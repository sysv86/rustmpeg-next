// SPDX-License-Identifier: Apache-2.0
//! FLAC subframe decoding (RFC 9639, Section 9.2).

use std::io::Read;

use super::FlacError;
use super::bitreader::BitReader;

/// Maximum LPC predictor order.
const MAX_LPC_ORDER: u32 = 32;

/// Decodes one subframe into `block_size` samples.
///
/// `side` selects the one-bit-deeper sample depth used by the side channel of
/// a stereo decorrelation.  The returned values have already had wasted bits
/// restored.
pub(crate) fn decode<R: Read>(
    reader: &mut BitReader<R>,
    block_size: u32,
    frame_bits_per_sample: u8,
    side: bool,
) -> Result<Vec<i64>, FlacError> {
    if reader.read_bit()? {
        return Err(FlacError::InvalidFrame(
            "subframe header padding bit is not zero",
        ));
    }
    let type_code = reader.read_bits(6)? as u32;
    // Frame bit depths are validated to 4..=32, so `base_bits` is at least 4.
    let base_bits = u32::from(frame_bits_per_sample) + u32::from(side);
    let wasted_bits = if reader.read_bit()? {
        // The field stores the number of wasted bits k minus one in unary
        // form.  At least one sample bit must remain, so k is at most
        // `base_bits - 1`; bounding the unary run rejects longer codes without
        // scanning an unbounded number of zero bits (RFC 9639, Section 9.2.2).
        let zeros = reader.read_unary_zeros(u64::from(base_bits - 1))?;
        // Bounded by `base_bits - 1`, which is at most 32.
        zeros as u32 + 1
    } else {
        0
    };

    if wasted_bits >= base_bits {
        return Err(FlacError::InvalidFrame(
            "wasted bits leave no sample bits in the subframe",
        ));
    }
    let coded_bits = base_bits - wasted_bits;

    let mut samples = match type_code {
        0 => {
            let sample = reader.read_signed_bits(coded_bits)?;
            vec![sample; block_size as usize]
        }
        1 => {
            let mut samples = Vec::with_capacity(block_size as usize);
            for _ in 0..block_size {
                samples.push(reader.read_signed_bits(coded_bits)?);
            }
            samples
        }
        8..=12 => decode_fixed(reader, block_size, type_code - 8, coded_bits)?,
        32..=63 => decode_lpc(reader, block_size, type_code - 31, coded_bits)?,
        _ => {
            return Err(FlacError::UnsupportedFeature(
                "reserved subframe type in frame",
            ));
        }
    };

    if wasted_bits != 0 {
        // A `coded_bits`-wide sample carries a sign bit plus `coded_bits - 1`
        // value bits, and `coded_bits + wasted_bits == base_bits <= 33`, so
        // restoring the wasted low bits cannot overflow an `i64`.
        for sample in &mut samples {
            *sample <<= wasted_bits;
        }
    }
    validate_sample_range(&samples, base_bits)?;
    Ok(samples)
}

/// Checks that every sample fits the declared (side-adjusted) bit depth.
fn validate_sample_range(samples: &[i64], bits: u32) -> Result<(), FlacError> {
    let limit = 1i64 << (bits - 1);
    if samples
        .iter()
        .any(|&sample| sample < -limit || sample >= limit)
    {
        return Err(FlacError::InvalidFrame(
            "subframe sample is outside the declared bit depth",
        ));
    }
    Ok(())
}

/// Decodes a fixed-predictor subframe.
fn decode_fixed<R: Read>(
    reader: &mut BitReader<R>,
    block_size: u32,
    order: u32,
    coded_bits: u32,
) -> Result<Vec<i64>, FlacError> {
    if order >= block_size {
        return Err(FlacError::InvalidFrame(
            "fixed predictor order is not smaller than the block size",
        ));
    }
    let mut samples = vec![0i64; block_size as usize];
    for sample in samples.iter_mut().take(order as usize) {
        *sample = reader.read_signed_bits(coded_bits)?;
    }
    decode_residual(reader, block_size, order, &mut samples[order as usize..])?;

    for index in order as usize..samples.len() {
        let prediction = fixed_prediction(&samples, index, order)?;
        samples[index] = samples[index]
            .checked_add(prediction)
            .ok_or(FlacError::InvalidFrame(
                "fixed predictor reconstruction overflow",
            ))?;
    }
    Ok(samples)
}

/// Computes the fixed predictor for one sample using 128-bit intermediates.
fn fixed_prediction(samples: &[i64], index: usize, order: u32) -> Result<i64, FlacError> {
    let sample = |back: usize| i128::from(samples[index - back]);
    let prediction = match order {
        0 => 0i128,
        1 => sample(1),
        2 => 2 * sample(1) - sample(2),
        3 => 3 * sample(1) - 3 * sample(2) + sample(3),
        4 => 4 * sample(1) - 6 * sample(2) + 4 * sample(3) - sample(4),
        _ => {
            return Err(FlacError::InvalidFrame(
                "fixed predictor order is outside 0 to 4",
            ));
        }
    };
    i64::try_from(prediction).map_err(|_| FlacError::InvalidFrame("fixed predictor value overflow"))
}

/// Decodes a linear-predictor subframe.
fn decode_lpc<R: Read>(
    reader: &mut BitReader<R>,
    block_size: u32,
    order: u32,
    coded_bits: u32,
) -> Result<Vec<i64>, FlacError> {
    if order == 0 || order > MAX_LPC_ORDER {
        return Err(FlacError::InvalidFrame("invalid LPC order"));
    }
    if order >= block_size {
        return Err(FlacError::InvalidFrame(
            "LPC order is not smaller than the block size",
        ));
    }

    let mut samples = vec![0i64; block_size as usize];
    for sample in samples.iter_mut().take(order as usize) {
        *sample = reader.read_signed_bits(coded_bits)?;
    }

    let precision_code = reader.read_bits(4)? as u32;
    if precision_code == 15 {
        return Err(FlacError::InvalidFrame(
            "forbidden LPC coefficient precision",
        ));
    }
    let precision = precision_code + 1;
    let shift = reader.read_signed_bits(5)?;
    if shift < 0 {
        return Err(FlacError::InvalidFrame("negative LPC right shift"));
    }
    let shift = shift as u32;

    let mut coefficients = [0i64; MAX_LPC_ORDER as usize];
    for coefficient in coefficients.iter_mut().take(order as usize) {
        *coefficient = reader.read_signed_bits(precision)?;
    }

    decode_residual(reader, block_size, order, &mut samples[order as usize..])?;

    for index in order as usize..samples.len() {
        let mut sum = 0i128;
        for (coefficient, past) in coefficients[..order as usize]
            .iter()
            .zip(samples[index - order as usize..index].iter().rev())
        {
            sum += i128::from(*coefficient) * i128::from(*past);
        }
        let prediction = i64::try_from(sum >> shift)
            .map_err(|_| FlacError::InvalidFrame("LPC prediction value overflow"))?;
        samples[index] = samples[index]
            .checked_add(prediction)
            .ok_or(FlacError::InvalidFrame("LPC reconstruction overflow"))?;
    }
    Ok(samples)
}

/// Decodes and writes a partitioned Rice residual into `output`.
fn decode_residual<R: Read>(
    reader: &mut BitReader<R>,
    block_size: u32,
    predictor_order: u32,
    output: &mut [i64],
) -> Result<(), FlacError> {
    let method = reader.read_bits(2)? as u32;
    let parameter_bits = match method {
        0 => 4,
        1 => 5,
        _ => {
            return Err(FlacError::UnsupportedFeature(
                "reserved residual coding method",
            ));
        }
    };
    let partition_order = reader.read_bits(4)? as u32;
    let partitions = 1u32 << partition_order;
    if !block_size.is_multiple_of(partitions) {
        return Err(FlacError::InvalidFrame(
            "block size is not divisible by the residual partition count",
        ));
    }
    let samples_per_partition = block_size / partitions;
    if samples_per_partition <= predictor_order {
        return Err(FlacError::InvalidFrame(
            "residual partition is too small for the predictor order",
        ));
    }
    let expected = block_size - predictor_order;
    if output.len() != expected as usize {
        return Err(FlacError::InvalidFrame(
            "residual output length does not match the block size",
        ));
    }

    let escape = (1u32 << parameter_bits) - 1;
    let max_folded = u64::from(u32::MAX - 1);
    let mut index = 0usize;

    for partition in 0..partitions {
        let count = if partition == 0 {
            samples_per_partition - predictor_order
        } else {
            samples_per_partition
        } as usize;
        let parameter = reader.read_bits(parameter_bits)? as u32;

        if parameter == escape {
            let raw_bits = reader.read_bits(5)? as u32;
            for _ in 0..count {
                let value = reader.read_signed_bits(raw_bits)?;
                let slot = output
                    .get_mut(index)
                    .ok_or(FlacError::InvalidFrame("residual partition overflow"))?;
                *slot = value;
                index += 1;
            }
        } else {
            let max_quotient = max_folded >> parameter;
            for _ in 0..count {
                let quotient = reader.read_unary_zeros(max_quotient)?;
                let remainder = if parameter == 0 {
                    0
                } else {
                    reader.read_bits(parameter)?
                };
                let folded = (quotient << parameter) | remainder;
                if folded > max_folded {
                    return Err(FlacError::InvalidFrame(
                        "residual sample exceeds the 32-bit limit",
                    ));
                }
                let value = if folded & 1 == 0 {
                    (folded >> 1) as i64
                } else {
                    -((folded >> 1) as i64) - 1
                };
                let slot = output
                    .get_mut(index)
                    .ok_or(FlacError::InvalidFrame("residual partition overflow"))?;
                *slot = value;
                index += 1;
            }
        }
    }

    if index != output.len() {
        return Err(FlacError::InvalidFrame(
            "residual partition sizes do not cover the block",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::decode;
    use crate::flac::FlacError;
    use crate::flac::bitreader::BitReader;

    /// A minimal MSB-first bit writer for hand-built subframes.
    struct BitWriter {
        bytes: Vec<u8>,
        current: u8,
        bits: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                current: 0,
                bits: 0,
            }
        }

        fn write_bit(&mut self, bit: bool) {
            self.current = (self.current << 1) | u8::from(bit);
            self.bits += 1;
            if self.bits == 8 {
                self.bytes.push(self.current);
                self.current = 0;
                self.bits = 0;
            }
        }

        fn write_bits(&mut self, value: u64, count: u32) {
            for shift in (0..count).rev() {
                self.write_bit((value >> shift) & 1 != 0);
            }
        }

        fn write_signed(&mut self, value: i64, count: u32) {
            self.write_bits(value as u64, count);
        }

        fn finish(mut self) -> Vec<u8> {
            if self.bits != 0 {
                self.current <<= 8 - self.bits;
                self.bytes.push(self.current);
            }
            self.bytes
        }
    }

    fn start_subframe(writer: &mut BitWriter, type_code: u32, wasted_bits: Option<u32>) {
        writer.write_bit(false);
        writer.write_bits(u64::from(type_code), 6);
        match wasted_bits {
            None => writer.write_bit(false),
            Some(count) => {
                writer.write_bit(true);
                for _ in 1..count {
                    writer.write_bit(false);
                }
                writer.write_bit(true);
            }
        }
    }

    fn write_zero_residuals(writer: &mut BitWriter, count: u32) {
        writer.write_bits(0, 2); // 4-bit Rice parameters
        writer.write_bits(0, 4); // partition order 0
        writer.write_bits(0, 4); // Rice parameter 0
        for _ in 0..count {
            writer.write_bit(true); // quotient 0
        }
    }

    fn decode_bytes(
        bytes: Vec<u8>,
        block_size: u32,
        bits: u8,
        side: bool,
    ) -> Result<Vec<i64>, FlacError> {
        let mut reader = BitReader::new(Cursor::new(bytes));
        decode(&mut reader, block_size, bits, side)
    }

    #[test]
    fn decodes_constant_subframes() {
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 0, None);
        writer.write_signed(-1234, 16);
        let samples = decode_bytes(writer.finish(), 4, 16, false).unwrap();
        assert_eq!(samples, vec![-1234; 4]);
    }

    #[test]
    fn decodes_verbatim_subframes_with_wasted_bits() {
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 1, Some(3));
        for sample in [10i64, -20, 30] {
            writer.write_signed(sample, 13);
        }
        let samples = decode_bytes(writer.finish(), 3, 16, false).unwrap();
        assert_eq!(samples, vec![80, -160, 240]);
    }

    #[test]
    fn decodes_fixed_predictor_subframes() {
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 10, None); // fixed predictor, order 2
        writer.write_signed(10, 16);
        writer.write_signed(20, 16);
        write_zero_residuals(&mut writer, 4);
        let samples = decode_bytes(writer.finish(), 6, 16, false).unwrap();
        assert_eq!(samples, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn decodes_lpc_subframes() {
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 33, None); // LPC, order 2
        writer.write_signed(5, 16);
        writer.write_signed(7, 16);
        writer.write_bits(2, 4); // coefficient precision 3
        writer.write_signed(0, 5); // right shift 0
        writer.write_signed(2, 3);
        writer.write_signed(-1, 3);
        write_zero_residuals(&mut writer, 2);
        let samples = decode_bytes(writer.finish(), 4, 16, false).unwrap();
        assert_eq!(samples, vec![5, 7, 9, 11]);
    }

    #[test]
    fn rejects_negative_lpc_shifts() {
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 33, None); // LPC, order 2
        writer.write_signed(5, 16);
        writer.write_signed(7, 16);
        writer.write_bits(2, 4); // coefficient precision 3
        writer.write_signed(-1, 5); // negative right shift
        writer.write_signed(0, 3);
        writer.write_signed(0, 3);
        write_zero_residuals(&mut writer, 2);
        let error = decode_bytes(writer.finish(), 4, 16, false).err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));
    }

    #[test]
    fn decodes_maximum_order_lpc_subframes() {
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 63, None); // LPC, order 32
        for sample in 0..32i64 {
            writer.write_signed(1000 + sample, 16);
        }
        writer.write_bits(0, 4); // coefficient precision 1
        writer.write_signed(0, 5); // right shift 0
        for _ in 0..32 {
            writer.write_signed(0, 1); // all coefficients are zero
        }
        write_zero_residuals(&mut writer, 1);
        let samples = decode_bytes(writer.finish(), 33, 16, false).unwrap();
        let mut expected: Vec<i64> = (0..32).map(|sample| 1000 + sample).collect();
        expected.push(0);
        assert_eq!(samples, expected);
    }

    #[test]
    fn decodes_escaped_residual_partitions() {
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 8, None); // fixed predictor, order 0
        writer.write_bits(0, 2); // 4-bit Rice parameters
        writer.write_bits(0, 4); // partition order 0
        writer.write_bits(0b1111, 4); // escape code
        writer.write_bits(5, 5); // five raw bits per sample
        for sample in [-16i64, 0, 15] {
            writer.write_signed(sample, 5);
        }
        let samples = decode_bytes(writer.finish(), 3, 16, false).unwrap();
        assert_eq!(samples, vec![-16, 0, 15]);
    }

    #[test]
    fn decodes_escaped_partitions_with_zero_raw_bits() {
        // RFC 9639, Section 9.2.7.1: zero raw bits means every residual in the
        // partition is zero and no bits are stored for the samples.
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 8, None); // fixed predictor, order 0
        writer.write_bits(0, 2); // 4-bit Rice parameters
        writer.write_bits(0, 4); // partition order 0
        writer.write_bits(0b1111, 4); // escape code
        writer.write_bits(0, 5); // zero raw bits per sample
        let samples = decode_bytes(writer.finish(), 4, 16, false).unwrap();
        assert_eq!(samples, vec![0; 4]);
    }

    #[test]
    fn decodes_five_bit_rice_parameters() {
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 8, None); // fixed predictor, order 0
        writer.write_bits(1, 2); // 5-bit Rice parameters
        writer.write_bits(0, 4); // partition order 0
        writer.write_bits(20, 5); // Rice parameter 20
        writer.write_bit(true); // quotient 0
        writer.write_bits(123_456, 20);
        let samples = decode_bytes(writer.finish(), 1, 24, false).unwrap();
        assert_eq!(samples, vec![61_728]);
    }

    #[test]
    fn decodes_extended_sample_depths() {
        for (bits, value) in [(12u8, -2048i64), (20, 524_287)] {
            let mut writer = BitWriter::new();
            start_subframe(&mut writer, 0, None);
            writer.write_signed(value, u32::from(bits));
            let samples = decode_bytes(writer.finish(), 2, bits, false).unwrap();
            assert_eq!(samples, vec![value; 2]);
        }
    }

    #[test]
    fn rejects_invalid_partition_orders() {
        // Block size 5 cannot be split into two partitions.
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 8, None);
        writer.write_bits(0, 2);
        writer.write_bits(1, 4);
        writer.write_bits(0, 4);
        let error = decode_bytes(writer.finish(), 5, 16, false).err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));

        // A partition must contain more samples than the predictor order.
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 11, None); // fixed predictor, order 3
        for sample in [1i64, 2, 3] {
            writer.write_signed(sample, 16);
        }
        writer.write_bits(0, 2);
        writer.write_bits(1, 4);
        writer.write_bits(0, 4);
        let error = decode_bytes(writer.finish(), 6, 16, false).err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));

        // The maximum partition order, 15, implies 32768 partitions, which no
        // legal block size of at most 65535 samples can be split into evenly.
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 8, None);
        writer.write_bits(0, 2);
        writer.write_bits(15, 4);
        let error = decode_bytes(writer.finish(), 16, 16, false).err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));
    }

    #[test]
    fn bounds_wasted_bit_unary_codes() {
        // A wasted-bits code longer than the sample depth is rejected without
        // reading an unbounded number of zero bits.
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 1, Some(40));
        let error = decode_bytes(writer.finish(), 1, 16, false).err();
        assert!(matches!(error, Some(FlacError::LimitExceeded(_))));
    }

    #[test]
    fn rejects_reserved_subframe_types() {
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 2, None);
        let error = decode_bytes(writer.finish(), 1, 16, false).err();
        assert!(matches!(error, Some(FlacError::UnsupportedFeature(_))));
    }

    #[test]
    fn rejects_wasted_bits_that_consume_the_sample() {
        let mut writer = BitWriter::new();
        start_subframe(&mut writer, 1, Some(16));
        let error = decode_bytes(writer.finish(), 1, 16, false).err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));
    }
}
