// SPDX-License-Identifier: Apache-2.0
//! LSB-first bit reader and writer for ISO/IEC 18181-1 (JPEG XL) codestreams.
//!
//! Per ISO/IEC 18181-1 Annex B, JPEG XL packs bits in least-significant-bit-first
//! order within each byte, with bytes ordered sequentially. Variable-length
//! integers (`U32`, `U64`), IEEE 754 binary16 floats (`F16`), enumerations, and
//! UTF-8 `Name` strings are built on top of these primitives.

use super::JxlError;

/// Descriptor for one branch of a `U32(d0, d1, d2, d3)` syntax element
/// (ISO/IEC 18181-1 Annex B.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum U32Specifier {
    /// Constant value (`c`), consuming 0 extra bits after the 2-bit selector.
    Constant(u32),
    /// Base offset plus `n` extra unsigned bits (`c + u(n)`).
    Bits {
        /// Constant base added to the `n`-bit unsigned integer.
        offset: u32,
        /// Number of extra bits (`1..=32`).
        bits: u8,
    },
}

impl U32Specifier {
    /// Constant branch `c`.
    #[must_use]
    pub const fn c(value: u32) -> Self {
        Self::Constant(value)
    }

    /// Branch `offset + u(bits)`.
    #[must_use]
    pub const fn u(offset: u32, bits: u8) -> Self {
        if bits == 0 {
            Self::Constant(offset)
        } else {
            Self::Bits { offset, bits }
        }
    }
}

/// Unpacks a non-negative integer into a signed 32-bit integer (`UnpackSigned`
/// in ISO/IEC 18181-1 Annex B.2.4).
#[must_use]
#[inline]
pub const fn unpack_signed(x: u32) -> i32 {
    let bit = x & 1;
    let base = x >> 1;
    let flip = 0u32.wrapping_sub(bit);
    (base ^ flip) as i32
}

/// Packs a signed 32-bit integer into a non-negative integer (`PackSigned`,
/// inverse of [`unpack_signed`]).
#[must_use]
#[inline]
pub const fn pack_signed(v: i32) -> u32 {
    if v >= 0 {
        (v as u32) << 1
    } else {
        (((!v) as u32) << 1) | 1
    }
}

/// Unpacks a 64-bit unsigned integer into a signed 64-bit integer.
#[must_use]
#[inline]
pub const fn unpack_signed_u64(x: u64) -> i64 {
    let bit = x & 1;
    let base = x >> 1;
    let flip = 0u64.wrapping_sub(bit);
    (base ^ flip) as i64
}

/// Packs a signed 64-bit integer into an unsigned 64-bit integer.
#[must_use]
#[inline]
pub const fn pack_signed_i64(v: i64) -> u64 {
    if v >= 0 {
        (v as u64) << 1
    } else {
        (((!v) as u64) << 1) | 1
    }
}

/// Converts IEEE 754 binary16 bits into `f32`, rejecting NaN and infinity per
/// ISO/IEC 18181-1 Annex B.2.6.
///
/// # Errors
///
/// Returns [`JxlError::InvalidBitstream`] if the 16-bit float is NaN or +/-infinity.
pub fn f16_bits_to_f32(bits: u16) -> Result<f32, JxlError> {
    let sign = u32::from(bits >> 15);
    let exp = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);

    if exp == 0x1f {
        return Err(JxlError::InvalidBitstream(
            "non-finite F16 (NaN or infinity) in codestream",
        ));
    }
    if exp == 0 {
        if mantissa == 0 {
            return Ok(f32::from_bits(sign << 31));
        }
        // Subnormal binary16: value = (-1)^sign * 2^-14 * (mantissa / 1024).
        let val = (mantissa as f32) * (1.0 / 16_777_216.0);
        return Ok(if sign != 0 { -val } else { val });
    }
    let f32_exp = exp + (127 - 15);
    let f32_bits = (sign << 31) | (f32_exp << 23) | (mantissa << 13);
    Ok(f32::from_bits(f32_bits))
}

/// Converts a finite `f32` to IEEE 754 binary16 bits (round-to-nearest-even).
///
/// # Errors
///
/// Returns [`JxlError::InvalidConfig`] if `val` is non-finite or overflows binary16.
pub fn f32_to_f16_bits(val: f32) -> Result<u16, JxlError> {
    if !val.is_finite() {
        return Err(JxlError::InvalidConfig("cannot encode non-finite F16"));
    }
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs_bits = bits & 0x7fff_ffff;
    if abs_bits == 0 {
        return Ok(sign);
    }
    let exp32 = ((abs_bits >> 23) & 0xff) as i32 - 127;
    let mant32 = abs_bits & 0x007f_ffff;
    if exp32 > 15 {
        return Err(JxlError::InvalidConfig("value overflows F16 range"));
    }
    if exp32 < -24 {
        return Ok(sign);
    }
    if exp32 < -14 {
        let shift = -14 - exp32;
        let full_mant = (mant32 | 0x0080_0000) >> (shift + 13);
        return Ok(sign | (full_mant as u16));
    }
    let exp16 = ((exp32 + 15) as u16) << 10;
    let round_bit = (mant32 >> 12) & 1;
    let sticky = (mant32 & 0x0fff) != 0;
    let mut mant16 = (mant32 >> 13) as u16;
    if round_bit != 0 && (sticky || (mant16 & 1) != 0) {
        mant16 += 1;
        if mant16 >= 0x0400 {
            if exp32 == 15 {
                return Err(JxlError::InvalidConfig(
                    "value overflows F16 after rounding",
                ));
            }
            return Ok(sign | (((exp32 + 16) as u16) << 10));
        }
    }
    Ok(sign | exp16 | mant16)
}

/// LSB-first bitstream reader over a borrowed byte slice.
#[derive(Clone, Debug)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    buf: u64,
    num_read_bits: usize,
    remaining_buf_bits: usize,
}

impl<'a> BitReader<'a> {
    /// Creates a new [`BitReader`] positioned at bit 0 of `bytes`.
    #[must_use]
    #[inline]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            buf: 0,
            num_read_bits: 0,
            remaining_buf_bits: 0,
        }
    }

    /// Total number of bits consumed so far.
    #[must_use]
    #[inline]
    pub const fn num_read_bits(&self) -> usize {
        self.num_read_bits
    }

    /// Current byte offset if aligned, or the floor byte offset.
    #[must_use]
    #[inline]
    pub const fn bytes_read(&self) -> usize {
        self.num_read_bits / 8
    }

    /// Refills the internal 64-bit reservoir so at least 56 bits are available
    /// whenever the underlying slice has remaining bytes.
    #[inline]
    pub fn refill(&mut self) {
        while self.remaining_buf_bits <= 56 {
            let Some((&b, rest)) = self.bytes.split_first() else {
                break;
            };
            self.buf |= u64::from(b) << self.remaining_buf_bits;
            self.remaining_buf_bits += 8;
            self.bytes = rest;
        }
    }

    /// Peeks `n` bits (`0..=32`) without consuming them, refilling first.
    #[must_use]
    #[inline]
    pub fn peek_bits(&mut self, n: usize) -> u32 {
        debug_assert!(n <= 32);
        if n == 0 {
            return 0;
        }
        self.refill();
        let mask = (1u64 << n) - 1;
        (self.buf & mask) as u32
    }

    /// Peeks a compile-time constant number of bits (`N <= 32`), refilling first.
    #[must_use]
    #[inline]
    pub fn peek_bits_const<const N: usize>(&mut self) -> u32 {
        debug_assert!(N <= 32);
        if N == 0 {
            return 0;
        }
        self.refill();
        let mask = (1u64 << N) - 1;
        (self.buf & mask) as u32
    }

    /// Peeks `n` bits (`0..=32`) from the already-refilled reservoir.
    #[must_use]
    #[inline]
    pub fn peek_bits_prefilled(&self, n: usize) -> u32 {
        debug_assert!(n <= 32);
        if n == 0 {
            return 0;
        }
        let mask = (1u64 << n) - 1;
        (self.buf & mask) as u32
    }

    /// Consumes `n` bits from the reservoir.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::UnexpectedEof`] if fewer than `n` bits remain.
    #[inline]
    pub fn consume_bits(&mut self, n: usize) -> Result<(), JxlError> {
        if n == 0 {
            return Ok(());
        }
        self.remaining_buf_bits = self
            .remaining_buf_bits
            .checked_sub(n)
            .ok_or(JxlError::UnexpectedEof)?;
        self.num_read_bits += n;
        self.buf >>= n;
        Ok(())
    }

    /// Reads and consumes `n` bits (`0..=32`) LSB-first.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::UnexpectedEof`] if fewer than `n` bits remain.
    #[inline]
    pub fn read_bits(&mut self, n: usize) -> Result<u32, JxlError> {
        if n == 0 {
            return Ok(0);
        }
        let val = self.peek_bits(n);
        self.consume_bits(n)?;
        Ok(val)
    }

    /// Reads a single boolean bit (`Bool` in ISO/IEC 18181-1 Annex B.2.2).
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::UnexpectedEof`] on end of stream.
    #[inline]
    pub fn read_bool(&mut self) -> Result<bool, JxlError> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Skips `n` bits forward.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::UnexpectedEof`] if `n` exceeds the remaining bits.
    pub fn skip_bits(&mut self, mut n: usize) -> Result<(), JxlError> {
        if let Some(rem) = self.remaining_buf_bits.checked_sub(n) {
            self.num_read_bits += n;
            self.remaining_buf_bits = rem;
            self.buf >>= n;
            return Ok(());
        }
        n -= self.remaining_buf_bits;
        self.num_read_bits += self.remaining_buf_bits;
        self.buf = 0;
        self.remaining_buf_bits = 0;
        if n > self.bytes.len().saturating_mul(8) {
            return Err(JxlError::UnexpectedEof);
        }
        let skip_bytes = n / 8;
        self.bytes = &self.bytes[skip_bytes..];
        self.num_read_bits += skip_bytes * 8;
        let rem_bits = n % 8;
        self.refill();
        self.consume_bits(rem_bits)
    }

    /// Aligns to the next byte boundary and verifies that padding bits are zero
    /// (`ZeroPadToByte` in ISO/IEC 18181-1 Annex B.2.7).
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::InvalidBitstream`] if any padding bit is non-zero.
    pub fn zero_pad_to_byte(&mut self) -> Result<(), JxlError> {
        let byte_boundary = self.num_read_bits.div_ceil(8) * 8;
        let pad_bits = byte_boundary - self.num_read_bits;
        if pad_bits > 0 && self.read_bits(pad_bits)? != 0 {
            return Err(JxlError::InvalidBitstream(
                "non-zero byte alignment padding",
            ));
        }
        Ok(())
    }

    /// Reads a `U32(d0, d1, d2, d3)` variable-length integer (ISO/IEC 18181-1 Annex B.2.1).
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::UnexpectedEof`] on premature end of stream.
    #[inline]
    pub fn read_u32(
        &mut self,
        d0: U32Specifier,
        d1: U32Specifier,
        d2: U32Specifier,
        d3: U32Specifier,
    ) -> Result<u32, JxlError> {
        let selector = self.read_bits(2)?;
        let spec = match selector {
            0 => d0,
            1 => d1,
            2 => d2,
            3 => d3,
            _ => unreachable!(),
        };
        match spec {
            U32Specifier::Constant(c) => Ok(c),
            U32Specifier::Bits { offset, bits } => {
                let extra = self.read_bits(bits as usize)?;
                Ok(offset.wrapping_add(extra))
            }
        }
    }

    /// Reads a `U64()` variable-length integer (ISO/IEC 18181-1 Annex B.2.3).
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::UnexpectedEof`] on premature end of stream.
    pub fn read_u64(&mut self) -> Result<u64, JxlError> {
        let selector = self.read_bits(2)?;
        match selector {
            0 => Ok(0),
            1 => Ok(1 + u64::from(self.read_bits(4)?)),
            2 => Ok(17 + u64::from(self.read_bits(8)?)),
            3 => {
                let mut value = u64::from(self.read_bits(12)?);
                let mut shift = 12u32;
                while self.read_bool()? {
                    if shift == 60 {
                        value |= u64::from(self.read_bits(4)?) << shift;
                        break;
                    }
                    value |= u64::from(self.read_bits(8)?) << shift;
                    shift += 8;
                }
                Ok(value)
            }
            _ => unreachable!(),
        }
    }

    /// Reads an `F16()` 16-bit floating-point number as `f32` (ISO/IEC 18181-1 Annex B.2.6).
    ///
    /// # Errors
    ///
    /// Returns [`JxlError`] on EOF or if the value is NaN/infinity.
    #[inline]
    pub fn read_f16(&mut self) -> Result<f32, JxlError> {
        let bits = self.read_bits(16)? as u16;
        f16_bits_to_f32(bits)
    }

    /// Reads an `Enum()` value encoded with `U32(0, 1, 2 + u(4), 18 + u(6))`
    /// (ISO/IEC 18181-1 Annex B.2.5).
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::UnexpectedEof`] on premature end of stream.
    #[inline]
    pub fn read_enum_raw(&mut self) -> Result<u32, JxlError> {
        self.read_u32(
            U32Specifier::c(0),
            U32Specifier::c(1),
            U32Specifier::u(2, 4),
            U32Specifier::u(18, 6),
        )
    }

    /// Reads a UTF-8 `Name` field (`U32(0, u(4), 16 + u(5), 48 + u(10))` length followed by bytes).
    ///
    /// # Errors
    ///
    /// Returns [`JxlError`] on EOF or invalid UTF-8.
    pub fn read_name(&mut self) -> Result<String, JxlError> {
        let len = self.read_u32(
            U32Specifier::c(0),
            U32Specifier::u(0, 4),
            U32Specifier::u(16, 5),
            U32Specifier::u(48, 10),
        )? as usize;
        let mut bytes = Vec::with_capacity(len);
        for _ in 0..len {
            bytes.push(self.read_bits(8)? as u8);
        }
        String::from_utf8(bytes)
            .map_err(|_| JxlError::InvalidBitstream("invalid UTF-8 in Name field"))
    }
}

/// LSB-first bitstream writer.
#[derive(Clone, Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    buf: u64,
    bits_in_buf: usize,
}

impl BitWriter {
    /// Creates an empty [`BitWriter`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bytes: Vec::new(),
            buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Creates a [`BitWriter`] with pre-allocated byte capacity.
    #[must_use]
    pub fn with_capacity(bytes_cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes_cap),
            buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Total number of bits written so far.
    #[must_use]
    #[inline]
    pub fn num_written_bits(&self) -> usize {
        self.bytes.len() * 8 + self.bits_in_buf
    }

    /// Flushes complete bytes from the internal bit accumulator.
    #[inline]
    fn flush_whole_bytes(&mut self) {
        while self.bits_in_buf >= 8 {
            self.bytes.push((self.buf & 0xff) as u8);
            self.buf >>= 8;
            self.bits_in_buf -= 8;
        }
    }

    /// Writes `n` bits (`0..=32`) from the low bits of `value` in LSB-first order.
    #[inline]
    pub fn write_bits(&mut self, n: usize, value: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        let mask = if n == 32 {
            u64::MAX >> 32
        } else {
            (1u64 << n) - 1
        };
        let clean = u64::from(value) & mask;
        self.buf |= clean << self.bits_in_buf;
        self.bits_in_buf += n;
        if self.bits_in_buf >= 32 {
            self.flush_whole_bytes();
        }
    }

    /// Writes a single boolean bit (`0` or `1`).
    #[inline]
    pub fn write_bool(&mut self, value: bool) {
        self.write_bits(1, u32::from(value));
    }

    /// Pads with zero bits to the next byte boundary (`ZeroPadToByte`).
    #[inline]
    pub fn zero_pad_to_byte(&mut self) {
        self.flush_whole_bytes();
        if self.bits_in_buf > 0 {
            self.bytes.push((self.buf & 0xff) as u8);
            self.buf = 0;
            self.bits_in_buf = 0;
        }
    }

    /// Appends raw byte-aligned bytes after zero-padding to a byte boundary.
    pub fn write_aligned_bytes(&mut self, slice: &[u8]) {
        self.zero_pad_to_byte();
        self.bytes.extend_from_slice(slice);
    }

    /// Appends all bits from another [`BitWriter`] without forcing byte alignment.
    pub fn append_bits_from(&mut self, other: &Self) {
        for &b in &other.bytes {
            self.write_bits(8, u32::from(b));
        }
        if other.bits_in_buf > 0 {
            self.write_bits(other.bits_in_buf, other.buf as u32);
        }
    }

    /// Encodes `value` using the shortest matching branch of `U32(d0, d1, d2, d3)`.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::InvalidConfig`] if `value` does not fit any branch.
    pub fn write_u32(
        &mut self,
        value: u32,
        d0: U32Specifier,
        d1: U32Specifier,
        d2: U32Specifier,
        d3: U32Specifier,
    ) -> Result<(), JxlError> {
        let specs = [d0, d1, d2, d3];
        for (selector, &spec) in specs.iter().enumerate() {
            match spec {
                U32Specifier::Constant(c) if value == c => {
                    self.write_bits(2, selector as u32);
                    return Ok(());
                }
                U32Specifier::Bits { offset, bits } if value >= offset => {
                    let delta = value - offset;
                    let max_delta = if bits >= 32 {
                        u32::MAX
                    } else {
                        (1u32 << bits) - 1
                    };
                    if delta <= max_delta {
                        self.write_bits(2, selector as u32);
                        self.write_bits(bits as usize, delta);
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
        Err(JxlError::InvalidConfig(
            "value cannot be represented by U32 specifier",
        ))
    }

    /// Writes a `U64()` variable-length integer (ISO/IEC 18181-1 Annex B.2.3).
    pub fn write_u64(&mut self, mut value: u64) {
        if value == 0 {
            self.write_bits(2, 0);
        } else if value <= 16 {
            self.write_bits(2, 1);
            self.write_bits(4, (value - 1) as u32);
        } else if value <= 272 {
            self.write_bits(2, 2);
            self.write_bits(8, (value - 17) as u32);
        } else {
            self.write_bits(2, 3);
            self.write_bits(12, (value & 0x0fff) as u32);
            value >>= 12;
            let mut shift = 12u32;
            while value > 0 {
                self.write_bool(true);
                if shift == 60 {
                    self.write_bits(4, (value & 0x0f) as u32);
                    return;
                }
                self.write_bits(8, (value & 0xff) as u32);
                value >>= 8;
                shift += 8;
            }
            self.write_bool(false);
        }
    }

    /// Writes an `F16()` 16-bit floating-point number.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::InvalidConfig`] if `value` is not representable in binary16.
    pub fn write_f16(&mut self, value: f32) -> Result<(), JxlError> {
        let bits = f32_to_f16_bits(value)?;
        self.write_bits(16, u32::from(bits));
        Ok(())
    }

    /// Writes an `Enum()` value (`U32(0, 1, 2 + u(4), 18 + u(6))`).
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::InvalidConfig`] if `value > 81`.
    pub fn write_enum_raw(&mut self, value: u32) -> Result<(), JxlError> {
        self.write_u32(
            value,
            U32Specifier::c(0),
            U32Specifier::c(1),
            U32Specifier::u(2, 4),
            U32Specifier::u(18, 6),
        )
    }

    /// Writes a UTF-8 `Name` field.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::InvalidConfig`] if `name.len() > 1071`.
    pub fn write_name(&mut self, name: &str) -> Result<(), JxlError> {
        let bytes = name.as_bytes();
        let len = u32::try_from(bytes.len())
            .map_err(|_| JxlError::InvalidConfig("Name string too long"))?;
        self.write_u32(
            len,
            U32Specifier::c(0),
            U32Specifier::u(0, 4),
            U32Specifier::u(16, 5),
            U32Specifier::u(48, 10),
        )?;
        for &b in bytes {
            self.write_bits(8, u32::from(b));
        }
        Ok(())
    }

    /// Zero-pads to a byte boundary and returns the owned byte buffer.
    #[must_use]
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.zero_pad_to_byte();
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_packing_round_trips() {
        for v in [-100_000, -255, -2, -1, 0, 1, 2, 255, 100_000] {
            assert_eq!(unpack_signed(pack_signed(v)), v);
        }
        for v in [-1_000_000_000i64, -1, 0, 1, 1_000_000_000i64] {
            assert_eq!(unpack_signed_u64(pack_signed_i64(v)), v);
        }
    }

    #[test]
    fn u32_u64_f16_and_name_round_trip() {
        let mut w = BitWriter::new();
        let d0 = U32Specifier::c(0);
        let d1 = U32Specifier::c(1);
        let d2 = U32Specifier::u(2, 4);
        let d3 = U32Specifier::u(18, 8);

        for &val in &[0, 1, 2, 17, 18, 200] {
            w.write_u32(val, d0, d1, d2, d3).expect("u32");
        }
        for &val in &[0u64, 1, 16, 17, 272, 273, 1_000_000, (1u64 << 42) + 7] {
            w.write_u64(val);
        }
        w.write_f16(0.5).expect("f16");
        w.write_f16(-1.25).expect("f16");
        w.write_name("rustmpeg-jxl").expect("name");
        let bytes = w.into_bytes();

        let mut r = BitReader::new(&bytes);
        for &val in &[0, 1, 2, 17, 18, 200] {
            assert_eq!(r.read_u32(d0, d1, d2, d3).expect("u32"), val);
        }
        for &val in &[0u64, 1, 16, 17, 272, 273, 1_000_000, (1u64 << 42) + 7] {
            assert_eq!(r.read_u64().expect("u64"), val);
        }
        assert!((r.read_f16().expect("f16") - 0.5).abs() < 1e-4);
        assert!((r.read_f16().expect("f16") + 1.25).abs() < 1e-4);
        assert_eq!(r.read_name().expect("name"), "rustmpeg-jxl");
        r.zero_pad_to_byte().expect("pad");
    }
}
