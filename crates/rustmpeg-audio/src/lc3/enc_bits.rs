// SPDX-License-Identifier: Apache-2.0
//! LC3 payload writer and arithmetic encoder (Bluetooth LC3 v1.0.1 §3.3.13).
//!
//! Side information and residual bits are written backwards from the last
//! byte of the payload, the arithmetically coded data forwards from the first
//! byte. The two ends must not cross.

use super::Lc3Error;
use super::bits::ilog2;

/// Backward bit writer for side information and residual data.
pub(crate) struct BitWriter<'a> {
    bytes: &'a mut [u8],
    bp_side: i64,
    mask_side: u8,
}

impl<'a> BitWriter<'a> {
    /// Starts writing at the end of `bytes`.
    pub(crate) fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            bp_side: bytes.len() as i64 - 1,
            mask_side: 1,
            bytes,
        }
    }

    /// Writes one bit.
    pub(crate) fn write_bit(&mut self, bit: bool) -> Result<(), Lc3Error> {
        if self.bp_side < 0 {
            return Err(Lc3Error::Encode("LC3 side information overflow"));
        }
        let byte = &mut self.bytes[self.bp_side as usize];
        if bit {
            *byte |= self.mask_side;
        } else {
            *byte &= !self.mask_side;
        }
        if self.mask_side == 0x80 {
            self.mask_side = 1;
            self.bp_side -= 1;
        } else {
            self.mask_side <<= 1;
        }
        Ok(())
    }

    /// Writes the low `nbits` bits of `value`, least significant first.
    pub(crate) fn write_uint(&mut self, value: u32, nbits: u32) -> Result<(), Lc3Error> {
        for bit in 0..nbits {
            self.write_bit((value >> bit) & 1 != 0)?;
        }
        Ok(())
    }

    /// Number of bits written so far.
    pub(crate) fn bits_written(&self, nbits: i32) -> i32 {
        let side = 8 * (self.bytes.len() as i64 - 1 - self.bp_side) as i32
            + ilog2(u32::from(self.mask_side));
        nbits - side
    }

    /// Byte index of the next free byte below the written side information.
    pub(crate) fn bottom(&self) -> i64 {
        self.bp_side
    }
}

/// Range encoder state (Equations of §3.3.13.6).
pub(crate) struct AcEncoder {
    low: u32,
    range: u32,
    cache: i32,
    carry: bool,
    carry_count: u32,
    out: Vec<u8>,
    /// Set when termination used the carry path with a full final byte.
    #[cfg(test)]
    pub(crate) full_tail_with_carry: bool,
}

impl AcEncoder {
    /// Creates an initialized arithmetic encoder.
    pub(crate) fn new() -> Self {
        Self {
            low: 0,
            range: 0x00ff_ffff,
            cache: -1,
            carry: false,
            carry_count: 0,
            out: Vec::new(),
            #[cfg(test)]
            full_tail_with_carry: false,
        }
    }

    /// Encodes one symbol given its cumulative frequency and frequency.
    pub(crate) fn encode(&mut self, cumulative: u16, frequency: u16) {
        let r = self.range >> 10;
        self.low += r * u32::from(cumulative);
        if self.low >> 24 != 0 {
            self.carry = true;
        }
        self.low &= 0x00ff_ffff;
        self.range = r * u32::from(frequency);
        while self.range < 0x1_0000 {
            self.range <<= 8;
            self.shift();
        }
    }

    /// Emits one byte, resolving pending carries.
    fn shift(&mut self) {
        if self.low < 0x00ff_0000 || self.carry {
            if self.cache >= 0 {
                self.out.push(self.cache as u8 + u8::from(self.carry));
            }
            while self.carry_count > 0 {
                self.out.push(u8::from(self.carry).wrapping_add(0xff));
                self.carry_count -= 1;
            }
            self.cache = (self.low >> 16) as i32;
            self.carry = false;
        } else {
            self.carry_count += 1;
        }
        self.low = (self.low << 8) & 0x00ff_ffff;
    }

    /// Upper bound of the bits emitted so far, including the pending flush.
    pub(crate) fn bits_estimate(&self) -> i32 {
        let mut bits = (self.out.len() * 8) as i32 + (25 - ilog2(self.range));
        if self.cache >= 0 {
            bits += 8;
        }
        bits + (self.carry_count * 8) as i32
    }

    /// Terminates the arithmetic code (§3.3.13.6 `ac_enc_finish`).
    pub(crate) fn finish(&mut self) {
        let mut bits = 1;
        while (self.range >> (24 - bits)) == 0 {
            bits += 1;
        }
        let mut mask = 0x00ff_ffffu32 >> bits;
        let mut val = self.low + mask;
        let over1 = val >> 24;
        val &= 0x00ff_ffff;
        let high = self.low + self.range;
        let over2 = high >> 24;
        let high = high & 0x00ff_ffff;
        val &= !mask;
        if over1 == over2 {
            if val + mask >= high {
                bits += 1;
                mask >>= 1;
                val = ((self.low + mask) & 0x00ff_ffff) & !mask;
            }
            if val < self.low {
                self.carry = true;
            }
        }
        self.low = val;
        while bits > 0 {
            self.shift();
            bits -= 8;
        }
        bits += 8;
        let carries = self.carry_count > 0;
        #[cfg(test)]
        {
            self.full_tail_with_carry = carries && bits == 8;
        }
        let tail = if carries {
            let cache = self.cache;
            self.out.push(cache as u8);
            while self.carry_count > 1 {
                self.out.push(0xff);
                self.carry_count -= 1;
            }
            // The write below keeps only the top `bits` bits, so a value
            // with `bits` low ones terminates with `bits` ones only when
            // `bits == 8`; shorter tails terminate with zeros.
            0xff >> (8 - bits)
        } else {
            self.cache as u32
        };
        // Write the top `bits` bits of `tail` into the next byte.
        let byte = self.out.len();
        self.out.push(0);
        let mut source = 0x80u32;
        for k in 0..bits {
            let values = tail & source;
            if values != 0 {
                self.out[byte] |= 0x80 >> k;
            }
            source >>= 1;
        }
    }

    /// Bytes produced by the encoder.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lc3::tables::{AC_SPEC_CUMFREQ, AC_SPEC_FREQ};

    #[test]
    fn backward_writer_reads_back_in_order() {
        let mut payload = [0u8; 8];
        let mut writer = BitWriter::new(&mut payload);
        writer.write_uint(5, 3).unwrap();
        writer.write_uint(0x2a, 8).unwrap();
        let mut reader = crate::lc3::bits::BitReader::new(&payload).unwrap();
        assert_eq!(reader.read_uint(3).unwrap(), 5);
        assert_eq!(reader.read_uint(8).unwrap(), 0x2a);
    }

    #[test]
    fn arithmetic_round_trip() {
        let symbols = [0usize, 3, 16, 1, 5, 5, 9];
        let mut encoder = AcEncoder::new();
        for &symbol in &symbols {
            encoder.encode(AC_SPEC_CUMFREQ[7][symbol], AC_SPEC_FREQ[7][symbol]);
        }
        encoder.finish();
        let mut payload = vec![0u8; 16];
        payload[..encoder.bytes().len()].copy_from_slice(encoder.bytes());
        let mut reader = crate::lc3::bits::BitReader::new(&payload).unwrap();
        for &symbol in &symbols {
            let decoded = reader
                .ac_decode(&AC_SPEC_CUMFREQ[7], &AC_SPEC_FREQ[7])
                .unwrap();
            assert_eq!(decoded, symbol);
        }
    }

    #[test]
    fn arithmetic_finish_with_a_full_carry_tail_round_trips() {
        // The terminator must write `0xff` when a pending carry reaches a
        // full final byte; a random search over short symbol sequences covers
        // that rare path.
        let mut triggering = 0usize;
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for case in 0..20_000u32 {
            let count = 1 + (next() % 40) as usize;
            let symbols: Vec<usize> = (0..count).map(|_| (next() % 17) as usize).collect();
            let mut encoder = AcEncoder::new();
            for &symbol in &symbols {
                encoder.encode(AC_SPEC_CUMFREQ[7][symbol], AC_SPEC_FREQ[7][symbol]);
            }
            encoder.finish();
            if encoder.full_tail_with_carry {
                triggering += 1;
                assert_eq!(
                    encoder.bytes().last(),
                    Some(&0xff),
                    "case {case}: full carry tail"
                );
            }
            let mut payload = encoder.bytes().to_vec();
            payload.extend_from_slice(&[0, 0, 0, 0]);
            let mut reader = crate::lc3::bits::BitReader::new(&payload).unwrap();
            for &symbol in &symbols {
                let decoded = reader
                    .ac_decode(&AC_SPEC_CUMFREQ[7], &AC_SPEC_FREQ[7])
                    .unwrap();
                assert_eq!(decoded, symbol, "case {case}");
            }
        }
        assert!(triggering > 0, "no full carry tail was generated");
    }
}
