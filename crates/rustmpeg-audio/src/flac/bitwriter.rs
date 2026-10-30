// SPDX-License-Identifier: Apache-2.0
//! MSB-first bit writer used by the FLAC encoder.
//!
//! FLAC stores every field MSB-first (RFC 9639, Section 5).  The writer keeps
//! the partially filled byte so callers can mix bit fields and byte-aligned
//! values without manual bookkeeping.

/// MSB-first bit writer.
#[derive(Debug, Default, Clone)]
pub(crate) struct BitWriter {
    data: Vec<u8>,
    current: u8,
    used: u8,
}

impl BitWriter {
    /// Creates an empty writer with pre-allocated byte capacity.
    pub(crate) fn with_capacity(bytes: usize) -> Self {
        Self {
            data: Vec::with_capacity(bytes),
            current: 0,
            used: 0,
        }
    }

    /// Returns `true` when the writer is byte-aligned.
    pub(crate) fn is_aligned(&self) -> bool {
        self.used == 0
    }

    /// Writes one bit.
    #[inline]
    pub(crate) fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current |= 1 << (7 - self.used);
        }
        self.used += 1;
        if self.used == 8 {
            self.data.push(self.current);
            self.current = 0;
            self.used = 0;
        }
    }

    /// Writes the low `n` (`0..=64`) bits of `value`, most significant first.
    pub(crate) fn write_bits(&mut self, n: u32, value: u64) {
        debug_assert!(n <= 64);
        for shift in (0..n).rev() {
            self.write_bit((value >> shift) & 1 != 0);
        }
    }

    /// Writes `n` (`1..=64`) bits of a signed value in two's complement.
    pub(crate) fn write_signed(&mut self, n: u32, value: i64) {
        debug_assert!((1..=64).contains(&n));
        let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        self.write_bits(n, (value as u64) & mask);
    }

    /// Writes `zeros` zero bits followed by a one bit (`unary`).
    pub(crate) fn write_unary(&mut self, zeros: u32) {
        for _ in 0..zeros {
            self.write_bit(false);
        }
        self.write_bit(true);
    }

    /// Writes a byte slice. The writer must be byte-aligned.
    pub(crate) fn write_bytes(&mut self, bytes: &[u8]) {
        debug_assert!(self.is_aligned());
        self.data.extend_from_slice(bytes);
    }

    /// Pads with zero bits to the next byte boundary.
    pub(crate) fn align_zero(&mut self) {
        if self.used != 0 {
            self.data.push(self.current);
            self.current = 0;
            self.used = 0;
        }
    }

    /// Returns the completed byte buffer, zero-padding the final byte.
    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        self.align_zero();
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::BitWriter;

    #[test]
    fn writes_fields_msb_first() {
        let mut writer = BitWriter::default();
        writer.write_bits(3, 0b101);
        writer.write_bit(true);
        writer.write_bits(4, 0b0011);
        writer.write_signed(5, -3);
        writer.align_zero();
        assert_eq!(writer.into_bytes(), vec![0b1011_0011, 0b1110_1000]);
    }

    #[test]
    fn writes_unary_and_bytes() {
        let mut writer = BitWriter::default();
        writer.write_unary(3);
        writer.write_bits(4, 0b1010);
        writer.align_zero();
        writer.write_bytes(&[0xde, 0xad]);
        assert_eq!(writer.into_bytes(), vec![0b0001_1010, 0xde, 0xad]);
    }

    #[test]
    fn signed_zero_extremes() {
        let mut writer = BitWriter::default();
        writer.write_signed(4, -8);
        writer.write_signed(4, 7);
        assert_eq!(writer.into_bytes(), vec![0x87]);
    }
}
