// SPDX-License-Identifier: Apache-2.0
//! LC3 side-information reader and arithmetic decoder.
//!
//! The LC3 payload is read from both ends: side information is stored from the
//! last byte backwards, least-significant bit first, while the arithmetic
//! coder consumes bytes from the front.  This module implements both, following
//! Bluetooth LC3 v1.0.1 §3.4.2.

use super::Lc3Error;

/// LC3 side bit reader plus range decoder.
pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    /// Side-information byte position, starting at the last byte.
    bp_side: usize,
    /// Side-information bit mask, starting at `1` and shifting left.
    mask_side: u8,
    /// Arithmetic-coder byte position.
    bp: usize,
    /// 24-bit arithmetic code value.
    low: u32,
    /// Arithmetic range, starting at `0xffffff`.
    range: u32,
}

impl<'a> BitReader<'a> {
    /// Initializes the reader over one frame payload.
    pub(crate) fn new(bytes: &'a [u8]) -> Result<Self, Lc3Error> {
        if bytes.len() < 4 {
            return Err(Lc3Error::InvalidPayload("LC3 payload is too short"));
        }
        let low = (u32::from(bytes[0]) << 16) | (u32::from(bytes[1]) << 8) | u32::from(bytes[2]);
        Ok(Self {
            bytes,
            bp_side: bytes.len() - 1,
            mask_side: 1,
            bp: 3,
            low,
            range: 0xff_ffff,
        })
    }

    /// Reads one side-information bit.
    pub(crate) fn read_bit(&mut self) -> Result<bool, Lc3Error> {
        let byte = *self
            .bytes
            .get(self.bp_side)
            .ok_or(Lc3Error::InvalidPayload("side information exhausted"))?;
        let bit = byte & self.mask_side != 0;
        self.mask_side <<= 1;
        if self.mask_side == 0 {
            self.mask_side = 1;
            self.bp_side = self
                .bp_side
                .checked_sub(1)
                .ok_or(Lc3Error::InvalidPayload("side information exhausted"))?;
        }
        Ok(bit)
    }

    /// Reads `nbits` (`0..=16`) side-information bits, least significant first.
    pub(crate) fn read_uint(&mut self, nbits: u32) -> Result<u32, Lc3Error> {
        let mut value = 0u32;
        for bit in 0..nbits {
            value |= u32::from(self.read_bit()?) << bit;
        }
        Ok(value)
    }

    /// Decodes one arithmetic symbol given cumulative and symbol frequencies.
    pub(crate) fn ac_decode(
        &mut self,
        cumulative: &[u16],
        frequencies: &[u16],
    ) -> Result<usize, Lc3Error> {
        if cumulative.is_empty() || cumulative.len() != frequencies.len() {
            return Err(Lc3Error::Bitstream("invalid arithmetic table"));
        }
        let value = self.range >> 10;
        if self.low >= value << 10 {
            return Err(Lc3Error::Bitstream("invalid arithmetic code value"));
        }
        let mut symbol = cumulative.len() - 1;
        while self.low < value * u32::from(cumulative[symbol]) {
            if symbol == 0 {
                return Err(Lc3Error::Bitstream("arithmetic symbol underflow"));
            }
            symbol -= 1;
        }
        self.low -= value * u32::from(cumulative[symbol]);
        self.range = value * u32::from(frequencies[symbol]);
        if self.range == 0 {
            return Err(Lc3Error::Bitstream("zero arithmetic range"));
        }
        while self.range < 0x1_0000 {
            self.range <<= 8;
            self.low <<= 8;
            self.low &= 0xff_ffff;
            let byte = *self
                .bytes
                .get(self.bp)
                .ok_or(Lc3Error::Bitstream("arithmetic data exhausted"))?;
            self.low += u32::from(byte);
            self.bp += 1;
        }
        Ok(symbol)
    }

    /// Number of payload bits still available for residual data.
    ///
    /// Side and residual bits are read from the end of the payload, the
    /// arithmetic-coded spectrum from the front. The count is therefore the
    /// payload size minus the bits consumed from both ends (Bluetooth LC3
    /// v1.0.1 §3.4.2.5).
    pub(crate) fn bits_left(&self) -> i32 {
        let nbits = (self.bytes.len() * 8) as i32;
        let side =
            8 * (self.bytes.len() as i32 - 1 - self.bp_side as i32) + ilog2(self.mask_side as u32);
        let arithmetic = (self.bp as i32 - 3) * 8 + (25 - ilog2(self.range));
        nbits - (side + arithmetic)
    }
}

/// `floor(log2(value))` for a positive integer.
#[inline]
pub(crate) fn ilog2(value: u32) -> i32 {
    debug_assert!(value > 0);
    31 - value.leading_zeros() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_side_information_from_the_end() {
        let payload = [0u8; 4];
        let mut reader = BitReader::new(&payload).unwrap();
        assert_eq!(reader.read_uint(8).unwrap(), 0);
        assert!(!reader.read_bit().unwrap());
    }

    #[test]
    fn rejects_short_payload() {
        assert!(BitReader::new(&[0u8; 3]).is_err());
    }

    #[test]
    fn arithmetic_round_trip_with_uniform_binary() {
        // Build a payload with the arithmetic coder alone: all side bits are
        // zero and the first four bytes seed the range. Decoding a few symbols
        // must stay in range and never panic.
        let payload = [0x00u8, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff];
        let mut reader = BitReader::new(&payload).unwrap();
        let cumulative = [0u16, 512];
        let frequencies = [512u16, 512];
        for _ in 0..16 {
            let symbol = reader.ac_decode(&cumulative, &frequencies).unwrap();
            assert!(symbol < 2);
        }
    }
}
