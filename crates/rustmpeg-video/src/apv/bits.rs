// SPDX-License-Identifier: Apache-2.0
//! MSB-first bit reader and writer for the APV syntax.

use super::ApvError;

/// Bounds-checked MSB-first bit reader.
#[derive(Debug, Clone)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    pub(crate) fn read_bit(&mut self) -> Result<u32, ApvError> {
        if self.position >= self.data.len() * 8 {
            return Err(ApvError::Eof);
        }
        let byte = self.data[self.position >> 3];
        let bit = (byte >> (7 - (self.position & 7))) & 1;
        self.position += 1;
        Ok(u32::from(bit))
    }

    pub(crate) fn read_bits(&mut self, count: u32) -> Result<u32, ApvError> {
        if count > 32 {
            return Err(ApvError::Invalid("bit count exceeds 32"));
        }
        let mut value = 0_u32;
        for _ in 0..count {
            value = (value << 1) | self.read_bit()?;
        }
        Ok(value)
    }

    pub(crate) fn skip_bits(&mut self, count: usize) -> Result<(), ApvError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(ApvError::Limit("bit position"))?;
        if end > self.data.len() * 8 {
            return Err(ApvError::Eof);
        }
        self.position = end;
        Ok(())
    }

    pub(crate) fn align_zero(&mut self) -> Result<(), ApvError> {
        while !self.is_aligned() {
            if self.read_bit()? != 0 {
                return Err(ApvError::Invalid("non-zero alignment bit"));
            }
        }
        Ok(())
    }

    pub(crate) fn is_aligned(&self) -> bool {
        self.position & 7 == 0
    }

    pub(crate) fn byte_position(&self) -> usize {
        self.position / 8
    }
}

/// MSB-first bit writer.
#[derive(Debug, Clone, Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn write_bit(&mut self, bit: u32) {
        self.current = (self.current << 1) | ((bit & 1) as u8);
        self.used += 1;
        if self.used == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
        }
    }

    pub(crate) fn write_bits(&mut self, value: u32, count: u32) {
        for shift in (0..count).rev() {
            self.write_bit(value >> shift);
        }
    }

    pub(crate) fn align_zero(&mut self) {
        while self.used != 0 {
            self.write_bit(0);
        }
    }

    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        self.align_zero();
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_round_trip() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.write_bits(0x1234, 16);
        writer.write_bit(1);
        writer.align_zero();
        let bytes = writer.into_bytes();

        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_bits(3).expect("bits"), 0b101);
        assert_eq!(reader.read_bits(16).expect("bits"), 0x1234);
        assert_eq!(reader.read_bit().expect("bit"), 1);
        reader.align_zero().expect("alignment");
        assert!(reader.read_bit().is_err());
    }

    #[test]
    fn non_zero_alignment_is_rejected() {
        let mut reader = BitReader::new(&[0b1000_0001]);
        assert_eq!(reader.read_bit().expect("bit"), 1);
        assert!(reader.align_zero().is_err());
    }

    #[test]
    fn reading_past_the_end_fails() {
        let mut reader = BitReader::new(&[0xff]);
        assert_eq!(reader.read_bits(8).expect("bits"), 0xff);
        assert!(reader.read_bit().is_err());
    }
}
