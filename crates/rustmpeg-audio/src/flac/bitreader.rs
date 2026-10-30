// SPDX-License-Identifier: Apache-2.0
//! MSB-first bit reader with FLAC CRC tracking.
//!
//! The reader fetches exactly one byte at a time from the underlying
//! [`Read`], so it never consumes bytes beyond the end of the current frame.
//! Every fetched byte is folded into the frame CRC-16 accumulator; the header
//! CRC-8 accumulator is reset and read separately because the CRC-8 byte
//! itself is excluded from its own check value.

use std::io::Read;

use super::FlacError;
use super::crc::{crc8_update, crc16_update};

/// A byte-at-a-time MSB-first bit reader.
pub(crate) struct BitReader<R> {
    reader: R,
    current: u8,
    bits_left: u8,
    crc8: u8,
    crc16: u16,
    consumed: u64,
    limit: u64,
}

impl<R: Read> BitReader<R> {
    /// Creates a reader with no byte limit.
    pub(crate) fn new(reader: R) -> Self {
        Self {
            reader,
            current: 0,
            bits_left: 0,
            crc8: 0,
            crc16: 0,
            consumed: 0,
            limit: u64::MAX,
        }
    }

    /// Borrows the underlying reader, for callers that seek and then call
    /// [`Self::reset`].
    pub(crate) fn get_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Clears buffered bits and checksums after the underlying reader has been
    /// repositioned.
    pub(crate) fn reset(&mut self) {
        self.current = 0;
        self.bits_left = 0;
        self.crc8 = 0;
        self.crc16 = 0;
        self.consumed = 0;
        self.limit = u64::MAX;
    }

    /// Starts a new frame: resets both checksums and applies the frame byte
    /// limit relative to the current stream position.
    pub(crate) fn begin_frame(&mut self, max_frame_bytes: u64) {
        self.crc8 = 0;
        self.crc16 = 0;
        self.limit = self.consumed.saturating_add(max_frame_bytes);
    }

    /// Returns the number of bytes fetched from the underlying reader so far.
    pub(crate) fn consumed(&self) -> u64 {
        self.consumed
    }

    /// Returns the running CRC-8 of the bytes fetched since [`Self::begin_frame`].
    pub(crate) fn crc8(&self) -> u8 {
        self.crc8
    }

    /// Returns the running CRC-16 of the bytes fetched since [`Self::begin_frame`].
    pub(crate) fn crc16(&self) -> u16 {
        self.crc16
    }

    /// Ensures at least one bit is buffered and reports whether the stream had
    /// data left.  Used at a frame boundary to distinguish a clean end of
    /// stream from an error.
    pub(crate) fn ensure_data(&mut self) -> Result<bool, FlacError> {
        if self.bits_left > 0 {
            return Ok(true);
        }
        match self.fetch_byte(true, true)? {
            Some(byte) => {
                self.current = byte;
                self.bits_left = 8;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Reads one bit.
    pub(crate) fn read_bit(&mut self) -> Result<bool, FlacError> {
        if self.bits_left == 0 {
            self.fill_required()?;
        }
        let bit = self.current & (1 << (self.bits_left - 1)) != 0;
        self.bits_left -= 1;
        Ok(bit)
    }

    /// Reads `count` bits (`0..=64`) as an unsigned MSB-first value.
    pub(crate) fn read_bits(&mut self, count: u32) -> Result<u64, FlacError> {
        if count > 64 {
            return Err(FlacError::InvalidFrame("bit field wider than 64 bits"));
        }
        let mut value = 0u64;
        let mut remaining = count;
        while remaining > 0 {
            if self.bits_left == 0 {
                self.fill_required()?;
            }
            let take = remaining.min(u32::from(self.bits_left));
            let shift = u32::from(self.bits_left) - take;
            let mask = if take == 8 { 0xff } else { (1u16 << take) - 1 };
            let chunk = (u16::from(self.current) >> shift) & mask;
            value = (value << take) | u64::from(chunk);
            self.bits_left -= take as u8;
            remaining -= take;
        }
        Ok(value)
    }

    /// Reads `count` bits (`0..=64`) as a signed two's complement value.
    pub(crate) fn read_signed_bits(&mut self, count: u32) -> Result<i64, FlacError> {
        if count == 0 {
            return Ok(0);
        }
        let value = self.read_bits(count)?;
        if count == 64 {
            return Ok(value as i64);
        }
        let sign_bit = 1u64 << (count - 1);
        if value & sign_bit != 0 {
            Ok((value | (!0u64 << count)) as i64)
        } else {
            Ok(value as i64)
        }
    }

    /// Counts zero bits up to and including the next one bit.
    ///
    /// `max_zeros` bounds the work done for malformed streams; exceeding it is
    /// reported as [`FlacError::LimitExceeded`].
    pub(crate) fn read_unary_zeros(&mut self, max_zeros: u64) -> Result<u64, FlacError> {
        let mut count = 0u64;
        loop {
            if self.bits_left == 0 {
                self.fill_required()?;
            }
            let remaining = self.bits_left;
            let mask = if remaining == 8 {
                0xff
            } else {
                (1u8 << remaining) - 1
            };
            let bits = self.current & mask;
            if bits == 0 {
                count = count
                    .checked_add(u64::from(remaining))
                    .ok_or(FlacError::LimitExceeded("unary code overflow"))?;
                if count > max_zeros {
                    return Err(FlacError::LimitExceeded("unary code is too long"));
                }
                self.bits_left = 0;
            } else {
                let zeros = u32::from(remaining) + bits.leading_zeros() - 8;
                count = count
                    .checked_add(u64::from(zeros))
                    .ok_or(FlacError::LimitExceeded("unary code overflow"))?;
                if count > max_zeros {
                    return Err(FlacError::LimitExceeded("unary code is too long"));
                }
                self.bits_left = remaining - (zeros + 1) as u8;
                return Ok(count);
            }
        }
    }

    /// Reads one byte and folds it into both checksums.
    pub(crate) fn read_u8(&mut self) -> Result<u8, FlacError> {
        self.require_alignment()?;
        match self.fetch_byte(true, true)? {
            Some(byte) => Ok(byte),
            None => Err(unexpected_eof()),
        }
    }

    /// Reads the frame header CRC-8 byte without folding it into the CRC-8
    /// accumulator.  The byte is still part of the frame CRC-16.
    pub(crate) fn read_crc8_byte(&mut self) -> Result<u8, FlacError> {
        self.require_alignment()?;
        match self.fetch_byte(false, true)? {
            Some(byte) => Ok(byte),
            None => Err(unexpected_eof()),
        }
    }

    /// Reads the two-byte frame footer CRC-16 without folding either byte into
    /// either checksum.
    pub(crate) fn read_crc16(&mut self) -> Result<u16, FlacError> {
        self.require_alignment()?;
        let high = match self.fetch_byte(false, false)? {
            Some(byte) => byte,
            None => return Err(unexpected_eof()),
        };
        let low = match self.fetch_byte(false, false)? {
            Some(byte) => byte,
            None => return Err(unexpected_eof()),
        };
        Ok(u16::from_be_bytes([high, low]))
    }

    /// Verifies that any buffered bits are zero and discards them.
    ///
    /// RFC 9639, Section 9.3 zero-pads the frame to a byte boundary before the
    /// frame CRC.
    pub(crate) fn align_zero(&mut self) -> Result<(), FlacError> {
        if self.bits_left == 0 {
            return Ok(());
        }
        let mask = if self.bits_left == 8 {
            0xff
        } else {
            (1u8 << self.bits_left) - 1
        };
        if self.current & mask != 0 {
            return Err(FlacError::InvalidFrame("nonzero frame padding bits"));
        }
        self.bits_left = 0;
        Ok(())
    }

    /// Sets an absolute byte limit.  Only used by tests.
    #[cfg(test)]
    pub(crate) fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn require_alignment(&self) -> Result<(), FlacError> {
        if self.bits_left == 0 {
            Ok(())
        } else {
            Err(FlacError::InvalidFrame(
                "byte-aligned field read while bits are buffered",
            ))
        }
    }

    fn fill_required(&mut self) -> Result<(), FlacError> {
        if self.bits_left > 0 {
            return Ok(());
        }
        match self.fetch_byte(true, true)? {
            Some(byte) => {
                self.current = byte;
                self.bits_left = 8;
                Ok(())
            }
            None => Err(unexpected_eof()),
        }
    }

    fn fetch_byte(
        &mut self,
        update_crc8: bool,
        update_crc16: bool,
    ) -> Result<Option<u8>, FlacError> {
        if self.consumed >= self.limit {
            return Err(FlacError::LimitExceeded("frame exceeds the byte limit"));
        }
        let mut buffer = [0u8; 1];
        loop {
            match self.reader.read(&mut buffer) {
                Ok(0) => return Ok(None),
                Ok(_) => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(FlacError::Io(error)),
            }
        }
        self.consumed += 1;
        if update_crc8 {
            self.crc8 = crc8_update(self.crc8, buffer[0]);
        }
        if update_crc16 {
            self.crc16 = crc16_update(self.crc16, buffer[0]);
        }
        Ok(Some(buffer[0]))
    }
}

fn unexpected_eof() -> FlacError {
    FlacError::Io(std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        "unexpected end of FLAC frame",
    ))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::BitReader;
    use crate::flac::FlacError;
    use crate::flac::crc::{crc8, crc16};

    fn reader(data: &[u8]) -> BitReader<Cursor<Vec<u8>>> {
        BitReader::new(Cursor::new(data.to_vec()))
    }

    #[test]
    fn reads_bits_across_byte_boundaries() {
        let mut bits = reader(&[0b1011_0010, 0b0110_1100]);
        assert_eq!(bits.read_bits(3).unwrap(), 0b101);
        assert_eq!(bits.read_bits(7).unwrap(), 0b1001001);
        assert_eq!(bits.read_bits(6).unwrap(), 0b101100);
    }

    #[test]
    fn reads_signed_fields() {
        let mut bits = reader(&[0b1111_0000, 0b0111_1111]);
        assert_eq!(bits.read_signed_bits(4).unwrap(), -1);
        assert_eq!(bits.read_signed_bits(4).unwrap(), 0);
        assert_eq!(bits.read_signed_bits(7).unwrap(), 0b0111111);
        assert_eq!(bits.read_signed_bits(1).unwrap(), -1);
    }

    #[test]
    fn reads_wide_values() {
        let data = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
        let mut bits = reader(&data);
        assert_eq!(bits.read_bits(64).unwrap(), 0x1234_5678_9abc_def0);
        assert_eq!(bits.read_bits(0).unwrap(), 0);
    }

    #[test]
    fn reads_unary_codes() {
        let mut bits = reader(&[0b0010_0000, 0b0000_0001]);
        assert_eq!(bits.read_unary_zeros(20).unwrap(), 2);
        assert_eq!(bits.read_unary_zeros(20).unwrap(), 12);
    }

    #[test]
    fn rejects_overlong_unary_codes() {
        let mut bits = reader(&[0, 0, 0, 0]);
        assert!(matches!(
            bits.read_unary_zeros(8),
            Err(FlacError::LimitExceeded(_))
        ));
    }

    #[test]
    fn aligns_on_zero_padding_and_rejects_other_bits() {
        let mut good = reader(&[0b1010_0000]);
        good.read_bits(3).unwrap();
        good.align_zero().unwrap();

        let mut bad = reader(&[0b1010_0001]);
        bad.read_bits(3).unwrap();
        assert!(matches!(bad.align_zero(), Err(FlacError::InvalidFrame(_))));
    }

    #[test]
    fn tracks_both_checksums() {
        let mut bits = reader(&[0x11, 0x22]);
        assert_eq!(bits.read_u8().unwrap(), 0x11);
        assert_eq!(bits.crc8(), crc8(&[0x11]));
        assert_eq!(bits.crc16(), crc16(&[0x11]));
        assert_eq!(bits.read_crc8_byte().unwrap(), 0x22);
        assert_eq!(bits.crc8(), crc8(&[0x11]));
        assert_eq!(bits.crc16(), crc16(&[0x11, 0x22]));
    }

    #[test]
    fn byte_reads_require_alignment() {
        let mut bits = reader(&[0xff]);
        bits.read_bit().unwrap();
        assert!(matches!(bits.read_u8(), Err(FlacError::InvalidFrame(_))));
    }

    #[test]
    fn reports_clean_end_of_stream() {
        let mut bits = reader(&[]);
        assert!(!bits.ensure_data().unwrap());
        assert!(matches!(bits.read_bit(), Err(FlacError::Io(_))));
    }

    #[test]
    fn enforces_the_byte_limit() {
        let mut bits = reader(&[0x00, 0x00]);
        bits.set_limit(1);
        assert_eq!(bits.read_bits(8).unwrap(), 0);
        assert!(matches!(
            bits.read_bits(1),
            Err(FlacError::LimitExceeded(_))
        ));
    }
}
