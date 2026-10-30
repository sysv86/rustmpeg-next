// SPDX-License-Identifier: Apache-2.0
//! Range decoder for Opus frames, defined by RFC 6716, Section 4.1.
//!
//! Opus entropy-codes its symbols with a range coder that operates on 16-bit
//! cumulative frequency tables.  This module implements the decoder half of
//! that coder, including the alternate entry points of RFC 6716, Section 4.1.3
//! and the raw-bit reader of Section 4.1.4.
//!
//! All arithmetic is performed with the exact 32-bit integer operations
//! specified by the RFC; the decoder keeps the invariant `range > 2**23`
//! between symbols.
//!
//! The decoder continues to read zero bytes once the supplied frame data is
//! exhausted, as required by RFC 6716, Section 4.1.2.1.  Raw bits are read from
//! the end of the same buffer and may overlap the bytes still buffered by the
//! range coder, as described in RFC 6716, Section 4.1.4.

use super::OpusError;

/// Smallest range that is renormalized (`2**23`).
const CODE_BOT: u32 = 1 << 23;

/// Mask applied to `val` after shifting in a byte.
const VAL_MASK: u32 = 0x7fff_ffff;

/// Entropy decoder for one Opus frame.
///
/// A range decoder borrows the frame payload; it copies nothing.  Decoding a
/// stream that is shorter than the symbols would require is allowed by the
/// RFC and returns zero bits, so the caller is responsible for validating how
/// many bits a frame should contain.
#[derive(Debug)]
pub struct RangeDecoder<'a> {
    /// Frame payload the decoder reads from.
    data: &'a [u8],
    /// Next byte to read from the front of `data`.
    offs: usize,
    /// Number of bytes already taken from the end of `data` for raw bits.
    end_offs: usize,
    /// Buffered raw bits, filled least significant byte first.
    end_window: u64,
    /// Number of valid bits in `end_window`.
    nend_bits: u32,
    /// Difference between the top of the current range and the coded value.
    val: u32,
    /// Size of the current range.
    rng: u32,
    /// Bit buffered from the last input byte, used as the top bit of the next
    /// renormalization symbol.
    rem: u32,
    /// Conservative count of bits consumed so far.
    nbits_total: u32,
}

impl<'a> RangeDecoder<'a> {
    /// Creates a decoder over one Opus frame payload.
    ///
    /// The payload is the contents of a single frame, without the packet's TOC
    /// byte or other framing bytes.  An empty slice is valid and behaves as a
    /// stream of zero bytes, as specified for the initialization and
    /// renormalization procedures.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        // RFC 6716, Section 4.1.1: initialize rng to 128 and val to
        // 127 - (b0 >> 1), buffering the low bit of b0 for renormalization.
        let first = data.first().copied().unwrap_or(0);
        let mut decoder = Self {
            data,
            offs: if data.is_empty() { 0 } else { 1 },
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            val: 127 - u32::from(first >> 1),
            rng: 128,
            rem: u32::from(first & 1),
            nbits_total: 9,
        };
        decoder.normalize();
        decoder
    }

    /// Returns the current range, `rng`.
    ///
    /// After initialization and after every symbol this value is greater than
    /// `2**23`, as required by the RFC.  An encoder implementing RFC 6716,
    /// Section 5.1 produces the same value after encoding the same symbols.
    #[must_use]
    pub fn range(&self) -> u32 {
        self.rng
    }

    /// Returns a conservative upper bound, in whole bits, of how much of the
    /// frame has been consumed so far.
    ///
    /// This is `ec_tell()` from RFC 6716, Section 4.1.6.1.  A freshly
    /// initialized decoder reports one bit, the bit reserved for termination.
    #[must_use]
    pub fn tell(&self) -> u32 {
        self.nbits_total.saturating_sub(ilog(self.rng))
    }

    /// Decodes a single binary symbol whose probability of being `1` is
    /// `2**-logp`.
    ///
    /// `logp` must be in `1..=16`, which caps the binary context's total
    /// frequency at `1 << 16` and keeps the `rng / ft` scale exact.  Larger
    /// values are rejected rather than silently losing precision.  Returns
    /// the decoded bit.  This is `ec_dec_bit_logp()` from RFC 6716, Section
    /// 4.1.3.2.
    ///
    /// # Errors
    ///
    /// Returns [`OpusError::RangeDecoder`] if `logp` is outside `1..=16` or the
    /// decoder state cannot represent the symbol's total frequency.
    pub fn decode_bit_logp(&mut self, logp: u32) -> Result<bool, OpusError> {
        if !(1..=16).contains(&logp) {
            return Err(OpusError::RangeDecoder("logp is outside 1..=16"));
        }
        let ft = 1u32 << logp;
        let fs = self.decode(ft)?;
        if fs < ft - 1 {
            self.update(0, ft - 1, ft)?;
            Ok(false)
        } else {
            self.update(ft - 1, ft, ft)?;
            Ok(true)
        }
    }

    /// Decodes a uniformly distributed integer in `0..ft`.
    ///
    /// `ft` must be at least one.  Values with more than eight bits are split
    /// into a range-coded high part and raw low bits, exactly as described in
    /// RFC 6716, Section 4.1.5.
    ///
    /// # Errors
    ///
    /// Returns [`OpusError::RangeDecoder`] if `ft` is zero or the raw bits
    /// decode a value outside `0..ft`, which indicates a corrupt frame.
    pub fn decode_uint(&mut self, ft: u32) -> Result<u32, OpusError> {
        if ft == 0 {
            return Err(OpusError::RangeDecoder("total frequency is zero"));
        }
        let ftb = ilog(ft - 1);
        let value = if ftb <= 8 {
            let value = self.decode(ft)?;
            self.update(value, value + 1, ft)?;
            value
        } else {
            let high_ft = ((ft - 1) >> (ftb - 8)) + 1;
            let high = self.decode(high_ft)?;
            self.update(high, high + 1, high_ft)?;
            let low = self.decode_raw_bits(ftb - 8)?;
            (high << (ftb - 8)) | low
        };
        if value >= ft {
            return Err(OpusError::RangeValue { value, ft });
        }
        Ok(value)
    }

    /// Decodes one symbol using an inverse cumulative distribution table.
    ///
    /// `icdf` holds `(1 << ftb) - fh[k]` for each symbol, in decreasing order,
    /// and must end with zero; `ftb` is at most eight.  Table-based contexts
    /// are the primary interface of the SILK layer (RFC 6716, Section
    /// 4.1.3.3).
    ///
    /// # Errors
    ///
    /// Returns [`OpusError::RangeDecoder`] if `ftb` exceeds eight, the table is
    /// empty, reaches past the total frequency, or is not a valid inverse
    /// distribution.
    pub fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> Result<u8, OpusError> {
        if ftb > 8 {
            return Err(OpusError::RangeDecoder("icdf ftb exceeds 8"));
        }
        if icdf.is_empty() || icdf.len() > 256 {
            return Err(OpusError::RangeDecoder("invalid icdf table length"));
        }
        let ft = 1u32 << ftb;
        let fs = self.decode(ft)?;
        let mut fl = 0u32;
        for (symbol, &entry) in icdf.iter().enumerate() {
            let Some(fh) = ft.checked_sub(u32::from(entry)) else {
                return Err(OpusError::RangeDecoder("icdf entry exceeds total"));
            };
            if fs < fh {
                self.update(fl, fh, ft)?;
                return Ok(symbol as u8);
            }
            fl = fh;
        }
        Err(OpusError::RangeDecoder("icdf table has no matching symbol"))
    }

    /// Computes `fs` for a context whose total frequency is `1 << ftb`.
    ///
    /// This is the value returned by `ec_decode_bin()` (RFC 6716, Section
    /// 4.1.3.1).  The caller must map `fs` to a symbol and then call
    /// [`RangeDecoder::update`] with that symbol's three-tuple.
    ///
    /// # Errors
    ///
    /// Returns [`OpusError::RangeDecoder`] if `ftb` exceeds 15 or the current
    /// range is too small for the total frequency.
    pub fn decode_bin(&mut self, ftb: u32) -> Result<u32, OpusError> {
        // RFC 6716, Section 4.1 bounds every total frequency by 65535, so the
        // largest power-of-two context this entry point accepts is 1 << 15.
        if ftb > 15 {
            return Err(OpusError::RangeDecoder("ftb exceeds 15"));
        }
        self.decode(1u32 << ftb)
    }

    /// Applies the three-tuple `(fl, fh, ft)` of the symbol selected by a
    /// previous [`RangeDecoder::decode_bin`] call.
    ///
    /// This is `ec_dec_update()` from RFC 6716, Section 4.1.2.
    ///
    /// # Errors
    ///
    /// Returns [`OpusError::RangeDecoder`] if the tuple is empty or outside the
    /// current range, or if the range cannot represent it.
    pub fn update(&mut self, fl: u32, fh: u32, ft: u32) -> Result<(), OpusError> {
        if ft == 0 || fl >= fh || fh > ft {
            return Err(OpusError::RangeDecoder("invalid frequency tuple"));
        }
        let scale = self.rng / ft;
        if scale == 0 {
            return Err(OpusError::RangeDecoder("total frequency exceeds range"));
        }
        let high = scale * (ft - fh);
        let Some(val) = self.val.checked_sub(high) else {
            return Err(OpusError::RangeDecoder("decoder value underflow"));
        };
        self.val = val;
        let next_rng = if fl > 0 {
            scale * (fh - fl)
        } else {
            self.rng - high
        };
        if next_rng == 0 {
            return Err(OpusError::RangeDecoder("decoder range collapsed"));
        }
        self.rng = next_rng;
        self.normalize();
        Ok(())
    }

    /// Decodes `bits` raw bits from the end of the frame.
    ///
    /// Raw bits are packed least significant bit first, starting at the least
    /// significant bit of the last byte (RFC 6716, Section 4.1.4), and bypass
    /// the range coder.  `bits` may be zero.
    ///
    /// If the frame is exhausted, the missing bytes are read as zero, exactly
    /// as the range decoder does during renormalization (RFC 6716, Section
    /// 4.1.2.1).  A caller that needs to reject over-long raw-bit reads must
    /// compare `tell()` with the frame length; this method zero-fills instead
    /// of returning an error.
    ///
    /// # Errors
    ///
    /// Returns [`OpusError::RangeDecoder`] if `bits` exceeds 32.
    pub fn decode_raw_bits(&mut self, bits: u32) -> Result<u32, OpusError> {
        if bits > 32 {
            return Err(OpusError::RangeDecoder("raw bit count exceeds 32"));
        }
        if bits == 0 {
            return Ok(0);
        }
        while self.nend_bits < bits {
            let byte = self.read_byte_from_end();
            self.end_window |= u64::from(byte) << self.nend_bits;
            self.nend_bits += 8;
        }
        let value = (self.end_window & ((1u64 << bits) - 1)) as u32;
        self.end_window >>= bits;
        self.nend_bits -= bits;
        self.nbits_total += bits;
        Ok(value)
    }

    /// Computes `fs` for a context with total frequency `ft`.
    fn decode(&mut self, ft: u32) -> Result<u32, OpusError> {
        let scale = self.rng / ft;
        if scale == 0 {
            return Err(OpusError::RangeDecoder("total frequency exceeds range"));
        }
        let fs = self.val / scale + 1;
        Ok(ft - fs.min(ft))
    }

    /// Renormalizes until `rng > 2**23`, reading bytes as needed.
    fn normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.nbits_total += 8;
            self.rng <<= 8;
            let byte = self.read_byte();
            let symbol = (self.rem << 7) | u32::from(byte >> 1);
            self.rem = u32::from(byte & 1);
            self.val = ((self.val << 8) + (255 - symbol)) & VAL_MASK;
        }
    }

    /// Reads the next byte from the front of the frame, or zero past the end.
    fn read_byte(&mut self) -> u8 {
        self.data.get(self.offs).map_or(0, |&byte| {
            self.offs += 1;
            byte
        })
    }

    /// Reads the next byte from the end of the frame, or zero past the start.
    fn read_byte_from_end(&mut self) -> u8 {
        if self.end_offs < self.data.len() {
            self.end_offs += 1;
            self.data[self.data.len() - self.end_offs]
        } else {
            0
        }
    }
}

/// Returns the number of bits needed to store `value` in two's complement
/// notation, or zero for zero (RFC 6716, Section 4.1.5).
pub(crate) fn ilog(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        32 - value.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ilog_matches_bit_width() {
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(2), 2);
        assert_eq!(ilog(3), 2);
        assert_eq!(ilog(255), 8);
        assert_eq!(ilog(256), 9);
        assert_eq!(ilog(u32::MAX), 32);
    }

    #[test]
    fn empty_payload_initializes_with_zero_bytes() {
        let decoder = RangeDecoder::new(&[]);
        assert_eq!(decoder.range(), 1 << 31);
        assert_eq!(decoder.tell(), 1);
    }

    #[test]
    fn initialization_consumes_four_bytes() {
        // The initial renormalization reads b0 plus three more bytes.
        let data = [0x12, 0x34, 0x56, 0x78, 0x9a];
        let decoder = RangeDecoder::new(&data);
        assert_eq!(decoder.offs, 4);
        assert!(decoder.range() > CODE_BOT);
    }

    #[test]
    fn invalid_parameters_are_errors() {
        let mut decoder = RangeDecoder::new(&[]);
        assert!(decoder.decode_bit_logp(0).is_err());
        assert!(decoder.decode_bit_logp(17).is_err());
        assert!(decoder.decode_bit_logp(32).is_err());
        assert!(decoder.decode_uint(0).is_err());
        assert!(decoder.decode_icdf(&[], 4).is_err());
        assert!(decoder.decode_icdf(&[0], 9).is_err());
        assert!(decoder.decode_bin(16).is_err());
        assert!(decoder.decode_bin(17).is_err());
        assert!(decoder.update(1, 1, 2).is_err());
        assert!(decoder.update(2, 1, 2).is_err());
        assert!(decoder.update(0, 3, 2).is_err());
        assert!(decoder.update(0, 1, 0).is_err());
        assert!(decoder.decode_raw_bits(33).is_err());
        assert_eq!(decoder.decode_raw_bits(0).unwrap(), 0);
    }

    #[test]
    fn icdf_table_reaching_past_total_is_rejected() {
        let mut decoder = RangeDecoder::new(&[]);
        // ftb = 4 means ft = 16; 17 is out of range.
        assert!(decoder.decode_icdf(&[17], 4).is_err());
    }
}
