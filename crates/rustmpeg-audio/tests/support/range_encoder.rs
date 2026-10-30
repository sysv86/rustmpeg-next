// SPDX-License-Identifier: Apache-2.0
//! Test-only range encoder for round-tripping [`rustmpeg_audio::RangeDecoder`].
//!
//! Implemented from the arithmetic of RFC 6716, Section 5.1.  It exists only to
//! validate the decoder: the project does not ship an Opus encoder yet, and
//! this module must not be moved into the crate.
//!
//! The encoder keeps the same `rng` as the RFC's reference algorithm, so tests
//! can compare [`RangeEncoder::range`] with the decoder's range after every
//! symbol (RFC 6716, Section 5.1 states that these values must match exactly).
//! It also implements `ec_tell()` so tests can assert that encoder and decoder
//! agree on the current bit position after every symbol and raw-bit read.

/// Returns the number of bits needed to store `value`, or zero for zero.
fn ilog(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        32 - value.leading_zeros()
    }
}

/// Smallest range that is renormalized (`2**23`).
const CODE_BOT: u32 = 1 << 23;

/// Mask applied to the encoder value after the carry-out shift.
const VAL_MASK: u32 = 0x7fff_ffff;

/// Initial value of `nbits_total`, from which [`RangeEncoder::tell`] subtracts
/// `ilog(rng)`.  RFC 6716, Section 5.1.6 defines `ec_tell()` as
/// `nbits_total - ilog(rng)`; the encoder starts fully normalized, unlike the
/// decoder, so the offset is the decoder's post-initialization value of 33.
const INITIAL_NBITS: u32 = 33;

/// Range encoder over a `Vec<u8>`.
pub struct RangeEncoder {
    /// Low end of the current range.
    val: u32,
    /// Size of the current range.
    rng: u32,
    /// Buffered byte that may still receive a carry, or -1 when empty.
    rem: i32,
    /// Number of buffered 255 bytes awaiting carry propagation.
    ext: u32,
    /// Range-coded output bytes.
    out: Vec<u8>,
    /// Raw bits, packed least significant bit first.
    raw: Vec<u8>,
    /// Number of raw bits written.
    raw_bits: usize,
    /// Bit counter used by [`RangeEncoder::tell`].
    nbits_total: u32,
}

impl RangeEncoder {
    /// Creates an encoder with the RFC's initial state `(0, 2**31, -1, 0)`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            val: 0,
            rng: 1 << 31,
            rem: -1,
            ext: 0,
            out: Vec::new(),
            raw: Vec::new(),
            raw_bits: 0,
            nbits_total: INITIAL_NBITS,
        }
    }

    /// Returns the current range, which must match the decoder's.
    #[must_use]
    pub fn range(&self) -> u32 {
        self.rng
    }

    /// Returns the current bit position, matching the decoder's `tell()`.
    #[must_use]
    pub fn tell(&self) -> u32 {
        self.nbits_total.saturating_sub(ilog(self.rng))
    }

    /// Encodes a symbol with the three-tuple `(fl, fh, ft)`.
    ///
    /// # Panics
    ///
    /// Panics if the tuple is empty or outside `ft`.  Callers in tests pass
    /// valid tuples.
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        assert!(ft > 0);
        assert!(fl < fh && fh <= ft);
        let scale = self.rng / ft;
        assert!(scale > 0);
        if fl > 0 {
            self.val += self.rng - scale * (ft - fl);
            self.rng = scale * (fh - fl);
        } else {
            // RFC 6716, Section 5.1.1 prints `(fh - fl)` in this branch, but
            // that contradicts the decoder's update in Section 4.1.2 and does
            // not preserve the `val + rng` invariant.  This is an erratum; use
            // `(ft - fh)` exactly as the decoder does.
            self.rng -= scale * (ft - fh);
        }
        self.normalize();
    }

    /// Encodes a binary symbol with probability `2**-logp` of being one.
    pub fn encode_bit_logp(&mut self, logp: u32, bit: bool) {
        let ft = 1u32 << logp;
        if bit {
            self.encode(ft - 1, ft, ft);
        } else {
            self.encode(0, ft - 1, ft);
        }
    }

    /// Encodes one symbol from an inverse cumulative distribution table.
    pub fn encode_icdf(&mut self, icdf: &[u8], ftb: u32, symbol: usize) {
        let ft = 1u32 << ftb;
        let fh = ft - u32::from(icdf[symbol]);
        let fl = if symbol == 0 {
            0
        } else {
            ft - u32::from(icdf[symbol - 1])
        };
        self.encode(fl, fh, ft);
    }

    /// Encodes a uniformly distributed integer in `0..ft`.
    pub fn encode_uint(&mut self, ft: u32, value: u32) {
        assert!(value < ft);
        let ftb = ilog(ft - 1);
        if ftb <= 8 {
            self.encode(value, value + 1, ft);
        } else {
            let high_ft = ((ft - 1) >> (ftb - 8)) + 1;
            let shift = ftb - 8;
            let high = value >> shift;
            self.encode(high, high + 1, high_ft);
            self.write_raw_bits(shift, value & ((1u32 << shift) - 1));
        }
    }

    /// Appends `bits` raw bits, least significant bit first.
    pub fn write_raw_bits(&mut self, bits: u32, value: u32) {
        for index in 0..bits {
            let bit = ((value >> index) & 1) as u8;
            let position = self.raw_bits;
            if position / 8 == self.raw.len() {
                self.raw.push(0);
            }
            self.raw[position / 8] |= bit << (position % 8);
            self.raw_bits += 1;
        }
        self.nbits_total += bits;
    }

    /// Finalizes the stream and returns its bytes.
    ///
    /// This follows RFC 6716, Section 5.1.5.  The termination chooses a value
    /// in `[val, val + rng)` with enough trailing zero bits that the highest
    /// raw byte can be ORed into the final range-coded byte.  One zero range
    /// byte is reserved for that purpose (and to guarantee that a frame always
    /// has at least one range-coded byte), after which the remaining raw bytes
    /// are packed towards the end of the frame as Section 5.1.3 requires.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        // `l` is the number of low bits of `val` that are not needed to keep
        // the decoded symbols unique.  `msk` rounds `val` up to a multiple of
        // the corresponding power of two; `(end | msk)` then checks that the
        // whole block lies inside `[val, val + rng)`.
        let mut free_bits = 32 - ilog(self.rng);
        let mut msk = VAL_MASK >> free_bits;
        let mut end = (self.val + msk) & !msk;
        if (end | msk) >= self.val + self.rng {
            free_bits += 1;
            msk >>= 1;
            end = (self.val + msk) & !msk;
        }
        while free_bits > 0 {
            self.carry_out(end >> 23);
            end = (end << 8) & VAL_MASK;
            free_bits = free_bits.saturating_sub(8);
        }
        // Flush the buffered final byte.  `rem >= 0` deliberately includes
        // zero: the reference implementation emits that byte too, and it is
        // the byte the raw bits may be ORed into.  A buffered 255 is never
        // stored in `rem` (it is counted in `ext`), so `carry_out` cannot lose
        // a carry here.
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }

        let raw_len = self.raw_bits.div_ceil(8);
        if raw_len > 0 || self.out.is_empty() {
            // Reserve one range-coded zero byte for the highest raw byte.  The
            // termination above leaves well over eight zero bits below the
            // last emitted range byte, so ORing raw bits into this byte cannot
            // change any decoded symbol.
            self.out.push(0);
        }
        if raw_len == 0 {
            return self.out;
        }

        // Raw bits fill from the end: bit 0 goes into the least significant
        // bit of the last byte, and the highest raw byte meets the range-coded
        // tail at `out.len() - 1` (RFC 6716, Sections 4.1.4 and 5.1.5).
        let total = self.out.len() + raw_len - 1;
        let mut output = vec![0u8; total];
        output[..self.out.len()].copy_from_slice(&self.out);
        for index in 0..self.raw_bits {
            let byte = total - 1 - index / 8;
            let bit = (self.raw[index / 8] >> (index % 8)) & 1;
            output[byte] |= bit << (index % 8);
        }
        output
    }

    /// Propagates one 9-bit value into the output buffer.
    fn carry_out(&mut self, value: u32) {
        if value == 255 {
            // RFC 6716, Section 5.1.1.2: a 255 is deferred in `ext` and is not
            // stored in `rem`, so a later carry can never overflow a buffered
            // 255.
            self.ext += 1;
            return;
        }
        let carry = value >> 8;
        if self.rem >= 0 {
            debug_assert_ne!(self.rem, 255, "a carry-out can never drop a carry");
            self.out.push(((self.rem as u32 + carry) & 0xff) as u8);
        }
        if self.ext > 0 {
            let fill = if carry != 0 { 0 } else { 255 };
            for _ in 0..self.ext {
                self.out.push(fill);
            }
            self.ext = 0;
        }
        self.rem = (value & 0xff) as i32;
    }

    /// Renormalizes until `rng > 2**23`.
    fn normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.carry_out(self.val >> 23);
            self.val = (self.val << 8) & VAL_MASK;
            self.rng <<= 8;
            self.nbits_total += 8;
        }
    }
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}
