// SPDX-License-Identifier: Apache-2.0
//! MSB-first bit writer/reader for ITU-T H.266 RBSP payloads.
//!
//! H.266 packs syntax elements MSB first.  `BitWriter` is used for both the
//! parameter sets (where every write is followed by `rbsp_trailing_bits()`) and
//! the slice header (where the write is followed by `byte_alignment()`).
//!
//! Everything in this module is crate-internal: the writer panics on violated
//! invariants, and those invariants are guaranteed by the builders in
//! [`super::hls`] rather than by untrusted input.  The reader returns errors for
//! every malformed input.

/// MSB-first bit writer.
#[derive(Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Bits not yet flushed to `bytes`, right-aligned.
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    /// Create an empty writer with a pre-allocated byte buffer.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
            acc: 0,
            nbits: 0,
        }
    }

    /// Write one bit.
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.acc = (self.acc << 1) | u64::from(bit);
        self.nbits += 1;
        if self.nbits == 8 {
            self.bytes.push(self.acc as u8);
            self.acc = 0;
            self.nbits = 0;
        }
    }

    /// Write the `n` least-significant bits of `value`, MSB first.
    ///
    /// `n` must be in `0..=32`; bits above `n` are ignored (this matches the
    /// permissive VTM `OutputBitstream::write` contract for `n < 32`).
    pub fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            self.write_bit(((value >> i) & 1) != 0);
        }
    }

    /// Unsigned Exp-Golomb `ue(v)` (H.266 9.2).
    pub fn write_ue(&mut self, value: u32) {
        self.write_ue_code(u64::from(value));
    }

    /// Signed Exp-Golomb `se(v)` (H.266 9.2).
    ///
    /// The mapping is evaluated in `u64` so that `i32::MIN`, whose code number
    /// is `2^32`, neither wraps nor panics.
    pub fn write_se(&mut self, value: i32) {
        let code_num = if value <= 0 {
            (-i64::from(value) as u64) * 2
        } else {
            (value as u64) * 2 - 1
        };
        self.write_ue_code(code_num);
    }

    /// Write the `ue(v)` code word for the code number `code_num`.
    fn write_ue_code(&mut self, code_num: u64) {
        let value = code_num + 1;
        let bits = 64 - value.leading_zeros(); // number of significant bits
        for _ in 0..(bits - 1) {
            self.write_bit(false);
        }
        for i in (0..bits).rev() {
            self.write_bit(((value >> i) & 1) != 0);
        }
    }

    /// Number of bits written so far.
    #[cfg(test)]
    pub fn bits_written(&self) -> usize {
        self.bytes.len() * 8 + self.nbits as usize
    }

    /// `true` when no partial byte is pending.
    pub fn is_byte_aligned(&self) -> bool {
        self.nbits == 0
    }

    /// Pad with zero bits until the next byte boundary.
    pub fn align_zero(&mut self) {
        if self.nbits != 0 {
            let pad = 8 - self.nbits;
            self.write_bits(0, pad);
        }
    }

    /// `byte_alignment()` (H.266 7.3.2.11): a single `1` bit then zeros.
    ///
    /// Bit-identical to `rbsp_trailing_bits()` but used mid-structure (between
    /// the slice header and `slice_data()`).
    pub fn byte_alignment(&mut self) {
        self.write_bit(true);
        self.align_zero();
    }

    /// `rbsp_trailing_bits()` (H.266 7.3.2.11) at the end of an RBSP.
    pub fn rbsp_trailing_bits(&mut self) {
        self.write_bit(true);
        self.align_zero();
    }

    /// Append already byte-aligned bytes.
    ///
    /// Internal invariant: callers must be byte-aligned.  Every call site in
    /// this crate aligns first, so this is not reachable from public input.
    pub fn append_bytes(&mut self, bytes: &[u8]) {
        assert!(
            self.is_byte_aligned(),
            "append_bytes requires byte alignment"
        );
        self.bytes.extend_from_slice(bytes);
    }

    /// Finish the writer.
    ///
    /// Internal invariant: callers must end with a trailing-bits or alignment
    /// construct.  Every builder in this crate does, so this is not reachable
    /// from public input.
    pub fn into_bytes(self) -> Vec<u8> {
        assert!(
            self.nbits == 0,
            "BitWriter finished with {} pending bits",
            self.nbits
        );
        self.bytes
    }
}

/// MSB-first bit reader with bounds-checked reads.
///
/// Used for the subset decoder's headers and for the high-level-syntax fields
/// of an SPS.  Every read is checked, so malformed input produces an error
/// instead of a panic.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // bit position
}

impl<'a> BitReader<'a> {
    /// Create a reader over `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Absolute bit position from the start of the buffer.
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    /// Read one bit.
    pub fn read_bit(&mut self) -> Result<bool, &'static str> {
        if self.pos >= self.data.len() * 8 {
            return Err("bit reader exhausted");
        }
        let byte = self.data[self.pos >> 3];
        let bit = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(bit != 0)
    }

    /// Read `n` bits into a `u32`, MSB first.
    pub fn read_bits(&mut self, n: u32) -> Result<u32, &'static str> {
        if n > 32 {
            return Err("read_bits supports at most 32 bits");
        }
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Ok(value)
    }

    /// Unsigned Exp-Golomb `ue(v)`.
    pub fn read_ue(&mut self) -> Result<u32, &'static str> {
        let mut leading = 0u32;
        while !self.read_bit()? {
            leading += 1;
            if leading > 31 {
                return Err("invalid ue(v) prefix");
            }
        }
        if leading == 0 {
            return Ok(0);
        }
        let rest = self.read_bits(leading)?;
        Ok(((1u32 << leading) - 1) + rest)
    }

    /// Signed Exp-Golomb `se(v)`.
    pub fn read_se(&mut self) -> Result<i32, &'static str> {
        let code_num = self.read_ue()?;
        let magnitude = (code_num + 1) >> 1;
        Ok(if code_num & 1 == 0 {
            -(magnitude as i32)
        } else {
            magnitude as i32
        })
    }

    /// Advance to the next byte boundary.
    pub fn byte_align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }

    /// `true` when the reader sits on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.pos & 7 == 0
    }
}

/// Insert RBSP emulation-prevention bytes: a `0x03` before any byte `<= 0x03`
/// that follows two zero bytes (H.266 7.4.2).
pub fn rbsp_to_ebsp(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 16);
    let mut zeros = 0u32;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(b);
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    out
}

/// Remove emulation-prevention bytes (`0x00 0x00 0x03` -> `0x00 0x00`).
pub fn ebsp_to_rbsp(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    let mut i = 0usize;
    while i < ebsp.len() {
        if i + 2 < ebsp.len() && ebsp[i] == 0 && ebsp[i + 1] == 0 && ebsp[i + 2] == 3 {
            out.push(0);
            out.push(0);
            i += 3;
        } else {
            out.push(ebsp[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb_matches_reference_encodings() {
        let cases = [
            (0u32, vec![true]),
            (1, vec![false, true, false]),
            (2, vec![false, true, true]),
        ];
        for (value, bits) in cases {
            let mut w = BitWriter::with_capacity(16);
            w.write_ue(value);
            assert_eq!(w.bits_written(), bits.len());
            w.align_zero();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            for expected in bits {
                assert_eq!(r.read_bit().unwrap(), expected);
            }
        }
    }

    #[test]
    fn signed_exp_golomb_round_trips() {
        for value in [-5i32, -1, 0, 1, 7] {
            let mut w = BitWriter::with_capacity(16);
            w.write_se(value);
            w.align_zero();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_se().unwrap(), value);
        }
    }

    #[test]
    fn signed_exp_golomb_extremes_do_not_overflow() {
        // `se(i32::MIN)` maps to code number 2^32, which does not fit in a
        // `u32`; the writer must emit it instead of panicking in debug builds.
        let mut w = BitWriter::with_capacity(16);
        w.write_se(i32::MIN);
        // 32 leading zeros, then the 33 bits of code number 2^32 + 1.
        assert_eq!(w.bits_written(), 65);
        w.align_zero();
        assert_eq!(w.into_bytes().len(), 9);

        let mut w = BitWriter::with_capacity(16);
        w.write_se(i32::MAX);
        // Code number 2^32 - 3 -> value 2^32 - 2 -> 31 leading zeros + 32 bits.
        assert_eq!(w.bits_written(), 63);
        w.align_zero();
        assert_eq!(w.into_bytes().len(), 8);

        // The code numbers keep the H.266 9.2 mapping.
        for (value, code_num) in [(-1i32, 2u32), (0, 0), (1, 1), (-7, 14), (7, 13)] {
            let significant = 32 - (code_num + 1).leading_zeros();
            let mut w = BitWriter::with_capacity(16);
            w.write_se(value);
            assert_eq!(
                w.bits_written(),
                (2 * significant - 1) as usize,
                "se({value})"
            );
        }
    }

    #[test]
    fn emulation_prevention_round_trips() {
        let rbsp = vec![0u8, 0, 0, 1, 0, 0, 2, 0, 0, 3, 9];
        let ebsp = rbsp_to_ebsp(&rbsp);
        assert_eq!(ebsp_to_rbsp(&ebsp), rbsp);
    }

    #[test]
    fn truncated_input_returns_errors_instead_of_panicking() {
        let empty: [u8; 0] = [];
        let mut r = BitReader::new(&empty);
        assert!(r.read_bit().is_err());
        assert!(r.read_bits(1).is_err());
        assert!(r.read_ue().is_err());

        let data = [0xFFu8; 1];
        let mut r = BitReader::new(&data);
        assert!(r.read_bits(8).is_ok());
        assert!(r.read_bits(8).is_err());
    }
}
