// SPDX-License-Identifier: Apache-2.0
//! ISO/IEC 18181-1 Annex B/C entropy coding (decoder and clean-room encoder).
//!
//! Implements:
//! * context distribution clustering (`read_clusters`, `write_clusters`);
//! * `HybridUintConfig` token + extra-bits integer coding (`decode_uint`, `encode_uint`);
//! * Brotli-style canonical prefix codes up to 15 bits (`PrefixHistogram`);
//! * 12-bit rANS with alias-table lookup and `0x130000` final state verification (`AnsHistogram`);
//! * LZ77 back-references with the 120 2D spatial `SPECIAL_DISTANCES` and RLE fast path;
//!   Lehmer-code permutation decoding (`read_permutation`).

use std::collections::HashSet;
use std::sync::Arc;

use super::JxlError;
use super::bits::{BitReader, BitWriter, U32Specifier};

/// Maximum prefix code length in JPEG XL (ISO/IEC 18181-1 Annex B).
pub const MAX_PREFIX_BITS: usize = 15;
const MAX_TOPLEVEL_BITS: usize = 10;

/// 120 special 2D spatial distance pairs `(dx, dy)` for LZ77 in ISO/IEC 18181-1.
#[rustfmt::skip]
const SPECIAL_DISTANCES: [[i8; 2]; 120] = [
    [0, 1], [1, 0], [1, 1], [-1, 1], [0, 2], [2, 0], [1, 2], [-1, 2], [2, 1], [-2, 1],
    [2, 2], [-2, 2], [0, 3], [3, 0], [1, 3], [-1, 3], [3, 1], [-3, 1], [2, 3], [-2, 3],
    [3, 2], [-3, 2], [0, 4], [4, 0], [1, 4], [-1, 4], [4, 1], [-4, 1], [3, 3], [-3, 3],
    [2, 4], [-2, 4], [4, 2], [-4, 2], [0, 5], [3, 4], [-3, 4], [4, 3], [-4, 3], [5, 0],
    [1, 5], [-1, 5], [5, 1], [-5, 1], [2, 5], [-2, 5], [5, 2], [-5, 2], [4, 4], [-4, 4],
    [3, 5], [-3, 5], [5, 3], [-5, 3], [0, 6], [6, 0], [1, 6], [-1, 6], [6, 1], [-6, 1],
    [2, 6], [-2, 6], [6, 2], [-6, 2], [4, 5], [-4, 5], [5, 4], [-5, 4], [3, 6], [-3, 6],
    [6, 3], [-6, 3], [0, 7], [7, 0], [1, 7], [-1, 7], [5, 5], [-5, 5], [7, 1], [-7, 1],
    [4, 6], [-4, 6], [6, 4], [-6, 4], [2, 7], [-2, 7], [7, 2], [-7, 2], [3, 7], [-3, 7],
    [7, 3], [-7, 3], [5, 6], [-5, 6], [6, 5], [-6, 5], [8, 0], [4, 7], [-4, 7], [7, 4],
    [-7, 4], [8, 1], [8, 2], [6, 6], [-6, 6], [8, 3], [5, 7], [-5, 7], [7, 5], [-7, 5],
    [8, 4], [6, 7], [-6, 7], [7, 6], [-7, 6], [8, 5], [7, 7], [-7, 7], [8, 6], [8, 7],
];

/// Hybrid unsigned integer configuration (`HybridUintConfig` in ISO/IEC 18181-1 Annex B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridUintConfig {
    /// Exponent `k` defining `split = 1 << k`.
    pub split_exponent: u32,
    /// Threshold `1 << split_exponent` below which values are coded directly as tokens.
    pub split: u32,
    /// Number of most-significant extra bits included in the token.
    pub msb_in_token: u32,
    /// Number of least-significant extra bits included in the token.
    pub lsb_in_token: u32,
}

impl HybridUintConfig {
    /// Creates a `HybridUintConfig` with validated parameters.
    #[must_use]
    pub const fn new(split_exponent: u32, msb_in_token: u32, lsb_in_token: u32) -> Self {
        Self {
            split_exponent,
            split: 1u32 << split_exponent,
            msb_in_token,
            lsb_in_token,
        }
    }

    /// Standard `(4, 2, 0)` configuration for general Modular/VarDCT residuals.
    pub const DEFAULT_4_2_0: Self = Self::new(4, 2, 0);

    /// Pure-exponent `(0, 0, 0)` configuration where `token = floor(log2(v)) + 1`.
    pub const EXP_0_0_0: Self = Self::new(0, 0, 0);

    /// Parses a `HybridUintConfig` from the bitstream given `log_alphabet_size`.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError`] on EOF or if `msb_in_token + lsb_in_token > split_exponent`.
    pub fn parse(reader: &mut BitReader<'_>, log_alphabet_size: u32) -> Result<Self, JxlError> {
        let split_bits = add_log2_ceil(log_alphabet_size) as usize;
        let split_exponent = reader.read_bits(split_bits)?;
        let (msb_in_token, lsb_in_token) = if split_exponent != log_alphabet_size {
            let msb_bits = add_log2_ceil(split_exponent) as usize;
            let msb = reader.read_bits(msb_bits)?;
            if msb > split_exponent {
                return Err(JxlError::InvalidBitstream(
                    "HybridUintConfig msb_in_token > split_exponent",
                ));
            }
            let lsb_bits = add_log2_ceil(split_exponent - msb) as usize;
            let lsb = reader.read_bits(lsb_bits)?;
            (msb, lsb)
        } else {
            (0, 0)
        };
        if msb_in_token + lsb_in_token > split_exponent {
            return Err(JxlError::InvalidBitstream(
                "HybridUintConfig msb_in_token + lsb_in_token > split_exponent",
            ));
        }
        Ok(Self::new(split_exponent, msb_in_token, lsb_in_token))
    }

    /// Writes this `HybridUintConfig` to `writer` for a given `log_alphabet_size`.
    pub fn write(&self, writer: &mut BitWriter, log_alphabet_size: u32) {
        let split_bits = add_log2_ceil(log_alphabet_size) as usize;
        writer.write_bits(split_bits, self.split_exponent);
        if self.split_exponent != log_alphabet_size {
            let msb_bits = add_log2_ceil(self.split_exponent) as usize;
            writer.write_bits(msb_bits, self.msb_in_token);
            let lsb_bits = add_log2_ceil(self.split_exponent - self.msb_in_token) as usize;
            writer.write_bits(lsb_bits, self.lsb_in_token);
        }
    }

    /// Decodes the unsigned integer from `token` and extra bits in `reader`.
    #[inline]
    pub fn decode_uint(&self, reader: &mut BitReader<'_>, token: u32) -> Result<u32, JxlError> {
        if token < self.split {
            return Ok(token);
        }
        let n = self.split_exponent - (self.msb_in_token + self.lsb_in_token)
            + ((token - self.split) >> (self.msb_in_token + self.lsb_in_token));
        if n >= 32 {
            return Err(JxlError::InvalidBitstream("HybridUint extra bits >= 32"));
        }
        let rest_bits = u64::from(reader.read_bits(n as usize)?);
        let low_bits = u64::from(token & ((1u32 << self.lsb_in_token) - 1));
        let token_msb = (token >> self.lsb_in_token) & ((1u32 << self.msb_in_token) - 1);
        let top = u64::from(token_msb | (1u32 << self.msb_in_token));
        let result = (((top << n) | rest_bits) << self.lsb_in_token) | low_bits;
        u32::try_from(result).map_err(|_| JxlError::InvalidBitstream("HybridUint value overflow"))
    }

    /// Encodes `value` into `(token, extra_bits_count, extra_bits_value)`.
    #[must_use]
    #[inline]
    pub const fn encode_uint(&self, value: u32) -> (u32, u8, u32) {
        if value < self.split {
            return (value, 0, 0);
        }
        let n_total = 31 - value.leading_zeros();
        let m = self.msb_in_token + self.lsb_in_token;
        let extra_bits = n_total - m;
        let msb_val = (value >> (n_total - self.msb_in_token)) & ((1u32 << self.msb_in_token) - 1);
        let lsb_val = value & ((1u32 << self.lsb_in_token) - 1);
        let token = self.split
            + (((n_total - self.split_exponent) << m) | (msb_val << self.lsb_in_token) | lsb_val);
        let extra_mask = if extra_bits == 0 {
            0
        } else {
            (1u32 << extra_bits) - 1
        };
        let extra_val = (value >> self.lsb_in_token) & extra_mask;
        (token, extra_bits as u8, extra_val)
    }
}

#[derive(Debug, Copy, Clone, Default)]
struct PrefixEntry {
    nested: bool,
    bits_or_mask: u8,
    symbol_or_offset: u16,
}

/// Canonical prefix-code histogram (up to 15-bit codewords).
#[derive(Debug, Clone)]
pub struct PrefixHistogram {
    toplevel_bits: usize,
    toplevel_mask: u32,
    toplevel_entries: Vec<PrefixEntry>,
    second_level_entries: Vec<PrefixEntry>,
}

impl PrefixHistogram {
    /// Creates a histogram that decodes a single constant symbol with 0 bits.
    #[must_use]
    pub fn with_single_symbol(symbol: u16) -> Self {
        let entry = PrefixEntry {
            nested: false,
            bits_or_mask: 0,
            symbol_or_offset: symbol,
        };
        Self {
            toplevel_bits: 0,
            toplevel_mask: 0,
            toplevel_entries: vec![entry],
            second_level_entries: Vec::new(),
        }
    }

    /// Builds a two-level lookup table from per-symbol code lengths (`1..=15`).
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::InvalidBitstream`] if the Kraft sum does not equal `1`.
    pub fn with_code_lengths(code_lengths: Vec<u8>) -> Result<Self, JxlError> {
        let mut syms_for_length: Vec<Vec<u16>> = Vec::with_capacity(MAX_PREFIX_BITS);
        for (sym, len) in code_lengths.into_iter().enumerate() {
            let sym = sym as u16;
            if len > 0 {
                if len as usize > MAX_PREFIX_BITS {
                    return Err(JxlError::InvalidBitstream("prefix code length > 15"));
                }
                if syms_for_length.len() < len as usize {
                    syms_for_length.resize_with(len as usize, Vec::new);
                }
                syms_for_length[len as usize - 1].push(sym);
            }
        }

        let toplevel_bits = syms_for_length.len().min(MAX_TOPLEVEL_BITS);
        let mut entries = vec![PrefixEntry::default(); 1 << toplevel_bits];
        let mut current_bits = 0u32;
        for (idx, syms) in syms_for_length.iter().enumerate().take(toplevel_bits) {
            let shifts = toplevel_bits - 1 - idx;
            for &sym in syms {
                let entry = PrefixEntry {
                    nested: false,
                    bits_or_mask: (idx + 1) as u8,
                    symbol_or_offset: sym,
                };
                let span = 1usize << shifts;
                if (current_bits as usize) + span > entries.len() {
                    return Err(JxlError::InvalidBitstream("oversubscribed prefix tree"));
                }
                entries[current_bits as usize..][..span].fill(entry);
                current_bits += span as u32;
            }
        }

        let mut second_level_entries = Vec::new();
        if toplevel_bits < syms_for_length.len() {
            let mut remaining_entries = Vec::new();
            let mut remaining_entry_bits = 0usize;
            for (idx, syms) in syms_for_length.iter().enumerate().skip(toplevel_bits) {
                if syms.is_empty() {
                    continue;
                }
                let chunk_size_bits = idx + 1 - toplevel_bits;
                let chunk_size = 1usize << chunk_size_bits;
                let mut chunk = Vec::with_capacity(chunk_size);
                if !remaining_entries.is_empty() {
                    let mult = 1usize << (chunk_size_bits - remaining_entry_bits);
                    for entry in remaining_entries {
                        for _ in 0..mult {
                            chunk.push(entry);
                        }
                    }
                }
                for &sym in syms {
                    let entry = PrefixEntry {
                        nested: false,
                        bits_or_mask: (idx + 1) as u8,
                        symbol_or_offset: sym,
                    };
                    chunk.push(entry);
                    if chunk.len() == chunk_size {
                        if current_bits as usize >= entries.len() {
                            return Err(JxlError::InvalidBitstream("oversubscribed prefix tree"));
                        }
                        entries[current_bits as usize] = PrefixEntry {
                            nested: true,
                            bits_or_mask: (chunk_size - 1) as u8,
                            symbol_or_offset: second_level_entries.len() as u16,
                        };
                        vec_reverse_bits(&chunk, &mut second_level_entries);
                        current_bits += 1;
                        chunk = Vec::with_capacity(chunk_size);
                    }
                }
                remaining_entries = chunk;
                remaining_entry_bits = chunk_size_bits;
            }
            if !remaining_entries.is_empty() {
                return Err(JxlError::InvalidBitstream(
                    "incomplete second-level prefix tree",
                ));
            }
        }

        if current_bits != (1u32 << toplevel_bits) {
            return Err(JxlError::InvalidBitstream(
                "incomplete prefix code Kraft sum",
            ));
        }

        let mut toplevel_entries = Vec::with_capacity(entries.len());
        vec_reverse_bits(&entries, &mut toplevel_entries);
        Ok(Self {
            toplevel_bits,
            toplevel_mask: (1u32 << toplevel_bits) - 1,
            toplevel_entries,
            second_level_entries,
        })
    }

    /// Parses a prefix histogram from `reader` for `alphabet_size` symbols.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError`] on malformed prefix tree data.
    pub fn parse(reader: &mut BitReader<'_>, alphabet_size: u32) -> Result<Self, JxlError> {
        if alphabet_size == 1 {
            return Ok(Self::with_single_symbol(0));
        }
        if alphabet_size > (1u32 << MAX_PREFIX_BITS) {
            return Err(JxlError::InvalidBitstream("prefix alphabet size > 32768"));
        }
        let hskip = reader.read_bits(2)?;
        if hskip == 1 {
            Self::parse_simple(reader, alphabet_size)
        } else {
            Self::parse_complex(reader, alphabet_size, hskip)
        }
    }

    fn parse_simple(reader: &mut BitReader<'_>, alphabet_size: u32) -> Result<Self, JxlError> {
        let alphabet_bits = alphabet_size.next_power_of_two().trailing_zeros() as usize;
        let nsym = reader.read_bits(2)? + 1;
        let pairs: Vec<(usize, u8)> = match nsym {
            1 => {
                let sym = reader.read_bits(alphabet_bits)?;
                if sym >= alphabet_size {
                    return Err(JxlError::InvalidBitstream(
                        "prefix simple symbol out of range",
                    ));
                }
                return Ok(Self::with_single_symbol(sym as u16));
            }
            2 => {
                let s0 = reader.read_bits(alphabet_bits)? as usize;
                let s1 = reader.read_bits(alphabet_bits)? as usize;
                if s0 == s1 {
                    return Err(JxlError::InvalidBitstream(
                        "duplicate symbol in simple prefix",
                    ));
                }
                vec![(s0, 1), (s1, 1)]
            }
            3 => {
                let s0 = reader.read_bits(alphabet_bits)? as usize;
                let s1 = reader.read_bits(alphabet_bits)? as usize;
                let s2 = reader.read_bits(alphabet_bits)? as usize;
                if s0 == s1 || s0 == s2 || s1 == s2 {
                    return Err(JxlError::InvalidBitstream(
                        "duplicate symbol in simple prefix",
                    ));
                }
                vec![(s0, 1), (s1, 2), (s2, 2)]
            }
            4 => {
                let s0 = reader.read_bits(alphabet_bits)? as usize;
                let s1 = reader.read_bits(alphabet_bits)? as usize;
                let s2 = reader.read_bits(alphabet_bits)? as usize;
                let s3 = reader.read_bits(alphabet_bits)? as usize;
                let tree_selector = reader.read_bool()?;
                let lens = if tree_selector {
                    [1u8, 2, 3, 3]
                } else {
                    [2u8, 2, 2, 2]
                };
                vec![(s0, lens[0]), (s1, lens[1]), (s2, lens[2]), (s3, lens[3])]
            }
            _ => unreachable!(),
        };

        let mut code_lengths = vec![0u8; alphabet_size as usize];
        for (sym, len) in pairs {
            let slot = code_lengths
                .get_mut(sym)
                .ok_or(JxlError::InvalidBitstream("prefix symbol out of bounds"))?;
            if *slot != 0 {
                return Err(JxlError::InvalidBitstream(
                    "duplicate symbol in simple prefix",
                ));
            }
            *slot = len;
        }
        Self::with_code_lengths(code_lengths)
    }

    fn parse_complex(
        reader: &mut BitReader<'_>,
        alphabet_size: u32,
        hskip: u32,
    ) -> Result<Self, JxlError> {
        const CODE_LENGTH_ORDER: [usize; 18] =
            [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut cl_lens = [0u8; 18];
        let mut bitacc = 0usize;
        let mut nonzero_count = 0usize;
        let mut nonzero_sym = 0usize;

        for idx in CODE_LENGTH_ORDER.into_iter().skip(hskip as usize) {
            let base = reader.read_u32(
                U32Specifier::c(0),
                U32Specifier::c(4),
                U32Specifier::c(3),
                U32Specifier::c(8),
            )? as u8;
            let len = if base == 8 {
                if reader.read_bool()? {
                    if reader.read_bool()? { 5 } else { 1 }
                } else {
                    2
                }
            } else {
                base
            };
            cl_lens[idx] = len;
            if len != 0 {
                nonzero_count += 1;
                nonzero_sym = idx;
                bitacc += 32 >> len;
                if bitacc > 32 {
                    return Err(JxlError::InvalidBitstream(
                        "invalid code-length prefix tree",
                    ));
                }
                if bitacc == 32 {
                    break;
                }
            }
        }

        let cl_hist = if nonzero_count == 1 {
            Self::with_single_symbol(nonzero_sym as u16)
        } else if bitacc != 32 {
            return Err(JxlError::InvalidBitstream(
                "incomplete code-length prefix tree",
            ));
        } else {
            Self::with_code_lengths(cl_lens.to_vec())?
        };

        let mut code_lengths = vec![0u8; alphabet_size as usize];
        let mut bitacc = 0usize;
        let mut prev_sym = 8u8;
        let mut last_nonzero_sym = 8u8;
        let mut last_repeat_count = 0usize;
        let mut repeat_count = 0usize;
        let mut repeat_sym = 0u8;

        for len in &mut code_lengths {
            if repeat_count > 0 {
                *len = repeat_sym;
                repeat_count -= 1;
            } else {
                let sym = cl_hist.read_symbol(reader)? as u8;
                match sym {
                    0 => {}
                    1..=15 => {
                        *len = sym;
                        last_nonzero_sym = sym;
                    }
                    16 => {
                        repeat_count = reader.read_bits(2)? as usize + 3;
                        if prev_sym == 16 {
                            repeat_count += last_repeat_count * 3 - 8;
                            last_repeat_count += repeat_count;
                        } else {
                            last_repeat_count = repeat_count;
                        }
                        repeat_sym = last_nonzero_sym;
                        *len = repeat_sym;
                        repeat_count -= 1;
                    }
                    17 => {
                        repeat_count = reader.read_bits(3)? as usize + 3;
                        if prev_sym == 17 {
                            repeat_count += last_repeat_count * 7 - 16;
                            last_repeat_count += repeat_count;
                        } else {
                            last_repeat_count = repeat_count;
                        }
                        repeat_sym = 0;
                        *len = repeat_sym;
                        repeat_count -= 1;
                    }
                    _ => {
                        return Err(JxlError::InvalidBitstream("invalid code-length symbol"));
                    }
                }
                prev_sym = sym;
            }

            if *len != 0 {
                bitacc += 1usize << MAX_PREFIX_BITS.saturating_sub(*len as usize);
                if bitacc > (1usize << MAX_PREFIX_BITS) {
                    return Err(JxlError::InvalidBitstream("prefix tree Kraft sum > 1"));
                }
                if bitacc == (1usize << MAX_PREFIX_BITS) && repeat_count == 0 {
                    break;
                }
            }
        }

        if bitacc != (1usize << MAX_PREFIX_BITS) || repeat_count > 0 {
            return Err(JxlError::InvalidBitstream("incomplete complex prefix tree"));
        }
        Self::with_code_lengths(code_lengths)
    }

    /// Reads one symbol from `reader` using this prefix histogram.
    #[inline]
    pub fn read_symbol(&self, reader: &mut BitReader<'_>) -> Result<u32, JxlError> {
        let peeked = reader.peek_bits_const::<MAX_PREFIX_BITS>();
        let top_idx = (peeked & self.toplevel_mask) as usize;
        let top_entry = self.toplevel_entries[top_idx];
        if top_entry.nested {
            let chunk_offset = (peeked >> self.toplevel_bits) & u32::from(top_entry.bits_or_mask);
            let second_idx = (u32::from(top_entry.symbol_or_offset) + chunk_offset) as usize;
            let second_entry = self.second_level_entries[second_idx];
            reader.consume_bits(second_entry.bits_or_mask as usize)?;
            Ok(u32::from(second_entry.symbol_or_offset))
        } else {
            reader.consume_bits(top_entry.bits_or_mask as usize)?;
            Ok(u32::from(top_entry.symbol_or_offset))
        }
    }

    /// Returns `Some(symbol)` when this histogram has a 0-bit single symbol.
    #[must_use]
    #[inline]
    pub fn single_symbol(&self) -> Option<u32> {
        if let [
            PrefixEntry {
                nested: false,
                bits_or_mask: 0,
                symbol_or_offset: sym,
            },
        ] = self.toplevel_entries.as_slice()
        {
            Some(u32::from(*sym))
        } else {
            None
        }
    }
}

fn vec_reverse_bits(v: &[PrefixEntry], out: &mut Vec<PrefixEntry>) {
    let len = v.len();
    debug_assert!(len.is_power_of_two());
    let bits = len.trailing_zeros();
    if bits == 0 {
        out.extend_from_slice(v);
        return;
    }
    let shift = usize::BITS - bits;
    for idx in 0..len {
        let rev_idx = idx.reverse_bits() >> shift;
        out.push(v[rev_idx]);
    }
}

/// Safe 12-bit rANS bucket entry (`ANS_LOG_TAB_SIZE = 12`).
#[derive(Debug, Copy, Clone, Default)]
struct AnsBucket {
    alias_symbol: u8,
    alias_cutoff: u8,
    dist: u16,
    alias_offset: u16,
    alias_dist_xor: u16,
}

/// 12-bit rANS distribution histogram.
#[derive(Debug, Clone)]
pub struct AnsHistogram {
    buckets: Vec<AnsBucket>,
    log_bucket_size: u32,
    bucket_mask: u32,
    single_symbol: Option<u32>,
}

impl AnsHistogram {
    /// Parses a 12-bit rANS histogram from `reader` given `log_alphabet_size` (`5..=8`).
    ///
    /// # Errors
    ///
    /// Returns [`JxlError`] on malformed rANS distribution headers.
    pub fn parse(reader: &mut BitReader<'_>, log_alphabet_size: u32) -> Result<Self, JxlError> {
        #[derive(Debug, Clone, Copy)]
        struct WorkingBucket {
            dist: u16,
            alias_symbol: u16,
            alias_offset: u16,
            alias_cutoff: u16,
        }

        let table_size = 1usize << log_alphabet_size;
        let log_bucket_size = 12 - log_alphabet_size;
        let bucket_size = 1u16 << log_bucket_size;

        let alphabet_size: usize;
        let mut dist = vec![0u16; table_size];
        if reader.read_bool()? {
            if reader.read_bool()? {
                // Binary distribution
                let v0 = Self::read_ans_u8(reader)? as usize;
                let v1 = Self::read_ans_u8(reader)? as usize;
                if v0 == v1 {
                    return Err(JxlError::InvalidBitstream(
                        "rANS binary symbols are identical",
                    ));
                }
                alphabet_size = v0.max(v1) + 1;
                if alphabet_size > table_size {
                    return Err(JxlError::InvalidBitstream(
                        "rANS binary symbol > table_size",
                    ));
                }
                let prob = reader.read_bits(12)? as u16;
                dist[v0] = prob;
                dist[v1] = (1u16 << 12) - prob;
            } else {
                // Unary (single-symbol) distribution
                let val = Self::read_ans_u8(reader)? as usize;
                alphabet_size = val + 1;
                if alphabet_size > table_size {
                    return Err(JxlError::InvalidBitstream("rANS unary symbol > table_size"));
                }
                dist[val] = 1u16 << 12;
            }
        } else if reader.read_bool()? {
            // Evenly distributed
            alphabet_size = Self::read_ans_u8(reader)? as usize + 1;
            if alphabet_size > table_size {
                return Err(JxlError::InvalidBitstream("rANS uniform size > table_size"));
            }
            let base = (1usize << 12) / alphabet_size;
            let leftover = (1usize << 12) % alphabet_size;
            dist[0..leftover].fill(base as u16 + 1);
            dist[leftover..alphabet_size].fill(base as u16);
        } else {
            // Compressed distribution
            let mut len = 0usize;
            while len < 3 {
                if reader.read_bool()? {
                    len += 1;
                } else {
                    break;
                }
            }
            let shift = (reader.read_bits(len)? + (1u32 << len) - 1) as i16;
            if shift > 13 {
                return Err(JxlError::InvalidBitstream("rANS shift > 13"));
            }
            alphabet_size = Self::read_ans_u8(reader)? as usize + 3;
            if alphabet_size > table_size {
                return Err(JxlError::InvalidBitstream(
                    "rANS alphabet_size > table_size",
                ));
            }
            let mut repeat_ranges = Vec::new();
            let mut omit_data: Option<(u16, usize)> = None;
            let mut idx = 0usize;
            while idx < alphabet_size {
                dist[idx] = read_ans_prefix(reader)?;
                if dist[idx] == 13 {
                    let repeat_count = Self::read_ans_u8(reader)? as usize + 4;
                    if idx + repeat_count > alphabet_size {
                        return Err(JxlError::InvalidBitstream("rANS repeat range overflow"));
                    }
                    repeat_ranges.push(idx..(idx + repeat_count));
                    idx += repeat_count;
                    continue;
                }
                match &mut omit_data {
                    Some((log, pos)) => {
                        if dist[idx] > *log {
                            *log = dist[idx];
                            *pos = idx;
                        }
                    }
                    slot => {
                        *slot = Some((dist[idx], idx));
                    }
                }
                idx += 1;
            }
            let Some((_, omit_pos)) = omit_data else {
                return Err(JxlError::InvalidBitstream("rANS missing omit symbol"));
            };
            if dist.get(omit_pos + 1) == Some(&13) {
                return Err(JxlError::InvalidBitstream(
                    "rANS repeat immediately after omit",
                ));
            }

            let mut repeat_range_idx = 0usize;
            let mut acc = 0u16;
            let mut prev_dist = 0u16;
            for (i, code) in dist.iter_mut().enumerate() {
                if repeat_range_idx < repeat_ranges.len()
                    && repeat_ranges[repeat_range_idx].start <= i
                {
                    if repeat_ranges[repeat_range_idx].end == i {
                        repeat_range_idx += 1;
                    } else {
                        *code = prev_dist;
                        acc = acc.saturating_add(*code);
                        if acc > (1u16 << 12) {
                            return Err(JxlError::InvalidBitstream("rANS probability sum > 4096"));
                        }
                        continue;
                    }
                }
                if *code == 0 || i == omit_pos {
                    prev_dist = 0;
                    continue;
                }
                if *code > 1 {
                    let zeros = (*code - 1) as i16;
                    let bitcount = (shift - ((12 - zeros) >> 1)).clamp(0, zeros);
                    let extra = reader.read_bits(bitcount as usize)? as u16;
                    *code = (1u16 << zeros) + (extra << (zeros - bitcount));
                }
                prev_dist = *code;
                acc = acc.saturating_add(*code);
                if acc > (1u16 << 12) {
                    return Err(JxlError::InvalidBitstream("rANS probability sum > 4096"));
                }
            }
            dist[omit_pos] = (1u16 << 12) - acc;
        }

        if let Some(single_sym_idx) = dist.iter().position(|&d| d == (1u16 << 12)) {
            let buckets = dist
                .into_iter()
                .enumerate()
                .map(|(i, d)| AnsBucket {
                    dist: d,
                    alias_symbol: single_sym_idx as u8,
                    alias_offset: bucket_size * (i as u16),
                    alias_cutoff: 0,
                    alias_dist_xor: d ^ (1u16 << 12),
                })
                .collect();
            return Ok(Self {
                buckets,
                log_bucket_size,
                bucket_mask: (1u32 << log_bucket_size) - 1,
                single_symbol: Some(single_sym_idx as u32),
            });
        }

        let mut working: Vec<WorkingBucket> = dist
            .into_iter()
            .enumerate()
            .map(|(i, d)| WorkingBucket {
                dist: d,
                alias_symbol: if i < alphabet_size { i as u16 } else { 0 },
                alias_offset: 0,
                alias_cutoff: d,
            })
            .collect();

        let mut underfull = Vec::new();
        let mut overfull = Vec::new();
        for (idx, wb) in working.iter().enumerate() {
            match wb.dist.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(idx),
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => overfull.push(idx),
            }
        }
        while let (Some(o), Some(u)) = (overfull.pop(), underfull.pop()) {
            let by = bucket_size - working[u].alias_cutoff;
            working[o].alias_cutoff -= by;
            working[u].alias_symbol = o as u16;
            working[u].alias_offset = working[o].alias_cutoff;
            match working[o].alias_cutoff.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(o),
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => overfull.push(o),
            }
        }

        let buckets = working
            .iter()
            .enumerate()
            .map(|(idx, wb)| {
                if wb.alias_cutoff == bucket_size {
                    AnsBucket {
                        dist: wb.dist,
                        alias_symbol: idx as u8,
                        alias_offset: 0,
                        alias_cutoff: 0,
                        alias_dist_xor: 0,
                    }
                } else {
                    AnsBucket {
                        dist: wb.dist,
                        alias_symbol: wb.alias_symbol as u8,
                        alias_offset: wb.alias_offset - wb.alias_cutoff,
                        alias_cutoff: wb.alias_cutoff as u8,
                        alias_dist_xor: wb.dist ^ working[wb.alias_symbol as usize].dist,
                    }
                }
            })
            .collect();

        Ok(Self {
            buckets,
            log_bucket_size,
            bucket_mask: (1u32 << log_bucket_size) - 1,
            single_symbol: None,
        })
    }

    fn read_ans_u8(reader: &mut BitReader<'_>) -> Result<u8, JxlError> {
        if reader.read_bool()? {
            let n = reader.read_bits(3)?;
            Ok(((1u32 << n) + reader.read_bits(n as usize)?) as u8)
        } else {
            Ok(0)
        }
    }

    /// Reads one symbol from `reader` and updates the 32-bit rANS `state`.
    #[inline]
    pub fn read_symbol(
        &self,
        reader: &mut BitReader<'_>,
        state: &mut u32,
    ) -> Result<u32, JxlError> {
        let idx = *state & 0x0fff;
        let i = (idx >> self.log_bucket_size) as usize;
        let pos = idx & self.bucket_mask;
        let bucket = self.buckets[i];

        let map_to_alias = pos >= u32::from(bucket.alias_cutoff);
        let (offset, dist_xor) = if map_to_alias {
            (
                u32::from(bucket.alias_offset),
                u32::from(bucket.alias_dist_xor),
            )
        } else {
            (0, 0)
        };
        let dist = u32::from(bucket.dist) ^ dist_xor;
        let symbol = if map_to_alias {
            u32::from(bucket.alias_symbol)
        } else {
            i as u32
        };
        let offset = offset + pos;

        let next_state = (*state >> 12) * dist + offset;
        if next_state < (1u32 << 16) {
            let low16 = reader.read_bits(16)?;
            *state = (next_state << 16) | low16;
        } else {
            *state = next_state;
        }
        Ok(symbol)
    }

    /// Returns `Some(symbol)` when this histogram represents a single symbol with probability 1.
    #[must_use]
    #[inline]
    pub const fn single_symbol(&self) -> Option<u32> {
        self.single_symbol
    }
}

fn read_ans_prefix(reader: &mut BitReader<'_>) -> Result<u16, JxlError> {
    Ok(match reader.read_bits(3)? {
        0 => 10,
        1 => {
            for val in [4, 0, 11, 13] {
                if reader.read_bool()? {
                    return Ok(val);
                }
            }
            12
        }
        2 => 7,
        3 => {
            if reader.read_bool()? {
                1
            } else {
                3
            }
        }
        4 => 6,
        5 => 8,
        6 => 9,
        7 => {
            if reader.read_bool()? {
                2
            } else {
                5
            }
        }
        _ => unreachable!(),
    })
}

#[derive(Debug, Clone)]
enum CoderBackend {
    PrefixCode(Arc<Vec<PrefixHistogram>>),
    Ans {
        dist: Arc<Vec<AnsHistogram>>,
        state: u32,
        initial: bool,
    },
}

impl CoderBackend {
    #[inline]
    fn read_symbol(&mut self, reader: &mut BitReader<'_>, cluster: u8) -> Result<u32, JxlError> {
        match self {
            Self::PrefixCode(dist) => dist[cluster as usize].read_symbol(reader),
            Self::Ans {
                dist,
                state,
                initial,
            } => {
                if *initial {
                    *state = reader.read_bits(32)?;
                    *initial = false;
                }
                dist[cluster as usize].read_symbol(reader, state)
            }
        }
    }

    #[inline]
    fn single_symbol(&self, cluster: u8) -> Option<u32> {
        match self {
            Self::PrefixCode(dist) => dist[cluster as usize].single_symbol(),
            Self::Ans { dist, .. } => dist[cluster as usize].single_symbol(),
        }
    }

    fn begin(&mut self, reader: &mut BitReader<'_>) -> Result<(), JxlError> {
        match self {
            Self::PrefixCode(_) => Ok(()),
            Self::Ans { state, initial, .. } => {
                *state = reader.read_bits(32)?;
                *initial = false;
                Ok(())
            }
        }
    }

    fn finalize(&self) -> Result<(), JxlError> {
        match *self {
            Self::PrefixCode(_) => Ok(()),
            Self::Ans { state, initial, .. } => {
                if initial || state == 0x0013_0000 {
                    Ok(())
                } else {
                    Err(JxlError::InvalidBitstream(
                        "rANS final checksum state != 0x130000",
                    ))
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Lz77State {
    lz_len_conf: HybridUintConfig,
    window: Vec<u32>,
    num_to_copy: u32,
    copy_pos: u32,
    num_decoded: u32,
}

impl Lz77State {
    fn new(lz_len_conf: HybridUintConfig) -> Self {
        Self {
            lz_len_conf,
            window: Vec::new(),
            num_to_copy: 0,
            copy_pos: 0,
            num_decoded: 0,
        }
    }
}

#[derive(Debug, Clone)]
enum Lz77Mode {
    Disabled,
    Enabled {
        min_symbol: u32,
        min_length: u32,
        state: Lz77State,
    },
}

impl Lz77Mode {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        if reader.read_bool()? {
            let min_symbol = reader.read_u32(
                U32Specifier::c(224),
                U32Specifier::c(512),
                U32Specifier::c(4096),
                U32Specifier::u(8, 15),
            )?;
            let min_length = reader.read_u32(
                U32Specifier::c(3),
                U32Specifier::c(4),
                U32Specifier::u(5, 2),
                U32Specifier::u(9, 8),
            )?;
            let lz_len_conf = HybridUintConfig::parse(reader, 8)?;
            Ok(Self::Enabled {
                min_symbol,
                min_length,
                state: Lz77State::new(lz_len_conf),
            })
        } else {
            Ok(Self::Disabled)
        }
    }
}

/// Multi-context JPEG XL entropy decoder (PrefixCode + rANS + HybridUint + LZ77).
#[derive(Debug, Clone)]
pub struct EntropyDecoder {
    lz77: Lz77Mode,
    clusters: Vec<u8>,
    configs: Vec<HybridUintConfig>,
    code: CoderBackend,
}

impl EntropyDecoder {
    /// Parses an entropy decoder configuration for `num_dist` contexts from `reader`.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError`] on invalid cluster maps, histograms, or integer configs.
    pub fn parse(reader: &mut BitReader<'_>, num_dist: u32) -> Result<Self, JxlError> {
        let lz77 = Lz77Mode::parse(reader)?;
        let actual_num_dist = match lz77 {
            Lz77Mode::Disabled => num_dist,
            Lz77Mode::Enabled { .. } => num_dist + 1,
        };
        Self::parse_inner(reader, actual_num_dist, lz77)
    }

    fn parse_assume_no_lz77(reader: &mut BitReader<'_>, num_dist: u32) -> Result<Self, JxlError> {
        if reader.read_bool()? {
            return Err(JxlError::InvalidBitstream(
                "LZ77 not allowed in nested cluster histogram",
            ));
        }
        Self::parse_inner(reader, num_dist, Lz77Mode::Disabled)
    }

    fn parse_inner(
        reader: &mut BitReader<'_>,
        num_dist: u32,
        lz77: Lz77Mode,
    ) -> Result<Self, JxlError> {
        let (num_clusters, clusters) = read_clusters(reader, num_dist)?;
        let use_prefix_code = reader.read_bool()?;
        let log_alphabet_size = if use_prefix_code {
            15
        } else {
            reader.read_bits(2)? + 5
        };
        let mut configs = Vec::with_capacity(num_clusters as usize);
        for _ in 0..num_clusters {
            configs.push(HybridUintConfig::parse(reader, log_alphabet_size)?);
        }
        let code = if use_prefix_code {
            let mut counts = Vec::with_capacity(num_clusters as usize);
            for _ in 0..num_clusters {
                let count = if reader.read_bool()? {
                    let n = reader.read_bits(4)? as usize;
                    1 + (1u32 << n) + reader.read_bits(n)?
                } else {
                    1
                };
                if count > (1u32 << 15) {
                    return Err(JxlError::InvalidBitstream("prefix alphabet count > 32768"));
                }
                counts.push(count);
            }
            let mut dist = Vec::with_capacity(num_clusters as usize);
            for count in counts {
                dist.push(PrefixHistogram::parse(reader, count)?);
            }
            CoderBackend::PrefixCode(Arc::new(dist))
        } else {
            let mut dist = Vec::with_capacity(num_clusters as usize);
            for _ in 0..num_clusters {
                dist.push(AnsHistogram::parse(reader, log_alphabet_size)?);
            }
            CoderBackend::Ans {
                dist: Arc::new(dist),
                state: 0,
                initial: true,
            }
        };
        Ok(Self {
            lz77,
            clusters,
            configs,
            code,
        })
    }

    /// Initializes the entropy stream state (reads the 32-bit initial state for rANS).
    pub fn begin(&mut self, reader: &mut BitReader<'_>) -> Result<(), JxlError> {
        self.code.begin(reader)
    }

    /// Verifies the final stream state (`0x130000` for rANS; no-op for PrefixCode).
    pub fn finalize(&self) -> Result<(), JxlError> {
        self.code.finalize()
    }

    /// Context-to-cluster mapping array.
    #[must_use]
    #[inline]
    pub fn cluster_map(&self) -> &[u8] {
        &self.clusters
    }

    /// Returns `Some(token)` if `cluster` deterministically produces a single constant value
    /// with 0 extra bits and LZ77 is disabled.
    #[must_use]
    #[inline]
    pub fn single_token(&self, cluster: u8) -> Option<u32> {
        if matches!(self.lz77, Lz77Mode::Enabled { .. }) {
            return None;
        }
        let sym = self.code.single_symbol(cluster)?;
        let conf = &self.configs[cluster as usize];
        (sym < conf.split).then_some(sym)
    }

    /// Reads one unsigned integer from context `ctx` with distance multiplier `0`.
    #[inline]
    pub fn read_varint(&mut self, reader: &mut BitReader<'_>, ctx: u32) -> Result<u32, JxlError> {
        self.read_varint_with_multiplier(reader, ctx, 0)
    }

    /// Reads one unsigned integer from context `ctx` with LZ77 `dist_multiplier`.
    #[inline]
    pub fn read_varint_with_multiplier(
        &mut self,
        reader: &mut BitReader<'_>,
        ctx: u32,
        dist_multiplier: u32,
    ) -> Result<u32, JxlError> {
        let cluster = self.clusters[ctx as usize];
        self.read_varint_with_multiplier_clustered(reader, cluster, dist_multiplier)
    }

    /// Reads one unsigned integer from pre-mapped `cluster` with LZ77 `dist_multiplier`.
    pub fn read_varint_with_multiplier_clustered(
        &mut self,
        reader: &mut BitReader<'_>,
        cluster: u8,
        dist_multiplier: u32,
    ) -> Result<u32, JxlError> {
        match &mut self.lz77 {
            Lz77Mode::Disabled => {
                let token = self.code.read_symbol(reader, cluster)?;
                self.configs[cluster as usize].decode_uint(reader, token)
            }
            Lz77Mode::Enabled {
                min_symbol,
                min_length,
                state,
            } => {
                let min_symbol = *min_symbol;
                let min_length = *min_length;
                let lz_dist_cluster = *self.clusters.last().unwrap();
                let r = if state.num_to_copy > 0 {
                    let val = state.window[(state.copy_pos & 0x000f_ffff) as usize];
                    state.copy_pos += 1;
                    state.num_to_copy -= 1;
                    val
                } else {
                    let token = self.code.read_symbol(reader, cluster)?;
                    if token >= min_symbol {
                        if state.num_decoded == 0 {
                            return Err(JxlError::InvalidBitstream(
                                "LZ77 repeat before any decoded symbol",
                            ));
                        }
                        let add_len = state.lz_len_conf.decode_uint(reader, token - min_symbol)?;
                        let num_to_copy = add_len
                            .checked_add(min_length)
                            .ok_or(JxlError::InvalidBitstream("LZ77 copy length overflow"))?;
                        state.num_to_copy = num_to_copy;

                        let dist_token = self.code.read_symbol(reader, lz_dist_cluster)?;
                        let raw_dist = self.configs[lz_dist_cluster as usize]
                            .decode_uint(reader, dist_token)?;
                        let distance = if dist_multiplier == 0 {
                            raw_dist
                        } else if raw_dist < 120 {
                            let [offset, mult] = SPECIAL_DISTANCES[raw_dist as usize];
                            let d = i32::from(offset) + (dist_multiplier as i32) * i32::from(mult);
                            (d - 1).max(0) as u32
                        } else {
                            raw_dist - 120
                        };
                        let clamped = (((1u32 << 20) - 1).min(distance) + 1).min(state.num_decoded);
                        state.copy_pos = state.num_decoded - clamped;
                        let val = state.window[(state.copy_pos & 0x000f_ffff) as usize];
                        state.copy_pos += 1;
                        state.num_to_copy -= 1;
                        val
                    } else {
                        self.configs[cluster as usize].decode_uint(reader, token)?
                    }
                };
                let win_idx = (state.num_decoded & 0x000f_ffff) as usize;
                if state.window.len() <= win_idx {
                    state.window.push(r);
                } else {
                    state.window[win_idx] = r;
                }
                state.num_decoded += 1;
                Ok(r)
            }
        }
    }
}

/// Reads a context-to-cluster mapping (`num_dist` entries) from `reader`.
///
/// # Errors
///
/// Returns [`JxlError`] if cluster IDs have gaps or exceed limits.
pub fn read_clusters(
    reader: &mut BitReader<'_>,
    num_dist: u32,
) -> Result<(u32, Vec<u8>), JxlError> {
    if num_dist == 0 {
        return Err(JxlError::InvalidBitstream("num_dist cannot be zero"));
    }
    if num_dist == 1 {
        return Ok((1, vec![0u8]));
    }
    let cluster = if reader.read_bool()? {
        let nbits = reader.read_bits(2)? as usize;
        let mut out = Vec::with_capacity(num_dist as usize);
        for _ in 0..num_dist {
            out.push(reader.read_bits(nbits)? as u8);
        }
        out
    } else {
        let use_mtf = reader.read_bool()?;
        let mut dec = if num_dist <= 2 {
            EntropyDecoder::parse_assume_no_lz77(reader, 1)?
        } else {
            EntropyDecoder::parse(reader, 1)?
        };
        dec.begin(reader)?;
        let mut ret = Vec::with_capacity(num_dist as usize);
        for _ in 0..num_dist {
            let b = dec.read_varint(reader, 0)?;
            let c = u8::try_from(b).map_err(|_| JxlError::InvalidBitstream("cluster ID >= 256"))?;
            ret.push(c);
        }
        dec.finalize()?;
        if use_mtf {
            let mut mtf: [u8; 256] = std::array::from_fn(|i| i as u8);
            for c in &mut ret {
                let idx = *c as usize;
                *c = mtf[idx];
                mtf.copy_within(0..idx, 1);
                mtf[0] = *c;
            }
        }
        ret
    };

    let num_clusters = u32::from(*cluster.iter().max().unwrap()) + 1;
    let unique: HashSet<u8> = cluster.iter().copied().collect();
    if unique.len() as u32 != num_clusters {
        return Err(JxlError::InvalidBitstream("cluster map contains a hole"));
    }
    Ok((num_clusters, cluster))
}

/// Decodes a Lehmer-coded permutation of `0..size` skipping the first `skip` elements
/// (ISO/IEC 18181-1 Annex B).
///
/// # Errors
///
/// Returns [`JxlError`] if a Lehmer code entry is out of bounds.
pub fn read_permutation(
    reader: &mut BitReader<'_>,
    decoder: &mut EntropyDecoder,
    size: u32,
    skip: u32,
) -> Result<Vec<usize>, JxlError> {
    let end = decoder.read_varint(reader, add_log2_ceil(size).min(7))?;
    if end > size - skip {
        return Err(JxlError::InvalidBitstream("permutation end out of bounds"));
    }
    let mut lehmer = vec![0u32; size as usize];
    let mut prev_val = 0u32;
    for slot in &mut lehmer[skip as usize..(skip + end) as usize] {
        let ctx = (prev_val + 1).next_power_of_two().trailing_zeros().min(7);
        *slot = decoder.read_varint(reader, ctx)?;
        prev_val = *slot;
    }
    let mut temp: Vec<usize> = (0..size as usize).collect();
    let mut permutation = Vec::with_capacity(size as usize);
    for idx in lehmer {
        let idx = idx as usize;
        if idx >= temp.len() {
            return Err(JxlError::InvalidBitstream(
                "Lehmer code index out of bounds",
            ));
        }
        permutation.push(temp.remove(idx));
    }
    Ok(permutation)
}

// ============================================================================
// Clean-room PrefixCode + 12-bit rANS + HybridUint + 2D LZ77 Entropy Encoder
// ============================================================================

/// Entropy coding backend selection for [`EntropyEncoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum JxlEntropyMode {
    /// Canonical Huffman prefix coding (`use_prefix_code = true`).
    #[default]
    Prefix,
    /// 12-bit rANS alias-table coding (`use_prefix_code = false`, `log_alphabet_size = 8`).
    Ans,
}

/// One recorded symbol emission in a cluster during encoding.
#[derive(Debug, Clone, Copy)]
struct RecordedToken {
    cluster: u8,
    token: u16,
    extra_bits: u8,
    extra_val: u32,
}

/// Raw pre-LZ77 symbol emission recorded by the caller.
#[derive(Debug, Clone, Copy)]
struct RawEmission {
    cluster: u8,
    value: u32,
    dist_multiplier: u32,
}

/// Clean-room multi-cluster `PrefixCode` / `12-bit rANS` + `HybridUint` + `2D LZ77` entropy encoder.
#[derive(Debug, Clone)]
pub struct EntropyEncoder {
    clusters: Vec<u8>,
    configs: Vec<HybridUintConfig>,
    mode: JxlEntropyMode,
    lz77_enabled: bool,
    raw_emissions: Vec<RawEmission>,
    recorded: Vec<RecordedToken>,
    frequencies: Vec<Vec<u64>>,
}

impl EntropyEncoder {
    /// Creates an encoder with `clusters` mapping contexts to `configs.len()` clusters.
    #[must_use]
    pub fn new(clusters: Vec<u8>, configs: Vec<HybridUintConfig>) -> Self {
        Self::with_options(clusters, configs, JxlEntropyMode::Prefix, false)
    }

    /// Creates an encoder with explicit `mode` (`Prefix` or `Ans`) and `lz77_enabled`.
    #[must_use]
    pub fn with_options(
        mut clusters: Vec<u8>,
        mut configs: Vec<HybridUintConfig>,
        mode: JxlEntropyMode,
        lz77_enabled: bool,
    ) -> Self {
        if lz77_enabled {
            let dist_cluster = configs.len() as u8;
            clusters.push(dist_cluster);
            configs.push(HybridUintConfig::DEFAULT_4_2_0);
        }
        let num_clusters = configs.len();
        Self {
            clusters,
            configs,
            mode,
            lz77_enabled,
            raw_emissions: Vec::new(),
            recorded: Vec::new(),
            frequencies: vec![Vec::new(); num_clusters],
        }
    }

    /// Creates a single-cluster encoder where all `num_dist` contexts share cluster `0`.
    #[must_use]
    pub fn single_cluster(num_dist: u32, config: HybridUintConfig) -> Self {
        Self::new(vec![0u8; num_dist as usize], vec![config])
    }

    /// Creates a single-cluster encoder with explicit [`JxlEntropyMode`] and `lz77_enabled`.
    #[must_use]
    pub fn single_cluster_with_options(
        num_dist: u32,
        config: HybridUintConfig,
        mode: JxlEntropyMode,
        lz77_enabled: bool,
    ) -> Self {
        Self::with_options(
            vec![0u8; num_dist as usize],
            vec![config],
            mode,
            lz77_enabled,
        )
    }

    /// Records an unsigned integer `value` in context `ctx`.
    #[inline]
    pub fn record_varint(&mut self, ctx: u32, value: u32) {
        self.record_varint_with_multiplier(ctx, value, 0);
    }

    /// Records an unsigned integer `value` in context `ctx` with 2D LZ77 `dist_multiplier`.
    #[inline]
    pub fn record_varint_with_multiplier(&mut self, ctx: u32, value: u32, dist_multiplier: u32) {
        let cluster = self.clusters[ctx as usize];
        if self.lz77_enabled {
            self.raw_emissions.push(RawEmission {
                cluster,
                value,
                dist_multiplier,
            });
        } else {
            self.record_varint_clustered(cluster, value);
        }
    }

    /// Records an unsigned integer `value` directly in `cluster`.
    #[inline]
    pub fn record_varint_clustered(&mut self, cluster: u8, value: u32) {
        let conf = &self.configs[cluster as usize];
        let (token, extra_bits, extra_val) = conf.encode_uint(value);
        self.push_token(cluster, token as u16, extra_bits, extra_val);
    }

    #[inline]
    fn push_token(&mut self, cluster: u8, token: u16, extra_bits: u8, extra_val: u32) {
        let freq = &mut self.frequencies[cluster as usize];
        let tok_idx = token as usize;
        if freq.len() <= tok_idx {
            freq.resize(tok_idx + 1, 0);
        }
        freq[tok_idx] += 1;
        self.recorded.push(RecordedToken {
            cluster,
            token,
            extra_bits,
            extra_val,
        });
    }

    /// Materializes recorded tokens when `lz77_enabled` is active.
    fn build_final_tokens(&self) -> (Vec<RecordedToken>, Vec<Vec<u64>>) {
        if !self.lz77_enabled {
            return (self.recorded.clone(), self.frequencies.clone());
        }
        let num_clusters = self.configs.len();
        let lz_dist_cluster = (num_clusters - 1) as u8;
        let lz_len_conf = HybridUintConfig::DEFAULT_4_2_0;
        let min_symbol = 224u32;
        let min_length = 3usize;
        let max_length = 34usize; // add_len <= 31 -> len_tok <= 23 -> token <= 247 < 256

        let mut recorded = Vec::with_capacity(self.raw_emissions.len());
        let mut frequencies = vec![Vec::new(); num_clusters];
        let push_tok = |rec: &mut Vec<RecordedToken>,
                        freqs: &mut [Vec<u64>],
                        cluster: u8,
                        token: u16,
                        extra_bits: u8,
                        extra_val: u32| {
            let f = &mut freqs[cluster as usize];
            let ti = token as usize;
            if f.len() <= ti {
                f.resize(ti + 1, 0);
            }
            f[ti] += 1;
            rec.push(RecordedToken {
                cluster,
                token,
                extra_bits,
                extra_val,
            });
        };

        // Hash table of 3-value sequences -> recent index in `raw_emissions`
        let n = self.raw_emissions.len();
        let mut hash_table = vec![usize::MAX; 4096];
        let hash3 = |a: u32, b: u32, c: u32| -> usize {
            ((a.wrapping_mul(2654435761) ^ b.wrapping_mul(2246822519) ^ c.wrapping_mul(3266489917))
                >> 20) as usize
                & 4095
        };

        let mut idx = 0usize;
        while idx < n {
            let cur = self.raw_emissions[idx];
            let mut best_len = 0usize;
            let mut best_raw_dist = 0u32;

            if idx > 0 && idx + min_length <= n {
                let max_l = (n - idx).min(max_length);
                // 1. Check immediate predecessor (RLE repeat, distance = 1)
                let prev_pos = idx - 1;
                let mut l = 0usize;
                while l < max_l
                    && self.raw_emissions[prev_pos + l].value == self.raw_emissions[idx + l].value
                {
                    l += 1;
                }
                if l >= min_length {
                    best_len = l;
                    best_raw_dist = encode_lz77_raw_dist(1, cur.dist_multiplier);
                }

                // 2. Check hash-table candidate
                if best_len < max_l {
                    let h = hash3(
                        self.raw_emissions[idx].value,
                        self.raw_emissions[idx + 1].value,
                        self.raw_emissions[idx + 2].value,
                    );
                    let cand = hash_table[h];
                    if cand < idx && idx - cand <= (1usize << 20) {
                        let mut cl = 0usize;
                        while cl < max_l
                            && self.raw_emissions[cand + cl].value
                                == self.raw_emissions[idx + cl].value
                        {
                            cl += 1;
                        }
                        if cl > best_len && cl >= min_length {
                            best_len = cl;
                            let dist = (idx - cand) as u32;
                            best_raw_dist = encode_lz77_raw_dist(dist, cur.dist_multiplier);
                        }
                    }
                }
            }

            if idx + 2 < n {
                let h = hash3(
                    self.raw_emissions[idx].value,
                    self.raw_emissions[idx + 1].value,
                    self.raw_emissions[idx + 2].value,
                );
                hash_table[h] = idx;
            }

            if best_len >= min_length {
                let add_len = (best_len - min_length) as u32;
                let (len_tok, len_eb, len_ev) = lz_len_conf.encode_uint(add_len);
                push_tok(
                    &mut recorded,
                    &mut frequencies,
                    cur.cluster,
                    (min_symbol + len_tok) as u16,
                    len_eb,
                    len_ev,
                );
                let dist_conf = &self.configs[lz_dist_cluster as usize];
                let (dist_tok, dist_eb, dist_ev) = dist_conf.encode_uint(best_raw_dist);
                push_tok(
                    &mut recorded,
                    &mut frequencies,
                    lz_dist_cluster,
                    dist_tok as u16,
                    dist_eb,
                    dist_ev,
                );
                for step in 1..best_len {
                    if idx + step + 2 < n {
                        let h = hash3(
                            self.raw_emissions[idx + step].value,
                            self.raw_emissions[idx + step + 1].value,
                            self.raw_emissions[idx + step + 2].value,
                        );
                        hash_table[h] = idx + step;
                    }
                }
                idx += best_len;
            } else {
                let conf = &self.configs[cur.cluster as usize];
                let (tok, eb, ev) = conf.encode_uint(cur.value);
                push_tok(
                    &mut recorded,
                    &mut frequencies,
                    cur.cluster,
                    tok as u16,
                    eb,
                    ev,
                );
                idx += 1;
            }
        }

        (recorded, frequencies)
    }

    /// Writes the complete entropy-coded header and body to `writer`.
    pub fn finish(&self, writer: &mut BitWriter) -> Result<(), JxlError> {
        let (header_bits, body_bits) = self.encode_split()?;
        writer.append_bits_from(&header_bits);
        writer.append_bits_from(&body_bits);
        Ok(())
    }

    /// Encodes the header (distributions) and symbol stream into two separate
    /// [`BitWriter`] buffers.
    pub fn encode_split(&self) -> Result<(BitWriter, BitWriter), JxlError> {
        let (recorded, frequencies) = self.build_final_tokens();
        let (header, mut bodies) =
            self.encode_from_tokens(&recorded, &frequencies, &[recorded.len()])?;
        let body = bodies.pop().unwrap_or_default();
        Ok((header, body))
    }

    /// Encodes the distribution header once and splits the recorded token stream
    /// into one [`BitWriter`] per group according to `group_token_counts`.
    pub fn encode_split_groups(
        &self,
        group_token_counts: &[usize],
    ) -> Result<(BitWriter, Vec<BitWriter>), JxlError> {
        let (recorded, frequencies) = self.build_final_tokens();
        self.encode_from_tokens(&recorded, &frequencies, group_token_counts)
    }

    fn encode_from_tokens(
        &self,
        recorded: &[RecordedToken],
        frequencies: &[Vec<u64>],
        group_token_counts: &[usize],
    ) -> Result<(BitWriter, Vec<BitWriter>), JxlError> {
        let mut header = BitWriter::new();
        if self.lz77_enabled {
            header.write_bool(true);
            // min_symbol = 224 (selector 0 -> c(224))
            header.write_u32(
                224,
                U32Specifier::c(224),
                U32Specifier::c(512),
                U32Specifier::c(4096),
                U32Specifier::u(8, 15),
            )?;
            // min_length = 3 (selector 0 -> c(3))
            header.write_u32(
                3,
                U32Specifier::c(3),
                U32Specifier::c(4),
                U32Specifier::u(5, 2),
                U32Specifier::u(9, 8),
            )?;
            HybridUintConfig::DEFAULT_4_2_0.write(&mut header, 8);
        } else {
            header.write_bool(false);
        }

        write_clusters(&mut header, &self.clusters)?;

        // Check if all tokens fit in 8-bit alphabet (<= 255) when Ans mode is requested.
        let max_sym = frequencies.iter().map(|f| f.len()).max().unwrap_or(1);
        let effective_mode = if self.mode == JxlEntropyMode::Ans && max_sym <= 256 {
            JxlEntropyMode::Ans
        } else {
            JxlEntropyMode::Prefix
        };

        let num_clusters = self.configs.len();
        match effective_mode {
            JxlEntropyMode::Prefix => {
                header.write_bool(true); // use_prefix_code = true
                for conf in &self.configs {
                    conf.write(&mut header, 15);
                }
                let mut codebooks: Vec<Vec<(u32, u8)>> = Vec::with_capacity(num_clusters);
                let mut alphabet_sizes: Vec<u32> = Vec::with_capacity(num_clusters);

                for freqs in frequencies.iter().take(num_clusters) {
                    let alphabet_size = if freqs.is_empty() {
                        1u32
                    } else {
                        freqs.len() as u32
                    };
                    if alphabet_size > (1u32 << MAX_PREFIX_BITS) {
                        return Err(JxlError::InvalidConfig("prefix alphabet size > 32768"));
                    }
                    alphabet_sizes.push(alphabet_size);
                    write_prefix_alphabet_size(&mut header, alphabet_size);
                }

                for (freqs, &alphabet_size) in frequencies.iter().zip(&alphabet_sizes) {
                    let book = write_prefix_histogram(&mut header, freqs, alphabet_size)?;
                    codebooks.push(book);
                }

                let mut bodies = Vec::with_capacity(group_token_counts.len());
                let mut offset = 0usize;
                for &count in group_token_counts {
                    let mut body = BitWriter::with_capacity(count);
                    for rec in &recorded[offset..offset + count] {
                        let (code, len) = codebooks[rec.cluster as usize][rec.token as usize];
                        if len > 0 {
                            body.write_bits(len as usize, code);
                        }
                        if rec.extra_bits > 0 {
                            body.write_bits(rec.extra_bits as usize, rec.extra_val);
                        }
                    }
                    offset += count;
                    bodies.push(body);
                }
                Ok((header, bodies))
            }
            JxlEntropyMode::Ans => {
                header.write_bool(false); // use_prefix_code = false
                header.write_bits(2, 3); // log_alphabet_size = 5 + 3 = 8 (table_size = 256)
                for conf in &self.configs {
                    conf.write(&mut header, 8);
                }
                let mut ans_tables = Vec::with_capacity(num_clusters);
                for freqs in frequencies.iter().take(num_clusters) {
                    ans_tables.push(write_ans_histogram(&mut header, freqs)?);
                }

                let mut bodies = Vec::with_capacity(group_token_counts.len());
                let mut offset = 0usize;
                for &count in group_token_counts {
                    let slice = &recorded[offset..offset + count];
                    let body = encode_ans_slice(slice, &ans_tables);
                    offset += count;
                    bodies.push(body);
                }
                Ok((header, bodies))
            }
        }
    }
}

fn encode_lz77_raw_dist(distance: u32, dist_multiplier: u32) -> u32 {
    if dist_multiplier == 0 {
        distance.saturating_sub(1)
    } else {
        let target_d = distance as i32;
        for (idx, &[offset, mult]) in SPECIAL_DISTANCES.iter().enumerate() {
            let d = (i32::from(offset) + (dist_multiplier as i32) * i32::from(mult)).max(1);
            if d == target_d {
                return idx as u32;
            }
        }
        distance.saturating_sub(1) + 120
    }
}

/// Inverse alias lookup table for a single 12-bit rANS cluster.
#[derive(Debug, Clone)]
struct AnsEncodeTable {
    dist: Vec<u16>,
    inv_idx: Vec<Vec<u16>>,
}

fn write_ans_u8(writer: &mut BitWriter, val: u8) {
    if val == 0 {
        writer.write_bool(false);
    } else {
        writer.write_bool(true);
        let n = 7 - val.leading_zeros();
        writer.write_bits(3, n);
        writer.write_bits(n as usize, u32::from(val) - (1u32 << n));
    }
}

fn write_ans_prefix(writer: &mut BitWriter, code: u16) {
    match code {
        0 => {
            writer.write_bits(3, 1);
            writer.write_bool(false);
            writer.write_bool(true);
        }
        1 => {
            writer.write_bits(3, 3);
            writer.write_bool(true);
        }
        2 => {
            writer.write_bits(3, 7);
            writer.write_bool(true);
        }
        3 => {
            writer.write_bits(3, 3);
            writer.write_bool(false);
        }
        4 => {
            writer.write_bits(3, 1);
            writer.write_bool(true);
        }
        5 => {
            writer.write_bits(3, 7);
            writer.write_bool(false);
        }
        6 => writer.write_bits(3, 4),
        7 => writer.write_bits(3, 2),
        8 => writer.write_bits(3, 5),
        9 => writer.write_bits(3, 6),
        10 => writer.write_bits(3, 0),
        11 => {
            writer.write_bits(3, 1);
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write_bool(true);
        }
        12 => {
            writer.write_bits(3, 1);
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write_bool(false);
        }
        _ => unreachable!("rANS prefix code must be in 0..=12"),
    }
}

fn write_ans_histogram(writer: &mut BitWriter, freqs: &[u64]) -> Result<AnsEncodeTable, JxlError> {
    const LOG_ALPHA: u32 = 8;
    const TABLE_SIZE: usize = 1 << LOG_ALPHA;
    const LOG_BUCKET: u32 = 12 - LOG_ALPHA;
    const BUCKET_SIZE: u16 = 1 << LOG_BUCKET;

    let nonzero: Vec<(usize, u64)> = freqs
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, f)| f > 0)
        .collect();

    let mut dist = vec![0u16; TABLE_SIZE];
    let alphabet_size: usize;

    if nonzero.len() <= 1 {
        let sym = nonzero.first().map(|&(s, _)| s).unwrap_or(0);
        if sym >= TABLE_SIZE {
            return Err(JxlError::InvalidConfig("rANS unary symbol >= 256"));
        }
        writer.write_bool(true);
        writer.write_bool(false);
        write_ans_u8(writer, sym as u8);
        dist[sym] = 1u16 << 12;
        return Ok(AnsEncodeTable {
            dist,
            inv_idx: Vec::new(),
        });
    } else if nonzero.len() == 2 {
        let (v0, f0) = nonzero[0];
        let (v1, f1) = nonzero[1];
        if v0.max(v1) >= TABLE_SIZE {
            return Err(JxlError::InvalidConfig("rANS binary symbol >= 256"));
        }
        let total = u128::from(f0 + f1);
        let prob0 = ((u128::from(f0) * 4096 + total / 2) / total).clamp(1, 4095) as u16;
        writer.write_bool(true);
        writer.write_bool(true);
        write_ans_u8(writer, v0 as u8);
        write_ans_u8(writer, v1 as u8);
        writer.write_bits(12, u32::from(prob0));
        dist[v0] = prob0;
        dist[v1] = 4096 - prob0;
        alphabet_size = v0.max(v1) + 1;
    } else {
        alphabet_size = (nonzero.iter().map(|&(s, _)| s).max().unwrap() + 1).max(3);
        if alphabet_size > TABLE_SIZE {
            return Err(JxlError::InvalidConfig("rANS alphabet_size > 256"));
        }
        let total: u128 = nonzero.iter().map(|&(_, f)| u128::from(f)).sum();
        let mut allocated = 0u16;
        let mut max_sym = nonzero[0].0;
        let mut max_f = 0u64;
        for &(s, f) in &nonzero {
            if f > max_f {
                max_f = f;
                max_sym = s;
            }
            let p = ((u128::from(f) * 4096) / total).clamp(1, 4095) as u16;
            dist[s] = p;
            allocated += p;
        }
        if allocated < 4096 {
            dist[max_sym] += 4096 - allocated;
        } else if allocated > 4096 {
            let mut excess = allocated - 4096;
            // Reduce largest probabilities first while keeping every active symbol >= 1.
            while excess > 0 {
                for &(s, _) in nonzero.iter().rev() {
                    if dist[s] > 1 && excess > 0 {
                        dist[s] -= 1;
                        excess -= 1;
                    }
                }
            }
        }

        // Compressed rANS header with `shift = 13` (`len = 3`, `bits = 6`) so `bitcount == zeros` is lossless.
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(true);
        writer.write_bool(true);
        writer.write_bool(true);
        writer.write_bits(3, 6); // shift = 6 + (1 << 3) - 1 = 13
        write_ans_u8(writer, (alphabet_size - 3) as u8);

        let mut codes = vec![0u16; alphabet_size];
        let mut omit_data: Option<(u16, usize)> = None;
        for i in 0..alphabet_size {
            let d = dist[i];
            let code = if d == 0 {
                0
            } else {
                (32 - (u32::from(d)).leading_zeros()) as u16
            };
            codes[i] = code;
            write_ans_prefix(writer, code);
            match &mut omit_data {
                Some((log, pos)) => {
                    if code > *log {
                        *log = code;
                        *pos = i;
                    }
                }
                slot => {
                    *slot = Some((code, i));
                }
            }
        }
        let omit_pos = omit_data.unwrap().1;
        for i in 0..alphabet_size {
            let code = codes[i];
            if code <= 1 || i == omit_pos {
                continue;
            }
            let zeros = (code - 1) as usize;
            let extra = dist[i] - (1u16 << zeros);
            writer.write_bits(zeros, u32::from(extra));
        }
    }

    // Build alias table identical to `AnsHistogram::parse` and invert it.
    #[derive(Clone, Copy)]
    struct WorkingBucket {
        dist: u16,
        alias_symbol: u16,
        alias_offset: u16,
        alias_cutoff: u16,
    }

    let mut working: Vec<WorkingBucket> = dist
        .iter()
        .copied()
        .enumerate()
        .map(|(i, d)| WorkingBucket {
            dist: d,
            alias_symbol: if i < alphabet_size { i as u16 } else { 0 },
            alias_offset: 0,
            alias_cutoff: d,
        })
        .collect();

    let mut underfull = Vec::new();
    let mut overfull = Vec::new();
    for (idx, wb) in working.iter().enumerate() {
        match wb.dist.cmp(&BUCKET_SIZE) {
            std::cmp::Ordering::Less => underfull.push(idx),
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => overfull.push(idx),
        }
    }
    while let (Some(o), Some(u)) = (overfull.pop(), underfull.pop()) {
        let by = BUCKET_SIZE - working[u].alias_cutoff;
        working[o].alias_cutoff -= by;
        working[u].alias_symbol = o as u16;
        working[u].alias_offset = working[o].alias_cutoff;
        match working[o].alias_cutoff.cmp(&BUCKET_SIZE) {
            std::cmp::Ordering::Less => underfull.push(o),
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => overfull.push(o),
        }
    }

    let buckets: Vec<AnsBucket> = working
        .iter()
        .enumerate()
        .map(|(idx, wb)| {
            if wb.alias_cutoff == BUCKET_SIZE {
                AnsBucket {
                    dist: wb.dist,
                    alias_symbol: idx as u8,
                    alias_offset: 0,
                    alias_cutoff: 0,
                    alias_dist_xor: 0,
                }
            } else {
                AnsBucket {
                    dist: wb.dist,
                    alias_symbol: wb.alias_symbol as u8,
                    alias_offset: wb.alias_offset - wb.alias_cutoff,
                    alias_cutoff: wb.alias_cutoff as u8,
                    alias_dist_xor: wb.dist ^ working[wb.alias_symbol as usize].dist,
                }
            }
        })
        .collect();

    let mut inv_idx: Vec<Vec<u16>> = dist.iter().map(|&d| vec![0u16; d as usize]).collect();
    let bucket_mask = (1u32 << LOG_BUCKET) - 1;
    for idx in 0..4096u32 {
        let i = (idx >> LOG_BUCKET) as usize;
        let pos = idx & bucket_mask;
        let bucket = buckets[i];
        let map_to_alias = pos >= u32::from(bucket.alias_cutoff);
        let sym = if map_to_alias {
            bucket.alias_symbol as usize
        } else {
            i
        };
        let offset = if map_to_alias {
            u32::from(bucket.alias_offset) + pos
        } else {
            pos
        };
        inv_idx[sym][offset as usize] = idx as u16;
    }

    Ok(AnsEncodeTable { dist, inv_idx })
}

fn encode_ans_slice(recorded: &[RecordedToken], tables: &[AnsEncodeTable]) -> BitWriter {
    let mut state: u32 = 0x0013_0000;
    let mut rev_ops: Vec<(u8, u32)> = Vec::with_capacity(recorded.len() * 2);

    for rec in recorded.iter().rev() {
        if rec.extra_bits > 0 {
            rev_ops.push((rec.extra_bits, rec.extra_val));
        }
        let tbl = &tables[rec.cluster as usize];
        let sym = rec.token as usize;
        let d = u32::from(tbl.dist[sym]);
        if d < 4096 {
            let max_x = u64::from(d) << 20;
            if u64::from(state) >= max_x {
                rev_ops.push((16, state & 0xffff));
                state >>= 16;
            }
            let q = state / d;
            let r = (state % d) as usize;
            let idx = u32::from(tbl.inv_idx[sym][r]);
            state = (q << 12) | idx;
        }
    }

    let mut body = BitWriter::with_capacity(4 + rev_ops.len() * 2);
    body.write_bits(32, state);
    for &(nbits, val) in rev_ops.iter().rev() {
        body.write_bits(nbits as usize, val);
    }
    body
}

/// Writes a context-to-cluster map to `writer`.
///
/// Uses the simple `nbits` encoding when `num_clusters <= 8`, and falls back to
/// an uncompressed prefix-coded cluster stream when `num_clusters > 8`.
pub fn write_clusters(writer: &mut BitWriter, clusters: &[u8]) -> Result<(), JxlError> {
    if clusters.len() <= 1 {
        return Ok(());
    }
    let max_cluster = *clusters.iter().max().unwrap_or(&0);
    if max_cluster < 8 {
        // Simple clustering: 1 bit `true` + 2-bit `nbits` + `nbits` per entry.
        writer.write_bool(true);
        let nbits = if max_cluster == 0 {
            0usize
        } else if max_cluster <= 1 {
            1usize
        } else if max_cluster <= 3 {
            2usize
        } else {
            3usize
        };
        writer.write_bits(2, nbits as u32);
        for &c in clusters {
            writer.write_bits(nbits, u32::from(c));
        }
        Ok(())
    } else {
        // General clustering with `use_mtf = false` and a 1-cluster prefix stream.
        writer.write_bool(false);
        writer.write_bool(false); // use_mtf = false
        let mut nested = EntropyEncoder::single_cluster(1, HybridUintConfig::DEFAULT_4_2_0);
        for &c in clusters {
            nested.record_varint(0, u32::from(c));
        }
        nested.finish(writer)
    }
}

fn write_prefix_alphabet_size(writer: &mut BitWriter, count: u32) {
    if count <= 1 {
        writer.write_bool(false);
    } else {
        writer.write_bool(true);
        let v = count - 1;
        let n = 31 - v.leading_zeros();
        writer.write_bits(4, n);
        writer.write_bits(n as usize, v - (1u32 << n));
    }
}

/// Builds canonical prefix code lengths (`1..=15`) from `freqs`, writes the
/// prefix histogram header (simple or complex) to `writer`, and returns the
/// LSB-reversed `(codeword, bit_length)` table for every symbol `0..alphabet_size`.
fn write_prefix_histogram(
    writer: &mut BitWriter,
    freqs: &[u64],
    alphabet_size: u32,
) -> Result<Vec<(u32, u8)>, JxlError> {
    if alphabet_size == 1 {
        return Ok(vec![(0, 0)]);
    }

    let nonzero: Vec<(usize, u64)> = freqs
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, f)| f > 0)
        .collect();

    let alphabet_bits = alphabet_size.next_power_of_two().trailing_zeros() as usize;

    // 0 or 1 active symbol -> simple prefix with `nsym = 1`.
    if nonzero.len() <= 1 {
        let sym = nonzero.first().map(|&(s, _)| s).unwrap_or(0);
        writer.write_bits(2, 1); // hskip = 1 (simple)
        writer.write_bits(2, 0); // nsym - 1 = 0
        writer.write_bits(alphabet_bits, sym as u32);
        let mut book = vec![(0u32, 0u8); alphabet_size as usize];
        book[sym] = (0, 0);
        return Ok(book);
    }

    // 2 active symbols -> simple prefix with `nsym = 2` (lengths [1, 1]).
    if nonzero.len() == 2 {
        let mut syms = [nonzero[0].0, nonzero[1].0];
        syms.sort_unstable();
        writer.write_bits(2, 1); // hskip = 1 (simple)
        writer.write_bits(2, 1); // nsym - 1 = 1
        writer.write_bits(alphabet_bits, syms[0] as u32);
        writer.write_bits(alphabet_bits, syms[1] as u32);
        let mut lengths = vec![0u8; alphabet_size as usize];
        lengths[syms[0]] = 1;
        lengths[syms[1]] = 1;
        return Ok(build_canonical_codebook(&lengths));
    }

    // 3 active symbols -> simple prefix with `nsym = 3` (lengths [1, 2, 2]).
    if nonzero.len() == 3 {
        let mut sorted = nonzero;
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let s0 = sorted[0].0;
        let mut rest = [sorted[1].0, sorted[2].0];
        rest.sort_unstable();
        writer.write_bits(2, 1); // hskip = 1 (simple)
        writer.write_bits(2, 2); // nsym - 1 = 2
        writer.write_bits(alphabet_bits, s0 as u32);
        writer.write_bits(alphabet_bits, rest[0] as u32);
        writer.write_bits(alphabet_bits, rest[1] as u32);
        let mut lengths = vec![0u8; alphabet_size as usize];
        lengths[s0] = 1;
        lengths[rest[0]] = 2;
        lengths[rest[1]] = 2;
        return Ok(build_canonical_codebook(&lengths));
    }

    // 4 active symbols -> simple prefix with `nsym = 4` (lengths [1, 2, 3, 3] or [2, 2, 2, 2]).
    if nonzero.len() == 4 {
        let mut sorted = nonzero;
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let cost_flat = (sorted[0].1 + sorted[1].1 + sorted[2].1 + sorted[3].1) * 2;
        let cost_skew = sorted[0].1 + sorted[1].1 * 2 + (sorted[2].1 + sorted[3].1) * 3;
        writer.write_bits(2, 1); // hskip = 1 (simple)
        writer.write_bits(2, 3); // nsym - 1 = 3
        let mut lengths = vec![0u8; alphabet_size as usize];
        if cost_skew < cost_flat {
            let s0 = sorted[0].0;
            let s1 = sorted[1].0;
            let mut tail = [sorted[2].0, sorted[3].0];
            tail.sort_unstable();
            writer.write_bits(alphabet_bits, s0 as u32);
            writer.write_bits(alphabet_bits, s1 as u32);
            writer.write_bits(alphabet_bits, tail[0] as u32);
            writer.write_bits(alphabet_bits, tail[1] as u32);
            writer.write_bool(true); // tree_selector = true ([1, 2, 3, 3])
            lengths[s0] = 1;
            lengths[s1] = 2;
            lengths[tail[0]] = 3;
            lengths[tail[1]] = 3;
        } else {
            let mut syms = [sorted[0].0, sorted[1].0, sorted[2].0, sorted[3].0];
            syms.sort_unstable();
            for &s in &syms {
                writer.write_bits(alphabet_bits, s as u32);
                lengths[s] = 2;
            }
            writer.write_bool(false); // tree_selector = false ([2, 2, 2, 2])
        }
        return Ok(build_canonical_codebook(&lengths));
    }

    // >= 5 active symbols -> build length-limited (<= 15 bits) canonical Huffman lengths
    // and write a complex prefix code header (`hskip = 0`).
    let lengths =
        compute_length_limited_code_lengths(freqs, alphabet_size as usize, MAX_PREFIX_BITS);
    write_complex_prefix_lengths(writer, &lengths)?;
    Ok(build_canonical_codebook(&lengths))
}

/// Computes length-limited (`<= max_bits`) prefix code lengths whose Kraft sum
/// `sum(2^(max_bits - len_i))` equals `2^max_bits` exactly.
fn compute_length_limited_code_lengths(
    freqs: &[u64],
    alphabet_size: usize,
    max_bits: usize,
) -> Vec<u8> {
    let mut active: Vec<(usize, u64)> = freqs
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, f)| f > 0)
        .collect();
    let n = active.len();
    debug_assert!(n >= 2);

    // Sort descending by frequency, ties broken by symbol index.
    active.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Initial lengths via standard Huffman tree on active symbols, clamped to `max_bits`.
    let raw_lens = huffman_lengths_for_sorted(&active, max_bits);
    let target_kraft = 1u32 << max_bits;
    let mut kraft: u32 = raw_lens
        .iter()
        .map(|&l| 1u32 << (max_bits - l as usize))
        .sum();

    let mut adj_lens = raw_lens;
    // If clamping caused Kraft sum > target_kraft, lengthen symbols with smallest frequency first.
    while kraft > target_kraft {
        for idx in (0..n).rev() {
            if (adj_lens[idx] as usize) < max_bits {
                let delta = 1u32 << (max_bits - 1 - adj_lens[idx] as usize);
                adj_lens[idx] += 1;
                kraft -= delta;
                if kraft <= target_kraft {
                    break;
                }
            }
        }
    }
    // If Kraft sum < target_kraft, shorten symbols with largest frequency first.
    while kraft < target_kraft {
        let mut changed = false;
        for len_ref in adj_lens.iter_mut().take(n) {
            if *len_ref > 1 {
                let gain = 1u32 << (max_bits - *len_ref as usize);
                if kraft + gain <= target_kraft {
                    *len_ref -= 1;
                    kraft += gain;
                    changed = true;
                    if kraft == target_kraft {
                        break;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut out = vec![0u8; alphabet_size];
    for ((sym, _), len) in active.into_iter().zip(adj_lens) {
        out[sym] = len;
    }
    out
}

fn huffman_lengths_for_sorted(active: &[(usize, u64)], max_bits: usize) -> Vec<u8> {
    let n = active.len();
    if n == 1 {
        return vec![1];
    }
    #[derive(Clone)]
    struct Node {
        weight: u64,
        leaves: Vec<usize>,
    }
    let mut nodes: Vec<Node> = (0..n)
        .map(|i| Node {
            weight: active[i].1.max(1),
            leaves: vec![i],
        })
        .collect();
    let mut depths = vec![0u8; n];
    while nodes.len() > 1 {
        nodes.sort_by_key(|a| std::cmp::Reverse(a.weight));
        let n1 = nodes.pop().unwrap();
        let n2 = nodes.pop().unwrap();
        for &idx in n1.leaves.iter().chain(&n2.leaves) {
            depths[idx] = depths[idx].saturating_add(1);
        }
        let mut merged_leaves = n1.leaves;
        merged_leaves.extend(n2.leaves);
        nodes.push(Node {
            weight: n1.weight.saturating_add(n2.weight),
            leaves: merged_leaves,
        });
    }
    for d in &mut depths {
        *d = (*d).clamp(1, max_bits as u8);
    }
    depths
}

/// Writes a complex prefix code header (`hskip = 0`) for `lengths` (`0..=15`).
fn write_complex_prefix_lengths(writer: &mut BitWriter, lengths: &[u8]) -> Result<(), JxlError> {
    const CODE_LENGTH_ORDER: [usize; 18] =
        [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    // Trim trailing zeros: the decoder stops reading `code_lengths` as soon as
    // `bitacc == 1 << 15`.
    let last_nonzero = lengths.iter().rposition(|&l| l != 0).unwrap_or(0);
    let trimmed = &lengths[..=last_nonzero];

    // Determine which of the 16 literal code-length symbols (0..=15) appear in `trimmed`.
    let mut cl_freqs = [0u64; 18];
    for &l in trimmed {
        cl_freqs[l as usize] += 1;
    }
    let active_cl: Vec<usize> = (0..18).filter(|&s| cl_freqs[s] > 0).collect();

    writer.write_bits(2, 0); // hskip = 0

    let cl_lens: Vec<u8> = if active_cl.len() == 1 {
        let mut v = vec![0u8; 18];
        v[active_cl[0]] = 1;
        v
    } else {
        compute_length_limited_code_lengths(&cl_freqs, 18, 5)
    };

    // Emit `cl_lens` in `CODE_LENGTH_ORDER` until `bitacc == 32` (or single symbol).
    let mut bitacc = 0usize;
    for &sym in &CODE_LENGTH_ORDER {
        let len = cl_lens[sym];
        write_cl_code_length(writer, len);
        if len != 0 && active_cl.len() > 1 {
            bitacc += 32 >> len;
            if bitacc == 32 {
                break;
            }
        }
    }

    if active_cl.len() == 1 {
        // Single code-length symbol consumes 0 bits per entry in `trimmed`!
        return Ok(());
    }

    let cl_book = build_canonical_codebook(&cl_lens);
    for &l in trimmed {
        let (code, bit_len) = cl_book[l as usize];
        writer.write_bits(bit_len as usize, code);
    }
    Ok(())
}

fn write_cl_code_length(writer: &mut BitWriter, len: u8) {
    match len {
        0 => {
            writer.write_bits(2, 0);
        }
        4 => {
            writer.write_bits(2, 1);
        }
        3 => {
            writer.write_bits(2, 2);
        }
        2 => {
            writer.write_bits(2, 3);
            writer.write_bool(false);
        }
        1 => {
            writer.write_bits(2, 3);
            writer.write_bool(true);
            writer.write_bool(false);
        }
        5 => {
            writer.write_bits(2, 3);
            writer.write_bool(true);
            writer.write_bool(true);
        }
        _ => unreachable!("code-length code length must be in 0..=5"),
    }
}

/// Builds the canonical LSB-reversed `(codeword, length)` table from per-symbol `code_lengths`.
#[must_use]
pub fn build_canonical_codebook(code_lengths: &[u8]) -> Vec<(u32, u8)> {
    let mut syms_by_len: Vec<Vec<usize>> = vec![Vec::new(); MAX_PREFIX_BITS + 1];
    for (sym, &len) in code_lengths.iter().enumerate() {
        if len > 0 {
            syms_by_len[len as usize].push(sym);
        }
    }
    let mut book = vec![(0u32, 0u8); code_lengths.len()];
    let mut code = 0u32;
    for (len, syms) in syms_by_len
        .iter()
        .enumerate()
        .take(MAX_PREFIX_BITS + 1)
        .skip(1)
    {
        for &sym in syms {
            let rev = code.reverse_bits() >> (32 - len);
            book[sym] = (rev, len as u8);
            code += 1;
        }
        code <<= 1;
    }
    book
}

fn add_log2_ceil(x: u32) -> u32 {
    if x >= 0x8000_0000 {
        32
    } else {
        (x + 1).next_power_of_two().trailing_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_uint_round_trips() {
        let conf = HybridUintConfig::DEFAULT_4_2_0;
        let mut w = BitWriter::new();
        let mut tokens = Vec::new();
        let values = [0u32, 1, 15, 16, 17, 31, 32, 100, 1024, 65535, 1_000_000];
        for &v in &values {
            let (tok, nbits, extra) = conf.encode_uint(v);
            tokens.push(tok);
            w.write_bits(nbits as usize, extra);
        }
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        for (&v, &tok) in values.iter().zip(&tokens) {
            assert_eq!(conf.decode_uint(&mut r, tok).expect("decode"), v);
        }
    }

    #[test]
    fn entropy_encoder_and_decoder_round_trip_all_alphabet_regimes() {
        for n_distinct in [1usize, 2, 3, 4, 5, 12, 40] {
            let mut enc = EntropyEncoder::new(
                vec![0, 1],
                vec![HybridUintConfig::DEFAULT_4_2_0, HybridUintConfig::EXP_0_0_0],
            );
            let mut expected = Vec::new();
            for i in 0..200u32 {
                let ctx = i % 2;
                let val = (i * 7) % (n_distinct as u32 * 5 + 1);
                enc.record_varint(ctx, val);
                expected.push((ctx, val));
            }
            let mut w = BitWriter::new();
            enc.finish(&mut w).expect("finish");
            let bytes = w.into_bytes();

            let mut r = BitReader::new(&bytes);
            let mut dec = EntropyDecoder::parse(&mut r, 2).expect("parse");
            dec.begin(&mut r).expect("begin");
            for (ctx, val) in expected {
                let got = dec.read_varint(&mut r, ctx).expect("read");
                assert_eq!(got, val, "mismatch for n_distinct={n_distinct}");
            }
            dec.finalize().expect("finalize");
        }
    }

    #[test]
    fn ans_and_lz77_entropy_round_trip_all_regimes() {
        for mode in [JxlEntropyMode::Prefix, JxlEntropyMode::Ans] {
            for lz77 in [false, true] {
                for n_distinct in [1usize, 2, 3, 5, 16] {
                    let mut enc = EntropyEncoder::with_options(
                        vec![0, 1],
                        vec![
                            HybridUintConfig::DEFAULT_4_2_0,
                            HybridUintConfig::DEFAULT_4_2_0,
                        ],
                        mode,
                        lz77,
                    );
                    let mut expected = Vec::new();
                    for i in 0..300u32 {
                        let ctx = i % 2;
                        // Include repetitive runs so LZ77 actively triggers!
                        let val = if (40..90).contains(&i) {
                            777
                        } else if (150..210).contains(&i) {
                            (i % 4) * 11
                        } else {
                            (i * 13) % (n_distinct as u32 * 3 + 1)
                        };
                        enc.record_varint_with_multiplier(ctx, val, 16);
                        expected.push((ctx, val));
                    }
                    let mut w = BitWriter::new();
                    enc.finish(&mut w).expect("finish");
                    let bytes = w.into_bytes();

                    let mut r = BitReader::new(&bytes);
                    let mut dec = EntropyDecoder::parse(&mut r, 2).expect("parse");
                    dec.begin(&mut r).expect("begin");
                    for (ctx, val) in expected {
                        let got = dec
                            .read_varint_with_multiplier(&mut r, ctx, 16)
                            .expect("read");
                        assert_eq!(
                            got, val,
                            "mismatch for mode={mode:?}, lz77={lz77}, n_distinct={n_distinct}"
                        );
                    }
                    dec.finalize().expect("finalize");
                }
            }
        }
    }

    #[test]
    fn test_ans_alpha_gradient_exact() {
        let mut enc = EntropyEncoder::single_cluster_with_options(
            1,
            HybridUintConfig::DEFAULT_4_2_0,
            JxlEntropyMode::Ans,
            false,
        );
        let mut grid = crate::jxl::modular::ModularGrid::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                grid.set(x, y, ((x * 3 + y * 5) & 0xff) as i32);
            }
        }
        let mut expected = Vec::new();
        let mut w = 0i32;
        for x in 0..64 {
            let s = grid.get(x, 0);
            expected.push(crate::jxl::bits::pack_signed(s.wrapping_sub(w)));
            w = s;
        }
        for y in 1..64 {
            let n0 = grid.get(0, y - 1);
            let s0 = grid.get(0, y);
            expected.push(crate::jxl::bits::pack_signed(s0.wrapping_sub(n0)));
            w = s0;
            for x in 1..64 {
                let nw = grid.get(x - 1, y - 1);
                let n = grid.get(x, y - 1);
                let pred = crate::jxl::modular::grad_clamped_i32(n, w, nw);
                let s = grid.get(x, y);
                expected.push(crate::jxl::bits::pack_signed(s.wrapping_sub(pred)));
                w = s;
            }
        }
        for &v in &expected {
            enc.record_varint(0, v);
        }
        let mut bw = BitWriter::new();
        for _ in 0..108 {
            bw.write_bool(true);
        }
        enc.finish(&mut bw).unwrap();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        for _ in 0..108 {
            br.read_bool().unwrap();
        }
        let mut dec = EntropyDecoder::parse(&mut br, 1).unwrap();
        dec.begin(&mut br).unwrap();
        let mut actual = Vec::new();
        for _ in 0..expected.len() {
            actual.push(dec.read_varint(&mut br, 0).unwrap());
        }
        dec.finalize().unwrap();
        assert_eq!(&actual[..16], &expected[..16]);
    }
}
