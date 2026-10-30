// SPDX-License-Identifier: Apache-2.0

//! Clean-room RFC 1950 (`zlib`) and RFC 1951 (`DEFLATE`) compressor and decompressor.
//!
//! Implemented from the IETF RFC 1950 and RFC 1951 normative specifications to support
//! OpenEXR `ZIPS` (`2`), `ZIP` (`3`), `PXR24` (`5`), `DWAA` (`8`), and `DWAB` (`9`) codecs
//! without external crates.
//!
//! On compression, `zlib_compress` evaluates Stored (`BTYPE=00`), Fixed Huffman (`BTYPE=01`),
//! and Dynamic Huffman (`BTYPE=10`) encodings and emits whichever produces the smallest output.

use super::ExrError;

const MAX_BITS: usize = 15;

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Computes the RFC 1950 Adler-32 checksum over `data`.
pub fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521;
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;

    for chunk in data.chunks(5552) {
        for &b in chunk {
            s1 += u32::from(b);
            s2 += s1;
        }
        s1 %= MOD_ADLER;
        s2 %= MOD_ADLER;
    }
    (s2 << 16) | s1
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u64,
    bit_count: u32,
}

impl<'a> BitReader<'a> {
    #[inline]
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bit_count: 0,
        }
    }

    #[inline]
    fn ensure_bits(&mut self, n: u32) -> Result<(), ExrError> {
        while self.bit_count < n {
            if self.pos >= self.data.len() {
                return Err(ExrError::CorruptChunk(
                    "unexpected EOF in DEFLATE bitstream",
                ));
            }
            self.bit_buf |= u64::from(self.data[self.pos]) << self.bit_count;
            self.pos += 1;
            self.bit_count += 8;
        }
        Ok(())
    }

    #[inline]
    fn read_bits(&mut self, n: u32) -> Result<u32, ExrError> {
        if n == 0 {
            return Ok(0);
        }
        self.ensure_bits(n)?;
        let val = (self.bit_buf & ((1u64 << n) - 1)) as u32;
        self.bit_buf >>= n;
        self.bit_count -= n;
        Ok(val)
    }

    #[inline]
    fn align_to_byte(&mut self) {
        let drop_bits = self.bit_count & 7;
        self.bit_buf >>= drop_bits;
        self.bit_count -= drop_bits;
    }

    fn read_aligned_u16_le(&mut self) -> Result<u16, ExrError> {
        let b0 = self.read_bits(8)? as u16;
        let b1 = self.read_bits(8)? as u16;
        Ok(b0 | (b1 << 8))
    }

    fn copy_aligned_bytes(&mut self, len: usize, out: &mut Vec<u8>) -> Result<(), ExrError> {
        for _ in 0..len {
            let b = self.read_bits(8)? as u8;
            out.push(b);
        }
        Ok(())
    }
}

/// Canonical Huffman decoder table for RFC 1951 (LSB-first bit order).
struct HuffTree {
    counts: [u16; MAX_BITS + 1],
    symbols: Vec<u16>,
    max_len: u32,
}

impl HuffTree {
    fn from_lengths(lengths: &[u8]) -> Result<Self, ExrError> {
        let mut counts = [0u16; MAX_BITS + 1];
        let mut max_len = 0u32;
        for &len in lengths {
            if usize::from(len) > MAX_BITS {
                return Err(ExrError::CorruptChunk("DEFLATE Huffman code length > 15"));
            }
            if len > 0 {
                counts[usize::from(len)] += 1;
                if u32::from(len) > max_len {
                    max_len = u32::from(len);
                }
            }
        }

        let mut left = 1i32;
        for &cnt in counts.iter().take(MAX_BITS + 1).skip(1) {
            left <<= 1;
            left -= i32::from(cnt);
            if left < 0 {
                return Err(ExrError::CorruptChunk(
                    "oversubscribed DEFLATE Huffman tree",
                ));
            }
        }

        let mut offsets = [0u16; MAX_BITS + 1];
        let mut total = 0u16;
        for bits in 1..=MAX_BITS {
            offsets[bits] = total;
            total += counts[bits];
        }

        let mut symbols = vec![0u16; usize::from(total)];
        for (sym, &len) in lengths.iter().enumerate() {
            if len > 0 {
                let idx = usize::from(offsets[usize::from(len)]);
                offsets[usize::from(len)] += 1;
                symbols[idx] = sym as u16;
            }
        }

        Ok(Self {
            counts,
            symbols,
            max_len,
        })
    }

    fn fixed_lit_len() -> Result<Self, ExrError> {
        let mut lengths = [0u8; 288];
        lengths[..=143].fill(8);
        lengths[144..=255].fill(9);
        lengths[256..=279].fill(7);
        lengths[280..=287].fill(8);
        Self::from_lengths(&lengths)
    }

    fn fixed_dist() -> Result<Self, ExrError> {
        let lengths = [5u8; 32];
        Self::from_lengths(&lengths)
    }

    #[inline]
    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16, ExrError> {
        if self.max_len == 0 {
            return Err(ExrError::CorruptChunk("empty DEFLATE Huffman tree"));
        }
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..=(self.max_len as usize) {
            let bit = reader.read_bits(1)? as i32;
            code |= bit;
            let count = i32::from(self.counts[len]);
            if code - count < first {
                let sym_idx = (index + (code - first)) as usize;
                return self
                    .symbols
                    .get(sym_idx)
                    .copied()
                    .ok_or(ExrError::CorruptChunk("invalid DEFLATE symbol index"));
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err(ExrError::CorruptChunk("unmatched DEFLATE Huffman code"))
    }
}

/// Decompresses an RFC 1950 `zlib`-wrapped stream into `expected_max_len` bytes.
pub fn zlib_decompress(data: &[u8], expected_max_len: usize) -> Result<Vec<u8>, ExrError> {
    if data.len() < 6 {
        return Err(ExrError::CorruptChunk("zlib stream shorter than 6 bytes"));
    }
    let cmf = data[0];
    let flg = data[1];
    let cm = cmf & 0x0f;
    let cinfo = cmf >> 4;
    if cm != 8 || cinfo > 7 {
        return Err(ExrError::CorruptChunk(
            "invalid zlib compression method/window",
        ));
    }
    if ((u16::from(cmf) << 8) | u16::from(flg)) % 31 != 0 {
        return Err(ExrError::CorruptChunk("zlib header FCHECK mismatch"));
    }
    if (flg & 0x20) != 0 {
        return Err(ExrError::CorruptChunk("zlib preset dictionary not allowed"));
    }

    let deflate_payload = &data[2..data.len() - 4];
    let expected_adler = u32::from_be_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);

    let out = deflate_decompress(deflate_payload, expected_max_len)?;
    let actual_adler = adler32(&out);
    if actual_adler != expected_adler {
        return Err(ExrError::CorruptChunk("zlib Adler-32 checksum mismatch"));
    }
    Ok(out)
}

/// Decompresses a raw RFC 1951 `DEFLATE` stream.
pub fn deflate_decompress(data: &[u8], expected_max_len: usize) -> Result<Vec<u8>, ExrError> {
    let mut reader = BitReader::new(data);
    let mut out = Vec::with_capacity(expected_max_len.min(1 << 20));

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;

        match btype {
            0 => {
                reader.align_to_byte();
                let len = reader.read_aligned_u16_le()?;
                let nlen = reader.read_aligned_u16_le()?;
                if len != !nlen {
                    return Err(ExrError::CorruptChunk(
                        "DEFLATE stored block LEN/NLEN mismatch",
                    ));
                }
                if out.len().saturating_add(usize::from(len)) > expected_max_len {
                    return Err(ExrError::CorruptChunk(
                        "DEFLATE stored block exceeds expected output size",
                    ));
                }
                reader.copy_aligned_bytes(usize::from(len), &mut out)?;
            }
            1 => {
                let lit_tree = HuffTree::fixed_lit_len()?;
                let dist_tree = HuffTree::fixed_dist()?;
                decode_huffman_block(
                    &mut reader,
                    &lit_tree,
                    &dist_tree,
                    &mut out,
                    expected_max_len,
                )?;
            }
            2 => {
                let (lit_tree, dist_tree) = read_dynamic_trees(&mut reader)?;
                decode_huffman_block(
                    &mut reader,
                    &lit_tree,
                    &dist_tree,
                    &mut out,
                    expected_max_len,
                )?;
            }
            _ => {
                return Err(ExrError::CorruptChunk("reserved DEFLATE block type 3"));
            }
        }

        if bfinal != 0 {
            break;
        }
    }

    Ok(out)
}

fn read_dynamic_trees(reader: &mut BitReader<'_>) -> Result<(HuffTree, HuffTree), ExrError> {
    let hlit = (reader.read_bits(5)? as usize) + 257;
    let hdist = (reader.read_bits(5)? as usize) + 1;
    let hclen = (reader.read_bits(4)? as usize) + 4;

    if hlit > 286 || hdist > 30 {
        return Err(ExrError::CorruptChunk("invalid DEFLATE HLIT/HDIST count"));
    }

    let mut cl_lengths = [0u8; 19];
    for i in 0..hclen {
        cl_lengths[CL_ORDER[i]] = reader.read_bits(3)? as u8;
    }
    let cl_tree = HuffTree::from_lengths(&cl_lengths)?;

    let total_codes = hlit + hdist;
    let mut lengths = Vec::with_capacity(total_codes);
    while lengths.len() < total_codes {
        let sym = cl_tree.decode(reader)?;
        match sym {
            0..=15 => {
                lengths.push(sym as u8);
            }
            16 => {
                let prev = *lengths
                    .last()
                    .ok_or(ExrError::CorruptChunk("DEFLATE repeat code 16 at start"))?;
                let repeat = (reader.read_bits(2)? as usize) + 3;
                if lengths.len() + repeat > total_codes {
                    return Err(ExrError::CorruptChunk("DEFLATE code length overflow"));
                }
                lengths.resize(lengths.len() + repeat, prev);
            }
            17 => {
                let repeat = (reader.read_bits(3)? as usize) + 3;
                if lengths.len() + repeat > total_codes {
                    return Err(ExrError::CorruptChunk("DEFLATE code length overflow"));
                }
                lengths.resize(lengths.len() + repeat, 0);
            }
            18 => {
                let repeat = (reader.read_bits(7)? as usize) + 11;
                if lengths.len() + repeat > total_codes {
                    return Err(ExrError::CorruptChunk("DEFLATE code length overflow"));
                }
                lengths.resize(lengths.len() + repeat, 0);
            }
            _ => {
                return Err(ExrError::CorruptChunk("invalid DEFLATE code length symbol"));
            }
        }
    }

    let lit_tree = HuffTree::from_lengths(&lengths[..hlit])?;
    let dist_tree = HuffTree::from_lengths(&lengths[hlit..])?;
    Ok((lit_tree, dist_tree))
}

fn decode_huffman_block(
    reader: &mut BitReader<'_>,
    lit_tree: &HuffTree,
    dist_tree: &HuffTree,
    out: &mut Vec<u8>,
    expected_max_len: usize,
) -> Result<(), ExrError> {
    loop {
        let sym = lit_tree.decode(reader)?;
        if sym < 256 {
            if out.len() >= expected_max_len {
                return Err(ExrError::CorruptChunk(
                    "DEFLATE literal exceeds expected output size",
                ));
            }
            out.push(sym as u8);
        } else if sym == 256 {
            break;
        } else {
            let len_idx = (sym as usize) - 257;
            if len_idx >= LENGTH_BASE.len() {
                return Err(ExrError::CorruptChunk("invalid DEFLATE length symbol"));
            }
            let extra_bits = u32::from(LENGTH_EXTRA[len_idx]);
            let length =
                usize::from(LENGTH_BASE[len_idx]) + (reader.read_bits(extra_bits)? as usize);

            let dist_sym = dist_tree.decode(reader)? as usize;
            if dist_sym >= DIST_BASE.len() {
                return Err(ExrError::CorruptChunk("invalid DEFLATE distance symbol"));
            }
            let dist_extra = u32::from(DIST_EXTRA[dist_sym]);
            let distance =
                usize::from(DIST_BASE[dist_sym]) + (reader.read_bits(dist_extra)? as usize);

            if distance == 0 || distance > out.len() {
                return Err(ExrError::CorruptChunk(
                    "DEFLATE backward distance out of bounds",
                ));
            }
            if out.len().saturating_add(length) > expected_max_len {
                return Err(ExrError::CorruptChunk(
                    "DEFLATE match exceeds expected output size",
                ));
            }

            let start = out.len() - distance;
            for i in 0..length {
                let b = out[start + i];
                out.push(b);
            }
        }
    }
    Ok(())
}

struct BitWriter {
    out: Vec<u8>,
    bit_buf: u64,
    bit_count: u32,
}

impl BitWriter {
    fn with_capacity(cap: usize) -> Self {
        Self {
            out: Vec::with_capacity(cap),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    #[inline]
    fn write_bits(&mut self, val: u32, n: u32) {
        if n == 0 {
            return;
        }
        self.bit_buf |= u64::from(val & ((1u32 << n) - 1)) << self.bit_count;
        self.bit_count += n;
        while self.bit_count >= 8 {
            self.out.push((self.bit_buf & 0xff) as u8);
            self.bit_buf >>= 8;
            self.bit_count -= 8;
        }
    }

    #[inline]
    fn write_huff_code(&mut self, code: u16, len: u8) {
        let mut rev = 0u32;
        let mut c = u32::from(code);
        for _ in 0..len {
            rev = (rev << 1) | (c & 1);
            c >>= 1;
        }
        self.write_bits(rev, u32::from(len));
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            self.out.push((self.bit_buf & 0xff) as u8);
        }
        self.out
    }
}

#[derive(Debug, Clone, Copy)]
enum LzToken {
    Literal(u8),
    Match {
        len_sym: u16,
        len_extra_bits: u8,
        len_extra_val: u16,
        dist_sym: u8,
        dist_extra_bits: u8,
        dist_extra_val: u16,
    },
}

fn lz77_tokenize(data: &[u8]) -> Vec<LzToken> {
    const HASH_BITS: usize = 14;
    const HASH_SIZE: usize = 1 << HASH_BITS;
    const MAX_WINDOW: usize = 32768;
    const MAX_CHAIN: usize = 32;

    let mut head = vec![usize::MAX; HASH_SIZE];
    let mut prev = vec![usize::MAX; data.len().max(1)];
    let mut tokens = Vec::with_capacity(data.len());

    let hash3 = |pos: usize| -> usize {
        let v = (u32::from(data[pos]) << 16)
            | (u32::from(data[pos + 1]) << 8)
            | u32::from(data[pos + 2]);
        ((v.wrapping_mul(0x1e35_a7bd)) >> (32 - HASH_BITS)) as usize
    };

    let mut pos = 0usize;
    while pos < data.len() {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;

        if pos + 3 <= data.len() {
            let h = hash3(pos);
            let mut cand = head[h];
            prev[pos] = cand;
            head[h] = pos;

            let max_len = (data.len() - pos).min(258);
            let mut chain_steps = 0;
            while cand != usize::MAX && pos - cand <= MAX_WINDOW && chain_steps < MAX_CHAIN {
                if data[cand + best_len] == data[pos + best_len]
                    && data[cand..cand + 3] == data[pos..pos + 3]
                {
                    let mut l = 3;
                    while l < max_len && data[cand + l] == data[pos + l] {
                        l += 1;
                    }
                    if l > best_len {
                        best_len = l;
                        best_dist = pos - cand;
                        if l == max_len {
                            break;
                        }
                    }
                }
                cand = prev[cand];
                chain_steps += 1;
            }
        }

        if best_len >= 3 {
            let (len_sym, len_extra_bits, len_extra_val) = encode_length(best_len as u16);
            let (dist_sym, dist_extra_bits, dist_extra_val) = encode_distance(best_dist as u16);
            tokens.push(LzToken::Match {
                len_sym: len_sym as u16,
                len_extra_bits,
                len_extra_val,
                dist_sym: dist_sym as u8,
                dist_extra_bits,
                dist_extra_val,
            });
            for step in 1..best_len {
                let p = pos + step;
                if p + 3 <= data.len() {
                    let h = hash3(p);
                    prev[p] = head[h];
                    head[h] = p;
                }
            }
            pos += best_len;
        } else {
            tokens.push(LzToken::Literal(data[pos]));
            pos += 1;
        }
    }
    tokens
}

fn build_limited_code_lengths(freqs: &[u64], max_len: u8) -> Vec<u8> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = freqs.len();
    let active: Vec<usize> = (0..n).filter(|&i| freqs[i] > 0).collect();
    let mut lengths = vec![0u8; n];
    if active.is_empty() {
        return lengths;
    }
    if active.len() == 1 {
        lengths[active[0]] = 1;
        return lengths;
    }

    let mut scale = 0u32;
    loop {
        let mut hlink = (0..n).collect::<Vec<usize>>();
        let mut scode = vec![0u8; n];
        let mut heap = BinaryHeap::new();
        for &i in &active {
            let w = (freqs[i] >> scale).max(1);
            heap.push(Reverse((w, i)));
        }
        while heap.len() > 1 {
            let Reverse((f1, i1)) = heap.pop().unwrap();
            let Reverse((f2, i2)) = heap.pop().unwrap();
            heap.push(Reverse((f1 + f2, i2)));
            let mut j = i2;
            loop {
                scode[j] += 1;
                if hlink[j] == j {
                    hlink[j] = i1;
                    break;
                }
                j = hlink[j];
            }
            let mut k = i1;
            loop {
                scode[k] += 1;
                if hlink[k] == k {
                    break;
                }
                k = hlink[k];
            }
        }
        if scode.iter().all(|&l| l <= max_len) {
            return scode;
        }
        scale += 1;
    }
}

fn canonical_codes_from_lengths(lengths: &[u8]) -> Vec<u16> {
    let mut bl_count = [0u16; MAX_BITS + 1];
    for &l in lengths {
        if l > 0 {
            bl_count[usize::from(l)] += 1;
        }
    }
    let mut next_code = [0u16; MAX_BITS + 1];
    let mut code = 0u16;
    for bits in 1..=MAX_BITS {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    let mut codes = vec![0u16; lengths.len()];
    for (i, &l) in lengths.iter().enumerate() {
        if l > 0 {
            codes[i] = next_code[usize::from(l)];
            next_code[usize::from(l)] += 1;
        }
    }
    codes
}

fn emit_fixed_deflate(tokens: &[LzToken], data_len: usize) -> Vec<u8> {
    let mut writer = BitWriter::with_capacity(data_len + 16);
    writer.write_bits(0x78, 8);
    writer.write_bits(0x9c, 8);
    writer.write_bits(1, 1); // BFINAL=1
    writer.write_bits(1, 2); // BTYPE=01 (Fixed Huffman)

    let mut lit_lens = [0u8; 288];
    lit_lens[..=143].fill(8);
    lit_lens[144..=255].fill(9);
    lit_lens[256..=279].fill(7);
    lit_lens[280..=287].fill(8);
    let lit_codes = canonical_codes_from_lengths(&lit_lens);

    for tok in tokens {
        match *tok {
            LzToken::Literal(b) => {
                let s = usize::from(b);
                writer.write_huff_code(lit_codes[s], lit_lens[s]);
            }
            LzToken::Match {
                len_sym,
                len_extra_bits,
                len_extra_val,
                dist_sym,
                dist_extra_bits,
                dist_extra_val,
            } => {
                let ls = usize::from(len_sym);
                writer.write_huff_code(lit_codes[ls], lit_lens[ls]);
                writer.write_bits(u32::from(len_extra_val), u32::from(len_extra_bits));
                writer.write_huff_code(u16::from(dist_sym), 5);
                writer.write_bits(u32::from(dist_extra_val), u32::from(dist_extra_bits));
            }
        }
    }
    writer.write_huff_code(lit_codes[256], lit_lens[256]);
    writer.finish()
}

fn emit_dynamic_deflate(tokens: &[LzToken], data_len: usize) -> Vec<u8> {
    let mut lit_freqs = [0u64; 286];
    let mut dist_freqs = [0u64; 30];
    lit_freqs[256] = 1;
    for tok in tokens {
        match *tok {
            LzToken::Literal(b) => lit_freqs[usize::from(b)] += 1,
            LzToken::Match {
                len_sym, dist_sym, ..
            } => {
                lit_freqs[usize::from(len_sym)] += 1;
                dist_freqs[usize::from(dist_sym)] += 1;
            }
        }
    }
    if dist_freqs.iter().all(|&f| f == 0) {
        dist_freqs[0] = 1;
    }

    let lit_lens = build_limited_code_lengths(&lit_freqs, 15);
    let dist_lens = build_limited_code_lengths(&dist_freqs, 15);
    let lit_codes = canonical_codes_from_lengths(&lit_lens);
    let dist_codes = canonical_codes_from_lengths(&dist_lens);

    let mut hlit = 286usize;
    while hlit > 257 && lit_lens[hlit - 1] == 0 {
        hlit -= 1;
    }
    let mut hdist = 30usize;
    while hdist > 1 && dist_lens[hdist - 1] == 0 {
        hdist -= 1;
    }

    // Encode combined lit_lens[..hlit] + dist_lens[..hdist] with RLE symbols 0..18
    let mut combined = Vec::with_capacity(hlit + hdist);
    combined.extend_from_slice(&lit_lens[..hlit]);
    combined.extend_from_slice(&dist_lens[..hdist]);

    let mut cl_tokens: Vec<(u8, u8, u8)> = Vec::new();
    let mut i = 0usize;
    while i < combined.len() {
        let v = combined[i];
        let mut run = 1usize;
        while i + run < combined.len() && combined[i + run] == v {
            run += 1;
        }
        i += run;
        if v == 0 {
            while run >= 11 {
                let step = run.min(138);
                cl_tokens.push((18, 7, (step - 11) as u8));
                run -= step;
            }
            if run >= 3 {
                cl_tokens.push((17, 3, (run - 3) as u8));
                run = 0;
            }
            for _ in 0..run {
                cl_tokens.push((0, 0, 0));
            }
        } else {
            cl_tokens.push((v, 0, 0));
            run -= 1;
            while run >= 3 {
                let step = run.min(6);
                cl_tokens.push((16, 2, (step - 3) as u8));
                run -= step;
            }
            for _ in 0..run {
                cl_tokens.push((v, 0, 0));
            }
        }
    }

    let mut cl_freqs = [0u64; 19];
    for &(sym, _, _) in &cl_tokens {
        cl_freqs[usize::from(sym)] += 1;
    }
    let cl_lens = build_limited_code_lengths(&cl_freqs, 7);
    let cl_codes = canonical_codes_from_lengths(&cl_lens);

    let mut hclen = 19usize;
    while hclen > 4 && cl_lens[CL_ORDER[hclen - 1]] == 0 {
        hclen -= 1;
    }

    let mut writer = BitWriter::with_capacity(data_len + 32);
    writer.write_bits(0x78, 8);
    writer.write_bits(0x9c, 8);
    writer.write_bits(1, 1); // BFINAL=1
    writer.write_bits(2, 2); // BTYPE=10 (Dynamic Huffman)
    writer.write_bits((hlit - 257) as u32, 5);
    writer.write_bits((hdist - 1) as u32, 5);
    writer.write_bits((hclen - 4) as u32, 4);

    for idx in 0..hclen {
        writer.write_bits(u32::from(cl_lens[CL_ORDER[idx]]), 3);
    }
    for &(sym, eb, ev) in &cl_tokens {
        let s = usize::from(sym);
        writer.write_huff_code(cl_codes[s], cl_lens[s]);
        if eb > 0 {
            writer.write_bits(u32::from(ev), u32::from(eb));
        }
    }

    for tok in tokens {
        match *tok {
            LzToken::Literal(b) => {
                let s = usize::from(b);
                writer.write_huff_code(lit_codes[s], lit_lens[s]);
            }
            LzToken::Match {
                len_sym,
                len_extra_bits,
                len_extra_val,
                dist_sym,
                dist_extra_bits,
                dist_extra_val,
            } => {
                let ls = usize::from(len_sym);
                let ds = usize::from(dist_sym);
                writer.write_huff_code(lit_codes[ls], lit_lens[ls]);
                writer.write_bits(u32::from(len_extra_val), u32::from(len_extra_bits));
                writer.write_huff_code(dist_codes[ds], dist_lens[ds]);
                writer.write_bits(u32::from(dist_extra_val), u32::from(dist_extra_bits));
            }
        }
    }
    writer.write_huff_code(lit_codes[256], lit_lens[256]);
    writer.finish()
}

/// Compresses `data` into an RFC 1950 `zlib` stream by choosing the smallest of
/// Dynamic Huffman (`BTYPE=10`), Fixed Huffman (`BTYPE=01`), and Stored (`BTYPE=00`) blocks.
pub fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let tokens = lz77_tokenize(data);
    let fixed = emit_fixed_deflate(&tokens, data.len());
    let mut best = fixed;

    if data.len() >= 64 {
        let dynamic = emit_dynamic_deflate(&tokens, data.len());
        if dynamic.len() < best.len() {
            best = dynamic;
        }
    }

    // Compare against Stored block if <= 65535 bytes and Huffman expanded the stream
    if data.len() <= 65535 && best.len() > data.len() + 7 {
        let mut stored = Vec::with_capacity(data.len() + 11);
        stored.push(0x78);
        stored.push(0x9c);
        stored.push(0x01); // BFINAL=1, BTYPE=00, byte-aligned
        let len = data.len() as u16;
        stored.extend_from_slice(&len.to_le_bytes());
        stored.extend_from_slice(&(!len).to_le_bytes());
        stored.extend_from_slice(data);
        best = stored;
    }

    let checksum = adler32(data);
    best.extend_from_slice(&checksum.to_be_bytes());
    best
}

fn encode_length(len: u16) -> (usize, u8, u16) {
    for i in (0..LENGTH_BASE.len()).rev() {
        if len >= LENGTH_BASE[i] {
            return (257 + i, LENGTH_EXTRA[i], len - LENGTH_BASE[i]);
        }
    }
    (257, 0, 0)
}

fn encode_distance(dist: u16) -> (usize, u8, u16) {
    for i in (0..DIST_BASE.len()).rev() {
        if dist >= DIST_BASE[i] {
            return (i, DIST_EXTRA[i], dist - DIST_BASE[i]);
        }
    }
    (0, 0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zlib_roundtrip_patterns() {
        let mut payload = Vec::new();
        for i in 0..4096u32 {
            payload.extend_from_slice(&(i % 17).to_le_bytes());
        }
        let compressed = zlib_compress(&payload);
        assert!(compressed.len() < payload.len());
        let decompressed = zlib_decompress(&compressed, payload.len()).unwrap();
        assert_eq!(decompressed, payload);
    }

    #[test]
    fn test_zlib_empty_roundtrip() {
        let compressed = zlib_compress(&[]);
        let decompressed = zlib_decompress(&compressed, 0).unwrap();
        assert!(decompressed.is_empty());
    }
}
