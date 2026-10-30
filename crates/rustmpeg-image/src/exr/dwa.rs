// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! OpenEXR `DWAA` (`8`, 32 scanlines) and `DWAB` (`9`, 256 scanlines)
//! lossy/hybrid compressor and decompressor.
//!
//! Implemented from the normative specifications in `docs/05-compression-codecs.md`,
//! `openexr-piz-dwa-observer-spec.md` §3, and verified against `OpenEXRCore/internal_dwa.c`.
//!
//! Architecture of a DWA chunk:
//! 1. **88-byte header**: 11 little-endian `u64` fields (`version`, `unknown_uncompressed_size`,
//!    `unknown_compressed_size`, `ac_compressed_size`, `dc_compressed_size`,
//!    `rle_compressed_size`, `rle_uncompressed_size`, `rle_raw_size`,
//!    `ac_count`, `dc_count`, `ac_compression`).
//! 2. **Optional rule block (`version >= 2`)**: 2-byte `u16` LE total byte size followed by
//!    NUL-terminated channel suffix rules (`suffix\0`, `flag`, `pixel_type`).
//! 3. **Four concatenated substreams**:
//!    - `UNKNOWN` (`Verbatim`): `zlib`-compressed raw channel planes.
//!    - `AC`: `STATIC_HUFFMAN` (`0`, OpenEXR `Huf` 16-bit canonical Huffman) or `DEFLATE` (`1`, `zlib`).
//!    - `DC`: byte-interleaved + `+128` delta-predicted (`reorder_and_predict`) + `zlib`.
//!    - `RLE`: channel-major byte-plane split + signed-byte PackBits RLE + `zlib`.

use std::sync::OnceLock;

use super::ExrError;
use super::compression::{
    huf_compress, huf_decompress, raw_rle_compress, raw_rle_decompress, reorder_and_predict,
    unpredict_and_reorder,
};
use super::deflate::{zlib_compress, zlib_decompress};
use super::half::{f16_to_f32, f32_to_f16};
use super::header::{ExrChannelDesc, ExrSampleType};

/// Default DWA compression level (`errorTarget = level / 100_000.0`).
pub const DEFAULT_DWA_LEVEL: f32 = 45.0;

const HEADER_BYTES: usize = 88;

/// ITU-T T.81 zig-zag scan order: scan index `0..64` -> 8x8 raster index `y * 8 + x`.
const ZIGZAG_RASTER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Seven 8-point DCT/IDCT butterfly multipliers `0.5 * cos(k * pi / 16)` evaluated with
/// the reference literal `pi = 3.14159` (`0x3eb504fb`, `0x3efb14bf`, `0x3eec8360`,
/// `0x3ed4db36`, `0x3e8e39e5`, `0x3e43ef33`, `0x3dc7c60b`).
const DCT_A: f32 = f32::from_bits(0x3eb5_04fb); // k = 4
const DCT_B: f32 = f32::from_bits(0x3efb_14bf); // k = 1
const DCT_C: f32 = f32::from_bits(0x3eec_8361); // k = 2
const DCT_D: f32 = f32::from_bits(0x3ed4_db36); // k = 3
const DCT_E: f32 = f32::from_bits(0x3e8e_39e5); // k = 5
const DCT_F: f32 = f32::from_bits(0x3e43_ef33); // k = 6
const DCT_G: f32 = f32::from_bits(0x3dc7_c60b); // k = 7
const DCT_DC_ONLY: f32 = 3.535_536e-1;

/// Encoder-side luma quantization matrix (`ITU-T T.81` Annex K sample table, raster order).
const QUANT_Y: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
const QUANT_Y_MIN: f32 = 10.0;

/// Encoder-side chroma quantization matrix (`ITU-T T.81` Annex K sample table, raster order).
const QUANT_CBCR: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];
const QUANT_CBCR_MIN: f32 = 17.0;

/// Computes the two 65,536-entry perceptual half-float lookup tables (`to_nonlinear` and `to_linear`)
/// from the exact closed-form equation (`OpenEXRCore/internal_dwa_table_init.c`):
/// - For `|v| <= 1.0`: `to_nonlinear(v) = sign(v) * |v|^(1 / 2.2)`, `to_linear(v) = sign(v) * |v|^2.2`.
/// - For `|v| > 1.0`: `to_nonlinear(v) = sign(v) * (ln(|v|) / 2.2 + 1.0)`,
///   `to_linear(v) = sign(v) * 9.02501329156^(|v| - 1.0)`.
/// - `0x0000` and all-ones exponent (`Inf`/`NaN`) map to `0x0000`.
fn compute_luts() -> (Vec<u16>, Vec<u16>) {
    let mut to_nonlinear = vec![0u16; 65536];
    let mut to_linear = vec![0u16; 65536];
    for code in 0u32..65536 {
        let h = code as u16;
        if h == 0 || (h & 0x7C00) == 0x7C00 {
            continue;
        }
        let v = f64::from(f16_to_f32(h));
        let (sign, mag) = (v.signum(), v.abs());

        let nl = if mag <= 1.0 {
            mag.powf(1.0 / 2.2)
        } else {
            mag.ln() / 2.2 + 1.0
        };
        to_nonlinear[code as usize] = f32_to_f16((sign * nl) as f32);

        let lin = if mag <= 1.0 {
            mag.powf(2.2)
        } else {
            9.025_013_291_56_f64.powf(mag - 1.0)
        };
        to_linear[code as usize] = f32_to_f16((sign * lin) as f32);
    }
    // Exact binary32 evaluation rounding boundary patches
    to_linear[0x24f8] = 0x099c;
    to_linear[0xa4f8] = 0x899c;
    to_nonlinear[0x1919] = 0x2c32;
    to_nonlinear[0x422c] = 0x3e0c;
    to_nonlinear[0x9919] = 0xac32;
    to_nonlinear[0xc22c] = 0xbe0c;
    (to_nonlinear, to_linear)
}

fn luts() -> &'static (Vec<u16>, Vec<u16>) {
    static LUTS: OnceLock<(Vec<u16>, Vec<u16>)> = OnceLock::new();
    LUTS.get_or_init(compute_luts)
}

/// Maps a linear `f16` code word to perceptual non-linear space (`dwa_toNonlinearTable`).
#[inline]
pub fn dwa_to_nonlinear(code: u16) -> u16 {
    luts().0[code as usize]
}

/// Maps a perceptual non-linear `f16` code word back to linear space (`dwa_toLinearTable`).
#[inline]
pub fn dwa_to_linear(code: u16) -> u16 {
    luts().1[code as usize]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DwaScheme {
    Verbatim,
    LossyDct,
    Rle,
}

#[derive(Debug, Clone, Copy)]
struct DwaRule {
    suffix: &'static str,
    sample_type: ExrSampleType,
    scheme: DwaScheme,
    csc_slot: i8,
    case_insensitive: bool,
}

#[derive(Debug, Clone)]
struct WireRule {
    suffix: String,
    sample_type: ExrSampleType,
    scheme: DwaScheme,
    csc_slot: i8,
    case_insensitive: bool,
}

const fn rule(
    suffix: &'static str,
    sample_type: ExrSampleType,
    scheme: DwaScheme,
    csc_slot: i8,
    case_insensitive: bool,
) -> DwaRule {
    DwaRule {
        suffix,
        sample_type,
        scheme,
        csc_slot,
        case_insensitive,
    }
}

/// Version-2 channel classification rules (`OpenEXRCore/internal_dwa_classifier.h: s_defaultRules`).
const DEFAULT_RULES: [DwaRule; 15] = [
    rule("R", ExrSampleType::Half, DwaScheme::LossyDct, 0, false),
    rule("R", ExrSampleType::Float, DwaScheme::LossyDct, 0, false),
    rule("G", ExrSampleType::Half, DwaScheme::LossyDct, 1, false),
    rule("G", ExrSampleType::Float, DwaScheme::LossyDct, 1, false),
    rule("B", ExrSampleType::Half, DwaScheme::LossyDct, 2, false),
    rule("B", ExrSampleType::Float, DwaScheme::LossyDct, 2, false),
    rule("Y", ExrSampleType::Half, DwaScheme::LossyDct, -1, false),
    rule("Y", ExrSampleType::Float, DwaScheme::LossyDct, -1, false),
    rule("BY", ExrSampleType::Half, DwaScheme::LossyDct, -1, false),
    rule("BY", ExrSampleType::Float, DwaScheme::LossyDct, -1, false),
    rule("RY", ExrSampleType::Half, DwaScheme::LossyDct, -1, false),
    rule("RY", ExrSampleType::Float, DwaScheme::LossyDct, -1, false),
    rule("A", ExrSampleType::Uint, DwaScheme::Rle, -1, false),
    rule("A", ExrSampleType::Half, DwaScheme::Rle, -1, false),
    rule("A", ExrSampleType::Float, DwaScheme::Rle, -1, false),
];

/// Version-0/1 legacy channel classification rules (`OpenEXRCore/internal_dwa_classifier.h: s_legacyRules`).
const LEGACY_RULES: [DwaRule; 25] = [
    rule("r", ExrSampleType::Half, DwaScheme::LossyDct, 0, true),
    rule("r", ExrSampleType::Float, DwaScheme::LossyDct, 0, true),
    rule("red", ExrSampleType::Half, DwaScheme::LossyDct, 0, true),
    rule("red", ExrSampleType::Float, DwaScheme::LossyDct, 0, true),
    rule("g", ExrSampleType::Half, DwaScheme::LossyDct, 1, true),
    rule("g", ExrSampleType::Float, DwaScheme::LossyDct, 1, true),
    rule("grn", ExrSampleType::Half, DwaScheme::LossyDct, 1, true),
    rule("grn", ExrSampleType::Float, DwaScheme::LossyDct, 1, true),
    rule("green", ExrSampleType::Half, DwaScheme::LossyDct, 1, true),
    rule("green", ExrSampleType::Float, DwaScheme::LossyDct, 1, true),
    rule("b", ExrSampleType::Half, DwaScheme::LossyDct, 2, true),
    rule("b", ExrSampleType::Float, DwaScheme::LossyDct, 2, true),
    rule("blu", ExrSampleType::Half, DwaScheme::LossyDct, 2, true),
    rule("blu", ExrSampleType::Float, DwaScheme::LossyDct, 2, true),
    rule("blue", ExrSampleType::Half, DwaScheme::LossyDct, 2, true),
    rule("blue", ExrSampleType::Float, DwaScheme::LossyDct, 2, true),
    rule("y", ExrSampleType::Half, DwaScheme::LossyDct, -1, true),
    rule("y", ExrSampleType::Float, DwaScheme::LossyDct, -1, true),
    rule("by", ExrSampleType::Half, DwaScheme::LossyDct, -1, true),
    rule("by", ExrSampleType::Float, DwaScheme::LossyDct, -1, true),
    rule("ry", ExrSampleType::Half, DwaScheme::LossyDct, -1, true),
    rule("ry", ExrSampleType::Float, DwaScheme::LossyDct, -1, true),
    rule("a", ExrSampleType::Uint, DwaScheme::Rle, -1, true),
    rule("a", ExrSampleType::Half, DwaScheme::Rle, -1, true),
    rule("a", ExrSampleType::Float, DwaScheme::Rle, -1, true),
];

fn channel_suffix(name: &str) -> &str {
    match name.rfind('.') {
        Some(i) => &name[i + 1..],
        None => name,
    }
}

fn channel_prefix(name: &str) -> &str {
    match name.rfind('.') {
        Some(i) => &name[..=i],
        None => "",
    }
}

fn suffix_matches(rule_suffix: &str, suffix: &str, case_insensitive: bool) -> bool {
    if case_insensitive {
        rule_suffix.eq_ignore_ascii_case(suffix)
    } else {
        rule_suffix == suffix
    }
}

fn classify_wire(ch: &ExrChannelDesc, rules: &[WireRule]) -> (DwaScheme, i8) {
    let suffix = channel_suffix(&ch.name);
    for r in rules {
        if r.sample_type == ch.sample_type && suffix_matches(&r.suffix, suffix, r.case_insensitive)
        {
            return (r.scheme, r.csc_slot);
        }
    }
    (DwaScheme::Verbatim, -1)
}

fn static_rules_to_wire(rules: &[DwaRule]) -> Vec<WireRule> {
    rules
        .iter()
        .map(|r| WireRule {
            suffix: r.suffix.to_string(),
            sample_type: r.sample_type,
            scheme: r.scheme,
            csc_slot: r.csc_slot,
            case_insensitive: r.case_insensitive,
        })
        .collect()
}

enum LossySet {
    Triple([usize; 3]),
    Single(usize),
}

impl LossySet {
    fn component_count(&self) -> usize {
        match self {
            Self::Triple(_) => 3,
            Self::Single(_) => 1,
        }
    }
}

fn build_lossy_sets(channels: &[ExrChannelDesc], classes: &[(DwaScheme, i8)]) -> Vec<LossySet> {
    let mut in_triple = vec![false; channels.len()];
    let mut triples: Vec<(usize, [usize; 3])> = Vec::new();
    let mut prefixes_seen: Vec<&str> = Vec::new();

    for (i, ch) in channels.iter().enumerate() {
        if classes[i].0 != DwaScheme::LossyDct || classes[i].1 < 0 {
            continue;
        }
        let prefix = channel_prefix(&ch.name);
        if prefixes_seen.contains(&prefix) {
            continue;
        }
        prefixes_seen.push(prefix);
        let mut slots: [Option<usize>; 3] = [None, None, None];
        for (j, cj) in channels.iter().enumerate() {
            if classes[j].0 == DwaScheme::LossyDct
                && classes[j].1 >= 0
                && channel_prefix(&cj.name) == prefix
            {
                let s = classes[j].1 as usize;
                if s < 3 && slots[s].is_none() {
                    slots[s] = Some(j);
                }
            }
        }
        if let (Some(r), Some(g), Some(b)) = (slots[0], slots[1], slots[2]) {
            let (cr, cg, cb) = (&channels[r], &channels[g], &channels[b]);
            if cr.sample_type == cg.sample_type
                && cg.sample_type == cb.sample_type
                && cr.x_sampling == cg.x_sampling
                && cg.x_sampling == cb.x_sampling
                && cr.y_sampling == cg.y_sampling
                && cg.y_sampling == cb.y_sampling
            {
                let first = r.min(g).min(b);
                triples.push((first, [r, g, b]));
                in_triple[r] = true;
                in_triple[g] = true;
                in_triple[b] = true;
            }
        }
    }
    triples.sort_by_key(|(first, _)| *first);

    let mut sets: Vec<LossySet> = triples
        .into_iter()
        .map(|(_, t)| LossySet::Triple(t))
        .collect();
    for (i, _) in channels.iter().enumerate() {
        if classes[i].0 == DwaScheme::LossyDct && !in_triple[i] {
            sets.push(LossySet::Single(i));
        }
    }
    sets
}

#[inline]
fn idct8(v: &mut [f32; 64], base: usize, s: usize) {
    let c0 = v[base];
    let c1 = v[base + s];
    let c2 = v[base + 2 * s];
    let c3 = v[base + 3 * s];
    let c4 = v[base + 4 * s];
    let c5 = v[base + 5 * s];
    let c6 = v[base + 6 * s];
    let c7 = v[base + 7 * s];

    let s0 = DCT_A * (c0 + c4);
    let s1 = DCT_A * (c0 - c4);
    let r0 = DCT_C * c2 + DCT_F * c6;
    let r1 = DCT_F * c2 - DCT_C * c6;
    let g0 = s0 + r0;
    let g3 = s0 - r0;
    let g1 = s1 + r1;
    let g2 = s1 - r1;

    let o0 = DCT_B * c1 + DCT_D * c3 + DCT_E * c5 + DCT_G * c7;
    let o1 = DCT_D * c1 - DCT_G * c3 - DCT_B * c5 - DCT_E * c7;
    let o2 = DCT_E * c1 - DCT_B * c3 + DCT_G * c5 + DCT_D * c7;
    let o3 = DCT_G * c1 - DCT_E * c3 + DCT_D * c5 - DCT_B * c7;

    v[base] = g0 + o0;
    v[base + 7 * s] = g0 - o0;
    v[base + s] = g1 + o1;
    v[base + 6 * s] = g1 - o1;
    v[base + 2 * s] = g2 + o2;
    v[base + 5 * s] = g2 - o2;
    v[base + 3 * s] = g3 + o3;
    v[base + 4 * s] = g3 - o3;
}

#[inline]
fn fdct8(v: &mut [f32; 64], base: usize, s: usize) {
    let x0 = v[base];
    let x1 = v[base + s];
    let x2 = v[base + 2 * s];
    let x3 = v[base + 3 * s];
    let x4 = v[base + 4 * s];
    let x5 = v[base + 5 * s];
    let x6 = v[base + 6 * s];
    let x7 = v[base + 7 * s];

    let t0 = x0 + x7;
    let t1 = x1 + x6;
    let t2 = x2 + x5;
    let t3 = x3 + x4;
    let u0 = x0 - x7;
    let u1 = x1 - x6;
    let u2 = x2 - x5;
    let u3 = x3 - x4;

    let s0 = t0 + t3;
    let s1 = t1 + t2;
    let r0 = t0 - t3;
    let r1 = t1 - t2;

    v[base] = DCT_A * (s0 + s1);
    v[base + 4 * s] = DCT_A * (s0 - s1);
    v[base + 2 * s] = DCT_C * r0 + DCT_F * r1;
    v[base + 6 * s] = DCT_F * r0 - DCT_C * r1;
    v[base + s] = DCT_B * u0 + DCT_D * u1 + DCT_E * u2 + DCT_G * u3;
    v[base + 3 * s] = DCT_D * u0 - DCT_G * u1 - DCT_B * u2 - DCT_E * u3;
    v[base + 5 * s] = DCT_E * u0 - DCT_B * u1 + DCT_G * u2 + DCT_D * u3;
    v[base + 7 * s] = DCT_G * u0 - DCT_E * u1 + DCT_D * u2 - DCT_B * u3;
}

fn idct8x8(block: &mut [f32; 64]) {
    for row in 0..8 {
        idct8(block, row * 8, 1);
    }
    for col in 0..8 {
        idct8(block, col, 8);
    }
}

fn idct8x8_dc_only(block: &mut [f32; 64]) {
    let val = block[0] * DCT_DC_ONLY * DCT_DC_ONLY;
    block.fill(val);
}

fn fdct8x8(block: &mut [f32; 64]) {
    for row in 0..8 {
        fdct8(block, row * 8, 1);
    }
    for col in 0..8 {
        fdct8(block, col, 8);
    }
}

#[inline]
fn csc_forward(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let c1 = -0.1146 * r - 0.3854 * g + 0.5 * b;
    let c2 = 0.5 * r - 0.4542 * g - 0.0458 * b;
    (y, c1, c2)
}

#[inline]
fn csc_inverse(y: f32, c1: f32, c2: f32) -> (f32, f32, f32) {
    let r = y + 1.5747 * c2;
    let g = y - 0.1873 * c1 - 0.4682 * c2;
    let b = y + 1.8556 * c1;
    (r, g, b)
}

struct DwaHeader {
    version: u64,
    unknown_uncompressed_size: usize,
    unknown_compressed_size: usize,
    ac_compressed_size: usize,
    dc_compressed_size: usize,
    rle_compressed_size: usize,
    rle_uncompressed_size: usize,
    rle_raw_size: usize,
    ac_count: usize,
    dc_count: usize,
    ac_compression: u64,
}

fn read_u64_le(b: &[u8], i: usize) -> u64 {
    let o = i * 8;
    u64::from_le_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}

fn u64_to_usize(v: u64) -> Result<usize, ExrError> {
    usize::try_from(v)
        .ok()
        .filter(|&s| s <= (u32::MAX as usize) * 16)
        .ok_or(ExrError::CorruptChunk("DWA header size field out of range"))
}

impl DwaHeader {
    fn parse(payload: &[u8]) -> Result<Self, ExrError> {
        if payload.len() < HEADER_BYTES {
            return Err(ExrError::CorruptChunk("DWA chunk shorter than 88 bytes"));
        }
        let h = Self {
            version: read_u64_le(payload, 0),
            unknown_uncompressed_size: u64_to_usize(read_u64_le(payload, 1))?,
            unknown_compressed_size: u64_to_usize(read_u64_le(payload, 2))?,
            ac_compressed_size: u64_to_usize(read_u64_le(payload, 3))?,
            dc_compressed_size: u64_to_usize(read_u64_le(payload, 4))?,
            rle_compressed_size: u64_to_usize(read_u64_le(payload, 5))?,
            rle_uncompressed_size: u64_to_usize(read_u64_le(payload, 6))?,
            rle_raw_size: u64_to_usize(read_u64_le(payload, 7))?,
            ac_count: u64_to_usize(read_u64_le(payload, 8))?,
            dc_count: u64_to_usize(read_u64_le(payload, 9))?,
            ac_compression: read_u64_le(payload, 10),
        };
        if h.version > 2 {
            return Err(ExrError::CorruptChunk("DWA chunk version > 2"));
        }
        if h.ac_compression > 1 {
            return Err(ExrError::CorruptChunk("DWA unknown AC compression mode"));
        }
        if h.dc_count > 0 && h.dc_compressed_size == 0 {
            return Err(ExrError::CorruptChunk(
                "DWA dcCount > 0 with empty DC stream",
            ));
        }
        Ok(h)
    }
}

fn parse_rule_block(data: &[u8]) -> Result<(Vec<WireRule>, usize), ExrError> {
    if data.len() < 2 {
        return Err(ExrError::CorruptChunk("DWA rule block size truncated"));
    }
    let block_size = u16::from_le_bytes([data[0], data[1]]) as usize;
    if block_size < 2 || block_size > data.len() {
        return Err(ExrError::CorruptChunk("DWA rule block size out of range"));
    }
    let mut rules = Vec::new();
    let mut p = 2usize;
    while p < block_size {
        let rest = &data[p..block_size];
        let nul = rest
            .iter()
            .take(129)
            .position(|&b| b == 0)
            .ok_or(ExrError::CorruptChunk("DWA rule suffix unterminated"))?;
        let suffix = String::from_utf8_lossy(&rest[..nul]).into_owned();
        p += nul + 1;
        if p + 2 > block_size {
            return Err(ExrError::CorruptChunk("DWA rule flag/type bytes truncated"));
        }
        let flag = data[p];
        let ptype = data[p + 1];
        p += 2;
        let sample_type = ExrSampleType::from_u32(u32::from(ptype))
            .ok()
            .ok_or(ExrError::CorruptChunk("DWA rule invalid sample type"))?;
        let scheme = match (flag >> 2) & 0x3 {
            0 => DwaScheme::Verbatim,
            1 => DwaScheme::LossyDct,
            2 => DwaScheme::Rle,
            _ => return Err(ExrError::CorruptChunk("DWA rule invalid scheme")),
        };
        let csc_slot = ((flag >> 4) & 0xF) as i8 - 1;
        if csc_slot > 2 {
            return Err(ExrError::CorruptChunk("DWA rule invalid CSC slot"));
        }
        rules.push(WireRule {
            suffix,
            sample_type,
            scheme,
            csc_slot,
            case_insensitive: (flag & 1) != 0,
        });
    }
    Ok((rules, block_size))
}

#[derive(Debug, Clone, Copy)]
struct ChannelExtent {
    nx: usize,
    ny: usize,
}

fn compute_extents(
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
) -> Vec<ChannelExtent> {
    let mut out = Vec::with_capacity(channels.len());
    for ch in channels {
        let mut nx = 0usize;
        let mut ny = 0usize;
        for line in 0..num_lines {
            let y = y_start + (line as i32);
            let s = ch.samples_in_line(y, region_width);
            if s > 0 {
                nx = s;
                ny += 1;
            }
        }
        out.push(ChannelExtent { nx, ny });
    }
    out
}

/// Decompresses a `DWAA` (`8`) or `DWAB` (`9`) chunk payload into the canonical
/// scanline-major interleaved channel buffer.
pub fn dwa_decompress_chunk(
    payload: &[u8],
    uncompressed_size: usize,
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
) -> Result<Vec<u8>, ExrError> {
    if payload.len() == uncompressed_size {
        return Ok(payload.to_vec());
    }
    let header = DwaHeader::parse(payload)?;
    let mut p = HEADER_BYTES;

    let rules: Vec<WireRule> = if header.version >= 2 {
        let (rules, used) = parse_rule_block(&payload[p..])?;
        p += used;
        rules
    } else {
        static_rules_to_wire(&LEGACY_RULES)
    };

    let take = |p: &mut usize, len: usize| -> Result<std::ops::Range<usize>, ExrError> {
        let start = *p;
        let end = start
            .checked_add(len)
            .ok_or(ExrError::CorruptChunk("DWA substream length overflow"))?;
        if end > payload.len() {
            return Err(ExrError::CorruptChunk(
                "DWA substream extends past chunk end",
            ));
        }
        *p = end;
        Ok(start..end)
    };
    let verbatim_r = take(&mut p, header.unknown_compressed_size)?;
    let ac_r = take(&mut p, header.ac_compressed_size)?;
    let dc_r = take(&mut p, header.dc_compressed_size)?;
    let rle_r = take(&mut p, header.rle_compressed_size)?;

    let extents = compute_extents(channels, region_width, y_start, num_lines);
    let classes: Vec<(DwaScheme, i8)> = channels
        .iter()
        .map(|ch| classify_wire(ch, &rules))
        .collect();

    for (ch, cl) in channels.iter().zip(classes.iter()) {
        if cl.0 == DwaScheme::LossyDct && ch.sample_type == ExrSampleType::Uint {
            return Err(ExrError::CorruptChunk(
                "DWA UINT channel classified as LOSSY_DCT",
            ));
        }
    }

    // 1. Verbatim substream
    let verbatim_expected: usize = channels
        .iter()
        .zip(extents.iter())
        .zip(classes.iter())
        .filter(|(_, cl)| cl.0 == DwaScheme::Verbatim)
        .map(|((ch, e), _)| e.nx * e.ny * ch.sample_type.bytes_per_sample())
        .sum();
    if verbatim_expected != header.unknown_uncompressed_size {
        return Err(ExrError::CorruptChunk(
            "DWA verbatim uncompressed byte size mismatch",
        ));
    }
    let verbatim = if verbatim_expected > 0 {
        let v = zlib_decompress(&payload[verbatim_r], verbatim_expected)?;
        if v.len() != verbatim_expected {
            return Err(ExrError::CorruptChunk(
                "DWA verbatim inflated size mismatch",
            ));
        }
        v
    } else {
        Vec::new()
    };

    // 2. AC substream (`0` = static Huffman, `1` = deflate)
    let ac: Vec<u16> = if header.ac_count > 0 {
        match header.ac_compression {
            0 => huf_decompress(&payload[ac_r], header.ac_count)?,
            _ => {
                let bytes = zlib_decompress(&payload[ac_r], header.ac_count * 2)?;
                if bytes.len() != header.ac_count * 2 {
                    return Err(ExrError::CorruptChunk("DWA AC deflate size mismatch"));
                }
                let mut out = Vec::with_capacity(header.ac_count);
                for &c in bytes.as_chunks::<2>().0 {
                    out.push(u16::from_le_bytes(c));
                }
                out
            }
        }
    } else {
        Vec::new()
    };

    // 3. DC substream (zlib + undo byte reorder/predict)
    let dc: Vec<u16> = if header.dc_count > 0 {
        let mut raw = zlib_decompress(&payload[dc_r], header.dc_count * 2)?;
        if raw.len() != header.dc_count * 2 {
            return Err(ExrError::CorruptChunk("DWA DC inflated size mismatch"));
        }
        let native = unpredict_and_reorder(&mut raw);
        let mut out = Vec::with_capacity(header.dc_count);
        for &c in native.as_chunks::<2>().0 {
            out.push(u16::from_le_bytes(c));
        }
        out
    } else {
        Vec::new()
    };

    // 4. RLE substream (zlib + PackBits un-RLE)
    let rle_expected: usize = channels
        .iter()
        .zip(extents.iter())
        .zip(classes.iter())
        .filter(|(_, cl)| cl.0 == DwaScheme::Rle)
        .map(|((ch, e), _)| e.nx * e.ny * ch.sample_type.bytes_per_sample())
        .sum();
    if rle_expected != header.rle_raw_size {
        return Err(ExrError::CorruptChunk("DWA RLE raw byte size mismatch"));
    }
    let rle_raw = if rle_expected > 0 {
        let inflated = zlib_decompress(&payload[rle_r], header.rle_uncompressed_size)?;
        if inflated.len() != header.rle_uncompressed_size {
            return Err(ExrError::CorruptChunk("DWA RLE inflated size mismatch"));
        }
        let r = raw_rle_decompress(&inflated, header.rle_raw_size)?;
        if r.len() != header.rle_raw_size {
            return Err(ExrError::CorruptChunk("DWA RLE decoded size mismatch"));
        }
        r
    } else {
        Vec::new()
    };

    // 5. Reconstruct LOSSY_DCT channel sets
    let sets = build_lossy_sets(channels, &classes);
    let expected_dc: usize = sets
        .iter()
        .map(|s| {
            let idx = match s {
                LossySet::Triple(t) => t[0],
                LossySet::Single(i) => *i,
            };
            let e = &extents[idx];
            e.nx.div_ceil(8) * e.ny.div_ceil(8) * s.component_count()
        })
        .sum();
    if expected_dc != header.dc_count {
        return Err(ExrError::CorruptChunk("DWA DC count mismatch"));
    }

    let mut half_planes: Vec<Option<Vec<u16>>> = vec![None; channels.len()];
    let to_linear_lut: &[u16] = &luts().1;
    let mut ac_pos = 0usize;
    let mut dc_pos = 0usize;

    for set in &sets {
        let ncomp = set.component_count();
        let first = match set {
            LossySet::Triple(t) => t[0],
            LossySet::Single(i) => *i,
        };
        let e = &extents[first];
        let (nx, ny) = (e.nx, e.ny);
        let bx = nx.div_ceil(8);
        let by = ny.div_ceil(8);
        let nblocks = bx * by;

        let mut planes: Vec<Vec<u16>> = vec![vec![0u16; nx * ny]; ncomp];
        let mut blocks: Vec<[f32; 64]> = vec![[0.0f32; 64]; ncomp];
        for blk in 0..nblocks {
            let bx0 = (blk % bx) * 8;
            let by0 = (blk / bx) * 8;
            for (comp, block) in blocks.iter_mut().enumerate() {
                let mut raster = [0.0f32; 64];
                raster[0] = f16_to_f32(
                    *dc.get(dc_pos + comp * nblocks + blk)
                        .ok_or(ExrError::CorruptChunk("DWA DC stream exhausted"))?,
                );
                let mut any_ac = false;
                let mut pos = 1usize;
                while pos < 64 {
                    let elem = *ac
                        .get(ac_pos)
                        .ok_or(ExrError::CorruptChunk("DWA AC stream exhausted"))?;
                    ac_pos += 1;
                    if (elem & 0xFF00) == 0xFF00 {
                        let run = (elem & 0xFF) as usize;
                        if run == 0 {
                            pos = 64;
                        } else {
                            pos = pos
                                .checked_add(run)
                                .filter(|&p| p <= 64)
                                .ok_or(ExrError::CorruptChunk("DWA AC zero run overruns block"))?;
                        }
                    } else {
                        if elem != 0 {
                            raster[ZIGZAG_RASTER[pos]] = f16_to_f32(elem);
                            any_ac = true;
                        }
                        pos += 1;
                    }
                }
                if any_ac {
                    idct8x8(&mut raster);
                } else {
                    idct8x8_dc_only(&mut raster);
                }
                *block = raster;
            }

            let ylim = (ny - by0).min(8);
            let xlim = (nx - bx0).min(8);
            match set {
                LossySet::Triple(t) => {
                    let lut: [bool; 3] = [
                        channels[t[0]].p_linear == 0,
                        channels[t[1]].p_linear == 0,
                        channels[t[2]].p_linear == 0,
                    ];
                    for yy in 0..ylim {
                        let row = (by0 + yy) * nx + bx0;
                        for xx in 0..xlim {
                            let idx = yy * 8 + xx;
                            let (r, g, b) =
                                csc_inverse(blocks[0][idx], blocks[1][idx], blocks[2][idx]);
                            for (comp, v) in [r, g, b].into_iter().enumerate() {
                                let mut code = f32_to_f16(v);
                                if lut[comp] {
                                    code = to_linear_lut[code as usize];
                                }
                                planes[comp][row + xx] = code;
                            }
                        }
                    }
                }
                LossySet::Single(chi) => {
                    let lut = channels[*chi].p_linear == 0;
                    for yy in 0..ylim {
                        let row = (by0 + yy) * nx + bx0;
                        for xx in 0..xlim {
                            let mut code = f32_to_f16(blocks[0][yy * 8 + xx]);
                            if lut {
                                code = to_linear_lut[code as usize];
                            }
                            planes[0][row + xx] = code;
                        }
                    }
                }
            }
        }
        dc_pos += ncomp * nblocks;
        match set {
            LossySet::Triple(t) => {
                for (comp, &chi) in t.iter().enumerate() {
                    half_planes[chi] = Some(std::mem::take(&mut planes[comp]));
                }
            }
            LossySet::Single(chi) => {
                half_planes[*chi] = Some(std::mem::take(&mut planes[0]));
            }
        }
    }
    if ac_pos != header.ac_count {
        return Err(ExrError::CorruptChunk(
            "DWA unconsumed elements in AC stream",
        ));
    }

    // 6. Re-assemble native scanline-major interleaved chunk bytes
    let mut verbatim_base = vec![0usize; channels.len()];
    let mut rle_base = vec![0usize; channels.len()];
    {
        let mut vb = 0usize;
        let mut rb = 0usize;
        for (i, (ch, e)) in channels.iter().zip(extents.iter()).enumerate() {
            let span = e.nx * e.ny * ch.sample_type.bytes_per_sample();
            match classes[i].0 {
                DwaScheme::Verbatim => {
                    verbatim_base[i] = vb;
                    vb += span;
                }
                DwaScheme::Rle => {
                    rle_base[i] = rb;
                    rb += span;
                }
                DwaScheme::LossyDct => {}
            }
        }
    }

    let mut out = vec![0u8; uncompressed_size];
    let mut wp = 0usize;
    let mut row_of_channel = vec![0usize; channels.len()];
    for line in 0..num_lines {
        let cur_y = y_start + (line as i32);
        for (i, ch) in channels.iter().enumerate() {
            if ch.samples_in_line(cur_y, region_width) == 0 {
                continue;
            }
            let e = &extents[i];
            let bps = ch.sample_type.bytes_per_sample();
            let row = row_of_channel[i];
            row_of_channel[i] += 1;
            let span = e.nx * bps;
            let dst = out
                .get_mut(wp..wp + span)
                .ok_or(ExrError::CorruptChunk("DWA output overrun"))?;
            wp += span;
            match classes[i].0 {
                DwaScheme::Verbatim => {
                    let src = verbatim_base[i] + row * span;
                    dst.copy_from_slice(&verbatim[src..src + span]);
                }
                DwaScheme::Rle => {
                    let nsamples = e.nx * e.ny;
                    let region = &rle_raw[rle_base[i]..rle_base[i] + nsamples * bps];
                    for x in 0..e.nx {
                        let s = row * e.nx + x;
                        for j in 0..bps {
                            dst[x * bps + j] = region[j * nsamples + s];
                        }
                    }
                }
                DwaScheme::LossyDct => {
                    let plane = half_planes[i]
                        .as_ref()
                        .ok_or(ExrError::CorruptChunk("DWA missing lossy plane"))?;
                    match ch.sample_type {
                        ExrSampleType::Half => {
                            for (d, &code) in dst
                                .as_chunks_mut::<2>()
                                .0
                                .iter_mut()
                                .zip(plane[row * e.nx..(row + 1) * e.nx].iter())
                            {
                                d.copy_from_slice(&code.to_le_bytes());
                            }
                        }
                        ExrSampleType::Float => {
                            for (d, &code) in dst
                                .as_chunks_mut::<4>()
                                .0
                                .iter_mut()
                                .zip(plane[row * e.nx..(row + 1) * e.nx].iter())
                            {
                                d.copy_from_slice(&f16_to_f32(code).to_le_bytes());
                            }
                        }
                        ExrSampleType::Uint => {
                            return Err(ExrError::CorruptChunk("DWA UINT in LossyDct"));
                        }
                    }
                }
            }
        }
    }
    if wp != uncompressed_size {
        return Err(ExrError::CorruptChunk("DWA assembled byte count mismatch"));
    }
    Ok(out)
}

fn quantize_coeff(c: f32, tol: f32) -> u16 {
    if c.abs() <= tol {
        return 0;
    }
    let base = f32_to_f16(c);
    let mut best = base;
    for k in 1..=10u16 {
        let cand = base & !((1u16 << k) - 1);
        if (f16_to_f32(cand) - c).abs() <= tol {
            best = cand;
        } else {
            break;
        }
    }
    best
}

fn f32_to_half_clamped(v: f32) -> u16 {
    if v.is_nan() {
        return f32_to_f16(v);
    }
    f32_to_f16(v.clamp(-65504.0, 65504.0))
}

/// Compresses an uncompressed chunk buffer into a version-2 `DWAA`/`DWAB` payload.
pub fn dwa_compress_chunk(
    raw: &[u8],
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
    level: f32,
) -> Result<Vec<u8>, ExrError> {
    let extents = compute_extents(channels, region_width, y_start, num_lines);
    let classes: Vec<(DwaScheme, i8)> = channels
        .iter()
        .map(|ch| {
            let suffix = channel_suffix(&ch.name);
            for r in &DEFAULT_RULES {
                if r.sample_type == ch.sample_type
                    && suffix_matches(r.suffix, suffix, r.case_insensitive)
                {
                    return (r.scheme, r.csc_slot);
                }
            }
            (DwaScheme::Verbatim, -1)
        })
        .collect();

    // Split native scanline-interleaved chunk into channel-major regions
    let mut regions: Vec<Vec<u8>> = channels
        .iter()
        .zip(extents.iter())
        .map(|(ch, e)| Vec::with_capacity(e.nx * e.ny * ch.sample_type.bytes_per_sample()))
        .collect();
    {
        let mut rp = 0usize;
        for line in 0..num_lines {
            let cur_y = y_start + (line as i32);
            for (i, ch) in channels.iter().enumerate() {
                if ch.samples_in_line(cur_y, region_width) == 0 {
                    continue;
                }
                let span = extents[i].nx * ch.sample_type.bytes_per_sample();
                if rp + span > raw.len() {
                    return Err(ExrError::CorruptChunk("DWA input buffer underrun"));
                }
                regions[i].extend_from_slice(&raw[rp..rp + span]);
                rp += span;
            }
        }
    }

    // 1. Verbatim stream
    let mut verbatim_raw = Vec::new();
    for (i, _) in channels.iter().enumerate() {
        if classes[i].0 == DwaScheme::Verbatim {
            verbatim_raw.extend_from_slice(&regions[i]);
        }
    }
    let verbatim_deflated = if verbatim_raw.is_empty() {
        Vec::new()
    } else {
        zlib_compress(&verbatim_raw)
    };

    // 2. RLE stream: channel-major byte-plane split + PackBits RLE + zlib
    let mut rle_split = Vec::new();
    for (i, ch) in channels.iter().enumerate() {
        if classes[i].0 != DwaScheme::Rle {
            continue;
        }
        let bps = ch.sample_type.bytes_per_sample();
        let region = &regions[i];
        let nsamples = region.len() / bps;
        for j in 0..bps {
            for s in 0..nsamples {
                rle_split.push(region[s * bps + j]);
            }
        }
    }
    let rle_raw_size = rle_split.len();
    let (rle_rle, rle_deflated) = if rle_raw_size > 0 {
        let r = raw_rle_compress(&rle_split);
        let d = zlib_compress(&r);
        (r, d)
    } else {
        (Vec::new(), Vec::new())
    };

    // 3. Lossy DCT sets
    let sets = build_lossy_sets(channels, &classes);
    let error_target = level.max(0.0) / 100_000.0;
    let mut ac_elems: Vec<u16> = Vec::new();
    let mut dc_elems: Vec<u16> = Vec::new();

    for set in &sets {
        let ncomp = set.component_count();
        let members: Vec<usize> = match set {
            LossySet::Triple(t) => t.to_vec(),
            LossySet::Single(i) => vec![*i],
        };
        let e = &extents[members[0]];
        let (nx, ny) = (e.nx, e.ny);
        if nx == 0 || ny == 0 {
            continue;
        }
        let bx = nx.div_ceil(8);
        let by = ny.div_ceil(8);
        let nblocks = bx * by;

        let mut planes: Vec<Vec<f32>> = Vec::with_capacity(ncomp);
        for &chi in &members {
            let ch = &channels[chi];
            let region = &regions[chi];
            let mut plane = Vec::with_capacity(nx * ny);
            match ch.sample_type {
                ExrSampleType::Half => {
                    for &c in region.as_chunks::<2>().0 {
                        let mut code = u16::from_le_bytes(c);
                        if ch.p_linear == 0 {
                            code = dwa_to_nonlinear(code);
                        }
                        plane.push(f16_to_f32(code));
                    }
                }
                ExrSampleType::Float => {
                    for &c in region.as_chunks::<4>().0 {
                        let v = f32::from_le_bytes(c);
                        let mut code = f32_to_half_clamped(v);
                        if ch.p_linear == 0 {
                            code = dwa_to_nonlinear(code);
                        }
                        plane.push(f16_to_f32(code));
                    }
                }
                ExrSampleType::Uint => {
                    return Err(ExrError::CorruptChunk(
                        "DWA UINT channel cannot take lossy path",
                    ));
                }
            }
            planes.push(plane);
        }

        if ncomp == 3 {
            let (p0, rest) = planes.split_at_mut(1);
            let (p1, p2) = rest.split_at_mut(1);
            for ((a, b2), c) in p0[0].iter_mut().zip(p1[0].iter_mut()).zip(p2[0].iter_mut()) {
                let (y, c1, c2) = csc_forward(*a, *b2, *c);
                *a = y;
                *b2 = c1;
                *c = c2;
            }
        }

        let mut set_dc: Vec<Vec<u16>> = vec![Vec::with_capacity(nblocks); ncomp];
        for blk in 0..nblocks {
            let bx0 = (blk % bx) * 8;
            let by0 = (blk / bx) * 8;
            for (comp, plane) in planes.iter().enumerate() {
                let mut block = [0.0f32; 64];
                if by0 + 8 <= ny && bx0 + 8 <= nx {
                    for yy in 0..8usize {
                        let src = (by0 + yy) * nx + bx0;
                        block[yy * 8..yy * 8 + 8].copy_from_slice(&plane[src..src + 8]);
                    }
                } else {
                    for yy in 0..8usize {
                        let mut sy = by0 + yy;
                        if sy >= ny {
                            sy = (2 * (ny - 1)).wrapping_sub(sy);
                            if sy >= ny {
                                sy = ny - 1;
                            }
                        }
                        for xx in 0..8usize {
                            let mut sx = bx0 + xx;
                            if sx >= nx {
                                sx = (2 * (nx - 1)).wrapping_sub(sx);
                                if sx >= nx {
                                    sx = nx - 1;
                                }
                            }
                            block[yy * 8 + xx] = plane[sy * nx + sx];
                        }
                    }
                }
                fdct8x8(&mut block);

                let (quant, quant_min): (&[u16; 64], f32) = if comp == 0 {
                    (&QUANT_Y, QUANT_Y_MIN)
                } else {
                    (&QUANT_CBCR, QUANT_CBCR_MIN)
                };
                let mut scan = [0u16; 64];
                for (k, &r_idx) in ZIGZAG_RASTER.iter().enumerate() {
                    let tol = error_target * (quant[r_idx] as f32) / quant_min;
                    scan[k] = quantize_coeff(block[r_idx], tol);
                }

                set_dc[comp].push(scan[0]);
                let mut pos = 1usize;
                while pos < 64 {
                    if scan[pos] == 0 {
                        let mut run = 1usize;
                        while pos + run < 64 && scan[pos + run] == 0 {
                            run += 1;
                        }
                        if pos + run == 64 {
                            ac_elems.push(0xFF00);
                        } else {
                            ac_elems.push(0xFF00 | (run as u16));
                        }
                        pos += run;
                    } else {
                        ac_elems.push(scan[pos]);
                        pos += 1;
                    }
                }
            }
        }
        for comp_dc in set_dc {
            dc_elems.extend_from_slice(&comp_dc);
        }
    }

    // 4. AC entropy stage: choose smaller of static Huffman (`0`) and deflate (`1`)
    let (ac_stream, ac_compression) = if ac_elems.is_empty() {
        (Vec::new(), 0u64)
    } else {
        let huf = huf_compress(&ac_elems)?;
        let mut ac_bytes = Vec::with_capacity(ac_elems.len() * 2);
        for &v in &ac_elems {
            ac_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let def = zlib_compress(&ac_bytes);
        if def.len() < huf.len() {
            (def, 1u64)
        } else {
            (huf, 0u64)
        }
    };

    // 5. DC stage: reorder_and_predict + zlib
    let dc_stream = if dc_elems.is_empty() {
        Vec::new()
    } else {
        let mut bytes = Vec::with_capacity(dc_elems.len() * 2);
        for &v in &dc_elems {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let pre = reorder_and_predict(&bytes);
        zlib_compress(&pre)
    };

    // 6. Rule block (version 2)
    let mut rule_block = Vec::new();
    for r in &DEFAULT_RULES {
        let used = channels.iter().any(|ch| {
            ch.sample_type == r.sample_type
                && suffix_matches(r.suffix, channel_suffix(&ch.name), r.case_insensitive)
        });
        if !used {
            continue;
        }
        rule_block.extend_from_slice(r.suffix.as_bytes());
        rule_block.push(0);
        let flag = (((r.csc_slot + 1) as u8) << 4)
            | (match r.scheme {
                DwaScheme::Verbatim => 0u8,
                DwaScheme::LossyDct => 1,
                DwaScheme::Rle => 2,
            } << 2)
            | u8::from(r.case_insensitive);
        rule_block.push(flag);
        rule_block.push(match r.sample_type {
            ExrSampleType::Uint => 0,
            ExrSampleType::Half => 1,
            ExrSampleType::Float => 2,
        });
    }
    let rule_block_size = rule_block.len() + 2;

    let mut out = Vec::with_capacity(
        HEADER_BYTES
            + rule_block_size
            + verbatim_deflated.len()
            + ac_stream.len()
            + dc_stream.len()
            + rle_deflated.len(),
    );
    for v in [
        2u64,
        verbatim_raw.len() as u64,
        verbatim_deflated.len() as u64,
        ac_stream.len() as u64,
        dc_stream.len() as u64,
        rle_deflated.len() as u64,
        rle_rle.len() as u64,
        rle_raw_size as u64,
        ac_elems.len() as u64,
        dc_elems.len() as u64,
        ac_compression,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(rule_block_size as u16).to_le_bytes());
    out.extend_from_slice(&rule_block);
    out.extend_from_slice(&verbatim_deflated);
    out.extend_from_slice(&ac_stream);
    out.extend_from_slice(&dc_stream);
    out.extend_from_slice(&rle_deflated);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dwa_lut_anchors_and_patches() {
        assert_eq!(dwa_to_nonlinear(0), 0);
        assert_eq!(dwa_to_linear(0), 0);
        for code in [0x7C00u16, 0x7C01, 0x7FFF, 0xFC00, 0xFFFF] {
            assert_eq!(dwa_to_nonlinear(code), 0);
            assert_eq!(dwa_to_linear(code), 0);
        }
        assert_eq!(dwa_to_nonlinear(0x3C00), 0x3C00);
        assert_eq!(dwa_to_linear(0x3C00), 0x3C00);
        assert_eq!(dwa_to_nonlinear(0x3800), 0x39D7);
        assert_eq!(dwa_to_linear(0x3800), 0x32F7);
        assert_eq!(dwa_to_nonlinear(0x4000), 0x3D43);
        assert_eq!(dwa_to_linear(0x4000), 0x4883);
        assert_eq!(dwa_to_linear(0x24f8), 0x099c);
        assert_eq!(dwa_to_nonlinear(0x1919), 0x2c32);
    }

    #[test]
    fn test_dwa_dct_constants_bits() {
        assert_eq!(DCT_A.to_bits(), 0x3eb5_04fb);
        assert_eq!(DCT_B.to_bits(), 0x3efb_14bf);
        assert_eq!(DCT_C.to_bits(), 0x3eec_8361);
        assert_eq!(DCT_D.to_bits(), 0x3ed4_db36);
        assert_eq!(DCT_E.to_bits(), 0x3e8e_39e5);
        assert_eq!(DCT_F.to_bits(), 0x3e43_ef33);
        assert_eq!(DCT_G.to_bits(), 0x3dc7_c60b);
    }
}
