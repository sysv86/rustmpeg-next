// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! OpenEXR chunk compression and decompression codecs:
//! `None` (`0`), `Rle` (`1`), `Zips` (`2`), `Zip` (`3`), `Piz` (`4`),
//! `Pxr24` (`5`), `B44` (`6`), and `B44a` (`7`).
//!
//! All mathematical transforms, bit packings, and edge-case rules are implemented
//! directly from the normative specifications in `docs/05-compression-codecs.md`
//! and verified against `OpenEXRCore` (`internal_rle.c`, `internal_zip.c`,
//! `internal_piz.c`, `internal_huf.c`, `internal_pxr24.c`, `internal_b44.c`).

use super::ExrError;
use super::deflate::{zlib_compress, zlib_decompress};
use super::dwa::{DEFAULT_DWA_LEVEL, dwa_compress_chunk, dwa_decompress_chunk};
use super::half::{
    b44_convert_from_linear, b44_convert_to_linear, float_to_float24, float24_to_float,
};
use super::header::{ExrChannelDesc, ExrCompression, ExrSampleType};
use super::zstd::{ZstdGridScenario, zstd_compress_chunk, zstd_decompress_chunk};

/// Decompresses a chunk payload according to `compression`, handling the universal
/// OpenEXR rule (`docs/01-file-layout.md` Section 7.6 & `OpenEXRCore/compression.c`):
/// if `packed.len() == expected_uncompressed_size`, the chunk was stored raw because
/// the compressor did not shrink the data.
pub fn decompress_chunk_data(
    compression: ExrCompression,
    packed: &[u8],
    expected_uncompressed_size: usize,
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
) -> Result<Vec<u8>, ExrError> {
    if expected_uncompressed_size == 0 {
        return Ok(Vec::new());
    }
    // Universal OpenEXR raw-bypass rule: if packed_size == uncompressed_size, data is uncompressed!
    if packed.len() == expected_uncompressed_size {
        return Ok(packed.to_vec());
    }

    match compression {
        ExrCompression::None => {
            if packed.len() != expected_uncompressed_size {
                return Err(ExrError::CorruptChunk(
                    "NO_COMPRESSION chunk size does not match expected uncompressed size",
                ));
            }
            Ok(packed.to_vec())
        }
        ExrCompression::Rle => rle_decompress_chunk(packed, expected_uncompressed_size),
        ExrCompression::Zips | ExrCompression::Zip => {
            zip_decompress_chunk(packed, expected_uncompressed_size)
        }
        ExrCompression::Piz => piz_decompress_chunk(
            packed,
            expected_uncompressed_size,
            channels,
            region_width,
            y_start,
            num_lines,
        ),
        ExrCompression::Pxr24 => pxr24_decompress_chunk(
            packed,
            expected_uncompressed_size,
            channels,
            region_width,
            y_start,
            num_lines,
        ),
        ExrCompression::B44 | ExrCompression::B44a => b44_decompress_chunk(
            packed,
            expected_uncompressed_size,
            channels,
            region_width,
            y_start,
            num_lines,
        ),
        ExrCompression::Dwaa | ExrCompression::Dwab => dwa_decompress_chunk(
            packed,
            expected_uncompressed_size,
            channels,
            region_width,
            y_start,
            num_lines,
        ),
        ExrCompression::Zstd => {
            let scenario = if channels.is_empty() {
                ZstdGridScenario::DeepSampleCountTable
            } else {
                ZstdGridScenario::FlatScanOrTile
            };
            zstd_decompress_chunk(
                packed,
                expected_uncompressed_size,
                channels,
                region_width,
                y_start,
                num_lines,
                scenario,
            )
        }
        ExrCompression::Htj2k256 | ExrCompression::Lj2k | ExrCompression::Htj2k32 => {
            Err(ExrError::UnsupportedCompression(compression))
        }
    }
}

/// Compresses an uncompressed chunk buffer using `compression`.
/// Enforces the universal OpenEXR rule: if `compressed.len() >= uncompressed.len()`,
/// returns `uncompressed` verbatim.
pub fn compress_chunk_data(
    compression: ExrCompression,
    uncompressed: &[u8],
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
) -> Result<Vec<u8>, ExrError> {
    if uncompressed.is_empty() || compression == ExrCompression::None {
        return Ok(uncompressed.to_vec());
    }

    let compressed = match compression {
        ExrCompression::None => uncompressed.to_vec(),
        ExrCompression::Rle => rle_compress_chunk(uncompressed),
        ExrCompression::Zips | ExrCompression::Zip => zip_compress_chunk(uncompressed),
        ExrCompression::Piz => {
            piz_compress_chunk(uncompressed, channels, region_width, y_start, num_lines)?
        }
        ExrCompression::Pxr24 => {
            pxr24_compress_chunk(uncompressed, channels, region_width, y_start, num_lines)?
        }
        ExrCompression::B44 => b44_compress_chunk(
            uncompressed,
            channels,
            region_width,
            y_start,
            num_lines,
            false,
        )?,
        ExrCompression::B44a => b44_compress_chunk(
            uncompressed,
            channels,
            region_width,
            y_start,
            num_lines,
            true,
        )?,
        ExrCompression::Dwaa | ExrCompression::Dwab => dwa_compress_chunk(
            uncompressed,
            channels,
            region_width,
            y_start,
            num_lines,
            DEFAULT_DWA_LEVEL,
        )?,
        ExrCompression::Zstd => {
            let scenario = if channels.is_empty() {
                ZstdGridScenario::DeepSampleCountTable
            } else {
                ZstdGridScenario::FlatScanOrTile
            };
            zstd_compress_chunk(
                uncompressed,
                channels,
                region_width,
                y_start,
                num_lines,
                scenario,
            )?
        }
        other => return Err(ExrError::UnsupportedCompression(other)),
    };

    if compressed.len() >= uncompressed.len() {
        Ok(uncompressed.to_vec())
    } else {
        Ok(compressed)
    }
}

/// Decompresses a deep pixel data chunk payload (`ZSTD` requires per-scanline deep sample counts).
#[allow(clippy::too_many_arguments)]
pub fn decompress_deep_samples_chunk_data(
    compression: ExrCompression,
    packed: &[u8],
    expected_uncompressed_size: usize,
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
    row_sample_counts: &[usize],
) -> Result<Vec<u8>, ExrError> {
    if expected_uncompressed_size == 0 {
        return Ok(Vec::new());
    }
    if compression == ExrCompression::Zstd {
        return zstd_decompress_chunk(
            packed,
            expected_uncompressed_size,
            channels,
            region_width,
            y_start,
            num_lines,
            ZstdGridScenario::DeepPixels(row_sample_counts),
        );
    }
    decompress_chunk_data(
        compression,
        packed,
        expected_uncompressed_size,
        channels,
        region_width,
        y_start,
        num_lines,
    )
}

/// Compresses a deep pixel data chunk payload (`ZSTD` requires per-scanline deep sample counts).
pub fn compress_deep_samples_chunk_data(
    compression: ExrCompression,
    uncompressed: &[u8],
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
    row_sample_counts: &[usize],
) -> Result<Vec<u8>, ExrError> {
    if uncompressed.is_empty() || compression == ExrCompression::None {
        return Ok(uncompressed.to_vec());
    }
    if compression == ExrCompression::Zstd {
        return zstd_compress_chunk(
            uncompressed,
            channels,
            region_width,
            y_start,
            num_lines,
            ZstdGridScenario::DeepPixels(row_sample_counts),
        );
    }
    compress_chunk_data(
        compression,
        uncompressed,
        channels,
        region_width,
        y_start,
        num_lines,
    )
}

/// Compresses a flat chunk with an explicit `dwa_level` when `compression` is `Dwaa` or `Dwab`.
pub fn compress_chunk_data_with_dwa_level(
    compression: ExrCompression,
    uncompressed: &[u8],
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
    dwa_level: f32,
) -> Result<Vec<u8>, ExrError> {
    if uncompressed.is_empty() || compression == ExrCompression::None {
        return Ok(uncompressed.to_vec());
    }
    if matches!(compression, ExrCompression::Dwaa | ExrCompression::Dwab) {
        let compressed = dwa_compress_chunk(
            uncompressed,
            channels,
            region_width,
            y_start,
            num_lines,
            dwa_level,
        )?;
        if compressed.len() >= uncompressed.len() {
            return Ok(uncompressed.to_vec());
        }
        return Ok(compressed);
    }
    compress_chunk_data(
        compression,
        uncompressed,
        channels,
        region_width,
        y_start,
        num_lines,
    )
}

// ============================================================================
// 1. Byte Interleave + Delta Predictor (`internal_rle.c` & `internal_zip.c`)
// ============================================================================

/// Splits `src` into even and odd byte halves and applies `+128` delta prediction
/// (`OpenEXRCore/internal_zip.c: internal_zip_deconstruct_bytes`).
pub fn reorder_and_predict(src: &[u8]) -> Vec<u8> {
    let n = src.len();
    if n == 0 {
        return Vec::new();
    }
    let mut scratch = vec![0u8; n];
    let half = n.div_ceil(2);
    for (i, &b) in src.iter().enumerate() {
        if (i & 1) == 0 {
            scratch[i / 2] = b;
        } else {
            scratch[half + (i / 2)] = b;
        }
    }

    let mut p = scratch[0];
    for item in scratch.iter_mut().skip(1) {
        let cur = *item;
        let d = (i32::from(cur) - i32::from(p) + (128 + 256)) as u8;
        p = cur;
        *item = d;
    }
    scratch
}

/// Reverses the `+128` delta predictor and interleaves the two halves back
/// (`OpenEXRCore/internal_zip.c: internal_zip_reconstruct_bytes`).
pub fn unpredict_and_reorder(scratch: &mut [u8]) -> Vec<u8> {
    let n = scratch.len();
    if n == 0 {
        return Vec::new();
    }
    let mut p = scratch[0];
    for item in scratch.iter_mut().skip(1) {
        let d = (i32::from(p) + i32::from(*item) - 128) as u8;
        *item = d;
        p = d;
    }

    let half = n.div_ceil(2);
    let mut out = vec![0u8; n];
    for (i, slot) in out.iter_mut().enumerate() {
        if (i & 1) == 0 {
            *slot = scratch[i / 2];
        } else {
            *slot = scratch[half + (i / 2)];
        }
    }
    out
}

// ============================================================================
// 2. RLE (`1`) — PackBits + Reorder/Predict (`OpenEXRCore/internal_rle.c`)
// ============================================================================

/// Decompresses an OpenEXR `RLE` chunk (`internal_rle_decompress` + `unpredict_and_reorder`).
pub fn rle_decompress_chunk(packed: &[u8], expected_len: usize) -> Result<Vec<u8>, ExrError> {
    let mut scratch = raw_rle_decompress(packed, expected_len)?;
    if scratch.len() != expected_len {
        return Err(ExrError::CorruptChunk(
            "RLE decompressed byte count mismatch",
        ));
    }
    Ok(unpredict_and_reorder(&mut scratch))
}

/// Compresses an OpenEXR `RLE` chunk (`reorder_and_predict` + `internal_rle_compress`).
pub fn rle_compress_chunk(uncompressed: &[u8]) -> Vec<u8> {
    let scratch = reorder_and_predict(uncompressed);
    raw_rle_compress(&scratch)
}

/// Raw signed-byte PackBits RLE decompressor (`internal_rle_decompress`).
pub fn raw_rle_decompress(src: &[u8], max_len: usize) -> Result<Vec<u8>, ExrError> {
    let mut out = Vec::with_capacity(max_len);
    let mut pos = 0usize;
    while pos < src.len() {
        let count = src[pos] as i8;
        pos += 1;
        if count < 0 {
            let n = (-i32::from(count)) as usize;
            if pos + n > src.len() || out.len() + n > max_len {
                return Err(ExrError::CorruptChunk("RLE literal run overflow"));
            }
            out.extend_from_slice(&src[pos..pos + n]);
            pos += n;
        } else {
            let n = (i32::from(count) + 1) as usize;
            if pos >= src.len() || out.len() + n > max_len {
                return Err(ExrError::CorruptChunk("RLE replicate run overflow"));
            }
            let val = src[pos];
            pos += 1;
            out.resize(out.len() + n, val);
        }
    }
    Ok(out)
}

/// Raw signed-byte PackBits RLE compressor (`internal_rle_compress`).
pub fn raw_rle_compress(src: &[u8]) -> Vec<u8> {
    const MIN_RUN_LENGTH: usize = 3;
    const MAX_RUN_LENGTH: usize = 127;

    let mut out = Vec::with_capacity(src.len() + src.len() / 128 + 4);
    let mut run_start = 0usize;
    let mut run_end = 1usize;

    while run_start < src.len() {
        while run_end < src.len()
            && src[run_start] == src[run_end]
            && (run_end - run_start) < MAX_RUN_LENGTH + 1
        {
            run_end += 1;
        }

        if run_end - run_start >= MIN_RUN_LENGTH {
            out.push(((run_end - run_start) - 1) as u8);
            out.push(src[run_start]);
            run_start = run_end;
        } else {
            while run_end < src.len()
                && ((run_end + 1 >= src.len() || src[run_end] != src[run_end + 1])
                    || (run_end + 2 >= src.len() || src[run_end + 1] != src[run_end + 2]))
                && (run_end - run_start) < MAX_RUN_LENGTH
            {
                run_end += 1;
            }
            out.push((run_start as isize - run_end as isize) as i8 as u8);
            out.extend_from_slice(&src[run_start..run_end]);
            run_start = run_end;
        }
        run_end += 1;
    }
    out
}

// ============================================================================
// 3. ZIPS (`2`) & ZIP (`3`) — Reorder/Predict + Zlib (`internal_zip.c`)
// ============================================================================

/// Decompresses an OpenEXR `ZIPS` or `ZIP` chunk.
pub fn zip_decompress_chunk(packed: &[u8], expected_len: usize) -> Result<Vec<u8>, ExrError> {
    let mut scratch = zlib_decompress(packed, expected_len)?;
    if scratch.len() != expected_len {
        return Err(ExrError::CorruptChunk(
            "ZIP decompressed size does not match expected chunk size",
        ));
    }
    Ok(unpredict_and_reorder(&mut scratch))
}

/// Compresses an OpenEXR `ZIPS` or `ZIP` chunk.
pub fn zip_compress_chunk(uncompressed: &[u8]) -> Vec<u8> {
    let scratch = reorder_and_predict(uncompressed);
    zlib_compress(&scratch)
}

// ============================================================================
// 4. PXR24 (`5`) — Pixar 24-bit float + Big-Endian Byte Transpose + Zlib (`internal_pxr24.c`)
// ============================================================================

/// Decompresses a `PXR24` chunk (`OpenEXRCore/internal_pxr24.c: undo_pxr24_impl`).
pub fn pxr24_decompress_chunk(
    packed: &[u8],
    expected_uncompressed_size: usize,
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
) -> Result<Vec<u8>, ExrError> {
    // Compute expected transposed size (where FLOAT takes 3 bytes per sample instead of 4)
    let mut expected_transposed = 0usize;
    for line in 0..num_lines {
        let y = y_start + (line as i32);
        for ch in channels {
            let n = ch.samples_in_line(y, region_width);
            let bps = match ch.sample_type {
                ExrSampleType::Uint => 4,
                ExrSampleType::Half => 2,
                ExrSampleType::Float => 3,
            };
            expected_transposed += n * bps;
        }
    }

    let transposed = zlib_decompress(packed, expected_transposed)?;
    if transposed.len() != expected_transposed {
        return Err(ExrError::CorruptChunk(
            "PXR24 decompressed byte count does not match expected transposed size",
        ));
    }

    let mut out = Vec::with_capacity(expected_uncompressed_size);
    let mut pos = 0usize;

    for line in 0..num_lines {
        let y = y_start + (line as i32);
        for ch in channels {
            let n = ch.samples_in_line(y, region_width);
            if n == 0 {
                continue;
            }
            match ch.sample_type {
                ExrSampleType::Uint => {
                    let p0 = &transposed[pos..pos + n];
                    let p1 = &transposed[pos + n..pos + 2 * n];
                    let p2 = &transposed[pos + 2 * n..pos + 3 * n];
                    let p3 = &transposed[pos + 3 * n..pos + 4 * n];
                    pos += 4 * n;
                    let mut pixel = 0u32;
                    for i in 0..n {
                        let diff = (u32::from(p0[i]) << 24)
                            | (u32::from(p1[i]) << 16)
                            | (u32::from(p2[i]) << 8)
                            | u32::from(p3[i]);
                        pixel = pixel.wrapping_add(diff);
                        out.extend_from_slice(&pixel.to_le_bytes());
                    }
                }
                ExrSampleType::Half => {
                    let p0 = &transposed[pos..pos + n];
                    let p1 = &transposed[pos + n..pos + 2 * n];
                    pos += 2 * n;
                    let mut pixel = 0u32;
                    for i in 0..n {
                        let diff = (u32::from(p0[i]) << 8) | u32::from(p1[i]);
                        pixel = pixel.wrapping_add(diff);
                        out.extend_from_slice(&(pixel as u16).to_le_bytes());
                    }
                }
                ExrSampleType::Float => {
                    let p0 = &transposed[pos..pos + n];
                    let p1 = &transposed[pos + n..pos + 2 * n];
                    let p2 = &transposed[pos + 2 * n..pos + 3 * n];
                    pos += 3 * n;
                    let mut pixel = 0u32;
                    for i in 0..n {
                        let diff = (u32::from(p0[i]) << 24)
                            | (u32::from(p1[i]) << 16)
                            | (u32::from(p2[i]) << 8);
                        pixel = pixel.wrapping_add(diff);
                        out.extend_from_slice(&pixel.to_le_bytes());
                    }
                }
            }
        }
    }

    if out.len() != expected_uncompressed_size {
        return Err(ExrError::CorruptChunk("PXR24 output size mismatch"));
    }
    let _ = float24_to_float;
    Ok(out)
}

/// Compresses a `PXR24` chunk (`OpenEXRCore/internal_pxr24.c: apply_pxr24_impl`).
pub fn pxr24_compress_chunk(
    uncompressed: &[u8],
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
) -> Result<Vec<u8>, ExrError> {
    let mut transposed = Vec::with_capacity(uncompressed.len());
    let mut pos = 0usize;

    for line in 0..num_lines {
        let y = y_start + (line as i32);
        for ch in channels {
            let n = ch.samples_in_line(y, region_width);
            if n == 0 {
                continue;
            }
            match ch.sample_type {
                ExrSampleType::Uint => {
                    if pos + 4 * n > uncompressed.len() {
                        return Err(ExrError::CorruptChunk("PXR24 input buffer underrun"));
                    }
                    let mut p0 = Vec::with_capacity(n);
                    let mut p1 = Vec::with_capacity(n);
                    let mut p2 = Vec::with_capacity(n);
                    let mut p3 = Vec::with_capacity(n);
                    let mut prev = 0u32;
                    for i in 0..n {
                        let off = pos + i * 4;
                        let cur = u32::from_le_bytes([
                            uncompressed[off],
                            uncompressed[off + 1],
                            uncompressed[off + 2],
                            uncompressed[off + 3],
                        ]);
                        let diff = cur.wrapping_sub(prev);
                        prev = cur;
                        p0.push((diff >> 24) as u8);
                        p1.push((diff >> 16) as u8);
                        p2.push((diff >> 8) as u8);
                        p3.push(diff as u8);
                    }
                    pos += 4 * n;
                    transposed.extend_from_slice(&p0);
                    transposed.extend_from_slice(&p1);
                    transposed.extend_from_slice(&p2);
                    transposed.extend_from_slice(&p3);
                }
                ExrSampleType::Half => {
                    if pos + 2 * n > uncompressed.len() {
                        return Err(ExrError::CorruptChunk("PXR24 input buffer underrun"));
                    }
                    let mut p0 = Vec::with_capacity(n);
                    let mut p1 = Vec::with_capacity(n);
                    let mut prev = 0u32;
                    for i in 0..n {
                        let off = pos + i * 2;
                        let cur = u32::from(u16::from_le_bytes([
                            uncompressed[off],
                            uncompressed[off + 1],
                        ]));
                        let diff = cur.wrapping_sub(prev);
                        prev = cur;
                        p0.push((diff >> 8) as u8);
                        p1.push(diff as u8);
                    }
                    pos += 2 * n;
                    transposed.extend_from_slice(&p0);
                    transposed.extend_from_slice(&p1);
                }
                ExrSampleType::Float => {
                    if pos + 4 * n > uncompressed.len() {
                        return Err(ExrError::CorruptChunk("PXR24 input buffer underrun"));
                    }
                    let mut p0 = Vec::with_capacity(n);
                    let mut p1 = Vec::with_capacity(n);
                    let mut p2 = Vec::with_capacity(n);
                    let mut prev = 0u32;
                    for i in 0..n {
                        let off = pos + i * 4;
                        let f = f32::from_le_bytes([
                            uncompressed[off],
                            uncompressed[off + 1],
                            uncompressed[off + 2],
                            uncompressed[off + 3],
                        ]);
                        let cur24 = float_to_float24(f);
                        let diff = cur24.wrapping_sub(prev);
                        prev = cur24;
                        p0.push((diff >> 16) as u8);
                        p1.push((diff >> 8) as u8);
                        p2.push(diff as u8);
                    }
                    pos += 4 * n;
                    transposed.extend_from_slice(&p0);
                    transposed.extend_from_slice(&p1);
                    transposed.extend_from_slice(&p2);
                }
            }
        }
    }

    Ok(zlib_compress(&transposed))
}

// ============================================================================
// 5. B44 (`6`) & B44A (`7`) — 4x4 Half-Float Block Compression (`internal_b44.c`)
// ============================================================================

#[inline]
fn b44_shift_and_round(x: i32, shift: u32) -> i32 {
    let x2 = x << 1;
    let a = (1i32 << shift) - 1;
    let shift1 = shift + 1;
    let b = (x2 >> shift1) & 1;
    (x2 + a + b) >> shift1
}

/// Packs a 4x4 block of 16-bit half-floats (`s[0..16]`) into 14 (or 3 if `flat_field` and all deltas are 0) bytes
/// (`OpenEXRCore/internal_b44.c: pack`).
fn b44_pack_block(s: &[u16; 16], flat_field: bool, b: &mut [u8; 14]) -> usize {
    let mut t = [0u16; 16];
    for i in 0..16 {
        if (s[i] & 0x7c00) == 0x7c00 {
            t[i] = 0x8000;
        } else if (s[i] & 0x8000) != 0 {
            t[i] = !s[i];
        } else {
            t[i] = s[i] | 0x8000;
        }
    }

    let t_max = *t.iter().max().unwrap_or(&0);

    let mut shift: i32 = -1;
    let mut d = [0i32; 16];
    let mut r = [0i32; 15];
    let mut r_min: i32;
    let mut r_max: i32;

    loop {
        shift += 1;
        let sh = shift as u32;
        for i in 0..16 {
            d[i] = b44_shift_and_round(i32::from(t_max) - i32::from(t[i]), sh);
        }

        const BIAS: i32 = 0x20;
        r[0] = d[0] - d[4] + BIAS;
        r[1] = d[4] - d[8] + BIAS;
        r[2] = d[8] - d[12] + BIAS;

        r[3] = d[0] - d[1] + BIAS;
        r[4] = d[4] - d[5] + BIAS;
        r[5] = d[8] - d[9] + BIAS;
        r[6] = d[12] - d[13] + BIAS;

        r[7] = d[1] - d[2] + BIAS;
        r[8] = d[5] - d[6] + BIAS;
        r[9] = d[9] - d[10] + BIAS;
        r[10] = d[13] - d[14] + BIAS;

        r[11] = d[2] - d[3] + BIAS;
        r[12] = d[6] - d[7] + BIAS;
        r[13] = d[10] - d[11] + BIAS;
        r[14] = d[14] - d[15] + BIAS;

        r_min = r[0];
        r_max = r[0];
        for &rv in &r[1..] {
            if rv < r_min {
                r_min = rv;
            }
            if rv > r_max {
                r_max = rv;
            }
        }

        if r_min >= 0 && r_max <= 0x3f {
            break;
        }
    }

    if r_min == 0x20 && r_max == 0x20 && flat_field {
        b[0] = (t[0] >> 8) as u8;
        b[1] = (t[0] & 0xff) as u8;
        b[2] = 0xfc;
        return 3;
    }

    b[0] = (t[0] >> 8) as u8;
    b[1] = (t[0] & 0xff) as u8;
    b[2] = ((shift as u8) << 2) | (((r[0] >> 4) & 0x03) as u8);
    b[3] = (((r[0] & 0x0f) << 4) | ((r[1] >> 2) & 0x0f)) as u8;
    b[4] = (((r[1] & 0x03) << 6) | (r[2] & 0x3f)) as u8;

    b[5] = (((r[3] & 0x3f) << 2) | ((r[4] >> 4) & 0x03)) as u8;
    b[6] = (((r[4] & 0x0f) << 4) | ((r[5] >> 2) & 0x0f)) as u8;
    b[7] = (((r[5] & 0x03) << 6) | (r[6] & 0x3f)) as u8;

    b[8] = (((r[7] & 0x3f) << 2) | ((r[8] >> 4) & 0x03)) as u8;
    b[9] = (((r[8] & 0x0f) << 4) | ((r[9] >> 2) & 0x0f)) as u8;
    b[10] = (((r[9] & 0x03) << 6) | (r[10] & 0x3f)) as u8;

    b[11] = (((r[11] & 0x3f) << 2) | ((r[12] >> 4) & 0x03)) as u8;
    b[12] = (((r[12] & 0x0f) << 4) | ((r[13] >> 2) & 0x0f)) as u8;
    b[13] = (((r[13] & 0x03) << 6) | (r[14] & 0x3f)) as u8;

    14
}

/// Unpacks a 14-byte `B44` block into 16 16-bit values (`OpenEXRCore/internal_b44.c: unpack14`).
fn b44_unpack14(b: &[u8], s: &mut [u16; 16]) {
    s[0] = (u16::from(b[0]) << 8) | u16::from(b[1]);

    let shift = u32::from(b[2] >> 2);
    let bias = 0x20u32 << shift;

    s[4] = s[0]
        .wrapping_add(((((u32::from(b[2]) << 4) | (u32::from(b[3]) >> 4)) & 0x3f) << shift) as u16)
        .wrapping_sub(bias as u16);
    s[8] = s[4]
        .wrapping_add(((((u32::from(b[3]) << 2) | (u32::from(b[4]) >> 6)) & 0x3f) << shift) as u16)
        .wrapping_sub(bias as u16);
    s[12] = s[8]
        .wrapping_add(((u32::from(b[4]) & 0x3f) << shift) as u16)
        .wrapping_sub(bias as u16);

    s[1] = s[0]
        .wrapping_add((u32::from(b[5] >> 2) << shift) as u16)
        .wrapping_sub(bias as u16);
    s[5] = s[4]
        .wrapping_add(((((u32::from(b[5]) << 4) | (u32::from(b[6]) >> 4)) & 0x3f) << shift) as u16)
        .wrapping_sub(bias as u16);
    s[9] = s[8]
        .wrapping_add(((((u32::from(b[6]) << 2) | (u32::from(b[7]) >> 6)) & 0x3f) << shift) as u16)
        .wrapping_sub(bias as u16);
    s[13] = s[12]
        .wrapping_add(((u32::from(b[7]) & 0x3f) << shift) as u16)
        .wrapping_sub(bias as u16);

    s[2] = s[1]
        .wrapping_add((u32::from(b[8] >> 2) << shift) as u16)
        .wrapping_sub(bias as u16);
    s[6] = s[5]
        .wrapping_add(((((u32::from(b[8]) << 4) | (u32::from(b[9]) >> 4)) & 0x3f) << shift) as u16)
        .wrapping_sub(bias as u16);
    s[10] = s[9]
        .wrapping_add(((((u32::from(b[9]) << 2) | (u32::from(b[10]) >> 6)) & 0x3f) << shift) as u16)
        .wrapping_sub(bias as u16);
    s[14] = s[13]
        .wrapping_add(((u32::from(b[10]) & 0x3f) << shift) as u16)
        .wrapping_sub(bias as u16);

    s[3] = s[2]
        .wrapping_add((u32::from(b[11] >> 2) << shift) as u16)
        .wrapping_sub(bias as u16);
    s[7] = s[6]
        .wrapping_add(
            ((((u32::from(b[11]) << 4) | (u32::from(b[12]) >> 4)) & 0x3f) << shift) as u16,
        )
        .wrapping_sub(bias as u16);
    s[11] = s[10]
        .wrapping_add(
            ((((u32::from(b[12]) << 2) | (u32::from(b[13]) >> 6)) & 0x3f) << shift) as u16,
        )
        .wrapping_sub(bias as u16);
    s[15] = s[14]
        .wrapping_add(((u32::from(b[13]) & 0x3f) << shift) as u16)
        .wrapping_sub(bias as u16);

    for val in s.iter_mut() {
        if (*val & 0x8000) != 0 {
            *val &= 0x7fff;
        } else {
            *val = !*val;
        }
    }
}

/// Unpacks a 3-byte `B44A` flat-field block (`OpenEXRCore/internal_b44.c: unpack3`).
fn b44_unpack3(b: &[u8], s: &mut [u16; 16]) {
    let mut v = (u16::from(b[0]) << 8) | u16::from(b[1]);
    if (v & 0x8000) != 0 {
        v &= 0x7fff;
    } else {
        v = !v;
    }
    s.fill(v);
}

/// Decompresses a `B44` or `B44A` chunk (`OpenEXRCore/internal_b44.c: undo_b44_impl`).
pub fn b44_decompress_chunk(
    packed: &[u8],
    expected_uncompressed_size: usize,
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
) -> Result<Vec<u8>, ExrError> {
    // Prepare per-channel plane buffers (channels are stored plane-by-plane in packed B44 stream,
    // then interleaved scanline-by-scanline in the output chunk buffer).
    let mut channel_planes: Vec<Vec<u8>> = Vec::with_capacity(channels.len());
    let mut channel_nx: Vec<usize> = Vec::with_capacity(channels.len());
    let mut channel_ny: Vec<usize> = Vec::with_capacity(channels.len());

    for ch in channels {
        let mut ny = 0usize;
        let mut nx = 0usize;
        for line in 0..num_lines {
            let y = y_start + (line as i32);
            let s = ch.samples_in_line(y, region_width);
            if s > 0 {
                nx = s;
                ny += 1;
            }
        }
        channel_nx.push(nx);
        channel_ny.push(ny);
        channel_planes.push(vec![0u8; nx * ny * ch.sample_type.bytes_per_sample()]);
    }

    let mut in_pos = 0usize;
    for (c_idx, ch) in channels.iter().enumerate() {
        let nx = channel_nx[c_idx];
        let ny = channel_ny[c_idx];
        if nx == 0 || ny == 0 {
            continue;
        }
        let plane = &mut channel_planes[c_idx];

        if ch.sample_type != ExrSampleType::Half {
            let nbytes = nx * ny * 4;
            if in_pos + nbytes > packed.len() {
                return Err(ExrError::CorruptChunk(
                    "B44 non-HALF channel truncated in stream",
                ));
            }
            plane.copy_from_slice(&packed[in_pos..in_pos + nbytes]);
            in_pos += nbytes;
            continue;
        }

        let mut y = 0usize;
        while y < ny {
            let mut x = 0usize;
            while x < nx {
                if in_pos + 3 > packed.len() {
                    return Err(ExrError::CorruptChunk("B44 block header truncated"));
                }
                let mut s = [0u16; 16];
                if packed[in_pos + 2] >= (13 << 2) {
                    if packed[in_pos + 2] != 0xfc {
                        return Err(ExrError::CorruptChunk("B44 invalid shift > 12 in block"));
                    }
                    b44_unpack3(&packed[in_pos..in_pos + 3], &mut s);
                    in_pos += 3;
                } else {
                    if in_pos + 14 > packed.len() {
                        return Err(ExrError::CorruptChunk("B44 14-byte block truncated"));
                    }
                    b44_unpack14(&packed[in_pos..in_pos + 14], &mut s);
                    in_pos += 14;
                }

                if ch.p_linear != 0 {
                    for v in &mut s {
                        *v = b44_convert_to_linear(*v);
                    }
                }

                let max_dy = (ny - y).min(4);
                let max_dx = (nx - x).min(4);
                for dy in 0..max_dy {
                    for dx in 0..max_dx {
                        let dst_idx = ((y + dy) * nx + (x + dx)) * 2;
                        let bytes = s[dy * 4 + dx].to_le_bytes();
                        plane[dst_idx] = bytes[0];
                        plane[dst_idx + 1] = bytes[1];
                    }
                }
                x += 4;
            }
            y += 4;
        }
    }

    // Re-interleave per-channel planes into canonical scanline-major chunk layout
    let mut out = Vec::with_capacity(expected_uncompressed_size);
    let mut ch_line_cursors = vec![0usize; channels.len()];
    for line in 0..num_lines {
        let y = y_start + (line as i32);
        for (c_idx, ch) in channels.iter().enumerate() {
            let n = ch.samples_in_line(y, region_width);
            if n == 0 {
                continue;
            }
            let bps = ch.sample_type.bytes_per_sample();
            let row_bytes = n * bps;
            let row_idx = ch_line_cursors[c_idx];
            ch_line_cursors[c_idx] += 1;
            let start = row_idx * row_bytes;
            out.extend_from_slice(&channel_planes[c_idx][start..start + row_bytes]);
        }
    }

    if out.len() != expected_uncompressed_size {
        return Err(ExrError::CorruptChunk("B44 output size mismatch"));
    }
    Ok(out)
}

/// Compresses a `B44` or `B44A` chunk (`OpenEXRCore/internal_b44.c: apply_b44_impl`).
pub fn b44_compress_chunk(
    uncompressed: &[u8],
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
    flat_field: bool,
) -> Result<Vec<u8>, ExrError> {
    // De-interleave scanlines into contiguous per-channel planes
    let mut channel_planes: Vec<Vec<u8>> = vec![Vec::new(); channels.len()];
    let mut channel_nx: Vec<usize> = vec![0; channels.len()];
    let mut channel_ny: Vec<usize> = vec![0; channels.len()];

    let mut pos = 0usize;
    for line in 0..num_lines {
        let y = y_start + (line as i32);
        for (c_idx, ch) in channels.iter().enumerate() {
            let n = ch.samples_in_line(y, region_width);
            if n == 0 {
                continue;
            }
            channel_nx[c_idx] = n;
            channel_ny[c_idx] += 1;
            let row_bytes = n * ch.sample_type.bytes_per_sample();
            if pos + row_bytes > uncompressed.len() {
                return Err(ExrError::CorruptChunk("B44 input buffer underrun"));
            }
            channel_planes[c_idx].extend_from_slice(&uncompressed[pos..pos + row_bytes]);
            pos += row_bytes;
        }
    }

    let mut out = Vec::with_capacity(uncompressed.len());
    for (c_idx, ch) in channels.iter().enumerate() {
        let nx = channel_nx[c_idx];
        let ny = channel_ny[c_idx];
        if nx == 0 || ny == 0 {
            continue;
        }
        let plane = &channel_planes[c_idx];
        if ch.sample_type != ExrSampleType::Half {
            out.extend_from_slice(plane);
            continue;
        }

        let mut y = 0usize;
        while y < ny {
            let mut x = 0usize;
            while x < nx {
                let mut s = [0u16; 16];
                for dy in 0..4 {
                    let sy = (y + dy).min(ny - 1);
                    for dx in 0..4 {
                        let sx = (x + dx).min(nx - 1);
                        let idx = (sy * nx + sx) * 2;
                        let mut val = u16::from_le_bytes([plane[idx], plane[idx + 1]]);
                        if ch.p_linear != 0 {
                            val = b44_convert_from_linear(val);
                        }
                        s[dy * 4 + dx] = val;
                    }
                }
                let mut blk = [0u8; 14];
                let nbytes = b44_pack_block(&s, flat_field, &mut blk);
                out.extend_from_slice(&blk[..nbytes]);
                x += 4;
            }
            y += 4;
        }
    }

    Ok(out)
}

// ============================================================================
// 6. PIZ (`4`) — 16-bit Bitmap LUT + 2D Haar Wavelet + 16-bit Canonical Huffman
//    (`OpenEXRCore/internal_piz.c` & `internal_huf.c`)
// ============================================================================

const BITMAP_SIZE: usize = 65536 / 8; // 8192 bytes

/// Decompresses a `PIZ` chunk (`OpenEXRCore/internal_piz.c: undo_piz_impl`).
pub fn piz_decompress_chunk(
    packed: &[u8],
    expected_uncompressed_size: usize,
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
) -> Result<Vec<u8>, ExrError> {
    if expected_uncompressed_size % 2 != 0 {
        return Err(ExrError::CorruptChunk(
            "PIZ expected uncompressed size is not a multiple of 2",
        ));
    }
    let n_u16 = expected_uncompressed_size / 2;

    if packed.len() < 4 {
        return Err(ExrError::CorruptChunk("PIZ stream shorter than 4 bytes"));
    }
    let min_non_zero = u16::from_le_bytes([packed[0], packed[1]]) as usize;
    let max_non_zero = u16::from_le_bytes([packed[2], packed[3]]) as usize;
    if max_non_zero >= BITMAP_SIZE {
        return Err(ExrError::CorruptChunk(
            "PIZ max_non_zero exceeds BITMAP_SIZE (8192)",
        ));
    }

    let mut pos = 4usize;
    let mut bitmap = [0u8; BITMAP_SIZE];
    if min_non_zero <= max_non_zero {
        let bm_bytes = max_non_zero - min_non_zero + 1;
        if pos + bm_bytes > packed.len() {
            return Err(ExrError::CorruptChunk("PIZ bitmap truncated"));
        }
        bitmap[min_non_zero..=max_non_zero].copy_from_slice(&packed[pos..pos + bm_bytes]);
        pos += bm_bytes;
    }

    let mut lut = [0u16; 65536];
    let max_value = reverse_lut_from_bitmap(&bitmap, &mut lut);

    if pos + 4 > packed.len() {
        return Err(ExrError::CorruptChunk("PIZ Huffman length field truncated"));
    }
    let huf_len = u32::from_le_bytes([
        packed[pos],
        packed[pos + 1],
        packed[pos + 2],
        packed[pos + 3],
    ]) as usize;
    pos += 4;
    if pos + huf_len > packed.len() {
        return Err(ExrError::CorruptChunk("PIZ Huffman payload truncated"));
    }

    let mut tmp_buf = huf_decompress(&packed[pos..pos + huf_len], n_u16)?;

    // Apply 2D inverse Haar wavelet per channel plane
    let mut plane_offset = 0usize;
    for ch in channels {
        let mut ny = 0usize;
        let mut nx = 0usize;
        for line in 0..num_lines {
            let y = y_start + (line as i32);
            let s = ch.samples_in_line(y, region_width);
            if s > 0 {
                nx = s;
                ny += 1;
            }
        }
        let u16_per_sample = ch.sample_type.bytes_per_sample() / 2;
        let plane_u16_len = nx * ny * u16_per_sample;
        if plane_offset + plane_u16_len > tmp_buf.len() {
            return Err(ExrError::CorruptChunk("PIZ channel plane bounds mismatch"));
        }
        if nx > 0 && ny > 0 {
            let plane_slice = &mut tmp_buf[plane_offset..plane_offset + plane_u16_len];
            for sub in 0..u16_per_sample {
                wav2d_decode(
                    &mut plane_slice[sub..],
                    nx,
                    u16_per_sample,
                    ny,
                    nx * u16_per_sample,
                    max_value,
                );
            }
        }
        plane_offset += plane_u16_len;
    }

    // Apply reverse LUT
    for v in &mut tmp_buf {
        if (*v as usize) <= (max_value as usize) {
            *v = lut[*v as usize];
        } else {
            return Err(ExrError::CorruptChunk(
                "PIZ wavelet output exceeds LUT max_value",
            ));
        }
    }

    // Interleave channel planes back into scanline-major chunk layout
    let mut out = Vec::with_capacity(expected_uncompressed_size);
    let mut ch_plane_offsets = Vec::with_capacity(channels.len());
    let mut ch_row_u16_lens = Vec::with_capacity(channels.len());
    let mut cur_off = 0usize;
    for ch in channels {
        let mut ny = 0usize;
        let mut nx = 0usize;
        for line in 0..num_lines {
            let y = y_start + (line as i32);
            let s = ch.samples_in_line(y, region_width);
            if s > 0 {
                nx = s;
                ny += 1;
            }
        }
        let u16_per_sample = ch.sample_type.bytes_per_sample() / 2;
        let row_u16 = nx * u16_per_sample;
        ch_plane_offsets.push(cur_off);
        ch_row_u16_lens.push(row_u16);
        cur_off += row_u16 * ny;
    }

    let mut ch_line_cursors = vec![0usize; channels.len()];
    for line in 0..num_lines {
        let y = y_start + (line as i32);
        for (c_idx, ch) in channels.iter().enumerate() {
            let s = ch.samples_in_line(y, region_width);
            if s == 0 {
                continue;
            }
            let row_u16 = ch_row_u16_lens[c_idx];
            let r_idx = ch_line_cursors[c_idx];
            ch_line_cursors[c_idx] += 1;
            let start = ch_plane_offsets[c_idx] + r_idx * row_u16;
            for &val in &tmp_buf[start..start + row_u16] {
                out.extend_from_slice(&val.to_le_bytes());
            }
        }
    }

    Ok(out)
}

/// Compresses a `PIZ` chunk (`OpenEXRCore/internal_piz.c: apply_piz_impl`).
pub fn piz_compress_chunk(
    uncompressed: &[u8],
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
) -> Result<Vec<u8>, ExrError> {
    if uncompressed.len() % 2 != 0 {
        return Err(ExrError::CorruptChunk(
            "PIZ uncompressed length not divisible by 2",
        ));
    }
    let n_u16 = uncompressed.len() / 2;

    // De-interleave scanlines into contiguous per-channel u16 planes
    let mut ch_planes: Vec<Vec<u16>> = vec![Vec::new(); channels.len()];
    let mut ch_nx: Vec<usize> = vec![0; channels.len()];
    let mut ch_ny: Vec<usize> = vec![0; channels.len()];

    let mut pos = 0usize;
    for line in 0..num_lines {
        let y = y_start + (line as i32);
        for (c_idx, ch) in channels.iter().enumerate() {
            let s = ch.samples_in_line(y, region_width);
            if s == 0 {
                continue;
            }
            ch_nx[c_idx] = s;
            ch_ny[c_idx] += 1;
            let u16_per_sample = ch.sample_type.bytes_per_sample() / 2;
            let row_u16 = s * u16_per_sample;
            for _ in 0..row_u16 {
                let v = u16::from_le_bytes([uncompressed[pos], uncompressed[pos + 1]]);
                pos += 2;
                ch_planes[c_idx].push(v);
            }
        }
    }

    let mut tmp_buf = Vec::with_capacity(n_u16);
    for plane in &ch_planes {
        tmp_buf.extend_from_slice(plane);
    }

    // Build bitmap & forward LUT
    let mut bitmap = [0u8; BITMAP_SIZE];
    let mut min_non_zero = BITMAP_SIZE;
    let mut max_non_zero = 0usize;
    bitmap_from_data(&tmp_buf, &mut bitmap, &mut min_non_zero, &mut max_non_zero);

    let mut lut = [0u16; 65536];
    let max_value = forward_lut_from_bitmap(&bitmap, &mut lut);
    for v in &mut tmp_buf {
        *v = lut[*v as usize];
    }

    // Apply 2D forward Haar wavelet per channel plane
    let mut plane_offset = 0usize;
    for (c_idx, ch) in channels.iter().enumerate() {
        let nx = ch_nx[c_idx];
        let ny = ch_ny[c_idx];
        let u16_per_sample = ch.sample_type.bytes_per_sample() / 2;
        let plane_u16_len = nx * ny * u16_per_sample;
        if nx > 0 && ny > 0 {
            let plane_slice = &mut tmp_buf[plane_offset..plane_offset + plane_u16_len];
            for sub in 0..u16_per_sample {
                wav2d_encode(
                    &mut plane_slice[sub..],
                    nx,
                    u16_per_sample,
                    ny,
                    nx * u16_per_sample,
                    max_value,
                );
            }
        }
        plane_offset += plane_u16_len;
    }

    // Huffman encode `tmp_buf`
    let huf_bytes = huf_compress(&tmp_buf)?;

    let mut out = Vec::with_capacity(8 + BITMAP_SIZE + huf_bytes.len());
    out.extend_from_slice(&(min_non_zero as u16).to_le_bytes());
    out.extend_from_slice(&(max_non_zero as u16).to_le_bytes());
    if min_non_zero <= max_non_zero {
        out.extend_from_slice(&bitmap[min_non_zero..=max_non_zero]);
    }
    out.extend_from_slice(&(huf_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&huf_bytes);
    Ok(out)
}

fn bitmap_from_data(
    data: &[u16],
    bitmap: &mut [u8; BITMAP_SIZE],
    min_non_zero: &mut usize,
    max_non_zero: &mut usize,
) {
    bitmap.fill(0);
    for &v in data {
        bitmap[(v >> 3) as usize] |= 1u8 << (v & 7);
    }
    bitmap[0] &= !1u8; // zero is implicit

    *min_non_zero = BITMAP_SIZE - 1;
    *max_non_zero = 0;
    for (i, &b) in bitmap.iter().enumerate() {
        if b != 0 {
            if *min_non_zero > i {
                *min_non_zero = i;
            }
            if *max_non_zero < i {
                *max_non_zero = i;
            }
        }
    }
}

fn forward_lut_from_bitmap(bitmap: &[u8; BITMAP_SIZE], lut: &mut [u16; 65536]) -> u16 {
    let mut k = 0u16;
    for i in 0..65536usize {
        if i == 0 || (bitmap[i >> 3] & (1u8 << (i & 7))) != 0 {
            lut[i] = k;
            k = k.wrapping_add(1);
        } else {
            lut[i] = 0;
        }
    }
    k.saturating_sub(1)
}

fn reverse_lut_from_bitmap(bitmap: &[u8; BITMAP_SIZE], lut: &mut [u16; 65536]) -> u16 {
    let mut k = 0usize;
    for i in 0..65536usize {
        if i == 0 || (bitmap[i >> 3] & (1u8 << (i & 7))) != 0 {
            lut[k] = i as u16;
            k += 1;
        }
    }
    let n = k.saturating_sub(1);
    while k < 65536 {
        lut[k] = 0;
        k += 1;
    }
    n as u16
}

// ============================================================================
// 6a. 2D Haar Wavelet (`wenc14`/`wdec14` and `wenc16`/`wdec16` in `internal_piz.c`)
// ============================================================================

const NBITS: u32 = 16;
const A_OFFSET: i32 = 1 << (NBITS - 1);
const M_OFFSET: i32 = 1 << (NBITS - 1);
const MOD_MASK: i32 = (1 << NBITS) - 1;

#[inline]
fn wenc14(a: u16, b: u16) -> (u16, u16) {
    let as_ = a as i16;
    let bs = b as i16;
    let ms = (as_.wrapping_add(bs)) >> 1;
    let ds = as_.wrapping_sub(bs);
    (ms as u16, ds as u16)
}

#[inline]
fn wdec14(l: u16, h: u16) -> (u16, u16) {
    let ls = l as i16;
    let hs = h as i16;
    let hi = i32::from(hs);
    let ai = i32::from(ls) + (hi & 1) + (hi >> 1);
    let as_ = ai as i16;
    let bs = (ai - hi) as i16;
    (as_ as u16, bs as u16)
}

#[inline]
fn wenc16(a: u16, b: u16) -> (u16, u16) {
    let ao = (i32::from(a) + A_OFFSET) & MOD_MASK;
    let mut m = (ao + i32::from(b)) >> 1;
    let d = ao - i32::from(b);
    if d < 0 {
        m = (m + M_OFFSET) & MOD_MASK;
    }
    let d_masked = d & MOD_MASK;
    (m as u16, d_masked as u16)
}

#[inline]
fn wdec16(l: u16, h: u16) -> (u16, u16) {
    let m = i32::from(l);
    let d = i32::from(h);
    let bb = (m - (d >> 1)) & MOD_MASK;
    let aa = (d + bb - A_OFFSET) & MOD_MASK;
    (aa as u16, bb as u16)
}

fn wav2d_encode(in_buf: &mut [u16], nx: usize, ox: usize, ny: usize, oy: usize, max_val: u16) {
    let w14 = max_val < (1u16 << 14);
    let n = nx.min(ny);
    let mut p = 1usize;
    let mut p2 = 2usize;

    while p2 <= n {
        let oy1 = oy * p;
        let oy2 = oy * p2;
        let ox1 = ox * p;
        let ox2 = ox * p2;

        // Y loop
        let mut py = 0isize;
        let ey = (oy as isize) * ((ny as isize) - (p2 as isize));

        while py <= ey {
            let mut px = py;
            let ex = py + (ox as isize) * ((nx as isize) - (p2 as isize));
            while px <= ex {
                let p00 = px as usize;
                let p01 = p00 + ox1;
                let p10 = p00 + oy1;
                let p11 = p10 + ox1;

                if w14 {
                    let (i00, i01) = wenc14(in_buf[p00], in_buf[p01]);
                    let (i10, i11) = wenc14(in_buf[p10], in_buf[p11]);
                    let (o00, o10) = wenc14(i00, i10);
                    let (o01, o11) = wenc14(i01, i11);
                    in_buf[p00] = o00;
                    in_buf[p10] = o10;
                    in_buf[p01] = o01;
                    in_buf[p11] = o11;
                } else {
                    let (i00, i01) = wenc16(in_buf[p00], in_buf[p01]);
                    let (i10, i11) = wenc16(in_buf[p10], in_buf[p11]);
                    let (o00, o10) = wenc16(i00, i10);
                    let (o01, o11) = wenc16(i01, i11);
                    in_buf[p00] = o00;
                    in_buf[p10] = o10;
                    in_buf[p01] = o01;
                    in_buf[p11] = o11;
                }
                px += ox2 as isize;
            }
            if (nx & p) != 0 {
                let p00 = px as usize;
                let p10 = p00 + oy1;
                let (o00, o10) = if w14 {
                    wenc14(in_buf[p00], in_buf[p10])
                } else {
                    wenc16(in_buf[p00], in_buf[p10])
                };
                in_buf[p00] = o00;
                in_buf[p10] = o10;
            }
            py += oy2 as isize;
        }

        if (ny & p) != 0 {
            let mut px = py;
            let ex = py + (ox as isize) * ((nx as isize) - (p2 as isize));
            while px <= ex {
                let p00 = px as usize;
                let p01 = p00 + ox1;
                let (o00, o01) = if w14 {
                    wenc14(in_buf[p00], in_buf[p01])
                } else {
                    wenc16(in_buf[p00], in_buf[p01])
                };
                in_buf[p00] = o00;
                in_buf[p01] = o01;
                px += ox2 as isize;
            }
        }

        p = p2;
        p2 <<= 1;
    }
}

fn wav2d_decode(in_buf: &mut [u16], nx: usize, ox: usize, ny: usize, oy: usize, max_val: u16) {
    let w14 = max_val < (1u16 << 14);
    let n = nx.min(ny);
    let mut p = 1usize;
    while p <= n {
        p <<= 1;
    }
    p >>= 2;

    while p >= 1 {
        let p2 = p << 1;
        let oy1 = oy * p;
        let oy2 = oy * p2;
        let ox1 = ox * p;
        let ox2 = ox * p2;

        let mut py = 0isize;
        let ey = (oy as isize) * ((ny as isize) - (p2 as isize));

        while py <= ey {
            let mut px = py;
            let ex = py + (ox as isize) * ((nx as isize) - (p2 as isize));
            while px <= ex {
                let p00 = px as usize;
                let p01 = p00 + ox1;
                let p10 = p00 + oy1;
                let p11 = p10 + ox1;

                if w14 {
                    let (i00, i10) = wdec14(in_buf[p00], in_buf[p10]);
                    let (i01, i11) = wdec14(in_buf[p01], in_buf[p11]);
                    let (o00, o01) = wdec14(i00, i01);
                    let (o10, o11) = wdec14(i10, i11);
                    in_buf[p00] = o00;
                    in_buf[p01] = o01;
                    in_buf[p10] = o10;
                    in_buf[p11] = o11;
                } else {
                    let (i00, i10) = wdec16(in_buf[p00], in_buf[p10]);
                    let (i01, i11) = wdec16(in_buf[p01], in_buf[p11]);
                    let (o00, o01) = wdec16(i00, i01);
                    let (o10, o11) = wdec16(i10, i11);
                    in_buf[p00] = o00;
                    in_buf[p01] = o01;
                    in_buf[p10] = o10;
                    in_buf[p11] = o11;
                }
                px += ox2 as isize;
            }
            if (nx & p) != 0 {
                let p00 = px as usize;
                let p10 = p00 + oy1;
                let (o00, o10) = if w14 {
                    wdec14(in_buf[p00], in_buf[p10])
                } else {
                    wdec16(in_buf[p00], in_buf[p10])
                };
                in_buf[p00] = o00;
                in_buf[p10] = o10;
            }
            py += oy2 as isize;
        }

        if (ny & p) != 0 {
            let mut px = py;
            let ex = py + (ox as isize) * ((nx as isize) - (p2 as isize));
            while px <= ex {
                let p00 = px as usize;
                let p01 = p00 + ox1;
                let (o00, o01) = if w14 {
                    wdec14(in_buf[p00], in_buf[p01])
                } else {
                    wdec16(in_buf[p00], in_buf[p01])
                };
                in_buf[p00] = o00;
                in_buf[p01] = o01;
                px += ox2 as isize;
            }
        }

        p >>= 1;
    }
}

// ============================================================================
// 6b. 16-bit Canonical Huffman (`Huf` in `OpenEXRCore/internal_huf.c`)
// ============================================================================

const HUF_ENCBITS: usize = 16;
const HUF_ENCSIZE: usize = (1 << HUF_ENCBITS) + 1; // 65537
const HUF_MAX_LEN: usize = 58;
const SHORT_ZEROCODE_RUN: u64 = 59;
const LONG_ZEROCODE_RUN: u64 = 63;
const SHORTEST_LONG_RUN: usize = 2 + (LONG_ZEROCODE_RUN - SHORT_ZEROCODE_RUN) as usize; // 6
const LONGEST_LONG_RUN: usize = 255 + SHORTEST_LONG_RUN; // 261

struct MsbBitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u64,
    bit_count: u32,
}

impl<'a> MsbBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bit_count: 0,
        }
    }

    fn read_bits(&mut self, n: u32) -> Result<u64, ExrError> {
        if n == 0 {
            return Ok(0);
        }
        while self.bit_count < n {
            if self.pos >= self.data.len() {
                return Err(ExrError::CorruptChunk("PIZ Huf table unexpected EOF"));
            }
            self.bit_buf = (self.bit_buf << 8) | u64::from(self.data[self.pos]);
            self.pos += 1;
            self.bit_count += 8;
        }
        self.bit_count -= n;
        let val = (self.bit_buf >> self.bit_count) & ((1u64 << n) - 1);
        Ok(val)
    }

    fn byte_pos_after_align(&self) -> usize {
        self.pos - ((self.bit_count / 8) as usize)
    }
}

struct MsbBitWriter {
    out: Vec<u8>,
    bit_buf: u64,
    bit_count: u32,
}

impl MsbBitWriter {
    fn with_capacity(cap: usize) -> Self {
        Self {
            out: Vec::with_capacity(cap),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    #[inline]
    fn write_bits(&mut self, count: u32, bits: u64) {
        self.bit_buf = (self.bit_buf << count) | (bits & ((1u64 << count) - 1));
        self.bit_count += count;
        while self.bit_count >= 8 {
            self.bit_count -= 8;
            self.out
                .push(((self.bit_buf >> self.bit_count) & 0xff) as u8);
        }
    }

    fn flush_align(&mut self) -> u32 {
        if self.bit_count > 0 {
            let rem = self.bit_count;
            let pad = 8 - rem;
            self.out.push(((self.bit_buf << pad) & 0xff) as u8);
            self.bit_buf = 0;
            self.bit_count = 0;
            rem
        } else {
            8
        }
    }
}

fn huf_canonical_code_table(hcode: &mut [u64]) {
    let mut n = [0u64; HUF_MAX_LEN + 1];
    for &l in hcode.iter() {
        if (l as usize) <= HUF_MAX_LEN {
            n[l as usize] += 1;
        }
    }
    let mut c = 0u64;
    for i in (1..=HUF_MAX_LEN).rev() {
        let nc = (c + n[i]) >> 1;
        n[i] = c;
        c = nc;
    }
    for entry in hcode.iter_mut() {
        let l = *entry as usize;
        if l > 0 {
            *entry = l as u64 | (n[l] << 6);
            n[l] += 1;
        }
    }
}

/// Decompresses an OpenEXR 16-bit canonical Huffman stream (`internal_huf_decompress`).
pub fn huf_decompress(compressed: &[u8], n_raw_u16: usize) -> Result<Vec<u16>, ExrError> {
    if n_raw_u16 == 0 {
        return Ok(Vec::new());
    }
    if compressed.len() < 20 {
        return Err(ExrError::CorruptChunk("PIZ Huf header < 20 bytes"));
    }

    let im =
        u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]) as usize;
    let i_m =
        u32::from_le_bytes([compressed[4], compressed[5], compressed[6], compressed[7]]) as usize;
    let _table_length =
        u32::from_le_bytes([compressed[8], compressed[9], compressed[10], compressed[11]]);
    let n_bits = u32::from_le_bytes([
        compressed[12],
        compressed[13],
        compressed[14],
        compressed[15],
    ]) as usize;
    let future = u32::from_le_bytes([
        compressed[16],
        compressed[17],
        compressed[18],
        compressed[19],
    ]);

    if future != 0 || im >= HUF_ENCSIZE || i_m >= HUF_ENCSIZE || im > i_m {
        return Err(ExrError::CorruptChunk("invalid PIZ Huf header fields"));
    }

    // 1. Unpack code lengths (`hufUnpackEncTable`)
    let mut reader = MsbBitReader::new(&compressed[20..]);
    let mut freq = vec![0u64; HUF_ENCSIZE];
    let mut sym = im;
    while sym <= i_m {
        let l = reader.read_bits(6)?;
        freq[sym] = l;
        if l == LONG_ZEROCODE_RUN {
            let zerun = (reader.read_bits(8)? as usize) + SHORTEST_LONG_RUN;
            if sym + zerun > i_m + 1 {
                return Err(ExrError::CorruptChunk("PIZ Huf long zero run overflow"));
            }
            for k in 0..zerun {
                freq[sym + k] = 0;
            }
            sym += zerun;
        } else if l >= SHORT_ZEROCODE_RUN {
            let zerun = (l - SHORT_ZEROCODE_RUN + 2) as usize;
            if sym + zerun > i_m + 1 {
                return Err(ExrError::CorruptChunk("PIZ Huf short zero run overflow"));
            }
            for k in 0..zerun {
                freq[sym + k] = 0;
            }
            sym += zerun;
        } else {
            sym += 1;
        }
    }

    let data_start = 20 + reader.byte_pos_after_align();
    let n_bytes = n_bits.div_ceil(8);
    if data_start + n_bytes > compressed.len() {
        return Err(ExrError::CorruptChunk("PIZ Huf bitstream truncated"));
    }

    // 2. Build canonical code lengths & sorted lookup table (`hufBuildDecTable`)
    let mut counts = [0u32; HUF_MAX_LEN + 1];
    for &l in &freq[..=i_m] {
        let len = l as usize;
        if len > HUF_MAX_LEN {
            return Err(ExrError::CorruptChunk("PIZ Huf code length > 58"));
        }
        if len > 0 {
            counts[len] += 1;
        }
    }

    let mut base_code = [0u64; HUF_MAX_LEN + 1];
    let mut c = 0u64;
    for l in (1..=HUF_MAX_LEN).rev() {
        let nc = (c + u64::from(counts[l])) >> 1;
        base_code[l] = c;
        c = nc;
    }
    if c != 1 && !(c == 0 && im == i_m && counts.iter().sum::<u32>() == 0) {
        return Err(ExrError::CorruptChunk(
            "PIZ Huf code lengths do not form valid tree",
        ));
    }

    let mut offsets = [0usize; HUF_MAX_LEN + 1];
    let mut cur_off = 0usize;
    for l in 1..=HUF_MAX_LEN {
        offsets[l] = cur_off;
        cur_off += counts[l] as usize;
    }
    let mut symbols = vec![0u32; cur_off];
    let mut write_off = offsets;
    for (s, &l) in freq[..=i_m].iter().enumerate() {
        let len = l as usize;
        if len > 0 {
            symbols[write_off[len]] = s as u32;
            write_off[len] += 1;
        }
    }

    // 3. Decode MSB-first bitstream (`hufDecode`)
    let rlc_sym = i_m as u32;
    let payload = &compressed[data_start..data_start + n_bytes];
    let mut out = Vec::with_capacity(n_raw_u16);

    let mut bit_buf = 0u64;
    let mut bit_cnt = 0usize;
    let mut byte_idx = 0usize;
    let mut bits_consumed = 0usize;

    let read_one_bit = |byte_idx: &mut usize,
                        bit_buf: &mut u64,
                        bit_cnt: &mut usize,
                        bits_consumed: &mut usize|
     -> Result<u64, ExrError> {
        if *bits_consumed >= n_bits {
            return Err(ExrError::CorruptChunk("PIZ Huf bitstream exhausted"));
        }
        if *bit_cnt == 0 {
            if *byte_idx >= payload.len() {
                return Err(ExrError::CorruptChunk("PIZ Huf payload exhausted"));
            }
            *bit_buf = u64::from(payload[*byte_idx]);
            *byte_idx += 1;
            *bit_cnt = 8;
        }
        *bit_cnt -= 1;
        *bits_consumed += 1;
        Ok((*bit_buf >> *bit_cnt) & 1)
    };

    while bits_consumed < n_bits && out.len() < n_raw_u16 {
        let mut code = 0u64;
        let mut matched_sym: Option<u32> = None;
        for l in 1..=HUF_MAX_LEN {
            let bit = read_one_bit(
                &mut byte_idx,
                &mut bit_buf,
                &mut bit_cnt,
                &mut bits_consumed,
            )?;
            code = (code << 1) | bit;
            let cnt = u64::from(counts[l]);
            if cnt > 0 && code >= base_code[l] && code < base_code[l] + cnt {
                let idx = offsets[l] + ((code - base_code[l]) as usize);
                matched_sym = Some(symbols[idx]);
                break;
            }
        }
        let s = matched_sym.ok_or(ExrError::CorruptChunk("PIZ Huf unmatched code"))?;
        if s == rlc_sym {
            let mut run = 0usize;
            for _ in 0..8 {
                let bit = read_one_bit(
                    &mut byte_idx,
                    &mut bit_buf,
                    &mut bit_cnt,
                    &mut bits_consumed,
                )?;
                run = (run << 1) | (bit as usize);
            }
            let prev = *out
                .last()
                .ok_or(ExrError::CorruptChunk("PIZ Huf RLC at start of stream"))?;
            if out.len() + run > n_raw_u16 {
                return Err(ExrError::CorruptChunk(
                    "PIZ Huf RLC run exceeds output size",
                ));
            }
            out.resize(out.len() + run, prev);
        } else {
            out.push(s as u16);
        }
    }

    if out.len() != n_raw_u16 {
        return Err(ExrError::CorruptChunk(
            "PIZ Huf decoded sample count does not match expected size",
        ));
    }
    Ok(out)
}

/// Compresses a `u16` slice into an OpenEXR 16-bit canonical Huffman stream (`internal_huf_compress`).
pub fn huf_compress(raw: &[u16]) -> Result<Vec<u8>, ExrError> {
    if raw.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Count symbol frequencies (`countFrequencies`)
    let mut freq = vec![0u64; HUF_ENCSIZE];
    for &s in raw {
        freq[usize::from(s)] += 1;
    }

    // Find `im` (first non-zero) and `iM` (last non-zero + 1 for RLC symbol)
    let mut im = 0usize;
    while im < HUF_ENCSIZE && freq[im] == 0 {
        im += 1;
    }
    let mut i_m = HUF_ENCSIZE - 1;
    while i_m > im && freq[i_m] == 0 {
        i_m -= 1;
    }
    i_m += 1; // Reserve `iM` as the RLC escape symbol
    freq[i_m] = 1;

    // 2. Build optimal length-limited Huffman code lengths (`hufBuildEncTable`)
    let mut hcode = build_huf_code_lengths(&freq, im, i_m);

    // 3. Pack encoding table (`hufPackEncTable`)
    let mut table_writer = MsbBitWriter::with_capacity(1024);
    let mut s = im;
    while s <= i_m {
        let mut l = hcode[s];
        if l == 0 {
            let mut zerun = 1usize;
            while s + zerun <= i_m && zerun < LONGEST_LONG_RUN && hcode[s + zerun] == 0 {
                zerun += 1;
            }
            if zerun >= 2 {
                if zerun >= SHORTEST_LONG_RUN {
                    table_writer.write_bits(6, LONG_ZEROCODE_RUN);
                    table_writer.write_bits(8, (zerun - SHORTEST_LONG_RUN) as u64);
                } else {
                    table_writer.write_bits(6, SHORT_ZEROCODE_RUN + (zerun as u64) - 2);
                }
                s += zerun;
                continue;
            }
        }
        table_writer.write_bits(6, l);
        s += 1;
        let _ = &mut l;
    }
    let _ = table_writer.flush_align();
    let table_bytes = table_writer.out;

    // 4. Convert lengths into `(code << 6) | length` (`hufCanonicalCodeTable`)
    huf_canonical_code_table(&mut hcode);

    // 5. Encode symbols + RLC runs (`hufEncode`)
    let mut data_writer = MsbBitWriter::with_capacity(raw.len() * 2);
    let rlc_code = hcode[i_m];
    let rlc_len = (rlc_code & 63) as u32;
    let rlc_bits = rlc_code >> 6;

    let mut s_cur = raw[0];
    let mut cs = 0usize;
    for &sym in &raw[1..] {
        if s_cur == sym && cs < 255 {
            cs += 1;
        } else {
            let code = hcode[usize::from(s_cur)];
            let clen = (code & 63) as u32;
            let cbits = code >> 6;
            data_writer.write_bits(clen, cbits);
            if cs * (clen as usize) > (rlc_len as usize) + 8 {
                data_writer.write_bits(rlc_len, rlc_bits);
                data_writer.write_bits(8, cs as u64);
            } else {
                for _ in 0..cs {
                    data_writer.write_bits(clen, cbits);
                }
            }
            cs = 0;
        }
        s_cur = sym;
    }
    // Flush trailing run
    let code = hcode[usize::from(s_cur)];
    let clen = (code & 63) as u32;
    let cbits = code >> 6;
    data_writer.write_bits(clen, cbits);
    if cs * (clen as usize) > (rlc_len as usize) + 8 {
        data_writer.write_bits(rlc_len, rlc_bits);
        data_writer.write_bits(8, cs as u64);
    } else {
        for _ in 0..cs {
            data_writer.write_bits(clen, cbits);
        }
    }

    let valid_last_bits = data_writer.flush_align();
    let data_bytes = data_writer.out;
    let n_bits = if data_bytes.is_empty() {
        0u32
    } else {
        ((data_bytes.len() as u32) - 1) * 8 + valid_last_bits
    };

    // 6. Write 20-byte header + table_bytes + data_bytes
    let mut out = Vec::with_capacity(20 + table_bytes.len() + data_bytes.len());
    out.extend_from_slice(&(im as u32).to_le_bytes());
    out.extend_from_slice(&(i_m as u32).to_le_bytes());
    out.extend_from_slice(&(table_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&n_bits.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&table_bytes);
    out.extend_from_slice(&data_bytes);
    Ok(out)
}

/// Builds Huffman code lengths `<= 58` for `freq[im..=i_m]` using `OpenEXRCore/internal_huf.c`'s
/// merge-chain algorithm (`hufBuildEncTable`).
fn build_huf_code_lengths(freq: &[u64], im: usize, i_m: usize) -> Vec<u64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut hlink = vec![0usize; HUF_ENCSIZE];
    let mut fheap = BinaryHeap::new();
    let mut scode = vec![0u64; HUF_ENCSIZE];
    let mut working_freq = freq.to_vec();

    for i in im..=i_m {
        hlink[i] = i;
        if working_freq[i] > 0 {
            fheap.push(Reverse((working_freq[i], i)));
        }
    }

    while fheap.len() > 1 {
        let Reverse((f_mm, mm)) = fheap.pop().unwrap();
        let Reverse((f_m, m)) = fheap.pop().unwrap();
        let combined = f_mm + f_m;
        working_freq[m] = combined;
        fheap.push(Reverse((combined, m)));

        let mut j = m;
        loop {
            scode[j] += 1;
            assert!(scode[j] <= (HUF_MAX_LEN as u64));
            if hlink[j] == j {
                hlink[j] = mm;
                break;
            }
            j = hlink[j];
        }

        let mut k = mm;
        loop {
            scode[k] += 1;
            assert!(scode[k] <= (HUF_MAX_LEN as u64));
            if hlink[k] == k {
                break;
            }
            k = hlink[k];
        }
    }

    scode
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rle_roundtrip() {
        let mut data = Vec::new();
        for i in 0..512u16 {
            data.extend_from_slice(&(i / 4).to_le_bytes());
        }
        let packed = rle_compress_chunk(&data);
        assert!(packed.len() < data.len());
        let unpacked = rle_decompress_chunk(&packed, data.len()).unwrap();
        assert_eq!(unpacked, data);
    }

    #[test]
    fn test_piz_huf_roundtrip() {
        let mut raw = Vec::new();
        for i in 0..1024u16 {
            raw.push((i % 29) * 7);
        }
        let comp = huf_compress(&raw).unwrap();
        let dec = huf_decompress(&comp, raw.len()).unwrap();
        assert_eq!(dec, raw);
    }

    #[test]
    fn test_piz_wav2d_roundtrip() {
        let mut plane14 = vec![0u16; 16 * 16];
        let mut plane16 = vec![0u16; 16 * 16];
        for (i, v) in plane14.iter_mut().enumerate() {
            *v = ((i * 73) & 0x1fff) as u16;
        }
        for (i, v) in plane16.iter_mut().enumerate() {
            *v = ((i * 1021) & 0xffff) as u16;
        }
        let orig14 = plane14.clone();
        wav2d_encode(&mut plane14, 16, 1, 16, 16, 0x1fff);
        wav2d_decode(&mut plane14, 16, 1, 16, 16, 0x1fff);
        assert_eq!(plane14, orig14);

        let orig16 = plane16.clone();
        wav2d_encode(&mut plane16, 16, 1, 16, 16, 0xffff);
        wav2d_decode(&mut plane16, 16, 1, 16, 16, 0xffff);
        assert_eq!(plane16, orig16);
    }
}
