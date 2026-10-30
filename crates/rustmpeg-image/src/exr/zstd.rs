// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! OpenEXR `ZSTD` (`13`) chunk compressor and decompressor, including
//! the OpenEXR 24-byte `zstd-exr` container, 16-bit/32-bit channel sorting,
//! per-row delta prediction, Blosc 2-byte/4-byte planar byte shuffle, and a
//! pure-Rust RFC 8878 Zstandard frame decompressor and compressor.
//!
//! Verified against `OpenEXRCore/internal_zstd.c`, `internal_zstd_shuffle.c`,
//! `internal_zstd_delta.c`, and RFC 8878 (`external/zstd/lib/decompress/`).

use super::ExrError;
use super::header::ExrChannelDesc;

/// 8-byte wire magic `b"zstd-exr"` (`8248453963162350458` LE = `0x7278_652d_6474_737a`).
pub const ZSTD_EXR_MAGIC: u64 = 8_248_453_963_162_350_458;
const ZSTD_EXR_HEADER_SIZE: usize = 24;
const ZSTD_EXR_FORMAT_V1: u32 = 1;
const ZSTD_EXR_FORMAT_V2: u32 = 2;
const ZSTD_EXR_FLAG_DELTA_AFTER_SORT: u32 = 1;

const ZSTD_FRAME_MAGIC: u32 = 0xFD2F_B528;
const ZSTD_BLOCKSIZE_MAX: usize = 128 * 1024;

/// Grid layout scenario for `ZSTD` (`OpenEXRCore/internal_zstd.c: exr_zstd_channel_grid_scenario_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZstdGridScenario<'a> {
    /// Deep sample-count table chunk: 1 synthetic 4-byte (`UINT`) channel with `region_width` samples/row.
    DeepSampleCountTable,
    /// Deep pixel data chunk: every channel on row `h` has `row_sample_counts[h]` samples.
    DeepPixels(&'a [usize]),
    /// Flat scanline or tile chunk: per-channel row sample count from `ch.samples_in_line(y, width)`.
    FlatScanOrTile,
}

// ============================================================================
// 1. OpenEXR ZSTD Pre/Post-Processing (`internal_zstd.c`, `shuffle.c`, `delta.c`)
// ============================================================================

fn build_sample_grid(
    scenario: ZstdGridScenario<'_>,
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
) -> (Vec<usize>, Vec<usize>) {
    match scenario {
        ZstdGridScenario::DeepSampleCountTable => {
            let bpe = vec![4usize];
            let grid = vec![region_width; num_lines];
            (bpe, grid)
        }
        ZstdGridScenario::DeepPixels(row_counts) => {
            let bpe: Vec<usize> = channels
                .iter()
                .map(|c| c.sample_type.bytes_per_sample())
                .collect();
            let nch = bpe.len();
            let mut grid = vec![0usize; num_lines * nch];
            for h in 0..num_lines {
                let cnt = row_counts.get(h).copied().unwrap_or(0);
                for i in 0..nch {
                    grid[h * nch + i] = cnt;
                }
            }
            (bpe, grid)
        }
        ZstdGridScenario::FlatScanOrTile => {
            let bpe: Vec<usize> = channels
                .iter()
                .map(|c| c.sample_type.bytes_per_sample())
                .collect();
            let nch = bpe.len();
            let mut grid = vec![0usize; num_lines * nch];
            for h in 0..num_lines {
                let y = y_start + (h as i32);
                for (i, ch) in channels.iter().enumerate() {
                    grid[h * nch + i] = ch.samples_in_line(y, region_width);
                }
            }
            (bpe, grid)
        }
    }
}

fn compute_sorting_lookup(
    grid: &[usize],
    height: usize,
    bpe: &[usize],
    sorting_lookup: &mut [usize],
) -> usize {
    let nch = bpe.len();
    let mut write_count = 0usize;
    for h in 0..height {
        for i in 0..nch {
            if bpe[i] == 2 {
                sorting_lookup[h * nch + i] = write_count;
                write_count += grid[h * nch + i] * 2;
            }
        }
    }
    let split_point = write_count;
    for h in 0..height {
        for i in 0..nch {
            if bpe[i] == 4 {
                sorting_lookup[h * nch + i] = write_count;
                write_count += grid[h * nch + i] * 4;
            }
        }
    }
    split_point
}

fn sort_2_4_byte_channels(
    src: &[u8],
    grid: &[usize],
    bpe: &[usize],
    forward: bool,
    height: usize,
    dst: &mut [u8],
) -> Result<usize, ExrError> {
    let nch = bpe.len();
    let mut sorting_lookup = vec![0usize; height * nch];
    let split_point = compute_sorting_lookup(grid, height, bpe, &mut sorting_lookup);

    let mut line_start_read = 0usize;
    for h in 0..height {
        let mut row_byte_off = 0usize;
        for i in 0..nch {
            let n_hi = grid[h * nch + i];
            let write_idx = sorting_lookup[h * nch + i];
            let read_idx = line_start_read + row_byte_off;
            let byte_size = n_hi * bpe[i];
            if byte_size > 0 {
                if forward {
                    if read_idx + byte_size > src.len() || write_idx + byte_size > dst.len() {
                        return Err(ExrError::CorruptChunk("ZSTD sort forward bounds overflow"));
                    }
                    dst[write_idx..write_idx + byte_size]
                        .copy_from_slice(&src[read_idx..read_idx + byte_size]);
                } else {
                    if write_idx + byte_size > src.len() || read_idx + byte_size > dst.len() {
                        return Err(ExrError::CorruptChunk("ZSTD sort inverse bounds overflow"));
                    }
                    dst[read_idx..read_idx + byte_size]
                        .copy_from_slice(&src[write_idx..write_idx + byte_size]);
                }
            }
            row_byte_off += byte_size;
        }
        line_start_read += row_byte_off;
    }
    Ok(split_point)
}

#[inline]
fn delta_encode_row_u16(p: &mut [u8], n: usize) {
    if n <= 1 {
        return;
    }
    for k in (1..n).rev() {
        let a = u16::from_le_bytes([p[k * 2], p[k * 2 + 1]]);
        let b = u16::from_le_bytes([p[(k - 1) * 2], p[(k - 1) * 2 + 1]]);
        let d = a.wrapping_sub(b).to_le_bytes();
        p[k * 2] = d[0];
        p[k * 2 + 1] = d[1];
    }
}

#[inline]
fn delta_decode_row_u16(p: &mut [u8], n: usize) {
    for k in 1..n {
        let a = u16::from_le_bytes([p[k * 2], p[k * 2 + 1]]);
        let b = u16::from_le_bytes([p[(k - 1) * 2], p[(k - 1) * 2 + 1]]);
        let s = a.wrapping_add(b).to_le_bytes();
        p[k * 2] = s[0];
        p[k * 2 + 1] = s[1];
    }
}

#[inline]
fn delta_encode_row_u32(p: &mut [u8], n: usize) {
    if n <= 1 {
        return;
    }
    for k in (1..n).rev() {
        let a = u32::from_le_bytes([p[k * 4], p[k * 4 + 1], p[k * 4 + 2], p[k * 4 + 3]]);
        let b = u32::from_le_bytes([
            p[(k - 1) * 4],
            p[(k - 1) * 4 + 1],
            p[(k - 1) * 4 + 2],
            p[(k - 1) * 4 + 3],
        ]);
        let d = a.wrapping_sub(b).to_le_bytes();
        p[k * 4..k * 4 + 4].copy_from_slice(&d);
    }
}

#[inline]
fn delta_decode_row_u32(p: &mut [u8], n: usize) {
    for k in 1..n {
        let a = u32::from_le_bytes([p[k * 4], p[k * 4 + 1], p[k * 4 + 2], p[k * 4 + 3]]);
        let b = u32::from_le_bytes([
            p[(k - 1) * 4],
            p[(k - 1) * 4 + 1],
            p[(k - 1) * 4 + 2],
            p[(k - 1) * 4 + 3],
        ]);
        let s = a.wrapping_add(b).to_le_bytes();
        p[k * 4..k * 4 + 4].copy_from_slice(&s);
    }
}

fn delta_encode_sorted_layout(
    buf: &mut [u8],
    split_point: usize,
    grid: &[usize],
    height: usize,
    bpe: &[usize],
) -> Result<(), ExrError> {
    let nch = bpe.len();
    let mut off = 0usize;
    for h in 0..height {
        for i in 0..nch {
            if bpe[i] != 2 {
                continue;
            }
            let n = grid[h * nch + i];
            if n > 0 {
                delta_encode_row_u16(&mut buf[off..off + n * 2], n);
            }
            off += n * 2;
        }
    }
    if off != split_point {
        return Err(ExrError::CorruptChunk("ZSTD delta encode split mismatch"));
    }
    for h in 0..height {
        for i in 0..nch {
            if bpe[i] != 4 {
                continue;
            }
            let n = grid[h * nch + i];
            if n > 0 {
                delta_encode_row_u32(&mut buf[off..off + n * 4], n);
            }
            off += n * 4;
        }
    }
    if off != buf.len() {
        return Err(ExrError::CorruptChunk("ZSTD delta encode total mismatch"));
    }
    Ok(())
}

fn delta_decode_sorted_layout(
    buf: &mut [u8],
    split_point: usize,
    grid: &[usize],
    height: usize,
    bpe: &[usize],
) -> Result<(), ExrError> {
    let nch = bpe.len();
    let mut off = 0usize;
    for h in 0..height {
        for i in 0..nch {
            if bpe[i] != 2 {
                continue;
            }
            let n = grid[h * nch + i];
            if n > 0 {
                delta_decode_row_u16(&mut buf[off..off + n * 2], n);
            }
            off += n * 2;
        }
    }
    if off != split_point {
        return Err(ExrError::CorruptChunk("ZSTD delta decode split mismatch"));
    }
    for h in 0..height {
        for i in 0..nch {
            if bpe[i] != 4 {
                continue;
            }
            let n = grid[h * nch + i];
            if n > 0 {
                delta_decode_row_u32(&mut buf[off..off + n * 4], n);
            }
            off += n * 4;
        }
    }
    if off != buf.len() {
        return Err(ExrError::CorruptChunk("ZSTD delta decode total mismatch"));
    }
    Ok(())
}

fn shuffle_encode_2(src: &[u8], dst: &mut [u8]) {
    let elements = src.len() / 2;
    for i in 0..elements {
        dst[i] = src[i * 2];
        dst[elements + i] = src[i * 2 + 1];
    }
}

fn shuffle_encode_4(src: &[u8], dst: &mut [u8]) {
    let elements = src.len() / 4;
    for i in 0..elements {
        dst[i] = src[i * 4];
        dst[elements + i] = src[i * 4 + 1];
        dst[elements * 2 + i] = src[i * 4 + 2];
        dst[elements * 3 + i] = src[i * 4 + 3];
    }
}

fn shuffle_decode_2(src: &[u8], dst: &mut [u8]) {
    let elements = src.len() / 2;
    for i in 0..elements {
        dst[i * 2] = src[i];
        dst[i * 2 + 1] = src[elements + i];
    }
}

fn shuffle_decode_4(src: &[u8], dst: &mut [u8]) {
    let elements = src.len() / 4;
    for i in 0..elements {
        dst[i * 4] = src[i];
        dst[i * 4 + 1] = src[elements + i];
        dst[i * 4 + 2] = src[elements * 2 + i];
        dst[i * 4 + 3] = src[elements * 3 + i];
    }
}

fn segment_layout(split: usize, total: usize) -> [(usize, usize); 2] {
    let rest = total - split;
    if split > 0 && rest > 0 {
        [(split, 2), (rest, 4)]
    } else {
        [(total, if split == 0 { 4 } else { 2 }), (0, 0)]
    }
}

/// Decompresses an OpenEXR `ZSTD` (`13`) chunk payload.
pub fn zstd_decompress_chunk(
    packed: &[u8],
    expected_uncompressed_size: usize,
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
    scenario: ZstdGridScenario<'_>,
) -> Result<Vec<u8>, ExrError> {
    if expected_uncompressed_size == 0 {
        return Ok(Vec::new());
    }
    if packed.len() < ZSTD_EXR_HEADER_SIZE {
        if packed.len() == expected_uncompressed_size {
            return Ok(packed.to_vec());
        }
        return Err(ExrError::CorruptChunk("ZSTD chunk shorter than 24 bytes"));
    }
    let magic = u64::from_le_bytes([
        packed[0], packed[1], packed[2], packed[3], packed[4], packed[5], packed[6], packed[7],
    ]);
    if magic != ZSTD_EXR_MAGIC {
        if packed.len() == expected_uncompressed_size {
            return Ok(packed.to_vec());
        }
        return Err(ExrError::CorruptChunk("ZSTD chunk invalid zstd-exr magic"));
    }

    let format = u32::from_le_bytes([packed[8], packed[9], packed[10], packed[11]]);
    let flags = u32::from_le_bytes([packed[12], packed[13], packed[14], packed[15]]);
    let zstd_payload_size = u64::from_le_bytes([
        packed[16], packed[17], packed[18], packed[19], packed[20], packed[21], packed[22],
        packed[23],
    ]) as usize;

    let apply_delta = match format {
        ZSTD_EXR_FORMAT_V1 => {
            if flags != 0 {
                return Err(ExrError::CorruptChunk("ZSTD v1 non-zero flags"));
            }
            false
        }
        ZSTD_EXR_FORMAT_V2 => {
            if (flags & !ZSTD_EXR_FLAG_DELTA_AFTER_SORT) != 0 {
                return Err(ExrError::CorruptChunk("ZSTD v2 unknown flag bits"));
            }
            (flags & ZSTD_EXR_FLAG_DELTA_AFTER_SORT) != 0
        }
        _ => return Err(ExrError::CorruptChunk("ZSTD unknown wire format version")),
    };

    if ZSTD_EXR_HEADER_SIZE.saturating_add(zstd_payload_size) != packed.len() {
        return Err(ExrError::CorruptChunk(
            "ZSTD payload size does not match chunk size",
        ));
    }

    let max_inner_size = expected_uncompressed_size
        .checked_add(16)
        .ok_or(ExrError::DimensionsTooLarge)?;
    let inner = zstd_decompress_frame(&packed[ZSTD_EXR_HEADER_SIZE..], max_inner_size)?;

    let (bpe, grid) = build_sample_grid(scenario, channels, region_width, y_start, num_lines);
    let mut dummy_lookup = vec![0usize; num_lines * bpe.len()];
    let split = compute_sorting_lookup(&grid, num_lines, &bpe, &mut dummy_lookup);
    if split > expected_uncompressed_size {
        return Err(ExrError::CorruptChunk(
            "ZSTD split exceeds uncompressed size",
        ));
    }

    let segs = segment_layout(split, expected_uncompressed_size);
    let mut sorted_buf = vec![0u8; expected_uncompressed_size];
    let mut q = 0usize;
    let mut tgt_off = 0usize;

    for &(seg_len, el_bytes) in &segs {
        if seg_len == 0 {
            continue;
        }
        if q + 8 > inner.len() {
            return Err(ExrError::CorruptChunk(
                "ZSTD inner segment header truncated",
            ));
        }
        let ilen = u64::from_le_bytes([
            inner[q],
            inner[q + 1],
            inner[q + 2],
            inner[q + 3],
            inner[q + 4],
            inner[q + 5],
            inner[q + 6],
            inner[q + 7],
        ]) as usize;
        q += 8;
        if ilen != seg_len || q + ilen > inner.len() || !ilen.is_multiple_of(el_bytes) {
            return Err(ExrError::CorruptChunk("ZSTD inner segment length mismatch"));
        }
        let dst_slice = &mut sorted_buf[tgt_off..tgt_off + ilen];
        let src_slice = &inner[q..q + ilen];
        if el_bytes == 4 {
            shuffle_decode_4(src_slice, dst_slice);
        } else {
            shuffle_decode_2(src_slice, dst_slice);
        }
        q += ilen;
        tgt_off += ilen;
    }
    if q != inner.len() || tgt_off != expected_uncompressed_size {
        return Err(ExrError::CorruptChunk(
            "ZSTD inner stream trailing bytes or length mismatch",
        ));
    }

    if apply_delta {
        delta_decode_sorted_layout(&mut sorted_buf, split, &grid, num_lines, &bpe)?;
    }

    let mut out = vec![0u8; expected_uncompressed_size];
    sort_2_4_byte_channels(&sorted_buf, &grid, &bpe, false, num_lines, &mut out)?;
    Ok(out)
}

/// Compresses an uncompressed chunk buffer into an OpenEXR `ZSTD` (`13`) payload.
pub fn zstd_compress_chunk(
    uncompressed: &[u8],
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
    scenario: ZstdGridScenario<'_>,
) -> Result<Vec<u8>, ExrError> {
    if uncompressed.is_empty() {
        return Ok(Vec::new());
    }
    let (bpe, grid) = build_sample_grid(scenario, channels, region_width, y_start, num_lines);
    let mut sorted_buf = vec![0u8; uncompressed.len()];
    let split =
        sort_2_4_byte_channels(uncompressed, &grid, &bpe, true, num_lines, &mut sorted_buf)?;

    let (hdr_format, hdr_flags, apply_delta) = match scenario {
        ZstdGridScenario::FlatScanOrTile => {
            (ZSTD_EXR_FORMAT_V2, ZSTD_EXR_FLAG_DELTA_AFTER_SORT, true)
        }
        ZstdGridScenario::DeepSampleCountTable | ZstdGridScenario::DeepPixels(_) => {
            (ZSTD_EXR_FORMAT_V1, 0, false)
        }
    };

    if apply_delta {
        delta_encode_sorted_layout(&mut sorted_buf, split, &grid, num_lines, &bpe)?;
    }

    let segs = segment_layout(split, uncompressed.len());
    let mut inner = Vec::with_capacity(uncompressed.len() + 16);
    let mut seg_off = 0usize;
    for &(seg_len, el_bytes) in &segs {
        if seg_len == 0 {
            continue;
        }
        inner.extend_from_slice(&(seg_len as u64).to_le_bytes());
        let dst_start = inner.len();
        inner.resize(dst_start + seg_len, 0);
        let src_slice = &sorted_buf[seg_off..seg_off + seg_len];
        if el_bytes == 4 {
            shuffle_encode_4(src_slice, &mut inner[dst_start..dst_start + seg_len]);
        } else {
            shuffle_encode_2(src_slice, &mut inner[dst_start..dst_start + seg_len]);
        }
        seg_off += seg_len;
    }

    let zstd_frame = zstd_compress_frame(&inner);
    let total_size = ZSTD_EXR_HEADER_SIZE + zstd_frame.len();
    if total_size >= uncompressed.len() {
        return Ok(uncompressed.to_vec());
    }

    let mut out = Vec::with_capacity(total_size);
    out.extend_from_slice(&ZSTD_EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&hdr_format.to_le_bytes());
    out.extend_from_slice(&hdr_flags.to_le_bytes());
    out.extend_from_slice(&(zstd_frame.len() as u64).to_le_bytes());
    out.extend_from_slice(&zstd_frame);
    Ok(out)
}

// ============================================================================
// 2. Pure-Rust RFC 8878 Zstandard Frame Decompressor & Compressor
// ============================================================================

const LL_BITS: [u8; 36] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];

const LL_BASE: [u32; 36] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64,
    0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000,
];

const LL_DEFAULT_NORM: [i16; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];
const LL_DEFAULT_NORM_LOG: u32 = 6;

const ML_BITS: [u8; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 0x83, 0x103, 0x203,
    0x403, 0x803, 0x1003, 0x2003, 0x4003, 0x8003, 0x10003,
];

const ML_DEFAULT_NORM: [i16; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];
const ML_DEFAULT_NORM_LOG: u32 = 6;

const OF_BITS: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];

const OF_BASE: [u32; 32] = [
    0, 1, 1, 5, 0xD, 0x1D, 0x3D, 0x7D, 0xFD, 0x1FD, 0x3FD, 0x7FD, 0xFFD, 0x1FFD, 0x3FFD, 0x7FFD,
    0xFFFD, 0x1FFFD, 0x3FFFD, 0x7FFFD, 0xFFFFD, 0x1FFFFD, 0x3FFFFD, 0x7FFFFD, 0xFFFFFD, 0x1FFFFFD,
    0x3FFFFFD, 0x7FFFFFD, 0xFFFFFFD, 0x1FFFFFFD, 0x3FFFFFFD, 0x7FFFFFFD,
];

const OF_DEFAULT_NORM: [i16; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];
const OF_DEFAULT_NORM_LOG: u32 = 5;

#[inline]
fn highbit32(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}

#[inline]
fn fse_tablestep(table_size: usize) -> usize {
    (table_size >> 1) + (table_size >> 3) + 3
}

/// Backward bitstream reader (`BIT_DStream_t` in `external/zstd/lib/common/bitstream.h`).
struct BitDStream<'a> {
    src: &'a [u8],
    ptr: usize,
    bit_container: u64,
    bits_consumed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DStreamStatus {
    Unfinished,
    EndOfBuffer,
    Completed,
    Overflow,
}

impl<'a> BitDStream<'a> {
    fn init(src: &'a [u8]) -> Result<Self, ExrError> {
        if src.is_empty() {
            return Err(ExrError::CorruptChunk("ZSTD empty backward bitstream"));
        }
        let last_byte = src[src.len() - 1];
        if last_byte == 0 {
            return Err(ExrError::CorruptChunk(
                "ZSTD backward bitstream missing endMark",
            ));
        }
        let mut bits_consumed = 8 - highbit32(u32::from(last_byte));
        let (ptr, bit_container) = if src.len() >= 8 {
            let p = src.len() - 8;
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&src[p..p + 8]);
            (p, u64::from_le_bytes(bytes))
        } else {
            let mut bytes = [0u8; 8];
            bytes[..src.len()].copy_from_slice(src);
            bits_consumed += ((8 - src.len()) as u32) * 8;
            (0, u64::from_le_bytes(bytes))
        };
        Ok(Self {
            src,
            ptr,
            bit_container,
            bits_consumed,
        })
    }

    #[inline]
    fn look_bits(&self, nb_bits: u32) -> u64 {
        if nb_bits == 0 {
            return 0;
        }
        let start = 64u32.wrapping_sub(self.bits_consumed).wrapping_sub(nb_bits) & 63;
        (self.bit_container >> start) & ((1u64 << nb_bits) - 1)
    }

    #[inline]
    fn look_bits_fast(&self, nb_bits: u32) -> u64 {
        debug_assert!(nb_bits >= 1);
        (self.bit_container << (self.bits_consumed & 63)) >> ((64 - nb_bits) & 63)
    }

    #[inline]
    fn skip_bits(&mut self, nb_bits: u32) {
        self.bits_consumed += nb_bits;
    }

    #[inline]
    fn read_bits(&mut self, nb_bits: u32) -> u64 {
        let v = self.look_bits(nb_bits);
        self.skip_bits(nb_bits);
        v
    }

    #[inline]
    fn read_bits_fast(&mut self, nb_bits: u32) -> u64 {
        let v = self.look_bits_fast(nb_bits);
        self.skip_bits(nb_bits);
        v
    }

    #[inline]
    fn reload(&mut self) -> DStreamStatus {
        if self.bits_consumed > 64 {
            return DStreamStatus::Overflow;
        }
        if self.ptr >= 8 {
            self.ptr -= (self.bits_consumed >> 3) as usize;
            self.bits_consumed &= 7;
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&self.src[self.ptr..self.ptr + 8]);
            self.bit_container = u64::from_le_bytes(bytes);
            return DStreamStatus::Unfinished;
        }
        if self.ptr == 0 {
            if self.bits_consumed < 64 {
                return DStreamStatus::EndOfBuffer;
            }
            return DStreamStatus::Completed;
        }
        let mut nb_bytes = (self.bits_consumed >> 3) as usize;
        let mut res = DStreamStatus::Unfinished;
        if nb_bytes > self.ptr {
            nb_bytes = self.ptr;
            res = DStreamStatus::EndOfBuffer;
        }
        self.ptr -= nb_bytes;
        self.bits_consumed -= (nb_bytes as u32) * 8;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.src[self.ptr..self.ptr + 8]);
        self.bit_container = u64::from_le_bytes(bytes);
        res
    }

    #[inline]
    fn is_end(&self) -> bool {
        self.ptr == 0 && self.bits_consumed == 64
    }
}

/// Read normalized FSE distribution table (`FSE_readNCount` in `entropy_common.c`).
fn fse_read_ncount(
    header_buf: &[u8],
    max_sv_in: usize,
    max_table_log: u32,
) -> Result<(Vec<i16>, usize, u32, usize), ExrError> {
    if header_buf.is_empty() {
        return Err(ExrError::CorruptChunk("ZSTD empty FSE NCount header"));
    }
    let mut padded = [0u8; 8];
    let slice: &[u8] = if header_buf.len() < 8 {
        padded[..header_buf.len()].copy_from_slice(header_buf);
        &padded
    } else {
        header_buf
    };

    let read_le32 = |p: usize| -> u32 {
        u32::from_le_bytes([slice[p], slice[p + 1], slice[p + 2], slice[p + 3]])
    };

    let mut norm = vec![0i16; max_sv_in + 1];
    let mut ip = 0usize;
    let iend = slice.len();
    let mut bit_stream = read_le32(ip);
    let mut nb_bits = (bit_stream & 0xF) + 5;
    if nb_bits > max_table_log.max(15) || nb_bits > max_table_log {
        return Err(ExrError::CorruptChunk("ZSTD FSE tableLog too large"));
    }
    bit_stream >>= 4;
    let mut bit_count = 4i32;
    let table_log = nb_bits;
    let mut remaining = (1i32 << nb_bits) + 1;
    let mut threshold = 1i32 << nb_bits;
    nb_bits += 1;

    let mut charnum = 0usize;
    let max_sv1 = max_sv_in + 1;
    let mut previous0 = false;

    loop {
        if previous0 {
            let mut repeats = ((!bit_stream) | 0x8000_0000).trailing_zeros() >> 1;
            while repeats >= 12 {
                charnum += 3 * 12;
                if ip + 7 <= iend {
                    ip += 3;
                } else {
                    bit_count -= (8 * ((iend as isize) - 7 - (ip as isize))) as i32;
                    bit_count &= 31;
                    ip = iend - 4;
                }
                bit_stream = read_le32(ip) >> (bit_count as u32);
                repeats = ((!bit_stream) | 0x8000_0000).trailing_zeros() >> 1;
            }
            charnum += 3 * (repeats as usize);
            bit_stream >>= 2 * repeats;
            bit_count += 2 * (repeats as i32);
            charnum += (bit_stream & 3) as usize;
            bit_count += 2;
            if charnum >= max_sv1 {
                break;
            }
            if (ip + 7 <= iend) || (ip + ((bit_count >> 3) as usize) <= iend - 4) {
                ip += (bit_count >> 3) as usize;
                bit_count &= 7;
            } else {
                bit_count -= (8 * ((iend as isize) - 4 - (ip as isize))) as i32;
                bit_count &= 31;
                ip = iend - 4;
            }
            bit_stream = read_le32(ip) >> (bit_count as u32);
        }

        let max = (2 * threshold - 1) - remaining;
        let mut count: i32;
        if ((bit_stream as i32) & (threshold - 1)) < max {
            count = (bit_stream as i32) & (threshold - 1);
            bit_count += (nb_bits as i32) - 1;
        } else {
            count = (bit_stream as i32) & (2 * threshold - 1);
            if count >= threshold {
                count -= max;
            }
            bit_count += nb_bits as i32;
        }
        count -= 1;
        if count >= 0 {
            remaining -= count;
        } else {
            remaining += count;
        }
        if charnum >= norm.len() {
            return Err(ExrError::CorruptChunk("ZSTD FSE symbol count overflow"));
        }
        norm[charnum] = count as i16;
        charnum += 1;
        previous0 = count == 0;

        if remaining < threshold {
            if remaining <= 1 {
                break;
            }
            nb_bits = highbit32(remaining as u32) + 1;
            threshold = 1i32 << (nb_bits - 1);
        }
        if charnum >= max_sv1 {
            break;
        }
        if (ip + 7 <= iend) || (ip + ((bit_count >> 3) as usize) <= iend - 4) {
            ip += (bit_count >> 3) as usize;
            bit_count &= 7;
        } else {
            bit_count -= (8 * ((iend as isize) - 4 - (ip as isize))) as i32;
            bit_count &= 31;
            ip = iend - 4;
        }
        bit_stream = read_le32(ip) >> (bit_count as u32);
    }

    if remaining != 1 || charnum > max_sv1 || bit_count > 32 {
        return Err(ExrError::CorruptChunk("ZSTD corrupt FSE NCount table"));
    }
    let bytes_read = ip + (((bit_count + 7) >> 3) as usize);
    if bytes_read > header_buf.len() {
        return Err(ExrError::CorruptChunk("ZSTD FSE NCount read past buffer"));
    }
    Ok((norm, charnum - 1, table_log, bytes_read))
}

#[derive(Debug, Clone, Copy, Default)]
struct FseDecodeCell {
    new_state: u16,
    symbol: u8,
    nb_bits: u8,
}

fn fse_build_byte_dtable(
    norm: &[i16],
    max_sv: usize,
    table_log: u32,
) -> Result<Vec<FseDecodeCell>, ExrError> {
    let table_size = 1usize << table_log;
    let table_mask = table_size - 1;
    let step = fse_tablestep(table_size);
    let mut table = vec![FseDecodeCell::default(); table_size];
    let mut symbol_next = vec![0u16; max_sv + 1];
    let mut high_threshold = table_size - 1;

    for s in 0..=max_sv {
        if norm[s] == -1 {
            table[high_threshold].symbol = s as u8;
            high_threshold = high_threshold.wrapping_sub(1);
            symbol_next[s] = 1;
        } else if norm[s] >= 0 {
            symbol_next[s] = norm[s] as u16;
        } else {
            return Err(ExrError::CorruptChunk("ZSTD invalid FSE norm count"));
        }
    }

    let mut position = 0usize;
    for (s, &n) in norm.iter().enumerate().take(max_sv + 1) {
        if n > 0 {
            for _ in 0..n {
                table[position].symbol = s as u8;
                position = (position + step) & table_mask;
                while position > high_threshold {
                    position = (position + step) & table_mask;
                }
            }
        }
    }
    if position != 0 {
        return Err(ExrError::CorruptChunk(
            "ZSTD FSE symbol spread did not reach 0",
        ));
    }

    for cell in &mut table {
        let sym = usize::from(cell.symbol);
        let next_state = u32::from(symbol_next[sym]);
        symbol_next[sym] += 1;
        let nb_bits = (table_log - highbit32(next_state)) as u8;
        cell.nb_bits = nb_bits;
        cell.new_state = ((next_state << nb_bits) - (table_size as u32)) as u16;
    }
    Ok(table)
}

fn fse_decompress_bytes(
    src: &[u8],
    max_dst_size: usize,
    max_log: u32,
) -> Result<Vec<u8>, ExrError> {
    let (norm, max_sv, table_log, header_size) = fse_read_ncount(src, 255, max_log)?;
    let dtable = fse_build_byte_dtable(&norm, max_sv, table_log)?;
    let mut bit_d = BitDStream::init(&src[header_size..])?;

    let mut state1 = bit_d.read_bits(table_log) as usize;
    let mut state2 = bit_d.read_bits(table_log) as usize;
    if bit_d.reload() == DStreamStatus::Overflow {
        return Err(ExrError::CorruptChunk("ZSTD FSE initial reload overflow"));
    }

    let decode_sym = |state: &mut usize, bd: &mut BitDStream<'_>| -> u8 {
        let info = dtable[*state];
        let low = bd.read_bits(u32::from(info.nb_bits)) as usize;
        *state = usize::from(info.new_state) + low;
        info.symbol
    };

    let mut out = Vec::with_capacity(max_dst_size);
    while bit_d.reload() == DStreamStatus::Unfinished && out.len() + 4 <= max_dst_size {
        out.push(decode_sym(&mut state1, &mut bit_d));
        out.push(decode_sym(&mut state2, &mut bit_d));
        out.push(decode_sym(&mut state1, &mut bit_d));
        out.push(decode_sym(&mut state2, &mut bit_d));
    }

    loop {
        if out.len() + 2 > max_dst_size {
            return Err(ExrError::CorruptChunk("ZSTD FSE output overflow"));
        }
        out.push(decode_sym(&mut state1, &mut bit_d));
        if bit_d.reload() == DStreamStatus::Overflow {
            out.push(decode_sym(&mut state2, &mut bit_d));
            break;
        }
        if out.len() + 2 > max_dst_size {
            return Err(ExrError::CorruptChunk("ZSTD FSE output overflow"));
        }
        out.push(decode_sym(&mut state2, &mut bit_d));
        if bit_d.reload() == DStreamStatus::Overflow {
            out.push(decode_sym(&mut state1, &mut bit_d));
            break;
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, Default)]
struct HufDEltX1 {
    byte: u8,
    nb_bits: u8,
}

#[derive(Debug, Clone)]
struct HufDTable {
    table_log: u32,
    entries: Vec<HufDEltX1>,
}

impl HufDTable {
    fn read_x1(src: &[u8]) -> Result<(Self, usize), ExrError> {
        if src.is_empty() {
            return Err(ExrError::CorruptChunk("ZSTD empty HUF header"));
        }
        let i_size_byte = usize::from(src[0]);
        let mut huff_weight = [0u8; 256];
        let (o_size, i_size) = if i_size_byte >= 128 {
            let o_sz = i_size_byte - 127;
            let i_sz = o_sz.div_ceil(2);
            if i_sz + 1 > src.len() || o_sz >= 256 {
                return Err(ExrError::CorruptChunk("ZSTD direct HUF weights truncated"));
            }
            for n in (0..o_sz).step_by(2) {
                let b = src[1 + n / 2];
                huff_weight[n] = b >> 4;
                if n + 1 < o_sz {
                    huff_weight[n + 1] = b & 15;
                }
            }
            (o_sz, i_sz)
        } else {
            let i_sz = i_size_byte;
            if i_sz + 1 > src.len() {
                return Err(ExrError::CorruptChunk("ZSTD FSE HUF weights truncated"));
            }
            let decoded = fse_decompress_bytes(&src[1..1 + i_sz], 255, 6)?;
            if decoded.len() >= 256 {
                return Err(ExrError::CorruptChunk("ZSTD HUF decoded weights >= 256"));
            }
            huff_weight[..decoded.len()].copy_from_slice(&decoded);
            (decoded.len(), i_sz)
        };

        let mut rank_val = [0u32; 13];
        let mut weight_total = 0u32;
        for &w in &huff_weight[..o_size] {
            if w > 12 {
                return Err(ExrError::CorruptChunk("ZSTD HUF weight > 12"));
            }
            rank_val[usize::from(w)] += 1;
            weight_total += (1u32 << w) >> 1;
        }
        if weight_total == 0 {
            return Err(ExrError::CorruptChunk("ZSTD HUF weight_total == 0"));
        }
        let table_log = highbit32(weight_total) + 1;
        if table_log > 12 {
            return Err(ExrError::CorruptChunk("ZSTD HUF tableLog > 12"));
        }
        let total = 1u32 << table_log;
        let rest = total - weight_total;
        let last_weight = highbit32(rest) + 1;
        if (1u32 << (last_weight - 1)) != rest {
            return Err(ExrError::CorruptChunk(
                "ZSTD HUF last weight not power of 2",
            ));
        }
        huff_weight[o_size] = last_weight as u8;
        rank_val[last_weight as usize] += 1;
        if rank_val[1] < 2 || (rank_val[1] & 1) != 0 {
            return Err(ExrError::CorruptChunk("ZSTD HUF invalid rank_val[1]"));
        }
        let nb_symbols = o_size + 1;

        let mut rank_start = [0u32; 14];
        let mut next_rank_start = 0u32;
        for n in 0..=(table_log as usize) {
            let curr = next_rank_start;
            next_rank_start += rank_val[n];
            rank_start[n] = curr;
        }
        let mut symbols = [0u8; 256];
        for (n, &w) in huff_weight[..nb_symbols].iter().enumerate() {
            let idx = rank_start[usize::from(w)] as usize;
            rank_start[usize::from(w)] += 1;
            symbols[idx] = n as u8;
        }

        let mut entries = vec![HufDEltX1::default(); 1usize << table_log];
        let mut sym_idx = rank_val[0] as usize;
        let mut u_start = 0usize;
        for (w, &rv) in rank_val
            .iter()
            .enumerate()
            .take(table_log as usize + 1)
            .skip(1)
        {
            let sym_count = rv as usize;
            let length = (1usize << w) >> 1;
            let nb_bits = (table_log as u8) + 1 - (w as u8);
            for s in 0..sym_count {
                let d = HufDEltX1 {
                    byte: symbols[sym_idx + s],
                    nb_bits,
                };
                entries[u_start..u_start + length].fill(d);
                u_start += length;
            }
            sym_idx += sym_count;
        }

        Ok((Self { table_log, entries }, i_size + 1))
    }

    fn decode_stream_x1(&self, c_src: &[u8], dst: &mut [u8]) -> Result<(), ExrError> {
        let mut bit_d = BitDStream::init(c_src)?;
        let dt_log = self.table_log;
        let mut p = 0usize;
        let p_end = dst.len();

        let decode_one = |bd: &mut BitDStream<'_>| -> u8 {
            let val = bd.look_bits_fast(dt_log) as usize;
            let elt = self.entries[val];
            bd.skip_bits(u32::from(elt.nb_bits));
            elt.byte
        };

        if p_end > 3 {
            while bit_d.reload() == DStreamStatus::Unfinished && p < p_end - 3 {
                dst[p] = decode_one(&mut bit_d);
                dst[p + 1] = decode_one(&mut bit_d);
                dst[p + 2] = decode_one(&mut bit_d);
                dst[p + 3] = decode_one(&mut bit_d);
                p += 4;
            }
        } else {
            let _ = bit_d.reload();
        }
        while p < p_end {
            dst[p] = decode_one(&mut bit_d);
            p += 1;
        }
        if !bit_d.is_end() {
            return Err(ExrError::CorruptChunk(
                "ZSTD HUF 1X1 stream did not end cleanly",
            ));
        }
        Ok(())
    }

    fn decode_stream_4x1(&self, c_src: &[u8], dst: &mut [u8]) -> Result<(), ExrError> {
        if c_src.len() < 10 || dst.len() < 6 {
            return Err(ExrError::CorruptChunk("ZSTD HUF 4X1 stream too short"));
        }
        let len1 = u16::from_le_bytes([c_src[0], c_src[1]]) as usize;
        let len2 = u16::from_le_bytes([c_src[2], c_src[3]]) as usize;
        let len3 = u16::from_le_bytes([c_src[4], c_src[5]]) as usize;
        let start1 = 6usize;
        let start2 = start1 + len1;
        let start3 = start2 + len2;
        let start4 = start3 + len3;
        if start4 > c_src.len() {
            return Err(ExrError::CorruptChunk("ZSTD HUF 4X1 jump table overflow"));
        }
        let seg_size = dst.len().div_ceil(4);
        let (s1, rest) = dst.split_at_mut(seg_size);
        let (s2, rest) = rest.split_at_mut(seg_size);
        let (s3, s4) = rest.split_at_mut(seg_size);
        self.decode_stream_x1(&c_src[start1..start2], s1)?;
        self.decode_stream_x1(&c_src[start2..start3], s2)?;
        self.decode_stream_x1(&c_src[start3..start4], s3)?;
        self.decode_stream_x1(&c_src[start4..], s4)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SeqSymbol {
    next_state: u16,
    nb_additional_bits: u8,
    nb_bits: u8,
    base_value: u32,
}

#[derive(Debug, Clone)]
struct SeqDTable {
    table_log: u32,
    entries: Vec<SeqSymbol>,
}

fn build_seq_fse_table(
    norm: &[i16],
    max_sv: usize,
    base_value: &[u32],
    nb_additional_bits: &[u8],
    table_log: u32,
) -> Result<SeqDTable, ExrError> {
    let table_size = 1usize << table_log;
    let table_mask = table_size - 1;
    let step = fse_tablestep(table_size);
    let mut entries = vec![SeqSymbol::default(); table_size];
    let mut symbol_next = vec![0u16; max_sv + 1];
    let mut high_threshold = table_size - 1;

    for s in 0..=max_sv {
        if norm[s] == -1 {
            entries[high_threshold].base_value = s as u32;
            high_threshold = high_threshold.wrapping_sub(1);
            symbol_next[s] = 1;
        } else if norm[s] >= 0 {
            symbol_next[s] = norm[s] as u16;
        } else {
            return Err(ExrError::CorruptChunk("ZSTD invalid sequence FSE norm"));
        }
    }

    let mut position = 0usize;
    for (s, &n) in norm.iter().enumerate().take(max_sv + 1) {
        if n > 0 {
            for _ in 0..n {
                entries[position].base_value = s as u32;
                position = (position + step) & table_mask;
                while position > high_threshold {
                    position = (position + step) & table_mask;
                }
            }
        }
    }
    if position != 0 {
        return Err(ExrError::CorruptChunk(
            "ZSTD sequence FSE spread did not reach 0",
        ));
    }

    for cell in &mut entries {
        let sym = cell.base_value as usize;
        let next_state = u32::from(symbol_next[sym]);
        symbol_next[sym] += 1;
        let nb_bits = (table_log - highbit32(next_state)) as u8;
        cell.nb_bits = nb_bits;
        cell.next_state = ((next_state << nb_bits) - (table_size as u32)) as u16;
        cell.nb_additional_bits = nb_additional_bits[sym];
        cell.base_value = base_value[sym];
    }

    Ok(SeqDTable { table_log, entries })
}

fn build_seq_rle_table(base_value: u32, nb_add_bits: u8) -> SeqDTable {
    SeqDTable {
        table_log: 0,
        entries: vec![SeqSymbol {
            next_state: 0,
            nb_additional_bits: nb_add_bits,
            nb_bits: 0,
            base_value,
        }],
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_seq_table(
    mode: u8,
    src: &[u8],
    max_sv: usize,
    max_log: u32,
    base_val: &[u32],
    nb_add_bits: &[u8],
    default_norm: &[i16],
    default_log: u32,
    prev_table: &Option<SeqDTable>,
) -> Result<(SeqDTable, usize), ExrError> {
    match mode {
        0 => {
            // Predefined_Mode
            let tbl = build_seq_fse_table(
                default_norm,
                default_norm.len() - 1,
                base_val,
                nb_add_bits,
                default_log,
            )?;
            Ok((tbl, 0))
        }
        1 => {
            // RLE_Mode
            if src.is_empty() || usize::from(src[0]) > max_sv {
                return Err(ExrError::CorruptChunk("ZSTD RLE sequence symbol invalid"));
            }
            let sym = usize::from(src[0]);
            Ok((build_seq_rle_table(base_val[sym], nb_add_bits[sym]), 1))
        }
        2 => {
            // FSE_Compressed_Mode
            let (norm, actual_max_sv, table_log, bytes_read) =
                fse_read_ncount(src, max_sv, max_log)?;
            let tbl = build_seq_fse_table(&norm, actual_max_sv, base_val, nb_add_bits, table_log)?;
            Ok((tbl, bytes_read))
        }
        3 => {
            // Repeat_Mode
            let tbl = prev_table
                .as_ref()
                .ok_or(ExrError::CorruptChunk("ZSTD Repeat_Mode without table"))?
                .clone();
            Ok((tbl, 0))
        }
        _ => Err(ExrError::CorruptChunk("ZSTD invalid sequence table mode")),
    }
}

/// Decompresses an RFC 8878 Zstandard frame into at most `max_dst_size` bytes.
pub fn zstd_decompress_frame(src: &[u8], max_dst_size: usize) -> Result<Vec<u8>, ExrError> {
    if src.len() < 5 {
        return Err(ExrError::CorruptChunk("ZSTD frame shorter than 5 bytes"));
    }
    let magic = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
    if magic != ZSTD_FRAME_MAGIC {
        return Err(ExrError::CorruptChunk("ZSTD frame invalid magic"));
    }

    let fhd = src[4];
    if (fhd & 0x08) != 0 {
        return Err(ExrError::CorruptChunk("ZSTD frame reserved bit set"));
    }
    let dict_id_flag = fhd & 3;
    let checksum_flag = (fhd >> 2) & 1;
    let single_segment = (fhd >> 5) & 1;
    let fcs_flag = (fhd >> 6) & 3;

    let mut pos = 5usize;
    if single_segment == 0 {
        if pos >= src.len() {
            return Err(ExrError::CorruptChunk("ZSTD frame window byte truncated"));
        }
        pos += 1;
    }
    let dict_id_bytes = match dict_id_flag {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => 4,
    };
    if pos + dict_id_bytes > src.len() {
        return Err(ExrError::CorruptChunk("ZSTD frame dictID truncated"));
    }
    for &b in &src[pos..pos + dict_id_bytes] {
        if b != 0 {
            return Err(ExrError::CorruptChunk(
                "ZSTD external dictionary not allowed",
            ));
        }
    }
    pos += dict_id_bytes;

    let fcs_bytes = match fcs_flag {
        0 => usize::from(single_segment != 0),
        1 => 2,
        2 => 4,
        _ => 8,
    };
    if pos + fcs_bytes > src.len() {
        return Err(ExrError::CorruptChunk("ZSTD frame FCS truncated"));
    }
    let declared_fcs: Option<usize> = match fcs_bytes {
        0 => None,
        1 => Some(usize::from(src[pos])),
        2 => Some((u16::from_le_bytes([src[pos], src[pos + 1]]) as usize) + 256),
        4 => {
            Some(u32::from_le_bytes([src[pos], src[pos + 1], src[pos + 2], src[pos + 3]]) as usize)
        }
        8 => {
            let v = u64::from_le_bytes([
                src[pos],
                src[pos + 1],
                src[pos + 2],
                src[pos + 3],
                src[pos + 4],
                src[pos + 5],
                src[pos + 6],
                src[pos + 7],
            ]);
            Some(usize::try_from(v).map_err(|_| ExrError::DimensionsTooLarge)?)
        }
        _ => unreachable!(),
    };
    pos += fcs_bytes;

    if let Some(fcs) = declared_fcs
        && fcs > max_dst_size
    {
        return Err(ExrError::CorruptChunk(
            "ZSTD declared frameContentSize exceeds chunk limit",
        ));
    }

    let mut out = Vec::with_capacity(declared_fcs.unwrap_or(max_dst_size).min(max_dst_size));
    let mut rep = [1usize, 4usize, 8usize];
    let mut prev_huf: Option<HufDTable> = None;
    let mut prev_ll: Option<SeqDTable> = None;
    let mut prev_of: Option<SeqDTable> = None;
    let mut prev_ml: Option<SeqDTable> = None;

    loop {
        if pos + 3 > src.len() {
            return Err(ExrError::CorruptChunk("ZSTD block header truncated"));
        }
        let bh =
            u32::from(src[pos]) | (u32::from(src[pos + 1]) << 8) | (u32::from(src[pos + 2]) << 16);
        pos += 3;
        let last_block = (bh & 1) != 0;
        let block_type = (bh >> 1) & 3;
        let block_size = (bh >> 3) as usize;

        match block_type {
            0 => {
                // Raw_Block
                if pos + block_size > src.len() || out.len() + block_size > max_dst_size {
                    return Err(ExrError::CorruptChunk("ZSTD Raw_Block overflow"));
                }
                out.extend_from_slice(&src[pos..pos + block_size]);
                pos += block_size;
            }
            1 => {
                // RLE_Block
                if pos >= src.len() || out.len() + block_size > max_dst_size {
                    return Err(ExrError::CorruptChunk("ZSTD RLE_Block overflow"));
                }
                let val = src[pos];
                pos += 1;
                out.resize(out.len() + block_size, val);
            }
            2 => {
                // Compressed_Block
                if pos + block_size > src.len() {
                    return Err(ExrError::CorruptChunk("ZSTD Compressed_Block truncated"));
                }
                let block_slice = &src[pos..pos + block_size];
                pos += block_size;
                decompress_zstd_block(
                    block_slice,
                    &mut out,
                    max_dst_size,
                    &mut rep,
                    &mut prev_huf,
                    &mut prev_ll,
                    &mut prev_of,
                    &mut prev_ml,
                )?;
            }
            _ => return Err(ExrError::CorruptChunk("ZSTD reserved block type 3")),
        }

        if last_block {
            break;
        }
    }

    if checksum_flag != 0 {
        if pos + 4 > src.len() {
            return Err(ExrError::CorruptChunk("ZSTD frame checksum truncated"));
        }
        pos += 4;
    }
    let _ = pos;

    if let Some(fcs) = declared_fcs
        && out.len() != fcs
    {
        return Err(ExrError::CorruptChunk(
            "ZSTD decoded size does not match frameContentSize",
        ));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn decompress_zstd_block(
    block: &[u8],
    out: &mut Vec<u8>,
    max_dst_size: usize,
    rep: &mut [usize; 3],
    prev_huf: &mut Option<HufDTable>,
    prev_ll: &mut Option<SeqDTable>,
    prev_of: &mut Option<SeqDTable>,
    prev_ml: &mut Option<SeqDTable>,
) -> Result<(), ExrError> {
    if block.len() < 2 {
        return Err(ExrError::CorruptChunk("ZSTD block shorter than 2 bytes"));
    }

    // 1. Decode Literals_Section
    let lit_enc_type = block[0] & 3;
    let lhl_code = (block[0] >> 2) & 3;
    let (literals, lit_header_and_body_len) = match lit_enc_type {
        0 => {
            // Raw_Literals_Block (`set_basic`)
            let (lh_size, lit_size) = match lhl_code {
                0 | 2 => (1usize, usize::from(block[0] >> 3)),
                1 => {
                    if block.len() < 2 {
                        return Err(ExrError::CorruptChunk("ZSTD raw literals header truncated"));
                    }
                    (
                        2usize,
                        (u16::from_le_bytes([block[0], block[1]]) >> 4) as usize,
                    )
                }
                _ => {
                    if block.len() < 3 {
                        return Err(ExrError::CorruptChunk("ZSTD raw literals header truncated"));
                    }
                    let v = u32::from(block[0])
                        | (u32::from(block[1]) << 8)
                        | (u32::from(block[2]) << 16);
                    (3usize, (v >> 4) as usize)
                }
            };
            if lh_size + lit_size > block.len() {
                return Err(ExrError::CorruptChunk(
                    "ZSTD raw literals payload truncated",
                ));
            }
            (
                block[lh_size..lh_size + lit_size].to_vec(),
                lh_size + lit_size,
            )
        }
        1 => {
            // RLE_Literals_Block (`set_rle`)
            let (lh_size, lit_size) = match lhl_code {
                0 | 2 => (1usize, usize::from(block[0] >> 3)),
                1 => {
                    if block.len() < 3 {
                        return Err(ExrError::CorruptChunk("ZSTD RLE literals header truncated"));
                    }
                    (
                        2usize,
                        (u16::from_le_bytes([block[0], block[1]]) >> 4) as usize,
                    )
                }
                _ => {
                    if block.len() < 4 {
                        return Err(ExrError::CorruptChunk("ZSTD RLE literals header truncated"));
                    }
                    let v = u32::from(block[0])
                        | (u32::from(block[1]) << 8)
                        | (u32::from(block[2]) << 16);
                    (3usize, (v >> 4) as usize)
                }
            };
            if lh_size + 1 > block.len() || lit_size > ZSTD_BLOCKSIZE_MAX {
                return Err(ExrError::CorruptChunk("ZSTD RLE literals byte truncated"));
            }
            (vec![block[lh_size]; lit_size], lh_size + 1)
        }
        2 | 3 => {
            // Compressed_Literals_Block (`2`) or Treeless_Literals_Block (`3`)
            if block.len() < 5 {
                return Err(ExrError::CorruptChunk("ZSTD compressed literals < 5 bytes"));
            }
            let lhc = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
            let (single_stream, lh_size, lit_size, lit_csize) = match lhl_code {
                0 | 1 => (
                    lhl_code == 0,
                    3usize,
                    ((lhc >> 4) & 0x3FF) as usize,
                    ((lhc >> 14) & 0x3FF) as usize,
                ),
                2 => (
                    false,
                    4usize,
                    ((lhc >> 4) & 0x3FFF) as usize,
                    (lhc >> 18) as usize,
                ),
                _ => (
                    false,
                    5usize,
                    ((lhc >> 4) & 0x3FFFF) as usize,
                    ((lhc >> 22) as usize) + (usize::from(block[4]) << 10),
                ),
            };
            if lh_size + lit_csize > block.len() || lit_size > ZSTD_BLOCKSIZE_MAX {
                return Err(ExrError::CorruptChunk(
                    "ZSTD compressed literals out of bounds",
                ));
            }
            let c_slice = &block[lh_size..lh_size + lit_csize];
            let huf_payload = if lit_enc_type == 2 {
                let (tbl, huf_hdr_len) = HufDTable::read_x1(c_slice)?;
                *prev_huf = Some(tbl);
                &c_slice[huf_hdr_len..]
            } else {
                if prev_huf.is_none() {
                    return Err(ExrError::CorruptChunk(
                        "ZSTD treeless literals without previous HUF table",
                    ));
                }
                c_slice
            };
            let tbl = prev_huf.as_ref().unwrap();
            let mut lits = vec![0u8; lit_size];
            if single_stream {
                tbl.decode_stream_x1(huf_payload, &mut lits)?;
            } else {
                tbl.decode_stream_4x1(huf_payload, &mut lits)?;
            }
            (lits, lh_size + lit_csize)
        }
        _ => unreachable!(),
    };

    // 2. Decode Sequences_Section
    let seq_section = &block[lit_header_and_body_len..];
    if seq_section.is_empty() {
        return Err(ExrError::CorruptChunk("ZSTD missing sequences section"));
    }
    let mut sp = 0usize;
    let byte0 = usize::from(seq_section[sp]);
    sp += 1;
    let nb_seq = if byte0 == 0 {
        0usize
    } else if byte0 < 0x80 {
        byte0
    } else if byte0 < 0xFF {
        if sp >= seq_section.len() {
            return Err(ExrError::CorruptChunk("ZSTD 2-byte nbSeq truncated"));
        }
        let b1 = usize::from(seq_section[sp]);
        sp += 1;
        ((byte0 - 0x80) << 8) + b1
    } else {
        if sp + 2 > seq_section.len() {
            return Err(ExrError::CorruptChunk("ZSTD 3-byte nbSeq truncated"));
        }
        let w = u16::from_le_bytes([seq_section[sp], seq_section[sp + 1]]) as usize;
        sp += 2;
        w + 0x7F00
    };

    if nb_seq == 0 {
        if sp != seq_section.len() {
            return Err(ExrError::CorruptChunk("ZSTD trailing bytes after nbSeq=0"));
        }
        if out.len() + literals.len() > max_dst_size {
            return Err(ExrError::CorruptChunk("ZSTD block literals exceed max_dst"));
        }
        out.extend_from_slice(&literals);
        return Ok(());
    }

    if sp >= seq_section.len() {
        return Err(ExrError::CorruptChunk("ZSTD sequence modes byte truncated"));
    }
    let modes = seq_section[sp];
    sp += 1;
    if (modes & 3) != 0 {
        return Err(ExrError::CorruptChunk(
            "ZSTD sequence modes reserved bits non-zero",
        ));
    }
    let ll_type = (modes >> 6) & 3;
    let of_type = (modes >> 4) & 3;
    let ml_type = (modes >> 2) & 3;

    let (ll_table, ll_bytes) = decode_seq_table(
        ll_type,
        &seq_section[sp..],
        35,
        9,
        &LL_BASE,
        &LL_BITS,
        &LL_DEFAULT_NORM,
        LL_DEFAULT_NORM_LOG,
        prev_ll,
    )?;
    sp += ll_bytes;

    let (of_table, of_bytes) = decode_seq_table(
        of_type,
        &seq_section[sp..],
        31,
        8,
        &OF_BASE,
        &OF_BITS,
        &OF_DEFAULT_NORM,
        OF_DEFAULT_NORM_LOG,
        prev_of,
    )?;
    sp += of_bytes;

    let (ml_table, ml_bytes) = decode_seq_table(
        ml_type,
        &seq_section[sp..],
        52,
        9,
        &ML_BASE,
        &ML_BITS,
        &ML_DEFAULT_NORM,
        ML_DEFAULT_NORM_LOG,
        prev_ml,
    )?;
    sp += ml_bytes;

    *prev_ll = Some(ll_table.clone());
    *prev_of = Some(of_table.clone());
    *prev_ml = Some(ml_table.clone());

    let mut bit_d = BitDStream::init(&seq_section[sp..])?;
    let mut state_ll = bit_d.read_bits(ll_table.table_log) as usize;
    let _ = bit_d.reload();
    let mut state_of = bit_d.read_bits(of_table.table_log) as usize;
    let _ = bit_d.reload();
    let mut state_ml = bit_d.read_bits(ml_table.table_log) as usize;
    let _ = bit_d.reload();

    let mut lit_ptr = 0usize;
    for seq_idx in 0..nb_seq {
        let is_last = seq_idx + 1 == nb_seq;
        let ll_info = ll_table.entries[state_ll];
        let ml_info = ml_table.entries[state_ml];
        let of_info = of_table.entries[state_of];

        let of_base = of_info.base_value as usize;
        let ll_bits = u32::from(ll_info.nb_additional_bits);
        let ml_bits = u32::from(ml_info.nb_additional_bits);
        let of_bits = u32::from(of_info.nb_additional_bits);
        let total_bits = ll_bits + ml_bits + of_bits;

        let offset: usize;
        if of_bits > 1 {
            offset = of_base + (bit_d.read_bits_fast(of_bits) as usize);
            rep[2] = rep[1];
            rep[1] = rep[0];
            rep[0] = offset;
        } else {
            let ll0 = usize::from(ll_info.base_value == 0);
            if of_bits == 0 {
                offset = rep[ll0];
                rep[1] = rep[1 - ll0];
                rep[0] = offset;
            } else {
                let off_code = of_base + ll0 + (bit_d.read_bits_fast(1) as usize);
                let temp = if off_code == 3 {
                    rep[0].wrapping_sub(1)
                } else {
                    rep[off_code]
                };
                if temp == 0 {
                    return Err(ExrError::CorruptChunk("ZSTD sequence repeat offset == 0"));
                }
                if off_code != 1 {
                    rep[2] = rep[1];
                }
                rep[1] = rep[0];
                rep[0] = temp;
                offset = temp;
            }
        }

        let mut match_length = ml_info.base_value as usize;
        if ml_bits > 0 {
            match_length += bit_d.read_bits_fast(ml_bits) as usize;
        }

        if total_bits >= 64 - 7 - (9 + 9 + 8) {
            let _ = bit_d.reload();
        }

        let mut lit_length = ll_info.base_value as usize;
        if ll_bits > 0 {
            lit_length += bit_d.read_bits_fast(ll_bits) as usize;
        }

        if !is_last {
            let low_ll = bit_d.read_bits(u32::from(ll_info.nb_bits)) as usize;
            state_ll = usize::from(ll_info.next_state) + low_ll;
            let low_ml = bit_d.read_bits(u32::from(ml_info.nb_bits)) as usize;
            state_ml = usize::from(ml_info.next_state) + low_ml;
            let low_of = bit_d.read_bits(u32::from(of_info.nb_bits)) as usize;
            state_of = usize::from(of_info.next_state) + low_of;
            let _ = bit_d.reload();
        }

        // Execute sequence
        if lit_ptr + lit_length > literals.len() {
            return Err(ExrError::CorruptChunk("ZSTD sequence literal overrun"));
        }
        if out.len() + lit_length + match_length > max_dst_size {
            return Err(ExrError::CorruptChunk("ZSTD sequence output overrun"));
        }
        out.extend_from_slice(&literals[lit_ptr..lit_ptr + lit_length]);
        lit_ptr += lit_length;

        if offset == 0 || offset > out.len() {
            return Err(ExrError::CorruptChunk(
                "ZSTD sequence match offset out of bounds",
            ));
        }
        let match_start = out.len() - offset;
        for i in 0..match_length {
            let b = out[match_start + i];
            out.push(b);
        }
    }

    if !bit_d.is_end() {
        return Err(ExrError::CorruptChunk(
            "ZSTD sequence bitstream did not reach exact end",
        ));
    }

    // Copy remaining trailing literals
    let rem_lits = literals.len() - lit_ptr;
    if out.len() + rem_lits > max_dst_size {
        return Err(ExrError::CorruptChunk("ZSTD trailing literals overrun"));
    }
    out.extend_from_slice(&literals[lit_ptr..]);
    Ok(())
}

// ============================================================================
// 3. Pure-Rust RFC 8878 Zstandard Frame Compressor (`Predefined_Mode` FSE + LZ77)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
struct FseSymTT {
    delta_find_state: i32,
    delta_nb_bits: u32,
}

#[derive(Debug, Clone)]
struct FseCTable {
    table_log: u32,
    state_table: Vec<u16>,
    sym_tt: Vec<FseSymTT>,
}

impl FseCTable {
    fn build(norm: &[i16], table_log: u32) -> Self {
        let max_sv = norm.len() - 1;
        let table_size = 1usize << table_log;
        let table_mask = table_size - 1;
        let step = fse_tablestep(table_size);

        let mut cumul = vec![0u16; max_sv + 2];
        let mut table_symbol = vec![0u8; table_size];
        let mut high_threshold = table_size - 1;

        for u in 1..=(max_sv + 1) {
            if norm[u - 1] == -1 {
                cumul[u] = cumul[u - 1] + 1;
                table_symbol[high_threshold] = (u - 1) as u8;
                high_threshold = high_threshold.wrapping_sub(1);
            } else {
                cumul[u] = cumul[u - 1] + (norm[u - 1] as u16);
            }
        }
        cumul[max_sv + 1] = (table_size as u16) + 1;

        let mut position = 0usize;
        for (sym, &freq) in norm.iter().enumerate() {
            if freq > 0 {
                for _ in 0..freq {
                    table_symbol[position] = sym as u8;
                    position = (position + step) & table_mask;
                    while position > high_threshold {
                        position = (position + step) & table_mask;
                    }
                }
            }
        }
        debug_assert_eq!(position, 0);

        let mut state_table = vec![0u16; table_size];
        for (u, &s) in table_symbol.iter().enumerate() {
            let idx = usize::from(cumul[usize::from(s)]);
            cumul[usize::from(s)] += 1;
            state_table[idx] = (table_size + u) as u16;
        }

        let mut sym_tt = vec![FseSymTT::default(); max_sv + 1];
        let mut total = 0i32;
        for (s, &c) in norm.iter().enumerate() {
            match c {
                0 => {
                    sym_tt[s].delta_nb_bits = ((table_log + 1) << 16) - (1u32 << table_log);
                }
                -1 | 1 => {
                    sym_tt[s].delta_nb_bits = (table_log << 16) - (1u32 << table_log);
                    sym_tt[s].delta_find_state = total - 1;
                    total += 1;
                }
                _ => {
                    let max_bits_out = table_log - highbit32((c as u32) - 1);
                    let min_state_plus = (c as u32) << max_bits_out;
                    sym_tt[s].delta_nb_bits = (max_bits_out << 16) - min_state_plus;
                    sym_tt[s].delta_find_state = total - i32::from(c);
                    total += i32::from(c);
                }
            }
        }

        Self {
            table_log,
            state_table,
            sym_tt,
        }
    }

    #[inline]
    fn init_cstate2(&self, symbol: u8) -> isize {
        let tt = self.sym_tt[usize::from(symbol)];
        let nb_bits_out = (tt.delta_nb_bits + (1 << 15)) >> 16;
        let val = ((nb_bits_out << 16) - tt.delta_nb_bits) as isize;
        let idx = ((val >> nb_bits_out) + (tt.delta_find_state as isize)) as usize;
        self.state_table[idx] as isize
    }

    #[inline]
    fn encode_symbol(&self, bit_c: &mut BitCStream, state: &mut isize, symbol: u8) {
        let tt = self.sym_tt[usize::from(symbol)];
        let nb_bits_out = ((*state as u32).wrapping_add(tt.delta_nb_bits)) >> 16;
        bit_c.add_bits(*state as u64, nb_bits_out);
        let idx = ((*state >> nb_bits_out) + (tt.delta_find_state as isize)) as usize;
        *state = self.state_table[idx] as isize;
    }
}

struct BitCStream {
    out: Vec<u8>,
    bit_container: u64,
    bit_pos: u32,
}

impl BitCStream {
    fn new(cap: usize) -> Self {
        Self {
            out: Vec::with_capacity(cap),
            bit_container: 0,
            bit_pos: 0,
        }
    }

    #[inline]
    fn add_bits(&mut self, value: u64, nb_bits: u32) {
        if nb_bits == 0 {
            return;
        }
        let mask = if nb_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << nb_bits) - 1
        };
        self.bit_container |= (value & mask) << self.bit_pos;
        self.bit_pos += nb_bits;
    }

    #[inline]
    fn flush_bits(&mut self) {
        while self.bit_pos >= 8 {
            self.out.push((self.bit_container & 0xFF) as u8);
            self.bit_container >>= 8;
            self.bit_pos -= 8;
        }
    }

    fn close(mut self) -> Vec<u8> {
        self.add_bits(1, 1); // endMark
        self.flush_bits();
        if self.bit_pos > 0 {
            self.out.push((self.bit_container & 0xFF) as u8);
        }
        self.out
    }
}

#[derive(Debug, Clone, Copy)]
struct ZstdSeq {
    lit_len: u32,
    off_base: u32,
    ml_base: u32,
}

#[inline]
fn ll_code(lit_length: u32) -> u8 {
    for (i, &base) in LL_BASE.iter().enumerate().rev() {
        if lit_length >= base {
            return i as u8;
        }
    }
    0
}

#[inline]
fn ml_code(ml_base: u32) -> u8 {
    let match_len = ml_base + 3;
    for (i, &base) in ML_BASE.iter().enumerate().rev() {
        if match_len >= base {
            return i as u8;
        }
    }
    0
}

/// Compresses `data` into a standard RFC 8878 Zstandard frame.
pub fn zstd_compress_frame(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 32);
    out.extend_from_slice(&ZSTD_FRAME_MAGIC.to_le_bytes());

    // Single_Segment_Flag = 1 (`0x20`), FCS_Field_Size based on `data.len()`
    if data.len() < 256 {
        out.push(0x20); // fcs_flag = 0, single_segment = 1 -> 1 byte FCS
        out.push(data.len() as u8);
    } else if data.len() <= 65535 + 256 {
        out.push(0x60); // fcs_flag = 1, single_segment = 1 -> 2 byte FCS
        out.extend_from_slice(&((data.len() - 256) as u16).to_le_bytes());
    } else {
        out.push(0xA0); // fcs_flag = 2, single_segment = 1 -> 4 byte FCS
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    }

    if data.is_empty() {
        // Empty Raw_Block with last_block = 1
        out.extend_from_slice(&[0x01, 0x00, 0x00]);
        return out;
    }

    let ll_ct = FseCTable::build(&LL_DEFAULT_NORM, LL_DEFAULT_NORM_LOG);
    let of_ct = FseCTable::build(&OF_DEFAULT_NORM, OF_DEFAULT_NORM_LOG);
    let ml_ct = FseCTable::build(&ML_DEFAULT_NORM, ML_DEFAULT_NORM_LOG);

    let mut offset = 0usize;
    while offset < data.len() {
        let blk_len = (data.len() - offset).min(ZSTD_BLOCKSIZE_MAX);
        let is_last = offset + blk_len == data.len();
        let blk = &data[offset..offset + blk_len];
        offset += blk_len;

        if let Some(cblock) = compress_zstd_block(blk, &ll_ct, &of_ct, &ml_ct)
            && cblock.len() < blk.len()
        {
            let bh = u32::from(is_last) | (2u32 << 1) | ((cblock.len() as u32) << 3);
            out.push((bh & 0xFF) as u8);
            out.push(((bh >> 8) & 0xFF) as u8);
            out.push(((bh >> 16) & 0xFF) as u8);
            out.extend_from_slice(&cblock);
        } else {
            let bh = u32::from(is_last) | ((blk.len() as u32) << 3);
            out.push((bh & 0xFF) as u8);
            out.push(((bh >> 8) & 0xFF) as u8);
            out.push(((bh >> 16) & 0xFF) as u8);
            out.extend_from_slice(blk);
        }
    }

    out
}

fn compress_zstd_block(
    blk: &[u8],
    ll_ct: &FseCTable,
    of_ct: &FseCTable,
    ml_ct: &FseCTable,
) -> Option<Vec<u8>> {
    if blk.len() < 16 {
        return None;
    }

    // Fast 4-byte hash table LZ77 match finder
    const HASH_BITS: usize = 14;
    let mut table = vec![usize::MAX; 1 << HASH_BITS];
    let hash4 = |p: usize| -> usize {
        let v = u32::from_le_bytes([blk[p], blk[p + 1], blk[p + 2], blk[p + 3]]);
        (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize
    };

    let mut literals = Vec::with_capacity(blk.len());
    let mut sequences: Vec<ZstdSeq> = Vec::new();
    let mut anchor = 0usize;
    let mut ip = 0usize;
    let ilimit = blk.len().saturating_sub(8);

    while ip <= ilimit {
        let h = hash4(ip);
        let cand = table[h];
        table[h] = ip;

        if cand != usize::MAX
            && ip > cand
            && (ip - cand) <= (1 << 24)
            && blk[cand..cand + 4] == blk[ip..ip + 4]
        {
            let mut mlen = 4usize;
            let max_m = (blk.len() - ip).min(0x10002);
            while mlen < max_m && blk[cand + mlen] == blk[ip + mlen] {
                mlen += 1;
            }
            let lit_len = (ip - anchor) as u32;
            let match_off = (ip - cand) as u32;
            // In Zstd, offBase = offset + 3 (`OFFSET_TO_OFFBASE(offset)`)
            let off_base = match_off + 3;
            sequences.push(ZstdSeq {
                lit_len,
                off_base,
                ml_base: (mlen - 3) as u32,
            });
            literals.extend_from_slice(&blk[anchor..ip]);
            ip += mlen;
            anchor = ip;
        } else {
            ip += 1;
        }
    }

    if sequences.is_empty() {
        return None;
    }
    literals.extend_from_slice(&blk[anchor..]);

    // Encode Raw_Literals_Block header + literals
    let mut out = Vec::with_capacity(blk.len());
    let lit_size = literals.len();
    if lit_size < 32 {
        out.push((lit_size as u8) << 3);
    } else if lit_size < 4096 {
        let h = ((lit_size as u16) << 4) | (1 << 2);
        out.extend_from_slice(&h.to_le_bytes());
    } else {
        let h = ((lit_size as u32) << 4) | (3 << 2);
        out.push((h & 0xFF) as u8);
        out.push(((h >> 8) & 0xFF) as u8);
        out.push(((h >> 16) & 0xFF) as u8);
    }
    out.extend_from_slice(&literals);

    // Encode Sequences_Section header
    let nb_seq = sequences.len();
    if nb_seq < 0x80 {
        out.push(nb_seq as u8);
    } else if nb_seq < 0x7F00 {
        out.push(((nb_seq >> 8) + 0x80) as u8);
        out.push((nb_seq & 0xFF) as u8);
    } else {
        out.push(0xFF);
        out.extend_from_slice(&((nb_seq - 0x7F00) as u16).to_le_bytes());
    }
    // Compression modes byte = 0x00 (`Predefined_Mode` for LL, OF, and ML)
    out.push(0x00);

    // Precompute codes for each sequence
    let mut ll_codes = Vec::with_capacity(nb_seq);
    let mut of_codes = Vec::with_capacity(nb_seq);
    let mut ml_codes = Vec::with_capacity(nb_seq);
    for seq in &sequences {
        let llc = ll_code(seq.lit_len);
        let ofc = highbit32(seq.off_base) as u8;
        if ofc > 28 {
            return None; // Predefined OF table covers codes 0..=28
        }
        let mlc = ml_code(seq.ml_base);
        ll_codes.push(llc);
        of_codes.push(ofc);
        ml_codes.push(mlc);
    }

    // Encode sequences backward (`ZSTD_encodeSequences_body`)
    let mut bit_c = BitCStream::new(nb_seq * 8 + 16);
    let last = nb_seq - 1;
    let mut state_ml = ml_ct.init_cstate2(ml_codes[last]);
    let mut state_of = of_ct.init_cstate2(of_codes[last]);
    let mut state_ll = ll_ct.init_cstate2(ll_codes[last]);

    let last_ll_extra = sequences[last].lit_len - LL_BASE[usize::from(ll_codes[last])];
    let last_ml_extra = (sequences[last].ml_base + 3) - ML_BASE[usize::from(ml_codes[last])];
    bit_c.add_bits(
        u64::from(last_ll_extra),
        u32::from(LL_BITS[usize::from(ll_codes[last])]),
    );
    bit_c.add_bits(
        u64::from(last_ml_extra),
        u32::from(ML_BITS[usize::from(ml_codes[last])]),
    );
    bit_c.add_bits(
        u64::from(sequences[last].off_base),
        u32::from(of_codes[last]),
    );
    bit_c.flush_bits();

    for n in (0..last).rev() {
        let llc = ll_codes[n];
        let ofc = of_codes[n];
        let mlc = ml_codes[n];
        let ll_b = u32::from(LL_BITS[usize::from(llc)]);
        let of_b = u32::from(ofc);
        let ml_b = u32::from(ML_BITS[usize::from(mlc)]);

        of_ct.encode_symbol(&mut bit_c, &mut state_of, ofc);
        ml_ct.encode_symbol(&mut bit_c, &mut state_ml, mlc);
        ll_ct.encode_symbol(&mut bit_c, &mut state_ll, llc);
        bit_c.flush_bits();

        let ll_extra = sequences[n].lit_len - LL_BASE[usize::from(llc)];
        let ml_extra = (sequences[n].ml_base + 3) - ML_BASE[usize::from(mlc)];
        bit_c.add_bits(u64::from(ll_extra), ll_b);
        bit_c.add_bits(u64::from(ml_extra), ml_b);
        bit_c.add_bits(u64::from(sequences[n].off_base), of_b);
        bit_c.flush_bits();
    }

    bit_c.add_bits(state_ml as u64, ml_ct.table_log);
    bit_c.flush_bits();
    bit_c.add_bits(state_of as u64, of_ct.table_log);
    bit_c.flush_bits();
    bit_c.add_bits(state_ll as u64, ll_ct.table_log);
    bit_c.flush_bits();

    let seq_bytes = bit_c.close();
    out.extend_from_slice(&seq_bytes);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_frame_roundtrip() {
        let mut input = Vec::new();
        for i in 0..2048u32 {
            input.extend_from_slice(&(i / 8).to_le_bytes());
        }
        let compressed = zstd_compress_frame(&input);
        assert!(compressed.len() < input.len());
        let decompressed = zstd_decompress_frame(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_zstd_delta_and_shuffle_roundtrip() {
        let mut orig = Vec::new();
        for i in 0..128u16 {
            orig.extend_from_slice(&(i * 3).to_le_bytes());
        }
        let mut buf = orig.clone();
        delta_encode_row_u16(&mut buf, 128);
        let mut shuf = vec![0u8; buf.len()];
        shuffle_encode_2(&buf, &mut shuf);
        let mut unshuf = vec![0u8; buf.len()];
        shuffle_decode_2(&shuf, &mut unshuf);
        delta_decode_row_u16(&mut unshuf, 128);
        assert_eq!(unshuf, orig);
    }
}
