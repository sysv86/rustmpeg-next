// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! OpenEXR decoder supporting single-part and multi-part scanline,
//! tiled (`OneLevel`, `MipmapLevels`, `RipmapLevels`), deep scanline, and deep tiled parts.

use rustmpeg_core::{
    ChromaLocation, ColorDescription, ColorPrimaries, ColorRange, MatrixCoefficients, PixelFormat,
    TransferFunction, VideoFrame,
};

use super::compression::{decompress_chunk_data, decompress_deep_samples_chunk_data};
use super::half::f16_to_f32;
use super::header::{
    ExrChannelDesc, ExrCompression, ExrFileHeader, ExrLevelMode, ExrPartHeader, ExrPartType,
    ExrSampleType, parse_file_header,
};
use super::{
    ExrChannelData, ExrChannelPlane, ExrConfig, ExrDeepSamples, ExrError, ExrImage, ExrLevelImage,
    ExrPart,
};

/// High-level OpenEXR decoder.
#[derive(Debug, Clone)]
pub struct ExrDecoder {
    config: ExrConfig,
}

impl Default for ExrDecoder {
    fn default() -> Self {
        Self::new(ExrConfig::default())
    }
}

impl ExrDecoder {
    /// Creates a new OpenEXR decoder with the given [`ExrConfig`].
    #[must_use]
    pub const fn new(config: ExrConfig) -> Self {
        Self { config }
    }

    /// Parses and validates only the OpenEXR file header (and all part headers) without decoding pixel chunks.
    pub fn decode_headers(&self, data: &[u8]) -> Result<ExrFileHeader, ExrError> {
        let (hdr, _) = parse_file_header(
            data,
            self.config.max_width,
            self.config.max_height,
            self.config.strict,
        )?;
        Ok(hdr)
    }

    /// Decodes a complete OpenEXR file (single-part or multi-part, flat or deep, scanline or tiled).
    pub fn decode(&self, data: &[u8]) -> Result<ExrImage, ExrError> {
        let (file_header, mut pos) = parse_file_header(
            data,
            self.config.max_width,
            self.config.max_height,
            self.config.strict,
        )?;

        let is_multipart = file_header.is_multipart();
        let num_parts = file_header.parts.len();

        // Read offset tables for all parts (`docs/01-file-layout.md` Section 6)
        let mut part_chunk_counts = Vec::with_capacity(num_parts);
        let mut total_chunks = 0usize;
        for part_hdr in &file_header.parts {
            let cc = part_hdr.compute_expected_chunk_count()?;
            part_chunk_counts.push(cc);
            total_chunks = total_chunks
                .checked_add(cc)
                .ok_or(ExrError::DimensionsTooLarge)?;
        }

        let offset_table_bytes = total_chunks
            .checked_mul(8)
            .ok_or(ExrError::DimensionsTooLarge)?;
        if pos + offset_table_bytes > data.len() {
            return Err(ExrError::UnexpectedEof);
        }

        let chunks_start_pos = pos + offset_table_bytes;
        let mut part_offsets: Vec<Vec<u64>> = Vec::with_capacity(num_parts);
        let mut any_invalid_offset = false;

        for &cc in &part_chunk_counts {
            let mut offsets = Vec::with_capacity(cc);
            for _ in 0..cc {
                let off = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]);
                pos += 8;
                if off < (chunks_start_pos as u64) || off >= (data.len() as u64) {
                    any_invalid_offset = true;
                }
                offsets.push(off);
            }
            part_offsets.push(offsets);
        }

        if any_invalid_offset {
            if self.config.strict {
                return Err(ExrError::CorruptChunk(
                    "invalid or zero chunk offset in chunk offset table",
                ));
            }
            part_offsets = reconstruct_chunk_offsets(
                data,
                chunks_start_pos,
                &file_header.parts,
                &part_chunk_counts,
                is_multipart,
            )?;
        }

        // Allocate part accumulators
        let mut decoded_parts = Vec::with_capacity(num_parts);
        for part_hdr in &file_header.parts {
            decoded_parts.push(allocate_part_storage(part_hdr)?);
        }

        // Decode chunks for each part
        for part_idx in 0..num_parts {
            let part_hdr = &file_header.parts[part_idx];
            let offsets = &part_offsets[part_idx];
            let mut seen_chunks = vec![false; offsets.len()];

            for (chunk_idx, &chunk_off_u64) in offsets.iter().enumerate() {
                let chunk_off = usize::try_from(chunk_off_u64)
                    .map_err(|_| ExrError::CorruptChunk("chunk offset exceeds usize"))?;
                if chunk_off >= data.len() {
                    return Err(ExrError::CorruptChunk("chunk offset past EOF"));
                }

                decode_single_chunk(
                    data,
                    chunk_off,
                    part_idx,
                    chunk_idx,
                    is_multipart,
                    part_hdr,
                    &mut decoded_parts[part_idx],
                    &mut seen_chunks,
                    self.config.strict,
                )?;
            }
        }

        Ok(ExrImage {
            version: file_header.version,
            flags: file_header.flags,
            parts: decoded_parts,
        })
    }

    /// Decodes Part 0 of an OpenEXR file and converts it into an 8-bit [`VideoFrame`]
    /// (`Rgba8` if an `"A"` channel is present, `Rgb8` if `"R"`/`"G"`/`"B"` are present,
    /// or `I400` for single-channel `"Y"`/`"Z"`).
    ///
    /// Linear scene-referred float values in `[0.0, 1.0]` are mapped with the IEC 61966-2-1
    /// sRGB electro-optical transfer function (`linear_to_srgb_u8`).
    pub fn decode_to_video_frame(&self, data: &[u8]) -> Result<VideoFrame, ExrError> {
        let img = self.decode(data)?;
        let part = img
            .parts
            .first()
            .ok_or(ExrError::InvalidHeader("OpenEXR file has no parts"))?;
        part.to_video_frame()
    }
}

fn allocate_part_storage(hdr: &ExrPartHeader) -> Result<ExrPart, ExrError> {
    let w = hdr.data_window.width() as usize;
    let h = hdr.data_window.height() as usize;

    let mut channels = Vec::with_capacity(hdr.channels.len());
    if !hdr.part_type.is_deep() {
        for ch in &hdr.channels {
            let xs = (ch.x_sampling.max(1)) as usize;
            let ys = (ch.y_sampling.max(1)) as usize;
            let cw = w / xs;
            let ch_h = h / ys;
            let count = cw.checked_mul(ch_h).ok_or(ExrError::DimensionsTooLarge)?;
            let data = match ch.sample_type {
                ExrSampleType::Uint => ExrChannelData::Uint(vec![0u32; count]),
                ExrSampleType::Half => ExrChannelData::Half(vec![0u16; count]),
                ExrSampleType::Float => ExrChannelData::Float(vec![0.0f32; count]),
            };
            channels.push(ExrChannelPlane {
                desc: ch.clone(),
                width: cw as u32,
                height: ch_h as u32,
                data,
            });
        }
    }

    // Allocate extra mipmap / ripmap level buffers if tiled with multi-resolution levels
    let mut extra_levels = Vec::new();
    if hdr.part_type == ExrPartType::TiledImage
        && let Some(td) = hdr.tiles
    {
        let (nlx, nly) = td.level_counts(hdr.data_window);
        match td.level_mode {
            ExrLevelMode::OneLevel => {}
            ExrLevelMode::MipmapLevels => {
                for l in 1..nlx {
                    extra_levels.push(allocate_level_image(hdr, td, l, l)?);
                }
            }
            ExrLevelMode::RipmapLevels => {
                for ly in 0..nly {
                    for lx in 0..nlx {
                        if lx == 0 && ly == 0 {
                            continue;
                        }
                        extra_levels.push(allocate_level_image(hdr, td, lx, ly)?);
                    }
                }
            }
        }
    }

    let deep_samples = if hdr.part_type.is_deep() {
        let px_count = w.checked_mul(h).ok_or(ExrError::DimensionsTooLarge)?;
        let mut ch_samples = Vec::with_capacity(hdr.channels.len());
        for ch in &hdr.channels {
            let data = match ch.sample_type {
                ExrSampleType::Uint => ExrChannelData::Uint(Vec::new()),
                ExrSampleType::Half => ExrChannelData::Half(Vec::new()),
                ExrSampleType::Float => ExrChannelData::Float(Vec::new()),
            };
            ch_samples.push(data);
        }
        Some(ExrDeepSamples {
            width: w as u32,
            height: h as u32,
            sample_counts: vec![0u32; px_count],
            channels: ch_samples,
        })
    } else {
        None
    };

    Ok(ExrPart {
        header: hdr.clone(),
        channels,
        extra_levels,
        deep_samples,
    })
}

fn allocate_level_image(
    hdr: &ExrPartHeader,
    td: super::header::ExrTileDesc,
    lx: u32,
    ly: u32,
) -> Result<ExrLevelImage, ExrError> {
    let (lw, lh) = td.level_dimensions(hdr.data_window, lx, ly);
    let count = (lw as usize)
        .checked_mul(lh as usize)
        .ok_or(ExrError::DimensionsTooLarge)?;
    let mut channels = Vec::with_capacity(hdr.channels.len());
    for ch in &hdr.channels {
        let data = match ch.sample_type {
            ExrSampleType::Uint => ExrChannelData::Uint(vec![0u32; count]),
            ExrSampleType::Half => ExrChannelData::Half(vec![0u16; count]),
            ExrSampleType::Float => ExrChannelData::Float(vec![0.0f32; count]),
        };
        channels.push(ExrChannelPlane {
            desc: ch.clone(),
            width: lw,
            height: lh,
            data,
        });
    }
    Ok(ExrLevelImage {
        level_x: lx,
        level_y: ly,
        width: lw,
        height: lh,
        channels,
    })
}

#[allow(clippy::too_many_arguments)]
fn decode_single_chunk(
    data: &[u8],
    mut pos: usize,
    expected_part_idx: usize,
    _chunk_idx: usize,
    is_multipart: bool,
    hdr: &ExrPartHeader,
    part: &mut ExrPart,
    seen_chunks: &mut [bool],
    strict: bool,
) -> Result<(), ExrError> {
    if is_multipart {
        if pos + 4 > data.len() {
            return Err(ExrError::UnexpectedEof);
        }
        let part_num = i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        if part_num < 0 || (part_num as usize) != expected_part_idx {
            return Err(ExrError::CorruptChunk(
                "multipart chunk part_number does not match part index",
            ));
        }
    }

    match hdr.part_type {
        ExrPartType::ScanlineImage => {
            decode_flat_scanline_chunk(data, pos, hdr, part, seen_chunks, strict)
        }
        ExrPartType::TiledImage => {
            decode_flat_tile_chunk(data, pos, hdr, part, seen_chunks, strict)
        }
        ExrPartType::DeepScanline => {
            decode_deep_scanline_chunk(data, pos, hdr, part, seen_chunks, strict)
        }
        ExrPartType::DeepTile => decode_deep_tile_chunk(data, pos, hdr, part, seen_chunks, strict),
    }
}

fn decode_flat_scanline_chunk(
    data: &[u8],
    pos: usize,
    hdr: &ExrPartHeader,
    part: &mut ExrPart,
    seen_chunks: &mut [bool],
    strict: bool,
) -> Result<(), ExrError> {
    if pos + 8 > data.len() {
        return Err(ExrError::UnexpectedEof);
    }
    let y = i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let packed_size_i32 =
        i32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
    if packed_size_i32 < 0 {
        return Err(ExrError::CorruptChunk(
            "negative scanline chunk packed_size",
        ));
    }
    let packed_size = packed_size_i32 as usize;
    let payload_start = pos + 8;
    if payload_start + packed_size > data.len() {
        return Err(ExrError::CorruptChunk("scanline chunk payload truncated"));
    }

    let dw = hdr.data_window;
    if y < dw.min_y || y > dw.max_y {
        return Err(ExrError::CorruptChunk(
            "scanline chunk Y coordinate outside dataWindow",
        ));
    }
    let lines_per_chunk = hdr.compression.scanlines_per_chunk() as i32;
    let rel_y = y - dw.min_y;
    if strict && (rel_y % lines_per_chunk) != 0 {
        return Err(ExrError::CorruptChunk(
            "scanline chunk Y coordinate is not aligned to scanlines_per_chunk",
        ));
    }
    let block_idx = (rel_y / lines_per_chunk) as usize;
    if block_idx < seen_chunks.len() {
        if strict && seen_chunks[block_idx] {
            return Err(ExrError::CorruptChunk(
                "duplicate scanline chunk Y coordinate",
            ));
        }
        seen_chunks[block_idx] = true;
    }

    let num_lines = ((dw.max_y - y + 1) as usize).min(lines_per_chunk as usize);
    let width = dw.width() as usize;

    let expected_uncompressed =
        compute_uncompressed_region_bytes(&hdr.channels, width, y, num_lines);
    let packed = &data[payload_start..payload_start + packed_size];

    let raw = decompress_chunk_data(
        hdr.compression,
        packed,
        expected_uncompressed,
        &hdr.channels,
        width,
        y,
        num_lines,
    )?;

    // Copy unpacked scanlines into `part.channels`
    let mut cursor = 0usize;
    for line in 0..num_lines {
        let cur_y = y + (line as i32);
        for (c_idx, ch) in hdr.channels.iter().enumerate() {
            let n = ch.samples_in_line(cur_y, width);
            if n == 0 {
                continue;
            }
            let ys = ch.y_sampling.max(1);
            let dst_row = ((cur_y - dw.min_y) / ys) as usize;
            let plane = &mut part.channels[c_idx];
            copy_samples_into_plane(&mut plane.data, dst_row * n, n, &raw, &mut cursor);
        }
    }

    Ok(())
}

fn decode_flat_tile_chunk(
    data: &[u8],
    pos: usize,
    hdr: &ExrPartHeader,
    part: &mut ExrPart,
    _seen_chunks: &mut [bool],
    _strict: bool,
) -> Result<(), ExrError> {
    if pos + 20 > data.len() {
        return Err(ExrError::UnexpectedEof);
    }
    let tile_x = i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let tile_y = i32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
    let level_x =
        i32::from_le_bytes([data[pos + 8], data[pos + 9], data[pos + 10], data[pos + 11]]);
    let level_y = i32::from_le_bytes([
        data[pos + 12],
        data[pos + 13],
        data[pos + 14],
        data[pos + 15],
    ]);
    let packed_size_i32 = i32::from_le_bytes([
        data[pos + 16],
        data[pos + 17],
        data[pos + 18],
        data[pos + 19],
    ]);

    if packed_size_i32 < 0 || tile_x < 0 || tile_y < 0 || level_x < 0 || level_y < 0 {
        return Err(ExrError::CorruptChunk(
            "negative coordinate or packed_size in tile chunk header",
        ));
    }
    let packed_size = packed_size_i32 as usize;
    let payload_start = pos + 20;
    if payload_start + packed_size > data.len() {
        return Err(ExrError::CorruptChunk("tile chunk payload truncated"));
    }

    let td = hdr
        .tiles
        .ok_or(ExrError::InvalidHeader("missing tiles attribute"))?;
    let (nlx, nly) = td.level_counts(hdr.data_window);
    let lx = level_x as u32;
    let ly = level_y as u32;

    if lx >= nlx || ly >= nly {
        return Err(ExrError::CorruptChunk("tile level index out of bounds"));
    }
    if td.level_mode == ExrLevelMode::OneLevel && (lx != 0 || ly != 0) {
        return Err(ExrError::CorruptChunk(
            "non-zero level in ONE_LEVEL tiled image",
        ));
    }
    if td.level_mode == ExrLevelMode::MipmapLevels && lx != ly {
        return Err(ExrError::CorruptChunk(
            "lx != ly in MIPMAP_LEVELS tiled image",
        ));
    }

    let (num_tx, num_ty) = td.tile_counts(hdr.data_window, lx, ly);
    let tx = tile_x as u32;
    let ty = tile_y as u32;
    if tx >= num_tx || ty >= num_ty {
        return Err(ExrError::CorruptChunk("tile (dx, dy) index out of bounds"));
    }

    let (lvl_w, lvl_h) = td.level_dimensions(hdr.data_window, lx, ly);
    let px_x0 = (tx * td.x_size) as usize;
    let px_y0 = (ty * td.y_size) as usize;
    let tile_w = ((lvl_w - tx * td.x_size).min(td.x_size)) as usize;
    let tile_h = ((lvl_h - ty * td.y_size).min(td.y_size)) as usize;
    let y_start = hdr.data_window.min_y + (px_y0 as i32);

    let expected_uncompressed =
        compute_uncompressed_region_bytes(&hdr.channels, tile_w, y_start, tile_h);
    let packed = &data[payload_start..payload_start + packed_size];

    let raw = decompress_chunk_data(
        hdr.compression,
        packed,
        expected_uncompressed,
        &hdr.channels,
        tile_w,
        y_start,
        tile_h,
    )?;

    let target_planes: &mut [ExrChannelPlane] = if lx == 0 && ly == 0 {
        &mut part.channels
    } else {
        let lvl_img = part
            .extra_levels
            .iter_mut()
            .find(|l| l.level_x == lx && l.level_y == ly)
            .ok_or(ExrError::CorruptChunk("missing level storage"))?;
        &mut lvl_img.channels
    };

    let mut cursor = 0usize;
    for row in 0..tile_h {
        let dst_y = px_y0 + row;
        for plane in target_planes.iter_mut() {
            let dst_offset = dst_y * (lvl_w as usize) + px_x0;
            copy_samples_into_plane(&mut plane.data, dst_offset, tile_w, &raw, &mut cursor);
        }
    }

    Ok(())
}

fn decode_deep_scanline_chunk(
    data: &[u8],
    pos: usize,
    hdr: &ExrPartHeader,
    part: &mut ExrPart,
    _seen_chunks: &mut [bool],
    _strict: bool,
) -> Result<(), ExrError> {
    if pos + 28 > data.len() {
        return Err(ExrError::UnexpectedEof);
    }
    let y = i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let packed_count_sz = u64::from_le_bytes([
        data[pos + 4],
        data[pos + 5],
        data[pos + 6],
        data[pos + 7],
        data[pos + 8],
        data[pos + 9],
        data[pos + 10],
        data[pos + 11],
    ]) as usize;
    let packed_data_sz = u64::from_le_bytes([
        data[pos + 12],
        data[pos + 13],
        data[pos + 14],
        data[pos + 15],
        data[pos + 16],
        data[pos + 17],
        data[pos + 18],
        data[pos + 19],
    ]) as usize;
    let unpacked_data_sz = u64::from_le_bytes([
        data[pos + 20],
        data[pos + 21],
        data[pos + 22],
        data[pos + 23],
        data[pos + 24],
        data[pos + 25],
        data[pos + 26],
        data[pos + 27],
    ]) as usize;

    let count_start = pos + 28;
    let data_start = count_start
        .checked_add(packed_count_sz)
        .ok_or(ExrError::CorruptChunk("deep scanline count size overflow"))?;
    let chunk_end = data_start
        .checked_add(packed_data_sz)
        .ok_or(ExrError::CorruptChunk("deep scanline data size overflow"))?;
    if chunk_end > data.len() {
        return Err(ExrError::CorruptChunk("deep scanline chunk truncated"));
    }

    let dw = hdr.data_window;
    if y < dw.min_y || y > dw.max_y {
        return Err(ExrError::CorruptChunk(
            "deep scanline Y coordinate outside dataWindow",
        ));
    }
    let lines_per_chunk = hdr.compression.scanlines_per_chunk() as i32;
    let num_lines = ((dw.max_y - y + 1) as usize).min(lines_per_chunk as usize);
    let width = dw.width() as usize;
    let num_pixels = width * num_lines;
    let expected_count_bytes = num_pixels * 4;

    let raw_count_bytes = decompress_chunk_data(
        hdr.compression,
        &data[count_start..data_start],
        expected_count_bytes,
        &[],
        width,
        y,
        num_lines,
    )?;

    // Parse per-scanline cumulative sample count table (`OpenEXRCore/decoding.c: unpack_sample_table`)
    let mut pixel_counts = Vec::with_capacity(num_pixels);
    let mut row_sample_counts = Vec::with_capacity(num_lines);
    let mut total_chunk_samples = 0usize;
    for line in 0..num_lines {
        let mut prev_cum = 0u32;
        for x in 0..width {
            let off = (line * width + x) * 4;
            let cum_i32 = i32::from_le_bytes([
                raw_count_bytes[off],
                raw_count_bytes[off + 1],
                raw_count_bytes[off + 2],
                raw_count_bytes[off + 3],
            ]);
            if cum_i32 < 0 || (cum_i32 as u32) < prev_cum {
                return Err(ExrError::CorruptChunk(
                    "non-monotonic or negative cumulative count in deep scanline table",
                ));
            }
            let cum = cum_i32 as u32;
            pixel_counts.push(cum - prev_cum);
            prev_cum = cum;
        }
        row_sample_counts.push(prev_cum as usize);
        total_chunk_samples += prev_cum as usize;
    }

    let bytes_per_pixel_sample: usize = hdr
        .channels
        .iter()
        .map(|c| c.sample_type.bytes_per_sample())
        .sum();
    let expected_unpacked = total_chunk_samples
        .checked_mul(bytes_per_pixel_sample)
        .ok_or(ExrError::CorruptChunk("deep sample byte size overflow"))?;
    if expected_unpacked != unpacked_data_sz {
        return Err(ExrError::CorruptChunk(
            "deep scanline unpacked_size does not match cumulative sample count",
        ));
    }

    let raw_samples = decompress_deep_samples_chunk_data(
        hdr.compression,
        &data[data_start..chunk_end],
        unpacked_data_sz,
        &hdr.channels,
        width,
        y,
        num_lines,
        &row_sample_counts,
    )?;

    let deep = part
        .deep_samples
        .as_mut()
        .ok_or(ExrError::CorruptChunk("missing deep_samples storage"))?;
    let rel_y0 = (y - dw.min_y) as usize;
    let mut cursor = 0usize;

    for line in 0..num_lines {
        let row_px_start = (rel_y0 + line) * width;
        let line_counts = &pixel_counts[line * width..(line + 1) * width];
        deep.sample_counts[row_px_start..row_px_start + width].copy_from_slice(line_counts);
        let line_total_samples: usize = line_counts.iter().map(|&c| c as usize).sum();
        for (c_idx, _ch) in hdr.channels.iter().enumerate() {
            append_deep_samples(
                &mut deep.channels[c_idx],
                line_total_samples,
                &raw_samples,
                &mut cursor,
            );
        }
    }

    Ok(())
}

fn decode_deep_tile_chunk(
    data: &[u8],
    pos: usize,
    hdr: &ExrPartHeader,
    part: &mut ExrPart,
    _seen_chunks: &mut [bool],
    _strict: bool,
) -> Result<(), ExrError> {
    if pos + 40 > data.len() {
        return Err(ExrError::UnexpectedEof);
    }
    let tile_x = i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let tile_y = i32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
    let level_x =
        i32::from_le_bytes([data[pos + 8], data[pos + 9], data[pos + 10], data[pos + 11]]);
    let level_y = i32::from_le_bytes([
        data[pos + 12],
        data[pos + 13],
        data[pos + 14],
        data[pos + 15],
    ]);
    let packed_count_sz = u64::from_le_bytes([
        data[pos + 16],
        data[pos + 17],
        data[pos + 18],
        data[pos + 19],
        data[pos + 20],
        data[pos + 21],
        data[pos + 22],
        data[pos + 23],
    ]) as usize;
    let packed_data_sz = u64::from_le_bytes([
        data[pos + 24],
        data[pos + 25],
        data[pos + 26],
        data[pos + 27],
        data[pos + 28],
        data[pos + 29],
        data[pos + 30],
        data[pos + 31],
    ]) as usize;
    let unpacked_data_sz = u64::from_le_bytes([
        data[pos + 32],
        data[pos + 33],
        data[pos + 34],
        data[pos + 35],
        data[pos + 36],
        data[pos + 37],
        data[pos + 38],
        data[pos + 39],
    ]) as usize;

    if tile_x < 0 || tile_y < 0 || level_x < 0 || level_y < 0 {
        return Err(ExrError::CorruptChunk("negative deep tile coordinates"));
    }

    let count_start = pos + 40;
    let data_start = count_start
        .checked_add(packed_count_sz)
        .ok_or(ExrError::CorruptChunk("deep tile count size overflow"))?;
    let chunk_end = data_start
        .checked_add(packed_data_sz)
        .ok_or(ExrError::CorruptChunk("deep tile data size overflow"))?;
    if chunk_end > data.len() {
        return Err(ExrError::CorruptChunk("deep tile chunk truncated"));
    }

    let td = hdr
        .tiles
        .ok_or(ExrError::InvalidHeader("missing tiles attribute"))?;
    let lx = level_x as u32;
    let ly = level_y as u32;
    let tx = tile_x as u32;
    let ty = tile_y as u32;
    let (lvl_w, lvl_h) = td.level_dimensions(hdr.data_window, lx, ly);
    let tile_w = ((lvl_w - tx * td.x_size).min(td.x_size)) as usize;
    let tile_h = ((lvl_h - ty * td.y_size).min(td.y_size)) as usize;
    let num_pixels = tile_w * tile_h;
    let y_start = hdr.data_window.min_y + ((ty * td.y_size) as i32);

    let raw_count_bytes = decompress_chunk_data(
        hdr.compression,
        &data[count_start..data_start],
        num_pixels * 4,
        &[],
        tile_w,
        y_start,
        tile_h,
    )?;

    let mut pixel_counts = Vec::with_capacity(num_pixels);
    let mut row_sample_counts = Vec::with_capacity(tile_h);
    let mut total_tile_samples = 0usize;
    for row in 0..tile_h {
        let mut prev_cum = 0u32;
        for x in 0..tile_w {
            let off = (row * tile_w + x) * 4;
            let cum_i32 = i32::from_le_bytes([
                raw_count_bytes[off],
                raw_count_bytes[off + 1],
                raw_count_bytes[off + 2],
                raw_count_bytes[off + 3],
            ]);
            if cum_i32 < 0 || (cum_i32 as u32) < prev_cum {
                return Err(ExrError::CorruptChunk(
                    "non-monotonic or negative cumulative count in deep tile table",
                ));
            }
            let cum = cum_i32 as u32;
            pixel_counts.push(cum - prev_cum);
            prev_cum = cum;
        }
        row_sample_counts.push(prev_cum as usize);
        total_tile_samples += prev_cum as usize;
    }

    let bytes_per_pixel_sample: usize = hdr
        .channels
        .iter()
        .map(|c| c.sample_type.bytes_per_sample())
        .sum();
    if total_tile_samples * bytes_per_pixel_sample != unpacked_data_sz {
        return Err(ExrError::CorruptChunk(
            "deep tile unpacked_size does not match cumulative sample count",
        ));
    }

    let raw_samples = decompress_deep_samples_chunk_data(
        hdr.compression,
        &data[data_start..chunk_end],
        unpacked_data_sz,
        &hdr.channels,
        tile_w,
        y_start,
        tile_h,
        &row_sample_counts,
    )?;

    if lx == 0 && ly == 0 {
        let deep = part
            .deep_samples
            .as_mut()
            .ok_or(ExrError::CorruptChunk("missing deep_samples storage"))?;
        let px_x0 = (tx * td.x_size) as usize;
        let px_y0 = (ty * td.y_size) as usize;
        let mut cursor = 0usize;
        for row in 0..tile_h {
            let line_counts = &pixel_counts[row * tile_w..(row + 1) * tile_w];
            let dst_row_start = (px_y0 + row) * (lvl_w as usize) + px_x0;
            deep.sample_counts[dst_row_start..dst_row_start + tile_w].copy_from_slice(line_counts);
            let line_total: usize = line_counts.iter().map(|&c| c as usize).sum();
            for (c_idx, _ch) in hdr.channels.iter().enumerate() {
                append_deep_samples(
                    &mut deep.channels[c_idx],
                    line_total,
                    &raw_samples,
                    &mut cursor,
                );
            }
        }
    }

    Ok(())
}

pub(crate) fn compute_uncompressed_region_bytes(
    channels: &[ExrChannelDesc],
    region_width: usize,
    y_start: i32,
    num_lines: usize,
) -> usize {
    let mut total = 0usize;
    for line in 0..num_lines {
        let y = y_start + (line as i32);
        for ch in channels {
            let n = ch.samples_in_line(y, region_width);
            total += n * ch.sample_type.bytes_per_sample();
        }
    }
    total
}

fn copy_samples_into_plane(
    plane: &mut ExrChannelData,
    dst_offset: usize,
    count: usize,
    raw: &[u8],
    cursor: &mut usize,
) {
    match plane {
        ExrChannelData::Uint(dst) => {
            for i in 0..count {
                let off = *cursor + i * 4;
                dst[dst_offset + i] =
                    u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            }
            *cursor += count * 4;
        }
        ExrChannelData::Half(dst) => {
            for i in 0..count {
                let off = *cursor + i * 2;
                dst[dst_offset + i] = u16::from_le_bytes([raw[off], raw[off + 1]]);
            }
            *cursor += count * 2;
        }
        ExrChannelData::Float(dst) => {
            for i in 0..count {
                let off = *cursor + i * 4;
                dst[dst_offset + i] =
                    f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            }
            *cursor += count * 4;
        }
    }
}

fn append_deep_samples(plane: &mut ExrChannelData, count: usize, raw: &[u8], cursor: &mut usize) {
    match plane {
        ExrChannelData::Uint(dst) => {
            for i in 0..count {
                let off = *cursor + i * 4;
                dst.push(u32::from_le_bytes([
                    raw[off],
                    raw[off + 1],
                    raw[off + 2],
                    raw[off + 3],
                ]));
            }
            *cursor += count * 4;
        }
        ExrChannelData::Half(dst) => {
            for i in 0..count {
                let off = *cursor + i * 2;
                dst.push(u16::from_le_bytes([raw[off], raw[off + 1]]));
            }
            *cursor += count * 2;
        }
        ExrChannelData::Float(dst) => {
            for i in 0..count {
                let off = *cursor + i * 4;
                dst.push(f32::from_le_bytes([
                    raw[off],
                    raw[off + 1],
                    raw[off + 2],
                    raw[off + 3],
                ]));
            }
            *cursor += count * 4;
        }
    }
}

/// Reconstructs chunk offsets sequentially from `chunks_start_pos` when `strict == false`
/// and the file has a zeroed/missing offset table (`OpenEXRCore/chunk.c: reconstruct_chunk_table`).
fn reconstruct_chunk_offsets(
    data: &[u8],
    mut pos: usize,
    parts: &[ExrPartHeader],
    part_chunk_counts: &[usize],
    is_multipart: bool,
) -> Result<Vec<Vec<u64>>, ExrError> {
    let mut table: Vec<Vec<u64>> = part_chunk_counts
        .iter()
        .map(|&cc| Vec::with_capacity(cc))
        .collect();
    let total_chunks: usize = part_chunk_counts.iter().sum();

    for _ in 0..total_chunks {
        if pos >= data.len() {
            return Err(ExrError::CorruptChunk(
                "truncated chunk stream during offset table reconstruction",
            ));
        }
        let chunk_start = pos;
        let part_idx = if is_multipart {
            if pos + 4 > data.len() {
                return Err(ExrError::UnexpectedEof);
            }
            let p = i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            if p < 0 || (p as usize) >= parts.len() {
                return Err(ExrError::CorruptChunk("invalid part index in chunk"));
            }
            p as usize
        } else {
            0
        };

        let hdr = &parts[part_idx];
        let step = match hdr.part_type {
            ExrPartType::ScanlineImage => {
                if pos + 8 > data.len() {
                    return Err(ExrError::UnexpectedEof);
                }
                let psize = i32::from_le_bytes([
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]);
                if psize < 0 {
                    return Err(ExrError::CorruptChunk("negative chunk size"));
                }
                8 + (psize as usize)
            }
            ExrPartType::TiledImage => {
                if pos + 20 > data.len() {
                    return Err(ExrError::UnexpectedEof);
                }
                let psize = i32::from_le_bytes([
                    data[pos + 16],
                    data[pos + 17],
                    data[pos + 18],
                    data[pos + 19],
                ]);
                if psize < 0 {
                    return Err(ExrError::CorruptChunk("negative tile chunk size"));
                }
                20 + (psize as usize)
            }
            ExrPartType::DeepScanline => {
                if pos + 28 > data.len() {
                    return Err(ExrError::UnexpectedEof);
                }
                let csz = u64::from_le_bytes([
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                    data[pos + 8],
                    data[pos + 9],
                    data[pos + 10],
                    data[pos + 11],
                ]) as usize;
                let dsz = u64::from_le_bytes([
                    data[pos + 12],
                    data[pos + 13],
                    data[pos + 14],
                    data[pos + 15],
                    data[pos + 16],
                    data[pos + 17],
                    data[pos + 18],
                    data[pos + 19],
                ]) as usize;
                28 + csz + dsz
            }
            ExrPartType::DeepTile => {
                if pos + 40 > data.len() {
                    return Err(ExrError::UnexpectedEof);
                }
                let csz = u64::from_le_bytes([
                    data[pos + 16],
                    data[pos + 17],
                    data[pos + 18],
                    data[pos + 19],
                    data[pos + 20],
                    data[pos + 21],
                    data[pos + 22],
                    data[pos + 23],
                ]) as usize;
                let dsz = u64::from_le_bytes([
                    data[pos + 24],
                    data[pos + 25],
                    data[pos + 26],
                    data[pos + 27],
                    data[pos + 28],
                    data[pos + 29],
                    data[pos + 30],
                    data[pos + 31],
                ]) as usize;
                40 + csz + dsz
            }
        };

        pos = pos
            .checked_add(step)
            .ok_or(ExrError::CorruptChunk("chunk step overflow"))?;
        if pos > data.len() {
            return Err(ExrError::CorruptChunk("chunk extends past EOF"));
        }
        table[part_idx].push(chunk_start as u64);
    }

    Ok(table)
}

/// Converts a linear scene-referred float value in `[0.0, 1.0]` to an 8-bit sRGB value
/// using IEC 61966-2-1 transfer function.
#[inline]
pub fn linear_to_srgb_u8(val: f32) -> u8 {
    if val.is_nan() || val <= 0.0 {
        return 0;
    }
    if val >= 1.0 {
        return 255;
    }
    let srgb = if val <= 0.003_130_8 {
        val * 12.92
    } else {
        1.055 * val.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

impl ExrPart {
    /// Converts this flat OpenEXR part into an 8-bit [`VideoFrame`] (`Rgba8`, `Rgb8`, or `I400`).
    pub fn to_video_frame(&self) -> Result<VideoFrame, ExrError> {
        use rustmpeg_core::{Buffer, Plane};
        if self.header.part_type.is_deep() {
            return Err(ExrError::InvalidHeader(
                "cannot convert deep OpenEXR part directly to flat VideoFrame",
            ));
        }
        let w = self.header.data_window.width();
        let h = self.header.data_window.height();
        let px_count = (w as usize) * (h as usize);
        let srgb_color = ColorDescription {
            range: ColorRange::Full,
            primaries: ColorPrimaries::Bt709,
            transfer: TransferFunction::Srgb,
            matrix: MatrixCoefficients::Identity,
            chroma_location: ChromaLocation::Left,
        };

        let a_ch = self.channel_by_name("A");
        if let Some(rgb) = self.reconstruct_rgb_f32()? {
            if let Some(a) = a_ch {
                let mut dst = vec![0u8; px_count * 4];
                for i in 0..px_count {
                    dst[i * 4] = linear_to_srgb_u8(rgb.r[i]);
                    dst[i * 4 + 1] = linear_to_srgb_u8(rgb.g[i]);
                    dst[i * 4 + 2] = linear_to_srgb_u8(rgb.b[i]);
                    let alpha = (a.sample_as_f32(i).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    dst[i * 4 + 3] = alpha;
                }
                let plane0 = Plane::new(Buffer::from_vec(dst), (w as usize) * 4, w, h);
                return Ok(VideoFrame::from_planes(
                    PixelFormat::Rgba8,
                    8,
                    w,
                    h,
                    srgb_color,
                    [plane0, Plane::empty(), Plane::empty()],
                )?);
            }
            let mut dst = vec![0u8; px_count * 3];
            for i in 0..px_count {
                dst[i * 3] = linear_to_srgb_u8(rgb.r[i]);
                dst[i * 3 + 1] = linear_to_srgb_u8(rgb.g[i]);
                dst[i * 3 + 2] = linear_to_srgb_u8(rgb.b[i]);
            }
            let plane0 = Plane::new(Buffer::from_vec(dst), (w as usize) * 3, w, h);
            return Ok(VideoFrame::from_planes(
                PixelFormat::Rgb8,
                8,
                w,
                h,
                srgb_color,
                [plane0, Plane::empty(), Plane::empty()],
            )?);
        }

        // Single-channel fallback (`Y` or first channel)
        let first = self
            .channels
            .first()
            .ok_or(ExrError::InvalidHeader("part has no channels"))?;
        let mut dst = vec![0u8; px_count];
        for (i, slot) in dst.iter_mut().enumerate().take(px_count) {
            *slot = linear_to_srgb_u8(first.sample_as_f32(i));
        }
        let plane0 = Plane::new(Buffer::from_vec(dst), w as usize, w, h);
        let _ = f16_to_f32;
        let _ = ExrCompression::None;
        Ok(VideoFrame::from_planes(
            PixelFormat::I400,
            8,
            w,
            h,
            srgb_color,
            [plane0, Plane::empty(), Plane::empty()],
        )?)
    }
}
