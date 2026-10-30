// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! Clean-room OpenEXR encoder supporting single-part and multi-part scanline,
//! tiled (`OneLevel`, `MipmapLevels`, `RipmapLevels`), deep scanline, and deep tiled parts.

use rustmpeg_core::{PixelFormat, VideoFrame};

use super::compression::{
    compress_chunk_data, compress_chunk_data_with_dwa_level, compress_deep_samples_chunk_data,
};
use super::decoder::compute_uncompressed_region_bytes;
use super::half::f32_to_f16;
use super::header::{
    ExrChannelDesc, ExrCompression, ExrLevelMode, ExrLineOrder, ExrPartHeader, ExrPartType,
    ExrSampleType, write_file_header,
};
use super::{ExrChannelData, ExrChannelPlane, ExrConfig, ExrError, ExrImage, ExrPart};

/// High-level OpenEXR encoder.
#[derive(Debug, Clone)]
pub struct ExrEncoder {
    config: ExrConfig,
}

impl Default for ExrEncoder {
    fn default() -> Self {
        Self::new(ExrConfig::default())
    }
}

impl ExrEncoder {
    /// Creates a new OpenEXR encoder with the given [`ExrConfig`].
    #[must_use]
    pub const fn new(config: ExrConfig) -> Self {
        Self { config }
    }

    /// Encodes a complete [`ExrImage`] (single-part or multi-part, flat or deep, scanline or tiled)
    /// into an OpenEXR byte buffer.
    pub fn encode(&self, image: &ExrImage) -> Result<Vec<u8>, ExrError> {
        if image.parts.is_empty() {
            return Err(ExrError::InvalidHeader(
                "cannot encode ExrImage with 0 parts",
            ));
        }

        let is_multipart = image.parts.len() > 1;
        let headers: Vec<ExrPartHeader> = image.parts.iter().map(|p| p.header.clone()).collect();

        let mut out = Vec::new();
        write_file_header(&headers, &mut out)?;

        // Compute total chunk count across all parts and reserve space for the chunk offset table
        let mut part_chunk_counts = Vec::with_capacity(image.parts.len());
        let mut total_chunks = 0usize;
        for part in &image.parts {
            let cc = part.header.compute_expected_chunk_count()?;
            part_chunk_counts.push(cc);
            total_chunks = total_chunks
                .checked_add(cc)
                .ok_or(ExrError::DimensionsTooLarge)?;
        }

        let offset_table_pos = out.len();
        out.resize(offset_table_pos + total_chunks * 8, 0);

        let mut chunk_offsets: Vec<u64> = Vec::with_capacity(total_chunks);
        for (part_idx, part) in image.parts.iter().enumerate() {
            encode_part_chunks(
                part_idx,
                is_multipart,
                part,
                self.config.dwa_compression_level,
                &mut out,
                &mut chunk_offsets,
            )?;
        }

        if chunk_offsets.len() != total_chunks {
            return Err(ExrError::InvalidHeader(
                "encoded chunk count does not match expected chunk count",
            ));
        }

        // Backfill the 64-bit little-endian chunk offset table
        for (i, &off) in chunk_offsets.iter().enumerate() {
            let pos = offset_table_pos + i * 8;
            out[pos..pos + 8].copy_from_slice(&off.to_le_bytes());
        }

        Ok(out)
    }

    /// Encodes an 8-bit [`VideoFrame`] (`Rgba8`, `Rgb8`, `Bgra8`, or `I400`) into a single-part
    /// scanline OpenEXR image using `self.config.compression` and `HALF` (`f16`) linear channels.
    pub fn encode_video_frame(&self, frame: &VideoFrame) -> Result<Vec<u8>, ExrError> {
        let w = frame.width;
        let h = frame.height;
        let px_count = (w as usize) * (h as usize);

        let mut planes: Vec<ExrChannelPlane> = Vec::new();
        match frame.format {
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
                let is_bgra = frame.format == PixelFormat::Bgra8;
                let src_plane = frame
                    .plane(0)
                    .ok_or(ExrError::InvalidHeader("missing VideoFrame plane 0"))?;
                let mut a_data = vec![0u16; px_count];
                let mut b_data = vec![0u16; px_count];
                let mut g_data = vec![0u16; px_count];
                let mut r_data = vec![0u16; px_count];

                for y in 0..(h as usize) {
                    let row = src_plane.row(y)?;
                    for x in 0..(w as usize) {
                        let idx = y * (w as usize) + x;
                        let (r8, g8, b8, a8) = if is_bgra {
                            (row[x * 4 + 2], row[x * 4 + 1], row[x * 4], row[x * 4 + 3])
                        } else {
                            (row[x * 4], row[x * 4 + 1], row[x * 4 + 2], row[x * 4 + 3])
                        };
                        r_data[idx] = f32_to_f16(srgb_u8_to_linear(r8));
                        g_data[idx] = f32_to_f16(srgb_u8_to_linear(g8));
                        b_data[idx] = f32_to_f16(srgb_u8_to_linear(b8));
                        a_data[idx] = f32_to_f16(f32::from(a8) / 255.0);
                    }
                }

                for (name, buf) in [("A", a_data), ("B", b_data), ("G", g_data), ("R", r_data)] {
                    planes.push(ExrChannelPlane {
                        desc: ExrChannelDesc::new(name, ExrSampleType::Half),
                        width: w,
                        height: h,
                        data: ExrChannelData::Half(buf),
                    });
                }
            }
            PixelFormat::Rgb8 => {
                let src_plane = frame
                    .plane(0)
                    .ok_or(ExrError::InvalidHeader("missing VideoFrame plane 0"))?;
                let mut b_data = vec![0u16; px_count];
                let mut g_data = vec![0u16; px_count];
                let mut r_data = vec![0u16; px_count];

                for y in 0..(h as usize) {
                    let row = src_plane.row(y)?;
                    for x in 0..(w as usize) {
                        let idx = y * (w as usize) + x;
                        r_data[idx] = f32_to_f16(srgb_u8_to_linear(row[x * 3]));
                        g_data[idx] = f32_to_f16(srgb_u8_to_linear(row[x * 3 + 1]));
                        b_data[idx] = f32_to_f16(srgb_u8_to_linear(row[x * 3 + 2]));
                    }
                }

                for (name, buf) in [("B", b_data), ("G", g_data), ("R", r_data)] {
                    planes.push(ExrChannelPlane {
                        desc: ExrChannelDesc::new(name, ExrSampleType::Half),
                        width: w,
                        height: h,
                        data: ExrChannelData::Half(buf),
                    });
                }
            }
            PixelFormat::I400 => {
                let src_plane = frame
                    .plane(0)
                    .ok_or(ExrError::InvalidHeader("missing VideoFrame plane 0"))?;
                let mut y_data = vec![0u16; px_count];
                for y in 0..(h as usize) {
                    let row = src_plane.row(y)?;
                    for x in 0..(w as usize) {
                        y_data[y * (w as usize) + x] = f32_to_f16(srgb_u8_to_linear(row[x]));
                    }
                }
                planes.push(ExrChannelPlane {
                    desc: ExrChannelDesc::new("Y", ExrSampleType::Half),
                    width: w,
                    height: h,
                    data: ExrChannelData::Half(y_data),
                });
            }
            _ => {
                return Err(ExrError::InvalidHeader(
                    "unsupported VideoFrame PixelFormat for OpenEXR encoding",
                ));
            }
        }

        let part = ExrPart::from_planar_channels(w, h, self.config.compression, planes)?;
        let img = ExrImage::from_single_part(part);
        self.encode(&img)
    }
}

fn resolve_dwa_level(hdr: &ExrPartHeader, fallback: f32) -> f32 {
    for attr in &hdr.other_attributes {
        if attr.name == "dwaCompressionLevel"
            && let super::header::ExrAttributeValue::Float(lvl) = attr.value
        {
            return lvl;
        }
    }
    fallback
}

fn encode_part_chunks(
    part_idx: usize,
    is_multipart: bool,
    part: &ExrPart,
    default_dwa_level: f32,
    out: &mut Vec<u8>,
    chunk_offsets: &mut Vec<u64>,
) -> Result<(), ExrError> {
    let dwa_level = resolve_dwa_level(&part.header, default_dwa_level);
    match part.header.part_type {
        ExrPartType::ScanlineImage => {
            encode_flat_scanline_chunks(part_idx, is_multipart, part, dwa_level, out, chunk_offsets)
        }
        ExrPartType::TiledImage => {
            encode_flat_tile_chunks(part_idx, is_multipart, part, dwa_level, out, chunk_offsets)
        }
        ExrPartType::DeepScanline => {
            encode_deep_scanline_chunks(part_idx, is_multipart, part, out, chunk_offsets)
        }
        ExrPartType::DeepTile => {
            encode_deep_tile_chunks(part_idx, is_multipart, part, out, chunk_offsets)
        }
    }
}

fn encode_flat_scanline_chunks(
    part_idx: usize,
    is_multipart: bool,
    part: &ExrPart,
    dwa_level: f32,
    out: &mut Vec<u8>,
    chunk_offsets: &mut Vec<u64>,
) -> Result<(), ExrError> {
    let hdr = &part.header;
    let dw = hdr.data_window;
    let width = dw.width() as usize;
    let height = dw.height() as usize;
    let lines_per_chunk = hdr.compression.scanlines_per_chunk();
    let num_chunks = height.div_ceil(lines_per_chunk);

    let mut block_indices: Vec<usize> = (0..num_chunks).collect();
    if hdr.line_order == ExrLineOrder::DecreasingY {
        block_indices.reverse();
    }

    for blk_idx in block_indices {
        chunk_offsets.push(out.len() as u64);
        let y_start = dw.min_y + ((blk_idx * lines_per_chunk) as i32);
        let num_lines = (height - blk_idx * lines_per_chunk).min(lines_per_chunk);

        let uncompressed_size =
            compute_uncompressed_region_bytes(&hdr.channels, width, y_start, num_lines);
        let mut raw = Vec::with_capacity(uncompressed_size);

        for line in 0..num_lines {
            let cur_y = y_start + (line as i32);
            for (c_idx, ch) in hdr.channels.iter().enumerate() {
                let n = ch.samples_in_line(cur_y, width);
                if n == 0 {
                    continue;
                }
                let ys = ch.y_sampling.max(1);
                let src_row = ((cur_y - dw.min_y) / ys) as usize;
                let plane = &part.channels[c_idx];
                write_samples_from_plane(&plane.data, src_row * n, n, &mut raw);
            }
        }

        let packed = compress_chunk_data_with_dwa_level(
            hdr.compression,
            &raw,
            &hdr.channels,
            width,
            y_start,
            num_lines,
            dwa_level,
        )?;

        if is_multipart {
            out.extend_from_slice(&(part_idx as i32).to_le_bytes());
        }
        out.extend_from_slice(&y_start.to_le_bytes());
        out.extend_from_slice(&(packed.len() as i32).to_le_bytes());
        out.extend_from_slice(&packed);
    }

    Ok(())
}

fn encode_flat_tile_chunks(
    part_idx: usize,
    is_multipart: bool,
    part: &ExrPart,
    dwa_level: f32,
    out: &mut Vec<u8>,
    chunk_offsets: &mut Vec<u64>,
) -> Result<(), ExrError> {
    let hdr = &part.header;
    let td = hdr
        .tiles
        .ok_or(ExrError::InvalidHeader("missing tiles attribute"))?;
    let (nlx, nly) = td.level_counts(hdr.data_window);

    let mut levels = Vec::new();
    match td.level_mode {
        ExrLevelMode::OneLevel => levels.push((0u32, 0u32)),
        ExrLevelMode::MipmapLevels => {
            for l in 0..nlx {
                levels.push((l, l));
            }
        }
        ExrLevelMode::RipmapLevels => {
            for ly in 0..nly {
                for lx in 0..nlx {
                    levels.push((lx, ly));
                }
            }
        }
    }

    for (lx, ly) in levels {
        let (num_tx, num_ty) = td.tile_counts(hdr.data_window, lx, ly);
        let (lvl_w, lvl_h) = td.level_dimensions(hdr.data_window, lx, ly);

        let src_planes: &[ExrChannelPlane] = if lx == 0 && ly == 0 {
            &part.channels
        } else {
            let lvl_img = part
                .extra_levels
                .iter()
                .find(|l| l.level_x == lx && l.level_y == ly)
                .ok_or(ExrError::InvalidHeader(
                    "missing extra_levels entry for mipmap/ripmap level",
                ))?;
            &lvl_img.channels
        };

        let mut ty_list: Vec<u32> = (0..num_ty).collect();
        if hdr.line_order == ExrLineOrder::DecreasingY {
            ty_list.reverse();
        }

        for ty in ty_list {
            for tx in 0..num_tx {
                chunk_offsets.push(out.len() as u64);
                let px_x0 = (tx * td.x_size) as usize;
                let px_y0 = (ty * td.y_size) as usize;
                let tile_w = ((lvl_w - tx * td.x_size).min(td.x_size)) as usize;
                let tile_h = ((lvl_h - ty * td.y_size).min(td.y_size)) as usize;
                let y_start = hdr.data_window.min_y + (px_y0 as i32);

                let uncompressed_size =
                    compute_uncompressed_region_bytes(&hdr.channels, tile_w, y_start, tile_h);
                let mut raw = Vec::with_capacity(uncompressed_size);

                for row in 0..tile_h {
                    let src_y = px_y0 + row;
                    for plane in src_planes {
                        let src_offset = src_y * (lvl_w as usize) + px_x0;
                        write_samples_from_plane(&plane.data, src_offset, tile_w, &mut raw);
                    }
                }

                let packed = compress_chunk_data_with_dwa_level(
                    hdr.compression,
                    &raw,
                    &hdr.channels,
                    tile_w,
                    y_start,
                    tile_h,
                    dwa_level,
                )?;

                if is_multipart {
                    out.extend_from_slice(&(part_idx as i32).to_le_bytes());
                }
                out.extend_from_slice(&(tx as i32).to_le_bytes());
                out.extend_from_slice(&(ty as i32).to_le_bytes());
                out.extend_from_slice(&(lx as i32).to_le_bytes());
                out.extend_from_slice(&(ly as i32).to_le_bytes());
                out.extend_from_slice(&(packed.len() as i32).to_le_bytes());
                out.extend_from_slice(&packed);
            }
        }
    }

    Ok(())
}

fn encode_deep_scanline_chunks(
    part_idx: usize,
    is_multipart: bool,
    part: &ExrPart,
    out: &mut Vec<u8>,
    chunk_offsets: &mut Vec<u64>,
) -> Result<(), ExrError> {
    let hdr = &part.header;
    let deep = part
        .deep_samples
        .as_ref()
        .ok_or(ExrError::InvalidHeader("deep part missing deep_samples"))?;
    let dw = hdr.data_window;
    let width = dw.width() as usize;
    let height = dw.height() as usize;
    let lines_per_chunk = hdr.compression.scanlines_per_chunk();
    let num_chunks = height.div_ceil(lines_per_chunk);

    // Precompute sample start offset per scanline across the deep channels
    let mut line_sample_offsets = Vec::with_capacity(height);
    let mut running_samples = 0usize;
    for r in 0..height {
        line_sample_offsets.push(running_samples);
        let row_sum: usize = deep.sample_counts[r * width..(r + 1) * width]
            .iter()
            .map(|&c| c as usize)
            .sum();
        running_samples += row_sum;
    }

    for blk_idx in 0..num_chunks {
        chunk_offsets.push(out.len() as u64);
        let row0 = blk_idx * lines_per_chunk;
        let num_lines = (height - row0).min(lines_per_chunk);
        let y_start = dw.min_y + (row0 as i32);

        // Build per-scanline cumulative sample count table (`OpenEXRCore/decoding.c`)
        let mut count_bytes = Vec::with_capacity(num_lines * width * 4);
        for l in 0..num_lines {
            let mut cum = 0i32;
            let row_counts = &deep.sample_counts[(row0 + l) * width..(row0 + l + 1) * width];
            for &c in row_counts {
                cum += c as i32;
                count_bytes.extend_from_slice(&cum.to_le_bytes());
            }
        }

        // Build uncompressed deep sample payload
        let mut raw_samples = Vec::new();
        let mut row_sample_counts = Vec::with_capacity(num_lines);
        for l in 0..num_lines {
            let r = row0 + l;
            let s_off = line_sample_offsets[r];
            let s_cnt: usize = deep.sample_counts[r * width..(r + 1) * width]
                .iter()
                .map(|&c| c as usize)
                .sum();
            row_sample_counts.push(s_cnt);
            for ch_data in &deep.channels {
                write_samples_from_plane(ch_data, s_off, s_cnt, &mut raw_samples);
            }
        }

        let packed_counts = compress_chunk_data(
            hdr.compression,
            &count_bytes,
            &[],
            width,
            y_start,
            num_lines,
        )?;
        let packed_samples = compress_deep_samples_chunk_data(
            hdr.compression,
            &raw_samples,
            &hdr.channels,
            width,
            y_start,
            num_lines,
            &row_sample_counts,
        )?;

        if is_multipart {
            out.extend_from_slice(&(part_idx as i32).to_le_bytes());
        }
        out.extend_from_slice(&y_start.to_le_bytes());
        out.extend_from_slice(&(packed_counts.len() as u64).to_le_bytes());
        out.extend_from_slice(&(packed_samples.len() as u64).to_le_bytes());
        out.extend_from_slice(&(raw_samples.len() as u64).to_le_bytes());
        out.extend_from_slice(&packed_counts);
        out.extend_from_slice(&packed_samples);
    }

    Ok(())
}

fn encode_deep_tile_chunks(
    part_idx: usize,
    is_multipart: bool,
    part: &ExrPart,
    out: &mut Vec<u8>,
    chunk_offsets: &mut Vec<u64>,
) -> Result<(), ExrError> {
    let hdr = &part.header;
    let td = hdr
        .tiles
        .ok_or(ExrError::InvalidHeader("missing tiles attribute"))?;
    let deep = part.deep_samples.as_ref().ok_or(ExrError::InvalidHeader(
        "deep tile part missing deep_samples",
    ))?;

    let (num_tx, num_ty) = td.tile_counts(hdr.data_window, 0, 0);
    let (lvl_w, lvl_h) = td.level_dimensions(hdr.data_window, 0, 0);

    // Precompute prefix sums of sample_counts across the entire level 0 image so we can slice per-pixel
    let total_px = (lvl_w as usize) * (lvl_h as usize);
    let mut px_sample_offsets = Vec::with_capacity(total_px);
    let mut running = 0usize;
    for &c in &deep.sample_counts {
        px_sample_offsets.push(running);
        running += c as usize;
    }

    for ty in 0..num_ty {
        for tx in 0..num_tx {
            chunk_offsets.push(out.len() as u64);
            let px_x0 = (tx * td.x_size) as usize;
            let px_y0 = (ty * td.y_size) as usize;
            let tile_w = ((lvl_w - tx * td.x_size).min(td.x_size)) as usize;
            let tile_h = ((lvl_h - ty * td.y_size).min(td.y_size)) as usize;
            let y_start = hdr.data_window.min_y + (px_y0 as i32);

            let mut count_bytes = Vec::with_capacity(tile_w * tile_h * 4);
            let mut raw_samples = Vec::new();
            let mut row_sample_counts = Vec::with_capacity(tile_h);

            for row in 0..tile_h {
                let mut cum = 0i32;
                let py = px_y0 + row;
                let row_px_start = py * (lvl_w as usize) + px_x0;
                let row_counts = &deep.sample_counts[row_px_start..row_px_start + tile_w];
                for &c in row_counts {
                    cum += c as i32;
                    count_bytes.extend_from_slice(&cum.to_le_bytes());
                }
                let s_off = px_sample_offsets[row_px_start];
                let s_cnt: usize = row_counts.iter().map(|&c| c as usize).sum();
                row_sample_counts.push(s_cnt);
                for ch_data in &deep.channels {
                    write_samples_from_plane(ch_data, s_off, s_cnt, &mut raw_samples);
                }
            }

            let packed_counts =
                compress_chunk_data(hdr.compression, &count_bytes, &[], tile_w, y_start, tile_h)?;
            let packed_samples = compress_deep_samples_chunk_data(
                hdr.compression,
                &raw_samples,
                &hdr.channels,
                tile_w,
                y_start,
                tile_h,
                &row_sample_counts,
            )?;

            if is_multipart {
                out.extend_from_slice(&(part_idx as i32).to_le_bytes());
            }
            out.extend_from_slice(&(tx as i32).to_le_bytes());
            out.extend_from_slice(&(ty as i32).to_le_bytes());
            out.extend_from_slice(&0i32.to_le_bytes());
            out.extend_from_slice(&0i32.to_le_bytes());
            out.extend_from_slice(&(packed_counts.len() as u64).to_le_bytes());
            out.extend_from_slice(&(packed_samples.len() as u64).to_le_bytes());
            out.extend_from_slice(&(raw_samples.len() as u64).to_le_bytes());
            out.extend_from_slice(&packed_counts);
            out.extend_from_slice(&packed_samples);
        }
    }

    Ok(())
}

fn write_samples_from_plane(
    plane: &ExrChannelData,
    src_offset: usize,
    count: usize,
    out: &mut Vec<u8>,
) {
    match plane {
        ExrChannelData::Uint(src) => {
            for &v in &src[src_offset..src_offset + count] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        ExrChannelData::Half(src) => {
            for &v in &src[src_offset..src_offset + count] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        ExrChannelData::Float(src) => {
            for &v in &src[src_offset..src_offset + count] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
}

/// Converts an 8-bit sRGB value (`0..=255`) into linear scene-referred light (`[0.0, 1.0]`).
#[inline]
pub fn srgb_u8_to_linear(v: u8) -> f32 {
    let s = f32::from(v) / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

const _: ExrCompression = ExrCompression::None;
