// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! OpenEXR Luminance/Chroma (`Y` + `RY` + `BY`) ↔ `RGB` color conversion,
//! sub-sampled chroma bilinear upsampling and separable tent downsampling,
//! `displayWindow` vs `dataWindow` framing, and multi-resolution Mipmap/Ripmap
//! pyramid generation.
//!
//! Normative equations (`OpenEXR/ImfRgbaYca.cpp` & `docs/03-channel-types-and-pixel-formats.md`):
//! - Given luminance weights `Yw = [wr, wg, wb]` (from the `chromaticities` attribute's
//!   RGB → CIE XYZ row `Y`, or Rec. ITU-R BT.709 `[0.2126, 0.7152, 0.0722]` by default):
//!   ```text
//!   Y  = wr * R + wg * G + wb * B
//!   RY = (R - Y) / Y   (0 when Y == 0)
//!   BY = (B - Y) / Y   (0 when Y == 0)
//!   ```
//! - Exact inverse:
//!   ```text
//!   R = (RY + 1) * Y
//!   B = (BY + 1) * Y
//!   G = (Y - wr * R - wb * B) / wg
//!   ```

use rustmpeg_core::{
    Buffer, ChromaLocation, ColorDescription, ColorPrimaries, ColorRange, MatrixCoefficients,
    PixelFormat, Plane, TransferFunction, VideoFrame,
};

use super::decoder::linear_to_srgb_u8;
use super::half::{f16_to_f32, f32_to_f16};
use super::header::{
    ExrAttribute, ExrAttributeValue, ExrChromaticities, ExrLevelMode, ExrLevelRoundingMode,
    ExrPartType, ExrSampleType, ExrTileDesc,
};
use super::{ExrChannelData, ExrChannelPlane, ExrError, ExrLevelImage, ExrPart};

/// Rec. ITU-R BT.709 primaries with D65 white point (default OpenEXR chromaticities).
pub const BT709_CHROMATICITIES: ExrChromaticities = ExrChromaticities {
    red: (0.6400, 0.3300),
    green: (0.3000, 0.6000),
    blue: (0.1500, 0.0600),
    white: (0.3127, 0.3290),
};

/// Computes the luminance weights `[wr, wg, wb]` (the `Y` row of the RGB → CIE XYZ matrix)
/// for the given [`ExrChromaticities`] via Cramer's rule in `f64`.
/// Falls back to [`BT709_CHROMATICITIES`] if the primary triangle is degenerate.
#[must_use]
pub fn luminance_weights(c: &ExrChromaticities) -> [f32; 3] {
    fn column(x: f32, y: f32) -> Option<[f64; 3]> {
        let (x, y) = (f64::from(x), f64::from(y));
        if y == 0.0 || !y.is_finite() || !x.is_finite() {
            return None;
        }
        Some([x / y, 1.0, (1.0 - x - y) / y])
    }
    let solve = || -> Option<[f32; 3]> {
        let r = column(c.red.0, c.red.1)?;
        let g = column(c.green.0, c.green.1)?;
        let b = column(c.blue.0, c.blue.1)?;
        let w = column(c.white.0, c.white.1)?;
        let det = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| -> f64 {
            a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1])
                + c[0] * (a[1] * b[2] - a[2] * b[1])
        };
        let d = det(r, g, b);
        if d == 0.0 || !d.is_finite() {
            return None;
        }
        let s = [det(w, g, b) / d, det(r, w, b) / d, det(r, g, w) / d];
        if s.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some([s[0] as f32, s[1] as f32, s[2] as f32])
    };
    solve().unwrap_or_else(|| luminance_weights(&BT709_CHROMATICITIES))
}

/// Extracts the `chromaticities` attribute from an attribute slice if present.
#[must_use]
pub fn chromaticities_of(attrs: &[ExrAttribute]) -> Option<ExrChromaticities> {
    attrs
        .iter()
        .find_map(|a| match (a.name.as_str(), &a.value) {
            ("chromaticities", ExrAttributeValue::Chromaticities(c)) => Some(*c),
            _ => None,
        })
}

/// Returns the luminance weights for a part's header attributes (`chromaticities` or BT.709 default).
#[must_use]
pub fn luminance_weights_of(attrs: &[ExrAttribute]) -> [f32; 3] {
    luminance_weights(&chromaticities_of(attrs).unwrap_or(BT709_CHROMATICITIES))
}

/// Reconstructs a full-resolution `width * height` plane from a sub-sampled plane
/// (`pw = width / sx`, `ph = height / sy`) using bilinear interpolation.
pub fn upsample_bilinear(
    sub: &[f32],
    width: u32,
    height: u32,
    sx: u32,
    sy: u32,
) -> Result<Vec<f32>, ExrError> {
    let (sx, sy) = (sx.max(1) as usize, sy.max(1) as usize);
    let (w, h) = (width as usize, height as usize);
    let pw = (w / sx).max(usize::from(w > 0));
    let ph = (h / sy).max(usize::from(h > 0));
    if sub.len() != pw * ph {
        return Err(ExrError::InvalidHeader(
            "sub-sampled chroma plane sample count does not match dimensions",
        ));
    }
    if sx == 1 && sy == 1 {
        return Ok(sub.to_vec());
    }
    let mut out = vec![0.0f32; w * h];
    let (sxf, syf) = (sx as f32, sy as f32);
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(ph);
    for jy in 0..ph {
        let src = &sub[jy * pw..(jy + 1) * pw];
        let mut row = vec![0.0f32; w];
        for (x, v) in row.iter_mut().enumerate() {
            let i0 = (x / sx).min(pw - 1);
            let i1 = (i0 + 1).min(pw - 1);
            let t = ((x.saturating_sub(i0 * sx)) as f32 / sxf).clamp(0.0, 1.0);
            *v = src[i0] + (src[i1] - src[i0]) * t;
        }
        rows.push(row);
    }
    for y in 0..h {
        let j0 = (y / sy).min(ph - 1);
        let j1 = (j0 + 1).min(ph - 1);
        let t = ((y.saturating_sub(j0 * sy)) as f32 / syf).clamp(0.0, 1.0);
        let (r0, r1) = (&rows[j0], &rows[j1]);
        let dst = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            dst[x] = r0[x] + (r1[x] - r0[x]) * t;
        }
    }
    Ok(out)
}

/// Downsamples a full-resolution `width * height` plane to `(sx, sy)` sub-sampling
/// using a symmetric separable tent filter.
pub fn downsample_tent(
    full: &[f32],
    width: u32,
    height: u32,
    sx: u32,
    sy: u32,
) -> Result<Vec<f32>, ExrError> {
    let (sx, sy) = (sx.max(1) as usize, sy.max(1) as usize);
    let (w, h) = (width as usize, height as usize);
    if full.len() != w * h {
        return Err(ExrError::InvalidHeader(
            "full-resolution plane sample count does not match width * height",
        ));
    }
    if sx == 1 && sy == 1 {
        return Ok(full.to_vec());
    }
    let pw = w / sx;
    let ph = h / sy;
    let (sxi, syi) = (sx as isize, sy as isize);

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(h);
    for y in 0..h {
        let src = &full[y * w..(y + 1) * w];
        let mut row = Vec::with_capacity(pw);
        for jx in 0..pw {
            let cx = (jx * sx) as isize;
            let (mut acc, mut wsum) = (0.0f64, 0.0f64);
            for dx in (1 - sxi)..sxi {
                let x = cx + dx;
                if x < 0 || x >= (w as isize) {
                    continue;
                }
                let wt = (sxi - dx.abs()) as f64;
                acc += f64::from(src[x as usize]) * wt;
                wsum += wt;
            }
            row.push((acc / wsum) as f32);
        }
        rows.push(row);
    }

    let mut out = Vec::with_capacity(pw * ph);
    for jy in 0..ph {
        let cy = (jy * sy) as isize;
        let mut acc = vec![0.0f64; pw];
        let mut wsum = 0.0f64;
        for dy in (1 - syi)..syi {
            let y = cy + dy;
            if y < 0 || y >= (h as isize) {
                continue;
            }
            let wt = (syi - dy.abs()) as f64;
            for (a, &v) in acc.iter_mut().zip(&rows[y as usize]) {
                *a += f64::from(v) * wt;
            }
            wsum += wt;
        }
        out.extend(acc.iter().map(|&a| (a / wsum) as f32));
    }
    Ok(out)
}

/// Borrowed chroma plane with its horizontal and vertical sub-sampling factors.
#[derive(Debug, Clone, Copy)]
pub struct ExrChromaSlice<'a> {
    /// Sub-sampled chroma samples.
    pub samples: &'a [f32],
    /// Horizontal sub-sampling step (`x_sampling`, typically `2`).
    pub x_sampling: u32,
    /// Vertical sub-sampling step (`y_sampling`, typically `2`).
    pub y_sampling: u32,
}

/// Full-resolution linear `R`, `G`, `B` planes reconstructed from `Y` + `RY` + `BY`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExrRgbPlanes {
    /// Linear red samples (`width * height`).
    pub r: Vec<f32>,
    /// Linear green samples (`width * height`).
    pub g: Vec<f32>,
    /// Linear blue samples (`width * height`).
    pub b: Vec<f32>,
}

/// Luminance (`Y`) and sub-sampled chroma (`RY`, `BY`) planes produced from linear `RGB`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExrLumaChromaPlanes {
    /// Full-resolution luminance plane (`width * height`).
    pub y: Vec<f32>,
    /// Sub-sampled `RY` plane (`(width / x_sampling) * (height / y_sampling)`).
    pub ry: Vec<f32>,
    /// Sub-sampled `BY` plane (`(width / x_sampling) * (height / y_sampling)`).
    pub by: Vec<f32>,
    /// Horizontal sub-sampling factor.
    pub x_sampling: u32,
    /// Vertical sub-sampling factor.
    pub y_sampling: u32,
}

/// Reconstructs full-resolution `R`, `G`, `B` planes from full-resolution `Y` and
/// sub-sampled `RY` / `BY` chroma planes.
pub fn luma_chroma_to_rgb(
    width: u32,
    height: u32,
    y: &[f32],
    ry: ExrChromaSlice<'_>,
    by: ExrChromaSlice<'_>,
    weights: [f32; 3],
) -> Result<ExrRgbPlanes, ExrError> {
    let pixels = (width as usize) * (height as usize);
    if y.len() != pixels {
        return Err(ExrError::InvalidHeader(
            "Y plane sample count does not match width * height",
        ));
    }
    let ry_full = upsample_bilinear(ry.samples, width, height, ry.x_sampling, ry.y_sampling)?;
    let by_full = upsample_bilinear(by.samples, width, height, by.x_sampling, by.y_sampling)?;
    let [wr, wg, wb] = weights;
    let mut r = Vec::with_capacity(pixels);
    let mut g = Vec::with_capacity(pixels);
    let mut b = Vec::with_capacity(pixels);
    for i in 0..pixels {
        let yy = y[i];
        let rr = (ry_full[i] + 1.0) * yy;
        let bb = (by_full[i] + 1.0) * yy;
        r.push(rr);
        b.push(bb);
        g.push((yy - wr * rr - wb * bb) / wg);
    }
    Ok(ExrRgbPlanes { r, g, b })
}

/// Converts full-resolution linear `R`, `G`, `B` planes into a full-resolution `Y` plane
/// and tent-downsampled `RY` / `BY` chroma planes at `sampling = (x_sampling, y_sampling)`.
pub fn rgb_to_luma_chroma(
    width: u32,
    height: u32,
    rgb: [&[f32]; 3],
    weights: [f32; 3],
    sampling: (u32, u32),
) -> Result<ExrLumaChromaPlanes, ExrError> {
    let [r, g, b] = rgb;
    let (x_sampling, y_sampling) = (sampling.0.max(1), sampling.1.max(1));
    let pixels = (width as usize) * (height as usize);
    if r.len() != pixels || g.len() != pixels || b.len() != pixels {
        return Err(ExrError::InvalidHeader(
            "RGB input plane length does not match width * height",
        ));
    }
    let [wr, wg, wb] = weights;
    let mut y = Vec::with_capacity(pixels);
    let mut ry_full = Vec::with_capacity(pixels);
    let mut by_full = Vec::with_capacity(pixels);
    for i in 0..pixels {
        let yy = wr * r[i] + wg * g[i] + wb * b[i];
        y.push(yy);
        if yy == 0.0 || !yy.is_finite() {
            ry_full.push(0.0);
            by_full.push(0.0);
        } else {
            ry_full.push((r[i] - yy) / yy);
            by_full.push((b[i] - yy) / yy);
        }
    }
    let ry = downsample_tent(&ry_full, width, height, x_sampling, y_sampling)?;
    let by = downsample_tent(&by_full, width, height, x_sampling, y_sampling)?;
    Ok(ExrLumaChromaPlanes {
        y,
        ry,
        by,
        x_sampling,
        y_sampling,
    })
}

impl ExrChannelPlane {
    /// Collects all samples in this channel plane as `f32`.
    #[must_use]
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match &self.data {
            ExrChannelData::Uint(v) => v.iter().map(|&x| x as f32).collect(),
            ExrChannelData::Half(v) => v.iter().map(|&x| f16_to_f32(x)).collect(),
            ExrChannelData::Float(v) => v.clone(),
        }
    }
}

impl ExrPart {
    /// Reconstructs linear `[R, G, B]` `f32` planes from either direct `"R"`, `"G"`, `"B"` channels
    /// or Luminance/Chroma `"Y"`, `"RY"`, `"BY"` channels (using the part's `chromaticities` attribute).
    pub fn reconstruct_rgb_f32(&self) -> Result<Option<ExrRgbPlanes>, ExrError> {
        let w = self.header.data_window.width();
        let h = self.header.data_window.height();
        if let (Some(r), Some(g), Some(b)) = (
            self.channel_by_name("R"),
            self.channel_by_name("G"),
            self.channel_by_name("B"),
        ) {
            return Ok(Some(ExrRgbPlanes {
                r: r.to_f32_vec(),
                g: g.to_f32_vec(),
                b: b.to_f32_vec(),
            }));
        }
        if let (Some(y_ch), Some(ry_ch), Some(by_ch)) = (
            self.channel_by_name("Y"),
            self.channel_by_name("RY"),
            self.channel_by_name("BY"),
        ) {
            let weights = luminance_weights_of(&self.header.other_attributes);
            let y_vec = y_ch.to_f32_vec();
            let ry_vec = ry_ch.to_f32_vec();
            let by_vec = by_ch.to_f32_vec();
            let planes = luma_chroma_to_rgb(
                w,
                h,
                &y_vec,
                ExrChromaSlice {
                    samples: &ry_vec,
                    x_sampling: ry_ch.desc.x_sampling.max(1) as u32,
                    y_sampling: ry_ch.desc.y_sampling.max(1) as u32,
                },
                ExrChromaSlice {
                    samples: &by_vec,
                    x_sampling: by_ch.desc.x_sampling.max(1) as u32,
                    y_sampling: by_ch.desc.y_sampling.max(1) as u32,
                },
                weights,
            )?;
            return Ok(Some(planes));
        }
        Ok(None)
    }

    /// Converts this flat OpenEXR part into an 8-bit [`VideoFrame`] framed to `header.display_window`
    /// rather than `header.data_window`:
    /// - Pixels inside `display_window` that fall outside `data_window` are zero-filled (`0, 0, 0, 0`).
    /// - Pixels inside `data_window` that fall outside `display_window` are cropped.
    pub fn to_display_video_frame(&self) -> Result<VideoFrame, ExrError> {
        let data_frame = self.to_video_frame()?;
        let disp = self.header.display_window;
        let data = self.header.data_window;
        if disp == data {
            return Ok(data_frame);
        }
        let disp_w = disp.width();
        let disp_h = disp.height();
        let data_w = data.width() as i32;
        let data_h = data.height() as i32;
        let bpp = match data_frame.format {
            PixelFormat::Rgba8 => 4usize,
            PixelFormat::Rgb8 => 3usize,
            PixelFormat::I400 => 1usize,
            _ => {
                return Err(ExrError::InvalidHeader(
                    "unsupported PixelFormat in to_display_video_frame",
                ));
            }
        };

        let src_plane = data_frame
            .plane(0)
            .ok_or(ExrError::InvalidHeader("missing source plane 0"))?;
        let mut dst = vec![0u8; (disp_w as usize) * (disp_h as usize) * bpp];

        for dy in 0..(disp_h as i32) {
            let world_y = disp.min_y + dy;
            let sy = world_y - data.min_y;
            if sy < 0 || sy >= data_h {
                continue;
            }
            let src_row = src_plane.row(sy as usize)?;
            let dst_row_off = (dy as usize) * (disp_w as usize) * bpp;
            for dx in 0..(disp_w as i32) {
                let world_x = disp.min_x + dx;
                let sx = world_x - data.min_x;
                if sx < 0 || sx >= data_w {
                    continue;
                }
                let s_off = (sx as usize) * bpp;
                let d_off = dst_row_off + (dx as usize) * bpp;
                dst[d_off..d_off + bpp].copy_from_slice(&src_row[s_off..s_off + bpp]);
            }
        }

        let srgb_color = ColorDescription {
            range: ColorRange::Full,
            primaries: ColorPrimaries::Bt709,
            transfer: TransferFunction::Srgb,
            matrix: MatrixCoefficients::Identity,
            chroma_location: ChromaLocation::Left,
        };
        let plane0 = Plane::new(
            Buffer::from_vec(dst),
            (disp_w as usize) * bpp,
            disp_w,
            disp_h,
        );
        let _ = linear_to_srgb_u8;
        Ok(VideoFrame::from_planes(
            data_frame.format,
            8,
            disp_w,
            disp_h,
            srgb_color,
            [plane0, Plane::empty(), Plane::empty()],
        )?)
    }

    /// Generates all `MipmapLevels` resolution levels (`(1, 1) .. (nlx - 1, nlx - 1)`)
    /// from the level `(0, 0)` channels using 2x2 box-filter averaging and populates
    /// `self.extra_levels` and `self.header.tiles`.
    pub fn generate_mipmaps(
        &mut self,
        tile_x_size: u32,
        tile_y_size: u32,
        rounding_mode: ExrLevelRoundingMode,
    ) -> Result<(), ExrError> {
        let td = ExrTileDesc {
            x_size: tile_x_size.max(1),
            y_size: tile_y_size.max(1),
            level_mode: ExrLevelMode::MipmapLevels,
            rounding_mode,
        };
        self.header.part_type = ExrPartType::TiledImage;
        self.header.tiles = Some(td);
        let (nlx, _) = td.level_counts(self.header.data_window);
        self.extra_levels.clear();

        for l in 1..nlx {
            let (lw, lh) = td.level_dimensions(self.header.data_window, l, l);
            let prev_channels = if l == 1 {
                &self.channels
            } else {
                &self.extra_levels.last().unwrap().channels
            };
            let mut next_channels = Vec::with_capacity(prev_channels.len());
            for prev_ch in prev_channels {
                next_channels.push(downsample_channel_plane(prev_ch, lw, lh));
            }
            self.extra_levels.push(ExrLevelImage {
                level_x: l,
                level_y: l,
                width: lw,
                height: lh,
                channels: next_channels,
            });
        }
        Ok(())
    }

    /// Generates all `RipmapLevels` resolution levels (`(lx, ly) != (0, 0)`)
    /// from the level `(0, 0)` channels and populates `self.extra_levels` and `self.header.tiles`.
    pub fn generate_ripmaps(
        &mut self,
        tile_x_size: u32,
        tile_y_size: u32,
        rounding_mode: ExrLevelRoundingMode,
    ) -> Result<(), ExrError> {
        let td = ExrTileDesc {
            x_size: tile_x_size.max(1),
            y_size: tile_y_size.max(1),
            level_mode: ExrLevelMode::RipmapLevels,
            rounding_mode,
        };
        self.header.part_type = ExrPartType::TiledImage;
        self.header.tiles = Some(td);
        let (nlx, nly) = td.level_counts(self.header.data_window);
        self.extra_levels.clear();

        for ly in 0..nly {
            for lx in 0..nlx {
                if lx == 0 && ly == 0 {
                    continue;
                }
                let (lw, lh) = td.level_dimensions(self.header.data_window, lx, ly);
                let src_channels: &[ExrChannelPlane] = if lx > 0 {
                    if lx == 1 && ly == 0 {
                        &self.channels
                    } else {
                        let prev = self
                            .extra_levels
                            .iter()
                            .find(|img| img.level_x == lx - 1 && img.level_y == ly)
                            .unwrap();
                        &prev.channels
                    }
                } else if ly == 1 {
                    &self.channels
                } else {
                    let prev = self
                        .extra_levels
                        .iter()
                        .find(|img| img.level_x == 0 && img.level_y == ly - 1)
                        .unwrap();
                    &prev.channels
                };
                let mut next_channels = Vec::with_capacity(src_channels.len());
                for prev_ch in src_channels {
                    next_channels.push(downsample_channel_plane(prev_ch, lw, lh));
                }
                self.extra_levels.push(ExrLevelImage {
                    level_x: lx,
                    level_y: ly,
                    width: lw,
                    height: lh,
                    channels: next_channels,
                });
            }
        }
        Ok(())
    }
}

fn downsample_channel_plane(src: &ExrChannelPlane, dst_w: u32, dst_h: u32) -> ExrChannelPlane {
    let sw = src.width as usize;
    let sh = src.height as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;

    let sample_at = |x: usize, y: usize| -> f32 {
        let cx = x.min(sw.saturating_sub(1));
        let cy = y.min(sh.saturating_sub(1));
        src.sample_as_f32(cy * sw + cx)
    };

    let count = dw * dh;
    let mut avg = Vec::with_capacity(count);
    let rx = sw > dw;
    let ry = sh > dh;
    for y in 0..dh {
        let sy0 = if ry { y * 2 } else { y };
        let sy1 = if ry { (y * 2 + 1).min(sh - 1) } else { sy0 };
        for x in 0..dw {
            let sx0 = if rx { x * 2 } else { x };
            let sx1 = if rx { (x * 2 + 1).min(sw - 1) } else { sx0 };
            let v = 0.25
                * (sample_at(sx0, sy0)
                    + sample_at(sx1, sy0)
                    + sample_at(sx0, sy1)
                    + sample_at(sx1, sy1));
            avg.push(v);
        }
    }

    let data = match src.desc.sample_type {
        ExrSampleType::Uint => {
            ExrChannelData::Uint(avg.into_iter().map(|v| v.max(0.0).round() as u32).collect())
        }
        ExrSampleType::Half => ExrChannelData::Half(avg.into_iter().map(f32_to_f16).collect()),
        ExrSampleType::Float => ExrChannelData::Float(avg),
    };

    ExrChannelPlane {
        desc: src.desc.clone(),
        width: dst_w,
        height: dst_h,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bt709_luminance_weights() {
        let w = luminance_weights(&BT709_CHROMATICITIES);
        assert!((w[0] - 0.2126).abs() < 2e-4);
        assert!((w[1] - 0.7152).abs() < 2e-4);
        assert!((w[2] - 0.0722).abs() < 2e-4);
        assert!((w[0] + w[1] + w[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_luma_chroma_roundtrip_1x1_and_2x2() {
        let w = 8u32;
        let h = 8u32;
        let weights = luminance_weights(&BT709_CHROMATICITIES);
        let r = vec![0.6f32; 64];
        let g = vec![0.4f32; 64];
        let b = vec![0.2f32; 64];

        let lc = rgb_to_luma_chroma(w, h, [&r, &g, &b], weights, (2, 2)).unwrap();
        assert_eq!(lc.y.len(), 64);
        assert_eq!(lc.ry.len(), 16);
        assert_eq!(lc.by.len(), 16);

        let rgb = luma_chroma_to_rgb(
            w,
            h,
            &lc.y,
            ExrChromaSlice {
                samples: &lc.ry,
                x_sampling: 2,
                y_sampling: 2,
            },
            ExrChromaSlice {
                samples: &lc.by,
                x_sampling: 2,
                y_sampling: 2,
            },
            weights,
        )
        .unwrap();

        for i in 0..64 {
            assert!((rgb.r[i] - r[i]).abs() < 1e-5);
            assert!((rgb.g[i] - g[i]).abs() < 1e-5);
            assert!((rgb.b[i] - b[i]).abs() < 1e-5);
        }
    }
}
