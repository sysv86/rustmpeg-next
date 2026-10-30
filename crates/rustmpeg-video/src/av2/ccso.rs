// SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause
// Copyright 2026 The rustmpeg-next Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Clean-room AV2 Cross-Component Sample Offset (`ccso.rs`, AV2 Specification v1.0.0 §5.18.7.12, §5.20.10.2, §6.18.7.8, §7.19, §8.3).
//!
//! Classifies reconstructed samples using luma guidance (edge/band offsets), solves for the
//! sample offset per band, and applies normative CCSO offsets after CDEF.

use crate::av2::frame::{Frame, Plane};
use crate::av2::headers::{FrameHeader, PixelLayout, SequenceHeader};
use crate::av2::pixel::{BitDepth, Pixel};

/// Number of CCSO classification bands (`16`).
pub const N_CCSO_BANDS: usize = 16;

/// Normative CCSO direction offsets (AV2 Spec v1.0.0 §8.3 / `Ccso_Pos`).
pub const CCSO_POS: [[i8; 2]; 7] = [
    [-1, 0],
    [0, -1],
    [-1, -1],
    [-1, 1],
    [-1, -2],
    [1, -2],
    [0, 2],
];

/// Normative CCSO quantization step sizes (AV2 Spec v1.0.0 §8.3 / `CCSO_Quant_Sz`).
pub const CCSO_QUANT_SZ: [[u16; 4]; 4] = [
    [16, 8, 32, 0],
    [56, 40, 64, 128],
    [48, 24, 96, 192],
    [80, 112, 160, 256],
];

/// Normative CCSO offset lookup table (AV2 Spec v1.0.0 §5.18.7.12 / §8.3 / `Ccso_Offset`).
pub const CCSO_OFFSET: [[i8; 8]; 4] = [
    [0, 1, -1, 3, -3, 7, -7, -10],
    [0, 2, -2, 6, -6, 14, -14, -20],
    [0, 3, -3, 9, -9, 21, -21, -30],
    [0, 4, -4, 12, -12, 28, -28, -40],
];

/// Classifies a directional difference into one of 3 edge classes (AV2 Spec v1.0.0 §7.19.1).
#[inline]
pub fn ccso_score(diff: i32, quant_step: i32, edge_classifier: u8) -> usize {
    if diff > quant_step && edge_classifier == 0 {
        2
    } else if diff < -quant_step {
        0
    } else {
        1
    }
}

/// Samples luma pixel with clamping to specified tile/frame bounds (AV2 Spec v1.0.0 §7.19.1).
#[inline]
pub fn get_ccso_luma<P: Pixel>(
    luma: &Plane<P>,
    x: i32,
    y: i32,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
) -> i32 {
    let cx = x.clamp(min_x, max_x) as usize;
    let cy = y.clamp(min_y, max_y) as usize;
    luma.data[cy * luma.stride + cx].to_i32()
}

/// Applies normative AV2 Cross-Component Sample Offset (`CCSO`) to all enabled planes
/// (AV2 Spec v1.0.0 §7.19).
pub fn ccso_frame_normative<BD: BitDepth>(
    bd: BD,
    rec: &mut Frame<BD::Pixel>,
    orig_luma: &Plane<BD::Pixel>,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    ccso_blks: &[Vec<bool>; 3],
) {
    if fh.ccso.enabled == 0 {
        return;
    }
    let n_planes = if seq.layout == PixelLayout::I400 {
        1
    } else {
        3
    };

    let f_bw4 = 2 * ((fh.width.max(0) as usize + 7) >> 3);
    let f_bh4 = 2 * ((fh.height.max(0) as usize + 7) >> 3);
    let unit_size = 64 * fh.ccso.b64size.max(1) as usize;
    let ccso_step4 = 16 * fh.ccso.b64size.max(1) as usize;
    let ccso_cols = f_bw4.div_ceil(ccso_step4);
    let ccso_rows = f_bh4.div_ceil(ccso_step4);

    let n_cols = (fh.tiling.t.cols.max(1) as usize).max(1);
    let n_rows = (fh.tiling.t.rows.max(1) as usize).max(1);
    let sb_step4 = if fh.sb128 != 0 { 32usize } else { 16usize };

    let mut col_starts4 = Vec::with_capacity(n_cols + 1);
    for tc in 0..n_cols {
        col_starts4.push(fh.tiling.t.col_start_sb[tc] as usize * sb_step4);
    }
    col_starts4.push(f_bw4);

    let mut row_starts4 = Vec::with_capacity(n_rows + 1);
    for tr in 0..n_rows {
        row_starts4.push(fh.tiling.t.row_start_sb[tr] as usize * sb_step4);
    }
    row_starts4.push(f_bh4);

    for pl in 0..n_planes.min(rec.planes.len()) {
        let p_hdr = &fh.ccso.planes[pl];
        if p_hdr.enabled == 0 {
            continue;
        }

        let ss_x = rec.planes[pl].ss_x;
        let ss_y = rec.planes[pl].ss_y;
        let blk_w = unit_size >> ss_x;
        let blk_h = unit_size >> ss_y;

        let scale_idx = (p_hdr.scale_idx as usize) & 3;
        let quant_idx = (p_hdr.quant_idx as usize) & 3;
        let quant_step = CCSO_QUANT_SZ[scale_idx][quant_idx];

        let ext_filter = (p_hdr.ext_filter_support as usize).min(6);
        let dy = CCSO_POS[ext_filter][0] as i32;
        let dx = CCSO_POS[ext_filter][1] as i32;

        let shift = (bd.bitdepth() as u32).saturating_sub(p_hdr.max_band_log2 as u32);
        let stride = rec.planes[pl].stride;

        for uby in 0..ccso_rows {
            let y_start = uby * blk_h;
            if y_start >= rec.planes[pl].height {
                break;
            }
            let y_end = (y_start + blk_h).min(rec.planes[pl].height);

            for ubx in 0..ccso_cols {
                let blk_idx = uby * ccso_cols + ubx;
                if blk_idx >= ccso_blks[pl].len() || !ccso_blks[pl][blk_idx] {
                    continue;
                }
                let x_start = ubx * blk_w;
                if x_start >= rec.planes[pl].width {
                    break;
                }
                let x_end = (x_start + blk_w).min(rec.planes[pl].width);

                let (min_x, max_x, min_y, max_y) = if seq.disable_loopfilters_across_tiles {
                    let x_luma_start = ubx * unit_size;
                    let y_luma_start = uby * unit_size;
                    let tc = col_starts4
                        .partition_point(|&s| s <= (x_luma_start / 4))
                        .saturating_sub(1);
                    let tr = row_starts4
                        .partition_point(|&s| s <= (y_luma_start / 4))
                        .saturating_sub(1);
                    let min_x = (col_starts4[tc] * 4) as i32;
                    let max_x =
                        ((col_starts4[tc + 1] * 4).min(orig_luma.width) as i32 - 1).max(min_x);
                    let min_y = (row_starts4[tr] * 4) as i32;
                    let max_y =
                        ((row_starts4[tr + 1] * 4).min(orig_luma.height) as i32 - 1).max(min_y);
                    (min_x, max_x, min_y, max_y)
                } else {
                    (
                        0,
                        (orig_luma.width as i32 - 1).max(0),
                        0,
                        (orig_luma.height as i32 - 1).max(0),
                    )
                };

                for y in y_start..y_end {
                    let y_luma = (y << ss_y) as i32;
                    let row_off = y * stride;
                    for x in x_start..x_end {
                        let x_luma = (x << ss_x) as i32;
                        let c =
                            get_ccso_luma(orig_luma, x_luma, y_luma, min_x, max_x, min_y, max_y);
                        let band = (c >> shift) as usize;

                        let (cls0, cls1) = if p_hdr.bo_only != 0 {
                            (0, 0)
                        } else {
                            let p0 = get_ccso_luma(
                                orig_luma,
                                x_luma + dx,
                                y_luma + dy,
                                min_x,
                                max_x,
                                min_y,
                                max_y,
                            );
                            let p1 = get_ccso_luma(
                                orig_luma,
                                x_luma - dx,
                                y_luma - dy,
                                min_x,
                                max_x,
                                min_y,
                                max_y,
                            );
                            (
                                ccso_score(p0 - c, quant_step as i32, p_hdr.edge_clf),
                                ccso_score(p1 - c, quant_step as i32, p_hdr.edge_clf),
                            )
                        };

                        let byte_idx = cls0 * 16 + cls1 * 4 + (band >> 1);
                        let nibble = (p_hdr.filter_off[byte_idx] >> (4 * (band & 1))) & 0x7;
                        let offset = CCSO_OFFSET[scale_idx][nibble as usize];

                        let cur = rec.planes[pl].data[row_off + x].to_i32();
                        rec.planes[pl].data[row_off + x] = bd.clip_pixel(cur + offset as i32);
                    }
                }
            }
        }
    }
}

/// One candidate frame-level CCSO parameter set tried by the encoder.
#[derive(Clone, Copy, Debug, Default)]
struct CcsoCandidate {
    bo_only: bool,
    scale_idx: u8,
    quant_idx: u8,
    ext_filter: u8,
    edge_clf: u8,
    max_band_log2: u8,
}

/// Classifies one sample with the normative CCSO band/edge classifier
/// (AV2 Spec v1.0.0 §7.19.1). `dx`/`dy` are zero for band-offset-only mode.
#[inline]
fn ccso_class<P: Pixel>(
    luma: &Plane<P>,
    x_luma: i32,
    y_luma: i32,
    dx: i32,
    dy: i32,
    quant_step: i32,
    edge_clf: u8,
    shift: u32,
    max_x: i32,
    max_y: i32,
) -> (usize, usize, usize) {
    let sample = |x: i32, y: i32| -> i32 {
        let cx = x.clamp(0, max_x) as usize;
        let cy = y.clamp(0, max_y) as usize;
        luma.data[cy * luma.stride + cx].to_i32()
    };
    let c = sample(x_luma, y_luma);
    let band = (c >> shift) as usize;
    if dx == 0 && dy == 0 {
        (band, 0, 0)
    } else {
        (
            band,
            ccso_score(sample(x_luma + dx, y_luma + dy) - c, quant_step, edge_clf),
            ccso_score(sample(x_luma - dx, y_luma - dy) - c, quant_step, edge_clf),
        )
    }
}

/// Solves the per-class offsets for one candidate parameter set and returns the packed
/// `filter_off` nibbles together with the resulting SSE over the enabled units.
fn ccso_trial<BD: BitDepth>(
    bd: BD,
    src_plane: &Plane<BD::Pixel>,
    rec_plane: &Plane<BD::Pixel>,
    luma: &Plane<BD::Pixel>,
    cand: CcsoCandidate,
    units: &[bool],
    unit_cols: usize,
    unit_rows: usize,
    unit_luma: usize,
) -> ([u8; 64], u64) {
    let scale_idx = (cand.scale_idx as usize) & 3;
    let quant_idx = (cand.quant_idx as usize) & 3;
    let quant_step = CCSO_QUANT_SZ[scale_idx][quant_idx] as i32;
    let (dx, dy) = if cand.bo_only {
        (0, 0)
    } else {
        let ext = (cand.ext_filter as usize).min(6);
        (CCSO_POS[ext][1] as i32, CCSO_POS[ext][0] as i32)
    };
    let shift = (bd.bitdepth() as u32).saturating_sub(cand.max_band_log2 as u32);
    let max_band = 1usize << (cand.max_band_log2 as usize).min(3);

    let mut sum = [[[0i64; 3]; 3]; 8];
    let mut count = [[[0u64; 3]; 3]; 8];

    let unit_w = (unit_luma >> rec_plane.ss_x).max(1);
    let unit_h = (unit_luma >> rec_plane.ss_y).max(1);
    let max_x = rec_plane.width.saturating_sub(1) as i32;
    let max_y = rec_plane.height.saturating_sub(1) as i32;

    for ur in 0..unit_rows {
        let y0 = ur * unit_h;
        if y0 >= rec_plane.height {
            break;
        }
        let y1 = (y0 + unit_h).min(rec_plane.height);
        for uc in 0..unit_cols {
            if !units.get(ur * unit_cols + uc).copied().unwrap_or(false) {
                continue;
            }
            let x0 = uc * unit_w;
            if x0 >= rec_plane.width {
                break;
            }
            let x1 = (x0 + unit_w).min(rec_plane.width);
            for y in y0..y1 {
                let s_row = y * src_plane.stride;
                let r_row = y * rec_plane.stride;
                let y_luma = (y << rec_plane.ss_y) as i32;
                for x in x0..x1 {
                    let x_luma = (x << rec_plane.ss_x) as i32;
                    let (band, cls0, cls1) = ccso_class(
                        luma,
                        x_luma,
                        y_luma,
                        dx,
                        dy,
                        quant_step,
                        cand.edge_clf,
                        shift,
                        max_x,
                        max_y,
                    );
                    if band >= max_band {
                        continue;
                    }
                    let err =
                        src_plane.data[s_row + x].to_i32() - rec_plane.data[r_row + x].to_i32();
                    sum[band][cls0][cls1] += err as i64;
                    count[band][cls0][cls1] += 1;
                }
            }
        }
    }

    let mut filter_off = [0u8; 64];
    for (band, band_sum) in sum.iter().enumerate().take(max_band) {
        for cls0 in 0..3 {
            for cls1 in 0..3 {
                let n = count[band][cls0][cls1];
                if n == 0 {
                    continue;
                }
                let avg = (band_sum[cls0][cls1] + (n as i64 / 2)) / n as i64;
                let mut best_k = 0usize;
                let mut best_diff = i64::MAX;
                for (k, &value) in CCSO_OFFSET[scale_idx].iter().enumerate() {
                    let diff = (value as i64 - avg).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_k = k;
                    }
                }
                let byte_idx = cls0 * 16 + cls1 * 4 + (band >> 1);
                filter_off[byte_idx] |= (best_k as u8) << (4 * (band & 1));
            }
        }
    }

    let mut sse = 0u64;
    for ur in 0..unit_rows {
        let y0 = ur * unit_h;
        if y0 >= rec_plane.height {
            break;
        }
        let y1 = (y0 + unit_h).min(rec_plane.height);
        for uc in 0..unit_cols {
            if !units.get(ur * unit_cols + uc).copied().unwrap_or(false) {
                continue;
            }
            let x0 = uc * unit_w;
            if x0 >= rec_plane.width {
                break;
            }
            let x1 = (x0 + unit_w).min(rec_plane.width);
            for y in y0..y1 {
                let s_row = y * src_plane.stride;
                let r_row = y * rec_plane.stride;
                let y_luma = (y << rec_plane.ss_y) as i32;
                for x in x0..x1 {
                    let x_luma = (x << rec_plane.ss_x) as i32;
                    let (band, cls0, cls1) = ccso_class(
                        luma,
                        x_luma,
                        y_luma,
                        dx,
                        dy,
                        quant_step,
                        cand.edge_clf,
                        shift,
                        max_x,
                        max_y,
                    );
                    if band >= max_band {
                        continue;
                    }
                    let byte_idx = cls0 * 16 + cls1 * 4 + (band >> 1);
                    let nibble = (filter_off[byte_idx] >> (4 * (band & 1))) & 0x7;
                    let off = CCSO_OFFSET[scale_idx][nibble as usize] as i32;
                    let value = bd
                        .clip_pixel(rec_plane.data[r_row + x].to_i32() + off)
                        .to_i32();
                    let diff = src_plane.data[s_row + x].to_i32() - value;
                    sse += (diff * diff) as u64;
                }
            }
        }
    }

    (filter_off, sse)
}

/// Selects which CCSO units are enabled from source activity before tile coding, so the
/// per-block `ccso_blk` flags can be written during entropy coding. Units without any
/// source variance are skipped; the offsets are still solved afterwards.
pub fn select_ccso_units<P: Pixel>(
    src: &Frame<P>,
    layout: PixelLayout,
    b64size: u8,
    unit_cols: usize,
    unit_rows: usize,
    bit_depth: u8,
) -> [Vec<bool>; 3] {
    let n_planes = if layout == PixelLayout::I400 { 1 } else { 3 };
    let unit_luma = 64 * b64size.max(1) as usize;
    let threshold = 1i64 << (2 * bit_depth.saturating_sub(8) as u32);
    let mut out = [
        vec![false; unit_cols * unit_rows],
        vec![false; unit_cols * unit_rows],
        vec![false; unit_cols * unit_rows],
    ];
    for plane_idx in 0..n_planes.min(src.planes.len()) {
        let plane = &src.planes[plane_idx];
        let unit_w = (unit_luma >> plane.ss_x).max(1);
        let unit_h = (unit_luma >> plane.ss_y).max(1);
        for ur in 0..unit_rows {
            let y0 = ur * unit_h;
            if y0 >= plane.height {
                break;
            }
            let y1 = (y0 + unit_h).min(plane.height);
            for uc in 0..unit_cols {
                let x0 = uc * unit_w;
                if x0 >= plane.width {
                    break;
                }
                let x1 = (x0 + unit_w).min(plane.width);
                let mut sum = 0i64;
                let mut sum_sq = 0i64;
                let mut n = 0i64;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let v = plane.data[y * plane.stride + x].to_i32() as i64;
                        sum += v;
                        sum_sq += v * v;
                        n += 1;
                    }
                }
                if n > 0 {
                    out[plane_idx][ur * unit_cols + uc] =
                        sum_sq * n - sum * sum > threshold * n * n;
                }
            }
        }
    }
    out
}

/// Chooses band and edge CCSO parameters per plane and applies the normative filter to
/// `rec` (AV2 Spec v1.0.0 §5.18.7.12, §7.19.1). `ccso_blks` was fixed before tile coding
/// because the flags are part of the tile payload.
pub fn optimize_and_apply_ccso<BD: BitDepth>(
    bd: BD,
    src: &Frame<BD::Pixel>,
    rec: &mut Frame<BD::Pixel>,
    orig_luma: &Plane<BD::Pixel>,
    seq: &SequenceHeader,
    fh: &mut FrameHeader,
    ccso_blks: &[Vec<bool>; 3],
) {
    if fh.ccso.enabled == 0 {
        return;
    }
    let n_planes = if rec.layout == PixelLayout::I400 {
        1
    } else {
        3
    };
    let unit_luma = 64 * fh.ccso.b64size.max(1) as usize;
    let ccso_step4 = 16 * fh.ccso.b64size.max(1) as usize;
    let f_bw4 = 2 * ((fh.width.max(0) as usize + 7) >> 3);
    let f_bh4 = 2 * ((fh.height.max(0) as usize + 7) >> 3);
    let unit_cols = f_bw4.div_ceil(ccso_step4);
    let unit_rows = f_bh4.div_ceil(ccso_step4);

    for plane_idx in 0..n_planes.min(rec.planes.len()) {
        if fh.ccso.planes[plane_idx].enabled == 0 {
            continue;
        }
        let units = &ccso_blks[plane_idx];
        let mut candidates = Vec::with_capacity(24);
        for scale_idx in 0..4u8 {
            candidates.push(CcsoCandidate {
                bo_only: true,
                scale_idx,
                quant_idx: 0,
                ext_filter: 0,
                edge_clf: 0,
                max_band_log2: 3,
            });
        }
        for ext_filter in 0..7u8 {
            for edge_clf in [0u8, 1u8] {
                candidates.push(CcsoCandidate {
                    bo_only: false,
                    scale_idx: 0,
                    quant_idx: 1,
                    ext_filter,
                    edge_clf,
                    max_band_log2: 3,
                });
            }
        }
        candidates.push(CcsoCandidate {
            bo_only: false,
            scale_idx: 0,
            quant_idx: 1,
            ext_filter: 1,
            edge_clf: 0,
            max_band_log2: 2,
        });
        candidates.push(CcsoCandidate {
            bo_only: false,
            scale_idx: 1,
            quant_idx: 0,
            ext_filter: 1,
            edge_clf: 0,
            max_band_log2: 3,
        });
        candidates.push(CcsoCandidate {
            bo_only: false,
            scale_idx: 1,
            quant_idx: 1,
            ext_filter: 1,
            edge_clf: 0,
            max_band_log2: 3,
        });

        let mut best: Option<(CcsoCandidate, [u8; 64], u64)> = None;
        for cand in candidates {
            let (filter_off, sse) = ccso_trial(
                bd,
                &src.planes[plane_idx],
                &rec.planes[plane_idx],
                orig_luma,
                cand,
                units,
                unit_cols,
                unit_rows,
                unit_luma,
            );
            if best
                .as_ref()
                .map(|(_, _, best_sse)| sse < *best_sse)
                .unwrap_or(true)
            {
                best = Some((cand, filter_off, sse));
            }
        }

        if let Some((cand, filter_off, _)) = best {
            let plane_hdr = &mut fh.ccso.planes[plane_idx];
            plane_hdr.bo_only = cand.bo_only as u8;
            plane_hdr.scale_idx = cand.scale_idx;
            plane_hdr.quant_idx = cand.quant_idx;
            plane_hdr.ext_filter_support = cand.ext_filter;
            plane_hdr.edge_clf = cand.edge_clf;
            plane_hdr.max_band_log2 = cand.max_band_log2;
            plane_hdr.filter_off = filter_off;
        }
    }

    ccso_frame_normative(bd, rec, orig_luma, seq, fh, ccso_blks);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::headers::FhCcsoPlane;
    use crate::av2::pixel::BitDepth8;

    #[test]
    fn test_ccso_score_and_constants() {
        assert_eq!(CCSO_POS.len(), 7);
        assert_eq!(CCSO_POS[0], [-1, 0]);
        assert_eq!(CCSO_POS[6], [0, 2]);

        assert_eq!(CCSO_QUANT_SZ[0][0], 16);
        assert_eq!(CCSO_QUANT_SZ[3][3], 256);

        assert_eq!(CCSO_OFFSET[0][1], 1);
        assert_eq!(CCSO_OFFSET[1][1], 2);
        assert_eq!(CCSO_OFFSET[2][1], 3);
        assert_eq!(CCSO_OFFSET[3][1], 4);
        assert_eq!(CCSO_OFFSET[0][7], -10);
        assert_eq!(CCSO_OFFSET[3][7], -40);

        // Score tests
        // diff > quant_step: 2 if edge_clf == 0, 1 if edge_clf != 0
        assert_eq!(ccso_score(20, 16, 0), 2);
        assert_eq!(ccso_score(20, 16, 1), 1);
        // diff < -quant_step: 0 regardless of edge_clf
        assert_eq!(ccso_score(-20, 16, 0), 0);
        assert_eq!(ccso_score(-20, 16, 1), 0);
        // inside quant_step: 1
        assert_eq!(ccso_score(10, 16, 0), 1);
        assert_eq!(ccso_score(-10, 16, 0), 1);
        assert_eq!(ccso_score(0, 16, 0), 1);
    }

    #[test]
    fn test_get_ccso_luma_clamping() {
        let mut luma = Plane {
            data: vec![0u8; 16 * 16],
            width: 16,
            height: 16,
            stride: 16,
            alloc_height: 16,
            coded_width: 16,
            coded_height: 16,
            ss_x: 0,
            ss_y: 0,
        };
        luma.data[0] = 42;
        luma.data[5 * 16 + 5] = 99;
        luma.data[15 * 16 + 15] = 200;

        assert_eq!(get_ccso_luma(&luma, 5, 5, 0, 15, 0, 15), 99);
        // Clamping x and y below min
        assert_eq!(get_ccso_luma(&luma, -5, -3, 0, 15, 0, 15), 42);
        // Clamping x and y above max
        assert_eq!(get_ccso_luma(&luma, 20, 20, 0, 15, 0, 15), 200);
        // Strict tile bounds
        assert_eq!(get_ccso_luma(&luma, 0, 0, 5, 10, 5, 10), 99);
    }

    #[test]
    fn test_ccso_frame_normative_bo_only() {
        let bd = BitDepth8;
        let mut rec = Frame::new(64, 64, 8, PixelLayout::I420, 0);
        let orig_luma = rec.planes[0].clone();

        let seq = SequenceHeader {
            layout: PixelLayout::I420,
            ss_hor: 1,
            ss_ver: 1,
            disable_loopfilters_across_tiles: false,
            ..Default::default()
        };

        let mut fh = FrameHeader {
            width: 64,
            height: 64,
            ..Default::default()
        };
        fh.ccso.enabled = 1;
        fh.ccso.b64size = 1;
        // Enable CCSO on plane 1 (U)
        fh.ccso.planes[1] = FhCcsoPlane {
            enabled: 1,
            bo_only: 1,
            scale_idx: 0,     // CCSO_OFFSET[0]
            max_band_log2: 2, // 4 bands: shift = 8 - 2 = 6
            ..Default::default()
        };
        // Set offset index 1 for band 0 (byte_idx = 0 >> 1 = 0, nibble 0)
        // CCSO_OFFSET[0][1] = 1
        fh.ccso.planes[1].filter_off[0] = 0x01;

        let ccso_blks = [vec![false; 1], vec![true; 1], vec![false; 1]];

        // Plane 1 initial value is 0. Since luma is 0, band is 0.
        // Offset is +1.
        ccso_frame_normative(bd, &mut rec, &orig_luma, &seq, &fh, &ccso_blks);

        // U plane should now have value 1 everywhere in the enabled block
        assert_eq!(rec.planes[1].data[0], 1);
        assert_eq!(rec.planes[1].data[31 * rec.planes[1].stride + 31], 1);
        // Y plane (pl=0) and V plane (pl=2) should remain 0
        assert_eq!(rec.planes[0].data[0], 0);
        assert_eq!(rec.planes[2].data[0], 0);
    }

    #[test]
    fn test_ccso_frame_normative_edge_clf() {
        let bd = BitDepth8;
        let mut rec = Frame::new(64, 64, 8, PixelLayout::I444, 0);
        // Create an edge in orig_luma:
        // center at (32, 32) is 100
        // (32 + dx, 32 + dy) = (32 - 1, 32 + 0) = (31, 32) -> 130 (diff = 30)
        // (32 - dx, 32 - dy) = (33, 32) -> 70 (diff = -30)
        let mut orig_luma = rec.planes[0].clone();
        for y in 0..64 {
            for x in 0..64 {
                orig_luma.data[y * orig_luma.stride + x] = 100;
            }
        }
        orig_luma.data[32 * orig_luma.stride + 31] = 130;
        orig_luma.data[32 * orig_luma.stride + 33] = 70;

        let seq = SequenceHeader {
            layout: PixelLayout::I444,
            ss_hor: 0,
            ss_ver: 0,
            disable_loopfilters_across_tiles: false,
            ..Default::default()
        };

        let mut fh = FrameHeader {
            width: 64,
            height: 64,
            ..Default::default()
        };
        fh.ccso.enabled = 1;
        fh.ccso.b64size = 1;
        // Enable on plane 0 (Y)
        // ext_filter = 0: dy = -1, dx = 0. Wait! CCSO_POS[0] is [-1, 0] (dy = -1, dx = 0).
        // Let's use ext_filter = 1: CCSO_POS[1] = [0, -1] (dy = 0, dx = -1).
        fh.ccso.planes[0] = FhCcsoPlane {
            enabled: 1,
            bo_only: 0,
            scale_idx: 1,          // CCSO_OFFSET[1] = [0, 2, -2, 6, -6, 14, -14, -20]
            quant_idx: 0, // CCSO_QUANT_SZ[1][0] = 56. Wait! If quant_step is 56, diff=30 is inside quant!
            ext_filter_support: 1, // dy = 0, dx = -1
            edge_clf: 0,
            max_band_log2: 1, // 2 bands
            ..Default::default()
        };
        // With scale_idx = 0, quant_idx = 1: CCSO_QUANT_SZ[0][1] = 8.
        fh.ccso.planes[0].scale_idx = 0;
        fh.ccso.planes[0].quant_idx = 1;
        // quant_step = 8.
        // At (32, 32):
        // p0 = luma(32 - 1, 32) = 130. p0 - c = 130 - 100 = 30 > 8 => cls0 = 2.
        // p1 = luma(32 + 1, 32) = 70. p1 - c = 70 - 100 = -30 < -8 => cls1 = 0.
        // c = 100. shift = 8 - 1 = 7. band = 100 >> 7 = 0.
        // byte_idx = cls0 * 16 + cls1 * 4 + (band >> 1) = 2 * 16 + 0 * 4 + 0 = 32.
        // (band & 1) = 0.
        // Set nibble = 3 (offset index 3). CCSO_OFFSET[0][3] = 3.
        fh.ccso.planes[0].filter_off[32] = 0x03;

        let ccso_blks = [vec![true; 1], vec![false; 1], vec![false; 1]];

        rec.planes[0].data[32 * rec.planes[0].stride + 32] = 100;
        ccso_frame_normative(bd, &mut rec, &orig_luma, &seq, &fh, &ccso_blks);

        assert_eq!(rec.planes[0].data[32 * rec.planes[0].stride + 32], 103);
    }

    #[test]
    fn test_ccso_frame_normative_10bit_12bit() {
        use crate::av2::pixel::BitDepth16;

        let bd10 = BitDepth16::new(10);
        let mut rec10 = Frame::<u16>::new(64, 64, 10, PixelLayout::I420, 0);
        let orig_luma10 = rec10.planes[0].clone();
        let seq = SequenceHeader {
            layout: PixelLayout::I420,
            ss_hor: 1,
            ss_ver: 1,
            ..Default::default()
        };
        let mut fh = FrameHeader {
            width: 64,
            height: 64,
            ..Default::default()
        };
        fh.ccso.enabled = 1;
        fh.ccso.b64size = 1;
        fh.ccso.planes[1] = FhCcsoPlane {
            enabled: 1,
            bo_only: 1,
            scale_idx: 0,
            max_band_log2: 2,
            ..Default::default()
        };
        fh.ccso.planes[1].filter_off[0] = 0x01;
        let ccso_blks = [vec![false; 1], vec![true; 1], vec![false; 1]];

        ccso_frame_normative(bd10, &mut rec10, &orig_luma10, &seq, &fh, &ccso_blks);
        assert_eq!(rec10.planes[1].data[0], 1);

        let bd12 = BitDepth16::new(12);
        let mut rec12 = Frame::<u16>::new(64, 64, 12, PixelLayout::I420, 0);
        let orig_luma12 = rec12.planes[0].clone();
        ccso_frame_normative(bd12, &mut rec12, &orig_luma12, &seq, &fh, &ccso_blks);
        assert_eq!(rec12.planes[1].data[0], 1);
    }
}
