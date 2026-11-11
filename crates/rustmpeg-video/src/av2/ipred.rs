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

//! Clean-room AV2 Spatial Intra Prediction (`ipred.rs`), including:
//! - Standard non-directional modes (`Dc`, `TopDc`, `LeftDc`, `Dc128`, `Smooth`, `SmoothV`, `SmoothH`, `Paeth`)
//! - Directional modes (`Vert`, `Hor`, `Z1`, `Z2`, `Z3`) with 4-tap cubic/quadratic subpel interpolation,
//!   wide-angle remapping, and Wave 2 directional angle delta refinement (`angle_delta = -3..=3`)
//! - Wave 2 Multiple Reference Lines (`mrl_idx = 0..=3`) edge extraction
//! - Wave 2 5-mode 7-tap Filter Intra prediction (`predict_filter_intra_block`, `select_best_filter_intra_mode`)

use crate::av2::intops::{approx_divide, ulog2};
use crate::av2::levels::{
    DC_128_PRED, FilterIntraMode, LEFT_DC_PRED, TOP_DC_PRED, Z1_PRED, Z2_PRED, Z3_PRED,
};
use crate::av2::pixel::{BitDepth, Pixel};
use crate::av2::tables::{DIP_WEIGHTS, DR_INTRA_DERIVATIVE, FILTER_INTRA_TAPS, MODE_TO_ANGLE_MAP};

/// 4-tap directional intra interpolation filter coefficients (`sum = 128`, AV2 Spec v1.0.0 §7.13.2.8).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DrFilter4Tap {
    pub f: [i16; 4],
}

pub static DR_INTERP_FILTER: [DrFilter4Tap; 32] = [
    DrFilter4Tap { f: [0, 128, 0, 0] },
    DrFilter4Tap {
        f: [-2, 127, 4, -1],
    },
    DrFilter4Tap {
        f: [-3, 125, 8, -2],
    },
    DrFilter4Tap {
        f: [-5, 123, 13, -3],
    },
    DrFilter4Tap {
        f: [-6, 121, 17, -4],
    },
    DrFilter4Tap {
        f: [-7, 118, 22, -5],
    },
    DrFilter4Tap {
        f: [-9, 116, 27, -6],
    },
    DrFilter4Tap {
        f: [-9, 112, 32, -7],
    },
    DrFilter4Tap {
        f: [-10, 109, 37, -8],
    },
    DrFilter4Tap {
        f: [-11, 106, 41, -8],
    },
    DrFilter4Tap {
        f: [-11, 102, 46, -9],
    },
    DrFilter4Tap {
        f: [-12, 98, 52, -10],
    },
    DrFilter4Tap {
        f: [-12, 94, 56, -10],
    },
    DrFilter4Tap {
        f: [-12, 90, 61, -11],
    },
    DrFilter4Tap {
        f: [-12, 85, 66, -11],
    },
    DrFilter4Tap {
        f: [-12, 81, 71, -12],
    },
    DrFilter4Tap {
        f: [-12, 76, 76, -12],
    },
    DrFilter4Tap {
        f: [-12, 71, 81, -12],
    },
    DrFilter4Tap {
        f: [-11, 66, 85, -12],
    },
    DrFilter4Tap {
        f: [-11, 61, 90, -12],
    },
    DrFilter4Tap {
        f: [-10, 56, 94, -12],
    },
    DrFilter4Tap {
        f: [-10, 52, 98, -12],
    },
    DrFilter4Tap {
        f: [-9, 46, 102, -11],
    },
    DrFilter4Tap {
        f: [-8, 41, 106, -11],
    },
    DrFilter4Tap {
        f: [-8, 37, 109, -10],
    },
    DrFilter4Tap {
        f: [-7, 32, 112, -9],
    },
    DrFilter4Tap {
        f: [-6, 27, 116, -9],
    },
    DrFilter4Tap {
        f: [-5, 22, 118, -7],
    },
    DrFilter4Tap {
        f: [-4, 17, 121, -6],
    },
    DrFilter4Tap {
        f: [-3, 13, 123, -5],
    },
    DrFilter4Tap {
        f: [-2, 8, 125, -3],
    },
    DrFilter4Tap {
        f: [-1, 4, 127, -2],
    },
];

/// Remaps nominal angle for rectangular blocks according to AV2 wide-angle intra prediction.
#[inline]
pub fn wide_angle_remap(angle: i32, w: usize, h: usize) -> i32 {
    if w == h {
        return angle;
    }
    let ratio = if w > h { w / h } else { h / w };
    let delta = match ratio {
        2 => 5,
        4 => 10,
        8 => 14,
        16 => 16,
        _ => 0,
    };
    if w > h && angle > 180 {
        angle + delta
    } else if h > w && angle < 90 {
        angle - delta
    } else {
        angle
    }
}

/// Computes the above-right and below-left reference availability for intra
/// prediction (AV2 §7.13.2.9).
///
/// Above-right samples may come from the previous superblock row (they are
/// already reconstructed), but must stay inside the current superblock when the
/// block lies inside a superblock. Below-left samples are only usable while the
/// block bottom is inside the current superblock row; otherwise they belong to
/// a superblock that has not been decoded yet.
#[allow(clippy::too_many_arguments)]
pub fn intra_edge_availability(
    px_x: usize,
    px_y: usize,
    w: usize,
    h: usize,
    tile_end_x: usize,
    tile_end_y: usize,
    sb_px: usize,
    have_left: bool,
    have_top: bool,
) -> (bool, bool) {
    let sb_px = sb_px.max(1);
    let bx_in_sb = px_x % sb_px;
    let by_in_sb = px_y % sb_px;
    // Above-right samples must stay inside the tile; when the block starts at a
    // superblock top boundary they come from the previous (already decoded) SB
    // row, otherwise they must stay inside the current superblock and belong to
    // a Z-order ancestor half that was already decoded before `by_in_sb`.
    let have_top_right = have_top
        && px_x + w < tile_end_x
        && (by_in_sb == 0
            || (bx_in_sb + w < sb_px
                && by_in_sb.trailing_zeros() >= (bx_in_sb + w).trailing_zeros()));
    // Below-left samples are available at the left superblock boundary (`x - 1`
    // in the already-decoded left superblock), or inside the superblock when
    // the Z-order block at `(bx_in_sb - 4, by_in_sb + h)` belongs to a vertical
    // ancestor half that was already decoded before `bx_in_sb`.
    let have_bottom_left = have_left
        && px_y + h < tile_end_y
        && by_in_sb + h < sb_px
        && (bx_in_sb == 0 || bx_in_sb.trailing_zeros() > (by_in_sb + h).trailing_zeros());
    (have_top_right, have_bottom_left)
}

/// Populates the 1D reference edge array `edge` centered at `tl_idx` from reconstructed plane `rec`,
/// supporting Multiple Reference Line offset `mrl_idx` (`0..=3` -> line distance `1 + mrl_idx`).
///
/// `rec_w`/`rec_h` are the coded dimensions of the plane. AV2 §7.13.2.1 clamps every
/// reference read to `maxX`/`maxY`, so a block that overhangs the right or bottom frame
/// boundary reuses the last in-frame row/column instead of the plane padding.
#[allow(clippy::too_many_arguments)]
pub fn prepare_intra_edges_mrl<BD: BitDepth>(
    bd: BD,
    edge: &mut [BD::Pixel; 512],
    tl_idx: usize,
    rec: &[BD::Pixel],
    rec_off: usize,
    rec_stride: usize,
    have_left: bool,
    have_top: bool,
    have_top_right: bool,
    have_bottom_left: bool,
    w: usize,
    h: usize,
    mrl_idx: u8,
    rec_w: usize,
    rec_h: usize,
) {
    let stride = rec_stride.max(1);
    let mid = BD::Pixel::from_i32(1 << (bd.bitdepth() - 1));
    let mid_lo = BD::Pixel::from_i32((1 << (bd.bitdepth() - 1)) - 1);
    let mid_hi = BD::Pixel::from_i32((1 << (bd.bitdepth() - 1)) + 1);
    let max_wh = w + h;
    let px_x = rec_off % stride;
    let px_y = rec_off / stride;
    let max_x = rec_w.saturating_sub(1);
    let max_y = rec_h.saturating_sub(1);
    let line_dist = (1 + mrl_idx as usize).min(px_y + 1);
    // Top row base (column zero) and left column position for the current block.
    let top_base = (px_y.saturating_sub(line_dist)) * stride;
    let left_col = px_x.saturating_sub(line_dist);

    // Top-left corner
    edge[tl_idx] = if have_top && have_left && px_x >= line_dist && px_y >= line_dist {
        rec[top_base + left_col]
    } else if have_top && px_y >= line_dist {
        rec[top_base + px_x.min(max_x)]
    } else if have_left && px_x >= line_dist {
        rec[px_y.min(max_y) * stride + left_col]
    } else {
        mid
    };

    // Top and Top-Right reference samples (`edge[tl_idx + 1 ..= tl_idx + w + h]`)
    if have_top && px_y >= line_dist {
        let px_have_w = w.min(rec_w.saturating_sub(px_x));
        for x in 0..w {
            let col = (px_x + x).min(max_x);
            edge[tl_idx + 1 + x] = rec[top_base + col];
        }
        let last_top = edge[tl_idx + w];
        let tr_avail = if have_top_right && px_x + w < rec_w {
            (max_wh - w).min(rec_w - (px_x + w))
        } else {
            0
        };
        for x in w..w + tr_avail {
            edge[tl_idx + 1 + x] = rec[top_base + px_x + x];
        }
        let pad_top = edge[tl_idx + w + tr_avail];
        let _ = (px_have_w, last_top);
        for x in w + tr_avail..max_wh {
            edge[tl_idx + 1 + x] = pad_top;
        }
    } else {
        let fill = if have_left && px_x >= line_dist {
            rec[px_y.min(max_y) * stride + left_col]
        } else {
            mid_lo
        };
        for x in 0..max_wh {
            edge[tl_idx + 1 + x] = fill;
        }
    }

    // Left and Bottom-Left reference samples (`edge[tl_idx - 1]` down to `edge[tl_idx - (w + h)]`)
    if have_left && px_x >= line_dist {
        for y in 0..h {
            let row = (px_y + y).min(max_y);
            edge[tl_idx - 1 - y] = rec[row * stride + left_col];
        }
        let bl_avail = if have_bottom_left && px_y + h < rec_h {
            (max_wh - h).min(rec_h - (px_y + h))
        } else {
            0
        };
        for y in h..h + bl_avail {
            edge[tl_idx - 1 - y] = rec[(px_y + y) * stride + left_col];
        }
        let pad_left = edge[tl_idx - (h + bl_avail)];
        for y in h + bl_avail..max_wh {
            edge[tl_idx - 1 - y] = pad_left;
        }
    } else {
        let fill = if have_top && px_y >= line_dist {
            rec[top_base + px_x.min(max_x)]
        } else {
            mid_hi
        };
        for y in 0..max_wh {
            edge[tl_idx - 1 - y] = fill;
        }
    }
}

/// Returns the coded plane extent `(width, height)` used to clamp intra reference
/// reads for a plane with subsampling shifts `(ss_x, ss_y)`.
///
/// AV2 Spec v1.0.0 §7.13.2.1 derives `maxX = (MiCols * MI_SIZE) - 1` and
/// `maxY = (MiRows * MI_SIZE) - 1`, shifted by the plane subsampling, so blocks
/// overhanging the right or bottom frame edge reuse the last coded row/column
/// instead of the plane padding.
pub fn intra_reference_extent(
    mi_cols: usize,
    mi_rows: usize,
    ss_x: u8,
    ss_y: u8,
) -> (usize, usize) {
    ((mi_cols * 4) >> ss_x, (mi_rows * 4) >> ss_y)
}

/// Standard single-reference-line (`mrl_idx = 0`) wrapper around [`prepare_intra_edges_mrl`].
#[allow(clippy::too_many_arguments)]
pub fn prepare_intra_edges<BD: BitDepth>(
    bd: BD,
    edge: &mut [BD::Pixel; 512],
    tl_idx: usize,
    rec: &[BD::Pixel],
    rec_off: usize,
    rec_stride: usize,
    have_left: bool,
    have_top: bool,
    have_top_right: bool,
    have_bottom_left: bool,
    w: usize,
    h: usize,
    rec_w: usize,
    rec_h: usize,
) {
    prepare_intra_edges_mrl(
        bd,
        edge,
        tl_idx,
        rec,
        rec_off,
        rec_stride,
        have_left,
        have_top,
        have_top_right,
        have_bottom_left,
        w,
        h,
        0,
        rec_w,
        rec_h,
    );
}

/// Predicts an intra block of size `w x h` into `dst` at `dst_off` with stride `dst_stride`.
///
/// `plane` selects the directional interpolation stage: AV2 Spec v1.0.0
/// §7.13.2.8 sets `enableIdif = plane == 0`, so luma uses the 4-tap IDIF filter
/// while chroma uses two-tap linear interpolation. `mrl_idx` is the multiple
/// reference line index already applied to `edge` (`0` when MRL is not used).
#[allow(clippy::too_many_arguments)]
pub fn predict_intra_block<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    dst_stride: usize,
    edge: &[BD::Pixel; 512],
    tl_idx: usize,
    mode: u8,
    angle_delta: i8,
    w: usize,
    h: usize,
    have_left: bool,
    have_top: bool,
    plane: usize,
    mrl_idx: u8,
) {
    let mid = 1i32 << (bd.bitdepth() - 1);

    // DC mode (0)
    if mode == 0 {
        let dc_val = match (have_top, have_left) {
            (true, true) => {
                let mut sum = 0i32;
                for x in 0..w {
                    sum += edge[tl_idx + 1 + x].to_i32();
                }
                for y in 0..h {
                    sum += edge[tl_idx - 1 - y].to_i32();
                }
                // AV2 §7.13.2.10: the average uses the approximate divide,
                // which differs from a rounded division when `w + h` is not a
                // power of two.
                approx_divide(sum, (w + h) as i32)
            }
            (true, false) => {
                let mut sum = 0i32;
                for x in 0..w {
                    sum += edge[tl_idx + 1 + x].to_i32();
                }
                let sh = ulog2(w as u32);
                (sum + (w as i32 >> 1)) >> sh
            }
            (false, true) => {
                let mut sum = 0i32;
                for y in 0..h {
                    sum += edge[tl_idx - 1 - y].to_i32();
                }
                let sh = ulog2(h as u32);
                (sum + (h as i32 >> 1)) >> sh
            }
            (false, false) => mid,
        };
        let px = bd.clip_pixel(dc_val);
        for y in 0..h {
            let row = dst_off + y * dst_stride;
            dst[row..row + w].fill(px);
        }
        let _ = (
            LEFT_DC_PRED,
            TOP_DC_PRED,
            DC_128_PRED,
            Z1_PRED,
            Z2_PRED,
            Z3_PRED,
        );
        return;
    }

    // Smooth (9), SmoothV (10), SmoothH (11) per AV2 Spec v1.0.0 §7.13.2.13
    if (9..=11).contains(&mode) {
        const BLEND_WEIGHT_MAX: i32 = 32;
        // AV2 §7.13.2.13: `bl = LeftCol[h]` and `tr = AboveRow[w]` are the
        // below-left and above-right reference samples of the edge buffer.
        let bl = edge[tl_idx - h - 1].to_i32();
        let tr = edge[tl_idx + w + 1].to_i32();
        let log2_w = ulog2(w as u32);
        let log2_h = ulog2(h as u32);
        let scale = ((log2_w + log2_h - 4 + 2) >> 2).max(0);
        let round_w = if log2_w > 0 { 1 << (log2_w - 1) } else { 0 };
        let round_h = if log2_h > 0 { 1 << (log2_h - 1) } else { 0 };

        for y in 0..h {
            let row = dst_off + y * dst_stride;
            let shift_y = (((y as i32) << 1) >> scale).clamp(0, 6);
            let s_top = BLEND_WEIGHT_MAX >> shift_y;
            let left = edge[tl_idx - 1 - y].to_i32();
            for x in 0..w {
                let shift_x = (((x as i32) << 1) >> scale).clamp(0, 6);
                let s_left = BLEND_WEIGHT_MAX >> shift_x;
                let top = edge[tl_idx + 1 + x].to_i32();

                let pred_h = tr + (((left - tr) * ((w - 1 - x) as i32) + round_w) >> log2_w);
                let pred_v = bl + (((top - bl) * ((h - 1 - y) as i32) + round_h) >> log2_h);
                let pred_h2 = pred_h + (((left - pred_h) * s_left + 32) >> 6);
                let pred_v2 = pred_v + (((top - pred_v) * s_top + 32) >> 6);

                let pred = match mode {
                    10 => pred_v2,
                    11 => pred_h2,
                    _ => (pred_v2 + pred_h2 + 1) >> 1,
                };
                dst[row + x] = bd.clip_pixel(pred);
            }
        }
        return;
    }

    // Paeth (12) per AV2 Spec v1.0.0 §7.13.2.2
    if mode == 12 {
        let tl = edge[tl_idx].to_i32();
        for y in 0..h {
            let row = dst_off + y * dst_stride;
            let l = edge[tl_idx - 1 - y].to_i32();
            for x in 0..w {
                let t = edge[tl_idx + 1 + x].to_i32();
                let base = t + l - tl;
                let dt = (base - t).abs();
                let dl = (base - l).abs();
                let dtl = (base - tl).abs();
                let p = if dt <= dl && dt <= dtl {
                    t
                } else if dl <= dtl {
                    l
                } else {
                    tl
                };
                dst[row + x] = BD::Pixel::from_i32(p);
            }
        }
        return;
    }

    // Directional modes (1..=8) per AV2 Spec v1.0.0 §7.13.2.7 & §7.13.2.8
    let nom_angle = MODE_TO_ANGLE_MAP[(mode.saturating_sub(1) as usize).min(7)] as i32;
    let angle = wide_angle_remap(nom_angle + (angle_delta as i32) * 3, w, h);
    let enable_idif = plane == 0;
    let mrl = mrl_idx as i32;

    if angle == 90 {
        let src_slice = &edge[tl_idx + 1..tl_idx + 1 + w];
        for y in 0..h {
            let row = dst_off + y * dst_stride;
            dst[row..row + w].copy_from_slice(src_slice);
        }
        return;
    }
    if angle == 180 {
        for y in 0..h {
            let row = dst_off + y * dst_stride;
            dst[row..row + w].fill(left_at(edge, tl_idx, (y as i32).max(0)));
        }
        return;
    }

    // §7.13.2.8 extends the reference arrays before filtering. Only luma runs
    // the IDIF stage, so only luma needs the extension; it is applied to local
    // copies because the caller's edge buffer is reused across mode trials.
    let min_base = -1 - mrl;
    let max_base = (w + h - 1) as i32 + (mrl << 1);
    // `w + h <= 128` and `mrl_index <= 3`, so the copies never exceed 138 slots.
    const LOCAL_EDGE: usize = 160;
    debug_assert!((max_base + 5) as usize <= LOCAL_EDGE);
    let mut above = [BD::Pixel::default(); LOCAL_EDGE];
    let mut left = [BD::Pixel::default(); LOCAL_EDGE];
    // Reference index `k` lives at buffer slot `k + 2`; the low slots hold the
    // `AboveRow[-2]` / `LeftCol[-2]` extension requested by §7.13.2.8.
    let conv = |k: i32| (k + 2) as usize;
    // The reference arrays are indexed from `minBase - 1`: §7.13.2.8 fills that
    // slot with `LeftCol[minBase]` / `AboveRow[minBase]`, which is the
    // above-left corner for the outermost multiple reference line.
    for k in min_base - 1..max_base + 3 {
        above[conv(k)] = above_at(edge, tl_idx, k);
        left[conv(k)] = left_at(edge, tl_idx, k);
    }
    if enable_idif {
        if angle > 90 && angle < 180 {
            let l = left[conv(h as i32 - 1)];
            left[conv(h as i32)] = l;
            left[conv(h as i32 + 1)] = l;
            let a = above[conv(w as i32 - 1)];
            above[conv(w as i32)] = a;
            above[conv(w as i32 + 1)] = a;
        } else {
            let a = above[conv(max_base)];
            above[conv(max_base + 1)] = a;
            above[conv(max_base + 2)] = a;
            let l = left[conv(max_base)];
            left[conv(max_base + 1)] = l;
            left[conv(max_base + 2)] = l;
        }
    }

    // 2-tap linear interpolation used when `enableIdif` is 0 (chroma).
    let lerp = |arr: &[BD::Pixel], k: i32, shift: i32| -> i32 {
        let a = arr[conv(k)].to_i32();
        let b = arr[conv(k + 1)].to_i32();
        (a * (32 - shift) + b * shift + 16) >> 5
    };
    // 4-tap IDIF filter used when `enableIdif` is 1 (luma).
    let idif = |arr: &[BD::Pixel], k: i32, shift: usize| -> i32 {
        let taps = &DR_INTERP_FILTER[shift].f;
        let mut sum = 0i32;
        for (t, tap) in taps.iter().enumerate() {
            sum += arr[conv(k + t as i32 - 1)].to_i32() * (*tap as i32);
        }
        (sum + 64) >> 7
    };

    if angle < 90 {
        // Zone 1: project onto the above row.
        let dx = DR_INTRA_DERIVATIVE[angle.clamp(1, 89) as usize] as i32;
        let limit = max_base + enable_idif as i32;
        for y in 0..h {
            let row = dst_off + y * dst_stride;
            let idx = (y as i32 + 1 + mrl) * dx;
            let base = idx >> 6;
            let shift = ((idx >> 1) & 0x1F) as usize;
            for x in 0..w {
                let b = base + x as i32;
                dst[row + x] = if b < limit {
                    let p = if enable_idif {
                        idif(&above, b, shift)
                    } else {
                        lerp(&above, b, shift as i32)
                    };
                    bd.clip_pixel(p)
                } else {
                    above[conv(max_base)]
                };
            }
        }
    } else if angle > 180 {
        // Zone 3: project onto the left column.
        let dy = DR_INTRA_DERIVATIVE[(270 - angle).clamp(1, 89) as usize] as i32;
        let limit = max_base + enable_idif as i32;
        for x in 0..w {
            let idx = (x as i32 + 1 + mrl) * dy;
            let base = idx >> 6;
            let shift = ((idx >> 1) & 0x1F) as usize;
            let mut y = 0usize;
            while y < h {
                let b = base + y as i32;
                dst[dst_off + y * dst_stride + x] = if b < limit {
                    let p = if enable_idif {
                        idif(&left, b, shift)
                    } else {
                        lerp(&left, b, shift as i32)
                    };
                    bd.clip_pixel(p)
                } else {
                    left[conv(max_base)]
                };
                y += 1;
            }
        }
    } else {
        // Zone 2: project onto the above row, or the left column past the corner.
        let dx = DR_INTRA_DERIVATIVE[(180 - angle).clamp(1, 89) as usize] as i32;
        let dy = DR_INTRA_DERIVATIVE[(angle - 90).clamp(1, 89) as usize] as i32;
        for y in 0..h {
            let row = dst_off + y * dst_stride;
            for x in 0..w {
                let idx_top = ((x as i32) << 6) - (y as i32 + 1 + mrl) * dx;
                let base_top = idx_top >> 6;
                let p = if base_top >= min_base {
                    let shift = ((idx_top >> 1) & 0x1F) as usize;
                    if enable_idif {
                        idif(&above, base_top, shift)
                    } else {
                        lerp(&above, base_top, shift as i32)
                    }
                } else {
                    let idx_left = ((y as i32) << 6) - (x as i32 + 1 + mrl) * dy;
                    let base_left = idx_left >> 6;
                    let shift = ((idx_left >> 1) & 0x1F) as usize;
                    if enable_idif {
                        idif(&left, base_left, shift)
                    } else {
                        lerp(&left, base_left, shift as i32)
                    }
                };
                dst[row + x] = bd.clip_pixel(p);
            }
        }
    }
}

/// Reads `AboveRow[k]` from the edge buffer (AV2 §7.13.2.1): `k = -1` is the
/// above-left corner and indices below `-1` repeat it.
#[inline]
fn above_at<P: Pixel>(edge: &[P], tl_idx: usize, k: i32) -> P {
    if k >= 0 {
        edge[(tl_idx + 1 + k as usize).min(edge.len() - 1)]
    } else {
        edge[tl_idx]
    }
}

/// Reads `LeftCol[k]` from the edge buffer (AV2 §7.13.2.1): `k = -1` is the
/// above-left corner and indices below `-1` repeat it.
#[inline]
fn left_at<P: Pixel>(edge: &[P], tl_idx: usize, k: i32) -> P {
    if k >= 0 {
        edge[tl_idx.saturating_sub(1 + k as usize)]
    } else {
        edge[tl_idx]
    }
}

/// Normative AV2 v1.0.0 Data-driven Intra Prediction (`DIP`, §7.13.2.3 - §7.13.2.6)
/// using `DIP_WEIGHTS` (`dip_weights.h`) for blocks with `w >= 4 && h >= 4`.
#[allow(clippy::too_many_arguments)]
pub fn predict_filter_intra_block<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    dst_stride: usize,
    edge: &[BD::Pixel; 512],
    tl_idx: usize,
    fi_mode: FilterIntraMode,
    w: usize,
    h: usize,
) {
    let _ = &FILTER_INTRA_TAPS;
    let dip_mode = (fi_mode as usize).min(5);
    let down_w = (w >> 2).max(1);
    let down_h = (h >> 2).max(1);

    // §7.13.2.4 DIP features process (`dip_transpose = 0`)
    let mut f_above = [0i32; 5];
    let mut f_left = [0i32; 5];
    for i in 0..5 {
        let mut ta = 0i32;
        for j in 0..down_w {
            let idx = (tl_idx + 1 + i * down_w + j).min(511);
            ta += edge[idx].to_i32();
        }
        f_above[i] = (ta + (down_w as i32 >> 1)) / (down_w as i32);

        let mut tl = 0i32;
        for j in 0..down_h {
            let idx = tl_idx.saturating_sub(1 + i * down_h + j);
            tl += edge[idx].to_i32();
        }
        f_left[i] = (tl + (down_h as i32 >> 1)) / (down_h as i32);
    }

    let mut feat = [0i32; 11];
    feat[0] = edge[tl_idx].to_i32();
    for i in 0..4 {
        feat[i + 1] = f_above[i];
        feat[i + 5] = f_left[i];
    }
    feat[9] = f_above[4];
    feat[10] = f_left[4];

    // §7.13.2.5 DIP transform process (produces 8x8 `dip_pred`)
    let mut dip_pred = [[0i32; 8]; 8];
    let weights = &DIP_WEIGHTS[dip_mode];
    for i in 0..8 {
        for j in 0..8 {
            let mut c = 0i32;
            let w_row = &weights[i * 8 + j];
            for k in 0..11 {
                c += (w_row[k] as i32) * feat[k];
            }
            dip_pred[i][j] = bd.clip_pixel((c + 512) >> 10).to_i32();
        }
    }

    // §7.13.2.6 DIP resample process (`dip_pred` 8x8 -> `w x h`)
    let upx = (w / 8).max(1);
    let upy = (h / 8).max(1);
    let downx = (8 / w).max(1);
    let downy = (8 / h).max(1);
    let min_h = h.min(8);
    let min_w = w.min(8);

    let mut horz_interp = [[0i32; 64]; 8];
    for i in 0..min_h {
        let y_edge = (i + 1) * upy - 1;
        let left_y = edge[tl_idx.saturating_sub(1 + y_edge)].to_i32();
        for j in 0..min_w {
            let p0 = if j == 0 {
                left_y
            } else {
                dip_pred[i * downy][(j - 1) * downx]
            };
            let p1 = dip_pred[i * downy][j * downx];
            for k in 0..upx {
                let x = j * upx + k;
                if x < w && x < 64 {
                    let w1 = (k + 1) as i32;
                    horz_interp[i][x] = ((upx as i32 - w1) * p0 + w1 * p1) / (upx as i32);
                }
            }
        }
    }

    for x in 0..w.min(64) {
        let above_x = edge[(tl_idx + 1 + x).min(511)].to_i32();
        for i in 0..min_h {
            let p0 = if i == 0 {
                above_x
            } else {
                horz_interp[i - 1][x]
            };
            let p1 = horz_interp[i][x];
            for k in 0..upy {
                let y = i * upy + k;
                if y < h {
                    let w1 = (k + 1) as i32;
                    let val = ((upy as i32 - w1) * p0 + w1 * p1) / (upy as i32);
                    dst[dst_off + y * dst_stride + x] = bd.clip_pixel(val);
                }
            }
        }
    }
}

/// Sum of Squared Errors (SSE) between two 2D blocks.
#[inline]
pub fn compute_sse<P: Pixel>(
    a: &[P],
    a_off: usize,
    a_stride: usize,
    b: &[P],
    b_off: usize,
    b_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut sse = 0u64;
    for r in 0..h {
        let ra = a_off + r * a_stride;
        let rb = b_off + r * b_stride;
        sse += P::row_sse(&a[ra..ra + w], &b[rb..rb + w], w);
    }
    sse
}

/// Sum of Absolute Differences (SAD) between two 2D blocks.
#[inline]
pub fn compute_sad<P: Pixel>(
    a: &[P],
    a_off: usize,
    a_stride: usize,
    b: &[P],
    b_off: usize,
    b_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut sad = 0u64;
    for r in 0..h {
        let ra = a_off + r * a_stride;
        let rb = b_off + r * b_stride;
        sad += P::row_sad(&a[ra..ra + w], &b[rb..rb + w], w);
    }
    sad
}

/// Sum of Absolute Transformed Differences (SATD, 4x4 Hadamard) between two 2D blocks.
#[inline]
pub fn compute_satd<P: Pixel>(
    a: &[P],
    a_off: usize,
    a_stride: usize,
    b: &[P],
    b_off: usize,
    b_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    use std::simd::prelude::*;
    use std::simd::{i32x16, simd_swizzle};

    if (w & 3) != 0 || (h & 3) != 0 {
        return compute_sad(a, a_off, a_stride, b, b_off, b_stride, w, h);
    }

    const SIGN1: i32x16 =
        i32x16::from_array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]);
    const SIGN2: i32x16 =
        i32x16::from_array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]);

    #[inline(always)]
    fn hadamard4_horiz_x16(v: i32x16) -> i32x16 {
        let swap1 = simd_swizzle!(v, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]);
        let s = swap1 + v * SIGN1;
        let swap2 = simd_swizzle!(s, [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]);
        let h = swap2 + s * SIGN2;
        h.abs()
    }

    let mut acc16 = i32x16::splat(0);
    let mut total = 0u64;
    let w16 = w & !15;

    for by in (0..h).step_by(4) {
        let ra0 = a_off + by * a_stride;
        let ra1 = ra0 + a_stride;
        let ra2 = ra1 + a_stride;
        let ra3 = ra2 + a_stride;
        let rb0 = b_off + by * b_stride;
        let rb1 = rb0 + b_stride;
        let rb2 = rb1 + b_stride;
        let rb3 = rb2 + b_stride;

        for bx in (0..w16).step_by(16) {
            let r0 = P::diff16(&a[ra0 + bx..], &b[rb0 + bx..]);
            let r1 = P::diff16(&a[ra1 + bx..], &b[rb1 + bx..]);
            let r2 = P::diff16(&a[ra2 + bx..], &b[rb2 + bx..]);
            let r3 = P::diff16(&a[ra3 + bx..], &b[rb3 + bx..]);

            let s0 = r0 + r1;
            let s1 = r0 - r1;
            let s2 = r2 + r3;
            let s3 = r2 - r3;

            let v0 = s0 + s2;
            let v1 = s1 + s3;
            let v2 = s0 - s2;
            let v3 = s1 - s3;

            acc16 += hadamard4_horiz_x16(v0)
                + hadamard4_horiz_x16(v1)
                + hadamard4_horiz_x16(v2)
                + hadamard4_horiz_x16(v3);
        }

        for bx in (w16..w).step_by(4) {
            let d0 = P::diff4(&a[ra0 + bx..], &b[rb0 + bx..]);
            let d1 = P::diff4(&a[ra1 + bx..], &b[rb1 + bx..]);
            let d2 = P::diff4(&a[ra2 + bx..], &b[rb2 + bx..]);
            let d3 = P::diff4(&a[ra3 + bx..], &b[rb3 + bx..]);
            let mut d = [0i32; 16];
            for (r, row) in [d0, d1, d2, d3].iter().enumerate() {
                let s0 = row[0] + row[1];
                let s1 = row[0] - row[1];
                let s2 = row[2] + row[3];
                let s3 = row[2] - row[3];
                d[r * 4] = s0 + s2;
                d[r * 4 + 1] = s1 + s3;
                d[r * 4 + 2] = s0 - s2;
                d[r * 4 + 3] = s1 - s3;
            }
            for c in 0..4 {
                let s0 = d[c] + d[4 + c];
                let s1 = d[c] - d[4 + c];
                let s2 = d[8 + c] + d[12 + c];
                let s3 = d[8 + c] - d[12 + c];
                total += (s0 + s2).unsigned_abs() as u64;
                total += (s1 + s3).unsigned_abs() as u64;
                total += (s0 - s2).unsigned_abs() as u64;
                total += (s1 - s3).unsigned_abs() as u64;
            }
        }
    }
    total += acc16.cast::<u64>().reduce_sum();
    (total + 2) >> 2
}

/// Wave 2 Luma Intra Mode RDO Selector:
/// Searches non-directional modes, directional modes (`1..=8`), and directional angle deltas
/// (`-3..=3`), returning `(best_mode, best_angle_delta, midx, best_cost)`.
#[allow(clippy::too_many_arguments)]
pub fn select_best_intra_mode_y<BD: BitDepth>(
    bd: BD,
    src: &[BD::Pixel],
    src_off: usize,
    src_stride: usize,
    rec: &[BD::Pixel],
    rec_off: usize,
    rec_stride: usize,
    have_left: bool,
    have_top: bool,
    w: usize,
    h: usize,
    _tx: usize,
    lambda_sad: u64,
    speed: u8,
    rec_w: usize,
    rec_h: usize,
) -> (u8, i8, u8, u64) {
    let mut edge = [BD::Pixel::default(); 512];
    let tl_idx = 256usize;
    prepare_intra_edges(
        bd, &mut edge, tl_idx, rec, rec_off, rec_stride, have_left, have_top, false, false, w, h,
        rec_w, rec_h,
    );

    let mut pred_buf = [BD::Pixel::default(); 64 * 64];
    let pred = &mut pred_buf[..w * h];

    if !have_left && !have_top {
        predict_intra_block(
            bd, pred, 0, w, &edge, tl_idx, 0, 0, w, h, false, false, 0, 0,
        );
        let satd = compute_satd(src, src_off, src_stride, pred, 0, w, w, h);
        return (0, 0, 0xFF, satd + lambda_sad);
    }

    let nondir_modes: &[u8] = if speed >= 4 {
        &[0, 9, 12]
    } else {
        &[0, 9, 10, 11, 12]
    };

    // Stage 1: Screen non-directional and directional modes using fast SIMD SAD to pick top 2.
    let mut top1 = (0u8, u64::MAX);
    let mut top2 = (9u8, u64::MAX);
    let mut dc_sad = u64::MAX;

    for &m in nondir_modes {
        predict_intra_block(
            bd, pred, 0, w, &edge, tl_idx, m, 0, w, h, have_left, have_top, 0, 0,
        );
        let sad = compute_sad(src, src_off, src_stride, pred, 0, w, w, h);
        if m == 0 {
            dc_sad = sad;
        }
        let cost_sad = sad + if m == 0 { lambda_sad >> 1 } else { lambda_sad };
        if cost_sad < top1.1 {
            top2 = top1;
            top1 = (m, cost_sad);
        } else if cost_sad < top2.1 {
            top2 = (m, cost_sad);
        }
    }

    // Screen cardinal/diagonal directional modes (`1..=4`) first.
    let mut best_card = (1u8, u64::MAX);
    for m in 1..=4u8 {
        predict_intra_block(
            bd, pred, 0, w, &edge, tl_idx, m, 0, w, h, have_left, have_top, 0, 0,
        );
        let sad = compute_sad(src, src_off, src_stride, pred, 0, w, w, h);
        if sad < best_card.1 {
            best_card = (m, sad);
        }
        let cost_sad = sad + lambda_sad;
        if cost_sad < top1.1 {
            top2 = top1;
            top1 = (m, cost_sad);
        } else if cost_sad < top2.1 {
            top2 = (m, cost_sad);
        }
        // On high-entropy noise blocks, if Vert (1) and Hor (2) show no directional advantage over DC,
        // skip the expensive 4-tap IDIF diagonal/intermediate directions (`3..=8`).
        if m == 2 && dc_sad > (w * h * 16) as u64 && best_card.1 + (best_card.1 >> 5) >= dc_sad {
            break;
        }
    }

    // Screen the 2 intermediate directional modes adjacent to the winning cardinal/diagonal mode.
    if dc_sad <= (w * h * 16) as u64 || best_card.1 + (best_card.1 >> 5) < dc_sad {
        let adj_modes: &[u8] = if speed >= 4 {
            match best_card.0 {
                1 => &[5, 8],
                2 => &[6, 7],
                3 => &[8],
                _ => &[5, 6],
            }
        } else {
            &[5, 6, 7, 8]
        };
        for &m in adj_modes {
            predict_intra_block(
                bd, pred, 0, w, &edge, tl_idx, m, 0, w, h, have_left, have_top, 0, 0,
            );
            let sad = compute_sad(src, src_off, src_stride, pred, 0, w, w, h);
            let cost_sad = sad + lambda_sad;
            if cost_sad < top1.1 {
                top2 = top1;
                top1 = (m, cost_sad);
            } else if cost_sad < top2.1 {
                top2 = (m, cost_sad);
            }
        }
    }

    // Stage 2: Evaluate top 2 overall candidates with exact SIMD SATD.
    let mut best_mode = top1.0;
    let mut best_angle = 0i8;
    let mut best_cost = u64::MAX;
    for &(m, _) in &[top1, top2] {
        predict_intra_block(
            bd, pred, 0, w, &edge, tl_idx, m, 0, w, h, have_left, have_top, 0, 0,
        );
        let satd = compute_satd(src, src_off, src_stride, pred, 0, w, w, h);
        let rate_penalty = if m == 0 { lambda_sad } else { lambda_sad * 2 };
        let cost = satd + rate_penalty;
        if cost < best_cost {
            best_cost = cost;
            best_mode = m;
            best_angle = 0;
        }
    }

    // Stage 3: Hierarchical angle delta refinement (`±2` then `±1`) around best directional mode.
    if (1..=8).contains(&best_mode)
        && speed <= 6
        && best_cost > (w * h) as u64
        && best_cost < (w * h * 16) as u64
    {
        let base_dir_mode = best_mode;
        for delta in [-2i8, 2] {
            predict_intra_block(
                bd,
                pred,
                0,
                w,
                &edge,
                tl_idx,
                base_dir_mode,
                delta,
                w,
                h,
                have_left,
                have_top,
                0,
                0,
            );
            let satd = compute_satd(src, src_off, src_stride, pred, 0, w, w, h);
            let cost = satd + lambda_sad * 3;
            if cost < best_cost {
                best_cost = cost;
                best_angle = delta;
            }
        }
        let center = best_angle;
        for step in [-1i8, 1] {
            let delta = (center + step).clamp(-3, 3);
            if delta == center || delta == -2 || delta == 2 {
                continue;
            }
            predict_intra_block(
                bd,
                pred,
                0,
                w,
                &edge,
                tl_idx,
                base_dir_mode,
                delta,
                w,
                h,
                have_left,
                have_top,
                0,
                0,
            );
            let satd = compute_satd(src, src_off, src_stride, pred, 0, w, w, h);
            let cost = satd + lambda_sad * 3;
            if cost < best_cost {
                best_cost = cost;
                best_angle = delta;
            }
        }
    }

    let midx = if (1..=8).contains(&best_mode) {
        best_mode - 1
    } else {
        0xFF
    };
    (best_mode, best_angle, midx, best_cost)
}

/// Evaluates all 5 7-tap Filter Intra modes and returns `Some((best_fi_mode, best_cost))`
/// if Filter Intra beats `current_best_cost`.
#[allow(clippy::too_many_arguments)]
pub fn select_best_filter_intra_mode<BD: BitDepth>(
    bd: BD,
    src: &[BD::Pixel],
    src_off: usize,
    src_stride: usize,
    edge: &[BD::Pixel; 512],
    tl_idx: usize,
    w: usize,
    h: usize,
    lambda_sad: u64,
    current_best_cost: u64,
) -> Option<(FilterIntraMode, u64)> {
    if w > 32 || h > 32 {
        return None;
    }
    let mut pred_buf = [BD::Pixel::default(); 32 * 32];
    let pred = &mut pred_buf[..w * h];
    let mut best_fi = None;
    let mut best_cost = current_best_cost;

    for fi in [
        FilterIntraMode::Dc,
        FilterIntraMode::Vertical,
        FilterIntraMode::Horizontal,
        FilterIntraMode::Diagonal,
        FilterIntraMode::Paeth,
    ] {
        predict_filter_intra_block(bd, pred, 0, w, edge, tl_idx, fi, w, h);
        let satd = compute_satd(src, src_off, src_stride, pred, 0, w, w, h);
        let cost = satd + lambda_sad * 3;
        if cost < best_cost {
            best_cost = cost;
            best_fi = Some((fi, cost));
        }
    }
    best_fi
}

/// Selects the best spatial chroma intra prediction mode `(uv_mode, uv_angle)` across U and V planes.
#[allow(clippy::too_many_arguments)]
pub fn select_best_intra_mode_uv<BD: BitDepth>(
    bd: BD,
    src_u: &[BD::Pixel],
    src_v: &[BD::Pixel],
    src_off: usize,
    src_stride: usize,
    rec_u: &[BD::Pixel],
    rec_v: &[BD::Pixel],
    rec_off: usize,
    rec_stride: usize,
    have_left: bool,
    have_top: bool,
    cw: usize,
    ch: usize,
    _uvtx: usize,
    _luma_midx: u8,
    lambda_sad: u64,
    rec_w: usize,
    rec_h: usize,
) -> (u8, i8) {
    let mut edge_u = [BD::Pixel::default(); 512];
    let mut edge_v = [BD::Pixel::default(); 512];
    let tl_idx = 256usize;
    prepare_intra_edges(
        bd,
        &mut edge_u,
        tl_idx,
        rec_u,
        rec_off,
        rec_stride,
        have_left,
        have_top,
        false,
        false,
        cw,
        ch,
        rec_w,
        rec_h,
    );
    prepare_intra_edges(
        bd,
        &mut edge_v,
        tl_idx,
        rec_v,
        rec_off,
        rec_stride,
        have_left,
        have_top,
        false,
        false,
        cw,
        ch,
        rec_w,
        rec_h,
    );

    let mut pred_u_buf = [BD::Pixel::default(); 64 * 64];
    let mut pred_v_buf = [BD::Pixel::default(); 64 * 64];
    let pred_u = &mut pred_u_buf[..cw * ch];
    let pred_v = &mut pred_v_buf[..cw * ch];
    let mut best_mode = 0u8;
    let mut best_angle = 0i8;
    let mut best_cost = u64::MAX;

    for &m in &[0u8, 1, 2, 3, 4, 5] {
        if m != 0 && !have_left && !have_top {
            continue;
        }
        predict_intra_block(
            bd, pred_u, 0, cw, &edge_u, tl_idx, m, 0, cw, ch, have_left, have_top, 1, 0,
        );
        predict_intra_block(
            bd, pred_v, 0, cw, &edge_v, tl_idx, m, 0, cw, ch, have_left, have_top, 2, 0,
        );
        let sad_u = compute_sad(src_u, src_off, src_stride, pred_u, 0, cw, cw, ch);
        let sad_v = compute_sad(src_v, src_off, src_stride, pred_v, 0, cw, cw, ch);
        let cost = sad_u + sad_v + (if m == 0 { lambda_sad } else { lambda_sad * 2 });
        if cost < best_cost {
            best_cost = cost;
            best_mode = m;
            best_angle = 0;
        }
    }

    (best_mode, best_angle)
}

/// `Ii_Weights_1d[128]` (AV2 Specification v1.0.0 §7.13.3.29).
pub static II_WEIGHTS_1D: [u8; 128] = [
    60, 58, 56, 54, 52, 50, 48, 47, 45, 44, 42, 41, 39, 38, 37, 35, 34, 33, 32, 31, 30, 29, 28, 27,
    26, 25, 24, 23, 22, 22, 21, 20, 19, 19, 18, 18, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 12,
    11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4,
    4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

/// `interintra_mode` value for `II_DC_PRED` (AV2 §6.19.7.14, Table 6.24).
pub const II_DC_PRED: u8 = 0;
/// `interintra_mode` value for `II_V_PRED`.
pub const II_V_PRED: u8 = 1;
/// `interintra_mode` value for `II_H_PRED`.
pub const II_H_PRED: u8 = 2;
/// `interintra_mode` value for `II_SMOOTH_PRED`.
pub const II_SMOOTH_PRED: u8 = 3;

/// Intra prediction mode used by an `interintra_mode` (§5.20.7.22).
pub fn interintra_intra_mode(mode: u8) -> u8 {
    match mode {
        II_V_PRED => 1,
        II_H_PRED => 2,
        II_SMOOTH_PRED => 9,
        _ => 0,
    }
}

/// `intra_mode_variant_mask()` (§7.13.3.29) for a `w` x `h` plane region.
pub fn interintra_mask(mode: u8, w: usize, h: usize) -> Vec<u8> {
    let mut mask = vec![32u8; w * h];
    if mode == II_DC_PRED || w == 0 || h == 0 {
        return mask;
    }
    let size_scale = 128 / w.max(h);
    for i in 0..h {
        for j in 0..w {
            let idx = match mode {
                II_V_PRED => i * size_scale,
                II_H_PRED => j * size_scale,
                II_SMOOTH_PRED => i.min(j) * size_scale,
                _ => 0,
            };
            mask[i * w + j] = II_WEIGHTS_1D[idx.min(II_WEIGHTS_1D.len() - 1)];
        }
    }
    mask
}

/// `mask_blend()` (§7.13.3.30) for an inter-intra block: blends the saved
/// intra prediction `intra` with the inter prediction already in `dst`.
pub fn blend_interintra<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    dst_stride: usize,
    intra: &[BD::Pixel],
    mask: &[u8],
    w: usize,
    h: usize,
) {
    for y in 0..h {
        for x in 0..w {
            let m = mask[y * w + x] as i32;
            let pred0 = intra[y * w + x].to_i32();
            let pred1 = dst[dst_off + y * dst_stride + x].to_i32();
            let v = (m * pred0 + (64 - m) * pred1 + 32) >> 6;
            dst[dst_off + y * dst_stride + x] = bd.clip_pixel(v);
        }
    }
}
