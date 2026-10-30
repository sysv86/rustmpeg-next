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

//! Clean-room AV2 In-Loop Deblocking Filter (`deblock.rs`, AV2 Specification v1.0.0 §5.18.5.2, §7.17, §8.3).
//!
//! Computes frame-level and per-block deblocking parameters and applies normative boundary filtering
//! across 4x4 vertical and horizontal block edges according to transform sizes, prediction block
//! boundaries, skip flags, and lossless mode.

use crate::accelerator::AcceleratorError;
use crate::av2::frame::{Frame, Plane};
use crate::av2::headers::{
    FhDeblock, FhQuant, FrameHeader, FrameType, PixelLayout, SequenceHeader,
};
use crate::av2::intops::iclip;
use crate::av2::pixel::{BitDepth, Pixel};

/// Optional hardware implementation of the in-loop deblocking filter.
///
/// The codec calls this trait instead of [`deblock_frame`] when a provider is
/// installed on the encoder. An implementation must be bit-exact with
/// [`deblock_frame`] and must leave the planes untouched when it returns an
/// error, so the caller can fall back to the CPU path.
pub trait DeblockFilter<P: Pixel>: Send + Sync {
    /// Applies the deblocking filter to the three reconstructed planes.
    ///
    /// # Errors
    ///
    /// Returns [`AcceleratorError`] when the device is unavailable or a
    /// dispatch fails. The planes must not be modified in that case.
    fn deblock(
        &self,
        planes: [&mut Plane<P>; 3],
        params: &FhDeblock,
    ) -> Result<(), AcceleratorError>;
}

/// Derives the frame-level `FhDeblock` apply flags from base quantizer `qidx`.
///
/// AV2 does not signal a filter level: `apply_deblocking_filter` is a flag and
/// the actual strength is derived from the block qindex by the decoder
/// (AV2 Spec v1.0.0 §7.17.5/§7.17.6). The `level_*` fields below are therefore
/// non-zero sentinels, not an encoder-side RD level search; there is nothing to
/// search because a different value would not change the filtered samples.
pub fn derive_deblock_params(qidx: u16, frame_type: FrameType, lossless: bool) -> FhDeblock {
    let mut db = FhDeblock::default();
    if lossless || qidx <= 20 {
        return db;
    }
    // Scale filter strength smoothly with quantizer index above qidx = 20
    let base_lvl = match frame_type {
        FrameType::Key | FrameType::Intra => ((qidx as u32).saturating_sub(16) * 44) / 240,
        FrameType::Inter | FrameType::Switch => ((qidx as u32).saturating_sub(20) * 48) / 236,
    }
    .clamp(0, 63) as u8;

    db.level_y = [base_lvl, base_lvl];
    db.level_u = (base_lvl * 3) / 4;
    db.level_v = (base_lvl * 3) / 4;
    db.sharpness = if qidx < 80 { 1 } else { 0 };
    db
}

/// Evaluates whether the 4 samples `[p1, p0, q0, q1]` across a block boundary satisfy
/// the AV2 deblocking flatness/step thresholds (`blimit`, `limit`, `thresh`) and applies
/// 4-tap or 6-tap edge filtering in-place.
#[inline]
#[allow(clippy::too_many_arguments)]
fn filter_edge_sample<BD: BitDepth>(
    bd: BD,
    p2: i32,
    p1: i32,
    p0: i32,
    q0: i32,
    q1: i32,
    q2: i32,
    blimit: i32,
    limit: i32,
    thresh: i32,
) -> (BD::Pixel, BD::Pixel, BD::Pixel, BD::Pixel) {
    let mask = (p1 - p0).abs() <= limit
        && (q1 - q0).abs() <= limit
        && ((p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1)) <= blimit;

    if !mask {
        return (
            BD::Pixel::from_i32(p1),
            BD::Pixel::from_i32(p0),
            BD::Pixel::from_i32(q0),
            BD::Pixel::from_i32(q1),
        );
    }

    let flat = (p2 - p1).abs() <= thresh && (q2 - q1).abs() <= thresh;
    let hev = (p1 - p0).abs() > thresh || (q1 - q0).abs() > thresh;

    if flat && !hev {
        // 6-tap wide smooth filter across `[p1, p0, q0, q1]`
        let np1 = (p2 * 3 + p1 * 2 + p0 * 2 + q0 + 4) >> 3;
        let np0 = (p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + 4) >> 3;
        let nq0 = (p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + 4) >> 3;
        let nq1 = (p0 + q0 * 2 + q1 * 2 + q2 * 3 + 4) >> 3;
        (
            bd.clip_pixel(np1),
            bd.clip_pixel(np0),
            bd.clip_pixel(nq0),
            bd.clip_pixel(nq1),
        )
    } else {
        // 4-tap narrow filter across `[p1, p0, q0, q1]`
        let shift = (bd.bitdepth() - 8) as i32;
        let max_delta = 127i32 << shift;
        let ps1 = iclip(p1 - p0, -max_delta, max_delta);
        let qs1 = iclip(q1 - q0, -max_delta, max_delta);
        let delta_raw = if hev {
            3 * (q0 - p0) + iclip(p1 - q1, -max_delta, max_delta)
        } else {
            3 * (q0 - p0)
        };
        let d = iclip((delta_raw + 4) >> 3, -(16 << shift), 16 << shift);
        let np0 = bd.clip_pixel(p0 + d);
        let nq0 = bd.clip_pixel(q0 - d);
        let (np1, nq1) = if !hev {
            let d2 = (d + 1) >> 1;
            let _ = (ps1, qs1);
            (bd.clip_pixel(p1 + d2), bd.clip_pixel(q1 - d2))
        } else {
            (BD::Pixel::from_i32(p1), BD::Pixel::from_i32(q1))
        };
        (np1, np0, nq0, nq1)
    }
}

/// Filters all vertical and horizontal `step`-pixel grid boundaries on a single plane.
fn deblock_plane<BD: BitDepth>(
    bd: BD,
    plane: &mut Plane<BD::Pixel>,
    level: u8,
    sharpness: u8,
    grid_step: usize,
) {
    if level == 0 || plane.width < grid_step || plane.height < grid_step {
        return;
    }
    let shift = (bd.bitdepth() - 8) as i32;
    let lvl = level as i32;
    let limit = ((lvl >> (sharpness.min(2) as i32)).max(2)) << shift;
    let blimit = ((2 * lvl + limit) + (4 << shift)).max(4 << shift);
    let thresh = ((lvl >> 4).max(1)) << shift;

    // 1. Filter vertical edges (`x = grid_step, 2 * grid_step, ...`)
    let mut x = grid_step;
    while x + 3 <= plane.width {
        if x >= 3 {
            for y in 0..plane.height {
                let row = y * plane.stride;
                let p2 = plane.data[row + x - 3].to_i32();
                let p1 = plane.data[row + x - 2].to_i32();
                let p0 = plane.data[row + x - 1].to_i32();
                let q0 = plane.data[row + x].to_i32();
                let q1 = plane.data[row + x + 1].to_i32();
                let q2 = plane.data[row + x + 2].to_i32();
                let (np1, np0, nq0, nq1) =
                    filter_edge_sample(bd, p2, p1, p0, q0, q1, q2, blimit, limit, thresh);
                plane.data[row + x - 2] = np1;
                plane.data[row + x - 1] = np0;
                plane.data[row + x] = nq0;
                plane.data[row + x + 1] = nq1;
            }
        }
        x += grid_step;
    }

    // 2. Filter horizontal edges (`y = grid_step, 2 * grid_step, ...`)
    let s = plane.stride;
    let mut y = grid_step;
    while y + 3 <= plane.height {
        if y >= 3 {
            for c in 0..plane.width {
                let p2 = plane.data[(y - 3) * s + c].to_i32();
                let p1 = plane.data[(y - 2) * s + c].to_i32();
                let p0 = plane.data[(y - 1) * s + c].to_i32();
                let q0 = plane.data[y * s + c].to_i32();
                let q1 = plane.data[(y + 1) * s + c].to_i32();
                let q2 = plane.data[(y + 2) * s + c].to_i32();
                let (np1, np0, nq0, nq1) =
                    filter_edge_sample(bd, p2, p1, p0, q0, q1, q2, blimit, limit, thresh);
                plane.data[(y - 2) * s + c] = np1;
                plane.data[(y - 1) * s + c] = np0;
                plane.data[y * s + c] = nq0;
                plane.data[(y + 1) * s + c] = nq1;
            }
        }
        y += grid_step;
    }
}

/// Applies the simplified in-loop deblocking filter across all active planes of `rec`.
///
/// Kept for compatibility with [`DeblockFilter`] accelerator providers and baseline tests.
pub fn deblock_frame<BD: BitDepth>(bd: BD, rec: &mut Frame<BD::Pixel>, params: &FhDeblock) {
    let luma_lvl = params.level_y[0].max(params.level_y[1]);
    if luma_lvl == 0 {
        return;
    }
    deblock_plane(bd, &mut rec.planes[0], luma_lvl, params.sharpness, 8);
    if rec.layout != PixelLayout::I400 {
        let uv_step = 8usize >> rec.planes[1].ss_x.min(1);
        deblock_plane(
            bd,
            &mut rec.planes[1],
            params.level_u,
            params.sharpness,
            uv_step.max(4),
        );
        deblock_plane(
            bd,
            &mut rec.planes[2],
            params.level_v,
            params.sharpness,
            uv_step.max(4),
        );
    }
}

// ============================================================================
// Normative AV2 Deblocking Filter Implementation (AV2 Spec v1.0.0 §7.17)
// ============================================================================

/// AV2 Specification v1.0.0 §7.17.6 `Side_Thresholds` lookup table (296 entries).
pub static SIDE_THRESHOLDS: [i16; 296] = [
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -14, -13, -11, -10, -9, -7, -6, -4, -3, -2, 0, 0, 2, 3, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 19,
    20, 22, 23, 24, 26, 27, 29, 30, 32, 33, 34, 36, 37, 39, 40, 42, 43, 44, 46, 47, 49, 50, 51, 53,
    54, 56, 57, 59, 60, 61, 63, 64, 66, 67, 69, 70, 71, 73, 74, 76, 77, 78, 80, 81, 83, 84, 86, 87,
    88, 90, 91, 93, 94, 96, 101, 111, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230,
    240, 249, 259, 269, 279, 289, 299, 309, 319, 329, 339, 349, 359, 368, 378, 388, 398, 408, 418,
    428, 438, 448, 458, 468, 478, 488, 497, 507, 517, 527, 537, 547, 557, 567, 577, 587, 597, 607,
    616, 626, 636, 646, 656, 666, 676, 686, 696, 706, 716, 726, 736, 745, 755, 765, 775, 785, 795,
    805, 815, 825, 835, 845, 855, 864, 874, 884, 894, 904, 914, 924, 934, 944, 954, 964, 974, 984,
    993, 1003, 1013, 1023, 1033, 1043, 1053, 1063, 1073, 1083, 1093, 1103, 1112, 1122, 1132, 1142,
    1152, 1162, 1172, 1182, 1192, 1202, 1212, 1222, 1232, 1241, 1251, 1261, 1271, 1281, 1291, 1301,
    1311, 1321, 1331, 1341, 1351, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460,
    1470, 1480, 1489, 1499, 1509, 1519, 1529, 1539, 1549, 1559, 1569, 1579, 1589, 1599, 1608, 1618,
    1628, 1638, 1648, 1658, 1668, 1678,
];

/// Multipliers for wide filter transition check (§7.17.7.2).
pub const Q_FIRST: [i32; 3] = [45, 40, 32];

/// Multipliers for quantization threshold delta clamping (§7.17.7.1).
pub const Q_THRESH_MULTS: [i32; 8] = [32, 25, 19, 19, 0, 18, 0, 17];

/// Weight multipliers for sample correction blending (§7.17.7.1).
pub const W_MULT: [i32; 8] = [85, 51, 37, 28, 0, 20, 0, 15];

/// Dequantizer step lookup for AV2 deblocking threshold derivation (§7.17.5).
#[inline]
pub fn deblock_dq_lookup(qidx: i32, qmax: i32) -> i32 {
    let qidx = qidx.clamp(0, qmax);
    if qidx == 0 {
        return 64;
    }
    static DQ_LOOKUP_TBL: [u8; 24] = [
        40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 54, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76,
        78,
    ];
    let qidx = qidx - 1;
    let shift = qidx / 24;
    let idx = (qidx % 24) as usize;
    (DQ_LOOKUP_TBL[idx] as i32) << shift
}

/// Derives the quantization threshold `qThr` for level `lvl_seg` (§7.17.5).
#[inline]
pub fn deblock_quant_thr(lvl_seg: i32, bitdepth: u8) -> i32 {
    let hbd = (bitdepth as i32 - 8) / 2;
    let qmax = 255 + 48 * hbd;
    let dq = deblock_dq_lookup(lvl_seg, qmax);
    (dq + 4) >> 9
}

/// Derives the side threshold `side` for level `lvl_seg` (§7.17.6).
#[inline]
pub fn deblock_side_thr(lvl_seg: i32, bitdepth: u8) -> i32 {
    let bd = bitdepth as i32;
    let q_ind = (lvl_seg - 24 * (bd - 8)).clamp(0, 295) as usize;
    let side_val = SIDE_THRESHOLDS[q_ind] as i32;
    (side_val + (1 << (12 - bd))).max(0) >> (13 - bd)
}

/// Per-4x4 block unit deblocking properties recorded during decoding.
#[derive(Debug, Clone, Copy)]
pub struct BlockDeblockInfo {
    /// Transform size in pixels in horizontal direction (4, 8, 16, 32, 64).
    pub tx_w: u8,
    /// Transform size in pixels in vertical direction (4, 8, 16, 32, 64).
    pub tx_h: u8,
    /// Whether block has zero transform coefficients.
    pub skip: bool,
    /// Whether block is coded losslessly.
    pub lossless: bool,
    /// Whether left 4x4 boundary is a coding block edge.
    pub block_left: bool,
    /// Whether top 4x4 boundary is a coding block edge.
    pub block_top: bool,
    /// Whether left 4x4 boundary is a transform edge.
    pub tx_left: bool,
    /// Whether top 4x4 boundary is a transform edge.
    pub tx_top: bool,
}

impl Default for BlockDeblockInfo {
    fn default() -> Self {
        Self {
            tx_w: 4,
            tx_h: 4,
            skip: true,
            lossless: false,
            block_left: false,
            block_top: false,
            tx_left: false,
            tx_top: false,
        }
    }
}

/// Grid of per-4x4 block information for luma and chroma across the frame.
#[derive(Debug, Clone)]
pub struct DeblockGrid {
    /// Width in 4x4 units for luma.
    pub mi_w: usize,
    /// Height in 4x4 units for luma.
    pub mi_h: usize,
    /// Luma per-4x4 block info.
    pub luma: Vec<BlockDeblockInfo>,
    /// Width in 4x4 units for chroma.
    pub chroma_w: usize,
    /// Height in 4x4 units for chroma.
    pub chroma_h: usize,
    /// Chroma per-4x4 block info.
    pub chroma: Vec<BlockDeblockInfo>,
    /// Per-4x4 block non-zero coefficient flag for loop restoration (`lr_noskip`).
    pub lr_noskip: Vec<bool>,
}

impl DeblockGrid {
    /// Allocates an empty deblock grid for a frame with `mi_w` by `mi_h` 4x4 luma blocks.
    pub fn new(mi_w: usize, mi_h: usize, layout: PixelLayout) -> Self {
        let (ss_x, ss_y, has_chroma) = match layout {
            PixelLayout::I400 => (1, 1, false),
            PixelLayout::I420 => (1, 1, true),
            PixelLayout::I422 => (1, 0, true),
            PixelLayout::I444 => (0, 0, true),
        };
        let (cw, ch) = if has_chroma {
            (
                (mi_w + (1 << ss_x) - 1) >> ss_x,
                (mi_h + (1 << ss_y) - 1) >> ss_y,
            )
        } else {
            (0, 0)
        };
        Self {
            mi_w,
            mi_h,
            luma: vec![BlockDeblockInfo::default(); mi_w * mi_h],
            chroma_w: cw,
            chroma_h: ch,
            chroma: vec![BlockDeblockInfo::default(); cw * ch],
            lr_noskip: vec![false; mi_w * mi_h],
        }
    }

    /// Sets non-zero coefficient flag (`lr_noskip`) at `(x4, y4)`.
    #[inline]
    pub fn set_lr_noskip(&mut self, x4: usize, y4: usize, val: bool) {
        if x4 < self.mi_w && y4 < self.mi_h {
            self.lr_noskip[y4 * self.mi_w + x4] = val;
        }
    }

    /// Returns non-zero coefficient flag (`lr_noskip`) at `(x4, y4)`.
    #[inline]
    pub fn get_lr_noskip(&self, x4: usize, y4: usize) -> bool {
        if x4 < self.mi_w && y4 < self.mi_h {
            self.lr_noskip[y4 * self.mi_w + x4]
        } else {
            false
        }
    }

    /// Sets luma block info at `(x4, y4)`.
    #[inline]
    pub fn set_luma(&mut self, x4: usize, y4: usize, info: BlockDeblockInfo) {
        if x4 < self.mi_w && y4 < self.mi_h {
            self.luma[y4 * self.mi_w + x4] = info;
        }
    }

    /// Returns luma block info at `(x4, y4)`.
    #[inline]
    pub fn get_luma(&self, x4: usize, y4: usize) -> BlockDeblockInfo {
        if x4 < self.mi_w && y4 < self.mi_h {
            self.luma[y4 * self.mi_w + x4]
        } else {
            BlockDeblockInfo::default()
        }
    }

    /// Sets chroma block info at `(cx4, cy4)`.
    #[inline]
    pub fn set_chroma(&mut self, cx4: usize, cy4: usize, info: BlockDeblockInfo) {
        if cx4 < self.chroma_w && cy4 < self.chroma_h {
            self.chroma[cy4 * self.chroma_w + cx4] = info;
        }
    }

    /// Returns chroma block info at `(cx4, cy4)`.
    #[inline]
    pub fn get_chroma(&self, cx4: usize, cy4: usize) -> BlockDeblockInfo {
        if cx4 < self.chroma_w && cy4 < self.chroma_h {
            self.chroma[cy4 * self.chroma_w + cx4]
        } else {
            BlockDeblockInfo::default()
        }
    }
}

#[inline]
fn get_sample<P: Pixel>(data: &[P], base: usize, dist: isize, stride_b: isize) -> i32 {
    let idx = base as isize + dist * stride_b;
    if idx >= 0 && (idx as usize) < data.len() {
        data[idx as usize].to_i32()
    } else {
        0
    }
}

/// Normative AV2 filter choice algorithm (§7.17.7.2).
///
/// Determines the filter width (0, 1, 2, 3, 4, 6, or 8) based on second derivatives across
/// the 4-pixel segment ends `s` and `t`.
pub fn filter_choice<P: Pixel>(
    data: &[P],
    s_off: usize,
    t_off: usize,
    stride_b: isize,
    max_width_neg: usize,
    max_width_pos: usize,
    q_thr: i32,
    side_thr: i32,
) -> usize {
    let mut second_deriv = [0i32; 4];
    for dist in -2..=1 {
        let s_prev = get_sample(data, s_off, dist - 1, stride_b);
        let s_curr = get_sample(data, s_off, dist, stride_b);
        let s_next = get_sample(data, s_off, dist + 1, stride_b);
        let deriv_s = (s_prev - 2 * s_curr + s_next).abs();

        let t_prev = get_sample(data, t_off, dist - 1, stride_b);
        let t_curr = get_sample(data, t_off, dist, stride_b);
        let t_next = get_sample(data, t_off, dist + 1, stride_b);
        let deriv_t = (t_prev - 2 * t_curr + t_next).abs();

        second_deriv[(dist + 2) as usize] = (deriv_s + deriv_t + 1) >> 1;
    }

    let high_deriv = second_deriv[0].max(second_deriv[3]);
    if high_deriv > side_thr {
        return 0;
    }
    if max_width_pos == 1 {
        return 1;
    }

    let side_thr2 = side_thr >> 2;
    let mut transition = second_deriv[1] + second_deriv[2];
    if high_deriv > side_thr2 {
        return 1;
    }
    if transition > q_thr * 4 {
        return 1;
    }

    let side_thr3 = side_thr >> 3;
    if high_deriv > side_thr3 {
        return 2;
    }
    if transition > q_thr * 3 {
        return 2;
    }

    let end_thr = (side_thr * 3) >> 4;
    if max_width_neg >= 3 {
        let s_m1 = get_sample(data, s_off, -1, stride_b);
        let s_m4 = get_sample(data, s_off, -4, stride_b);
        let s_m2 = get_sample(data, s_off, -2, stride_b);
        let deriv_s = (s_m1 - s_m4 - 3 * (s_m1 - s_m2)).abs();

        let t_m1 = get_sample(data, t_off, -1, stride_b);
        let t_m4 = get_sample(data, t_off, -4, stride_b);
        let t_m2 = get_sample(data, t_off, -2, stride_b);
        let deriv_t = (t_m1 - t_m4 - 3 * (t_m1 - t_m2)).abs();

        if ((deriv_s + deriv_t + 1) >> 1) > end_thr {
            return 2;
        }
    }

    let s_0 = get_sample(data, s_off, 0, stride_b);
    let s_3 = get_sample(data, s_off, 3, stride_b);
    let s_1 = get_sample(data, s_off, 1, stride_b);
    let deriv_s = (s_0 - s_3 - 3 * (s_0 - s_1)).abs();

    let t_0 = get_sample(data, t_off, 0, stride_b);
    let t_3 = get_sample(data, t_off, 3, stride_b);
    let t_1 = get_sample(data, t_off, 1, stride_b);
    let deriv_t = (t_0 - t_3 - 3 * (t_0 - t_1)).abs();

    if ((deriv_s + deriv_t + 1) >> 1) > end_thr {
        return 2;
    }

    if max_width_pos == 3 {
        return 3;
    }

    transition <<= 4;
    let mut prev_dist = 3;
    let mut dist = 4;
    while dist <= max_width_pos {
        let q_first_idx = (dist - 4) >> 1;
        let q_thr4 = q_thr * Q_FIRST[q_first_idx];
        let end_thr4 = (side_thr * dist as i32) >> 4;

        if transition > q_thr4 {
            return prev_dist;
        }

        let dist2 = dist.min(7);
        if max_width_neg >= dist2 {
            let s_m1 = get_sample(data, s_off, -1, stride_b);
            let s_dist = get_sample(data, s_off, -(dist2 as isize) - 1, stride_b);
            let s_m2 = get_sample(data, s_off, -2, stride_b);
            let deriv_s = (s_m1 - s_dist - (dist2 as i32) * (s_m1 - s_m2)).abs();

            let t_m1 = get_sample(data, t_off, -1, stride_b);
            let t_dist = get_sample(data, t_off, -(dist2 as isize) - 1, stride_b);
            let t_m2 = get_sample(data, t_off, -2, stride_b);
            let deriv_t = (t_m1 - t_dist - (dist2 as i32) * (t_m1 - t_m2)).abs();

            if ((deriv_s + deriv_t + 1) >> 1) > end_thr4 {
                return prev_dist;
            }
        }

        let s_0 = get_sample(data, s_off, 0, stride_b);
        let s_dist = get_sample(data, s_off, dist2 as isize, stride_b);
        let s_1 = get_sample(data, s_off, 1, stride_b);
        let deriv_s = (s_0 - s_dist - (dist2 as i32) * (s_0 - s_1)).abs();

        let t_0 = get_sample(data, t_off, 0, stride_b);
        let t_dist = get_sample(data, t_off, dist2 as isize, stride_b);
        let t_1 = get_sample(data, t_off, 1, stride_b);
        let deriv_t = (t_0 - t_dist - (dist2 as i32) * (t_0 - t_1)).abs();

        if ((deriv_s + deriv_t + 1) >> 1) > end_thr4 {
            return prev_dist;
        }

        prev_dist = dist;
        dist += 2;
    }

    max_width_pos
}

/// Filters a 4-pixel edge segment (§7.17.7.1).
#[allow(clippy::too_many_arguments)]
fn filter_edge_segment<BD: BitDepth>(
    bd: BD,
    plane: &mut Plane<BD::Pixel>,
    base_off: usize,
    num_lines: usize,
    stride_a: usize,
    stride_b: isize,
    max_width_neg: usize,
    max_width_pos: usize,
    q_thr: i32,
    side_thr: i32,
    prev_lossless: bool,
    curr_lossless: bool,
) {
    if num_lines == 0 || (prev_lossless && curr_lossless) {
        return;
    }
    let s_off = base_off;
    let t_off = base_off + (num_lines - 1) * stride_a;

    let width = filter_choice(
        &plane.data,
        s_off,
        t_off,
        stride_b,
        max_width_neg,
        max_width_pos,
        q_thr,
        side_thr,
    );
    if width < 1 {
        return;
    }

    let width_neg = width.min(max_width_neg);
    let width_pos = width;
    let q_thr_clamp = q_thr * Q_THRESH_MULTS[width - 1];

    for line_idx in 0..num_lines {
        let line_off = base_off + line_idx * stride_a;
        let p1 = get_sample(&plane.data, line_off, -2, stride_b);
        let p0 = get_sample(&plane.data, line_off, -1, stride_b);
        let q0 = get_sample(&plane.data, line_off, 0, stride_b);
        let q1 = get_sample(&plane.data, line_off, 1, stride_b);

        let delta_raw = 4 * (3 * (q0 - p0) - (q1 - p1));
        let delta_m2 = delta_raw.clamp(-q_thr_clamp, q_thr_clamp);

        if !prev_lossless && width_neg > 0 {
            let delta_m2_neg = delta_m2 * W_MULT[width_neg - 1];
            for j in 0..width_neg {
                let diff = (delta_m2_neg * ((width_neg - j) as i32) + 1024) >> 11;
                let idx = line_off as isize - (j as isize + 1) * stride_b;
                if idx >= 0 && (idx as usize) < plane.data.len() {
                    let sample = plane.data[idx as usize].to_i32();
                    plane.data[idx as usize] = bd.clip_pixel(sample + diff);
                }
            }
        }

        if !curr_lossless && width_pos > 0 {
            let delta_m2_pos = delta_m2 * W_MULT[width_pos - 1];
            for j in 0..width_pos {
                let diff = (delta_m2_pos * ((width_pos - j) as i32) + 1024) >> 11;
                let idx = line_off as isize + (j as isize) * stride_b;
                if idx >= 0 && (idx as usize) < plane.data.len() {
                    let sample = plane.data[idx as usize].to_i32();
                    plane.data[idx as usize] = bd.clip_pixel(sample - diff);
                }
            }
        }
    }
}

/// Filters a single plane using the normative AV2 deblocking algorithm (§7.17).
#[allow(clippy::too_many_arguments)]
fn deblock_plane_normative<BD: BitDepth>(
    bd: BD,
    plane: &mut Plane<BD::Pixel>,
    plane_idx: usize,
    enable_v: bool,
    enable_h: bool,
    delta_q_v: i8,
    delta_q_h: i8,
    yac: u16,
    sub_pu_edge: bool,
    grid: &DeblockGrid,
    disable_across_tiles: bool,
    tile_col_starts: &[usize],
    tile_row_starts: &[usize],
) {
    if (!enable_v && !enable_h) || plane.width < 4 || plane.height < 4 {
        return;
    }
    // Deblocking runs over the coded extent (`MiCols * 4` x `MiRows * 4`), not
    // the visible picture: the bottom/right partial block is reconstructed and
    // filtered in full (AV2 Spec v1.0.0 §7.17.2). The extra samples live in the
    // padded plane storage.
    let plane_w = plane.coded_width.max(plane.width);
    let plane_h = plane.coded_height.max(plane.height);
    let is_chroma = plane_idx != 0;
    let (grid_w, grid_h) = if is_chroma {
        (grid.chroma_w, grid.chroma_h)
    } else {
        (grid.mi_w, grid.mi_h)
    };
    let get_block = |x4: usize, y4: usize| -> BlockDeblockInfo {
        if is_chroma {
            grid.get_chroma(x4, y4)
        } else {
            grid.get_luma(x4, y4)
        }
    };

    // Pass 0: Vertical edges (filtering horizontally)
    if enable_v {
        let lvl_seg = (yac as i32) + 8 * (delta_q_v as i32);
        let q_thr = deblock_quant_thr(lvl_seg, bd.bitdepth());
        let side_thr = deblock_side_thr(lvl_seg, bd.bitdepth());

        let stride_a = plane.stride;
        let stride_b = 1isize;
        let max_col4 = plane_w.div_ceil(4);

        for col4 in 1..max_col4.min(grid_w) {
            let x = col4 * 4;
            if x >= plane_w {
                break;
            }
            let is_tile_col = tile_col_starts.contains(&x);
            if disable_across_tiles && is_tile_col {
                continue;
            }
            let tile_vert_edge = is_tile_col;

            let mut y = 0;
            while y < plane_h {
                let y4 = y / 4;
                if y4 >= grid_h {
                    break;
                }
                let prev = get_block(col4 - 1, y4);
                let curr = get_block(col4, y4);

                let is_tx_edge = curr.tx_left;
                let is_block_edge = curr.block_left;
                // `Skips[row][col]` is the block on the positive side of the
                // edge (AV2 Spec v1.0.0 §7.17.2).
                let skip = curr.skip;

                let mut edge_q_thr = q_thr;
                let mut edge_side_thr = side_thr;
                if sub_pu_edge && !is_tx_edge {
                    edge_q_thr >>= 3;
                    edge_side_thr >>= 3;
                }

                // The reference decoder does not early-terminate when `qThr`
                // or `side` is zero; it lets `filter_choice` decide and the
                // clamp below then yields a zero delta. Match that behaviour
                // so the dav2d comparison is byte-exact (AV2 Spec v1.0.0
                // §7.17.7.2 describes the early return, but real 10/12-bit
                // streams at low qindex otherwise diverge by 1-3 LSB).
                let can_filter =
                    (is_tx_edge || sub_pu_edge) && (is_block_edge || !skip || sub_pu_edge);

                if can_filter {
                    let prev_tx_sz = prev.tx_w as usize;
                    let curr_tx_sz = curr.tx_w as usize;
                    let mut filter_size = prev_tx_sz.min(curr_tx_sz);

                    // Clamped near picture boundary (§7.17.3)
                    if !is_chroma {
                        if x + 16 > plane_w {
                            filter_size = filter_size.min(16);
                        }
                    } else if x + 8 > plane_w {
                        filter_size = filter_size.min(8);
                    }

                    let max_width_pos = match filter_size {
                        fs if fs <= 4 => 1,
                        8 => 3,
                        16 => {
                            if is_chroma {
                                4
                            } else {
                                6
                            }
                        }
                        _ => {
                            if is_chroma {
                                4
                            } else {
                                8
                            }
                        }
                    };

                    let sb_edge = tile_vert_edge;
                    let max_width_neg = if sb_edge {
                        max_width_pos.min(if is_chroma { 2 } else { 6 })
                    } else {
                        max_width_pos
                    };

                    let base_off = y * plane.stride + x;
                    let num_lines = (plane_h - y).min(4);

                    filter_edge_segment(
                        bd,
                        plane,
                        base_off,
                        num_lines,
                        stride_a,
                        stride_b,
                        max_width_neg,
                        max_width_pos,
                        edge_q_thr,
                        edge_side_thr,
                        prev.lossless,
                        curr.lossless,
                    );
                }
                y += 4;
            }
        }
    }

    // Pass 1: Horizontal edges (filtering vertically)
    if enable_h {
        let lvl_seg = (yac as i32) + 8 * (delta_q_h as i32);
        let q_thr = deblock_quant_thr(lvl_seg, bd.bitdepth());
        let side_thr = deblock_side_thr(lvl_seg, bd.bitdepth());

        let stride_a = 1usize;
        let stride_b = plane.stride as isize;
        let max_row4 = plane_h.div_ceil(4);

        for row4 in 1..max_row4.min(grid_h) {
            let y = row4 * 4;
            if y >= plane_h {
                break;
            }
            let is_tile_row = tile_row_starts.contains(&y);
            if disable_across_tiles && is_tile_row {
                continue;
            }

            let mut x = 0;
            while x < plane_w {
                let x4 = x / 4;
                if x4 >= grid_w {
                    break;
                }
                let prev = get_block(x4, row4 - 1);
                let curr = get_block(x4, row4);

                let is_tx_edge = curr.tx_top;
                let is_block_edge = curr.block_top;
                // `Skips[row][col]` is the block on the positive side of the
                // edge (AV2 Spec v1.0.0 §7.17.2).
                let skip = curr.skip;

                let mut edge_q_thr = q_thr;
                let mut edge_side_thr = side_thr;
                if sub_pu_edge && !is_tx_edge {
                    edge_q_thr >>= 3;
                    edge_side_thr >>= 3;
                }

                // Same zero-threshold behaviour as the vertical pass above.
                let can_filter =
                    (is_tx_edge || sub_pu_edge) && (is_block_edge || !skip || sub_pu_edge);

                if can_filter {
                    let prev_tx_sz = prev.tx_h as usize;
                    let curr_tx_sz = curr.tx_h as usize;
                    let mut filter_size = prev_tx_sz.min(curr_tx_sz);

                    // Clamped near picture boundary (§7.17.3)
                    if !is_chroma {
                        if y + 16 > plane_h {
                            filter_size = filter_size.min(16);
                        }
                    } else if y + 8 > plane_h {
                        filter_size = filter_size.min(8);
                    }

                    let max_width_pos = match filter_size {
                        fs if fs <= 4 => 1,
                        8 => 3,
                        16 => {
                            if is_chroma {
                                4
                            } else {
                                6
                            }
                        }
                        _ => {
                            if is_chroma {
                                4
                            } else {
                                8
                            }
                        }
                    };

                    let horz64_edge = (y % (64usize >> plane.ss_y.min(1))) == 0;
                    let sb_edge = horz64_edge;
                    let max_width_neg = if sb_edge {
                        max_width_pos.min(if is_chroma { 2 } else { 6 })
                    } else {
                        max_width_pos
                    };

                    let base_off = y * plane.stride + x;
                    let num_lines = (plane_w - x).min(4);

                    filter_edge_segment(
                        bd,
                        plane,
                        base_off,
                        num_lines,
                        stride_a,
                        stride_b,
                        max_width_neg,
                        max_width_pos,
                        edge_q_thr,
                        edge_side_thr,
                        prev.lossless,
                        curr.lossless,
                    );
                }
                x += 4;
            }
        }
    }
}

/// Applies normative AV2 in-loop deblocking across all active planes of `rec`.
#[allow(clippy::too_many_arguments)]
pub fn deblock_planes_normative<BD: BitDepth>(
    bd: BD,
    rec: &mut Frame<BD::Pixel>,
    params: &FhDeblock,
    quant: &FhQuant,
    grid: &DeblockGrid,
    disable_across_tiles: bool,
    tile_col_starts: &[usize],
    tile_row_starts: &[usize],
) {
    let sub_pu_edge = params.sub_pu != 0;
    // Plane 0: Luma (Y)
    deblock_plane_normative(
        bd,
        &mut rec.planes[0],
        0,
        params.level_y[0] != 0,
        params.level_y[1] != 0,
        params.delta_q_y[0],
        params.delta_q_y[1],
        quant.yac,
        sub_pu_edge,
        grid,
        disable_across_tiles,
        tile_col_starts,
        tile_row_starts,
    );

    if rec.layout != PixelLayout::I400 {
        let ss_x = rec.planes[1].ss_x;
        let ss_y = rec.planes[1].ss_y;
        let uv_col_starts: Vec<usize> = tile_col_starts.iter().map(|&c| c >> ss_x).collect();
        let uv_row_starts: Vec<usize> = tile_row_starts.iter().map(|&r| r >> ss_y).collect();

        // Plane 1: Cb (U). `apply_deblocking_filter[2]` enables both chroma
        // directions (AV2 Spec v1.0.0 §7.17.1).
        if params.level_u != 0 {
            let u_yac = (quant.yac as i32 + quant.uac_delta as i32).max(0) as u16;
            deblock_plane_normative(
                bd,
                &mut rec.planes[1],
                1,
                params.level_u != 0,
                params.level_u != 0,
                params.delta_q_u,
                params.delta_q_u,
                u_yac,
                sub_pu_edge,
                grid,
                disable_across_tiles,
                &uv_col_starts,
                &uv_row_starts,
            );
        }

        // Plane 2: Cr (V). `apply_deblocking_filter[3]` enables both chroma
        // directions (AV2 Spec v1.0.0 §7.17.1).
        if params.level_v != 0 {
            let v_yac = (quant.yac as i32 + quant.vac_delta as i32).max(0) as u16;
            deblock_plane_normative(
                bd,
                &mut rec.planes[2],
                2,
                params.level_v != 0,
                params.level_v != 0,
                params.delta_q_v,
                params.delta_q_v,
                v_yac,
                sub_pu_edge,
                grid,
                disable_across_tiles,
                &uv_col_starts,
                &uv_row_starts,
            );
        }
    }
}

/// Applies normative AV2 in-loop deblocking given sequence & frame headers and the decoded block grid.
pub fn deblock_frame_normative<BD: BitDepth>(
    bd: BD,
    rec: &mut Frame<BD::Pixel>,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    grid: &DeblockGrid,
) {
    let sb_step4 = if fh.sb128 != 0 { 32 } else { 16 };
    let n_cols = (fh.tiling.t.cols.max(1) as usize).max(1);
    let n_rows = (fh.tiling.t.rows.max(1) as usize).max(1);
    let mut tile_col_starts = Vec::with_capacity(n_cols);
    for tc in 0..n_cols {
        tile_col_starts.push(fh.tiling.t.col_start_sb[tc] as usize * sb_step4 * 4);
    }
    let mut tile_row_starts = Vec::with_capacity(n_rows);
    for tr in 0..n_rows {
        tile_row_starts.push(fh.tiling.t.row_start_sb[tr] as usize * sb_step4 * 4);
    }
    deblock_planes_normative(
        bd,
        rec,
        &fh.deblock,
        &fh.quant,
        grid,
        seq.disable_loopfilters_across_tiles,
        &tile_col_starts,
        &tile_row_starts,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::pixel::{BitDepth8, BitDepth16};

    #[test]
    fn test_threshold_derivation_8bit() {
        // BitDepth 8: qmax = 255
        let q0 = deblock_quant_thr(0, 8);
        assert_eq!(q0, 0); // (64 + 4) >> 9 == 0

        // At qidx = 20
        let q20 = deblock_quant_thr(20, 8);
        assert!(q20 >= 0);

        // At qidx = 100
        let q100 = deblock_quant_thr(100, 8);
        let s100 = deblock_side_thr(100, 8);
        assert!(q100 > 0);
        assert!(s100 > 0);

        // High qidx = 240
        let q240 = deblock_quant_thr(240, 8);
        let s240 = deblock_side_thr(240, 8);
        assert!(q240 > q100);
        assert!(s240 > s100);
    }

    #[test]
    fn test_threshold_derivation_10bit_and_12bit() {
        let q10 = deblock_quant_thr(150, 10);
        let s10 = deblock_side_thr(150, 10);
        assert!(q10 > 0);
        assert!(s10 > 0);

        let q12 = deblock_quant_thr(200, 12);
        let s12 = deblock_side_thr(200, 12);
        assert!(q12 > 0);
        assert!(s12 > 0);
    }

    #[test]
    fn test_filter_choice_flat_and_steep() {
        // Flat boundary: samples are all 128
        let flat_data = vec![128u8; 64];
        let choice_flat = filter_choice(&flat_data, 16, 16 + 8, 1, 8, 8, 10, 10);
        // Flat surface should choose maximum available filter width (8)
        assert_eq!(choice_flat, 8);

        // Step edge with flat sides: transition > q_thr * 4 clamps filter width to 1
        let mut step_data = vec![128u8; 64];
        for i in 0..16 {
            step_data[i] = 10;
        }
        for i in 16..32 {
            step_data[i] = 240;
        }
        let choice_step = filter_choice(&step_data, 16, 16, 1, 8, 8, 2, 2);
        assert_eq!(choice_step, 1);

        // Highly textured / rough side where second_deriv[-2] > side_thr disables filtering (returns 0)
        let mut rough_data = vec![128u8; 64];
        rough_data[16 - 3] = 10;
        rough_data[16 - 2] = 200;
        rough_data[16 - 1] = 10;
        let choice_rough = filter_choice(&rough_data, 16, 16, 1, 8, 8, 2, 2);
        assert_eq!(choice_rough, 0);
    }

    #[test]
    fn test_filter_edge_segment_smoothing() {
        let mut plane = Plane::<u8>::new(32, 32, 0, 0, 8);
        // Put a small step at x = 16: left = 120, right = 136
        for y in 0..32 {
            for x in 0..16 {
                plane.data[y * plane.stride + x] = 120;
            }
            for x in 16..32 {
                plane.data[y * plane.stride + x] = 136;
            }
        }

        let base_off = 16;
        let stride_a = plane.stride;
        let stride_b = 1;
        let q_thr = 8;
        let side_thr = 12;

        filter_edge_segment(
            BitDepth8, &mut plane, base_off, 4, stride_a, stride_b, 3, 3, q_thr, side_thr, false,
            false,
        );

        // After filtering, samples near boundary should be smoothed towards each other
        let p0 = plane.data[base_off - 1];
        let q0 = plane.data[base_off];
        assert!(p0 >= 120);
        assert!(q0 <= 136);
        assert!(q0 - p0 < 136 - 120);
    }

    #[test]
    fn test_deblock_planes_normative_all_layouts() {
        for layout in [
            PixelLayout::I400,
            PixelLayout::I420,
            PixelLayout::I422,
            PixelLayout::I444,
        ] {
            let mut frame = Frame::<u8>::new(32, 32, 8, layout, 0);
            for (idx, plane) in frame.planes.iter_mut().enumerate() {
                for (i, sample) in plane.data.iter_mut().enumerate() {
                    *sample = ((i * 17 + idx * 31) % 256) as u8;
                }
            }

            let mut grid = DeblockGrid::new(8, 8, layout);
            // Mark all 4x4 blocks as filtered
            for y in 0..grid.mi_h {
                for x in 0..grid.mi_w {
                    grid.set_luma(
                        x,
                        y,
                        BlockDeblockInfo {
                            tx_w: 8,
                            tx_h: 8,
                            skip: false,
                            lossless: false,
                            block_left: x % 2 == 0,
                            block_top: y % 2 == 0,
                            tx_left: x % 2 == 0,
                            tx_top: y % 2 == 0,
                        },
                    );
                }
            }
            if layout != PixelLayout::I400 {
                for y in 0..grid.chroma_h {
                    for x in 0..grid.chroma_w {
                        grid.set_chroma(
                            x,
                            y,
                            BlockDeblockInfo {
                                tx_w: 8,
                                tx_h: 8,
                                skip: false,
                                lossless: false,
                                block_left: true,
                                block_top: true,
                                tx_left: true,
                                tx_top: true,
                            },
                        );
                    }
                }
            }

            let params = FhDeblock {
                level_y: [1, 1],
                level_u: 1,
                level_v: 1,
                delta_q_y: [0, 0],
                delta_q_u: 0,
                delta_q_v: 0,
                sub_pu: 0,
                sharpness: 0,
            };
            let quant = FhQuant {
                yac: 100,
                ..Default::default()
            };

            deblock_planes_normative(
                BitDepth8,
                &mut frame,
                &params,
                &quant,
                &grid,
                false,
                &[],
                &[],
            );
        }
    }

    #[test]
    fn test_deblock_filters_step_edge_at_multiple_levels() {
        // A vertical step at x = 16 crosses a transform/block edge. The
        // normative qThr is zero at QP 20 (no filtering) and grows with the
        // quantizer index, so QP 100 and QP 140 must both change samples and
        // the higher quantizer must produce the stronger threshold.
        for bd_val in [8u8, 10, 12] {
            let q20 = deblock_quant_thr(20, bd_val);
            assert_eq!(q20, 0, "{bd_val}-bit QP 20 qThr must be zero");
            assert!(
                deblock_quant_thr(140, bd_val) > deblock_quant_thr(100, bd_val),
                "{bd_val}-bit qThr must increase from QP 100 to 140"
            );
            assert!(
                deblock_side_thr(140, bd_val) >= deblock_side_thr(100, bd_val),
                "{bd_val}-bit side threshold must not decrease"
            );
        }

        macro_rules! run {
            ($p:ty, $bd:expr) => {{
                let bd = $bd;
                let bd_val: u8 = bd.bitdepth();
                let mut changed = 0usize;
                for qidx in [100u16, 140] {
                    let mut frame = Frame::<$p>::new(32, 32, bd_val, PixelLayout::I420, 0);
                    for (pi, plane) in frame.planes.iter_mut().enumerate() {
                        let (w, h) = (plane.width, plane.height);
                        for y in 0..h {
                            for x in 0..w {
                                let v = if x < w / 2 {
                                    96 + pi as i32 * 8
                                } else {
                                    160 + pi as i32 * 8
                                };
                                plane.data[y * plane.stride + x] =
                                    bd.clip_pixel(v << (bd_val as i32 - 8));
                            }
                        }
                    }
                    let before = frame.planes[0].data.clone();
                    let mut grid = DeblockGrid::new(8, 8, PixelLayout::I420);
                    for y in 0..grid.mi_h {
                        for x in 0..grid.mi_w {
                            grid.set_luma(
                                x,
                                y,
                                BlockDeblockInfo {
                                    tx_w: 8,
                                    tx_h: 8,
                                    skip: false,
                                    lossless: false,
                                    block_left: x % 2 == 0,
                                    block_top: y % 2 == 0,
                                    tx_left: x % 2 == 0,
                                    tx_top: y % 2 == 0,
                                },
                            );
                        }
                    }
                    for y in 0..grid.chroma_h {
                        for x in 0..grid.chroma_w {
                            grid.set_chroma(
                                x,
                                y,
                                BlockDeblockInfo {
                                    tx_w: 4,
                                    tx_h: 4,
                                    skip: false,
                                    lossless: false,
                                    block_left: true,
                                    block_top: true,
                                    tx_left: true,
                                    tx_top: true,
                                },
                            );
                        }
                    }
                    let params = FhDeblock {
                        level_y: [1, 1],
                        level_u: 1,
                        level_v: 1,
                        ..Default::default()
                    };
                    let quant = FhQuant {
                        yac: qidx,
                        ..Default::default()
                    };
                    deblock_planes_normative(
                        bd,
                        &mut frame,
                        &params,
                        &quant,
                        &grid,
                        false,
                        &[],
                        &[],
                    );
                    let diff = before
                        .iter()
                        .zip(frame.planes[0].data.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                    assert!(diff > 0, "{bd_val}-bit q{qidx}: deblock changed no samples");
                    changed += diff;
                }
                assert!(changed > 0);
            }};
        }
        run!(u8, BitDepth8);
        run!(u16, BitDepth16::new(10));
        run!(u16, BitDepth16::new(12));
    }

    #[test]
    fn test_deblock_normative_high_bit_depth() {
        for bd_val in [10u8, 12] {
            let bd = BitDepth16::new(bd_val);
            let mut frame = Frame::<u16>::new(32, 32, bd_val, PixelLayout::I420, 0);
            for plane in &mut frame.planes {
                for (i, sample) in plane.data.iter_mut().enumerate() {
                    *sample = ((i * 37) % (1 << bd_val)) as u16;
                }
            }

            let mut grid = DeblockGrid::new(8, 8, PixelLayout::I420);
            for y in 0..grid.mi_h {
                for x in 0..grid.mi_w {
                    grid.set_luma(
                        x,
                        y,
                        BlockDeblockInfo {
                            tx_w: 8,
                            tx_h: 8,
                            skip: false,
                            lossless: false,
                            block_left: x % 2 == 0,
                            block_top: y % 2 == 0,
                            tx_left: x % 2 == 0,
                            tx_top: y % 2 == 0,
                        },
                    );
                }
            }
            for y in 0..grid.chroma_h {
                for x in 0..grid.chroma_w {
                    grid.set_chroma(
                        x,
                        y,
                        BlockDeblockInfo {
                            tx_w: 4,
                            tx_h: 4,
                            skip: false,
                            lossless: false,
                            block_left: true,
                            block_top: true,
                            tx_left: true,
                            tx_top: true,
                        },
                    );
                }
            }

            let params = FhDeblock {
                level_y: [1, 1],
                level_u: 1,
                level_v: 1,
                delta_q_y: [0, 0],
                delta_q_u: 0,
                delta_q_v: 0,
                sub_pu: 0,
                sharpness: 0,
            };
            let quant = FhQuant {
                yac: 120,
                ..Default::default()
            };

            deblock_planes_normative(bd, &mut frame, &params, &quant, &grid, false, &[], &[]);
        }
    }
}
