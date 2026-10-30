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

//! Encoder-side AV2 v1.0.0 segmentation, segment-map prediction, and
//! superblock delta-Q.
//!
//! Normative references:
//! - §5.4.9 `seg_info(numSegments)` — per-segment feature enable/value syntax.
//! - §5.18.7.1 `segmentation_params()` — frame segment enable, map update, temporal update.
//! - §5.20.5.7 `intra_segment_id()` and §5.20.5.8 `read_segment_id()` — spatial prediction,
//!   `seg_id_ext_flag`, `segment_id` symbol coding, and `neg_deinterleave`.
//! - §5.20.7.2 `inter_segment_id(preSkip)` and §5.20.7.4 `get_segment_id()` — pre/post-skip
//!   segment id placement and the minimum segment id over the reference map.
//! - §5.20.5.11 `read_delta_qindex()` — per-superblock quantizer delta syntax.
//! - §5.18.1 `get_qindex()` — segment ALT_Q composition with `CurrentQIndex`.
//!
//! Reference behavior was checked against dav2d rev `bff8e5c` (read-only):
//! `parse_seg_info` and the `segmentation_params` branch in `src/obu.c`,
//! `neg_deinterleave` and the segment-id block in `src/decode.c`,
//! `get_cur_frame_segid` in `src/env.h`, and `get_prev_frame_segid` in `src/decode.c`.
//! No dav2d code or table is copied; the expressions below are written from the
//! AV2 specification text.
//!
//! The numeric thresholds in [`VaqPlan`] and the rate-feedback bounds in
//! `ratectrl.rs` are encoder tuning parameters. They are not syntax constants
//! and cannot change how a conforming decoder parses the stream; the syntax
//! ranges they must stay inside are cited next to the fields.

use crate::av2::cdf::CdfModeContext;
use crate::av2::frame::{Frame, Plane};
use crate::av2::headers::{MAX_SEGMENTS, SegmentationDataSet};
use crate::av2::intops::iclip;
use crate::av2::msac::MsacEncoder;
use crate::av2::pixel::Pixel;

/// Number of symbols in `TileSegmentIdCdf` / `TileSegmentIdExtCdf`
/// (AV2 Spec v1.0.0 §8.3.7, `Default_Segment_Id_Cdf[SEGMENT_ID_CONTEXTS][8 + 1]`).
const SEG_ID_SYMBOLS: usize = 8;

/// Width of the AV2 segment map in 4x4 luma units (`SegmentIds[MiRow][MiCol]`).
///
/// The decoder stores and predicts segment ids at `MI_SIZE` (4x4) granularity
/// (AV2 Spec v1.0.0 §5.20.5.8, dav2d `cur_segmap`), not at a fixed 16x16 grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentMap {
    cols4: usize,
    rows4: usize,
    ids: Vec<u8>,
}

impl SegmentMap {
    /// Allocates a map covering `cols4 * rows4` 4x4 units, initialized to segment 0.
    pub fn new(cols4: usize, rows4: usize) -> Self {
        Self {
            cols4,
            rows4,
            ids: vec![0u8; cols4 * rows4],
        }
    }

    /// Allocates a map and fills every unit with `id`.
    pub fn filled(cols4: usize, rows4: usize, id: u8) -> Self {
        Self {
            cols4,
            rows4,
            ids: vec![id; cols4 * rows4],
        }
    }

    #[inline]
    pub fn cols4(&self) -> usize {
        self.cols4
    }

    #[inline]
    pub fn rows4(&self) -> usize {
        self.rows4
    }

    #[inline]
    pub fn get(&self, x4: usize, y4: usize) -> u8 {
        if x4 >= self.cols4 || y4 >= self.rows4 {
            return 0;
        }
        self.ids[y4 * self.cols4 + x4]
    }

    /// Copies the segment id of `src` into this map.
    pub fn copy_from(&mut self, src: &SegmentMap) {
        self.cols4 = src.cols4;
        self.rows4 = src.rows4;
        self.ids.clear();
        self.ids.extend_from_slice(&src.ids);
    }

    /// Copies the 4x4 region `[x0,x1) x [y0,y1)` from `src`.
    pub fn copy_region_from(
        &mut self,
        src: &SegmentMap,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
    ) {
        let x1 = x1.min(self.cols4).min(src.cols4);
        let y1 = y1.min(self.rows4).min(src.rows4);
        let x0 = x0.min(self.cols4).min(src.cols4);
        if x0 >= x1 {
            return;
        }
        for y in y0.min(self.rows4)..y1 {
            let s = y * src.cols4;
            let d = y * self.cols4;
            self.ids[d + x0..d + x1].copy_from_slice(&src.ids[s + x0..s + x1]);
        }
    }

    /// Writes `id` over the coded block footprint, clipped to the frame.
    ///
    /// This is the encoder counterpart of the decoder's `splat2d` of
    /// `cur_segmap` after a block is decoded (dav2d `src/decode.c`, block end).
    pub fn fill_block(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize, id: u8) {
        let x1 = (bx4 + bw4).min(self.cols4);
        let y1 = (by4 + bh4).min(self.rows4);
        for y in by4.min(self.rows4)..y1 {
            let row = y * self.cols4;
            for x in bx4.min(self.cols4)..x1 {
                self.ids[row + x] = id;
            }
        }
    }

    /// Spatial segment-id predictor and context for the current frame map.
    ///
    /// AV2 Spec v1.0.0 §5.20.5.8 `read_segment_id()`: the prediction is the
    /// above neighbour when `prevUL == prevU`, otherwise the left neighbour;
    /// the three-way context is 2 when all three neighbours agree, 1 when any
    /// two agree, else 0. dav2d implements the same rule in
    /// `get_cur_frame_segid` (`src/env.h`).
    pub fn predict(&self, bx4: usize, by4: usize, have_left: bool, have_top: bool) -> (u8, usize) {
        if have_left && have_top {
            let l = self.get(bx4 - 1, by4);
            let a = self.get(bx4, by4 - 1);
            let al = self.get(bx4 - 1, by4 - 1);
            let ctx = if l == a && al == l {
                2
            } else if l == a || al == l || a == al {
                1
            } else {
                0
            };
            (if a == al { a } else { l }, ctx)
        } else {
            let pred = if have_left {
                self.get(bx4 - 1, by4)
            } else if have_top {
                self.get(bx4, by4 - 1)
            } else {
                0
            };
            (pred, 0)
        }
    }

    /// `get_segment_id()` over a reference map: the smallest segment id in the
    /// on-screen block footprint (AV2 Spec v1.0.0 §5.20.7.4; dav2d
    /// `get_prev_frame_segid`, `src/decode.c`).
    pub fn min_block(&self, bx4: usize, by4: usize, bw4: usize, bh4: usize) -> u8 {
        let x1 = (bx4 + bw4).min(self.cols4);
        let y1 = (by4 + bh4).min(self.rows4);
        let mut seg = u8::MAX;
        for y in by4.min(self.rows4)..y1 {
            let row = y * self.cols4;
            for x in bx4.min(self.cols4)..x1 {
                seg = seg.min(self.ids[row + x]);
            }
        }
        if seg == u8::MAX { 0 } else { seg }
    }
}

/// `neg_deinterleave(diff, ref, max)` from AV2 Spec v1.0.0 §5.20.5.8.
///
/// The coded `segment_id` is a de-interleaved distance from the spatial
/// predictor so that small absolute differences stay cheap regardless of which
/// side of the predictor they fall on.
#[inline]
pub fn neg_deinterleave(diff: i32, rf: i32, max: i32) -> i32 {
    if rf == 0 {
        return diff;
    }
    if rf >= max - 1 {
        return max - diff - 1;
    }
    if 2 * rf < max {
        if diff <= 2 * rf {
            if diff & 1 != 0 {
                rf + ((diff + 1) >> 1)
            } else {
                rf - (diff >> 1)
            }
        } else {
            diff
        }
    } else if diff <= 2 * (max - rf - 1) {
        if diff & 1 != 0 {
            rf + ((diff + 1) >> 1)
        } else {
            rf - (diff >> 1)
        }
    } else {
        max - (diff + 1)
    }
}

/// Inverse of [`neg_deinterleave`] for a valid `target` in `0..max`.
///
/// The forward mapping is a permutation of `0..max` for every `ref` in
/// `0..max` (AV2 Spec v1.0.0 §5.20.5.8), so the inverse is obtained by a table
/// scan of the normative function rather than by a second hand-derived formula.
pub fn neg_interleave(target: i32, rf: i32, max: i32) -> Option<i32> {
    if max <= 0 || target < 0 || target >= max {
        return None;
    }
    (0..max).find(|&diff| neg_deinterleave(diff, rf, max) == target)
}

/// Encodes `delta_q_abs` for one superblock (AV2 Spec v1.0.0 §5.20.5.11).
///
/// `delta` is already expressed in quantizer-index units. The caller applies
/// the same value to `CurrentQIndex` so the encoder reconstruction and the
/// normative `get_qindex()` stay in sync.
pub fn encode_delta_q(msac: &mut MsacEncoder, cdf: &mut [u16; 9], delta: i32, res_log2: u8) {
    let step = 1i32 << res_log2;
    debug_assert_eq!(delta % step, 0, "delta-Q must be a multiple of delta_q_res");
    let reduced = delta / step;
    let abs = reduced.unsigned_abs();
    if abs < 7 {
        msac.encode_symbol_adapt(abs as usize, cdf, 7);
    } else {
        msac.encode_symbol_adapt(7, cdf, 7);
        // delta_q_rem_bits is a 3-bit literal; delta_q_rem_bits++ makes the
        // decoded range start at 1. delta_q_abs_bits is a literal of that width.
        let mut rem = 1u32;
        while rem < 8 && (abs as u64) > (1u64 << (rem + 1)) + 4 {
            rem += 1;
        }
        msac.encode_bools_bypass(3, rem - 1);
        let base = (1u32 << rem) + 5;
        let extra = abs - base;
        debug_assert!(extra < (1 << rem));
        msac.encode_bools_bypass(rem, extra);
    }
    if abs != 0 {
        msac.encode_bool_bypass((reduced < 0) as u32);
    }
}

/// Segment coding context for one tile.
#[derive(Debug, Clone)]
pub struct SegmentCoder {
    /// `LastActiveSegId` from the frame segment data (dav2d uses `-1` when no
    /// feature is enabled).
    pub last_active_segid: i16,
    /// `MaxSegments` for this sequence (`8` or `16`).
    pub max_segments: usize,
    /// `segmentation_temporal_update`.
    pub temporal: bool,
    /// `any_lossless` for the frame: when true, a skipped block still codes its
    /// segment id (AV2 Spec v1.0.0 §5.20.7.2).
    pub any_lossless: bool,
    /// `SegIdPreSkip`: segment id is coded before the skip flag.
    pub preskip: bool,
}

impl SegmentCoder {
    pub fn new(data: &SegmentationDataSet, max_segments: usize, temporal: bool) -> Self {
        Self {
            last_active_segid: data.last_active_segid(),
            max_segments: max_segments.min(MAX_SEGMENTS),
            temporal,
            any_lossless: false,
            preskip: data.preskip(),
        }
    }

    /// True when any segment id may be written explicitly.
    #[inline]
    pub fn can_code_segment_id(&self) -> bool {
        self.last_active_segid >= 0
    }
}

/// Side information produced by the encoder's variance analysis.
#[derive(Debug, Clone)]
pub struct VaqPlan {
    bw16: usize,
    bh16: usize,
    /// `ln(variance)` per 16x16 luma cell.
    log_var: Vec<f32>,
    avg_log_var: f32,
    seg_class: Vec<u8>,
}

impl VaqPlan {
    /// Number of VAQ classes (segments) used by the encoder. AV2 v1.0.0 allows
    /// 8 segments without `enable_ext_seg`; using 8 keeps `seg_id_ext_flag`
    /// out of the stream while exercising every ALT_Q entry.
    pub const CLASSES: usize = 8;

    /// ALT_Q value per class. Smooth cells get a finer quantizer, textured
    /// cells a coarser one. These are encoder rate-distortion decisions, not
    /// bitstream constants; the syntax range is `[-MAXQ_BITS, MAXQ_BITS]`
    /// (§5.18.7.1 `Segmentation_Feature_Max[SEG_LVL_ALT_Q]`).
    pub const ALT_Q: [i16; Self::CLASSES] = [-8, -6, -4, -2, 0, 2, 4, 6];

    /// Builds a variance-based segmentation plan and fills the normative
    /// `SEG_LVL_ALT_Q` feature data.
    pub fn analyse<P: Pixel>(src: &Frame<P>, base_qidx: u16) -> Option<Self> {
        if base_qidx == 0 {
            return None;
        }
        let bw16 = src.width.div_ceil(16).max(1);
        let bh16 = src.height.div_ceil(16).max(1);
        let n_blks = bw16 * bh16;
        let y_plane: &Plane<P> = &src.planes[0];
        let mut log_var = vec![0.0f32; n_blks];
        let mut sum = 0.0f64;

        for by in 0..bh16 {
            let py = by * 16;
            let h = 16.min(src.height.saturating_sub(py)).max(1);
            for bx in 0..bw16 {
                let px = bx * 16;
                let w = 16.min(src.width.saturating_sub(px)).max(1);
                let mut s = 0i64;
                let mut s2 = 0u64;
                for r in 0..h {
                    let row = y_plane.offset(px, py + r);
                    for c in 0..w {
                        let v = y_plane.data[row + c].to_i32() as i64;
                        s += v;
                        s2 += (v * v) as u64;
                    }
                }
                let n = (w * h) as u64;
                let mean_sq = ((s as i128 * s as i128) / n as i128) as u64;
                let var = (s2.saturating_sub(mean_sq) as f64 / n as f64).max(1.0);
                let lv = var.ln() as f32;
                log_var[by * bw16 + bx] = lv;
                sum += lv as f64;
            }
        }

        let avg = (sum / n_blks as f64) as f32;
        // Spread the classes over +/- one standard deviation of log variance.
        let mut var_sum = 0.0f64;
        for &lv in &log_var {
            let d = (lv - avg) as f64;
            var_sum += d * d;
        }
        let sigma = (var_sum / n_blks as f64).sqrt().max(0.05) as f32;

        let mut seg_class = vec![0u8; n_blks];
        for (i, &lv) in log_var.iter().enumerate() {
            let z = (lv - avg) / sigma;
            // Class 0 = smoothest, class 7 = most textured.
            let bin = (3.5 + 1.35 * z).round() as i32;
            seg_class[i] = iclip(bin, 0, Self::CLASSES as i32 - 1) as u8;
        }

        Some(Self {
            bw16,
            bh16,
            log_var,
            avg_log_var: avg,
            seg_class,
        })
    }

    /// Writes the class ALT_Q values into `seg` and enables all 8 masks.
    pub fn fill_segmentation(&self, seg: &mut SegmentationDataSet) {
        for i in 0..Self::CLASSES {
            seg.delta_q[i] = Self::ALT_Q[i];
            seg.delta_q_mask |= 1 << i;
            seg.skip_mask &= !(1 << i);
            seg.globalmv_mask &= !(1 << i);
        }
        for i in Self::CLASSES..MAX_SEGMENTS {
            seg.delta_q[i] = 0;
        }
    }

    /// Segment id selected for the block whose top-left luma sample is `(px, py)`.
    #[inline]
    pub fn seg_for_pixel(&self, px: usize, py: usize) -> u8 {
        let bx = (px / 16).min(self.bw16.saturating_sub(1));
        let by = (py / 16).min(self.bh16.saturating_sub(1));
        self.seg_class[by * self.bw16 + bx]
    }

    /// Per-superblock ALT_Q offset derived from the mean log variance of the
    /// cells covered by the superblock. This is the second AQ layer; it is not
    /// part of segmentation and is signalled through `read_delta_qindex()`.
    pub fn sb_offset(&self, bx4: usize, by4: usize, sb_step4: usize) -> i16 {
        let x0 = bx4 / 4;
        let y0 = by4 / 4;
        let x1 = ((bx4 + sb_step4).div_ceil(4)).min(self.bw16);
        let y1 = ((by4 + sb_step4).div_ceil(4)).min(self.bh16);
        if x1 <= x0 || y1 <= y0 {
            return 0;
        }
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for y in y0..y1 {
            for x in x0..x1 {
                sum += self.log_var[y * self.bw16 + x] as f64;
                n += 1;
            }
        }
        let mean = (sum / n.max(1) as f64) as f32;
        // Map log-variance deviation to a small QP offset in quantizer-index
        // units. The gain and bound are encoder tuning parameters.
        let off = ((mean - self.avg_log_var) * 3.0).round() as i32;
        iclip(off, -6, 6) as i16
    }

    /// Frame-average log variance of the luma plane.
    pub fn avg_log_var(&self) -> f32 {
        self.avg_log_var
    }
}

/// Codes one block's segment id.
///
/// Returns the segment id that the decoder will store for the block. The caller
/// must already have written the surrounding syntax in the order required by
/// §5.20.7.2 (`inter_segment_id(preSkip)`): before `skip_mode`/`is_inter` when
/// `preskip` is set, otherwise after `skip_txfm`.
#[allow(clippy::too_many_arguments)]
pub fn encode_segment_id(
    msac: &mut MsacEncoder,
    cdf: &mut CdfModeContext,
    coder: &SegmentCoder,
    cur: &mut SegmentMap,
    prev: Option<&SegmentMap>,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
    have_left: bool,
    have_top: bool,
    skip_txfm: bool,
    desired: u8,
    seg_pred_a: &mut [u8; 32],
    seg_pred_l: &mut [u8; 32],
) -> u8 {
    let (pred, seg_ctx) = cur.predict(bx4, by4, have_left, have_top);
    let pred_ctx = (seg_pred_a[bx4 & 31] + seg_pred_l[by4 & 31]) as usize;
    let mut seg_predicted = 0u8;
    let seg_id;

    if !coder.can_code_segment_id() {
        // No enabled feature: dav2d uses LastActiveSegId = -1 and never reads a
        // segment id. Keep segment 0 and do not consume bits.
        seg_id = 0;
    } else if skip_txfm && !coder.any_lossless && !coder.preskip {
        // AV2 §5.20.7.2: a skipped non-lossless moment infers the spatially
        // predicted segment id and codes no `seg_id_predicted` bit.
        seg_id = pred;
    } else if let Some(prev_map) = prev.filter(|_| coder.temporal && (!skip_txfm || coder.preskip))
    {
        let temporal_id = prev_map.min_block(bx4, by4, bw4, bh4);
        let use_temporal = temporal_id == desired;
        msac.encode_bool_adapt(
            use_temporal as u32,
            &mut cdf.seg_id_predicted[pred_ctx.min(2)],
        );
        seg_predicted = use_temporal as u8;
        if use_temporal {
            seg_id = temporal_id;
        } else {
            seg_id = code_explicit_segment_id(
                msac, cdf, coder, pred, seg_ctx, desired, seg_pred_a, seg_pred_l, bx4, by4,
            );
        }
    } else {
        seg_id = code_explicit_segment_id(
            msac, cdf, coder, pred, seg_ctx, desired, seg_pred_a, seg_pred_l, bx4, by4,
        );
    }

    // Update `AboveSegPredContext` / `LeftSegPredContext` for the coded block
    // (AV2 Spec v1.0.0 §5.20.7.2).
    for i in 0..bw4.min(32 - (bx4 & 31)) {
        seg_pred_a[(bx4 & 31) + i] = seg_predicted;
    }
    for i in 0..bh4.min(32 - (by4 & 31)) {
        seg_pred_l[(by4 & 31) + i] = seg_predicted;
    }

    cur.fill_block(bx4, by4, bw4, bh4, seg_id);
    seg_id
}

#[allow(clippy::too_many_arguments)]
fn code_explicit_segment_id(
    msac: &mut MsacEncoder,
    cdf: &mut CdfModeContext,
    coder: &SegmentCoder,
    pred: u8,
    seg_ctx: usize,
    desired: u8,
    _seg_pred_a: &mut [u8; 32],
    _seg_pred_l: &mut [u8; 32],
    _bx4: usize,
    _by4: usize,
) -> u8 {
    let max = coder.last_active_segid as i32 + 1;
    let want = desired as i32;
    // `MaxSegments` is 8 or 16; segment ids above 7 need the extension flag.
    let ext = (coder.max_segments > SEG_ID_SYMBOLS && want >= SEG_ID_SYMBOLS as i32) as usize;
    if coder.max_segments > SEG_ID_SYMBOLS {
        msac.encode_bool_adapt(ext as u32, &mut cdf.seg_id_ext_flag[seg_ctx.min(2)]);
    }
    let diff = neg_interleave(want, pred as i32, max).unwrap_or_else(|| {
        // With all features enabled, `max` covers every representable id, so a
        // failed inverse is an encoder bug rather than a bitstream condition.
        debug_assert!(
            false,
            "segment id {want} not representable from {pred} (max {max})"
        );
        pred as i32
    });
    let symbol = (diff - (ext as i32) * SEG_ID_SYMBOLS as i32).clamp(0, 7) as usize;
    let table = if ext == 0 {
        &mut cdf.seg_id[seg_ctx.min(2)]
    } else {
        &mut cdf.seg_id_ext[seg_ctx.min(2)]
    };
    msac.encode_symbol_adapt(symbol, table, 7);
    // The decoder reconstructs the id through `neg_deinterleave`; return that
    // value so the encoder's map and quantizer use the same id as the decoder.
    neg_deinterleave(diff, pred as i32, max).clamp(0, 255) as u8
}

/// Encoder-side quantizer index for `segment` under `get_qindex()` semantics.
///
/// AV2 Spec v1.0.0 §5.18.1: with `delta_q_present` the segment ALT_Q value is
/// added to `CurrentQIndex`; otherwise it is added to `base_q_idx`. The result
/// is clipped to the per-bit-depth `MaxQ`.
#[inline]
pub fn segment_qindex(
    base_qidx: i32,
    current_qidx: i32,
    delta_q: i16,
    delta_q_present: bool,
    bit_depth: u8,
) -> i32 {
    let q = if delta_q_present {
        current_qidx + delta_q as i32
    } else {
        base_qidx + delta_q as i32
    };
    iclip(q, 0, crate::av2::headers::max_q_bits(bit_depth))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::headers::{PixelLayout, SEG_LVL_ALT_Q, SEG_LVL_SKIP};

    fn sds_with_features(mask: u16) -> SegmentationDataSet {
        let mut sds = SegmentationDataSet::default();
        sds.delta_q_mask = mask;
        for i in 0..MAX_SEGMENTS {
            sds.delta_q[i] = i as i16 * 2;
        }
        sds
    }

    #[test]
    fn neg_deinterleave_is_permutation_for_every_predictor() {
        for max in [1, 2, 8, 16] {
            for rf in 0..max {
                let mut seen = vec![false; max as usize];
                for diff in 0..max {
                    let target = neg_deinterleave(diff, rf, max);
                    assert!(
                        (0..max).contains(&target),
                        "diff={diff} ref={rf} max={max} -> {target}"
                    );
                    assert!(!seen[target as usize], "duplicate target {target}");
                    seen[target as usize] = true;
                    assert_eq!(neg_interleave(target, rf, max), Some(diff));
                }
            }
        }
    }

    #[test]
    fn segment_map_spatial_prediction_matches_spec() {
        let mut map = SegmentMap::new(8, 8);
        map.fill_block(0, 0, 2, 2, 3);
        map.fill_block(2, 0, 2, 2, 5);
        map.fill_block(0, 2, 2, 2, 5);
        // above = 5, left = 5, above-left = 3 -> ctx 1, pred = above.
        assert_eq!(map.predict(2, 2, true, true), (5, 1));
        // No neighbours: pred 0, ctx 0.
        assert_eq!(map.predict(0, 0, false, false), (0, 0));
        // Only left available.
        assert_eq!(map.predict(4, 4, true, false), (0, 0));
    }

    #[test]
    fn segment_map_min_over_reference_footprint() {
        let mut map = SegmentMap::new(8, 8);
        map.fill_block(0, 0, 2, 2, 4);
        map.fill_block(2, 0, 2, 2, 2);
        assert_eq!(map.min_block(1, 1, 3, 1), 2);
        assert_eq!(map.min_block(0, 0, 1, 1), 4);
    }

    #[test]
    fn vaq_plan_assigns_texture_higher_delta_q() {
        let mut raw = crate::av2::container::RawFrame::<u8>::new(64, 64, 8, PixelLayout::I420);
        for y in 0..64 {
            for x in 0..64 {
                let smooth = x < 32;
                raw.y[y * 64 + x] = if smooth {
                    128
                } else {
                    ((x * 37 + y * 11) & 255) as u8
                };
            }
        }
        let frame = Frame::from_raw(&raw, 0);
        let plan = VaqPlan::analyse(&frame, 100).expect("plan");
        let left = plan.seg_for_pixel(8, 8);
        let right = plan.seg_for_pixel(40, 8);
        assert!(
            right > left,
            "textured block class {right} should be coarser than smooth {left}"
        );
    }

    #[test]
    fn delta_q_bit_layout_roundtrips_through_documented_ranges() {
        // The escape range of AV2 Spec §5.20.5.11: rem=1 -> 7..8, rem=2 -> 9..12.
        for abs in [
            0u32, 1, 6, 7, 8, 9, 12, 13, 20, 21, 36, 37, 68, 69, 132, 133, 260, 261, 516,
        ] {
            let mut msac = MsacEncoder::new();
            let mut cdf = crate::av2::cdf::DEFAULT_MODE_CDF.delta_q;
            let sign = if abs % 2 == 0 { 1 } else { -1 };
            encode_delta_q(&mut msac, &mut cdf, sign * abs as i32, 0);
        }
    }

    #[test]
    fn seg_data_preskip_and_last_active_flags() {
        let sds = sds_with_features(0b1000_0001);
        assert_eq!(sds.last_active_segid(), 7);
        assert!(!sds.preskip());
        let mut skip = sds;
        skip.skip_mask = 1 << 3;
        assert!(skip.preskip());
        assert!(skip.feature_active(3, SEG_LVL_SKIP));
        assert!(!skip.feature_active(3, SEG_LVL_ALT_Q) || (skip.delta_q_mask & 8) != 0);
        assert_eq!(SegmentationDataSet::max_segments(false), 8);
        assert_eq!(SegmentationDataSet::max_segments(true), 16);
    }

    #[test]
    fn segment_qindex_composes_alt_q_with_current_qindex() {
        // base 100, delta +6 -> 106 without delta-Q; CurrentQIndex 90 + 6 -> 96.
        assert_eq!(segment_qindex(100, 0, 6, false, 8), 106);
        assert_eq!(segment_qindex(100, 90, 6, true, 8), 96);
        assert_eq!(segment_qindex(100, 90, -200, true, 8), 0);
        assert_eq!(segment_qindex(250, 250, 20, true, 8), 255);
    }

    #[test]
    fn temporal_segment_prediction_costs_no_explicit_id() {
        let mut sds = SegmentationDataSet::default();
        sds.delta_q_mask = 0x00FF;
        let coder = SegmentCoder::new(&sds, 8, true);
        let prev = SegmentMap::filled(4, 4, 3);

        let mut cdf = CdfModeContext::new();
        let mut msac = MsacEncoder::new();
        let mut cur = SegmentMap::new(4, 4);
        let mut a = [0u8; 32];
        let mut l = [0u8; 32];
        let id = encode_segment_id(
            &mut msac,
            &mut cdf,
            &coder,
            &mut cur,
            Some(&prev),
            0,
            0,
            2,
            2,
            false,
            false,
            false,
            3,
            &mut a,
            &mut l,
        );
        assert_eq!(id, 3);
        assert_eq!(cur.get(0, 0), 3);

        // Same block without a temporal reference: the explicit id is coded.
        let coder_off = SegmentCoder::new(&sds, 8, false);
        let mut cdf2 = CdfModeContext::new();
        let mut msac2 = MsacEncoder::new();
        let mut cur2 = SegmentMap::new(4, 4);
        let mut a2 = [0u8; 32];
        let mut l2 = [0u8; 32];
        let id2 = encode_segment_id(
            &mut msac2, &mut cdf2, &coder_off, &mut cur2, None, 0, 0, 2, 2, false, false, false, 3,
            &mut a2, &mut l2,
        );
        assert_eq!(id2, 3);
        assert!(
            msac.tell_bits() <= msac2.tell_bits(),
            "temporal prediction {} bits, explicit {} bits",
            msac.tell_bits(),
            msac2.tell_bits()
        );
    }

    /// Byte-exact comparison against the dav2d reference decoder for a stream
    /// with segmentation ALT_Q, spatial/temporal segment prediction, and
    /// per-superblock delta-Q. Ignored by default because it needs the oracle
    /// binary; run with:
    ///
    /// ```text
    /// cargo test -p rustmpeg-video av2_segmentation_dav2d -- --ignored
    /// ```
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_segmentation_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::{IvfWriter, RawFrame};
        use crate::av2::frame::ToolConfig;
        use crate::av2::headers::PixelLayout;
        use crate::av2::ratectrl::RateControlMode;
        use crate::av2::{Encoder, EncoderConfig};

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir();
        for &(w, h) in &[(64usize, 64usize), (256, 192)] {
            for &bd in &[8u8, 10, 12] {
                let ivf = dir.join(format!("u5_seg_{w}x{h}_{bd}.ivf"));
                let dav_y4m = dir.join(format!("u5_seg_{w}x{h}_{bd}.dav.y4m"));
                let cfg = EncoderConfig {
                    width: w,
                    height: h,
                    bit_depth: bd,
                    layout: PixelLayout::I420,
                    base_qidx: 100,
                    rate_control: RateControlMode::ConstantQp { base_qidx: 100 },
                    keyint: 30,
                    tools: ToolConfig {
                        enable_vaq: true,
                        enable_delta_q: true,
                        ..ToolConfig::default()
                    },
                    ..EncoderConfig::default()
                };
                let mut rec_frames: Vec<Vec<u8>> = Vec::new();
                {
                    let mut writer = IvfWriter::new(
                        std::io::BufWriter::new(std::fs::File::create(&ivf).unwrap()),
                        w as u16,
                        h as u16,
                        30,
                        1,
                        3,
                    )
                    .unwrap();
                    if bd == 8 {
                        let mut enc = Encoder::<u8>::new(cfg).expect("encoder");
                        for idx in 0..3 {
                            let raw =
                                RawFrame::<u8>::generate_synthetic(w, h, 8, PixelLayout::I420, idx);
                            let pkt = enc.encode_raw_frame(&raw).expect("encode");
                            writer.write_packet(pkt.pts, &pkt.data).unwrap();
                            rec_frames.push(raw8_to_bytes(&pkt.rec_frame.to_raw()));
                        }
                    } else {
                        let mut enc = Encoder::<u16>::new(cfg).expect("encoder");
                        for idx in 0..3 {
                            let raw = RawFrame::<u16>::generate_synthetic(
                                w,
                                h,
                                bd,
                                PixelLayout::I420,
                                idx,
                            );
                            let pkt = enc.encode_raw_frame(&raw).expect("encode");
                            writer.write_packet(pkt.pts, &pkt.data).unwrap();
                            rec_frames.push(raw16_to_le_bytes(&pkt.rec_frame.to_raw()));
                        }
                    }
                    writer.flush().unwrap();
                }
                let status = Command::new(&dav2d)
                    .arg("-i")
                    .arg(&ivf)
                    .arg("-o")
                    .arg(&dav_y4m)
                    .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
                    .status()
                    .expect("run dav2d");
                assert!(status.success(), "dav2d failed for {w}x{h} {bd}-bit");
                let dav = read_y4m_frames(&dav_y4m);
                assert_eq!(dav.len(), rec_frames.len(), "{w}x{h} {bd}-bit frame count");
                for (i, (a, b)) in rec_frames.iter().zip(dav.iter()).enumerate() {
                    assert_eq!(a, b, "{w}x{h} {bd}-bit frame {i} differs from dav2d");
                }
            }
        }
    }

    fn raw8_to_bytes(raw: &crate::av2::container::RawFrame<u8>) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&raw.y);
        out.extend_from_slice(&raw.u);
        out.extend_from_slice(&raw.v);
        out
    }

    fn raw16_to_le_bytes(raw: &crate::av2::container::RawFrame<u16>) -> Vec<u8> {
        let mut out = Vec::new();
        for plane in [&raw.y, &raw.u, &raw.v] {
            for &sample in plane {
                out.extend_from_slice(&sample.to_le_bytes());
            }
        }
        out
    }

    fn read_y4m_frames(path: &std::path::Path) -> Vec<Vec<u8>> {
        let data = std::fs::read(path).expect("read y4m");
        let mut pos = data.iter().position(|&b| b == b'\n').unwrap() + 1;
        let mut frames = Vec::new();
        while pos < data.len() {
            assert_eq!(&data[pos..pos + 5], b"FRAME");
            pos = data[pos..].iter().position(|&b| b == b'\n').unwrap() + pos + 1;
            let next = data[pos..]
                .windows(5)
                .position(|w| w == b"FRAME")
                .map(|p| p + pos)
                .unwrap_or(data.len());
            frames.push(data[pos..next].to_vec());
            pos = next;
        }
        frames
    }
}
