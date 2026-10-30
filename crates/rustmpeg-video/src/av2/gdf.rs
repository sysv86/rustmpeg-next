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

//! AV2 Guided Detail Filter (`GDF`, AV2 Spec v1.0.0 § 7.20.5).

use crate::av2::frame::{Frame, Plane};
use crate::av2::gdf_tables::{GDF_INTER_PARAMS, GDF_INTRA_PARAMS};
use crate::av2::headers::{AdaptiveBoolean, FrameHeader, SequenceHeader};
use crate::av2::intops::{apply_sign, iclip};
use crate::av2::ipred::compute_sse;
use crate::av2::pixel::{BitDepth, Pixel};

/// Normative 18 symmetric coordinate offsets for GDF guided filter taps (AV2 § 7.20.5).
pub static GDF_COORDS: [[i8; 2]; 18] = [
    [6, 0],
    [5, 0],
    [4, 0],
    [3, 0],
    [2, 1],
    [2, 0],
    [2, -1],
    [1, 2],
    [1, 1],
    [1, 0],
    [1, -1],
    [1, -2],
    [0, 6],
    [0, 5],
    [0, 4],
    [0, 3],
    [0, 2],
    [0, 1],
];

/// Computes the reference distance table index (0..4 for inter).
#[inline]
pub fn compute_gdf_ref_dst_idx(fh: &FrameHeader, ref_dists: &[i32]) -> usize {
    if fh.gdf.enabled == AdaptiveBoolean::Off || !fh.is_inter_or_switch() {
        return 0;
    }
    let mut max_dist = 0i32;
    for &d in ref_dists.iter().take((fh.n_ref_frames as usize).min(2)) {
        max_dist = max_dist.max(d.abs());
    }
    const REF_DST_IDX_TBL: [usize; 12] = [4, 0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4];
    REF_DST_IDX_TBL[(max_dist as usize).min(11)]
}

/// Fetches a luma sample for GDF filtering with normative boundary and stripe rules (AV2 § 7.20.1).
/// Samples within the stripe are taken from `cdef_luma`, outside from `orig_luma` (up to 2 lines).
/// Returns a 10-bit normalized sample.
#[inline(always)]
fn get_gdf_sample<P: Pixel>(
    cdef_luma: &Plane<P>,
    orig_luma: &Plane<P>,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    s_start: i32,
    s_end: i32,
    up_shift: u32,
    down_shift: u32,
) -> i32 {
    let cx = x.clamp(0, w.saturating_sub(1) as i32) as usize;
    let raw = if y < s_start {
        if s_start == 0 {
            cdef_luma.data[cx].to_i32()
        } else {
            let y_fetch = y.max(s_start - 2) as usize;
            orig_luma.data[y_fetch * orig_luma.stride + cx].to_i32()
        }
    } else if y > s_end {
        if s_end >= (h.saturating_sub(1) as i32) {
            cdef_luma.data[(h - 1) * cdef_luma.stride + cx].to_i32()
        } else {
            let y_fetch = y.min(s_end + 2) as usize;
            orig_luma.data[y_fetch * orig_luma.stride + cx].to_i32()
        }
    } else {
        cdef_luma.data[y as usize * cdef_luma.stride + cx].to_i32()
    };
    (raw << up_shift) >> down_shift
}

#[inline(always)]
fn gdf_sum(a: i32, b: i32, alpha: i32, m: i32) -> i32 {
    let diff_a = (a - m).clamp(-alpha, alpha);
    let diff_b = (b - m).clamp(-alpha, alpha);
    (diff_a + diff_b).clamp(-512, 511)
}

#[inline(always)]
fn swap_pair_if(cond: bool, a0: &mut i32, a1: &mut i32, b0: &mut i32, b1: &mut i32) {
    if cond {
        std::mem::swap(a0, b0);
        std::mem::swap(a1, b1);
    }
}

/// Applies normative AV2 Guided Detail Filter (`GDF`) to the reconstructed luma plane (AV2 § 7.20.5).
pub fn gdf_frame_normative<BD: BitDepth>(
    bd: BD,
    rec: &mut Frame<BD::Pixel>,
    cdef_luma: &Plane<BD::Pixel>,
    orig_luma: &Plane<BD::Pixel>,
    _seqhdr: &SequenceHeader,
    frame_hdr: &FrameHeader,
    ref_dists: &[i32],
    gdf_blks: &[bool],
) {
    if frame_hdr.gdf.enabled == AdaptiveBoolean::Off {
        return;
    }
    let qp_idx = (frame_hdr.gdf.qp_idx as usize).min(5);
    let params = if frame_hdr.is_inter_or_switch() {
        let ref_dst_idx = compute_gdf_ref_dst_idx(frame_hdr, ref_dists);
        &GDF_INTER_PARAMS[ref_dst_idx][qp_idx]
    } else {
        &GDF_INTRA_PARAMS[qp_idx]
    };

    let bitdepth = bd.bitdepth();
    let down_shift: u32 = if bitdepth == 12 { 2 } else { 0 };
    let up_shift: u32 = if bitdepth == 8 { 2 } else { 0 };
    let shift = 12 - bitdepth as i32;
    let rnd = if shift > 0 { 1 << (shift - 1) } else { 0 };
    let gdf_scale = frame_hdr.gdf.scale as i32;

    let w = rec.planes[0].width;
    let h = rec.planes[0].height;
    let stride = rec.planes[0].stride;

    let f_bw4 = 2 * ((frame_hdr.width.max(0) as usize + 7) >> 3);
    let gdf_step4 = 16 * frame_hdr.gdf.b64size.max(1) as usize;
    let gdf_cols = f_bw4.div_ceil(gdf_step4);

    // Iterate across vertical stripes.
    // The first stripe of the frame has height 56; subsequent stripes have height 64 (AV2 § 7.20.1).
    let mut stripe_start_y = 0usize;
    let mut is_first_stripe = true;

    static OFFS: [[i32; 2]; 4] = [[1, 0], [0, 1], [1, 1], [-1, 1]];

    while stripe_start_y < h {
        let stripe_h = if is_first_stripe {
            56usize.min(h - stripe_start_y)
        } else {
            64usize.min(h - stripe_start_y)
        };
        is_first_stripe = false;
        let stripe_end_y = stripe_start_y + stripe_h - 1;

        let s_start = stripe_start_y as i32;
        let s_end = stripe_end_y as i32;

        let mut y2 = stripe_start_y;
        while y2 < stripe_start_y + stripe_h {
            let cur_h = 2.min(h.saturating_sub(y2));

            let mut x2 = 0usize;
            while x2 < w {
                let cur_w = 2.min(w.saturating_sub(x2));

                // Check block-level enable flag (mapped to 64x64 or 128x128 superblocks).
                let uc = x2 / (gdf_step4 * 4);
                let ur = y2 / (gdf_step4 * 4);
                let blk_idx = ur * gdf_cols + uc;
                let is_enabled = if blk_idx < gdf_blks.len() {
                    gdf_blks[blk_idx]
                } else {
                    frame_hdr.gdf.enabled != AdaptiveBoolean::Off
                };
                if !is_enabled {
                    x2 += 2;
                    continue;
                }

                // Compute 4-directional gradient sums over the 4x4 window:
                // [-1, 2] in Y and [-1, 2] in X around (y2, x2).
                let mut grad_sums = [0i32; 4];
                for d in 0..4 {
                    let dy = OFFS[d][0];
                    let dx = OFFS[d][1];
                    let mut g = 0i32;
                    for sy in -1..=2 {
                        let y = y2 as i32 + sy;
                        for sx in -1..=2 {
                            let x = x2 as i32 + sx;
                            let a = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x - dx,
                                y - dy,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let b = get_gdf_sample(
                                cdef_luma, orig_luma, x, y, w, h, s_start, s_end, up_shift,
                                down_shift,
                            );
                            let c = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x + dx,
                                y + dy,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            g += (b * 2 - a - c).abs();
                        }
                    }
                    grad_sums[d] = g;
                }

                let mut bias = params.bias[2] as i32;
                for d in 0..4 {
                    let v = (grad_sums[d] >> 4).min(params.gradient_alpha[d] as i32);
                    bias += v * params.gradient_weight[d] as i32;
                }
                let filter_class = ((grad_sums[0] <= grad_sums[1]) as usize)
                    | (((grad_sums[2] <= grad_sums[3]) as usize) << 1);

                // Apply per-pixel GDF tap filtering to each pixel in this 2x2 block.
                for dy in 0..cur_h {
                    let y = y2 + dy;
                    let row_off = y * stride;
                    for dx in 0..cur_w {
                        let x = x2 + dx;
                        let m = get_gdf_sample(
                            cdef_luma, orig_luma, x as i32, y as i32, w, h, s_start, s_end,
                            up_shift, down_shift,
                        );

                        let mut idx_vals = [params.bias[0] as i32, params.bias[1] as i32, bias];

                        // Tap group k = 0..5 ({1, 0}, {0, 1} .. {6, 0}, {0, 6})
                        for k in 0..6 {
                            let mut above = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32,
                                y as i32 - (k as i32 + 1),
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut below = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32,
                                y as i32 + (k as i32 + 1),
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut left = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 - (k as i32 + 1),
                                y as i32,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut right = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 + (k as i32 + 1),
                                y as i32,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            swap_pair_if(
                                filter_class & 1 != 0,
                                &mut above,
                                &mut below,
                                &mut left,
                                &mut right,
                            );
                            let sum_v = gdf_sum(above, below, params.alpha[k][0] as i32, m);
                            let sum_h = gdf_sum(left, right, params.alpha[k][1] as i32, m);
                            if k < 2 {
                                for idx in 0..3 {
                                    idx_vals[idx] += sum_v * params.weight1[k * 3 + idx][0] as i32
                                        + sum_h * params.weight1[k * 3 + idx][1] as i32;
                                }
                            } else {
                                idx_vals[0] += sum_v * params.weight2[0][k - 2] as i32;
                                idx_vals[1] += sum_h * params.weight2[1][k - 2] as i32;
                                if k == 2 {
                                    idx_vals[2] += sum_v * params.weight1[6][0] as i32
                                        + sum_h * params.weight1[6][1] as i32;
                                }
                            }
                        }

                        // Diagonal 1 ({1, 1}, {1, -1})
                        {
                            let mut tl = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 - 1,
                                y as i32 - 1,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut br = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 + 1,
                                y as i32 + 1,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut tr = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 + 1,
                                y as i32 - 1,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut bl = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 - 1,
                                y as i32 + 1,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            swap_pair_if(filter_class & 2 != 0, &mut tl, &mut br, &mut tr, &mut bl);
                            let sum_dr = gdf_sum(tl, br, params.alpha[6][0] as i32, m);
                            let sum_dl = gdf_sum(tr, bl, params.alpha[6][1] as i32, m);
                            idx_vals[0] += sum_dr * params.weight3[0][0] as i32
                                + sum_dl * params.weight3[0][1] as i32;
                            idx_vals[1] += sum_dr * params.weight3[1][0] as i32
                                + sum_dl * params.weight3[1][1] as i32;
                        }

                        // Diagonal 2 ({1, 2}, {1, -2}, {2, 1}, {2, -1})
                        {
                            let mut ttl = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 - 1,
                                y as i32 - 2,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut bbr = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 + 1,
                                y as i32 + 2,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut tll = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 - 2,
                                y as i32 - 1,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut brr = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 + 2,
                                y as i32 + 1,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut ttr = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 + 1,
                                y as i32 - 2,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut bbl = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 - 1,
                                y as i32 + 2,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut trr = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 + 2,
                                y as i32 - 1,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            let mut bll = get_gdf_sample(
                                cdef_luma,
                                orig_luma,
                                x as i32 - 2,
                                y as i32 + 1,
                                w,
                                h,
                                s_start,
                                s_end,
                                up_shift,
                                down_shift,
                            );
                            swap_pair_if(
                                filter_class & 1 != 0,
                                &mut ttl,
                                &mut bbr,
                                &mut tll,
                                &mut brr,
                            );
                            swap_pair_if(
                                filter_class & 1 != 0,
                                &mut ttr,
                                &mut bbl,
                                &mut trr,
                                &mut bll,
                            );
                            swap_pair_if(
                                filter_class & 2 != 0,
                                &mut ttl,
                                &mut bbr,
                                &mut ttr,
                                &mut bbl,
                            );
                            swap_pair_if(
                                filter_class & 2 != 0,
                                &mut tll,
                                &mut brr,
                                &mut trr,
                                &mut bll,
                            );
                            let sum_ddr = gdf_sum(ttl, bbr, params.alpha[7][0] as i32, m);
                            let sum_ddl = gdf_sum(ttr, bbl, params.alpha[7][1] as i32, m);
                            let sum_drr = gdf_sum(tll, brr, params.alpha[8][0] as i32, m);
                            let sum_dll = gdf_sum(trr, bll, params.alpha[8][1] as i32, m);
                            idx_vals[0] += sum_ddr * params.weight4[0][0] as i32
                                + sum_ddl * params.weight4[0][1] as i32;
                            idx_vals[1] += sum_drr * params.weight4[1][0] as i32
                                + sum_dll * params.weight4[1][1] as i32;
                        }

                        let p_scale = params.scale as i32;
                        let mut full_idx = 0usize;
                        for idx in 0..3 {
                            let sv = idx_vals[idx] * p_scale;
                            let v = apply_sign((sv.abs() + (1 << 14)) >> 15, sv);
                            let sub_idx = (iclip(v, -p_scale, p_scale - 1) + p_scale) as usize;
                            full_idx = full_idx * (p_scale as usize * 2) + sub_idx;
                        }

                        let err = params.error_lut[full_idx] as i32;
                        let diff = err * gdf_scale;
                        let delta = apply_sign((diff.abs() + rnd) >> shift, diff);
                        let orig = rec.planes[0].data[row_off + x].to_i32();
                        rec.planes[0].data[row_off + x] = bd.clip_pixel(orig + delta);
                    }
                }

                x2 += 2;
            }
            y2 += 2;
        }

        stripe_start_y += stripe_h;
    }
}

/// Selects which GDF blocks are enabled from source luma activity before tile coding, so
/// the `use_gdf` flags can be written during entropy coding (AV2 Spec v1.0.0 §5.20.10.3).
/// `step4` is the GDF block size in 4x4 units; flat blocks are skipped.
pub fn select_gdf_units<P: Pixel>(
    src: &Frame<P>,
    unit_cols: usize,
    unit_rows: usize,
    step4: usize,
    bit_depth: u8,
) -> Vec<bool> {
    let plane = &src.planes[0];
    let block = step4 * 4;
    let threshold = 1i64 << (2 * bit_depth.saturating_sub(8) as u32);
    let mut out = vec![false; unit_cols * unit_rows];
    for ur in 0..unit_rows {
        let y0 = ur * block;
        if y0 >= plane.height {
            break;
        }
        let y1 = (y0 + block).min(plane.height);
        for uc in 0..unit_cols {
            let x0 = uc * block;
            if x0 >= plane.width {
                break;
            }
            let x1 = (x0 + block).min(plane.width);
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
                out[ur * unit_cols + uc] = sum_sq * n - sum * sum > threshold * n * n;
            }
        }
    }
    out
}

/// Evaluates and applies the normative AV2 Guided Detail Filter (`GDF`) across luma using
/// the per-block enable map fixed before tile coding. `gdf_blks` must match the flags that
/// were written into the bitstream; the parameter search only changes `scale`/`qp_idx`.
pub fn optimize_and_apply_gdf<BD: BitDepth>(
    bd: BD,
    src: &Frame<BD::Pixel>,
    rec: &mut Frame<BD::Pixel>,
    cdef_luma: &Plane<BD::Pixel>,
    orig_luma: &Plane<BD::Pixel>,
    seqhdr: &SequenceHeader,
    frame_hdr: &mut FrameHeader,
    ref_dists: &[i32],
    gdf_blks: &[bool],
) -> bool {
    if frame_hdr.gdf.enabled == AdaptiveBoolean::Off {
        return false;
    }
    let w = rec.width;
    let h = rec.height;

    let mut best_sse = u64::MAX;
    let mut best_scale = 1u8;
    let mut best_qp_idx = 0u8;
    let mut best_plane: Option<Plane<BD::Pixel>> = None;

    for scale in [1u8, 2u8] {
        for qp_offset in [0u8, 1u8] {
            let mut trial = rec.clone();
            let mut trial_fh = frame_hdr.clone();
            trial_fh.gdf.scale = scale;

            let qp_base = if frame_hdr.is_key_or_intra() { 85 } else { 110 };
            let qp_diff = frame_hdr.quant.yac as i32 - qp_base - 48 * seqhdr.hbd as i32;
            let base = iclip((qp_diff - 37) / 25, 0, 2);
            let qp_idx = (base + qp_offset as i32).clamp(0, 5) as u8;
            trial_fh.gdf.qp_idx = qp_idx;

            gdf_frame_normative(
                bd, &mut trial, cdef_luma, orig_luma, seqhdr, &trial_fh, ref_dists, gdf_blks,
            );

            let sse = compute_sse(
                &src.planes[0].data,
                0,
                src.planes[0].stride,
                &trial.planes[0].data,
                0,
                trial.planes[0].stride,
                w,
                h,
            );
            if sse < best_sse {
                best_sse = sse;
                best_scale = scale;
                best_qp_idx = qp_idx;
                best_plane = Some(trial.planes[0].clone());
            }
        }
    }

    if let Some(trial_plane) = best_plane {
        rec.planes[0] = trial_plane;
        frame_hdr.gdf.scale = best_scale;
        frame_hdr.gdf.qp_idx = best_qp_idx;
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::headers::{AdaptiveBoolean, FrameHeader, SequenceHeader};
    use crate::av2::pixel::BitDepth8;

    #[test]
    fn test_gdf_ref_dst_idx() {
        let mut fh = FrameHeader::default();
        fh.gdf.enabled = AdaptiveBoolean::On;
        fh.frame_type = crate::av2::headers::FrameType::Inter;
        fh.n_ref_frames = 2;

        let dists = [1, 2];
        assert_eq!(compute_gdf_ref_dst_idx(&fh, &dists), 1);

        let dists_zero = [0, 0];
        assert_eq!(compute_gdf_ref_dst_idx(&fh, &dists_zero), 4);
    }

    #[test]
    fn test_gdf_filter_runs_without_panic() {
        let w = 64;
        let h = 64;
        let mut rec = Frame::new(w, h, 8, crate::av2::headers::PixelLayout::I420, 0);
        let orig_luma = rec.planes[0].clone();
        let seq = SequenceHeader::default();
        let mut fh = FrameHeader::default();
        fh.gdf.enabled = AdaptiveBoolean::On;
        fh.gdf.scale = 1;
        fh.gdf.qp_idx = 0;
        let gdf_blks = vec![true; 16];

        let cdef_luma = rec.planes[0].clone();
        gdf_frame_normative(
            BitDepth8,
            &mut rec,
            &cdef_luma,
            &orig_luma,
            &seq,
            &fh,
            &[1, 1],
            &gdf_blks,
        );
        assert_eq!(rec.planes[0].width, 64);
    }

    #[test]
    fn test_gdf_high_bit_depth() {
        let w = 64;
        let h = 64;
        let mut rec = Frame::new(w, h, 10, crate::av2::headers::PixelLayout::I420, 0);
        let orig_luma = rec.planes[0].clone();
        let seq = SequenceHeader::default();
        let mut fh = FrameHeader::default();
        fh.gdf.enabled = AdaptiveBoolean::On;
        fh.gdf.scale = 2;
        fh.gdf.qp_idx = 1;
        let gdf_blks = vec![true; 16];

        let cdef_luma = rec.planes[0].clone();
        gdf_frame_normative(
            crate::av2::pixel::BitDepth16::new(10),
            &mut rec,
            &cdef_luma,
            &orig_luma,
            &seq,
            &fh,
            &[2, 2],
            &gdf_blks,
        );
        assert_eq!(rec.planes[0].height, 64);
    }
}
