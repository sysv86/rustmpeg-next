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

//! Clean-room AV2 Chroma from Luma (`CfL`) Intra Prediction (AV2 Specification v1.0.0 §7.11.5).
//!
//! Computes zero-mean Q3 subsampled luma AC contributions from reconstructed luma samples,
//! finds optimal signed scaling parameters `(alpha_u, alpha_v)` in `[-16, 16]` via closed-form
//! least-squares initialization and RDO refinement, and synthesizes CfL chroma predictions.

use crate::av2::intops::iclip;
use crate::av2::ipred::{compute_sad, predict_intra_block, prepare_intra_edges};
use crate::av2::pixel::{BitDepth, Pixel};

/// Extracts subsampled Q3 (`8x` precision) zero-mean AC luma values `ac_q3` (`cw * ch` elements)
/// from the reconstructed luma plane `rec_y` at `(luma_x, luma_y)` with subsampling `(ss_x, ss_y)`.
#[allow(clippy::too_many_arguments)]
pub fn compute_cfl_ac_luma<P: Pixel>(
    rec_y: &[P],
    rec_y_stride: usize,
    luma_x: usize,
    luma_y: usize,
    cw: usize,
    ch: usize,
    ss_x: u8,
    ss_y: u8,
    ac_q3: &mut [i16],
) {
    let n = cw * ch;
    debug_assert!(ac_q3.len() >= n);
    let mut sum_q3 = 0i32;

    for cy in 0..ch {
        for cx in 0..cw {
            let lx = luma_x + (cx << ss_x);
            let ly = luma_y + (cy << ss_y);
            let val_q3 = match (ss_x, ss_y) {
                (1, 1) => {
                    // 4:2:0: sum of 2x2 luma samples (4 samples), shifted by `<< 1` to reach Q3 (8x)
                    let r0 = ly * rec_y_stride + lx;
                    let r1 = (ly + 1) * rec_y_stride + lx;
                    let s = rec_y[r0].to_i32()
                        + rec_y[r0 + 1].to_i32()
                        + rec_y[r1].to_i32()
                        + rec_y[r1 + 1].to_i32();
                    s << 1
                }
                (1, 0) => {
                    // 4:2:2: sum of 2x1 luma samples (2 samples), shifted by `<< 2` to reach Q3 (8x)
                    let r0 = ly * rec_y_stride + lx;
                    let s = rec_y[r0].to_i32() + rec_y[r0 + 1].to_i32();
                    s << 2
                }
                _ => {
                    // 4:4:4: single luma sample shifted by `<< 3` to reach Q3 (8x)
                    let r0 = ly * rec_y_stride + lx;
                    rec_y[r0].to_i32() << 3
                }
            };
            ac_q3[cy * cw + cx] = val_q3 as i16;
            sum_q3 += val_q3;
        }
    }

    // Subtract the block average to obtain the zero-mean Q3 AC luma. AV2
    // §7.13.2.12 uses `approx_divide(lumaSum, lumaCount)` for the average.
    let avg_q3 = crate::av2::intops::approx_divide(sum_q3, n as i32);

    for v in &mut ac_q3[..n] {
        *v = (*v as i32 - avg_q3).clamp(-4095, 4095) as i16;
    }
}

/// Synthesizes a CfL chroma prediction plane `dst` (`cw x ch`) by adding scaled Q3 luma AC
/// `(alpha * ac_q3 + 32) >> 6` onto the DC chroma prediction `dc_pred`.
#[allow(clippy::too_many_arguments)]
pub fn predict_cfl_plane<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    dst_stride: usize,
    dc_pred: &[BD::Pixel],
    ac_q3: &[i16],
    alpha: i8,
    cw: usize,
    ch: usize,
) {
    let a = alpha as i32;
    for r in 0..ch {
        let dst_row = dst_off + r * dst_stride;
        let idx_row = r * cw;
        for c in 0..cw {
            let dc = dc_pred[idx_row + c].to_i32();
            let ac = ac_q3[idx_row + c] as i32;
            let scaled = (a * ac + 32) >> 6;
            dst[dst_row + c] = bd.clip_pixel(dc + scaled);
        }
    }
}

/// Estimates the optimal signed CfL scaling factor `alpha in [-16, 16]` for a single chroma plane
/// using closed-form least-squares $\alpha \approx 8 \cdot \frac{\sum \text{ac} \cdot (C - DC)}{\sum \text{ac}^2}$
/// followed by local RDO search around the least-squares estimate.
#[allow(clippy::too_many_arguments)]
fn search_cfl_alpha_single_plane<BD: BitDepth>(
    bd: BD,
    src_c: &[BD::Pixel],
    src_off: usize,
    src_stride: usize,
    dc_pred: &[BD::Pixel],
    ac_q3: &[i16],
    cw: usize,
    ch: usize,
    lambda_sad: u64,
) -> (i8, u64) {
    let mut num = 0i64;
    let mut den = 0i64;
    for r in 0..ch {
        let s_row = src_off + r * src_stride;
        let p_row = r * cw;
        for c in 0..cw {
            let diff = (src_c[s_row + c].to_i32() - dc_pred[p_row + c].to_i32()) as i64;
            let ac = ac_q3[p_row + c] as i64;
            num += diff * ac;
            den += ac * ac;
        }
    }

    if den <= 64 {
        let sad0 = compute_sad(src_c, src_off, src_stride, dc_pred, 0, cw, cw, ch);
        return (0, sad0);
    }

    // Since `scaled = (alpha * ac_q3) / 64` and `diff ~ scaled`, `alpha ~ 64 * num / den`.
    let ls_alpha = iclip(((num * 64 + (den / 2)) / den) as i32, -16, 16) as i8;

    let mut cand_buf = vec![BD::Pixel::default(); cw * ch];
    let mut best_alpha = 0i8;
    let mut best_cost = compute_sad(src_c, src_off, src_stride, dc_pred, 0, cw, cw, ch);

    for delta in -2i8..=2i8 {
        let cand = (ls_alpha.saturating_add(delta)).clamp(-16, 16);
        if cand == 0 {
            continue;
        }
        predict_cfl_plane(bd, &mut cand_buf, 0, cw, dc_pred, ac_q3, cand, cw, ch);
        let sad = compute_sad(src_c, src_off, src_stride, &cand_buf, 0, cw, cw, ch);
        let rate_penalty = lambda_sad + (lambda_sad * (cand.unsigned_abs() as u64)) / 8;
        let cost = sad + rate_penalty;
        if cost < best_cost {
            best_cost = cost;
            best_alpha = cand;
        }
    }

    (best_alpha, best_cost)
}

/// Evaluates Chroma-from-Luma (`CfL`) for U and V planes jointly and returns
/// `Some((alpha_u, alpha_v, total_cost))` when CfL is valid (`(alpha_u, alpha_v) != (0, 0)`)
/// and within the allowed `cw <= 32 && ch <= 32` block size.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_cfl_uv<BD: BitDepth>(
    bd: BD,
    src_u: &[BD::Pixel],
    src_v: &[BD::Pixel],
    src_c_off: usize,
    src_c_stride: usize,
    rec_y: &[BD::Pixel],
    rec_y_stride: usize,
    rec_u: &[BD::Pixel],
    rec_v: &[BD::Pixel],
    rec_c_off: usize,
    rec_c_stride: usize,
    luma_x: usize,
    luma_y: usize,
    cw: usize,
    ch: usize,
    ss_x: u8,
    ss_y: u8,
    have_left: bool,
    have_top: bool,
    lambda_sad: u64,
    rec_w: usize,
    rec_h: usize,
) -> Option<(i8, i8, u64)> {
    if cw > 32 || ch > 32 || cw == 0 || ch == 0 {
        return None;
    }

    let mut ac_q3 = [0i16; 32 * 32];
    compute_cfl_ac_luma(
        rec_y,
        rec_y_stride,
        luma_x,
        luma_y,
        cw,
        ch,
        ss_x,
        ss_y,
        &mut ac_q3,
    );

    let mut edge_u = [BD::Pixel::default(); 512];
    let mut edge_v = [BD::Pixel::default(); 512];
    let tl_idx = 256usize;
    prepare_intra_edges(
        bd,
        &mut edge_u,
        tl_idx,
        rec_u,
        rec_c_off,
        rec_c_stride,
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
        rec_c_off,
        rec_c_stride,
        have_left,
        have_top,
        false,
        false,
        cw,
        ch,
        rec_w,
        rec_h,
    );

    let mut dc_u = vec![BD::Pixel::default(); cw * ch];
    let mut dc_v = vec![BD::Pixel::default(); cw * ch];
    predict_intra_block(
        bd, &mut dc_u, 0, cw, &edge_u, tl_idx, 0, 0, cw, ch, have_left, have_top, 1, 0,
    );
    predict_intra_block(
        bd, &mut dc_v, 0, cw, &edge_v, tl_idx, 0, 0, cw, ch, have_left, have_top, 2, 0,
    );

    let (alpha_u, cost_u) = search_cfl_alpha_single_plane(
        bd,
        src_u,
        src_c_off,
        src_c_stride,
        &dc_u,
        &ac_q3,
        cw,
        ch,
        lambda_sad,
    );
    let (alpha_v, cost_v) = search_cfl_alpha_single_plane(
        bd,
        src_v,
        src_c_off,
        src_c_stride,
        &dc_v,
        &ac_q3,
        cw,
        ch,
        lambda_sad,
    );

    if alpha_u == 0 && alpha_v == 0 {
        return None;
    }

    Some((alpha_u, alpha_v, cost_u + cost_v + lambda_sad * 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::pixel::BitDepth8;

    #[test]
    fn test_cfl_luma_ac_and_alpha_search() {
        let mut luma = [128u8; 8 * 8];
        let mut src_u = [128u8; 4 * 4];
        let mut src_v = [128u8; 4 * 4];
        for r in 0..8 {
            for c in 0..8 {
                let delta = if c < 4 { -20i32 } else { 20i32 };
                luma[r * 8 + c] = (128 + delta) as u8;
            }
        }
        for r in 0..4 {
            for c in 0..4 {
                let delta = if c < 2 { -10i32 } else { 10i32 };
                src_u[r * 4 + c] = (128 + delta) as u8;
                src_v[r * 4 + c] = (128 - delta) as u8;
            }
        }
        let rec_u = [128u8; 4 * 4];
        let rec_v = [128u8; 4 * 4];
        let res = evaluate_cfl_uv(
            BitDepth8, &src_u, &src_v, 0, 4, &luma, 8, &rec_u, &rec_v, 0, 4, 0, 0, 4, 4, 1, 1,
            false, false, 2, 64, 64,
        );
        assert!(res.is_some());
        let (au, av, _) = res.unwrap();
        assert!(au > 0);
        assert!(av < 0);
    }
}
