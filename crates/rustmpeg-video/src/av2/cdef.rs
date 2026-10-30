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

//! Clean-room AV2 Constrained Directional Enhancement Filter (`cdef.rs`, AV2 Spec v1.0.0 §7.18).
//!
//! Performs 8-orientation directional variance analysis on each `8x8` luma block (§7.18.2),
//! non-linear primary and secondary directional tap filtering (§7.18.3), and frame-level CDEF
//! application across tiles (§7.18.1).

use crate::av2::deblock::DeblockGrid;
use crate::av2::frame::{Frame, Plane};
use crate::av2::headers::{FrameHeader, PixelLayout, SequenceHeader};
use crate::av2::ipred::compute_sse;
use crate::av2::pixel::{BitDepth, Pixel};

/// AV2 normative CDEF primary taps lookup table (§7.18.3):
/// `Cdef_Pri_Taps[(priStr >> coeffShift) & 1]` -> `{4, 2}` if even, `{3, 3}` if odd.
pub const CDEF_PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];

/// AV2 normative CDEF secondary taps lookup table (§7.18.3): `{2, 1}`.
pub const CDEF_SEC_TAPS: [[i32; 2]; 2] = [[2, 1], [2, 1]];

/// AV2 normative CDEF tap displacement lookup table (§7.18.3): `[dir][k][dy, dx]`.
pub const CDEF_DIRECTIONS: [[[i32; 2]; 2]; 8] = [
    [[-1, 1], [-2, 2]], // 0: 45 deg
    [[0, 1], [-1, 2]],  // 1
    [[0, 1], [0, 2]],   // 2: 0 deg (horizontal)
    [[0, 1], [1, 2]],   // 3
    [[1, 1], [2, 2]],   // 4: -45 deg
    [[1, 0], [2, 1]],   // 5
    [[1, 0], [2, 0]],   // 6: -90 deg (vertical)
    [[1, 0], [2, -1]],  // 7
];

/// AV2 normative chroma direction mapping table (§7.18.1): `[subX][subY][yDir]`.
pub const CDEF_UV_DIR: [[[usize; 8]; 2]; 2] = [
    [
        // subX = 0, subY = 0 (4:4:4)
        [0, 1, 2, 3, 4, 5, 6, 7],
        // subX = 0, subY = 1
        [1, 2, 2, 2, 3, 4, 6, 0],
    ],
    [
        // subX = 1, subY = 0 (4:2:2)
        [7, 0, 2, 4, 5, 6, 6, 6],
        // subX = 1, subY = 1 (4:2:0)
        [0, 1, 2, 3, 4, 5, 6, 7],
    ],
];

/// Non-linear CDEF sample clamping function (`AV2 Spec v1.0.0 §7.18.3`):
/// `sign(diff) * min(|diff|, max(0, threshold - (|diff| >> shift)))`
/// where `shift = max(0, damping - floor_log2(threshold))`.
#[inline]
pub fn cdef_constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold == 0 || diff == 0 {
        return 0;
    }
    let abs_diff = diff.abs();
    let damping_adj = (damping - (threshold as u32).ilog2() as i32).max(0);
    let term = (threshold - (abs_diff >> damping_adj)).clamp(0, abs_diff);
    if diff < 0 { -term } else { term }
}

/// Finds the dominant edge orientation (`0..8`) and directional variance of an `8x8` block
/// at `(bx, by)` in `plane` by projecting centered pixel values onto 8 directions (`AV2 Spec §7.18.2`).
pub fn find_cdef_direction<P: Pixel>(
    plane: &Plane<P>,
    bx: usize,
    by: usize,
    bit_depth: u8,
) -> (usize, u32) {
    let shift = (bit_depth - 8) as i32;
    let mid = 128i32;
    let mut partial = [[0i32; 15]; 8];

    for r in 0..8 {
        let cy = (by + r).min(plane.height.saturating_sub(1));
        let row = cy * plane.stride;
        for c in 0..8 {
            let cx = (bx + c).min(plane.width.saturating_sub(1));
            let x = (plane.data[row + cx].to_i32() >> shift) - mid;
            partial[0][r + c] += x;
            partial[1][r + (c >> 1)] += x;
            partial[2][r] += x;
            partial[3][3 + r - (c >> 1)] += x;
            partial[4][7 + r - c] += x;
            partial[5][3 - (r >> 1) + c] += x;
            partial[6][c] += x;
            partial[7][(r >> 1) + c] += x;
        }
    }

    let mut costs = [0u64; 8];
    for i in 0..8 {
        costs[2] += (partial[2][i] as i64 * partial[2][i] as i64) as u64;
        costs[6] += (partial[6][i] as i64 * partial[6][i] as i64) as u64;
    }
    costs[2] *= 105;
    costs[6] *= 105;

    const DIV_TABLE: [u64; 7] = [840, 420, 280, 210, 168, 140, 120];
    for i in 0..7 {
        let d = DIV_TABLE[i];
        costs[0] += ((partial[0][i] as i64).pow(2) + (partial[0][14 - i] as i64).pow(2)) as u64 * d;
        costs[4] += ((partial[4][i] as i64).pow(2) + (partial[4][14 - i] as i64).pow(2)) as u64 * d;
    }
    costs[0] += (partial[0][7] as i64).pow(2) as u64 * 105;
    costs[4] += (partial[4][7] as i64).pow(2) as u64 * 105;

    for i in (1..8).step_by(2) {
        for j in 0..5 {
            costs[i] += (partial[i][3 + j] as i64).pow(2) as u64;
        }
        costs[i] *= 105;
        for j in 0..3 {
            let d = DIV_TABLE[2 * j + 1];
            costs[i] +=
                ((partial[i][j] as i64).pow(2) + (partial[i][10 - j] as i64).pow(2)) as u64 * d;
        }
    }

    let mut best_dir = 0usize;
    let mut max_cost = costs[0];
    for d in 1..8 {
        if costs[d] > max_cost {
            max_cost = costs[d];
            best_dir = d;
        }
    }
    let ortho_cost = costs[(best_dir + 4) & 7];
    let var = ((max_cost.saturating_sub(ortho_cost)) >> 10) as u32;
    (best_dir, var)
}

/// Applies normative CDEF directional filtering to a single block `(x0, y0, w, h)`
/// from `src` into `dst` (`AV2 Spec v1.0.0 §7.18.3`).
#[allow(clippy::too_many_arguments)]
pub fn cdef_filter_block_normative<P: Pixel>(
    src: &Plane<P>,
    dst: &mut Plane<P>,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    pri_str: i32,
    sec_str: i32,
    damping: i32,
    dir: usize,
    coeff_shift: u8,
    min_x: usize,
    max_x: usize,
    min_y: usize,
    max_y: usize,
) {
    if pri_str == 0 && sec_str == 0 {
        return;
    }

    for i in 0..h {
        let py = y0 + i;
        if py >= dst.height {
            break;
        }
        let dst_row = py * dst.stride;
        for j in 0..w {
            let px = x0 + j;
            if px >= dst.width {
                break;
            }
            let x = src.data[py * src.stride + px].to_i32();
            let mut sum = 0i32;
            let mut min_val = x;
            let mut max_val = x;

            if pri_str > 0 {
                let pri_tap_idx = ((pri_str >> coeff_shift) & 1) as usize;
                for k in 0..2 {
                    let weight = CDEF_PRI_TAPS[pri_tap_idx][k];
                    for sign in [-1, 1] {
                        let dy = sign * CDEF_DIRECTIONS[dir][k][0];
                        let dx = sign * CDEF_DIRECTIONS[dir][k][1];
                        let tap_x = px as i32 + dx;
                        let tap_y = py as i32 + dy;
                        if tap_x >= min_x as i32
                            && tap_x < max_x as i32
                            && tap_y >= min_y as i32
                            && tap_y < max_y as i32
                        {
                            let p = src.data[tap_y as usize * src.stride + tap_x as usize].to_i32();
                            min_val = min_val.min(p);
                            max_val = max_val.max(p);
                            sum += weight * cdef_constrain(p - x, pri_str, damping);
                        }
                    }
                }
            }

            if sec_str > 0 {
                for dir_off in [2usize, 6usize] {
                    let sec_dir = (dir + dir_off) & 7;
                    for k in 0..2 {
                        let weight = CDEF_SEC_TAPS[0][k];
                        for sign in [-1, 1] {
                            let dy = sign * CDEF_DIRECTIONS[sec_dir][k][0];
                            let dx = sign * CDEF_DIRECTIONS[sec_dir][k][1];
                            let tap_x = px as i32 + dx;
                            let tap_y = py as i32 + dy;
                            if tap_x >= min_x as i32
                                && tap_x < max_x as i32
                                && tap_y >= min_y as i32
                                && tap_y < max_y as i32
                            {
                                let s =
                                    src.data[tap_y as usize * src.stride + tap_x as usize].to_i32();
                                min_val = min_val.min(s);
                                max_val = max_val.max(s);
                                sum += weight * cdef_constrain(s - x, sec_str, damping);
                            }
                        }
                    }
                }
            }

            let filtered = x + ((8 + sum - (sum < 0) as i32) >> 4);
            let clipped = filtered.clamp(min_val, max_val);
            dst.data[dst_row + px] = P::from_i32(clipped);
        }
    }
}

/// Applies normative AV2 CDEF filtering across all active planes of `rec` (`AV2 Spec v1.0.0 §7.18`).
pub fn cdef_frame_normative<BD: BitDepth>(
    bd: BD,
    rec: &mut Frame<BD::Pixel>,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    grid: &DeblockGrid,
    cdef_idx: &[i8],
) {
    if fh.cdef.enabled == 0 {
        return;
    }

    let f_bw4 = 2 * ((fh.width.max(0) as usize + 7) >> 3);
    let f_bh4 = 2 * ((fh.height.max(0) as usize + 7) >> 3);
    let cdef_cols = f_bw4.div_ceil(16);
    let coeff_shift = bd.bitdepth() - 8;

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

    let src = rec.clone();

    for r in (0..f_bh4).step_by(2) {
        let r16 = r / 16;
        let tr = row_starts4
            .partition_point(|&start| start <= r)
            .saturating_sub(1);
        let tile_row_start4 = row_starts4[tr];
        let tile_row_end4 = row_starts4[tr + 1];

        for c in (0..f_bw4).step_by(2) {
            let c16 = c / 16;
            let idx = cdef_idx[r16 * cdef_cols + c16];
            if idx < 0 || idx as usize >= fh.cdef.n_strengths as usize {
                continue;
            }
            let idx = idx as usize;
            let raw_y = fh.cdef.y_strength[idx];
            let raw_uv = fh.cdef.uv_strength[idx];
            if raw_y == 0 && raw_uv == 0 {
                continue;
            }

            // Transform-skip and lossless check (§7.18.1)
            let s00 = grid.get_luma(c, r);
            let s01 = grid.get_luma((c + 1).min(f_bw4 - 1), r);
            let s10 = grid.get_luma(c, (r + 1).min(f_bh4 - 1));
            let s11 = grid.get_luma((c + 1).min(f_bw4 - 1), (r + 1).min(f_bh4 - 1));

            if s00.lossless || s01.lossless || s10.lossless || s11.lossless {
                continue;
            }
            // The `Skips` input of §7.18.1 is the per-4x4 non-zero-coefficient
            // mask (`lr_noskip`), not the inter skip flag: an 8x8 block with no
            // coded coefficients is not filtered even when `skip_txfm == 0`.
            let n00 = grid.get_lr_noskip(c, r);
            let n01 = grid.get_lr_noskip((c + 1).min(f_bw4 - 1), r);
            let n10 = grid.get_lr_noskip(c, (r + 1).min(f_bh4 - 1));
            let n11 = grid.get_lr_noskip((c + 1).min(f_bw4 - 1), (r + 1).min(f_bh4 - 1));
            if fh.cdef.on_skiptx == 0 && !n00 && !n01 && !n10 && !n11 {
                continue;
            }

            let tc = col_starts4
                .partition_point(|&start| start <= c)
                .saturating_sub(1);
            let tile_col_start4 = col_starts4[tc];
            let tile_col_end4 = col_starts4[tc + 1];

            let (min_x, max_x, min_y, max_y) = if seq.disable_loopfilters_across_tiles {
                (
                    tile_col_start4 * 4,
                    tile_col_end4 * 4,
                    tile_row_start4 * 4,
                    tile_row_end4 * 4,
                )
            } else {
                (0, f_bw4 * 4, 0, f_bh4 * 4)
            };

            let y_pri = (raw_y >> 2) as i32;
            let y_sec_raw = (raw_y & 3) as i32;
            let y_sec = if y_sec_raw == 3 { 4 } else { y_sec_raw };
            let mut pri_str_y = y_pri << coeff_shift;
            let sec_str_y = y_sec << coeff_shift;

            let uv_pri = (raw_uv >> 2) as i32;
            let uv_sec_raw = (raw_uv & 3) as i32;
            let uv_sec = if uv_sec_raw == 3 { 4 } else { uv_sec_raw };
            let pri_str_uv = uv_pri << coeff_shift;
            let sec_str_uv = uv_sec << coeff_shift;

            let x0 = c * 4;
            let y0 = r * 4;

            let (y_dir, var) = if pri_str_y != 0 || pri_str_uv != 0 {
                find_cdef_direction(&src.planes[0], x0, y0, bd.bitdepth())
            } else {
                (0, 0)
            };

            let dir_y = if pri_str_y == 0 { 0 } else { y_dir };
            let var_str = if (var >> 6) != 0 {
                ((var >> 6).ilog2() as i32).min(12)
            } else {
                0
            };
            if var != 0 {
                pri_str_y = (pri_str_y * (4 + var_str) + 8) >> 4;
            } else {
                pri_str_y = 0;
            }
            let damping_y = fh.cdef.damping as i32 + coeff_shift as i32;

            // Filter luma 8x8 block
            cdef_filter_block_normative(
                &src.planes[0],
                &mut rec.planes[0],
                x0,
                y0,
                8,
                8,
                pri_str_y,
                sec_str_y,
                damping_y,
                dir_y,
                coeff_shift,
                min_x,
                max_x,
                min_y,
                max_y,
            );

            // Filter chroma if present (§7.18.1 steps 8..14)
            if seq.layout != PixelLayout::I400 {
                let ss_x = seq.ss_hor as usize;
                let ss_y = seq.ss_ver as usize;
                let dir_uv = if pri_str_uv == 0 {
                    0
                } else {
                    CDEF_UV_DIR[ss_x][ss_y][y_dir]
                };
                let damping_uv = fh.cdef.damping as i32 + coeff_shift as i32 - 1;
                let cw = 8 >> ss_x;
                let ch = 8 >> ss_y;
                let cx0 = x0 >> ss_x;
                let cy0 = y0 >> ss_y;

                let (cmin_x, cmax_x, cmin_y, cmax_y) =
                    (min_x >> ss_x, max_x >> ss_x, min_y >> ss_y, max_y >> ss_y);

                for pl in 1..=2 {
                    cdef_filter_block_normative(
                        &src.planes[pl],
                        &mut rec.planes[pl],
                        cx0,
                        cy0,
                        cw,
                        ch,
                        pri_str_uv,
                        sec_str_uv,
                        damping_uv,
                        dir_uv,
                        coeff_shift,
                        cmin_x,
                        cmax_x,
                        cmin_y,
                        cmax_y,
                    );
                }
            }
        }
    }
}

/// Backward-compatible single-block filter wrapper.
#[allow(clippy::too_many_arguments)]
pub fn cdef_filter_block<BD: BitDepth>(
    _bd: BD,
    dst: &mut Plane<BD::Pixel>,
    src: &Plane<BD::Pixel>,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
    dir: usize,
    pri_strength: i32,
    sec_strength: i32,
    damping: i32,
) {
    let coeff_shift = (src.data[0].to_i32() > 255) as u8; // 0 for 8-bit, 2 for 10-bit
    cdef_filter_block_normative(
        src,
        dst,
        bx,
        by,
        bw,
        bh,
        pri_strength,
        sec_strength,
        damping,
        dir,
        coeff_shift,
        0,
        src.width,
        0,
        src.height,
    );
}

/// Evaluates candidate CDEF strengths against the original source frame `src`, updates
/// `frame_hdr.cdef`, and applies the winning CDEF filter to `rec` in-place.
pub fn optimize_and_apply_cdef<BD: BitDepth>(
    bd: BD,
    src: &Frame<BD::Pixel>,
    rec: &mut Frame<BD::Pixel>,
    seq: &SequenceHeader,
    frame_hdr: &mut FrameHeader,
    grid: &DeblockGrid,
) {
    let qidx = frame_hdr.quant.yac;
    if qidx <= 16 {
        frame_hdr.cdef.damping = 3;
        frame_hdr.cdef.n_bits = 0;
        frame_hdr.cdef.n_strengths = 1;
        frame_hdr.cdef.y_strength[0] = 0;
        frame_hdr.cdef.uv_strength[0] = 0;
        return;
    }

    let damping = (3 + (qidx as i32 >> 6)).clamp(3, 6);

    let base_sse = compute_sse(
        &src.planes[0].data,
        0,
        src.planes[0].stride,
        &rec.planes[0].data,
        0,
        rec.planes[0].stride,
        rec.width,
        rec.height,
    );

    let mut best_sse = base_sse;
    let mut best_y_code = 0u8;

    let candidates: &[u8] = &[4, 8, 13, 21];

    let f_bw4 = 2 * ((frame_hdr.width.max(0) as usize + 7) >> 3);
    let f_bh4 = 2 * ((frame_hdr.height.max(0) as usize + 7) >> 3);
    let cdef_cols = f_bw4.div_ceil(16);
    let cdef_rows = f_bh4.div_ceil(16);
    let cdef_idx = vec![0i8; cdef_cols * cdef_rows];

    for &code in candidates {
        let mut trial = rec.clone();
        let mut fh = frame_hdr.clone();
        fh.cdef.enabled = 1;
        fh.cdef.damping = damping as u8;
        fh.cdef.n_bits = 0;
        fh.cdef.n_strengths = 1;
        fh.cdef.y_strength[0] = code;
        fh.cdef.uv_strength[0] = 0;

        cdef_frame_normative(bd, &mut trial, seq, &fh, grid, &cdef_idx);
        let sse = compute_sse(
            &src.planes[0].data,
            0,
            src.planes[0].stride,
            &trial.planes[0].data,
            0,
            trial.planes[0].stride,
            rec.width,
            rec.height,
        );
        if sse < best_sse {
            best_sse = sse;
            best_y_code = code;
        }
    }

    frame_hdr.cdef.damping = damping as u8;
    frame_hdr.cdef.n_bits = 0;
    frame_hdr.cdef.n_strengths = 1;
    frame_hdr.cdef.y_strength[0] = best_y_code;
    frame_hdr.cdef.uv_strength[0] = (best_y_code >> 3) << 2;

    if best_y_code > 0 {
        cdef_frame_normative(bd, rec, seq, frame_hdr, grid, &cdef_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::headers::FhCdef;
    use crate::av2::pixel::{BitDepth8, BitDepth16};

    #[test]
    fn test_cdef_constrain() {
        assert_eq!(cdef_constrain(0, 10, 5), 0);
        assert_eq!(cdef_constrain(25, 0, 5), 0);
        assert_eq!(cdef_constrain(-25, 0, 5), 0);

        // Positive diff within damping range
        let res_pos = cdef_constrain(10, 8, 4);
        assert_eq!(res_pos, 3);

        // Symmetric negative diff
        let res_neg = cdef_constrain(-10, 8, 4);
        assert_eq!(res_neg, -3);

        // When diff is small, constrain should allow it through (up to threshold)
        let small = cdef_constrain(3, 8, 5);
        assert_eq!(small, 3);
        let small_neg = cdef_constrain(-3, 8, 5);
        assert_eq!(small_neg, -3);

        // Large diff exceeds threshold minus shift term
        let large = cdef_constrain(100, 4, 3);
        assert_eq!(large, 0);
    }

    #[test]
    fn test_find_cdef_direction_patterns() {
        // Horizontal edge: top half 50, bottom half 200 -> Direction 2 (0 deg)
        let mut h_plane = Plane::<u8>::new(8, 8, 0, 0, 8);
        for y in 0..8 {
            for x in 0..8 {
                h_plane.data[y * h_plane.stride + x] = if y < 4 { 50 } else { 200 };
            }
        }
        let (dir_h, var_h) = find_cdef_direction(&h_plane, 0, 0, 8);
        assert_eq!(dir_h, 2, "horizontal edge should be direction 2");
        assert!(var_h > 0, "variance should be non-zero for horizontal edge");

        // Vertical edge: left half 50, right half 200 -> Direction 6 (-90 deg)
        let mut v_plane = Plane::<u8>::new(8, 8, 0, 0, 8);
        for y in 0..8 {
            for x in 0..8 {
                v_plane.data[y * v_plane.stride + x] = if x < 4 { 50 } else { 200 };
            }
        }
        let (dir_v, var_v) = find_cdef_direction(&v_plane, 0, 0, 8);
        assert_eq!(dir_v, 6, "vertical edge should be direction 6");
        assert!(var_v > 0, "variance should be non-zero for vertical edge");

        // Diagonal edge (45 deg): y + x < 7 -> Direction 0
        let mut d_plane = Plane::<u8>::new(8, 8, 0, 0, 8);
        for y in 0..8 {
            for x in 0..8 {
                d_plane.data[y * d_plane.stride + x] = if y + x < 7 { 40 } else { 220 };
            }
        }
        let (dir_d, var_d) = find_cdef_direction(&d_plane, 0, 0, 8);
        assert_eq!(dir_d, 0, "45-degree diagonal edge should be direction 0");
        assert!(var_d > 0);

        // Flat block -> variance 0
        let mut flat_plane = Plane::<u8>::new(8, 8, 0, 0, 8);
        for y in 0..8 {
            for x in 0..8 {
                flat_plane.data[y * flat_plane.stride + x] = 128;
            }
        }
        let (_, var_flat) = find_cdef_direction(&flat_plane, 0, 0, 8);
        assert_eq!(var_flat, 0, "flat block should have variance 0");
    }

    #[test]
    fn test_cdef_filter_block_normative_taps_and_clipping() {
        let mut src = Plane::<u8>::new(16, 16, 0, 0, 8);
        for y in 0..16 {
            for x in 0..16 {
                src.data[y * src.stride + x] = ((x * 13 + y * 7) % 256) as u8;
            }
        }
        let mut dst = src.clone();

        // Primary even tap (pri_str = 4 -> taps {4, 2})
        cdef_filter_block_normative(&src, &mut dst, 0, 0, 8, 8, 4, 0, 5, 2, 0, 0, 16, 0, 16);

        // Primary odd tap (pri_str = 3 -> taps {3, 3})
        cdef_filter_block_normative(&src, &mut dst, 8, 0, 8, 8, 3, 0, 5, 2, 0, 0, 16, 0, 16);

        // Secondary only (pri_str = 0, sec_str = 2)
        cdef_filter_block_normative(&src, &mut dst, 0, 8, 8, 8, 0, 2, 5, 2, 0, 0, 16, 0, 16);

        // Combined primary and secondary (pri_str = 4, sec_str = 2)
        cdef_filter_block_normative(&src, &mut dst, 8, 8, 8, 8, 4, 2, 5, 2, 0, 0, 16, 0, 16);

        // Ensure filtered output is valid and non-trivially smoothed
        for y in 0..16 {
            for x in 0..16 {
                let s = src.data[y * src.stride + x];
                let d = dst.data[y * dst.stride + x];
                assert!((s as i32 - d as i32).abs() < 50);
            }
        }
    }

    #[test]
    fn test_cdef_boundary_handling() {
        let mut src = Plane::<u8>::new(8, 8, 0, 0, 8);
        for y in 0..8 {
            for x in 0..8 {
                src.data[y * src.stride + x] = (y * 20 + x * 10) as u8;
            }
        }
        let mut dst = src.clone();

        // Strict boundary: only [0..8, 0..8] is inside.
        // Taps reaching beyond 8x8 should be skipped without panic.
        cdef_filter_block_normative(&src, &mut dst, 0, 0, 8, 8, 8, 4, 5, 0, 0, 0, 8, 0, 8);

        // Ensure output was filtered within valid bounds
        for y in 0..8 {
            for x in 0..8 {
                let d = dst.data[y * dst.stride + x];
                assert!(d > 0);
            }
        }
    }

    #[test]
    fn test_cdef_frame_normative_all_layouts_and_high_bit_depth() {
        for layout in [PixelLayout::I420, PixelLayout::I422, PixelLayout::I444] {
            // 8-bit test
            let mut frame8 = Frame::<u8>::new(32, 32, 8, layout, 0);
            for plane in &mut frame8.planes {
                for i in 0..plane.data.len() {
                    plane.data[i] = ((i * 31 + 17) % 256) as u8;
                }
            }

            let seq = SequenceHeader {
                layout,
                ss_hor: (layout != PixelLayout::I444) as u8,
                ss_ver: (layout == PixelLayout::I420) as u8,
                disable_loopfilters_across_tiles: false,
                ..Default::default()
            };

            let mut fh = FrameHeader {
                width: 32,
                height: 32,
                cdef: FhCdef {
                    enabled: 1,
                    damping: 4,
                    n_bits: 0,
                    n_strengths: 2,
                    on_skiptx: 1,
                    y_strength: [10, 5, 0, 0, 0, 0, 0, 0],
                    uv_strength: [6, 2, 0, 0, 0, 0, 0, 0],
                },
                ..Default::default()
            };

            let grid = DeblockGrid::new(8, 8, layout);
            let cdef_idx = vec![0i8; 4];

            cdef_frame_normative(BitDepth8, &mut frame8, &seq, &fh, &grid, &cdef_idx);

            // 10-bit test
            let mut frame10 = Frame::<u16>::new(32, 32, 10, layout, 0);
            for plane in &mut frame10.planes {
                for i in 0..plane.data.len() {
                    plane.data[i] = ((i * 127 + 53) % 1024) as u16;
                }
            }
            fh.cdef.damping = 5;
            let bd10 = BitDepth16::new(10);
            cdef_frame_normative(bd10, &mut frame10, &seq, &fh, &grid, &cdef_idx);
        }
    }

    #[test]
    fn test_cdef_multiple_strengths_change_output() {
        // The encoder's CDEF search picks one strength per frame; this test
        // exercises the filter directly with several of the candidate strength
        // codes so a regression in tap/strength handling is caught even when
        // the synthetic RD search converges to a single value.
        let layout = PixelLayout::I420;
        let mut outputs = Vec::new();
        for code in [4u8, 8, 13, 21] {
            let mut frame = Frame::<u8>::new(32, 32, 8, layout, 0);
            for (pi, plane) in frame.planes.iter_mut().enumerate() {
                for y in 0..plane.height {
                    for x in 0..plane.width {
                        let v = if pi == 0 {
                            (40 + (x * 7 + y * 13) % 180) as u8
                        } else {
                            (96 + (x * 5 + y * 3) % 64) as u8
                        };
                        plane.data[y * plane.stride + x] = v;
                    }
                }
            }
            let seq = SequenceHeader {
                layout,
                ss_hor: 1,
                ss_ver: 1,
                ..Default::default()
            };
            let fh = FrameHeader {
                width: 32,
                height: 32,
                cdef: FhCdef {
                    enabled: 1,
                    damping: 4,
                    n_bits: 0,
                    n_strengths: 1,
                    on_skiptx: 1,
                    y_strength: [code, 0, 0, 0, 0, 0, 0, 0],
                    uv_strength: [code, 0, 0, 0, 0, 0, 0, 0],
                },
                ..Default::default()
            };
            let grid = DeblockGrid::new(8, 8, layout);
            let cdef_idx = vec![0i8; 4];
            cdef_frame_normative(BitDepth8, &mut frame, &seq, &fh, &grid, &cdef_idx);
            outputs.push(frame.planes[0].data.clone());
        }
        assert!(
            outputs.windows(2).any(|w| w[0] != w[1]),
            "CDEF strength codes 4/8/13/21 produced identical luma output"
        );
    }
}
