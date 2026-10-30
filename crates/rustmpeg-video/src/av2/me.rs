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

//! Clean-room AV2 Motion Estimation (`ME`), 8-tap Subpel Motion Compensation (`MC`),
//! Switchable Subpel Filter RDO (`Regular8Tap`, `Smooth8Tap`, `Sharp8Tap`), and
//! Normative AV2 Specification v1.0.0 §5.20.7.21 Joint Shell Motion Vector Residual Coding (`encode_mv_residual`).

use crate::av2::cdf::CdfMvContext;
use crate::av2::dec::msac::MsacDecoder;
use crate::av2::error::Result;
use crate::av2::intops::{iclip, ulog2};
use crate::av2::ipred::compute_sad;
use crate::av2::levels::{CompInterType, MvXY};
use crate::av2::msac::MsacEncoder;
use crate::av2::pixel::{BitDepth, Pixel};
use crate::av2::tables::MC_SUBPEL_FILTERS;

/// Computes the AV2 MV joint state (`0..=3`) for a motion vector difference `(dy, dx)`:
/// - `0`: `dy == 0 && dx == 0`
/// - `1`: `dy == 0 && dx != 0`
/// - `2`: `dy != 0 && dx == 0`
/// - `3`: `dy != 0 && dx != 0`
#[inline]
pub const fn mv_joint_type(dy: i32, dx: i32) -> usize {
    ((dy != 0) as usize) * 2 + ((dx != 0) as usize)
}

/// Clamps a motion vector coordinate to integer bounds per AV2 Spec §5.20.7.13.
#[inline]
pub fn mv_clamp_to_integer(v: i32) -> i32 {
    if v < -32767 {
        -32760
    } else if v > 32766 {
        32759
    } else {
        v
    }
}

/// Decodes the magnitude of a 2D motion vector difference `mvd` from `msac` using Joint Shell syntax.
pub fn read_mv_magnitude(
    msac: &mut MsacDecoder,
    cdf_mv: &mut CdfMvContext,
    mv_prec: usize,
) -> Result<(i16, i16)> {
    let n_syms = 9 + mv_prec;
    let h_syms = n_syms >> 1;

    let shell_set = msac.decode_bool_adapt(&mut cdf_mv.joint_shell_set)? != 0;
    let sh_class = if shell_set {
        let h_syms2 = n_syms - h_syms;
        let sym = msac.decode_symbol_adapt(cdf_mv.shell_upper_mut(mv_prec), h_syms2.min(7))?;
        let mut sc = h_syms + 1 + sym;
        if mv_prec + sc == 21 {
            let tip_bit = msac.decode_bool_adapt(&mut cdf_mv.joint_shell_last_two_classes)?;
            sc += tip_bit as usize;
        }
        sc
    } else {
        msac.decode_symbol_adapt(cdf_mv.shell_lower_mut(mv_prec), h_syms)?
    };

    let mut sh_index = if sh_class < 2 {
        msac.decode_bool_adapt(&mut cdf_mv.shell_offset_low_class[sh_class])? as usize
    } else if sh_class == 2 {
        let mut idx = msac.decode_bool_adapt(&mut cdf_mv.shell_offset_class2)? as usize;
        if idx != 0 {
            let b1 = msac.decode_bool_bypass();
            idx += b1 as usize;
            if idx == 2 {
                let b2 = msac.decode_bool_bypass();
                idx += b2 as usize;
            }
        }
        idx
    } else {
        let mut idx = 0usize;
        for i in 0..sh_class {
            let b = msac.decode_bool_adapt(&mut cdf_mv.shell_offset_other_class[i.min(15)])?;
            idx |= (b as usize) << i;
        }
        idx
    };

    if sh_class != 0 {
        sh_index += 1usize << sh_class;
    }
    if sh_index == 0 {
        return Ok((0, 0));
    }

    let mut pair_index = 0usize;
    if sh_index >= 2 {
        let p0 = msac.decode_bool_adapt(&mut cdf_mv.col_mv_greater[0])?;
        if p0 != 0 {
            pair_index = 1;
            if sh_index >= 4 {
                let p1 = msac.decode_bool_adapt(&mut cdf_mv.col_mv_greater[1])?;
                if p1 != 0 {
                    pair_index = 2;
                    if sh_index >= 6 {
                        let extra = msac.decode_uniform(((sh_index >> 1) - 1) as u32) as usize;
                        pair_index += extra;
                    }
                }
            }
        }
    }

    let sh = 6 - mv_prec;
    if pair_index * 2 == sh_index {
        let v = ((sh_index >> 1) << sh) as i16;
        Ok((v, v))
    } else {
        let b = msac.decode_bool_adapt(&mut cdf_mv.col_mv_index[sh_class.min(3)])?;
        if b != 0 {
            let y = (pair_index << sh) as i16;
            let x = ((sh_index - pair_index) << sh) as i16;
            Ok((x, y))
        } else {
            let x = (pair_index << sh) as i16;
            let y = ((sh_index - pair_index) << sh) as i16;
            Ok((x, y))
        }
    }
}

/// Decodes a 2D motion vector difference `mvd` from `msac` using the normative
/// AV2 Specification v1.0.0 §5.20.7.20 / §5.20.7.13 Joint Shell MVD syntax.
pub fn read_mv_residual(
    msac: &mut MsacDecoder,
    cdf_mv: &mut CdfMvContext,
    mv_prec: usize,
    mvd_sign_derive: bool,
) -> Result<MvXY> {
    let (mut mv_x, mut mv_y) = read_mv_magnitude(msac, cdf_mv, mv_prec)?;
    let sh = 6 - mv_prec;
    let mut nnzc = (mv_x != 0) as usize + (mv_y != 0) as usize;
    let sum_mvd = ((mv_x.unsigned_abs() as usize) + (mv_y.unsigned_abs() as usize)) >> sh;
    if !mvd_sign_derive || mv_prec >= 5 {
        nnzc = 3;
    }

    let mut nnzc2 = 0usize;
    if mv_y != 0 {
        nnzc2 += 1;
        let s = if nnzc2 == nnzc {
            (sum_mvd & 1) != 0
        } else {
            msac.decode_bool_bypass() != 0
        };
        if s {
            mv_y = -mv_y;
        }
    }
    if mv_x != 0 {
        nnzc2 += 1;
        let s = if nnzc2 == nnzc {
            (sum_mvd & 1) != 0
        } else {
            msac.decode_bool_bypass() != 0
        };
        if s {
            mv_x = -mv_x;
        }
    }

    Ok(MvXY { x: mv_x, y: mv_y })
}

/// Encodes the magnitude of a 2D motion vector difference `mv - ref_mv` into `msac`.
/// Returns `(dx, dy)` in subpel units.
pub fn encode_mv_magnitude(
    msac: &mut MsacEncoder,
    cdf_mv: &mut CdfMvContext,
    mv: MvXY,
    ref_mv: MvXY,
    mv_prec: usize,
) -> (i32, i32) {
    let mvd = MvXY {
        x: mv.x - ref_mv.x,
        y: mv.y - ref_mv.y,
    };
    let shift = 6 - mv_prec;
    let dy = (mvd.y as i32) >> shift;
    let dx = (mvd.x as i32) >> shift;
    let abs_y = dy.unsigned_abs() as usize;
    let abs_x = dx.unsigned_abs() as usize;
    let shell_index = abs_y + abs_x;

    let (shell_class, shell_offset) = if shell_index == 0 {
        (0usize, 0usize)
    } else if shell_index == 1 {
        (0usize, 1usize)
    } else {
        let sc = (ulog2(shell_index as u32).max(0) as usize).min(15);
        (sc, shell_index - (1usize << sc))
    };

    let n_syms = 9 + mv_prec;
    let h_syms = n_syms >> 1;
    let shell_set = shell_class > h_syms;

    if !shell_set {
        msac.encode_bool_adapt(0, &mut cdf_mv.joint_shell_set);
        msac.encode_symbol_adapt(shell_class, cdf_mv.shell_lower_mut(mv_prec), h_syms);
    } else {
        msac.encode_bool_adapt(1, &mut cdf_mv.joint_shell_set);
        let h_syms2 = n_syms - h_syms;
        let rel_class = shell_class - (h_syms + 1);
        let sym = rel_class.min(7);
        msac.encode_symbol_adapt(sym, cdf_mv.shell_upper_mut(mv_prec), h_syms2.min(7));
        if mv_prec + shell_class == 21 {
            let tip = (rel_class > 7) as u32;
            msac.encode_bool_adapt(tip, &mut cdf_mv.joint_shell_last_two_classes);
        }
    }

    if shell_class < 2 {
        msac.encode_bool_adapt(
            (shell_offset & 1) as u32,
            &mut cdf_mv.shell_offset_low_class[shell_class],
        );
    } else if shell_class == 2 {
        let off = shell_offset.min(3);
        let b0 = (off > 0) as u32;
        msac.encode_bool_adapt(b0, &mut cdf_mv.shell_offset_class2);
        if off > 0 {
            let b1 = (off > 1) as u32;
            msac.encode_bool_bypass(b1);
            if off > 1 {
                let b2 = (off > 2) as u32;
                msac.encode_bool_bypass(b2);
            }
        }
    } else {
        for i in 0..shell_class {
            let bit = ((shell_offset >> i) & 1) as u32;
            msac.encode_bool_adapt(bit, &mut cdf_mv.shell_offset_other_class[i.min(15)]);
        }
    }

    if shell_index > 0 {
        let max_pair_idx = shell_index >> 1;
        let col = abs_x.min(abs_y);
        if shell_index >= 2 {
            let p0 = (col > 0) as u32;
            msac.encode_bool_adapt(p0, &mut cdf_mv.col_mv_greater[0]);
            if p0 != 0 && shell_index >= 4 {
                let p1 = (col > 1) as u32;
                msac.encode_bool_adapt(p1, &mut cdf_mv.col_mv_greater[1]);
                if p1 != 0 && shell_index >= 6 {
                    msac.encode_uniform(((shell_index >> 1) - 1) as u32, (col - 2) as u32);
                }
            }
        }

        let skip_col_bit = (col == max_pair_idx) && ((shell_index & 1) == 0);
        if !skip_col_bit {
            let col_mv_idx = (abs_y == col) as u32;
            msac.encode_bool_adapt(col_mv_idx, &mut cdf_mv.col_mv_index[shell_class.min(3)]);
        }
    }

    (dx, dy)
}

/// Encodes a 2D motion vector difference `mv - ref_mv` into `msac` using the normative
/// AV2 Specification v1.0.0 §5.20.7.20 / §5.20.7.13 Joint Shell MVD syntax.
pub fn encode_mv_residual(
    msac: &mut MsacEncoder,
    cdf_mv: &mut CdfMvContext,
    mv: MvXY,
    ref_mv: MvXY,
    mv_prec: usize,
    mvd_sign_derive: bool,
) {
    let (dx, dy) = encode_mv_magnitude(msac, cdf_mv, mv, ref_mv, mv_prec);
    let shift = 6 - mv_prec;
    let mut nnzc = (dx != 0) as usize + (dy != 0) as usize;
    let _sum_mvd = ((dx.unsigned_abs() as usize) + (dy.unsigned_abs() as usize)) >> shift;
    if !mvd_sign_derive || mv_prec >= 5 {
        nnzc = 3;
    }

    let mut nnzc2 = 0usize;
    if dy != 0 {
        nnzc2 += 1;
        if nnzc2 != nnzc {
            msac.encode_bool_bypass((dy < 0) as u32);
        }
    }
    if dx != 0 {
        nnzc2 += 1;
        if nnzc2 != nnzc {
            msac.encode_bool_bypass((dx < 0) as u32);
        }
    }
}

/// Estimates the bit cost (in `Q8`, i.e., `1/256` bit units) of coding `mv - ref_mv`.
#[inline]
pub fn estimate_mv_bits_q8(mv: MvXY, ref_mv: MvXY) -> u64 {
    let dy = (mv.y as i32 - ref_mv.y as i32).unsigned_abs();
    let dx = (mv.x as i32 - ref_mv.x as i32).unsigned_abs();
    let comp_cost = |d: u32| -> u64 {
        if d == 0 {
            192
        } else {
            let lg = (32 - d.leading_zeros()) as u64;
            384 + lg * 384
        }
    };
    comp_cost(dy) + comp_cost(dx)
}

/// Evaluates an integer motion vector `(mv_x_px, mv_y_px)` in full-pel coordinates by computing
/// the SAD between `src` at `(bx_px, by_px)` and `ref_plane` at `(bx_px + mv_x_px, by_px + mv_y_px)`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_fullpel_sad<P: Pixel>(
    src: &[P],
    src_stride: usize,
    ref_data: &[P],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    bx_px: i32,
    by_px: i32,
    bw: usize,
    bh: usize,
    mv_x_px: i32,
    mv_y_px: i32,
) -> u64 {
    let rx = (bx_px + mv_x_px).clamp(0, (ref_w - bw as i32).max(0)) as usize;
    let ry = (by_px + mv_y_px).clamp(0, (ref_h - bh as i32).max(0)) as usize;
    let src_off = (by_px as usize) * src_stride + (bx_px as usize);
    let ref_off = ry * ref_stride + rx;
    compute_sad(
        src, src_off, src_stride, ref_data, ref_off, ref_stride, bw, bh,
    )
}

/// Motion-compensates one `bw x bh` block from `ref_data` using a `1/16`-pel motion vector
/// (`mv_*_q16`) and filter kernel `filter` (`0` = `Regular8Tap`, `1` = `Smooth8Tap`,
/// `2` = `Sharp8Tap`), writing into `dst` at `dst_off` with stride `dst_stride`.
///
/// The `1/16`-pel vector is converted to the `1/8`-pel `MvXY` representation used by
/// [`interpolate_subpel_block`] by halving it.
#[allow(clippy::too_many_arguments)]
pub fn mc_predict_block<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    dst_stride: usize,
    ref_data: &[BD::Pixel],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    bx_px: i32,
    by_px: i32,
    bw: usize,
    bh: usize,
    mv_y_q16: i32,
    mv_x_q16: i32,
    filter: usize,
) {
    let mv = MvXY {
        y: (mv_y_q16 / 2) as i16,
        x: (mv_x_q16 / 2) as i16,
    };
    interpolate_subpel_block(
        bd,
        ref_data,
        ref_stride,
        ref_w.max(0) as usize,
        ref_h.max(0) as usize,
        bx_px,
        by_px,
        bw,
        bh,
        mv,
        filter,
        dst,
        dst_off,
        dst_stride,
    );
}

/// Applies 8-tap separable sub-pixel interpolation at quarter-pel (`1/8`-pel `MvXY` units) offset
/// using filter kernel `filter_type` (`0` = `Regular8Tap`, `1` = `Smooth8Tap`, `2` = `Sharp8Tap`).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_subpel_block<BD: BitDepth>(
    bd: BD,
    ref_data: &[BD::Pixel],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    bx_px: i32,
    by_px: i32,
    bw: usize,
    bh: usize,
    mv: MvXY,
    filter_type: usize,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    dst_stride: usize,
) {
    let int_x = bx_px + ((mv.x as i32) >> 3);
    let int_y = by_px + ((mv.y as i32) >> 3);
    let sub_x = ((mv.x as i32) & 7) << 1; // 0..15 in 1/16-pel table index
    let sub_y = ((mv.y as i32) & 7) << 1;

    let max_x = (ref_w as i32 - 1).max(0);
    let max_y = (ref_h as i32 - 1).max(0);

    if sub_x == 0 && sub_y == 0 {
        let ix = int_x as usize;
        let iy = int_y as usize;
        if int_x >= 0 && ix + bw <= ref_w && int_y >= 0 && iy + bh <= ref_h {
            for r in 0..bh {
                let dst_row = dst_off + r * dst_stride;
                let src_row = (iy + r) * ref_stride + ix;
                dst[dst_row..dst_row + bw].copy_from_slice(&ref_data[src_row..src_row + bw]);
            }
        } else if int_x >= 0 && ix + bw <= ref_w {
            for r in 0..bh {
                let y = (int_y + r as i32).clamp(0, max_y) as usize;
                let dst_row = dst_off + r * dst_stride;
                let src_row = y * ref_stride + ix;
                dst[dst_row..dst_row + bw].copy_from_slice(&ref_data[src_row..src_row + bw]);
            }
        } else {
            for r in 0..bh {
                let y = (int_y + r as i32).clamp(0, max_y) as usize;
                let dst_row = dst_off + r * dst_stride;
                let row_base = y * ref_stride;
                for c in 0..bw {
                    let x = (int_x + c as i32).clamp(0, max_x) as usize;
                    dst[dst_row + c] = ref_data[row_base + x];
                }
            }
        }
        return;
    }

    let flt = &MC_SUBPEL_FILTERS[filter_type.min(2)];
    let fx = if sub_x > 0 {
        flt[(sub_x - 1) as usize]
    } else {
        [0, 0, 0, 64, 0, 0, 0, 0]
    };
    let fy = if sub_y > 0 {
        flt[(sub_y - 1) as usize]
    } else {
        [0, 0, 0, 64, 0, 0, 0, 0]
    };

    // Horizontal pass with 7 extra rows (3 above, 4 below) for vertical 8-tap convolution.
    // The normative filters sum to 128 (`FILTER_BITS = 7`); `MC_SUBPEL_FILTERS`
    // stores them halved (sum 64), so the AV2 `InterRound0 = 3` horizontal
    // rounding `Round2(s, 3)` is `(2*acc + 4) >> 3 == (acc + 2) >> 2` here.
    let ext_h = bh + 7;
    let mut tmp = [0i32; 128 * 135];
    for r in 0..ext_h {
        let y = (int_y + (r as i32) - 3).clamp(0, max_y) as usize;
        let row_base = y * ref_stride;
        for c in 0..bw {
            let mut acc = 0i32;
            for k in 0..8 {
                let x = (int_x + (c as i32) + (k as i32) - 3).clamp(0, max_x) as usize;
                acc += ref_data[row_base + x].to_i32() * (fx[k] as i32);
            }
            tmp[r * bw + c] = (acc + 2) >> 2;
        }
    }

    // Vertical pass. `InterRound1 = 11` for a single reference and the filters
    // are halved, so `Round2(2*acc, 11) == (acc + 512) >> 10`.
    for r in 0..bh {
        let dst_row = dst_off + r * dst_stride;
        for c in 0..bw {
            let mut acc = 0i32;
            for k in 0..8 {
                acc += tmp[(r + k) * bw + c] * (fy[k] as i32);
            }
            dst[dst_row + c] = bd.iclip_pixel((acc + 512) >> 10);
        }
    }
}

/// Runs Diamond + Hexagonal full-pel motion estimation followed by quarter-pel refinement and
/// 3-way subpel filter RDO (`Regular8Tap`, `Smooth8Tap`, `Sharp8Tap`).
///
/// Returns `(best_mv, best_filter_idx, best_rd_cost)`.
#[allow(clippy::too_many_arguments)]
pub fn search_motion_vector<BD: BitDepth>(
    bd: BD,
    src_data: &[BD::Pixel],
    src_stride: usize,
    ref_data: &[BD::Pixel],
    ref_stride: usize,
    frame_w: usize,
    frame_h: usize,
    bx_px: usize,
    by_px: usize,
    bw: usize,
    bh: usize,
    mvp: MvXY,
    lambda_sad: u64,
    speed: u8,
) -> (MvXY, u8, u64) {
    let bx = bx_px as i32;
    let by = by_px as i32;
    let rw = frame_w as i32;
    let rh = frame_h as i32;

    let start_x = ((mvp.x as i32) >> 3).clamp(-bx, (rw - bw as i32 - bx).max(0));
    let start_y = ((mvp.y as i32) >> 3).clamp(-by, (rh - bh as i32 - by).max(0));

    let eval_cand = |cx: i32, cy: i32| -> u64 {
        let sad = eval_fullpel_sad(
            src_data, src_stride, ref_data, ref_stride, rw, rh, bx, by, bw, bh, cx, cy,
        );
        let cand_mv = MvXY {
            x: (cx * 8) as i16,
            y: (cy * 8) as i16,
        };
        let bits_q8 = estimate_mv_bits_q8(cand_mv, mvp);
        sad + ((lambda_sad * bits_q8) >> 8)
    };

    let mut best_x = start_x;
    let mut best_y = start_y;
    let mut best_cost = eval_cand(best_x, best_y);

    // Also check zero MV `(0, 0)`
    let zero_cost = eval_cand(0, 0);
    if zero_cost < best_cost {
        best_cost = zero_cost;
        best_x = 0;
        best_y = 0;
    }

    let max_rounds = if speed >= 7 { 4 } else { 8 };
    const DIAMOND: [(i32, i32); 8] = [
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ];

    for _ in 0..max_rounds {
        let mut improved = false;
        for &(dx, dy) in &DIAMOND {
            let nx = iclip(best_x + dx, -32, 32);
            let ny = iclip(best_y + dy, -32, 32);
            if bx + nx < 0
                || by + ny < 0
                || bx + nx + (bw as i32) > rw
                || by + ny + (bh as i32) > rh
            {
                continue;
            }
            let c = eval_cand(nx, ny);
            if c < best_cost {
                best_cost = c;
                best_x = nx;
                best_y = ny;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    // 1-pel small diamond refinement
    for &(dx, dy) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
        let nx = best_x + dx;
        let ny = best_y + dy;
        if bx + nx >= 0
            && by + ny >= 0
            && bx + nx + (bw as i32) <= rw
            && by + ny + (bh as i32) <= rh
        {
            let c = eval_cand(nx, ny);
            if c < best_cost {
                best_cost = c;
                best_x = nx;
                best_y = ny;
            }
        }
    }

    // Quarter-pel (`2` units in `1/8`-pel MvXY) refinement + 3-filter RDO
    let mut best_mv = MvXY {
        x: (best_x * 8) as i16,
        y: (best_y * 8) as i16,
    };
    let mut best_filter = 0u8;

    if speed <= 6 {
        let mut pred_buf = vec![BD::Pixel::default(); bw * bh];
        let src_off = by_px * src_stride + bx_px;

        for &(qdx, qdy) in &[
            (0i16, 0i16),
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
            (-4, 0),
            (4, 0),
            (0, -4),
            (0, 4),
        ] {
            let cand_mv = MvXY {
                x: (best_x * 8) as i16 + qdx,
                y: (best_y * 8) as i16 + qdy,
            };
            let n_filters = if qdx == 0 && qdy == 0 { 1usize } else { 3usize };
            for flt_idx in 0..n_filters {
                interpolate_subpel_block(
                    bd,
                    ref_data,
                    ref_stride,
                    frame_w,
                    frame_h,
                    bx,
                    by,
                    bw,
                    bh,
                    cand_mv,
                    flt_idx,
                    &mut pred_buf,
                    0,
                    bw,
                );
                let sad = compute_sad(src_data, src_off, src_stride, &pred_buf, 0, bw, bw, bh);
                let bits_q8 = estimate_mv_bits_q8(cand_mv, mvp) + (flt_idx as u64) * 64;
                let cost = sad + ((lambda_sad * bits_q8) >> 8);
                if cost < best_cost {
                    best_cost = cost;
                    best_mv = cand_mv;
                    best_filter = flt_idx as u8;
                }
            }
        }
    }

    (best_mv, best_filter, best_cost)
}

/// Runs the Diamond + Hexagonal full-pel search and quarter-pel refinement for block
/// `(bx_px, by_px, bw, bh)` in the full `src_data` plane against `ref_data`.
///
/// `src_off` is the full-plane offset of the source block; it must agree with
/// `by_px * src_stride + bx_px`. `search_range` is accepted for API symmetry but the search
/// window is already bounded by the frame dimensions and the internal `+/-32`-pel diamond limit.
///
/// Returns `(best_mv, best_rd_cost)` in the speed-6 configuration.
#[allow(clippy::too_many_arguments)]
pub fn estimate_motion_block<BD: BitDepth>(
    bd: BD,
    src_data: &[BD::Pixel],
    src_off: usize,
    src_stride: usize,
    ref_data: &[BD::Pixel],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    bx_px: i32,
    by_px: i32,
    bw: usize,
    bh: usize,
    mvp: MvXY,
    search_range: i32,
    lambda_sad: u64,
) -> (MvXY, u64) {
    let bx = bx_px.max(0) as usize;
    let by = by_px.max(0) as usize;
    debug_assert_eq!(src_off, by * src_stride + bx);
    let _ = search_range;
    let (mv, _filter, cost) = search_motion_vector(
        bd,
        src_data,
        src_stride,
        ref_data,
        ref_stride,
        ref_w.max(1) as usize,
        ref_h.max(1) as usize,
        bx,
        by,
        bw,
        bh,
        mvp,
        lambda_sad,
        6,
    );
    (mv, cost)
}

/// Selects the cheapest of the three switchable 8-tap subpel filters
/// (`Regular8Tap`, `Smooth8Tap`, `Sharp8Tap`) for `mv` by SAD against the source block.
///
/// Returns `(best_filter_idx, best_sad)`. Ties keep the lowest filter index.
#[allow(clippy::too_many_arguments)]
pub fn select_best_subpel_filter<BD: BitDepth>(
    bd: BD,
    src_data: &[BD::Pixel],
    src_off: usize,
    src_stride: usize,
    ref_data: &[BD::Pixel],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    bx_px: i32,
    by_px: i32,
    bw: usize,
    bh: usize,
    mv: MvXY,
) -> (u8, u64) {
    let mut pred = vec![BD::Pixel::default(); bw * bh];
    let mut best_filter = 0u8;
    let mut best_cost = u64::MAX;
    for filter in 0..=2usize {
        mc_predict_block(
            bd,
            &mut pred,
            0,
            bw,
            ref_data,
            ref_stride,
            ref_w,
            ref_h,
            bx_px,
            by_px,
            bw,
            bh,
            (mv.y as i32) * 2,
            (mv.x as i32) * 2,
            filter,
        );
        let sad = compute_sad(src_data, src_off, src_stride, &pred, 0, bw, bw, bh);
        if sad < best_cost {
            best_cost = sad;
            best_filter = filter as u8;
        }
    }
    (best_filter, best_cost)
}

/// Motion-compensates one `bw x bh` block into intermediate 16-bit / 32-bit representation `dst`
/// with `intermediate_bits` (4 for 8/10-bit, 2 for 12-bit) subpel precision (AV2 Spec §7.13.3.16).
#[allow(clippy::too_many_arguments)]
pub fn mc_predict_block_prep<BD: BitDepth>(
    bd: BD,
    dst: &mut [i32],
    ref_data: &[BD::Pixel],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    bx_px: i32,
    by_px: i32,
    bw: usize,
    bh: usize,
    mv_y_q16: i32,
    mv_x_q16: i32,
    filter_type: usize,
) {
    let mv = MvXY {
        y: (mv_y_q16 / 2) as i16,
        x: (mv_x_q16 / 2) as i16,
    };
    let int_x = bx_px + (mv.x as i32) / 8;
    let int_y = by_px + (mv.y as i32) / 8;
    let sub_x = ((mv.x as i32) & 7) << 1;
    let sub_y = ((mv.y as i32) & 7) << 1;

    let max_x = (ref_w - 1).max(0);
    let max_y = (ref_h - 1).max(0);

    let intermediate_bits = if bd.bitdepth() <= 10 { 4 } else { 2 };

    if sub_x == 0 && sub_y == 0 {
        for r in 0..bh {
            let y = (int_y + r as i32).clamp(0, max_y) as usize;
            let dst_row = r * bw;
            let row_base = y * ref_stride;
            for c in 0..bw {
                let x = (int_x + c as i32).clamp(0, max_x) as usize;
                dst[dst_row + c] = (ref_data[row_base + x].to_i32()) << intermediate_bits;
            }
        }
        return;
    }

    let flt = &MC_SUBPEL_FILTERS[filter_type.min(2)];
    let fx = if sub_x > 0 {
        flt[(sub_x - 1) as usize]
    } else {
        [0, 0, 0, 64, 0, 0, 0, 0]
    };
    let fy = if sub_y > 0 {
        flt[(sub_y - 1) as usize]
    } else {
        [0, 0, 0, 64, 0, 0, 0, 0]
    };

    let sh_h = 6 - intermediate_bits;
    let rnd_h = 1 << (sh_h - 1);

    if sub_y == 0 {
        for r in 0..bh {
            let y = (int_y + r as i32).clamp(0, max_y) as usize;
            let dst_row = r * bw;
            let row_base = y * ref_stride;
            for c in 0..bw {
                let mut acc = 0i32;
                for k in 0..8 {
                    let x = (int_x + (c as i32) + (k as i32) - 3).clamp(0, max_x) as usize;
                    acc += ref_data[row_base + x].to_i32() * (fx[k] as i32);
                }
                dst[dst_row + c] = (acc + rnd_h) >> sh_h;
            }
        }
    } else if sub_x == 0 {
        for r in 0..bh {
            let dst_row = r * bw;
            for c in 0..bw {
                let x = (int_x + c as i32).clamp(0, max_x) as usize;
                let mut acc = 0i32;
                for k in 0..8 {
                    let y = (int_y + (r as i32) + (k as i32) - 3).clamp(0, max_y) as usize;
                    acc += ref_data[y * ref_stride + x].to_i32() * (fy[k] as i32);
                }
                dst[dst_row + c] = (acc + rnd_h) >> sh_h;
            }
        }
    } else {
        let ext_h = bh + 7;
        let mut mid = [0i32; 128 * 135];
        for r in 0..ext_h {
            let y = (int_y + (r as i32) - 3).clamp(0, max_y) as usize;
            let row_base = y * ref_stride;
            for c in 0..bw {
                let mut acc = 0i32;
                for k in 0..8 {
                    let x = (int_x + (c as i32) + (k as i32) - 3).clamp(0, max_x) as usize;
                    acc += ref_data[row_base + x].to_i32() * (fx[k] as i32);
                }
                mid[r * bw + c] = (acc + rnd_h) >> sh_h;
            }
        }

        for r in 0..bh {
            let dst_row = r * bw;
            for c in 0..bw {
                let mut acc = 0i32;
                for k in 0..8 {
                    acc += mid[(r + k) * bw + c] * (fy[k] as i32);
                }
                dst[dst_row + c] = (acc + 32) >> 6;
            }
        }
    }
}

/// Blends two intermediate predictions into `dst` according to `comp_type` and optional mask.
#[allow(clippy::too_many_arguments)]
pub fn blend_compound_prep<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    dst_stride: usize,
    p0: &[i32],
    p1: &[i32],
    w: usize,
    h: usize,
    comp_type: CompInterType,
    weight: i32,
    mask: Option<&[u8]>,
) {
    let intermediate_bits = if bd.bitdepth() <= 10 { 4 } else { 2 };
    match comp_type {
        CompInterType::Average => {
            let sh = intermediate_bits + 1;
            let rnd = 1 << intermediate_bits;
            for r in 0..h {
                let d_row = dst_off + r * dst_stride;
                let s_row = r * w;
                for c in 0..w {
                    let val = (p0[s_row + c] + p1[s_row + c] + rnd) >> sh;
                    dst[d_row + c] = bd.clip_pixel(val);
                }
            }
        }
        CompInterType::DistanceWeighted => {
            let sh = intermediate_bits + 4;
            let rnd = 8 << intermediate_bits;
            let w0 = weight;
            let w1 = 16 - weight;
            for r in 0..h {
                let d_row = dst_off + r * dst_stride;
                let s_row = r * w;
                for c in 0..w {
                    let val = (w0 * p0[s_row + c] + w1 * p1[s_row + c] + rnd) >> sh;
                    dst[d_row + c] = bd.clip_pixel(val);
                }
            }
        }
        CompInterType::Wedge | CompInterType::DifferenceWeighted => {
            let sh = intermediate_bits + 6;
            let rnd = 32 << intermediate_bits;
            let mask = mask.expect("mask must be provided for Wedge and DifferenceWeighted");
            for r in 0..h {
                let d_row = dst_off + r * dst_stride;
                let s_row = r * w;
                for c in 0..w {
                    let m = mask[s_row + c] as i32;
                    let val = (m * p0[s_row + c] + (64 - m) * p1[s_row + c] + rnd) >> sh;
                    dst[d_row + c] = bd.clip_pixel(val);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mv_clamp_bounds() {
        assert_eq!(mv_clamp_to_integer(0), 0);
        assert_eq!(mv_clamp_to_integer(100), 100);
        assert_eq!(mv_clamp_to_integer(-100), -100);
        assert_eq!(mv_clamp_to_integer(-40000), -32760);
        assert_eq!(mv_clamp_to_integer(40000), 32759);
    }

    #[test]
    fn test_encode_decode_mv_residual_roundtrip() {
        let test_mvs = [
            (0, 0),
            (2, 0),
            (0, 2),
            (-2, 0),
            (0, -2),
            (2, 2),
            (-2, 2),
            (2, -2),
            (-2, -2),
            (4, 6),
            (-8, 10),
            (14, -18),
            (-32, -40),
            (128, -256),
            (-512, 1024),
        ];

        for mv_prec in [5usize, 6usize] {
            let shift = 6 - mv_prec;
            for &(rx, ry) in &test_mvs {
                let mvd = MvXY {
                    x: (rx << shift) as i16,
                    y: (ry << shift) as i16,
                };
                let mut enc_cdf = CdfMvContext::new();
                let mut msac_enc = MsacEncoder::new();
                encode_mv_residual(
                    &mut msac_enc,
                    &mut enc_cdf,
                    mvd,
                    MvXY::default(),
                    mv_prec,
                    false,
                );
                let bitstream = msac_enc.finalize();

                let mut dec_cdf = CdfMvContext::new();
                let mut msac_dec = MsacDecoder::new(&bitstream, true);
                let decoded =
                    read_mv_residual(&mut msac_dec, &mut dec_cdf, mv_prec, false).unwrap();
                assert_eq!(
                    decoded, mvd,
                    "Failed roundtrip for MVD ({rx}, {ry}) at mv_prec {mv_prec}"
                );
            }
        }
    }

    #[test]
    fn test_subpel_constant_plane_is_preserved() {
        use crate::av2::pixel::BitDepth8;
        let ref_plane = vec![100u8; 8 * 8];
        let mut dst = vec![0u8; 4 * 4];
        for filter in 0..=2usize {
            interpolate_subpel_block(
                BitDepth8,
                &ref_plane,
                8,
                8,
                8,
                2,
                2,
                4,
                4,
                MvXY { x: 1, y: 1 },
                filter,
                &mut dst,
                0,
                4,
            );
            assert!(dst.iter().all(|&p| p == 100), "filter {filter}: {dst:?}");
        }
    }

    #[test]
    fn test_subpel_rounding_matches_normative_two_pass() {
        use crate::av2::pixel::BitDepth8;
        // Horizontal phase 1 (1/8 pel) with the regular filter. The tap
        // pattern gives `acc = 30`; the normative `Round2(acc, 3)` horizontal
        // pass followed by `Round2(..., 11)` yields 1, while a single-pass
        // `(acc + 32) >> 6` would yield 0.
        let mut ref_plane = vec![0u8; 8 * 8];
        for row in 0..8 {
            ref_plane[row * 8 + 4] = 4;
            ref_plane[row * 8 + 5] = 3;
        }
        let mut dst = vec![0u8; 4 * 4];
        interpolate_subpel_block(
            BitDepth8,
            &ref_plane,
            8,
            8,
            8,
            2,
            2,
            4,
            4,
            MvXY { x: 1, y: 0 },
            0,
            &mut dst,
            0,
            4,
        );
        assert_eq!(dst[1], 1, "normative rounding expected 1, got {dst:?}");
    }
}
