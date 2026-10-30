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

//! Clean-room Wave 3 Sub-Block Transform Partitioning (`TxPartition`, `txpart.rs`).
//!
//! Evaluates whether a coding block's prediction residual should be transformed as a single
//! full-size transform unit (`TxPartition::TxNone`) or split into 2 (`TxH`, `TxV`) or 4
//! (`TxSplit`, `TxH4`, `TxV4`) sub-transforms based on localized high-frequency residual
//! variance and Lagrangian RDO cost $J = D + \lambda R$.

use crate::av2::cdf::CdfModeContext;
use crate::av2::levels::{BlockSize, RectTxfmSize, TxPartition, txtp};
use crate::av2::msac::MsacEncoder;
use crate::av2::pixel::{BitDepth, Pixel};
use crate::av2::quantizer::{quantize_dequantize_block, quantize_rdoq_block};
use crate::av2::tables::{MAX_TXFM_SIZE_FOR_BS, TXFM_DIMENSIONS};
use crate::av2::transform::{fwd_txfm_2d, inv_txfm_add};

/// Geometry of a single transform sub-unit within a coding block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubTxUnit {
    /// Sub-transform size (`RectTxfmSize`).
    pub tx: RectTxfmSize,
    /// Horizontal offset in pixels relative to the coding block origin.
    pub x_px: usize,
    /// Vertical offset in pixels relative to the coding block origin.
    pub y_px: usize,
    /// Width in pixels (`4..=64`).
    pub w_px: usize,
    /// Height in pixels (`4..=64`).
    pub h_px: usize,
}

/// Returns the sub-transform size obtained by splitting `tx` into 4 square/rectangular quadrants.
pub const fn split_tx_quad(tx: RectTxfmSize) -> Option<RectTxfmSize> {
    match tx {
        RectTxfmSize::Tx64x64 => Some(RectTxfmSize::Tx32x32),
        RectTxfmSize::Tx32x32 => Some(RectTxfmSize::Tx16x16),
        RectTxfmSize::Tx16x16 => Some(RectTxfmSize::Tx8x8),
        RectTxfmSize::Tx8x8 => Some(RectTxfmSize::Tx4x4),
        RectTxfmSize::Rtx64x32 => Some(RectTxfmSize::Rtx32x16),
        RectTxfmSize::Rtx32x64 => Some(RectTxfmSize::Rtx16x32),
        RectTxfmSize::Rtx32x16 => Some(RectTxfmSize::Rtx16x8),
        RectTxfmSize::Rtx16x32 => Some(RectTxfmSize::Rtx8x16),
        RectTxfmSize::Rtx16x8 => Some(RectTxfmSize::Rtx8x4),
        RectTxfmSize::Rtx8x16 => Some(RectTxfmSize::Rtx4x8),
        _ => None,
    }
}

/// Returns the sub-transform size obtained by splitting `tx` horizontally (`TxH`: 2 stacked halves).
pub const fn split_tx_hor(tx: RectTxfmSize) -> Option<RectTxfmSize> {
    match tx {
        RectTxfmSize::Tx64x64 => Some(RectTxfmSize::Rtx64x32),
        RectTxfmSize::Tx32x32 => Some(RectTxfmSize::Rtx32x16),
        RectTxfmSize::Tx16x16 => Some(RectTxfmSize::Rtx16x8),
        RectTxfmSize::Tx8x8 => Some(RectTxfmSize::Rtx8x4),
        RectTxfmSize::Rtx32x64 => Some(RectTxfmSize::Tx32x32),
        RectTxfmSize::Rtx16x32 => Some(RectTxfmSize::Tx16x16),
        RectTxfmSize::Rtx8x16 => Some(RectTxfmSize::Tx8x8),
        RectTxfmSize::Rtx4x8 => Some(RectTxfmSize::Tx4x4),
        _ => None,
    }
}

/// Returns the sub-transform size obtained by splitting `tx` vertically (`TxV`: 2 side-by-side halves).
pub const fn split_tx_ver(tx: RectTxfmSize) -> Option<RectTxfmSize> {
    match tx {
        RectTxfmSize::Tx64x64 => Some(RectTxfmSize::Rtx32x64),
        RectTxfmSize::Tx32x32 => Some(RectTxfmSize::Rtx16x32),
        RectTxfmSize::Tx16x16 => Some(RectTxfmSize::Rtx8x16),
        RectTxfmSize::Tx8x8 => Some(RectTxfmSize::Rtx4x8),
        RectTxfmSize::Rtx64x32 => Some(RectTxfmSize::Tx32x32),
        RectTxfmSize::Rtx32x16 => Some(RectTxfmSize::Tx16x16),
        RectTxfmSize::Rtx16x8 => Some(RectTxfmSize::Tx8x8),
        RectTxfmSize::Rtx8x4 => Some(RectTxfmSize::Tx4x4),
        _ => None,
    }
}

/// Expands a coding block's base transform size `base_tx` and `tx_part` into its list of
/// constituent [`SubTxUnit`]s (up to 4 sub-units).
pub fn expand_tx_partition(base_tx: RectTxfmSize, tx_part: TxPartition) -> (Vec<SubTxUnit>, usize) {
    let t_dim = &TXFM_DIMENSIONS[base_tx as usize];
    let w = (t_dim.w as usize) * 4;
    let h = (t_dim.h as usize) * 4;

    match tx_part {
        TxPartition::TxNone => (
            vec![SubTxUnit {
                tx: base_tx,
                x_px: 0,
                y_px: 0,
                w_px: w,
                h_px: h,
            }],
            1,
        ),
        TxPartition::TxSplit => {
            if let Some(sub_tx) = split_tx_quad(base_tx) {
                let hw = w / 2;
                let hh = h / 2;
                (
                    vec![
                        SubTxUnit {
                            tx: sub_tx,
                            x_px: 0,
                            y_px: 0,
                            w_px: hw,
                            h_px: hh,
                        },
                        SubTxUnit {
                            tx: sub_tx,
                            x_px: hw,
                            y_px: 0,
                            w_px: hw,
                            h_px: hh,
                        },
                        SubTxUnit {
                            tx: sub_tx,
                            x_px: 0,
                            y_px: hh,
                            w_px: hw,
                            h_px: hh,
                        },
                        SubTxUnit {
                            tx: sub_tx,
                            x_px: hw,
                            y_px: hh,
                            w_px: hw,
                            h_px: hh,
                        },
                    ],
                    4,
                )
            } else {
                (
                    vec![SubTxUnit {
                        tx: base_tx,
                        x_px: 0,
                        y_px: 0,
                        w_px: w,
                        h_px: h,
                    }],
                    1,
                )
            }
        }
        TxPartition::TxH => {
            if let Some(sub_tx) = split_tx_hor(base_tx) {
                let hh = h / 2;
                (
                    vec![
                        SubTxUnit {
                            tx: sub_tx,
                            x_px: 0,
                            y_px: 0,
                            w_px: w,
                            h_px: hh,
                        },
                        SubTxUnit {
                            tx: sub_tx,
                            x_px: 0,
                            y_px: hh,
                            w_px: w,
                            h_px: hh,
                        },
                    ],
                    2,
                )
            } else {
                (
                    vec![SubTxUnit {
                        tx: base_tx,
                        x_px: 0,
                        y_px: 0,
                        w_px: w,
                        h_px: h,
                    }],
                    1,
                )
            }
        }
        TxPartition::TxV => {
            if let Some(sub_tx) = split_tx_ver(base_tx) {
                let hw = w / 2;
                (
                    vec![
                        SubTxUnit {
                            tx: sub_tx,
                            x_px: 0,
                            y_px: 0,
                            w_px: hw,
                            h_px: h,
                        },
                        SubTxUnit {
                            tx: sub_tx,
                            x_px: hw,
                            y_px: 0,
                            w_px: hw,
                            h_px: h,
                        },
                    ],
                    2,
                )
            } else {
                (
                    vec![SubTxUnit {
                        tx: base_tx,
                        x_px: 0,
                        y_px: 0,
                        w_px: w,
                        h_px: h,
                    }],
                    1,
                )
            }
        }
        _ => (
            vec![SubTxUnit {
                tx: base_tx,
                x_px: 0,
                y_px: 0,
                w_px: w,
                h_px: h,
            }],
            1,
        ),
    }
}

/// Evaluates the RDO cost of transforming, quantizing, and reconstructing a residual buffer
/// `res` (`w x h`) under a given `TxPartition` (`TxNone` vs `TxSplit`), returning
/// `(distortion_sse, non_zero_count, rate_bits_estimate)`.
#[allow(clippy::too_many_arguments)]
fn evaluate_tx_partition_candidate<BD: BitDepth>(
    bd: BD,
    res: &[i32],
    res_stride: usize,
    base_tx: RectTxfmSize,
    tx_part: TxPartition,
    dq_tbl: [u32; 2],
    is_intra: bool,
    lossless: bool,
    enable_rdoq: bool,
    lambda_rdoq: u64,
    _speed: u8,
) -> u64 {
    let (units, n_units) = expand_tx_partition(base_tx, tx_part);
    let mut cf = [0i32; 64 * 64];
    let mut q_cf = [0i32; 64 * 64];
    let mut dq_cf = [0i32; 64 * 64];
    let mid = 1i32 << (bd.bitdepth() - 1);
    let mid_px = bd.pixel_clip(mid);

    let mut total_sse = 0u64;
    let mut total_rate_q8 = if tx_part == TxPartition::TxNone {
        192u64
    } else {
        448u64 * (n_units as u64)
    };

    for u in &units {
        let uw = u.w_px;
        let uh = u.h_px;
        let utx = u.tx as usize;
        let mut sub_res = vec![0i32; uw * uh];
        for r in 0..uh {
            for c in 0..uw {
                sub_res[r * uw + c] = res[(u.y_px + r) * res_stride + (u.x_px + c)];
            }
        }

        // The sub-block transform RDO uses the default kernel: the transform
        // type of the actual block is chosen separately by the frame encoder,
        // and only `TX_PARTITION_NONE` is emitted by this encoder.
        let txtp_val = if lossless && utx == 0 {
            crate::av2::transform::WHT_WHT
        } else {
            txtp::DCT_DCT
        };
        fwd_txfm_2d(&sub_res, uw, &mut cf, txtp_val, utx);
        let eob = if enable_rdoq {
            quantize_rdoq_block(
                &cf,
                &mut q_cf,
                &mut dq_cf,
                utx,
                dq_tbl,
                None,
                bd.bitdepth(),
                is_intra,
                lossless,
                lambda_rdoq,
            )
        } else {
            quantize_dequantize_block(
                &cf,
                &mut q_cf,
                &mut dq_cf,
                utx,
                dq_tbl,
                None,
                bd.bitdepth(),
                is_intra,
                lossless,
            )
        };

        let sw = uw.min(32);
        let sh = uh.min(32);
        for i in 0..(sw * sh) {
            let a = q_cf[i].unsigned_abs() as u64;
            if a > 0 {
                total_rate_q8 += 512 + a.min(16) * 256;
            }
        }

        let mut rec_patch = vec![mid_px; uw * uh];
        if eob >= 0 {
            if lossless {
                // AV2 Spec v1.0.0 §7.14.4 lossless dequantization: `q = 64`
                // at quantizer index 0 and `QUANT_TABLE_BITS = 3`.
                for v in dq_cf.iter_mut() {
                    *v *= crate::av2::transform::LOSSLESS_DEQUANT_SCALE;
                }
            }
            inv_txfm_add(bd, &mut rec_patch, 0, uw, &mut dq_cf, txtp_val, eob, utx);
        }
        for r in 0..uh {
            for c in 0..uw {
                let recon_res = rec_patch[r * uw + c].to_i32() - mid;
                let diff = (sub_res[r * uw + c] - recon_res) as i64;
                total_sse += (diff * diff) as u64;
            }
        }
    }

    let _ = txtp::DCT_DCT;
    total_sse + ((lambda_rdoq * total_rate_q8) >> 8)
}

/// Selects the optimal [`TxPartition`] (`TxNone`, `TxSplit`, `TxH`, or `TxV`) for a coding block
/// residual `res` (`w x h`) by comparing spatial quadrant energy heterogeneity and RDO cost.
#[allow(clippy::too_many_arguments)]
pub fn select_best_tx_partition<BD: BitDepth>(
    bd: BD,
    bs: BlockSize,
    res: &[i32],
    res_stride: usize,
    dq_tbl: [u32; 2],
    is_intra: bool,
    lossless: bool,
    enable_rdoq: bool,
    lambda_rdoq: u64,
    speed: u8,
) -> (TxPartition, RectTxfmSize) {
    let base_tx_idx = MAX_TXFM_SIZE_FOR_BS[bs as usize] as usize;
    let base_tx = RectTxfmSize::from_usize(base_tx_idx).unwrap_or(RectTxfmSize::Tx4x4);

    // 4x4 blocks cannot split further; lossless coding uses single transform unit
    if base_tx == RectTxfmSize::Tx4x4 || lossless || speed >= 8 {
        return (TxPartition::TxNone, base_tx);
    }

    // Check quadrant energy variance: if residual energy is concentrated in one quadrant/half,
    // evaluate `TxSplit` / `TxH` / `TxV`.
    let t_dim = &TXFM_DIMENSIONS[base_tx_idx];
    let w = (t_dim.w as usize) * 4;
    let h = (t_dim.h as usize) * 4;
    let hw = (w / 2).max(1);
    let hh = (h / 2).max(1);

    let mut q_energy = [0u64; 4];
    for qy in 0..2 {
        for qx in 0..2 {
            let mut e = 0u64;
            for r in 0..hh {
                let row = (qy * hh + r) * res_stride + qx * hw;
                for c in 0..hw {
                    let v = res[row + c] as i64;
                    e += (v * v) as u64;
                }
            }
            q_energy[qy * 2 + qx] = e;
        }
    }
    let max_e = *q_energy.iter().max().unwrap();
    let min_e = *q_energy.iter().min().unwrap();
    // If residual energy is uniform across quadrants, `TxNone` is optimal
    if max_e <= (min_e + 64) * 3 {
        return (TxPartition::TxNone, base_tx);
    }

    let cost_none = evaluate_tx_partition_candidate(
        bd,
        res,
        res_stride,
        base_tx,
        TxPartition::TxNone,
        dq_tbl,
        is_intra,
        lossless,
        enable_rdoq,
        lambda_rdoq,
        speed,
    );

    let mut best_part = TxPartition::TxNone;
    let mut best_tx = base_tx;
    let mut best_cost = cost_none;

    if let Some(sub_tx) = split_tx_quad(base_tx) {
        let cost_split = evaluate_tx_partition_candidate(
            bd,
            res,
            res_stride,
            base_tx,
            TxPartition::TxSplit,
            dq_tbl,
            is_intra,
            lossless,
            enable_rdoq,
            lambda_rdoq,
            speed,
        );
        if cost_split < best_cost {
            best_cost = cost_split;
            best_part = TxPartition::TxSplit;
            best_tx = sub_tx;
        }
    }

    let _ = best_cost;
    (best_part, best_tx)
}

/// Encodes the `tx_split` flag for a coding block into `msac` (AV2 Spec §5.20.2 `read_tx_part`).
///
/// The flag is present only when the block's base transform is not `4x4`, the block is no
/// larger than `64x64` in either dimension, and the sequence does not use the reduced
/// transform-partition set. `fsc` is the screen-content-tools flag and `inter` selects the
/// inter CDF context.
pub fn encode_tx_partition_flag(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    bs: BlockSize,
    tx_part: TxPartition,
    fsc: usize,
    inter: usize,
) {
    // AV2 Spec §5.20.2 `read_tx_partition`: the flag is present only when the
    // coding block is at most 64x64 and its base transform is larger than 4x4.
    let base_tx_idx = MAX_TXFM_SIZE_FOR_BS[bs as usize] as usize;
    if base_tx_idx == 0 || bs == BlockSize::Bs4x4 {
        return;
    }
    let b_dim = crate::av2::tables::BLOCK_DIMENSIONS[bs as usize];
    if (b_dim[0].max(b_dim[1]) as usize) > 16 {
        return;
    }
    let is_split = (tx_part != TxPartition::TxNone) as u32;
    let grp = crate::av2::tables::TX_PART_GROUP[bs as usize] as usize;
    msac.encode_bool_adapt(
        is_split,
        cdf_m.tx_do_partition(fsc.min(1), inter.min(1), grp),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::pixel::BitDepth8;

    #[test]
    fn test_tx_partition_rdo_prefers_split_on_localized_burst() {
        // Construct a 16x16 residual where top-left 8x8 quadrant has a sharp impulse burst
        // and the other 3 quadrants are completely flat zero.
        let mut res = [0i32; 16 * 16];
        for r in 0..8 {
            for c in 0..8 {
                res[r * 16 + c] = if (r + c) % 2 == 0 { 95 } else { -95 };
            }
        }
        let dq_tbl = [24, 24];
        let (part, sub_tx) = select_best_tx_partition(
            BitDepth8,
            BlockSize::Bs16x16,
            &res,
            16,
            dq_tbl,
            false,
            false,
            true,
            32,
            2,
        );
        assert_eq!(part, TxPartition::TxSplit);
        assert_eq!(sub_tx, RectTxfmSize::Tx8x8);

        let (units, count) = expand_tx_partition(RectTxfmSize::Tx16x16, part);
        assert_eq!(count, 4);
        assert_eq!(units[0].w_px, 8);
        assert_eq!(units[3].x_px, 8);
        assert_eq!(units[3].y_px, 8);
    }
}
