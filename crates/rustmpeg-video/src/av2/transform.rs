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

//! Forward and Inverse 1D/2D Transforms for AV2 (DCT, ADST, FLIPADST, DDT, FLIPDDT, Identity, WHT).
//!
//! Implements both the forward transform (`fwd_txfm_2d`) for the encoder residual path and
//! the normative AV2 inverse transform (`inv_txfm_add`) for the in-loop reconstruction path,
//! supporting all 25 square and rectangular transform sizes (`4x4` through `64x64`) and
//! lossless `4x4` Walsh-Hadamard Transform (`WHT_WHT`).

use crate::av2::entropy::TxSet;
use crate::av2::intops::{iclip, imin};
use crate::av2::levels::txtp;
use crate::av2::levels::txtp::DCT_DCT;
use crate::av2::levels::{N_RECT_TX_SIZES, N_TX_1D_TYPES, N_TX_SIZES};
use crate::av2::pixel::{BitDepth, BitDepth8, Pixel};
use crate::av2::tables::{TX_SHIFT, TXFM_DIMENSIONS, TxfmInfo};

pub type Tx1dFn = fn(c: &mut [i32], stride: usize);

/// Packed kernel id of the lossless 4x4 Walsh-Hadamard transform.
pub const WHT_WHT: u32 = 6 | (6 << 5);

/// `QUANT_TABLE_BITS` (AV2 Spec v1.0.0 §3 Table 3.1).
///
/// At quantizer index 0 (`get_q(0, delta <= 0) == 64`) the dequantization
/// process of §7.14.4 computes `Round2(|q| * 64, 3)`, i.e. it scales every
/// coded lossless level by this constant before the inverse WHT.
pub const LOSSLESS_DEQUANT_SCALE: i32 = 8;

// ============================================================================
// Normative transform-type set (AV2 Spec v1.0.0 §3 Table 3.1, §5.20.8.2,
// §5.20.8.3, §9.6). The spec `TxType` enum is a dense index `0..=15`; the
// kernels above use the packed `horizontal_1d | (class << 3) | (vertical_1d << 5)`
// representation, so both directions of the mapping are needed.
// ============================================================================

/// `DCT_DCT` (`TxType` 0).
pub const SPEC_DCT_DCT: u8 = 0;
/// `ADST_DCT` (`TxType` 1): rows DCT, columns ADST.
pub const SPEC_ADST_DCT: u8 = 1;
/// `DCT_ADST` (`TxType` 2): rows ADST, columns DCT.
pub const SPEC_DCT_ADST: u8 = 2;
/// `ADST_ADST` (`TxType` 3).
pub const SPEC_ADST_ADST: u8 = 3;
/// `FLIPADST_DCT` (`TxType` 4).
pub const SPEC_FLIPADST_DCT: u8 = 4;
/// `DCT_FLIPADST` (`TxType` 5).
pub const SPEC_DCT_FLIPADST: u8 = 5;
/// `FLIPADST_FLIPADST` (`TxType` 6).
pub const SPEC_FLIPADST_FLIPADST: u8 = 6;
/// `ADST_FLIPADST` (`TxType` 7).
pub const SPEC_ADST_FLIPADST: u8 = 7;
/// `FLIPADST_ADST` (`TxType` 8).
pub const SPEC_FLIPADST_ADST: u8 = 8;
/// `IDTX` (`TxType` 9).
pub const SPEC_IDTX: u8 = 9;
/// `V_DCT` (`TxType` 10): rows identity, columns DCT.
pub const SPEC_V_DCT: u8 = 10;
/// `H_DCT` (`TxType` 11): rows DCT, columns identity.
pub const SPEC_H_DCT: u8 = 11;
/// `V_ADST` (`TxType` 12).
pub const SPEC_V_ADST: u8 = 12;
/// `H_ADST` (`TxType` 13).
pub const SPEC_H_ADST: u8 = 13;
/// `V_FLIPADST` (`TxType` 14).
pub const SPEC_V_FLIPADST: u8 = 14;
/// `H_FLIPADST` (`TxType` 15).
pub const SPEC_H_FLIPADST: u8 = 15;

/// `TxType` -> packed kernel id.
pub static TX_TYPE_TO_TXTP: [u32; 16] = [
    txtp::DCT_DCT,
    txtp::ADST_DCT,
    txtp::DCT_ADST,
    txtp::ADST_ADST,
    txtp::FLIPADST_DCT,
    txtp::DCT_FLIPADST,
    txtp::FLIPADST_FLIPADST,
    txtp::ADST_FLIPADST,
    txtp::FLIPADST_ADST,
    txtp::IDTX,
    txtp::V_DCT,
    txtp::H_DCT,
    txtp::V_ADST,
    txtp::H_ADST,
    txtp::V_FLIPADST,
    txtp::H_FLIPADST,
];

/// `Size_Class[txSz]` (AV2 Spec v1.0.0 §9, indexed like `RectTxfmSize`).
pub static SIZE_CLASS: [u8; N_RECT_TX_SIZES] = [
    0, 1, 2, 3, 3, 0, 0, 1, 1, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
];

/// `Mode_To_Angle[INTRA_MODES]` (AV2 Spec v1.0.0 §9).
pub static MODE_TO_ANGLE: [i16; 13] = [0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0];

/// `Mrl_Index_To_Delta[4]` (AV2 Spec v1.0.0 §7.13.2.5).
pub static MRL_INDEX_TO_DELTA: [i16; 4] = [0, 1, -1, 0];

/// `ANGLE_STEP` (AV2 Spec v1.0.0 §3 Table 3.1).
pub const ANGLE_STEP: i32 = 3;

/// `Md_Idx_To_Type[Size_Class][intraDir][intra_tx_type]` (AV2 Spec v1.0.0 §9).
///
/// The entries are `TxType` values; the intra transform-type syntax codes the
/// index into this table, not the type itself.
pub static MD_IDX_TO_TYPE: [[[u8; 7]; 13]; 4] = [
    [
        [0, 3, 1, 2, 7, 8, 13],
        [0, 3, 1, 2, 7, 10, 12],
        [0, 3, 1, 2, 8, 11, 13],
        [0, 3, 1, 2, 6, 7, 8],
        [0, 3, 1, 2, 7, 8, 13],
        [0, 3, 1, 2, 7, 12, 14],
        [0, 3, 1, 2, 7, 8, 13],
        [0, 3, 1, 2, 8, 11, 13],
        [0, 3, 1, 2, 7, 10, 12],
        [0, 3, 1, 2, 6, 7, 8],
        [0, 3, 1, 2, 7, 8, 12],
        [0, 3, 1, 2, 7, 8, 13],
        [0, 3, 2, 10, 11, 12, 13],
    ],
    [
        [0, 3, 1, 2, 4, 7, 8],
        [0, 3, 1, 2, 7, 6, 8],
        [0, 3, 1, 2, 8, 4, 7],
        [0, 3, 1, 2, 6, 7, 8],
        [0, 3, 1, 2, 5, 7, 8],
        [0, 3, 1, 2, 5, 7, 8],
        [0, 3, 1, 2, 4, 7, 8],
        [0, 3, 1, 2, 4, 8, 7],
        [0, 3, 1, 2, 5, 6, 7],
        [0, 3, 1, 2, 6, 7, 8],
        [0, 3, 1, 2, 4, 7, 8],
        [0, 3, 1, 2, 5, 7, 8],
        [0, 3, 1, 2, 10, 11, 13],
    ],
    [
        [0, 3, 1, 2, 4, 7, 8],
        [0, 3, 1, 2, 5, 7, 8],
        [0, 3, 1, 2, 4, 8, 7],
        [0, 3, 1, 2, 4, 7, 8],
        [0, 3, 1, 2, 4, 7, 8],
        [0, 3, 1, 2, 5, 7, 8],
        [0, 3, 1, 2, 4, 7, 8],
        [0, 3, 1, 2, 4, 6, 8],
        [0, 3, 1, 2, 5, 7, 8],
        [0, 3, 1, 2, 4, 7, 8],
        [0, 3, 1, 2, 4, 7, 8],
        [0, 3, 1, 2, 5, 7, 8],
        [0, 3, 1, 2, 10, 11, 12],
    ],
    [
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
        [0, 3, 1, 2, 4, 5, 6],
    ],
];

/// `Tx_Type_Inter_Inv_Set1[16]` (AV2 Spec v1.0.0 §5.20.8.2).
pub static TX_TYPE_INTER_INV_SET1: [u8; 16] =
    [9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 4, 5, 3, 6, 7, 8];

/// `Tx_Type_Inter_Inv_Set2[12]` (AV2 Spec v1.0.0 §5.20.8.2).
pub static TX_TYPE_INTER_INV_SET2: [u8; 12] = [9, 10, 11, 0, 1, 2, 4, 5, 3, 6, 7, 8];

/// `Tx_Type_Inter_Inv_Set3[2]` (AV2 Spec v1.0.0 §5.20.8.2).
pub static TX_TYPE_INTER_INV_SET3: [u8; 2] = [9, 0];

/// `Tx_Type_Inter_Inv_Set4[4]` (AV2 Spec v1.0.0 §5.20.8.2).
pub static TX_TYPE_INTER_INV_SET4: [u8; 4] = [0, 10, 11, 9];

/// `Tx_Type_Inv_Long[is_long_side_dct][orientation][symbol]`
/// (AV2 Spec v1.0.0 §5.20.8.2). `orientation` is 0 for wide transforms and 1
/// for tall transforms.
pub static TX_TYPE_INV_LONG: [[[u8; 4]; 2]; 2] = [
    [[10, 12, 14, 9], [11, 13, 15, 9]],
    [[0, 1, 4, 11], [0, 2, 5, 10]],
];

/// Returns `true` for `V_PRED..D67_PRED` (AV2 Spec v1.0.0 §5.20.7.31).
#[inline]
pub fn is_directional_mode(mode: u8) -> bool {
    (1..=8).contains(&mode)
}

/// `wide_angle_mapping(mode, w, h, pAngle)` (AV2 Spec v1.0.0 §5.20.8.2).
///
/// Unlike the prediction-time remap, this also substitutes the mode: tall
/// blocks map to `D203_PRED` and wide blocks to `D45_PRED`, which changes the
/// row of `Md_Idx_To_Type` used to derive an intra transform type.
pub fn wide_angle_mapping(mode: u8, w: usize, h: usize, p_angle: i32) -> (u8, i32) {
    const D45: u8 = 3;
    const D203: u8 = 7;
    const THRES_2: i32 = 61;
    const THRES_4: i32 = 73;
    const THRES_8: i32 = 82;
    const THRES_16: i32 = 86;
    if (h == 2 * w && p_angle < THRES_2)
        || (h == 4 * w && p_angle < THRES_4)
        || (h == 8 * w && p_angle < THRES_8)
        || (h == 16 * w && p_angle < THRES_16)
    {
        return (D203, 180 + p_angle);
    }
    if (w == 2 * h && p_angle > 270 - THRES_2)
        || (w == 4 * h && p_angle > 270 - THRES_4)
        || (w == 8 * h && p_angle > 270 - THRES_8)
        || (w == 16 * h && p_angle > 270 - THRES_16)
    {
        return (D45, p_angle - 180);
    }
    (mode, p_angle)
}

/// Returns the `TxType` for a packed kernel id.
pub fn spec_tx_type(txtp: u32) -> Option<u8> {
    TX_TYPE_TO_TXTP
        .iter()
        .position(|&t| t == txtp)
        .map(|i| i as u8)
}

/// Returns `true` when the packed kernel id uses only 2D (non 1D-vertical or
/// 1D-horizontal) kernels, i.e. the coefficient scan is the 2D anti-diagonal
/// scan of AV2 Spec v1.0.0 §5.20.7.30.
#[inline]
pub fn is_2d_tx_class(txtp: u32) -> bool {
    !matches!(
        spec_tx_type(txtp),
        Some(
            SPEC_V_DCT | SPEC_H_DCT | SPEC_V_ADST | SPEC_H_ADST | SPEC_V_FLIPADST | SPEC_H_FLIPADST
        )
    )
}

/// One transform type that can be signalled for a block.
///
/// `is_long_side_dct` is only coded for `TX_SET_WIDE_32` / `TX_SET_HIGH_32`;
/// `symbol` is the `intra_tx_type` / `inter_tx_type` value, and `symbol2` the
/// `inter_tx_type_offset` for `TX_SET_INTER_1` / `TX_SET_INTER_2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxTypeChoice {
    /// Packed kernel id.
    pub txtp: u32,
    /// `is_long_side_dct` value (inferred to 1 for the 64-sample long sets).
    pub is_long_side_dct: u8,
    /// Primary transform-type symbol.
    pub symbol: u8,
    /// Secondary `inter_tx_type_offset` symbol (0 when unused).
    pub symbol2: u8,
}

/// Derives the signalled symbols of a chosen intra luma transform type
/// (AV2 Spec v1.0.0 §5.20.8.2 `transform_type`).
///
/// Returns `None` when the type cannot be expressed by `set` for this block.
pub fn intra_tx_type_choice(
    t_dim: &TxfmInfo,
    tx: usize,
    y_mode: u8,
    angle: i8,
    mrl: u8,
    set: TxSet,
    txtp: u32,
) -> Option<TxTypeChoice> {
    let want = spec_tx_type(txtp)?;
    let w = 4 * t_dim.w as usize;
    let h = 4 * t_dim.h as usize;
    let wide = t_dim.w > t_dim.h;
    match set {
        TxSet::Wide64 | TxSet::High64 | TxSet::Wide32 | TxSet::High32 => {
            let orient = if wide { 0 } else { 1 };
            let isld = if matches!(set, TxSet::Wide32 | TxSet::High32) {
                let row = &TX_TYPE_INV_LONG[1][orient];
                if let Some(sym) = row.iter().position(|&t| t == want) {
                    return Some(TxTypeChoice {
                        txtp,
                        is_long_side_dct: 1,
                        symbol: sym as u8,
                        symbol2: 0,
                    });
                }
                let row0 = &TX_TYPE_INV_LONG[0][orient];
                if let Some(sym) = row0.iter().position(|&t| t == want) {
                    return Some(TxTypeChoice {
                        txtp,
                        is_long_side_dct: 0,
                        symbol: sym as u8,
                        symbol2: 0,
                    });
                }
                return None;
            } else {
                1
            };
            let sym = TX_TYPE_INV_LONG[isld as usize][orient]
                .iter()
                .position(|&t| t == want)?;
            Some(TxTypeChoice {
                txtp,
                is_long_side_dct: isld,
                symbol: sym as u8,
                symbol2: 0,
            })
        }
        TxSet::Intra1 | TxSet::Intra2 => {
            let size_class = SIZE_CLASS[tx] as usize;
            let mut intra_dir = y_mode.min(12);
            if is_directional_mode(intra_dir) {
                let p_angle = MODE_TO_ANGLE[intra_dir as usize] as i32
                    + angle as i32 * ANGLE_STEP
                    + MRL_INDEX_TO_DELTA[mrl.min(3) as usize] as i32;
                intra_dir = wide_angle_mapping(intra_dir, w, h, p_angle).0;
            }
            let row = &MD_IDX_TO_TYPE[size_class][intra_dir as usize];
            let n = if set == TxSet::Intra1 { 7 } else { 2 };
            let sym = row[..n].iter().position(|&t| t == want)?;
            Some(TxTypeChoice {
                txtp,
                is_long_side_dct: 1,
                symbol: sym as u8,
                symbol2: 0,
            })
        }
        _ => None,
    }
}

/// Enumerates the intra luma transform types that `set` can signal for this
/// block, in symbol order (AV2 Spec v1.0.0 §5.20.8.2 / §9 `Md_Idx_To_Type`).
///
/// Only 2D-class kernels are returned: the encoder's quantizer and coefficient
/// scan are defined for the 2D class, so the 1D (`V_*` / `H_*`) entries of the
/// long and inter sets are not selectable here.
pub fn intra_tx_type_choices(
    t_dim: &TxfmInfo,
    tx: usize,
    y_mode: u8,
    angle: i8,
    mrl: u8,
    set: TxSet,
) -> ([TxTypeChoice; 7], usize) {
    let mut out = [TxTypeChoice {
        txtp: DCT_DCT,
        is_long_side_dct: 1,
        symbol: 0,
        symbol2: 0,
    }; 7];
    let mut n = 0;
    let w = 4 * t_dim.w as usize;
    let h = 4 * t_dim.h as usize;
    let wide = t_dim.w > t_dim.h;
    match set {
        TxSet::Wide64 | TxSet::High64 | TxSet::Wide32 | TxSet::High32 => {
            // Only `is_long_side_dct = 1` entries are 2D kernels; the
            // `is_long_side_dct = 0` rows of the 32-sample sets are 1D.
            let orient = if wide { 0 } else { 1 };
            let isld = 1usize;
            for (sym, &t) in TX_TYPE_INV_LONG[isld][orient].iter().enumerate() {
                let txtp = TX_TYPE_TO_TXTP[t as usize];
                if is_2d_tx_class(txtp) {
                    out[n] = TxTypeChoice {
                        txtp,
                        is_long_side_dct: isld as u8,
                        symbol: sym as u8,
                        symbol2: 0,
                    };
                    n += 1;
                }
            }
        }
        TxSet::Intra1 | TxSet::Intra2 => {
            let size_class = SIZE_CLASS[tx] as usize;
            let mut intra_dir = y_mode.min(12);
            if is_directional_mode(intra_dir) {
                let p_angle = MODE_TO_ANGLE[intra_dir as usize] as i32
                    + angle as i32 * ANGLE_STEP
                    + MRL_INDEX_TO_DELTA[mrl.min(3) as usize] as i32;
                intra_dir = wide_angle_mapping(intra_dir, w, h, p_angle).0;
            }
            let row = &MD_IDX_TO_TYPE[size_class][intra_dir as usize];
            let count = if set == TxSet::Intra1 { 7 } else { 2 };
            for (sym, &t) in row[..count].iter().enumerate() {
                let txtp = TX_TYPE_TO_TXTP[t as usize];
                if is_2d_tx_class(txtp) {
                    out[n] = TxTypeChoice {
                        txtp,
                        is_long_side_dct: 1,
                        symbol: sym as u8,
                        symbol2: 0,
                    };
                    n += 1;
                }
            }
        }
        _ => {}
    }
    (out, n)
}

/// Enumerates the inter luma transform types that `set` can signal
/// (AV2 Spec v1.0.0 §5.20.8.2 `Tx_Type_Inter_Inv_Set*` / `Tx_Type_Inv_Long`).
pub fn inter_tx_type_choices(t_dim: &TxfmInfo, set: TxSet) -> ([TxTypeChoice; 16], usize) {
    let mut out = [TxTypeChoice {
        txtp: DCT_DCT,
        is_long_side_dct: 1,
        symbol: 0,
        symbol2: 0,
    }; 16];
    let mut n = 0;
    let wide = t_dim.w > t_dim.h;
    let push = |txtp: u32,
                isld: u8,
                symbol: u8,
                symbol2: u8,
                out: &mut [TxTypeChoice; 16],
                n: &mut usize| {
        // The identity kernels are not selectable by this encoder: its forward
        // identity normalization is not byte-exact against the reference
        // decoder, so `IDTX` stays decoder-only (see
        // docs/av2/UNIT9_ENCODER.md).
        if is_2d_tx_class(txtp) && !out[..*n].iter().any(|c| c.txtp == txtp) {
            out[*n] = TxTypeChoice {
                txtp,
                is_long_side_dct: isld,
                symbol,
                symbol2,
            };
            *n += 1;
        }
    };
    match set {
        TxSet::DctOnly => push(DCT_DCT, 1, 0, 0, &mut out, &mut n),
        TxSet::Wide64 | TxSet::High64 | TxSet::Wide32 | TxSet::High32 => {
            let orient = if wide { 0 } else { 1 };
            for (sym, &t) in TX_TYPE_INV_LONG[1][orient].iter().enumerate() {
                push(
                    TX_TYPE_TO_TXTP[t as usize],
                    1,
                    sym as u8,
                    0,
                    &mut out,
                    &mut n,
                );
            }
            if matches!(set, TxSet::Wide32 | TxSet::High32) {
                for (sym, &t) in TX_TYPE_INV_LONG[0][orient].iter().enumerate() {
                    push(
                        TX_TYPE_TO_TXTP[t as usize],
                        0,
                        sym as u8,
                        0,
                        &mut out,
                        &mut n,
                    );
                }
            }
        }
        TxSet::DctIdtx => {
            for (sym, &t) in TX_TYPE_INTER_INV_SET3.iter().enumerate() {
                push(
                    TX_TYPE_TO_TXTP[t as usize],
                    1,
                    sym as u8,
                    0,
                    &mut out,
                    &mut n,
                );
            }
        }
        TxSet::DctIdtxIddct => {
            for (sym, &t) in TX_TYPE_INTER_INV_SET4.iter().enumerate() {
                push(
                    TX_TYPE_TO_TXTP[t as usize],
                    1,
                    sym as u8,
                    0,
                    &mut out,
                    &mut n,
                );
            }
        }
        TxSet::Inter1 => {
            for (idx, &t) in TX_TYPE_INTER_INV_SET1.iter().enumerate() {
                push(
                    TX_TYPE_TO_TXTP[t as usize],
                    1,
                    (idx / 8) as u8,
                    (idx % 8) as u8,
                    &mut out,
                    &mut n,
                );
            }
        }
        TxSet::Inter2 => {
            for (idx, &t) in TX_TYPE_INTER_INV_SET2.iter().enumerate() {
                push(
                    TX_TYPE_TO_TXTP[t as usize],
                    1,
                    (idx / 8) as u8,
                    (idx % 8) as u8,
                    &mut out,
                    &mut n,
                );
            }
        }
        TxSet::Intra1 | TxSet::Intra2 => {}
    }
    (out, n)
}

/// Derives the signalled symbols of a chosen inter luma transform type.
pub fn inter_tx_type_choice(t_dim: &TxfmInfo, set: TxSet, txtp: u32) -> Option<TxTypeChoice> {
    let (choices, n) = inter_tx_type_choices(t_dim, set);
    choices[..n].iter().find(|c| c.txtp == txtp).copied()
}

static DCT8_KERNEL: [i8; 16] = [
    89, 75, 50, 18, 75, -18, -89, -50, 50, -89, 18, 75, 18, -50, 75, -89,
];

static DCT16_KERNEL: [i8; 64] = [
    90, 87, 80, 70, 57, 43, 26, 9, 87, 57, 9, -43, -80, -90, -70, -26, 80, 9, -70, -87, -26, 57,
    90, 43, 70, -43, -87, 9, 90, 26, -80, -57, 57, -80, -26, 90, -9, -87, 43, 70, 43, -90, 57, 26,
    -87, 70, 9, -80, 26, -70, 90, -80, 43, 9, -57, 87, 9, -26, 43, -57, 70, -80, 87, -90,
];

static DCT32_KERNEL: [i8; 256] = [
    90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 47, 39, 30, 22, 13, 4, 90, 82, 67, 47, 22, -4, -30,
    -54, -73, -85, -90, -88, -78, -61, -39, -13, 88, 67, 30, -13, -54, -82, -90, -78, -47, -4, 39,
    73, 90, 85, 61, 22, 85, 47, -13, -67, -90, -73, -22, 39, 82, 88, 54, -4, -61, -90, -78, -30,
    82, 22, -54, -90, -61, 13, 78, 85, 30, -47, -90, -67, 4, 73, 88, 39, 78, -4, -82, -73, 13, 85,
    67, -22, -88, -61, 30, 90, 54, -39, -90, -47, 73, -30, -90, -22, 78, 67, -39, -90, -13, 82, 61,
    -47, -88, -4, 85, 54, 67, -54, -78, 39, 85, -22, -90, 4, 90, 13, -88, -30, 82, 47, -73, -61,
    61, -73, -47, 82, 30, -88, -13, 90, -4, -90, 22, 85, -39, -78, 54, 67, 54, -85, -4, 88, -47,
    -61, 82, 13, -90, 39, 67, -78, -22, 90, -30, -73, 47, -90, 39, 54, -90, 30, 61, -88, 22, 67,
    -85, 13, 73, -82, 4, 78, 39, -88, 73, -4, -67, 90, -47, -30, 85, -78, 13, 61, -90, 54, 22, -82,
    30, -78, 90, -61, 4, 54, -88, 82, -39, -22, 73, -90, 67, -13, -47, 85, 22, -61, 85, -90, 73,
    -39, -4, 47, -78, 90, -82, 54, -13, -30, 67, -88, 13, -39, 61, -78, 88, -90, 85, -73, 54, -30,
    4, 22, -47, 67, -82, 90, 4, -13, 22, -30, 39, -47, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90,
];

static ADST4_KERNEL: [i8; 16] = [
    18, 50, 75, 89, 50, 89, 18, -75, 75, 18, -89, 50, 89, -75, 50, -18,
];

static ADST8_KERNEL: [i8; 64] = [
    11, 34, 54, 71, 84, 88, 79, 50, 28, 74, 89, 68, 17, -44, -83, -69, 44, 89, 48, -41, -89, -44,
    50, 81, 58, 76, -34, -86, 10, 88, 6, -84, 70, 39, -87, 1, 86, -44, -59, 78, 79, -12, -66, 87,
    -35, -44, 86, -62, 86, -58, 12, 38, -75, 88, -74, 40, 89, -86, 79, -70, 58, -44, 29, -14,
];

static ADST16_KERNEL: [i8; 256] = [
    8, 25, 41, 55, 67, 77, 84, 88, 89, 87, 81, 73, 62, 48, 33, 17, 17, 48, 73, 87, 88, 77, 55, 25,
    -8, -41, -67, -84, -89, -81, -62, -33, 25, 67, 88, 81, 48, 0, -48, -81, -88, -67, -25, 25, 67,
    88, 81, 48, 33, 81, 84, 41, -25, -77, -87, -48, 17, 73, 88, 55, -8, -67, -89, -62, 41, 88, 62,
    -17, -81, -77, -8, 67, 87, 33, -48, -89, -55, 25, 84, 73, 48, 88, 25, -67, -81, 0, 81, 67, -25,
    -88, -48, 48, 88, 25, -67, -81, 55, 81, -17, -89, -25, 77, 62, -48, -84, 8, 88, 33, -73, -67,
    41, 87, 62, 67, -55, -73, 48, 77, -41, -81, 33, 84, -25, -87, 17, 88, -8, -89, 67, 48, -81,
    -25, 88, 0, -88, 25, 81, -48, -67, 67, 48, -81, -25, 88, 73, 25, -89, 33, 67, -77, -17, 88,
    -41, -62, 81, 8, -87, 48, 55, -84, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77,
    77, 81, -25, -48, 88, -67, 0, 67, -88, 48, 25, -81, 81, -25, -48, 88, -67, 84, -48, -8, 62,
    -88, 77, -33, -25, 73, -89, 67, -17, -41, 81, -87, 55, 87, -67, 33, 8, -48, 77, -89, 81, -55,
    17, 25, -62, 84, -88, 73, -41, 88, -81, 67, -48, 25, 0, -25, 48, -67, 81, -88, 88, -81, 67,
    -48, 25, 89, -88, 87, -84, 81, -77, 73, -67, 62, -55, 48, -41, 33, -25, 17, -8,
];

static FLIPADST4_KERNEL: [i8; 16] = [
    89, 75, 50, 18, 75, -18, -89, -50, 50, -89, 18, 75, 18, -50, 75, -89,
];

static FLIPADST16_KERNEL: [i8; 256] = [
    89, 88, 87, 84, 81, 77, 73, 67, 62, 55, 48, 41, 33, 25, 17, 8, 88, 81, 67, 48, 25, 0, -25, -48,
    -67, -81, -88, -88, -81, -67, -48, -25, 87, 67, 33, -8, -48, -77, -89, -81, -55, -17, 25, 62,
    84, 88, 73, 41, 84, 48, -8, -62, -88, -77, -33, 25, 73, 89, 67, 17, -41, -81, -87, -55, 81, 25,
    -48, -88, -67, 0, 67, 88, 48, -25, -81, -81, -25, 48, 88, 67, 77, 0, -77, -77, 0, 77, 77, 0,
    -77, -77, 0, 77, 77, 0, -77, -77, 73, -25, -89, -33, 67, 77, -17, -88, -41, 62, 81, -8, -87,
    -48, 55, 84, 67, -48, -81, 25, 88, 0, -88, -25, 81, 48, -67, -67, 48, 81, -25, -88, 62, -67,
    -55, 73, 48, -77, -41, 81, 33, -84, -25, 87, 17, -88, -8, 89, 55, -81, -17, 89, -25, -77, 62,
    48, -84, -8, 88, -33, -73, 67, 41, -87, 48, -88, 25, 67, -81, 0, 81, -67, -25, 88, -48, -48,
    88, -25, -67, 81, 41, -88, 62, 17, -81, 77, -8, -67, 87, -33, -48, 89, -55, -25, 84, -73, 33,
    -81, 84, -41, -25, 77, -87, 48, 17, -73, 88, -55, -8, 67, -89, 62, 25, -67, 88, -81, 48, 0,
    -48, 81, -88, 67, -25, -25, 67, -88, 81, -48, 17, -48, 73, -87, 88, -77, 55, -25, -8, 41, -67,
    84, -89, 81, -62, 33, 8, -25, 41, -55, 67, -77, 84, -88, 89, -87, 81, -73, 62, -48, 33, -17,
];

static DDT8_KERNEL: [i8; 64] = [
    4, 6, 22, 57, 96, 103, 78, 56, 7, 14, 48, 94, 73, -17, -79, -96, 15, 36, 85, 76, -43, -80, 7,
    98, 33, 77, 88, -26, -69, 56, 56, -77, 65, 100, 0, -73, 55, 15, -82, 54, 98, 45, -86, 34, 20,
    -66, 79, -33, 106, -57, -23, 54, -71, 75, -56, 19, 80, -98, 82, -66, 53, -41, 26, -6,
];

static DDT16_KERNEL: [i8; 256] = [
    12, 17, 37, 45, 47, 60, 64, 82, 89, 100, 92, 84, 69, 50, 51, 44, 15, 23, 49, 60, 60, 74, 70,
    73, 48, 9, -35, -71, -83, -79, -89, -95, 19, 30, 60, 69, 61, 64, 40, 3, -53, -99, -91, -46, 2,
    47, 73, 124, 23, 38, 69, 73, 49, 28, -19, -80, -96, -45, 42, 88, 75, 14, -17, -126, 30, 48, 75,
    66, 19, -31, -79, -91, -5, 84, 71, -16, -78, -60, -45, 108, 39, 61, 75, 40, -29, -87, -78, 10,
    89, 36, -69, -67, 18, 67, 89, -81, 51, 76, 61, -8, -77, -82, 11, 94, 16, -81, -22, 79, 50, -37,
    -103, 54, 66, 87, 29, -65, -83, 4, 92, 18, -83, 4, 85, -22, -85, -6, 97, -30, 78, 83, -18, -91,
    -16, 88, 28, -84, 12, 73, -60, -46, 81, 49, -83, 16, 88, 59, -67, -57, 75, 54, -85, -5, 75,
    -60, -17, 84, -43, -80, 71, -6, 94, 19, -96, 21, 93, -55, -41, 80, -51, -17, 77, -68, -6, 98,
    -56, 1, 97, -30, -83, 86, 3, -77, 82, -17, -43, 76, -70, 15, 53, -99, 44, 3, 93, -73, -28, 81,
    -92, 29, 39, -70, 81, -55, 11, 46, -81, 90, -31, -4, 83, -99, 40, 8, -74, 88, -83, 47, -14,
    -21, 56, -83, 88, -71, 22, 5, 68, -99, 84, -69, 32, 3, -37, 55, -75, 81, -83, 82, -69, 48, -11,
    -3, 50, -76, 83, -90, 97, -86, 83, -68, 67, -56, 49, -40, 32, -19, 5, 2,
];

// ============================================================================
// 1D INVERSE TRANSFORMS (Matching AV2 Spec §7.15 & AV2 Spec)
// ============================================================================

const fn transpose_kernel<const N: usize, const NN: usize>(m: &[i8; NN]) -> [i8; NN] {
    let mut out = [0i8; NN];
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < N {
            out[j * N + i] = m[i * N + j];
            j += 1;
        }
        i += 1;
    }
    out
}

static DCT8_KERNEL_T: [i8; 16] = transpose_kernel::<4, 16>(&DCT8_KERNEL);
static DCT16_KERNEL_T: [i8; 64] = transpose_kernel::<8, 64>(&DCT16_KERNEL);
static DCT32_KERNEL_T: [i8; 256] = transpose_kernel::<16, 256>(&DCT32_KERNEL);

macro_rules! impl_inv_dct_stage_simd {
    ($fn_name:ident, $n:expr) => {
        #[inline]
        fn $fn_name(c: &mut [i32], stride: usize, mat_t: &[i8]) {
            use std::simd::Simd;
            use std::simd::prelude::*;

            const N: usize = $n;
            let k = N * 2 - 1;
            let mut b_vec = Simd::<i32, N>::splat(0);
            for j_idx in 0..N {
                let cj = c[(j_idx * 2 + 1) * stride];
                if cj != 0 {
                    let col =
                        Simd::<i8, N>::from_slice(&mat_t[j_idx * N..(j_idx + 1) * N]).cast::<i32>();
                    b_vec += col * Simd::<i32, N>::splat(cj);
                }
            }
            let b = b_vec.to_array();
            let mut a = [0i32; N];
            for (i, ai) in a.iter_mut().enumerate() {
                *ai = c[i * 2 * stride];
            }
            for i in 0..N {
                c[i * stride] = a[i] + b[i];
                c[(k - i) * stride] = a[i] - b[i];
            }
        }
    };
}

impl_inv_dct_stage_simd!(inv_dct_stage_simd4, 4);
impl_inv_dct_stage_simd!(inv_dct_stage_simd8, 8);
impl_inv_dct_stage_simd!(inv_dct_stage_simd16, 16);

pub fn inv_dct4_1d(c: &mut [i32], stride: usize) {
    let a0 = (c[0] + c[2 * stride]) * 64;
    let a1 = (c[0] - c[2 * stride]) * 64;
    let b0 = c[stride] * 83 + c[3 * stride] * 35;
    let b1 = c[stride] * 35 - c[3 * stride] * 83;

    c[0] = a0 + b0;
    c[stride] = a1 + b1;
    c[2 * stride] = a1 - b1;
    c[3 * stride] = a0 - b0;
}

pub fn inv_dct8_1d(c: &mut [i32], stride: usize) {
    inv_dct4_1d(c, 2 * stride);
    inv_dct_stage_simd4(c, stride, &DCT8_KERNEL_T);
}

pub fn inv_dct16_1d(c: &mut [i32], stride: usize) {
    inv_dct8_1d(c, 2 * stride);
    inv_dct_stage_simd8(c, stride, &DCT16_KERNEL_T);
}

pub fn inv_dct32_1d(c: &mut [i32], stride: usize) {
    inv_dct16_1d(c, 2 * stride);
    inv_dct_stage_simd16(c, stride, &DCT32_KERNEL_T);
}

fn inv_dst_1d(c: &mut [i32], stride: usize, mat: &[i8], n: usize, flip: bool) {
    let mut sums = [0i32; 16];
    let mut mi = 0;

    for i in 0..n {
        let mut sum = 0i32;
        for j in 0..n {
            sum += mat[mi] as i32 * c[j * stride];
            mi += 1;
        }
        sums[i] = sum;
    }

    if flip {
        for i in 0..n {
            c[(n - 1 - i) * stride] = sums[i];
        }
    } else {
        for i in 0..n {
            c[i * stride] = sums[i];
        }
    }
}

pub fn inv_adst4_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &ADST4_KERNEL, 4, false);
}

pub fn inv_adst8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &ADST8_KERNEL, 8, false);
}

pub fn inv_adst16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &ADST16_KERNEL, 16, false);
}

pub fn inv_flipadst4_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &FLIPADST4_KERNEL, 4, false);
}

pub fn inv_flipadst8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &ADST8_KERNEL, 8, true);
}

pub fn inv_flipadst16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &FLIPADST16_KERNEL, 16, false);
}

pub fn inv_ddt8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &DDT8_KERNEL, 8, false);
}

pub fn inv_ddt16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &DDT16_KERNEL, 16, false);
}

pub fn inv_flipddt8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &DDT8_KERNEL, 8, true);
}

pub fn inv_flipddt16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, stride, &DDT16_KERNEL, 16, true);
}

pub fn inv_identity4_1d(c: &mut [i32], stride: usize) {
    for i in 0..4 {
        c[stride * i] *= 128;
    }
}

pub fn inv_identity8_1d(c: &mut [i32], stride: usize) {
    for i in 0..8 {
        c[stride * i] *= 181;
    }
}

pub fn inv_identity16_1d(c: &mut [i32], stride: usize) {
    for i in 0..16 {
        c[stride * i] *= 256;
    }
}

pub fn inv_identity32_1d(c: &mut [i32], stride: usize) {
    for i in 0..32 {
        c[stride * i] *= 362;
    }
}

pub fn inv_wht4_1d(c: &mut [i32], stride: usize) {
    let in0 = c[0];
    let in1 = c[stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];

    let t0 = in0 + in1;
    let t2 = in2 - in3;
    let t4 = (t0 - t2) >> 1;
    let t3 = t4 - in3;
    let t1 = t4 - in1;

    c[0] = t0 - t3;
    c[stride] = t3;
    c[2 * stride] = t1;
    c[3 * stride] = t2 + t1;
}

pub fn inv_wht_wht_4x4(coeff: &[i32; 16], tmp: &mut [i32; 16]) {
    for y in 0..4 {
        for x in 0..4 {
            tmp[y * 4 + x] = coeff[y + x * 4] >> 3;
        }
        inv_wht4_1d(&mut tmp[y * 4..], 1);
    }
    for x in 0..4 {
        inv_wht4_1d(&mut tmp[x..], 4);
    }
}

// ============================================================================
// 1D FORWARD TRANSFORMS (Exact transpose/inverse of 1D inverse kernels)
// ============================================================================

macro_rules! impl_fwd_dct_stage_simd {
    ($fn_name:ident, $n:expr) => {
        #[inline]
        fn $fn_name(c: &mut [i32], stride: usize, mat: &[i8]) {
            use std::simd::Simd;
            use std::simd::prelude::*;

            const N: usize = $n;
            let k = N * 2 - 1;
            let mut a = [0i32; N];
            let mut b = [0i32; N];

            for i in 0..N {
                let c0 = c[i * stride];
                let c1 = c[(k - i) * stride];
                a[i] = c0 + c1;
                b[i] = c0 - c1;
            }

            for (i, &ai) in a.iter().enumerate() {
                c[i * 2 * stride] = ai;
            }

            let mut sum_vec = Simd::<i32, N>::splat(0);
            for (i, &bi) in b.iter().enumerate() {
                if bi != 0 {
                    let row = Simd::<i8, N>::from_slice(&mat[i * N..(i + 1) * N]).cast::<i32>();
                    sum_vec += row * Simd::<i32, N>::splat(bi);
                }
            }
            let sums = sum_vec.to_array();
            for (j_idx, &s) in sums.iter().enumerate() {
                c[(j_idx * 2 + 1) * stride] = s;
            }
        }
    };
}

impl_fwd_dct_stage_simd!(fwd_dct_stage_simd4, 4);
impl_fwd_dct_stage_simd!(fwd_dct_stage_simd8, 8);
impl_fwd_dct_stage_simd!(fwd_dct_stage_simd16, 16);

pub fn fwd_dct4_1d(c: &mut [i32], stride: usize) {
    let s0 = c[0] + c[3 * stride];
    let s1 = c[stride] + c[2 * stride];
    let d0 = c[0] - c[3 * stride];
    let d1 = c[stride] - c[2 * stride];

    c[0] = (s0 + s1) * 64;
    c[2 * stride] = (s0 - s1) * 64;
    c[stride] = d0 * 83 + d1 * 35;
    c[3 * stride] = d0 * 35 - d1 * 83;
}

pub fn fwd_dct8_1d(c: &mut [i32], stride: usize) {
    fwd_dct_stage_simd4(c, stride, &DCT8_KERNEL);
    fwd_dct4_1d(c, 2 * stride);
}

pub fn fwd_dct16_1d(c: &mut [i32], stride: usize) {
    fwd_dct_stage_simd8(c, stride, &DCT16_KERNEL);
    fwd_dct8_1d(c, 2 * stride);
}

pub fn fwd_dct32_1d(c: &mut [i32], stride: usize) {
    fwd_dct_stage_simd16(c, stride, &DCT32_KERNEL);
    fwd_dct16_1d(c, 2 * stride);
}

fn fwd_dst_1d(c: &mut [i32], stride: usize, mat: &[i8], n: usize, flip: bool) {
    let mut inp = [0i32; 16];
    if flip {
        for i in 0..n {
            inp[i] = c[(n - 1 - i) * stride];
        }
    } else {
        for i in 0..n {
            inp[i] = c[i * stride];
        }
    }

    for j in 0..n {
        let mut sum = 0i32;
        for i in 0..n {
            sum += (mat[i * n + j] as i32) * inp[i];
        }
        c[j * stride] = sum;
    }
}

pub fn fwd_adst4_1d(c: &mut [i32], stride: usize) {
    fwd_dst_1d(c, stride, &ADST4_KERNEL, 4, false);
}

pub fn fwd_adst8_1d(c: &mut [i32], stride: usize) {
    fwd_dst_1d(c, stride, &ADST8_KERNEL, 8, false);
}

pub fn fwd_adst16_1d(c: &mut [i32], stride: usize) {
    fwd_dst_1d(c, stride, &ADST16_KERNEL, 16, false);
}

pub fn fwd_flipadst4_1d(c: &mut [i32], stride: usize) {
    fwd_dst_1d(c, stride, &FLIPADST4_KERNEL, 4, false);
}

pub fn fwd_flipadst8_1d(c: &mut [i32], stride: usize) {
    fwd_dst_1d(c, stride, &ADST8_KERNEL, 8, true);
}

pub fn fwd_flipadst16_1d(c: &mut [i32], stride: usize) {
    fwd_dst_1d(c, stride, &FLIPADST16_KERNEL, 16, false);
}

pub fn fwd_ddt8_1d(c: &mut [i32], stride: usize) {
    fwd_dst_1d(c, stride, &DDT8_KERNEL, 8, false);
}

pub fn fwd_ddt16_1d(c: &mut [i32], stride: usize) {
    fwd_dst_1d(c, stride, &DDT16_KERNEL, 16, false);
}

pub fn fwd_flipddt8_1d(c: &mut [i32], stride: usize) {
    fwd_dst_1d(c, stride, &DDT8_KERNEL, 8, true);
}

pub fn fwd_flipddt16_1d(c: &mut [i32], stride: usize) {
    fwd_dst_1d(c, stride, &DDT16_KERNEL, 16, true);
}

pub fn fwd_identity4_1d(c: &mut [i32], stride: usize) {
    for i in 0..4 {
        c[stride * i] *= 128;
    }
}

pub fn fwd_identity8_1d(c: &mut [i32], stride: usize) {
    for i in 0..8 {
        c[stride * i] *= 181;
    }
}

pub fn fwd_identity16_1d(c: &mut [i32], stride: usize) {
    for i in 0..16 {
        c[stride * i] *= 256;
    }
}

pub fn fwd_identity32_1d(c: &mut [i32], stride: usize) {
    for i in 0..32 {
        c[stride * i] *= 362;
    }
}

/// Exact integer inverse of `inv_wht4_1d` for lossless 4x4 coding.
pub fn fwd_wht4_1d(c: &mut [i32], stride: usize) {
    let out0 = c[0];
    let out1 = c[stride];
    let out2 = c[2 * stride];
    let out3 = c[3 * stride];

    let t0 = out0 + out1;
    let t2 = out3 - out2;
    let t4 = (t0 - t2) >> 1;
    let in3 = t4 - out1;
    let in1 = t4 - out2;
    let in0 = t0 - in1;
    let in2 = t2 + in3;

    c[0] = in0;
    c[stride] = in1;
    c[2 * stride] = in2;
    c[3 * stride] = in3;
}

/// Forward 4x4 Walsh-Hadamard transform producing column-major quantized
/// levels (`coeff_out[y + x * 4]`) for lossless coding.
///
/// The levels are the coded `Quant` values of AV2 Spec v1.0.0 §7.14.4: the
/// inverse process dequantizes them with `Round2(|q| * 64, QUANT_TABLE_BITS)`
/// (`LOSSLESS_DEQUANT_SCALE`), then `inv_wht_wht_4x4` pre-shifts by 3 and
/// applies the inverse WHT pair, so `inv(dequant(q)) == res`.
pub fn fwd_wht_wht_4x4(res: &[i32], stride: usize, coeff_out: &mut [i32; 16]) {
    let mut tmp = [0i32; 16];
    for y in 0..4 {
        for x in 0..4 {
            tmp[y * 4 + x] = res[y * stride + x];
        }
    }
    for x in 0..4 {
        fwd_wht4_1d(&mut tmp[x..], 4);
    }
    for y in 0..4 {
        fwd_wht4_1d(&mut tmp[y * 4..], 1);
        for x in 0..4 {
            coeff_out[y + x * 4] = tmp[y * 4 + x];
        }
    }
}

/// 64-point 1D Inverse DCT (operates on the 32 low-frequency coefficients per AV2 Spec §7.13.2).
pub fn inv_dct64_1d(c: &mut [i32], stride: usize) {
    inv_dct32_1d(c, stride);
}

/// 64-point 1D Forward DCT (projects onto the 32 low-frequency basis vectors per AV2 Spec §7.13.2).
pub fn fwd_dct64_1d(c: &mut [i32], stride: usize) {
    fwd_dct32_1d(c, stride);
}

/// 64-point 1D Inverse Identity transform (`4 * sqrt(2)` scaling).
pub fn inv_identity64_1d(c: &mut [i32], stride: usize) {
    inv_identity32_1d(c, stride);
}

/// 64-point 1D Forward Identity transform (`4 * sqrt(2)` scaling).
pub fn fwd_identity64_1d(c: &mut [i32], stride: usize) {
    fwd_identity32_1d(c, stride);
}

pub static INV_TX1D_FNS: [[Option<Tx1dFn>; N_TX_1D_TYPES - 1]; N_TX_SIZES] = {
    const DCT: usize = 0;
    const IDENTITY: usize = 1;
    const ADST: usize = 2;
    const FLIPADST: usize = 3;
    const DDT: usize = 4;
    const FLIPDDT: usize = 5;
    const NONE: Option<Tx1dFn> = None;

    let mut t = [[NONE; N_TX_1D_TYPES - 1]; N_TX_SIZES];

    t[0][DCT] = Some(inv_dct4_1d);
    t[0][IDENTITY] = Some(inv_identity4_1d);
    t[0][ADST] = Some(inv_adst4_1d);
    t[0][FLIPADST] = Some(inv_flipadst4_1d);

    t[1][DCT] = Some(inv_dct8_1d);
    t[1][IDENTITY] = Some(inv_identity8_1d);
    t[1][ADST] = Some(inv_adst8_1d);
    t[1][FLIPADST] = Some(inv_flipadst8_1d);
    t[1][DDT] = Some(inv_ddt8_1d);
    t[1][FLIPDDT] = Some(inv_flipddt8_1d);

    t[2][DCT] = Some(inv_dct16_1d);
    t[2][IDENTITY] = Some(inv_identity16_1d);
    t[2][ADST] = Some(inv_adst16_1d);
    t[2][FLIPADST] = Some(inv_flipadst16_1d);
    t[2][DDT] = Some(inv_ddt16_1d);
    t[2][FLIPDDT] = Some(inv_flipddt16_1d);

    t[3][DCT] = Some(inv_dct32_1d);
    t[3][IDENTITY] = Some(inv_identity32_1d);

    t[4][DCT] = Some(inv_dct64_1d);
    t[4][IDENTITY] = Some(inv_identity64_1d);

    t
};

pub static FWD_TX1D_FNS: [[Option<Tx1dFn>; N_TX_1D_TYPES - 1]; N_TX_SIZES] = {
    const DCT: usize = 0;
    const IDENTITY: usize = 1;
    const ADST: usize = 2;
    const FLIPADST: usize = 3;
    const DDT: usize = 4;
    const FLIPDDT: usize = 5;
    const NONE: Option<Tx1dFn> = None;

    let mut t = [[NONE; N_TX_1D_TYPES - 1]; N_TX_SIZES];

    t[0][DCT] = Some(fwd_dct4_1d);
    t[0][IDENTITY] = Some(fwd_identity4_1d);
    t[0][ADST] = Some(fwd_adst4_1d);
    t[0][FLIPADST] = Some(fwd_flipadst4_1d);

    t[1][DCT] = Some(fwd_dct8_1d);
    t[1][IDENTITY] = Some(fwd_identity8_1d);
    t[1][ADST] = Some(fwd_adst8_1d);
    t[1][FLIPADST] = Some(fwd_flipadst8_1d);
    t[1][DDT] = Some(fwd_ddt8_1d);
    t[1][FLIPDDT] = Some(fwd_flipddt8_1d);

    t[2][DCT] = Some(fwd_dct16_1d);
    t[2][IDENTITY] = Some(fwd_identity16_1d);
    t[2][ADST] = Some(fwd_adst16_1d);
    t[2][FLIPADST] = Some(fwd_flipadst16_1d);
    t[2][DDT] = Some(fwd_ddt16_1d);
    t[2][FLIPDDT] = Some(fwd_flipddt16_1d);

    t[3][DCT] = Some(fwd_dct32_1d);
    t[3][IDENTITY] = Some(fwd_identity32_1d);

    t[4][DCT] = Some(fwd_dct64_1d);
    t[4][IDENTITY] = Some(fwd_identity64_1d);

    t
};

/// Forward 2D transform of spatial residual `res` (with row stride `res_stride`) into
/// column-major coefficient buffer `coeff_out` (`coeff_out[y + x * sh]`).
///
/// The output coefficients are scaled to the exact domain expected by [`inv_txfm_add`].
pub fn fwd_txfm_2d(res: &[i32], res_stride: usize, coeff_out: &mut [i32], txtp: u32, tx: usize) {
    if (txtp & 0xFF) == WHT_WHT {
        debug_assert_eq!(tx, 0);
        let mut out4 = [0i32; 16];
        fwd_wht_wht_4x4(res, res_stride, &mut out4);
        coeff_out[..16].copy_from_slice(&out4);
        return;
    }

    let t_dim = &TXFM_DIMENSIONS[tx];
    let tx_sh = &TX_SHIFT[tx];
    let w = 4 * t_dim.w as usize;
    let h = 4 * t_dim.h as usize;
    let sw = w.min(32);
    let sh = h.min(32);
    let is_rect2 = ((t_dim.lw + t_dim.lh) & 1) != 0;

    let mut tmp = [0i32; 32 * 32];

    // Step 1: Load spatial residual into sw x sh grid (averaging 2x1/1x2/2x2 if w > 32 or h > 32)
    let sx_step = w / sw;
    let sy_step = h / sh;
    let area = (sx_step * sy_step) as i32;
    let area_rnd = area >> 1;

    for y in 0..sh {
        for x in 0..sw {
            let mut sum = 0i32;
            for dy in 0..sy_step {
                for dx in 0..sx_step {
                    sum += res[(y * sy_step + dy) * res_stride + (x * sx_step + dx)];
                }
            }
            tmp[y * sw + x] = (sum + area_rnd) / area;
        }
    }

    let hor_fn = FWD_TX1D_FNS[t_dim.lw as usize][(txtp & 7) as usize].unwrap_or(fwd_dct4_1d);
    let ver_fn = FWD_TX1D_FNS[t_dim.lh as usize][((txtp >> 5) & 7) as usize].unwrap_or(fwd_dct4_1d);

    // Step 2: Vertical 1D forward transform along each column x in 0..sw
    for x in 0..sw {
        ver_fn(&mut tmp[x..], sw);
    }

    let l_sw = (t_dim.lw.min(3) + 2) as i32;
    let l_sh = (t_dim.lh.min(3) + 2) as i32;
    let total_shift = 24 + l_sw + l_sh - (tx_sh[0] as i32 + tx_sh[1] as i32) - (is_rect2 as i32);
    let shift0 = 6i32;
    let rnd0 = 1i32 << (shift0 - 1);
    for v in tmp[..sw * sh].iter_mut() {
        *v = (*v + rnd0) >> shift0;
    }

    // Step 3: Horizontal 1D forward transform along each row y in 0..sh
    for y in 0..sh {
        hor_fn(&mut tmp[y * sw..], 1);
    }

    // Step 4: Normalize and store in column-major order coeff_out[y + x * sh]
    let shift1 = total_shift - shift0;
    let rnd1 = if shift1 > 0 { 1i64 << (shift1 - 1) } else { 0 };

    for y in 0..sh {
        for x in 0..sw {
            let mut v = tmp[y * sw + x] as i64;
            if is_rect2 {
                v = (v * 181 + 128) >> 8;
            }
            let c = if shift1 >= 0 {
                ((v + rnd1) >> shift1) as i32
            } else {
                (v << (-shift1)) as i32
            };
            coeff_out[y + x * sh] = c;
        }
    }
}

/// Residual addition helper for `inv_txfm_add`.
pub fn residual_add<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    c: &[i32],
    w: usize,
    h: usize,
    rnd: i32,
    shift: i32,
    dpcm_flag: u8,
) {
    match dpcm_flag {
        1 => {
            let mut ci = 0;
            for y in 0..h {
                let mut acc = 0i32;
                for x in 0..w {
                    acc += (c[ci] + rnd) >> shift;
                    let p = dst[y * stride + x].into();
                    dst[y * stride + x] = bd.pixel_clip(p + acc);
                    ci += 1;
                }
            }
        }
        2 => {
            for x in 0..w {
                let mut acc = 0i32;
                for y in 0..h {
                    acc += (c[y * w + x] + rnd) >> shift;
                    let p = dst[y * stride + x].into();
                    dst[y * stride + x] = bd.pixel_clip(p + acc);
                }
            }
        }
        _ => {
            let max_val = bd.bitdepth_max();
            for y in 0..h {
                let row = y * stride;
                if row >= dst.len() {
                    break;
                }
                let cw = y * w;
                let n = w.min(dst.len() - row).min(c.len().saturating_sub(cw));
                BD::Pixel::add_residual_clip(
                    &mut dst[row..row + n],
                    &c[cw..cw + n],
                    rnd,
                    shift,
                    max_val,
                );
            }
        }
    }
}

/// Normative AV2 8-bit inverse transform + residual add.
#[inline]
pub fn inv_txfm_add_8bpc(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    coeff: &mut [i32],
    txtp: u32,
    eob: i32,
    tx: usize,
) {
    inv_txfm_add(BitDepth8, dst, dst_off, stride, coeff, txtp, eob, tx);
}

/// Normative AV2 inverse transform of `coeff` followed by clipped add into `dst`.
pub fn inv_txfm_add<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    coeff: &mut [i32],
    txtp: u32,
    eob: i32,
    tx: usize,
) {
    if eob < 0 {
        return;
    }

    if (txtp & 0xFF) == WHT_WHT {
        assert_eq!(tx, 0);
        let mut tmp = [0i32; 16];
        inv_wht_wht_4x4(&coeff[..16].try_into().unwrap(), &mut tmp);
        coeff[..16].fill(0);
        let dpcm_flag = (txtp >> 8) as u8;
        residual_add(bd, &mut dst[dst_off..], stride, &tmp, 4, 4, 0, 0, dpcm_flag);
        return;
    }

    let t_dim = &TXFM_DIMENSIONS[tx];
    let tx_sh = &TX_SHIFT[tx];
    let w = 4 * t_dim.w as usize;
    let h = 4 * t_dim.h as usize;
    let is_rect2 = ((t_dim.lw + t_dim.lh) & 1) != 0;

    if eob + txtp as i32 == 0 {
        let shift_p1 = tx_sh[0] as i32;
        let shift = shift_p1 + tx_sh[1] as i32 - 12;
        let rnd = (1 << (shift - 1)) + shift_p1 - 6;
        let mut dc = coeff[0];
        coeff[0] = 0;
        if is_rect2 {
            dc = (dc * 181 + 128) >> 8;
        }
        dc = (dc + rnd) >> shift;
        let max_val = bd.bitdepth_max();
        for y in 0..h {
            let row = dst_off + y * stride;
            if row >= dst.len() {
                break;
            }
            let n = w.min(dst.len() - row);
            BD::Pixel::add_dc_clip(&mut dst[row..row + n], dc, max_val);
        }
        return;
    }

    let first_1d_fn = INV_TX1D_FNS[t_dim.lw as usize][(txtp & 7) as usize].unwrap();
    let second_1d_fn = INV_TX1D_FNS[t_dim.lh as usize][((txtp >> 5) & 7) as usize].unwrap();
    let sh = imin(h as i32, 32) as usize;
    let sw = imin(w as i32, 32) as usize;
    let (row_clip_min, row_clip_max) = if BD::BPC == 8 {
        (i16::MIN as i32, i16::MAX as i32)
    } else {
        let min = ((!bd.bitdepth_max() as u32) << 7) as i32;
        (min, !min)
    };

    let mut tmp = [0i32; 32 * 32];
    let mut col = 0usize;
    let tx_class = (txtp >> 3) & 0x3;
    let shift0 = tx_sh[0] as i32;
    let rnd0 = (1 << shift0) >> 1;

    if tx_class == 0 {
        let max_c = sh.min(eob as usize + 1);
        let max_x = sw.min(eob as usize + 1);
        let mut max_nz_col = 0usize;
        let mut max_nz_x = 0usize;
        while col < max_c {
            let mut any_nz = false;
            let mut row_max_x = 0usize;
            let row_off = col * sw;
            for x in 0..max_x {
                let idx = col + x * sh;
                let v = coeff[idx];
                if v != 0 {
                    coeff[idx] = 0;
                    any_nz = true;
                    row_max_x = x;
                    tmp[row_off + x] = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
                }
            }
            if any_nz {
                max_nz_col = col;
                max_nz_x = max_nz_x.max(row_max_x);
                if row_max_x == 0 && (txtp & 3) == 0 {
                    let dc = iclip(
                        (tmp[row_off] * 64 + rnd0) >> shift0,
                        row_clip_min,
                        row_clip_max,
                    );
                    tmp[row_off..row_off + sw].fill(dc);
                } else {
                    first_1d_fn(&mut tmp[row_off..], 1);
                    for v in tmp[row_off..row_off + sw].iter_mut() {
                        *v = iclip((*v + rnd0) >> shift0, row_clip_min, row_clip_max);
                    }
                }
            }
            col += 1;
        }
        if max_nz_col == 0 && txtp == 0 {
            let shift1 = tx_sh[1] as i32;
            let rnd1 = (1 << shift1) >> 1;
            let max_val = bd.bitdepth_max();
            for v in tmp[..sw].iter_mut() {
                *v *= 64;
            }
            let c_row = &tmp[..sw];
            if w > sw {
                for y in 0..h {
                    let d = dst_off + y * stride;
                    BD::Pixel::add_residual_upsample2x_row_clip(
                        &mut dst[d..d + w],
                        c_row,
                        rnd1,
                        shift1,
                        max_val,
                    );
                }
            } else {
                for y in 0..h {
                    let d = dst_off + y * stride;
                    BD::Pixel::add_residual_clip(&mut dst[d..d + w], c_row, rnd1, shift1, max_val);
                }
            }
            return;
        }
        if max_nz_x == 0 && txtp == 0 {
            let shift1 = tx_sh[1] as i32;
            let rnd1 = (1 << shift1) >> 1;
            let max_val = bd.bitdepth_max();
            second_1d_fn(&mut tmp[0..], sw);
            if h > sh {
                for y in (0..h).step_by(2) {
                    let dy = (tmp[(y / 2) * sw] + rnd1) >> shift1;
                    let d0 = dst_off + y * stride;
                    let d1 = dst_off + (y + 1) * stride;
                    BD::Pixel::add_dc_clip(&mut dst[d0..d0 + w], dy, max_val);
                    BD::Pixel::add_dc_clip(&mut dst[d1..d1 + w], dy, max_val);
                }
            } else {
                for y in 0..h {
                    let dy = (tmp[y * sw] + rnd1) >> shift1;
                    let d = dst_off + y * stride;
                    BD::Pixel::add_dc_clip(&mut dst[d..d + w], dy, max_val);
                }
            }
            return;
        }
    } else {
        let last_nz_col = if tx_class == 2 {
            imin(sh as i32 - 1, eob) as usize
        } else if tx_class == 3 {
            (eob as usize) >> (t_dim.lw as usize + 2)
        } else {
            sh - 1
        };
        loop {
            let row_off = col * sw;
            for x in 0..sw {
                let idx = col + x * sh;
                let v = coeff[idx];
                if v != 0 {
                    coeff[idx] = 0;
                    tmp[row_off + x] = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
                }
            }
            first_1d_fn(&mut tmp[row_off..], 1);
            for v in tmp[row_off..row_off + sw].iter_mut() {
                *v = iclip((*v + rnd0) >> shift0, row_clip_min, row_clip_max);
            }
            col += 1;
            if col > last_nz_col {
                break;
            }
        }
    }

    for x in 0..sw {
        second_1d_fn(&mut tmp[x..], sw);
    }

    let shift1 = tx_sh[1] as i32;
    let rnd1 = (1 << shift1) >> 1;
    let max_val = bd.bitdepth_max();

    if w > sw {
        if h > sh {
            for y in (0..h).step_by(2) {
                let c_row = &tmp[(y / 2) * sw..(y / 2 + 1) * sw];
                let d0 = dst_off + y * stride;
                let d1 = dst_off + (y + 1) * stride;
                BD::Pixel::add_residual_upsample2x_row_clip(
                    &mut dst[d0..d0 + w],
                    c_row,
                    rnd1,
                    shift1,
                    max_val,
                );
                BD::Pixel::add_residual_upsample2x_row_clip(
                    &mut dst[d1..d1 + w],
                    c_row,
                    rnd1,
                    shift1,
                    max_val,
                );
            }
        } else {
            for y in 0..h {
                let c_row = &tmp[y * sw..(y + 1) * sw];
                let d = dst_off + y * stride;
                BD::Pixel::add_residual_upsample2x_row_clip(
                    &mut dst[d..d + w],
                    c_row,
                    rnd1,
                    shift1,
                    max_val,
                );
            }
        }
    } else if h > sh {
        for y in (0..h).step_by(2) {
            let c_row = &tmp[(y / 2) * sw..(y / 2 + 1) * sw];
            let d0 = dst_off + y * stride;
            let d1 = dst_off + (y + 1) * stride;
            BD::Pixel::add_residual_clip(&mut dst[d0..d0 + w], c_row, rnd1, shift1, max_val);
            BD::Pixel::add_residual_clip(&mut dst[d1..d1 + w], c_row, rnd1, shift1, max_val);
        }
    } else {
        let dpcm_flag = (txtp >> 8) as u8;
        residual_add(
            bd,
            &mut dst[dst_off..],
            stride,
            &tmp,
            w,
            h,
            rnd1,
            shift1,
            dpcm_flag,
        );
    }
}

/// Derives the chroma plane transform type from the seed type selected by the
/// intra chroma prediction mode, applying the same restrictions as the
/// reference decoder (`compute_tx_type`): large transform dimensions only allow
/// the DCT kernel in the corresponding direction, and 16x16 forbids the
/// mixed directional combinations.
pub fn derive_chroma_txtp(seed: u32, t_dim: &TxfmInfo, tx: usize) -> u32 {
    let t = seed;
    if (t_dim.w >= 8 && (t & 0x02) != 0)
        || (t_dim.h >= 8 && (t & 0x40) != 0)
        || (tx == 2 && ((t & 0x47) == 0x41 || (t & 0xe2) == 0x22))
    {
        txtp::DCT_DCT
    } else if t == txtp::IDTX_INV {
        txtp::IDTX
    } else {
        t
    }
}

/// Picks the transform kernel with the best coefficient compaction from the
/// transform types that the block's transform set can actually signal.
///
/// The score is `sum(|cf[i]| * (8 + i/4))`, a cheap proxy for the coefficient
/// rate, plus a small penalty for non-default kernels so ties stay on
/// `DCT_DCT`. Returns the packed kernel id; the caller derives the signalled
/// symbols with [`intra_tx_type_choice`] / [`inter_tx_type_choice`].
pub fn select_best_tx_type(
    res: &[i32],
    res_stride: usize,
    tx: usize,
    choices: &[TxTypeChoice],
    speed: u8,
) -> u32 {
    if choices.len() <= 1 {
        return choices.first().map_or(DCT_DCT, |c| c.txtp);
    }
    let t_dim = &TXFM_DIMENSIONS[tx];
    let w = 4 * t_dim.w as usize;
    let h = 4 * t_dim.h as usize;
    let n_coeffs = (w.min(32)) * (h.min(32));
    let mut cf_buf = [0i32; 32 * 32];
    let mut best_txtp = DCT_DCT;
    let mut best_cost = u64::MAX;
    // The compaction heuristic is only meaningful for the small transforms the
    // intra/inter sets cover; keep the default above that and at high speed.
    let evaluate = speed < 7 && w <= 16 && h <= 16;

    for choice in choices.iter() {
        if !evaluate {
            break;
        }
        fwd_txfm_2d(res, res_stride, &mut cf_buf, choice.txtp, tx);
        let mut l1_energy = 0u64;
        for i in 0..n_coeffs {
            l1_energy += (cf_buf[i].unsigned_abs() as u64) * (8 + (i as u64) / 4);
        }
        // Rate penalty for signalling a non-default transform kernel: the
        // symbol costs a couple of bits, which is small relative to the
        // coefficient-rate proxy but keeps ties on `DCT_DCT`.
        let penalty = if choice.txtp == DCT_DCT {
            0
        } else {
            l1_energy / 64 + 16
        };
        let cost = l1_energy + penalty;
        if cost < best_cost {
            best_cost = cost;
            best_txtp = choice.txtp;
        }
    }
    best_txtp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwd_inv_wht_lossless_roundtrip() {
        let mut res = [0i32; 16];
        for i in 0..16 {
            res[i] = (i as i32 * 7) - 45;
        }
        let mut coeff = [0i32; 16];
        fwd_txfm_2d(&res, 4, &mut coeff, WHT_WHT, 0);

        // The inverse WHT is fed the dequantized coefficients of AV2 Spec
        // v1.0.0 §7.14.4: `Round2(|q| * 64, 3) == 8 * q` at quantizer index 0.
        let mut dq: [i32; 16] = core::array::from_fn(|i| coeff[i] * LOSSLESS_DEQUANT_SCALE);
        let mut dst = [100u8; 16];
        inv_txfm_add_8bpc(&mut dst, 0, 4, &mut dq, WHT_WHT, 15, 0);
        for i in 0..16 {
            assert_eq!(dst[i] as i32, 100 + res[i]);
        }
    }

    #[test]
    fn test_tx_type_round_trips_through_the_packed_representation() {
        for (spec, &packed) in TX_TYPE_TO_TXTP.iter().enumerate() {
            assert_eq!(spec_tx_type(packed), Some(spec as u8));
        }
        // `V_*` / `H_*` are the only 1D classes (AV2 Spec v1.0.0 §5.20.7.30).
        assert!(is_2d_tx_class(txtp::DCT_DCT));
        assert!(is_2d_tx_class(txtp::ADST_ADST));
        assert!(is_2d_tx_class(txtp::IDTX));
        assert!(!is_2d_tx_class(txtp::V_DCT));
        assert!(!is_2d_tx_class(txtp::H_FLIPADST));
    }

    #[test]
    fn test_intra_tx_type_choices_follow_md_idx_to_type() {
        // 4x4 DC intra, `TX_SET_INTRA_2`: the table row is {0, 3} and both
        // entries are 2D kernels.
        let td = &TXFM_DIMENSIONS[0];
        let (choices, n) = intra_tx_type_choices(td, 0, 0, 0, 0, TxSet::Intra2);
        assert_eq!(n, 2);
        assert_eq!(choices[0].txtp, txtp::DCT_DCT);
        assert_eq!(choices[1].txtp, txtp::ADST_ADST);
        assert_eq!(choices[1].symbol, 1);

        // 8x8 directional mode 1 (V_PRED, 90 degrees) with `TX_SET_INTRA_1`:
        // `Md_Idx_To_Type[1][1] = {0, 3, 1, 2, 7, 6, 8}`; the 1D `H_*`/`V_*`
        // entries are filtered out for the encoder.
        let td8 = &TXFM_DIMENSIONS[1];
        let (choices, n) = intra_tx_type_choices(td8, 1, 1, 0, 0, TxSet::Intra1);
        assert_eq!(n, 7);
        assert_eq!(choices[0].txtp, txtp::DCT_DCT);
        assert_eq!(choices[1].txtp, txtp::ADST_ADST);
        assert_eq!(choices[2].txtp, txtp::ADST_DCT);
        assert_eq!(choices[3].txtp, txtp::DCT_ADST);
        assert_eq!(choices[4].txtp, txtp::ADST_FLIPADST);
        assert_eq!(choices[5].txtp, txtp::FLIPADST_FLIPADST);
        assert_eq!(choices[6].txtp, txtp::FLIPADST_ADST);
    }

    #[test]
    fn test_wide_angle_mapping_changes_the_mode() {
        // Tall 4x8 block, pAngle 45 degrees: maps to D203 with 225 degrees.
        assert_eq!(wide_angle_mapping(3, 4, 8, 45), (7, 225));
        // Wide 8x4 block, pAngle 225 degrees: maps to D45 with 45 degrees.
        assert_eq!(wide_angle_mapping(7, 8, 4, 225), (3, 45));
        // Square blocks are unchanged.
        assert_eq!(wide_angle_mapping(3, 8, 8, 45), (3, 45));
    }

    #[test]
    fn test_inter_tx_type_choices_match_the_inverse_sets() {
        let td = &TXFM_DIMENSIONS[1];
        let (choices, n) = inter_tx_type_choices(td, TxSet::DctIdtx);
        assert_eq!(n, 2);
        assert_eq!(choices[0].txtp, txtp::IDTX);
        assert_eq!(choices[1].txtp, txtp::DCT_DCT);

        let (choices, n) = inter_tx_type_choices(td, TxSet::Inter1);
        assert_eq!(n, 10);
        assert_eq!(choices[0].txtp, txtp::IDTX);
        assert_eq!(choices[0].symbol, 0);
        assert_eq!(choices[0].symbol2, 0);
        assert_eq!(choices[1].txtp, txtp::DCT_DCT);
        assert_eq!(choices[1].symbol, 0);
        assert_eq!(choices[1].symbol2, 7);
        assert_eq!(choices[2].txtp, txtp::ADST_DCT);
        assert_eq!(choices[2].symbol, 1);
        assert_eq!(choices[2].symbol2, 0);
    }

    #[test]
    fn test_fwd_inv_idtx_2d_roundtrip() {
        for (tx, w) in [(0usize, 4usize), (1, 8), (2, 16), (3, 32)] {
            let mut res = vec![0i32; w * w];
            for y in 0..w {
                for x in 0..w {
                    res[y * w + x] = ((x as i32 * 5) - (y as i32 * 3) + 7).clamp(-40, 40);
                }
            }
            let mut coeff = vec![0i32; w * w];
            fwd_txfm_2d(&res, w, &mut coeff, txtp::IDTX, tx);
            let mut dst = vec![128u8; w * w];
            inv_txfm_add_8bpc(
                &mut dst,
                0,
                w,
                &mut coeff,
                txtp::IDTX,
                (w * w - 1) as i32,
                tx,
            );
            let mut worst = 0i32;
            for i in 0..w * w {
                worst = worst.max((dst[i] as i32 - (128 + res[i])).abs());
            }
            assert!(worst <= 2, "tx {tx} w {w} worst {worst}");
        }
    }

    #[test]
    fn test_fwd_inv_dct_2d_roundtrip() {
        let mut res = [0i32; 64];
        for y in 0..8 {
            for x in 0..8 {
                res[y * 8 + x] = ((x as i32 - 3) * 6) + ((y as i32 - 4) * 4);
            }
        }
        let mut coeff = [0i32; 64];
        fwd_txfm_2d(&res, 8, &mut coeff, 0, 1);

        let mut dst = [128u8; 64];
        inv_txfm_add_8bpc(&mut dst, 0, 8, &mut coeff, 0, 63, 1);
        for i in 0..64 {
            let diff = (dst[i] as i32 - (128 + res[i])).abs();
            assert!(diff <= 2, "pixel {i} diff={diff}");
        }
    }
}
