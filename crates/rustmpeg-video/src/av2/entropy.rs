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

//! Normative AV2 Specification v1.0.0 §5.20 & §8.3 context-adaptive entropy coding:
//! - Superblock and coding-tree partition contexts (`BlockContext`, `get_partition_ctx`, `update_partition_ctx`)
//! - Joint intra luma mode coding (`encode_intra_y_mode`: `y_mode_set`, `y_mode_index`, `y_mode_offset`, `y_second_mode`)
//! - Intra chroma & CfL coding (`encode_intra_uv_mode`, `encode_cfl_alphas`, `encode_mrl_idx`, `encode_filter_intra_mode`)
//! - Inter mode, reference frame, compound type, DRL index, and subpel filter coding
//! - 2D transform coefficient token coding (`encode_coefs`) matching AV2 Spec v1.0.0 §5.20.7.27–28 (`coeffs` & `read_quant`)

use crate::av2::cdf::{CdfCoefContext, CdfModeContext};
use crate::av2::intops::imin;
use crate::av2::levels::{BlockSize, CompInterType, FilterIntraMode, InterPredMode, MvXY, txtp};
use crate::av2::msac::MsacEncoder;
use crate::av2::scan::SCANS;
use crate::av2::tables::{
    BLOCK_DIMENSIONS, DEFAULT_MODE_LIST_Y, REORDERED_DIR_Y_MODE, REORDERED_NONDIR_Y_MODE,
    SIZE_GROUP, TXFM_DIMENSIONS, TxfmInfo,
};

/// Spatial neighbor context arrays across a 128-pixel superblock edge (`32` 4-pixel units).
#[derive(Debug, Clone)]
pub struct BlockContext {
    pub intra: [u8; 32],
    pub mode: [u8; 32],
    /// Joint intra luma mode index (`IntraJointModes`, `0..=55`, `0xFF` when the
    /// neighbour is not intra directional), used by the intra mode reordering.
    pub midx: [u8; 32],
    pub uvmode: [u8; 32],
    pub partition: [u8; 32],
    pub skip: [u8; 32],
    pub lcoef: [u8; 32],
    pub ccoef: [[u8; 32]; 2],
    pub tx: [u8; 32],
    pub tx_luma: [u8; 32],
    pub tx_chroma: [u8; 32],
    pub txtp: [u16; 32],
    pub ref_frame: [[i8; 32]; 2],
    pub comp_type: [u8; 32],
    pub filter: [[u8; 32]; 2],
    pub mv: [[MvXY; 32]; 2],
    pub seg_id: [u8; 32],
}

impl Default for BlockContext {
    fn default() -> Self {
        Self {
            intra: [0; 32],
            mode: [0; 32],
            midx: [0xFF; 32],
            uvmode: [0; 32],
            partition: [0; 32],
            skip: [0; 32],
            lcoef: [0; 32],
            ccoef: [[0; 32]; 2],
            tx: [0; 32],
            tx_luma: [0; 32],
            tx_chroma: [0; 32],
            txtp: [0; 32],
            ref_frame: [[-1; 32]; 2],
            comp_type: [0xFF; 32],
            filter: [[0; 32]; 2],
            mv: [[MvXY::default(); 32]; 2],
            seg_id: [0; 32],
        }
    }
}

/// Resets `BlockContext` at the start of a tile column or superblock row.
pub fn reset_context(ctx: &mut BlockContext, keyframe: bool, is_tip_frame: bool) {
    let intra_default = (keyframe || is_tip_frame) as u8;
    ctx.intra.fill(intra_default);
    ctx.mode.fill(0);
    ctx.midx.fill(0xFF);
    ctx.uvmode.fill(0);
    // A zero partition mask means "neighbour at least as large as any block that
    // can query it", which is what the reference decoder uses outside the frame
    // and at tile/SB-row boundaries (AV2 Spec v1.0.0 §5.20.3.2 note on AvailU /
    // AvailL: an unavailable neighbour never counts as smaller).
    ctx.partition.fill(0);
    ctx.skip.fill(0);
    // A zero DC category is 1 (0x40), matching the coefficient `res_ctx`
    // representation (dc_category << 6 | cul_level).
    ctx.lcoef.fill(0x40);
    ctx.ccoef[0].fill(0x40);
    ctx.ccoef[1].fill(0x40);
    ctx.tx.fill(4);
    ctx.tx_luma.fill(4);
    ctx.tx_chroma.fill(4);
    ctx.txtp.fill(txtp::DCT_DCT as u16);
    // Inter neighbour context at frame/tile/SB-row boundaries: no reference
    // and NEARMV, matching the reference decoder's reset (mode 13, refs -1).
    ctx.ref_frame[0].fill(-1);
    ctx.ref_frame[1].fill(-1);
    ctx.comp_type.fill(0);
    if !keyframe {
        ctx.mode.fill(13);
        ctx.filter[0].fill(3);
        ctx.filter[1].fill(3);
    } else {
        ctx.filter[0].fill(0);
        ctx.filter[1].fill(0);
    }
    ctx.mv[0].fill(MvXY::default());
    ctx.mv[1].fill(MvXY::default());
    ctx.seg_id.fill(0);
}

/// Computes the two-bit partition context `left << 1 | above` from the above (`a`)
/// and left (`l`) neighbour masks for block size `bs` at `(bx4, by4)`.
///
/// This is the `ctx1 * 2 + ctx2` term of the `do_split`, `rect_type` and
/// `do_ext_partition` context derivations in AV2 Spec v1.0.0 §5.20.3.2:
/// bit 0 is 1 when the above neighbour is smaller than `bs` in width, bit 1 is 1
/// when the left neighbour is smaller than `bs` in height. Each stored mask byte
/// holds `~(dim4 - 1)` for the covered 4x4 units, so testing the bit at
/// `log2(dim4) - 1` compares the neighbour's dimension against the current one.
#[inline]
pub fn get_partition_ctx(
    a: &BlockContext,
    l: &BlockContext,
    bs: BlockSize,
    bx4: usize,
    by4: usize,
) -> usize {
    let b_dim = BLOCK_DIMENSIONS[bs as usize];
    let shift_a = (b_dim[2] as u32).saturating_sub(1) as u8;
    let shift_l = (b_dim[3] as u32).saturating_sub(1) as u8;
    let above = (a.partition[bx4 & 31] >> shift_a) & 1;
    let left = (l.partition[by4 & 31] >> shift_l) & 1;
    above as usize | ((left as usize) << 1)
}

/// Records the coded block `bs` at `(bx4, by4)` in the `a.partition` and
/// `l.partition` neighbour masks.
///
/// This is the `AboveMiSizes` / `LeftMiSizes` update of AV2 Spec v1.0.0 §5.20.4.1
/// (`decode_block`): only *coded blocks* update the neighbour context, so it must
/// be called for leaf blocks only. Updating it for a `H`, `V` or `Split` parent
/// instead would clobber the context that the child nodes read, so partition
/// nodes above a leaf never touch the masks.
///
/// Each covered 4x4 unit stores `~(dim4 - 1)`, which makes the bit at
/// `log2(dim4) - 1` compare a neighbour's dimension against the current block
/// (see [`get_partition_ctx`]).
pub fn update_partition_ctx(
    a: &mut BlockContext,
    l: &mut BlockContext,
    bs: BlockSize,
    bx4: usize,
    by4: usize,
) {
    let b_dim = BLOCK_DIMENSIONS[bs as usize];
    let bw4 = b_dim[0] as usize;
    let bh4 = b_dim[1] as usize;
    let a_mask = !(bw4 as u8 - 1);
    let l_mask = !(bh4 as u8 - 1);
    for i in 0..bw4.min(32 - (bx4 & 31)) {
        a.partition[(bx4 & 31) + i] = a_mask;
    }
    for i in 0..bh4.min(32 - (by4 & 31)) {
        l.partition[(by4 & 31) + i] = l_mask;
    }
}

/// Computes the transform-block `all_zero` (`txb_skip` / `v_txb_skip`) context
/// index per AV2 Spec §8.3.2.
///
/// The neighbouring level-context bytes use the layout
/// `dc_category << 6 | cul_level`, where `dc_category` is 0 for a negative DC,
/// 1 for a zero DC and 2 for a positive DC.
#[allow(clippy::too_many_arguments)]
pub fn get_skip_ctx(
    t_dim: &TxfmInfo,
    bs: usize,
    a: &[u8],
    l: &[u8],
    plane: i32,
    u_has_cf: i32,
    ss_hor: bool,
    ss_ver: bool,
) -> u32 {
    let b_dim = BLOCK_DIMENSIONS[bs];
    let bw4 = b_dim[0];
    let bh4 = b_dim[1];

    if plane != 0 {
        // `bw * bh > w * h` (AV2 Spec v1.0.0 §5.20.7.27, `all_zero`): the
        // plane residual block is larger than the transform block.
        let _ = (ss_hor, ss_ver);
        let bw = b_dim[0] as usize * 4;
        let bh = b_dim[1] as usize * 4;
        let not_one_blk = bw * bh > (t_dim.w as usize * 4) * (t_dim.h as usize * 4);
        let ca = a[..t_dim.w as usize].iter().any(|&v| v != 0x40);
        let cl = l[..t_dim.h as usize].iter().any(|&v| v != 0x40);
        let offset = if plane == 1 {
            6
        } else {
            6 * u_has_cf + not_one_blk as i32 * 3
        } as u32;
        offset + ca as u32 + cl as u32
    } else if bw4 == t_dim.w && bh4 == t_dim.h {
        0
    } else {
        let mut la = 0u8;
        let mut ll = 0u8;
        for &v in &a[..t_dim.w as usize] {
            la |= v;
        }
        for &v in &l[..t_dim.h as usize] {
            ll |= v;
        }
        ((la & 0x3F).min(4) as u32 + (ll & 0x3F).min(4) as u32 + 3) >> 1
    }
}

/// Computes the DC sign context (`0`, `1`, or `2`) from above (`a`) and left (`l`)
/// sign categories per AV2 Spec §8.3.2.
pub fn get_dc_sign_ctx(t_dim: &TxfmInfo, a: &[u8], l: &[u8]) -> u32 {
    let mut balance = 0i32;
    for &v in &a[..t_dim.w as usize] {
        match v >> 6 {
            0 => balance -= 1,
            2 => balance += 1,
            _ => {}
        }
    }
    for &v in &l[..t_dim.h as usize] {
        match v >> 6 {
            0 => balance -= 1,
            2 => balance += 1,
            _ => {}
        }
    }
    if balance == 0 {
        0
    } else if balance < 0 {
        1
    } else {
        2
    }
}

/// Computes the base-token context and high-range (BR) context for the current
/// scan position from the already-coded neighbouring base tokens.
///
/// `off` is the position of the current coefficient in the `levels` scratch
/// array, `xy` is `row + col` for the 2D transform class and `row` for the
/// 1-D classes, and `stride` is the distance between two columns.
pub fn get_lo_ctx(
    levels: &[i8],
    off: usize,
    tx_class: u8,
    hi_mag: &mut u32,
    xy: u32,
    plane: i32,
    stride: usize,
) -> u32 {
    let chroma = plane != 0;
    let lo_freq = xy
        < if chroma {
            1
        } else if tx_class == 0 {
            4
        } else {
            2
        };
    let mut lim: u32 = if lo_freq { 5 } else { 3 };
    let mut lo_mag: u32 = 0;
    let mut hi: u32 = 0;

    #[inline]
    fn add(v: i8, lim: u32, lo_mag: &mut u32, hi: &mut u32) {
        let val = v as u8 as u32;
        *lo_mag += val.min(lim);
        *hi += val.min(5);
    }

    add(levels[off + 1], lim, &mut lo_mag, &mut hi);
    add(levels[off + stride], lim, &mut lo_mag, &mut hi);

    let offset: u32;
    if tx_class == 0 {
        add(levels[off + stride + 1], lim, &mut lo_mag, &mut hi);
        if !chroma {
            lo_mag += (levels[off + 2] as u8 as u32).min(lim)
                + (levels[off + 2 * stride] as u8 as u32).min(lim);
            if lo_freq {
                offset = if xy == 0 {
                    0
                } else if xy < 2 {
                    9
                } else {
                    16
                };
                lim = if xy == 0 {
                    8
                } else if xy < 2 {
                    6
                } else {
                    4
                };
            } else {
                offset = if xy < 6 {
                    0
                } else if xy < 8 {
                    5
                } else {
                    10
                };
                lim = 4;
            }
        } else {
            lim = 3;
            offset = if plane == 1 { 0 } else { 4 };
        }
    } else {
        if !chroma {
            lim = 3;
            add(levels[off + 2], lim, &mut lo_mag, &mut hi);
            lo_mag += (levels[off + 3] as u8 as u32).min(3) + (levels[off + 4] as u8 as u32).min(3);
            if lo_freq {
                offset = if xy == 0 { 21 } else { 28 };
                lim = if xy == 0 { 6 } else { 4 };
            } else {
                offset = 15;
                lim = 4;
            }
        } else {
            offset = 8;
            lim = 3;
        }
    }

    *hi_mag = (if !chroma && lo_freq && (xy > 0 || tx_class != 0) {
        7
    } else {
        0
    }) + ((hi + 1) >> 1).min(if chroma { 3 } else { 6 });
    offset + ((lo_mag + 1) >> 1).min(lim)
}

/// Builds the 5-element custom directional luma intra mode list from above/left neighbor modes.
pub fn build_custom_dir_mode_list(a_mode: u8, l_mode: u8) -> [u8; 5] {
    let a_dir = (1..=8).contains(&a_mode);
    let l_dir = (1..=8).contains(&l_mode);
    let mut list = [0u8; 5];
    let mut count = 0usize;

    let mut push_unique = |m: u8| {
        if count < 5 && (1..=8).contains(&m) && !list[..count].contains(&m) {
            list[count] = m;
            count += 1;
        }
    };

    if a_dir {
        push_unique(a_mode);
    }
    if l_dir {
        push_unique(l_mode);
    }
    for &m in &REORDERED_DIR_Y_MODE {
        push_unique(m);
    }
    let _ = DEFAULT_MODE_LIST_Y;
    list
}

/// Builds the `get_intra_y_mode_set` directional ordering for a block with the
/// given above/left neighbour joint mode indices (`0xFF` when absent) and
/// 4x4-unit dimensions (`bw4`, `bh4`), per AV2 Spec v1.0.0 §5.20.7.15.
///
/// The neighbour entries are the *left* joint mode first, then the above one,
/// followed by the `+/-1..4` angle neighbours of each and finally the remaining
/// `DEFAULT_MODE_LIST_Y` entries.
pub fn build_intra_y_mode_order(a_midx: u8, l_midx: u8, bw4: i32, bh4: i32) -> ([u8; 56], usize) {
    let mut is_selected = [false; 56];
    let mut ordered = [0u8; 56];
    let mut out_len = 0usize;

    if bw4 >= 2 && bh4 >= 2 {
        let mut nb_dirs = [0u8; 2];
        let mut nb_count = 0usize;
        for nb_midx in [l_midx, a_midx] {
            if nb_midx == 0xFF {
                continue;
            }
            let idx = nb_midx as usize;
            if idx >= 56 || is_selected[idx] {
                continue;
            }
            is_selected[idx] = true;
            ordered[out_len] = nb_midx;
            out_len += 1;
            nb_dirs[nb_count] = nb_midx;
            nb_count += 1;
        }
        if bw4 * bh4 > 4 {
            for step in 1..=4i32 {
                for j in 0..nb_count {
                    for sgn in [-1i32, 1i32] {
                        let cand = (nb_dirs[j] as i32 + step * sgn).rem_euclid(56) as usize;
                        if !is_selected[cand] {
                            is_selected[cand] = true;
                            ordered[out_len] = cand as u8;
                            out_len += 1;
                        }
                    }
                }
            }
        }
    }
    for &m in &DEFAULT_MODE_LIST_Y {
        let mu = m as usize;
        if mu < 56 && !is_selected[mu] {
            is_selected[mu] = true;
            ordered[out_len] = m;
            out_len += 1;
        }
    }
    (ordered, out_len)
}

/// Transform set selected by `get_tx_set` (AV2 Spec v1.0.0 §5.20.8.3).
///
/// Only the members reachable with `reduced_tx_set` 1-3 are named; the
/// unreduced sets (`TX_SET_INTER_1/2`, `TX_SET_INTRA_1`) are still represented
/// so the selector is total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxSet {
    /// `TX_SET_DCTONLY`: `DCT_DCT` is inferred, no symbol is coded.
    DctOnly,
    /// `TX_SET_WIDE_64`: wide transform whose long side is 64 samples.
    Wide64,
    /// `TX_SET_HIGH_64`: tall transform whose long side is 64 samples.
    High64,
    /// `TX_SET_WIDE_32`: wide transform whose long side is 32 samples.
    Wide32,
    /// `TX_SET_HIGH_32`: tall transform whose long side is 32 samples.
    High32,
    /// `TX_SET_INTER_1`.
    Inter1,
    /// `TX_SET_INTER_2`.
    Inter2,
    /// `TX_SET_DCT_IDTX`: a single DCT/IDTX flag.
    DctIdtx,
    /// `TX_SET_DCT_IDTX_IDDCT`.
    DctIdtxIddct,
    /// `TX_SET_INTRA_1`.
    Intra1,
    /// `TX_SET_INTRA_2`.
    Intra2,
}

/// `get_tx_set(txSz, plane)` (AV2 Spec v1.0.0 §5.20.8.3).
///
/// `reduced_tx_set` is the frame-level `reduced_tx_set` for luma and
/// `enable_chroma_dctonly` for chroma, exactly as the specification's
/// `reducedTxSet` local.
pub fn get_tx_set(t_dim: &TxfmInfo, is_inter: bool, reduced_tx_set: i32) -> TxSet {
    use TxSet::*;
    // `Tx_Size_Sqr` / `Tx_Size_Sqr_Up` are the shorter and longer side logs.
    let sqr = t_dim.min as i32;
    let sqr_up = t_dim.max as i32;
    let wide = t_dim.w > t_dim.h;
    // TX_32X32 is 3 in the square-size enum, TX_64X64 is 4.
    if sqr_up > 3 {
        if sqr >= 3 {
            return DctOnly;
        }
        return if wide { Wide64 } else { High64 };
    }
    if sqr_up == 3 && sqr != 3 {
        return if wide { Wide32 } else { High32 };
    }
    if !is_inter && sqr_up == 3 {
        return DctOnly;
    }
    if sqr_up == 3 || reduced_tx_set == 1 {
        return if is_inter { DctIdtx } else { Intra2 };
    }
    if reduced_tx_set == 2 {
        return DctIdtx;
    }
    if reduced_tx_set == 3 {
        return if is_inter { DctIdtxIddct } else { Intra2 };
    }
    if is_inter {
        return if sqr == 2 { Inter2 } else { Inter1 };
    }
    Intra1
}

/// Whether `transform_type` codes a transform type for this block
/// (AV2 Spec v1.0.0 §5.20.8.2).
pub fn tx_type_is_coded(set: TxSet, is_inter: bool, lossless: bool, reduced_tx_set: i32) -> bool {
    set != TxSet::DctOnly && !lossless && !(reduced_tx_set == 2 && !is_inter)
}

/// Encodes `lossless_tx_size` for one lossless coding block
/// (AV2 Spec v1.0.0 §5.20.6.1).
///
/// The symbol is present when `MiSize != BLOCK_4X4`, the block is inter (or an
/// intra block uses `fsc_mode`), and the block carries a residual. `1` selects
/// the single full-block transform; `0` keeps `TX_4X4` for every unit.
pub fn encode_lossless_tx_size(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    bs: BlockSize,
    full_size: bool,
    is_inter: usize,
) {
    if bs == BlockSize::Bs4x4 {
        return;
    }
    let group = crate::av2::frame::SPEC_SIZE_GROUP[crate::av2::frame::mi_size_index(bs)] as usize;
    msac.encode_bool_adapt(full_size as u32, cdf_m.lossless_tx_size(group, is_inter));
}

/// Context and square-size index (`ctx`, `sqrSz`) of the `inter_tx_type` symbol
/// (AV2 Spec v1.0.0 §8 Table 8.3).
///
/// `eob` is the last non-zero coefficient in scan order, which is the
/// specification's `eob - 1`.
#[inline]
pub fn inter_txtp_ctx(t_dim: &TxfmInfo, eob: i32) -> (usize, usize) {
    // `bwl = Min(Tx_Width_Log2[txSz], 5)`.
    let bwl = (t_dim.lw as i32 + 2).min(5);
    let eob = eob.max(0);
    let eoby = eob >> bwl;
    let eobx = eob - (eoby << bwl);
    let diag = eobx + eoby;
    // `Min(Tx_Width, 32) + Min(Tx_Height, 32) - 4`, in samples.
    let limit = 4 * (t_dim.w.min(8) as i32 + t_dim.h.min(8) as i32) - 4;
    let ctx = if diag < 2 {
        1
    } else if diag > limit {
        2
    } else {
        0
    };
    (ctx, t_dim.min as usize)
}

/// Encodes the transform type of a block whose coefficients end at `eob`
/// (AV2 Spec v1.0.0 §5.20.8.2 `transform_type`).
///
/// `set` is `get_tx_set` for the block; `choice` carries the `TxType` symbol(s)
/// derived from the selected kernel with [`crate::av2::transform::intra_tx_type_choice`]
/// or [`crate::av2::transform::inter_tx_type_choice`]. The long sets code
/// `is_long_side_dct` for a 32-sample long side (the 64-sample sets infer 1);
/// `TX_SET_INTER_1` / `TX_SET_INTER_2` code a second offset symbol; and
/// `TX_SET_DCTONLY` codes nothing.
pub fn encode_tx_type_choice(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    t_dim: &TxfmInfo,
    set: TxSet,
    is_inter: bool,
    eob: i32,
    choice: &crate::av2::transform::TxTypeChoice,
) {
    use crate::av2::transform::TxTypeChoice;
    let TxTypeChoice {
        is_long_side_dct,
        symbol,
        symbol2,
        ..
    } = *choice;
    let (ctx, sqr) = inter_txtp_ctx(t_dim, eob);
    let sqr_i = (t_dim.min as usize).min(3);
    let inter = is_inter as usize;
    match set {
        TxSet::DctOnly => {}
        TxSet::Wide64 | TxSet::High64 => {
            if is_inter {
                msac.encode_symbol_adapt(symbol as usize, cdf_m.inter_tx_type_long(ctx, sqr), 3);
            } else {
                msac.encode_symbol_adapt(symbol as usize, &mut cdf_m.intra_tx_type_long[sqr_i], 3);
            }
        }
        TxSet::Wide32 | TxSet::High32 => {
            msac.encode_bool_adapt(is_long_side_dct as u32, cdf_m.is_long_side_dct(inter));
            if is_inter {
                msac.encode_symbol_adapt(symbol as usize, cdf_m.inter_tx_type_long(ctx, sqr), 3);
            } else {
                msac.encode_symbol_adapt(symbol as usize, &mut cdf_m.intra_tx_type_long[sqr_i], 3);
            }
        }
        TxSet::Intra1 => {
            msac.encode_symbol_adapt(symbol as usize, &mut cdf_m.intra_tx_type_set1[sqr_i], 6);
        }
        TxSet::Intra2 => {
            msac.encode_symbol_adapt(symbol as usize, &mut cdf_m.intra_tx_type_set2[sqr_i], 1);
        }
        TxSet::DctIdtx => {
            if is_inter {
                msac.encode_symbol_adapt(
                    symbol as usize,
                    &mut cdf_m.inter_tx_type_set3[ctx][sqr_i],
                    1,
                );
            } else {
                msac.encode_symbol_adapt(symbol as usize, &mut cdf_m.intra_tx_type_set2[sqr_i], 1);
            }
        }
        TxSet::DctIdtxIddct => {
            msac.encode_symbol_adapt(
                symbol as usize,
                &mut cdf_m.inter_tx_type_set4[ctx][sqr_i],
                3,
            );
        }
        TxSet::Inter1 => {
            msac.encode_symbol_adapt(
                symbol as usize,
                &mut cdf_m.inter_tx_type_set1[ctx][sqr_i.min(1)],
                1,
            );
            let cdf = if symbol == 0 {
                &mut cdf_m.inter_tx_type_index_set1[ctx]
            } else {
                &mut cdf_m.inter_tx_type_offset_set1[ctx]
            };
            msac.encode_symbol_adapt(symbol2 as usize, cdf, 7);
        }
        TxSet::Inter2 => {
            msac.encode_symbol_adapt(symbol as usize, &mut cdf_m.inter_tx_type_set2[ctx], 1);
            if symbol == 0 {
                msac.encode_symbol_adapt(
                    symbol2 as usize,
                    &mut cdf_m.inter_tx_type_index_set2[ctx],
                    7,
                );
            } else {
                msac.encode_symbol_adapt(
                    symbol2 as usize,
                    &mut cdf_m.inter_tx_type_offset_set2[ctx],
                    3,
                );
            }
        }
    }
}

/// Encodes joint luma intra prediction mode (`y_mode` + `y_angle`) per AV2 Specification v1.0.0 §5.20.7.5 & §5.20.7.15
/// (`y_mode_set`, `y_mode_index`, `y_mode_offset`, `y_second_mode`, `get_intra_y_mode_set`).
#[allow(clippy::too_many_arguments)]
pub fn encode_intra_y_mode(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    a: &BlockContext,
    l: &BlockContext,
    bx4: usize,
    by4: usize,
    bw4: i32,
    bh4: i32,
    w4: i32,
    h4: i32,
    seg_lossless: bool,
    y_mode: u8,
    y_angle: i8,
    dpcm: Option<bool>,
) -> u8 {
    if seg_lossless {
        // `use_dpcm_y` / `dpcm_mode_y` (AV2 Spec v1.0.0 §5.20.5.5). DPCM
        // replaces the mode syntax: `dpcm_mode_y == 0` means `V_PRED`, `1`
        // means `H_PRED`, `AngleDeltaY` and `mrl_index` become 0.
        msac.encode_bool_adapt(dpcm.is_some() as u32, cdf_m.dpcm(0));
        if let Some(horizontal) = dpcm {
            msac.encode_bool_adapt(horizontal as u32, cdf_m.dpcm_mode(0));
            // `IntraJointMode` for the neighbour context. The specification
            // stores `5 + joint index` (`50` for `H_PRED`, `22` for `V_PRED`,
            // AV2 Spec v1.0.0 §5.20.5.5); the context arrays here and in the
            // reference decoder use the joint index itself (`45` / `17`).
            return if horizontal { 45 } else { 17 };
        }
    }

    // `get_joint_mode(0)` is the left neighbour at the block's last row and
    // `get_joint_mode(1)` the above neighbour at its last column (AV2 Spec
    // v1.0.0 §5.20.7.15); a non-directional or absent neighbour contributes 0.
    let a_midx = a.midx[(bx4 + bw4 as usize - 1) & 31];
    let l_midx = l.midx[(by4 + bh4 as usize - 1) & 31];
    let a_is_dir = (a_midx != 0xFF) as usize;
    let l_is_dir = (l_midx != 0xFF) as usize;
    let ctx = a_is_dir + l_is_dir;

    // Invert get_intra_y_mode_set(modeIdx) from AV2 Spec v1.0.0 §5.20.7.15:
    // Non-directional modes map directly to mode_idx = 0..4;
    // Directional modes (56 total = 8 base directional modes * 7 angle deltas) map via
    // neighbor joint modes followed by DEFAULT_MODE_LIST_Y[56].
    let (mode_idx, midx) = if (1..=8).contains(&y_mode) {
        let reord_dir_idx = REORDERED_DIR_Y_MODE
            .iter()
            .position(|&m| m == y_mode)
            .unwrap_or(0);
        let angle_idx = ((y_angle + 3).clamp(0, 6)) as usize;
        let target_dir_mode = (reord_dir_idx * 7 + angle_idx) as u8;

        let (ordered, _out_len) = build_intra_y_mode_order(a_midx, l_midx, bw4, bh4);
        let rank = ordered
            .iter()
            .position(|&m| m == target_dir_mode)
            .unwrap_or(0);
        (5 + rank, target_dir_mode)
    } else {
        let non_dir_pos = REORDERED_NONDIR_Y_MODE
            .iter()
            .position(|&m| m == y_mode)
            .unwrap_or(0);
        (non_dir_pos.min(4), 0xFFu8)
    };

    let _ = (w4, h4, SIZE_GROUP[0]);

    // FIRST_MODE_COUNT = 13 (0..=12 coded via y_mode_set == 0; 13..=60 via y_mode_set in 1..=3 + L(4))
    if mode_idx < 13 {
        msac.encode_symbol_adapt(0, cdf_m.y_mode_set(), 3);
        if mode_idx < 7 {
            msac.encode_symbol_adapt(mode_idx, cdf_m.y_mode_index(ctx), 7);
        } else {
            msac.encode_symbol_adapt(7, cdf_m.y_mode_index(ctx), 7);
            msac.encode_symbol_adapt((mode_idx - 7).min(5), cdf_m.y_mode_offset(ctx), 5);
        }
    } else {
        let rem = mode_idx - 13;
        let y_mode_set = (1 + rem / 16).min(3);
        let y_second_mode = (rem % 16) as u32;
        msac.encode_symbol_adapt(y_mode_set, cdf_m.y_mode_set(), 3);
        msac.encode_bools_bypass(4, y_second_mode);
    }

    midx
}

/// Encodes Multiple Reference Line index (`mrl_idx = 0..=3`) for directional luma intra prediction
/// per AV2 Specification v1.0.0 §5.20.7.5.
pub fn encode_mrl_idx(msac: &mut MsacEncoder, cdf_m: &mut CdfModeContext, ctx: usize, mrl_idx: u8) {
    let idx = (mrl_idx as usize).min(3);
    msac.encode_symbol_adapt(idx, cdf_m.mrl_idx(ctx), 3);
    if idx > 0 {
        msac.encode_bool_adapt(0, cdf_m.mrl_sec_idx(ctx));
    }
}

/// Encodes DIP / Filter Intra enable flag, `dip_transpose L(1)`, and `dip_mode S()` when enabled
/// per AV2 Specification v1.0.0 §5.20.7.9 (`dip_mode_info`).
pub fn encode_filter_intra_mode(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    bs: BlockSize,
    mode: Option<FilterIntraMode>,
) {
    let enabled = mode.is_some() as u32;
    let ctx = (bs as usize).min(2);
    msac.encode_bool_adapt(enabled, cdf_m.use_dip(ctx));
    if let Some(m) = mode {
        msac.encode_bool_bypass(0); // dip_transpose L(1) = 0
        msac.encode_symbol_adapt((m as usize).min(5), cdf_m.dip_mode(), 5);
    }
}

/// Returns whether `uv_mode` can be represented through the default chroma
/// directional mode list when the luma mode is directional. The decoder's
/// index mapping skips one list position depending on the luma direction, so a
/// single directional chroma mode is unreachable for each luma direction.
pub fn chroma_uv_mode_codeable(uv_mode: u8, luma_midx: u8) -> bool {
    if !(1..=8).contains(&uv_mode) || luma_midx == 0xFF {
        return true;
    }
    const DEFAULT_MODE_LIST_UV: [u8; 8] = [1, 2, 3, 4, 8, 5, 6, 7];
    const INTRA_DIR_MODE_Y_TO_UV_IDX: [u8; 8] = [2, 4, 0, 5, 3, 6, 1, 7];
    let base = DEFAULT_MODE_LIST_UV
        .iter()
        .position(|&m| m == uv_mode)
        .unwrap_or(0);
    let luma_dir = (luma_midx / 7) as usize;
    base != INTRA_DIR_MODE_Y_TO_UV_IDX[luma_dir.min(7)] as usize
}

/// Encodes chroma intra prediction mode (`uv_mode`) per AV2 Specification
/// \u00a75.20.7.6 & \u00a75.20.7.16 (`is_cfl`, `uv_mode_cfl_not_allowed`,
/// `uv_mode_idx`, `get_intra_uv_mode_set`).
///
/// This encoder always codes directional chroma modes through the default
/// chroma mode list with a zero angle delta, so the reconstructed mode is
/// exactly `uv_mode` (it never uses the "same directional mode as luma" path).
#[allow(clippy::too_many_arguments)]
pub fn encode_intra_uv_mode(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    a: &BlockContext,
    l: &BlockContext,
    cbx4: usize,
    cby4: usize,
    cfl_allowed: bool,
    seg_lossless: bool,
    luma_midx: u8,
    uv_mode: u8,
    uv_angle: i8,
    dpcm: Option<bool>,
) {
    if seg_lossless {
        // `use_dpcm_uv` / `dpcm_mode_uv` (AV2 Spec v1.0.0 §5.20.5.6). The
        // symbol replaces the chroma mode syntax entirely.
        msac.encode_bool_adapt(dpcm.is_some() as u32, cdf_m.dpcm(1));
        if let Some(horizontal) = dpcm {
            msac.encode_bool_adapt(horizontal as u32, cdf_m.dpcm_mode(1));
            return;
        }
    }
    let _ = (a, l, cbx4, cby4, uv_angle);
    if cfl_allowed {
        let is_cfl = (uv_mode == 13) as u32;
        msac.encode_bool_adapt(is_cfl, cdf_m.is_cfl(0));
        if is_cfl != 0 {
            return;
        }
    }

    // Order of the inverse mapping must match the decoder's
    // `get_intra_uv_mode_set`: non-directional modes first, then the eight
    // default directional chroma modes.
    const DEFAULT_MODE_LIST_UV: [u8; 8] = [1, 2, 3, 4, 8, 5, 6, 7];
    const INTRA_DIR_MODE_Y_TO_UV_IDX: [u8; 8] = [2, 4, 0, 5, 3, 6, 1, 7];

    let uv_mode_ctx = (luma_midx != 0xFF) as usize;
    let uv_idx = if (1..=8).contains(&uv_mode) {
        let base = DEFAULT_MODE_LIST_UV
            .iter()
            .position(|&m| m == uv_mode)
            .unwrap_or(0);
        let mut idx = base;
        if uv_mode_ctx != 0 {
            let luma_dir = (luma_midx / 7) as usize;
            let threshold = INTRA_DIR_MODE_Y_TO_UV_IDX[luma_dir.min(7)] as usize;
            if idx >= threshold {
                idx -= 1;
            }
        }
        5 + uv_mode_ctx + idx
    } else {
        let pos = REORDERED_NONDIR_Y_MODE
            .iter()
            .position(|&m| m == uv_mode)
            .unwrap_or(0);
        uv_mode_ctx + pos
    }
    .min(12);

    if uv_idx < 7 {
        msac.encode_symbol_adapt(uv_idx, cdf_m.uv_mode_cfl_not_allowed(uv_mode_ctx), 7);
    } else {
        msac.encode_symbol_adapt(7, cdf_m.uv_mode_cfl_not_allowed(uv_mode_ctx), 7);
        msac.encode_bools_bypass(3, (uv_idx - 7) as u32);
    }
}

/// Encodes Chroma-from-Luma (`CfL`) scaling parameters `(alpha_u, alpha_v)`
/// according to AV2 Specification v1.0.0 §5.20.7.29 (`cfl_index`, `cfl_alpha_signs`, `cfl_alpha_u`, `cfl_alpha_v`).
pub fn encode_cfl_alphas(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    alpha_u: i8,
    alpha_v: i8,
) {
    // cfl_index = 0 (CFL_EXPLICIT)
    msac.encode_bool_adapt(0, cdf_m.cfl_index());

    let sign_u = if alpha_u == 0 {
        0usize
    } else if alpha_u < 0 {
        1usize
    } else {
        2usize
    };
    let sign_v = if alpha_v == 0 {
        0usize
    } else if alpha_v < 0 {
        1usize
    } else {
        2usize
    };
    let joint_sign = (sign_u * 3 + sign_v).saturating_sub(1).min(7);
    msac.encode_symbol_adapt(joint_sign, cdf_m.cfl_signs(), 7);

    if sign_u != 0 {
        let abs_u = (alpha_u.unsigned_abs() as usize).clamp(1, 8) - 1;
        let ctx_u = (sign_u - 1) * 3 + sign_v;
        msac.encode_symbol_adapt(abs_u, cdf_m.cfl_alpha(ctx_u.min(5)), 7);
    }
    if sign_v != 0 {
        let abs_v = (alpha_v.unsigned_abs() as usize).clamp(1, 8) - 1;
        let ctx_v = (sign_v - 1) * 3 + sign_u;
        msac.encode_symbol_adapt(abs_v, cdf_m.cfl_alpha(ctx_v.min(5)), 7);
    }
}

/// Encodes inter prediction mode, reference frame selection, compound type, DRL index, and subpel filter.
#[allow(clippy::too_many_arguments)]
pub fn encode_inter_block_mode(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    is_compound: bool,
    comp_type: CompInterType,
    ref_slot_idx: u8,
    inter_mode: InterPredMode,
    drl_idx: u8,
    interp_filter: u8,
) {
    msac.encode_bool_adapt(is_compound as u32, cdf_m.comp(0));
    if is_compound {
        msac.encode_bool_adapt(((comp_type as usize) & 1) as u32, cdf_m.comp_type());
        msac.encode_symbol_adapt(
            (inter_mode as usize).min(4),
            cdf_m.compound_mode_non_joint(0),
            4,
        );
    } else {
        msac.encode_bool_adapt((ref_slot_idx & 1) as u32, cdf_m.single_ref(0, 0));
        msac.encode_symbol_adapt((inter_mode as usize).min(2), cdf_m.single_mode(0), 2);
    }
    if matches!(inter_mode, InterPredMode::NearMv | InterPredMode::NewMv) {
        msac.encode_bool_adapt((drl_idx != 0) as u32, cdf_m.drl_bit(0, 0));
    }
    msac.encode_symbol_adapt((interp_filter as usize).min(2), cdf_m.interp_filter(0), 2);
}

/// Derives the 0..=3 context index for switchable subpel interpolation filter coding
/// (AV2 Spec v1.0.0 §5.20.7 / §8.3; dav2d `src/env.h:get_filter_ctx`).
///
/// This is the reduced two-neighbour form kept for the decoder half: it uses
/// the near corner only. The encoder uses
/// [`get_filter_ctx_for_block`], which reproduces the normative
/// `boff[0]`/`boff[1]` selection (`decode.c`).
pub fn get_filter_ctx(
    top_ctx: Option<&BlockContext>,
    left_ctx: Option<&BlockContext>,
    bx4_in_sb: usize,
    by4_in_sb: usize,
    ref_idx: i8,
) -> usize {
    get_filter_ctx_for_block(
        top_ctx,
        left_ctx,
        bx4_in_sb,
        by4_in_sb,
        1,
        1,
        top_ctx.is_some(),
        left_ctx.is_some(),
        false,
        ref_idx,
    )
}

/// Derives the switchable subpel filter context with the normative neighbour
/// selection of AV2 Spec v1.0.0 §8.3 / dav2d `src/env.h:get_filter_ctx` and
/// `decode.c` `boff[]`.
///
/// The first candidate is the far edge of the left block (`by4 + bh4 - 1`)
/// when the block spans its full partition height, otherwise its top edge;
/// the second is the far edge of the above block (`bx4 + bw4 - 1`), otherwise
/// its left edge. A missing candidate contributes type `3` (no filter).
#[allow(clippy::too_many_arguments)]
pub fn get_filter_ctx_for_block(
    top_ctx: Option<&BlockContext>,
    left_ctx: Option<&BlockContext>,
    bx4_in_sb: usize,
    by4_in_sb: usize,
    bw4: usize,
    bh4: usize,
    have_top_in_sb: bool,
    have_left: bool,
    is_compound: bool,
    ref_idx: i8,
) -> usize {
    let mut nb: [(Option<&BlockContext>, usize); 2] = [(None, 0); 2];
    let mut idx = 0usize;
    if have_left && bh4 >= 1 {
        nb[idx] = (left_ctx, by4_in_sb + bh4 - 1);
        idx += 1;
    }
    if have_top_in_sb && bw4 >= 1 && idx < 2 {
        nb[idx] = (top_ctx, bx4_in_sb + bw4 - 1);
        idx += 1;
    }
    if have_left && idx < 2 {
        nb[idx] = (left_ctx, by4_in_sb);
        idx += 1;
    }
    if have_top_in_sb && idx < 2 {
        nb[idx] = (top_ctx, bx4_in_sb);
    }

    let filter_type = |slot: (Option<&BlockContext>, usize)| -> usize {
        let (ctx, off) = slot;
        match ctx {
            Some(ctx) if ctx.ref_frame[0][off] == ref_idx || ctx.ref_frame[1][off] == ref_idx => {
                ctx.filter[0][off].min(3) as usize
            }
            _ => 3,
        }
    };
    let flt0 = filter_type(nb[0]);
    let flt1 = filter_type(nb[1]);
    let base = if is_compound { 4 } else { 0 };

    if flt0 == flt1 || flt1 == 3 {
        base + flt0.min(3)
    } else if flt0 == 3 {
        base + flt1.min(3)
    } else {
        base + 3
    }
}

/// Parameters for coding a 2D transform block's coefficients.
pub struct CoefCodingParams<'a> {
    pub tx: usize,
    pub bs: usize,
    pub plane: i32,
    pub intra: bool,
    pub fsc: bool,
    pub lossless: bool,
    pub y_mode: usize,
    /// Luma intra angle delta (`AngleDeltaY`), used by `Md_Idx_To_Type`.
    pub y_angle: i8,
    /// Luma intra multiple-reference-line index (`MrlIndex`).
    pub mrl_index: u8,
    pub uv_mode: usize,
    pub reduced_txtp_set: i32,
    pub u_has_cf: i32,
    pub dq_tbl: [u32; 2],
    pub bitdepth: u8,
    pub qm: Option<&'a [u8]>,
    pub ss_hor: bool,
    pub ss_ver: bool,
}

/// Encodes quantized transform coefficient levels `q_cf` (column-major, indexed
/// by scan position) into `msac` following the AV2 base-token coefficient
/// syntax (`coeffs` / `read_quant`).
///
/// Returns the last non-zero scan index (`eob`), or `-1` when `all_zero == 1`.
#[allow(clippy::too_many_arguments)]
pub fn encode_coefs(
    msac: &mut MsacEncoder,
    coef: &mut CdfCoefContext,
    mode: &mut CdfModeContext,
    a: &[u8],
    l: &[u8],
    p: &CoefCodingParams,
    q_cf: &[i32],
    txtp: &mut u16,
    res_ctx: &mut u8,
) -> i32 {
    let t_dim = &TXFM_DIMENSIONS[p.tx];
    let chroma = p.plane != 0;
    let slw = imin(t_dim.lw as i32, 3) as usize;
    let slh = imin(t_dim.lh as i32, 3) as usize;
    let tx2dszctx = slw + slh;
    let n_coeffs = 16usize << tx2dszctx;
    let scan = SCANS[p.tx];
    // Scan index below which the low-frequency token alphabet is used.
    let hi_to_low = if chroma { 1usize } else { 10usize };
    let stride = 4usize << slh;
    let shift = slh + 2;
    let mask = stride - 1;
    let tx_class = ((*txtp >> 3) & 3) as u8;

    // Last non-zero scan index (0 = DC only, -1 = all zero).
    let mut eob_idx: i32 = -1;
    for i in (0..n_coeffs).rev() {
        if q_cf[scan[i] as usize] != 0 {
            eob_idx = i as i32;
            break;
        }
    }

    let sctx = get_skip_ctx(t_dim, p.bs, a, l, p.plane, p.u_has_cf, p.ss_hor, p.ss_ver) as usize;
    let all_zero = (eob_idx < 0) as u32;
    if p.plane == 2 {
        msac.encode_bool_adapt(all_zero, coef.skip_v(sctx));
    } else {
        let is_inter_or_fsc = (!p.intra || p.fsc) as usize;
        msac.encode_bool_adapt(
            all_zero,
            coef.skip(is_inter_or_fsc, t_dim.ctx as usize, sctx),
        );
    }
    if all_zero != 0 {
        *res_ctx = 0x40;
        return -1;
    }

    // EOB bin: the coded value is the last non-zero scan index (0 = DC only).
    let eob_bin_val = 32 - (eob_idx as u32).leading_zeros();
    let eob_ctx = if chroma { 2usize } else { (!p.intra) as usize };
    match tx2dszctx {
        0 => msac.encode_symbol_adapt((eob_bin_val as usize).min(4), coef.eob_bin_16(eob_ctx), 4),
        1 => msac.encode_symbol_adapt((eob_bin_val as usize).min(5), coef.eob_bin_32(eob_ctx), 5),
        2 => msac.encode_symbol_adapt((eob_bin_val as usize).min(6), coef.eob_bin_64(eob_ctx), 6),
        3 => msac.encode_symbol_adapt((eob_bin_val as usize).min(7), coef.eob_bin_128(eob_ctx), 7),
        4 => {
            let sym = (eob_bin_val as usize).min(7);
            msac.encode_symbol_adapt(sym, coef.eob_bin_256(eob_ctx), 7);
            if sym == 7 {
                msac.encode_bool_bypass((eob_bin_val - 7) & 1);
            }
        }
        5 => {
            let sym = (eob_bin_val as usize).min(7);
            msac.encode_symbol_adapt(sym, coef.eob_bin_512(eob_ctx), 7);
            if sym == 7 {
                msac.encode_bools_bypass(2, eob_bin_val - 7);
            }
        }
        _ => {
            let sym = (eob_bin_val as usize).min(7);
            msac.encode_symbol_adapt(sym, coef.eob_bin_1024(eob_ctx), 7);
            if sym == 7 {
                msac.encode_bools_bypass(2, eob_bin_val - 7);
            }
        }
    }
    if eob_bin_val > 1 {
        let n = eob_bin_val - 2;
        let hi = (eob_idx as u32 >> n) & 1;
        msac.encode_bool_adapt(hi, coef.eob_hi_bit());
        for i in (0..n).rev() {
            msac.encode_bool_bypass((eob_idx as u32 >> i) & 1);
        }
    }

    // Transform type (AV2 Spec v1.0.0 §5.20.8.2 `transform_type`). Luma codes
    // a type from its transform set unless the set is `TX_SET_DCTONLY`, the
    // block is lossless, the intra block is DC-only (`eob == 1`), or the frame
    // uses `reduced_tx_set == 2` on an intra block. Chroma derives its type
    // from luma.
    //
    // Lossless luma inter blocks code `txtp_lossless` (WHT_WHT vs IDTX) when
    // the transform is 4x4 and infer `IDTX` above that; lossless chroma
    // inherits the masked luma kernel. Intra lossless blocks always use the
    // WHT and code no type.
    if p.lossless {
        if !chroma && !p.intra {
            if t_dim.max == 0 {
                let is_idtx = (*txtp & 0xFF) as u32 == txtp::IDTX;
                msac.encode_bool_adapt(is_idtx as u32, mode.lossless_inter_tx_type());
                *txtp = if is_idtx {
                    txtp::IDTX as u16
                } else {
                    crate::av2::transform::WHT_WHT as u16
                };
            } else {
                *txtp = txtp::IDTX as u16;
            }
        } else if chroma {
            // `IDTX_INV -> IDTX`; `WHT_WHT` and `IDTX` stay as inherited.
            *txtp &= 0x00E7;
        }
    } else if !chroma {
        let set = get_tx_set(t_dim, !p.intra, p.reduced_txtp_set);
        let dc_only = p.intra && eob_idx == 0;
        if tx_type_is_coded(set, !p.intra, p.lossless, p.reduced_txtp_set) && !dc_only {
            let choice = if p.intra {
                crate::av2::transform::intra_tx_type_choice(
                    t_dim,
                    p.tx,
                    p.y_mode as u8,
                    p.y_angle,
                    p.mrl_index,
                    set,
                    *txtp as u32,
                )
            } else {
                crate::av2::transform::inter_tx_type_choice(t_dim, set, *txtp as u32)
            };
            match choice {
                Some(choice) => {
                    encode_tx_type_choice(msac, mode, t_dim, set, !p.intra, eob_idx, &choice)
                }
                // The selected kernel is not representable in this block's set;
                // fall back to the inferred `DCT_DCT` so the stream stays valid.
                None => {
                    *txtp = txtp::DCT_DCT as u16;
                }
            }
        } else {
            *txtp = txtp::DCT_DCT as u16;
        }
    }

    let mut hr_avg: u32 = 0;
    let mut cul_level: u32 = 0;

    if eob_idx == 0 {
        // DC-only block: a dedicated low-frequency DC token.
        let level = q_cf[0].unsigned_abs();
        let max_br = if chroma { 5 } else { 8 };
        let token = if chroma {
            let tok = level.min(5);
            if tok >= 1 {
                msac.encode_symbol_adapt((tok - 1) as usize, coef.eob_base_uv_tok_lf(0), 4);
            }
            tok
        } else {
            if level <= 4 {
                msac.encode_symbol_adapt(
                    (level - 1) as usize,
                    coef.eob_base_y_tok_lf(t_dim.ctx as usize, 0),
                    4,
                );
                level
            } else {
                msac.encode_symbol_adapt(4, coef.eob_base_y_tok_lf(t_dim.ctx as usize, 0), 4);
                let br = (level - 5).min(3);
                msac.encode_symbol_adapt(br as usize, coef.br_y_tok_lf(0), 3);
                5 + br
            }
        };

        let sign = q_cf[0] < 0;
        let dc_sign_level = if sign { 0u32 } else { 2 << 6 };
        if chroma {
            msac.encode_bool_bypass(sign as u32);
        } else {
            let dc_ctx = get_dc_sign_ctx(t_dim, a, l) as usize;
            msac.encode_bool_adapt(sign as u32, coef.dc_sign(0, 0, dc_ctx));
        }
        let mut final_level = token;
        if token >= max_br {
            let hr = level - token;
            msac.encode_quant_hr(hr, 0, &mut hr_avg);
            final_level = token + hr;
        }
        cul_level = final_level;
        *res_ctx = (cul_level.min(63) | dc_sign_level) as u8;
        return 0;
    }

    let mut levels = [0i8; 1089];

    // EOB base token.
    let lf_eob = (eob_idx as usize) < hi_to_low;
    let ctx_init =
        1 + (eob_idx as usize > 2 << tx2dszctx) as u32 + (eob_idx as usize > 4 << tx2dszctx) as u32;
    {
        let eob_level = q_cf[scan[eob_idx as usize] as usize].unsigned_abs();
        let base_max = if lf_eob { 4 } else { 2 };
        let token = eob_level.min(if lf_eob { 8 } else { 6 });
        let (sym, br) = if token <= base_max {
            (token - 1, 0)
        } else {
            (base_max, token - base_max - 1)
        };
        if chroma {
            if lf_eob {
                msac.encode_symbol_adapt(
                    sym as usize,
                    coef.eob_base_uv_tok_lf(ctx_init as usize),
                    4,
                );
            } else {
                msac.encode_symbol_adapt(
                    sym as usize,
                    coef.eob_base_uv_tok_hf(ctx_init as usize),
                    2,
                );
                if sym == base_max {
                    msac.encode_symbol_adapt(br as usize, coef.br_uv_tok_hf(0), 3);
                }
            }
        } else if lf_eob {
            msac.encode_symbol_adapt(
                sym as usize,
                coef.eob_base_y_tok_lf(t_dim.ctx as usize, ctx_init as usize),
                4,
            );
            if sym == base_max {
                msac.encode_symbol_adapt(br as usize, coef.br_y_tok_lf(7), 3);
            }
        } else {
            msac.encode_symbol_adapt(
                sym as usize,
                coef.eob_base_y_tok_hf(t_dim.ctx as usize, ctx_init as usize),
                2,
            );
            if sym == base_max {
                msac.encode_symbol_adapt(br as usize, coef.br_y_tok_hf(0), 3);
            }
        }
        levels[scan[eob_idx as usize] as usize] = token as i8;
    }

    // AC base tokens (high to low scan order).
    for i in (1..eob_idx as usize).rev() {
        let rc = scan[i] as usize;
        let x = rc >> shift;
        let y = rc & mask;
        let lf = i < hi_to_low;
        let mut hr_ctx = 0u32;
        let ctx = get_lo_ctx(
            &levels,
            rc,
            tx_class,
            &mut hr_ctx,
            (x + y) as u32,
            p.plane,
            stride,
        );
        let level = q_cf[rc].unsigned_abs();
        let base_max = if lf { 4 } else { 2 };
        let token = level.min(if lf { 8 } else { 6 });
        let (sym, br) = if token <= base_max {
            (token, 0)
        } else {
            (base_max + 1, token - base_max - 1)
        };
        if chroma {
            if lf {
                msac.encode_symbol_adapt(sym as usize, coef.base_uv_tok_lf(ctx as usize), 5);
            } else {
                msac.encode_symbol_adapt(sym as usize, coef.base_uv_tok_hf(ctx as usize), 3);
                if sym == base_max + 1 {
                    msac.encode_symbol_adapt(br as usize, coef.br_uv_tok_hf(hr_ctx as usize), 3);
                }
            }
        } else if lf {
            msac.encode_symbol_adapt(
                sym as usize,
                coef.base_y_tok_lf(t_dim.ctx as usize, ctx as usize, 0),
                5,
            );
            if sym == base_max + 1 {
                msac.encode_symbol_adapt(br as usize, coef.br_y_tok_lf(hr_ctx as usize), 3);
            }
        } else {
            msac.encode_symbol_adapt(
                sym as usize,
                coef.base_y_tok_hf(t_dim.ctx as usize, ctx as usize, 0),
                3,
            );
            if sym == base_max + 1 {
                msac.encode_symbol_adapt(br as usize, coef.br_y_tok_hf(hr_ctx as usize), 3);
            }
        }
        levels[rc] = token as i8;
    }

    // DC base token. The DC always uses the low-frequency token alphabet; for
    // chroma there is no BR (high-range) CDF in that branch, so the token is
    // capped at the last base symbol (5).
    {
        let mut hr_ctx = 0u32;
        let ctx = get_lo_ctx(&levels, 0, tx_class, &mut hr_ctx, 0, p.plane, stride);
        let level = q_cf[0].unsigned_abs();
        let base_max = 4u32;
        let token = level.min(if chroma { 5 } else { 8 });
        let (sym, br) = if token <= base_max {
            (token, 0)
        } else {
            (base_max + 1, token - base_max - 1)
        };
        if chroma {
            msac.encode_symbol_adapt(sym as usize, coef.base_uv_tok_lf(ctx as usize), 5);
        } else {
            msac.encode_symbol_adapt(
                sym as usize,
                coef.base_y_tok_lf(t_dim.ctx as usize, ctx as usize, 0),
                5,
            );
            if sym == base_max + 1 {
                msac.encode_symbol_adapt(br as usize, coef.br_y_tok_lf(hr_ctx as usize), 3);
            }
        }
        levels[0] = token as i8;
    }

    // Signs and high-range residuals, high to low scan order.
    for i in (1..=eob_idx as usize).rev() {
        let rc = scan[i] as usize;
        let token = levels[rc] as u32;
        if token == 0 {
            continue;
        }
        let level = q_cf[rc].unsigned_abs();
        let sign = q_cf[rc] < 0;
        let y = rc & mask;
        // 2D class: all AC signs are bypass-coded.
        let _ = y;
        msac.encode_bool_bypass(sign as u32);
        let max_br = if i < hi_to_low {
            if chroma { 5 } else { 8 }
        } else {
            6
        };
        if token >= max_br {
            let hr = level - token;
            msac.encode_quant_hr(hr, 0, &mut hr_avg);
            cul_level += token + hr;
        } else {
            cul_level += token;
        }
    }

    // DC sign and residual.
    let dc_level = q_cf[0].unsigned_abs();
    let token = levels[0] as u32;
    let dc_sign_level: u32;
    if token != 0 {
        let sign = q_cf[0] < 0;
        dc_sign_level = if sign { 0 } else { 2 << 6 };
        if chroma {
            msac.encode_bool_bypass(sign as u32);
        } else {
            let dc_ctx = get_dc_sign_ctx(t_dim, a, l) as usize;
            msac.encode_bool_adapt(sign as u32, coef.dc_sign(0, 0, dc_ctx));
        }
        let max_br = if chroma { 5 } else { 8 };
        if token >= max_br {
            let hr = dc_level - token;
            msac.encode_quant_hr(hr, 0, &mut hr_avg);
            cul_level += token + hr;
        } else {
            cul_level += token;
        }
    } else {
        dc_sign_level = 1 << 6;
    }

    *res_ctx = (cul_level.min(63) | dc_sign_level) as u8;
    eob_idx
}

/// Neighbour context gathered for inter/intra mode decisions (AV2 §8.3.2).
///
/// The spec selects up to two spatial neighbours with a specific priority:
/// the left neighbour's bottom row, the above neighbour's rightmost column,
/// then the plain left/above entries. The same set drives the intra/inter flag,
/// the skip flag, reference-list selection, and the single-reference context.
#[derive(Debug, Clone, Copy, Default)]
pub struct InterNeighborCtx {
    /// Number of gathered neighbours (`0..=2`).
    pub n_ctx: usize,
    /// `(is_above, offset)` for each gathered neighbour.
    pub entries: [(bool, usize); 2],
    /// Skip flags of the gathered neighbours.
    pub skip: [u8; 2],
    /// Intra flags (excluding IntraBC) of the gathered neighbours.
    pub intra: [u8; 2],
    /// First reference slot of the gathered neighbours.
    pub ref0: [i8; 2],
    /// Second reference slot of the gathered neighbours.
    pub ref1: [i8; 2],
    /// Inter prediction modes of the gathered neighbours.
    pub mode: [u8; 2],
    /// Compound blending types of the gathered neighbours.
    pub comp_type: [u8; 2],
    /// Above neighbour `(ref0, ref1, mode)` at `bx4` when `have_top`.
    pub top: Option<(i8, i8, u8)>,
    /// Above-right neighbour `(ref0, ref1, mode)` at `bx4 + bw4 - 1` when available.
    pub top_right: Option<(i8, i8, u8)>,
    /// Left neighbour `(ref0, ref1, mode)` at `by4` when `have_left`.
    pub left: Option<(i8, i8, u8)>,
    /// Bottom-left neighbour `(ref0, ref1, mode)` at `by4 + bh4 - 1` when available.
    pub bottom_left: Option<(i8, i8, u8)>,
}

const NEWMV0_MODE_MASK: u32 =
    (1 << 15) | (1 << 20) | (1 << 22) | (1 << 23) | (1 << 26) | (1 << 27) | (1 << 28);
const NEWMV1_MODE_MASK: u32 = (1 << 19) | (1 << 22) | (1 << 25) | (1 << 27);

/// Gathers the normative inter/intra context neighbours for a block.
#[allow(clippy::too_many_arguments)]
pub fn gather_inter_neighbors(
    a: &BlockContext,
    l: &BlockContext,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
    have_left: bool,
    have_top: bool,
    tile_col_end4: usize,
    tile_row_end4: usize,
) -> InterNeighborCtx {
    let mut out = InterNeighborCtx::default();
    let mut idx = 0usize;

    if have_left && by4 + bh4 <= tile_row_end4 {
        let off = (by4 + bh4).saturating_sub(1) & 31;
        out.entries[idx] = (false, off);
        out.skip[idx] = l.skip[off];
        out.intra[idx] = (l.intra[off] != 0) as u8;
        out.ref0[idx] = l.ref_frame[0][off];
        out.ref1[idx] = l.ref_frame[1][off];
        out.mode[idx] = l.mode[off];
        out.comp_type[idx] = l.comp_type[off];
        idx += 1;
    }
    if have_top && bx4 + bw4 <= tile_col_end4 {
        let off = (bx4 + bw4).saturating_sub(1) & 31;
        out.entries[idx] = (true, off);
        out.skip[idx] = a.skip[off];
        out.intra[idx] = (a.intra[off] != 0) as u8;
        out.ref0[idx] = a.ref_frame[0][off];
        out.ref1[idx] = a.ref_frame[1][off];
        out.mode[idx] = a.mode[off];
        out.comp_type[idx] = a.comp_type[off];
        idx += 1;
    }
    if idx < 2 && have_left {
        let off = by4 & 31;
        out.entries[idx] = (false, off);
        out.skip[idx] = l.skip[off];
        out.intra[idx] = (l.intra[off] != 0) as u8;
        out.ref0[idx] = l.ref_frame[0][off];
        out.ref1[idx] = l.ref_frame[1][off];
        out.mode[idx] = l.mode[off];
        out.comp_type[idx] = l.comp_type[off];
        idx += 1;
    }
    if idx < 2 {
        let off = bx4 & 31;
        out.entries[idx] = (true, off);
        out.skip[idx] = a.skip[off];
        out.intra[idx] = (a.intra[off] != 0) as u8;
        out.ref0[idx] = a.ref_frame[0][off];
        out.ref1[idx] = a.ref_frame[1][off];
        out.mode[idx] = a.mode[off];
        out.comp_type[idx] = a.comp_type[off];
        if idx == 0 {
            out.skip[1] = out.skip[0];
            out.intra[1] = out.intra[0];
            out.ref0[1] = out.ref0[0];
            out.ref1[1] = out.ref1[0];
            out.mode[1] = out.mode[0];
            out.comp_type[1] = out.comp_type[0];
            out.entries[1] = out.entries[0];
        }
        idx = if have_top { idx + 1 } else { idx };
    }
    out.n_ctx = idx;

    if have_top {
        let off = bx4 & 31;
        out.top = Some((a.ref_frame[0][off], a.ref_frame[1][off], a.mode[off]));
        if bx4 + bw4 <= tile_col_end4 {
            let off_tr = (bx4 + bw4).saturating_sub(1) & 31;
            out.top_right = Some((
                a.ref_frame[0][off_tr],
                a.ref_frame[1][off_tr],
                a.mode[off_tr],
            ));
        }
    }
    if have_left {
        let off = by4 & 31;
        out.left = Some((l.ref_frame[0][off], l.ref_frame[1][off], l.mode[off]));
        if by4 + bh4 <= tile_row_end4 {
            let off_bl = (by4 + bh4).saturating_sub(1) & 31;
            out.bottom_left = Some((
                l.ref_frame[0][off_bl],
                l.ref_frame[1][off_bl],
                l.mode[off_bl],
            ));
        }
    }
    out
}

impl InterNeighborCtx {
    /// `get_intra_ctx`: 0..=3 context for the intra/inter flag.
    pub fn intra_ctx(&self) -> usize {
        if self.n_ctx == 0 {
            return 0;
        }
        let i = self.n_ctx - 1;
        let sum = self.intra[0] as usize + self.intra[i] as usize;
        sum + (sum == self.n_ctx) as usize
    }

    /// Skip-flag context (two gathered neighbours; skip mode is disabled here).
    pub fn skip_ctx(&self) -> usize {
        self.skip[0] as usize + self.skip[1] as usize
    }

    /// Reference-list selection context for reference-list index `ref_list_idx`.
    pub fn single_ref_ctx(&self, ref_list_idx: usize) -> usize {
        if self.n_ctx == 0 {
            return 1;
        }
        let mut cnt = [0i32; 9];
        for i in 0..self.n_ctx {
            cnt[(self.ref0[i].clamp(-1, 7) + 1) as usize] += 1;
            cnt[(self.ref1[i].clamp(-1, 7) + 1) as usize] += 1;
        }
        let mut cnt_rem = self.n_ctx as i32 * 2 - cnt[0] - cnt[8];
        let cnt_cur = cnt[(ref_list_idx + 1).min(8)];
        cnt_rem -= cnt_cur;
        (cnt_cur - cnt_rem + 1).clamp(0, 2) as usize
    }

    /// `get_snglref_ctx`: 0..=4 context for the single-reference mode symbol.
    pub fn snglref_ctx(&self, ref_idx: i8) -> usize {
        let mut row = 0i32;
        let mut col = 0i32;
        let mut newmv = 0i32;
        let add = |is_above: bool,
                   ref0: i8,
                   ref1: i8,
                   mode: u8,
                   row: &mut i32,
                   col: &mut i32,
                   newmv: &mut i32| {
            if ref0 == ref_idx {
                if is_above {
                    *col += 1;
                } else {
                    *row += 1;
                }
                *newmv += ((1u32 << mode.min(31)) & NEWMV0_MODE_MASK != 0) as i32;
            } else if ref1 == ref_idx {
                if is_above {
                    *col += 1;
                } else {
                    *row += 1;
                }
                *newmv += ((1u32 << mode.min(31)) & NEWMV1_MODE_MASK != 0) as i32;
            }
        };
        if let Some((r0, r1, m)) = self.top {
            add(true, r0, r1, m, &mut row, &mut col, &mut newmv);
        }
        if let Some((r0, r1, m)) = self.top_right {
            add(true, r0, r1, m, &mut row, &mut col, &mut newmv);
        }
        if let Some((r0, r1, m)) = self.left {
            add(false, r0, r1, m, &mut row, &mut col, &mut newmv);
        }
        if let Some((r0, r1, m)) = self.bottom_left {
            add(false, r0, r1, m, &mut row, &mut col, &mut newmv);
        }
        (row != 0) as usize + (col != 0) as usize + 2 * (newmv != 0) as usize
    }

    /// `get_comp_ctx`: 0..=4 context for the `comp` (is_compound) bit.
    pub fn comp_ctx(&self, refdir: &[i8]) -> usize {
        let is_future = |r: i8| -> bool {
            if r >= 0 && (r as usize) < refdir.len() {
                refdir[r as usize] == 1
            } else {
                false
            }
        };
        let is_intra_or_future = |r: i8| -> bool {
            if r == -1 {
                true
            } else if r >= 0 && (r as usize) < refdir.len() {
                refdir[r as usize] == 1
            } else {
                false
            }
        };

        match self.n_ctx {
            2 => {
                let refa2 = self.ref1[0];
                let refb2 = self.ref1[1];
                if refa2 == -1 {
                    let refa1 = self.ref0[0];
                    if refb2 == -1 {
                        let refb1 = self.ref0[1];
                        (is_future(refa1) ^ is_future(refb1)) as usize
                    } else {
                        2 + is_intra_or_future(refa1) as usize
                    }
                } else if refb2 == -1 {
                    let refb1 = self.ref0[1];
                    2 + is_intra_or_future(refb1) as usize
                } else {
                    4
                }
            }
            1 => {
                let ref2 = self.ref1[0];
                if ref2 == -1 {
                    let ref1 = self.ref0[0];
                    is_intra_or_future(ref1) as usize
                } else {
                    3
                }
            }
            _ => 1,
        }
    }

    /// `get_compref_ctx`: 0..=4 context for the compound mode symbol.
    pub fn compref_ctx(&self, ref0: i8, ref1: i8) -> usize {
        let mut row = 0i32;
        let mut col = 0i32;
        let mut newmv = 0i32;
        const NEWMV_MASK: u32 = (1 << 3)
            | (1 << 19)
            | (1 << 20)
            | (1 << 22)
            | (1 << 23)
            | (1 << 25)
            | (1 << 26)
            | (1 << 27)
            | (1 << 28);
        let mut add = |is_above: bool, r0: i8, r1: i8, mode: u8| {
            if r0 == ref0 && r1 == ref1 {
                if is_above {
                    col += 1;
                } else {
                    row += 1;
                }
                let is_newmv = (1u32 << (mode as usize).min(31)) & NEWMV_MASK != 0;
                newmv += is_newmv as i32;
            }
        };
        if let Some((r0, r1, m)) = self.top {
            add(true, r0, r1, m);
        }
        if let Some((r0, r1, m)) = self.top_right {
            add(true, r0, r1, m);
        }
        if let Some((r0, r1, m)) = self.left {
            add(false, r0, r1, m);
        }
        if let Some((r0, r1, m)) = self.bottom_left {
            add(false, r0, r1, m);
        }
        ((row != 0) as usize) + ((col != 0) as usize) + 2 * ((newmv != 0) as usize)
    }

    /// `comp_group_idx_ctx`: 0..=11 context for `comp_group_idx` (masked compound decision).
    pub fn comp_group_idx_ctx(&self, dist0: i32, dist1: i32, furthest_future: i8) -> usize {
        let mut ctx0 = 0usize;
        let mut ctx1 = 0usize;
        if self.n_ctx > 0 {
            if self.ref1[0] != -1 {
                ctx0 = (self.comp_type[0] > 1) as usize;
            } else if self.ref0[0] == furthest_future {
                ctx0 = 2;
            }
        }
        if self.n_ctx > 1 {
            if self.ref1[1] != -1 {
                ctx1 = (self.comp_type[1] > 1) as usize;
            } else if self.ref0[1] == furthest_future {
                ctx1 = 2;
            }
        }
        let offset = (dist0.abs() == dist1.abs()) as usize;
        ctx0 + ctx1 + ((ctx0 > 0 && ctx1 > 0) as usize) + offset * 6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tx(w4: u8, h4: u8) -> TxfmInfo {
        // Locate the rectangular transform size matching the requested extent.
        for t in &crate::av2::tables::TXFM_DIMENSIONS {
            if t.w == w4 && t.h == h4 {
                return *t;
            }
        }
        panic!("no transform size {w4}x{h4} in 4x4 units");
    }

    #[test]
    fn test_get_tx_set_matches_the_specification() {
        // AV2 Spec v1.0.0 §5.20.8.3 `get_tx_set`, reduced set 2.
        assert_eq!(get_tx_set(&tx(16, 16), true, 2), TxSet::DctOnly); // 64x64
        assert_eq!(get_tx_set(&tx(16, 8), true, 2), TxSet::DctOnly); // 64x32
        assert_eq!(get_tx_set(&tx(8, 16), true, 2), TxSet::DctOnly); // 32x64
        // Long 64-sample sides code `inter_tx_type` from the long table.
        assert_eq!(get_tx_set(&tx(16, 4), true, 2), TxSet::Wide64); // 64x16
        assert_eq!(get_tx_set(&tx(16, 2), true, 2), TxSet::Wide64); // 64x8
        assert_eq!(get_tx_set(&tx(4, 16), true, 2), TxSet::High64); // 16x64
        // Long 32-sample sides additionally code `is_long_side_dct`.
        assert_eq!(get_tx_set(&tx(8, 4), true, 2), TxSet::Wide32); // 32x16
        assert_eq!(get_tx_set(&tx(4, 8), true, 2), TxSet::High32); // 16x32
        // Reduced set 2 collapses everything else to the DCT/IDTX flag.
        assert_eq!(get_tx_set(&tx(8, 8), true, 2), TxSet::DctIdtx); // 32x32
        assert_eq!(get_tx_set(&tx(4, 4), true, 2), TxSet::DctIdtx); // 16x16
        assert_eq!(get_tx_set(&tx(1, 1), true, 2), TxSet::DctIdtx); // 4x4
        // Intra with the reduced set 2 codes nothing at any size.
        assert!(!tx_type_is_coded(
            get_tx_set(&tx(4, 4), false, 2),
            false,
            false,
            2
        ));
        // 32x32 intra is `TX_SET_DCTONLY` outright.
        assert_eq!(get_tx_set(&tx(8, 8), false, 0), TxSet::DctOnly);
        // Lossless never codes a type.
        assert!(!tx_type_is_coded(
            get_tx_set(&tx(16, 4), true, 2),
            true,
            true,
            2
        ));
    }

    #[test]
    fn test_inter_txtp_ctx_uses_the_last_scan_index() {
        // 64x16: `bwl = Min(6, 5) = 5`, so the context diagonal uses the low 5
        // bits of the last non-zero scan index (AV2 Spec v1.0.0 Table 8.3).
        let t = tx(16, 4);
        assert_eq!(inter_txtp_ctx(&t, 0).0, 1);
        assert_eq!(inter_txtp_ctx(&t, 1).0, 1);
        assert_eq!(inter_txtp_ctx(&t, 3).0, 0);
        // The square-size index is `Tx_Size_Sqr[txSz]` (16x16 -> 2).
        assert_eq!(inter_txtp_ctx(&t, 0).1, 2);
        assert_eq!(inter_txtp_ctx(&tx(8, 8), 0).1, 3);
        assert_eq!(inter_txtp_ctx(&tx(1, 1), 0).1, 0);
    }
}
