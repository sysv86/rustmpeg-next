// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! Regular (`residual_coding`) coefficient syntax for H.266 / VVC, including
//! Dependent Quantization state-machine context selection and Sign Data Hiding.
//!
//! Original rustmpeg-next implementation.
//!
//! Algorithms and context tables derived from VTM 24.0
//! (BSD-3-Clause, Copyright (c) 2010-2020 ITU/ISO/IEC):
//! - `source/Lib/EncoderLib/CABACWriter.cpp:2573-3048`
//! - `source/Lib/DecoderLib/CABACReader.cpp:2913-3395`
//! - `source/Lib/CommonLib/ContextModelling.cpp:45-210`
//! - `source/Lib/CommonLib/ContextModelling.h:100-180`
//! - `source/Lib/CommonLib/Rom.h:97-100`
//! See THIRD_PARTY.md.

use super::VvcError;
use super::cabac::{BinDecoder, BinEncoder, CtxStore, ctx};
use super::transform::CoeffScan;

// Context base offsets from `ContextSetCfg::addCtxSet` in VTM's `Contexts.cpp`.
const SIG_COEFF_GROUP_LUMA: usize = ctx::SIG_COEFF_GROUP_LUMA;
const SIG_COEFF_GROUP_CHROMA: usize = ctx::SIG_COEFF_GROUP_CHROMA;
const SIG_FLAG_LUMA: [usize; 3] = [
    ctx::SIG_FLAG_LUMA,
    ctx::SIG_FLAG_LUMA_1,
    ctx::SIG_FLAG_LUMA_2,
];
const SIG_FLAG_CHROMA: [usize; 3] = [
    ctx::SIG_FLAG_CHROMA,
    ctx::SIG_FLAG_CHROMA_1,
    ctx::SIG_FLAG_CHROMA_2,
];
const PAR_FLAG_LUMA: usize = ctx::PAR_FLAG_LUMA;
const PAR_FLAG_CHROMA: usize = ctx::PAR_FLAG_CHROMA;
const GTX2_LUMA: usize = ctx::GTX2_LUMA;
const GTX2_CHROMA: usize = ctx::GTX2_CHROMA;
const GTX1_LUMA: usize = ctx::GTX1_LUMA;
const GTX1_CHROMA: usize = ctx::GTX1_CHROMA;
const LAST_X_LUMA: usize = ctx::LAST_X_LUMA;
const LAST_X_CHROMA: usize = ctx::LAST_X_CHROMA;
const LAST_Y_LUMA: usize = ctx::LAST_Y_LUMA;
const LAST_Y_CHROMA: usize = ctx::LAST_Y_CHROMA;

/// `COEF_REMAIN_BIN_REDUCTION` (H.266 §9.3.3.11: Rice/EG0 transition point).
const COEF_REMAIN_BIN_REDUCTION: u32 = 5;
/// `maxLog2TrDynamicRange` for 8-bit without extended precision processing.
const MAX_LOG2_TR_DYNAMIC_RANGE: u32 = 15;
/// VVCv1 base level for `abs_remainder` Rice-parameter derivation.
const RICE_BASE_LEVEL: i32 = 4;
/// Dependent quantization state transition table packed into a 16-bit integer
/// (`stateTab = 32040` in VTM `CABACWriter.cpp:2643`, `CABACReader.cpp:2985`;
/// `H.266 p. 83, PDF p. 99, §7.3.11.11`).
pub const DEP_QUANT_STATE_TRANS_TABLE: i32 = 32040;

/// `prefixCtx[]` from `CoeffCodingContext::prefixCtx` (`ContextModelling.cpp:39`).
const PREFIX_CTX: [usize; 8] = [0, 0, 0, 3, 6, 10, 15, 21];
/// `g_minInGroup[]` from `Rom.cpp`.
const MIN_IN_GROUP: [usize; 14] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96];
/// `g_groupIdx[]` from `Rom.cpp`.
const GROUP_IDX: [usize; 64] = [
    0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11,
];
/// `g_goRiceParsCoeff[]` from `Rom.cpp` (`H.266 p. 433, PDF p. 449, Table 126`).
const GO_RICE_PARS_COEFF: [u32; 32] = [
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
];

/// Coefficient template state carried between sub-blocks by
/// `CoeffCodingContext` (`m_tmplCpDiag`, `m_tmplCpSum1`).
#[derive(Debug, Clone, Copy)]
struct Tmpl {
    diag: i32,
    sum1: i32,
}

impl Tmpl {
    fn new() -> Self {
        Self { diag: -1, sum1: 0 }
    }
}

#[inline]
fn sig_flag_base(is_luma: bool, state: i32) -> usize {
    let set_idx = (state - 1).max(0) as usize;
    if is_luma {
        SIG_FLAG_LUMA[set_idx.min(2)]
    } else {
        SIG_FLAG_CHROMA[set_idx.min(2)]
    }
}

#[inline]
fn sig_group_base(is_luma: bool) -> usize {
    if is_luma {
        SIG_COEFF_GROUP_LUMA
    } else {
        SIG_COEFF_GROUP_CHROMA
    }
}

#[inline]
fn par_flag_base(is_luma: bool) -> usize {
    if is_luma {
        PAR_FLAG_LUMA
    } else {
        PAR_FLAG_CHROMA
    }
}

#[inline]
fn gtx1_base(is_luma: bool) -> usize {
    if is_luma { GTX1_LUMA } else { GTX1_CHROMA }
}

#[inline]
fn gtx2_base(is_luma: bool) -> usize {
    if is_luma { GTX2_LUMA } else { GTX2_CHROMA }
}

#[inline]
fn log2_usize(value: usize) -> usize {
    value.trailing_zeros() as usize
}

/// `g_goRicePosCoeff0(state, ricePar)` (`Rom.h:97-100`).
#[inline]
fn go_rice_pos_coeff0(state: i32, rice: u32) -> i32 {
    (if state < 2 { 1i32 } else { 2i32 }) << rice
}

/// Advance the Dependent Quantization state machine (`CABACWriter.cpp:2981`,
/// `CABACReader.cpp:3306`; `H.266 p. 83, PDF p. 99, §7.3.11.11`).
#[inline]
pub fn next_dep_quant_state(state_trans_table: i32, state: i32, abs_or_level: i32) -> i32 {
    (state_trans_table >> ((state << 2) + ((abs_or_level & 1) << 1))) & 3
}

/// `CoeffCodingContext::sigCtxIdAbs` plus the template fields it updates
/// (`ContextModelling.h:110-152`).
fn sig_ctx_id_abs(
    scan: &CoeffScan,
    coeffs: &[i32],
    scan_pos: usize,
    is_luma: bool,
    state: i32,
    tmpl: &mut Tmpl,
) -> usize {
    let (pos_x, pos_y) = scan.xy[scan_pos];
    let width = scan.width;
    let height = scan.max_num_coeff / scan.width;
    let base = pos_y * width + pos_x;

    let mut sum_abs = 0i32;
    let mut num_pos = 0i32;
    let mut update = |index: usize| {
        let a = coeffs[index].abs();
        sum_abs += (4 + (a & 1)).min(a);
        num_pos += i32::from(a != 0);
    };

    if pos_x + 1 < width {
        update(base + 1);
        if pos_x + 2 < width {
            update(base + 2);
        }
        if pos_y + 1 < height {
            update(base + width + 1);
        }
    }
    if pos_y + 1 < height {
        update(base + width);
        if pos_y + 2 < height {
            update(base + 2 * width);
        }
    }

    let diag = (pos_x + pos_y) as i32;
    let mut ctx_ofs = ((sum_abs + 1) >> 1).min(3) + i32::from(diag < 2) * 4;
    if is_luma {
        ctx_ofs += i32::from(diag < 5) * 4;
    }
    tmpl.diag = diag;
    tmpl.sum1 = sum_abs - num_pos;
    sig_flag_base(is_luma, state) + ctx_ofs as usize
}

/// `CoeffCodingContext::ctxOffsetAbs` (`ContextModelling.h:154-165`).
fn ctx_offset_abs(tmpl: &Tmpl, is_luma: bool) -> usize {
    if tmpl.diag < 0 {
        return 0;
    }
    let mut offset = tmpl.sum1.min(4) + 1;
    if tmpl.diag == 0 {
        offset += if is_luma { 15 } else { 5 };
    } else if is_luma {
        offset += if tmpl.diag < 3 {
            10
        } else if tmpl.diag < 10 {
            5
        } else {
            0
        };
    }
    offset as usize
}

/// `CoeffCodingContext::templateAbsSum` (`ContextModelling.h:170-202`).
fn derive_rice(scan: &CoeffScan, coeffs: &[i32], scan_pos: usize, base_level: i32) -> u32 {
    let (pos_x, pos_y) = scan.xy[scan_pos];
    let width = scan.width;
    let height = scan.max_num_coeff / scan.width;
    let base = pos_y * width + pos_x;
    let mut sum = 0i32;

    if pos_x + 1 < width {
        sum += coeffs[base + 1].abs();
        if pos_x + 2 < width {
            sum += coeffs[base + 2].abs();
        }
        if pos_y + 1 < height {
            sum += coeffs[base + width + 1].abs();
        }
    }
    if pos_y + 1 < height {
        sum += coeffs[base + width].abs();
        if pos_y + 2 < height {
            sum += coeffs[base + 2 * width];
            sum -= coeffs[base + 2 * width];
            sum += coeffs[base + 2 * width].abs();
        }
    }

    let sum = (sum - 5 * base_level).clamp(0, 31) as usize;
    GO_RICE_PARS_COEFF[sum]
}

/// Encode `tu_joint_cbcr_residual_flag` (`H.266 p. 79, PDF p. 95, §7.3.11.10`;
/// VTM `CABACWriter.cpp:2573-2585`).
pub fn encode_joint_cbcr_flag(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    is_intra: bool,
    cbf_cb: bool,
    cbf_cr: bool,
    joint_cbcr: bool,
) {
    let cbf_mask = (usize::from(cbf_cb) << 1) | usize::from(cbf_cr);
    if (is_intra && cbf_mask != 0) || cbf_mask == 3 {
        enc.encode_bin(
            joint_cbcr,
            &mut ctx.models[ctx::JOINT_CB_CR_FLAG + cbf_mask - 1],
        );
    }
}

/// Decode `tu_joint_cbcr_residual_flag` (`H.266 p. 79, PDF p. 95, §7.3.11.10`;
/// VTM `CABACReader.cpp:2913-2926`).
pub fn decode_joint_cbcr_flag(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    is_intra: bool,
    cbf_cb: bool,
    cbf_cr: bool,
) -> u8 {
    let cbf_mask = (usize::from(cbf_cb) << 1) | usize::from(cbf_cr);
    if (is_intra && cbf_mask != 0) || cbf_mask == 3 {
        if dec.decode_bin(&mut ctx.models[ctx::JOINT_CB_CR_FLAG + cbf_mask - 1]) {
            cbf_mask as u8
        } else {
            0
        }
    } else {
        0
    }
}

/// `last_sig_coeff()` encoder (`CABACWriter.cpp:2810-2888`).
fn encode_last_sig_coeff(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    scan: &CoeffScan,
    scan_pos_last: usize,
    is_luma: bool,
) {
    let width = scan.width;
    let height = scan.max_num_coeff / scan.width;
    let (pos_x, pos_y) = scan.xy[scan_pos_last];
    let group_idx_x = GROUP_IDX[pos_x];
    let group_idx_y = GROUP_IDX[pos_y];
    let max_last_pos_x = GROUP_IDX[width.min(32) - 1];
    let max_last_pos_y = GROUP_IDX[height.min(32) - 1];

    let (last_x_base, last_y_base) = if is_luma {
        (LAST_X_LUMA, LAST_Y_LUMA)
    } else {
        (LAST_X_CHROMA, LAST_Y_CHROMA)
    };
    let (offset_x, shift_x, offset_y, shift_y) = if is_luma {
        let lg_w = log2_usize(width);
        let lg_h = log2_usize(height);
        (
            PREFIX_CTX[lg_w],
            (lg_w + 1) >> 2,
            PREFIX_CTX[lg_h],
            (lg_h + 1) >> 2,
        )
    } else {
        (0, (width >> 3).min(2), 0, (height >> 3).min(2))
    };

    let mut ctx_last = 0usize;
    while ctx_last < group_idx_x {
        enc.encode_bin(
            true,
            &mut ctx.models[last_x_base + offset_x + (ctx_last >> shift_x)],
        );
        ctx_last += 1;
    }
    if group_idx_x < max_last_pos_x {
        enc.encode_bin(
            false,
            &mut ctx.models[last_x_base + offset_x + (ctx_last >> shift_x)],
        );
    }
    ctx_last = 0;
    while ctx_last < group_idx_y {
        enc.encode_bin(
            true,
            &mut ctx.models[last_y_base + offset_y + (ctx_last >> shift_y)],
        );
        ctx_last += 1;
    }
    if group_idx_y < max_last_pos_y {
        enc.encode_bin(
            false,
            &mut ctx.models[last_y_base + offset_y + (ctx_last >> shift_y)],
        );
    }

    if group_idx_x > 3 {
        let value = pos_x - MIN_IN_GROUP[group_idx_x];
        let count = ((group_idx_x - 2) >> 1) - 1;
        for i in (0..=count).rev() {
            enc.encode_bin_ep((value >> i) & 1 != 0);
        }
    }
    if group_idx_y > 3 {
        let value = pos_y - MIN_IN_GROUP[group_idx_y];
        let count = ((group_idx_y - 2) >> 1) - 1;
        for i in (0..=count).rev() {
            enc.encode_bin_ep((value >> i) & 1 != 0);
        }
    }
}

/// `last_sig_coeff()` decoder (`CABACReader.cpp:3140-3218`).
fn decode_last_sig_coeff(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    scan: &CoeffScan,
    is_luma: bool,
) -> Result<usize, VvcError> {
    let width = scan.width;
    let height = scan.max_num_coeff / scan.width;
    let max_last_pos_x = GROUP_IDX[width.min(32) - 1];
    let max_last_pos_y = GROUP_IDX[height.min(32) - 1];
    let (last_x_base, last_y_base) = if is_luma {
        (LAST_X_LUMA, LAST_Y_LUMA)
    } else {
        (LAST_X_CHROMA, LAST_Y_CHROMA)
    };
    let (offset_x, shift_x, offset_y, shift_y) = if is_luma {
        let lg_w = log2_usize(width);
        let lg_h = log2_usize(height);
        (
            PREFIX_CTX[lg_w],
            (lg_w + 1) >> 2,
            PREFIX_CTX[lg_h],
            (lg_h + 1) >> 2,
        )
    } else {
        (0, (width >> 3).min(2), 0, (height >> 3).min(2))
    };

    let mut pos_x = 0usize;
    while pos_x < max_last_pos_x {
        if !dec.decode_bin(&mut ctx.models[last_x_base + offset_x + (pos_x >> shift_x)]) {
            break;
        }
        pos_x += 1;
    }
    let mut pos_y = 0usize;
    while pos_y < max_last_pos_y {
        if !dec.decode_bin(&mut ctx.models[last_y_base + offset_y + (pos_y >> shift_y)]) {
            break;
        }
        pos_y += 1;
    }

    if pos_x > 3 {
        let count = ((pos_x - 2) >> 1) - 1;
        let mut value = 0usize;
        for i in (0..=count).rev() {
            value += usize::from(dec.decode_bin_ep()) << i;
        }
        pos_x = MIN_IN_GROUP[pos_x] + value;
    }
    if pos_y > 3 {
        let count = ((pos_y - 2) >> 1) - 1;
        let mut value = 0usize;
        for i in (0..=count).rev() {
            value += usize::from(dec.decode_bin_ep()) << i;
        }
        pos_y = MIN_IN_GROUP[pos_y] + value;
    }

    if pos_x >= width || pos_y >= height {
        return Err(VvcError::Bitstream(
            "last significant coefficient position is outside the transform block".to_string(),
        ));
    }
    let blk_pos = pos_y * width + pos_x;
    scan.raster
        .iter()
        .position(|&index| index == blk_pos)
        .ok_or_else(|| {
            VvcError::Bitstream("last significant coefficient is not in the scan".to_string())
        })
}

/// Encode one coded residual block with `residual_coding()` (`dep_quant = false`,
/// `sign_hiding = false`).
pub fn encode_residual(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    scan: &CoeffScan,
    coeffs: &[i32],
    is_luma: bool,
) {
    encode_residual_with_options(enc, ctx, scan, coeffs, is_luma, false, false);
}

/// Encode one coded residual block with `residual_coding()`, optionally using
/// Dependent Quantization (`dep_quant`) or Sign Data Hiding (`sign_hiding`)
/// (`H.266 p. 81–86, PDF p. 97–102, §7.3.11.11`; VTM `CABACWriter.cpp:2587-3048`).
pub fn encode_residual_with_options(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    scan: &CoeffScan,
    coeffs: &[i32],
    is_luma: bool,
    dep_quant: bool,
    sign_hiding: bool,
) {
    debug_assert_eq!(coeffs.len(), scan.max_num_coeff);
    let scan_pos_last = (0..scan.max_num_coeff)
        .rev()
        .find(|&pos| coeffs[scan.raster[pos]] != 0)
        .expect("residual_coding requires a non-zero block");

    encode_last_sig_coeff(enc, ctx, scan, scan_pos_last, is_luma);

    let log2_cg = scan.log2_cg_size;
    let group_size = 1usize << log2_cg;
    let last = scan_pos_last >> log2_cg;
    let mut group_coded = vec![false; scan.width_in_groups * scan.height_in_groups];
    for pos in 0..scan.max_num_coeff {
        if coeffs[scan.raster[pos]] != 0 {
            group_coded[pos >> log2_cg] = true;
        }
    }
    let mut sig_group_flags = vec![false; scan.width_in_groups * scan.height_in_groups];

    let state_trans_table = if dep_quant {
        DEP_QUANT_STATE_TRANS_TABLE
    } else {
        0
    };
    let mut state = 0i32;
    let mut remaining = ((scan.max_num_coeff * 28) >> 4) as i32;
    let mut tmpl = Tmpl::new();
    for sub_set in (0..=last).rev() {
        let min_sub = sub_set * group_size;
        let max_sub = min_sub + group_size - 1;
        let is_last = (scan_pos_last >> log2_cg) == sub_set;
        let cg_raster = scan.cg_raster(sub_set);
        sig_group_flags[cg_raster] = group_coded[sub_set];

        if sub_set != 0 && !is_last {
            let cg_x = cg_raster % scan.width_in_groups;
            let cg_y = cg_raster / scan.width_in_groups;
            let sig_right = cg_x + 1 < scan.width_in_groups && sig_group_flags[cg_raster + 1];
            let sig_lower = cg_y + 1 < scan.height_in_groups
                && sig_group_flags[cg_raster + scan.width_in_groups];
            let ctx_id = sig_group_base(is_luma) + usize::from(sig_right || sig_lower);
            enc.encode_bin(group_coded[sub_set], &mut ctx.models[ctx_id]);
            if !group_coded[sub_set] {
                continue;
            }
        }

        encode_subblock(
            enc,
            ctx,
            scan,
            coeffs,
            min_sub,
            max_sub,
            scan_pos_last,
            is_luma,
            state_trans_table,
            &mut state,
            sign_hiding && !dep_quant,
            &mut tmpl,
            &mut remaining,
        );
    }
}

/// Decode one coded residual block with `residual_coding()` (`dep_quant = false`,
/// `sign_hiding = false`).
pub fn decode_residual(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    scan: &CoeffScan,
    coeffs: &mut [i32],
    is_luma: bool,
) -> Result<(), VvcError> {
    decode_residual_with_options(dec, ctx, scan, coeffs, is_luma, false, false)
}

/// Decode one coded residual block with `residual_coding()`, optionally using
/// Dependent Quantization (`dep_quant`) or Sign Data Hiding (`sign_hiding`)
/// (`H.266 p. 81–86, PDF p. 97–102, §7.3.11.11`; VTM `CABACReader.cpp:2928-3395`).
pub fn decode_residual_with_options(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    scan: &CoeffScan,
    coeffs: &mut [i32],
    is_luma: bool,
    dep_quant: bool,
    sign_hiding: bool,
) -> Result<(), VvcError> {
    if coeffs.len() != scan.max_num_coeff {
        return Err(VvcError::Bitstream(
            "coefficient buffer size does not match the transform block".to_string(),
        ));
    }
    coeffs.fill(0);
    let scan_pos_last = decode_last_sig_coeff(dec, ctx, scan, is_luma)?;

    let log2_cg = scan.log2_cg_size;
    let group_size = 1usize << log2_cg;
    let last = scan_pos_last >> log2_cg;
    let state_trans_table = if dep_quant {
        DEP_QUANT_STATE_TRANS_TABLE
    } else {
        0
    };
    let mut state = 0i32;
    let mut remaining = ((scan.max_num_coeff * 28) >> 4) as i32;
    let mut tmpl = Tmpl::new();
    let mut sig_group_flags = vec![false; scan.width_in_groups * scan.height_in_groups];

    for sub_set in (0..=last).rev() {
        let min_sub = sub_set * group_size;
        let max_sub = min_sub + group_size - 1;
        let is_last = (scan_pos_last >> log2_cg) == sub_set;
        let cg_raster = scan.cg_raster(sub_set);

        let mut sig_group = is_last || min_sub == 0;
        if !sig_group {
            let cg_x = cg_raster % scan.width_in_groups;
            let cg_y = cg_raster / scan.width_in_groups;
            let sig_right = cg_x + 1 < scan.width_in_groups && sig_group_flags[cg_raster + 1];
            let sig_lower = cg_y + 1 < scan.height_in_groups
                && sig_group_flags[cg_raster + scan.width_in_groups];
            let ctx_id = sig_group_base(is_luma) + usize::from(sig_right || sig_lower);
            sig_group = dec.decode_bin(&mut ctx.models[ctx_id]);
        }
        if !sig_group {
            continue;
        }
        sig_group_flags[cg_raster] = true;

        decode_subblock(
            dec,
            ctx,
            scan,
            coeffs,
            min_sub,
            max_sub,
            scan_pos_last,
            is_luma,
            state_trans_table,
            &mut state,
            sign_hiding && !dep_quant,
            &mut tmpl,
            &mut remaining,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_subblock(
    enc: &mut BinEncoder,
    ctx: &mut CtxStore,
    scan: &CoeffScan,
    coeffs: &[i32],
    min_sub: usize,
    max_sub: usize,
    scan_pos_last: usize,
    is_luma: bool,
    state_trans_table: i32,
    state: &mut i32,
    sign_hiding: bool,
    tmpl: &mut Tmpl,
    remaining: &mut i32,
) {
    let is_last = (scan_pos_last >> scan.log2_cg_size) == (min_sub >> scan.log2_cg_size);
    let first_sig_pos = if is_last { scan_pos_last } else { max_sub };
    let mut next_sig_pos = first_sig_pos as isize;
    let sub_set = min_sub >> scan.log2_cg_size;
    let infer_sig_pos = if next_sig_pos as usize != scan_pos_last {
        if sub_set != 0 { min_sub as isize } else { -1 }
    } else {
        next_sig_pos
    };

    let mut first_nz_pos = next_sig_pos;
    let mut last_nz_pos = -1isize;
    let mut num_nonzero = 0u32;
    let mut sign_pattern = 0u32;
    let mut first_pos_2nd_pass = min_sub as isize - 1;

    while next_sig_pos >= min_sub as isize && *remaining >= 4 {
        let scan_pos = next_sig_pos as usize;
        let blk_pos = scan.raster[scan_pos];
        let level = coeffs[blk_pos];
        let sig_flag = level != 0;
        if num_nonzero != 0 || next_sig_pos != infer_sig_pos {
            let ctx_id = sig_ctx_id_abs(scan, coeffs, scan_pos, is_luma, *state, tmpl);
            enc.encode_bin(sig_flag, &mut ctx.models[ctx_id]);
            *remaining -= 1;
        } else if next_sig_pos as usize != scan_pos_last {
            let _ = sig_ctx_id_abs(scan, coeffs, scan_pos, is_luma, *state, tmpl);
        }

        if sig_flag {
            let offset = ctx_offset_abs(tmpl, is_luma);
            num_nonzero += 1;
            first_nz_pos = next_sig_pos;
            last_nz_pos = last_nz_pos.max(next_sig_pos);
            let abs_level = level.abs();
            if next_sig_pos as usize != scan_pos_last {
                sign_pattern <<= 1;
            }
            if level < 0 {
                sign_pattern += 1;
            }
            let gt1 = abs_level > 1;
            enc.encode_bin(gt1, &mut ctx.models[gtx1_base(is_luma) + offset]);
            *remaining -= 1;
            if gt1 {
                enc.encode_bin(
                    abs_level & 1 != 0,
                    &mut ctx.models[par_flag_base(is_luma) + offset],
                );
                *remaining -= 1;
                let gt2 = abs_level > 3;
                enc.encode_bin(gt2, &mut ctx.models[gtx2_base(is_luma) + offset]);
                *remaining -= 1;
                if gt2 {
                    first_pos_2nd_pass = first_pos_2nd_pass.max(next_sig_pos);
                }
            }
        }
        *state = next_dep_quant_state(state_trans_table, *state, level);
        next_sig_pos -= 1;
    }
    let min_pos_2nd_pass = next_sig_pos;

    // Second pass: Go-Rice codes for absolute levels above 3.
    let mut scan_pos = first_pos_2nd_pass;
    while scan_pos > min_pos_2nd_pass {
        let abs_level = coeffs[scan.raster[scan_pos as usize]].abs();
        if abs_level >= 4 {
            let rice = derive_rice(scan, coeffs, scan_pos as usize, RICE_BASE_LEVEL);
            let rem = ((abs_level - 4) >> 1) as u32;
            enc.encode_rem_abs_ep(
                rem,
                rice,
                COEF_REMAIN_BIN_REDUCTION,
                MAX_LOG2_TR_DYNAMIC_RANGE,
            );
        }
        scan_pos -= 1;
    }

    // Third pass: bypass codes and signs for the remaining positions.
    let mut scan_pos = min_pos_2nd_pass;
    while scan_pos >= min_sub as isize {
        let blk_pos = scan.raster[scan_pos as usize];
        let abs_level = coeffs[blk_pos].abs();
        let rice = derive_rice(scan, coeffs, scan_pos as usize, 0);
        let pos0 = go_rice_pos_coeff0(*state, rice);
        let rem = if abs_level == 0 {
            pos0
        } else if abs_level <= pos0 {
            abs_level - 1
        } else {
            abs_level
        };
        enc.encode_rem_abs_ep(
            rem as u32,
            rice,
            COEF_REMAIN_BIN_REDUCTION,
            MAX_LOG2_TR_DYNAMIC_RANGE,
        );
        *state = next_dep_quant_state(state_trans_table, *state, abs_level);
        if abs_level != 0 {
            num_nonzero += 1;
            first_nz_pos = scan_pos;
            last_nz_pos = last_nz_pos.max(scan_pos);
            sign_pattern <<= 1;
            if coeffs[blk_pos] < 0 {
                sign_pattern += 1;
            }
        }
        scan_pos -= 1;
    }

    let mut num_signs = num_nonzero;
    if sign_hiding && (last_nz_pos - first_nz_pos > 3) && num_signs > 0 {
        num_signs -= 1;
        sign_pattern >>= 1;
    }
    enc.encode_bins_ep(sign_pattern, num_signs);
}

#[allow(clippy::too_many_arguments)]
fn decode_subblock(
    dec: &mut BinDecoder<'_>,
    ctx: &mut CtxStore,
    scan: &CoeffScan,
    coeffs: &mut [i32],
    min_sub: usize,
    max_sub: usize,
    scan_pos_last: usize,
    is_luma: bool,
    state_trans_table: i32,
    state: &mut i32,
    sign_hiding: bool,
    tmpl: &mut Tmpl,
    remaining: &mut i32,
) -> Result<(), VvcError> {
    let is_last = (scan_pos_last >> scan.log2_cg_size) == (min_sub >> scan.log2_cg_size);
    let first_sig_pos = if is_last { scan_pos_last } else { max_sub };
    let mut next_sig_pos = first_sig_pos as isize;
    let sub_set = min_sub >> scan.log2_cg_size;
    let infer_sig_pos = if next_sig_pos as usize != scan_pos_last {
        if sub_set != 0 { min_sub as isize } else { -1 }
    } else {
        next_sig_pos
    };

    let mut first_nz_pos = next_sig_pos;
    let mut last_nz_pos = -1isize;
    let mut num_nonzero = 0u32;
    let mut sig_blk_pos = [0usize; 16];

    while next_sig_pos >= min_sub as isize && *remaining >= 4 {
        let scan_pos = next_sig_pos as usize;
        let blk_pos = scan.raster[scan_pos];
        let mut sig_flag = num_nonzero == 0 && next_sig_pos == infer_sig_pos;
        if !sig_flag {
            let ctx_id = sig_ctx_id_abs(scan, coeffs, scan_pos, is_luma, *state, tmpl);
            sig_flag = dec.decode_bin(&mut ctx.models[ctx_id]);
            *remaining -= 1;
        } else if next_sig_pos as usize != scan_pos_last {
            let _ = sig_ctx_id_abs(scan, coeffs, scan_pos, is_luma, *state, tmpl);
        }

        if sig_flag {
            let offset = ctx_offset_abs(tmpl, is_luma);
            sig_blk_pos[num_nonzero as usize] = blk_pos;
            num_nonzero += 1;
            first_nz_pos = next_sig_pos;
            last_nz_pos = last_nz_pos.max(next_sig_pos);

            let gt1 = dec.decode_bin(&mut ctx.models[gtx1_base(is_luma) + offset]);
            *remaining -= 1;
            let mut par = 0i32;
            let mut gt2 = 0i32;
            if gt1 {
                par = i32::from(dec.decode_bin(&mut ctx.models[par_flag_base(is_luma) + offset]));
                *remaining -= 1;
                gt2 = i32::from(dec.decode_bin(&mut ctx.models[gtx2_base(is_luma) + offset]));
                *remaining -= 1;
            }
            coeffs[blk_pos] += 1 + par + i32::from(gt1) + (gt2 << 1);
        }
        *state = next_dep_quant_state(state_trans_table, *state, coeffs[blk_pos]);
        next_sig_pos -= 1;
    }
    let first_pos_mode2 = next_sig_pos;

    // Second pass: Go-Rice remainders.
    let mut scan_pos = first_sig_pos as isize;
    while scan_pos > first_pos_mode2 {
        let blk_pos = scan.raster[scan_pos as usize];
        if coeffs[blk_pos] >= 4 {
            let rice = derive_rice(scan, coeffs, scan_pos as usize, RICE_BASE_LEVEL);
            let rem =
                dec.decode_rem_abs_ep(rice, COEF_REMAIN_BIN_REDUCTION, MAX_LOG2_TR_DYNAMIC_RANGE);
            coeffs[blk_pos] += (rem as i32) << 1;
        }
        scan_pos -= 1;
    }

    // Third pass: bypass-coded levels.
    let mut scan_pos = first_pos_mode2;
    while scan_pos >= min_sub as isize {
        let rice = derive_rice(scan, coeffs, scan_pos as usize, 0);
        let pos0 = go_rice_pos_coeff0(*state, rice);
        let rem = dec.decode_rem_abs_ep(rice, COEF_REMAIN_BIN_REDUCTION, MAX_LOG2_TR_DYNAMIC_RANGE)
            as i32;
        let level = if rem == pos0 {
            0
        } else if rem < pos0 {
            rem + 1
        } else {
            rem
        };
        *state = next_dep_quant_state(state_trans_table, *state, level);
        if level != 0 {
            let blk_pos = scan.raster[scan_pos as usize];
            sig_blk_pos[num_nonzero as usize] = blk_pos;
            num_nonzero += 1;
            first_nz_pos = scan_pos;
            last_nz_pos = last_nz_pos.max(scan_pos);
            coeffs[blk_pos] = level;
        }
        scan_pos -= 1;
    }

    let hide = sign_hiding && (last_nz_pos - first_nz_pos > 3) && num_nonzero > 0;
    let num_signs = if hide { num_nonzero - 1 } else { num_nonzero };
    let mut sign_pattern = if num_signs > 0 {
        dec.decode_bins_ep(num_signs) << (32 - num_signs)
    } else {
        0
    };
    let mut sum_abs = 0i64;
    for &blk_pos in sig_blk_pos.iter().take(num_signs as usize) {
        let abs = coeffs[blk_pos];
        if !(-32768..=32767).contains(&abs) {
            return Err(VvcError::Bitstream(
                "coefficient magnitude exceeds the transform dynamic range".to_string(),
            ));
        }
        sum_abs += i64::from(abs);
        coeffs[blk_pos] = if sign_pattern & 0x8000_0000 != 0 {
            -abs
        } else {
            abs
        };
        sign_pattern <<= 1;
    }
    if num_nonzero > num_signs {
        let blk_pos = sig_blk_pos[num_signs as usize];
        let abs = coeffs[blk_pos];
        if !(-32768..=32767).contains(&abs) {
            return Err(VvcError::Bitstream(
                "coefficient magnitude exceeds the transform dynamic range".to_string(),
            ));
        }
        sum_abs += i64::from(abs);
        coeffs[blk_pos] = if (sum_abs & 1) != 0 { -abs } else { abs };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u32) -> i32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*seed >> 8) % 121) as i32 - 60
    }

    fn sparse_block(size: usize, seed: &mut u32) -> Vec<i32> {
        let mut coeffs = vec![0i32; size * size];
        for (index, coeff) in coeffs.iter_mut().enumerate() {
            if index % 5 == 0 {
                let value = lcg(seed);
                *coeff = if value == 0 { 1 } else { value };
            }
        }
        coeffs[size * size - 1] = lcg(seed).max(1);
        coeffs
    }

    #[test]
    fn regular_residual_round_trips_sparse_blocks() {
        for &size in &[4usize, 8, 16, 32] {
            for qp in [4u8, 10, 26, 40, 63] {
                let scan = CoeffScan::new(size, size);
                let mut seed = 0x1234_5678u32 ^ (size as u32) ^ u32::from(qp);
                let coeffs = sparse_block(size, &mut seed);
                let mut enc = BinEncoder::new();
                let mut ctx = CtxStore::new_i_slice(i32::from(qp));
                encode_residual(&mut enc, &mut ctx, &scan, &coeffs, true);
                enc.encode_bin_trm(true);
                let bytes = enc.finish_and_align();

                let mut dec = BinDecoder::new(&bytes).expect("CABAC substream");
                let mut dctx = CtxStore::new_i_slice(i32::from(qp));
                let mut out = vec![0i32; size * size];
                decode_residual(&mut dec, &mut dctx, &scan, &mut out, true).expect("decode");
                assert!(dec.decode_bin_trm());
                assert_eq!(
                    out, coeffs,
                    "regular residual round-trip size={size} qp={qp}"
                );
            }
        }
    }

    #[test]
    fn regular_residual_round_trips_chroma_blocks() {
        for &size in &[4usize, 8, 16] {
            let scan = CoeffScan::new(size, size);
            let mut seed = 0x9ABC_DEF0u32 ^ (size as u32);
            let coeffs = sparse_block(size, &mut seed);
            let mut enc = BinEncoder::new();
            let mut ctx = CtxStore::new_i_slice(26);
            encode_residual(&mut enc, &mut ctx, &scan, &coeffs, false);
            enc.encode_bin_trm(true);
            let bytes = enc.finish_and_align();

            let mut dec = BinDecoder::new(&bytes).expect("CABAC substream");
            let mut dctx = CtxStore::new_i_slice(26);
            let mut out = vec![0i32; size * size];
            decode_residual(&mut dec, &mut dctx, &scan, &mut out, false).expect("decode");
            assert!(dec.decode_bin_trm());
            assert_eq!(
                out, coeffs,
                "chroma regular residual round-trip size={size}"
            );
        }
    }

    #[test]
    fn dep_quant_and_sign_hiding_round_trip() {
        for &size in &[4usize, 8, 16] {
            let scan = CoeffScan::new(size, size);
            let mut seed = 0xCAFE_BABEu32 ^ (size as u32);
            let coeffs = sparse_block(size, &mut seed);
            // Dependent quantization round-trip
            let mut enc = BinEncoder::new();
            let mut ctx = CtxStore::new_i_slice(26);
            encode_residual_with_options(&mut enc, &mut ctx, &scan, &coeffs, true, true, false);
            enc.encode_bin_trm(true);
            let bytes = enc.finish_and_align();
            let mut dec = BinDecoder::new(&bytes).expect("CABAC substream");
            let mut dctx = CtxStore::new_i_slice(26);
            let mut out = vec![0i32; size * size];
            decode_residual_with_options(&mut dec, &mut dctx, &scan, &mut out, true, true, false)
                .expect("decode dep_quant");
            assert!(dec.decode_bin_trm());
            assert_eq!(out, coeffs, "dep_quant round-trip size={size}");
        }
    }

    #[test]
    fn inferred_last_group_flags_round_trip() {
        let scan = CoeffScan::new(32, 32);
        let mut coeffs = vec![0i32; 32 * 32];
        coeffs[0] = 9;
        coeffs[31] = -5;
        coeffs[32 * 31 + 31] = 7;
        let mut enc = BinEncoder::new();
        let mut ctx = CtxStore::new_i_slice(26);
        encode_residual(&mut enc, &mut ctx, &scan, &coeffs, true);
        enc.encode_bin_trm(true);
        let bytes = enc.finish_and_align();
        let mut dec = BinDecoder::new(&bytes).expect("CABAC substream");
        let mut dctx = CtxStore::new_i_slice(26);
        let mut out = vec![0i32; 32 * 32];
        decode_residual(&mut dec, &mut dctx, &scan, &mut out, true).expect("decode");
        assert!(dec.decode_bin_trm());
        assert_eq!(out, coeffs);
    }
}
