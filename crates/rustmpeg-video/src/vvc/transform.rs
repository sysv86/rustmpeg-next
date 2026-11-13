// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! Transform-skip coefficient helpers, Subblock Transform (SBT), Joint CbCr
//! Inverse Color Transform (ICT), and QT/MTT coding tree split geometry for
//! H.266/VVC (ITU-T Rec. H.266 §7.4.9.4, §8.7.4).
//!
//! Algorithms derived from VTM 24.0 (`TrQuant.cpp`, `UnitTools.cpp`,
//! BSD-3-Clause, Copyright (c) 2010-2026 ITU/ISO/IEC). See `THIRD_PARTY.md`.

use super::dct::TrType;

/// Grouped diagonal scan over a transform block (VTM `g_scanOrder`
/// `SCAN_GROUPED_4x4` with `DIAG`). For every block size >= 4x4 the coefficient
/// groups are 4x4.
#[derive(Debug, Clone)]
pub struct CoeffScan {
    /// Transform block width in samples.
    pub width: usize,
    /// Base-2 logarithm of the number of coefficients per group (`4` for 4x4 CGs).
    pub log2_cg_size: u32,
    /// Width of the block in coefficient groups.
    pub width_in_groups: usize,
    /// Height of the block in coefficient groups.
    pub height_in_groups: usize,
    /// Maximum number of scanned coefficients after 32x32 high-frequency zeroing.
    pub max_num_coeff: usize,
    /// Raster index for every scan position.
    pub raster: Vec<usize>,
    /// `(x, y)` sample index for every scan position.
    pub xy: Vec<(usize, usize)>,
    /// `(x, y)` position of every coefficient group in the group grid.
    pub group_xy: Vec<(usize, usize)>,
}

impl CoeffScan {
    /// Build the grouped scan for a `width x height` block.
    ///
    /// Internal invariant: both dimensions are powers of two and at least 2.
    /// For 64-point transforms, only the top-left 32x32 coefficient region is scanned.
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width.is_power_of_two() && height.is_power_of_two());
        let (log2_cg_w, log2_cg_h) = coefficient_group_log2(width, height);
        let cg_w = 1usize << log2_cg_w;
        let cg_h = 1usize << log2_cg_h;
        let capped_w = width.min(32);
        let capped_h = height.min(32);
        let width_in_groups = capped_w >> log2_cg_w;
        let height_in_groups = capped_h >> log2_cg_h;

        let group_order = diagonal_scan(width_in_groups, height_in_groups);
        let inner = diagonal_scan(cg_w, cg_h);
        let group_xy = group_order.clone();
        let max_num_coeff = capped_w * capped_h;
        let mut raster = Vec::with_capacity(max_num_coeff);
        let mut xy = Vec::with_capacity(max_num_coeff);
        for (gx, gy) in group_order {
            for &(ix, iy) in &inner {
                let x = gx * cg_w + ix;
                let y = gy * cg_h + iy;
                raster.push(y * width + x);
                xy.push((x, y));
            }
        }
        debug_assert_eq!(raster.len(), max_num_coeff);
        Self {
            width,
            log2_cg_size: (log2_cg_w + log2_cg_h) as u32,
            width_in_groups,
            height_in_groups,
            max_num_coeff,
            raster,
            xy,
            group_xy,
        }
    }

    /// Index of the last coefficient group.
    pub fn last_sub_set(&self) -> usize {
        (self.max_num_coeff - 1) >> self.log2_cg_size
    }

    /// Raster index of a coefficient group in the group grid
    /// (VTM `m_scanCG[sub_set].idx`).
    #[inline]
    pub fn cg_raster(&self, sub_set: usize) -> usize {
        let (x, y) = self.group_xy[sub_set];
        y * self.width_in_groups + x
    }
}

/// Normative coefficient-group size for a transform block
/// (`g_log2TxSubblockSize`, ITU-T Rec. H.266 §8.7.4; VTM `Rom.cpp:267-276`).
///
/// Wide or tall blocks use non-square groups (for example `2x8` for a `2x32`
/// transform) so the group grid stays close to square.
fn coefficient_group_log2(width: usize, height: usize) -> (usize, usize) {
    const TABLE: [[(usize, usize); 8]; 8] = [
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 4),
            (0, 4),
            (0, 4),
        ],
        [
            (1, 0),
            (1, 1),
            (1, 1),
            (1, 3),
            (1, 3),
            (1, 3),
            (1, 3),
            (1, 3),
        ],
        [
            (2, 0),
            (1, 1),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        ],
        [
            (3, 0),
            (3, 1),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        ],
        [
            (4, 0),
            (3, 1),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        ],
        [
            (4, 0),
            (3, 1),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        ],
        [
            (4, 0),
            (3, 1),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        ],
        [
            (4, 0),
            (3, 1),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        ],
    ];
    let lx = width.trailing_zeros() as usize;
    let ly = height.trailing_zeros() as usize;
    TABLE[lx.min(7)][ly.min(7)]
}

/// VTM `ScanGenerator` with `CoeffScanType::DIAG`, returning `(x, y)`.
fn diagonal_scan(w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(w * h);
    if w == 0 || h == 0 {
        return out;
    }
    let (mut line, mut column) = (0usize, 0usize);
    for _ in 0..w * h {
        out.push((column, line));
        if column == w - 1 || line == 0 {
            line += column + 1;
            column = 0;
            if line >= h {
                column += line - (h - 1);
                line = h - 1;
            }
        } else {
            column += 1;
            line -= 1;
        }
    }
    out
}

/// Forward quantisation for a transform-skip residual sample
/// (VTM `Quant::quant`, transform-skip path, 8-bit, IRAP).
pub fn quantize_ts(residual: i32, qp: u8) -> i32 {
    let qp = u32::from(qp.max(4));
    let qp_per = qp / 6;
    let qp_rem = qp % 6;
    const QUANT_SCALES: [i32; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
    let i_qbits = 14 + qp_per;
    let scale = QUANT_SCALES[qp_rem as usize];
    let i_add = 171i64 << (i_qbits - 9); // IRAP
    let sign = residual.signum();
    let level = ((i64::from(residual.abs()) * i64::from(scale) + i_add) >> i_qbits) as i32;
    (sign * level).clamp(-32768, 32767)
}

/// Normative inverse quantisation for a transform-skip level
/// (VTM `Quant::dequant`, scaling lists off).
pub fn dequantize_ts(level: i32, qp: u8) -> i32 {
    let qp = u32::from(qp.max(4));
    let qp_per = (qp / 6) as i32;
    let qp_rem = qp % 6;
    const INV_QUANT_SCALES: [i32; 6] = [40, 45, 51, 57, 64, 72];
    let scale = INV_QUANT_SCALES[qp_rem as usize];
    let right_shift = 6 - qp_per;
    let value = i64::from(level) * i64::from(scale);
    let scaled = if right_shift > 0 {
        (value + (1i64 << (right_shift - 1))) >> right_shift
    } else {
        value << (-right_shift)
    };
    (scaled as i32).clamp(-32768, 32767)
}

/// Subblock Transform (SBT) split mode for inter-coded CUs (H.266 §7.4.9.5,
/// §8.7.4.1, VTM `TrQuant.cpp:693-741`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SbtIdx {
    /// Vertical split into two halves (`SBT_VER_HALF = 1`).
    VerHalf,
    /// Horizontal split into two halves (`SBT_HOR_HALF = 2`).
    HorHalf,
    /// Vertical split with 1/4 active transform unit (`SBT_VER_QUAD = 3`).
    VerQuad,
    /// Horizontal split with 1/4 active transform unit (`SBT_HOR_QUAD = 4`).
    HorQuad,
}

/// Derive the horizontal and vertical 1-D transform types `(tr_hor, tr_ver)` for a
/// luma transform unit coded with Subblock Transform (SBT).
///
/// Normative source: ITU-T Rec. H.266 §8.7.4.1 Table 39 (PDF p. 408);
/// VTM `TrQuant.cpp:693-741` (`TrQuant::getTrTypes`).
pub fn derive_sbt_tr_types(
    sbt_idx: SbtIdx,
    sbt_pos_is_second: bool,
    tu_w: usize,
    tu_h: usize,
) -> (TrType, TrType) {
    const MTS_INTER_MAX_CU_SIZE: usize = 32;
    match sbt_idx {
        SbtIdx::VerHalf | SbtIdx::VerQuad => {
            if tu_h > MTS_INTER_MAX_CU_SIZE || tu_w > MTS_INTER_MAX_CU_SIZE {
                (TrType::Dct2, TrType::Dct2)
            } else if !sbt_pos_is_second {
                (TrType::Dct8, TrType::Dst7)
            } else {
                (TrType::Dst7, TrType::Dst7)
            }
        }
        SbtIdx::HorHalf | SbtIdx::HorQuad => {
            if tu_w > MTS_INTER_MAX_CU_SIZE || tu_h > MTS_INTER_MAX_CU_SIZE {
                (TrType::Dct2, TrType::Dct2)
            } else if !sbt_pos_is_second {
                (TrType::Dst7, TrType::Dct8)
            } else {
                (TrType::Dst7, TrType::Dst7)
            }
        }
    }
}

/// Return the active transform sub-rectangle `(tu_bx, tu_by, tu_w, tu_h)` inside an
/// inter CU `(bx, by, w, h)` using SBT (`sbt_idx`, `sbt_pos_is_second`).
pub fn sbt_active_tu_rect(
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    sbt_idx: SbtIdx,
    sbt_pos_is_second: bool,
) -> (usize, usize, usize, usize) {
    match sbt_idx {
        SbtIdx::VerHalf => {
            let hw = w / 2;
            (if sbt_pos_is_second { bx + hw } else { bx }, by, hw, h)
        }
        SbtIdx::HorHalf => {
            let hh = h / 2;
            (bx, if sbt_pos_is_second { by + hh } else { by }, w, hh)
        }
        SbtIdx::VerQuad => {
            let qw = w / 4;
            (if sbt_pos_is_second { bx + w - qw } else { bx }, by, qw, h)
        }
        SbtIdx::HorQuad => {
            let qh = h / 4;
            (bx, if sbt_pos_is_second { by + h - qh } else { by }, w, qh)
        }
    }
}

/// Apply the Joint CbCr Inverse Color Transform (ICT) in-place to `res_cb` and `res_cr`.
///
/// `ict_mode` is the signed ICT mode (`1, -1, 2, -2, 3, -3`) derived from
/// `(tu_cbf_cb, tu_cbf_cr)` and `sps_joint_cbcr_sign_flag` (or `ph_joint_cbcr_sign_flag`).
///
/// Normative source: ITU-T Rec. H.266 §8.7.4.5 Equations 1192–1197 (PDF p. 419);
/// VTM `TrQuant.cpp:125-160` (`invTransformCbCr`).
pub fn apply_ict_joint_cbcr(res_cb: &mut [i32], res_cr: &mut [i32], ict_mode: i8) {
    for (cb, cr) in res_cb.iter_mut().zip(res_cr.iter_mut()) {
        match ict_mode {
            1 => *cr = *cb >> 1,
            -1 => *cr = (-*cb) >> 1,
            2 => *cr = *cb,
            -2 => *cr = if *cb == -32768 { 32767 } else { -*cb },
            3 => *cb = *cr >> 1,
            -3 => *cb = (-*cr) >> 1,
            _ => {}
        }
    }
}

/// Quadtree plus Multi-Type Tree (QT/MTT) coding tree split mode (H.266 §7.4.9.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodingTreeSplit {
    /// Leaf coding unit (`SPLIT_NONE`).
    #[default]
    NoSplit,
    /// Quadtree split into 4 equal quadrants (`SPLIT_QT`).
    QuadTree,
    /// Horizontal binary tree split into top and bottom halves (`SPLIT_BT_HOR`).
    BinaryHor,
    /// Vertical binary tree split into left and right halves (`SPLIT_BT_VER`).
    BinaryVer,
    /// Horizontal ternary tree split into `1/4 : 1/2 : 1/4` rows (`SPLIT_TT_HOR`).
    TernaryHor,
    /// Vertical ternary tree split into `1/4 : 1/2 : 1/4` columns (`SPLIT_TT_VER`).
    TernaryVer,
}

/// Subdivide a coding tree node `(bx, by, w, h)` according to `split` (H.266 §6.4.1).
pub fn split_coding_tree_node(
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    split: CodingTreeSplit,
) -> Vec<(usize, usize, usize, usize)> {
    match split {
        CodingTreeSplit::NoSplit => vec![(bx, by, w, h)],
        CodingTreeSplit::QuadTree => {
            let hw = w / 2;
            let hh = h / 2;
            vec![
                (bx, by, hw, hh),
                (bx + hw, by, hw, hh),
                (bx, by + hh, hw, hh),
                (bx + hw, by + hh, hw, hh),
            ]
        }
        CodingTreeSplit::BinaryHor => {
            let hh = h / 2;
            vec![(bx, by, w, hh), (bx, by + hh, w, hh)]
        }
        CodingTreeSplit::BinaryVer => {
            let hw = w / 2;
            vec![(bx, by, hw, h), (bx + hw, by, hw, h)]
        }
        CodingTreeSplit::TernaryHor => {
            let qh = h / 4;
            vec![
                (bx, by, w, qh),
                (bx, by + qh, w, 2 * qh),
                (bx, by + 3 * qh, w, qh),
            ]
        }
        CodingTreeSplit::TernaryVer => {
            let qw = w / 4;
            vec![
                (bx, by, qw, h),
                (bx + qw, by, 2 * qw, h),
                (bx + 3 * qw, by, qw, h),
            ]
        }
    }
}

/// Normative Adaptive Colour Transform (ACT) per-component QP offset `[-5, 1, 3]`
/// for `(Y/Cg/Co)` (`H.266 §8.7.1`, VTM `CommonDef.h:552`, `Quant.cpp:89`).
pub const DELTA_QP_ACT: [i32; 3] = [-5, 1, 3];

/// Forward quantisation for a transform-skip residual sample with explicit intra/inter
/// rounding offset (`Quant.cpp:64-103, 1120-1150`).
pub fn quantize_ts_with_mode(residual: i32, qp_scaled: u8, is_intra: bool) -> i32 {
    let qp = u32::from(qp_scaled.max(4));
    let qp_per = qp / 6;
    let qp_rem = qp % 6;
    const QUANT_SCALES: [i32; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
    let i_qbits = 14 + qp_per;
    let scale = QUANT_SCALES[qp_rem as usize];
    let offset_num = if is_intra { 171i64 } else { 85i64 };
    let i_add = offset_num << (i_qbits - 9);
    let sign = residual.signum();
    let level = ((i64::from(residual.abs()) * i64::from(scale) + i_add) >> i_qbits) as i32;
    (sign * level).clamp(-32768, 32767)
}

/// Apply the forward Adaptive Colour Transform (ACT, YCgCo-R) to 4:4:4 residual blocks
/// `(res_g, res_b, res_r)` -> `(res_y, res_cg, res_co)`.
///
/// In H.266 / VTM (`Buffer.cpp:765-792`), component 0 is G (`y0`), component 1 is B (`cg`),
/// and component 2 is R (`co`).
pub fn apply_act_forward(
    res_g: &[i32],
    res_b: &[i32],
    res_r: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let n = res_g.len().min(res_b.len()).min(res_r.len());
    let mut out_y = vec![0i32; n];
    let mut out_cg = vec![0i32; n];
    let mut out_co = vec![0i32; n];
    for i in 0..n {
        let g = res_g[i];
        let b = res_b[i];
        let r = res_r[i];
        let co = r - b;
        let t = b + (co >> 1);
        let cg = g - t;
        let y0 = t + (cg >> 1);
        out_y[i] = y0;
        out_cg[i] = cg;
        out_co[i] = co;
    }
    (out_y, out_cg, out_co)
}

/// Apply the normative inverse Adaptive Colour Transform (ACT, YCgCo-R) in-place to
/// `(res_y0, res_cg, res_co)` -> `(res_g, res_b, res_r)` (ITU-T Rec. H.266 §8.7.4.6
/// Eq. 1198–1201, VTM `Buffer.cpp:794-825`).
pub fn apply_act_inverse(
    res_y0: &mut [i32],
    res_cg: &mut [i32],
    res_co: &mut [i32],
    bit_depth: u8,
) {
    let max_abs = 1i32 << (u32::from(bit_depth.clamp(8, 12)) + 1);
    let min_val = -max_abs;
    let max_val = max_abs - 1;
    let n = res_y0.len().min(res_cg.len()).min(res_co.len());
    for i in 0..n {
        let y0 = res_y0[i].clamp(min_val, max_val);
        let cg = res_cg[i].clamp(min_val, max_val);
        let co = res_co[i].clamp(min_val, max_val);
        let t = y0 - (cg >> 1);
        let g = cg + t;
        let b = t - (co >> 1);
        let r = co + b;
        res_y0[i] = g;
        res_cg[i] = b;
        res_co[i] = r;
    }
}

/// Allowed split flags for a QT/MTT coding tree node (ITU-T Rec. H.266 §6.4.1–§6.4.3,
/// VTM `Partitioner::canSplit`, `UnitPartitioner.cpp:383-645`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QtMttCanSplit {
    /// Node may terminate without splitting (`canNo`).
    pub can_no: bool,
    /// Node may split via Quadtree (`canQt`).
    pub can_qt: bool,
    /// Node may split via Horizontal Binary Tree (`canBh`).
    pub can_bh: bool,
    /// Node may split via Vertical Binary Tree (`canBv`).
    pub can_bv: bool,
    /// Node may split via Horizontal Ternary Tree (`canTh`).
    pub can_th: bool,
    /// Node may split via Vertical Ternary Tree (`canTv`).
    pub can_tv: bool,
}

impl QtMttCanSplit {
    /// True if any split mode (`QT`, `BH`, `BV`, `TH`, `TV`) is permitted.
    #[inline]
    pub fn any_split(self) -> bool {
        self.can_qt || self.can_bh || self.can_bv || self.can_th || self.can_tv
    }

    /// True if any MTT split mode (`BH`, `BV`, `TH`, `TV`) is permitted.
    #[inline]
    pub fn any_btt(self) -> bool {
        self.can_bh || self.can_bv || self.can_th || self.can_tv
    }
}

/// Normative CABAC context increments for `split_cu_flag`, `split_qt_flag`,
/// `mtt_split_cu_vertical_flag`, and `mtt_split_cu_binary_flag`
/// (ITU-T Rec. H.266 §9.3.4.2.2, VTM `DeriveCtx::CtxSplit`, `ContextModelling.cpp:170-294`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QtMttSplitContexts {
    /// Context offset `0..=8` for `split_cu_flag` (`ctx::SPLIT_FLAG`).
    pub ctx_split: usize,
    /// Context offset `0..=5` for `split_qt_flag` (`ctx::SPLIT_QT_FLAG`).
    pub ctx_qt: usize,
    /// Context offset `0..=4` for `mtt_split_cu_vertical_flag` (`ctx::SPLIT_HV_FLAG`).
    pub ctx_hv: usize,
    /// Context offset `0..=1` for horizontal `mtt_split_cu_binary_flag` (`ctx::SPLIT_12_FLAG`).
    pub ctx_hor_bt: usize,
    /// Context offset `2..=3` for vertical `mtt_split_cu_binary_flag` (`ctx::SPLIT_12_FLAG`).
    pub ctx_ver_bt: usize,
}

/// Compute which split modes (`canNo`, `canQt`, `canBh`, `canBv`, `canTh`, `canTv`) are
/// normatively allowed at coding tree node `(bx, by, w, h)` given picture bounds and
/// SPS/SH partition limits (`H.266 §6.4.1–§6.4.3`, VTM `UnitPartitioner.cpp:383-645`).
#[allow(clippy::too_many_arguments)]
pub fn can_split_coding_tree(
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    pic_w: usize,
    pic_h: usize,
    qt_depth: u8,
    mtt_depth: u8,
    min_qt_size: usize,
    max_bt_size: usize,
    max_tt_size: usize,
    max_mtt_depth: u8,
    min_cb_size: usize,
    last_split: CodingTreeSplit,
) -> QtMttCanSplit {
    let right_out = bx + w > pic_w;
    let bottom_out = by + h > pic_h;
    let can_no = !right_out && !bottom_out;

    let can_qt = w == h && mtt_depth == 0 && w > min_qt_size.max(min_cb_size) && w > 4;

    if right_out && bottom_out {
        return QtMttCanSplit {
            can_no: false,
            can_qt: can_qt || (w == h && w > min_cb_size),
            can_bh: !can_qt && h > min_cb_size,
            can_bv: false,
            can_th: false,
            can_tv: false,
        };
    }
    if right_out {
        return QtMttCanSplit {
            can_no: false,
            can_qt,
            can_bh: false,
            can_bv: w > min_cb_size,
            can_th: false,
            can_tv: false,
        };
    }
    if bottom_out {
        return QtMttCanSplit {
            can_no: false,
            can_qt,
            can_bh: h > min_cb_size,
            can_bv: false,
            can_th: false,
            can_tv: false,
        };
    }

    if mtt_depth >= max_mtt_depth {
        return QtMttCanSplit {
            can_no,
            can_qt,
            can_bh: false,
            can_bv: false,
            can_th: false,
            can_tv: false,
        };
    }

    let _ = qt_depth;
    let can_bh = h > min_cb_size && w <= max_bt_size && h <= max_bt_size;
    let can_bv = w > min_cb_size && w <= max_bt_size && h <= max_bt_size;
    let can_th = h >= 4 * min_cb_size
        && w <= max_tt_size
        && h <= max_tt_size
        && last_split != CodingTreeSplit::TernaryHor;
    let can_tv = w >= 4 * min_cb_size
        && w <= max_tt_size
        && h <= max_tt_size
        && last_split != CodingTreeSplit::TernaryVer;

    QtMttCanSplit {
        can_no,
        can_qt,
        can_bh,
        can_bv,
        can_th,
        can_tv,
    }
}

/// Derive the normative CABAC context offsets for a QT/MTT coding tree node
/// (ITU-T Rec. H.266 §9.3.4.2.2, VTM `DeriveCtx::CtxSplit`, `ContextModelling.cpp:170-294`).
pub fn derive_split_contexts(
    w: usize,
    h: usize,
    qt_depth: u8,
    mtt_depth: u8,
    left_height_and_qt: Option<(usize, u8)>,
    above_width_and_qt: Option<(usize, u8)>,
    can: QtMttCanSplit,
) -> QtMttSplitContexts {
    let mut ctx_split = 0usize;
    if let Some((h_left, _)) = left_height_and_qt {
        if h_left < h {
            ctx_split += 1;
        }
    }
    if let Some((w_above, _)) = above_width_and_qt {
        if w_above < w {
            ctx_split += 1;
        }
    }
    let mut num_split = 0usize;
    if can.can_qt {
        num_split += 2;
    }
    if can.can_bh {
        num_split += 1;
    }
    if can.can_bv {
        num_split += 1;
    }
    if can.can_th {
        num_split += 1;
    }
    if can.can_tv {
        num_split += 1;
    }
    if num_split > 0 {
        num_split = num_split.saturating_sub(1);
    }
    ctx_split += 3 * (num_split >> 1);

    let mut ctx_qt = 0usize;
    if let Some((_, qt_left)) = left_height_and_qt {
        if qt_left > qt_depth {
            ctx_qt += 1;
        }
    }
    if let Some((_, qt_above)) = above_width_and_qt {
        if qt_above > qt_depth {
            ctx_qt += 1;
        }
    }
    if qt_depth >= 2 {
        ctx_qt += 3;
    }

    let num_hor = usize::from(can.can_bh) + usize::from(can.can_th);
    let num_ver = usize::from(can.can_bv) + usize::from(can.can_tv);
    let ctx_hv = if num_ver == num_hor {
        match (left_height_and_qt, above_width_and_qt) {
            (Some((h_left, _)), Some((w_above, _))) => {
                let dep_above = w / w_above.max(1);
                let dep_left = h / h_left.max(1);
                if dep_above == dep_left {
                    0
                } else if dep_above < dep_left {
                    1
                } else {
                    2
                }
            }
            _ => 0,
        }
    } else if num_ver < num_hor {
        3
    } else {
        4
    };

    let ctx_hor_bt = if mtt_depth <= 1 { 1 } else { 0 };
    let ctx_ver_bt = if mtt_depth <= 1 { 3 } else { 2 };

    QtMttSplitContexts {
        ctx_split: ctx_split.min(8),
        ctx_qt: ctx_qt.min(5),
        ctx_hv: ctx_hv.min(4),
        ctx_hor_bt,
        ctx_ver_bt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_scan_is_anti_diagonal_with_x_increasing() {
        let scan = diagonal_scan(4, 4);
        assert_eq!(
            scan,
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (0, 2),
                (1, 1),
                (2, 0),
                (0, 3),
                (1, 2),
                (2, 1),
                (3, 0),
                (1, 3),
                (2, 2),
                (3, 1),
                (2, 3),
                (3, 2),
                (3, 3),
            ]
        );
    }

    #[test]
    fn grouped_scan_covers_every_coefficient_once() {
        for size in [8usize, 16, 32] {
            let cs = CoeffScan::new(size, size);
            let mut seen = vec![false; size * size];
            for &idx in &cs.raster {
                assert!(!seen[idx]);
                seen[idx] = true;
            }
            assert!(seen.iter().all(|&v| v));
        }
    }

    #[test]
    fn qp_four_is_lossless_for_transform_skip() {
        for value in [-200i32, -1, 0, 1, 37, 200] {
            let level = quantize_ts(value, 4);
            assert_eq!(level, value);
            assert_eq!(dequantize_ts(level, 4), value);
        }
    }

    #[test]
    fn dequant_matches_reference_table() {
        assert_eq!(dequantize_ts(1, 22), 8);
        assert_eq!(dequantize_ts(-1, 22), -8);
        assert_eq!(dequantize_ts(1000, 63), 32767);
        assert_eq!(dequantize_ts(-1000, 63), -32768);
    }

    #[test]
    fn sbt_tr_types_and_ict_modes_match_spec() {
        assert_eq!(
            derive_sbt_tr_types(SbtIdx::VerHalf, false, 16, 16),
            (TrType::Dct8, TrType::Dst7)
        );
        assert_eq!(
            derive_sbt_tr_types(SbtIdx::HorHalf, false, 16, 16),
            (TrType::Dst7, TrType::Dct8)
        );
        let mut cb = vec![20i32, -40];
        let mut cr = vec![0i32, 0];
        apply_ict_joint_cbcr(&mut cb, &mut cr, -2);
        assert_eq!(cr, vec![-20, 40]);
    }

    #[test]
    fn qt_mtt_splits_preserve_area() {
        for split in [
            CodingTreeSplit::QuadTree,
            CodingTreeSplit::BinaryHor,
            CodingTreeSplit::BinaryVer,
            CodingTreeSplit::TernaryHor,
            CodingTreeSplit::TernaryVer,
        ] {
            let parts = split_coding_tree_node(0, 0, 32, 32, split);
            let total_area: usize = parts.iter().map(|&(_, _, pw, ph)| pw * ph).sum();
            assert_eq!(total_area, 32 * 32);
        }
    }
}
