// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! In-loop filters for H.266/VVC (ITU-T Rec. H.266 §8.8):
//! 1. Luma Mapping with Chroma Scaling (LMCS, §8.8.2, VTM `Reshape.cpp`).
//! 2. Deblocking Filter (DBF, §8.8.3, VTM `DeblockingFilter.cpp`).
//! 3. Sample Adaptive Offset (SAO, §8.8.4, VTM `SampleAdaptiveOffset.cpp`).
//! 4. Adaptive Loop Filter & Cross-Component ALF (ALF / CCALF, §8.8.5, VTM `AdaptiveLoopFilter.cpp`).
//!
//! Algorithms and tables derived from VTM 24.0 (BSD-3-Clause, Copyright (c)
//! 2010-2026 ITU/ISO/IEC). See `THIRD_PARTY.md`.

use super::hls::{VvcAlfAps, VvcLmcsAps};
use super::inter::Mv;
use super::predict::Plane;

// ============================================================================
// Stage 1: Luma Mapping with Chroma Scaling (LMCS, H.266 §8.8.2)
// ============================================================================

/// Number of piecewise-linear bins in LMCS (`PIC_CODE_CW_BINS = 16`, VTM `Reshape.h`).
pub const LMCS_NUM_BINS: usize = 16;
/// Fixed-point bit precision for LMCS forward/inverse scaling (`FP_PREC = 11`, VTM `Reshape.h`).
pub const LMCS_FP_PREC: i32 = 11;

/// Luma Mapping with Chroma Scaling (LMCS) lookup tables and piecewise-linear
/// parameters derived from an LMCS APS (`VvcLmcsAps`).
///
/// Normative source: ITU-T Rec. H.266 §8.8.2 (PDF pp. 421–424);
/// VTM `Reshape.cpp:63-317` (`Reshape::constructReshaper`).
#[derive(Debug, Clone)]
pub struct LmcsReshaper {
    /// Minimum active bin index (`lmcs_min_bin_idx`).
    pub min_bin_idx: usize,
    /// Maximum active bin index (`15 - lmcs_delta_max_bin_idx`).
    pub max_bin_idx: usize,
    /// Codeword counts per bin `bin_cw[0..16]`.
    pub bin_cw: [i32; 16],
    /// Original domain pivot points `input_pivot[0..=16]` (`i * 16` for 8-bit).
    pub input_pivot: [i32; 17],
    /// Reshaped domain pivot points `reshape_pivot[0..=16]`.
    pub reshape_pivot: [i32; 17],
    /// Forward scaling coefficients `fwd_scale_coef[0..16]`.
    pub fwd_scale_coef: [i32; 16],
    /// Inverse scaling coefficients `inv_scale_coef[0..16]`.
    pub inv_scale_coef: [i32; 16],
    /// Chroma residual scaling factors per bin `chroma_adj_lut[0..16]`.
    pub chroma_adj_lut: [i32; 16],
    /// Forward 8-bit sample lookup table `fwd_lut[0..256]`.
    pub fwd_lut: [u8; 256],
    /// Inverse 8-bit sample lookup table `inv_lut[0..256]`.
    pub inv_lut: [u8; 256],
}

impl LmcsReshaper {
    /// Construct an 8-bit `LmcsReshaper` from a parsed `VvcLmcsAps` (`Reshape.cpp:257-317`).
    pub fn from_aps_8bit(aps: &VvcLmcsAps) -> Self {
        let init_cw = 16i32; // 256 / 16
        let min_bin = usize::from(aps.min_bin_idx.min(15));
        let max_bin = usize::from(aps.max_bin_idx.min(15)).max(min_bin);

        let mut bin_cw = [0i32; 16];
        for i in min_bin..=max_bin {
            bin_cw[i] = (init_cw + i32::from(aps.bin_cw_delta[i])).max(0);
        }

        let mut input_pivot = [0i32; 17];
        let mut reshape_pivot = [0i32; 17];
        let mut fwd_scale_coef = [1i32 << LMCS_FP_PREC; 16];
        let mut inv_scale_coef = [1i32 << LMCS_FP_PREC; 16];
        let mut chroma_adj_lut = [1i32 << LMCS_FP_PREC; 16];

        for i in 0..16 {
            reshape_pivot[i + 1] = reshape_pivot[i] + bin_cw[i];
            input_pivot[i + 1] = input_pivot[i] + init_cw;
            // log2(init_cw) = 4
            fwd_scale_coef[i] = ((bin_cw[i] << LMCS_FP_PREC) + (1 << 3)) >> 4;
            if bin_cw[i] == 0 {
                inv_scale_coef[i] = 0;
                chroma_adj_lut[i] = 1 << LMCS_FP_PREC;
            } else {
                inv_scale_coef[i] = (init_cw << LMCS_FP_PREC) / bin_cw[i];
                let denom = (bin_cw[i] + i32::from(aps.chroma_res_scaling_offset)).max(1);
                chroma_adj_lut[i] = (init_cw << LMCS_FP_PREC) / denom;
            }
        }

        let get_pwl_idx_inv = |luma_val: i32| -> usize {
            let mut idx_s = min_bin;
            while idx_s <= max_bin {
                if luma_val < reshape_pivot[idx_s + 1] {
                    break;
                }
                idx_s += 1;
            }
            idx_s.min(15)
        };

        let mut fwd_lut = [0u8; 256];
        let mut inv_lut = [0u8; 256];
        for sample in 0..256i32 {
            let idx_y = (sample / init_cw) as usize;
            let fwd_val = reshape_pivot[idx_y]
                + ((fwd_scale_coef[idx_y] * (sample - input_pivot[idx_y])
                    + (1 << (LMCS_FP_PREC - 1)))
                    >> LMCS_FP_PREC);
            fwd_lut[sample as usize] = fwd_val.clamp(0, 255) as u8;

            let idx_inv = get_pwl_idx_inv(sample);
            let inv_val = input_pivot[idx_inv]
                + ((inv_scale_coef[idx_inv] * (sample - reshape_pivot[idx_inv])
                    + (1 << (LMCS_FP_PREC - 1)))
                    >> LMCS_FP_PREC);
            inv_lut[sample as usize] = inv_val.clamp(0, 255) as u8;
        }

        Self {
            min_bin_idx: min_bin,
            max_bin_idx: max_bin,
            bin_cw,
            input_pivot,
            reshape_pivot,
            fwd_scale_coef,
            inv_scale_coef,
            chroma_adj_lut,
            fwd_lut,
            inv_lut,
        }
    }

    /// Map a luma plane forward into the reshaped domain (`Reshape.cpp:307-311`).
    pub fn apply_forward_plane(&self, plane: &mut Plane) {
        for s in &mut plane.data {
            *s = self.fwd_lut[usize::from(*s)];
        }
    }

    /// Map a reconstructed luma plane from the reshaped domain back to the output
    /// sample domain before deblocking (`H.266 §8.8.2.3`, `Reshape.cpp:313-316`).
    pub fn apply_inverse_plane(&self, plane: &mut Plane) {
        for s in &mut plane.data {
            *s = self.inv_lut[usize::from(*s)];
        }
    }

    /// Derive the chroma residual scaling factor `cScale` from the average reshaped
    /// luma value `avg_luma` (`Reshape::calculateChromaAdj`, `Reshape.cpp:106-110`).
    pub fn chroma_scale_for_luma(&self, avg_luma: i32) -> i32 {
        let mut idx_s = self.min_bin_idx;
        while idx_s <= self.max_bin_idx {
            if avg_luma < self.reshape_pivot[idx_s + 1] {
                break;
            }
            idx_s += 1;
        }
        self.chroma_adj_lut[idx_s.min(15)]
    }

    /// Scale a chroma residual block in-place using the LMCS chroma residual scale
    /// (`H.266 §8.7.5.3 Eq. 1222`).
    pub fn scale_chroma_residual(&self, residual: &mut [i32], avg_luma: i32) {
        let scale = self.chroma_scale_for_luma(avg_luma);
        for r in residual.iter_mut() {
            *r = ((*r * scale) + (1 << (LMCS_FP_PREC - 1))) >> LMCS_FP_PREC;
        }
    }
}

// ============================================================================
// Stage 2: Deblocking Filter (DBF, H.266 §8.8.3)
// ============================================================================

/// Normative deblocking `t_C` lookup table `sm_tcTable[66]` (at 10-bit base precision;
/// for 8-bit video `tc = (DBF_TC_TABLE[idx] + 2) >> 2`).
///
/// Normative source: ITU-T Rec. H.266 §8.8.3.6.2 Table 43 (PDF p. 441);
/// VTM `DeblockingFilter.cpp:62-66`.
pub const DBF_TC_TABLE: [i32; 66] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 4, 5, 5, 5, 5, 7, 7, 8, 9,
    10, 10, 11, 13, 14, 15, 17, 19, 21, 24, 25, 29, 33, 36, 41, 45, 51, 57, 64, 71, 80, 89, 100,
    112, 125, 141, 157, 177, 198, 222, 250, 280, 314, 352, 395,
];

/// Normative deblocking `beta` lookup table `sm_betaTable[64]` (at 8-bit base precision).
///
/// Normative source: ITU-T Rec. H.266 §8.8.3.6.2 Table 43 (PDF p. 441);
/// VTM `DeblockingFilter.cpp:68-71`.
pub const DBF_BETA_TABLE: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66,
    68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
];

/// Derive Boundary Strength `Bs` (`0`, `1`, or `2`) across a 4-sample block boundary
/// between blocks `P` and `Q` (H.266 §8.8.3.5, VTM `DeblockingFilter.cpp:475-620`).
pub fn derive_boundary_strength(
    p_is_intra: bool,
    q_is_intra: bool,
    p_has_cbf: bool,
    q_has_cbf: bool,
    is_tu_edge: bool,
    mv_p: Mv,
    mv_q: Mv,
    ref_poc_p: i32,
    ref_poc_q: i32,
) -> u8 {
    if p_is_intra || q_is_intra {
        2
    } else if is_tu_edge && (p_has_cbf || q_has_cbf) {
        1
    } else if ref_poc_p != ref_poc_q
        || (mv_p.hor - mv_q.hor).abs() >= 8
        || (mv_p.ver - mv_q.ver).abs() >= 8
    {
        // >= 1/2 luma pel (8 in 1/16-pel units)
        1
    } else {
        0
    }
}

/// Luma deblocking filter length for a transform side of `size` samples
/// (H.266 §8.8.3.5; VTM `xSetMaxFilterLengthPQFromTransformSizes`,
/// `DeblockingFilter.cpp:519-640`).
///
/// Returns `7` for a side of at least 32 samples, `3` for 8-16 samples, and `1` when
/// the side is at most 4 samples (which disables luma filtering).
#[inline]
pub fn luma_filter_length(size: usize) -> usize {
    if size <= 4 {
        1
    } else if size >= 32 {
        7
    } else {
        3
    }
}

/// Per-edge luma deblocking decision (ITU-T Rec. H.266 §8.8.3.6.2).
#[derive(Clone, Copy, Debug)]
struct LumaEdgeDecision {
    /// Decision `dE`: `0` = no filtering, `1` = weak, `2` = strong short, `3` = long.
    de: u8,
    /// `dEp` (filter `p1`).
    dep: bool,
    /// `dEq` (filter `q1`).
    deq: bool,
    /// `maxFilterLengthP` as modified by the decision process.
    len_p: usize,
    /// `maxFilterLengthQ` as modified by the decision process.
    len_q: usize,
    /// Threshold `tC`.
    tc: i32,
}

/// `p_i` of a line stored as `[p7, p6, ..., p0, q0, ..., q7]`.
#[inline]
fn p_at(line: &[i32; 16], i: usize) -> i32 {
    line[7 - i]
}

/// `q_j` of a line stored as `[p7, p6, ..., p0, q0, ..., q7]`.
#[inline]
fn q_at(line: &[i32; 16], j: usize) -> i32 {
    line[8 + j]
}

/// Decision process for one luma sample, i.e. `dSam` (H.266 §8.8.3.6.6, Eq. 1369-1374).
#[allow(clippy::too_many_arguments)]
fn luma_sample_decision(
    p0: i32,
    p3: i32,
    q0: i32,
    q3: i32,
    dpq: i32,
    sp_in: i32,
    sq_in: i32,
    spq: i32,
    side_p_large: bool,
    side_q_large: bool,
    beta: i32,
    tc: i32,
) -> bool {
    let mut sp = sp_in;
    let mut sq = sq_in;
    if side_p_large {
        sp = (sp + (p3 - p0).abs() + 1) >> 1;
    }
    if side_q_large {
        sq = (sq + (q3 - q0).abs() + 1) >> 1;
    }
    let (s_thr1, s_thr2) = if side_p_large || side_q_large {
        ((3 * beta) >> 5, beta >> 4)
    } else {
        (beta >> 3, beta >> 2)
    };
    dpq < s_thr2 && sp + sq < s_thr1 && spq < ((5 * tc + 1) >> 1)
}

/// Decision process for a luma block edge (H.266 §8.8.3.6.2, Eq. 1280-1322).
///
/// `lines` holds four consecutive lines `[p7, p6, ..., p0, q0, ..., q7]`; the decision
/// is taken from the first (`k = 0`) and last (`k = 3`) line only. `at_ctb_hor_boundary`
/// is step 6: for a horizontal edge on a CTB row boundary the P side is never large.
fn decide_luma_edge(
    lines: &[[i32; 16]; 4],
    beta: i32,
    tc: i32,
    len_p_in: usize,
    len_q_in: usize,
    at_ctb_hor_boundary: bool,
) -> Option<LumaEdgeDecision> {
    let l0 = &lines[0];
    let l3 = &lines[3];

    let dp0 = (p_at(l0, 2) - 2 * p_at(l0, 1) + p_at(l0, 0)).abs();
    let dp3 = (p_at(l3, 2) - 2 * p_at(l3, 1) + p_at(l3, 0)).abs();
    let dq0 = (q_at(l0, 2) - 2 * q_at(l0, 1) + q_at(l0, 0)).abs();
    let dq3 = (q_at(l3, 2) - 2 * q_at(l3, 1) + q_at(l3, 0)).abs();
    let dpq0 = dp0 + dq0;
    let dpq3 = dp3 + dq3;

    let sp0 = (p_at(l0, 3) - p_at(l0, 0)).abs();
    let sp3 = (p_at(l3, 3) - p_at(l3, 0)).abs();
    let sq0 = (q_at(l0, 0) - q_at(l0, 3)).abs();
    let sq3 = (q_at(l3, 0) - q_at(l3, 3)).abs();
    let spq0 = (p_at(l0, 0) - q_at(l0, 0)).abs();
    let spq3 = (p_at(l3, 0) - q_at(l3, 0)).abs();

    let mut len_p = len_p_in.clamp(1, 7);
    let mut len_q = len_q_in.clamp(1, 7);
    let side_p_large = len_p > 3 && !at_ctb_hor_boundary;
    let side_q_large = len_q > 3;

    // Steps 7-8: long-tap decision.
    if side_p_large || side_q_large {
        let dp0_l = if side_p_large {
            (dp0 + (p_at(l0, 5) - 2 * p_at(l0, 4) + p_at(l0, 3)).abs() + 1) >> 1
        } else {
            dp0
        };
        let dp3_l = if side_p_large {
            (dp3 + (p_at(l3, 5) - 2 * p_at(l3, 4) + p_at(l3, 3)).abs() + 1) >> 1
        } else {
            dp3
        };
        let dq0_l = if side_q_large {
            (dq0 + (q_at(l0, 5) - 2 * q_at(l0, 4) + q_at(l0, 3)).abs() + 1) >> 1
        } else {
            dq0
        };
        let dq3_l = if side_q_large {
            (dq3 + (q_at(l3, 5) - 2 * q_at(l3, 4) + q_at(l3, 3)).abs() + 1) >> 1
        } else {
            dq3
        };
        if !side_p_large {
            len_p = 3;
        }
        if !side_q_large {
            len_q = 3;
        }

        let sp0_l = if len_p == 7 {
            sp0 + (p_at(l0, 7) - p_at(l0, 6) - p_at(l0, 5) + p_at(l0, 4)).abs()
        } else {
            sp0
        };
        let sp3_l = if len_p == 7 {
            sp3 + (p_at(l3, 7) - p_at(l3, 6) - p_at(l3, 5) + p_at(l3, 4)).abs()
        } else {
            sp3
        };
        let sq0_l = if len_q == 7 {
            sq0 + (q_at(l0, 4) - q_at(l0, 5) - q_at(l0, 6) + q_at(l0, 7)).abs()
        } else {
            sq0
        };
        let sq3_l = if len_q == 7 {
            sq3 + (q_at(l3, 4) - q_at(l3, 5) - q_at(l3, 6) + q_at(l3, 7)).abs()
        } else {
            sq3
        };

        let d0_l = dp0_l + dq0_l;
        let d3_l = dp3_l + dq3_l;
        if d0_l + d3_l < beta {
            let d_sam0 = luma_sample_decision(
                if side_p_large { p_at(l0, len_p) } else { 0 },
                if side_p_large { p_at(l0, 3) } else { 0 },
                if side_q_large { q_at(l0, len_q) } else { 0 },
                if side_q_large { q_at(l0, 3) } else { 0 },
                2 * d0_l,
                sp0_l,
                sq0_l,
                spq0,
                side_p_large,
                side_q_large,
                beta,
                tc,
            );
            let d_sam3 = luma_sample_decision(
                if side_p_large { p_at(l3, len_p) } else { 0 },
                if side_p_large { p_at(l3, 3) } else { 0 },
                if side_q_large { q_at(l3, len_q) } else { 0 },
                if side_q_large { q_at(l3, 3) } else { 0 },
                2 * d3_l,
                sp3_l,
                sq3_l,
                spq3,
                side_p_large,
                side_q_large,
                beta,
                tc,
            );
            if d_sam0 && d_sam3 {
                return Some(LumaEdgeDecision {
                    de: 3,
                    dep: true,
                    deq: true,
                    len_p,
                    len_q,
                    tc,
                });
            }
        }
    }

    // Step 9: short-filter decision.
    let dp = dp0 + dp3;
    let dq = dq0 + dq3;
    if dpq0 + dpq3 >= beta {
        return None;
    }
    let mut d_sam0 = false;
    let mut d_sam3 = false;
    if len_p > 2 && len_q > 2 {
        d_sam0 = luma_sample_decision(0, 0, 0, 0, 2 * dpq0, sp0, sq0, spq0, false, false, beta, tc);
        d_sam3 = luma_sample_decision(0, 0, 0, 0, 2 * dpq3, sp3, sq3, spq3, false, false, beta, tc);
    }
    let mut de = 1;
    if d_sam0 && d_sam3 {
        de = 2;
        len_p = 3;
        len_q = 3;
    }
    let side_threshold = (beta + (beta >> 1)) >> 3;
    let dep = len_p > 1 && len_q > 1 && dp < side_threshold;
    let deq = len_p > 1 && len_q > 1 && dq < side_threshold;
    if de == 1 {
        len_p = 1 + usize::from(dep);
        len_q = 1 + usize::from(deq);
    }
    Some(LumaEdgeDecision {
        de,
        dep,
        deq,
        len_p,
        len_q,
        tc,
    })
}

/// Filtering process for one luma sample using long filters (H.266 §8.8.3.6.8,
/// Eq. 1389-1410; VTM `xFilteringPandQ`, `DeblockingFilter.cpp:1389-1475`).
///
/// H.266 V4 Eq. 1393/1394 for the asymmetric `7/3` and `3/7` cases describe a
/// different `refMiddle` than VTM 24.0 and VVdeC 3.3, which agree with each other;
/// this implementation follows the reference decoders so that reconstructed samples
/// stay byte-identical to them.
fn long_tap_luma_line(line: &mut [i32; 16], len_p: usize, len_q: usize, tc: i32, max_val: i32) {
    let m = *line;
    let p = |i: usize| m[7 - i];
    let q = |j: usize| m[8 + j];
    let (np, nq) = (len_p.min(7), len_q.min(7));
    let ref_middle = match (np, nq) {
        (5, 5) => {
            (2 * (p(0) + q(0) + p(1) + q(1) + p(2) + q(2)) + p(3) + q(3) + p(4) + q(4) + 8) >> 4
        }
        (7, 5) | (5, 7) => {
            (2 * (p(0) + q(0) + p(1) + q(1))
                + p(2)
                + q(2)
                + p(3)
                + q(3)
                + p(4)
                + q(4)
                + p(5)
                + q(5)
                + 8)
                >> 4
        }
        (5, 3) | (3, 5) => (p(0) + q(0) + p(1) + q(1) + p(2) + q(2) + p(3) + q(3) + 4) >> 3,
        (7, 3) => {
            (2 * (p(0) + q(0))
                + q(0)
                + 2 * (q(1) + q(2))
                + p(1)
                + q(1)
                + p(2)
                + p(3)
                + p(4)
                + p(5)
                + p(6)
                + 8)
                >> 4
        }
        (3, 7) => {
            (2 * (p(0) + q(0))
                + p(0)
                + 2 * (p(1) + p(2))
                + p(1)
                + q(1)
                + q(2)
                + q(3)
                + q(4)
                + q(5)
                + q(6)
                + 8)
                >> 4
        }
        _ => {
            debug_assert_eq!((np, nq), (7, 7));
            (2 * (p(0) + q(0))
                + p(1)
                + q(1)
                + p(2)
                + q(2)
                + p(3)
                + q(3)
                + p(4)
                + q(4)
                + p(5)
                + q(5)
                + p(6)
                + q(6)
                + 8)
                >> 4
        }
    };

    let ref_p = (p(np) + p(np - 1) + 1) >> 1;
    let ref_q = (q(nq) + q(nq - 1) + 1) >> 1;

    let coeffs = |len: usize| -> &'static [i32] {
        match len {
            7 => &[59, 50, 41, 32, 23, 14, 5],
            5 => &[58, 45, 32, 19, 6],
            _ => &[53, 32, 11],
        }
    };
    let tc_mult = |len: usize| -> &'static [i32] {
        match len {
            7 => &[6, 5, 4, 3, 2, 1, 1],
            5 => &[6, 5, 4, 3, 2],
            _ => &[6, 4, 2],
        }
    };

    let cp = coeffs(np);
    let cq = coeffs(nq);
    let tcp = tc_mult(np);
    let tcq = tc_mult(nq);
    for (pos, (&f, &t)) in cp.iter().zip(tcp.iter()).enumerate() {
        let orig = p(pos);
        let cval = (tc * t) >> 1;
        let interp = (ref_middle * f + ref_p * (64 - f) + 32) >> 6;
        line[7 - pos] = interp.clamp(orig - cval, orig + cval).clamp(0, max_val);
    }
    for (pos, (&g, &t)) in cq.iter().zip(tcq.iter()).enumerate() {
        let orig = q(pos);
        let cval = (tc * t) >> 1;
        let interp = (ref_middle * g + ref_q * (64 - g) + 32) >> 6;
        line[8 + pos] = interp.clamp(orig - cval, orig + cval).clamp(0, max_val);
    }
}

/// Short-filter part of the luma filtering process (H.266 §8.8.3.6.7, Eq. 1375-1388)
/// for one line of `[p7, ..., p0, q0, ..., q7]`.
fn short_filter_luma_line(line: &mut [i32; 16], edge: &LumaEdgeDecision, max_val: i32) {
    let m = *line;
    let tc = edge.tc;
    if edge.de == 2 {
        line[7] =
            ((p_at(&m, 2) + 2 * p_at(&m, 1) + 2 * p_at(&m, 0) + 2 * q_at(&m, 0) + q_at(&m, 1) + 4)
                >> 3)
                .clamp(p_at(&m, 0) - 3 * tc, p_at(&m, 0) + 3 * tc);
        line[6] = ((p_at(&m, 2) + p_at(&m, 1) + p_at(&m, 0) + q_at(&m, 0) + 2) >> 2)
            .clamp(p_at(&m, 1) - 2 * tc, p_at(&m, 1) + 2 * tc);
        line[5] =
            ((2 * p_at(&m, 3) + 3 * p_at(&m, 2) + p_at(&m, 1) + p_at(&m, 0) + q_at(&m, 0) + 4)
                >> 3)
                .clamp(p_at(&m, 2) - tc, p_at(&m, 2) + tc);
        line[8] =
            ((p_at(&m, 1) + 2 * p_at(&m, 0) + 2 * q_at(&m, 0) + 2 * q_at(&m, 1) + q_at(&m, 2) + 4)
                >> 3)
                .clamp(q_at(&m, 0) - 3 * tc, q_at(&m, 0) + 3 * tc);
        line[9] = ((p_at(&m, 0) + q_at(&m, 0) + q_at(&m, 1) + q_at(&m, 2) + 2) >> 2)
            .clamp(q_at(&m, 1) - 2 * tc, q_at(&m, 1) + 2 * tc);
        line[10] =
            ((p_at(&m, 0) + q_at(&m, 0) + q_at(&m, 1) + 3 * q_at(&m, 2) + 2 * q_at(&m, 3) + 4)
                >> 3)
                .clamp(q_at(&m, 2) - tc, q_at(&m, 2) + tc);
        return;
    }

    let delta = (9 * (q_at(&m, 0) - p_at(&m, 0)) - 3 * (q_at(&m, 1) - p_at(&m, 1)) + 8) >> 4;
    if delta.abs() >= tc * 10 {
        return;
    }
    let delta = delta.clamp(-tc, tc);
    line[7] = (p_at(&m, 0) + delta).clamp(0, max_val);
    line[8] = (q_at(&m, 0) - delta).clamp(0, max_val);
    let tc2 = tc >> 1;
    if edge.dep {
        let delta_p =
            ((((p_at(&m, 2) + p_at(&m, 0) + 1) >> 1) - p_at(&m, 1) + delta) >> 1).clamp(-tc2, tc2);
        line[6] = (p_at(&m, 1) + delta_p).clamp(0, max_val);
    }
    if edge.deq {
        let delta_q =
            ((((q_at(&m, 2) + q_at(&m, 0) + 1) >> 1) - q_at(&m, 1) - delta) >> 1).clamp(-tc2, tc2);
        line[9] = (q_at(&m, 1) + delta_q).clamp(0, max_val);
    }
}

/// Run the H.266 §8.8.3.6.2-§8.8.3.6.8 luma deblocking process on a 4-line edge segment.
///
/// `lines` holds `[p7, p6, ..., p0, q0, ..., q7]` for each of the four lines. `bit_depth`
/// selects the sample range and the `tC`/`β` scaling.
#[allow(clippy::too_many_arguments)]
fn deblock_luma_edge_bd(
    lines: &mut [[i32; 16]; 4],
    qp_p: i32,
    qp_q: i32,
    bs: u8,
    beta_offset_div2: i8,
    tc_offset_div2: i8,
    bit_depth: u8,
    max_len_p: usize,
    max_len_q: usize,
    at_ctb_hor_boundary: bool,
) {
    if bs == 0 {
        return;
    }
    let bd = i32::from(bit_depth.clamp(8, 12));
    let max_val = (1i32 << bd) - 1;
    let qp = (qp_p + qp_q + 1) >> 1;
    let idx_tc =
        (qp + 2 * (i32::from(bs) - 1) + 2 * i32::from(tc_offset_div2)).clamp(0, 65) as usize;
    let idx_b = (qp + 2 * i32::from(beta_offset_div2)).clamp(0, 63) as usize;
    let tc = if bd < 10 {
        (DBF_TC_TABLE[idx_tc] + (1 << (9 - bd))) >> (10 - bd)
    } else {
        DBF_TC_TABLE[idx_tc] << (bd - 10)
    };
    let beta = DBF_BETA_TABLE[idx_b] << (bd - 8);
    if beta == 0 || tc == 0 {
        return;
    }
    let Some(edge) = decide_luma_edge(lines, beta, tc, max_len_p, max_len_q, at_ctb_hor_boundary)
    else {
        return;
    };
    for line in lines.iter_mut() {
        if edge.de == 3 {
            long_tap_luma_line(line, edge.len_p, edge.len_q, edge.tc, max_val);
        } else {
            short_filter_luma_line(line, &edge, max_val);
        }
    }
}

/// Apply the normative luma deblocking filter to a 4-line edge segment whose lines are
/// supplied as `[p7, p6, ..., p0, q0, ..., q7]` (H.266 §8.8.3.6.2-§8.8.3.6.8).
///
/// `max_len_p`/`max_len_q` are the maximum filter lengths derived from the transform
/// sizes (`7` for a transform side of at least 32 samples, `3` for 8-16, `1` below);
/// `at_ctb_hor_boundary` disables the long filter on the P side of a horizontal edge
/// that lies on a CTB row boundary (step 6 of the decision process).
#[allow(clippy::too_many_arguments)]
pub fn deblock_luma_edge_normative(
    lines: &mut [[i32; 16]; 4],
    qp_p: u8,
    qp_q: u8,
    bs: u8,
    beta_offset_div2: i8,
    tc_offset_div2: i8,
    max_len_p: usize,
    max_len_q: usize,
    at_ctb_hor_boundary: bool,
) {
    deblock_luma_edge_bd(
        lines,
        i32::from(qp_p),
        i32::from(qp_q),
        bs,
        beta_offset_div2,
        tc_offset_div2,
        8,
        max_len_p,
        max_len_q,
        at_ctb_hor_boundary,
    );
}

/// Filter a 4-line luma edge segment (`lines` of 8 samples `[p3, p2, p1, p0, q0, q1, q2, q3]`
/// per line) in-place using the H.266 §8.8.3.6.2-§8.8.3.6.7 short/long deblocking filters.
///
/// Equivalent to [`deblock_luma_edge_normative`] with `maxFilterLengthP = maxFilterLengthQ = 3`,
/// i.e. the long filter is never selected.
pub fn deblock_luma_edge_4(
    lines: &mut [[i32; 8]; 4],
    qp_p: u8,
    qp_q: u8,
    bs: u8,
    beta_offset_div2: i8,
    tc_offset_div2: i8,
) {
    let mut wide = [[0i32; 16]; 4];
    for (dst, src) in wide.iter_mut().zip(lines.iter()) {
        dst[4..12].copy_from_slice(src);
    }
    deblock_luma_edge_normative(
        &mut wide,
        qp_p,
        qp_q,
        bs,
        beta_offset_div2,
        tc_offset_div2,
        3,
        3,
        false,
    );
    for (dst, src) in lines.iter_mut().zip(wide.iter()) {
        dst.copy_from_slice(&src[4..12]);
    }
}

/// Filter a long-tap luma edge line of 16 samples `[p7..p0, q0..q7]` in-place
/// (H.266 §8.8.3.6.8; VTM `xFilteringPandQ`, `DeblockingFilter.cpp:1389-1475`).
pub fn deblock_luma_long_tap_line(line: &mut [i32; 16], len_p: usize, len_q: usize, tc: i32) {
    long_tap_luma_line(line, len_p, len_q, tc, 255);
}

/// Filter a 4-sample chroma edge `[p1, p0, q0, q1]` using the H.266 §8.8.3.6.3
/// normal chroma deblocking filter (`DeblockingFilter.cpp:1613-1617`).
pub fn deblock_chroma_sample(
    p1: i32,
    p0: &mut i32,
    q0: &mut i32,
    q1: i32,
    qp_p: u8,
    qp_q: u8,
    bs: u8,
    tc_offset_div2: i8,
) {
    if bs < 2 {
        return;
    }
    let qp = (i32::from(qp_p) + i32::from(qp_q) + 1) >> 1;
    let idx_tc =
        (qp + 2 * (i32::from(bs) - 1) + 2 * i32::from(tc_offset_div2)).clamp(0, 65) as usize;
    let tc = (DBF_TC_TABLE[idx_tc] + 2) >> 2;
    if tc == 0 {
        return;
    }
    let delta = ((4 * (*q0 - *p0) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
    *p0 = (*p0 + delta).clamp(0, 255);
    *q0 = (*q0 - delta).clamp(0, 255);
}

/// Apply vertical-edge then horizontal-edge deblocking across an entire luma `Plane`.
///
/// Edges are filtered on the coding-block lattice `cu_step`; the transform side is
/// `cu_step` on both sides, so the maximum filter length follows
/// [`luma_filter_length`]. `ctb_size` is `CtbSizeY` and is only used for the step-6
/// rule that disables the long filter on the P side of a horizontal CTB row boundary
/// (H.266 §8.8.3.6.2).
pub fn deblock_luma_plane(
    plane: &mut Plane,
    cu_step: usize,
    ctb_size: usize,
    qp: u8,
    bs: u8,
    beta_offset_div2: i8,
    tc_offset_div2: i8,
) {
    let step = cu_step.max(8);
    let ctb = ctb_size.max(step);
    let max_len = luma_filter_length(step);
    let w = plane.width;
    let h = plane.height;

    // 1. Vertical edges at x = step, 2*step, ...
    let mut x = step;
    while x + 8 <= w && x >= 8 {
        let mut y = 0;
        while y + 4 <= h {
            let mut lines = [[0i32; 16]; 4];
            for dy in 0..4 {
                for k in 0..16 {
                    lines[dy][k] = i32::from(plane.get(x - 8 + k, y + dy));
                }
            }
            deblock_luma_edge_normative(
                &mut lines,
                qp,
                qp,
                bs,
                beta_offset_div2,
                tc_offset_div2,
                max_len,
                max_len,
                false,
            );
            for dy in 0..4 {
                for k in 0..16 {
                    plane.set(x - 8 + k, y + dy, lines[dy][k].clamp(0, 255) as u8);
                }
            }
            y += 4;
        }
        x += step;
    }

    // 2. Horizontal edges at y = step, 2*step, ...
    let mut y = step;
    while y + 8 <= h && y >= 8 {
        let at_ctb_row_boundary = y.is_multiple_of(ctb);
        let mut bx = 0;
        while bx + 4 <= w {
            let mut lines = [[0i32; 16]; 4];
            for dx in 0..4 {
                for k in 0..16 {
                    lines[dx][k] = i32::from(plane.get(bx + dx, y - 8 + k));
                }
            }
            deblock_luma_edge_normative(
                &mut lines,
                qp,
                qp,
                bs,
                beta_offset_div2,
                tc_offset_div2,
                max_len,
                max_len,
                at_ctb_row_boundary,
            );
            for dx in 0..4 {
                for k in 0..16 {
                    plane.set(bx + dx, y - 8 + k, lines[dx][k].clamp(0, 255) as u8);
                }
            }
            bx += 4;
        }
        y += step;
    }
}

/// Chroma deblocking filter length for a chroma transform side of `size` samples
/// (H.266 §8.8.3.5): `3` when the side is at least 8 samples, otherwise `1`.
#[inline]
pub fn chroma_filter_length(size: usize) -> usize {
    if size >= 8 { 3 } else { 1 }
}

/// Decision process for a chroma sample (`dSam`, H.266 §8.8.3.6.9).
#[inline]
fn chroma_sample_decision(
    p0: i32,
    p3: i32,
    q0: i32,
    q3: i32,
    dpq: i32,
    beta: i32,
    tc: i32,
) -> bool {
    dpq < (beta >> 2)
        && (p3 - p0).abs() + (q0 - q3).abs() < (beta >> 3)
        && (p0 - q0).abs() < ((5 * tc + 1) >> 1)
}

/// Normative chroma deblocking for one sample group of a coding-block edge
/// (H.266 §8.8.3.6.4, §8.8.3.6.9, §8.8.3.6.10).
///
/// Each line is `[p3, p2, p1, p0, q0, q1, q2, q3]`. `lines` holds two lines when
/// `subSampleC == 2` (`4:2:0` vertical edges and `4:2:2` horizontal edges) and
/// four lines when `subSampleC == 1` (`4:4:4`, and `4:2:2` vertical edges).
///
/// `max_len_p`/`max_len_q` are `maxFilterLengthP`/`maxFilterLengthQ` from
/// H.266 §8.8.3.5: `3` when both transform sides are at least 8 chroma samples,
/// `1` otherwise. On a horizontal chroma CTB boundary the P side is `1` and the
/// Q side `3`, which is exactly the asymmetric `1/3` filter of Eq. 1417-1420.
#[allow(clippy::too_many_arguments)]
pub fn deblock_chroma_edge(
    lines: &mut [[i32; 8]],
    qp_chroma: i32,
    bs: u8,
    beta_offset_div2: i8,
    tc_offset_div2: i8,
    max_len_p: usize,
    max_len_q: usize,
    bit_depth: u8,
) {
    if bs == 0 || !matches!(lines.len(), 2 | 4) {
        return;
    }
    let mut len_p = max_len_p.min(3);
    let mut len_q = max_len_q.min(3);
    // H.266 §8.8.3.6.4: `maxFilterLengthP`/`Q` both become 0 (no filtering)
    // when both are 1 and `bS` is not 2.
    if len_p == 1 && len_q == 1 && bs != 2 {
        return;
    }

    let bd = i32::from(bit_depth.clamp(8, 12));
    let max_val = (1i32 << bd) - 1;
    let idx_tc =
        (qp_chroma + 2 * (i32::from(bs) - 1) + 2 * i32::from(tc_offset_div2)).clamp(0, 65) as usize;
    let idx_b = (qp_chroma + 2 * i32::from(beta_offset_div2)).clamp(0, 63) as usize;
    let tc = if bd < 10 {
        (DBF_TC_TABLE[idx_tc] + (1 << (9 - bd))) >> (10 - bd)
    } else {
        DBF_TC_TABLE[idx_tc] << (bd - 10)
    };
    let beta = DBF_BETA_TABLE[idx_b] << (bd - 8);

    let n1 = lines.len() - 1;
    if len_q == 3 {
        // Step 2: when the P side cannot use three samples, p3 and p2 are set
        // equal to p1 for the decision only.
        let mut line0 = lines[0];
        let mut line1 = lines[n1];
        if len_p == 1 {
            line0[0] = line0[2];
            line0[1] = line0[2];
            line1[0] = line1[2];
            line1[1] = line1[2];
        }
        let dp0 = (line0[1] - 2 * line0[2] + line0[3]).abs();
        let dp1 = (line1[1] - 2 * line1[2] + line1[3]).abs();
        let dq0 = (line0[6] - 2 * line0[5] + line0[4]).abs();
        let dq1 = (line1[6] - 2 * line1[5] + line1[4]).abs();
        let dpq0 = dp0 + dq0;
        let dpq1 = dp1 + dq1;

        let strong = (dpq0 + dpq1) < beta
            && chroma_sample_decision(line0[3], line0[0], line0[4], line0[7], 2 * dpq0, beta, tc)
            && chroma_sample_decision(line1[3], line1[0], line1[4], line1[7], 2 * dpq1, beta, tc);
        if !strong {
            // Step 6: `dSam0 == 0 || dSam1 == 0` resets both lengths to 1.
            len_p = 1;
            len_q = 1;
        }
    }

    for line in lines.iter_mut() {
        match (len_p, len_q) {
            (3, 3) => chroma_strong_line(line, tc, max_val),
            (1, 3) => chroma_weak_line_p1q3(line, tc, max_val),
            _ => chroma_weak_line(line, tc, max_val),
        }
    }
}

/// Strong chroma filter, H.266 Eq. 1411-1416.
fn chroma_strong_line(line: &mut [i32; 8], tc: i32, max_val: i32) {
    let m = *line;
    let clip = |orig: i32, val: i32| val.clamp(orig - tc, orig + tc).clamp(0, max_val);
    line[1] = clip(m[1], (3 * m[0] + 2 * m[1] + m[2] + m[3] + m[4] + 4) >> 3);
    line[2] = clip(
        m[2],
        (2 * m[0] + m[1] + 2 * m[2] + m[3] + m[4] + m[5] + 4) >> 3,
    );
    line[3] = clip(
        m[3],
        (m[0] + m[1] + m[2] + 2 * m[3] + m[4] + m[5] + m[6] + 4) >> 3,
    );
    line[4] = clip(
        m[4],
        (m[1] + m[2] + m[3] + 2 * m[4] + m[5] + m[6] + m[7] + 4) >> 3,
    );
    line[5] = clip(
        m[5],
        (m[2] + m[3] + m[4] + 2 * m[5] + m[6] + 2 * m[7] + 4) >> 3,
    );
    line[6] = clip(m[6], (m[3] + m[4] + m[5] + 2 * m[6] + 3 * m[7] + 4) >> 3);
}

/// Asymmetric chroma filter with `maxFilterLengthP = 1`, `maxFilterLengthQ = 3`,
/// H.266 Eq. 1417-1420.
fn chroma_weak_line_p1q3(line: &mut [i32; 8], tc: i32, max_val: i32) {
    let m = *line;
    let clip = |orig: i32, val: i32| val.clamp(orig - tc, orig + tc).clamp(0, max_val);
    line[3] = clip(m[3], (3 * m[2] + 2 * m[3] + m[4] + m[5] + m[6] + 4) >> 3);
    line[4] = clip(
        m[4],
        (2 * m[2] + m[3] + 2 * m[4] + m[5] + m[6] + m[7] + 4) >> 3,
    );
    line[5] = clip(
        m[5],
        (m[2] + m[3] + m[4] + 2 * m[5] + m[6] + 2 * m[7] + 4) >> 3,
    );
    line[6] = clip(m[6], (m[3] + m[4] + m[5] + 2 * m[6] + 3 * m[7] + 4) >> 3);
}

/// Weak chroma filter, H.266 Eq. 1421-1423.
fn chroma_weak_line(line: &mut [i32; 8], tc: i32, max_val: i32) {
    let (p1, p0, q0, q1) = (line[2], line[3], line[4], line[5]);
    let delta = ((4 * (q0 - p0) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
    line[3] = (p0 + delta).clamp(0, max_val);
    line[4] = (q0 - delta).clamp(0, max_val);
}

/// Deblock one chroma block edge inside a `Plane` for the sample range `start..end`
/// along the edge (H.266 §8.8.3.6.4-§8.8.3.6.10).
///
/// `vertical` selects a vertical edge at column `edge` (rows `start..end`) or a
/// horizontal edge at row `edge` (columns `start..end`). `sub_sample_c` is
/// `SubHeightC` for a vertical edge and `SubWidthC` for a horizontal edge and
/// selects whether a decision group spans two or four lines.
#[allow(clippy::too_many_arguments)]
pub fn deblock_chroma_edge_region(
    plane: &mut Plane,
    vertical: bool,
    edge: usize,
    start: usize,
    end: usize,
    qp_chroma: i32,
    bs: u8,
    beta_offset_div2: i8,
    tc_offset_div2: i8,
    max_len_p: usize,
    max_len_q: usize,
    sub_sample_c: usize,
) {
    let group = if sub_sample_c == 2 { 2 } else { 4 };
    let limit = if vertical { plane.width } else { plane.height };
    if edge < 4 || edge + 4 > limit {
        return;
    }
    let mut pos = start;
    while pos + group <= end {
        let mut lines = [[0i32; 8]; 4];
        for (k, line) in lines.iter_mut().take(group).enumerate() {
            for (i, sample) in line.iter_mut().enumerate() {
                let d = edge - 4 + i;
                *sample = i32::from(if vertical {
                    plane.get(d, pos + k)
                } else {
                    plane.get(pos + k, d)
                });
            }
        }
        deblock_chroma_edge(
            &mut lines[..group],
            qp_chroma,
            bs,
            beta_offset_div2,
            tc_offset_div2,
            max_len_p,
            max_len_q,
            8,
        );
        for (k, line) in lines.iter().take(group).enumerate() {
            for (i, &sample) in line.iter().enumerate() {
                let d = edge - 4 + i;
                if vertical {
                    plane.set(d, pos + k, sample.clamp(0, 255) as u8);
                } else {
                    plane.set(pos + k, d, sample.clamp(0, 255) as u8);
                }
            }
        }
        pos += group;
    }
}

/// Apply vertical-edge then horizontal-edge deblocking across a chroma `Plane`.
///
/// `cu_step` is the chroma coding-block size, `ctb_height_c` is `CtbHeightC`, and
/// `sub_width_c`/`sub_height_c` are `SubWidthC`/`SubHeightC`. The pair that forms
/// one decision group is two lines for `SubHeightC == 2` vertical edges and two
/// columns for `SubWidthC == 2` horizontal edges; otherwise a group is four.
#[allow(clippy::too_many_arguments)]
pub fn deblock_chroma_plane(
    plane: &mut Plane,
    cu_step: usize,
    ctb_height_c: usize,
    qp_chroma: u8,
    bs: u8,
    beta_offset_div2: i8,
    tc_offset_div2: i8,
    sub_width_c: usize,
    sub_height_c: usize,
) {
    let step = cu_step.max(8);
    let ctb = ctb_height_c.max(step);
    let len = chroma_filter_length(step);
    let w = plane.width;
    let h = plane.height;
    let qp = i32::from(qp_chroma);
    let group_v = if sub_height_c == 1 { 4 } else { 2 };
    let group_h = if sub_width_c == 1 { 4 } else { 2 };

    let mut x = step;
    while x >= 4 && x + 4 <= w {
        let mut y = 0;
        while y + group_v <= h {
            let mut lines = [[0i32; 8]; 4];
            for (k, line) in lines.iter_mut().take(group_v).enumerate() {
                for (i, sample) in line.iter_mut().enumerate() {
                    let xx = x - 4 + i;
                    *sample = i32::from(plane.get(xx, y + k));
                }
            }
            deblock_chroma_edge(
                &mut lines[..group_v],
                qp,
                bs,
                beta_offset_div2,
                tc_offset_div2,
                len,
                len,
                8,
            );
            for (k, line) in lines.iter().take(group_v).enumerate() {
                for (i, &sample) in line.iter().enumerate() {
                    let xx = x - 4 + i;
                    plane.set(xx, y + k, sample.clamp(0, 255) as u8);
                }
            }
            y += group_v;
        }
        x += step;
    }

    let mut y = step;
    while y >= 4 && y + 4 <= h {
        let (len_p, len_q) = if y.is_multiple_of(ctb) {
            (1, len)
        } else {
            (len, len)
        };
        let mut bx = 0;
        while bx + group_h <= w {
            let mut lines = [[0i32; 8]; 4];
            for (k, line) in lines.iter_mut().take(group_h).enumerate() {
                for (i, sample) in line.iter_mut().enumerate() {
                    let yy = y - 4 + i;
                    *sample = i32::from(plane.get(bx + k, yy));
                }
            }
            deblock_chroma_edge(
                &mut lines[..group_h],
                qp,
                bs,
                beta_offset_div2,
                tc_offset_div2,
                len_p,
                len_q,
                8,
            );
            for (k, line) in lines.iter().take(group_h).enumerate() {
                for (i, &sample) in line.iter().enumerate() {
                    let yy = y - 4 + i;
                    plane.set(bx + k, yy, sample.clamp(0, 255) as u8);
                }
            }
            bx += group_h;
        }
        y += step;
    }
}

// ============================================================================
// Stage 3: Sample Adaptive Offset (SAO, H.266 §8.8.4)
// ============================================================================

/// Sample Adaptive Offset (SAO) filter mode for one colour component in a CTU
/// (H.266 §8.8.4, VTM `SampleAdaptiveOffset.cpp:285-480`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaoType {
    /// SAO disabled (`sao_type_idx == 0`).
    #[default]
    Off,
    /// Band Offset (`sao_type_idx == 1`): starting band `band_pos` (`0..32`) and 4 signed offsets.
    BandOffset {
        /// Starting band position (`0..32`).
        band_pos: u8,
        /// Signed offsets for bands `(band_pos + i) % 32` (`i in 0..4`).
        offsets: [i32; 4],
    },
    /// Edge Offset (`sao_type_idx == 2`): directional class `eo_class` (`0..4`: `0 deg`,
    /// `90 deg`, `135 deg`, `45 deg`) and 4 signed offsets for categories `1, 2, 3, 4`
    /// (`edgeIdx` `0, 1, 3, 4` where `edgeIdx = 2 + sign(c - a) + sign(c - b)`).
    EdgeOffset {
        /// Edge direction (`0` = horizontal, `1` = vertical, `2` = 135° diag, `3` = 45° diag).
        eo_class: u8,
        /// Signed offsets `[cat1 (>=0), cat2 (>=0), cat3 (<=0), cat4 (<=0)]`.
        offsets: [i32; 4],
    },
}

/// Apply SAO to a CTU region `(bx, by, w, h)` reading from `src` and writing to `dst`
/// (`SampleAdaptiveOffset::offsetBlock`, `SampleAdaptiveOffset.cpp:285-540`).
pub fn apply_sao_region(
    src: &Plane,
    dst: &mut Plane,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    sao: SaoType,
) {
    match sao {
        SaoType::Off => {}
        SaoType::BandOffset { band_pos, offsets } => {
            let mut band_table = [0i32; 32];
            for i in 0..4 {
                band_table[((band_pos as usize) + i) & 31] = offsets[i];
            }
            for y in by..(by + h).min(src.height) {
                for x in bx..(bx + w).min(src.width) {
                    let s = i32::from(src.get(x, y));
                    let band = (s >> 3) as usize & 31;
                    dst.set(x, y, (s + band_table[band]).clamp(0, 255) as u8);
                }
            }
        }
        SaoType::EdgeOffset { eo_class, offsets } => {
            // `edge_idx = 2 + sign(c - a) + sign(c - b)` in `0..=4`:
            // 0 -> cat 1 (offsets[0]), 1 -> cat 2 (offsets[1]), 2 -> cat 0 (0),
            // 3 -> cat 3 (offsets[2]), 4 -> cat 4 (offsets[3]).
            let eo_lut = [offsets[0], offsets[1], 0, offsets[2], offsets[3]];
            let (dx, dy): (isize, isize) = match eo_class & 3 {
                0 => (1, 0),
                1 => (0, 1),
                2 => (1, 1),
                _ => (-1, 1),
            };
            let pw = src.width as isize;
            let ph = src.height as isize;

            for y in by..(by + h).min(src.height) {
                for x in bx..(bx + w).min(src.width) {
                    let ix = x as isize;
                    let iy = y as isize;
                    let ax = ix - dx;
                    let ay = iy - dy;
                    let bx_n = ix + dx;
                    let by_n = iy + dy;
                    if ax < 0
                        || ay < 0
                        || ax >= pw
                        || ay >= ph
                        || bx_n < 0
                        || by_n < 0
                        || bx_n >= pw
                        || by_n >= ph
                    {
                        continue;
                    }
                    let c = i32::from(src.get(x, y));
                    let a = i32::from(src.get(ax as usize, ay as usize));
                    let b = i32::from(src.get(bx_n as usize, by_n as usize));
                    let edge_idx = (2 + (c - a).signum() + (c - b).signum()) as usize;
                    dst.set(x, y, (c + eo_lut[edge_idx]).clamp(0, 255) as u8);
                }
            }
        }
    }
}

/// Apply SAO across an entire plane given per-CTU `SaoType` parameters.
pub fn apply_sao_plane(plane: &mut Plane, ctu_size: usize, ctu_params: &[SaoType]) {
    if ctu_params.iter().all(|&p| p == SaoType::Off) {
        return;
    }
    let src = plane.clone();
    let cols = plane.width.div_ceil(ctu_size);
    let rows = plane.height.div_ceil(ctu_size);
    for cy in 0..rows {
        for cx in 0..cols {
            let idx = cy * cols + cx;
            if let Some(&sao) = ctu_params.get(idx) {
                let bx = cx * ctu_size;
                let by = cy * ctu_size;
                let w = (plane.width - bx).min(ctu_size);
                let h = (plane.height - by).min(ctu_size);
                apply_sao_region(&src, plane, bx, by, w, h, sao);
            }
        }
    }
}

// ============================================================================
// Stage 4: Adaptive Loop Filter & Cross-Component ALF (ALF/CCALF, H.266 §8.8.5)
// ============================================================================

/// Normative 64 fixed 7x7 luma ALF filter coefficient sets (`m_fixedFilterSetCoeff[64][13]`,
/// H.266 §8.8.5.2 Table 49, VTM `AdaptiveLoopFilter.cpp:212-245`).
pub const ALF_FIXED_FILTER_COEFF: [[i16; 13]; 64] = [
    [0, 0, 2, -3, 1, -4, 1, 7, -1, 1, -1, 5, 0],
    [0, 0, 0, 0, 0, -1, 0, 1, 0, 0, -1, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
    [2, 2, -7, -3, 0, -5, 13, 22, 12, -3, -3, 17, 0],
    [-1, 0, 6, -8, 1, -5, 1, 23, 0, 2, -5, 10, 0],
    [0, 0, -1, -1, 0, -1, 2, 1, 0, 0, -1, 4, 0],
    [0, 0, 3, -11, 1, 0, -1, 35, 5, 2, -9, 9, 0],
    [0, 0, 8, -8, -2, -7, 4, 4, 2, 1, -1, 25, 0],
    [0, 0, 1, -1, 0, -3, 1, 3, -1, 1, -1, 3, 0],
    [0, 0, 3, -3, 0, -6, 5, -1, 2, 1, -4, 21, 0],
    [-7, 1, 5, 4, -3, 5, 11, 13, 12, -8, 11, 12, 0],
    [-5, -3, 6, -2, -3, 8, 14, 15, 2, -7, 11, 16, 0],
    [2, -1, -6, -5, -2, -2, 20, 14, -4, 0, -3, 25, 0],
    [3, 1, -8, -4, 0, -8, 22, 5, -3, 2, -10, 29, 0],
    [2, 1, -7, -1, 2, -11, 23, -5, 0, 2, -10, 29, 0],
    [-6, -3, 8, 9, -4, 8, 9, 7, 14, -2, 8, 9, 0],
    [2, 1, -4, -7, 0, -8, 17, 22, 1, -1, -4, 23, 0],
    [3, 0, -5, -7, 0, -7, 15, 18, -5, 0, -5, 27, 0],
    [2, 0, 0, -7, 1, -10, 13, 13, -4, 2, -7, 24, 0],
    [3, 3, -13, 4, -2, -5, 9, 21, 25, -2, -3, 12, 0],
    [-5, -2, 7, -3, -7, 9, 8, 9, 16, -2, 15, 12, 0],
    [0, -1, 0, -7, -5, 4, 11, 11, 8, -6, 12, 21, 0],
    [3, -2, -3, -8, -4, -1, 16, 15, -2, -3, 3, 26, 0],
    [2, 1, -5, -4, -1, -8, 16, 4, -2, 1, -7, 33, 0],
    [2, 1, -4, -2, 1, -10, 17, -2, 0, 2, -11, 33, 0],
    [1, -2, 7, -15, -16, 10, 8, 8, 20, 11, 14, 11, 0],
    [2, 2, 3, -13, -13, 4, 8, 12, 2, -3, 16, 24, 0],
    [1, 4, 0, -7, -8, -4, 9, 9, -2, -2, 8, 29, 0],
    [1, 1, 2, -4, -1, -6, 6, 3, -1, -1, -3, 30, 0],
    [-7, 3, 2, 10, -2, 3, 7, 11, 19, -7, 8, 10, 0],
    [0, -2, -5, -3, -2, 4, 20, 15, -1, -3, -1, 22, 0],
    [3, -1, -8, -4, -1, -4, 22, 8, -4, 2, -8, 28, 0],
    [0, 3, -14, 3, 0, 1, 19, 17, 8, -3, -7, 20, 0],
    [0, 2, -1, -8, 3, -6, 5, 21, 1, 1, -9, 13, 0],
    [-4, -2, 8, 20, -2, 2, 3, 5, 21, 4, 6, 1, 0],
    [2, -2, -3, -9, -4, 2, 14, 16, 3, -6, 8, 24, 0],
    [2, 1, 5, -16, -7, 2, 3, 11, 15, -3, 11, 22, 0],
    [1, 2, 3, -11, -2, -5, 4, 8, 9, -3, -2, 26, 0],
    [0, -1, 10, -9, -1, -8, 2, 3, 4, 0, 0, 29, 0],
    [1, 2, 0, -5, 1, -9, 9, 3, 0, 1, -7, 20, 0],
    [-2, 8, -6, -4, 3, -9, -8, 45, 14, 2, -13, 7, 0],
    [1, -1, 16, -19, -8, -4, -3, 2, 19, 0, 4, 30, 0],
    [1, 1, -3, 0, 2, -11, 15, -5, 1, 2, -9, 24, 0],
    [0, 1, -2, 0, 1, -4, 4, 0, 0, 1, -4, 7, 0],
    [0, 1, 2, -5, 1, -6, 4, 10, -2, 1, -4, 10, 0],
    [3, 0, -3, -6, -2, -6, 14, 8, -1, -1, -3, 31, 0],
    [0, 1, 0, -2, 1, -6, 5, 1, 0, 1, -5, 13, 0],
    [3, 1, 9, -19, -21, 9, 7, 6, 13, 5, 15, 21, 0],
    [2, 4, 3, -12, -13, 1, 7, 8, 3, 0, 12, 26, 0],
    [3, 1, -8, -2, 0, -6, 18, 2, -2, 3, -10, 23, 0],
    [1, 1, -4, -1, 1, -5, 8, 1, -1, 2, -5, 10, 0],
    [0, 1, -1, 0, 0, -2, 2, 0, 0, 1, -2, 3, 0],
    [1, 1, -2, -7, 1, -7, 14, 18, 0, 0, -7, 21, 0],
    [0, 1, 0, -2, 0, -7, 8, 1, -2, 0, -3, 24, 0],
    [0, 1, 1, -2, 2, -10, 10, 0, -2, 1, -7, 23, 0],
    [0, 2, 2, -11, 2, -4, -3, 39, 7, 1, -10, 9, 0],
    [1, 0, 13, -16, -5, -6, -1, 8, 6, 0, 6, 29, 0],
    [1, 3, 1, -6, -4, -7, 9, 6, -3, -2, 3, 33, 0],
    [4, 0, -17, -1, -1, 5, 26, 8, -2, 3, -15, 30, 0],
    [0, 1, -2, 0, 2, -8, 12, -6, 1, 1, -6, 16, 0],
    [0, 0, 0, -1, 1, -4, 4, 0, 0, 0, -3, 11, 0],
    [0, 1, 2, -8, 2, -6, 5, 15, 0, 2, -7, 9, 0],
    [1, -1, 12, -15, -7, -2, 3, 6, 6, -1, 7, 30, 0],
];

/// Mapping from `(fixed_filter_set_idx, class_idx)` to `ALF_FIXED_FILTER_COEFF` index
/// (`m_classToFilterMapping[16][25]`, H.266 §8.8.5.2 Table 48, VTM `AdaptiveLoopFilter.cpp:248-266`).
pub const ALF_CLASS_TO_FILTER_MAPPING: [[u8; 25]; 16] = [
    [
        8, 2, 2, 2, 3, 4, 53, 9, 9, 52, 4, 4, 5, 9, 2, 8, 10, 9, 1, 3, 39, 39, 10, 9, 52,
    ],
    [
        11, 12, 13, 14, 15, 30, 11, 17, 18, 19, 16, 20, 20, 4, 53, 21, 22, 23, 14, 25, 26, 26, 27,
        28, 10,
    ],
    [
        16, 12, 31, 32, 14, 16, 30, 33, 53, 34, 35, 16, 20, 4, 7, 16, 21, 36, 18, 19, 21, 26, 37,
        38, 39,
    ],
    [
        35, 11, 13, 14, 43, 35, 16, 4, 34, 62, 35, 35, 30, 56, 7, 35, 21, 38, 24, 40, 16, 21, 48,
        57, 39,
    ],
    [
        11, 31, 32, 43, 44, 16, 4, 17, 34, 45, 30, 20, 20, 7, 5, 21, 22, 46, 40, 47, 26, 48, 63,
        58, 10,
    ],
    [
        12, 13, 50, 51, 52, 11, 17, 53, 45, 9, 30, 4, 53, 19, 0, 22, 23, 25, 43, 44, 37, 27, 28,
        10, 55,
    ],
    [
        30, 33, 62, 51, 44, 20, 41, 56, 34, 45, 20, 41, 41, 56, 5, 30, 56, 38, 40, 47, 11, 37, 42,
        57, 8,
    ],
    [
        35, 11, 23, 32, 14, 35, 20, 4, 17, 18, 21, 20, 20, 20, 4, 16, 21, 36, 46, 25, 41, 26, 48,
        49, 58,
    ],
    [
        12, 31, 59, 59, 3, 33, 33, 59, 59, 52, 4, 33, 17, 59, 55, 22, 36, 59, 59, 60, 22, 36, 59,
        25, 55,
    ],
    [
        31, 25, 15, 60, 60, 22, 17, 19, 55, 55, 20, 20, 53, 19, 55, 22, 46, 25, 43, 60, 37, 28, 10,
        55, 52,
    ],
    [
        12, 31, 32, 50, 51, 11, 33, 53, 19, 45, 16, 4, 4, 53, 5, 22, 36, 18, 25, 43, 26, 27, 27,
        28, 10,
    ],
    [
        5, 2, 44, 52, 3, 4, 53, 45, 9, 3, 4, 56, 5, 0, 2, 5, 10, 47, 52, 3, 63, 39, 10, 9, 52,
    ],
    [
        12, 34, 44, 44, 3, 56, 56, 62, 45, 9, 56, 56, 7, 5, 0, 22, 38, 40, 47, 52, 48, 57, 39, 10,
        9,
    ],
    [
        35, 11, 23, 14, 51, 35, 20, 41, 56, 62, 16, 20, 41, 56, 7, 16, 21, 38, 24, 40, 26, 26, 42,
        57, 39,
    ],
    [
        33, 34, 51, 51, 52, 41, 41, 34, 62, 0, 41, 41, 56, 7, 5, 56, 38, 38, 40, 44, 37, 42, 57,
        39, 10,
    ],
    [
        16, 31, 32, 15, 60, 30, 4, 17, 19, 25, 22, 20, 4, 53, 19, 21, 22, 46, 25, 55, 26, 48, 63,
        58, 55,
    ],
];

/// 8-bit non-linear ALF clipping values `[256, 32, 8, 2]` (`AdaptiveLoopFilter.cpp:714-726`).
pub const ALF_CLIP_VALS_8BIT: [i32; 4] = [256, 32, 8, 2];

/// Derive the ALF class index (`0..25`) and transpose index (`0..4`) for a `4x4` luma
/// block at `(bx, by)` using the 1D Laplacian gradients (`AdaptiveLoopFilter.cpp:829-1045`).
pub fn derive_alf_class_4x4(plane: &Plane, bx: usize, by: usize) -> (usize, usize) {
    const TH: [usize; 16] = [0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4];
    const TRANSPOSE_TABLE: [usize; 8] = [0, 1, 0, 2, 2, 3, 1, 3];

    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, plane.width as isize - 1) as usize;
        let cy = y.clamp(0, plane.height as isize - 1) as usize;
        i32::from(plane.get(cx, cy))
    };

    let mut sum_v = 0i32;
    let mut sum_h = 0i32;
    let mut sum_d0 = 0i32;
    let mut sum_d1 = 0i32;

    for dy in (-2..6isize).step_by(2) {
        for dx in (-2..6isize).step_by(2) {
            let x = bx as isize + dx;
            let y = by as isize + dy;
            let y0 = get_p(x, y) << 1;
            let yup1 = get_p(x + 1, y + 1) << 1;

            sum_v += (y0 - get_p(x, y - 1) - get_p(x, y + 1)).abs()
                + (yup1 - get_p(x + 1, y) - get_p(x + 1, y + 2)).abs();
            sum_h += (y0 - get_p(x + 1, y) - get_p(x - 1, y)).abs()
                + (yup1 - get_p(x + 2, y + 1) - get_p(x, y + 1)).abs();
            sum_d0 += (y0 - get_p(x - 1, y - 1) - get_p(x + 1, y + 1)).abs()
                + (yup1 - get_p(x, y) - get_p(x + 2, y + 2)).abs();
            sum_d1 += (y0 - get_p(x - 1, y + 1) - get_p(x + 1, y - 1)).abs()
                + (yup1 - get_p(x, y + 2) - get_p(x + 2, y)).abs();
        }
    }

    let temp_act = sum_v + sum_h;
    // shift = bitDepth + 4 = 12 for 8-bit
    let activity = ((temp_act * 64) >> 12).clamp(0, 15) as usize;
    let mut class_idx = TH[activity];

    let (hv1, hv0, dir_temp_hv) = if sum_v > sum_h {
        (sum_v, sum_h, 1usize)
    } else {
        (sum_h, sum_v, 3usize)
    };
    let (d1, d0, dir_temp_d) = if sum_d0 > sum_d1 {
        (sum_d0, sum_d1, 0usize)
    } else {
        (sum_d1, sum_d0, 2usize)
    };

    let (hvd1, hvd0, main_dir, sec_dir) = if (d1 as i64) * (hv0 as i64) > (hv1 as i64) * (d0 as i64)
    {
        (d1, d0, dir_temp_d, dir_temp_hv)
    } else {
        (hv1, hv0, dir_temp_hv, dir_temp_d)
    };

    let mut dir_strength = 0usize;
    if hvd1 > 2 * hvd0 {
        dir_strength = 1;
    }
    if hvd1 * 2 > 9 * hvd0 {
        dir_strength = 2;
    }
    if dir_strength > 0 {
        class_idx += (((main_dir & 1) << 1) + dir_strength) * 5;
    }
    let transpose_idx = TRANSPOSE_TABLE[main_dir * 2 + (sec_dir >> 1)];
    (class_idx, transpose_idx)
}

#[inline]
fn clip_alf(clip_val: i32, curr: i32, p0: i32, p1: i32) -> i32 {
    (p0 - curr).clamp(-clip_val, clip_val) + (p1 - curr).clamp(-clip_val, clip_val)
}

/// Apply the 7x7 diamond luma Adaptive Loop Filter across `plane` using a `VvcAlfAps`
/// (`AdaptiveLoopFilter::filterBlk<ALF_FILTER_7>`, `AdaptiveLoopFilter.cpp:1048-1277`).
pub fn apply_alf_luma_plane(plane: &mut Plane, aps: &VvcAlfAps) {
    if aps.luma_filters.is_empty() {
        return;
    }
    let src = plane.clone();
    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, src.width as isize - 1) as usize;
        let cy = y.clamp(0, src.height as isize - 1) as usize;
        i32::from(src.get(cx, cy))
    };

    let mut by = 0usize;
    while by < plane.height {
        let mut bx = 0usize;
        while bx < plane.width {
            let (class_idx, transpose_idx) = derive_alf_class_4x4(&src, bx, by);
            let filt_idx = usize::from(
                aps.filter_coeff_delta_idx[class_idx]
                    .min((aps.luma_filters.len().saturating_sub(1)) as u8),
            );
            let raw_c = &aps.luma_filters[filt_idx];
            let raw_clip_idx = aps
                .luma_clip_idx
                .get(filt_idx)
                .copied()
                .unwrap_or([0u8; 12]);
            let mut c = [0i32; 12];
            let mut cl = [256i32; 12];
            for k in 0..12 {
                c[k] = i32::from(raw_c[k]);
                let cidx = if aps.luma_clip {
                    usize::from(raw_clip_idx[k] & 3)
                } else {
                    0
                };
                cl[k] = ALF_CLIP_VALS_8BIT[cidx];
            }

            let perm: [usize; 12] = match transpose_idx {
                1 => [9, 4, 10, 8, 1, 5, 11, 7, 3, 0, 2, 6],
                2 => [0, 3, 2, 1, 8, 7, 6, 5, 4, 9, 10, 11],
                3 => [9, 8, 10, 4, 3, 7, 11, 5, 1, 0, 2, 6],
                _ => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            };
            let fc: [i32; 12] = core::array::from_fn(|i| c[perm[i]]);
            let fclip: [i32; 12] = core::array::from_fn(|i| cl[perm[i]]);

            for dy in 0..4 {
                let y = by + dy;
                if y >= plane.height {
                    break;
                }
                let iy = y as isize;
                for dx in 0..4 {
                    let x = bx + dx;
                    if x >= plane.width {
                        break;
                    }
                    let ix = x as isize;
                    let curr = get_p(ix, iy);
                    let mut sum = 0i32;
                    sum += fc[0] * clip_alf(fclip[0], curr, get_p(ix, iy + 3), get_p(ix, iy - 3));
                    sum += fc[1]
                        * clip_alf(fclip[1], curr, get_p(ix + 1, iy + 2), get_p(ix - 1, iy - 2));
                    sum += fc[2] * clip_alf(fclip[2], curr, get_p(ix, iy + 2), get_p(ix, iy - 2));
                    sum += fc[3]
                        * clip_alf(fclip[3], curr, get_p(ix - 1, iy + 2), get_p(ix + 1, iy - 2));
                    sum += fc[4]
                        * clip_alf(fclip[4], curr, get_p(ix + 2, iy + 1), get_p(ix - 2, iy - 1));
                    sum += fc[5]
                        * clip_alf(fclip[5], curr, get_p(ix + 1, iy + 1), get_p(ix - 1, iy - 1));
                    sum += fc[6] * clip_alf(fclip[6], curr, get_p(ix, iy + 1), get_p(ix, iy - 1));
                    sum += fc[7]
                        * clip_alf(fclip[7], curr, get_p(ix - 1, iy + 1), get_p(ix + 1, iy - 1));
                    sum += fc[8]
                        * clip_alf(fclip[8], curr, get_p(ix - 2, iy + 1), get_p(ix + 2, iy - 1));
                    sum += fc[9] * clip_alf(fclip[9], curr, get_p(ix + 3, iy), get_p(ix - 3, iy));
                    sum += fc[10] * clip_alf(fclip[10], curr, get_p(ix + 2, iy), get_p(ix - 2, iy));
                    sum += fc[11] * clip_alf(fclip[11], curr, get_p(ix + 1, iy), get_p(ix - 1, iy));

                    let val = curr + ((sum + 64) >> 7);
                    plane.set(x, y, val.clamp(0, 255) as u8);
                }
            }
            bx += 4;
        }
        by += 4;
    }
}

/// Apply the 5x5 diamond chroma Adaptive Loop Filter across `plane` using alternative
/// `alt_idx` from `VvcAlfAps` (`AdaptiveLoopFilter::filterBlk<ALF_FILTER_5>`,
/// `AdaptiveLoopFilter.cpp:1233-1254`).
pub fn apply_alf_chroma_plane(plane: &mut Plane, aps: &VvcAlfAps, alt_idx: usize) {
    if aps.chroma_filters.is_empty() {
        return;
    }
    let idx = alt_idx.min(aps.chroma_filters.len() - 1);
    let raw_c = &aps.chroma_filters[idx];
    let raw_clip_idx = aps.chroma_clip_idx.get(idx).copied().unwrap_or([0u8; 6]);
    let mut fc = [0i32; 6];
    let mut fclip = [256i32; 6];
    for k in 0..6 {
        fc[k] = i32::from(raw_c[k]);
        let cidx = if aps.chroma_clip {
            usize::from(raw_clip_idx[k] & 3)
        } else {
            0
        };
        fclip[k] = ALF_CLIP_VALS_8BIT[cidx];
    }

    let src = plane.clone();
    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, src.width as isize - 1) as usize;
        let cy = y.clamp(0, src.height as isize - 1) as usize;
        i32::from(src.get(cx, cy))
    };

    for y in 0..plane.height {
        let iy = y as isize;
        for x in 0..plane.width {
            let ix = x as isize;
            let curr = get_p(ix, iy);
            let mut sum = 0i32;
            sum += fc[0] * clip_alf(fclip[0], curr, get_p(ix, iy + 2), get_p(ix, iy - 2));
            sum += fc[1] * clip_alf(fclip[1], curr, get_p(ix + 1, iy + 1), get_p(ix - 1, iy - 1));
            sum += fc[2] * clip_alf(fclip[2], curr, get_p(ix, iy + 1), get_p(ix, iy - 1));
            sum += fc[3] * clip_alf(fclip[3], curr, get_p(ix - 1, iy + 1), get_p(ix + 1, iy - 1));
            sum += fc[4] * clip_alf(fclip[4], curr, get_p(ix + 2, iy), get_p(ix - 2, iy));
            sum += fc[5] * clip_alf(fclip[5], curr, get_p(ix + 1, iy), get_p(ix - 1, iy));

            let val = curr + ((sum + 64) >> 7);
            plane.set(x, y, val.clamp(0, 255) as u8);
        }
    }
}

/// Apply Cross-Component ALF (CCALF) to a 4:2:0 chroma plane using the pre-ALF
/// luma plane `luma_src` and 7 signed filter coefficients `coeffs`
/// (`AdaptiveLoopFilter::filterBlkCcAlf`, `AdaptiveLoopFilter.cpp:1280-1375`).
pub fn apply_ccalf_plane_420(chroma_plane: &mut Plane, luma_src: &Plane, coeffs: &[i16; 7]) {
    let get_y = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, luma_src.width as isize - 1) as usize;
        let cy = y.clamp(0, luma_src.height as isize - 1) as usize;
        i32::from(luma_src.get(cx, cy))
    };

    for cy in 0..chroma_plane.height {
        let ly = (cy * 2) as isize;
        for cx in 0..chroma_plane.width {
            let lx = (cx * 2) as isize;
            let curr = get_y(lx, ly);
            let mut sum = 0i32;
            sum += i32::from(coeffs[0]) * (get_y(lx, ly - 1) - curr);
            sum += i32::from(coeffs[1]) * (get_y(lx - 1, ly) - curr);
            sum += i32::from(coeffs[2]) * (get_y(lx + 1, ly) - curr);
            sum += i32::from(coeffs[3]) * (get_y(lx - 1, ly + 1) - curr);
            sum += i32::from(coeffs[4]) * (get_y(lx, ly + 1) - curr);
            sum += i32::from(coeffs[5]) * (get_y(lx + 1, ly + 1) - curr);
            sum += i32::from(coeffs[6]) * (get_y(lx, ly + 2) - curr);

            let delta = ((sum + 64) >> 7).clamp(-128, 127);
            let c_val = i32::from(chroma_plane.get(cx, cy)) + delta;
            chroma_plane.set(cx, cy, c_val.clamp(0, 255) as u8);
        }
    }
}

use super::predict::Plane16;

/// Normative 10-bit ALF non-linear clipping table `AlfClip = { 1024, 128, 32, 8 }`
/// (`H.266 §8.8.5.2 Table 44`, VTM `AdaptiveLoopFilter.cpp:714-725`).
pub const ALF_CLIP_VALS_10BIT: [i32; 4] = [1024, 128, 32, 8];

/// Normative 12-bit ALF non-linear clipping table `AlfClip = { 4096, 512, 128, 32 }`
/// (`H.266 §8.8.5.2 Table 44`, VTM `AdaptiveLoopFilter.cpp:714-725`).
pub const ALF_CLIP_VALS_12BIT: [i32; 4] = [4096, 512, 128, 32];

/// Apply Sample Adaptive Offset (SAO) to a region `(bx, by, w, h)` of a 16-bit [`Plane16`]
/// (`bit_depth` in `8..=12`, ITU-T Rec. H.266 §8.8.4, VTM `SampleAdaptiveOffset.cpp`).
pub fn apply_sao_region_16(
    src: &Plane16,
    dst: &mut Plane16,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    sao: SaoType,
) {
    let bd = src.bit_depth.clamp(8, 12);
    let max_val = src.max_sample();
    let band_shift = bd - 5;
    let w_lim = (bx + w).min(src.width);
    let h_lim = (by + h).min(src.height);

    match sao {
        SaoType::Off => {}
        SaoType::BandOffset { band_pos, offsets } => {
            for y in by..h_lim {
                for x in bx..w_lim {
                    let s = i32::from(src.get(x, y));
                    let band = ((s >> band_shift) as u8).wrapping_sub(band_pos) & 31;
                    if band < 4 {
                        let off = i32::from(offsets[band as usize]) << (bd - 8);
                        dst.set(x, y, (s + off).clamp(0, max_val) as u16);
                    }
                }
            }
        }
        SaoType::EdgeOffset { eo_class, offsets } => {
            let (dx, dy): (isize, isize) = match eo_class & 3 {
                0 => (1, 0),
                1 => (0, 1),
                2 => (1, 1),
                _ => (1, -1),
            };
            for y in by..h_lim {
                for x in bx..w_lim {
                    let ix = x as isize;
                    let iy = y as isize;
                    if ix - dx < 0
                        || ix + dx >= src.width as isize
                        || iy - dy < 0
                        || iy + dy >= src.height as isize
                    {
                        continue;
                    }
                    let c = i32::from(src.get(x, y));
                    let a = i32::from(src.get((ix - dx) as usize, (iy - dy) as usize));
                    let b = i32::from(src.get((ix + dx) as usize, (iy + dy) as usize));
                    let edge_idx = 2 + (c - a).signum() + (c - b).signum();
                    let off = match edge_idx {
                        0 => i32::from(offsets[0]),
                        1 => i32::from(offsets[1]),
                        3 => i32::from(offsets[2]),
                        4 => i32::from(offsets[3]),
                        _ => 0,
                    } << (bd - 8);
                    dst.set(x, y, (c + off).clamp(0, max_val) as u16);
                }
            }
        }
    }
}

/// ALF coefficient transposition permutations (`ALF_TRANSPOSE_PERM[transpose_idx]`).
pub const ALF_TRANSPOSE_PERM: [[usize; 12]; 4] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [9, 4, 10, 8, 1, 5, 11, 7, 3, 0, 2, 6],
    [0, 3, 2, 1, 8, 7, 6, 5, 4, 9, 10, 11],
    [9, 8, 10, 4, 3, 7, 11, 5, 1, 0, 2, 6],
];

/// Apply 7x7 diamond luma Adaptive Loop Filter (ALF) to a 16-bit [`Plane16`] using
/// [`ALF_CLIP_VALS_10BIT`] for 10-bit samples (`H.266 §8.8.5.2`, `AdaptiveLoopFilter.cpp`).
pub fn apply_alf_luma_plane_16(plane: &mut Plane16, aps: &VvcAlfAps) {
    let bd = plane.bit_depth.clamp(8, 12);
    let max_val = plane.max_sample();
    let clip_table = if bd >= 12 {
        ALF_CLIP_VALS_12BIT
    } else if bd >= 10 {
        ALF_CLIP_VALS_10BIT
    } else {
        ALF_CLIP_VALS_8BIT
    };
    let src = plane.clone();
    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, src.width as isize - 1) as usize;
        let cy = y.clamp(0, src.height as isize - 1) as usize;
        i32::from(src.get(cx, cy))
    };

    for by in (0..plane.height).step_by(4) {
        for bx in (0..plane.width).step_by(4) {
            let (class_idx, transpose_idx) = classify_alf_4x4_16(&src, bx, by);
            let filt_idx = if !aps.luma_filters.is_empty() {
                usize::from(aps.filter_coeff_delta_idx[class_idx]).min(aps.luma_filters.len() - 1)
            } else {
                0
            };
            let base_coeffs: [i16; 12] = if !aps.luma_filters.is_empty() {
                aps.luma_filters[filt_idx]
            } else {
                let fixed = ALF_CLASS_TO_FILTER_MAPPING[class_idx][0] as usize;
                let mut arr = [0i16; 12];
                arr.copy_from_slice(&ALF_FIXED_FILTER_COEFF[fixed][..12]);
                arr
            };
            let base_clips: [u8; 12] = if aps.luma_clip && filt_idx < aps.luma_clip_idx.len() {
                aps.luma_clip_idx[filt_idx]
            } else {
                [0u8; 12]
            };
            let perm = &ALF_TRANSPOSE_PERM[transpose_idx & 3];
            let mut fc = [0i32; 12];
            let mut fclip = [clip_table[0]; 12];
            for k in 0..12 {
                fc[k] = i32::from(base_coeffs[perm[k]]);
                fclip[k] = clip_table[usize::from(base_clips[perm[k]] & 3)];
            }

            let h_end = (by + 4).min(plane.height);
            let w_end = (bx + 4).min(plane.width);
            for y in by..h_end {
                let iy = y as isize;
                for x in bx..w_end {
                    let ix = x as isize;
                    let curr = get_p(ix, iy);
                    let mut sum = 0i32;
                    sum += fc[0] * clip_alf(fclip[0], curr, get_p(ix, iy + 3), get_p(ix, iy - 3));
                    sum += fc[1]
                        * clip_alf(fclip[1], curr, get_p(ix + 1, iy + 2), get_p(ix - 1, iy - 2));
                    sum += fc[2] * clip_alf(fclip[2], curr, get_p(ix, iy + 2), get_p(ix, iy - 2));
                    sum += fc[3]
                        * clip_alf(fclip[3], curr, get_p(ix - 1, iy + 2), get_p(ix + 1, iy - 2));
                    sum += fc[4]
                        * clip_alf(fclip[4], curr, get_p(ix + 2, iy + 1), get_p(ix - 2, iy - 1));
                    sum += fc[5]
                        * clip_alf(fclip[5], curr, get_p(ix + 1, iy + 1), get_p(ix - 1, iy - 1));
                    sum += fc[6] * clip_alf(fclip[6], curr, get_p(ix, iy + 1), get_p(ix, iy - 1));
                    sum += fc[7]
                        * clip_alf(fclip[7], curr, get_p(ix - 1, iy + 1), get_p(ix + 1, iy - 1));
                    sum += fc[8]
                        * clip_alf(fclip[8], curr, get_p(ix - 2, iy + 1), get_p(ix + 2, iy - 1));
                    sum += fc[9] * clip_alf(fclip[9], curr, get_p(ix + 3, iy), get_p(ix - 3, iy));
                    sum += fc[10] * clip_alf(fclip[10], curr, get_p(ix + 2, iy), get_p(ix - 2, iy));
                    sum += fc[11] * clip_alf(fclip[11], curr, get_p(ix + 1, iy), get_p(ix - 1, iy));

                    let val = curr + ((sum + 64) >> 7);
                    plane.set(x, y, val.clamp(0, max_val) as u16);
                }
            }
        }
    }
}

fn classify_alf_4x4_16(plane: &Plane16, bx: usize, by: usize) -> (usize, usize) {
    const TH: [usize; 16] = [0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4];
    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, plane.width as isize - 1) as usize;
        let cy = y.clamp(0, plane.height as isize - 1) as usize;
        i32::from(plane.get(cx, cy))
    };
    let mut sum_h = 0i32;
    let mut sum_v = 0i32;
    let mut sum_d0 = 0i32;
    let mut sum_d1 = 0i32;
    for dy in -1isize..5 {
        for dx in -1isize..5 {
            if (dy & 1) == 0 && (dx & 1) == 0 {
                let x = bx as isize + dx;
                let y = by as isize + dy;
                let c2 = get_p(x, y) << 1;
                sum_h += (c2 - get_p(x - 1, y) - get_p(x + 1, y)).abs();
                sum_v += (c2 - get_p(x, y - 1) - get_p(x, y + 1)).abs();
                sum_d0 += (c2 - get_p(x - 1, y - 1) - get_p(x + 1, y + 1)).abs();
                sum_d1 += (c2 - get_p(x + 1, y - 1) - get_p(x - 1, y + 1)).abs();
            }
        }
    }
    let temp_act = sum_h + sum_v;
    let shift = u32::from(plane.bit_depth.clamp(8, 12)) + 4;
    let activity = ((i64::from(temp_act) * 64) >> shift).clamp(0, 15) as usize;
    let q_act = TH[activity];

    let (hv1, hv0, dir_hv) = if sum_v > sum_h {
        (sum_v, sum_h, 1usize)
    } else {
        (sum_h, sum_v, 3usize)
    };
    let (d1, d0, dir_d) = if sum_d0 > sum_d1 {
        (sum_d0, sum_d1, 0usize)
    } else {
        (sum_d1, sum_d0, 2usize)
    };
    let (hvd1, hvd0, main_dir, sec_dir) =
        if i64::from(d1) * i64::from(hv0) > i64::from(hv1) * i64::from(d0) {
            (d1, d0, dir_d, dir_hv)
        } else {
            (hv1, hv0, dir_hv, dir_d)
        };
    let mut dir_strength = 0usize;
    if hvd1 > 2 * hvd0 {
        dir_strength = if hvd1 * 2 > 9 * hvd0 { 2 } else { 1 };
    }
    let transpose_table: [usize; 8] = [0, 1, 0, 2, 2, 3, 1, 3];
    let transpose_idx = transpose_table[main_dir * 2 + (sec_dir >> 1)];
    let mut class_idx = q_act;
    if dir_strength != 0 {
        class_idx += (((main_dir & 1) << 1) + dir_strength - 1 + 1) * 5;
    }
    (class_idx.min(24), transpose_idx)
}

/// Apply 5x5 diamond chroma ALF to a 16-bit [`Plane16`] (`H.266 §8.8.5.4`,
/// VTM `AdaptiveLoopFilter.cpp`).
pub fn apply_alf_chroma_plane_16(plane: &mut Plane16, aps: &VvcAlfAps, alt_idx: usize) {
    if plane.width == 0 || plane.height == 0 || aps.chroma_filters.is_empty() {
        return;
    }
    let bd = plane.bit_depth.clamp(8, 12);
    let max_val = plane.max_sample();
    let clip_table = if bd >= 12 {
        ALF_CLIP_VALS_12BIT
    } else if bd >= 10 {
        ALF_CLIP_VALS_10BIT
    } else {
        ALF_CLIP_VALS_8BIT
    };
    let idx = alt_idx.min(aps.chroma_filters.len() - 1);
    let fc: [i32; 6] = [
        i32::from(aps.chroma_filters[idx][0]),
        i32::from(aps.chroma_filters[idx][1]),
        i32::from(aps.chroma_filters[idx][2]),
        i32::from(aps.chroma_filters[idx][3]),
        i32::from(aps.chroma_filters[idx][4]),
        i32::from(aps.chroma_filters[idx][5]),
    ];
    let fclip: [i32; 6] = if aps.chroma_clip && idx < aps.chroma_clip_idx.len() {
        let ci = &aps.chroma_clip_idx[idx];
        [
            clip_table[usize::from(ci[0] & 3)],
            clip_table[usize::from(ci[1] & 3)],
            clip_table[usize::from(ci[2] & 3)],
            clip_table[usize::from(ci[3] & 3)],
            clip_table[usize::from(ci[4] & 3)],
            clip_table[usize::from(ci[5] & 3)],
        ]
    } else {
        [clip_table[0]; 6]
    };
    let src = plane.clone();
    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, src.width as isize - 1) as usize;
        let cy = y.clamp(0, src.height as isize - 1) as usize;
        i32::from(src.get(cx, cy))
    };
    for y in 0..plane.height {
        let iy = y as isize;
        for x in 0..plane.width {
            let ix = x as isize;
            let curr = get_p(ix, iy);
            let mut sum = 0i32;
            sum += fc[0] * clip_alf(fclip[0], curr, get_p(ix, iy + 2), get_p(ix, iy - 2));
            sum += fc[1] * clip_alf(fclip[1], curr, get_p(ix + 1, iy + 1), get_p(ix - 1, iy - 1));
            sum += fc[2] * clip_alf(fclip[2], curr, get_p(ix, iy + 1), get_p(ix, iy - 1));
            sum += fc[3] * clip_alf(fclip[3], curr, get_p(ix - 1, iy + 1), get_p(ix + 1, iy - 1));
            sum += fc[4] * clip_alf(fclip[4], curr, get_p(ix + 2, iy), get_p(ix - 2, iy));
            sum += fc[5] * clip_alf(fclip[5], curr, get_p(ix + 1, iy), get_p(ix - 1, iy));
            let val = curr + ((sum + 64) >> 7);
            plane.set(x, y, val.clamp(0, max_val) as u16);
        }
    }
}

/// Apply 7-tap Cross-Component ALF (CCALF) to a 16-bit chroma [`Plane16`] using
/// pre-ALF luma [`Plane16`] for any [`super::predict::ChromaFormatIdc`]
/// (`H.266 §8.8.5.7`, VTM `AdaptiveLoopFilter.cpp:1280-1375`).
pub fn apply_ccalf_plane_16(
    chroma_plane: &mut Plane16,
    luma_src: &Plane16,
    coeffs: &[i16; 7],
    chroma_format: super::predict::ChromaFormatIdc,
) {
    if chroma_plane.width == 0 || chroma_plane.height == 0 || !chroma_format.is_chroma_enabled() {
        return;
    }
    let bd = chroma_plane.bit_depth.clamp(8, 12);
    let max_val = chroma_plane.max_sample();
    let offset = 1i32 << (bd - 1);
    let sub_w = chroma_format.sub_width_c() as isize;
    let sub_h = chroma_format.sub_height_c() as isize;
    let get_y = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, luma_src.width as isize - 1) as usize;
        let cy = y.clamp(0, luma_src.height as isize - 1) as usize;
        i32::from(luma_src.get(cx, cy))
    };
    for cy in 0..chroma_plane.height {
        let ly = (cy as isize) * sub_h;
        for cx in 0..chroma_plane.width {
            let lx = (cx as isize) * sub_w;
            let y0 = get_y(lx, ly);
            let mut sum = 0i32;
            sum += i32::from(coeffs[0]) * (get_y(lx, ly - 1) - y0);
            sum += i32::from(coeffs[1]) * (get_y(lx - 1, ly) - y0);
            sum += i32::from(coeffs[2]) * (get_y(lx + 1, ly) - y0);
            sum += i32::from(coeffs[3]) * (get_y(lx - 1, ly + 1) - y0);
            sum += i32::from(coeffs[4]) * (get_y(lx, ly + 1) - y0);
            sum += i32::from(coeffs[5]) * (get_y(lx + 1, ly + 1) - y0);
            sum += i32::from(coeffs[6]) * (get_y(lx, ly + 2) - y0);
            let delta = ((sum + 64) >> 7).clamp(-offset, offset - 1);
            let c_val = i32::from(chroma_plane.get(cx, cy)) + delta;
            chroma_plane.set(cx, cy, c_val.clamp(0, max_val) as u16);
        }
    }
}

/// Apply Sample Adaptive Offset (SAO) across an entire 16-bit [`Plane16`] using per-CTU
/// [`SaoType`] parameters (`H.266 §8.8.4`, VTM `SampleAdaptiveOffset.cpp`).
pub fn apply_sao_plane_16(plane: &mut Plane16, ctu_size: usize, ctu_params: &[SaoType]) {
    if plane.width == 0 || plane.height == 0 || ctu_params.is_empty() {
        return;
    }
    let step = ctu_size.max(4);
    let cols = plane.width.div_ceil(step);
    let rows = plane.height.div_ceil(step);
    let src = plane.clone();
    for ry in 0..rows {
        for rx in 0..cols {
            let idx = ry * cols + rx;
            let sao = ctu_params.get(idx).copied().unwrap_or(SaoType::Off);
            apply_sao_region_16(&src, plane, rx * step, ry * step, step, step, sao);
        }
    }
}

/// Normative 16-bit luma 4-sample edge deblocking (`bit_depth` in `8..=12`,
/// ITU-T Rec. H.266 §8.8.3.6.2-§8.8.3.6.7, VTM `DeblockingFilter.cpp:1094-1180`).
///
/// The 8-sample form cannot express a long filter, so it always runs with
/// `maxFilterLengthP = maxFilterLengthQ = 3`.
pub fn deblock_luma_edge_4_16(
    lines: &mut [[i32; 8]; 4],
    qp: i32,
    bs: u8,
    beta_offset_div2: i8,
    tc_offset_div2: i8,
    bit_depth: u8,
) {
    let mut wide = [[0i32; 16]; 4];
    for (dst, src) in wide.iter_mut().zip(lines.iter()) {
        dst[4..12].copy_from_slice(src);
    }
    deblock_luma_edge_bd(
        &mut wide,
        qp,
        qp,
        bs,
        beta_offset_div2,
        tc_offset_div2,
        bit_depth,
        3,
        3,
        false,
    );
    for (dst, src) in lines.iter_mut().zip(wide.iter()) {
        dst.copy_from_slice(&src[4..12]);
    }
}

/// Deblock all vertical and horizontal luma edges on a coding-block lattice of a 16-bit
/// [`Plane16`] (`bit_depth` in `8..=12`, ITU-T Rec. H.266 §8.8.3).
///
/// `ctb_size` is `CtbSizeY`; see [`deblock_luma_plane`].
pub fn deblock_luma_plane_16(
    plane: &mut Plane16,
    cu_step: usize,
    ctb_size: usize,
    qp: i32,
    bs: u8,
    beta_offset_div2: i8,
    tc_offset_div2: i8,
) {
    let step = cu_step.max(8);
    let ctb = ctb_size.max(step);
    let max_len = luma_filter_length(step);
    let w = plane.width;
    let h = plane.height;
    let bd = plane.bit_depth.clamp(8, 12);
    let max_val = plane.max_sample();

    let mut x = step;
    while x + 8 <= w && x >= 8 {
        let mut y = 0usize;
        while y + 4 <= h {
            let mut lines = [[0i32; 16]; 4];
            for dy in 0..4 {
                for k in 0..16 {
                    lines[dy][k] = i32::from(plane.get(x - 8 + k, y + dy));
                }
            }
            deblock_luma_edge_bd(
                &mut lines,
                qp,
                qp,
                bs,
                beta_offset_div2,
                tc_offset_div2,
                bd,
                max_len,
                max_len,
                false,
            );
            for dy in 0..4 {
                for k in 0..16 {
                    plane.set(x - 8 + k, y + dy, lines[dy][k].clamp(0, max_val) as u16);
                }
            }
            y += 4;
        }
        x += step;
    }

    let mut y = step;
    while y + 8 <= h && y >= 8 {
        let at_ctb_row_boundary = y.is_multiple_of(ctb);
        let mut bx = 0usize;
        while bx + 4 <= w {
            let mut lines = [[0i32; 16]; 4];
            for dx in 0..4 {
                for k in 0..16 {
                    lines[dx][k] = i32::from(plane.get(bx + dx, y - 8 + k));
                }
            }
            deblock_luma_edge_bd(
                &mut lines,
                qp,
                qp,
                bs,
                beta_offset_div2,
                tc_offset_div2,
                bd,
                max_len,
                max_len,
                at_ctb_row_boundary,
            );
            for dx in 0..4 {
                for k in 0..16 {
                    plane.set(bx + dx, y - 8 + k, lines[dx][k].clamp(0, max_val) as u16);
                }
            }
            bx += 4;
        }
        y += step;
    }
}

/// Deblock all vertical and horizontal chroma edges on a `cu_step` lattice of a 16-bit
/// [`Plane16`] (`bit_depth` in `8..=12`, ITU-T Rec. H.266 §8.8.3.6.4-§8.8.3.6.10).
///
/// `ctb_height_c` is `CtbHeightC`; `sub_width_c`/`sub_height_c` are
/// `SubWidthC`/`SubHeightC`. See [`deblock_chroma_plane`].
#[allow(clippy::too_many_arguments)]
pub fn deblock_chroma_plane_16(
    plane: &mut Plane16,
    cu_step: usize,
    ctb_height_c: usize,
    qp_chroma: i32,
    bs: u8,
    beta_offset_div2: i8,
    tc_offset_div2: i8,
    sub_width_c: usize,
    sub_height_c: usize,
) {
    if bs == 0 || plane.width == 0 || plane.height == 0 {
        return;
    }
    let bd = plane.bit_depth.clamp(8, 12);
    let max_val = i32::from(plane.max_sample());
    let step = cu_step.max(8);
    let ctb = ctb_height_c.max(step);
    let len = chroma_filter_length(step);
    let w = plane.width;
    let h = plane.height;
    let group_v = if sub_height_c == 1 { 4 } else { 2 };
    let group_h = if sub_width_c == 1 { 4 } else { 2 };

    let mut x = step;
    while x >= 4 && x + 4 <= w {
        let mut y = 0;
        while y + group_v <= h {
            let mut lines = [[0i32; 8]; 4];
            for (k, line) in lines.iter_mut().take(group_v).enumerate() {
                for (i, sample) in line.iter_mut().enumerate() {
                    let xx = x - 4 + i;
                    *sample = i32::from(plane.get(xx, y + k));
                }
            }
            deblock_chroma_edge(
                &mut lines[..group_v],
                qp_chroma,
                bs,
                beta_offset_div2,
                tc_offset_div2,
                len,
                len,
                bd,
            );
            for (k, line) in lines.iter().take(group_v).enumerate() {
                for (i, &sample) in line.iter().enumerate() {
                    let xx = x - 4 + i;
                    plane.set(xx, y + k, sample.clamp(0, max_val) as u16);
                }
            }
            y += group_v;
        }
        x += step;
    }

    let mut y = step;
    while y >= 4 && y + 4 <= h {
        let (len_p, len_q) = if y.is_multiple_of(ctb) {
            (1, len)
        } else {
            (len, len)
        };
        let mut bx = 0;
        while bx + group_h <= w {
            let mut lines = [[0i32; 8]; 4];
            for (k, line) in lines.iter_mut().take(group_h).enumerate() {
                for (i, sample) in line.iter_mut().enumerate() {
                    let yy = y - 4 + i;
                    *sample = i32::from(plane.get(bx + k, yy));
                }
            }
            deblock_chroma_edge(
                &mut lines[..group_h],
                qp_chroma,
                bs,
                beta_offset_div2,
                tc_offset_div2,
                len_p,
                len_q,
                bd,
            );
            for (k, line) in lines.iter().take(group_h).enumerate() {
                for (i, &sample) in line.iter().enumerate() {
                    let yy = y - 4 + i;
                    plane.set(bx + k, yy, sample.clamp(0, max_val) as u16);
                }
            }
            bx += group_h;
        }
        y += step;
    }
}

/// Apply display-side Film Grain synthesis (`payloadType == 19`) to a 16-bit [`Plane16`]
/// (`bit_depth` in `8..=12`, ITU-T Rec. H.274 §8.5).
pub fn apply_film_grain_plane_16(
    plane: &mut Plane16,
    log2_scale_factor: u8,
    intervals: &[(u8, u8, i16)],
    seed: u32,
) {
    if intervals.is_empty() || plane.width == 0 || plane.height == 0 {
        return;
    }
    let bd = plane.bit_depth.clamp(8, 12);
    let bd_shift = bd - 8;
    let max_val = plane.max_sample();
    let shift = u32::from(log2_scale_factor.clamp(2, 7));
    let mut prng = seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x85EB_CA6B);
    for y in 0..plane.height {
        for x in 0..plane.width {
            let s16 = plane.get(x, y);
            let s8 = (s16 >> bd_shift).min(255) as u8;
            let mut scale = 0i32;
            for &(lo, hi, val) in intervals {
                if s8 >= lo && s8 <= hi {
                    scale = i32::from(val);
                    break;
                }
            }
            if scale != 0 {
                prng ^= prng << 13;
                prng ^= prng >> 17;
                prng ^= prng << 5;
                let noise = ((prng & 0x3F) as i32) - 32;
                let delta = ((scale * noise + (1 << (shift - 1))) >> shift) << bd_shift;
                plane.set(x, y, (i32::from(s16) + delta).clamp(0, max_val) as u16);
            }
        }
    }
}

/// Apply display-side Film Grain synthesis (ITU-T Rec. H.266 Annex D.3.10 / SMPTE RDD 5,
/// `payloadType == 19`) to an 8-bit plane using piecewise-linear intensity intervals
/// `(lower_bound, upper_bound, comp_model_value)` and a deterministic PRNG seed (`poc`).
pub fn apply_film_grain_plane(
    plane: &mut Plane,
    log2_scale_factor: u8,
    intervals: &[(u8, u8, i16)],
    seed: u32,
) {
    if intervals.is_empty() {
        return;
    }
    let shift = u32::from(log2_scale_factor.clamp(2, 7));
    let mut prng = seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x85EB_CA6B);
    for y in 0..plane.height {
        for x in 0..plane.width {
            let s = plane.get(x, y);
            let mut scale = 0i32;
            for &(lo, hi, val) in intervals {
                if s >= lo && s <= hi {
                    scale = i32::from(val);
                    break;
                }
            }
            if scale != 0 {
                prng ^= prng << 13;
                prng ^= prng >> 17;
                prng ^= prng << 5;
                let noise = ((prng & 0x3F) as i32) - 32;
                let delta = (scale * noise + (1 << (shift - 1))) >> shift;
                plane.set(x, y, (i32::from(s) + delta).clamp(0, 255) as u8);
            }
        }
    }
}

/// 16-bit Luma Mapping with Chroma Scaling (`LMCS`, `bit_depth` in `8..=12`,
/// ITU-T Rec. H.266 §8.8.2, VTM `Reshape.cpp:257-317`).
#[derive(Debug, Clone)]
pub struct LmcsReshaper16 {
    /// Sample bit depth (`8..=12`).
    pub bit_depth: u8,
    /// Minimum active bin index (`lmcs_min_bin_idx`).
    pub min_bin_idx: usize,
    /// Maximum active bin index (`15 - lmcs_delta_max_bin_idx`).
    pub max_bin_idx: usize,
    /// Codeword counts per bin `bin_cw[0..16]`.
    pub bin_cw: [i32; 16],
    /// Original domain pivot points `input_pivot[0..=16]`.
    pub input_pivot: [i32; 17],
    /// Reshaped domain pivot points `reshape_pivot[0..=16]`.
    pub reshape_pivot: [i32; 17],
    /// Chroma residual scaling factors per bin `chroma_adj_lut[0..16]`.
    pub chroma_adj_lut: [i32; 16],
    /// Forward sample lookup table `fwd_lut[0..(1 << bit_depth)]`.
    pub fwd_lut: Vec<u16>,
    /// Inverse sample lookup table `inv_lut[0..(1 << bit_depth)]`.
    pub inv_lut: Vec<u16>,
}

impl LmcsReshaper16 {
    /// Construct a 16-bit [`LmcsReshaper16`] from `aps` at `bit_depth` (`8..=12`).
    pub fn from_aps(aps: &VvcLmcsAps, bit_depth: u8) -> Self {
        let bd = bit_depth.clamp(8, 12);
        let max_val = (1i32 << u32::from(bd)) - 1;
        let log2_cw = i32::from(bd) - 4;
        let init_cw = 1i32 << log2_cw;
        let min_bin = usize::from(aps.min_bin_idx.min(15));
        let max_bin = usize::from(aps.max_bin_idx.min(15)).max(min_bin);

        let mut bin_cw = [0i32; 16];
        for i in min_bin..=max_bin {
            let delta = i32::from(aps.bin_cw_delta[i]) << (bd - 8);
            bin_cw[i] = (init_cw + delta).max(0);
        }

        let mut input_pivot = [0i32; 17];
        let mut reshape_pivot = [0i32; 17];
        let mut fwd_scale_coef = [1i32 << LMCS_FP_PREC; 16];
        let mut inv_scale_coef = [1i32 << LMCS_FP_PREC; 16];
        let mut chroma_adj_lut = [1i32 << LMCS_FP_PREC; 16];

        for i in 0..16 {
            reshape_pivot[i + 1] = reshape_pivot[i] + bin_cw[i];
            input_pivot[i + 1] = input_pivot[i] + init_cw;
            fwd_scale_coef[i] = ((bin_cw[i] << LMCS_FP_PREC) + (1 << (log2_cw - 1))) >> log2_cw;
            if bin_cw[i] == 0 {
                inv_scale_coef[i] = 0;
                chroma_adj_lut[i] = 1 << LMCS_FP_PREC;
            } else {
                inv_scale_coef[i] = (init_cw << LMCS_FP_PREC) / bin_cw[i];
                let c_off = i32::from(aps.chroma_res_scaling_offset) << (bd - 8);
                let denom = (bin_cw[i] + c_off).max(1);
                chroma_adj_lut[i] = (init_cw << LMCS_FP_PREC) / denom;
            }
        }

        let get_pwl_idx_inv = |luma_val: i32| -> usize {
            let mut idx_s = min_bin;
            while idx_s <= max_bin {
                if luma_val < reshape_pivot[idx_s + 1] {
                    break;
                }
                idx_s += 1;
            }
            idx_s.min(15)
        };

        let table_len = (max_val + 1) as usize;
        let mut fwd_lut = vec![0u16; table_len];
        let mut inv_lut = vec![0u16; table_len];
        for sample in 0..=max_val {
            let idx_y = ((sample >> log2_cw) as usize).min(15);
            let fwd_val = reshape_pivot[idx_y]
                + ((fwd_scale_coef[idx_y] * (sample - input_pivot[idx_y])
                    + (1 << (LMCS_FP_PREC - 1)))
                    >> LMCS_FP_PREC);
            fwd_lut[sample as usize] = fwd_val.clamp(0, max_val) as u16;

            let idx_inv = get_pwl_idx_inv(sample);
            let inv_val = input_pivot[idx_inv]
                + ((inv_scale_coef[idx_inv] * (sample - reshape_pivot[idx_inv])
                    + (1 << (LMCS_FP_PREC - 1)))
                    >> LMCS_FP_PREC);
            inv_lut[sample as usize] = inv_val.clamp(0, max_val) as u16;
        }

        Self {
            bit_depth: bd,
            min_bin_idx: min_bin,
            max_bin_idx: max_bin,
            bin_cw,
            input_pivot,
            reshape_pivot,
            chroma_adj_lut,
            fwd_lut,
            inv_lut,
        }
    }

    /// Map a 16-bit luma [`Plane16`] forward into the reshaped domain (`Reshape.cpp:307-311`).
    pub fn apply_forward_plane_16(&self, plane: &mut Plane16) {
        let cap = self.fwd_lut.len().saturating_sub(1);
        for s in &mut plane.data {
            *s = self.fwd_lut[usize::from(*s).min(cap)];
        }
    }

    /// Map a reconstructed 16-bit luma [`Plane16`] from the reshaped domain back to the output
    /// sample domain before deblocking (`H.266 §8.8.2.3`, `Reshape.cpp:313-316`).
    pub fn apply_inverse_plane_16(&self, plane: &mut Plane16) {
        let cap = self.inv_lut.len().saturating_sub(1);
        for s in &mut plane.data {
            *s = self.inv_lut[usize::from(*s).min(cap)];
        }
    }

    /// Derive the 11-bit fixed-point chroma residual scaling factor `cScale` from `avg_luma`.
    pub fn chroma_scale_for_luma(&self, avg_luma: i32) -> i32 {
        let mut idx_s = self.min_bin_idx;
        while idx_s <= self.max_bin_idx {
            if avg_luma < self.reshape_pivot[idx_s + 1] {
                break;
            }
            idx_s += 1;
        }
        self.chroma_adj_lut[idx_s.min(15)]
    }
}

/// Normative 16-bit luma long-tap deblocking line filter (`maxFilterLengthP/Q` in `{3, 5, 7}`,
/// `bit_depth` in `8..=12`, ITU-T Rec. H.266 §8.8.3.6.8, VTM `DeblockingFilter.cpp:1389-1475`).
pub fn deblock_luma_long_tap_line_16(
    line: &mut [i32; 16],
    len_p: usize,
    len_q: usize,
    tc: i32,
    bit_depth: u8,
) {
    let bd = i32::from(bit_depth.clamp(8, 12));
    let max_val = (1i32 << bd) - 1;
    long_tap_luma_line(line, len_p, len_q, tc, max_val);
}

/// Derive optimal Sample Adaptive Offset (`SaoType::BandOffset` vs `SaoType::Off`) for an
/// 8-bit CTU region `(bx, by, w, h)` by minimising SSD against `src`
/// (VTM `EncSampleAdaptiveOffset.cpp:440-620`).
pub fn derive_sao_ctu_params(
    src: &Plane,
    recon: &Plane,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
) -> SaoType {
    let w_lim = (bx + w).min(src.width).min(recon.width);
    let h_lim = (by + h).min(src.height).min(recon.height);
    if w_lim <= bx || h_lim <= by {
        return SaoType::Off;
    }
    let mut sum_diff = [0i64; 32];
    let mut count = [0i64; 32];
    for y in by..h_lim {
        for x in bx..w_lim {
            let r = recon.get(x, y);
            let s = src.get(x, y);
            let band = usize::from(r >> 3);
            sum_diff[band] += i64::from(s) - i64::from(r);
            count[band] += 1;
        }
    }
    let mut band_offs = [0i32; 32];
    for b in 0..32 {
        if count[b] > 0 {
            let avg = (sum_diff[b] + (count[b] >> 1)) / count[b];
            band_offs[b] = (avg as i32).clamp(-7, 7);
        }
    }
    let mut best_pos = 0u8;
    let mut best_gain = 0i64;
    for pos in 0..28usize {
        let mut gain = 0i64;
        for k in 0..4 {
            let b = pos + k;
            let o = i64::from(band_offs[b]);
            if o != 0 {
                gain += 2 * o * sum_diff[b] - count[b] * o * o;
            }
        }
        if gain > best_gain {
            best_gain = gain;
            best_pos = pos as u8;
        }
    }
    if best_gain <= 0 {
        SaoType::Off
    } else {
        let p = usize::from(best_pos);
        SaoType::BandOffset {
            band_pos: best_pos,
            offsets: [
                band_offs[p],
                band_offs[p + 1],
                band_offs[p + 2],
                band_offs[p + 3],
            ],
        }
    }
}

/// Derive optimal Sample Adaptive Offset (`SaoType::BandOffset` vs `SaoType::Off`) for a
/// 16-bit [`Plane16`] CTU region `(bx, by, w, h)` (`bit_depth` in `8..=12`).
pub fn derive_sao_ctu_params_16(
    src: &Plane16,
    recon: &Plane16,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
) -> SaoType {
    let s8 = src.to_plane_8bit();
    let r8 = recon.to_plane_8bit();
    derive_sao_ctu_params(&s8, &r8, bx, by, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_lmcs_reshaper_preserves_all_256_levels() {
        let aps = VvcLmcsAps {
            min_bin_idx: 0,
            max_bin_idx: 15,
            bin_cw_delta: [0; 16],
            chroma_res_scaling_offset: 0,
        };
        let reshaper = LmcsReshaper::from_aps_8bit(&aps);
        for v in 0..=255u8 {
            assert_eq!(reshaper.fwd_lut[usize::from(v)], v);
            assert_eq!(reshaper.inv_lut[usize::from(v)], v);
        }
        assert_eq!(reshaper.chroma_scale_for_luma(128), 1 << LMCS_FP_PREC);
    }

    #[test]
    fn deblocking_smooths_step_edge_when_below_beta() {
        let mut lines = [[100, 100, 100, 100, 108, 108, 108, 108]; 4];
        deblock_luma_edge_4(&mut lines, 30, 30, 2, 0, 0);
        // Step discontinuity across p0=100, q0=108 must be reduced
        assert!(lines[0][3] > 100);
        assert!(lines[0][4] < 108);
    }

    #[test]
    fn sao_band_and_edge_offsets_modify_matching_samples() {
        let src = Plane::from_slice(8, 8, &[64u8; 64]).expect("src");
        let mut dst = src.clone();
        // 64 >> 3 == 8
        apply_sao_region(
            &src,
            &mut dst,
            0,
            0,
            8,
            8,
            SaoType::BandOffset {
                band_pos: 8,
                offsets: [5, -2, 0, 0],
            },
        );
        assert!(dst.data.iter().all(|&v| v == 69));
    }

    #[test]
    fn alf_preserves_constant_plane() {
        let mut plane = Plane::from_slice(16, 16, &[140u8; 256]).expect("plane");
        let aps = VvcAlfAps {
            luma_clip: true,
            luma_filters: vec![ALF_FIXED_FILTER_COEFF[4][..12].try_into().unwrap()],
            luma_clip_idx: vec![[0u8; 12]],
            filter_coeff_delta_idx: [0u8; 25],
            chroma_clip: false,
            chroma_filters: Vec::new(),
            chroma_clip_idx: Vec::new(),
            cc_cb_filters: Vec::new(),
            cc_cr_filters: Vec::new(),
        };
        apply_alf_luma_plane(&mut plane, &aps);
        assert!(plane.data.iter().all(|&v| v == 140));
    }
}

/// Minimal per-superblock SAO parameters used by the slice-level filter pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SaoBlkParams {
    /// Luma SAO type for the superblock.
    pub y: SaoType,
}
