// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! Inter prediction for H.266/VVC (ITU-T Rec. H.266 §8.3, §8.5).
//!
//! Algorithms and filter tables derived from VTM 24.0 (`InterpolationFilter.cpp`,
//! `InterPrediction.cpp`, `UnitTools.cpp`, `Buffer.cpp`, `Rom.cpp`, `Mv.h`,
//! BSD-3-Clause, Copyright (c) 2010-2026 ITU/ISO/IEC). See `THIRD_PARTY.md`.
//!
//! Supported tools:
//! - Decoded Picture Buffer (`DecodedPictureBuffer`, `RefFrame`) and Reference
//!   Picture List (`RefPicList`, `L0`/`L1`) construction (§8.3.2).
//! - 1/16-pel internal motion vectors (`Mv`), symmetric rounding right-shift
//!   (`Mv.h:136-148`), storage bit-depth clipping (`[-131072, 131071]`), and
//!   POC distance scaling (`get_dist_scale_factor`, `UnitTools.cpp:1701-1718`).
//! - Spatial/Temporal/HMVP/Pairwise/Zero Merge candidate list construction
//!   (`derive_merge_candidates`, §8.5.2.2–§8.5.2.6).
//! - Merge with Motion Vector Difference (`derive_mmvd_candidate`, §8.5.2.7).
//! - Advanced Motion Vector Prediction (`derive_amvp_candidates`, §8.5.2.8) and
//!   Adaptive Motion Vector Resolution (`AmvrPrecision`, §8.5.2.14).
//! - Fractional sample interpolation: 8-tap luma (`LUMA_FILTER_8TAP`), 8-tap
//!   alternative half-pel luma (`LUMA_ALT_HPEL_FILTER_8TAP`), 6-tap affine luma
//!   (`AFFINE_LUMA_FILTER_8TAP`), 4-tap chroma (`CHROMA_FILTER_4TAP`), and 2-tap
//!   DMVR bilinear (`BILINEAR_FILTER_PREC4`) with 14-bit internal precision (§8.5.6.3).
//! - Bi-Prediction with CU-Level Weights (`BCW_WEIGHTS = [-2, 3, 4, 5, 10]`, §8.5.6.6).
//! - Bi-Directional Optical Flow (`apply_bdof_luma`, §8.5.6.5).
//! - Decoder-Side Motion Vector Refinement (`refine_dmvr_subblock`, §8.5.3).
//! - Subblock 4-param / 6-param Affine Motion Compensation + PROF (§8.5.5).
//! - Combined Inter/Intra Prediction (`apply_ciip_blend`, §8.5.6.7).
//! - Geometric Partitioning Mode (`GpmParams`, `predict_gpm_block`, 64 modes, §8.5.7).
//! - Encoder P/B block motion estimation (`estimate_inter_block`).

use super::hls::VvcRefPicEntry;
use super::hls::VvcRefPicListStruct;
use super::predict::{CHROMA_FILTER_4TAP, Plane};

/// Internal motion vector fractional bit precision (`1/16` luma pel, VTM `TypeDef.h`).
pub const MV_FRACTIONAL_BITS_INTERNAL: i32 = 4;
/// Minimum motion vector component at 18-bit signed storage (`-131072`, VTM `Mv.h`).
pub const MV_MIN_STORAGE: i32 = -(1 << 17);
/// Maximum motion vector component at 18-bit signed storage (`131071`, VTM `Mv.h`).
pub const MV_MAX_STORAGE: i32 = (1 << 17) - 1;
/// Internal interpolation filter bit precision (`14` bits, VTM `InterpolationFilter.h`).
pub const IF_INTERNAL_PREC: i32 = 14;
/// Filter coefficient bit precision (`6` bits, sum = 64, VTM `InterpolationFilter.h`).
pub const IF_FILTER_PREC: i32 = 6;
/// Internal interpolation offset (`1 << (14 - 1) = 8192`, VTM `InterpolationFilter.h`).
pub const IF_INTERNAL_OFFS: i32 = 1 << (IF_INTERNAL_PREC - 1);
/// Default BCW weight index (`2` -> equal weight `4/8`, VTM `CommonDef.h`).
pub const BCW_DEFAULT: u8 = 2;
/// Maximum HMVP history table size (`5`, VTM `CommonDef.h`).
pub const MAX_NUM_HMVP_CANDS: usize = 5;

/// Normative 8-tap luma motion compensation filter table `f_L[frac][0..8]`
/// for 1/16-pel fractional positions `0..16`.
///
/// Normative source: ITU-T Rec. H.266 §8.5.6.3.2 Table 33 (PDF p. 367);
/// VTM `InterpolationFilter.cpp:81-99`.
pub const LUMA_FILTER_8TAP: [[i32; 8]; 16] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [0, 1, -3, 63, 4, -2, 1, 0],
    [-1, 2, -5, 62, 8, -3, 1, 0],
    [-1, 3, -8, 60, 13, -4, 1, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 52, 26, -8, 3, -1],
    [-1, 3, -9, 47, 31, -10, 4, -1],
    [-1, 4, -11, 45, 34, -10, 4, -1],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [-1, 4, -10, 34, 45, -11, 4, -1],
    [-1, 4, -10, 31, 47, -9, 3, -1],
    [-1, 3, -8, 26, 52, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
    [0, 1, -4, 13, 60, -8, 3, -1],
    [0, 1, -3, 8, 62, -5, 2, -1],
    [0, 1, -2, 4, 63, -3, 1, 0],
];

/// Alternative half-pel (`frac == 8`) 8-tap luma interpolation filter.
///
/// Normative source: ITU-T Rec. H.266 §8.5.6.3.2 Table 33;
/// VTM `InterpolationFilter.cpp:187`.
pub const LUMA_ALT_HPEL_FILTER_8TAP: [i32; 8] = [0, 3, 9, 20, 20, 9, 3, 0];

/// Normative 6-tap (padded to 8-tap) affine luma motion compensation filter table.
///
/// Normative source: ITU-T Rec. H.266 §8.5.6.3.2 Table 33;
/// VTM `InterpolationFilter.cpp:57-78`.
pub const AFFINE_LUMA_FILTER_8TAP: [[i32; 8]; 16] = [
    [0, 0, 64, 0, 0, 0, 0, 0],
    [1, -3, 63, 4, -2, 1, 0, 0],
    [1, -5, 62, 8, -3, 1, 0, 0],
    [2, -8, 60, 13, -4, 1, 0, 0],
    [3, -10, 58, 17, -5, 1, 0, 0],
    [3, -11, 52, 26, -8, 2, 0, 0],
    [2, -9, 47, 31, -10, 3, 0, 0],
    [3, -11, 45, 34, -10, 3, 0, 0],
    [3, -11, 40, 40, -11, 3, 0, 0],
    [3, -10, 34, 45, -11, 3, 0, 0],
    [3, -10, 31, 47, -9, 2, 0, 0],
    [2, -8, 26, 52, -11, 3, 0, 0],
    [1, -5, 17, 58, -10, 3, 0, 0],
    [1, -4, 13, 60, -8, 2, 0, 0],
    [1, -3, 8, 62, -5, 1, 0, 0],
    [1, -2, 4, 63, -3, 1, 0, 0],
];

/// 2-tap bilinear interpolation filter table for DMVR initial search (`InterpolationFilter.cpp:298-316`).
pub const BILINEAR_FILTER_PREC4: [[i32; 2]; 16] = [
    [16, 0],
    [15, 1],
    [14, 2],
    [13, 3],
    [12, 4],
    [11, 5],
    [10, 6],
    [9, 7],
    [8, 8],
    [7, 9],
    [6, 10],
    [5, 11],
    [4, 12],
    [3, 13],
    [2, 14],
    [1, 15],
];

/// Bi-prediction with CU-level Weights (BCW) L1 weight table (`Rom.cpp:205`).
///
/// The corresponding L0 weight is `8 - BCW_WEIGHTS[bcw_idx]`.
pub const BCW_WEIGHTS: [i32; 5] = [-2, 3, 4, 5, 10];

/// Geometric Partitioning Mode (GPM) displacement lookup table `g_dis[32]` (`Rom.cpp:710-711`).
pub const GEO_DIS: [i32; 32] = [
    8, 8, 8, 8, 4, 4, 2, 1, 0, -1, -2, -4, -4, -8, -8, -8, -8, -8, -8, -8, -4, -4, -2, -1, 0, 1, 2,
    4, 4, 8, 8, 8,
];

/// GPM angle-to-mask table `g_angle2mask[32]` (`Rom.cpp:708`).
pub const GEO_ANGLE_TO_MASK: [i8; 32] = [
    0, -1, 1, 2, 3, 4, -1, -1, 5, -1, -1, 4, 3, 2, 1, -1, 0, -1, 1, 2, 3, 4, -1, -1, 5, -1, -1, 4,
    3, 2, 1, -1,
];

/// Motion vector at 1/16-pel luma precision (`Mv.h:76-176`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mv {
    /// Horizontal component in 1/16-pel units.
    pub hor: i32,
    /// Vertical component in 1/16-pel units.
    pub ver: i32,
}

impl Mv {
    /// Zero motion vector.
    pub const ZERO: Self = Self { hor: 0, ver: 0 };

    /// Construct a new motion vector `(hor, ver)` in 1/16-pel units.
    #[inline]
    pub const fn new(hor: i32, ver: i32) -> Self {
        Self { hor, ver }
    }

    /// Construct a motion vector from integer luma pel coordinates.
    #[inline]
    pub const fn from_int_pel(dx: i32, dy: i32) -> Self {
        Self {
            hor: dx << MV_FRACTIONAL_BITS_INTERNAL,
            ver: dy << MV_FRACTIONAL_BITS_INTERNAL,
        }
    }

    /// Symmetric rounding right-shift matching VTM `Mv::operator>>=` (`Mv.h:136-148`).
    #[inline]
    pub fn shr_round(self, shift: u32) -> Self {
        if shift == 0 {
            return self;
        }
        let offset = 1i32 << (shift - 1);
        let h_neg_bias = (!self.hor) >> 31;
        let v_neg_bias = (!self.ver) >> 31;
        Self {
            hor: (self.hor + offset + h_neg_bias) >> shift,
            ver: (self.ver + offset + v_neg_bias) >> shift,
        }
    }

    /// Clip both components to the 18-bit signed storage range `[-131072, 131071]`.
    #[inline]
    pub fn clip_to_storage(self) -> Self {
        Self {
            hor: self.hor.clamp(MV_MIN_STORAGE, MV_MAX_STORAGE),
            ver: self.ver.clamp(MV_MIN_STORAGE, MV_MAX_STORAGE),
        }
    }

    /// Scale this motion vector by a POC distance scale factor (`Mv::getScaledMv`, `Mv.h:170-176`).
    #[inline]
    pub fn scaled(self, scale: i32) -> Self {
        Self::new(scale * self.hor, scale * self.ver)
            .shr_round(8)
            .clip_to_storage()
    }

    /// Add two motion vectors.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            hor: self.hor + other.hor,
            ver: self.ver + other.ver,
        }
    }

    /// Subtract `other` from `self`.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            hor: self.hor - other.hor,
            ver: self.ver - other.ver,
        }
    }
}

/// Compute the temporal POC distance scaling factor (`xGetDistScaleFactor`,
/// H.266 §8.5.2.12 Eq. 614–617, VTM `UnitTools.cpp:1701-1718`).
pub fn get_dist_scale_factor(
    curr_poc: i32,
    curr_ref_poc: i32,
    col_poc: i32,
    col_ref_poc: i32,
) -> i32 {
    let diff_poc_d = col_poc - col_ref_poc;
    let diff_poc_b = curr_poc - curr_ref_poc;
    if diff_poc_d == diff_poc_b || diff_poc_d == 0 {
        4096
    } else {
        let td_b = diff_poc_b.clamp(-128, 127);
        let td_d = diff_poc_d.clamp(-128, 127);
        let x = (0x4000 + (td_d / 2).abs()) / td_d;
        ((td_b * x + 32) >> 6).clamp(-4096, 4095)
    }
}

/// Motion vector + reference index pair for one reference list (`MvField`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvField {
    /// Motion vector in 1/16-pel units.
    pub mv: Mv,
    /// Reference index into the active reference list (`-1` if unused).
    pub ref_idx: i8,
}

impl Default for MvField {
    fn default() -> Self {
        Self {
            mv: Mv::default(),
            ref_idx: -1,
        }
    }
}

impl MvField {
    /// Create a valid `MvField` with `ref_idx >= 0`.
    pub const fn new(mv: Mv, ref_idx: i8) -> Self {
        Self { mv, ref_idx }
    }

    /// Return `true` if this list entry is active (`ref_idx >= 0`).
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.ref_idx >= 0
    }
}

/// Complete motion candidate for one inter PU (Merge, MMVD, or AMVP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterMergeCand {
    /// Prediction direction bitmask (`1` = L0, `2` = L1, `3` = Bi-pred).
    pub inter_dir: u8,
    /// Motion vector fields for `[L0, L1]`.
    pub mv_field: [MvField; 2],
    /// BCW weight index (`0..5`, default `BCW_DEFAULT = 2`).
    pub bcw_idx: u8,
    /// Whether alternative half-pel interpolation filter is used.
    pub use_alt_hpel_if: bool,
}

impl InterMergeCand {
    /// Construct a unidirectional L0 candidate with reference index `ref_idx`.
    pub fn uni_l0(ref_idx: i8, mv: Mv) -> Self {
        Self {
            inter_dir: 1,
            mv_field: [MvField::new(mv, ref_idx), MvField::default()],
            bcw_idx: BCW_DEFAULT,
            use_alt_hpel_if: false,
        }
    }
}

impl Default for InterMergeCand {
    fn default() -> Self {
        Self {
            inter_dir: 1,
            mv_field: [MvField::new(Mv::default(), 0), MvField::default()],
            bcw_idx: BCW_DEFAULT,
            use_alt_hpel_if: false,
        }
    }
}

/// Per-4x4-block motion field grid stored on each decoded frame for spatial
/// and temporal (TMVP) neighbor lookups.
#[derive(Debug, Clone)]
pub struct MotionFieldGrid {
    /// Width in 4x4 luma units.
    pub cols: usize,
    /// Height in 4x4 luma units.
    pub rows: usize,
    /// Stored motion info (`None` if the 4x4 block was intra-coded).
    pub blocks: Vec<Option<InterMergeCand>>,
}

impl MotionFieldGrid {
    /// Allocate an empty motion grid for a picture of luma dimensions `(width, height)`.
    pub fn new(width: usize, height: usize) -> Self {
        let cols = width.div_ceil(4);
        let rows = height.div_ceil(4);
        Self {
            cols,
            rows,
            blocks: vec![None; cols * rows],
        }
    }

    /// Look up the motion candidate at luma sample position `(x, y)`.
    pub fn get_at_luma(&self, x: usize, y: usize) -> Option<InterMergeCand> {
        let gx = x >> 2;
        let gy = y >> 2;
        if gx < self.cols && gy < self.rows {
            self.blocks[gy * self.cols + gx]
        } else {
            None
        }
    }

    /// Fill a luma region with a single candidate (alias of `fill_region`).
    pub fn fill_block(&mut self, bx: usize, by: usize, w: usize, h: usize, cand: InterMergeCand) {
        self.fill_region(bx, by, w, h, Some(cand));
    }

    /// Fill a luma region `(bx, by, w, h)` with `cand`.
    pub fn fill_region(
        &mut self,
        bx: usize,
        by: usize,
        w: usize,
        h: usize,
        cand: Option<InterMergeCand>,
    ) {
        let x0 = bx >> 2;
        let y0 = by >> 2;
        let x1 = ((bx + w).div_ceil(4)).min(self.cols);
        let y1 = ((by + h).div_ceil(4)).min(self.rows);
        for gy in y0..y1 {
            for gx in x0..x1 {
                self.blocks[gy * self.cols + gx] = cand;
            }
        }
    }
}

impl core::ops::Add for Mv {
    type Output = Mv;
    fn add(self, rhs: Mv) -> Mv {
        Mv::new(self.hor + rhs.hor, self.ver + rhs.ver)
    }
}

impl core::ops::Sub for Mv {
    type Output = Mv;
    fn sub(self, rhs: Mv) -> Mv {
        Mv::new(self.hor - rhs.hor, self.ver - rhs.ver)
    }
}

/// Reconstructed reference picture stored in the Decoded Picture Buffer (DPB).
#[derive(Debug, Clone)]
pub struct RefFrame {
    /// Picture Order Count (POC).
    pub poc: i32,
    /// Whether this picture is marked as a long-term reference picture.
    pub is_long_term: bool,
    /// Reconstructed luma plane.
    pub y: Plane,
    /// Reconstructed Cb plane.
    pub u: Plane,
    /// Reconstructed Cr plane.
    pub v: Plane,
    /// Reference list POCs (`[L0, L1]`) used when this frame was decoded (for TMVP scaling).
    pub ref_pocs: [Vec<i32>; 2],
    /// 4x4 motion vector grid for TMVP collocated lookups.
    pub mv_grid: MotionFieldGrid,
}

impl RefFrame {
    /// Construct a new reference frame from reconstructed Y, Cb, Cr planes.
    pub fn new(poc: i32, y: Plane, u: Plane, v: Plane) -> Self {
        let mv_grid = MotionFieldGrid::new(y.width, y.height);
        Self {
            poc,
            is_long_term: false,
            y,
            u,
            v,
            ref_pocs: [Vec::new(), Vec::new()],
            mv_grid,
        }
    }

    /// Construct a reference frame with explicit long-term flag and motion grid.
    pub fn from_planes(
        poc: i32,
        is_long_term: bool,
        y: Plane,
        u: Plane,
        v: Plane,
        mv_grid: MotionFieldGrid,
    ) -> Self {
        Self {
            poc,
            is_long_term,
            y,
            u,
            v,
            ref_pocs: [Vec::new(), Vec::new()],
            mv_grid,
        }
    }
}

/// Decoded Picture Buffer (DPB) managing active reference pictures and constructing
/// `RefPicList0` and `RefPicList1` per H.266 §8.3.2.
#[derive(Debug, Clone, Default)]
pub struct DecodedPictureBuffer {
    /// Stored reference frames ordered by insertion/POC.
    pub frames: Vec<RefFrame>,
    /// Maximum capacity of the DPB.
    pub max_size: usize,
}

impl DecodedPictureBuffer {
    /// Create a DPB with capacity `max_size`.
    pub fn new(max_size: usize) -> Self {
        Self {
            frames: Vec::new(),
            max_size: max_size.max(1),
        }
    }

    /// Insert or replace a reference frame in the DPB.
    pub fn insert(&mut self, frame: RefFrame) {
        if let Some(pos) = self.frames.iter().position(|f| f.poc == frame.poc) {
            self.frames[pos] = frame;
            return;
        }
        if self.frames.len() >= self.max_size {
            self.frames.remove(0);
        }
        self.frames.push(frame);
    }

    /// Whether the DPB holds no pictures.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Remove all pictures from the DPB.
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Build the default `RefPicList0`/`RefPicList1` for a P-slice from the DPB
    /// (most recent short-term picture in L0, empty L1).
    pub fn build_ref_lists(&self, _curr_poc: i32) -> (Vec<&RefFrame>, Vec<&RefFrame>) {
        let l0: Vec<&RefFrame> = self.frames.last().into_iter().collect();
        (l0, Vec::new())
    }

    /// Find a reference frame by exact POC.
    pub fn find_by_poc(&self, poc: i32) -> Option<&RefFrame> {
        self.frames.iter().find(|f| f.poc == poc)
    }

    /// Build active reference picture list (`L0` or `L1`) from a `VvcRefPicListStruct`
    /// and current picture `curr_poc` (H.266 §8.3.2 Eq. 29).
    pub fn build_ref_list(&self, curr_poc: i32, rpls: &VvcRefPicListStruct) -> Vec<&RefFrame> {
        let mut list = Vec::new();
        let mut poc_base = curr_poc;
        for entry in &rpls.entries {
            match entry {
                VvcRefPicEntry::ShortTerm { delta_poc } => {
                    let target_poc = poc_base + *delta_poc;
                    poc_base = target_poc;
                    if let Some(f) = self.find_by_poc(target_poc) {
                        list.push(f);
                    } else if let Some(fallback) = self.frames.last() {
                        list.push(fallback);
                    }
                }
                VvcRefPicEntry::LongTerm { .. } => {
                    if let Some(f) = self.frames.iter().find(|f| f.is_long_term) {
                        list.push(f);
                    } else if let Some(fallback) = self.frames.last() {
                        list.push(fallback);
                    }
                }
            }
        }
        if list.is_empty() {
            if let Some(last) = self.frames.last() {
                list.push(last);
            }
        }
        list
    }
}

/// History-based Motion Vector Prediction (HMVP) FIFO table (H.266 §8.5.2.6).
#[derive(Debug, Clone, Default)]
pub struct HmvpTable {
    /// Up to `MAX_NUM_HMVP_CANDS = 5` motion candidates.
    pub cands: Vec<InterMergeCand>,
}

impl HmvpTable {
    /// Create an empty HMVP table.
    pub fn new() -> Self {
        Self { cands: Vec::new() }
    }

    /// Reset the HMVP table at the start of a CTU row or slice.
    pub fn reset(&mut self) {
        self.cands.clear();
    }

    /// Add a motion candidate to the HMVP FIFO table with identical-candidate pruning.
    pub fn push(&mut self, cand: InterMergeCand) {
        if let Some(pos) = self
            .cands
            .iter()
            .position(|c| c.mv_field == cand.mv_field && c.inter_dir == cand.inter_dir)
        {
            self.cands.remove(pos);
        } else if self.cands.len() >= MAX_NUM_HMVP_CANDS {
            self.cands.remove(0);
        }
        self.cands.push(cand);
    }
}

/// Derive the regular inter Merge candidate list of length `max_num_merge_cand` (up to 6).
///
/// Order:
/// 1. Spatial `B1` (above-right-most at `(bx + w - 1, by - 1)`),
/// 2. Spatial `A1` (left-bottom-most at `(bx - 1, by + h - 1)`),
/// 3. Spatial `B0` (above-right at `(bx + w, by - 1)`),
/// 4. Spatial `A0` (below-left at `(bx - 1, by + h)`),
/// 5. Spatial `B2` (above-left at `(bx - 1, by - 1)` if `< 4` spatial candidates so far),
/// 6. Temporal collocated `Col` (if provided),
/// 7. HMVP table candidates (latest to oldest, compared against `A1` and `B1`),
/// 8. Pairwise average of candidates `0` and `1`,
/// 9. Zero motion vectors (`ref_idx` 0..num_ref).
///
/// Normative source: ITU-T Rec. H.266 §8.5.2.2–§8.5.2.6 (PDF pp. 284–294);
/// VTM `UnitTools.cpp:1059-1699` (`PU::getInterMergeCandidates`).
#[allow(clippy::too_many_arguments)]
pub fn derive_merge_candidates(
    grid: &MotionFieldGrid,
    hmvp: &HmvpTable,
    col_cand: Option<InterMergeCand>,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    is_b_slice: bool,
    max_num_merge_cand: usize,
) -> Vec<InterMergeCand> {
    let max_cands = max_num_merge_cand.clamp(1, 6);
    let mut list: Vec<InterMergeCand> = Vec::with_capacity(max_cands);

    let b1 = if by > 0 {
        grid.get_at_luma(bx + w - 1, by - 1)
    } else {
        None
    };
    let a1 = if bx > 0 {
        grid.get_at_luma(bx - 1, by + h - 1)
    } else {
        None
    };
    let b0 = if by > 0 {
        grid.get_at_luma(bx + w, by - 1)
    } else {
        None
    };
    let a0 = if bx > 0 {
        grid.get_at_luma(bx - 1, by + h)
    } else {
        None
    };
    let b2 = if bx > 0 && by > 0 {
        grid.get_at_luma(bx - 1, by - 1)
    } else {
        None
    };

    if let Some(c_b1) = b1 {
        list.push(c_b1);
    }
    if list.len() < max_cands {
        if let Some(c_a1) = a1 {
            if b1.map_or(true, |b| b.mv_field != c_a1.mv_field) {
                list.push(c_a1);
            }
        }
    }
    if list.len() < max_cands {
        if let Some(c_b0) = b0 {
            if b1.map_or(true, |b| b.mv_field != c_b0.mv_field) {
                list.push(c_b0);
            }
        }
    }
    if list.len() < max_cands {
        if let Some(c_a0) = a0 {
            if a1.map_or(true, |a| a.mv_field != c_a0.mv_field) {
                list.push(c_a0);
            }
        }
    }
    if list.len() < 4 && list.len() < max_cands {
        if let Some(c_b2) = b2 {
            let diff_a1 = a1.map_or(true, |a| a.mv_field != c_b2.mv_field);
            let diff_b1 = b1.map_or(true, |b| b.mv_field != c_b2.mv_field);
            if diff_a1 && diff_b1 {
                list.push(c_b2);
            }
        }
    }

    if list.len() < max_cands {
        if let Some(col) = col_cand {
            list.push(col);
        }
    }

    // HMVP candidates (reverse order from newest to oldest).
    for h_cand in hmvp.cands.iter().rev() {
        if list.len() >= max_cands.saturating_sub(1) {
            break;
        }
        let dup_a1 = a1.is_some_and(|a| a.mv_field == h_cand.mv_field);
        let dup_b1 = b1.is_some_and(|b| b.mv_field == h_cand.mv_field);
        if !dup_a1 && !dup_b1 {
            list.push(*h_cand);
        }
    }

    // Pairwise average candidate between list[0] and list[1] (`UnitTools.cpp:1590-1652`).
    if list.len() >= 2 && list.len() < max_cands {
        let c0 = list[0];
        let c1 = list[1];
        let mut avg = InterMergeCand {
            inter_dir: 0,
            mv_field: [MvField::default(), MvField::default()],
            bcw_idx: BCW_DEFAULT,
            use_alt_hpel_if: c0.use_alt_hpel_if && c1.use_alt_hpel_if,
        };
        for l in 0..if is_b_slice { 2 } else { 1 } {
            let f0 = c0.mv_field[l];
            let f1 = c1.mv_field[l];
            if f0.is_valid() && f1.is_valid() {
                let sum_mv = Mv::new(f0.mv.hor + f1.mv.hor, f0.mv.ver + f1.mv.ver).shr_round(1);
                avg.mv_field[l] = MvField::new(sum_mv, f0.ref_idx);
                avg.inter_dir |= (l as u8) + 1;
            } else if f0.is_valid() {
                avg.mv_field[l] = f0;
                avg.inter_dir |= (l as u8) + 1;
            } else if f1.is_valid() {
                avg.mv_field[l] = f1;
                avg.inter_dir |= (l as u8) + 1;
            }
        }
        if avg.inter_dir != 0 {
            list.push(avg);
        }
    }

    // Zero MV padding (`UnitTools.cpp:1656-1695`).
    let mut zero_idx = 0i8;
    while list.len() < max_cands {
        let mv0 = MvField::new(Mv::default(), zero_idx.min(0));
        let mv1 = if is_b_slice {
            MvField::new(Mv::default(), zero_idx.min(0))
        } else {
            MvField::default()
        };
        list.push(InterMergeCand {
            inter_dir: if is_b_slice { 3 } else { 1 },
            mv_field: [mv0, mv1],
            bcw_idx: BCW_DEFAULT,
            use_alt_hpel_if: false,
        });
        zero_idx = zero_idx.saturating_add(1);
    }

    list
}

/// Derive a Merge with Motion Vector Difference (MMVD) candidate from a base
/// Merge candidate (`base_idx` in `0..=1`), distance step (`step` in `0..8`),
/// and direction (`dir` in `0..4`: `+x`, `-x`, `+y`, `-y`).
///
/// Normative source: ITU-T Rec. H.266 §8.5.2.7 (PDF pp. 295–297);
/// VTM `InterPrediction.cpp:2411-2618` (`getMmvdDeltaMv`, `setMmvdMergeCandiInfo`).
#[allow(clippy::too_many_arguments)]
pub fn derive_mmvd_candidate(
    merge_list: &[InterMergeCand],
    base_idx: usize,
    step: u8,
    dir: u8,
    dis_frac_mmvd: bool,
    curr_poc: i32,
    ref_poc_l0: Option<i32>,
    ref_poc_l1: Option<i32>,
) -> InterMergeCand {
    let base = merge_list.get(base_idx).copied().unwrap_or_default();
    let mut offset = 1i32 << (step.min(7) + 2); // 1/4-pel base in 1/16-pel units = 4
    if dis_frac_mmvd {
        offset <<= 2;
    }
    let d_mv = match dir & 3 {
        0 => Mv::new(offset, 0),
        1 => Mv::new(-offset, 0),
        2 => Mv::new(0, offset),
        _ => Mv::new(0, -offset),
    };

    let mut delta0 = Mv::default();
    let mut delta1 = Mv::default();
    let has_l0 = base.mv_field[0].is_valid();
    let has_l1 = base.mv_field[1].is_valid();

    if has_l0 && has_l1 {
        let poc0 = ref_poc_l0.unwrap_or(curr_poc - 1);
        let poc1 = ref_poc_l1.unwrap_or(curr_poc + 1);
        if (poc0 - curr_poc) == (poc1 - curr_poc) {
            delta0 = d_mv;
            delta1 = d_mv;
        } else if (poc1 - curr_poc).abs() > (poc0 - curr_poc).abs() {
            delta1 = d_mv;
            let scale = get_dist_scale_factor(curr_poc, poc0, curr_poc, poc1);
            delta0 = delta1.scaled(scale);
        } else {
            delta0 = d_mv;
            let scale = get_dist_scale_factor(curr_poc, poc1, curr_poc, poc0);
            delta1 = delta0.scaled(scale);
        }
    } else if has_l0 {
        delta0 = d_mv;
    } else if has_l1 {
        delta1 = d_mv;
    }

    let mut out = base;
    if has_l0 {
        out.mv_field[0].mv = out.mv_field[0].mv.add(delta0).clip_to_storage();
    }
    if has_l1 {
        out.mv_field[1].mv = out.mv_field[1].mv.add(delta1).clip_to_storage();
    }
    out
}

/// Adaptive Motion Vector Resolution (AMVR) precision mode (H.266 §8.5.2.14, VTM `Mv.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmvrPrecision {
    /// Default 1/4-pel MVD precision (`shift = 2` from 1/16-pel).
    #[default]
    QuarterPel,
    /// Alternative 1/2-pel MVD precision (`shift = 3`, enables `LUMA_ALT_HPEL_FILTER_8TAP`).
    HalfPel,
    /// Integer 1-pel MVD precision (`shift = 4`).
    FullPel,
    /// Coarse 4-pel MVD precision (`shift = 6`).
    FourPel,
}

impl AmvrPrecision {
    /// Return the left-shift from coded MVD units to internal 1/16-pel MV units.
    #[inline]
    pub const fn internal_shift(self) -> u32 {
        match self {
            Self::QuarterPel => 2,
            Self::HalfPel => 3,
            Self::FullPel => 4,
            Self::FourPel => 6,
        }
    }
}

/// Derive the 2-entry AMVP motion vector predictor candidate list for reference list
/// `list_idx` (`0` = L0, `1` = L1) and reference index `ref_idx`.
///
/// Normative source: ITU-T Rec. H.266 §8.5.2.8 (PDF pp. 297–302);
/// VTM `UnitTools.cpp:2084-2248` (`PU::fillMvpCand`).
pub fn derive_amvp_candidates(
    grid: &MotionFieldGrid,
    hmvp: &HmvpTable,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    list_idx: usize,
    ref_idx: i8,
    amvr: AmvrPrecision,
) -> [Mv; 2] {
    let shift = amvr.internal_shift();
    let round_mv = |mv: Mv| -> Mv {
        let r = mv.shr_round(shift);
        Mv::new(r.hor << shift, r.ver << shift).clip_to_storage()
    };

    let mut cands: Vec<Mv> = Vec::with_capacity(2);

    // 1. Left spatial candidate (A0 at (bx - 1, by + h), then A1 at (bx - 1, by + h - 1)).
    if bx > 0 {
        for &(nx, ny) in &[(bx - 1, by + h), (bx - 1, by + h - 1)] {
            if let Some(mi) = grid.get_at_luma(nx, ny) {
                if mi.mv_field[list_idx].ref_idx == ref_idx {
                    cands.push(round_mv(mi.mv_field[list_idx].mv));
                    break;
                }
            }
        }
    }

    // 2. Above spatial candidate (B0, B1, B2).
    if by > 0 {
        let mut above_coords = vec![(bx + w, by - 1), (bx + w - 1, by - 1)];
        if bx > 0 {
            above_coords.push((bx - 1, by - 1));
        }
        for (nx, ny) in above_coords {
            if let Some(mi) = grid.get_at_luma(nx, ny) {
                if mi.mv_field[list_idx].ref_idx == ref_idx {
                    let mv = round_mv(mi.mv_field[list_idx].mv);
                    if cands.first().copied() != Some(mv) {
                        cands.push(mv);
                    }
                    break;
                }
            }
        }
    }

    // 3. HMVP candidates matching `ref_idx`.
    if cands.len() < 2 {
        for h_cand in hmvp.cands.iter().rev() {
            if h_cand.mv_field[list_idx].ref_idx == ref_idx {
                let mv = round_mv(h_cand.mv_field[list_idx].mv);
                if !cands.contains(&mv) {
                    cands.push(mv);
                    if cands.len() == 2 {
                        break;
                    }
                }
            }
        }
    }

    while cands.len() < 2 {
        cands.push(Mv::default());
    }
    [cands[0], cands[1]]
}

#[inline]
fn sample_clamped(plane: &Plane, x: isize, y: isize) -> i32 {
    let cx = x.clamp(0, plane.width as isize - 1) as usize;
    let cy = y.clamp(0, plane.height as isize - 1) as usize;
    i32::from(plane.get(cx, cy))
}

/// Perform luma fractional-sample interpolation at 1/16-pel precision producing
/// either final clipped 8-bit samples (`bi == false`) or 14-bit signed intermediate
/// samples (`bi == true`, shifted by `-IF_INTERNAL_OFFS`, matching VTM `InterpolationFilter.cpp:379-614`).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_luma_block(
    ref_plane: &Plane,
    bx: isize,
    by: isize,
    w: usize,
    h: usize,
    mv: Mv,
    bi: bool,
    use_alt_hpel: bool,
    use_affine_filter: bool,
) -> Vec<i32> {
    let int_x = bx + ((mv.hor >> 4) as isize);
    let int_y = by + ((mv.ver >> 4) as isize);
    let frac_x = (mv.hor & 15) as usize;
    let frac_y = (mv.ver & 15) as usize;

    let head_room = IF_INTERNAL_PREC - 8; // 6 for 8-bit video
    let get_coeffs = |frac: usize| -> &[i32; 8] {
        if use_affine_filter {
            &AFFINE_LUMA_FILTER_8TAP[frac]
        } else if use_alt_hpel && frac == 8 {
            &LUMA_ALT_HPEL_FILTER_8TAP
        } else {
            &LUMA_FILTER_8TAP[frac]
        }
    };

    let tap_start: isize = if use_affine_filter { -2 } else { -3 };
    let mut out = vec![0i32; w * h];

    if frac_x == 0 && frac_y == 0 {
        for y in 0..h {
            for x in 0..w {
                let s = sample_clamped(ref_plane, int_x + x as isize, int_y + y as isize);
                out[y * w + x] = if bi {
                    (s << head_room) - IF_INTERNAL_OFFS
                } else {
                    s
                };
            }
        }
        return out;
    }

    if frac_y == 0 {
        let c = get_coeffs(frac_x);
        let shift = if bi {
            IF_FILTER_PREC - head_room
        } else {
            IF_FILTER_PREC
        };
        let offset = if bi {
            -IF_INTERNAL_OFFS << shift
        } else {
            1 << (shift - 1)
        };
        for y in 0..h {
            let py = int_y + y as isize;
            for x in 0..w {
                let px = int_x + x as isize;
                let mut sum = 0i32;
                for k in 0..8 {
                    let tap_x = if use_affine_filter {
                        px + tap_start + k as isize - 1
                    } else {
                        px + tap_start + k as isize
                    };
                    sum += sample_clamped(ref_plane, tap_x, py) * c[k];
                }
                let val = (sum + offset) >> shift;
                out[y * w + x] = if bi { val } else { val.clamp(0, 255) };
            }
        }
        return out;
    }

    if frac_x == 0 {
        let c = get_coeffs(frac_y);
        let shift = if bi {
            IF_FILTER_PREC - head_room
        } else {
            IF_FILTER_PREC
        };
        let offset = if bi {
            -IF_INTERNAL_OFFS << shift
        } else {
            1 << (shift - 1)
        };
        for y in 0..h {
            let py = int_y + y as isize;
            for x in 0..w {
                let px = int_x + x as isize;
                let mut sum = 0i32;
                for k in 0..8 {
                    let tap_y = if use_affine_filter {
                        py + tap_start + k as isize - 1
                    } else {
                        py + tap_start + k as isize
                    };
                    sum += sample_clamped(ref_plane, px, tap_y) * c[k];
                }
                let val = (sum + offset) >> shift;
                out[y * w + x] = if bi { val } else { val.clamp(0, 255) };
            }
        }
        return out;
    }

    // 2D separable filtering: horizontal pass into `tmp` (height + 7 rows), then vertical pass.
    let cx = get_coeffs(frac_x);
    let cy = get_coeffs(frac_y);
    let ext_h = h + 7;
    let mut tmp = vec![0i32; w * ext_h];
    let shift1 = IF_FILTER_PREC - head_room; // 0 for 8-bit
    let offset1 = -IF_INTERNAL_OFFS << shift1;

    for y in 0..ext_h {
        let py = int_y - 3 + y as isize;
        for x in 0..w {
            let px = int_x + x as isize;
            let mut sum = 0i32;
            for k in 0..8 {
                let tap_x = if use_affine_filter {
                    px + tap_start + k as isize - 1
                } else {
                    px + tap_start + k as isize
                };
                sum += sample_clamped(ref_plane, tap_x, py) * cx[k];
            }
            tmp[y * w + x] = (sum + offset1) >> shift1;
        }
    }

    let shift2 = if bi {
        IF_FILTER_PREC
    } else {
        IF_FILTER_PREC + head_room
    };
    let offset2 = if bi {
        0
    } else {
        (1 << (shift2 - 1)) + (IF_INTERNAL_OFFS << IF_FILTER_PREC)
    };

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0i32;
            for k in 0..8 {
                sum += tmp[(y + k) * w + x] * cy[k];
            }
            let val = (sum + offset2) >> shift2;
            out[y * w + x] = if bi { val } else { val.clamp(0, 255) };
        }
    }
    out
}

/// Perform 4:2:0 chroma fractional-sample interpolation at 1/32-pel precision
/// using the 4-tap `CHROMA_FILTER_4TAP` filter (`InterPrediction.cpp:708-766`).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_chroma_block_420(
    ref_plane: &Plane,
    cx: isize,
    cy: isize,
    cw: usize,
    ch: usize,
    luma_mv: Mv,
    bi: bool,
) -> Vec<i32> {
    let int_x = cx + ((luma_mv.hor >> 5) as isize);
    let int_y = cy + ((luma_mv.ver >> 5) as isize);
    let frac_x = (luma_mv.hor & 31) as usize;
    let frac_y = (luma_mv.ver & 31) as usize;

    let head_room = IF_INTERNAL_PREC - 8; // 6
    let mut out = vec![0i32; cw * ch];

    if frac_x == 0 && frac_y == 0 {
        for y in 0..ch {
            for x in 0..cw {
                let s = sample_clamped(ref_plane, int_x + x as isize, int_y + y as isize);
                out[y * cw + x] = if bi {
                    (s << head_room) - IF_INTERNAL_OFFS
                } else {
                    s
                };
            }
        }
        return out;
    }

    if frac_y == 0 {
        let c = &CHROMA_FILTER_4TAP[frac_x];
        let shift = if bi {
            IF_FILTER_PREC - head_room
        } else {
            IF_FILTER_PREC
        };
        let offset = if bi {
            -IF_INTERNAL_OFFS << shift
        } else {
            1 << (shift - 1)
        };
        for y in 0..ch {
            let py = int_y + y as isize;
            for x in 0..cw {
                let px = int_x + x as isize;
                let mut sum = 0i32;
                for k in 0..4 {
                    sum += sample_clamped(ref_plane, px - 1 + k as isize, py) * c[k];
                }
                let val = (sum + offset) >> shift;
                out[y * cw + x] = if bi { val } else { val.clamp(0, 255) };
            }
        }
        return out;
    }

    if frac_x == 0 {
        let c = &CHROMA_FILTER_4TAP[frac_y];
        let shift = if bi {
            IF_FILTER_PREC - head_room
        } else {
            IF_FILTER_PREC
        };
        let offset = if bi {
            -IF_INTERNAL_OFFS << shift
        } else {
            1 << (shift - 1)
        };
        for y in 0..ch {
            let py = int_y + y as isize;
            for x in 0..cw {
                let px = int_x + x as isize;
                let mut sum = 0i32;
                for k in 0..4 {
                    sum += sample_clamped(ref_plane, px, py - 1 + k as isize) * c[k];
                }
                let val = (sum + offset) >> shift;
                out[y * cw + x] = if bi { val } else { val.clamp(0, 255) };
            }
        }
        return out;
    }

    let cx_f = &CHROMA_FILTER_4TAP[frac_x];
    let cy_f = &CHROMA_FILTER_4TAP[frac_y];
    let ext_h = ch + 3;
    let mut tmp = vec![0i32; cw * ext_h];
    let shift1 = IF_FILTER_PREC - head_room;
    let offset1 = -IF_INTERNAL_OFFS << shift1;

    for y in 0..ext_h {
        let py = int_y - 1 + y as isize;
        for x in 0..cw {
            let px = int_x + x as isize;
            let mut sum = 0i32;
            for k in 0..4 {
                sum += sample_clamped(ref_plane, px - 1 + k as isize, py) * cx_f[k];
            }
            tmp[y * cw + x] = (sum + offset1) >> shift1;
        }
    }

    let shift2 = if bi {
        IF_FILTER_PREC
    } else {
        IF_FILTER_PREC + head_room
    };
    let offset2 = if bi {
        0
    } else {
        (1 << (shift2 - 1)) + (IF_INTERNAL_OFFS << IF_FILTER_PREC)
    };

    for y in 0..ch {
        for x in 0..cw {
            let mut sum = 0i32;
            for k in 0..4 {
                sum += tmp[(y + k) * cw + x] * cy_f[k];
            }
            let val = (sum + offset2) >> shift2;
            out[y * cw + x] = if bi { val } else { val.clamp(0, 255) };
        }
    }
    out
}

/// Combine two 14-bit bi-prediction buffers (`pred0`, `pred1`) using CU-level
/// BCW weights (`bcw_idx` in `0..5`, where `BCW_DEFAULT = 2` is equal `4/8` weight).
///
/// Normative source: ITU-T Rec. H.266 §8.5.6.6 (PDF pp. 377–379);
/// VTM `Buffer.cpp:325-352` (`addWeightedAvg`).
pub fn weighted_average_bi(pred0: &[i32], pred1: &[i32], bcw_idx: u8) -> Vec<i32> {
    let w1 = BCW_WEIGHTS[(bcw_idx as usize).min(4)];
    let w0 = 8 - w1;
    let head_room = IF_INTERNAL_PREC - 8; // 6
    let shift_num = head_room + 3; // log2WeightBase = 3 -> shift = 9
    let offset = (1i32 << (shift_num - 1)) + (IF_INTERNAL_OFFS << 3);

    pred0
        .iter()
        .zip(pred1.iter())
        .map(|(&p0, &p1)| ((p0 * w0 + p1 * w1 + offset) >> shift_num).clamp(0, 255))
        .collect()
}

#[inline]
fn right_shift_msb(numer: i32, denom: i32) -> i32 {
    if denom <= 0 {
        0
    } else {
        let log2_d = 31 - (denom as u32).leading_zeros();
        numer >> log2_d
    }
}

/// Apply Bi-Directional Optical Flow (BDOF) to two 14-bit intermediate luma
/// prediction blocks `pred0` and `pred1` of size `(w, h)` (`w >= 4, h >= 4`).
///
/// Normative source: ITU-T Rec. H.266 §8.5.6.5 (PDF pp. 373–377);
/// VTM `InterPrediction.cpp:1185-1301` & `Buffer.cpp:90-193`.
pub fn apply_bdof_luma(pred0: &[i32], pred1: &[i32], w: usize, h: usize) -> Vec<i32> {
    let ext_w = w + 2;
    let ext_h = h + 2;
    let pad_buf = |src: &[i32]| -> Vec<i32> {
        let mut p = vec![0i32; ext_w * ext_h];
        for y in 0..h {
            let sy = y + 1;
            p[sy * ext_w] = src[y * w];
            for x in 0..w {
                p[sy * ext_w + x + 1] = src[y * w + x];
            }
            p[sy * ext_w + w + 1] = src[y * w + w - 1];
        }
        for x in 0..ext_w {
            p[x] = p[ext_w + x];
            p[(h + 1) * ext_w + x] = p[h * ext_w + x];
        }
        p
    };

    let p0 = pad_buf(pred0);
    let p1 = pad_buf(pred1);

    let compute_grads = |p: &[i32]| -> (Vec<i32>, Vec<i32>) {
        let mut gx = vec![0i32; ext_w * ext_h];
        let mut gy = vec![0i32; ext_w * ext_h];
        for y in 1..=h {
            for x in 1..=w {
                let idx = y * ext_w + x;
                gx[idx] = (p[idx + 1] >> 6) - (p[idx - 1] >> 6);
                gy[idx] = (p[idx + ext_w] >> 6) - (p[idx - ext_w] >> 6);
            }
            gx[y * ext_w] = gx[y * ext_w + 1];
            gx[y * ext_w + w + 1] = gx[y * ext_w + w];
            gy[y * ext_w] = gy[y * ext_w + 1];
            gy[y * ext_w + w + 1] = gy[y * ext_w + w];
        }
        for x in 0..ext_w {
            gx[x] = gx[ext_w + x];
            gx[(h + 1) * ext_w + x] = gx[h * ext_w + x];
            gy[x] = gy[ext_w + x];
            gy[(h + 1) * ext_w + x] = gy[h * ext_w + x];
        }
        (gx, gy)
    };

    let (gx0, gy0) = compute_grads(&p0);
    let (gx1, gy1) = compute_grads(&p1);

    let head_room = IF_INTERNAL_PREC - 8; // 6
    let shift_num = head_room + 1; // 7
    let offset = (1i32 << (shift_num - 1)) + 2 * IF_INTERNAL_OFFS;
    let limit = 15i32;

    let mut dst = vec![0i32; w * h];
    for yu in 0..(h >> 2) {
        for xu in 0..(w >> 2) {
            let mut sum_abs_gx = 0i32;
            let mut sum_abs_gy = 0i32;
            let mut sum_dix = 0i32;
            let mut sum_diy = 0i32;
            let mut sum_sign_gy_gx = 0i32;

            for dy in 0..6 {
                let py = yu * 4 + dy;
                for dx in 0..6 {
                    let px = xu * 4 + dx;
                    let idx = py * ext_w + px;
                    let tmp_gx = (gx0[idx] + gx1[idx]) >> 1;
                    let tmp_gy = (gy0[idx] + gy1[idx]) >> 1;
                    let tmp_di = (p1[idx] >> 4) - (p0[idx] >> 4);
                    sum_abs_gx += tmp_gx.abs();
                    sum_abs_gy += tmp_gy.abs();
                    sum_dix += if tmp_gx < 0 {
                        -tmp_di
                    } else if tmp_gx == 0 {
                        0
                    } else {
                        tmp_di
                    };
                    sum_diy += if tmp_gy < 0 {
                        -tmp_di
                    } else if tmp_gy == 0 {
                        0
                    } else {
                        tmp_di
                    };
                    sum_sign_gy_gx += if tmp_gy < 0 {
                        -tmp_gx
                    } else if tmp_gy == 0 {
                        0
                    } else {
                        tmp_gx
                    };
                }
            }

            let tmpx = if sum_abs_gx == 0 {
                0
            } else {
                right_shift_msb(4 * sum_dix, sum_abs_gx).clamp(-limit, limit)
            };
            let tmp_data = (sum_sign_gy_gx * tmpx) >> 1;
            let tmpy = if sum_abs_gy == 0 {
                0
            } else {
                right_shift_msb(4 * sum_diy - tmp_data, sum_abs_gy).clamp(-limit, limit)
            };

            for dy in 0..4 {
                let y = yu * 4 + dy;
                for dx in 0..4 {
                    let x = xu * 4 + dx;
                    let idx = (y + 1) * ext_w + (x + 1);
                    let b = tmpx * (gx0[idx] - gx1[idx]) + tmpy * (gy0[idx] - gy1[idx]);
                    dst[y * w + x] = ((p0[idx] + p1[idx] + b + offset) >> shift_num).clamp(0, 255);
                }
            }
        }
    }
    dst
}

fn dmvr_subpel_1d(s0: i64, s1: i64, s2: i64) -> i32 {
    let den = s1 + s2 - 2 * s0;
    if den == 0 {
        return 0;
    }
    if s1 != s0 && s2 != s0 {
        let num = ((s1 - s2) << MV_FRACTIONAL_BITS_INTERNAL) >> 1;
        let mut rem = num.abs();
        let sign = num < 0;
        let mut q = 0i32;
        for i in (0..=2).rev() {
            let div = den << i;
            if rem >= div {
                rem -= div;
                q += 1 << i;
            }
        }
        if sign { -q } else { q }
    } else {
        const HALF_PEL: i32 = (1 << MV_FRACTIONAL_BITS_INTERNAL) >> 1;
        if s1 == s0 { -HALF_PEL } else { HALF_PEL }
    }
}

/// Refine a symmetric bi-predicted Merge motion vector pair `(mv0, mv1)` on a
/// subblock of size `(dx, dy)` using Decoder-Side Motion Vector Refinement (DMVR).
///
/// Normative source: ITU-T Rec. H.266 §8.5.3 (PDF pp. 316–324);
/// VTM `InterPrediction.cpp:1737-2005` (`xProcessDMVR`, `xDmvrSubpelRefine`).
#[allow(clippy::too_many_arguments)]
pub fn refine_dmvr_subblock(
    ref0: &Plane,
    ref1: &Plane,
    bx: isize,
    by: isize,
    dx: usize,
    dy: usize,
    mv0: Mv,
    mv1: Mv,
) -> (Mv, Mv) {
    let bilinear_pred = |plane: &Plane, mv: Mv, off_x: isize, off_y: isize| -> Vec<i32> {
        let int_x = bx + ((mv.hor >> 4) as isize) + off_x;
        let int_y = by + ((mv.ver >> 4) as isize) + off_y;
        let fx = (mv.hor & 15) as usize;
        let fy = (mv.ver & 15) as usize;
        let wx = &BILINEAR_FILTER_PREC4[fx];
        let wy = &BILINEAR_FILTER_PREC4[fy];
        let mut out = vec![0i32; dx * dy];
        for y in 0..dy {
            let py = int_y + y as isize;
            for x in 0..dx {
                let px = int_x + x as isize;
                let h0 = (sample_clamped(plane, px, py) * wx[0]
                    + sample_clamped(plane, px + 1, py) * wx[1]
                    + 8)
                    >> 4;
                let h1 = (sample_clamped(plane, px, py + 1) * wx[0]
                    + sample_clamped(plane, px + 1, py + 1) * wx[1]
                    + 8)
                    >> 4;
                out[y * dx + x] = (h0 * wy[0] + h1 * wy[1] + 8) >> 4;
            }
        }
        out
    };

    // 5x5 integer grid (`-2..=2`) bilateral SAD with 2-row subsampling (`InterPrediction.cpp:1666-1710`).
    let mut sads = [[i64::MAX / 4; 5]; 5];
    let eval_cost = |ox: isize, oy: isize| -> i64 {
        let p0 = bilinear_pred(ref0, mv0, ox, oy);
        let p1 = bilinear_pred(ref1, mv1, -ox, -oy);
        let mut cost = 0i64;
        for y in (0..dy).step_by(2) {
            for x in 0..dx {
                cost += i64::from((p0[y * dx + x] - p1[y * dx + x]).abs());
            }
        }
        cost
    };

    let mut min_cost = eval_cost(0, 0);
    min_cost -= min_cost >> 2;
    sads[2][2] = min_cost;

    if min_cost < (dx * dy) as i64 {
        return (mv0, mv1);
    }

    let mut best_ix = 0isize;
    let mut best_iy = 0isize;
    for oy in -2..=2isize {
        for ox in -2..=2isize {
            if ox == 0 && oy == 0 {
                continue;
            }
            let c = eval_cost(ox, oy);
            sads[(oy + 2) as usize][(ox + 2) as usize] = c;
            if c < min_cost {
                min_cost = c;
                best_ix = ox;
                best_iy = oy;
            }
        }
    }

    let mut delta_mv = Mv::from_int_pel(best_ix as i32, best_iy as i32);
    if best_ix.abs() < 2 && best_iy.abs() < 2 {
        let ux = (best_ix + 2) as usize;
        let uy = (best_iy + 2) as usize;
        let sub_x = dmvr_subpel_1d(sads[uy][ux], sads[uy][ux - 1], sads[uy][ux + 1]);
        let sub_y = dmvr_subpel_1d(sads[uy][ux], sads[uy - 1][ux], sads[uy + 1][ux]);
        delta_mv.hor += sub_x;
        delta_mv.ver += sub_y;
    }

    (
        mv0.add(delta_mv).clip_to_storage(),
        mv1.sub(delta_mv).clip_to_storage(),
    )
}

/// Affine motion model degree (`4-parameter` or `6-parameter`, H.266 §8.5.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineModel {
    /// 4-parameter affine model using 2 control points `(mv_lt, mv_rt)`.
    FourParam,
    /// 6-parameter affine model using 3 control points `(mv_lt, mv_rt, mv_lb)`.
    SixParam,
}

/// Check whether subblock affine motion vectors exceed the spread limit and
/// must fall back to the CU-center MV (`InterPrediction.cpp:812-850`).
pub fn is_subblock_vector_spread_over_limit(a: i32, b: i32, c: i32, d: i32, pred_type: u8) -> bool {
    let s4 = 4 << 11;
    let filter_tap = 6;
    if pred_type == 3 {
        let max_w = 0.max(4 * a + s4).max((4 * c).max(4 * a + 4 * c + s4));
        let min_w = 0.min(4 * a + s4).min((4 * c).min(4 * a + 4 * c + s4));
        let max_h = 0.max(4 * b).max((4 * d + s4).max(4 * b + 4 * d + s4));
        let min_h = 0.min(4 * b).min((4 * d + s4).min(4 * b + 4 * d + s4));
        let ref_w = ((max_w - min_w) >> 11) + filter_tap + 3;
        let ref_h = ((max_h - min_h) >> 11) + filter_tap + 3;
        ref_w * ref_h > (filter_tap + 9) * (filter_tap + 9)
    } else {
        let ref_w0 = (((4 * a + s4).abs()) >> 11) + filter_tap + 3;
        let ref_h0 = (((4 * b).abs()) >> 11) + filter_tap + 3;
        if ref_w0 * ref_h0 > (filter_tap + 9) * (filter_tap + 5) {
            return true;
        }
        let ref_w1 = (((4 * c).abs()) >> 11) + filter_tap + 3;
        let ref_h1 = (((4 * d + s4).abs()) >> 11) + filter_tap + 3;
        ref_w1 * ref_h1 > (filter_tap + 5) * (filter_tap + 9)
    }
}

/// Derive the 4x4 subblock luma motion vectors and optional PROF `4x4` delta MVs
/// for an affine-coded CU of size `(w, h)`.
///
/// Normative source: ITU-T Rec. H.266 §8.5.5.1 & §8.5.5.9;
/// VTM `InterPrediction.cpp:908-1016`.
#[allow(clippy::too_many_arguments)]
pub fn derive_affine_subblock_mvs(
    model: AffineModel,
    cp_mvs: [Mv; 3],
    w: usize,
    h: usize,
    inter_dir: u8,
) -> (Vec<Mv>, [i32; 16], [i32; 16], bool) {
    const PREC: u32 = 7; // MAX_CU_DEPTH = 7
    let log2_w = w.trailing_zeros();
    let log2_h = h.trailing_zeros();
    let mv_lt = cp_mvs[0];
    let mv_rt = cp_mvs[1];
    let mv_lb = cp_mvs[2];

    let dmv_hor_x = (mv_rt.hor - mv_lt.hor) << (PREC - log2_w);
    let dmv_hor_y = (mv_rt.ver - mv_lt.ver) << (PREC - log2_w);
    let (dmv_ver_x, dmv_ver_y) = match model {
        AffineModel::SixParam => (
            (mv_lb.hor - mv_lt.hor) << (PREC - log2_h),
            (mv_lb.ver - mv_lt.ver) << (PREC - log2_h),
        ),
        AffineModel::FourParam => (-dmv_hor_y, dmv_hor_x),
    };

    let large_grad =
        is_subblock_vector_spread_over_limit(dmv_hor_x, dmv_hor_y, dmv_ver_x, dmv_ver_y, inter_dir);
    let enable_prof =
        !large_grad && (dmv_hor_x != 0 || dmv_hor_y != 0 || dmv_ver_x != 0 || dmv_ver_y != 0);

    let base_hor = mv_lt.hor << PREC;
    let base_ver = mv_lt.ver << PREC;
    let sb_cols = w / 4;
    let sb_rows = h / 4;
    let mut sub_mvs = Vec::with_capacity(sb_cols * sb_rows);

    for sy in 0..sb_rows {
        for sx in 0..sb_cols {
            let weight_hor = if large_grad {
                (w >> 1) as i32
            } else {
                (2 + sx * 4) as i32
            };
            let weight_ver = if large_grad {
                (h >> 1) as i32
            } else {
                (2 + sy * 4) as i32
            };
            let raw = Mv::new(
                base_hor + dmv_hor_x * weight_hor + dmv_ver_x * weight_ver,
                base_ver + dmv_hor_y * weight_hor + dmv_ver_y * weight_ver,
            );
            sub_mvs.push(raw.shr_round(PREC).clip_to_storage());
        }
    }

    let mut dmv_scale_h = [0i32; 16];
    let mut dmv_scale_v = [0i32; 16];
    if enable_prof {
        const DMV_LIMIT: i32 = (1 << 5) - 1;
        for y in 0..4 {
            for x in 0..4 {
                let wx = 2 * (x as i32) - 3;
                let wy = 2 * (y as i32) - 3;
                let raw = Mv::new(
                    wx * dmv_hor_x + wy * dmv_ver_x,
                    wx * dmv_hor_y + wy * dmv_ver_y,
                )
                .shr_round(7);
                dmv_scale_h[y * 4 + x] = raw.hor.clamp(-DMV_LIMIT, DMV_LIMIT);
                dmv_scale_v[y * 4 + x] = raw.ver.clamp(-DMV_LIMIT, DMV_LIMIT);
            }
        }
    }

    (sub_mvs, dmv_scale_h, dmv_scale_v, enable_prof)
}

/// Apply Prediction Refinement with Optical Flow (PROF) to a 4x4 14-bit intermediate
/// subblock (`Buffer.cpp:45-72`, `applyPROFCore`).
pub fn apply_prof_4x4(
    src_6x6: &[i32; 36],
    dmv_x: &[i32; 16],
    dmv_y: &[i32; 16],
    bi: bool,
) -> [i32; 16] {
    let di_limit = 1i32 << 13;
    let shift_num = IF_INTERNAL_PREC - 8; // 6
    let offset = (1i32 << (shift_num - 1)) + IF_INTERNAL_OFFS;
    let mut out = [0i32; 16];

    for y in 0..4 {
        for x in 0..4 {
            let c_idx = (y + 1) * 6 + (x + 1);
            let gx = (src_6x6[c_idx + 1] >> 6) - (src_6x6[c_idx - 1] >> 6);
            let gy = (src_6x6[c_idx + 6] >> 6) - (src_6x6[c_idx - 6] >> 6);
            let idx = y * 4 + x;
            let di = (dmv_x[idx] * gx + dmv_y[idx] * gy).clamp(-di_limit, di_limit - 1);
            let val = src_6x6[c_idx] + di;
            out[idx] = if bi {
                val
            } else {
                ((val + offset) >> shift_num).clamp(0, 255)
            };
        }
    }
    out
}

/// Blend an inter prediction block and a Planar intra prediction block using
/// Combined Inter/Intra Prediction (CIIP).
///
/// Normative source: ITU-T Rec. H.266 §8.5.6.7 (PDF pp. 379–380);
/// VTM `IntraPrediction.cpp:678-707` (`geneWeightedPred`).
pub fn apply_ciip_blend(
    inter_pred: &[i32],
    intra_planar_pred: &[i32],
    left_is_intra: bool,
    above_is_intra: bool,
) -> Vec<i32> {
    let w_intra = 1 + i32::from(left_is_intra) + i32::from(above_is_intra);
    inter_pred
        .iter()
        .zip(intra_planar_pred.iter())
        .map(|(&p_inter, &p_intra)| {
            (p_inter + ((w_intra * (p_intra - p_inter) + 2) >> 2)).clamp(0, 255)
        })
        .collect()
}

/// Geometric Partitioning Mode (GPM) angle and distance pair (`Rom.h:215-220`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeoParam {
    /// Angle index (`0..32`).
    pub angle_idx: u8,
    /// Distance index (`0..4`).
    pub distance_idx: u8,
}

/// Derive the 64-entry GPM `(angle_idx, distance_idx)` table (`Rom.cpp:620-637`).
pub fn derive_gpm_params_table() -> [GeoParam; 64] {
    let mut table = [GeoParam {
        angle_idx: 0,
        distance_idx: 0,
    }; 64];
    let mut mode_idx = 0usize;
    for angle_idx in 0..32u8 {
        for distance_idx in 0..4u8 {
            let mask = GEO_ANGLE_TO_MASK[angle_idx as usize];
            if (distance_idx == 0 && angle_idx >= 16)
                || ((distance_idx == 2 || distance_idx == 0) && (mask == 0 || mask == 5))
                || mask == -1
            {
                continue;
            }
            if mode_idx < 64 {
                table[mode_idx] = GeoParam {
                    angle_idx,
                    distance_idx,
                };
                mode_idx += 1;
            }
        }
    }
    table
}

/// Blend two uni-predicted blocks `pred0` and `pred1` of size `(w, h)` using
/// Geometric Partitioning Mode (`split_dir` in `0..64`).
///
/// Normative source: ITU-T Rec. H.266 §8.5.7 (PDF pp. 383–390, Tables 36–38);
/// VTM `Rom.cpp:647-700`, `InterpolationFilter.cpp:740-810`.
pub fn predict_gpm_block(
    pred0: &[i32],
    pred1: &[i32],
    w: usize,
    h: usize,
    split_dir: u8,
    is_luma: bool,
    scale_x: usize,
    scale_y: usize,
) -> Vec<i32> {
    let table = derive_gpm_params_table();
    let param = table[(split_dir as usize) & 63];
    let angle = param.angle_idx as usize;
    let distance = param.distance_idx as i32;

    let luma_w = (w << scale_x) as i32;
    let luma_h = (h << scale_y) as i32;

    let mut offset_x = -(luma_w >> 1);
    let mut offset_y = -(luma_h >> 1);
    if distance > 0 {
        let angle_mod = angle % 16;
        if angle_mod == 8 || (angle_mod != 0 && luma_h >= luma_w) {
            let d = (distance * luma_h) >> 3;
            offset_y += if angle < 16 { d } else { -d };
        } else {
            let d = (distance * luma_w) >> 3;
            offset_x += if angle < 16 { d } else { -d };
        }
    }

    let dist_x = angle;
    let dist_y = (dist_x + 8) % 32;
    let mut out = vec![0i32; w * h];
    let step_x = 1i32 << scale_x;
    let step_y = 1i32 << scale_y;

    for y in 0..h {
        let ly = (y as i32) * step_y;
        let lookup_y = (((ly + offset_y) << 1) + 1) * GEO_DIS[dist_y];
        for x in 0..w {
            let lx = (x as i32) * step_x;
            let sx_i = ((lx + offset_x) << 1) + 1;
            let weight_idx = sx_i * GEO_DIS[dist_x] + lookup_y;
            let w0 = ((32 + weight_idx + 4) >> 3).clamp(0, 8);
            let w1 = 8 - w0;
            let p0 = pred0[y * w + x];
            let p1 = pred1[y * w + x];
            if is_luma {
                out[y * w + x] = ((w0 * p0 + w1 * p1 + 4) >> 3).clamp(0, 255);
            } else {
                out[y * w + x] = ((w0 * p0 + w1 * p1 + 4) >> 3).clamp(0, 255);
            }
        }
    }
    out
}

/// Produce the luma and 4:2:0 chroma inter prediction blocks `(pred_y, pred_u, pred_v)`
/// for a PU at luma position `(bx, by)` of luma size `(w, h)` using `cand`.
#[allow(clippy::too_many_arguments)]
pub fn predict_inter_pu(
    cand: &InterMergeCand,
    ref_l0: &[&RefFrame],
    ref_l1: &[&RefFrame],
    curr_poc: i32,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    enable_bdof: bool,
    enable_dmvr: bool,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let cw = w / 2;
    let ch = h / 2;
    let cx = (bx / 2) as isize;
    let cy = (by / 2) as isize;

    let f0 = cand.mv_field[0];
    let f1 = cand.mv_field[1];
    let has_l0 = (cand.inter_dir & 1) != 0 && f0.is_valid() && !ref_l0.is_empty();
    let has_l1 = (cand.inter_dir & 2) != 0 && f1.is_valid() && !ref_l1.is_empty();

    if has_l0 && has_l1 {
        let r0 = ref_l0[(f0.ref_idx as usize).min(ref_l0.len() - 1)];
        let r1 = ref_l1[(f1.ref_idx as usize).min(ref_l1.len() - 1)];

        let opp_poc = (curr_poc - r0.poc) * (curr_poc - r1.poc) < 0;
        let sym_poc = (curr_poc - r0.poc) == -(curr_poc - r1.poc);

        let (mv0, mv1) = if enable_dmvr
            && sym_poc
            && cand.bcw_idx == BCW_DEFAULT
            && w >= 8
            && h >= 8
            && w * h >= 128
        {
            refine_dmvr_subblock(&r0.y, &r1.y, bx as isize, by as isize, w, h, f0.mv, f1.mv)
        } else {
            (f0.mv, f1.mv)
        };

        let y0 = interpolate_luma_block(
            &r0.y,
            bx as isize,
            by as isize,
            w,
            h,
            mv0,
            true,
            cand.use_alt_hpel_if,
            false,
        );
        let y1 = interpolate_luma_block(
            &r1.y,
            bx as isize,
            by as isize,
            w,
            h,
            mv1,
            true,
            cand.use_alt_hpel_if,
            false,
        );

        let pred_y = if enable_bdof && opp_poc && cand.bcw_idx == BCW_DEFAULT && w >= 4 && h >= 4 {
            apply_bdof_luma(&y0, &y1, w, h)
        } else {
            weighted_average_bi(&y0, &y1, cand.bcw_idx)
        };

        let u0 = interpolate_chroma_block_420(&r0.u, cx, cy, cw, ch, mv0, true);
        let u1 = interpolate_chroma_block_420(&r1.u, cx, cy, cw, ch, mv1, true);
        let pred_u = weighted_average_bi(&u0, &u1, cand.bcw_idx);

        let v0 = interpolate_chroma_block_420(&r0.v, cx, cy, cw, ch, mv0, true);
        let v1 = interpolate_chroma_block_420(&r1.v, cx, cy, cw, ch, mv1, true);
        let pred_v = weighted_average_bi(&v0, &v1, cand.bcw_idx);

        (pred_y, pred_u, pred_v)
    } else if has_l0 {
        let r0 = ref_l0[(f0.ref_idx as usize).min(ref_l0.len() - 1)];
        let pred_y = interpolate_luma_block(
            &r0.y,
            bx as isize,
            by as isize,
            w,
            h,
            f0.mv,
            false,
            cand.use_alt_hpel_if,
            false,
        );
        let pred_u = interpolate_chroma_block_420(&r0.u, cx, cy, cw, ch, f0.mv, false);
        let pred_v = interpolate_chroma_block_420(&r0.v, cx, cy, cw, ch, f0.mv, false);
        (pred_y, pred_u, pred_v)
    } else if has_l1 {
        let r1 = ref_l1[(f1.ref_idx as usize).min(ref_l1.len() - 1)];
        let pred_y = interpolate_luma_block(
            &r1.y,
            bx as isize,
            by as isize,
            w,
            h,
            f1.mv,
            false,
            cand.use_alt_hpel_if,
            false,
        );
        let pred_u = interpolate_chroma_block_420(&r1.u, cx, cy, cw, ch, f1.mv, false);
        let pred_v = interpolate_chroma_block_420(&r1.v, cx, cy, cw, ch, f1.mv, false);
        (pred_y, pred_u, pred_v)
    } else {
        (vec![128; w * h], vec![128; cw * ch], vec![128; cw * ch])
    }
}

/// Encoder motion estimation result for a single block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionSearchResult {
    /// Selected motion candidate.
    pub cand: InterMergeCand,
    /// Whether the block uses the Merge path (`true`) or explicit AMVP (`false`).
    pub is_merge: bool,
    /// Index into the Merge list (`0..max_num_merge_cand`) when `is_merge` is `true`.
    pub merge_idx: usize,
    /// Best luma SAD achieved.
    pub sad: u64,
}

/// Estimate the best inter motion candidate (across the Merge list and integer +
/// quarter-pel diamond search around the AMVP predictor) for a luma block at `(bx, by)`.
#[allow(clippy::too_many_arguments)]
pub fn estimate_inter_block(
    src_y: &Plane,
    ref_l0: &[&RefFrame],
    ref_l1: &[&RefFrame],
    merge_cands: &[InterMergeCand],
    amvp_l0: [Mv; 2],
    curr_poc: i32,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    search_range_pel: i32,
) -> MotionSearchResult {
    let calc_sad = |pred: &[i32]| -> u64 {
        let mut total = 0u64;
        for y in 0..h {
            for x in 0..w {
                let s = i32::from(src_y.get(bx + x, by + y));
                total += u64::from((s - pred[y * w + x]).unsigned_abs());
            }
        }
        total
    };

    let mut best_merge_idx = 0usize;
    let mut best_merge_sad = u64::MAX;
    let mut best_merge_cand = merge_cands.first().copied().unwrap_or_default();

    for (idx, cand) in merge_cands.iter().enumerate() {
        let (pred_y, _, _) =
            predict_inter_pu(cand, ref_l0, ref_l1, curr_poc, bx, by, w, h, false, false);
        let sad = calc_sad(&pred_y);
        if sad < best_merge_sad {
            best_merge_sad = sad;
            best_merge_idx = idx;
            best_merge_cand = *cand;
        }
    }

    if ref_l0.is_empty() {
        return MotionSearchResult {
            cand: best_merge_cand,
            is_merge: true,
            merge_idx: best_merge_idx,
            sad: best_merge_sad,
        };
    }

    // Integer pel diamond search on L0 ref 0 around `amvp_l0[0]`.
    let r0 = &ref_l0[0].y;
    let start_pel_x = amvp_l0[0].hor >> 4;
    let start_pel_y = amvp_l0[0].ver >> 4;
    let mut best_mv = Mv::from_int_pel(start_pel_x, start_pel_y);
    let init_pred = interpolate_luma_block(
        r0,
        bx as isize,
        by as isize,
        w,
        h,
        best_mv,
        false,
        false,
        false,
    );
    let mut best_amvp_sad = calc_sad(&init_pred);

    const DIAMOND: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for _iter in 0..search_range_pel.max(1) {
        let mut improved = false;
        let center = best_mv;
        for &(dx, dy) in &DIAMOND {
            let trial = Mv::from_int_pel((center.hor >> 4) + dx, (center.ver >> 4) + dy);
            let pred = interpolate_luma_block(
                r0,
                bx as isize,
                by as isize,
                w,
                h,
                trial,
                false,
                false,
                false,
            );
            let sad = calc_sad(&pred);
            if sad < best_amvp_sad {
                best_amvp_sad = sad;
                best_mv = trial;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    // Quarter-pel refinement (`step = 4` in 1/16-pel units).
    let int_center = best_mv;
    for &(qx, qy) in &[(4, 0), (-4, 0), (0, 4), (0, -4)] {
        let trial = Mv::new(int_center.hor + qx, int_center.ver + qy);
        let pred = interpolate_luma_block(
            r0,
            bx as isize,
            by as isize,
            w,
            h,
            trial,
            false,
            false,
            false,
        );
        let sad = calc_sad(&pred);
        if sad < best_amvp_sad {
            best_amvp_sad = sad;
            best_mv = trial;
        }
    }

    if best_merge_sad <= best_amvp_sad {
        MotionSearchResult {
            cand: best_merge_cand,
            is_merge: true,
            merge_idx: best_merge_idx,
            sad: best_merge_sad,
        }
    } else {
        MotionSearchResult {
            cand: InterMergeCand {
                inter_dir: 1,
                mv_field: [MvField::new(best_mv, 0), MvField::default()],
                bcw_idx: BCW_DEFAULT,
                use_alt_hpel_if: false,
            },
            is_merge: false,
            merge_idx: 0,
            sad: best_amvp_sad,
        }
    }
}

use super::predict::{ChromaFormatIdc, Plane16};

/// Normative 8-tap luma fractional motion compensation on a 16-bit [`Plane16`]
/// (`bit_depth` in `8..=12`, ITU-T Rec. H.266 §8.5.6.3.2).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_luma_block_16(
    ref_plane: &Plane16,
    bx: isize,
    by: isize,
    w: usize,
    h: usize,
    mv: Mv,
    for_bi_pred: bool,
    is_affine: bool,
    use_alt_hpel_if: bool,
) -> Vec<i32> {
    let dx = (mv.hor >> 4) as isize;
    let dy = (mv.ver >> 4) as isize;
    let fx = (mv.hor & 15) as usize;
    let fy = (mv.ver & 15) as usize;
    let bd = i32::from(ref_plane.bit_depth.clamp(8, 12));
    let max_val = (1i32 << bd) - 1;

    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, ref_plane.width as isize - 1) as usize;
        let cy = y.clamp(0, ref_plane.height as isize - 1) as usize;
        i32::from(ref_plane.get(cx, cy))
    };

    let get_filter = |frac: usize| -> &[i32; 8] {
        if is_affine {
            &AFFINE_LUMA_FILTER_8TAP[frac]
        } else if use_alt_hpel_if && frac == 8 {
            &LUMA_ALT_HPEL_FILTER_8TAP
        } else {
            &LUMA_FILTER_8TAP[frac]
        }
    };

    let head_room = (14 - bd).max(2);
    let shift1 = 6 - head_room;
    let shift2 = if for_bi_pred { 6 } else { 12 - bd };
    let add2 = if for_bi_pred { 0 } else { 1i32 << (shift2 - 1) };

    let mut out = vec![0i32; w * h];
    if fx == 0 && fy == 0 {
        for y in 0..h {
            for x in 0..w {
                let s = get_p(bx + x as isize + dx, by + y as isize + dy);
                out[y * w + x] = if for_bi_pred {
                    (s << head_room) - (1 << (bd + head_room - 1))
                } else {
                    s
                };
            }
        }
        return out;
    }

    if fy == 0 {
        let fh = get_filter(fx);
        for y in 0..h {
            let py = by + y as isize + dy;
            for x in 0..w {
                let px = bx + x as isize + dx;
                let mut sum = 0i32;
                for k in 0..8 {
                    sum += fh[k] * get_p(px + k as isize - 3, py);
                }
                if for_bi_pred {
                    out[y * w + x] = (sum >> shift1) - (1 << (bd + head_room - 1));
                } else {
                    out[y * w + x] = ((sum + 32) >> 6).clamp(0, max_val);
                }
            }
        }
        return out;
    }

    if fx == 0 {
        let fv = get_filter(fy);
        for y in 0..h {
            let py = by + y as isize + dy;
            for x in 0..w {
                let px = bx + x as isize + dx;
                let mut sum = 0i32;
                for k in 0..8 {
                    sum += fv[k] * get_p(px, py + k as isize - 3);
                }
                if for_bi_pred {
                    out[y * w + x] = (sum >> shift1) - (1 << (bd + head_room - 1));
                } else {
                    out[y * w + x] = ((sum + 32) >> 6).clamp(0, max_val);
                }
            }
        }
        return out;
    }

    let fh = get_filter(fx);
    let fv = get_filter(fy);
    let tmp_h = h + 7;
    let mut tmp = vec![0i32; w * tmp_h];
    for ty in 0..tmp_h {
        let py = by + ty as isize - 3 + dy;
        for x in 0..w {
            let px = bx + x as isize + dx;
            let mut sum = 0i32;
            for k in 0..8 {
                sum += fh[k] * get_p(px + k as isize - 3, py);
            }
            tmp[ty * w + x] = sum >> shift1;
        }
    }

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0i32;
            for k in 0..8 {
                sum += fv[k] * tmp[(y + k) * w + x];
            }
            if for_bi_pred {
                out[y * w + x] = (sum >> shift2) - (1 << (bd + head_room - 1));
            } else {
                out[y * w + x] = ((sum + add2) >> shift2).clamp(0, max_val);
            }
        }
    }
    out
}

/// Normative 4-tap chroma fractional motion compensation supporting `4:2:0`, `4:2:2`,
/// and `4:4:4` chroma formats (`H.266 §8.5.6.3.1, §8.5.6.3.3`, VTM `InterPrediction.cpp:415-520`).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_chroma_block_format(
    ref_plane: &Plane,
    bx_chroma: isize,
    by_chroma: isize,
    cw: usize,
    ch: usize,
    luma_mv: Mv,
    for_bi_pred: bool,
    chroma_format: ChromaFormatIdc,
) -> Vec<i32> {
    if chroma_format == ChromaFormatIdc::Chroma420 {
        return interpolate_chroma_block_420(
            ref_plane,
            bx_chroma,
            by_chroma,
            cw,
            ch,
            luma_mv,
            for_bi_pred,
        );
    }
    // Scale luma 1/16-pel MV to 1/32-pel chroma fractional units:
    // For SubWidthC == 2, 1/16 luma pel = 1/32 chroma pel (scale = 1).
    // For SubWidthC == 1 (4:4:4), 1/16 luma pel = 2/32 chroma pel (scale = 2).
    let mv_cx_32 = luma_mv.hor
        * if chroma_format.sub_width_c() == 1 {
            2
        } else {
            1
        };
    let mv_cy_32 = luma_mv.ver
        * if chroma_format.sub_height_c() == 1 {
            2
        } else {
            1
        };
    let scaled_mv = Mv::new(mv_cx_32, mv_cy_32);
    interpolate_chroma_block_420(
        ref_plane,
        bx_chroma,
        by_chroma,
        cw,
        ch,
        scaled_mv,
        for_bi_pred,
    )
}

/// Check whether an Intra Block Copy (IBC) integer-pel block vector `(bv_x_pel, bv_y_pel)`
/// is normatively valid for CU `(bx, by, w, h)` within a picture of `(pic_w, pic_h)` and
/// `ctu_size` (ITU-T Rec. H.266 §8.6.2.5, VTM `InterPrediction.cpp:1405-1465`).
#[allow(clippy::too_many_arguments)]
pub fn is_ibc_bv_valid(
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    bv_x_pel: isize,
    bv_y_pel: isize,
    ctu_size: usize,
    pic_w: usize,
    pic_h: usize,
) -> bool {
    let ref_x = bx as isize + bv_x_pel;
    let ref_y = by as isize + bv_y_pel;
    if ref_x < 0 || ref_y < 0 {
        return false;
    }
    let rx = ref_x as usize;
    let ry = ref_y as usize;
    if rx + w > pic_w || ry + h > pic_h {
        return false;
    }
    // Must stay within the same CTU row (`ry / ctu_size == by / ctu_size` and `(ry + h - 1) / ctu_size == by / ctu_size`)
    let ctu = ctu_size.max(32);
    let cur_ctu_y = by / ctu;
    if ry / ctu != cur_ctu_y || (ry + h - 1) / ctu != cur_ctu_y {
        return false;
    }
    let cur_ctu_x = bx / ctu;
    let ref_ctu_x_end = (rx + w - 1) / ctu;
    if ref_ctu_x_end > cur_ctu_x || rx / ctu + 1 < cur_ctu_x {
        return false;
    }
    // Reference block must be strictly prior to current CU (no overlap and already reconstructed)
    if ref_ctu_x_end == cur_ctu_x {
        let overlap_x = rx < bx + w && bx < rx + w;
        let overlap_y = ry < by + h && by < ry + h;
        if overlap_x && overlap_y {
            return false;
        }
        // Within current CTU, reference bottom-right must precede current CU in raster/Z order
        if ry + h > by && rx + w > bx {
            return false;
        }
    }
    true
}

/// Derive the Intra Block Copy (IBC) merge/AMVP candidate list of length `max_num_ibc_cand`
/// from spatial neighbours (`A1` left, `B1` above), the HMVP-IBC history table, and zero-BV
/// padding (ITU-T Rec. H.266 §8.6.2.2–§8.6.2.4, VTM `UnitTools.cpp:1780-1890`).
pub fn derive_ibc_merge_candidates(
    left_bv: Option<Mv>,
    above_bv: Option<Mv>,
    hmvp_ibc: &[Mv],
    max_num_ibc_cand: usize,
) -> Vec<Mv> {
    let cap = max_num_ibc_cand.clamp(1, 6);
    let mut list: Vec<Mv> = Vec::with_capacity(cap);
    if let Some(a1) = left_bv {
        list.push(a1);
    }
    if let Some(b1) = above_bv {
        if list.first() != Some(&b1) && list.len() < cap {
            list.push(b1);
        }
    }
    for &hbv in hmvp_ibc.iter().rev() {
        if list.len() >= cap {
            break;
        }
        if !list.contains(&hbv) {
            list.push(hbv);
        }
    }
    while list.len() < cap {
        list.push(Mv::from_int_pel(
            -(w_default_ibc_offset(list.len()) as i32),
            0,
        ));
    }
    list
}

#[inline]
fn w_default_ibc_offset(idx: usize) -> usize {
    4 * (idx + 1)
}

/// Reconstruct a luma or chroma block using Intra Block Copy (IBC) integer-pel block
/// vector `(bv_x_pel, bv_y_pel)` from the already-reconstructed region of `recon_plane`
/// (ITU-T Rec. H.266 §8.6.3, VTM `InterPrediction.cpp:1405-1465`).
pub fn predict_ibc_block(
    recon_plane: &Plane,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    bv_x_pel: isize,
    bv_y_pel: isize,
) -> Vec<u8> {
    let mut out = vec![128u8; w * h];
    for y in 0..h {
        let ry = (by as isize + y as isize + bv_y_pel).clamp(0, recon_plane.height as isize - 1)
            as usize;
        for x in 0..w {
            let rx = (bx as isize + x as isize + bv_x_pel).clamp(0, recon_plane.width as isize - 1)
                as usize;
            out[y * w + x] = recon_plane.get(rx, ry);
        }
    }
    out
}

/// Search for the best valid integer-pel Intra Block Copy (IBC) block vector `(bv_x_pel, bv_y_pel)`
/// in the current CTU and left CTU for screen-content encoding.
#[allow(clippy::too_many_arguments)]
pub fn search_ibc_block_vector(
    recon_plane: &Plane,
    src_plane: &Plane,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    ctu_size: usize,
) -> Option<(isize, isize, u64)> {
    let mut best: Option<(isize, isize, u64)> = None;
    let min_dx = -(bx.min(ctu_size) as isize);
    let min_dy = -((by % ctu_size.max(32)) as isize);
    for dy in (min_dy..=0).step_by(2) {
        for dx in (min_dx..=0).step_by(2) {
            if !is_ibc_bv_valid(
                bx,
                by,
                w,
                h,
                dx,
                dy,
                ctu_size,
                recon_plane.width,
                recon_plane.height,
            ) {
                continue;
            }
            let mut sad = 0u64;
            for y in 0..h {
                let ry = (by as isize + y as isize + dy) as usize;
                for x in 0..w {
                    let rx = (bx as isize + x as isize + dx) as usize;
                    sad += u64::from(
                        recon_plane
                            .get(rx, ry)
                            .abs_diff(src_plane.get(bx + x, by + y)),
                    );
                }
            }
            if best.map_or(true, |(_, _, b_sad)| sad < b_sad) {
                best = Some((dx, dy, sad));
            }
        }
    }
    best
}

/// Unit scaling ratio (`1 << 14 = 16384`) for Reference Picture Resampling (RPR,
/// ITU-T Rec. H.266 §8.5.6.3.1 Eq. 944–945).
pub const RPR_SCALE_1X: u32 = 1 << 14;

/// Normative 14-bit fixed-point horizontal and vertical Reference Picture Resampling (RPR)
/// scaling ratios `(RefPicScale[i][j][0], RefPicScale[i][j][1])`
/// (ITU-T Rec. H.266 §8.5.6.3.1, VTM `Slice.cpp:1395-1425`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RprScalingRatio {
    /// Horizontal scaling ratio in 2.14 fixed-point (`16384` = 1.0x).
    pub hor_scale: u32,
    /// Vertical scaling ratio in 2.14 fixed-point (`16384` = 1.0x).
    pub ver_scale: u32,
}

impl Default for RprScalingRatio {
    fn default() -> Self {
        Self {
            hor_scale: RPR_SCALE_1X,
            ver_scale: RPR_SCALE_1X,
        }
    }
}

impl RprScalingRatio {
    /// True when either horizontal or vertical scaling differs from `1.0x` (`16384`).
    #[inline]
    pub fn is_scaled(self) -> bool {
        self.hor_scale != RPR_SCALE_1X || self.ver_scale != RPR_SCALE_1X
    }
}

/// Derive the normative 14-bit fixed-point RPR scaling ratio between a reference picture's
/// conformance window size `(ref_vis_w, ref_vis_h)` and the current picture's conformance
/// window size `(cur_vis_w, cur_vis_h)` (ITU-T Rec. H.266 §7.4.8 Eq. 142, VTM `Slice.cpp:1395-1425`).
#[inline]
pub fn derive_rpr_scaling_ratio(
    ref_vis_w: u32,
    ref_vis_h: u32,
    cur_vis_w: u32,
    cur_vis_h: u32,
) -> RprScalingRatio {
    let cw = u64::from(cur_vis_w.max(1));
    let ch = u64::from(cur_vis_h.max(1));
    let hor_scale = (((u64::from(ref_vis_w.max(1)) << 14) + (cw >> 1)) / cw) as u32;
    let ver_scale = (((u64::from(ref_vis_h.max(1)) << 14) + (ch >> 1)) / ch) as u32;
    RprScalingRatio {
        hor_scale,
        ver_scale,
    }
}

/// Normative luma motion compensation with Reference Picture Resampling (RPR) scaling
/// ratio `scale` (ITU-T Rec. H.266 §8.5.6.3.1 Eq. 944–952, VTM `InterPrediction.cpp:415-610`).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_luma_rpr_block(
    ref_plane: &Plane,
    bx: isize,
    by: isize,
    w: usize,
    h: usize,
    mv: Mv,
    scale: RprScalingRatio,
) -> Vec<u8> {
    if !scale.is_scaled() {
        return interpolate_luma_block(ref_plane, bx, by, w, h, mv, false, false, false)
            .into_iter()
            .map(|v| v.clamp(0, 255) as u8)
            .collect();
    }
    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, ref_plane.width as isize - 1) as usize;
        let cy = y.clamp(0, ref_plane.height as isize - 1) as usize;
        i32::from(ref_plane.get(cx, cy))
    };
    let mut out = vec![0u8; w * h];
    // H.266 Eq. 944-949: reference position in 1/16-pel units
    let ref_x_base = ((bx as i64) << 4) + i64::from(mv.hor);
    let ref_y_base = ((by as i64) << 4) + i64::from(mv.ver);

    for y in 0..h {
        let cur_y_16 = ref_y_base + ((y as i64) << 4);
        let scaled_y_16 = ((cur_y_16 * i64::from(scale.ver_scale) + (1 << 13)) >> 14) as isize;
        let iy = scaled_y_16 >> 4;
        let fy = (scaled_y_16 & 15) as usize;
        let fv = &LUMA_FILTER_8TAP[fy];

        for x in 0..w {
            let cur_x_16 = ref_x_base + ((x as i64) << 4);
            let scaled_x_16 = ((cur_x_16 * i64::from(scale.hor_scale) + (1 << 13)) >> 14) as isize;
            let ix = scaled_x_16 >> 4;
            let fx = (scaled_x_16 & 15) as usize;
            let fh = &LUMA_FILTER_8TAP[fx];

            let mut sum_v = 0i32;
            for ky in 0..8 {
                let py = iy + ky as isize - 3;
                let mut sum_h = 0i32;
                for kx in 0..8 {
                    sum_h += fh[kx] * get_p(ix + kx as isize - 3, py);
                }
                sum_v += fv[ky] * ((sum_h + 32) >> 6);
            }
            out[y * w + x] = ((sum_v + 32) >> 6).clamp(0, 255) as u8;
        }
    }
    out
}

/// Normative chroma motion compensation with Reference Picture Resampling (RPR) scaling
/// ratio `scale` for any [`ChromaFormatIdc`] (ITU-T Rec. H.266 §8.5.6.3.1 Eq. 950–952).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_chroma_rpr_block(
    ref_plane: &Plane,
    bx_c: isize,
    by_c: isize,
    cw: usize,
    ch: usize,
    luma_mv: Mv,
    scale: RprScalingRatio,
    chroma_format: ChromaFormatIdc,
) -> Vec<u8> {
    if !scale.is_scaled() {
        return interpolate_chroma_block_format(
            ref_plane,
            bx_c,
            by_c,
            cw,
            ch,
            luma_mv,
            false,
            chroma_format,
        )
        .into_iter()
        .map(|v| v.clamp(0, 255) as u8)
        .collect();
    }
    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, ref_plane.width as isize - 1) as usize;
        let cy = y.clamp(0, ref_plane.height as isize - 1) as usize;
        i32::from(ref_plane.get(cx, cy))
    };
    let shift_x = if chroma_format.sub_width_c() == 1 {
        1
    } else {
        0
    };
    let shift_y = if chroma_format.sub_height_c() == 1 {
        1
    } else {
        0
    };
    let ref_x_base_32 = ((bx_c as i64) << 5) + (i64::from(luma_mv.hor) << shift_x);
    let ref_y_base_32 = ((by_c as i64) << 5) + (i64::from(luma_mv.ver) << shift_y);

    let mut out = vec![0u8; cw * ch];
    for y in 0..ch {
        let cur_y_32 = ref_y_base_32 + ((y as i64) << 5);
        let scaled_y_32 = ((cur_y_32 * i64::from(scale.ver_scale) + (1 << 13)) >> 14) as isize;
        let iy = scaled_y_32 >> 5;
        let fy = (scaled_y_32 & 31) as usize;
        let fv = &CHROMA_FILTER_4TAP[fy];

        for x in 0..cw {
            let cur_x_32 = ref_x_base_32 + ((x as i64) << 5);
            let scaled_x_32 = ((cur_x_32 * i64::from(scale.hor_scale) + (1 << 13)) >> 14) as isize;
            let ix = scaled_x_32 >> 5;
            let fx = (scaled_x_32 & 31) as usize;
            let fh = &CHROMA_FILTER_4TAP[fx];

            let mut sum_v = 0i32;
            for ky in 0..4 {
                let py = iy + ky as isize - 1;
                let mut sum_h = 0i32;
                for kx in 0..4 {
                    sum_h += fh[kx] * get_p(ix + kx as isize - 1, py);
                }
                sum_v += fv[ky] * ((sum_h + 32) >> 6);
            }
            out[y * cw + x] = ((sum_v + 32) >> 6).clamp(0, 255) as u8;
        }
    }
    out
}

/// Normative 4-tap chroma fractional motion compensation on a 16-bit [`Plane16`] for
/// `4:0:0`, `4:2:0`, `4:2:2`, and `4:4:4` chroma formats (`bit_depth` in `8..=12`,
/// ITU-T Rec. H.266 §8.5.6.3.1, §8.5.6.3.3, VTM `InterPrediction.cpp:415-520`).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_chroma_block_format_16(
    ref_plane: &Plane16,
    bx_chroma: isize,
    by_chroma: isize,
    cw: usize,
    ch: usize,
    luma_mv: Mv,
    for_bi_pred: bool,
    chroma_format: ChromaFormatIdc,
) -> Vec<i32> {
    if cw == 0 || ch == 0 || !chroma_format.is_chroma_enabled() {
        return Vec::new();
    }
    let mv_cx_32 = luma_mv.hor
        * if chroma_format.sub_width_c() == 1 {
            2
        } else {
            1
        };
    let mv_cy_32 = luma_mv.ver
        * if chroma_format.sub_height_c() == 1 {
            2
        } else {
            1
        };
    let dx = (mv_cx_32 >> 5) as isize;
    let dy = (mv_cy_32 >> 5) as isize;
    let fx = (mv_cx_32 & 31) as usize;
    let fy = (mv_cy_32 & 31) as usize;
    let bd = i32::from(ref_plane.bit_depth.clamp(8, 12));
    let max_val = (1i32 << bd) - 1;

    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, ref_plane.width as isize - 1) as usize;
        let cy = y.clamp(0, ref_plane.height as isize - 1) as usize;
        i32::from(ref_plane.get(cx, cy))
    };

    let head_room = (14 - bd).max(2);
    let shift1 = 6 - head_room;
    let shift2 = if for_bi_pred { 6 } else { 12 - bd };
    let add2 = if for_bi_pred { 0 } else { 1i32 << (shift2 - 1) };

    let mut out = vec![0i32; cw * ch];
    if fx == 0 && fy == 0 {
        for y in 0..ch {
            for x in 0..cw {
                let s = get_p(bx_chroma + x as isize + dx, by_chroma + y as isize + dy);
                out[y * cw + x] = if for_bi_pred {
                    (s << head_room) - (1 << (bd + head_room - 1))
                } else {
                    s
                };
            }
        }
        return out;
    }

    if fy == 0 {
        let fh = &CHROMA_FILTER_4TAP[fx];
        for y in 0..ch {
            let py = by_chroma + y as isize + dy;
            for x in 0..cw {
                let px = bx_chroma + x as isize + dx;
                let mut sum = 0i32;
                for k in 0..4 {
                    sum += fh[k] * get_p(px + k as isize - 1, py);
                }
                if for_bi_pred {
                    out[y * cw + x] = (sum >> shift1) - (1 << (bd + head_room - 1));
                } else {
                    out[y * cw + x] = ((sum + 32) >> 6).clamp(0, max_val);
                }
            }
        }
        return out;
    }

    if fx == 0 {
        let fv = &CHROMA_FILTER_4TAP[fy];
        for y in 0..ch {
            let py = by_chroma + y as isize + dy;
            for x in 0..cw {
                let px = bx_chroma + x as isize + dx;
                let mut sum = 0i32;
                for k in 0..4 {
                    sum += fv[k] * get_p(px, py + k as isize - 1);
                }
                if for_bi_pred {
                    out[y * cw + x] = (sum >> shift1) - (1 << (bd + head_room - 1));
                } else {
                    out[y * cw + x] = ((sum + 32) >> 6).clamp(0, max_val);
                }
            }
        }
        return out;
    }

    let fh = &CHROMA_FILTER_4TAP[fx];
    let fv = &CHROMA_FILTER_4TAP[fy];
    let tmp_h = ch + 3;
    let mut tmp = vec![0i32; cw * tmp_h];
    for ty in 0..tmp_h {
        let py = by_chroma + ty as isize - 1 + dy;
        for x in 0..cw {
            let px = bx_chroma + x as isize + dx;
            let mut sum = 0i32;
            for k in 0..4 {
                sum += fh[k] * get_p(px + k as isize - 1, py);
            }
            tmp[ty * cw + x] = sum >> shift1;
        }
    }

    for y in 0..ch {
        for x in 0..cw {
            let mut sum = 0i32;
            for k in 0..4 {
                sum += fv[k] * tmp[(y + k) * cw + x];
            }
            if for_bi_pred {
                out[y * cw + x] = (sum >> shift2) - (1 << (bd + head_room - 1));
            } else {
                out[y * cw + x] = ((sum + add2) >> shift2).clamp(0, max_val);
            }
        }
    }
    out
}

/// Combine two 14-bit bi-prediction buffers (`pred0`, `pred1`) at `bit_depth` (`8..=12`)
/// using CU-level BCW weights (`bcw_idx` in `0..5`, ITU-T Rec. H.266 §8.5.6.6,
/// VTM `Buffer.cpp:325-352`).
pub fn weighted_average_bi_16(
    pred0: &[i32],
    pred1: &[i32],
    bcw_idx: u8,
    bit_depth: u8,
) -> Vec<u16> {
    let bd = i32::from(bit_depth.clamp(8, 12));
    let max_val = (1i32 << bd) - 1;
    let w1 = BCW_WEIGHTS[(bcw_idx as usize).min(4)];
    let w0 = 8 - w1;
    let head_room = (14 - bd).max(2);
    let shift_num = head_room + 3;
    let internal_offs = 1i32 << (bd + head_room - 1);
    let offset = (1i32 << (shift_num - 1)) + (internal_offs << 3);

    pred0
        .iter()
        .zip(pred1.iter())
        .map(|(&p0, &p1)| (((p0 * w0 + p1 * w1 + offset) >> shift_num).clamp(0, max_val)) as u16)
        .collect()
}

/// Normative 16-bit luma motion compensation with Reference Picture Resampling (RPR)
/// scaling ratio `scale` (`H.266 §8.5.6.3.1 Eq. 944–952`, VTM `InterPrediction.cpp:2172-2310`).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_luma_rpr_block_16(
    ref_plane: &Plane16,
    bx: isize,
    by: isize,
    w: usize,
    h: usize,
    mv: Mv,
    scale: RprScalingRatio,
) -> Vec<u16> {
    let bd = i32::from(ref_plane.bit_depth.clamp(8, 12));
    let max_val = (1i32 << bd) - 1;
    if !scale.is_scaled() {
        return interpolate_luma_block_16(ref_plane, bx, by, w, h, mv, false, false, false)
            .into_iter()
            .map(|v| v.clamp(0, max_val) as u16)
            .collect();
    }
    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, ref_plane.width as isize - 1) as usize;
        let cy = y.clamp(0, ref_plane.height as isize - 1) as usize;
        i32::from(ref_plane.get(cx, cy))
    };
    let mut out = vec![0u16; w * h];
    let ref_x_base = ((bx as i64) << 4) + i64::from(mv.hor);
    let ref_y_base = ((by as i64) << 4) + i64::from(mv.ver);

    for y in 0..h {
        let cur_y_16 = ref_y_base + ((y as i64) << 4);
        let scaled_y_16 = ((cur_y_16 * i64::from(scale.ver_scale) + (1 << 13)) >> 14) as isize;
        let iy = scaled_y_16 >> 4;
        let fy = (scaled_y_16 & 15) as usize;
        let fv = &LUMA_FILTER_8TAP[fy];

        for x in 0..w {
            let cur_x_16 = ref_x_base + ((x as i64) << 4);
            let scaled_x_16 = ((cur_x_16 * i64::from(scale.hor_scale) + (1 << 13)) >> 14) as isize;
            let ix = scaled_x_16 >> 4;
            let fx = (scaled_x_16 & 15) as usize;
            let fh = &LUMA_FILTER_8TAP[fx];

            let mut sum_v = 0i32;
            for ky in 0..8 {
                let py = iy + ky as isize - 3;
                let mut sum_h = 0i32;
                for kx in 0..8 {
                    sum_h += fh[kx] * get_p(ix + kx as isize - 3, py);
                }
                sum_v += fv[ky] * ((sum_h + 32) >> 6);
            }
            out[y * w + x] = ((sum_v + 32) >> 6).clamp(0, max_val) as u16;
        }
    }
    out
}

/// Normative 16-bit chroma motion compensation with Reference Picture Resampling (RPR)
/// scaling ratio `scale` for any [`ChromaFormatIdc`] (`H.266 §8.5.6.3.1 Eq. 950–952`).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_chroma_rpr_block_16(
    ref_plane: &Plane16,
    bx_c: isize,
    by_c: isize,
    cw: usize,
    ch: usize,
    luma_mv: Mv,
    scale: RprScalingRatio,
    chroma_format: ChromaFormatIdc,
) -> Vec<u16> {
    if cw == 0 || ch == 0 || !chroma_format.is_chroma_enabled() {
        return Vec::new();
    }
    let bd = i32::from(ref_plane.bit_depth.clamp(8, 12));
    let max_val = (1i32 << bd) - 1;
    if !scale.is_scaled() {
        return interpolate_chroma_block_format_16(
            ref_plane,
            bx_c,
            by_c,
            cw,
            ch,
            luma_mv,
            false,
            chroma_format,
        )
        .into_iter()
        .map(|v| v.clamp(0, max_val) as u16)
        .collect();
    }
    let get_p = |x: isize, y: isize| -> i32 {
        let cx = x.clamp(0, ref_plane.width as isize - 1) as usize;
        let cy = y.clamp(0, ref_plane.height as isize - 1) as usize;
        i32::from(ref_plane.get(cx, cy))
    };
    let shift_x = if chroma_format.sub_width_c() == 1 {
        1
    } else {
        0
    };
    let shift_y = if chroma_format.sub_height_c() == 1 {
        1
    } else {
        0
    };
    let ref_x_base_32 = ((bx_c as i64) << 5) + (i64::from(luma_mv.hor) << shift_x);
    let ref_y_base_32 = ((by_c as i64) << 5) + (i64::from(luma_mv.ver) << shift_y);

    let mut out = vec![0u16; cw * ch];
    for y in 0..ch {
        let cur_y_32 = ref_y_base_32 + ((y as i64) << 5);
        let scaled_y_32 = ((cur_y_32 * i64::from(scale.ver_scale) + (1 << 13)) >> 14) as isize;
        let iy = scaled_y_32 >> 5;
        let fy = (scaled_y_32 & 31) as usize;
        let fv = &CHROMA_FILTER_4TAP[fy];

        for x in 0..cw {
            let cur_x_32 = ref_x_base_32 + ((x as i64) << 5);
            let scaled_x_32 = ((cur_x_32 * i64::from(scale.hor_scale) + (1 << 13)) >> 14) as isize;
            let ix = scaled_x_32 >> 5;
            let fx = (scaled_x_32 & 31) as usize;
            let fh = &CHROMA_FILTER_4TAP[fx];

            let mut sum_v = 0i32;
            for ky in 0..4 {
                let py = iy + ky as isize - 1;
                let mut sum_h = 0i32;
                for kx in 0..4 {
                    sum_h += fh[kx] * get_p(ix + kx as isize - 1, py);
                }
                sum_v += fv[ky] * ((sum_h + 32) >> 6);
            }
            out[y * cw + x] = ((sum_v + 32) >> 6).clamp(0, max_val) as u16;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luma_8tap_filter_rows_sum_to_64() {
        for (frac, row) in LUMA_FILTER_8TAP.iter().enumerate() {
            let sum: i32 = row.iter().sum();
            assert_eq!(sum, 64, "LUMA_FILTER_8TAP[{frac}] sum != 64");
        }
        let alt_sum: i32 = LUMA_ALT_HPEL_FILTER_8TAP.iter().sum();
        assert_eq!(alt_sum, 64);
        for (frac, row) in AFFINE_LUMA_FILTER_8TAP.iter().enumerate() {
            let sum: i32 = row.iter().sum();
            assert_eq!(sum, 64, "AFFINE_LUMA_FILTER_8TAP[{frac}] sum != 64");
        }
    }

    #[test]
    fn fractional_mc_reproduces_constant_plane_at_all_16_subpel_positions() {
        let plane = Plane::from_slice(32, 32, &[115u8; 32 * 32]).expect("plane");
        for fy in 0..16 {
            for fx in 0..16 {
                let mv = Mv::new(fx, fy);
                let pred = interpolate_luma_block(&plane, 8, 8, 8, 8, mv, false, false, false);
                assert!(
                    pred.iter().all(|&p| p == 115),
                    "failed at subpel ({fx}, {fy})"
                );
            }
        }
    }

    #[test]
    fn bi_prediction_bcw_weights_reproduce_flat_planes() {
        let p0 = Plane::from_slice(16, 16, &[80u8; 16 * 16]).expect("p0");
        let p1 = Plane::from_slice(16, 16, &[160u8; 16 * 16]).expect("p1");
        let y0 = interpolate_luma_block(&p0, 4, 4, 8, 8, Mv::default(), true, false, false);
        let y1 = interpolate_luma_block(&p1, 4, 4, 8, 8, Mv::default(), true, false, false);
        // Equal weight (bcw_idx = 2 -> w0 = 4, w1 = 4): (80 + 160) / 2 = 120
        let avg = weighted_average_bi(&y0, &y1, BCW_DEFAULT);
        assert!(avg.iter().all(|&v| v == 120));
    }

    #[test]
    fn bdof_preserves_identical_bi_predictions() {
        let p = Plane::from_slice(16, 16, &[99u8; 16 * 16]).expect("p");
        let y0 = interpolate_luma_block(&p, 4, 4, 8, 8, Mv::default(), true, false, false);
        let bdof = apply_bdof_luma(&y0, &y0, 8, 8);
        assert!(bdof.iter().all(|&v| v == 99));
    }

    #[test]
    fn gpm_derives_all_64_valid_partition_modes() {
        let table = derive_gpm_params_table();
        assert_eq!(table.len(), 64);
        assert_eq!(
            table[0],
            GeoParam {
                angle_idx: 0,
                distance_idx: 1
            }
        );
        let p0 = vec![60i32; 64];
        let p1 = vec![60i32; 64];
        for dir in 0..64u8 {
            let out = predict_gpm_block(&p0, &p1, 8, 8, dir, true, 0, 0);
            assert!(out.iter().all(|&v| v == 60));
        }
    }

    #[test]
    fn merge_and_mmvd_candidate_derivation() {
        let mut grid = MotionFieldGrid::new(32, 32);
        let left_cand = InterMergeCand {
            inter_dir: 1,
            mv_field: [MvField::new(Mv::from_int_pel(3, -2), 0), MvField::default()],
            bcw_idx: BCW_DEFAULT,
            use_alt_hpel_if: false,
        };
        grid.fill_region(0, 8, 8, 8, Some(left_cand));
        let hmvp = HmvpTable::default();
        let cands = derive_merge_candidates(&grid, &hmvp, None, 8, 8, 8, 8, false, 6);
        assert_eq!(cands.len(), 6);
        assert_eq!(cands[0].mv_field[0].mv, Mv::from_int_pel(3, -2));

        let mmvd = derive_mmvd_candidate(&cands, 0, 0, 0, false, 1, Some(0), None);
        assert_eq!(mmvd.mv_field[0].mv, Mv::new((3 << 4) + 4, -2 << 4));
    }
}
