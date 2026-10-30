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

//! Clean-room Wave 2 Advanced Inter Prediction (`inter.rs`):
//! - Dynamic Reference List (`DRL`, AV2 Spec §7.10): spatial & temporal MV candidate scanning,
//!   deduplication, and ranking (`NearestMv`, `NearMv`, `GlobalMv`, `NewMv`, `ZeroMv`).
//! - Multi-Reference Frame Search (`LAST_FRAME` + `GOLDEN_FRAME` / `ALTREF_FRAME`).
//! - Compound Inter Prediction (`CompInterType::Average`, `DistanceWeighted`, `DifferenceWeighted`).
//! - Switchable 8-tap Subpel Interpolation Filter RDO (`Regular8Tap`, `Smooth8Tap`, `Sharp8Tap`).

use crate::av2::entropy::BlockContext;
use crate::av2::frame::Frame;
use crate::av2::intops::iclip;
use crate::av2::ipred::compute_sad;
use crate::av2::levels::{CompInterType, InterPredMode, MvXY};
use crate::av2::me::{
    estimate_motion_block, estimate_mv_bits_q8, mc_predict_block, select_best_subpel_filter,
};
use crate::av2::pixel::{BitDepth, Pixel};
use crate::av2::warp::WarpMiInfo;

/// Maximum number of ranked MV predictors in the AV2 Dynamic Reference List (`DRL`).
pub const MAX_DRL_CANDIDATES: usize = 4;

/// Ranked list of spatial/temporal motion vector predictors for a block.
#[derive(Debug, Clone, Copy, Default)]
pub struct DrlCandidateList {
    pub mvs: [MvXY; MAX_DRL_CANDIDATES],
    pub weights: [u16; MAX_DRL_CANDIDATES],
    pub count: usize,
}

impl DrlCandidateList {
    /// Returns the predictor at index `idx` (falling back to `(0, 0)` if out of bounds).
    #[inline]
    pub fn get(&self, idx: usize) -> MvXY {
        if idx < self.count {
            self.mvs[idx]
        } else {
            MvXY::default()
        }
    }

    fn insert_or_accumulate(&mut self, mv: MvXY, weight: u16) {
        for i in 0..self.count {
            if self.mvs[i] == mv {
                self.weights[i] = self.weights[i].saturating_add(weight);
                // Bubble up if weight exceeds predecessor
                let mut k = i;
                while k > 0 && self.weights[k] > self.weights[k - 1] {
                    self.mvs.swap(k, k - 1);
                    self.weights.swap(k, k - 1);
                    k -= 1;
                }
                return;
            }
        }
        if self.count < MAX_DRL_CANDIDATES {
            let idx = self.count;
            self.mvs[idx] = mv;
            self.weights[idx] = weight;
            self.count += 1;
            let mut k = idx;
            while k > 0 && self.weights[k] > self.weights[k - 1] {
                self.mvs.swap(k, k - 1);
                self.weights.swap(k, k - 1);
                k -= 1;
            }
        }
    }
}

/// Maximum number of entries in the normative `RefStackMv` (`MAX_REF_MV_STACK_SIZE`, §7.3).
pub const MAX_REF_MV_STACK: usize = 6;
/// Maximum number of stack comparisons before pruning is skipped (`MAX_PR_NUM`, §7.3).
const MAX_PR_NUM: usize = 16;
/// `MV_BORDER` used by `clamp_mv_row` / `clamp_mv_col` (§5.20.9.4–5).
const MV_BORDER_CLAMP: i32 = 128;

/// A motion-vector candidate stack built by the normative `find_mv_stack`
/// process (AV2 Spec v1.0.0 §7.12.2) for a single-reference block.
#[derive(Debug, Clone, Copy)]
pub struct RefMvStack {
    /// Candidate motion vectors in 1/8-pel units.
    pub mvs: [MvXY; MAX_REF_MV_STACK],
    /// Candidate weights (`WeightStack`).
    pub weights: [u16; MAX_REF_MV_STACK],
    /// `RefStackRowOffset` in 4x4 units.
    pub row_offsets: [i16; MAX_REF_MV_STACK],
    /// `RefStackColOffset` in 4x4 units.
    pub col_offsets: [i16; MAX_REF_MV_STACK],
    /// `NumMvFound`.
    pub count: usize,
}

impl RefMvStack {
    /// Returns candidate `idx`, or `(0, 0)` when the stack is shorter.
    #[inline]
    pub fn get(&self, idx: usize) -> MvXY {
        if idx < self.count {
            self.mvs[idx]
        } else {
            MvXY::default()
        }
    }
}

/// `clamp_mv_row()` (AV2 Spec v1.0.0 §5.20.9.4).
#[inline]
pub fn clamp_mv_row(mv: i32, mi_row: usize, bh4: usize, mi_rows: usize) -> i32 {
    let low = -((mi_row + bh4) as i32) * 4 * 8 - MV_BORDER_CLAMP;
    let high = ((mi_rows - mi_row) as i32) * 4 * 8 + MV_BORDER_CLAMP;
    mv.clamp(low, high)
}

/// `clamp_mv_col()` (AV2 Spec v1.0.0 §5.20.9.5).
#[inline]
pub fn clamp_mv_col(mv: i32, mi_col: usize, bw4: usize, mi_cols: usize) -> i32 {
    let low = -((mi_col + bw4) as i32) * 4 * 8 - MV_BORDER_CLAMP;
    let high = ((mi_cols - mi_col) as i32) * 4 * 8 + MV_BORDER_CLAMP;
    mv.clamp(low, high)
}

/// Normative `find_mv_stack` (AV2 Spec v1.0.0 §7.12.2) for a single-reference
/// inter block, with `use_ref_frame_mvs = 0`, `enable_refmvbank = 0`,
/// `DrlReorder = DRL_REORDER_OFF`, no TIP, and no compound prediction.
///
/// The stack is built from the spatial scan points of §7.12.2.6, the scan-col
/// process of §7.12.2.5, the extra search of §7.12.2.20, and the clamping
/// process of §7.12.2.23. Temporal, derived-compound, and reference-bank
/// candidates are absent because the encoder disables those tools.
#[allow(clippy::too_many_arguments)]
pub fn find_mv_stack(
    mi_map: &[WarpMiInfo],
    mi_rows: usize,
    mi_cols: usize,
    sb_h4: usize,
    tile_row_start4: usize,
    tile_row_end4: usize,
    tile_col_start4: usize,
    tile_col_end4: usize,
    mi_row: usize,
    mi_col: usize,
    bw4: usize,
    bh4: usize,
    ref_id: i8,
) -> RefMvStack {
    struct Builder<'a> {
        map: &'a [WarpMiInfo],
        mi_rows: usize,
        mi_cols: usize,
        tile_row_start4: usize,
        tile_row_end4: usize,
        tile_col_start4: usize,
        tile_col_end4: usize,
        ref_id: i8,
        stack: RefMvStack,
        prune_count: usize,
    }

    impl Builder<'_> {
        fn inside(&self, row: isize, col: isize) -> bool {
            row >= self.tile_row_start4 as isize
                && row < self.tile_row_end4 as isize
                && col >= self.tile_col_start4 as isize
                && col < self.tile_col_end4 as isize
                && row < self.mi_rows as isize
                && col < self.mi_cols as isize
        }

        fn entry(&self, row: usize, col: usize) -> &WarpMiInfo {
            &self.map[row * self.mi_cols + col]
        }

        /// `get_mv(mvRow, mvCol, 0, 0)` with identity global motion: the coded
        /// block MV is the candidate.
        fn candidate_mv(&self, row: usize, col: usize) -> MvXY {
            self.entry(row, col).mv
        }

        /// `search_stack()` (§7.12.2.12).
        fn search_stack(&mut self, row: usize, col: usize, weight: u16) {
            let cand = self.candidate_mv(row, col);
            let mut found = false;
            if self.prune_count < MAX_PR_NUM {
                for idx in 0..self.stack.count {
                    self.prune_count += 1;
                    if self.stack.mvs[idx] == cand {
                        self.stack.weights[idx] = self.stack.weights[idx].saturating_add(weight);
                        found = true;
                        break;
                    }
                }
            }
            if !found && self.stack.count < MAX_REF_MV_STACK {
                let idx = self.stack.count;
                self.stack.mvs[idx] = cand;
                self.stack.weights[idx] = weight;
                self.stack.row_offsets[idx] = 0;
                self.stack.col_offsets[idx] = 0;
                self.stack.count += 1;
            }
        }

        /// `add_reference_motion_vector()` (§7.12.2.10) for `isCompound = 0`.
        fn add_ref_mv(&mut self, row: usize, col: usize, weight: u16) {
            let info = *self.entry(row, col);
            if !info.inter {
                return;
            }
            if info.ref0 == self.ref_id {
                self.search_stack(row, col, weight);
            }
        }

        /// `scan_point()` (§7.12.2.6).
        fn scan_point(
            &mut self,
            mi_row: usize,
            mi_col: usize,
            sb_h4: usize,
            delta_row: isize,
            mut delta_col: isize,
        ) {
            let mv_row = mi_row as isize + delta_row;
            let mut mv_col = mi_col as isize + delta_col;
            let is_sb_border = mi_row.is_multiple_of(sb_h4.max(1));
            if delta_row < 0 && is_sb_border {
                mv_col = (mv_col >> 1) << 1;
                delta_col = mv_col - mi_col as isize;
            }
            let weight = if (delta_row == -1 && delta_col == -1) || delta_col < -1 {
                0
            } else {
                1
            };
            if !self.inside(mv_row, mv_col) {
                return;
            }
            let (row, col) = (mv_row as usize, mv_col as usize);
            if !self.entry(row, col).written {
                return;
            }
            if self.stack.count >= MAX_REF_MV_STACK {
                return;
            }
            self.add_ref_mv(row, col, weight);
        }

        /// `scan_point_if_valid()` inside `scan_col()` (§7.12.2.5).
        fn scan_point_if_valid(
            &mut self,
            mi_row: usize,
            mi_col: usize,
            sb_h4: usize,
            delta_row: isize,
            delta_col: isize,
        ) {
            let mv_row = mi_row as isize + delta_row;
            let mv_col = mi_col as isize + delta_col;
            let mv_other_col = mi_col as isize - 1;
            if !self.inside(mv_row, mv_col) || !self.inside(mv_row, mv_other_col) {
                return;
            }
            let (row, col, other) = (mv_row as usize, mv_col as usize, mv_other_col as usize);
            if self.entry(row, col).origin_x4 != self.entry(row, other).origin_x4 {
                self.scan_point(mi_row, mi_col, sb_h4, delta_row, delta_col);
            }
        }
    }

    let mut b = Builder {
        map: mi_map,
        mi_rows,
        mi_cols,
        tile_row_start4,
        tile_row_end4,
        tile_col_start4,
        tile_col_end4,
        ref_id,
        stack: RefMvStack {
            mvs: [MvXY::default(); MAX_REF_MV_STACK],
            weights: [0; MAX_REF_MV_STACK],
            row_offsets: [0; MAX_REF_MV_STACK],
            col_offsets: [0; MAX_REF_MV_STACK],
            count: 0,
        },
        prune_count: 0,
    };

    let is_sb_border = mi_row.is_multiple_of(sb_h4.max(1));

    // §7.12.2 ordered steps 7–14 (spatial scan points).
    b.scan_point(mi_row, mi_col, sb_h4, bh4 as isize - 1, -1);
    b.scan_point(
        mi_row,
        mi_col,
        sb_h4,
        -1,
        (bw4 as isize - 1 - is_sb_border as isize).max(0),
    );
    if bh4 >= 2 {
        b.scan_point(mi_row, mi_col, sb_h4, 0, -1);
    }
    if bw4 >= if is_sb_border { 4 } else { 2 } {
        b.scan_point(mi_row, mi_col, sb_h4, -1, 0);
    }
    if bh4 <= 16 {
        b.scan_point(mi_row, mi_col, sb_h4, bh4 as isize, -1);
    }
    if bw4 <= 16 {
        b.scan_point(
            mi_row,
            mi_col,
            sb_h4,
            -1,
            if is_sb_border {
                (bw4.max(2)) as isize
            } else {
                bw4 as isize
            },
        );
    }
    b.scan_point(mi_row, mi_col, sb_h4, -1, -1 - is_sb_border as isize);

    // §7.12.2 step 16: scan col with deltaCol = -3.
    let mut delta_col = -3isize;
    if bw4 == 1 {
        delta_col += (mi_col & 1) as isize;
    }
    b.scan_point_if_valid(mi_row, mi_col, sb_h4, bh4 as isize - 1, delta_col);
    if bh4 > 1 {
        b.scan_point_if_valid(mi_row, mi_col, sb_h4, 0, delta_col);
    }

    // §7.12.2.20 extra search: clamp the stack, then add the global (identity)
    // candidate when absent.
    for idx in 0..b.stack.count {
        b.stack.mvs[idx].y = clamp_mv_row(b.stack.mvs[idx].y as i32, mi_row, bh4, mi_rows) as i16;
        b.stack.mvs[idx].x = clamp_mv_col(b.stack.mvs[idx].x as i32, mi_col, bw4, mi_cols) as i16;
    }
    let global = MvXY::default();
    let mut found = false;
    if b.prune_count < MAX_PR_NUM {
        for idx in 0..b.stack.count {
            b.prune_count += 1;
            if b.stack.mvs[idx] == global {
                found = true;
                break;
            }
        }
    }
    if !found && b.stack.count < MAX_REF_MV_STACK {
        let idx = b.stack.count;
        b.stack.mvs[idx] = global;
        b.stack.weights[idx] = 0;
        b.stack.count += 1;
    }

    // Extra mixtures for blocks larger than 32x32 (§7.12.2.20).
    if bw4 * 4 > 32 && bh4 * 4 > 32 {
        let num = b.stack.count;
        let mut insert = |y: usize, x: usize| {
            let cand = MvXY {
                y: b.stack.mvs[y].y,
                x: b.stack.mvs[x].x,
            };
            let mut found = false;
            if b.prune_count < MAX_PR_NUM {
                for idx in 0..b.stack.count {
                    b.prune_count += 1;
                    if b.stack.mvs[idx] == cand {
                        found = true;
                        break;
                    }
                }
            }
            if !found && b.stack.count < MAX_REF_MV_STACK {
                let idx = b.stack.count;
                b.stack.mvs[idx] = cand;
                b.stack.count += 1;
            }
        };
        if num > 1 {
            insert(0, 1);
            insert(1, 0);
        }
        if num > 2 {
            insert(0, 2);
            insert(2, 0);
            insert(1, 2);
            insert(2, 1);
        }
    }

    // §7.12.2.23 clamping process.
    for idx in 0..b.stack.count {
        b.stack.mvs[idx].y = clamp_mv_row(b.stack.mvs[idx].y as i32, mi_row, bh4, mi_rows) as i16;
        b.stack.mvs[idx].x = clamp_mv_col(b.stack.mvs[idx].x as i32, mi_col, bw4, mi_cols) as i16;
    }

    b.stack
}

/// Maximum number of derived compound candidates (`MAX_DR_STACK_SIZE`, §7.3).
pub const MAX_DR_STACK_SIZE: usize = 4;
/// Derived-stack prune limit (`MAX_DR_PR_NUM`, §7.3).
const MAX_DR_PR_NUM: usize = 2;

/// Per-4x4 neighbour mode information for the compound branch of
/// `find_mv_stack` (AV2 Spec v1.0.0 §7.12.2.10 / §7.12.2.13).
///
/// The single-reference [`WarpMiInfo`] map stores only list 0. Compound
/// prediction needs the second reference index, the second motion vector, and
/// the neighbour's `CwpIdx`, so the encoder keeps a parallel map for the
/// compound stack.
#[derive(Debug, Clone, Copy)]
pub struct CompoundMiInfo {
    /// Block origin in 4x4 luma units (used by `scan_col`, §7.12.2.5).
    pub origin_x4: u16,
    /// `IsInters` and availability of the neighbour.
    pub inter: bool,
    /// `RefFrames[..][0]` and `RefFrames[..][1]` (reference indices, `-1` for none).
    pub refs: [i8; 2],
    /// `SubMvs[..][0]` and `SubMvs[..][1]` in 1/8-pel units.
    pub mvs: [MvXY; 2],
    /// `CwpIdxs[..]` (8 = `CWP_EQUAL`).
    pub cwp: u8,
}

impl Default for CompoundMiInfo {
    fn default() -> Self {
        Self {
            origin_x4: 0,
            inter: false,
            refs: [-1, -1],
            mvs: [MvXY::default(); 2],
            cwp: 8,
        }
    }
}

/// A compound motion-vector candidate stack built by the compound branch of
/// `find_mv_stack` (AV2 Spec v1.0.0 §7.12.2, `isCompound = 1`).
#[derive(Debug, Clone, Copy)]
pub struct CompoundMvStack {
    /// `RefStackMv[ idx ][ list ]`.
    pub mvs: [[MvXY; 2]; MAX_REF_MV_STACK],
    /// `WeightStack[ idx ]`.
    pub weights: [u16; MAX_REF_MV_STACK],
    /// `RefStackCwp[ idx ]`.
    pub cwp: [u8; MAX_REF_MV_STACK],
    /// `NumMvFound`.
    pub count: usize,
}

impl CompoundMvStack {
    /// Returns the candidate pair at `idx`, or a zero pair when the stack is shorter.
    #[inline]
    pub fn get(&self, idx: usize) -> [MvXY; 2] {
        if idx < self.count {
            self.mvs[idx]
        } else {
            [MvXY::default(); 2]
        }
    }
}

/// Normative compound branch of `find_mv_stack` (AV2 Spec v1.0.0 §7.12.2)
/// with `use_ref_frame_mvs = 0`, `enable_refmvbank = 0`,
/// `DrlReorder = DRL_REORDER_OFF`, no TIP, and identity global motion.
///
/// It implements the scan points (§7.12.2.6), scan col (§7.12.2.5), compound
/// search stack (§7.12.2.13), compound add derived (§7.12.2.14), fill from
/// derived smvp (§7.12.2.22), extra search (§7.12.2.20), and clamping
/// (§7.12.2.23). Derived candidates that require temporal projection are
/// absent because the encoder disables reference-frame MVs.
#[allow(clippy::too_many_arguments)]
pub fn find_compound_mv_stack(
    mi_map: &[CompoundMiInfo],
    mi_rows: usize,
    mi_cols: usize,
    sb_h4: usize,
    tile_row_start4: usize,
    tile_row_end4: usize,
    tile_col_start4: usize,
    tile_col_end4: usize,
    mi_row: usize,
    mi_col: usize,
    bw4: usize,
    bh4: usize,
    ref0: i8,
    ref1: i8,
    max_drl_bits: usize,
) -> CompoundMvStack {
    struct Builder<'a> {
        map: &'a [CompoundMiInfo],
        mi_rows: usize,
        mi_cols: usize,
        tile_row_start4: usize,
        tile_row_end4: usize,
        tile_col_start4: usize,
        tile_col_end4: usize,
        refs: [i8; 2],
        stack: CompoundMvStack,
        derived: [[MvXY; 2]; MAX_DR_STACK_SIZE],
        derived_count: usize,
        single_ref: [i8; MAX_DR_STACK_SIZE],
        single_mv: [MvXY; MAX_DR_STACK_SIZE],
        single_count: usize,
        prune_count: usize,
        single_prune_count: usize,
        derived_prune_count: usize,
        max_ref_mv_count: usize,
    }

    impl Builder<'_> {
        fn inside(&self, row: isize, col: isize) -> bool {
            row >= self.tile_row_start4 as isize
                && row < self.tile_row_end4 as isize
                && col >= self.tile_col_start4 as isize
                && col < self.tile_col_end4 as isize
                && row < self.mi_rows as isize
                && col < self.mi_cols as isize
        }

        fn entry(&self, row: usize, col: usize) -> &CompoundMiInfo {
            &self.map[row * self.mi_cols + col]
        }

        fn pair_in_stack(&mut self, pair: [MvXY; 2]) -> bool {
            if self.prune_count >= MAX_PR_NUM {
                return false;
            }
            for idx in 0..self.stack.count {
                self.prune_count += 1;
                if self.stack.mvs[idx] == pair {
                    return true;
                }
            }
            false
        }

        /// `compound_search_stack()` (§7.12.2.13).
        fn compound_search_stack(&mut self, row: usize, col: usize, weight: u16) {
            let info = *self.entry(row, col);
            let pair = info.mvs;
            if self.prune_count < MAX_PR_NUM {
                for idx in 0..self.stack.count {
                    self.prune_count += 1;
                    if self.stack.mvs[idx] == pair {
                        self.stack.weights[idx] = self.stack.weights[idx].saturating_add(weight);
                        return;
                    }
                }
            }
            if self.stack.count < MAX_REF_MV_STACK {
                let idx = self.stack.count;
                self.stack.mvs[idx] = pair;
                self.stack.weights[idx] = weight;
                self.stack.cwp[idx] = info.cwp;
                self.stack.count += 1;
            }
        }

        fn add_derived_pair(&mut self, pair: [MvXY; 2]) {
            if self.derived_count >= MAX_DR_STACK_SIZE {
                return;
            }
            if self.derived_prune_count < MAX_DR_PR_NUM {
                for idx in 0..self.derived_count {
                    self.derived_prune_count += 1;
                    if self.derived[idx] == pair {
                        return;
                    }
                }
            }
            if self.derived_count < MAX_DR_STACK_SIZE {
                self.derived[self.derived_count] = pair;
                self.derived_count += 1;
            }
        }

        fn add_single(&mut self, reference: i8, mv: MvXY) {
            if self.single_prune_count < MAX_DR_PR_NUM {
                for idx in 0..self.single_count {
                    self.single_prune_count += 1;
                    if self.single_ref[idx] == reference && self.single_mv[idx] == mv {
                        return;
                    }
                }
            }
            if self.single_count < MAX_DR_STACK_SIZE {
                self.single_ref[self.single_count] = reference;
                self.single_mv[self.single_count] = mv;
                self.single_count += 1;
            }
        }

        /// `compound_add_derived()` (§7.12.2.14) without temporal projection.
        fn compound_add_derived(&mut self, row: usize, col: usize) {
            let info = *self.entry(row, col);
            let cand_ref_idx0 = if info.refs[0] == self.refs[0] || info.refs[1] == self.refs[0] {
                0usize
            } else if info.refs[0] == self.refs[1] || info.refs[1] == self.refs[1] {
                1usize
            } else {
                return;
            };
            let cand_ref_idx1 = 1 - cand_ref_idx0;
            let cand_list = if info.refs[0] == self.refs[cand_ref_idx0] {
                0usize
            } else {
                1usize
            };
            let cand_mv = info.mvs[cand_list];

            for cand_idx in 0..self.single_count {
                if self.single_ref[cand_idx] == self.refs[cand_ref_idx1] {
                    let other = self.single_mv[cand_idx];
                    let pair = if cand_ref_idx0 == 0 {
                        [cand_mv, other]
                    } else {
                        [other, cand_mv]
                    };
                    self.add_derived_pair(pair);
                    break;
                }
            }
            self.add_single(self.refs[cand_ref_idx0], cand_mv);
        }

        /// `add_reference_motion_vector()` (§7.12.2.10) for `isCompound = 1`.
        fn add_ref_mv(&mut self, row: usize, col: usize, weight: u16) {
            let info = *self.entry(row, col);
            if !info.inter {
                return;
            }
            if info.refs[0] == self.refs[0] && info.refs[1] == self.refs[1] {
                self.compound_search_stack(row, col, weight);
            } else {
                self.compound_add_derived(row, col);
            }
        }

        /// `scan_point()` (§7.12.2.6).
        fn scan_point(
            &mut self,
            mi_row: usize,
            mi_col: usize,
            sb_h4: usize,
            delta_row: isize,
            mut delta_col: isize,
        ) {
            let mv_row = mi_row as isize + delta_row;
            let mut mv_col = mi_col as isize + delta_col;
            let is_sb_border = mi_row.is_multiple_of(sb_h4.max(1));
            if delta_row < 0 && is_sb_border {
                mv_col = (mv_col >> 1) << 1;
                delta_col = mv_col - mi_col as isize;
            }
            let weight = if (delta_row == -1 && delta_col == -1) || delta_col < -1 {
                0
            } else {
                1
            };
            if !self.inside(mv_row, mv_col) {
                return;
            }
            let (row, col) = (mv_row as usize, mv_col as usize);
            if self.stack.count >= MAX_REF_MV_STACK {
                return;
            }
            self.add_ref_mv(row, col, weight);
        }

        fn scan_point_if_valid(
            &mut self,
            mi_row: usize,
            mi_col: usize,
            sb_h4: usize,
            delta_row: isize,
            delta_col: isize,
        ) {
            let mv_row = mi_row as isize + delta_row;
            let mv_col = mi_col as isize + delta_col;
            let mv_other_col = mi_col as isize - 1;
            if !self.inside(mv_row, mv_col) || !self.inside(mv_row, mv_other_col) {
                return;
            }
            let (row, col, other) = (mv_row as usize, mv_col as usize, mv_other_col as usize);
            if self.entry(row, col).origin_x4 != self.entry(row, other).origin_x4 {
                self.scan_point(mi_row, mi_col, sb_h4, delta_row, delta_col);
            }
        }
    }

    let mut b = Builder {
        map: mi_map,
        mi_rows,
        mi_cols,
        tile_row_start4,
        tile_row_end4,
        tile_col_start4,
        tile_col_end4,
        refs: [ref0, ref1],
        stack: CompoundMvStack {
            mvs: [[MvXY::default(); 2]; MAX_REF_MV_STACK],
            weights: [0; MAX_REF_MV_STACK],
            cwp: [8; MAX_REF_MV_STACK],
            count: 0,
        },
        derived: [[MvXY::default(); 2]; MAX_DR_STACK_SIZE],
        derived_count: 0,
        single_ref: [-1; MAX_DR_STACK_SIZE],
        single_mv: [MvXY::default(); MAX_DR_STACK_SIZE],
        single_count: 0,
        prune_count: 0,
        single_prune_count: 0,
        derived_prune_count: 0,
        max_ref_mv_count: (1 + max_drl_bits).min(MAX_REF_MV_STACK),
    };

    let is_sb_border = mi_row.is_multiple_of(sb_h4.max(1));

    b.scan_point(mi_row, mi_col, sb_h4, bh4 as isize - 1, -1);
    b.scan_point(
        mi_row,
        mi_col,
        sb_h4,
        -1,
        (bw4 as isize - 1 - is_sb_border as isize).max(0),
    );
    if bh4 >= 2 {
        b.scan_point(mi_row, mi_col, sb_h4, 0, -1);
    }
    if bw4 >= if is_sb_border { 4 } else { 2 } {
        b.scan_point(mi_row, mi_col, sb_h4, -1, 0);
    }
    if bh4 <= 16 {
        b.scan_point(mi_row, mi_col, sb_h4, bh4 as isize, -1);
    }
    if bw4 <= 16 {
        b.scan_point(
            mi_row,
            mi_col,
            sb_h4,
            -1,
            if is_sb_border {
                bw4.max(2) as isize
            } else {
                bw4 as isize
            },
        );
    }
    b.scan_point(mi_row, mi_col, sb_h4, -1, -1 - is_sb_border as isize);

    let mut delta_col = -3isize;
    if bw4 == 1 {
        delta_col += (mi_col & 1) as isize;
    }
    b.scan_point_if_valid(mi_row, mi_col, sb_h4, bh4 as isize - 1, delta_col);
    if bh4 > 1 {
        b.scan_point_if_valid(mi_row, mi_col, sb_h4, 0, delta_col);
    }

    // §7.12.2 step 19: fill mvp from derived smvp.
    for derived_idx in 0..b.derived_count {
        if b.stack.count >= b.max_ref_mv_count {
            break;
        }
        let pair = b.derived[derived_idx];
        if b.pair_in_stack(pair) {
            continue;
        }
        if b.stack.count < b.max_ref_mv_count {
            let idx = b.stack.count;
            b.stack.mvs[idx] = pair;
            b.stack.weights[idx] = 0;
            b.stack.cwp[idx] = 8;
            b.stack.count += 1;
        }
    }

    // §7.12.2.23: clamp before the extra search.
    for idx in 0..b.stack.count {
        for list in 0..2 {
            b.stack.mvs[idx][list].y =
                clamp_mv_row(b.stack.mvs[idx][list].y as i32, mi_row, bh4, mi_rows) as i16;
            b.stack.mvs[idx][list].x =
                clamp_mv_col(b.stack.mvs[idx][list].x as i32, mi_col, bw4, mi_cols) as i16;
        }
    }

    // §7.12.2.20 extra search: the identity global pair is the zero pair.
    let global = [MvXY::default(); 2];
    if b.stack.count < MAX_REF_MV_STACK && !b.pair_in_stack(global) {
        let idx = b.stack.count;
        b.stack.mvs[idx] = global;
        b.stack.weights[idx] = 0;
        b.stack.cwp[idx] = 8;
        b.stack.count += 1;
    }

    // Extra mixtures for blocks larger than 32x32 (§7.12.2.20).
    if bw4 * 4 > 32 && bh4 * 4 > 32 {
        let num = b.stack.count;
        let mut insert = |y: usize, x: usize| {
            let cand = [
                MvXY {
                    y: b.stack.mvs[y][0].y,
                    x: b.stack.mvs[x][0].x,
                },
                MvXY {
                    y: b.stack.mvs[y][1].y,
                    x: b.stack.mvs[x][1].x,
                },
            ];
            if b.pair_in_stack(cand) {
                return;
            }
            if b.stack.count < MAX_REF_MV_STACK {
                let idx = b.stack.count;
                b.stack.mvs[idx] = cand;
                b.stack.weights[idx] = 0;
                b.stack.cwp[idx] = 8;
                b.stack.count += 1;
            }
        };
        if num > 1 {
            insert(0, 1);
            insert(1, 0);
        }
        if num > 2 {
            insert(0, 2);
            insert(2, 0);
            insert(1, 2);
            insert(2, 1);
        }
    }

    for idx in 0..b.stack.count {
        for list in 0..2 {
            b.stack.mvs[idx][list].y =
                clamp_mv_row(b.stack.mvs[idx][list].y as i32, mi_row, bh4, mi_rows) as i16;
            b.stack.mvs[idx][list].x =
                clamp_mv_col(b.stack.mvs[idx][list].x as i32, mi_col, bw4, mi_cols) as i16;
        }
    }

    b.stack
}

/// Builds the ranked Dynamic Reference List (`DRL`) of up to 4 MV predictors by scanning
/// above (`a`) and left (`l`) spatial neighbors across the block span `(bx4_in_sb, by4_in_sb, bw4, bh4)`.
pub fn build_drl_candidates(
    a: &BlockContext,
    l: &BlockContext,
    bx4_in_sb: usize,
    by4_in_sb: usize,
    bw4: usize,
    bh4: usize,
    have_left: bool,
    have_top: bool,
    ref_list_idx: usize,
) -> DrlCandidateList {
    let mut list = DrlCandidateList::default();
    let r_idx = ref_list_idx.min(1);

    if have_top {
        let mut i = 0usize;
        while i < bw4 && (bx4_in_sb + i) < 32 {
            let pos = bx4_in_sb + i;
            if a.intra[pos] == 0 {
                let mv = a.mv[r_idx][pos];
                if !mv.is_zero() {
                    list.insert_or_accumulate(mv, 4);
                }
            }
            i += 1;
        }
        // Top-right neighbor if within superblock
        let tr_pos = bx4_in_sb + bw4;
        if tr_pos < 32 && a.intra[tr_pos] == 0 {
            let mv = a.mv[r_idx][tr_pos];
            if !mv.is_zero() {
                list.insert_or_accumulate(mv, 2);
            }
        }
    }

    if have_left {
        let mut j = 0usize;
        while j < bh4 && (by4_in_sb + j) < 32 {
            let pos = by4_in_sb + j;
            if l.intra[pos] == 0 {
                let mv = l.mv[r_idx][pos];
                if !mv.is_zero() {
                    list.insert_or_accumulate(mv, 4);
                }
            }
            j += 1;
        }
        // Bottom-left neighbor if within superblock
        let bl_pos = by4_in_sb + bh4;
        if bl_pos < 32 && l.intra[bl_pos] == 0 {
            let mv = l.mv[r_idx][bl_pos];
            if !mv.is_zero() {
                list.insert_or_accumulate(mv, 2);
            }
        }
    }

    // Ensure at least one zero predictor fallback is available
    if list.count == 0 {
        list.insert_or_accumulate(MvXY::default(), 1);
    }
    list
}

/// Computes POC distance weights `(w0, w1)` summing to `16` for `CompInterType::DistanceWeighted`
/// based on the relative temporal distances `|cur_poc - ref0_poc|` and `|cur_poc - ref1_poc|`.
#[inline]
pub fn compute_distance_weights(cur_poc: u8, ref0_poc: u8, ref1_poc: u8) -> (i32, i32) {
    let d0 = (cur_poc as i32 - ref0_poc as i32).abs().max(1);
    let d1 = (cur_poc as i32 - ref1_poc as i32).abs().max(1);
    if d0 == d1 {
        (8, 8)
    } else if d0 < d1 {
        let ratio = d1 / d0;
        if ratio >= 3 {
            (12, 4)
        } else if ratio == 2 {
            (11, 5)
        } else {
            (9, 7)
        }
    } else {
        let ratio = d0 / d1;
        if ratio >= 3 {
            (4, 12)
        } else if ratio == 2 {
            (5, 11)
        } else {
            (7, 9)
        }
    }
}

/// Blends two single-reference motion-compensated prediction buffers `pred0` and `pred1` (`w x h`)
/// into `dst` according to `comp_type` (`Average`, `DistanceWeighted`, or `DifferenceWeighted`).
#[allow(clippy::too_many_arguments)]
pub fn blend_compound_predictions<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    dst_stride: usize,
    pred0: &[BD::Pixel],
    pred1: &[BD::Pixel],
    w: usize,
    h: usize,
    comp_type: CompInterType,
    dist_weights: (i32, i32),
) {
    match comp_type {
        CompInterType::Average => {
            for r in 0..h {
                let d_row = dst_off + r * dst_stride;
                let p_row = r * w;
                for c in 0..w {
                    let p0 = pred0[p_row + c].to_i32();
                    let p1 = pred1[p_row + c].to_i32();
                    dst[d_row + c] = bd.clip_pixel((p0 + p1 + 1) >> 1);
                }
            }
        }
        CompInterType::DistanceWeighted => {
            let (w0, w1) = dist_weights;
            for r in 0..h {
                let d_row = dst_off + r * dst_stride;
                let p_row = r * w;
                for c in 0..w {
                    let p0 = pred0[p_row + c].to_i32();
                    let p1 = pred1[p_row + c].to_i32();
                    dst[d_row + c] = bd.clip_pixel((w0 * p0 + w1 * p1 + 8) >> 4);
                }
            }
        }
        CompInterType::DifferenceWeighted | CompInterType::Wedge => {
            let shift = (bd.bitdepth() - 8) as i32;
            let thresh = 16i32 << shift;
            for r in 0..h {
                let d_row = dst_off + r * dst_stride;
                let p_row = r * w;
                for c in 0..w {
                    let p0 = pred0[p_row + c].to_i32();
                    let p1 = pred1[p_row + c].to_i32();
                    let diff = (p0 - p1).abs();
                    let w0 = iclip(8 + (diff - thresh) / (4 << shift), 4, 12);
                    let w1 = 16 - w0;
                    dst[d_row + c] = bd.clip_pixel((w0 * p0 + w1 * p1 + 8) >> 4);
                }
            }
        }
    }
}

/// Complete inter prediction RDO decision for a coding block.
#[derive(Debug, Clone, Copy)]
pub struct InterBlockDecision {
    pub is_compound: bool,
    /// Coded compound mode (18..=23); 0 for single-reference blocks.
    pub compound_mode: u8,
    /// Compound DRL index used for both reference lists (`NEARMV_NEARMV`).
    pub compound_drl_idx: u8,
    pub comp_type: CompInterType,
    pub wedge_idx: u8,
    pub wedge_sign: bool,
    pub mask_sign: bool,
    pub cwp_idx: u8,
    pub ref_slot_idx: u8,
    pub mode: InterPredMode,
    pub drl_idx: u8,
    pub mv: [MvXY; 2],
    pub ref_mv: [MvXY; 2],
    pub interp_filter: u8,
    pub cost: u64,
}

/// Wave 2 Inter Prediction Mode RDO Selector:
/// Evaluates DRL candidates (`NearestMv`, `NearMv`, `ZeroMv`), full motion search (`NewMv`)
/// across primary (`ref0`) and secondary (`ref1`) reference frames, compound bi-prediction
/// (`Average`, `DistanceWeighted`, `DifferenceWeighted`), and switchable 8-tap subpel filters.
#[allow(clippy::too_many_arguments)]
pub fn select_best_inter_mode<BD: BitDepth>(
    bd: BD,
    src_y: &[BD::Pixel],
    src_off: usize,
    src_stride: usize,
    ref0: &Frame<BD::Pixel>,
    ref1: Option<&Frame<BD::Pixel>>,
    cur_poc: u8,
    a: &BlockContext,
    l: &BlockContext,
    bx4_in_sb: usize,
    by4_in_sb: usize,
    bw4: usize,
    bh4: usize,
    bx_px: i32,
    by_px: i32,
    have_left: bool,
    have_top: bool,
    allow_compound: bool,
    lambda_sad: u64,
    speed: u8,
) -> InterBlockDecision {
    let w = bw4 * 4;
    let h = bh4 * 4;
    let drl0 = build_drl_candidates(a, l, bx4_in_sb, by4_in_sb, bw4, bh4, have_left, have_top, 0);
    let pred_mv0 = drl0.get(0);

    let mut pred_buf0 = vec![BD::Pixel::default(); w * h];
    let mut best_dec = InterBlockDecision {
        is_compound: false,
        compound_mode: 0,
        compound_drl_idx: 0,
        comp_type: CompInterType::Average,
        wedge_idx: 0,
        wedge_sign: false,
        mask_sign: false,
        cwp_idx: 0,
        ref_slot_idx: 0,
        mode: InterPredMode::NearestMv,
        drl_idx: 0,
        mv: [pred_mv0, MvXY::default()],
        ref_mv: [pred_mv0, MvXY::default()],
        interp_filter: 0,
        cost: u64::MAX,
    };

    // 1. Evaluate `NearestMv` (`drl_idx = 0`), `NearMv` (`drl_idx = 1`), and `ZeroMv` on `ref0`
    for &(mode, drl_idx, cand_mv, mode_bits) in &[
        (InterPredMode::NearestMv, 0u8, drl0.get(0), 1u64),
        (InterPredMode::NearMv, 1u8, drl0.get(1), 2u64),
        (InterPredMode::ZeroMv, 0u8, MvXY::default(), 2u64),
    ] {
        mc_predict_block(
            bd,
            &mut pred_buf0,
            0,
            w,
            &ref0.planes[0].data,
            ref0.planes[0].stride,
            ref0.planes[0].coded_width as i32,
            ref0.planes[0].coded_height as i32,
            bx_px,
            by_px,
            w,
            h,
            (cand_mv.y as i32) * 2,
            (cand_mv.x as i32) * 2,
            0,
        );
        let sad = compute_sad(src_y, src_off, src_stride, &pred_buf0, 0, w, w, h);
        let cost = sad + lambda_sad * mode_bits;
        if cost < best_dec.cost {
            best_dec.cost = cost;
            best_dec.mode = mode;
            best_dec.drl_idx = drl_idx;
            best_dec.mv[0] = cand_mv;
            best_dec.ref_mv[0] = drl0.get(drl_idx as usize);
        }
    }

    // 2. Evaluate `NewMv` via motion estimation on `ref0`
    let search_range = if speed >= 7 { 12 } else { 20 };
    let (new_mv0, me_cost0) = estimate_motion_block(
        bd,
        src_y,
        src_off,
        src_stride,
        &ref0.planes[0].data,
        ref0.planes[0].stride,
        ref0.planes[0].coded_width as i32,
        ref0.planes[0].coded_height as i32,
        bx_px,
        by_px,
        w,
        h,
        pred_mv0,
        search_range,
        lambda_sad,
    );
    let new_mv0_total = me_cost0 + lambda_sad * 2;
    if new_mv0_total < best_dec.cost {
        best_dec.cost = new_mv0_total;
        best_dec.mode = InterPredMode::NewMv;
        best_dec.drl_idx = 0;
        best_dec.mv[0] = new_mv0;
        best_dec.ref_mv[0] = pred_mv0;
    }

    // 3. Evaluate secondary reference (`ref1`) and Compound Bi-Prediction if available
    if let Some(r1) = ref1 {
        let drl1 =
            build_drl_candidates(a, l, bx4_in_sb, by4_in_sb, bw4, bh4, have_left, have_top, 1);
        let pred_mv1 = drl1.get(0);
        let (new_mv1, me_cost1) = estimate_motion_block(
            bd,
            src_y,
            src_off,
            src_stride,
            &r1.planes[0].data,
            r1.planes[0].stride,
            r1.planes[0].coded_width as i32,
            r1.planes[0].coded_height as i32,
            bx_px,
            by_px,
            w,
            h,
            pred_mv1,
            search_range,
            lambda_sad,
        );
        let single_r1_cost = me_cost1 + lambda_sad * 3;
        if single_r1_cost < best_dec.cost {
            best_dec.cost = single_r1_cost;
            best_dec.is_compound = false;
            best_dec.ref_slot_idx = 1;
            best_dec.mode = InterPredMode::NewMv;
            best_dec.drl_idx = 0;
            best_dec.mv[0] = new_mv1;
            best_dec.ref_mv[0] = pred_mv1;
        }

        // Compound prediction (`w >= 8 && h >= 8` per AV2 spec)
        if allow_compound && w >= 8 && h >= 8 {
            let mut pred_buf1 = vec![BD::Pixel::default(); w * h];
            let mut comp_buf = vec![BD::Pixel::default(); w * h];
            mc_predict_block(
                bd,
                &mut pred_buf0,
                0,
                w,
                &ref0.planes[0].data,
                ref0.planes[0].stride,
                ref0.planes[0].coded_width as i32,
                ref0.planes[0].coded_height as i32,
                bx_px,
                by_px,
                w,
                h,
                (new_mv0.y as i32) * 2,
                (new_mv0.x as i32) * 2,
                0,
            );
            mc_predict_block(
                bd,
                &mut pred_buf1,
                0,
                w,
                &r1.planes[0].data,
                r1.planes[0].stride,
                r1.planes[0].coded_width as i32,
                r1.planes[0].coded_height as i32,
                bx_px,
                by_px,
                w,
                h,
                (new_mv1.y as i32) * 2,
                (new_mv1.x as i32) * 2,
                0,
            );

            let dist_w = compute_distance_weights(cur_poc, ref0.poc, r1.poc);
            let mv_bits_q8 =
                estimate_mv_bits_q8(new_mv0, pred_mv0) + estimate_mv_bits_q8(new_mv1, pred_mv1);
            let mv_rate_cost = ((lambda_sad * mv_bits_q8) >> 8) + lambda_sad * 3;

            for comp_type in [
                CompInterType::Average,
                CompInterType::DistanceWeighted,
                CompInterType::DifferenceWeighted,
            ] {
                blend_compound_predictions(
                    bd,
                    &mut comp_buf,
                    0,
                    w,
                    &pred_buf0,
                    &pred_buf1,
                    w,
                    h,
                    comp_type,
                    dist_w,
                );
                let sad = compute_sad(src_y, src_off, src_stride, &comp_buf, 0, w, w, h);
                let total_comp_cost = sad + mv_rate_cost;
                if total_comp_cost < best_dec.cost {
                    best_dec.cost = total_comp_cost;
                    best_dec.is_compound = true;
                    best_dec.comp_type = comp_type;
                    best_dec.ref_slot_idx = 0;
                    best_dec.mode = InterPredMode::NewMv;
                    best_dec.drl_idx = 0;
                    best_dec.mv = [new_mv0, new_mv1];
                    best_dec.ref_mv = [pred_mv0, pred_mv1];
                }
            }
        }
    }

    // 4. Switchable 8-tap subpel filter RDO (`Regular8Tap`, `Smooth8Tap`, `Sharp8Tap`)
    if speed <= 6 {
        let active_ref = if best_dec.ref_slot_idx == 1 {
            ref1.unwrap_or(ref0)
        } else {
            ref0
        };
        let (best_f, _) = select_best_subpel_filter(
            bd,
            src_y,
            src_off,
            src_stride,
            &active_ref.planes[0].data,
            active_ref.planes[0].stride,
            active_ref.planes[0].coded_width as i32,
            active_ref.planes[0].coded_height as i32,
            bx_px,
            by_px,
            w,
            h,
            best_dec.mv[0],
        );
        best_dec.interp_filter = best_f;
    }

    best_dec
}

/// Scales motion vector `mv` according to the ratio of reference distances `num / den`
/// (AV2 Spec §7.12.3 / dav2d `dav2d_mv_projection`).
pub fn scale_projected_mv(mv: MvXY, num: i32, den: i32) -> MvXY {
    const DIV_MULT: [u16; 32] = [
        0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170,
        1092, 1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
    ];
    let d = den.clamp(1, 31) as usize;
    let n = num.clamp(-31, 31);
    let frac = n * DIV_MULT[d] as i32;
    let y = mv.y as i64 * frac as i64;
    let x = mv.x as i64 * frac as i64;
    let py = ((y + 0x2000 - (y < 0) as i64) >> 14).clamp(-0x7fff, 0x7fff) as i16;
    let px = ((x + 0x2000 - (x < 0) as i64) >> 14).clamp(-0x7fff, 0x7fff) as i16;
    MvXY { x: px, y: py }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_mv_stack_scans_left_then_above() {
        let mi_rows = 16usize;
        let mi_cols = 16usize;
        let mut map = vec![WarpMiInfo::default(); mi_rows * mi_cols];

        let mut left = WarpMiInfo::block(0, 0, 8, 16);
        left.inter = true;
        left.ref0 = 0;
        left.mv = MvXY { x: 8, y: 0 };
        for y in 0..16 {
            for x in 0..8 {
                map[y * mi_cols + x] = left;
            }
        }

        let mut above = WarpMiInfo::block(8, 0, 8, 8);
        above.inter = true;
        above.ref0 = 0;
        above.mv = MvXY { x: 0, y: 8 };
        for y in 0..8 {
            for x in 8..16 {
                map[y * mi_cols + x] = above;
            }
        }

        let stack = find_mv_stack(&map, mi_rows, mi_cols, 16, 0, 16, 0, 16, 8, 8, 4, 4, 0);
        assert!(stack.count >= 2);
        assert_eq!(stack.mvs[0], MvXY { x: 8, y: 0 });
        assert_eq!(stack.mvs[1], MvXY { x: 0, y: 8 });
        assert!(stack.mvs[..stack.count].contains(&MvXY::default()));
    }

    #[test]
    fn test_find_compound_mv_stack_prefers_exact_pair() {
        let mi_rows = 16usize;
        let mi_cols = 16usize;
        let mut map = vec![CompoundMiInfo::default(); mi_rows * mi_cols];

        // Left neighbour with the exact compound pair.
        let mut left = CompoundMiInfo {
            origin_x4: 0,
            inter: true,
            refs: [0, 1],
            mvs: [MvXY { x: 8, y: 0 }, MvXY { x: 0, y: 8 }],
            cwp: 8,
        };
        for y in 0..16 {
            for x in 0..8 {
                map[y * mi_cols + x] = left;
            }
        }
        // Above neighbour with a single reference (must not enter the stack
        // directly, but seeds the derived single list).
        left.refs = [0, -1];
        left.mvs[0] = MvXY { x: 16, y: 16 };
        for y in 0..8 {
            for x in 8..16 {
                map[y * mi_cols + x] = left;
            }
        }

        let stack = find_compound_mv_stack(
            &map, mi_rows, mi_cols, 16, 0, 16, 0, 16, 8, 8, 4, 4, 0, 1, 2,
        );
        assert!(stack.count >= 1);
        assert_eq!(stack.mvs[0][0], MvXY { x: 8, y: 0 });
        assert_eq!(stack.mvs[0][1], MvXY { x: 0, y: 8 });
    }

    #[test]
    fn test_find_compound_mv_stack_adds_global_candidate() {
        let map = vec![CompoundMiInfo::default(); 16 * 16];
        let stack = find_compound_mv_stack(&map, 16, 16, 16, 0, 16, 0, 16, 8, 8, 4, 4, 0, 1, 2);
        assert_eq!(stack.count, 1);
        assert_eq!(stack.mvs[0], [MvXY::default(); 2]);
    }

    #[test]
    fn test_find_mv_stack_clamps_candidates() {
        let mi_rows = 16usize;
        let mi_cols = 16usize;
        let mut map = vec![WarpMiInfo::default(); mi_rows * mi_cols];
        let mut left = WarpMiInfo::block(0, 0, 8, 16);
        left.inter = true;
        left.ref0 = 0;
        left.mv = MvXY {
            x: i16::MAX,
            y: i16::MIN,
        };
        for y in 0..16 {
            for x in 0..8 {
                map[y * mi_cols + x] = left;
            }
        }
        let stack = find_mv_stack(&map, mi_rows, mi_cols, 16, 0, 16, 0, 16, 8, 8, 4, 4, 0);
        assert_eq!(
            stack.mvs[0].y as i32,
            clamp_mv_row(i16::MIN as i32, 8, 4, mi_rows)
        );
        assert_eq!(
            stack.mvs[0].x as i32,
            clamp_mv_col(i16::MAX as i32, 8, 4, mi_cols)
        );
    }
}
