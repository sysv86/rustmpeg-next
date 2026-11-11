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

//! AV2 motion-vector prediction and compound blending.
//!
//! This module implements the normative `Find MV stack` process of
//! AV2 v1.0.0 §7.12.2 for both single (`isCompound = 0`) and compound
//! (`isCompound = 1`) prediction: the `RefStackMv` / `RefStackCwp` /
//! `WeightStack` / `RefStackRowOffset` / `RefStackColOffset` arrays, the
//! `WarpParamStack`, the `NumMvFound` truncation and `MaxPrNum` pruning limits,
//! the §7.12.2.19 sorting process, the temporal/derived projection of
//! §7.12.2.14/§7.12.2.16/§7.12.2.22, the reference-vector bank of
//! §7.12.2.21, the extra search of §7.12.2.20, and the clamping process of
//! §7.12.2.23.
//!
//! The candidate maps are per-4x4: `RefFrames`, `Mvs`, `SubMvs`, `IsInters`,
//! `MotionModes`, `WarpParams`, `MiColBase`, `MiRowBase`, `MiSizes` and
//! `CwpIdxs` live in [`WarpMiInfo`]. The processes are shared by the encoder
//! and the decoder so both sides reconstruct the same predictor from the same
//! already-coded neighbours.
//!
//! Unsupported inputs are gated in the headers, never silently skipped:
//! temporal motion-field scan (`use_ref_frame_mvs = 0`; §7.12.2.7 needs the
//! TIP motion field, and the decoder rejects the flag), TIP prediction
//! (`RefFrame == TIP_FRAME`), IntraBC (which uses its own BVP stack), and
//! global motion (`GmType == IDENTITY`, and the decoder rejects
//! `global_motion`).

use crate::av2::intops::iclip;
use crate::av2::levels::{CompInterType, InterPredMode, MvXY};
use crate::av2::pixel::{BitDepth, Pixel};
use crate::av2::warp::{
    DEFAULT_WARP_PARAMS, MAX_WARP_REF_CANDIDATES, WarpMiInfo, generate_points_from_corners,
};

/// Maximum number of ranked MV predictors in `RefStackMv` (`MAX_REF_MV_STACK_SIZE`, §7.3).
pub const MAX_REF_MV_STACK: usize = 6;
/// Maximum number of stack comparisons before pruning is skipped (`MAX_PR_NUM`, §7.3).
const MAX_PR_NUM: usize = 16;
/// Maximum number of derived compound candidates (`MAX_DR_STACK_SIZE`, §7.3).
pub const MAX_DR_STACK_SIZE: usize = 4;
/// Derived-stack prune limit (`MAX_DR_PR_NUM`, §7.3).
const MAX_DR_PR_NUM: usize = 2;
/// Size of one reference-motion-vector bank (`REF_MV_BANK_SIZE`, §7.3).
pub const REF_MV_BANK_SIZE: usize = 4;
/// Number of reference-motion-vector banks (`BANK_REFS_PER_FRAME`, §7.3).
pub const BANK_REFS_PER_FRAME: usize = 9;
/// `CWP_EQUAL` (§7.3): the neutral compound weighting.
pub const CWP_EQUAL: u8 = 8;
/// `MV_BORDER` used by `clamp_mv_row` / `clamp_mv_col` (§5.20.9.4–5).
const MV_BORDER_CLAMP: i32 = 128;

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

/// `RefMvBank`: the bank of recently used motion vectors of §7.12.3
/// (`RefMvBankParams`, `RefMvBankSize`, `RefMvBankStart`).
///
/// Entry layout matches the specification: `[cwp, key, y0, x0, y1, x1]`.
#[derive(Debug, Clone)]
pub struct RefMvBank {
    /// `RefMvBankSize[ bank ]`.
    pub size: [u8; BANK_REFS_PER_FRAME],
    /// `RefMvBankStart[ bank ]`.
    pub start: [u8; BANK_REFS_PER_FRAME],
    /// `RefMvBankParams[ bank ][ idx ]`.
    pub params: [[[i32; 6]; REF_MV_BANK_SIZE]; BANK_REFS_PER_FRAME],
}

impl Default for RefMvBank {
    fn default() -> Self {
        Self {
            size: [0; BANK_REFS_PER_FRAME],
            start: [0; BANK_REFS_PER_FRAME],
            params: [[[0; 6]; REF_MV_BANK_SIZE]; BANK_REFS_PER_FRAME],
        }
    }
}

impl RefMvBank {
    /// Clears every bank, as `clear_left_context()` does at a tile start.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// `get_rmb_list_index( refFrames )` (§7.12.2.21).
    #[inline]
    fn list_index(refs: [i8; 2]) -> usize {
        let r0 = refs[0];
        let r1 = refs[1];
        if r1 < 0 && (0..=5).contains(&r0) {
            return r0 as usize;
        }
        if r0 == 0 && r1 == 0 {
            6
        } else if r0 == 0 && r1 == 1 {
            7
        } else {
            8
        }
    }

    /// `update_ref_mv_bank( refFrames, mvs, cwpIdx, 1 )` (§7.12.3).
    ///
    /// The hit counters of `update_ref_mv_count()` are maintained by the
    /// callers (see [`RefMvBankHits`]).
    pub fn update(&mut self, refs: [i8; 2], mvs: [MvXY; 2], cwp: u8) {
        let is_compound = refs[1] >= 0;
        let bank = Self::list_index(refs);
        let mut p = [0i32; 6];
        p[0] = cwp as i32;
        p[1] = if is_compound {
            refs[0] as i32 + (refs[1] as i32 + 1) * BANK_REFS_PER_FRAME as i32
        } else {
            refs[0] as i32
        };
        p[2] = mvs[0].y as i32;
        p[3] = mvs[0].x as i32;
        if is_compound {
            p[4] = mvs[1].y as i32;
            p[5] = mvs[1].x as i32;
        }
        let count = self.size[bank] as usize;
        let start = self.start[bank] as usize;
        let mut found = None;
        for i in 0..count {
            let idx = (start + i) % REF_MV_BANK_SIZE;
            if self.params[bank][idx][1..6] == p[1..6] {
                found = Some(i);
                break;
            }
        }
        if let Some(found) = found {
            let idx = (start + found) % REF_MV_BANK_SIZE;
            let tmp = self.params[bank][idx];
            for i in found..count.saturating_sub(1) {
                let a = (start + i) % REF_MV_BANK_SIZE;
                let b = (start + i + 1) % REF_MV_BANK_SIZE;
                self.params[bank][a] = self.params[bank][b];
            }
            let tail = (start + count - 1) % REF_MV_BANK_SIZE;
            self.params[bank][tail] = tmp;
            return;
        }
        let idx = (start + count) % REF_MV_BANK_SIZE;
        self.params[bank][idx] = p;
        if count < REF_MV_BANK_SIZE {
            self.size[bank] = count as u8 + 1;
        } else {
            self.start[bank] = ((start + 1) % REF_MV_BANK_SIZE) as u8;
        }
    }
}

/// `RefMvBankHits`, `RefMvUnitHits` and `RefMvRemainHits` (§7.12.3).
#[derive(Debug, Clone, Copy, Default)]
pub struct RefMvBankHits {
    /// `RefMvBankHits`.
    pub bank_hits: u32,
    /// `RefMvUnitHits`.
    pub unit_hits: u32,
    /// `RefMvRemainHits`.
    pub remain_hits: u32,
}

/// `MAX_RMB_SB_HITS` (§7.3).
pub const MAX_RMB_SB_HITS: u32 = 64;

impl RefMvBankHits {
    /// `update_ref_mv_count()` (§7.12.3).
    pub fn update_count(
        &mut self,
        bw4: usize,
        bh4: usize,
        mi_row: usize,
        mi_col: usize,
        sb4: usize,
    ) {
        let sb4 = sb4.max(1);
        let unit4 = (sb4 >> 3).max(1);
        let unit_count = (bw4 / unit4).max(1) * (bh4 / unit4).max(1);
        if mi_row.is_multiple_of(sb4) && mi_col.is_multiple_of(sb4) {
            self.remain_hits = unit_count.max(4) as u32;
            self.unit_hits = 0;
        } else if mi_row.is_multiple_of(unit4) && mi_col.is_multiple_of(unit4) {
            self.remain_hits += unit_count as u32;
            self.unit_hits = 0;
        }
    }

    /// `update_ref_mv_bank(..., fromWithinSb = 1)` gate (§7.12.3).
    pub fn allow_update(
        &mut self,
        bw4: usize,
        bh4: usize,
        mi_row: usize,
        mi_col: usize,
        sb4: usize,
    ) -> bool {
        self.update_count(bw4, bh4, mi_row, mi_col, sb4);
        if self.remain_hits == 0 || self.unit_hits >= 16 {
            return false;
        }
        self.remain_hits -= 1;
        self.unit_hits += 1;
        self.bank_hits += 1;
        true
    }
}

/// Inputs of the `Find MV stack` process (§7.12.2).
///
/// The struct is filled by the encoder and the decoder from their per-4x4
/// candidate maps. `max_drl_bits` is the frame-header value
/// (`max_drl_bits_minus_1 + 1`), so the process caps the stack at
/// `max_drl_bits + 1` entries as §7.12.2.21/.22 do.
pub struct MvStackParams<'a> {
    /// Per-4x4 candidate map.
    pub map: &'a [WarpMiInfo],
    /// Frame height in 4x4 units.
    pub mi_rows: usize,
    /// Frame width in 4x4 units.
    pub mi_cols: usize,
    /// `Num_4x4_Blocks_High[ SbSize ]`.
    pub sb_h4: usize,
    /// Tile bounds in 4x4 units.
    pub tile_row_start4: usize,
    /// Tile bounds in 4x4 units.
    pub tile_row_end4: usize,
    /// Tile bounds in 4x4 units.
    pub tile_col_start4: usize,
    /// Tile bounds in 4x4 units.
    pub tile_col_end4: usize,
    /// `MiRow` of the current block.
    pub mi_row: usize,
    /// `MiCol` of the current block.
    pub mi_col: usize,
    /// `Num_4x4_Blocks_Wide[ MiSize ]`.
    pub bw4: usize,
    /// `Num_4x4_Blocks_High[ MiSize ]`.
    pub bh4: usize,
    /// `RefFrame[ 0 ]` and `RefFrame[ 1 ]` (`-1` for `NONE`).
    pub refs: [i8; 2],
    /// `isCompound`.
    pub is_compound: bool,
    /// `DeriveWrl` (§5.20.7.6): the `WarpParamStack` is built.
    pub derive_wrl: bool,
    /// `use_ref_frame_mvs` (always 0 for a stream this codec supports).
    pub use_ref_frame_mvs: bool,
    /// `DrlReorder`: 0 `DRL_REORDER_DISABLED`, 1 `CONSTRAINT`, 2 `ALWAYS`.
    pub drl_reorder: u8,
    /// `enable_refmvbank`.
    pub enable_refmvbank: bool,
    /// `max_drl_bits` = `max_drl_bits_minus_1 + 1` (§5.18).
    pub max_drl_bits: usize,
    /// `FrameDistance[ ref ]` by reference slot (`get_relative_dist`).
    pub frame_distance: &'a [i32; 7],
    /// `WarpBankParams[ ref ]`, most recent entry first.
    pub warp_bank: &'a [Vec<[i32; 6]>],
    /// `RefMvBankParams`, or `None` when `enable_refmvbank = 0`.
    pub refmv_bank: Option<&'a RefMvBank>,
}

/// `RefStackMv` and the arrays that accompany it (§7.12.2).
#[derive(Debug, Clone, Copy)]
pub struct RefMvStack {
    /// `RefStackMv[ idx ][ list ]` in 1/8-pel units.
    pub mvs: [[MvXY; 2]; MAX_REF_MV_STACK],
    /// `RefStackRefFrame[ idx ][ list ]`: the candidate's reference frames.
    pub refs: [[i8; 2]; MAX_REF_MV_STACK],
    /// `WeightStack[ idx ]`.
    pub weights: [u16; MAX_REF_MV_STACK],
    /// `RefStackCwp[ idx ]`.
    pub cwp: [u8; MAX_REF_MV_STACK],
    /// `RefStackRowOffset[ idx ]` in 4x4 units.
    pub row_offsets: [i16; MAX_REF_MV_STACK],
    /// `RefStackColOffset[ idx ]` in 4x4 units.
    pub col_offsets: [i16; MAX_REF_MV_STACK],
    /// `NumMvFound`.
    pub count: usize,
    /// `WarpParamStack[ idx ]`.
    pub warp: [[i32; 6]; MAX_WARP_REF_CANDIDATES],
    /// `NumWarpFound`.
    pub num_warp: usize,
}

impl RefMvStack {
    fn new() -> Self {
        Self {
            mvs: [[MvXY::default(); 2]; MAX_REF_MV_STACK],
            refs: [[-1; 2]; MAX_REF_MV_STACK],
            weights: [0; MAX_REF_MV_STACK],
            cwp: [CWP_EQUAL; MAX_REF_MV_STACK],
            row_offsets: [0; MAX_REF_MV_STACK],
            col_offsets: [0; MAX_REF_MV_STACK],
            count: 0,
            warp: [DEFAULT_WARP_PARAMS; MAX_WARP_REF_CANDIDATES],
            num_warp: 0,
        }
    }

    /// Returns the list-0 motion vector at `idx`, or `(0, 0)` when the stack is shorter.
    #[inline]
    pub fn get(&self, idx: usize) -> MvXY {
        if idx < self.count {
            self.mvs[idx][0]
        } else {
            MvXY::default()
        }
    }

    /// Returns both lists at `idx`, or a zero pair when the stack is shorter.
    #[inline]
    pub fn get_pair(&self, idx: usize) -> [MvXY; 2] {
        if idx < self.count {
            self.mvs[idx]
        } else {
            [MvXY::default(); 2]
        }
    }
}

struct StackBuilder<'a, 'b> {
    p: &'a MvStackParams<'b>,
    stack: RefMvStack,
    derived: [[MvXY; 2]; MAX_DR_STACK_SIZE],
    derived_count: usize,
    single_ref: [i8; MAX_DR_STACK_SIZE],
    single_mv: [MvXY; MAX_DR_STACK_SIZE],
    single_count: usize,
    prune_count: usize,
    derived_prune_count: usize,
    single_prune_count: usize,
}

impl StackBuilder<'_, '_> {
    #[inline]
    fn inside(&self, row: isize, col: isize) -> bool {
        row >= self.p.tile_row_start4 as isize
            && row < self.p.tile_row_end4 as isize
            && col >= self.p.tile_col_start4 as isize
            && col < self.p.tile_col_end4 as isize
            && row < self.p.mi_rows as isize
            && col < self.p.mi_cols as isize
    }

    #[inline]
    fn entry(&self, row: usize, col: usize) -> &WarpMiInfo {
        &self.p.map[row * self.p.mi_cols + col]
    }

    #[inline]
    fn frame_distance(&self, reference: i8) -> i32 {
        if reference >= 0 && (reference as usize) < self.p.frame_distance.len() {
            self.p.frame_distance[reference as usize]
        } else {
            0
        }
    }

    /// `get_mv( mvRow, mvCol, refList, candList )` (§7.12.2.14).
    ///
    /// Global motion is not signalled by this codec (`GmType == IDENTITY` and
    /// the decoder rejects `global_motion`), so the `GlobalMvs` branch of the
    /// specification is unreachable and the sub-block vector is returned.
    #[inline]
    fn get_mv(&self, row: usize, col: usize, _ref_list: isize, cand_list: usize) -> MvXY {
        self.entry(row, col).sub_list_mv(cand_list)
    }

    /// `insert warp candidate` (§7.12.2.11).
    fn add_warp(&mut self, params: [i32; 6]) {
        if self.stack.num_warp >= MAX_WARP_REF_CANDIDATES {
            return;
        }
        self.stack.warp[self.stack.num_warp] = params;
        self.stack.num_warp += 1;
    }

    /// `swap_stack( i, j )` (§7.12.2.19).
    fn swap_stack(&mut self, i: usize, j: usize) {
        self.stack.weights.swap(i, j);
        self.stack.cwp.swap(i, j);
        self.stack.row_offsets.swap(i, j);
        self.stack.col_offsets.swap(i, j);
        self.stack.mvs.swap(i, j);
        self.stack.refs.swap(i, j);
    }

    /// `sorting process` (§7.12.2.19): move the highest-weight entry to `start`.
    fn sort(&mut self, start: usize, end: usize) {
        if end <= start + 1 {
            return;
        }
        let mut max_weight = self.stack.weights[start];
        let mut max_idx = start;
        for idx in start + 1..end {
            if max_weight < self.stack.weights[idx] {
                max_weight = self.stack.weights[idx];
                max_idx = idx;
            }
        }
        if max_idx != start {
            self.swap_stack(start, max_idx);
        }
    }

    /// `search stack` (§7.12.2.12) for single-reference prediction.
    fn search_stack(&mut self, row: usize, col: usize, weight: u16, cand_list: usize) {
        let cand = self.get_mv(row, col, 0, cand_list);
        let mut found = false;
        if self.prune_count < MAX_PR_NUM {
            for idx in 0..self.stack.count {
                self.prune_count += 1;
                if self.stack.mvs[idx][0] == cand {
                    self.stack.weights[idx] = self.stack.weights[idx].saturating_add(weight);
                    found = true;
                    break;
                }
            }
        }
        if !found && self.stack.count < MAX_REF_MV_STACK {
            let idx = self.stack.count;
            let (ref0, ref1) = {
                let info = self.entry(row, col);
                (info.ref0, info.ref1)
            };
            self.stack.mvs[idx] = [cand, MvXY::default()];
            self.stack.refs[idx] = [ref0, ref1];
            self.stack.row_offsets[idx] = (row as i16).wrapping_sub(self.p.mi_row as i16);
            self.stack.col_offsets[idx] = (col as i16).wrapping_sub(self.p.mi_col as i16);
            self.stack.weights[idx] = weight;
            self.stack.cwp[idx] = CWP_EQUAL;
            self.stack.count += 1;
        }
    }

    /// `compound search stack` (§7.12.2.13).
    fn compound_search_stack(&mut self, row: usize, col: usize, weight: u16) {
        let info = *self.entry(row, col);
        let cand = [info.sub_mv, info.sub_mv1];
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
            self.stack.refs[idx] = [info.ref0, info.ref1];
            self.stack.cwp[idx] = if info.cwp == 0 { CWP_EQUAL } else { info.cwp };
            self.stack.row_offsets[idx] = (row as i16).wrapping_sub(self.p.mi_row as i16);
            self.stack.col_offsets[idx] = (col as i16).wrapping_sub(self.p.mi_col as i16);
            self.stack.weights[idx] = weight;
            self.stack.count += 1;
        }
    }

    /// `comp_mv_in_stack()` (§7.12.2.14).
    fn comp_mv_in_stack(&mut self, list0: MvXY, list1: MvXY) -> bool {
        if self.derived_prune_count < MAX_DR_PR_NUM {
            for i in 0..self.derived_count {
                self.derived_prune_count += 1;
                if self.derived[i][0] == list0 && self.derived[i][1] == list1 {
                    return true;
                }
            }
        }
        false
    }

    /// `single add derived` (§7.12.2.16).
    fn single_add_derived(&mut self, row: usize, col: usize, cand_list: usize) {
        let cand_ref = self.entry(row, col).list_ref(cand_list);
        if cand_ref < 0 {
            return;
        }
        let cand_mv = self.get_mv(row, col, -1, cand_list);
        let cur_dist = self.frame_distance(self.p.refs[0]);
        let cand_dist = self.frame_distance(cand_ref);
        let have_proj = (cur_dist > 0 && cand_dist > 0) || (cur_dist < 0 && cand_dist < 0);
        if !have_proj {
            return;
        }
        let proj = scale_projected_mv(cand_mv, cur_dist.abs(), cand_dist.abs());
        if self.derived_prune_count < MAX_DR_PR_NUM {
            for i in 0..self.derived_count {
                self.derived_prune_count += 1;
                if self.derived[i][0] == proj {
                    return;
                }
            }
        }
        if self.derived_count < MAX_DR_STACK_SIZE {
            self.derived[self.derived_count] = [proj, MvXY::default()];
            self.derived_count += 1;
        }
    }

    /// `compound add derived` (§7.12.2.14), without the `enable_mv_traj` path.
    fn compound_add_derived(&mut self, row: usize, col: usize) {
        let info = *self.entry(row, col);
        let (cand_ref_idx0, cand_ref_idx1) =
            if info.ref0 == self.p.refs[0] || info.ref1 == self.p.refs[0] {
                (0usize, 1usize)
            } else if info.ref0 == self.p.refs[1] || info.ref1 == self.p.refs[1] {
                (1usize, 0usize)
            } else {
                return;
            };
        let cand_list = usize::from(info.ref0 != self.p.refs[cand_ref_idx0]);
        let cand_mv = self.get_mv(row, col, cand_ref_idx0 as isize, cand_list);
        let mut cand_idx = 0usize;
        while cand_idx < self.single_count && self.derived_count < MAX_DR_STACK_SIZE {
            if self.single_ref[cand_idx] == self.p.refs[cand_ref_idx1] {
                let other = self.single_mv[cand_idx];
                let (l0, l1) = if cand_ref_idx0 == 0 {
                    (cand_mv, other)
                } else {
                    (other, cand_mv)
                };
                if !self.comp_mv_in_stack(l0, l1) {
                    self.derived[self.derived_count] = [l0, l1];
                    self.derived_count += 1;
                }
                break;
            }
            cand_idx += 1;
        }
        if self.single_prune_count < MAX_DR_PR_NUM {
            for cand_idx in 0..self.single_count {
                self.single_prune_count += 1;
                if self.single_ref[cand_idx] == self.p.refs[cand_ref_idx0]
                    && self.single_mv[cand_idx] == cand_mv
                {
                    return;
                }
            }
        }
        if self.single_count < MAX_DR_STACK_SIZE {
            self.single_ref[self.single_count] = self.p.refs[cand_ref_idx0];
            self.single_mv[self.single_count] = cand_mv;
            self.single_count += 1;
        }
    }

    /// `add reference motion vector` (§7.12.2.10).
    fn add_ref_mv(&mut self, row: usize, col: usize, weight: u16) {
        let info = *self.entry(row, col);
        if !info.inter {
            return;
        }
        if !self.p.is_compound {
            for cand_list in 0..2usize {
                if info.list_ref(cand_list) == self.p.refs[0] {
                    self.search_stack(row, col, weight, cand_list);
                } else if info.list_ref(cand_list) >= 0 {
                    // `is_derivable_ref_frame( candRefFrames, candList )`.
                    self.single_add_derived(row, col, cand_list);
                }
            }
        } else if info.ref0 == self.p.refs[0] && info.ref1 == self.p.refs[1] {
            self.compound_search_stack(row, col, weight);
        } else {
            self.compound_add_derived(row, col);
        }
    }

    /// `scan point` (§7.12.2.6).
    fn scan_point(&mut self, delta_row: isize, delta_col: isize) {
        let mut delta_col = delta_col;
        let mv_row = self.p.mi_row as isize + delta_row;
        let mut mv_col = self.p.mi_col as isize + delta_col;
        let is_sb_border = self.p.mi_row.is_multiple_of(self.p.sb_h4.max(1));
        if delta_row < 0 && is_sb_border {
            mv_col = (mv_col >> 1) << 1;
            delta_col = mv_col - self.p.mi_col as isize;
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
        let info = *self.entry(row, col);
        if !info.written {
            return;
        }
        // `add warp motion vector` (§7.12.2.9).
        if self.p.derive_wrl && info.motion_mode >= 2 && info.inter && info.ref0 == self.p.refs[0] {
            self.add_warp(info.warp);
        }
        if self.stack.count >= MAX_REF_MV_STACK {
            return;
        }
        self.add_ref_mv(row, col, weight);
    }

    /// `scan point if valid` inside `scan col` (§7.12.2.5).
    fn scan_point_if_valid(&mut self, delta_row: isize, delta_col: isize) {
        let mv_row = self.p.mi_row as isize + delta_row;
        let mv_col = self.p.mi_col as isize + delta_col;
        let mv_other_col = self.p.mi_col as isize - 1;
        if !self.inside(mv_row, mv_col) || !self.inside(mv_row, mv_other_col) {
            return;
        }
        let (row, col, other) = (mv_row as usize, mv_col as usize, mv_other_col as usize);
        if self.entry(row, col).origin_x4 != self.entry(row, other).origin_x4 {
            self.scan_point(delta_row, delta_col);
        }
    }

    /// `fill mvp from derived smvp` (§7.12.2.22).
    fn fill_from_derived(&mut self) {
        let is_compound = self.p.is_compound;
        let max_ref = (self.p.max_drl_bits + 1).min(MAX_REF_MV_STACK);
        if self.stack.count >= max_ref {
            return;
        }
        for derived_idx in 0..self.derived_count {
            let mut found = false;
            if self.prune_count < MAX_PR_NUM {
                for idx in 0..self.stack.count {
                    self.prune_count += 1;
                    let mut equal = true;
                    for list in 0..=usize::from(is_compound) {
                        if self.derived[derived_idx][list] != self.stack.mvs[idx][list] {
                            equal = false;
                            break;
                        }
                    }
                    if equal {
                        found = true;
                        break;
                    }
                }
            }
            if !found && self.stack.count < max_ref {
                let idx = self.stack.count;
                for list in 0..=usize::from(is_compound) {
                    self.stack.mvs[idx][list] = self.derived[derived_idx][list];
                }
                self.stack.refs[idx] = if is_compound {
                    [self.p.refs[0], self.p.refs[1]]
                } else {
                    [self.p.refs[0], -1]
                };
                self.stack.cwp[idx] = CWP_EQUAL;
                self.stack.row_offsets[idx] = 0;
                self.stack.col_offsets[idx] = 0;
                self.stack.count += 1;
            }
        }
    }

    /// `check rmb cand` inside `fill mvp from ref mv bank` (§7.12.2.21).
    fn check_rmb_cand(&mut self, cand: [MvXY; 2], cwp: u8) {
        let is_compound = self.p.is_compound;
        if self.prune_count < MAX_PR_NUM {
            for idx in 0..self.stack.count {
                self.prune_count += 1;
                if cand[0] == self.stack.mvs[idx][0]
                    && (!is_compound || cand[1] == self.stack.mvs[idx][1])
                {
                    return;
                }
            }
        }
        let bw = (self.p.bw4 * 4) as i32;
        let bh = (self.p.bh4 * 4) as i32;
        for mv in cand.iter().take(1 + usize::from(is_compound)) {
            let ref_y = (self.p.mi_row * 4) as i32 + (mv.y as i32) / 8;
            let ref_x = (self.p.mi_col * 4) as i32 + (mv.x as i32) / 8;
            if ref_x <= -bw
                || ref_y <= -bh
                || ref_x >= (self.p.mi_cols * 4) as i32
                || ref_y >= (self.p.mi_rows * 4) as i32
            {
                return;
            }
        }
        if self.stack.count < MAX_REF_MV_STACK {
            let idx = self.stack.count;
            self.stack.mvs[idx] = cand;
            self.stack.refs[idx] = [self.p.refs[0], self.p.refs[1]];
            self.stack.cwp[idx] = cwp;
            self.stack.row_offsets[idx] = 0;
            self.stack.col_offsets[idx] = 0;
            self.stack.weights[idx] = 0;
            self.stack.count += 1;
        }
    }

    /// `fill mvp from ref mv bank` (§7.12.2.21).
    fn fill_from_bank(&mut self) {
        let Some(bank) = self.p.refmv_bank else {
            return;
        };
        let refs = self.p.refs;
        let is_compound = self.p.is_compound;
        let bank_idx = RefMvBank::list_index(refs);
        let key = if is_compound {
            refs[0] as i32 + (refs[1] as i32 + 1) * BANK_REFS_PER_FRAME as i32
        } else {
            refs[0] as i32
        };
        let max_ref = (self.p.max_drl_bits + 1).min(MAX_REF_MV_STACK);
        let count = bank.size[bank_idx] as usize;
        let start = bank.start[bank_idx] as usize;
        for i in (0..count).rev() {
            if self.stack.count >= max_ref {
                break;
            }
            let idx = (start + i) % REF_MV_BANK_SIZE;
            let params = bank.params[bank_idx][idx];
            if params[1] != key {
                continue;
            }
            let cand = [
                MvXY {
                    y: params[2] as i16,
                    x: params[3] as i16,
                },
                MvXY {
                    y: params[4] as i16,
                    x: params[5] as i16,
                },
            ];
            self.check_rmb_cand(cand, params[0].clamp(0, 255) as u8);
        }
    }

    /// `insert mvp candidate` inside the extra search (§7.12.2.20).
    fn insert_mvp_candidate(&mut self, y_cand: usize, x_cand: usize) {
        let is_compound = self.p.is_compound;
        let cand = [
            MvXY {
                y: self.stack.mvs[y_cand][0].y,
                x: self.stack.mvs[x_cand][0].x,
            },
            MvXY {
                y: self.stack.mvs[y_cand][1].y,
                x: self.stack.mvs[x_cand][1].x,
            },
        ];
        if self.prune_count < MAX_PR_NUM {
            for idx in 0..self.stack.count {
                self.prune_count += 1;
                if cand[0] == self.stack.mvs[idx][0]
                    && (!is_compound || cand[1] == self.stack.mvs[idx][1])
                {
                    return;
                }
            }
        }
        if self.stack.count < MAX_REF_MV_STACK {
            let idx = self.stack.count;
            self.stack.mvs[idx] = cand;
            self.stack.refs[idx] = [self.p.refs[0], self.p.refs[1]];
            self.stack.cwp[idx] = CWP_EQUAL;
            self.stack.row_offsets[idx] = 0;
            self.stack.col_offsets[idx] = 0;
            self.stack.weights[idx] = 0;
            self.stack.count += 1;
        }
    }

    /// `extra search process` (§7.12.2.20).
    fn extra_search(&mut self) {
        let is_compound = self.p.is_compound;
        let num_lists = 1 + usize::from(is_compound);
        for list in 0..num_lists {
            for idx in 0..self.stack.count {
                self.stack.mvs[idx][list].y = clamp_mv_row(
                    self.stack.mvs[idx][list].y as i32,
                    self.p.mi_row,
                    self.p.bh4,
                    self.p.mi_rows,
                ) as i16;
                self.stack.mvs[idx][list].x = clamp_mv_col(
                    self.stack.mvs[idx][list].x as i32,
                    self.p.mi_col,
                    self.p.bw4,
                    self.p.mi_cols,
                ) as i16;
            }
        }
        if self.stack.count < MAX_REF_MV_STACK {
            let global = [MvXY::default(); 2];
            let mut found = false;
            if self.prune_count < MAX_PR_NUM {
                for idx in 0..self.stack.count {
                    self.prune_count += 1;
                    if global[0] == self.stack.mvs[idx][0]
                        && (!is_compound || global[1] == self.stack.mvs[idx][1])
                    {
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                let idx = self.stack.count;
                for list in 0..num_lists {
                    self.stack.mvs[idx][list] = global[list];
                }
                self.stack.refs[idx] = [
                    self.p.refs[0],
                    if is_compound { self.p.refs[1] } else { -1 },
                ];
                self.stack.cwp[idx] = CWP_EQUAL;
                self.stack.row_offsets[idx] = 0;
                self.stack.col_offsets[idx] = 0;
                self.stack.weights[idx] = 0;
                self.stack.count += 1;
            }
        }
        if self.p.bw4 * 4 > 32 && self.p.bh4 * 4 > 32 {
            let num = self.stack.count;
            if num > 1 {
                self.insert_mvp_candidate(0, 1);
                self.insert_mvp_candidate(1, 0);
            }
            if num > 2 {
                self.insert_mvp_candidate(0, 2);
                self.insert_mvp_candidate(2, 0);
                self.insert_mvp_candidate(1, 2);
                self.insert_mvp_candidate(2, 1);
            }
        }
        if self.p.derive_wrl {
            let reference = self.p.refs[0];
            if reference >= 0 {
                if let Some(bank) = self.p.warp_bank.get(reference as usize) {
                    for params in bank.iter() {
                        self.add_warp(*params);
                    }
                }
                self.add_warp(DEFAULT_WARP_PARAMS);
                self.add_warp(DEFAULT_WARP_PARAMS);
            }
        }
    }

    /// `clamping process` (§7.12.2.23).
    fn clamp_all(&mut self) {
        let num_lists = 1 + usize::from(self.p.is_compound);
        for list in 0..num_lists {
            for idx in 0..self.stack.count {
                self.stack.mvs[idx][list].y = clamp_mv_row(
                    self.stack.mvs[idx][list].y as i32,
                    self.p.mi_row,
                    self.p.bh4,
                    self.p.mi_rows,
                ) as i16;
                self.stack.mvs[idx][list].x = clamp_mv_col(
                    self.stack.mvs[idx][list].x as i32,
                    self.p.mi_col,
                    self.p.bw4,
                    self.p.mi_cols,
                ) as i16;
            }
        }
    }
}

/// `find_mv_stack( isCompound )` (AV2 Spec v1.0.0 §7.12.2).
///
/// Builds `RefStackMv`, `RefStackCwp`, `WeightStack`, the stack offsets, and —
/// when `derive_wrl` is set — `WarpParamStack`, for the current block.
///
/// # Panics
/// Debug builds assert that `use_ref_frame_mvs` is 0: the temporal scan of
/// §7.12.2.7 consumes the TIP motion field (§7.10), which this codec does not
/// build, and the decoder rejects any stream that signals the flag.
pub fn find_mv_stack(p: &MvStackParams<'_>) -> RefMvStack {
    debug_assert!(
        !p.use_ref_frame_mvs,
        "temporal motion-field candidates require use_ref_frame_mvs = 0"
    );

    let mut b = StackBuilder {
        p,
        stack: RefMvStack::new(),
        derived: [[MvXY::default(); 2]; MAX_DR_STACK_SIZE],
        derived_count: 0,
        single_ref: [-1; MAX_DR_STACK_SIZE],
        single_mv: [MvXY::default(); MAX_DR_STACK_SIZE],
        single_count: 0,
        prune_count: 0,
        derived_prune_count: 0,
        single_prune_count: 0,
    };

    let is_sb_border = p.mi_row.is_multiple_of(p.sb_h4.max(1));

    // Steps 4–5: `generate points from corners` (§7.12.2.3) for the warp stack.
    if p.derive_wrl {
        if let Some(c) = generate_points_from_corners(
            p.map, p.mi_cols, p.mi_rows, p.mi_row, p.mi_col, p.bw4, p.bh4, p.sb_h4, p.sb_h4, 0,
            p.refs[0],
        ) {
            b.add_warp(c);
        } else if p.bw4 <= 16
            && let Some(c) = generate_points_from_corners(
                p.map, p.mi_cols, p.mi_rows, p.mi_row, p.mi_col, p.bw4, p.bh4, p.sb_h4, p.sb_h4, 1,
                p.refs[0],
            )
        {
            b.add_warp(c);
        }
    }

    // Steps 7–12: ordered spatial scan points.
    b.scan_point(p.bh4 as isize - 1, -1);
    b.scan_point(
        -1,
        if is_sb_border {
            (p.bw4 as isize - 2).max(0)
        } else {
            p.bw4 as isize - 1
        },
    );
    if p.bh4 >= 2 {
        b.scan_point(0, -1);
    }
    if p.bw4 >= if is_sb_border { 4 } else { 2 } {
        b.scan_point(-1, 0);
    }
    if p.bh4 <= 16 {
        b.scan_point(p.bh4 as isize, -1);
    }
    if p.bw4 <= 16 {
        b.scan_point(
            -1,
            if is_sb_border {
                p.bw4.max(2) as isize
            } else {
                p.bw4 as isize
            },
        );
    }
    // Steps 13–14: the temporal scan is gated off; the diagonal scan point
    // follows.
    b.scan_point(-1, -1 - is_sb_border as isize);

    // Step 15: `numNearest`.
    let num_nearest = b.stack.count;

    // Step 16: scan col with deltaCol = -3.
    let mut delta_col = -3isize;
    if p.bw4 == 1 {
        delta_col += (p.mi_col & 1) as isize;
    }
    b.scan_point_if_valid(p.bh4 as isize - 1, delta_col);
    if p.bh4 > 1 {
        b.scan_point_if_valid(0, delta_col);
    }

    // Steps 17–18: sorting.
    let use_sort = p.drl_reorder == 2 || (p.drl_reorder == 1 && num_nearest >= 4);
    if use_sort {
        b.sort(0, num_nearest);
    }

    // Steps 19–21: derived candidates and the reference MV bank.
    if p.is_compound {
        b.fill_from_derived();
    }
    if p.enable_refmvbank {
        b.fill_from_bank();
    }
    if !p.is_compound {
        b.fill_from_derived();
    }

    // Steps 22–23.
    b.extra_search();
    b.clamp_all();

    b.stack
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

/// Scales motion vector `mv` according to the ratio of reference distances `num / den`
/// (AV2 Spec §7.9.4 / dav2d `dav2d_mv_projection`).
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

    fn params<'a>(
        map: &'a [WarpMiInfo],
        mi_rows: usize,
        mi_cols: usize,
        refs: [i8; 2],
        is_compound: bool,
        warp_bank: &'a [Vec<[i32; 6]>],
        frame_distance: &'a [i32; 7],
    ) -> MvStackParams<'a> {
        MvStackParams {
            map,
            mi_rows,
            mi_cols,
            sb_h4: 16,
            tile_row_start4: 0,
            tile_row_end4: mi_rows,
            tile_col_start4: 0,
            tile_col_end4: mi_cols,
            mi_row: 8,
            mi_col: 8,
            bw4: 4,
            bh4: 4,
            refs,
            is_compound,
            derive_wrl: false,
            use_ref_frame_mvs: false,
            drl_reorder: 0,
            enable_refmvbank: false,
            max_drl_bits: 2,
            frame_distance,
            warp_bank,
            refmv_bank: None,
        }
    }

    #[test]
    fn test_find_mv_stack_scans_left_then_above() {
        let mi_rows = 16usize;
        let mi_cols = 16usize;
        let mut map = vec![WarpMiInfo::default(); mi_rows * mi_cols];

        let mut left = WarpMiInfo::block(0, 0, 8, 16);
        left.inter = true;
        left.ref0 = 0;
        left.mv = MvXY { x: 8, y: 0 };
        left.sub_mv = left.mv;
        for y in 0..16 {
            for x in 0..8 {
                map[y * mi_cols + x] = left;
            }
        }

        let mut above = WarpMiInfo::block(8, 0, 8, 8);
        above.inter = true;
        above.ref0 = 0;
        above.mv = MvXY { x: 0, y: 8 };
        above.sub_mv = above.mv;
        for y in 0..8 {
            for x in 8..16 {
                map[y * mi_cols + x] = above;
            }
        }

        let bank: Vec<Vec<[i32; 6]>> = vec![Vec::new(); 8];
        let dist = [0i32; 7];
        let stack = find_mv_stack(&params(
            &map,
            mi_rows,
            mi_cols,
            [0, -1],
            false,
            &bank,
            &dist,
        ));
        assert!(stack.count >= 2);
        assert_eq!(stack.mvs[0][0], MvXY { x: 8, y: 0 });
        assert_eq!(stack.mvs[1][0], MvXY { x: 0, y: 8 });
        assert!(
            stack.mvs[..stack.count]
                .iter()
                .any(|m| m[0] == MvXY::default())
        );
    }

    #[test]
    fn test_find_mv_stack_adds_global_candidate() {
        let map = vec![WarpMiInfo::default(); 16 * 16];
        let bank: Vec<Vec<[i32; 6]>> = vec![Vec::new(); 8];
        let dist = [0i32; 7];
        let stack = find_mv_stack(&params(&map, 16, 16, [0, -1], false, &bank, &dist));
        assert_eq!(stack.count, 1);
        assert_eq!(stack.mvs[0][0], MvXY::default());
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
        left.sub_mv = left.mv;
        for y in 0..16 {
            for x in 0..8 {
                map[y * mi_cols + x] = left;
            }
        }
        let bank: Vec<Vec<[i32; 6]>> = vec![Vec::new(); 8];
        let dist = [0i32; 7];
        let stack = find_mv_stack(&params(
            &map,
            mi_rows,
            mi_cols,
            [0, -1],
            false,
            &bank,
            &dist,
        ));
        assert_eq!(
            stack.mvs[0][0].y as i32,
            clamp_mv_row(i16::MIN as i32, 8, 4, mi_rows)
        );
        assert_eq!(
            stack.mvs[0][0].x as i32,
            clamp_mv_col(i16::MAX as i32, 8, 4, mi_cols)
        );
    }

    #[test]
    fn test_compound_stack_prefers_exact_pair() {
        let mi_rows = 16usize;
        let mi_cols = 16usize;
        let mut map = vec![WarpMiInfo::default(); mi_rows * mi_cols];

        // Left neighbour with the exact compound pair.
        let mut left = WarpMiInfo::block(0, 0, 8, 16);
        left.inter = true;
        left.ref0 = 0;
        left.ref1 = 1;
        left.mv = MvXY { x: 8, y: 0 };
        left.mv1 = MvXY { x: 0, y: 8 };
        left.sub_mv = left.mv;
        left.sub_mv1 = left.mv1;
        for y in 0..16 {
            for x in 0..8 {
                map[y * mi_cols + x] = left;
            }
        }
        // Above neighbour with a single reference: it seeds the derived list.
        let mut above = WarpMiInfo::block(8, 0, 8, 8);
        above.inter = true;
        above.ref0 = 0;
        above.mv = MvXY { x: 16, y: 16 };
        above.sub_mv = above.mv;
        for y in 0..8 {
            for x in 8..16 {
                map[y * mi_cols + x] = above;
            }
        }

        let bank: Vec<Vec<[i32; 6]>> = vec![Vec::new(); 8];
        let dist = [4i32, -4, 0, 0, 0, 0, 0];
        let mut p = params(&map, mi_rows, mi_cols, [0, 1], true, &bank, &dist);
        p.derive_wrl = false;
        let stack = find_mv_stack(&p);
        assert!(stack.count >= 1);
        assert_eq!(stack.mvs[0][0], MvXY { x: 8, y: 0 });
        assert_eq!(stack.mvs[0][1], MvXY { x: 0, y: 8 });
    }

    #[test]
    fn test_refmv_bank_round_trip() {
        let mut bank = RefMvBank::default();
        let mvs = [MvXY { x: 8, y: 16 }, MvXY::default()];
        bank.update([2, -1], mvs, CWP_EQUAL);
        let idx = RefMvBank::list_index([2, -1]);
        assert_eq!(bank.size[idx], 1);
        assert_eq!(bank.params[idx][0][2], 16);
        assert_eq!(bank.params[idx][0][3], 8);
        assert_eq!(bank.params[idx][0][1], 2);
    }
}
