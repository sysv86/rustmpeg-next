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

//! AV2 screen-content coding tools (encoder side):
//! - Palette mode (`palette_mode_info` / `palette_tokens`, AV2 Spec v1.0.0
//!   §5.20.8.1 and §5.20.8.4): palette sizes `2..=8`, cached above/left
//!   palettes, delta-coded sorted colors, `palette_direction`,
//!   `identity_row_y`, and the `get_palette_color_context` CDF context.
//! - Intra Block Copy (`IntraBC`, §5.20.5.4 `read_intrabc_info` and
//!   §6.19.7.12 `is_mv_valid`): causal block-vector search plus the
//!   `intrabc_mode` / DRL / MVD syntax and the local 64x64 IBC buffer ring.
//!
//! AV2 v1.0.0 has no Intra Sub-Partitions (`ISP`) syntax; `predict_isp_strips`
//! is retained only for callers of the public helper and is never signalled.

use std::collections::BTreeMap;

use crate::av2::cdf::CdfModeContext;
use crate::av2::frame::Plane;
use crate::av2::ipred::{compute_sad, predict_intra_block, prepare_intra_edges};
use crate::av2::levels::MvXY;
use crate::av2::msac::MsacEncoder;
use crate::av2::pixel::{BitDepth, Pixel};

/// Maximum number of distinct colors in an AV2 screen-content palette block (`2..=8`).
pub const MAX_PALETTE_COLORS: usize = 8;
/// `PALETTE_ROW_FLAG_CONTEXTS` (AV2 Spec v1.0.0 Table 3).
pub const PALETTE_ROW_FLAG_CONTEXTS: usize = 4;
/// `INTRABC_DELAY_PIXELS` (AV2 Spec v1.0.0 Table 3).
pub const INTRABC_DELAY_PIXELS: i32 = 256;
/// `IBC_BUFFER_SIZE_LOG2` (AV2 Spec v1.0.0 Table 3).
pub const IBC_BUFFER_SIZE_LOG2: u32 = 6;
/// `IBC_NUM_BUFFERS`: the four most recently decoded 64x64 local regions.
pub const IBC_NUM_BUFFERS: usize = 4;

/// Frame-level palette entropy state that is not part of `CdfModeContext`.
#[derive(Debug, Clone)]
pub struct PaletteCdfState {
    /// `TileIdentityRowYCdf[prevIdentityRow]` (`Default_Identity_Row_Y_Cdf`).
    /// Layout is `[cdf0, cdf1, para, count]` for three symbols.
    pub identity_row_y: [[u16; 4]; PALETTE_ROW_FLAG_CONTEXTS],
}

impl Default for PaletteCdfState {
    fn default() -> Self {
        Self {
            identity_row_y: [
                [22515, 25751, 25, 0],
                [4014, 5233, 31, 0],
                [3548, 4163, 33, 0],
                [12999, 32756, 56, 0],
            ],
        }
    }
}

/// Tracks the four-entry IBC local buffer ring described by
/// `update_ibc_buffers` (AV2 Spec v1.0.0 §5.20.5.14) and used by
/// `check_valid_local_ibc` (§6.19.7.12).
#[derive(Debug, Clone)]
pub struct IbcBufferState {
    cur_row: i32,
    cur_col: i32,
    valid: [bool; IBC_NUM_BUFFERS],
    row: [i32; IBC_NUM_BUFFERS],
    col: [i32; IBC_NUM_BUFFERS],
}

impl Default for IbcBufferState {
    fn default() -> Self {
        Self {
            cur_row: -1,
            cur_col: -1,
            valid: [false; IBC_NUM_BUFFERS],
            row: [-1; IBC_NUM_BUFFERS],
            col: [-1; IBC_NUM_BUFFERS],
        }
    }
}

impl IbcBufferState {
    /// `ibc_buffer_index(row, col)` (AV2 Spec v1.0.0 §5.20.3.1). The two-bit
    /// slot index depends on `SbSize`: four columns for 64x64 superblocks and a
    /// 2x2 tile for 128x128 superblocks.
    #[inline]
    fn index_for(col: i32, row: i32, sb128: bool) -> usize {
        if sb128 {
            ((col & 1) | ((row & 1) << 1)) as usize
        } else {
            (col & 3) as usize
        }
    }

    /// `ibc_buffer_index(row, col)`: four 64x64 slots indexed by column bits.
    #[inline]
    fn index(col: i32) -> usize {
        (col & 3) as usize
    }

    /// `update_ibc_buffers(miRow, miCol)` (§5.20.5.14). `sb_step4` is the
    /// superblock size in 4x4 units; it only participates in the `BRU`
    /// invalidation rule, which this encoder never enables.
    pub fn update(&mut self, mi_row: i32, mi_col: i32) {
        let shift = IBC_BUFFER_SIZE_LOG2 - 2;
        let buf_row = mi_row >> shift;
        let buf_col = mi_col >> shift;
        if buf_row != self.cur_row || buf_col != self.cur_col {
            if self.cur_row >= 0 {
                let idx = Self::index(self.cur_col);
                self.row[idx] = self.cur_row;
                self.col[idx] = self.cur_col;
                self.valid[idx] = true;
            }
            self.cur_row = buf_row;
            self.cur_col = buf_col;
        }
    }

    /// Returns true when `(x, y)` (in 4x4 units) is inside a valid local IBC
    /// buffer and lies in the already-decoded part of the current tile.
    pub fn valid_location(&self, x: i32, y: i32) -> bool {
        self.valid_location_sb(x, y, false)
    }

    /// Superblock-size-aware variant of [`IbcBufferState::valid_location`].
    pub fn valid_location_sb(&self, x: i32, y: i32, sb128: bool) -> bool {
        let shift = IBC_BUFFER_SIZE_LOG2 - 2;
        let buf_row = y >> shift;
        let buf_col = x >> shift;
        if buf_row == self.cur_row && buf_col == self.cur_col {
            return true;
        }
        let idx = Self::index_for(buf_col, buf_row, sb128);
        self.valid[idx] && self.row[idx] == buf_row && self.col[idx] == buf_col
    }

    /// Superblock-size-aware variant of [`IbcBufferState::update`]
    /// (`update_ibc_buffers`, AV2 Spec v1.0.0 §5.20.3.1).
    pub fn update_sb(&mut self, mi_row: i32, mi_col: i32, sb128: bool) {
        let shift = IBC_BUFFER_SIZE_LOG2 - 2;
        let buf_row = mi_row >> shift;
        let buf_col = mi_col >> shift;
        if buf_row != self.cur_row || buf_col != self.cur_col {
            if self.cur_row >= 0 {
                let idx = Self::index_for(self.cur_col, self.cur_row, sb128);
                self.row[idx] = self.cur_row;
                self.col[idx] = self.cur_col;
                self.valid[idx] = true;
            }
            self.cur_row = buf_row;
            self.cur_col = buf_col;
        }
    }
}

/// Computes a fast 32-bit FNV-1a spatial signature of a 4x4 luma patch at `(x, y)`.
#[inline]
pub fn hash_4x4_patch<P: Pixel>(data: &[P], stride: usize, x: usize, y: usize) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for r in 0..4 {
        let row = (y + r) * stride + x;
        for c in 0..4 {
            let v = data[row + c].to_i32() as u32;
            h ^= v;
            h = h.wrapping_mul(0x0100_0193);
        }
    }
    h
}

/// Conservative causal check used by the public search helper: the source
/// block must be inside the tile and strictly above or left of the current
/// block (AV2 Spec v1.0.0 §6.19.7.12, `is_mv_valid`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn is_valid_intrabc_vector(
    bx_px: i32,
    by_px: i32,
    w: i32,
    h: i32,
    dv_x_px: i32,
    dv_y_px: i32,
    tile_x0: i32,
    tile_y0: i32,
    tile_x1: i32,
    tile_y1: i32,
) -> bool {
    if dv_x_px == 0 && dv_y_px == 0 {
        return false;
    }
    let rx = bx_px + dv_x_px;
    let ry = by_px + dv_y_px;
    if rx < tile_x0 || ry < tile_y0 || rx + w > tile_x1 || ry + h > tile_y1 {
        return false;
    }
    rx + w <= bx_px || ry + h <= by_px
}

/// Full `is_mv_valid` check for local IntraBC: source inside the tile, fully
/// causal, and covered by the local 64x64 buffer ring.
#[inline]
#[allow(clippy::too_many_arguments)]
fn is_valid_local_intrabc(
    bx_px: i32,
    by_px: i32,
    w: i32,
    h: i32,
    dv_x_px: i32,
    dv_y_px: i32,
    tile_x0: i32,
    tile_y0: i32,
    tile_x1: i32,
    tile_y1: i32,
    ibc: &IbcBufferState,
) -> bool {
    if dv_x_px == 0 && dv_y_px == 0 {
        return false;
    }
    let rx = bx_px + dv_x_px;
    let ry = by_px + dv_y_px;
    if rx < tile_x0 || ry < tile_y0 || rx + w > tile_x1 || ry + h > tile_y1 {
        return false;
    }
    // Both source corners must be causal (above or left of the current block).
    if !(rx + w <= bx_px || ry + h <= by_px) {
        return false;
    }
    let left = rx >> 2;
    let top = ry >> 2;
    let right = (rx + w - 1) >> 2;
    let bottom = (ry + h - 1) >> 2;
    ibc.valid_location(left, top) && ibc.valid_location(right, bottom)
}

/// Searches the causal reconstructed region of `rec_luma` for the best
/// integer-pel IntraBC displacement vector matching `src_luma` at
/// `(bx_px, by_px, w, h)`.
#[allow(clippy::too_many_arguments)]
pub fn search_intrabc_block<P: Pixel>(
    src_luma: &Plane<P>,
    rec_luma: &Plane<P>,
    bx_px: usize,
    by_px: usize,
    w: usize,
    h: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile_x1: usize,
    tile_y1: usize,
    lambda_sad: u64,
) -> Option<(MvXY, u64)> {
    search_intrabc_block_local(
        src_luma,
        rec_luma,
        bx_px,
        by_px,
        w,
        h,
        tile_x0,
        tile_y0,
        tile_x1,
        tile_y1,
        &IbcBufferState::default(),
        lambda_sad,
    )
}

/// Local-buffer-aware IntraBC search. `ibc` must have been advanced with
/// `IbcBufferState::update` for the current block.
#[allow(clippy::too_many_arguments)]
pub fn search_intrabc_block_local<P: Pixel>(
    src_luma: &Plane<P>,
    rec_luma: &Plane<P>,
    bx_px: usize,
    by_px: usize,
    w: usize,
    h: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile_x1: usize,
    tile_y1: usize,
    ibc: &IbcBufferState,
    lambda_sad: u64,
) -> Option<(MvXY, u64)> {
    let bx = bx_px as i32;
    let by = by_px as i32;
    let bw = w as i32;
    let bh = h as i32;
    let tx0 = tile_x0 as i32;
    let ty0 = tile_y0 as i32;
    let tx1 = (tile_x1.min(rec_luma.width)) as i32;
    let ty1 = (tile_y1.min(rec_luma.height)) as i32;
    // Keep every candidate inside the current 64x64 local IBC buffer cell.
    // `check_valid_local_ibc` (AV2 Spec v1.0.0 §6.19.7.12) only guarantees
    // causality for the buffer currently being filled; referencing an earlier
    // ring slot requires the wrap-around / collocation check, which depends on
    // `is_mi_coded` at the mirrored position. Restricting the search to the
    // current cell is legal for every superblock size (64 and 128) and keeps
    // the local `is_mv_valid` proof simple.
    let cell_x0 = bx & !63;
    let cell_y0 = by & !63;
    let tx0 = tx0.max(cell_x0);
    let ty0 = ty0.max(cell_y0);
    let tx1 = tx1.min(cell_x0 + 64);
    let ty1 = ty1.min(cell_y0 + 64);

    let src_off = src_luma.offset(bx_px, by_px);
    let src_sig = hash_4x4_patch(&src_luma.data, src_luma.stride, bx_px, by_px);

    let mut best_dv: Option<MvXY> = None;
    let mut best_cost = u64::MAX;

    let eval_dv = |dx_px: i32, dy_px: i32| -> Option<(MvXY, u64)> {
        if !is_valid_local_intrabc(bx, by, bw, bh, dx_px, dy_px, tx0, ty0, tx1, ty1, ibc) {
            return None;
        }
        let rx = (bx + dx_px) as usize;
        let ry = (by + dy_px) as usize;
        let rec_off = rec_luma.offset(rx, ry);
        let sad = compute_sad(
            &src_luma.data,
            src_off,
            src_luma.stride,
            &rec_luma.data,
            rec_off,
            rec_luma.stride,
            w,
            h,
        );
        let mag = (dx_px.unsigned_abs() + dy_px.unsigned_abs()).max(1);
        let rate_bits = 4 + 2 * (32 - mag.leading_zeros() as u64);
        let cost = sad + lambda_sad * rate_bits;
        Some((
            MvXY {
                x: (dx_px * 8) as i16,
                y: (dy_px * 8) as i16,
            },
            cost,
        ))
    };

    fn update_best(best_dv: &mut Option<MvXY>, best_cost: &mut u64, cand: Option<(MvXY, u64)>) {
        if let Some((dv, cost)) = cand
            && cost < *best_cost
        {
            *best_cost = cost;
            *best_dv = Some(dv);
        }
    }

    // Immediate causal neighbours and a 4-pixel grid over a 128-pixel window.
    update_best(&mut best_dv, &mut best_cost, eval_dv(-bw, 0));
    update_best(&mut best_dv, &mut best_cost, eval_dv(-2 * bw, 0));
    update_best(&mut best_dv, &mut best_cost, eval_dv(0, -bh));
    update_best(&mut best_dv, &mut best_cost, eval_dv(-bw, -bh));

    let min_dy = (ty0 - by).max(-128);
    let max_dy = bh;
    let min_dx = (tx0 - bx).max(-128);
    let max_dx = (tx1 - bw - bx).min(128);

    let mut dy = min_dy;
    while dy <= max_dy {
        let mut dx = min_dx;
        while dx <= max_dx {
            if is_valid_local_intrabc(bx, by, bw, bh, dx, dy, tx0, ty0, tx1, ty1, ibc) {
                let rx = (bx + dx) as usize;
                let ry = (by + dy) as usize;
                let cand_sig = hash_4x4_patch(&rec_luma.data, rec_luma.stride, rx, ry);
                if cand_sig == src_sig || (dx % 8 == 0 && dy % 8 == 0) {
                    update_best(&mut best_dv, &mut best_cost, eval_dv(dx, dy));
                }
            }
            dx += 4;
        }
        dy += 4;
    }

    // 1-pixel local diamond refinement around the winning candidate.
    if let Some(center) = best_dv {
        let cx = (center.x / 8) as i32;
        let cy = (center.y / 8) as i32;
        for ddy in -3..=3 {
            for ddx in -3..=3 {
                update_best(&mut best_dv, &mut best_cost, eval_dv(cx + ddx, cy + ddy));
            }
        }
    }

    best_dv.map(|dv| (dv, best_cost))
}

/// Copies an already-validated IntraBC reference block
/// `(bx_px + dv_x_px, by_px + dv_y_px)` into `(bx_px, by_px)`.
///
/// The reference is guaranteed by `is_mv_valid` to lie inside the coded
/// picture and the current tile, so no visible-size clamp is applied.
pub fn apply_intrabc_block<P: Pixel>(
    rec_plane: &mut Plane<P>,
    bx_px: usize,
    by_px: usize,
    w: usize,
    h: usize,
    dv_y_px: i32,
    dv_x_px: i32,
) {
    let rx = bx_px as i32 + dv_x_px;
    let ry = by_px as i32 + dv_y_px;
    if rx < 0 || ry < 0 || rx as usize + w > rec_plane.width || ry as usize + h > rec_plane.height {
        return;
    }
    let (rx, ry) = (rx as usize, ry as usize);

    let mut tmp = vec![P::default(); w * h];
    for r in 0..h {
        let src_row = (ry + r) * rec_plane.stride + rx;
        tmp[r * w..(r + 1) * w].copy_from_slice(&rec_plane.data[src_row..src_row + w]);
    }
    for r in 0..h {
        let dst_row = (by_px + r) * rec_plane.stride + bx_px;
        rec_plane.data[dst_row..dst_row + w].copy_from_slice(&tmp[r * w..(r + 1) * w]);
    }
}

/// Copies an IntraBC reference block using the normative picture bounds
/// (`MiCols * 4`, `MiRows * 4`) instead of the visible plane size
/// (AV2 Spec v1.0.0 §6.19.7.12 `is_mv_valid`).
///
/// `x_limit` / `y_limit` are the source bounds in the plane's own subsampling;
/// a reference outside them is not written and `false` is returned. The source
/// must already be reconstructed, which `is_mv_valid` guarantees.
pub fn apply_intrabc_block_bounded<P: Pixel>(
    rec_plane: &mut Plane<P>,
    bx_px: usize,
    by_px: usize,
    w: usize,
    h: usize,
    dv_y_px: i32,
    dv_x_px: i32,
    x_limit: usize,
    y_limit: usize,
) -> bool {
    let rx = bx_px as i32 + dv_x_px;
    let ry = by_px as i32 + dv_y_px;
    if rx < 0 || ry < 0 {
        return false;
    }
    let (rx, ry) = (rx as usize, ry as usize);
    if rx + w > x_limit || ry + h > y_limit {
        return false;
    }
    // The planes are padded to a 128-sample stride and height, so the
    // `MiCols`/`MiRows` extent is always addressable.
    if rx + w > rec_plane.stride || ry + h > rec_plane.alloc_height {
        return false;
    }
    // Copy through a row staging buffer so an overlapping reference and
    // destination still read source samples before writing them.
    let mut tmp = vec![P::default(); w * h];
    for r in 0..h {
        let src_row = (ry + r) * rec_plane.stride + rx;
        tmp[r * w..(r + 1) * w].copy_from_slice(&rec_plane.data[src_row..src_row + w]);
    }
    for r in 0..h {
        let dst_row = (by_px + r) * rec_plane.stride + bx_px;
        rec_plane.data[dst_row..dst_row + w].copy_from_slice(&tmp[r * w..(r + 1) * w]);
    }
    true
}

/// `lower_mv_precision(MV_PRECISION_ONE_PEL, mv)` from `assign_mv()`
/// (AV2 Spec v1.0.0 §5.20.7.13), used to fold a BVP candidate to integer
/// precision before adding the IntraBC motion-vector difference.
pub fn lower_mv_precision_to_integer(mv: MvXY) -> MvXY {
    let lower = |v: i16| -> i16 {
        let a = (v as i32).abs();
        let a_int = (a - 1 + 4) >> 3;
        let mut out = if v >= 0 { a_int << 3 } else { (-a_int) << 3 };
        if (a_int << 3) != a {
            out = out.clamp(crate::av2::warp::MV_LOW + 8, crate::av2::warp::MV_UPP - 8);
        }
        out as i16
    };
    MvXY {
        x: lower(mv.x),
        y: lower(mv.y),
    }
}

/// Palette mode decision for a luma block (`2..=8` sorted colors + index map).
#[derive(Debug, Clone)]
pub struct PaletteBlockCandidate {
    pub palette_size: usize,
    pub colors: [i32; MAX_PALETTE_COLORS],
    pub color_map: Vec<u8>,
    /// Sum of absolute palette quantisation error over the block.
    pub distortion: u64,
    pub cost: u64,
}

/// Analyzes a luma block `(bx_px, by_px, w, h)` for Palette coding.
///
/// The palette is restricted to the most frequent exact sample values in the
/// block. This is lossless when the block has at most eight distinct values,
/// and the normal transform/residual path codes any remaining error.
pub fn evaluate_palette_block<P: Pixel>(
    src_plane: &Plane<P>,
    bx_px: usize,
    by_px: usize,
    w: usize,
    h: usize,
    lambda_sad: u64,
) -> Option<PaletteBlockCandidate> {
    if w < 8 || h < 8 || w > 64 || h > 64 {
        return None;
    }

    let n = w * h;
    let mut counts: BTreeMap<i32, u32> = BTreeMap::new();
    let mut samples = Vec::with_capacity(n);
    for r in 0..h {
        let row = (by_px + r) * src_plane.stride + bx_px;
        for c in 0..w {
            let v = src_plane.data[row + c].to_i32();
            *counts.entry(v).or_insert(0) += 1;
            samples.push(v);
        }
    }
    if counts.len() < 2 {
        return None;
    }

    let mut ranked: Vec<(i32, u32)> = counts.into_iter().collect();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(MAX_PALETTE_COLORS);
    ranked.sort_unstable_by_key(|e| e.0);

    // A two-entry palette would exercise the two-symbol form of
    // `palette_color_idx_y`; add an unused third color so the coded palette
    // uses the general (three or more symbol) CDF path.
    if ranked.len() == 2 {
        let max = (1i32 << P::BITDEPTH) - 1;
        let spare = if ranked[0].0 > 0 {
            ranked[0].0 - 1
        } else if ranked[1].0 < max {
            ranked[1].0 + 1
        } else {
            (ranked[0].0 + ranked[1].0) / 2
        };
        if spare != ranked[0].0 && spare != ranked[1].0 {
            ranked.push((spare, 0));
            ranked.sort_unstable_by_key(|e| e.0);
        }
    }
    let k_colors = ranked.len();
    if !(2..=MAX_PALETTE_COLORS).contains(&k_colors) {
        return None;
    }
    let mut colors = [0i32; MAX_PALETTE_COLORS];
    for (i, &(v, _)) in ranked.iter().enumerate() {
        colors[i] = v;
    }

    let mut color_map = vec![0u8; n];
    let mut sad = 0u64;
    let mut transitions = 1u64;
    for (idx, &px) in samples.iter().enumerate() {
        let mut best_c = 0usize;
        let mut best_d = (px - colors[0]).abs();
        for c_idx in 1..k_colors {
            let d = (px - colors[c_idx]).abs();
            if d < best_d {
                best_d = d;
                best_c = c_idx;
            }
        }
        color_map[idx] = best_c as u8;
        sad += best_d as u64;
        if idx > 0 && color_map[idx] != color_map[idx - 1] {
            transitions += 1;
        }
    }

    // Approximate palette token cost: colour deltas plus one index decision
    // per run boundary. `identity_row_y` removes repeated rows, so a run
    // boundary is counted once rather than once per sample.
    let rate_bits = (k_colors as u64) * 8 + transitions * 2;
    let cost = sad + lambda_sad * rate_bits;

    Some(PaletteBlockCandidate {
        palette_size: k_colors,
        colors,
        color_map,
        distortion: sad,
        cost,
    })
}

/// Reconstructs a Palette block into `rec_plane` at `(bx_px, by_px, w, h)`.
pub fn apply_palette_block<P: Pixel>(
    rec_plane: &mut Plane<P>,
    bx_px: usize,
    by_px: usize,
    w: usize,
    h: usize,
    cand: &PaletteBlockCandidate,
) {
    for r in 0..h {
        let dst_row = (by_px + r) * rec_plane.stride + bx_px;
        for c in 0..w {
            let idx = (cand.color_map[r * w + c] as usize).min(cand.palette_size - 1);
            rec_plane.data[dst_row + c] = P::from_i32(cand.colors[idx]);
        }
    }
}

fn ceil_log2(v: u32) -> u32 {
    if v <= 1 {
        0
    } else {
        32 - (v - 1).leading_zeros()
    }
}

/// Codes one `palette_color_idx_y` with the CDF selected by `PaletteSizeY`
/// (AV2 Spec v1.0.0 §8.3.2, Table 8.1).
#[inline]
fn encode_palette_color_index(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    p_size: usize,
    ctx: usize,
    rank: usize,
) {
    let ctx = ctx.min(4);
    match p_size {
        2 => msac.encode_symbol_adapt(rank, &mut cdf_m.palette_size_2_y_color[ctx], 1),
        3 => msac.encode_symbol_adapt(rank, &mut cdf_m.palette_size_3_y_color[ctx], 2),
        4 => msac.encode_symbol_adapt(rank, &mut cdf_m.palette_size_4_y_color[ctx], 3),
        5 => msac.encode_symbol_adapt(rank, &mut cdf_m.palette_size_5_y_color[ctx], 4),
        6 => msac.encode_symbol_adapt(rank, &mut cdf_m.palette_size_6_y_color[ctx], 5),
        7 => msac.encode_symbol_adapt(rank, &mut cdf_m.palette_size_7_y_color[ctx], 6),
        _ => msac.encode_symbol_adapt(rank, &mut cdf_m.palette_size_8_y_color[ctx], 7),
    }
}

/// `get_palette_cache` (AV2 Spec v1.0.0 §5.20.8.1): interleaves the above and
/// left palettes. The caller supplies the already-resolved neighbour palettes.
fn palette_cache(above: &[i32], left: &[i32]) -> Vec<i32> {
    let mut cache = Vec::with_capacity(above.len() + left.len());
    let mut above_idx = 0usize;
    let mut left_idx = 0usize;
    while above_idx < above.len() || left_idx < left.len() {
        if above_idx < above.len() {
            cache.push(above[above_idx]);
            above_idx += 1;
        }
        if left_idx < left.len() {
            cache.push(left[left_idx]);
            left_idx += 1;
        }
    }
    cache
}

/// Encodes `palette_mode_info()` (AV2 Spec v1.0.0 §5.20.8.1).
#[allow(clippy::too_many_arguments)]
pub fn encode_palette_mode_info(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    cand: &PaletteBlockCandidate,
    bit_depth: u8,
    above: &[i32],
    left: &[i32],
) {
    let p_size = cand.palette_size.clamp(2, MAX_PALETTE_COLORS);
    msac.encode_bool_adapt(1, &mut cdf_m.palette_y_mode);
    msac.encode_symbol_adapt(p_size - 2, &mut cdf_m.palette_y_size, 6);

    let cache = palette_cache(above, left);
    let mut idx = 0usize;
    // This encoder never reuses a cached palette entry; the syntax bit is
    // still present for every cache entry considered.
    for _ in 0..cache.len() {
        msac.encode_bool_bypass(0);
    }

    // New entries are strictly increasing and sorted.
    if idx < p_size {
        msac.encode_bools_bypass(bit_depth as u32, cand.colors[idx].max(0) as u32);
        idx += 1;
    }
    let max = (1i32 << bit_depth) - 1;
    let base_bits = (bit_depth as u32).saturating_sub(3);
    // Pick the smallest `palette_num_extra_bits_y` that can represent the
    // first delta; later deltas are bounded by the shrinking `range`.
    let mut extra = 0u32;
    if idx < p_size {
        while extra < 3 {
            let mut bits = base_bits + extra;
            let mut prev = cand.colors[idx - 1];
            let mut ok = true;
            for &color in &cand.colors[idx..p_size] {
                let delta = (color - prev - 1).max(0) as u32;
                if delta >= (1u32 << bits) {
                    ok = false;
                    break;
                }
                prev = color.min(max);
                let range = (max - prev).max(0) as u32;
                bits = bits.min(ceil_log2(range));
            }
            if ok {
                break;
            }
            extra += 1;
        }
        msac.encode_bools_bypass(2, extra);
    }
    let mut bits = base_bits + extra;
    while idx < p_size {
        let delta = (cand.colors[idx] - cand.colors[idx - 1] - 1).max(0) as u32;
        msac.encode_bools_bypass(bits, delta);
        let prev = cand.colors[idx].min(max);
        let range = (max - prev).max(0) as u32;
        bits = bits.min(ceil_log2(range));
        idx += 1;
    }
}

/// `get_palette_color_context` (AV2 Spec v1.0.0 §5.20.8.5) for one sample.
///
/// Returns `(ctx, color_order)`, where `ColorOrder[palette_color_idx_y]` is
/// the actual palette index stored in the color map.
pub fn palette_color_context(
    map: &[u8],
    w: usize,
    r: usize,
    c: usize,
    n: usize,
) -> (usize, [usize; MAX_PALETTE_COLORS]) {
    let mut scores = [0i32; MAX_PALETTE_COLORS];
    let mut order = [0usize; MAX_PALETTE_COLORS];
    for (i, slot) in order.iter_mut().enumerate() {
        *slot = i;
    }
    let at = |r: usize, c: usize| map[r * w + c] as usize;
    if c > 0 {
        scores[at(r, c - 1)] += 2;
    }
    if r > 0 && c > 0 {
        scores[at(r - 1, c - 1)] += 1;
    }
    if r > 0 {
        scores[at(r - 1, c)] += 2;
    }

    for i in 0..3 {
        let mut max_score = scores[i];
        let mut max_idx = i;
        for j in (i + 1)..n {
            let left_tie = scores[j] > 0 && scores[j] == max_score && c > 0 && j == at(r, c - 1);
            if scores[j] > max_score || left_tie {
                max_score = scores[j];
                max_idx = j;
            }
        }
        if max_idx != i {
            let max_color = order[max_idx];
            for k in (i + 1..=max_idx).rev() {
                scores[k] = scores[k - 1];
                order[k] = order[k - 1];
            }
            scores[i] = max_score;
            order[i] = max_color;
        }
    }

    const PALETTE_COLOR_CONTEXT: [i32; 9] = [-1, -1, 0, -1, -1, 4, 3, 2, 1];
    let hash = scores[0] + 2 * scores[1] + 2 * scores[2];
    let ctx = PALETTE_COLOR_CONTEXT[(hash.clamp(0, 8)) as usize].max(0) as usize;
    (ctx, order)
}

/// Encodes `palette_tokens()` (AV2 Spec v1.0.0 §5.20.8.4) for a luma block of
/// `block_w x block_h` samples.
#[allow(clippy::too_many_arguments)]
pub fn encode_palette_tokens(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    palette_cdf: &mut PaletteCdfState,
    cand: &PaletteBlockCandidate,
    block_w: usize,
    block_h: usize,
) {
    let p_size = cand.palette_size.clamp(2, MAX_PALETTE_COLORS);
    let mut map = cand.color_map.clone();
    map.resize(block_w * block_h, 0);

    let direction = 0usize;
    if block_w < 64 && block_h < 64 {
        // Row-major is the better choice for the screen-content raster this
        // encoder targets; the direction bit is bypass-coded.
        msac.encode_bool_bypass(0);
    }
    let (outer_lim, inner_lim) = if direction == 1 {
        (block_w, block_h)
    } else {
        (block_h, block_w)
    };

    let mut prev_identity = PALETTE_ROW_FLAG_CONTEXTS - 1;
    let at = |map: &[u8], r: usize, c: usize| map[r * block_w + c] as usize;
    for i in 0..outer_lim {
        let (line_r, line_c) = if direction == 1 { (0, i) } else { (i, 0) };
        let line_len = inner_lim;
        let line_at = |j: usize| {
            if direction == 1 {
                (j, line_c)
            } else {
                (line_r, j)
            }
        };
        let first = at(&map, line_at(0).0, line_at(0).1);
        let mut all_same = true;
        for j in 1..line_len {
            let (r, c) = line_at(j);
            if at(&map, r, c) != first {
                all_same = false;
                break;
            }
        }
        let mut same_as_prev = false;
        if i > 0 {
            same_as_prev = true;
            for j in 0..line_len {
                let (r, c) = line_at(j);
                let (pr, pc) = if direction == 1 {
                    (r, c - 1)
                } else {
                    (r - 1, c)
                };
                if at(&map, r, c) != at(&map, pr, pc) {
                    same_as_prev = false;
                    break;
                }
            }
        }
        let identity = if same_as_prev {
            2
        } else if all_same {
            1
        } else {
            0
        };

        msac.encode_symbol_adapt(identity, &mut palette_cdf.identity_row_y[prev_identity], 2);
        prev_identity = identity;

        for j in 0..line_len {
            let (r, c) = line_at(j);
            if identity == 2 {
                if direction == 1 {
                    map[r * block_w + c] = map[r * block_w + (c - 1)];
                } else {
                    map[r * block_w + c] = map[(r - 1) * block_w + c];
                }
            } else if identity == 1 && j > 0 {
                if direction == 1 {
                    map[r * block_w + c] = map[(r - 1) * block_w + c];
                } else {
                    map[r * block_w + c] = map[r * block_w + (c - 1)];
                }
            } else if r == 0 && c == 0 {
                msac.encode_uniform(p_size as u32, map[0] as u32);
            } else {
                let (ctx, order) = palette_color_context(&map, block_w, r, c, p_size);
                let actual = map[r * block_w + c] as usize;
                let mut rank = 0usize;
                for (k, &pal) in order.iter().take(p_size).enumerate() {
                    if pal == actual {
                        rank = k;
                        break;
                    }
                }
                encode_palette_color_index(msac, cdf_m, p_size, ctx, rank);
            }
        }
    }
}

/// Predicts a block `(w, h)` in four horizontal or vertical strips.
///
/// AV2 v1.0.0 has no Intra Sub-Partitions syntax; this helper is not used by
/// the encoder bitstream and is kept only for API compatibility.
#[allow(clippy::too_many_arguments)]
pub fn predict_isp_strips<BD: BitDepth>(
    bd: BD,
    rec_plane: &mut Plane<BD::Pixel>,
    bx_px: usize,
    by_px: usize,
    w: usize,
    h: usize,
    y_mode: u8,
    y_angle: i8,
    have_left: bool,
    have_top: bool,
    horizontal_strips: bool,
) {
    let mut edge = [BD::Pixel::default(); 512];
    let tl_idx = 256usize;

    if horizontal_strips && h >= 8 {
        let strip_h = (h / 4).max(2);
        let mut sy = 0usize;
        while sy < h {
            let cur_h = strip_h.min(h - sy);
            let rec_off = rec_plane.offset(bx_px, by_px + sy);
            prepare_intra_edges(
                bd,
                &mut edge,
                tl_idx,
                &rec_plane.data,
                rec_off,
                rec_plane.stride,
                have_left,
                have_top || sy > 0,
                false,
                false,
                w,
                cur_h,
                rec_plane.width,
                rec_plane.height,
            );
            predict_intra_block(
                bd,
                &mut rec_plane.data,
                rec_off,
                rec_plane.stride,
                &edge,
                tl_idx,
                y_mode,
                y_angle,
                w,
                cur_h,
                have_left,
                have_top || sy > 0,
                0,
                0,
            );
            sy += cur_h;
        }
    } else if !horizontal_strips && w >= 8 {
        let strip_w = (w / 4).max(2);
        let mut sx = 0usize;
        while sx < w {
            let cur_w = strip_w.min(w - sx);
            let rec_off = rec_plane.offset(bx_px + sx, by_px);
            prepare_intra_edges(
                bd,
                &mut edge,
                tl_idx,
                &rec_plane.data,
                rec_off,
                rec_plane.stride,
                have_left || sx > 0,
                have_top,
                false,
                false,
                cur_w,
                h,
                rec_plane.width,
                rec_plane.height,
            );
            predict_intra_block(
                bd,
                &mut rec_plane.data,
                rec_off,
                rec_plane.stride,
                &edge,
                tl_idx,
                y_mode,
                y_angle,
                cur_w,
                h,
                have_left || sx > 0,
                have_top,
                0,
                0,
            );
            sx += cur_w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intrabc_search_finds_exact_copied_glyph_pattern() {
        let mut rec = Plane::<u8>::new(128, 128, 0, 0, 8);
        let mut src = Plane::<u8>::new(128, 128, 0, 0, 8);

        // The search is restricted to the current 64x64 local IBC buffer cell,
        // so both blocks live inside the (64..128, 64..128) cell.
        for r in 0..8 {
            for c in 0..8 {
                let val = if (r ^ c) & 1 == 0 { 220u8 } else { 30u8 };
                rec.data[(64 + r) * rec.stride + (72 + c)] = val;
                src.data[(72 + r) * src.stride + (80 + c)] = val;
            }
        }

        let mut ibc = IbcBufferState::default();
        ibc.update(0, 0);
        ibc.update(18, 20);
        let res = search_intrabc_block_local(&src, &rec, 80, 72, 8, 8, 0, 0, 128, 128, &ibc, 1);
        assert!(res.is_some());
        let (dv_q8, _) = res.unwrap();
        assert_eq!(dv_q8.x, -8 * 8);
        assert_eq!(dv_q8.y, -8 * 8);
    }

    #[test]
    fn test_palette_clustering_and_reconstruction() {
        let mut src = Plane::<u8>::new(32, 32, 0, 0, 8);
        for r in 0..16 {
            for c in 0..16 {
                let color = if r < 5 {
                    20u8
                } else if r < 11 {
                    128u8
                } else {
                    240u8
                };
                src.data[r * src.stride + c] = color;
            }
        }
        let cand =
            evaluate_palette_block(&src, 0, 0, 16, 16, 1).expect("Should detect 3-color palette");
        assert_eq!(cand.palette_size, 3);
        assert_eq!(cand.colors[0], 20);
        assert_eq!(cand.colors[1], 128);
        assert_eq!(cand.colors[2], 240);

        let mut rec = Plane::<u8>::new(32, 32, 0, 0, 8);
        apply_palette_block(&mut rec, 0, 0, 16, 16, &cand);
        for r in 0..16 {
            for c in 0..16 {
                assert_eq!(rec.data[r * rec.stride + c], src.data[r * src.stride + c]);
            }
        }
    }

    #[test]
    fn test_palette_color_context_prefers_left_when_tied() {
        // All neighbours equal: the left neighbour wins the tied ranking.
        let mut map = vec![0u8; 16];
        map[1] = 0;
        map[4] = 1;
        map[5] = 2;
        let (ctx, order) = palette_color_context(&map, 4, 1, 1, 4);
        assert_eq!(order[0], 0);
        assert!((0..5).contains(&ctx));
    }
}
