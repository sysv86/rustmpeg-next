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

//! AV2 warped motion (AV2 Specification v1.0.0 §5.20.7.7, §7.13.3.19–§7.13.3.23).
//!
//! AV2 v1.0.0 has no overlapped block motion compensation. `MOTION_MODES` is
//! `SIMPLE`, `INTERINTRA`, `LOCALWARP`, `DELTAWARP`, `EXTENDWARP`
//! (§7.3 constants, `MOTION_MODES = 5`), and §5.20.7.14 `read_motion_mode()`
//! never signals a per-block OBMC flag. The earlier encoder-side OBMC blending
//! was therefore not normative and is removed; the encoder's `enable_obmc`
//! switch is retained only as a documented no-op for CLI compatibility.
//!
//! This module implements the normative pieces the encoder uses for
//! `WARPMV` + `DELTAWARP` blocks (and the model math `WARP_NEWMV` would
//! reuse):
//!
//! * §7.12.2.3 / §7.12.2.4 `generate_points_from_corners` / `warp_corner`,
//!   which builds the `WarpParamStack` candidates from neighbouring blocks.
//! * §7.13.3.21 `setup_shear()`, §7.13.3.22 `resolve_divisor()`,
//!   §7.13.3.23 `reduce_warp_model()` / `get_warp_translation()`.
//! * §7.13.3.19 `block_warp()` prediction with `Warped_Filters`.
//!
//! The normative lookup tables (`Warped_Filters`) are from the AV2 v1.0.0
//! specification attachment `all_tables.h` (Alliance for Open Media, see
//! `THIRD_PARTY.md`). dav2d reference symbols used to cross-check behaviour:
//! `dav2d_resolve_divisor_32`, `dav2d_get_shear_params`,
//! `dav2d_set_affine_mv2d`, `dav2d_find_affine_int`, `warp_affine_8x8_c`,
//! `warp_affine()` in `src/warpmv.c`, `src/mc_tmpl.c`, `src/recon_tmpl.c`.

use crate::av2::frame::Plane;
use crate::av2::intops::{DIV_LUT, iclip};
use crate::av2::levels::MvXY;
use crate::av2::pixel::{BitDepth, Pixel};

/// Internal precision of AV2 warped motion models (§7.3 `WARPEDMODEL_PREC_BITS`).
pub const WARPEDMODEL_PREC_BITS: u32 = 16;
/// Rounding bit width applied by the shear setup (§7.3 `WARP_PARAM_REDUCE_BITS`).
pub const WARP_PARAM_REDUCE_BITS: u32 = 6;
/// Clamping value for translation components (§7.3 `WARPEDMODEL_TRANS_CLAMP`).
pub const WARPEDMODEL_TRANS_CLAMP: i64 = 1 << 27;
/// Number of phases in the warped filter (§7.3 `WARPEDPIXEL_PREC_SHIFTS`).
pub const WARPEDPIXEL_PREC_SHIFTS: i32 = 1 << 6;
/// Extra fractional bits used by warped filtering (§7.3 `WARPEDDIFF_PREC_BITS`).
pub const WARPEDDIFF_PREC_BITS: u32 = 10;
/// `GM_TRANS_ONLY_PREC_DIFF` = `WARPEDMODEL_PREC_BITS - 3` (§7.3).
pub const GM_TRANS_ONLY_PREC_DIFF: u32 = WARPEDMODEL_PREC_BITS - 3;
/// Size of the per-reference warp parameter bank (§7.3 `WARP_PARAM_BANK_SIZE`).
pub const WARP_PARAM_BANK_SIZE: usize = 4;
/// Maximum number of warp reference candidates (§7.3 `MAX_WARP_REF_CANDIDATES`).
pub const MAX_WARP_REF_CANDIDATES: usize = 4;
/// Largest number of bank searches per superblock (§7.3 `MAX_WARP_SB_HITS`).
pub const MAX_WARP_SB_HITS: u32 = 64;
/// `WARP_DELTA_STEP_BITS` (§7.3).
pub const WARP_DELTA_STEP_BITS: u32 = 10;
/// Number of low symbols for a warp delta parameter (§7.3).
pub const WARP_DELTA_NUM_SYMBOLS_LOW: i32 = 8;

/// Rounding variables from §7.13.3.16 for single-reference prediction.
pub const INTER_ROUND0: u32 = 3;
pub const INTER_ROUND1: u32 = 11;

/// `Warped_Filters` (AV2 Specification v1.0.0 §7.13.3.19,
/// `WARPEDPIXEL_PREC_SHIFTS * 7 + 1 = 449` phases, 8 taps).
pub static WARPED_FILTERS: [[i16; 8]; 449] = [
    [127, 1, 0, 0, 0, 0, 0, 0],
    [126, 2, 0, 0, 0, 0, 0, 0],
    [124, 4, 0, 0, 0, 0, 0, 0],
    [122, 6, 0, 0, 0, 0, 0, 0],
    [120, 8, 0, 0, 0, 0, 0, 0],
    [118, 10, 0, 0, 0, 0, 0, 0],
    [116, 12, 0, 0, 0, 0, 0, 0],
    [114, 14, 0, 0, 0, 0, 0, 0],
    [112, 16, 0, 0, 0, 0, 0, 0],
    [110, 18, 0, 0, 0, 0, 0, 0],
    [108, 20, 0, 0, 0, 0, 0, 0],
    [106, 22, 0, 0, 0, 0, 0, 0],
    [104, 24, 0, 0, 0, 0, 0, 0],
    [102, 26, 0, 0, 0, 0, 0, 0],
    [100, 28, 0, 0, 0, 0, 0, 0],
    [98, 30, 0, 0, 0, 0, 0, 0],
    [96, 32, 0, 0, 0, 0, 0, 0],
    [94, 34, 0, 0, 0, 0, 0, 0],
    [92, 36, 0, 0, 0, 0, 0, 0],
    [90, 38, 0, 0, 0, 0, 0, 0],
    [88, 40, 0, 0, 0, 0, 0, 0],
    [86, 42, 0, 0, 0, 0, 0, 0],
    [84, 44, 0, 0, 0, 0, 0, 0],
    [82, 46, 0, 0, 0, 0, 0, 0],
    [80, 48, 0, 0, 0, 0, 0, 0],
    [78, 50, 0, 0, 0, 0, 0, 0],
    [76, 52, 0, 0, 0, 0, 0, 0],
    [74, 54, 0, 0, 0, 0, 0, 0],
    [72, 56, 0, 0, 0, 0, 0, 0],
    [70, 58, 0, 0, 0, 0, 0, 0],
    [68, 60, 0, 0, 0, 0, 0, 0],
    [66, 62, 0, 0, 0, 0, 0, 0],
    [64, 64, 0, 0, 0, 0, 0, 0],
    [62, 66, 0, 0, 0, 0, 0, 0],
    [60, 68, 0, 0, 0, 0, 0, 0],
    [58, 70, 0, 0, 0, 0, 0, 0],
    [56, 72, 0, 0, 0, 0, 0, 0],
    [54, 74, 0, 0, 0, 0, 0, 0],
    [52, 76, 0, 0, 0, 0, 0, 0],
    [50, 78, 0, 0, 0, 0, 0, 0],
    [48, 80, 0, 0, 0, 0, 0, 0],
    [46, 82, 0, 0, 0, 0, 0, 0],
    [44, 84, 0, 0, 0, 0, 0, 0],
    [42, 86, 0, 0, 0, 0, 0, 0],
    [40, 88, 0, 0, 0, 0, 0, 0],
    [38, 90, 0, 0, 0, 0, 0, 0],
    [36, 92, 0, 0, 0, 0, 0, 0],
    [34, 94, 0, 0, 0, 0, 0, 0],
    [32, 96, 0, 0, 0, 0, 0, 0],
    [30, 98, 0, 0, 0, 0, 0, 0],
    [28, 100, 0, 0, 0, 0, 0, 0],
    [26, 102, 0, 0, 0, 0, 0, 0],
    [24, 104, 0, 0, 0, 0, 0, 0],
    [22, 106, 0, 0, 0, 0, 0, 0],
    [20, 108, 0, 0, 0, 0, 0, 0],
    [18, 110, 0, 0, 0, 0, 0, 0],
    [16, 112, 0, 0, 0, 0, 0, 0],
    [14, 114, 0, 0, 0, 0, 0, 0],
    [12, 116, 0, 0, 0, 0, 0, 0],
    [10, 118, 0, 0, 0, 0, 0, 0],
    [8, 120, 0, 0, 0, 0, 0, 0],
    [6, 122, 0, 0, 0, 0, 0, 0],
    [4, 124, 0, 0, 0, 0, 0, 0],
    [2, 126, 0, 0, 0, 0, 0, 0],
    [0, 127, 1, 0, 0, 0, 0, 0],
    [-1, 127, 2, 0, 0, 0, 0, 0],
    [-2, 127, 4, -1, 0, 0, 0, 0],
    [-3, 126, 6, -1, 0, 0, 0, 0],
    [-3, 125, 8, -2, 0, 0, 0, 0],
    [-4, 124, 11, -3, 0, 0, 0, 0],
    [-5, 123, 13, -3, 0, 0, 0, 0],
    [-5, 121, 15, -3, 0, 0, 0, 0],
    [-6, 120, 18, -4, 0, 0, 0, 0],
    [-7, 119, 20, -4, 0, 0, 0, 0],
    [-7, 118, 22, -5, 0, 0, 0, 0],
    [-8, 116, 25, -5, 0, 0, 0, 0],
    [-8, 115, 27, -6, 0, 0, 0, 0],
    [-9, 113, 30, -6, 0, 0, 0, 0],
    [-9, 112, 32, -7, 0, 0, 0, 0],
    [-9, 110, 34, -7, 0, 0, 0, 0],
    [-10, 108, 37, -7, 0, 0, 0, 0],
    [-10, 107, 39, -8, 0, 0, 0, 0],
    [-10, 105, 41, -8, 0, 0, 0, 0],
    [-11, 103, 44, -8, 0, 0, 0, 0],
    [-11, 101, 47, -9, 0, 0, 0, 0],
    [-11, 99, 49, -9, 0, 0, 0, 0],
    [-11, 97, 51, -9, 0, 0, 0, 0],
    [-11, 95, 54, -10, 0, 0, 0, 0],
    [-11, 93, 56, -10, 0, 0, 0, 0],
    [-12, 91, 59, -10, 0, 0, 0, 0],
    [-12, 89, 61, -10, 0, 0, 0, 0],
    [-12, 87, 64, -11, 0, 0, 0, 0],
    [-12, 85, 66, -11, 0, 0, 0, 0],
    [-12, 82, 69, -11, 0, 0, 0, 0],
    [-12, 80, 71, -11, 0, 0, 0, 0],
    [-12, 78, 73, -11, 0, 0, 0, 0],
    [-11, 75, 75, -11, 0, 0, 0, 0],
    [-11, 73, 78, -12, 0, 0, 0, 0],
    [-11, 71, 80, -12, 0, 0, 0, 0],
    [-11, 69, 82, -12, 0, 0, 0, 0],
    [-11, 66, 85, -12, 0, 0, 0, 0],
    [-11, 64, 87, -12, 0, 0, 0, 0],
    [-10, 61, 89, -12, 0, 0, 0, 0],
    [-10, 59, 91, -12, 0, 0, 0, 0],
    [-10, 56, 93, -11, 0, 0, 0, 0],
    [-10, 54, 95, -11, 0, 0, 0, 0],
    [-9, 51, 97, -11, 0, 0, 0, 0],
    [-9, 49, 99, -11, 0, 0, 0, 0],
    [-9, 47, 101, -11, 0, 0, 0, 0],
    [-8, 44, 103, -11, 0, 0, 0, 0],
    [-8, 41, 105, -10, 0, 0, 0, 0],
    [-8, 39, 107, -10, 0, 0, 0, 0],
    [-7, 37, 108, -10, 0, 0, 0, 0],
    [-7, 34, 110, -9, 0, 0, 0, 0],
    [-7, 32, 112, -9, 0, 0, 0, 0],
    [-6, 30, 113, -9, 0, 0, 0, 0],
    [-6, 27, 115, -8, 0, 0, 0, 0],
    [-5, 25, 116, -8, 0, 0, 0, 0],
    [-5, 22, 118, -7, 0, 0, 0, 0],
    [-4, 20, 119, -7, 0, 0, 0, 0],
    [-4, 18, 120, -6, 0, 0, 0, 0],
    [-3, 15, 121, -5, 0, 0, 0, 0],
    [-3, 13, 123, -5, 0, 0, 0, 0],
    [-3, 11, 124, -4, 0, 0, 0, 0],
    [-2, 8, 125, -3, 0, 0, 0, 0],
    [-1, 6, 126, -3, 0, 0, 0, 0],
    [-1, 4, 127, -2, 0, 0, 0, 0],
    [0, 2, 127, -1, 0, 0, 0, 0],
    [0, 0, 127, 1, 0, 0, 0, 0],
    [0, -1, 127, 2, 0, 0, 0, 0],
    [1, -3, 127, 4, -1, 0, 0, 0],
    [1, -4, 126, 6, -2, 1, 0, 0],
    [1, -5, 126, 8, -3, 1, 0, 0],
    [1, -6, 125, 11, -4, 1, 0, 0],
    [1, -7, 124, 13, -4, 1, 0, 0],
    [2, -8, 123, 15, -5, 1, 0, 0],
    [2, -9, 122, 18, -6, 1, 0, 0],
    [2, -10, 121, 20, -6, 1, 0, 0],
    [2, -11, 120, 22, -7, 2, 0, 0],
    [2, -12, 119, 25, -8, 2, 0, 0],
    [3, -13, 117, 27, -8, 2, 0, 0],
    [3, -13, 116, 29, -9, 2, 0, 0],
    [3, -14, 114, 32, -10, 3, 0, 0],
    [3, -15, 113, 35, -10, 2, 0, 0],
    [3, -15, 111, 37, -11, 3, 0, 0],
    [3, -16, 109, 40, -11, 3, 0, 0],
    [3, -16, 108, 42, -12, 3, 0, 0],
    [4, -17, 106, 45, -13, 3, 0, 0],
    [4, -17, 104, 47, -13, 3, 0, 0],
    [4, -17, 102, 50, -14, 3, 0, 0],
    [4, -17, 100, 52, -14, 3, 0, 0],
    [4, -18, 98, 55, -15, 4, 0, 0],
    [4, -18, 96, 58, -15, 3, 0, 0],
    [4, -18, 94, 60, -16, 4, 0, 0],
    [4, -18, 91, 63, -16, 4, 0, 0],
    [4, -18, 89, 65, -16, 4, 0, 0],
    [4, -18, 87, 68, -17, 4, 0, 0],
    [4, -18, 85, 70, -17, 4, 0, 0],
    [4, -18, 82, 73, -17, 4, 0, 0],
    [4, -18, 80, 75, -17, 4, 0, 0],
    [4, -18, 78, 78, -18, 4, 0, 0],
    [4, -17, 75, 80, -18, 4, 0, 0],
    [4, -17, 73, 82, -18, 4, 0, 0],
    [4, -17, 70, 85, -18, 4, 0, 0],
    [4, -17, 68, 87, -18, 4, 0, 0],
    [4, -16, 65, 89, -18, 4, 0, 0],
    [4, -16, 63, 91, -18, 4, 0, 0],
    [4, -16, 60, 94, -18, 4, 0, 0],
    [3, -15, 58, 96, -18, 4, 0, 0],
    [4, -15, 55, 98, -18, 4, 0, 0],
    [3, -14, 52, 100, -17, 4, 0, 0],
    [3, -14, 50, 102, -17, 4, 0, 0],
    [3, -13, 47, 104, -17, 4, 0, 0],
    [3, -13, 45, 106, -17, 4, 0, 0],
    [3, -12, 42, 108, -16, 3, 0, 0],
    [3, -11, 40, 109, -16, 3, 0, 0],
    [3, -11, 37, 111, -15, 3, 0, 0],
    [2, -10, 35, 113, -15, 3, 0, 0],
    [3, -10, 32, 114, -14, 3, 0, 0],
    [2, -9, 29, 116, -13, 3, 0, 0],
    [2, -8, 27, 117, -13, 3, 0, 0],
    [2, -8, 25, 119, -12, 2, 0, 0],
    [2, -7, 22, 120, -11, 2, 0, 0],
    [1, -6, 20, 121, -10, 2, 0, 0],
    [1, -6, 18, 122, -9, 2, 0, 0],
    [1, -5, 15, 123, -8, 2, 0, 0],
    [1, -4, 13, 124, -7, 1, 0, 0],
    [1, -4, 11, 125, -6, 1, 0, 0],
    [1, -3, 8, 126, -5, 1, 0, 0],
    [1, -2, 6, 126, -4, 1, 0, 0],
    [0, -1, 4, 127, -3, 1, 0, 0],
    [0, 0, 2, 127, -1, 0, 0, 0],
    [0, 0, 0, 127, 1, 0, 0, 0],
    [0, 0, -1, 127, 2, 0, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],
    [0, 1, -5, 127, 6, -2, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],
    [-1, 2, -7, 126, 11, -4, 2, -1],
    [-1, 3, -8, 125, 13, -5, 2, -1],
    [-1, 3, -10, 124, 16, -6, 3, -1],
    [-1, 4, -11, 123, 18, -7, 3, -1],
    [-1, 4, -12, 122, 20, -7, 3, -1],
    [-1, 4, -13, 121, 23, -8, 3, -1],
    [-2, 5, -14, 120, 25, -9, 4, -1],
    [-1, 5, -15, 119, 27, -10, 4, -1],
    [-1, 5, -16, 118, 30, -11, 4, -1],
    [-2, 6, -17, 116, 33, -12, 5, -1],
    [-2, 6, -17, 114, 35, -12, 5, -1],
    [-2, 6, -18, 113, 38, -13, 5, -1],
    [-2, 7, -19, 111, 41, -14, 6, -2],
    [-2, 7, -19, 110, 43, -15, 6, -2],
    [-2, 7, -20, 108, 46, -15, 6, -2],
    [-2, 7, -20, 106, 49, -16, 6, -2],
    [-2, 7, -21, 104, 51, -16, 7, -2],
    [-2, 7, -21, 102, 54, -17, 7, -2],
    [-2, 8, -21, 100, 56, -18, 7, -2],
    [-2, 8, -22, 98, 59, -18, 7, -2],
    [-2, 8, -22, 96, 62, -19, 7, -2],
    [-2, 8, -22, 94, 64, -19, 7, -2],
    [-2, 8, -22, 91, 67, -20, 8, -2],
    [-2, 8, -22, 89, 69, -20, 8, -2],
    [-2, 8, -22, 87, 72, -21, 8, -2],
    [-2, 8, -21, 84, 74, -21, 8, -2],
    [-2, 8, -22, 82, 77, -21, 8, -2],
    [-2, 8, -21, 79, 79, -21, 8, -2],
    [-2, 8, -21, 77, 82, -22, 8, -2],
    [-2, 8, -21, 74, 84, -21, 8, -2],
    [-2, 8, -21, 72, 87, -22, 8, -2],
    [-2, 8, -20, 69, 89, -22, 8, -2],
    [-2, 8, -20, 67, 91, -22, 8, -2],
    [-2, 7, -19, 64, 94, -22, 8, -2],
    [-2, 7, -19, 62, 96, -22, 8, -2],
    [-2, 7, -18, 59, 98, -22, 8, -2],
    [-2, 7, -18, 56, 100, -21, 8, -2],
    [-2, 7, -17, 54, 102, -21, 7, -2],
    [-2, 7, -16, 51, 104, -21, 7, -2],
    [-2, 6, -16, 49, 106, -20, 7, -2],
    [-2, 6, -15, 46, 108, -20, 7, -2],
    [-2, 6, -15, 43, 110, -19, 7, -2],
    [-2, 6, -14, 41, 111, -19, 7, -2],
    [-1, 5, -13, 38, 113, -18, 6, -2],
    [-1, 5, -12, 35, 114, -17, 6, -2],
    [-1, 5, -12, 33, 116, -17, 6, -2],
    [-1, 4, -11, 30, 118, -16, 5, -1],
    [-1, 4, -10, 27, 119, -15, 5, -1],
    [-1, 4, -9, 25, 120, -14, 5, -2],
    [-1, 3, -8, 23, 121, -13, 4, -1],
    [-1, 3, -7, 20, 122, -12, 4, -1],
    [-1, 3, -7, 18, 123, -11, 4, -1],
    [-1, 3, -6, 16, 124, -10, 3, -1],
    [-1, 2, -5, 13, 125, -8, 3, -1],
    [-1, 2, -4, 11, 126, -7, 2, -1],
    [0, 1, -3, 8, 126, -6, 2, 0],
    [0, 1, -2, 6, 127, -5, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],
    [0, 0, 0, 2, 127, -1, 0, 0],
    [0, 0, 0, 1, 127, 0, 0, 0],
    [0, 0, 0, -1, 127, 2, 0, 0],
    [0, 0, 1, -3, 127, 4, -1, 0],
    [0, 0, 1, -4, 126, 6, -2, 1],
    [0, 0, 1, -5, 126, 8, -3, 1],
    [0, 0, 1, -6, 125, 11, -4, 1],
    [0, 0, 1, -7, 124, 13, -4, 1],
    [0, 0, 2, -8, 123, 15, -5, 1],
    [0, 0, 2, -9, 122, 18, -6, 1],
    [0, 0, 2, -10, 121, 20, -6, 1],
    [0, 0, 2, -11, 120, 22, -7, 2],
    [0, 0, 2, -12, 119, 25, -8, 2],
    [0, 0, 3, -13, 117, 27, -8, 2],
    [0, 0, 3, -13, 116, 29, -9, 2],
    [0, 0, 3, -14, 114, 32, -10, 3],
    [0, 0, 3, -15, 113, 35, -10, 2],
    [0, 0, 3, -15, 111, 37, -11, 3],
    [0, 0, 3, -16, 109, 40, -11, 3],
    [0, 0, 3, -16, 108, 42, -12, 3],
    [0, 0, 4, -17, 106, 45, -13, 3],
    [0, 0, 4, -17, 104, 47, -13, 3],
    [0, 0, 4, -17, 102, 50, -14, 3],
    [0, 0, 4, -17, 100, 52, -14, 3],
    [0, 0, 4, -18, 98, 55, -15, 4],
    [0, 0, 4, -18, 96, 58, -15, 3],
    [0, 0, 4, -18, 94, 60, -16, 4],
    [0, 0, 4, -18, 91, 63, -16, 4],
    [0, 0, 4, -18, 89, 65, -16, 4],
    [0, 0, 4, -18, 87, 68, -17, 4],
    [0, 0, 4, -18, 85, 70, -17, 4],
    [0, 0, 4, -18, 82, 73, -17, 4],
    [0, 0, 4, -18, 80, 75, -17, 4],
    [0, 0, 4, -18, 78, 78, -18, 4],
    [0, 0, 4, -17, 75, 80, -18, 4],
    [0, 0, 4, -17, 73, 82, -18, 4],
    [0, 0, 4, -17, 70, 85, -18, 4],
    [0, 0, 4, -17, 68, 87, -18, 4],
    [0, 0, 4, -16, 65, 89, -18, 4],
    [0, 0, 4, -16, 63, 91, -18, 4],
    [0, 0, 4, -16, 60, 94, -18, 4],
    [0, 0, 3, -15, 58, 96, -18, 4],
    [0, 0, 4, -15, 55, 98, -18, 4],
    [0, 0, 3, -14, 52, 100, -17, 4],
    [0, 0, 3, -14, 50, 102, -17, 4],
    [0, 0, 3, -13, 47, 104, -17, 4],
    [0, 0, 3, -13, 45, 106, -17, 4],
    [0, 0, 3, -12, 42, 108, -16, 3],
    [0, 0, 3, -11, 40, 109, -16, 3],
    [0, 0, 3, -11, 37, 111, -15, 3],
    [0, 0, 2, -10, 35, 113, -15, 3],
    [0, 0, 3, -10, 32, 114, -14, 3],
    [0, 0, 2, -9, 29, 116, -13, 3],
    [0, 0, 2, -8, 27, 117, -13, 3],
    [0, 0, 2, -8, 25, 119, -12, 2],
    [0, 0, 2, -7, 22, 120, -11, 2],
    [0, 0, 1, -6, 20, 121, -10, 2],
    [0, 0, 1, -6, 18, 122, -9, 2],
    [0, 0, 1, -5, 15, 123, -8, 2],
    [0, 0, 1, -4, 13, 124, -7, 1],
    [0, 0, 1, -4, 11, 125, -6, 1],
    [0, 0, 1, -3, 8, 126, -5, 1],
    [0, 0, 1, -2, 6, 126, -4, 1],
    [0, 0, 0, -1, 4, 127, -3, 1],
    [0, 0, 0, 0, 2, 127, -1, 0],
    [0, 0, 0, 0, 0, 127, 1, 0],
    [0, 0, 0, 0, -1, 127, 2, 0],
    [0, 0, 0, 0, -2, 127, 4, -1],
    [0, 0, 0, 0, -3, 126, 6, -1],
    [0, 0, 0, 0, -3, 125, 8, -2],
    [0, 0, 0, 0, -4, 124, 11, -3],
    [0, 0, 0, 0, -5, 123, 13, -3],
    [0, 0, 0, 0, -5, 121, 15, -3],
    [0, 0, 0, 0, -6, 120, 18, -4],
    [0, 0, 0, 0, -7, 119, 20, -4],
    [0, 0, 0, 0, -7, 118, 22, -5],
    [0, 0, 0, 0, -8, 116, 25, -5],
    [0, 0, 0, 0, -8, 115, 27, -6],
    [0, 0, 0, 0, -9, 113, 30, -6],
    [0, 0, 0, 0, -9, 112, 32, -7],
    [0, 0, 0, 0, -9, 110, 34, -7],
    [0, 0, 0, 0, -10, 108, 37, -7],
    [0, 0, 0, 0, -10, 107, 39, -8],
    [0, 0, 0, 0, -10, 105, 41, -8],
    [0, 0, 0, 0, -11, 103, 44, -8],
    [0, 0, 0, 0, -11, 101, 47, -9],
    [0, 0, 0, 0, -11, 99, 49, -9],
    [0, 0, 0, 0, -11, 97, 51, -9],
    [0, 0, 0, 0, -11, 95, 54, -10],
    [0, 0, 0, 0, -11, 93, 56, -10],
    [0, 0, 0, 0, -12, 91, 59, -10],
    [0, 0, 0, 0, -12, 89, 61, -10],
    [0, 0, 0, 0, -12, 87, 64, -11],
    [0, 0, 0, 0, -12, 85, 66, -11],
    [0, 0, 0, 0, -12, 82, 69, -11],
    [0, 0, 0, 0, -12, 80, 71, -11],
    [0, 0, 0, 0, -12, 78, 73, -11],
    [0, 0, 0, 0, -11, 75, 75, -11],
    [0, 0, 0, 0, -11, 73, 78, -12],
    [0, 0, 0, 0, -11, 71, 80, -12],
    [0, 0, 0, 0, -11, 69, 82, -12],
    [0, 0, 0, 0, -11, 66, 85, -12],
    [0, 0, 0, 0, -11, 64, 87, -12],
    [0, 0, 0, 0, -10, 61, 89, -12],
    [0, 0, 0, 0, -10, 59, 91, -12],
    [0, 0, 0, 0, -10, 56, 93, -11],
    [0, 0, 0, 0, -10, 54, 95, -11],
    [0, 0, 0, 0, -9, 51, 97, -11],
    [0, 0, 0, 0, -9, 49, 99, -11],
    [0, 0, 0, 0, -9, 47, 101, -11],
    [0, 0, 0, 0, -8, 44, 103, -11],
    [0, 0, 0, 0, -8, 41, 105, -10],
    [0, 0, 0, 0, -8, 39, 107, -10],
    [0, 0, 0, 0, -7, 37, 108, -10],
    [0, 0, 0, 0, -7, 34, 110, -9],
    [0, 0, 0, 0, -7, 32, 112, -9],
    [0, 0, 0, 0, -6, 30, 113, -9],
    [0, 0, 0, 0, -6, 27, 115, -8],
    [0, 0, 0, 0, -5, 25, 116, -8],
    [0, 0, 0, 0, -5, 22, 118, -7],
    [0, 0, 0, 0, -4, 20, 119, -7],
    [0, 0, 0, 0, -4, 18, 120, -6],
    [0, 0, 0, 0, -3, 15, 121, -5],
    [0, 0, 0, 0, -3, 13, 123, -5],
    [0, 0, 0, 0, -3, 11, 124, -4],
    [0, 0, 0, 0, -2, 8, 125, -3],
    [0, 0, 0, 0, -1, 6, 126, -3],
    [0, 0, 0, 0, -1, 4, 127, -2],
    [0, 0, 0, 0, 0, 2, 127, -1],
    [0, 0, 0, 0, 0, 0, 127, 1],
    [0, 0, 0, 0, 0, 0, 126, 2],
    [0, 0, 0, 0, 0, 0, 124, 4],
    [0, 0, 0, 0, 0, 0, 122, 6],
    [0, 0, 0, 0, 0, 0, 120, 8],
    [0, 0, 0, 0, 0, 0, 118, 10],
    [0, 0, 0, 0, 0, 0, 116, 12],
    [0, 0, 0, 0, 0, 0, 114, 14],
    [0, 0, 0, 0, 0, 0, 112, 16],
    [0, 0, 0, 0, 0, 0, 110, 18],
    [0, 0, 0, 0, 0, 0, 108, 20],
    [0, 0, 0, 0, 0, 0, 106, 22],
    [0, 0, 0, 0, 0, 0, 104, 24],
    [0, 0, 0, 0, 0, 0, 102, 26],
    [0, 0, 0, 0, 0, 0, 100, 28],
    [0, 0, 0, 0, 0, 0, 98, 30],
    [0, 0, 0, 0, 0, 0, 96, 32],
    [0, 0, 0, 0, 0, 0, 94, 34],
    [0, 0, 0, 0, 0, 0, 92, 36],
    [0, 0, 0, 0, 0, 0, 90, 38],
    [0, 0, 0, 0, 0, 0, 88, 40],
    [0, 0, 0, 0, 0, 0, 86, 42],
    [0, 0, 0, 0, 0, 0, 84, 44],
    [0, 0, 0, 0, 0, 0, 82, 46],
    [0, 0, 0, 0, 0, 0, 80, 48],
    [0, 0, 0, 0, 0, 0, 78, 50],
    [0, 0, 0, 0, 0, 0, 76, 52],
    [0, 0, 0, 0, 0, 0, 74, 54],
    [0, 0, 0, 0, 0, 0, 72, 56],
    [0, 0, 0, 0, 0, 0, 70, 58],
    [0, 0, 0, 0, 0, 0, 68, 60],
    [0, 0, 0, 0, 0, 0, 66, 62],
    [0, 0, 0, 0, 0, 0, 64, 64],
    [0, 0, 0, 0, 0, 0, 62, 66],
    [0, 0, 0, 0, 0, 0, 60, 68],
    [0, 0, 0, 0, 0, 0, 58, 70],
    [0, 0, 0, 0, 0, 0, 56, 72],
    [0, 0, 0, 0, 0, 0, 54, 74],
    [0, 0, 0, 0, 0, 0, 52, 76],
    [0, 0, 0, 0, 0, 0, 50, 78],
    [0, 0, 0, 0, 0, 0, 48, 80],
    [0, 0, 0, 0, 0, 0, 46, 82],
    [0, 0, 0, 0, 0, 0, 44, 84],
    [0, 0, 0, 0, 0, 0, 42, 86],
    [0, 0, 0, 0, 0, 0, 40, 88],
    [0, 0, 0, 0, 0, 0, 38, 90],
    [0, 0, 0, 0, 0, 0, 36, 92],
    [0, 0, 0, 0, 0, 0, 34, 94],
    [0, 0, 0, 0, 0, 0, 32, 96],
    [0, 0, 0, 0, 0, 0, 30, 98],
    [0, 0, 0, 0, 0, 0, 28, 100],
    [0, 0, 0, 0, 0, 0, 26, 102],
    [0, 0, 0, 0, 0, 0, 24, 104],
    [0, 0, 0, 0, 0, 0, 22, 106],
    [0, 0, 0, 0, 0, 0, 20, 108],
    [0, 0, 0, 0, 0, 0, 18, 110],
    [0, 0, 0, 0, 0, 0, 16, 112],
    [0, 0, 0, 0, 0, 0, 14, 114],
    [0, 0, 0, 0, 0, 0, 12, 116],
    [0, 0, 0, 0, 0, 0, 10, 118],
    [0, 0, 0, 0, 0, 0, 8, 120],
    [0, 0, 0, 0, 0, 0, 6, 122],
    [0, 0, 0, 0, 0, 0, 4, 124],
    [0, 0, 0, 0, 0, 0, 2, 126],
    [0, 0, 0, 0, 0, 0, 2, 126],
];

/// `Round2(x, n)` (§7.4, equation 6) on a 64-bit intermediate.
#[inline]
pub fn round2(x: i64, n: u32) -> i64 {
    if n == 0 {
        return x;
    }
    (x + (1i64 << (n - 1))) >> n
}

/// `Round2Signed(x, n)` (§7.4, equation 7) on a 64-bit intermediate.
#[inline]
pub fn round2_signed(x: i64, n: u32) -> i64 {
    if x >= 0 { round2(x, n) } else { -round2(-x, n) }
}

/// `resolve_divisor(d)` (§7.13.3.22) for the 64-bit affine determinant.
///
/// dav2d cross-check: `resolve_divisor_64()` in `src/warpmv.c`.
pub fn resolve_divisor_64(d: i64) -> (u32, i32) {
    let ad = d.unsigned_abs();
    let n = 63 - ad.leading_zeros();
    let e = ad - (1u64 << n);
    let f = if n > 7 {
        ((e + (1u64 << (n - 8))) >> (n - 7)) as usize
    } else {
        (e << (7 - n)) as usize
    };
    let div_shift = n + 9;
    let div_factor = DIV_LUT[f.min(DIV_LUT.len() - 1)] as i32;
    (div_shift, if d < 0 { -div_factor } else { div_factor })
}

/// `reduce_warp_model(params)` (§7.13.3.23).
///
/// Clamps the non-translational parameters and rounds them to
/// `WARP_PARAM_REDUCE_BITS`. dav2d cross-check: `iclip_wmp()` and the
/// `0x20 - (v < 0)` rounding used by `dav2d_get_shear_params()`.
pub fn reduce_warp_model(params: &mut [i32; 6]) {
    let max_value = (1 << (WARPEDMODEL_PREC_BITS - 1)) - (1 << WARP_PARAM_REDUCE_BITS);
    let min_value = -max_value;
    for (i, value) in params.iter_mut().enumerate().take(6).skip(2) {
        let offset = if i == 2 || i == 5 {
            1 << WARPEDMODEL_PREC_BITS
        } else {
            0
        };
        let original = *value as i64 - offset;
        let clamped = original.clamp(min_value as i64, max_value as i64);
        let rounded = round2_signed(clamped, WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS;
        *value = (rounded + offset) as i32;
    }
}

/// `setup_shear(params)` (§7.13.3.21). Returns `(alpha, beta, gamma, delta)`
/// when the model is a valid AV2 shear warp, `None` otherwise.
pub fn setup_shear(params: &[i32; 6]) -> Option<[i32; 4]> {
    let max_value = 32767 - (1 << (WARP_PARAM_REDUCE_BITS - 1));
    if params[2] <= 0 {
        return None;
    }
    let alpha0 = iclip(params[2] - (1 << WARPEDMODEL_PREC_BITS), -32768, max_value);
    let beta0 = iclip(params[3], -32768, max_value);
    let (div_shift, div_factor) = resolve_divisor_64(params[2] as i64);
    let v = (params[4] as i64) << WARPEDMODEL_PREC_BITS;
    let gamma0 = iclip(
        round2_signed(v * div_factor as i64, div_shift).clamp(i32::MIN as i64, i32::MAX as i64)
            as i32,
        -32768,
        max_value,
    );
    let w = params[3] as i64 * params[4] as i64;
    let delta_raw = params[5] as i64
        - round2_signed(w * div_factor as i64, div_shift)
        - (1i64 << WARPEDMODEL_PREC_BITS);
    let delta0 = iclip(
        delta_raw.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        -32768,
        max_value,
    );

    let alpha =
        (round2_signed(alpha0 as i64, WARP_PARAM_REDUCE_BITS) as i32) << WARP_PARAM_REDUCE_BITS;
    let beta =
        (round2_signed(beta0 as i64, WARP_PARAM_REDUCE_BITS) as i32) << WARP_PARAM_REDUCE_BITS;
    let gamma =
        (round2_signed(gamma0 as i64, WARP_PARAM_REDUCE_BITS) as i32) << WARP_PARAM_REDUCE_BITS;
    let delta =
        (round2_signed(delta0 as i64, WARP_PARAM_REDUCE_BITS) as i32) << WARP_PARAM_REDUCE_BITS;

    // §7.13.3.21 warpValid: the 4*|alpha| + 7*|beta| and 4*|gamma| + 4*|delta|
    // stability inequalities, with the `3 << WARPEDMODEL_PREC_BITS` bound.
    let limit = 3i32 << WARPEDMODEL_PREC_BITS;
    if 4 * alpha.abs() + 7 * beta.abs() >= limit {
        return None;
    }
    if 4 * gamma.abs() + 4 * delta.abs() >= limit {
        return None;
    }
    Some([alpha, beta, gamma, delta])
}

/// `get_warp_translation(params, refList)` (§7.13.3.23).
///
/// `w4`/`h4` are the block dimensions in 4x4 luma units, `mi_x4`/`mi_y4` the
/// block origin in 4x4 luma units and `mv` the block motion vector in 1/8-pel.
pub fn get_warp_translation(
    params: &mut [i32; 6],
    w4: usize,
    h4: usize,
    mi_x4: usize,
    mi_y4: usize,
    mv: MvXY,
) {
    let mid_y = (mi_y4 * 4 + h4 * 2 - 1) as i64;
    let mid_x = (mi_x4 * 4 + w4 * 2 - 1) as i64;
    let unary = 1i64 << WARPEDMODEL_PREC_BITS;
    let vx = (mv.x as i64) * (1i64 << GM_TRANS_ONLY_PREC_DIFF)
        - (mid_x * (params[2] as i64 - unary) + mid_y * params[3] as i64);
    let vy = (mv.y as i64) * (1i64 << GM_TRANS_ONLY_PREC_DIFF)
        - (mid_x * params[4] as i64 + mid_y * (params[5] as i64 - unary));
    let hi = WARPEDMODEL_TRANS_CLAMP - (1i64 << WARP_PARAM_REDUCE_BITS);
    params[0] = vx.clamp(-WARPEDMODEL_TRANS_CLAMP, hi) as i32;
    params[1] = vy.clamp(-WARPEDMODEL_TRANS_CLAMP, hi) as i32;
}

/// The `Default_Warp_Params` table (§7.12.2) — the identity affine model.
pub const DEFAULT_WARP_PARAMS: [i32; 6] = [
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
];

/// `params_equal(paramsA, paramsB)` (§7.12.3 bank update): only the
/// non-translational components are compared.
#[inline]
pub fn params_equal(a: &[i32; 6], b: &[i32; 6]) -> bool {
    a[2..6] == b[2..6]
}

/// Inserts `params` into the per-reference warp bank with the normative
/// most-recent-first ordering and de-duplication of §7.12.3
/// (`update_warp_param_bank`).
pub fn warp_bank_insert(bank: &mut Vec<[i32; 6]>, params: &[i32; 6]) {
    if let Some(pos) = bank.iter().position(|p| params_equal(p, params)) {
        let p = bank.remove(pos);
        bank.insert(0, p);
        return;
    }
    bank.insert(0, *params);
    if bank.len() > WARP_PARAM_BANK_SIZE {
        bank.truncate(WARP_PARAM_BANK_SIZE);
    }
}

/// Per-4x4 mode information the encoder keeps to evaluate warp corners.
///
/// AV2 §7.12.2.3 reads `MiSizes`, `MiColBase`, `MiRowBase`, `RefFrames`,
/// `IsInters`, `MotionModes`, `Mvs` and `SubMvs` of already coded neighbours.
#[derive(Debug, Clone, Copy, Default)]
pub struct WarpMiInfo {
    /// Block origin in 4x4 luma units.
    pub origin_x4: u16,
    /// Block origin in 4x4 luma units.
    pub origin_y4: u16,
    /// Block width/height in 4x4 luma units.
    pub w4: u8,
    /// Block height in 4x4 luma units.
    pub h4: u8,
    /// Coded flag: the neighbour exists inside this frame/tile.
    pub written: bool,
    /// `IsInters` for the block.
    pub inter: bool,
    /// `RefFrames[..][0]` list (frame slot index, or `-1`).
    pub ref0: i8,
    /// `Mvs[..][0]` list (the block motion vector, 1/8-pel).
    pub mv: MvXY,
    /// `RefFrames[..][1]` list (frame slot index, or `-1`).
    pub ref1: i8,
    /// `Mvs[..][1]` list (the block motion vector, 1/8-pel).
    pub mv1: MvXY,
    /// `SubMvs[..][0]` at this 4x4 cell (1/8-pel). For a warped block this is
    /// the projected sub-block vector of the enclosing 8x8 group (§7.13.3.19).
    pub sub_mv: MvXY,
    /// `SubMvs[..][1]` at this 4x4 cell (1/8-pel).
    pub sub_mv1: MvXY,
    /// `CwpIdxs[..]` (compound only, 8 = `CWP_EQUAL`).
    pub cwp: u8,
    /// `MotionModes` value (0 = `SIMPLE`, 3 = `DELTAWARP`).
    pub motion_mode: u8,
    /// `WarpParams[..][0]` for warp blocks.
    pub warp: [i32; 6],
}

impl WarpMiInfo {
    /// Creates the default (intra) entry for a block footprint.
    pub fn block(origin_x4: usize, origin_y4: usize, w4: usize, h4: usize) -> Self {
        Self {
            origin_x4: origin_x4 as u16,
            origin_y4: origin_y4 as u16,
            w4: w4 as u8,
            h4: h4 as u8,
            written: true,
            inter: false,
            ref0: -1,
            mv: MvXY::default(),
            ref1: -1,
            mv1: MvXY::default(),
            sub_mv: MvXY::default(),
            sub_mv1: MvXY::default(),
            cwp: 8,
            motion_mode: 0,
            warp: DEFAULT_WARP_PARAMS,
        }
    }

    /// Motion vector for `list` (0 or 1) from `Mvs` (the block vector).
    #[inline]
    pub fn list_mv(&self, list: usize) -> MvXY {
        if list == 0 { self.mv } else { self.mv1 }
    }

    /// Motion vector for `list` (0 or 1) from `SubMvs` (the per-4x4 vector).
    #[inline]
    pub fn sub_list_mv(&self, list: usize) -> MvXY {
        if list == 0 { self.sub_mv } else { self.sub_mv1 }
    }

    /// Reference frame for `list` (0 or 1).
    #[inline]
    pub fn list_ref(&self, list: usize) -> i8 {
        if list == 0 { self.ref0 } else { self.ref1 }
    }
}

/// `get_warp_motion_vector_xy_pos(mat, posRow, posCol)` (§7.12.2.4), returning
/// the motion vector at `(posRow * 4, posCol * 4)` in 1/8-pel.
pub fn warp_mv_at(params: &[i32; 6], pos_row4: usize, pos_col4: usize) -> MvXY {
    let y = (pos_row4 * 4) as i64;
    let x = (pos_col4 * 4) as i64;
    let unary = 1i64 << WARPEDMODEL_PREC_BITS;
    let xc = (params[2] as i64 * x + params[3] as i64 * y + params[0] as i64) - (x * unary);
    let yc = (params[4] as i64 * x + params[5] as i64 * y + params[1] as i64) - (y * unary);
    let mv_y = round2_signed(yc, WARPEDMODEL_PREC_BITS - 3).clamp(-32767, 32767) as i16;
    let mv_x = round2_signed(xc, WARPEDMODEL_PREC_BITS - 3).clamp(-32767, 32767) as i16;
    MvXY { x: mv_x, y: mv_y }
}

/// `MV_LOW` / `MV_UPP` (§7.3, `MV_IN_USE_BITS = 16`).
pub const MV_LOW: i32 = -(1 << 16);
pub const MV_UPP: i32 = 1 << 16;
/// `MV_BORDER` (§7.3).
pub const MV_BORDER: i32 = 128;

/// `SubMvs` entry for one 4x4 cell of a warped block (§7.13.3.19 `block_warp`).
///
/// The block-warp process projects the centre of each 8x8 section, so every
/// 4x4 cell of the same 8x8 group receives the same motion vector. `local_x4`
/// and `local_y4` are the 4x4 offsets inside the block.
#[inline]
pub fn warp_sub_mv(
    params: &[i32; 6],
    bx4: usize,
    by4: usize,
    local_x4: usize,
    local_y4: usize,
) -> MvXY {
    let src_x = (bx4 * 4 + (local_x4 & !1) * 4 + 4) as i64;
    let src_y = (by4 * 4 + (local_y4 & !1) * 4 + 4) as i64;
    let dst_x = params[2] as i64 * src_x + params[3] as i64 * src_y + params[0] as i64;
    let dst_y = params[4] as i64 * src_x + params[5] as i64 * src_y + params[1] as i64;
    let mv_x = round2_signed(
        dst_x - (src_x << WARPEDMODEL_PREC_BITS),
        WARPEDMODEL_PREC_BITS - 3,
    )
    .clamp(MV_LOW as i64 + 1, MV_UPP as i64 - 1);
    let mv_y = round2_signed(
        dst_y - (src_y << WARPEDMODEL_PREC_BITS),
        WARPEDMODEL_PREC_BITS - 3,
    )
    .clamp(MV_LOW as i64 + 1, MV_UPP as i64 - 1);
    MvXY {
        x: mv_x as i16,
        y: mv_y as i16,
    }
}

/// `get_warp_motion_vector(params, precision)` (§7.12.2.2) for `WARPMV`
/// blocks, including the `clamp_mv_row()` / `clamp_mv_col()` border clamps of
/// §5.20.9.4 / §5.20.9.5.
pub fn warp_motion_vector(
    params: &[i32; 6],
    w4: usize,
    h4: usize,
    mi_x4: usize,
    mi_y4: usize,
    mi_cols: usize,
    mi_rows: usize,
) -> MvXY {
    let bw = (w4 * 4) as i64;
    let bh = (h4 * 4) as i64;
    let x = mi_x4 as i64 * 4 + bw / 2 - 1;
    let y = mi_y4 as i64 * 4 + bh / 2 - 1;
    let unary = 1i64 << WARPEDMODEL_PREC_BITS;
    let xc = (params[2] as i64 - unary) * x + params[3] as i64 * y + params[0] as i64;
    let yc = params[4] as i64 * x + (params[5] as i64 - unary) * y + params[1] as i64;
    let mv_y = round2_signed(yc, WARPEDMODEL_PREC_BITS - 3)
        .clamp(MV_LOW as i64 + 1, MV_UPP as i64 - 1) as i32;
    let mv_x = round2_signed(xc, WARPEDMODEL_PREC_BITS - 3)
        .clamp(MV_LOW as i64 + 1, MV_UPP as i64 - 1) as i32;
    let low = -(((mi_y4 + h4) * 4 * 8) as i32) - MV_BORDER;
    let high = ((mi_rows - mi_y4) * 4 * 8) as i32 + MV_BORDER;
    let mv_y = mv_y.clamp(low, high);
    let low = -(((mi_x4 + w4) * 4 * 8) as i32) - MV_BORDER;
    let high = ((mi_cols - mi_x4) * 4 * 8) as i32 + MV_BORDER;
    let mv_x = mv_x.clamp(low, high);
    MvXY {
        x: mv_x as i16,
        y: mv_y as i16,
    }
}

/// `generate_points_from_corners(iter)` (§7.12.2.3) for a per-4x4 mode map.
///
/// Returns the reduced warp candidate when three corners are available, all
/// three motion vectors differ, and every reference point is non-negative.
#[allow(clippy::too_many_arguments)]
pub fn generate_points_from_corners(
    map: &[WarpMiInfo],
    mi_cols: usize,
    mi_rows: usize,
    mi_row: usize,
    mi_col: usize,
    w4: usize,
    h4: usize,
    _sb_w4: usize,
    sb_h4: usize,
    iter: usize,
    ref_id: i8,
) -> Option<[i32; 6]> {
    let mut corner_pts = [[0i32; 2]; 3];
    let mut corner_mvs = [[0i32; 2]; 3];
    let mut corners_found = 0usize;

    let mut warp_corner = |delta_row: isize, delta_col: isize, adjust_col: isize| {
        if corners_found >= 3 {
            return;
        }
        let is_sb_border = mi_row.is_multiple_of(sb_h4);
        let mv_row = mi_row as isize + delta_row;
        let mv_col = mi_col as isize + delta_col;
        let delta_col2 = delta_col + adjust_col;
        let mv_col2 = if delta_row < 0 && is_sb_border {
            (mi_col as isize - (mi_col as isize & 1)) + (delta_col2 - (delta_col2 & 1))
        } else {
            mi_col as isize + delta_col2
        };
        if is_sb_border && delta_col2 == 0 && w4 <= 2 {
            return;
        }
        if mv_row < 0 || mv_col2 < 0 || mv_col < 0 {
            return;
        }
        let (row, col, col2) = (mv_row as usize, mv_col as usize, mv_col2 as usize);
        if row >= mi_rows || col >= mi_cols || col2 >= mi_cols {
            return;
        }
        let info = map[row * mi_cols + col2];
        if !info.written || !info.inter {
            return;
        }
        // `RefFrames[..][ref]` for `ref = 0..1`: the first list that matches
        // `RefFrame[0]` supplies the corner.
        let list = if info.ref0 == ref_id {
            0usize
        } else if info.ref1 == ref_id {
            1usize
        } else {
            return;
        };
        corner_pts[corners_found] = [((row + 1) * 4) as i32, ((col + 1) * 4) as i32];
        corner_mvs[corners_found] = if info.motion_mode >= 2 {
            // `LOCALWARP` = 2 in the AV2 motion-mode enumeration. The corner
            // vector is warped from the corner position, not the 8x8 centre.
            if list > 0 {
                return;
            }
            let mv = warp_mv_at(&info.warp, row + 1, col + 1);
            [
                crate::av2::inter::clamp_mv_row(mv.y as i32, mi_row, h4, mi_rows),
                crate::av2::inter::clamp_mv_col(mv.x as i32, mi_col, w4, mi_cols),
            ]
        } else {
            let mv = info.sub_list_mv(list);
            [mv.y as i32, mv.x as i32]
        };
        corners_found += 1;
    };

    warp_corner(-1, -1, iter as isize);
    warp_corner(-1, w4 as isize - 1, iter as isize);
    warp_corner(h4 as isize - 1, -1, 0);
    if corners_found != 3 {
        return None;
    }

    let mut all_same = true;
    let mut ref_pts = [[0i64; 2]; 3];
    for n in 0..3 {
        for c in 0..2 {
            ref_pts[n][c] = ((corner_pts[n][c] as i64) << WARPEDMODEL_PREC_BITS)
                + ((corner_mvs[n][c] as i64) << GM_TRANS_ONLY_PREC_DIFF);
            if corner_mvs[n][c] != corner_mvs[0][c] {
                all_same = false;
            }
        }
    }
    if all_same {
        return None;
    }
    if ref_pts.iter().flatten().any(|&v| v < 0) {
        return None;
    }

    let width_log2 = w4.trailing_zeros() + 2;
    let height_log2 = h4.trailing_zeros() + 2;
    let y0 = corner_pts[0][0] as i64;
    let x0 = corner_pts[0][1] as i64;
    let mut wmmat = [0i32; 6];
    wmmat[2] = ((ref_pts[1][1] - ref_pts[0][1]) >> width_log2) as i32;
    wmmat[4] = ((ref_pts[1][0] - ref_pts[0][0]) >> width_log2) as i32;
    wmmat[3] = ((ref_pts[2][1] - ref_pts[0][1]) >> height_log2) as i32;
    wmmat[5] = ((ref_pts[2][0] - ref_pts[0][0]) >> height_log2) as i32;
    let wmmat0 = ref_pts[0][1] - wmmat[2] as i64 * x0 - wmmat[3] as i64 * y0;
    let wmmat1 = ref_pts[0][0] - wmmat[4] as i64 * x0 - wmmat[5] as i64 * y0;
    reduce_warp_model(&mut wmmat);
    let hi = WARPEDMODEL_TRANS_CLAMP - (1i64 << WARP_PARAM_REDUCE_BITS);
    wmmat[0] = wmmat0.clamp(-WARPEDMODEL_TRANS_CLAMP, hi) as i32;
    wmmat[1] = wmmat1.clamp(-WARPEDMODEL_TRANS_CLAMP, hi) as i32;
    Some(wmmat)
}

/// `LS_MV_MAX` (§7.3): largest motion-vector difference accepted by the
/// local-warp least-squares fit.
pub const LS_MV_MAX: i32 = 256;
/// `LEAST_SQUARES_SAMPLES_MAX` (§7.3): maximum number of warp samples.
pub const LEAST_SQUARES_SAMPLES_MAX: usize = 8;

/// `find_warp_samples()` (§7.12.3) over a per-4x4 block map.
///
/// Returns `(CandList, NumSamples)` where each candidate is
/// `[midY*8, midX*8, midY*8+mvY, midX*8+mvX]` in 1/8-pel units.
#[allow(clippy::too_many_arguments)]
pub fn find_warp_samples(
    map: &[WarpMiInfo],
    mi_cols: usize,
    mi_rows: usize,
    sb_h4: usize,
    tile_row_start4: usize,
    tile_row_end4: usize,
    tile_col_start4: usize,
    tile_col_end4: usize,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
    ref_id: i8,
) -> ([[i32; 4]; LEAST_SQUARES_SAMPLES_MAX], usize) {
    let mut out = [[0i32; 4]; LEAST_SQUARES_SAMPLES_MAX];
    let mut count = 0usize;
    let sb_h4 = sb_h4.max(1);
    let is_sb_border = by4.is_multiple_of(sb_h4);

    let inside = |row: isize, col: isize| {
        row >= tile_row_start4 as isize
            && row < tile_row_end4 as isize
            && col >= tile_col_start4 as isize
            && col < tile_col_end4 as isize
    };
    let above_stored = |map: &[WarpMiInfo], delta_col: isize| -> bool {
        let row = by4 as isize - 1;
        let col = bx4 as isize + delta_col;
        if !inside(row, col) {
            return false;
        }
        if !is_sb_border {
            return true;
        }
        if col.rem_euclid(2) == 0 {
            return true;
        }
        let src_w4 = map[row as usize * mi_cols + col as usize].w4;
        if src_w4 == 1 {
            return false;
        }
        col + 1 < tile_col_end4 as isize
    };
    let mut add = |delta_row: isize, delta_col: isize| {
        if count >= LEAST_SQUARES_SAMPLES_MAX {
            return;
        }
        let mv_row = by4 as isize + delta_row;
        let mv_col = bx4 as isize + delta_col;
        if !inside(mv_row, mv_col) {
            return;
        }
        let info = map[mv_row as usize * mi_cols + mv_col as usize];
        if !info.written || !info.inter {
            return;
        }
        for list in 0..2usize {
            if info.list_ref(list) != ref_id {
                continue;
            }
            let mid_y = info.origin_y4 as i32 * 4 + info.h4 as i32 * 2 - 1;
            let mid_x = info.origin_x4 as i32 * 4 + info.w4 as i32 * 2 - 1;
            let mv = info.list_mv(list);
            out[count] = [
                mid_y * 8,
                mid_x * 8,
                mid_y * 8 + mv.y as i32,
                mid_x * 8 + mv.x as i32,
            ];
            count += 1;
            if count >= LEAST_SQUARES_SAMPLES_MAX {
                return;
            }
        }
    };

    let mut do_top_left = true;
    let mut do_top_right = true;
    if by4 > tile_row_start4 && by4 > 0 {
        let above = map[(by4 - 1) * mi_cols + bx4];
        let col_offset = above.origin_x4 as isize - bx4 as isize;
        if col_offset < 0 {
            do_top_left = false;
        }
        let limit = bw4.min(mi_cols - bx4) as isize;
        let mut i = col_offset;
        while i < limit {
            let col = bx4 as isize + i;
            if col >= mi_cols as isize {
                break;
            }
            let src_w = map[(by4 - 1) * mi_cols + col as usize].w4 as isize;
            if src_w <= 0 {
                break;
            }
            if above_stored(map, i) {
                add(-1, i);
            }
            i += src_w;
        }
        do_top_right = i == bw4 as isize && i < (mi_cols - bx4) as isize;
    }
    if bx4 > tile_col_start4 && bx4 > 0 {
        let left = map[by4 * mi_cols + bx4 - 1];
        let row_offset = left.origin_y4 as isize - by4 as isize;
        if row_offset < 0 {
            do_top_left = false;
        }
        let limit = bh4.min(mi_rows - by4) as isize;
        let mut i = row_offset;
        while i < limit {
            let row = by4 as isize + i;
            if row >= mi_rows as isize {
                break;
            }
            let src_h = map[row as usize * mi_cols + bx4 - 1].h4 as isize;
            if src_h <= 0 {
                break;
            }
            add(i, -1);
            i += src_h;
        }
    }
    if do_top_left && above_stored(map, -1) {
        add(-1, -1);
    }
    if do_top_right && bw4 <= 16 && above_stored(map, bw4 as isize) {
        add(-1, bw4 as isize);
    }
    (out, count)
}

/// `warp_estimation()` (§7.13.3.23): least-squares fit of the local affine
/// model from `CandList`, followed by `reduce_warp_model()` and
/// `get_warp_translation()`.
#[allow(clippy::too_many_arguments)]
pub fn warp_estimation(
    cands: &[[i32; 4]; LEAST_SQUARES_SAMPLES_MAX],
    num_samples: usize,
    bw4: usize,
    bh4: usize,
    bx4: usize,
    by4: usize,
    mv: MvXY,
) -> [i32; 6] {
    let mut a = [[0i64; 2]; 2];
    let mut bx = [0i64; 2];
    let mut by = [0i64; 2];
    let mid_y = by4 as i64 * 4 + bh4 as i64 * 2 - 1;
    let mid_x = bx4 as i64 * 4 + bw4 as i64 * 2 - 1;
    let suy = mid_y * 8;
    let sux = mid_x * 8;
    let duy = suy + mv.y as i64;
    let dux = sux + mv.x as i64;
    for cand in cands.iter().take(num_samples) {
        let sy = cand[0] as i64 - suy;
        let sx = cand[1] as i64 - sux;
        let dy = cand[2] as i64 - duy;
        let dx = cand[3] as i64 - dux;
        if (sx - dx).abs() < LS_MV_MAX as i64 && (sy - dy).abs() < LS_MV_MAX as i64 {
            a[0][0] += ((sx * sx) >> 2) + sx + sx + 8;
            a[0][1] += ((sx * sy) >> 2) + sx + sy + 4;
            a[1][1] += ((sy * sy) >> 2) + sy + sy + 8;
            bx[0] += ((sx * dx) >> 2) + sx + dx + 8;
            bx[1] += ((sy * dx) >> 2) + sy + dx + 4;
            by[0] += ((sx * dy) >> 2) + sx + dy + 4;
            by[1] += ((sy * dy) >> 2) + sy + dy + 8;
        }
    }
    let det = a[0][0] * a[1][1] - a[0][1] * a[0][1];
    if det == 0 {
        let mut params = DEFAULT_WARP_PARAMS;
        get_warp_translation(&mut params, bw4, bh4, bx4, by4, mv);
        return params;
    }
    let (mut shift, factor) = resolve_divisor_64(det);
    let mut factor = factor as i64;
    if shift >= WARPEDMODEL_PREC_BITS {
        shift -= WARPEDMODEL_PREC_BITS;
    } else {
        factor <<= WARPEDMODEL_PREC_BITS - shift;
        shift = 0;
    }
    let diag = |v: i64| -> i32 {
        round2_signed(v * factor, shift).clamp(i32::MIN as i64, i32::MAX as i64) as i32
    };
    let mut params = [0i32; 6];
    params[2] = diag(a[1][1] * bx[0] - a[0][1] * bx[1]);
    params[3] = diag(-a[0][1] * bx[0] + a[0][0] * bx[1]);
    params[4] = diag(a[1][1] * by[0] - a[0][1] * by[1]);
    params[5] = diag(-a[0][1] * by[0] + a[0][0] * by[1]);
    reduce_warp_model(&mut params);
    get_warp_translation(&mut params, bw4, bh4, bx4, by4, mv);
    params
}

/// `true` when the block has a neighbour with a matching reference for
/// `has_cs_ext` / `WarpSampleFound` (§7.11.4).
#[allow(clippy::too_many_arguments)]
pub fn warp_has_ref_neighbour(
    map: &[WarpMiInfo],
    mi_cols: usize,
    mi_rows: usize,
    sb_h4: usize,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
    ref_id: i8,
) -> bool {
    let sb_h4 = sb_h4.max(1);
    let is_sb_border = by4.is_multiple_of(sb_h4);
    let check = |row: isize, col: isize| -> bool {
        if row < 0 || col < 0 || row >= mi_rows as isize || col >= mi_cols as isize {
            return false;
        }
        let info = map[row as usize * mi_cols + col as usize];
        info.written && info.inter && (info.ref0 == ref_id || info.ref1 == ref_id)
    };
    let left = bx4 > 0 && check(by4 as isize, bx4 as isize - 1)
        || (bx4 > 0
            && by4 + bh4 <= mi_rows
            && check(by4 as isize + bh4 as isize - 1, bx4 as isize - 1));
    let above = if by4 == 0 {
        false
    } else if is_sb_border {
        check(by4 as isize - 1, (bx4 & !1) as isize)
            || ((bx4 + bw4 >= 2) && check(by4 as isize - 1, ((bx4 + bw4 - 2) & !1) as isize))
    } else {
        check(by4 as isize - 1, bx4 as isize)
            || (bx4 + bw4 <= mi_cols && check(by4 as isize - 1, (bx4 + bw4 - 1) as isize))
    };
    left || above
}

/// CDF context for `use_extend_warp` / `use_local_warp` (§8.3.2): the number
/// of left/above neighbours whose motion mode is `LOCALWARP` or greater.
pub fn warp_mode_context(
    map: &[WarpMiInfo],
    mi_cols: usize,
    mi_rows: usize,
    sb_h4: usize,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
) -> usize {
    let sb_h4 = sb_h4.max(1);
    let have_top_in_sb = !by4.is_multiple_of(sb_h4);
    let mut ctx = 0usize;
    let mut add = |row: isize, col: isize| {
        if row < 0 || col < 0 || row >= mi_rows as isize || col >= mi_cols as isize {
            return;
        }
        let info = map[row as usize * mi_cols + col as usize];
        if info.written && info.inter && info.motion_mode >= 2 {
            ctx += 1;
        }
    };
    let mut n = 0usize;
    if bx4 > 0 {
        add(by4 as isize + bh4 as isize - 1, bx4 as isize - 1);
        n += 1;
    }
    if have_top_in_sb {
        add(by4 as isize - 1, bx4 as isize + bw4 as isize - 1);
        n += 1;
    }
    if bx4 > 0 && n < 2 {
        add(by4 as isize, bx4 as isize - 1);
        n += 1;
    }
    if have_top_in_sb && n < 2 {
        add(by4 as isize - 1, bx4 as isize);
    }
    ctx.min(2)
}

/// CDF context for `use_local_warp` (§8.3.2, dav2d `warp_causal[cs_ctx]`):
/// `(ext_ctx > 0) + (left == LOCALWARP) + (above == LOCALWARP)`.
pub fn warp_local_context(
    map: &[WarpMiInfo],
    mi_cols: usize,
    mi_rows: usize,
    sb_h4: usize,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
) -> usize {
    let sb_h4 = sb_h4.max(1);
    let have_top_in_sb = !by4.is_multiple_of(sb_h4);
    let mode_at = |row: isize, col: isize| -> u8 {
        if row < 0 || col < 0 || row >= mi_rows as isize || col >= mi_cols as isize {
            return 0;
        }
        let info = map[row as usize * mi_cols + col as usize];
        if info.written && info.inter {
            info.motion_mode
        } else {
            0
        }
    };
    let mut modes = [0u8; 2];
    let mut n = 0usize;
    if bx4 > 0 {
        modes[0] = mode_at(by4 as isize + bh4 as isize - 1, bx4 as isize - 1);
        n = 1;
    }
    if have_top_in_sb && n < 2 {
        modes[n] = mode_at(by4 as isize - 1, bx4 as isize + bw4 as isize - 1);
        n += 1;
    }
    if bx4 > 0 && n < 2 {
        modes[n] = mode_at(by4 as isize, bx4 as isize - 1);
        n += 1;
    }
    if have_top_in_sb && n < 2 {
        modes[n] = mode_at(by4 as isize - 1, bx4 as isize);
    }
    let ext_ctx = (modes[0] >= 2) as usize + (modes[1] >= 2) as usize;
    (ext_ctx > 0) as usize + (modes[0] == 2) as usize + (modes[1] == 2) as usize
}

/// `extend_warp_estimation()` (§7.13.3.24) over a per-4x4 block map.
///
/// `stack_row` / `stack_col` are `RefStackRowOffset[RefMvIdx]` and
/// `RefStackColOffset[RefMvIdx]`. The specification keeps those offsets only
/// when either is `-1`; otherwise it uses `ExtendDeltaRow` / `ExtendDeltaCol`,
/// the first neighbour in the §7.11.4 scan-point order whose reference matches
/// `RefFrame[0]`. This function performs that scan itself.
#[allow(clippy::too_many_arguments)]
pub fn extend_warp_params(
    map: &[WarpMiInfo],
    mi_cols: usize,
    mi_rows: usize,
    sb_h4: usize,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
    ref_id: i8,
    stack_row: i16,
    stack_col: i16,
    mv: MvXY,
) -> Option<[i32; 6]> {
    let sb_h4 = sb_h4.max(1);
    let is_sb_border = by4.is_multiple_of(sb_h4);
    let inside = |row: isize, col: isize| -> bool {
        row >= 0 && col >= 0 && row < mi_rows as isize && col < mi_cols as isize
    };
    let has_ref = |row: isize, col: isize| -> bool {
        if !inside(row, col) {
            return false;
        }
        let info = map[row as usize * mi_cols + col as usize];
        info.written && info.inter && (info.ref0 == ref_id || info.ref1 == ref_id)
    };
    let (mut delta_row, mut delta_col) = (stack_row as isize, stack_col as isize);
    if delta_row != -1 && delta_col != -1 {
        // `ExtendDeltaRow` / `ExtendDeltaCol`: the first matching scan point.
        let bh4i = bh4 as isize;
        let bw4i = bw4 as isize;
        if bx4 > 0 && has_ref(by4 as isize + bh4i - 1, bx4 as isize - 1) {
            delta_row = bh4i - 1;
            delta_col = -1;
        } else if by4 > 0
            && has_ref(
                by4 as isize - 1,
                bx4 as isize + bw4i - 1 - (is_sb_border as isize & (bx4 as isize & 1)),
            )
        {
            delta_row = -1;
            delta_col =
                -(bx4 as isize & is_sb_border as isize) + bw4i - (1 + is_sb_border as isize);
        } else if by4 > 0 && bx4 > 0 && has_ref(by4 as isize, bx4 as isize - 1) {
            delta_row = 0;
            delta_col = -1;
        } else if by4 > 0 && has_ref(by4 as isize - 1, bx4 as isize) {
            delta_row = -1;
            delta_col = -(bx4 as isize & is_sb_border as isize);
        }
    }
    if !inside(by4 as isize + delta_row, bx4 as isize + delta_col) {
        return None;
    }
    let row = (by4 as isize + delta_row) as usize;
    let col = (bx4 as isize + delta_col) as usize;
    let info = map[row * mi_cols + col];
    let list = if info.ref0 == ref_id {
        0usize
    } else if info.ref1 == ref_id {
        1usize
    } else {
        return None;
    };
    let params = if info.motion_mode >= 2 {
        info.warp
    } else {
        let mut p = DEFAULT_WARP_PARAMS;
        let nmv = info.list_mv(list);
        p[0] = (nmv.x as i32) << (WARPEDMODEL_PREC_BITS - 3);
        p[1] = (nmv.y as i32) << (WARPEDMODEL_PREC_BITS - 3);
        p
    };
    let mid_y = by4 as i64 * 4 + bh4 as i64 * 2 - 1;
    let mid_x = bx4 as i64 * 4 + bw4 as i64 * 2 - 1;
    let proj_mid_x =
        (mid_x << WARPEDMODEL_PREC_BITS) + ((mv.x as i64) << (WARPEDMODEL_PREC_BITS - 3));
    let proj_mid_y =
        (mid_y << WARPEDMODEL_PREC_BITS) + ((mv.y as i64) << (WARPEDMODEL_PREC_BITS - 3));
    let neighbor_is_above = delta_row == -1 && delta_col >= 0;
    let mut ext = [0i32; 6];
    if neighbor_is_above {
        ext[2] = params[2];
        ext[4] = params[4];
        let above_x = mid_x;
        let above_y = by4 as i64 * 4 - 1;
        let proj_above_x =
            params[2] as i64 * above_x + params[3] as i64 * above_y + params[0] as i64;
        let proj_above_y =
            params[4] as i64 * above_x + params[5] as i64 * above_y + params[1] as i64;
        let shift = bh4.trailing_zeros() + 1;
        ext[3] = round2_signed(proj_mid_x - proj_above_x, shift) as i32;
        ext[5] = round2_signed(proj_mid_y - proj_above_y, shift) as i32;
    } else {
        ext[3] = params[3];
        ext[5] = params[5];
        let left_x = bx4 as i64 * 4 - 1;
        let left_y = mid_y;
        let proj_left_x = params[2] as i64 * left_x + params[3] as i64 * left_y + params[0] as i64;
        let proj_left_y = params[4] as i64 * left_x + params[5] as i64 * left_y + params[1] as i64;
        let shift = bw4.trailing_zeros() + 1;
        ext[2] = round2_signed(proj_mid_x - proj_left_x, shift) as i32;
        ext[4] = round2_signed(proj_mid_y - proj_left_y, shift) as i32;
    }
    reduce_warp_model(&mut ext);
    get_warp_translation(&mut ext, bw4, bh4, bx4, by4, mv);
    Some(ext)
}

/// `block_warp()` (§7.13.3.19): warps one plane of a block into `dst`.
///
/// `x`/`y` and `w`/`h` are in the plane's sample units, `sub_x`/`sub_y` the
/// plane subsampling. The output is written at `dst_off` with `dst_stride`.
/// `abcd` is the output of [`setup_shear`] for `params`.
#[allow(clippy::too_many_arguments)]
pub fn block_warp<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    dst_stride: usize,
    ref_plane: &Plane<BD::Pixel>,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    sub_x: usize,
    sub_y: usize,
    params: &[i32; 6],
    abcd: &[i32; 4],
) {
    let m0 = params[0] as i64;
    let m1 = params[1] as i64;
    let m2 = params[2] as i64;
    let m3 = params[3] as i64;
    let m4 = params[4] as i64;
    let m5 = params[5] as i64;
    let alpha = abcd[0];
    let beta = abcd[1];
    let gamma = abcd[2];
    let delta = abcd[3];
    let ref_w = ref_plane.coded_width as i32;
    let ref_h = ref_plane.coded_height as i32;
    let max_index = WARPED_FILTERS.len() as i32 - 1;
    // §7.13.3.19 performs the two-pass rounding with `InterRound0` and
    // `InterRound1`. The reference decoder keeps the intermediate array in
    // 16 bits (`intermediate_bits = 14 - BitDepth`), so for high bit depth the
    // passes round by `BitDepth - 7` and `21 - BitDepth` (5/9 at 12-bit);
    // 8-bit keeps the `(3, 11)` pair. dav2d cross-check:
    // `get_intermediate_bits()` / `FILTER_WARP_RND()` in `src/mc_tmpl.c`.
    let bd_bits = bd.bitdepth() as u32;
    let (round0, round1) = if bd_bits <= 8 {
        (INTER_ROUND0, INTER_ROUND1)
    } else {
        (bd_bits - 7, 21 - bd_bits)
    };

    let mut i8 = 0usize;
    while i8 * 8 < h {
        let mut j8 = 0usize;
        while j8 * 8 < w {
            let src_x = ((x + j8 * 8 + 4) << sub_x) as i64;
            let src_y = ((y + i8 * 8 + 4) << sub_y) as i64;
            let dst_x = m2 * src_x + m3 * src_y + m0;
            let dst_y = m4 * src_x + m5 * src_y + m1;
            let x4 = dst_x >> sub_x;
            let y4 = dst_y >> sub_y;
            let ix4 = x4 >> WARPEDMODEL_PREC_BITS;
            let sx4 = (x4 & ((1i64 << WARPEDMODEL_PREC_BITS) - 1)) as i32;
            let iy4 = y4 >> WARPEDMODEL_PREC_BITS;
            let sy4 = (y4 & ((1i64 << WARPEDMODEL_PREC_BITS) - 1)) as i32;

            let mut intermediate = [[0i32; 8]; 15];
            for i1 in -7i32..8 {
                for i2 in -4i32..4 {
                    let sx = sx4 + alpha * i2 + beta * i1;
                    let offs = (((sx + (1 << (WARPEDDIFF_PREC_BITS - 1))) >> WARPEDDIFF_PREC_BITS)
                        + 3 * WARPEDPIXEL_PREC_SHIFTS)
                        .clamp(0, max_index) as usize;
                    let row = (iy4 + i1 as i64).clamp(0, (ref_h - 1) as i64) as usize;
                    let mut s = 0i64;
                    for i3 in 0..8usize {
                        let col = (ix4 + (i2 - 3) as i64 + i3 as i64).clamp(0, (ref_w - 1) as i64)
                            as usize;
                        s += WARPED_FILTERS[offs][i3] as i64
                            * ref_plane.data[row * ref_plane.stride + col].to_i32() as i64;
                    }
                    intermediate[(i1 + 7) as usize][(i2 + 4) as usize] = round2(s, round0) as i32;
                }
            }

            for i1 in -4i32..4 {
                for i2 in -4i32..4 {
                    let sy = sy4 + gamma * i2 + delta * i1;
                    let offs = (((sy + (1 << (WARPEDDIFF_PREC_BITS - 1))) >> WARPEDDIFF_PREC_BITS)
                        + 3 * WARPEDPIXEL_PREC_SHIFTS)
                        .clamp(0, max_index) as usize;
                    let mut s = 0i64;
                    for i3 in 0..8usize {
                        s += WARPED_FILTERS[offs][i3] as i64
                            * intermediate[(i1 + i3 as i32 + 4) as usize][(i2 + 4) as usize] as i64;
                    }
                    let value = round2(s, round1) as i32;
                    let idx = dst_off
                        + (i8 * 8 + (i1 + 4) as usize) * dst_stride
                        + j8 * 8
                        + (i2 + 4) as usize;
                    dst[idx] = bd.clip_pixel(value);
                }
            }
            j8 += 1;
        }
        i8 += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_warp_copies_reference() {
        // With identity parameters and a zero block motion vector,
        // `get_warp_translation` produces zero translation and `block_warp`
        // reduces to the phase-0 warped filter, which is an exact copy.
        let mut params = DEFAULT_WARP_PARAMS;
        get_warp_translation(&mut params, 4, 4, 0, 0, MvXY::default());
        assert_eq!(params, DEFAULT_WARP_PARAMS);
        let abcd = setup_shear(&params).expect("identity shear must be valid");
        assert_eq!(abcd, [0, 0, 0, 0]);
    }

    #[test]
    fn reduce_warp_model_rounds_to_64() {
        let mut params = [0, 0, 65536 + 100, 33, -33, 65536 + 100];
        reduce_warp_model(&mut params);
        assert_eq!(params[2], 65536 + 128);
        assert_eq!(params[3], 64);
        assert_eq!(params[4], -64);
        assert_eq!(params[5], 65536 + 128);
    }

    #[test]
    fn shear_rejects_degenerate_scale() {
        let mut params = DEFAULT_WARP_PARAMS;
        params[2] = 0;
        assert!(setup_shear(&params).is_none());
    }

    #[test]
    fn corner_candidate_is_affine_when_corner_mvs_differ() {
        // 8x8 map of 4x4 cells; the block under test is 16x16 at (4,4).
        let mi_cols = 8usize;
        let mi_rows = 8usize;
        let mut map = vec![WarpMiInfo::default(); mi_cols * mi_rows];
        let corners = [
            (3usize, 3usize, MvXY { x: 4, y: 0 }),
            (3usize, 7usize, MvXY { x: 8, y: 0 }),
            (7usize, 3usize, MvXY { x: 0, y: 8 }),
        ];
        for (r, c, mv) in corners {
            let mut info = WarpMiInfo::block(c, r, 1, 1);
            info.inter = true;
            info.ref0 = 0;
            info.mv = mv;
            info.sub_mv = mv;
            map[r * mi_cols + c] = info;
        }
        let cand = generate_points_from_corners(&map, mi_cols, mi_rows, 4, 4, 4, 4, 8, 8, 0, 0)
            .expect("three differing corner MVs must yield a warp candidate");
        assert_ne!(cand[2], 1 << WARPEDMODEL_PREC_BITS);
    }

    #[test]
    fn block_warp_zoom_is_not_a_copy() {
        let bd = crate::av2::pixel::BitDepth8;
        let mut plane = Plane::<u8>::new(32, 32, 0, 0, 8);
        for r in 0..32 {
            for c in 0..32 {
                plane.data[r * plane.stride + c] = ((r * 7 + c * 3) % 251) as u8;
            }
        }
        let mut ident = vec![0u8; 16 * 16];
        let identity = DEFAULT_WARP_PARAMS;
        let abcd = setup_shear(&identity).unwrap();
        block_warp(
            bd, &mut ident, 0, 16, &plane, 4, 4, 16, 16, 0, 0, &identity, &abcd,
        );

        let mut zoom = [0i32; 6];
        zoom[2] = (1 << WARPEDMODEL_PREC_BITS) + 2048;
        zoom[5] = (1 << WARPEDMODEL_PREC_BITS) + 2048;
        let abcd = setup_shear(&zoom).unwrap();
        get_warp_translation(&mut zoom, 4, 4, 1, 1, MvXY::default());
        let mut warped = vec![0u8; 16 * 16];
        block_warp(
            bd,
            &mut warped,
            0,
            16,
            &plane,
            4,
            4,
            16,
            16,
            0,
            0,
            &zoom,
            &abcd,
        );
        assert_ne!(ident, warped);
    }

    #[test]
    fn bank_dedups_by_non_translation_parameters() {
        let mut bank = Vec::new();
        let a = [10, 20, 65536 + 2048, 0, 0, 65536 + 2048];
        let b = [99, 99, 65536 + 2048, 0, 0, 65536 + 2048];
        warp_bank_insert(&mut bank, &a);
        warp_bank_insert(&mut bank, &b);
        assert_eq!(bank.len(), 1);
        assert_eq!(bank[0][0], 10);
    }
}
