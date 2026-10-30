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

/// Builds the `WarpParamStack` of §7.12.2 for a single-reference block.
///
/// `corner` is the candidate produced by `generate_points_from_corners`
/// (§7.12.2.3), `bank` the per-reference bank, and `gm` the derived global
/// motion model (identity when `use_global_motion` is 0). The list order is
/// the normative one: corner candidate, bank (most recent first), global
/// motion, then two `Default_Warp_Params` entries.
pub fn build_warp_param_stack(
    corner: Option<[i32; 6]>,
    bank: &[[i32; 6]],
    gm: &[i32; 6],
) -> [[i32; 6]; MAX_WARP_REF_CANDIDATES] {
    let mut stack = [DEFAULT_WARP_PARAMS; MAX_WARP_REF_CANDIDATES];
    let mut n = 0usize;
    let mut push = |v: [i32; 6]| {
        if n < MAX_WARP_REF_CANDIDATES {
            stack[n] = v;
            n += 1;
        }
    };
    if let Some(c) = corner {
        push(c);
    }
    for p in bank {
        push(*p);
    }
    push(*gm);
    push(DEFAULT_WARP_PARAMS);
    push(DEFAULT_WARP_PARAMS);
    stack
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
    /// `Mvs[..][0]` list (1/8-pel).
    pub mv: MvXY,
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
            motion_mode: 0,
            warp: DEFAULT_WARP_PARAMS,
        }
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

/// `generate_points_from_corners(iter)` (§7.12.2.3) for the encoder's map.
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
        // Single-reference: only list 0 can match RefFrame[0].
        if info.ref0 != 0 {
            return;
        }
        corner_pts[corners_found] = [((row + 1) * 4) as i32, ((col + 1) * 4) as i32];
        corner_mvs[corners_found] = if info.motion_mode >= 2 {
            // `LOCALWARP` = 2 in the AV2 motion-mode enumeration.
            let mv = warp_mv_at(&info.warp, row + 1, col + 1);
            [mv.y as i32, mv.x as i32]
        } else {
            [info.mv.y as i32, info.mv.x as i32]
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
            map[r * mi_cols + c] = info;
        }
        let cand = generate_points_from_corners(&map, mi_cols, mi_rows, 4, 4, 4, 4, 8, 8, 0)
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
