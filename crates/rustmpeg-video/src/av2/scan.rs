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

//! Clean-room compile-time generator for all 25 AV2 2D anti-diagonal zig-zag scan tables
//! (`4x4` up to `64x64`) and per-column maximum EOB bounds (`LAST_EOB_PER_COL`),
//! derived directly from the AV2 Specification v1.0.0 mathematical scan definition.

use crate::av2::levels::N_RECT_TX_SIZES;

/// Generates a 2D anti-diagonal scan table for a transform block of width `W` and height `H`
/// stored in column-major order (`index = col * H + row`), ordered by increasing diagonal
/// `d = row + col` and within each diagonal by increasing column `col` (top-right to bottom-left).
pub const fn generate_antidiag_scan<const W: usize, const H: usize, const N: usize>() -> [u16; N] {
    let mut out = [0u16; N];
    let mut k = 0usize;
    let max_d = (W - 1) + (H - 1);
    let mut d = 0usize;
    while d <= max_d {
        let c_start = if d >= H { d - (H - 1) } else { 0 };
        let c_end = if d < W { d } else { W - 1 };
        let mut c = c_start;
        while c <= c_end {
            let r = d - c;
            out[k] = (c * H + r) as u16;
            k += 1;
            c += 1;
        }
        d += 1;
    }
    out
}

pub const SCAN_4X4: [u16; 16] = generate_antidiag_scan::<4, 4, 16>();
pub const SCAN_4X8: [u16; 32] = generate_antidiag_scan::<4, 8, 32>();
pub const SCAN_4X16: [u16; 64] = generate_antidiag_scan::<4, 16, 64>();
pub const SCAN_4X32: [u16; 128] = generate_antidiag_scan::<4, 32, 128>();
pub const SCAN_8X4: [u16; 32] = generate_antidiag_scan::<8, 4, 32>();
pub const SCAN_8X8: [u16; 64] = generate_antidiag_scan::<8, 8, 64>();
pub const SCAN_8X16: [u16; 128] = generate_antidiag_scan::<8, 16, 128>();
pub const SCAN_8X32: [u16; 256] = generate_antidiag_scan::<8, 32, 256>();
pub const SCAN_8X64: [u16; 256] = generate_antidiag_scan::<8, 32, 256>();
pub const SCAN_16X4: [u16; 64] = generate_antidiag_scan::<16, 4, 64>();
pub const SCAN_16X8: [u16; 128] = generate_antidiag_scan::<16, 8, 128>();
pub const SCAN_16X16: [u16; 256] = generate_antidiag_scan::<16, 16, 256>();
pub const SCAN_16X32: [u16; 512] = generate_antidiag_scan::<16, 32, 512>();
pub const SCAN_16X64: [u16; 512] = generate_antidiag_scan::<16, 32, 512>();
pub const SCAN_32X4: [u16; 128] = generate_antidiag_scan::<32, 4, 128>();
pub const SCAN_32X8: [u16; 256] = generate_antidiag_scan::<32, 8, 256>();
pub const SCAN_32X16: [u16; 512] = generate_antidiag_scan::<32, 16, 512>();
pub const SCAN_32X32: [u16; 1024] = generate_antidiag_scan::<32, 32, 1024>();
pub const SCAN_32X64: [u16; 1024] = generate_antidiag_scan::<32, 32, 1024>();
pub const SCAN_64X8: [u16; 256] = generate_antidiag_scan::<32, 8, 256>();
pub const SCAN_64X16: [u16; 512] = generate_antidiag_scan::<32, 16, 512>();
pub const SCAN_64X32: [u16; 1024] = generate_antidiag_scan::<32, 32, 1024>();
pub const SCAN_64X64: [u16; 1024] = generate_antidiag_scan::<32, 32, 1024>();

/// Indexed by `RectTxfmSize as usize` (`0..25`).
/// Note: In AV2, 64-point transforms zero out high frequencies beyond the lowest 32x32 sub-block,
/// so their effective coded coefficient scan covers up to `min(W, 32) x min(H, 32)`.
pub const SCANS: [&[u16]; N_RECT_TX_SIZES] = [
    &SCAN_4X4,   // Tx4x4
    &SCAN_8X8,   // Tx8x8
    &SCAN_16X16, // Tx16x16
    &SCAN_32X32, // Tx32x32
    &SCAN_64X64, // Tx64x64
    &SCAN_4X8,   // Rtx4x8
    &SCAN_8X4,   // Rtx8x4
    &SCAN_8X16,  // Rtx8x16
    &SCAN_16X8,  // Rtx16x8
    &SCAN_16X32, // Rtx16x32
    &SCAN_32X16, // Rtx32x16
    &SCAN_32X64, // Rtx32x64
    &SCAN_64X32, // Rtx64x32
    &SCAN_4X16,  // Rtx4x16
    &SCAN_16X4,  // Rtx16x4
    &SCAN_8X32,  // Rtx8x32
    &SCAN_32X8,  // Rtx32x8
    &SCAN_16X64, // Rtx16x64
    &SCAN_64X16, // Rtx64x16
    &SCAN_4X32,  // Rtx4x32
    &SCAN_32X4,  // Rtx32x4
    &SCAN_8X64,  // Rtx8x64
    &SCAN_64X8,  // Rtx64x8
    &SCAN_4X32,  // Rtx4x64 (effective 4x32 coded region)
    &SCAN_32X4,  // Rtx64x4 (effective 32x4 coded region)
];

/// Computes the maximum scan index in each column for a `W x H` coded transform region.
const fn generate_last_eob_col<const W: usize, const H: usize>() -> [u16; W] {
    let n = W * H;
    let mut scan = [0u16; 1024];
    let mut k = 0usize;
    let max_d = (W - 1) + (H - 1);
    let mut d = 0usize;
    while d <= max_d {
        let c_start = if d >= H { d - (H - 1) } else { 0 };
        let c_end = if d < W { d } else { W - 1 };
        let mut c = c_start;
        while c <= c_end {
            let r = d - c;
            scan[k] = (c * H + r) as u16;
            k += 1;
            c += 1;
        }
        d += 1;
    }
    let mut last_col = [0u16; W];
    let mut i = 0usize;
    while i < n {
        let rc = scan[i] as usize;
        let col = rc / H;
        if col < W {
            last_col[col] = (i + 1) as u16;
        }
        i += 1;
    }
    last_col
}

/// Per-column EOB bounds for 1D horizontal/vertical transform classes (`TxClass::H` / `TxClass::V`).
pub struct LastEob {
    pub eob_4x4: [u16; 4],
    pub eob_4x8: [u16; 4],
    pub eob_4x16: [u16; 4],
    pub eob_8x4: [u16; 8],
    pub eob_8x8: [u16; 8],
    pub eob_8x16: [u16; 8],
    pub eob_8x32: [u16; 8],
    pub eob_16x4: [u16; 16],
    pub eob_16x8: [u16; 16],
    pub eob_16x16: [u16; 16],
    pub eob_16x32: [u16; 16],
    pub eob_32x8: [u16; 32],
    pub eob_32x16: [u16; 32],
    pub eob_32x32: [u16; 32],
}

pub const LAST_EOB_PER_COL: LastEob = LastEob {
    eob_4x4: generate_last_eob_col::<4, 4>(),
    eob_4x8: generate_last_eob_col::<4, 8>(),
    eob_4x16: generate_last_eob_col::<4, 16>(),
    eob_8x4: generate_last_eob_col::<8, 4>(),
    eob_8x8: generate_last_eob_col::<8, 8>(),
    eob_8x16: generate_last_eob_col::<8, 16>(),
    eob_8x32: generate_last_eob_col::<8, 32>(),
    eob_16x4: generate_last_eob_col::<16, 4>(),
    eob_16x8: generate_last_eob_col::<16, 8>(),
    eob_16x16: generate_last_eob_col::<16, 16>(),
    eob_16x32: generate_last_eob_col::<16, 32>(),
    eob_32x8: generate_last_eob_col::<32, 8>(),
    eob_32x16: generate_last_eob_col::<32, 16>(),
    eob_32x32: generate_last_eob_col::<32, 32>(),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_antidiag_scan_4x4() {
        assert_eq!(
            SCAN_4X4,
            [0, 1, 4, 2, 5, 8, 3, 6, 9, 12, 7, 10, 13, 11, 14, 15]
        );
        assert_eq!(LAST_EOB_PER_COL.eob_4x4, [7, 11, 14, 16]);
    }
}
