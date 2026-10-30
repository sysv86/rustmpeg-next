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

//! AV2 normative Wedge Masks and Mask Blending (AV2 Spec §5.20.7.18, §7.13.3.27, §7.13.3.30).
//!
//! Wedge prediction splits a rectangular or square block into two regions along an arbitrary
//! angled line, using pre-computed smooth transition masks (weights 0..64) with 20 angles
//! and 68 codebook index offsets.

use crate::av2::intops::iclip;
use crate::av2::levels::BlockSize;
use crate::av2::pixel::{BitDepth, Pixel};
use crate::av2::tables::BLOCK_DIMENSIONS;
use std::sync::LazyLock;

/// Number of discrete wedge directions.
pub const N_WEDGE_DIRECTIONS: usize = 20;

/// Number of entries in the AV2 wedge codebook.
pub const WEDGE_TYPES: usize = 68;

/// Master mask dimensions (128x128).
pub const MASK_MASTER_SIZE: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct WedgeCode {
    pub direction: u8,
    pub x_offset: u8,
    pub y_offset: u8,
}

/// Normative 68-entry wedge codebook (direction, x_offset, y_offset).
pub const WEDGE_CODEBOOK_16: [WedgeCode; 68] = [
    WedgeCode {
        direction: 0,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCode {
        direction: 0,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 0,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCode {
        direction: 1,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 1,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCode {
        direction: 1,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 1,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 2,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCode {
        direction: 3,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 3,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCode {
        direction: 3,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 3,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCode {
        direction: 4,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 4,
        x_offset: 4,
        y_offset: 3,
    },
    WedgeCode {
        direction: 4,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCode {
        direction: 4,
        x_offset: 4,
        y_offset: 1,
    },
    WedgeCode {
        direction: 5,
        x_offset: 4,
        y_offset: 3,
    },
    WedgeCode {
        direction: 5,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCode {
        direction: 5,
        x_offset: 4,
        y_offset: 1,
    },
    WedgeCode {
        direction: 6,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 6,
        x_offset: 4,
        y_offset: 3,
    },
    WedgeCode {
        direction: 6,
        x_offset: 4,
        y_offset: 2,
    },
    WedgeCode {
        direction: 6,
        x_offset: 4,
        y_offset: 1,
    },
    WedgeCode {
        direction: 7,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 7,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCode {
        direction: 7,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 7,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCode {
        direction: 8,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 8,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCode {
        direction: 8,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 8,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCode {
        direction: 9,
        x_offset: 4,
        y_offset: 4,
    },
    WedgeCode {
        direction: 9,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCode {
        direction: 9,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 9,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCode {
        direction: 10,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCode {
        direction: 10,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 10,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCode {
        direction: 11,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCode {
        direction: 11,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 11,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCode {
        direction: 12,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCode {
        direction: 12,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 12,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCode {
        direction: 13,
        x_offset: 3,
        y_offset: 4,
    },
    WedgeCode {
        direction: 13,
        x_offset: 2,
        y_offset: 4,
    },
    WedgeCode {
        direction: 13,
        x_offset: 1,
        y_offset: 4,
    },
    WedgeCode {
        direction: 14,
        x_offset: 4,
        y_offset: 5,
    },
    WedgeCode {
        direction: 14,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCode {
        direction: 14,
        x_offset: 4,
        y_offset: 7,
    },
    WedgeCode {
        direction: 15,
        x_offset: 4,
        y_offset: 5,
    },
    WedgeCode {
        direction: 15,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCode {
        direction: 15,
        x_offset: 4,
        y_offset: 7,
    },
    WedgeCode {
        direction: 16,
        x_offset: 4,
        y_offset: 5,
    },
    WedgeCode {
        direction: 16,
        x_offset: 4,
        y_offset: 6,
    },
    WedgeCode {
        direction: 16,
        x_offset: 4,
        y_offset: 7,
    },
    WedgeCode {
        direction: 17,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCode {
        direction: 17,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 17,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCode {
        direction: 18,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCode {
        direction: 18,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 18,
        x_offset: 7,
        y_offset: 4,
    },
    WedgeCode {
        direction: 19,
        x_offset: 5,
        y_offset: 4,
    },
    WedgeCode {
        direction: 19,
        x_offset: 6,
        y_offset: 4,
    },
    WedgeCode {
        direction: 19,
        x_offset: 7,
        y_offset: 4,
    },
];

/// Normative lookup from `(wedge_angle, wedge_dist)` to `WedgeIndex` (AV2 Spec §5.20.7.18).
pub const WEDGE_ANGLE_DIST_TO_IDX: [[i8; 4]; 20] = [
    [-1, 0, 1, 2],
    [3, 4, 5, 6],
    [7, 8, 9, 10],
    [11, 12, 13, 14],
    [15, 16, 17, 18],
    [-1, 19, 20, 21],
    [22, 23, 24, 25],
    [26, 27, 28, 29],
    [30, 31, 32, 33],
    [34, 35, 36, 37],
    [-1, 38, 39, 40],
    [-1, 41, 42, 43],
    [-1, 44, 45, 46],
    [-1, 47, 48, 49],
    [-1, 50, 51, 52],
    [-1, 53, 54, 55],
    [-1, 56, 57, 58],
    [-1, 59, 60, 61],
    [-1, 62, 63, 64],
    [-1, 65, 66, 67],
];

/// Look up table from `wedge_idx` (0..67) to `(quad, angle, dist)`.
pub const WEDGE_IDX_TO_ANGLE_DIST: [(u8, u8, u8); 68] = {
    let mut table = [(0u8, 0u8, 0u8); 68];
    let mut a = 0usize;
    while a < 20 {
        let quad = (a / 5) as u8;
        let angle = (a % 5) as u8;
        let mut d = 0usize;
        while d < 4 {
            let idx = WEDGE_ANGLE_DIST_TO_IDX[a][d];
            if idx >= 0 && (idx as usize) < 68 {
                table[idx as usize] = (quad, angle, d as u8);
            }
            d += 1;
        }
        a += 1;
    }
    table
};

const COS_LUT: [i8; 20] = [
    4, 4, 4, 2, 2, 0, -2, -2, -4, -4, -4, -4, -4, -2, -2, 0, 2, 2, 4, 4,
];
const SIN_LUT: [i8; 20] = [
    0, -1, -2, -2, -4, -4, -4, -2, -2, -1, 0, 1, 2, 2, 4, 4, 4, 2, 2, 1,
];
const WEIGHT: [u8; 29] = [
    8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
];

/// Generates a 128x128 master wedge mask according to AV2 Spec §7.13.3.27.
pub fn gen_master_mask(mul: i32, wd: usize) -> [u8; 128 * 128] {
    let mut master = [0u8; 128 * 128];
    let s = (SIN_LUT[wd] as i32) * mul;
    let c = (COS_LUT[wd] as i32) * mul;
    for y in 0..128 {
        let dy = (2 * (y as i32) - 127) * s;
        let row = y * 128;
        for x in 0..128 {
            let d = iclip((2 * (x as i32) - 127) * c + dy, -28, 28);
            master[row + x] = 4
                * (if d >= 0 {
                    16 - WEIGHT[d as usize]
                } else {
                    WEIGHT[(-d) as usize]
                });
        }
    }
    master
}

/// Slices a `(w, h)` mask from `master` using block offsets `(x_off, y_off)`.
pub fn copy2d_mask(w: usize, h: usize, x_off: usize, y_off: usize, master: &[u8]) -> Vec<u8> {
    let w8 = w / 8;
    let h8 = h / 8;
    let src_start_y = 64 - y_off * h8;
    let src_start_x = 64 - x_off * w8;
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let src_row = (src_start_y + y) * 128 + src_start_x;
        let dst_row = y * w;
        out[dst_row..dst_row + w].copy_from_slice(&master[src_row..src_row + w]);
    }
    out
}

/// Horizontally subsamples a mask by 2 (4:2:2 chroma).
pub fn subsample_mask_422(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let cw = w / 2;
    let mut out = vec![0u8; cw * h];
    for y in 0..h {
        let s_row = y * w;
        let d_row = y * cw;
        for x in 0..cw {
            out[d_row + x] =
                ((src[s_row + 2 * x] as u16 + src[s_row + 2 * x + 1] as u16 + 1) >> 1) as u8;
        }
    }
    out
}

/// Subsamples a mask by 2x2 (4:2:0 chroma).
pub fn subsample_mask_420(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let cw = w / 2;
    let ch = h / 2;
    let mut out = vec![0u8; cw * ch];
    for y in 0..ch {
        let s_row0 = 2 * y * w;
        let s_row1 = (2 * y + 1) * w;
        let d_row = y * cw;
        for x in 0..cw {
            let sum = src[s_row0 + 2 * x] as u32
                + src[s_row0 + 2 * x + 1] as u32
                + src[s_row1 + 2 * x] as u32
                + src[s_row1 + 2 * x + 1] as u32;
            out[d_row + x] = ((sum + 2) >> 2) as u8;
        }
    }
    out
}

/// Precomputed tables for all supported wedge block sizes and layouts.
pub struct WedgeTables {
    /// Masks indexed by: `bs_idx` (0..16), `widx` (0..67), `subsample_idx` (0: 444, 1: 422, 2: 420).
    masks: Vec<Vec<[Vec<u8>; 3]>>,
}

impl WedgeTables {
    fn new() -> Self {
        // Supported block sizes for wedge: min(w, h) >= 8 and max(w, h) <= 32
        // Block sizes: 8x8, 8x16, 16x8, 16x16, 8x32, 32x8, 16x32, 32x16, 32x32, 8x64, 64x8, 16x64, 64x16, 32x64, 64x32, 64x64
        let n_bs = 22; // N_BS_SIZES
        let mut masks = vec![Vec::new(); n_bs];

        let mut masters_sharp = [[0u8; 128 * 128]; 20];
        let mut masters_soft = [[0u8; 128 * 128]; 20];
        for wd in 0..20 {
            masters_sharp[wd] = gen_master_mask(2, wd);
            masters_soft[wd] = gen_master_mask(1, wd);
        }

        for bs_idx in 0..n_bs {
            let b_dim = BLOCK_DIMENSIONS[bs_idx];
            let w = b_dim[0] as usize * 4;
            let h = b_dim[1] as usize * 4;
            if w < 8 || h < 8 || w > 64 || h > 64 {
                continue;
            }
            let is_sharp = w <= 16 && h <= 16;
            let masters = if is_sharp {
                &masters_sharp
            } else {
                &masters_soft
            };

            let mut bs_masks = Vec::with_capacity(68);
            for widx in 0..68 {
                let code = &WEDGE_CODEBOOK_16[widx];
                let m444 = copy2d_mask(
                    w,
                    h,
                    code.x_offset as usize,
                    code.y_offset as usize,
                    &masters[code.direction as usize],
                );
                let m422 = subsample_mask_422(&m444, w, h);
                let m420 = subsample_mask_420(&m444, w, h);
                bs_masks.push([m444, m422, m420]);
            }
            masks[bs_idx] = bs_masks;
        }

        Self { masks }
    }

    /// Retrieves the wedge mask for block size `bs`, wedge index `widx`, sign, and subsampling.
    #[inline]
    pub fn get_mask(&self, bs: BlockSize, widx: usize, ss_x: usize, ss_y: usize) -> Option<&[u8]> {
        let bs_idx = bs as usize;
        let sub_idx = if ss_x == 0 && ss_y == 0 {
            0
        } else if ss_x == 1 && ss_y == 0 {
            1
        } else {
            2
        };
        self.masks
            .get(bs_idx)?
            .get(widx)
            .map(|m| m[sub_idx].as_slice())
    }
}

/// Global lazily initialized wedge mask tables.
pub static WEDGE_TABLES: LazyLock<WedgeTables> = LazyLock::new(WedgeTables::new);

/// Retrieves the normative wedge mask for a block, inverted if `sign == true`.
pub fn get_normative_wedge_mask(
    bs: BlockSize,
    widx: usize,
    sign: bool,
    ss_x: usize,
    ss_y: usize,
    out: &mut [u8],
) -> bool {
    if let Some(mask) = WEDGE_TABLES.get_mask(bs, widx, ss_x, ss_y)
        && out.len() >= mask.len()
    {
        if sign {
            for i in 0..mask.len() {
                out[i] = 64 - mask[i];
            }
        } else {
            out[..mask.len()].copy_from_slice(mask);
        }
        return true;
    }
    false
}

/// Blends two motion-compensated predictions using a 6-bit mask (0..=64) per AV2 Spec §7.13.3.30.
///
/// `dst[x] = (mask[x] * p0[x] + (64 - mask[x]) * p1[x] + 32) >> 6`.
pub fn blend_wedge_mask<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    p0: &[BD::Pixel],
    p1: &[BD::Pixel],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    for y in 0..h {
        let d_row = y * dst_stride;
        let s_row = y * w;
        for x in 0..w {
            let m = mask[s_row + x] as i32;
            let val0 = p0[s_row + x].to_i32();
            let val1 = p1[s_row + x].to_i32();
            let blended = (m * val0 + (64 - m) * val1 + 32) >> 6;
            dst[d_row + x] = bd.clip_pixel(blended);
        }
    }
}

/// Generates difference-weighted blending mask from two predictions per AV2 Spec §7.13.3.28.
pub fn compute_difference_mask<BD: BitDepth>(
    bd: BD,
    p0: &[BD::Pixel],
    p1: &[BD::Pixel],
    w: usize,
    h: usize,
    mask_sign: bool,
    out_mask: &mut [u8],
) {
    let shift = bd.bitdepth().saturating_sub(8) as usize;
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let diff = (p0[row + x].to_i32() - p1[row + x].to_i32()).abs();
            let diff_rounded = (diff + (1 << shift >> 1)) >> shift;
            let m = iclip(38 + diff_rounded / 16, 0, 64) as u8;
            out_mask[row + x] = if mask_sign { 64 - m } else { m };
        }
    }
}

/// Computes difference-weighted blending masks from intermediate predictions per AV2 Spec §7.13.3.28 / dav2d `w_mask_c`.
pub fn compute_diff_wtd_mask_intermediate<BD: BitDepth>(
    bd: BD,
    tmp1: &[i32],
    tmp2: &[i32],
    w: usize,
    h: usize,
    mask_sign: bool,
    ss_hor: usize,
    ss_ver: usize,
    luma_mask: &mut [u8],
    chroma_mask: &mut [u8],
) {
    let intermediate_bits = if bd.bitdepth() <= 10 { 4 } else { 2 };
    let bitdepth = bd.bitdepth() as usize;
    let mask_sh = bitdepth + intermediate_bits - 4;
    let mask_rnd = 1 << (mask_sh - 5);
    let sign = mask_sign as i32;

    for y in 0..h {
        let r_in = y * w;
        for x in 0..w {
            let tmpdiff = tmp1[r_in + x] - tmp2[r_in + x];
            let m = (38 + ((tmpdiff.abs() + mask_rnd) >> mask_sh)).min(64) as u8;
            luma_mask[r_in + x] = m;
        }
    }

    if ss_hor != 0 || ss_ver != 0 {
        let cw = (w >> ss_hor).max(1);
        let ch = (h >> ss_ver).max(1);
        if ss_hor == 1 && ss_ver == 1 {
            for cy in 0..ch {
                let ly0 = 2 * cy;
                let ly1 = 2 * cy + 1;
                let d_row = cy * cw;
                for cx in 0..cw {
                    let lx0 = 2 * cx;
                    let lx1 = 2 * cx + 1;
                    let m0 = luma_mask[ly0 * w + lx0] as i32;
                    let n0 = luma_mask[ly0 * w + lx1] as i32;
                    let m1 = luma_mask[ly1 * w + lx0] as i32;
                    let n1 = luma_mask[ly1 * w + lx1] as i32;
                    chroma_mask[d_row + cx] =
                        ((m0 + n0 + m1 + n1 + 2 - sign) >> 2).clamp(0, 64) as u8;
                }
            }
        } else if ss_hor == 1 && ss_ver == 0 {
            for cy in 0..ch {
                let d_row = cy * cw;
                let ly = cy;
                for cx in 0..cw {
                    let lx0 = 2 * cx;
                    let lx1 = 2 * cx + 1;
                    let m = luma_mask[ly * w + lx0] as i32;
                    let n = luma_mask[ly * w + lx1] as i32;
                    chroma_mask[d_row + cx] = ((m + n + 1 - sign) >> 1).clamp(0, 64) as u8;
                }
            }
        } else {
            let len = cw * ch;
            chroma_mask[..len].copy_from_slice(&luma_mask[..len]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::pixel::BitDepth8;

    #[test]
    fn test_wedge_codebook_and_angle_dist() {
        for (widx, &(quad, angle, dist)) in WEDGE_IDX_TO_ANGLE_DIST.iter().enumerate() {
            assert!(quad < 4);
            assert!(angle < 5);
            assert!(dist < 4);
            let dir = (quad * 5 + angle) as usize;
            let mapped_idx = WEDGE_ANGLE_DIST_TO_IDX[dir][dist as usize];
            assert_eq!(mapped_idx as usize, widx);
        }
    }

    #[test]
    fn test_wedge_masks_generation() {
        let mask = WEDGE_TABLES
            .get_mask(BlockSize::Bs16x16, 0, 0, 0)
            .expect("16x16 wedge mask 0 must exist");
        assert_eq!(mask.len(), 16 * 16);
        // All mask values must be between 0 and 64 inclusive
        for &val in mask {
            assert!(val <= 64);
        }

        // Subsampled 4:2:0 mask for 16x16 is 8x8 = 64
        let mask_uv = WEDGE_TABLES
            .get_mask(BlockSize::Bs16x16, 0, 1, 1)
            .expect("16x16 4:2:0 mask must exist");
        assert_eq!(mask_uv.len(), 8 * 8);
        for &val in mask_uv {
            assert!(val <= 64);
        }
    }

    #[test]
    fn test_wedge_blend_endpoints() {
        let w = 8;
        let h = 8;
        let p0 = vec![100u8; w * h];
        let p1 = vec![200u8; w * h];
        let mut mask = vec![64u8; w * h];
        let mut dst = vec![0u8; w * h];

        // Mask == 64 should reproduce p0
        blend_wedge_mask(BitDepth8, &mut dst, w, &p0, &p1, w, h, &mask);
        assert_eq!(&dst[..], &p0[..]);

        // Mask == 0 should reproduce p1
        mask.fill(0);
        blend_wedge_mask(BitDepth8, &mut dst, w, &p0, &p1, w, h, &mask);
        assert_eq!(&dst[..], &p1[..]);

        // Mask == 32 should produce average (150)
        mask.fill(32);
        blend_wedge_mask(BitDepth8, &mut dst, w, &p0, &p1, w, h, &mask);
        assert_eq!(dst[0], 150);
    }
}
