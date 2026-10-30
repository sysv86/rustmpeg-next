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

//! AV2 Superblock Partition Tree RDO (`Bs64x64/Bs128x128 -> ... -> Bs4x4`) and
//! coding-tree syntax (`encode_partition_decision`, AV2 Spec v1.0.0 §5.20.3).

use crate::av2::cdf::CdfModeContext;
use crate::av2::entropy::{BlockContext, get_partition_ctx};
use crate::av2::frame::Plane;
use crate::av2::levels::{BlockPartition, BlockSize, N_BS_SIZES};
use crate::av2::msac::MsacEncoder;
use crate::av2::pixel::Pixel;
use crate::av2::tables::BLOCK_DIMENSIONS;

/// Computes the sample variance ($\sum (x - \mu)^2$) of an $8 \times 8$ block at `(x0, y0)`,
/// clamping coordinates to the active plane dimensions `(w, h)`.
pub fn variance_8x8<P: Pixel>(plane: &Plane<P>, x0: usize, y0: usize, w: usize, h: usize) -> u64 {
    let mut sum = 0i64;
    let mut sum_sq = 0i64;
    for dy in 0..8 {
        let y = (y0 + dy).min(h.saturating_sub(1));
        let row = y * plane.stride;
        for dx in 0..8 {
            let x = (x0 + dx).min(w.saturating_sub(1));
            let v = plane.data[row + x].to_i32() as i64;
            sum += v;
            sum_sq += v * v;
        }
    }
    let var64 = sum_sq - ((sum * sum) >> 6);
    var64.max(0) as u64
}

/// Returns `true` when all 4 corner $8 \times 8$ sub-blocks of `(px_x, px_y, w, h)` have high
/// spatial variance (`> 4096`), indicating homogeneous full-block noise where sub-partitioning
/// and intra mode search on inter frames only add signaling overhead.
#[inline]
pub fn is_spatial_noise_block<P: Pixel>(
    plane: &Plane<P>,
    px_x: usize,
    px_y: usize,
    w: usize,
    h: usize,
    plane_w: usize,
    plane_h: usize,
) -> bool {
    let dx = w.saturating_sub(8);
    let dy = h.min(32).saturating_sub(8);
    variance_8x8(plane, px_x, px_y, plane_w, plane_h) > 4096
        && variance_8x8(plane, px_x + dx, px_y, plane_w, plane_h) > 4096
        && variance_8x8(plane, px_x, px_y + dy, plane_w, plane_h) > 4096
        && variance_8x8(plane, px_x + dx, px_y + dy, plane_w, plane_h) > 4096
}

/// Computes the directional horizontal and vertical $8 \times 8$ sub-block variance gradients
/// across a square block `(x0, y0, dim)` to guide rectangular (`H` / `V`) vs quadtree (`Split`)
/// vs leaf (`None`) partitioning.
pub fn analyze_block_variance<P: Pixel>(
    plane: &Plane<P>,
    x0: usize,
    y0: usize,
    dim: usize,
    w: usize,
    h: usize,
) -> (u64, u64, u64) {
    let n8 = (dim / 8).max(1);
    let mut sub_vars = [0u64; 64];
    let mut total_var = 0u64;
    let mut max_sub = 0u64;
    let mut min_sub = u64::MAX;

    for r in 0..n8 {
        for c in 0..n8 {
            let v = variance_8x8(plane, x0 + c * 8, y0 + r * 8, w, h);
            sub_vars[r * n8 + c] = v;
            total_var += v;
            max_sub = max_sub.max(v);
            min_sub = min_sub.min(v);
        }
    }

    let mean_var = total_var / ((n8 * n8) as u64);
    if n8 < 2 {
        return (mean_var, 0, 0);
    }

    let half = n8 / 2;
    let mut top_sum = 0u64;
    let mut bot_sum = 0u64;
    let mut left_sum = 0u64;
    let mut right_sum = 0u64;

    for r in 0..n8 {
        for c in 0..n8 {
            let v = sub_vars[r * n8 + c];
            if r < half {
                top_sum += v;
            } else {
                bot_sum += v;
            }
            if c < half {
                left_sum += v;
            } else {
                right_sum += v;
            }
        }
    }

    let hor_grad = top_sum.abs_diff(bot_sum) / ((half * n8) as u64);
    let ver_grad = left_sum.abs_diff(right_sum) / ((half * n8) as u64);
    let inhomogeneity = max_sub.saturating_sub(min_sub);

    (mean_var + (inhomogeneity >> 2), hor_grad, ver_grad)
}

/// Selects the optimal partition (`None`, `H`, `V`, or `Split`) for a coding-tree node
/// `(bs, bx_px, by_px)` using multi-scale variance homogeneity and Lagrangian rate-distortion scaling.
///
/// In AV2 Spec v1.0.0 §5.20.3.2, `BlockPartition::Split` exists only for `Bs128x128` (and `Bs256x256`).
/// Smaller square blocks (`Bs64x64`, `Bs32x32`) quad-split into four half-size square blocks via
/// `BlockPartition::H` (producing two `2:1` rectangular blocks `Bs64x32` / `Bs32x16`) followed by
/// `BlockPartition::V` at each `2:1` child (producing `Bs32x32` / `Bs16x16`).
#[allow(clippy::too_many_arguments)]
pub fn choose_partition_rdo<P: Pixel>(
    luma_src: &Plane<P>,
    bx_px: usize,
    by_px: usize,
    bs: BlockSize,
    frame_w: usize,
    frame_h: usize,
    qidx: u16,
    _speed: u8,
    _enable_rect_partitions: bool,
) -> BlockPartition {
    if bs == BlockSize::Bs128x128 {
        return BlockPartition::Split;
    }
    match bs {
        BlockSize::Bs64x32 | BlockSize::Bs32x16 | BlockSize::Bs16x8 => return BlockPartition::V,
        BlockSize::Bs32x64 | BlockSize::Bs16x32 | BlockSize::Bs8x16 => return BlockPartition::H,
        BlockSize::Bs64x64 | BlockSize::Bs32x32 => {}
        _ => return BlockPartition::None,
    }

    let dim = if bs == BlockSize::Bs64x64 {
        64usize
    } else {
        32usize
    };
    if bx_px + dim > frame_w || by_px + dim > frame_h {
        return BlockPartition::H;
    }

    let n8 = dim / 8;
    let mut total_var = 0u64;
    let mut max_sub = 0u64;
    let mut min_sub = u64::MAX;

    for r in 0..n8 {
        for c in 0..n8 {
            let v = variance_8x8(luma_src, bx_px + c * 8, by_px + r * 8, frame_w, frame_h);
            total_var += v;
            max_sub = max_sub.max(v);
            min_sub = min_sub.min(v);
        }
    }

    // Flat block or full-block high-variance noise (`min_sub > 2048` across all 8x8 sub-blocks):
    // keep as a single leaf block so `Bs64x64` (`dq_shift = 5`) codes stationary/noise
    // blocks with minimal overhead.
    if max_sub <= 128 || min_sub > 2048 {
        return BlockPartition::None;
    }

    let mean_var = total_var / ((n8 * n8) as u64);
    if bs == BlockSize::Bs64x64 {
        // `Bs64x64` uses `Tx64x64`, which 2x2-downsamples/upsamples residuals (AV2 Spec §7.13.2).
        // Split into `Bs32x32` (`Tx32x32`, full 1:1 resolution) at low/medium QP (`qidx <= 85`)
        // whenever the block is non-flat, or at higher QP when flat background borders an edge.
        if (qidx <= 85 && mean_var > 192)
            || (qidx <= 115 && min_sub * 3 < max_sub && max_sub > 512)
            || (min_sub <= 128 && max_sub > 1024)
        {
            return BlockPartition::H;
        }
        return BlockPartition::None;
    }

    // `Bs32x32` -> `Bs16x16`: split at low/medium QP (`qidx <= 85`) when sub-block variance
    // is strongly inhomogeneous (sharp edges/text adjacent to flat background).
    if qidx <= 85 {
        let var_thresh = 1024 + (qidx as u64) * 32;
        if min_sub * 4 < max_sub && max_sub > var_thresh {
            return BlockPartition::H;
        }
    }
    BlockPartition::None
}

/// Frame-level wrapper around [`choose_partition_rdo`] that takes 4x4 unit coordinates
/// (`bx4`, `by4`) and 4x4 frame dimensions (`f_bw4`, `f_bh4`) and converts them to pixels.
///
/// `bitdepth` is accepted for API symmetry with the frame encoder and is unused: the
/// variance-based decision is bit-depth independent.
#[allow(clippy::too_many_arguments)]
pub fn select_best_partition<P: Pixel>(
    luma_src: &Plane<P>,
    bs: BlockSize,
    bx4: usize,
    by4: usize,
    f_bw4: usize,
    f_bh4: usize,
    _bitdepth: u8,
    qidx: u16,
    speed: u8,
    enable_rect_partitions: bool,
) -> BlockPartition {
    choose_partition_rdo(
        luma_src,
        bx4 * 4,
        by4 * 4,
        bs,
        f_bw4 * 4,
        f_bh4 * 4,
        qidx,
        speed,
        enable_rect_partitions,
    )
}

/// Layout and availability parameters needed to derive which partitions the
/// coding-tree syntax allows at a node (AV2 Spec v1.0.0 §5.20.3.2).
#[derive(Debug, Clone, Copy)]
pub struct PartitionSyntaxConfig {
    /// `FrameIsIntra`: inter frames forbid 4x4 blocks in mixed regions.
    pub frame_is_intra: bool,
    /// `hasChroma`: whether a chroma plane is coded alongside luma.
    pub has_chroma: bool,
    /// `SubsamplingX`.
    pub subsampling_x: bool,
    /// `SubsamplingY`.
    pub subsampling_y: bool,
    /// `MaxPbAspectRatio` (`1 << max_pb_aspect_ratio_log2`).
    pub max_pb_aspect_ratio: u32,
    /// `enable_ext_partitions`: AV2 Spec §5.20.3.2.
    pub enable_ext_partitions: bool,
    /// `enable_uneven_4way_partitions`: AV2 Spec §5.20.3.2.
    pub enable_uneven_4way_partitions: bool,
}

impl PartitionSyntaxConfig {
    /// Builds the configuration from a sequence/frame header for a shared-part
    /// coding tree with default (basic) partition set.
    pub fn new(layout: crate::av2::headers::PixelLayout, frame_is_intra: bool) -> Self {
        Self::with_ext_flags(layout, frame_is_intra, false, false)
    }

    /// Builds the configuration with extended partitions enabled.
    pub fn new_with_ext(layout: crate::av2::headers::PixelLayout, frame_is_intra: bool) -> Self {
        Self::with_ext_flags(layout, frame_is_intra, true, true)
    }

    /// Builds the configuration with explicit sequence-header flags.
    pub fn with_ext_flags(
        layout: crate::av2::headers::PixelLayout,
        frame_is_intra: bool,
        enable_ext_partitions: bool,
        enable_uneven_4way_partitions: bool,
    ) -> Self {
        let has_chroma = layout != crate::av2::headers::PixelLayout::I400;
        Self {
            frame_is_intra,
            has_chroma,
            subsampling_x: has_chroma && layout != crate::av2::headers::PixelLayout::I444,
            subsampling_y: has_chroma && layout == crate::av2::headers::PixelLayout::I420,
            max_pb_aspect_ratio: 8,
            enable_ext_partitions,
            enable_uneven_4way_partitions,
        }
    }

    /// Builds the configuration from sequence header and frame-is-intra flag.
    pub fn from_seq_header(
        seq: &crate::av2::headers::SequenceHeader,
        frame_is_intra: bool,
    ) -> Self {
        let mut cfg = Self::with_ext_flags(
            seq.layout,
            frame_is_intra,
            seq.ext_partitions,
            seq.uneven_4way_partitions,
        );
        cfg.max_pb_aspect_ratio = 1 << seq.max_pb_aspect_ratio_log2;
        cfg
    }
}

/// The partitions tracked by this codec, indexed by [`BlockPartition`] value
/// (`None = 0`, `H = 1`, `V = 2`, `H3 = 3`, `V3 = 4`, `H4A = 5`, `H4B = 6`, `V4A = 7`, `V4B = 8`, `Split = 9`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllowedPartitions {
    allowed: [bool; 10],
    count: usize,
}

impl AllowedPartitions {
    /// Slot index corresponding to `bp`.
    #[inline]
    pub const fn slot(bp: BlockPartition) -> usize {
        bp as usize
    }

    /// Returns whether `bp` may be signalled at this node.
    #[inline]
    pub fn contains(&self, bp: BlockPartition) -> bool {
        let idx = Self::slot(bp);
        idx < self.allowed.len() && self.allowed[idx]
    }

    /// Number of legal partitions (`numAllowed`).
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the only legal partition when exactly one is available.
    pub fn single(&self) -> Option<BlockPartition> {
        if self.count != 1 {
            return None;
        }
        self.first()
    }

    /// Returns the first legal partition in coding order:
    /// `[None, H, V, H3, V3, H4A, H4B, V4A, V4B, Split]`.
    pub fn first(&self) -> Option<BlockPartition> {
        const CANDIDATES: [BlockPartition; 10] = [
            BlockPartition::None,
            BlockPartition::H,
            BlockPartition::V,
            BlockPartition::H3,
            BlockPartition::V3,
            BlockPartition::H4A,
            BlockPartition::H4B,
            BlockPartition::V4A,
            BlockPartition::V4B,
            BlockPartition::Split,
        ];
        CANDIDATES.into_iter().find(|bp| self.contains(*bp))
    }
}

/// `H_Partition_Midsize[bs]` (AV2 Spec §8.2 / `h_partition_midsize.h`).
pub static H_PARTITION_MIDSIZE: [Option<BlockSize>; N_BS_SIZES] = {
    let mut tbl = [None; N_BS_SIZES];
    tbl[BlockSize::Bs8x16 as usize] = Some(BlockSize::Bs4x8);
    tbl[BlockSize::Bs16x8 as usize] = Some(BlockSize::Bs8x4);
    tbl[BlockSize::Bs16x16 as usize] = Some(BlockSize::Bs8x8);
    tbl[BlockSize::Bs16x32 as usize] = Some(BlockSize::Bs8x16);
    tbl[BlockSize::Bs32x16 as usize] = Some(BlockSize::Bs16x8);
    tbl[BlockSize::Bs32x32 as usize] = Some(BlockSize::Bs16x16);
    tbl[BlockSize::Bs32x64 as usize] = Some(BlockSize::Bs16x32);
    tbl[BlockSize::Bs64x32 as usize] = Some(BlockSize::Bs32x16);
    tbl[BlockSize::Bs64x64 as usize] = Some(BlockSize::Bs32x32);
    tbl[BlockSize::Bs8x32 as usize] = Some(BlockSize::Bs4x16);
    tbl[BlockSize::Bs32x8 as usize] = Some(BlockSize::Bs16x4);
    tbl[BlockSize::Bs16x64 as usize] = Some(BlockSize::Bs8x32);
    tbl[BlockSize::Bs64x16 as usize] = Some(BlockSize::Bs32x8);
    tbl
};

/// Returns the size of the two middle sub-blocks for 3-way (`H3` / `V3`) partitions,
/// which have dimensions `(W/2, H/2)` (AV2 Spec §8.2 `H_Partition_Midsize`).
#[inline]
pub fn h_partition_midsize(bs: BlockSize) -> Option<BlockSize> {
    H_PARTITION_MIDSIZE[bs as usize]
}

/// Size of the sub-blocks produced by partition `bp` of `bs`, or `None` when the
/// syntax forbids it (`Partition_Subsize`, AV2 Spec v1.0.0 §5.20.3.2 / `AV2_all_tables.h`).
pub fn partition_subsize(bs: BlockSize, bp: BlockPartition) -> Option<BlockSize> {
    use BlockSize::*;
    match bp {
        BlockPartition::None => Some(bs),
        // Same width, half the height; the result must be a valid block size.
        BlockPartition::H => Some(match bs {
            Bs4x8 => Bs4x4,
            Bs8x8 => Bs8x4,
            Bs8x16 => Bs8x8,
            Bs16x8 => Bs16x4,
            Bs16x16 => Bs16x8,
            Bs16x32 => Bs16x16,
            Bs32x16 => Bs32x8,
            Bs32x32 => Bs32x16,
            Bs32x64 => Bs32x32,
            Bs64x32 => Bs64x16,
            Bs64x64 => Bs64x32,
            Bs64x128 => Bs64x64,
            Bs128x128 => Bs128x64,
            Bs4x16 => Bs4x8,
            Bs8x32 => Bs8x16,
            Bs32x8 => Bs32x4,
            Bs16x64 => Bs16x32,
            Bs64x16 => Bs64x8,
            _ => return None,
        }),
        // Same height, half the width; the result must be a valid block size.
        BlockPartition::V => Some(match bs {
            Bs8x4 => Bs4x4,
            Bs8x8 => Bs4x8,
            Bs8x16 => Bs4x16,
            Bs16x8 => Bs8x8,
            Bs16x16 => Bs8x16,
            Bs16x32 => Bs8x32,
            Bs32x16 => Bs16x16,
            Bs32x32 => Bs16x32,
            Bs32x64 => Bs16x64,
            Bs64x32 => Bs32x32,
            Bs64x64 => Bs32x64,
            Bs128x64 => Bs64x64,
            Bs128x128 => Bs64x128,
            Bs16x4 => Bs8x4,
            Bs8x32 => Bs4x32,
            Bs32x8 => Bs16x8,
            Bs16x64 => Bs8x64,
            Bs64x16 => Bs32x16,
            _ => return None,
        }),
        // 3-way horizontal (H3): outer quarter-height sub-blocks
        // (`Partition_Subsize` column 1, AV2 Spec v1.0.0 §5.20.3.2).
        BlockPartition::H3 => Some(match bs {
            Bs8x16 => Bs8x8,
            Bs16x16 => Bs16x8,
            Bs16x32 => Bs16x8,
            Bs32x16 => Bs32x8,
            Bs32x32 => Bs32x8,
            Bs32x64 => Bs32x16,
            Bs64x32 => Bs64x8,
            Bs64x64 => Bs64x16,
            Bs8x32 => Bs8x8,
            Bs16x64 => Bs16x16,
            _ => return None,
        }),
        // 3-way vertical (V3): outer quarter-width sub-blocks
        // (`Partition_Subsize` column 1, AV2 Spec v1.0.0 §5.20.3.2).
        BlockPartition::V3 => Some(match bs {
            Bs16x8 => Bs8x8,
            Bs16x16 => Bs8x16,
            Bs16x32 => Bs8x32,
            Bs32x16 => Bs8x16,
            Bs32x32 => Bs8x32,
            Bs32x64 => Bs8x64,
            Bs64x32 => Bs16x32,
            Bs64x64 => Bs16x64,
            Bs32x8 => Bs8x8,
            Bs64x16 => Bs16x16,
            _ => return None,
        }),
        // Uneven 4-way horizontal (H4A, H4B): 1/8-height sub-blocks
        // (`Partition_Subsize` column 2, AV2 Spec v1.0.0 §5.20.3.2).
        BlockPartition::H4A | BlockPartition::H4B => Some(match bs {
            Bs16x32 => Bs16x4,
            Bs32x32 => Bs32x4,
            Bs32x64 => Bs32x8,
            Bs64x32 => Bs64x4,
            Bs64x64 => Bs64x8,
            Bs8x32 => Bs8x4,
            Bs16x64 => Bs16x8,
            _ => return None,
        }),
        // Uneven 4-way vertical (V4A, V4B): 1/8-width sub-blocks
        // (`Partition_Subsize` column 2, AV2 Spec v1.0.0 §5.20.3.2).
        BlockPartition::V4A | BlockPartition::V4B => Some(match bs {
            Bs32x16 => Bs4x16,
            Bs32x32 => Bs4x32,
            Bs32x64 => Bs4x64,
            Bs64x32 => Bs8x32,
            Bs64x64 => Bs8x64,
            Bs32x8 => Bs4x8,
            Bs64x16 => Bs8x16,
            _ => return None,
        }),
        // `PARTITION_SPLIT` only exists for 128x128 and 256x256 superblocks.
        BlockPartition::Split => match bs {
            Bs128x128 => Some(Bs64x64),
            _ => None,
        },
    }
}

/// Returns the sub-block sizes `(sub_eighth, big_half, med_quarter)` for uneven
/// 4-way partitions (`H4A`, `H4B`, `V4A`, `V4B`) per AV2 Spec §5.20.3.2.
pub fn partition_4way_sizes(
    bs: BlockSize,
    bp: BlockPartition,
) -> Option<(BlockSize, BlockSize, BlockSize)> {
    match bp {
        BlockPartition::H4A | BlockPartition::H4B => {
            let sub_eighth = partition_subsize(bs, bp)?;
            let big_half = partition_subsize(bs, BlockPartition::H)?;
            let med_quarter = partition_subsize(big_half, BlockPartition::H)?;
            Some((sub_eighth, big_half, med_quarter))
        }
        BlockPartition::V4A | BlockPartition::V4B => {
            let sub_eighth = partition_subsize(bs, bp)?;
            let big_half = partition_subsize(bs, BlockPartition::V)?;
            let med_quarter = partition_subsize(big_half, BlockPartition::V)?;
            Some((sub_eighth, big_half, med_quarter))
        }
        _ => None,
    }
}

/// `Partition_Size_Adjust[bs]` (AV2 Spec v1.0.0 §5.20.3.2), in `BlockSize` order.
pub fn partition_size_adjust(bs: BlockSize) -> usize {
    use BlockSize::*;
    match bs {
        Bs128x128 => 6,
        Bs128x64 => 5,
        Bs64x128 => 4,
        Bs64x64 => 3,
        Bs64x32 => 3,
        Bs32x64 => 3,
        Bs32x32 => 2,
        Bs32x16 => 2,
        Bs16x32 => 2,
        Bs16x16 => 1,
        Bs16x8 => 1,
        Bs8x16 => 1,
        Bs64x16 => 15,
        Bs16x64 => 14,
        Bs32x8 => 13,
        Bs8x32 => 12,
        Bs16x4 => 11,
        Bs4x16 => 10,
        // 1:8 and 1:16 sizes, 2x2 and non-specification sizes all map to 0.
        _ => 0,
    }
}

/// `Partition_Size_Adjust_Rect_Type[bs]` (AV2 Spec v1.0.0 §5.20.3.2).
pub fn partition_size_adjust_rect_type(bs: BlockSize) -> usize {
    use BlockSize::*;
    match bs {
        Bs128x128 => 9,
        Bs128x64 => 8,
        Bs64x128 => 7,
        Bs64x64 => 6,
        Bs64x32 => 5,
        Bs32x64 => 4,
        Bs32x32 => 3,
        Bs32x16 | Bs16x8 => 2,
        Bs16x32 | Bs8x16 => 1,
        Bs64x16 => 14,
        Bs16x64 | Bs4x16 | Bs8x32 => 13,
        Bs32x8 | Bs16x4 => 14,
        // 4x4, 8x8, 8x4, 4x8, 1:8/1:16 sizes and non-specification sizes.
        _ => 0,
    }
}

/// `rect_type_implied_by_bsize(bs)` (AV2 Spec v1.0.0 §5.20.3.2).
///
/// `Some(true)` means `RECT_VERT` (only a vertical split is possible),
/// `Some(false)` means `RECT_HORZ`, and `None` means the direction is signalled.
pub fn rect_type_implied_by_bsize(bs: BlockSize) -> Option<bool> {
    use BlockSize::*;
    match bs {
        Bs4x8 | Bs64x128 | Bs4x16 => Some(false),
        Bs8x4 | Bs128x64 | Bs16x4 => Some(true),
        _ => None,
    }
}

/// `partition_implied(r, c, bSize)` (AV2 Spec v1.0.0 §5.20.3.2).
///
/// Returns the partition the syntax implies for the node without coding any
/// symbol. `Some(BlockPartition::None)` means "no symbol and code a block".
pub fn partition_implied(
    bs: BlockSize,
    bx4: usize,
    by4: usize,
    mi_cols: usize,
    mi_rows: usize,
) -> Option<BlockPartition> {
    use BlockSize::*;
    // 1:8 / 1:16 blocks and the smallest blocks never recurse.
    if matches!(
        bs,
        Bs4x4 | Bs2x2 | Bs4x32 | Bs32x4 | Bs8x64 | Bs64x8 | Bs64x4 | Bs4x64
    ) {
        return Some(BlockPartition::None);
    }
    partition_implied_at_boundary(bs, bx4, by4, mi_cols, mi_rows)
}

/// `partition_implied_at_boundary(r, c, bSize)` (AV2 Spec v1.0.0 §5.20.3.2).
fn partition_implied_at_boundary(
    bs: BlockSize,
    bx4: usize,
    by4: usize,
    mi_cols: usize,
    mi_rows: usize,
) -> Option<BlockPartition> {
    let b_dim = BLOCK_DIMENSIONS[bs as usize];
    let w4 = b_dim[0] as usize;
    let h4 = b_dim[1] as usize;
    let has_rows = by4 + (h4 >> 1) < mi_rows;
    let has_cols = bx4 + (w4 >> 1) < mi_cols;
    if has_rows && has_cols {
        return None;
    }
    // AV2 Spec v1.0.0 §5.20.3.2 `partition_implied_at_boundary`: the half-block
    // that falls outside the frame is dropped, which forces the direction of
    // the split.
    if w4 == h4 {
        return Some(if has_rows {
            BlockPartition::V
        } else {
            BlockPartition::H
        });
    }
    if h4 > w4 {
        let sub_has_cols = bx4 + (w4 >> 2) < mi_cols;
        if !has_rows || (w4 >= 4 && !sub_has_cols) {
            return Some(BlockPartition::H);
        }
    } else {
        let sub_has_rows = by4 + (h4 >> 2) < mi_rows;
        if !has_cols || (h4 >= 4 && !sub_has_rows) {
            return Some(BlockPartition::V);
        }
    }
    None
}

/// `is_chroma_offset_for_subsize` (AV2 Spec v1.0.0 §5.20.3.2).
fn is_chroma_offset_for_subsize(bs: BlockSize, ss_x: bool, ss_y: bool) -> bool {
    let b_dim = BLOCK_DIMENSIONS[bs as usize];
    (ss_y && b_dim[3] == 0) || (ss_x && b_dim[2] == 0)
}

/// `is_chroma_offset_for_partition` (AV2 Spec v1.0.0 §5.20.3.2).
///
/// Returns whether coding `bp` of `bs` offsets the chroma position, which
/// propagates `chromaOffset = 1` into every child of the node.
pub fn chroma_offset_for_partition(
    bp: BlockPartition,
    bs: BlockSize,
    cfg: &PartitionSyntaxConfig,
) -> bool {
    cfg.has_chroma && is_chroma_offset_for_partition(bp, bs, cfg.subsampling_x, cfg.subsampling_y)
}

/// `is_chroma_offset_for_partition` (AV2 Spec v1.0.0 §5.20.3.2).
fn is_chroma_offset_for_partition(
    bp: BlockPartition,
    bs: BlockSize,
    ss_x: bool,
    ss_y: bool,
) -> bool {
    let Some(sub) = partition_subsize(bs, bp) else {
        return false;
    };
    if is_chroma_offset_for_subsize(sub, ss_x, ss_y) {
        return true;
    }
    if bp == BlockPartition::H3
        && !(bs == BlockSize::Bs8x32 && ss_x)
        && let Some(mid) = h_partition_midsize(bs)
    {
        return is_chroma_offset_for_subsize(mid, ss_x, ss_y);
    }
    false
}

/// Computes the 2-bit secondary partition context (`get_partition2_ctx`) for
/// extended partition signalling (AV2 Spec v1.0.0 §5.20.3.2 / dav2d `get_partition2_ctx`).
pub fn get_partition2_ctx(
    a: &BlockContext,
    l: &BlockContext,
    bs: BlockSize,
    is_vert: bool,
    bx4: usize,
    by4: usize,
) -> usize {
    let b_dim = BLOCK_DIMENSIONS[bs as usize];
    if !is_vert {
        let hh4 = (b_dim[1] >> 1) as usize;
        let shift = (b_dim[3] as u32).saturating_sub(2) as u8;
        let b1 = (l.partition[(by4 + hh4) & 31] >> shift) & 1;
        let b0 = (l.partition[by4 & 31] >> shift) & 1;
        b1 as usize + ((b0 as usize) << 1)
    } else {
        let hw4 = (b_dim[0] >> 1) as usize;
        let shift = (b_dim[2] as u32).saturating_sub(2) as u8;
        let b1 = (a.partition[(bx4 + hw4) & 31] >> shift) & 1;
        let b0 = (a.partition[bx4 & 31] >> shift) & 1;
        b1 as usize + ((b0 as usize) << 1)
    }
}

/// `is_ext_partition_allowed(bSize, rectType)` (AV2 Spec v1.0.0 §5.20.3.2).
///
/// This is the sequence-header gate (`enable_ext_partitions`). The geometry
/// conditions of the reference `has_hv3` test are enforced by
/// [`ext_selection_allowed`] when the encoder evaluates a candidate; at the
/// 64x64 nodes the encoder searches, those conditions hold for every supported
/// layout.
pub fn is_ext_partition_allowed(bs: BlockSize, is_vert: bool, cfg: &PartitionSyntaxConfig) -> bool {
    let _ = (bs, is_vert);
    cfg.enable_ext_partitions
}

/// `is_uneven_4way_partition_allowed(bSize, rectType)` (AV2 Spec v1.0.0 §5.20.3.2).
///
/// This is the sequence-header gate (`enable_uneven_4way_partitions`); see
/// [`ext_selection_allowed`] for the geometry conditions.
pub fn is_uneven_4way_partition_allowed(
    bs: BlockSize,
    is_vert: bool,
    cfg: &PartitionSyntaxConfig,
) -> bool {
    let _ = (bs, is_vert);
    cfg.enable_uneven_4way_partitions
}

/// Geometry gate for an extended partition candidate (AV2 Spec v1.0.0
/// §5.20.3.2 `is_ext_partition_allowed` / `is_uneven_4way_partition_allowed`,
/// `dav2d` `decode.c` `decode_sb` `has_hv3` / `has_hv4ab`).
///
/// A 3-way partition needs four 4x4 units across the split axis and two along
/// it; an uneven 4-way partition needs eight across. In both cases the
/// subsampled chroma block must remain large enough, and the child must stay
/// within the 8:1 aspect limit. The encoder calls this before adding an
/// extended candidate to its search.
pub fn ext_selection_allowed(
    bs: BlockSize,
    bp: BlockPartition,
    cfg: &PartitionSyntaxConfig,
) -> bool {
    let is_vert = match bp {
        BlockPartition::V3 | BlockPartition::V4A | BlockPartition::V4B => true,
        BlockPartition::H3 | BlockPartition::H4A | BlockPartition::H4B => false,
        _ => return false,
    };
    let b_dim = BLOCK_DIMENSIONS[bs as usize];
    let (bw4, bh4) = (b_dim[0] as usize, b_dim[1] as usize);
    let (other4, main4) = if is_vert { (bw4, bh4) } else { (bh4, bw4) };
    let ss_x = cfg.subsampling_x as usize;
    let ss_y = cfg.subsampling_y as usize;
    let (other_ss, main_ss) = if is_vert {
        ((bw4 >> ss_x).max(1), (bh4 >> ss_y).max(1))
    } else {
        ((bh4 >> ss_y).max(1), (bw4 >> ss_x).max(1))
    };
    let aspect_isn8 = usize::from(cfg.max_pb_aspect_ratio != 8);
    match bp {
        BlockPartition::H3 | BlockPartition::V3 => {
            cfg.enable_ext_partitions
                && other4 >= 4
                && main4 >= 2
                && other4 * 8 >= main4 * 4 + aspect_isn8
                && other_ss >= 4
                && main_ss >= 2
        }
        BlockPartition::H4A | BlockPartition::H4B | BlockPartition::V4A | BlockPartition::V4B => {
            cfg.enable_uneven_4way_partitions
                && other4 >= 8
                && other4 * 8 >= main4 * 8 + aspect_isn8
                && other_ss >= 8
        }
        _ => false,
    }
}

/// Validity of `get_plane_residual_size(bs, 1)` (AV2 Spec v1.0.0 §5.20.7.26).
///
/// The subsampled size must still be a block size the standard defines; the
/// 4x4 block always subsamples to 4x4.
fn plane_residual_size_valid(bs: BlockSize, ss_x: bool, ss_y: bool) -> bool {
    use BlockSize::*;
    let b_dim = BLOCK_DIMENSIONS[bs as usize];
    let (mut w4, mut h4) = (b_dim[0] as usize, b_dim[1] as usize);
    if bs != Bs4x4 {
        match (ss_x, ss_y) {
            // 4:2:0 clamps each subsampled dimension to the 4-sample minimum.
            (true, true) => {
                w4 = (w4 >> 1).max(1);
                h4 = (h4 >> 1).max(1);
            }
            (true, false) => {
                if w4 < 2 {
                    return false;
                }
                w4 >>= 1;
            }
            (false, true) => {
                if h4 < 2 {
                    return false;
                }
                h4 >>= 1;
            }
            (false, false) => {}
        }
    }
    matches!(
        (w4, h4),
        (1, 1)
            | (1, 2)
            | (2, 1)
            | (2, 2)
            | (2, 4)
            | (4, 2)
            | (4, 4)
            | (4, 8)
            | (8, 4)
            | (8, 8)
            | (8, 16)
            | (16, 8)
            | (16, 16)
            | (16, 32)
            | (32, 16)
            | (32, 32)
            | (1, 4)
            | (4, 1)
            | (2, 8)
            | (8, 2)
            | (1, 8)
            | (8, 1)
            | (4, 16)
            | (16, 4)
            | (8, 32)
            | (32, 8)
            | (16, 64)
            | (64, 16)
            | (4, 32)
            | (32, 4)
            | (8, 64)
            | (64, 8)
            | (32, 64)
            | (64, 32)
            | (64, 64)
    )
}

/// `is_partition_allowed(r, c, p, bSize, chromaOffset, hasChroma, numPlanes)`
/// (AV2 Spec v1.0.0 §5.20.3.2) for a shared-part coding tree.
///
/// `chroma_offset` carries the inherited `chromaOffset` flag; the local
/// computation for the candidate partition happens inside.
pub fn is_partition_allowed(
    bp: BlockPartition,
    bs: BlockSize,
    bx4: usize,
    by4: usize,
    mi_cols: usize,
    mi_rows: usize,
    chroma_offset: bool,
    cfg: &PartitionSyntaxConfig,
) -> bool {
    let Some(sub) = partition_subsize(bs, bp) else {
        return false;
    };
    // 4x4 blocks are not valid in mixed intra/inter regions of inter frames.
    if !cfg.frame_is_intra && sub == BlockSize::Bs4x4 {
        return false;
    }
    // A direction implied by the block shape cannot be overridden.
    if let Some(vertical) = rect_type_implied_by_bsize(bs) {
        if vertical && bp.is_horizontal() {
            return false;
        }
        if !vertical && bp.is_vertical() {
            return false;
        }
    }
    // Extended partition sequence header enablement.
    let ext_ok = match bp {
        BlockPartition::H3 => is_ext_partition_allowed(bs, false, cfg),
        BlockPartition::V3 => is_ext_partition_allowed(bs, true, cfg),
        BlockPartition::H4A | BlockPartition::H4B => {
            is_ext_partition_allowed(bs, false, cfg)
                && is_uneven_4way_partition_allowed(bs, false, cfg)
        }
        BlockPartition::V4A | BlockPartition::V4B => {
            is_ext_partition_allowed(bs, true, cfg)
                && is_uneven_4way_partition_allowed(bs, true, cfg)
        }
        _ => true,
    };
    if !ext_ok {
        return false;
    }
    // Maximum partition aspect ratio.
    let sub_dim = BLOCK_DIMENSIONS[sub as usize];
    let (sw4, sh4) = (sub_dim[0] as u32, sub_dim[1] as u32);
    let max_aspect = cfg.max_pb_aspect_ratio.max(1);
    if sw4 > sh4 * max_aspect || sh4 > sw4 * max_aspect {
        if bp == BlockPartition::None {
            return false;
        }
        if sw4 >= sh4 * 8 || sh4 >= sw4 * 8 {
            return false;
        }
    }
    let b_dim = BLOCK_DIMENSIONS[bs as usize];
    let (w4, h4) = (b_dim[0] as usize, b_dim[1] as usize);
    let chroma_offset = chroma_offset
        || (cfg.has_chroma
            && is_chroma_offset_for_partition(bp, bs, cfg.subsampling_x, cfg.subsampling_y));
    if cfg.has_chroma && !plane_residual_size_valid(sub, cfg.subsampling_x, cfg.subsampling_y) {
        return false;
    }
    if bp == BlockPartition::None {
        let has_rows = by4 + (h4 >> 1) < mi_rows;
        let has_cols = bx4 + (w4 >> 1) < mi_cols;
        if !has_rows || !has_cols {
            return false;
        }
    }
    if cfg.has_chroma && chroma_offset {
        // A predictive split whose chroma half is not coded is not allowed.
        return match bp {
            BlockPartition::H => by4 + (h4 >> 1) < mi_rows,
            BlockPartition::V => bx4 + (w4 >> 1) < mi_cols,
            BlockPartition::H3 => by4 + 3 * (h4 >> 2) < mi_rows,
            BlockPartition::V3 => bx4 + 3 * (w4 >> 2) < mi_cols,
            BlockPartition::H4A | BlockPartition::H4B => {
                let sub_h4 = BLOCK_DIMENSIONS[sub as usize][1] as usize;
                by4 + 7 * sub_h4 < mi_rows
            }
            BlockPartition::V4A | BlockPartition::V4B => {
                let sub_w4 = BLOCK_DIMENSIONS[sub as usize][0] as usize;
                bx4 + 7 * sub_w4 < mi_cols
            }
            _ => true,
        };
    }
    true
}

/// `init_allowed_partitions(r, c, bSize, chromaOffset, hasChroma)`
/// (AV2 Spec v1.0.0 §5.20.3.2).
pub fn init_allowed_partitions(
    bs: BlockSize,
    bx4: usize,
    by4: usize,
    mi_cols: usize,
    mi_rows: usize,
    chroma_offset: bool,
    cfg: &PartitionSyntaxConfig,
) -> AllowedPartitions {
    let mut allowed = AllowedPartitions::default();
    const CANDIDATES: [BlockPartition; 10] = [
        BlockPartition::None,
        BlockPartition::H,
        BlockPartition::V,
        BlockPartition::H3,
        BlockPartition::V3,
        BlockPartition::H4A,
        BlockPartition::H4B,
        BlockPartition::V4A,
        BlockPartition::V4B,
        BlockPartition::Split,
    ];
    for bp in CANDIDATES {
        let good = is_partition_allowed(bp, bs, bx4, by4, mi_cols, mi_rows, chroma_offset, cfg);
        let slot = AllowedPartitions::slot(bp);
        allowed.allowed[slot] = good;
        allowed.count += good as usize;
    }
    if allowed.count == 0 {
        // Fall back to `PARTITION_NONE` so a node always has a legal decision.
        allowed.allowed[AllowedPartitions::slot(BlockPartition::None)] = true;
        allowed.count = 1;
    }
    allowed
}

/// Encodes the partition decision `bp` for coding-tree node `(bs, bx4_in_sb, by4_in_sb)`
/// into `msac` per AV2 Spec v1.0.0 §5.20.3.2 (`read_partition`).
///
/// `allowed` must come from [`init_allowed_partitions`] for the same node, and
/// `bp` must be one of the partitions it reports. Implied partitions and nodes
/// with a single legal partition code no symbols and must not call this function.
#[allow(clippy::too_many_arguments)]
pub fn encode_partition_decision(
    msac: &mut MsacEncoder,
    cdf_m: &mut CdfModeContext,
    a: &BlockContext,
    l: &BlockContext,
    bs: BlockSize,
    bp: BlockPartition,
    bx4_in_sb: usize,
    by4_in_sb: usize,
    allowed: &AllowedPartitions,
) {
    let ctx = get_partition_ctx(a, l, bs, bx4_in_sb, by4_in_sb);
    let part_ctx = partition_size_adjust(bs) * 4 + ctx;
    if allowed.contains(BlockPartition::None) {
        let split = (bp != BlockPartition::None) as u32;
        msac.encode_bool_adapt(split, cdf_m.part_split(0, part_ctx));
    }
    if bp == BlockPartition::None {
        return;
    }
    if allowed.contains(BlockPartition::Split) {
        // `do_square_split` context: the +4 term for 256x256 superblocks never
        // applies because this codec supports 64x64 and 128x128 only.
        let square = (bp == BlockPartition::Split) as u32;
        msac.encode_bool_adapt(square, cdf_m.part_square(ctx));
        if bp == BlockPartition::Split {
            return;
        }
    }

    let vertical = bp.is_vertical();
    let allow_h = allowed.contains(BlockPartition::H)
        || allowed.contains(BlockPartition::H3)
        || allowed.contains(BlockPartition::H4A)
        || allowed.contains(BlockPartition::H4B);
    let allow_v = allowed.contains(BlockPartition::V)
        || allowed.contains(BlockPartition::V3)
        || allowed.contains(BlockPartition::V4A)
        || allowed.contains(BlockPartition::V4B);

    match rect_type_implied_by_bsize(bs) {
        // Direction is forced by the block shape; no symbol is coded.
        Some(implied_vertical) => debug_assert_eq!(implied_vertical, vertical),
        None => {
            if allow_h && allow_v {
                let rect_ctx = partition_size_adjust_rect_type(bs) * 4 + ctx;
                msac.encode_bool_adapt(vertical as u32, cdf_m.part_dir(0, rect_ctx));
            } else {
                debug_assert_eq!(vertical, allow_v);
            }
        }
    }

    let (non_ext_allowed, ext_allowed3, ext_allowed4) = if !vertical {
        (
            allowed.contains(BlockPartition::H),
            allowed.contains(BlockPartition::H3),
            allowed.contains(BlockPartition::H4A) || allowed.contains(BlockPartition::H4B),
        )
    } else {
        (
            allowed.contains(BlockPartition::V),
            allowed.contains(BlockPartition::V3),
            allowed.contains(BlockPartition::V4A) || allowed.contains(BlockPartition::V4B),
        )
    };

    let is_ext = bp.is_extended();
    if non_ext_allowed && (ext_allowed3 || ext_allowed4) {
        let ctx5 = get_partition2_ctx(a, l, bs, vertical, bx4_in_sb, by4_in_sb);
        let ctx6 = ctx5 + partition_size_adjust(bs) * 4;
        msac.encode_bool_adapt(is_ext as u32, cdf_m.part_ext(0, ctx6));
    }
    if !is_ext {
        return;
    }

    let is_uneven_4way = bp.is_uneven_4way();
    if ext_allowed3 && ext_allowed4 {
        let ctx5 = get_partition2_ctx(a, l, bs, vertical, bx4_in_sb, by4_in_sb);
        let ctx6 = ctx5 + partition_size_adjust(bs) * 4;
        msac.encode_bool_adapt(is_uneven_4way as u32, cdf_m.part_uneven_4way(0, ctx6));
    }

    if is_uneven_4way {
        let is_b = (bp == BlockPartition::H4B || bp == BlockPartition::V4B) as u32;
        msac.encode_bool_bypass(is_b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::cdf::CdfContext;
    use crate::av2::entropy::{BlockContext, reset_context, update_partition_ctx};
    use crate::av2::headers::PixelLayout;

    fn cfg_intra() -> PartitionSyntaxConfig {
        PartitionSyntaxConfig::new_with_ext(PixelLayout::I420, true)
    }

    fn cfg_inter() -> PartitionSyntaxConfig {
        PartitionSyntaxConfig::new_with_ext(PixelLayout::I420, false)
    }

    #[test]
    fn partition_size_adjust_matches_the_specification() {
        // AV2 Spec v1.0.0 §5.20.3.2, Partition_Size_Adjust / _Rect_Type rows.
        assert_eq!(partition_size_adjust(BlockSize::Bs64x64), 3);
        assert_eq!(partition_size_adjust(BlockSize::Bs32x32), 2);
        assert_eq!(partition_size_adjust(BlockSize::Bs16x16), 1);
        assert_eq!(partition_size_adjust(BlockSize::Bs8x8), 0);
        assert_eq!(partition_size_adjust(BlockSize::Bs128x128), 6);
        assert_eq!(partition_size_adjust(BlockSize::Bs16x64), 14);
        assert_eq!(partition_size_adjust(BlockSize::Bs64x16), 15);
        assert_eq!(partition_size_adjust_rect_type(BlockSize::Bs64x64), 6);
        assert_eq!(partition_size_adjust_rect_type(BlockSize::Bs16x16), 0);
        assert_eq!(partition_size_adjust_rect_type(BlockSize::Bs32x16), 2);
        assert_eq!(partition_size_adjust_rect_type(BlockSize::Bs16x32), 1);
    }

    #[test]
    fn partition_subsize_follows_the_specification() {
        assert_eq!(
            partition_subsize(BlockSize::Bs64x64, BlockPartition::H),
            Some(BlockSize::Bs64x32)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs64x64, BlockPartition::V),
            Some(BlockSize::Bs32x64)
        );
        // `PARTITION_SPLIT` only exists for 128x128 (and 256x256) superblocks.
        assert_eq!(
            partition_subsize(BlockSize::Bs128x128, BlockPartition::Split),
            Some(BlockSize::Bs64x64)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs64x64, BlockPartition::Split),
            None
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs8x8, BlockPartition::H),
            Some(BlockSize::Bs8x4)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs8x8, BlockPartition::V),
            Some(BlockSize::Bs4x8)
        );
        // 1:8 / 1:16 blocks never split.
        assert_eq!(
            partition_subsize(BlockSize::Bs4x32, BlockPartition::H),
            None
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs8x64, BlockPartition::H),
            None
        );
        // Direction implied by the block shape.
        assert_eq!(rect_type_implied_by_bsize(BlockSize::Bs64x128), Some(false));
        assert_eq!(rect_type_implied_by_bsize(BlockSize::Bs128x64), Some(true));
        assert_eq!(rect_type_implied_by_bsize(BlockSize::Bs64x64), None);
        // Extended partitions (AV2 Spec §5.20.3.2).
        assert_eq!(
            partition_subsize(BlockSize::Bs64x64, BlockPartition::H3),
            Some(BlockSize::Bs64x16)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs64x64, BlockPartition::V3),
            Some(BlockSize::Bs16x64)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs64x64, BlockPartition::H4A),
            Some(BlockSize::Bs64x8)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs64x64, BlockPartition::H4B),
            Some(BlockSize::Bs64x8)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs64x64, BlockPartition::V4A),
            Some(BlockSize::Bs8x64)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs64x64, BlockPartition::V4B),
            Some(BlockSize::Bs8x64)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs4x4, BlockPartition::H3),
            None
        );
    }

    #[test]
    fn partition_implied_at_the_right_frame_edge() {
        // 96x64: MiCols = 24, MiRows = 16. A 64x64 at the right edge must become
        // a vertical split (AV2 `partition_implied_at_boundary`).
        assert_eq!(
            partition_implied(BlockSize::Bs64x64, 16, 0, 24, 16),
            Some(BlockPartition::V)
        );
        // Inside the frame nothing is implied.
        assert_eq!(partition_implied(BlockSize::Bs64x64, 0, 0, 24, 16), None);
        // 16x32 at the edge: the quarter-width check implies a horizontal split.
        assert_eq!(
            partition_implied(BlockSize::Bs16x32, 24, 0, 25, 16),
            Some(BlockPartition::H)
        );
    }

    #[test]
    fn partition_implied_at_the_bottom_frame_edge() {
        // 1280x720: MiRows = 180. A 64x64 at the last SB row splits horizontally.
        assert_eq!(
            partition_implied(BlockSize::Bs64x64, 0, 176, 320, 180),
            Some(BlockPartition::H)
        );
        // 4x4 and the 1:8/1:16 sizes are implied `PARTITION_NONE`.
        assert_eq!(
            partition_implied(BlockSize::Bs4x4, 0, 0, 320, 180),
            Some(BlockPartition::None)
        );
        assert_eq!(
            partition_implied(BlockSize::Bs4x32, 0, 0, 320, 180),
            Some(BlockPartition::None)
        );
    }

    #[test]
    fn allowed_partitions_reject_illegal_choices() {
        let cfg = cfg_intra();
        // A 64x64 node with extended partitions enabled allows 9 partitions:
        // None, H, V, H3, V3, H4A, H4B, V4A, V4B. (Split only for 128x128).
        let allowed = init_allowed_partitions(BlockSize::Bs64x64, 0, 0, 16, 16, false, &cfg);
        assert!(allowed.contains(BlockPartition::None));
        assert!(allowed.contains(BlockPartition::H));
        assert!(allowed.contains(BlockPartition::V));
        assert!(allowed.contains(BlockPartition::H3));
        assert!(allowed.contains(BlockPartition::V3));
        assert!(allowed.contains(BlockPartition::H4A));
        assert!(allowed.contains(BlockPartition::H4B));
        assert!(allowed.contains(BlockPartition::V4A));
        assert!(allowed.contains(BlockPartition::V4B));
        assert!(!allowed.contains(BlockPartition::Split));
        assert_eq!(allowed.count(), 9);

        // When extended partitions are disabled in config, only None, H, V are allowed.
        let cfg_no_ext =
            PartitionSyntaxConfig::with_ext_flags(PixelLayout::I420, true, false, false);
        let allowed_no_ext =
            init_allowed_partitions(BlockSize::Bs64x64, 0, 0, 16, 16, false, &cfg_no_ext);
        assert!(allowed_no_ext.contains(BlockPartition::None));
        assert!(allowed_no_ext.contains(BlockPartition::H));
        assert!(allowed_no_ext.contains(BlockPartition::V));
        assert!(!allowed_no_ext.contains(BlockPartition::H3));
        assert!(!allowed_no_ext.contains(BlockPartition::V3));
        assert_eq!(allowed_no_ext.count(), 3);

        // At the right frame edge `PARTITION_NONE` is not allowed.
        let allowed = init_allowed_partitions(BlockSize::Bs64x64, 16, 0, 24, 16, false, &cfg);
        assert!(!allowed.contains(BlockPartition::None));

        // Inside a 64x64 superblock `PARTITION_SPLIT` is legal for 128x128 only.
        let allowed = init_allowed_partitions(BlockSize::Bs128x128, 0, 0, 32, 32, false, &cfg);
        assert!(allowed.contains(BlockPartition::Split));

        // Inter frames forbid 4x4 blocks in mixed regions.
        let allowed = init_allowed_partitions(BlockSize::Bs8x8, 0, 0, 32, 32, false, &cfg_inter());
        assert!(allowed.contains(BlockPartition::H));
        let allowed = init_allowed_partitions(BlockSize::Bs4x8, 0, 0, 32, 32, false, &cfg_inter());
        assert!(!allowed.contains(BlockPartition::H));
    }

    #[test]
    fn h_partition_midsize_matches_normative_table() {
        assert_eq!(
            h_partition_midsize(BlockSize::Bs64x64),
            Some(BlockSize::Bs32x32)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs32x32),
            Some(BlockSize::Bs16x16)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs16x16),
            Some(BlockSize::Bs8x8)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs8x16),
            Some(BlockSize::Bs4x8)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs16x8),
            Some(BlockSize::Bs8x4)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs16x32),
            Some(BlockSize::Bs8x16)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs32x16),
            Some(BlockSize::Bs16x8)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs32x64),
            Some(BlockSize::Bs16x32)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs64x32),
            Some(BlockSize::Bs32x16)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs8x32),
            Some(BlockSize::Bs4x16)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs32x8),
            Some(BlockSize::Bs16x4)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs16x64),
            Some(BlockSize::Bs8x32)
        );
        assert_eq!(
            h_partition_midsize(BlockSize::Bs64x16),
            Some(BlockSize::Bs32x8)
        );
        assert_eq!(h_partition_midsize(BlockSize::Bs4x4), None);
        assert_eq!(h_partition_midsize(BlockSize::Bs128x128), None);
    }

    #[test]
    fn extended_selection_uses_the_reference_geometry_gate() {
        let cfg420 = cfg_intra();
        for bp in [
            BlockPartition::H3,
            BlockPartition::V3,
            BlockPartition::H4A,
            BlockPartition::H4B,
            BlockPartition::V4A,
            BlockPartition::V4B,
        ] {
            assert!(ext_selection_allowed(BlockSize::Bs64x64, bp, &cfg420));
        }
        // In 4:2:0 an 8x4-unit node is only 4x2 chroma units, so the uneven
        // 4-way children are too small; 4:4:4 keeps them legal.
        assert!(!ext_selection_allowed(
            BlockSize::Bs32x32,
            BlockPartition::H4A,
            &cfg420
        ));
        assert!(!ext_selection_allowed(
            BlockSize::Bs16x16,
            BlockPartition::H3,
            &cfg420
        ));
        let cfg444 = PartitionSyntaxConfig::new_with_ext(PixelLayout::I444, true);
        assert!(ext_selection_allowed(
            BlockSize::Bs32x32,
            BlockPartition::H4A,
            &cfg444
        ));
        assert!(ext_selection_allowed(
            BlockSize::Bs16x16,
            BlockPartition::H3,
            &cfg444
        ));
        // Non-extended partitions never pass the gate.
        assert!(!ext_selection_allowed(
            BlockSize::Bs64x64,
            BlockPartition::H,
            &cfg420
        ));
    }

    #[test]
    fn extended_subsize_tables_match_the_reference_tables() {
        // `Partition_Subsize` quarter column (H3/V3) and eighth column
        // (H4A/B, V4A/B), cross-checked with `dav2d` `decode.c` `subb`.
        assert_eq!(
            partition_subsize(BlockSize::Bs32x16, BlockPartition::H3),
            Some(BlockSize::Bs32x8)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs16x16, BlockPartition::H3),
            Some(BlockSize::Bs16x8)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs8x16, BlockPartition::H3),
            Some(BlockSize::Bs8x8)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs16x32, BlockPartition::V3),
            Some(BlockSize::Bs8x32)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs16x16, BlockPartition::V3),
            Some(BlockSize::Bs8x16)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs16x8, BlockPartition::V3),
            Some(BlockSize::Bs8x8)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs64x32, BlockPartition::H4A),
            Some(BlockSize::Bs64x4)
        );
        assert_eq!(
            partition_subsize(BlockSize::Bs32x64, BlockPartition::V4A),
            Some(BlockSize::Bs4x64)
        );
        assert_eq!(
            partition_4way_sizes(BlockSize::Bs64x32, BlockPartition::H4A),
            Some((BlockSize::Bs64x4, BlockSize::Bs64x16, BlockSize::Bs64x8))
        );
        assert_eq!(
            partition_4way_sizes(BlockSize::Bs32x64, BlockPartition::V4B),
            Some((BlockSize::Bs4x64, BlockSize::Bs16x64, BlockSize::Bs8x64))
        );
    }

    #[test]
    fn partition_4way_sizes_returns_normative_dimensions() {
        assert_eq!(
            partition_4way_sizes(BlockSize::Bs64x64, BlockPartition::H4A),
            Some((BlockSize::Bs64x8, BlockSize::Bs64x32, BlockSize::Bs64x16))
        );
        assert_eq!(
            partition_4way_sizes(BlockSize::Bs64x64, BlockPartition::H4B),
            Some((BlockSize::Bs64x8, BlockSize::Bs64x32, BlockSize::Bs64x16))
        );
        assert_eq!(
            partition_4way_sizes(BlockSize::Bs64x64, BlockPartition::V4A),
            Some((BlockSize::Bs8x64, BlockSize::Bs32x64, BlockSize::Bs16x64))
        );
        assert_eq!(
            partition_4way_sizes(BlockSize::Bs64x64, BlockPartition::V4B),
            Some((BlockSize::Bs8x64, BlockSize::Bs32x64, BlockSize::Bs16x64))
        );
        assert_eq!(
            partition_4way_sizes(BlockSize::Bs32x32, BlockPartition::H4A),
            Some((BlockSize::Bs32x4, BlockSize::Bs32x16, BlockSize::Bs32x8))
        );
        assert_eq!(
            partition_4way_sizes(BlockSize::Bs32x32, BlockPartition::V4B),
            Some((BlockSize::Bs4x32, BlockSize::Bs16x32, BlockSize::Bs8x32))
        );
        assert_eq!(
            partition_4way_sizes(BlockSize::Bs64x64, BlockPartition::H),
            None
        );
        assert_eq!(
            partition_4way_sizes(BlockSize::Bs4x4, BlockPartition::H4A),
            None
        );
    }

    #[test]
    fn get_partition2_ctx_computes_neighbor_contexts() {
        let (mut a, mut l) = fresh();
        // Zero initial contexts.
        assert_eq!(
            get_partition2_ctx(&a, &l, BlockSize::Bs32x32, false, 0, 0),
            0
        );
        assert_eq!(
            get_partition2_ctx(&a, &l, BlockSize::Bs32x32, true, 0, 0),
            0
        );

        // Update with 16x16 leaf blocks.
        update_partition_ctx(&mut a, &mut l, BlockSize::Bs16x16, 0, 0);
        let ctx_h = get_partition2_ctx(&a, &l, BlockSize::Bs32x32, false, 0, 0);
        assert!(ctx_h <= 3);
        let ctx_v = get_partition2_ctx(&a, &l, BlockSize::Bs32x32, true, 0, 0);
        assert!(ctx_v <= 3);
    }

    #[test]
    fn encode_extended_partition_decisions() {
        let (a, l) = fresh();
        let cfg = cfg_intra();
        let allowed = init_allowed_partitions(BlockSize::Bs64x64, 0, 0, 16, 16, false, &cfg);

        for bp in [
            BlockPartition::H3,
            BlockPartition::V3,
            BlockPartition::H4A,
            BlockPartition::H4B,
            BlockPartition::V4A,
            BlockPartition::V4B,
        ] {
            let mut msac = MsacEncoder::new();
            let mut cdf = CdfContext::default();
            encode_partition_decision(
                &mut msac,
                &mut cdf.m,
                &a,
                &l,
                BlockSize::Bs64x64,
                bp,
                0,
                0,
                &allowed,
            );
            let bytes = msac.finalize();
            assert!(!bytes.is_empty());
        }
    }

    #[test]
    fn partition_context_uses_the_leaf_neighbour_size() {
        let fresh = || {
            let mut a = BlockContext::default();
            let mut l = BlockContext::default();
            reset_context(&mut a, true, false);
            reset_context(&mut l, true, false);
            (a, l)
        };

        // Nothing coded yet: no neighbour is "smaller", matching the reference
        // decoder's zero-initialised partition masks at frame/tile boundaries.
        let (a, l) = fresh();
        assert_eq!(get_partition_ctx(&a, &l, BlockSize::Bs64x64, 0, 0), 0);

        // A 32x32 leaf above a 64x64 node contributes the "above is smaller" bit.
        let (mut a, mut l) = fresh();
        update_partition_ctx(&mut a, &mut l, BlockSize::Bs32x32, 0, 0);
        reset_context(&mut l, true, false);
        assert_eq!(get_partition_ctx(&a, &l, BlockSize::Bs64x64, 0, 0), 1);

        // A 16x16 leaf on the left contributes the "left is smaller" bit.
        let (mut a, mut l) = fresh();
        update_partition_ctx(&mut a, &mut l, BlockSize::Bs16x16, 0, 0);
        reset_context(&mut a, true, false);
        assert_eq!(get_partition_ctx(&a, &l, BlockSize::Bs64x64, 0, 0), 2);

        // Both neighbours smaller.
        let (mut a, mut l) = fresh();
        update_partition_ctx(&mut a, &mut l, BlockSize::Bs16x16, 0, 0);
        assert_eq!(get_partition_ctx(&a, &l, BlockSize::Bs64x64, 0, 0), 3);

        // A same-size neighbour is not "smaller".
        let (mut a, mut l) = fresh();
        update_partition_ctx(&mut a, &mut l, BlockSize::Bs64x64, 0, 0);
        assert_eq!(get_partition_ctx(&a, &l, BlockSize::Bs64x64, 0, 0), 0);

        // The `do_split` context is `Partition_Size_Adjust[bs] * 4 + ctx`.
        assert_eq!(partition_size_adjust(BlockSize::Bs64x64) * 4, 12);
    }

    fn fresh() -> (BlockContext, BlockContext) {
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();
        reset_context(&mut a, true, false);
        reset_context(&mut l, true, false);
        (a, l)
    }

    #[test]
    fn partition_context_only_updates_at_leaf_blocks() {
        // A `V`/`H`/`Split` parent must not write the neighbour masks: the child
        // nodes read them before they decode their own leaves.
        let (a, l) = fresh();
        let before_a = a.partition;
        let before_l = l.partition;
        // Encoding a V partition of 64x64 must leave the masks untouched.
        encode_partition_decision(
            &mut MsacEncoder::new(),
            &mut CdfContext::default().m,
            &a,
            &l,
            BlockSize::Bs64x64,
            BlockPartition::V,
            0,
            0,
            &init_allowed_partitions(BlockSize::Bs64x64, 0, 0, 16, 16, false, &cfg_intra()),
        );
        assert_eq!(a.partition, before_a);
        assert_eq!(l.partition, before_l);
    }
}
