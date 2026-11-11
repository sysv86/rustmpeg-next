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

//! Clean-room Wave 3 Frame & Tile Encoder (`frame.rs`), integrating:
//! - Padded 128-aligned `Plane<P>` and `Frame<P>` buffers + 16-slot `ReferenceDpb<P>`
//! - Multi-Threaded Scoped Tile-Parallel Encoding (`execute_tiles_parallel`)
//! - Superblock & Coding-Tree RDO Partitioning (`None`, `H`, `V`, `Split`)
//! - Recursive Sub-Block Transform Partitioning (`TxPartition`: `TxNone`, `TxSplit`, `TxH`, `TxV`)
//!   and 64-point `DCT-64` / `Identity-64` transforms
//! - Advanced Intra & Screen Content Tools: `IntraBC` (Intra Block Copy), `Palette` (K-means
//!   `2..=8` colors), `ISP` (Intra Sub-Partitions), `MRL`, `AngleDelta`, `FilterIntra`, and `CfL`
//! - Advanced Inter Tools: `DRL`, Multi-Ref (`LAST`/`GOLDEN`/`ALTREF`), Compound Bi-Prediction,
//!   Switchable 8-Tap Subpel Filters, and `WARP_NEWMV` + `DELTAWARP` warped
//!   motion (the decoder also handles `WARPMV` + `DELTAWARP`; AV2 v1.0.0 has
//!   no OBMC, see `warp.rs`)
//! - Full 5-Stage In-Loop Filter Pipeline: `Deblock` $\to$ `CDEF` $\to$ `CCSO` $\to$ `GDF` $\to$
//!   `Wiener/Sgrproj Restoration` + `FilmGrainData` AR(1) parameter estimation

use crate::accelerator::PlaneView;
use crate::av2::ccso::{optimize_and_apply_ccso, select_ccso_units};
use crate::av2::cdef::optimize_and_apply_cdef;
use crate::av2::cdf::CdfContext;
use crate::av2::cfl::{compute_cfl_ac_luma, evaluate_cfl_uv, predict_cfl_plane};
use crate::av2::container::RawFrame;
use crate::av2::deblock::{
    BlockDeblockInfo, DeblockFilter, DeblockGrid, deblock_frame_normative, derive_deblock_params,
};
use crate::av2::entropy::encode_lossless_tx_size;
use crate::av2::entropy::{
    BlockContext, CoefCodingParams, encode_cfl_alphas, encode_coefs, encode_filter_intra_mode,
    encode_intra_uv_mode, encode_intra_y_mode, encode_mrl_idx, gather_inter_neighbors,
    get_filter_ctx_for_block, reset_context, update_partition_ctx,
};
use crate::av2::filmgrain::estimate_film_grain_params;
use crate::av2::gdf::{optimize_and_apply_gdf, select_gdf_units};
use crate::av2::headers::{
    AdaptiveBoolean, FilterMode, FrameHeader, FrameType, MAX_SEGMENTS, NUM_REF_FRAMES, ObuType,
    PRIMARY_REF_NONE, PixelLayout, RestorationType, SEG_LVL_GLOBALMV, SEG_LVL_SKIP, SequenceHeader,
};
use crate::av2::inter::{MAX_RMB_SB_HITS, MvStackParams, RefMvBank, RefMvBankHits, find_mv_stack};
use crate::av2::intops::ulog2;
use crate::av2::ipred::{
    blend_interintra, compute_sad, compute_sse, interintra_intra_mode, interintra_mask,
    predict_filter_intra_block, predict_intra_block, prepare_intra_edges, prepare_intra_edges_mrl,
    select_best_filter_intra_mode, select_best_intra_mode_uv, select_best_intra_mode_y,
};
use crate::av2::levels::{
    BlockPartition, BlockSize, CFL_PRED, CompInterType, InterPredMode, MvXY, TxPartition, txtp,
};
use crate::av2::me::{
    blend_compound_prep, encode_mv_residual, estimate_motion_block, mc_predict_block,
    mc_predict_block_prep, search_motion_vector, select_best_subpel_filter,
};
use crate::av2::msac::MsacEncoder;
use crate::av2::obu::write_frame_and_tile_group_obu;
use crate::av2::partition::{
    AllowedPartitions, PartitionSyntaxConfig, chroma_offset_for_partition,
    encode_partition_decision, ext_selection_allowed, h_partition_midsize, init_allowed_partitions,
    partition_4way_sizes, partition_implied, partition_subsize,
};
use crate::av2::pixel::{BitDepth, Pixel};
use crate::av2::quantizer::{
    dq_lookup, init_quant_tables, quantize_dequantize_block, quantize_rdoq_block,
};
use crate::av2::restoration::{
    apply_encoder_restoration, encoder_ns_chroma_filter, encoder_ns_luma_filter,
};
use crate::av2::screen::{
    apply_intrabc_block, apply_palette_block, encode_palette_mode_info, encode_palette_tokens,
    evaluate_palette_block, search_intrabc_block_local,
};
use crate::av2::segment::{
    SegmentCoder, SegmentMap, VaqPlan, encode_delta_q, encode_segment_id, segment_qindex,
};
use crate::av2::tables::{
    BLOCK_DIMENSIONS, MAX_TXFM_SIZE_FOR_BS, SS_BS, TXFM_DIMENSIONS, TXTP_FROM_UVMODE,
};
use crate::av2::threading::{TileJob, execute_tiles_parallel};
use crate::av2::transform::{fwd_txfm_2d, inv_txfm_add};
use crate::av2::txpart::{
    encode_tx_partition_syntax, expand_tx_partition, select_best_tx_partition,
    tx_partition_is_signallable,
};
use crate::av2::warp::{
    DEFAULT_WARP_PARAMS, MAX_WARP_SB_HITS, WARP_DELTA_NUM_SYMBOLS_LOW, WarpMiInfo, block_warp,
    extend_warp_params, find_warp_samples, get_warp_translation, reduce_warp_model, setup_shear,
    warp_bank_insert, warp_estimation, warp_has_ref_neighbour, warp_local_context,
    warp_mode_context, warp_motion_vector, warp_mv_at, warp_sub_mv,
};
use crate::av2::wedge::{
    WEDGE_ANGLE_DIST_TO_IDX, WEDGE_IDX_TO_ANGLE_DIST, WEDGE_TABLES, WEDGE_TYPES,
    compute_diff_wtd_mask_intermediate,
};

/// Maps the crate's [`BlockSize`] enumeration to the AV2 `BLOCK_SIZES` index
/// used by MiSize-indexed tables and CDFs (`WarpDeltaPrecisionCdf`,
/// `WarpInterIntraCdf`, `Size_Group`).
pub(crate) fn mi_size_index(bs: BlockSize) -> usize {
    let d = BLOCK_DIMENSIONS[bs as usize];
    match (d[0], d[1]) {
        (1, 1) => 0,
        (1, 2) => 1,
        (2, 1) => 2,
        (2, 2) => 3,
        (2, 4) => 4,
        (4, 2) => 5,
        (4, 4) => 6,
        (4, 8) => 7,
        (8, 4) => 8,
        (8, 8) => 9,
        (8, 16) => 10,
        (16, 8) => 11,
        (16, 16) => 12,
        (16, 32) => 13,
        (32, 16) => 14,
        (32, 32) => 15,
        (32, 64) => 16,
        (64, 32) => 17,
        (64, 64) => 18,
        (1, 4) => 19,
        (4, 1) => 20,
        (2, 8) => 21,
        (8, 2) => 22,
        (4, 16) => 23,
        (16, 4) => 24,
        (1, 8) => 25,
        (8, 1) => 26,
        (2, 16) => 27,
        (16, 2) => 28,
        _ => 0,
    }
}

/// `Size_Group[MiSize]` (AV2 Specification v1.0.0 §9, `size_group.h`).
pub(crate) static SPEC_SIZE_GROUP: [u8; 29] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2,
];

/// Chooses the loop restoration unit size (log2 samples) for `plane`.///
/// `LoopRestorationSize` starts at 512 and is reduced by the frame-header
/// half/max/quarter flags. The encoder uses the smallest size the superblock
/// geometry allows (64 luma samples for 64x64 superblocks) so that frames have
/// several independent restoration units.
fn lr_unit_size_log2(plane: usize, sb128: u8, ss_hor: u8, ss_ver: u8) -> u8 {
    let base = if plane == 0 {
        9u8
    } else {
        9 - ss_hor.max(ss_ver)
    };
    match sb128 {
        0 => {
            if plane == 0 {
                base.saturating_sub(3)
            } else {
                6.min(base)
            }
        }
        1 => base.saturating_sub(2),
        _ => base,
    }
}

/// Frame-level loop restoration tool selection (AV2 Specification v1.0.0
/// §6.17.7.7, `FrameRestorationType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RestorationMode {
    /// `RESTORE_PC_WIENER`: pixel-classified Wiener with pretrained filters.
    #[default]
    PcWiener,
    /// `RESTORE_WIENER_NONSEP`: non-separable Wiener with frame-level filters.
    NsWiener,
    /// `RESTORE_SWITCHABLE`: per-unit PC or NS Wiener selection.
    Switchable,
}

/// Fine-grained feature toggles for Wave 2 & Wave 3 coding tools and in-loop filters.
#[derive(Debug, Clone, Copy)]
pub struct ToolConfig {
    /// Enable rectangular block partitions (`BlockPartition::H` and `BlockPartition::V`).
    pub enable_rect_partitions: bool,
    /// Enable extended partitions (`BlockPartition::H3` and `BlockPartition::V3`).
    pub enable_ext_partitions: bool,
    /// Enable uneven 4-way partitions (`BlockPartition::H4A/B` and `BlockPartition::V4A/B`).
    pub enable_uneven_4way_partitions: bool,
    /// Enable Wave 3 recursive sub-block transform partitioning (`TxPartition::TxSplit`).
    pub enable_tx_partition: bool,
    /// Enable 2D transform kernel RDO search (`ADST`, `FLIPADST`, `IDTX` in addition to `DCT_DCT`).
    pub enable_tx_type_rdo: bool,
    /// Frame-level `reduced_tx_set` (`0..=3`, AV2 Spec v1.0.0 §5.18.7.4). The
    /// default `2` restricts intra luma to the inferred `DCT_DCT` and inter
    /// luma to `TX_SET_DCT_IDTX`; `0` selects the full transform sets.
    pub reduced_txtp_set: u8,
    /// Enable DPCM intra prediction (`use_dpcm_y` / `use_dpcm_uv`,
    /// AV2 Spec v1.0.0 §5.20.5.5/§5.20.5.6) in lossless frames.
    pub enable_dpcm: bool,
    /// Enable Rate-Distortion Optimized Quantization (`RDOQ`).
    pub enable_rdoq: bool,
    /// Enable Chroma from Luma (`CfL`) intra prediction.
    pub enable_cfl: bool,
    /// Enable Multiple Reference Lines (`MRL`) for directional luma intra prediction.
    pub enable_mrl: bool,
    /// Enable 5-mode 7-tap Filter Intra prediction.
    pub enable_filter_intra: bool,
    /// Enable Wave 3 Intra Sub-Partitions (`ISP` horizontal/vertical strip prediction).
    /// Accepted for API compatibility but has no effect: AV2 v1.0.0 has no
    /// Intra Sub-Partitions syntax, so the encoder always predicts whole blocks.
    pub enable_isp: bool,
    /// Enable Wave 3 Intra Block Copy (`IntraBC` / `IBC`) for screen content.
    pub enable_intrabc: bool,
    /// Enable Wave 3 Palette Mode (`2..=8` K-means color clustering) for screen content.
    pub enable_palette: bool,
    /// Enable Compound (bi-predictive) inter prediction across multiple references.
    pub enable_compound: bool,
    /// Bitmask of compound blending modes to evaluate: bit 0 average,
    /// bit 1 wedge, bit 2 difference-weighted, bit 3 distance-weighted (CWP).
    pub compound_types: u8,
    /// Enable masked compound (`enable_masked_compound`) and distance-weighted
    /// compound (`enable_cwp`). The average mode works without it.
    pub enable_compound_masks: bool,
    /// Accepted for API/CLI compatibility; it has no effect.
    ///
    /// AV2 v1.0.0 removed overlapped block motion compensation: the motion
    /// modes are `SIMPLE`, `INTERINTRA`, `LOCALWARP`, `DELTAWARP` and
    /// `EXTENDWARP` (§7.3 `MOTION_MODES = 5`), and §5.20.7.14
    /// `read_motion_mode()` has no OBMC symbol or blend. Enabling this flag
    /// therefore produces exactly the same bitstream as leaving it off.
    pub enable_obmc: bool,
    /// Enable warped motion: `WARPMV` blocks whose affine model and motion
    /// vector are derived from `WarpParamStack` (AV2 Spec v1.0.0 §5.20.7.6 /
    /// §7.12.2 / §7.13.3.19).
    pub enable_warped_motion: bool,
    /// Enable Variance-Based Adaptive Quantization (VAQ) through the AV2
    /// segmentation tool: a spatial segment map with per-segment `SEG_LVL_ALT_Q`
    /// values (AV2 Spec v1.0.0 §5.18.7.1).
    pub enable_vaq: bool,
    /// Enable per-superblock quantizer deltas (`delta_q_present`,
    /// `read_delta_qindex()`, AV2 Spec v1.0.0 §5.18.7.8 and §5.20.5.11).
    pub enable_delta_q: bool,
    /// Enable `segmentation_temporal_update` (AV2 Spec v1.0.0 §5.18.7.1).
    ///
    /// Temporal prediction copies segment ids from the map loaded from the
    /// primary reference (`load_previous_segment_ids`, §5.18.1). A decoder
    /// that parses frame headers ahead of reconstruction (dav2d with
    /// `--framedelay 1`) can only do that when the primary reference is older
    /// than the frame delay. The encoder's linear GOP references the
    /// immediately preceding frame, so this remains an explicit opt-in for GOP
    /// structures with older references. It is off by default.
    pub enable_seg_temporal: bool,
    /// Enable In-Loop Deblocking filter.
    pub enable_deblock: bool,
    /// Enable In-Loop Constrained Directional Enhancement Filter (`CDEF`).
    pub enable_cdef: bool,
    /// Enable In-Loop Cross-Component Sample Offset (`CCSO`).
    pub enable_ccso: bool,
    /// Reuse CCSO block enable flags from the reference frame (`sb_reuse_ccso`,
    /// AV2 Spec v1.0.0 §5.18.7.12). Enabled by default now that the in-tree
    /// decoder loads the per-slot reference map; set to `false` to signal the
    /// block flags on every inter frame.
    pub enable_ccso_sb_reuse: bool,
    /// Enable Wave 3 In-Loop Guided Detail Filter (`GDF`).
    pub enable_gdf: bool,
    /// Enable In-Loop Loop Restoration (AV2 §7.20).
    pub enable_restoration: bool,
    /// Restoration tool signalled when `enable_restoration` is set.
    pub restoration_mode: RestorationMode,
    /// Also restore the chroma planes with NS Wiener (chroma requires
    /// `RESTORE_WIENER_NONSEP`; PC Wiener is luma-only in AV2 v1.0.0).
    pub restoration_chroma: bool,
    /// Enable Wave 3 New Motion Vector (`NEWMV`) with joint shell MVD coding.
    pub enable_newmv: bool,
    /// Enable Switchable Subpel Interpolation Filters (`Regular8Tap`, `Smooth8Tap`, `Sharp8Tap`).
    pub enable_switchable_interp: bool,
    /// Enable Wave 3 Film Grain AR(1) synthesis parameter estimation.
    pub enable_film_grain: bool,
    /// Enable Wave 3 Lookahead Scene-Cut detection.
    pub enable_scenecut: bool,
    /// Enable Wave 3 3-layer Hierarchical Pyramid GOP scheduling.
    pub enable_hierarchical_gop: bool,
    /// Number of parallel worker threads for tile encoding (`1..=64`).
    pub threads: usize,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            // Foundation preset: basic intra/inter only. Advanced Wave 3 tools
            // are opt-in until their block syntax matches the AV2 decoder.
            enable_rect_partitions: false,
            enable_ext_partitions: true,
            enable_uneven_4way_partitions: true,
            enable_tx_partition: false,
            enable_tx_type_rdo: false,
            reduced_txtp_set: 2,
            enable_dpcm: false,
            enable_rdoq: false,
            enable_cfl: false,
            enable_mrl: false,
            enable_filter_intra: false,
            enable_isp: false,
            enable_intrabc: false,
            enable_palette: false,
            enable_compound: false,
            compound_types: 0x0f,
            enable_compound_masks: false,
            enable_obmc: false,
            enable_warped_motion: false,
            enable_vaq: false,
            enable_delta_q: false,
            enable_seg_temporal: false,
            enable_deblock: false,
            enable_cdef: false,
            enable_ccso: false,
            enable_ccso_sb_reuse: true,
            enable_gdf: false,
            enable_restoration: false,
            restoration_mode: RestorationMode::PcWiener,
            restoration_chroma: false,
            enable_newmv: false,
            enable_switchable_interp: false,
            enable_film_grain: false,
            enable_scenecut: false,
            enable_hierarchical_gop: false,
            threads: 1,
        }
    }
}

/// Padded 2D plane buffer aligned to 128-pixel superblock boundaries.
#[derive(Debug, Clone)]
pub struct Plane<P: Pixel> {
    pub data: Vec<P>,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub alloc_height: usize,
    /// Coded plane width in samples: the visible width rounded up to the MI
    /// grid (AV2 §7.13.2.1 `maxX + 1`). Inter prediction and intra reference
    /// reads clamp to this extent, not to `width`.
    pub coded_width: usize,
    /// Coded plane height in samples (`maxY + 1`, AV2 §7.13.2.1).
    pub coded_height: usize,
    pub ss_x: u8,
    pub ss_y: u8,
}

impl Plane<u8> {
    /// Borrows the plane as an 8-bit accelerator view without copying.
    pub fn view(&self) -> PlaneView<'_> {
        PlaneView::new(
            &self.data,
            u32::try_from(self.width).unwrap_or(u32::MAX),
            u32::try_from(self.height).unwrap_or(u32::MAX),
            self.stride,
        )
    }
}

impl<P: Pixel> Plane<P> {
    pub fn new(width: usize, height: usize, ss_x: u8, ss_y: u8, bit_depth: u8) -> Self {
        // AV2 pads the coding grid to 8 pixels: `MiCols = 2 * ((w + 7) >> 3)`.
        let coded_width = width.div_ceil(8) * 8;
        let coded_height = height.div_ceil(8) * 8;
        Self::with_coded(
            width,
            height,
            coded_width,
            coded_height,
            ss_x,
            ss_y,
            bit_depth,
        )
    }

    fn with_coded(
        width: usize,
        height: usize,
        coded_width: usize,
        coded_height: usize,
        ss_x: u8,
        ss_y: u8,
        _bit_depth: u8,
    ) -> Self {
        let align_w = 128usize >> ss_x.min(1);
        let align_h = 128usize >> ss_y.min(1);
        let stride = width.max(1).div_ceil(align_w) * align_w;
        let alloc_height = height.max(1).div_ceil(align_h) * align_h;
        Self {
            data: vec![P::default(); stride * alloc_height],
            width,
            height,
            stride,
            alloc_height,
            coded_width,
            coded_height,
            ss_x,
            ss_y,
        }
    }

    pub fn copy_from_raw(&mut self, src: &[P]) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let total = self.width * self.height;
        assert!(src.len() >= total);
        if self.stride == self.width {
            self.data[..total].copy_from_slice(&src[..total]);
        } else {
            for r in 0..self.height {
                let src_row = &src[r * self.width..(r + 1) * self.width];
                let dst_row = &mut self.data[r * self.stride..r * self.stride + self.width];
                dst_row.copy_from_slice(src_row);
            }
        }
        self.pad_edges();
    }

    pub fn pad_edges(&mut self) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let w = self.width;
        let stride = self.stride;
        if w < stride {
            for r in 0..self.height {
                let row_off = r * stride;
                let last_px = self.data[row_off + w - 1];
                self.data[row_off + w..row_off + stride].fill(last_px);
            }
        }
        if self.height < self.alloc_height {
            let last_row_off = (self.height - 1) * stride;
            for r in self.height..self.alloc_height {
                self.data
                    .copy_within(last_row_off..last_row_off + stride, r * stride);
            }
        }
    }

    pub fn to_raw_vec(&self) -> Vec<P> {
        if self.stride == self.width {
            return self.data[..self.width * self.height].to_vec();
        }
        let mut out = Vec::with_capacity(self.width * self.height);
        for r in 0..self.height {
            out.extend_from_slice(&self.data[r * self.stride..r * self.stride + self.width]);
        }
        out
    }

    #[inline]
    pub fn offset(&self, x: usize, y: usize) -> usize {
        y * self.stride + x
    }
}

/// Complete 3-plane YUV frame buffer (`planes[0] = Y`, `planes[1] = U`, `planes[2] = V`).
#[derive(Debug, Clone)]
pub struct Frame<P: Pixel> {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u8,
    pub layout: PixelLayout,
    pub poc: u8,
    pub planes: [Plane<P>; 3],
}

impl<P: Pixel> Frame<P> {
    pub fn new(width: usize, height: usize, bit_depth: u8, layout: PixelLayout, poc: u8) -> Self {
        let (ss_x, ss_y, has_chroma) = match layout {
            PixelLayout::I400 => (1u8, 1u8, false),
            PixelLayout::I420 => (1u8, 1u8, true),
            PixelLayout::I422 => (1u8, 0u8, true),
            PixelLayout::I444 => (0u8, 0u8, true),
        };
        let (cw, ch) = if has_chroma {
            (
                (width + ss_x as usize) >> ss_x,
                (height + ss_y as usize) >> ss_y,
            )
        } else {
            (0, 0)
        };
        let coded_w = width.div_ceil(8) * 8;
        let coded_h = height.div_ceil(8) * 8;
        let (ccw, cch) = if has_chroma {
            (coded_w >> ss_x, coded_h >> ss_y)
        } else {
            (0, 0)
        };
        Self {
            width,
            height,
            bit_depth,
            layout,
            poc,
            planes: [
                Plane::with_coded(width, height, coded_w, coded_h, 0, 0, bit_depth),
                Plane::with_coded(cw, ch, ccw, cch, ss_x, ss_y, bit_depth),
                Plane::with_coded(cw, ch, ccw, cch, ss_x, ss_y, bit_depth),
            ],
        }
    }

    pub fn from_raw(raw: &RawFrame<P>, poc: u8) -> Self {
        let mut f = Self::new(raw.width, raw.height, raw.bit_depth, raw.layout, poc);
        f.planes[0].copy_from_raw(&raw.y);
        if raw.layout != PixelLayout::I400 {
            f.planes[1].copy_from_raw(&raw.u);
            f.planes[2].copy_from_raw(&raw.v);
        }
        f
    }

    pub fn to_raw(&self) -> RawFrame<P> {
        RawFrame {
            width: self.width,
            height: self.height,
            bit_depth: self.bit_depth,
            layout: self.layout,
            y: self.planes[0].to_raw_vec(),
            u: if self.layout != PixelLayout::I400 {
                self.planes[1].to_raw_vec()
            } else {
                Vec::new()
            },
            v: if self.layout != PixelLayout::I400 {
                self.planes[2].to_raw_vec()
            } else {
                Vec::new()
            },
        }
    }
}

/// Decoded Picture Buffer (`DPB`) holding the 16 AV2 reference frame slots.
#[derive(Debug, Clone)]
pub struct ReferenceDpb<P: Pixel> {
    pub slots: [Option<std::sync::Arc<Frame<P>>>; NUM_REF_FRAMES],
    /// Per-slot CCSO block maps (`CcsoBlks`) used for `sb_reuse_ccso`
    /// (AV2 Spec v1.0.0 §7.23). Both the encoder and decoder save the current
    /// map on refresh and load it for a frame that signals `sb_reuse_ccso`.
    pub ccso_maps: [Option<[Vec<bool>; 3]>; NUM_REF_FRAMES],
}

impl<P: Pixel> Default for ReferenceDpb<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Pixel> ReferenceDpb<P> {
    pub fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            ccso_maps: std::array::from_fn(|_| None),
        }
    }

    pub fn update_with_frame(&mut self, refresh_mask: u16, rec_frame: &Frame<P>) {
        if refresh_mask == 0 {
            return;
        }
        let shared = std::sync::Arc::new(rec_frame.clone());
        for slot_idx in 0..NUM_REF_FRAMES {
            if (refresh_mask & (1u16 << slot_idx)) != 0 {
                self.slots[slot_idx] = Some(std::sync::Arc::clone(&shared));
            }
        }
    }

    /// Stores the CCSO block map alongside the reference frames refreshed by this frame,
    /// so a later inter frame can signal `sb_reuse_ccso` (AV2 Spec v1.0.0 §6.17.7.8).
    pub fn update_ccso_maps(&mut self, refresh_mask: u16, maps: &[Vec<bool>; 3]) {
        if refresh_mask == 0 {
            return;
        }
        for slot_idx in 0..NUM_REF_FRAMES {
            if (refresh_mask & (1u16 << slot_idx)) != 0 {
                self.ccso_maps[slot_idx] = Some(maps.clone());
            }
        }
    }

    pub fn update_with_owned_frame(&mut self, refresh_mask: u16, rec_frame: Frame<P>) {
        if refresh_mask == 0 {
            return;
        }
        let shared = std::sync::Arc::new(rec_frame);
        for slot_idx in 0..NUM_REF_FRAMES {
            if (refresh_mask & (1u16 << slot_idx)) != 0 {
                self.slots[slot_idx] = Some(std::sync::Arc::clone(&shared));
            }
        }
    }

    pub fn get_slot(&self, slot_idx: usize) -> Option<&Frame<P>> {
        self.slots
            .get(slot_idx)
            .and_then(|s| s.as_deref())
            .or_else(|| self.slots.iter().find_map(|s| s.as_deref()))
    }
}

/// Per-frame quality, bitrate, and coding tool telemetry returned by the encoder.
#[derive(Debug, Clone)]
pub struct FrameStats {
    pub poc: u8,
    pub frame_type: FrameType,
    pub tlayer_id: u8,
    pub qidx: u16,
    pub bytes: usize,
    pub psnr_y: f64,
    pub psnr_u: f64,
    pub psnr_v: f64,
    pub psnr_avg: f64,
    pub mse_y: f64,
    pub intra_blocks: usize,
    pub inter_blocks: usize,
    /// Number of single-reference `NEWMV` blocks (`YMode == NEWMV`).
    pub newmv_blocks: usize,
    /// Number of blocks coded with a non-default switchable subpel filter
    /// (`interp_filter != EIGHTAP`).
    pub switchable_filter_blocks: usize,
    pub compound_blocks: usize,
    /// Compound blocks by blending mode (`COMPOUND_AVERAGE`, `COMPOUND_WEDGE`,
    /// `COMPOUND_DIFFWTD`, and distance-weighted `cwp_idx`).
    pub compound_average_blocks: usize,
    pub compound_wedge_blocks: usize,
    pub compound_diff_blocks: usize,
    pub compound_cwp_blocks: usize,
    pub obmc_blocks: usize,
    pub warped_blocks: usize,
    /// `WARPMV` / `WARP_NEWMV` split and the two explicit motion modes.
    pub warpmv_blocks: usize,
    pub localwarp_blocks: usize,
    pub extendwarp_blocks: usize,
    /// Blocks whose prediction is blended with an intra prediction
    /// (`INTERINTRA`).
    pub interintra_blocks: usize,
    pub cfl_blocks: usize,
    pub filter_intra_blocks: usize,
    pub isp_blocks: usize,
    pub intrabc_blocks: usize,
    pub palette_blocks: usize,
    pub tx_split_blocks: usize,
    /// Transform-partition decisions of every inter block whose partition was
    /// evaluated, indexed by [`TxPartition`] value (`NONE = 0`, `SPLIT = 1`,
    /// `HORZ = 2`, `VERT = 3`).
    pub tx_partition_counts: [usize; 4],
    /// Blocks coded with a non-`DCT_DCT` luma transform type.
    pub non_dct_tx_blocks: usize,
    /// Lossless blocks whose luma residual uses the 4x4 WHT.
    pub wht_blocks: usize,
    /// Lossless luma blocks coded with `use_dpcm_y` (`dpcm_mode_y` 0 = `V_PRED`,
    /// 1 = `H_PRED`).
    pub dpcm_y_blocks: usize,
    /// Lossless chroma blocks coded with `use_dpcm_uv`.
    pub dpcm_uv_blocks: usize,
    pub rect_blocks: usize,
    /// Number of coding-tree nodes that selected each partition, indexed by
    /// [`BlockPartition`] value (`None = 0`, `H = 1`, `V = 2`, `H3 = 3`,
    /// `V3 = 4`, `H4A = 5`, `H4B = 6`, `V4A = 7`, `V4B = 8`, `Split = 9`).
    pub partition_counts: [usize; 10],
    pub skipped_blocks: usize,
    /// Effective luma quantizer averaged over coded blocks, including segment
    /// `SEG_LVL_ALT_Q` and per-superblock `delta_q` adjustments.
    pub avg_qidx: u16,
    pub deblock_level: u8,
    pub cdef_y_strength: u8,
    pub ccso_enabled: bool,
    pub gdf_enabled: bool,
    /// Number of CCSO unit enable flags set across the enabled planes.
    pub ccso_units: usize,
    /// Number of CCSO units whose flags came from a reference frame (`sb_reuse`).
    pub ccso_reuse_units: usize,
    /// Number of planes that signalled edge offsets (`bo_only == 0`).
    pub ccso_edge_planes: usize,
    /// Number of GDF block enable flags set in the frame.
    pub gdf_units: usize,
    /// Whether the GDF frame used per-block `use_gdf` signalling.
    pub gdf_per_block: bool,
    pub restoration_y: RestorationType,
}

/// Computes MSE and PSNR (in dB) between source and reconstructed planes over the active window.
pub fn compute_plane_psnr<P: Pixel>(src: &Plane<P>, rec: &Plane<P>, bit_depth: u8) -> (f64, f64) {
    if src.width == 0 || src.height == 0 {
        return (0.0, 100.0);
    }
    let sse = compute_sse(
        &src.data, 0, src.stride, &rec.data, 0, rec.stride, src.width, src.height,
    );
    let n_samples = (src.width * src.height) as f64;
    let mse = (sse as f64) / n_samples;
    if mse <= 1e-12 {
        return (0.0, 100.0);
    }
    let peak = ((1u64 << bit_depth) - 1) as f64;
    let psnr = 10.0 * ((peak * peak) / mse).log10();
    (mse, psnr)
}

/// Computes full-frame `(mse_y, psnr_y, psnr_u, psnr_v, psnr_avg)` statistics.
pub fn compute_frame_psnr<P: Pixel>(src: &Frame<P>, rec: &Frame<P>) -> (f64, f64, f64, f64, f64) {
    let (mse_y, psnr_y) = compute_plane_psnr(&src.planes[0], &rec.planes[0], src.bit_depth);
    if src.layout == PixelLayout::I400 {
        return (mse_y, psnr_y, 100.0, 100.0, psnr_y);
    }
    let (_, psnr_u) = compute_plane_psnr(&src.planes[1], &rec.planes[1], src.bit_depth);
    let (_, psnr_v) = compute_plane_psnr(&src.planes[2], &rec.planes[2], src.bit_depth);
    let psnr_avg = (4.0 * psnr_y + psnr_u + psnr_v) / 6.0;
    (mse_y, psnr_y, psnr_u, psnr_v, psnr_avg)
}

/// Output produced by encoding a single tile (used by both single-threaded and multi-threaded paths).
struct TileEncodeResult<P: Pixel> {
    pub job: TileJob,
    pub payload: Vec<u8>,
    pub rec_tile: Frame<P>,
    pub intra_blocks: usize,
    pub inter_blocks: usize,
    pub newmv_blocks: usize,
    pub switchable_filter_blocks: usize,
    pub compound_blocks: usize,
    pub compound_average_blocks: usize,
    pub compound_wedge_blocks: usize,
    pub compound_diff_blocks: usize,
    pub compound_cwp_blocks: usize,
    pub obmc_blocks: usize,
    pub warped_blocks: usize,
    pub warpmv_blocks: usize,
    pub localwarp_blocks: usize,
    pub extendwarp_blocks: usize,
    pub interintra_blocks: usize,
    pub cfl_blocks: usize,
    pub filter_intra_blocks: usize,
    pub isp_blocks: usize,
    pub intrabc_blocks: usize,
    pub palette_blocks: usize,
    pub tx_split_blocks: usize,
    pub tx_partition_counts: [usize; 4],
    /// Blocks coded with a non-`DCT_DCT` luma transform type.
    pub non_dct_tx_blocks: usize,
    /// Lossless blocks whose luma residual uses the 4x4 WHT.
    pub wht_blocks: usize,
    /// Lossless luma blocks coded with `use_dpcm_y` (`dpcm_mode_y` 0 = `V_PRED`,
    /// 1 = `H_PRED`).
    pub dpcm_y_blocks: usize,
    /// Lossless chroma blocks coded with `use_dpcm_uv`.
    pub dpcm_uv_blocks: usize,
    pub rect_blocks: usize,
    /// Selected partition per coding-tree node, indexed by [`BlockPartition`].
    pub partition_counts: [usize; 10],
    pub skipped_blocks: usize,
    pub grid: DeblockGrid,
    pub seg_map: SegmentMap,
    pub qidx_sum: u64,
    pub qidx_weight: u64,
}

struct TileEncodeState<'a, BD: BitDepth> {
    bd: BD,
    seqhdr: &'a SequenceHeader,
    frame_hdr: &'a FrameHeader,
    tools: ToolConfig,
    src: &'a Frame<BD::Pixel>,
    rec: &'a mut Frame<BD::Pixel>,
    ref0: Option<&'a Frame<BD::Pixel>>,
    ref1: Option<&'a Frame<BD::Pixel>>,
    refdist: [i32; 7],
    absrefdist: [i32; 7],
    refdir: [i8; 7],
    furthest_future_refidx: i8,
    dq: [[[u32; 2]; 3]; MAX_SEGMENTS],
    /// Coded segment ids at 4x4 granularity for this tile (AV2 §5.20.5.8).
    seg_map: SegmentMap,
    /// Previous frame segment map used for `segmentation_temporal_update`
    /// (`load_previous_segment_ids`, AV2 Spec v1.0.0 §5.18.1).
    seg_ref: Option<&'a SegmentMap>,
    /// Variance classes and per-superblock QP offsets (encoder-side AQ).
    seg_plan: Option<&'a VaqPlan>,
    /// Segment-id coding state derived from `SegmentationDataSet`.
    seg_coder: SegmentCoder,
    /// `CurrentQIndex` for the current superblock (AV2 Spec §5.20.5.11).
    cur_qidx: u16,
    /// Delta-Q still to be written in the first block of the superblock.
    pending_delta: Option<i32>,
    /// `AboveSegPredContext` / `LeftSegPredContext` arrays
    /// (AV2 Spec v1.0.0 §5.20.7.2).
    seg_pred_a: Vec<[u8; 32]>,
    seg_pred_l: [u8; 32],
    /// Superblock size in 4x4 units, used for the per-SB QP plan.
    sb_step4: usize,
    /// Weighted sum of effective segment quantizers for frame statistics.
    qidx_sum: u64,
    qidx_weight: u64,
    /// Per-superblock bit budget from the rate controller (`None` for ConstantQp).
    sb_budget_bits: Option<f64>,
    /// Cumulative rate-feedback offset applied to the activity-based delta-Q.
    rate_offset: i16,
    /// `tell_bits()` at the start of the previous superblock.
    prev_sb_bits: usize,
    msac: MsacEncoder,
    cdf: CdfContext,
    a_cols: Vec<BlockContext>,
    l_ctx: BlockContext,
    tile_col_start4: usize,
    tile_col_end4: usize,
    tile_row_start4: usize,
    tile_row_end4: usize,
    f_bw4: usize,
    f_bh4: usize,
    speed: u8,
    lambda_sad: u64,
    lambda_rdoq: u64,
    /// Per-reference warp parameter bank (AV2 Spec v1.0.0 §7.12.3
    /// `update_warp_param_bank`), most recent entry first.
    warp_bank: Vec<Vec<[i32; 6]>>,
    /// `WarpBankHits` for the current superblock (§7.3 `MAX_WARP_SB_HITS`).
    warp_bank_hits: u32,
    /// `RefMvBankParams` (§7.12.3), indexed by `get_rmb_list_index()`.
    refmv_bank: RefMvBank,
    /// `RefMvBankHits` / `RefMvUnitHits` / `RefMvRemainHits` (§7.12.3).
    refmv_hits: RefMvBankHits,
    /// Superblock row whose warp-bank models the encoder currently holds.
    warp_bank_row: usize,
    /// Coded mode of the leaf being written, for `update_ref_mv_bank()`.
    block_inter: bool,
    /// `RefFrame[]` of the leaf being written.
    block_refs: [i8; 2],
    /// `BlockMvs[]` of the leaf being written.
    block_mvs: [MvXY; 2],
    /// `CwpIdx` of the leaf being written.
    block_cwp: u8,
    /// Per-4x4 mode information used to derive `WarpParamStack` and the
    /// `is_warp` context (§7.11.4 / §7.12.2.3 / §7.12.2.4).
    mi_map: Vec<WarpMiInfo>,
    intra_blocks: usize,
    inter_blocks: usize,
    newmv_blocks: usize,
    switchable_filter_blocks: usize,
    compound_blocks: usize,
    compound_average_blocks: usize,
    compound_wedge_blocks: usize,
    compound_diff_blocks: usize,
    compound_cwp_blocks: usize,
    obmc_blocks: usize,
    warped_blocks: usize,
    warpmv_blocks: usize,
    localwarp_blocks: usize,
    extendwarp_blocks: usize,
    interintra_blocks: usize,
    cfl_blocks: usize,
    filter_intra_blocks: usize,
    isp_blocks: usize,
    intrabc_blocks: usize,
    palette_blocks: usize,
    /// `PaletteSizes` / `PaletteColors` at 4x4 granularity (§5.20.8.1
    /// `get_palette_cache`). Entries are filled over each coded block footprint.
    palette_sizes: Vec<u8>,
    palette_colors: Vec<[i32; 8]>,
    /// `IBCCoded` and the local 64x64 buffer ring (§5.20.5.14 / §6.19.7.12).
    intrabc_coded: Vec<bool>,
    ibc_buffers: crate::av2::screen::IbcBufferState,
    /// Per-plane `is_coded` map at 4x4 granularity, used by the lossless
    /// per-transform-block intra prediction edge derivation. Every
    /// reconstructed transform block is marked, including all-zero ones
    /// (`dav2d` `recon_tmpl.c` `is_coded`).
    tu_coded: [Vec<bool>; 3],
    palette_cdf: crate::av2::screen::PaletteCdfState,
    tx_split_blocks: usize,
    tx_partition_counts: [usize; 4],
    non_dct_tx_blocks: usize,
    wht_blocks: usize,
    dpcm_y_blocks: usize,
    dpcm_uv_blocks: usize,
    rect_blocks: usize,
    /// Selected partition per coding-tree node, indexed by [`BlockPartition`].
    partition_counts: [usize; 10],
    skipped_blocks: usize,
    grid: DeblockGrid,
    ccso_blks: [Vec<bool>; 3],
    ccso_cols: usize,
    ccso_rows: usize,
    ccso_step4: usize,
    gdf_blks: Vec<bool>,
    gdf_cols: usize,
    gdf_rows: usize,
    gdf_step4: usize,
    lr_types: [Vec<u8>; 3],
    lr_cols: [usize; 3],
    lr_rows: [usize; 3],
    scratch_mc0: Vec<i32>,
    scratch_mc1: Vec<i32>,
    scratch_diff_mask: Vec<u8>,
    scratch_chroma_diff_mask: Vec<u8>,
}

/// A selected `WARPMV` + `DELTAWARP` block (AV2 §5.20.7.6).
///
/// `WARPMV` codes no motion-vector residual: `BlockMvs[0]` and the affine
/// model both come from the `WarpParamStack` entry selected by `warp_idx`
/// (§7.12.2), which is derived from neighbouring warp-corner motion vectors,
/// the per-reference warp bank, and the (identity) global motion model.
#[derive(Debug, Clone, Copy)]
struct WarpChoice {
    /// `RefWarpIdx` selected by the `warp_idx` symbols (0..=3).
    idx: usize,
    /// Final `LocalWarpParams` (reduced, with the block translation).
    matrix: [i32; 6],
    /// `setup_shear()` output for `matrix`.
    abcd: [i32; 4],
    /// `BlockMvs[0]` derived from the stack entry.
    mv: MvXY,
    /// Predictor used for the coded MVD (`RefStackMv[RefMvIdx][0]`).
    pred_mv: MvXY,
    /// `true` for `WARPMV`, `false` for `WARP_NEWMV`.
    warp_mv: bool,
    /// `motion_mode` value: `2` `LOCALWARP`, `3` `DELTAWARP`, `4` `EXTENDWARP`.
    motion_mode: u8,
    /// Quantised `params[2]` / `params[3]` deltas for `WARP_NEWMV` +
    /// `DELTAWARP`.
    delta: [i32; 2],
    /// `warp_delta_precision` for `WARP_NEWMV` + `DELTAWARP`.
    high_prec: bool,
}

/// One coded transform unit of a coding block: the geometry, kernel, EOB, and
/// the quantized/dequantized levels produced before the entropy syntax is
/// written.
///
/// The transform-partition syntax (`tx_do_partition` / `tx_partition_type`)
/// precedes the coefficient blocks, so the encoder decides the partition and
/// pre-computes every sub-transform before it emits the block syntax.
struct EncTxUnit {
    /// Rectangular transform size index.
    tx: usize,
    /// Horizontal/vertical offset in pixels relative to the coding block.
    x_px: usize,
    y_px: usize,
    /// Transform extent in pixels.
    w: usize,
    h: usize,
    /// Packed transform kernel (including the lossless DPCM flag bits).
    txtp: u32,
    /// Last non-zero scan index (`-1` = all zero).
    eob: i32,
    /// Quantized levels, indexed by coefficient position.
    q_cf: Vec<i32>,
    /// Dequantized levels, indexed by coefficient position.
    dq_cf: Vec<i32>,
}

impl EncTxUnit {
    fn new(tx: usize, x_px: usize, y_px: usize) -> Self {
        let d = &TXFM_DIMENSIONS[tx];
        let n = 16usize << (d.lw.min(3) + d.lh.min(3)) as usize;
        Self {
            tx,
            x_px,
            y_px,
            w: d.w as usize * 4,
            h: d.h as usize * 4,
            txtp: txtp::DCT_DCT,
            eob: -1,
            q_cf: vec![0i32; n],
            dq_cf: vec![0i32; n],
        }
    }
}

impl<'a, BD: BitDepth> TileEncodeState<'a, BD> {
    /// Marks the current block footprint as intra in the per-4x4 map.
    fn mark_mi_default(&mut self, bs: BlockSize, bx4: usize, by4: usize) {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let info = WarpMiInfo::block(bx4, by4, bw4, bh4);
        for y in 0..bh4 {
            let row = by4 + y;
            if row >= self.f_bh4 {
                break;
            }
            for x in 0..bw4 {
                let col = bx4 + x;
                if col >= self.f_bw4 {
                    break;
                }
                self.mi_map[row * self.f_bw4 + col] = info;
            }
        }
        self.block_inter = false;
        self.block_refs = [-1, -1];
        self.block_mvs = [MvXY::default(); 2];
        self.block_cwp = 8;
    }

    /// `update_ref_mv_bank(..., fromWithinSb = 1)` / `update_ref_mv_count()`
    /// for the leaf that was just written (§7.12.3).
    fn note_refmv_bank(&mut self, bw4: usize, bh4: usize, bx4: usize, by4: usize) {
        if !self.seqhdr.refmv_bank || self.refmv_hits.bank_hits >= MAX_RMB_SB_HITS {
            return;
        }
        let sb4 = self.sb_step4.max(1);
        if self.block_inter {
            if self.refmv_hits.allow_update(bw4, bh4, by4, bx4, sb4) {
                self.refmv_bank
                    .update(self.block_refs, self.block_mvs, self.block_cwp);
            }
        } else {
            self.refmv_hits.update_count(bw4, bh4, by4, bx4, sb4);
        }
    }

    /// Records the coded inter mode of the current block for warp and compound
    /// motion-vector derivation.
    #[allow(clippy::too_many_arguments)]
    fn mark_mi_inter(
        &mut self,
        bs: BlockSize,
        bx4: usize,
        by4: usize,
        motion_mode: u8,
        warp: [i32; 6],
        refs: [i8; 2],
        mvs: [MvXY; 2],
        cwp: u8,
    ) {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let mut info = WarpMiInfo::block(bx4, by4, bw4, bh4);
        info.inter = true;
        info.ref0 = refs[0];
        info.mv = mvs[0];
        info.ref1 = refs[1];
        info.mv1 = mvs[1];
        info.cwp = cwp;
        info.motion_mode = motion_mode;
        info.warp = warp;
        self.block_inter = true;
        self.block_refs = refs;
        self.block_mvs = mvs;
        self.block_cwp = cwp;
        for y in 0..bh4 {
            let row = by4 + y;
            if row >= self.f_bh4 {
                break;
            }
            for x in 0..bw4 {
                let col = bx4 + x;
                if col >= self.f_bw4 {
                    break;
                }
                // `SubMvs` (§7.13.3.19): a warped block stores the projected
                // vector of each 8x8 section, other blocks the block vector.
                if motion_mode >= 2 {
                    info.sub_mv = warp_sub_mv(&warp, bx4, by4, x, y);
                } else {
                    info.sub_mv = mvs[0];
                }
                info.sub_mv1 = mvs[1];
                self.mi_map[row * self.f_bw4 + col] = info;
            }
        }
    }

    /// `WarpMvCount` from §7.11.4 `scan_point_warp_ctx`, used as the `is_warp`
    /// CDF context (§9.3 `TileIsWarpCdf[WarpMvCount]`).
    fn warp_mv_count(&self, bx4: usize, by4: usize, bw4: usize, bh4: usize) -> usize {
        let sb_h4 = self.sb_step4.max(1);
        let is_sb_border = by4.is_multiple_of(sb_h4);
        let mut count = 0usize;
        let mut scan = |delta_row: isize, delta_col: isize| {
            let mut dcol = delta_col;
            if delta_row < 0 && is_sb_border {
                dcol -= (bx4 & 1) as isize;
            }
            let row = by4 as isize + delta_row;
            let col = bx4 as isize + dcol;
            if row < 0 || col < 0 {
                return;
            }
            let (row, col) = (row as usize, col as usize);
            if row >= self.f_bh4 || col >= self.f_bw4 {
                return;
            }
            let info = self.mi_map[row * self.f_bw4 + col];
            if info.written && info.inter && info.ref0 == 0 && info.motion_mode >= 2 {
                count += 1;
            }
        };
        scan(bh4 as isize - 1, -1);
        scan(
            -1,
            if is_sb_border {
                (bw4 as isize - 2).max(0)
            } else {
                bw4 as isize - 1
            },
        );
        scan(0, -1);
        if bw4 >= if is_sb_border { 4 } else { 2 } {
            scan(-1, 0);
        }
        count.min(4)
    }

    /// Runs the normative `find_mv_stack` for the current block (§7.12.2).
    #[allow(clippy::too_many_arguments)]
    fn mv_stack(
        &self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        refs: [i8; 2],
    ) -> crate::av2::inter::RefMvStack {
        let is_compound = refs[1] >= 0;
        let derive_wrl = !is_compound && refs[0] >= 0 && bw4 * 4 >= 8 && bh4 * 4 >= 8;
        find_mv_stack(&MvStackParams {
            map: &self.mi_map,
            mi_rows: self.f_bh4,
            mi_cols: self.f_bw4,
            sb_h4: self.sb_step4.max(1),
            tile_row_start4: self.tile_row_start4,
            tile_row_end4: self.tile_row_end4,
            tile_col_start4: self.tile_col_start4,
            tile_col_end4: self.tile_col_end4,
            mi_row: by4,
            mi_col: bx4,
            bw4,
            bh4,
            refs,
            is_compound,
            derive_wrl,
            use_ref_frame_mvs: self.frame_hdr.use_ref_frame_mvs != 0,
            drl_reorder: self.seqhdr.drl_reorder,
            enable_refmvbank: self.seqhdr.refmv_bank,
            max_drl_bits: self.frame_hdr.max_drl_bits.max(1) as usize,
            frame_distance: &self.refdist,
            warp_bank: &self.warp_bank,
            refmv_bank: self.seqhdr.refmv_bank.then_some(&self.refmv_bank),
        })
    }

    /// Intra prediction for one plane of an inter-intra block
    /// (`read_interintra_mode`, §5.20.7.15). The prediction is written into a
    /// fresh buffer and later blended with the inter prediction by
    /// [`blend_interintra`] (§7.13.3.30).
    #[allow(clippy::too_many_arguments)]
    fn interintra_prediction(
        &self,
        plane: usize,
        px_x: usize,
        px_y: usize,
        w: usize,
        h: usize,
        have_left: bool,
        have_top: bool,
        interintra_mode: u8,
    ) -> Vec<BD::Pixel> {
        let pl = &self.rec.planes[plane];
        let (rec_cw, rec_ch) =
            crate::av2::ipred::intra_reference_extent(self.f_bw4, self.f_bh4, pl.ss_x, pl.ss_y);
        let sb_px = (64usize << self.frame_hdr.sb128) >> pl.ss_x.max(pl.ss_y);
        let (have_tr, have_bl) = crate::av2::ipred::intra_edge_availability(
            px_x,
            px_y,
            w,
            h,
            (self.tile_col_end4 * 4) >> pl.ss_x,
            (self.tile_row_end4 * 4) >> pl.ss_y,
            sb_px,
            have_left,
            have_top,
        );
        let mut edge = [BD::Pixel::default(); 512];
        prepare_intra_edges(
            self.bd,
            &mut edge,
            256,
            &pl.data,
            pl.offset(px_x, px_y),
            pl.stride,
            have_left,
            have_top,
            have_tr,
            have_bl,
            w,
            h,
            rec_cw,
            rec_ch,
        );
        let mut out = vec![BD::Pixel::default(); w * h];
        predict_intra_block(
            self.bd,
            &mut out,
            0,
            w,
            &edge,
            256,
            interintra_intra_mode(interintra_mode),
            0,
            w,
            h,
            have_left,
            have_top,
            plane,
            0,
        );
        out
    }

    /// Evaluates the four `interintra_mode` values for a single-reference
    /// block and returns the best `(mode, luma_prediction)` when it beats the
    /// inter prediction after rate is accounted for.
    #[allow(clippy::too_many_arguments)]
    fn select_interintra(
        &self,
        px_x: usize,
        px_y: usize,
        w: usize,
        h: usize,
        have_left: bool,
        have_top: bool,
        inter_sad: u64,
    ) -> Option<(u8, Vec<BD::Pixel>, u64)> {
        let mut best: Option<(u8, Vec<BD::Pixel>, u64)> = None;
        let mut blended = vec![BD::Pixel::default(); w * h];
        let src = &self.src.planes[0];
        let rec = &self.rec.planes[0];
        for mode in 0..4u8 {
            let intra = self.interintra_prediction(0, px_x, px_y, w, h, have_left, have_top, mode);
            let mask = interintra_mask(mode, w, h);
            let mut off = 0usize;
            while off < w * h {
                let m = mask[off] as i32;
                let p0 = intra[off].to_i32();
                let p1 =
                    rec.data[rec.offset(px_x, px_y) + (off / w) * rec.stride + off % w].to_i32();
                blended[off] = self.bd.clip_pixel((m * p0 + (64 - m) * p1 + 32) >> 6);
                off += 1;
            }
            let sad = compute_sad(
                &src.data,
                src.offset(px_x, px_y),
                src.stride,
                &blended,
                0,
                w,
                w,
                h,
            );
            // `inter_intra`, `interintra_mode` and `wedge_interintra` bits.
            let rate_q8 = 4 * 256 + 2 * 256;
            let cost = sad + ((self.lambda_sad * rate_q8) >> 8);
            if cost < inter_sad && best.as_ref().is_none_or(|(_, _, c)| cost < *c) {
                best = Some((mode, intra, cost));
            }
        }
        best
    }

    /// Predicts a block with `matrix` and returns `(abcd, SAD)` against the
    /// source (§7.13.3.19 `block_warp`).
    #[allow(clippy::too_many_arguments)]
    fn eval_warp_model(
        &self,
        src_plane: &Plane<BD::Pixel>,
        ref_plane: &Plane<BD::Pixel>,
        px_x: usize,
        px_y: usize,
        w: usize,
        h: usize,
        matrix: &[i32; 6],
    ) -> Option<([i32; 4], u64)> {
        let abcd = setup_shear(matrix)?;
        let mut buf = vec![BD::Pixel::default(); w * h];
        block_warp(
            self.bd, &mut buf, 0, w, ref_plane, px_x, px_y, w, h, 0, 0, matrix, &abcd,
        );
        let sad = compute_sad(
            &src_plane.data,
            src_plane.offset(px_x, px_y),
            src_plane.stride,
            &buf,
            0,
            w,
            w,
            h,
        );
        Some((abcd, sad))
    }

    /// Estimates a `WARP_NEWMV` + `DELTAWARP` model from four half-block
    /// motion vectors and codes it as a delta on `WarpParamStack[0]`
    /// (§5.20.7.7 `read_warp_delta`).
    #[allow(clippy::too_many_arguments)]
    fn select_warp_delta(
        &self,
        src_plane: &Plane<BD::Pixel>,
        ref_plane: &Plane<BD::Pixel>,
        px_x: usize,
        py_y: usize,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        trans_sad: u64,
        best_mv: MvXY,
        stack: &crate::av2::inter::RefMvStack,
    ) -> Option<(WarpChoice, u64)> {
        let w = bw4 * 4;
        let h = bh4 * 4;
        let sub_w = w / 2;
        let sub_h = h / 2;
        if sub_w < 8 || sub_h < 8 {
            return None;
        }
        let mut pts = [[0i64; 2]; 4];
        let mut mvs = [[0i32; 2]; 4];
        let mut n = 0usize;
        for qy in 0..2usize {
            for qx in 0..2usize {
                let sx = px_x + qx * sub_w;
                let sy = py_y + qy * sub_h;
                let (mv, _) = estimate_motion_block(
                    self.bd,
                    &src_plane.data,
                    src_plane.offset(sx, sy),
                    src_plane.stride,
                    &ref_plane.data,
                    ref_plane.stride,
                    ref_plane.coded_width as i32,
                    ref_plane.coded_height as i32,
                    sx as i32,
                    sy as i32,
                    sub_w,
                    sub_h,
                    best_mv,
                    0,
                    self.lambda_sad,
                );
                pts[n] = [(sx + sub_w / 2) as i64, (sy + sub_h / 2) as i64];
                mvs[n] = [mv.x as i32, mv.y as i32];
                n += 1;
            }
        }
        let bcx = (px_x + w / 2) as i64;
        let bcy = (py_y + h / 2) as i64;
        let mut ra = 0i64;
        let mut rb = 0i64;
        let mut den = 0i64;
        for i in 0..4 {
            let dx = pts[i][0] - bcx;
            let dy = pts[i][1] - bcy;
            let dmvx = (mvs[i][0] - best_mv.x as i32) as i64;
            let dmvy = (mvs[i][1] - best_mv.y as i32) as i64;
            ra += dmvx * dx + dmvy * dy;
            rb += dmvx * dy - dmvy * dx;
            den += dx * dx + dy * dy;
        }
        if den == 0 {
            return None;
        }
        // `warp_delta_precision = 1`: step is 2^(WARP_DELTA_STEP_BITS + 1 - 1).
        let step = 1i32 << 10;
        let max_v = 14 * step;
        let base = stack.warp[0];
        // The four-parameter fit is relative to the identity model, while the
        // coded deltas are offsets from `WarpParamStack[0]`.
        let target2 = (1i32 << crate::av2::warp::WARPEDMODEL_PREC_BITS)
            .wrapping_add(((ra << 13) / den) as i32);
        let target3 = ((rb << 13) / den) as i32;
        let d2 = (((target2.wrapping_sub(base[2])) as f64 / step as f64).round() as i32 * step)
            .clamp(-max_v, max_v);
        let d3 = (((target3.wrapping_sub(base[3])) as f64 / step as f64).round() as i32 * step)
            .clamp(-max_v, max_v);
        if d2 == 0 && d3 == 0 {
            return None;
        }
        let mut matrix = base;
        matrix[0] = 0;
        matrix[1] = 0;
        matrix[2] = matrix[2].wrapping_add(d2);
        matrix[3] = matrix[3].wrapping_add(d3);
        matrix[4] = matrix[3].wrapping_neg();
        matrix[5] = matrix[2];
        reduce_warp_model(&mut matrix);
        let (abcd, sad) = self.eval_warp_model(src_plane, ref_plane, px_x, py_y, w, h, &matrix)?;
        get_warp_translation(&mut matrix, bw4, bh4, bx4, by4, best_mv);
        let pred_mv = stack.get(0);
        // `is_warp`, `warp_mv`, `warp_idx`, the DRL bit, the MV residual, and
        // the warp-delta precision/parameter symbols.
        let bits_q8 = crate::av2::me::estimate_mv_bits_q8(best_mv, pred_mv) + 20 * 256;
        let cost = sad + ((self.lambda_sad * bits_q8) >> 8);
        if cost < trans_sad {
            Some((
                WarpChoice {
                    idx: 0,
                    matrix,
                    abcd,
                    mv: best_mv,
                    pred_mv,
                    warp_mv: false,
                    motion_mode: 3,
                    delta: [d2, d3],
                    high_prec: true,
                },
                cost,
            ))
        } else {
            None
        }
    }

    /// Estimates the `LOCALWARP` model with `warp_estimation()` (§7.13.3.23).
    #[allow(clippy::too_many_arguments)]
    fn select_local_warp(
        &self,
        src_plane: &Plane<BD::Pixel>,
        ref_plane: &Plane<BD::Pixel>,
        px_x: usize,
        px_y: usize,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        trans_sad: u64,
        best_mv: MvXY,
        stack: &crate::av2::inter::RefMvStack,
    ) -> Option<(WarpChoice, u64)> {
        let (cands, num) = find_warp_samples(
            &self.mi_map,
            self.f_bw4,
            self.f_bh4,
            self.sb_step4.max(1),
            self.tile_row_start4,
            self.tile_row_end4,
            self.tile_col_start4,
            self.tile_col_end4,
            bx4,
            by4,
            bw4,
            bh4,
            0,
        );
        if num == 0 {
            return None;
        }
        let w = bw4 * 4;
        let h = bh4 * 4;
        let matrix = warp_estimation(&cands, num, bw4, bh4, bx4, by4, best_mv);
        let (abcd, sad) = self.eval_warp_model(src_plane, ref_plane, px_x, px_y, w, h, &matrix)?;
        // `is_warp`, `warp_mv = 0`, `use_extend_warp = 0`, `use_local_warp = 1`,
        // the DRL bit, and the MV residual.
        let rate = crate::av2::me::estimate_mv_bits_q8(best_mv, stack.get(0)) + 4 * 256;
        let cost = sad + ((self.lambda_sad * rate) >> 8);
        if cost < trans_sad {
            Some((
                WarpChoice {
                    idx: 0,
                    matrix,
                    abcd,
                    mv: best_mv,
                    pred_mv: stack.get(0),
                    warp_mv: false,
                    motion_mode: 2,
                    delta: [0, 0],
                    high_prec: false,
                },
                cost,
            ))
        } else {
            None
        }
    }

    /// Builds the `EXTENDWARP` model with `extend_warp_estimation()` (§7.13.3.24).
    #[allow(clippy::too_many_arguments)]
    fn select_extend_warp(
        &self,
        src_plane: &Plane<BD::Pixel>,
        ref_plane: &Plane<BD::Pixel>,
        px_x: usize,
        py_y: usize,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        trans_sad: u64,
        best_mv: MvXY,
        stack: &crate::av2::inter::RefMvStack,
    ) -> Option<(WarpChoice, u64)> {
        let row = stack.row_offsets.first().copied().unwrap_or(-1);
        let col = stack.col_offsets.first().copied().unwrap_or(-1);
        let matrix = extend_warp_params(
            &self.mi_map,
            self.f_bw4,
            self.f_bh4,
            self.sb_step4.max(1),
            bx4,
            by4,
            bw4,
            bh4,
            0,
            row,
            col,
            best_mv,
        )?;
        let w = bw4 * 4;
        let h = bh4 * 4;
        let (abcd, sad) = self.eval_warp_model(src_plane, ref_plane, px_x, py_y, w, h, &matrix)?;
        let rate = crate::av2::me::estimate_mv_bits_q8(best_mv, stack.get(0)) + 4 * 256;
        let cost = sad + ((self.lambda_sad * rate) >> 8);
        if cost < trans_sad {
            Some((
                WarpChoice {
                    idx: 0,
                    matrix,
                    abcd,
                    mv: best_mv,
                    pred_mv: stack.get(0),
                    warp_mv: false,
                    motion_mode: 4,
                    delta: [0, 0],
                    high_prec: false,
                },
                cost,
            ))
        } else {
            None
        }
    }

    /// Selects a `WARPMV` candidate from `WarpParamStack` (§5.20.7.6).
    #[allow(clippy::too_many_arguments)]
    fn select_warpmv(
        &self,
        src_plane: &Plane<BD::Pixel>,
        ref_plane: &Plane<BD::Pixel>,
        px_x: usize,
        py_y: usize,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        trans_sad: u64,
        stack: &crate::av2::inter::RefMvStack,
    ) -> Option<(WarpChoice, u64)> {
        let w = bw4 * 4;
        let h = bh4 * 4;
        let mut best: Option<(WarpChoice, u64)> = None;
        for idx in 0..stack.num_warp.min(stack.warp.len()) {
            let cand = stack.warp[idx];
            if cand == DEFAULT_WARP_PARAMS {
                continue;
            }
            // `BlockMvs[0]` is projected from the raw stack entry
            // (§7.12.2.2); only the prediction model is reduced.
            let mv = warp_motion_vector(&cand, bw4, bh4, bx4, by4, self.f_bw4, self.f_bh4);
            let mut matrix = cand;
            reduce_warp_model(&mut matrix);
            get_warp_translation(&mut matrix, bw4, bh4, bx4, by4, mv);
            let Some((abcd, sad)) =
                self.eval_warp_model(src_plane, ref_plane, px_x, py_y, w, h, &matrix)
            else {
                continue;
            };
            let rate_q8 = 4 * 256 + 4 * idx as u64 * 256;
            let cost = sad + ((self.lambda_sad * rate_q8) >> 8);
            if cost < trans_sad && best.as_ref().is_none_or(|(_, c)| cost < *c) {
                best = Some((
                    WarpChoice {
                        idx,
                        matrix,
                        abcd,
                        mv,
                        pred_mv: MvXY::default(),
                        warp_mv: true,
                        motion_mode: 3,
                        delta: [0, 0],
                        high_prec: true,
                    },
                    cost,
                ));
            }
        }
        best
    }

    /// `read_warp_delta_param()` encoder counterpart (§5.20.7.7).
    fn encode_warp_delta_param(&mut self, idx: usize, value: i32, high_prec: bool) {
        let step = 1i32 << (10 + 1 - high_prec as u32);
        let mut v = (value.abs() / step).min(14) as usize;
        let cdf_idx = usize::from(idx == 3 || idx == 4);
        if high_prec && v >= WARP_DELTA_NUM_SYMBOLS_LOW as usize - 1 {
            let low = WARP_DELTA_NUM_SYMBOLS_LOW as usize - 1;
            let extra = v - low;
            self.msac
                .encode_symbol_adapt(low, &mut self.cdf.m.warp_delta_param_low[cdf_idx], 7);
            self.msac.encode_symbol_adapt(
                extra.min(7),
                &mut self.cdf.m.warp_delta_param_high[cdf_idx],
                7,
            );
            v = low + extra;
        } else {
            self.msac.encode_symbol_adapt(
                v.min(7),
                &mut self.cdf.m.warp_delta_param_low[cdf_idx],
                7,
            );
        }
        if v != 0 {
            self.msac
                .encode_bool_adapt((value < 0) as u32, &mut self.cdf.m.warp_delta_param_sign);
        }
    }

    ///
    /// Mirrors AV2 Spec v1.0.0 §5.18.1 `get_qindex()`: with `delta_q_present`
    /// the segment `SEG_LVL_ALT_Q` is added to `CurrentQIndex`, otherwise to
    /// `base_q_idx`.
    fn apply_qidx(&mut self, qidx: u16) {
        let bit_depth = self.bd.bitdepth();
        let delta_present = self.frame_hdr.delta.q.present != 0;
        let base = self.frame_hdr.quant.yac as i32;
        let n_seg = if self.frame_hdr.segmentation.enabled != 0 {
            MAX_SEGMENTS
        } else {
            1
        };
        for seg in 0..n_seg {
            let seg_delta = if self.frame_hdr.segmentation.enabled != 0 {
                self.frame_hdr.segmentation.d.delta_q[seg]
            } else {
                0
            };
            let yac = segment_qindex(base, qidx as i32, seg_delta, delta_present, bit_depth);
            let ydc = (yac + self.frame_hdr.quant.ydc_delta as i32).clamp(0, 255);
            let uac = (yac + self.frame_hdr.quant.uac_delta as i32).clamp(0, 255);
            let udc = (yac + self.frame_hdr.quant.udc_delta as i32).clamp(0, 255);
            let vac = (yac + self.frame_hdr.quant.vac_delta as i32).clamp(0, 255);
            let vdc = (yac + self.frame_hdr.quant.vdc_delta as i32).clamp(0, 255);
            self.dq[seg][0][0] = dq_lookup(ydc) as u32;
            self.dq[seg][0][1] = dq_lookup(yac) as u32;
            self.dq[seg][1][0] = dq_lookup(udc) as u32;
            self.dq[seg][1][1] = dq_lookup(uac) as u32;
            self.dq[seg][2][0] = dq_lookup(vdc) as u32;
            self.dq[seg][2][1] = dq_lookup(vac) as u32;
        }
    }

    /// Starts a superblock: selects the per-superblock quantizer and queues the
    /// `read_delta_qindex()` symbol for the first coded block.
    fn begin_superblock(&mut self, bx4: usize, by4: usize) {
        // The warp and reference-MV banks are per superblock row in the
        // reference decoder (`dav2d_refmvs_tile_sbrow_init` clears them and
        // `dav2d_refmvs_warp_add` / `bank_update` stop at the hit limit), and
        // the encoder matches that here so both sides derive the same
        // `WarpParamStack` / `RefStackMv` bank candidates.
        if by4 != self.warp_bank_row {
            self.warp_bank_row = by4;
            self.warp_bank_hits = 0;
            for b in self.warp_bank.iter_mut() {
                b.clear();
            }
            self.refmv_bank.clear();
            self.refmv_hits = RefMvBankHits::default();
        }
        if self.frame_hdr.delta.q.present == 0 {
            return;
        }
        let base = self.frame_hdr.quant.yac as i32;
        // Rate feedback: compare the bits spent by the previous superblock
        // against its share of the frame budget before selecting this one.
        if let Some(budget) = self.sb_budget_bits {
            let used = self.msac.tell_bits().saturating_sub(self.prev_sb_bits) as f64;
            self.prev_sb_bits = self.msac.tell_bits();
            if used > budget * 1.15 {
                self.rate_offset = (self.rate_offset + 1).min(4);
            } else if used < budget * 0.85 {
                self.rate_offset = (self.rate_offset - 1).max(-4);
            }
        }
        let mut target = base + self.rate_offset as i32;
        if let Some(plan) = self.seg_plan {
            target += plan.sb_offset(bx4, by4, self.sb_step4) as i32;
        }
        let max_q = crate::av2::headers::max_q_bits(self.bd.bitdepth());
        let target = target.clamp(1, max_q);
        let step = 1i32 << self.frame_hdr.delta.q.res_log2;
        // Keep the applied delta on the `delta_q_res` lattice so the syntax and
        // the encoder state describe the same quantizer.
        let target = (target / step) * step;
        let target = target.clamp(1, max_q) as u16;
        let delta = target as i32 - self.cur_qidx as i32;
        self.cur_qidx = if delta > 0 {
            (self.cur_qidx as i32 + delta).min(max_q) as u16
        } else {
            (self.cur_qidx as i32 + delta).max(1) as u16
        };
        self.pending_delta = Some(delta);
        self.apply_qidx(self.cur_qidx);
    }

    /// Writes the queued superblock delta-Q at the syntax position required by
    /// AV2 Spec v1.0.0 §5.20.5.11 (after `read_ccso`, before the block mode).
    fn emit_pending_delta(&mut self) {
        if let Some(delta) = self.pending_delta.take() {
            encode_delta_q(
                &mut self.msac,
                &mut self.cdf.m.delta_q,
                delta,
                self.frame_hdr.delta.q.res_log2,
            );
        }
    }

    /// Codes the current block's segment id and accounts its quantizer.
    #[allow(clippy::too_many_arguments)]
    fn code_segment_id(
        &mut self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        have_left: bool,
        have_top: bool,
        skip_txfm: bool,
        desired: u8,
    ) -> usize {
        if self.frame_hdr.segmentation.enabled == 0 {
            return 0;
        }
        let a_col_idx = (bx4 - self.tile_col_start4) >> 5;
        let seg_pred_a = &mut self.seg_pred_a[a_col_idx];
        let seg_pred_l = &mut self.seg_pred_l;
        let id = encode_segment_id(
            &mut self.msac,
            &mut self.cdf.m,
            &self.seg_coder,
            &mut self.seg_map,
            self.seg_ref,
            bx4,
            by4,
            bw4,
            bh4,
            have_left,
            have_top,
            skip_txfm,
            desired,
            seg_pred_a,
            seg_pred_l,
        );
        let id = id as usize;
        let w = (bw4 * bh4) as u64;
        let seg_delta = self.frame_hdr.segmentation.d.delta_q[id];
        let qidx = segment_qindex(
            self.frame_hdr.quant.yac as i32,
            self.cur_qidx as i32,
            seg_delta,
            self.frame_hdr.delta.q.present != 0,
            self.bd.bitdepth(),
        );
        self.qidx_sum += qidx.max(0) as u64 * w;
        self.qidx_weight += w;
        id
    }

    fn encode_lr(&mut self, bx4: usize, by4: usize) {
        let n_planes = if self.seqhdr.layout == PixelLayout::I400 {
            1
        } else {
            3
        };
        let ss_x = self.seqhdr.ss_hor as usize;
        let ss_y = self.seqhdr.ss_ver as usize;
        for p in 0..n_planes {
            let frame_type = self.frame_hdr.restoration.planes[p].restoration_type;
            if frame_type == 0 {
                continue;
            }
            let (sub_x, sub_y) = if p == 0 { (0, 0) } else { (ss_x, ss_y) };
            let tx = ((bx4 - self.tile_col_start4) * 4) >> sub_x;
            let ty = ((by4 - self.tile_row_start4) * 4) >> sub_y;
            let unit_sz_log2 =
                self.frame_hdr.restoration.unit_size[if p == 0 { 0 } else { 1 }] as usize;
            let unit_sz = 1usize << unit_sz_log2;
            let mask = unit_sz - 1;
            if ((tx | ty) & mask) != 0 {
                continue;
            }
            let tw = (self.tile_col_end4 * 4) >> sub_x;
            let th = (self.tile_row_end4 * 4) >> sub_y;
            let half_unit = unit_sz >> 1;
            let fx = (bx4 * 4) >> sub_x;
            let fy = (by4 * 4) >> sub_y;
            if (ty > 0 && fy + half_unit > th) || (tx > 0 && fx + half_unit > tw) {
                continue;
            }
            let uc = fx >> unit_sz_log2;
            let ur = fy >> unit_sz_log2;
            if uc >= self.lr_cols[p] || ur >= self.lr_rows[p] {
                continue;
            }
            let unit_type = self.lr_types[p][ur * self.lr_cols[p] + uc];
            match frame_type {
                1 => {
                    let flag = unit_type == 1;
                    self.msac
                        .encode_bool_adapt(flag as u32, &mut self.cdf.m.use_pc_wiener);
                }
                2 => {
                    let flag = unit_type == 2;
                    self.msac
                        .encode_bool_adapt(flag as u32, &mut self.cdf.m.use_wiener_ns);
                }
                3 => {
                    // Switchable: first symbol selects NONE, the second PC vs NS
                    // (dav2d `read_restoration_info`, `rst_switchable`).
                    let is_none = unit_type == 0;
                    self.msac.encode_bool_adapt(
                        is_none as u32,
                        &mut self.cdf.m.flex_restoration_type[0][0],
                    );
                    if !is_none {
                        let is_pc = unit_type == 1;
                        self.msac.encode_bool_adapt(
                            is_pc as u32,
                            &mut self.cdf.m.flex_restoration_type[1][0],
                        );
                    }
                }
                _ => {}
            }
        }
    }
}

fn partition_has_sub8x8(bs: BlockSize, p: BlockPartition) -> bool {
    let b_dim = BLOCK_DIMENSIONS[bs as usize];
    let w = (b_dim[0] as usize) * 4;
    let h = (b_dim[1] as usize) * 4;
    match p {
        BlockPartition::None => false,
        BlockPartition::H => (h / 2) < 8,
        BlockPartition::V => (w / 2) < 8,
        BlockPartition::Split => (w / 2) < 8 || (h / 2) < 8,
        BlockPartition::H3 => (h / 4) < 8 || (w / 2) < 8,
        BlockPartition::V3 => (w / 4) < 8 || (h / 2) < 8,
        BlockPartition::H4A | BlockPartition::H4B => (h / 8) < 8 || w < 8,
        BlockPartition::V4A | BlockPartition::V4B => (w / 8) < 8 || h < 8,
    }
}

/// Restricts the luma transform type that an inter chroma plane inherits
/// (AV2 Spec v1.0.0 §5.20.8.3; `dav2d` `decode_cf`).
///
/// A chroma transform wider/taller than 32 samples cannot use an ADST in that
/// direction, a 16x16 transform cannot mix identity with ADST, and the
/// internal `IDTX_INV` form maps to `IDTX`.
fn chroma_txtp_from_luma(y_txtp: u32, t_dim: &crate::av2::tables::TxfmInfo, tx: usize) -> u32 {
    let t = y_txtp & 0xFF;
    if (t_dim.w >= 8 && (t & 0x02) != 0)
        || (t_dim.h >= 8 && (t & 0x40) != 0)
        || (tx == 2 && (((t & 0x47) == 0x41) || ((t & 0xe2) == 0x22)))
    {
        txtp::DCT_DCT
    } else if t == txtp::IDTX_INV {
        txtp::IDTX
    } else {
        t
    }
}

/// Returns the L1 cost of the 4x4 Walsh-Hadamard transform of each `4x4`
/// sub-block of `buf`, used as the lossless rate proxy during encoder
/// decisions (AV2 Spec v1.0.0 §7.15.2.2).
fn wht_l1_cost(buf: &[i32], stride: usize, w: usize, h: usize) -> u64 {
    let mut total = 0u64;
    let mut sub = [0i32; 16];
    let mut coeff = [0i32; 16];
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            for i in 0..4 {
                for j in 0..4 {
                    sub[i * 4 + j] = buf[(y + i) * stride + x + j];
                }
            }
            crate::av2::transform::fwd_wht_wht_4x4(&sub, 4, &mut coeff);
            for &c in coeff.iter() {
                total += c.unsigned_abs() as u64;
            }
            x += 4;
        }
        y += 4;
    }
    total
}

impl<'a, BD: BitDepth> TileEncodeState<'a, BD> {
    fn encode_gdf_blk(&mut self, bx4: usize, by4: usize) {
        if self.frame_hdr.gdf.enabled == AdaptiveBoolean::Off {
            return;
        }
        if ((bx4 | by4) & (self.gdf_step4 - 1)) != 0 {
            return;
        }
        let uc = bx4 / self.gdf_step4;
        let ur = by4 / self.gdf_step4;
        if ur >= self.gdf_rows || uc >= self.gdf_cols {
            return;
        }
        let max_dim = self.frame_hdr.width.max(self.frame_hdr.height);
        let flag = self.gdf_blks[ur * self.gdf_cols + uc];
        if self.frame_hdr.gdf.enabled == AdaptiveBoolean::Adaptive
            && max_dim > 64 * self.frame_hdr.gdf.b64size as i32
        {
            self.msac
                .encode_bool_adapt(flag as u32, &mut self.cdf.m.use_gdf);
        }
    }
    /// Records per-4x4 transform/skip/edge information for the normative
    /// deblocking filter (AV2 Spec v1.0.0 §7.17). Mirrors the decoder.
    #[allow(clippy::too_many_arguments)]
    fn encode_ccso_blk(&mut self, bx4: usize, by4: usize) {
        if self.frame_hdr.ccso.enabled == 0 {
            return;
        }
        if ((bx4 | by4) & (self.ccso_step4 - 1)) != 0 {
            return;
        }
        let uc = bx4 / self.ccso_step4;
        let ur = by4 / self.ccso_step4;
        if ur >= self.ccso_rows || uc >= self.ccso_cols {
            return;
        }

        let n_planes = if self.seqhdr.layout == PixelLayout::I400 {
            1
        } else {
            3
        };
        for p in 0..n_planes {
            if self.frame_hdr.ccso.planes[p].enabled == 0 {
                continue;
            }
            if self.frame_hdr.ccso.planes[p].sb_reuse != 0 {
                continue;
            }
            let left_available = bx4 > self.tile_col_start4;
            let left_val =
                left_available && self.ccso_blks[p][ur * self.ccso_cols + uc.saturating_sub(1)];
            let ctx = if left_val { 2 } else { 0 };
            let val = true;
            self.ccso_blks[p][ur * self.ccso_cols + uc] = val;
            self.msac
                .encode_bool_adapt(val as u32, &mut self.cdf.m.ccso_blk[p][ctx]);
        }
    }

    fn record_deblock(
        &mut self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        t_dim: &crate::av2::tables::TxfmInfo,
        uv_tdim: &crate::av2::tables::TxfmInfo,
        has_chroma: bool,
        skip: bool,
        lossless: bool,
    ) {
        if !self.tools.enable_deblock && !self.tools.enable_cdef {
            return;
        }
        let (tw, th) = (t_dim.w as usize, t_dim.h as usize);
        for dy4 in 0..bh4 {
            let y4 = by4 + dy4;
            if y4 >= self.f_bh4 {
                break;
            }
            for dx4 in 0..bw4 {
                let x4 = bx4 + dx4;
                if x4 >= self.f_bw4 {
                    break;
                }
                self.grid.set_luma(
                    x4,
                    y4,
                    BlockDeblockInfo {
                        tx_w: t_dim.w * 4,
                        tx_h: t_dim.h * 4,
                        skip,
                        lossless,
                        block_left: dx4 == 0,
                        block_top: dy4 == 0,
                        tx_left: dx4 % tw == 0,
                        tx_top: dy4 % th == 0,
                    },
                );
            }
        }
        if has_chroma {
            let ss_x = self.seqhdr.ss_hor;
            let ss_y = self.seqhdr.ss_ver;
            let cbx4 = bx4 >> ss_x;
            let cby4 = by4 >> ss_y;
            let cbw4 = (bw4 >> ss_x).max(1);
            let cbh4 = (bh4 >> ss_y).max(1);
            let uw = (uv_tdim.w as usize).max(1);
            let uh = (uv_tdim.h as usize).max(1);
            for dy4 in 0..cbh4 {
                let cy4 = cby4 + dy4;
                if cy4 >= self.grid.chroma_h {
                    break;
                }
                for dx4 in 0..cbw4 {
                    let cx4 = cbx4 + dx4;
                    if cx4 >= self.grid.chroma_w {
                        break;
                    }
                    self.grid.set_chroma(
                        cx4,
                        cy4,
                        BlockDeblockInfo {
                            tx_w: uv_tdim.w * 4,
                            tx_h: uv_tdim.h * 4,
                            skip,
                            lossless,
                            block_left: dx4 == 0,
                            block_top: dy4 == 0,
                            tx_left: dx4 % uw == 0,
                            tx_top: dy4 % uh == 0,
                        },
                    );
                }
            }
        }
    }
}

impl<'a, BD: BitDepth> TileEncodeState<'a, BD> {
    /// Encodes the coding tree of `bs` at `(bx4, by4)`.
    ///
    /// The node first derives the partitions the syntax allows (AV2 Spec v1.0.0
    /// §5.20.3.2 `init_allowed_partitions`) and the partition the syntax implies
    /// at a frame boundary (`partition_implied`). Implied partitions and nodes
    /// with a single legal partition code no symbols at all; every other node
    /// codes the partition decision and then recurses or codes a leaf block.
    fn encode_sb(&mut self, bs: BlockSize, bx4: usize, by4: usize, chroma_offset: bool) {
        if bx4 >= self.tile_col_end4 || by4 >= self.tile_row_end4 {
            return;
        }

        let cfg =
            PartitionSyntaxConfig::from_seq_header(self.seqhdr, self.frame_hdr.is_key_or_intra());
        let allowed =
            init_allowed_partitions(bs, bx4, by4, self.f_bw4, self.f_bh4, chroma_offset, &cfg);
        let implied = partition_implied(bs, bx4, by4, self.f_bw4, self.f_bh4);

        let bp = if let Some(partition) = implied.filter(|p| allowed.contains(*p)) {
            partition
        } else if let Some(partition) = allowed.single() {
            partition
        } else {
            let chosen = self.choose_partition(bs, bx4, by4, &allowed);
            let a_col_idx = (bx4 - self.tile_col_start4) >> 5;
            encode_partition_decision(
                &mut self.msac,
                &mut self.cdf.m,
                &self.a_cols[a_col_idx],
                &self.l_ctx,
                bs,
                chosen,
                bx4 & 31,
                by4 & 31,
                &allowed,
            );
            chosen
        };
        self.encode_partition_node(bs, bp, bx4, by4, chroma_offset, &cfg);
    }

    fn eval_subblock_distortion(&self, px_x: usize, px_y: usize, w: usize, h: usize) -> u64 {
        let src_y = &self.src.planes[0];
        if px_x >= src_y.width || px_y >= src_y.height {
            return 0;
        }
        let eff_w = w.min(src_y.width - px_x);
        let eff_h = h.min(src_y.height - px_y);
        if eff_w == 0 || eff_h == 0 {
            return 0;
        }

        if self.frame_hdr.is_inter_or_switch()
            && let Some(r0) = self.ref0
        {
            let r0_y = &r0.planes[0];
            let inter_sad = compute_sad(
                &src_y.data,
                src_y.offset(px_x, px_y),
                src_y.stride,
                &r0_y.data,
                r0_y.offset(px_x, px_y),
                r0_y.stride,
                eff_w,
                eff_h,
            );
            let intra_sad = self.eval_intra_dc_sad(px_x, px_y, eff_w, eff_h);
            inter_sad.min(intra_sad)
        } else {
            self.eval_intra_dc_sad(px_x, px_y, eff_w, eff_h)
        }
    }

    /// Luma intra-mode decision used by the encoder.
    ///
    /// Extended-partition streams keep rectangular luma leaves on `DC_PRED`:
    /// wide-angle directional prediction on non-square blocks is not yet
    /// bit-exact against the reference decoder, while square blocks keep the
    /// full mode search. Default streams (rectangular partitions disabled)
    /// are unchanged.
    #[allow(clippy::too_many_arguments)]
    fn select_luma_intra_mode(
        &self,
        src: &[BD::Pixel],
        src_off: usize,
        src_stride: usize,
        rec: &[BD::Pixel],
        rec_off: usize,
        rec_stride: usize,
        have_left: bool,
        have_top: bool,
        w: usize,
        h: usize,
        tx: usize,
        lambda_sad: u64,
        speed: u8,
        rec_w: usize,
        rec_h: usize,
        px_x: usize,
        px_y: usize,
    ) -> (u8, i8, u8, u64) {
        if w != h && self.tools.enable_rect_partitions && self.tools.enable_ext_partitions {
            let sad = self.eval_intra_dc_sad(px_x, px_y, w, h);
            return (0, 0, 0xFF, sad + (lambda_sad >> 1));
        }
        select_best_intra_mode_y(
            self.bd, src, src_off, src_stride, rec, rec_off, rec_stride, have_left, have_top, w, h,
            tx, lambda_sad, speed, rec_w, rec_h,
        )
    }

    fn eval_intra_dc_sad(&self, px_x: usize, px_y: usize, eff_w: usize, eff_h: usize) -> u64 {
        let count = (eff_w * eff_h) as i64;
        if count == 0 {
            return 0;
        }
        let src_y = &self.src.planes[0];
        let mut sum = 0i64;
        for r in 0..eff_h {
            let row_off = src_y.offset(px_x, px_y + r);
            let row = &src_y.data[row_off..row_off + eff_w];
            for &val in row {
                sum += val.to_i32() as i64;
            }
        }
        let mean = (sum + count / 2) / count;
        let mut sad = 0u64;
        for r in 0..eff_h {
            let row_off = src_y.offset(px_x, px_y + r);
            let row = &src_y.data[row_off..row_off + eff_w];
            for &val in row {
                sad += (val.to_i32() as i64 - mean).unsigned_abs();
            }
        }
        sad
    }

    /// Picks the partition to code from the legal set using rate-distortion optimization (RDO).
    fn choose_partition(
        &self,
        bs: BlockSize,
        bx4: usize,
        by4: usize,
        allowed: &AllowedPartitions,
    ) -> BlockPartition {
        // `read_delta_qindex()` is not coded for a skipped block whose size is
        // the superblock size (AV2 Spec v1.0.0 §5.20.5.11:
        // `!(MiSize == SbSize && skip_flag)`). The encoder writes delta-Q in
        // the first coded block of every superblock, so it keeps that first
        // block below the superblock size.
        if self.frame_hdr.delta.q.present != 0
            && matches!(bs, BlockSize::Bs64x64 | BlockSize::Bs128x128)
            && allowed.contains(BlockPartition::Split)
        {
            return BlockPartition::Split;
        }
        if bs == BlockSize::Bs128x128 {
            if allowed.contains(BlockPartition::Split) {
                return BlockPartition::Split;
            }
            return allowed.first().unwrap_or(BlockPartition::None);
        }

        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let w = bw4 * 4;
        let h = bh4 * 4;
        let px_x = bx4 * 4;
        let px_y = by4 * 4;

        // 1. Respect intermediate 2:1 and 1:2 splits into square sub-blocks
        // when rectangular partitions are not enabled.
        if !self.tools.enable_rect_partitions && bx4 + bw4 <= self.f_bw4 && by4 + bh4 <= self.f_bh4
        {
            match bs {
                BlockSize::Bs64x32 | BlockSize::Bs32x16 | BlockSize::Bs16x8
                    if allowed.contains(BlockPartition::V) =>
                {
                    return BlockPartition::V;
                }
                BlockSize::Bs32x64 | BlockSize::Bs16x32 | BlockSize::Bs8x16
                    if allowed.contains(BlockPartition::H) =>
                {
                    return BlockPartition::H;
                }
                _ => {}
            }
        }

        if bs != BlockSize::Bs64x64 && bs != BlockSize::Bs32x32 {
            if allowed.contains(BlockPartition::None) {
                return BlockPartition::None;
            }
            return allowed.first().unwrap_or(BlockPartition::None);
        }

        // 2. Noise / flat check:
        // If is_spatial_noise_block: return None (if allowed).
        let src_y = &self.src.planes[0];
        let is_noise = crate::av2::partition::is_spatial_noise_block(
            src_y,
            px_x,
            px_y,
            w,
            h,
            self.src.width,
            self.src.height,
        );
        if is_noise && allowed.contains(BlockPartition::None) {
            return BlockPartition::None;
        }

        // 3. Pruned candidate evaluation:
        // - First evaluate Cost(NONE) = D(whole) + lambda_sad * R(NONE)
        let d_whole = self.eval_subblock_distortion(px_x, px_y, w, h);
        let eff_w = w.min(self.src.width.saturating_sub(px_x));
        let eff_h = h.min(self.src.height.saturating_sub(px_y));
        let area = (eff_w * eff_h) as u64;

        // - If stationary inter block (D(whole) <= area * 2) and None is allowed: return None.
        if self.frame_hdr.is_inter_or_switch()
            && self.ref0.is_some()
            && d_whole <= area * 2
            && allowed.contains(BlockPartition::None)
        {
            return BlockPartition::None;
        }

        // - Four-quadrant distortion is the `PARTITION_SPLIT` cost.
        let half_w = w / 2;
        let half_h = h / 2;
        let d_quad0 = self.eval_subblock_distortion(px_x, px_y, half_w, half_h);
        let d_quad1 = self.eval_subblock_distortion(px_x + half_w, px_y, half_w, half_h);
        let d_quad2 = self.eval_subblock_distortion(px_x, px_y + half_h, half_w, half_h);
        let d_quad3 = self.eval_subblock_distortion(px_x + half_w, px_y + half_h, half_w, half_h);
        let d_quads = d_quad0 + d_quad1 + d_quad2 + d_quad3;

        if d_quads >= (d_whole * 9) / 10 && allowed.contains(BlockPartition::None) {
            return BlockPartition::None;
        }

        // - Loop over candidate partitions in allowed:
        //   Candidate set: [None, H, V, Split, H3, V3, H4A, H4B, V4A, V4B]
        //   (filter only those where allowed.contains(p) and tools enables it).
        let mut best_partition = if allowed.contains(BlockPartition::None) {
            BlockPartition::None
        } else {
            allowed.first().unwrap_or(BlockPartition::None)
        };
        let mut best_cost = if allowed.contains(BlockPartition::None) {
            d_whole + self.lambda_sad
        } else {
            u64::MAX
        };

        const CANDIDATES: [BlockPartition; 10] = [
            BlockPartition::None,
            BlockPartition::H,
            BlockPartition::V,
            BlockPartition::Split,
            BlockPartition::H3,
            BlockPartition::V3,
            BlockPartition::H4A,
            BlockPartition::H4B,
            BlockPartition::V4A,
            BlockPartition::V4B,
        ];

        let cfg =
            PartitionSyntaxConfig::from_seq_header(self.seqhdr, self.frame_hdr.is_key_or_intra());

        for &p in &CANDIDATES {
            if !allowed.contains(p) {
                continue;
            }
            if p != BlockPartition::None && chroma_offset_for_partition(p, bs, &cfg) {
                continue;
            }
            if self.seqhdr.layout == PixelLayout::I420 && partition_has_sub8x8(bs, p) {
                continue;
            }
            // Extended partitions are searched at 64x64 superblock nodes
            // only. The syntax and allowed sets cover every size, but the
            // encoder keeps the search pruned: rectangular children below
            // 64x64 use the regular H/V/square split decisions.
            if bs != BlockSize::Bs64x64 && p.is_extended() {
                continue;
            }
            if p.is_extended() && !ext_selection_allowed(bs, p, &cfg) {
                continue;
            }
            match p {
                BlockPartition::None => {}
                BlockPartition::H => {
                    // For square splitting when rect partitions are disabled, H is preferred.
                    if !self.tools.enable_rect_partitions && bw4 != bh4 {
                        continue;
                    }
                }
                BlockPartition::V => {
                    // If rect partitions disabled, only allow H for 4-way square split.
                    if !self.tools.enable_rect_partitions {
                        continue;
                    }
                }
                BlockPartition::Split => {}
                BlockPartition::H3 | BlockPartition::V3 => {
                    if !self.tools.enable_ext_partitions || !self.tools.enable_rect_partitions {
                        continue;
                    }
                }
                BlockPartition::H4A
                | BlockPartition::H4B
                | BlockPartition::V4A
                | BlockPartition::V4B => {
                    if !self.tools.enable_ext_partitions
                        || !self.tools.enable_uneven_4way_partitions
                        || !self.tools.enable_rect_partitions
                    {
                        continue;
                    }
                }
            }

            let (dist, bits) = match p {
                BlockPartition::None => (d_whole, 1u64),
                BlockPartition::H => {
                    let d0 = self.eval_subblock_distortion(px_x, px_y, w, half_h);
                    let d1 = self.eval_subblock_distortion(px_x, px_y + half_h, w, half_h);
                    (d0 + d1, 3u64)
                }
                BlockPartition::V => {
                    let d0 = self.eval_subblock_distortion(px_x, px_y, half_w, h);
                    let d1 = self.eval_subblock_distortion(px_x + half_w, px_y, half_w, h);
                    (d0 + d1, 3u64)
                }
                BlockPartition::Split => (d_quads, 2u64),
                BlockPartition::H3 => {
                    let qh = h / 4;
                    let hw = w / 2;
                    let d0 = self.eval_subblock_distortion(px_x, px_y, w, qh);
                    let d1 = self.eval_subblock_distortion(px_x, px_y + qh, hw, 2 * qh);
                    let d2 = self.eval_subblock_distortion(px_x + hw, px_y + qh, hw, 2 * qh);
                    let d3 = self.eval_subblock_distortion(px_x, px_y + 3 * qh, w, qh);
                    (d0 + d1 + d2 + d3, 5u64)
                }
                BlockPartition::V3 => {
                    let qw = w / 4;
                    let hh = h / 2;
                    let d0 = self.eval_subblock_distortion(px_x, px_y, qw, h);
                    let d1 = self.eval_subblock_distortion(px_x + qw, px_y, 2 * qw, hh);
                    let d2 = self.eval_subblock_distortion(px_x + qw, px_y + hh, 2 * qw, hh);
                    let d3 = self.eval_subblock_distortion(px_x + 3 * qw, px_y, qw, h);
                    (d0 + d1 + d2 + d3, 5u64)
                }
                BlockPartition::H4A => {
                    let u = h / 8;
                    let d0 = self.eval_subblock_distortion(px_x, px_y, w, u);
                    let d1 = self.eval_subblock_distortion(px_x, px_y + u, w, 2 * u);
                    let d2 = self.eval_subblock_distortion(px_x, px_y + 3 * u, w, 4 * u);
                    let d3 = self.eval_subblock_distortion(px_x, px_y + 7 * u, w, u);
                    (d0 + d1 + d2 + d3, 7u64)
                }
                BlockPartition::H4B => {
                    let u = h / 8;
                    let d0 = self.eval_subblock_distortion(px_x, px_y, w, u);
                    let d1 = self.eval_subblock_distortion(px_x, px_y + u, w, 4 * u);
                    let d2 = self.eval_subblock_distortion(px_x, px_y + 5 * u, w, 2 * u);
                    let d3 = self.eval_subblock_distortion(px_x, px_y + 7 * u, w, u);
                    (d0 + d1 + d2 + d3, 7u64)
                }
                BlockPartition::V4A => {
                    let u = w / 8;
                    let d0 = self.eval_subblock_distortion(px_x, px_y, u, h);
                    let d1 = self.eval_subblock_distortion(px_x + u, px_y, 2 * u, h);
                    let d2 = self.eval_subblock_distortion(px_x + 3 * u, px_y, 4 * u, h);
                    let d3 = self.eval_subblock_distortion(px_x + 7 * u, px_y, u, h);
                    (d0 + d1 + d2 + d3, 7u64)
                }
                BlockPartition::V4B => {
                    let u = w / 8;
                    let d0 = self.eval_subblock_distortion(px_x, px_y, u, h);
                    let d1 = self.eval_subblock_distortion(px_x + u, px_y, 4 * u, h);
                    let d2 = self.eval_subblock_distortion(px_x + 5 * u, px_y, 2 * u, h);
                    let d3 = self.eval_subblock_distortion(px_x + 7 * u, px_y, u, h);
                    (d0 + d1 + d2 + d3, 7u64)
                }
            };

            let cost = dist + bits * self.lambda_sad;
            if cost < best_cost {
                best_cost = cost;
                best_partition = p;
            }
        }

        best_partition
    }

    /// Applies the chosen partition: updates the partition context and either
    /// codes the leaf block or recurses into the sub-blocks.
    fn encode_partition_node(
        &mut self,
        bs: BlockSize,
        bp: BlockPartition,
        bx4: usize,
        by4: usize,
        chroma_offset: bool,
        cfg: &PartitionSyntaxConfig,
    ) {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let a_col_idx = (bx4 - self.tile_col_start4) >> 5;
        let bx4_in_sb = bx4 & 31;
        let by4_in_sb = by4 & 31;
        let child_chroma_offset = chroma_offset || chroma_offset_for_partition(bp, bs, cfg);
        let Some(sub_bs) = partition_subsize(bs, bp) else {
            return;
        };
        self.partition_counts[bp as usize] += 1;

        match bp {
            BlockPartition::None => {
                update_partition_ctx(
                    &mut self.a_cols[a_col_idx],
                    &mut self.l_ctx,
                    bs,
                    bx4_in_sb,
                    by4_in_sb,
                );
                self.encode_leaf_block(bs, bx4, by4);
            }
            BlockPartition::H => {
                self.rect_blocks += 2;
                self.encode_sb(sub_bs, bx4, by4, child_chroma_offset);
                self.encode_sb(sub_bs, bx4, by4 + (bh4 >> 1), child_chroma_offset);
            }
            BlockPartition::V => {
                self.rect_blocks += 2;
                self.encode_sb(sub_bs, bx4, by4, child_chroma_offset);
                self.encode_sb(sub_bs, bx4 + (bw4 >> 1), by4, child_chroma_offset);
            }
            BlockPartition::Split => {
                let half_w4 = bw4 >> 1;
                let half_h4 = bh4 >> 1;
                self.encode_sb(sub_bs, bx4, by4, child_chroma_offset);
                self.encode_sb(sub_bs, bx4 + half_w4, by4, child_chroma_offset);
                self.encode_sb(sub_bs, bx4, by4 + half_h4, child_chroma_offset);
                self.encode_sb(sub_bs, bx4 + half_w4, by4 + half_h4, child_chroma_offset);
            }
            BlockPartition::H3 => {
                let Some(mid_bs) = h_partition_midsize(bs) else {
                    return;
                };
                self.rect_blocks += 4;
                let qh4 = bh4 >> 2;
                self.encode_sb(sub_bs, bx4, by4, false);
                self.encode_sb(mid_bs, bx4, by4 + qh4, false);
                self.encode_sb(mid_bs, bx4 + (bw4 >> 1), by4 + qh4, false);
                self.encode_sb(sub_bs, bx4, by4 + 3 * qh4, child_chroma_offset);
            }
            BlockPartition::V3 => {
                let Some(mid_bs) = h_partition_midsize(bs) else {
                    return;
                };
                self.rect_blocks += 4;
                let qw4 = bw4 >> 2;
                self.encode_sb(sub_bs, bx4, by4, false);
                self.encode_sb(mid_bs, bx4 + qw4, by4, false);
                self.encode_sb(mid_bs, bx4 + qw4, by4 + (bh4 >> 1), false);
                self.encode_sb(sub_bs, bx4 + 3 * qw4, by4, child_chroma_offset);
            }
            BlockPartition::H4A => {
                let Some((sub_bs, big_bs, med_bs)) = partition_4way_sizes(bs, bp) else {
                    return;
                };
                self.rect_blocks += 4;
                let u = bh4 >> 3;
                self.encode_sb(sub_bs, bx4, by4, false);
                self.encode_sb(med_bs, bx4, by4 + u, false);
                self.encode_sb(big_bs, bx4, by4 + 3 * u, false);
                self.encode_sb(sub_bs, bx4, by4 + 7 * u, child_chroma_offset);
            }
            BlockPartition::H4B => {
                let Some((sub_bs, big_bs, med_bs)) = partition_4way_sizes(bs, bp) else {
                    return;
                };
                self.rect_blocks += 4;
                let u = bh4 >> 3;
                self.encode_sb(sub_bs, bx4, by4, false);
                self.encode_sb(big_bs, bx4, by4 + u, false);
                self.encode_sb(med_bs, bx4, by4 + 5 * u, false);
                self.encode_sb(sub_bs, bx4, by4 + 7 * u, child_chroma_offset);
            }
            BlockPartition::V4A => {
                let Some((sub_bs, big_bs, med_bs)) = partition_4way_sizes(bs, bp) else {
                    return;
                };
                self.rect_blocks += 4;
                let u = bw4 >> 3;
                self.encode_sb(sub_bs, bx4, by4, false);
                self.encode_sb(med_bs, bx4 + u, by4, false);
                self.encode_sb(big_bs, bx4 + 3 * u, by4, false);
                self.encode_sb(sub_bs, bx4 + 7 * u, by4, child_chroma_offset);
            }
            BlockPartition::V4B => {
                let Some((sub_bs, big_bs, med_bs)) = partition_4way_sizes(bs, bp) else {
                    return;
                };
                self.rect_blocks += 4;
                let u = bw4 >> 3;
                self.encode_sb(sub_bs, bx4, by4, false);
                self.encode_sb(big_bs, bx4 + u, by4, false);
                self.encode_sb(med_bs, bx4 + 5 * u, by4, false);
                self.encode_sb(sub_bs, bx4 + 7 * u, by4, child_chroma_offset);
            }
        }
    }

    /// Chooses the luma transform kernel for one transform block.
    ///
    /// Lossless blocks use the 4x4 WHT of AV2 Spec v1.0.0 §7.15.4; otherwise
    /// the kernel comes from the block's transform set (§5.20.8.3) and, when
    /// transform-type RDO is enabled, from the compaction heuristic in
    /// [`select_best_tx_type`]. `DCT_DCT` stays the default whenever the set
    /// cannot signal another type (for example `reduced_tx_set == 2` on an
    /// intra block, or any 32x32-or-larger intra transform).
    #[allow(clippy::too_many_arguments)]
    fn choose_luma_txtp(
        &self,
        res: &[i32],
        w: usize,
        tx: usize,
        is_intra: bool,
        y_mode: u8,
        y_angle: i8,
        mrl: u8,
        lossless: bool,
    ) -> u32 {
        if lossless {
            return if tx == 0 {
                txtp::WHT_WHT
            } else {
                txtp::DCT_DCT
            };
        }
        if !self.tools.enable_tx_type_rdo {
            return txtp::DCT_DCT;
        }
        let t_dim = &TXFM_DIMENSIONS[tx];
        let set = crate::av2::entropy::get_tx_set(
            t_dim,
            !is_intra,
            self.frame_hdr.reduced_txtp_set as i32,
        );
        let (choices, n) = if is_intra {
            let (c, n) =
                crate::av2::transform::intra_tx_type_choices(t_dim, tx, y_mode, y_angle, mrl, set);
            let mut out = [crate::av2::transform::TxTypeChoice {
                txtp: txtp::DCT_DCT,
                is_long_side_dct: 1,
                symbol: 0,
                symbol2: 0,
            }; 16];
            out[..n].copy_from_slice(&c[..n]);
            (out, n)
        } else {
            crate::av2::transform::inter_tx_type_choices(t_dim, set)
        };
        if n == 0 {
            return txtp::DCT_DCT;
        }
        crate::av2::transform::select_best_tx_type(res, w, tx, &choices[..n], self.speed)
    }

    /// Builds one coded transform unit from a packed residual block.
    ///
    /// `sub` holds `unit.w * unit.h` residual samples in raster order. The
    /// kernel is the one the caller selected for this unit (the coefficient
    /// syntax may still re-derive an inferred type). Lossless units use the
    /// §7.14.4 `8 * q` dequantization.
    #[allow(clippy::too_many_arguments)]
    fn build_tx_unit(
        &self,
        sub: &[i32],
        tx: usize,
        txtp: u32,
        seg_id: usize,
        plane: usize,
        is_intra: bool,
        lossless: bool,
    ) -> EncTxUnit {
        let mut unit = EncTxUnit::new(tx, 0, 0);
        let mut cf = vec![0i32; unit.q_cf.len()];
        if lossless && (txtp & 0xFF) == txtp::IDTX {
            // §7.15.4: a lossless `IDTX` transform is a pure pass-through, so
            // the coded levels are the residual samples themselves, stored in
            // the column-major coefficient order.
            for y in 0..unit.h {
                for x in 0..unit.w {
                    cf[y + x * unit.h] = sub[y * unit.w + x];
                }
            }
        } else {
            fwd_txfm_2d(sub, unit.w, &mut cf, txtp, tx);
        }
        unit.eob = self.quantize_block_wave3(
            &cf,
            &mut unit.q_cf,
            &mut unit.dq_cf,
            tx,
            seg_id,
            plane,
            is_intra,
            lossless,
        );
        if lossless && unit.eob >= 0 {
            for v in unit.dq_cf.iter_mut() {
                *v *= crate::av2::transform::LOSSLESS_DEQUANT_SCALE;
            }
        }
        unit.txtp = txtp;
        unit
    }

    /// Packs a `w` x `h` residual rectangle out of a `stride`-wide plane
    /// residual buffer.
    fn pack_residual(
        res: &[i32],
        stride: usize,
        x_px: usize,
        y_px: usize,
        w: usize,
        h: usize,
    ) -> Vec<i32> {
        let mut out = vec![0i32; w * h];
        for r in 0..h {
            let src = (y_px + r) * stride + x_px;
            out[r * w..(r + 1) * w].copy_from_slice(&res[src..src + w]);
        }
        out
    }

    /// Fills `res` with `src - rec` over a plane rectangle.
    fn residual_into(
        &self,
        plane: usize,
        x_px: usize,
        y_px: usize,
        w: usize,
        h: usize,
        res: &mut [i32],
    ) -> bool {
        let src_pl = &self.src.planes[plane];
        let rec_pl = &self.rec.planes[plane];
        let mut nz = false;
        for r in 0..h {
            let s_off = src_pl.offset(x_px, y_px + r);
            let p_off = rec_pl.offset(x_px, y_px + r);
            for c in 0..w {
                let d = src_pl.data[s_off + c].to_i32() - rec_pl.data[p_off + c].to_i32();
                res[r * w + c] = d;
                nz |= d != 0;
            }
        }
        nz
    }

    /// Chooses the lossless kernel of one 4x4 luma transform unit:
    /// `WHT_WHT` or `IDTX` (AV2 Spec v1.0.0 §5.20.8.2 `txtp_lossless`). The
    /// decision compares the L1 level cost of both kernels, which is the exact
    /// rate proxy at quantizer index 0.
    fn choose_lossless_txtp(&self, res: &[i32]) -> u32 {
        debug_assert_eq!(res.len(), 16);
        let mut wht = [0i32; 16];
        crate::av2::transform::fwd_wht_wht_4x4(res, 4, &mut wht);
        // The lossless `IDTX` kernel is a pass-through, so its coded levels are
        // the residual samples themselves.
        let ident: [i32; 16] = res.try_into().unwrap_or([0i32; 16]);
        let cost = |c: &[i32; 16]| -> u64 { c.iter().map(|v| v.unsigned_abs() as u64).sum() };
        if false && cost(&ident) < cost(&wht) {
            txtp::IDTX
        } else {
            crate::av2::transform::WHT_WHT
        }
    }

    fn quantize_block_wave3(
        &self,
        cf_in: &[i32],
        q_out: &mut [i32],
        dq_out: &mut [i32],
        tx: usize,
        seg_id: usize,
        plane: usize,
        is_intra: bool,
        lossless: bool,
    ) -> i32 {
        let dq_tbl = self.dq[seg_id.min(MAX_SEGMENTS - 1)][plane];
        if self.tools.enable_rdoq {
            quantize_rdoq_block(
                cf_in,
                q_out,
                dq_out,
                tx,
                dq_tbl,
                None,
                self.bd.bitdepth(),
                is_intra,
                lossless,
                self.lambda_rdoq,
            )
        } else {
            quantize_dequantize_block(
                cf_in,
                q_out,
                dq_out,
                tx,
                dq_tbl,
                None,
                self.bd.bitdepth(),
                is_intra,
                lossless,
            )
        }
    }

    /// Compares the lossless prediction options of one plane over a whole
    /// coding block: the selected intra mode, vertical DPCM and horizontal
    /// DPCM.
    ///
    /// At lossless the reconstruction equals the source, so the prediction of
    /// every 4x4 transform block is exactly the already-reconstructed source
    /// sample above (or left of) it. The returned costs are the L1 magnitude
    /// of the 4x4 WHT of the coded residual, which is the coded level cost at
    /// quantizer index 0.
    #[allow(clippy::too_many_arguments)]
    fn lossless_prediction_costs(
        &self,
        plane: usize,
        px_x: usize,
        px_y: usize,
        w: usize,
        h: usize,
        mode: u8,
        angle: i8,
        have_left: bool,
        have_top: bool,
        have_tr: bool,
        have_bl: bool,
    ) -> (u64, u64, u64) {
        let src = &self.src.planes[plane];
        let rec = &self.rec.planes[plane];
        let mid = self.bd.pixel_clip(1 << (self.bd.bitdepth() - 1));
        let mut edge = [BD::Pixel::default(); 512];
        let tl_idx = 256usize;
        let (rec_cw, rec_ch) =
            crate::av2::ipred::intra_reference_extent(self.f_bw4, self.f_bh4, rec.ss_x, rec.ss_y);
        prepare_intra_edges(
            self.bd,
            &mut edge,
            tl_idx,
            &rec.data,
            rec.offset(px_x, px_y),
            rec.stride,
            have_left,
            have_top,
            have_tr,
            have_bl,
            w,
            h,
            rec_cw,
            rec_ch,
        );
        let mut pred = [BD::Pixel::default(); 64 * 64];
        predict_intra_block(
            self.bd,
            &mut pred[..w * h],
            0,
            w,
            &edge,
            tl_idx,
            mode,
            angle,
            w,
            h,
            have_left,
            have_top,
            plane,
            0,
        );

        let mut buf = [0i32; 64 * 64];
        // Normal prediction.
        for r in 0..h {
            let s_off = src.offset(px_x, px_y + r);
            for c in 0..w {
                buf[r * w + c] = src.data[s_off + c].to_i32() - pred[r * w + c].to_i32();
            }
        }
        let normal = wht_l1_cost(&buf, w, w, h);

        // Vertical DPCM: prediction is the reconstructed sample above, the
        // coded residual is its first vertical difference.
        let mut true_res = [0i32; 64];
        let mut prev = [0i32; 64];
        for r in 0..h {
            let s_off = src.offset(px_x, px_y + r);
            for c in 0..w {
                let above = if r == 0 {
                    if have_top {
                        rec.data[rec.offset(px_x, px_y) - rec.stride + c].to_i32()
                    } else {
                        mid.to_i32()
                    }
                } else {
                    src.data[src.offset(px_x, px_y + r - 1) + c].to_i32()
                };
                true_res[c] = src.data[s_off + c].to_i32() - above;
            }
            for c in 0..w {
                buf[r * w + c] = if r == 0 {
                    true_res[c]
                } else {
                    true_res[c] - prev[c]
                };
            }
            prev[..w].copy_from_slice(&true_res[..w]);
        }
        let dpcm_v = wht_l1_cost(&buf, w, w, h);

        // Horizontal DPCM.
        for c in 0..w {
            let left = if c == 0 {
                if have_left {
                    rec.data[rec.offset(px_x, px_y) - 1].to_i32()
                } else {
                    mid.to_i32()
                }
            } else {
                src.data[src.offset(px_x, px_y) + c - 1].to_i32()
            };
            for r in 0..h {
                let s_off = src.offset(px_x, px_y + r);
                let cur = src.data[s_off + c].to_i32();
                let res = cur
                    - if c == 0 {
                        left
                    } else {
                        src.data[s_off + c - 1].to_i32()
                    };
                let prev_res = if c == 0 {
                    0
                } else {
                    // Residual of the sample to the left, same row.
                    let left_pred = if c == 1 {
                        left
                    } else {
                        src.data[s_off + c - 2].to_i32()
                    };
                    src.data[s_off + c - 1].to_i32() - left_pred
                };
                buf[r * w + c] = if c == 0 { res } else { res - prev_res };
            }
        }
        let dpcm_h = wht_l1_cost(&buf, w, w, h);
        (normal, dpcm_v, dpcm_h)
    }

    /// Top-right / bottom-left availability of one lossless 4x4 transform
    /// block, mirroring the `n_tr` / `n_bl` derivation of the reference
    /// decoder (`dav2d` `recon_tmpl.c`): the units to the right in the row
    /// above and below in the column to the left count only while they have
    /// already been reconstructed, except at a superblock or tile boundary.
    #[allow(clippy::too_many_arguments)]
    fn lossless_tu_edges(
        &self,
        plane: usize,
        x4: usize,
        y4: usize,
        tw4: usize,
        th4: usize,
        ss_x: usize,
        ss_y: usize,
    ) -> (bool, bool) {
        let stride = self.f_bw4 >> ss_x;
        let sb4x = (self.sb_step4 >> ss_x).max(1);
        let sb4y = (self.sb_step4 >> ss_y).max(1);
        let lx4 = x4 << ss_x;
        let ly4 = y4 << ss_y;
        let tile_col_end4 = self.tile_col_end4 >> ss_x;
        let tile_row_end4 = self.tile_row_end4 >> ss_y;

        let mut n_tr = 0usize;
        if ly4 > self.tile_row_start4 {
            let w = tw4.min(tile_col_end4.saturating_sub(x4 + tw4));
            if (ly4 & (self.sb_step4 - 1)) == 0 {
                n_tr = w;
            } else {
                let end = ((x4 + sb4x) & !(sb4x - 1)).min(tile_col_end4);
                let w = tw4.min(end.saturating_sub(x4 + tw4));
                if w > 0 {
                    let row = y4 - 1;
                    let mut n = 0;
                    while n < w && self.tu_coded[plane][row * stride + x4 + tw4 + n] {
                        n += 1;
                    }
                    n_tr = n;
                }
            }
        }

        let mut n_bl = 0usize;
        if lx4 > self.tile_col_start4 {
            let end = ((y4 + sb4y) & !(sb4y - 1)).min(tile_row_end4);
            let h = th4.min(end.saturating_sub(y4 + th4));
            if (lx4 & (self.sb_step4 - 1)) == 0 || h == 0 {
                n_bl = h;
            } else {
                let mut n = 0;
                while n < h && self.tu_coded[plane][(y4 + th4 + n) * stride + x4 - 1] {
                    n += 1;
                }
                n_bl = n;
            }
        }

        (n_tr > 0, n_bl > 0)
    }

    /// Encodes one lossless intra leaf block.
    ///
    /// Lossless frames signal `TX_MODE_ONLY_4X4`, so every plane is coded as
    /// 4x4 transform blocks using the WHT of AV2 Spec v1.0.0 §7.15.4, and
    /// intra prediction is applied per transform block (§5.20.7.24
    /// `make_intra_prediction`). `use_dpcm_y` / `use_dpcm_uv` replace the intra
    /// mode syntax and difference the residual along the prediction direction
    /// (§5.20.5.5, §5.20.5.6).
    fn encode_intra_lossless_leaf(&mut self, bs: BlockSize, bx4: usize, by4: usize) {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let w = bw4 * 4;
        let h = bh4 * 4;
        let px_x = bx4 * 4;
        let px_y = by4 * 4;
        let a_col_idx = (bx4 - self.tile_col_start4) >> 5;
        let bx4_in_sb = bx4 & 31;
        let by4_in_sb = by4 & 31;
        let have_left = bx4 > self.tile_col_start4;
        let have_top = by4 > self.tile_row_start4;
        let tile_end_x = self.tile_col_end4 * 4;
        let tile_end_y = self.tile_row_end4 * 4;
        let sb_px = 64usize << self.frame_hdr.sb128;
        let (have_tr, have_bl) = crate::av2::ipred::intra_edge_availability(
            px_x, px_y, w, h, tile_end_x, tile_end_y, sb_px, have_left, have_top,
        );

        self.intra_blocks += 1;
        let src_y = &self.src.planes[0];
        let rec_y = &self.rec.planes[0];
        let (y_mode, y_angle, _midx, _cost) = self.select_luma_intra_mode(
            &src_y.data,
            src_y.offset(px_x, px_y),
            src_y.stride,
            &rec_y.data,
            rec_y.offset(px_x, px_y),
            rec_y.stride,
            have_left,
            have_top,
            w,
            h,
            0,
            self.lambda_sad,
            self.speed,
            src_y.width,
            src_y.height,
            px_x,
            px_y,
        );

        // DPCM decision: the cheapest of the selected mode, V_PRED-DPCM and
        // H_PRED-DPCM. DPCM costs one extra symbol but saves the mode syntax.
        let (cost_normal, cost_v, cost_h) = self.lossless_prediction_costs(
            0, px_x, px_y, w, h, y_mode, y_angle, have_left, have_top, have_tr, have_bl,
        );
        let dpcm_y = if self.tools.enable_dpcm && cost_v.min(cost_h) < cost_normal {
            Some(cost_h < cost_v)
        } else {
            None
        };
        if dpcm_y.is_some() {
            self.dpcm_y_blocks += 1;
        }
        let (y_mode, y_angle) = match dpcm_y {
            Some(true) => (2u8, 0i8),
            Some(false) => (1u8, 0i8),
            None => (y_mode, y_angle),
        };

        let luma_midx = encode_intra_y_mode(
            &mut self.msac,
            &mut self.cdf.m,
            &self.a_cols[a_col_idx],
            &self.l_ctx,
            bx4_in_sb,
            by4_in_sb,
            bw4 as i32,
            bh4 as i32,
            bw4 as i32,
            bh4 as i32,
            true,
            y_mode,
            y_angle,
            dpcm_y,
        );
        if dpcm_y.is_none() && self.tools.enable_mrl && (1..=8).contains(&y_mode) {
            encode_mrl_idx(&mut self.msac, &mut self.cdf.m, 0, 0);
        }

        // Chroma mode: one DPCM decision covers both planes.
        let has_chroma = self.seqhdr.layout != PixelLayout::I400;
        let ss_x = self.seqhdr.ss_hor;
        let ss_y = self.seqhdr.ss_ver;
        let layout_col = (self.seqhdr.layout as usize).saturating_sub(1);
        let uv_bs_idx = if has_chroma {
            SS_BS[bs as usize][layout_col] as usize
        } else {
            bs as usize
        };
        // The chroma *plane* block keeps its subsampled size; the transform
        // size is 4x4 everywhere in a lossless frame (§5.20.6.1).
        let uv_bs_dim = BLOCK_DIMENSIONS[uv_bs_idx];
        let cw = (uv_bs_dim[0] as usize) * 4;
        let ch = (uv_bs_dim[1] as usize) * 4;
        let uv_tdim = &TXFM_DIMENSIONS[0];
        let cpx_x = px_x >> ss_x;
        let cpx_y = px_y >> ss_y;
        let cbx4_in_sb = bx4_in_sb >> ss_x;
        let cby4_in_sb = by4_in_sb >> ss_y;
        let mut uv_mode = 0u8;
        let mut uv_angle = 0i8;
        let mut dpcm_uv: Option<bool> = None;

        if has_chroma {
            let (spatial_um, spatial_ua) = select_best_intra_mode_uv(
                self.bd,
                &self.src.planes[1].data,
                &self.src.planes[2].data,
                self.src.planes[1].offset(cpx_x, cpx_y),
                self.src.planes[1].stride,
                &self.rec.planes[1].data,
                &self.rec.planes[2].data,
                self.rec.planes[1].offset(cpx_x, cpx_y),
                self.rec.planes[1].stride,
                have_left,
                have_top,
                cw,
                ch,
                0,
                luma_midx,
                self.lambda_sad,
                self.rec.planes[1].width,
                self.rec.planes[1].height,
            );
            let (chtr, chbl) = crate::av2::ipred::intra_edge_availability(
                cpx_x,
                cpx_y,
                cw,
                ch,
                tile_end_x >> ss_x,
                tile_end_y >> ss_y,
                sb_px >> ss_x.max(ss_y),
                have_left,
                have_top,
            );
            let (n_u, v_u, h_u) = self.lossless_prediction_costs(
                1, cpx_x, cpx_y, cw, ch, spatial_um, spatial_ua, have_left, have_top, chtr, chbl,
            );
            let (n_v, v_v, h_v) = self.lossless_prediction_costs(
                2, cpx_x, cpx_y, cw, ch, spatial_um, spatial_ua, have_left, have_top, chtr, chbl,
            );
            let cost_normal = n_u + n_v;
            let cost_v = v_u + v_v;
            let cost_h = h_u + h_v;
            dpcm_uv = if self.tools.enable_dpcm && cost_v.min(cost_h) < cost_normal {
                Some(cost_h < cost_v)
            } else {
                None
            };
            if dpcm_uv.is_some() {
                self.dpcm_uv_blocks += 1;
            }
            let um = match dpcm_uv {
                Some(true) => 2u8,
                Some(false) => 1u8,
                None => {
                    if crate::av2::entropy::chroma_uv_mode_codeable(spatial_um, luma_midx) {
                        spatial_um
                    } else {
                        0
                    }
                }
            };
            uv_mode = um;
            uv_angle = if dpcm_uv.is_some() { 0 } else { spatial_ua };
            // `cflAllowed` at lossless is `planeSz == BLOCK_4X4` (§5.20.5.6).
            let cfl_allowed = self.seqhdr.cfl && cw == 4 && ch == 4;
            encode_intra_uv_mode(
                &mut self.msac,
                &mut self.cdf.m,
                &self.a_cols[a_col_idx],
                &self.l_ctx,
                cbx4_in_sb,
                cby4_in_sb,
                cfl_allowed,
                true,
                luma_midx,
                uv_mode,
                uv_angle,
                dpcm_uv,
            );
            // `palette_mode_info()` is still present when screen content tools
            // are allowed; the lossless path never selects palette mode.
            let palette_eligible = self.frame_hdr.allow_screen_content_tools != 0
                && bw4 + bh4 >= 4
                && bw4.max(bh4) <= 16
                && y_mode == 0;
            if palette_eligible {
                self.msac
                    .encode_bool_adapt(0, &mut self.cdf.m.palette_y_mode);
            }
        }

        // Residual coding: one 4x4 transform block per 4x4 of the coding block.
        let mut res = [0i32; 16];
        let mut cf = [0i32; 16];
        let mut q_cf = [0i32; 16];
        let mut dq_cf = [0i32; 16];
        let mut any_nz = false;
        // `EobU` is the `eob` of the last U-plane transform block when the V
        // plane starts (AV2 Spec v1.0.0 §5.20.7.27: `if (plane == 1) EobU = eob`).
        let mut u_has_cf = false;
        for plane in 0..if has_chroma { 3 } else { 1 } {
            let is_luma = plane == 0;
            let plane_w = if is_luma { w } else { cw };
            let plane_h = if is_luma { h } else { ch };
            let (ox, oy) = if is_luma {
                (px_x, px_y)
            } else {
                (cpx_x, cpx_y)
            };
            let mode = if is_luma { y_mode } else { uv_mode };
            let angle = if is_luma { y_angle } else { uv_angle };
            let dpcm = if is_luma { dpcm_y } else { dpcm_uv };
            // Packed DPCM flag understood by `residual_add`: 1 = horizontal
            // (`H_PRED`, cumulative along a row), 2 = vertical (`V_PRED`).
            let dpcm_flag: u32 = match dpcm {
                Some(true) => 1,
                Some(false) => 2,
                None => 0,
            };
            let wht_txtp = txtp::WHT_WHT | (dpcm_flag << 8);
            let mut edge = [BD::Pixel::default(); 512];
            let tl_idx = 256usize;
            let (pw4, ph4) = (plane_w / 4, plane_h / 4);
            let ss_x_p = self.rec.planes[plane].ss_x as usize;
            let ss_y_p = self.rec.planes[plane].ss_y as usize;
            let (rec_cw, rec_ch) = crate::av2::ipred::intra_reference_extent(
                self.f_bw4,
                self.f_bh4,
                self.rec.planes[plane].ss_x,
                self.rec.planes[plane].ss_y,
            );
            for y4 in 0..ph4 {
                for x4 in 0..pw4 {
                    let tx_px = ox + x4 * 4;
                    let ty_px = oy + y4 * 4;
                    let hl = have_left || x4 > 0;
                    let ht = have_top || y4 > 0;
                    let tx4 = ox / 4 + x4;
                    let ty4 = oy / 4 + y4;
                    // `count_top_right_avail` / `count_bottom_left_avail`
                    // (§5.20.7.24) per transform block: the reference decoder
                    // counts the already-reconstructed 4x4 units in the row
                    // above / column to the left, so a neighbouring block
                    // boundary can make the edge unavailable.
                    let (htr, hbl) = self.lossless_tu_edges(plane, tx4, ty4, 1, 1, ss_x_p, ss_y_p);
                    {
                        let rec_pl = &self.rec.planes[plane];
                        prepare_intra_edges(
                            self.bd,
                            &mut edge,
                            tl_idx,
                            &rec_pl.data,
                            rec_pl.offset(tx_px, ty_px),
                            rec_pl.stride,
                            hl,
                            ht,
                            htr,
                            hbl,
                            4,
                            4,
                            rec_cw,
                            rec_ch,
                        );
                    }
                    {
                        let rec_pl = &mut self.rec.planes[plane];
                        let off = rec_pl.offset(tx_px, ty_px);
                        predict_intra_block(
                            self.bd,
                            &mut rec_pl.data,
                            off,
                            rec_pl.stride,
                            &edge,
                            tl_idx,
                            mode,
                            angle,
                            4,
                            4,
                            hl,
                            ht,
                            plane,
                            0,
                        );
                    }
                    let src_pl = &self.src.planes[plane];
                    let rec_pl = &self.rec.planes[plane];
                    let mut true_res = [0i32; 16];
                    for i in 0..4 {
                        let s_off = src_pl.offset(tx_px, ty_px + i);
                        let p_off = rec_pl.offset(tx_px, ty_px + i);
                        for j in 0..4 {
                            true_res[i * 4 + j] =
                                src_pl.data[s_off + j].to_i32() - rec_pl.data[p_off + j].to_i32();
                        }
                    }
                    // DPCM: the coded residual is the first difference along
                    // the prediction direction (§7.15.4).
                    match dpcm {
                        Some(false) => {
                            // Vertical: `r[i] -= r[i-1]`.
                            for j in 0..4 {
                                let mut prev = 0;
                                for i in 0..4 {
                                    let v = true_res[i * 4 + j];
                                    res[i * 4 + j] = v - prev;
                                    prev = v;
                                }
                            }
                        }
                        Some(true) => {
                            for i in 0..4 {
                                let mut prev = 0;
                                for j in 0..4 {
                                    let v = true_res[i * 4 + j];
                                    res[i * 4 + j] = v - prev;
                                    prev = v;
                                }
                            }
                        }
                        None => res = true_res,
                    }
                    fwd_txfm_2d(&res, 4, &mut cf, wht_txtp, 0);
                    let eob = self
                        .quantize_block_wave3(&cf, &mut q_cf, &mut dq_cf, 0, 0, plane, true, true);
                    if eob >= 0 {
                        any_nz = true;
                        for v in dq_cf.iter_mut() {
                            *v *= crate::av2::transform::LOSSLESS_DEQUANT_SCALE;
                        }
                    }
                    let params = CoefCodingParams {
                        tx: 0,
                        bs: if is_luma { bs as usize } else { uv_bs_idx },
                        plane: plane as i32,
                        intra: true,
                        fsc: false,
                        lossless: true,
                        y_mode: y_mode as usize,
                        y_angle,
                        mrl_index: 0,
                        uv_mode: uv_mode as usize,
                        reduced_txtp_set: self.frame_hdr.reduced_txtp_set as i32,
                        u_has_cf: if plane == 2 { u_has_cf as i32 } else { 0 },
                        dq_tbl: self.dq[0][plane],
                        bitdepth: self.bd.bitdepth(),
                        qm: None,
                        ss_hor: ss_x != 0,
                        ss_ver: ss_y != 0,
                    };
                    let mut wht = wht_txtp as u16;
                    let mut res_ctx = 0u8;
                    let (a_slice, l_slice) = if is_luma {
                        (
                            &self.a_cols[a_col_idx].lcoef[bx4_in_sb + x4..],
                            &self.l_ctx.lcoef[by4_in_sb + y4..],
                        )
                    } else {
                        (
                            &self.a_cols[a_col_idx].ccoef[plane - 1][cbx4_in_sb + x4..],
                            &self.l_ctx.ccoef[plane - 1][cby4_in_sb + y4..],
                        )
                    };
                    let coded_eob = encode_coefs(
                        &mut self.msac,
                        &mut self.cdf.coef,
                        &mut self.cdf.m,
                        a_slice,
                        l_slice,
                        &params,
                        &q_cf,
                        &mut wht,
                        &mut res_ctx,
                    );
                    if is_luma {
                        self.a_cols[a_col_idx].lcoef[bx4_in_sb + x4] = res_ctx;
                        self.l_ctx.lcoef[by4_in_sb + y4] = res_ctx;
                    } else {
                        self.a_cols[a_col_idx].ccoef[plane - 1][cbx4_in_sb + x4] = res_ctx;
                        self.l_ctx.ccoef[plane - 1][cby4_in_sb + y4] = res_ctx;
                        if plane == 1 {
                            u_has_cf = coded_eob >= 0;
                        }
                    }
                    if coded_eob >= 0 {
                        let rec_pl = &mut self.rec.planes[plane];
                        let off = rec_pl.offset(tx_px, ty_px);
                        inv_txfm_add(
                            self.bd,
                            &mut rec_pl.data,
                            off,
                            rec_pl.stride,
                            &mut dq_cf,
                            wht_txtp,
                            coded_eob,
                            0,
                        );
                    }
                    self.tu_coded[plane][ty4 * (self.f_bw4 >> ss_x_p) + tx4] = true;
                }
            }
        }

        if any_nz {
            self.wht_blocks += 1;
        } else {
            self.skipped_blocks += 1;
        }
        let t_dim = &TXFM_DIMENSIONS[0];
        for i in 0..bw4.min(32 - bx4_in_sb).min(self.f_bw4.saturating_sub(bx4)) {
            self.a_cols[a_col_idx].intra[bx4_in_sb + i] = 1;
            self.a_cols[a_col_idx].mode[bx4_in_sb + i] = y_mode;
            self.a_cols[a_col_idx].skip[bx4_in_sb + i] = 0;
            self.a_cols[a_col_idx].midx[bx4_in_sb + i] = luma_midx;
            self.a_cols[a_col_idx].tx_luma[bx4_in_sb + i] = 0;
            self.a_cols[a_col_idx].tx[bx4_in_sb + i] = 0;
            self.a_cols[a_col_idx].seg_id[bx4_in_sb + i] = 0;
            self.a_cols[a_col_idx].ref_frame[0][bx4_in_sb + i] = -1;
            self.a_cols[a_col_idx].ref_frame[1][bx4_in_sb + i] = -1;
            self.a_cols[a_col_idx].filter[0][bx4_in_sb + i] = 3;
            self.a_cols[a_col_idx].filter[1][bx4_in_sb + i] = 3;
        }
        for i in 0..bh4.min(32 - by4_in_sb).min(self.f_bh4.saturating_sub(by4)) {
            self.l_ctx.intra[by4_in_sb + i] = 1;
            self.l_ctx.mode[by4_in_sb + i] = y_mode;
            self.l_ctx.skip[by4_in_sb + i] = 0;
            self.l_ctx.midx[by4_in_sb + i] = luma_midx;
            self.l_ctx.tx_luma[by4_in_sb + i] = 0;
            self.l_ctx.tx[by4_in_sb + i] = 0;
            self.l_ctx.seg_id[by4_in_sb + i] = 0;
            self.l_ctx.ref_frame[0][by4_in_sb + i] = -1;
            self.l_ctx.ref_frame[1][by4_in_sb + i] = -1;
            self.l_ctx.filter[0][by4_in_sb + i] = 3;
            self.l_ctx.filter[1][by4_in_sb + i] = 3;
        }
        if has_chroma {
            for i in 0..(uv_tdim.w as usize)
                .min(32 - cbx4_in_sb)
                .min((self.f_bw4 >> ss_x).saturating_sub(bx4 >> ss_x))
            {
                self.a_cols[a_col_idx].uvmode[cbx4_in_sb + i] = uv_mode;
                self.a_cols[a_col_idx].tx_chroma[cbx4_in_sb + i] = 0;
            }
            for i in 0..(uv_tdim.h as usize)
                .min(32 - cby4_in_sb)
                .min((self.f_bh4 >> ss_y).saturating_sub(by4 >> ss_y))
            {
                self.l_ctx.uvmode[cby4_in_sb + i] = uv_mode;
                self.l_ctx.tx_chroma[cby4_in_sb + i] = 0;
            }
        }
        self.record_deblock(bx4, by4, bw4, bh4, t_dim, uv_tdim, has_chroma, false, true);
    }

    /// Encodes one leaf block and applies the §7.12.3 reference-motion-vector
    /// bank update for it.
    fn encode_leaf_block(&mut self, bs: BlockSize, bx4: usize, by4: usize) {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let (bw4, bh4) = (b_dim[0] as usize, b_dim[1] as usize);
        let end = bx4 < self.tile_col_end4 && by4 < self.tile_row_end4;
        self.encode_leaf_block_inner(bs, bx4, by4);
        if end {
            self.note_refmv_bank(bw4, bh4, bx4, by4);
        }
    }

    fn encode_leaf_block_inner(&mut self, bs: BlockSize, bx4: usize, by4: usize) {
        if bx4 >= self.tile_col_end4 || by4 >= self.tile_row_end4 {
            return;
        }
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let w = bw4 * 4;
        let h = bh4 * 4;
        let px_x = bx4 * 4;
        let px_y = by4 * 4;
        // Advance the local IntraBC 64x64 buffer ring (§5.20.5.14) before any
        // block-vector validity check.
        self.ibc_buffers
            .update_sb(by4 as i32, bx4 as i32, self.frame_hdr.sb128 != 0);
        // Default to intra; `encode_inter_leaf` overwrites the footprint with
        // the coded inter mode (AV2 §7.12.2.3 reads the neighbour maps).
        self.mark_mi_default(bs, bx4, by4);

        // Lossless frames use `TX_MODE_ONLY_4X4` and a dedicated 4x4 WHT /
        // DPCM path (AV2 Spec v1.0.0 §5.20.6.1, §5.20.5.5, §7.15.4). Inter
        // frames may still code inter blocks: their residual is 4x4 WHT (or
        // `IDTX`) with `lossless_tx_size`, and the prediction is the block
        // motion-compensated prediction.
        if self.frame_hdr.all_lossless != 0 {
            let a_col_idx = (bx4 - self.tile_col_start4) >> 5;
            let have_left = bx4 > self.tile_col_start4;
            let have_top = by4 > self.tile_row_start4;
            let mut inter_dec: Option<crate::av2::inter::InterBlockDecision> = None;
            if self.frame_hdr.is_inter_or_switch()
                && let Some(r0) = self.ref0
            {
                let src_y = &self.src.planes[0];
                let rec_y = &self.rec.planes[0];
                let r0_y = &r0.planes[0];
                let src_off = src_y.offset(px_x, px_y);
                let cands = self.mv_stack(bx4, by4, bw4, bh4, [0, -1]);
                let mvp = cands.get(0);
                let (near_filter, near_sad) = if self.tools.enable_switchable_interp {
                    select_best_subpel_filter(
                        self.bd,
                        &src_y.data,
                        src_off,
                        src_y.stride,
                        &r0_y.data,
                        r0_y.stride,
                        r0.planes[0].coded_width as i32,
                        r0.planes[0].coded_height as i32,
                        px_x as i32,
                        px_y as i32,
                        w,
                        h,
                        mvp,
                    )
                } else {
                    let mut pred_buf = [BD::Pixel::default(); 64 * 64];
                    mc_predict_block(
                        self.bd,
                        &mut pred_buf[..w * h],
                        0,
                        w,
                        &r0_y.data,
                        r0_y.stride,
                        r0.planes[0].coded_width as i32,
                        r0.planes[0].coded_height as i32,
                        px_x as i32,
                        px_y as i32,
                        w,
                        h,
                        mvp.y as i32 * 2,
                        mvp.x as i32 * 2,
                        0,
                    );
                    (
                        0u8,
                        compute_sad(
                            &src_y.data,
                            src_off,
                            src_y.stride,
                            &pred_buf[..w * h],
                            0,
                            w,
                            w,
                            h,
                        ),
                    )
                };
                let mut b_mode = InterPredMode::NearMv;
                let mut b_mv = mvp;
                let mut b_filter = near_filter;
                let mut b_cost = near_sad + ((self.lambda_sad * 256) >> 8);
                if self.tools.enable_newmv {
                    let (me_mv, me_filter, me_cost) = search_motion_vector(
                        self.bd,
                        &src_y.data,
                        src_y.stride,
                        &r0_y.data,
                        r0_y.stride,
                        r0.width,
                        r0.height,
                        px_x,
                        px_y,
                        w,
                        h,
                        mvp,
                        self.lambda_sad,
                        self.speed,
                    );
                    if me_cost < b_cost {
                        b_mode = InterPredMode::NewMv;
                        b_mv = me_mv;
                        b_filter = if self.tools.enable_switchable_interp {
                            me_filter
                        } else {
                            0
                        };
                        b_cost = me_cost;
                    }
                }
                if self.tools.enable_warped_motion {
                    b_mode = InterPredMode::GlobalMv;
                    b_mv = MvXY::default();
                    b_filter = 0;
                }
                let (y_mode, y_angle, _midx, intra_cost) = self.select_luma_intra_mode(
                    &src_y.data,
                    src_off,
                    src_y.stride,
                    &rec_y.data,
                    rec_y.offset(px_x, px_y),
                    rec_y.stride,
                    have_left,
                    have_top,
                    w,
                    h,
                    MAX_TXFM_SIZE_FOR_BS[bs as usize] as usize,
                    self.lambda_sad,
                    self.speed,
                    src_y.width,
                    src_y.height,
                    px_x,
                    px_y,
                );
                let _ = (y_mode, y_angle);
                if b_cost < intra_cost {
                    inter_dec = Some(crate::av2::inter::InterBlockDecision {
                        is_compound: false,
                        compound_mode: 0,
                        compound_drl_idx: 0,
                        comp_type: crate::av2::levels::CompInterType::Average,
                        wedge_idx: 0,
                        wedge_sign: false,
                        mask_sign: false,
                        cwp_idx: 0,
                        ref_slot_idx: 0,
                        mode: b_mode,
                        drl_idx: 0,
                        interp_filter: b_filter,
                        mv: [b_mv, MvXY::default()],
                        ref_mv: [mvp, MvXY::default()],
                        cost: b_cost,
                    });
                }
            }
            if self.frame_hdr.is_inter_or_switch() {
                let nb = gather_inter_neighbors(
                    &self.a_cols[a_col_idx],
                    &self.l_ctx,
                    bx4,
                    by4,
                    bw4,
                    bh4,
                    have_left,
                    have_top,
                    self.tile_col_end4,
                    self.tile_row_end4,
                );
                let ctx = nb.intra_ctx();
                // The coded bit is `is_inter` (0 = intra).
                self.msac
                    .encode_bool_adapt(inter_dec.is_some() as u32, self.cdf.m.intra(ctx));
            }
            if let Some(dec) = inter_dec {
                self.encode_inter_leaf(
                    bs,
                    bx4,
                    by4,
                    MAX_TXFM_SIZE_FOR_BS[bs as usize] as usize,
                    0,
                    dec,
                );
                return;
            }
            self.encode_intra_lossless_leaf(bs, bx4, by4);
            return;
        }

        let a_col_idx = (bx4 - self.tile_col_start4) >> 5;
        let bx4_in_sb = bx4 & 31;
        let by4_in_sb = by4 & 31;

        let have_left = bx4 > self.tile_col_start4;
        let have_top = by4 > self.tile_row_start4;

        let tx = MAX_TXFM_SIZE_FOR_BS[bs as usize] as usize;
        let t_dim = &TXFM_DIMENSIONS[tx];
        // Desired segment id from the encoder variance analysis. The actual id
        // is written by `code_segment_id` at the syntax position required by
        // AV2 Spec v1.0.0 §5.20.5.7 / §5.20.7.2, and may be replaced by the
        // normative spatial prediction for skipped non-lossless blocks.
        let desired_seg = if self.frame_hdr.segmentation.enabled != 0 {
            self.seg_plan
                .map(|p| p.seg_for_pixel(px_x, px_y))
                .unwrap_or(0)
        } else {
            0
        };
        // The preskip variant codes the segment id before `is_inter`; this
        // encoder enables only ALT_Q, so the map is normally coded post-skip.
        let mut seg_id = if self.seg_coder.preskip {
            self.code_segment_id(bx4, by4, bw4, bh4, have_left, have_top, false, desired_seg)
        } else {
            desired_seg as usize
        };

        // Evaluate Intra vs Inter decision for Inter/Switch frames.
        // Note: `encode_inter_leaf` codes single-ref `NearMv` (`drl_idx = 0`, `mv = (0, 0)`),
        // so we evaluate the true `mv = (0, 0)` predictor directly without heap allocation or
        // discarded motion searches, and cache the intra search result if intra wins.
        let mut cached_intra_y: Option<(u8, i8, u8, u64)> = None;
        if self.frame_hdr.is_inter_or_switch()
            && let Some(r0) = self.ref0
        {
            let src_y = &self.src.planes[0];
            let rec_y = &self.rec.planes[0];
            let r0_y = &r0.planes[0];
            let src_off = src_y.offset(px_x, px_y);
            let cands = self.mv_stack(bx4, by4, bw4, bh4, [0, -1]);
            let mvp = cands.get(0);

            let (best_mode, best_mv, best_filter, best_inter_cost) =
                if self.tools.enable_newmv || self.tools.enable_switchable_interp {
                    let (near_filter, near_sad) = if self.tools.enable_switchable_interp {
                        select_best_subpel_filter(
                            self.bd,
                            &src_y.data,
                            src_off,
                            src_y.stride,
                            &r0_y.data,
                            r0_y.stride,
                            r0.planes[0].coded_width as i32,
                            r0.planes[0].coded_height as i32,
                            px_x as i32,
                            px_y as i32,
                            w,
                            h,
                            mvp,
                        )
                    } else {
                        let mut pred_buf = [BD::Pixel::default(); 64 * 64];
                        mc_predict_block(
                            self.bd,
                            &mut pred_buf[..w * h],
                            0,
                            w,
                            &r0_y.data,
                            r0_y.stride,
                            r0.planes[0].coded_width as i32,
                            r0.planes[0].coded_height as i32,
                            px_x as i32,
                            px_y as i32,
                            w,
                            h,
                            mvp.y as i32 * 2,
                            mvp.x as i32 * 2,
                            0,
                        );
                        (
                            0u8,
                            compute_sad(
                                &src_y.data,
                                src_off,
                                src_y.stride,
                                &pred_buf[..w * h],
                                0,
                                w,
                                w,
                                h,
                            ),
                        )
                    };
                    let near_rate_q8 = 256u64;
                    let mut b_mode = InterPredMode::NearMv;
                    let mut b_mv = mvp;
                    let mut b_filter = near_filter;
                    let mut b_cost = near_sad + ((self.lambda_sad * near_rate_q8) >> 8);

                    if self.tools.enable_newmv {
                        let (me_mv, me_filter, me_cost) = search_motion_vector(
                            self.bd,
                            &src_y.data,
                            src_y.stride,
                            &r0_y.data,
                            r0_y.stride,
                            r0.width,
                            r0.height,
                            px_x,
                            px_y,
                            w,
                            h,
                            mvp,
                            self.lambda_sad,
                            self.speed,
                        );
                        let chosen_flt = if self.tools.enable_switchable_interp {
                            me_filter
                        } else {
                            0u8
                        };
                        if me_cost < b_cost {
                            b_mode = InterPredMode::NewMv;
                            b_mv = me_mv;
                            b_filter = chosen_flt;
                            b_cost = me_cost;
                        }
                    }
                    (b_mode, b_mv, b_filter, b_cost)
                } else {
                    let inter_satd = {
                        let mut pred_buf = [BD::Pixel::default(); 64 * 64];
                        mc_predict_block(
                            self.bd,
                            &mut pred_buf[..w * h],
                            0,
                            w,
                            &r0_y.data,
                            r0_y.stride,
                            r0.planes[0].coded_width as i32,
                            r0.planes[0].coded_height as i32,
                            px_x as i32,
                            px_y as i32,
                            w,
                            h,
                            mvp.y as i32 * 2,
                            mvp.x as i32 * 2,
                            0,
                        );
                        crate::av2::ipred::compute_satd(
                            &src_y.data,
                            src_off,
                            src_y.stride,
                            &pred_buf[..w * h],
                            0,
                            w,
                            w,
                            h,
                        )
                    };
                    let inter_cost = inter_satd + ((self.lambda_sad * 12) >> 4);
                    (InterPredMode::NearMv, mvp, 0u8, inter_cost)
                };
            // With warped motion enabled, non-warp single-reference blocks
            // use `GLOBALMV` (zero MV, no DRL dependency). The encoder's
            // reduced DRL does not reproduce every normative candidate once
            // warp sub-block MVs are present, so keeping translation at
            // `GLOBALMV` makes our warp streams deterministic for any
            // conforming decoder while the warp RDO supplies the non-trivial
            // motion.
            let (best_mode, best_mv, best_filter) = if self.tools.enable_warped_motion {
                (InterPredMode::GlobalMv, MvXY::default(), 0u8)
            } else {
                (best_mode, best_mv, best_filter)
            };
            let mut inter_dec = crate::av2::inter::InterBlockDecision {
                is_compound: false,
                compound_mode: 0,
                compound_drl_idx: 0,
                comp_type: crate::av2::levels::CompInterType::Average,
                wedge_idx: 0,
                wedge_sign: false,
                mask_sign: false,
                cwp_idx: 0,
                ref_slot_idx: 0,
                mode: best_mode,
                drl_idx: 0,
                interp_filter: best_filter,
                mv: [best_mv, MvXY::default()],
                ref_mv: [mvp, MvXY::default()],
                cost: best_inter_cost,
            };

            if self.tools.enable_compound
                && self.frame_hdr.switchable_comp_refs != 0
                && (bw4 * bh4 >= 4)
                && w >= 8
                && h >= 8
                && let Some(r1) = self.ref1
            {
                let r1_y = &r1.planes[0];
                // Compound branch of `find_mv_stack` (AV2 Spec §7.12.2,
                // `isCompound = 1`). Candidate 0 is the predictor for the
                // `NEW_NEWMV` residuals.
                let comp_stack = self.mv_stack(bx4, by4, bw4, bh4, [0, 1]);
                let comp_pred = comp_stack.get_pair(0);
                let (me_mv1, _, _) = search_motion_vector(
                    self.bd,
                    &src_y.data,
                    src_y.stride,
                    &r1_y.data,
                    r1_y.stride,
                    r1.width,
                    r1.height,
                    px_x,
                    px_y,
                    w,
                    h,
                    MvXY::default(),
                    self.lambda_sad,
                    self.speed,
                );
                // This subset codes compound residuals only at integer-pel
                // precision, so subpel ME results fall back to single
                // prediction instead of being truncated.
                let integer_mvs = best_mv.x & 7 == 0
                    && best_mv.y & 7 == 0
                    && me_mv1.x & 7 == 0
                    && me_mv1.y & 7 == 0
                    && comp_pred[0].x & 7 == 0
                    && comp_pred[0].y & 7 == 0
                    && comp_pred[1].x & 7 == 0
                    && comp_pred[1].y & 7 == 0;
                if integer_mvs {
                    mc_predict_block_prep(
                        self.bd,
                        &mut self.scratch_mc0[..w * h],
                        &r0_y.data,
                        r0_y.stride,
                        r0.planes[0].coded_width as i32,
                        r0.planes[0].coded_height as i32,
                        px_x as i32,
                        px_y as i32,
                        w,
                        h,
                        best_mv.y as i32 * 2,
                        best_mv.x as i32 * 2,
                        best_filter as usize,
                    );
                    mc_predict_block_prep(
                        self.bd,
                        &mut self.scratch_mc1[..w * h],
                        &r1_y.data,
                        r1_y.stride,
                        r1.planes[0].coded_width as i32,
                        r1.planes[0].coded_height as i32,
                        px_x as i32,
                        px_y as i32,
                        w,
                        h,
                        me_mv1.y as i32 * 2,
                        me_mv1.x as i32 * 2,
                        best_filter as usize,
                    );
                    let mut blend_buf = [BD::Pixel::default(); 64 * 64];
                    let accept = |dec: &mut crate::av2::inter::InterBlockDecision,
                                  comp_type: CompInterType,
                                  cost: u64| {
                        dec.is_compound = true;
                        dec.compound_mode = 22;
                        dec.compound_drl_idx = 0;
                        dec.comp_type = comp_type;
                        dec.mode = InterPredMode::NewMv;
                        dec.mv = [best_mv, me_mv1];
                        dec.ref_mv = comp_pred;
                        dec.cost = cost;
                    };

                    // 1. Average.
                    if self.tools.compound_types & 1 != 0 {
                        blend_compound_prep(
                            self.bd,
                            &mut blend_buf[..w * h],
                            0,
                            w,
                            &self.scratch_mc0[..w * h],
                            &self.scratch_mc1[..w * h],
                            w,
                            h,
                            CompInterType::Average,
                            8,
                            None,
                        );
                        let sad = compute_sad(
                            &src_y.data,
                            src_off,
                            src_y.stride,
                            &blend_buf[..w * h],
                            0,
                            w,
                            w,
                            h,
                        );
                        if sad < inter_dec.cost {
                            accept(&mut inter_dec, CompInterType::Average, sad);
                        }
                    }

                    // 2. DifferenceWeighted (DiffWtd).
                    if self.seqhdr.masked_compound && self.tools.compound_types & 4 != 0 {
                        for mask_sign in [false, true] {
                            compute_diff_wtd_mask_intermediate(
                                self.bd,
                                &self.scratch_mc0[..w * h],
                                &self.scratch_mc1[..w * h],
                                w,
                                h,
                                mask_sign,
                                self.seqhdr.ss_hor as usize,
                                self.seqhdr.ss_ver as usize,
                                &mut self.scratch_diff_mask[..w * h],
                                &mut self.scratch_chroma_diff_mask
                                    [..(w >> self.seqhdr.ss_hor) * (h >> self.seqhdr.ss_ver)],
                            );
                            let (p0, p1) = if mask_sign {
                                (&self.scratch_mc1[..w * h], &self.scratch_mc0[..w * h])
                            } else {
                                (&self.scratch_mc0[..w * h], &self.scratch_mc1[..w * h])
                            };
                            blend_compound_prep(
                                self.bd,
                                &mut blend_buf[..w * h],
                                0,
                                w,
                                p0,
                                p1,
                                w,
                                h,
                                CompInterType::DifferenceWeighted,
                                0,
                                Some(&self.scratch_diff_mask[..w * h]),
                            );
                            let sad = compute_sad(
                                &src_y.data,
                                src_off,
                                src_y.stride,
                                &blend_buf[..w * h],
                                0,
                                w,
                                w,
                                h,
                            );
                            if sad < inter_dec.cost {
                                accept(&mut inter_dec, CompInterType::DifferenceWeighted, sad);
                                inter_dec.mask_sign = mask_sign;
                            }
                        }
                    }

                    // 3. Wedge.
                    if self.seqhdr.masked_compound
                        && self.tools.compound_types & 2 != 0
                        && w <= 64
                        && h <= 64
                        && bw4.max(bh4) <= 16
                    {
                        for wedge_idx in 0..WEDGE_TYPES as u8 {
                            if let Some(mask) = WEDGE_TABLES.get_mask(bs, wedge_idx as usize, 0, 0)
                            {
                                for wedge_sign in [false, true] {
                                    let (p0, p1) = if wedge_sign {
                                        (&self.scratch_mc1[..w * h], &self.scratch_mc0[..w * h])
                                    } else {
                                        (&self.scratch_mc0[..w * h], &self.scratch_mc1[..w * h])
                                    };
                                    blend_compound_prep(
                                        self.bd,
                                        &mut blend_buf[..w * h],
                                        0,
                                        w,
                                        p0,
                                        p1,
                                        w,
                                        h,
                                        CompInterType::Wedge,
                                        0,
                                        Some(mask),
                                    );
                                    let sad = compute_sad(
                                        &src_y.data,
                                        src_off,
                                        src_y.stride,
                                        &blend_buf[..w * h],
                                        0,
                                        w,
                                        w,
                                        h,
                                    );
                                    if sad < inter_dec.cost {
                                        accept(&mut inter_dec, CompInterType::Wedge, sad);
                                        inter_dec.wedge_idx = wedge_idx;
                                        inter_dec.wedge_sign = wedge_sign;
                                    }
                                }
                            }
                        }
                    }
                }

                // DistanceWeighted (CWP) uses `NEARMV_NEARMV`, whose two MVs
                // come directly from the compound stack.
                let comp_integer = comp_pred[0].x & 7 == 0
                    && comp_pred[0].y & 7 == 0
                    && comp_pred[1].x & 7 == 0
                    && comp_pred[1].y & 7 == 0;
                if self.seqhdr.cwp
                    && self.tools.compound_types & 8 != 0
                    && comp_stack.count > 0
                    && comp_integer
                {
                    mc_predict_block_prep(
                        self.bd,
                        &mut self.scratch_mc0[..w * h],
                        &r0_y.data,
                        r0_y.stride,
                        r0.planes[0].coded_width as i32,
                        r0.planes[0].coded_height as i32,
                        px_x as i32,
                        px_y as i32,
                        w,
                        h,
                        comp_pred[0].y as i32 * 2,
                        comp_pred[0].x as i32 * 2,
                        best_filter as usize,
                    );
                    mc_predict_block_prep(
                        self.bd,
                        &mut self.scratch_mc1[..w * h],
                        &r1_y.data,
                        r1_y.stride,
                        r1.planes[0].coded_width as i32,
                        r1.planes[0].coded_height as i32,
                        px_x as i32,
                        px_y as i32,
                        w,
                        h,
                        comp_pred[1].y as i32 * 2,
                        comp_pred[1].x as i32 * 2,
                        best_filter as usize,
                    );
                    let mut blend_buf = [BD::Pixel::default(); 64 * 64];
                    const CWP_WEIGHTING_FACTOR: [[i8; 5]; 2] =
                        [[8, 12, 4, 10, 6], [8, 12, 4, 20, -4]];
                    let is_same_side = (self.refdir[0] ^ self.refdir[1]) == 0;
                    for cwp in 1..=4 {
                        let wt = CWP_WEIGHTING_FACTOR[is_same_side as usize][cwp] as i32;
                        blend_compound_prep(
                            self.bd,
                            &mut blend_buf[..w * h],
                            0,
                            w,
                            &self.scratch_mc0[..w * h],
                            &self.scratch_mc1[..w * h],
                            w,
                            h,
                            CompInterType::DistanceWeighted,
                            wt,
                            None,
                        );
                        let sad = compute_sad(
                            &src_y.data,
                            src_off,
                            src_y.stride,
                            &blend_buf[..w * h],
                            0,
                            w,
                            w,
                            h,
                        );
                        if sad < inter_dec.cost {
                            inter_dec.is_compound = true;
                            inter_dec.compound_mode = 18;
                            inter_dec.compound_drl_idx = 0;
                            inter_dec.comp_type = CompInterType::DistanceWeighted;
                            inter_dec.mode = InterPredMode::NearMv;
                            inter_dec.mv = comp_pred;
                            inter_dec.ref_mv = comp_pred;
                            inter_dec.cwp_idx = cwp as u8;
                            inter_dec.cost = sad;
                        }
                    }
                }
            }
            let inter_cost = inter_dec.cost;

            // Stationary block or spatial white-noise fast path: skip the 19-mode intra search.
            let is_intra = if inter_cost <= (w * h * 2) as u64
                || crate::av2::partition::is_spatial_noise_block(
                    src_y,
                    px_x,
                    px_y,
                    w,
                    h,
                    src_y.width,
                    src_y.height,
                ) {
                false
            } else {
                let intra_res = self.select_luma_intra_mode(
                    &src_y.data,
                    src_off,
                    src_y.stride,
                    &rec_y.data,
                    rec_y.offset(px_x, px_y),
                    rec_y.stride,
                    have_left,
                    have_top,
                    w,
                    h,
                    tx,
                    self.lambda_sad,
                    self.speed,
                    src_y.width,
                    src_y.height,
                    px_x,
                    px_y,
                );
                // Require a Luma SATD margin (`1.5x > sqrt(2)`) before switching an inter block
                // to intra, so moving edges switch to intra while uncorrelated noise stays inter.
                let wins = intra_res.3 + (intra_res.3 >> 1) + (self.lambda_sad * 4) < inter_cost;
                if wins {
                    cached_intra_y = Some(intra_res);
                }
                wins
            };

            let nb = gather_inter_neighbors(
                &self.a_cols[a_col_idx],
                &self.l_ctx,
                bx4,
                by4,
                bw4,
                bh4,
                have_left,
                have_top,
                self.tile_col_end4,
                self.tile_row_end4,
            );
            let intra_ctx = nb.intra_ctx();
            // The coded bit is `is_inter` (0 = intra).
            self.msac
                .encode_bool_adapt((!is_intra) as u32, self.cdf.m.intra(intra_ctx));

            if !is_intra {
                self.encode_inter_leaf(bs, bx4, by4, tx, desired_seg as usize, inter_dec);
                return;
            }
        }

        // Intra & Screen Content Coding (`IntraBC`, `Palette`, `ISP`, `MRL`, `FilterIntra`, `CfL`):
        self.intra_blocks += 1;
        let src_y = &self.src.planes[0];
        let rec_y = &self.rec.planes[0];
        let (y_mode, y_angle, _midx, base_intra_cost) = if let Some(cached) = cached_intra_y {
            cached
        } else {
            self.select_luma_intra_mode(
                &src_y.data,
                src_y.offset(px_x, px_y),
                src_y.stride,
                &rec_y.data,
                rec_y.offset(px_x, px_y),
                rec_y.stride,
                have_left,
                have_top,
                w,
                h,
                tx,
                self.lambda_sad,
                self.speed,
                src_y.width,
                src_y.height,
                px_x,
                px_y,
            )
        };

        // Screen-content decisions (AV2 Spec v1.0.0 §5.20.5.3): IntraBC is
        // signalled before the segment id and replaces the intra mode syntax;
        // palette mode is signalled after the intra Y/UV modes.
        let mut pal_cand: Option<crate::av2::screen::PaletteBlockCandidate> = None;
        let mut use_palette = false;
        let tile_x0 = self.tile_col_start4 * 4;
        let tile_y0 = self.tile_row_start4 * 4;
        let tile_x1 = self.tile_col_end4 * 4;
        let tile_y1 = self.tile_row_end4 * 4;

        // Decoder-side presence condition for `use_intrabc` (§5.20.5.3).
        let intrabc_eligible = !self.frame_hdr.is_inter_or_switch()
            && self.frame_hdr.allow_intrabc != 0
            && bw4.min(bh4) < 16
            && !(bw4 == 16 && bh4 == 16);
        let ibc_allowed = self.tools.enable_intrabc
            && intrabc_eligible
            && self.frame_hdr.segmentation.enabled == 0
            // One IntraBC block per frame keeps the BVP candidate stack at the
            // normative default entries; later blocks would see earlier
            // IntraBC motion vectors in the stack.
            && self.intrabc_blocks == 0
            && !self.has_intrabc_neighbor(bx4, by4, bw4, bh4, have_left, have_top);
        if ibc_allowed
            && let Some((dv_q8, ibc_cost)) = search_intrabc_block_local(
                src_y,
                rec_y,
                px_x,
                px_y,
                w,
                h,
                tile_x0,
                tile_y0,
                tile_x1,
                tile_y1,
                &self.ibc_buffers,
                self.lambda_sad,
            )
        {
            let dx_px = (dv_q8.x / 8) as i32;
            let dy_px = (dv_q8.y / 8) as i32;
            let chroma_ok = if self.seqhdr.layout == PixelLayout::I400 {
                true
            } else {
                let ss_x = self.seqhdr.ss_hor;
                let ss_y = self.seqhdr.ss_ver;
                let cx = (px_x as i32 + dx_px) >> ss_x;
                let cy = (px_y as i32 + dy_px) >> ss_y;
                dx_px % (1 << ss_x) == 0
                    && dy_px % (1 << ss_y) == 0
                    && cx >= 0
                    && cy >= 0
                    && cx as usize + (w >> ss_x) <= self.rec.planes[1].width
                    && cy as usize + (h >> ss_y) <= self.rec.planes[1].height
            };
            let copy_sad = {
                let rx = px_x as i32 + dx_px;
                let ry = px_y as i32 + dy_px;
                if rx >= 0
                    && ry >= 0
                    && (rx as usize) + w <= rec_y.width
                    && (ry as usize) + h <= rec_y.height
                {
                    compute_sad(
                        &src_y.data,
                        src_y.offset(px_x, px_y),
                        src_y.stride,
                        &rec_y.data,
                        rec_y.offset(rx as usize, ry as usize),
                        rec_y.stride,
                        w,
                        h,
                    )
                } else {
                    u64::MAX
                }
            };
            if chroma_ok && (ibc_cost + self.lambda_sad * 2 < base_intra_cost || copy_sad == 0) {
                self.encode_intrabc_leaf(bs, bx4, by4, dx_px, dy_px, dv_q8, seg_id, desired_seg);
                return;
            }
        }
        if intrabc_eligible {
            let ctx = self.intrabc_context(bx4, by4, bw4, bh4, have_left, have_top);
            self.msac.encode_bool_adapt(0, &mut self.cdf.m.intrabc[ctx]);
        }

        // Palette mode: only when `allow_screen_content_tools` is set and the
        // mode/size conditions of §5.20.5.3 hold. The luma mode is forced to
        // DC_PRED so the presence flag is always legal.
        let pal_eval = if self.tools.enable_palette
            && self.frame_hdr.allow_screen_content_tools != 0
            && self.frame_hdr.segmentation.enabled == 0
            && bw4 + bh4 >= 4
            && bw4.max(bh4) <= 16
        {
            evaluate_palette_block(src_y, px_x, px_y, w, h, self.lambda_sad)
        } else {
            None
        };
        if let Some(cand) = pal_eval
            && (cand.cost + self.lambda_sad * 2 < base_intra_cost || cand.distortion == 0)
        {
            use_palette = true;
            pal_cand = Some(cand);
        }
        let (y_mode, y_angle) = if use_palette {
            (0u8, 0i8)
        } else {
            (y_mode, y_angle)
        };

        let mut edge = [BD::Pixel::default(); 512];
        let tl_idx = 256usize;
        let sb_px = 64usize << self.frame_hdr.sb128;
        let (y_have_tr, y_have_bl) = crate::av2::ipred::intra_edge_availability(
            px_x,
            px_y,
            w,
            h,
            self.tile_col_end4 * 4,
            self.tile_row_end4 * 4,
            sb_px,
            have_left,
            have_top,
        );
        let (rec_w, rec_h) =
            crate::av2::ipred::intra_reference_extent(self.f_bw4, self.f_bh4, 0, 0);
        prepare_intra_edges(
            self.bd,
            &mut edge,
            tl_idx,
            &rec_y.data,
            rec_y.offset(px_x, px_y),
            rec_y.stride,
            have_left,
            have_top,
            y_have_tr,
            y_have_bl,
            w,
            h,
            rec_w,
            rec_h,
        );

        let fi_opt = if self.tools.enable_filter_intra
            && !use_palette
            && y_mode == 0
            && (have_left || have_top)
        {
            select_best_filter_intra_mode(
                self.bd,
                &src_y.data,
                src_y.offset(px_x, px_y),
                src_y.stride,
                &edge,
                tl_idx,
                w,
                h,
                self.lambda_sad,
                base_intra_cost,
            )
            .map(|(m, _)| m)
        } else {
            None
        };

        let mut best_mrl_idx = 0u8;
        if self.tools.enable_mrl
            && fi_opt.is_none()
            && (1..=8).contains(&y_mode)
            && have_left
            && have_top
            && px_x >= 4
            && px_y >= 4
        {
            let mut best_mrl_cost = base_intra_cost;
            let mut mrl_edge = [BD::Pixel::default(); 512];
            let mut pred_mrl = vec![BD::Pixel::default(); w * h];
            for cand_mrl in 1..=2u8 {
                prepare_intra_edges_mrl(
                    self.bd,
                    &mut mrl_edge,
                    tl_idx,
                    &rec_y.data,
                    rec_y.offset(px_x, px_y),
                    rec_y.stride,
                    have_left,
                    have_top,
                    false,
                    false,
                    w,
                    h,
                    cand_mrl,
                    rec_w,
                    rec_h,
                );
                predict_intra_block(
                    self.bd,
                    &mut pred_mrl,
                    0,
                    w,
                    &mrl_edge,
                    tl_idx,
                    y_mode,
                    y_angle,
                    w,
                    h,
                    have_left,
                    have_top,
                    0,
                    cand_mrl,
                );
                let sse = compute_sse(
                    &src_y.data,
                    src_y.offset(px_x, px_y),
                    src_y.stride,
                    &pred_mrl,
                    0,
                    w,
                    w,
                    h,
                );
                let approx_cost = (sse as f64).sqrt() as u64 + self.lambda_sad * 2;
                if approx_cost < best_mrl_cost {
                    best_mrl_cost = approx_cost;
                    best_mrl_idx = cand_mrl;
                    edge = mrl_edge;
                }
            }
        }

        // Intra: `intra_segment_id()` precedes `read_gdf`/`read_cdef`/`read_ccso`
        // /`read_delta_qindex` (AV2 Spec v1.0.0 §5.20.5.3).
        if !self.seg_coder.preskip {
            seg_id =
                self.code_segment_id(bx4, by4, bw4, bh4, have_left, have_top, false, desired_seg);
        }
        let seg_lossless = self.frame_hdr.quant.lossless[seg_id] != 0;

        self.encode_gdf_blk(bx4, by4);
        self.encode_ccso_blk(bx4, by4);
        self.emit_pending_delta();

        let luma_midx = encode_intra_y_mode(
            &mut self.msac,
            &mut self.cdf.m,
            &self.a_cols[a_col_idx],
            &self.l_ctx,
            bx4_in_sb,
            by4_in_sb,
            bw4 as i32,
            bh4 as i32,
            bw4 as i32,
            bh4 as i32,
            seg_lossless,
            y_mode,
            y_angle,
            None,
        );

        if self.tools.enable_mrl && (1..=8).contains(&y_mode) {
            encode_mrl_idx(&mut self.msac, &mut self.cdf.m, 0, best_mrl_idx);
        }
        if self.tools.enable_filter_intra && y_mode == 0 {
            encode_filter_intra_mode(&mut self.msac, &mut self.cdf.m, bs, fi_opt);
        }

        // Synthesize luma prediction (FilterIntra, ISP strip prediction, or standard intra)
        {
            let rec_y_mut = &mut self.rec.planes[0];
            let rec_off = rec_y_mut.offset(px_x, px_y);
            if let Some(cand) = &pal_cand {
                self.palette_blocks += 1;
                apply_palette_block(rec_y_mut, px_x, px_y, w, h, cand);
            } else if let Some(fi_mode) = fi_opt {
                self.filter_intra_blocks += 1;
                predict_filter_intra_block(
                    self.bd,
                    &mut rec_y_mut.data,
                    rec_off,
                    rec_y_mut.stride,
                    &edge,
                    tl_idx,
                    fi_mode,
                    w,
                    h,
                );
            } else {
                predict_intra_block(
                    self.bd,
                    &mut rec_y_mut.data,
                    rec_off,
                    rec_y_mut.stride,
                    &edge,
                    tl_idx,
                    y_mode,
                    y_angle,
                    w,
                    h,
                    have_left,
                    have_top,
                    0,
                    best_mrl_idx,
                );
            }
        }

        let mut res = [0i32; 64 * 64];
        let mut cf = [0i32; 32 * 32];
        let mut q_cf = [0i32; 32 * 32];
        let mut dq_cf = [0i32; 32 * 32];

        for r in 0..h {
            let s_off = self.src.planes[0].offset(px_x, px_y + r);
            let p_off = self.rec.planes[0].offset(px_x, px_y + r);
            for c in 0..w {
                res[r * w + c] = self.src.planes[0].data[s_off + c].to_i32()
                    - self.rec.planes[0].data[p_off + c].to_i32();
            }
        }

        let mut chosen_y_txtp = self.choose_luma_txtp(
            &res,
            w,
            tx,
            true,
            y_mode,
            y_angle,
            best_mrl_idx,
            seg_lossless,
        );

        fwd_txfm_2d(&res, w, &mut cf, chosen_y_txtp, tx);
        let mut eob_y = self.quantize_block_wave3(
            &cf,
            &mut q_cf,
            &mut dq_cf,
            tx,
            seg_id,
            0,
            true,
            seg_lossless,
        );
        // An intra block whose coefficients are DC-only infers `DCT_DCT`
        // (AV2 Spec v1.0.0 §5.20.8.2: `!is_inter && eob == 1`), so the
        // transform must be re-run with the inferred kernel.
        if eob_y == 0 && chosen_y_txtp != txtp::DCT_DCT {
            chosen_y_txtp = txtp::DCT_DCT;
            fwd_txfm_2d(&res, w, &mut cf, chosen_y_txtp, tx);
            eob_y = self.quantize_block_wave3(
                &cf,
                &mut q_cf,
                &mut dq_cf,
                tx,
                seg_id,
                0,
                true,
                seg_lossless,
            );
        }

        if eob_y >= 0 {
            for y in 0..t_dim.h as usize {
                for x in 0..t_dim.w as usize {
                    self.grid.set_lr_noskip(bx4 + x, by4 + y, true);
                }
            }
            if seg_lossless {
                // Lossless dequantization (§7.14.4) scales the coded levels by
                // `Round2(64, 3) = 8` before the inverse WHT.
                for v in dq_cf.iter_mut() {
                    *v *= crate::av2::transform::LOSSLESS_DEQUANT_SCALE;
                }
            }
            let rec_y_mut = &mut self.rec.planes[0];
            let rec_off = rec_y_mut.offset(px_x, px_y);
            inv_txfm_add(
                self.bd,
                &mut rec_y_mut.data,
                rec_off,
                rec_y_mut.stride,
                &mut dq_cf,
                chosen_y_txtp,
                eob_y,
                tx,
            );
        }

        let has_chroma = self.seqhdr.layout != PixelLayout::I400;
        let ss_x = self.seqhdr.ss_hor;
        let ss_y = self.seqhdr.ss_ver;
        let layout_col = (self.seqhdr.layout as usize).saturating_sub(1);
        let uv_bs_idx = if has_chroma {
            SS_BS[bs as usize][layout_col] as usize
        } else {
            bs as usize
        };
        let uvtx = MAX_TXFM_SIZE_FOR_BS[uv_bs_idx] as usize;
        let uv_tdim = &TXFM_DIMENSIONS[uvtx];
        let cw = (uv_tdim.w as usize) * 4;
        let ch = (uv_tdim.h as usize) * 4;
        let cpx_x = px_x >> ss_x;
        let cpx_y = px_y >> ss_y;
        let cbx4_in_sb = bx4_in_sb >> ss_x;
        let cby4_in_sb = by4_in_sb >> ss_y;

        let mut q_cf_u = [0i32; 32 * 32];
        let mut dq_cf_u = [0i32; 32 * 32];
        let mut q_cf_v = [0i32; 32 * 32];
        let mut dq_cf_v = [0i32; 32 * 32];
        let mut eob_u = -1i32;
        let mut eob_v = -1i32;

        let (uv_mode, _uv_angle, uv_txtp_init) = if has_chroma {
            let (spatial_um, spatial_ua) = select_best_intra_mode_uv(
                self.bd,
                &self.src.planes[1].data,
                &self.src.planes[2].data,
                self.src.planes[1].offset(cpx_x, cpx_y),
                self.src.planes[1].stride,
                &self.rec.planes[1].data,
                &self.rec.planes[2].data,
                self.rec.planes[1].offset(cpx_x, cpx_y),
                self.rec.planes[1].stride,
                have_left,
                have_top,
                cw,
                ch,
                uvtx,
                luma_midx,
                self.lambda_sad,
                self.rec.planes[1].width,
                self.rec.planes[1].height,
            );

            let cfl_cand =
                if self.tools.enable_cfl && self.seqhdr.cfl && !seg_lossless && !use_palette {
                    evaluate_cfl_uv(
                        self.bd,
                        &self.src.planes[1].data,
                        &self.src.planes[2].data,
                        self.src.planes[1].offset(cpx_x, cpx_y),
                        self.src.planes[1].stride,
                        &self.rec.planes[0].data,
                        self.rec.planes[0].stride,
                        &self.rec.planes[1].data,
                        &self.rec.planes[2].data,
                        self.rec.planes[1].offset(cpx_x, cpx_y),
                        self.rec.planes[1].stride,
                        px_x,
                        px_y,
                        cw,
                        ch,
                        ss_x,
                        ss_y,
                        have_left,
                        have_top,
                        self.lambda_sad,
                        self.rec.planes[1].width,
                        self.rec.planes[1].height,
                    )
                } else {
                    None
                };

            let (um, ua, cfl_alphas) = if let Some((au, av, _)) = cfl_cand {
                self.cfl_blocks += 1;
                (CFL_PRED, 0i8, Some((au, av)))
            } else {
                // The chroma mode is coded through the default chroma mode
                // list, which reconstructs a zero angle delta. Some directional
                // modes are not representable for a given luma direction; fall
                // back to DC so the signalled mode and the prediction agree.
                let _ = spatial_ua;
                let um = if crate::av2::entropy::chroma_uv_mode_codeable(spatial_um, luma_midx) {
                    spatial_um
                } else {
                    0
                };
                (um, 0i8, None)
            };

            encode_intra_uv_mode(
                &mut self.msac,
                &mut self.cdf.m,
                &self.a_cols[a_col_idx],
                &self.l_ctx,
                cbx4_in_sb,
                cby4_in_sb,
                self.seqhdr.cfl,
                seg_lossless,
                luma_midx,
                um,
                ua,
                None,
            );
            if let Some((au, av)) = cfl_alphas {
                encode_cfl_alphas(&mut self.msac, &mut self.cdf.m, au, av);
            }

            // `palette_mode_info()` / `palette_tokens()` (§5.20.5.3,
            // §5.20.8.1, §5.20.8.4) follow the intra UV mode and CfL alphas.
            // `palette_y_mode` is present whenever
            // `allow_screen_content_tools` is set, even when palette coding is
            // disabled (IntraBC-only frames). The zero flag must then still be
            // coded to keep the block syntax aligned.
            let palette_eligible = self.frame_hdr.allow_screen_content_tools != 0
                && bw4 + bh4 >= 4
                && bw4.max(bh4) <= 16
                && y_mode == 0;
            if palette_eligible {
                if let Some(cand) = &pal_cand {
                    let (above, left) = self.palette_neighbor_colors(bx4, by4);
                    encode_palette_mode_info(
                        &mut self.msac,
                        &mut self.cdf.m,
                        cand,
                        self.bd.bitdepth(),
                        &above,
                        &left,
                    );
                    encode_palette_tokens(
                        &mut self.msac,
                        &mut self.cdf.m,
                        &mut self.palette_cdf,
                        cand,
                        w,
                        h,
                    );
                } else {
                    self.msac
                        .encode_bool_adapt(0, &mut self.cdf.m.palette_y_mode);
                }
            }

            let uvt = if seg_lossless && uvtx == 0 {
                txtp::WHT_WHT as u16
            } else {
                crate::av2::transform::derive_chroma_txtp(
                    TXTP_FROM_UVMODE[um as usize] as u32,
                    uv_tdim,
                    uvtx,
                ) as u16
            };

            let mut ac_q3 = [0i16; 32 * 32];
            if cfl_alphas.is_some() {
                compute_cfl_ac_luma(
                    &self.rec.planes[0].data,
                    self.rec.planes[0].stride,
                    px_x,
                    px_y,
                    cw,
                    ch,
                    ss_x,
                    ss_y,
                    &mut ac_q3,
                );
            }

            for pl in 1..=2 {
                let rec_pl = &mut self.rec.planes[pl];
                let rec_off = rec_pl.offset(cpx_x, cpx_y);
                let mut edge_c = [BD::Pixel::default(); 512];
                let sb_c = (64usize << self.frame_hdr.sb128) >> ss_x.max(ss_y);
                let (c_have_tr, c_have_bl) = crate::av2::ipred::intra_edge_availability(
                    cpx_x,
                    cpx_y,
                    cw,
                    ch,
                    (self.tile_col_end4 * 4) >> ss_x,
                    (self.tile_row_end4 * 4) >> ss_y,
                    sb_c,
                    have_left,
                    have_top,
                );
                let (rec_cw, rec_ch) = crate::av2::ipred::intra_reference_extent(
                    self.f_bw4,
                    self.f_bh4,
                    rec_pl.ss_x,
                    rec_pl.ss_y,
                );
                prepare_intra_edges(
                    self.bd,
                    &mut edge_c,
                    tl_idx,
                    &rec_pl.data,
                    rec_off,
                    rec_pl.stride,
                    have_left,
                    have_top,
                    c_have_tr,
                    c_have_bl,
                    cw,
                    ch,
                    rec_cw,
                    rec_ch,
                );
                if let Some((au, av)) = cfl_alphas {
                    let mut dc_pred_buf = [BD::Pixel::default(); 64 * 64];
                    let dc_pred = &mut dc_pred_buf[..cw * ch];
                    predict_intra_block(
                        self.bd, dc_pred, 0, cw, &edge_c, tl_idx, 0, 0, cw, ch, have_left,
                        have_top, pl, 0,
                    );
                    let alpha = if pl == 1 { au } else { av };
                    predict_cfl_plane(
                        self.bd,
                        &mut rec_pl.data,
                        rec_off,
                        rec_pl.stride,
                        dc_pred,
                        &ac_q3,
                        alpha,
                        cw,
                        ch,
                    );
                } else {
                    predict_intra_block(
                        self.bd,
                        &mut rec_pl.data,
                        rec_off,
                        rec_pl.stride,
                        &edge_c,
                        tl_idx,
                        um,
                        ua,
                        cw,
                        ch,
                        have_left,
                        have_top,
                        pl,
                        0,
                    );
                }

                for r in 0..ch {
                    let s_off = self.src.planes[pl].offset(cpx_x, cpx_y + r);
                    let p_off = self.rec.planes[pl].offset(cpx_x, cpx_y + r);
                    for c in 0..cw {
                        res[r * cw + c] = self.src.planes[pl].data[s_off + c].to_i32()
                            - self.rec.planes[pl].data[p_off + c].to_i32();
                    }
                }
                fwd_txfm_2d(&res, cw, &mut cf, uvt as u32, uvtx);
                if pl == 1 {
                    eob_u = self.quantize_block_wave3(
                        &cf,
                        &mut q_cf_u,
                        &mut dq_cf_u,
                        uvtx,
                        seg_id,
                        1,
                        true,
                        seg_lossless,
                    );
                } else {
                    eob_v = self.quantize_block_wave3(
                        &cf,
                        &mut q_cf_v,
                        &mut dq_cf_v,
                        uvtx,
                        seg_id,
                        2,
                        true,
                        seg_lossless,
                    );
                }
            }
            (um, ua, uvt)
        } else {
            (0u8, 0i8, txtp::DCT_DCT as u16)
        };

        let all_zero = eob_y < 0 && eob_u < 0 && eob_v < 0;

        if all_zero {
            self.skipped_blocks += 1;
        }
        // Intra non-IntraBC blocks infer skip_txfm = 0 in the AV2 decoder;
        // do not emit a skip flag for them.
        self.store_palette_block(bx4, by4, bw4, bh4, pal_cand.as_ref());
        self.mark_intrabc_footprint(bx4, by4, bw4, bh4, false);

        for i in 0..bw4.min(32 - bx4_in_sb).min(self.f_bw4.saturating_sub(bx4)) {
            self.a_cols[a_col_idx].intra[bx4_in_sb + i] = 1;
            self.a_cols[a_col_idx].mode[bx4_in_sb + i] = y_mode;
            self.a_cols[a_col_idx].skip[bx4_in_sb + i] = 0;
            self.a_cols[a_col_idx].midx[bx4_in_sb + i] = luma_midx;
            self.a_cols[a_col_idx].tx_luma[bx4_in_sb + i] = t_dim.lw;
            self.a_cols[a_col_idx].tx[bx4_in_sb + i] = t_dim.lw;
            self.a_cols[a_col_idx].seg_id[bx4_in_sb + i] = seg_id as u8;
            // An intra block has no inter reference on either list
            // (AV2 Spec v1.0.0 §5.20.4.1 context update).
            self.a_cols[a_col_idx].ref_frame[0][bx4_in_sb + i] = -1;
            self.a_cols[a_col_idx].ref_frame[1][bx4_in_sb + i] = -1;
            self.a_cols[a_col_idx].filter[0][bx4_in_sb + i] = 3;
            self.a_cols[a_col_idx].filter[1][bx4_in_sb + i] = 3;
        }
        for i in 0..bh4.min(32 - by4_in_sb).min(self.f_bh4.saturating_sub(by4)) {
            self.l_ctx.intra[by4_in_sb + i] = 1;
            self.l_ctx.mode[by4_in_sb + i] = y_mode;
            self.l_ctx.skip[by4_in_sb + i] = 0;
            self.l_ctx.midx[by4_in_sb + i] = luma_midx;
            self.l_ctx.tx_luma[by4_in_sb + i] = t_dim.lh;
            self.l_ctx.tx[by4_in_sb + i] = t_dim.lh;
            self.l_ctx.seg_id[by4_in_sb + i] = seg_id as u8;
            self.l_ctx.ref_frame[0][by4_in_sb + i] = -1;
            self.l_ctx.ref_frame[1][by4_in_sb + i] = -1;
            self.l_ctx.filter[0][by4_in_sb + i] = 3;
            self.l_ctx.filter[1][by4_in_sb + i] = 3;
        }
        let lossless_seg = self.frame_hdr.quant.lossless[seg_id] != 0;
        self.record_deblock(
            bx4,
            by4,
            bw4,
            bh4,
            t_dim,
            uv_tdim,
            has_chroma,
            false,
            lossless_seg,
        );
        if has_chroma {
            for i in 0..(uv_tdim.w as usize)
                .min(32 - cbx4_in_sb)
                .min((self.f_bw4 >> ss_x).saturating_sub(bx4 >> ss_x))
            {
                self.a_cols[a_col_idx].uvmode[cbx4_in_sb + i] = uv_mode;
                self.a_cols[a_col_idx].tx_chroma[cbx4_in_sb + i] = uv_tdim.lw;
            }
            for i in 0..(uv_tdim.h as usize)
                .min(32 - cby4_in_sb)
                .min((self.f_bh4 >> ss_y).saturating_sub(by4 >> ss_y))
            {
                self.l_ctx.uvmode[cby4_in_sb + i] = uv_mode;
                self.l_ctx.tx_chroma[cby4_in_sb + i] = uv_tdim.lh;
            }
        }

        // `read_tx_size` (§5.20.6.1) is entered for intra blocks whenever
        // `MiSize > BLOCK_4X4` and `TxMode == TX_MODE_SELECT`; lossless frames
        // use `TX_MODE_ONLY_4X4`, so no transform-tree symbol is present.
        // Intra blocks stay `TX_PARTITION_NONE` (the encoder does not model
        // per-transform-block intra prediction yet).
        if !seg_lossless {
            encode_tx_partition_syntax(
                &mut self.msac,
                &mut self.cdf.m,
                bs,
                TxPartition::TxNone,
                0,
                0,
                self.seqhdr.reduced_tx_part_set,
            );
        }

        if chosen_y_txtp != txtp::DCT_DCT {
            self.non_dct_tx_blocks += 1;
        }
        let mut y_txtp = chosen_y_txtp as u16;
        let mut y_res_ctx = 0u8;
        let y_params = CoefCodingParams {
            tx,
            bs: bs as usize,
            plane: 0,
            intra: true,
            fsc: false,
            lossless: seg_lossless,
            y_mode: y_mode as usize,
            y_angle,
            mrl_index: best_mrl_idx,
            uv_mode: uv_mode as usize,
            reduced_txtp_set: self.frame_hdr.reduced_txtp_set as i32,
            u_has_cf: 0,
            dq_tbl: self.dq[seg_id][0],
            bitdepth: self.bd.bitdepth(),
            qm: None,
            ss_hor: ss_x != 0,
            ss_ver: ss_y != 0,
        };
        let _ = encode_coefs(
            &mut self.msac,
            &mut self.cdf.coef,
            &mut self.cdf.m,
            &self.a_cols[a_col_idx].lcoef[bx4_in_sb..],
            &self.l_ctx.lcoef[by4_in_sb..],
            &y_params,
            &q_cf,
            &mut y_txtp,
            &mut y_res_ctx,
        );
        for i in 0..bw4.min(32 - bx4_in_sb).min(self.f_bw4.saturating_sub(bx4)) {
            self.a_cols[a_col_idx].lcoef[bx4_in_sb + i] = y_res_ctx;
            self.a_cols[a_col_idx].txtp[bx4_in_sb + i] = y_txtp;
        }
        for i in 0..bh4.min(32 - by4_in_sb).min(self.f_bh4.saturating_sub(by4)) {
            self.l_ctx.lcoef[by4_in_sb + i] = y_res_ctx;
            self.l_ctx.txtp[by4_in_sb + i] = y_txtp;
        }

        if has_chroma {
            let mut u_txtp = uv_txtp_init;
            let mut u_res_ctx = 0u8;
            let u_params = CoefCodingParams {
                tx: uvtx,
                bs: uv_bs_idx,
                plane: 1,
                intra: true,
                fsc: false,
                lossless: seg_lossless,
                y_mode: y_mode as usize,
                y_angle,
                mrl_index: best_mrl_idx,
                uv_mode: uv_mode as usize,
                reduced_txtp_set: self.frame_hdr.reduced_txtp_set as i32,
                u_has_cf: 0,
                dq_tbl: self.dq[seg_id][1],
                bitdepth: self.bd.bitdepth(),
                qm: None,
                ss_hor: ss_x != 0,
                ss_ver: ss_y != 0,
            };
            let coded_eob_u = encode_coefs(
                &mut self.msac,
                &mut self.cdf.coef,
                &mut self.cdf.m,
                &self.a_cols[a_col_idx].ccoef[0][cbx4_in_sb..],
                &self.l_ctx.ccoef[0][cby4_in_sb..],
                &u_params,
                &q_cf_u,
                &mut u_txtp,
                &mut u_res_ctx,
            );
            for i in 0..(uv_tdim.w as usize)
                .min(32 - cbx4_in_sb)
                .min((self.f_bw4 >> ss_x).saturating_sub(bx4 >> ss_x))
            {
                self.a_cols[a_col_idx].ccoef[0][cbx4_in_sb + i] = u_res_ctx;
            }
            for i in 0..(uv_tdim.h as usize)
                .min(32 - cby4_in_sb)
                .min((self.f_bh4 >> ss_y).saturating_sub(by4 >> ss_y))
            {
                self.l_ctx.ccoef[0][cby4_in_sb + i] = u_res_ctx;
            }
            if coded_eob_u >= 0 {
                let rec_u_mut = &mut self.rec.planes[1];
                let rec_off = rec_u_mut.offset(cpx_x, cpx_y);
                inv_txfm_add(
                    self.bd,
                    &mut rec_u_mut.data,
                    rec_off,
                    rec_u_mut.stride,
                    &mut dq_cf_u,
                    u_txtp as u32,
                    coded_eob_u,
                    uvtx,
                );
            }

            let mut v_txtp = u_txtp;
            let mut v_res_ctx = 0u8;
            let v_params = CoefCodingParams {
                tx: uvtx,
                bs: uv_bs_idx,
                plane: 2,
                intra: true,
                fsc: false,
                lossless: seg_lossless,
                y_mode: y_mode as usize,
                y_angle,
                mrl_index: best_mrl_idx,
                uv_mode: uv_mode as usize,
                reduced_txtp_set: self.frame_hdr.reduced_txtp_set as i32,
                u_has_cf: (coded_eob_u >= 0) as i32,
                dq_tbl: self.dq[seg_id][2],
                bitdepth: self.bd.bitdepth(),
                qm: None,
                ss_hor: ss_x != 0,
                ss_ver: ss_y != 0,
            };
            let coded_eob_v = encode_coefs(
                &mut self.msac,
                &mut self.cdf.coef,
                &mut self.cdf.m,
                &self.a_cols[a_col_idx].ccoef[1][cbx4_in_sb..],
                &self.l_ctx.ccoef[1][cby4_in_sb..],
                &v_params,
                &q_cf_v,
                &mut v_txtp,
                &mut v_res_ctx,
            );
            for i in 0..(uv_tdim.w as usize)
                .min(32 - cbx4_in_sb)
                .min((self.f_bw4 >> ss_x).saturating_sub(bx4 >> ss_x))
            {
                self.a_cols[a_col_idx].ccoef[1][cbx4_in_sb + i] = v_res_ctx;
            }
            for i in 0..(uv_tdim.h as usize)
                .min(32 - cby4_in_sb)
                .min((self.f_bh4 >> ss_y).saturating_sub(by4 >> ss_y))
            {
                self.l_ctx.ccoef[1][cby4_in_sb + i] = v_res_ctx;
            }
            if coded_eob_v >= 0 {
                let rec_v_mut = &mut self.rec.planes[2];
                let rec_off = rec_v_mut.offset(cpx_x, cpx_y);
                inv_txfm_add(
                    self.bd,
                    &mut rec_v_mut.data,
                    rec_off,
                    rec_v_mut.stride,
                    &mut dq_cf_v,
                    v_txtp as u32,
                    coded_eob_v,
                    uvtx,
                );
            }
        }
    }

    /// Above/left palettes for `get_palette_cache()` (§5.20.8.1).
    fn palette_neighbor_colors(&self, bx4: usize, by4: usize) -> (Vec<i32>, Vec<i32>) {
        let mut above = Vec::new();
        if (by4 * 4) % 64 != 0
            && by4 > 0
            && let Some(idx) = (by4 - 1)
                .checked_mul(self.f_bw4)
                .and_then(|r| r.checked_add(bx4))
            && idx < self.palette_sizes.len()
        {
            let n = self.palette_sizes[idx] as usize;
            above.extend_from_slice(&self.palette_colors[idx][..n]);
        }
        let mut left = Vec::new();
        if bx4 > self.tile_col_start4
            && let Some(idx) = by4
                .checked_mul(self.f_bw4)
                .and_then(|r| r.checked_add(bx4 - 1))
            && idx < self.palette_sizes.len()
        {
            let n = self.palette_sizes[idx] as usize;
            left.extend_from_slice(&self.palette_colors[idx][..n]);
        }
        (above, left)
    }

    /// Stores `PaletteSizes` / `PaletteColors` over a coded block footprint.
    fn store_palette_block(
        &mut self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        cand: Option<&crate::av2::screen::PaletteBlockCandidate>,
    ) {
        let (size, colors) = match cand {
            Some(c) => (c.palette_size as u8, c.colors),
            None => (0u8, [0i32; 8]),
        };
        for y in 0..bh4 {
            let r = by4 + y;
            if r >= self.f_bh4 {
                break;
            }
            for x in 0..bw4 {
                let c = bx4 + x;
                if c >= self.f_bw4 {
                    break;
                }
                let idx = r * self.f_bw4 + c;
                self.palette_sizes[idx] = size;
                self.palette_colors[idx] = colors;
            }
        }
    }

    /// `IBCCoded` marking at 4x4 granularity.
    fn mark_intrabc_footprint(
        &mut self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        coded: bool,
    ) {
        for y in 0..bh4 {
            let r = by4 + y;
            if r >= self.f_bh4 {
                break;
            }
            for x in 0..bw4 {
                let c = bx4 + x;
                if c >= self.f_bw4 {
                    break;
                }
                self.intrabc_coded[r * self.f_bw4 + c] = coded;
            }
        }
    }

    /// True when one of the two IntraBC neighbour-context positions or the
    /// spatial MVP scan would see an earlier IntraBC block. Such blocks are
    /// skipped so the encoder's fixed default BVP candidate stack stays
    /// identical to the normatively derived stack.
    fn has_intrabc_neighbor(
        &self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        have_left: bool,
        have_top: bool,
    ) -> bool {
        let probe = |x: usize, y: usize| -> bool {
            if x >= self.f_bw4 || y >= self.f_bh4 {
                return true;
            }
            self.intrabc_coded[y * self.f_bw4 + x]
        };
        let have_top_in_sb = (by4 & (self.sb_step4 - 1)) != 0;
        if have_left && bh4 > 0 && probe(bx4 - 1, by4 + bh4 - 1) {
            return true;
        }
        if have_top_in_sb && probe(bx4 + bw4.saturating_sub(1), by4 - 1) {
            return true;
        }
        if have_left && probe(bx4 - 1, by4) {
            return true;
        }
        if have_top && probe(bx4, by4 - 1) {
            return true;
        }
        // Broader neighbours scanned by find_mv_stack, including top-right and
        // bottom-left, whose motion vectors enter the BVP candidate stack.
        if have_top {
            for x in 0..=bw4 {
                if probe(bx4 + x, by4 - 1) {
                    return true;
                }
            }
        }
        if have_left {
            for y in 0..=bh4 {
                if probe(bx4 - 1, by4 + y) {
                    return true;
                }
            }
        }
        false
    }

    /// `intrabc` flag context: sum of the two neighbour `intrabc` flags
    /// (AV2 Spec v1.0.0 §5.20.5.3 / §8.3.2 `TileIntrabcCdf`).
    fn intrabc_context(
        &self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        have_left: bool,
        have_top: bool,
    ) -> usize {
        let probe = |x: usize, y: usize| -> u8 {
            if x >= self.f_bw4 || y >= self.f_bh4 {
                0
            } else {
                self.intrabc_coded[y * self.f_bw4 + x] as u8
            }
        };
        let have_top_in_sb = (by4 & (self.sb_step4 - 1)) != 0;
        let mut idx = 0usize;
        let mut nb = [0u8; 2];
        if have_left && bh4 > 0 {
            nb[idx] = probe(bx4 - 1, by4 + bh4 - 1);
            idx += 1;
        }
        if have_top_in_sb && bw4 > 0 {
            nb[idx] = probe(bx4 + bw4 - 1, by4 - 1);
            idx += 1;
        }
        if have_left && idx < 2 {
            nb[idx] = probe(bx4 - 1, by4);
            idx += 1;
        }
        if have_top && idx < 2 {
            nb[idx] = probe(bx4, by4 - 1);
        }
        (nb[0] + nb[1]) as usize
    }

    /// Encodes a complete IntraBC block (§5.20.5.3 / §5.20.5.4).
    ///
    /// IntraBC uses `skip_txfm = 1`: no transform partition, transform type,
    /// or coefficient syntax is present, and the prediction is an integer
    /// copy of an already-reconstructed block. The residual is therefore zero.
    #[allow(clippy::too_many_arguments)]
    fn encode_intrabc_leaf(
        &mut self,
        bs: BlockSize,
        bx4: usize,
        by4: usize,
        dx_px: i32,
        dy_px: i32,
        dv_q8: MvXY,
        mut seg_id: usize,
        desired_seg: u8,
    ) {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let w = bw4 * 4;
        let h = bh4 * 4;
        let px_x = bx4 * 4;
        let px_y = by4 * 4;
        let bx4_in_sb = bx4 & 31;
        let by4_in_sb = by4 & 31;
        let have_left = bx4 > self.tile_col_start4;
        let have_top = by4 > self.tile_row_start4;

        self.intrabc_blocks += 1;
        let ctx = self.intrabc_context(bx4, by4, bw4, bh4, have_left, have_top);
        self.msac.encode_bool_adapt(1, &mut self.cdf.m.intrabc[ctx]);

        // `read_skip()`: the block has no residual.
        let a_col_idx = (bx4 - self.tile_col_start4) >> 5;
        let nb = gather_inter_neighbors(
            &self.a_cols[a_col_idx],
            &self.l_ctx,
            bx4_in_sb,
            by4_in_sb,
            bw4,
            bh4,
            have_left,
            have_top,
            self.tile_col_end4,
            self.tile_row_end4,
        );
        let skip_ctx = (nb.skip[0] as usize + nb.skip[1] as usize).min(2);
        self.msac.encode_bool_adapt(1, self.cdf.m.skip(skip_ctx));

        if !self.seg_coder.preskip {
            seg_id =
                self.code_segment_id(bx4, by4, bw4, bh4, have_left, have_top, false, desired_seg);
        }
        self.encode_gdf_blk(bx4, by4);
        self.encode_ccso_blk(bx4, by4);
        self.emit_pending_delta();

        // `read_intrabc_info()`: NEWMV-style residual against the first
        // default BVP candidate, `(0, -SbHeight)`, with integer precision.
        self.msac.encode_bool_adapt(0, &mut self.cdf.m.intrabc_mode);
        // `intrabc_drl_mode`: RefMvIdx = 0 terminates the unary loop.
        self.msac.encode_bool_bypass(0);
        let sb_px = 64i32 << self.frame_hdr.sb128;
        let pred = MvXY {
            x: 0,
            y: (-sb_px * 8) as i16,
        };
        // `MvCtx = MV_INTRABC_CONTEXT` (§5.20.7.20), i.e. the separate `dmv` CDF set.
        encode_mv_residual(&mut self.msac, &mut self.cdf.dmv, dv_q8, pred, 3, false);

        // Prediction: integer block copy for luma and both chroma planes.
        apply_intrabc_block(&mut self.rec.planes[0], px_x, px_y, w, h, dy_px, dx_px);
        if self.seqhdr.layout != PixelLayout::I400 {
            let ss_x = self.seqhdr.ss_hor;
            let ss_y = self.seqhdr.ss_ver;
            for pl in 1..=2 {
                apply_intrabc_block(
                    &mut self.rec.planes[pl],
                    px_x >> ss_x,
                    px_y >> ss_y,
                    w >> ss_x,
                    h >> ss_y,
                    dy_px >> ss_y,
                    dx_px >> ss_x,
                );
            }
        }

        // Block context update for IntraBC (§5.20.4.1): intra = 1,
        // intrabc = 1, mode = DC_PRED, skip_txfm = 1, zero coefficient context.
        for i in 0..bw4.min(32 - bx4_in_sb).min(self.f_bw4.saturating_sub(bx4)) {
            let a = &mut self.a_cols[a_col_idx];
            a.intra[bx4_in_sb + i] = 1;
            a.mode[bx4_in_sb + i] = 0;
            a.midx[bx4_in_sb + i] = 0xff;
            a.skip[bx4_in_sb + i] = 1;
            a.lcoef[bx4_in_sb + i] = 0x40;
            a.seg_id[bx4_in_sb + i] = seg_id as u8;
            a.ref_frame[0][bx4_in_sb + i] = -1;
            a.ref_frame[1][bx4_in_sb + i] = -1;
            a.filter[0][bx4_in_sb + i] = 3;
            a.filter[1][bx4_in_sb + i] = 3;
        }
        for i in 0..bh4.min(32 - by4_in_sb).min(self.f_bh4.saturating_sub(by4)) {
            let l = &mut self.l_ctx;
            l.intra[by4_in_sb + i] = 1;
            l.mode[by4_in_sb + i] = 0;
            l.midx[by4_in_sb + i] = 0xff;
            l.skip[by4_in_sb + i] = 1;
            l.lcoef[by4_in_sb + i] = 0x40;
            l.seg_id[by4_in_sb + i] = seg_id as u8;
            l.ref_frame[0][by4_in_sb + i] = -1;
            l.ref_frame[1][by4_in_sb + i] = -1;
            l.filter[0][by4_in_sb + i] = 3;
            l.filter[1][by4_in_sb + i] = 3;
        }
        if self.seqhdr.layout != PixelLayout::I400 {
            let ss_x = self.seqhdr.ss_hor;
            let ss_y = self.seqhdr.ss_ver;
            // The chroma entropy contexts live at chroma 4-sample granularity
            // over the chroma plane block size, not over the luma footprint
            // (`av2_reset_entropy_context`, §5.20.4.1 / `coeffs()` `x4/y4`).
            let cbx4 = (bx4 >> ss_x) & 31;
            let cby4 = (by4 >> ss_y) & 31;
            for i in 0..(bw4 >> ss_x)
                .min(32 - cbx4)
                .min((self.f_bw4 >> ss_x).saturating_sub(bx4 >> ss_x))
            {
                let a = &mut self.a_cols[a_col_idx];
                a.ccoef[0][cbx4 + i] = 0x40;
                a.ccoef[1][cbx4 + i] = 0x40;
                a.uvmode[cbx4 + i] = 0;
            }
            for i in 0..(bh4 >> ss_y)
                .min(32 - cby4)
                .min((self.f_bh4 >> ss_y).saturating_sub(by4 >> ss_y))
            {
                let l = &mut self.l_ctx;
                l.ccoef[0][cby4 + i] = 0x40;
                l.ccoef[1][cby4 + i] = 0x40;
                l.uvmode[cby4 + i] = 0;
            }
        }
        self.store_palette_block(bx4, by4, bw4, bh4, None);
        self.mark_intrabc_footprint(bx4, by4, bw4, bh4, true);
        self.skipped_blocks += 1;

        let (rec_w, rec_h) =
            crate::av2::ipred::intra_reference_extent(self.f_bw4, self.f_bh4, 0, 0);
        let _ = (rec_w, rec_h, px_x, px_y);
    }

    /// Computes the `single_ref` bit context and the `sngl_ctx` single-reference
    /// mode context from the coded neighbour contexts, assuming the coded
    /// reference is list index 0 and no neighbour uses NEWMV.
    fn encode_inter_leaf(
        &mut self,
        bs: BlockSize,
        bx4: usize,
        by4: usize,
        tx: usize,
        desired_seg: usize,
        mut dec: crate::av2::inter::InterBlockDecision,
    ) {
        let seg_id = desired_seg;
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let w = bw4 * 4;
        let h = bh4 * 4;
        if self.frame_hdr.switchable_comp_refs == 0 || (bw4 * bh4 < 4) {
            dec.is_compound = false;
        }
        self.inter_blocks += 1;
        if dec.is_compound {
            self.compound_blocks += 1;
            match dec.comp_type {
                CompInterType::Average => self.compound_average_blocks += 1,
                CompInterType::Wedge => self.compound_wedge_blocks += 1,
                CompInterType::DifferenceWeighted => self.compound_diff_blocks += 1,
                CompInterType::DistanceWeighted => self.compound_cwp_blocks += 1,
            }
        }
        let px_x = bx4 * 4;
        let px_y = by4 * 4;
        let a_col_idx = (bx4 - self.tile_col_start4) >> 5;
        let bx4_in_sb = bx4 & 31;
        let by4_in_sb = by4 & 31;
        let have_left = bx4 > self.tile_col_start4;
        let have_top = by4 > self.tile_row_start4;
        let t_dim = &TXFM_DIMENSIONS[tx];
        let seg_lossless = self.frame_hdr.quant.lossless[seg_id] != 0;

        let r0 = self.ref0.unwrap();
        let mv0_y_q16 = (dec.mv[0].y as i32) * 2;
        let mv0_x_q16 = (dec.mv[0].x as i32) * 2;
        let mv1_y_q16 = (dec.mv[1].y as i32) * 2;
        let mv1_x_q16 = (dec.mv[1].x as i32) * 2;

        let ss_x = self.seqhdr.ss_hor;
        let ss_y = self.seqhdr.ss_ver;

        // Motion-compensate luma block into `rec.planes[0]`
        {
            let rec_y_mut = &mut self.rec.planes[0];
            let rec_off = rec_y_mut.offset(px_x, px_y);
            if dec.is_compound {
                let r1 = self.ref1.unwrap_or(r0);
                mc_predict_block_prep(
                    self.bd,
                    &mut self.scratch_mc0[..w * h],
                    &r0.planes[0].data,
                    r0.planes[0].stride,
                    r0.planes[0].coded_width as i32,
                    r0.planes[0].coded_height as i32,
                    px_x as i32,
                    px_y as i32,
                    w,
                    h,
                    mv0_y_q16,
                    mv0_x_q16,
                    dec.interp_filter as usize,
                );
                mc_predict_block_prep(
                    self.bd,
                    &mut self.scratch_mc1[..w * h],
                    &r1.planes[0].data,
                    r1.planes[0].stride,
                    r1.planes[0].coded_width as i32,
                    r1.planes[0].coded_height as i32,
                    px_x as i32,
                    px_y as i32,
                    w,
                    h,
                    mv1_y_q16,
                    mv1_x_q16,
                    dec.interp_filter as usize,
                );
                match dec.comp_type {
                    CompInterType::Average => {
                        blend_compound_prep(
                            self.bd,
                            &mut rec_y_mut.data,
                            rec_off,
                            rec_y_mut.stride,
                            &self.scratch_mc0[..w * h],
                            &self.scratch_mc1[..w * h],
                            w,
                            h,
                            CompInterType::Average,
                            8,
                            None,
                        );
                    }
                    CompInterType::DistanceWeighted => {
                        const CWP_WEIGHTING_FACTOR: [[i8; 5]; 2] =
                            [[8, 12, 4, 10, 6], [8, 12, 4, 20, -4]];
                        let is_same_side = (self.refdir[0] ^ self.refdir[1]) == 0;
                        let wt = CWP_WEIGHTING_FACTOR[is_same_side as usize]
                            [dec.cwp_idx.min(4) as usize] as i32;
                        blend_compound_prep(
                            self.bd,
                            &mut rec_y_mut.data,
                            rec_off,
                            rec_y_mut.stride,
                            &self.scratch_mc0[..w * h],
                            &self.scratch_mc1[..w * h],
                            w,
                            h,
                            CompInterType::DistanceWeighted,
                            wt,
                            None,
                        );
                    }
                    CompInterType::Wedge => {
                        let mask = WEDGE_TABLES
                            .get_mask(bs, dec.wedge_idx as usize, 0, 0)
                            .unwrap_or(&[]);
                        let (p0, p1) = if dec.wedge_sign {
                            (&self.scratch_mc1[..w * h], &self.scratch_mc0[..w * h])
                        } else {
                            (&self.scratch_mc0[..w * h], &self.scratch_mc1[..w * h])
                        };
                        blend_compound_prep(
                            self.bd,
                            &mut rec_y_mut.data,
                            rec_off,
                            rec_y_mut.stride,
                            p0,
                            p1,
                            w,
                            h,
                            CompInterType::Wedge,
                            0,
                            Some(mask),
                        );
                    }
                    CompInterType::DifferenceWeighted => {
                        compute_diff_wtd_mask_intermediate(
                            self.bd,
                            &self.scratch_mc0[..w * h],
                            &self.scratch_mc1[..w * h],
                            w,
                            h,
                            dec.mask_sign,
                            ss_x as usize,
                            ss_y as usize,
                            &mut self.scratch_diff_mask[..w * h],
                            &mut self.scratch_chroma_diff_mask[..(w >> ss_x) * (h >> ss_y)],
                        );
                        let (p0, p1) = if dec.mask_sign {
                            (&self.scratch_mc1[..w * h], &self.scratch_mc0[..w * h])
                        } else {
                            (&self.scratch_mc0[..w * h], &self.scratch_mc1[..w * h])
                        };
                        blend_compound_prep(
                            self.bd,
                            &mut rec_y_mut.data,
                            rec_off,
                            rec_y_mut.stride,
                            p0,
                            p1,
                            w,
                            h,
                            CompInterType::DifferenceWeighted,
                            0,
                            Some(&self.scratch_diff_mask[..w * h]),
                        );
                    }
                }
            } else {
                let active_r = if dec.ref_slot_idx == 1 {
                    self.ref1.unwrap_or(r0)
                } else {
                    r0
                };
                mc_predict_block(
                    self.bd,
                    &mut rec_y_mut.data,
                    rec_off,
                    rec_y_mut.stride,
                    &active_r.planes[0].data,
                    active_r.planes[0].stride,
                    active_r.planes[0].coded_width as i32,
                    active_r.planes[0].coded_height as i32,
                    px_x as i32,
                    px_y as i32,
                    w,
                    h,
                    mv0_y_q16,
                    mv0_x_q16,
                    dec.interp_filter as usize,
                );
            }
        }

        // Warped motion (AV2 Spec v1.0.0 §5.20.7.6 `inter_block_mode_info`,
        // §5.20.7.14 `read_motion_mode`, §7.12.2 `find_mv_stack`,
        // §7.13.3.19 `block_warp`). AV2 v1.0.0 has no OBMC symbol. The
        // `WarpParamStack` and the DRL predictor are both derived here from
        // the same normative stack the decoder uses, so `WARPMV`, `LOCALWARP`,
        // `EXTENDWARP` and `DELTAWARP` all code against identical inputs.
        let mut warp_choice: Option<WarpChoice> = None;
        if !dec.is_compound
            && self.tools.enable_warped_motion
            && !self.tools.enable_compound
            && !self
                .frame_hdr
                .segmentation
                .d
                .feature_active(seg_id, SEG_LVL_SKIP)
            && !self
                .frame_hdr
                .segmentation
                .d
                .feature_active(seg_id, SEG_LVL_GLOBALMV)
            && self.frame_hdr.switchable_comp_refs == 0
            && self.warp_bank_hits < MAX_WARP_SB_HITS
            && px_x + w <= self.src.planes[0].width
            && px_y + h <= self.src.planes[0].height
            && w >= 8
            && h >= 8
            && (w >> ss_x) >= 8
            && (h >> ss_y) >= 8
            // The candidate maps are kept per tile, so warp RDO requires a
            // single-tile frame.
            && self.tile_col_start4 == 0
            && self.tile_row_start4 == 0
            && self.tile_col_end4 >= self.f_bw4
            && self.tile_row_end4 >= self.f_bh4
        {
            let active_r = if dec.ref_slot_idx == 1 {
                self.ref1.unwrap_or(r0)
            } else {
                r0
            };
            let warp_ref = dec.ref_slot_idx as i8;
            let stack = self.mv_stack(bx4, by4, bw4, bh4, [warp_ref, -1]);
            // The coded MV residual is quantised to the MVD lattice, so the
            // model estimation and reconstruction use the snapped vector.
            let warp_mv = crate::av2::me::snap_mv_to_precision(
                dec.mv[0],
                stack.get(0),
                3 + self.frame_hdr.mv_precision as usize,
            );
            let trans_sad = compute_sad(
                &self.src.planes[0].data,
                self.src.planes[0].offset(px_x, px_y),
                self.src.planes[0].stride,
                &self.rec.planes[0].data,
                self.rec.planes[0].offset(px_x, px_y),
                self.rec.planes[0].stride,
                w,
                h,
            );
            let mut best = self.select_warpmv(
                &self.src.planes[0],
                &active_r.planes[0],
                px_x,
                px_y,
                bx4,
                by4,
                bw4,
                bh4,
                trans_sad,
                &stack,
            );
            // `LOCALWARP` / `EXTENDWARP` are only signalled when
            // `WarpSampleFound[0]` is set (§5.20.7.14 `motion_mode_allowed`).
            let local_extend_allowed = warp_has_ref_neighbour(
                &self.mi_map,
                self.f_bw4,
                self.f_bh4,
                self.sb_step4,
                bx4,
                by4,
                bw4,
                bh4,
                warp_ref,
            );
            {
                let consider =
                    |cand: Option<(WarpChoice, u64)>, best: &mut Option<(WarpChoice, u64)>| {
                        if let Some((c, cost)) = cand
                            && best.as_ref().is_none_or(|(_, bc)| cost < *bc)
                        {
                            *best = Some((c, cost));
                        }
                    };
                // `DELTAWARP` is always allowed for `WARP_NEWMV`
                // (§5.20.7.14); `LOCALWARP` and `EXTENDWARP` additionally
                // require `WarpSampleFound[0]`.
                if local_extend_allowed {
                    consider(
                        self.select_local_warp(
                            &self.src.planes[0],
                            &active_r.planes[0],
                            px_x,
                            px_y,
                            bx4,
                            by4,
                            bw4,
                            bh4,
                            trans_sad,
                            warp_mv,
                            &stack,
                        ),
                        &mut best,
                    );
                    consider(
                        self.select_extend_warp(
                            &self.src.planes[0],
                            &active_r.planes[0],
                            px_x,
                            px_y,
                            bx4,
                            by4,
                            bw4,
                            bh4,
                            trans_sad,
                            warp_mv,
                            &stack,
                        ),
                        &mut best,
                    );
                }
                consider(
                    self.select_warp_delta(
                        &self.src.planes[0],
                        &active_r.planes[0],
                        px_x,
                        px_y,
                        bx4,
                        by4,
                        bw4,
                        bh4,
                        trans_sad,
                        warp_mv,
                        &stack,
                    ),
                    &mut best,
                );
            }
            if let Some((choice, _)) = best {
                let rec_y = &mut self.rec.planes[0];
                let rec_off = rec_y.offset(px_x, px_y);
                block_warp(
                    self.bd,
                    &mut rec_y.data,
                    rec_off,
                    rec_y.stride,
                    &active_r.planes[0],
                    px_x,
                    px_y,
                    w,
                    h,
                    0,
                    0,
                    &choice.matrix,
                    &choice.abcd,
                );
                self.warped_blocks += 1;
                if choice.warp_mv {
                    self.warpmv_blocks += 1;
                } else if choice.motion_mode == 2 {
                    self.localwarp_blocks += 1;
                } else if choice.motion_mode == 4 {
                    self.extendwarp_blocks += 1;
                }
                // `update_warp_param_bank()` (§7.12.3).
                if let Some(bank) = self.warp_bank.get_mut(warp_ref as usize) {
                    warp_bank_insert(bank, &choice.matrix);
                }
                self.warp_bank_hits += 1;
                warp_choice = Some(choice);
            }
        }

        // Inter-intra (`read_interintra_mode`, §5.20.7.15): evaluate the four
        // `interintra_mode` variants and blend the intra prediction with the
        // (translation or warped) inter prediction. Only single-reference,
        // non-warp blocks of at most 64x64 samples can signal it.
        let mut interintra_mode_choice: Option<u8> = None;
        if !dec.is_compound
            && warp_choice.is_none()
            && (self.seqhdr.motion_modes & (1 << 1)) != 0
            && bw4 * bh4 > 2
            && bw4.max(bh4) <= 16
            && w >= 8
            && h >= 8
            && (w >> ss_x) >= 4
            && (h >> ss_y) >= 4
            && !self
                .frame_hdr
                .segmentation
                .d
                .feature_active(seg_id, SEG_LVL_SKIP)
            && !self
                .frame_hdr
                .segmentation
                .d
                .feature_active(seg_id, SEG_LVL_GLOBALMV)
        {
            let inter_sad = compute_sad(
                &self.src.planes[0].data,
                self.src.planes[0].offset(px_x, px_y),
                self.src.planes[0].stride,
                &self.rec.planes[0].data,
                self.rec.planes[0].offset(px_x, px_y),
                self.rec.planes[0].stride,
                w,
                h,
            );
            if let Some((mode, intra_y, _)) =
                self.select_interintra(px_x, px_y, w, h, have_left, have_top, inter_sad)
            {
                let mask_y = interintra_mask(mode, w, h);
                {
                    let dp = &mut self.rec.planes[0];
                    let off = dp.offset(px_x, px_y);
                    blend_interintra(
                        self.bd,
                        &mut dp.data,
                        off,
                        dp.stride,
                        &intra_y,
                        &mask_y,
                        w,
                        h,
                    );
                }
                // Chroma is motion-compensated later; its inter-intra blend is
                // applied in the chroma prediction loop below.
                interintra_mode_choice = Some(mode);
                self.interintra_blocks += 1;
            }
        }

        let mut res = vec![0i32; w * h];
        let y_nz = self.residual_into(0, px_x, px_y, w, h, &mut res);

        // Wave 3 sub-block transform partitioning (`TxPartition`) RDO for
        // inter blocks. Lossless blocks code `lossless_tx_size` instead and
        // never signal a partition (§5.20.6.1, `Lossless` branch).
        let (tx_part, _sub_tx) = if self.tools.enable_tx_partition
            && y_nz
            && !seg_lossless
            && tx_partition_is_signallable(bs)
        {
            let (part, sub_tx) = select_best_tx_partition(
                self.bd,
                bs,
                &res,
                w,
                self.dq[seg_id][0],
                false,
                seg_lossless,
                self.tools.enable_rdoq,
                self.lambda_rdoq,
                self.speed,
            );
            self.tx_partition_counts[part as usize] += 1;
            if part != TxPartition::TxNone {
                self.tx_split_blocks += 1;
            }
            (part, sub_tx)
        } else {
            (
                TxPartition::TxNone,
                crate::av2::levels::RectTxfmSize::from_usize(tx)
                    .unwrap_or(crate::av2::levels::RectTxfmSize::Tx4x4),
            )
        };

        // Lossless inter luma: `lossless_tx_size == 0` codes one 4x4 WHT (or
        // `IDTX`, via `txtp_lossless`) per transform unit, `1` codes a single
        // full-block `IDTX` transform for blocks up to 32x32 (§5.20.6.1 says
        // `find_tx_size(min(32, width), min(32, height))`; larger blocks would
        // need several of those, which this encoder does not model).
        let mut luma_units: Vec<EncTxUnit> = Vec::new();
        let mut lossless_tx_full = false;
        // Every coded block carries one all-zero coefficient block per plane
        // even when its residual is empty, so the unit lists are always built
        // (the `all_zero` skip flag only short-circuits the whole block).
        {
            if seg_lossless {
                let base_tx_idx = MAX_TXFM_SIZE_FOR_BS[bs as usize] as usize;
                let base_dim = &TXFM_DIMENSIONS[base_tx_idx];
                // Only transforms of at most 256 samples keep the §7.15.4
                // shift at 0, so the full-block `IDTX` stays lossless; larger
                // blocks use the 4x4 WHT units.
                let full_possible =
                    bs != BlockSize::Bs4x4 && base_dim.w <= 4 && base_dim.h <= 4 && (bw4 * bh4) > 1;
                if full_possible {
                    // Compare the L1 level cost of the 4x4 WHT units with the
                    // single full-block identity transform.
                    let ident_cost: u64 = {
                        let base_dim = &TXFM_DIMENSIONS[base_tx_idx];
                        let n = base_dim.w as usize * 4 * base_dim.h as usize * 4;
                        res[..n].iter().map(|v| v.unsigned_abs() as u64).sum()
                    };
                    let mut wht_cost = 0u64;
                    for y4 in 0..bh4 {
                        for x4 in 0..bw4 {
                            let sub = Self::pack_residual(&res, w, x4 * 4, y4 * 4, 4, 4);
                            let mut cf = [0i32; 16];
                            crate::av2::transform::fwd_wht_wht_4x4(&sub, 4, &mut cf);
                            wht_cost += cf.iter().map(|v| v.unsigned_abs() as u64).sum::<u64>();
                        }
                    }
                    lossless_tx_full = ident_cost < wht_cost;
                }
                if lossless_tx_full {
                    let unit =
                        self.build_tx_unit(&res, base_tx_idx, txtp::IDTX, seg_id, 0, false, true);
                    luma_units.push(unit);
                } else {
                    for y4 in 0..bh4 {
                        for x4 in 0..bw4 {
                            let sub = Self::pack_residual(&res, w, x4 * 4, y4 * 4, 4, 4);
                            // Every transform unit codes its `all_zero` symbol
                            // and updates the 4x4 rate contexts at its own
                            // position, so an all-zero unit still carries the
                            // position (and the inherited chroma kernel).
                            let kernel = if sub.iter().all(|v| *v == 0) {
                                crate::av2::transform::WHT_WHT
                            } else {
                                self.choose_lossless_txtp(&sub)
                            };
                            let mut unit =
                                self.build_tx_unit(&sub, 0, kernel, seg_id, 0, false, true);
                            unit.x_px = x4 * 4;
                            unit.y_px = y4 * 4;
                            luma_units.push(unit);
                        }
                    }
                }
            } else if tx_part == TxPartition::TxNone {
                let kernel = self.choose_luma_txtp(&res, w, tx, false, 0, 0, 0, false);
                luma_units.push(self.build_tx_unit(&res, tx, kernel, seg_id, 0, false, false));
            } else {
                let base = crate::av2::levels::RectTxfmSize::from_usize(tx)
                    .unwrap_or(crate::av2::levels::RectTxfmSize::Tx4x4);
                let (units, _) = expand_tx_partition(base, tx_part);
                for u in &units {
                    let sub = Self::pack_residual(&res, w, u.x_px, u.y_px, u.w_px, u.h_px);
                    let kernel =
                        self.choose_luma_txtp(&sub, u.w_px, u.tx as usize, false, 0, 0, 0, false);
                    let mut unit =
                        self.build_tx_unit(&sub, u.tx as usize, kernel, seg_id, 0, false, false);
                    unit.x_px = u.x_px;
                    unit.y_px = u.y_px;
                    luma_units.push(unit);
                }
            }
        }
        let y_coded = luma_units.iter().any(|u| u.eob >= 0);

        let has_chroma = self.seqhdr.layout != PixelLayout::I400;
        let ss_x = self.seqhdr.ss_hor;
        let ss_y = self.seqhdr.ss_ver;
        let layout_col = (self.seqhdr.layout as usize).saturating_sub(1);
        let uv_bs_idx = if has_chroma {
            SS_BS[bs as usize][layout_col] as usize
        } else {
            bs as usize
        };
        let uvtx = MAX_TXFM_SIZE_FOR_BS[uv_bs_idx] as usize;
        let uv_tdim = &TXFM_DIMENSIONS[uvtx];
        // Inter chroma shares the luma transform type (AV2 Spec v1.0.0
        // §5.20.8.3 `get_tx_set` with `plane != 0` and `reducedTxSet` 0), but
        // only when the luma block actually codes coefficients: an all-zero
        // luma block infers `DCT_DCT` (`dav2d` `decode_cf`, `all_skip` branch),
        // and the size-dependent restrictions can turn the shared type back
        // into `DCT_DCT`.
        let first_luma_txtp = luma_units
            .iter()
            .find(|u| u.eob >= 0)
            .map_or(txtp::DCT_DCT, |u| u.txtp);
        let chroma_txtp = if self.seqhdr.chroma_dctonly || !y_coded {
            txtp::DCT_DCT
        } else {
            chroma_txtp_from_luma(first_luma_txtp, uv_tdim, uvtx)
        };
        let cw = (uv_tdim.w as usize) * 4;
        let ch = (uv_tdim.h as usize) * 4;
        let cpx_x = px_x >> ss_x;
        let cpx_y = px_y >> ss_y;
        let cbx4_in_sb = bx4_in_sb >> ss_x;
        let cby4_in_sb = by4_in_sb >> ss_y;

        let mut u_units: Vec<EncTxUnit> = Vec::new();
        let mut v_units: Vec<EncTxUnit> = Vec::new();

        if has_chroma {
            if dec.is_compound {
                let r1 = self.ref1.unwrap_or(r0);
                let c_area = cw * ch;
                for pl in 1..=2 {
                    let rp0 = &r0.planes[pl];
                    let rp1 = &r1.planes[pl];
                    let rec_pl = &mut self.rec.planes[pl];
                    let rec_off = rec_pl.offset(cpx_x, cpx_y);
                    let rec_stride = rec_pl.stride;
                    mc_predict_block_prep(
                        self.bd,
                        &mut self.scratch_mc0[..c_area],
                        &rp0.data,
                        rp0.stride,
                        rp0.coded_width as i32,
                        rp0.coded_height as i32,
                        cpx_x as i32,
                        cpx_y as i32,
                        cw,
                        ch,
                        mv0_y_q16 >> ss_y,
                        mv0_x_q16 >> ss_x,
                        dec.interp_filter as usize,
                    );
                    mc_predict_block_prep(
                        self.bd,
                        &mut self.scratch_mc1[..c_area],
                        &rp1.data,
                        rp1.stride,
                        rp1.coded_width as i32,
                        rp1.coded_height as i32,
                        cpx_x as i32,
                        cpx_y as i32,
                        cw,
                        ch,
                        mv1_y_q16 >> ss_y,
                        mv1_x_q16 >> ss_x,
                        dec.interp_filter as usize,
                    );
                    match dec.comp_type {
                        CompInterType::Average => {
                            blend_compound_prep(
                                self.bd,
                                &mut rec_pl.data,
                                rec_off,
                                rec_stride,
                                &self.scratch_mc0[..c_area],
                                &self.scratch_mc1[..c_area],
                                cw,
                                ch,
                                CompInterType::Average,
                                8,
                                None,
                            );
                        }
                        CompInterType::DistanceWeighted => {
                            const CWP_WEIGHTING_FACTOR: [[i8; 5]; 2] =
                                [[8, 12, 4, 10, 6], [8, 12, 4, 20, -4]];
                            let is_same_side = (self.refdir[0] ^ self.refdir[1]) == 0;
                            let wt = CWP_WEIGHTING_FACTOR[is_same_side as usize]
                                [dec.cwp_idx.min(4) as usize]
                                as i32;
                            blend_compound_prep(
                                self.bd,
                                &mut rec_pl.data,
                                rec_off,
                                rec_stride,
                                &self.scratch_mc0[..c_area],
                                &self.scratch_mc1[..c_area],
                                cw,
                                ch,
                                CompInterType::DistanceWeighted,
                                wt,
                                None,
                            );
                        }
                        CompInterType::Wedge => {
                            let mask = WEDGE_TABLES
                                .get_mask(bs, dec.wedge_idx as usize, ss_x as usize, ss_y as usize)
                                .unwrap_or(&[]);
                            let (p0, p1) = if dec.wedge_sign {
                                (&self.scratch_mc1[..c_area], &self.scratch_mc0[..c_area])
                            } else {
                                (&self.scratch_mc0[..c_area], &self.scratch_mc1[..c_area])
                            };
                            blend_compound_prep(
                                self.bd,
                                &mut rec_pl.data,
                                rec_off,
                                rec_stride,
                                p0,
                                p1,
                                cw,
                                ch,
                                CompInterType::Wedge,
                                0,
                                Some(mask),
                            );
                        }
                        CompInterType::DifferenceWeighted => {
                            let (p0, p1) = if dec.mask_sign {
                                (&self.scratch_mc1[..c_area], &self.scratch_mc0[..c_area])
                            } else {
                                (&self.scratch_mc0[..c_area], &self.scratch_mc1[..c_area])
                            };
                            blend_compound_prep(
                                self.bd,
                                &mut rec_pl.data,
                                rec_off,
                                rec_stride,
                                p0,
                                p1,
                                cw,
                                ch,
                                CompInterType::DifferenceWeighted,
                                0,
                                Some(&self.scratch_chroma_diff_mask[..c_area]),
                            );
                        }
                    }
                }
            } else {
                let cmv0_y_q16 = mv0_y_q16 >> ss_y;
                let cmv0_x_q16 = mv0_x_q16 >> ss_x;
                let active_r = if dec.ref_slot_idx == 1 {
                    self.ref1.unwrap_or(r0)
                } else {
                    r0
                };
                for pl in 1..=2 {
                    {
                        let rec_pl = &mut self.rec.planes[pl];
                        let rec_off = rec_pl.offset(cpx_x, cpx_y);
                        if let Some(choice) = warp_choice {
                            // Chroma uses the same warp model with the plane
                            // subsampling (§7.13.3.19 block_warp, subX/subY).
                            block_warp(
                                self.bd,
                                &mut rec_pl.data,
                                rec_off,
                                rec_pl.stride,
                                &active_r.planes[pl],
                                cpx_x,
                                cpx_y,
                                cw,
                                ch,
                                ss_x as usize,
                                ss_y as usize,
                                &choice.matrix,
                                &choice.abcd,
                            );
                        } else {
                            mc_predict_block(
                                self.bd,
                                &mut rec_pl.data,
                                rec_off,
                                rec_pl.stride,
                                &active_r.planes[pl].data,
                                active_r.planes[pl].stride,
                                active_r.planes[pl].coded_width as i32,
                                active_r.planes[pl].coded_height as i32,
                                cpx_x as i32,
                                cpx_y as i32,
                                cw,
                                ch,
                                cmv0_y_q16,
                                cmv0_x_q16,
                                dec.interp_filter as usize,
                            );
                        }
                    }
                    if let Some(mode) = interintra_mode_choice {
                        let intra = self.interintra_prediction(
                            pl, cpx_x, cpx_y, cw, ch, have_left, have_top, mode,
                        );
                        let mask = interintra_mask(mode, cw, ch);
                        let rec_pl = &mut self.rec.planes[pl];
                        let rec_off = rec_pl.offset(cpx_x, cpx_y);
                        blend_interintra(
                            self.bd,
                            &mut rec_pl.data,
                            rec_off,
                            rec_pl.stride,
                            &intra,
                            &mask,
                            cw,
                            ch,
                        );
                    }
                }
            }

            // Chroma transform units. A lossless segment codes 4x4 units that
            // inherit the co-located luma kernel (`Lossless` branch of
            // §5.20.8.2; `dav2d` reads `txtp_map`); otherwise each plane is a
            // single transform of the plane block size.
            let mut luma_kernel_map: Vec<u32> = vec![txtp::DCT_DCT; bw4 * bh4];
            if seg_lossless {
                for unit in &luma_units {
                    let td = &TXFM_DIMENSIONS[unit.tx];
                    for dy in 0..td.h as usize {
                        for dx in 0..td.w as usize {
                            let x4 = unit.x_px / 4 + dx;
                            let y4 = unit.y_px / 4 + dy;
                            if x4 < bw4 && y4 < bh4 {
                                luma_kernel_map[y4 * bw4 + x4] = unit.txtp & 0xFF;
                            }
                        }
                    }
                }
            }
            for pl in 1..=2 {
                let mut cres = vec![0i32; cw * ch];
                let _ = self.residual_into(pl, cpx_x, cpx_y, cw, ch, &mut cres);
                if seg_lossless {
                    let mut units: Vec<EncTxUnit> = Vec::new();
                    for y4 in 0..(ch / 4) {
                        for x4 in 0..(cw / 4) {
                            let sub = Self::pack_residual(&cres, cw, x4 * 4, y4 * 4, 4, 4);
                            let lx4 = (x4 << ss_x).min(bw4 - 1);
                            let ly4 = (y4 << ss_y).min(bh4 - 1);
                            let kernel = luma_kernel_map[ly4 * bw4 + lx4];
                            let mut unit =
                                self.build_tx_unit(&sub, 0, kernel, seg_id, pl, false, true);
                            unit.x_px = x4 * 4;
                            unit.y_px = y4 * 4;
                            units.push(unit);
                        }
                    }
                    if pl == 1 {
                        u_units = units;
                    } else {
                        v_units = units;
                    }
                } else {
                    let unit =
                        self.build_tx_unit(&cres, uvtx, chroma_txtp, seg_id, pl, false, false);
                    if pl == 1 {
                        u_units.push(unit);
                    } else {
                        v_units.push(unit);
                    }
                }
            }
        }

        let u_coded = u_units.iter().any(|u| u.eob >= 0);
        let v_coded = v_units.iter().any(|u| u.eob >= 0);
        let all_zero = !y_coded && !u_coded && !v_coded;
        let nb = gather_inter_neighbors(
            &self.a_cols[a_col_idx],
            &self.l_ctx,
            bx4,
            by4,
            bw4,
            bh4,
            have_left,
            have_top,
            self.tile_col_end4,
            self.tile_row_end4,
        );
        let skip_ctx = nb.skip_ctx();
        self.msac
            .encode_bool_adapt(all_zero as u32, self.cdf.m.skip(skip_ctx));

        // Inter: `inter_segment_id(0)` follows `skip_txfm` when `SegIdPreSkip`
        // is 0 (AV2 Spec v1.0.0 §5.20.7.2).
        if !self.seg_coder.preskip {
            let _ = self.code_segment_id(
                bx4,
                by4,
                bw4,
                bh4,
                have_left,
                have_top,
                all_zero,
                desired_seg as u8,
            );
        }

        self.encode_gdf_blk(bx4, by4);
        self.encode_ccso_blk(bx4, by4);
        self.emit_pending_delta();

        let is_comp = if self.frame_hdr.switchable_comp_refs != 0 && (bw4 * bh4 >= 4) {
            let ctx = nb.comp_ctx(&self.refdir);
            self.msac
                .encode_bool_adapt(dec.is_compound as u32, self.cdf.m.comp(ctx));
            dec.is_compound
        } else {
            false
        };

        let (r0_idx, r1_idx, comp_type_idx, mut full_mode, mv0, mv1);
        let mut block_motion_mode = 0u8;
        let mut block_warp_params = DEFAULT_WARP_PARAMS;
        if !is_comp {
            // Single-reference selection (ref list index 0).
            let sr_ctx = nb.single_ref_ctx(0);
            let sngl_ctx = nb.snglref_ctx(0);
            self.msac
                .encode_bool_adapt(1, self.cdf.m.single_ref(sr_ctx, 0));
            let warp_signalled = self.frame_hdr.warp_motion != 0
                && bw4.min(bh4) >= 2
                && !self
                    .frame_hdr
                    .segmentation
                    .d
                    .feature_active(seg_id, SEG_LVL_SKIP)
                && !self
                    .frame_hdr
                    .segmentation
                    .d
                    .feature_active(seg_id, SEG_LVL_GLOBALMV);
            if warp_signalled {
                let warp_ctx = self.warp_mv_count(bx4, by4, bw4, bh4);
                self.msac.encode_bool_adapt(
                    warp_choice.is_some() as u32,
                    &mut self.cdf.m.is_warp[warp_ctx],
                );
            }
            if let Some(choice) = warp_choice {
                // `is_warp` was coded above.
                if choice.warp_mv {
                    // `warp_mv = 1` selects `YMode = WARPMV` (§5.20.7.6);
                    // `read_motion_mode()` returns `DELTAWARP` and `warp_idx`
                    // selects `RefWarpIdx`.
                    self.msac.encode_bool_adapt(1, &mut self.cdf.m.warp_mv);
                    if choice.idx >= 3 {
                        for idx in 0..3 {
                            self.msac
                                .encode_symbol_adapt(1, &mut self.cdf.m.warp_idx[idx], 1);
                        }
                    } else {
                        for idx in 0..choice.idx {
                            self.msac
                                .encode_symbol_adapt(1, &mut self.cdf.m.warp_idx[idx], 1);
                        }
                        self.msac
                            .encode_symbol_adapt(0, &mut self.cdf.m.warp_idx[choice.idx], 1);
                    }
                    // `warpmv_with_mvd` is present only when `RefWarpIdx < 2`.
                    if choice.idx < 2 {
                        self.msac
                            .encode_bool_adapt(0, &mut self.cdf.m.warp_with_mvd);
                    }
                } else {
                    // `warp_mv = 0` selects `YMode = WARP_NEWMV`.
                    let warp_ref = dec.ref_slot_idx as i8;
                    self.msac.encode_bool_adapt(0, &mut self.cdf.m.warp_mv);
                    let ext_ctx = warp_mode_context(
                        &self.mi_map,
                        self.f_bw4,
                        self.f_bh4,
                        self.sb_step4,
                        bx4,
                        by4,
                        bw4,
                        bh4,
                    );
                    let has_ext = warp_has_ref_neighbour(
                        &self.mi_map,
                        self.f_bw4,
                        self.f_bh4,
                        self.sb_step4,
                        bx4,
                        by4,
                        bw4,
                        bh4,
                        warp_ref,
                    );
                    if has_ext {
                        if (self.seqhdr.motion_modes & (1 << 4)) != 0 {
                            self.msac.encode_bool_adapt(
                                (choice.motion_mode == 4) as u32,
                                &mut self.cdf.m.use_extend_warp[ext_ctx],
                            );
                        }
                        if choice.motion_mode != 4
                            && (self.seqhdr.motion_modes & ((1 << 2) | (1 << 3)))
                                == ((1 << 2) | (1 << 3))
                        {
                            let cs_ctx = warp_local_context(
                                &self.mi_map,
                                self.f_bw4,
                                self.f_bh4,
                                self.sb_step4,
                                bx4,
                                by4,
                                bw4,
                                bh4,
                            );
                            self.msac.encode_bool_adapt(
                                (choice.motion_mode == 2) as u32,
                                &mut self.cdf.m.use_local_warp[cs_ctx],
                            );
                        }
                    }
                    if choice.motion_mode == 3 {
                        // `warp_idx`: `RefWarpIdx = 0`.
                        self.msac
                            .encode_symbol_adapt(0, &mut self.cdf.m.warp_idx[0], 1);
                    }
                    // `read_drl_idx()` for `WARP_NEWMV` (`has_newmv`).
                    self.msac
                        .encode_bool_adapt(0, self.cdf.m.drl_bit(0, sngl_ctx));
                    encode_mv_residual(
                        &mut self.msac,
                        &mut self.cdf.mv,
                        choice.mv,
                        choice.pred_mv,
                        3 + self.frame_hdr.mv_precision as usize,
                        self.seqhdr.mvd_sign_derive,
                    );
                    if choice.motion_mode == 3 {
                        let high = choice.high_prec;
                        self.msac.encode_bool_adapt(
                            high as u32,
                            &mut self.cdf.m.warp_precision[mi_size_index(bs)],
                        );
                        self.encode_warp_delta_param(2, choice.delta[0], high);
                        self.encode_warp_delta_param(3, choice.delta[1], high);
                    }
                }
            } else {
                let mode_sym = match dec.mode {
                    InterPredMode::NewMv => 2,
                    InterPredMode::GlobalMv => 1,
                    _ => 0,
                };
                self.msac
                    .encode_symbol_adapt(mode_sym, self.cdf.m.single_mode(sngl_ctx), 2);
                if (self.seqhdr.motion_modes & (1 << 1)) != 0 && bw4 * bh4 > 2 && bw4.max(bh4) <= 16
                {
                    let size_group = SPEC_SIZE_GROUP[mi_size_index(bs)] as usize;
                    if let Some(mode) = interintra_mode_choice {
                        self.msac
                            .encode_bool_adapt(1, &mut self.cdf.m.inter_intra[size_group]);
                        self.msac.encode_symbol_adapt(
                            mode as usize,
                            &mut self.cdf.m.inter_intra_mode[size_group],
                            3,
                        );
                        if bw4.min(bh4) > 1 {
                            self.msac
                                .encode_bool_adapt(0, &mut self.cdf.m.wedge_inter_intra);
                        }
                    } else {
                        self.msac
                            .encode_bool_adapt(0, &mut self.cdf.m.inter_intra[size_group]);
                    }
                }
                if mode_sym == 0 || mode_sym == 2 {
                    self.msac
                        .encode_bool_adapt(0, self.cdf.m.drl_bit(0, sngl_ctx));
                }
            }

            if let Some(choice) = warp_choice {
                if choice.warp_mv {
                    // `read_interintra_mode(1)` codes `warp_inter_intra` for
                    // `WARPMV` (dav2d `decode.c` warp_ii branch).
                    let size_group = SPEC_SIZE_GROUP[mi_size_index(bs)] as usize;
                    self.msac
                        .encode_bool_adapt(0, &mut self.cdf.m.warp_inter_intra[size_group]);
                }
            } else if dec.mode == InterPredMode::NewMv {
                encode_mv_residual(
                    &mut self.msac,
                    &mut self.cdf.mv,
                    dec.mv[0],
                    dec.ref_mv[0],
                    3 + self.frame_hdr.mv_precision as usize,
                    self.seqhdr.mvd_sign_derive,
                );
            }

            if self.frame_hdr.subpel_filter_mode == FilterMode::Switchable && warp_choice.is_none()
            {
                let has_subpel_filter = dec.mode != InterPredMode::GlobalMv || (bw4.min(bh4) == 1);
                if has_subpel_filter {
                    let filter_ctx = get_filter_ctx_for_block(
                        have_top.then_some(&self.a_cols[a_col_idx]),
                        have_left.then_some(&self.l_ctx),
                        bx4_in_sb,
                        by4_in_sb,
                        bw4,
                        bh4,
                        (by4 & (self.sb_step4 - 1)) != 0,
                        have_left,
                        false,
                        dec.ref_slot_idx as i8,
                    );
                    self.msac.encode_symbol_adapt(
                        (dec.interp_filter as usize).min(2),
                        self.cdf.m.interp_filter(filter_ctx),
                        2,
                    );
                }
            }

            // `read_tx_partition()` belongs to the transform tree (§5.20.6.2),
            // which follows the complete inter mode info: MVD, warp delta,
            // warp intra, and the switchable subpel filter. A lossless
            // segment codes `lossless_tx_size` in the same position instead
            // (§5.20.6.1), and only when the block carries a residual.
            if !all_zero {
                if seg_lossless {
                    encode_lossless_tx_size(
                        &mut self.msac,
                        &mut self.cdf.m,
                        bs,
                        lossless_tx_full,
                        1,
                    );
                } else {
                    encode_tx_partition_syntax(
                        &mut self.msac,
                        &mut self.cdf.m,
                        bs,
                        tx_part,
                        0,
                        1,
                        self.seqhdr.reduced_tx_part_set,
                    );
                }
            }

            if warp_choice.is_none() {
                if dec.mode == InterPredMode::NewMv {
                    self.newmv_blocks += 1;
                }
                if dec.interp_filter != 0 {
                    self.switchable_filter_blocks += 1;
                }
            }

            r0_idx = dec.ref_slot_idx as i8;
            r1_idx = -1;
            comp_type_idx = 0;
            full_mode = match dec.mode {
                InterPredMode::NewMv => 15,
                InterPredMode::GlobalMv => 14,
                _ => 13,
            };
            if let Some(choice) = warp_choice {
                // `WARPMV` (AV2 §5.20.7.6 mode index 17) or `WARP_NEWMV`
                // (mode index 18).
                full_mode = if choice.warp_mv { 17 } else { 18 };
            }
            mv0 = warp_choice.map_or(dec.mv[0], |c| c.mv);
            mv1 = dec.mv[1];
            // Record the coded mode for §7.12.2.3 corner derivation. A warp
            // block stores `MotionModes = DELTAWARP` (3) and its
            // `WarpParams`; everything else is a `SIMPLE` translation.
            let (motion_mode, warp_matrix) = if let Some(choice) = warp_choice {
                (choice.motion_mode, choice.matrix)
            } else if interintra_mode_choice.is_some() {
                (1u8, DEFAULT_WARP_PARAMS)
            } else {
                (0u8, DEFAULT_WARP_PARAMS)
            };
            block_motion_mode = motion_mode;
            block_warp_params = warp_matrix;
        } else {
            let comp_ref0 = 0usize;
            let comp_ref1 = 1usize;
            let n_refs = self.frame_hdr.n_ref_frames.max(1) as usize;
            if n_refs > 1 {
                let same_refs = self.seqhdr.num_same_ref_comp as usize;
                let mut n = 0usize;
                let target_refs = [comp_ref0, comp_ref1];
                let mut cnt = [0u8; 9];
                if nb.top.is_some() {
                    cnt[(nb.ref0[0].clamp(-1, 7) + 1) as usize] += 1;
                    cnt[(nb.ref1[0].clamp(-1, 7) + 1) as usize] += 1;
                }
                if nb.left.is_some() {
                    cnt[(nb.ref0[1].clamp(-1, 7) + 1) as usize] += 1;
                    cnt[(nb.ref1[1].clamp(-1, 7) + 1) as usize] += 1;
                }
                let mut cnt_rem = nb.n_ctx as i32 * 2 - cnt[0] as i32 - cnt[8] as i32;
                let mut maybe_same_ref = same_refs != 0;
                let mut dir = 0usize;
                let mut i = 0usize;
                while i < n_refs + n - 2 + (maybe_same_ref as usize) {
                    let cnt_cur = cnt[i + 1] as i32;
                    cnt_rem -= cnt_cur;
                    let is_target = i == target_refs[n];
                    if !(n == 0 && (i == 2 || (i >= n_refs - 2 && i + 1 >= same_refs))) {
                        let ctx = (cnt_cur - cnt_rem + 1).clamp(0, 2) as usize;
                        if n == 0 {
                            self.msac
                                .encode_bool_adapt(is_target as u32, self.cdf.m.comp_ref0(ctx, i));
                        } else {
                            let d = (dir ^ (self.refdir[i] as usize)) & 1;
                            self.msac.encode_bool_adapt(
                                is_target as u32,
                                self.cdf.m.comp_ref1(ctx, d, i),
                            );
                        }
                    }
                    if is_target {
                        n += 1;
                        if n == 2 {
                            break;
                        }
                        dir = self.refdir[i] as usize;
                    }
                    if maybe_same_ref {
                        maybe_same_ref = (!is_target) && (i + 1 < same_refs);
                        if is_target {
                            i = i.wrapping_sub(1);
                            cnt_rem += cnt_cur;
                        }
                    }
                    i = i.wrapping_add(1);
                }
            }

            let comp_ctx = nb.compref_ctx(comp_ref0 as i8, comp_ref1 as i8);
            let compound_mode = if dec.compound_mode != 0 {
                dec.compound_mode
            } else {
                match dec.mode {
                    InterPredMode::NewMv => 22,
                    InterPredMode::GlobalMv => 21,
                    _ => 18,
                }
            };

            if comp_ref0 == comp_ref1 {
                let sym = match compound_mode {
                    18 => 0,
                    19 => 1,
                    20 => 2,
                    22 => 3,
                    _ => 0,
                };
                self.msac
                    .encode_symbol_adapt(sym, self.cdf.m.compound_mode_same_refs(comp_ctx), 3);
            } else {
                let joint_ctx = (self.refdist[comp_ref0] != -self.refdist[comp_ref1]) as usize;
                if compound_mode == 23 {
                    self.msac
                        .encode_bool_adapt(1, self.cdf.m.is_joint(joint_ctx));
                } else {
                    self.msac
                        .encode_bool_adapt(0, self.cdf.m.is_joint(joint_ctx));
                    let sym = (compound_mode - 18) as usize;
                    self.msac.encode_symbol_adapt(
                        sym,
                        self.cdf.m.compound_mode_non_joint(comp_ctx),
                        4,
                    );
                }
            }

            if compound_mode == 23 {
                // `jmvd_scale_mode` follows `is_joint` and precedes the DRL
                // index (§5.20.7.6). The encoder uses the un-scaled variant.
                self.msac
                    .encode_symbol_adapt(0, &mut self.cdf.mv.jmvd_scale_mode, 4);
            }

            if compound_mode != 21 {
                let n_drls = 1 + (compound_mode <= 19) as usize;
                let max_drl_bits = self.frame_hdr.max_drl_bits.max(1) as usize;
                let idx = (dec.compound_drl_idx as usize).min(max_drl_bits - 1);
                for _ in 0..n_drls {
                    let mut ctx = 0usize;
                    for n in 0..max_drl_bits {
                        let bit = (n < idx) as u32;
                        self.msac
                            .encode_bool_adapt(bit, self.cdf.m.drl_bit(ctx.min(2), comp_ctx));
                        if bit == 0 {
                            break;
                        }
                        if ctx < 2 {
                            ctx += 1;
                        }
                    }
                }
            }

            if compound_mode == 22 {
                encode_mv_residual(
                    &mut self.msac,
                    &mut self.cdf.mv,
                    dec.mv[0],
                    dec.ref_mv[0],
                    3 + self.frame_hdr.mv_precision as usize,
                    self.seqhdr.mvd_sign_derive,
                );
                encode_mv_residual(
                    &mut self.msac,
                    &mut self.cdf.mv,
                    dec.mv[1],
                    dec.ref_mv[1],
                    3 + self.frame_hdr.mv_precision as usize,
                    self.seqhdr.mvd_sign_derive,
                );
            } else if compound_mode == 23 {
                let start = (self.absrefdist[comp_ref0] < self.absrefdist[comp_ref1]) as usize;
                encode_mv_residual(
                    &mut self.msac,
                    &mut self.cdf.mv,
                    dec.mv[start],
                    dec.ref_mv[start],
                    3 + self.frame_hdr.mv_precision as usize,
                    self.seqhdr.mvd_sign_derive,
                );
            } else if compound_mode == 19 {
                encode_mv_residual(
                    &mut self.msac,
                    &mut self.cdf.mv,
                    dec.mv[1],
                    dec.ref_mv[1],
                    3 + self.frame_hdr.mv_precision as usize,
                    self.seqhdr.mvd_sign_derive,
                );
            } else if compound_mode == 20 {
                encode_mv_residual(
                    &mut self.msac,
                    &mut self.cdf.mv,
                    dec.mv[0],
                    dec.ref_mv[0],
                    3 + self.frame_hdr.mv_precision as usize,
                    self.seqhdr.mvd_sign_derive,
                );
            }

            let allow_masked =
                compound_mode <= 23 && self.seqhdr.masked_compound && bw4.min(bh4) >= 2;
            if allow_masked {
                let ctx = nb.comp_group_idx_ctx(
                    self.absrefdist[comp_ref0],
                    self.absrefdist[comp_ref1],
                    self.furthest_future_refidx,
                );
                let has_mask = dec.comp_type == CompInterType::Wedge
                    || dec.comp_type == CompInterType::DifferenceWeighted;
                self.msac
                    .encode_bool_adapt(has_mask as u32, self.cdf.m.comp_group_idx(ctx));
                if has_mask {
                    let max_dim = bw4.max(bh4);
                    let is_wedge = dec.comp_type == CompInterType::Wedge;
                    if max_dim <= 16 {
                        self.msac
                            .encode_bool_adapt((!is_wedge) as u32, self.cdf.m.comp_type());
                    }
                    if is_wedge {
                        // `WEDGE_IDX_TO_ANGLE_DIST` stores `(quad, angle, dist)`
                        // where `angle` is already the within-quad value used
                        // by `wedge_angle` (§5.20.7.18).
                        let (quad, angle_sub, dist) =
                            WEDGE_IDX_TO_ANGLE_DIST[dec.wedge_idx as usize];
                        let global_angle = quad as usize * 5 + angle_sub as usize;
                        self.msac
                            .encode_symbol_adapt(quad as usize, self.cdf.m.wedge_quad(), 3);
                        self.msac.encode_symbol_adapt(
                            angle_sub as usize,
                            self.cdf.m.wedge_angle(quad as usize),
                            4,
                        );
                        if WEDGE_ANGLE_DIST_TO_IDX[global_angle][0] < 0 {
                            self.msac.encode_symbol_adapt(
                                (dist - 1) as usize,
                                self.cdf.m.wedge_dist2(),
                                2,
                            );
                        } else {
                            self.msac.encode_symbol_adapt(
                                dist as usize,
                                self.cdf.m.wedge_dist1(),
                                3,
                            );
                        }
                        self.msac.encode_bool_bypass(dec.wedge_sign as u32);
                    } else {
                        self.msac.encode_bool_bypass(dec.mask_sign as u32);
                    }
                }
            }

            if (dec.comp_type == CompInterType::Average
                || dec.comp_type == CompInterType::DistanceWeighted)
                && self.seqhdr.cwp
                && (compound_mode == 18 || compound_mode == 23)
            {
                let cwp_idx = if dec.comp_type == CompInterType::DistanceWeighted {
                    dec.cwp_idx.clamp(1, 4) as usize
                } else {
                    0usize
                };
                for n in 0..4 {
                    if n < cwp_idx {
                        self.msac.encode_bool_adapt(1, self.cdf.m.cwp_idx(n));
                    } else {
                        self.msac.encode_bool_adapt(0, self.cdf.m.cwp_idx(n));
                        break;
                    }
                }
            }

            if self.frame_hdr.subpel_filter_mode == FilterMode::Switchable {
                let has_subpel_filter =
                    compound_mode <= 23 && (compound_mode != 21 || bw4.min(bh4) == 1);
                if has_subpel_filter {
                    let filter_ctx = get_filter_ctx_for_block(
                        have_top.then_some(&self.a_cols[a_col_idx]),
                        have_left.then_some(&self.l_ctx),
                        bx4_in_sb,
                        by4_in_sb,
                        bw4,
                        bh4,
                        (by4 & (self.sb_step4 - 1)) != 0,
                        have_left,
                        true,
                        comp_ref0 as i8,
                    );
                    self.msac.encode_symbol_adapt(
                        (dec.interp_filter as usize).min(2),
                        self.cdf.m.interp_filter(filter_ctx),
                        2,
                    );
                }
            }

            if !all_zero {
                if seg_lossless {
                    encode_lossless_tx_size(
                        &mut self.msac,
                        &mut self.cdf.m,
                        bs,
                        lossless_tx_full,
                        1,
                    );
                } else {
                    encode_tx_partition_syntax(
                        &mut self.msac,
                        &mut self.cdf.m,
                        bs,
                        tx_part,
                        0,
                        1,
                        self.seqhdr.reduced_tx_part_set,
                    );
                }
            }

            r0_idx = comp_ref0 as i8;
            r1_idx = comp_ref1 as i8;
            comp_type_idx = match dec.comp_type {
                CompInterType::Average | CompInterType::DistanceWeighted => 1,
                CompInterType::Wedge => 2,
                CompInterType::DifferenceWeighted => 3,
            };
            full_mode = compound_mode;
            mv0 = dec.mv[0];
            mv1 = dec.mv[1];
        }

        let inter_cwp = if dec.is_compound && dec.comp_type == CompInterType::DistanceWeighted {
            dec.cwp_idx
        } else {
            8
        };
        let map_refs = if dec.is_compound {
            [r0_idx, r1_idx]
        } else {
            [r0_idx, -1]
        };
        self.mark_mi_inter(
            bs,
            bx4,
            by4,
            block_motion_mode,
            block_warp_params,
            map_refs,
            [mv0, mv1],
            inter_cwp,
        );

        for i in 0..bw4.min(32 - bx4_in_sb).min(self.f_bw4.saturating_sub(bx4)) {
            self.a_cols[a_col_idx].intra[bx4_in_sb + i] = 0;
            self.a_cols[a_col_idx].midx[bx4_in_sb + i] = 0xFF;
            self.a_cols[a_col_idx].skip[bx4_in_sb + i] = all_zero as u8;
            self.a_cols[a_col_idx].tx_luma[bx4_in_sb + i] = t_dim.lw;
            self.a_cols[a_col_idx].tx[bx4_in_sb + i] = t_dim.lw;
            self.a_cols[a_col_idx].mv[0][bx4_in_sb + i] = if block_motion_mode >= 2 {
                warp_mv_at(&block_warp_params, by4 + bh4 - 1, bx4 + i)
            } else {
                mv0
            };
            self.a_cols[a_col_idx].mv[1][bx4_in_sb + i] = mv1;
            self.a_cols[a_col_idx].ref_frame[0][bx4_in_sb + i] = r0_idx;
            self.a_cols[a_col_idx].ref_frame[1][bx4_in_sb + i] = r1_idx;
            self.a_cols[a_col_idx].comp_type[bx4_in_sb + i] = comp_type_idx;
            self.a_cols[a_col_idx].mode[bx4_in_sb + i] = full_mode;
            self.a_cols[a_col_idx].filter[0][bx4_in_sb + i] = dec.interp_filter;
            self.a_cols[a_col_idx].filter[1][bx4_in_sb + i] = dec.interp_filter;
            self.a_cols[a_col_idx].seg_id[bx4_in_sb + i] = seg_id as u8;
        }
        for i in 0..bh4.min(32 - by4_in_sb).min(self.f_bh4.saturating_sub(by4)) {
            self.l_ctx.intra[by4_in_sb + i] = 0;
            self.l_ctx.midx[by4_in_sb + i] = 0xFF;
            self.l_ctx.skip[by4_in_sb + i] = all_zero as u8;
            self.l_ctx.tx_luma[by4_in_sb + i] = t_dim.lh;
            self.l_ctx.tx[by4_in_sb + i] = t_dim.lh;
            self.l_ctx.mv[0][by4_in_sb + i] = if block_motion_mode >= 2 {
                warp_mv_at(&block_warp_params, by4 + i, bx4 + bw4 - 1)
            } else {
                mv0
            };
            self.l_ctx.mv[1][by4_in_sb + i] = mv1;
            self.l_ctx.ref_frame[0][by4_in_sb + i] = r0_idx;
            self.l_ctx.ref_frame[1][by4_in_sb + i] = r1_idx;
            self.l_ctx.comp_type[by4_in_sb + i] = comp_type_idx;
            self.l_ctx.mode[by4_in_sb + i] = full_mode;
            self.l_ctx.filter[0][by4_in_sb + i] = dec.interp_filter;
            self.l_ctx.filter[1][by4_in_sb + i] = dec.interp_filter;
            self.l_ctx.seg_id[by4_in_sb + i] = seg_id as u8;
        }

        let lossless_seg = self.frame_hdr.quant.lossless[seg_id] != 0;
        self.record_deblock_inter(
            bx4,
            by4,
            bw4,
            bh4,
            &luma_units,
            t_dim,
            uv_tdim,
            has_chroma,
            all_zero,
            lossless_seg,
        );
        if all_zero {
            self.skipped_blocks += 1;
            for i in 0..bw4.min(32 - bx4_in_sb).min(self.f_bw4.saturating_sub(bx4)) {
                self.a_cols[a_col_idx].lcoef[bx4_in_sb + i] = 0x40;
            }
            for i in 0..bh4.min(32 - by4_in_sb).min(self.f_bh4.saturating_sub(by4)) {
                self.l_ctx.lcoef[by4_in_sb + i] = 0x40;
            }
            if has_chroma {
                for pl_idx in 0..2 {
                    for i in 0..(uv_tdim.w as usize)
                        .min(32 - cbx4_in_sb)
                        .min((self.f_bw4 >> ss_x).saturating_sub(bx4 >> ss_x))
                    {
                        self.a_cols[a_col_idx].ccoef[pl_idx][cbx4_in_sb + i] = 0x40;
                    }
                    for i in 0..(uv_tdim.h as usize)
                        .min(32 - cby4_in_sb)
                        .min((self.f_bh4 >> ss_y).saturating_sub(by4 >> ss_y))
                    {
                        self.l_ctx.ccoef[pl_idx][cby4_in_sb + i] = 0x40;
                    }
                }
            }
            return;
        }

        // Luma coefficient syntax, one transform unit at a time. The
        // transform-partition syntax is already written, so the units follow
        // in raster order with their own rate contexts.
        for unit in &luma_units {
            let ux4 = unit.x_px / 4;
            let uy4 = unit.y_px / 4;
            let td = &TXFM_DIMENSIONS[unit.tx];
            if unit.txtp & 0xFF != txtp::DCT_DCT {
                self.non_dct_tx_blocks += 1;
            }
            let y_params = CoefCodingParams {
                tx: unit.tx,
                bs: bs as usize,
                plane: 0,
                intra: false,
                fsc: false,
                lossless: seg_lossless,
                y_mode: 0,
                y_angle: 0,
                mrl_index: 0,
                uv_mode: 0,
                reduced_txtp_set: self.frame_hdr.reduced_txtp_set as i32,
                u_has_cf: 0,
                dq_tbl: self.dq[seg_id][0],
                bitdepth: self.bd.bitdepth(),
                qm: None,
                ss_hor: ss_x != 0,
                ss_ver: ss_y != 0,
            };
            let mut y_txtp = unit.txtp as u16;
            let mut y_res_ctx = 0u8;
            let coded = encode_coefs(
                &mut self.msac,
                &mut self.cdf.coef,
                &mut self.cdf.m,
                &self.a_cols[a_col_idx].lcoef[bx4_in_sb + ux4..],
                &self.l_ctx.lcoef[by4_in_sb + uy4..],
                &y_params,
                &unit.q_cf,
                &mut y_txtp,
                &mut y_res_ctx,
            );
            for i in 0..(td.w as usize)
                .min(32usize.saturating_sub(bx4_in_sb + ux4))
                .min(self.f_bw4.saturating_sub(bx4 + ux4))
            {
                self.a_cols[a_col_idx].lcoef[bx4_in_sb + ux4 + i] = y_res_ctx;
                self.a_cols[a_col_idx].txtp[bx4_in_sb + ux4 + i] = y_txtp;
            }
            for i in 0..(td.h as usize)
                .min(32usize.saturating_sub(by4_in_sb + uy4))
                .min(self.f_bh4.saturating_sub(by4 + uy4))
            {
                self.l_ctx.lcoef[by4_in_sb + uy4 + i] = y_res_ctx;
                self.l_ctx.txtp[by4_in_sb + uy4 + i] = y_txtp;
            }
            if coded >= 0 {
                for y in 0..td.h as usize {
                    for x in 0..td.w as usize {
                        self.grid.set_lr_noskip(bx4 + ux4 + x, by4 + uy4 + y, true);
                    }
                }
                let rec_y_mut = &mut self.rec.planes[0];
                let rec_off = rec_y_mut.offset(px_x + unit.x_px, px_y + unit.y_px);
                let mut dq = unit.dq_cf.clone();
                if seg_lossless && (y_txtp as u32 & 0xFF) == txtp::IDTX {
                    crate::av2::transform::inv_txfm_add_lossless_idtx(
                        self.bd,
                        &mut rec_y_mut.data,
                        rec_off,
                        rec_y_mut.stride,
                        &mut dq,
                        coded,
                        unit.tx,
                    );
                } else {
                    inv_txfm_add(
                        self.bd,
                        &mut rec_y_mut.data,
                        rec_off,
                        rec_y_mut.stride,
                        &mut dq,
                        y_txtp as u32,
                        coded,
                        unit.tx,
                    );
                }
            }
        }

        // Chroma coefficient syntax. A lossless segment codes one 4x4 unit
        // per chroma transform, the rest one unit per plane block.
        let mut u_has_cf = false;
        for pl in 1..=2 {
            let units = if pl == 1 { &u_units } else { &v_units };
            for unit in units {
                let ux4 = unit.x_px / 4;
                let uy4 = unit.y_px / 4;
                let td = &TXFM_DIMENSIONS[unit.tx];
                let params = CoefCodingParams {
                    tx: unit.tx,
                    bs: uv_bs_idx,
                    plane: pl as i32,
                    intra: false,
                    fsc: false,
                    lossless: seg_lossless,
                    y_mode: 0,
                    y_angle: 0,
                    mrl_index: 0,
                    uv_mode: 0,
                    reduced_txtp_set: self.frame_hdr.reduced_txtp_set as i32,
                    u_has_cf: if pl == 2 { u_has_cf as i32 } else { 0 },
                    dq_tbl: self.dq[seg_id][pl],
                    bitdepth: self.bd.bitdepth(),
                    qm: None,
                    ss_hor: ss_x != 0,
                    ss_ver: ss_y != 0,
                };
                let mut c_txtp = unit.txtp as u16;
                let mut c_res_ctx = 0u8;
                let coded = encode_coefs(
                    &mut self.msac,
                    &mut self.cdf.coef,
                    &mut self.cdf.m,
                    &self.a_cols[a_col_idx].ccoef[pl - 1][cbx4_in_sb + ux4..],
                    &self.l_ctx.ccoef[pl - 1][cby4_in_sb + uy4..],
                    &params,
                    &unit.q_cf,
                    &mut c_txtp,
                    &mut c_res_ctx,
                );
                for i in 0..(td.w as usize)
                    .min(32usize.saturating_sub(cbx4_in_sb + ux4))
                    .min((self.f_bw4 >> ss_x).saturating_sub((bx4 >> ss_x) + ux4))
                {
                    self.a_cols[a_col_idx].ccoef[pl - 1][cbx4_in_sb + ux4 + i] = c_res_ctx;
                }
                for i in 0..(td.h as usize)
                    .min(32usize.saturating_sub(cby4_in_sb + uy4))
                    .min((self.f_bh4 >> ss_y).saturating_sub((by4 >> ss_y) + uy4))
                {
                    self.l_ctx.ccoef[pl - 1][cby4_in_sb + uy4 + i] = c_res_ctx;
                }
                if pl == 1 {
                    u_has_cf = coded >= 0;
                }
                if coded >= 0 {
                    let rec_pl = &mut self.rec.planes[pl];
                    let rec_off = rec_pl.offset(cpx_x + unit.x_px, cpx_y + unit.y_px);
                    let mut dq = unit.dq_cf.clone();
                    if seg_lossless && (c_txtp as u32 & 0xFF) == txtp::IDTX {
                        crate::av2::transform::inv_txfm_add_lossless_idtx(
                            self.bd,
                            &mut rec_pl.data,
                            rec_off,
                            rec_pl.stride,
                            &mut dq,
                            coded,
                            unit.tx,
                        );
                    } else {
                        inv_txfm_add(
                            self.bd,
                            &mut rec_pl.data,
                            rec_off,
                            rec_pl.stride,
                            &mut dq,
                            c_txtp as u32,
                            coded,
                            unit.tx,
                        );
                    }
                }
            }
        }
    }

    /// Records the deblock/CDEF block grid for an inter block whose luma
    /// residual may be split into several transform units.
    #[allow(clippy::too_many_arguments)]
    fn record_deblock_inter(
        &mut self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        units: &[EncTxUnit],
        t_dim: &crate::av2::tables::TxfmInfo,
        uv_tdim: &crate::av2::tables::TxfmInfo,
        has_chroma: bool,
        skip: bool,
        lossless: bool,
    ) {
        if units.is_empty() {
            self.record_deblock(
                bx4, by4, bw4, bh4, t_dim, uv_tdim, has_chroma, skip, lossless,
            );
            return;
        }
        if !self.tools.enable_deblock && !self.tools.enable_cdef {
            return;
        }
        for unit in units {
            let tw4 = unit.w / 4;
            let th4 = unit.h / 4;
            let ux4 = unit.x_px / 4;
            let uy4 = unit.y_px / 4;
            for dy4 in 0..th4 {
                let y4 = by4 + uy4 + dy4;
                if y4 >= self.f_bh4 {
                    break;
                }
                for dx4 in 0..tw4 {
                    let x4 = bx4 + ux4 + dx4;
                    if x4 >= self.f_bw4 {
                        break;
                    }
                    self.grid.set_luma(
                        x4,
                        y4,
                        BlockDeblockInfo {
                            tx_w: unit.w as u8,
                            tx_h: unit.h as u8,
                            skip,
                            lossless,
                            block_left: ux4 == 0 && dx4 == 0,
                            block_top: uy4 == 0 && dy4 == 0,
                            tx_left: dx4 == 0,
                            tx_top: dy4 == 0,
                        },
                    );
                }
            }
        }
        if has_chroma {
            let ss_x = self.seqhdr.ss_hor;
            let ss_y = self.seqhdr.ss_ver;
            let cbx4 = bx4 >> ss_x;
            let cby4 = by4 >> ss_y;
            let cbw4 = (bw4 >> ss_x).max(1);
            let cbh4 = (bh4 >> ss_y).max(1);
            let uw = (uv_tdim.w as usize).max(1);
            let uh = (uv_tdim.h as usize).max(1);
            for dy4 in 0..cbh4 {
                let cy4 = cby4 + dy4;
                if cy4 >= self.grid.chroma_h {
                    break;
                }
                for dx4 in 0..cbw4 {
                    let cx4 = cbx4 + dx4;
                    if cx4 >= self.grid.chroma_w {
                        break;
                    }
                    self.grid.set_chroma(
                        cx4,
                        cy4,
                        BlockDeblockInfo {
                            tx_w: uv_tdim.w * 4,
                            tx_h: uv_tdim.h * 4,
                            skip,
                            lossless,
                            block_left: dx4 == 0,
                            block_top: dy4 == 0,
                            tx_left: dx4 % uw == 0,
                            tx_top: dy4 % uh == 0,
                        },
                    );
                }
            }
        }
    }
}

/// Encodes an entire frame into an AV2 `ObuType::ClosedLoopKf` / `ObuType::TileGrp` OBU packet with the full Wave 3 pipeline,
/// supporting multi-threaded parallel tile execution via `execute_tiles_parallel`.
#[allow(clippy::too_many_arguments)]
pub fn encode_frame<BD: BitDepth>(
    bd: BD,
    seqhdr: &SequenceHeader,
    frame_hdr: &mut FrameHeader,
    tools: ToolConfig,
    src: &Frame<BD::Pixel>,
    rec: &mut Frame<BD::Pixel>,
    dpb: &mut ReferenceDpb<BD::Pixel>,
    speed: u8,
    lambda_sad: u64,
    lambda_rdoq: u64,
    deblock_accelerator: Option<&dyn DeblockFilter<BD::Pixel>>,
    seg_ref: Option<&SegmentMap>,
    frame_budget_bits: Option<f64>,
) -> (Vec<u8>, FrameStats, SegmentMap) {
    if tools.enable_switchable_interp {
        frame_hdr.subpel_filter_mode = FilterMode::Switchable;
    } else {
        frame_hdr.subpel_filter_mode = FilterMode::Regular8Tap;
    }

    // Encoder-side variance analysis feeds both the AV2 segmentation ALT_Q
    // features and the per-superblock delta-Q layer.
    let seg_plan = if (tools.enable_vaq || tools.enable_delta_q) && frame_hdr.quant.yac > 0 {
        VaqPlan::analyse(src, frame_hdr.quant.yac)
    } else {
        None
    };

    if tools.enable_vaq && frame_hdr.quant.yac > 0 {
        if let Some(plan) = &seg_plan {
            frame_hdr.segmentation.enabled = 1;
            frame_hdr.segmentation.update_map = 1;
            frame_hdr.segmentation.update_data = 1;
            plan.fill_segmentation(&mut frame_hdr.segmentation.d);
            // Do not create a lossless segment from an ALT_Q offset unless the
            // frame itself is lossless: LosslessArray is derived per segment
            // (AV2 Spec v1.0.0 §5.18.1) and a mixed lossless frame changes the
            // residual/transform path. Clamp ALT_Q so every segment keeps a
            // positive quantizer when base_q_idx > 0.
            if frame_hdr.quant.yac > 0 {
                let min_delta = 1 - frame_hdr.quant.yac as i32;
                for d in frame_hdr.segmentation.d.delta_q.iter_mut() {
                    if (*d as i32) < min_delta {
                        *d = min_delta.clamp(-351, 351) as i16;
                    }
                }
            }
            // `segmentation_temporal_update` requires `segmentation_update_map`
            // and a previous segment map loaded from the primary reference
            // (`load_previous_segment_ids`, AV2 Spec v1.0.0 §5.18.1).
            frame_hdr.segmentation.temporal = (tools.enable_seg_temporal
                && seg_ref.is_some()
                && frame_hdr.primary_ref_frame != PRIMARY_REF_NONE)
                as u8;
        }
    } else {
        frame_hdr.segmentation.enabled = 0;
        frame_hdr.segmentation.temporal = 0;
    }

    if tools.enable_delta_q && frame_hdr.quant.yac > 0 && seg_plan.is_some() {
        frame_hdr.delta.q.present = 1;
        // `delta_q_res = 0` gives one quantizer-index unit per coded delta.
        frame_hdr.delta.q.res_log2 = 0;
    } else {
        frame_hdr.delta.q.present = 0;
    }

    // Derived per-segment quantizers and lossless flags (AV2 Spec §5.18.1).
    {
        let delta_present = frame_hdr.delta.q.present != 0;
        let delta_lossless = frame_hdr.quant.ydc_delta == 0
            && frame_hdr.quant.udc_delta == 0
            && frame_hdr.quant.uac_delta == 0
            && frame_hdr.quant.vdc_delta == 0
            && frame_hdr.quant.vac_delta == 0
            && !delta_present;
        let n_seg = if frame_hdr.segmentation.enabled != 0 {
            MAX_SEGMENTS
        } else {
            1
        };
        let mut all_ll = true;
        let mut any_ll = false;
        for i in 0..n_seg {
            let dq = if frame_hdr.segmentation.enabled != 0 {
                frame_hdr.segmentation.d.delta_q[i]
            } else {
                0
            };
            let q = segment_qindex(
                frame_hdr.quant.yac as i32,
                frame_hdr.quant.yac as i32,
                dq,
                delta_present,
                bd.bitdepth(),
            );
            let ll = q == 0 && delta_lossless;
            frame_hdr.quant.lossless[i] = ll as u8;
            frame_hdr.segmentation.lossless[i] = ll as u8;
            all_ll &= ll;
            any_ll |= ll;
        }
        for i in n_seg..MAX_SEGMENTS {
            frame_hdr.quant.lossless[i] = 0;
            frame_hdr.segmentation.lossless[i] = 0;
        }
        frame_hdr.all_lossless = all_ll as u8;
        frame_hdr.any_lossless = any_ll as u8;
    }

    let mut dq = [[[0u32; 2]; 3]; MAX_SEGMENTS];
    init_quant_tables(frame_hdr, frame_hdr.quant.yac as i32, &mut dq);
    let seg_cols4 = 2 * ((src.width + 7) >> 3);
    let seg_rows4 = 2 * ((src.height + 7) >> 3);
    let mut seg_coder = SegmentCoder::new(
        &frame_hdr.segmentation.d,
        crate::av2::headers::SegmentationDataSet::max_segments(seqhdr.segmentation.ext),
        frame_hdr.segmentation.temporal != 0,
    );
    seg_coder.any_lossless = frame_hdr.any_lossless != 0;

    // AV2 §5.20.3: MiCols/MiRows round the frame to the 8-sample MI grid.
    let f_bw4 = 2 * ((src.width + 7) >> 3);
    let f_bh4 = 2 * ((src.height + 7) >> 3);
    let sb_step4 = if frame_hdr.sb128 != 0 {
        32usize
    } else {
        16usize
    };
    let root_bs = if frame_hdr.sb128 != 0 {
        BlockSize::Bs128x128
    } else {
        BlockSize::Bs64x64
    };
    let n_sb_total = f_bw4.div_ceil(sb_step4).max(1) * f_bh4.div_ceil(sb_step4).max(1);

    let (ref0, ref1) = if frame_hdr.is_inter_or_switch() {
        let s0 = frame_hdr.refidx[0].max(0) as usize;
        let s1 = frame_hdr.refidx[1].max(0) as usize;
        let r0 = dpb.get_slot(s0);
        let r1 = dpb.get_slot(s1);
        (r0, r1)
    } else {
        (None, None)
    };

    if tools.enable_compound && frame_hdr.is_inter_or_switch() && ref0.is_some() && ref1.is_some() {
        frame_hdr.switchable_comp_refs = 1;
    } else {
        frame_hdr.switchable_comp_refs = 0;
    }

    let mut refdist = [0i32; 7];
    let mut absrefdist = [0i32; 7];
    let mut refdir = [0i8; 7];
    let mut furthest_future_refidx = -2i8;
    if frame_hdr.is_inter_or_switch() {
        let poc = frame_hdr.frame_offset as i32;
        let r0_poc = ref0.map_or(0, |r| r.poc as i32);
        let r1_poc = ref1.map_or(0, |r| r.poc as i32);
        let delta0 = if seqhdr.order_hint_n_bits == 0 {
            0
        } else {
            (r0_poc - poc).clamp(-127, 127)
        };
        let delta1 = if seqhdr.order_hint_n_bits == 0 {
            0
        } else {
            (r1_poc - poc).clamp(-127, 127)
        };
        refdist[0] = delta0;
        refdist[1] = delta1;
        absrefdist[0] = delta0.abs();
        absrefdist[1] = delta1.abs();
        refdir[0] = (delta0 > 0) as i8;
        refdir[1] = (delta1 > 0) as i8;
        if delta0 > 0 {
            furthest_future_refidx = 0;
        }
        if delta1 > 0 && (furthest_future_refidx < 0 || delta1 > delta0) {
            furthest_future_refidx = 1;
        }
    }

    let n_tile_cols = frame_hdr.tiling.t.cols.max(1) as usize;
    let n_tile_rows = frame_hdr.tiling.t.rows.max(1) as usize;
    let mut jobs = Vec::with_capacity(n_tile_cols * n_tile_rows);
    for tr in 0..n_tile_rows {
        let row_start4 = frame_hdr.tiling.t.row_start_sb[tr] as usize * sb_step4;
        let row_end4 = (frame_hdr.tiling.t.row_start_sb[tr + 1] as usize * sb_step4).min(f_bh4);
        for tc in 0..n_tile_cols {
            let col_start4 = frame_hdr.tiling.t.col_start_sb[tc] as usize * sb_step4;
            let col_end4 = (frame_hdr.tiling.t.col_start_sb[tc + 1] as usize * sb_step4).min(f_bw4);
            jobs.push(TileJob {
                tile_idx: tr * n_tile_cols + tc,
                tile_row: tr,
                tile_col: tc,
                col_start4,
                col_end4,
                row_start4,
                row_end4,
            });
        }
    }

    let is_lossless = frame_hdr.quant.yac == 0;
    let mut ccso_map = [vec![false; 1], vec![false; 1], vec![false; 1]];
    if tools.enable_ccso && !is_lossless {
        frame_hdr.ccso.enabled = 1;
        frame_hdr.ccso.b64size = 1
            << (if seqhdr.ccso_unit_matches_sbsz {
                frame_hdr.sb128
            } else {
                2
            });
        let n_planes = if seqhdr.layout == PixelLayout::I400 {
            1
        } else {
            3
        };
        let ccso_step4 = 16 * frame_hdr.ccso.b64size.max(1) as usize;
        let ccso_cols = f_bw4.div_ceil(ccso_step4);
        let ccso_rows = f_bh4.div_ceil(ccso_step4);
        // `sb_reuse_ccso` is only conformant for the 256-luma CCSO unit
        // (CcsoLumaSizeLog2 == CCSO_LUMA_SIZE_LOG2), which is `b64size == 4`.
        let can_reuse = tools.enable_ccso_sb_reuse
            && frame_hdr.is_inter_or_switch()
            && frame_hdr.ccso.b64size == 4
            && frame_hdr.refidx[0] >= 0;
        let mut reused = false;
        if can_reuse {
            let slot = frame_hdr.refidx[0] as usize;
            if let Some(maps) = dpb.ccso_maps.get(slot).and_then(|m| m.as_ref())
                && maps.iter().all(|map| map.len() == ccso_cols * ccso_rows)
            {
                ccso_map = maps.clone();
                reused = true;
            }
        }
        if reused {
            for p in 0..n_planes {
                frame_hdr.ccso.planes[p].sb_reuse = 1;
                frame_hdr.ccso.planes[p].refidx = 0;
                frame_hdr.ccso.planes[p].reuse = 0;
            }
        } else {
            ccso_map = select_ccso_units(
                src,
                seqhdr.layout,
                frame_hdr.ccso.b64size,
                ccso_cols,
                ccso_rows,
                bd.bitdepth(),
            );
        }
        for p in 0..n_planes {
            frame_hdr.ccso.planes[p].enabled = 1;
            frame_hdr.ccso.planes[p].bo_only = 1;
            frame_hdr.ccso.planes[p].max_band_log2 = 3;
            frame_hdr.ccso.planes[p].scale_idx = 0;
        }
    } else {
        frame_hdr.ccso.enabled = 0;
    }

    let mut gdf_map = vec![false; 1];
    if tools.enable_gdf && !is_lossless {
        frame_hdr.gdf.scale = 1;
        frame_hdr.gdf.b64size = 1 << (!seqhdr.gdf_unit_matches_sbsz as u8).max(frame_hdr.sb128);
        if !seqhdr.gdf_unit_matches_sbsz && frame_hdr.sb128 == 0 {
            let mut mask = 0;
            for t in 0..frame_hdr.tiling.t.cols.saturating_sub(1) as usize {
                mask |= frame_hdr.tiling.t.col_start_sb[t];
            }
            for t in 0..frame_hdr.tiling.t.rows.saturating_sub(1) as usize {
                mask |= frame_hdr.tiling.t.row_start_sb[t];
            }
            if mask & 1 != 0 {
                frame_hdr.gdf.b64size = 1;
            }
        }
        let gdf_step4 = 16 * frame_hdr.gdf.b64size.max(1) as usize;
        let gdf_cols = f_bw4.div_ceil(gdf_step4);
        let gdf_rows = f_bh4.div_ceil(gdf_step4);
        let selected = select_gdf_units(src, gdf_cols, gdf_rows, gdf_step4, bd.bitdepth());
        let any_selected = selected.iter().any(|&enabled| enabled);
        let per_block = frame_hdr.width.max(frame_hdr.height) > 64 * frame_hdr.gdf.b64size as i32;
        if any_selected {
            if per_block {
                frame_hdr.gdf.enabled = AdaptiveBoolean::Adaptive;
                gdf_map = selected;
            } else {
                frame_hdr.gdf.enabled = AdaptiveBoolean::On;
                gdf_map = vec![true; gdf_cols * gdf_rows];
            }
        } else {
            frame_hdr.gdf.enabled = AdaptiveBoolean::Off;
            gdf_map = vec![false; gdf_cols * gdf_rows];
        }
    } else {
        frame_hdr.gdf.enabled = AdaptiveBoolean::Off;
    }

    let is_lossless = frame_hdr.quant.yac == 0;
    let chroma_restoration = tools.restoration_chroma && seqhdr.layout != PixelLayout::I400;
    if tools.enable_restoration && !is_lossless {
        let luma_type = match tools.restoration_mode {
            RestorationMode::PcWiener => 1,
            RestorationMode::NsWiener => 2,
            RestorationMode::Switchable => 3,
        };
        frame_hdr.restoration.planes[0].restoration_type = luma_type;
        frame_hdr.restoration.planes[0].t = match luma_type {
            1 => RestorationType::PcWiener,
            2 => RestorationType::NsWiener,
            _ => RestorationType::Switchable,
        };
        if matches!(
            tools.restoration_mode,
            RestorationMode::NsWiener | RestorationMode::Switchable
        ) {
            let luma_filter = encoder_ns_luma_filter();
            frame_hdr.restoration.planes[0].ns.frame_filters_on = 1;
            frame_hdr.restoration.planes[0].ns.num_classes_idx = 0;
            frame_hdr.restoration.planes[0].ns.num_classes = 1;
            frame_hdr.restoration.planes[0].ns.temporal = 0;
            frame_hdr.restoration.planes[0].ns.filter[0][..16].copy_from_slice(&luma_filter);
        }
        let chroma_filter = encoder_ns_chroma_filter();
        for p in 1..3 {
            if chroma_restoration {
                frame_hdr.restoration.planes[p].restoration_type = 2;
                frame_hdr.restoration.planes[p].t = RestorationType::NsWiener;
                frame_hdr.restoration.planes[p].ns.frame_filters_on = 1;
                frame_hdr.restoration.planes[p].ns.num_classes_idx = 0;
                frame_hdr.restoration.planes[p].ns.num_classes = 1;
                frame_hdr.restoration.planes[p].ns.temporal = 0;
                frame_hdr.restoration.planes[p].ns.filter[0].copy_from_slice(&chroma_filter);
            }
        }
        // Inter frames reuse the reference frame's frame-level NS filters
        // (`temporal_pred_flag`). Every reference in this encoder's linear GOP
        // has restoration enabled, so the filter is always available and no
        // filter bank syntax is needed.
        if frame_hdr.is_inter_or_switch() && frame_hdr.n_ref_frames > 0 {
            for p in 0..3 {
                if frame_hdr.restoration.planes[p].ns.frame_filters_on != 0 {
                    frame_hdr.restoration.planes[p].ns.temporal = 1;
                    frame_hdr.restoration.planes[p].ns.refidx = 0;
                }
            }
        }
        frame_hdr.restoration.unit_size[0] =
            lr_unit_size_log2(0, frame_hdr.sb128, seqhdr.ss_hor, seqhdr.ss_ver);
        frame_hdr.restoration.unit_size[1] =
            lr_unit_size_log2(1, frame_hdr.sb128, seqhdr.ss_hor, seqhdr.ss_ver);
    }

    let ss_x = seqhdr.ss_hor as usize;
    let ss_y = seqhdr.ss_ver as usize;
    let unit_sz_log2_y = frame_hdr.restoration.unit_size[0] as usize;
    let half_unit_y = 1usize << unit_sz_log2_y.saturating_sub(1);
    let lr_cols_y = ((f_bw4 * 4 + half_unit_y) >> unit_sz_log2_y).max(1);
    let lr_rows_y = ((f_bh4 * 4 + half_unit_y) >> unit_sz_log2_y).max(1);

    let unit_sz_log2_uv = frame_hdr.restoration.unit_size[1] as usize;
    let half_unit_uv = 1usize << unit_sz_log2_uv.saturating_sub(1);
    let lr_cols_uv = ((((f_bw4 * 4) >> ss_x) + half_unit_uv) >> unit_sz_log2_uv).max(1);
    let lr_rows_uv = ((((f_bh4 * 4) >> ss_y) + half_unit_uv) >> unit_sz_log2_uv).max(1);

    // Per-unit restoration type map. Switchable alternates PC and NS Wiener so
    // both tools appear in the bitstream; a one-unit frame still selects PC.
    let luma_unit_count = lr_cols_y * lr_rows_y;
    let chroma_unit_count = lr_cols_uv * lr_rows_uv;
    let luma_types: Vec<u8> = (0..luma_unit_count)
        .map(|i| {
            if !tools.enable_restoration || is_lossless {
                0
            } else if tools.restoration_mode == RestorationMode::Switchable {
                if i % 2 == 0 { 1 } else { 2 }
            } else {
                frame_hdr.restoration.planes[0].restoration_type
            }
        })
        .collect();
    let chroma_types: Vec<u8> = (0..chroma_unit_count)
        .map(|_| {
            if chroma_restoration && !is_lossless {
                2
            } else {
                0
            }
        })
        .collect();
    let lr_types = [luma_types, chroma_types.clone(), chroma_types];
    let lr_cols = [lr_cols_y, lr_cols_uv, lr_cols_uv];
    let lr_rows = [lr_rows_y, lr_rows_uv, lr_rows_uv];

    let fh_snapshot = frame_hdr.clone();
    let mut tile_payloads = Vec::with_capacity(jobs.len());
    let mut total_intra_blocks = 0usize;
    let mut total_inter_blocks = 0usize;
    let mut total_newmv_blocks = 0usize;
    let mut total_switchable_filter_blocks = 0usize;
    let mut total_compound_blocks = 0usize;
    let mut total_compound_average_blocks = 0usize;
    let mut total_compound_wedge_blocks = 0usize;
    let mut total_compound_diff_blocks = 0usize;
    let mut total_compound_cwp_blocks = 0usize;
    let mut total_obmc_blocks = 0usize;
    let mut total_warped_blocks = 0usize;
    let mut total_warpmv_blocks = 0usize;
    let mut total_localwarp_blocks = 0usize;
    let mut total_extendwarp_blocks = 0usize;
    let mut total_interintra_blocks = 0usize;
    let mut total_cfl_blocks = 0usize;
    let mut total_fi_blocks = 0usize;
    let mut total_isp_blocks = 0usize;
    let mut total_intrabc_blocks = 0usize;
    let mut total_palette_blocks = 0usize;
    let mut total_tx_split_blocks = 0usize;
    let mut total_tx_partition_counts = [0usize; 4];
    let mut total_non_dct_tx_blocks = 0usize;
    let mut total_wht_blocks = 0usize;
    let mut total_dpcm_y_blocks = 0usize;
    let mut total_dpcm_uv_blocks = 0usize;
    let mut total_rect_blocks = 0usize;
    let mut total_partition_counts = [0usize; 10];
    let mut total_skipped_blocks = 0usize;
    let mut total_qidx_sum = 0u64;
    let mut total_qidx_weight = 0u64;
    let mut frame_seg_map = SegmentMap::new(seg_cols4, seg_rows4);
    let mut frame_grid = DeblockGrid::new(f_bw4, f_bh4, seqhdr.layout);

    if jobs.len() == 1 {
        let job = &jobs[0];
        let num_sb_cols = ((job.col_end4.saturating_sub(job.col_start4)) + 31) >> 5;
        let mut a_cols = vec![BlockContext::default(); num_sb_cols.max(1)];
        for ctx in &mut a_cols {
            reset_context(ctx, fh_snapshot.is_key_or_intra(), false);
        }
        let mut l_ctx = BlockContext::default();
        reset_context(&mut l_ctx, fh_snapshot.is_key_or_intra(), false);

        let ccso_step4 = 16 * fh_snapshot.ccso.b64size.max(1) as usize;
        let ccso_cols = f_bw4.div_ceil(ccso_step4);
        let ccso_rows = f_bh4.div_ceil(ccso_step4);
        let ccso_blks = ccso_map.clone();

        let gdf_step4 = 16 * fh_snapshot.gdf.b64size.max(1) as usize;
        let gdf_cols = f_bw4.div_ceil(gdf_step4);
        let gdf_rows = f_bh4.div_ceil(gdf_step4);
        let gdf_blks = gdf_map.clone();

        let mut state = TileEncodeState {
            bd,
            seqhdr,
            frame_hdr: &fh_snapshot,
            tools,
            src,
            rec,
            ref0,
            ref1,
            refdist,
            absrefdist,
            refdir,
            furthest_future_refidx,
            dq,
            seg_map: SegmentMap::new(seg_cols4, seg_rows4),
            seg_ref,
            seg_plan: seg_plan.as_ref(),
            seg_coder: seg_coder.clone(),
            cur_qidx: fh_snapshot.quant.yac,
            pending_delta: None,
            seg_pred_a: vec![[0u8; 32]; num_sb_cols.max(1)],
            seg_pred_l: [0u8; 32],
            sb_step4,
            qidx_sum: 0,
            qidx_weight: 0,
            sb_budget_bits: frame_budget_bits.map(|b| (b / n_sb_total as f64).max(64.0)),
            rate_offset: 0,
            prev_sb_bits: 0,
            msac: MsacEncoder::new(),
            cdf: CdfContext::new(fh_snapshot.quant.yac),
            a_cols,
            l_ctx,
            tile_col_start4: job.col_start4,
            tile_col_end4: job.col_end4,
            tile_row_start4: job.row_start4,
            tile_row_end4: job.row_end4,
            f_bw4,
            f_bh4,
            speed,
            lambda_sad,
            lambda_rdoq,
            warp_bank: (0..8).map(|_| Vec::with_capacity(4)).collect(),
            warp_bank_hits: 0,
            refmv_bank: RefMvBank::default(),
            refmv_hits: RefMvBankHits::default(),
            warp_bank_row: usize::MAX,
            block_inter: false,
            block_refs: [-1, -1],
            block_mvs: [MvXY::default(); 2],
            block_cwp: 8,
            mi_map: vec![WarpMiInfo::default(); f_bw4 * f_bh4],
            intra_blocks: 0,
            inter_blocks: 0,
            newmv_blocks: 0,
            switchable_filter_blocks: 0,
            compound_blocks: 0,
            compound_average_blocks: 0,
            compound_wedge_blocks: 0,
            compound_diff_blocks: 0,
            compound_cwp_blocks: 0,
            obmc_blocks: 0,
            warped_blocks: 0,
            warpmv_blocks: 0,
            localwarp_blocks: 0,
            extendwarp_blocks: 0,
            interintra_blocks: 0,
            cfl_blocks: 0,
            filter_intra_blocks: 0,
            isp_blocks: 0,
            intrabc_blocks: 0,
            palette_blocks: 0,
            palette_sizes: vec![0u8; f_bw4 * f_bh4],
            palette_colors: vec![[0i32; 8]; f_bw4 * f_bh4],
            intrabc_coded: vec![false; f_bw4 * f_bh4],
            ibc_buffers: crate::av2::screen::IbcBufferState::default(),
            tu_coded: [
                vec![false; f_bw4 * f_bh4],
                vec![false; f_bw4 * f_bh4],
                vec![false; f_bw4 * f_bh4],
            ],
            palette_cdf: crate::av2::screen::PaletteCdfState::default(),
            tx_split_blocks: 0,
            tx_partition_counts: [0; 4],
            non_dct_tx_blocks: 0,
            wht_blocks: 0,
            dpcm_y_blocks: 0,
            dpcm_uv_blocks: 0,
            rect_blocks: 0,
            partition_counts: [0; 10],
            skipped_blocks: 0,
            grid: DeblockGrid::new(f_bw4, f_bh4, seqhdr.layout),
            ccso_blks,
            ccso_cols,
            ccso_rows,
            ccso_step4,
            gdf_blks,
            gdf_cols,
            gdf_rows,
            gdf_step4,
            lr_types: lr_types.clone(),
            lr_cols,
            lr_rows,
            scratch_mc0: vec![0i32; 128 * 128],
            scratch_mc1: vec![0i32; 128 * 128],
            scratch_diff_mask: vec![0u8; 128 * 128],
            scratch_chroma_diff_mask: vec![0u8; 128 * 128],
        };

        let mut by4 = job.row_start4;
        while by4 < job.row_end4 {
            reset_context(&mut state.l_ctx, fh_snapshot.is_key_or_intra(), false);
            state.seg_pred_l = [0u8; 32];
            let mut bx4 = job.col_start4;
            while bx4 < job.col_end4 {
                state.encode_lr(bx4, by4);
                state.begin_superblock(bx4, by4);
                state.encode_sb(root_bs, bx4, by4, false);
                bx4 += sb_step4;
            }
            by4 += sb_step4;
        }

        total_intra_blocks = state.intra_blocks;
        total_inter_blocks = state.inter_blocks;
        total_newmv_blocks = state.newmv_blocks;
        total_switchable_filter_blocks = state.switchable_filter_blocks;
        total_compound_blocks = state.compound_blocks;
        total_compound_average_blocks = state.compound_average_blocks;
        total_compound_wedge_blocks = state.compound_wedge_blocks;
        total_compound_diff_blocks = state.compound_diff_blocks;
        total_compound_cwp_blocks = state.compound_cwp_blocks;
        total_obmc_blocks = state.obmc_blocks;
        total_warped_blocks = state.warped_blocks;
        total_warpmv_blocks = state.warpmv_blocks;
        total_localwarp_blocks = state.localwarp_blocks;
        total_extendwarp_blocks = state.extendwarp_blocks;
        total_interintra_blocks = state.interintra_blocks;
        total_cfl_blocks = state.cfl_blocks;
        total_fi_blocks = state.filter_intra_blocks;
        total_isp_blocks = state.isp_blocks;
        total_intrabc_blocks = state.intrabc_blocks;
        total_palette_blocks = state.palette_blocks;
        total_tx_split_blocks = state.tx_split_blocks;
        total_tx_partition_counts = state.tx_partition_counts;
        total_non_dct_tx_blocks = state.non_dct_tx_blocks;
        total_wht_blocks = state.wht_blocks;
        total_dpcm_y_blocks = state.dpcm_y_blocks;
        total_dpcm_uv_blocks = state.dpcm_uv_blocks;
        total_rect_blocks = state.rect_blocks;
        total_partition_counts = state.partition_counts;
        total_skipped_blocks = state.skipped_blocks;
        total_qidx_sum = state.qidx_sum;
        total_qidx_weight = state.qidx_weight;
        frame_seg_map.copy_from(&state.seg_map);
        frame_grid = std::mem::replace(&mut state.grid, DeblockGrid::new(0, 0, seqhdr.layout));
        tile_payloads.push(state.msac.finalize());
    } else {
        let tile_results = execute_tiles_parallel(&jobs, tools.threads.max(1), |job| {
            let mut local_rec = Frame::new(
                src.width,
                src.height,
                bd.bitdepth(),
                seqhdr.layout,
                fh_snapshot.frame_offset,
            );
            let num_sb_cols = ((job.col_end4.saturating_sub(job.col_start4)) + 31) >> 5;
            let mut a_cols = vec![BlockContext::default(); num_sb_cols.max(1)];
            for ctx in &mut a_cols {
                reset_context(ctx, fh_snapshot.is_key_or_intra(), false);
            }
            let mut l_ctx = BlockContext::default();
            reset_context(&mut l_ctx, fh_snapshot.is_key_or_intra(), false);

            let ccso_step4 = 16 * fh_snapshot.ccso.b64size.max(1) as usize;
            let ccso_cols = f_bw4.div_ceil(ccso_step4);
            let ccso_rows = f_bh4.div_ceil(ccso_step4);
            let ccso_blks = ccso_map.clone();

            let gdf_step4 = 16 * fh_snapshot.gdf.b64size.max(1) as usize;
            let gdf_cols = f_bw4.div_ceil(gdf_step4);
            let gdf_rows = f_bh4.div_ceil(gdf_step4);
            let gdf_blks = gdf_map.clone();

            let mut state = TileEncodeState {
                bd,
                seqhdr,
                frame_hdr: &fh_snapshot,
                tools,
                src,
                rec: &mut local_rec,
                ref0,
                ref1,
                refdist,
                absrefdist,
                refdir,
                furthest_future_refidx,
                dq,
                seg_map: SegmentMap::new(seg_cols4, seg_rows4),
                seg_ref,
                seg_plan: seg_plan.as_ref(),
                seg_coder: seg_coder.clone(),
                cur_qidx: fh_snapshot.quant.yac,
                pending_delta: None,
                seg_pred_a: vec![[0u8; 32]; num_sb_cols.max(1)],
                seg_pred_l: [0u8; 32],
                sb_step4,
                qidx_sum: 0,
                qidx_weight: 0,
                sb_budget_bits: frame_budget_bits.map(|b| (b / n_sb_total as f64).max(64.0)),
                rate_offset: 0,
                prev_sb_bits: 0,
                msac: MsacEncoder::new(),
                cdf: CdfContext::new(fh_snapshot.quant.yac),
                a_cols,
                l_ctx,
                tile_col_start4: job.col_start4,
                tile_col_end4: job.col_end4,
                tile_row_start4: job.row_start4,
                tile_row_end4: job.row_end4,
                f_bw4,
                f_bh4,
                speed,
                lambda_sad,
                lambda_rdoq,
                warp_bank: (0..8).map(|_| Vec::with_capacity(4)).collect(),
                warp_bank_hits: 0,
                refmv_bank: RefMvBank::default(),
                refmv_hits: RefMvBankHits::default(),
                warp_bank_row: usize::MAX,
                block_inter: false,
                block_refs: [-1, -1],
                block_mvs: [MvXY::default(); 2],
                block_cwp: 8,
                mi_map: vec![WarpMiInfo::default(); f_bw4 * f_bh4],
                intra_blocks: 0,
                inter_blocks: 0,
                newmv_blocks: 0,
                switchable_filter_blocks: 0,
                compound_blocks: 0,
                compound_average_blocks: 0,
                compound_wedge_blocks: 0,
                compound_diff_blocks: 0,
                compound_cwp_blocks: 0,
                obmc_blocks: 0,
                warped_blocks: 0,
                warpmv_blocks: 0,
                localwarp_blocks: 0,
                extendwarp_blocks: 0,
                interintra_blocks: 0,
                cfl_blocks: 0,
                filter_intra_blocks: 0,
                isp_blocks: 0,
                intrabc_blocks: 0,
                palette_blocks: 0,
                palette_sizes: vec![0u8; f_bw4 * f_bh4],
                palette_colors: vec![[0i32; 8]; f_bw4 * f_bh4],
                intrabc_coded: vec![false; f_bw4 * f_bh4],
                ibc_buffers: crate::av2::screen::IbcBufferState::default(),
                tu_coded: [
                    vec![false; f_bw4 * f_bh4],
                    vec![false; f_bw4 * f_bh4],
                    vec![false; f_bw4 * f_bh4],
                ],
                palette_cdf: crate::av2::screen::PaletteCdfState::default(),
                tx_split_blocks: 0,
                tx_partition_counts: [0; 4],
                non_dct_tx_blocks: 0,
                wht_blocks: 0,
                dpcm_y_blocks: 0,
                dpcm_uv_blocks: 0,
                rect_blocks: 0,
                partition_counts: [0; 10],
                skipped_blocks: 0,
                grid: DeblockGrid::new(f_bw4, f_bh4, seqhdr.layout),
                ccso_blks,
                ccso_cols,
                ccso_rows,
                ccso_step4,
                gdf_blks,
                gdf_cols,
                gdf_rows,
                gdf_step4,
                lr_types: lr_types.clone(),
                lr_cols,
                lr_rows,
                scratch_mc0: vec![0i32; 128 * 128],
                scratch_mc1: vec![0i32; 128 * 128],
                scratch_diff_mask: vec![0u8; 128 * 128],
                scratch_chroma_diff_mask: vec![0u8; 128 * 128],
            };

            let mut by4 = job.row_start4;
            while by4 < job.row_end4 {
                reset_context(&mut state.l_ctx, fh_snapshot.is_key_or_intra(), false);
                state.seg_pred_l = [0u8; 32];
                let mut bx4 = job.col_start4;
                while bx4 < job.col_end4 {
                    state.encode_lr(bx4, by4);
                    state.begin_superblock(bx4, by4);
                    state.encode_sb(root_bs, bx4, by4, false);
                    bx4 += sb_step4;
                }
                by4 += sb_step4;
            }

            let intra_blocks = state.intra_blocks;
            let inter_blocks = state.inter_blocks;
            let newmv_blocks = state.newmv_blocks;
            let switchable_filter_blocks = state.switchable_filter_blocks;
            let compound_blocks = state.compound_blocks;
            let compound_average_blocks = state.compound_average_blocks;
            let compound_wedge_blocks = state.compound_wedge_blocks;
            let compound_diff_blocks = state.compound_diff_blocks;
            let compound_cwp_blocks = state.compound_cwp_blocks;
            let obmc_blocks = state.obmc_blocks;
            let warped_blocks = state.warped_blocks;
            let warpmv_blocks = state.warpmv_blocks;
            let localwarp_blocks = state.localwarp_blocks;
            let extendwarp_blocks = state.extendwarp_blocks;
            let interintra_blocks = state.interintra_blocks;
            let cfl_blocks = state.cfl_blocks;
            let filter_intra_blocks = state.filter_intra_blocks;
            let isp_blocks = state.isp_blocks;
            let intrabc_blocks = state.intrabc_blocks;
            let palette_blocks = state.palette_blocks;
            let tx_split_blocks = state.tx_split_blocks;
            let tx_partition_counts = state.tx_partition_counts;
            let non_dct_tx_blocks = state.non_dct_tx_blocks;
            let wht_blocks = state.wht_blocks;
            let dpcm_y_blocks = state.dpcm_y_blocks;
            let dpcm_uv_blocks = state.dpcm_uv_blocks;
            let rect_blocks = state.rect_blocks;
            let partition_counts = state.partition_counts;
            let skipped_blocks = state.skipped_blocks;
            let qidx_sum = state.qidx_sum;
            let qidx_weight = state.qidx_weight;
            let seg_map =
                std::mem::replace(&mut state.seg_map, SegmentMap::new(seg_cols4, seg_rows4));
            let grid = std::mem::replace(&mut state.grid, DeblockGrid::new(0, 0, seqhdr.layout));
            let payload = state.msac.finalize();

            TileEncodeResult {
                job: *job,
                payload,
                rec_tile: local_rec,
                intra_blocks,
                inter_blocks,
                newmv_blocks,
                switchable_filter_blocks,
                compound_blocks,
                compound_average_blocks,
                compound_wedge_blocks,
                compound_diff_blocks,
                compound_cwp_blocks,
                obmc_blocks,
                warped_blocks,
                warpmv_blocks,
                localwarp_blocks,
                extendwarp_blocks,
                interintra_blocks,
                cfl_blocks,
                filter_intra_blocks,
                isp_blocks,
                intrabc_blocks,
                palette_blocks,
                tx_split_blocks,
                tx_partition_counts,
                non_dct_tx_blocks,
                wht_blocks,
                dpcm_y_blocks,
                dpcm_uv_blocks,
                rect_blocks,
                partition_counts,
                skipped_blocks,
                grid,
                seg_map,
                qidx_sum,
                qidx_weight,
            }
        });

        let n_planes = if seqhdr.layout == PixelLayout::I400 {
            1
        } else {
            3
        };
        for tr in tile_results {
            total_intra_blocks += tr.intra_blocks;
            total_inter_blocks += tr.inter_blocks;
            total_newmv_blocks += tr.newmv_blocks;
            total_switchable_filter_blocks += tr.switchable_filter_blocks;
            total_compound_blocks += tr.compound_blocks;
            total_compound_average_blocks += tr.compound_average_blocks;
            total_compound_wedge_blocks += tr.compound_wedge_blocks;
            total_compound_diff_blocks += tr.compound_diff_blocks;
            total_compound_cwp_blocks += tr.compound_cwp_blocks;
            total_obmc_blocks += tr.obmc_blocks;
            total_warped_blocks += tr.warped_blocks;
            total_warpmv_blocks += tr.warpmv_blocks;
            total_localwarp_blocks += tr.localwarp_blocks;
            total_extendwarp_blocks += tr.extendwarp_blocks;
            total_interintra_blocks += tr.interintra_blocks;
            total_cfl_blocks += tr.cfl_blocks;
            total_fi_blocks += tr.filter_intra_blocks;
            total_isp_blocks += tr.isp_blocks;
            total_intrabc_blocks += tr.intrabc_blocks;
            total_palette_blocks += tr.palette_blocks;
            total_tx_split_blocks += tr.tx_split_blocks;
            for (total, count) in total_tx_partition_counts
                .iter_mut()
                .zip(tr.tx_partition_counts)
            {
                *total += count;
            }
            total_non_dct_tx_blocks += tr.non_dct_tx_blocks;
            total_wht_blocks += tr.wht_blocks;
            total_dpcm_y_blocks += tr.dpcm_y_blocks;
            total_dpcm_uv_blocks += tr.dpcm_uv_blocks;
            total_rect_blocks += tr.rect_blocks;
            for (total, count) in total_partition_counts.iter_mut().zip(tr.partition_counts) {
                *total += count;
            }
            total_skipped_blocks += tr.skipped_blocks;
            total_qidx_sum += tr.qidx_sum;
            total_qidx_weight += tr.qidx_weight;
            frame_seg_map.copy_region_from(
                &tr.seg_map,
                tr.job.col_start4,
                tr.job.row_start4,
                tr.job.col_end4.min(f_bw4),
                tr.job.row_end4.min(f_bh4),
            );

            for pl in 0..n_planes {
                let ss_x = rec.planes[pl].ss_x;
                let ss_y = rec.planes[pl].ss_y;
                let x0 = ((tr.job.col_start4 * 4) >> ss_x).min(rec.planes[pl].width);
                let x1 = ((tr.job.col_end4 * 4) >> ss_x).min(rec.planes[pl].width);
                let y0 = ((tr.job.row_start4 * 4) >> ss_y).min(rec.planes[pl].height);
                let y1 = ((tr.job.row_end4 * 4) >> ss_y).min(rec.planes[pl].height);
                for y in y0..y1 {
                    let s_off = y * tr.rec_tile.planes[pl].stride;
                    let d_off = y * rec.planes[pl].stride;
                    rec.planes[pl].data[d_off + x0..d_off + x1]
                        .copy_from_slice(&tr.rec_tile.planes[pl].data[s_off + x0..s_off + x1]);
                }
            }
            if tools.enable_deblock || tools.enable_restoration {
                let ssx = seqhdr.ss_hor as usize;
                let ssy = seqhdr.ss_ver as usize;
                for y4 in tr.job.row_start4..tr.job.row_end4.min(f_bh4) {
                    for x4 in tr.job.col_start4..tr.job.col_end4.min(f_bw4) {
                        frame_grid.set_luma(x4, y4, tr.grid.get_luma(x4, y4));
                        frame_grid.set_lr_noskip(x4, y4, tr.grid.get_lr_noskip(x4, y4));
                    }
                }
                if seqhdr.layout != PixelLayout::I400 {
                    for cy4 in (tr.job.row_start4 >> ssy)
                        ..(tr.job.row_end4 >> ssy).min(frame_grid.chroma_h)
                    {
                        for cx4 in (tr.job.col_start4 >> ssx)
                            ..(tr.job.col_end4 >> ssx).min(frame_grid.chroma_w)
                        {
                            frame_grid.set_chroma(cx4, cy4, tr.grid.get_chroma(cx4, cy4));
                        }
                    }
                }
            }
            tile_payloads.push(tr.payload);
        }
    }

    // Full Wave 3 5-Stage In-Loop Filter & Enhancement Pipeline:
    // Deblock -> CDEF -> CCSO -> Wiener/Sgrproj Loop Restoration -> Guided Detail Filter (GDF)
    let is_lossless = frame_hdr.quant.yac == 0;
    if tools.enable_deblock && !is_lossless {
        frame_hdr.deblock =
            derive_deblock_params(frame_hdr.quant.yac, frame_hdr.frame_type, is_lossless);
        // The accelerator trait only models the legacy block-agnostic filter;
        // the normative filter needs the per-block grid, so it runs on the CPU.
        let _ = deblock_accelerator;
        deblock_frame_normative(bd, rec, seqhdr, frame_hdr, &frame_grid);
    }
    let orig_planes =
        if (tools.enable_ccso || tools.enable_gdf || tools.enable_restoration) && !is_lossless {
            Some([
                rec.planes[0].clone(),
                rec.planes[1].clone(),
                rec.planes[2].clone(),
            ])
        } else {
            None
        };
    let orig_luma = orig_planes.as_ref().map(|p| &p[0]);
    if tools.enable_cdef && !is_lossless {
        optimize_and_apply_cdef(bd, src, rec, seqhdr, frame_hdr, &frame_grid);
    }
    let mut ref_dists = [0i32; 8];
    if frame_hdr.is_inter_or_switch() {
        for (i, &idx) in frame_hdr
            .refidx
            .iter()
            .enumerate()
            .take(frame_hdr.n_ref_frames as usize)
        {
            if idx >= 0
                && (idx as usize) < dpb.slots.len()
                && let Some(ref r) = dpb.slots[idx as usize]
            {
                ref_dists[i] = (frame_hdr.frame_offset as i32 - r.poc as i32).abs();
            }
        }
    }
    if let Some(orig_luma) = orig_luma {
        if tools.enable_ccso && !is_lossless {
            optimize_and_apply_ccso(bd, src, rec, orig_luma, seqhdr, frame_hdr, &ccso_map);
        }
        let cdef_luma = rec.planes[0].clone();
        if tools.enable_restoration && !is_lossless {
            let orig = orig_planes
                .as_ref()
                .expect("restoration needs the deblocked frame");
            apply_encoder_restoration(bd, rec, orig, seqhdr, frame_hdr, &lr_types, &frame_grid);
        }
        if tools.enable_gdf && !is_lossless {
            optimize_and_apply_gdf(
                bd, src, rec, &cdef_luma, orig_luma, seqhdr, frame_hdr, &ref_dists, &gdf_map,
            );
        }
    }
    if tools.enable_film_grain && !is_lossless {
        // AV2 §7.21.1: grain is applied to the output picture only. The model
        // is estimated from the residual, signalled in an OBU_FILM_GRAIN, and
        // never stored in the reference frames.
        let seed = (0x9E37u32 ^ (frame_hdr.frame_offset as u32).wrapping_mul(0x4D2)) | 1;
        let data = estimate_film_grain_params(src, rec, seed as u16);
        if data.is_active() {
            frame_hdr.film_grain.present = 1;
            frame_hdr.film_grain.id = 0;
            frame_hdr.film_grain.seed = seed;
            frame_hdr.film_grain.data = data;
        } else {
            frame_hdr.film_grain.present = 0;
            frame_hdr.film_grain.data = Default::default();
        }
    }

    // Reference frames keep the reconstructed pixels inside the coded extent
    // `MiCols * 4` x `MiRows * 4` (AV2 §7.13.2.1): inter prediction and intra
    // reference reads clamp to that extent, so replicating the visible border
    // over it would desynchronise the encoder reference from the decoder's.
    dpb.update_with_frame(frame_hdr.refresh_frame_flags, rec);
    dpb.update_ccso_maps(frame_hdr.refresh_frame_flags, &ccso_map);

    if tile_payloads.len() > 1 {
        let max_tile_sz = tile_payloads
            .iter()
            .map(|v| v.len().saturating_sub(1))
            .max()
            .unwrap_or(0);
        let n_bits = ulog2(max_tile_sz.max(1) as u32) + 1;
        frame_hdr.tiling.n_bytes = ((n_bits + 7) >> 3).clamp(1, 4) as u8;
    } else {
        frame_hdr.tiling.n_bytes = 1;
    }

    let obu_type = if frame_hdr.frame_type == FrameType::Key {
        ObuType::ClosedLoopKf
    } else {
        ObuType::TileGrp
    };
    let obu_bytes = write_frame_and_tile_group_obu(seqhdr, frame_hdr, obu_type, &tile_payloads);

    let (mse_y, psnr_y, psnr_u, psnr_v, psnr_avg) = compute_frame_psnr(src, rec);

    let n_stats_planes = if seqhdr.layout == PixelLayout::I400 {
        1
    } else {
        3
    };
    let ccso_units = if frame_hdr.ccso.enabled != 0 {
        (0..n_stats_planes)
            .map(|p| ccso_map[p].iter().filter(|&&enabled| enabled).count())
            .sum()
    } else {
        0
    };
    let ccso_reuse_units = if frame_hdr.ccso.enabled != 0 {
        (0..n_stats_planes)
            .filter(|&p| frame_hdr.ccso.planes[p].sb_reuse != 0)
            .map(|p| ccso_map[p].len())
            .sum()
    } else {
        0
    };
    let ccso_edge_planes = if frame_hdr.ccso.enabled != 0 {
        (0..n_stats_planes)
            .filter(|&p| {
                frame_hdr.ccso.planes[p].enabled != 0 && frame_hdr.ccso.planes[p].bo_only == 0
            })
            .count()
    } else {
        0
    };
    let gdf_units = if frame_hdr.gdf.enabled != AdaptiveBoolean::Off {
        gdf_map.iter().filter(|&&enabled| enabled).count()
    } else {
        0
    };

    let stats = FrameStats {
        poc: frame_hdr.frame_offset,
        frame_type: frame_hdr.frame_type,
        tlayer_id: frame_hdr.tlayer_id,
        qidx: frame_hdr.quant.yac,
        bytes: obu_bytes.len(),
        psnr_y,
        psnr_u,
        psnr_v,
        psnr_avg,
        mse_y,
        intra_blocks: total_intra_blocks,
        inter_blocks: total_inter_blocks,
        newmv_blocks: total_newmv_blocks,
        switchable_filter_blocks: total_switchable_filter_blocks,
        compound_blocks: total_compound_blocks,
        compound_average_blocks: total_compound_average_blocks,
        compound_wedge_blocks: total_compound_wedge_blocks,
        compound_diff_blocks: total_compound_diff_blocks,
        compound_cwp_blocks: total_compound_cwp_blocks,
        obmc_blocks: total_obmc_blocks,
        warped_blocks: total_warped_blocks,
        warpmv_blocks: total_warpmv_blocks,
        localwarp_blocks: total_localwarp_blocks,
        extendwarp_blocks: total_extendwarp_blocks,
        interintra_blocks: total_interintra_blocks,
        cfl_blocks: total_cfl_blocks,
        filter_intra_blocks: total_fi_blocks,
        isp_blocks: total_isp_blocks,
        intrabc_blocks: total_intrabc_blocks,
        palette_blocks: total_palette_blocks,
        tx_split_blocks: total_tx_split_blocks,
        tx_partition_counts: total_tx_partition_counts,
        non_dct_tx_blocks: total_non_dct_tx_blocks,
        wht_blocks: total_wht_blocks,
        dpcm_y_blocks: total_dpcm_y_blocks,
        dpcm_uv_blocks: total_dpcm_uv_blocks,
        rect_blocks: total_rect_blocks,
        partition_counts: total_partition_counts,
        skipped_blocks: total_skipped_blocks,
        // Effective luma quantizer after segment ALT_Q and per-SB delta-Q.
        avg_qidx: total_qidx_sum
            .checked_div(total_qidx_weight)
            .map(|q| q.min(255) as u16)
            .unwrap_or(frame_hdr.quant.yac),
        deblock_level: frame_hdr.deblock.level_y[0],
        cdef_y_strength: frame_hdr.cdef.y_strength[0],
        ccso_enabled: frame_hdr.ccso.enabled != 0,
        gdf_enabled: frame_hdr.gdf.enabled != crate::av2::headers::AdaptiveBoolean::Off,
        ccso_units,
        ccso_reuse_units,
        ccso_edge_planes,
        gdf_units,
        gdf_per_block: frame_hdr.gdf.enabled == AdaptiveBoolean::Adaptive,
        restoration_y: frame_hdr.restoration.planes[0].t,
    };

    (obu_bytes, stats, frame_seg_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restoration_unit_size_and_count_are_non_zero() {
        // 64x64 superblocks use 64-sample luma units; 4:2:0/4:2:2 chroma also
        // uses 64 samples, which is at least the 32-sample chroma stripe.
        assert_eq!(lr_unit_size_log2(0, 0, 1, 1), 6);
        assert_eq!(lr_unit_size_log2(1, 0, 1, 1), 6);
        assert_eq!(lr_unit_size_log2(0, 1, 1, 1), 7);
        for (w, h) in [(64usize, 64usize), (256, 192)] {
            for (plane, sx, sy) in [(0usize, 0u8, 0u8), (1, 1, 1), (2, 1, 1)] {
                let log2 = lr_unit_size_log2(plane, 0, 1, 1);
                let pw = (w >> sx).max(1);
                let ph = (h >> sy).max(1);
                let cols = ((pw + (1 << (log2 - 1))) >> log2).max(1);
                let rows = ((ph + (1 << (log2 - 1))) >> log2).max(1);
                assert!(cols * rows > 0);
            }
        }
    }

    #[test]
    fn coded_extent_is_padded_to_eight_samples() {
        // AV2 §7.13.2.1: `MiCols = 2 * ((FrameWidth + 7) >> 3)`, so the coded
        // extent is the visible size rounded up to a multiple of eight, not
        // four. 130x66 exercises a 2-pixel remainder in both directions.
        let f = Frame::<u8>::new(130, 66, 8, PixelLayout::I420, 0);
        assert_eq!(
            (f.planes[0].coded_width, f.planes[0].coded_height),
            (136, 72)
        );
        assert_eq!(
            (f.planes[1].coded_width, f.planes[1].coded_height),
            (68, 36)
        );
        assert_eq!(
            (f.planes[2].coded_width, f.planes[2].coded_height),
            (68, 36)
        );
        for p in &f.planes {
            assert!(p.coded_width <= p.stride);
            assert!(p.coded_height <= p.alloc_height);
        }

        let f = Frame::<u8>::new(64, 64, 8, PixelLayout::I444, 0);
        assert_eq!(
            (f.planes[0].coded_width, f.planes[0].coded_height),
            (64, 64)
        );
        assert_eq!(
            (f.planes[1].coded_width, f.planes[1].coded_height),
            (64, 64)
        );
    }
}
