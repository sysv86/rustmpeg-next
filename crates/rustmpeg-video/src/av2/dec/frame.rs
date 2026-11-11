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

//! AV2 tile and coding-block decoder.
//!
//! The block syntax mirrors the `av2` encoder so that every stream the encoder
//! produces can be decoded bit-exactly. Reconstruction reuses the encoder's
//! intra predictors, motion compensation, dequantizer, and inverse transform,
//! so equal syntax implies equal samples.

use crate::av2::ccso::ccso_frame_normative;
use crate::av2::cdef::cdef_frame_normative;
use crate::av2::cdf::CdfContext;
use crate::av2::cfl::{compute_cfl_ac_luma, predict_cfl_plane};
use crate::av2::deblock::{
    BlockDeblockInfo, DeblockGrid, deblock_frame_normative as deblock_frame,
};
use crate::av2::dec::msac::MsacDecoder;
use crate::av2::dec::syntax::{
    CoefDecodeParams, PaletteBlock, SegSyntax, decode_cfl_alphas, decode_coefs,
    decode_delta_qindex, decode_intra_uv_mode, decode_intra_y_mode, decode_palette_mode_info,
    decode_partition_decision, decode_segment_id,
};
use crate::av2::entropy::{
    BlockContext, gather_inter_neighbors, get_filter_ctx_for_block, reset_context,
    update_partition_ctx,
};
use crate::av2::error::{Av2Error, Result};
use crate::av2::frame::{Frame, ReferenceDpb, SPEC_SIZE_GROUP, mi_size_index};
use crate::av2::gdf::gdf_frame_normative;
use crate::av2::headers::{
    AdaptiveBoolean, FilterMode, FrameHeader, MAX_SEGMENTS, PixelLayout, SequenceHeader, TxfmMode,
};
use crate::av2::inter::{
    CompoundMiInfo, find_compound_mv_stack, find_mv_stack, scale_projected_mv,
};
use crate::av2::intops::iclip;
use crate::av2::ipred::{
    blend_interintra, interintra_intra_mode, interintra_mask, predict_filter_intra_block,
    predict_intra_block, prepare_intra_edges, prepare_intra_edges_mrl,
};
use crate::av2::levels::{
    BlockPartition, BlockSize, CompInterType, FilterIntraMode, InterPredMode, MvXY, txtp,
};
use crate::av2::me::{
    blend_compound_prep, mc_predict_block, mc_predict_block_prep, mv_clamp_to_integer,
    read_mv_magnitude, read_mv_residual,
};
use crate::av2::partition::{
    PartitionSyntaxConfig, chroma_offset_for_partition, h_partition_midsize,
    init_allowed_partitions, partition_4way_sizes, partition_implied, partition_subsize,
};
use crate::av2::pixel::BitDepth;
use crate::av2::quantizer::dq_lookup;
use crate::av2::restoration::{apply_ns_wiener_plane, apply_pc_wiener_units};
use crate::av2::screen::{
    IbcBufferState, MAX_PALETTE_COLORS, apply_intrabc_block_bounded, lower_mv_precision_to_integer,
};
use crate::av2::segment::SegmentMap;
use crate::av2::tables::{
    BLOCK_DIMENSIONS, MAX_TXFM_SIZE_FOR_BS, SS_BS, TX_PART_GROUP, TXFM_DIMENSIONS,
};
use crate::av2::transform::inv_txfm_add;
use crate::av2::warp::{
    DEFAULT_WARP_PARAMS, MAX_WARP_REF_CANDIDATES, MV_BORDER, MV_LOW, MV_UPP, WARPEDMODEL_PREC_BITS,
    WarpMiInfo, block_warp, build_warp_param_stack, extend_warp_params, find_warp_samples,
    generate_points_from_corners, get_warp_translation, reduce_warp_model, round2_signed,
    setup_shear, warp_bank_insert, warp_estimation, warp_has_ref_neighbour, warp_local_context,
    warp_mode_context, warp_motion_vector, warp_mv_at,
};
use crate::av2::wedge::{
    WEDGE_ANGLE_DIST_TO_IDX, WEDGE_TABLES, compute_diff_wtd_mask_intermediate,
};

const MAX_TX_AREA: usize = 32 * 32;

/// Return value of [`decode_frame`]: the reconstructed picture, the saved
/// segment map, and the current frame's `CcsoBlks` map for `sb_reuse_ccso`.
type DecodedFrameParts<P> = (Frame<P>, SegmentMap, [Vec<bool>; 3]);

/// `Wedge_Bits[MiSize]` (AV2 Spec v1.0.0 §9.2): non-zero exactly for blocks
/// whose width and height are in `8..=64` samples, i.e. where the wedge
/// codebook has an entry.
#[inline]
fn wedge_bits(bw4: usize, bh4: usize) -> u8 {
    let w = bw4 * 4;
    let h = bh4 * 4;
    if (8..=64).contains(&w) && (8..=64).contains(&h) {
        4
    } else {
        0
    }
}

/// `is_thin_4xn_nx4_block()` (AV2 Spec v1.0.0 §5.20.7.16).
#[inline]
fn is_thin_4xn_nx4_block(bw4: usize, bh4: usize) -> bool {
    let w = bw4 * 4;
    let h = bh4 * 4;
    (w == 4 && h >= 16) || (h == 4 && w >= 16)
}

/// Normative dequantizer lookup including `qidx > 255` for 10/12-bit content.
///
/// `dq_lookup` clamps at 255, while AV2 `MaxQ` is `255 + 24 * (BitDepth - 8)`
/// (AV2 Spec v1.0.0 Table 5.2). The base table is the normative `Ac_Qlookup`.
#[inline]
fn dq_lookup_qmax(qidx: i32) -> i32 {
    if qidx <= 255 {
        return dq_lookup(qidx);
    }
    static DQ_LOOKUP_TBL: [u8; 24] = [
        40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 54, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76,
        78,
    ];
    let q = qidx - 1;
    (DQ_LOOKUP_TBL[(q % 24) as usize] as i32) << (q / 24)
}

/// `get_q(qindex, delta)` for one plane (AV2 Spec v1.0.0 §7.14.2).
#[inline]
fn get_q_step(qindex: i32, delta: i8, max_q: i32) -> u32 {
    let q = if qindex == 0 && delta <= 0 {
        0
    } else {
        iclip(qindex + delta as i32, 1, max_q)
    };
    dq_lookup_qmax(q) as u32
}

/// Recomputes the per-segment dequantizer tables for `qidx`
/// (`init_quant_tables` in AV2 Spec v1.0.0 §5.18.1; dav2d `init_quant_tables`).
fn init_segment_quant_tables(
    fh: &FrameHeader,
    qidx: i32,
    bit_depth: u8,
    dq: &mut [[[u32; 2]; 3]; MAX_SEGMENTS],
) {
    let max_q = crate::av2::headers::max_q_bits(bit_depth);
    let n_seg = if fh.segmentation.enabled != 0 {
        MAX_SEGMENTS
    } else {
        1
    };
    for i in 0..n_seg {
        let yac = if fh.segmentation.enabled != 0 {
            iclip(qidx + fh.segmentation.d.delta_q[i] as i32, 0, max_q)
        } else {
            iclip(qidx, 0, max_q)
        };
        dq[i][0][0] = get_q_step(yac, fh.quant.ydc_delta, max_q);
        dq[i][0][1] = get_q_step(yac, 0, max_q);
        dq[i][1][0] = get_q_step(yac, fh.quant.udc_delta, max_q);
        dq[i][1][1] = get_q_step(yac, fh.quant.uac_delta, max_q);
        dq[i][2][0] = get_q_step(yac, fh.quant.vdc_delta, max_q);
        dq[i][2][1] = get_q_step(yac, fh.quant.vac_delta, max_q);
    }
}

/// Decodes one frame from its tile payloads.
///
/// Returns the reconstructed frame and its 4x4 segment map
/// (`SavedSegmentIds` for later temporal prediction, AV2 Spec v1.0.0 §5.18.1).
pub fn decode_frame<BD: BitDepth>(
    bd: BD,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    tiles: &[&[u8]],
    dpb: &ReferenceDpb<BD::Pixel>,
    n_cols: usize,
    prev_segmap: Option<&SegmentMap>,
) -> Result<DecodedFrameParts<BD::Pixel>> {
    validate_tools(seq, fh)?;
    if fh.width <= 0 || fh.height <= 0 || fh.width > 16384 || fh.height > 16384 {
        return Err(Av2Error::BitstreamError(format!(
            "frame dimensions {}x{} are outside the supported range",
            fh.width, fh.height
        )));
    }

    let mut rec = Frame::new(
        fh.width.max(1) as usize,
        fh.height.max(1) as usize,
        bd.bitdepth(),
        seq.layout,
        fh.frame_offset,
    );
    let mut dq = [[[0u32; 2]; 3]; MAX_SEGMENTS];
    init_segment_quant_tables(fh, fh.quant.yac as i32, bd.bitdepth(), &mut dq);

    let mut ref_frames: [Option<&Frame<BD::Pixel>>; 7] = [None; 7];
    let mut refdist = [0i32; 7];
    let mut absrefdist = [0i32; 7];
    let mut refdir = [0i8; 7];
    let mut furthest_future_refidx = -2i8;
    if fh.is_inter_or_switch() {
        let poc = fh.frame_offset as i32;
        for i in 0..((fh.n_ref_frames as usize).min(7)) {
            let slot = fh.refidx[i];
            if slot >= 0
                && (slot as usize) < dpb.slots.len()
                && let Some(r) = dpb.slots[slot as usize].as_ref()
            {
                ref_frames[i] = Some(r);
                let rpoc = r.poc as i32;
                let delta = if seq.order_hint_n_bits == 0 {
                    0
                } else {
                    (rpoc - poc).clamp(-127, 127)
                };
                refdist[i] = delta;
                absrefdist[i] = delta.abs();
                refdir[i] = (delta > 0) as i8;
                if delta > 0
                    && (furthest_future_refidx < 0
                        || refdist[furthest_future_refidx as usize] < delta)
                {
                    furthest_future_refidx = i as i8;
                }
            }
        }
        if ref_frames[0].is_none() && fh.n_ref_frames > 0 {
            return Err(Av2Error::BitstreamError(format!(
                "inter frame references missing slot {}",
                fh.refidx[0]
            )));
        }
    }

    let n_cols = n_cols.max(1);
    // AV2 §5.20.3: MiCols/MiRows round the frame to the 8-sample MI grid.
    let f_bw4 = 2 * ((fh.width.max(0) as usize + 7) >> 3);
    let f_bh4 = 2 * ((fh.height.max(0) as usize + 7) >> 3);
    let sb_step4 = if fh.sb128 != 0 { 32usize } else { 16usize };
    let root_bs = if fh.sb128 != 0 {
        BlockSize::Bs128x128
    } else {
        BlockSize::Bs64x64
    };
    let n_rows = (fh.tiling.t.rows.max(1) as usize).max(1);
    if tiles.len() != n_cols * n_rows {
        return Err(Av2Error::BitstreamError(format!(
            "tile group has {} tiles but the frame header declares {}",
            tiles.len(),
            n_cols * n_rows
        )));
    }

    // `SkipSegFrame` (AV2 Spec v1.0.0 §7.8): a segment with `SEG_LVL_SKIP` or
    // `SEG_LVL_GLOBALMV` always references this list entry.
    let mut skip_seg_frame = 0usize;
    if fh.is_inter_or_switch() {
        if let Some(i) =
            (0..fh.n_ref_frames as usize).find(|&i| ref_frames[i].is_some() && refdist[i] == 0)
        {
            skip_seg_frame = i;
        } else if let Some(i) = (0..fh.n_ref_frames as usize)
            .filter(|&i| ref_frames[i].is_some() && refdist[i] < 0)
            .max_by_key(|&i| refdist[i])
        {
            skip_seg_frame = i;
        }
    }

    // Current-frame segment map. With `segmentation_update_map == 0` the map
    // is loaded from the primary reference (`load_previous_segment_ids`,
    // AV2 Spec v1.0.0 §5.18.1).
    let mut seg_map = SegmentMap::new(f_bw4, f_bh4);
    if fh.segmentation.enabled != 0
        && fh.segmentation.update_map == 0
        && let Some(map) = prev_segmap
        && map.cols4() == f_bw4
        && map.rows4() == f_bh4
    {
        seg_map.copy_from(map);
    }
    let seg_syntax = SegSyntax::from_frame_header(fh, seq.segmentation.ext);
    let max_q = crate::av2::headers::max_q_bits(bd.bitdepth());

    let mut grid = DeblockGrid::new(f_bw4, f_bh4, seq.layout);
    let cdef_cols = f_bw4.div_ceil(16);
    let cdef_rows = f_bh4.div_ceil(16);
    let mut cdef_idx = vec![-1i8; cdef_cols * cdef_rows];

    let ccso_step4 = 16 * fh.ccso.b64size.max(1) as usize;
    let ccso_cols = f_bw4.div_ceil(ccso_step4);
    let ccso_rows = f_bh4.div_ceil(ccso_step4);
    let mut ccso_blks = [
        vec![false; ccso_cols * ccso_rows],
        vec![false; ccso_cols * ccso_rows],
        vec![false; ccso_cols * ccso_rows],
    ];

    // `sb_reuse_ccso`: the block level enable flags are loaded from the saved
    // `CcsoBlks` of the referenced frame instead of being read from the tile
    // payload (AV2 Spec v1.0.0 §5.18.7.12, `load_ccso_params`, §7.23).
    if fh.ccso.enabled != 0 {
        let n_planes = if seq.layout == PixelLayout::I400 {
            1
        } else {
            3
        };
        for p in 0..n_planes {
            let plane = &fh.ccso.planes[p];
            if plane.enabled == 0 || plane.sb_reuse == 0 {
                continue;
            }
            let ref_list_idx = plane.refidx as usize;
            if ref_list_idx >= fh.n_ref_frames as usize {
                return Err(Av2Error::BitstreamError(format!(
                    "CCSO sb_reuse plane {p} references ref list entry {ref_list_idx} outside {}",
                    fh.n_ref_frames
                )));
            }
            let slot = fh.refidx[ref_list_idx];
            if slot < 0 {
                return Err(Av2Error::BitstreamError(format!(
                    "CCSO sb_reuse plane {p} references an unavailable frame"
                )));
            }
            let maps = dpb
                .ccso_maps
                .get(slot as usize)
                .and_then(|m| m.as_ref())
                .ok_or_else(|| {
                    Av2Error::BitstreamError(format!(
                        "CCSO sb_reuse plane {p} references slot {slot} without a saved CCSO map"
                    ))
                })?;
            if maps[p].len() != ccso_cols * ccso_rows {
                return Err(Av2Error::BitstreamError(format!(
                    "CCSO sb_reuse plane {p} map has {} units, expected {}",
                    maps[p].len(),
                    ccso_cols * ccso_rows
                )));
            }
            ccso_blks[p].copy_from_slice(&maps[p]);
        }
    }

    let gdf_step4 = 16 * fh.gdf.b64size.max(1) as usize;
    let gdf_cols = f_bw4.div_ceil(gdf_step4);
    let gdf_rows = f_bh4.div_ceil(gdf_step4);
    let mut gdf_blks = vec![false; gdf_cols * gdf_rows];

    let ss_x = seq.ss_hor as usize;
    let ss_y = seq.ss_ver as usize;
    let unit_sz_log2_y = fh.restoration.unit_size[0] as usize;
    let half_unit_y = 1usize << unit_sz_log2_y.saturating_sub(1);
    let lr_cols_y = ((f_bw4 * 4 + half_unit_y) >> unit_sz_log2_y).max(1);
    let lr_rows_y = ((f_bh4 * 4 + half_unit_y) >> unit_sz_log2_y).max(1);

    let unit_sz_log2_uv = fh.restoration.unit_size[1] as usize;
    let half_unit_uv = 1usize << unit_sz_log2_uv.saturating_sub(1);
    let lr_cols_uv = ((((f_bw4 * 4) >> ss_x) + half_unit_uv) >> unit_sz_log2_uv).max(1);
    let lr_rows_uv = ((((f_bh4 * 4) >> ss_y) + half_unit_uv) >> unit_sz_log2_uv).max(1);

    let mut lr_types = [
        vec![0u8; lr_cols_y * lr_rows_y],
        vec![0u8; lr_cols_uv * lr_rows_uv],
        vec![0u8; lr_cols_uv * lr_rows_uv],
    ];
    let lr_cols = [lr_cols_y, lr_cols_uv, lr_cols_uv];
    let lr_rows = [lr_rows_y, lr_rows_uv, lr_rows_uv];

    let mut palette_sizes = vec![0u8; f_bw4 * f_bh4];
    let mut palette_colors = vec![[0i32; MAX_PALETTE_COLORS]; f_bw4 * f_bh4];
    let mut ibc_coded = vec![false; f_bw4 * f_bh4];
    let mut intrabc_flags = vec![false; f_bw4 * f_bh4];

    for (idx, tile_data) in tiles.iter().enumerate() {
        let tr = idx / n_cols;
        let tc = idx % n_cols;
        let col_start4 = fh.tiling.t.col_start_sb[tc] as usize * sb_step4;
        let col_end4 = ((fh.tiling.t.col_start_sb[tc + 1] as usize) * sb_step4).min(f_bw4);
        let row_start4 = fh.tiling.t.row_start_sb[tr] as usize * sb_step4;
        let row_end4 = ((fh.tiling.t.row_start_sb[tr + 1] as usize) * sb_step4).min(f_bh4);
        if col_start4 >= col_end4 || row_start4 >= row_end4 {
            continue;
        }
        let mut decoder = TileDecoder {
            bd,
            seq,
            fh,
            rec: &mut rec,
            grid: &mut grid,
            ref_frames,
            refdist,
            absrefdist,
            refdir,
            furthest_future_refidx,
            scratch_mc0: vec![0i32; 128 * 128],
            scratch_mc1: vec![0i32; 128 * 128],
            scratch_diff_mask: vec![0u8; 128 * 128],
            scratch_chroma_diff_mask: vec![0u8; 128 * 128],
            skip_seg_frame,
            dq,
            seg_map: &mut seg_map,
            seg_prev: prev_segmap,
            seg_syntax,
            seg_pred_a: Vec::new(),
            seg_pred_l: [0u8; 32],
            cur_qidx: fh.quant.yac as i32,
            max_q,
            root_bs,
            msac: MsacDecoder::new(tile_data, fh.disable_cdf_update == 0),
            cdf: CdfContext::new(fh.quant.yac),
            a_cols: Vec::new(),
            l_ctx: BlockContext::default(),
            dq_y: [0i32; MAX_TX_AREA],
            dq_u: [0i32; MAX_TX_AREA],
            dq_v: [0i32; MAX_TX_AREA],
            tile_col_start4: col_start4,
            tile_col_end4: col_end4,
            tile_row_start4: row_start4,
            tile_row_end4: row_end4,
            frame_mi_w: f_bw4,
            frame_mi_h: f_bh4,
            cdef_idx: &mut cdef_idx,
            cdef_cols,
            cdef_rows,
            sb_step4,
            ccso_blks: &mut ccso_blks,
            ccso_cols,
            ccso_rows,
            ccso_step4,
            gdf_blks: &mut gdf_blks,
            gdf_cols,
            gdf_rows,
            gdf_step4,
            lr_types: &mut lr_types,
            lr_cols,
            lr_rows,
            warp_mi: vec![WarpMiInfo::default(); f_bw4 * f_bh4],
            comp_mi: vec![CompoundMiInfo::default(); f_bw4 * f_bh4],
            ref_mi_scratch: vec![WarpMiInfo::default(); f_bw4 * f_bh4],
            warp_bank: (0..8).map(|_| Vec::with_capacity(4)).collect(),
            warped_blocks: 0,
            palette_sizes: &mut palette_sizes,
            palette_colors: &mut palette_colors,
            ibc_coded: &mut ibc_coded,
            intrabc_flags: &mut intrabc_flags,
            ibc: IbcBufferState::default(),
            tu_coded: [
                vec![false; f_bw4 * f_bh4],
                vec![false; f_bw4 * f_bh4],
                vec![false; f_bw4 * f_bh4],
            ],
        };
        decoder.decode_tile(root_bs, sb_step4)?;
    }

    if fh.deblock.level_y != [0, 0] || fh.deblock.level_u != 0 || fh.deblock.level_v != 0 {
        deblock_frame(bd, &mut rec, seq, fh, &grid);
    }

    let restore_chroma = fh.restoration.planes[1].restoration_type == 2
        || fh.restoration.planes[2].restoration_type == 2;
    let orig_luma = if fh.ccso.enabled != 0
        || fh.gdf.enabled != AdaptiveBoolean::Off
        || fh.restoration.planes[0].restoration_type != 0
    {
        Some(rec.planes[0].clone())
    } else {
        None
    };
    let orig_chroma = if restore_chroma {
        Some([rec.planes[1].clone(), rec.planes[2].clone()])
    } else {
        None
    };

    if fh.cdef.enabled != 0 {
        cdef_frame_normative(bd, &mut rec, seq, fh, &grid, &cdef_idx);
    }

    if let Some(orig_luma) = orig_luma {
        if fh.ccso.enabled != 0 {
            ccso_frame_normative(bd, &mut rec, &orig_luma, seq, fh, &ccso_blks);
        }
        let cdef_luma = rec.planes[0].clone();
        let luma_type = fh.restoration.planes[0].restoration_type;
        if luma_type == 1 || luma_type == 3 {
            let pc_units: Vec<bool> = lr_types[0].iter().map(|&t| t == 1).collect();
            apply_pc_wiener_units(
                bd,
                &mut rec.planes[0],
                &cdef_luma,
                &orig_luma,
                seq,
                fh,
                &pc_units,
                &grid,
            );
        }
        if luma_type == 2 || luma_type == 3 {
            apply_ns_wiener_plane(
                bd,
                0,
                &mut rec.planes[0],
                &cdef_luma,
                &orig_luma,
                &cdef_luma,
                &orig_luma,
                seq,
                &lr_types[0],
                &fh.restoration.planes[0].ns.filter[0],
                fh.restoration.unit_size[0] as usize,
                0,
                0,
            );
        }
        if let Some(orig_chroma) = orig_chroma.as_ref() {
            let cdef_chroma = [rec.planes[1].clone(), rec.planes[2].clone()];
            for p in 1..3 {
                if fh.restoration.planes[p].restoration_type != 2 {
                    continue;
                }
                apply_ns_wiener_plane(
                    bd,
                    p,
                    &mut rec.planes[p],
                    &cdef_chroma[p - 1],
                    &orig_chroma[p - 1],
                    &cdef_luma,
                    &orig_luma,
                    seq,
                    &lr_types[p],
                    &fh.restoration.planes[p].ns.filter[0],
                    fh.restoration.unit_size[1] as usize,
                    seq.ss_hor as i32,
                    seq.ss_ver as i32,
                );
            }
        }
        if fh.gdf.enabled != AdaptiveBoolean::Off {
            let mut ref_dists = [0i32; 8];
            if fh.is_inter_or_switch() {
                for (i, &idx) in fh.refidx.iter().enumerate().take(fh.n_ref_frames as usize) {
                    if idx >= 0
                        && (idx as usize) < dpb.slots.len()
                        && let Some(ref r) = dpb.slots[idx as usize]
                    {
                        ref_dists[i] = (fh.frame_offset as i32 - r.poc as i32).abs();
                    }
                }
            }
            gdf_frame_normative(
                bd, &mut rec, &cdef_luma, &orig_luma, seq, fh, &ref_dists, &gdf_blks,
            );
        }
    }

    // `save_ccso_params`: return the current frame's CCSO block map so the
    // caller can store it in the refreshed reference slots for a later
    // `sb_reuse_ccso` frame (AV2 Spec v1.0.0 §7.23).
    Ok((rec, seg_map, ccso_blks))
}

fn validate_tools(seq: &SequenceHeader, fh: &FrameHeader) -> Result<()> {
    let unsupported = |what: &str| Err(Av2Error::Unsupported(what.to_string()));
    // Palette mode and Intra Block Copy are decoded (`dec/frame.rs`), so
    // `allow_screen_content_tools` no longer rejects the stream. The tools only
    // add per-block syntax when the sequence/frame headers enable them.
    if seq.sdp || seq.ext_sdp {
        return unsupported("separate chroma partitioning (SDP)");
    }
    if seq.fsc {
        return unsupported("frequency-separated coefficient coding (FSC)");
    }
    if seq.cctx || seq.mhccp || seq.ibp {
        return unsupported("cross-component / MHCCP / IBP intra tools");
    }
    if seq.tip_mode != 0 || seq.mv_traj || seq.bawp || seq.imp_msk_bld {
        return unsupported("TIP / BAWP / implicit mask inter tools");
    }
    if seq.ref_frame_mvs && fh.use_ref_frame_mvs != 0 {
        return unsupported("reference frame motion vectors");
    }
    if seq.opfl_refine != 0 || seq.refine_mv {
        return unsupported("OPFL / MV refinement");
    }
    if seq.adaptive_mvd || seq.mvd_sign_derive || seq.flex_mvres {
        return unsupported("adaptive MVD / flexible MV resolution");
    }
    if seq.global_motion && fh.is_inter_or_switch() {
        return unsupported("global motion");
    }
    if seq.ist[0] || seq.ist[1] || seq.inter_ddt {
        return unsupported("implicit transform selection / DDT");
    }
    if fh.tcq != 0 || fh.parity_hiding != 0 {
        return unsupported("trellis-coded quantization / parity hiding");
    }
    if fh
        .restoration
        .planes
        .iter()
        .any(|p| p.restoration_type >= 2 && p.ns.frame_filters_on == 0)
    {
        return unsupported("per-unit Wiener NS filter bank");
    }
    if fh
        .restoration
        .planes
        .iter()
        .any(|p| p.restoration_type >= 2 && p.ns.num_classes > 1)
    {
        return unsupported("multi-class frame-level Wiener NS filters");
    }
    if fh.skip_mode_enabled != 0 {
        return unsupported("skip mode");
    }
    if fh.bawp != 0 {
        return unsupported("block-adaptive weighted prediction");
    }
    if fh.warp_motion != 0 && (seq.motion_modes & (1 << 3)) == 0 {
        return Err(Av2Error::BitstreamError(
            "allow_warpmv_mode without sequence DELTAWARP".to_string(),
        ));
    }
    if seq.layout == PixelLayout::I400 && fh.is_inter_or_switch() {
        // Monochrome inter is supported by the same path; nothing to reject.
    }
    Ok(())
}

struct TileDecoder<'a, BD: BitDepth> {
    bd: BD,
    seq: &'a SequenceHeader,
    fh: &'a FrameHeader,
    rec: &'a mut Frame<BD::Pixel>,
    grid: &'a mut DeblockGrid,
    ref_frames: [Option<&'a Frame<BD::Pixel>>; 7],
    refdist: [i32; 7],
    absrefdist: [i32; 7],
    refdir: [i8; 7],
    furthest_future_refidx: i8,
    skip_seg_frame: usize,
    scratch_mc0: Vec<i32>,
    scratch_mc1: Vec<i32>,
    scratch_diff_mask: Vec<u8>,
    scratch_chroma_diff_mask: Vec<u8>,
    dq: [[[u32; 2]; 3]; MAX_SEGMENTS],
    seg_map: &'a mut SegmentMap,
    seg_prev: Option<&'a SegmentMap>,
    seg_syntax: SegSyntax,
    seg_pred_a: Vec<[u8; 32]>,
    seg_pred_l: [u8; 32],
    cur_qidx: i32,
    max_q: i32,
    root_bs: BlockSize,
    msac: MsacDecoder<'a>,
    cdf: CdfContext,
    a_cols: Vec<BlockContext>,
    l_ctx: BlockContext,
    dq_y: [i32; MAX_TX_AREA],
    dq_u: [i32; MAX_TX_AREA],
    dq_v: [i32; MAX_TX_AREA],
    tile_col_start4: usize,
    tile_col_end4: usize,
    tile_row_start4: usize,
    tile_row_end4: usize,
    frame_mi_w: usize,
    frame_mi_h: usize,
    cdef_idx: &'a mut [i8],
    cdef_cols: usize,
    cdef_rows: usize,
    sb_step4: usize,
    ccso_blks: &'a mut [Vec<bool>; 3],
    ccso_cols: usize,
    ccso_rows: usize,
    ccso_step4: usize,
    gdf_blks: &'a mut [bool],
    gdf_cols: usize,
    gdf_rows: usize,
    gdf_step4: usize,
    lr_types: &'a mut [Vec<u8>; 3],
    lr_cols: [usize; 3],
    lr_rows: [usize; 3],
    /// Per-4x4 mode map used by `generate_points_from_corners` (§7.12.2.3).
    warp_mi: Vec<WarpMiInfo>,
    /// Per-4x4 neighbour mode information for the compound `find_mv_stack`
    /// branch (both reference lists, §7.12.2.13 / §7.12.2.14).
    comp_mi: Vec<CompoundMiInfo>,
    /// Scratch map used to evaluate the single-reference stack of a compound
    /// block for one list (§7.12.2.10 `candList`).
    ref_mi_scratch: Vec<WarpMiInfo>,
    /// Per-reference warp parameter bank (§7.12.3 `update_warp_param_bank`).
    warp_bank: Vec<Vec<[i32; 6]>>,
    /// Number of `WARPMV` / `WARP_NEWMV` blocks decoded for this tile.
    warped_blocks: usize,
    /// `PaletteSizes[MiRow][MiCol]` (§5.20.8.1 `get_palette_cache`).
    palette_sizes: &'a mut [u8],
    /// `PaletteColors[MiRow][MiCol]` (§5.20.8.1 `get_palette_cache`).
    palette_colors: &'a mut [[i32; MAX_PALETTE_COLORS]],
    /// `IBCCoded[MiRow][MiCol]` (§5.20.3.1 `decode_partition`/`decode_block`).
    ibc_coded: &'a mut [bool],
    /// Neighbour `intrabc` flags for the `TileIntrabcCdf` context
    /// (§5.20.5.3 / §8.3.2).
    intrabc_flags: &'a mut [bool],
    /// Local IntraBC buffer ring (`update_ibc_buffers`, §5.20.3.1).
    ibc: IbcBufferState,
    /// Per-plane `is_coded` map at 4x4 granularity for the lossless
    /// per-transform-block intra prediction (`dav2d` `recon_tmpl.c`).
    tu_coded: [Vec<bool>; 3],
}

impl<BD: BitDepth> TileDecoder<'_, BD> {
    fn read_wedge_idx(&mut self) -> Result<u8> {
        let quad = self.msac.decode_symbol_adapt(self.cdf.m.wedge_quad(), 3)?;
        let angle_sub = self
            .msac
            .decode_symbol_adapt(self.cdf.m.wedge_angle(quad), 4)?;
        let angle = 5 * quad + angle_sub;
        let dist = if angle.wrapping_sub(1) >= 9 || angle == 5 {
            1 + self.msac.decode_symbol_adapt(self.cdf.m.wedge_dist2(), 2)?
        } else {
            self.msac.decode_symbol_adapt(self.cdf.m.wedge_dist1(), 3)?
        };
        let idx = WEDGE_ANGLE_DIST_TO_IDX[angle][dist];
        if idx < 0 {
            return Err(Av2Error::BitstreamError(format!(
                "invalid wedge angle {angle} dist {dist}"
            )));
        }
        Ok(idx as u8)
    }

    fn decode_lr(&mut self, bx4: usize, by4: usize) -> Result<()> {
        let n_planes = if self.seq.layout == PixelLayout::I400 {
            1
        } else {
            3
        };
        let ss_x = self.seq.ss_hor as usize;
        let ss_y = self.seq.ss_ver as usize;
        for p in 0..n_planes {
            let frame_type = self.fh.restoration.planes[p].restoration_type;
            if frame_type == 0 {
                continue;
            }
            let (sub_x, sub_y) = if p == 0 { (0, 0) } else { (ss_x, ss_y) };
            let tx = ((bx4 - self.tile_col_start4) * 4) >> sub_x;
            let ty = ((by4 - self.tile_row_start4) * 4) >> sub_y;
            let unit_sz_log2 = self.fh.restoration.unit_size[if p == 0 { 0 } else { 1 }] as usize;
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
            let unit_type = match frame_type {
                1 => {
                    let flag = self.msac.decode_bool_adapt(&mut self.cdf.m.use_pc_wiener)? != 0;
                    u8::from(flag)
                }
                2 => {
                    let flag = self.msac.decode_bool_adapt(&mut self.cdf.m.use_wiener_ns)? != 0;
                    if flag { 2 } else { 0 }
                }
                3 => {
                    // Switchable (`rst_switchable`): first symbol selects
                    // RESTORE_NONE, the second PC vs NS (AV2 §5.20.10.5).
                    let is_none = self
                        .msac
                        .decode_bool_adapt(&mut self.cdf.m.flex_restoration_type[0][0])?
                        != 0;
                    if is_none {
                        0
                    } else {
                        let is_pc = self
                            .msac
                            .decode_bool_adapt(&mut self.cdf.m.flex_restoration_type[1][0])?
                            != 0;
                        if is_pc { 1 } else { 2 }
                    }
                }
                _ => 0,
            };
            if unit_type == 2 && self.fh.restoration.planes[p].ns.frame_filters_on == 0 {
                return Err(Av2Error::Unsupported(
                    "per-unit Wiener NS filter bank".to_string(),
                ));
            }
            self.lr_types[p][ur * self.lr_cols[p] + uc] = unit_type;
        }
        Ok(())
    }

    fn decode_gdf_blk(&mut self, bx4: usize, by4: usize) -> Result<()> {
        if self.fh.gdf.enabled == AdaptiveBoolean::Off {
            return Ok(());
        }
        if ((bx4 | by4) & (self.gdf_step4 - 1)) != 0 {
            return Ok(());
        }
        let uc = bx4 / self.gdf_step4;
        let ur = by4 / self.gdf_step4;
        if ur >= self.gdf_rows || uc >= self.gdf_cols {
            return Ok(());
        }
        let flag = if self.fh.gdf.enabled == AdaptiveBoolean::Adaptive {
            self.msac.decode_bool_adapt(&mut self.cdf.m.use_gdf)? != 0
        } else {
            true
        };
        self.gdf_blks[ur * self.gdf_cols + uc] = flag;
        Ok(())
    }
    fn decode_cdef_idx(
        &mut self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        skip: bool,
    ) -> Result<()> {
        if self.fh.cdef.enabled == 0 || (skip && self.fh.cdef.on_skiptx == 0) {
            return Ok(());
        }
        let r16 = by4 / 16;
        let c16 = bx4 / 16;
        let idx64 = r16 * self.cdef_cols + c16;
        if self.cdef_idx[idx64] != -1 {
            return Ok(());
        }

        let v = if self.fh.cdef.n_strengths == 1 {
            0i8
        } else {
            let left = if bx4 >= 16 + self.tile_col_start4 {
                self.cdef_idx[r16 * self.cdef_cols + c16 - 1]
            } else {
                -1
            };
            let above = if by4 >= 16 + self.tile_row_start4
                && (by4 - 16) / self.sb_step4 == by4 / self.sb_step4
            {
                self.cdef_idx[(r16 - 1) * self.cdef_cols + c16]
            } else {
                -1
            };

            let ctx = if left != -1 && above != -1 {
                let mut c = (left == 0) as usize + (above == 0) as usize;
                if c == 2 {
                    c += 1;
                }
                c
            } else if left != -1 || above != -1 {
                let avail = if left != -1 { left } else { above };
                if avail == 0 { 2 } else { 0 }
            } else {
                0
            };

            let cdef_index0 = self
                .msac
                .decode_bool_adapt(&mut self.cdf.m.cdef_index0[ctx])?;
            if cdef_index0 != 0 {
                0i8
            } else if self.fh.cdef.n_strengths == 2 {
                1i8
            } else {
                let rem = (self.fh.cdef.n_strengths - 3) as usize;
                let n_symbols = (self.fh.cdef.n_strengths - 2) as usize;
                let sym = match rem {
                    0 => self
                        .msac
                        .decode_symbol_adapt(&mut self.cdf.m.cdef_index_minus1_with3, n_symbols)?,
                    1 => self
                        .msac
                        .decode_symbol_adapt(&mut self.cdf.m.cdef_index_minus1_with4, n_symbols)?,
                    2 => self
                        .msac
                        .decode_symbol_adapt(&mut self.cdf.m.cdef_index_minus1_with5, n_symbols)?,
                    3 => self
                        .msac
                        .decode_symbol_adapt(&mut self.cdf.m.cdef_index_minus1_with6, n_symbols)?,
                    4 => self
                        .msac
                        .decode_symbol_adapt(&mut self.cdf.m.cdef_index_minus1_with7, n_symbols)?,
                    _ => self
                        .msac
                        .decode_symbol_adapt(&mut self.cdf.m.cdef_index_minus1_with8, n_symbols)?,
                };
                (sym + 1) as i8
            }
        };

        let rows = ((by4 & 15) + bh4).div_ceil(16);
        let cols = ((bx4 & 15) + bw4).div_ceil(16);
        for dy16 in 0..rows {
            let r = r16 + dy16;
            if r >= self.cdef_rows {
                break;
            }
            for dx16 in 0..cols {
                let c = c16 + dx16;
                if c >= self.cdef_cols {
                    break;
                }
                self.cdef_idx[r * self.cdef_cols + c] = v;
            }
        }
        Ok(())
    }

    fn decode_ccso_blk(&mut self, bx4: usize, by4: usize) -> Result<()> {
        if self.fh.ccso.enabled == 0 {
            return Ok(());
        }
        if ((bx4 | by4) & (self.ccso_step4 - 1)) != 0 {
            return Ok(());
        }
        let uc = bx4 / self.ccso_step4;
        let ur = by4 / self.ccso_step4;
        if ur >= self.ccso_rows || uc >= self.ccso_cols {
            return Ok(());
        }

        let n_planes = if self.seq.layout == PixelLayout::I400 {
            1
        } else {
            3
        };
        for p in 0..n_planes {
            if self.fh.ccso.planes[p].enabled == 0 {
                continue;
            }
            if self.fh.ccso.planes[p].sb_reuse != 0 {
                // The map was preloaded from the reference slot before tile
                // decoding; no `ccso_blk` syntax is present (§5.20.10.2).
                continue;
            }
            let left_available = bx4 > self.tile_col_start4;
            let left_val =
                left_available && self.ccso_blks[p][ur * self.ccso_cols + uc.saturating_sub(1)];
            let ctx = if left_val { 2 } else { 0 };
            let val = self
                .msac
                .decode_bool_adapt(&mut self.cdf.m.ccso_blk[p][ctx])?
                != 0;
            self.ccso_blks[p][ur * self.ccso_cols + uc] = val;
        }
        Ok(())
    }

    fn decode_tile(&mut self, root_bs: BlockSize, sb_step4: usize) -> Result<()> {
        let num_sb_cols = ((self.tile_col_end4 - self.tile_col_start4) + 31) >> 5;
        self.a_cols = vec![BlockContext::default(); num_sb_cols.max(1)];
        for ctx in &mut self.a_cols {
            reset_context(ctx, self.fh.is_key_or_intra(), false);
        }
        reset_context(&mut self.l_ctx, self.fh.is_key_or_intra(), false);
        self.seg_pred_a = vec![[0u8; 32]; num_sb_cols.max(1)];

        let mut by4 = self.tile_row_start4;
        while by4 < self.tile_row_end4 {
            reset_context(&mut self.l_ctx, self.fh.is_key_or_intra(), false);
            self.seg_pred_l = [0u8; 32];
            let mut bx4 = self.tile_col_start4;
            while bx4 < self.tile_col_end4 {
                self.decode_lr(bx4, by4)?;
                self.decode_sb(root_bs, bx4, by4, false)?;
                bx4 += sb_step4;
            }
            by4 += sb_step4;
        }
        Ok(())
    }

    /// Decodes the coding tree of `bs` at `(bx4, by4)`, mirroring the encoder's
    /// [`crate::av2::frame`] tree walk and AV2 Spec v1.0.0 §5.20.3.1/§5.20.3.2.
    fn decode_sb(
        &mut self,
        bs: BlockSize,
        bx4: usize,
        by4: usize,
        chroma_offset: bool,
    ) -> Result<()> {
        if bx4 >= self.tile_col_end4 || by4 >= self.tile_row_end4 {
            return Ok(());
        }
        let cfg = PartitionSyntaxConfig::from_seq_header(self.seq, self.fh.is_key_or_intra());
        let allowed = init_allowed_partitions(
            bs,
            bx4,
            by4,
            self.frame_mi_w,
            self.frame_mi_h,
            chroma_offset,
            &cfg,
        );
        let bp = match partition_implied(bs, bx4, by4, self.frame_mi_w, self.frame_mi_h)
            .filter(|p| allowed.contains(*p))
        {
            Some(partition) => partition,
            None => match allowed.single() {
                Some(partition) => partition,
                None => {
                    let a_idx = ((bx4 - self.tile_col_start4) >> 5).min(self.a_cols.len() - 1);
                    decode_partition_decision(
                        &mut self.msac,
                        &mut self.cdf.m,
                        &self.a_cols[a_idx],
                        &self.l_ctx,
                        bs,
                        bx4 & 31,
                        by4 & 31,
                        &allowed,
                    )?
                }
            },
        };
        self.decode_partition_node(bs, bp, bx4, by4, chroma_offset, &cfg)
    }

    /// Applies the decoded partition: updates the partition context and either
    /// decodes the leaf block or recurses into the sub-blocks.
    fn decode_partition_node(
        &mut self,
        bs: BlockSize,
        bp: BlockPartition,
        bx4: usize,
        by4: usize,
        chroma_offset: bool,
        cfg: &PartitionSyntaxConfig,
    ) -> Result<()> {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let a_idx = ((bx4 - self.tile_col_start4) >> 5).min(self.a_cols.len() - 1);
        let child_chroma_offset = chroma_offset || chroma_offset_for_partition(bp, bs, cfg);
        let Some(sub_bs) = partition_subsize(bs, bp) else {
            return Err(Av2Error::BitstreamError(format!(
                "partition {bp:?} is not legal for block size {bs:?}"
            )));
        };

        match bp {
            BlockPartition::None => {
                update_partition_ctx(
                    &mut self.a_cols[a_idx],
                    &mut self.l_ctx,
                    bs,
                    bx4 & 31,
                    by4 & 31,
                );
                self.decode_leaf(bs, bx4, by4)
            }
            BlockPartition::H => {
                self.decode_sb(sub_bs, bx4, by4, child_chroma_offset)?;
                self.decode_sb(sub_bs, bx4, by4 + (bh4 >> 1), child_chroma_offset)
            }
            BlockPartition::V => {
                self.decode_sb(sub_bs, bx4, by4, child_chroma_offset)?;
                self.decode_sb(sub_bs, bx4 + (bw4 >> 1), by4, child_chroma_offset)
            }
            BlockPartition::Split => {
                let half_w4 = bw4 >> 1;
                let half_h4 = bh4 >> 1;
                self.decode_sb(sub_bs, bx4, by4, child_chroma_offset)?;
                self.decode_sb(sub_bs, bx4 + half_w4, by4, child_chroma_offset)?;
                self.decode_sb(sub_bs, bx4, by4 + half_h4, child_chroma_offset)?;
                self.decode_sb(sub_bs, bx4 + half_w4, by4 + half_h4, child_chroma_offset)
            }
            BlockPartition::H3 => {
                let Some(mid_bs) = h_partition_midsize(bs) else {
                    return Err(Av2Error::BitstreamError(format!(
                        "midsize not defined for partition H3 on {bs:?}"
                    )));
                };
                let qh4 = bh4 >> 2;
                self.decode_sb(sub_bs, bx4, by4, false)?;
                self.decode_sb(mid_bs, bx4, by4 + qh4, false)?;
                self.decode_sb(mid_bs, bx4 + (bw4 >> 1), by4 + qh4, false)?;
                self.decode_sb(sub_bs, bx4, by4 + 3 * qh4, child_chroma_offset)
            }
            BlockPartition::V3 => {
                let Some(mid_bs) = h_partition_midsize(bs) else {
                    return Err(Av2Error::BitstreamError(format!(
                        "midsize not defined for partition V3 on {bs:?}"
                    )));
                };
                let qw4 = bw4 >> 2;
                self.decode_sb(sub_bs, bx4, by4, false)?;
                self.decode_sb(mid_bs, bx4 + qw4, by4, false)?;
                self.decode_sb(mid_bs, bx4 + qw4, by4 + (bh4 >> 1), false)?;
                self.decode_sb(sub_bs, bx4 + 3 * qw4, by4, child_chroma_offset)
            }
            BlockPartition::H4A => {
                let Some((sub_bs, big_bs, med_bs)) = partition_4way_sizes(bs, bp) else {
                    return Err(Av2Error::BitstreamError(format!(
                        "4-way sizes not defined for partition H4A on {bs:?}"
                    )));
                };
                let u = bh4 >> 3;
                self.decode_sb(sub_bs, bx4, by4, false)?;
                self.decode_sb(med_bs, bx4, by4 + u, false)?;
                self.decode_sb(big_bs, bx4, by4 + 3 * u, false)?;
                self.decode_sb(sub_bs, bx4, by4 + 7 * u, child_chroma_offset)
            }
            BlockPartition::H4B => {
                let Some((sub_bs, big_bs, med_bs)) = partition_4way_sizes(bs, bp) else {
                    return Err(Av2Error::BitstreamError(format!(
                        "4-way sizes not defined for partition H4B on {bs:?}"
                    )));
                };
                let u = bh4 >> 3;
                self.decode_sb(sub_bs, bx4, by4, false)?;
                self.decode_sb(big_bs, bx4, by4 + u, false)?;
                self.decode_sb(med_bs, bx4, by4 + 5 * u, false)?;
                self.decode_sb(sub_bs, bx4, by4 + 7 * u, child_chroma_offset)
            }
            BlockPartition::V4A => {
                let Some((sub_bs, big_bs, med_bs)) = partition_4way_sizes(bs, bp) else {
                    return Err(Av2Error::BitstreamError(format!(
                        "4-way sizes not defined for partition V4A on {bs:?}"
                    )));
                };
                let u = bw4 >> 3;
                self.decode_sb(sub_bs, bx4, by4, false)?;
                self.decode_sb(med_bs, bx4 + u, by4, false)?;
                self.decode_sb(big_bs, bx4 + 3 * u, by4, false)?;
                self.decode_sb(sub_bs, bx4 + 7 * u, by4, child_chroma_offset)
            }
            BlockPartition::V4B => {
                let Some((sub_bs, big_bs, med_bs)) = partition_4way_sizes(bs, bp) else {
                    return Err(Av2Error::BitstreamError(format!(
                        "4-way sizes not defined for partition V4B on {bs:?}"
                    )));
                };
                let u = bw4 >> 3;
                self.decode_sb(sub_bs, bx4, by4, false)?;
                self.decode_sb(big_bs, bx4 + u, by4, false)?;
                self.decode_sb(med_bs, bx4 + 5 * u, by4, false)?;
                self.decode_sb(sub_bs, bx4 + 7 * u, by4, child_chroma_offset)
            }
        }
    }

    fn decode_segment_id_at(
        &mut self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        have_left: bool,
        have_top: bool,
        spatial_only: bool,
        pre_skip: bool,
        skip_txfm: bool,
    ) -> Result<u8> {
        let a_idx = ((bx4 - self.tile_col_start4) >> 5).min(self.seg_pred_a.len() - 1);
        let seg_map = &mut *self.seg_map;
        decode_segment_id(
            &mut self.msac,
            &mut self.cdf.m,
            &self.seg_syntax,
            seg_map,
            self.seg_prev,
            bx4,
            by4,
            bw4,
            bh4,
            have_left,
            have_top,
            spatial_only,
            pre_skip,
            skip_txfm,
            &mut self.seg_pred_a[a_idx],
            &mut self.seg_pred_l,
        )
    }

    /// Reads `read_delta_qindex()` on the first block of a superblock and
    /// refreshes the per-segment dequantizer tables (AV2 Spec v1.0.0 §5.20.5.11
    /// and §5.18.1; dav2d `delta-q/lf`).
    fn decode_delta_q_at(
        &mut self,
        bs: BlockSize,
        bx4: usize,
        by4: usize,
        skip: bool,
    ) -> Result<()> {
        if self.fh.delta.q.present == 0 {
            return Ok(());
        }
        if ((bx4 | by4) & (self.sb_step4 - 1)) != 0 {
            return Ok(());
        }
        // `read_delta_qindex()` is skipped for a whole-superblock skip block.
        if bs == self.root_bs && skip {
            return Ok(());
        }
        let new_qidx = decode_delta_qindex(
            &mut self.msac,
            &mut self.cdf.m,
            self.cur_qidx,
            self.fh.delta.q.res_log2,
            self.max_q,
        )?;
        if new_qidx != self.cur_qidx {
            self.cur_qidx = new_qidx;
            init_segment_quant_tables(self.fh, new_qidx, self.bd.bitdepth(), &mut self.dq);
        }
        Ok(())
    }

    fn decode_leaf(&mut self, bs: BlockSize, bx4: usize, by4: usize) -> Result<()> {
        if bx4 >= self.tile_col_end4 || by4 >= self.tile_row_end4 {
            return Ok(());
        }
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let tx = MAX_TXFM_SIZE_FOR_BS[bs as usize] as usize;
        let have_left = bx4 > self.tile_col_start4;
        let have_top = by4 > self.tile_row_start4;
        let seg_enabled = self.seg_syntax.enabled;
        let inter_frame = self.fh.is_inter_or_switch();

        // With `SegIdPreSkip`, the segment id is read before `is_inter`
        // (§5.20.7.1 `inter_frame_mode_info`) and before `use_intrabc`
        // (§5.20.5.3 `intra_frame_mode_info`).
        let mut seg_id = 0u8;
        if seg_enabled && self.seg_syntax.preskip {
            seg_id = self.decode_segment_id_at(
                bx4, by4, bw4, bh4, have_left, have_top, false, true, false,
            )?;
        }

        let forced_globalmv = seg_enabled && self.seg_syntax.globalmv_feature(self.fh, seg_id);
        let is_intra = if inter_frame {
            if forced_globalmv {
                false
            } else {
                let a_idx = ((bx4 - self.tile_col_start4) >> 5).min(self.a_cols.len() - 1);
                let nb = gather_inter_neighbors(
                    &self.a_cols[a_idx],
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
                self.msac.decode_bool_adapt(self.cdf.m.intra(ctx))? == 0
            }
        } else {
            true
        };

        if is_intra {
            let seg_id = if seg_enabled && !self.seg_syntax.preskip {
                // The post-skip segment id is decoded inside `decode_intra_block`
                // after `use_intrabc`, per §5.20.5.3.
                0
            } else {
                seg_id
            };
            self.decode_intra_block(bs, bx4, by4, tx, seg_id)
        } else {
            // `read_skip()`: `SEG_LVL_SKIP` forces `skip_flag` to 1 and codes
            // no symbol; otherwise the flag is decoded with the neighbour
            // context (§5.20.5.10).
            let seg_skip = seg_enabled && self.seg_syntax.skip_feature(self.fh, seg_id);
            let all_zero = if seg_skip {
                true
            } else {
                let a_idx = ((bx4 - self.tile_col_start4) >> 5).min(self.a_cols.len() - 1);
                let nb = gather_inter_neighbors(
                    &self.a_cols[a_idx],
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
                let skip_ctx = nb.skip_ctx().min(5);
                self.msac.decode_bool_adapt(self.cdf.m.skip(skip_ctx))? != 0
            };

            if seg_enabled && !self.seg_syntax.preskip {
                seg_id = self.decode_segment_id_at(
                    bx4, by4, bw4, bh4, have_left, have_top, false, false, all_zero,
                )?;
            }
            let forced_globalmv = seg_enabled && self.seg_syntax.globalmv_feature(self.fh, seg_id);
            self.decode_inter_block(bs, bx4, by4, tx, seg_id, all_zero, forced_globalmv)
        }
    }

    /// Top-right / bottom-left availability of one lossless 4x4 transform
    /// block, mirroring the reference decoder's `n_tr` / `n_bl` derivation
    /// (`dav2d` `recon_tmpl.c`).
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
        let stride = self.frame_mi_w >> ss_x;
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

    /// Decodes a lossless intra coding block: 4x4 WHT transforms with
    /// per-transform-block intra prediction, optional DPCM, and the `EobU`
    /// V-plane context (AV2 Spec v1.0.0 §5.20.7.24, §5.20.7.27, §7.14.4,
    /// §7.15.4).
    #[allow(clippy::too_many_arguments)]
    fn decode_lossless_intra_block(
        &mut self,
        bs: BlockSize,
        bx4: usize,
        by4: usize,
        seg_id: u8,
        y_mode: u8,
        y_angle: i8,
        midx: u8,
        uv_mode: u8,
        uv_angle: i8,
        dpcm_y: Option<bool>,
        dpcm_uv: Option<bool>,
    ) -> Result<()> {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let px_x = bx4 * 4;
        let px_y = by4 * 4;
        let a_idx = ((bx4 - self.tile_col_start4) >> 5).min(self.a_cols.len() - 1);
        let bx4_in_sb = bx4 & 31;
        let by4_in_sb = by4 & 31;
        let have_left = bx4 > self.tile_col_start4;
        let have_top = by4 > self.tile_row_start4;
        let has_chroma = self.seq.layout != PixelLayout::I400;
        let ss_x = self.seq.ss_hor;
        let ss_y = self.seq.ss_ver;
        let layout_col = (self.seq.layout as usize).saturating_sub(1);
        let uv_bs_idx = if has_chroma {
            SS_BS[bs as usize][layout_col] as usize
        } else {
            bs as usize
        };
        let uv_bs_dim = BLOCK_DIMENSIONS[uv_bs_idx];
        let cw = (uv_bs_dim[0] as usize) * 4;
        let ch = (uv_bs_dim[1] as usize) * 4;
        let cpx_x = px_x >> ss_x;
        let cpx_y = px_y >> ss_y;
        let cbx4_in_sb = bx4_in_sb >> ss_x;
        let cby4_in_sb = by4_in_sb >> ss_y;
        let seg = seg_id.min((MAX_SEGMENTS - 1) as u8) as usize;

        let mut u_has_cf = false;
        for plane in 0..if has_chroma { 3 } else { 1 } {
            let is_luma = plane == 0;
            let plane_w = if is_luma { bw4 * 4 } else { cw };
            let plane_h = if is_luma { bh4 * 4 } else { ch };
            let (ox, oy) = if is_luma {
                (px_x, px_y)
            } else {
                (cpx_x, cpx_y)
            };
            let mode = if is_luma { y_mode } else { uv_mode };
            let angle = if is_luma { y_angle } else { uv_angle };
            let dpcm = if is_luma { dpcm_y } else { dpcm_uv };
            let dpcm_flag: u32 = match dpcm {
                Some(true) => 1,
                Some(false) => 2,
                None => 0,
            };
            let wht_txtp = txtp::WHT_WHT | (dpcm_flag << 8);
            let (pw4, ph4) = (plane_w / 4, plane_h / 4);
            let ss_x_p = self.rec.planes[plane].ss_x as usize;
            let ss_y_p = self.rec.planes[plane].ss_y as usize;
            let (rec_cw, rec_ch) = crate::av2::ipred::intra_reference_extent(
                self.frame_mi_w,
                self.frame_mi_h,
                self.rec.planes[plane].ss_x,
                self.rec.planes[plane].ss_y,
            );
            let mut edge = [BD::Pixel::default(); 512];
            let tl_idx = 256usize;
            for y4 in 0..ph4 {
                for x4 in 0..pw4 {
                    let tx_px = ox + x4 * 4;
                    let ty_px = oy + y4 * 4;
                    let hl = have_left || x4 > 0;
                    let ht = have_top || y4 > 0;
                    let tx4 = ox / 4 + x4;
                    let ty4 = oy / 4 + y4;
                    let (htr, hbl) = self.lossless_tu_edges(plane, tx4, ty4, 1, 1, ss_x_p, ss_y_p);
                    {
                        let pl = &self.rec.planes[plane];
                        prepare_intra_edges(
                            self.bd,
                            &mut edge,
                            tl_idx,
                            &pl.data,
                            pl.offset(tx_px, ty_px),
                            pl.stride,
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
                        let pl = &mut self.rec.planes[plane];
                        let off = pl.offset(tx_px, ty_px);
                        predict_intra_block(
                            self.bd,
                            &mut pl.data,
                            off,
                            pl.stride,
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

                    let params = CoefDecodeParams {
                        tx: 0,
                        bs: if is_luma { bs as usize } else { uv_bs_idx },
                        plane: plane as i32,
                        intra: true,
                        fsc: false,
                        lossless: true,
                        uv_mode: uv_mode as usize,
                        y_mode: y_mode as usize,
                        y_angle,
                        mrl_index: 0,
                        y_txtp: 0,
                        reduced_txtp_set: self.fh.reduced_txtp_set as i32,
                        u_has_cf: if plane == 2 { u_has_cf as i32 } else { 0 },
                        dq_tbl: self.dq[seg][plane],
                        bitdepth: self.bd.bitdepth(),
                        ss_hor: ss_x != 0,
                        ss_ver: ss_y != 0,
                        chroma_dctonly: self.seq.chroma_dctonly,
                    };
                    let (a_slice, l_slice) = if is_luma {
                        (
                            &self.a_cols[a_idx].lcoef[bx4_in_sb + x4..],
                            &self.l_ctx.lcoef[by4_in_sb + y4..],
                        )
                    } else {
                        (
                            &self.a_cols[a_idx].ccoef[plane - 1][cbx4_in_sb + x4..],
                            &self.l_ctx.ccoef[plane - 1][cby4_in_sb + y4..],
                        )
                    };
                    let dq_buf = match plane {
                        0 => &mut self.dq_y,
                        1 => &mut self.dq_u,
                        _ => &mut self.dq_v,
                    };
                    let res = decode_coefs(
                        &mut self.msac,
                        &mut self.cdf.coef,
                        &mut self.cdf.m,
                        a_slice,
                        l_slice,
                        &params,
                        dq_buf,
                    )?;
                    if is_luma {
                        self.a_cols[a_idx].lcoef[bx4_in_sb + x4] = res.res_ctx;
                        self.l_ctx.lcoef[by4_in_sb + y4] = res.res_ctx;
                    } else {
                        self.a_cols[a_idx].ccoef[plane - 1][cbx4_in_sb + x4] = res.res_ctx;
                        self.l_ctx.ccoef[plane - 1][cby4_in_sb + y4] = res.res_ctx;
                        if plane == 1 {
                            u_has_cf = res.eob >= 0;
                        }
                    }
                    if res.eob >= 0 {
                        let pl = &mut self.rec.planes[plane];
                        let off = pl.offset(tx_px, ty_px);
                        inv_txfm_add(
                            self.bd,
                            &mut pl.data,
                            off,
                            pl.stride,
                            dq_buf,
                            wht_txtp,
                            res.eob,
                            0,
                        );
                    }
                    self.tu_coded[plane][ty4 * (self.frame_mi_w >> ss_x_p) + tx4] = true;
                }
            }
        }

        // Neighbour context bookkeeping (mirrors the encoder and the lossy
        // intra path). Every transform is 4x4 in a lossless frame.
        self.store_palette_block(bx4, by4, bw4, bh4, None);
        for i in 0..bw4
            .min(32 - bx4_in_sb)
            .min(self.frame_mi_w.saturating_sub(bx4))
        {
            self.a_cols[a_idx].intra[bx4_in_sb + i] = 1;
            self.a_cols[a_idx].mode[bx4_in_sb + i] = y_mode;
            self.a_cols[a_idx].skip[bx4_in_sb + i] = 0;
            self.a_cols[a_idx].midx[bx4_in_sb + i] = midx;
            self.a_cols[a_idx].tx_luma[bx4_in_sb + i] = 0;
            self.a_cols[a_idx].tx[bx4_in_sb + i] = 0;
            self.a_cols[a_idx].ref_frame[0][bx4_in_sb + i] = -1;
            self.a_cols[a_idx].ref_frame[1][bx4_in_sb + i] = -1;
            self.a_cols[a_idx].filter[0][bx4_in_sb + i] = 3;
            self.a_cols[a_idx].filter[1][bx4_in_sb + i] = 3;
        }
        for i in 0..bh4
            .min(32 - by4_in_sb)
            .min(self.frame_mi_h.saturating_sub(by4))
        {
            self.l_ctx.intra[by4_in_sb + i] = 1;
            self.l_ctx.mode[by4_in_sb + i] = y_mode;
            self.l_ctx.skip[by4_in_sb + i] = 0;
            self.l_ctx.midx[by4_in_sb + i] = midx;
            self.l_ctx.tx_luma[by4_in_sb + i] = 0;
            self.l_ctx.tx[by4_in_sb + i] = 0;
            self.l_ctx.ref_frame[0][by4_in_sb + i] = -1;
            self.l_ctx.ref_frame[1][by4_in_sb + i] = -1;
            self.l_ctx.filter[0][by4_in_sb + i] = 3;
            self.l_ctx.filter[1][by4_in_sb + i] = 3;
        }
        if has_chroma {
            for i in 0..(uv_bs_dim[0] as usize)
                .min(32 - cbx4_in_sb)
                .min((self.frame_mi_w >> ss_x).saturating_sub(bx4 >> ss_x))
            {
                self.a_cols[a_idx].uvmode[cbx4_in_sb + i] = uv_mode;
                self.a_cols[a_idx].tx_chroma[cbx4_in_sb + i] = 0;
            }
            for i in 0..(uv_bs_dim[1] as usize)
                .min(32 - cby4_in_sb)
                .min((self.frame_mi_h >> ss_y).saturating_sub(by4 >> ss_y))
            {
                self.l_ctx.uvmode[cby4_in_sb + i] = uv_mode;
                self.l_ctx.tx_chroma[cby4_in_sb + i] = 0;
            }
        }

        // Deblocking block info: each 4x4 is its own transform block.
        for dy4 in 0..bh4 {
            let y4 = by4 + dy4;
            if y4 >= self.frame_mi_h {
                break;
            }
            for dx4 in 0..bw4 {
                let x4 = bx4 + dx4;
                if x4 >= self.frame_mi_w {
                    break;
                }
                self.grid.set_luma(
                    x4,
                    y4,
                    BlockDeblockInfo {
                        tx_w: 4,
                        tx_h: 4,
                        skip: false,
                        lossless: true,
                        block_left: dx4 == 0,
                        block_top: dy4 == 0,
                        tx_left: true,
                        tx_top: true,
                    },
                );
            }
        }
        if has_chroma {
            let cbx4 = bx4 >> ss_x;
            let cby4 = by4 >> ss_y;
            let cbw4 = (bw4 >> ss_x).max(1);
            let cbh4 = (bh4 >> ss_y).max(1);
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
                            tx_w: 4,
                            tx_h: 4,
                            skip: false,
                            lossless: true,
                            block_left: dx4 == 0,
                            block_top: dy4 == 0,
                            tx_left: true,
                            tx_top: true,
                        },
                    );
                }
            }
        }

        Ok(())
    }

    fn decode_intra_block(
        &mut self,
        bs: BlockSize,
        bx4: usize,
        by4: usize,
        tx: usize,
        preskip_seg_id: u8,
    ) -> Result<()> {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let w = bw4 * 4;
        let h = bh4 * 4;
        let px_x = bx4 * 4;
        let px_y = by4 * 4;
        let a_idx = ((bx4 - self.tile_col_start4) >> 5).min(self.a_cols.len() - 1);
        let bx4_in_sb = bx4 & 31;
        let by4_in_sb = by4 & 31;
        let have_left = bx4 > self.tile_col_start4;
        let have_top = by4 > self.tile_row_start4;
        let t_dim = &TXFM_DIMENSIONS[tx];
        let seg_enabled = self.seg_syntax.enabled;
        let inter_frame = self.fh.is_inter_or_switch();
        let allow_sct = match self.seq.screen_content_tools {
            AdaptiveBoolean::On => true,
            AdaptiveBoolean::Off => false,
            AdaptiveBoolean::Adaptive => self.fh.allow_screen_content_tools != 0,
        };
        let mut seg_id = preskip_seg_id;

        self.mark_mi_default(bs, bx4, by4);
        // `IBCCoded[r][c] = 1` and `update_ibc_buffers(r, c)` run at the start
        // of `decode_block()` (AV2 Spec v1.0.0 §5.20.4.1).
        self.mark_ibc_block(bx4, by4, bw4, bh4);

        // `use_intrabc` (§5.20.5.3). The presence condition is
        // `allow_intrabc && Block_Width <= 64 && Block_Height <= 64 &&
        // MiSize != BLOCK_64X64`; an inter-frame intra block reads the same
        // flag via `read_is_inter()` (§5.20.7.3).
        let intrabc_present =
            self.fh.allow_intrabc != 0 && w <= 64 && h <= 64 && bs != BlockSize::Bs64x64;
        let use_intrabc = if intrabc_present {
            let ctx = self.intrabc_context(bx4, by4, bw4, bh4, have_left, have_top);
            self.msac.decode_bool_adapt(&mut self.cdf.m.intrabc[ctx])? != 0
        } else {
            false
        };

        if use_intrabc {
            let mut skip_txfm = true;
            if !inter_frame {
                let seg_skip = seg_enabled && self.seg_syntax.skip_feature(self.fh, seg_id);
                skip_txfm = if seg_skip {
                    true
                } else {
                    let nb = gather_inter_neighbors(
                        &self.a_cols[a_idx],
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
                    let skip_ctx = nb.skip_ctx().min(5);
                    self.msac.decode_bool_adapt(self.cdf.m.skip(skip_ctx))? != 0
                };
            }
            if seg_enabled && !self.seg_syntax.preskip {
                seg_id = self.decode_segment_id_at(
                    bx4, by4, bw4, bh4, have_left, have_top, true, false, skip_txfm,
                )?;
            }
            self.decode_gdf_blk(bx4, by4)?;
            self.decode_cdef_idx(bx4, by4, bw4, bh4, false)?;
            self.decode_ccso_blk(bx4, by4)?;
            self.decode_delta_q_at(bs, bx4, by4, skip_txfm)?;
            return self.decode_intrabc_block(bs, bx4, by4, seg_id, skip_txfm, allow_sct);
        }

        if seg_enabled && !self.seg_syntax.preskip {
            seg_id = self.decode_segment_id_at(
                bx4, by4, bw4, bh4, have_left, have_top, true, false, false,
            )?;
        }
        let seg_lossless =
            self.fh.quant.lossless[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize] != 0;

        self.decode_gdf_blk(bx4, by4)?;
        self.decode_cdef_idx(bx4, by4, bw4, bh4, false)?;
        self.decode_ccso_blk(bx4, by4)?;
        self.decode_delta_q_at(bs, bx4, by4, false)?;

        let (y_mode, y_angle, midx, dpcm_y) = decode_intra_y_mode(
            &mut self.msac,
            &mut self.cdf.m,
            &self.a_cols[a_idx],
            &self.l_ctx,
            bx4_in_sb,
            by4_in_sb,
            bw4 as i32,
            bh4 as i32,
            seg_lossless,
        )?;

        let mut mrl_idx = 0u8;
        if dpcm_y.is_none() && self.seq.mrls && (1..=8).contains(&y_mode) {
            mrl_idx = self.msac.decode_symbol_adapt(self.cdf.m.mrl_idx(0), 3)? as u8;
            if mrl_idx > 0 {
                let _second = self.msac.decode_bool_adapt(self.cdf.m.mrl_sec_idx(0))?;
            }
        }

        let has_chroma = self.seq.layout != PixelLayout::I400;
        let ss_x = self.seq.ss_hor;
        let ss_y = self.seq.ss_ver;
        let layout_col = (self.seq.layout as usize).saturating_sub(1);
        let uv_bs_idx = if has_chroma {
            SS_BS[bs as usize][layout_col] as usize
        } else {
            bs as usize
        };
        let uvtx = MAX_TXFM_SIZE_FOR_BS[uv_bs_idx] as usize;
        let uv_tdim = &TXFM_DIMENSIONS[uvtx];
        let cw = uv_tdim.w as usize * 4;
        let ch = uv_tdim.h as usize * 4;
        let cpx_x = px_x >> ss_x;
        let cpx_y = px_y >> ss_y;

        let (uv_mode, uv_angle, dpcm_uv, cfl_alphas) = if has_chroma {
            let (um, ua, duv) = decode_intra_uv_mode(
                &mut self.msac,
                &mut self.cdf.m,
                self.seq.cfl,
                seg_lossless,
                midx,
            )?;
            if um == 13 {
                let (au, av) = decode_cfl_alphas(&mut self.msac, &mut self.cdf.m)?;
                (um, ua, duv, Some((au, av)))
            } else {
                (um, ua, duv, None)
            }
        } else {
            (0u8, 0i8, None, None)
        };

        if !cfl_alphas.is_some() && uv_mode == 13 {
            return Err(Av2Error::BitstreamError(
                "CfL mode without alpha parameters".to_string(),
            ));
        }

        // `palette_mode_info()` runs after the intra UV mode and CfL alphas and
        // before `dip_mode_info()` (§5.20.5.3, §5.20.7.5). It is present for
        // `MiSize >= BLOCK_8X8`, `Block_Width/Height <= 64`, an enabled
        // `allow_screen_content_tools`, and a `DC_PRED` luma mode.
        let palette = if allow_sct
            && bs != BlockSize::Bs4x4
            && bs != BlockSize::Bs4x8
            && bs != BlockSize::Bs8x4
            && w <= 64
            && h <= 64
            && y_mode == 0
        {
            let mut above: Vec<i32> = Vec::new();
            if by4 > 0 && (by4 * 4) % 64 != 0 {
                let n = self.palette_sizes[(by4 - 1) * self.frame_mi_w + bx4] as usize;
                above.extend_from_slice(
                    &self.palette_colors[(by4 - 1) * self.frame_mi_w + bx4][..n],
                );
            }
            let mut left: Vec<i32> = Vec::new();
            if have_left {
                let n = self.palette_sizes[by4 * self.frame_mi_w + bx4 - 1] as usize;
                left.extend_from_slice(&self.palette_colors[by4 * self.frame_mi_w + bx4 - 1][..n]);
            }
            let onscreen_w = w.min((self.frame_mi_w * 4).saturating_sub(px_x));
            let onscreen_h = h.min((self.frame_mi_h * 4).saturating_sub(px_y));
            decode_palette_mode_info(
                &mut self.msac,
                &mut self.cdf.m,
                self.bd.bitdepth(),
                &above,
                &left,
                w,
                h,
                onscreen_w,
                onscreen_h,
            )?
        } else {
            None
        };

        let mut dip_mode: Option<u8> = None;
        if self.seq.intra_dip && y_mode == 0 {
            let ctx = bs as usize;
            if self
                .msac
                .decode_bool_adapt(self.cdf.m.use_dip(ctx.min(2)))?
                != 0
            {
                let transpose = self.msac.decode_bool_bypass();
                let mode = self.msac.decode_symbol_adapt(self.cdf.m.dip_mode(), 5)? as u8;
                if transpose != 0 {
                    return Err(Av2Error::Unsupported(
                        "transposed DIP prediction".to_string(),
                    ));
                }
                dip_mode = Some(mode);
            }
        }

        // Transform partition flag (always `TxNone` for the encoder subset).
        self.decode_tx_part_flag(bs, 0, 0)?;

        if seg_lossless {
            if palette.is_some() {
                return Err(Av2Error::Unsupported(
                    "palette intra prediction in a lossless frame".to_string(),
                ));
            }
            if dip_mode.is_some() {
                return Err(Av2Error::Unsupported(
                    "DIP intra prediction in a lossless frame".to_string(),
                ));
            }
            if cfl_alphas.is_some() {
                return Err(Av2Error::Unsupported(
                    "CfL intra prediction in a lossless frame".to_string(),
                ));
            }
            return self.decode_lossless_intra_block(
                bs, bx4, by4, seg_id, y_mode, y_angle, midx, uv_mode, uv_angle, dpcm_y, dpcm_uv,
            );
        }

        // Luma coefficients.
        let y_params = CoefDecodeParams {
            tx,
            bs: bs as usize,
            plane: 0,
            intra: true,
            fsc: false,
            lossless: seg_lossless,
            uv_mode: uv_mode as usize,
            y_mode: y_mode as usize,
            y_angle,
            mrl_index: mrl_idx,
            y_txtp: 0,
            reduced_txtp_set: self.fh.reduced_txtp_set as i32,
            u_has_cf: 0,
            dq_tbl: self.dq[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize][0],
            bitdepth: self.bd.bitdepth(),
            ss_hor: ss_x != 0,
            ss_ver: ss_y != 0,
            chroma_dctonly: self.seq.chroma_dctonly,
        };
        let y_res = decode_coefs(
            &mut self.msac,
            &mut self.cdf.coef,
            &mut self.cdf.m,
            &self.a_cols[a_idx].lcoef[bx4_in_sb..],
            &self.l_ctx.lcoef[by4_in_sb..],
            &y_params,
            &mut self.dq_y,
        )?;

        // Luma prediction and reconstruction.
        let y_txtp = y_res.txtp;
        if let Some(pal) = &palette {
            // Palette prediction: `rec[r][c] = palette_colors_y[ColorMapY[r][c]]`
            // for the on-screen part of the block (§5.20.8.4).
            let onscreen_w = w.min((self.frame_mi_w * 4).saturating_sub(px_x));
            let onscreen_h = h.min((self.frame_mi_h * 4).saturating_sub(px_y));
            let pl = &mut self.rec.planes[0];
            for r in 0..onscreen_h {
                let dst_row = (px_y + r) * pl.stride + px_x;
                for c in 0..onscreen_w {
                    let idx = pal.map[r * w + c] as usize;
                    pl.data[dst_row + c] = self.bd.clip_pixel(pal.colors[idx.min(pal.size - 1)]);
                }
            }
        } else if let Some(dip) = dip_mode {
            let edge = self.edge_buffer(0, px_x, px_y, w, h, have_left, have_top, 0);
            let pl = &mut self.rec.planes[0];
            let off = pl.offset(px_x, px_y);
            predict_filter_intra_block(
                self.bd,
                &mut pl.data,
                off,
                pl.stride,
                &edge,
                256,
                filter_intra_mode(dip),
                w,
                h,
            );
        } else {
            let edge = self.edge_buffer(0, px_x, px_y, w, h, have_left, have_top, mrl_idx);
            let pl = &mut self.rec.planes[0];
            let off = pl.offset(px_x, px_y);
            predict_intra_block(
                self.bd,
                &mut pl.data,
                off,
                pl.stride,
                &edge,
                256,
                y_mode,
                y_angle,
                w,
                h,
                have_left,
                have_top,
                0,
                mrl_idx,
            );
        }
        if y_res.eob >= 0 {
            for y in 0..t_dim.h as usize {
                for x in 0..t_dim.w as usize {
                    self.grid.set_lr_noskip(bx4 + x, by4 + y, true);
                }
            }
            let pl = &mut self.rec.planes[0];
            let off = pl.offset(px_x, px_y);
            inv_txfm_add(
                self.bd,
                &mut pl.data,
                off,
                pl.stride,
                &mut self.dq_y,
                y_txtp as u32,
                y_res.eob,
                tx,
            );
        }

        // Chroma coefficients and reconstruction.
        if has_chroma {
            let u_params = CoefDecodeParams {
                tx: uvtx,
                bs: uv_bs_idx,
                plane: 1,
                intra: true,
                fsc: false,
                lossless: seg_lossless,
                uv_mode: uv_mode as usize,
                y_mode: y_mode as usize,
                y_angle,
                mrl_index: mrl_idx,
                y_txtp: y_txtp as u32,
                reduced_txtp_set: self.fh.reduced_txtp_set as i32,
                u_has_cf: 0,
                dq_tbl: self.dq[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize][1],
                bitdepth: self.bd.bitdepth(),
                ss_hor: ss_x != 0,
                ss_ver: ss_y != 0,
                chroma_dctonly: self.seq.chroma_dctonly,
            };
            let u_res = decode_coefs(
                &mut self.msac,
                &mut self.cdf.coef,
                &mut self.cdf.m,
                &self.a_cols[a_idx].ccoef[0][(bx4_in_sb >> ss_x)..],
                &self.l_ctx.ccoef[0][(by4_in_sb >> ss_y)..],
                &u_params,
                &mut self.dq_u,
            )?;
            let eob_u = u_res.eob;
            let u_res_ctx = u_res.res_ctx;
            let txtp_u = u_res.txtp;

            let v_params = CoefDecodeParams {
                u_has_cf: (eob_u >= 0) as i32,
                plane: 2,
                dq_tbl: self.dq[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize][2],
                ..u_params
            };
            let v_res = decode_coefs(
                &mut self.msac,
                &mut self.cdf.coef,
                &mut self.cdf.m,
                &self.a_cols[a_idx].ccoef[1][(bx4_in_sb >> ss_x)..],
                &self.l_ctx.ccoef[1][(by4_in_sb >> ss_y)..],
                &v_params,
                &mut self.dq_v,
            )?;
            let eob_v = v_res.eob;
            let v_res_ctx = v_res.res_ctx;
            let txtp_v = v_res.txtp;

            let mut ac_q3 = [0i16; MAX_TX_AREA];
            let cfl_data = if cfl_alphas.is_some() {
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
                Some(&ac_q3)
            } else {
                None
            };
            for pl_idx in 1..=2 {
                let edge = self.edge_buffer(pl_idx, cpx_x, cpx_y, cw, ch, have_left, have_top, 0);
                let dst = &mut self.rec.planes[pl_idx];
                let off = dst.offset(cpx_x, cpx_y);
                if let (Some((au, av)), Some(ac)) = (cfl_alphas, cfl_data) {
                    let mut dc_pred_buf = [BD::Pixel::default(); 64 * 64];
                    let dc_pred = &mut dc_pred_buf[..cw * ch];
                    predict_intra_block(
                        self.bd, dc_pred, 0, cw, &edge, 256, 0, 0, cw, ch, have_left, have_top,
                        pl_idx, 0,
                    );
                    predict_cfl_plane(
                        self.bd,
                        &mut dst.data,
                        off,
                        dst.stride,
                        dc_pred,
                        ac,
                        if pl_idx == 1 { au } else { av },
                        cw,
                        ch,
                    );
                } else {
                    predict_intra_block(
                        self.bd,
                        &mut dst.data,
                        off,
                        dst.stride,
                        &edge,
                        256,
                        uv_mode,
                        0,
                        cw,
                        ch,
                        have_left,
                        have_top,
                        pl_idx,
                        0,
                    );
                }
            }
            if eob_u >= 0 {
                let pl = &mut self.rec.planes[1];
                let off = pl.offset(cpx_x, cpx_y);
                inv_txfm_add(
                    self.bd,
                    &mut pl.data,
                    off,
                    pl.stride,
                    &mut self.dq_u,
                    txtp_u as u32,
                    eob_u,
                    uvtx,
                );
            }
            if eob_v >= 0 {
                let pl = &mut self.rec.planes[2];
                let off = pl.offset(cpx_x, cpx_y);
                inv_txfm_add(
                    self.bd,
                    &mut pl.data,
                    off,
                    pl.stride,
                    &mut self.dq_v,
                    txtp_v as u32,
                    eob_v,
                    uvtx,
                );
            }
            let cbx4 = bx4_in_sb >> ss_x;
            let cby4 = by4_in_sb >> ss_y;
            for i in 0..(uv_tdim.w as usize)
                .min(32 - cbx4)
                .min((self.frame_mi_w >> ss_x).saturating_sub(bx4 >> ss_x))
            {
                self.a_cols[a_idx].uvmode[cbx4 + i] = uv_mode;
                self.a_cols[a_idx].tx_chroma[cbx4 + i] = uv_tdim.lw;
                self.a_cols[a_idx].ccoef[0][cbx4 + i] = u_res_ctx;
                self.a_cols[a_idx].ccoef[1][cbx4 + i] = v_res_ctx;
            }
            for i in 0..(uv_tdim.h as usize)
                .min(32 - cby4)
                .min((self.frame_mi_h >> ss_y).saturating_sub(by4 >> ss_y))
            {
                self.l_ctx.uvmode[cby4 + i] = uv_mode;
                self.l_ctx.tx_chroma[cby4 + i] = uv_tdim.lh;
                self.l_ctx.ccoef[0][cby4 + i] = u_res_ctx;
                self.l_ctx.ccoef[1][cby4 + i] = v_res_ctx;
            }
        }

        // Neighbour context bookkeeping (mirrors the encoder).
        self.store_palette_block(bx4, by4, bw4, bh4, palette.as_ref());
        for i in 0..bw4
            .min(32 - bx4_in_sb)
            .min(self.frame_mi_w.saturating_sub(bx4))
        {
            self.a_cols[a_idx].intra[bx4_in_sb + i] = 1;
            self.a_cols[a_idx].mode[bx4_in_sb + i] = y_mode;
            self.a_cols[a_idx].skip[bx4_in_sb + i] = 0;
            self.a_cols[a_idx].midx[bx4_in_sb + i] = midx;
            self.a_cols[a_idx].tx_luma[bx4_in_sb + i] = t_dim.lw;
            self.a_cols[a_idx].tx[bx4_in_sb + i] = t_dim.lw;
            // An intra block has no inter reference on either list
            // (AV2 Spec v1.0.0 §5.20.4.1 context update).
            self.a_cols[a_idx].ref_frame[0][bx4_in_sb + i] = -1;
            self.a_cols[a_idx].ref_frame[1][bx4_in_sb + i] = -1;
            self.a_cols[a_idx].filter[0][bx4_in_sb + i] = 3;
            self.a_cols[a_idx].filter[1][bx4_in_sb + i] = 3;
        }
        for i in 0..bw4
            .min(32 - bx4_in_sb)
            .min(self.frame_mi_w.saturating_sub(bx4))
        {
            self.a_cols[a_idx].lcoef[bx4_in_sb + i] = y_res.res_ctx;
            self.a_cols[a_idx].txtp[bx4_in_sb + i] = y_txtp;
        }
        for i in 0..bh4
            .min(32 - by4_in_sb)
            .min(self.frame_mi_h.saturating_sub(by4))
        {
            self.l_ctx.intra[by4_in_sb + i] = 1;
            self.l_ctx.mode[by4_in_sb + i] = y_mode;
            self.l_ctx.skip[by4_in_sb + i] = 0;
            self.l_ctx.midx[by4_in_sb + i] = midx;
            self.l_ctx.tx_luma[by4_in_sb + i] = t_dim.lh;
            self.l_ctx.tx[by4_in_sb + i] = t_dim.lh;
            self.l_ctx.ref_frame[0][by4_in_sb + i] = -1;
            self.l_ctx.ref_frame[1][by4_in_sb + i] = -1;
            self.l_ctx.filter[0][by4_in_sb + i] = 3;
            self.l_ctx.filter[1][by4_in_sb + i] = 3;
        }
        for i in 0..bh4
            .min(32 - by4_in_sb)
            .min(self.frame_mi_h.saturating_sub(by4))
        {
            self.l_ctx.lcoef[by4_in_sb + i] = y_res.res_ctx;
            self.l_ctx.txtp[by4_in_sb + i] = y_txtp;
        }

        // Record deblocking block info for luma
        let tx_w = t_dim.w * 4;
        let tx_h = t_dim.h * 4;
        for dy4 in 0..bh4 {
            let y4 = by4 + dy4;
            if y4 >= self.frame_mi_h {
                break;
            }
            for dx4 in 0..bw4 {
                let x4 = bx4 + dx4;
                if x4 >= self.frame_mi_w {
                    break;
                }
                self.grid.set_luma(
                    x4,
                    y4,
                    BlockDeblockInfo {
                        tx_w,
                        tx_h,
                        skip: false,
                        lossless: seg_lossless,
                        block_left: dx4 == 0,
                        block_top: dy4 == 0,
                        tx_left: dx4 % (t_dim.w as usize) == 0,
                        tx_top: dy4 % (t_dim.h as usize) == 0,
                    },
                );
            }
        }

        // Record deblocking block info for chroma
        if has_chroma {
            let uv_tx_w = uv_tdim.w * 4;
            let uv_tx_h = uv_tdim.h * 4;
            let cbx4 = bx4 >> ss_x;
            let cby4 = by4 >> ss_y;
            let cbw4 = (bw4 >> ss_x).max(1);
            let cbh4 = (bh4 >> ss_y).max(1);
            let uv_w4 = (uv_tdim.w as usize).max(1);
            let uv_h4 = (uv_tdim.h as usize).max(1);
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
                            tx_w: uv_tx_w,
                            tx_h: uv_tx_h,
                            skip: false,
                            lossless: seg_lossless,
                            block_left: dx4 == 0,
                            block_top: dy4 == 0,
                            tx_left: dx4 % uv_w4 == 0,
                            tx_top: dy4 % uv_h4 == 0,
                        },
                    );
                }
            }
        }

        Ok(())
    }

    fn decode_inter_block(
        &mut self,
        bs: BlockSize,
        bx4: usize,
        by4: usize,
        tx: usize,
        seg_id: u8,
        all_zero: bool,
        forced_globalmv: bool,
    ) -> Result<()> {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let w = bw4 * 4;
        let h = bh4 * 4;
        let px_x = bx4 * 4;
        let px_y = by4 * 4;
        let a_idx = ((bx4 - self.tile_col_start4) >> 5).min(self.a_cols.len() - 1);
        let bx4_in_sb = bx4 & 31;
        let by4_in_sb = by4 & 31;
        let have_left = bx4 > self.tile_col_start4;
        let have_top = by4 > self.tile_row_start4;
        let t_dim = &TXFM_DIMENSIONS[tx];
        let seg_lossless =
            self.fh.quant.lossless[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize] != 0;

        self.mark_ibc_block(bx4, by4, bw4, bh4);
        self.store_palette_block(bx4, by4, bw4, bh4, None);

        let nb = gather_inter_neighbors(
            &self.a_cols[a_idx],
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

        self.decode_gdf_blk(bx4, by4)?;
        self.decode_cdef_idx(bx4, by4, bw4, bh4, all_zero)?;
        self.decode_ccso_blk(bx4, by4)?;
        self.decode_delta_q_at(bs, bx4, by4, all_zero)?;

        let has_chroma = self.seq.layout != PixelLayout::I400;
        let ss_x = self.seq.ss_hor;
        let ss_y = self.seq.ss_ver;
        let layout_col = (self.seq.layout as usize).saturating_sub(1);
        let uv_bs_idx = if has_chroma {
            SS_BS[bs as usize][layout_col] as usize
        } else {
            bs as usize
        };
        let uvtx = MAX_TXFM_SIZE_FOR_BS[uv_bs_idx] as usize;
        let uv_tdim = &TXFM_DIMENSIONS[uvtx];
        let cw = uv_tdim.w as usize * 4;
        let ch = uv_tdim.h as usize * 4;
        let cpx_x = px_x >> ss_x;
        let cpx_y = px_y >> ss_y;

        let is_comp = if forced_globalmv {
            false
        } else if self.fh.switchable_comp_refs != 0 && (bw4 * bh4 >= 4) {
            let ctx = nb.comp_ctx(&self.refdir);
            self.msac.decode_bool_adapt(self.cdf.m.comp(ctx))? != 0
        } else {
            false
        };

        let (r0_idx, r1_idx, comp_type_idx, saved_mode, mv0, mv1, filter);
        let mut block_motion_mode = 0u8;
        let mut block_warp_params = DEFAULT_WARP_PARAMS;
        let mut cwp_map = 8u8;
        if !is_comp {
            // Single-reference selection. A segment with `SEG_LVL_SKIP` or
            // `SEG_LVL_GLOBALMV` infers `SkipSegFrame` and `GLOBALMV` without
            // coding reference or mode symbols (AV2 Spec v1.0.0 §5.20.7.3,
            // §5.20.7.6, §5.20.7.10).
            let seg_skip = self.seg_syntax.enabled && self.seg_syntax.skip_feature(self.fh, seg_id);
            let seg_globalmv =
                self.seg_syntax.enabled && self.seg_syntax.globalmv_feature(self.fh, seg_id);
            let n_refs = self.fh.n_ref_frames.max(1) as usize;
            let ref_idx = if forced_globalmv {
                self.skip_seg_frame
            } else if n_refs > 1 {
                let mut r = 0usize;
                loop {
                    let ctx = nb.single_ref_ctx(r);
                    let bit = self.msac.decode_bool_adapt(self.cdf.m.single_ref(ctx, r))?;
                    if bit != 0 {
                        break;
                    }
                    r += 1;
                    if r >= n_refs - 1 {
                        break;
                    }
                }
                r
            } else {
                0
            };
            let sngl_ctx = nb.snglref_ctx(ref_idx as i8);
            let warp_signalled = !forced_globalmv
                && self.fh.warp_motion != 0
                && bw4.min(bh4) >= 2
                && !seg_skip
                && !seg_globalmv;
            let mut mode = InterPredMode::GlobalMv as u8;
            let mut drl_idx = 0usize;
            let mut is_warp = false;
            let mut warp_mv = true;
            let mut ref_warp_idx = 0usize;
            let mut warpmv_with_mvd = false;
            // `SIMPLE = 0`, `INTERINTRA = 1`, `LOCALWARP = 2`, `DELTAWARP = 3`,
            // `EXTENDWARP = 4` (§7.3 `MOTION_MODES`).
            let mut motion_mode = 0u8;
            let mut inter_intra = false;
            let mut inter_intra_mode = 0u8;
            if !forced_globalmv {
                if warp_signalled {
                    let ctx = self.warp_mv_count(bx4, by4, bw4, bh4);
                    is_warp = self.msac.decode_bool_adapt(&mut self.cdf.m.is_warp[ctx])? != 0;
                }
                if is_warp {
                    warp_mv = if self.fh.force_integer_mv != 0 {
                        true
                    } else {
                        self.msac.decode_bool_adapt(&mut self.cdf.m.warp_mv)? != 0
                    };
                    motion_mode = 3;
                    if !warp_mv {
                        // `read_motion_mode()` for `WARP_NEWMV`
                        // (§5.20.7.14): `EXTENDWARP` when the block has a
                        // neighbour with a matching reference, otherwise a
                        // `LOCALWARP` symbol when both local and delta warp
                        // are enabled (dav2d `decode.c` WARPNEWMV branch).
                        let ext_ctx = warp_mode_context(
                            &self.warp_mi,
                            self.frame_mi_w,
                            self.frame_mi_h,
                            self.sb_step4,
                            bx4,
                            by4,
                            bw4,
                            bh4,
                        );
                        let has_ext = warp_has_ref_neighbour(
                            &self.warp_mi,
                            self.frame_mi_w,
                            self.frame_mi_h,
                            self.sb_step4,
                            bx4,
                            by4,
                            bw4,
                            bh4,
                            ref_idx as i8,
                        );
                        if has_ext {
                            if (self.seq.motion_modes & (1 << 4)) != 0
                                && self
                                    .msac
                                    .decode_bool_adapt(&mut self.cdf.m.use_extend_warp[ext_ctx])?
                                    != 0
                            {
                                motion_mode = 4;
                            }
                            if motion_mode == 3
                                && (self.seq.motion_modes & ((1 << 2) | (1 << 3)))
                                    == ((1 << 2) | (1 << 3))
                            {
                                let cs_ctx = warp_local_context(
                                    &self.warp_mi,
                                    self.frame_mi_w,
                                    self.frame_mi_h,
                                    self.sb_step4,
                                    bx4,
                                    by4,
                                    bw4,
                                    bh4,
                                );
                                if self
                                    .msac
                                    .decode_bool_adapt(&mut self.cdf.m.use_local_warp[cs_ctx])?
                                    != 0
                                {
                                    motion_mode = 2;
                                }
                            }
                        }
                    }
                    if motion_mode == 3 {
                        // `warp_idx` (§5.20.7.6).
                        for idx in 0..MAX_WARP_REF_CANDIDATES - 1 {
                            let more = self
                                .msac
                                .decode_symbol_adapt(&mut self.cdf.m.warp_idx[idx], 1)?;
                            if more == 0 {
                                ref_warp_idx = idx;
                                break;
                            }
                            ref_warp_idx = idx + 1;
                        }
                    }
                    if warp_mv && ref_warp_idx < 2 {
                        warpmv_with_mvd =
                            self.msac.decode_bool_adapt(&mut self.cdf.m.warp_with_mvd)? != 0;
                    }
                    if !warp_mv {
                        // `read_drl_idx()` for `WARP_NEWMV` (`has_newmv`).
                        let max_drl = self.fh.max_drl_bits.max(1) as usize;
                        let mut n = 0usize;
                        while n < max_drl {
                            if self
                                .msac
                                .decode_bool_adapt(self.cdf.m.drl_bit(n.min(2), sngl_ctx))?
                                == 0
                            {
                                break;
                            }
                            n += 1;
                        }
                        drl_idx = n;
                    }
                } else {
                    let sym = self
                        .msac
                        .decode_symbol_adapt(self.cdf.m.single_mode(sngl_ctx), 2)?;
                    mode = InterPredMode::NearMv as u8 + sym as u8;
                    // `read_interintra_mode(0)` (§5.20.7.15) for
                    // `NEARMV`/`GLOBALMV`/`NEWMV` blocks of at most 64x64
                    // samples (dav2d `decode.c` inter-intra branch).
                    if (self.seq.motion_modes & (1 << 1)) != 0
                        && bw4 * bh4 > 2
                        && bw4.max(bh4) <= 16
                    {
                        let size_group = SPEC_SIZE_GROUP[mi_size_index(bs)] as usize;
                        if self
                            .msac
                            .decode_bool_adapt(&mut self.cdf.m.inter_intra[size_group])?
                            != 0
                        {
                            motion_mode = 1;
                            inter_intra = true;
                            inter_intra_mode = self.msac.decode_symbol_adapt(
                                &mut self.cdf.m.inter_intra_mode[size_group],
                                3,
                            )? as u8;
                            if bw4.min(bh4) > 1
                                && self
                                    .msac
                                    .decode_bool_adapt(&mut self.cdf.m.wedge_inter_intra)?
                                    != 0
                            {
                                let _ = self.read_wedge_idx()?;
                                return Err(Av2Error::Unsupported(
                                    "inter-intra wedge blending".to_string(),
                                ));
                            }
                        }
                    }
                    if mode == InterPredMode::NearMv as u8 || mode == InterPredMode::NewMv as u8 {
                        let max_drl = self.fh.max_drl_bits.max(1) as usize;
                        let mut n = 0usize;
                        while n < max_drl {
                            if self
                                .msac
                                .decode_bool_adapt(self.cdf.m.drl_bit(n.min(2), sngl_ctx))?
                                == 0
                            {
                                break;
                            }
                            n += 1;
                        }
                        drl_idx = n;
                    }
                }
            }

            // Normative `find_mv_stack` (AV2 Spec v1.0.0 §7.12.2) over the
            // per-4x4 mode map, matching the encoder's single-reference stack
            // (`use_ref_frame_mvs = 0`, no reference MV bank, no TIP).
            let cands = find_mv_stack(
                &self.warp_mi,
                self.frame_mi_h,
                self.frame_mi_w,
                self.sb_step4,
                self.tile_row_start4,
                self.tile_row_end4,
                self.tile_col_start4,
                self.tile_col_end4,
                by4,
                bx4,
                bw4,
                bh4,
                ref_idx as i8,
            );
            let pred_mv = cands.get(drl_idx);

            let mut warp_model: Option<([i32; 6], [i32; 4])> = None;
            let mv;
            let flt;
            if is_warp {
                let stack = self.warp_param_stack(ref_idx, bx4, by4, bw4, bh4);
                let params = stack[ref_warp_idx.min(MAX_WARP_REF_CANDIDATES - 1)];
                let block_mv = {
                    let mut m = if warp_mv {
                        self.warp_mv_precision(&params, bw4, bh4, bx4, by4, !warpmv_with_mvd)
                    } else {
                        pred_mv
                    };
                    if (warp_mv && warpmv_with_mvd) || !warp_mv {
                        let diff_mv = read_mv_residual(
                            &mut self.msac,
                            &mut self.cdf.mv,
                            3 + self.fh.mv_precision as usize,
                            self.seq.mvd_sign_derive,
                        )?;
                        m = MvXY {
                            x: mv_clamp_to_integer(m.x as i32 + diff_mv.x as i32) as i16,
                            y: mv_clamp_to_integer(m.y as i32 + diff_mv.y as i32) as i16,
                        };
                    }
                    m
                };
                mv = block_mv;
                let mut matrix;
                match motion_mode {
                    2 => {
                        // `LOCALWARP`: `warp_estimation()` (§7.13.3.23).
                        let (cands, num) = find_warp_samples(
                            &self.warp_mi,
                            self.frame_mi_w,
                            self.frame_mi_h,
                            self.sb_step4,
                            self.tile_row_start4,
                            self.tile_row_end4,
                            self.tile_col_start4,
                            self.tile_col_end4,
                            bx4,
                            by4,
                            bw4,
                            bh4,
                            ref_idx as i8,
                        );
                        matrix = warp_estimation(&cands, num, bw4, bh4, bx4, by4, block_mv);
                    }
                    4 => {
                        // `EXTENDWARP`: `extend_warp_estimation()` (§7.13.3.24).
                        let row = cands.row_offsets.get(drl_idx).copied().unwrap_or(-1);
                        let col = cands.col_offsets.get(drl_idx).copied().unwrap_or(-1);
                        matrix = extend_warp_params(
                            &self.warp_mi,
                            self.frame_mi_w,
                            self.frame_mi_h,
                            self.sb_step4,
                            bx4,
                            by4,
                            bw4,
                            bh4,
                            ref_idx as i8,
                            row,
                            col,
                            block_mv,
                        )
                        .ok_or_else(|| {
                            Av2Error::BitstreamError(
                                "extended warp without a usable neighbour".to_string(),
                            )
                        })?;
                    }
                    _ => {
                        // `DELTAWARP`: `read_warp_delta()` (§5.20.7.7).
                        matrix = params;
                        if !warp_mv {
                            // Only `RefWarpIdx == 0` is reachable without
                            // six-parameter warp delta.
                            let six_param = self.seq.six_param_warp_delta && ref_warp_idx == 1;
                            if six_param || ref_warp_idx == 0 {
                                let high_prec = self.msac.decode_bool_adapt(
                                    &mut self.cdf.m.warp_precision[mi_size_index(bs)],
                                )? != 0;
                                matrix[0] = 0;
                                matrix[1] = 0;
                                let d2 = self.read_warp_delta_param(2, high_prec)?;
                                let d3 = self.read_warp_delta_param(3, high_prec)?;
                                matrix[2] = matrix[2].wrapping_add(d2);
                                matrix[3] = matrix[3].wrapping_add(d3);
                                if six_param {
                                    let d4 = self.read_warp_delta_param(4, high_prec)?;
                                    let d5 = self.read_warp_delta_param(5, high_prec)?;
                                    matrix[4] = matrix[4].wrapping_add(d4);
                                    matrix[5] = matrix[5].wrapping_add(d5);
                                } else {
                                    matrix[4] = matrix[3].wrapping_neg();
                                    matrix[5] = matrix[2];
                                }
                            }
                        }
                        reduce_warp_model(&mut matrix);
                        get_warp_translation(&mut matrix, bw4, bh4, bx4, by4, mv);
                    }
                }
                if warp_mv && w <= 64 && h <= 64 {
                    // `read_interintra_mode(1)` for `WARPMV`.
                    let size_group = SPEC_SIZE_GROUP[mi_size_index(bs)] as usize;
                    let warp_ii = self
                        .msac
                        .decode_bool_adapt(&mut self.cdf.m.warp_inter_intra[size_group])?;
                    if warp_ii != 0 {
                        inter_intra = true;
                        inter_intra_mode = self
                            .msac
                            .decode_symbol_adapt(&mut self.cdf.m.inter_intra_mode[size_group], 3)?
                            as u8;
                        if self
                            .msac
                            .decode_bool_adapt(&mut self.cdf.m.wedge_inter_intra)?
                            != 0
                        {
                            let _ = self.read_wedge_idx()?;
                            return Err(Av2Error::Unsupported(
                                "warp inter-intra wedge blending".to_string(),
                            ));
                        }
                    }
                }
                let abcd = setup_shear(&matrix).ok_or_else(|| {
                    Av2Error::BitstreamError("invalid warped-motion model".to_string())
                })?;
                warp_model = Some((matrix, abcd));
                if !all_zero {
                    self.decode_tx_part_flag(bs, 0, 1)?;
                }
                flt = 0usize;
                block_motion_mode = motion_mode;
                block_warp_params = matrix;
            } else {
                mv = match mode {
                    m if m == InterPredMode::GlobalMv as u8 => MvXY::default(),
                    m if m == InterPredMode::NewMv as u8 => {
                        let diff_mv = read_mv_residual(
                            &mut self.msac,
                            &mut self.cdf.mv,
                            3 + self.fh.mv_precision as usize,
                            self.seq.mvd_sign_derive,
                        )?;
                        MvXY {
                            x: mv_clamp_to_integer(pred_mv.x as i32 + diff_mv.x as i32) as i16,
                            y: mv_clamp_to_integer(pred_mv.y as i32 + diff_mv.y as i32) as i16,
                        }
                    }
                    _ => pred_mv,
                };
                flt = if self.fh.subpel_filter_mode == FilterMode::Switchable {
                    let has_subpel_filter =
                        mode != InterPredMode::GlobalMv as u8 || bw4.min(bh4) == 1;
                    if has_subpel_filter {
                        let filter_ctx = get_filter_ctx_for_block(
                            have_top.then_some(&self.a_cols[a_idx]),
                            have_left.then_some(&self.l_ctx),
                            bx4_in_sb,
                            by4_in_sb,
                            bw4,
                            bh4,
                            (by4 & (self.sb_step4 - 1)) != 0,
                            have_left,
                            false,
                            ref_idx as i8,
                        );
                        self.msac
                            .decode_symbol_adapt(self.cdf.m.interp_filter(filter_ctx), 2)?
                    } else {
                        0usize
                    }
                } else {
                    self.fh.subpel_filter_mode as usize
                };
                block_motion_mode = motion_mode;
                // `read_tx_partition()` follows the complete inter mode info
                // (§5.20.6.2): the MVD and the switchable subpel filter come
                // first, then the transform tree flag.
                if !all_zero {
                    self.decode_tx_part_flag(bs, 0, 1)?;
                }
            }

            let ref_frame = self.ref_frames[ref_idx].ok_or_else(|| {
                Av2Error::BitstreamError("inter block references a missing frame".to_string())
            })?;

            // Inter-intra: predict the intra half before the inter prediction
            // overwrites the block (`read_interintra_mode`, §5.20.7.15).
            let ii_saved = if inter_intra {
                let iy = self.save_interintra_pred(
                    0,
                    px_x,
                    px_y,
                    w,
                    h,
                    have_left,
                    have_top,
                    inter_intra_mode,
                );
                if has_chroma {
                    let iu = self.save_interintra_pred(
                        1,
                        cpx_x,
                        cpx_y,
                        cw,
                        ch,
                        have_left,
                        have_top,
                        inter_intra_mode,
                    );
                    let iv = self.save_interintra_pred(
                        2,
                        cpx_x,
                        cpx_y,
                        cw,
                        ch,
                        have_left,
                        have_top,
                        inter_intra_mode,
                    );
                    Some((iy, Some((iu, iv))))
                } else {
                    Some((iy, None))
                }
            } else {
                None
            };

            if let Some((matrix, abcd)) = warp_model {
                {
                    let rp = &ref_frame.planes[0];
                    let dp = &mut self.rec.planes[0];
                    let off = dp.offset(px_x, px_y);
                    block_warp(
                        self.bd,
                        &mut dp.data,
                        off,
                        dp.stride,
                        rp,
                        px_x,
                        px_y,
                        w,
                        h,
                        0,
                        0,
                        &matrix,
                        &abcd,
                    );
                }
                if has_chroma {
                    for pl_idx in 1..=2 {
                        let rp = &ref_frame.planes[pl_idx];
                        let dp = &mut self.rec.planes[pl_idx];
                        let off = dp.offset(cpx_x, cpx_y);
                        block_warp(
                            self.bd,
                            &mut dp.data,
                            off,
                            dp.stride,
                            rp,
                            cpx_x,
                            cpx_y,
                            cw,
                            ch,
                            ss_x as usize,
                            ss_y as usize,
                            &matrix,
                            &abcd,
                        );
                    }
                }
                self.warped_blocks += 1;
                let bank_pos = ref_idx.min(self.warp_bank.len() - 1);
                warp_bank_insert(&mut self.warp_bank[bank_pos], &matrix);
            } else {
                let mv_y_q16 = mv.y as i32 * 2;
                let mv_x_q16 = mv.x as i32 * 2;
                {
                    let rp = &ref_frame.planes[0];
                    let dp = &mut self.rec.planes[0];
                    let off = dp.offset(px_x, px_y);
                    let stride = dp.stride;
                    mc_predict_block(
                        self.bd,
                        &mut dp.data,
                        off,
                        stride,
                        &rp.data,
                        rp.stride,
                        rp.coded_width as i32,
                        rp.coded_height as i32,
                        px_x as i32,
                        px_y as i32,
                        w,
                        h,
                        mv_y_q16,
                        mv_x_q16,
                        flt,
                    );
                }

                if has_chroma {
                    for pl_idx in 1..=2 {
                        let rp = &ref_frame.planes[pl_idx];
                        let dp = &mut self.rec.planes[pl_idx];
                        let off = dp.offset(cpx_x, cpx_y);
                        let stride = dp.stride;
                        mc_predict_block(
                            self.bd,
                            &mut dp.data,
                            off,
                            stride,
                            &rp.data,
                            rp.stride,
                            rp.coded_width as i32,
                            rp.coded_height as i32,
                            cpx_x as i32,
                            cpx_y as i32,
                            cw,
                            ch,
                            mv_y_q16 >> ss_y,
                            mv_x_q16 >> ss_x,
                            flt,
                        );
                    }
                }
            }

            if let Some((iy, chroma)) = ii_saved {
                let mask = interintra_mask(inter_intra_mode, w, h);
                let dp = &mut self.rec.planes[0];
                let off = dp.offset(px_x, px_y);
                blend_interintra(self.bd, &mut dp.data, off, dp.stride, &iy, &mask, w, h);
                if let Some((iu, iv)) = chroma {
                    let mask_u = interintra_mask(inter_intra_mode, cw, ch);
                    let dp = &mut self.rec.planes[1];
                    let off = dp.offset(cpx_x, cpx_y);
                    blend_interintra(self.bd, &mut dp.data, off, dp.stride, &iu, &mask_u, cw, ch);
                    let mask_v = interintra_mask(inter_intra_mode, cw, ch);
                    let dp = &mut self.rec.planes[2];
                    let off = dp.offset(cpx_x, cpx_y);
                    blend_interintra(self.bd, &mut dp.data, off, dp.stride, &iv, &mask_v, cw, ch);
                }
            }

            r0_idx = ref_idx as i8;
            r1_idx = -1;
            comp_type_idx = 0;
            saved_mode = if is_warp {
                if warp_mv { 17 } else { 18 }
            } else {
                12 + mode
            };
            mv0 = mv;
            mv1 = MvXY::default();
            filter = flt;
        } else {
            // Compound reference pair decoding
            let n_refs = self.fh.n_ref_frames.max(1) as usize;
            let (comp_ref0, comp_ref1) = if n_refs > 1 {
                let same_refs = self.seq.num_same_ref_comp as usize;
                let mut n = 0usize;
                let mut refs = [0usize; 2];
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
                    let bit = if n == 0 && (i == 2 || (i >= n_refs - 2 && i + 1 >= same_refs)) {
                        1u32
                    } else {
                        let ctx = (cnt_cur - cnt_rem + 1).clamp(0, 2) as usize;
                        if n == 0 {
                            self.msac.decode_bool_adapt(self.cdf.m.comp_ref0(ctx, i))?
                        } else {
                            let d = (dir ^ (self.refdir[i] as usize)) & 1;
                            self.msac
                                .decode_bool_adapt(self.cdf.m.comp_ref1(ctx, d, i))?
                        }
                    };
                    if bit != 0 {
                        refs[n] = i;
                        n += 1;
                        if n == 2 {
                            break;
                        }
                        dir = self.refdir[i] as usize;
                    }
                    if maybe_same_ref {
                        maybe_same_ref = (bit == 0) && (i + 1 < same_refs);
                        if bit != 0 {
                            i = i.wrapping_sub(1);
                            cnt_rem += cnt_cur;
                        }
                    }
                    i = i.wrapping_add(1);
                }
                if n < 2 {
                    refs[1] = n_refs - 1;
                    if n == 0 {
                        refs[0] = n_refs - 1 - (same_refs < n_refs) as usize;
                    }
                }
                (refs[0], refs[1])
            } else {
                (0, 0)
            };

            let comp_ctx = nb.compref_ctx(comp_ref0 as i8, comp_ref1 as i8);
            let compound_mode = if comp_ref0 == comp_ref1 {
                let sym = self
                    .msac
                    .decode_symbol_adapt(self.cdf.m.compound_mode_same_refs(comp_ctx), 3)?;
                let mut m = 18 + sym;
                if m >= 21 {
                    m += 1;
                }
                m
            } else {
                let joint_ctx = (self.refdist[comp_ref0] != -self.refdist[comp_ref1]) as usize;
                let is_joint = self
                    .msac
                    .decode_bool_adapt(self.cdf.m.is_joint(joint_ctx))?
                    != 0;
                if is_joint {
                    23 // JOINT_NEWMV
                } else {
                    let sym = self
                        .msac
                        .decode_symbol_adapt(self.cdf.m.compound_mode_non_joint(comp_ctx), 4)?;
                    18 + sym
                }
            };

            // `jmvd_scale_mode` follows `is_joint` and precedes the DRL
            // index (§5.20.7.6).
            let jmvd_scale_mode = if compound_mode == 23 {
                self.msac
                    .decode_symbol_adapt(&mut self.cdf.mv.jmvd_scale_mode, 4)?
            } else {
                0usize
            };

            let mut drl_idx = [0usize; 2];
            if compound_mode != 21
            /* GLOBALMV_GLOBALMV */
            {
                let n_drls = 1 + (compound_mode <= 19) as usize;
                let max_drl_bits = self.fh.max_drl_bits.max(1) as usize;
                let mut n = 0usize;
                let mut ctx = 0usize;
                for r in 0..n_drls {
                    while n < max_drl_bits {
                        if self
                            .msac
                            .decode_bool_adapt(self.cdf.m.drl_bit(ctx.min(2), comp_ctx))?
                            == 0
                        {
                            break;
                        }
                        n += 1;
                        if ctx < 2 {
                            ctx += 1;
                        }
                    }
                    drl_idx[r] = n;
                    n = if compound_mode == 18 && comp_ref0 == comp_ref1 {
                        drl_idx[0] + (drl_idx[0] < max_drl_bits) as usize
                    } else {
                        0
                    };
                    ctx = n.min(2);
                }
                if n_drls == 1 {
                    drl_idx[1] = drl_idx[0];
                }
            }

            // §7.12.2: `has_second_drl` modes (`NEAR_NEARMV`,
            // `NEAR_NEWMV`) build one single-reference stack per list, while
            // every other compound mode uses the compound stack and
            // `RefStackMv[RefMvIdx]` for both lists.
            let (pred_mv0, pred_mv1) = if compound_mode <= 19 {
                // `candList` 0..1 of §7.12.2.10: a compound neighbour also
                // contributes its second-list MV to the stack of that list.
                let mut ref_mi = core::mem::take(&mut self.ref_mi_scratch);
                for (dst, cand) in ref_mi.iter_mut().zip(self.comp_mi.iter()) {
                    let list = if cand.refs[0] == comp_ref0 as i8 {
                        0
                    } else if cand.refs[1] == comp_ref0 as i8 {
                        1
                    } else {
                        -1
                    };
                    *dst = WarpMiInfo {
                        origin_x4: cand.origin_x4,
                        written: true,
                        inter: cand.inter,
                        ref0: if list >= 0 { comp_ref0 as i8 } else { -1 },
                        mv: if list >= 0 {
                            cand.mvs[list as usize]
                        } else {
                            MvXY::default()
                        },
                        ..WarpMiInfo::default()
                    };
                }
                let s0 = find_mv_stack(
                    &ref_mi,
                    self.frame_mi_h,
                    self.frame_mi_w,
                    self.sb_step4,
                    self.tile_row_start4,
                    self.tile_row_end4,
                    self.tile_col_start4,
                    self.tile_col_end4,
                    by4,
                    bx4,
                    bw4,
                    bh4,
                    comp_ref0 as i8,
                );
                for (dst, cand) in ref_mi.iter_mut().zip(self.comp_mi.iter()) {
                    let list = if cand.refs[0] == comp_ref1 as i8 {
                        0
                    } else if cand.refs[1] == comp_ref1 as i8 {
                        1
                    } else {
                        -1
                    };
                    *dst = WarpMiInfo {
                        origin_x4: cand.origin_x4,
                        written: true,
                        inter: cand.inter,
                        ref0: if list >= 0 { comp_ref1 as i8 } else { -1 },
                        mv: if list >= 0 {
                            cand.mvs[list as usize]
                        } else {
                            MvXY::default()
                        },
                        ..WarpMiInfo::default()
                    };
                }
                let s1 = find_mv_stack(
                    &ref_mi,
                    self.frame_mi_h,
                    self.frame_mi_w,
                    self.sb_step4,
                    self.tile_row_start4,
                    self.tile_row_end4,
                    self.tile_col_start4,
                    self.tile_col_end4,
                    by4,
                    bx4,
                    bw4,
                    bh4,
                    comp_ref1 as i8,
                );
                self.ref_mi_scratch = ref_mi;
                (s0.get(drl_idx[0]), s1.get(drl_idx[1]))
            } else {
                let stack = find_compound_mv_stack(
                    &self.comp_mi,
                    self.frame_mi_h,
                    self.frame_mi_w,
                    self.sb_step4,
                    self.tile_row_start4,
                    self.tile_row_end4,
                    self.tile_col_start4,
                    self.tile_col_end4,
                    by4,
                    bx4,
                    bw4,
                    bh4,
                    comp_ref0 as i8,
                    comp_ref1 as i8,
                    self.fh.max_drl_bits.max(1) as usize,
                );
                let pair = stack.get(drl_idx[0]);
                (pair[0], pair[1])
            };

            let (cmv0, cmv1) = match compound_mode {
                18 /* NEARMV_NEARMV */ => (pred_mv0, pred_mv1),
                19 /* NEARMV_NEWMV */ => {
                    let diff_mv = read_mv_residual(
                        &mut self.msac,
                        &mut self.cdf.mv,
                        3 + self.fh.mv_precision as usize,
                        self.seq.mvd_sign_derive,
                    )?;
                    let mv1 = MvXY {
                        x: mv_clamp_to_integer(pred_mv1.x as i32 + diff_mv.x as i32) as i16,
                        y: mv_clamp_to_integer(pred_mv1.y as i32 + diff_mv.y as i32) as i16,
                    };
                    (pred_mv0, mv1)
                }
                20 /* NEWMV_NEARMV */ => {
                    let diff_mv = read_mv_residual(
                        &mut self.msac,
                        &mut self.cdf.mv,
                        3 + self.fh.mv_precision as usize,
                        self.seq.mvd_sign_derive,
                    )?;
                    let mv0 = MvXY {
                        x: mv_clamp_to_integer(pred_mv0.x as i32 + diff_mv.x as i32) as i16,
                        y: mv_clamp_to_integer(pred_mv0.y as i32 + diff_mv.y as i32) as i16,
                    };
                    (mv0, pred_mv1)
                }
                21 /* GLOBALMV_GLOBALMV */ => (MvXY::default(), MvXY::default()),
                22 /* NEWMV_NEWMV */ => {
                    let prec = 3 + self.fh.mv_precision as usize;
                    let (mut x0, mut y0) = read_mv_magnitude(&mut self.msac, &mut self.cdf.mv, prec)?;
                    let (mut x1, mut y1) = read_mv_magnitude(&mut self.msac, &mut self.cdf.mv, prec)?;
                    if y0 != 0 && self.msac.decode_bool_bypass() != 0 {
                        y0 = -y0;
                    }
                    if x0 != 0 && self.msac.decode_bool_bypass() != 0 {
                        x0 = -x0;
                    }
                    if y1 != 0 && self.msac.decode_bool_bypass() != 0 {
                        y1 = -y1;
                    }
                    if x1 != 0 && self.msac.decode_bool_bypass() != 0 {
                        x1 = -x1;
                    }
                    let mv0 = MvXY {
                        x: mv_clamp_to_integer(pred_mv0.x as i32 + x0 as i32) as i16,
                        y: mv_clamp_to_integer(pred_mv0.y as i32 + y0 as i32) as i16,
                    };
                    let mv1 = MvXY {
                        x: mv_clamp_to_integer(pred_mv1.x as i32 + x1 as i32) as i16,
                        y: mv_clamp_to_integer(pred_mv1.y as i32 + y1 as i32) as i16,
                    };
                    (mv0, mv1)
                }
                23 /* JOINT_NEWMV */ => {
                    // `baseList` selection of §5.20.7.13 uses the absolute
                    // frame distances; the projected component carries the
                    // sign of the opposite side.
                    let mut first_dist = self.absrefdist[comp_ref0];
                    let mut second_dist = self.absrefdist[comp_ref1];
                    let mut base = 0usize;
                    if first_dist < second_dist {
                        base = 1;
                        core::mem::swap(&mut first_dist, &mut second_dist);
                    }
                    if (self.refdir[comp_ref0] ^ self.refdir[comp_ref1]) != 0 {
                        second_dist = -second_dist;
                    }
                    let diff_mv = read_mv_residual(
                        &mut self.msac,
                        &mut self.cdf.mv,
                        3 + self.fh.mv_precision as usize,
                        self.seq.mvd_sign_derive,
                    )?;
                    let mut proj = scale_projected_mv(diff_mv, second_dist, first_dist);
                    // `jmvd_scale_mode` (§5.20.7.13): the non-adaptive
                    // variant scales one component only.
                    match jmvd_scale_mode {
                        1 => proj.y = proj.y.saturating_mul(2),
                        2 => proj.x = proj.x.saturating_mul(2),
                        3 => proj.y /= 2,
                        4 => proj.x /= 2,
                        _ => {}
                    }
                    let mut mvs = [pred_mv0, pred_mv1];
                    mvs[base] = MvXY {
                        x: mv_clamp_to_integer(mvs[base].x as i32 + diff_mv.x as i32) as i16,
                        y: mv_clamp_to_integer(mvs[base].y as i32 + diff_mv.y as i32) as i16,
                    };
                    mvs[1 - base] = MvXY {
                        x: mv_clamp_to_integer(mvs[1 - base].x as i32 + proj.x as i32) as i16,
                        y: mv_clamp_to_integer(mvs[1 - base].y as i32 + proj.y as i32) as i16,
                    };
                    (mvs[0], mvs[1])
                }
                _ => (pred_mv0, pred_mv1),
            };

            let mut comp_type = CompInterType::Average;
            let mut cwp_weight = 8i32;
            let mut wedge_idx = 0u8;
            let mut wedge_sign = false;
            let mut mask_sign = false;

            // `read_compound_type()` (§5.20.7.16): `comp_group_idx` is coded
            // for every compound block except thin 4xN/Nx4 blocks, and
            // `compound_type` only when `Wedge_Bits[MiSize]` is non-zero.
            let allow_masked = self.seq.masked_compound && !is_thin_4xn_nx4_block(bw4, bh4);

            if allow_masked {
                let ctx = nb.comp_group_idx_ctx(
                    self.absrefdist[comp_ref0],
                    self.absrefdist[comp_ref1],
                    self.furthest_future_refidx,
                );
                let has_mask = self
                    .msac
                    .decode_bool_adapt(self.cdf.m.comp_group_idx(ctx))?
                    != 0;
                if has_mask {
                    let is_wedge = if wedge_bits(bw4, bh4) != 0 {
                        self.msac.decode_bool_adapt(self.cdf.m.comp_type())? == 0
                    } else {
                        false
                    };
                    if is_wedge {
                        comp_type = CompInterType::Wedge;
                        wedge_idx = self.read_wedge_idx()?;
                        wedge_sign = self.msac.decode_bool_bypass() != 0;
                    } else {
                        comp_type = CompInterType::DifferenceWeighted;
                        mask_sign = self.msac.decode_bool_bypass() != 0;
                    }
                }
            }

            if comp_type == CompInterType::Average
                && self.seq.cwp
                && (compound_mode == 18 || compound_mode == 23)
            {
                let mut n = 0usize;
                while n < 4 {
                    if self.msac.decode_bool_adapt(self.cdf.m.cwp_idx(n))? == 0 {
                        break;
                    }
                    n += 1;
                }
                const CWP_WEIGHTING_FACTOR: [[i8; 5]; 2] = [[8, 12, 4, 10, 6], [8, 12, 4, 20, -4]];
                let is_same_side = (self.refdir[comp_ref0] ^ self.refdir[comp_ref1]) == 0;
                let wt = CWP_WEIGHTING_FACTOR[is_same_side as usize][n] as i32;
                if wt != 8 {
                    comp_type = CompInterType::DistanceWeighted;
                    cwp_weight = wt;
                    cwp_map = wt as u8;
                }
            }

            // The switchable subpel filter and the transform partition follow
            // the complete compound mode info (§5.20.6.2 / §5.20.7.6).
            let flt = if self.fh.subpel_filter_mode == FilterMode::Switchable {
                let has_subpel_filter =
                    compound_mode <= 23 && (compound_mode != 21 || bw4.min(bh4) == 1);
                if has_subpel_filter {
                    let filter_ctx = get_filter_ctx_for_block(
                        have_top.then_some(&self.a_cols[a_idx]),
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
                    self.msac
                        .decode_symbol_adapt(self.cdf.m.interp_filter(filter_ctx), 2)?
                } else {
                    0usize
                }
            } else {
                self.fh.subpel_filter_mode as usize
            };

            if !all_zero {
                self.decode_tx_part_flag(bs, 0, 1)?;
            }

            let r0 = self.ref_frames[comp_ref0].ok_or_else(|| {
                Av2Error::BitstreamError(format!(
                    "inter compound block references missing ref0 {}",
                    comp_ref0
                ))
            })?;
            let r1 = self.ref_frames[comp_ref1].ok_or_else(|| {
                Av2Error::BitstreamError(format!(
                    "inter compound block references missing ref1 {}",
                    comp_ref1
                ))
            })?;

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
                cmv0.y as i32 * 2,
                cmv0.x as i32 * 2,
                flt,
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
                cmv1.y as i32 * 2,
                cmv1.x as i32 * 2,
                flt,
            );

            let dp0 = &mut self.rec.planes[0];
            let off0 = dp0.offset(px_x, px_y);
            let stride0 = dp0.stride;

            match comp_type {
                CompInterType::Average => {
                    blend_compound_prep(
                        self.bd,
                        &mut dp0.data,
                        off0,
                        stride0,
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
                    blend_compound_prep(
                        self.bd,
                        &mut dp0.data,
                        off0,
                        stride0,
                        &self.scratch_mc0[..w * h],
                        &self.scratch_mc1[..w * h],
                        w,
                        h,
                        CompInterType::DistanceWeighted,
                        cwp_weight,
                        None,
                    );
                }
                CompInterType::Wedge => {
                    let mask = WEDGE_TABLES
                        .get_mask(bs, wedge_idx as usize, 0, 0)
                        .ok_or_else(|| {
                            Av2Error::BitstreamError(format!(
                                "missing wedge mask bs {:?} idx {}",
                                bs, wedge_idx
                            ))
                        })?;
                    let (p0, p1) = if wedge_sign {
                        (&self.scratch_mc1[..w * h], &self.scratch_mc0[..w * h])
                    } else {
                        (&self.scratch_mc0[..w * h], &self.scratch_mc1[..w * h])
                    };
                    blend_compound_prep(
                        self.bd,
                        &mut dp0.data,
                        off0,
                        stride0,
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
                        mask_sign,
                        ss_x as usize,
                        ss_y as usize,
                        &mut self.scratch_diff_mask[..w * h],
                        &mut self.scratch_chroma_diff_mask[..cw * ch],
                    );
                    let (p0, p1) = if mask_sign {
                        (&self.scratch_mc1[..w * h], &self.scratch_mc0[..w * h])
                    } else {
                        (&self.scratch_mc0[..w * h], &self.scratch_mc1[..w * h])
                    };
                    blend_compound_prep(
                        self.bd,
                        &mut dp0.data,
                        off0,
                        stride0,
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

            if has_chroma {
                let c_area = cw * ch;
                for pl in 1..=2 {
                    let rp0 = &r0.planes[pl];
                    let rp1 = &r1.planes[pl];
                    let dp = &mut self.rec.planes[pl];
                    let off = dp.offset(cpx_x, cpx_y);
                    let stride = dp.stride;

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
                        (cmv0.y as i32 * 2) >> ss_y,
                        (cmv0.x as i32 * 2) >> ss_x,
                        flt,
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
                        (cmv1.y as i32 * 2) >> ss_y,
                        (cmv1.x as i32 * 2) >> ss_x,
                        flt,
                    );

                    match comp_type {
                        CompInterType::Average => {
                            blend_compound_prep(
                                self.bd,
                                &mut dp.data,
                                off,
                                stride,
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
                            blend_compound_prep(
                                self.bd,
                                &mut dp.data,
                                off,
                                stride,
                                &self.scratch_mc0[..c_area],
                                &self.scratch_mc1[..c_area],
                                cw,
                                ch,
                                CompInterType::DistanceWeighted,
                                cwp_weight,
                                None,
                            );
                        }
                        CompInterType::Wedge => {
                            let mask = WEDGE_TABLES
                                .get_mask(bs, wedge_idx as usize, ss_x as usize, ss_y as usize)
                                .ok_or_else(|| {
                                    Av2Error::BitstreamError(format!(
                                        "missing chroma wedge mask bs {:?} idx {}",
                                        bs, wedge_idx
                                    ))
                                })?;
                            let (p0, p1) = if wedge_sign {
                                (&self.scratch_mc1[..c_area], &self.scratch_mc0[..c_area])
                            } else {
                                (&self.scratch_mc0[..c_area], &self.scratch_mc1[..c_area])
                            };
                            blend_compound_prep(
                                self.bd,
                                &mut dp.data,
                                off,
                                stride,
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
                            let (p0, p1) = if mask_sign {
                                (&self.scratch_mc1[..c_area], &self.scratch_mc0[..c_area])
                            } else {
                                (&self.scratch_mc0[..c_area], &self.scratch_mc1[..c_area])
                            };
                            blend_compound_prep(
                                self.bd,
                                &mut dp.data,
                                off,
                                stride,
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
            }

            r0_idx = comp_ref0 as i8;
            r1_idx = comp_ref1 as i8;
            comp_type_idx = match comp_type {
                CompInterType::Average | CompInterType::DistanceWeighted => 1,
                CompInterType::Wedge => 2,
                CompInterType::DifferenceWeighted => 3,
            };
            saved_mode = compound_mode as u8;
            mv0 = cmv0;
            mv1 = cmv1;
            filter = flt;
        }

        let mut y_res = crate::av2::dec::syntax::CoefDecodeResult {
            eob: -1,
            txtp: 0,
            res_ctx: 0x40,
        };
        let mut u_res_ctx = 0x40u8;
        let mut v_res_ctx = 0x40u8;
        if !all_zero {
            let y_params = CoefDecodeParams {
                tx,
                bs: bs as usize,
                plane: 0,
                intra: false,
                fsc: false,
                lossless: seg_lossless,
                uv_mode: 0,
                y_mode: 0,
                y_angle: 0,
                mrl_index: 0,
                y_txtp: 0,
                reduced_txtp_set: self.fh.reduced_txtp_set as i32,
                u_has_cf: 0,
                dq_tbl: self.dq[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize][0],
                bitdepth: self.bd.bitdepth(),
                ss_hor: ss_x != 0,
                ss_ver: ss_y != 0,
                chroma_dctonly: self.seq.chroma_dctonly,
            };
            y_res = decode_coefs(
                &mut self.msac,
                &mut self.cdf.coef,
                &mut self.cdf.m,
                &self.a_cols[a_idx].lcoef[bx4_in_sb..],
                &self.l_ctx.lcoef[by4_in_sb..],
                &y_params,
                &mut self.dq_y,
            )?;
            if y_res.eob >= 0 {
                for y in 0..t_dim.h as usize {
                    for x in 0..t_dim.w as usize {
                        self.grid.set_lr_noskip(bx4 + x, by4 + y, true);
                    }
                }
                let dp = &mut self.rec.planes[0];
                let off = dp.offset(px_x, px_y);
                let stride = dp.stride;
                inv_txfm_add(
                    self.bd,
                    &mut dp.data,
                    off,
                    stride,
                    &mut self.dq_y,
                    y_res.txtp as u32,
                    y_res.eob,
                    tx,
                );
            }

            if has_chroma {
                let u_params = CoefDecodeParams {
                    tx: uvtx,
                    bs: uv_bs_idx,
                    plane: 1,
                    intra: false,
                    fsc: false,
                    lossless: seg_lossless,
                    uv_mode: 0,
                    y_mode: 0,
                    y_angle: 0,
                    mrl_index: 0,
                    y_txtp: y_res.txtp as u32,
                    reduced_txtp_set: self.fh.reduced_txtp_set as i32,
                    u_has_cf: 0,
                    dq_tbl: self.dq[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize][1],
                    bitdepth: self.bd.bitdepth(),
                    ss_hor: ss_x != 0,
                    ss_ver: ss_y != 0,
                    chroma_dctonly: self.seq.chroma_dctonly,
                };
                let u_res = decode_coefs(
                    &mut self.msac,
                    &mut self.cdf.coef,
                    &mut self.cdf.m,
                    &self.a_cols[a_idx].ccoef[0][(bx4_in_sb >> ss_x)..],
                    &self.l_ctx.ccoef[0][(by4_in_sb >> ss_y)..],
                    &u_params,
                    &mut self.dq_u,
                )?;
                let eob_u = u_res.eob;
                u_res_ctx = u_res.res_ctx;
                if eob_u >= 0 {
                    let dp = &mut self.rec.planes[1];
                    let off = dp.offset(cpx_x, cpx_y);
                    let stride = dp.stride;
                    inv_txfm_add(
                        self.bd,
                        &mut dp.data,
                        off,
                        stride,
                        &mut self.dq_u,
                        u_res.txtp as u32,
                        eob_u,
                        uvtx,
                    );
                }
                let v_params = CoefDecodeParams {
                    plane: 2,
                    u_has_cf: (eob_u >= 0) as i32,
                    dq_tbl: self.dq[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize][2],
                    ..u_params
                };
                let v_res = decode_coefs(
                    &mut self.msac,
                    &mut self.cdf.coef,
                    &mut self.cdf.m,
                    &self.a_cols[a_idx].ccoef[1][(bx4_in_sb >> ss_x)..],
                    &self.l_ctx.ccoef[1][(by4_in_sb >> ss_y)..],
                    &v_params,
                    &mut self.dq_v,
                )?;
                let eob_v = v_res.eob;
                v_res_ctx = v_res.res_ctx;
                if eob_v >= 0 {
                    let dp = &mut self.rec.planes[2];
                    let off = dp.offset(cpx_x, cpx_y);
                    let stride = dp.stride;
                    inv_txfm_add(
                        self.bd,
                        &mut dp.data,
                        off,
                        stride,
                        &mut self.dq_v,
                        v_res.txtp as u32,
                        eob_v,
                        uvtx,
                    );
                }
            }
        }

        // Neighbour context bookkeeping (mirrors the encoder).
        for i in 0..bw4
            .min(32 - bx4_in_sb)
            .min(self.frame_mi_w.saturating_sub(bx4))
        {
            self.a_cols[a_idx].intra[bx4_in_sb + i] = 0;
            self.a_cols[a_idx].midx[bx4_in_sb + i] = 0xFF;
            self.a_cols[a_idx].skip[bx4_in_sb + i] = all_zero as u8;
            self.a_cols[a_idx].tx_luma[bx4_in_sb + i] = t_dim.lw;
            self.a_cols[a_idx].tx[bx4_in_sb + i] = t_dim.lw;
            self.a_cols[a_idx].mv[0][bx4_in_sb + i] = if block_motion_mode >= 2 {
                warp_mv_at(&block_warp_params, by4 + bh4 - 1, bx4 + i)
            } else {
                mv0
            };
            self.a_cols[a_idx].mv[1][bx4_in_sb + i] = mv1;
            self.a_cols[a_idx].ref_frame[0][bx4_in_sb + i] = r0_idx;
            self.a_cols[a_idx].ref_frame[1][bx4_in_sb + i] = r1_idx;
            self.a_cols[a_idx].comp_type[bx4_in_sb + i] = comp_type_idx;
            self.a_cols[a_idx].mode[bx4_in_sb + i] = saved_mode;
            self.a_cols[a_idx].filter[0][bx4_in_sb + i] = filter as u8;
            self.a_cols[a_idx].filter[1][bx4_in_sb + i] = filter as u8;
            self.a_cols[a_idx].lcoef[bx4_in_sb + i] = y_res.res_ctx;
        }
        for i in 0..bh4
            .min(32 - by4_in_sb)
            .min(self.frame_mi_h.saturating_sub(by4))
        {
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
            self.l_ctx.mode[by4_in_sb + i] = saved_mode;
            self.l_ctx.filter[0][by4_in_sb + i] = filter as u8;
            self.l_ctx.filter[1][by4_in_sb + i] = filter as u8;
            self.l_ctx.lcoef[by4_in_sb + i] = y_res.res_ctx;
        }
        if has_chroma {
            let cbx4 = bx4_in_sb >> ss_x;
            let cby4 = by4_in_sb >> ss_y;
            for i in 0..(uv_tdim.w as usize)
                .min(32 - cbx4)
                .min((self.frame_mi_w >> ss_x).saturating_sub(bx4 >> ss_x))
            {
                self.a_cols[a_idx].ccoef[0][cbx4 + i] = u_res_ctx;
                self.a_cols[a_idx].ccoef[1][cbx4 + i] = v_res_ctx;
            }
            for i in 0..(uv_tdim.h as usize)
                .min(32 - cby4)
                .min((self.frame_mi_h >> ss_y).saturating_sub(by4 >> ss_y))
            {
                self.l_ctx.ccoef[0][cby4 + i] = u_res_ctx;
                self.l_ctx.ccoef[1][cby4 + i] = v_res_ctx;
            }
        }

        // Record deblocking block info for luma
        let tx_w = t_dim.w * 4;
        let tx_h = t_dim.h * 4;
        for dy4 in 0..bh4 {
            let y4 = by4 + dy4;
            if y4 >= self.frame_mi_h {
                break;
            }
            for dx4 in 0..bw4 {
                let x4 = bx4 + dx4;
                if x4 >= self.frame_mi_w {
                    break;
                }
                self.grid.set_luma(
                    x4,
                    y4,
                    BlockDeblockInfo {
                        tx_w,
                        tx_h,
                        skip: all_zero,
                        lossless: seg_lossless,
                        block_left: dx4 == 0,
                        block_top: dy4 == 0,
                        tx_left: dx4 % (t_dim.w as usize) == 0,
                        tx_top: dy4 % (t_dim.h as usize) == 0,
                    },
                );
            }
        }

        // Record deblocking block info for chroma
        if has_chroma {
            let uv_tx_w = uv_tdim.w * 4;
            let uv_tx_h = uv_tdim.h * 4;
            let cbx4 = bx4 >> ss_x;
            let cby4 = by4 >> ss_y;
            let cbw4 = (bw4 >> ss_x).max(1);
            let cbh4 = (bh4 >> ss_y).max(1);
            let uv_w4 = (uv_tdim.w as usize).max(1);
            let uv_h4 = (uv_tdim.h as usize).max(1);
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
                            tx_w: uv_tx_w,
                            tx_h: uv_tx_h,
                            skip: all_zero,
                            lossless: seg_lossless,
                            block_left: dx4 == 0,
                            block_top: dy4 == 0,
                            tx_left: dx4 % uv_w4 == 0,
                            tx_top: dy4 % uv_h4 == 0,
                        },
                    );
                }
            }
        }

        // Record the decoded inter mode for `generate_points_from_corners`
        // (§7.12.2.3) and the compound motion-vector stack before the next
        // block can use it.
        let (map_refs, map_mvs, map_cwp) = if r1_idx >= 0 {
            (
                [r0_idx, r1_idx],
                [mv0, mv1],
                if comp_type_idx == 3 { cwp_map } else { 8 },
            )
        } else {
            ([r0_idx, -1], [mv0, MvXY::default()], 8)
        };
        self.mark_mi_inter(
            bs,
            bx4,
            by4,
            block_motion_mode,
            block_warp_params,
            map_refs,
            map_mvs,
            map_cwp,
        );

        Ok(())
    }

    /// Marks the current block footprint as intra in the per-4x4 warp map.
    fn mark_mi_default(&mut self, bs: BlockSize, bx4: usize, by4: usize) {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let info = WarpMiInfo::block(bx4, by4, bw4, bh4);
        for y in 0..bh4 {
            let row = by4 + y;
            if row >= self.frame_mi_h {
                break;
            }
            for x in 0..bw4 {
                let col = bx4 + x;
                if col >= self.frame_mi_w {
                    break;
                }
                self.warp_mi[row * self.frame_mi_w + col] = info;
                self.comp_mi[row * self.frame_mi_w + col] = CompoundMiInfo {
                    origin_x4: bx4 as u16,
                    ..CompoundMiInfo::default()
                };
            }
        }
    }

    /// Records the decoded inter mode for `generate_points_from_corners` and the
    /// compound `find_mv_stack` branch (§7.12.2.3 / §7.12.2.10).
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
        info.motion_mode = motion_mode;
        info.warp = warp;
        let comp_info = CompoundMiInfo {
            origin_x4: bx4 as u16,
            inter: true,
            refs,
            mvs,
            cwp,
        };
        for y in 0..bh4 {
            let row = by4 + y;
            if row >= self.frame_mi_h {
                break;
            }
            for x in 0..bw4 {
                let col = bx4 + x;
                if col >= self.frame_mi_w {
                    break;
                }
                self.warp_mi[row * self.frame_mi_w + col] = info;
                self.comp_mi[row * self.frame_mi_w + col] = comp_info;
            }
        }
    }

    /// `WarpMvCount` (§7.11.4 `scan_point_warp_ctx`) for the `is_warp` context.
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
            if row >= self.frame_mi_h || col >= self.frame_mi_w {
                return;
            }
            let info = self.warp_mi[row * self.frame_mi_w + col];
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

    /// Builds the `WarpParamStack` for the current block (§7.12.2): corner
    /// candidate, warp bank, global motion (identity), then defaults.
    fn warp_param_stack(
        &self,
        ref_idx: usize,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
    ) -> [[i32; 6]; 4] {
        let sb = self.sb_step4.max(1);
        let mut corner = generate_points_from_corners(
            &self.warp_mi,
            self.frame_mi_w,
            self.frame_mi_h,
            by4,
            bx4,
            bw4,
            bh4,
            sb,
            sb,
            0,
        );
        if corner.is_none() && bw4 <= 16 {
            corner = generate_points_from_corners(
                &self.warp_mi,
                self.frame_mi_w,
                self.frame_mi_h,
                by4,
                bx4,
                bw4,
                bh4,
                sb,
                sb,
                1,
            );
        }
        let bank = &self.warp_bank[ref_idx.min(self.warp_bank.len() - 1)];
        build_warp_param_stack(corner, bank, &DEFAULT_WARP_PARAMS)
    }

    /// `get_warp_motion_vector` (§7.12.2.2) for an arbitrary precision; only
    /// the eighth-pel and frame-precision variants are used by AV2 syntax.
    #[allow(clippy::too_many_arguments)]
    fn warp_mv_precision(
        &self,
        params: &[i32; 6],
        bw4: usize,
        bh4: usize,
        bx4: usize,
        by4: usize,
        eighth_pel: bool,
    ) -> MvXY {
        if eighth_pel {
            return warp_motion_vector(
                params,
                bw4,
                bh4,
                bx4,
                by4,
                self.frame_mi_w,
                self.frame_mi_h,
            );
        }
        let bw = (bw4 * 4) as i64;
        let bh = (bh4 * 4) as i64;
        let x = bx4 as i64 * 4 + bw / 2 - 1;
        let y = by4 as i64 * 4 + bh / 2 - 1;
        let unary = 1i64 << WARPEDMODEL_PREC_BITS;
        let xc = (params[2] as i64 - unary) * x + params[3] as i64 * y + params[0] as i64;
        let yc = params[4] as i64 * x + (params[5] as i64 - unary) * y + params[1] as i64;
        let mut mv_y = (round2_signed(yc, WARPEDMODEL_PREC_BITS - 2) * 2)
            .clamp(MV_LOW as i64 + 1, MV_UPP as i64 - 1) as i32;
        let mut mv_x = (round2_signed(xc, WARPEDMODEL_PREC_BITS - 2) * 2)
            .clamp(MV_LOW as i64 + 1, MV_UPP as i64 - 1) as i32;
        let low = -(((by4 + bh4) * 4 * 8) as i32) - MV_BORDER;
        let high = ((self.frame_mi_h - by4) * 4 * 8) as i32 + MV_BORDER;
        mv_y = mv_y.clamp(low, high);
        let low = -(((bx4 + bw4) * 4 * 8) as i32) - MV_BORDER;
        let high = ((self.frame_mi_w - bx4) * 4 * 8) as i32 + MV_BORDER;
        mv_x = mv_x.clamp(low, high);
        MvXY {
            x: mv_x as i16,
            y: mv_y as i16,
        }
    }

    /// `read_warp_delta_param()` (§5.20.7.7).
    fn read_warp_delta_param(&mut self, idx: usize, high_prec: bool) -> Result<i32> {
        let cdf_idx = usize::from(idx == 3 || idx == 4);
        let mut v = self
            .msac
            .decode_symbol_adapt(&mut self.cdf.m.warp_delta_param_low[cdf_idx], 7)?;
        if high_prec && v == 7 {
            v += self
                .msac
                .decode_symbol_adapt(&mut self.cdf.m.warp_delta_param_high[cdf_idx], 7)?;
        }
        let mut value = v as i32;
        if v != 0
            && self
                .msac
                .decode_bool_adapt(&mut self.cdf.m.warp_delta_param_sign)?
                != 0
        {
            value = -value;
        }
        Ok(value << (10 + 1 - high_prec as u32))
    }

    /// Marks the block footprint in `IBCCoded`, clears the neighbour
    /// `intrabc` flags, and advances the local IBC buffer ring
    /// (`decode_block()` / `update_ibc_buffers`, AV2 Spec v1.0.0 §5.20.4.1).
    fn mark_ibc_block(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize) {
        for y in 0..bh4 {
            let row = by4 + y;
            if row >= self.frame_mi_h {
                break;
            }
            for x in 0..bw4 {
                let col = bx4 + x;
                if col >= self.frame_mi_w {
                    break;
                }
                let idx = row * self.frame_mi_w + col;
                self.ibc_coded[idx] = true;
                self.intrabc_flags[idx] = false;
            }
        }
        self.ibc
            .update_sb(by4 as i32, bx4 as i32, self.fh.sb128 != 0);
    }

    /// `TileIntrabcCdf` context: sum of the two neighbour `intrabc` flags
    /// (§5.20.5.3, §8.3.2).
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
            if x >= self.frame_mi_w || y >= self.frame_mi_h {
                0
            } else {
                self.intrabc_flags[y * self.frame_mi_w + x] as u8
            }
        };
        let have_top_in_sb = (by4 & (self.sb_step4 - 1)) != 0;
        let mut nb = [0u8; 2];
        let mut idx = 0usize;
        if have_left && bh4 > 0 {
            nb[idx] = probe(bx4 - 1, by4 + bh4 - 1);
            idx += 1;
        }
        if have_top_in_sb && bw4 > 0 && idx < 2 {
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

    /// Records `PaletteSizes` / `PaletteColors` across the block footprint
    /// (§5.20.4.1 block decode bookkeeping).
    fn store_palette_block(
        &mut self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        pal: Option<&PaletteBlock>,
    ) {
        let (size, colors) = match pal {
            Some(p) => (p.size as u8, p.colors),
            None => (0u8, [0i32; MAX_PALETTE_COLORS]),
        };
        for y in 0..bh4 {
            let row = by4 + y;
            if row >= self.frame_mi_h {
                break;
            }
            for x in 0..bw4 {
                let col = bx4 + x;
                if col >= self.frame_mi_w {
                    break;
                }
                let idx = row * self.frame_mi_w + col;
                self.palette_sizes[idx] = size;
                self.palette_colors[idx] = colors;
            }
        }
    }

    /// Builds the `RefStackMv` candidate list used by `read_intrabc_info()`
    /// (§7.12.2.20): the spatial candidates followed by the four
    /// `add_to_ref_bv` defaults, capped at `max_bvp_drl_bits_minus_1 + 2`.
    fn bvp_stack(
        &self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        have_left: bool,
        have_top: bool,
    ) -> Vec<MvXY> {
        let max_entries = self.fh.max_bvp_drl_bits.max(1) as usize + 1;
        let mut stack: Vec<MvXY> = Vec::with_capacity(max_entries);
        let probe = |x: usize, y: usize| -> Option<MvXY> {
            if x >= self.frame_mi_w || y >= self.frame_mi_h {
                return None;
            }
            let idx = y * self.frame_mi_w + x;
            if self.intrabc_flags[idx] {
                None
            } else {
                Some(self.warp_mi[idx].mv)
            }
        };
        let push = |mv: MvXY, stack: &mut Vec<MvXY>| {
            if !mv.is_zero() && !stack.contains(&mv) && stack.len() < max_entries {
                stack.push(mv);
            }
        };
        if have_top {
            for i in 0..bw4 {
                if let Some(mv) = probe(bx4 + i, by4 - 1) {
                    push(mv, &mut stack);
                }
            }
        }
        if have_left {
            for j in 0..bh4 {
                if let Some(mv) = probe(bx4 - 1, by4 + j) {
                    push(mv, &mut stack);
                }
            }
        }
        // `add_to_ref_bv(0, -Block_Height[SbSize])` then the three smaller
        // defaults, added in this order while `NumMvFound < max + 1`.
        let sb_px = 64i32 << self.fh.sb128;
        let defaults = [
            MvXY {
                x: 0,
                y: -(sb_px * 8) as i16,
            },
            MvXY {
                x: (-8 * (sb_px + 256)) as i16,
                y: 0,
            },
            MvXY {
                x: 0,
                y: -((bh4 as i32) * 32) as i16,
            },
            MvXY {
                x: -((bw4 as i32) * 32) as i16,
                y: 0,
            },
        ];
        for mv in defaults {
            if stack.len() >= max_entries {
                break;
            }
            stack.push(mv);
        }
        stack
    }

    /// `is_mv_valid(BlockMvs[0])` for IntraBC (§6.19.7.12): tile bounds,
    /// causality, and the local IBC buffer constraints.
    fn check_intrabc_mv(&self, bs: BlockSize, bx4: usize, by4: usize, mv: MvXY) -> Result<()> {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as i32;
        let bh4 = b_dim[1] as i32;
        let bottom_border = ((mv.y & 7) != 0) as i32;
        let right_border = ((mv.x & 7) != 0) as i32;
        let delta_row = (mv.y >> 3) as i32;
        let delta_col = (mv.x >> 3) as i32;
        let src_top = (by4 as i32) * 4 + delta_row;
        let src_left = (bx4 as i32) * 4 + delta_col;
        let src_bottom = src_top + bh4 * 4 + bottom_border;
        let src_right = src_left + bw4 * 4 + right_border;
        let row_start = (self.tile_row_start4 as i32) * 4;
        let col_start = (self.tile_col_start4 as i32) * 4;
        let row_end = (self.tile_row_end4 as i32) * 4;
        let col_end = (self.tile_col_end4 as i32) * 4;
        if src_top < row_start
            || src_left < col_start
            || src_bottom > row_end
            || src_right > col_end
        {
            return Err(Av2Error::BitstreamError(format!(
                "IntraBC reference {src_left},{src_top}..{src_right},{src_bottom} outside the tile"
            )));
        }
        if self.fh.allow_local_intrabc != 0 {
            let sb128 = self.fh.sb128 != 0;
            let check = |x: i32, y: i32| -> bool {
                if x >= (bx4 as i32) * 4 && y >= (by4 as i32) * 4 {
                    return false;
                }
                let mx = (x >> 2) as usize;
                let my = (y >> 2) as usize;
                if mx >= self.frame_mi_w || my >= self.frame_mi_h {
                    return false;
                }
                if !self.ibc_coded[my * self.frame_mi_w + mx] {
                    return false;
                }
                self.ibc.valid_location_sb(x >> 2, y >> 2, sb128)
            };
            if !check(src_left, src_top)
                || !check(src_right - 1 - right_border, src_bottom - 1 - bottom_border)
            {
                return Err(Av2Error::BitstreamError(
                    "IntraBC reference outside the local IBC buffer".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Decodes `read_intrabc_info()` and reconstructs the block
    /// (AV2 Spec v1.0.0 §5.20.5.4).
    fn decode_intrabc_block(
        &mut self,
        bs: BlockSize,
        bx4: usize,
        by4: usize,
        seg_id: u8,
        skip_txfm: bool,
        allow_sct: bool,
    ) -> Result<()> {
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        let bw4 = b_dim[0] as usize;
        let bh4 = b_dim[1] as usize;
        let w = bw4 * 4;
        let h = bh4 * 4;
        let px_x = bx4 * 4;
        let px_y = by4 * 4;
        let a_idx = ((bx4 - self.tile_col_start4) >> 5).min(self.a_cols.len() - 1);
        let bx4_in_sb = bx4 & 31;
        let by4_in_sb = by4 & 31;
        let have_left = bx4 > self.tile_col_start4;
        let have_top = by4 > self.tile_row_start4;
        let seg_lossless =
            self.fh.quant.lossless[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize] != 0;
        let ss_x = self.seq.ss_hor;
        let ss_y = self.seq.ss_ver;
        let has_chroma = self.seq.layout != PixelLayout::I400;

        let intrabc_mode = self.msac.decode_bool_adapt(&mut self.cdf.m.intrabc_mode)?;
        let max_m = self.fh.max_bvp_drl_bits.max(1) as usize;
        let mut drl_idx = 0usize;
        for idx in 0..max_m {
            if self.msac.decode_bool_bypass() == 0 {
                drl_idx = idx;
                break;
            }
            drl_idx = idx + 1;
        }
        let force_integer = match self.seq.force_integer_mv {
            AdaptiveBoolean::On => true,
            AdaptiveBoolean::Off => false,
            AdaptiveBoolean::Adaptive => self.fh.force_integer_mv != 0,
        };
        let mut is_qpel = !force_integer;
        if !force_integer && intrabc_mode == 0 {
            is_qpel = self
                .msac
                .decode_bool_adapt(&mut self.cdf.m.intrabc_precision)?
                != 0;
        }
        let stack = self.bvp_stack(bx4, by4, bw4, bh4, have_left, have_top);
        let pred = stack.get(drl_idx).copied().unwrap_or_default();
        let mv = if intrabc_mode != 0 {
            pred
        } else {
            let base = if is_qpel {
                pred
            } else {
                lower_mv_precision_to_integer(pred)
            };
            let mv_prec = if is_qpel { 5 } else { 3 };
            let diff = read_mv_residual(&mut self.msac, &mut self.cdf.dmv, mv_prec, false)?;
            MvXY {
                x: base.x.saturating_add(diff.x),
                y: base.y.saturating_add(diff.y),
            }
        };
        if self.seq.bawp && allow_sct {
            return Err(Av2Error::Unsupported(
                "IntraBC morph prediction (BAWP)".to_string(),
            ));
        }
        self.check_intrabc_mv(bs, bx4, by4, mv)?;

        let dx_px = (mv.x as i32) / 8;
        let dy_px = (mv.y as i32) / 8;
        if mv.x % 8 != 0 || mv.y % 8 != 0 {
            return Err(Av2Error::Unsupported(
                "sub-pixel IntraBC prediction".to_string(),
            ));
        }
        let x_limit = self.frame_mi_w * 4;
        let y_limit = self.frame_mi_h * 4;
        apply_intrabc_block_bounded(
            &mut self.rec.planes[0],
            px_x,
            px_y,
            w,
            h,
            dy_px,
            dx_px,
            x_limit,
            y_limit,
        );
        if has_chroma {
            for pl in 1..=2 {
                apply_intrabc_block_bounded(
                    &mut self.rec.planes[pl],
                    px_x >> ss_x,
                    px_y >> ss_y,
                    w >> ss_x,
                    h >> ss_y,
                    dy_px >> ss_y,
                    dx_px >> ss_x,
                    x_limit >> ss_x,
                    y_limit >> ss_y,
                );
            }
        }

        // Residual syntax: `read_tx_part` and `read_coef_blocks` are present
        // whenever `skip_txfm == 0`; IntraBC is coded as inter (`!b->intra ||
        // b->intrabc`) for transform-set selection (§5.20.8.2).
        let mut l_res_ctx = 0x40u8;
        let mut u_res_ctx = 0x40u8;
        let mut v_res_ctx = 0x40u8;
        if !skip_txfm {
            let tx = MAX_TXFM_SIZE_FOR_BS[bs as usize] as usize;
            let t_dim = &TXFM_DIMENSIONS[tx];
            self.decode_tx_part_flag(bs, 0, 1)?;
            let y_params = CoefDecodeParams {
                tx,
                bs: bs as usize,
                plane: 0,
                intra: false,
                fsc: false,
                lossless: seg_lossless,
                uv_mode: 0,
                y_mode: 0,
                y_angle: 0,
                mrl_index: 0,
                y_txtp: 0,
                reduced_txtp_set: self.fh.reduced_txtp_set as i32,
                u_has_cf: 0,
                dq_tbl: self.dq[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize][0],
                bitdepth: self.bd.bitdepth(),
                ss_hor: ss_x != 0,
                ss_ver: ss_y != 0,
                chroma_dctonly: self.seq.chroma_dctonly,
            };
            let y_res = decode_coefs(
                &mut self.msac,
                &mut self.cdf.coef,
                &mut self.cdf.m,
                &self.a_cols[a_idx].lcoef[bx4_in_sb..],
                &self.l_ctx.lcoef[by4_in_sb..],
                &y_params,
                &mut self.dq_y,
            )?;
            l_res_ctx = y_res.res_ctx;
            if y_res.eob >= 0 {
                let pl = &mut self.rec.planes[0];
                let off = pl.offset(px_x, px_y);
                inv_txfm_add(
                    self.bd,
                    &mut pl.data,
                    off,
                    pl.stride,
                    &mut self.dq_y,
                    y_res.txtp as u32,
                    y_res.eob,
                    tx,
                );
            }
            let _ = t_dim;
            if has_chroma {
                let layout_col = (self.seq.layout as usize).saturating_sub(1);
                let uv_bs_idx = SS_BS[bs as usize][layout_col] as usize;
                let uvtx = MAX_TXFM_SIZE_FOR_BS[uv_bs_idx] as usize;
                let uv_tdim = &TXFM_DIMENSIONS[uvtx];
                let cw = uv_tdim.w as usize * 4;
                let ch = uv_tdim.h as usize * 4;
                let cpx_x = px_x >> ss_x;
                let cpx_y = px_y >> ss_y;
                let mut u_params = CoefDecodeParams {
                    tx: uvtx,
                    bs: uv_bs_idx,
                    plane: 1,
                    intra: false,
                    fsc: false,
                    lossless: seg_lossless,
                    uv_mode: 0,
                    y_mode: 0,
                    y_angle: 0,
                    mrl_index: 0,
                    y_txtp: y_res.txtp as u32,
                    reduced_txtp_set: self.fh.reduced_txtp_set as i32,
                    u_has_cf: 0,
                    dq_tbl: self.dq[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize][1],
                    bitdepth: self.bd.bitdepth(),
                    ss_hor: ss_x != 0,
                    ss_ver: ss_y != 0,
                    chroma_dctonly: self.seq.chroma_dctonly,
                };
                let u_res = decode_coefs(
                    &mut self.msac,
                    &mut self.cdf.coef,
                    &mut self.cdf.m,
                    &self.a_cols[a_idx].ccoef[0][(bx4_in_sb >> ss_x)..],
                    &self.l_ctx.ccoef[0][(by4_in_sb >> ss_y)..],
                    &u_params,
                    &mut self.dq_u,
                )?;
                u_res_ctx = u_res.res_ctx;
                u_params.u_has_cf = (u_res.eob >= 0) as i32;
                u_params.plane = 2;
                u_params.dq_tbl = self.dq[seg_id.min((MAX_SEGMENTS - 1) as u8) as usize][2];
                let v_res = decode_coefs(
                    &mut self.msac,
                    &mut self.cdf.coef,
                    &mut self.cdf.m,
                    &self.a_cols[a_idx].ccoef[1][(bx4_in_sb >> ss_x)..],
                    &self.l_ctx.ccoef[1][(by4_in_sb >> ss_y)..],
                    &u_params,
                    &mut self.dq_v,
                )?;
                v_res_ctx = v_res.res_ctx;
                if u_res.eob >= 0 {
                    let pl = &mut self.rec.planes[1];
                    let off = pl.offset(cpx_x, cpx_y);
                    inv_txfm_add(
                        self.bd,
                        &mut pl.data,
                        off,
                        pl.stride,
                        &mut self.dq_u,
                        u_res.txtp as u32,
                        u_res.eob,
                        uvtx,
                    );
                }
                if v_res.eob >= 0 {
                    let pl = &mut self.rec.planes[2];
                    let off = pl.offset(cpx_x, cpx_y);
                    inv_txfm_add(
                        self.bd,
                        &mut pl.data,
                        off,
                        pl.stride,
                        &mut self.dq_v,
                        v_res.txtp as u32,
                        v_res.eob,
                        uvtx,
                    );
                }
                let _ = (cw, ch);
            }
        }

        // Contexts: `intra = 1`, `intrabc = 1`, `mode = DC_PRED`, zero palette.
        for i in 0..bw4
            .min(32 - bx4_in_sb)
            .min(self.frame_mi_w.saturating_sub(bx4))
        {
            let a = &mut self.a_cols[a_idx];
            a.intra[bx4_in_sb + i] = 1;
            a.mode[bx4_in_sb + i] = 0;
            a.midx[bx4_in_sb + i] = 0xFF;
            a.skip[bx4_in_sb + i] = skip_txfm as u8;
            a.lcoef[bx4_in_sb + i] = l_res_ctx;
            a.ref_frame[0][bx4_in_sb + i] = -1;
            a.ref_frame[1][bx4_in_sb + i] = -1;
            a.filter[0][bx4_in_sb + i] = 3;
            a.filter[1][bx4_in_sb + i] = 3;
        }
        for i in 0..bh4
            .min(32 - by4_in_sb)
            .min(self.frame_mi_h.saturating_sub(by4))
        {
            let l = &mut self.l_ctx;
            l.intra[by4_in_sb + i] = 1;
            l.mode[by4_in_sb + i] = 0;
            l.midx[by4_in_sb + i] = 0xFF;
            l.skip[by4_in_sb + i] = skip_txfm as u8;
            l.lcoef[by4_in_sb + i] = l_res_ctx;
            l.ref_frame[0][by4_in_sb + i] = -1;
            l.ref_frame[1][by4_in_sb + i] = -1;
            l.filter[0][by4_in_sb + i] = 3;
            l.filter[1][by4_in_sb + i] = 3;
        }
        if has_chroma {
            // The chroma entropy contexts live at chroma 4-sample granularity
            // over the chroma plane block size, not over the luma footprint
            // (`av2_reset_entropy_context`, §5.20.4.1 / `coeffs()` `x4/y4`,
            // §5.20.7.27). Writing them at luma indices desynchronised the
            // chroma `txb_skip` context of the block below the IntraBC block.
            let cbx4 = bx4_in_sb >> ss_x;
            let cby4 = by4_in_sb >> ss_y;
            for i in 0..(bw4 >> ss_x)
                .min(32 - cbx4)
                .min((self.frame_mi_w >> ss_x).saturating_sub(bx4 >> ss_x))
            {
                let a = &mut self.a_cols[a_idx];
                a.ccoef[0][cbx4 + i] = u_res_ctx;
                a.ccoef[1][cbx4 + i] = v_res_ctx;
                a.uvmode[cbx4 + i] = 0;
            }
            for i in 0..(bh4 >> ss_y)
                .min(32 - cby4)
                .min((self.frame_mi_h >> ss_y).saturating_sub(by4 >> ss_y))
            {
                let l = &mut self.l_ctx;
                l.ccoef[0][cby4 + i] = u_res_ctx;
                l.ccoef[1][cby4 + i] = v_res_ctx;
                l.uvmode[cby4 + i] = 0;
            }
        }
        self.store_palette_block(bx4, by4, bw4, bh4, None);
        for y in 0..bh4 {
            let row = by4 + y;
            if row >= self.frame_mi_h {
                break;
            }
            for x in 0..bw4 {
                let col = bx4 + x;
                if col >= self.frame_mi_w {
                    break;
                }
                self.intrabc_flags[row * self.frame_mi_w + col] = true;
            }
        }
        Ok(())
    }

    fn decode_tx_part_flag(&mut self, bs: BlockSize, fsc: usize, inter: usize) -> Result<()> {
        // `read_tx_partition()` is only present when `TxMode == TX_MODE_SELECT`
        // (`read_tx_size()`, AV2 Spec v1.0.0 §5.20.6.1).
        if self.fh.txfm_mode != TxfmMode::Switchable {
            return Ok(());
        }
        if self.seq.reduced_tx_part_set {
            return Ok(());
        }
        let base_tx_idx = MAX_TXFM_SIZE_FOR_BS[bs as usize] as usize;
        if base_tx_idx == 0 || bs == BlockSize::Bs4x4 {
            return Ok(());
        }
        let b_dim = BLOCK_DIMENSIONS[bs as usize];
        if (b_dim[0].max(b_dim[1]) as usize) > 16 {
            return Ok(());
        }
        let grp = TX_PART_GROUP[bs as usize] as usize;
        let split = self
            .msac
            .decode_bool_adapt(self.cdf.m.tx_do_partition(fsc, inter, grp))?;
        if split != 0 {
            return Err(Av2Error::Unsupported(
                "sub-block transform partitioning".to_string(),
            ));
        }
        Ok(())
    }

    fn edge_buffer(
        &self,
        pl: usize,
        px_x: usize,
        px_y: usize,
        w: usize,
        h: usize,
        have_left: bool,
        have_top: bool,
        mrl_idx: u8,
    ) -> [BD::Pixel; 512] {
        let plane = &self.rec.planes[pl];
        let sb_px = (64usize << self.fh.sb128) >> plane.ss_x.max(plane.ss_y);
        let (have_tr, have_bl) = crate::av2::ipred::intra_edge_availability(
            px_x,
            px_y,
            w,
            h,
            (self.tile_col_end4 * 4) >> plane.ss_x,
            (self.tile_row_end4 * 4) >> plane.ss_y,
            sb_px,
            have_left,
            have_top,
        );
        let (rec_w, rec_h) = crate::av2::ipred::intra_reference_extent(
            self.frame_mi_w,
            self.frame_mi_h,
            plane.ss_x,
            plane.ss_y,
        );
        let mut edge = [BD::Pixel::default(); 512];
        prepare_intra_edges_mrl(
            self.bd,
            &mut edge,
            256,
            &plane.data,
            plane.offset(px_x, px_y),
            plane.stride,
            have_left,
            have_top,
            have_tr,
            have_bl,
            w,
            h,
            mrl_idx,
            rec_w,
            rec_h,
        );
        edge
    }

    /// Returns the intra prediction for one plane of an inter-intra block
    /// (`read_interintra_mode`, §5.20.7.15 / §5.20.7.22). The prediction is
    /// written into the current frame, copied out, and later blended with the
    /// inter prediction by [`blend_interintra`] (§7.13.3.30).
    #[allow(clippy::too_many_arguments)]
    fn save_interintra_pred(
        &mut self,
        plane: usize,
        px_x: usize,
        px_y: usize,
        w: usize,
        h: usize,
        have_left: bool,
        have_top: bool,
        interintra_mode: u8,
    ) -> Vec<BD::Pixel> {
        let edge = self.edge_buffer(plane, px_x, px_y, w, h, have_left, have_top, 0);
        let mode = interintra_intra_mode(interintra_mode);
        {
            let pl = &mut self.rec.planes[plane];
            let off = pl.offset(px_x, px_y);
            predict_intra_block(
                self.bd,
                &mut pl.data,
                off,
                pl.stride,
                &edge,
                256,
                mode,
                0,
                w,
                h,
                have_left,
                have_top,
                plane,
                0,
            );
        }
        let pl = &self.rec.planes[plane];
        let mut out = vec![BD::Pixel::default(); w * h];
        for r in 0..h {
            let off = pl.offset(px_x, px_y + r);
            out[r * w..(r + 1) * w].copy_from_slice(&pl.data[off..off + w]);
        }
        out
    }
}

fn filter_intra_mode(mode: u8) -> FilterIntraMode {
    match mode {
        1 => FilterIntraMode::Vertical,
        2 => FilterIntraMode::Horizontal,
        3 => FilterIntraMode::Diagonal,
        4 => FilterIntraMode::Paeth,
        _ => FilterIntraMode::Dc,
    }
}
