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

//! Normative AV2 Specification v1.0.0 §5.2, §5.4, §5.18, §5.19 Open Bitstream Unit (`OBU`)
//! writer and `SequenceHeader`, `FrameHeader`, `TemporalDelimiter`, and `TileGroup` packers.

use crate::av2::ccso::CCSO_QUANT_SZ;
use crate::av2::headers::{
    AdaptiveBoolean, ContentLightLevel, FilterMode, FrameHeader, FrameType, MAX_CDEF_STRENGTHS,
    MAX_SEGMENTS, MAX_TILE_COLS, MAX_TILE_ROWS, MasteringDisplay, ObuType, PRIMARY_REF_NONE,
    PixelLayout, RestorationType, SegmentationDataSet, SequenceHeader, TileInfo, TxfmMode,
};
use crate::av2::intops::{iclip, imax, imin, tile_log2, ulog2};
use crate::av2::levels::ObuMetaType;
use crate::av2::putbits::PutBits;

/// Packages an already-serialized OBU payload into a complete AV2 OBU with its
/// leading `leb128()` size field (`sz = hdr_bytes + payload.len()`) followed by the
/// 1-byte (or 2-byte extended) `obu_header()` and payload (`AV2 Spec v1.0.0 §5.2 & Annex B`).
pub fn write_obu(
    obu_type: ObuType,
    tlayer_id: u8,
    mlayer_id: u8,
    xlayer_id: u8,
    payload: &[u8],
) -> Vec<u8> {
    let has_extension = mlayer_id != 0 || xlayer_id != 0;
    let hdr_bytes = if has_extension { 2usize } else { 1usize };
    let obu_size = (hdr_bytes + payload.len()) as u32;

    let mut pb = PutBits::new();
    pb.put_uleb128(obu_size);
    pb.put_bit(has_extension as u32);
    pb.put_bits(5, obu_type as u32);
    pb.put_bits(2, (tlayer_id & 3) as u32);
    if has_extension {
        pb.put_bits(3, (mlayer_id & 7) as u32);
        pb.put_bits(5, (xlayer_id & 31) as u32);
    }
    pb.put_bytes(payload);
    pb.into_bytes()
}

/// Alias for [`write_obu`] with base layer IDs `(0, 0, 0)`.
#[inline]
pub fn pack_obu(obu_type: ObuType, payload: &[u8]) -> Vec<u8> {
    write_obu(obu_type, 0, 0, 0, payload)
}

/// Writes a Temporal Delimiter OBU (`ObuType::Td`, empty payload, 2 bytes `[0x01, 0x08]`).
pub fn write_temporal_delimiter_obu() -> Vec<u8> {
    write_obu(ObuType::Td, 0, 0, 0, &[])
}

/// Writes a Padding OBU (`ObuType::Padding`) containing `n_padding_bytes` zero bytes.
pub fn write_padding_obu(n_padding_bytes: usize) -> Vec<u8> {
    let mut pb = PutBits::new();
    for _ in 0..n_padding_bytes {
        pb.put_bits(8, 0);
    }
    pb.put_trailing_bits();
    write_obu(ObuType::Padding, 0, 0, 0, &pb.into_bytes())
}

/// Writes an HDR Content Light Level (`CLL`) Metadata OBU.
pub fn write_metadata_cll_obu(cll: &ContentLightLevel) -> Vec<u8> {
    let mut pb = PutBits::new();
    pb.put_uleb128(ObuMetaType::HdrCll as u32);
    pb.put_bits(16, cll.max_content_light_level as u32);
    pb.put_bits(16, cll.max_frame_average_light_level as u32);
    pb.put_trailing_bits();
    write_obu(ObuType::Metadata, 0, 0, 0, &pb.into_bytes())
}

/// Writes an HDR Mastering Display Color Volume (`MDCV`) Metadata OBU.
pub fn write_metadata_mdcv_obu(md: &MasteringDisplay) -> Vec<u8> {
    let mut pb = PutBits::new();
    pb.put_uleb128(ObuMetaType::HdrMdcv as u32);
    for i in 0..3 {
        pb.put_bits(16, md.primaries[i][0] as u32);
        pb.put_bits(16, md.primaries[i][1] as u32);
    }
    pb.put_bits(16, md.white_point[0] as u32);
    pb.put_bits(16, md.white_point[1] as u32);
    pb.put_bits(32, md.max_luminance);
    pb.put_bits(32, md.min_luminance);
    pb.put_trailing_bits();
    write_obu(ObuType::Metadata, 0, 0, 0, &pb.into_bytes())
}

/// Writes `SegmentationDataSet` (`seg_info(numSegments)` in AV2 Spec v1.0.0 §5.4.4.1) into `pb`.
pub fn write_seg_info(seg: &SegmentationDataSet, pb: &mut PutBits, n_seg: usize) {
    for n in 0..n_seg {
        let m = 1u16 << n;
        let has_q = (seg.delta_q_mask & m) != 0 || seg.delta_q[n] != 0;
        pb.put_bit(has_q as u32);
        if has_q {
            let dq = (seg.delta_q[n] as i32).clamp(-351, 351);
            pb.put_sbits(10, dq);
        }
        pb.put_bit(((seg.skip_mask & m) != 0) as u32);
        pb.put_bit(((seg.globalmv_mask & m) != 0) as u32);
    }
}

/// Initializes a uniform `TileInfo` grid for frame dimensions `(width, height)` per AV2 Spec §5.18.4.
pub fn init_uniform_tile_info(
    t: &mut TileInfo,
    _sbmul: i32,
    sb128: u8,
    seq_sb128: u8,
    width: i32,
    height: i32,
    level: u8,
    tier: u8,
    log2_cols_req: u8,
    log2_rows_req: u8,
) {
    let sbsz_log2 = 6 + sb128 as i32;
    let sbsz_min1 = (64 << sb128) - 1;
    let sbw = (width + sbsz_min1) >> sbsz_log2;
    let sbh = (height + sbsz_min1) >> sbsz_log2;
    let w_adj = (level >= 18) as i32 + ((level >= 14 && tier != 0) as i32);
    let max_tile_width_sb = 4096 >> (sbsz_log2 - w_adj);
    let sz_adj = (level >= 14) as i32 + (level >= 18) as i32 + ((level >= 14 && tier != 0) as i32);
    let max_tile_area_sb = (4096 * 2304) >> (2 * sbsz_log2 - sz_adj);

    t.min_log2_cols = tile_log2(max_tile_width_sb, sbw) as u8;
    t.max_log2_cols = tile_log2(1, imin(sbw, MAX_TILE_COLS as i32)) as u8;
    t.max_log2_rows = tile_log2(1, imin(sbh, MAX_TILE_ROWS as i32)) as u8;
    let min_log2_tiles = imax(
        tile_log2(max_tile_area_sb, sbw * sbh),
        t.min_log2_cols as i32,
    );

    t.uniform = true;
    t.log2_cols = iclip(
        log2_cols_req as i32,
        t.min_log2_cols as i32,
        t.max_log2_cols as i32,
    ) as u8;
    t.min_log2_rows = imax(min_log2_tiles - t.log2_cols as i32, 0) as u8;
    t.log2_rows = iclip(
        log2_rows_req as i32,
        t.min_log2_rows as i32,
        t.max_log2_rows as i32,
    ) as u8;

    let seq_sbsz_log2 = 6 + seq_sb128 as i32;
    let sbmul = 1i32 << (seq_sbsz_log2 - sbsz_log2).max(0);
    let fsbw = imax(1, (width + 7) >> seq_sbsz_log2);
    let fsbh = imax(1, (height + 7) >> seq_sbsz_log2);

    let tile_w = imax(1, fsbw >> t.log2_cols);
    let mut extra_w = imax(0, fsbw - (tile_w << t.log2_cols));
    t.cols = 0;
    let mut sbx = 0i32;
    while sbx < fsbw && (t.cols as usize) < MAX_TILE_COLS {
        t.col_start_sb[t.cols as usize] = (sbx * sbmul) as u16;
        let step = tile_w + (extra_w > 0) as i32;
        extra_w -= (extra_w > 0) as i32;
        sbx += step;
        t.cols += 1;
    }
    t.col_start_sb[t.cols as usize] = sbw as u16;
    t.log2_cols = tile_log2(1, t.cols as i32) as u8;

    let tile_h = imax(1, fsbh >> t.log2_rows);
    let mut extra_h = imax(0, fsbh - (tile_h << t.log2_rows));
    t.rows = 0;
    let mut sby = 0i32;
    while sby < fsbh && (t.rows as usize) < MAX_TILE_ROWS {
        t.row_start_sb[t.rows as usize] = (sby * sbmul) as u16;
        let step = tile_h + (extra_h > 0) as i32;
        extra_h -= (extra_h > 0) as i32;
        sby += step;
        t.rows += 1;
    }
    t.row_start_sb[t.rows as usize] = sbh as u16;
    t.log2_rows = tile_log2(1, t.rows as i32) as u8;
}

/// Writes `tile_params` syntax bits into `pb` per AV2 Specification v1.0.0 §5.4.10.1 & §5.18.4.
#[allow(clippy::too_many_arguments)]
pub fn write_tile_info(
    t: &TileInfo,
    pb: &mut PutBits,
    sb128: u8,
    width: i32,
    height: i32,
    level: u8,
    tier: u8,
) {
    let sbsz_log2 = 6 + sb128 as i32;
    let sbsz_min1 = (64 << sb128) - 1;
    let sbw = (width + sbsz_min1) >> sbsz_log2;
    let sbh = (height + sbsz_min1) >> sbsz_log2;
    let w_adj = (level >= 18) as i32 + ((level >= 14 && tier != 0) as i32);
    let max_tile_width_sb = 4096 >> (sbsz_log2 - w_adj);
    let sz_adj = (level >= 14) as i32 + (level >= 18) as i32 + ((level >= 14 && tier != 0) as i32);
    let max_tile_area_sb = (4096 * 2304) >> (2 * sbsz_log2 - sz_adj);

    let min_log2_cols = tile_log2(max_tile_width_sb, sbw);
    let max_log2_cols = tile_log2(1, imin(sbw, MAX_TILE_COLS as i32));
    let max_log2_rows = tile_log2(1, imin(sbh, MAX_TILE_ROWS as i32));
    let min_log2_tiles = imax(tile_log2(max_tile_area_sb, sbw * sbh), min_log2_cols);

    pb.put_bit(t.uniform as u32);
    if t.uniform {
        for _ in min_log2_cols..(t.log2_cols as i32) {
            pb.put_bit(1);
        }
        if (t.log2_cols as i32) < max_log2_cols {
            pb.put_bit(0);
        }
        let min_log2_rows = imax(min_log2_tiles - (t.log2_cols as i32), 0);
        for _ in min_log2_rows..(t.log2_rows as i32) {
            pb.put_bit(1);
        }
        if (t.log2_rows as i32) < max_log2_rows {
            pb.put_bit(0);
        }
    } else {
        let mut widest_tile_sb = 1i32;
        let mut start_sb = 0i32;
        let mut i = 0usize;
        while start_sb < sbw && i < t.cols as usize {
            let next_sb = t.col_start_sb[i + 1] as i32;
            let size_sb = (next_sb - start_sb).max(1);
            let n = imin(sbw - start_sb, max_tile_width_sb) as u32;
            pb.put_uniform(n, (size_sb - 1) as u32);
            widest_tile_sb = imax(size_sb, widest_tile_sb);
            start_sb += size_sb;
            i += 1;
        }
        let max_area = if min_log2_tiles > 0 {
            (sbh * sbw) >> (min_log2_tiles + 1)
        } else {
            sbh * sbw
        };
        let max_tile_height_sb = imax(max_area / widest_tile_sb, 1);
        start_sb = 0;
        i = 0;
        while start_sb < sbh && i < t.rows as usize {
            let next_sb = t.row_start_sb[i + 1] as i32;
            let size_sb = (next_sb - start_sb).max(1);
            let n = imin(sbh - start_sb, max_tile_height_sb) as u32;
            pb.put_uniform(n, (size_sb - 1) as u32);
            start_sb += size_sb;
            i += 1;
        }
    }
}

/// Writes the normative AV2 `sequence_header_obu()` payload into `pb` (`AV2 Spec v1.0.0 §5.4`).
pub fn write_seq_hdr(hdr: &SequenceHeader, pb: &mut PutBits) {
    pb.put_vlc(hdr.id as u32);
    pb.put_bits(5, (hdr.profile & 0x1F) as u32);
    pb.put_bit(hdr.reduced_still_picture_header as u32);
    pb.put_bits(5, (hdr.level & 0x1F) as u32);
    if hdr.level >= 4 && !hdr.reduced_still_picture_header {
        pb.put_bit((hdr.tier & 1) as u32);
    }

    let layout_idx = match hdr.layout {
        PixelLayout::I420 => 0u32,
        PixelLayout::I400 => 1u32,
        PixelLayout::I444 => 2u32,
        PixelLayout::I422 => 3u32,
    };
    pb.put_vlc(layout_idx);

    // AV2 Spec Table 6.4: bit_depth_idc = 0 (10-bit, hbd=1), 1 (8-bit, hbd=0), 2 (12-bit, hbd=2)
    let coded_hbd = if hdr.hbd < 2 { hdr.hbd ^ 1 } else { hdr.hbd };
    pb.put_vlc(coded_hbd as u32);

    if !hdr.reduced_still_picture_header {
        pb.put_bits(3, (hdr.lcr_id & 7) as u32);
        pb.put_bit(hdr.still_picture as u32);
        pb.put_bits(2, (hdr.max_tlayer_id & 3) as u32);
        pb.put_bits(3, (hdr.max_mlayer_id & 7) as u32);
        pb.put_bit(hdr.monotonic as u32);
    }

    let w_bits = hdr.width_n_bits.clamp(1, 16);
    let h_bits = hdr.height_n_bits.clamp(1, 16);
    pb.put_bits(4, (w_bits - 1) as u32);
    pb.put_bits(4, (h_bits - 1) as u32);
    pb.put_bits(w_bits as i32, (hdr.max_width.max(1) - 1) as u32);
    pb.put_bits(h_bits as i32, (hdr.max_height.max(1) - 1) as u32);

    pb.put_bit(hdr.crop.enabled as u32);
    if hdr.crop.enabled {
        pb.put_vlc(hdr.crop.left);
        pb.put_vlc(hdr.crop.right);
        pb.put_vlc(hdr.crop.top);
        pb.put_vlc(hdr.crop.bottom);
    }

    if !hdr.reduced_still_picture_header {
        pb.put_bit(0); // seq_initial_display_delay_present_flag = 0
        pb.put_bit(0); // decoder_model_info_present_flag = 0
    }

    if hdr.max_tlayer_id > 0 {
        pb.put_bit(hdr.tlayer_dependency_present as u32);
        if hdr.tlayer_dependency_present {
            for n in 1..hdr.max_tlayer_id as usize {
                pb.put_bits(n as i32, hdr.tlayer_dependencies[n] as u32);
            }
        }
    }
    if hdr.max_mlayer_id > 0 {
        pb.put_bit(hdr.mlayer_dependency_present as u32);
        if hdr.mlayer_dependency_present {
            for n in 1..hdr.max_mlayer_id as usize {
                pb.put_bits(n as i32, hdr.mlayer_dependencies[n] as u32);
            }
        }
    }

    // §5.4.3 sequence_partition_config()
    if hdr.sb128 == 2 {
        pb.put_bit(1); // use_256x256_superblock = 1
    } else {
        pb.put_bit(0); // use_256x256_superblock = 0
        pb.put_bit((hdr.sb128 != 0) as u32); // use_128x128_superblock
    }
    if hdr.layout != PixelLayout::I400 {
        pb.put_bit(hdr.sdp as u32);
        if hdr.sdp && !hdr.reduced_still_picture_header {
            pb.put_bit(hdr.ext_sdp as u32);
        }
    }
    pb.put_bit(hdr.ext_partitions as u32);
    if hdr.ext_partitions {
        pb.put_bit(hdr.uneven_4way_partitions as u32);
    }
    if hdr.max_pb_aspect_ratio_log2 < 3 {
        pb.put_bit(1); // reduce_pb_aspect_ratio = 1
        pb.put_bit((hdr.max_pb_aspect_ratio_log2.saturating_sub(1) & 1) as u32);
    } else {
        pb.put_bit(0); // reduce_pb_aspect_ratio = 0 -> MaxPbAspectRatio = 8 (log2 = 3)
    }

    // §5.4.4 sequence_segment_config()
    pb.put_bit(hdr.segmentation.ext as u32);
    pb.put_bit(hdr.segmentation.info_present as u32);
    if hdr.segmentation.info_present {
        pb.put_bit(hdr.segmentation.adaptive as u32);
        write_seg_info(
            &hdr.segmentation.d,
            pb,
            8usize << (hdr.segmentation.ext as usize),
        );
    }

    // §5.4.5 sequence_intra_config()
    pb.put_bit(hdr.intra_dip as u32);
    pb.put_bit(hdr.intra_edge_filter as u32);
    pb.put_bit(hdr.mrls as u32);
    pb.put_bit(hdr.cfl as u32);
    if hdr.layout != PixelLayout::I400 {
        pb.put_bits(2, (hdr.cfl_ds_filter_index & 3) as u32);
    }
    pb.put_bit(hdr.mhccp as u32);
    pb.put_bit(hdr.ibp as u32);

    // §5.4.6 sequence_inter_config()
    if !hdr.reduced_still_picture_header {
        for shift in [1, 2, 3, 4] {
            pb.put_bit(((hdr.motion_modes >> shift) & 1) as u32);
        }
        if (hdr.motion_modes & !1) != 0 {
            pb.put_bit(hdr.frame_motion_modes_present as u32);
        }
        if (hdr.motion_modes & (1 << 3)) != 0 {
            pb.put_bit(hdr.six_param_warp_delta as u32);
        }
        pb.put_bit(hdr.masked_compound as u32);
        pb.put_bit(hdr.ref_frame_mvs as u32);
        if hdr.ref_frame_mvs {
            pb.put_bit((hdr.reduced_ref_frame_mvs_mode & 1) as u32);
        }
        pb.put_bits(4, (hdr.order_hint_n_bits.clamp(1, 16) - 1) as u32);
    }

    pb.put_bit(hdr.refmv_bank as u32);
    if hdr.drl_reorder == 0 {
        pb.put_bit(1); // disable_drl_reorder = 1
    } else {
        pb.put_bit(0); // disable_drl_reorder = 0
        pb.put_bit((hdr.drl_reorder == 1) as u32); // constrain_drl_reorder
    }

    if !hdr.reduced_still_picture_header {
        pb.put_bit(hdr.explicit_ref_frame_map as u32);
        if hdr.ref_frames != 8 {
            pb.put_bit(1);
            pb.put_bits(4, (hdr.ref_frames.clamp(1, 16) - 1) as u32);
        } else {
            pb.put_bit(0);
        }
        pb.put_bits(3, (hdr.number_of_bits_for_lt_frame_id & 7) as u32);
        pb.put_uniform(5, (hdr.def_max_drl_bits.clamp(1, 5) - 1) as u32);
        pb.put_bit(hdr.allow_frame_max_drl_bits as u32);
    }

    pb.put_uniform(3, (hdr.def_max_bvp_drl_bits.clamp(1, 3) - 1) as u32);
    pb.put_bit(hdr.allow_max_bvp_drl_bits as u32);

    if !hdr.reduced_still_picture_header {
        pb.put_bits(2, (hdr.num_same_ref_comp & 3) as u32);
        if hdr.tip_mode == 0 {
            pb.put_bit(0);
        } else {
            pb.put_bit(1);
            pb.put_bit(((hdr.tip_mode - 1) & 1) as u32);
            pb.put_bit(hdr.tip_hole_fill as u32);
        }
        pb.put_bit(hdr.mv_traj as u32);
    }

    pb.put_bit(hdr.bawp as u32);

    if !hdr.reduced_still_picture_header {
        pb.put_bit(hdr.cwp as u32);
        pb.put_bit(hdr.imp_msk_bld as u32);
        pb.put_bit(hdr.db_sub_pu as u32);
        if hdr.tip_mode == 1 && hdr.db_sub_pu {
            pb.put_bit(hdr.tip_explicit_qp as u32);
        }
        pb.put_bits(2, (hdr.opfl_refine & 3) as u32);
        pb.put_bit(hdr.refine_mv as u32);
        if hdr.tip_mode != 0 && (hdr.opfl_refine != 0 || hdr.refine_mv) {
            pb.put_bit(hdr.tip_refine_mv as u32);
        }
        pb.put_bit(hdr.bru as u32);
        pb.put_bit(hdr.adaptive_mvd as u32);
        pb.put_bit(hdr.mvd_sign_derive as u32);
        pb.put_bit(hdr.flex_mvres as u32);
        pb.put_bit(hdr.global_motion as u32);
        pb.put_bit(hdr.short_refresh_frame_flags as u32);
    }

    // §5.4.7 sequence_scc_config()
    if !hdr.reduced_still_picture_header {
        match hdr.screen_content_tools {
            AdaptiveBoolean::Adaptive => pb.put_bit(1),
            AdaptiveBoolean::On => {
                pb.put_bit(0);
                pb.put_bit(1);
            }
            AdaptiveBoolean::Off => {
                pb.put_bit(0);
                pb.put_bit(0);
            }
        }
        if hdr.screen_content_tools != AdaptiveBoolean::Off {
            match hdr.force_integer_mv {
                AdaptiveBoolean::Adaptive => pb.put_bit(1),
                AdaptiveBoolean::On => {
                    pb.put_bit(0);
                    pb.put_bit(1);
                }
                AdaptiveBoolean::Off => {
                    pb.put_bit(0);
                    pb.put_bit(0);
                }
            }
        }
    }

    // §5.4.8 sequence_transform_quant_entropy_config()
    pb.put_bit(hdr.fsc as u32);
    if !hdr.fsc {
        pb.put_bit(hdr.idtx_intra as u32);
    }
    pb.put_bit(hdr.ist[0] as u32);
    pb.put_bit(hdr.ist[1] as u32);
    if hdr.layout != PixelLayout::I400 {
        pb.put_bit(hdr.chroma_dctonly as u32);
    }
    if !hdr.reduced_still_picture_header {
        pb.put_bit(hdr.inter_ddt as u32);
    }
    pb.put_bit(hdr.reduced_tx_part_set as u32);
    if hdr.layout != PixelLayout::I400 {
        pb.put_bit(hdr.cctx as u32);
    }

    match hdr.tcq {
        AdaptiveBoolean::Off => pb.put_bit(0),
        AdaptiveBoolean::On => {
            pb.put_bit(1);
            if !hdr.reduced_still_picture_header {
                pb.put_bit(0);
            }
        }
        AdaptiveBoolean::Adaptive => {
            pb.put_bit(1);
            if !hdr.reduced_still_picture_header {
                pb.put_bit(1);
            }
        }
    }
    if hdr.tcq != AdaptiveBoolean::On {
        pb.put_bit(hdr.parity_hiding as u32);
    }

    if !hdr.reduced_still_picture_header {
        pb.put_bit(hdr.avg_cdf as u32);
        if hdr.avg_cdf {
            pb.put_bit((hdr.avg_cdf_type != 0) as u32);
        }
    }

    if hdr.layout != PixelLayout::I400 {
        pb.put_bit(hdr.separate_uv_delta_q as u32);
    }
    pb.put_bit(hdr.equal_ac_dc_q as u32);
    if !hdr.equal_ac_dc_q {
        pb.put_bits(5, ((hdr.base_ydc_dq as i32) + 23).clamp(0, 31) as u32);
        pb.put_bit(hdr.ydc_dq_enabled as u32);
    }
    if hdr.layout != PixelLayout::I400 {
        if !hdr.equal_ac_dc_q {
            pb.put_bits(
                5,
                ((hdr.base_uvdc_dq as i8 as i32) + 23).clamp(0, 31) as u32,
            );
            pb.put_bit(hdr.uvdc_dq_enabled as u32);
        }
        pb.put_bits(
            5,
            ((hdr.base_uvac_dq as i8 as i32) + 23).clamp(0, 31) as u32,
        );
        pb.put_bit(hdr.uvac_dq_enabled as u32);
    }

    // §5.4.9 sequence_filter_config()
    pb.put_bit(hdr.disable_loopfilters_across_tiles as u32);
    pb.put_bit(hdr.cdef as u32);
    pb.put_bit(hdr.gdf as u32);
    if hdr.gdf && hdr.sb128 == 0 {
        pb.put_bit(hdr.gdf_unit_matches_sbsz as u32);
    }
    pb.put_bit(hdr.restoration as u32);
    if hdr.restoration {
        pb.put_bit((hdr.rst_disable_mask[0] & 1) as u32);
        pb.put_bit(((hdr.rst_disable_mask[0] >> 1) & 1) as u32);
        pb.put_bit(1); // lr_tools_uv_present = 1
        pb.put_bit(((hdr.rst_disable_mask[1] >> 1) & 1) as u32);
    }
    pb.put_bit(hdr.ccso as u32);
    if hdr.ccso {
        pb.put_bit(hdr.ccso_unit_matches_sbsz as u32);
    }
    if !hdr.reduced_still_picture_header {
        match hdr.cdef_on_skiptx {
            AdaptiveBoolean::On => pb.put_bit(1),
            AdaptiveBoolean::Off => {
                pb.put_bit(0);
                pb.put_bit(1);
            }
            AdaptiveBoolean::Adaptive => {
                pb.put_bit(0);
                pb.put_bit(0);
            }
        }
    }
    pb.put_bits(2, (hdr.df_par_bits.saturating_sub(2) & 3) as u32);

    // §5.4.10 sequence_tile_config()
    if hdr.tiling.present == AdaptiveBoolean::Off {
        pb.put_bit(0);
    } else {
        pb.put_bit(1);
        pb.put_bit((hdr.tiling.present == AdaptiveBoolean::Adaptive) as u32);
        write_tile_info(
            &hdr.tiling.t,
            pb,
            hdr.sb128,
            hdr.max_width,
            hdr.max_height,
            hdr.level,
            hdr.tier,
        );
    }

    pb.put_bit(hdr.film_grain_present as u32);

    // §5.2 open_bitstream_unit(): extensible OBU has f(1) obu_extension_flag = 0 before trailing_bits()
    pb.put_bit(0);
    pb.put_trailing_bits();
}

/// Encodes `SequenceHeader` into a complete `ObuType::SeqHdr` OBU byte vector.
pub fn write_seq_hdr_obu(hdr: &SequenceHeader) -> Vec<u8> {
    let mut pb = PutBits::new();
    write_seq_hdr(hdr, &mut pb);
    write_obu(ObuType::SeqHdr, 0, 0, 0, &pb.into_bytes())
}

/// Computes `tile_start_mask` for a frame header.
pub fn tile_start_mask(hdr: &FrameHeader) -> u16 {
    let mut mask = 0u16;
    for i in 0..hdr.tiling.t.cols as usize {
        mask |= hdr.tiling.t.col_start_sb[i];
    }
    for i in 0..hdr.tiling.t.rows as usize {
        mask |= hdr.tiling.t.row_start_sb[i];
    }
    mask
}

/// Computes `gdf_unit_count` for a frame header.
pub fn gdf_unit_count(hdr: &FrameHeader) -> i32 {
    let b64 = hdr.gdf.b64size.max(1) as i32;
    let unit_px = 64 * b64;
    let w_units = (hdr.width + unit_px - 1) / unit_px;
    let h_units = (hdr.height + unit_px - 1) / unit_px;
    let n_tiles = (hdr.tiling.t.cols.max(1) as i32) * (hdr.tiling.t.rows.max(1) as i32);
    (w_units * h_units).max(n_tiles)
}

/// Writes the normative AV2 `frame_header_info()` bits into `pb` (`AV2 Spec v1.0.0 §5.18`).
/// Writes the `lr_luma_*`/`lr_chroma_*` unit-size flags for one plane so that
/// the decoder derives `unit_size_log2`.
fn write_lr_unit_size(pb: &mut PutBits, seq: &SequenceHeader, plane: usize, unit_size_log2: u8) {
    let base = if plane == 0 {
        9u8
    } else {
        9 - seq.ss_hor.max(seq.ss_ver)
    };
    let shift = base.saturating_sub(unit_size_log2);
    if shift == 1 {
        pb.put_bit(1);
        return;
    }
    pb.put_bit(0);
    if seq.sb128 >= 2 {
        return;
    }
    if shift == 0 {
        pb.put_bit(1);
        return;
    }
    pb.put_bit(0);
    if seq.sb128 == 0 {
        pb.put_bit((shift == 2) as u32);
    }
}

/// Writes one frame-level NS Wiener filter (`read_wienerns_filter(..., 1)`,
/// AV2 §5.20.10.6). The encoder signals a single class with no reference match,
/// then the full coefficient subset.
fn write_wienerns_frame_filter(
    pb: &mut PutBits,
    plane: usize,
    ns: &crate::av2::headers::NSWienerPlane,
) {
    use crate::av2::wiener_tables::{
        WIENER_NS_TAPS_K, WIENER_NS_TAPS_MIN, WIENER_NS_TAPS_PRESENT_UV, WIENER_NS_TAPS_PRESENT_Y,
    };
    let (n_coeffs, n_subsets) = if plane == 0 {
        (16usize, 4usize)
    } else {
        (18, 3)
    };
    // One class. Group 0 (zeros, no reference filter) is not the predicted
    // group, so one group bit selects it.
    pb.put_bit(1);
    // exact_match_mask = 0.
    pb.put_bit(0);
    let mut subset = 0usize;
    for s in 0..n_subsets - 1 {
        pb.put_bit(1);
        subset = s + 1;
    }
    // dav2d's frame-filter parser does not consume `wiener_ns_uv_sym`
    // (src/obu.c, "FIXME read sym bit"); it always codes every present
    // coefficient, so the encoder does the same.
    let present: &[bool] = if plane == 0 {
        &WIENER_NS_TAPS_PRESENT_Y[subset]
    } else {
        &WIENER_NS_TAPS_PRESENT_UV[subset]
    };
    let min = &WIENER_NS_TAPS_MIN[usize::from(plane > 0)];
    let k = &WIENER_NS_TAPS_K[usize::from(plane > 0)];
    let mut j = 0usize;
    while j < n_coeffs {
        if !present[j] {
            j += 1;
            continue;
        }
        let low = min[j] as i32;
        let mx = 1u32 << k[j];
        let coeff = ns.filter[0][j] as i32;
        debug_assert!(coeff >= low && coeff < low + mx as i32);
        pb.put_bits_subexp_u((0 - low) as u32, mx, k[j] as i32 - 3, (coeff - low) as u32);
        j += 1;
    }
}

pub fn write_frame_hdr(
    seqhdr: &SequenceHeader,
    hdr: &FrameHeader,
    obu_type: ObuType,
    pb: &mut PutBits,
) {
    if obu_type != ObuType::Bridge {
        pb.put_vlc(hdr.id as u32); // cur_mfh_id
    }
    if hdr.id == 0 {
        pb.put_vlc(seqhdr.id as u32); // seq_header_id_in_frame_header
    }

    if obu_type == ObuType::Sef {
        pb.put_bits(
            seqhdr.ref_frames_log2 as i32,
            hdr.existing_frame_idx.max(0) as u32,
        );
        pb.put_bit(1); // derive_sef_order_hint = 1
        if seqhdr.film_grain_present && !seqhdr.reduced_still_picture_header {
            pb.put_bit(0); // apply_grain = 0
        }
        return;
    }

    if !seqhdr.reduced_still_picture_header {
        if !matches!(
            obu_type,
            ObuType::ClosedLoopKf
                | ObuType::OpenLoopKf
                | ObuType::Ras
                | ObuType::Switch
                | ObuType::LeadingTip
                | ObuType::Tip
                | ObuType::Bridge
        ) {
            pb.put_bit((hdr.frame_type == FrameType::Inter) as u32);
        }
        if hdr.frame_type == FrameType::Key && seqhdr.number_of_bits_for_lt_frame_id > 0 {
            pb.put_bits(
                seqhdr.number_of_bits_for_lt_frame_id as i32,
                (hdr.ltr_id + 1).max(0) as u32,
            );
        }
        if obu_type != ObuType::Bridge {
            if obu_type != ObuType::OpenLoopKf {
                pb.put_bit((hdr.show_immediate != 0) as u32);
            }
            if hdr.show_immediate == 0 && !seqhdr.monotonic {
                pb.put_bit((hdr.show_implicit != 0) as u32);
            }
        }
    }

    let size_matches_seq = hdr.width == seqhdr.max_width && hdr.height == seqhdr.max_height;
    let frame_size_override = !size_matches_seq || hdr.frame_size_override != 0;

    if !seqhdr.reduced_still_picture_header {
        if hdr.frame_type != FrameType::Switch {
            pb.put_bit(frame_size_override as u32);
        }
        pb.put_bits(
            seqhdr.order_hint_n_bits.max(1) as i32,
            hdr.frame_offset as u32,
        );
        if hdr.frame_type == FrameType::Inter {
            let sig_pri = hdr.primary_ref_signaled != 0;
            pb.put_bit(sig_pri as u32);
            if obu_type != ObuType::LeadingTip && obu_type != ObuType::Tip {
                pb.put_bit((hdr.no_cross_frame_context != 0) as u32);
            }
            if sig_pri {
                pb.put_bits(3, (hdr.primary_ref_frame & 7) as u32);
            }
        }
    }

    // refresh_frame_flags
    if !(obu_type == ObuType::ClosedLoopKf && seqhdr.max_mlayer_id == 0) {
        if obu_type == ObuType::OpenLoopKf || seqhdr.max_mlayer_id > 0 {
            if seqhdr.short_refresh_frame_flags {
                let slot = hdr.refresh_frame_flags.trailing_zeros().min(15);
                pb.put_bits(seqhdr.ref_frames_log2 as i32, slot);
            } else {
                pb.put_bits(seqhdr.ref_frames as i32, hdr.refresh_frame_flags as u32);
            }
        } else if hdr.frame_type != FrameType::Switch && seqhdr.short_refresh_frame_flags {
            if hdr.refresh_frame_flags != 0 {
                pb.put_bit(1);
                let slot = hdr.refresh_frame_flags.trailing_zeros().min(15);
                pb.put_bits(seqhdr.ref_frames_log2 as i32, slot);
            } else {
                pb.put_bit(0);
            }
        } else {
            pb.put_bits(seqhdr.ref_frames as i32, hdr.refresh_frame_flags as u32);
        }
    }

    // Reference frames (for Inter / Switch)
    if hdr.is_inter_or_switch() {
        if hdr.frame_type != FrameType::Switch && seqhdr.explicit_ref_frame_map {
            pb.put_bit(1); // frame_explicit_ref_frame_map = 1
        }
        if hdr.frame_type == FrameType::Switch || seqhdr.explicit_ref_frame_map {
            let n_refs = hdr.n_ref_frames.clamp(1, 7);
            pb.put_bits(3, n_refs as u32);
            for i in 0..n_refs as usize {
                pb.put_bits(
                    seqhdr.ref_frames_log2 as i32,
                    (hdr.refidx[i].max(0) as u32) % (seqhdr.ref_frames.max(1) as u32),
                );
            }
        }
    }

    // Frame size
    if frame_size_override {
        if hdr.is_inter_or_switch() {
            let n_refs = hdr.n_ref_frames.clamp(1, 7) as usize;
            for _ in 0..n_refs {
                pb.put_bit(0); // found_ref = 0 -> code explicit dimensions
            }
        }
        pb.put_bits(seqhdr.width_n_bits as i32, (hdr.width.max(1) - 1) as u32);
        pb.put_bits(seqhdr.height_n_bits as i32, (hdr.height.max(1) - 1) as u32);
    }

    if hdr.is_inter_or_switch() {
        if seqhdr.ref_frame_mvs {
            pb.put_bit((hdr.use_ref_frame_mvs != 0) as u32);
            if hdr.use_ref_frame_mvs != 0 && hdr.n_ref_frames > 1 && seqhdr.sb128 != 0 {
                pb.put_bit((hdr.tmvp_sample_step > 1) as u32);
            }
        }
        if seqhdr.opfl_refine == 3 {
            if hdr.opfl_refine_type == 1 {
                pb.put_bit(1);
            } else {
                pb.put_bit(0);
                pb.put_bit((hdr.opfl_refine_type == 2) as u32);
            }
        }
    }

    // screen_content_params() & intrabc_params()
    if seqhdr.screen_content_tools == AdaptiveBoolean::Adaptive {
        pb.put_bit((hdr.allow_screen_content_tools != 0) as u32);
    }
    let allow_sct = match seqhdr.screen_content_tools {
        AdaptiveBoolean::On => true,
        AdaptiveBoolean::Off => false,
        AdaptiveBoolean::Adaptive => hdr.allow_screen_content_tools != 0,
    };
    if allow_sct && seqhdr.force_integer_mv == AdaptiveBoolean::Adaptive {
        pb.put_bit((hdr.force_integer_mv != 0) as u32);
    }
    let force_int_mv = if allow_sct {
        match seqhdr.force_integer_mv {
            AdaptiveBoolean::On => true,
            AdaptiveBoolean::Off => false,
            AdaptiveBoolean::Adaptive => hdr.force_integer_mv != 0,
        }
    } else {
        false
    };

    pb.put_bit((hdr.allow_intrabc != 0) as u32);
    if hdr.allow_intrabc != 0 {
        if hdr.is_key_or_intra() {
            pb.put_bit((hdr.allow_global_intrabc != 0) as u32);
            if hdr.allow_global_intrabc != 0 {
                pb.put_bit((hdr.allow_local_intrabc != 0) as u32);
            }
        }
        if seqhdr.allow_max_bvp_drl_bits {
            pb.put_bit(0); // change_bvp_drl = 0
        }
    }

    if hdr.is_inter_or_switch() {
        if seqhdr.allow_frame_max_drl_bits {
            pb.put_bit(0); // change_drl = 0
        }
        if !force_int_mv {
            if hdr.mv_precision == 2 {
                pb.put_bit(1); // use_qtr_precision_mv = 1
            } else {
                pb.put_bit(0);
                pb.put_bit((hdr.mv_precision >= 3) as u32); // allow_high_precision_mv
            }
        }
        if hdr.subpel_filter_mode == FilterMode::Switchable {
            pb.put_bit(1);
        } else {
            pb.put_bit(0);
            pb.put_bits(2, (hdr.subpel_filter_mode as u32) & 3);
        }
        if seqhdr.frame_motion_modes_present {
            for shift in [1, 2, 3, 4] {
                if (seqhdr.motion_modes & (1 << shift)) != 0 {
                    pb.put_bit(((hdr.motion_modes >> shift) & 1) as u32);
                }
            }
        }
    }

    pb.put_bit((hdr.disable_cdf_update != 0) as u32);

    // tile_info()
    let sb128 = if hdr.is_inter_or_switch() {
        seqhdr.sb128
    } else {
        (seqhdr.sb128 != 0) as u8
    };
    if seqhdr.tiling.present == AdaptiveBoolean::Off {
        write_tile_info(
            &hdr.tiling.t,
            pb,
            sb128,
            hdr.width,
            hdr.height,
            seqhdr.level,
            seqhdr.tier,
        );
    } else if seqhdr.tiling.present == AdaptiveBoolean::Adaptive {
        pb.put_bit(1); // reuse_tile_info = 1
    }
    if hdr.tiling.t.log2_cols != 0 || hdr.tiling.t.log2_rows != 0 {
        if !seqhdr.avg_cdf || seqhdr.avg_cdf_type == 0 {
            let n = (hdr.tiling.t.log2_cols + hdr.tiling.t.log2_rows) as i32;
            pb.put_bits(n, hdr.tiling.update as u32);
        }
        pb.put_bits(2, (hdr.tiling.n_bytes.clamp(1, 4) - 1) as u32);
    }

    // quantization_params()
    let q_bits = 8 + (seqhdr.hbd != 0) as i32;
    pb.put_bits(q_bits, hdr.quant.yac as u32);
    if seqhdr.ydc_dq_enabled {
        let has_ydc = hdr.quant.ydc_delta != 0;
        pb.put_bit(has_ydc as u32);
        if has_ydc {
            pb.put_sbits(7, hdr.quant.ydc_delta as i32);
        }
    }
    if seqhdr.layout != PixelLayout::I400 && (seqhdr.uvdc_dq_enabled || seqhdr.uvac_dq_enabled) {
        let diff_uv = seqhdr.separate_uv_delta_q
            && (hdr.quant.vdc_delta != hdr.quant.udc_delta
                || hdr.quant.vac_delta != hdr.quant.uac_delta);
        if seqhdr.separate_uv_delta_q {
            pb.put_bit(diff_uv as u32);
        }
        if seqhdr.uvdc_dq_enabled {
            let has_udc = hdr.quant.udc_delta != 0;
            pb.put_bit(has_udc as u32);
            if has_udc {
                pb.put_sbits(7, hdr.quant.udc_delta as i32);
            }
        }
        if seqhdr.uvac_dq_enabled {
            let has_uac = hdr.quant.uac_delta != 0;
            pb.put_bit(has_uac as u32);
            if has_uac {
                pb.put_sbits(7, hdr.quant.uac_delta as i32);
            }
        }
        if diff_uv {
            if seqhdr.uvdc_dq_enabled {
                let has_vdc = hdr.quant.vdc_delta != 0;
                pb.put_bit(has_vdc as u32);
                if has_vdc {
                    pb.put_sbits(7, hdr.quant.vdc_delta as i32);
                }
            }
            if seqhdr.uvac_dq_enabled {
                let has_vac = hdr.quant.vac_delta != 0;
                pb.put_bit(has_vac as u32);
                if has_vac {
                    pb.put_sbits(7, hdr.quant.vac_delta as i32);
                }
            }
        }
    }

    // segmentation_params()
    pb.put_bit((hdr.segmentation.enabled != 0) as u32);
    if hdr.segmentation.enabled != 0 {
        if seqhdr.segmentation.info_present && seqhdr.segmentation.adaptive {
            pb.put_bit(0); // reuse_seg_info = 0 -> write explicit seg_info
        }
        if !seqhdr.segmentation.info_present || seqhdr.segmentation.adaptive {
            write_seg_info(
                &hdr.segmentation.d,
                pb,
                8usize << (seqhdr.segmentation.ext as usize),
            );
        }
        if hdr.primary_ref_frame != PRIMARY_REF_NONE {
            pb.put_bit((hdr.segmentation.update_map != 0) as u32);
            if hdr.segmentation.update_map != 0 && hdr.frame_type != FrameType::Switch {
                pb.put_bit((hdr.segmentation.temporal != 0) as u32);
            }
        }
    }

    // setup_qm_params()
    pb.put_bit((hdr.quant.qm.enabled != 0) as u32);
    if hdr.quant.qm.enabled != 0 {
        let qm_num = if hdr.segmentation.enabled != 0 {
            let num = hdr.quant.qm.num.clamp(1, 4);
            pb.put_bits(2, (num - 1) as u32);
            num as usize
        } else {
            1usize
        };
        for i in 0..qm_num {
            pb.put_bits(4, (hdr.quant.qm.y[i] & 0xF) as u32);
            if seqhdr.layout != PixelLayout::I400 {
                let uv_same = hdr.quant.qm.u[i] == hdr.quant.qm.y[i]
                    && hdr.quant.qm.v[i] == hdr.quant.qm.y[i];
                pb.put_bit(uv_same as u32);
                if !uv_same {
                    pb.put_bits(4, (hdr.quant.qm.u[i] & 0xF) as u32);
                    if seqhdr.separate_uv_delta_q {
                        pb.put_bits(4, (hdr.quant.qm.v[i] & 0xF) as u32);
                    }
                }
            }
        }
    }

    // delta_q_params()
    if hdr.quant.yac != 0 {
        pb.put_bit((hdr.delta.q.present != 0) as u32);
        if hdr.delta.q.present != 0 {
            pb.put_bits(2, (hdr.delta.q.res_log2 & 3) as u32);
        }
    }

    // Compute CodedLossless per AV2 Spec §5.18.1
    let delta_lossless = hdr.quant.ydc_delta == 0
        && hdr.quant.udc_delta == 0
        && hdr.quant.uac_delta == 0
        && hdr.quant.vdc_delta == 0
        && hdr.quant.vac_delta == 0
        && hdr.delta.q.present == 0;
    let mut coded_lossless = hdr.quant.yac == 0 && delta_lossless;
    if hdr.segmentation.enabled != 0 {
        for i in 0..MAX_SEGMENTS {
            let q = (hdr.quant.yac as i32 + hdr.segmentation.d.delta_q[i] as i32).clamp(0, 255);
            if q != 0 || !delta_lossless {
                coded_lossless = false;
            }
        }
    }

    if !coded_lossless {
        if seqhdr.tcq == AdaptiveBoolean::Adaptive {
            pb.put_bit((hdr.tcq != 0) as u32);
        }
        let allow_tcq = match seqhdr.tcq {
            AdaptiveBoolean::On => true,
            AdaptiveBoolean::Off => false,
            AdaptiveBoolean::Adaptive => hdr.tcq != 0,
        };
        if !allow_tcq && seqhdr.parity_hiding {
            pb.put_bit((hdr.parity_hiding != 0) as u32);
        }

        // deblocking_filter_params()
        if hdr.frame_type == FrameType::Inter && seqhdr.db_sub_pu {
            pb.put_bit((hdr.deblock.sub_pu != 0) as u32);
        }
        let apply_y0 = hdr.deblock.level_y[0] != 0;
        let apply_y1 = hdr.deblock.level_y[1] != 0;
        let apply_u = hdr.deblock.level_u != 0;
        let apply_v = hdr.deblock.level_v != 0;
        pb.put_bit(apply_y0 as u32);
        pb.put_bit(apply_y1 as u32);
        if seqhdr.layout != PixelLayout::I400 && (apply_y0 || apply_y1) {
            pb.put_bit(apply_u as u32);
            pb.put_bit(apply_v as u32);
        }
        let df_bits = seqhdr.df_par_bits.max(2) as i32;
        let df_off = 1i32 << (df_bits - 1);
        for (idx, &(apply, dq)) in [
            (apply_y0, hdr.deblock.delta_q_y[0]),
            (apply_y1, hdr.deblock.delta_q_y[1]),
            (
                seqhdr.layout != PixelLayout::I400 && (apply_y0 || apply_y1) && apply_u,
                hdr.deblock.delta_q_u,
            ),
            (
                seqhdr.layout != PixelLayout::I400 && (apply_y0 || apply_y1) && apply_v,
                hdr.deblock.delta_q_v,
            ),
        ]
        .iter()
        .enumerate()
        {
            if apply {
                let default_dq = if idx == 1 {
                    hdr.deblock.delta_q_y[0]
                } else {
                    0
                };
                let has_dq = dq != default_dq;
                pb.put_bit(has_dq as u32);
                if has_dq {
                    let coded_dq = ((dq as i32) + df_off).clamp(0, (1i32 << df_bits) - 1) as u32;
                    pb.put_bits(df_bits, coded_dq);
                }
            }
        }

        // gdf_params()
        if seqhdr.gdf {
            let gdf_on = hdr.gdf.enabled != AdaptiveBoolean::Off;
            if !seqhdr.reduced_still_picture_header {
                pb.put_bit(gdf_on as u32);
            }
            if gdf_on || seqhdr.reduced_still_picture_header {
                let gdf_bs = 128 << (hdr.sb128 == 2) as usize;
                if hdr.width.max(hdr.height) > gdf_bs {
                    pb.put_bit((hdr.gdf.enabled == AdaptiveBoolean::Adaptive) as u32);
                }
                let qp_base = if hdr.is_key_or_intra() { 85 } else { 110 };
                let qp_diff = hdr.quant.yac as i32 - qp_base - 48 * seqhdr.hbd as i32;
                let base = crate::av2::intops::iclip((qp_diff - 37) / 25, 0, 2);
                let qp_idx_offset = (hdr.gdf.qp_idx as i32 - base).clamp(0, 3) as u32;
                pb.put_bits(2, qp_idx_offset);
                pb.put_bits(2, (hdr.gdf.scale.saturating_sub(1) & 3) as u32);
            }
        }

        // cdef_params()
        if seqhdr.cdef {
            let cdef_on = hdr.cdef.enabled != 0
                || hdr.cdef.y_strength[0] != 0
                || hdr.cdef.uv_strength[0] != 0;
            if !seqhdr.reduced_still_picture_header {
                pb.put_bit(cdef_on as u32);
            }
            if cdef_on || seqhdr.reduced_still_picture_header {
                pb.put_bits(2, (hdr.cdef.damping.clamp(3, 6) - 3) as u32);
                let n_str = hdr
                    .cdef
                    .n_strengths
                    .max(1u8 << hdr.cdef.n_bits)
                    .clamp(1, MAX_CDEF_STRENGTHS as u8);
                pb.put_bits(3, (n_str - 1) as u32);
                if seqhdr.cdef_on_skiptx == AdaptiveBoolean::Adaptive {
                    pb.put_bit((hdr.cdef.on_skiptx != 0) as u32);
                }
                for i in 0..n_str as usize {
                    let y_str = hdr.cdef.y_strength[i];
                    let y_pri = y_str >> 2;
                    let y_sec = (y_str & 3).min(3);
                    if y_pri == 0 {
                        pb.put_bit(1); // cdef_y_pri_zero = 1
                    } else {
                        pb.put_bit(0);
                        pb.put_bits(4, (y_pri & 0xF) as u32);
                    }
                    pb.put_bits(2, y_sec as u32);
                    if seqhdr.layout != PixelLayout::I400 {
                        let uv_str = hdr.cdef.uv_strength[i];
                        let uv_pri = uv_str >> 2;
                        let uv_sec = (uv_str & 3).min(3);
                        if uv_pri == 0 {
                            pb.put_bit(1); // cdef_uv_pri_zero = 1
                        } else {
                            pb.put_bit(0);
                            pb.put_bits(4, (uv_pri & 0xF) as u32);
                        }
                        pb.put_bits(2, uv_sec as u32);
                    }
                }
            }
        }

        // lr_params()
        if seqhdr.restoration {
            let n_planes = if seqhdr.layout == PixelLayout::I400 {
                1usize
            } else {
                3usize
            };
            for p in 0..n_planes {
                let disable_mask = seqhdr.rst_disable_mask[if p != 0 { 1 } else { 0 }];
                let rtype = hdr.restoration.planes[p].restoration_type;
                if disable_mask == 0 {
                    pb.put_bits(2, (rtype & 3) as u32);
                } else if disable_mask != 3 {
                    pb.put_bit((rtype != 0) as u32);
                }
                if rtype >= RestorationType::NsWiener as u8 {
                    let ns = &hdr.restoration.planes[p].ns;
                    pb.put_bit(ns.frame_filters_on as u32);
                    if ns.frame_filters_on != 0 {
                        if hdr.is_inter_or_switch() && hdr.n_ref_frames > 0 {
                            pb.put_bit(ns.temporal as u32);
                        }
                        if ns.temporal != 0 {
                            if hdr.n_ref_frames > 1 {
                                let n = if hdr.n_ref_frames <= 2 {
                                    1
                                } else {
                                    ulog2(hdr.n_ref_frames as u32 - 1) + 1
                                };
                                pb.put_bits(n, ns.refidx as u32);
                            }
                        } else {
                            // dav2d reads `num_filter_classes_idx` for chroma as
                            // well even though AVM `max_num_classes` is 1 there;
                            // signalling it keeps the dav2d oracle in sync.
                            pb.put_bits(3, ns.num_classes_idx as u32);
                        }
                    }
                }
            }
            if hdr.restoration.planes[0].restoration_type != 0 {
                write_lr_unit_size(pb, seqhdr, 0, hdr.restoration.unit_size[0]);
            }
            if n_planes > 1
                && (hdr.restoration.planes[1].restoration_type != 0
                    || hdr.restoration.planes[2].restoration_type != 0)
            {
                write_lr_unit_size(pb, seqhdr, 1, hdr.restoration.unit_size[1]);
            }
            for p in 0..n_planes {
                let ns = &hdr.restoration.planes[p].ns;
                if hdr.restoration.planes[p].restoration_type >= RestorationType::NsWiener as u8
                    && ns.frame_filters_on != 0
                    && ns.temporal == 0
                {
                    write_wienerns_frame_filter(pb, p, ns);
                }
            }
        }

        // ccso_params() (AV2 §5.18.7.12)
        if seqhdr.ccso {
            if !seqhdr.reduced_still_picture_header {
                pb.put_bit((hdr.ccso.enabled != 0) as u32);
            }
            if hdr.ccso.enabled != 0 {
                let n_planes = if seqhdr.layout == PixelLayout::I400 {
                    1
                } else {
                    3
                };
                for p in 0..n_planes {
                    let p_hdr = &hdr.ccso.planes[p];
                    pb.put_bit((p_hdr.enabled != 0) as u32);
                    if p_hdr.enabled == 0 {
                        continue;
                    }
                    if hdr.is_inter_or_switch() {
                        pb.put_bit(p_hdr.reuse as u32);
                        pb.put_bit(p_hdr.sb_reuse as u32);
                        if p_hdr.reuse != 0 || p_hdr.sb_reuse != 0 {
                            let n_bits = if hdr.n_ref_frames <= 2 {
                                hdr.n_ref_frames.saturating_sub(1) as i32
                            } else {
                                1 + ulog2(hdr.n_ref_frames as u32 - 1)
                            };
                            if n_bits > 0 {
                                pb.put_bits(n_bits, p_hdr.refidx as u32);
                            }
                        }
                    }
                    if p_hdr.reuse == 0 {
                        pb.put_bit(p_hdr.bo_only as u32);
                        pb.put_bits(2, p_hdr.scale_idx as u32);
                        if p_hdr.bo_only != 0 {
                            pb.put_bits(3, p_hdr.max_band_log2 as u32);
                        } else {
                            pb.put_bits(2, p_hdr.quant_idx as u32);
                            pb.put_bits(3, p_hdr.ext_filter_support as u32);
                            if CCSO_QUANT_SZ[p_hdr.scale_idx as usize][p_hdr.quant_idx as usize]
                                != 0
                            {
                                pb.put_bit(p_hdr.edge_clf as u32);
                            }
                            pb.put_bits(2, p_hdr.max_band_log2 as u32);
                        }
                        let n_intervals = if p_hdr.bo_only != 0 {
                            1
                        } else {
                            3 - p_hdr.edge_clf as usize
                        };
                        let max_band = 1 << p_hdr.max_band_log2;
                        for n in 0..n_intervals {
                            for m in 0..n_intervals {
                                for o in 0..max_band {
                                    let byte_idx = n * 16 + m * 4 + (o >> 1);
                                    let off = (p_hdr.filter_off[byte_idx] >> (4 * (o & 1))) & 0x7;
                                    for _ in 0..off {
                                        pb.put_bit(1);
                                    }
                                    if off < 7 {
                                        pb.put_bit(0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // read_tx_mode()
        pb.put_bit((hdr.txfm_mode == TxfmMode::Switchable) as u32);
    }

    // frame_reference_mode(), skip_mode_params(), allow_bawp, allow_warpmv_mode
    if hdr.is_inter_or_switch() {
        pb.put_bit((hdr.switchable_comp_refs != 0) as u32);
        pb.put_bit((hdr.skip_mode_enabled != 0) as u32);
        if seqhdr.bawp {
            pb.put_bit((hdr.bawp != 0) as u32);
        }
        if (seqhdr.motion_modes & (1 << 3)) != 0 {
            pb.put_bit((hdr.warp_motion != 0) as u32);
        }
    }

    pb.put_bits(2, (hdr.reduced_txtp_set & 3) as u32);

    // global_motion_params()
    if hdr.is_inter_or_switch() && seqhdr.global_motion {
        pb.put_bit(0); // use_global_motion = 0
    }

    // film_grain_config()
    if seqhdr.film_grain_present && (hdr.show_immediate != 0 || hdr.show_implicit != 0) {
        if !seqhdr.reduced_still_picture_header {
            pb.put_bit((hdr.film_grain.present != 0) as u32);
        }
        if hdr.film_grain.present != 0 || seqhdr.reduced_still_picture_header {
            pb.put_bits(3, (hdr.film_grain.id & 7) as u32);
            pb.put_bits(16, hdr.film_grain.seed & 0xFFFF);
        }
    }
}

/// Packs a `FrameHeader` and all tile payloads into a complete AV2 Tile Group OBU
/// (`ObuType::ClosedLoopKf` for keyframes, `ObuType::TileGrp` for inter frames).
pub fn write_frame_and_tile_group_obu(
    seqhdr: &SequenceHeader,
    hdr: &FrameHeader,
    obu_type: ObuType,
    tile_payloads: &[Vec<u8>],
) -> Vec<u8> {
    let mut pb = PutBits::new();

    if !matches!(obu_type, ObuType::Sef | ObuType::Tip | ObuType::Bridge) {
        pb.put_bit(1); // is_first_tile_group = 1 -> frame_header_present_flag is implied 1
    }

    write_frame_hdr(seqhdr, hdr, obu_type, &mut pb);

    if matches!(obu_type, ObuType::Sef | ObuType::Tip | ObuType::Bridge) {
        pb.put_trailing_bits();
    } else {
        let n_tiles = (hdr.tiling.t.cols as usize) * (hdr.tiling.t.rows as usize);
        if n_tiles > 1 {
            pb.put_bit(0); // tile_start_and_end_present_flag = 0 -> covers all tiles 0..n_tiles-1
        }
        // Per AV2 Spec §5.19.1 tile_group_obu(): byte_alignment() pads with 0 bits to byte boundary
        pb.bytealign();

        let n_bytes = hdr.tiling.n_bytes.clamp(1, 4) as usize;
        for (idx, tile_bytes) in tile_payloads.iter().enumerate() {
            if idx + 1 < tile_payloads.len() {
                let sz_minus_1 = tile_bytes.len().saturating_sub(1);
                for k in 0..n_bytes {
                    pb.put_bits(8, ((sz_minus_1 >> (k * 8)) & 0xFF) as u32);
                }
            }
            pb.put_bytes(tile_bytes);
        }
    }

    write_obu(
        obu_type,
        hdr.tlayer_id,
        hdr.mlayer_id,
        hdr.xlayer_id,
        &pb.into_bytes(),
    )
}

/// Builds a default normative [`SequenceHeader`] for the given dimensions, layout, and bit depth.
pub fn build_default_sequence_header(
    width: i32,
    height: i32,
    layout: PixelLayout,
    bitdepth: u8,
    use_sb128: bool,
) -> SequenceHeader {
    let mut seq = SequenceHeader::default();
    seq.id = 0;
    seq.profile = match (layout, bitdepth) {
        (PixelLayout::I420 | PixelLayout::I400, 8 | 10) => 0,
        (PixelLayout::I444, 8 | 10) => 1,
        _ => 2,
    };
    seq.reduced_still_picture_header = false;
    seq.level = 8;
    seq.tier = 0;
    seq.layout = layout;
    match layout {
        PixelLayout::I420 | PixelLayout::I400 => {
            seq.ss_hor = 1;
            seq.ss_ver = 1;
        }
        PixelLayout::I422 => {
            seq.ss_hor = 1;
            seq.ss_ver = 0;
        }
        PixelLayout::I444 => {
            seq.ss_hor = 0;
            seq.ss_ver = 0;
        }
    }
    seq.hbd = match bitdepth {
        10 => 1,
        12 => 2,
        _ => 0,
    };
    seq.still_picture = false;
    seq.monotonic = true;
    seq.max_width = width;
    seq.max_height = height;
    seq.width_n_bits = (ulog2(width.max(1) as u32) + 1).clamp(1, 16) as u8;
    seq.height_n_bits = (ulog2(height.max(1) as u32) + 1).clamp(1, 16) as u8;
    seq.sb128 = if use_sb128 { 1 } else { 0 };
    seq.max_pb_aspect_ratio_log2 = 3;
    // The minimal intra path does not implement directional reference-edge
    // filtering yet, so advertise it as disabled to keep prediction exact.
    seq.intra_edge_filter = false;
    seq.mrls = true;
    seq.cfl = layout != PixelLayout::I400;
    seq.cfl_ds_filter_index = 0;
    seq.motion_modes = 1;
    seq.order_hint_n_bits = 8;
    seq.refmv_bank = true;
    seq.drl_reorder = 0;
    seq.explicit_ref_frame_map = true;
    seq.ref_frames = 8;
    seq.ref_frames_log2 = 3;
    seq.def_max_drl_bits = 2;
    seq.def_max_bvp_drl_bits = 2;
    seq.screen_content_tools = AdaptiveBoolean::Off;
    seq.force_integer_mv = AdaptiveBoolean::Adaptive;
    seq.equal_ac_dc_q = true;
    // With equal_ac_dc_q set, the DC delta-q enable flags are not present in the
    // sequence header and are inferred to 0 (AV2 v1.0.0 §5.4.7/§6.4.7). The frame
    // header must therefore not carry the corresponding delta-q present bits.
    seq.ydc_dq_enabled = false;
    seq.uvdc_dq_enabled = false;
    seq.uvac_dq_enabled = layout != PixelLayout::I400;
    seq.cdef = true;
    seq.gdf = true;
    seq.restoration = false;
    seq.ccso = false;
    seq.cdef_on_skiptx = AdaptiveBoolean::Off;
    seq.df_par_bits = 2;
    seq.tiling.present = AdaptiveBoolean::Off;
    seq.ext_partitions = true;
    seq.uneven_4way_partitions = true;
    seq
}

/// Builds a [`FrameHeader`] matching `seqhdr` for the given frame type, POC, and base quantizer `qidx`.
pub fn build_frame_header(
    seqhdr: &SequenceHeader,
    frame_type: FrameType,
    poc: u8,
    qidx: u16,
    log2_tile_cols: u8,
    log2_tile_rows: u8,
) -> FrameHeader {
    let mut fh = FrameHeader::default();
    fh.id = 0;
    fh.frame_type = frame_type;
    fh.width = seqhdr.max_width;
    fh.height = seqhdr.max_height;
    fh.frame_offset = poc;
    fh.show_immediate = 1;
    fh.primary_ref_frame = PRIMARY_REF_NONE;
    fh.secondary_ref_frame = PRIMARY_REF_NONE;
    fh.refresh_frame_flags = if frame_type == FrameType::Key {
        ((1u32 << seqhdr.ref_frames) - 1) as u16
    } else {
        1u16 << (poc as u16 % seqhdr.ref_frames.max(1) as u16)
    };
    if frame_type == FrameType::Inter {
        fh.no_cross_frame_context = 1;
        fh.n_ref_frames = 2;
        fh.refidx[0] = ((poc.wrapping_sub(1)) % seqhdr.ref_frames.max(1)) as i8;
        fh.refidx[1] = 0;
        fh.has_past_refs = 1;
        fh.mv_precision = 2;
        fh.subpel_filter_mode = FilterMode::Regular8Tap;
        fh.motion_modes = 1;
        fh.max_drl_bits = seqhdr.def_max_drl_bits;
    }
    fh.sb128 = if fh.is_inter_or_switch() {
        seqhdr.sb128
    } else {
        (seqhdr.sb128 != 0) as u8
    };
    let sbmul = if seqhdr.sb128 == 2 && fh.is_key_or_intra() {
        2
    } else {
        1
    };
    init_uniform_tile_info(
        &mut fh.tiling.t,
        sbmul,
        fh.sb128,
        seqhdr.sb128,
        fh.width,
        fh.height,
        seqhdr.level,
        seqhdr.tier,
        log2_tile_cols,
        log2_tile_rows,
    );
    fh.tiling.n_bytes = 1;
    fh.quant.yac = qidx.min(255);
    let is_lossless = (fh.quant.yac == 0
        && fh.quant.ydc_delta == 0
        && fh.quant.udc_delta == 0
        && fh.quant.uac_delta == 0
        && fh.quant.vdc_delta == 0
        && fh.quant.vac_delta == 0) as u8;
    fh.quant.lossless[0] = is_lossless;
    fh.segmentation.lossless[0] = is_lossless;
    fh.all_lossless = is_lossless;
    fh.any_lossless = is_lossless;
    fh.txfm_mode = if is_lossless != 0 {
        TxfmMode::Only4x4
    } else {
        TxfmMode::Switchable
    };
    fh.reduced_txtp_set = 2;
    fh.cdef.enabled = seqhdr.cdef as u8;
    fh.cdef.damping = 3;
    fh.cdef.n_bits = 0;
    fh.cdef.n_strengths = if seqhdr.cdef { 1 } else { 0 };
    fh.gdf.b64size = 1 << (!seqhdr.gdf_unit_matches_sbsz as u8).max(fh.sb128);
    if !seqhdr.gdf_unit_matches_sbsz && fh.sb128 == 0 {
        let mut mask = 0;
        for t in 0..fh.tiling.t.cols.saturating_sub(1) as usize {
            mask |= fh.tiling.t.col_start_sb[t];
        }
        for t in 0..fh.tiling.t.rows.saturating_sub(1) as usize {
            mask |= fh.tiling.t.row_start_sb[t];
        }
        if mask & 1 != 0 {
            fh.gdf.b64size = 1;
        }
    }
    fh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_delimiter_obu_bytes() {
        let td = write_temporal_delimiter_obu();
        // ULEB128(1) = 0x01, header: ext=0, type=2(00010), tlayer=00 -> 0x08
        assert_eq!(td, vec![0x01, 0x08]);
    }
}
