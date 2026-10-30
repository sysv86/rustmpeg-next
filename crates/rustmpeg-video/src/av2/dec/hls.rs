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

//! AV2 high-level syntax parsing: OBU framing, sequence headers, frame headers,
//! and tile-group splitting (AV2 Specification v1.0.0 §5.2, §5.4, §5.18, §5.19).

use crate::av2::ccso::CCSO_QUANT_SZ;
use crate::av2::dec::bitio::BitReader;
use crate::av2::error::{Av2Error, Result};
use crate::av2::headers::{
    AdaptiveBoolean, CropInfo, FilmGrainData, FilterMode, FrameHeader, FrameType, MAX_SEGMENTS,
    MAX_TILE_COLS, MAX_TILE_ROWS, NSWienerPlane, ObuType, PRIMARY_REF_NONE, PixelLayout,
    RestorationType, SequenceHeader, TileInfo, TxfmMode,
};
use crate::av2::intops::{get_poc_diff, iclip, ulog2};
use crate::av2::obu::init_uniform_tile_info;

/// One reference slot's metadata needed to derive the primary reference
/// (`DerivedPrimaryRefFrame`, AV2 Spec v1.0.0 §7.8 / dav2d `derive_pri_sec_ref`).
#[derive(Debug, Clone, Copy, Default)]
pub struct RefFrameInfo {
    /// The slot holds a key or intra frame (such frames are not eligible).
    pub is_intra: bool,
    /// The slot is occupied by a decoded frame.
    pub present: bool,
    /// Reference frame `base_q_idx`.
    pub qidx: u16,
    /// Reference frame order hint (`frame_offset`).
    pub poc: i32,
    /// Resolved frame-level NS Wiener filters per plane (`RefFrameLrWienerNs`,
    /// `RefFrameFiltersOn`; AV2 §5.18.7.11). Used by `temporal_pred_flag`.
    pub ns: [NSWienerPlane; 3],
}

/// One parsed OBU with a borrowed payload.
#[derive(Debug, Clone)]
pub struct Obu<'a> {
    /// OBU type.
    pub obu_type: ObuType,
    /// Temporal layer id from the OBU header.
    pub tlayer_id: u8,
    /// Payload bytes following the OBU header.
    pub payload: &'a [u8],
}

/// Splits a temporal unit into OBUs and validates the framing.
pub fn split_obus(data: &[u8]) -> Result<Vec<Obu<'_>>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        let (obu_size, n) = read_uleb128(&data[pos..])?;
        pos += n;
        let obu_size = obu_size as usize;
        if obu_size == 0 || pos + obu_size > data.len() {
            return Err(Av2Error::BitstreamError(format!(
                "OBU size {obu_size} exceeds the remaining {} bytes",
                data.len().saturating_sub(pos)
            )));
        }
        let mut br = BitReader::new(&data[pos..pos + obu_size]);
        let ext = br.get_bit()? != 0;
        let obu_type_raw = br.get_bits(5)? as u8;
        let tlayer_id = br.get_bits(2)? as u8;
        let hdr_bytes = if ext {
            let _mlayer_id = br.get_bits(3)?;
            let _xlayer_id = br.get_bits(5)?;
            2usize
        } else {
            1usize
        };
        let obu_type = obu_type_from_u8(obu_type_raw)?;
        if obu_size < hdr_bytes {
            return Err(Av2Error::BitstreamError(
                "OBU size is smaller than its header".to_string(),
            ));
        }
        let payload = &data[pos + hdr_bytes..pos + obu_size];
        out.push(Obu {
            obu_type,
            tlayer_id,
            payload,
        });
        pos += obu_size;
    }
    Ok(out)
}

fn read_uleb128(data: &[u8]) -> Result<(u32, usize)> {
    let mut value = 0u64;
    for (i, &byte) in data.iter().enumerate().take(8) {
        value |= ((byte & 0x7F) as u64) << (i * 7);
        if byte & 0x80 == 0 {
            if value > u32::MAX as u64 {
                return Err(Av2Error::BitstreamError(
                    "uleb128 value exceeds 32 bits".to_string(),
                ));
            }
            return Ok((value as u32, i + 1));
        }
    }
    Err(Av2Error::BitstreamError(
        "invalid leb128 OBU size".to_string(),
    ))
}

fn obu_type_from_u8(v: u8) -> Result<ObuType> {
    use ObuType::*;
    Ok(match v {
        1 => SeqHdr,
        2 => Td,
        3 => MultiFrameHdr,
        4 => ClosedLoopKf,
        5 => OpenLoopKf,
        6 => LeadingTileGrp,
        7 => TileGrp,
        8 => Metadata,
        9 => MetadataGrp,
        10 => Switch,
        11 => LeadingSef,
        12 => Sef,
        13 => LeadingTip,
        14 => Tip,
        15 => BufRmTiming,
        16 => LayerCfgRec,
        17 => AtlasSeg,
        18 => OpPtSet,
        19 => Bridge,
        20 => Msdo,
        21 => Ras,
        22 => Qm,
        23 => Fgm,
        24 => ContentInterp,
        25 => Padding,
        other => {
            return Err(Av2Error::BitstreamError(format!(
                "reserved OBU type {other}"
            )));
        }
    })
}

fn read_layout(idx: u32) -> Result<PixelLayout> {
    Ok(match idx {
        0 => PixelLayout::I420,
        1 => PixelLayout::I400,
        2 => PixelLayout::I444,
        3 => PixelLayout::I422,
        other => {
            return Err(Av2Error::BitstreamError(format!(
                "invalid chroma layout index {other}"
            )));
        }
    })
}

fn read_adaptive_boolean(br: &mut BitReader<'_>) -> Result<AdaptiveBoolean> {
    Ok(if br.get_bit()? != 0 {
        AdaptiveBoolean::Adaptive
    } else if br.get_bit()? != 0 {
        AdaptiveBoolean::On
    } else {
        AdaptiveBoolean::Off
    })
}

fn read_seg_info(
    br: &mut BitReader<'_>,
    n_seg: usize,
) -> Result<crate::av2::headers::SegmentationDataSet> {
    let mut seg = crate::av2::headers::SegmentationDataSet::default();
    for n in 0..n_seg {
        let m = 1u16 << n;
        if br.get_bit()? != 0 {
            seg.delta_q_mask |= m;
            // `seg_info()` stores the 10-bit signed value clipped to
            // [-351, 351] (AV2 Spec v1.0.0 §5.4.9; dav2d `parse_seg_info`).
            seg.delta_q[n] = iclip(br.get_sbits(10)?, -351, 351) as i16;
        }
        if br.get_bit()? != 0 {
            seg.skip_mask |= m;
        }
        if br.get_bit()? != 0 {
            seg.globalmv_mask |= m;
        }
    }
    Ok(seg)
}

fn tile_log2(blk_size: i32, target: i32) -> i32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// Reads the raw `tile_info()` syntax and leaves the grid to
/// [`init_uniform_tile_info`].
fn read_tile_info(
    br: &mut BitReader<'_>,
    t: &mut TileInfo,
    sb128: u8,
    width: i32,
    height: i32,
    level: u8,
    tier: u8,
) -> Result<()> {
    let sbsz_log2 = 6 + sb128 as i32;
    let sbsz_min1 = (64 << sb128) - 1;
    let sbw = (width + sbsz_min1) >> sbsz_log2;
    let sbh = (height + sbsz_min1) >> sbsz_log2;
    let w_adj = (level >= 18) as i32 + ((level >= 14 && tier != 0) as i32);
    let max_tile_width_sb = 4096 >> (sbsz_log2 - w_adj);
    let sz_adj = (level >= 14) as i32 + (level >= 18) as i32 + ((level >= 14 && tier != 0) as i32);
    let max_tile_area_sb = (4096 * 2304) >> (2 * sbsz_log2 - sz_adj);
    let min_log2_cols = tile_log2(max_tile_width_sb, sbw);
    let max_log2_cols = tile_log2(1, sbw.min(MAX_TILE_COLS as i32));
    let max_log2_rows = tile_log2(1, sbh.min(MAX_TILE_ROWS as i32));
    let min_log2_tiles = tile_log2(max_tile_area_sb, sbw * sbh).max(min_log2_cols);

    t.uniform = br.get_bit()? != 0;
    if !t.uniform {
        return Err(Av2Error::Unsupported("non-uniform tile layout".to_string()));
    }
    let mut log2_cols = min_log2_cols;
    while log2_cols < max_log2_cols && br.get_bit()? != 0 {
        log2_cols += 1;
    }
    let min_log2_rows = (min_log2_tiles - log2_cols).max(0);
    let mut log2_rows = min_log2_rows;
    while log2_rows < max_log2_rows && br.get_bit()? != 0 {
        log2_rows += 1;
    }
    t.log2_cols = log2_cols as u8;
    t.log2_rows = log2_rows as u8;
    Ok(())
}

fn finish_tile_grid(
    t: &mut TileInfo,
    sb128: u8,
    seq_sb128: u8,
    width: i32,
    height: i32,
    level: u8,
    tier: u8,
) {
    init_uniform_tile_info(
        t,
        1,
        sb128,
        seq_sb128,
        width,
        height,
        level,
        tier,
        t.log2_cols,
        t.log2_rows,
    );
}

/// Parses a `sequence_header_obu()` payload.
pub fn parse_sequence_header(payload: &[u8]) -> Result<SequenceHeader> {
    let mut br = BitReader::new(payload);
    let mut seq = SequenceHeader::default();
    seq.id = br.get_vlc()? as u8;
    seq.profile = br.get_bits(5)? as u8;
    seq.reduced_still_picture_header = br.get_bit()? != 0;
    seq.level = br.get_bits(5)? as u8;
    if seq.level >= 4 && !seq.reduced_still_picture_header {
        seq.tier = br.get_bit()? as u8;
    }
    seq.layout = read_layout(br.get_vlc()?)?;
    seq.ss_hor = match seq.layout {
        PixelLayout::I400 | PixelLayout::I420 | PixelLayout::I422 => 1,
        PixelLayout::I444 => 0,
    };
    seq.ss_ver = match seq.layout {
        PixelLayout::I400 | PixelLayout::I420 => 1,
        PixelLayout::I422 | PixelLayout::I444 => 0,
    };
    seq.hbd = match br.get_vlc()? {
        1 => 0,
        0 => 1,
        2 => 2,
        other => {
            return Err(Av2Error::BitstreamError(format!(
                "invalid bit_depth_idc {other}"
            )));
        }
    };
    if !seq.reduced_still_picture_header {
        seq.lcr_id = br.get_bits(3)? as u8;
        seq.still_picture = br.get_bit()? != 0;
        seq.max_tlayer_id = br.get_bits(2)? as u8;
        seq.max_mlayer_id = br.get_bits(3)? as u8;
        seq.monotonic = br.get_bit()? != 0;
    }
    seq.width_n_bits = (br.get_bits(4)? + 1) as u8;
    seq.height_n_bits = (br.get_bits(4)? + 1) as u8;
    seq.max_width = br.get_bits(seq.width_n_bits as u32)? as i32 + 1;
    seq.max_height = br.get_bits(seq.height_n_bits as u32)? as i32 + 1;
    let mut crop = CropInfo::default();
    crop.enabled = br.get_bit()? != 0;
    if crop.enabled {
        crop.left = br.get_vlc()?;
        crop.right = br.get_vlc()?;
        crop.top = br.get_vlc()?;
        crop.bottom = br.get_vlc()?;
    }
    seq.crop = crop;
    if !seq.reduced_still_picture_header {
        let delay_present = br.get_bit()?;
        let model_info_present = br.get_bit()?;
        if delay_present != 0 || model_info_present != 0 {
            return Err(Av2Error::Unsupported(
                "sequence-level display delay / decoder model info".to_string(),
            ));
        }
    }
    if seq.max_tlayer_id > 0 {
        seq.tlayer_dependency_present = br.get_bit()? != 0;
        if seq.tlayer_dependency_present {
            for n in 1..seq.max_tlayer_id as usize {
                seq.tlayer_dependencies[n] = br.get_bits(n as u32)? as u8;
            }
        }
    }
    if seq.max_mlayer_id > 0 {
        seq.mlayer_dependency_present = br.get_bit()? != 0;
        if seq.mlayer_dependency_present {
            for n in 1..seq.max_mlayer_id as usize {
                seq.mlayer_dependencies[n] = br.get_bits(n as u32)? as u8;
            }
        }
    }

    seq.sb128 = if br.get_bit()? != 0 {
        2
    } else if br.get_bit()? != 0 {
        1
    } else {
        0
    };
    if seq.layout != PixelLayout::I400 {
        seq.sdp = br.get_bit()? != 0;
        if seq.sdp && !seq.reduced_still_picture_header {
            seq.ext_sdp = br.get_bit()? != 0;
        }
    }
    seq.ext_partitions = br.get_bit()? != 0;
    if seq.ext_partitions {
        seq.uneven_4way_partitions = br.get_bit()? != 0;
    }
    seq.max_pb_aspect_ratio_log2 = if br.get_bit()? != 0 {
        1 + br.get_bit()? as u8
    } else {
        3
    };

    // sequence_segment_config()
    seq.segmentation.ext = br.get_bit()? != 0;
    seq.segmentation.info_present = br.get_bit()? != 0;
    if seq.segmentation.info_present {
        seq.segmentation.adaptive = br.get_bit()? != 0;
        seq.segmentation.d = read_seg_info(&mut br, 8usize << (seq.segmentation.ext as usize))?;
    }

    // sequence_intra_config()
    seq.intra_dip = br.get_bit()? != 0;
    seq.intra_edge_filter = br.get_bit()? != 0;
    seq.mrls = br.get_bit()? != 0;
    seq.cfl = br.get_bit()? != 0;
    if seq.layout != PixelLayout::I400 {
        seq.cfl_ds_filter_index = br.get_bits(2)? as u8;
    }
    seq.mhccp = br.get_bit()? != 0;
    seq.ibp = br.get_bit()? != 0;

    // sequence_inter_config()
    seq.motion_modes = 1;
    if !seq.reduced_still_picture_header {
        for shift in [1, 2, 3, 4] {
            seq.motion_modes |= (br.get_bit()? as u8) << shift;
        }
        if (seq.motion_modes & !1) != 0 {
            seq.frame_motion_modes_present = br.get_bit()? != 0;
        }
        if (seq.motion_modes & (1 << 3)) != 0 {
            seq.six_param_warp_delta = br.get_bit()? != 0;
        }
        seq.masked_compound = br.get_bit()? != 0;
        seq.ref_frame_mvs = br.get_bit()? != 0;
        if seq.ref_frame_mvs {
            seq.reduced_ref_frame_mvs_mode = br.get_bit()? as u8;
        }
        seq.order_hint_n_bits = (br.get_bits(4)? + 1) as u8;
    } else {
        // `sequence_inter_config()`: `OrderHintBits = 0` when
        // `single_picture_header_flag` is set (AV2 Spec v1.0.0 §5.4.9).
        seq.order_hint_n_bits = 0;
    }
    seq.refmv_bank = br.get_bit()? != 0;
    if br.get_bit()? != 0 {
        seq.drl_reorder = 0;
    } else {
        seq.drl_reorder = if br.get_bit()? != 0 { 1 } else { 2 };
    }
    if !seq.reduced_still_picture_header {
        seq.explicit_ref_frame_map = br.get_bit()? != 0;
        seq.ref_frames = if br.get_bit()? != 0 {
            (br.get_bits(4)? + 1) as u8
        } else {
            8
        };
        seq.ref_frames_log2 = ulog2(seq.ref_frames as u32) as u8;
        seq.number_of_bits_for_lt_frame_id = br.get_bits(3)? as u8;
        seq.def_max_drl_bits = (br.get_uniform(5)? + 1) as u8;
        seq.allow_frame_max_drl_bits = br.get_bit()? != 0;
    }
    seq.def_max_bvp_drl_bits = (br.get_uniform(3)? + 1) as u8;
    seq.allow_max_bvp_drl_bits = br.get_bit()? != 0;
    if !seq.reduced_still_picture_header {
        seq.num_same_ref_comp = br.get_bits(2)? as u8;
        if br.get_bit()? != 0 {
            seq.tip_mode = 1 + br.get_bit()? as u8;
            seq.tip_hole_fill = br.get_bit()? != 0;
        }
        seq.mv_traj = br.get_bit()? != 0;
    }
    seq.bawp = br.get_bit()? != 0;
    if !seq.reduced_still_picture_header {
        seq.cwp = br.get_bit()? != 0;
        seq.imp_msk_bld = br.get_bit()? != 0;
        seq.db_sub_pu = br.get_bit()? != 0;
        if seq.tip_mode == 1 && seq.db_sub_pu {
            seq.tip_explicit_qp = br.get_bit()? != 0;
        }
        seq.opfl_refine = br.get_bits(2)? as u8;
        seq.refine_mv = br.get_bit()? != 0;
        if seq.tip_mode != 0 && (seq.opfl_refine != 0 || seq.refine_mv) {
            seq.tip_refine_mv = br.get_bit()? != 0;
        }
        seq.bru = br.get_bit()? != 0;
        seq.adaptive_mvd = br.get_bit()? != 0;
        seq.mvd_sign_derive = br.get_bit()? != 0;
        seq.flex_mvres = br.get_bit()? != 0;
        seq.global_motion = br.get_bit()? != 0;
        seq.short_refresh_frame_flags = br.get_bit()? != 0;
    }

    // sequence_scc_config()
    if seq.reduced_still_picture_header {
        // `single_picture_header_flag`: both values are `SELECT_*`, i.e. coded
        // per frame instead of in the sequence header (AV2 Spec v1.0.0 §5.4.7).
        seq.screen_content_tools = AdaptiveBoolean::Adaptive;
        seq.force_integer_mv = AdaptiveBoolean::Adaptive;
    } else {
        seq.screen_content_tools = read_adaptive_boolean(&mut br)?;
        if seq.screen_content_tools != AdaptiveBoolean::Off {
            seq.force_integer_mv = read_adaptive_boolean(&mut br)?;
        }
    }

    // sequence_transform_quant_entropy_config()
    seq.fsc = br.get_bit()? != 0;
    if !seq.fsc {
        seq.idtx_intra = br.get_bit()? != 0;
    }
    seq.ist = [br.get_bit()? != 0, br.get_bit()? != 0];
    if seq.layout != PixelLayout::I400 {
        seq.chroma_dctonly = br.get_bit()? != 0;
    }
    if !seq.reduced_still_picture_header {
        seq.inter_ddt = br.get_bit()? != 0;
    }
    seq.reduced_tx_part_set = br.get_bit()? != 0;
    if seq.layout != PixelLayout::I400 {
        seq.cctx = br.get_bit()? != 0;
    }
    if br.get_bit()? != 0 {
        if seq.reduced_still_picture_header {
            seq.tcq = AdaptiveBoolean::On;
        } else {
            seq.tcq = if br.get_bit()? != 0 {
                AdaptiveBoolean::Adaptive
            } else {
                AdaptiveBoolean::On
            };
        }
    }
    if seq.tcq != AdaptiveBoolean::On {
        seq.parity_hiding = br.get_bit()? != 0;
    }
    if !seq.reduced_still_picture_header {
        seq.avg_cdf = br.get_bit()? != 0;
        if seq.avg_cdf {
            seq.avg_cdf_type = br.get_bit()? as u8;
        }
    }
    if seq.layout != PixelLayout::I400 {
        seq.separate_uv_delta_q = br.get_bit()? != 0;
    }
    seq.equal_ac_dc_q = br.get_bit()? != 0;
    if !seq.equal_ac_dc_q {
        seq.base_ydc_dq = (br.get_bits(5)? as i32 - 23) as i8;
        seq.ydc_dq_enabled = br.get_bit()? != 0;
    }
    if seq.layout != PixelLayout::I400 {
        if !seq.equal_ac_dc_q {
            seq.base_uvdc_dq = (br.get_bits(5)? as i32 - 23) as u8;
            seq.uvdc_dq_enabled = br.get_bit()? != 0;
        }
        seq.base_uvac_dq = (br.get_bits(5)? as i32 - 23) as u8;
        seq.uvac_dq_enabled = br.get_bit()? != 0;
    }

    // sequence_filter_config()
    seq.disable_loopfilters_across_tiles = br.get_bit()? != 0;
    seq.cdef = br.get_bit()? != 0;
    seq.gdf = br.get_bit()? != 0;
    if seq.gdf && seq.sb128 == 0 {
        seq.gdf_unit_matches_sbsz = br.get_bit()? != 0;
    }
    seq.restoration = br.get_bit()? != 0;
    if seq.restoration {
        let b0 = br.get_bit()? as u8;
        let b1 = br.get_bit()? as u8;
        seq.rst_disable_mask[0] = b0 | (b1 << 1);
        let lr_tools_uv_present = br.get_bit()? != 0;
        if lr_tools_uv_present {
            seq.rst_disable_mask[1] = ((br.get_bit()? as u8) << 1) | 1;
        } else {
            seq.rst_disable_mask[1] = seq.rst_disable_mask[0] | 1;
        }
    }
    seq.ccso = br.get_bit()? != 0;
    if seq.ccso {
        seq.ccso_unit_matches_sbsz = br.get_bit()? != 0;
    }
    if !seq.reduced_still_picture_header {
        seq.cdef_on_skiptx = if br.get_bit()? != 0 {
            AdaptiveBoolean::On
        } else if br.get_bit()? != 0 {
            AdaptiveBoolean::Off
        } else {
            AdaptiveBoolean::Adaptive
        };
    } else {
        // `single_picture_header_flag`: `CdefOnSkipTxfm` is inferred to be
        // `CDEF_ON_SKIP_TXFM_ADAPTIVE` (§5.18.7.10, §5.4.9), so the frame
        // header still carries `cdef_on_skip_txfm_frame_enable`.
        seq.cdef_on_skiptx = AdaptiveBoolean::Adaptive;
    }
    seq.df_par_bits = br.get_bits(2)? as u8 + 2;

    // sequence_tile_config()
    if br.get_bit()? == 0 {
        seq.tiling.present = AdaptiveBoolean::Off;
    } else {
        seq.tiling.present = if br.get_bit()? != 0 {
            AdaptiveBoolean::Adaptive
        } else {
            AdaptiveBoolean::On
        };
        read_tile_info(
            &mut br,
            &mut seq.tiling.t,
            seq.sb128,
            seq.max_width,
            seq.max_height,
            seq.level,
            seq.tier,
        )?;
        finish_tile_grid(
            &mut seq.tiling.t,
            seq.sb128,
            seq.sb128,
            seq.max_width,
            seq.max_height,
            seq.level,
            seq.tier,
        );
    }
    seq.film_grain_present = br.get_bit()? != 0;
    if br.get_bit()? != 0 {
        return Err(Av2Error::Unsupported(
            "OBU extension headers in sequence_header_obu()".to_string(),
        ));
    }
    br.trailing_bits()?;
    Ok(seq)
}

/// Parses a complete tile-group OBU payload without reference metadata.
///
/// This is the convenience form used by tests and callers that do not track
/// reference slots; it does not derive `DerivedPrimaryRefFrame`. The decoder
/// uses [`parse_tile_group_with_refs`].
pub fn parse_tile_group<'a>(
    seq: &SequenceHeader,
    obu: &Obu<'a>,
) -> Result<(FrameHeader, Vec<&'a [u8]>)> {
    parse_tile_group_with_refs(seq, obu, &[RefFrameInfo::default(); 16])
}

/// Parses a complete tile-group OBU payload with reference-slot metadata.
///
/// `ref_info[slot]` is used to derive `DerivedPrimaryRefFrame` when the frame
/// header does not signal a primary reference (AV2 Spec v1.0.0 §5.18.1 and
/// §7.8). The segmentation syntax depends on that derived value.
pub fn parse_tile_group_with_refs<'a>(
    seq: &SequenceHeader,
    obu: &Obu<'a>,
    ref_info: &[RefFrameInfo; 16],
) -> Result<(FrameHeader, Vec<&'a [u8]>)> {
    let mut br = BitReader::new(obu.payload);
    let is_first = br.get_bit()?;
    if is_first == 0 {
        return Err(Av2Error::Unsupported(
            "tile groups that are not the first group of a frame".to_string(),
        ));
    }
    let mut fh = parse_frame_header(seq, &mut br, obu.obu_type, ref_info)?;
    fh.tlayer_id = obu.tlayer_id;
    let n_tiles = (fh.tiling.t.cols.max(1) as usize) * (fh.tiling.t.rows.max(1) as usize);
    if n_tiles > 1 {
        let flag = br.get_bit()?;
        if flag != 0 {
            return Err(Av2Error::Unsupported("partial tile groups".to_string()));
        }
    }
    br.byte_align();
    let mut pos = br.bit_pos() / 8;
    let payload = obu.payload;
    if pos > payload.len() {
        return Err(Av2Error::BitstreamError(
            "tile group header exceeds the payload".to_string(),
        ));
    }
    let n_bytes = fh.tiling.n_bytes.max(1) as usize;
    let mut tiles = Vec::with_capacity(n_tiles);
    for idx in 0..n_tiles {
        if idx + 1 < n_tiles {
            if pos + n_bytes > payload.len() {
                return Err(Av2Error::BitstreamError(
                    "truncated tile size field".to_string(),
                ));
            }
            let mut size = 0usize;
            for k in 0..n_bytes {
                size |= (payload[pos + k] as usize) << (8 * k);
            }
            pos += n_bytes;
            size += 1;
            if pos + size > payload.len() {
                return Err(Av2Error::BitstreamError(
                    "tile payload exceeds the tile group".to_string(),
                ));
            }
            tiles.push(&payload[pos..pos + size]);
            pos += size;
        } else {
            tiles.push(&payload[pos..]);
        }
    }
    Ok((fh, tiles))
}

/// Derives `DerivedPrimaryRefFrame` when the frame header does not signal one.
///
/// AV2 Spec v1.0.0 §7.8 and dav2d `derive_pri_sec_ref`: choose the two eligible
/// references (non-key/non-intra) with the smallest `|qidx - base_q_idx|`, then
/// the smallest POC distance; the first of the two is the primary reference.
fn derive_primary_ref(seq: &SequenceHeader, fh: &FrameHeader, ref_info: &[RefFrameInfo; 16]) -> u8 {
    let mut refs = [PRIMARY_REF_NONE, PRIMARY_REF_NONE];
    let mut best = 0usize;
    let mut best_qdiff = [0i32; 2];
    let mut best_pocdiff = [0i32; 2];
    let mut best_poc = [0i32; 2];
    let nbits = seq.order_hint_n_bits as i32;
    for i in 0..(fh.n_ref_frames as usize).min(fh.refidx.len()) {
        let slot = fh.refidx[i];
        if slot < 0 {
            continue;
        }
        let info = ref_info[slot as usize];
        if !info.present || info.is_intra {
            continue;
        }
        let qdiff = (info.qidx as i32 - fh.quant.yac as i32).abs();
        let pocdiff = get_poc_diff(nbits, fh.frame_offset as i32, info.poc).abs();
        for n in 0..2 {
            let m = if n == 0 { best } else { 1 - best };
            if refs[m] == PRIMARY_REF_NONE
                || qdiff < best_qdiff[m]
                || (qdiff == best_qdiff[m]
                    && (pocdiff < best_pocdiff[m]
                        || (pocdiff == best_pocdiff[m]
                            && get_poc_diff(nbits, best_poc[m], info.poc) < 0)))
            {
                let other = 1 - m;
                refs[other] = i as u8;
                best_pocdiff[other] = pocdiff;
                best_qdiff[other] = qdiff;
                best_poc[other] = info.poc;
                if n == 0 {
                    best = other;
                }
                break;
            }
        }
    }
    if best != 0 {
        refs.swap(0, 1);
    }
    refs[0]
}

/// Parses a `frame_header_info()` from the current reader position.
/// Reads one frame-level NS Wiener filter bank (`read_wienerns_filter(...,
/// readFrameFilters = 1)`, AV2 Spec v1.0.0 §5.20.10.6).
///
/// The `dav2d` parser (the acceptance oracle) does not consume
/// `wiener_ns_uv_sym` for frame filters, so this reader follows it: chroma
/// subsets are always fully coded.
fn read_wienerns_frame_filter(
    seq: &SequenceHeader,
    n_ref_frames: u8,
    refidx: &[i8],
    ref_info: &[RefFrameInfo; 16],
    br: &mut BitReader<'_>,
    plane: usize,
    ns: &mut NSWienerPlane,
) -> Result<()> {
    use crate::av2::wiener_tables::{
        WIENER_NS_TAPS_K, WIENER_NS_TAPS_MIN, WIENER_NS_TAPS_PRESENT_UV, WIENER_NS_TAPS_PRESENT_Y,
    };

    let n_coeffs = if plane == 0 { 16usize } else { 18 };
    let n_classes = ns.num_classes as usize;
    if n_classes == 0 || n_classes > 16 {
        return Err(Av2Error::BitstreamError(format!(
            "invalid Wiener NS class count {n_classes}"
        )));
    }
    let mask = seq.rst_disable_mask[usize::from(plane > 0)];
    let n_ref_filters = if mask & 1 != 0 {
        16usize
    } else {
        48usize.saturating_sub(n_classes)
    };

    // Collect reference frame filters in `RefLrWienerNs` order: the current
    // plane first, then the other chroma plane for chroma.
    let mut ref_filters = [[0i8; 18]; 48];
    let mut n_refs = 0usize;
    for r in 0..(n_ref_frames as usize).min(refidx.len()) {
        let slot = refidx[r];
        if slot < 0 {
            continue;
        }
        let info = ref_info.get(slot as usize).copied().unwrap_or_default();
        let planes: [usize; 2] = if plane == 0 {
            [0, 0]
        } else if plane == 1 {
            [1, 2]
        } else {
            [2, 1]
        };
        for (idx, &p2) in planes.iter().enumerate() {
            if idx == 1 && plane == 0 {
                break;
            }
            let rpd = &info.ns[p2];
            if rpd.frame_filters_on == 0 {
                continue;
            }
            let n = (n_ref_filters - n_refs).min(rpd.num_classes as usize);
            for c in 0..n {
                ref_filters[n_refs][..n_coeffs].copy_from_slice(&rpd.filter[c][..n_coeffs]);
                n_refs += 1;
            }
        }
    }

    let n_filters = if mask & 1 != 0 { 16usize } else { 64 };
    let mut grp_cnt = [0usize; 3];
    grp_cnt[0] = n_classes;
    grp_cnt[1] = n_refs;
    grp_cnt[2] = n_filters.saturating_sub(grp_cnt[0] + grp_cnt[1]);
    let mut filter_refs = [0usize; 64];
    let mut pred_grp = 2 - usize::from(grp_cnt[1] > 2);
    let nnz_grps = 1 + usize::from(grp_cnt[1] > 0) + usize::from(grp_cnt[2] > 0);
    let mut grp_ref_cnt = [0usize; 3];
    for n in 0..n_classes {
        let group = if nnz_grps == 1 || br.get_bit()? == 0 {
            pred_grp
        } else if nnz_grps == 2 {
            2 - usize::from(grp_cnt[2] == 0) - pred_grp
        } else if br.get_bit()? != 0 {
            2 - usize::from(pred_grp == 2)
        } else {
            usize::from(pred_grp == 0)
        };
        grp_ref_cnt[group] += 1;
        if grp_ref_cnt[group] + usize::from(group < pred_grp) > grp_ref_cnt[pred_grp] {
            pred_grp = group;
        }
        let base = grp_cnt[0] * usize::from(group > 0) + grp_cnt[1] * usize::from(group == 2);
        let range = if group != 0 { grp_cnt[group] } else { n + 1 };
        filter_refs[n] = base
            + if range == 1 {
                0
            } else {
                br.get_bits_subexp_u((range >> 1) as u32, range as u32, 4)? as usize
            };
    }

    let mut exact_match_mask = 0u32;
    for n in 0..n_classes {
        exact_match_mask |= br.get_bit()? << n;
    }

    for n in 0..n_classes {
        let r = filter_refs[n];
        let mut ref_filter = [0i8; 18];
        if r != 0 {
            if r < n_classes {
                ref_filter = ns.filter[r - 1];
            } else if r < n_classes + grp_cnt[1] {
                ref_filter = ref_filters[r - n_classes];
            } else {
                return Err(Av2Error::Unsupported(
                    "pretrained Wiener NS filter bank reference".to_string(),
                ));
            }
        }
        if exact_match_mask & (1u32 << n) != 0 {
            ns.filter[n][..n_coeffs].copy_from_slice(&ref_filter[..n_coeffs]);
            continue;
        }
        ns.filter[n] = [0i8; 18];
        let mut subset = 0usize;
        let n_subsets = if plane == 0 { 4 } else { 3 };
        for s in 0..n_subsets - 1 {
            if br.get_bit()? == 0 {
                break;
            }
            subset = s + 1;
        }
        let present: &[bool] = if plane == 0 {
            &WIENER_NS_TAPS_PRESENT_Y[subset]
        } else {
            &WIENER_NS_TAPS_PRESENT_UV[subset]
        };
        let min = &WIENER_NS_TAPS_MIN[usize::from(plane > 0)];
        let k = &WIENER_NS_TAPS_K[usize::from(plane > 0)];
        for j in 0..n_coeffs {
            if !present[j] {
                continue;
            }
            let low = min[j] as i32;
            let mx = 1u32 << k[j];
            let v =
                br.get_bits_subexp_u((ref_filter[j] as i32 - low) as u32, mx, k[j] as i32 - 3)?
                    as i32
                    + low;
            ns.filter[n][j] = v as i8;
        }
    }
    Ok(())
}

pub fn parse_frame_header(
    seq: &SequenceHeader,
    br: &mut BitReader<'_>,
    obu_type: ObuType,
    ref_info: &[RefFrameInfo; 16],
) -> Result<FrameHeader> {
    let mut fh = FrameHeader::default();
    fh.primary_ref_frame = PRIMARY_REF_NONE;
    fh.secondary_ref_frame = PRIMARY_REF_NONE;

    fh.id = br.get_vlc()? as u8;
    if fh.id == 0 {
        let seq_id = br.get_vlc()? as u8;
        if seq_id != seq.id {
            return Err(Av2Error::BitstreamError(format!(
                "frame header references sequence header {seq_id}, active is {}",
                seq.id
            )));
        }
    }

    match obu_type {
        ObuType::ClosedLoopKf | ObuType::OpenLoopKf => fh.frame_type = FrameType::Key,
        ObuType::TileGrp => {
            fh.frame_type = if br.get_bit()? != 0 {
                FrameType::Inter
            } else {
                FrameType::Intra
            };
        }
        other => {
            return Err(Av2Error::Unsupported(format!("frame OBU type {other:?}")));
        }
    }

    if !seq.reduced_still_picture_header {
        if fh.frame_type == FrameType::Key && seq.number_of_bits_for_lt_frame_id > 0 {
            let _ltr_id = br.get_bits(seq.number_of_bits_for_lt_frame_id as u32)?;
        }
        if obu_type != ObuType::OpenLoopKf {
            fh.show_immediate = br.get_bit()? as u8;
        }
        if fh.show_immediate == 0 && !seq.monotonic {
            fh.show_implicit = br.get_bit()? as u8;
        }
    } else {
        fh.show_immediate = 1;
    }

    if fh.frame_type == FrameType::Switch {
        fh.frame_size_override = 1;
    } else if seq.reduced_still_picture_header {
        // `single_picture_header_flag`: `frame_size_override_flag = 0` and no
        // bit is coded (AV2 Spec v1.0.0 §5.18.3.1).
        fh.frame_size_override = 0;
    } else {
        fh.frame_size_override = br.get_bit()? as u8;
    }
    fh.frame_offset = br.get_bits(seq.order_hint_n_bits as u32)? as u8;
    if fh.frame_type == FrameType::Inter {
        fh.primary_ref_signaled = br.get_bit()? as u8;
        if obu_type != ObuType::LeadingTip && obu_type != ObuType::Tip {
            fh.no_cross_frame_context = br.get_bit()? as u8;
        }
        if fh.primary_ref_signaled != 0 {
            fh.primary_ref_frame = br.get_bits(3)? as u8;
        }
    }

    // refresh_frame_flags
    if !(obu_type == ObuType::ClosedLoopKf && seq.max_mlayer_id == 0) {
        if obu_type == ObuType::OpenLoopKf || seq.max_mlayer_id > 0 {
            if seq.short_refresh_frame_flags {
                let slot = br.get_bits(seq.ref_frames_log2 as u32)?;
                fh.refresh_frame_flags = 1u16 << slot;
            } else {
                fh.refresh_frame_flags = br.get_bits(seq.ref_frames as u32)? as u16;
            }
        } else if fh.frame_type != FrameType::Switch && seq.short_refresh_frame_flags {
            if br.get_bit()? != 0 {
                let slot = br.get_bits(seq.ref_frames_log2 as u32)?;
                fh.refresh_frame_flags = 1u16 << slot;
            }
        } else {
            fh.refresh_frame_flags = br.get_bits(seq.ref_frames as u32)? as u16;
        }
    } else {
        fh.refresh_frame_flags = ((1u32 << seq.ref_frames) - 1) as u16;
    }

    // Reference frame list
    if fh.is_inter_or_switch() {
        if fh.frame_type != FrameType::Switch && seq.explicit_ref_frame_map {
            let flag = br.get_bit()?;
            if flag == 0 {
                return Err(Av2Error::Unsupported(
                    "implicit reference frame map".to_string(),
                ));
            }
        }
        if fh.frame_type == FrameType::Switch || seq.explicit_ref_frame_map {
            fh.n_ref_frames = br.get_bits(3)? as u8;
            if fh.n_ref_frames == 0
                || fh.n_ref_frames as usize > crate::av2::headers::REFS_PER_FRAME
            {
                return Err(Av2Error::BitstreamError(
                    "invalid inter reference count".to_string(),
                ));
            }
            for i in 0..fh.n_ref_frames as usize {
                let r = br.get_bits(seq.ref_frames_log2 as u32)? as i8;
                if r < 0 || r as usize >= seq.ref_frames as usize {
                    return Err(Av2Error::BitstreamError(
                        "reference slot out of range".to_string(),
                    ));
                }
                fh.refidx[i] = r;
            }
        }
    }

    // Frame size
    if fh.frame_size_override != 0 {
        if fh.is_inter_or_switch() {
            for _ in 0..fh.n_ref_frames as usize {
                if br.get_bit()? != 0 {
                    return Err(Av2Error::Unsupported(
                        "reference-derived frame size".to_string(),
                    ));
                }
            }
        }
        fh.width = br.get_bits(seq.width_n_bits as u32)? as i32 + 1;
        fh.height = br.get_bits(seq.height_n_bits as u32)? as i32 + 1;
    } else {
        fh.width = seq.max_width;
        fh.height = seq.max_height;
    }

    if fh.is_inter_or_switch() {
        if seq.ref_frame_mvs {
            fh.use_ref_frame_mvs = br.get_bit()? as u8;
            if fh.use_ref_frame_mvs != 0 && fh.n_ref_frames > 1 && seq.sb128 != 0 {
                fh.tmvp_sample_step = 1 + br.get_bit()? as u8;
            }
        }
        if seq.opfl_refine == 3 {
            if br.get_bit()? != 0 {
                fh.opfl_refine_type = 1;
            } else {
                fh.opfl_refine_type = if br.get_bit()? != 0 { 2 } else { 0 };
            }
        }
    }

    // screen_content_params() & intrabc_params()
    if seq.screen_content_tools == AdaptiveBoolean::Adaptive {
        fh.allow_screen_content_tools = br.get_bit()? as u8;
    }
    let allow_sct = match seq.screen_content_tools {
        AdaptiveBoolean::On => true,
        AdaptiveBoolean::Off => false,
        AdaptiveBoolean::Adaptive => fh.allow_screen_content_tools != 0,
    };
    if allow_sct && seq.force_integer_mv == AdaptiveBoolean::Adaptive {
        fh.force_integer_mv = br.get_bit()? as u8;
    }
    let force_int_mv = if allow_sct {
        match seq.force_integer_mv {
            AdaptiveBoolean::On => true,
            AdaptiveBoolean::Off => false,
            AdaptiveBoolean::Adaptive => fh.force_integer_mv != 0,
        }
    } else {
        false
    };
    fh.allow_intrabc = br.get_bit()? as u8;
    if fh.allow_intrabc != 0 {
        if fh.is_key_or_intra() {
            fh.allow_global_intrabc = br.get_bit()? as u8;
            if fh.allow_global_intrabc != 0 {
                fh.allow_local_intrabc = br.get_bit()? as u8;
            }
        }
        if seq.allow_max_bvp_drl_bits {
            let change = br.get_bit()?;
            if change != 0 {
                return Err(Av2Error::Unsupported(
                    "frame-level BVP DRL bit count change".to_string(),
                ));
            }
        }
        fh.max_bvp_drl_bits = seq.def_max_bvp_drl_bits;
    }

    if fh.is_inter_or_switch() {
        if seq.allow_frame_max_drl_bits {
            let change = br.get_bit()?;
            if change != 0 {
                return Err(Av2Error::Unsupported(
                    "frame-level DRL bit count change".to_string(),
                ));
            }
        }
        fh.max_drl_bits = seq.def_max_drl_bits;
        if !force_int_mv {
            if br.get_bit()? != 0 {
                fh.mv_precision = 2;
            } else {
                fh.mv_precision = if br.get_bit()? != 0 { 3 } else { 2 };
            }
        }
        if br.get_bit()? != 0 {
            fh.subpel_filter_mode = FilterMode::Switchable;
        } else {
            fh.subpel_filter_mode = match br.get_bits(2)? {
                0 => FilterMode::Regular8Tap,
                1 => FilterMode::Smooth8Tap,
                2 => FilterMode::Sharp8Tap,
                _ => FilterMode::Bilinear,
            };
        }
        fh.motion_modes = 1;
        if seq.frame_motion_modes_present {
            for shift in [1, 2, 3, 4] {
                if (seq.motion_modes & (1 << shift)) != 0 {
                    fh.motion_modes |= (br.get_bit()? as u8) << shift;
                }
            }
        }
    }

    fh.disable_cdf_update = br.get_bit()? as u8;

    // tile_info()
    fh.sb128 = if fh.is_inter_or_switch() {
        seq.sb128
    } else {
        (seq.sb128 != 0) as u8
    };
    if seq.tiling.present == AdaptiveBoolean::Off {
        read_tile_info(
            br,
            &mut fh.tiling.t,
            fh.sb128,
            fh.width,
            fh.height,
            seq.level,
            seq.tier,
        )?;
        finish_tile_grid(
            &mut fh.tiling.t,
            fh.sb128,
            seq.sb128,
            fh.width,
            fh.height,
            seq.level,
            seq.tier,
        );
    } else if seq.tiling.present == AdaptiveBoolean::Adaptive {
        if br.get_bit()? == 0 {
            return Err(Av2Error::Unsupported(
                "explicit frame tile layout".to_string(),
            ));
        }
        fh.tiling.t = seq.tiling.t.clone();
    } else {
        fh.tiling.t = seq.tiling.t.clone();
    }
    if fh.tiling.t.log2_cols != 0 || fh.tiling.t.log2_rows != 0 {
        if !seq.avg_cdf || seq.avg_cdf_type == 0 {
            let n = (fh.tiling.t.log2_cols + fh.tiling.t.log2_rows) as u32;
            fh.tiling.update = br.get_bits(n)? as u16;
        }
        fh.tiling.n_bytes = br.get_bits(2)? as u8 + 1;
    } else {
        fh.tiling.n_bytes = 1;
    }

    // quantization_params()
    let q_bits = 8 + (seq.hbd != 0) as u32;
    fh.quant.yac = br.get_bits(q_bits)? as u16;
    if seq.ydc_dq_enabled && br.get_bit()? != 0 {
        fh.quant.ydc_delta = br.get_sbits(7)? as i8;
    }
    if seq.layout != PixelLayout::I400 && (seq.uvdc_dq_enabled || seq.uvac_dq_enabled) {
        let diff_uv = seq.separate_uv_delta_q && br.get_bit()? != 0;
        let mut udc = 0i8;
        let mut uac = 0i8;
        if seq.uvdc_dq_enabled && br.get_bit()? != 0 {
            udc = br.get_sbits(7)? as i8;
        }
        if seq.uvac_dq_enabled && br.get_bit()? != 0 {
            uac = br.get_sbits(7)? as i8;
        }
        let mut vdc = udc;
        let mut vac = uac;
        if diff_uv {
            if seq.uvdc_dq_enabled && br.get_bit()? != 0 {
                vdc = br.get_sbits(7)? as i8;
            }
            if seq.uvac_dq_enabled && br.get_bit()? != 0 {
                vac = br.get_sbits(7)? as i8;
            }
        }
        fh.quant.udc_delta = udc;
        fh.quant.uac_delta = uac;
        fh.quant.vdc_delta = vdc;
        fh.quant.vac_delta = vac;
    }

    // DerivedPrimaryRefFrame: when the primary reference is not signalled, the
    // value is derived from the reference slot metadata. Segmentation syntax
    // reads `segmentation_update_map` only for a non-NONE primary reference.
    if fh.is_inter_or_switch() && fh.primary_ref_signaled == 0 {
        fh.primary_ref_frame = derive_primary_ref(seq, &fh, ref_info);
    }

    // segmentation_params()
    fh.segmentation.enabled = br.get_bit()? as u8;
    if fh.segmentation.enabled != 0 {
        if seq.segmentation.info_present && seq.segmentation.adaptive {
            if br.get_bit()? != 0 {
                fh.segmentation.d = seq.segmentation.d;
            } else {
                fh.segmentation.d = read_seg_info(br, 8usize << (seq.segmentation.ext as usize))?;
            }
        } else if !seq.segmentation.info_present {
            fh.segmentation.d = read_seg_info(br, 8usize << (seq.segmentation.ext as usize))?;
        } else {
            fh.segmentation.d = seq.segmentation.d;
        }
        // AV2 Spec v1.0.0 §5.18.7.1: with no primary reference the segment map
        // is always updated; `update_map` is not coded. dav2d sets the same
        // default in `parse_frame_hdr`.
        if fh.primary_ref_frame == PRIMARY_REF_NONE {
            fh.segmentation.update_map = 1;
        } else {
            fh.segmentation.update_map = br.get_bit()? as u8;
            if fh.segmentation.update_map != 0 && fh.frame_type != FrameType::Switch {
                fh.segmentation.temporal = br.get_bit()? as u8;
            }
        }
        fh.segmentation.preskip = fh.segmentation.d.preskip() as u8;
        fh.segmentation.last_active_segid = fh.segmentation.d.last_active_segid() as i8;
    }

    // setup_qm_params()
    fh.quant.qm.enabled = br.get_bit()? as u8;
    if fh.quant.qm.enabled != 0 {
        let qm_num = if fh.segmentation.enabled != 0 {
            let num = br.get_bits(2)? as u8 + 1;
            fh.quant.qm.num = num;
            num as usize
        } else {
            1usize
        };
        for i in 0..qm_num {
            fh.quant.qm.y[i] = br.get_bits(4)? as u8;
            if seq.layout != PixelLayout::I400 {
                if br.get_bit()? != 0 {
                    fh.quant.qm.u[i] = fh.quant.qm.y[i];
                    fh.quant.qm.v[i] = fh.quant.qm.y[i];
                } else {
                    fh.quant.qm.u[i] = br.get_bits(4)? as u8;
                    fh.quant.qm.v[i] = if seq.separate_uv_delta_q {
                        br.get_bits(4)? as u8
                    } else {
                        fh.quant.qm.u[i]
                    };
                }
            } else {
                fh.quant.qm.u[i] = fh.quant.qm.y[i];
                fh.quant.qm.v[i] = fh.quant.qm.y[i];
            }
        }
        return Err(Av2Error::Unsupported("quantization matrices".to_string()));
    }

    // delta_q_params()
    if fh.quant.yac != 0 {
        fh.delta.q.present = br.get_bit()? as u8;
        if fh.delta.q.present != 0 {
            fh.delta.q.res_log2 = br.get_bits(2)? as u8;
        }
    }

    // LosslessArray derivation (AV2 Spec v1.0.0 §5.18.1 / §7.14.4). The
    // segment quantizer uses `get_qindex(1, segment)`, so ALT_Q is applied to
    // `base_q_idx` and the per-SB `CurrentQIndex` is ignored here.
    let max_q = crate::av2::headers::max_q_bits(if seq.hbd == 2 {
        12
    } else if seq.hbd == 1 {
        10
    } else {
        8
    });
    let delta_lossless = fh.delta.q.present == 0
        && fh.quant.ydc_delta <= 0
        && fh.quant.udc_delta <= 0
        && fh.quant.uac_delta <= 0
        && fh.quant.vdc_delta <= 0
        && fh.quant.vac_delta <= 0;
    let n_seg = if fh.segmentation.enabled != 0 {
        MAX_SEGMENTS
    } else {
        1
    };
    let mut has_lossless_segment = false;
    let mut coded_lossless = true;
    for i in 0..n_seg {
        let q = if fh.segmentation.enabled != 0 {
            iclip(
                fh.quant.yac as i32 + fh.segmentation.d.delta_q[i] as i32,
                0,
                max_q,
            )
        } else {
            fh.quant.yac as i32
        };
        let lossless = q == 0 && delta_lossless;
        fh.segmentation.qidx[i] = q.clamp(0, max_q) as u8;
        fh.quant.lossless[i] = lossless as u8;
        fh.segmentation.lossless[i] = lossless as u8;
        has_lossless_segment |= lossless;
        coded_lossless &= lossless;
    }
    for i in n_seg..MAX_SEGMENTS {
        fh.quant.lossless[i] = 0;
        fh.segmentation.lossless[i] = 0;
    }
    fh.all_lossless = coded_lossless as u8;
    fh.any_lossless = has_lossless_segment as u8;

    if !coded_lossless {
        if seq.tcq == AdaptiveBoolean::Adaptive {
            fh.tcq = br.get_bit()? as u8;
        } else if seq.tcq == AdaptiveBoolean::On {
            fh.tcq = 1;
        }
        let allow_tcq = seq.tcq == AdaptiveBoolean::On || fh.tcq != 0;
        if !allow_tcq && seq.parity_hiding {
            fh.parity_hiding = br.get_bit()? as u8;
        }

        // deblocking_filter_params()
        if fh.frame_type == FrameType::Inter && seq.db_sub_pu {
            fh.deblock.sub_pu = br.get_bit()? as u8;
        }
        let apply_y0 = br.get_bit()? != 0;
        let apply_y1 = br.get_bit()? != 0;
        let mut apply_u = false;
        let mut apply_v = false;
        if seq.layout != PixelLayout::I400 && (apply_y0 || apply_y1) {
            apply_u = br.get_bit()? != 0;
            apply_v = br.get_bit()? != 0;
        }
        let df_bits = seq.df_par_bits.max(2) as u32;
        let df_off = 1i32 << (df_bits - 1);
        let mut dq_y = [0i8; 2];
        let mut dq_u = 0i8;
        let mut dq_v = 0i8;
        for idx in 0..4 {
            let apply = match idx {
                0 => apply_y0,
                1 => apply_y1,
                2 => apply_u,
                _ => apply_v,
            };
            if !apply {
                continue;
            }
            let default_dq = if idx == 1 { dq_y[0] } else { 0 };
            let dq = if br.get_bit()? != 0 {
                (br.get_bits(df_bits)? as i32 - df_off) as i8
            } else {
                default_dq
            };
            match idx {
                0 => dq_y[0] = dq,
                1 => dq_y[1] = dq,
                2 => dq_u = dq,
                _ => dq_v = dq,
            }
        }
        fh.deblock.level_y = [apply_y0 as u8, apply_y1 as u8];
        fh.deblock.level_u = apply_u as u8;
        fh.deblock.level_v = apply_v as u8;
        fh.deblock.delta_q_y = dq_y;
        fh.deblock.delta_q_u = dq_u;
        fh.deblock.delta_q_v = dq_v;

        // gdf_params()
        if seq.gdf {
            // GdfBlkSize is fixed before `gdf_per_block` is read: it is the
            // superblock width when `gdf_unit_matches_sb_size` is set, and
            // `Max(Block_Width[SbSize], GDF_MIN_SIZE)` otherwise, with the
            // 64x64 tile-alignment override (AV2 §5.18.7.9).
            fh.gdf.b64size = 1 << (!seq.gdf_unit_matches_sbsz as u8).max(fh.sb128);
            if !seq.gdf_unit_matches_sbsz && fh.sb128 == 0 {
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
            let gdf_blk_size = 64 * fh.gdf.b64size.max(1) as usize;

            let gdf_on = if seq.reduced_still_picture_header {
                true
            } else {
                br.get_bit()? != 0
            };
            if gdf_on {
                let mi_cols = 2 * ((fh.width.max(0) as usize + 7) >> 3);
                let mi_rows = 2 * ((fh.height.max(0) as usize + 7) >> 3);
                let per_block = mi_cols * 4 > gdf_blk_size
                    || mi_rows * 4 > gdf_blk_size
                    || (seq.disable_loopfilters_across_tiles
                        && (fh.tiling.t.rows > 1 || fh.tiling.t.cols > 1));
                let adaptive = if per_block { br.get_bit()? != 0 } else { false };
                fh.gdf.enabled = if adaptive {
                    AdaptiveBoolean::Adaptive
                } else {
                    AdaptiveBoolean::On
                };
                let qp_base = if fh.is_key_or_intra() { 85 } else { 110 };
                let qp_diff = fh.quant.yac as i32 - qp_base - 48 * seq.hbd as i32;
                let qp_idx_offset = br.get_bits(2)? as u8;
                fh.gdf.qp_idx =
                    (iclip((qp_diff - 37) / 25, 0, 2) + qp_idx_offset as i32).clamp(0, 5) as u8;
                fh.gdf.scale = br.get_bits(2)? as u8 + 1;
            } else {
                fh.gdf.enabled = AdaptiveBoolean::Off;
            }
        }

        // cdef_params()
        if seq.cdef {
            let cdef_on = if seq.reduced_still_picture_header {
                true
            } else {
                br.get_bit()? != 0
            };
            if cdef_on {
                fh.cdef.enabled = 1;
                fh.cdef.damping = br.get_bits(2)? as u8 + 3;
                let n_str = br.get_bits(3)? as usize + 1;
                fh.cdef.n_strengths = n_str as u8;
                fh.cdef.on_skiptx = match seq.cdef_on_skiptx {
                    AdaptiveBoolean::On => 1,
                    AdaptiveBoolean::Off => 0,
                    AdaptiveBoolean::Adaptive => br.get_bit()? as u8,
                };
                for i in 0..n_str {
                    let y_pri = if br.get_bit()? != 0 {
                        0
                    } else {
                        br.get_bits(4)? as u8
                    };
                    let y_sec = br.get_bits(2)? as u8;
                    fh.cdef.y_strength[i] = (y_pri << 2) | y_sec.min(3);
                    if seq.layout != PixelLayout::I400 {
                        let uv_pri = if br.get_bit()? != 0 {
                            0
                        } else {
                            br.get_bits(4)? as u8
                        };
                        let uv_sec = br.get_bits(2)? as u8;
                        fh.cdef.uv_strength[i] = (uv_pri << 2) | uv_sec.min(3);
                    }
                }
            }
        }

        // lr_params() (AV2 §5.18.7.11)
        if seq.restoration && fh.all_lossless == 0 {
            let n_planes = if seq.layout == PixelLayout::I400 {
                1usize
            } else {
                3usize
            };
            for p in 0..n_planes {
                let mask = seq.rst_disable_mask[if p != 0 { 1 } else { 0 }];
                let rtype = if mask == 0 {
                    br.get_bits(2)? as u8
                } else if mask != 3 {
                    br.get_bit()? as u8 * (3 - mask)
                } else {
                    0
                };
                fh.restoration.planes[p].restoration_type = rtype;
                fh.restoration.planes[p].t = match rtype {
                    0 => RestorationType::None,
                    1 => RestorationType::PcWiener,
                    2 => RestorationType::NsWiener,
                    _ => RestorationType::Switchable,
                };
                if rtype >= RestorationType::NsWiener as u8 {
                    let frame_filters_on = br.get_bit()? as u8;
                    fh.restoration.planes[p].ns.frame_filters_on = frame_filters_on;
                    if frame_filters_on != 0 {
                        let num_ref_frames = if fh.frame_type == FrameType::Inter {
                            fh.n_ref_frames as usize
                        } else {
                            0
                        };
                        if num_ref_frames > 0 {
                            fh.restoration.planes[p].ns.temporal = br.get_bit()? as u8;
                        }
                        if fh.restoration.planes[p].ns.temporal != 0 {
                            let ref_idx = if num_ref_frames > 1 {
                                let n_bits = if num_ref_frames <= 2 {
                                    1u32
                                } else {
                                    (ulog2(num_ref_frames as u32 - 1) + 1) as u32
                                };
                                let idx = br.get_bits(n_bits)? as usize;
                                if idx >= num_ref_frames {
                                    return Err(Av2Error::BitstreamError(format!(
                                        "restoration reference index {idx} exceeds {num_ref_frames} references"
                                    )));
                                }
                                idx
                            } else {
                                0
                            };
                            fh.restoration.planes[p].ns.refidx = ref_idx as u8;
                            let slot = fh.refidx.get(ref_idx).copied().ok_or_else(|| {
                                Av2Error::BitstreamError(
                                    "restoration reference index is out of range".to_string(),
                                )
                            })?;
                            if slot < 0 {
                                return Err(Av2Error::BitstreamError(
                                    "restoration reference slot is invalid".to_string(),
                                ));
                            }
                            let info = ref_info.get(slot as usize).copied().unwrap_or_default();
                            let mut ref_plane = p;
                            if p > 0 && info.ns[p].frame_filters_on == 0 {
                                ref_plane = 3 - p;
                            }
                            let rpd = &info.ns[ref_plane];
                            if rpd.frame_filters_on == 0 {
                                return Err(Av2Error::BitstreamError(format!(
                                    "temporal restoration reference for plane {p} has no frame filters"
                                )));
                            }
                            fh.restoration.planes[p].ns.num_classes_idx = rpd.num_classes_idx;
                            fh.restoration.planes[p].ns.num_classes = rpd.num_classes;
                            fh.restoration.planes[p].ns.filter = rpd.filter;
                        } else {
                            let idx = br.get_bits(3)? as u8;
                            fh.restoration.planes[p].ns.num_classes_idx = idx;
                            fh.restoration.planes[p].ns.num_classes =
                                1 + idx + idx.saturating_sub(3) + 2 * idx.saturating_sub(5);
                        }
                    } else {
                        fh.restoration.planes[p].ns.num_classes_idx = 0;
                        fh.restoration.planes[p].ns.num_classes = 1;
                    }
                }
            }
            let sb_size_idx = seq.sb128;
            let mut unit_size_y = 9u8;
            if fh.restoration.planes[0].restoration_type != 0 {
                if br.get_bit()? != 0 {
                    unit_size_y -= 1;
                } else if sb_size_idx < 2 && br.get_bit()? == 0 {
                    let quarter_size = if sb_size_idx == 0 {
                        br.get_bit()? != 0
                    } else {
                        true
                    };
                    unit_size_y -= 2 + (!quarter_size as u8);
                }
            }
            fh.restoration.unit_size[0] = unit_size_y;

            let ss = (seq.layout != PixelLayout::I444) as u8;
            let mut unit_size_uv = 9u8 - ss;
            if n_planes > 1
                && (fh.restoration.planes[1].restoration_type != 0
                    || fh.restoration.planes[2].restoration_type != 0)
            {
                if br.get_bit()? != 0 {
                    unit_size_uv -= 1;
                } else if sb_size_idx < 2 && br.get_bit()? == 0 {
                    let quarter_size = if sb_size_idx == 0 {
                        br.get_bit()? != 0
                    } else {
                        true
                    };
                    unit_size_uv -= 2 + (!quarter_size as u8);
                }
            }
            fh.restoration.unit_size[1] = unit_size_uv;

            for p in 0..n_planes {
                let n_ref_frames = fh.n_ref_frames;
                let refidx = fh.refidx;
                let ns = &mut fh.restoration.planes[p].ns;
                if ns.frame_filters_on != 0 && ns.temporal == 0 {
                    read_wienerns_frame_filter(seq, n_ref_frames, &refidx, ref_info, br, p, ns)?;
                }
            }
        }

        // ccso_params() (AV2 §5.18.7.12)
        if seq.ccso {
            let ccso_on = if seq.reduced_still_picture_header {
                true
            } else {
                br.get_bit()? != 0
            };
            fh.ccso.enabled = ccso_on as u8;
            if ccso_on {
                let n_planes = if seq.layout == PixelLayout::I400 {
                    1
                } else {
                    3
                };
                for p in 0..n_planes {
                    fh.ccso.planes[p].enabled = br.get_bit()? as u8;
                    if fh.ccso.planes[p].enabled == 0 {
                        continue;
                    }
                    if fh.is_inter_or_switch() {
                        fh.ccso.planes[p].reuse = br.get_bit()? as u8;
                        fh.ccso.planes[p].sb_reuse = br.get_bit()? as u8;
                        if fh.ccso.planes[p].reuse != 0 || fh.ccso.planes[p].sb_reuse != 0 {
                            let n_bits = if fh.n_ref_frames <= 2 {
                                fh.n_ref_frames.saturating_sub(1) as u32
                            } else {
                                1 + ulog2(fh.n_ref_frames as u32 - 1) as u32
                            };
                            if n_bits > 0 {
                                fh.ccso.planes[p].refidx = br.get_bits(n_bits)? as u8;
                            }
                            if fh.ccso.planes[p].reuse != 0 {
                                return Err(Av2Error::Unsupported("CCSO reuse".to_string()));
                            }
                        }
                    }
                    if fh.ccso.planes[p].reuse == 0 {
                        fh.ccso.planes[p].bo_only = br.get_bit()? as u8;
                        let si = br.get_bits(2)? as u8;
                        fh.ccso.planes[p].scale_idx = si;
                        if fh.ccso.planes[p].bo_only != 0 {
                            fh.ccso.planes[p].max_band_log2 = br.get_bits(3)? as u8;
                        } else {
                            let qi = br.get_bits(2)? as u8;
                            fh.ccso.planes[p].quant_idx = qi;
                            let ext_filter = br.get_bits(3)? as u8;
                            if ext_filter == 7 {
                                return Err(Av2Error::BitstreamError(
                                    "ccso_ext_filter cannot be 7".to_string(),
                                ));
                            }
                            fh.ccso.planes[p].ext_filter_support = ext_filter;
                            if CCSO_QUANT_SZ[si as usize][qi as usize] != 0 {
                                fh.ccso.planes[p].edge_clf = br.get_bit()? as u8;
                            }
                            fh.ccso.planes[p].max_band_log2 = br.get_bits(2)? as u8;
                        }
                        let n_intervals = if fh.ccso.planes[p].bo_only != 0 {
                            1
                        } else {
                            3 - fh.ccso.planes[p].edge_clf as usize
                        };
                        let max_band = 1 << fh.ccso.planes[p].max_band_log2;
                        fh.ccso.planes[p].filter_off = [0u8; 64];
                        for n in 0..n_intervals {
                            for m in 0..n_intervals {
                                for o in 0..max_band {
                                    let mut off = 0u8;
                                    while off < 7 {
                                        if br.get_bit()? == 0 {
                                            break;
                                        }
                                        off += 1;
                                    }
                                    let byte_idx = n * 16 + m * 4 + (o >> 1);
                                    fh.ccso.planes[p].filter_off[byte_idx] |= off << (4 * (o & 1));
                                }
                            }
                        }
                    }
                }
            }
            fh.ccso.b64size = 1
                << (if seq.ccso_unit_matches_sbsz {
                    fh.sb128
                } else {
                    2
                });
            if !seq.ccso_unit_matches_sbsz && fh.sb128 < 2 {
                let mut mask = 0;
                for t in 0..fh.tiling.t.cols.saturating_sub(1) as usize {
                    mask |= fh.tiling.t.col_start_sb[t];
                }
                for t in 0..fh.tiling.t.rows.saturating_sub(1) as usize {
                    mask |= fh.tiling.t.row_start_sb[t];
                }
                if mask & 3 != 0 && fh.sb128 == 0 {
                    fh.ccso.b64size = 2 - (mask & 1) as u8;
                } else if mask & 1 != 0 {
                    fh.ccso.b64size = 2;
                }
            }
        }

        // read_tx_mode()
        fh.txfm_mode = if br.get_bit()? != 0 {
            TxfmMode::Switchable
        } else {
            TxfmMode::Largest
        };
    } else {
        fh.txfm_mode = TxfmMode::Only4x4;
    }

    if fh.is_inter_or_switch() {
        fh.switchable_comp_refs = br.get_bit()? as u8;
        fh.skip_mode_enabled = br.get_bit()? as u8;
        if seq.bawp {
            fh.bawp = br.get_bit()? as u8;
        }
        if (seq.motion_modes & (1 << 3)) != 0 {
            fh.warp_motion = br.get_bit()? as u8;
        }
    }

    fh.reduced_txtp_set = br.get_bits(2)? as u8;

    if fh.is_inter_or_switch() && seq.global_motion && br.get_bit()? != 0 {
        return Err(Av2Error::Unsupported(
            "global motion parameters".to_string(),
        ));
    }

    if seq.film_grain_present && (fh.show_immediate != 0 || fh.show_implicit != 0) {
        fh.film_grain.present = if seq.reduced_still_picture_header {
            1
        } else {
            br.get_bit()? as u8
        };
        if fh.film_grain.present != 0 || seq.reduced_still_picture_header {
            fh.film_grain.id = br.get_bits(3)? as u8;
            fh.film_grain.seed = br.get_bits(16)?;
        }
    }

    // Derived per-segment quantizers and lossless flags were computed directly
    // above (AV2 Spec §5.18.1 / §7.14.4).

    Ok(fh)
}

/// Parses a `film_grain_obu()` payload (§5.14) into the models it updates.
///
/// `seq_layout` is the active sequence chroma layout; §6.17.10.1 requires
/// `fgm_chroma_idc` to match it, and the reference decoder rejects a mismatch.
pub fn parse_film_grain_obu(
    payload: &[u8],
    seq_layout: PixelLayout,
) -> Result<[Option<FilmGrainData>; 8]> {
    let mut br = BitReader::new(payload);
    let update_flags = br.get_bits(8)? as u8;
    if update_flags == 0 {
        return Err(Av2Error::BitstreamError(
            "fgm_update_flags cannot be zero".to_string(),
        ));
    }
    let chroma_idc = br.get_vlc()?;
    let layout = match chroma_idc {
        0 => PixelLayout::I420,
        1 => PixelLayout::I400,
        2 => PixelLayout::I444,
        3 => PixelLayout::I422,
        other => {
            return Err(Av2Error::BitstreamError(format!(
                "invalid fgm_chroma_idc {other}"
            )));
        }
    };
    if layout != seq_layout {
        return Err(Av2Error::BitstreamError(format!(
            "fgm_chroma_idc {chroma_idc} does not match sequence layout {seq_layout:?}"
        )));
    }

    let mut models: [Option<FilmGrainData>; 8] = [None; 8];
    for (slot, model) in models.iter_mut().enumerate() {
        if update_flags & (1 << slot) != 0 {
            *model = Some(parse_film_grain_model(&mut br, layout)?);
        }
    }
    br.trailing_bits()?;
    Ok(models)
}

/// Parses `film_grain_model()` (§5.18.10.2) with the §6.17.10.2 bounds checks.
fn parse_film_grain_model(br: &mut BitReader<'_>, layout: PixelLayout) -> Result<FilmGrainData> {
    let mut data = FilmGrainData::default();
    let monochrome = layout == PixelLayout::I400;
    if !monochrome {
        data.chroma_scaling_from_luma = br.get_bit()? != 0;
    }
    let num_planes = if monochrome || data.chroma_scaling_from_luma {
        1
    } else {
        3
    };
    for plane in 0..num_planes {
        let num_points = br.get_bits(4)? as i32;
        if num_points > 14 {
            return Err(Av2Error::BitstreamError(format!(
                "film grain plane {plane} has {num_points} points"
            )));
        }
        data.num_points[plane] = num_points;
        if num_points == 0 {
            continue;
        }
        let index_bits = 1 + br.get_bits(3)?;
        let scaling_bits = 5 + br.get_bits(2)?;
        let mut base = 0u32;
        for i in 0..num_points as usize {
            base += br.get_bits(index_bits)?;
            if base > 255 {
                return Err(Av2Error::BitstreamError(
                    "film grain scaling point exceeds 255".to_string(),
                ));
            }
            if i > 0 && base <= data.points[plane][i - 1][0] as u32 {
                return Err(Av2Error::BitstreamError(
                    "film grain scaling points are not increasing".to_string(),
                ));
            }
            data.points[plane][i][0] = base as u8;
            data.points[plane][i][1] = br.get_bits(scaling_bits)? as u8;
        }
    }
    if layout == PixelLayout::I420 && (data.num_points[1] > 0) != (data.num_points[2] > 0) {
        return Err(Av2Error::BitstreamError(
            "4:2:0 film grain must enable both chroma planes or neither".to_string(),
        ));
    }

    data.scaling_shift = br.get_bits(2)? as i32 + 8;
    data.ar_coeff_lag = br.get_bits(2)? as i32;
    let num_pos = 2 * data.ar_coeff_lag * (data.ar_coeff_lag + 1);
    for plane in 0..3 {
        if data.num_points[plane] == 0 && (plane == 0 || !data.chroma_scaling_from_luma) {
            continue;
        }
        // Chroma gains one coefficient for the co-located luma template
        // (§5.18.10.2 `numPosChroma`).
        let num_plane_pos = num_pos + i32::from(plane > 0 && data.num_points[0] > 0);
        let coef_bits = 5 + br.get_bits(2)?;
        let half = 1i32 << (coef_bits - 1);
        for i in 0..num_plane_pos as usize {
            data.ar_coeffs[plane][i] = (br.get_bits(coef_bits)? as i32 - half) as i8;
        }
    }
    data.ar_coeff_shift = (br.get_bits(2)? + 6) as u64;
    data.grain_scale_shift = br.get_bits(2)? as i32;
    for plane in 1..3 {
        if data.num_points[plane] == 0 {
            continue;
        }
        data.uv_mult[plane - 1] = br.get_bits(8)? as i32 - 128;
        data.uv_luma_mult[plane - 1] = br.get_bits(8)? as i32 - 128;
        data.uv_offset[plane - 1] = br.get_bits(9)? as i32 - 256;
    }
    data.overlap_flag = br.get_bit()? != 0;
    data.clip_to_restricted_range = br.get_bit()? != 0;
    if data.clip_to_restricted_range {
        data.mc_identity = br.get_bit()? != 0;
    }
    data.block_size = br.get_bit()? as i32;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::headers::PixelLayout;
    use crate::av2::obu::{build_default_sequence_header, write_seq_hdr_obu};

    #[test]
    fn test_sequence_header_roundtrip() {
        for bitdepth in [8u8, 10, 12] {
            let seq = build_default_sequence_header(64, 48, PixelLayout::I420, bitdepth, false);
            let obu = write_seq_hdr_obu(&seq);
            let obus = split_obus(&obu).unwrap();
            assert_eq!(obus.len(), 1);
            assert_eq!(obus[0].obu_type, ObuType::SeqHdr);
            let parsed = parse_sequence_header(obus[0].payload).unwrap();
            assert_eq!(parsed.max_width, seq.max_width);
            assert_eq!(parsed.max_height, seq.max_height);
            assert_eq!(parsed.hbd, seq.hbd);
            assert_eq!(parsed.layout, seq.layout);
            assert_eq!(parsed.sb128, seq.sb128);
            assert_eq!(parsed.ref_frames, seq.ref_frames);
            assert_eq!(parsed.order_hint_n_bits, seq.order_hint_n_bits);
            assert_eq!(parsed.reduced_tx_part_set, seq.reduced_tx_part_set);
            assert_eq!(parsed.tiling.present, seq.tiling.present);
            assert_eq!(parsed.motion_modes, seq.motion_modes);
            assert_eq!(parsed.cdef, seq.cdef);
            assert_eq!(parsed.gdf, seq.gdf);
            assert_eq!(parsed.mrls, seq.mrls);
            assert_eq!(parsed.cfl, seq.cfl);
        }
    }

    #[test]
    fn test_film_grain_obu_roundtrip() {
        use crate::av2::filmgrain::write_film_grain_obu;

        let mut data = FilmGrainData::default();
        data.num_points[0] = 3;
        data.points[0][0] = [16, 0];
        data.points[0][1] = [40, 120];
        data.points[0][2] = [90, 200];
        data.num_points[1] = 2;
        data.points[1][0] = [0, 64];
        data.points[1][1] = [255, 96];
        data.num_points[2] = 2;
        data.points[2][0] = [0, 80];
        data.points[2][1] = [200, 100];
        data.scaling_shift = 11;
        data.ar_coeff_lag = 2;
        data.ar_coeffs[0][..12].copy_from_slice(&[0, 0, -58, 0, 0, 0, -76, 100, -43, 0, -51, 82]);
        data.ar_coeffs[1][..13].copy_from_slice(&[0, 0, -49, 0, 0, 0, -36, 22, -30, 0, -38, 7, 39]);
        data.ar_coeffs[2][..13]
            .copy_from_slice(&[0, 0, -47, 0, 0, 0, -31, 31, -25, 0, -32, 13, -100]);
        data.ar_coeff_shift = 8;
        data.grain_scale_shift = 1;
        data.uv_mult = [119, 101];
        data.uv_luma_mult = [64, 64];
        data.uv_offset = [-238, -202];
        data.overlap_flag = true;
        data.clip_to_restricted_range = true;
        data.mc_identity = true;
        data.block_size = 1;

        let obu = write_film_grain_obu(&data, PixelLayout::I420, 2, 0, 0, 0);
        let obus = split_obus(&obu).unwrap();
        assert_eq!(obus.len(), 1);
        assert_eq!(obus[0].obu_type, ObuType::Fgm);
        let models = parse_film_grain_obu(obus[0].payload, PixelLayout::I420).unwrap();
        let parsed = models[2].expect("slot 2");
        assert_eq!(parsed.num_points, data.num_points);
        assert_eq!(&parsed.points[..], &data.points[..]);
        assert_eq!(parsed.scaling_shift, data.scaling_shift);
        assert_eq!(parsed.ar_coeff_lag, data.ar_coeff_lag);
        assert_eq!(&parsed.ar_coeffs[..], &data.ar_coeffs[..]);
        assert_eq!(parsed.ar_coeff_shift, data.ar_coeff_shift);
        assert_eq!(parsed.grain_scale_shift, data.grain_scale_shift);
        assert_eq!(parsed.uv_mult, data.uv_mult);
        assert_eq!(parsed.uv_luma_mult, data.uv_luma_mult);
        assert_eq!(parsed.uv_offset, data.uv_offset);
        assert_eq!(parsed.overlap_flag, data.overlap_flag);
        assert_eq!(
            parsed.clip_to_restricted_range,
            data.clip_to_restricted_range
        );
        assert_eq!(parsed.mc_identity, data.mc_identity);
        assert_eq!(parsed.block_size, data.block_size);

        // The sequence layout must match `fgm_chroma_idc` (§6.17.10.1).
        assert!(parse_film_grain_obu(obus[0].payload, PixelLayout::I444).is_err());
    }

    #[test]
    fn test_split_obus_handles_temporal_delimiter() {
        let seq = build_default_sequence_header(64, 64, PixelLayout::I420, 8, false);
        let mut data = crate::av2::obu::write_temporal_delimiter_obu();
        data.extend_from_slice(&write_seq_hdr_obu(&seq));
        let obus = split_obus(&data).unwrap();
        assert_eq!(obus.len(), 2);
        assert_eq!(obus[0].obu_type, ObuType::Td);
        assert_eq!(obus[1].obu_type, ObuType::SeqHdr);
    }
}

#[cfg(test)]
mod frame_header_tests {
    use super::*;
    use crate::av2::container::RawFrame;
    use crate::av2::headers::PixelLayout;
    use crate::av2::{Encoder, EncoderConfig};

    #[test]
    fn test_frame_header_roundtrip_against_encoder() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            layout: PixelLayout::I420,
            bit_depth: 8,
            ..EncoderConfig::default()
        };
        let mut enc = Encoder::<u8>::new(cfg).unwrap();
        let mut seq = None;
        for idx in 0..3usize {
            let raw = RawFrame::<u8>::generate_synthetic(64, 64, 8, PixelLayout::I420, idx);
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            let obus = split_obus(&pkt.data).unwrap();
            for obu in &obus {
                if obu.obu_type == ObuType::SeqHdr {
                    seq = Some(parse_sequence_header(obu.payload).unwrap());
                }
            }
            let seqhdr = seq.as_ref().unwrap();
            let frame_obu = obus
                .iter()
                .find(|o| matches!(o.obu_type, ObuType::ClosedLoopKf | ObuType::TileGrp))
                .unwrap();
            let (fh, tiles) = parse_tile_group(seqhdr, frame_obu).unwrap();
            assert_eq!(fh.width, 64);
            assert_eq!(fh.height, 64);
            assert_eq!(fh.quant.yac, pkt.stats.qidx);
            assert_eq!(fh.tiling.t.cols, 1);
            assert_eq!(fh.tiling.t.rows, 1);
            assert_eq!(tiles.len(), 1);
            assert_eq!(fh.frame_type, pkt.frame_type);
            if idx > 0 {
                assert_eq!(fh.frame_type, FrameType::Inter);
                assert_eq!(fh.n_ref_frames, 2);
            }
        }
    }
}
