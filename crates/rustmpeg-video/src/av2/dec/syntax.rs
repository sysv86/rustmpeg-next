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

//! Entropy-syntax decoders that mirror the encoder's block coding decisions.
//!
//! Each function here is the exact inverse of the corresponding encoder
//! function in [`crate::av2::entropy`] and [`crate::av2::partition`], using the
//! same CDF tables and the same neighbour-context derivations. Keeping the two
//! halves in lockstep is what makes the decoder reproducible against the
//! encoder reconstruction and the reference decoder.

use crate::av2::cdf::{CdfCoefContext, CdfModeContext};
use crate::av2::dec::msac::MsacDecoder;
use crate::av2::entropy::{
    BlockContext, build_intra_y_mode_order, get_dc_sign_ctx, get_lo_ctx, get_partition_ctx,
    get_skip_ctx,
};
use crate::av2::error::{Av2Error, Result};
use crate::av2::headers::{FrameHeader, MAX_SEGMENTS, SEG_LVL_GLOBALMV, SEG_LVL_SKIP};
use crate::av2::intops::{iclip, imin};
use crate::av2::levels::{BlockPartition, BlockSize, txtp};
use crate::av2::partition::{
    AllowedPartitions, get_partition2_ctx, partition_size_adjust, partition_size_adjust_rect_type,
    rect_type_implied_by_bsize,
};
use crate::av2::quantizer::dequantize_coef;
use crate::av2::scan::SCANS;
use crate::av2::screen::{MAX_PALETTE_COLORS, palette_color_context};
use crate::av2::segment::{SegmentMap, neg_deinterleave};
use crate::av2::tables::{
    REORDERED_DIR_Y_MODE, REORDERED_NONDIR_Y_MODE, TXFM_DIMENSIONS, TXTP_FROM_UVMODE,
};
use crate::av2::transform::derive_chroma_txtp;

/// `CeilLog2(x)` for the palette delta bit-width update (AV2 Spec v1.0.0
/// §5.20.8.1).
#[inline]
fn ceil_log2_u32(v: u32) -> u32 {
    if v <= 1 {
        0
    } else {
        32 - (v - 1).leading_zeros()
    }
}

/// One decoded palette block: `PaletteSizeY`, `palette_colors_y`, and the
/// `ColorMapY` index map in row-major block coordinates.
#[derive(Debug, Clone)]
pub struct PaletteBlock {
    /// `PaletteSizeY` (`2..=8`).
    pub size: usize,
    /// `palette_colors_y[0..size]`, sorted ascending.
    pub colors: [i32; MAX_PALETTE_COLORS],
    /// `ColorMapY[r][c]` for the full block footprint.
    pub map: Vec<u8>,
}

/// Decodes `palette_mode_info()` and `palette_tokens()` (AV2 Spec v1.0.0
/// §5.20.8.1 and §5.20.8.4).
///
/// Returns `Ok(None)` when `has_palette_y == 0`. `above` and `left` are the
/// neighbour palettes that `get_palette_cache()` interleaves; `block_w` /
/// `block_h` are `Block_Width` / `Block_Height`, and `onscreen_w` /
/// `onscreen_h` clamp the token loop to `MiCols * 4` / `MiRows * 4`.
#[allow(clippy::too_many_arguments)]
pub fn decode_palette_mode_info(
    msac: &mut MsacDecoder<'_>,
    cdf_m: &mut CdfModeContext,
    bit_depth: u8,
    above: &[i32],
    left: &[i32],
    block_w: usize,
    block_h: usize,
    onscreen_w: usize,
    onscreen_h: usize,
) -> Result<Option<PaletteBlock>> {
    if msac.decode_bool_adapt(&mut cdf_m.palette_y_mode)? == 0 {
        return Ok(None);
    }
    let size = msac.decode_symbol_adapt(&mut cdf_m.palette_y_size, 6)? + 2;
    let max = ((1i32 << bit_depth) - 1).max(0);
    let mut colors = [0i32; MAX_PALETTE_COLORS];

    // `get_palette_cache()`: interleave the above and left palettes.
    let mut cache = Vec::with_capacity(above.len() + left.len());
    let (mut ai, mut li) = (0usize, 0usize);
    while ai < above.len() || li < left.len() {
        if ai < above.len() {
            cache.push(above[ai]);
            ai += 1;
        }
        if li < left.len() {
            cache.push(left[li]);
            li += 1;
        }
    }

    let mut idx = 0usize;
    let mut ci = 0usize;
    while ci < cache.len() && idx < size {
        if msac.decode_bool_bypass() != 0 {
            colors[idx] = cache[ci].clamp(0, max);
            idx += 1;
        }
        ci += 1;
    }
    if idx < size {
        colors[idx] = msac.decode_bools_bypass(bit_depth as u32) as i32;
        idx += 1;
    }
    let mut bits = 0u32;
    if idx < size {
        bits = (bit_depth as u32).saturating_sub(3) + msac.decode_bools_bypass(2);
    }
    while idx < size {
        let delta = msac.decode_bools_bypass(bits) + 1;
        colors[idx] = (colors[idx - 1] + delta as i32).clamp(0, max);
        let range = (max - colors[idx]).max(0) as u32;
        bits = bits.min(ceil_log2_u32(range));
        idx += 1;
    }
    colors[..size].sort_unstable();

    // `palette_tokens()`.
    let direction = if block_w < 64 && block_h < 64 {
        msac.decode_bool_bypass() as usize
    } else {
        0
    };
    let (outer_lim, inner_lim) = if direction != 0 {
        (onscreen_w, onscreen_h)
    } else {
        (onscreen_h, onscreen_w)
    };
    let mut map = vec![0u8; block_w * block_h];
    let mut prev_identity = 3usize;
    for i in 0..outer_lim {
        let identity = msac.decode_symbol_adapt(&mut cdf_m.identity_row_y[prev_identity], 2)? as u8;
        for j in 0..inner_lim {
            let (r, c) = if direction != 0 { (j, i) } else { (i, j) };
            if identity == 2 {
                map[r * block_w + c] = if direction != 0 {
                    map[r * block_w + (c - 1)]
                } else {
                    map[(r - 1) * block_w + c]
                };
            } else if identity == 1 && j > 0 {
                map[r * block_w + c] = if direction != 0 {
                    map[(r - 1) * block_w + c]
                } else {
                    map[r * block_w + (c - 1)]
                };
            } else if r == 0 && c == 0 {
                map[0] = msac.decode_uniform(size as u32) as u8;
            } else {
                let (ctx, order) = palette_color_context(&map, block_w, r, c, size);
                let ctx = ctx.min(4);
                let rank = match size {
                    2 => msac.decode_symbol_adapt(&mut cdf_m.palette_size_2_y_color[ctx], 1)?,
                    3 => msac.decode_symbol_adapt(&mut cdf_m.palette_size_3_y_color[ctx], 2)?,
                    4 => msac.decode_symbol_adapt(&mut cdf_m.palette_size_4_y_color[ctx], 3)?,
                    5 => msac.decode_symbol_adapt(&mut cdf_m.palette_size_5_y_color[ctx], 4)?,
                    6 => msac.decode_symbol_adapt(&mut cdf_m.palette_size_6_y_color[ctx], 5)?,
                    7 => msac.decode_symbol_adapt(&mut cdf_m.palette_size_7_y_color[ctx], 6)?,
                    _ => msac.decode_symbol_adapt(&mut cdf_m.palette_size_8_y_color[ctx], 7)?,
                };
                map[r * block_w + c] = order[rank] as u8;
            }
        }
        prev_identity = identity as usize;
    }

    Ok(Some(PaletteBlock { size, colors, map }))
}

/// Default directional order of the eight chroma intra prediction modes.
const DEFAULT_MODE_LIST_UV: [u8; 8] = [1, 2, 3, 4, 8, 5, 6, 7];
/// Position in [`DEFAULT_MODE_LIST_UV`] replaced by "same direction as luma".
const INTRA_DIR_MODE_Y_TO_UV_IDX: [u8; 8] = [2, 4, 0, 5, 3, 6, 1, 7];

/// Decodes one coding-tree partition decision for `bs` at `(bx4_in_sb, by4_in_sb)`
/// per AV2 Spec v1.0.0 §5.20.3.2 (`read_partition`).
///
/// `allowed` is the legal-partition set for this node. Implied partitions and
/// nodes with a single legal partition are resolved by the caller and never
/// reach this function, because the syntax codes no symbol for them.
pub fn decode_partition_decision(
    msac: &mut MsacDecoder<'_>,
    cdf_m: &mut CdfModeContext,
    a: &BlockContext,
    l: &BlockContext,
    bs: BlockSize,
    bx4_in_sb: usize,
    by4_in_sb: usize,
    allowed: &AllowedPartitions,
) -> Result<BlockPartition> {
    let ctx = get_partition_ctx(a, l, bs, bx4_in_sb, by4_in_sb);
    let part_ctx = partition_size_adjust(bs) * 4 + ctx;

    if allowed.contains(BlockPartition::None)
        && msac.decode_bool_adapt(cdf_m.part_split(0, part_ctx))? == 0
    {
        return Ok(BlockPartition::None);
    }

    if allowed.contains(BlockPartition::Split) {
        // The `+4` context term only applies to 256x256 superblocks, which this
        // decoder does not support.
        if msac.decode_bool_adapt(cdf_m.part_square(ctx))? != 0 {
            return Ok(BlockPartition::Split);
        }
    }

    let vertical = match rect_type_implied_by_bsize(bs) {
        Some(v) => v,
        None => {
            let allow_h = allowed.contains(BlockPartition::H)
                || allowed.contains(BlockPartition::H3)
                || allowed.contains(BlockPartition::H4A)
                || allowed.contains(BlockPartition::H4B);
            let allow_v = allowed.contains(BlockPartition::V)
                || allowed.contains(BlockPartition::V3)
                || allowed.contains(BlockPartition::V4A)
                || allowed.contains(BlockPartition::V4B);
            if allow_h && allow_v {
                let rect_ctx = partition_size_adjust_rect_type(bs) * 4 + ctx;
                msac.decode_bool_adapt(cdf_m.part_dir(0, rect_ctx))? != 0
            } else {
                !allow_h
            }
        }
    };

    let (non_ext, ext3, ext4) = if !vertical {
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

    if ext3 || ext4 {
        let do_ext = if non_ext {
            let ctx5 = get_partition2_ctx(a, l, bs, vertical, bx4_in_sb, by4_in_sb);
            let ctx6 = ctx5 + partition_size_adjust(bs) * 4;
            msac.decode_bool_adapt(cdf_m.part_ext(0, ctx6))? != 0
        } else {
            true
        };

        if do_ext {
            if ext4 {
                let do_4way = if ext3 {
                    let ctx5 = get_partition2_ctx(a, l, bs, vertical, bx4_in_sb, by4_in_sb);
                    let ctx6 = ctx5 + partition_size_adjust(bs) * 4;
                    msac.decode_bool_adapt(cdf_m.part_uneven_4way(0, ctx6))? != 0
                } else {
                    true
                };

                if do_4way {
                    let is_b = msac.decode_bool_bypass() != 0;
                    let bp = if !vertical {
                        if is_b {
                            BlockPartition::H4B
                        } else {
                            BlockPartition::H4A
                        }
                    } else {
                        if is_b {
                            BlockPartition::V4B
                        } else {
                            BlockPartition::V4A
                        }
                    };
                    if !allowed.contains(bp) {
                        return Err(Av2Error::BitstreamError(format!(
                            "partition {bp:?} not legal for {bs:?}"
                        )));
                    }
                    return Ok(bp);
                }
            }
            let bp = if !vertical {
                BlockPartition::H3
            } else {
                BlockPartition::V3
            };
            if !allowed.contains(bp) {
                return Err(Av2Error::BitstreamError(format!(
                    "partition {bp:?} not legal for {bs:?}"
                )));
            }
            return Ok(bp);
        }
    }

    let bp = if vertical {
        BlockPartition::V
    } else {
        BlockPartition::H
    };
    if !allowed.contains(bp) {
        return Err(Av2Error::BitstreamError(format!(
            "partition {bp:?} not legal for {bs:?}"
        )));
    }
    Ok(bp)
}

/// Decodes the joint luma intra mode (`y_mode`, angle delta, and the
/// directional mode index used by chroma coding) and the lossless DPCM flag
/// (`use_dpcm_y` / `dpcm_mode_y`, AV2 Spec v1.0.0 §5.20.5.5).
pub fn decode_intra_y_mode(
    msac: &mut MsacDecoder<'_>,
    cdf_m: &mut CdfModeContext,
    a: &BlockContext,
    l: &BlockContext,
    bx4: usize,
    by4: usize,
    bw4: i32,
    bh4: i32,
    seg_lossless: bool,
) -> Result<(u8, i8, u8, Option<bool>)> {
    if seg_lossless {
        let dpcm = msac.decode_bool_adapt(cdf_m.dpcm(0))?;
        if dpcm != 0 {
            // `dpcm_mode_y`: 0 = `V_PRED`, 1 = `H_PRED`. The joint mode index
            // stored in the neighbour context is the joint index itself
            // (`45` for `H_PRED`, `17` for `V_PRED`).
            let horizontal = msac.decode_bool_adapt(cdf_m.dpcm_mode(0))? != 0;
            let mode = if horizontal { 2u8 } else { 1u8 };
            let midx = if horizontal { 45u8 } else { 17u8 };
            return Ok((mode, 0, midx, Some(horizontal)));
        }
    }

    // `get_joint_mode(0)` is the left neighbour at the block's last row and
    // `get_joint_mode(1)` the above neighbour at its last column (AV2 Spec
    // v1.0.0 §5.20.7.15); a non-directional or absent neighbour contributes 0.
    let a_midx = a.midx[(bx4 + bw4 as usize - 1) & 31];
    let l_midx = l.midx[(by4 + bh4 as usize - 1) & 31];
    let a_is_dir = (a_midx != 0xFF) as usize;
    let l_is_dir = (l_midx != 0xFF) as usize;
    let ctx = a_is_dir + l_is_dir;

    let set = msac.decode_symbol_adapt(cdf_m.y_mode_set(), 3)?;
    let mode_idx = if set == 0 {
        let idx = msac.decode_symbol_adapt(cdf_m.y_mode_index(ctx), 7)?;
        if idx < 7 {
            idx
        } else {
            7 + msac.decode_symbol_adapt(cdf_m.y_mode_offset(ctx), 5)?
        }
    } else {
        let second = msac.decode_bools_bypass(4) as usize;
        13 + (set - 1) * 16 + second
    };

    if mode_idx < 5 {
        Ok((REORDERED_NONDIR_Y_MODE[mode_idx], 0, 0xFF, None))
    } else if mode_idx <= 60 {
        let rank = mode_idx - 5;
        let (ordered, len) = build_intra_y_mode_order(a_midx, l_midx, bw4, bh4);
        if rank >= len {
            return Err(Av2Error::BitstreamError(format!(
                "intra luma mode rank {rank} outside the neighbour list of {len}"
            )));
        }
        let target = ordered[rank] as usize;
        let reord = target / 7;
        let angle_idx = target % 7;
        if reord >= REORDERED_DIR_Y_MODE.len() {
            return Err(Av2Error::BitstreamError(
                "intra luma directional mode index out of range".to_string(),
            ));
        }
        Ok((
            REORDERED_DIR_Y_MODE[reord],
            angle_idx as i8 - 3,
            target as u8,
            None,
        ))
    } else {
        Err(Av2Error::BitstreamError(
            "intra luma mode index out of range".to_string(),
        ))
    }
}

/// Decodes the chroma intra mode, its angle delta, and the lossless DPCM flag
/// (`use_dpcm_uv` / `dpcm_mode_uv`, AV2 Spec v1.0.0 §5.20.5.6).
pub fn decode_intra_uv_mode(
    msac: &mut MsacDecoder<'_>,
    cdf_m: &mut CdfModeContext,
    cfl_allowed: bool,
    seg_lossless: bool,
    luma_midx: u8,
) -> Result<(u8, i8, Option<bool>)> {
    if seg_lossless {
        let dpcm = msac.decode_bool_adapt(cdf_m.dpcm(1))?;
        if dpcm != 0 {
            let horizontal = msac.decode_bool_adapt(cdf_m.dpcm_mode(1))? != 0;
            let mode = if horizontal { 2u8 } else { 1u8 };
            return Ok((mode, 0, Some(horizontal)));
        }
    }
    if cfl_allowed && msac.decode_bool_adapt(cdf_m.is_cfl(0))? != 0 {
        return Ok((13, 0, None));
    }

    let uv_mode_ctx = (luma_midx != 0xFF) as usize;
    let sym = msac.decode_symbol_adapt(cdf_m.uv_mode_cfl_not_allowed(uv_mode_ctx), 7)?;
    let uv_idx = if sym < 7 {
        sym
    } else {
        7 + msac.decode_bools_bypass(3) as usize
    };

    let (uv_mode, uv_angle) = if uv_idx < uv_mode_ctx {
        (
            REORDERED_DIR_Y_MODE[(luma_midx / 7) as usize],
            (luma_midx % 7) as i8 - 3,
        )
    } else if uv_idx - uv_mode_ctx < 5 {
        (REORDERED_NONDIR_Y_MODE[uv_idx - uv_mode_ctx], 0)
    } else {
        let mut idx = uv_idx - 5 - uv_mode_ctx;
        if uv_mode_ctx != 0 {
            let luma_dir = (luma_midx / 7) as usize;
            let threshold = INTRA_DIR_MODE_Y_TO_UV_IDX[luma_dir.min(7)] as usize;
            if idx >= threshold {
                idx += 1;
            }
        }
        if idx > 7 {
            eprintln!(
                "DEBUG uv_mode: uv_idx={} uv_mode_ctx={} luma_midx={} idx={}",
                uv_idx, uv_mode_ctx, luma_midx, idx
            );
            return Err(Av2Error::BitstreamError(
                "chroma intra mode index out of range".to_string(),
            ));
        }
        (DEFAULT_MODE_LIST_UV[idx], 0)
    };
    Ok((uv_mode, uv_angle, None))
}

/// Decodes the CfL alpha parameters (`alpha_u`, `alpha_v`).
pub fn decode_cfl_alphas(
    msac: &mut MsacDecoder<'_>,
    cdf_m: &mut CdfModeContext,
) -> Result<(i8, i8)> {
    if msac.decode_bool_adapt(cdf_m.cfl_index())? != 0 {
        return Err(Av2Error::Unsupported(
            "implicit CfL (MHCCP) signalling".to_string(),
        ));
    }
    let joint = msac.decode_symbol_adapt(cdf_m.cfl_signs(), 7)? as u32;
    let combined = joint + 1;
    let sign_u = (combined / 3) as usize;
    let sign_v = (combined % 3) as usize;
    let alpha_u = if sign_u != 0 {
        let abs = msac.decode_symbol_adapt(cdf_m.cfl_alpha((sign_u - 1) * 3 + sign_v), 7)? + 1;
        if sign_u == 1 { -(abs as i8) } else { abs as i8 }
    } else {
        0
    };
    let alpha_v = if sign_v != 0 {
        let abs = msac.decode_symbol_adapt(cdf_m.cfl_alpha((sign_v - 1) * 3 + sign_u), 7)? + 1;
        if sign_v == 1 { -(abs as i8) } else { abs as i8 }
    } else {
        0
    };
    Ok((alpha_u, alpha_v))
}

/// Decoder-side view of the frame segmentation syntax
/// (AV2 Spec v1.0.0 §5.18.7.1 and §5.20.5.7/§5.20.7.2).
#[derive(Debug, Clone, Copy)]
pub struct SegSyntax {
    /// `segmentation_enabled`.
    pub enabled: bool,
    /// `segmentation_update_map`.
    pub update_map: bool,
    /// `segmentation_temporal_update`.
    pub temporal: bool,
    /// `SegIdPreSkip`: any `SEG_LVL_SKIP` / `SEG_LVL_GLOBALMV` feature is on.
    pub preskip: bool,
    /// `HasLosslessSegment`.
    pub any_lossless: bool,
    /// `LastActiveSegId`, or `-1` when no feature is enabled.
    pub last_active_segid: i16,
    /// `MaxSegments` (`8`, or `16` with `enable_ext_seg`).
    pub max_segments: usize,
}

impl SegSyntax {
    /// Derives the decoder segmentation state from a parsed frame header.
    pub fn from_frame_header(fh: &FrameHeader, seq_ext_seg: bool) -> Self {
        Self {
            enabled: fh.segmentation.enabled != 0,
            update_map: fh.segmentation.update_map != 0,
            temporal: fh.segmentation.temporal != 0,
            preskip: fh.segmentation.preskip != 0,
            any_lossless: fh.any_lossless != 0,
            last_active_segid: fh.segmentation.last_active_segid as i16,
            max_segments: if seq_ext_seg {
                MAX_SEGMENTS
            } else {
                MAX_SEGMENTS / 2
            },
        }
    }

    /// `seg_feature_active_idx(idx, feature)` for the current segment.
    #[inline]
    pub fn feature_active(&self, fh: &FrameHeader, segment: u8, feature: usize) -> bool {
        fh.segmentation.enabled != 0 && fh.segmentation.d.feature_active(segment as usize, feature)
    }

    /// True when `SEG_LVL_SKIP` is enabled for `segment`.
    #[inline]
    pub fn skip_feature(&self, fh: &FrameHeader, segment: u8) -> bool {
        self.feature_active(fh, segment, SEG_LVL_SKIP)
    }

    /// True when `SEG_LVL_GLOBALMV` is enabled for `segment`.
    #[inline]
    pub fn globalmv_feature(&self, fh: &FrameHeader, segment: u8) -> bool {
        self.feature_active(fh, segment, SEG_LVL_GLOBALMV)
    }
}

/// Reads the spatial segment-id predictor and three-way context from the
/// current-frame map (AV2 Spec v1.0.0 §5.20.5.8).
fn spatial_segment_id(
    msac: &mut MsacDecoder<'_>,
    cdf: &mut CdfModeContext,
    syn: &SegSyntax,
    cur: &mut SegmentMap,
    bx4: usize,
    by4: usize,
    have_left: bool,
    have_top: bool,
) -> Result<(u8, u8)> {
    let (pred, ctx) = cur.predict(bx4, by4, have_left, have_top);
    let ctx = ctx.min(2);
    let ext = if syn.max_segments > 8 {
        msac.decode_bool_adapt(&mut cdf.seg_id_ext_flag[ctx])? as i32
    } else {
        0
    };
    let symbol = if ext == 0 {
        msac.decode_symbol_adapt(&mut cdf.seg_id[ctx], 7)?
    } else {
        msac.decode_symbol_adapt(&mut cdf.seg_id_ext[ctx], 7)?
    } as i32;
    let diff = symbol + (ext << 3);
    let max = syn.last_active_segid as i32 + 1;
    if max <= 0 {
        // No feature is active: `LastActiveSegId` is -1 and no explicit id is
        // representable. dav2d never reaches this state for a conformant
        // stream, but returning 0 keeps a malformed header from reading bits
        // that are not there.
        return Ok((0, 0));
    }
    let id = neg_deinterleave(diff, pred as i32, max);
    if id < 0 || id >= max || id >= MAX_SEGMENTS as i32 {
        return Err(Av2Error::BitstreamError(format!(
            "decoded segment id {id} outside LastActiveSegId {}",
            syn.last_active_segid
        )));
    }
    Ok((id as u8, 0))
}

/// Reads and applies one block's segment id.
///
/// This is the inverse of [`crate::av2::segment::encode_segment_id`] and follows
/// AV2 Spec v1.0.0 §5.20.5.7 (`intra_segment_id()`), §5.20.5.8
/// (`read_segment_id()`), and §5.20.7.2 (`inter_segment_id(preSkip)`).
///
/// `spatial_only` is set for `intra_segment_id()`, which never reads a temporal
/// prediction bit. `pre_skip` selects the `inter_segment_id(1)` call that runs
/// before `skip_flag`; `skip_txfm` is the already-decoded skip flag for the
/// post-skip call.
#[allow(clippy::too_many_arguments)]
pub fn decode_segment_id(
    msac: &mut MsacDecoder<'_>,
    cdf: &mut CdfModeContext,
    syn: &SegSyntax,
    cur: &mut SegmentMap,
    prev: Option<&SegmentMap>,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
    have_left: bool,
    have_top: bool,
    spatial_only: bool,
    pre_skip: bool,
    skip_txfm: bool,
    seg_pred_a: &mut [u8; 32],
    seg_pred_l: &mut [u8; 32],
) -> Result<u8> {
    if !syn.enabled {
        return Ok(0);
    }

    let mut seg_predicted = 0u8;
    let id;

    if spatial_only {
        (id, _) = spatial_segment_id(msac, cdf, syn, cur, bx4, by4, have_left, have_top)?;
    } else if !syn.update_map {
        // The frame reuses the loaded previous-frame map; no segment id bits
        // are coded for luma (AV2 Spec §5.20.7.2, `get_segment_id()`).
        let source = prev.unwrap_or(&*cur);
        id = source.min_block(bx4, by4, bw4, bh4);
    } else if pre_skip && !syn.preskip {
        // `inter_segment_id(1)` is a no-op when `SegIdPreSkip` is 0; the
        // post-skip call codes the id.
        return Ok(0);
    } else if !pre_skip && skip_txfm && !syn.any_lossless {
        // A skipped non-lossless block infers the spatial predictor and codes
        // no `seg_id_predicted` bit (§5.20.7.2).
        let (pred, _) = cur.predict(bx4, by4, have_left, have_top);
        id = pred;
    } else if syn.temporal && (!skip_txfm || pre_skip) {
        let ctx = (seg_pred_a[bx4 & 31] + seg_pred_l[by4 & 31]).min(2) as usize;
        let bit = msac.decode_bool_adapt(&mut cdf.seg_id_predicted[ctx])?;
        seg_predicted = bit as u8;
        if bit != 0 {
            let predicted = prev
                .map(|map| map.min_block(bx4, by4, bw4, bh4))
                .unwrap_or(0);
            if predicted as usize >= syn.max_segments {
                return Err(Av2Error::BitstreamError(format!(
                    "temporal segment id {predicted} exceeds MaxSegments {}",
                    syn.max_segments
                )));
            }
            id = predicted;
        } else {
            (id, _) = spatial_segment_id(msac, cdf, syn, cur, bx4, by4, have_left, have_top)?;
        }
    } else {
        (id, _) = spatial_segment_id(msac, cdf, syn, cur, bx4, by4, have_left, have_top)?;
    }

    cur.fill_block(bx4, by4, bw4, bh4, id);
    for i in 0..bw4.min(32 - (bx4 & 31)) {
        seg_pred_a[(bx4 & 31) + i] = seg_predicted;
    }
    for i in 0..bh4.min(32 - (by4 & 31)) {
        seg_pred_l[(by4 & 31) + i] = seg_predicted;
    }
    Ok(id)
}

/// Reads `read_delta_qindex()` and returns the updated `CurrentQIndex`
/// (AV2 Spec v1.0.0 §5.20.5.11).
///
/// This is the inverse of [`crate::av2::segment::encode_delta_q`] and matches
/// dav2d's `delta-q/lf` block: the escape value `DELTA_Q_SMALL` is followed by
/// a 3-bit `delta_q_rem_bits` plus that many `delta_q_abs_bits`, then an
/// optional sign, and the reduced value is shifted by `delta_q_res`.
pub fn decode_delta_qindex(
    msac: &mut MsacDecoder<'_>,
    cdf: &mut CdfModeContext,
    current_qidx: i32,
    res_log2: u8,
    max_q: i32,
) -> Result<i32> {
    let sym = msac.decode_symbol_adapt(&mut cdf.delta_q, 7)?;
    let abs = if sym == 7 {
        let n_bits = 1 + msac.decode_bools_bypass(3);
        msac.decode_bools_bypass(n_bits) as i32 + 5 + (1 << n_bits)
    } else {
        sym as i32
    };
    let delta = if abs != 0 {
        let signed = if msac.decode_bool_bypass() != 0 {
            -abs
        } else {
            abs
        };
        signed << res_log2
    } else {
        0
    };
    Ok(iclip(current_qidx + delta, 1, max_q))
}

/// Parameters for decoding one transform block's coefficients.
#[derive(Debug, Clone, Copy)]
pub struct CoefDecodeParams {
    /// Rectangular transform size index.
    pub tx: usize,
    /// Block size index for the skip/level contexts.
    pub bs: usize,
    /// Plane index (`0` luma, `1` U, `2` V).
    pub plane: i32,
    /// Whether the block is intra.
    pub intra: bool,
    /// Whether the block uses the frequency-separated coefficient mode.
    pub fsc: bool,
    /// Whether the segment is lossless.
    pub lossless: bool,
    /// Intra chroma mode (used to derive the chroma transform kernel).
    pub uv_mode: usize,
    /// Luma intra prediction mode and angle delta (`Md_Idx_To_Type`).
    pub y_mode: usize,
    /// Luma intra angle delta (`AngleDeltaY`).
    pub y_angle: i8,
    /// Luma multiple-reference-line index (`MrlIndex`).
    pub mrl_index: u8,
    /// Luma transform type, used by an inter chroma plane.
    pub y_txtp: u32,
    /// Frame-level `reduced_txtp_set` value.
    pub reduced_txtp_set: i32,
    /// Whether the U plane has a coded coefficient block (V-plane context).
    pub u_has_cf: i32,
    /// Dequantizer DC/AC step pair for this plane.
    pub dq_tbl: [u32; 2],
    /// Sample bit depth.
    pub bitdepth: u8,
    /// Horizontal chroma subsampling shift.
    pub ss_hor: bool,
    /// Vertical chroma subsampling shift.
    pub ss_ver: bool,
    /// Whether chroma transforms are restricted to DCT.
    pub chroma_dctonly: bool,
}

/// Result of decoding one transform block.
#[derive(Debug, Clone, Copy)]
pub struct CoefDecodeResult {
    /// Last non-zero coefficient index in scan order (`-1` when all zero).
    pub eob: i32,
    /// Derived transform kernel.
    pub txtp: u16,
    /// Coefficient level context propagated to neighbours.
    pub res_ctx: u8,
}

/// Derives (and, where the syntax codes it, reads) the transform kernel for a
/// block from the frame and block syntax (AV2 Spec v1.0.0 §5.20.8.2/§5.20.8.3;
/// `dav2d` `recon_tmpl.c` `decode_cf`).
fn derive_txtp(
    msac: &mut MsacDecoder<'_>,
    cdf_m: &mut CdfModeContext,
    p: &CoefDecodeParams,
    t_dim: &crate::av2::tables::TxfmInfo,
    eob: i32,
) -> Result<u16> {
    use crate::av2::transform::{
        ANGLE_STEP, MD_IDX_TO_TYPE, MODE_TO_ANGLE, MRL_INDEX_TO_DELTA, TX_TYPE_INTER_INV_SET1,
        TX_TYPE_INTER_INV_SET2, TX_TYPE_INV_LONG, TX_TYPE_TO_TXTP, is_directional_mode,
        wide_angle_mapping,
    };

    if p.lossless {
        if p.plane == 0 {
            if p.intra {
                return Ok(txtp::WHT_WHT as u16);
            }
            if t_dim.max == 0 {
                // `txtp_lossless` (WHT vs IDTX for a 4x4 inter transform) is
                // not part of this decoder's CDF context; the encoder is
                // intra-only at quantizer index 0.
                return Err(Av2Error::Unsupported(
                    "inter lossless 4x4 transform type signalling".to_string(),
                ));
            }
            return Ok(txtp::IDTX as u16);
        }
        if p.intra {
            return Ok(txtp::WHT_WHT as u16);
        }
        // Inter chroma inherits the (masked) luma kernel.
        return Ok(derive_chroma_txtp(p.y_txtp & 0xFF, t_dim, p.tx) as u16);
    }

    if p.plane != 0 {
        if p.chroma_dctonly {
            return Ok(txtp::DCT_DCT as u16);
        }
        let seed = if p.intra {
            TXTP_FROM_UVMODE[p.uv_mode.min(TXTP_FROM_UVMODE.len() - 1)] as u32
        } else {
            p.y_txtp & 0xFF
        };
        return Ok(derive_chroma_txtp(seed, t_dim, p.tx) as u16);
    }

    if p.intra {
        // `compute_tx_type` for luma intra (§5.20.8.2).
        if t_dim.max > 3 && t_dim.min >= 3 {
            return Ok(txtp::DCT_DCT as u16);
        }
        if p.fsc {
            return Ok(txtp::IDTX as u16);
        }
        if eob == 0 || p.tx == 3 || p.reduced_txtp_set == 2 {
            return Ok(txtp::DCT_DCT as u16);
        }
        if t_dim.max >= 3 {
            // Long 64/32 side: `is_long_side_dct` then the short 1D index.
            let long_dct = if t_dim.max == 4 {
                1usize
            } else {
                msac.decode_bool_adapt(cdf_m.is_long_side_dct(0))? as usize
            };
            let sym =
                msac.decode_symbol_adapt(&mut cdf_m.intra_tx_type_long[t_dim.min as usize], 3)?;
            let orient = (t_dim.w < t_dim.h) as usize;
            let t = TX_TYPE_INV_LONG[long_dct][orient][sym.min(3)] as usize;
            return Ok(TX_TYPE_TO_TXTP[t] as u16);
        }
        // `EXT_NEW_TX_SET` / reduced intra sets.
        let sz_ctx = ((t_dim.lw + t_dim.lh) >> 1) as usize;
        let tx_idx = if p.reduced_txtp_set != 0 {
            msac.decode_bool_adapt(&mut cdf_m.intra_tx_type_set2[t_dim.min as usize])? as usize
        } else {
            msac.decode_symbol_adapt(&mut cdf_m.intra_tx_type_set1[t_dim.min as usize], 6)?
        };
        let mut intra_dir = (p.y_mode as u8).min(12);
        if is_directional_mode(intra_dir) {
            let p_angle = MODE_TO_ANGLE[intra_dir as usize] as i32
                + p.y_angle as i32 * ANGLE_STEP
                + MRL_INDEX_TO_DELTA[p.mrl_index.min(3) as usize] as i32;
            intra_dir = wide_angle_mapping(
                intra_dir,
                4 * t_dim.w as usize,
                4 * t_dim.h as usize,
                p_angle,
            )
            .0;
        }
        let t = MD_IDX_TO_TYPE[sz_ctx.min(3)][intra_dir as usize][tx_idx.min(6)] as usize;
        return Ok(TX_TYPE_TO_TXTP[t] as u16);
    }

    // Inter luma transform type (AV2 Spec v1.0.0 §5.20.8.2 `transform_type`).
    if t_dim.max > 3 && t_dim.min >= 3 {
        return Ok(txtp::DCT_DCT as u16);
    }
    let (ctx, sqr) = crate::av2::entropy::inter_txtp_ctx(t_dim, eob);
    if p.tx == 3 {
        let dct = msac.decode_bool_adapt(cdf_m.txtp_inter_dct_idtx(ctx, sqr))? != 0;
        return Ok(if dct {
            txtp::DCT_DCT as u16
        } else {
            txtp::IDTX as u16
        });
    }
    if t_dim.max >= 3 {
        let long_dct = if t_dim.max == 4 {
            1usize
        } else {
            msac.decode_bool_adapt(cdf_m.is_long_side_dct(1))? as usize
        };
        let sym = msac.decode_symbol_adapt(cdf_m.inter_tx_type_long(ctx, sqr), 3)?;
        let orient = (t_dim.w < t_dim.h) as usize;
        let t = TX_TYPE_INV_LONG[long_dct][orient][sym.min(3)] as usize;
        return Ok(TX_TYPE_TO_TXTP[t] as u16);
    }
    let kernel = match p.reduced_txtp_set {
        1 | 2 => {
            let dct = msac
                .decode_bool_adapt(&mut cdf_m.inter_tx_type_set3[ctx][t_dim.min as usize])?
                != 0;
            if dct { txtp::DCT_DCT } else { txtp::IDTX }
        }
        3 => {
            let sym = msac
                .decode_symbol_adapt(&mut cdf_m.inter_tx_type_set4[ctx][t_dim.min as usize], 3)?;
            let t = crate::av2::transform::TX_TYPE_INTER_INV_SET4[sym.min(3)] as usize;
            TX_TYPE_TO_TXTP[t]
        }
        _ => {
            if p.tx == 2 {
                // `TX_SET_INTER_2` (`Tx_Type_Inter_Inv_Set2`).
                let set = msac.decode_bool_adapt(&mut cdf_m.inter_tx_type_set2[ctx])? != 0;
                let idx = if !set {
                    msac.decode_symbol_adapt(&mut cdf_m.inter_tx_type_index_set2[ctx], 7)?
                } else {
                    8 + msac.decode_symbol_adapt(&mut cdf_m.inter_tx_type_offset_set2[ctx], 3)?
                };
                let t = TX_TYPE_INTER_INV_SET2[idx.min(11)] as usize;
                TX_TYPE_TO_TXTP[t]
            } else {
                // `TX_SET_INTER_1` (`Tx_Type_Inter_Inv_Set1`).
                let set = msac.decode_bool_adapt(
                    &mut cdf_m.inter_tx_type_set1[ctx][t_dim.min.min(1) as usize],
                )? != 0;
                let idx = if !set {
                    msac.decode_symbol_adapt(&mut cdf_m.inter_tx_type_index_set1[ctx], 7)?
                } else {
                    8 + msac.decode_symbol_adapt(&mut cdf_m.inter_tx_type_offset_set1[ctx], 7)?
                };
                let t = TX_TYPE_INTER_INV_SET1[idx.min(15)] as usize;
                TX_TYPE_TO_TXTP[t]
            }
        }
    };
    Ok(kernel as u16)
}

/// Decodes one transform block's coefficients into dequantized `dq_out`.
///
/// This is the inverse of [`crate::av2::entropy::encode_coefs`]. `a` and `l`
/// are the above/left level-context bytes (`res_ctx` history), starting at the
/// current 4x4 column/row.
#[allow(clippy::too_many_arguments)]
pub fn decode_coefs(
    msac: &mut MsacDecoder<'_>,
    coef: &mut CdfCoefContext,
    cdf_m: &mut CdfModeContext,
    a: &[u8],
    l: &[u8],
    p: &CoefDecodeParams,
    dq_out: &mut [i32],
) -> Result<CoefDecodeResult> {
    let t_dim = &TXFM_DIMENSIONS[p.tx];
    let chroma = p.plane != 0;
    let slw = imin(t_dim.lw as i32, 3) as usize;
    let slh = imin(t_dim.lh as i32, 3) as usize;
    let tx2dszctx = slw + slh;
    let n_coeffs = 16usize << tx2dszctx;
    // AV2 Spec v1.0.0 §7.14.4: at quantizer index 0 the dequantization is
    // `Round2(|q| * 64, QUANT_TABLE_BITS) == 8 * |q|`. `dequantize_coef` returns
    // the coded level for the lossless path, so apply the normative scale here.
    let dq_scale = if p.lossless {
        crate::av2::transform::LOSSLESS_DEQUANT_SCALE
    } else {
        1
    };
    let scan = SCANS[p.tx];
    let hi_to_low = if chroma { 1usize } else { 10usize };
    let stride = 4usize << slh;
    let shift = slh + 2;
    let mask = stride - 1;

    if dq_out.len() < n_coeffs {
        return Err(Av2Error::BitstreamError(
            "coefficient output buffer too small".to_string(),
        ));
    }

    // 1. all_zero flag.
    let sctx = get_skip_ctx(t_dim, p.bs, a, l, p.plane, p.u_has_cf, p.ss_hor, p.ss_ver) as usize;
    let all_zero = if p.plane == 2 {
        msac.decode_bool_adapt(coef.skip_v(sctx))?
    } else {
        let is_inter_or_fsc = (!p.intra || p.fsc) as usize;
        msac.decode_bool_adapt(coef.skip(is_inter_or_fsc, t_dim.ctx as usize, sctx))?
    };
    if all_zero != 0 {
        return Ok(CoefDecodeResult {
            eob: -1,
            txtp: if p.lossless {
                txtp::WHT_WHT as u16
            } else {
                txtp::DCT_DCT as u16
            },
            res_ctx: 0x40,
        });
    }

    // 2. EOB position.
    let eob_ctx = if chroma { 2usize } else { (!p.intra) as usize };
    let eob_bin_val: usize = match tx2dszctx {
        0 => msac.decode_symbol_adapt(coef.eob_bin_16(eob_ctx), 4)?,
        1 => msac.decode_symbol_adapt(coef.eob_bin_32(eob_ctx), 5)?,
        2 => msac.decode_symbol_adapt(coef.eob_bin_64(eob_ctx), 6)?,
        3 => msac.decode_symbol_adapt(coef.eob_bin_128(eob_ctx), 7)?,
        4 => {
            let mut v = msac.decode_symbol_adapt(coef.eob_bin_256(eob_ctx), 7)?;
            if v == 7 {
                v += msac.decode_bool_bypass() as usize;
            }
            v
        }
        5 => {
            let mut v = msac.decode_symbol_adapt(coef.eob_bin_512(eob_ctx), 7)?;
            if v == 7 {
                v += msac.decode_bools_bypass(2) as usize;
            }
            v
        }
        _ => {
            let mut v = msac.decode_symbol_adapt(coef.eob_bin_1024(eob_ctx), 7)?;
            if v == 7 {
                v += msac.decode_bools_bypass(2) as usize;
            }
            v
        }
    };
    let eob_idx: i32 = if eob_bin_val > 1 {
        let n = eob_bin_val - 2;
        let hi = msac.decode_bool_adapt(coef.eob_hi_bit())? as i32;
        let low = if n > 0 {
            msac.decode_bools_bypass(n as u32) as i32
        } else {
            0
        };
        (1i32 << (n + 1)) | (hi << n) | low
    } else {
        eob_bin_val as i32
    };
    if eob_idx < 0 || eob_idx as usize >= n_coeffs {
        return Err(Av2Error::BitstreamError(format!(
            "decoded EOB {eob_idx} outside the {n_coeffs}-coefficient block"
        )));
    }

    // 3. Transform kernel (derived; no syntax for the encoder's subset).
    let txtp = derive_txtp(msac, cdf_m, p, t_dim, eob_idx)?;
    let tx_class = ((txtp >> 3) & 3) as u8;
    let mut hr_avg = 0u32;
    let mut cul_level = 0u32;

    // 4. DC-only block.
    if eob_idx == 0 {
        let dc_sign_level: u32;
        let level: u32;
        if chroma {
            let sym = msac.decode_symbol_adapt(coef.eob_base_uv_tok_lf(0), 4)?;
            let token = (sym + 1) as u32;
            let sign = msac.decode_bool_bypass() != 0;
            dc_sign_level = if sign { 0 } else { 2 << 6 };
            level = if token >= 5 {
                token + msac.decode_quant_hr(0, &mut hr_avg)
            } else {
                token
            };
        } else {
            let sym = msac.decode_symbol_adapt(coef.eob_base_y_tok_lf(t_dim.ctx as usize, 0), 4)?;
            let token = if sym < 4 {
                (sym + 1) as u32
            } else {
                5 + msac.decode_symbol_adapt(coef.br_y_tok_lf(0), 3)? as u32
            };
            let dc_ctx = get_dc_sign_ctx(t_dim, a, l) as usize;
            let sign = msac.decode_bool_adapt(coef.dc_sign(0, 0, dc_ctx))? != 0;
            dc_sign_level = if sign { 0 } else { 2 << 6 };
            level = if token >= 8 {
                token + msac.decode_quant_hr(0, &mut hr_avg)
            } else {
                token
            };
        }
        let signed = if dc_sign_level == 0 {
            -(level as i32)
        } else {
            level as i32
        };
        dq_out[0] = dequantize_coef(
            signed, 0, 0, p.dq_tbl, None, t_dim.ctx, p.bitdepth, p.lossless,
        ) * dq_scale;
        let res_ctx = (level.min(63) as u8) | (dc_sign_level as u8);
        return Ok(CoefDecodeResult {
            eob: 0,
            txtp,
            res_ctx,
        });
    }

    let mut levels = [0i8; 1089];

    // 5. EOB base token.
    {
        let rc = scan[eob_idx as usize] as usize;
        let lf_eob = (eob_idx as usize) < hi_to_low;
        let ctx_init = 1
            + (eob_idx as usize > 2usize << tx2dszctx) as usize
            + (eob_idx as usize > 4usize << tx2dszctx) as usize;
        let token = if chroma {
            if lf_eob {
                let sym = msac.decode_symbol_adapt(coef.eob_base_uv_tok_lf(ctx_init), 4)?;
                (sym + 1) as u32
            } else {
                let sym = msac.decode_symbol_adapt(coef.eob_base_uv_tok_hf(ctx_init), 2)?;
                if sym == 2 {
                    3 + msac.decode_symbol_adapt(coef.br_uv_tok_hf(0), 3)? as u32
                } else {
                    (sym + 1) as u32
                }
            }
        } else if lf_eob {
            let sym =
                msac.decode_symbol_adapt(coef.eob_base_y_tok_lf(t_dim.ctx as usize, ctx_init), 4)?;
            if sym == 4 {
                5 + msac.decode_symbol_adapt(coef.br_y_tok_lf(7), 3)? as u32
            } else {
                (sym + 1) as u32
            }
        } else {
            let sym =
                msac.decode_symbol_adapt(coef.eob_base_y_tok_hf(t_dim.ctx as usize, ctx_init), 2)?;
            if sym == 2 {
                3 + msac.decode_symbol_adapt(coef.br_y_tok_hf(0), 3)? as u32
            } else {
                (sym + 1) as u32
            }
        };
        levels[rc] = token.min(127) as i8;
    }

    // 6. AC base tokens, high to low scan order.
    for i in (1..eob_idx as usize).rev() {
        let rc = scan[i] as usize;
        let x = rc >> shift;
        let y = rc & mask;
        let lf = i < hi_to_low;
        let mut hr_ctx = 0u32;
        let ctx = get_lo_ctx(
            &levels,
            rc,
            tx_class,
            &mut hr_ctx,
            (x + y) as u32,
            p.plane,
            stride,
        ) as usize;
        let token = if chroma {
            if lf {
                msac.decode_symbol_adapt(coef.base_uv_tok_lf(ctx), 5)? as u32
            } else {
                let sym = msac.decode_symbol_adapt(coef.base_uv_tok_hf(ctx), 3)?;
                if sym == 3 {
                    3 + msac.decode_symbol_adapt(coef.br_uv_tok_hf(hr_ctx as usize), 3)? as u32
                } else {
                    sym as u32
                }
            }
        } else if lf {
            let sym =
                msac.decode_symbol_adapt(coef.base_y_tok_lf(t_dim.ctx as usize, ctx, 0), 5)?;
            if sym == 5 {
                5 + msac.decode_symbol_adapt(coef.br_y_tok_lf(hr_ctx as usize), 3)? as u32
            } else {
                sym as u32
            }
        } else {
            let sym =
                msac.decode_symbol_adapt(coef.base_y_tok_hf(t_dim.ctx as usize, ctx, 0), 3)?;
            if sym == 3 {
                3 + msac.decode_symbol_adapt(coef.br_y_tok_hf(hr_ctx as usize), 3)? as u32
            } else {
                sym as u32
            }
        };
        levels[rc] = token.min(127) as i8;
    }

    // 7. DC base token.
    {
        let mut hr_ctx = 0u32;
        let ctx = get_lo_ctx(&levels, 0, tx_class, &mut hr_ctx, 0, p.plane, stride) as usize;
        let token = if chroma {
            msac.decode_symbol_adapt(coef.base_uv_tok_lf(ctx), 5)? as u32
        } else {
            let sym =
                msac.decode_symbol_adapt(coef.base_y_tok_lf(t_dim.ctx as usize, ctx, 0), 5)?;
            if sym == 5 {
                5 + msac.decode_symbol_adapt(coef.br_y_tok_lf(hr_ctx as usize), 3)? as u32
            } else {
                sym as u32
            }
        };
        levels[0] = token.min(127) as i8;
    }

    // 8. Signs and high-range residuals, high to low scan order.
    for i in (1..=eob_idx as usize).rev() {
        let rc = scan[i] as usize;
        let token = levels[rc] as u32;
        if token == 0 {
            continue;
        }
        let sign = msac.decode_bool_bypass() != 0;
        let max_br = if i < hi_to_low {
            if chroma { 5 } else { 8 }
        } else {
            6
        };
        let level = if token >= max_br {
            let hr = msac.decode_quant_hr(0, &mut hr_avg);
            token + hr
        } else {
            token
        };
        cul_level += level;
        let signed = if sign { -(level as i32) } else { level as i32 };
        dq_out[rc] = dequantize_coef(
            signed, i, rc, p.dq_tbl, None, t_dim.ctx, p.bitdepth, p.lossless,
        ) * dq_scale;
    }

    // 9. DC sign and residual.
    let dc_token = levels[0] as u32;
    let dc_sign_level: u32;
    if dc_token != 0 {
        let sign = if chroma {
            msac.decode_bool_bypass() != 0
        } else {
            let dc_ctx = get_dc_sign_ctx(t_dim, a, l) as usize;
            msac.decode_bool_adapt(coef.dc_sign(0, 0, dc_ctx))? != 0
        };
        dc_sign_level = if sign { 0 } else { 2 << 6 };
        let max_br = if chroma { 5 } else { 8 };
        let level = if dc_token >= max_br {
            dc_token + msac.decode_quant_hr(0, &mut hr_avg)
        } else {
            dc_token
        };
        cul_level += level;
        let signed = if sign { -(level as i32) } else { level as i32 };
        dq_out[0] = dequantize_coef(
            signed, 0, 0, p.dq_tbl, None, t_dim.ctx, p.bitdepth, p.lossless,
        ) * dq_scale;
    } else {
        dc_sign_level = 1 << 6;
    }

    Ok(CoefDecodeResult {
        eob: eob_idx,
        txtp,
        res_ctx: (cul_level.min(63) as u8) | (dc_sign_level as u8),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::cdf::CdfContext;
    use crate::av2::entropy::reset_context;
    use crate::av2::headers::{PixelLayout, SegmentationDataSet};
    use crate::av2::msac::MsacEncoder;
    use crate::av2::partition::{
        PartitionSyntaxConfig, encode_partition_decision, init_allowed_partitions,
    };
    use crate::av2::segment::{SegmentCoder, encode_delta_q, encode_segment_id};

    #[test]
    fn partition_decision_roundtrip_all_types() {
        let mut a = BlockContext::default();
        let mut l = BlockContext::default();
        reset_context(&mut a, true, false);
        reset_context(&mut l, true, false);

        let cfg = PartitionSyntaxConfig::new_with_ext(PixelLayout::I420, true);
        let allowed_64 = init_allowed_partitions(BlockSize::Bs64x64, 0, 0, 16, 16, false, &cfg);
        let allowed_128 = init_allowed_partitions(BlockSize::Bs128x128, 0, 0, 32, 32, false, &cfg);

        // Test Bs128x128 with Split and None
        for bp in [BlockPartition::None, BlockPartition::Split] {
            let mut msac_enc = MsacEncoder::new();
            let mut cdf_enc = CdfContext::default();
            encode_partition_decision(
                &mut msac_enc,
                &mut cdf_enc.m,
                &a,
                &l,
                BlockSize::Bs128x128,
                bp,
                0,
                0,
                &allowed_128,
            );
            let bytes = msac_enc.finalize();
            let mut msac_dec = MsacDecoder::new(&bytes, true);
            let mut cdf_dec = CdfContext::default();
            let decoded = decode_partition_decision(
                &mut msac_dec,
                &mut cdf_dec.m,
                &a,
                &l,
                BlockSize::Bs128x128,
                0,
                0,
                &allowed_128,
            )
            .expect("decode partition decision failed");
            assert_eq!(decoded, bp);
        }

        // Test Bs64x64 with all supported partitions
        for bp in [
            BlockPartition::None,
            BlockPartition::H,
            BlockPartition::V,
            BlockPartition::H3,
            BlockPartition::V3,
            BlockPartition::H4A,
            BlockPartition::H4B,
            BlockPartition::V4A,
            BlockPartition::V4B,
        ] {
            let mut msac_enc = MsacEncoder::new();
            let mut cdf_enc = CdfContext::default();
            encode_partition_decision(
                &mut msac_enc,
                &mut cdf_enc.m,
                &a,
                &l,
                BlockSize::Bs64x64,
                bp,
                0,
                0,
                &allowed_64,
            );
            let bytes = msac_enc.finalize();
            let mut msac_dec = MsacDecoder::new(&bytes, true);
            let mut cdf_dec = CdfContext::default();
            let decoded = decode_partition_decision(
                &mut msac_dec,
                &mut cdf_dec.m,
                &a,
                &l,
                BlockSize::Bs64x64,
                0,
                0,
                &allowed_64,
            )
            .expect("decode partition decision failed");
            assert_eq!(decoded, bp);
        }
    }

    fn seg_syntax(max_segments: usize, last_active: i16, temporal: bool) -> SegSyntax {
        SegSyntax {
            enabled: true,
            update_map: true,
            temporal,
            preskip: false,
            any_lossless: false,
            last_active_segid: last_active,
            max_segments,
        }
    }

    fn seg_dataset(last_active: usize) -> SegmentationDataSet {
        let mut sds = SegmentationDataSet::default();
        for i in 0..=last_active {
            sds.delta_q_mask |= 1 << i;
            sds.delta_q[i] = i as i16;
        }
        sds
    }

    #[test]
    fn segment_id_explicit_roundtrip_every_id_and_extension() {
        for (max_segments, last_active) in [(8usize, 7usize), (16, 15)] {
            for desired in 0..=last_active as u8 {
                let sds = seg_dataset(last_active);
                let coder = SegmentCoder::new(&sds, max_segments, false);
                let mut enc = MsacEncoder::new();
                let mut enc_cdf = CdfContext::default();
                let mut enc_map = SegmentMap::new(8, 8);
                let mut a = [0u8; 32];
                let mut l = [0u8; 32];
                let encoded = encode_segment_id(
                    &mut enc,
                    &mut enc_cdf.m,
                    &coder,
                    &mut enc_map,
                    None,
                    0,
                    0,
                    2,
                    2,
                    false,
                    false,
                    false,
                    desired,
                    &mut a,
                    &mut l,
                );
                assert_eq!(encoded, desired, "encoder id for {desired}");

                let bytes = enc.finalize();
                let mut dec = MsacDecoder::new(&bytes, true);
                let mut dec_cdf = CdfContext::default();
                let mut dec_map = SegmentMap::new(8, 8);
                let mut a = [0u8; 32];
                let mut l = [0u8; 32];
                let decoded = decode_segment_id(
                    &mut dec,
                    &mut dec_cdf.m,
                    &seg_syntax(max_segments, last_active as i16, false),
                    &mut dec_map,
                    None,
                    0,
                    0,
                    2,
                    2,
                    false,
                    false,
                    false,
                    false,
                    false,
                    &mut a,
                    &mut l,
                )
                .expect("decode segment id");
                assert_eq!(decoded, desired);
                assert_eq!(dec_map.get(0, 0), desired);
            }
        }
    }

    #[test]
    fn segment_id_spatial_prediction_roundtrip() {
        let sds = seg_dataset(7);
        let coder = SegmentCoder::new(&sds, 8, false);
        let mut enc = MsacEncoder::new();
        let mut enc_cdf = CdfContext::default();
        let mut enc_map = SegmentMap::new(8, 8);
        let mut ea = [0u8; 32];
        let mut el = [0u8; 32];
        for (bx, by, have_left, have_top, id) in
            [(0usize, 0usize, false, false, 3u8), (2, 0, true, false, 5)]
        {
            let got = encode_segment_id(
                &mut enc,
                &mut enc_cdf.m,
                &coder,
                &mut enc_map,
                None,
                bx,
                by,
                2,
                2,
                have_left,
                have_top,
                false,
                id,
                &mut ea,
                &mut el,
            );
            assert_eq!(got, id);
        }
        assert_eq!(enc_map.predict(2, 0, true, false), (3, 0));

        let bytes = enc.finalize();
        let mut dec = MsacDecoder::new(&bytes, true);
        let mut dec_cdf = CdfContext::default();
        let mut dec_map = SegmentMap::new(8, 8);
        let mut da = [0u8; 32];
        let mut dl = [0u8; 32];
        let syn = seg_syntax(8, 7, false);
        for (bx, by, have_left, have_top, id) in
            [(0usize, 0usize, false, false, 3u8), (2, 0, true, false, 5)]
        {
            let got = decode_segment_id(
                &mut dec,
                &mut dec_cdf.m,
                &syn,
                &mut dec_map,
                None,
                bx,
                by,
                2,
                2,
                have_left,
                have_top,
                false,
                false,
                false,
                &mut da,
                &mut dl,
            )
            .expect("decode segment id");
            assert_eq!(got, id);
        }
        assert_eq!(dec_map.predict(2, 0, true, false), (3, 0));
    }

    #[test]
    fn segment_id_skip_inference_codes_no_symbol() {
        let sds = seg_dataset(7);
        let coder = SegmentCoder::new(&sds, 8, false);
        let mut enc = MsacEncoder::new();
        let mut enc_cdf = CdfContext::default();
        let mut enc_map = SegmentMap::new(4, 4);
        let mut ea = [0u8; 32];
        let mut el = [0u8; 32];
        // Seed the spatial predictor with segment 2.
        encode_segment_id(
            &mut enc,
            &mut enc_cdf.m,
            &coder,
            &mut enc_map,
            None,
            0,
            0,
            2,
            2,
            false,
            false,
            false,
            2,
            &mut ea,
            &mut el,
        );
        let before = enc.tell_bits();
        let id = encode_segment_id(
            &mut enc,
            &mut enc_cdf.m,
            &coder,
            &mut enc_map,
            None,
            2,
            0,
            2,
            2,
            true,
            false,
            true,
            2,
            &mut ea,
            &mut el,
        );
        assert_eq!(id, 2);
        assert_eq!(enc.tell_bits(), before, "skip inference must code no bits");

        let bytes = enc.finalize();
        let mut dec = MsacDecoder::new(&bytes, true);
        let mut dec_cdf = CdfContext::default();
        let mut dec_map = SegmentMap::new(4, 4);
        let mut da = [0u8; 32];
        let mut dl = [0u8; 32];
        let syn = seg_syntax(8, 7, false);
        for (bx, skip) in [(0usize, false), (2, true)] {
            let got = decode_segment_id(
                &mut dec,
                &mut dec_cdf.m,
                &syn,
                &mut dec_map,
                None,
                bx,
                0,
                2,
                2,
                bx != 0,
                false,
                false,
                false,
                skip,
                &mut da,
                &mut dl,
            )
            .expect("decode segment id");
            assert_eq!(got, 2);
        }
    }

    #[test]
    fn segment_id_preskip_with_skip_feature_roundtrip() {
        let mut sds = seg_dataset(7);
        sds.skip_mask |= 1 << 2;
        let coder = SegmentCoder::new(&sds, 8, false);
        assert!(coder.preskip);
        let mut enc = MsacEncoder::new();
        let mut enc_cdf = CdfContext::default();
        let mut enc_map = SegmentMap::new(4, 4);
        let mut ea = [0u8; 32];
        let mut el = [0u8; 32];
        let id = encode_segment_id(
            &mut enc,
            &mut enc_cdf.m,
            &coder,
            &mut enc_map,
            None,
            0,
            0,
            2,
            2,
            false,
            false,
            false,
            2,
            &mut ea,
            &mut el,
        );
        assert_eq!(id, 2);
        let bytes = enc.finalize();
        let mut dec = MsacDecoder::new(&bytes, true);
        let mut dec_cdf = CdfContext::default();
        let mut dec_map = SegmentMap::new(4, 4);
        let mut da = [0u8; 32];
        let mut dl = [0u8; 32];
        let mut syn = seg_syntax(8, 7, false);
        syn.preskip = true;
        let got = decode_segment_id(
            &mut dec,
            &mut dec_cdf.m,
            &syn,
            &mut dec_map,
            None,
            0,
            0,
            2,
            2,
            false,
            false,
            false,
            true,
            false,
            &mut da,
            &mut dl,
        )
        .expect("decode preskip segment id");
        assert_eq!(got, 2);
    }

    #[test]
    fn segment_id_temporal_prediction_roundtrip() {
        let sds = seg_dataset(7);
        let coder = SegmentCoder::new(&sds, 8, true);
        let prev = SegmentMap::filled(4, 4, 3);
        let mut enc = MsacEncoder::new();
        let mut enc_cdf = CdfContext::default();
        let mut enc_map = SegmentMap::new(4, 4);
        let mut ea = [0u8; 32];
        let mut el = [0u8; 32];
        let id = encode_segment_id(
            &mut enc,
            &mut enc_cdf.m,
            &coder,
            &mut enc_map,
            Some(&prev),
            0,
            0,
            2,
            2,
            false,
            false,
            false,
            3,
            &mut ea,
            &mut el,
        );
        assert_eq!(id, 3);
        let bytes = enc.finalize();

        let mut dec = MsacDecoder::new(&bytes, true);
        let mut dec_cdf = CdfContext::default();
        let mut dec_map = SegmentMap::new(4, 4);
        let mut da = [0u8; 32];
        let mut dl = [0u8; 32];
        let got = decode_segment_id(
            &mut dec,
            &mut dec_cdf.m,
            &seg_syntax(8, 7, true),
            &mut dec_map,
            Some(&prev),
            0,
            0,
            2,
            2,
            false,
            false,
            false,
            false,
            false,
            &mut da,
            &mut dl,
        )
        .expect("decode temporal segment id");
        assert_eq!(got, 3);
    }

    #[test]
    fn delta_qindex_roundtrip_all_escape_ranges() {
        for res_log2 in [0u8, 2] {
            let step = 1i32 << res_log2;
            for abs in [
                0i32, 1, 6, 7, 8, 9, 12, 13, 20, 21, 36, 37, 68, 69, 132, 133,
            ] {
                for sign in [1i32, -1] {
                    let delta = sign * abs * step;
                    let mut enc = MsacEncoder::new();
                    let mut enc_cdf = CdfContext::default();
                    encode_delta_q(&mut enc, &mut enc_cdf.m.delta_q, delta, res_log2);
                    let bytes = enc.finalize();
                    let mut dec = MsacDecoder::new(&bytes, true);
                    let mut dec_cdf = CdfContext::default();
                    let got = decode_delta_qindex(&mut dec, &mut dec_cdf.m, 100, res_log2, 351)
                        .expect("decode delta q");
                    assert_eq!(
                        got,
                        iclip(100 + delta, 1, 351),
                        "delta {delta} res {res_log2}"
                    );
                }
            }
        }
    }
}
