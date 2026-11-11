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

//! **av2** — Independent AV2 video encoder and matching byte-stream decoder
//! written in pure safe Rust.
//!
//! [`Encoder`] produces AV2 temporal units; [`Decoder`] consumes them, keeps a
//! reference picture buffer, and reconstructs frames. Both halves share the
//! CDF tables and prediction/transform primitives so that a decoded stream is
//! bit-exact against the encoder reconstruction.

#![allow(missing_docs)]
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::field_reassign_with_default,
    clippy::manual_memcpy,
    clippy::chunks_exact_to_as_chunks
)]
// Imported AV2 foundation: documentation/style lint cleanup follows later.

pub mod ccso;
pub mod cdef;
pub mod cdf;
pub mod cfl;
pub mod coef_cdf;
pub mod container;
pub mod deblock;
pub mod dec;
pub mod entropy;
pub mod error;
pub mod filmgrain;
pub mod frame;
pub mod gdf;
pub mod gdf_tables;
pub mod gop;
pub mod headers;
pub mod inter;
pub mod intops;
pub mod ipred;
pub mod levels;
pub mod me;
pub mod msac;
pub mod obu;
pub mod partition;
pub mod pixel;
pub mod putbits;
pub mod quantizer;
pub mod ratectrl;
pub mod restoration;
pub mod scan;
pub mod screen;
pub mod segment;
pub mod tables;
#[cfg(test)]
mod test_support;
pub mod threading;
pub mod transform;
pub mod txpart;
pub mod warp;
pub mod wedge;
pub mod wiener_tables;

use std::sync::Arc;

use crate::accelerator::FrameMetricsProvider;
use crate::av2::container::RawFrame;
use crate::av2::deblock::DeblockFilter;
pub use crate::av2::dec::Decoder;
use crate::av2::error::{Av2Error, Result};
use crate::av2::filmgrain::write_film_grain_obu;
use crate::av2::frame::{Frame, FrameStats, ReferenceDpb, ToolConfig, encode_frame};
use crate::av2::gop::{detect_scenecut, plan_gop_frame};
use crate::av2::headers::{
    AdaptiveBoolean, FhFilmGrain, FrameHeader, FrameType, PRIMARY_REF_NONE, PixelLayout,
    SequenceHeader,
};
use crate::av2::intops::iclip;
use crate::av2::obu::{
    build_default_sequence_header, build_frame_header, write_seq_hdr_obu,
    write_temporal_delimiter_obu,
};
use crate::av2::pixel::{BitDepth8, BitDepth16, Pixel};
use crate::av2::ratectrl::{RateControlMode, RateController};
use crate::av2::segment::SegmentMap;

/// Configuration parameters for [`Encoder`].
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Active frame width in luma pixels (`1..=16384`).
    pub width: usize,
    /// Active frame height in luma pixels (`1..=16384`).
    pub height: usize,
    /// Sample bit depth (`8`, `10`, or `12`).
    pub bit_depth: u8,
    /// Chroma subsampling layout (`I400`, `I420`, `I422`, `I444`).
    pub layout: PixelLayout,
    /// Framerate numerator (e.g. `30` or `60`).
    pub framerate_num: u32,
    /// Framerate denominator (e.g. `1`).
    pub framerate_den: u32,
    /// Base quantizer index (`0..=255`, where `0` is lossless and `100` is high quality).
    pub base_qidx: u16,
    /// Rate control operating mode (`ConstantQp`, `Vbr`, `Cbr`).
    pub rate_control: RateControlMode,
    /// Keyframe interval in frames (`1` = all-intra, `> 1` = periodic keyframes with inter frames).
    pub keyint: usize,
    /// Encoder speed preset (`0..=10`, where `0` is highest quality and `10` is fastest).
    pub speed: u8,
    /// Whether to use `128x128` superblocks (`true`) or `64x64` superblocks (`false`).
    pub use_sb128: bool,
    /// Log2 number of tile columns (`0..=6`).
    pub log2_tile_cols: u8,
    /// Log2 number of tile rows (`0..=6`).
    pub log2_tile_rows: u8,
    /// Wave 2 & Wave 3 coding tool, loop filter, GOP, and threading configuration.
    pub tools: ToolConfig,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            bit_depth: 8,
            layout: PixelLayout::I420,
            framerate_num: 30,
            framerate_den: 1,
            base_qidx: 100,
            rate_control: RateControlMode::ConstantQp { base_qidx: 100 },
            keyint: 30,
            speed: 4,
            use_sb128: false,
            log2_tile_cols: 0,
            log2_tile_rows: 0,
            tools: ToolConfig::default(),
        }
    }
}

impl EncoderConfig {
    /// Validates the encoder configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.width > 16384 || self.height > 16384 {
            return Err(Av2Error::InvalidConfig(format!(
                "Invalid frame dimensions {}x{} (must be within 1..=16384)",
                self.width, self.height
            )));
        }
        if !matches!(self.bit_depth, 8 | 10 | 12) {
            return Err(Av2Error::InvalidConfig(format!(
                "Unsupported bit depth {} (must be 8, 10, or 12)",
                self.bit_depth
            )));
        }
        if self.base_qidx > 255 {
            return Err(Av2Error::InvalidConfig(format!(
                "Invalid base_qidx {} (must be 0..=255)",
                self.base_qidx
            )));
        }
        Ok(())
    }
}

/// Complete temporal unit packet emitted by [`Encoder::encode_raw_frame`].
#[derive(Debug, Clone)]
pub struct EncodedPacket<P: Pixel> {
    /// Presentation timestamp / frame index (`0, 1, 2, ...`).
    pub pts: u64,
    /// Frame coding type (`FrameType::Key` or `FrameType::Inter`).
    pub frame_type: FrameType,
    /// Complete concatenated AV2 OBU bitstream for this temporal unit (`TD` + optional `SeqHdr` + `Frame`).
    pub data: Vec<u8>,
    /// Quality and coding statistics for this frame.
    pub stats: FrameStats,
    /// In-loop reconstructed and post-filtered frame (`Y`, `U`, `V` planes).
    pub rec_frame: Frame<P>,
    /// Film grain configuration signalled for this frame (§5.18.10.1). The
    /// reconstruction itself stays clean; call
    /// [`filmgrain::apply_film_grain`] to produce the output picture.
    pub film_grain: FhFilmGrain,
}

/// Stateful AV2 video encoder for pixel type `P` (`u8` for 8-bit, `u16` for 10/12-bit).
pub struct Encoder<P: Pixel> {
    pub config: EncoderConfig,
    pub seqhdr: SequenceHeader,
    pub dpb: ReferenceDpb<P>,
    pub rc: RateController,
    pub prev_raw: Option<RawFrame<P>>,
    pub frames_encoded: u64,
    /// Segment map of the most recently encoded frame. Used as the temporal
    /// reference map when the next inter frame can load segment ids from its
    /// primary reference (AV2 Spec v1.0.0 §5.18.1 `load_previous_segment_ids`).
    pub prev_seg_map: Option<SegmentMap>,
    /// Frame type of the most recently encoded frame, used to decide whether the
    /// derived primary reference can supply a previous segment map.
    pub prev_frame_type: Option<FrameType>,
    metrics: Option<Arc<dyn FrameMetricsProvider>>,
    deblock: Option<Arc<dyn DeblockFilter<P>>>,
}

impl<P: Pixel> Encoder<P> {
    /// Creates a new stateful AV2 [`Encoder`] from the provided [`EncoderConfig`].
    pub fn new(config: EncoderConfig) -> Result<Self> {
        config.validate()?;
        let mut seqhdr = build_default_sequence_header(
            config.width as i32,
            config.height as i32,
            config.layout,
            config.bit_depth,
            config.use_sb128,
        );
        if config.tools.enable_intrabc || config.tools.enable_palette {
            seqhdr.screen_content_tools = AdaptiveBoolean::Adaptive;
        }
        if config.tools.enable_intrabc {
            // The reference MV bank can seed the IntraBC BVP stack with MVs
            // from earlier blocks. The encoder uses the fixed default BVP
            // candidates, so keep the bank out of the IntraBC stream.
            seqhdr.refmv_bank = false;
        }
        if config.tools.enable_newmv
            || config.tools.enable_switchable_interp
            || config.tools.enable_compound
        {
            // The encoder's DRL derives candidates from the spatial scan only.
            // The optional reference MV bank (§7.12.2.21) would inject
            // candidates that this encoder does not model, so it is disabled
            // in the sequence header for streams that signal NEWMV,
            // switchable subpel filters, or compound prediction.
            seqhdr.refmv_bank = false;
        }
        if config.tools.enable_compound && config.tools.enable_compound_masks {
            // `enable_masked_compound` opens `comp_group_idx`/`compound_type`
            // (§5.20.7.16); `enable_cwp` opens the distance-weighted
            // `cwp_idx` syntax for `NEARMV_NEARMV` / `JOINT_NEWMV`.
            seqhdr.masked_compound = true;
            seqhdr.cwp = true;
        }
        seqhdr.mrls = config.tools.enable_mrl;
        seqhdr.cfl = config.tools.enable_cfl && config.layout != PixelLayout::I400;
        seqhdr.cdef = config.tools.enable_cdef;
        seqhdr.gdf = config.tools.enable_gdf;
        seqhdr.ccso = config.tools.enable_ccso;
        seqhdr.restoration = config.tools.enable_restoration;
        if config.tools.enable_restoration {
            // PC Wiener and NS Wiener are both enabled so `RESTORE_SWITCHABLE`
            // is signallable; chroma PC Wiener is always disabled by syntax.
            seqhdr.rst_disable_mask = [0, 1];
        }
        seqhdr.film_grain_present = config.tools.enable_film_grain;
        if config.tools.enable_warped_motion {
            // `seq_enabled_motion_modes[]` indexes `INTERINTRA..EXTENDWARP`
            // (§5.4.6). All five AV2 v1.0.0 modes are enabled: the encoder
            // emits `INTERINTRA`, `WARPMV`/`WARP_NEWMV` + `DELTAWARP`,
            // `LOCALWARP` and `EXTENDWARP`.
            seqhdr.motion_modes = 1 | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
        }
        let rc = RateController::new(
            config.rate_control,
            config.width,
            config.height,
            config.framerate_num,
            config.framerate_den,
            config.keyint,
        );
        Ok(Self {
            config,
            seqhdr,
            dpb: ReferenceDpb::new(),
            rc,
            prev_raw: None,
            frames_encoded: 0,
            prev_seg_map: None,
            prev_frame_type: None,
            metrics: None,
            deblock: None,
        })
    }

    /// Routes frame-quality metrics through an optional accelerator.
    ///
    /// The bitstream is unaffected: only the `FrameStats` PSNR/MSE fields
    /// change. A provider error is logged and the CPU-computed statistics are
    /// kept, so a failing device can never corrupt a stream.
    pub fn set_frame_metrics(&mut self, provider: Arc<dyn FrameMetricsProvider>) {
        self.metrics = Some(provider);
    }

    /// Returns to the CPU-only metrics path.
    pub fn clear_frame_metrics(&mut self) {
        self.metrics = None;
    }

    /// Routes the in-loop deblocking filter through an optional accelerator.
    ///
    /// The filter is applied at the same point in the pipeline as the software
    /// path, and the bitstream is unaffected. A provider error is logged and
    /// the CPU filter runs instead.
    pub fn set_deblock_filter(&mut self, filter: Arc<dyn DeblockFilter<P>>) {
        self.deblock = Some(filter);
    }

    /// Returns to the CPU-only deblocking path.
    pub fn clear_deblock_filter(&mut self) {
        self.deblock = None;
    }

    /// Returns the standalone encoded `ObuType::SeqHdr` OBU bytes for this stream.
    pub fn sequence_header_obu(&self) -> Vec<u8> {
        write_seq_hdr_obu(&self.seqhdr)
    }

    /// Builds the [`FrameHeader`] and `(lambda_sad, lambda_rdoq)` for `raw` at `frame_idx`,
    /// incorporating Wave 3 Scene-Cut detection and Hierarchical Pyramid GOP scheduling.
    pub fn prepare_frame(&self, frame_idx: u64, raw: &RawFrame<P>) -> (FrameHeader, u64, u64) {
        let is_scenecut = if self.config.tools.enable_scenecut && frame_idx > 0 {
            if let Some(ref prev) = self.prev_raw {
                detect_scenecut(prev, raw)
            } else {
                false
            }
        } else {
            false
        };

        // Lossless coding is intra-only in this encoder: an inter frame would
        // have to signal `lossless_tx_size` and code its residual as 4x4 WHT
        // blocks, which the encoder does not model yet. Every frame of a
        // quantizer-index-0 stream is therefore a key frame.
        let lossless = matches!(
            self.config.rate_control,
            RateControlMode::ConstantQp { base_qidx: 0 }
        ) || self.config.base_qidx == 0;
        let keyint = if lossless { 1 } else { self.config.keyint };
        let gop_plan = plan_gop_frame(
            frame_idx,
            keyint,
            self.config.tools.enable_hierarchical_gop,
            is_scenecut,
            self.seqhdr.ref_frames,
        );

        let (base_qidx, lambda_sad, lambda_rdoq) = self
            .rc
            .begin_frame(gop_plan.frame_type, self.config.bit_depth);
        let qidx = if base_qidx == 0 {
            0
        } else {
            iclip(base_qidx as i32 + gop_plan.qp_offset as i32, 1, 255) as u16
        };

        let poc = (frame_idx & 0xFF) as u8;
        let mut fh = build_frame_header(
            &self.seqhdr,
            gop_plan.frame_type,
            poc,
            qidx,
            self.config.log2_tile_cols,
            self.config.log2_tile_rows,
        );
        fh.reduced_txtp_set = self.config.tools.reduced_txtp_set.min(3);
        if self.config.tools.enable_warped_motion && fh.frame_type == FrameType::Inter {
            // `allow_warpmv_mode` (§5.18.x, written when
            // `frame_enabled_motion_modes[DELTAWARP]` is set).
            fh.warp_motion = 1;
        }
        fh.tlayer_id = gop_plan.tlayer_id;
        fh.refresh_frame_flags = gop_plan.refresh_frame_flags;
        fh.refidx[0] = gop_plan.ref_slots[0];
        fh.refidx[1] = gop_plan.ref_slots[1];
        if !self.config.tools.enable_compound && fh.frame_type == FrameType::Inter {
            fh.refidx[0] = 0;
            fh.refidx[1] = 0;
        }
        // Segment temporal prediction loads the map of the primary reference
        // (`load_previous_segment_ids`, AV2 Spec v1.0.0 §5.18.1). Signal list
        // index 0 explicitly when it points at the previous inter frame; this
        // keeps the derived primary reference and the stored map in sync.
        if fh.frame_type == FrameType::Inter
            && self.config.tools.enable_vaq
            && self.config.tools.enable_seg_temporal
            && self.prev_frame_type == Some(FrameType::Inter)
            && self.prev_seg_map.is_some()
            && fh.refidx[0] == (poc.wrapping_sub(1) % self.seqhdr.ref_frames.max(1)) as i8
        {
            fh.primary_ref_signaled = 1;
            fh.primary_ref_frame = 0;
        }
        // Screen-content frame header (`screen_content_params()` and
        // `intrabc_params()`, AV2 Spec v1.0.0 §5.18.3.3/§5.18.3.4).
        if self.config.tools.enable_palette {
            fh.allow_screen_content_tools = 1;
        }
        if self.config.tools.enable_intrabc && !fh.is_inter_or_switch() {
            fh.allow_screen_content_tools = 1;
            fh.allow_intrabc = 1;
            // Local-only IntraBC keeps the conformance rule simple and
            // deterministic (`allow_global_intrabc = 0` implies
            // `allow_local_intrabc = 1`).
            fh.allow_global_intrabc = 0;
            fh.allow_local_intrabc = 1;
            fh.force_integer_mv = 1;
            fh.max_bvp_drl_bits = self.seqhdr.def_max_bvp_drl_bits;
        } else if fh.allow_screen_content_tools != 0 {
            fh.force_integer_mv = 0;
        }
        (fh, lambda_sad, lambda_rdoq)
    }
}

impl Encoder<u8> {
    /// Encodes a single 8-bit [`RawFrame<u8>`] into an AV2 [`EncodedPacket<u8>`].
    pub fn encode_raw_frame(&mut self, raw: &RawFrame<u8>) -> Result<EncodedPacket<u8>> {
        if raw.width != self.config.width || raw.height != self.config.height {
            return Err(Av2Error::InvalidDimension {
                width: raw.width,
                height: raw.height,
            });
        }
        let pts = self.frames_encoded;
        let (mut frame_hdr, lambda_sad, lambda_rdoq) = self.prepare_frame(pts, raw);
        let src_frame = Frame::from_raw(raw, frame_hdr.frame_offset);
        let mut rec_frame = Frame::new(
            self.config.width,
            self.config.height,
            8,
            self.config.layout,
            frame_hdr.frame_offset,
        );

        let seg_ref = if frame_hdr.primary_ref_frame != PRIMARY_REF_NONE {
            self.prev_seg_map.as_ref()
        } else {
            None
        };
        let frame_budget = self.rc.frame_budget_bits(frame_hdr.frame_type);
        let (frame_obu, mut stats, seg_map) = encode_frame(
            BitDepth8,
            &self.seqhdr,
            &mut frame_hdr,
            self.config.tools,
            &src_frame,
            &mut rec_frame,
            &mut self.dpb,
            self.config.speed,
            lambda_sad,
            lambda_rdoq,
            self.deblock.as_deref(),
            seg_ref,
            frame_budget,
        );

        if let Some(provider) = &self.metrics {
            let reference = [
                src_frame.planes[0].view(),
                src_frame.planes[1].view(),
                src_frame.planes[2].view(),
            ];
            let distorted = [
                rec_frame.planes[0].view(),
                rec_frame.planes[1].view(),
                rec_frame.planes[2].view(),
            ];
            match provider.frame_metrics(reference, distorted) {
                Ok(metrics) => {
                    let [psnr_y, psnr_u, psnr_v] = metrics.psnr(8);
                    stats.mse_y = metrics.mse_y();
                    stats.psnr_y = psnr_y;
                    stats.psnr_u = psnr_u;
                    stats.psnr_v = psnr_v;
                    stats.psnr_avg = metrics.psnr_average(8);
                }
                Err(error) => {
                    tracing::warn!(%error, "frame metrics accelerator failed; keeping CPU statistics");
                }
            }
        }

        let mut tu_bytes = write_temporal_delimiter_obu();
        if frame_hdr.frame_type == FrameType::Key {
            tu_bytes.extend_from_slice(&write_seq_hdr_obu(&self.seqhdr));
        }
        if frame_hdr.film_grain.present != 0 {
            // The model must be available before the frame that references it
            // (§7.3.8.8); send it in the same temporal unit, ahead of the frame.
            tu_bytes.extend_from_slice(&write_film_grain_obu(
                &frame_hdr.film_grain.data,
                self.config.layout,
                frame_hdr.film_grain.id,
                frame_hdr.tlayer_id,
                frame_hdr.mlayer_id,
                frame_hdr.xlayer_id,
            ));
        }
        tu_bytes.extend_from_slice(&frame_obu);
        stats.bytes = tu_bytes.len();
        self.rc
            .end_frame(frame_hdr.frame_type, stats.avg_qidx, stats.bytes);
        self.prev_seg_map = if frame_hdr.segmentation.enabled != 0 {
            Some(seg_map)
        } else {
            None
        };
        self.prev_frame_type = Some(frame_hdr.frame_type);
        if self.config.tools.enable_scenecut {
            self.prev_raw = Some(raw.clone());
        }
        self.frames_encoded += 1;

        Ok(EncodedPacket {
            pts,
            frame_type: frame_hdr.frame_type,
            data: tu_bytes,
            stats,
            rec_frame,
            film_grain: frame_hdr.film_grain,
        })
    }
}

impl Encoder<u16> {
    /// Encodes a single 10-bit or 12-bit [`RawFrame<u16>`] into an AV2 [`EncodedPacket<u16>`].
    pub fn encode_raw_frame(&mut self, raw: &RawFrame<u16>) -> Result<EncodedPacket<u16>> {
        if raw.width != self.config.width || raw.height != self.config.height {
            return Err(Av2Error::InvalidDimension {
                width: raw.width,
                height: raw.height,
            });
        }
        let pts = self.frames_encoded;
        let (mut frame_hdr, lambda_sad, lambda_rdoq) = self.prepare_frame(pts, raw);
        let src_frame = Frame::from_raw(raw, frame_hdr.frame_offset);
        let mut rec_frame = Frame::new(
            self.config.width,
            self.config.height,
            self.config.bit_depth,
            self.config.layout,
            frame_hdr.frame_offset,
        );
        let bd = BitDepth16::new(self.config.bit_depth);

        let seg_ref = if frame_hdr.primary_ref_frame != PRIMARY_REF_NONE {
            self.prev_seg_map.as_ref()
        } else {
            None
        };
        let frame_budget = self.rc.frame_budget_bits(frame_hdr.frame_type);
        let (frame_obu, mut stats, seg_map) = encode_frame(
            bd,
            &self.seqhdr,
            &mut frame_hdr,
            self.config.tools,
            &src_frame,
            &mut rec_frame,
            &mut self.dpb,
            self.config.speed,
            lambda_sad,
            lambda_rdoq,
            None,
            seg_ref,
            frame_budget,
        );

        let mut tu_bytes = write_temporal_delimiter_obu();
        if frame_hdr.frame_type == FrameType::Key {
            tu_bytes.extend_from_slice(&write_seq_hdr_obu(&self.seqhdr));
        }
        if frame_hdr.film_grain.present != 0 {
            // The model must be available before the frame that references it
            // (§7.3.8.8); send it in the same temporal unit, ahead of the frame.
            tu_bytes.extend_from_slice(&write_film_grain_obu(
                &frame_hdr.film_grain.data,
                self.config.layout,
                frame_hdr.film_grain.id,
                frame_hdr.tlayer_id,
                frame_hdr.mlayer_id,
                frame_hdr.xlayer_id,
            ));
        }
        tu_bytes.extend_from_slice(&frame_obu);
        stats.bytes = tu_bytes.len();
        self.rc
            .end_frame(frame_hdr.frame_type, stats.avg_qidx, stats.bytes);
        self.prev_seg_map = if frame_hdr.segmentation.enabled != 0 {
            Some(seg_map)
        } else {
            None
        };
        self.prev_frame_type = Some(frame_hdr.frame_type);
        if self.config.tools.enable_scenecut {
            self.prev_raw = Some(raw.clone());
        }
        self.frames_encoded += 1;

        Ok(EncodedPacket {
            pts,
            frame_type: frame_hdr.frame_type,
            data: tu_bytes,
            stats,
            rec_frame,
            film_grain: frame_hdr.film_grain,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// External oracle for release unit 9: the encoder reconstruction must
    /// match `dav2d` byte for byte for lossless WHT/DPCM streams and for
    /// streams that signal non-DCT transform types and reduced sets 0/1/3.
    ///
    /// `DAV2D` overrides the reference decoder path. The test is ignored by
    /// default because it needs the external binary.
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_unit9_encoder_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::IvfWriter;

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir().join("rustmpeg-u9enc");
        std::fs::create_dir_all(&dir).unwrap();

        // The pixel type is part of the encoder API (`Encoder<u8>` and
        // `Encoder<u16>` implement `encode_raw_frame` separately), so the case
        // body is a macro rather than a generic function.
        macro_rules! run_case {
            ($p:ty, $ser:ident, $w:expr, $h:expr, $bd:expr, $q:expr, $reduced:expr,
             $tx_rdo:expr, $dpcm:expr) => {{
                let (w, h, bd, q, reduced, tx_rdo, dpcm): (usize, usize, u8, u16, u8, bool, bool) =
                    ($w, $h, $bd, $q, $reduced, $tx_rdo, $dpcm);
                let tag = format!(
                    "u9_{w}x{h}_{bd}_{q}_r{reduced}_{}{}",
                    tx_rdo as u8, dpcm as u8
                );
                let cfg = EncoderConfig {
                    width: w,
                    height: h,
                    bit_depth: bd,
                    layout: PixelLayout::I420,
                    base_qidx: q,
                    rate_control: RateControlMode::ConstantQp { base_qidx: q },
                    keyint: 60,
                    tools: ToolConfig {
                        enable_tx_type_rdo: tx_rdo,
                        reduced_txtp_set: reduced,
                        enable_dpcm: dpcm,
                        ..ToolConfig::default()
                    },
                    ..EncoderConfig::default()
                };
                let ivf = dir.join(format!("{tag}.ivf"));
                let mut recon: Vec<Vec<u8>> = Vec::new();
                // (wht, dpcm_y, dpcm_uv, non_dct, intra, inter)
                let mut counts = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
                {
                    let mut writer = IvfWriter::new(
                        std::io::BufWriter::new(std::fs::File::create(&ivf).unwrap()),
                        w as u16,
                        h as u16,
                        30,
                        1,
                        3,
                    )
                    .unwrap();
                    let mut enc = Encoder::<$p>::new(cfg).unwrap();
                    for idx in 0..3 {
                        let raw =
                            RawFrame::<$p>::generate_synthetic(w, h, bd, PixelLayout::I420, idx);
                        let pkt = enc.encode_raw_frame(&raw).unwrap();
                        writer.write_packet(pkt.pts, &pkt.data).unwrap();
                        recon.push($ser(&pkt.rec_frame.to_raw()));
                        counts.0 += pkt.stats.wht_blocks;
                        counts.1 += pkt.stats.dpcm_y_blocks;
                        counts.2 += pkt.stats.dpcm_uv_blocks;
                        counts.3 += pkt.stats.non_dct_tx_blocks;
                        counts.4 += pkt.stats.intra_blocks;
                        counts.5 += pkt.stats.inter_blocks;
                    }
                    writer.flush().unwrap();
                }
                let out = dir.join(format!("{tag}.dav.y4m"));
                let status = Command::new(&dav2d)
                    .arg("-i")
                    .arg(&ivf)
                    .arg("-o")
                    .arg(&out)
                    .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
                    .status()
                    .expect("run dav2d");
                assert!(status.success(), "dav2d failed for {tag}");
                let dav = read_y4m_frames(&out);
                assert_eq!(dav.len(), recon.len(), "{tag}: frame count");
                for (i, (a, b)) in dav.iter().zip(recon.iter()).enumerate() {
                    assert_eq!(a, b, "{tag} frame {i}: dav2d mismatch");
                }
                counts
            }};
        }

        // Lossless WHT + DPCM at quantizer index 0, 8-bit and 10-bit.
        for (w, h) in [(64usize, 64usize), (256, 192)] {
            let (wht, dpcm_y, dpcm_uv, ..) = run_case!(u8, raw8_bytes, w, h, 8, 0, 2, false, true);
            assert!(wht > 0, "no WHT blocks at {w}x{h}");
            assert!(
                dpcm_y > 0 && dpcm_uv > 0,
                "no DPCM blocks at {w}x{h}: {:?}",
                (dpcm_y, dpcm_uv)
            );
        }
        run_case!(u16, raw16_bytes, 256, 192, 10, 0, 2, false, true);

        // Reduced transform sets 0/1/3 with transform-type RDO.
        for reduced in [0u8, 1, 3] {
            for q in [20u16, 100, 140] {
                run_case!(u8, raw8_bytes, 64, 64, 8, q, reduced, true, false);
                run_case!(u8, raw8_bytes, 256, 192, 8, q, reduced, true, false);
            }
        }
        run_case!(u16, raw16_bytes, 256, 192, 10, 100, 0, true, false);
        // Non-DCT selection is asserted on the 64x64 synthetic source, whose
        // 16x16 blocks carry a steep gradient plus an 8x8 detail pattern.
        let (_, _, _, non_dct, ..) = run_case!(u8, raw8_bytes, 64, 64, 8, 100, 0, true, false);
        assert!(non_dct > 0, "no non-DCT transform type selected");

        // 16x16 inter blocks signal `TX_SET_INTER_2` (reduced 0),
        // `TX_SET_DCT_IDTX` (reduced 1/2) and `TX_SET_DCT_IDTX_IDDCT`
        // (reduced 3) at 16x16 frames.
        for reduced in [0u8, 2, 3] {
            let (_, _, _, _, _, inter) =
                run_case!(u8, raw8_bytes, 16, 16, 8, 100, reduced, true, false);
            assert!(inter > 0, "no inter blocks at 16x16");
        }
    }

    /// External oracle for release unit 3: CCSO (band and edge offsets, `sb_reuse`)
    /// and GDF encoder reconstructions must match `dav2d` byte for byte.
    ///
    /// `DAV2D` overrides the reference decoder path. Ignored by default because
    /// it needs the external binary.
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_unit3_encoder_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::IvfWriter;

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir().join("rustmpeg-u3enc");
        std::fs::create_dir_all(&dir).unwrap();

        // (ccso_units, ccso_reuse_units, ccso_edge_planes, gdf_units, gdf_per_block_frames)
        macro_rules! run_case {
            ($p:ty, $ser:ident, $w:expr, $h:expr, $bd:expr, $q:expr, $layout:expr,
             $ccso:expr, $gdf:expr) => {{
                let (w, h, bd, q): (usize, usize, u8, u16) = ($w, $h, $bd, $q);
                let layout: PixelLayout = $layout;
                let tag = format!(
                    "u3_{}x{}_{}_{}_{}c{}g{}",
                    w,
                    h,
                    bd,
                    q,
                    match layout {
                        PixelLayout::I420 => "420",
                        PixelLayout::I422 => "422",
                        PixelLayout::I444 => "444",
                        PixelLayout::I400 => "400",
                    },
                    $ccso as u8,
                    $gdf as u8
                );
                let cfg = EncoderConfig {
                    width: w,
                    height: h,
                    bit_depth: bd,
                    layout,
                    base_qidx: q,
                    rate_control: RateControlMode::ConstantQp { base_qidx: q },
                    keyint: 60,
                    tools: ToolConfig {
                        enable_ccso: $ccso,
                        enable_ccso_sb_reuse: $ccso,
                        enable_gdf: $gdf,
                        ..ToolConfig::default()
                    },
                    ..EncoderConfig::default()
                };
                let ivf = dir.join(format!("{tag}.ivf"));
                let mut recon: Vec<Vec<u8>> = Vec::new();
                let mut counts = (0usize, 0usize, 0usize, 0usize, 0usize);
                {
                    let mut writer = IvfWriter::new(
                        std::io::BufWriter::new(std::fs::File::create(&ivf).unwrap()),
                        w as u16,
                        h as u16,
                        30,
                        1,
                        3,
                    )
                    .unwrap();
                    let mut enc = Encoder::<$p>::new(cfg).unwrap();
                    for idx in 0..3 {
                        let raw = RawFrame::<$p>::generate_synthetic(w, h, bd, layout, idx);
                        let pkt = enc.encode_raw_frame(&raw).unwrap();
                        writer.write_packet(pkt.pts, &pkt.data).unwrap();
                        recon.push($ser(&pkt.rec_frame.to_raw()));
                        counts.0 += pkt.stats.ccso_units;
                        counts.1 += pkt.stats.ccso_reuse_units;
                        counts.2 += pkt.stats.ccso_edge_planes;
                        counts.3 += pkt.stats.gdf_units;
                        counts.4 += pkt.stats.gdf_per_block as usize;
                    }
                    writer.flush().unwrap();
                }
                let out = dir.join(format!("{tag}.dav.y4m"));
                let status = Command::new(&dav2d)
                    .arg("-i")
                    .arg(&ivf)
                    .arg("-o")
                    .arg(&out)
                    .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
                    .status()
                    .expect("run dav2d");
                assert!(status.success(), "dav2d failed for {tag}");
                let dav = read_y4m_frames(&out);
                assert_eq!(dav.len(), recon.len(), "{tag}: frame count");
                for (i, (a, b)) in dav.iter().zip(recon.iter()).enumerate() {
                    assert_eq!(a, b, "{tag} frame {i}: dav2d mismatch");
                }
                counts
            }};
        }

        for (w, h) in [(64usize, 64usize), (256, 192)] {
            for q in [20u16, 100, 140] {
                for (ccso, gdf) in [(true, false), (false, true), (true, true)] {
                    let (ccso_units, reuse_units, edge_planes, gdf_units, per_block) =
                        run_case!(u8, raw8_bytes, w, h, 8, q, PixelLayout::I420, ccso, gdf);
                    if ccso {
                        assert!(ccso_units > 0, "{w}x{h} q{q}: no CCSO units");
                        assert!(reuse_units > 0, "{w}x{h} q{q}: no CCSO sb_reuse units");
                        assert!(edge_planes > 0, "{w}x{h} q{q}: no CCSO edge-offset plane");
                    }
                    if gdf {
                        assert!(gdf_units > 0, "{w}x{h} q{q}: no GDF units");
                        if w.max(h) > 128 {
                            assert!(per_block > 0, "{w}x{h} q{q}: no GDF per-block signalling");
                        }
                    }
                }
            }
        }

        for (bd, layout, ccso, gdf) in [
            (10u8, PixelLayout::I420, true, false),
            (10, PixelLayout::I420, false, true),
            (10, PixelLayout::I420, true, true),
        ] {
            let (ccso_units, reuse_units, edge_planes, gdf_units, _) =
                run_case!(u16, raw16_bytes, 256, 192, bd, 100, layout, ccso, gdf);
            if ccso {
                assert!(ccso_units > 0 && reuse_units > 0 && edge_planes > 0);
            }
            if gdf {
                assert!(gdf_units > 0);
            }
        }
        for layout in [PixelLayout::I422, PixelLayout::I444] {
            for (ccso, gdf) in [(true, false), (false, true), (true, true)] {
                let (ccso_units, reuse_units, edge_planes, gdf_units, _) =
                    run_case!(u8, raw8_bytes, 256, 192, 8, 100, layout, ccso, gdf);
                if ccso {
                    assert!(ccso_units > 0 && reuse_units > 0 && edge_planes > 0);
                }
                if gdf {
                    assert!(gdf_units > 0);
                }
            }
        }
    }

    #[test]
    fn test_encoder_selects_ccso_and_gdf_units() {
        // Encoder-side selection: CCSO must choose edge offsets, use
        // `sb_reuse` on inter frames, and GDF must use per-block signalling at
        // 256x192 (GDF block = 128x128, frame larger than one block).
        let cfg = EncoderConfig {
            width: 256,
            height: 192,
            bit_depth: 8,
            layout: PixelLayout::I420,
            base_qidx: 100,
            tools: ToolConfig {
                enable_ccso: true,
                enable_ccso_sb_reuse: true,
                enable_gdf: true,
                ..ToolConfig::default()
            },
            ..EncoderConfig::default()
        };
        let mut enc = Encoder::<u8>::new(cfg).unwrap();
        let mut ccso_units = 0usize;
        let mut reuse_units = 0usize;
        let mut edge_planes = 0usize;
        let mut gdf_units = 0usize;
        let mut gdf_per_block = 0usize;
        for idx in 0..3 {
            let raw = RawFrame::generate_synthetic(256, 192, 8, PixelLayout::I420, idx);
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            ccso_units += pkt.stats.ccso_units;
            reuse_units += pkt.stats.ccso_reuse_units;
            edge_planes += pkt.stats.ccso_edge_planes;
            gdf_units += pkt.stats.gdf_units;
            gdf_per_block += pkt.stats.gdf_per_block as usize;
        }
        assert!(ccso_units > 0, "no CCSO unit selected");
        assert!(reuse_units > 0, "no CCSO sb_reuse unit selected");
        assert!(edge_planes > 0, "no CCSO edge-offset plane selected");
        assert!(gdf_units > 0, "no GDF block selected");
        assert!(gdf_per_block > 0, "no GDF per-block signalling");
    }

    /// Release unit 1: deblocking must actually filter at the quantizers where
    /// the normative `qThr` is non-zero, must be a no-op at QP 20 (where
    /// `qThr == 0`), and the decoder must reproduce the encoder reconstruction
    /// for every layout and bit depth. The signaled level is only an apply
    /// flag; the filter strength is derived from `qindex` (AV2 Spec v1.0.0
    /// §7.17.5/§7.17.6), so there is no encoder-side level search.
    #[test]
    fn av2_unit1_deblock_changes_recon_and_levels() {
        macro_rules! case {
            ($p:ty, $ser:ident, $bd:expr, $layout:expr, $q:expr) => {{
                let (bd, layout, q): (u8, PixelLayout, u16) = ($bd, $layout, $q);
                let mut recon = [Vec::new(), Vec::new()];
                let mut level = 0u8;
                for (slot, enable) in recon.iter_mut().zip([true, false]) {
                    let cfg = EncoderConfig {
                        width: 64,
                        height: 64,
                        bit_depth: bd,
                        layout,
                        base_qidx: q,
                        rate_control: RateControlMode::ConstantQp { base_qidx: q },
                        keyint: 60,
                        tools: ToolConfig {
                            enable_deblock: enable,
                            ..ToolConfig::default()
                        },
                        ..EncoderConfig::default()
                    };
                    let mut enc = Encoder::<$p>::new(cfg).unwrap();
                    let raw = RawFrame::<$p>::generate_synthetic(64, 64, bd, layout, 0);
                    let pkt = enc.encode_raw_frame(&raw).unwrap();
                    *slot = $ser(&pkt.rec_frame.to_raw());
                    if enable {
                        level = pkt.stats.deblock_level;
                    }
                    let mut dec = Decoder::<$p>::new();
                    let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                    assert_eq!(
                        $ser(&decoded.frame),
                        *slot,
                        "deblock={enable} {bd}-bit {layout:?} q{q}: decoder != encoder recon"
                    );
                }
                if q >= 140 {
                    assert!(level > 0, "deblock not applied at {bd}-bit {layout:?} q{q}");
                    assert_ne!(
                        recon[0], recon[1],
                        "deblock did not change the recon at {bd}-bit {layout:?} q{q}"
                    );
                } else if q >= 100 {
                    // The apply flag is set, but the synthetic 64x64 source
                    // has no edge that survives the QP 100 `qThr`/`side`
                    // thresholds; the filtered-edge assertion is q140 plus the
                    // controlled step-edge unit test in `deblock.rs`.
                    assert!(level > 0, "deblock not applied at {bd}-bit {layout:?} q{q}");
                } else {
                    assert_eq!(level, 0, "unexpected deblock level at q{q}");
                    assert_eq!(
                        recon[0], recon[1],
                        "QP 20 must be a deblock no-op (qThr == 0)"
                    );
                }
            }};
        }

        for bd in [8u8, 10, 12] {
            for layout in [PixelLayout::I420, PixelLayout::I422, PixelLayout::I444] {
                for q in [20u16, 100, 140] {
                    if bd == 8 {
                        case!(u8, raw8_bytes, bd, layout, q);
                    } else {
                        case!(u16, raw16_bytes, bd, layout, q);
                    }
                }
            }
        }
    }

    /// Release unit 2: CDEF must actually filter (non-zero blocks) at the
    /// quantizers where the encoder selects a non-zero strength, must be
    /// reproducible by the decoder for every layout and bit depth, and the
    /// selected strengths and detected directions must vary across content.
    #[test]
    fn av2_unit2_cdef_changes_recon_and_varies() {
        let mut strengths = std::collections::BTreeSet::new();
        macro_rules! case {
            ($p:ty, $ser:ident, $bd:expr, $layout:expr, $q:expr, $w:expr, $h:expr) => {{
                let (bd, layout, q, w, h): (u8, PixelLayout, u16, usize, usize) =
                    ($bd, $layout, $q, $w, $h);
                let mut recon = [Vec::new(), Vec::new()];
                let mut strength = 0u8;
                for (slot, enable) in recon.iter_mut().zip([true, false]) {
                    let cfg = EncoderConfig {
                        width: w,
                        height: h,
                        bit_depth: bd,
                        layout,
                        base_qidx: q,
                        rate_control: RateControlMode::ConstantQp { base_qidx: q },
                        keyint: 60,
                        tools: ToolConfig {
                            enable_cdef: enable,
                            ..ToolConfig::default()
                        },
                        ..EncoderConfig::default()
                    };
                    let mut enc = Encoder::<$p>::new(cfg).unwrap();
                    let mut dec = Decoder::<$p>::new();
                    let mut frames = Vec::new();
                    for idx in 0..3 {
                        let raw = RawFrame::<$p>::generate_synthetic(w, h, bd, layout, idx);
                        let pkt = enc.encode_raw_frame(&raw).unwrap();
                        let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                        let reference = pkt.rec_frame.to_raw();
                        assert_eq!(
                            $ser(&decoded.frame),
                            $ser(&reference),
                            "cdef={enable} {w}x{h} {bd}-bit {layout:?} q{q}: decoder != encoder recon"
                        );
                        if enable {
                            strength = strength.max(pkt.stats.cdef_y_strength);
                        }
                        frames.push($ser(&reference));
                    }
                    *slot = frames;
                }
                if q >= 100 {
                    assert!(
                        strength > 0,
                        "CDEF did not select a strength at {w}x{h} {bd}-bit {layout:?} q{q}"
                    );
                    assert_ne!(
                        recon[0], recon[1],
                        "CDEF did not change the recon at {w}x{h} {bd}-bit {layout:?} q{q}"
                    );
                    strengths.insert(strength);
                }
            }};
        }

        for bd in [8u8, 10, 12] {
            for layout in [PixelLayout::I420, PixelLayout::I422, PixelLayout::I444] {
                for q in [20u16, 100, 140] {
                    if bd == 8 {
                        case!(u8, raw8_bytes, bd, layout, q, 64, 64);
                        case!(u8, raw8_bytes, bd, layout, q, 256, 192);
                    } else {
                        case!(u16, raw16_bytes, bd, layout, q, 64, 64);
                        case!(u16, raw16_bytes, bd, layout, q, 256, 192);
                    }
                }
            }
        }
        assert!(
            !strengths.is_empty(),
            "CDEF never selected a strength: {strengths:?}"
        );
        assert!(
            strengths.iter().all(|&s| s > 0),
            "CDEF selected a zero strength: {strengths:?}"
        );

        // Direction search must report more than one orientation on real
        // content: horizontal, vertical and diagonal edges.
        let mut plane = crate::av2::frame::Plane::<u8>::new(24, 8, 0, 0, 8);
        for y in 0..8 {
            for x in 0..24 {
                let v = match x / 8 {
                    0 => {
                        if y < 4 {
                            50
                        } else {
                            200
                        }
                    }
                    1 => {
                        if x % 8 < 4 {
                            50
                        } else {
                            200
                        }
                    }
                    _ => {
                        if y + (x % 8) < 7 {
                            40
                        } else {
                            220
                        }
                    }
                };
                plane.data[y * plane.stride + x] = v;
            }
        }
        let mut dirs = std::collections::BTreeSet::new();
        for bx in [0usize, 8, 16] {
            dirs.insert(crate::av2::cdef::find_cdef_direction(&plane, bx, 0, 8).0);
        }
        assert!(
            dirs.len() >= 3,
            "CDEF direction search did not vary: {dirs:?}"
        );
    }

    /// External oracle for release unit 1: encoder reconstruction with the
    /// deblocking filter enabled and disabled must match `dav2d` byte for byte
    /// for 8/10/12-bit, 4:2:0/4:2:2/4:4:4, QP 20/100/140 and 3 frames. The
    /// 130x66 case is restricted to 4:4:4 because `dav2d` crashes on the 4:2:0
    /// and 4:2:2 variants of this stream independently of deblocking.
    ///
    /// `DAV2D` overrides the reference decoder path. Ignored by default because
    /// it needs the external binary.
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_unit1_deblock_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::IvfWriter;

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir().join("rustmpeg-u1deblock");
        std::fs::create_dir_all(&dir).unwrap();

        macro_rules! case {
            ($p:ty, $ser:ident, $w:expr, $h:expr, $bd:expr, $layout:expr, $q:expr) => {{
                let (w, h, bd, layout, q): (usize, usize, u8, PixelLayout, u16) =
                    ($w, $h, $bd, $layout, $q);
                let mut recon = [Vec::new(), Vec::new()];
                let mut level = 0u8;
                for (slot, enable) in recon.iter_mut().zip([true, false]) {
                    let tag = format!("u1_{w}x{h}_{bd}_{layout:?}_q{q}_db{}", enable as u8);
                    let cfg = EncoderConfig {
                        width: w,
                        height: h,
                        bit_depth: bd,
                        layout,
                        base_qidx: q,
                        rate_control: RateControlMode::ConstantQp { base_qidx: q },
                        keyint: 60,
                        tools: ToolConfig {
                            enable_deblock: enable,
                            ..ToolConfig::default()
                        },
                        ..EncoderConfig::default()
                    };
                    let ivf = dir.join(format!("{tag}.ivf"));
                    let mut frames = Vec::new();
                    {
                        let mut writer = IvfWriter::new(
                            std::io::BufWriter::new(std::fs::File::create(&ivf).unwrap()),
                            w as u16,
                            h as u16,
                            30,
                            1,
                            3,
                        )
                        .unwrap();
                        let mut enc = Encoder::<$p>::new(cfg).unwrap();
                        for idx in 0..3 {
                            let raw = RawFrame::<$p>::generate_synthetic(w, h, bd, layout, idx);
                            let pkt = enc.encode_raw_frame(&raw).unwrap();
                            writer.write_packet(pkt.pts, &pkt.data).unwrap();
                            if enable {
                                level = level.max(pkt.stats.deblock_level);
                            }
                            frames.push($ser(&pkt.rec_frame.to_raw()));
                        }
                        writer.flush().unwrap();
                    }
                    *slot = frames;
                    let out = dir.join(format!("{tag}.dav.y4m"));
                    let status = Command::new(&dav2d)
                        .arg("-i")
                        .arg(&ivf)
                        .arg("-o")
                        .arg(&out)
                        .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
                        .status()
                        .expect("run dav2d");
                    assert!(status.success(), "dav2d failed for {tag}");
                    let dav = read_y4m_frames(&out);
                    assert_eq!(dav.len(), slot.len(), "{tag}: frame count");
                    for (i, (a, b)) in dav.iter().zip(slot.iter()).enumerate() {
                        assert_eq!(a, b, "{tag} frame {i}: dav2d mismatch");
                    }
                }
                if q >= 140 {
                    assert!(level > 0, "deblock not applied at {w}x{h} q{q}");
                    assert_ne!(
                        recon[0], recon[1],
                        "deblock did not change the recon at {w}x{h} {bd}-bit q{q}"
                    );
                } else if q >= 100 {
                    assert!(level > 0, "deblock not applied at {w}x{h} q{q}");
                } else {
                    // Rate control maps QP 20 to a small qindex whose `qThr`
                    // is still zero, so the filter is signalled but changes
                    // nothing.
                    assert_eq!(recon[0], recon[1], "QP 20 must be a no-op");
                }
            }};
        }

        for bd in [8u8, 10, 12] {
            for layout in [PixelLayout::I420, PixelLayout::I422, PixelLayout::I444] {
                for q in [20u16, 100, 140] {
                    if bd == 8 {
                        case!(u8, raw8_bytes, 64, 64, bd, layout, q);
                        case!(u8, raw8_bytes, 256, 192, bd, layout, q);
                    } else {
                        case!(u16, raw16_bytes, 64, 64, bd, layout, q);
                        case!(u16, raw16_bytes, 256, 192, bd, layout, q);
                    }
                }
                // 130x66 exercises the coded-extent (partial bottom-right
                // superblock) deblocking path. Only 4:4:4 is used here because
                // dav2d crashes on the 4:2:0/4:2:2 variants of this stream.
                if layout == PixelLayout::I444 {
                    for q in [20u16, 100, 140] {
                        if bd == 8 {
                            case!(u8, raw8_bytes, 130, 66, bd, layout, q);
                        } else {
                            case!(u16, raw16_bytes, 130, 66, bd, layout, q);
                        }
                    }
                }
            }
        }
    }

    /// External oracle for release unit 2: encoder reconstruction with CDEF
    /// enabled must match `dav2d` byte for byte for 8/10/12-bit,
    /// 4:2:0/4:2:2/4:4:4, QP 20/100/140, 64x64/256x192 and 3 frames. Deblocking
    /// is disabled so the comparison isolates CDEF. For QP 100/140 the test
    /// also proves that CDEF selected a non-zero strength and changed the
    /// reconstruction relative to the CDEF-off encode.
    ///
    /// `DAV2D` overrides the reference decoder path. Ignored by default because
    /// it needs the external binary.
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_unit2_cdef_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::IvfWriter;

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir().join("rustmpeg-u2cdef");
        std::fs::create_dir_all(&dir).unwrap();
        let mut strengths = std::collections::BTreeSet::new();

        macro_rules! encode_case {
            ($p:ty, $ser:ident, $w:expr, $h:expr, $bd:expr, $layout:expr, $q:expr, $enable:expr) => {{
                let (w, h, bd, layout, q, enable): (usize, usize, u8, PixelLayout, u16, bool) =
                    ($w, $h, $bd, $layout, $q, $enable);
                let tag = format!("u2_{w}x{h}_{bd}_{layout:?}_q{q}_cd{}", enable as u8);
                let cfg = EncoderConfig {
                    width: w,
                    height: h,
                    bit_depth: bd,
                    layout,
                    base_qidx: q,
                    rate_control: RateControlMode::ConstantQp { base_qidx: q },
                    keyint: 60,
                    tools: ToolConfig {
                        enable_cdef: enable,
                        ..ToolConfig::default()
                    },
                    ..EncoderConfig::default()
                };
                let ivf = dir.join(format!("{tag}.ivf"));
                let mut frames = Vec::new();
                let mut strength = 0u8;
                {
                    let mut writer = IvfWriter::new(
                        std::io::BufWriter::new(std::fs::File::create(&ivf).unwrap()),
                        w as u16,
                        h as u16,
                        30,
                        1,
                        3,
                    )
                    .unwrap();
                    let mut enc = Encoder::<$p>::new(cfg).unwrap();
                    let mut dec = Decoder::<$p>::new();
                    for idx in 0..3 {
                        let raw = RawFrame::<$p>::generate_synthetic(w, h, bd, layout, idx);
                        let pkt = enc.encode_raw_frame(&raw).unwrap();
                        writer.write_packet(pkt.pts, &pkt.data).unwrap();
                        let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                        let reference = pkt.rec_frame.to_raw();
                        assert_eq!(
                            $ser(&decoded.frame),
                            $ser(&reference),
                            "{tag} frame {idx}: decoder != encoder recon"
                        );
                        if enable {
                            strength = strength.max(pkt.stats.cdef_y_strength);
                        }
                        frames.push($ser(&reference));
                    }
                    writer.flush().unwrap();
                }
                let out = dir.join(format!("{tag}.dav.y4m"));
                let status = Command::new(&dav2d)
                    .arg("-i")
                    .arg(&ivf)
                    .arg("-o")
                    .arg(&out)
                    .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
                    .status()
                    .expect("run dav2d");
                assert!(status.success(), "dav2d failed for {tag}");
                let dav = read_y4m_frames(&out);
                assert_eq!(dav.len(), frames.len(), "{tag}: frame count");
                for (i, (a, b)) in dav.iter().zip(frames.iter()).enumerate() {
                    assert_eq!(a, b, "{tag} frame {i}: dav2d mismatch");
                }
                (frames, strength)
            }};
        }

        for bd in [8u8, 10, 12] {
            for layout in [PixelLayout::I420, PixelLayout::I422, PixelLayout::I444] {
                for q in [20u16, 100, 140] {
                    for (w, h) in [(64usize, 64usize), (256, 192)] {
                        let (on, strength) = if bd == 8 {
                            encode_case!(u8, raw8_bytes, w, h, bd, layout, q, true)
                        } else {
                            encode_case!(u16, raw16_bytes, w, h, bd, layout, q, true)
                        };
                        if q >= 100 {
                            assert!(
                                strength > 0,
                                "CDEF did not select a strength at {w}x{h} {bd}-bit {layout:?} q{q}"
                            );
                            let (off, _) = if bd == 8 {
                                encode_case!(u8, raw8_bytes, w, h, bd, layout, q, false)
                            } else {
                                encode_case!(u16, raw16_bytes, w, h, bd, layout, q, false)
                            };
                            assert_ne!(
                                on, off,
                                "CDEF did not change the recon at {w}x{h} {bd}-bit {layout:?} q{q}"
                            );
                            strengths.insert(strength);
                        }
                    }
                }
            }
        }
        assert!(
            !strengths.is_empty(),
            "CDEF never selected a strength: {strengths:?}"
        );
        assert!(
            strengths.iter().all(|&s| s > 0),
            "CDEF selected a zero strength: {strengths:?}"
        );
    }

    fn raw8_bytes(raw: &RawFrame<u8>) -> Vec<u8> {
        let mut out = Vec::with_capacity(raw.y.len() + raw.u.len() + raw.v.len());
        out.extend_from_slice(&raw.y);
        out.extend_from_slice(&raw.u);
        out.extend_from_slice(&raw.v);
        out
    }

    fn raw16_bytes(raw: &RawFrame<u16>) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 * (raw.y.len() + raw.u.len() + raw.v.len()));
        for plane in [&raw.y, &raw.u, &raw.v] {
            for &sample in plane {
                out.extend_from_slice(&sample.to_le_bytes());
            }
        }
        out
    }

    fn read_y4m_frames(path: &std::path::Path) -> Vec<Vec<u8>> {
        let data = std::fs::read(path).expect("read y4m");
        let mut pos = data.iter().position(|&b| b == b'\n').unwrap() + 1;
        let mut frames = Vec::new();
        while pos < data.len() {
            assert_eq!(&data[pos..pos + 5], b"FRAME");
            pos = data[pos..].iter().position(|&b| b == b'\n').unwrap() + pos + 1;
            let next = data[pos..]
                .windows(5)
                .position(|w| w == b"FRAME")
                .map(|p| p + pos)
                .unwrap_or(data.len());
            frames.push(data[pos..next].to_vec());
            pos = next;
        }
        frames
    }

    #[test]
    fn test_lossless_encoder_is_exact_and_selects_wht_and_dpcm() {
        // Quantizer index 0 is lossless (AV2 Spec v1.0.0 §5.18.1): the
        // reconstruction must equal the source exactly, every luma transform
        // block uses the 4x4 WHT, and the DPCM decision selects both planes.
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            bit_depth: 8,
            layout: PixelLayout::I420,
            base_qidx: 0,
            rate_control: RateControlMode::ConstantQp { base_qidx: 0 },
            tools: ToolConfig {
                enable_dpcm: true,
                ..ToolConfig::default()
            },
            ..EncoderConfig::default()
        };
        let mut enc = Encoder::<u8>::new(cfg).unwrap();
        let mut wht = 0usize;
        let mut dpcm_y = 0usize;
        let mut dpcm_uv = 0usize;
        for idx in 0..3 {
            let raw = RawFrame::generate_synthetic(64, 64, 8, PixelLayout::I420, idx);
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            assert_eq!(pkt.frame_type, FrameType::Key);
            for pl in 0..3 {
                let rec = &pkt.rec_frame.planes[pl];
                let src = raw_plane(&raw, pl);
                for r in 0..rec.height {
                    let row = &rec.data[r * rec.stride..r * rec.stride + rec.width];
                    let src_row = &src[r * rec.width..(r + 1) * rec.width];
                    assert_eq!(
                        row, src_row,
                        "lossless reconstruction differs on plane {pl}, row {r} of frame {idx}"
                    );
                }
            }
            wht += pkt.stats.wht_blocks;
            dpcm_y += pkt.stats.dpcm_y_blocks;
            dpcm_uv += pkt.stats.dpcm_uv_blocks;
        }
        assert!(wht > 0, "expected lossless WHT blocks, got {wht}");
        assert!(dpcm_y > 0, "expected luma DPCM blocks, got {dpcm_y}");
        assert!(dpcm_uv > 0, "expected chroma DPCM blocks, got {dpcm_uv}");
    }

    #[test]
    fn test_encoder_selects_non_dct_transform_types() {
        // With `reduced_tx_set = 0` and transform-type RDO the encoder must be
        // able to select a kernel other than `DCT_DCT` (§5.20.8.2/§5.20.8.3).
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            bit_depth: 8,
            layout: PixelLayout::I420,
            base_qidx: 100,
            tools: ToolConfig {
                enable_tx_type_rdo: true,
                reduced_txtp_set: 0,
                ..ToolConfig::default()
            },
            ..EncoderConfig::default()
        };
        let mut enc = Encoder::<u8>::new(cfg).unwrap();
        let mut non_dct = 0usize;
        for idx in 0..2 {
            let raw = RawFrame::generate_synthetic(64, 64, 8, PixelLayout::I420, idx);
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            non_dct += pkt.stats.non_dct_tx_blocks;
        }
        assert!(
            non_dct > 0,
            "expected non-DCT transform types, got {non_dct}"
        );
    }

    /// Copies one plane of a raw frame into a flat buffer for comparisons.
    fn raw_plane(raw: &RawFrame<u8>, plane: usize) -> Vec<u8> {
        match plane {
            0 => raw.y.clone(),
            1 => raw.u.clone(),
            _ => raw.v.clone(),
        }
    }

    #[test]
    fn test_encoder_newmv_and_switchable_stats() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            bit_depth: 8,
            layout: PixelLayout::I420,
            base_qidx: 20,
            tools: ToolConfig {
                enable_newmv: true,
                enable_switchable_interp: true,
                ..ToolConfig::default()
            },
            ..EncoderConfig::default()
        };
        let mut enc = Encoder::<u8>::new(cfg).unwrap();
        let mut newmv = 0usize;
        let mut subpel = 0usize;
        for idx in 0..3 {
            let mut raw = RawFrame::<u8>::new(64, 64, 8, PixelLayout::I420);
            let shift = idx as f32 * 0.5;
            for r in 0..64 {
                for c in 0..64 {
                    raw.y[r * 64 + c] = (((c as f32 + shift) * 4.0) as i32 % 256) as u8;
                }
            }
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            newmv += pkt.stats.newmv_blocks;
            subpel += pkt.stats.switchable_filter_blocks;
        }
        assert!(newmv > 0, "expected NEWMV blocks, got {newmv}");
        assert!(
            subpel > 0,
            "expected non-default switchable filters, got {subpel}"
        );
    }

    #[test]
    fn test_end_to_end_8bit_and_10bit_wave3_encoding_and_multithreaded_tiles() {
        let mut cfg8 = EncoderConfig {
            width: 128,
            height: 96,
            bit_depth: 8,
            layout: PixelLayout::I420,
            framerate_num: 30,
            framerate_den: 1,
            base_qidx: 80,
            rate_control: RateControlMode::Vbr {
                target_kbps: 450,
                max_qidx: 220,
            },
            keyint: 8,
            speed: 4,
            use_sb128: false,
            log2_tile_cols: 1,
            log2_tile_rows: 1,
            tools: ToolConfig {
                threads: 4,
                enable_intrabc: true,
                enable_palette: true,
                enable_film_grain: true,
                ..ToolConfig::default()
            },
        };
        let mut enc8 = Encoder::<u8>::new(cfg8.clone()).unwrap();
        for idx in 0..4 {
            let f = RawFrame::<u8>::generate_synthetic(128, 96, 8, PixelLayout::I420, idx);
            let pkt = enc8.encode_raw_frame(&f).unwrap();
            assert!(!pkt.data.is_empty());
            assert!(
                pkt.stats.psnr_y > 26.0,
                "Frame {idx} PSNR Y={}",
                pkt.stats.psnr_y
            );
        }

        // Verify 10-bit pipeline
        cfg8.bit_depth = 10;
        let mut enc10 = Encoder::<u16>::new(cfg8).unwrap();
        let f10_0 = RawFrame::<u16>::generate_synthetic(128, 96, 10, PixelLayout::I420, 0);
        let f10_1 = RawFrame::<u16>::generate_synthetic(128, 96, 10, PixelLayout::I420, 1);
        let pkt10_0 = enc10.encode_raw_frame(&f10_0).unwrap();
        let pkt10_1 = enc10.encode_raw_frame(&f10_1).unwrap();
        assert_eq!(pkt10_0.frame_type, FrameType::Key);
        assert_eq!(pkt10_1.frame_type, FrameType::Inter);
        assert!(pkt10_0.stats.psnr_y > 28.0);
        assert!(pkt10_1.stats.psnr_y > 26.0);
    }
}
