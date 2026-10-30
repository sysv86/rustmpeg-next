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

//! AV2 byte-stream decoder.
//!
//! The decoder consumes the OBU temporal units produced by
//! [`crate::av2::Encoder`] (or a conformant encoder using the same tool subset),
//! keeps a reference picture buffer, and returns reconstructed
//! [`RawFrame`]s. It is deliberately strict: syntax that the encoder cannot
//! currently produce is rejected with [`Av2Error::Unsupported`] instead of being
//! silently misdecoded.
//!
//! Supported subset:
//!
//! * 8/10/12-bit I420, I422, I444, and I400;
//! * intra and single-reference translational inter frames;
//! * compound inter prediction: reference pairs, `NEARMV_NEARMV` through
//!   `JOINT_NEWMV`, average, wedge, difference-weighted, and CWP blending
//!   (§5.20.7.6, §5.20.7.10–11, §5.20.7.16–18, §7.12.2, §7.13.3.27–30);
//! * deblock, CDEF, CCSO, GDF, and PC/NS Wiener loop restoration for luma and
//!   chroma, including switchable units and temporal filter reuse;
//! * 64x64 and 128x128 superblocks, square/rectangular partitions;
//! * DCT-based transforms with `reduced_txtp_set = 2`, WHT 4x4 lossless intra;
//! * MRL, DIP/filter-intra, CfL, and per-block DCT/idtx chroma kernels when the
//!   corresponding sequence flags are enabled;
//! * segmentation (`SEG_LVL_ALT_Q`, `SEG_LVL_SKIP`, `SEG_LVL_GLOBALMV`), spatial
//!   and temporal segment-map prediction, and per-superblock delta-Q;
//! * warped motion for `WARPMV` + `DELTAWARP` and `WARP_NEWMV` + `DELTAWARP`
//!   (§5.20.7.6–7, §7.12.2, §7.13.3.19). AV2 v1.0.0 has no OBMC tool.
//!
//! Film grain models are parsed from `OBU_FILM_GRAIN` (§5.14) and the output
//! picture receives the normative synthesis of §7.21.7; reference frames stay
//! clean (§7.21.1).
//!
//! Not yet decoded: inter-intra/local/extended warp modes, `NEWMV` residuals
//! outside the warp path, and switchable subpel filters. Lossless (QP 0)
//! streams are rejected because the encoder's current lossless transform path
//! is not reference-conformant.
//!
//! The decoder mirrors the encoder bitstream exactly. For streams produced by
//! other encoders only the syntax subset above is supported; unsupported syntax
//! is reported as [`Av2Error::Unsupported`] rather than silently misdecoded.

pub mod bitio;
pub mod frame;
pub mod hls;
pub mod msac;
pub mod syntax;

use crate::av2::container::RawFrame;
use crate::av2::dec::hls::RefFrameInfo;
use crate::av2::error::{Av2Error, Result};
use crate::av2::frame::{Frame, ReferenceDpb};
use crate::av2::headers::{
    FilmGrainData, FrameHeader, FrameType, NSWienerPlane, NUM_REF_FRAMES, ObuType,
    PRIMARY_REF_NONE, SequenceHeader,
};
use crate::av2::pixel::{BitDepth8, BitDepth16, Pixel};
use crate::av2::segment::SegmentMap;

/// Parsed tile group for one coded frame.
pub(crate) struct DecodedTileGroup<'a> {
    pub fh: FrameHeader,
    pub tiles: Vec<&'a [u8]>,
    pub n_cols: usize,
}

/// Finds and parses the coded frame in a temporal unit.
pub(crate) fn decode_group<'a>(
    seq: &SequenceHeader,
    obus: &[Obu<'a>],
    ref_info: &[RefFrameInfo; 16],
) -> Result<Option<DecodedTileGroup<'a>>> {
    let mut out = None;
    for obu in obus {
        if matches!(
            obu.obu_type,
            ObuType::ClosedLoopKf | ObuType::OpenLoopKf | ObuType::TileGrp
        ) {
            let (fh, tiles) = hls::parse_tile_group_with_refs(seq, obu, ref_info)?;
            let n_cols = fh.tiling.t.cols.max(1) as usize;
            out = Some(DecodedTileGroup { fh, tiles, n_cols });
        }
    }
    Ok(out)
}

/// One decoded frame with its timing and coding metadata.
#[derive(Debug, Clone)]
pub struct DecodedFrame<P: Pixel> {
    /// Reconstructed frame planes.
    pub frame: RawFrame<P>,
    /// Coding type of the frame.
    pub frame_type: FrameType,
    /// Frame order hint (`frame_offset`).
    pub poc: u8,
    /// Frame-level base quantizer index.
    pub qidx: u16,
}

/// Stateful AV2 decoder for pixel type `P`.
pub struct Decoder<P: Pixel> {
    seq: Option<SequenceHeader>,
    dpb: ReferenceDpb<P>,
    /// `SavedSegmentIds` per reference slot (AV2 Spec v1.0.0 §5.18.1).
    seg_maps: [Option<SegmentMap>; NUM_REF_FRAMES],
    /// Reference slot metadata for `DerivedPrimaryRefFrame`.
    ref_info: [RefFrameInfo; NUM_REF_FRAMES],
    /// Film grain models received in `OBU_FILM_GRAIN`, indexed by slot (§6.13).
    grain_models: [Option<FilmGrainData>; 8],
    /// Whether output-side film grain synthesis is applied (§7.21.1).
    apply_grain: bool,
    frames_decoded: u64,
}

impl<P: Pixel> Default for Decoder<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Pixel> Decoder<P> {
    /// Creates an empty decoder with a fresh reference picture buffer.
    pub fn new() -> Self {
        Self {
            seq: None,
            dpb: ReferenceDpb::new(),
            seg_maps: std::array::from_fn(|_| None),
            ref_info: [RefFrameInfo::default(); NUM_REF_FRAMES],
            grain_models: [None; 8],
            apply_grain: true,
            frames_decoded: 0,
        }
    }

    /// Enables or disables output-side film grain synthesis (§7.21.1).
    ///
    /// This mirrors the reference decoder's `apply_grain` option: the decoded
    /// picture and the reference frame store are unaffected, only the returned
    /// output picture changes. Enabled by default.
    pub fn set_apply_grain(&mut self, enabled: bool) {
        self.apply_grain = enabled;
    }

    /// Parses the sequence header and film grain OBUs of a temporal unit.
    fn update_headers(&mut self, obus: &[hls::Obu<'_>]) -> Result<()> {
        for obu in obus {
            if obu.obu_type == ObuType::SeqHdr {
                self.seq = Some(hls::parse_sequence_header(obu.payload)?);
            }
        }
        let Some(seq) = self.seq.as_ref() else {
            return Ok(());
        };
        for obu in obus {
            if obu.obu_type != ObuType::Fgm {
                continue;
            }
            let models = hls::parse_film_grain_obu(obu.payload, seq.layout)?;
            for (slot, model) in models.iter().enumerate() {
                if let Some(model) = model {
                    self.grain_models[slot] = Some(*model);
                }
            }
        }
        Ok(())
    }

    /// Resolves the film grain model and seed referenced by a frame header.
    fn frame_grain(&self, fh: &FrameHeader) -> Result<Option<(FilmGrainData, u16)>> {
        if fh.film_grain.present == 0 {
            return Ok(None);
        }
        let id = fh.film_grain.id as usize;
        let data = self
            .grain_models
            .get(id)
            .copied()
            .flatten()
            .ok_or_else(|| {
                Av2Error::BitstreamError(format!("film grain model {id} was not received"))
            })?;
        Ok(Some((data, fh.film_grain.seed as u16)))
    }

    /// Number of frames decoded so far.
    #[inline]
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Returns the active sequence header, if one has been parsed.
    #[inline]
    pub fn sequence_header(&self) -> Option<&SequenceHeader> {
        self.seq.as_ref()
    }

    fn previous_segment_map(&self, fh: &FrameHeader) -> Option<&SegmentMap> {
        if fh.segmentation.enabled == 0
            || fh.primary_ref_frame == PRIMARY_REF_NONE
            || fh.primary_ref_frame as usize >= fh.n_ref_frames as usize
        {
            return None;
        }
        let slot = fh.refidx[fh.primary_ref_frame as usize];
        if slot < 0 {
            return None;
        }
        self.seg_maps
            .get(slot as usize)
            .and_then(|map| map.as_ref())
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_frame(
        &mut self,
        rec: Frame<P>,
        grain: Option<(FilmGrainData, u16)>,
        frame_type: FrameType,
        poc: u8,
        qidx: u16,
        refresh_flags: u16,
        seg_map: SegmentMap,
        ccso_map: [Vec<bool>; 3],
        segmentation_enabled: bool,
        restoration_ns: [NSWienerPlane; 3],
    ) -> DecodedFrame<P> {
        // §7.21.1: grain is added to the output picture only. The frame that
        // enters the DPB is the clean reconstruction.
        let frame = match grain {
            Some((data, seed)) if self.apply_grain => {
                let mut out = rec.clone();
                crate::av2::filmgrain::apply_film_grain(&mut out, &data, seed);
                out.to_raw()
            }
            _ => rec.to_raw(),
        };
        self.dpb.update_with_owned_frame(refresh_flags, rec);
        self.dpb.update_ccso_maps(refresh_flags, &ccso_map);
        for (slot, entry) in self.seg_maps.iter_mut().enumerate() {
            if refresh_flags & (1u16 << slot) != 0 {
                *entry = if segmentation_enabled {
                    Some(seg_map.clone())
                } else {
                    None
                };
            }
        }
        let ref_is_intra = matches!(frame_type, FrameType::Key | FrameType::Intra);
        for (slot, entry) in self.ref_info.iter_mut().enumerate() {
            if refresh_flags & (1u16 << slot) != 0 {
                *entry = RefFrameInfo {
                    is_intra: ref_is_intra,
                    present: true,
                    qidx,
                    poc: poc as i32,
                    ns: restoration_ns,
                };
            }
        }
        self.frames_decoded += 1;
        DecodedFrame {
            frame,
            frame_type,
            poc,
            qidx,
        }
    }
}

impl Decoder<u8> {
    /// Decodes one temporal unit (OBU packet) into a reconstructed 8-bit frame.
    ///
    /// Returns `Ok(None)` when the packet contains no coded frame (for example a
    /// temporal delimiter or a standalone sequence header).
    pub fn decode_packet(&mut self, data: &[u8]) -> Result<Option<DecodedFrame<u8>>> {
        let obus = hls::split_obus(data)?;
        self.update_headers(&obus)?;
        let seq = self
            .seq
            .clone()
            .ok_or_else(|| Av2Error::BitstreamError("frame before sequence header".to_string()))?;
        let Some(group) = decode_group(&seq, &obus, &self.ref_info)? else {
            return Ok(None);
        };
        let grain = self.frame_grain(&group.fh)?;
        let prev_segmap = self.previous_segment_map(&group.fh);
        let (rec, seg_map, ccso_map) = frame::decode_frame(
            BitDepth8,
            &seq,
            &group.fh,
            &group.tiles,
            &self.dpb,
            group.n_cols,
            prev_segmap,
        )?;
        let out = self.finish_frame(
            rec,
            grain,
            group.fh.frame_type,
            group.fh.frame_offset,
            group.fh.quant.yac,
            group.fh.refresh_frame_flags,
            seg_map,
            ccso_map,
            group.fh.segmentation.enabled != 0,
            [
                group.fh.restoration.planes[0].ns,
                group.fh.restoration.planes[1].ns,
                group.fh.restoration.planes[2].ns,
            ],
        );
        Ok(Some(out))
    }
}

impl Decoder<u16> {
    /// Decodes one temporal unit (OBU packet) into a reconstructed high-bit-depth frame.
    pub fn decode_packet(&mut self, data: &[u8]) -> Result<Option<DecodedFrame<u16>>> {
        let obus = hls::split_obus(data)?;
        self.update_headers(&obus)?;
        let seq = self
            .seq
            .clone()
            .ok_or_else(|| Av2Error::BitstreamError("frame before sequence header".to_string()))?;
        let Some(group) = decode_group(&seq, &obus, &self.ref_info)? else {
            return Ok(None);
        };
        if !matches!(seq.hbd, 1 | 2) {
            return Err(Av2Error::InvalidInput(
                "16-bit decoder requires 10-bit or 12-bit AV2 input".to_string(),
            ));
        }
        let bd = BitDepth16::new(if seq.hbd == 2 { 12 } else { 10 });
        let grain = self.frame_grain(&group.fh)?;
        let prev_segmap = self.previous_segment_map(&group.fh);
        let (rec, seg_map, ccso_map) = frame::decode_frame(
            bd,
            &seq,
            &group.fh,
            &group.tiles,
            &self.dpb,
            group.n_cols,
            prev_segmap,
        )?;
        let out = self.finish_frame(
            rec,
            grain,
            group.fh.frame_type,
            group.fh.frame_offset,
            group.fh.quant.yac,
            group.fh.refresh_frame_flags,
            seg_map,
            ccso_map,
            group.fh.segmentation.enabled != 0,
            [
                group.fh.restoration.planes[0].ns,
                group.fh.restoration.planes[1].ns,
                group.fh.restoration.planes[2].ns,
            ],
        );
        Ok(Some(out))
    }
}

/// Convenience wrapper: decode every frame in an IVF file.
pub fn decode_ivf<R: std::io::Read>(reader: R) -> Result<Vec<DecodedFrame<u8>>> {
    let mut ivf = crate::av2::container::IvfReader::new(reader)?;
    let mut decoder = Decoder::<u8>::new();
    let mut out = Vec::new();
    while let Some(packet) = ivf.read_packet()? {
        if let Some(frame) = decoder.decode_packet(&packet.data)? {
            out.push(frame);
        }
    }
    Ok(out)
}

pub use hls::Obu;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::headers::PixelLayout;
    use crate::av2::{Encoder, EncoderConfig};

    fn assert_frames_equal(a: &RawFrame<u8>, b: &RawFrame<u8>) {
        assert_eq!(a.width, b.width);
        assert_eq!(a.height, b.height);
        assert_eq!(a.y, b.y, "luma mismatch");
        assert_eq!(a.u, b.u, "U mismatch");
        assert_eq!(a.v, b.v, "V mismatch");
    }

    fn roundtrip(width: usize, height: usize, qidx: u16, frames: usize) {
        let cfg = EncoderConfig {
            width,
            height,
            layout: PixelLayout::I420,
            bit_depth: 8,
            base_qidx: qidx,
            rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
            keyint: 1,
            ..EncoderConfig::default()
        };
        let mut enc = Encoder::<u8>::new(cfg).unwrap();
        let mut dec = Decoder::<u8>::new();
        for idx in 0..frames {
            let raw = RawFrame::<u8>::generate_synthetic(width, height, 8, PixelLayout::I420, idx);
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            let decoded = dec
                .decode_packet(&pkt.data)
                .unwrap()
                .unwrap_or_else(|| panic!("frame {idx} decoded to nothing"));
            assert_eq!(decoded.frame_type, pkt.frame_type);
            assert_eq!(decoded.qidx, pkt.stats.qidx);
            let reference = pkt.rec_frame.to_raw();
            assert_frames_equal(&decoded.frame, &reference);
        }
    }

    #[test]
    fn test_decode_matches_encoder_recon_intra_64() {
        roundtrip(64, 64, 100, 1);
    }

    #[test]
    fn test_decode_matches_encoder_recon_inter_64() {
        roundtrip(64, 64, 100, 3);
    }

    #[test]
    fn test_decode_matches_encoder_recon_intra_128() {
        roundtrip(128, 128, 100, 1);
    }

    fn roundtrip_layout(
        width: usize,
        height: usize,
        layout: PixelLayout,
        qidx: u16,
        frames: usize,
    ) {
        let cfg = EncoderConfig {
            width,
            height,
            layout,
            bit_depth: 8,
            base_qidx: qidx,
            rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
            keyint: if frames > 1 { 60 } else { 1 },
            ..EncoderConfig::default()
        };
        let mut enc = Encoder::<u8>::new(cfg).unwrap();
        let mut dec = Decoder::<u8>::new();
        for idx in 0..frames {
            let raw = RawFrame::<u8>::generate_synthetic(width, height, 8, layout, idx);
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
            assert_eq!(decoded.frame_type, pkt.frame_type);
            assert_frames_equal(&decoded.frame, &pkt.rec_frame.to_raw());
        }
    }

    /// Fills a frame with a pattern that makes the extended partition search
    /// prefer `family` at every 64x64 superblock: each family's four children
    /// get distinct constant luma values, tiled across the frame.
    fn fill_extended_pattern<P: crate::av2::pixel::Pixel>(
        frame: &mut RawFrame<P>,
        family: crate::av2::levels::BlockPartition,
    ) {
        use crate::av2::levels::BlockPartition;

        let w = frame.width;
        let h = frame.height;
        let mut put = |x0: usize, y0: usize, x1: usize, y1: usize, value: i32| {
            let x1 = x1.min(w);
            let y1 = y1.min(h);
            if x1 <= x0 || y1 <= y0 {
                return;
            }
            for r in y0..y1 {
                for c in x0..x1 {
                    frame.y[r * w + c] = P::from_i32(value);
                }
            }
        };
        for y in (0..h).step_by(64) {
            for x in (0..w).step_by(64) {
                match family {
                    BlockPartition::H3 => {
                        put(x, y, x + 64, y + 16, 40);
                        put(x, y + 16, x + 32, y + 48, 90);
                        put(x + 32, y + 16, x + 64, y + 48, 170);
                        put(x, y + 48, x + 64, y + 64, 230);
                    }
                    BlockPartition::V3 => {
                        put(x, y, x + 16, y + 64, 40);
                        put(x + 16, y, x + 48, y + 32, 90);
                        put(x + 16, y + 32, x + 48, y + 64, 170);
                        put(x + 48, y, x + 64, y + 64, 230);
                    }
                    BlockPartition::H4A => {
                        put(x, y, x + 64, y + 8, 40);
                        put(x, y + 8, x + 64, y + 24, 90);
                        put(x, y + 24, x + 64, y + 56, 170);
                        put(x, y + 56, x + 64, y + 64, 230);
                    }
                    BlockPartition::H4B => {
                        put(x, y, x + 64, y + 8, 40);
                        put(x, y + 8, x + 64, y + 40, 90);
                        put(x, y + 40, x + 64, y + 56, 170);
                        put(x, y + 56, x + 64, y + 64, 230);
                    }
                    BlockPartition::V4A => {
                        put(x, y, x + 8, y + 64, 40);
                        put(x + 8, y, x + 24, y + 64, 90);
                        put(x + 24, y, x + 56, y + 64, 170);
                        put(x + 56, y, x + 64, y + 64, 230);
                    }
                    BlockPartition::V4B => {
                        put(x, y, x + 8, y + 64, 40);
                        put(x + 8, y, x + 40, y + 64, 90);
                        put(x + 40, y, x + 56, y + 64, 170);
                        put(x + 56, y, x + 64, y + 64, 230);
                    }
                    _ => unreachable!("not an extended partition"),
                }
            }
        }
    }

    fn extended_partition_config(width: usize, height: usize, qidx: u16) -> EncoderConfig {
        EncoderConfig {
            width,
            height,
            layout: PixelLayout::I420,
            bit_depth: 8,
            base_qidx: qidx,
            rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
            keyint: 60,
            tools: crate::av2::frame::ToolConfig {
                enable_rect_partitions: true,
                enable_ext_partitions: true,
                enable_uneven_4way_partitions: true,
                ..Default::default()
            },
            ..EncoderConfig::default()
        }
    }

    /// Every extended partition family must be selected, decoded back to the
    /// encoder reconstruction byte-exactly, and its syntax must be exercised on
    /// both the 64x64 and 256x192 frame sizes (AV2 Spec v1.0.0 §5.20.3.2).
    #[test]
    fn test_decode_extended_partitions_roundtrip() {
        use crate::av2::levels::BlockPartition;

        for family in [
            BlockPartition::H3,
            BlockPartition::V3,
            BlockPartition::H4A,
            BlockPartition::H4B,
            BlockPartition::V4A,
            BlockPartition::V4B,
        ] {
            for &(width, height) in &[(64usize, 64usize), (256, 192)] {
                for qidx in [20u16, 100, 140] {
                    let cfg = extended_partition_config(width, height, qidx);
                    let mut enc = Encoder::<u8>::new(cfg).unwrap();
                    let mut dec = Decoder::<u8>::new();
                    let mut family_count = 0usize;
                    for idx in 0..3 {
                        let mut raw = RawFrame::<u8>::new(width, height, 8, PixelLayout::I420);
                        fill_extended_pattern(&mut raw, family);
                        let pkt = enc.encode_raw_frame(&raw).unwrap();
                        family_count += pkt.stats.partition_counts[family as usize];
                        let decoded = dec
                            .decode_packet(&pkt.data)
                            .unwrap()
                            .unwrap_or_else(|| panic!("frame {idx} decoded to nothing"));
                        assert_eq!(decoded.frame_type, pkt.frame_type);
                        assert_frames_equal(&decoded.frame, &pkt.rec_frame.to_raw());
                    }
                    assert!(
                        family_count > 0,
                        "{family:?} was never selected at {width}x{height} q{qidx}"
                    );
                }
            }
        }
    }

    /// The 10-bit path must carry the same extended syntax. The synthetic
    /// 64x64 source selects `V3`; decode must equal the encoder reconstruction.
    #[test]
    fn test_decode_extended_partitions_10bit_roundtrip() {
        use crate::av2::levels::BlockPartition;

        let width = 64usize;
        let height = 64usize;
        let qidx = 100u16;
        let mut cfg = extended_partition_config(width, height, qidx);
        cfg.bit_depth = 10;
        let mut enc = Encoder::<u16>::new(cfg).unwrap();
        let mut dec = Decoder::<u16>::new();
        let mut v3_count = 0usize;
        for _ in 0..3 {
            let raw = RawFrame::<u16>::generate_synthetic(width, height, 10, PixelLayout::I420, 0);
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            v3_count += pkt.stats.partition_counts[BlockPartition::V3 as usize];
            let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
            let reference = pkt.rec_frame.to_raw();
            assert_eq!(decoded.frame.y, reference.y, "luma mismatch");
            assert_eq!(decoded.frame.u, reference.u, "U mismatch");
            assert_eq!(decoded.frame.v, reference.v, "V mismatch");
        }
        assert!(v3_count > 0, "V3 was never selected in the 10-bit stream");
    }

    #[test]
    fn test_decode_applies_film_grain_to_output_only() {
        use crate::av2::filmgrain::apply_film_grain;
        use crate::av2::frame::ToolConfig;

        let qidx = 100u16;
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            layout: PixelLayout::I420,
            bit_depth: 8,
            base_qidx: qidx,
            rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
            keyint: 60,
            tools: ToolConfig {
                enable_film_grain: true,
                ..Default::default()
            },
            ..EncoderConfig::default()
        };
        let mut enc = Encoder::<u8>::new(cfg).unwrap();
        let mut grain_on = Decoder::<u8>::new();
        let mut grain_off = Decoder::<u8>::new();
        grain_off.set_apply_grain(false);

        for idx in 0..3 {
            let raw = RawFrame::<u8>::generate_synthetic(64, 64, 8, PixelLayout::I420, idx);
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            assert_eq!(
                pkt.film_grain.present, 1,
                "frame {idx} did not signal grain"
            );

            // The reference decoder's grained reconstruction is the clean
            // reconstruction plus §7.21.7 synthesis with the signalled model.
            let mut grained = pkt.rec_frame.clone();
            apply_film_grain(
                &mut grained,
                &pkt.film_grain.data,
                pkt.film_grain.seed as u16,
            );

            let on = grain_on.decode_packet(&pkt.data).unwrap().unwrap();
            let off = grain_off.decode_packet(&pkt.data).unwrap().unwrap();
            let clean = raw8_bytes(&pkt.rec_frame.to_raw());
            assert_eq!(
                raw8_bytes(&off.frame),
                clean,
                "frame {idx}: grain-off output != clean reconstruction"
            );
            assert_eq!(
                raw8_bytes(&on.frame),
                raw8_bytes(&grained.to_raw()),
                "frame {idx}: grained output != signalled-model synthesis"
            );
            assert_ne!(
                raw8_bytes(&on.frame),
                clean,
                "frame {idx}: film grain did not change the output"
            );
        }
    }

    macro_rules! roundtrip_deblock {
        ($p:ty, $width:expr, $height:expr, $layout:expr, $bit_depth:expr, $qidx:expr, $frames:expr, $tiles:expr) => {{
            let (width, height, layout, bit_depth, qidx): (usize, usize, PixelLayout, u8, u16) =
                ($width, $height, $layout, $bit_depth, $qidx);
            let cfg = EncoderConfig {
                width,
                height,
                layout,
                bit_depth,
                base_qidx: qidx,
                rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
                keyint: 60,
                log2_tile_cols: $tiles.0,
                log2_tile_rows: $tiles.1,
                tools: crate::av2::frame::ToolConfig {
                    enable_deblock: true,
                    ..Default::default()
                },
                ..EncoderConfig::default()
            };
            let mut enc = Encoder::<$p>::new(cfg).unwrap();
            let mut dec = Decoder::<$p>::new();
            for idx in 0..$frames {
                let raw = RawFrame::<$p>::generate_synthetic(width, height, bit_depth, layout, idx);
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                let reference = pkt.rec_frame.to_raw();
                assert!(decoded.frame.y == reference.y, "luma mismatch");
                assert!(decoded.frame.u == reference.u, "U mismatch");
                assert!(decoded.frame.v == reference.v, "V mismatch");
            }
        }};
    }

    #[test]
    fn test_decode_matches_encoder_recon_with_deblocking() {
        for layout in [PixelLayout::I420, PixelLayout::I422, PixelLayout::I444] {
            for q in [60u16, 140, 220] {
                roundtrip_deblock!(u8, 192, 128, layout, 8, q, 3, (0u8, 0u8));
            }
        }
        roundtrip_deblock!(u8, 256, 192, PixelLayout::I420, 8, 140, 2, (1u8, 1u8));
        roundtrip_deblock!(u16, 128, 128, PixelLayout::I420, 10, 140, 2, (0u8, 0u8));
        roundtrip_deblock!(u16, 128, 128, PixelLayout::I444, 12, 140, 2, (0u8, 0u8));
    }

    macro_rules! roundtrip_cdef {
        ($p:ty, $width:expr, $height:expr, $layout:expr, $bit_depth:expr, $qidx:expr, $frames:expr, $tiles:expr) => {{
            let (width, height, layout, bit_depth, qidx): (usize, usize, PixelLayout, u8, u16) =
                ($width, $height, $layout, $bit_depth, $qidx);
            let cfg = EncoderConfig {
                width,
                height,
                layout,
                bit_depth,
                base_qidx: qidx,
                rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
                keyint: 60,
                log2_tile_cols: $tiles.0,
                log2_tile_rows: $tiles.1,
                tools: crate::av2::frame::ToolConfig {
                    enable_cdef: true,
                    ..Default::default()
                },
                ..EncoderConfig::default()
            };
            let mut enc = Encoder::<$p>::new(cfg).unwrap();
            let mut dec = Decoder::<$p>::new();
            for idx in 0..$frames {
                let raw = RawFrame::<$p>::generate_synthetic(width, height, bit_depth, layout, idx);
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                let reference = pkt.rec_frame.to_raw();
                assert!(decoded.frame.y == reference.y, "luma mismatch");
                assert!(decoded.frame.u == reference.u, "U mismatch");
                assert!(decoded.frame.v == reference.v, "V mismatch");
            }
        }};
    }

    #[test]
    fn test_decode_matches_encoder_recon_with_cdef() {
        for layout in [PixelLayout::I420, PixelLayout::I422, PixelLayout::I444] {
            for q in [60u16, 140, 220] {
                roundtrip_cdef!(u8, 192, 128, layout, 8, q, 3, (0u8, 0u8));
            }
        }
        roundtrip_cdef!(u8, 256, 192, PixelLayout::I420, 8, 140, 2, (1u8, 1u8));
        roundtrip_cdef!(u16, 128, 128, PixelLayout::I420, 10, 140, 2, (0u8, 0u8));
        roundtrip_cdef!(u16, 128, 128, PixelLayout::I444, 12, 140, 2, (0u8, 0u8));
    }

    #[test]
    fn test_decode_matches_encoder_recon_with_ccso_and_gdf() {
        use crate::av2::frame::ToolConfig;

        for &(width, height) in &[(64usize, 64usize), (256usize, 192usize)] {
            let qidx = 100u16;
            let cfg = EncoderConfig {
                width,
                height,
                layout: PixelLayout::I420,
                bit_depth: 8,
                base_qidx: qidx,
                rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
                keyint: 60,
                tools: ToolConfig {
                    enable_ccso: true,
                    enable_ccso_sb_reuse: true,
                    enable_gdf: true,
                    ..Default::default()
                },
                ..EncoderConfig::default()
            };
            let mut enc = Encoder::<u8>::new(cfg).unwrap();
            let mut dec = Decoder::<u8>::new();
            let mut reuse_units = 0usize;
            for idx in 0..3 {
                let raw =
                    RawFrame::<u8>::generate_synthetic(width, height, 8, PixelLayout::I420, idx);
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                reuse_units += pkt.stats.ccso_reuse_units;
                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                assert_frames_equal(&decoded.frame, &pkt.rec_frame.to_raw());
            }
            assert!(
                reuse_units > 0,
                "{width}x{height}: no CCSO sb_reuse units decoded"
            );
        }
    }

    macro_rules! roundtrip_restoration {
        ($p:ty, $width:expr, $height:expr, $bit_depth:expr, $qidx:expr, $frames:expr, $mode:expr, $chroma:expr, $tiles:expr) => {{
            let (width, height, bit_depth, qidx): (usize, usize, u8, u16) =
                ($width, $height, $bit_depth, $qidx);
            let cfg = EncoderConfig {
                width,
                height,
                layout: PixelLayout::I420,
                bit_depth,
                base_qidx: qidx,
                rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
                keyint: 60,
                log2_tile_cols: $tiles.0,
                log2_tile_rows: $tiles.1,
                tools: crate::av2::frame::ToolConfig {
                    enable_restoration: true,
                    restoration_mode: $mode,
                    restoration_chroma: $chroma,
                    ..Default::default()
                },
                ..EncoderConfig::default()
            };
            let mut enc = Encoder::<$p>::new(cfg).unwrap();
            let mut dec = Decoder::<$p>::new();
            for idx in 0..$frames {
                let raw = RawFrame::<$p>::generate_synthetic(
                    width,
                    height,
                    bit_depth,
                    PixelLayout::I420,
                    idx,
                );
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                let reference = pkt.rec_frame.to_raw();
                assert!(decoded.frame.y == reference.y, "luma mismatch");
                assert!(decoded.frame.u == reference.u, "U mismatch");
                assert!(decoded.frame.v == reference.v, "V mismatch");
            }
        }};
    }

    #[test]
    fn test_decode_matches_encoder_recon_with_restoration() {
        use crate::av2::frame::RestorationMode;
        for mode in [
            RestorationMode::PcWiener,
            RestorationMode::NsWiener,
            RestorationMode::Switchable,
        ] {
            roundtrip_restoration!(u8, 64, 64, 8, 100, 3, mode, false, (0u8, 0u8));
            roundtrip_restoration!(u8, 256, 192, 8, 100, 3, mode, false, (0u8, 0u8));
        }
        roundtrip_restoration!(
            u8,
            64,
            64,
            8,
            100,
            3,
            RestorationMode::NsWiener,
            true,
            (0u8, 0u8)
        );
        roundtrip_restoration!(
            u8,
            256,
            192,
            8,
            100,
            3,
            RestorationMode::NsWiener,
            true,
            (0u8, 0u8)
        );
        roundtrip_restoration!(
            u16,
            64,
            64,
            10,
            100,
            3,
            RestorationMode::NsWiener,
            true,
            (0u8, 0u8)
        );
        roundtrip_restoration!(
            u8,
            256,
            192,
            8,
            100,
            3,
            RestorationMode::NsWiener,
            true,
            (1u8, 0u8)
        );
    }

    /// Byte-exact loop restoration comparison against dav2d: our decoder must
    /// equal the encoder reconstruction and `dav2d` for PC Wiener, NS Wiener,
    /// switchable units and NS luma+chroma. Ignored by default because it needs
    /// the oracle binary; run with:
    ///
    /// ```text
    /// cargo test -p rustmpeg-video av2_restoration_decoder_dav2d -- --ignored
    /// ```
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_restoration_decoder_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::IvfWriter;
        use crate::av2::frame::RestorationMode;

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir();
        struct Case {
            w: usize,
            h: usize,
            bd: u8,
            q: u16,
            tile_cols: u8,
            frames: usize,
            mode: RestorationMode,
            chroma: bool,
        }
        let modes = [
            RestorationMode::PcWiener,
            RestorationMode::NsWiener,
            RestorationMode::Switchable,
        ];
        let mut cases = Vec::new();
        for mode in modes {
            for (w, h) in [(64usize, 64usize), (256, 192)] {
                for q in [20u16, 100, 140] {
                    cases.push(Case {
                        w,
                        h,
                        bd: 8,
                        q,
                        tile_cols: 0,
                        frames: 3,
                        mode,
                        chroma: false,
                    });
                }
            }
        }
        for mode in modes {
            cases.push(Case {
                w: 64,
                h: 64,
                bd: 10,
                q: 100,
                tile_cols: 0,
                frames: 3,
                mode,
                chroma: false,
            });
        }
        for mode in modes {
            cases.push(Case {
                w: 64,
                h: 64,
                bd: 8,
                q: 100,
                tile_cols: 0,
                frames: 3,
                mode,
                chroma: true,
            });
            cases.push(Case {
                w: 256,
                h: 192,
                bd: 8,
                q: 100,
                tile_cols: 0,
                frames: 3,
                mode,
                chroma: true,
            });
        }
        cases.push(Case {
            w: 256,
            h: 192,
            bd: 8,
            q: 100,
            tile_cols: 1,
            frames: 3,
            mode: RestorationMode::NsWiener,
            chroma: true,
        });

        for (ci, case) in cases.iter().enumerate() {
            let Case {
                w,
                h,
                bd,
                q,
                tile_cols,
                frames,
                mode,
                chroma,
            } = *case;
            let ivf = dir.join(format!("u4d_{ci}_{w}x{h}_{bd}_{q}.ivf"));
            let dav_y4m = dir.join(format!("u4d_{ci}_{w}x{h}_{bd}_{q}.dav.y4m"));
            let cfg = EncoderConfig {
                width: w,
                height: h,
                layout: PixelLayout::I420,
                bit_depth: bd,
                base_qidx: q,
                rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: q },
                keyint: 60,
                log2_tile_cols: tile_cols,
                tools: crate::av2::frame::ToolConfig {
                    enable_restoration: true,
                    restoration_mode: mode,
                    restoration_chroma: chroma,
                    ..Default::default()
                },
                ..EncoderConfig::default()
            };
            let mut rec_frames: Vec<Vec<u8>> = Vec::new();
            let mut our_frames: Vec<Vec<u8>> = Vec::new();
            {
                let mut writer = IvfWriter::new(
                    std::io::BufWriter::new(std::fs::File::create(&ivf).unwrap()),
                    w as u16,
                    h as u16,
                    30,
                    1,
                    frames as u32,
                )
                .unwrap();
                if bd == 8 {
                    let mut enc = Encoder::<u8>::new(cfg).unwrap();
                    let mut dec = Decoder::<u8>::new();
                    for idx in 0..frames {
                        let raw =
                            RawFrame::<u8>::generate_synthetic(w, h, 8, PixelLayout::I420, idx);
                        let pkt = enc.encode_raw_frame(&raw).unwrap();
                        writer.write_packet(pkt.pts, &pkt.data).unwrap();
                        let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                        rec_frames.push(raw8_bytes(&pkt.rec_frame.to_raw()));
                        our_frames.push(raw8_bytes(&decoded.frame));
                    }
                } else {
                    let mut enc = Encoder::<u16>::new(cfg).unwrap();
                    let mut dec = Decoder::<u16>::new();
                    for idx in 0..frames {
                        let raw =
                            RawFrame::<u16>::generate_synthetic(w, h, bd, PixelLayout::I420, idx);
                        let pkt = enc.encode_raw_frame(&raw).unwrap();
                        writer.write_packet(pkt.pts, &pkt.data).unwrap();
                        let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                        rec_frames.push(raw16_bytes(&pkt.rec_frame.to_raw()));
                        our_frames.push(raw16_bytes(&decoded.frame));
                    }
                }
                writer.flush().unwrap();
            }
            for (i, (rec, ours)) in rec_frames.iter().zip(our_frames.iter()).enumerate() {
                assert_eq!(
                    rec, ours,
                    "{w}x{h} {bd}-bit QP {q} frame {i}: our decoder != encoder recon"
                );
            }
            let status = Command::new(&dav2d)
                .arg("-i")
                .arg(&ivf)
                .arg("-o")
                .arg(&dav_y4m)
                .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
                .status()
                .expect("run dav2d");
            assert!(status.success(), "dav2d failed for {w}x{h} {bd}-bit QP {q}");
            let dav = read_y4m_frames(&dav_y4m);
            assert_eq!(dav.len(), rec_frames.len());
            for (i, (rec, dav_frame)) in rec_frames.iter().zip(dav.iter()).enumerate() {
                assert_eq!(
                    rec, dav_frame,
                    "{w}x{h} {bd}-bit QP {q} frame {i}: dav2d != encoder recon"
                );
            }
        }
    }

    macro_rules! roundtrip_newmv {
        ($p:ty, $width:expr, $height:expr, $layout:expr, $bit_depth:expr, $qidx:expr, $frames:expr, $tiles:expr) => {{
            let (width, height, layout, bit_depth, qidx): (usize, usize, PixelLayout, u8, u16) =
                ($width, $height, $layout, $bit_depth, $qidx);
            let cfg = EncoderConfig {
                width,
                height,
                layout,
                bit_depth,
                base_qidx: qidx,
                rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
                keyint: 60,
                log2_tile_cols: $tiles.0,
                log2_tile_rows: $tiles.1,
                tools: crate::av2::frame::ToolConfig {
                    enable_newmv: true,
                    ..Default::default()
                },
                ..EncoderConfig::default()
            };
            let mut enc = Encoder::<$p>::new(cfg).unwrap();
            let mut dec = Decoder::<$p>::new();
            for idx in 0..$frames {
                let raw = RawFrame::<$p>::generate_synthetic(width, height, bit_depth, layout, idx);
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                let reference = pkt.rec_frame.to_raw();
                assert!(decoded.frame.y == reference.y, "luma mismatch");
                assert!(decoded.frame.u == reference.u, "U mismatch");
                assert!(decoded.frame.v == reference.v, "V mismatch");
            }
        }};
    }

    #[test]
    fn test_decode_matches_encoder_recon_with_newmv() {
        for layout in [PixelLayout::I420, PixelLayout::I422, PixelLayout::I444] {
            for q in [60u16, 140, 220] {
                roundtrip_newmv!(u8, 192, 128, layout, 8, q, 3, (0u8, 0u8));
            }
        }
        roundtrip_newmv!(u8, 256, 192, PixelLayout::I420, 8, 140, 2, (1u8, 1u8));
        roundtrip_newmv!(u16, 128, 128, PixelLayout::I420, 10, 140, 2, (0u8, 0u8));
        roundtrip_newmv!(u16, 128, 128, PixelLayout::I444, 12, 140, 2, (0u8, 0u8));
    }

    macro_rules! roundtrip_switchable {
        ($p:ty, $width:expr, $height:expr, $layout:expr, $bit_depth:expr, $qidx:expr, $frames:expr, $tiles:expr) => {{
            let (width, height, layout, bit_depth, qidx): (usize, usize, PixelLayout, u8, u16) =
                ($width, $height, $layout, $bit_depth, $qidx);
            let cfg = EncoderConfig {
                width,
                height,
                layout,
                bit_depth,
                base_qidx: qidx,
                rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
                keyint: 60,
                log2_tile_cols: $tiles.0,
                log2_tile_rows: $tiles.1,
                tools: crate::av2::frame::ToolConfig {
                    enable_switchable_interp: true,
                    ..Default::default()
                },
                ..EncoderConfig::default()
            };
            let mut enc = Encoder::<$p>::new(cfg).unwrap();
            let mut dec = Decoder::<$p>::new();
            for idx in 0..$frames {
                let raw = RawFrame::<$p>::generate_synthetic(width, height, bit_depth, layout, idx);
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                let reference = pkt.rec_frame.to_raw();
                assert!(decoded.frame.y == reference.y, "luma mismatch");
                assert!(decoded.frame.u == reference.u, "U mismatch");
                assert!(decoded.frame.v == reference.v, "V mismatch");
            }
        }};
    }

    #[test]
    fn test_decode_matches_encoder_recon_with_switchable_subpel() {
        for layout in [PixelLayout::I420, PixelLayout::I422, PixelLayout::I444] {
            for q in [60u16, 140, 220] {
                roundtrip_switchable!(u8, 192, 128, layout, 8, q, 3, (0u8, 0u8));
            }
        }
        roundtrip_switchable!(u8, 256, 192, PixelLayout::I420, 8, 140, 2, (1u8, 1u8));
        roundtrip_switchable!(u16, 128, 128, PixelLayout::I420, 10, 140, 2, (0u8, 0u8));
        roundtrip_switchable!(u16, 128, 128, PixelLayout::I444, 12, 140, 2, (0u8, 0u8));
    }

    macro_rules! roundtrip_newmv_and_switchable {
        ($p:ty, $width:expr, $height:expr, $layout:expr, $bit_depth:expr, $qidx:expr, $frames:expr, $tiles:expr) => {{
            let (width, height, layout, bit_depth, qidx): (usize, usize, PixelLayout, u8, u16) =
                ($width, $height, $layout, $bit_depth, $qidx);
            let cfg = EncoderConfig {
                width,
                height,
                layout,
                bit_depth,
                base_qidx: qidx,
                rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
                keyint: 60,
                log2_tile_cols: $tiles.0,
                log2_tile_rows: $tiles.1,
                tools: crate::av2::frame::ToolConfig {
                    enable_newmv: true,
                    enable_switchable_interp: true,
                    enable_deblock: true,
                    enable_cdef: true,
                    enable_ccso: true,
                    enable_gdf: true,
                    ..Default::default()
                },
                ..EncoderConfig::default()
            };
            let mut enc = Encoder::<$p>::new(cfg).unwrap();
            let mut dec = Decoder::<$p>::new();
            for idx in 0..$frames {
                let raw = RawFrame::<$p>::generate_synthetic(width, height, bit_depth, layout, idx);
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                let reference = pkt.rec_frame.to_raw();
                assert!(decoded.frame.y == reference.y, "luma mismatch");
                assert!(decoded.frame.u == reference.u, "U mismatch");
                assert!(decoded.frame.v == reference.v, "V mismatch");
            }
        }};
    }

    #[test]
    fn test_decode_matches_encoder_recon_with_newmv_and_switchable_and_filters() {
        for layout in [PixelLayout::I420, PixelLayout::I422, PixelLayout::I444] {
            for q in [60u16, 140, 220] {
                roundtrip_newmv_and_switchable!(u8, 192, 128, layout, 8, q, 3, (0u8, 0u8));
            }
        }
        roundtrip_newmv_and_switchable!(u8, 256, 192, PixelLayout::I420, 8, 140, 2, (1u8, 1u8));
        roundtrip_newmv_and_switchable!(u16, 128, 128, PixelLayout::I420, 10, 140, 2, (0u8, 0u8));
    }

    #[test]
    fn test_decode_matches_encoder_recon_inter_16px_superblock_remainder() {
        // A height with a 16-sample remainder leaves 64x16 blocks in the last
        // superblock row, whose inter luma transforms select `TX_SET_WIDE_64` in
        // `transform_type` and code a four-symbol `inter_tx_type`. The round trip
        // covers the encoder and decoder halves of that syntax.
        roundtrip_layout(64, 80, PixelLayout::I420, 220, 3);
        roundtrip_layout(128, 720, PixelLayout::I420, 20, 3);
    }

    fn segmentation_config(width: usize, height: usize, bit_depth: u8, qidx: u16) -> EncoderConfig {
        EncoderConfig {
            width,
            height,
            layout: PixelLayout::I420,
            bit_depth,
            base_qidx: qidx,
            rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
            keyint: 60,
            tools: crate::av2::frame::ToolConfig {
                enable_vaq: true,
                enable_delta_q: true,
                ..Default::default()
            },
            ..EncoderConfig::default()
        }
    }

    fn roundtrip_segmentation(width: usize, height: usize, bit_depth: u8, qidx: u16) {
        let cfg = segmentation_config(width, height, bit_depth, qidx);
        if bit_depth == 8 {
            let mut enc = Encoder::<u8>::new(cfg).unwrap();
            let mut dec = Decoder::<u8>::new();
            for idx in 0..3 {
                let raw =
                    RawFrame::<u8>::generate_synthetic(width, height, 8, PixelLayout::I420, idx);
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                assert_eq!(decoded.frame_type, pkt.frame_type);
                assert_frames_equal(&decoded.frame, &pkt.rec_frame.to_raw());
            }
        } else {
            let mut enc = Encoder::<u16>::new(cfg).unwrap();
            let mut dec = Decoder::<u16>::new();
            for idx in 0..3 {
                let raw = RawFrame::<u16>::generate_synthetic(
                    width,
                    height,
                    bit_depth,
                    PixelLayout::I420,
                    idx,
                );
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                assert_eq!(decoded.frame_type, pkt.frame_type);
                assert_eq!(decoded.frame.y, pkt.rec_frame.to_raw().y);
                assert_eq!(decoded.frame.u, pkt.rec_frame.to_raw().u);
                assert_eq!(decoded.frame.v, pkt.rec_frame.to_raw().v);
            }
        }
    }

    #[test]
    fn test_decode_matches_encoder_recon_with_segmentation_and_delta_q() {
        for &(w, h) in &[(64usize, 64usize), (256, 192)] {
            for &bd in &[8u8, 10, 12] {
                for &q in &[20u16, 100, 140] {
                    roundtrip_segmentation(w, h, bd, q);
                }
            }
        }
    }

    #[test]
    fn test_decode_matches_encoder_recon_with_temporal_segment_map() {
        let width = 64usize;
        let height = 64usize;
        let qidx = 100u16;
        let cfg = EncoderConfig {
            width,
            height,
            layout: PixelLayout::I420,
            bit_depth: 8,
            base_qidx: qidx,
            rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
            keyint: 60,
            tools: crate::av2::frame::ToolConfig {
                enable_vaq: true,
                enable_delta_q: true,
                enable_seg_temporal: true,
                enable_compound: true,
                ..Default::default()
            },
            ..EncoderConfig::default()
        };
        let mut enc = Encoder::<u8>::new(cfg).unwrap();
        let mut dec = Decoder::<u8>::new();
        let mut saw_temporal = false;
        for idx in 0..4 {
            let raw = RawFrame::<u8>::generate_synthetic(width, height, 8, PixelLayout::I420, idx);
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
            let seq = dec.sequence_header().cloned().unwrap();
            for obu in hls::split_obus(&pkt.data).unwrap() {
                if matches!(obu.obu_type, ObuType::ClosedLoopKf | ObuType::TileGrp) {
                    let (fh, _) = hls::parse_tile_group(&seq, &obu).unwrap();
                    saw_temporal |= fh.segmentation.temporal != 0;
                }
            }
            assert_frames_equal(&decoded.frame, &pkt.rec_frame.to_raw());
        }
        assert!(
            saw_temporal,
            "temporal segment-map update was never signalled"
        );
    }

    /// Byte-exact comparison of our encoder and decoder against the dav2d
    /// reference on segmentation + delta-Q streams. Ignored by default because
    /// it needs the oracle binary; run with:
    ///
    /// ```text
    /// cargo test -p rustmpeg-video av2_segmentation_decoder_dav2d -- --ignored
    /// ```
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_segmentation_decoder_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::IvfWriter;

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir();
        for &(w, h) in &[(64usize, 64usize), (256, 192)] {
            for &bd in &[8u8, 10, 12] {
                for &q in &[20u16, 100, 140] {
                    let ivf = dir.join(format!("u5_dec_{w}x{h}_{bd}_{q}.ivf"));
                    let dav_y4m = dir.join(format!("u5_dec_{w}x{h}_{bd}_{q}.dav.y4m"));
                    let cfg = segmentation_config(w, h, bd, q);
                    let mut rec_frames: Vec<Vec<u8>> = Vec::new();
                    let mut our_frames: Vec<Vec<u8>> = Vec::new();
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
                        if bd == 8 {
                            let mut enc = Encoder::<u8>::new(cfg).unwrap();
                            let mut dec = Decoder::<u8>::new();
                            for idx in 0..3 {
                                let raw = RawFrame::<u8>::generate_synthetic(
                                    w,
                                    h,
                                    8,
                                    PixelLayout::I420,
                                    idx,
                                );
                                let pkt = enc.encode_raw_frame(&raw).unwrap();
                                writer.write_packet(pkt.pts, &pkt.data).unwrap();
                                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                                let rec = pkt.rec_frame.to_raw();
                                rec_frames.push(raw8_bytes(&rec));
                                our_frames.push(raw8_bytes(&decoded.frame));
                            }
                        } else {
                            let mut enc = Encoder::<u16>::new(cfg).unwrap();
                            let mut dec = Decoder::<u16>::new();
                            for idx in 0..3 {
                                let raw = RawFrame::<u16>::generate_synthetic(
                                    w,
                                    h,
                                    bd,
                                    PixelLayout::I420,
                                    idx,
                                );
                                let pkt = enc.encode_raw_frame(&raw).unwrap();
                                writer.write_packet(pkt.pts, &pkt.data).unwrap();
                                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                                let rec = pkt.rec_frame.to_raw();
                                rec_frames.push(raw16_bytes(&rec));
                                our_frames.push(raw16_bytes(&decoded.frame));
                            }
                        }
                        writer.flush().unwrap();
                    }
                    for (i, (rec, ours)) in rec_frames.iter().zip(our_frames.iter()).enumerate() {
                        assert_eq!(
                            rec, ours,
                            "{w}x{h} {bd}-bit QP {q} frame {i}: our decoder != encoder recon"
                        );
                    }

                    let status = Command::new(&dav2d)
                        .arg("-i")
                        .arg(&ivf)
                        .arg("-o")
                        .arg(&dav_y4m)
                        .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
                        .status()
                        .expect("run dav2d");
                    assert!(status.success(), "dav2d failed for {w}x{h} {bd}-bit QP {q}");
                    let dav = read_y4m_frames(&dav_y4m);
                    assert_eq!(dav.len(), rec_frames.len());
                    for (i, (rec, dav_frame)) in rec_frames.iter().zip(dav.iter()).enumerate() {
                        assert_eq!(
                            rec, dav_frame,
                            "{w}x{h} {bd}-bit QP {q} frame {i}: dav2d != encoder recon"
                        );
                    }
                }
            }
        }
    }

    /// Byte-exact extended-partition check against the dav2d reference decoder.
    ///
    /// Ignored by default because it needs the oracle binary; run with:
    ///
    /// ```text
    /// cargo test -p rustmpeg-video av2_extended_partitions_decoder_dav2d -- --ignored
    /// ```
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_extended_partitions_decoder_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::IvfWriter;
        use crate::av2::levels::BlockPartition;

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir();

        for family in [
            BlockPartition::H3,
            BlockPartition::V3,
            BlockPartition::H4A,
            BlockPartition::H4B,
            BlockPartition::V4A,
            BlockPartition::V4B,
        ] {
            for &(w, h) in &[(64usize, 64usize), (256, 192)] {
                for q in [20u16, 100, 140] {
                    let ivf = dir.join(format!("u10_dec_{family:?}_{w}x{h}_{q}.ivf"));
                    let dav_y4m = dir.join(format!("u10_dec_{family:?}_{w}x{h}_{q}.dav.y4m"));
                    let cfg = extended_partition_config(w, h, q);
                    let mut rec_frames: Vec<Vec<u8>> = Vec::new();
                    let mut our_frames: Vec<Vec<u8>> = Vec::new();
                    let mut family_count = 0usize;
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
                        let mut enc = Encoder::<u8>::new(cfg).unwrap();
                        let mut dec = Decoder::<u8>::new();
                        for _ in 0..3 {
                            let mut raw = RawFrame::<u8>::new(w, h, 8, PixelLayout::I420);
                            fill_extended_pattern(&mut raw, family);
                            let pkt = enc.encode_raw_frame(&raw).unwrap();
                            writer.write_packet(pkt.pts, &pkt.data).unwrap();
                            let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                            family_count += pkt.stats.partition_counts[family as usize];
                            rec_frames.push(raw8_bytes(&pkt.rec_frame.to_raw()));
                            our_frames.push(raw8_bytes(&decoded.frame));
                        }
                        writer.flush().unwrap();
                    }
                    assert!(
                        family_count > 0,
                        "{family:?} was never selected at {w}x{h} QP {q}"
                    );
                    for (i, (rec, ours)) in rec_frames.iter().zip(our_frames.iter()).enumerate() {
                        assert_eq!(
                            rec, ours,
                            "{family:?} {w}x{h} QP {q} frame {i}: our decoder != encoder recon"
                        );
                    }

                    let status = Command::new(&dav2d)
                        .arg("-i")
                        .arg(&ivf)
                        .arg("-o")
                        .arg(&dav_y4m)
                        .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
                        .status()
                        .expect("run dav2d");
                    assert!(
                        status.success(),
                        "dav2d failed for {family:?} {w}x{h} QP {q}"
                    );
                    let dav = read_y4m_frames(&dav_y4m);
                    assert_eq!(dav.len(), rec_frames.len());
                    for (i, (rec, dav_frame)) in rec_frames.iter().zip(dav.iter()).enumerate() {
                        assert_eq!(
                            rec, dav_frame,
                            "{family:?} {w}x{h} QP {q} frame {i}: dav2d != encoder recon"
                        );
                    }
                }
            }
        }

        // One 10-bit case: the synthetic source selects V3.
        let ivf = dir.join("u10_dec_10bit.ivf");
        let dav_y4m = dir.join("u10_dec_10bit.dav.y4m");
        let mut cfg = extended_partition_config(64, 64, 100);
        cfg.bit_depth = 10;
        let mut rec_frames: Vec<Vec<u8>> = Vec::new();
        let mut our_frames: Vec<Vec<u8>> = Vec::new();
        let mut v3_count = 0usize;
        {
            let mut writer = IvfWriter::new(
                std::io::BufWriter::new(std::fs::File::create(&ivf).unwrap()),
                64,
                64,
                30,
                1,
                3,
            )
            .unwrap();
            let mut enc = Encoder::<u16>::new(cfg).unwrap();
            let mut dec = Decoder::<u16>::new();
            for _ in 0..3 {
                let raw = RawFrame::<u16>::generate_synthetic(64, 64, 10, PixelLayout::I420, 0);
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                writer.write_packet(pkt.pts, &pkt.data).unwrap();
                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                v3_count += pkt.stats.partition_counts[BlockPartition::V3 as usize];
                rec_frames.push(raw16_bytes(&pkt.rec_frame.to_raw()));
                our_frames.push(raw16_bytes(&decoded.frame));
            }
            writer.flush().unwrap();
        }
        assert!(v3_count > 0, "V3 was never selected in the 10-bit stream");
        for (i, (rec, ours)) in rec_frames.iter().zip(our_frames.iter()).enumerate() {
            assert_eq!(rec, ours, "10-bit frame {i}: our decoder != encoder recon");
        }
        let status = Command::new(&dav2d)
            .arg("-i")
            .arg(&ivf)
            .arg("-o")
            .arg(&dav_y4m)
            .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
            .status()
            .expect("run dav2d");
        assert!(status.success(), "dav2d failed for the 10-bit case");
        let dav = read_y4m_frames(&dav_y4m);
        assert_eq!(dav.len(), rec_frames.len());
        for (i, (rec, dav_frame)) in rec_frames.iter().zip(dav.iter()).enumerate() {
            assert_eq!(rec, dav_frame, "10-bit frame {i}: dav2d != encoder recon");
        }
    }

    /// Byte-exact film grain check against the dav2d reference decoder.
    ///
    /// Ignored by default because it needs the oracle binary; run with:
    ///
    /// ```text
    /// cargo test -p rustmpeg-video av2_film_grain_decoder_dav2d -- --ignored
    /// ```
    ///
    /// `AVM_FG_IVF` may point at an AVM `--film-grain-test=1` stream; when the
    /// file is absent that case is reported as skipped.
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_film_grain_decoder_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::IvfWriter;

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir().join("rustmpeg-u12dec");
        std::fs::create_dir_all(&dir).unwrap();

        macro_rules! case {
            ($w:expr, $h:expr, $layout:expr, $bd:expr, $q:expr, $p:ty, $ser:ident) => {{
                let (w, h, layout, bd, q): (usize, usize, PixelLayout, u8, u16) =
                    ($w, $h, $layout, $bd, $q);
                let cfg = EncoderConfig {
                    width: w,
                    height: h,
                    layout,
                    bit_depth: bd,
                    base_qidx: q,
                    rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp {
                        base_qidx: q,
                    },
                    keyint: 60,
                    tools: crate::av2::frame::ToolConfig {
                        enable_film_grain: true,
                        ..Default::default()
                    },
                    ..EncoderConfig::default()
                };
                let tag = format!("u12_fg_{w}x{h}_{layout:?}_{bd}_{q}");
                let ivf = dir.join(format!("{tag}.ivf"));
                let mut clean = Vec::new();
                let mut grained = Vec::new();
                let mut on = Vec::new();
                let mut off = Vec::new();
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
                    let mut dec_on = Decoder::<$p>::new();
                    let mut dec_off = Decoder::<$p>::new();
                    dec_off.set_apply_grain(false);
                    for idx in 0..3 {
                        let raw = RawFrame::<$p>::generate_synthetic(w, h, bd, layout, idx);
                        let pkt = enc.encode_raw_frame(&raw).unwrap();
                        assert_eq!(pkt.film_grain.present, 1, "{tag} frame {idx}: no grain");
                        writer.write_packet(pkt.pts, &pkt.data).unwrap();
                        let mut g = pkt.rec_frame.clone();
                        crate::av2::filmgrain::apply_film_grain(
                            &mut g,
                            &pkt.film_grain.data,
                            pkt.film_grain.seed as u16,
                        );
                        clean.push($ser(&pkt.rec_frame.to_raw()));
                        grained.push($ser(&g.to_raw()));
                        on.push($ser(
                            &dec_on.decode_packet(&pkt.data).unwrap().unwrap().frame,
                        ));
                        off.push($ser(
                            &dec_off.decode_packet(&pkt.data).unwrap().unwrap().frame,
                        ));
                    }
                    writer.flush().unwrap();
                }
                for i in 0..3 {
                    assert_eq!(
                        on[i], grained[i],
                        "{tag} frame {i}: decoder != signalled synthesis"
                    );
                    assert_eq!(
                        off[i], clean[i],
                        "{tag} frame {i}: grain-off != clean recon"
                    );
                }
                for (flag, expected, label) in
                    [("1", &grained, "filmgrain 1"), ("0", &clean, "filmgrain 0")]
                {
                    let out = dir.join(format!("{tag}_dav{flag}.y4m"));
                    let status = Command::new(&dav2d)
                        .arg("-i")
                        .arg(&ivf)
                        .arg("-o")
                        .arg(&out)
                        .args([
                            "--threads",
                            "1",
                            "--framedelay",
                            "1",
                            "--strict",
                            "1",
                            "--filmgrain",
                            flag,
                            "-q",
                        ])
                        .status()
                        .expect("run dav2d");
                    assert!(status.success(), "dav2d failed for {tag} {label}");
                    let dav = read_y4m_frames(&out);
                    assert_eq!(dav.len(), expected.len(), "{tag} {label}: frame count");
                    for (i, (a, b)) in dav.iter().zip(expected.iter()).enumerate() {
                        assert_eq!(a, b, "{tag} frame {i}: dav2d {label} mismatch");
                    }
                }
            }};
        }

        case!(64, 64, PixelLayout::I420, 8, 20, u8, raw8_bytes);
        case!(64, 64, PixelLayout::I420, 8, 100, u8, raw8_bytes);
        case!(64, 64, PixelLayout::I420, 8, 140, u8, raw8_bytes);
        case!(256, 192, PixelLayout::I420, 8, 20, u8, raw8_bytes);
        case!(256, 192, PixelLayout::I420, 8, 100, u8, raw8_bytes);
        case!(256, 192, PixelLayout::I420, 8, 140, u8, raw8_bytes);
        case!(256, 192, PixelLayout::I420, 10, 100, u16, raw16_bytes);
        case!(64, 64, PixelLayout::I422, 8, 100, u8, raw8_bytes);
        case!(64, 64, PixelLayout::I444, 8, 100, u8, raw8_bytes);

        // Independent encoder (AVM) `--film-grain-test=1` stream. The stream is
        // generated with tools outside this decoder's subset disabled; see
        // docs/av2/UNIT12_DECODER.md.
        let Some(avm) = crate::av2::test_support::oracle_file("AVM_FG_IVF") else {
            eprintln!("AVM_FG_IVF not set; skipping AVM film-grain case");
            return;
        };
        let mut on = Vec::new();
        let mut off = Vec::new();
        {
            let mut reader = crate::av2::container::IvfReader::new(std::io::BufReader::new(
                std::fs::File::open(&avm).unwrap(),
            ))
            .unwrap();
            let mut dec_on = Decoder::<u8>::new();
            let mut dec_off = Decoder::<u8>::new();
            dec_off.set_apply_grain(false);
            while let Some(packet) = reader.read_packet().unwrap() {
                if let Some(frame) = dec_on.decode_packet(&packet.data).unwrap() {
                    on.push(raw8_bytes(&frame.frame));
                }
                if let Some(frame) = dec_off.decode_packet(&packet.data).unwrap() {
                    off.push(raw8_bytes(&frame.frame));
                }
            }
        }
        assert!(!on.is_empty(), "AVM stream decoded no frames");
        for (flag, expected, label) in [("1", &on, "filmgrain 1"), ("0", &off, "filmgrain 0")] {
            let out = dir.join(format!("avm_fg_dav{flag}.y4m"));
            let status = Command::new(&dav2d)
                .arg("-i")
                .arg(&avm)
                .arg("-o")
                .arg(&out)
                .args([
                    "--threads",
                    "1",
                    "--framedelay",
                    "1",
                    "--strict",
                    "1",
                    "--filmgrain",
                    flag,
                    "-q",
                ])
                .status()
                .expect("run dav2d");
            assert!(
                status.success(),
                "dav2d failed for the AVM stream ({label})"
            );
            let dav = read_y4m_frames(&out);
            assert_eq!(dav.len(), expected.len(), "AVM {label}: frame count");
            for (i, (a, b)) in dav.iter().zip(expected.iter()).enumerate() {
                assert_eq!(a, b, "AVM frame {i}: dav2d {label} mismatch");
            }
        }
        assert_ne!(on, off, "AVM stream carries no film grain");
    }

    /// External oracle for release unit 3 decoder: CCSO (band and edge offsets,
    /// `sb_reuse`) and GDF reconstructions must satisfy
    /// `our decoder == encoder reconstruction == dav2d` on every frame.
    ///
    /// Ignored by default because it needs the external binary; run with:
    ///
    /// ```text
    /// cargo test -p rustmpeg-video av2_unit3_decoder_dav2d_byte_exact -- --ignored
    /// ```
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_unit3_decoder_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::IvfWriter;
        use crate::av2::frame::ToolConfig;

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir().join("rustmpeg-u3dec");
        std::fs::create_dir_all(&dir).unwrap();

        macro_rules! run_case {
            ($p:ty, $raw_fn:ident, $w:expr, $h:expr, $bd:expr, $q:expr, $layout:expr,
             $ccso:expr, $gdf:expr) => {{
                let (w, h, bd, q): (usize, usize, u8, u16) = ($w, $h, $bd, $q);
                let layout: PixelLayout = $layout;
                let tag = format!(
                    "u3d_{}x{}_{}_{}_{}_c{}g{}",
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
                let ivf = dir.join(format!("{tag}.ivf"));
                let dav_y4m = dir.join(format!("{tag}.dav.y4m"));
                let cfg = EncoderConfig {
                    width: w,
                    height: h,
                    layout,
                    bit_depth: bd,
                    base_qidx: q,
                    rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp {
                        base_qidx: q,
                    },
                    keyint: 60,
                    tools: ToolConfig {
                        enable_ccso: $ccso,
                        enable_ccso_sb_reuse: $ccso,
                        enable_gdf: $gdf,
                        ..ToolConfig::default()
                    },
                    ..EncoderConfig::default()
                };
                let mut rec_frames: Vec<Vec<u8>> = Vec::new();
                let mut our_frames: Vec<Vec<u8>> = Vec::new();
                let mut ccso_units = 0usize;
                let mut reuse_units = 0usize;
                let mut gdf_units = 0usize;
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
                        ccso_units += pkt.stats.ccso_units;
                        reuse_units += pkt.stats.ccso_reuse_units;
                        gdf_units += pkt.stats.gdf_units;
                        let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                        rec_frames.push($raw_fn(&pkt.rec_frame.to_raw()));
                        our_frames.push($raw_fn(&decoded.frame));
                    }
                    writer.flush().unwrap();
                }
                for (i, (rec, ours)) in rec_frames.iter().zip(our_frames.iter()).enumerate() {
                    assert_eq!(rec, ours, "{tag} frame {i}: our decoder != encoder recon");
                }
                let status = Command::new(&dav2d)
                    .arg("-i")
                    .arg(&ivf)
                    .arg("-o")
                    .arg(&dav_y4m)
                    .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
                    .status()
                    .expect("run dav2d");
                assert!(status.success(), "dav2d failed for {tag}");
                let dav = read_y4m_frames(&dav_y4m);
                assert_eq!(dav.len(), rec_frames.len(), "{tag}: frame count");
                for (i, (rec, dav_frame)) in rec_frames.iter().zip(dav.iter()).enumerate() {
                    assert_eq!(rec, dav_frame, "{tag} frame {i}: dav2d != encoder recon");
                }
                (ccso_units, reuse_units, gdf_units)
            }};
        }

        // 8-bit I420: CCSO-only (with `sb_reuse`), GDF-only and both.
        for (w, h) in [(64usize, 64usize), (256, 192)] {
            for q in [20u16, 100, 140] {
                for (ccso, gdf) in [(true, false), (false, true), (true, true)] {
                    let (ccso_units, reuse_units, gdf_units) =
                        run_case!(u8, raw8_bytes, w, h, 8, q, PixelLayout::I420, ccso, gdf);
                    if ccso {
                        assert!(ccso_units > 0, "{w}x{h} q{q}: no CCSO units");
                        assert!(reuse_units > 0, "{w}x{h} q{q}: no CCSO sb_reuse units");
                    }
                    if gdf {
                        assert!(gdf_units > 0, "{w}x{h} q{q}: no GDF units");
                    }
                }
            }
        }

        // 10-bit, 4:2:2 and 4:4:4 coverage with both filters.
        let (ccso_units, reuse_units, gdf_units) = run_case!(
            u16,
            raw16_bytes,
            256,
            192,
            10,
            100,
            PixelLayout::I420,
            true,
            true
        );
        assert!(ccso_units > 0 && reuse_units > 0 && gdf_units > 0);
        for layout in [PixelLayout::I422, PixelLayout::I444] {
            let (ccso_units, reuse_units, gdf_units) =
                run_case!(u8, raw8_bytes, 256, 192, 8, 100, layout, true, true);
            assert!(ccso_units > 0 && reuse_units > 0 && gdf_units > 0);
        }
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

    fn roundtrip_tiles(width: usize, height: usize, cols: u8, rows: u8, frames: usize) {
        let cfg = EncoderConfig {
            width,
            height,
            layout: PixelLayout::I420,
            bit_depth: 8,
            base_qidx: 100,
            rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: 100 },
            keyint: 60,
            log2_tile_cols: cols,
            log2_tile_rows: rows,
            ..EncoderConfig::default()
        };
        let mut enc = Encoder::<u8>::new(cfg).unwrap();
        let mut dec = Decoder::<u8>::new();
        for idx in 0..frames {
            let raw = RawFrame::<u8>::generate_synthetic(width, height, 8, PixelLayout::I420, idx);
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
            assert_frames_equal(&decoded.frame, &pkt.rec_frame.to_raw());
        }
    }

    #[test]
    fn test_decode_matches_encoder_recon_inter_128() {
        roundtrip(128, 128, 100, 3);
    }

    #[test]
    fn test_decode_quantizer_sweep() {
        for q in [20u16, 40, 60, 100, 140, 180, 220] {
            roundtrip(64, 64, q, 2);
        }
        roundtrip(192, 192, 20, 3);
        roundtrip(256, 128, 20, 3);
    }

    #[test]
    fn test_decode_small_and_edge_sizes() {
        roundtrip(32, 32, 100, 2);
        roundtrip(96, 64, 100, 2);
        roundtrip(64, 96, 100, 2);
        roundtrip(96, 96, 100, 2);
    }

    #[test]
    fn test_decode_all_layouts() {
        for layout in [
            PixelLayout::I420,
            PixelLayout::I422,
            PixelLayout::I444,
            PixelLayout::I400,
        ] {
            roundtrip_layout(64, 64, layout, 100, 2);
        }
    }

    #[test]
    fn test_decode_tiles() {
        roundtrip_tiles(128, 128, 1, 0, 1);
        roundtrip_tiles(128, 128, 0, 1, 1);
        roundtrip_tiles(128, 128, 1, 1, 1);
        roundtrip_tiles(256, 128, 1, 1, 1);
    }

    #[test]
    fn test_decode_ivf_roundtrip_and_truncated_input() {
        use crate::av2::container::IvfWriter;

        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            layout: PixelLayout::I420,
            bit_depth: 8,
            base_qidx: 100,
            rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: 100 },
            keyint: 60,
            ..EncoderConfig::default()
        };
        let mut enc = Encoder::<u8>::new(cfg).unwrap();
        let mut ivf_bytes = Vec::new();
        {
            let mut writer = IvfWriter::new(&mut ivf_bytes, 64, 64, 30, 1, 2).unwrap();
            for idx in 0..2 {
                let raw = RawFrame::<u8>::generate_synthetic(64, 64, 8, PixelLayout::I420, idx);
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                writer.write_packet(pkt.pts, &pkt.data).unwrap();
            }
            writer.flush().unwrap();
        }
        let decoded = decode_ivf(&ivf_bytes[..]).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].frame_type, FrameType::Key);
        assert_eq!(decoded[1].frame_type, FrameType::Inter);

        // A truncated temporal unit must report an error, not panic.
        let raw = RawFrame::<u8>::generate_synthetic(64, 64, 8, PixelLayout::I420, 2);
        let pkt = enc.encode_raw_frame(&raw).unwrap();
        let mut dec = Decoder::<u8>::new();
        assert!(dec.decode_packet(&pkt.data[..pkt.data.len() / 2]).is_err());
    }

    #[test]
    fn test_decoder_rejects_malformed_and_fuzzed_inputs() {
        let mut dec = Decoder::<u8>::new();
        // Empty input
        assert!(dec.decode_packet(&[]).is_err());

        // Single byte and short garbage prefixes
        for b in 0..=255u8 {
            let _ = dec.decode_packet(&[b]);
            let _ = dec.decode_packet(&[b, b ^ 0xAA]);
            let _ = dec.decode_packet(&[b, 0, 0, b]);
        }

        // Pseudo-random pseudo-fuzzed bytes of varying lengths
        let mut prng = 0x12345678u32;
        for len in [3, 7, 15, 31, 63, 127, 255, 511, 1023] {
            let mut buf = vec![0u8; len];
            for byte in buf.iter_mut() {
                prng = prng.wrapping_mul(1103515245).wrapping_add(12345);
                *byte = (prng >> 16) as u8;
            }
            let _ = dec.decode_packet(&buf);
        }

        // All-0x00 and all-0xFF buffers
        assert!(dec.decode_packet(&[0u8; 64]).is_err());
        assert!(dec.decode_packet(&[0xFFu8; 64]).is_err());
    }

    #[test]
    fn test_decode_high_bit_depth() {
        for depth in [10u8, 12] {
            let cfg = EncoderConfig {
                width: 64,
                height: 64,
                layout: PixelLayout::I420,
                bit_depth: depth,
                base_qidx: 100,
                rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: 100 },
                keyint: 60,
                ..EncoderConfig::default()
            };
            let mut enc = Encoder::<u16>::new(cfg).unwrap();
            let mut dec = Decoder::<u16>::new();
            for idx in 0..2 {
                let raw =
                    RawFrame::<u16>::generate_synthetic(64, 64, depth, PixelLayout::I420, idx);
                let pkt = enc.encode_raw_frame(&raw).unwrap();
                let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                assert_eq!(decoded.frame.y, pkt.rec_frame.to_raw().y, "depth {depth}");
                assert_eq!(decoded.frame.u, pkt.rec_frame.to_raw().u, "depth {depth}");
                assert_eq!(decoded.frame.v, pkt.rec_frame.to_raw().v, "depth {depth}");
            }
        }
    }

    fn compound_config(
        width: usize,
        height: usize,
        bit_depth: u8,
        qidx: u16,
        mask: u8,
    ) -> EncoderConfig {
        EncoderConfig {
            width,
            height,
            layout: PixelLayout::I420,
            bit_depth,
            base_qidx: qidx,
            rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qidx },
            keyint: 60,
            tools: crate::av2::frame::ToolConfig {
                enable_compound: true,
                enable_compound_masks: true,
                compound_types: mask,
                ..Default::default()
            },
            ..EncoderConfig::default()
        }
    }

    fn constant_raw(width: usize, height: usize, value: u8) -> RawFrame<u8> {
        let mut raw = RawFrame::<u8>::new(width, height, 8, PixelLayout::I420);
        raw.y.fill(value);
        raw.u.fill(128);
        raw.v.fill(128);
        raw
    }

    /// Builds a frame that is the exact wedge blend of the two constant
    /// references (64 and 192), so the encoder has a zero-SAD wedge candidate.
    fn wedge_raw(width: usize, height: usize) -> RawFrame<u8> {
        use crate::av2::levels::{BlockSize, CompInterType};
        use crate::av2::me::blend_compound_prep;
        use crate::av2::pixel::BitDepth8;
        use crate::av2::wedge::WEDGE_TABLES;

        let bs = match (width, height) {
            (64, 64) => BlockSize::Bs64x64,
            (32, 32) => BlockSize::Bs32x32,
            (16, 16) => BlockSize::Bs16x16,
            _ => BlockSize::Bs8x8,
        };
        let mask = WEDGE_TABLES.get_mask(bs, 33, 0, 0).unwrap();
        let p0 = vec![64i32 << 4; width * height];
        let p1 = vec![192i32 << 4; width * height];
        let mut y = vec![0u8; width * height];
        blend_compound_prep(
            BitDepth8,
            &mut y,
            0,
            width,
            &p0,
            &p1,
            width,
            height,
            CompInterType::Wedge,
            0,
            Some(mask),
        );
        let mut raw = constant_raw(width, height, 128);
        raw.y.copy_from_slice(&y);
        raw
    }

    /// Compound prediction round trip: reference pair, compound mode, wedge,
    /// difference-weighted, CWP, and the compound `find_mv_stack` branch.
    #[test]
    fn test_decode_matches_encoder_recon_compound() {
        // 64x64 needs constant references so the integer-pel compound RDO has a
        // predictable winner; 256x192 uses the synthetic source.
        let cases: [(usize, usize, u8, &[u8; 3]); 9] = [
            (64, 64, 1, &[64, 192, 128]),
            (64, 64, 4, &[64, 192, 100]),
            (64, 64, 8, &[64, 192, 96]),
            (64, 64, 15, &[64, 192, 96]),
            (256, 192, 1, &[0, 0, 0]),
            (256, 192, 2, &[0, 0, 0]),
            (256, 192, 4, &[0, 0, 0]),
            (256, 192, 15, &[0, 0, 0]),
            (256, 192, 8, &[64, 192, 96]),
        ];
        let mut seen = [0usize; 4];
        for q in [20u16, 100, 140] {
            for &(w, h, mask, consts) in &cases {
                let cfg = compound_config(w, h, 8, q, mask);
                let mut enc = Encoder::<u8>::new(cfg).unwrap();
                let mut dec = Decoder::<u8>::new();
                let mut compound = 0usize;
                let mut modes = [0usize; 4];
                for idx in 0..3 {
                    let raw = if consts[0] == 0 {
                        RawFrame::<u8>::generate_synthetic(w, h, 8, PixelLayout::I420, idx)
                    } else {
                        constant_raw(w, h, consts[idx])
                    };
                    let pkt = enc.encode_raw_frame(&raw).unwrap();
                    let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                    assert_frames_equal(&decoded.frame, &pkt.rec_frame.to_raw());
                    compound += pkt.stats.compound_blocks;
                    modes[0] += pkt.stats.compound_average_blocks;
                    modes[1] += pkt.stats.compound_wedge_blocks;
                    modes[2] += pkt.stats.compound_diff_blocks;
                    modes[3] += pkt.stats.compound_cwp_blocks;
                }
                assert!(
                    compound > 0,
                    "{w}x{h} QP {q} mask {mask}: no compound block was coded"
                );
                match mask {
                    1 => assert!(modes[0] > 0),
                    2 => assert!(modes[1] > 0),
                    4 => assert!(modes[2] > 0),
                    8 => assert!(modes[3] > 0),
                    _ => assert!(modes[1] > 0 || modes[3] > 0),
                }
                for (seen, got) in seen.iter_mut().zip(modes) {
                    *seen += got;
                }
            }
        }
        // 10-bit combined case.
        let cfg = compound_config(256, 192, 10, 100, 15);
        let mut enc = Encoder::<u16>::new(cfg).unwrap();
        let mut dec = Decoder::<u16>::new();
        let mut compound = 0usize;
        for idx in 0..3 {
            let raw = RawFrame::<u16>::generate_synthetic(256, 192, 10, PixelLayout::I420, idx);
            let pkt = enc.encode_raw_frame(&raw).unwrap();
            let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
            let rec = pkt.rec_frame.to_raw();
            assert_eq!(decoded.frame.y, rec.y);
            assert_eq!(decoded.frame.u, rec.u);
            assert_eq!(decoded.frame.v, rec.v);
            compound += pkt.stats.compound_blocks;
        }
        assert!(compound > 0, "10-bit compound case coded no compound block");
        assert!(
            seen.iter().all(|&n| n > 0),
            "a compound mode was not exercised"
        );
    }

    /// Byte-exact compound comparison against the dav2d reference decoder.
    /// Ignored by default because it needs the oracle binary; run with:
    ///
    /// ```text
    /// cargo test -p rustmpeg-video av2_compound_decoder_dav2d -- --ignored
    /// ```
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_compound_decoder_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::IvfWriter;

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir();
        let cases: [(usize, usize, u8, &[u8; 3]); 9] = [
            (64, 64, 1, &[64, 192, 128]),
            (64, 64, 2, &[64, 192, 0]),
            (64, 64, 4, &[64, 192, 100]),
            (64, 64, 8, &[64, 192, 96]),
            (256, 192, 1, &[0, 0, 0]),
            (256, 192, 2, &[0, 0, 0]),
            (256, 192, 4, &[0, 0, 0]),
            (256, 192, 15, &[0, 0, 0]),
            (256, 192, 8, &[64, 192, 96]),
        ];
        for &(w, h, mask, consts) in &cases {
            // At the highest quality the 64x64 wedge clip codes as intra; the
            // QP sweep for that size is covered by the external matrix.
            let qs: &[u16] = if w == 64 && mask == 2 {
                &[100]
            } else {
                &[20, 100, 140]
            };
            for &q in qs {
                // The 64x64 wedge source is built from the same mask the
                // encoder evaluates, so the RDO has an exact wedge candidate.
                let mut frames_raw: Vec<RawFrame<u8>> = Vec::new();
                for idx in 0..3 {
                    frames_raw.push(if consts[0] == 0 {
                        RawFrame::<u8>::generate_synthetic(w, h, 8, PixelLayout::I420, idx)
                    } else if mask == 2 && idx == 2 {
                        wedge_raw(w, h)
                    } else {
                        constant_raw(w, h, consts[idx])
                    });
                }
                let ivf = dir.join(format!("u7_dec_{w}x{h}_m{mask}_q{q}.ivf"));
                let dav_y4m = dir.join(format!("u7_dec_{w}x{h}_m{mask}_q{q}.dav.y4m"));
                let mut rec_frames: Vec<Vec<u8>> = Vec::new();
                let mut compound = 0usize;
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
                    let cfg = compound_config(w, h, 8, q, mask);
                    let mut enc = Encoder::<u8>::new(cfg).unwrap();
                    let mut dec = Decoder::<u8>::new();
                    for raw in &frames_raw {
                        let pkt = enc.encode_raw_frame(raw).unwrap();
                        writer.write_packet(pkt.pts, &pkt.data).unwrap();
                        let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                        assert_frames_equal(&decoded.frame, &pkt.rec_frame.to_raw());
                        compound += pkt.stats.compound_blocks;
                        rec_frames.push(raw8_bytes(&pkt.rec_frame.to_raw()));
                    }
                    writer.flush().unwrap();
                }
                assert!(
                    compound > 0,
                    "{w}x{h} QP {q} mask {mask}: no compound block"
                );
                let status = Command::new(&dav2d)
                    .arg("-i")
                    .arg(&ivf)
                    .arg("-o")
                    .arg(&dav_y4m)
                    .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
                    .status()
                    .expect("run dav2d");
                assert!(
                    status.success(),
                    "dav2d failed for {w}x{h} QP {q} mask {mask}"
                );
                let dav = read_y4m_frames(&dav_y4m);
                assert_eq!(dav.len(), rec_frames.len());
                for (i, (rec, dav_frame)) in rec_frames.iter().zip(dav.iter()).enumerate() {
                    assert_eq!(
                        rec, dav_frame,
                        "{w}x{h} QP {q} mask {mask} frame {i}: dav2d != encoder recon"
                    );
                }
            }
        }
    }

    /// Decoder half of release unit 9: lossless WHT with and without DPCM,
    /// every reduced transform set, and the full transform-type signalling
    /// must decode to the encoder's reconstruction.
    #[test]
    fn test_decoder_unit9_lossless_and_transform_sets() {
        use crate::av2::frame::ToolConfig;

        macro_rules! check_case {
            ($p:ty, $raw_fn:ident, $w:expr, $h:expr, $bd:expr, $q:expr, $reduced:expr,
             $tx_rdo:expr, $dpcm:expr) => {{
                let (w, h, bd, q, reduced, tx_rdo, dpcm): (usize, usize, u8, u16, u8, bool, bool) =
                    ($w, $h, $bd, $q, $reduced, $tx_rdo, $dpcm);
                let cfg = EncoderConfig {
                    width: w,
                    height: h,
                    bit_depth: bd,
                    layout: PixelLayout::I420,
                    base_qidx: q,
                    rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: q },
                    keyint: 60,
                    tools: ToolConfig {
                        enable_tx_type_rdo: tx_rdo,
                        reduced_txtp_set: reduced,
                        enable_dpcm: dpcm,
                        ..ToolConfig::default()
                    },
                    ..EncoderConfig::default()
                };
                let mut enc = Encoder::<$p>::new(cfg).unwrap();
                let mut dec = Decoder::<$p>::new();
                let (mut wht, mut dpcm_y, mut dpcm_uv, mut non_dct) = (0usize, 0, 0, 0);
                for idx in 0..3 {
                    let raw = RawFrame::<$p>::generate_synthetic(w, h, bd, PixelLayout::I420, idx);
                    let pkt = enc.encode_raw_frame(&raw).unwrap();
                    wht += pkt.stats.wht_blocks;
                    dpcm_y += pkt.stats.dpcm_y_blocks;
                    dpcm_uv += pkt.stats.dpcm_uv_blocks;
                    non_dct += pkt.stats.non_dct_tx_blocks;
                    let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                    let got = $raw_fn(&decoded.frame);
                    let want = $raw_fn(&pkt.rec_frame.to_raw());
                    assert_eq!(
                        got, want,
                        "{w}x{h} bd{bd} q{q} r{reduced} txrdo{tx_rdo} dpcm{dpcm} frame {idx}: decoder != encoder recon"
                    );
                    if q == 0 {
                        assert_eq!(
                            got,
                            $raw_fn(&raw),
                            "{w}x{h} frame {idx}: lossless decoder output != source"
                        );
                    }
                }
                (wht, dpcm_y, dpcm_uv, non_dct)
            }};
        }

        // Lossless WHT, with and without DPCM, including frame 2.
        let (wht, dpcm_y, dpcm_uv, _) = check_case!(u8, raw8_bytes, 64, 64, 8, 0, 2, false, true);
        assert!(wht > 0 && dpcm_y > 0 && dpcm_uv > 0);
        let (wht, dpcm_y, dpcm_uv, _) = check_case!(u8, raw8_bytes, 64, 64, 8, 0, 2, false, false);
        assert!(wht > 0 && dpcm_y == 0 && dpcm_uv == 0);
        for (w, h) in [(64usize, 64usize), (256, 192)] {
            for dpcm in [false, true] {
                let (wht, _, _, _) = check_case!(u8, raw8_bytes, w, h, 8, 0, 2, false, dpcm);
                assert!(wht > 0, "{w}x{h} dpcm{dpcm}: no WHT blocks");
            }
        }
        check_case!(u16, raw16_bytes, 256, 192, 10, 0, 2, false, true);

        // Reduced transform sets 0-3 at three quantizers.
        for reduced in [0u8, 1, 2, 3] {
            for q in [20u16, 100, 140] {
                check_case!(u8, raw8_bytes, 64, 64, 8, q, reduced, true, false);
                check_case!(u8, raw8_bytes, 256, 192, 8, q, reduced, true, false);
            }
        }
        let (_, _, _, non_dct) = check_case!(u8, raw8_bytes, 64, 64, 8, 100, 0, true, false);
        assert!(non_dct > 0, "no non-DCT transform type selected");
        check_case!(u16, raw16_bytes, 256, 192, 10, 100, 0, true, false);
    }

    /// External oracle for the decoder half of unit 9: our decoder output must
    /// match `dav2d` for lossless WHT/DPCM and every reduced transform set.
    #[test]
    #[ignore = "requires the dav2d reference decoder"]
    fn av2_unit9_decoder_dav2d_byte_exact() {
        use std::process::Command;

        use crate::av2::container::IvfWriter;
        use crate::av2::frame::ToolConfig;

        let Some(dav2d) = crate::av2::test_support::oracle_bin("DAV2D", "dav2d") else {
            eprintln!("dav2d not found (set DAV2D or add it to PATH); skipping external test");
            return;
        };
        let dir = std::env::temp_dir().join("rustmpeg-u9dec");
        std::fs::create_dir_all(&dir).unwrap();

        macro_rules! check_case {
            ($p:ty, $raw_fn:ident, $w:expr, $h:expr, $bd:expr, $q:expr, $reduced:expr,
             $tx_rdo:expr, $dpcm:expr) => {{
                let (w, h, bd, q, reduced, tx_rdo, dpcm): (usize, usize, u8, u16, u8, bool, bool) =
                    ($w, $h, $bd, $q, $reduced, $tx_rdo, $dpcm);
                let tag = format!(
                    "u9d_{w}x{h}_{bd}_{q}_r{reduced}_{}{}",
                    tx_rdo as u8, dpcm as u8
                );
                let cfg = EncoderConfig {
                    width: w,
                    height: h,
                    bit_depth: bd,
                    layout: PixelLayout::I420,
                    base_qidx: q,
                    rate_control: crate::av2::ratectrl::RateControlMode::ConstantQp {
                        base_qidx: q,
                    },
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
                let dav_y4m = dir.join(format!("{tag}.dav.y4m"));
                let mut ours: Vec<Vec<u8>> = Vec::new();
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
                        let raw =
                            RawFrame::<$p>::generate_synthetic(w, h, bd, PixelLayout::I420, idx);
                        let pkt = enc.encode_raw_frame(&raw).unwrap();
                        writer.write_packet(pkt.pts, &pkt.data).unwrap();
                        let decoded = dec.decode_packet(&pkt.data).unwrap().unwrap();
                        assert_eq!($raw_fn(&decoded.frame), $raw_fn(&pkt.rec_frame.to_raw()));
                        ours.push($raw_fn(&decoded.frame));
                    }
                    writer.flush().unwrap();
                }
                let status = Command::new(&dav2d)
                    .arg("-i")
                    .arg(&ivf)
                    .arg("-o")
                    .arg(&dav_y4m)
                    .args(["--threads", "1", "--framedelay", "1", "--strict", "1", "-q"])
                    .status()
                    .expect("run dav2d");
                assert!(status.success(), "dav2d failed for {tag}");
                let dav = read_y4m_frames(&dav_y4m);
                assert_eq!(dav.len(), ours.len(), "{tag}: frame count");
                for (i, (a, b)) in ours.iter().zip(dav.iter()).enumerate() {
                    assert_eq!(a, b, "{tag} frame {i}: decoder != dav2d");
                }
            }};
        }

        // Lossless WHT with and without DPCM, 8-bit and 10-bit.
        for dpcm in [false, true] {
            check_case!(u8, raw8_bytes, 64, 64, 8, 0, 2, false, dpcm);
            check_case!(u8, raw8_bytes, 256, 192, 8, 0, 2, false, dpcm);
        }
        check_case!(u16, raw16_bytes, 256, 192, 10, 0, 2, false, true);

        // Reduced sets 0-3 and the full transform-type signalling.
        for reduced in [0u8, 1, 2, 3] {
            for q in [20u16, 100, 140] {
                check_case!(u8, raw8_bytes, 64, 64, 8, q, reduced, true, false);
                check_case!(u8, raw8_bytes, 256, 192, 8, q, reduced, true, false);
            }
        }
        check_case!(u16, raw16_bytes, 256, 192, 10, 100, 0, true, false);
    }
}
