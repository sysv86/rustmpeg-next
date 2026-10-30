// SPDX-License-Identifier: Apache-2.0

//! JPEG XL v2 image and animation decoder (ISO/IEC 18181-1 & 18181-2).

use std::num::Wrapping;

use rustmpeg_core::{Buffer, ColorDescription, PixelFormat, Plane, VideoFrame};

use crate::cms::{CmsProfile, CmsRenderingIntent, CmsTransform};

use super::bits::{BitReader, unpack_signed};
use super::container::{ParsedContainer, parse_container};
use super::entropy::EntropyDecoder;
use super::headers::{
    BitDepthHeader, BlendMode, ColourEncoding, ColourSpace, EdgePreservingFilter, FrameEncoding,
    FrameHeader, FrameType, GaborFilter, ImageHeader, Primaries, Toc, TocGroupKind,
    TransferFunctionHeader, read_icc,
};
use super::modular::{ChannelShift, GlobalModularState, ModularChannelInfo, ModularGrid};
use super::vardct::{
    DecodedLfGroup, FloatGrid, HfBlockContext, HfGlobal, LfChannelCorrelation,
    LfChannelDequantization, Quantizer, apply_epf, apply_gaborish, decode_hf_pass_group,
    reconstruct_vardct_xyb_with_lf_frame, xyb_to_target_rgb,
};
use super::{JxlBitDepth, JxlColorEncoding, JxlError, JxlFrame, JxlImage, JxlPixelLayout};

/// JPEG XL v2 decoder supporting both bare codestreams and ISO/IEC 18181-2 containers,
/// Modular (lossless/XYB/YCbCr) and VarDCT (lossy) frames, `Patches`, `Splines`, `Noise`,
/// hierarchical `LfFrame`s, 8/16/32-bit depths, extra channels, ICCv4 color management via
/// `moxcms`, and multi-frame animations.
#[derive(Debug, Clone)]
pub struct JxlDecoder {
    /// Whether to apply embedded ICC profiles via `moxcms` when converting to sRGB output.
    pub apply_icc: bool,
}

impl Default for JxlDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl JxlDecoder {
    /// Creates a new `JxlDecoder` with default settings (`apply_icc = true`).
    #[must_use]
    pub const fn new() -> Self {
        Self { apply_icc: true }
    }

    /// Configures whether embedded ICC profiles should be applied using `moxcms`.
    #[must_use]
    pub const fn with_apply_icc(mut self, apply_icc: bool) -> Self {
        self.apply_icc = apply_icc;
        self
    }

    /// Parses only the container and [`ImageHeader`] without decoding frame pixel data.
    pub fn probe<'a>(
        &self,
        data: &'a [u8],
    ) -> Result<(ParsedContainer<'a>, ImageHeader), JxlError> {
        let parsed = parse_container(data)?;
        let mut br = BitReader::new(&parsed.codestream);
        let header = ImageHeader::parse(&mut br)?;
        Ok((parsed, header))
    }

    /// Decodes a complete JPEG XL image or animation from `data`.
    pub fn decode(&self, data: &[u8]) -> Result<JxlImage, JxlError> {
        let parsed = parse_container(data)?;
        let codestream = &parsed.codestream;
        let mut br = BitReader::new(codestream);
        let image_header = ImageHeader::parse(&mut br)?;

        let icc_profile = if image_header.metadata.colour_encoding.want_icc() {
            Some(read_icc(&mut br)?)
        } else {
            None
        };

        // Preview frame (if present) precedes regular frames and is byte-aligned.
        if let Some(ref preview) = image_header.metadata.preview {
            br.zero_pad_to_byte()?;
            let mut preview_header = image_header.clone();
            preview_header.size.width = preview.width;
            preview_header.size.height = preview.height;
            preview_header.metadata.preview = None;
            let _ = decode_one_frame_internal(
                &mut br,
                codestream,
                &preview_header,
                &mut [None, None, None, None],
                &mut [None, None, None, None],
                0,
                0,
            )?;
        }

        let width = image_header.width_with_orientation();
        let height = image_header.height_with_orientation();
        let unoriented_w = image_header.size.width as usize;
        let unoriented_h = image_header.size.height as usize;

        let is_gray = image_header.metadata.grayscale();
        let has_alpha = image_header.metadata.ec_info.iter().any(|ec| ec.is_alpha());
        let alpha_ec_idx = image_header
            .metadata
            .ec_info
            .iter()
            .position(|ec| ec.is_alpha());

        let layout = match (is_gray, has_alpha) {
            (true, false) => JxlPixelLayout::Gray,
            (true, true) => JxlPixelLayout::GrayAlpha,
            (false, false) => JxlPixelLayout::Rgb,
            (false, true) => JxlPixelLayout::Rgba,
        };

        let bits = image_header.metadata.bit_depth.bits_per_sample();
        let is_float = matches!(
            image_header.metadata.bit_depth,
            BitDepthHeader::FloatSample { .. }
        );
        let bit_depth = if is_float || bits > 16 {
            JxlBitDepth::Float32
        } else if bits > 8 {
            JxlBitDepth::Uint16(bits as u8)
        } else {
            JxlBitDepth::Uint8(bits as u8)
        };

        let total_ch = if is_gray { 1 } else { 3 } + image_header.metadata.ec_info.len();
        let mut ref_frames: [Option<Vec<FloatGrid>>; 4] = [None, None, None, None];
        let mut lf_frames: [Option<[FloatGrid; 3]>; 4] = [None, None, None, None];
        let mut canvas: Option<Vec<FloatGrid>> = None;
        let mut frames = Vec::new();
        let mut visible_frames_num = 0usize;
        let mut invisible_frames_num = 0usize;

        loop {
            br.zero_pad_to_byte()?;
            if br.bytes_read() >= codestream.len() {
                break;
            }
            let decoded_frame = decode_one_frame_internal(
                &mut br,
                codestream,
                &image_header,
                &mut ref_frames,
                &mut lf_frames,
                visible_frames_num,
                invisible_frames_num,
            )?;
            let is_last = decoded_frame.header.is_last;

            let visible = decoded_frame.header.frame_type.is_normal_frame()
                && (decoded_frame.header.duration > 0 || decoded_frame.header.is_last);
            if visible {
                visible_frames_num += 1;
                invisible_frames_num = 0;
            } else {
                invisible_frames_num += 1;
            }

            if decoded_frame.header.frame_type == FrameType::RegularFrame
                || decoded_frame.header.frame_type == FrameType::SkipProgressive
            {
                let mut next_canvas = canvas.take().unwrap_or_else(|| {
                    (0..total_ch)
                        .map(|_| FloatGrid::new(unoriented_w, unoriented_h))
                        .collect()
                });
                blend_onto_canvas(
                    &mut next_canvas,
                    &decoded_frame.channels,
                    &decoded_frame.header,
                    &image_header,
                    is_gray,
                );

                if visible {
                    let mut out_channels = if is_gray {
                        vec![next_canvas[0].clone()]
                    } else {
                        vec![
                            next_canvas[0].clone(),
                            next_canvas[1].clone(),
                            next_canvas[2].clone(),
                        ]
                    };
                    if let Some(a_idx) = alpha_ec_idx {
                        let c_offset = if is_gray { 1 } else { 3 };
                        if c_offset + a_idx < next_canvas.len() {
                            out_channels.push(next_canvas[c_offset + a_idx].clone());
                        }
                    }

                    // Apply orientation `1..=8`.
                    let mut oriented: Vec<FloatGrid> = out_channels
                        .into_iter()
                        .map(|g| apply_orientation(&g, image_header.metadata.orientation))
                        .collect();

                    // Optional ICC -> sRGB transform on 3-channel RGB when `apply_icc` is enabled.
                    if self.apply_icc
                        && !is_gray
                        && oriented.len() >= 3
                        && let Some(ref icc_bytes) = icc_profile
                        && let Ok(src_prof) = CmsProfile::from_icc(icc_bytes)
                    {
                        let dst_prof = CmsProfile::new_srgb();
                        if let Ok(xform) =
                            CmsTransform::new(&src_prof, &dst_prof, CmsRenderingIntent::Perceptual)
                        {
                            let mut planes = vec![
                                oriented[0].data.clone(),
                                oriented[1].data.clone(),
                                oriented[2].data.clone(),
                            ];
                            if xform.transform_planar(&mut planes).is_ok() {
                                oriented[0].data = planes.remove(0);
                                oriented[1].data = planes.remove(0);
                                oriented[2].data = planes.remove(0);
                            }
                        }
                    }

                    let packed = pack_float_channels(&oriented, bit_depth);
                    frames.push(JxlFrame {
                        width,
                        height,
                        layout,
                        bit_depth,
                        duration_ticks: decoded_frame.header.duration,
                        name: decoded_frame.header.name.clone(),
                        data: packed,
                    });
                }

                canvas = Some(next_canvas);
            }

            if is_last {
                break;
            }
        }

        if frames.is_empty() {
            return Err(JxlError::InvalidBitstream(
                "no visible frames decoded from JXL stream",
            ));
        }

        let color_encoding = if let Some(icc) = icc_profile {
            JxlColorEncoding::Icc(icc)
        } else {
            match &image_header.metadata.colour_encoding {
                ColourEncoding::Enum(e) => match (e.colour_space, e.primaries, e.tf) {
                    (ColourSpace::Grey, _, TransferFunctionHeader::Linear) => {
                        JxlColorEncoding::LinearGray
                    }
                    (ColourSpace::Grey, _, _) => JxlColorEncoding::GraySrgb,
                    (_, Primaries::P3, _) => JxlColorEncoding::DisplayP3,
                    (_, Primaries::Bt2100, TransferFunctionHeader::Pq) => {
                        JxlColorEncoding::Bt2020Pq
                    }
                    (_, Primaries::Bt2100, TransferFunctionHeader::Hlg) => {
                        JxlColorEncoding::Bt2020Hlg
                    }
                    (_, _, TransferFunctionHeader::Linear) => JxlColorEncoding::LinearSrgb,
                    _ => JxlColorEncoding::Srgb,
                },
                ColourEncoding::IccProfile(_) => JxlColorEncoding::Srgb,
            }
        };

        Ok(JxlImage {
            width,
            height,
            layout,
            bit_depth,
            color_encoding,
            container_kind: parsed.kind,
            exif: parsed.exif,
            xmp: parsed.xmp,
            animation: image_header.metadata.animation,
            frames,
        })
    }

    /// Decodes the first visible frame of `data` into a `rustmpeg_core::VideoFrame`.
    pub fn decode_video_frame(&self, data: &[u8]) -> Result<VideoFrame, JxlError> {
        let img = self.decode(data)?;
        img.frames[0].to_video_frame()
    }
}

struct InternalDecodedFrame {
    header: FrameHeader,
    channels: Vec<FloatGrid>,
}

#[allow(clippy::too_many_arguments)]
fn decode_one_frame_internal(
    br: &mut BitReader<'_>,
    codestream: &[u8],
    image_header: &ImageHeader,
    ref_frames: &mut [Option<Vec<FloatGrid>>; 4],
    lf_frames: &mut [Option<[FloatGrid; 3]>; 4],
    visible_frames_num: usize,
    invisible_frames_num: usize,
) -> Result<InternalDecodedFrame, JxlError> {
    let frame_header = FrameHeader::parse(br, image_header)?;
    let toc = Toc::parse(br, &frame_header)?;

    let payload_start = br.bytes_read();
    let payload_end = payload_start
        .checked_add(toc.total_size)
        .ok_or(JxlError::LimitExceeded("TOC total byte size overflow"))?;
    let frame_payload = codestream
        .get(payload_start..payload_end)
        .ok_or(JxlError::UnexpectedEof)?;
    br.skip_bits(toc.total_size * 8)?;

    let single_section = toc.is_single_entry();
    let mut single_br = BitReader::new(if single_section {
        toc.section_bytes(frame_payload, TocGroupKind::All)?
    } else {
        &[]
    });
    let mut lf_global_sec_br;
    let lf_global_br: &mut BitReader<'_> = if single_section {
        &mut single_br
    } else {
        lf_global_sec_br =
            BitReader::new(toc.section_bytes(frame_payload, TocGroupKind::LfGlobal)?);
        &mut lf_global_sec_br
    };

    // Parse optional `Patches`, `Splines`, `NoiseParameters` in `LfGlobal`.
    let patches = if frame_header.flags.patches() {
        Some(parse_patches(lf_global_br, image_header, &frame_header)?)
    } else {
        None
    };
    let splines = if frame_header.flags.splines() {
        Some(parse_splines(lf_global_br, &frame_header)?)
    } else {
        None
    };
    let noise = if frame_header.flags.noise() {
        Some(parse_noise(lf_global_br)?)
    } else {
        None
    };

    let lf_dequant = LfChannelDequantization::parse(lf_global_br)?;
    let vardct_global = if frame_header.encoding == FrameEncoding::VarDct {
        let q = Quantizer::parse(lf_global_br)?;
        let ctx = HfBlockContext::parse(lf_global_br)?;
        let corr = LfChannelCorrelation::parse(lf_global_br)?;
        Some((q, ctx, corr))
    } else {
        None
    };
    let base_correlations_xb = vardct_global
        .as_ref()
        .map(|(_, _, corr)| (corr.base_correlation_x, corr.base_correlation_b));

    // Build initial `ModularChannelInfo` list for `GlobalModular`.
    let color_w = frame_header.color_sample_width();
    let color_h = frame_header.color_sample_height();
    let mut mod_channels = Vec::new();
    if frame_header.encoding == FrameEncoding::Modular {
        if frame_header.do_ycbcr {
            for idx in 0..3 {
                mod_channels.push(ModularChannelInfo::new(
                    color_w,
                    color_h,
                    ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx),
                ));
            }
        } else {
            let n_col = frame_header.encoded_color_channels;
            for _ in 0..n_col {
                mod_channels.push(ModularChannelInfo::new(
                    color_w,
                    color_h,
                    ChannelShift::from_shift(0),
                ));
            }
        }
    }
    let extra_channel_from = mod_channels.len();
    let color_up_shift = frame_header.upsampling.trailing_zeros();
    for (&ec_up, ec_info) in frame_header
        .ec_upsampling
        .iter()
        .zip(image_header.metadata.ec_info.iter())
    {
        let ec_up_shift = ec_up.trailing_zeros();
        let actual_shift = (ec_up_shift + ec_info.dim_shift).saturating_sub(color_up_shift);
        mod_channels.push(ModularChannelInfo::new(
            color_w,
            color_h,
            ChannelShift::from_shift(actual_shift),
        ));
    }

    let pass_shifts = frame_header.pass_shifts();
    let mut gmodular = GlobalModularState::parse(
        lf_global_br,
        mod_channels,
        extra_channel_from,
        frame_header.group_dim(),
        image_header.metadata.bit_depth.bits_per_sample(),
        frame_header.width,
        frame_header.height,
        &pass_shifts,
    )?;

    let num_lf_groups = frame_header.num_lf_groups();
    let num_groups = frame_header.num_groups();
    let num_passes = frame_header.passes.num_passes;

    // Decode pre-color-transform color `[FloatGrid; 3]` and extra channel `Vec<FloatGrid>`.
    let (mut color_floats, mut ec_floats): ([FloatGrid; 3], Vec<FloatGrid>) = match frame_header
        .encoding
    {
        FrameEncoding::Modular => {
            for lf_idx in 0..num_lf_groups {
                let mut lfg_br;
                let active_br: &mut BitReader<'_> = if single_section {
                    lf_global_br
                } else {
                    lfg_br = BitReader::new(
                        toc.section_bytes(frame_payload, TocGroupKind::LfGroup(lf_idx))?,
                    );
                    &mut lfg_br
                };
                gmodular.decode_lf_group(active_br, lf_idx, num_lf_groups)?;
            }

            if !single_section {
                for pass_idx in 0..num_passes {
                    for group_idx in 0..num_groups {
                        let mut pg_br = BitReader::new(toc.section_bytes(
                            frame_payload,
                            TocGroupKind::GroupPass {
                                pass_idx,
                                group_idx,
                            },
                        )?);
                        gmodular.decode_pass_group(
                            &mut pg_br,
                            pass_idx,
                            group_idx,
                            num_lf_groups,
                            num_groups,
                        )?;
                    }
                }
            } else {
                for pass_idx in 0..num_passes {
                    for group_idx in 0..num_groups {
                        gmodular.decode_pass_group(
                            lf_global_br,
                            pass_idx,
                            group_idx,
                            num_lf_groups,
                            num_groups,
                        )?;
                    }
                }
            }

            let mod_grids = gmodular.finish();
            extract_modular_color_and_ec_floats(
                &mod_grids,
                image_header,
                &frame_header,
                &lf_dequant,
            )
        }
        FrameEncoding::VarDct => {
            let (quantizer, hf_block_ctx, lf_chan_corr) = vardct_global.unwrap();
            let global_ma = gmodular.global_ma_config.clone();

            let mut lf_groups = Vec::with_capacity(num_lf_groups as usize);
            for lf_idx in 0..num_lf_groups {
                let mut lfg_br;
                let active_br: &mut BitReader<'_> = if single_section {
                    lf_global_br
                } else {
                    lfg_br = BitReader::new(
                        toc.section_bytes(frame_payload, TocGroupKind::LfGroup(lf_idx))?,
                    );
                    &mut lfg_br
                };
                lf_groups.push(DecodedLfGroup::parse(
                    active_br,
                    &frame_header,
                    &quantizer,
                    global_ma.as_ref(),
                    &mut gmodular,
                    lf_idx,
                )?);
            }

            let mut hfg_br;
            let active_hfg_br: &mut BitReader<'_> = if single_section {
                lf_global_br
            } else {
                hfg_br = BitReader::new(toc.section_bytes(frame_payload, TocGroupKind::HfGlobal)?);
                &mut hfg_br
            };
            let hf_global = HfGlobal::parse(
                active_hfg_br,
                image_header.metadata.bit_depth.bits_per_sample(),
                &frame_header,
                global_ma.as_ref(),
                &hf_block_ctx,
            )?;

            let mut bw_total = (color_w as usize).div_ceil(8);
            let mut bh_total = (color_h as usize).div_ceil(8);
            if frame_header
                .jpeg_upsampling
                .into_iter()
                .any(|j| j == 1 || j == 2)
            {
                bw_total = bw_total.div_ceil(2) * 2;
            }
            if frame_header
                .jpeg_upsampling
                .into_iter()
                .any(|j| j == 1 || j == 3)
            {
                bh_total = bh_total.div_ceil(2) * 2;
            }
            let shifts: [ChannelShift; 3] = std::array::from_fn(|idx| {
                ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
            });
            let mut hf_coeff_grid: [ModularGrid; 3] = std::array::from_fn(|c| {
                let (sw, sh) = shifts[c].shift_size((bw_total as u32, bh_total as u32));
                ModularGrid::new(sw as usize * 8, sh as usize * 8)
            });

            let groups_per_row = frame_header.groups_per_row().max(1);
            let lf_groups_per_row = frame_header.lf_groups_per_row().max(1);

            for pass_idx in 0..num_passes {
                for group_idx in 0..num_groups {
                    let mut pg_br;
                    let active_pg_br: &mut BitReader<'_> = if single_section {
                        lf_global_br
                    } else {
                        pg_br = BitReader::new(toc.section_bytes(
                            frame_payload,
                            TocGroupKind::GroupPass {
                                pass_idx,
                                group_idx,
                            },
                        )?);
                        &mut pg_br
                    };
                    let gx = group_idx % groups_per_row;
                    let gy = group_idx / groups_per_row;
                    let lf_idx = ((gy / 8) * lf_groups_per_row + (gx / 8)) as usize;
                    decode_hf_pass_group(
                        active_pg_br,
                        &frame_header,
                        &hf_block_ctx,
                        &hf_global,
                        &lf_groups[lf_idx],
                        pass_idx,
                        group_idx,
                        &mut hf_coeff_grid,
                    )?;
                    gmodular.decode_pass_group(
                        active_pg_br,
                        pass_idx,
                        group_idx,
                        num_lf_groups,
                        num_groups,
                    )?;
                }
            }

            let ext_lf = lf_frames
                .get(frame_header.lf_level as usize)
                .and_then(|opt| opt.as_ref());
            let mut xyb = reconstruct_vardct_xyb_with_lf_frame(
                image_header,
                &frame_header,
                &lf_dequant,
                &quantizer,
                &lf_chan_corr,
                &hf_global,
                &lf_groups,
                &hf_coeff_grid,
                ext_lf,
            );

            if let GaborFilter::Enabled(weights) = frame_header.restoration_filter.gab {
                apply_gaborish(&mut xyb, weights);
            }
            if let EdgePreservingFilter::Enabled(ref epf_params) =
                frame_header.restoration_filter.epf
            {
                apply_epf(&mut xyb, &frame_header, epf_params, Some(&lf_groups));
            }

            let ec_grids = gmodular.finish();
            let mut ecs = Vec::with_capacity(ec_grids.len());
            for (ec_idx, ec_grid) in ec_grids.into_iter().enumerate() {
                let ec_bd = image_header
                    .metadata
                    .ec_info
                    .get(ec_idx)
                    .map(|e| e.bit_depth)
                    .unwrap_or(image_header.metadata.bit_depth);
                ecs.push(modular_grid_to_float_grid(&ec_grid, ec_bd));
            }
            (xyb, ecs)
        }
    };

    // Save hierarchical `LfFrame` if `frame_type == FrameType::LfFrame`.
    if frame_header.frame_type == FrameType::LfFrame && (1..=4).contains(&frame_header.lf_level) {
        lf_frames[(frame_header.lf_level - 1) as usize] = Some(color_floats.clone());
    }

    // Upsample color and extra channels to `(frame_header.width, frame_header.height)` before
    // rendering features (`Patches`, `Splines`, `Noise`) per ISO/IEC 18181-1 Annex J.
    let target_w = frame_header.width as usize;
    let target_h = frame_header.height as usize;
    for ch in &mut color_floats {
        if ch.width != target_w || ch.height != target_h {
            *ch = upsample_nn_grid(ch, target_w, target_h);
        }
    }
    for ch in &mut ec_floats {
        if ch.width != target_w || ch.height != target_h {
            *ch = upsample_nn_grid(ch, target_w, target_h);
        }
    }

    let is_gray = image_header.metadata.grayscale();
    let color_ch_count = if is_gray { 1 } else { 3 };

    // Apply `Splines` in XYB / pre-CT color space when 3 color channels are active.
    if let Some(ref spline_data) = splines
        && (!is_gray || image_header.metadata.xyb_encoded || frame_header.do_ycbcr)
    {
        render_splines(
            &frame_header,
            &mut color_floats,
            spline_data,
            base_correlations_xb,
        );
    }

    // Apply `Noise` in XYB / pre-CT color space when 3 color channels are active.
    if let Some(ref noise_params) = noise
        && (!is_gray || image_header.metadata.xyb_encoded || frame_header.do_ycbcr)
    {
        render_noise(
            &frame_header,
            visible_frames_num,
            invisible_frames_num,
            base_correlations_xb,
            &mut color_floats,
            noise_params,
        );
    }

    // Convert XYB or YCbCr to target RGB / Grey (unless `save_before_ct && !is_last`).
    if frame_header.do_ycbcr {
        ycbcr_grids_to_rgb(&mut color_floats);
    } else if image_header.metadata.xyb_encoded && !frame_header.save_before_ct {
        xyb_to_target_rgb(
            &mut color_floats,
            &image_header.metadata.opsin_inverse_matrix,
            image_header.metadata.tone_mapping.intensity_target,
            &image_header.metadata.colour_encoding,
        );
    }

    let mut color_and_ec_floats = if is_gray {
        vec![color_floats[0].clone()]
    } else {
        vec![
            color_floats[0].clone(),
            color_floats[1].clone(),
            color_floats[2].clone(),
        ]
    };
    color_and_ec_floats.extend(ec_floats);

    // Apply `Patches` if present.
    if let Some(ref patch_list) = patches {
        apply_patches(
            &mut color_and_ec_floats,
            patch_list,
            ref_frames,
            color_ch_count,
        );
    }

    // Save reference frame if requested.
    if (frame_header.frame_type == FrameType::ReferenceOnly || frame_header.can_reference())
        && (frame_header.save_as_reference as usize) < 4
    {
        ref_frames[frame_header.save_as_reference as usize] = Some(color_and_ec_floats.clone());
    }

    Ok(InternalDecodedFrame {
        header: frame_header,
        channels: color_and_ec_floats,
    })
}

fn modular_grid_to_float_grid(grid: &ModularGrid, bit_depth: BitDepthHeader) -> FloatGrid {
    let mut out = FloatGrid::new(grid.width, grid.height);
    for (o, &s) in out.data.iter_mut().zip(&grid.data) {
        *o = bit_depth.parse_integer_sample(s);
    }
    out
}

fn extract_modular_color_and_ec_floats(
    grids: &[ModularGrid],
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_dequant: &LfChannelDequantization,
) -> ([FloatGrid; 3], Vec<FloatGrid>) {
    let n_color = frame_header.encoded_color_channels;
    let bd = image_header.metadata.bit_depth;
    let w = frame_header.color_sample_width() as usize;
    let h = frame_header.color_sample_height() as usize;

    let mut color_floats: [FloatGrid; 3] = if n_color == 1 {
        let g0 = grids
            .first()
            .map(|g| modular_grid_to_float_grid(g, bd))
            .unwrap_or_else(|| FloatGrid::new(w, h));
        [g0.clone(), g0.clone(), g0]
    } else if image_header.metadata.xyb_encoded && grids.len() >= 3 {
        // Modular XYB stores `[Y, X, B - Y]`.
        let y_mod = &grids[0];
        let x_mod = &grids[1];
        let b_mod = &grids[2];
        let mut x_f = FloatGrid::new(x_mod.width, x_mod.height);
        let mut y_f = FloatGrid::new(y_mod.width, y_mod.height);
        let mut b_f = FloatGrid::new(b_mod.width, b_mod.height);
        let mx = lf_dequant.m_x_lf / 128.0;
        let my = lf_dequant.m_y_lf / 128.0;
        let mb = lf_dequant.m_b_lf / 128.0;
        for i in 0..y_mod.data.len() {
            let yv = y_mod.data[i];
            let xv = x_mod.data[i];
            let bv = b_mod.data[i].saturating_add(yv);
            x_f.data[i] = xv as f32 * mx;
            y_f.data[i] = yv as f32 * my;
            b_f.data[i] = bv as f32 * mb;
        }
        [x_f, y_f, b_f]
    } else {
        std::array::from_fn(|c| {
            let g = grids
                .get(c)
                .map(|m| modular_grid_to_float_grid(m, bd))
                .unwrap_or_else(|| FloatGrid::new(w, h));
            if g.width != w || g.height != h {
                upsample_nn_grid(&g, w, h)
            } else {
                g
            }
        })
    };

    // Apply Gaborish and EPF if enabled on Modular frame.
    if let GaborFilter::Enabled(weights) = frame_header.restoration_filter.gab {
        apply_gaborish(&mut color_floats, weights);
    }
    if let EdgePreservingFilter::Enabled(ref epf_params) = frame_header.restoration_filter.epf {
        apply_epf(&mut color_floats, frame_header, epf_params, None);
    }

    let mut ecs = Vec::with_capacity(grids.len().saturating_sub(n_color));
    for (ec_idx, ec_grid) in grids.iter().skip(n_color).enumerate() {
        let ec_bd = image_header
            .metadata
            .ec_info
            .get(ec_idx)
            .map(|e| e.bit_depth)
            .unwrap_or(bd);
        ecs.push(modular_grid_to_float_grid(ec_grid, ec_bd));
    }
    (color_floats, ecs)
}

fn ycbcr_grids_to_rgb(cbycr: &mut [FloatGrid; 3]) {
    let len = cbycr[0].data.len();
    for i in 0..len {
        let cb = cbycr[0].data[i];
        let y = cbycr[1].data[i] + 128.0 / 255.0;
        let cr = cbycr[2].data[i];
        let r = cr.mul_add(1.402, y);
        let g = cb.mul_add(
            -0.114 * 1.772 / 0.587,
            cr.mul_add(-0.299 * 1.402 / 0.587, y),
        );
        let b = cb.mul_add(1.772, y);
        cbycr[0].data[i] = r;
        cbycr[1].data[i] = g;
        cbycr[2].data[i] = b;
    }
}

fn upsample_nn_grid(src: &FloatGrid, dst_w: usize, dst_h: usize) -> FloatGrid {
    if src.width == 0 || src.height == 0 {
        return FloatGrid::new(dst_w, dst_h);
    }
    let mut out = FloatGrid::new(dst_w, dst_h);
    for y in 0..dst_h {
        let sy = (y * src.height / dst_h).min(src.height - 1);
        for x in 0..dst_w {
            let sx = (x * src.width / dst_w).min(src.width - 1);
            out.set(x, y, src.get(sx, sy));
        }
    }
    out
}

// ============================================================================
// ISO/IEC 18181-1 Annex J.1: Splines Decoding & Centripetal Catmull-Rom Rendering
// ============================================================================

#[derive(Debug, Clone)]
struct QuantSpline {
    quant_points: Vec<(i64, i64)>,
    xyb_dct: [[i32; 32]; 3],
    sigma_dct: [i32; 32],
}

#[derive(Debug, Clone)]
struct ParsedSplines {
    quant_splines: Vec<QuantSpline>,
    quant_adjust: i32,
}

fn parse_splines(
    br: &mut BitReader<'_>,
    frame_header: &FrameHeader,
) -> Result<ParsedSplines, JxlError> {
    const MAX_NUM_SPLINES: usize = 1 << 24;
    const MAX_NUM_CONTROL_POINTS: usize = 1 << 20;

    let mut dec = EntropyDecoder::parse(br, 6)?;
    dec.begin(br)?;

    let num_splines_minus1 = dec.read_varint(br, 2)? as usize;
    let num_pixels = (frame_header.width as usize).saturating_mul(frame_header.height as usize);
    let max_num_splines = MAX_NUM_SPLINES.min(num_pixels / 4);
    if num_splines_minus1 >= max_num_splines {
        return Err(JxlError::LimitExceeded("too many splines"));
    }
    let num_splines = num_splines_minus1 + 1;

    let mut start_points = vec![(0i64, 0i64); num_splines];
    let mut prev_point = (
        i64::from(dec.read_varint(br, 1)?),
        i64::from(dec.read_varint(br, 1)?),
    );
    start_points[0] = prev_point;
    for next_point in &mut start_points[1..] {
        let dx = i64::from(unpack_signed(dec.read_varint(br, 1)?));
        let dy = i64::from(unpack_signed(dec.read_varint(br, 1)?));
        prev_point.0 = prev_point.0.saturating_add(dx);
        prev_point.1 = prev_point.1.saturating_add(dy);
        *next_point = prev_point;
    }

    let quant_adjust = unpack_signed(dec.read_varint(br, 0)?);
    let mut quant_splines = Vec::with_capacity(num_splines);
    let mut acc_control_points = 0usize;
    let max_num_points = MAX_NUM_CONTROL_POINTS.min(num_pixels / 2);

    for start_point in start_points {
        let num_points = dec.read_varint(br, 3)? as usize;
        acc_control_points = acc_control_points.saturating_add(num_points);
        if acc_control_points > max_num_points {
            return Err(JxlError::LimitExceeded("too many spline control points"));
        }
        let mut quant_points = Vec::with_capacity(1 + num_points);
        let mut cur_value = start_point;
        let mut cur_delta = (0i64, 0i64);
        quant_points.push(cur_value);
        for _ in 0..num_points {
            let prev_value = cur_value;
            let delta_x = i64::from(unpack_signed(dec.read_varint(br, 4)?));
            let delta_y = i64::from(unpack_signed(dec.read_varint(br, 4)?));
            cur_delta.0 = cur_delta.0.saturating_add(delta_x);
            cur_delta.1 = cur_delta.1.saturating_add(delta_y);
            cur_value.0 = cur_value
                .0
                .checked_add(cur_delta.0)
                .ok_or(JxlError::InvalidBitstream("spline control point overflow"))?;
            cur_value.1 = cur_value
                .1
                .checked_add(cur_delta.1)
                .ok_or(JxlError::InvalidBitstream("spline control point overflow"))?;
            if cur_value == prev_value {
                return Err(JxlError::InvalidBitstream(
                    "duplicate consecutive spline control points",
                ));
            }
            quant_points.push(cur_value);
        }

        let mut xyb_dct = [[0i32; 32]; 3];
        for ch_dct in &mut xyb_dct {
            for coeff in ch_dct {
                *coeff = unpack_signed(dec.read_varint(br, 5)?);
            }
        }
        let mut sigma_dct = [0i32; 32];
        for coeff in &mut sigma_dct {
            *coeff = unpack_signed(dec.read_varint(br, 5)?);
        }
        quant_splines.push(QuantSpline {
            quant_points,
            xyb_dct,
            sigma_dct,
        });
    }

    dec.finalize()?;
    Ok(ParsedSplines {
        quant_splines,
        quant_adjust,
    })
}

#[derive(Debug, Default, Clone, Copy)]
struct SplinePoint {
    x: f32,
    y: f32,
}

impl SplinePoint {
    #[inline]
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    fn mirror(self, center: Self) -> Self {
        Self {
            x: center.x + center.x - self.x,
            y: center.y + center.y - self.y,
        }
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }

    #[inline]
    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }

    #[inline]
    fn norm_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    #[inline]
    fn norm(self) -> f32 {
        self.norm_squared().sqrt()
    }
}

struct SplineArc {
    point: SplinePoint,
    length: f32,
}

fn upsample_centripetal_catmull_rom(points: &[SplinePoint]) -> Vec<SplinePoint> {
    if points.len() <= 1 {
        return points.to_vec();
    }
    let mut ext = Vec::with_capacity(points.len() + 2);
    ext.push(points[1].mirror(points[0]));
    ext.extend_from_slice(points);
    ext.push(points[points.len() - 2].mirror(points[points.len() - 1]));

    let mut upsampled = Vec::with_capacity(16 * (ext.len() - 3) + 1);
    for i in 0..(ext.len() - 3) {
        let p = [ext[i], ext[i + 1], ext[i + 2], ext[i + 3]];
        upsampled.push(p[1]);
        let mut t = [0.0f32; 4];
        for k in 1..4 {
            t[k] = t[k - 1] + p[k].sub(p[k - 1]).norm_squared().powf(0.25);
        }
        for step in 1..16 {
            let knot = t[1] + (step as f32 / 16.0) * (t[2] - t[1]);
            let mut a = [SplinePoint::default(); 3];
            for k in 0..3 {
                a[k] = p[k].add(p[k + 1].sub(p[k]).mul((knot - t[k]) / (t[k + 1] - t[k])));
            }
            let mut b = [SplinePoint::default(); 2];
            for k in 0..2 {
                b[k] = a[k].add(a[k + 1].sub(a[k]).mul((knot - t[k]) / (t[k + 2] - t[k])));
            }
            upsampled.push(b[0].add(b[1].sub(b[0]).mul((knot - t[1]) / (t[2] - t[1]))));
        }
    }
    upsampled.push(points[points.len() - 1]);
    upsampled
}

fn sample_spline_arcs(points: &[SplinePoint]) -> Vec<SplineArc> {
    let upsampled = upsample_centripetal_catmull_rom(points);
    let mut current = upsampled[0];
    let mut next_idx = 0usize;
    let mut all_samples = vec![SplineArc {
        point: current,
        length: 1.0,
    }];

    while next_idx < upsampled.len() {
        let mut prev = current;
        let mut arclength = 0.0f32;
        loop {
            if next_idx >= upsampled.len() {
                all_samples.push(SplineArc {
                    point: prev,
                    length: arclength,
                });
                break;
            }
            let next = upsampled[next_idx];
            let d = next.sub(prev).norm();
            if arclength + d >= 1.0 {
                current = prev.add(next.sub(prev).mul((1.0 - arclength) / d));
                all_samples.push(SplineArc {
                    point: current,
                    length: 1.0,
                });
                break;
            }
            arclength += d;
            prev = next;
            next_idx += 1;
        }
    }
    all_samples
}

#[inline]
fn continuous_idct(dct: &[f32; 32], t: f32) -> f32 {
    let mut res = dct[0];
    for (i, &c) in dct.iter().enumerate().skip(1) {
        let theta = (i as f32) * (std::f32::consts::PI / 32.0) * (t + 0.5);
        res += std::f32::consts::SQRT_2 * c * theta.cos();
    }
    res
}

#[allow(clippy::excessive_precision)]
#[inline]
fn erf_approx(x: f32) -> f32 {
    let ax = x.abs();
    let d1 = ax * 7.77394369e-02 + 2.05260015e-04;
    let d2 = d1 * ax + 2.32120216e-01;
    let d3 = d2 * ax + 2.77820801e-01;
    let d4 = d3 * ax + 1.0;
    let d5 = d4 * d4;
    let inv_d5 = 1.0 / d5;
    let result = -inv_d5 * inv_d5 + 1.0;
    if x < 0.0 { -result } else { result }
}

fn render_splines(
    frame_header: &FrameHeader,
    color_floats: &mut [FloatGrid; 3],
    splines: &ParsedSplines,
    base_correlations_xb: Option<(f32, f32)>,
) {
    const CHANNEL_WEIGHTS: [f32; 4] = [0.0042, 0.075, 0.07, 0.3333];
    let qa = splines.quant_adjust as f32;
    let inverted_qa = if qa >= 0.0 {
        1.0 / (1.0 + qa / 8.0)
    } else {
        1.0 - qa / 8.0
    };
    let (corr_x, corr_b) = base_correlations_xb.unwrap_or((0.0, 1.0));

    for qs in &splines.quant_splines {
        let points: Vec<SplinePoint> = qs
            .quant_points
            .iter()
            .map(|&(x, y)| SplinePoint::new(x as f32, y as f32))
            .collect();
        let mut xyb_dct = [[0.0f32; 32]; 3];
        for c in 0..3 {
            for i in 0..32 {
                xyb_dct[c][i] = (qs.xyb_dct[c][i] as f32) * CHANNEL_WEIGHTS[c] * inverted_qa;
            }
        }
        for i in 0..32 {
            let y_val = xyb_dct[1][i];
            xyb_dct[0][i] += corr_x * y_val;
            xyb_dct[2][i] += corr_b * y_val;
        }
        let mut sigma_dct = [0.0f32; 32];
        for (dst, &q) in sigma_dct.iter_mut().zip(&qs.sigma_dct) {
            *dst = (q as f32) * CHANNEL_WEIGHTS[3] * inverted_qa;
        }

        let all_samples = sample_spline_arcs(&points);
        let arclength = (all_samples.len() as f32) - 2.0 + all_samples.last().unwrap().length;
        for (i, arc) in all_samples.iter().enumerate() {
            let arclength_from_start = if arclength <= 0.0 {
                0.0
            } else {
                ((i as f32) / arclength).min(1.0)
            };
            let t = 31.0 * arclength_from_start;
            let sigma = continuous_idct(&sigma_dct, t);
            if sigma.abs() < 1e-7 {
                continue;
            }
            let inv_sigma = 1.0 / sigma;
            let values = [
                continuous_idct(&xyb_dct[0], t) * arc.length,
                continuous_idct(&xyb_dct[1], t) * arc.length,
                continuous_idct(&xyb_dct[2], t) * arc.length,
            ];
            let max_color = 0.01f32.max(values[0].max(values[1]).max(values[2]));
            let max_distance = (2.0 * (10.0f32.ln() * 3.0 + max_color)).sqrt() * sigma.abs();

            let xbegin = 0.max((arc.point.x - max_distance + 0.5).floor() as i32) as usize;
            let xend = (frame_header.width as i32)
                .min((arc.point.x + max_distance + 1.5).floor() as i32)
                .max(0) as usize;
            let ybegin = 0.max((arc.point.y - max_distance + 0.5).floor() as i32) as usize;
            let yend = (frame_header.height as i32)
                .min((arc.point.y + max_distance + 1.5).floor() as i32)
                .max(0) as usize;

            for (ch, grid) in color_floats.iter_mut().enumerate() {
                let x_lim = xend.min(grid.width);
                let y_lim = yend.min(grid.height);
                for y in ybegin..y_lim {
                    let dy = (y as f32) - arc.point.y;
                    let row = grid.row_mut(y);
                    for (x, sample) in row.iter_mut().enumerate().take(x_lim).skip(xbegin) {
                        let dx = (x as f32) - arc.point.x;
                        let distance = (dx * dx + dy * dy).sqrt();
                        const SQRT_0125: f32 = 0.353_553_38;
                        let factor = erf_approx((0.5 * distance + SQRT_0125) * inv_sigma)
                            - erf_approx((0.5 * distance - SQRT_0125) * inv_sigma);
                        *sample += 0.25 * values[ch] * sigma * factor * factor;
                    }
                }
            }
        }
    }
}

// ============================================================================
// ISO/IEC 18181-1 Annex J.3: Noise Synthesis (`XorShift128Plus` + 5x5 Kernel)
// ============================================================================

#[derive(Debug, Clone)]
struct NoiseParameters {
    lut: [f32; 8],
}

fn parse_noise(br: &mut BitReader<'_>) -> Result<NoiseParameters, JxlError> {
    let mut lut = [0.0f32; 8];
    for slot in &mut lut {
        *slot = (br.read_bits(10)? as f32) / 1024.0;
    }
    Ok(NoiseParameters { lut })
}

struct XorShift128Plus {
    s0: [Wrapping<u64>; 8],
    s1: [Wrapping<u64>; 8],
}

impl XorShift128Plus {
    fn new(seed0: u64, seed1: u64) -> Self {
        let seed0 = Wrapping(seed0);
        let seed1 = Wrapping(seed1);
        let mut s0 = [Wrapping(0u64); 8];
        let mut s1 = [Wrapping(0u64); 8];
        s0[0] = split_mix_64(seed0 + Wrapping(0x9E37_79B9_7F4A_7C15));
        s1[0] = split_mix_64(seed1 + Wrapping(0x9E37_79B9_7F4A_7C15));
        for i in 1..8 {
            s0[i] = split_mix_64(s0[i - 1]);
            s1[i] = split_mix_64(s1[i - 1]);
        }
        Self { s0, s1 }
    }

    #[inline]
    fn get_u32_bits(&mut self) -> [u32; 16] {
        let mut out = [0u32; 16];
        for i in 0..8 {
            let mut s1 = self.s0[i];
            let s0 = self.s1[i];
            let ret = (s1 + s0).0;
            self.s0[i] = s0;
            s1 ^= s1 << 23;
            self.s1[i] = s1 ^ (s0 ^ (s1 >> 18) ^ (s0 >> 5));
            out[2 * i] = (ret & 0xffff_ffff) as u32;
            out[2 * i + 1] = (ret >> 32) as u32;
        }
        out
    }
}

#[inline]
fn split_mix_64(z: Wrapping<u64>) -> Wrapping<u64> {
    let z = (z ^ (z >> 30)) * Wrapping(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)) * Wrapping(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn render_noise(
    frame_header: &FrameHeader,
    visible_frames_num: usize,
    invisible_frames_num: usize,
    base_correlations_xb: Option<(f32, f32)>,
    color_floats: &mut [FloatGrid; 3],
    params: &NoiseParameters,
) {
    let width = frame_header.width as usize;
    let height = frame_header.height as usize;
    if width == 0 || height == 0 {
        return;
    }
    let group_dim = frame_header.group_dim() as usize;
    let groups_per_row = width.div_ceil(group_dim);
    let num_groups = groups_per_row * height.div_ceil(group_dim);
    let seed0 = ((visible_frames_num as u64) << 32) + (invisible_frames_num as u64);

    let mut raw_noise = [
        FloatGrid::new(width, height),
        FloatGrid::new(width, height),
        FloatGrid::new(width, height),
    ];

    for g_idx in 0..num_groups {
        let gx = g_idx % groups_per_row;
        let gy = g_idx / groups_per_row;
        let x0 = gx * group_dim;
        let y0 = gy * group_dim;
        let gw = group_dim.min(width - x0);
        let gh = group_dim.min(height - y0);
        let seed1 = ((x0 as u64) << 32) + (y0 as u64);
        let width_n2 = gw.div_ceil(16);
        let mut rng = XorShift128Plus::new(seed0, seed1);

        for c in &mut raw_noise {
            for y in 0..gh {
                for bx in 0..width_n2 {
                    let bits = rng.get_u32_bits();
                    for (k, &b) in bits.iter().enumerate() {
                        let dx = bx * 16 + k;
                        if dx < gw {
                            let f = f32::from_bits((b >> 9) | 0x3f80_0000);
                            c.set(x0 + dx, y0 + y, f);
                        }
                    }
                }
            }
        }
    }

    let mut convolved = [
        FloatGrid::new(width, height),
        FloatGrid::new(width, height),
        FloatGrid::new(width, height),
    ];
    for (src, dst) in raw_noise.iter().zip(convolved.iter_mut()) {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0f32;
                for dy in -2isize..=2 {
                    let sy = mirror_idx(y as isize + dy, height);
                    let row = src.row(sy);
                    for dx in -2isize..=2 {
                        let sx = mirror_idx(x as isize + dx, width);
                        sum += row[sx] * 0.16;
                    }
                }
                dst.set(x, y, sum - src.get(x, y) * 4.0);
            }
        }
    }

    let (corr_x, corr_b) = base_correlations_xb.unwrap_or((0.0, 1.0));
    let mut lut = [0.0f32; 9];
    lut[..8].copy_from_slice(&params.lut);
    lut[8] = params.lut[7];

    for y in 0..height {
        for x in 0..width {
            let gx = color_floats[0].get(x, y);
            let gy = color_floats[1].get(x, y);
            let nx_raw = convolved[0].get(x, y);
            let ny_raw = convolved[1].get(x, y);
            let nb_raw = convolved[2].get(x, y);

            let in_scaled_x = ((gx + gy) * 3.0).max(0.0);
            let in_scaled_y = ((gy - gx) * 3.0).max(0.0);
            let ix = (in_scaled_x as usize).min(7);
            let fx = in_scaled_x - (ix as f32);
            let iy = (in_scaled_y as usize).min(7);
            let fy = in_scaled_y - (iy as f32);

            let sx = (lut[ix + 1] - lut[ix]) * fx + lut[ix];
            let sy = (lut[iy + 1] - lut[iy]) * fy + lut[iy];
            let nx = 0.22 * sx * (0.0078125 * nx_raw + 0.9921875 * nb_raw);
            let ny = 0.22 * sy * (0.0078125 * ny_raw + 0.9921875 * nb_raw);

            color_floats[0].set(x, y, gx + corr_x * (nx + ny) + nx - ny);
            color_floats[1].set(x, y, gy + nx + ny);
            color_floats[2].set(x, y, color_floats[2].get(x, y) + corr_b * (nx + ny));
        }
    }
}

#[inline]
fn mirror_idx(mut idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    if idx < 0 {
        idx = -(idx + 1);
    }
    if (idx as usize) >= len {
        idx = (2 * len as isize - 1 - idx).max(0);
    }
    (idx as usize).min(len - 1)
}

// ============================================================================
// ISO/IEC 18181-1 Annex J.4: Patches Decoding & Application
// ============================================================================

#[derive(Debug)]
struct ParsedPatchTarget {
    x: i32,
    y: i32,
    modes: Vec<(u32, u32, bool)>,
}

#[derive(Debug)]
struct ParsedPatchRef {
    ref_idx: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    targets: Vec<ParsedPatchTarget>,
}

fn parse_patches(
    br: &mut BitReader<'_>,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
) -> Result<Vec<ParsedPatchRef>, JxlError> {
    let num_extra = image_header.metadata.ec_info.len();
    let alpha_indices: Vec<u32> = image_header
        .metadata
        .ec_info
        .iter()
        .enumerate()
        .filter_map(|(i, ec)| ec.is_alpha().then_some(i as u32))
        .collect();

    let mut dec = EntropyDecoder::parse(br, 10)?;
    dec.begin(br)?;

    let max_refs = (1u32 << 24).min(
        ((u64::from(frame_header.width) * u64::from(frame_header.height)) / 16).max(1024) as u32,
    );
    let num_refs = dec.read_varint(br, 0)?;
    if num_refs > max_refs {
        return Err(JxlError::LimitExceeded("too many patch references"));
    }

    let mut refs = Vec::with_capacity(num_refs as usize);
    let mut total_patches = 0u32;
    for _ in 0..num_refs {
        let ref_idx = dec.read_varint(br, 1)? as usize;
        if ref_idx >= 4 {
            return Err(JxlError::InvalidBitstream("patch ref_idx >= 4"));
        }
        let x0 = dec.read_varint(br, 3)? as usize;
        let y0 = dec.read_varint(br, 3)? as usize;
        let width = dec.read_varint(br, 2)? as usize + 1;
        let height = dec.read_varint(br, 2)? as usize + 1;
        let count = dec.read_varint(br, 7)? + 1;
        total_patches = total_patches.saturating_add(count);
        if total_patches > max_refs.saturating_mul(4) {
            return Err(JxlError::LimitExceeded("too many total patches"));
        }

        let mut prev_xy: Option<(i32, i32)> = None;
        let mut targets = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let (x, y) = if let Some((px, py)) = prev_xy {
                let dx = unpack_signed(dec.read_varint(br, 6)?);
                let dy = unpack_signed(dec.read_varint(br, 6)?);
                (
                    px.checked_add(dx)
                        .ok_or(JxlError::InvalidBitstream("patch x overflow"))?,
                    py.checked_add(dy)
                        .ok_or(JxlError::InvalidBitstream("patch y overflow"))?,
                )
            } else {
                (
                    dec.read_varint(br, 4)? as i32,
                    dec.read_varint(br, 4)? as i32,
                )
            };
            prev_xy = Some((x, y));

            let mut modes = Vec::with_capacity(num_extra + 1);
            for _ in 0..(num_extra + 1) {
                let raw_mode = dec.read_varint(br, 5)?;
                if raw_mode > 7 {
                    return Err(JxlError::InvalidBitstream("invalid PatchBlendMode"));
                }
                let alpha_ch = if raw_mode >= 4 && alpha_indices.len() >= 2 {
                    dec.read_varint(br, 8)?
                } else {
                    alpha_indices.first().copied().unwrap_or(0)
                };
                let clamp = if raw_mode >= 3 {
                    dec.read_varint(br, 9)? != 0
                } else {
                    false
                };
                modes.push((raw_mode, alpha_ch, clamp));
            }
            targets.push(ParsedPatchTarget { x, y, modes });
        }
        refs.push(ParsedPatchRef {
            ref_idx,
            x0,
            y0,
            width,
            height,
            targets,
        });
    }
    dec.finalize()?;
    Ok(refs)
}

fn apply_patches(
    channels: &mut [FloatGrid],
    patches: &[ParsedPatchRef],
    ref_frames: &[Option<Vec<FloatGrid>>; 4],
    color_ch_count: usize,
) {
    for pref in patches {
        let Some(ref ref_chs) = ref_frames[pref.ref_idx] else {
            continue;
        };
        for target in &pref.targets {
            for (ch_idx, dst_grid) in channels.iter_mut().enumerate() {
                let Some(src_grid) = ref_chs.get(ch_idx) else {
                    continue;
                };
                let mode_idx = if ch_idx < color_ch_count {
                    0
                } else {
                    1 + (ch_idx - color_ch_count)
                };
                let Some(&(raw_mode, _alpha_ch, clamp)) = target.modes.get(mode_idx) else {
                    continue;
                };
                if raw_mode == 0 {
                    continue;
                }
                for dy in 0..pref.height {
                    let sy = pref.y0 + dy;
                    let ty = target.y + dy as i32;
                    if sy >= src_grid.height || ty < 0 || (ty as usize) >= dst_grid.height {
                        continue;
                    }
                    for dx in 0..pref.width {
                        let sx = pref.x0 + dx;
                        let tx = target.x + dx as i32;
                        if sx >= src_grid.width || tx < 0 || (tx as usize) >= dst_grid.width {
                            continue;
                        }
                        let mut sv = src_grid.get(sx, sy);
                        if clamp {
                            sv = sv.clamp(0.0, 1.0);
                        }
                        let dv = dst_grid.get(tx as usize, ty as usize);
                        let out = match raw_mode {
                            1 => sv,
                            2 => dv + sv,
                            3 => dv * sv,
                            _ => sv,
                        };
                        dst_grid.set(tx as usize, ty as usize, out);
                    }
                }
            }
        }
    }
}

fn blend_onto_canvas(
    canvas: &mut [FloatGrid],
    frame_chs: &[FloatGrid],
    frame_header: &FrameHeader,
    image_header: &ImageHeader,
    is_gray: bool,
) {
    let color_ch = if is_gray { 1 } else { 3 };
    let x0 = frame_header.x0;
    let y0 = frame_header.y0;

    for (ch_idx, (dst, src)) in canvas.iter_mut().zip(frame_chs).enumerate() {
        let b_info = if ch_idx < color_ch {
            &frame_header.blending_info
        } else {
            frame_header
                .ec_blending_info
                .get(ch_idx - color_ch)
                .unwrap_or(&frame_header.blending_info)
        };
        let alpha_ch_idx = color_ch + b_info.alpha_channel as usize;
        let premul = image_header
            .metadata
            .ec_info
            .get(b_info.alpha_channel as usize)
            .and_then(|ec| ec.alpha_associated())
            .unwrap_or(false);

        for sy in 0..src.height {
            let dy = y0 + sy as i32;
            if dy < 0 || (dy as usize) >= dst.height {
                continue;
            }
            let dy = dy as usize;
            for sx in 0..src.width {
                let dx = x0 + sx as i32;
                if dx < 0 || (dx as usize) >= dst.width {
                    continue;
                }
                let dx = dx as usize;
                let mut sv = src.get(sx, sy);
                if b_info.clamp {
                    sv = sv.clamp(0.0, 1.0);
                }
                let dv = dst.get(dx, dy);
                let out = match b_info.mode {
                    BlendMode::Replace => sv,
                    BlendMode::Add => dv + sv,
                    BlendMode::Mul => dv * sv,
                    BlendMode::Blend => {
                        let a_new = frame_chs
                            .get(alpha_ch_idx)
                            .map(|ag| ag.get(sx, sy))
                            .unwrap_or(1.0);
                        let a_new = if b_info.clamp {
                            a_new.clamp(0.0, 1.0)
                        } else {
                            a_new
                        };
                        if ch_idx == alpha_ch_idx {
                            dv + a_new * (1.0 - dv)
                        } else if premul {
                            sv + dv * (1.0 - a_new)
                        } else {
                            sv * a_new + dv * (1.0 - a_new)
                        }
                    }
                    BlendMode::MulAdd => {
                        if ch_idx == alpha_ch_idx {
                            dv
                        } else {
                            let a_new = frame_chs
                                .get(alpha_ch_idx)
                                .map(|ag| ag.get(sx, sy))
                                .unwrap_or(1.0);
                            dv + a_new * sv
                        }
                    }
                };
                dst.set(dx, dy, out);
            }
        }
    }
}

fn apply_orientation(grid: &FloatGrid, orientation: u32) -> FloatGrid {
    let w = grid.width;
    let h = grid.height;
    match orientation {
        1 => grid.clone(),
        2 => {
            let mut out = FloatGrid::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    out.set(w - 1 - x, y, grid.get(x, y));
                }
            }
            out
        }
        3 => {
            let mut out = FloatGrid::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    out.set(w - 1 - x, h - 1 - y, grid.get(x, y));
                }
            }
            out
        }
        4 => {
            let mut out = FloatGrid::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    out.set(x, h - 1 - y, grid.get(x, y));
                }
            }
            out
        }
        5 => {
            let mut out = FloatGrid::new(h, w);
            for y in 0..h {
                for x in 0..w {
                    out.set(y, x, grid.get(x, y));
                }
            }
            out
        }
        6 => {
            let mut out = FloatGrid::new(h, w);
            for y in 0..h {
                for x in 0..w {
                    out.set(h - 1 - y, x, grid.get(x, y));
                }
            }
            out
        }
        7 => {
            let mut out = FloatGrid::new(h, w);
            for y in 0..h {
                for x in 0..w {
                    out.set(h - 1 - y, w - 1 - x, grid.get(x, y));
                }
            }
            out
        }
        8 => {
            let mut out = FloatGrid::new(h, w);
            for y in 0..h {
                for x in 0..w {
                    out.set(y, w - 1 - x, grid.get(x, y));
                }
            }
            out
        }
        _ => grid.clone(),
    }
}

fn pack_float_channels(channels: &[FloatGrid], bit_depth: JxlBitDepth) -> Vec<u8> {
    if channels.is_empty() {
        return Vec::new();
    }
    let num_px = channels[0].data.len();
    let n_ch = channels.len();
    match bit_depth {
        JxlBitDepth::Uint8(bits) => {
            let max_v = ((1u32 << bits) - 1) as f32;
            let scale = 255.0 / max_v.max(1.0);
            let mut out = Vec::with_capacity(num_px * n_ch);
            for i in 0..num_px {
                for ch in channels {
                    let raw = (ch.data[i].clamp(0.0, 1.0) * max_v).round();
                    let v8 = (raw * scale).round().clamp(0.0, 255.0) as u8;
                    out.push(v8);
                }
            }
            out
        }
        JxlBitDepth::Uint16(bits) => {
            let max_v = ((1u32 << bits) - 1) as f32;
            let scale = 65535.0 / max_v.max(1.0);
            let mut out = Vec::with_capacity(num_px * n_ch * 2);
            for i in 0..num_px {
                for ch in channels {
                    let raw = (ch.data[i].clamp(0.0, 1.0) * max_v).round();
                    let v16 = (raw * scale).round().clamp(0.0, 65535.0) as u16;
                    out.extend_from_slice(&v16.to_le_bytes());
                }
            }
            out
        }
        JxlBitDepth::Float32 => {
            let mut out = Vec::with_capacity(num_px * n_ch * 4);
            for i in 0..num_px {
                for ch in channels {
                    out.extend_from_slice(&ch.data[i].to_le_bytes());
                }
            }
            out
        }
    }
}

impl JxlFrame {
    /// Converts this `JxlFrame` into an 8-bit `Rgba8` `rustmpeg_core::VideoFrame`.
    pub fn to_video_frame(&self) -> Result<VideoFrame, JxlError> {
        let rgba8 = self.to_rgba8();
        let stride = (self.width as usize) * 4;
        let frame = VideoFrame::from_planes(
            PixelFormat::Rgba8,
            8,
            self.width,
            self.height,
            ColorDescription::default(),
            [
                Plane::new(Buffer::from_vec(rgba8), stride, self.width, self.height),
                Plane::empty(),
                Plane::empty(),
            ],
        )?;
        Ok(frame)
    }

    /// Converts the frame's pixel buffer into 8-bit interleaved RGBA (`width * height * 4` bytes).
    #[must_use]
    pub fn to_rgba8(&self) -> Vec<u8> {
        let num_px = (self.width * self.height) as usize;
        let n_ch = self.layout.channels();
        let mut out = Vec::with_capacity(num_px * 4);

        for i in 0..num_px {
            let mut ch_u8 = [0u8, 0, 0, 255];
            for (c, dst) in ch_u8.iter_mut().enumerate().take(n_ch) {
                let idx = i * n_ch + c;
                *dst = match self.bit_depth {
                    JxlBitDepth::Uint8(_) => self.data[idx],
                    JxlBitDepth::Uint16(_) => {
                        let lo = self.data[idx * 2];
                        let hi = self.data[idx * 2 + 1];
                        ((u32::from(u16::from_le_bytes([lo, hi])) * 255 + 32767) / 65535) as u8
                    }
                    JxlBitDepth::Float32 => {
                        let b = &self.data[idx * 4..idx * 4 + 4];
                        let f = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        (f.clamp(0.0, 1.0) * 255.0).round() as u8
                    }
                };
            }
            match self.layout {
                JxlPixelLayout::Gray => {
                    let g = ch_u8[0];
                    out.extend_from_slice(&[g, g, g, 255]);
                }
                JxlPixelLayout::GrayAlpha => {
                    let g = ch_u8[0];
                    let a = ch_u8[1];
                    out.extend_from_slice(&[g, g, g, a]);
                }
                JxlPixelLayout::Rgb => {
                    out.extend_from_slice(&[ch_u8[0], ch_u8[1], ch_u8[2], 255]);
                }
                JxlPixelLayout::Rgba => {
                    out.extend_from_slice(&ch_u8);
                }
            }
        }
        out
    }
}
