// SPDX-License-Identifier: Apache-2.0

//! Clean-room JPEG XL v2 encoder (ISO/IEC 18181-1 & 18181-2).
//!
//! Written from scratch in safe Rust under Apache-2.0. Supports:
//! - Lossless and near-lossless Modular encoding with 14 spatial predictors, `SelfCorrecting`
//!   weighted predictor, 4-context adaptive `MaConfig` trees, `YCoCg` RCT (`rct_type = 6`),
//!   automatic lossless `Palette` extraction, and Haar-like `Squeeze` transforms;
//! - Lossy XYB VarDCT encoding with `extra_precision` DC, Chroma-from-Luma (`CfL`) estimation,
//!   adaptive `Dct16`/`Dct8` varblock selection, activity-masked per-block `hf_mul` quantization,
//!   inverse-Gaborish pre-sharpening, and Edge-Preserving Filter (`EPF`) sharpness maps;
//! - Both `Prefix` and 12-bit `rANS` (`Ans`) entropy backends plus `2D LZ77` match finding;
//! - Wide-gamut & HDR color encodings (`Srgb`, `LinearSrgb`, `DisplayP3`, `Bt2020Pq`, `Bt2020Hlg`,
//!   and embedded `Icc` profiles via Annex E `write_icc`);
//! - Bare codestream and ISO/IEC 18181-2 container output (`Exif`, `xml `) and multi-frame animations.

use rustmpeg_core::{PixelFormat, VideoFrame};

use super::bits::BitWriter;
use super::container::{JxlContainerKind, write_container};
use super::entropy::{EntropyEncoder, HybridUintConfig};
use super::headers::{
    BitDepthHeader, BlendingInfo, ColourEncoding, ColourSpace, EdgePreservingFilter,
    EnumColourEncoding, EpfParams, ExtraChannelInfo, ExtraChannelType, FrameEncoding, FrameFlags,
    FrameHeader, FrameType, GaborFilter, ImageHeader, ImageMetadata, OpsinInverseMatrix, Passes,
    Primaries, RenderingIntent, RestorationFilter, SizeHeader, Toc, ToneMapping,
    TransferFunctionHeader, WhitePoint, write_icc,
};
use super::modular::{
    MaConfig, ModularEncodeOptions, ModularGrid, ModularHeader, Predictor, WpHeader,
    encode_modular_subimage_v2,
};
use super::vardct::{
    FloatGrid, LfChannelDequantization, VarDctEncodeOptions, apply_inverse_gaborish,
    apply_inverse_tf, encode_vardct_frame_sections_v2, linear_srgb_to_xyb_pixel,
    target_to_srgb_gamut_matrix,
};
use super::{
    JxlBitDepth, JxlColorEncoding, JxlConfig, JxlEncodingMode, JxlError, JxlFrame, JxlImage,
    JxlPixelLayout,
};

/// Clean-room JPEG XL v2 encoder.
#[derive(Debug, Clone, Default)]
pub struct JxlEncoder {
    config: JxlConfig,
}

impl JxlEncoder {
    /// Creates a new `JxlEncoder` with the given [`JxlConfig`].
    #[must_use]
    pub const fn new(config: JxlConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the encoder configuration.
    #[must_use]
    pub const fn config(&self) -> &JxlConfig {
        &self.config
    }

    /// Encodes a [`JxlImage`] into a JPEG XL byte vector (bare codestream or ISO/IEC 18181-2 container).
    pub fn encode(&self, image: &JxlImage) -> Result<Vec<u8>, JxlError> {
        if image.width == 0 || image.height == 0 {
            return Err(JxlError::InvalidConfig(
                "image width and height must be non-zero",
            ));
        }
        if image.frames.is_empty() {
            return Err(JxlError::InvalidConfig(
                "image must contain at least one frame",
            ));
        }

        let is_vardct = matches!(self.config.mode, JxlEncodingMode::VarDctLossy { .. });
        let is_gray = matches!(
            image.layout,
            JxlPixelLayout::Gray | JxlPixelLayout::GrayAlpha
        );
        let has_alpha = image.layout.has_alpha();

        let bit_depth_header = match image.bit_depth {
            JxlBitDepth::Uint8(bits) => BitDepthHeader::IntegerSample {
                bits_per_sample: u32::from(bits),
            },
            JxlBitDepth::Uint16(bits) => BitDepthHeader::IntegerSample {
                bits_per_sample: u32::from(bits),
            },
            JxlBitDepth::Float32 => BitDepthHeader::FloatSample {
                bits_per_sample: 32,
                exp_bits: 8,
            },
        };

        let mut ec_info = Vec::new();
        if has_alpha {
            ec_info.push(ExtraChannelInfo {
                ty: ExtraChannelType::Alpha {
                    alpha_associated: false,
                },
                bit_depth: bit_depth_header,
                dim_shift: 0,
                name: String::new(),
            });
        }

        let (colour_encoding, icc_bytes_opt) = match &image.color_encoding {
            JxlColorEncoding::Icc(icc) if !icc.is_empty() => {
                let cs = if is_gray && !is_vardct {
                    ColourSpace::Grey
                } else {
                    ColourSpace::Rgb
                };
                (ColourEncoding::IccProfile(cs), Some(icc.as_slice()))
            }
            JxlColorEncoding::Srgb | JxlColorEncoding::Icc(_) => {
                if is_gray && !is_vardct {
                    (ColourEncoding::Enum(EnumColourEncoding::gray_srgb()), None)
                } else {
                    (ColourEncoding::Enum(EnumColourEncoding::srgb()), None)
                }
            }
            JxlColorEncoding::LinearSrgb => {
                let cs = if is_gray && !is_vardct {
                    ColourSpace::Grey
                } else {
                    ColourSpace::Rgb
                };
                (
                    ColourEncoding::Enum(EnumColourEncoding {
                        colour_space: cs,
                        white_point: WhitePoint::D65,
                        primaries: Primaries::Srgb,
                        tf: TransferFunctionHeader::Linear,
                        rendering_intent: RenderingIntent::Relative,
                    }),
                    None,
                )
            }
            JxlColorEncoding::GraySrgb => {
                if is_vardct {
                    (ColourEncoding::Enum(EnumColourEncoding::srgb()), None)
                } else {
                    (ColourEncoding::Enum(EnumColourEncoding::gray_srgb()), None)
                }
            }
            JxlColorEncoding::LinearGray => {
                let cs = if is_vardct {
                    ColourSpace::Rgb
                } else {
                    ColourSpace::Grey
                };
                (
                    ColourEncoding::Enum(EnumColourEncoding {
                        colour_space: cs,
                        white_point: WhitePoint::D65,
                        primaries: Primaries::Srgb,
                        tf: TransferFunctionHeader::Linear,
                        rendering_intent: RenderingIntent::Relative,
                    }),
                    None,
                )
            }
            JxlColorEncoding::DisplayP3 => (
                ColourEncoding::Enum(EnumColourEncoding {
                    colour_space: ColourSpace::Rgb,
                    white_point: WhitePoint::D65,
                    primaries: Primaries::P3,
                    tf: TransferFunctionHeader::Srgb,
                    rendering_intent: RenderingIntent::Relative,
                }),
                None,
            ),
            JxlColorEncoding::Bt2020Pq => (
                ColourEncoding::Enum(EnumColourEncoding {
                    colour_space: ColourSpace::Rgb,
                    white_point: WhitePoint::D65,
                    primaries: Primaries::Bt2100,
                    tf: TransferFunctionHeader::Pq,
                    rendering_intent: RenderingIntent::Relative,
                }),
                None,
            ),
            JxlColorEncoding::Bt2020Hlg => (
                ColourEncoding::Enum(EnumColourEncoding {
                    colour_space: ColourSpace::Rgb,
                    white_point: WhitePoint::D65,
                    primaries: Primaries::Bt2100,
                    tf: TransferFunctionHeader::Hlg,
                    rendering_intent: RenderingIntent::Relative,
                }),
                None,
            ),
        };

        let image_header = ImageHeader {
            size: SizeHeader {
                width: image.width,
                height: image.height,
            },
            metadata: ImageMetadata {
                orientation: 1,
                intrinsic_size: None,
                preview: None,
                animation: image.animation,
                bit_depth: bit_depth_header,
                modular_16bit_buffers: true,
                ec_info,
                xyb_encoded: is_vardct && icc_bytes_opt.is_none(),
                colour_encoding,
                tone_mapping: ToneMapping::default(),
                opsin_inverse_matrix: OpsinInverseMatrix::default(),
            },
        };

        let mut bw = BitWriter::new();
        image_header.write(&mut bw)?;
        if let Some(icc) = icc_bytes_opt {
            write_icc(&mut bw, icc)?;
        }

        let num_frames = image.frames.len();
        for (idx, frame) in image.frames.iter().enumerate() {
            if frame.width != image.width || frame.height != image.height {
                return Err(JxlError::InvalidConfig(
                    "frame dimensions must match image dimensions",
                ));
            }
            let is_last = idx + 1 == num_frames;
            bw.zero_pad_to_byte();
            match self.config.mode {
                JxlEncodingMode::ModularLossless { predictor } => {
                    encode_modular_frame(
                        &mut bw,
                        &image_header,
                        frame,
                        predictor,
                        &self.config,
                        is_last,
                    )?;
                }
                JxlEncodingMode::VarDctLossy { quality } => {
                    encode_vardct_frame(
                        &mut bw,
                        &image_header,
                        frame,
                        quality,
                        &self.config,
                        is_last,
                    )?;
                }
            }
        }

        let codestream = bw.into_bytes();
        let need_container = self.config.force_container
            || image.container_kind == JxlContainerKind::Container
            || image.exif.is_some()
            || image.xmp.is_some();

        if need_container {
            write_container(
                &codestream,
                None,
                image.exif.as_deref(),
                image.xmp.as_deref(),
            )
        } else {
            Ok(codestream)
        }
    }

    /// Encodes a single `rustmpeg_core::VideoFrame` into a JPEG XL byte vector.
    pub fn encode_video_frame(&self, video_frame: &VideoFrame) -> Result<Vec<u8>, JxlError> {
        let jxl_frame = JxlFrame::from_video_frame(video_frame)?;
        let img = JxlImage::from_single_frame(jxl_frame);
        self.encode(&img)
    }
}

fn unpack_frame_to_modular_grids(frame: &JxlFrame) -> Result<Vec<ModularGrid>, JxlError> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let num_px = w * h;
    let n_ch = frame.layout.channels();
    let expected_bytes = num_px * n_ch * frame.bit_depth.bytes_per_sample();
    if frame.data.len() < expected_bytes {
        return Err(JxlError::InvalidConfig("frame data buffer is too short"));
    }

    let mut grids: Vec<ModularGrid> = (0..n_ch).map(|_| ModularGrid::new(w, h)).collect();
    match frame.bit_depth {
        JxlBitDepth::Uint8(bits) => {
            let shift = 8u8.saturating_sub(bits);
            for i in 0..num_px {
                for (c, g) in grids.iter_mut().enumerate() {
                    g.data[i] = i32::from(frame.data[i * n_ch + c] >> shift);
                }
            }
        }
        JxlBitDepth::Uint16(bits) => {
            let shift = 16u8.saturating_sub(bits);
            for i in 0..num_px {
                for (c, g) in grids.iter_mut().enumerate() {
                    let off = (i * n_ch + c) * 2;
                    let v16 = u16::from_le_bytes([frame.data[off], frame.data[off + 1]]);
                    g.data[i] = i32::from(v16 >> shift);
                }
            }
        }
        JxlBitDepth::Float32 => {
            for i in 0..num_px {
                for (c, g) in grids.iter_mut().enumerate() {
                    let off = (i * n_ch + c) * 4;
                    let f = f32::from_le_bytes([
                        frame.data[off],
                        frame.data[off + 1],
                        frame.data[off + 2],
                        frame.data[off + 3],
                    ]);
                    let bits = if f == 0.0 { 0u32 } else { f.to_bits() };
                    g.data[i] = bits as i32;
                }
            }
        }
    }
    Ok(grids)
}

fn encode_modular_frame(
    bw: &mut BitWriter,
    image_header: &ImageHeader,
    frame: &JxlFrame,
    predictor: Predictor,
    config: &JxlConfig,
    is_last: bool,
) -> Result<(), JxlError> {
    let ec_count = image_header.metadata.ec_info.len();
    let encoded_color_channels = if image_header.metadata.grayscale() {
        1
    } else {
        3
    };
    let frame_header = FrameHeader {
        frame_type: FrameType::RegularFrame,
        encoding: FrameEncoding::Modular,
        flags: FrameFlags(FrameFlags::SKIP_ADAPTIVE_LF_SMOOTHING),
        do_ycbcr: false,
        jpeg_upsampling: [0, 0, 0],
        upsampling: 1,
        ec_upsampling: vec![1; ec_count],
        group_size_shift: 1,
        x_qm_scale: 2,
        b_qm_scale: 2,
        passes: Passes::default(),
        lf_level: 0,
        have_crop: false,
        x0: 0,
        y0: 0,
        width: frame.width,
        height: frame.height,
        blending_info: BlendingInfo::default(),
        ec_blending_info: vec![BlendingInfo::default(); ec_count],
        duration: frame.duration_ticks,
        timecode: 0,
        is_last,
        save_as_reference: 0,
        save_before_ct: false,
        name: frame.name.clone(),
        restoration_filter: RestorationFilter::disabled(),
        encoded_color_channels,
        bit_depth: image_header.metadata.bit_depth,
    };
    frame_header.write(bw, image_header)?;

    let mut grids = unpack_frame_to_modular_grids(frame)?;
    if encoded_color_channels == 3 && grids.len() < 3 {
        let g0 = grids[0].clone();
        grids.insert(1, g0.clone());
        grids.insert(2, g0);
    }

    let mod_opts = ModularEncodeOptions {
        predictor,
        use_rct_ycocg: config.use_rct,
        enable_palette: config.enable_palette,
        enable_squeeze: config.enable_squeeze,
        adaptive_tree: config.modular_ma_tree,
        entropy_mode: config.entropy_mode,
        lz77: config.lz77,
        residual_multiplier: u32::from(config.modular_max_error) + 1,
    };

    let group_dim = frame_header.group_dim() as usize;
    let num_groups = frame_header.num_groups() as usize;
    let num_lf_groups = frame_header.num_lf_groups() as usize;

    let sections = if num_groups == 1 {
        let mut sec_bw = BitWriter::new();
        LfChannelDequantization::default().write(&mut sec_bw)?;
        sec_bw.write_bool(false); // GlobalModular: has_global_tree = false
        encode_modular_subimage_v2(&mut sec_bw, &grids, &mod_opts)?;
        vec![sec_bw.into_bytes()]
    } else {
        let mut lf_global_bw = BitWriter::new();
        LfChannelDequantization::default().write(&mut lf_global_bw)?;
        lf_global_bw.write_bool(false); // has_global_tree = false
        let mod_header = ModularHeader {
            use_global_tree: false,
            wp_params: WpHeader::default(),
            transform: Vec::new(),
        };
        mod_header.write(&mut lf_global_bw)?;
        MaConfig::write_single_leaf_tree(&mut lf_global_bw, predictor)?;
        let empty_enc = EntropyEncoder::single_cluster(1, HybridUintConfig::DEFAULT_4_2_0);
        empty_enc.finish(&mut lf_global_bw)?;

        let mut secs = Vec::with_capacity(2 + num_lf_groups + num_groups);
        secs.push(lf_global_bw.into_bytes());
        for _ in 0..num_lf_groups {
            secs.push(Vec::new());
        }
        secs.push(Vec::new()); // HfGlobal (empty for Modular)

        let num_cols = frame_header.groups_per_row() as usize;
        for g_idx in 0..num_groups {
            let gx = g_idx % num_cols;
            let gy = g_idx / num_cols;
            let x0 = gx * group_dim;
            let y0 = gy * group_dim;
            let tile_grids: Vec<ModularGrid> = grids
                .iter()
                .map(|g| g.extract_rect(x0, y0, group_dim, group_dim))
                .collect();
            let mut pg_bw = BitWriter::new();
            encode_modular_subimage_v2(&mut pg_bw, &tile_grids, &mod_opts)?;
            secs.push(pg_bw.into_bytes());
        }
        secs
    };

    let section_sizes: Vec<u32> = sections.iter().map(|s| s.len() as u32).collect();
    Toc::write(bw, &section_sizes)?;
    for sec in sections {
        bw.write_aligned_bytes(&sec);
    }
    Ok(())
}

fn encode_vardct_frame(
    bw: &mut BitWriter,
    image_header: &ImageHeader,
    frame: &JxlFrame,
    quality: f32,
    config: &JxlConfig,
    is_last: bool,
) -> Result<(), JxlError> {
    let ec_count = image_header.metadata.ec_info.len();
    let gab = if config.gaborish {
        GaborFilter::default()
    } else {
        GaborFilter::Disabled
    };
    let epf = if config.epf_iters == 0 {
        EdgePreservingFilter::Disabled
    } else {
        EdgePreservingFilter::Enabled(EpfParams {
            iters: u32::from(config.epf_iters.clamp(1, 3)),
            ..EpfParams::default()
        })
    };
    let restoration_filter = RestorationFilter { gab, epf };

    let frame_header = FrameHeader {
        frame_type: FrameType::RegularFrame,
        encoding: FrameEncoding::VarDct,
        flags: FrameFlags(FrameFlags::SKIP_ADAPTIVE_LF_SMOOTHING),
        do_ycbcr: false,
        jpeg_upsampling: [0, 0, 0],
        upsampling: 1,
        ec_upsampling: vec![1; ec_count],
        group_size_shift: 1,
        x_qm_scale: 2,
        b_qm_scale: 2,
        passes: Passes::default(),
        lf_level: 0,
        have_crop: false,
        x0: 0,
        y0: 0,
        width: frame.width,
        height: frame.height,
        blending_info: BlendingInfo::default(),
        ec_blending_info: vec![BlendingInfo::default(); ec_count],
        duration: frame.duration_ticks,
        timecode: 0,
        is_last,
        save_as_reference: 0,
        save_before_ct: false,
        name: frame.name.clone(),
        restoration_filter,
        encoded_color_channels: 3,
        bit_depth: image_header.metadata.bit_depth,
    };
    frame_header.write(bw, image_header)?;

    let w = frame.width as usize;
    let h = frame.height as usize;
    let num_px = w * h;
    let rgba8 = frame.to_rgba8();
    let mut xyb = [
        FloatGrid::new(w, h),
        FloatGrid::new(w, h),
        FloatGrid::new(w, h),
    ];

    let (tf, to_srgb_mat) = match &image_header.metadata.colour_encoding {
        ColourEncoding::Enum(e) => (
            e.tf,
            target_to_srgb_gamut_matrix(e.primaries, e.white_point),
        ),
        ColourEncoding::IccProfile(_) => (
            TransferFunctionHeader::Srgb,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ),
    };
    let is_identity_gamut = to_srgb_mat == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let intensity_target = image_header.metadata.tone_mapping.intensity_target;

    for i in 0..num_px {
        let rf = f32::from(rgba8[i * 4]) / 255.0;
        let gf = f32::from(rgba8[i * 4 + 1]) / 255.0;
        let bf = f32::from(rgba8[i * 4 + 2]) / 255.0;
        let mut rl = apply_inverse_tf(rf, tf, intensity_target);
        let mut gl = apply_inverse_tf(gf, tf, intensity_target);
        let mut bl = apply_inverse_tf(bf, tf, intensity_target);
        if !is_identity_gamut {
            let r_s = to_srgb_mat[0] * rl + to_srgb_mat[1] * gl + to_srgb_mat[2] * bl;
            let g_s = to_srgb_mat[3] * rl + to_srgb_mat[4] * gl + to_srgb_mat[5] * bl;
            let b_s = to_srgb_mat[6] * rl + to_srgb_mat[7] * gl + to_srgb_mat[8] * bl;
            rl = r_s;
            gl = g_s;
            bl = b_s;
        }
        let [x, y, b] = linear_srgb_to_xyb_pixel(rl, gl, bl);
        xyb[0].data[i] = x;
        xyb[1].data[i] = y;
        xyb[2].data[i] = b;
    }

    if let GaborFilter::Enabled(weights) = restoration_filter.gab {
        apply_inverse_gaborish(&mut xyb, weights);
    }

    let all_mod_grids = unpack_frame_to_modular_grids(frame)?;
    let extra_channels: Vec<ModularGrid> = if frame.layout.has_alpha() {
        vec![all_mod_grids.last().unwrap().clone()]
    } else {
        Vec::new()
    };

    let vardct_opts = VarDctEncodeOptions {
        quality,
        enable_cfl: config.enable_cfl,
        enable_adaptive_quant: config.enable_adaptive_quant,
        enable_adaptive_dct: config.enable_adaptive_dct,
        entropy_mode: config.entropy_mode,
        lz77: config.lz77,
    };
    let sections =
        encode_vardct_frame_sections_v2(&xyb, &extra_channels, &frame_header, &vardct_opts)?;
    let section_sizes: Vec<u32> = sections.iter().map(|s| s.len() as u32).collect();
    Toc::write(bw, &section_sizes)?;
    for sec in sections {
        bw.write_aligned_bytes(&sec);
    }
    Ok(())
}

impl JxlFrame {
    /// Creates a `JxlFrame` from a `rustmpeg_core::VideoFrame`.
    pub fn from_video_frame(frame: &VideoFrame) -> Result<Self, JxlError> {
        let w = frame.width as usize;
        let h = frame.height as usize;
        if w == 0 || h == 0 {
            return Err(JxlError::InvalidConfig("VideoFrame has zero dimensions"));
        }

        let (layout, bit_depth, data) = match frame.format {
            PixelFormat::Rgb8 => {
                let plane = &frame.planes[0];
                let mut out = Vec::with_capacity(w * h * 3);
                for y in 0..h {
                    let row = &plane.data.as_slice()[y * plane.stride..y * plane.stride + w * 3];
                    out.extend_from_slice(row);
                }
                (JxlPixelLayout::Rgb, JxlBitDepth::Uint8(8), out)
            }
            PixelFormat::Rgba8 => {
                let plane = &frame.planes[0];
                let mut out = Vec::with_capacity(w * h * 4);
                for y in 0..h {
                    let row = &plane.data.as_slice()[y * plane.stride..y * plane.stride + w * 4];
                    out.extend_from_slice(row);
                }
                (JxlPixelLayout::Rgba, JxlBitDepth::Uint8(8), out)
            }
            PixelFormat::Bgra8 => {
                let plane = &frame.planes[0];
                let mut out = Vec::with_capacity(w * h * 4);
                for y in 0..h {
                    let row = &plane.data.as_slice()[y * plane.stride..y * plane.stride + w * 4];
                    for px in row.as_chunks::<4>().0 {
                        out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                    }
                }
                (JxlPixelLayout::Rgba, JxlBitDepth::Uint8(8), out)
            }
            PixelFormat::I400 => {
                let plane = &frame.planes[0];
                if frame.bit_depth == 8 {
                    let mut out = Vec::with_capacity(w * h);
                    for y in 0..h {
                        let row = &plane.data.as_slice()[y * plane.stride..y * plane.stride + w];
                        out.extend_from_slice(row);
                    }
                    (JxlPixelLayout::Gray, JxlBitDepth::Uint8(8), out)
                } else {
                    let mut out = Vec::with_capacity(w * h * 2);
                    for y in 0..h {
                        let row =
                            &plane.data.as_slice()[y * plane.stride..y * plane.stride + w * 2];
                        out.extend_from_slice(row);
                    }
                    (
                        JxlPixelLayout::Gray,
                        JxlBitDepth::Uint16(frame.bit_depth),
                        out,
                    )
                }
            }
            PixelFormat::I420 | PixelFormat::I422 | PixelFormat::I444 => {
                let py = &frame.planes[0];
                let pu = &frame.planes[1];
                let pv = &frame.planes[2];
                let (sub_x, sub_y) = frame.format.subsampling();
                let sx = 1usize << sub_x;
                let sy = 1usize << sub_y;
                let y_slice = py.data.as_slice();
                let u_slice = pu.data.as_slice();
                let v_slice = pv.data.as_slice();
                let max_val = ((1u32 << frame.bit_depth) - 1) as f32;
                let mid = (1u32 << (frame.bit_depth - 1)) as f32;
                let mut out = Vec::with_capacity(w * h * 3);
                for y in 0..h {
                    for x in 0..w {
                        let (yv, uv, vv) = if frame.bit_depth == 8 {
                            (
                                f32::from(y_slice[y * py.stride + x]),
                                f32::from(u_slice[(y / sy) * pu.stride + (x / sx)]) - 128.0,
                                f32::from(v_slice[(y / sy) * pv.stride + (x / sx)]) - 128.0,
                            )
                        } else {
                            let yo = y * py.stride + x * 2;
                            let uo = (y / sy) * pu.stride + (x / sx) * 2;
                            let vo = (y / sy) * pv.stride + (x / sx) * 2;
                            let y16 = u16::from_le_bytes([y_slice[yo], y_slice[yo + 1]]);
                            let u16 = u16::from_le_bytes([u_slice[uo], u_slice[uo + 1]]);
                            let v16 = u16::from_le_bytes([v_slice[vo], v_slice[vo + 1]]);
                            (
                                f32::from(y16) * (255.0 / max_val),
                                (f32::from(u16) - mid) * (255.0 / max_val),
                                (f32::from(v16) - mid) * (255.0 / max_val),
                            )
                        };
                        let r = (yv + 1.402 * vv).round().clamp(0.0, 255.0) as u8;
                        let g = (yv - 0.344136 * uv - 0.714136 * vv)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                        let b = (yv + 1.772 * uv).round().clamp(0.0, 255.0) as u8;
                        out.extend_from_slice(&[r, g, b]);
                    }
                }
                (JxlPixelLayout::Rgb, JxlBitDepth::Uint8(8), out)
            }
        };

        Ok(Self {
            width: frame.width,
            height: frame.height,
            layout,
            bit_depth,
            duration_ticks: 0,
            name: String::new(),
            data,
        })
    }
}
