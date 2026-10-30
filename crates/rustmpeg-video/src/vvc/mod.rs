// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! H.266 / VVC (ITU-T H.266 | ISO/IEC 23090-3) intra-only software encoder.
//!
//! Algorithms derived from VTM 24.0 (BSD-3-Clause, Copyright (c) 2010-2020
//! ITU/ISO/IEC). See THIRD_PARTY.md.
//!
//! This is a conformant VVC bitstream writer for a deliberately small subset of
//! the standard:
//!
//! * Main 10 profile signalling, 8-bit 4:2:0, BT.709 limited range;
//! * one IDR_N_LP picture per access unit, one slice, one tile;
//! * `CTU == min CU == 32x32`, quadtree partitioning disabled (no
//!   `split_cu_flag` appears at all);
//! * every coding unit is intra PLANAR or DC (coded through the MPM path,
//!   chosen per block by SAD, PLANAR on ties) with `intra_chroma_pred_mode =
//!   DM`, so no MPM neighbour derivation is needed;
//! * QP 4 residual blocks use `transform_skip_flag = 1` and the normative
//!   `residual_ts_coding()` (Go-Rice with the transform-skip level remapping),
//!   which is lossless for 8-bit; QP > 4 uses DCT-2 / DST-7 / DCT-8 with
//!   regular `residual_coding()`;
//! * explicit intra MTS (all four DST-7/DCT-8 combinations) and 8x8 LFNST are
//!   selected per luma block by reconstruction SSE; MTS and LFNST are mutually
//!   exclusive as required by the specification;
//!
//! The multi-tool pipeline adds SAO / ALF / LMCS / deblocking / MIP / ISP /
//! BDPCM / dependent quantisation / sign data hiding on top of the strict
//! subset; see `docs/vvc/STATUS.md` for the supported set and
//! `VERIFICATION_DEBT.md` for what still needs external conformance testing.
//!
//! The output is Annex-B with spec 2-byte NAL headers and emulation
//! prevention.
//!
//! [`decode_annexb_to_i420`] is a *subset* decoder for the exact syntax
//! [`VvcEncoder`] emits.  It is not a general H.266 decoder: streams using
//! non-MPM intra modes, MPM indices above DC, transforms outside the implemented
//! set, loop filters, or other partitions are rejected with
//! [`VvcError::Unsupported`].

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::explicit_counter_loop,
    clippy::collapsible_if,
    clippy::if_same_then_else,
    clippy::match_single_binding,
    clippy::should_implement_trait,
    clippy::useless_conversion,
    clippy::unnecessary_map_or,
    clippy::items_after_test_module
)]

/// Bitstream reader, writer, and RBSP/EBSP emulation-prevention helpers.
pub mod bits;
/// H.266 CABAC arithmetic coder, 379-context state tables, and bypass binarizers.
pub mod cabac;
/// Core 1D/2D DCT-2, DST-7, DCT-8 transforms, quantization, and Dependent Quantization.
pub mod dct;
/// In-loop filters: LMCS, Deblocking, SAO, ALF, and CCALF (ITU-T H.266 clause 8.8).
pub mod filter;
/// High-level syntax (OPI, DCI, VPS, SPS, PPS, APS, Picture Header, Slice Header).
pub mod hls;
/// Inter prediction: Merge, MMVD, AMVP, 8-tap/4-tap MC, BCW, BDOF, DMVR, Affine/PROF, CIIP, GPM.
pub mod inter;
/// Low-Frequency Non-Separable Transform (LFNST, ITU-T H.266 clause 8.7.4.2).
pub mod lfnst;
/// Intra prediction: 67 luma modes, 6-MPM, wide-angle, PDPC, MRL, ISP, CCLM, MIP, BDPCM.
pub mod predict;
/// Regular and DepQuant/SDH transform coefficient coding and Joint CbCr flag coding.
pub mod residual;
/// Slice-level CTU/CU/TU encoding and decoding for Intra and Inter P/B slices.
pub mod slice;
mod tr_tables;
/// Transform-skip helpers, Subblock Transform (SBT), Joint CbCr ICT, and QT/MTT split geometry.
pub mod transform;

use crate::I420Frame;
use crate::accelerator::{AcceleratorError, CpuRgbaToI420, RgbaToI420Converter};

pub use hls::VvcSpsInfo;
pub use hls::parse_sps_info;

/// CTU size used by this encoder: 32x32 luma, 16x16 chroma per CU.
const CTU_LOG2: u8 = 5;

/// Largest picture dimension accepted by the encoder and subset decoder.
const MAX_DIMENSION: u32 = 16_384;

/// H.266 level 6.2 `MaxLumaPs`: the largest picture in luma samples.
const MAX_LUMA_SAMPLES: u64 = 35_651_584;

/// Largest NAL unit the Annex-B parser accepts (64 MiB).
const MAX_NAL_BYTES: usize = 64 * 1024 * 1024;

/// Largest Annex-B access unit the parser accepts (256 MiB).
const MAX_ACCESS_UNIT_BYTES: usize = 256 * 1024 * 1024;

/// Largest number of NAL units accepted in one access unit.
const MAX_NALS_PER_ACCESS_UNIT: usize = 65_536;

/// ITU-T H.266 / VVC Table 5 — NAL unit type codes (`nal_unit_type`, 5 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VvcNalType {
    /// Coded slice of a trailing picture (`TRAIL`).
    Trail,
    /// Coded slice of a step-wise temporal sub-layer access picture (`STSA`).
    Stsa,
    /// Coded slice of a random access decodable leading picture (`RADL`).
    Radl,
    /// Coded slice of a random access skipped leading picture (`RASL`).
    Rasl,
    /// Coded slice of an IDR picture with leading pictures (`IDR_W_RADL`).
    IdrWRadl,
    /// Coded slice of an IDR picture without leading pictures (`IDR_N_LP`).
    IdrNLp,
    /// Coded slice of a clean random access picture (`CRA`).
    Cra,
    /// Coded slice of a gradual decoding refresh picture (`GDR`).
    Gdr,
    /// Operating point information (`OPI`).
    Opi,
    /// Decoding capability information (`DCI`).
    Dci,
    /// Video parameter set (`VPS`).
    Vps,
    /// Sequence parameter set (`SPS`).
    Sps,
    /// Picture parameter set (`PPS`).
    Pps,
    /// Adaptation parameter set (`APS`).
    PrefixAps,
    /// Suffix adaptation parameter set (`SUFFIX_APS`).
    SuffixAps,
    /// Picture header (`PH`).
    Ph,
    /// Access unit delimiter (`AUD`).
    Aud,
    /// End of sequence (`EOS`).
    Eos,
    /// Prefix supplemental enhancement information (`PREFIX_SEI`).
    PrefixSei,
    /// Suffix supplemental enhancement information (`SUFFIX_SEI`).
    SuffixSei,
    /// Any other `nal_unit_type` code, kept verbatim.
    Unknown(u8),
}

impl VvcNalType {
    /// Decode a 5-bit `nal_unit_type`; the low 5 bits of `code` are used.
    pub fn from_u8(code: u8) -> Self {
        match code & 0x1F {
            0 => Self::Trail,
            1 => Self::Stsa,
            2 => Self::Radl,
            3 => Self::Rasl,
            7 => Self::IdrWRadl,
            8 => Self::IdrNLp,
            9 => Self::Cra,
            10 => Self::Gdr,
            12 => Self::Opi,
            13 => Self::Dci,
            14 => Self::Vps,
            15 => Self::Sps,
            16 => Self::Pps,
            17 => Self::PrefixAps,
            18 => Self::SuffixAps,
            19 => Self::Ph,
            20 => Self::Aud,
            21 => Self::Eos,
            23 => Self::PrefixSei,
            24 => Self::SuffixSei,
            other => Self::Unknown(other),
        }
    }

    /// Numeric `nal_unit_type` code.
    pub fn code(self) -> u8 {
        match self {
            Self::Trail => 0,
            Self::Stsa => 1,
            Self::Radl => 2,
            Self::Rasl => 3,
            Self::IdrWRadl => 7,
            Self::IdrNLp => 8,
            Self::Cra => 9,
            Self::Gdr => 10,
            Self::Opi => 12,
            Self::Dci => 13,
            Self::Vps => 14,
            Self::Sps => 15,
            Self::Pps => 16,
            Self::PrefixAps => 17,
            Self::SuffixAps => 18,
            Self::Ph => 19,
            Self::Aud => 20,
            Self::Eos => 21,
            Self::PrefixSei => 23,
            Self::SuffixSei => 24,
            Self::Unknown(c) => c & 0x1F,
        }
    }

    /// `true` for the IRAP picture NAL unit types.
    ///
    /// H.266 treats GDR pictures together with IRAP pictures
    /// (`ph_gdr_or_irap_pic_flag`), so `GDR` is included here.
    pub fn is_irap(self) -> bool {
        matches!(self, Self::IdrWRadl | Self::IdrNLp | Self::Cra | Self::Gdr)
    }
}

/// One parsed VVC NAL unit from an Annex-B byte stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcNalUnit {
    /// `nal_unit_type`.
    pub nal_type: VvcNalType,
    /// `nuh_layer_id`.
    pub layer_id: u8,
    /// `nuh_temporal_id_plus1 - 1`, saturated at zero.
    pub temporal_id: u8,
    /// Raw NAL bytes starting with the 2-byte VVC NAL unit header.
    pub raw: Vec<u8>,
    /// Unescaped RBSP payload after the 2-byte VVC NAL unit header.
    pub rbsp: Vec<u8>,
}

/// Errors returned by the VVC encoder and subset decoder.
#[derive(Debug, thiserror::Error)]
pub enum VvcError {
    /// The encoder configuration is outside the supported range.
    #[error("invalid VVC configuration: {0}")]
    InvalidConfig(String),
    /// A frame's dimensions differ from the configured picture size.
    #[error(
        "frame dimensions {width}x{height} do not match encoder {expected_width}x{expected_height}"
    )]
    FrameDimensions {
        /// Width of the frame handed to the encoder.
        width: u32,
        /// Height of the frame handed to the encoder.
        height: u32,
        /// Configured width, in luma samples.
        expected_width: u32,
        /// Configured height, in luma samples.
        expected_height: u32,
    },
    /// A plane does not contain the number of bytes its dimensions require.
    #[error("I420 plane {plane} has {actual} bytes, expected {expected}")]
    PlaneSize {
        /// Plane label (`"Y"`, `"U"` or `"V"`).
        plane: &'static str,
        /// Actual byte length.
        actual: usize,
        /// Required byte length.
        expected: usize,
    },
    /// The requested picture exceeds a VVC or parser limit.
    #[error("VVC limit exceeded: {0}")]
    LimitExceeded(String),
    /// The encoder has already been finished and accepts no more frames.
    #[error("VVC encoder is finished; no more frames are accepted")]
    EncoderFinished,
    /// A bitstream is truncated, inconsistent, or otherwise malformed.
    #[error("invalid VVC bitstream: {0}")]
    Bitstream(String),
    /// The bitstream uses syntax outside the subset this decoder implements.
    #[error("unsupported VVC feature: {0}")]
    Unsupported(&'static str),
    /// An optional accelerator failed while preparing the frame.
    #[error(transparent)]
    Accelerator(#[from] AcceleratorError),
}

/// Encoder configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VvcEncoderConfig {
    /// Visible picture width in luma samples.  Must be even and at least 2.
    pub width: u32,
    /// Visible picture height in luma samples.  Must be even and at least 2.
    pub height: u32,
    /// H.266 `SliceQpY`, in `4..=63`.
    ///
    /// QP 4 is the transform-skip lossless point for 8-bit, so a QP of 4
    /// produces a bit-exact reconstruction.  The chroma QP is derived through
    /// the normative chroma QP mapping table.
    pub qp: u8,
}

/// One Annex-B access unit produced by the encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedAccessUnit {
    /// Annex-B bytes (start codes, NAL headers, emulation prevention).
    pub data: Vec<u8>,
    /// Always `true`: every access unit is an IDR_N_LP keyframe.
    pub keyframe: bool,
}

/// Coded (CTU-padded) picture size for a visible size and `ctu_log2` (`5..=7`).
fn coded_size_for_ctu(
    visible_width: u32,
    visible_height: u32,
    ctu_log2: u8,
) -> Result<(u32, u32), VvcError> {
    if !(5..=7).contains(&ctu_log2) {
        return Err(VvcError::InvalidConfig(format!(
            "ctu_log2 {ctu_log2} is outside 5..=7 (CTU size 32..=128)"
        )));
    }
    let ctu_size = 1u64 << ctu_log2;
    let mask = ctu_size - 1;
    let coded_width = (u64::from(visible_width) + mask) & !mask;
    let coded_height = (u64::from(visible_height) + mask) & !mask;
    let coded_width = u32::try_from(coded_width)
        .map_err(|_| VvcError::LimitExceeded("coded picture width".to_string()))?;
    let coded_height = u32::try_from(coded_height)
        .map_err(|_| VvcError::LimitExceeded("coded picture height".to_string()))?;
    if u64::from(coded_width) * u64::from(coded_height) > MAX_LUMA_SAMPLES {
        return Err(VvcError::LimitExceeded(format!(
            "coded picture {coded_width}x{coded_height} exceeds {MAX_LUMA_SAMPLES} luma samples"
        )));
    }
    Ok((coded_width, coded_height))
}

/// Validate an encoder picture size for a given [`ChromaFormatIdc`].
fn validate_picture_size_for_format(
    width: u32,
    height: u32,
    chroma_format: ChromaFormatIdc,
) -> Result<(), VvcError> {
    let sub_w = chroma_format.sub_width_c() as u32;
    let sub_h = chroma_format.sub_height_c() as u32;
    if width < 2 || height < 2 {
        return Err(VvcError::InvalidConfig(format!(
            "picture needs at least 2x2 luma samples, got {width}x{height}"
        )));
    }
    if !width.is_multiple_of(sub_w) || !height.is_multiple_of(sub_h) {
        return Err(VvcError::InvalidConfig(format!(
            "picture size {width}x{height} is not aligned to chroma subsampling {sub_w}x{sub_h}"
        )));
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(VvcError::LimitExceeded(format!(
            "picture size {width}x{height} exceeds the {MAX_DIMENSION} sample dimension limit"
        )));
    }
    let luma = u64::from(width) * u64::from(height);
    if luma > MAX_LUMA_SAMPLES {
        return Err(VvcError::LimitExceeded(format!(
            "picture size {width}x{height} has {luma} luma samples, more than {MAX_LUMA_SAMPLES}"
        )));
    }
    Ok(())
}

/// Validate the parsed SPS before it is used to allocate and decode a picture.
fn validate_decoded_sps(info: &VvcSpsInfo) -> Result<(), VvcError> {
    if info.chroma_format_idc > 3 {
        return Err(VvcError::Bitstream(format!(
            "invalid sps_chroma_format_idc {}",
            info.chroma_format_idc
        )));
    }
    if !matches!(info.bit_depth_luma, 8 | 10 | 12) {
        return Err(VvcError::Unsupported("bit depth other than 8, 10, or 12"));
    }
    if info.multilayer_enabled {
        return Err(VvcError::Unsupported("multi-layer stream"));
    }
    if !(5..=7).contains(&info.ctu_log2) {
        return Err(VvcError::Unsupported("CTU size outside 32..=128"));
    }
    if info.coded_width > MAX_DIMENSION || info.coded_height > MAX_DIMENSION {
        return Err(VvcError::LimitExceeded(format!(
            "coded picture {}x{} exceeds the {MAX_DIMENSION} sample dimension limit",
            info.coded_width, info.coded_height
        )));
    }
    let coded_luma = u64::from(info.coded_width) * u64::from(info.coded_height);
    if coded_luma > MAX_LUMA_SAMPLES {
        return Err(VvcError::LimitExceeded(format!(
            "coded picture {}x{} has {coded_luma} luma samples, more than {MAX_LUMA_SAMPLES}",
            info.coded_width, info.coded_height
        )));
    }
    let ctu = 1u32 << info.ctu_log2;
    if info.coded_width < ctu
        || info.coded_height < ctu
        || !info.coded_width.is_multiple_of(ctu)
        || !info.coded_height.is_multiple_of(ctu)
    {
        return Err(VvcError::Unsupported(
            "coded picture not aligned to the CTU grid",
        ));
    }
    let cf = ChromaFormatIdc::from_idc(info.chroma_format_idc);
    let sub_w = cf.sub_width_c() as u32;
    let sub_h = cf.sub_height_c() as u32;
    if info.width < 2
        || info.height < 2
        || !info.width.is_multiple_of(sub_w)
        || !info.height.is_multiple_of(sub_h)
        || info.width > info.coded_width
        || info.height > info.coded_height
    {
        return Err(VvcError::Bitstream(format!(
            "visible picture {}x{} is not representable in a coded picture of {}x{}",
            info.width, info.height, info.coded_width, info.coded_height
        )));
    }
    let visible_luma = u64::from(info.width) * u64::from(info.height);
    if visible_luma > MAX_LUMA_SAMPLES {
        return Err(VvcError::LimitExceeded(format!(
            "visible picture {}x{} has {visible_luma} luma samples, more than {MAX_LUMA_SAMPLES}",
            info.width, info.height
        )));
    }
    Ok(())
}

/// A streaming H.266 / VVC intra-only software encoder.
///
/// Every access unit is an `IDR_N_LP` keyframe: H.266 inter prediction is not
/// implemented, so every picture is self-contained and parameter sets are
/// repeated in-band.  [`VvcEncoder::finish`] only marks the encoder as drained
/// and never returns delayed output.
#[derive(Debug)]
pub struct VvcEncoder {
    config: VvcEncoderConfig,
    bit_depth: u8,
    chroma_format: ChromaFormatIdc,
    frames_sent: u64,
    finished: bool,
    cfg: hls::VvcStreamConfig,
    visible_width: u32,
    visible_height: u32,
    aud_rbsp: Vec<u8>,
    vps_rbsp: Vec<u8>,
    sps_rbsp: Vec<u8>,
    pps_rbsp: Vec<u8>,
}

impl VvcEncoder {
    /// Create an 8-bit 4:2:0 encoder for the given picture size and QP.
    ///
    /// Fails with [`VvcError::InvalidConfig`] for zero or odd dimensions and
    /// for a QP outside `4..=63`, and with [`VvcError::LimitExceeded`] when the
    /// picture exceeds the VVC level 6.2 limits.
    pub fn new(config: VvcEncoderConfig) -> Result<Self, VvcError> {
        Self::new_with_format(config, 8, ChromaFormatIdc::Chroma420, CTU_LOG2)
    }

    /// Create an encoder for `(config, bit_depth, chroma_format, ctu_log2)` where
    /// `bit_depth` is in `{8, 10, 12}`, `chroma_format` is in `4:0:0..=4:4:4`, and
    /// `ctu_log2` is in `5..=7` (`CTU` size `32..=128`).
    ///
    /// `CTU` size 16 cannot be represented: `sps_log2_ctu_size_minus5` is a
    /// non-negative value, so the smallest signallable CTU is 32 samples wide
    /// (ITU-T Rec. H.266 §7.3.2.4). Passing `4` fails with
    /// [`VvcError::InvalidConfig`].
    pub fn new_with_format(
        config: VvcEncoderConfig,
        bit_depth: u8,
        chroma_format: ChromaFormatIdc,
        ctu_log2: u8,
    ) -> Result<Self, VvcError> {
        if !(4..=63).contains(&config.qp) {
            return Err(VvcError::InvalidConfig(format!(
                "qp {} is outside the SliceQpY range 4..=63",
                config.qp
            )));
        }
        if !matches!(bit_depth, 8 | 10 | 12) {
            return Err(VvcError::InvalidConfig(format!(
                "bit_depth {bit_depth} must be 8, 10, or 12"
            )));
        }
        if ctu_log2 == 4 {
            return Err(VvcError::InvalidConfig(
                "ctu_log2 4 (CTU size 16) cannot be signalled: \
                 sps_log2_ctu_size_minus5 has no value below CTU size 32"
                    .to_string(),
            ));
        }
        if !(5..=7).contains(&ctu_log2) {
            return Err(VvcError::InvalidConfig(format!(
                "ctu_log2 {ctu_log2} is outside 5..=7 (CTU size 32..=128)"
            )));
        }
        validate_picture_size_for_format(config.width, config.height, chroma_format)?;

        let (coded_width, coded_height) =
            coded_size_for_ctu(config.width, config.height, ctu_log2)?;
        let min_cb_log2 = ctu_log2.min(5);
        let cfg = hls::VvcStreamConfig {
            width: coded_width,
            height: coded_height,
            crop_right: coded_width - config.width,
            crop_bottom: coded_height - config.height,
            qp: config.qp,
            level_idc: hls::select_level_idc(coded_width, coded_height),
            ctu_log2,
            min_cb_log2,
            ts_log2: min_cb_log2,
        };
        let is_default_8bit_420 =
            bit_depth == 8 && chroma_format == ChromaFormatIdc::Chroma420 && ctu_log2 == CTU_LOG2;
        let sps_rbsp = if is_default_8bit_420 {
            hls::build_sps_rbsp(&cfg)
        } else {
            let tools = hls::VvcHlsTools {
                transform_skip: true,
                ..hls::VvcHlsTools::default()
            };
            hls::build_sps_rbsp_format_bd(
                &cfg,
                &tools,
                chroma_format as u8,
                bit_depth,
                false,
                false,
                false,
                false,
            )
        };
        Ok(Self {
            config,
            bit_depth,
            chroma_format,
            frames_sent: 0,
            finished: false,
            visible_width: config.width,
            visible_height: config.height,
            aud_rbsp: hls::build_aud_rbsp(),
            vps_rbsp: hls::build_vps_rbsp(&cfg),
            sps_rbsp,
            pps_rbsp: hls::build_pps_rbsp(&cfg),
            cfg,
        })
    }

    /// Configured sample bit depth (`8`, `10`, or `12`).
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    /// Configured chroma format (`4:0:0`..=`4:4:4`).
    pub fn chroma_format(&self) -> ChromaFormatIdc {
        self.chroma_format
    }

    /// The configuration this encoder was created with.
    pub fn config(&self) -> &VvcEncoderConfig {
        &self.config
    }

    /// Number of frames accepted so far.
    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// Coded (CTU-padded) picture size.
    ///
    /// This is the size the conformance window starts from; it is a multiple of
    /// 32 in both dimensions.
    pub fn coded_size(&self) -> (u32, u32) {
        (self.cfg.width, self.cfg.height)
    }

    /// Visible (conformance-window-cropped) picture size.
    ///
    /// This equals the configured size and the size of every frame passed to
    /// [`VvcEncoder::encode_i420`].
    pub fn visible_size(&self) -> (u32, u32) {
        (self.visible_width, self.visible_height)
    }

    /// Encode one I420 frame into an Annex-B access unit.
    ///
    /// The frame must match the configured visible size and carry exactly
    /// `width * height`, `ceil(width/2) * ceil(height/2)` and
    /// `ceil(width/2) * ceil(height/2)` bytes in its Y, U and V planes.
    pub fn encode_i420(&mut self, frame: &I420Frame) -> Result<EncodedAccessUnit, VvcError> {
        self.ensure_active()?;
        if frame.width != self.config.width || frame.height != self.config.height {
            return Err(VvcError::FrameDimensions {
                width: frame.width,
                height: frame.height,
                expected_width: self.config.width,
                expected_height: self.config.height,
            });
        }
        validate_frame_planes(frame)?;

        let padded = pad_i420(frame, self.cfg.width, self.cfg.height);
        let cabac = slice::encode_picture(&padded, self.cfg.qp, self.cfg.ctu_log2)?;

        let mut au = Vec::with_capacity(cabac.len() + 256);
        hls::push_nal(&mut au, hls::NAL_AUD, 0, 1, &self.aud_rbsp);
        // In-band parameter sets on every IDR so each access unit is
        // self-contained for Annex-B consumers.
        hls::push_nal(&mut au, hls::NAL_VPS, 0, 1, &self.vps_rbsp);
        hls::push_nal(&mut au, hls::NAL_SPS, 0, 1, &self.sps_rbsp);
        hls::push_nal(&mut au, hls::NAL_PPS, 0, 1, &self.pps_rbsp);
        // Each picture is an IDR, so its POC LSB restarts at 0.
        hls::push_nal(&mut au, hls::NAL_PH, 0, 1, &hls::build_ph_rbsp(0));

        let mut slice_rbsp = hls::build_slice_header_bits(0);
        slice_rbsp.extend_from_slice(&cabac);
        hls::push_nal(&mut au, hls::NAL_IDR_N_LP, 0, 1, &slice_rbsp);

        self.frames_sent += 1;
        Ok(EncodedAccessUnit {
            data: au,
            keyframe: true,
        })
    }

    /// Encode one [`VvcFrame`] (`bit_depth` in `8..=12`, `chroma_format` in `4:0:0`..=`4:4:4`)
    /// into an Annex-B access unit.
    pub fn encode_vvc_frame(&mut self, frame: &VvcFrame) -> Result<EncodedAccessUnit, VvcError> {
        self.ensure_active()?;
        if frame.width != self.config.width || frame.height != self.config.height {
            return Err(VvcError::FrameDimensions {
                width: frame.width,
                height: frame.height,
                expected_width: self.config.width,
                expected_height: self.config.height,
            });
        }
        validate_vvc_frame_planes(frame)?;
        if self.bit_depth == 8
            && self.chroma_format == ChromaFormatIdc::Chroma420
            && self.cfg.ctu_log2 == CTU_LOG2
            && frame.bit_depth == 8
            && frame.chroma_format == ChromaFormatIdc::Chroma420
        {
            return self.encode_i420(&frame.to_i420());
        }
        let padded = pad_vvc_frame(frame, self.cfg.width, self.cfg.height);
        let (cabac, _recon) = slice::encode_vvc_frame_picture(
            &padded,
            self.cfg.qp,
            self.cfg.ctu_log2,
            &SliceCodingOptions::default(),
        )?;
        let tools = hls::VvcHlsTools {
            transform_skip: true,
            ..hls::VvcHlsTools::default()
        };
        let mut au = Vec::with_capacity(cabac.len() + 256);
        hls::push_nal(&mut au, hls::NAL_AUD, 0, 1, &self.aud_rbsp);
        hls::push_nal(&mut au, hls::NAL_VPS, 0, 1, &self.vps_rbsp);
        hls::push_nal(&mut au, hls::NAL_SPS, 0, 1, &self.sps_rbsp);
        hls::push_nal(&mut au, hls::NAL_PPS, 0, 1, &self.pps_rbsp);
        hls::push_nal(
            &mut au,
            hls::NAL_PH,
            0,
            1,
            &hls::build_ph_rbsp_with_tools(0, &tools),
        );
        let mut slice_rbsp = hls::build_slice_header_bits_with_tools(0, &tools, false, false);
        slice_rbsp.extend_from_slice(&cabac);
        hls::push_nal(&mut au, hls::NAL_IDR_N_LP, 0, 1, &slice_rbsp);

        self.frames_sent += 1;
        Ok(EncodedAccessUnit {
            data: au,
            keyframe: true,
        })
    }

    /// Convert one RGBA8 frame to 4:2:0 and encode it.
    ///
    /// `rgba` must contain exactly `width * height * 4` bytes and is interpreted
    /// as premultiplied RGBA8 in BT.709 limited range; the alpha channel is not
    /// coded and video frames are expected to be opaque (alpha 255).
    ///
    /// # Errors
    ///
    /// Returns [`VvcError::InvalidConfig`] when the buffer length is wrong and
    /// the same errors as [`VvcEncoder::encode_i420`] otherwise.
    pub fn encode_rgba(&mut self, rgba: &[u8]) -> Result<EncodedAccessUnit, VvcError> {
        self.encode_rgba_with(rgba, &CpuRgbaToI420)
    }

    /// Encodes one RGBA8 frame through a caller-provided converter.
    ///
    /// The converter is the seam used by the heterogeneous scheduler: a Vulkan
    /// Compute implementation and [`CpuRgbaToI420`] produce the same I420 frame,
    /// so the resulting access unit is bit-identical. The encoder itself never
    /// depends on the Vulkan crate.
    ///
    /// # Errors
    ///
    /// Returns [`VvcError::InvalidConfig`] when the RGBA length does not match
    /// the configured picture size, and [`VvcError::Accelerator`] when the
    /// converter fails.
    pub fn encode_rgba_with(
        &mut self,
        rgba: &[u8],
        converter: &dyn RgbaToI420Converter,
    ) -> Result<EncodedAccessUnit, VvcError> {
        self.ensure_active()?;
        let width = self.config.width;
        let height = self.config.height;
        let expected = u64::from(width) * u64::from(height) * 4;
        let expected = usize::try_from(expected)
            .map_err(|_| VvcError::LimitExceeded(format!("RGBA buffer for {width}x{height}")))?;
        if rgba.len() != expected {
            return Err(VvcError::InvalidConfig(format!(
                "RGBA input has {} bytes, expected {expected} for {width}x{height}",
                rgba.len()
            )));
        }

        let source = converter.rgba_to_i420(rgba, width, height)?;
        if source.width != width || source.height != height || !source.is_valid() {
            return Err(VvcError::Accelerator(AcceleratorError::InvalidInput(
                "converter returned a frame with unexpected dimensions or planes".to_owned(),
            )));
        }
        self.encode_i420(&source)
    }

    /// Finish the stream.
    ///
    /// All frames are emitted immediately, so no delayed pictures exist and
    /// this produces no output.  It is idempotent; calls to
    /// [`VvcEncoder::encode_i420`] and [`VvcEncoder::encode_rgba`] after
    /// `finish` fail with [`VvcError::EncoderFinished`].
    pub fn finish(&mut self) {
        self.finished = true;
    }

    /// `true` once [`VvcEncoder::finish`] has been called.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    fn ensure_active(&self) -> Result<(), VvcError> {
        if self.finished {
            return Err(VvcError::EncoderFinished);
        }
        Ok(())
    }
}

/// Validate the three plane lengths of an I420 frame.
fn validate_frame_planes(frame: &I420Frame) -> Result<(), VvcError> {
    let luma = usize::try_from(u64::from(frame.width) * u64::from(frame.height))
        .map_err(|_| VvcError::LimitExceeded("luma plane size".to_string()))?;
    let (chroma_width, chroma_height) = frame.chroma_size();
    let chroma = usize::try_from(u64::from(chroma_width) * u64::from(chroma_height))
        .map_err(|_| VvcError::LimitExceeded("chroma plane size".to_string()))?;

    if !frame.is_valid() {
        for (plane, actual, expected) in [
            ("Y", frame.y.len(), luma),
            ("U", frame.u.len(), chroma),
            ("V", frame.v.len(), chroma),
        ] {
            if actual != expected {
                return Err(VvcError::PlaneSize {
                    plane,
                    actual,
                    expected,
                });
            }
        }
        return Err(VvcError::PlaneSize {
            plane: "Y",
            actual: frame.y.len(),
            expected: luma,
        });
    }
    Ok(())
}

/// Parse an ITU-T H.266 / VVC Annex-B byte stream into individual NAL units.
///
/// A stream larger than 256 MiB, a NAL unit larger than 64 MiB, or more than
/// 65 536 NAL units is rejected with [`VvcError::LimitExceeded`].  NAL units
/// shorter than the 2-byte VVC header are skipped.  Empty input yields an empty
/// vector; non-empty input that yields no NAL unit at all (no start code, or
/// only headers too short to be a NAL) is rejected with [`VvcError::Bitstream`].
pub fn parse_annexb_nals(data: &[u8]) -> Result<Vec<VvcNalUnit>, VvcError> {
    if data.len() > MAX_ACCESS_UNIT_BYTES {
        return Err(VvcError::LimitExceeded(format!(
            "access unit of {} bytes exceeds the {MAX_ACCESS_UNIT_BYTES}-byte parser limit",
            data.len()
        )));
    }

    let mut out = Vec::new();
    let mut i = 0usize;
    let mut nal_start: Option<usize> = None;
    while i < data.len() {
        let sc_len = if i + 4 <= data.len() && data[i..i + 4] == [0, 0, 0, 1] {
            4
        } else if i + 3 <= data.len() && data[i..i + 3] == [0, 0, 1] {
            3
        } else {
            0
        };
        if sc_len > 0 {
            if let Some(start) = nal_start
                && start < i
            {
                push_nal_unit(&mut out, &data[start..i])?;
            }
            i += sc_len;
            nal_start = Some(i);
        } else {
            i += 1;
        }
    }
    if let Some(start) = nal_start
        && start < data.len()
    {
        push_nal_unit(&mut out, &data[start..])?;
    }
    if out.is_empty() && !data.is_empty() {
        return Err(VvcError::Bitstream(
            "Annex-B input contains no start code or NAL unit".to_string(),
        ));
    }
    Ok(out)
}

/// Convert one raw Annex-B NAL unit and append it.
fn push_nal_unit(out: &mut Vec<VvcNalUnit>, raw: &[u8]) -> Result<(), VvcError> {
    if raw.len() < 2 {
        return Ok(());
    }
    if raw.len() > MAX_NAL_BYTES {
        return Err(VvcError::LimitExceeded(format!(
            "NAL unit of {} bytes exceeds the {MAX_NAL_BYTES}-byte limit",
            raw.len()
        )));
    }
    if out.len() >= MAX_NALS_PER_ACCESS_UNIT {
        return Err(VvcError::LimitExceeded(format!(
            "access unit has more than {MAX_NALS_PER_ACCESS_UNIT} NAL units"
        )));
    }
    let layer_id = raw[0] & 0x3F;
    let nal_type = VvcNalType::from_u8((raw[1] >> 3) & 0x1F);
    let temporal_id = (raw[1] & 0x07).saturating_sub(1);
    let rbsp = bits::ebsp_to_rbsp(&raw[2..]);
    out.push(VvcNalUnit {
        nal_type,
        layer_id,
        temporal_id,
        raw: raw.to_vec(),
        rbsp,
    });
    Ok(())
}

/// Replicate the right/bottom edges until the picture reaches the coded size.
fn pad_i420(frame: &I420Frame, width: u32, height: u32) -> I420Frame {
    let pad_plane = |src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32| -> Vec<u8> {
        let (sw, sh, dw, dh) = (sw as usize, sh as usize, dw as usize, dh as usize);
        let mut out = vec![0u8; dw * dh];
        for y in 0..dh {
            let sy = y.min(sh - 1);
            for x in 0..dw {
                let sx = x.min(sw - 1);
                out[y * dw + x] = src[sy * sw + sx];
            }
        }
        out
    };
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let scw = frame.width.div_ceil(2);
    let sch = frame.height.div_ceil(2);
    I420Frame {
        width,
        height,
        y: pad_plane(&frame.y, frame.width, frame.height, width, height),
        u: pad_plane(&frame.u, scw, sch, cw, ch),
        v: pad_plane(&frame.v, scw, sch, cw, ch),
    }
}

/// Map a checked header read onto a malformed-bitstream error.
fn header_read<T>(value: Result<T, &'static str>, context: &str) -> Result<T, VvcError> {
    value.map_err(|_| VvcError::Bitstream(format!("truncated {context}")))
}

/// Identity and initial QP carried by this encoder's PPS layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PpsInfo {
    /// `pps_pic_parameter_set_id` (6-bit).
    pps_id: u8,
    /// `pps_seq_parameter_set_id` (4-bit).
    sps_id: u8,
    /// `pps_init_qp_minus26`.
    init_qp_minus26: i32,
}

/// Parse this encoder's PPS layout through `pps_init_qp_minus26`.
///
/// The leading identifiers are retained so the decoder can link the picture
/// header to its PPS and the PPS to its SPS; the remaining fields are walked
/// only to reach `pps_init_qp_minus26`.
fn parse_pps_info(rbsp: &[u8]) -> Result<PpsInfo, VvcError> {
    let mut br = bits::BitReader::new(rbsp);
    let pps_id = header_read(br.read_bits(6), "PPS: pps_pic_parameter_set_id")? as u8;
    let sps_id = header_read(br.read_bits(4), "PPS: pps_seq_parameter_set_id")? as u8;
    header_read(br.read_bit(), "PPS: mixed_nalu_types flag")?;
    header_read(br.read_ue(), "PPS: picture width")?;
    header_read(br.read_ue(), "PPS: picture height")?;
    if header_read(br.read_bit(), "PPS: conformance window flag")? {
        // pps_conformance_window_flag
        for _ in 0..4 {
            header_read(br.read_ue(), "PPS: conformance window")?;
        }
    }
    header_read(br.read_bit(), "PPS: scaling window flag")?;
    header_read(br.read_bit(), "PPS: output flag")?;
    let no_pic_partition = header_read(br.read_bit(), "PPS: no_pic_partition flag")?;
    if header_read(br.read_bit(), "PPS: subpic id mapping")? {
        let num_subpics = if !no_pic_partition {
            header_read(br.read_ue(), "PPS: pps_num_subpics_minus1")?.saturating_add(1)
        } else {
            1
        };
        let id_len =
            header_read(br.read_ue(), "PPS: pps_subpic_id_len_minus1")?.saturating_add(1) as usize;
        if id_len == 0 || id_len > 16 || num_subpics > 1024 {
            return Err(VvcError::Bitstream(
                "PPS: invalid sub-picture ID mapping".to_string(),
            ));
        }
        for _ in 0..num_subpics {
            header_read(br.read_bits(id_len as u32), "PPS: pps_subpic_id")?;
        }
    }
    header_read(br.read_bit(), "PPS: cabac_init flag")?;
    header_read(br.read_ue(), "PPS: ref_idx_default_active[0]")?;
    header_read(br.read_ue(), "PPS: ref_idx_default_active[1]")?;
    header_read(br.read_bit(), "PPS: rpl1_idx flag")?;
    header_read(br.read_bit(), "PPS: weighted_pred flag")?;
    header_read(br.read_bit(), "PPS: weighted_bipred flag")?;
    header_read(br.read_bit(), "PPS: ref_wraparound flag")?;
    let init_qp_minus26 = header_read(br.read_se(), "PPS: pps_init_qp_minus26")?;
    Ok(PpsInfo {
        pps_id,
        sps_id,
        init_qp_minus26,
    })
}

/// Identity and ordering information carried by this encoder's PH layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PhInfo {
    /// `ph_pic_parameter_set_id` (`ue(v)`).
    pps_id: u8,
    /// `ph_pic_order_cnt_lsb` (8-bit).
    poc_lsb: u32,
}

/// Parse the picture header RBSP this encoder writes.
///
/// The header is deliberately minimal: an IRAP, reference, non-GDR picture whose
/// slices are all intra.  Anything else is outside the subset this decoder
/// implements, while a POC that does not restart at zero is malformed for an
/// all-IDR stream.
fn parse_ph_info(rbsp: &[u8]) -> Result<PhInfo, VvcError> {
    let mut br = bits::BitReader::new(rbsp);
    if !header_read(br.read_bit(), "PH: ph_gdr_or_irap_pic_flag")? {
        return Err(VvcError::Unsupported(
            "picture header: not a GDR or IRAP picture",
        ));
    }
    if header_read(br.read_bit(), "PH: ph_non_ref_pic_flag")? {
        return Err(VvcError::Unsupported(
            "picture header: non-reference picture",
        ));
    }
    if header_read(br.read_bit(), "PH: ph_gdr_pic_flag")? {
        return Err(VvcError::Unsupported(
            "picture header: gradual decoding refresh picture",
        ));
    }
    if header_read(br.read_bit(), "PH: ph_inter_slice_allowed_flag")? {
        return Err(VvcError::Unsupported(
            "picture header: inter slices allowed",
        ));
    }
    let pps_id = header_read(br.read_ue(), "PH: ph_pic_parameter_set_id")?;
    let pps_id = u8::try_from(pps_id)
        .map_err(|_| VvcError::Bitstream(format!("PH: PPS id {pps_id} does not fit in 6 bits")))?;
    let poc_lsb = header_read(br.read_bits(8), "PH: ph_pic_order_cnt_lsb")?;
    if poc_lsb != 0 {
        return Err(VvcError::Bitstream(format!(
            "PH: ph_pic_order_cnt_lsb is {poc_lsb}, but every picture is an IDR and POC restarts at zero"
        )));
    }
    Ok(PhInfo { pps_id, poc_lsb })
}

/// `true` for coded-slice NAL unit types (H.266 Table 5, codes `0..=11`).
fn is_vcl_nal(nal_type: VvcNalType) -> bool {
    nal_type.code() <= 11
}

/// Parse the slice header of this encoder's slices and return
/// `(sh_qp_delta, offset_of_slice_data_in_bytes)`.
fn parse_our_slice_header(
    rbsp: &[u8],
    options: &SliceCodingOptions,
    inter: bool,
) -> Result<(i32, usize), VvcError> {
    let mut br = bits::BitReader::new(rbsp);
    let ph_in_sh = header_read(br.read_bit(), "slice header")?;
    if ph_in_sh {
        let gdr_or_irap = header_read(br.read_bit(), "PH-in-SH: gdr_or_irap")?;
        let _non_ref = header_read(br.read_bit(), "PH-in-SH: non_ref")?;
        if gdr_or_irap {
            let _gdr = header_read(br.read_bit(), "PH-in-SH: gdr")?;
        }
        let inter_allowed = header_read(br.read_bit(), "PH-in-SH: inter_slice_allowed")?;
        if inter_allowed {
            let _intra_allowed = header_read(br.read_bit(), "PH-in-SH: intra_slice_allowed")?;
        }
        let _pps_id = header_read(br.read_ue(), "PH-in-SH: pps_id")?;
        let _poc_lsb = header_read(br.read_bits(8), "PH-in-SH: poc_lsb")?;
        if !gdr_or_irap || inter_allowed {
            let _ = header_read(br.read_bit(), "PH-in-SH: mvd_l1_zero")?;
        }
        let _ = header_read(br.read_bit(), "PH-in-SH: deblocking_Disabled_or_partition")?;
    }
    if inter {
        let slice_type = header_read(br.read_ue(), "slice sh_slice_type")?;
        if slice_type > 2 {
            return Err(VvcError::Bitstream(
                "slice: invalid sh_slice_type".to_string(),
            ));
        }
        if options.enable_alf {
            let alf = header_read(br.read_bit(), "slice sh_alf_enabled_flag")?;
            if alf {
                let num_aps = header_read(br.read_bits(3), "slice sh_num_alf_aps_ids_luma")?;
                for _ in 0..num_aps {
                    let _ = header_read(br.read_bits(3), "slice sh_alf_aps_id_luma")?;
                }
                let cb_alf = header_read(br.read_bit(), "slice sh_alf_cb_enabled_flag")?;
                let cr_alf = header_read(br.read_bit(), "slice sh_alf_cr_enabled_flag")?;
                if cb_alf || cr_alf {
                    let _ = header_read(br.read_bits(3), "slice sh_alf_aps_id_chroma")?;
                }
            }
        }
        if slice_type != 2 {
            let num_l0 = header_read(br.read_ue(), "slice num_ref_entries[0]")?;
            if num_l0 > 16 {
                return Err(VvcError::Bitstream(
                    "slice: L0 entry count exceeds 16".to_string(),
                ));
            }
            for _ in 0..num_l0 {
                let _abs_delta = header_read(br.read_ue(), "slice abs_delta_poc_st")?;
                let _sign = header_read(br.read_bit(), "slice strp_entry_sign_flag")?;
            }
            let num_l1 = header_read(br.read_ue(), "slice num_ref_entries[1]")?;
            if num_l1 > 16 {
                return Err(VvcError::Bitstream(
                    "slice: L1 entry count exceeds 16".to_string(),
                ));
            }
            for _ in 0..num_l1 {
                let _abs_delta = header_read(br.read_ue(), "slice abs_delta_poc_st L1")?;
                let _sign = header_read(br.read_bit(), "slice strp_entry_sign_flag L1")?;
            }
        }
    } else {
        let _no_output = header_read(br.read_bit(), "slice output flag")?;
        if options.enable_alf {
            let alf = header_read(br.read_bit(), "slice sh_alf_enabled_flag")?;
            if alf {
                let num_aps = header_read(br.read_bits(3), "slice sh_num_alf_aps_ids_luma")?;
                for _ in 0..num_aps {
                    let _ = header_read(br.read_bits(3), "slice sh_alf_aps_id_luma")?;
                }
                let cb_alf = header_read(br.read_bit(), "slice sh_alf_cb_enabled_flag")?;
                let cr_alf = header_read(br.read_bit(), "slice sh_alf_cr_enabled_flag")?;
                if cb_alf || cr_alf {
                    let _ = header_read(br.read_bits(3), "slice sh_alf_aps_id_chroma")?;
                }
            }
        }
    }
    let qp_delta = header_read(br.read_se(), "slice sh_qp_delta")?;
    if options.enable_sao {
        let _sao_luma = header_read(br.read_bit(), "slice sh_sao_luma_used_flag")?;
        let _sao_chroma = header_read(br.read_bit(), "slice sh_sao_chroma_used_flag")?;
    }
    let mut dep_quant = false;
    if options.dep_quant {
        dep_quant = header_read(br.read_bit(), "slice sh_dep_quant_used_flag")?;
    }
    let mut sdh = false;
    if options.sign_data_hiding && !dep_quant {
        sdh = header_read(br.read_bit(), "slice sh_sign_data_hiding_used_flag")?;
    }
    if !dep_quant && !sdh {
        let _ts_disabled = header_read(br.read_bit(), "slice transform-skip flag")?;
    }
    br.byte_align();
    let offset = br.bit_pos() / 8;
    Ok((qp_delta, offset))
}

/// Decode Annex-B access units produced by [`VvcEncoder`] back into
/// `I420Frame`s.
///
/// The accepted input is exactly the bitstream this crate's encoder emits: one
/// `IDR_N_LP` picture per access unit, in-band parameter sets repeated in every
/// access unit, a 32x32 CTU grid with one intra PLANAR or DC coding unit per
/// CTU, and either transform-skip or DCT-2 residual coding.
///
/// This is **not** a general H.266 decoder.  In particular:
///
/// * foreign conformant streams are not supported: any other CTU size,
///   partition, prediction mode, transform, or loop filter is rejected with
///   [`VvcError::Unsupported`];
/// * parameter-set linkage is validated: SPS and PPS are stored by their
///   signalled IDs, a repeated ID must carry identical content, and a picture
///   header must reference a known PPS whose SPS is present;
/// * picture headers are validated: the PH must describe an IRAP reference
///   picture (not GDR) with `ph_pic_order_cnt_lsb == 0` and no inter slices;
/// * each access unit must contain at most one slice, and only `IDR_N_LP`
///   slices on layer 0 at temporal sub-layer 0 are accepted;
/// * access units are not reordered; frames are returned in the order their
///   slices appear.
pub fn decode_annexb_to_i420(access_units: &[&[u8]]) -> Result<Vec<I420Frame>, VvcError> {
    let mut frames = Vec::new();
    let mut sps_info: [Option<VvcSpsInfo>; 16] = std::array::from_fn(|_| None);
    let mut pps_info: [Option<PpsInfo>; 64] = [None; 64];
    let mut active_ph: Option<PhInfo> = None;
    let mut vcl_seen = false;

    for au in access_units {
        let nals = parse_annexb_nals(au)?;
        for nal in &nals {
            match nal.nal_type {
                VvcNalType::Aud => {
                    // An AUD starts a new access unit.
                    active_ph = None;
                    vcl_seen = false;
                }
                VvcNalType::Sps => {
                    let info = parse_sps_info(&nal.rbsp)?;
                    validate_decoded_sps(&info)?;
                    let index = usize::from(info.sps_id);
                    if let Some(existing) = &sps_info[index] {
                        if existing != &info {
                            return Err(VvcError::Bitstream(format!(
                                "SPS {} is repeated with different content",
                                info.sps_id
                            )));
                        }
                    } else {
                        sps_info[index] = Some(info);
                    }
                }
                VvcNalType::Pps => {
                    let info = parse_pps_info(&nal.rbsp)?;
                    if sps_info[usize::from(info.sps_id)].is_none() {
                        return Err(VvcError::Bitstream(format!(
                            "PPS {} references unknown SPS {}",
                            info.pps_id, info.sps_id
                        )));
                    }
                    let index = usize::from(info.pps_id);
                    if let Some(existing) = &pps_info[index] {
                        if existing != &info {
                            return Err(VvcError::Bitstream(format!(
                                "PPS {} is repeated with different content",
                                info.pps_id
                            )));
                        }
                    } else {
                        pps_info[index] = Some(info);
                    }
                }
                VvcNalType::Ph => {
                    let ph = parse_ph_info(&nal.rbsp)?;
                    let pps = pps_info[usize::from(ph.pps_id)].ok_or_else(|| {
                        VvcError::Bitstream(format!("PH references unknown PPS {}", ph.pps_id))
                    })?;
                    if sps_info[usize::from(pps.sps_id)].is_none() {
                        return Err(VvcError::Bitstream(format!(
                            "PPS {} references unknown SPS {}",
                            pps.pps_id, pps.sps_id
                        )));
                    }
                    active_ph = Some(ph);
                    vcl_seen = false;
                }
                nal_type if is_vcl_nal(nal_type) => {
                    let ph = active_ph.ok_or_else(|| {
                        VvcError::Bitstream(
                            "VCL NAL unit without a preceding picture header".to_string(),
                        )
                    })?;
                    if nal.nal_type != VvcNalType::IdrNLp {
                        return Err(VvcError::Unsupported(
                            "VCL NAL unit type other than IDR_N_LP",
                        ));
                    }
                    if nal.layer_id != 0 {
                        return Err(VvcError::Unsupported(
                            "VCL NAL unit with a non-zero layer ID",
                        ));
                    }
                    // `VvcNalUnit::temporal_id` stores
                    // `nuh_temporal_id_plus1 - 1`, so temporal sub-layer 0 is
                    // the raw value 1; the illegal raw value 0 is rejected too.
                    if nal.raw[1] & 0x07 != 1 {
                        return Err(VvcError::Unsupported(
                            "VCL NAL unit above temporal sub-layer 0",
                        ));
                    }
                    if vcl_seen {
                        return Err(VvcError::Unsupported("multiple slices per access unit"));
                    }
                    vcl_seen = true;

                    let pps = pps_info[usize::from(ph.pps_id)].ok_or_else(|| {
                        VvcError::Bitstream(format!("PH references unknown PPS {}", ph.pps_id))
                    })?;
                    let info = sps_info[usize::from(pps.sps_id)].as_ref().ok_or_else(|| {
                        VvcError::Bitstream(format!(
                            "PPS {} references unknown SPS {}",
                            pps.pps_id, pps.sps_id
                        ))
                    })?;
                    let (qp_delta, data_off) =
                        parse_our_slice_header(&nal.rbsp, &SliceCodingOptions::default(), false)?;
                    let slice_qp = 26i32
                        .saturating_add(pps.init_qp_minus26)
                        .saturating_add(qp_delta)
                        .clamp(0, 63) as u8;
                    let cabac = nal.rbsp.get(data_off..).ok_or_else(|| {
                        VvcError::Bitstream("slice data offset out of range".to_string())
                    })?;
                    let frame = slice::decode_picture(
                        cabac,
                        slice_qp,
                        info.ctu_log2,
                        info.coded_width,
                        info.coded_height,
                    )?;
                    frames.push(crop_frame(&frame, info.width, info.height));
                }
                _ => {}
            }
        }
        // The active picture header is scoped to one access unit.
        active_ph = None;
        vcl_seen = false;
    }
    Ok(frames)
}

/// Crop a coded picture down to the conformance-window visible size.
fn crop_frame(frame: &I420Frame, width: u32, height: u32) -> I420Frame {
    if frame.width == width && frame.height == height {
        return frame.clone();
    }
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let crop_plane = |src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32| -> Vec<u8> {
        let (sw, sh, dw, dh) = (sw as usize, sh as usize, dw as usize, dh as usize);
        let mut out = vec![0u8; dw * dh];
        for y in 0..dh.min(sh) {
            for x in 0..dw.min(sw) {
                out[y * dw + x] = src[y * sw + x];
            }
        }
        out
    };
    let scw = frame.width.div_ceil(2);
    let sch = frame.height.div_ceil(2);
    I420Frame {
        width,
        height,
        y: crop_plane(&frame.y, frame.width, frame.height, width, height),
        u: crop_plane(&frame.u, scw, sch, cw, ch),
        v: crop_plane(&frame.v, scw, sch, cw, ch),
    }
}

// ===========================================================================
// Wave 7: Multi-Tool VVC Encoder Options & Stateful Annex-B I/P/B Decoder
// ===========================================================================

pub use filter::{
    ALF_CLASS_TO_FILTER_MAPPING, ALF_FIXED_FILTER_COEFF, DBF_BETA_TABLE, DBF_TC_TABLE,
    LmcsReshaper, SaoBlkParams, SaoType, apply_alf_chroma_plane, apply_alf_luma_plane,
    apply_ccalf_plane_420, apply_sao_plane, apply_sao_region, chroma_filter_length,
    deblock_chroma_edge, deblock_chroma_edge_region, deblock_chroma_plane, deblock_chroma_sample,
    deblock_luma_edge_4, deblock_luma_edge_normative, deblock_luma_long_tap_line,
    deblock_luma_plane, derive_alf_class_4x4, derive_boundary_strength, luma_filter_length,
};
pub use hls::{
    VvcAlfAps, VvcAps, VvcApsType, VvcChromaQpTable, VvcDci, VvcLmcsAps, VvcOpi, VvcPictureHeader,
    VvcPps, VvcRefPicEntry, VvcRefPicListStruct, VvcScalingListEntry, VvcSliceHeader, VvcSliceType,
    VvcSps, VvcVps,
};
pub use inter::{
    AFFINE_LUMA_FILTER_8TAP, AmvrPrecision, DecodedPictureBuffer, HmvpTable, InterMergeCand,
    LUMA_ALT_HPEL_FILTER_8TAP, LUMA_FILTER_8TAP, MotionFieldGrid, Mv, MvField, RefFrame,
    apply_bdof_luma, apply_ciip_blend, apply_prof_4x4, derive_affine_subblock_mvs,
    derive_amvp_candidates, derive_gpm_params_table, derive_merge_candidates,
    derive_mmvd_candidate, estimate_inter_block, get_dist_scale_factor,
    interpolate_chroma_block_420, interpolate_luma_block, predict_gpm_block, predict_inter_pu,
    refine_dmvr_subblock, weighted_average_bi,
};
pub use slice::{
    SliceCodingOptions, decode_chroma_intra_mode, decode_full_luma_intra_mode,
    decode_inter_picture, decode_isp_mode, decode_mip_mode, decode_mrl_idx, decode_mvd,
    decode_picture_multitool, decode_picture_with_options, encode_chroma_intra_mode,
    encode_full_luma_intra_mode, encode_inter_picture, encode_isp_mode, encode_mip_mode,
    encode_mrl_idx, encode_mvd, encode_picture_multitool, encode_picture_with_options,
};

/// Extended encoder options for multi-tool Intra and GOP (`I`/`P`/`B`) coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VvcEncoderOptions {
    /// Base picture dimensions and `SliceQpY`.
    pub config: VvcEncoderConfig,
    /// Keyframe interval (`1` = all-Intra `IDR_N_LP`; `> 1` emits `IDR_N_LP` followed by `TRAIL` P/B slices).
    pub keyint: u32,
    /// Slice-level coding tool switches (`67` intra modes, `CCLM`, `DepQuant`, `SDH`, loop filters).
    pub slice_options: SliceCodingOptions,
    /// Sample bit depth (`8`, `10`, or `12`).
    pub bit_depth: u8,
    /// Chroma subsampling format (`4:0:0`, `4:2:0`, `4:2:2`, or `4:4:4`).
    pub chroma_format: ChromaFormatIdc,
    /// Base-2 logarithm of the CTU size (`5..=7`, i.e. `32`, `64`, or `128`).
    /// `CTU` size 16 cannot be signalled by `sps_log2_ctu_size_minus5`.
    pub ctu_log2: u8,
    /// Number of B pictures per mini-GOP (`0` = I/P only; `>= 1` emits B slices using L0 and L1 references).
    pub b_frames: u8,
    /// Optional CQP / VBR rate control configuration (`None` uses constant `config.qp`).
    pub rate_control: Option<VvcRateControlMode>,
    /// Whether to emit an in-band Film Grain Characteristics `PREFIX_SEI_NUT` (`payloadType == 19`) on keyframes.
    pub emit_film_grain_sei: bool,
}

impl Default for VvcEncoderOptions {
    fn default() -> Self {
        Self {
            config: VvcEncoderConfig {
                width: 64,
                height: 64,
                qp: 26,
            },
            keyint: 1,
            slice_options: SliceCodingOptions::default(),
            bit_depth: 8,
            chroma_format: ChromaFormatIdc::Chroma420,
            ctu_log2: CTU_LOG2,
            b_frames: 0,
            rate_control: None,
            emit_film_grain_sei: false,
        }
    }
}

impl VvcEncoderOptions {
    /// Create multi-tool encoder options from a base [`VvcEncoderConfig`].
    pub fn new(config: VvcEncoderConfig) -> Self {
        Self {
            config,
            keyint: 1,
            slice_options: SliceCodingOptions::default(),
            bit_depth: 8,
            chroma_format: ChromaFormatIdc::Chroma420,
            ctu_log2: CTU_LOG2,
            b_frames: 0,
            rate_control: None,
            emit_film_grain_sei: false,
        }
    }
}

/// Stateful H.266 / VVC multi-frame encoder supporting both `IDR_N_LP` Intra
/// pictures and `TRAIL` Inter P/B pictures with an internal [`DecodedPictureBuffer`].
#[derive(Debug)]
pub struct VvcMultiToolEncoder {
    inner: VvcEncoder,
    options: VvcEncoderOptions,
    dpb: DecodedPictureBuffer,
    rate_controller: Option<VvcRateController>,
    vps_rbsp: Vec<u8>,
    sps_rbsp: Vec<u8>,
    pps_rbsp: Vec<u8>,
}

/// Map the slice-tool switches to the SPS/PPS enable flags that the slice-data
/// writer actually signals.
///
/// `mip`, `isp`, and `mrl` are deliberately kept disabled: the multi-tool slice
/// writer does not emit their CABAC syntax yet, so advertising them in the SPS
/// would make a decoder read bins that are not present. The same applies to
/// JCCR, SAO, and ALF until their per-CTU syntax is written.
fn hls_tools_for(options: &SliceCodingOptions, inter: bool) -> hls::VvcHlsTools {
    hls::VvcHlsTools {
        inter,
        mts: !(inter || options.any_tool()),
        lfnst: !(inter || options.any_tool()),
        transform_skip: true,
        bdpcm: options.enable_bdpcm,
        cclm: options.enable_cclm,
        mip: options.enable_mip,
        mrl: options.enable_mrl,
        isp: options.enable_isp,
        joint_cbcr: options.enable_jccr,
        dep_quant: options.dep_quant,
        sign_data_hiding: options.sign_data_hiding,
        sao: options.enable_sao,
        alf: options.enable_alf,
        deblock: options.enable_deblock,
    }
}

impl VvcMultiToolEncoder {
    /// Create a new multi-tool VVC encoder.
    pub fn new(options: VvcEncoderOptions) -> Result<Self, VvcError> {
        let inner = VvcEncoder::new_with_format(
            options.config,
            options.bit_depth,
            options.chroma_format,
            options.ctu_log2,
        )?;
        let tools = hls_tools_for(&options.slice_options, options.keyint > 1);
        let vps_rbsp = hls::build_vps_rbsp_with_tools(&inner.cfg, &tools);
        let is_default_8bit_420 = options.bit_depth == 8
            && options.chroma_format == ChromaFormatIdc::Chroma420
            && options.ctu_log2 == CTU_LOG2;
        let sps_rbsp = if is_default_8bit_420 {
            hls::build_sps_rbsp_with_tools(&inner.cfg, &tools)
        } else {
            hls::build_sps_rbsp_format_bd(
                &inner.cfg,
                &tools,
                options.chroma_format as u8,
                options.bit_depth,
                false,
                false,
                false,
                false,
            )
        };
        let rate_controller = options.rate_control.map(VvcRateController::new);
        Ok(Self {
            options,
            dpb: DecodedPictureBuffer::new(4),
            rate_controller,
            vps_rbsp,
            sps_rbsp,
            pps_rbsp: hls::build_pps_rbsp_with_tools(&inner.cfg, &tools),
            inner,
        })
    }

    /// Effective slice options for this encoder.
    ///
    /// For inter streams the loop filters are disabled (see [`Self::new`]), so
    /// a matching [`VvcDecoder`] must use these options.
    pub fn slice_options(&self) -> &SliceCodingOptions {
        &self.options.slice_options
    }

    /// Encode one `I420Frame` using the configured multi-tool Intra or Inter pipeline.
    pub fn encode_i420(&mut self, frame: &I420Frame) -> Result<EncodedAccessUnit, VvcError> {
        self.inner.ensure_active()?;
        if frame.width != self.options.config.width || frame.height != self.options.config.height {
            return Err(VvcError::FrameDimensions {
                width: frame.width,
                height: frame.height,
                expected_width: self.options.config.width,
                expected_height: self.options.config.height,
            });
        }
        validate_frame_planes(frame)?;

        let padded = pad_i420(frame, self.inner.cfg.width, self.inner.cfg.height);
        let poc = (self.inner.frames_sent % u64::from(self.options.keyint.max(1))) as i32;
        let is_idr = poc == 0 || self.dpb.is_empty();

        if is_idr {
            self.dpb.clear();
            let cabac = if self.options.keyint > 1 {
                slice::encode_picture_multitool(
                    &padded,
                    self.inner.cfg.qp,
                    self.inner.cfg.ctu_log2,
                    &self.options.slice_options,
                )?
            } else {
                slice::encode_picture_with_options(
                    &padded,
                    self.inner.cfg.qp,
                    self.inner.cfg.ctu_log2,
                    &self.options.slice_options,
                )?
            };
            let recon = if self.options.keyint > 1 {
                slice::decode_picture_multitool(
                    &cabac,
                    self.inner.cfg.qp,
                    self.inner.cfg.ctu_log2,
                    self.inner.cfg.width,
                    self.inner.cfg.height,
                    &self.options.slice_options,
                )?
            } else {
                slice::decode_picture_with_options(
                    &cabac,
                    self.inner.cfg.qp,
                    self.inner.cfg.ctu_log2,
                    self.inner.cfg.width,
                    self.inner.cfg.height,
                    &self.options.slice_options,
                )?
            };
            let cw = (self.inner.cfg.width as usize).div_ceil(2);
            let ch = (self.inner.cfg.height as usize).div_ceil(2);
            self.dpb.insert(RefFrame::from_planes(
                0,
                false,
                predict::Plane::from_slice(
                    self.inner.cfg.width as usize,
                    self.inner.cfg.height as usize,
                    &recon.y,
                )
                .expect("Y plane"),
                predict::Plane::from_slice(cw, ch, &recon.u).expect("U plane"),
                predict::Plane::from_slice(cw, ch, &recon.v).expect("V plane"),
                MotionFieldGrid::new(
                    self.inner.cfg.width as usize,
                    self.inner.cfg.height as usize,
                ),
            ));

            let mut au = Vec::with_capacity(cabac.len() + 256);
            hls::push_nal(&mut au, hls::NAL_AUD, 0, 1, &self.inner.aud_rbsp);
            hls::push_nal(&mut au, hls::NAL_VPS, 0, 1, &self.vps_rbsp);
            hls::push_nal(&mut au, hls::NAL_SPS, 0, 1, &self.sps_rbsp);
            hls::push_nal(&mut au, hls::NAL_PPS, 0, 1, &self.pps_rbsp);
            if self.options.emit_film_grain_sei {
                let fg = hls::VvcFilmGrainSei {
                    cancel_flag: false,
                    model_id: 0,
                    separate_colour_desc_present: false,
                    blending_mode_id: 0,
                    log2_scale_factor: 4,
                    comp_model_present: [true, false, false],
                    intervals: [vec![(0, 255, 4)], Vec::new(), Vec::new()],
                    persistence_flag: true,
                };
                hls::push_nal(
                    &mut au,
                    hls::NAL_PREFIX_SEI,
                    0,
                    1,
                    &hls::build_film_grain_sei_rbsp(&fg),
                );
            }
            let tools = hls_tools_for(&self.options.slice_options, false);
            hls::push_nal(
                &mut au,
                hls::NAL_PH,
                0,
                1,
                &hls::build_ph_rbsp_with_tools(0, &tools),
            );
            let mut slice_rbsp = hls::build_slice_header_bits_with_tools(
                0,
                &tools,
                self.options.slice_options.dep_quant,
                self.options.slice_options.sign_data_hiding,
            );
            slice_rbsp.extend_from_slice(&cabac);
            hls::push_nal(&mut au, hls::NAL_IDR_N_LP, 0, 1, &slice_rbsp);

            if let Some(ref mut rc) = self.rate_controller {
                rc.update_after_picture(au.len());
            }
            self.inner.frames_sent += 1;
            Ok(EncodedAccessUnit {
                data: au,
                keyframe: true,
            })
        } else {
            let (l0_refs, l1_refs) = self.dpb.build_ref_lists(poc);
            let l0_owned: Vec<RefFrame> = l0_refs.into_iter().cloned().collect();
            let l1_owned: Vec<RefFrame> = if self.options.b_frames > 0 {
                let mut v: Vec<RefFrame> = l1_refs.into_iter().cloned().collect();
                if v.is_empty() && !l0_owned.is_empty() {
                    v.push(l0_owned[0].clone());
                }
                v
            } else {
                Vec::new()
            };
            let slice_type = if self.options.b_frames > 0 && !l1_owned.is_empty() {
                VvcSliceType::B
            } else {
                VvcSliceType::P
            };
            let (cabac, recon, mv_grid) = slice::encode_inter_picture(
                &padded,
                self.inner.cfg.qp,
                self.inner.cfg.ctu_log2,
                slice_type,
                poc,
                &l0_owned,
                &l1_owned,
                &self.options.slice_options,
            )?;
            let cw = (self.inner.cfg.width as usize).div_ceil(2);
            let ch = (self.inner.cfg.height as usize).div_ceil(2);
            self.dpb.insert(RefFrame::from_planes(
                poc,
                false,
                predict::Plane::from_slice(
                    self.inner.cfg.width as usize,
                    self.inner.cfg.height as usize,
                    &recon.y,
                )
                .expect("Y plane"),
                predict::Plane::from_slice(cw, ch, &recon.u).expect("U plane"),
                predict::Plane::from_slice(cw, ch, &recon.v).expect("V plane"),
                mv_grid,
            ));

            let tools = hls_tools_for(&self.options.slice_options, true);
            let mut au = Vec::with_capacity(cabac.len() + 128);
            let aud_pic_type = if slice_type == VvcSliceType::B { 2 } else { 1 };
            hls::push_nal(
                &mut au,
                hls::NAL_AUD,
                0,
                1,
                &hls::build_aud_rbsp_pic_type(aud_pic_type),
            );
            hls::push_nal(
                &mut au,
                hls::NAL_PH,
                0,
                1,
                &hls::build_ph_rbsp_inter(poc as u32, &tools),
            );
            let mut slice_rbsp = if slice_type == VvcSliceType::B {
                let dp0 = l0_owned.first().map_or(-1, |r| r.poc - poc);
                let dp1 = l1_owned.first().map_or(-1, |r| r.poc - poc);
                hls::build_b_slice_header_bits(
                    0,
                    dp0,
                    dp1,
                    &tools,
                    self.options.slice_options.dep_quant,
                    self.options.slice_options.sign_data_hiding,
                )
            } else {
                hls::build_inter_slice_header_bits(
                    0,
                    &tools,
                    self.options.slice_options.dep_quant,
                    self.options.slice_options.sign_data_hiding,
                )
            };
            slice_rbsp.extend_from_slice(&cabac);
            hls::push_nal(&mut au, hls::NAL_TRAIL, 0, 1, &slice_rbsp);

            if let Some(ref mut rc) = self.rate_controller {
                rc.update_after_picture(au.len());
            }
            self.inner.frames_sent += 1;
            Ok(EncodedAccessUnit {
                data: au,
                keyframe: false,
            })
        }
    }

    /// Encode one [`VvcFrame`] (`bit_depth` in `8..=12`, `chroma_format` in `4:0:0`..=`4:4:4`)
    /// using the configured multi-tool pipeline.
    pub fn encode_vvc_frame(&mut self, frame: &VvcFrame) -> Result<EncodedAccessUnit, VvcError> {
        self.inner.ensure_active()?;
        if frame.width != self.options.config.width || frame.height != self.options.config.height {
            return Err(VvcError::FrameDimensions {
                width: frame.width,
                height: frame.height,
                expected_width: self.options.config.width,
                expected_height: self.options.config.height,
            });
        }
        validate_vvc_frame_planes(frame)?;
        if self.options.bit_depth == 8
            && self.options.chroma_format == ChromaFormatIdc::Chroma420
            && self.options.ctu_log2 == CTU_LOG2
            && frame.bit_depth == 8
            && frame.chroma_format == ChromaFormatIdc::Chroma420
        {
            return self.encode_i420(&frame.to_i420());
        }

        let padded = pad_vvc_frame(frame, self.inner.cfg.width, self.inner.cfg.height);
        let (cabac, recon) = slice::encode_vvc_frame_picture(
            &padded,
            self.inner.cfg.qp,
            self.inner.cfg.ctu_log2,
            &self.options.slice_options,
        )?;
        let recon_i420 = recon.to_i420();
        let cw = (self.inner.cfg.width as usize).div_ceil(2);
        let ch = (self.inner.cfg.height as usize).div_ceil(2);
        self.dpb.clear();
        self.dpb.insert(RefFrame::from_planes(
            0,
            false,
            predict::Plane::from_slice(
                self.inner.cfg.width as usize,
                self.inner.cfg.height as usize,
                &recon_i420.y,
            )
            .expect("Y plane"),
            predict::Plane::from_slice(cw, ch, &recon_i420.u).expect("U plane"),
            predict::Plane::from_slice(cw, ch, &recon_i420.v).expect("V plane"),
            MotionFieldGrid::new(
                self.inner.cfg.width as usize,
                self.inner.cfg.height as usize,
            ),
        ));

        let mut au = Vec::with_capacity(cabac.len() + 256);
        hls::push_nal(&mut au, hls::NAL_AUD, 0, 1, &self.inner.aud_rbsp);
        hls::push_nal(&mut au, hls::NAL_VPS, 0, 1, &self.vps_rbsp);
        hls::push_nal(&mut au, hls::NAL_SPS, 0, 1, &self.sps_rbsp);
        hls::push_nal(&mut au, hls::NAL_PPS, 0, 1, &self.pps_rbsp);
        let tools = hls_tools_for(&self.options.slice_options, false);
        hls::push_nal(
            &mut au,
            hls::NAL_PH,
            0,
            1,
            &hls::build_ph_rbsp_with_tools(0, &tools),
        );
        let mut slice_rbsp = hls::build_slice_header_bits_with_tools(
            0,
            &tools,
            self.options.slice_options.dep_quant,
            self.options.slice_options.sign_data_hiding,
        );
        slice_rbsp.extend_from_slice(&cabac);
        hls::push_nal(&mut au, hls::NAL_IDR_N_LP, 0, 1, &slice_rbsp);

        self.inner.frames_sent += 1;
        Ok(EncodedAccessUnit {
            data: au,
            keyframe: true,
        })
    }
}

/// Stateful H.266 / VVC Annex-B decoder supporting both [`VvcEncoder`] streams
/// and [`VvcMultiToolEncoder`] Intra/Inter streams.
#[derive(Debug)]
pub struct VvcDecoder {
    sps_table: [Option<VvcSpsInfo>; 16],
    full_sps_table: [Option<hls::VvcSps>; 16],
    pps_table: [Option<PpsInfo>; 64],
    full_pps_table: [Option<hls::VvcPps>; 64],
    aps_bank: hls::VvcActiveApsBank,
    active_film_grain: Option<hls::VvcFilmGrainSei>,
    dpb: DecodedPictureBuffer,
    slice_options: SliceCodingOptions,
    last_vvc_frame: Option<VvcFrame>,
}

impl Default for VvcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl VvcDecoder {
    /// Create a new stateful VVC decoder with default options.
    pub fn new() -> Self {
        Self::with_options(SliceCodingOptions::default())
    }

    /// Create a new stateful VVC decoder with explicit [`SliceCodingOptions`].
    pub fn with_options(slice_options: SliceCodingOptions) -> Self {
        Self {
            sps_table: std::array::from_fn(|_| None),
            full_sps_table: std::array::from_fn(|_| None),
            pps_table: [None; 64],
            full_pps_table: std::array::from_fn(|_| None),
            aps_bank: hls::VvcActiveApsBank::default(),
            active_film_grain: None,
            dpb: DecodedPictureBuffer::new(4),
            slice_options,
            last_vvc_frame: None,
        }
    }

    /// Decode a sequence of Annex-B access units into [`VvcFrame`]s (`bit_depth` in `8..=12`,
    /// `chroma_format` in `4:0:0`..=`4:4:4`).
    pub fn decode_annexb_stream_vvc_frames(
        &mut self,
        access_units: &[&[u8]],
    ) -> Result<Vec<VvcFrame>, VvcError> {
        let mut out = Vec::with_capacity(access_units.len());
        for au in access_units {
            if let Some(frame) = self.decode_access_unit_vvc_frame(au)? {
                out.push(frame);
            }
        }
        Ok(out)
    }

    /// Decode one Annex-B access unit into a [`VvcFrame`] (`bit_depth` in `8..=12`,
    /// `chroma_format` in `4:0:0`..=`4:4:4`).
    pub fn decode_access_unit_vvc_frame(
        &mut self,
        au: &[u8],
    ) -> Result<Option<VvcFrame>, VvcError> {
        self.last_vvc_frame = None;
        let i420_opt = self.decode_access_unit(au)?;
        if let Some(vf) = self.last_vvc_frame.take() {
            return Ok(Some(vf));
        }
        Ok(i420_opt.and_then(|f| VvcFrame::from_i420(&f)))
    }

    /// Return the currently active Adaptation Parameter Set (APS) bank (`ALF`, `LMCS`, `ScalingList`).
    pub fn active_aps_bank(&self) -> &hls::VvcActiveApsBank {
        &self.aps_bank
    }

    /// Return the most recently received Film Grain Characteristics SEI message, if active.
    pub fn active_film_grain(&self) -> Option<&hls::VvcFilmGrainSei> {
        self.active_film_grain.as_ref()
    }

    /// Decode a sequence of Annex-B access units into `I420Frame`s.
    pub fn decode_annexb_stream(
        &mut self,
        access_units: &[&[u8]],
    ) -> Result<Vec<I420Frame>, VvcError> {
        let mut out = Vec::with_capacity(access_units.len());
        for au in access_units {
            if let Some(frame) = self.decode_access_unit(au)? {
                out.push(frame);
            }
        }
        Ok(out)
    }

    /// Decode one Annex-B access unit and update the internal DPB.
    pub fn decode_access_unit(&mut self, au: &[u8]) -> Result<Option<I420Frame>, VvcError> {
        let nals = parse_annexb_nals(au)?;
        let mut active_pps_id: Option<u8> = None;
        let mut active_inter_allowed = false;
        let mut active_poc: i32 = 0;
        let mut decoded_frame: Option<I420Frame> = None;

        for nal in &nals {
            if nal.layer_id != 0 {
                return Err(VvcError::Unsupported("multi-layer stream"));
            }
            match nal.nal_type {
                VvcNalType::Opi => {
                    let opi = hls::parse_opi_rbsp(&nal.rbsp)?;
                    if let Some(ols_idx) = opi.ols_idx {
                        if ols_idx > 0 {
                            return Err(VvcError::Unsupported("multi-layer stream"));
                        }
                    }
                }
                VvcNalType::Dci => {
                    let _dci = hls::parse_dci_rbsp(&nal.rbsp)?;
                }
                VvcNalType::Vps => {
                    let vps = hls::parse_vps_rbsp(&nal.rbsp)?;
                    if vps.max_layers_minus1 != 0 {
                        return Err(VvcError::Unsupported("multi-layer stream"));
                    }
                }
                VvcNalType::Sps => {
                    let info = parse_sps_info(&nal.rbsp)?;
                    validate_decoded_sps(&info)?;
                    let sps_id = usize::from(info.sps_id);
                    if let Ok(full_sps) = hls::parse_full_sps_rbsp(&nal.rbsp) {
                        self.full_sps_table[sps_id] = Some(full_sps);
                    }
                    self.sps_table[sps_id] = Some(info);
                }
                VvcNalType::Pps => {
                    let info = parse_pps_info(&nal.rbsp)?;
                    if let Ok(full_pps) = hls::parse_full_pps_rbsp(&nal.rbsp) {
                        self.full_pps_table[usize::from(info.pps_id)] = Some(full_pps);
                    }
                    self.pps_table[usize::from(info.pps_id)] = Some(info);
                }
                VvcNalType::PrefixAps | VvcNalType::SuffixAps => {
                    let aps = hls::parse_aps_rbsp(&nal.rbsp)?;
                    self.aps_bank.activate(aps);
                }
                VvcNalType::PrefixSei | VvcNalType::SuffixSei => {
                    for msg in hls::parse_sei_rbsp(&nal.rbsp)? {
                        if let hls::VvcSeiMessage::FilmGrain(fg) = msg {
                            if fg.cancel_flag {
                                self.active_film_grain = None;
                            } else {
                                self.active_film_grain = Some(fg);
                            }
                        }
                    }
                }
                VvcNalType::Ph => {
                    // Prefer the complete `picture_header_structure` parser so
                    // that `ph_pic_order_cnt_lsb` uses `sps_log2_max_pic_order_
                    // cnt_lsb_minus4` and every tool flag is read at the right
                    // position (`H.266 §7.3.2.8`). Fall back to the minimal
                    // layout when the parameter sets are not fully parsed.
                    if let Ok(ph) = hls::parse_full_ph_rbsp(
                        &nal.rbsp,
                        &self.full_sps_table,
                        &self.full_pps_table,
                    ) {
                        active_pps_id = Some(ph.pps_id);
                        active_inter_allowed = ph.inter_slice_allowed;
                        active_poc = ph.poc_lsb as i32;
                    } else {
                        let mut br = bits::BitReader::new(&nal.rbsp);
                        let gdr_or_irap = header_read(br.read_bit(), "PH: gdr_or_irap")?;
                        let _non_ref = header_read(br.read_bit(), "PH: non_ref")?;
                        if gdr_or_irap {
                            let _gdr = header_read(br.read_bit(), "PH: gdr")?;
                        }
                        let inter_allowed = header_read(br.read_bit(), "PH: inter_slice_allowed")?;
                        if inter_allowed {
                            let _intra_allowed =
                                header_read(br.read_bit(), "PH: intra_slice_allowed")?;
                        }
                        let pps_id = header_read(br.read_ue(), "PH: pps_id")? as u8;
                        let poc_lsb = header_read(br.read_bits(8), "PH: poc_lsb")? as i32;
                        active_pps_id = Some(pps_id);
                        active_inter_allowed = inter_allowed;
                        active_poc = poc_lsb;
                    }
                }
                nal_type if is_vcl_nal(nal_type) => {
                    // `sh_picture_header_in_slice_header_flag` is the first bit
                    // of every slice header (`H.266 §7.3.7`). When it is set,
                    // `picture_header_structure` is embedded in the slice
                    // header and the access unit carries no PH NAL unit. That
                    // is VTM's default layout (`EnablePictureHeaderInSliceHeader`,
                    // `EncGOP.cpp:4456`, `VLCWriter.cpp:2227-2231`).
                    let embedded_header = if nal.rbsp.first().is_some_and(|byte| byte & 0x80 != 0) {
                        Some(hls::parse_full_slice_header(
                            &nal.rbsp,
                            nal_type.code(),
                            None,
                            &self.full_sps_table,
                            &self.full_pps_table,
                        )?)
                    } else {
                        None
                    };
                    let embedded_ph = match &embedded_header {
                        Some((sh, _)) => Some(sh.embedded_ph.as_ref().ok_or_else(|| {
                            VvcError::Bitstream(
                                "slice header: ph_in_slice_header_flag is set but the picture header is missing"
                                    .to_string(),
                            )
                        })?),
                        None => None,
                    };
                    let (pps_id, slice_poc, inter_allowed) = match embedded_ph {
                        Some(ph) => (ph.pps_id, ph.poc_lsb as i32, ph.inter_slice_allowed),
                        None => {
                            let pps_id = active_pps_id.ok_or_else(|| {
                                VvcError::Bitstream(
                                    "VCL NAL unit without a preceding picture header".to_string(),
                                )
                            })?;
                            (pps_id, active_poc, active_inter_allowed)
                        }
                    };
                    if embedded_ph.is_some() && inter_allowed {
                        return Err(VvcError::Unsupported(
                            "inter slice with an embedded picture header",
                        ));
                    }
                    let pps = self.pps_table[usize::from(pps_id)].ok_or_else(|| {
                        VvcError::Bitstream(format!("PH references unknown PPS {pps_id}"))
                    })?;
                    let info = self.sps_table[usize::from(pps.sps_id)]
                        .as_ref()
                        .ok_or_else(|| {
                            VvcError::Bitstream(format!(
                                "PPS {} references unknown SPS {}",
                                pps.pps_id, pps.sps_id
                            ))
                        })?;
                    let mut eff_opts = self.slice_options;
                    if let Some(full_sps) = &self.full_sps_table[usize::from(pps.sps_id)] {
                        eff_opts.enable_cclm |= full_sps.cclm_enabled;
                        eff_opts.enable_mip |= full_sps.mip_enabled;
                        eff_opts.enable_mrl |= full_sps.mrl_enabled;
                        eff_opts.enable_isp |= full_sps.isp_enabled;
                        eff_opts.enable_jccr |= full_sps.joint_cbcr_enabled;
                        eff_opts.dep_quant |= full_sps.dep_quant_enabled;
                        eff_opts.sign_data_hiding |= full_sps.sign_data_hiding_enabled;
                        eff_opts.enable_sao |= full_sps.sao_enabled;
                        eff_opts.enable_alf |= full_sps.alf_enabled;
                        eff_opts.enable_palette |= full_sps.palette_enabled;
                        eff_opts.enable_ibc |= full_sps.ibc_enabled;
                        eff_opts.enable_act |= full_sps.act_enabled;
                        if !(full_sps.mts_enabled && full_sps.lfnst_enabled) {
                            eff_opts.full_intra_modes = true;
                        }
                    }
                    if let Some(full_pps) = &self.full_pps_table[usize::from(pps_id)] {
                        eff_opts.enable_deblock |= !full_pps.deblocking_disabled;
                    }
                    // The slice reader below decodes one 32x32 coding unit per
                    // CTU quadrant and only understands a quadtree down to that
                    // unit. Reject other coding trees as unsupported instead of
                    // desynchronising the CABAC stream and reporting a missing
                    // terminating bit much later.
                    if let Some(full_sps) = &self.full_sps_table[usize::from(pps.sps_id)] {
                        let min_qt_intra = full_sps
                            .min_cb_log2
                            .saturating_add(full_sps.log2_diff_min_qt_min_cb_intra_luma);
                        let min_qt_inter = full_sps
                            .min_cb_log2
                            .saturating_add(full_sps.log2_diff_min_qt_min_cb_inter);
                        if min_qt_intra < 5
                            || min_qt_inter < 5
                            || full_sps.max_mtt_depth_intra_luma != 0
                            || full_sps.max_mtt_depth_inter != 0
                            || full_sps.dual_tree_intra
                        {
                            return Err(VvcError::Unsupported(
                                "coding tree with a minimum QT below 32, MTT, or dual-tree intra",
                            ));
                        }
                    }
                    let (qp_delta, data_off) = match &embedded_header {
                        Some((sh, data_off)) => {
                            // With `ph_in_slice_header_flag == 1` the slice
                            // header carries the flags that normally live in
                            // the PH and override the PPS defaults
                            // (`H.266 §7.3.7`).
                            eff_opts.dep_quant = sh.dep_quant_used;
                            eff_opts.sign_data_hiding = sh.sign_data_hiding_used;
                            eff_opts.enable_deblock = !sh.deblocking_disabled;
                            eff_opts.enable_lmcs = sh.lmcs_used;
                            (sh.qp_delta, *data_off)
                        }
                        None => parse_our_slice_header(&nal.rbsp, &eff_opts, !nal_type.is_irap())?,
                    };
                    let slice_qp = 26i32
                        .saturating_add(pps.init_qp_minus26)
                        .saturating_add(qp_delta)
                        .clamp(0, 63) as u8;
                    let cabac = nal.rbsp.get(data_off..).ok_or_else(|| {
                        VvcError::Bitstream("slice data offset out of range".to_string())
                    })?;

                    let cw = (info.coded_width as usize).div_ceil(2);
                    let ch = (info.coded_height as usize).div_ceil(2);

                    // `QpC` comes from the signalled chroma QP mapping table
                    // (`H.266 §7.4.3.4`), not from a hard-coded default: VTM
                    // signals `QpInValCb`/`QpOutValCb` in the SPS.
                    let chroma_qp_cb = match &self.full_sps_table[usize::from(pps.sps_id)] {
                        Some(full_sps) => {
                            full_sps.mapped_chroma_qp(1, i32::from(slice_qp), 0) as u8
                        }
                        None => hls::derived_chroma_qp(i32::from(slice_qp)).clamp(0, 63) as u8,
                    };

                    if nal_type.is_irap() && !inter_allowed {
                        self.dpb.clear();
                        let cf = ChromaFormatIdc::from_idc(info.chroma_format_idc);
                        let bd = info.bit_depth_luma;
                        if bd != 8 || cf != ChromaFormatIdc::Chroma420 || info.ctu_log2 != CTU_LOG2
                        {
                            let coded_vvc = slice::decode_vvc_frame_picture_with_chroma_qp(
                                cabac,
                                slice_qp,
                                chroma_qp_cb,
                                info.ctu_log2,
                                info.coded_width,
                                info.coded_height,
                                bd,
                                cf,
                                &eff_opts,
                            )?;
                            let mut cropped_vvc =
                                crop_vvc_frame(&coded_vvc, info.width, info.height);
                            if let Some(ref fg) = self.active_film_grain {
                                if !fg.cancel_flag {
                                    apply_film_grain_to_vvc_frame(
                                        &mut cropped_vvc,
                                        fg,
                                        slice_poc as u32,
                                    );
                                }
                            }
                            let i420 = cropped_vvc.to_i420();
                            self.last_vvc_frame = Some(cropped_vvc);
                            decoded_frame = Some(i420);
                            continue;
                        }
                        // An inter stream disables MTS/LFNST in the SPS and
                        // always uses the multi-tool intra writer, even when
                        // every optional slice tool is off.
                        let inter_stream =
                            info.gci_bits.get(1).is_some_and(|intra_only| !intra_only);
                        let coded_frame = if !eff_opts.any_tool() && !inter_stream {
                            slice::decode_picture(
                                cabac,
                                slice_qp,
                                info.ctu_log2,
                                info.coded_width,
                                info.coded_height,
                            )?
                        } else {
                            slice::decode_picture_multitool_with_chroma_qp(
                                cabac,
                                slice_qp,
                                chroma_qp_cb,
                                info.ctu_log2,
                                info.coded_width,
                                info.coded_height,
                                &eff_opts,
                            )?
                        };
                        self.dpb.insert(RefFrame::from_planes(
                            0,
                            false,
                            predict::Plane::from_slice(
                                info.coded_width as usize,
                                info.coded_height as usize,
                                &coded_frame.y,
                            )
                            .expect("Y"),
                            predict::Plane::from_slice(cw, ch, &coded_frame.u).expect("U"),
                            predict::Plane::from_slice(cw, ch, &coded_frame.v).expect("V"),
                            MotionFieldGrid::new(
                                info.coded_width as usize,
                                info.coded_height as usize,
                            ),
                        ));
                        let mut disp = crop_frame(&coded_frame, info.width, info.height);
                        if let Some(ref fg) = self.active_film_grain {
                            if !fg.cancel_flag {
                                apply_film_grain_to_i420(&mut disp, fg, slice_poc as u32);
                            }
                        }
                        decoded_frame = Some(disp);
                    } else {
                        let mut sh_br = bits::BitReader::new(&nal.rbsp);
                        let _ = sh_br.read_bit();
                        let raw_st = sh_br.read_ue().unwrap_or(1);
                        let st = if raw_st == 0 {
                            VvcSliceType::B
                        } else {
                            VvcSliceType::P
                        };
                        let (l0_refs, l1_refs) = self.dpb.build_ref_lists(slice_poc);
                        let l0_owned: Vec<RefFrame> = l0_refs.into_iter().cloned().collect();
                        let l1_owned: Vec<RefFrame> = if st == VvcSliceType::B {
                            let mut v: Vec<RefFrame> = l1_refs.into_iter().cloned().collect();
                            if v.is_empty() && !l0_owned.is_empty() {
                                v.push(l0_owned[0].clone());
                            }
                            v
                        } else {
                            Vec::new()
                        };
                        let (coded_frame, mv_grid) = slice::decode_inter_picture(
                            cabac,
                            slice_qp,
                            info.ctu_log2,
                            info.coded_width,
                            info.coded_height,
                            st,
                            &l0_owned,
                            &l1_owned,
                            &eff_opts,
                        )?;
                        self.dpb.insert(RefFrame::from_planes(
                            slice_poc,
                            false,
                            predict::Plane::from_slice(
                                info.coded_width as usize,
                                info.coded_height as usize,
                                &coded_frame.y,
                            )
                            .expect("Y"),
                            predict::Plane::from_slice(cw, ch, &coded_frame.u).expect("U"),
                            predict::Plane::from_slice(cw, ch, &coded_frame.v).expect("V"),
                            mv_grid,
                        ));
                        let mut disp = crop_frame(&coded_frame, info.width, info.height);
                        if let Some(ref fg) = self.active_film_grain {
                            if !fg.cancel_flag {
                                apply_film_grain_to_i420(&mut disp, fg, slice_poc as u32);
                            }
                        }
                        decoded_frame = Some(disp);
                    }
                }
                _ => {}
            }
        }
        Ok(decoded_frame)
    }
}

pub use hls::{
    SEI_PAYLOAD_FILM_GRAIN_CHARACTERISTICS, VvcActiveApsBank, VvcFilmGrainSei, VvcSeiMessage,
    VvcTileGrid, build_film_grain_sei_rbsp, build_ph_rbsp_with_tools, build_sps_rbsp_format_bd,
    parse_film_grain_sei_payload, parse_sei_rbsp,
};
pub use inter::{
    RPR_SCALE_1X, RprScalingRatio, derive_ibc_merge_candidates, derive_rpr_scaling_ratio,
    interpolate_chroma_block_format, interpolate_chroma_rpr_block, interpolate_luma_block_16,
    interpolate_luma_rpr_block, is_ibc_bv_valid, predict_ibc_block, search_ibc_block_vector,
};
pub use predict::{
    ChromaFormatIdc, PALETTE_RUN_LEFT_LUT, PALETTE_RUN_TOP_LUT, PaletteBlock, Plane16,
    ReferenceSamples16, build_palette_from_block, derive_palette_run_ctx,
    downsample_luma_for_cclm_format, gather_references_16, palette_traverse_coord,
    predict_angular_16, predict_block_16, predict_cclm_with_format, reconstruct_palette_block,
};
pub use slice::{
    choose_qt_mtt_partition_rdo, decode_act_flag, decode_coding_tree_split,
    decode_ctu_alf_luma_fixed, decode_ctu_sao, decode_ibc_cu_syntax, decode_palette_cu_syntax,
    encode_act_flag, encode_coding_tree_split, encode_ctu_alf_luma_fixed, encode_ctu_sao,
    encode_ibc_cu_syntax, encode_palette_cu_syntax,
};
pub use transform::{
    CodingTreeSplit, DELTA_QP_ACT, QtMttCanSplit, QtMttSplitContexts, apply_act_forward,
    apply_act_inverse, can_split_coding_tree, derive_split_contexts, quantize_ts_with_mode,
};

/// General H.266 / VVC planar frame supporting 8-bit and 10-bit sample depths and
/// `4:2:0`, `4:2:2`, and `4:4:4` chroma subsampling layouts (Wave A).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VvcFrame {
    /// Visible width in luma samples.
    pub width: u32,
    /// Visible height in luma samples.
    pub height: u32,
    /// Sample bit depth (`8` or `10`).
    pub bit_depth: u8,
    /// Chroma subsampling format (`4:2:0`, `4:2:2`, or `4:4:4`).
    pub chroma_format: ChromaFormatIdc,
    /// Luma (or G in RGB/ACT) plane (`width x height`).
    pub y: Plane16,
    /// Cb (or B in RGB/ACT) plane (`(width / SubWidthC) x (height / SubHeightC)`).
    pub u: Plane16,
    /// Cr (or R in RGB/ACT) plane (`(width / SubWidthC) x (height / SubHeightC)`).
    pub v: Plane16,
}

impl VvcFrame {
    /// Allocate a blank [`VvcFrame`] for `(width, height, bit_depth, chroma_format)`
    /// (`bit_depth` in `8..=12`, `chroma_format` in `4:0:0`..=`4:4:4`).
    pub fn new(width: u32, height: u32, bit_depth: u8, chroma_format: ChromaFormatIdc) -> Self {
        let w = width as usize;
        let h = height as usize;
        let (cw, ch) = if chroma_format.is_chroma_enabled() {
            (
                w.div_ceil(chroma_format.sub_width_c()),
                h.div_ceil(chroma_format.sub_height_c()),
            )
        } else {
            (0, 0)
        };
        let bd = bit_depth.clamp(8, 12);
        Self {
            width,
            height,
            bit_depth: bd,
            chroma_format,
            y: Plane16::new(w, h, bd),
            u: Plane16::new(cw, ch, bd),
            v: Plane16::new(cw, ch, bd),
        }
    }

    /// True when the Y, U, and V plane dimensions and buffer lengths match `(width, height, chroma_format)`.
    pub fn is_valid(&self) -> bool {
        let w = self.width as usize;
        let h = self.height as usize;
        let (cw, ch) = if self.chroma_format.is_chroma_enabled() {
            (
                w.div_ceil(self.chroma_format.sub_width_c()),
                h.div_ceil(self.chroma_format.sub_height_c()),
            )
        } else {
            (0, 0)
        };
        self.y.width == w
            && self.y.height == h
            && self.y.data.len() == w * h
            && self.u.width == cw
            && self.u.height == ch
            && self.u.data.len() == cw * ch
            && self.v.width == cw
            && self.v.height == ch
            && self.v.data.len() == cw * ch
    }

    /// Construct a [`VvcFrame`] from an 8-bit 4:2:0 [`I420Frame`].
    pub fn from_i420(frame: &I420Frame) -> Option<Self> {
        let w = frame.width as usize;
        let h = frame.height as usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let py = predict::Plane::from_slice(w, h, &frame.y)?;
        let pu = predict::Plane::from_slice(cw, ch, &frame.u)?;
        let pv = predict::Plane::from_slice(cw, ch, &frame.v)?;
        Some(Self {
            width: frame.width,
            height: frame.height,
            bit_depth: 8,
            chroma_format: ChromaFormatIdc::Chroma420,
            y: Plane16::from_plane_8bit(&py, 8),
            u: Plane16::from_plane_8bit(&pu, 8),
            v: Plane16::from_plane_8bit(&pv, 8),
        })
    }

    /// Convert this [`VvcFrame`] to an 8-bit 4:2:0 [`I420Frame`] (downshifting 10-bit
    /// samples by 2 bits and subsampling `4:2:2` / `4:4:4` chroma to `4:2:0` if needed).
    pub fn to_i420(&self) -> I420Frame {
        let y8 = self.y.to_plane_8bit();
        let u8_src = self.u.to_plane_8bit();
        let v8_src = self.v.to_plane_8bit();
        let cw = (self.width as usize).div_ceil(2);
        let ch = (self.height as usize).div_ceil(2);
        if self.chroma_format == ChromaFormatIdc::Chroma420 {
            return I420Frame {
                width: self.width,
                height: self.height,
                y: y8.data,
                u: u8_src.data,
                v: v8_src.data,
            };
        }
        if !self.chroma_format.is_chroma_enabled() || u8_src.width == 0 || u8_src.height == 0 {
            return I420Frame {
                width: self.width,
                height: self.height,
                y: y8.data,
                u: vec![128u8; cw * ch],
                v: vec![128u8; cw * ch],
            };
        }
        let sub_w = self.chroma_format.sub_width_c();
        let sub_h = self.chroma_format.sub_height_c();
        let mut u_out = vec![128u8; cw * ch];
        let mut v_out = vec![128u8; cw * ch];
        for cy in 0..ch {
            let sy = (cy * 2 / sub_h).min(u8_src.height.saturating_sub(1));
            for cx in 0..cw {
                let sx = (cx * 2 / sub_w).min(u8_src.width.saturating_sub(1));
                u_out[cy * cw + cx] = u8_src.get(sx, sy);
                v_out[cy * cw + cx] = v8_src.get(sx, sy);
            }
        }
        I420Frame {
            width: self.width,
            height: self.height,
            y: y8.data,
            u: u_out,
            v: v_out,
        }
    }
}

fn apply_film_grain_to_i420(frame: &mut I420Frame, fg: &hls::VvcFilmGrainSei, seed: u32) {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    if fg.comp_model_present[0] && !fg.intervals[0].is_empty() {
        if let Some(mut py) = predict::Plane::from_slice(w, h, &frame.y) {
            filter::apply_film_grain_plane(&mut py, fg.log2_scale_factor, &fg.intervals[0], seed);
            frame.y = py.data;
        }
    }
    if fg.comp_model_present[1] && !fg.intervals[1].is_empty() {
        if let Some(mut pu) = predict::Plane::from_slice(cw, ch, &frame.u) {
            filter::apply_film_grain_plane(
                &mut pu,
                fg.log2_scale_factor,
                &fg.intervals[1],
                seed.wrapping_add(1),
            );
            frame.u = pu.data;
        }
    }
    if fg.comp_model_present[2] && !fg.intervals[2].is_empty() {
        if let Some(mut pv) = predict::Plane::from_slice(cw, ch, &frame.v) {
            filter::apply_film_grain_plane(
                &mut pv,
                fg.log2_scale_factor,
                &fg.intervals[2],
                seed.wrapping_add(2),
            );
            frame.v = pv.data;
        }
    }
}

fn apply_film_grain_to_vvc_frame(frame: &mut VvcFrame, fg: &hls::VvcFilmGrainSei, seed: u32) {
    if fg.comp_model_present[0] && !fg.intervals[0].is_empty() {
        filter::apply_film_grain_plane_16(
            &mut frame.y,
            fg.log2_scale_factor,
            &fg.intervals[0],
            seed,
        );
    }
    if frame.chroma_format.is_chroma_enabled() {
        if fg.comp_model_present[1] && !fg.intervals[1].is_empty() {
            filter::apply_film_grain_plane_16(
                &mut frame.u,
                fg.log2_scale_factor,
                &fg.intervals[1],
                seed.wrapping_add(1),
            );
        }
        if fg.comp_model_present[2] && !fg.intervals[2].is_empty() {
            filter::apply_film_grain_plane_16(
                &mut frame.v,
                fg.log2_scale_factor,
                &fg.intervals[2],
                seed.wrapping_add(2),
            );
        }
    }
}

fn validate_vvc_frame_planes(frame: &VvcFrame) -> Result<(), VvcError> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let (cw, ch) = if frame.chroma_format.is_chroma_enabled() {
        (
            w.div_ceil(frame.chroma_format.sub_width_c()),
            h.div_ceil(frame.chroma_format.sub_height_c()),
        )
    } else {
        (0, 0)
    };
    if frame.y.data.len() != w * h {
        return Err(VvcError::PlaneSize {
            plane: "Y",
            actual: frame.y.data.len(),
            expected: w * h,
        });
    }
    if frame.u.data.len() != cw * ch {
        return Err(VvcError::PlaneSize {
            plane: "U",
            actual: frame.u.data.len(),
            expected: cw * ch,
        });
    }
    if frame.v.data.len() != cw * ch {
        return Err(VvcError::PlaneSize {
            plane: "V",
            actual: frame.v.data.len(),
            expected: cw * ch,
        });
    }
    Ok(())
}

fn pad_vvc_frame(frame: &VvcFrame, width: u32, height: u32) -> VvcFrame {
    let pad_p16 = |src: &Plane16, dw: usize, dh: usize| -> Plane16 {
        if dw == 0 || dh == 0 || src.width == 0 || src.height == 0 {
            return Plane16::new(dw, dh, src.bit_depth);
        }
        let mut out = Plane16::new(dw, dh, src.bit_depth);
        for y in 0..dh {
            let sy = y.min(src.height - 1);
            for x in 0..dw {
                let sx = x.min(src.width - 1);
                out.set(x, y, src.get(sx, sy));
            }
        }
        out
    };
    let cf = frame.chroma_format;
    let w = width as usize;
    let h = height as usize;
    let (cw, ch) = if cf.is_chroma_enabled() {
        (w.div_ceil(cf.sub_width_c()), h.div_ceil(cf.sub_height_c()))
    } else {
        (0, 0)
    };
    VvcFrame {
        width,
        height,
        bit_depth: frame.bit_depth,
        chroma_format: cf,
        y: pad_p16(&frame.y, w, h),
        u: pad_p16(&frame.u, cw, ch),
        v: pad_p16(&frame.v, cw, ch),
    }
}

fn crop_vvc_frame(frame: &VvcFrame, width: u32, height: u32) -> VvcFrame {
    if frame.width == width && frame.height == height {
        return frame.clone();
    }
    let crop_p16 = |src: &Plane16, dw: usize, dh: usize| -> Plane16 {
        if dw == 0 || dh == 0 || src.width == 0 || src.height == 0 {
            return Plane16::new(dw, dh, src.bit_depth);
        }
        let mut out = Plane16::new(dw, dh, src.bit_depth);
        for y in 0..dh.min(src.height) {
            for x in 0..dw.min(src.width) {
                out.set(x, y, src.get(x, y));
            }
        }
        out
    };
    let cf = frame.chroma_format;
    let w = width as usize;
    let h = height as usize;
    let (cw, ch) = if cf.is_chroma_enabled() {
        (w.div_ceil(cf.sub_width_c()), h.div_ceil(cf.sub_height_c()))
    } else {
        (0, 0)
    };
    VvcFrame {
        width,
        height,
        bit_depth: frame.bit_depth,
        chroma_format: cf,
        y: crop_p16(&frame.y, w, h),
        u: crop_p16(&frame.u, cw, ch),
        v: crop_p16(&frame.v, cw, ch),
    }
}

/// Encode and reconstruct an 8/10/12-bit / 4:0:0..4:4:4 [`VvcFrame`] (with optional 4:4:4
/// Adaptive Colour Transform `enable_act`) using the normative H.266 transform,
/// quantisation (`qp_bd_offset`), and intra prediction pipeline (Wave A / Wave H).
pub fn encode_and_reconstruct_vvc_frame(
    frame: &VvcFrame,
    qp: u8,
    enable_act: bool,
) -> Result<(Vec<u8>, VvcFrame), VvcError> {
    if !frame.chroma_format.is_chroma_enabled() {
        let padded = pad_vvc_frame(
            frame,
            frame.width.div_ceil(32) * 32,
            frame.height.div_ceil(32) * 32,
        );
        let (cabac, recon) =
            slice::encode_vvc_frame_picture(&padded, qp, 5, &SliceCodingOptions::default())?;
        return Ok((cabac, crop_vvc_frame(&recon, frame.width, frame.height)));
    }
    let ctu = 32usize;
    let bd = frame.bit_depth.clamp(8, 12);
    let cf = frame.chroma_format;
    let sub_w = cf.sub_width_c();
    let sub_h = cf.sub_height_c();
    let coded_w = (frame.width as usize).div_ceil(ctu) * ctu;
    let coded_h = (frame.height as usize).div_ceil(ctu) * ctu;
    let coded_cw = coded_w / sub_w;
    let coded_ch = coded_h / sub_h;
    let ctu_cw = ctu / sub_w;
    let ctu_ch = ctu / sub_h;

    let mut recon_y = Plane16::new(coded_w, coded_h, bd);
    let mut recon_u = Plane16::new(coded_cw, coded_ch, bd);
    let mut recon_v = Plane16::new(coded_cw, coded_ch, bd);

    let qp_bd = dct::qp_bd_offset(bd);
    let use_act = enable_act && cf == ChromaFormatIdc::Chroma444;
    let base_luma_qp = (i32::from(qp) + qp_bd + if use_act { DELTA_QP_ACT[0] } else { 0 })
        .clamp(4, 63 + qp_bd) as u8;
    let base_cb_qp =
        (hls::derived_chroma_qp(i32::from(qp)) + qp_bd + if use_act { DELTA_QP_ACT[1] } else { 0 })
            .clamp(4, 63 + qp_bd) as u8;
    let base_cr_qp =
        (hls::derived_chroma_qp(i32::from(qp)) + qp_bd + if use_act { DELTA_QP_ACT[2] } else { 0 })
            .clamp(4, 63 + qp_bd) as u8;

    let mut enc = cabac::BinEncoder::new();
    let mut ctx = cabac::CtxStore::new_i_slice(i32::from(qp));
    let max_val = (1i32 << u32::from(bd)) - 1;

    for by in (0..coded_h).step_by(ctu) {
        for bx in (0..coded_w).step_by(ctu) {
            let cx = bx / sub_w;
            let cy = by / sub_h;

            let pred_y = predict::predict_block_16(
                &recon_y,
                bx,
                by,
                ctu,
                ctu,
                true,
                predict::IntraPredMode::Planar,
            );
            let pred_u = predict::predict_block_16(
                &recon_u,
                cx,
                cy,
                ctu_cw,
                ctu_ch,
                false,
                predict::IntraPredMode::Planar,
            );
            let pred_v = predict::predict_block_16(
                &recon_v,
                cx,
                cy,
                ctu_cw,
                ctu_ch,
                false,
                predict::IntraPredMode::Planar,
            );

            let mut res_y = vec![0i32; ctu * ctu];
            for y in 0..ctu {
                let sy = (by + y).min(frame.height as usize - 1);
                for x in 0..ctu {
                    let sx = (bx + x).min(frame.width as usize - 1);
                    res_y[y * ctu + x] =
                        i32::from(frame.y.get(sx, sy)) - i32::from(pred_y[y * ctu + x]);
                }
            }
            let mut res_u = vec![0i32; ctu_cw * ctu_ch];
            let mut res_v = vec![0i32; ctu_cw * ctu_ch];
            for y in 0..ctu_ch {
                let sy = (cy + y).min(frame.u.height - 1);
                for x in 0..ctu_cw {
                    let sx = (cx + x).min(frame.u.width - 1);
                    res_u[y * ctu_cw + x] =
                        i32::from(frame.u.get(sx, sy)) - i32::from(pred_u[y * ctu_cw + x]);
                    res_v[y * ctu_cw + x] =
                        i32::from(frame.v.get(sx, sy)) - i32::from(pred_v[y * ctu_cw + x]);
                }
            }

            if use_act {
                encode_act_flag(&mut enc, &mut ctx, true);
                let (ay, acg, aco) = apply_act_forward(&res_y, &res_u, &res_v);
                res_y = ay;
                res_u = acg;
                res_v = aco;
            }

            let coeff_y = dct::forward_transform_rect_bd(
                ctu,
                ctu,
                &res_y,
                dct::TrType::Dct2,
                dct::TrType::Dct2,
                0,
                0,
                bd,
            )
            .map_err(VvcError::Unsupported)?;
            let levels_y = dct::quantize_rect_bd(ctu, ctu, &coeff_y, base_luma_qp, bd, true)
                .map_err(VvcError::Unsupported)?;
            let dq_y = dct::dequantize_rect_bd(ctu, ctu, &levels_y, base_luma_qp, bd)
                .map_err(VvcError::Unsupported)?;
            let mut inv_y = dct::inverse_transform_rect_bd(
                ctu,
                ctu,
                &dq_y,
                dct::TrType::Dct2,
                dct::TrType::Dct2,
                0,
                0,
                bd,
            )
            .map_err(VvcError::Unsupported)?;

            let coeff_u = dct::forward_transform_rect_bd(
                ctu_cw,
                ctu_ch,
                &res_u,
                dct::TrType::Dct2,
                dct::TrType::Dct2,
                0,
                0,
                bd,
            )
            .map_err(VvcError::Unsupported)?;
            let levels_u = dct::quantize_rect_bd(ctu_cw, ctu_ch, &coeff_u, base_cb_qp, bd, true)
                .map_err(VvcError::Unsupported)?;
            let dq_u = dct::dequantize_rect_bd(ctu_cw, ctu_ch, &levels_u, base_cb_qp, bd)
                .map_err(VvcError::Unsupported)?;
            let mut inv_u = dct::inverse_transform_rect_bd(
                ctu_cw,
                ctu_ch,
                &dq_u,
                dct::TrType::Dct2,
                dct::TrType::Dct2,
                0,
                0,
                bd,
            )
            .map_err(VvcError::Unsupported)?;

            let coeff_v = dct::forward_transform_rect_bd(
                ctu_cw,
                ctu_ch,
                &res_v,
                dct::TrType::Dct2,
                dct::TrType::Dct2,
                0,
                0,
                bd,
            )
            .map_err(VvcError::Unsupported)?;
            let levels_v = dct::quantize_rect_bd(ctu_cw, ctu_ch, &coeff_v, base_cr_qp, bd, true)
                .map_err(VvcError::Unsupported)?;
            let dq_v = dct::dequantize_rect_bd(ctu_cw, ctu_ch, &levels_v, base_cr_qp, bd)
                .map_err(VvcError::Unsupported)?;
            let mut inv_v = dct::inverse_transform_rect_bd(
                ctu_cw,
                ctu_ch,
                &dq_v,
                dct::TrType::Dct2,
                dct::TrType::Dct2,
                0,
                0,
                bd,
            )
            .map_err(VvcError::Unsupported)?;

            if use_act {
                apply_act_inverse(&mut inv_y, &mut inv_u, &mut inv_v, bd);
            }

            let scan_y = transform::CoeffScan::new(ctu, ctu);
            let scan_c = transform::CoeffScan::new(ctu_cw, ctu_ch);
            let cbf_y = levels_y.iter().any(|&v| v != 0);
            let cbf_u = levels_u.iter().any(|&v| v != 0);
            let cbf_v = levels_v.iter().any(|&v| v != 0);
            enc.encode_bin(cbf_u, &mut ctx.models[cabac::ctx::QT_CBF_CB]);
            enc.encode_bin(
                cbf_v,
                &mut ctx.models[cabac::ctx::QT_CBF_CR + usize::from(cbf_u)],
            );
            if !use_act || cbf_u || cbf_v {
                enc.encode_bin(cbf_y, &mut ctx.models[cabac::ctx::QT_CBF_Y]);
            }
            if cbf_y {
                residual::encode_residual(&mut enc, &mut ctx, &scan_y, &levels_y, true);
            }
            if cbf_u {
                residual::encode_residual(&mut enc, &mut ctx, &scan_c, &levels_u, false);
            }
            if cbf_v {
                residual::encode_residual(&mut enc, &mut ctx, &scan_c, &levels_v, false);
            }

            for y in 0..ctu {
                for x in 0..ctu {
                    let v = (i32::from(pred_y[y * ctu + x]) + inv_y[y * ctu + x]).clamp(0, max_val);
                    recon_y.set(bx + x, by + y, v as u16);
                }
            }
            for y in 0..ctu_ch {
                for x in 0..ctu_cw {
                    let u_val = (i32::from(pred_u[y * ctu_cw + x]) + inv_u[y * ctu_cw + x])
                        .clamp(0, max_val);
                    let v_val = (i32::from(pred_v[y * ctu_cw + x]) + inv_v[y * ctu_cw + x])
                        .clamp(0, max_val);
                    recon_u.set(cx + x, cy + y, u_val as u16);
                    recon_v.set(cx + x, cy + y, v_val as u16);
                }
            }
        }
    }

    enc.encode_bin_trm(true);
    let cabac_bytes = enc.finish_and_align();

    let vis_w = frame.width as usize;
    let vis_h = frame.height as usize;
    let vis_cw = vis_w.div_ceil(sub_w);
    let vis_ch = vis_h.div_ceil(sub_h);
    let mut out_frame = VvcFrame::new(frame.width, frame.height, bd, cf);
    for y in 0..vis_h {
        for x in 0..vis_w {
            out_frame.y.set(x, y, recon_y.get(x, y));
        }
    }
    for y in 0..vis_ch {
        for x in 0..vis_cw {
            out_frame.u.set(x, y, recon_u.get(x, y));
            out_frame.v.set(x, y, recon_v.get(x, y));
        }
    }
    Ok((cabac_bytes, out_frame))
}

/// Rate control operating mode for [`VvcRateController`] (Wave G).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VvcRateControlMode {
    /// Constant Quantisation Parameter (`CQP`) with hierarchical picture-type offset.
    Cqp {
        /// Base slice QP (`4..=63`).
        base_qp: u8,
    },
    /// Variable Bitrate (`VBR`) targeting `target_kbps` at `fps_num / fps_den` frames/s.
    Vbr {
        /// Target bitrate in kilobits per second (`1000` bits/s units).
        target_kbps: u32,
        /// Frame rate numerator.
        fps_num: u32,
        /// Frame rate denominator.
        fps_den: u32,
        /// Initial slice QP (`4..=63`).
        init_qp: u8,
    },
}

/// Non-oscillating CQP / VBR rate controller for H.266/VVC encoding (Wave G).
#[derive(Debug, Clone)]
pub struct VvcRateController {
    mode: VvcRateControlMode,
    current_qp: u8,
    virtual_buffer_bits: i64,
    buffer_capacity_bits: i64,
    target_bits_per_pic: i64,
    frames_coded: u64,
}

impl VvcRateController {
    /// Create a new [`VvcRateController`] for `mode`.
    pub fn new(mode: VvcRateControlMode) -> Self {
        match mode {
            VvcRateControlMode::Cqp { base_qp } => Self {
                mode,
                current_qp: base_qp.clamp(4, 51),
                virtual_buffer_bits: 0,
                buffer_capacity_bits: 1,
                target_bits_per_pic: 0,
                frames_coded: 0,
            },
            VvcRateControlMode::Vbr {
                target_kbps,
                fps_num,
                fps_den,
                init_qp,
            } => {
                let target_bps = i64::from(target_kbps.max(1)) * 1000;
                let fps_n = i64::from(fps_num.max(1));
                let fps_d = i64::from(fps_den.max(1));
                let target_bits_per_pic = ((target_bps * fps_d) / fps_n).max(256);
                let buffer_capacity_bits = (target_bps * 2).max(target_bits_per_pic * 4);
                Self {
                    mode,
                    current_qp: init_qp.clamp(10, 51),
                    virtual_buffer_bits: 0,
                    buffer_capacity_bits,
                    target_bits_per_pic,
                    frames_coded: 0,
                }
            }
        }
    }

    /// Select the slice QP (`4..=51`) for the next picture of `slice_type`.
    pub fn select_picture_qp(&self, slice_type: VvcSliceType) -> u8 {
        let offset: i32 = match slice_type {
            VvcSliceType::I => -2,
            VvcSliceType::P => 0,
            VvcSliceType::B => 2,
        };
        (i32::from(self.current_qp) + offset).clamp(4, 51) as u8
    }

    /// Record the actual encoded size (`actual_bytes`) of the picture and update the
    /// virtual buffer with a damped `[-2, +2]` QP step so consecutive pictures never oscillate.
    pub fn update_after_picture(&mut self, actual_bytes: usize) {
        self.frames_coded += 1;
        if let VvcRateControlMode::Vbr { .. } = self.mode {
            let actual_bits = (actual_bytes as i64) * 8;
            self.virtual_buffer_bits = (self.virtual_buffer_bits
                + (actual_bits - self.target_bits_per_pic))
                .clamp(-self.buffer_capacity_bits, self.buffer_capacity_bits);
            // Damped proportional step bounded to [-2, +2] QP units per picture
            let ratio_num = self.virtual_buffer_bits * 4;
            let step = (ratio_num / self.buffer_capacity_bits.max(1)).clamp(-2, 2) as i32;
            self.current_qp = (i32::from(self.current_qp) + step).clamp(10, 51) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frame(width: u32, height: u32, bar_x: u32) -> Vec<u8> {
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                if x >= bar_x && x < bar_x + 8 {
                    rgba.extend_from_slice(&[240, 90, 40, 255]);
                } else {
                    let g = ((y * 4) % 200) as u8;
                    rgba.extend_from_slice(&[20, g, 80, 255]);
                }
            }
        }
        rgba
    }

    fn gradient(width: u32, height: u32) -> I420Frame {
        let cw = width.div_ceil(2) as usize;
        let ch = height.div_ceil(2) as usize;
        let mut y = vec![0u8; (width * height) as usize];
        for j in 0..height as usize {
            for i in 0..width as usize {
                y[j * width as usize + i] = ((i * 3 + j * 5) % 251) as u8;
            }
        }
        I420Frame {
            width,
            height,
            y,
            u: vec![90u8; cw * ch],
            v: vec![160u8; cw * ch],
        }
    }

    fn encoder(width: u32, height: u32, qp: u8) -> VvcEncoder {
        VvcEncoder::new(VvcEncoderConfig { width, height, qp }).expect("create VVC encoder")
    }

    #[test]
    fn emits_a_conformant_slice_structure_and_round_trips_losslessly() {
        let mut enc = encoder(96, 64, 4); // QP 4: transform-skip lossless
        assert_eq!(enc.coded_size(), (96, 64)); // already a multiple of 32
        assert_eq!(enc.visible_size(), (96, 64));

        let source = test_frame(96, 64, 24);
        let au = enc.encode_rgba(&source).expect("encode frame");
        assert!(au.keyframe);
        assert_eq!(enc.frames_sent(), 1);

        // Real VVC syntax, not the old private `KVSL` payload.
        assert!(!au.data.windows(4).any(|w| w == b"KVSL"));
        assert!(!au.data.windows(4).any(|w| w == b"KVVC"));

        let nals = parse_annexb_nals(&au.data).expect("parse access unit");
        let types: Vec<VvcNalType> = nals.iter().map(|n| n.nal_type).collect();
        assert_eq!(
            types,
            vec![
                VvcNalType::Aud,
                VvcNalType::Vps,
                VvcNalType::Sps,
                VvcNalType::Pps,
                VvcNalType::Ph,
                VvcNalType::IdrNLp,
            ]
        );
        for nal in &nals {
            assert_eq!(nal.layer_id, 0);
            assert_eq!(nal.temporal_id, 0);
        }
        assert_eq!(nals[0].rbsp, vec![0x88]);
        let sps = parse_sps_info(&nals[2].rbsp).expect("parse SPS");
        assert_eq!((sps.width, sps.height), (96, 64));
        assert_eq!((sps.coded_width, sps.coded_height), (96, 64));
        assert_eq!(sps.ctu_log2, 5);

        let decoded = decode_annexb_to_i420(&[au.data.as_slice()]).expect("decode");
        assert_eq!(decoded.len(), 1);
        assert_eq!((decoded[0].width, decoded[0].height), (96, 64));
        let expected = crate::color::rgba_to_i420(&source, 96, 64);
        assert_eq!(decoded[0].y, expected.y);
        assert_eq!(decoded[0].u, expected.u);
        assert_eq!(decoded[0].v, expected.v);

        enc.finish();
        assert_eq!(enc.frames_sent(), 1);
    }

    #[test]
    fn conformance_window_handles_non_multiple_of_ctu_sizes() {
        let mut enc = encoder(100, 60, 4);
        assert_eq!(enc.coded_size(), (128, 64));
        assert_eq!(enc.visible_size(), (100, 60));
        let source = test_frame(100, 60, 10);
        let au = enc.encode_rgba(&source).expect("encode frame");
        let decoded = decode_annexb_to_i420(&[au.data.as_slice()]).expect("decode");
        assert_eq!((decoded[0].width, decoded[0].height), (100, 60));
        let expected = crate::color::rgba_to_i420(&source, 100, 60);
        assert_eq!(decoded[0].y.len(), expected.y.len());
        assert_eq!(decoded[0].y, expected.y);
        assert_eq!(decoded[0].u, expected.u);
        assert_eq!(decoded[0].v, expected.v);
    }

    #[test]
    fn three_frame_encode_decode_roundtrip_is_lossless_at_qp4() {
        let mut enc = encoder(64, 64, 4);
        let mut units = Vec::new();
        for index in 0..3u32 {
            let frame = gradient(64, 64);
            let mut shifted = frame.clone();
            for (i, sample) in shifted.y.iter_mut().enumerate() {
                *sample = sample
                    .wrapping_add((index * 7) as u8)
                    .wrapping_add(i as u8 % 3);
            }
            let au = enc.encode_i420(&shifted).expect("encode i420 frame");
            assert!(au.keyframe);
            // Encoding is deterministic: a second encoder produces the same
            // bytes for the same input.
            let mut other = encoder(64, 64, 4);
            assert_eq!(
                au.data,
                other.encode_i420(&shifted).expect("re-encode").data
            );
            units.push((au, shifted));
        }
        assert_eq!(enc.frames_sent(), 3);
        let slices: Vec<&[u8]> = units.iter().map(|(au, _)| au.data.as_slice()).collect();
        let decoded = decode_annexb_to_i420(&slices).expect("decode");
        assert_eq!(decoded.len(), 3);
        for (frame, (_, source)) in decoded.iter().zip(units.iter()) {
            assert_eq!(frame.y, source.y);
            assert_eq!(frame.u, source.u);
            assert_eq!(frame.v, source.v);
        }
    }

    #[test]
    fn every_access_unit_is_a_keyframe() {
        let mut enc = encoder(64, 64, 26);
        let mut units = Vec::new();
        for i in 0..3u32 {
            let au = enc.encode_rgba(&test_frame(64, 64, i * 8)).expect("encode");
            assert!(au.keyframe);
            units.push(au);
        }
        assert_eq!(units.len(), 3);
        assert!(units.iter().all(|u| u.keyframe));
        let slices: Vec<&[u8]> = units.iter().map(|u| u.data.as_slice()).collect();
        let decoded = decode_annexb_to_i420(&slices).expect("decode");
        assert_eq!(decoded.len(), 3);
    }

    #[test]
    fn rejects_zero_and_odd_dimensions() {
        for (width, height) in [(0, 48), (48, 0), (0, 0)] {
            let err = VvcEncoder::new(VvcEncoderConfig {
                width,
                height,
                qp: 26,
            })
            .expect_err("zero dimension must be rejected");
            assert!(matches!(err, VvcError::InvalidConfig(_)), "{err}");
        }
        for (width, height) in [(33, 48), (48, 33), (99, 61)] {
            let err = VvcEncoder::new(VvcEncoderConfig {
                width,
                height,
                qp: 26,
            })
            .expect_err("odd dimension must be rejected");
            assert!(matches!(err, VvcError::InvalidConfig(_)), "{err}");
        }
    }

    #[test]
    fn rejects_oversized_dimensions() {
        let err = VvcEncoder::new(VvcEncoderConfig {
            width: 16_386,
            height: 2,
            qp: 26,
        })
        .expect_err("dimension above 16384 must be rejected");
        assert!(matches!(err, VvcError::LimitExceeded(_)), "{err}");

        let err = VvcEncoder::new(VvcEncoderConfig {
            width: 16_384,
            height: 16_384,
            qp: 26,
        })
        .expect_err("luma samples above MaxLumaPs must be rejected");
        assert!(matches!(err, VvcError::LimitExceeded(_)), "{err}");
    }

    #[test]
    fn rejects_qp_out_of_range() {
        for qp in [0u8, 1, 3, 64, 100, 255] {
            let err = VvcEncoder::new(VvcEncoderConfig {
                width: 32,
                height: 32,
                qp,
            })
            .expect_err("qp outside 4..=63 must be rejected");
            assert!(matches!(err, VvcError::InvalidConfig(_)), "{err}");
        }
        assert!(
            VvcEncoder::new(VvcEncoderConfig {
                width: 32,
                height: 32,
                qp: 4
            })
            .is_ok()
        );
        assert!(
            VvcEncoder::new(VvcEncoderConfig {
                width: 32,
                height: 32,
                qp: 63
            })
            .is_ok()
        );
    }

    #[test]
    fn rejects_wrong_i420_plane_sizes() {
        let mut enc = encoder(32, 32, 26);
        let mut frame = gradient(32, 32);
        frame.y.pop();
        let err = enc.encode_i420(&frame).expect_err("short Y plane");
        assert!(
            matches!(err, VvcError::PlaneSize { plane: "Y", .. }),
            "{err}"
        );

        let mut frame = gradient(32, 32);
        frame.u.push(0);
        let err = enc.encode_i420(&frame).expect_err("long U plane");
        assert!(
            matches!(err, VvcError::PlaneSize { plane: "U", .. }),
            "{err}"
        );

        let mut frame = gradient(32, 32);
        frame.v.clear();
        let err = enc.encode_i420(&frame).expect_err("empty V plane");
        assert!(
            matches!(err, VvcError::PlaneSize { plane: "V", .. }),
            "{err}"
        );

        // A structurally valid frame of the wrong size is a dimension error.
        let err = enc
            .encode_i420(&gradient(48, 32))
            .expect_err("wrong dimensions");
        assert!(
            matches!(
                err,
                VvcError::FrameDimensions {
                    width: 48,
                    height: 32,
                    expected_width: 32,
                    expected_height: 32
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn rejects_wrong_rgba_buffer_size() {
        let mut enc = encoder(32, 32, 26);
        assert!(enc.encode_rgba(&[0u8; 12]).is_err());
        assert!(enc.encode_rgba(&[]).is_err());
        // One byte too many.
        let mut too_big = test_frame(32, 32, 0);
        too_big.push(0);
        assert!(enc.encode_rgba(&too_big).is_err());
        // One byte too few.
        let mut too_small = test_frame(32, 32, 0);
        too_small.pop();
        assert!(enc.encode_rgba(&too_small).is_err());
    }

    #[test]
    fn finish_marks_the_encoder_drained_and_rejects_more_frames() {
        let mut enc = encoder(32, 32, 26);
        assert!(!enc.is_finished());
        enc.finish();
        assert!(enc.is_finished());
        enc.finish(); // idempotent, produces no output
        assert_eq!(enc.frames_sent(), 0);
        assert!(matches!(
            enc.encode_rgba(&test_frame(32, 32, 0)),
            Err(VvcError::EncoderFinished)
        ));
        assert!(matches!(
            enc.encode_i420(&gradient(32, 32)),
            Err(VvcError::EncoderFinished)
        ));
    }

    #[test]
    fn decoder_rejects_streams_without_parameter_sets() {
        let mut enc = encoder(32, 32, 26);
        let au = enc.encode_rgba(&test_frame(32, 32, 0)).expect("encode");
        let nals = parse_annexb_nals(&au.data).expect("parse");
        let slice = nals
            .iter()
            .find(|nal| nal.nal_type == VvcNalType::IdrNLp)
            .expect("slice NAL");
        let mut isolated = vec![0x00, 0x00, 0x00, 0x01];
        isolated.extend_from_slice(&slice.raw);
        let err = decode_annexb_to_i420(&[isolated.as_slice()])
            .expect_err("slice without SPS/PPS must be rejected");
        assert!(matches!(err, VvcError::Bitstream(_)), "{err}");
    }

    #[test]
    fn decoder_decodes_non_mpm_and_angular_mpm_intra_modes() {
        let mut enc = cabac::BinEncoder::new();
        let mut ctx = cabac::CtxStore::new_i_slice(26);
        enc.encode_bin(true, &mut ctx.models[cabac::ctx::INTRA_LUMA_MPM_FLAG]);
        enc.encode_bin(
            true,
            &mut ctx.models[cabac::ctx::INTRA_LUMA_PLANAR_FLAG + 1],
        );
        enc.encode_bin_ep(true);
        enc.encode_bin_ep(false);
        enc.encode_bin(false, &mut ctx.models[cabac::ctx::INTRA_CHROMA_PRED_MODE]);
        enc.encode_bin(false, &mut ctx.models[cabac::ctx::QT_CBF_CB]);
        enc.encode_bin(false, &mut ctx.models[cabac::ctx::QT_CBF_CR]);
        enc.encode_bin(false, &mut ctx.models[cabac::ctx::QT_CBF_Y]);
        enc.encode_bin_trm(true);
        let bytes = enc.finish_and_align();
        let frame = slice::decode_picture(&bytes, 26, 5, 32, 32)
            .expect("angular MPM index 2 decodes cleanly");
        assert_eq!((frame.width, frame.height), (32, 32));
    }

    #[test]
    fn malformed_input_does_not_panic() {
        let mut enc = encoder(64, 64, 26);
        let au = enc.encode_rgba(&test_frame(64, 64, 0)).expect("encode");

        // Every truncation of a valid access unit.
        for len in 0..=au.data.len() {
            let truncated = &au.data[..len];
            let _ = parse_annexb_nals(truncated);
            let _ = decode_annexb_to_i420(&[truncated]);
        }

        // Deterministic pseudo-random bytes, including start-code-like runs.
        let mut seed = 0x1234_5678u32;
        for case in 0..64 {
            let len = (case * 37) % 4096;
            let mut data = Vec::with_capacity(len + 3);
            for _ in 0..len {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let byte = (seed >> 24) as u8;
                data.push(byte);
            }
            let _ = parse_annexb_nals(&data);
            let _ = decode_annexb_to_i420(&[data.as_slice()]);
        }

        // Start-code floods and degenerate NAL units.
        for data in [
            vec![],
            vec![0x00],
            vec![0x00, 0x00, 0x01],
            vec![0x00, 0x00, 0x00, 0x01],
            vec![0x00; 1024],
            vec![0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xFF, 0xFF],
        ] {
            let _ = parse_annexb_nals(&data);
            let _ = decode_annexb_to_i420(&[data.as_slice()]);
        }

        // A valid SPS followed by a garbage slice must not panic either.
        let nals = parse_annexb_nals(&au.data).expect("parse");
        let mut mutated = au.data.clone();
        if let Some(last) = nals.last() {
            let offset = mutated.len() - last.raw.len() - 4;
            for byte in &mut mutated[offset..] {
                *byte = 0xFF;
            }
        }
        let _ = decode_annexb_to_i420(&[mutated.as_slice()]);
    }

    #[test]
    fn nal_size_and_count_limits_are_documented_and_enforced() {
        assert_eq!(MAX_NAL_BYTES, 64 * 1024 * 1024);
        assert_eq!(MAX_ACCESS_UNIT_BYTES, 256 * 1024 * 1024);
        assert_eq!(MAX_LUMA_SAMPLES, 35_651_584);
        assert_eq!(MAX_DIMENSION, 16_384);
        // A start-code flood produces many tiny NAL units; the count cap keeps
        // the allocation bounded instead of panicking.
        let mut many = Vec::new();
        for _ in 0..(MAX_NALS_PER_ACCESS_UNIT + 8) {
            many.extend_from_slice(&[0x00, 0x00, 0x01, 0x00, 0x08]);
        }
        let err = parse_annexb_nals(&many).expect_err("NAL count limit");
        assert!(matches!(err, VvcError::LimitExceeded(_)), "{err}");
    }

    #[test]
    fn gdr_is_an_irap_nal_type() {
        assert_eq!(VvcNalType::from_u8(10), VvcNalType::Gdr);
        assert_eq!(VvcNalType::Gdr.code(), 10);
        assert!(VvcNalType::Gdr.is_irap());
        for nal_type in [VvcNalType::IdrWRadl, VvcNalType::IdrNLp, VvcNalType::Cra] {
            assert!(nal_type.is_irap(), "{nal_type:?}");
        }
        for nal_type in [
            VvcNalType::Trail,
            VvcNalType::Stsa,
            VvcNalType::Radl,
            VvcNalType::Rasl,
            VvcNalType::Ph,
        ] {
            assert!(!nal_type.is_irap(), "{nal_type:?}");
        }
    }

    #[test]
    fn non_empty_input_without_a_nal_is_rejected() {
        assert!(
            parse_annexb_nals(&[])
                .expect("empty input is not an error")
                .is_empty()
        );

        for data in [&[0xAAu8, 0xBB][..], &[0x00, 0x00, 0x02, 0x10][..]] {
            let err = parse_annexb_nals(data).expect_err("no start code");
            assert!(matches!(err, VvcError::Bitstream(_)), "{err}");
        }

        // A start code whose payload is shorter than the 2-byte VVC NAL
        // header yields no NAL unit and is therefore malformed as well.
        let err = parse_annexb_nals(&[0x00, 0x00, 0x01, 0x00]).expect_err("short NAL");
        assert!(matches!(err, VvcError::Bitstream(_)), "{err}");

        // A well-formed NAL still parses.
        let nals = parse_annexb_nals(&[0x00, 0x00, 0x01, 0x00, 0x78]).expect("one NAL");
        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0].nal_type, VvcNalType::Sps);
    }

    fn sps_info(ctu_log2: u8, coded: (u32, u32)) -> VvcSpsInfo {
        VvcSpsInfo {
            sps_id: 0,
            vps_id: 1,
            chroma_format_idc: 1,
            bit_depth_luma: 8,
            width: coded.0,
            height: coded.1,
            coded_width: coded.0,
            coded_height: coded.1,
            ctu_log2,
            profile_idc: 1,
            tier_flag: false,
            level_idc: 16,
            frame_only_constraint: true,
            multilayer_enabled: false,
            ref_pic_resampling: false,
            res_change_in_clvs_allowed: false,
            gci_bits: Vec::new(),
            num_sub_profiles: 0,
            sub_profile_idc: Vec::new(),
        }
    }

    #[test]
    fn decoded_sps_rejects_ctu_sizes_other_than_32_and_partial_ctus() {
        let err = validate_decoded_sps(&sps_info(3, (64, 64)))
            .expect_err("8x8 CTUs are outside 32..=128");
        assert!(matches!(err, VvcError::Unsupported(_)), "{err}");

        let err = validate_decoded_sps(&sps_info(8, (256, 256)))
            .expect_err("256x256 CTUs are outside 32..=128");
        assert!(matches!(err, VvcError::Unsupported(_)), "{err}");

        let err = validate_decoded_sps(&sps_info(5, (48, 48)))
            .expect_err("partial CTUs are legal but not implemented");
        assert!(matches!(err, VvcError::Unsupported(_)), "{err}");

        validate_decoded_sps(&sps_info(5, (32, 32))).expect("32x32 CTU, CTU-aligned");
        validate_decoded_sps(&sps_info(5, (96, 64))).expect("32x32 CTU, CTU-aligned");
        validate_decoded_sps(&sps_info(6, (64, 64))).expect("64x64 CTU, CTU-aligned");
        validate_decoded_sps(&sps_info(7, (128, 128))).expect("128x128 CTU, CTU-aligned");
    }

    /// Build the prefix of this encoder's PPS layout, choosing
    /// `pps_subpic_id_mapping_present_flag`.
    fn pps_with_subpic_mapping(subpic_mapping: bool) -> Vec<u8> {
        let mut bw = bits::BitWriter::with_capacity(8);
        bw.write_bits(0, 6); // pps_pic_parameter_set_id
        bw.write_bits(0, 4); // pps_seq_parameter_set_id
        bw.write_bit(false); // pps_mixed_nalu_types_in_pic_flag
        bw.write_ue(32); // pps_pic_width_in_luma_samples
        bw.write_ue(32); // pps_pic_height_in_luma_samples
        bw.write_bit(false); // pps_conformance_window_flag
        bw.write_bit(false); // pps_scaling_window_explicit_signalling_flag
        bw.write_bit(false); // pps_output_flag_present_flag
        bw.write_bit(true); // pps_no_pic_partition_flag
        bw.write_bit(subpic_mapping); // pps_subpic_id_mapping_present_flag
        if subpic_mapping {
            // `pps_no_pic_partition_flag` is 1, so `pps_num_subpics_minus1` is absent and
            // inferred to 0 (H.266 §7.3.2.4); one subpicture ID of length 1 is written.
            bw.write_ue(0); // pps_subpic_id_len_minus1
            bw.write_bits(0, 1); // pps_subpic_id[0]
        }
        bw.write_bit(false); // pps_cabac_init_present_flag
        bw.write_ue(0); // pps_num_ref_idx_default_active_minus1[0]
        bw.write_ue(0); // pps_num_ref_idx_default_active_minus1[1]
        bw.write_bit(false); // pps_rpl1_idx_present_flag
        bw.write_bit(false); // pps_weighted_pred_flag
        bw.write_bit(false); // pps_weighted_bipred_flag
        bw.write_bit(false); // pps_ref_wraparound_enabled_flag
        bw.write_se(0); // pps_init_qp_minus26
        bw.rbsp_trailing_bits();
        bw.into_bytes()
    }

    #[test]
    fn pps_subpic_id_mapping_is_parsed() {
        assert_eq!(
            parse_pps_info(&pps_with_subpic_mapping(false))
                .expect("own PPS layout")
                .init_qp_minus26,
            0
        );
        // A PPS with subpic_mapping=true and 1 subpic (id_len=1, id=0) parses cleanly.
        assert!(parse_pps_info(&pps_with_subpic_mapping(true)).is_ok());
    }

    /// Build a slice header prefix for this encoder's layout.
    fn slice_header_bits(ph_in_sh: bool, ts_disabled: bool, qp_delta: i32) -> Vec<u8> {
        let mut bw = bits::BitWriter::with_capacity(8);
        bw.write_bit(ph_in_sh); // sh_picture_header_in_slice_header_flag
        if !ph_in_sh {
            bw.write_bit(false); // sh_no_output_of_prior_pics_flag
            bw.write_se(qp_delta); // sh_qp_delta
            bw.write_bit(ts_disabled); // sh_ts_residual_coding_disabled_flag
        }
        bw.byte_alignment();
        bw.into_bytes()
    }

    #[test]
    fn slice_header_alternatives_are_unsupported() {
        let (qp_delta, offset) = parse_our_slice_header(
            &slice_header_bits(false, false, -3),
            &SliceCodingOptions::default(),
            false,
        )
        .expect("own slice header");
        assert_eq!(qp_delta, -3);
        assert_eq!(offset, 1);

        let (qp_d2, off2) = parse_our_slice_header(
            &slice_header_bits(false, true, 0),
            &SliceCodingOptions::default(),
            false,
        )
        .expect("sh_ts_residual_coding_disabled_flag == 1 is supported");
        assert_eq!((qp_d2, off2), (0, 1));
    }

    fn stream_config(width: u32, height: u32, qp: u8) -> hls::VvcStreamConfig {
        hls::VvcStreamConfig {
            width,
            height,
            crop_right: 0,
            crop_bottom: 0,
            qp,
            level_idc: 16,
            ctu_log2: 5,
            min_cb_log2: 5,
            ts_log2: 5,
        }
    }

    /// Build this encoder's PPS layout with explicit identifiers.
    fn pps_rbsp(pps_id: u32, sps_id: u32, qp: u8) -> Vec<u8> {
        let mut bw = bits::BitWriter::with_capacity(24);
        bw.write_bits(pps_id, 6); // pps_pic_parameter_set_id
        bw.write_bits(sps_id, 4); // pps_seq_parameter_set_id
        bw.write_bit(false); // pps_mixed_nalu_types_in_pic_flag
        bw.write_ue(32); // pps_pic_width_in_luma_samples
        bw.write_ue(32); // pps_pic_height_in_luma_samples
        bw.write_bit(false); // pps_conformance_window_flag
        bw.write_bit(false); // pps_scaling_window_explicit_signalling_flag
        bw.write_bit(false); // pps_output_flag_present_flag
        bw.write_bit(true); // pps_no_pic_partition_flag
        bw.write_bit(false); // pps_subpic_id_mapping_present_flag
        bw.write_bit(false); // pps_cabac_init_present_flag
        bw.write_ue(0); // pps_num_ref_idx_default_active_minus1[0]
        bw.write_ue(0); // pps_num_ref_idx_default_active_minus1[1]
        bw.write_bit(false); // pps_rpl1_idx_present_flag
        bw.write_bit(false); // pps_weighted_pred_flag
        bw.write_bit(false); // pps_weighted_bipred_flag
        bw.write_bit(false); // pps_ref_wraparound_enabled_flag
        bw.write_se(i32::from(qp) - 26); // pps_init_qp_minus26
        bw.write_bit(false); // pps_cu_qp_delta_enabled_flag
        bw.write_bit(false); // pps_chroma_tool_offsets_present_flag
        bw.write_bit(true); // pps_deblocking_filter_control_present_flag
        bw.write_bit(false); // pps_deblocking_filter_override_enabled_flag
        bw.write_bit(true); // pps_deblocking_filter_disabled_flag
        bw.write_bit(false); // pps_picture_header_extension_present_flag
        bw.write_bit(false); // pps_slice_header_extension_present_flag
        bw.write_bit(false); // pps_extension_flag
        bw.rbsp_trailing_bits();
        bw.into_bytes()
    }

    /// Build this encoder's PH layout with explicit fields.
    fn ph_rbsp(pps_id: u32, poc_lsb: u32, inter_slice_allowed: bool) -> Vec<u8> {
        let mut bw = bits::BitWriter::with_capacity(4);
        bw.write_bit(true); // ph_gdr_or_irap_pic_flag
        bw.write_bit(false); // ph_non_ref_pic_flag
        bw.write_bit(false); // ph_gdr_pic_flag
        bw.write_bit(inter_slice_allowed); // ph_inter_slice_allowed_flag
        bw.write_ue(pps_id); // ph_pic_parameter_set_id
        bw.write_bits(poc_lsb, 8); // ph_pic_order_cnt_lsb
        bw.rbsp_trailing_bits();
        bw.into_bytes()
    }

    /// A complete raw NAL unit (header plus escaped payload).
    fn raw_nal(nal_type: u8, layer_id: u8, temporal_id_plus1: u8, rbsp: &[u8]) -> Vec<u8> {
        let mut framed = Vec::new();
        hls::push_nal(&mut framed, nal_type, layer_id, temporal_id_plus1, rbsp);
        framed.split_off(4)
    }

    fn annexb(raws: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        for raw in raws {
            out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            out.extend_from_slice(raw);
        }
        out
    }

    fn raw_nal_type(raw: &[u8]) -> VvcNalType {
        VvcNalType::from_u8((raw[1] >> 3) & 0x1F)
    }

    fn replace_raw(raws: &mut [Vec<u8>], nal_type: VvcNalType, replacement: Vec<u8>) {
        let index = raws
            .iter()
            .position(|raw| raw_nal_type(raw) == nal_type)
            .expect("NAL unit of the requested type");
        raws[index] = replacement;
    }

    fn encoded_raw_nals(width: u32, height: u32, qp: u8) -> Vec<Vec<u8>> {
        let mut enc = encoder(width, height, qp);
        let au = enc.encode_i420(&gradient(width, height)).expect("encode");
        parse_annexb_nals(&au.data)
            .expect("parse")
            .into_iter()
            .map(|nal| nal.raw)
            .collect()
    }

    #[test]
    fn pps_must_reference_a_known_sps() {
        let mut raws = encoded_raw_nals(32, 32, 26);
        replace_raw(
            &mut raws,
            VvcNalType::Pps,
            raw_nal(hls::NAL_PPS, 0, 1, &pps_rbsp(0, 1, 26)),
        );
        let stream = annexb(&raws);
        let err = decode_annexb_to_i420(&[stream.as_slice()])
            .expect_err("PPS referencing an unknown SPS must fail");
        assert!(matches!(err, VvcError::Bitstream(_)), "{err}");
    }

    #[test]
    fn ph_must_reference_a_known_pps() {
        let mut raws = encoded_raw_nals(32, 32, 26);
        replace_raw(
            &mut raws,
            VvcNalType::Ph,
            raw_nal(hls::NAL_PH, 0, 1, &ph_rbsp(1, 0, false)),
        );
        let stream = annexb(&raws);
        let err = decode_annexb_to_i420(&[stream.as_slice()])
            .expect_err("PH referencing an unknown PPS must fail");
        assert!(matches!(err, VvcError::Bitstream(_)), "{err}");
    }

    #[test]
    fn ph_rejects_inter_slices_and_non_zero_poc() {
        // Inter slices are outside the intra-only subset.
        let mut raws = encoded_raw_nals(32, 32, 26);
        replace_raw(
            &mut raws,
            VvcNalType::Ph,
            raw_nal(hls::NAL_PH, 0, 1, &ph_rbsp(0, 0, true)),
        );
        let stream = annexb(&raws);
        let err = decode_annexb_to_i420(&[stream.as_slice()])
            .expect_err("ph_inter_slice_allowed_flag = 1 must be unsupported");
        assert!(
            matches!(
                err,
                VvcError::Unsupported("picture header: inter slices allowed")
            ),
            "{err}"
        );

        // Every coded picture is an IDR, so POC must restart at zero.
        let err = parse_ph_info(&ph_rbsp(0, 1, false)).expect_err("non-zero POC LSB");
        assert!(matches!(err, VvcError::Bitstream(_)), "{err}");
        assert!(parse_ph_info(&ph_rbsp(0, 0, false)).is_ok());
    }

    #[test]
    fn vcl_without_ph_is_a_bitstream_error() {
        let raws: Vec<Vec<u8>> = encoded_raw_nals(32, 32, 26)
            .into_iter()
            .filter(|raw| raw_nal_type(raw) != VvcNalType::Ph)
            .collect();
        let stream = annexb(&raws);
        let err = decode_annexb_to_i420(&[stream.as_slice()])
            .expect_err("a VCL NAL without a picture header must fail");
        assert!(matches!(err, VvcError::Bitstream(_)), "{err}");
    }

    #[test]
    fn picture_header_inside_the_slice_header_decodes() {
        // VTM writes `ph_in_slice_header_flag == 1` by default and omits the
        // PH NAL unit (`EncGOP.cpp:4456`). Rewrite this encoder's access unit
        // that way and check that the stateful decoder produces the same frame.
        let mut enc = encoder(64, 64, 26);
        let au = enc.encode_i420(&gradient(64, 64)).expect("encode");
        let nals = parse_annexb_nals(&au.data).expect("parse");

        let find = |nal_type: VvcNalType| {
            nals.iter()
                .find(|nal| nal.nal_type == nal_type)
                .expect("NAL unit")
                .rbsp
                .clone()
        };
        let mut sps_table: [Option<hls::VvcSps>; 16] = std::array::from_fn(|_| None);
        sps_table[0] = Some(hls::parse_full_sps_rbsp(&find(VvcNalType::Sps)).expect("SPS"));
        let mut pps_table: [Option<hls::VvcPps>; 64] = std::array::from_fn(|_| None);
        pps_table[0] = Some(hls::parse_full_pps_rbsp(&find(VvcNalType::Pps)).expect("PPS"));
        let ph = hls::parse_full_ph_rbsp(&find(VvcNalType::Ph), &sps_table, &pps_table)
            .expect("picture header");
        let sps = sps_table[0].clone().expect("SPS");
        let pps = pps_table[0].clone().expect("PPS");
        let slice_rbsp = find(VvcNalType::IdrNLp);
        let (mut sh, data_off) = hls::parse_full_slice_header(
            &slice_rbsp,
            VvcNalType::IdrNLp.code(),
            Some(&ph),
            &sps_table,
            &pps_table,
        )
        .expect("slice header");
        sh.ph_in_slice_header = true;
        sh.embedded_ph = Some(ph.clone());
        let mut new_slice =
            hls::build_full_slice_header_bits(&sh, VvcNalType::IdrNLp.code(), &ph, &sps, &pps);
        new_slice.extend_from_slice(&slice_rbsp[data_off..]);

        let mut raws: Vec<Vec<u8>> = nals
            .iter()
            .filter(|nal| nal.nal_type != VvcNalType::Ph)
            .map(|nal| nal.raw.clone())
            .collect();
        let index = raws
            .iter()
            .position(|raw| raw_nal_type(raw) == VvcNalType::IdrNLp)
            .expect("slice NAL");
        raws[index] = raw_nal(hls::NAL_IDR_N_LP, 0, 1, &new_slice);
        let stream = annexb(&raws);

        let mut dec = VvcDecoder::new();
        let embedded = dec
            .decode_annexb_stream_vvc_frames(&[stream.as_slice()])
            .expect("decode PH-in-slice-header stream");
        let mut reference = VvcDecoder::new();
        let standalone = reference
            .decode_annexb_stream_vvc_frames(&[au.data.as_slice()])
            .expect("decode PH-NAL stream");
        assert_eq!(embedded, standalone);
    }

    #[test]
    fn second_vcl_in_one_access_unit_is_unsupported() {
        let mut raws = encoded_raw_nals(32, 32, 26);
        let index = raws
            .iter()
            .position(|raw| raw_nal_type(raw) == VvcNalType::IdrNLp)
            .expect("slice NAL");
        let slice = raws[index].clone();
        raws.insert(index + 1, slice);
        let stream = annexb(&raws);
        let err = decode_annexb_to_i420(&[stream.as_slice()])
            .expect_err("a second slice in one access unit must be unsupported");
        assert!(
            matches!(
                err,
                VvcError::Unsupported("multiple slices per access unit")
            ),
            "{err}"
        );
    }

    #[test]
    fn vcl_layer_and_temporal_id_must_be_zero() {
        for (layer_id, temporal_id_plus1) in [(1u8, 1u8), (0, 2), (0, 0)] {
            let mut raws = encoded_raw_nals(32, 32, 26);
            let index = raws
                .iter()
                .position(|raw| raw_nal_type(raw) == VvcNalType::IdrNLp)
                .expect("slice NAL");
            raws[index][0] = (raws[index][0] & 0xC0) | layer_id;
            raws[index][1] = (raws[index][1] & 0xF8) | temporal_id_plus1;
            let stream = annexb(&raws);
            let err = decode_annexb_to_i420(&[stream.as_slice()])
                .expect_err("only layer 0 at temporal sub-layer 0 is part of the subset");
            assert!(matches!(err, VvcError::Unsupported(_)), "{err}");
        }
    }

    #[test]
    fn repeated_parameter_sets_must_be_identical() {
        // Identical SPS and PPS repeats are accepted.
        let mut raws = encoded_raw_nals(32, 32, 26);
        let sps_index = raws
            .iter()
            .position(|raw| raw_nal_type(raw) == VvcNalType::Sps)
            .expect("SPS NAL");
        let sps = raws[sps_index].clone();
        raws.insert(sps_index + 1, sps);
        let pps_index = raws
            .iter()
            .position(|raw| raw_nal_type(raw) == VvcNalType::Pps)
            .expect("PPS NAL");
        let pps = raws[pps_index].clone();
        raws.insert(pps_index + 1, pps);
        let stream = annexb(&raws);
        let decoded = decode_annexb_to_i420(&[stream.as_slice()])
            .expect("identical repeated parameter sets must decode");
        assert_eq!(decoded.len(), 1);

        // A conflicting SPS with the same ID is rejected.
        let mut raws = encoded_raw_nals(32, 32, 26);
        let sps_index = raws
            .iter()
            .position(|raw| raw_nal_type(raw) == VvcNalType::Sps)
            .expect("SPS NAL");
        let conflicting_sps = hls::build_sps_rbsp(&stream_config(64, 64, 26));
        raws.insert(sps_index + 1, raw_nal(hls::NAL_SPS, 0, 1, &conflicting_sps));
        let stream = annexb(&raws);
        let err = decode_annexb_to_i420(&[stream.as_slice()])
            .expect_err("conflicting duplicate SPS must fail");
        assert!(matches!(err, VvcError::Bitstream(_)), "{err}");

        // A conflicting PPS with the same ID is rejected.
        let mut raws = encoded_raw_nals(32, 32, 26);
        let pps_index = raws
            .iter()
            .position(|raw| raw_nal_type(raw) == VvcNalType::Pps)
            .expect("PPS NAL");
        let conflicting_pps = hls::build_pps_rbsp(&stream_config(32, 32, 40));
        raws.insert(pps_index + 1, raw_nal(hls::NAL_PPS, 0, 1, &conflicting_pps));
        let stream = annexb(&raws);
        let err = decode_annexb_to_i420(&[stream.as_slice()])
            .expect_err("conflicting duplicate PPS must fail");
        assert!(matches!(err, VvcError::Bitstream(_)), "{err}");
    }

    #[test]
    fn multi_tool_encoder_and_decoder_round_trip_gop_sequence() {
        let f0 = gradient(64, 64);
        let mut f1 = f0.clone();
        // Shift luma pattern slightly to exercise inter motion compensation and residual coding.
        for v in &mut f1.y {
            *v = v.saturating_add(2);
        }

        let opts = VvcEncoderOptions {
            config: VvcEncoderConfig {
                width: 64,
                height: 64,
                qp: 24,
            },
            keyint: 4,
            slice_options: SliceCodingOptions {
                full_intra_modes: true,
                enable_cclm: true,
                enable_mip: false,
                dep_quant: true,
                sign_data_hiding: false,
                enable_jccr: false,
                enable_deblock: true,
                enable_sao: false,
                enable_alf: false,
                ..SliceCodingOptions::default()
            },
            ..VvcEncoderOptions::default()
        };
        let mut enc = VvcMultiToolEncoder::new(opts).expect("multi-tool encoder");
        let au0 = enc.encode_i420(&f0).expect("AU 0 (IDR)");
        assert!(au0.keyframe);
        let au1 = enc.encode_i420(&f1).expect("AU 1 (TRAIL P)");
        assert!(!au1.keyframe);

        let mut dec = VvcDecoder::with_options(opts.slice_options);
        let decoded = dec
            .decode_annexb_stream(&[au0.data.as_slice(), au1.data.as_slice()])
            .expect("decode GOP");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].width, 64);
        assert_eq!(decoded[1].width, 64);
    }

    #[test]
    fn inter_p_slice_conformance_cases_qp4_qp10_qp40_round_trip() {
        for &(w, h, qp, dep_quant) in &[
            (64u32, 64u32, 4u8, false),
            (64, 64, 4, true),
            (64, 64, 10, false),
            (64, 64, 10, true),
            (200, 120, 40, false),
            (200, 120, 40, true),
        ] {
            let opts = VvcEncoderOptions {
                config: VvcEncoderConfig {
                    width: w,
                    height: h,
                    qp,
                },
                keyint: 4,
                slice_options: SliceCodingOptions {
                    full_intra_modes: true,
                    enable_cclm: true,
                    enable_mip: true,
                    dep_quant,
                    sign_data_hiding: !dep_quant,
                    enable_jccr: true,
                    enable_deblock: true,
                    enable_sao: true,
                    enable_alf: true,
                    ..SliceCodingOptions::default()
                },
                ..VvcEncoderOptions::default()
            };
            let mut enc = VvcMultiToolEncoder::new(opts).expect("multi-tool encoder");
            let mut aus = Vec::new();
            for idx in 0..3u32 {
                let mut f = gradient(w, h);
                for v in &mut f.y {
                    *v = v.wrapping_add((idx * 11) as u8);
                }
                aus.push(enc.encode_i420(&f).expect("encode AU").data);
            }
            let refs: Vec<&[u8]> = aus.iter().map(|a| a.as_slice()).collect();
            let mut dec = VvcDecoder::new();
            let decoded = dec.decode_annexb_stream(&refs).expect("decode 3-frame GOP");
            assert_eq!(decoded.len(), 3);
            assert_eq!((decoded[2].width, decoded[2].height), (w, h));
        }
    }

    #[test]
    fn wave_a_10bit_422_444_and_act_round_trip() {
        for (bd, cf, act) in [
            (10u8, ChromaFormatIdc::Chroma420, false),
            (10, ChromaFormatIdc::Chroma422, false),
            (10, ChromaFormatIdc::Chroma444, false),
            (10, ChromaFormatIdc::Chroma444, true),
            (8, ChromaFormatIdc::Chroma444, true),
        ] {
            let mut frame = VvcFrame::new(64, 64, bd, cf);
            let max_v = (1u16 << bd) - 1;
            for y in 0..frame.y.height {
                for x in 0..frame.y.width {
                    frame
                        .y
                        .set(x, y, (((x * 7 + y * 5) as u16) << (bd - 8)).min(max_v));
                }
            }
            for y in 0..frame.u.height {
                for x in 0..frame.u.width {
                    frame.u.set(x, y, (110u16 << (bd - 8)).min(max_v));
                    frame.v.set(x, y, (145u16 << (bd - 8)).min(max_v));
                }
            }
            let (cabac, recon) =
                encode_and_reconstruct_vvc_frame(&frame, 12, act).expect("encode VvcFrame");
            assert!(!cabac.is_empty());
            assert_eq!(recon.width, 64);
            assert_eq!(recon.height, 64);
            assert_eq!(recon.bit_depth, bd);
            assert_eq!(recon.chroma_format, cf);
        }
    }

    #[test]
    fn wave_c_d_e_g_tools_round_trip() {
        // Wave C: Tile grid derivation
        let pps = VvcPps {
            width: 128,
            height: 64,
            no_pic_partition: false,
            ctu_log2: 5,
            exp_tile_col_widths: vec![2, 2],
            exp_tile_row_heights: vec![2],
            ..VvcPps::default()
        };
        let grid = pps.derive_tile_grid();
        assert_eq!(grid.num_tiles(), 2);
        assert_eq!(grid.ctu_coords_in_tile_scan().len(), 8);

        // Wave D: Palette block reconstruction
        let y_in = vec![200u16; 16];
        let u_in = vec![100u16; 16];
        let v_in = vec![150u16; 16];
        let plt = build_palette_from_block(&y_in, &u_in, &v_in, 4, 4, 8);
        let (ry, ru, rv) = reconstruct_palette_block(&plt, 4, 4, 8);
        assert_eq!(ry, y_in);
        assert_eq!(ru, u_in);
        assert_eq!(rv, v_in);

        // Wave E: Film Grain SEI build & parse + RPR ratio
        let fg = VvcFilmGrainSei {
            cancel_flag: false,
            model_id: 0,
            separate_colour_desc_present: false,
            blending_mode_id: 0,
            log2_scale_factor: 4,
            comp_model_present: [true, false, false],
            intervals: [vec![(0, 255, 6)], Vec::new(), Vec::new()],
            persistence_flag: true,
        };
        let sei_rbsp = build_film_grain_sei_rbsp(&fg);
        let msgs = parse_sei_rbsp(&sei_rbsp).expect("parse SEI");
        assert_eq!(msgs, vec![VvcSeiMessage::FilmGrain(fg)]);
        let ratio = derive_rpr_scaling_ratio(64, 64, 128, 128);
        assert_eq!(ratio.hor_scale, RPR_SCALE_1X / 2);

        // Wave G: Rate controller stability
        let mut rc = VvcRateController::new(VvcRateControlMode::Vbr {
            target_kbps: 500,
            fps_num: 30,
            fps_den: 1,
            init_qp: 26,
        });
        let qp0 = rc.select_picture_qp(VvcSliceType::I);
        rc.update_after_picture(4000);
        let qp1 = rc.select_picture_qp(VvcSliceType::P);
        assert!((qp0 as i32 - qp1 as i32).abs() <= 4);
    }
}
