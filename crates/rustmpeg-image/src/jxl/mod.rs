// SPDX-License-Identifier: Apache-2.0

//! JPEG XL (ISO/IEC 18181-1 & 18181-2) v2 decoder and clean-room encoder.
//!
//! This module implements:
//! - [`container`]: ISO/IEC 18181-2 container parser and writer (`JXL `, `ftyp`, `jxll`, `jxlc`,
//!   `jxlp`, `Exif`, `xml `) and policy rejection of legacy JPEG reconstruction (`jbrd`).
//! - [`bits`]: LSB-first bitstream reader/writer (`U32Specifier`, `F16`, `UnpackSigned`, `Enum`, `Name`).
//! - [`entropy`]: Hybrid integer coding, canonical prefix codes, 12-bit rANS with alias mapping,
//!   2D LZ77 special distances, and clean-room `Prefix` / `12-bit rANS` + `2D LZ77` encoder.
//! - [`headers`]: `SizeHeader`, `ImageMetadata`, `ColourEncoding`, `ToneMapping`,
//!   `OpsinInverseMatrix`, `ExtraChannelInfo`, `AnimationHeader`, `read_icc`/`write_icc`,
//!   `FrameHeader`, `RestorationFilter`, and `Toc`.
//! - [`modular`]: Modular sub-bitstream decoder and clean-room v2 encoder (`WpHeader`, 14 spatial
//!   predictors, `SelfCorrectingPredictor`, multi-node `MaConfig` trees, and reversible `Rct`,
//!   `Palette`, `Squeeze` transforms).
//! - [`vardct`]: VarDCT sub-bitstream decoder and clean-room v2 encoder (`LfChannelDequantization`,
//!   `Quantizer`, `HfBlockContext`, `LfChannelCorrelation`, `DequantMatrixSet`, `LfCoeff`,
//!   `HfMetadata`, `HfGlobal`, `HfPass`, 2D DCT/IDCT, all 27 `TransformType`s, adaptive `Dct16`/`Dct8`
//!   selection, Chroma-from-Luma, adaptive quantization, XYB + CIE 1931 XYZ / Bradford gamut
//!   adaptation, Gaborish / inverse-Gaborish, and Edge-Preserving Filter).
//! - [`decoder`]: High-level [`JxlDecoder`] with full `Patches`, `Splines`, `Noise`, `LfFrame`,
//!   `moxcms` ICC profile support, and `VideoFrame` output.
//! - [`encoder`]: Clean-room Apache-2.0 [`JxlEncoder`] supporting `ModularLossless` and
//!   `VarDctLossy` with v2 tools (`Ans`, `LZ77`, `Palette`, `Squeeze`, `MaConfig`, `CfL`,
//!   adaptive `Dct16`/`Dct8`, `EPF`, wide-gamut `DisplayP3`/`Bt2020Pq`/`Bt2020Hlg`, and `write_icc`).

pub mod bits;
pub mod container;
pub mod decoder;
pub mod encoder;
pub mod entropy;
pub mod headers;
pub mod modular;
pub mod vardct;

use thiserror::Error;

use crate::cms::CmsError;

pub use self::container::{JxlContainerKind, ParsedContainer};
pub use self::decoder::JxlDecoder;
pub use self::encoder::JxlEncoder;
pub use self::entropy::JxlEntropyMode;
pub use self::headers::{AnimationHeader, ImageHeader, ImageMetadata};
pub use self::modular::Predictor;

/// Errors produced during JPEG XL decoding or encoding.
#[derive(Debug, Error)]
pub enum JxlError {
    /// Missing or invalid JPEG XL codestream (`0xFF 0x0A`) or ISO/IEC 18181-2 container signature.
    #[error("invalid JPEG XL signature")]
    InvalidSignature,
    /// Malformed ISO/IEC 18181-2 container structure or box sequence.
    #[error("invalid JXL container: {0}")]
    InvalidContainer(&'static str),
    /// Malformed or non-conformant ISO/IEC 18181-1 codestream syntax.
    #[error("invalid JXL bitstream: {0}")]
    InvalidBitstream(&'static str),
    /// Invalid caller input or encoder configuration.
    #[error("invalid JXL configuration or input: {0}")]
    InvalidConfig(&'static str),
    /// Unexpected end of bitstream or container payload.
    #[error("unexpected end of JXL bitstream")]
    UnexpectedEof,
    /// Resource, profile, or allocation limit exceeded.
    #[error("JXL limit exceeded: {0}")]
    LimitExceeded(&'static str),
    /// Unsupported feature or policy-excluded box (such as `jbrd` legacy JPEG reconstruction).
    #[error("unsupported JXL feature: {0}")]
    Unsupported(&'static str),
    /// ICC / CICP color management error.
    #[error("JXL color management error: {0}")]
    Cms(#[from] CmsError),
    /// Core frame or plane validation error.
    #[error("core media error: {0}")]
    Core(#[from] rustmpeg_core::CoreError),
}

/// Pixel channel layout of a [`JxlFrame`] or [`JxlImage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JxlPixelLayout {
    /// 1-channel grayscale (`Y`).
    Gray,
    /// 2-channel grayscale + alpha (`YA`).
    GrayAlpha,
    /// 3-channel RGB (`RGB`).
    Rgb,
    /// 4-channel RGBA (`RGBA`).
    Rgba,
}

impl JxlPixelLayout {
    /// Number of interleaved channels per pixel (`1..=4`).
    #[must_use]
    #[inline]
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    /// Returns true if the layout includes an alpha channel.
    #[must_use]
    #[inline]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::GrayAlpha | Self::Rgba)
    }
}

/// Sample bit depth and storage format of a [`JxlFrame`] or [`JxlImage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JxlBitDepth {
    /// Unsigned 8-bit container per sample with `1..=8` nominal bits.
    Uint8(u8),
    /// Unsigned little-endian 16-bit container per sample with `9..=16` nominal bits.
    Uint16(u8),
    /// IEEE-754 little-endian 32-bit float per sample (`0.0..=1.0`).
    Float32,
}

impl JxlBitDepth {
    /// Nominal bits per sample (`1..=32`).
    #[must_use]
    #[inline]
    pub const fn bits_per_sample(self) -> u8 {
        match self {
            Self::Uint8(b) | Self::Uint16(b) => b,
            Self::Float32 => 32,
        }
    }

    /// Bytes per channel sample in `JxlFrame::data` (`1`, `2`, or `4`).
    #[must_use]
    #[inline]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Uint8(_) => 1,
            Self::Uint16(_) => 2,
            Self::Float32 => 4,
        }
    }
}

/// Signalled color encoding of a [`JxlImage`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum JxlColorEncoding {
    /// Standard sRGB (D65, sRGB primaries, sRGB transfer function).
    #[default]
    Srgb,
    /// Linear-light sRGB (D65, sRGB primaries, linear transfer function).
    LinearSrgb,
    /// Grayscale with sRGB transfer curve.
    GraySrgb,
    /// Linear-light grayscale.
    LinearGray,
    /// Display P3 wide-gamut RGB (D65, P3 primaries, sRGB transfer function).
    DisplayP3,
    /// Rec. 2100 / BT.2020 HDR with SMPTE ST 2084 PQ transfer function.
    Bt2020Pq,
    /// Rec. 2100 / BT.2020 HDR with ITU-R BT.2100 HLG transfer function.
    Bt2020Hlg,
    /// Embedded raw ICC profile bytes.
    Icc(Vec<u8>),
}

/// Encoding mode for [`JxlEncoder`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JxlEncodingMode {
    /// Mathematically lossless (or near-lossless when `modular_max_error > 0`) Modular encoding.
    ModularLossless {
        /// Spatial predictor mode (`Predictor::Gradient` or `Predictor::SelfCorrecting` recommended).
        predictor: Predictor,
    },
    /// Lossy XYB VarDCT frequency-domain encoding.
    VarDctLossy {
        /// Quality factor in `1.0..=100.0` (`90.0` recommended).
        quality: f32,
    },
}

impl Default for JxlEncodingMode {
    fn default() -> Self {
        Self::ModularLossless {
            predictor: Predictor::Gradient,
        }
    }
}

/// Configuration for [`JxlEncoder`] (v1 and v2 tool switches).
#[derive(Debug, Clone, PartialEq)]
pub struct JxlConfig {
    /// Encoding mode (`ModularLossless` or `VarDctLossy`).
    pub mode: JxlEncodingMode,
    /// Whether to apply the reversible YCoCg colour transform (`rct_type = 6`) in `ModularLossless`
    /// mode on RGB/RGBA images.
    pub use_rct: bool,
    /// Whether to enable the 3x3 Gaborish deblocking filter (with encoder pre-sharpening) in `VarDctLossy` mode.
    pub gaborish: bool,
    /// Whether to wrap the codestream in an ISO/IEC 18181-2 container even when no metadata boxes
    /// are present.
    pub force_container: bool,
    /// Entropy coding backend (`Prefix` or 12-bit `Ans`).
    pub entropy_mode: JxlEntropyMode,
    /// Enable 2D LZ77 backward-reference match finding in Modular streams.
    pub lz77: bool,
    /// Automatically extract a lossless `Palette` transform when unique colors `<= 256`.
    pub enable_palette: bool,
    /// Apply Haar-like multi-resolution `Squeeze` transform in `ModularLossless` mode.
    pub enable_squeeze: bool,
    /// Emit a 4-context adaptive `MaConfig` decision tree in `ModularLossless` mode.
    pub modular_ma_tree: bool,
    /// Maximum per-sample quantization error in Modular mode (`0` = strictly lossless).
    pub modular_max_error: u8,
    /// Estimate Chroma-from-Luma (`x_from_y`, `b_from_y`) correlation in `VarDctLossy` mode.
    pub enable_cfl: bool,
    /// Enable adaptive `Dct16` / `Dct8` varblock transform selection in `VarDctLossy` mode.
    pub enable_adaptive_dct: bool,
    /// Enable activity-masked per-block `hf_mul` quantization in `VarDctLossy` mode.
    pub enable_adaptive_quant: bool,
    /// Number of Edge-Preserving Filter stages (`0..=3`) in `VarDctLossy` mode.
    pub epf_iters: u8,
}

impl Default for JxlConfig {
    fn default() -> Self {
        Self {
            mode: JxlEncodingMode::default(),
            use_rct: true,
            gaborish: false,
            force_container: false,
            entropy_mode: JxlEntropyMode::Prefix,
            lz77: false,
            enable_palette: false,
            enable_squeeze: false,
            modular_ma_tree: false,
            modular_max_error: 0,
            enable_cfl: true,
            enable_adaptive_dct: true,
            enable_adaptive_quant: true,
            epf_iters: 0,
        }
    }
}

impl JxlConfig {
    /// Creates a fast lossless Modular encoder configuration.
    #[must_use]
    pub fn lossless() -> Self {
        Self::default()
    }

    /// Creates a high-compression v2 lossless Modular configuration (`SelfCorrecting` predictor,
    /// 12-bit `Ans`, `2D LZ77`, automatic `Palette`, `YCoCg` RCT, and 4-context `MaConfig` tree).
    #[must_use]
    pub fn lossless_v2() -> Self {
        Self {
            mode: JxlEncodingMode::ModularLossless {
                predictor: Predictor::SelfCorrecting,
            },
            use_rct: true,
            gaborish: false,
            force_container: false,
            entropy_mode: JxlEntropyMode::Ans,
            lz77: true,
            enable_palette: true,
            enable_squeeze: false,
            modular_ma_tree: true,
            modular_max_error: 0,
            enable_cfl: false,
            enable_adaptive_dct: false,
            enable_adaptive_quant: false,
            epf_iters: 0,
        }
    }

    /// Creates a lossy VarDCT encoder configuration with `quality` in `1.0..=100.0`.
    #[must_use]
    pub fn lossy(quality: f32) -> Self {
        Self {
            mode: JxlEncodingMode::VarDctLossy { quality },
            use_rct: false,
            gaborish: false,
            force_container: false,
            entropy_mode: JxlEntropyMode::Prefix,
            lz77: false,
            enable_palette: false,
            enable_squeeze: false,
            modular_ma_tree: false,
            modular_max_error: 0,
            enable_cfl: true,
            enable_adaptive_dct: false,
            enable_adaptive_quant: false,
            epf_iters: 0,
        }
    }

    /// Creates a full v2 lossy VarDCT encoder configuration (`12-bit rANS`, `CfL`, adaptive
    /// `Dct16`/`Dct8`, adaptive quantization, Gaborish pre-sharpening + deblocking, and EPF).
    #[must_use]
    pub fn lossy_v2(quality: f32) -> Self {
        Self {
            mode: JxlEncodingMode::VarDctLossy { quality },
            use_rct: false,
            gaborish: true,
            force_container: false,
            entropy_mode: JxlEntropyMode::Ans,
            lz77: true,
            enable_palette: false,
            enable_squeeze: false,
            modular_ma_tree: false,
            modular_max_error: 0,
            enable_cfl: true,
            enable_adaptive_dct: true,
            enable_adaptive_quant: true,
            epf_iters: 1,
        }
    }
}

/// A single decoded or input frame in a [`JxlImage`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JxlFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel channel layout.
    pub layout: JxlPixelLayout,
    /// Sample bit depth and storage format.
    pub bit_depth: JxlBitDepth,
    /// Display duration in animation ticks (`0` for still images).
    pub duration_ticks: u32,
    /// Optional UTF-8 frame name.
    pub name: String,
    /// Interleaved pixel bytes (`width * height * layout.channels() * bit_depth.bytes_per_sample()`).
    pub data: Vec<u8>,
}

impl JxlFrame {
    /// Creates an 8-bit frame from raw interleaved pixel bytes.
    pub fn new_u8(
        width: u32,
        height: u32,
        layout: JxlPixelLayout,
        data: Vec<u8>,
    ) -> Result<Self, JxlError> {
        let expected = (width as usize) * (height as usize) * layout.channels();
        if data.len() != expected {
            return Err(JxlError::InvalidConfig(
                "data length does not match width * height * channels",
            ));
        }
        Ok(Self {
            width,
            height,
            layout,
            bit_depth: JxlBitDepth::Uint8(8),
            duration_ticks: 0,
            name: String::new(),
            data,
        })
    }

    /// Creates a 16-bit frame from native-endian `u16` samples.
    pub fn new_u16(
        width: u32,
        height: u32,
        layout: JxlPixelLayout,
        nominal_bits: u8,
        samples: &[u16],
    ) -> Result<Self, JxlError> {
        let expected = (width as usize) * (height as usize) * layout.channels();
        if samples.len() != expected || !(9..=16).contains(&nominal_bits) {
            return Err(JxlError::InvalidConfig(
                "invalid sample count or nominal_bits for 16-bit JxlFrame",
            ));
        }
        let mut data = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        Ok(Self {
            width,
            height,
            layout,
            bit_depth: JxlBitDepth::Uint16(nominal_bits),
            duration_ticks: 0,
            name: String::new(),
            data,
        })
    }
}

/// Complete decoded or input JPEG XL image (still image or multi-frame animation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JxlImage {
    /// Oriented image width in pixels.
    pub width: u32,
    /// Oriented image height in pixels.
    pub height: u32,
    /// Pixel channel layout.
    pub layout: JxlPixelLayout,
    /// Sample bit depth and storage format.
    pub bit_depth: JxlBitDepth,
    /// Color encoding metadata.
    pub color_encoding: JxlColorEncoding,
    /// Container format (`BareCodestream` or `Container`).
    pub container_kind: JxlContainerKind,
    /// Optional raw EXIF payload from an `Exif` box.
    pub exif: Option<Vec<u8>>,
    /// Optional raw XMP / XML payload from an `xml ` box.
    pub xmp: Option<Vec<u8>>,
    /// Optional animation timing header.
    pub animation: Option<AnimationHeader>,
    /// Ordered sequence of visible frames (`>= 1`).
    pub frames: Vec<JxlFrame>,
}

impl JxlImage {
    /// Wraps a single [`JxlFrame`] into a still [`JxlImage`].
    #[must_use]
    pub fn from_single_frame(frame: JxlFrame) -> Self {
        let color_encoding = match frame.layout {
            JxlPixelLayout::Gray | JxlPixelLayout::GrayAlpha => JxlColorEncoding::GraySrgb,
            JxlPixelLayout::Rgb | JxlPixelLayout::Rgba => JxlColorEncoding::Srgb,
        };
        Self {
            width: frame.width,
            height: frame.height,
            layout: frame.layout,
            bit_depth: frame.bit_depth,
            color_encoding,
            container_kind: JxlContainerKind::BareCodestream,
            exif: None,
            xmp: None,
            animation: None,
            frames: vec![frame],
        }
    }
}
