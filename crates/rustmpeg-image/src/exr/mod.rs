// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! OpenEXR (`exr`) v2 decoder and encoder for `rustmpeg-next`.
//!
//! This module implements the OpenEXR v2/v3 binary container, attribute catalog,
//! single-part and multi-part scanline, tiled (`OneLevel`, `MipmapLevels`, `RipmapLevels`),
//! deep scanline, and deep tiled pipelines, Luminance/Chroma (`Y` + `RY` + `BY`) ↔ `RGB`
//! reconstruction, `displayWindow` framing, Mipmap/Ripmap generation, and 11
//! compression codecs:
//! - [`ExrCompression::None`] (`0`) — uncompressed,
//! - [`ExrCompression::Rle`] (`1`) — byte-split delta prediction + PackBits RLE,
//! - [`ExrCompression::Zips`] (`2`) — byte-split delta prediction + RFC 1950 `zlib` (1 scanline),
//! - [`ExrCompression::Zip`] (`3`) — byte-split delta prediction + RFC 1950 `zlib` (16 scanlines),
//! - [`ExrCompression::Piz`] (`4`) — 16-bit bitmap LUT + 2D Haar wavelet (`wenc14`/`wdec14` and `wenc16`/`wdec16`) + 58-bit canonical Huffman (`Huf`),
//! - [`ExrCompression::Pxr24`] (`5`) — Pixar 24-bit float + big-endian byte-plane delta transpose + `zlib`,
//! - [`ExrCompression::B44`] (`6`) — 4x4 fixed-rate 14-byte half-float block compression,
//! - [`ExrCompression::B44a`] (`7`) — 4x4 half-float block compression with 3-byte (`0xfc`) flat-field mode,
//! - [`ExrCompression::Dwaa`] (`8`) — DreamWorks 8x8 DCT + perceptual LUT + BT.709 CSC + hybrid `Huf`/`zlib`/`RLE` (32 scanlines),
//! - [`ExrCompression::Dwab`] (`9`) — DreamWorks 8x8 DCT + perceptual LUT + BT.709 CSC + hybrid `Huf`/`zlib`/`RLE` (256 scanlines),
//! - [`ExrCompression::Zstd`] (`13`) — 16-bit/32-bit channel sort + row delta + Blosc byte shuffle + RFC 8878 Zstandard (1 scanline, flat & deep).

pub mod compression;
pub mod decoder;
pub mod deflate;
pub mod dwa;
pub mod encoder;
pub mod half;
pub mod header;
pub mod luma_chroma;
pub mod zstd;

use thiserror::Error;

pub use self::decoder::{ExrDecoder, linear_to_srgb_u8};
pub use self::dwa::{
    DEFAULT_DWA_LEVEL, dwa_compress_chunk, dwa_decompress_chunk, dwa_to_linear, dwa_to_nonlinear,
};
pub use self::encoder::{ExrEncoder, srgb_u8_to_linear};
pub use self::half::{
    b44_convert_from_linear, b44_convert_to_linear, f16_to_f32, f16_to_f32_bits, f32_bits_to_f16,
    f32_to_f16, float_to_float24, float24_to_float,
};
pub use self::header::{
    DEFAULT_MAX_IMAGE_DIM, DEFAULT_MAX_TILE_DIM, EXR_FLAG_DEEP, EXR_FLAG_LONG_NAMES,
    EXR_FLAG_MULTIPART, EXR_FLAG_TILED, EXR_MAGIC, ExrAttribute, ExrAttributeValue, ExrBox2f,
    ExrBox2i, ExrChannelDesc, ExrChromaticities, ExrCompression, ExrFileHeader, ExrLevelMode,
    ExrLevelRoundingMode, ExrLineOrder, ExrPartHeader, ExrPartType, ExrPreview, ExrSampleType,
    ExrTileDesc,
};
pub use self::luma_chroma::{
    BT709_CHROMATICITIES, ExrChromaSlice, ExrLumaChromaPlanes, ExrRgbPlanes, chromaticities_of,
    downsample_tent, luma_chroma_to_rgb, luminance_weights, luminance_weights_of,
    rgb_to_luma_chroma, upsample_bilinear,
};
pub use self::zstd::{
    ZSTD_EXR_MAGIC, ZstdGridScenario, zstd_compress_chunk, zstd_compress_frame,
    zstd_decompress_chunk, zstd_decompress_frame,
};

/// Errors produced during OpenEXR decoding or encoding.
#[derive(Debug, Error)]
pub enum ExrError {
    /// Missing or invalid OpenEXR 4-byte magic (`0x76, 0x2f, 0x31, 0x01`).
    #[error("invalid OpenEXR magic signature")]
    InvalidMagic,
    /// Unsupported OpenEXR file format version word.
    #[error("unsupported OpenEXR version field: {0:#010x}")]
    UnsupportedVersion(u32),
    /// Malformed or non-conformant OpenEXR file/part header or attribute.
    #[error("invalid OpenEXR header: {0}")]
    InvalidHeader(&'static str),
    /// Unsupported compression codec (`HTJ2K256`, `HTJ2K32`, `LJ2K`).
    #[error("unsupported OpenEXR compression codec: {0:?}")]
    UnsupportedCompression(ExrCompression),
    /// Image or tile dimensions exceed configured safety bounds.
    #[error("OpenEXR image or tile dimensions exceed configured safety limits")]
    DimensionsTooLarge,
    /// Corrupt chunk header, offset table, or compressed payload.
    #[error("corrupt OpenEXR chunk: {0}")]
    CorruptChunk(&'static str),
    /// Unexpected end of input buffer.
    #[error("unexpected end of OpenEXR stream")]
    UnexpectedEof,
    /// Core media frame error.
    #[error("core media error: {0}")]
    Core(#[from] rustmpeg_core::CoreError),
}

/// Configuration for [`ExrDecoder`] and [`ExrEncoder`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExrConfig {
    /// Default compression method when encoding a `VideoFrame` (`Zip` by default).
    pub compression: ExrCompression,
    /// Default `DWAA`/`DWAB` compression level (`45.0` by default; `errorTarget = level / 100_000.0`).
    pub dwa_compression_level: f32,
    /// Maximum allowed image width (`65536` by default; `0` disables check).
    pub max_width: u32,
    /// Maximum allowed image height (`65536` by default; `0` disables check).
    pub max_height: u32,
    /// Enforce strict header, offset-table, and chunk-alignment validation (`true` by default).
    pub strict: bool,
}

impl Default for ExrConfig {
    fn default() -> Self {
        Self {
            compression: ExrCompression::Zip,
            dwa_compression_level: DEFAULT_DWA_LEVEL,
            max_width: DEFAULT_MAX_IMAGE_DIM,
            max_height: DEFAULT_MAX_IMAGE_DIM,
            strict: true,
        }
    }
}

impl ExrConfig {
    /// Creates an encoder/decoder configuration with the specified [`ExrCompression`].
    #[must_use]
    pub fn with_compression(compression: ExrCompression) -> Self {
        Self {
            compression,
            ..Self::default()
        }
    }

    /// Sets the `DWAA`/`DWAB` compression level (`errorTarget = level / 100_000.0`).
    #[must_use]
    pub fn with_dwa_level(mut self, level: f32) -> Self {
        self.dwa_compression_level = level;
        self
    }
}

/// Typed sample storage for a single OpenEXR channel.
#[derive(Debug, Clone, PartialEq)]
pub enum ExrChannelData {
    /// 32-bit unsigned integer samples (`UINT = 0`).
    Uint(Vec<u32>),
    /// 16-bit IEEE 754 binary16 samples (`HALF = 1`, stored as raw `u16` bits).
    Half(Vec<u16>),
    /// 32-bit IEEE 754 binary32 samples (`FLOAT = 2`).
    Float(Vec<f32>),
}

impl ExrChannelData {
    /// Number of samples stored in this channel buffer.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Uint(v) => v.len(),
            Self::Half(v) => v.len(),
            Self::Float(v) => v.len(),
        }
    }

    /// Returns true if the buffer contains 0 samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the [`ExrSampleType`] corresponding to this buffer variant.
    #[inline]
    pub const fn sample_type(&self) -> ExrSampleType {
        match self {
            Self::Uint(_) => ExrSampleType::Uint,
            Self::Half(_) => ExrSampleType::Half,
            Self::Float(_) => ExrSampleType::Float,
        }
    }

    /// Converts sample at `index` to `f32` (`HALF` via `f16_to_f32`, `UINT` via `as f32`).
    #[inline]
    pub fn sample_as_f32(&self, index: usize) -> f32 {
        match self {
            Self::Uint(v) => v.get(index).copied().unwrap_or(0) as f32,
            Self::Half(v) => v.get(index).copied().map(f16_to_f32).unwrap_or(0.0),
            Self::Float(v) => v.get(index).copied().unwrap_or(0.0),
        }
    }
}

/// Planar pixel buffer for a single channel at a given resolution level.
#[derive(Debug, Clone, PartialEq)]
pub struct ExrChannelPlane {
    /// Channel descriptor (`name`, `sample_type`, `p_linear`, `x_sampling`, `y_sampling`).
    pub desc: ExrChannelDesc,
    /// Plane width in samples (`data_window.width() / x_sampling`).
    pub width: u32,
    /// Plane height in samples (`data_window.height() / y_sampling`).
    pub height: u32,
    /// Typed sample storage (`width * height` samples).
    pub data: ExrChannelData,
}

impl ExrChannelPlane {
    /// Reads sample `index` as an `f32`.
    #[inline]
    pub fn sample_as_f32(&self, index: usize) -> f32 {
        self.data.sample_as_f32(index)
    }
}

/// Decoded or input mipmap/ripmap resolution level `(level_x, level_y)` (`!= (0, 0)`).
#[derive(Debug, Clone, PartialEq)]
pub struct ExrLevelImage {
    /// Horizontal level index (`lx`).
    pub level_x: u32,
    /// Vertical level index (`ly`).
    pub level_y: u32,
    /// Level width in pixels.
    pub width: u32,
    /// Level height in pixels.
    pub height: u32,
    /// Sorted channel planes for this resolution level.
    pub channels: Vec<ExrChannelPlane>,
}

/// Deep pixel sample storage for `deepscanline` and `deeptile` parts.
#[derive(Debug, Clone, PartialEq)]
pub struct ExrDeepSamples {
    /// Part width in pixels.
    pub width: u32,
    /// Part height in pixels.
    pub height: u32,
    /// Number of deep samples at each pixel (`width * height` entries, row-major).
    pub sample_counts: Vec<u32>,
    /// Flattened sample arrays per channel (each of length `sample_counts.iter().sum()`).
    pub channels: Vec<ExrChannelData>,
}

/// A single decoded or input OpenEXR part (flat scanline, flat tiled, deep scanline, or deep tiled).
#[derive(Debug, Clone, PartialEq)]
pub struct ExrPart {
    /// Validated part header.
    pub header: ExrPartHeader,
    /// Level `(0, 0)` planar channels (sorted by ASCII byte order of `desc.name`).
    pub channels: Vec<ExrChannelPlane>,
    /// Additional mipmap/ripmap levels (`(lx, ly) != (0, 0)`) when `part_type == TiledImage`.
    pub extra_levels: Vec<ExrLevelImage>,
    /// Deep sample storage when `part_type` is `DeepScanline` or `DeepTile`.
    pub deep_samples: Option<ExrDeepSamples>,
}

impl ExrPart {
    /// Constructs a flat scanline [`ExrPart`] from a list of [`ExrChannelPlane`]s,
    /// automatically sorting channels into canonical ASCII byte order.
    pub fn from_planar_channels(
        width: u32,
        height: u32,
        compression: ExrCompression,
        mut channels: Vec<ExrChannelPlane>,
    ) -> Result<Self, ExrError> {
        if width == 0 || height == 0 || channels.is_empty() {
            return Err(ExrError::InvalidHeader(
                "width, height, and channel count must be non-zero",
            ));
        }
        channels.sort_by(|a, b| a.desc.name.as_bytes().cmp(b.desc.name.as_bytes()));
        for ch in &channels {
            let xs = (ch.desc.x_sampling.max(1)) as u32;
            let ys = (ch.desc.y_sampling.max(1)) as u32;
            let expected_w = width / xs;
            let expected_h = height / ys;
            if ch.width != expected_w
                || ch.height != expected_h
                || ch.data.len() != (expected_w as usize) * (expected_h as usize)
            {
                return Err(ExrError::InvalidHeader(
                    "channel plane dimensions do not match dataWindow / sampling",
                ));
            }
        }
        let descs: Vec<ExrChannelDesc> = channels.iter().map(|c| c.desc.clone()).collect();
        let header = ExrPartHeader::new_scanline(width, height, compression, descs);
        Ok(Self {
            header,
            channels,
            extra_levels: Vec::new(),
            deep_samples: None,
        })
    }

    /// Looks up a level-0 channel plane by its exact name (e.g. `"R"`, `"G"`, `"B"`, `"A"`, `"Z"`).
    #[must_use]
    pub fn channel_by_name(&self, name: &str) -> Option<&ExrChannelPlane> {
        self.channels.iter().find(|c| c.desc.name == name)
    }
}

/// Complete decoded or input OpenEXR file (`1` or more [`ExrPart`]s).
#[derive(Debug, Clone, PartialEq)]
pub struct ExrImage {
    /// File format version (`2`).
    pub version: u8,
    /// File format version flags (`EXR_FLAG_*`).
    pub flags: u32,
    /// Ordered parts (`>= 1`).
    pub parts: Vec<ExrPart>,
}

impl ExrImage {
    /// Wraps a single [`ExrPart`] into a single-part [`ExrImage`].
    #[must_use]
    pub fn from_single_part(part: ExrPart) -> Self {
        let mut flags = 0u32;
        if part.header.part_type.is_deep() {
            flags |= EXR_FLAG_DEEP;
        } else if part.header.part_type.is_tiled() {
            flags |= EXR_FLAG_TILED;
        }
        if part.header.requires_long_names() {
            flags |= EXR_FLAG_LONG_NAMES;
        }
        Self {
            version: 2,
            flags,
            parts: vec![part],
        }
    }
}
