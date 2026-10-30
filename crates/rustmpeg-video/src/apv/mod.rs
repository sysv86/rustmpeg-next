// SPDX-License-Identifier: Apache-2.0
//! Advanced Professional Video (APV) codec, RFC 9924.
//!
//! APV is an intra-only mezzanine codec: each frame is split into tiles of
//! 16x16 macroblocks, each macroblock into 8x8 transform blocks, and every
//! block is coded independently with an integer DCT and a differential
//! DC/run-level AC syntax.  There is no inter prediction.
//!
//! The implementation is written from RFC 9924.  No OpenAPV, FFmpeg, or other
//! reference source is transcribed.  Normative constants (the 8x8 transform
//! matrix, level scale, profiles, and level table) are taken from the RFC and
//! cited in the module that uses them.

mod bits;
mod decoder;
mod encoder;
mod syntax;
mod transform;
mod vlc;

pub use decoder::ApvDecoder;
pub use encoder::ApvEncoder;

use std::fmt;

use rustmpeg_core::ColorDescription;
use thiserror::Error;

/// Largest frame dimension accepted by the codec.
pub const MAX_DIMENSION: u32 = 16_384;
/// Largest number of luma samples accepted in one frame.
pub const MAX_LUMA_SAMPLES: u64 = 35_651_584;
/// Largest number of tiles accepted in one frame.
pub const MAX_TILES: usize = 400;
/// Largest number of macroblocks accepted in one frame.
pub const MAX_MACROBLOCKS: u64 = MAX_LUMA_SAMPLES / 256 + 1;

/// Chroma sampling layout of an APV frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApvChromaFormat {
    /// 4:0:0, one sample array.
    Mono,
    /// 4:2:2, chroma halved horizontally.
    Yuv422,
    /// 4:4:4, full-resolution chroma.
    Yuv444,
    /// 4:4:4:4, three colour arrays plus a fourth array (alpha or key).
    Yuv4444,
}

impl ApvChromaFormat {
    /// `chroma_format_idc` value from RFC 9924, Table 2.
    #[must_use]
    pub const fn idc(self) -> u8 {
        match self {
            Self::Mono => 0,
            Self::Yuv422 => 2,
            Self::Yuv444 => 3,
            Self::Yuv4444 => 4,
        }
    }

    /// Number of sample arrays.
    #[must_use]
    pub const fn components(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Yuv422 | Self::Yuv444 => 3,
            Self::Yuv4444 => 4,
        }
    }

    /// Horizontal chroma subsampling exponent.
    #[must_use]
    pub const fn sub_width(self) -> u32 {
        match self {
            Self::Yuv422 => 1,
            _ => 0,
        }
    }

    /// Vertical chroma subsampling exponent.
    #[must_use]
    pub const fn sub_height(self) -> u32 {
        0
    }

    /// Parses a `chroma_format_idc` value.
    ///
    /// # Errors
    ///
    /// Returns [`ApvError::Unsupported`] for reserved values.
    pub fn from_idc(idc: u8) -> Result<Self, ApvError> {
        match idc {
            0 => Ok(Self::Mono),
            2 => Ok(Self::Yuv422),
            3 => Ok(Self::Yuv444),
            4 => Ok(Self::Yuv4444),
            _ => Err(ApvError::Unsupported("reserved chroma_format_idc")),
        }
    }
}

/// One decoded APV frame with planar `u16` samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApvFrame {
    /// Visible width in luma samples.
    pub width: u32,
    /// Visible height in luma samples.
    pub height: u32,
    /// Chroma sampling layout.
    pub chroma_format: ApvChromaFormat,
    /// Sample bit depth, normally 10 or 12 for a conformant profile.
    pub bit_depth: u8,
    /// Capture-time distance signalled by the frame, in milliseconds.
    pub capture_time_distance: u8,
    /// Optional colour description from the frame header.
    pub color: Option<ColorDescription>,
    /// Planar samples in component order (Y, Cb, Cr, fourth).
    pub planes: Vec<Vec<u16>>,
}

impl ApvFrame {
    /// Allocates a zeroed frame.
    ///
    /// # Errors
    ///
    /// Rejects zero or oversized dimensions, an odd width in 4:2:2, a bit
    /// depth outside 10..=12, or an unsupported layout.
    pub fn new(
        chroma_format: ApvChromaFormat,
        bit_depth: u8,
        width: u32,
        height: u32,
    ) -> Result<Self, ApvError> {
        validate_geometry(chroma_format, bit_depth, width, height)?;
        let mut planes = Vec::with_capacity(chroma_format.components());
        for component in 0..chroma_format.components() {
            let (plane_width, plane_height) =
                plane_dimensions(chroma_format, width, height, component);
            let len = usize::try_from(u64::from(plane_width) * u64::from(plane_height))
                .map_err(|_| ApvError::Limit("frame plane"))?;
            planes.push(vec![0; len]);
        }
        Ok(Self {
            width,
            height,
            chroma_format,
            bit_depth,
            capture_time_distance: 0,
            color: None,
            planes,
        })
    }

    /// Builds a frame from caller-provided planes after validating their sizes.
    ///
    /// # Errors
    ///
    /// Rejects an invalid geometry or mismatched plane lengths.
    pub fn from_planes(
        chroma_format: ApvChromaFormat,
        bit_depth: u8,
        width: u32,
        height: u32,
        color: Option<ColorDescription>,
        planes: Vec<Vec<u16>>,
    ) -> Result<Self, ApvError> {
        validate_geometry(chroma_format, bit_depth, width, height)?;
        if planes.len() != chroma_format.components() {
            return Err(ApvError::Frame("wrong number of planes"));
        }
        for (component, plane) in planes.iter().enumerate() {
            let (plane_width, plane_height) =
                plane_dimensions(chroma_format, width, height, component);
            let expected = usize::try_from(u64::from(plane_width) * u64::from(plane_height))
                .map_err(|_| ApvError::Limit("frame plane"))?;
            if plane.len() != expected {
                return Err(ApvError::Frame("plane has the wrong length"));
            }
            let maximum = (1_u32 << bit_depth) - 1;
            if plane.iter().any(|sample| u32::from(*sample) > maximum) {
                return Err(ApvError::Frame("sample exceeds the bit depth"));
            }
        }
        Ok(Self {
            width,
            height,
            chroma_format,
            bit_depth,
            capture_time_distance: 0,
            color,
            planes,
        })
    }

    /// Dimensions of one component plane.
    #[must_use]
    pub fn plane_size(&self, component: usize) -> Option<(u32, u32)> {
        (component < self.chroma_format.components())
            .then(|| plane_dimensions(self.chroma_format, self.width, self.height, component))
    }

    /// Checks that every plane matches the declared geometry.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.planes.len() == self.chroma_format.components()
            && (0..self.chroma_format.components()).all(|component| {
                self.plane_size(component).is_some_and(|(width, height)| {
                    self.planes[component].len() == (width as usize) * (height as usize)
                })
            })
    }
}

/// Encoder configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApvConfig {
    /// Level identifier, 30 times the level number (for example 123 = 4.1).
    pub level_idc: u8,
    /// Band identifier, `0..=3`.
    pub band_idc: u8,
    /// Luma quantization parameter, `0..=51`; chroma shares it.
    pub qp: u8,
    /// Tile width in macroblocks, at least 16.
    pub tile_width_in_mbs: u16,
    /// Tile height in macroblocks, at least 8.
    pub tile_height_in_mbs: u16,
}

impl Default for ApvConfig {
    fn default() -> Self {
        Self {
            level_idc: 123,
            band_idc: 3,
            qp: 24,
            tile_width_in_mbs: 16,
            tile_height_in_mbs: 8,
        }
    }
}

/// Errors produced by the APV encoder and decoder.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ApvError {
    /// The input ends before the requested bits.
    #[error("unexpected end of APV bitstream")]
    Eof,
    /// The access-unit signature is not `aPv1`.
    #[error("invalid APV access-unit signature")]
    InvalidSignature,
    /// A syntax element or structure is invalid.
    #[error("invalid APV bitstream: {0}")]
    Invalid(&'static str),
    /// The bitstream uses a feature outside this implementation.
    #[error("unsupported APV feature: {0}")]
    Unsupported(&'static str),
    /// A resource limit was exceeded.
    #[error("APV limit exceeded: {0}")]
    Limit(&'static str),
    /// The caller provided an invalid frame or configuration.
    #[error("invalid APV frame or configuration: {0}")]
    Frame(&'static str),
}

impl fmt::Display for ApvChromaFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Mono => "4:0:0",
            Self::Yuv422 => "4:2:2",
            Self::Yuv444 => "4:4:4",
            Self::Yuv4444 => "4:4:4:4",
        })
    }
}

fn validate_geometry(
    chroma_format: ApvChromaFormat,
    bit_depth: u8,
    width: u32,
    height: u32,
) -> Result<(), ApvError> {
    if width == 0 || height == 0 {
        return Err(ApvError::Frame("frame dimensions are zero"));
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(ApvError::Limit("frame dimension"));
    }
    let samples = u64::from(width) * u64::from(height);
    if samples > MAX_LUMA_SAMPLES {
        return Err(ApvError::Limit("luma samples"));
    }
    if !(10..=12).contains(&bit_depth) {
        return Err(ApvError::Frame("bit depth must be 10 or 12"));
    }
    if chroma_format == ApvChromaFormat::Yuv422 && !width.is_multiple_of(2) {
        return Err(ApvError::Frame("4:2:2 width must be even"));
    }
    Ok(())
}

fn plane_dimensions(
    chroma_format: ApvChromaFormat,
    width: u32,
    height: u32,
    component: usize,
) -> (u32, u32) {
    if component == 0 || chroma_format == ApvChromaFormat::Yuv4444 {
        return (width, height);
    }
    (
        width >> chroma_format.sub_width(),
        height >> chroma_format.sub_height(),
    )
}

pub(crate) const fn qp_bd_offset(bit_depth: u8) -> u8 {
    (bit_depth - 8) * 6
}

pub(crate) const fn max_tile_qp(bit_depth: u8) -> u8 {
    63 + (bit_depth - 10) * 6
}

pub(crate) const fn valid_level_idc(level_idc: u8) -> bool {
    matches!(
        level_idc,
        30 | 33 | 60 | 63 | 90 | 93 | 120 | 123 | 150 | 153 | 180 | 183 | 210 | 213
    )
}

pub(crate) const fn profile_for(
    chroma_format: ApvChromaFormat,
    bit_depth: u8,
) -> Result<u8, ApvError> {
    match (chroma_format, bit_depth) {
        (ApvChromaFormat::Mono, 10) => Ok(99),
        (ApvChromaFormat::Yuv422, 10) => Ok(33),
        (ApvChromaFormat::Yuv422, 12) => Ok(44),
        (ApvChromaFormat::Yuv444, 10) => Ok(55),
        (ApvChromaFormat::Yuv444, 12) => Ok(66),
        (ApvChromaFormat::Yuv4444, 10) => Ok(77),
        (ApvChromaFormat::Yuv4444, 12) => Ok(88),
        _ => Err(ApvError::Unsupported("no APV profile for this layout")),
    }
}

pub(crate) fn profile_allows(
    profile_idc: u8,
    chroma_format: ApvChromaFormat,
    bit_depth: u8,
) -> bool {
    match profile_idc {
        33 => chroma_format == ApvChromaFormat::Yuv422 && bit_depth == 10,
        44 => chroma_format == ApvChromaFormat::Yuv422 && (10..=12).contains(&bit_depth),
        55 => {
            matches!(
                chroma_format,
                ApvChromaFormat::Yuv422 | ApvChromaFormat::Yuv444
            ) && bit_depth == 10
        }
        66 => {
            matches!(
                chroma_format,
                ApvChromaFormat::Yuv422 | ApvChromaFormat::Yuv444
            ) && (10..=12).contains(&bit_depth)
        }
        77 => chroma_format != ApvChromaFormat::Mono && bit_depth == 10,
        88 => chroma_format != ApvChromaFormat::Mono && (10..=12).contains(&bit_depth),
        99 => chroma_format == ApvChromaFormat::Mono && bit_depth == 10,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_mapping_covers_the_rfc_profiles() {
        assert_eq!(
            profile_for(ApvChromaFormat::Yuv422, 10).expect("422-10"),
            33
        );
        assert_eq!(
            profile_for(ApvChromaFormat::Yuv422, 12).expect("422-12"),
            44
        );
        assert_eq!(
            profile_for(ApvChromaFormat::Yuv444, 10).expect("444-10"),
            55
        );
        assert_eq!(
            profile_for(ApvChromaFormat::Yuv4444, 12).expect("4444-12"),
            88
        );
        assert_eq!(profile_for(ApvChromaFormat::Mono, 10).expect("400-10"), 99);
        assert!(profile_for(ApvChromaFormat::Mono, 12).is_err());
    }

    #[test]
    fn frame_geometry_and_planes_are_consistent() {
        let frame = ApvFrame::new(ApvChromaFormat::Yuv422, 10, 30, 18).expect("frame");
        assert_eq!(frame.planes.len(), 3);
        assert_eq!(frame.plane_size(0), Some((30, 18)));
        assert_eq!(frame.plane_size(1), Some((15, 18)));
        assert!(frame.is_valid());
        assert!(ApvFrame::new(ApvChromaFormat::Yuv422, 10, 31, 18).is_err());
        assert!(ApvFrame::new(ApvChromaFormat::Yuv422, 9, 32, 18).is_err());
    }
}
