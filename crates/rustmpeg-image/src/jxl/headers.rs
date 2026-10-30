// SPDX-License-Identifier: Apache-2.0
//! ISO/IEC 18181-1 Annex D, Annex E, and Annex F headers and Table of Contents (TOC).
//!
//! Implements both bitstream parsing and bit-exact header writing for:
//! * [`SizeHeader`], [`ImageMetadata`], and [`ImageHeader`] (Annex D);
//! * [`read_icc`] and [`decode_icc`] compressed ICC profile streams (Annex E);
//! * [`FrameHeader`], [`RestorationFilter`], and [`Toc`] (Annex F).

use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::num::Wrapping;

use super::JxlError;
use super::bits::{BitReader, BitWriter, U32Specifier, pack_signed, unpack_signed};
use super::entropy::{EntropyDecoder, EntropyEncoder, HybridUintConfig, read_permutation};

/// Image dimensions from ISO/IEC 18181-1 Annex D.2 (`SizeHeader`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeHeader {
    /// Coded width in pixels (before orientation).
    pub width: u32,
    /// Coded height in pixels (before orientation).
    pub height: u32,
}

impl SizeHeader {
    /// Parses `SizeHeader` from `reader`.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError`] on EOF or zero dimensions.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let div8 = reader.read_bool()?;
        let height = if div8 {
            8 * (1 + reader.read_bits(5)?)
        } else {
            reader.read_u32(
                U32Specifier::u(1, 9),
                U32Specifier::u(1, 13),
                U32Specifier::u(1, 18),
                U32Specifier::u(1, 30),
            )?
        };
        let ratio = reader.read_bits(3)?;
        let width = if ratio == 0 {
            if div8 {
                8 * (1 + reader.read_bits(5)?)
            } else {
                reader.read_u32(
                    U32Specifier::u(1, 9),
                    U32Specifier::u(1, 13),
                    U32Specifier::u(1, 18),
                    U32Specifier::u(1, 30),
                )?
            }
        } else {
            Self::compute_aspect_width(ratio, height)
        };
        if width == 0 || height == 0 {
            return Err(JxlError::InvalidBitstream(
                "SizeHeader has zero width or height",
            ));
        }
        Ok(Self { width, height })
    }

    /// Writes `SizeHeader` to `writer`.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::InvalidConfig`] if `width` or `height` is 0 or exceeds `2^30`.
    pub fn write(&self, writer: &mut BitWriter) -> Result<(), JxlError> {
        if self.width == 0 || self.height == 0 {
            return Err(JxlError::InvalidConfig(
                "SizeHeader dimensions must be non-zero",
            ));
        }
        if self.width.is_multiple_of(8)
            && self.height.is_multiple_of(8)
            && (self.width / 8) <= 32
            && (self.height / 8) <= 32
        {
            writer.write_bool(true); // div8 = true
            writer.write_bits(5, (self.height / 8) - 1);
            if self.width == self.height {
                writer.write_bits(3, 1); // ratio = 1 (1:1)
            } else {
                writer.write_bits(3, 0); // ratio = 0
                writer.write_bits(5, (self.width / 8) - 1);
            }
            return Ok(());
        }
        writer.write_bool(false); // div8 = false
        let spec0 = U32Specifier::u(1, 9);
        let spec1 = U32Specifier::u(1, 13);
        let spec2 = U32Specifier::u(1, 18);
        let spec3 = U32Specifier::u(1, 30);
        writer.write_u32(self.height, spec0, spec1, spec2, spec3)?;
        if self.width == self.height {
            writer.write_bits(3, 1);
        } else {
            writer.write_bits(3, 0);
            writer.write_u32(self.width, spec0, spec1, spec2, spec3)?;
        }
        Ok(())
    }

    /// Computes `width` from `height` and aspect `ratio` (`1..=7`) per Annex D.2.
    #[must_use]
    pub const fn compute_aspect_width(ratio: u32, height: u32) -> u32 {
        let h = height as u64;
        let w = match ratio {
            1 => h,
            2 => (h * 12) / 10,
            3 => (h * 4) / 3,
            4 => (h * 3) / 2,
            5 => (h * 16) / 9,
            6 => (h * 5) / 4,
            7 => h * 2,
            _ => 0,
        };
        w as u32
    }
}

/// Sample bit depth representation in `ImageMetadata` (`BitDepth` in ISO/IEC 18181-1 Annex D.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepthHeader {
    /// Unsigned/signed integer samples in `0..=(1 << bits_per_sample) - 1`.
    IntegerSample {
        /// Bits per sample (`1..=31`).
        bits_per_sample: u32,
    },
    /// Floating-point samples with `bits_per_sample` total bits and `exp_bits` exponent bits.
    FloatSample {
        /// Total bits per sample (`16`, `24`, `32`, etc.).
        bits_per_sample: u32,
        /// Exponent bits (`2..=8`).
        exp_bits: u32,
    },
}

impl Default for BitDepthHeader {
    fn default() -> Self {
        Self::IntegerSample { bits_per_sample: 8 }
    }
}

impl BitDepthHeader {
    /// Returns the number of bits per sample.
    #[must_use]
    #[inline]
    pub const fn bits_per_sample(self) -> u32 {
        match self {
            Self::IntegerSample { bits_per_sample }
            | Self::FloatSample {
                bits_per_sample, ..
            } => bits_per_sample,
        }
    }

    /// Converts a decoded Modular integer sample into normalized `f32` (`[0.0, 1.0]` nominal).
    #[must_use]
    #[inline]
    pub fn parse_integer_sample(self, sample: i32) -> f32 {
        match self {
            Self::IntegerSample { bits_per_sample } => {
                let max_val = ((1u64 << bits_per_sample) - 1) as f32;
                (sample as f32) / max_val
            }
            Self::FloatSample {
                bits_per_sample,
                exp_bits,
            } => {
                let sample = sample as u32;
                let mantissa_bits = bits_per_sample - exp_bits - 1;
                let mantissa_mask = (1u32 << mantissa_bits) - 1;
                let exp_mask = ((1u32 << (bits_per_sample - 1)) - 1) ^ mantissa_mask;
                let is_signed = (sample & (1u32 << (bits_per_sample - 1))) != 0;
                let mantissa = sample & mantissa_mask;
                let exp = ((sample & exp_mask) >> mantissa_bits) as i32;
                if exp == 0 && mantissa == 0 {
                    return if is_signed { -0.0 } else { 0.0 };
                }
                let unbiased_exp = exp - ((1i32 << (exp_bits - 1)) - 1);
                let f32_mant_bits = 23u32;
                let f32_mant = if mantissa_bits < f32_mant_bits {
                    mantissa << (f32_mant_bits - mantissa_bits)
                } else {
                    mantissa >> (mantissa_bits - f32_mant_bits)
                };
                let f32_exp = (unbiased_exp + 127).clamp(0, 255) as u32;
                let sign = u32::from(is_signed);
                f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
            }
        }
    }

    /// Parses `BitDepth` from `reader`.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        if reader.read_bool()? {
            let bits_per_sample = reader.read_u32(
                U32Specifier::c(32),
                U32Specifier::c(16),
                U32Specifier::c(24),
                U32Specifier::u(1, 6),
            )?;
            let exp_bits = reader.read_bits(4)? + 1;
            if !(2..=8).contains(&exp_bits) {
                return Err(JxlError::InvalidBitstream("invalid float exp_bits"));
            }
            let mantissa_bits = bits_per_sample.wrapping_sub(exp_bits + 1);
            if !(2..=23).contains(&mantissa_bits) {
                return Err(JxlError::InvalidBitstream("invalid float mantissa_bits"));
            }
            Ok(Self::FloatSample {
                bits_per_sample,
                exp_bits,
            })
        } else {
            let bits_per_sample = reader.read_u32(
                U32Specifier::c(8),
                U32Specifier::c(10),
                U32Specifier::c(12),
                U32Specifier::u(1, 6),
            )?;
            if bits_per_sample == 0 || bits_per_sample > 31 {
                return Err(JxlError::InvalidBitstream(
                    "invalid integer bits_per_sample",
                ));
            }
            Ok(Self::IntegerSample { bits_per_sample })
        }
    }

    /// Writes `BitDepth` to `writer`.
    pub fn write(self, writer: &mut BitWriter) -> Result<(), JxlError> {
        match self {
            Self::IntegerSample { bits_per_sample } => {
                writer.write_bool(false);
                writer.write_u32(
                    bits_per_sample,
                    U32Specifier::c(8),
                    U32Specifier::c(10),
                    U32Specifier::c(12),
                    U32Specifier::u(1, 6),
                )
            }
            Self::FloatSample {
                bits_per_sample,
                exp_bits,
            } => {
                writer.write_bool(true);
                writer.write_u32(
                    bits_per_sample,
                    U32Specifier::c(32),
                    U32Specifier::c(16),
                    U32Specifier::c(24),
                    U32Specifier::u(1, 6),
                )?;
                writer.write_bits(4, exp_bits.saturating_sub(1));
                Ok(())
            }
        }
    }
}

/// Extra channel type (`ExtraChannelType` in ISO/IEC 18181-1 Annex D.3.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtraChannelType {
    /// Alpha transparency channel.
    Alpha {
        /// Whether color channels are premultiplied by alpha.
        alpha_associated: bool,
    },
    /// Depth map channel.
    Depth,
    /// Spot color channel with sRGB tint and solidity.
    SpotColour {
        /// Red tint component.
        red: f32,
        /// Green tint component.
        green: f32,
        /// Blue tint component.
        blue: f32,
        /// Solidity factor.
        solidity: f32,
    },
    /// Selection mask channel.
    SelectionMask,
    /// CMYK Black (`K`) channel.
    Black,
    /// Bayer Color Filter Array channel.
    Cfa {
        /// CFA channel index.
        cfa_channel: u32,
    },
    /// Thermal channel.
    Thermal,
    /// Non-optional generic channel.
    NonOptional,
    /// Optional generic channel.
    Optional,
}

impl Default for ExtraChannelType {
    fn default() -> Self {
        Self::Alpha {
            alpha_associated: false,
        }
    }
}

/// Metadata for one extra channel (`ExtraChannelInfo` in ISO/IEC 18181-1 Annex D.3.5).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExtraChannelInfo {
    /// Extra channel type and parameters.
    pub ty: ExtraChannelType,
    /// Sample bit depth of this extra channel.
    pub bit_depth: BitDepthHeader,
    /// Subsampling exponent (`dim_shift`).
    pub dim_shift: u32,
    /// Optional UTF-8 channel name.
    pub name: String,
}

impl ExtraChannelInfo {
    /// Returns `true` when this channel is an alpha channel.
    #[must_use]
    #[inline]
    pub const fn is_alpha(&self) -> bool {
        matches!(self.ty, ExtraChannelType::Alpha { .. })
    }

    /// Returns `Some(alpha_associated)` if this is an alpha channel.
    #[must_use]
    #[inline]
    pub const fn alpha_associated(&self) -> Option<bool> {
        if let ExtraChannelType::Alpha { alpha_associated } = self.ty {
            Some(alpha_associated)
        } else {
            None
        }
    }

    /// Parses `ExtraChannelInfo` from `reader`.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        if reader.read_bool()? {
            return Ok(Self::default());
        }
        let raw_ty = reader.read_enum_raw()?;
        let bit_depth = BitDepthHeader::parse(reader)?;
        let dim_shift = reader.read_u32(
            U32Specifier::c(0),
            U32Specifier::c(3),
            U32Specifier::c(4),
            U32Specifier::u(1, 3),
        )?;
        let name = reader.read_name()?;
        let ty = match raw_ty {
            0 => ExtraChannelType::Alpha {
                alpha_associated: reader.read_bool()?,
            },
            1 => ExtraChannelType::Depth,
            2 => ExtraChannelType::SpotColour {
                red: reader.read_f16()?,
                green: reader.read_f16()?,
                blue: reader.read_f16()?,
                solidity: reader.read_f16()?,
            },
            3 => ExtraChannelType::SelectionMask,
            4 => ExtraChannelType::Black,
            5 => ExtraChannelType::Cfa {
                cfa_channel: reader.read_u32(
                    U32Specifier::c(1),
                    U32Specifier::u(0, 2),
                    U32Specifier::u(3, 4),
                    U32Specifier::u(19, 8),
                )?,
            },
            6 => ExtraChannelType::Thermal,
            15 => ExtraChannelType::NonOptional,
            16 => ExtraChannelType::Optional,
            _ => {
                return Err(JxlError::InvalidBitstream("unknown ExtraChannelType enum"));
            }
        };
        Ok(Self {
            ty,
            bit_depth,
            dim_shift,
            name,
        })
    }

    /// Writes `ExtraChannelInfo` to `writer`.
    pub fn write(&self, writer: &mut BitWriter) -> Result<(), JxlError> {
        if *self == Self::default() {
            writer.write_bool(true);
            return Ok(());
        }
        writer.write_bool(false);
        let raw_ty = match self.ty {
            ExtraChannelType::Alpha { .. } => 0,
            ExtraChannelType::Depth => 1,
            ExtraChannelType::SpotColour { .. } => 2,
            ExtraChannelType::SelectionMask => 3,
            ExtraChannelType::Black => 4,
            ExtraChannelType::Cfa { .. } => 5,
            ExtraChannelType::Thermal => 6,
            ExtraChannelType::NonOptional => 15,
            ExtraChannelType::Optional => 16,
        };
        writer.write_enum_raw(raw_ty)?;
        self.bit_depth.write(writer)?;
        writer.write_u32(
            self.dim_shift,
            U32Specifier::c(0),
            U32Specifier::c(3),
            U32Specifier::c(4),
            U32Specifier::u(1, 3),
        )?;
        writer.write_name(&self.name)?;
        match self.ty {
            ExtraChannelType::Alpha { alpha_associated } => {
                writer.write_bool(alpha_associated);
            }
            ExtraChannelType::SpotColour {
                red,
                green,
                blue,
                solidity,
            } => {
                writer.write_f16(red)?;
                writer.write_f16(green)?;
                writer.write_f16(blue)?;
                writer.write_f16(solidity)?;
            }
            ExtraChannelType::Cfa { cfa_channel } => {
                writer.write_u32(
                    cfa_channel,
                    U32Specifier::c(1),
                    U32Specifier::u(0, 2),
                    U32Specifier::u(3, 4),
                    U32Specifier::u(19, 8),
                )?;
            }
            _ => {}
        }
        Ok(())
    }
}

/// Colour space discriminant (`ColourSpace` in ISO/IEC 18181-1 Annex D.4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColourSpace {
    /// Tristimulus RGB (`0`).
    #[default]
    Rgb = 0,
    /// Grayscale luma (`1`).
    Grey = 1,
    /// Absolute perceptual XYB (`2`).
    Xyb = 2,
    /// Unknown colour space (`3`).
    Unknown = 3,
}

impl ColourSpace {
    fn from_raw(v: u32) -> Result<Self, JxlError> {
        match v {
            0 => Ok(Self::Rgb),
            1 => Ok(Self::Grey),
            2 => Ok(Self::Xyb),
            3 => Ok(Self::Unknown),
            _ => Err(JxlError::InvalidBitstream("invalid ColourSpace enum")),
        }
    }
}

/// Custom CIE 1931 `xy` chromaticity scaled by `1_000_000`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CustomXy {
    /// Scaled `x` coordinate.
    pub x: i32,
    /// Scaled `y` coordinate.
    pub y: i32,
}

impl CustomXy {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let d0 = U32Specifier::u(0, 19);
        let d1 = U32Specifier::u(524_288, 19);
        let d2 = U32Specifier::u(1_048_576, 20);
        let d3 = U32Specifier::u(2_097_152, 21);
        let x = unpack_signed(reader.read_u32(d0, d1, d2, d3)?);
        let y = unpack_signed(reader.read_u32(d0, d1, d2, d3)?);
        Ok(Self { x, y })
    }

    fn write(self, writer: &mut BitWriter) -> Result<(), JxlError> {
        let d0 = U32Specifier::u(0, 19);
        let d1 = U32Specifier::u(524_288, 19);
        let d2 = U32Specifier::u(1_048_576, 20);
        let d3 = U32Specifier::u(2_097_152, 21);
        writer.write_u32(pack_signed(self.x), d0, d1, d2, d3)?;
        writer.write_u32(pack_signed(self.y), d0, d1, d2, d3)
    }
}

/// White point (`WhitePoint` in ISO/IEC 18181-1 Annex D.4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WhitePoint {
    /// CIE Standard Illuminant D65 (`1`).
    #[default]
    D65,
    /// Custom chromaticity (`2`).
    Custom(CustomXy),
    /// Equal-energy illuminant E (`10`).
    E,
    /// DCI-P3 theater white point (`11`).
    Dci,
}

impl WhitePoint {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        match reader.read_enum_raw()? {
            1 => Ok(Self::D65),
            2 => Ok(Self::Custom(CustomXy::parse(reader)?)),
            10 => Ok(Self::E),
            11 => Ok(Self::Dci),
            _ => Err(JxlError::InvalidBitstream("invalid WhitePoint enum")),
        }
    }

    fn write(self, writer: &mut BitWriter) -> Result<(), JxlError> {
        match self {
            Self::D65 => writer.write_enum_raw(1),
            Self::Custom(xy) => {
                writer.write_enum_raw(2)?;
                xy.write(writer)
            }
            Self::E => writer.write_enum_raw(10),
            Self::Dci => writer.write_enum_raw(11),
        }
    }
}

/// Colour primaries (`Primaries` in ISO/IEC 18181-1 Annex D.4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Primaries {
    /// sRGB / BT.709 primaries (`1`).
    #[default]
    Srgb,
    /// Custom RGB primaries (`2`).
    Custom {
        /// Red primary.
        red: CustomXy,
        /// Green primary.
        green: CustomXy,
        /// Blue primary.
        blue: CustomXy,
    },
    /// BT.2100 / BT.2020 wide-gamut primaries (`9`).
    Bt2100,
    /// Display P3 / DCI-P3 primaries (`11`).
    P3,
}

impl Primaries {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        match reader.read_enum_raw()? {
            1 => Ok(Self::Srgb),
            2 => Ok(Self::Custom {
                red: CustomXy::parse(reader)?,
                green: CustomXy::parse(reader)?,
                blue: CustomXy::parse(reader)?,
            }),
            9 => Ok(Self::Bt2100),
            11 => Ok(Self::P3),
            _ => Err(JxlError::InvalidBitstream("invalid Primaries enum")),
        }
    }

    fn write(self, writer: &mut BitWriter) -> Result<(), JxlError> {
        match self {
            Self::Srgb => writer.write_enum_raw(1),
            Self::Custom { red, green, blue } => {
                writer.write_enum_raw(2)?;
                red.write(writer)?;
                green.write(writer)?;
                blue.write(writer)
            }
            Self::Bt2100 => writer.write_enum_raw(9),
            Self::P3 => writer.write_enum_raw(11),
        }
    }
}

/// Opto-electronic transfer function (`CustomTransferFunction` in ISO/IEC 18181-1 Annex D.4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TransferFunctionHeader {
    /// Explicit gamma value (`g` scaled by `1e7`, `g <= 10_000_000`).
    Gamma(u32),
    /// BT.709 transfer function (`1`).
    Bt709,
    /// Unknown transfer function (`2`).
    Unknown,
    /// Linear light (`8`).
    Linear,
    /// sRGB transfer function (`13`).
    #[default]
    Srgb,
    /// SMPTE ST 2084 PQ (`16`).
    Pq,
    /// DCI gamma 2.6 (`17`).
    Dci,
    /// ITU-R BT.2100 HLG (`18`).
    Hlg,
}

impl TransferFunctionHeader {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        if reader.read_bool()? {
            let g = reader.read_bits(24)?;
            if g == 0 || g > 10_000_000 {
                return Err(JxlError::InvalidBitstream("invalid custom gamma value"));
            }
            Ok(Self::Gamma(g))
        } else {
            match reader.read_enum_raw()? {
                1 => Ok(Self::Bt709),
                2 => Ok(Self::Unknown),
                8 => Ok(Self::Linear),
                13 => Ok(Self::Srgb),
                16 => Ok(Self::Pq),
                17 => Ok(Self::Dci),
                18 => Ok(Self::Hlg),
                _ => Err(JxlError::InvalidBitstream("invalid TransferFunction enum")),
            }
        }
    }

    fn write(self, writer: &mut BitWriter) -> Result<(), JxlError> {
        match self {
            Self::Gamma(g) => {
                writer.write_bool(true);
                writer.write_bits(24, g);
                Ok(())
            }
            other => {
                writer.write_bool(false);
                let code = match other {
                    Self::Bt709 => 1,
                    Self::Unknown => 2,
                    Self::Linear => 8,
                    Self::Srgb => 13,
                    Self::Pq => 16,
                    Self::Dci => 17,
                    Self::Hlg => 18,
                    Self::Gamma(_) => unreachable!(),
                };
                writer.write_enum_raw(code)
            }
        }
    }
}

/// ICC rendering intent (`RenderingIntent` in ISO/IEC 18181-1 Annex D.4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RenderingIntent {
    /// Perceptual (`0`).
    Perceptual = 0,
    /// Relative colorimetric (`1`).
    #[default]
    Relative = 1,
    /// Saturation (`2`).
    Saturation = 2,
    /// Absolute colorimetric (`3`).
    Absolute = 3,
}

impl RenderingIntent {
    fn from_raw(v: u32) -> Result<Self, JxlError> {
        match v {
            0 => Ok(Self::Perceptual),
            1 => Ok(Self::Relative),
            2 => Ok(Self::Saturation),
            3 => Ok(Self::Absolute),
            _ => Err(JxlError::InvalidBitstream("invalid RenderingIntent enum")),
        }
    }
}

/// Enumerated colour encoding (`ColourEncoding` when `want_icc == false`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EnumColourEncoding {
    /// Colour space (`Rgb`, `Grey`, `Xyb`, `Unknown`).
    pub colour_space: ColourSpace,
    /// White point.
    pub white_point: WhitePoint,
    /// Colour primaries.
    pub primaries: Primaries,
    /// Transfer function.
    pub tf: TransferFunctionHeader,
    /// Rendering intent.
    pub rendering_intent: RenderingIntent,
}

impl EnumColourEncoding {
    /// Standard sRGB colour encoding (`all_default = true`).
    #[must_use]
    pub const fn srgb() -> Self {
        Self {
            colour_space: ColourSpace::Rgb,
            white_point: WhitePoint::D65,
            primaries: Primaries::Srgb,
            tf: TransferFunctionHeader::Srgb,
            rendering_intent: RenderingIntent::Relative,
        }
    }

    /// Standard grayscale sRGB colour encoding.
    #[must_use]
    pub const fn gray_srgb() -> Self {
        Self {
            colour_space: ColourSpace::Grey,
            white_point: WhitePoint::D65,
            primaries: Primaries::Srgb,
            tf: TransferFunctionHeader::Srgb,
            rendering_intent: RenderingIntent::Relative,
        }
    }
}

/// Colour encoding of a JPEG XL image (`ColourEncoding` in ISO/IEC 18181-1 Annex D.4.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColourEncoding {
    /// Described by CICP-like enum values.
    Enum(EnumColourEncoding),
    /// Described by an embedded ICC profile stream in the codestream.
    IccProfile(ColourSpace),
}

impl Default for ColourEncoding {
    fn default() -> Self {
        Self::Enum(EnumColourEncoding::srgb())
    }
}

impl ColourEncoding {
    /// Returns `true` when an embedded ICC profile stream follows `ImageMetadata`.
    #[must_use]
    #[inline]
    pub const fn want_icc(&self) -> bool {
        matches!(self, Self::IccProfile(_))
    }

    /// Returns the declared [`ColourSpace`].
    #[must_use]
    #[inline]
    pub const fn colour_space(&self) -> ColourSpace {
        match self {
            Self::Enum(e) => e.colour_space,
            Self::IccProfile(cs) => *cs,
        }
    }

    /// Parses `ColourEncoding` from `reader`.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        if reader.read_bool()? {
            return Ok(Self::default());
        }
        let want_icc = reader.read_bool()?;
        let colour_space = ColourSpace::from_raw(reader.read_enum_raw()?)?;
        if want_icc {
            return Ok(Self::IccProfile(colour_space));
        }
        let white_point = if colour_space == ColourSpace::Xyb {
            WhitePoint::D65
        } else {
            WhitePoint::parse(reader)?
        };
        let primaries = if matches!(colour_space, ColourSpace::Xyb | ColourSpace::Grey) {
            Primaries::Srgb
        } else {
            Primaries::parse(reader)?
        };
        let tf = TransferFunctionHeader::parse(reader)?;
        let rendering_intent = RenderingIntent::from_raw(reader.read_enum_raw()?)?;
        Ok(Self::Enum(EnumColourEncoding {
            colour_space,
            white_point,
            primaries,
            tf,
            rendering_intent,
        }))
    }

    /// Writes `ColourEncoding` to `writer`.
    pub fn write(&self, writer: &mut BitWriter) -> Result<(), JxlError> {
        if *self == Self::default() {
            writer.write_bool(true);
            return Ok(());
        }
        writer.write_bool(false);
        match self {
            Self::IccProfile(cs) => {
                writer.write_bool(true);
                writer.write_enum_raw(*cs as u32)
            }
            Self::Enum(e) => {
                writer.write_bool(false);
                writer.write_enum_raw(e.colour_space as u32)?;
                if e.colour_space != ColourSpace::Xyb {
                    e.white_point.write(writer)?;
                }
                if !matches!(e.colour_space, ColourSpace::Xyb | ColourSpace::Grey) {
                    e.primaries.write(writer)?;
                }
                e.tf.write(writer)?;
                writer.write_enum_raw(e.rendering_intent as u32)
            }
        }
    }
}

/// HDR tone mapping metadata (`ToneMapping` in ISO/IEC 18181-1 Annex D.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneMapping {
    /// Peak display luminance in nits (default `255.0`).
    pub intensity_target: f32,
    /// Minimum luminance in nits (default `0.0`).
    pub min_nits: f32,
    /// Whether `linear_below` is relative to `intensity_target`.
    pub relative_to_max_display: bool,
    /// Threshold below which tone mapping stays linear.
    pub linear_below: f32,
}

impl Default for ToneMapping {
    fn default() -> Self {
        Self {
            intensity_target: 255.0,
            min_nits: 0.0,
            relative_to_max_display: false,
            linear_below: 0.0,
        }
    }
}

impl ToneMapping {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        if reader.read_bool()? {
            return Ok(Self::default());
        }
        let intensity_target = reader.read_f16()?;
        let min_nits = reader.read_f16()?;
        let relative_to_max_display = reader.read_bool()?;
        let linear_below = reader.read_f16()?;
        if intensity_target <= 0.0
            || min_nits < 0.0
            || min_nits > intensity_target
            || linear_below < 0.0
            || (relative_to_max_display && linear_below > 1.0)
        {
            return Err(JxlError::InvalidBitstream("invalid ToneMapping parameters"));
        }
        Ok(Self {
            intensity_target,
            min_nits,
            relative_to_max_display,
            linear_below,
        })
    }

    fn write(&self, writer: &mut BitWriter) -> Result<(), JxlError> {
        if *self == Self::default() {
            writer.write_bool(true);
            return Ok(());
        }
        writer.write_bool(false);
        writer.write_f16(self.intensity_target)?;
        writer.write_f16(self.min_nits)?;
        writer.write_bool(self.relative_to_max_display);
        writer.write_f16(self.linear_below)
    }
}

/// Opsin inverse matrix and quantization bias for XYB decoding (`OpsinInverseMatrix`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpsinInverseMatrix {
    /// 3x3 inverse matrix from mixed LMS to linear sRGB.
    pub inv_mat: [[f32; 3]; 3],
    /// Cube-root opsin bias per channel.
    pub opsin_bias: [f32; 3],
    /// Dequantization bias for `|q| == 1` per channel.
    pub quant_bias: [f32; 3],
    /// Dequantization bias numerator for `|q| > 1`.
    pub quant_bias_numerator: f32,
}

impl Default for OpsinInverseMatrix {
    #[allow(clippy::excessive_precision)]
    fn default() -> Self {
        Self {
            inv_mat: [
                [11.031566901960783, -9.866943921568629, -0.16462299647058826],
                [-3.254147380392157, 4.418770392156863, -0.16462299647058826],
                [-3.6588512862745097, 2.7129230470588235, 1.9459282392156863],
            ],
            opsin_bias: [-0.0037930732552754493; 3],
            quant_bias: [
                1.0 - 0.05465007330715401,
                1.0 - 0.07005449891748593,
                1.0 - 0.049935103337343655,
            ],
            quant_bias_numerator: 0.145,
        }
    }
}

impl OpsinInverseMatrix {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        if reader.read_bool()? {
            return Ok(Self::default());
        }
        let mut inv_mat = [[0.0f32; 3]; 3];
        for row in &mut inv_mat {
            for cell in row {
                *cell = reader.read_f16()?;
            }
        }
        let mut opsin_bias = [0.0f32; 3];
        for b in &mut opsin_bias {
            *b = reader.read_f16()?;
        }
        let mut quant_bias = [0.0f32; 3];
        for q in &mut quant_bias {
            *q = reader.read_f16()?;
        }
        let quant_bias_numerator = reader.read_f16()?;
        Ok(Self {
            inv_mat,
            opsin_bias,
            quant_bias,
            quant_bias_numerator,
        })
    }
}

/// Animation timing header (`AnimationHeader` in ISO/IEC 18181-1 Annex D.3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationHeader {
    /// Ticks per second numerator.
    pub tps_numerator: u32,
    /// Ticks per second denominator.
    pub tps_denominator: u32,
    /// Loop count (`0` means infinite).
    pub num_loops: u32,
    /// Whether frames carry 32-bit timecodes.
    pub have_timecodes: bool,
}

impl AnimationHeader {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let tps_numerator = reader.read_u32(
            U32Specifier::c(100),
            U32Specifier::c(1000),
            U32Specifier::u(1, 10),
            U32Specifier::u(1, 30),
        )?;
        let tps_denominator = reader.read_u32(
            U32Specifier::c(1),
            U32Specifier::c(1001),
            U32Specifier::u(1, 8),
            U32Specifier::u(1, 10),
        )?;
        let num_loops = reader.read_u32(
            U32Specifier::c(0),
            U32Specifier::u(0, 3),
            U32Specifier::u(0, 16),
            U32Specifier::u(0, 32),
        )?;
        let have_timecodes = reader.read_bool()?;
        Ok(Self {
            tps_numerator,
            tps_denominator,
            num_loops,
            have_timecodes,
        })
    }

    fn write(&self, writer: &mut BitWriter) -> Result<(), JxlError> {
        writer.write_u32(
            self.tps_numerator,
            U32Specifier::c(100),
            U32Specifier::c(1000),
            U32Specifier::u(1, 10),
            U32Specifier::u(1, 30),
        )?;
        writer.write_u32(
            self.tps_denominator,
            U32Specifier::c(1),
            U32Specifier::c(1001),
            U32Specifier::u(1, 8),
            U32Specifier::u(1, 10),
        )?;
        writer.write_u32(
            self.num_loops,
            U32Specifier::c(0),
            U32Specifier::u(0, 3),
            U32Specifier::u(0, 16),
            U32Specifier::u(0, 32),
        )?;
        writer.write_bool(self.have_timecodes);
        Ok(())
    }
}

/// Preview dimensions (`PreviewHeader` in ISO/IEC 18181-1 Annex D.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewHeader {
    /// Preview width in pixels.
    pub width: u32,
    /// Preview height in pixels.
    pub height: u32,
}

impl PreviewHeader {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let div8 = reader.read_bool()?;
        let height = if div8 {
            8 * reader.read_u32(
                U32Specifier::c(16),
                U32Specifier::c(32),
                U32Specifier::u(1, 5),
                U32Specifier::u(33, 9),
            )?
        } else {
            reader.read_u32(
                U32Specifier::u(1, 6),
                U32Specifier::u(65, 8),
                U32Specifier::u(321, 10),
                U32Specifier::u(1345, 12),
            )?
        };
        let ratio = reader.read_bits(3)?;
        let width = if ratio == 0 {
            if div8 {
                8 * reader.read_u32(
                    U32Specifier::c(16),
                    U32Specifier::c(32),
                    U32Specifier::u(1, 5),
                    U32Specifier::u(33, 9),
                )?
            } else {
                reader.read_u32(
                    U32Specifier::u(1, 6),
                    U32Specifier::u(65, 8),
                    U32Specifier::u(321, 10),
                    U32Specifier::u(1345, 12),
                )?
            }
        } else {
            SizeHeader::compute_aspect_width(ratio, height)
        };
        Ok(Self { width, height })
    }
}

/// Image metadata bundle (`ImageMetadata` in ISO/IEC 18181-1 Annex D.3).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageMetadata {
    /// EXIF-style orientation (`1..=8`, `1` = normal).
    pub orientation: u32,
    /// Optional intrinsic display dimensions.
    pub intrinsic_size: Option<SizeHeader>,
    /// Optional preview frame header.
    pub preview: Option<PreviewHeader>,
    /// Optional animation timing parameters.
    pub animation: Option<AnimationHeader>,
    /// Sample bit depth.
    pub bit_depth: BitDepthHeader,
    /// Whether 16-bit buffers are sufficient for Modular decoding.
    pub modular_16bit_buffers: bool,
    /// Extra channel descriptors (alpha, depth, spot, black, etc.).
    pub ec_info: Vec<ExtraChannelInfo>,
    /// Whether the image is coded in the perceptual XYB colour space.
    pub xyb_encoded: bool,
    /// Colour encoding (enum or embedded ICC profile).
    pub colour_encoding: ColourEncoding,
    /// HDR tone mapping parameters.
    pub tone_mapping: ToneMapping,
    /// Opsin inverse matrix for XYB decoding.
    pub opsin_inverse_matrix: OpsinInverseMatrix,
}

impl Default for ImageMetadata {
    fn default() -> Self {
        Self {
            orientation: 1,
            intrinsic_size: None,
            preview: None,
            animation: None,
            bit_depth: BitDepthHeader::default(),
            modular_16bit_buffers: true,
            ec_info: Vec::new(),
            xyb_encoded: true,
            colour_encoding: ColourEncoding::default(),
            tone_mapping: ToneMapping::default(),
            opsin_inverse_matrix: OpsinInverseMatrix::default(),
        }
    }
}

impl ImageMetadata {
    /// Returns `true` when the primary colour space is grayscale (`Grey`).
    #[must_use]
    #[inline]
    pub const fn grayscale(&self) -> bool {
        matches!(self.colour_encoding.colour_space(), ColourSpace::Grey)
    }

    /// Applies `orientation` (`1..=8`) to `(width, height)`.
    #[must_use]
    #[inline]
    pub const fn oriented_dimensions(&self, width: u32, height: u32) -> (u32, u32) {
        if self.orientation >= 5 {
            (height, width)
        } else {
            (width, height)
        }
    }

    /// Parses `ImageMetadata` from `reader`.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let all_default = reader.read_bool()?;
        let mut meta = Self::default();
        if !all_default {
            let extra_fields = reader.read_bool()?;
            if extra_fields {
                meta.orientation = 1 + reader.read_bits(3)?;
                if reader.read_bool()? {
                    meta.intrinsic_size = Some(SizeHeader::parse(reader)?);
                }
                if reader.read_bool()? {
                    meta.preview = Some(PreviewHeader::parse(reader)?);
                }
                if reader.read_bool()? {
                    meta.animation = Some(AnimationHeader::parse(reader)?);
                }
            }
            meta.bit_depth = BitDepthHeader::parse(reader)?;
            meta.modular_16bit_buffers = reader.read_bool()?;
            let num_extra = reader.read_u32(
                U32Specifier::c(0),
                U32Specifier::c(1),
                U32Specifier::u(2, 4),
                U32Specifier::u(1, 12),
            )?;
            if num_extra > 256 {
                return Err(JxlError::LimitExceeded("num_extra > 256"));
            }
            meta.ec_info.reserve(num_extra as usize);
            for _ in 0..num_extra {
                meta.ec_info.push(ExtraChannelInfo::parse(reader)?);
            }
            meta.xyb_encoded = reader.read_bool()?;
            meta.colour_encoding = ColourEncoding::parse(reader)?;
            if extra_fields {
                meta.tone_mapping = ToneMapping::parse(reader)?;
            }
            skip_extensions(reader)?;
        }
        let default_m = reader.read_bool()?;
        if !default_m {
            if meta.xyb_encoded {
                meta.opsin_inverse_matrix = OpsinInverseMatrix::parse(reader)?;
            }
            let cw_mask = reader.read_bits(3)?;
            if (cw_mask & 1) != 0 {
                for _ in 0..15 {
                    let _ = reader.read_f16()?;
                }
            }
            if (cw_mask & 2) != 0 {
                for _ in 0..55 {
                    let _ = reader.read_f16()?;
                }
            }
            if (cw_mask & 4) != 0 {
                for _ in 0..210 {
                    let _ = reader.read_f16()?;
                }
            }
        }
        Ok(meta)
    }

    /// Writes `ImageMetadata` to `writer`.
    pub fn write(&self, writer: &mut BitWriter) -> Result<(), JxlError> {
        let is_all_default = *self == Self::default();
        if is_all_default {
            writer.write_bool(true); // all_default = true
            writer.write_bool(true); // default_m = true
            return Ok(());
        }
        writer.write_bool(false); // all_default = false
        let need_extra = self.orientation != 1
            || self.intrinsic_size.is_some()
            || self.preview.is_some()
            || self.animation.is_some()
            || self.tone_mapping != ToneMapping::default();
        writer.write_bool(need_extra);
        if need_extra {
            writer.write_bits(3, self.orientation.saturating_sub(1));
            if let Some(intr) = &self.intrinsic_size {
                writer.write_bool(true);
                intr.write(writer)?;
            } else {
                writer.write_bool(false);
            }
            writer.write_bool(false); // have_preview = false
            if let Some(anim) = &self.animation {
                writer.write_bool(true);
                anim.write(writer)?;
            } else {
                writer.write_bool(false);
            }
        }
        self.bit_depth.write(writer)?;
        writer.write_bool(self.modular_16bit_buffers);
        let num_extra = u32::try_from(self.ec_info.len())
            .map_err(|_| JxlError::InvalidConfig("too many extra channels"))?;
        writer.write_u32(
            num_extra,
            U32Specifier::c(0),
            U32Specifier::c(1),
            U32Specifier::u(2, 4),
            U32Specifier::u(1, 12),
        )?;
        for ec in &self.ec_info {
            ec.write(writer)?;
        }
        writer.write_bool(self.xyb_encoded);
        self.colour_encoding.write(writer)?;
        if need_extra {
            self.tone_mapping.write(writer)?;
        }
        writer.write_u64(0); // extensions = 0
        writer.write_bool(true); // default_m = true
        Ok(())
    }
}

/// Top-level JPEG XL codestream header (`0xFF 0x0A` + `SizeHeader` + `ImageMetadata`).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageHeader {
    /// Coded canvas dimensions.
    pub size: SizeHeader,
    /// Image metadata.
    pub metadata: ImageMetadata,
}

impl ImageHeader {
    /// Parses `ImageHeader` (including the 16-bit `0x0AFF` LSB-first signature) from `reader`.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let sig = reader.read_bits(16)?;
        if sig != 0x0aff {
            return Err(JxlError::InvalidSignature);
        }
        let size = SizeHeader::parse(reader)?;
        let metadata = ImageMetadata::parse(reader)?;
        Ok(Self { size, metadata })
    }

    /// Writes `ImageHeader` (including the 16-bit `0x0AFF` signature) to `writer`.
    pub fn write(&self, writer: &mut BitWriter) -> Result<(), JxlError> {
        writer.write_bits(16, 0x0aff);
        self.size.write(writer)?;
        self.metadata.write(writer)
    }

    /// Visible width after applying `metadata.orientation`.
    #[must_use]
    #[inline]
    pub const fn width_with_orientation(&self) -> u32 {
        self.metadata
            .oriented_dimensions(self.size.width, self.size.height)
            .0
    }

    /// Visible height after applying `metadata.orientation`.
    #[must_use]
    #[inline]
    pub const fn height_with_orientation(&self) -> u32 {
        self.metadata
            .oriented_dimensions(self.size.width, self.size.height)
            .1
    }
}

fn skip_extensions(reader: &mut BitReader<'_>) -> Result<(), JxlError> {
    let mut ext_bits = reader.read_u64()?;
    if ext_bits == 0 {
        return Ok(());
    }
    let mut lengths = Vec::with_capacity(ext_bits.count_ones() as usize);
    for _ in 0..64 {
        if (ext_bits & 1) != 0 {
            lengths.push(reader.read_u64()?);
        }
        ext_bits >>= 1;
    }
    for len in lengths {
        let bits = usize::try_from(len)
            .map_err(|_| JxlError::LimitExceeded("extension bit length overflow"))?;
        reader.skip_bits(bits)?;
    }
    Ok(())
}

// ============================================================================
// ISO/IEC 18181-1 Annex E: Compressed ICC Profile Stream Decoder
// ============================================================================

/// Reads and entropy-decodes the compressed ICC stream from `reader`.
///
/// # Errors
///
/// Returns [`JxlError`] if the ICC stream exceeds limits or fails entropy decoding.
pub fn read_icc(reader: &mut BitReader<'_>) -> Result<Vec<u8>, JxlError> {
    let enc_size = reader.read_u64()?;
    if enc_size > (1u64 << 24) {
        return Err(JxlError::LimitExceeded("encoded ICC stream too large"));
    }
    let mut decoder = EntropyDecoder::parse(reader, 41)?;
    let enc_len = enc_size as usize;
    let mut encoded = vec![0u8; enc_len];
    let mut b1 = 0u8;
    let mut b2 = 0u8;
    decoder.begin(reader)?;
    for (idx, slot) in encoded.iter_mut().enumerate() {
        let sym = decoder.read_varint(reader, icc_context(idx, b1, b2))?;
        if sym >= 256 {
            return Err(JxlError::InvalidBitstream("ICC stream byte symbol >= 256"));
        }
        *slot = sym as u8;
        b2 = b1;
        b1 = *slot;
    }
    decoder.finalize()?;
    decode_icc(&encoded)
}

/// Compresses and entropy-encodes raw ICC profile bytes into `writer` (ISO/IEC 18181-1 Annex E).
pub fn write_icc(writer: &mut BitWriter, icc_bytes: &[u8]) -> Result<(), JxlError> {
    let encoded = encode_icc(icc_bytes);
    writer.write_u64(encoded.len() as u64);
    let mut enc = EntropyEncoder::single_cluster(41, HybridUintConfig::DEFAULT_4_2_0);
    let mut b1 = 0u8;
    let mut b2 = 0u8;
    for (idx, &b) in encoded.iter().enumerate() {
        enc.record_varint(icc_context(idx, b1, b2), u32::from(b));
        b2 = b1;
        b1 = b;
    }
    enc.finish(writer)
}

/// Encodes raw ICCv4 profile bytes into an ISO/IEC 18181-1 Annex E command/data stream.
#[must_use]
pub fn encode_icc(icc_bytes: &[u8]) -> Vec<u8> {
    let output_size = icc_bytes.len();
    let header_size = output_size.min(128);
    let mut header_data = vec![0u8; header_size];
    for idx in 0..header_size {
        let p = predict_icc_header(idx, output_size as u32, &header_data);
        header_data[idx] = icc_bytes[idx].wrapping_sub(p);
    }

    let mut commands = Vec::new();
    let mut body_data = Vec::new();
    if output_size > 128 {
        write_icc_varint(&mut commands, 0); // v = 0 -> no tag directory table
        commands.push(1); // command 1 = literal bytes run
        let rem = output_size - 128;
        write_icc_varint(&mut commands, rem as u64);
        body_data.extend_from_slice(&icc_bytes[128..]);
    }

    let mut stream = Vec::with_capacity(16 + commands.len() + header_data.len() + body_data.len());
    write_icc_varint(&mut stream, output_size as u64);
    write_icc_varint(&mut stream, commands.len() as u64);
    stream.extend_from_slice(&commands);
    stream.extend_from_slice(&header_data);
    stream.extend_from_slice(&body_data);
    stream
}

fn write_icc_varint(out: &mut Vec<u8>, mut val: u64) {
    loop {
        let byte = (val & 0x7f) as u8;
        val >>= 7;
        if val == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

fn icc_context(idx: usize, b1: u8, b2: u8) -> u32 {
    if idx <= 128 {
        return 0;
    }
    let p1 = match b1 {
        b'a'..=b'z' | b'A'..=b'Z' => 0,
        b'0'..=b'9' | b'.' | b',' => 1,
        0..=1 => 2 + u32::from(b1),
        2..=15 => 4,
        241..=254 => 5,
        255 => 6,
        _ => 7,
    };
    let p2 = match b2 {
        b'a'..=b'z' | b'A'..=b'Z' => 0,
        b'0'..=b'9' | b'.' | b',' => 1,
        0..=15 => 2,
        241..=255 => 3,
        _ => 4,
    };
    1 + p1 + 8 * p2
}

fn read_icc_varint(cursor: &mut Cursor<&[u8]>) -> Result<u64, JxlError> {
    let mut value = 0u64;
    let mut shift = 0u32;
    let mut b = 0u8;
    while shift < 63 {
        cursor
            .read_exact(std::slice::from_mut(&mut b))
            .map_err(|_| JxlError::InvalidBitstream("truncated ICC varint"))?;
        value |= u64::from(b & 0x7f) << shift;
        if (b & 0x80) == 0 {
            break;
        }
        shift += 7;
    }
    Ok(value)
}

fn predict_icc_header(idx: usize, output_size: u32, header: &[u8]) -> u8 {
    match idx {
        0..=3 => output_size.to_be_bytes()[idx],
        8 => 4,
        12..=23 => b"mntrRGB XYZ "[idx - 12],
        36..=39 => b"acsp"[idx - 36],
        41 | 42 if header[40] == b'A' => b'P',
        43 if header[40] == b'A' => b'L',
        41 if header[40] == b'M' => b'S',
        42 if header[40] == b'M' => b'F',
        43 if header[40] == b'M' => b'T',
        42 if header[40] == b'S' && header[41] == b'G' => b'I',
        43 if header[40] == b'S' && header[41] == b'G' => b' ',
        42 if header[40] == b'S' && header[41] == b'U' => b'N',
        43 if header[40] == b'S' && header[41] == b'U' => b'W',
        70 => 246,
        71 => 214,
        73 => 1,
        78 => 211,
        79 => 45,
        80..=83 => header[4 + idx - 80],
        _ => 0,
    }
}

fn icc_shuffle2(bytes: &[u8]) -> Vec<u8> {
    let len = bytes.len();
    let mut out = Vec::with_capacity(len);
    let height = len / 2;
    let odd = len % 2;
    for idx in 0..height {
        out.push(bytes[idx]);
        out.push(bytes[idx + height + odd]);
    }
    if odd != 0 {
        out.push(bytes[height]);
    }
    out
}

fn icc_shuffle4(bytes: &[u8]) -> Vec<u8> {
    let len = bytes.len();
    let mut out = Vec::with_capacity(len);
    let step = len / 4;
    let wide_count = len % 4;
    for idx in 0..step {
        let mut base = idx;
        for _ in 0..wide_count {
            out.push(bytes[base]);
            base += step + 1;
        }
        for _ in wide_count..4 {
            out.push(bytes[base]);
            base += step;
        }
    }
    for idx in 1..=wide_count {
        out.push(bytes[(step + 1) * idx - 1]);
    }
    out
}

/// Reconstructs raw ICCv4 profile bytes from a decoded ICC command/data stream
/// (ISO/IEC 18181-1 Annex E).
pub fn decode_icc(stream: &[u8]) -> Result<Vec<u8>, JxlError> {
    const COMMON_TAGS: [&[u8]; 19] = [
        b"rTRC", b"rXYZ", b"cprt", b"wtpt", b"bkpt", b"rXYZ", b"gXYZ", b"bXYZ", b"kXYZ", b"rTRC",
        b"gTRC", b"bTRC", b"kTRC", b"chad", b"desc", b"chrm", b"dmnd", b"dmdd", b"lumi",
    ];
    const COMMON_DATA: [&[u8]; 8] = [
        b"XYZ ", b"desc", b"text", b"mluc", b"para", b"curv", b"sf32", b"gbd ",
    ];

    let mut tmp = Cursor::new(stream);
    let output_size = read_icc_varint(&mut tmp)?;
    let commands_size = read_icc_varint(&mut tmp)?;
    let offset = tmp.position();
    if offset + commands_size > stream.len() as u64 {
        return Err(JxlError::InvalidBitstream("invalid ICC commands_size"));
    }
    if output_size > (1u64 << 24) {
        return Err(JxlError::LimitExceeded("ICC output_size too large"));
    }
    let (commands, data) = stream[offset as usize..].split_at(commands_size as usize);
    let header_size = output_size.min(128) as usize;
    if data.len() < header_size {
        return Err(JxlError::InvalidBitstream(
            "ICC data stream shorter than header",
        ));
    }
    let (header_data, mut data) = data.split_at(header_size);
    let mut commands_stream = Cursor::new(commands);
    let mut out = Vec::with_capacity(output_size as usize);

    for (idx, &e) in header_data.iter().enumerate() {
        let p = predict_icc_header(idx, output_size as u32, header_data);
        out.push(p.wrapping_add(e));
    }
    if output_size <= 128 {
        return Ok(out);
    }

    let v = read_icc_varint(&mut commands_stream)?;
    if let Some(num_tags) = v.checked_sub(1) {
        if (output_size - 128) / 12 < num_tags {
            return Err(JxlError::InvalidBitstream("ICC num_tags too large"));
        }
        let num_tags = num_tags as u32;
        out.extend_from_slice(&num_tags.to_be_bytes());
        let mut prev_tagstart = num_tags * 12 + 128;
        let mut prev_tagsize = 0u32;

        loop {
            let mut command = 0u8;
            if commands_stream
                .read_exact(std::slice::from_mut(&mut command))
                .is_err()
            {
                return Ok(out);
            }
            let tagcode = command & 63;
            let tag = match tagcode {
                0 => break,
                1 => {
                    if data.len() < 4 {
                        return Err(JxlError::InvalidBitstream("ICC truncated tag name"));
                    }
                    let (t, rest) = data.split_at(4);
                    data = rest;
                    t
                }
                2..=20 => COMMON_TAGS[(tagcode - 2) as usize],
                _ => return Err(JxlError::InvalidBitstream("invalid ICC tagcode")),
            };
            let tagstart = if (command & 64) == 0 {
                prev_tagstart + prev_tagsize
            } else {
                read_icc_varint(&mut commands_stream)? as u32
            };
            let tagsize = match tag {
                _ if (command & 128) != 0 => read_icc_varint(&mut commands_stream)? as u32,
                b"rXYZ" | b"gXYZ" | b"bXYZ" | b"kXYZ" | b"wtpt" | b"bkpt" | b"lumi" => 20,
                _ => prev_tagsize,
            };
            if u64::from(tagstart) + u64::from(tagsize) > output_size {
                return Err(JxlError::InvalidBitstream("ICC tag exceeds output_size"));
            }
            prev_tagstart = tagstart;
            prev_tagsize = tagsize;

            out.extend_from_slice(tag);
            out.extend_from_slice(&tagstart.to_be_bytes());
            out.extend_from_slice(&tagsize.to_be_bytes());
            if tagcode == 2 {
                for extra_tag in [b"gTRC", b"bTRC"] {
                    out.extend_from_slice(extra_tag);
                    out.extend_from_slice(&tagstart.to_be_bytes());
                    out.extend_from_slice(&tagsize.to_be_bytes());
                }
            } else if tagcode == 3 {
                out.extend_from_slice(b"gXYZ");
                out.extend_from_slice(&(tagstart + tagsize).to_be_bytes());
                out.extend_from_slice(&tagsize.to_be_bytes());
                out.extend_from_slice(b"bXYZ");
                out.extend_from_slice(&(tagstart + tagsize * 2).to_be_bytes());
                out.extend_from_slice(&tagsize.to_be_bytes());
            }
        }
    }

    let mut command = 0u8;
    while commands_stream
        .read_exact(std::slice::from_mut(&mut command))
        .is_ok()
    {
        match command {
            1 => {
                let num = read_icc_varint(&mut commands_stream)? as usize;
                if num > data.len() {
                    return Err(JxlError::InvalidBitstream("ICC literal run out of bounds"));
                }
                let (bytes, rest) = data.split_at(num);
                data = rest;
                out.extend_from_slice(bytes);
            }
            2 | 3 => {
                let num = read_icc_varint(&mut commands_stream)? as usize;
                if num > data.len() {
                    return Err(JxlError::InvalidBitstream("ICC shuffle run out of bounds"));
                }
                let (bytes, rest) = data.split_at(num);
                data = rest;
                let shuffled = if command == 2 {
                    icc_shuffle2(bytes)
                } else {
                    icc_shuffle4(bytes)
                };
                out.extend_from_slice(&shuffled);
            }
            4 => {
                let mut flags = 0u8;
                commands_stream
                    .read_exact(std::slice::from_mut(&mut flags))
                    .map_err(|_| JxlError::InvalidBitstream("ICC truncated command 4 flags"))?;
                let width = ((flags & 3) + 1) as usize;
                let order = (flags >> 2) & 3;
                if width == 3 || order == 3 {
                    return Err(JxlError::InvalidBitstream(
                        "invalid ICC command 4 width/order",
                    ));
                }
                let stride = if (flags & 16) == 0 {
                    width
                } else {
                    let s = read_icc_varint(&mut commands_stream)? as usize;
                    if s < width {
                        return Err(JxlError::InvalidBitstream("ICC stride < width"));
                    }
                    s
                };
                if stride.saturating_mul(4) >= out.len() {
                    return Err(JxlError::InvalidBitstream("ICC stride * 4 >= out.len()"));
                }
                let num = read_icc_varint(&mut commands_stream)? as usize;
                if data.len() < num {
                    return Err(JxlError::InvalidBitstream("ICC command 4 data truncated"));
                }
                let (bytes, rest) = data.split_at(num);
                data = rest;
                let shuffled;
                let bytes = match width {
                    1 => bytes,
                    2 => {
                        shuffled = icc_shuffle2(bytes);
                        &shuffled
                    }
                    4 => {
                        shuffled = icc_shuffle4(bytes);
                        &shuffled
                    }
                    _ => unreachable!(),
                };
                for i in (0..num).step_by(width) {
                    let mut prev = [Wrapping(0u32); 3];
                    for (j, p) in prev[..=order as usize].iter_mut().enumerate() {
                        let off = out.len() - stride * (j + 1);
                        let mut b4 = [0u8; 4];
                        b4[(4 - width)..].copy_from_slice(&out[off..][..width]);
                        p.0 = u32::from_be_bytes(b4);
                    }
                    let pred = match order {
                        0 => prev[0],
                        1 => Wrapping(2) * prev[0] - prev[1],
                        2 => Wrapping(3) * (prev[0] - prev[1]) + prev[2],
                        _ => unreachable!(),
                    };
                    for j in 0..width.min(num - i) {
                        let val =
                            Wrapping(u32::from(bytes[i + j])) + (pred >> (8 * (width - 1 - j)));
                        out.push(val.0 as u8);
                    }
                }
            }
            10 => {
                if data.len() < 12 {
                    return Err(JxlError::InvalidBitstream("ICC XYZ command truncated"));
                }
                out.extend_from_slice(b"XYZ \0\0\0\0");
                let (bytes, rest) = data.split_at(12);
                data = rest;
                out.extend_from_slice(bytes);
            }
            16..=23 => {
                out.extend_from_slice(COMMON_DATA[(command - 16) as usize]);
                out.extend_from_slice(&[0, 0, 0, 0]);
            }
            _ => {
                return Err(JxlError::InvalidBitstream("unknown ICC command opcode"));
            }
        }
    }
    if out.len() != output_size as usize {
        return Err(JxlError::InvalidBitstream("decoded ICC size mismatch"));
    }
    Ok(out)
}

// ============================================================================
// ISO/IEC 18181-1 Annex F: FrameHeader, RestorationFilter, and TOC
// ============================================================================

/// Frame type (`FrameType` in ISO/IEC 18181-1 Annex F.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameType {
    /// Displayed frame (`0`).
    #[default]
    RegularFrame = 0,
    /// 1:8 downsampled LF frame (`1`).
    LfFrame = 1,
    /// Reference-only hidden frame (`2`).
    ReferenceOnly = 2,
    /// Coalesced frame that skips progressive rendering (`3`).
    SkipProgressive = 3,
}

impl FrameType {
    /// Returns `true` for `RegularFrame` and `SkipProgressive`.
    #[must_use]
    #[inline]
    pub const fn is_normal_frame(self) -> bool {
        matches!(self, Self::RegularFrame | Self::SkipProgressive)
    }
}

/// Frame coding mode (`Encoding` in ISO/IEC 18181-1 Annex F.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameEncoding {
    /// Variable-block DCT mode (`0`).
    #[default]
    VarDct = 0,
    /// Integer Modular mode (`1`).
    Modular = 1,
}

/// Bitmask flags in `FrameHeader` (`Flags` in ISO/IEC 18181-1 Annex F.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameFlags(pub u64);

impl FrameFlags {
    /// Synthetic noise enabled (`0x01`).
    pub const NOISE: u64 = 0x01;
    /// Patch dictionary references enabled (`0x02`).
    pub const PATCHES: u64 = 0x02;
    /// Centripetal Catmull-Rom splines enabled (`0x10`).
    pub const SPLINES: u64 = 0x10;
    /// Uses a previously decoded `LfFrame` (`0x20`).
    pub const USE_LF_FRAME: u64 = 0x20;
    /// Skips adaptive LF smoothing in VarDCT (`0x80`).
    pub const SKIP_ADAPTIVE_LF_SMOOTHING: u64 = 0x80;

    /// Returns `true` if synthetic noise is enabled.
    #[must_use]
    #[inline]
    pub const fn noise(self) -> bool {
        (self.0 & Self::NOISE) != 0
    }

    /// Returns `true` if patches are enabled.
    #[must_use]
    #[inline]
    pub const fn patches(self) -> bool {
        (self.0 & Self::PATCHES) != 0
    }

    /// Returns `true` if splines are enabled.
    #[must_use]
    #[inline]
    pub const fn splines(self) -> bool {
        (self.0 & Self::SPLINES) != 0
    }

    /// Returns `true` if `USE_LF_FRAME` is set.
    #[must_use]
    #[inline]
    pub const fn use_lf_frame(self) -> bool {
        (self.0 & Self::USE_LF_FRAME) != 0
    }

    /// Returns `true` if `SKIP_ADAPTIVE_LF_SMOOTHING` is set.
    #[must_use]
    #[inline]
    pub const fn skip_adaptive_lf_smoothing(self) -> bool {
        (self.0 & Self::SKIP_ADAPTIVE_LF_SMOOTHING) != 0
    }
}

/// Multi-pass progressive configuration (`Passes` in ISO/IEC 18181-1 Annex F.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Passes {
    /// Number of HF passes (`1..=11`).
    pub num_passes: u32,
    /// Number of downsampling steps.
    pub num_ds: u32,
    /// Coefficient bit-shift per pass (`num_passes - 1` entries).
    pub shift: Vec<u32>,
    /// Downsampling factor per step (`1`, `2`, `4`, or `8`).
    pub downsample: Vec<u32>,
    /// Last pass index for each downsampling step.
    pub last_pass: Vec<u32>,
}

impl Default for Passes {
    fn default() -> Self {
        Self {
            num_passes: 1,
            num_ds: 0,
            shift: Vec::new(),
            downsample: Vec::new(),
            last_pass: Vec::new(),
        }
    }
}

impl Passes {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        let num_passes = reader.read_u32(
            U32Specifier::c(1),
            U32Specifier::c(2),
            U32Specifier::c(3),
            U32Specifier::u(4, 3),
        )?;
        if num_passes == 1 {
            return Ok(Self::default());
        }
        let num_ds = reader.read_u32(
            U32Specifier::c(0),
            U32Specifier::c(1),
            U32Specifier::c(2),
            U32Specifier::u(3, 1),
        )?;
        let mut shift = Vec::with_capacity((num_passes - 1) as usize);
        for _ in 0..(num_passes - 1) {
            shift.push(reader.read_bits(2)?);
        }
        let mut downsample = Vec::with_capacity(num_ds as usize);
        for _ in 0..num_ds {
            downsample.push(reader.read_u32(
                U32Specifier::c(1),
                U32Specifier::c(2),
                U32Specifier::c(4),
                U32Specifier::c(8),
            )?);
        }
        let mut last_pass = Vec::with_capacity(num_ds as usize);
        for _ in 0..num_ds {
            last_pass.push(reader.read_u32(
                U32Specifier::c(0),
                U32Specifier::c(1),
                U32Specifier::c(2),
                U32Specifier::u(0, 3),
            )?);
        }
        Ok(Self {
            num_passes,
            num_ds,
            shift,
            downsample,
            last_pass,
        })
    }
}

/// Frame blending mode (`BlendMode` in ISO/IEC 18181-1 Annex F.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Overwrite canvas samples (`0`).
    #[default]
    Replace = 0,
    /// Add to canvas samples (`1`).
    Add = 1,
    /// Alpha blend over canvas (`2`).
    Blend = 2,
    /// Alpha-weighted additive blend (`3`).
    MulAdd = 3,
    /// Multiply canvas samples (`4`).
    Mul = 4,
}

impl BlendMode {
    /// Returns `true` for modes that reference an alpha channel (`Blend`, `MulAdd`).
    #[must_use]
    #[inline]
    pub const fn use_alpha(self) -> bool {
        matches!(self, Self::Blend | Self::MulAdd)
    }

    fn parse(reader: &mut BitReader<'_>) -> Result<Self, JxlError> {
        match reader.read_u32(
            U32Specifier::c(0),
            U32Specifier::c(1),
            U32Specifier::c(2),
            U32Specifier::u(3, 2),
        )? {
            0 => Ok(Self::Replace),
            1 => Ok(Self::Add),
            2 => Ok(Self::Blend),
            3 => Ok(Self::MulAdd),
            4 => Ok(Self::Mul),
            _ => Err(JxlError::InvalidBitstream("invalid BlendMode")),
        }
    }
}

/// Per-frame / per-extra-channel blending specification (`BlendingInfo`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlendingInfo {
    /// Blending operation.
    pub mode: BlendMode,
    /// Index of the alpha extra channel used by `Blend` or `MulAdd`.
    pub alpha_channel: u32,
    /// Whether alpha/mul samples are clamped to `[0, 1]`.
    pub clamp: bool,
    /// Reference slot (`0..=3`) used as the base canvas.
    pub source: u32,
}

/// Gabor-like 3x3 deblocking filter configuration (`gab` in `RestorationFilter`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GaborFilter {
    /// Gaborish filter disabled.
    Disabled,
    /// Gaborish filter enabled with `[w_side, w_diag]` weights for X, Y, B.
    Enabled([[f32; 2]; 3]),
}

impl Default for GaborFilter {
    fn default() -> Self {
        Self::Enabled([[0.115169525, 0.061248592]; 3])
    }
}

impl GaborFilter {
    /// Returns `true` when the Gaborish filter is enabled.
    #[must_use]
    #[inline]
    pub const fn enabled(&self) -> bool {
        matches!(self, Self::Enabled(_))
    }
}

/// Sigma parameters for the Edge-Preserving Filter (`EpfSigma`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpfSigma {
    /// Quantization multiplier (default `0.46`).
    pub quant_mul: f32,
    /// Pass 0 sigma scale (default `0.9`).
    pub pass0_sigma_scale: f32,
    /// Pass 2 sigma scale (default `6.5`).
    pub pass2_sigma_scale: f32,
    /// Block border SAD multiplier (default `2.0 / 3.0`).
    pub border_sad_mul: f32,
}

impl Default for EpfSigma {
    fn default() -> Self {
        Self {
            quant_mul: 0.46,
            pass0_sigma_scale: 0.9,
            pass2_sigma_scale: 6.5,
            border_sad_mul: 2.0 / 3.0,
        }
    }
}

/// Edge-Preserving Filter parameters (`EpfParams`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpfParams {
    /// Number of EPF stages (`1..=3`).
    pub iters: u32,
    /// 8-entry sharpness lookup table.
    pub sharp_lut: [f32; 8],
    /// Per-channel distance weights (default `[40.0, 5.0, 3.5]`).
    pub channel_scale: [f32; 3],
    /// Sigma scaling parameters.
    pub sigma: EpfSigma,
    /// Constant sigma for Modular frames (default `1.0`).
    pub sigma_for_modular: f32,
}

impl Default for EpfParams {
    fn default() -> Self {
        Self {
            iters: 2,
            sharp_lut: [
                0.0,
                1.0 / 7.0,
                2.0 / 7.0,
                3.0 / 7.0,
                4.0 / 7.0,
                5.0 / 7.0,
                6.0 / 7.0,
                1.0,
            ],
            channel_scale: [40.0, 5.0, 3.5],
            sigma: EpfSigma::default(),
            sigma_for_modular: 1.0,
        }
    }
}

/// Edge-Preserving Filter state (`epf` in `RestorationFilter`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgePreservingFilter {
    /// EPF disabled (`iters == 0`).
    Disabled,
    /// EPF enabled (`iters` in `1..=3`).
    Enabled(EpfParams),
}

impl Default for EdgePreservingFilter {
    fn default() -> Self {
        Self::Enabled(EpfParams::default())
    }
}

impl EdgePreservingFilter {
    /// Returns `true` when EPF is enabled.
    #[must_use]
    #[inline]
    pub const fn enabled(&self) -> bool {
        matches!(self, Self::Enabled(_))
    }
}

/// In-loop restoration filters (`RestorationFilter` in ISO/IEC 18181-1 Annex F.2).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RestorationFilter {
    /// 3x3 Gabor-like deblocking filter.
    pub gab: GaborFilter,
    /// Bilateral-like Edge-Preserving Filter (1..3 stages).
    pub epf: EdgePreservingFilter,
}

impl RestorationFilter {
    /// Returns a `RestorationFilter` with both Gaborish and EPF disabled.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            gab: GaborFilter::Disabled,
            epf: EdgePreservingFilter::Disabled,
        }
    }

    fn parse(reader: &mut BitReader<'_>, encoding: FrameEncoding) -> Result<Self, JxlError> {
        if reader.read_bool()? {
            return Ok(Self::default());
        }
        let gab = if reader.read_bool()? {
            if reader.read_bool()? {
                let mut w = [[0.0f32; 2]; 3];
                for chan in &mut w {
                    for v in chan.iter_mut() {
                        *v = reader.read_f16()?;
                    }
                    if (1.0 + (chan[0] + chan[1]) * 4.0).abs() < f32::EPSILON {
                        return Err(JxlError::InvalidBitstream("degenerate Gaborish weights"));
                    }
                }
                GaborFilter::Enabled(w)
            } else {
                GaborFilter::default()
            }
        } else {
            GaborFilter::Disabled
        };

        let iters = reader.read_bits(2)?;
        let epf = if iters == 0 {
            EdgePreservingFilter::Disabled
        } else {
            let mut params = EpfParams {
                iters,
                ..EpfParams::default()
            };
            let sharp_custom = if encoding == FrameEncoding::VarDct {
                reader.read_bool()?
            } else {
                false
            };
            if sharp_custom {
                for v in &mut params.sharp_lut {
                    *v = reader.read_f16()?;
                }
            }
            if reader.read_bool()? {
                for v in &mut params.channel_scale {
                    *v = reader.read_f16()?;
                }
                let _ = reader.read_bits(32)?;
            }
            if reader.read_bool()? {
                params.sigma = EpfSigma {
                    quant_mul: if encoding == FrameEncoding::VarDct {
                        reader.read_f16()?
                    } else {
                        0.46
                    },
                    pass0_sigma_scale: reader.read_f16()?,
                    pass2_sigma_scale: reader.read_f16()?,
                    border_sad_mul: reader.read_f16()?,
                };
            }
            if encoding == FrameEncoding::Modular {
                params.sigma_for_modular = reader.read_f16()?;
            }
            EdgePreservingFilter::Enabled(params)
        };
        skip_extensions(reader)?;
        Ok(Self { gab, epf })
    }

    fn write(&self, writer: &mut BitWriter) -> Result<(), JxlError> {
        if *self == Self::default() {
            writer.write_bool(true);
            return Ok(());
        }
        writer.write_bool(false);
        match self.gab {
            GaborFilter::Disabled => writer.write_bool(false),
            GaborFilter::Enabled(w) => {
                writer.write_bool(true);
                if GaborFilter::Enabled(w) == GaborFilter::default() {
                    writer.write_bool(false);
                } else {
                    writer.write_bool(true);
                    for chan in w {
                        for v in chan {
                            writer.write_f16(v)?;
                        }
                    }
                }
            }
        }
        match self.epf {
            EdgePreservingFilter::Disabled => {
                writer.write_bits(2, 0);
            }
            EdgePreservingFilter::Enabled(params) => {
                writer.write_bits(2, params.iters.clamp(1, 3));
                writer.write_bool(false); // sharp_custom = false (VarDCT)
                writer.write_bool(false); // custom channel_scale = false
                writer.write_bool(false); // custom sigma = false
            }
        }
        writer.write_u64(0); // extensions = 0
        Ok(())
    }
}

/// Frame header (`FrameHeader` in ISO/IEC 18181-1 Annex F.2).
#[derive(Debug, Clone, PartialEq)]
pub struct FrameHeader {
    /// Frame type.
    pub frame_type: FrameType,
    /// Coding mode (`VarDct` or `Modular`).
    pub encoding: FrameEncoding,
    /// Feature flags (`NOISE`, `PATCHES`, `SPLINES`, `USE_LF_FRAME`, `SKIP_ADAPTIVE_LF_SMOOTHING`).
    pub flags: FrameFlags,
    /// Whether YCbCr transform is applied before coding (`!xyb_encoded` only).
    pub do_ycbcr: bool,
    /// Per-channel JPEG chroma subsampling code (`0..=3` for Cb, Y, Cr).
    pub jpeg_upsampling: [u32; 3],
    /// Colour upsampling factor (`1`, `2`, `4`, or `8`).
    pub upsampling: u32,
    /// Per-extra-channel upsampling factor (`1`, `2`, `4`, or `8`).
    pub ec_upsampling: Vec<u32>,
    /// Group dimension shift (`group_dim = 128 << group_size_shift`; default `1` -> `256`).
    pub group_size_shift: u32,
    /// X-channel quantization matrix scale (`0..=7`).
    pub x_qm_scale: u32,
    /// B-channel quantization matrix scale (`0..=7`).
    pub b_qm_scale: u32,
    /// Multi-pass configuration.
    pub passes: Passes,
    /// LF level (`0..=4`).
    pub lf_level: u32,
    /// Whether frame has custom crop/offset relative to canvas.
    pub have_crop: bool,
    /// Horizontal offset on canvas when `have_crop == true`.
    pub x0: i32,
    /// Vertical offset on canvas when `have_crop == true`.
    pub y0: i32,
    /// Coded frame width in pixels.
    pub width: u32,
    /// Coded frame height in pixels.
    pub height: u32,
    /// Colour channel blending parameters.
    pub blending_info: BlendingInfo,
    /// Per-extra-channel blending parameters.
    pub ec_blending_info: Vec<BlendingInfo>,
    /// Display duration in animation ticks (`0` for non-animation or coalesced layer).
    pub duration: u32,
    /// Optional 32-bit timecode.
    pub timecode: u32,
    /// Whether this is the final frame in the codestream.
    pub is_last: bool,
    /// Reference slot (`0..=3`) where this frame is saved when `!is_last`.
    pub save_as_reference: u32,
    /// Whether the reference is saved before the colour transform.
    pub save_before_ct: bool,
    /// Optional frame name.
    pub name: String,
    /// In-loop restoration filters (Gaborish + EPF).
    pub restoration_filter: RestorationFilter,
    /// Number of coded colour channels (`1` for non-XYB non-YCbCr Modular grayscale, else `3`).
    pub encoded_color_channels: usize,
    /// Sample bit depth inherited from `ImageMetadata`.
    pub bit_depth: BitDepthHeader,
}

impl FrameHeader {
    /// Parses `FrameHeader` from `reader` (does not auto-align; caller must call
    /// `reader.zero_pad_to_byte()` before reading `FrameHeader` per Annex F.1).
    pub fn parse(reader: &mut BitReader<'_>, image_header: &ImageHeader) -> Result<Self, JxlError> {
        let all_default = reader.read_bool()?;
        let meta = &image_header.metadata;
        let num_ec = meta.ec_info.len();

        let frame_type = if !all_default {
            match reader.read_bits(2)? {
                0 => FrameType::RegularFrame,
                1 => FrameType::LfFrame,
                2 => FrameType::ReferenceOnly,
                3 => FrameType::SkipProgressive,
                _ => unreachable!(),
            }
        } else {
            FrameType::RegularFrame
        };

        let encoding = if !all_default {
            if reader.read_bool()? {
                FrameEncoding::Modular
            } else {
                FrameEncoding::VarDct
            }
        } else {
            FrameEncoding::VarDct
        };

        let flags = if !all_default {
            FrameFlags(reader.read_u64()?)
        } else {
            FrameFlags(0)
        };

        let do_ycbcr = if !all_default && !meta.xyb_encoded {
            reader.read_bool()?
        } else {
            false
        };

        let encoded_color_channels = if encoding == FrameEncoding::Modular
            && !do_ycbcr
            && !meta.xyb_encoded
            && meta.grayscale()
        {
            1
        } else {
            3
        };

        let mut jpeg_upsampling = [0u32; 3];
        if do_ycbcr && !flags.use_lf_frame() {
            for slot in &mut jpeg_upsampling {
                *slot = reader.read_bits(2)?;
            }
        }

        let upsampling = if !all_default && !flags.use_lf_frame() {
            reader.read_u32(
                U32Specifier::c(1),
                U32Specifier::c(2),
                U32Specifier::c(4),
                U32Specifier::c(8),
            )?
        } else {
            1
        };

        let mut ec_upsampling = vec![1u32; num_ec];
        if !all_default && !flags.use_lf_frame() {
            for slot in &mut ec_upsampling {
                *slot = reader.read_u32(
                    U32Specifier::c(1),
                    U32Specifier::c(2),
                    U32Specifier::c(4),
                    U32Specifier::c(8),
                )?;
            }
        }

        let group_size_shift = if encoding == FrameEncoding::Modular {
            reader.read_bits(2)?
        } else {
            1
        };

        let default_x_qm = if meta.xyb_encoded && encoding == FrameEncoding::VarDct {
            3
        } else {
            2
        };
        let (x_qm_scale, b_qm_scale) =
            if !all_default && meta.xyb_encoded && encoding == FrameEncoding::VarDct {
                (reader.read_bits(3)?, reader.read_bits(3)?)
            } else {
                (default_x_qm, 2)
            };

        let passes = if !all_default && frame_type != FrameType::ReferenceOnly {
            Passes::parse(reader)?
        } else {
            Passes::default()
        };

        let lf_level = if frame_type == FrameType::LfFrame {
            1 + reader.read_bits(2)?
        } else {
            0
        };

        let have_crop = if !all_default && frame_type != FrameType::LfFrame {
            reader.read_bool()?
        } else {
            false
        };

        let crop_d0 = U32Specifier::u(0, 8);
        let crop_d1 = U32Specifier::u(256, 11);
        let crop_d2 = U32Specifier::u(2304, 14);
        let crop_d3 = U32Specifier::u(18688, 30);

        let (x0, y0) = if have_crop && frame_type != FrameType::ReferenceOnly {
            let x0 = unpack_signed(reader.read_u32(crop_d0, crop_d1, crop_d2, crop_d3)?);
            let y0 = unpack_signed(reader.read_u32(crop_d0, crop_d1, crop_d2, crop_d3)?);
            (x0, y0)
        } else {
            (0, 0)
        };

        let (width, height) = if have_crop {
            let w = reader.read_u32(crop_d0, crop_d1, crop_d2, crop_d3)?;
            let h = reader.read_u32(crop_d0, crop_d1, crop_d2, crop_d3)?;
            (w, h)
        } else {
            (image_header.size.width, image_header.size.height)
        };

        if width == 0 || height == 0 {
            return Err(JxlError::InvalidBitstream(
                "FrameHeader has zero width or height",
            ));
        }

        let full_image = !have_crop
            || (x0 <= 0
                && y0 <= 0
                && (i64::from(x0) + i64::from(width) >= i64::from(image_header.size.width))
                && (i64::from(y0) + i64::from(height) >= i64::from(image_header.size.height)));

        let has_extra = num_ec > 0;
        let blending_info = if !all_default && frame_type.is_normal_frame() {
            parse_blending_info(reader, has_extra, None, full_image)?
        } else {
            BlendingInfo::default()
        };

        let mut ec_blending_info = vec![BlendingInfo::default(); num_ec];
        if !all_default && frame_type.is_normal_frame() {
            for slot in &mut ec_blending_info {
                *slot =
                    parse_blending_info(reader, has_extra, Some(blending_info.mode), full_image)?;
            }
        }

        let duration = if !all_default && frame_type.is_normal_frame() && meta.animation.is_some() {
            reader.read_u32(
                U32Specifier::c(0),
                U32Specifier::c(1),
                U32Specifier::u(0, 8),
                U32Specifier::u(0, 32),
            )?
        } else {
            0
        };

        let timecode = if !all_default
            && frame_type.is_normal_frame()
            && meta.animation.is_some_and(|a| a.have_timecodes)
        {
            reader.read_bits(32)?
        } else {
            0
        };

        let is_last = if !all_default && frame_type.is_normal_frame() {
            reader.read_bool()?
        } else {
            frame_type == FrameType::RegularFrame
        };

        let save_as_reference = if !all_default && frame_type != FrameType::LfFrame && !is_last {
            reader.read_bits(2)?
        } else {
            0
        };

        let resets_canvas = blending_info.mode == BlendMode::Replace && full_image;
        let save_before_ct = if !all_default
            && (frame_type == FrameType::ReferenceOnly
                || (resets_canvas
                    && !is_last
                    && (duration == 0 || save_as_reference != 0)
                    && frame_type != FrameType::LfFrame))
        {
            reader.read_bool()?
        } else {
            !frame_type.is_normal_frame()
        };

        let name = if !all_default {
            reader.read_name()?
        } else {
            String::new()
        };

        let restoration_filter = if !all_default {
            RestorationFilter::parse(reader, encoding)?
        } else {
            RestorationFilter::default()
        };

        if !all_default {
            skip_extensions(reader)?;
        }

        Ok(Self {
            frame_type,
            encoding,
            flags,
            do_ycbcr,
            jpeg_upsampling,
            upsampling,
            ec_upsampling,
            group_size_shift,
            x_qm_scale,
            b_qm_scale,
            passes,
            lf_level,
            have_crop,
            x0,
            y0,
            width,
            height,
            blending_info,
            ec_blending_info,
            duration,
            timecode,
            is_last,
            save_as_reference,
            save_before_ct,
            name,
            restoration_filter,
            encoded_color_channels,
            bit_depth: meta.bit_depth,
        })
    }

    /// Writes `FrameHeader` to `writer` (aligning to byte boundary first per Annex F.1).
    pub fn write(
        &self,
        writer: &mut BitWriter,
        image_header: &ImageHeader,
    ) -> Result<(), JxlError> {
        writer.zero_pad_to_byte();
        let meta = &image_header.metadata;
        writer.write_bool(false); // all_default = false
        writer.write_bits(2, self.frame_type as u32);
        writer.write_bool(self.encoding == FrameEncoding::Modular);
        writer.write_u64(self.flags.0);
        if !meta.xyb_encoded {
            writer.write_bool(self.do_ycbcr);
        }
        if self.do_ycbcr && !self.flags.use_lf_frame() {
            for &u in &self.jpeg_upsampling {
                writer.write_bits(2, u);
            }
        }
        if !self.flags.use_lf_frame() {
            writer.write_u32(
                self.upsampling,
                U32Specifier::c(1),
                U32Specifier::c(2),
                U32Specifier::c(4),
                U32Specifier::c(8),
            )?;
            for &ecu in &self.ec_upsampling {
                writer.write_u32(
                    ecu,
                    U32Specifier::c(1),
                    U32Specifier::c(2),
                    U32Specifier::c(4),
                    U32Specifier::c(8),
                )?;
            }
        }
        if self.encoding == FrameEncoding::Modular {
            writer.write_bits(2, self.group_size_shift);
        }
        if meta.xyb_encoded && self.encoding == FrameEncoding::VarDct {
            writer.write_bits(3, self.x_qm_scale);
            writer.write_bits(3, self.b_qm_scale);
        }
        if self.frame_type != FrameType::ReferenceOnly {
            // Single pass (`num_passes = 1`)
            writer.write_u32(
                1,
                U32Specifier::c(1),
                U32Specifier::c(2),
                U32Specifier::c(3),
                U32Specifier::u(4, 3),
            )?;
        }
        if self.frame_type == FrameType::LfFrame {
            writer.write_bits(2, self.lf_level.saturating_sub(1));
        }
        if self.frame_type != FrameType::LfFrame {
            writer.write_bool(false); // have_crop = false
        }
        if self.frame_type.is_normal_frame() {
            // blending_info: Replace (mode = 0), resets_canvas = true
            writer.write_u32(
                0,
                U32Specifier::c(0),
                U32Specifier::c(1),
                U32Specifier::c(2),
                U32Specifier::u(3, 2),
            )?;
            for _ in 0..meta.ec_info.len() {
                writer.write_u32(
                    0,
                    U32Specifier::c(0),
                    U32Specifier::c(1),
                    U32Specifier::c(2),
                    U32Specifier::u(3, 2),
                )?;
            }
            if let Some(anim) = &meta.animation {
                writer.write_u32(
                    self.duration,
                    U32Specifier::c(0),
                    U32Specifier::c(1),
                    U32Specifier::u(0, 8),
                    U32Specifier::u(0, 32),
                )?;
                if anim.have_timecodes {
                    writer.write_bits(32, self.timecode);
                }
            }
            writer.write_bool(self.is_last);
        }
        if self.frame_type != FrameType::LfFrame && !self.is_last {
            writer.write_bits(2, self.save_as_reference);
        }
        if self.frame_type == FrameType::ReferenceOnly
            || (!self.is_last
                && (self.duration == 0 || self.save_as_reference != 0)
                && self.frame_type != FrameType::LfFrame)
        {
            writer.write_bool(self.save_before_ct);
        }
        writer.write_name(&self.name)?;
        self.restoration_filter.write(writer)?;
        writer.write_u64(0); // extensions = 0
        Ok(())
    }

    /// Returns `true` if this frame is a displayed keyframe (`is_last || duration != 0`).
    #[must_use]
    #[inline]
    pub const fn is_keyframe(&self) -> bool {
        self.frame_type.is_normal_frame() && (self.is_last || self.duration != 0)
    }

    /// Returns `true` if this frame can be stored in a reference slot.
    #[must_use]
    #[inline]
    pub const fn can_reference(&self) -> bool {
        !self.is_last
            && (self.duration == 0 || self.save_as_reference != 0)
            && !matches!(self.frame_type, FrameType::LfFrame)
    }

    /// Sample width at `upsampling` factor.
    #[must_use]
    #[inline]
    pub const fn sample_width(&self, upsampling: u32) -> u32 {
        let mut w = self.width;
        if upsampling > 1 {
            w = w.div_ceil(upsampling);
        }
        if self.lf_level > 0 {
            let div = 1u32 << (3 * self.lf_level);
            w = (w + div - 1) >> (3 * self.lf_level);
        }
        w
    }

    /// Sample height at `upsampling` factor.
    #[must_use]
    #[inline]
    pub const fn sample_height(&self, upsampling: u32) -> u32 {
        let mut h = self.height;
        if upsampling > 1 {
            h = h.div_ceil(upsampling);
        }
        if self.lf_level > 0 {
            let div = 1u32 << (3 * self.lf_level);
            h = (h + div - 1) >> (3 * self.lf_level);
        }
        h
    }

    /// Coded colour sample width.
    #[must_use]
    #[inline]
    pub const fn color_sample_width(&self) -> u32 {
        self.sample_width(self.upsampling)
    }

    /// Coded colour sample height.
    #[must_use]
    #[inline]
    pub const fn color_sample_height(&self) -> u32 {
        self.sample_height(self.upsampling)
    }

    /// Group dimension in samples (`128 << group_size_shift`).
    #[must_use]
    #[inline]
    pub const fn group_dim(&self) -> u32 {
        128u32 << self.group_size_shift
    }

    /// LF group dimension in samples (`group_dim * 8`).
    #[must_use]
    #[inline]
    pub const fn lf_group_dim(&self) -> u32 {
        self.group_dim() * 8
    }

    /// Number of groups per horizontal row.
    #[must_use]
    #[inline]
    pub const fn groups_per_row(&self) -> u32 {
        self.color_sample_width().div_ceil(self.group_dim())
    }

    /// Number of LF groups per horizontal row.
    #[must_use]
    #[inline]
    pub const fn lf_groups_per_row(&self) -> u32 {
        self.color_sample_width().div_ceil(self.lf_group_dim())
    }

    /// Total number of pass groups in one pass.
    #[must_use]
    #[inline]
    pub const fn num_groups(&self) -> u32 {
        let gd = self.group_dim();
        self.color_sample_width().div_ceil(gd) * self.color_sample_height().div_ceil(gd)
    }

    /// Total number of LF groups in the frame.
    #[must_use]
    #[inline]
    pub const fn num_lf_groups(&self) -> u32 {
        let lgd = self.lf_group_dim();
        self.color_sample_width().div_ceil(lgd) * self.color_sample_height().div_ceil(lgd)
    }

    /// Dimensions `(width, height)` of pass group `group_idx`.
    #[must_use]
    pub const fn group_size_for(&self, group_idx: u32) -> (u32, u32) {
        self.size_for(self.group_dim(), group_idx)
    }

    /// Dimensions `(width, height)` of LF group `lf_group_idx` in full-resolution pixels.
    #[must_use]
    pub const fn lf_group_size_for(&self, lf_group_idx: u32) -> (u32, u32) {
        self.size_for(self.lf_group_dim(), lf_group_idx)
    }

    const fn size_for(&self, dim: u32, idx: u32) -> (u32, u32) {
        let w = self.color_sample_width();
        let h = self.color_sample_height();
        let cols = w.div_ceil(dim);
        let row = idx / cols;
        let col = idx % cols;
        let gw = if (col + 1) * dim <= w {
            dim
        } else {
            w - col * dim
        };
        let gh = if (row + 1) * dim <= h {
            dim
        } else {
            h - row * dim
        };
        (gw, gh)
    }

    /// Computes the pass shift map `pass_idx -> (minshift, maxshift)` for Squeeze groups.
    #[must_use]
    pub fn pass_shifts(&self) -> BTreeMap<u32, (i32, i32)> {
        let mut map = BTreeMap::new();
        let mut maxshift = 3i32;
        for (&ds, &lp) in self.passes.downsample.iter().zip(&self.passes.last_pass) {
            let minshift = ds.trailing_zeros() as i32;
            map.insert(lp, (minshift, maxshift));
            maxshift = minshift;
        }
        map.insert(self.passes.num_passes - 1, (0, maxshift));
        map
    }
}

fn parse_blending_info(
    reader: &mut BitReader<'_>,
    has_extra: bool,
    color_mode: Option<BlendMode>,
    full_image: bool,
) -> Result<BlendingInfo, JxlError> {
    let mode = BlendMode::parse(reader)?;
    let alpha_channel = if has_extra && mode.use_alpha() {
        reader.read_u32(
            U32Specifier::c(0),
            U32Specifier::c(1),
            U32Specifier::c(2),
            U32Specifier::u(3, 3),
        )?
    } else {
        0
    };
    let clamp = if (has_extra && mode.use_alpha()) || mode == BlendMode::Mul {
        reader.read_bool()?
    } else {
        false
    };
    let effective_mode = color_mode.unwrap_or(mode);
    let resets_canvas = effective_mode == BlendMode::Replace && full_image;
    let source = if !resets_canvas {
        reader.read_bits(2)?
    } else {
        0
    };
    Ok(BlendingInfo {
        mode,
        alpha_channel,
        clamp,
        source,
    })
}

/// Logical section kind in a frame Table of Contents (`Toc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TocGroupKind {
    /// Combined single-entry section (`num_groups == 1 && num_passes == 1`).
    All,
    /// `LfGlobal` section.
    LfGlobal,
    /// `LfGroup(idx)` section.
    LfGroup(u32),
    /// `HfGlobal` section.
    HfGlobal,
    /// `GroupPass { pass_idx, group_idx }` section.
    GroupPass {
        /// Pass index (`0..num_passes`).
        pass_idx: u32,
        /// Group index (`0..num_groups`).
        group_idx: u32,
    },
}

/// One entry in the frame Table of Contents (`Toc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TocEntry {
    /// Section kind in logical frame order.
    pub kind: TocGroupKind,
    /// Byte offset from the end of the TOC (start of frame data payload).
    pub offset: usize,
    /// Size of this section in bytes.
    pub size: u32,
}

/// Frame Table of Contents (`Toc` in ISO/IEC 18181-1 Annex F.3).
#[derive(Debug, Clone)]
pub struct Toc {
    /// Number of LF groups.
    pub num_lf_groups: usize,
    /// Number of pass groups per pass.
    pub num_groups: usize,
    /// Entries in logical order (`LfGlobal`, `LfGroup(0..)`, `HfGlobal`, `GroupPass(..)`).
    pub entries: Vec<TocEntry>,
    /// Total byte size of all sections combined.
    pub total_size: usize,
}

impl Toc {
    /// Parses `Toc` from `reader` and leaves `reader` byte-aligned at the start
    /// of the first group's bitstream bytes.
    pub fn parse(reader: &mut BitReader<'_>, frame_header: &FrameHeader) -> Result<Self, JxlError> {
        let num_groups = frame_header.num_groups();
        let num_passes = frame_header.passes.num_passes;
        let num_lf_groups = frame_header.num_lf_groups();

        let entry_count = if num_groups == 1 && num_passes == 1 {
            1u32
        } else {
            1 + num_lf_groups + 1 + num_groups * num_passes
        };
        if entry_count > 65_536 {
            return Err(JxlError::LimitExceeded("TOC entry count > 65536"));
        }

        let permuted = reader.read_bool()?;
        let permutation = if permuted {
            let mut dec = EntropyDecoder::parse(reader, 8)?;
            dec.begin(reader)?;
            let perm = read_permutation(reader, &mut dec, entry_count, 0)?;
            dec.finalize()?;
            perm
        } else {
            Vec::new()
        };

        reader.zero_pad_to_byte()?;
        let d0 = U32Specifier::u(0, 10);
        let d1 = U32Specifier::u(1024, 14);
        let d2 = U32Specifier::u(17408, 22);
        let d3 = U32Specifier::u(4_211_712, 30);

        let mut bitstream_sizes = Vec::with_capacity(entry_count as usize);
        for _ in 0..entry_count {
            bitstream_sizes.push(reader.read_u32(d0, d1, d2, d3)?);
        }
        reader.zero_pad_to_byte()?;

        let mut bitstream_offsets = Vec::with_capacity(entry_count as usize);
        let mut total_size = 0usize;
        for &sz in &bitstream_sizes {
            bitstream_offsets.push(total_size);
            total_size = total_size
                .checked_add(sz as usize)
                .ok_or(JxlError::LimitExceeded("TOC total byte size overflow"))?;
        }

        let kinds = if entry_count == 1 {
            vec![TocGroupKind::All]
        } else {
            let mut v = Vec::with_capacity(entry_count as usize);
            v.push(TocGroupKind::LfGlobal);
            for i in 0..num_lf_groups {
                v.push(TocGroupKind::LfGroup(i));
            }
            v.push(TocGroupKind::HfGlobal);
            for p in 0..num_passes {
                for g in 0..num_groups {
                    v.push(TocGroupKind::GroupPass {
                        pass_idx: p,
                        group_idx: g,
                    });
                }
            }
            v
        };

        let mut entries = Vec::with_capacity(entry_count as usize);
        if permuted {
            for (logical_idx, kind) in kinds.into_iter().enumerate() {
                let bs_idx = permutation[logical_idx];
                entries.push(TocEntry {
                    kind,
                    offset: bitstream_offsets[bs_idx],
                    size: bitstream_sizes[bs_idx],
                });
            }
        } else {
            for ((kind, offset), size) in kinds
                .into_iter()
                .zip(bitstream_offsets)
                .zip(bitstream_sizes)
            {
                entries.push(TocEntry { kind, offset, size });
            }
        }

        Ok(Self {
            num_lf_groups: num_lf_groups as usize,
            num_groups: num_groups as usize,
            entries,
            total_size,
        })
    }

    /// Writes an unpermuted `Toc` for `section_sizes` to `writer` and aligns to a byte boundary.
    pub fn write(writer: &mut BitWriter, section_sizes: &[u32]) -> Result<(), JxlError> {
        writer.write_bool(false); // permutated_toc = false
        writer.zero_pad_to_byte();
        let d0 = U32Specifier::u(0, 10);
        let d1 = U32Specifier::u(1024, 14);
        let d2 = U32Specifier::u(17408, 22);
        let d3 = U32Specifier::u(4_211_712, 30);
        for &sz in section_sizes {
            writer.write_u32(sz, d0, d1, d2, d3)?;
        }
        writer.zero_pad_to_byte();
        Ok(())
    }

    /// Returns `true` when the TOC has only a single combined entry (`TocGroupKind::All`).
    #[must_use]
    #[inline]
    pub fn is_single_entry(&self) -> bool {
        self.entries.len() == 1
    }

    /// Returns the slice of `frame_payload` corresponding to `kind`.
    ///
    /// # Errors
    ///
    /// Returns [`JxlError::UnexpectedEof`] if `frame_payload` is shorter than the section range.
    pub fn section_bytes<'a>(
        &self,
        frame_payload: &'a [u8],
        kind: TocGroupKind,
    ) -> Result<&'a [u8], JxlError> {
        let idx = match kind {
            TocGroupKind::All => 0,
            TocGroupKind::LfGlobal => 0,
            TocGroupKind::LfGroup(i) => 1 + (i as usize),
            TocGroupKind::HfGlobal => 1 + self.num_lf_groups,
            TocGroupKind::GroupPass {
                pass_idx,
                group_idx,
            } => {
                1 + self.num_lf_groups
                    + 1
                    + (pass_idx as usize) * self.num_groups
                    + (group_idx as usize)
            }
        };
        let entry = self.entries.get(idx).ok_or(JxlError::InvalidBitstream(
            "TOC section index out of bounds",
        ))?;
        let end = entry
            .offset
            .checked_add(entry.size as usize)
            .ok_or(JxlError::InvalidBitstream("TOC section range overflow"))?;
        frame_payload
            .get(entry.offset..end)
            .ok_or(JxlError::UnexpectedEof)
    }
}
