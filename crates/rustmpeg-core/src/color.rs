// SPDX-License-Identifier: Apache-2.0
//! Colour description metadata shared by video and image pipelines.

/// Sample range of the coded signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorRange {
    /// Studio range (BT.709/BT.2020 limited range).
    #[default]
    Limited,
    /// Full range (`0..=2^depth - 1`).
    Full,
}

/// Colour primaries and white point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorPrimaries {
    /// BT.709 / sRGB primaries.
    #[default]
    Bt709,
    /// BT.2020 primaries.
    Bt2020,
    /// Display P3 primaries.
    DisplayP3,
    /// Unknown or unspecified.
    Unknown,
}

/// Opto-electronic transfer function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TransferFunction {
    /// Standard dynamic range (BT.709 / BT.2020).
    #[default]
    Sdr,
    /// SMPTE ST 2084 perceptual quantizer (HDR10).
    Pq,
    /// Hybrid log-gamma (HLG).
    Hlg,
    /// Linear light.
    Linear,
    /// sRGB transfer curve.
    Srgb,
    /// Unknown or unspecified.
    Unknown,
}

/// Matrix coefficients used to derive luma and chroma.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MatrixCoefficients {
    /// BT.709 / BT.601 derivation.
    #[default]
    Bt709,
    /// BT.2020 non-constant luminance.
    Bt2020NonConstant,
    /// Identity, used by RGB.
    Identity,
    /// Unknown or unspecified.
    Unknown,
}

/// Chroma sample location relative to the luma grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChromaLocation {
    /// Horizontally and vertically co-sited.
    #[default]
    Left,
    /// Horizontally centred.
    Center,
    /// Top-left convention used by some contributors.
    TopLeft,
}

/// Complete colour description of a picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ColorDescription {
    /// Limited or full range.
    pub range: ColorRange,
    /// Colour primaries.
    pub primaries: ColorPrimaries,
    /// Transfer function.
    pub transfer: TransferFunction,
    /// Matrix coefficients.
    pub matrix: MatrixCoefficients,
    /// Chroma sample location.
    pub chroma_location: ChromaLocation,
}

impl ColorDescription {
    /// BT.709 limited-range SDR, the default for 8-bit video.
    pub const BT709_LIMITED: Self = Self {
        range: ColorRange::Limited,
        primaries: ColorPrimaries::Bt709,
        transfer: TransferFunction::Sdr,
        matrix: MatrixCoefficients::Bt709,
        chroma_location: ChromaLocation::Left,
    };

    /// BT.2020 full-range PQ HDR.
    pub const BT2020_PQ: Self = Self {
        range: ColorRange::Full,
        primaries: ColorPrimaries::Bt2020,
        transfer: TransferFunction::Pq,
        matrix: MatrixCoefficients::Bt2020NonConstant,
        chroma_location: ChromaLocation::Left,
    };

    /// BT.2020 limited-range HLG HDR.
    pub const BT2020_HLG: Self = Self {
        range: ColorRange::Limited,
        primaries: ColorPrimaries::Bt2020,
        transfer: TransferFunction::Hlg,
        matrix: MatrixCoefficients::Bt2020NonConstant,
        chroma_location: ChromaLocation::Left,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_bt709_limited() {
        assert_eq!(ColorDescription::default(), ColorDescription::BT709_LIMITED);
    }

    #[test]
    fn constants_carry_expected_metadata() {
        assert_eq!(ColorDescription::BT2020_PQ.range, ColorRange::Full);
        assert_eq!(ColorDescription::BT2020_HLG.transfer, TransferFunction::Hlg);
    }
}
