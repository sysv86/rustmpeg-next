// SPDX-License-Identifier: Apache-2.0
//! Pure-Rust ICCv4 and CICP color management built on `moxcms`.
//!
//! Per `docs/CONCEPT.md`, `rustmpeg-image` handles ICCv4 profiles and wide-gamut
//! conversions (sRGB, Display P3, BT.2020) exclusively through the pure-Rust
//! `moxcms` crate without any C `lcms2` dependency.

use std::sync::Arc;

use moxcms::{ColorProfile, DataColorSpace, Layout, RenderingIntent, TransformOptions};
use rustmpeg_core::{
    ColorDescription, ColorPrimaries, ColorRange, MatrixCoefficients, TransferFunction,
};
use thiserror::Error;

/// Rendering intent for ICC color conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CmsRenderingIntent {
    /// Perceptual intent (`0` in ISO/IEC 18181-1).
    Perceptual,
    /// Media-relative colorimetric intent (`1` in ISO/IEC 18181-1).
    #[default]
    Relative,
    /// Saturation intent (`2` in ISO/IEC 18181-1).
    Saturation,
    /// ICC-absolute colorimetric intent (`3` in ISO/IEC 18181-1).
    Absolute,
}

impl CmsRenderingIntent {
    /// Converts this rendering intent into the corresponding `moxcms::RenderingIntent`.
    #[must_use]
    pub const fn to_moxcms(self) -> RenderingIntent {
        match self {
            Self::Perceptual => RenderingIntent::Perceptual,
            Self::Relative => RenderingIntent::RelativeColorimetric,
            Self::Saturation => RenderingIntent::Saturation,
            Self::Absolute => RenderingIntent::AbsoluteColorimetric,
        }
    }
}

/// Errors produced by the color management subsystem.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CmsError {
    /// The ICC profile bytes are malformed or rejected by `moxcms`.
    #[error("invalid or unsupported ICC profile: {0}")]
    InvalidProfile(String),
    /// The requested color space or channel count is not supported.
    #[error("unsupported CMS color space or layout: {0}")]
    UnsupportedLayout(&'static str),
    /// Channel buffer lengths do not match the expected geometry.
    #[error("CMS channel buffer mismatch: {0}")]
    BufferMismatch(&'static str),
}

/// A parsed ICCv4/ICCv2 color profile backed by `moxcms::ColorProfile`.
#[derive(Debug, Clone)]
pub struct CmsProfile {
    inner: ColorProfile,
    channels: usize,
    layout: Layout,
}

impl CmsProfile {
    /// Parses an ICC profile from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CmsError::InvalidProfile`] when `moxcms` rejects the profile, or
    /// [`CmsError::UnsupportedLayout`] when its data color space is unsupported.
    pub fn from_icc(icc_bytes: &[u8]) -> Result<Self, CmsError> {
        let inner = ColorProfile::new_from_slice(icc_bytes)
            .map_err(|err| CmsError::InvalidProfile(err.to_string()))?;
        let (layout, channels) = color_space_to_layout(inner.color_space)?;
        Ok(Self {
            inner,
            channels,
            layout,
        })
    }

    /// Creates the built-in sRGB profile.
    #[must_use]
    pub fn new_srgb() -> Self {
        Self {
            inner: ColorProfile::new_srgb(),
            channels: 3,
            layout: Layout::Rgb,
        }
    }

    /// Creates the built-in Display P3 profile.
    #[must_use]
    pub fn new_display_p3() -> Self {
        Self {
            inner: ColorProfile::new_display_p3(),
            channels: 3,
            layout: Layout::Rgb,
        }
    }

    /// Creates the built-in BT.2020 profile.
    #[must_use]
    pub fn new_bt2020() -> Self {
        Self {
            inner: ColorProfile::new_bt2020(),
            channels: 3,
            layout: Layout::Rgb,
        }
    }

    /// Creates a built-in grayscale profile with the sRGB transfer curve.
    #[must_use]
    pub fn new_gray() -> Self {
        Self {
            inner: ColorProfile::new_gray_with_gamma(2.2),
            channels: 1,
            layout: Layout::Gray,
        }
    }

    /// Builds a profile matching a [`rustmpeg_core::ColorDescription`].
    #[must_use]
    pub fn from_color_description(desc: ColorDescription) -> Self {
        match desc.primaries {
            ColorPrimaries::DisplayP3 => Self::new_display_p3(),
            ColorPrimaries::Bt2020 => Self::new_bt2020(),
            ColorPrimaries::Bt709 | ColorPrimaries::Unknown => Self::new_srgb(),
        }
    }

    /// Number of color channels expected by this profile (`1`, `3`, or `4`).
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Underlying `moxcms::ColorProfile`.
    #[must_use]
    pub const fn inner(&self) -> &ColorProfile {
        &self.inner
    }

    /// Maps this profile to an approximate [`ColorDescription`].
    #[must_use]
    pub const fn to_color_description(&self) -> ColorDescription {
        ColorDescription {
            range: ColorRange::Full,
            primaries: ColorPrimaries::Bt709,
            transfer: TransferFunction::Srgb,
            matrix: MatrixCoefficients::Identity,
            chroma_location: rustmpeg_core::ChromaLocation::Left,
        }
    }
}

/// A prepared floating-point color transform between two [`CmsProfile`] instances.
pub struct CmsTransform {
    executor: Arc<moxcms::TransformF32Executor>,
    from_channels: usize,
    to_channels: usize,
}

impl std::fmt::Debug for CmsTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CmsTransform")
            .field("from_channels", &self.from_channels)
            .field("to_channels", &self.to_channels)
            .finish_non_exhaustive()
    }
}

impl CmsTransform {
    /// Creates a floating-point transform between `src` and `dst`.
    ///
    /// # Errors
    ///
    /// Returns [`CmsError::InvalidProfile`] when `moxcms` cannot build the transform.
    pub fn new(
        src: &CmsProfile,
        dst: &CmsProfile,
        intent: CmsRenderingIntent,
    ) -> Result<Self, CmsError> {
        let options = TransformOptions {
            rendering_intent: intent.to_moxcms(),
            ..TransformOptions::default()
        };
        let executor = src
            .inner
            .create_transform_f32(src.layout, &dst.inner, dst.layout, options)
            .map_err(|err| CmsError::InvalidProfile(err.to_string()))?;
        Ok(Self {
            executor,
            from_channels: src.channels,
            to_channels: dst.channels,
        })
    }

    /// Number of input channels required by this transform.
    #[must_use]
    pub const fn input_channels(&self) -> usize {
        self.from_channels
    }

    /// Number of output channels produced by this transform.
    #[must_use]
    pub const fn output_channels(&self) -> usize {
        self.to_channels
    }

    /// Transforms planar `f32` channels in-place.
    ///
    /// # Errors
    ///
    /// Returns [`CmsError`] if `channels` has fewer planes than required or if
    /// plane lengths differ.
    pub fn transform_planar(&self, channels: &mut [Vec<f32>]) -> Result<usize, CmsError> {
        let needed_planes = self.from_channels.max(self.to_channels);
        if channels.len() < needed_planes {
            return Err(CmsError::BufferMismatch("insufficient channel planes"));
        }
        let len = channels[0].len();
        for plane in &channels[..self.from_channels] {
            if plane.len() != len {
                return Err(CmsError::BufferMismatch("channel plane length mismatch"));
            }
        }
        for plane in &mut channels[..self.to_channels] {
            if plane.len() != len {
                plane.resize(len, 0.0);
            }
        }

        let chunk_pixels = 1024usize;
        let mut buf_in = vec![0.0f32; chunk_pixels * self.from_channels];
        let mut buf_out = vec![0.0f32; chunk_pixels * self.to_channels];

        for base in (0..len).step_by(chunk_pixels) {
            let count = (len - base).min(chunk_pixels);
            for i in 0..count {
                for c in 0..self.from_channels {
                    buf_in[i * self.from_channels + c] = channels[c][base + i];
                }
            }
            self.executor
                .transform(
                    &buf_in[..count * self.from_channels],
                    &mut buf_out[..count * self.to_channels],
                )
                .map_err(|err| CmsError::InvalidProfile(err.to_string()))?;
            for i in 0..count {
                for c in 0..self.to_channels {
                    channels[c][base + i] = buf_out[i * self.to_channels + c];
                }
            }
        }

        Ok(self.to_channels)
    }
}

fn color_space_to_layout(color_space: DataColorSpace) -> Result<(Layout, usize), CmsError> {
    match color_space {
        DataColorSpace::Xyz
        | DataColorSpace::Lab
        | DataColorSpace::Luv
        | DataColorSpace::YCbr
        | DataColorSpace::Yxy
        | DataColorSpace::Rgb
        | DataColorSpace::Hsv
        | DataColorSpace::Hls
        | DataColorSpace::Cmy => Ok((Layout::Rgb, 3)),
        DataColorSpace::Gray => Ok((Layout::Gray, 1)),
        DataColorSpace::Cmyk => Ok((Layout::Rgba, 4)),
        _ => Err(CmsError::UnsupportedLayout(
            "unsupported ICC profile data color space",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_to_display_p3_round_trip_preserves_neutral_axis() {
        let srgb = CmsProfile::new_srgb();
        let p3 = CmsProfile::new_display_p3();
        let to_p3 = CmsTransform::new(&srgb, &p3, CmsRenderingIntent::Relative).expect("to_p3");
        let to_srgb = CmsTransform::new(&p3, &srgb, CmsRenderingIntent::Relative).expect("to_srgb");

        let mut planes = vec![
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
        ];
        let orig = planes.clone();
        assert_eq!(to_p3.transform_planar(&mut planes).expect("p3"), 3);
        assert_eq!(to_srgb.transform_planar(&mut planes).expect("srgb"), 3);
        for c in 0..3 {
            for (a, b) in orig[c].iter().zip(&planes[c]) {
                assert!((a - b).abs() < 1e-3, "diff too large: {a} vs {b}");
            }
        }
    }
}
