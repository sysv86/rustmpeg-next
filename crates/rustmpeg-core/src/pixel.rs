// SPDX-License-Identifier: Apache-2.0
//! Pixel layouts and plane geometry.

use crate::error::{CoreError, Result};
use crate::limits;

/// Pixel layout of a picture.
///
/// Packed RGB layouts are 8-bit only in the current matrix; high-bit-depth RGB
/// is carried by the image pipeline until a dedicated format is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// Monochrome luma.
    I400,
    /// Planar 4:2:0 luma plus half-resolution Cb/Cr.
    I420,
    /// Planar 4:2:2 luma plus horizontally subsampled Cb/Cr.
    I422,
    /// Planar 4:4:4 luma plus full-resolution Cb/Cr.
    I444,
    /// Packed 8-bit RGB.
    Rgb8,
    /// Packed 8-bit RGBA.
    Rgba8,
    /// Packed 8-bit BGRA.
    Bgra8,
}

impl PixelFormat {
    /// Number of planes exposed by this layout.
    pub const fn plane_count(self) -> usize {
        match self {
            Self::I400 | Self::Rgb8 | Self::Rgba8 | Self::Bgra8 => 1,
            Self::I420 | Self::I422 | Self::I444 => 3,
        }
    }

    /// Horizontal and vertical chroma subsampling exponents.
    pub const fn subsampling(self) -> (u8, u8) {
        match self {
            Self::I400 | Self::I444 => (0, 0),
            Self::I420 => (1, 1),
            Self::I422 => (1, 0),
            Self::Rgb8 | Self::Rgba8 | Self::Bgra8 => (0, 0),
        }
    }

    /// `true` for planar YUV layouts.
    pub const fn is_planar(self) -> bool {
        matches!(self, Self::I400 | Self::I420 | Self::I422 | Self::I444)
    }

    /// `true` for packed RGB layouts.
    pub const fn is_packed(self) -> bool {
        matches!(self, Self::Rgb8 | Self::Rgba8 | Self::Bgra8)
    }

    /// Bytes per sample for a planar layout.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Unsupported`] for packed layouts and bit depths
    /// outside the implemented matrix.
    pub fn bytes_per_sample(self, bit_depth: u8) -> Result<usize> {
        match (self, bit_depth) {
            (Self::Rgb8 | Self::Rgba8 | Self::Bgra8, 8) => Ok(1),
            (Self::Rgb8 | Self::Rgba8 | Self::Bgra8, _) => Err(CoreError::Unsupported(
                "packed RGB is currently 8-bit only".into(),
            )),
            (_, 8) => Ok(1),
            (_, 10 | 12 | 16) => Ok(2),
            (_, _) => Err(CoreError::Unsupported(
                "planar bit depth must be 8, 10, 12, or 16".into(),
            )),
        }
    }

    /// Number of packed bytes per pixel for RGB layouts.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Unsupported`] for planar layouts.
    pub fn packed_bytes_per_pixel(self) -> Result<usize> {
        match self {
            Self::Rgb8 => Ok(3),
            Self::Rgba8 | Self::Bgra8 => Ok(4),
            _ => Err(CoreError::Unsupported(
                "packed bytes per pixel are defined only for RGB layouts".into(),
            )),
        }
    }

    /// Dimensions of one plane.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidIndex`] when the plane is not present.
    pub fn plane_dimensions(self, width: u32, height: u32, plane: usize) -> Result<(u32, u32)> {
        if plane >= self.plane_count() {
            return Err(CoreError::index(plane, "plane"));
        }
        let (sub_x, sub_y) = self.subsampling();
        if plane == 0 || self.is_packed() {
            return Ok((width, height));
        }
        let divisor_x = 1_u32 << sub_x;
        let divisor_y = 1_u32 << sub_y;
        Ok((width.div_ceil(divisor_x), height.div_ceil(divisor_y)))
    }

    /// Byte size of one plane.
    ///
    /// # Errors
    ///
    /// Propagates dimension, subsampling, and overflow failures.
    pub fn plane_bytes(
        self,
        width: u32,
        height: u32,
        plane: usize,
        bit_depth: u8,
    ) -> Result<usize> {
        let (plane_width, plane_height) = self.plane_dimensions(width, height, plane)?;
        let sample = if self.is_packed() {
            self.bytes_per_sample(bit_depth)?;
            self.packed_bytes_per_pixel()?
        } else {
            self.bytes_per_sample(bit_depth)?
        };
        limits::plane_bytes(plane_width, plane_height, sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_counts_and_subsampling_match_the_layouts() {
        assert_eq!(PixelFormat::I400.plane_count(), 1);
        assert_eq!(PixelFormat::I420.plane_count(), 3);
        assert_eq!(PixelFormat::I420.subsampling(), (1, 1));
        assert_eq!(PixelFormat::I422.subsampling(), (1, 0));
        assert_eq!(PixelFormat::I444.subsampling(), (0, 0));
        assert!(PixelFormat::Rgba8.is_packed());
        assert!(!PixelFormat::Rgba8.is_planar());
    }

    #[test]
    fn chroma_planes_round_up_for_odd_sizes() {
        assert_eq!(
            PixelFormat::I420.plane_dimensions(65, 49, 1).expect("U"),
            (33, 25)
        );
        assert_eq!(
            PixelFormat::I422.plane_dimensions(65, 49, 1).expect("U"),
            (33, 49)
        );
        assert_eq!(
            PixelFormat::I444.plane_dimensions(65, 49, 1).expect("U"),
            (65, 49)
        );
        assert!(PixelFormat::I420.plane_dimensions(64, 48, 3).is_err());
    }

    #[test]
    fn plane_bytes_use_the_sample_width() {
        assert_eq!(
            PixelFormat::I420.plane_bytes(1920, 1080, 0, 8).expect("Y"),
            2_073_600
        );
        assert_eq!(
            PixelFormat::I420.plane_bytes(1920, 1080, 0, 10).expect("Y"),
            4_147_200
        );
        assert_eq!(
            PixelFormat::I420.plane_bytes(1920, 1080, 1, 8).expect("U"),
            518_400
        );
        assert_eq!(
            PixelFormat::Rgba8.plane_bytes(2, 2, 0, 8).expect("RGBA"),
            16
        );
        assert!(PixelFormat::Rgba8.bytes_per_sample(10).is_err());
    }

    #[test]
    fn packed_layouts_report_pixel_width() {
        assert_eq!(PixelFormat::Rgb8.packed_bytes_per_pixel().expect("rgb"), 3);
        assert_eq!(
            PixelFormat::Bgra8.packed_bytes_per_pixel().expect("bgra"),
            4
        );
        assert!(PixelFormat::I420.packed_bytes_per_pixel().is_err());
    }
}
