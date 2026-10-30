// SPDX-License-Identifier: Apache-2.0
//! Planar video frames and borrowed plane views.

use alloc::vec;
use alloc::vec::Vec;

use crate::buffer::Buffer;
use crate::color::ColorDescription;
use crate::error::{CoreError, Result};
use crate::limits;
use crate::pixel::PixelFormat;

/// One owned plane of a video frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    /// Visible bytes, including row padding between rows.
    pub data: Buffer,
    /// Distance between the first bytes of consecutive rows.
    pub stride: usize,
    /// Visible width in samples.
    pub width: u32,
    /// Visible height in samples.
    pub height: u32,
}

impl Plane {
    /// Creates an empty placeholder plane.
    pub fn empty() -> Self {
        Self {
            data: Buffer::empty(),
            stride: 0,
            width: 0,
            height: 0,
        }
    }

    /// Creates a plane from an owned buffer and geometry.
    pub const fn new(data: Buffer, stride: usize, width: u32, height: u32) -> Self {
        Self {
            data,
            stride,
            width,
            height,
        }
    }

    /// `true` when the plane has no visible samples.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// A borrowed plane with an explicit visible row length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneRef<'a> {
    /// Visible bytes, including row padding.
    pub data: &'a [u8],
    /// Distance between row starts.
    pub stride: usize,
    /// Visible width in samples.
    pub width: u32,
    /// Visible height in samples.
    pub height: u32,
    /// Visible bytes per row.
    pub row_bytes: usize,
}

impl<'a> PlaneRef<'a> {
    /// Creates a borrowed plane view.
    pub const fn new(
        data: &'a [u8],
        stride: usize,
        width: u32,
        height: u32,
        row_bytes: usize,
    ) -> Self {
        Self {
            data,
            stride,
            width,
            height,
            row_bytes,
        }
    }

    /// Returns one visible row, excluding padding.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidIndex`] when `row >= height`.
    pub fn row(&self, row: usize) -> Result<&'a [u8]> {
        if row >= self.height as usize {
            return Err(CoreError::index(row, "plane row"));
        }
        let start = row
            .checked_mul(self.stride)
            .ok_or(CoreError::Overflow("plane row offset"))?;
        let end = start
            .checked_add(self.row_bytes)
            .ok_or(CoreError::Overflow("plane row end"))?;
        self.data
            .get(start..end)
            .ok_or_else(|| CoreError::index(row, "plane row"))
    }

    /// `true` when every visible row is inside the backing buffer.
    pub fn is_valid(&self) -> bool {
        if self.width == 0 || self.height == 0 {
            return true;
        }
        if self.stride < self.row_bytes {
            return false;
        }
        let last = (self.height as usize - 1)
            .checked_mul(self.stride)
            .and_then(|offset| offset.checked_add(self.row_bytes));
        last.is_some_and(|required| required <= self.data.len())
    }
}

/// An owned planar video frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoFrame {
    /// Pixel layout.
    pub format: PixelFormat,
    /// Sample bit depth.
    pub bit_depth: u8,
    /// Colour metadata.
    pub color: ColorDescription,
    /// Visible width in luma samples.
    pub width: u32,
    /// Visible height in luma samples.
    pub height: u32,
    /// Planes in format order; unused entries are empty.
    pub planes: [Plane; 3],
}

impl VideoFrame {
    /// Allocates a zeroed frame with tightly packed planes.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid dimensions, bit depth, or size
    /// overflow.
    pub fn new(
        format: PixelFormat,
        bit_depth: u8,
        width: u32,
        height: u32,
        color: ColorDescription,
    ) -> Result<Self> {
        limits::validate_dimensions(width, height)?;
        let mut planes = [Plane::empty(), Plane::empty(), Plane::empty()];
        for (index, plane) in planes.iter_mut().enumerate().take(format.plane_count()) {
            let (plane_width, plane_height) = format.plane_dimensions(width, height, index)?;
            let bytes = format.plane_bytes(width, height, index, bit_depth)?;
            let sample_width = if format.is_packed() {
                format.packed_bytes_per_pixel()?
            } else {
                format.bytes_per_sample(bit_depth)?
            };
            let stride = usize::try_from(plane_width)
                .map_err(|_| CoreError::Overflow("plane width"))?
                .checked_mul(sample_width)
                .ok_or(CoreError::Overflow("plane stride"))?;
            *plane = Plane::new(
                Buffer::from_vec(vec![0; bytes]),
                stride,
                plane_width,
                plane_height,
            );
        }
        let frame = Self {
            format,
            bit_depth,
            color,
            width,
            height,
            planes,
        };
        frame.validate()?;
        Ok(frame)
    }

    /// Builds a frame from caller-provided planes.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidConfig`] when a plane's dimensions, stride,
    /// or backing length do not match the format and picture size.
    pub fn from_planes(
        format: PixelFormat,
        bit_depth: u8,
        width: u32,
        height: u32,
        color: ColorDescription,
        planes: [Plane; 3],
    ) -> Result<Self> {
        let frame = Self {
            format,
            bit_depth,
            color,
            width,
            height,
            planes,
        };
        frame.validate()?;
        Ok(frame)
    }

    /// Number of active planes.
    pub fn plane_count(&self) -> usize {
        self.format.plane_count()
    }

    /// Returns a borrowed plane view for an active plane.
    pub fn plane(&self, index: usize) -> Option<PlaneRef<'_>> {
        if index >= self.plane_count() {
            return None;
        }
        let plane = &self.planes[index];
        if plane.is_empty() {
            return None;
        }
        let row_bytes = self.row_bytes(index).ok()?;
        Some(PlaneRef::new(
            &plane.data,
            plane.stride,
            plane.width,
            plane.height,
            row_bytes,
        ))
    }

    /// `true` when every plane is internally consistent.
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Visible bytes per row for one plane.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid plane index or sample width.
    pub fn row_bytes(&self, index: usize) -> Result<usize> {
        let plane = self
            .planes
            .get(index)
            .ok_or_else(|| CoreError::index(index, "frame plane"))?;
        let sample = if self.format.is_packed() {
            self.format.packed_bytes_per_pixel()?
        } else {
            self.format.bytes_per_sample(self.bit_depth)?
        };
        usize::try_from(plane.width)
            .map_err(|_| CoreError::Overflow("plane width"))?
            .checked_mul(sample)
            .ok_or(CoreError::Overflow("plane row bytes"))
    }

    fn validate(&self) -> Result<()> {
        limits::validate_dimensions(self.width, self.height)?;
        if self.format.is_packed() {
            self.format.bytes_per_sample(self.bit_depth)?;
        }
        if self.format.plane_count() > self.planes.len() {
            return Err(CoreError::InvalidConfig(
                "format exposes more than three planes".into(),
            ));
        }
        for index in 0..self.planes.len() {
            let plane = &self.planes[index];
            if index >= self.format.plane_count() {
                if !plane.is_empty() {
                    return Err(CoreError::InvalidConfig(
                        "unused plane must be empty".into(),
                    ));
                }
                continue;
            }
            let (expected_width, expected_height) =
                self.format
                    .plane_dimensions(self.width, self.height, index)?;
            if plane.width != expected_width || plane.height != expected_height {
                return Err(CoreError::InvalidConfig(
                    "plane dimensions do not match the pixel format".into(),
                ));
            }
            let row_bytes = self.row_bytes(index)?;
            if plane.stride < row_bytes {
                return Err(CoreError::InvalidConfig(
                    "plane stride is smaller than one visible row".into(),
                ));
            }
            let needed = plane
                .stride
                .checked_mul(plane.height as usize)
                .ok_or(CoreError::Overflow("plane buffer size"))?;
            if plane.data.len() < needed {
                return Err(CoreError::BufferTooSmall {
                    needed,
                    got: plane.data.len(),
                });
            }
        }
        Ok(())
    }
}

/// Read-only frame interface implemented by the owned core frame and codec
/// frame types.
pub trait VideoFrameView {
    /// Pixel layout.
    fn format(&self) -> PixelFormat;
    /// Sample bit depth.
    fn bit_depth(&self) -> u8;
    /// Width in luma samples.
    fn width(&self) -> u32;
    /// Height in luma samples.
    fn height(&self) -> u32;
    /// Borrowed plane view.
    fn plane(&self, index: usize) -> Option<PlaneRef<'_>>;
    /// Number of active planes.
    fn plane_count(&self) -> usize {
        self.format().plane_count()
    }
}

impl VideoFrameView for VideoFrame {
    fn format(&self) -> PixelFormat {
        self.format
    }

    fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn plane(&self, index: usize) -> Option<PlaneRef<'_>> {
        VideoFrame::plane(self, index)
    }

    fn plane_count(&self) -> usize {
        VideoFrame::plane_count(self)
    }
}

/// Builds a tightly packed plane from bytes.
///
/// # Errors
///
/// Returns an error when the byte length does not match the geometry.
pub fn packed_plane(
    data: Vec<u8>,
    width: u32,
    height: u32,
    bytes_per_sample: usize,
) -> Result<Plane> {
    let row_bytes = usize::try_from(width)
        .map_err(|_| CoreError::Overflow("plane width"))?
        .checked_mul(bytes_per_sample)
        .ok_or(CoreError::Overflow("plane row bytes"))?;
    let needed = row_bytes
        .checked_mul(height as usize)
        .ok_or(CoreError::Overflow("plane buffer size"))?;
    if data.len() != needed {
        return Err(CoreError::InvalidData(
            "packed plane length does not match its geometry".into(),
        ));
    }
    Ok(Plane::new(Buffer::from_vec(data), row_bytes, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn color() -> ColorDescription {
        ColorDescription::BT709_LIMITED
    }

    #[test]
    fn allocation_matches_plane_geometry() {
        let frame = VideoFrame::new(PixelFormat::I420, 8, 65, 49, color()).expect("frame");
        assert_eq!(frame.plane_count(), 3);
        assert_eq!(frame.planes[0].width, 65);
        assert_eq!(frame.planes[1].width, 33);
        assert_eq!(frame.planes[1].height, 25);
        assert_eq!(frame.planes[0].stride, 65);
        assert_eq!(frame.planes[1].stride, 33);
        assert!(frame.is_valid());
        assert!(frame.plane(2).is_some());
        assert!(frame.plane(3).is_none());
    }

    #[test]
    fn borrowed_rows_exclude_padding() {
        let frame = VideoFrame::new(PixelFormat::I420, 8, 4, 4, color()).expect("frame");
        let view = frame.plane(0).expect("luma");
        assert_eq!(view.row(0).expect("row").len(), 4);
        assert!(view.row(4).is_err());
        assert!(view.is_valid());
    }

    #[test]
    fn padded_planes_are_accepted() {
        let mut data = vec![0_u8; 8 * 4];
        let plane = Plane::new(Buffer::from_vec(core::mem::take(&mut data)), 8, 4, 4);
        let frame = VideoFrame::from_planes(
            PixelFormat::I400,
            8,
            4,
            4,
            color(),
            [plane, Plane::empty(), Plane::empty()],
        )
        .expect("padded frame");
        assert!(frame.is_valid());
        assert_eq!(frame.plane(0).expect("luma").stride, 8);
    }

    #[test]
    fn wrong_plane_geometry_is_rejected() {
        let plane = Plane::new(Buffer::from_vec(vec![0; 16]), 4, 4, 4);
        let error = VideoFrame::from_planes(
            PixelFormat::I420,
            8,
            4,
            4,
            color(),
            [plane, Plane::empty(), Plane::empty()],
        )
        .expect_err("missing chroma planes");
        assert!(matches!(error, CoreError::InvalidConfig(_)));
    }

    #[test]
    fn packed_frames_use_pixel_width() {
        let frame = VideoFrame::new(PixelFormat::Rgba8, 8, 3, 2, color()).expect("frame");
        assert_eq!(frame.plane_count(), 1);
        assert_eq!(frame.planes[0].stride, 12);
        assert_eq!(frame.planes[0].data.len(), 24);
        assert!(frame.is_valid());
    }

    #[test]
    fn packed_plane_helper_checks_length() {
        let plane = packed_plane(vec![0; 12], 3, 2, 2).expect("plane");
        assert_eq!(plane.stride, 6);
        assert!(packed_plane(vec![0; 11], 3, 2, 2).is_err());
    }
}
