// SPDX-License-Identifier: Apache-2.0
//! Accelerator traits for frame-level video stages.
//!
//! The codec crates are lower in the dependency graph than the Vulkan backend,
//! so they cannot name `wgpu` types. These traits define the narrow boundary
//! between a codec and an optional accelerator: RGBA to I420 conversion and
//! plane distortion metrics. `rustmpeg-vulkan` implements them on top of
//! Vulkan Compute; [`CpuRgbaToI420`] and [`CpuFrameMetrics`] are the reference
//! implementations that always exist.
//!
//! A provider must be deterministic and bit-exact with its CPU counterpart. If
//! a GPU kernel disagrees with the reference, the scheduler disables the GPU
//! path instead of changing the encoded output.

use crate::{I420Frame, color};

/// Failure reported by an accelerator.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AcceleratorError {
    /// No usable device is present.
    #[error("accelerator is unavailable: {0}")]
    Unavailable(String),
    /// The input violates the stage contract.
    #[error("accelerator input is invalid: {0}")]
    InvalidInput(String),
    /// The input is valid but exceeds a device limit.
    #[error("accelerator limit exceeded: {0}")]
    LimitExceeded(String),
    /// The device rejected or aborted the operation.
    #[error("accelerator execution failed: {0}")]
    Execution(String),
}

/// One 8-bit plane with an explicit row pitch.
///
/// `stride` is measured in bytes. Rows may be padded, which lets the metrics
/// stage read codec planes in place without an intermediate copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneView<'a> {
    /// Plane samples in raster order, with padding between rows.
    pub data: &'a [u8],
    /// Visible width in samples.
    pub width: u32,
    /// Visible height in samples.
    pub height: u32,
    /// Distance between the first sample of consecutive rows, in bytes.
    pub stride: usize,
}

impl<'a> PlaneView<'a> {
    /// Creates a plane view.
    pub const fn new(data: &'a [u8], width: u32, height: u32, stride: usize) -> Self {
        Self {
            data,
            width,
            height,
            stride,
        }
    }

    /// `true` when the view has no visible samples.
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Number of visible samples.
    pub fn samples(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// `true` when every visible sample lies inside `data`.
    pub fn is_valid(&self) -> bool {
        if self.is_empty() {
            return true;
        }
        if self.stride < self.width as usize {
            return false;
        }
        let last_row = (self.height as usize - 1)
            .checked_mul(self.stride)
            .and_then(|offset| offset.checked_add(self.width as usize));
        last_row.is_some_and(|required| required <= self.data.len())
    }
}

/// Sum of squared errors and sum of absolute differences for one plane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneMetrics {
    /// Sum of squared sample differences.
    pub sse: u64,
    /// Sum of absolute sample differences.
    pub sad: u64,
    /// Number of compared samples.
    pub samples: u64,
}

impl PlaneMetrics {
    /// Mean squared error, or `0.0` for an empty plane.
    pub fn mse(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.sse as f64 / self.samples as f64
        }
    }

    /// Peak signal-to-noise ratio for a given coded bit depth.
    ///
    /// A zero error is reported as `100.0` dB, the same convention the AV2
    /// encoder uses for a lossless plane.
    pub fn psnr(&self, bit_depth: u8) -> f64 {
        let mse = self.mse();
        if mse <= 1e-12 {
            return 100.0;
        }
        let peak = ((1_u64 << bit_depth.min(16)) - 1) as f64;
        10.0 * ((peak * peak) / mse).log10()
    }
}

/// Distortion metrics for the three planes of a 4:2:0 frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameMetrics {
    /// Luma, Cb, and Cr metrics in that order.
    pub planes: [PlaneMetrics; 3],
}

impl FrameMetrics {
    /// Luma mean squared error.
    pub fn mse_y(&self) -> f64 {
        self.planes[0].mse()
    }

    /// Per-plane PSNR `[Y, U, V]`.
    pub fn psnr(&self, bit_depth: u8) -> [f64; 3] {
        [
            self.planes[0].psnr(bit_depth),
            self.planes[1].psnr(bit_depth),
            self.planes[2].psnr(bit_depth),
        ]
    }

    /// Weighted PSNR `(4 * Y + U + V) / 6`, or luma-only for a monochrome
    /// frame.
    pub fn psnr_average(&self, bit_depth: u8) -> f64 {
        let [luma, cb, cr] = self.psnr(bit_depth);
        if self.planes[1].samples == 0 || self.planes[2].samples == 0 {
            return luma;
        }
        (4.0 * luma + cb + cr) / 6.0
    }
}

/// Converts premultiplied RGBA8 to a BT.709 limited-range I420 frame.
pub trait RgbaToI420Converter: Send + Sync {
    /// Converts exactly `width * height * 4` bytes of RGBA.
    ///
    /// # Errors
    ///
    /// Returns [`AcceleratorError`] when the input length does not match the
    /// dimensions, the picture is empty, or the device fails.
    fn rgba_to_i420(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<I420Frame, AcceleratorError>;
}

/// Computes distortion metrics between two 8-bit planes.
pub trait FrameMetricsProvider: Send + Sync {
    /// Computes SSE, SAD, and the sample count for one plane pair.
    ///
    /// # Errors
    ///
    /// Returns [`AcceleratorError`] when the views are invalid or the device
    /// fails.
    fn plane_metrics(
        &self,
        reference: PlaneView<'_>,
        distorted: PlaneView<'_>,
    ) -> Result<PlaneMetrics, AcceleratorError>;

    /// Computes metrics for a complete frame.
    ///
    /// The default implementation issues one [`FrameMetricsProvider::plane_metrics`]
    /// call per plane, which also lets a backend batch the three planes.
    ///
    /// # Errors
    ///
    /// Propagates the first plane error.
    fn frame_metrics(
        &self,
        reference: [PlaneView<'_>; 3],
        distorted: [PlaneView<'_>; 3],
    ) -> Result<FrameMetrics, AcceleratorError> {
        let mut planes = [PlaneMetrics::default(); 3];
        for index in 0..3 {
            planes[index] = self.plane_metrics(reference[index], distorted[index])?;
        }
        Ok(FrameMetrics { planes })
    }
}

/// CPU implementation of [`RgbaToI420Converter`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuRgbaToI420;

impl RgbaToI420Converter for CpuRgbaToI420 {
    fn rgba_to_i420(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<I420Frame, AcceleratorError> {
        if width == 0 || height == 0 {
            return Err(AcceleratorError::InvalidInput(
                "picture dimensions must be non-zero".to_owned(),
            ));
        }
        let expected = u64::from(width) * u64::from(height) * 4;
        if expected != rgba.len() as u64 {
            return Err(AcceleratorError::InvalidInput(format!(
                "RGBA input has {} bytes, expected {expected} for {width}x{height}",
                rgba.len()
            )));
        }
        Ok(color::rgba_to_i420(rgba, width, height))
    }
}

/// CPU implementation of [`FrameMetricsProvider`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFrameMetrics;

impl FrameMetricsProvider for CpuFrameMetrics {
    fn plane_metrics(
        &self,
        reference: PlaneView<'_>,
        distorted: PlaneView<'_>,
    ) -> Result<PlaneMetrics, AcceleratorError> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(AcceleratorError::InvalidInput(format!(
                "plane dimensions differ: {}x{} and {}x{}",
                reference.width, reference.height, distorted.width, distorted.height
            )));
        }
        if !reference.is_valid() || !distorted.is_valid() {
            return Err(AcceleratorError::InvalidInput(
                "plane view exceeds its backing buffer".to_owned(),
            ));
        }
        if reference.is_empty() {
            return Ok(PlaneMetrics::default());
        }

        let width = reference.width as usize;
        let height = reference.height as usize;
        let mut sse = 0_u64;
        let mut sad = 0_u64;
        for row in 0..height {
            let a = &reference.data[row * reference.stride..][..width];
            let b = &distorted.data[row * distorted.stride..][..width];
            for (left, right) in a.iter().zip(b) {
                let difference = u64::from((i32::from(*left) - i32::from(*right)).unsigned_abs());
                sse += difference * difference;
                sad += difference;
            }
        }
        Ok(PlaneMetrics {
            sse,
            sad,
            samples: reference.samples(),
        })
    }
}

/// Builds the three plane views of an I420 frame.
pub fn i420_plane_views(frame: &I420Frame) -> [PlaneView<'_>; 3] {
    let (chroma_width, chroma_height) = frame.chroma_size();
    [
        PlaneView::new(&frame.y, frame.width, frame.height, frame.width as usize),
        PlaneView::new(&frame.u, chroma_width, chroma_height, chroma_width as usize),
        PlaneView::new(&frame.v, chroma_width, chroma_height, chroma_width as usize),
    ]
}

/// Computes full-frame metrics for two I420 frames.
///
/// # Errors
///
/// Returns [`AcceleratorError::InvalidInput`] when either frame is malformed
/// and propagates provider failures.
pub fn i420_metrics(
    provider: &dyn FrameMetricsProvider,
    reference: &I420Frame,
    distorted: &I420Frame,
) -> Result<FrameMetrics, AcceleratorError> {
    if !reference.is_valid() || !distorted.is_valid() {
        return Err(AcceleratorError::InvalidInput(
            "I420 frame dimensions do not match its planes".to_owned(),
        ));
    }
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(AcceleratorError::InvalidInput(format!(
            "frame dimensions differ: {}x{} and {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )));
    }
    provider.frame_metrics(i420_plane_views(reference), i420_plane_views(distorted))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, value: u8) -> I420Frame {
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        I420Frame {
            width,
            height,
            y: vec![value; (width * height) as usize],
            u: vec![value; (cw * ch) as usize],
            v: vec![value; (cw * ch) as usize],
        }
    }

    #[test]
    fn cpu_converter_matches_the_canonical_reference() {
        let rgba = vec![17_u8; 4 * 4 * 4];
        let converted = CpuRgbaToI420
            .rgba_to_i420(&rgba, 4, 4)
            .expect("conversion succeeds");
        assert_eq!(converted, color::rgba_to_i420(&rgba, 4, 4));
    }

    #[test]
    fn cpu_converter_rejects_a_short_buffer() {
        let error = CpuRgbaToI420
            .rgba_to_i420(&[0; 15], 2, 2)
            .expect_err("short input must be rejected");
        assert!(matches!(error, AcceleratorError::InvalidInput(_)));
    }

    #[test]
    fn cpu_metrics_are_zero_for_identical_frames() {
        let metrics = i420_metrics(&CpuFrameMetrics, &frame(4, 4, 9), &frame(4, 4, 9))
            .expect("metrics succeed");
        assert_eq!(metrics.planes[0].sse, 0);
        assert_eq!(metrics.planes[0].sad, 0);
        assert_eq!(metrics.planes[0].samples, 16);
        assert_eq!(metrics.psnr(8), [100.0; 3]);
        assert_eq!(metrics.psnr_average(8), 100.0);
    }

    #[test]
    fn cpu_metrics_count_sse_and_sad() {
        let metrics = i420_metrics(&CpuFrameMetrics, &frame(2, 2, 10), &frame(2, 2, 7))
            .expect("metrics succeed");
        assert_eq!(metrics.planes[0].sse, 4 * 9);
        assert_eq!(metrics.planes[0].sad, 4 * 3);
        assert_eq!(metrics.planes[1].samples, 1);
    }

    #[test]
    fn padded_planes_are_read_with_the_declared_stride() {
        let a = [1_u8, 2, 0, 0, 3, 4, 0, 0];
        let b = [1_u8, 2, 9, 9, 3, 5, 9, 9];
        let metrics = CpuFrameMetrics
            .plane_metrics(PlaneView::new(&a, 2, 2, 4), PlaneView::new(&b, 2, 2, 4))
            .expect("metrics succeed");
        assert_eq!(metrics.sse, 1);
        assert_eq!(metrics.sad, 1);
        assert_eq!(metrics.samples, 4);
    }

    #[test]
    fn invalid_views_are_rejected() {
        let metrics = CpuFrameMetrics.plane_metrics(
            PlaneView::new(&[0; 4], 4, 4, 4),
            PlaneView::new(&[0; 4], 4, 4, 4),
        );
        assert!(metrics.is_err());
    }
}
