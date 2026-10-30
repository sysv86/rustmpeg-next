// SPDX-License-Identifier: Apache-2.0
//! High-level accelerator that combines the compute kernels and the scheduler.
//!
//! [`Accelerator`] is the object a codec-facing application holds. It owns the
//! Vulkan device, the compiled kernels, and a calibrated [`Scheduler`], and it
//! implements the accelerator traits from `rustmpeg-video`, so it can be passed
//! directly to `VvcEncoder::encode_rgba_with` or
//! `Av2Encoder::set_frame_metrics`.
//!
//! Every operation consults the scheduler. If the GPU is predicted to be slower,
//! or if it fails at run time, the same operation runs on the CPU reference and
//! the GPU path is disabled for the rest of the session. The output never
//! changes: the CPU and GPU implementations are bit-exact.

use std::sync::atomic::{AtomicBool, Ordering};

use rustmpeg_video::accelerator::{
    AcceleratorError, CpuFrameMetrics, CpuRgbaToI420, FrameMetrics, FrameMetricsProvider,
    PlaneMetrics, PlaneView, RgbaToI420Converter,
};
use rustmpeg_video::{I420Frame, accelerator};

use crate::compute::color::i420_to_rgba_cpu;
use crate::compute::{
    Compute, I420Converter, PlaneMetricsKernel, RgbaConverter, accelerator_error,
};
use crate::error::VulkanError;
use crate::scheduler::{FrameSize, Placement, Scheduler, Stage};

/// A calibrated CPU/GPU accelerator for frame-level video stages.
#[derive(Debug)]
pub struct Accelerator {
    compute: Compute,
    scheduler: Scheduler,
    rgba: RgbaConverter,
    i420: I420Converter,
    metrics: PlaneMetricsKernel,
    gpu_failed: AtomicBool,
}

impl Accelerator {
    /// Creates an accelerator on the default Vulkan device.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Unavailable`] when no Vulkan adapter is present,
    /// or [`VulkanError::Kernel`] when a kernel does not compile. A machine
    /// without a GPU should use [`CpuRgbaToI420`] and [`CpuFrameMetrics`]
    /// directly.
    pub fn new() -> Result<Self, VulkanError> {
        Self::with_compute(Compute::new()?)
    }

    /// Creates an accelerator on an existing compute device.
    ///
    /// The kernels are compiled immediately and the scheduler is calibrated
    /// against the CPU reference. A calibration failure is logged and leaves
    /// the accelerator on the CPU path instead of failing construction.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Kernel`] when a shader does not compile.
    pub fn with_compute(compute: Compute) -> Result<Self, VulkanError> {
        let rgba = RgbaConverter::new(&compute)?;
        let i420 = I420Converter::new(&compute)?;
        let metrics = PlaneMetricsKernel::new(&compute)?;
        let mut scheduler = Scheduler::with_compute(&compute);
        if let Err(error) = scheduler.calibrate(&compute) {
            tracing::warn!(%error, "GPU calibration failed; using the CPU path");
            scheduler.disable_gpu();
        }
        Ok(Self {
            compute,
            scheduler,
            rgba,
            i420,
            metrics,
            gpu_failed: AtomicBool::new(false),
        })
    }

    /// The underlying compute device.
    pub fn compute(&self) -> &Compute {
        &self.compute
    }

    /// The current CPU/GPU placement policy and measured costs.
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Re-runs calibration and re-enables the GPU for a fresh measurement.
    ///
    /// # Errors
    ///
    /// Propagates kernel, buffer, and dispatch failures. A mismatch against the
    /// CPU reference disables the GPU and returns
    /// [`VulkanError::Execution`].
    pub fn recalibrate(&mut self) -> Result<(), VulkanError> {
        self.scheduler = Scheduler::with_compute(&self.compute);
        self.gpu_failed.store(false, Ordering::Relaxed);
        self.scheduler.calibrate(&self.compute)
    }

    /// Permanently disables the GPU path for this accelerator.
    pub fn disable_gpu(&mut self) {
        self.scheduler.disable_gpu();
        self.gpu_failed.store(true, Ordering::Relaxed);
    }

    /// Converts premultiplied RGBA8 to I420, choosing the execution unit.
    ///
    /// # Errors
    ///
    /// Returns [`AcceleratorError`] when the input is invalid. A GPU failure is
    /// logged, disables the GPU path, and retries on the CPU.
    pub fn convert_rgba(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<I420Frame, AcceleratorError> {
        if self.gpu_usable(Stage::RgbaToI420, FrameSize::new(width, height)) {
            match self.rgba.convert(rgba, width, height) {
                Ok(frame) => return Ok(frame),
                Err(error) => self.fallback(&accelerator_error(error)),
            }
        }
        CpuRgbaToI420.rgba_to_i420(rgba, width, height)
    }

    /// Converts an I420 frame to RGBA8, choosing the execution unit.
    ///
    /// # Errors
    ///
    /// Returns [`AcceleratorError`] when the frame is malformed. A GPU failure
    /// is logged and retried on the CPU.
    pub fn convert_i420(&self, frame: &I420Frame) -> Result<Vec<u8>, AcceleratorError> {
        if self.gpu_usable(Stage::I420ToRgba, FrameSize::new(frame.width, frame.height)) {
            match self.i420.convert(frame) {
                Ok(rgba) => return Ok(rgba),
                Err(error) => self.fallback(&accelerator_error(error)),
            }
        }
        i420_to_rgba_cpu(frame).map_err(accelerator_error)
    }

    /// Computes full-frame metrics for two I420 frames, choosing the execution
    /// unit.
    ///
    /// # Errors
    ///
    /// Returns [`AcceleratorError`] when either frame is malformed. A GPU
    /// failure is logged and retried on the CPU.
    pub fn i420_metrics(
        &self,
        reference: &I420Frame,
        distorted: &I420Frame,
    ) -> Result<FrameMetrics, AcceleratorError> {
        accelerator::i420_metrics(self, reference, distorted)
    }

    fn gpu_usable(&self, stage: Stage, size: FrameSize) -> bool {
        !self.gpu_failed.load(Ordering::Relaxed)
            && self.scheduler.place(stage, size) == Placement::Gpu
    }

    fn fallback(&self, error: &AcceleratorError) {
        tracing::warn!(%error, "GPU stage failed; disabling the GPU path and using the CPU");
        self.gpu_failed.store(true, Ordering::Relaxed);
    }
}

impl RgbaToI420Converter for Accelerator {
    fn rgba_to_i420(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<I420Frame, AcceleratorError> {
        self.convert_rgba(rgba, width, height)
    }
}

impl FrameMetricsProvider for Accelerator {
    fn frame_metrics(
        &self,
        reference: [PlaneView<'_>; 3],
        distorted: [PlaneView<'_>; 3],
    ) -> Result<FrameMetrics, AcceleratorError> {
        let size = FrameSize::new(reference[0].width, reference[0].height);
        if self.gpu_usable(Stage::PlaneMetrics, size) {
            match self.metrics.frame_metrics(reference, distorted) {
                Ok(planes) => return Ok(FrameMetrics { planes }),
                Err(error) => self.fallback(&accelerator_error(error)),
            }
        }
        CpuFrameMetrics.frame_metrics(reference, distorted)
    }

    fn plane_metrics(
        &self,
        reference: PlaneView<'_>,
        distorted: PlaneView<'_>,
    ) -> Result<PlaneMetrics, AcceleratorError> {
        let size = FrameSize::new(reference.width, reference.height);
        if self.gpu_usable(Stage::PlaneMetrics, size) {
            match self.metrics.compute(reference, distorted) {
                Ok(metrics) => return Ok(metrics),
                Err(error) => self.fallback(&accelerator_error(error)),
            }
        }
        CpuFrameMetrics.plane_metrics(reference, distorted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba(width: u32, height: u32) -> Vec<u8> {
        (0..width * height)
            .flat_map(|index| {
                [
                    u8::try_from((index * 11) % 256).unwrap_or(0),
                    u8::try_from((index * 29 + 7) % 256).unwrap_or(0),
                    u8::try_from((index * 47 + 13) % 256).unwrap_or(0),
                    255,
                ]
            })
            .collect()
    }

    #[test]
    fn accelerator_matches_the_cpu_reference() {
        let Ok(accelerator) = Accelerator::new() else {
            return;
        };
        let source = rgba(37, 21);
        let gpu = accelerator
            .convert_rgba(&source, 37, 21)
            .expect("conversion succeeds");
        let cpu = CpuRgbaToI420
            .rgba_to_i420(&source, 37, 21)
            .expect("reference");
        assert_eq!(gpu, cpu);

        let back = accelerator.convert_i420(&gpu).expect("conversion succeeds");
        let expected = crate::compute::color::i420_to_rgba_cpu(&gpu).expect("reference");
        assert_eq!(back, expected);
    }

    #[test]
    fn disabled_gpu_still_produces_the_cpu_result() {
        let Ok(mut accelerator) = Accelerator::new() else {
            return;
        };
        accelerator.disable_gpu();
        let source = rgba(16, 16);
        let result = accelerator
            .convert_rgba(&source, 16, 16)
            .expect("conversion succeeds");
        assert_eq!(
            result,
            CpuRgbaToI420
                .rgba_to_i420(&source, 16, 16)
                .expect("reference")
        );
    }

    #[test]
    fn malformed_input_is_reported_by_the_cpu_fallback() {
        let Ok(accelerator) = Accelerator::new() else {
            return;
        };
        let error = accelerator
            .convert_rgba(&[0; 3], 2, 2)
            .expect_err("short input must fail");
        assert!(matches!(error, AcceleratorError::InvalidInput(_)));
    }
}
