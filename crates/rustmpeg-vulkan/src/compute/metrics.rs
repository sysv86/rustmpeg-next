// SPDX-License-Identifier: Apache-2.0
//! Frame-level distortion metrics on Vulkan Compute.
//!
//! The kernel computes sum of squared errors (SSE) and sum of absolute
//! differences (SAD) for one 8-bit plane pair. Each workgroup owns one row, so
//! the reduction is race-free: threads aggregate into workgroup atomics and
//! thread zero writes the row partial. The host sums the partials in `u64`,
//! which avoids 32-bit overflow for 16K pictures.
//!
//! All three planes of a frame are recorded into one [`Batch`], so a frame
//! measurement performs one queue submission instead of three.

use rustmpeg_video::accelerator::{
    AcceleratorError, FrameMetrics, FrameMetricsProvider, PlaneMetrics, PlaneView,
};

use crate::compute::{Compute, GpuBuffer, Kernel};
use crate::error::VulkanError;

const SHADER: &str = r#"
struct Dims {
    width: u32,
    height: u32,
    stride: u32,
    _reserved: u32,
};

@group(0) @binding(0) var<storage, read> reference: array<u32>;
@group(0) @binding(1) var<storage, read> distorted: array<u32>;
@group(0) @binding(2) var<storage, read_write> partials: array<u32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> row_sse: atomic<u32>;
var<workgroup> row_sad: atomic<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if (lane == 0u) {
        atomicStore(&row_sse, 0u);
        atomicStore(&row_sad, 0u);
    }
    workgroupBarrier();

    let row = id.x;
    if (row < dims.height) {
        var column = lane;
        loop {
            if (column >= dims.width) {
                break;
            }
            let index = row * dims.stride + column;
            let a = i32((reference[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu);
            let b = i32((distorted[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu);
            let difference = a - b;
            atomicAdd(&row_sse, u32(difference * difference));
            atomicAdd(&row_sad, u32(abs(difference)));
            column = column + 256u;
        }
    }
    workgroupBarrier();

    if (lane == 0u) {
        partials[row * 2u] = atomicLoad(&row_sse);
        partials[row * 2u + 1u] = atomicLoad(&row_sad);
    }
}
"#;

/// A compiled per-plane SSE/SAD kernel.
///
/// The kernel owns a handle to one [`Compute`] device and can be reused for
/// every frame.
#[derive(Debug)]
pub struct PlaneMetricsKernel {
    compute: Compute,
    kernel: Kernel,
}

/// Buffers and dimensions prepared for one plane pair.
#[derive(Debug)]
struct PlaneJob {
    reference: GpuBuffer,
    distorted: GpuBuffer,
    partials: GpuBuffer,
    dims: GpuBuffer,
    partials_len: usize,
    samples: u64,
    height: u32,
}

impl PlaneMetricsKernel {
    /// Compiles the metrics kernel.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Kernel`] when the shader does not compile.
    pub fn new(compute: &Compute) -> Result<Self, VulkanError> {
        Ok(Self {
            compute: compute.clone(),
            kernel: compute.kernel(SHADER, "main")?,
        })
    }

    /// Computes SSE, SAD, and the sample count for one plane pair.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::InvalidConfig`] for malformed views,
    /// [`VulkanError::DispatchTooLarge`] when the plane is taller than the
    /// device dispatch limit, and propagates buffer and execution failures.
    pub fn compute(
        &self,
        reference: PlaneView<'_>,
        distorted: PlaneView<'_>,
    ) -> Result<PlaneMetrics, VulkanError> {
        let Some(job) = self.prepare(reference, distorted)? else {
            return Ok(PlaneMetrics::default());
        };
        let mut batch = self.compute.batch();
        batch.dispatch(
            &self.kernel,
            &[&job.reference, &job.distorted, &job.partials, &job.dims],
            [job.height, 1, 1],
        )?;
        let read = batch.read(&job.partials, job.partials_len)?;
        let results = batch.submit()?;
        Ok(reduce(&results[read], job.samples))
    }

    /// Computes all three planes in one submission.
    ///
    /// Empty planes (monochrome input) contribute [`PlaneMetrics::default`].
    ///
    /// # Errors
    ///
    /// Propagates the same validation, buffer, and execution errors as
    /// [`PlaneMetricsKernel::compute`].
    pub fn frame_metrics(
        &self,
        reference: [PlaneView<'_>; 3],
        distorted: [PlaneView<'_>; 3],
    ) -> Result<[PlaneMetrics; 3], VulkanError> {
        let mut jobs: [Option<PlaneJob>; 3] = [None, None, None];
        for index in 0..3 {
            jobs[index] = self.prepare(reference[index], distorted[index])?;
        }

        let mut batch = self.compute.batch();
        let mut reads = [usize::MAX; 3];
        for (index, job) in jobs.iter().enumerate() {
            if let Some(job) = job {
                batch.dispatch(
                    &self.kernel,
                    &[&job.reference, &job.distorted, &job.partials, &job.dims],
                    [job.height, 1, 1],
                )?;
                reads[index] = batch.read(&job.partials, job.partials_len)?;
            }
        }
        let results = batch.submit()?;

        let mut planes = [PlaneMetrics::default(); 3];
        for (index, job) in jobs.iter().enumerate() {
            if let Some(job) = job {
                planes[index] = reduce(&results[reads[index]], job.samples);
            }
        }
        Ok(planes)
    }

    fn prepare(
        &self,
        reference: PlaneView<'_>,
        distorted: PlaneView<'_>,
    ) -> Result<Option<PlaneJob>, VulkanError> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(VulkanError::InvalidConfig(format!(
                "plane dimensions differ: {}x{} and {}x{}",
                reference.width, reference.height, distorted.width, distorted.height
            )));
        }
        if !reference.is_valid() || !distorted.is_valid() {
            return Err(VulkanError::InvalidConfig(
                "plane view exceeds its backing buffer".to_owned(),
            ));
        }
        if reference.is_empty() {
            return Ok(None);
        }
        // One workgroup accumulates a whole row in a 32-bit atomic. The
        // largest possible squared difference is 255 * 255, so a row must be
        // narrow enough for the partial sum to fit in u32.
        if u64::from(reference.width) * 65_025 > u64::from(u32::MAX) {
            return Err(VulkanError::InvalidConfig(
                "plane row is too wide for the 32-bit metrics reduction".to_owned(),
            ));
        }
        let limit = self.compute.info().max_workgroups_per_dimension;
        if reference.height > limit {
            return Err(VulkanError::DispatchTooLarge {
                requested: [reference.height, 1, 1],
                limit,
            });
        }

        let reference_bytes = upload_bytes(&reference)?;
        let distorted_bytes = upload_bytes(&distorted)?;
        let partials_len = usize::try_from(u64::from(reference.height) * 8).map_err(|_| {
            VulkanError::Execution("metrics partial buffer is too large".to_owned())
        })?;

        Ok(Some(PlaneJob {
            reference: self.compute.upload(&reference_bytes)?,
            distorted: self.compute.upload(&distorted_bytes)?,
            partials: self.compute.allocate(partials_len)?,
            dims: self.compute.uniform(&dims_bytes(&reference))?,
            partials_len,
            samples: reference.samples(),
            height: reference.height,
        }))
    }
}

impl FrameMetricsProvider for PlaneMetricsKernel {
    fn plane_metrics(
        &self,
        reference: PlaneView<'_>,
        distorted: PlaneView<'_>,
    ) -> Result<PlaneMetrics, AcceleratorError> {
        self.compute(reference, distorted)
            .map_err(crate::compute::accelerator_error)
    }

    fn frame_metrics(
        &self,
        reference: [PlaneView<'_>; 3],
        distorted: [PlaneView<'_>; 3],
    ) -> Result<FrameMetrics, AcceleratorError> {
        let planes = PlaneMetricsKernel::frame_metrics(self, reference, distorted)
            .map_err(crate::compute::accelerator_error)?;
        Ok(FrameMetrics { planes })
    }
}

fn reduce(bytes: &[u8], samples: u64) -> PlaneMetrics {
    let rows = bytes.len() / 8;
    let mut sse = 0_u64;
    let mut sad = 0_u64;
    for row in 0..rows {
        let offset = row * 8;
        let row_sse = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        let row_sad = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        sse += u64::from(row_sse);
        sad += u64::from(row_sad);
    }
    PlaneMetrics { sse, sad, samples }
}

fn upload_bytes(view: &PlaneView<'_>) -> Result<Vec<u8>, VulkanError> {
    let required = view
        .stride
        .checked_mul(view.height as usize)
        .ok_or_else(|| VulkanError::Execution("plane stride overflow".to_owned()))?;
    let available = required.min(view.data.len());
    let mut bytes = view.data[..available].to_vec();
    let aligned = (bytes.len() + 3) & !3;
    bytes.resize(aligned.max(4), 0);
    Ok(bytes)
}

fn dims_bytes(view: &PlaneView<'_>) -> [u8; 16] {
    let mut bytes = [0_u8; 16];
    bytes[0..4].copy_from_slice(&view.width.to_le_bytes());
    bytes[4..8].copy_from_slice(&view.height.to_le_bytes());
    let stride = u32::try_from(view.stride).unwrap_or(u32::MAX);
    bytes[8..12].copy_from_slice(&stride.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmpeg_video::accelerator::CpuFrameMetrics;

    fn plane_data(width: u32, height: u32, seed: u32) -> Vec<u8> {
        (0..width * height)
            .map(|index| u8::try_from((index * 37 + seed) % 256).unwrap_or(0))
            .collect()
    }

    #[test]
    fn gpu_metrics_match_the_cpu_reference() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let kernel = PlaneMetricsKernel::new(&compute).expect("kernel compiles");
        for (width, height) in [(16, 16), (17, 9), (64, 48), (256, 3)] {
            let reference = plane_data(width, height, 11);
            let distorted = plane_data(width, height, 29);
            let a = PlaneView::new(&reference, width, height, width as usize);
            let b = PlaneView::new(&distorted, width, height, width as usize);
            let gpu = kernel.compute(a, b).expect("dispatch");
            let cpu = CpuFrameMetrics.plane_metrics(a, b).expect("reference");
            assert_eq!(gpu, cpu, "mismatch for {width}x{height}");
        }
    }

    #[test]
    fn gpu_frame_metrics_batch_all_three_planes() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let kernel = PlaneMetricsKernel::new(&compute).expect("kernel compiles");
        let reference = [
            plane_data(64, 48, 1),
            plane_data(32, 24, 2),
            plane_data(32, 24, 3),
        ];
        let distorted = [
            plane_data(64, 48, 11),
            plane_data(32, 24, 12),
            plane_data(32, 24, 13),
        ];
        fn views(planes: &[Vec<u8>; 3]) -> [PlaneView<'_>; 3] {
            [
                PlaneView::new(&planes[0], 64, 48, 64),
                PlaneView::new(&planes[1], 32, 24, 32),
                PlaneView::new(&planes[2], 32, 24, 32),
            ]
        }
        let gpu = kernel
            .frame_metrics(views(&reference), views(&distorted))
            .expect("batch");
        let cpu = CpuFrameMetrics
            .frame_metrics(views(&reference), views(&distorted))
            .expect("reference");
        assert_eq!(gpu, cpu.planes);
    }

    #[test]
    fn gpu_metrics_honour_a_padded_stride() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let kernel = PlaneMetricsKernel::new(&compute).expect("kernel compiles");
        let (width, height, stride) = (13_usize, 7_usize, 32_usize);
        let mut reference = vec![0_u8; stride * height];
        let mut distorted = vec![0_u8; stride * height];
        for row in 0..height {
            for column in 0..width {
                reference[row * stride + column] =
                    u8::try_from((row * 17 + column) % 256).unwrap_or(0);
                distorted[row * stride + column] =
                    u8::try_from((row * 17 + column + 3) % 256).unwrap_or(0);
            }
            for column in width..stride {
                reference[row * stride + column] = 200;
                distorted[row * stride + column] = 1;
            }
        }
        let a = PlaneView::new(&reference, width as u32, height as u32, stride);
        let b = PlaneView::new(&distorted, width as u32, height as u32, stride);
        let gpu = kernel.compute(a, b).expect("dispatch");
        let cpu = CpuFrameMetrics.plane_metrics(a, b).expect("reference");
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn empty_planes_are_zero_without_a_dispatch() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let kernel = PlaneMetricsKernel::new(&compute).expect("kernel compiles");
        let empty = PlaneView::new(&[], 0, 0, 0);
        let metrics = kernel.compute(empty, empty).expect("empty plane");
        assert_eq!(metrics, PlaneMetrics::default());
    }

    #[test]
    fn mismatched_views_are_rejected() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let kernel = PlaneMetricsKernel::new(&compute).expect("kernel compiles");
        let a = [0_u8; 16];
        let b = [0_u8; 16];
        let error = kernel
            .compute(PlaneView::new(&a, 4, 4, 4), PlaneView::new(&b, 2, 8, 2))
            .expect_err("dimension mismatch must fail");
        assert!(matches!(error, VulkanError::InvalidConfig(_)));
    }
}
