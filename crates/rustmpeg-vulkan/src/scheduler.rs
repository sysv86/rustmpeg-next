// SPDX-License-Identifier: Apache-2.0
//! Deterministic CPU/GPU placement for frame-level compute stages.
//!
//! The scheduler decides, per stage and per frame size, whether the CPU or the
//! GPU should run the work. It is deliberately conservative: the GPU is chosen
//! only when its predicted total cost, including upload, dispatch, and
//! read-back, is lower than the CPU cost by a margin. Until a calibration has
//! measured real device timings, every stage stays on the CPU.
//!
//! Calibration runs both paths on synthetic frames of two sizes, verifies that
//! the GPU output is bit-exact with the CPU reference, and fits one cost line
//! per stage. A kernel that disagrees with the reference disables the GPU
//! instead of changing the output.

use std::time::Instant;

use rustmpeg_video::I420Frame;
use rustmpeg_video::accelerator::{
    CpuFrameMetrics, CpuRgbaToI420, FrameMetricsProvider, PlaneMetrics, PlaneView,
    RgbaToI420Converter,
};

use crate::VulkanError;
use crate::compute::Compute;
use crate::compute::color::{I420Converter, RgbaConverter, i420_to_rgba_cpu};
use crate::compute::metrics::PlaneMetricsKernel;

/// Relative margin by which the GPU must beat the CPU.
const GPU_MARGIN: f64 = 1.1;

/// Lower bound for a fitted cost slope, in nanoseconds per pixel.
const MIN_NS_PER_PIXEL: f64 = 1e-6;

/// Number of timed runs per measurement; the best run is kept.
const TIMED_RUNS: usize = 5;

/// Frame sizes used to fit the per-stage cost line.
const CALIBRATION_SIZES: [FrameSize; 2] = [FrameSize::new(128, 128), FrameSize::new(1920, 1080)];

/// A frame-level stage that can run on either the CPU or the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Premultiplied RGBA8 to I420 conversion.
    RgbaToI420,
    /// I420 to RGBA8 conversion.
    I420ToRgba,
    /// Per-plane SSE/SAD measurement over a reference and a distorted frame.
    PlaneMetrics,
}

impl Stage {
    /// Short stable name used in logs and diagnostics.
    pub const fn label(self) -> &'static str {
        match self {
            Self::RgbaToI420 => "rgba_to_i420",
            Self::I420ToRgba => "i420_to_rgba",
            Self::PlaneMetrics => "plane_metrics",
        }
    }
}

/// Where a stage should execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Run on the CPU reference implementation.
    Cpu,
    /// Run on the Vulkan Compute device.
    Gpu,
}

/// Frame dimensions in luma samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSize {
    /// Frame width in luma samples.
    pub width: u32,
    /// Frame height in luma samples.
    pub height: u32,
}

impl FrameSize {
    /// Creates a frame size from its dimensions.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Number of luma pixels, or `0` when the product does not fit in `u64`.
    pub fn pixels(&self) -> u64 {
        pixels_from(u64::from(self.width), u64::from(self.height))
    }

    /// `true` when either dimension is zero.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Multiplies two dimensions, reporting an unrepresentable product as `0`.
fn pixels_from(width: u64, height: u64) -> u64 {
    width.checked_mul(height).unwrap_or(0)
}

/// Host CPU properties that influence the software cost of a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuProfile {
    /// Logical cores available to the process.
    pub logical_cores: usize,
    /// Width of the widest vector unit used by the software kernels, in bits.
    pub simd_bits: u32,
    /// Size of the level-1 data cache in bytes.
    pub l1_data_bytes: Option<u64>,
    /// Size of the level-2 cache in bytes.
    pub l2_bytes: Option<u64>,
    /// Size of the level-3 cache in bytes.
    pub l3_bytes: Option<u64>,
}

impl CpuProfile {
    /// Detects the host CPU profile.
    ///
    /// Cache sizes are read from `sysfs` on Linux and left as `None` when they
    /// cannot be determined. Every probe failure is ignored.
    pub fn detect() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        let caches = read_cache_entries();
        Self {
            logical_cores,
            simd_bits: detect_simd_bits(),
            l1_data_bytes: cache_bytes(&caches, 1),
            l2_bytes: cache_bytes(&caches, 2),
            l3_bytes: cache_bytes(&caches, 3),
        }
    }
}

/// Static properties of the Vulkan Compute device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuProfile {
    /// Adapter name reported by the driver.
    pub name: String,
    /// `true` for an integrated or virtual adapter.
    pub integrated: bool,
    /// Maximum size of a single buffer allocation.
    pub max_buffer_size: u64,
    /// Maximum workgroup size per dimension.
    pub max_workgroup_size: [u32; 3],
    /// Maximum invocations in one workgroup.
    pub max_invocations_per_workgroup: u32,
    /// Maximum dispatch grid per dimension.
    pub max_workgroups_per_dimension: u32,
}

impl GpuProfile {
    /// Copies the scheduling-relevant limits out of a [`ComputeInfo`].
    ///
    /// [`ComputeInfo`]: crate::compute::ComputeInfo
    pub fn from_info(info: &crate::compute::ComputeInfo) -> Self {
        Self {
            name: info.name.clone(),
            integrated: info.integrated,
            max_buffer_size: info.max_buffer_size,
            max_workgroup_size: info.max_workgroup_size,
            max_invocations_per_workgroup: info.max_invocations_per_workgroup,
            max_workgroups_per_dimension: info.max_workgroups_per_dimension,
        }
    }
}

/// Fitted cost of one stage on both execution units.
///
/// Both the CPU and the GPU cost are lines in the number of luma pixels: a
/// fixed per-call cost plus a per-pixel rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageCost {
    /// CPU cost that does not depend on the frame size, in nanoseconds.
    pub cpu_fixed_ns: f64,
    /// Measured CPU cost per luma pixel, in nanoseconds.
    pub cpu_ns_per_pixel: f64,
    /// GPU cost that does not depend on the frame size, in nanoseconds.
    pub gpu_fixed_ns: f64,
    /// GPU cost per luma pixel, in nanoseconds.
    pub gpu_ns_per_pixel: f64,
}

/// Fitted costs for every schedulable stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageCosts {
    /// Cost of RGBA to I420 conversion.
    pub rgba_to_i420: StageCost,
    /// Cost of I420 to RGBA conversion.
    pub i420_to_rgba: StageCost,
    /// Cost of whole-frame plane metrics (three plane calls).
    pub plane_metrics: StageCost,
}

impl StageCosts {
    /// Returns the cost entry for one stage.
    fn for_stage(&self, stage: Stage) -> StageCost {
        match stage {
            Stage::RgbaToI420 => self.rgba_to_i420,
            Stage::I420ToRgba => self.i420_to_rgba,
            Stage::PlaneMetrics => self.plane_metrics,
        }
    }
}

/// Chooses the execution unit for each frame-level stage.
///
/// A scheduler without a GPU profile, without calibrated costs, or with the
/// GPU explicitly disabled always places work on the CPU. Once a stage has
/// been calibrated, the GPU is selected only when its predicted cost is at
/// least ten percent below the CPU cost.
#[derive(Debug, Clone)]
pub struct Scheduler {
    cpu: CpuProfile,
    gpu: Option<GpuProfile>,
    gpu_enabled: bool,
    calibrated: bool,
    costs: Option<StageCosts>,
}

impl Scheduler {
    /// Creates a CPU-only scheduler with no GPU profile and no cost data.
    pub fn new() -> Self {
        Self {
            cpu: CpuProfile::detect(),
            gpu: None,
            gpu_enabled: false,
            calibrated: false,
            costs: None,
        }
    }

    /// Creates a scheduler bound to a compute device.
    ///
    /// The GPU stays unused until [`Scheduler::calibrate`] succeeds, because
    /// no costs are known yet.
    pub fn with_compute(compute: &Compute) -> Self {
        Self {
            cpu: CpuProfile::detect(),
            gpu: Some(GpuProfile::from_info(compute.info())),
            gpu_enabled: true,
            calibrated: false,
            costs: None,
        }
    }

    /// Creates a scheduler from an explicit CPU profile, GPU profile, and cost
    /// table.
    ///
    /// The scheduler counts as calibrated. The GPU starts enabled only when a
    /// GPU profile is supplied.
    pub fn with_costs(cpu: CpuProfile, gpu: Option<GpuProfile>, costs: StageCosts) -> Self {
        Self {
            cpu,
            gpu_enabled: gpu.is_some(),
            gpu,
            calibrated: true,
            costs: Some(costs),
        }
    }

    /// Returns the detected or supplied CPU profile.
    pub fn cpu_profile(&self) -> &CpuProfile {
        &self.cpu
    }

    /// Returns the GPU profile, if a device was bound.
    pub fn gpu_profile(&self) -> Option<&GpuProfile> {
        self.gpu.as_ref()
    }

    /// `true` when the GPU may be selected.
    pub fn gpu_enabled(&self) -> bool {
        self.gpu_enabled
    }

    /// `true` when stage costs have been fitted.
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Permanently disables the GPU for this scheduler.
    ///
    /// Calibration never re-enables it; a new scheduler is required to use the
    /// GPU again.
    pub fn disable_gpu(&mut self) {
        self.gpu_enabled = false;
    }

    /// Returns the fitted costs, if the scheduler is calibrated.
    pub fn costs(&self) -> Option<&StageCosts> {
        self.costs.as_ref()
    }

    /// Measures and fits the per-stage costs on `compute`.
    ///
    /// Both paths run on synthetic frames of two sizes. Every GPU result is
    /// compared with the CPU reference; a mismatch disables the GPU and returns
    /// [`VulkanError::Execution`], so a disagreeing kernel can never change the
    /// output.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Unavailable`] when the scheduler has no GPU
    /// profile, propagates kernel compilation, buffer, dispatch, and read-back
    /// failures, and returns [`VulkanError::Execution`] when a kernel disagrees
    /// with the CPU reference.
    pub fn calibrate(&mut self, compute: &Compute) -> Result<(), VulkanError> {
        if self.gpu.is_none() {
            return Err(VulkanError::Unavailable(
                "scheduler has no GPU profile".to_owned(),
            ));
        }

        let downscale = RgbaConverter::new(compute)?;
        let upscale = I420Converter::new(compute)?;
        let metrics = PlaneMetricsKernel::new(compute)?;

        let mut rgba_inputs = Vec::with_capacity(CALIBRATION_SIZES.len());
        let mut references = Vec::with_capacity(CALIBRATION_SIZES.len());
        for size in CALIBRATION_SIZES {
            let rgba = synthetic_rgba(size);
            let reference = rgba_to_i420_cpu(&rgba, size.width, size.height)?;
            rgba_inputs.push(rgba);
            references.push(reference);
        }

        // Raise the GPU clocks and warm the allocator before any measurement.
        // A cold laptop GPU can otherwise report a several-fold slower rate,
        // which biases the scheduler permanently towards the CPU.
        let large = CALIBRATION_SIZES[1];
        let large_rgba = &rgba_inputs[1];
        let large_reference = &references[1];
        let large_distorted = perturb(large_reference);
        for _ in 0..8 {
            downscale.convert(large_rgba, large.width, large.height)?;
            upscale.convert(large_reference)?;
            frame_metrics_gpu(&metrics, large_reference, &large_distorted)?;
        }

        // RGBA -> I420.
        let mut down_cpu_ns = [0.0_f64; CALIBRATION_SIZES.len()];
        let mut down_gpu_ns = [0.0_f64; CALIBRATION_SIZES.len()];
        for (index, size) in CALIBRATION_SIZES.iter().enumerate() {
            let rgba = &rgba_inputs[index];
            let reference = &references[index];

            if index == 0 {
                downscale.convert(rgba, size.width, size.height)?;
            }
            let produced = downscale.convert(rgba, size.width, size.height)?;
            if produced != *reference {
                self.disable_gpu();
                return Err(VulkanError::Execution(
                    "RGBA->I420 kernel disagrees with the CPU reference".to_owned(),
                ));
            }

            down_cpu_ns[index] = best_of_three(|| rgba_to_i420_cpu(rgba, size.width, size.height))?;
            down_gpu_ns[index] =
                best_of_three(|| downscale.convert(rgba, size.width, size.height))?;
        }

        // I420 -> RGBA.
        let mut up_cpu_ns = [0.0_f64; CALIBRATION_SIZES.len()];
        let mut up_gpu_ns = [0.0_f64; CALIBRATION_SIZES.len()];
        for (index, reference) in references.iter().enumerate() {
            let produced = upscale.convert(reference)?;
            let expected = i420_to_rgba_cpu(reference)?;
            if produced != expected {
                self.disable_gpu();
                return Err(VulkanError::Execution(
                    "I420->RGBA kernel disagrees with the CPU reference".to_owned(),
                ));
            }

            up_cpu_ns[index] = best_of_three(|| i420_to_rgba_cpu(reference))?;
            up_gpu_ns[index] = best_of_three(|| upscale.convert(reference))?;
        }

        // Plane metrics over the three planes of a whole frame.
        let mut metrics_cpu_ns = [0.0_f64; CALIBRATION_SIZES.len()];
        let mut metrics_gpu_ns = [0.0_f64; CALIBRATION_SIZES.len()];
        for (index, reference) in references.iter().enumerate() {
            let distorted = perturb(reference);
            if frame_metrics_cpu(reference, &distorted)?
                != frame_metrics_gpu(&metrics, reference, &distorted)?
            {
                self.disable_gpu();
                return Err(VulkanError::Execution(
                    "plane metrics kernel disagrees with the CPU reference".to_owned(),
                ));
            }

            metrics_cpu_ns[index] = best_of_three(|| frame_metrics_cpu(reference, &distorted))?;
            metrics_gpu_ns[index] =
                best_of_three(|| frame_metrics_gpu(&metrics, reference, &distorted))?;
        }

        let small_px = CALIBRATION_SIZES[0].pixels() as f64;
        let large_px = CALIBRATION_SIZES[1].pixels() as f64;
        let costs = StageCosts {
            rgba_to_i420: stage_cost(
                down_cpu_ns[0],
                down_cpu_ns[1],
                down_gpu_ns[0],
                down_gpu_ns[1],
                small_px,
                large_px,
            ),
            i420_to_rgba: stage_cost(
                up_cpu_ns[0],
                up_cpu_ns[1],
                up_gpu_ns[0],
                up_gpu_ns[1],
                small_px,
                large_px,
            ),
            plane_metrics: stage_cost(
                metrics_cpu_ns[0],
                metrics_cpu_ns[1],
                metrics_gpu_ns[0],
                metrics_gpu_ns[1],
                small_px,
                large_px,
            ),
        };

        self.costs = Some(costs);
        self.calibrated = true;

        let adapter = self.gpu.as_ref().map_or("unknown", |gpu| gpu.name.as_str());
        tracing::debug!(
            adapter,
            rgba_to_i420_cpu_fixed_ns = costs.rgba_to_i420.cpu_fixed_ns,
            rgba_to_i420_cpu_ns_per_pixel = costs.rgba_to_i420.cpu_ns_per_pixel,
            rgba_to_i420_gpu_ns_per_pixel = costs.rgba_to_i420.gpu_ns_per_pixel,
            rgba_to_i420_gpu_fixed_ns = costs.rgba_to_i420.gpu_fixed_ns,
            i420_to_rgba_cpu_fixed_ns = costs.i420_to_rgba.cpu_fixed_ns,
            i420_to_rgba_cpu_ns_per_pixel = costs.i420_to_rgba.cpu_ns_per_pixel,
            i420_to_rgba_gpu_ns_per_pixel = costs.i420_to_rgba.gpu_ns_per_pixel,
            i420_to_rgba_gpu_fixed_ns = costs.i420_to_rgba.gpu_fixed_ns,
            plane_metrics_cpu_fixed_ns = costs.plane_metrics.cpu_fixed_ns,
            plane_metrics_cpu_ns_per_pixel = costs.plane_metrics.cpu_ns_per_pixel,
            plane_metrics_gpu_ns_per_pixel = costs.plane_metrics.gpu_ns_per_pixel,
            plane_metrics_gpu_fixed_ns = costs.plane_metrics.gpu_fixed_ns,
            "calibrated CPU and GPU stage costs"
        );

        Ok(())
    }

    /// Predicts the CPU and GPU cost of a stage in nanoseconds.
    ///
    /// Returns `None` when the scheduler is uncalibrated or the frame has no
    /// pixels. The estimate stays available after [`Scheduler::disable_gpu`],
    /// which lets a caller log the potential gain.
    pub fn estimate(&self, stage: Stage, size: FrameSize) -> Option<(f64, f64)> {
        if !self.calibrated {
            return None;
        }
        let pixels = size.pixels();
        if pixels == 0 {
            return None;
        }
        let cost = self.costs?.for_stage(stage);
        let pixels = pixels as f64;
        Some((
            cost.cpu_fixed_ns + cost.cpu_ns_per_pixel * pixels,
            cost.gpu_fixed_ns + cost.gpu_ns_per_pixel * pixels,
        ))
    }

    /// Chooses the execution unit for one stage and frame size.
    ///
    /// Returns [`Placement::Cpu`] when the GPU is disabled or uncalibrated, the
    /// frame is empty, a cost is unusable, or the predicted GPU saving is below
    /// the ten percent margin.
    pub fn place(&self, stage: Stage, size: FrameSize) -> Placement {
        if !self.gpu_enabled || !self.calibrated || size.is_empty() {
            return Placement::Cpu;
        }
        let Some((cpu_ns, gpu_ns)) = self.estimate(stage, size) else {
            return Placement::Cpu;
        };
        if !cpu_ns.is_finite() || cpu_ns <= 0.0 {
            return Placement::Cpu;
        }
        if !gpu_ns.is_finite() || gpu_ns < 0.0 {
            return Placement::Cpu;
        }
        if gpu_ns * GPU_MARGIN < cpu_ns {
            Placement::Gpu
        } else {
            Placement::Cpu
        }
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts RGBA on the CPU, mapping the codec-facing error into the backend
/// error used by the scheduler.
fn rgba_to_i420_cpu(rgba: &[u8], width: u32, height: u32) -> Result<I420Frame, VulkanError> {
    CpuRgbaToI420
        .rgba_to_i420(rgba, width, height)
        .map_err(|error| VulkanError::Execution(error.to_string()))
}

/// Fits one stage cost from the two calibration measurements.
fn stage_cost(
    cpu_small_ns: f64,
    cpu_large_ns: f64,
    gpu_small_ns: f64,
    gpu_large_ns: f64,
    small_px: f64,
    large_px: f64,
) -> StageCost {
    let (cpu_ns_per_pixel, cpu_fixed_ns) = fit_line(cpu_small_ns, cpu_large_ns, small_px, large_px);
    let (gpu_ns_per_pixel, gpu_fixed_ns) = fit_line(gpu_small_ns, gpu_large_ns, small_px, large_px);
    StageCost {
        cpu_fixed_ns,
        cpu_ns_per_pixel,
        gpu_fixed_ns,
        gpu_ns_per_pixel,
    }
}

/// Fits a cost line through two measured points.
///
/// A non-finite slope falls back to the large-frame rate, and both parameters
/// are clamped so that a noisy measurement cannot produce a negative cost.
fn fit_line(small_ns: f64, large_ns: f64, small_px: f64, large_px: f64) -> (f64, f64) {
    let span = large_px - small_px;
    let slope = if span > 0.0 {
        (large_ns - small_ns) / span
    } else {
        f64::NAN
    };
    if !slope.is_finite() {
        let rate = if large_px > 0.0 {
            (large_ns / large_px).max(MIN_NS_PER_PIXEL)
        } else {
            MIN_NS_PER_PIXEL
        };
        return (rate, 0.0);
    }
    let slope = slope.max(MIN_NS_PER_PIXEL);
    let fixed = (small_ns - slope * small_px).max(0.0);
    (slope, fixed)
}

/// Returns the fastest wall-clock time of [`TIMED_RUNS`] calls to `run`, in
/// nanoseconds.
fn best_of_three<T>(mut run: impl FnMut() -> Result<T, VulkanError>) -> Result<f64, VulkanError> {
    // One untimed call raises the GPU clocks and warms the allocator before the
    // measurements, so a cold device does not bias the cost model towards the
    // CPU.
    let warmup = run()?;
    let _ = std::hint::black_box(&warmup);

    let mut best = f64::INFINITY;
    for _ in 0..TIMED_RUNS {
        let start = Instant::now();
        let value = run()?;
        let elapsed_ns = start.elapsed().as_secs_f64() * 1e9;
        let _ = std::hint::black_box(&value);
        if elapsed_ns < best {
            best = elapsed_ns;
        }
    }
    Ok(best)
}

/// Builds deterministic RGBA input where every pixel carries a distinct value.
fn synthetic_rgba(size: FrameSize) -> Vec<u8> {
    let count = usize::try_from(size.pixels()).unwrap_or(0);
    let mut rgba = Vec::with_capacity(count.saturating_mul(4));
    for index in 0..count {
        rgba.push(index.wrapping_mul(11).wrapping_add(3) as u8);
        rgba.push(index.wrapping_mul(29).wrapping_add(7) as u8);
        rgba.push(index.wrapping_mul(47).wrapping_add(13) as u8);
        rgba.push(index.wrapping_mul(61).wrapping_add(3) as u8);
    }
    rgba
}

/// Adds three to every seventh byte of every plane, wrapping at 255.
fn perturb(frame: &I420Frame) -> I420Frame {
    let mut distorted = frame.clone();
    for plane in [&mut distorted.y, &mut distorted.u, &mut distorted.v] {
        for (index, sample) in plane.iter_mut().enumerate() {
            if index % 7 == 0 {
                *sample = sample.wrapping_add(3);
            }
        }
    }
    distorted
}

/// Builds the three plane-view pairs of a reference and a distorted frame.
fn plane_pairs<'a>(
    reference: &'a I420Frame,
    distorted: &'a I420Frame,
) -> [(PlaneView<'a>, PlaneView<'a>); 3] {
    let (chroma_width, chroma_height) = reference.chroma_size();
    [
        plane_pair(
            (reference.y.as_slice(), distorted.y.as_slice()),
            (reference.width, reference.height),
        ),
        plane_pair(
            (reference.u.as_slice(), distorted.u.as_slice()),
            (chroma_width, chroma_height),
        ),
        plane_pair(
            (reference.v.as_slice(), distorted.v.as_slice()),
            (chroma_width, chroma_height),
        ),
    ]
}

/// Builds one plane-view pair with a tightly packed stride.
fn plane_pair<'a>(
    planes: (&'a [u8], &'a [u8]),
    dimensions: (u32, u32),
) -> (PlaneView<'a>, PlaneView<'a>) {
    let (width, height) = dimensions;
    let stride = width as usize;
    (
        PlaneView::new(planes.0, width, height, stride),
        PlaneView::new(planes.1, width, height, stride),
    )
}

/// Computes whole-frame plane metrics on the CPU.
fn frame_metrics_cpu(
    reference: &I420Frame,
    distorted: &I420Frame,
) -> Result<[PlaneMetrics; 3], VulkanError> {
    let mut metrics = [PlaneMetrics::default(); 3];
    for (slot, (a, b)) in metrics.iter_mut().zip(plane_pairs(reference, distorted)) {
        *slot = CpuFrameMetrics
            .plane_metrics(a, b)
            .map_err(|error| VulkanError::Execution(error.to_string()))?;
    }
    Ok(metrics)
}

/// Computes whole-frame plane metrics on the GPU.
fn frame_metrics_gpu(
    kernel: &PlaneMetricsKernel,
    reference: &I420Frame,
    distorted: &I420Frame,
) -> Result<[PlaneMetrics; 3], VulkanError> {
    let mut metrics = [PlaneMetrics::default(); 3];
    for (slot, (a, b)) in metrics.iter_mut().zip(plane_pairs(reference, distorted)) {
        *slot = kernel.compute(a, b)?;
    }
    Ok(metrics)
}

/// One cache level reported by the host.
#[derive(Debug, Clone, Copy)]
struct CacheEntry {
    level: u32,
    data: bool,
    bytes: u64,
}

/// Reads the host cache hierarchy from `sysfs`.
#[cfg(target_os = "linux")]
fn read_cache_entries() -> Vec<CacheEntry> {
    let mut entries = Vec::new();
    for index in 0..=4_u32 {
        let base = format!("/sys/devices/system/cpu/cpu0/cache/index{index}");
        let Some(level) = read_trimmed(&format!("{base}/level")).and_then(|text| text.parse().ok())
        else {
            continue;
        };
        let kind = read_trimmed(&format!("{base}/type")).unwrap_or_default();
        let Some(bytes) =
            read_trimmed(&format!("{base}/size")).and_then(|text| parse_cache_size(&text))
        else {
            continue;
        };
        entries.push(CacheEntry {
            level,
            data: kind.eq_ignore_ascii_case("Data"),
            bytes,
        });
    }
    entries
}

/// Returns no cache data on platforms without a `sysfs` cache hierarchy.
#[cfg(not(target_os = "linux"))]
fn read_cache_entries() -> Vec<CacheEntry> {
    Vec::new()
}

/// Reads one trimmed `sysfs` attribute.
#[cfg(target_os = "linux")]
fn read_trimmed(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|text| text.trim().to_owned())
}

/// Parses a `sysfs` cache size such as `32K`, `512K`, or `8M`.
#[cfg(target_os = "linux")]
fn parse_cache_size(text: &str) -> Option<u64> {
    let text = text.trim();
    let suffix = text.len().saturating_sub(1);
    let (digits, multiplier) = match text.as_bytes().last().copied()? {
        b'K' | b'k' => (&text[..suffix], 1024_u64),
        b'M' | b'm' => (&text[..suffix], 1024_u64 * 1024),
        b'G' | b'g' => (&text[..suffix], 1024_u64 * 1024 * 1024),
        _ => (text, 1_u64),
    };
    let value: u64 = digits.trim().parse().ok()?;
    value.checked_mul(multiplier)
}

/// Returns the size of the cache at `level`, preferring the data cache for
/// level one.
fn cache_bytes(entries: &[CacheEntry], level: u32) -> Option<u64> {
    if level == 1 {
        let data = entries
            .iter()
            .filter(|entry| entry.level == 1 && entry.data)
            .map(|entry| entry.bytes)
            .max();
        if data.is_some() {
            return data;
        }
    }
    entries
        .iter()
        .filter(|entry| entry.level == level)
        .map(|entry| entry.bytes)
        .max()
}

/// Detects the widest SIMD register width the software kernels can use.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn detect_simd_bits() -> u32 {
    if std::is_x86_feature_detected!("avx512f") {
        return 512;
    }
    if std::is_x86_feature_detected!("avx2") {
        return 256;
    }
    if std::is_x86_feature_detected!("sse2") {
        return 128;
    }
    64
}

/// Detects the widest SIMD register width the software kernels can use.
#[cfg(target_arch = "aarch64")]
fn detect_simd_bits() -> u32 {
    128
}

/// Detects the widest SIMD register width the software kernels can use.
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
fn detect_simd_bits() -> u32 {
    64
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL_HD: FrameSize = FrameSize::new(1920, 1080);
    const EIGHT_K: FrameSize = FrameSize::new(7680, 4320);

    fn cpu_profile() -> CpuProfile {
        CpuProfile {
            logical_cores: 8,
            simd_bits: 256,
            l1_data_bytes: Some(32 * 1024),
            l2_bytes: Some(512 * 1024),
            l3_bytes: Some(16 * 1024 * 1024),
        }
    }

    fn gpu_profile() -> GpuProfile {
        GpuProfile {
            name: "test adapter".to_owned(),
            integrated: false,
            max_buffer_size: 1 << 30,
            max_workgroup_size: [256, 256, 64],
            max_invocations_per_workgroup: 256,
            max_workgroups_per_dimension: 65_535,
        }
    }

    fn known_costs() -> StageCosts {
        let cost = StageCost {
            cpu_fixed_ns: 0.0,
            cpu_ns_per_pixel: 10.0,
            gpu_fixed_ns: 1_000_000.0,
            gpu_ns_per_pixel: 0.5,
        };
        StageCosts {
            rgba_to_i420: cost,
            i420_to_rgba: cost,
            plane_metrics: cost,
        }
    }

    fn calibrated_scheduler() -> Scheduler {
        Scheduler::with_costs(cpu_profile(), Some(gpu_profile()), known_costs())
    }

    #[test]
    fn known_costs_place_small_frames_on_the_cpu_and_large_frames_on_the_gpu() {
        let scheduler = calibrated_scheduler();
        for stage in [Stage::RgbaToI420, Stage::I420ToRgba, Stage::PlaneMetrics] {
            assert_eq!(
                scheduler.place(stage, FrameSize::new(64, 64)),
                Placement::Cpu,
                "{}",
                stage.label()
            );
            assert_eq!(
                scheduler.place(stage, FULL_HD),
                Placement::Gpu,
                "{}",
                stage.label()
            );
        }
    }

    #[test]
    fn new_scheduler_never_uses_the_gpu() {
        let scheduler = Scheduler::new();
        assert!(!scheduler.gpu_enabled());
        assert!(!scheduler.is_calibrated());
        assert!(scheduler.gpu_profile().is_none());
        assert!(scheduler.costs().is_none());
        for size in [FrameSize::new(1, 1), FULL_HD, EIGHT_K] {
            for stage in [Stage::RgbaToI420, Stage::I420ToRgba, Stage::PlaneMetrics] {
                assert_eq!(scheduler.place(stage, size), Placement::Cpu);
            }
        }
    }

    #[test]
    fn disabling_the_gpu_forces_the_cpu_even_for_eight_k() {
        let mut scheduler = calibrated_scheduler();
        assert_eq!(
            scheduler.place(Stage::PlaneMetrics, EIGHT_K),
            Placement::Gpu
        );

        scheduler.disable_gpu();
        assert!(!scheduler.gpu_enabled());
        assert!(scheduler.is_calibrated());
        for stage in [Stage::RgbaToI420, Stage::I420ToRgba, Stage::PlaneMetrics] {
            assert_eq!(scheduler.place(stage, EIGHT_K), Placement::Cpu);
        }
    }

    #[test]
    fn empty_frames_stay_on_the_cpu() {
        let scheduler = calibrated_scheduler();
        for size in [FrameSize::new(0, 1080), FrameSize::new(1920, 0)] {
            for stage in [Stage::RgbaToI420, Stage::I420ToRgba, Stage::PlaneMetrics] {
                assert_eq!(scheduler.place(stage, size), Placement::Cpu);
            }
        }
    }

    #[test]
    fn estimate_requires_calibration_and_matches_the_cost_model() {
        let size = FrameSize::new(64, 64);
        assert_eq!(Scheduler::new().estimate(Stage::RgbaToI420, size), None);

        let scheduler = calibrated_scheduler();
        let (cpu_ns, gpu_ns) = scheduler
            .estimate(Stage::PlaneMetrics, size)
            .expect("calibrated scheduler has costs");
        assert!((cpu_ns - 40_960.0).abs() < 1e-9);
        assert!((gpu_ns - 1_002_048.0).abs() < 1e-9);
        assert_eq!(
            scheduler.estimate(Stage::RgbaToI420, FrameSize::new(0, 0)),
            None
        );
    }

    #[test]
    fn estimate_stays_available_after_the_gpu_is_disabled() {
        let mut scheduler = calibrated_scheduler();
        scheduler.disable_gpu();
        assert!(scheduler.estimate(Stage::I420ToRgba, FULL_HD).is_some());
    }

    #[test]
    fn a_small_gpu_gain_stays_on_the_cpu_and_a_large_one_moves_to_the_gpu() {
        let size = FrameSize::new(640, 480);
        let five_percent = StageCost {
            cpu_fixed_ns: 0.0,
            cpu_ns_per_pixel: 1.0,
            gpu_fixed_ns: 0.0,
            gpu_ns_per_pixel: 0.95,
        };
        let twenty_percent = StageCost {
            cpu_fixed_ns: 0.0,
            cpu_ns_per_pixel: 1.0,
            gpu_fixed_ns: 0.0,
            gpu_ns_per_pixel: 0.8,
        };

        let timid = Scheduler::with_costs(
            cpu_profile(),
            Some(gpu_profile()),
            StageCosts {
                rgba_to_i420: five_percent,
                i420_to_rgba: five_percent,
                plane_metrics: five_percent,
            },
        );
        assert_eq!(timid.place(Stage::RgbaToI420, size), Placement::Cpu);

        let confident = Scheduler::with_costs(
            cpu_profile(),
            Some(gpu_profile()),
            StageCosts {
                rgba_to_i420: twenty_percent,
                i420_to_rgba: twenty_percent,
                plane_metrics: twenty_percent,
            },
        );
        assert_eq!(confident.place(Stage::RgbaToI420, size), Placement::Gpu);
    }

    #[test]
    fn a_scheduler_without_a_gpu_profile_stays_on_the_cpu() {
        let scheduler = Scheduler::with_costs(cpu_profile(), None, known_costs());
        assert!(!scheduler.gpu_enabled());
        assert!(scheduler.is_calibrated());
        assert!(scheduler.estimate(Stage::RgbaToI420, EIGHT_K).is_some());
        assert_eq!(scheduler.place(Stage::RgbaToI420, EIGHT_K), Placement::Cpu);
    }

    #[test]
    fn degenerate_costs_keep_the_cpu() {
        let size = FrameSize::new(320, 240);
        let cases = [
            StageCost {
                cpu_fixed_ns: 0.0,
                cpu_ns_per_pixel: 0.0,
                gpu_fixed_ns: 0.0,
                gpu_ns_per_pixel: 0.1,
            },
            StageCost {
                cpu_fixed_ns: 0.0,
                cpu_ns_per_pixel: 10.0,
                gpu_fixed_ns: 0.0,
                gpu_ns_per_pixel: -1.0,
            },
            StageCost {
                cpu_fixed_ns: 0.0,
                cpu_ns_per_pixel: 10.0,
                gpu_fixed_ns: f64::NAN,
                gpu_ns_per_pixel: 0.1,
            },
            StageCost {
                cpu_fixed_ns: 0.0,
                cpu_ns_per_pixel: f64::INFINITY,
                gpu_fixed_ns: 0.0,
                gpu_ns_per_pixel: 0.1,
            },
        ];
        for cost in cases {
            let scheduler = Scheduler::with_costs(
                cpu_profile(),
                Some(gpu_profile()),
                StageCosts {
                    rgba_to_i420: cost,
                    i420_to_rgba: cost,
                    plane_metrics: cost,
                },
            );
            assert_eq!(scheduler.place(Stage::RgbaToI420, size), Placement::Cpu);
        }
    }

    #[test]
    fn frame_size_arithmetic_uses_checked_maths() {
        assert_eq!(FULL_HD.pixels(), 2_073_600);
        assert_eq!(FrameSize::new(1, 1).pixels(), 1);
        assert!(FrameSize::new(0, 1080).is_empty());
        assert!(FrameSize::new(1920, 0).is_empty());
        assert!(!FrameSize::new(1, 1).is_empty());
        assert_eq!(pixels_from(u64::MAX, 2), 0);
        assert_eq!(pixels_from(2, u64::MAX), 0);
        assert_eq!(pixels_from(0, u64::MAX), 0);
    }

    #[test]
    fn stage_labels_are_stable() {
        assert_eq!(Stage::RgbaToI420.label(), "rgba_to_i420");
        assert_eq!(Stage::I420ToRgba.label(), "i420_to_rgba");
        assert_eq!(Stage::PlaneMetrics.label(), "plane_metrics");
    }

    #[test]
    fn gpu_profile_mirrors_the_compute_limits() {
        use crate::compute::ComputeInfo;

        let info = ComputeInfo {
            name: "adapter".to_owned(),
            vendor_id: 1,
            device_id: 2,
            backend: "Vulkan".to_owned(),
            device_type: "DiscreteGpu".to_owned(),
            integrated: false,
            driver: "driver".to_owned(),
            driver_info: "1.0".to_owned(),
            max_buffer_size: 4 << 30,
            max_storage_buffer_binding_size: 1 << 30,
            max_uniform_buffer_binding_size: 1 << 16,
            max_workgroup_size: [1024, 1024, 64],
            max_invocations_per_workgroup: 1024,
            max_workgroups_per_dimension: 65_535,
            max_storage_buffers_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 8,
            min_storage_buffer_offset_alignment: 16,
            min_uniform_buffer_offset_alignment: 64,
        };
        let profile = GpuProfile::from_info(&info);
        assert_eq!(profile.name, "adapter");
        assert_eq!(profile.max_buffer_size, 4 << 30);
        assert_eq!(profile.max_workgroup_size, [1024, 1024, 64]);
        assert_eq!(profile.max_invocations_per_workgroup, 1024);
        assert_eq!(profile.max_workgroups_per_dimension, 65_535);
        assert!(!profile.integrated);
    }

    #[test]
    fn cpu_detection_reports_plausible_values() {
        let profile = CpuProfile::detect();
        assert!(profile.logical_cores >= 1);
        assert!(matches!(profile.simd_bits, 64 | 128 | 256 | 512));
    }

    #[test]
    fn fit_gpu_recovers_a_two_point_line() {
        let (slope, fixed) = fit_line(1_000.0, 5_000.0, 4_096.0, 262_144.0);
        let expected = (5_000.0 - 1_000.0) / (262_144.0 - 4_096.0);
        assert!((slope - expected).abs() < 1e-12);
        assert!((fixed - (1_000.0 - expected * 4_096.0)).abs() < 1e-6);
        let reconstructed = fixed + slope * 262_144.0;
        assert!((reconstructed - 5_000.0).abs() < 1e-6);
    }

    #[test]
    fn fit_line_clamps_degenerate_lines() {
        let (slope, fixed) = fit_line(1_000.0, 5_000.0, 4_096.0, 4_096.0);
        assert!(slope.is_finite());
        assert!(slope >= MIN_NS_PER_PIXEL);
        assert!(fixed >= 0.0);

        let (slope, fixed) = fit_line(5_000.0, 1_000.0, 4_096.0, 262_144.0);
        assert!(slope >= MIN_NS_PER_PIXEL);
        assert!(fixed >= 0.0);

        let (slope, fixed) = fit_line(0.0, 0.0, 0.0, 0.0);
        assert!(slope >= MIN_NS_PER_PIXEL);
        assert!(fixed >= 0.0);
    }

    #[test]
    fn stage_cost_clamps_non_positive_measurements() {
        let cost = stage_cost(0.0, 0.0, 0.0, 0.0, 4_096.0, 262_144.0);
        assert!((cost.cpu_ns_per_pixel - MIN_NS_PER_PIXEL).abs() < 1e-18);
        assert!((cost.gpu_ns_per_pixel - MIN_NS_PER_PIXEL).abs() < 1e-18);
        assert!(cost.cpu_fixed_ns >= 0.0);
        assert!(cost.gpu_fixed_ns >= 0.0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_sizes_are_parsed_from_sysfs_units() {
        assert_eq!(parse_cache_size("32K"), Some(32 * 1024));
        assert_eq!(parse_cache_size("512K"), Some(512 * 1024));
        assert_eq!(parse_cache_size("8192K"), Some(8 * 1024 * 1024));
        assert_eq!(parse_cache_size("1M"), Some(1024 * 1024));
        assert_eq!(parse_cache_size("64"), Some(64));
        assert_eq!(parse_cache_size(""), None);
        assert_eq!(parse_cache_size("large"), None);
    }

    #[test]
    fn cache_lookup_prefers_the_level_one_data_cache() {
        let entries = [
            CacheEntry {
                level: 1,
                data: false,
                bytes: 64 * 1024,
            },
            CacheEntry {
                level: 1,
                data: true,
                bytes: 32 * 1024,
            },
            CacheEntry {
                level: 2,
                data: false,
                bytes: 512 * 1024,
            },
        ];
        assert_eq!(cache_bytes(&entries, 1), Some(32 * 1024));
        assert_eq!(cache_bytes(&entries, 2), Some(512 * 1024));
        assert_eq!(cache_bytes(&entries, 3), None);
    }

    #[test]
    fn cache_lookup_falls_back_to_unified_level_one() {
        let entries = [CacheEntry {
            level: 1,
            data: false,
            bytes: 64 * 1024,
        }];
        assert_eq!(cache_bytes(&entries, 1), Some(64 * 1024));
    }
}
