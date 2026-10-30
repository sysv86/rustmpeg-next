// SPDX-License-Identifier: Apache-2.0
//! Vulkan Compute runtime built on `wgpu`.
//!
//! The runtime owns one logical Vulkan device, compiles WGSL kernels at run
//! time, validates them with `naga`, allocates storage and uniform buffers,
//! dispatches compute passes, and reads results back to host memory.
//!
//! # Contracts
//!
//! * Bindings in a kernel must use group `0` and the contiguous binding
//!   numbers `0..n`. Callers pass buffers to [`Kernel::dispatch`] in ascending
//!   binding order.
//! * Every buffer is bounds-checked against the device limits before it is
//!   created or bound. A rejected request returns a typed [`VulkanError`];
//!   the runtime never panics on hostile dimensions or sizes.
//! * Read-back uses a bounded timeout so a wedged driver cannot hang the
//!   calling thread forever.

use std::future::Future;
use std::sync::Arc;
use std::sync::mpsc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use naga::valid::{Capabilities, ValidationFlags, Validator};
use rustmpeg_video::accelerator::AcceleratorError;

use crate::error::VulkanError;

pub mod color;
pub mod deblock;
pub mod metrics;

pub use color::{I420Converter, RgbaConverter};
pub use deblock::DeblockKernel;
pub use metrics::PlaneMetricsKernel;

/// Default upper bound for a queue submission and its read-back.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Preferred class of adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerPreference {
    /// Prefer an integrated or low-power adapter.
    LowPower,
    /// Prefer a discrete adapter.
    HighPerformance,
}

/// Device selection and execution policy.
#[derive(Debug, Clone)]
pub struct ComputeConfig {
    /// Adapter class to prefer when several Vulkan devices are present.
    pub power_preference: PowerPreference,
    /// Optional exact adapter name. When set, no other adapter is considered.
    pub adapter_name: Option<String>,
    /// Maximum time to wait for a submission or a read-back mapping.
    pub timeout: Duration,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            power_preference: PowerPreference::HighPerformance,
            adapter_name: None,
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

/// Static information about the selected compute device.
#[derive(Debug, Clone)]
pub struct ComputeInfo {
    /// Adapter name.
    pub name: String,
    /// PCI vendor identifier.
    pub vendor_id: u32,
    /// PCI device identifier.
    pub device_id: u32,
    /// Backend name, always `Vulkan` for this crate.
    pub backend: String,
    /// Device type reported by the driver.
    pub device_type: String,
    /// `true` for integrated or virtual devices.
    pub integrated: bool,
    /// Driver name.
    pub driver: String,
    /// Driver version and build information.
    pub driver_info: String,
    /// Maximum size of a single buffer allocation.
    pub max_buffer_size: u64,
    /// Maximum size of one storage-buffer binding.
    pub max_storage_buffer_binding_size: u64,
    /// Maximum size of one uniform-buffer binding.
    pub max_uniform_buffer_binding_size: u64,
    /// Maximum workgroup size per dimension.
    pub max_workgroup_size: [u32; 3],
    /// Maximum invocations in one workgroup.
    pub max_invocations_per_workgroup: u32,
    /// Maximum dispatch grid per dimension.
    pub max_workgroups_per_dimension: u32,
    /// Maximum storage-buffer bindings visible to a shader stage.
    pub max_storage_buffers_per_shader_stage: u32,
    /// Maximum uniform-buffer bindings visible to a shader stage.
    pub max_uniform_buffers_per_shader_stage: u32,
    /// Required alignment of a storage-buffer binding offset.
    pub min_storage_buffer_offset_alignment: u32,
    /// Required alignment of a uniform-buffer binding offset.
    pub min_uniform_buffer_offset_alignment: u32,
}

/// A Vulkan compute device created through `wgpu`.
///
/// Handles are cheap to clone: they refer to the same device and queue. Kernels
/// own one handle, which lets a codec keep an accelerator alive independently of
/// the device object that created it.
#[derive(Debug, Clone)]
pub struct Compute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: Arc<ComputeInfo>,
    timeout: Duration,
}

impl Compute {
    /// Creates a headless Vulkan compute device with the default policy.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Unavailable`] when no Vulkan adapter is present
    /// or the device cannot be created.
    pub fn new() -> Result<Self, VulkanError> {
        Self::with_config(&ComputeConfig::default())
    }

    /// Creates a headless Vulkan compute device with an explicit policy.
    ///
    /// Only the Vulkan backend is requested: the project has a single GPU
    /// stack and does not fall back to OpenGL, Metal, or D3D.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Unavailable`] when no matching Vulkan adapter is
    /// present or the device cannot be created.
    pub fn with_config(config: &ComputeConfig) -> Result<Self, VulkanError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);

        let mut adapters = block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        if adapters.is_empty() {
            return Err(VulkanError::Unavailable(
                "no Vulkan adapter was enumerated".to_owned(),
            ));
        }

        if let Some(name) = &config.adapter_name {
            adapters.retain(|adapter| adapter.get_info().name == *name);
            if adapters.is_empty() {
                return Err(VulkanError::Unavailable(format!(
                    "no Vulkan adapter named `{name}`"
                )));
            }
        }

        let adapter = adapters
            .into_iter()
            .max_by_key(|adapter| adapter_score(adapter, config.power_preference))
            .ok_or_else(|| VulkanError::Unavailable("no Vulkan adapter was selected".to_owned()))?;

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();

        // Raise only the limits that depend on the physical device. The
        // remaining limits stay at the portable defaults, which is enough for
        // every kernel in this crate and avoids requesting optional features.
        let required_limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_uniform_buffer_binding_size: adapter_limits.max_uniform_buffer_binding_size,
            max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: adapter_limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: adapter_limits.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: adapter_limits
                .max_compute_invocations_per_workgroup,
            max_compute_workgroups_per_dimension: adapter_limits
                .max_compute_workgroups_per_dimension,
            ..wgpu::Limits::default()
        };

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rustmpeg compute"),
            required_features: wgpu::Features::empty(),
            required_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|error| VulkanError::Unavailable(error.to_string()))?;

        let limits = device.limits();
        let info = ComputeInfo {
            name: adapter_info.name,
            vendor_id: adapter_info.vendor,
            device_id: adapter_info.device,
            backend: format!("{:?}", adapter_info.backend),
            device_type: format!("{:?}", adapter_info.device_type),
            integrated: matches!(
                adapter_info.device_type,
                wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu
            ),
            driver: adapter_info.driver,
            driver_info: adapter_info.driver_info,
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size,
            max_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            max_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            max_uniform_buffers_per_shader_stage: limits.max_uniform_buffers_per_shader_stage,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
            min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
        };

        tracing::debug!(
            adapter = %info.name,
            backend = %info.backend,
            driver = %info.driver,
            "Vulkan compute device created"
        );

        Ok(Self {
            device,
            queue,
            info: Arc::new(info),
            timeout: config.timeout,
        })
    }

    /// Information about the selected adapter.
    pub fn info(&self) -> &ComputeInfo {
        self.info.as_ref()
    }

    /// Maximum time a submission or read-back may take.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Uploads bytes into a storage buffer.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::BufferTooLarge`] when the allocation exceeds the
    /// device limit.
    pub fn upload(&self, data: &[u8]) -> Result<GpuBuffer, VulkanError> {
        self.upload_with(
            data,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            "rustmpeg storage upload",
            4,
        )
    }

    /// Uploads bytes into a uniform buffer.
    ///
    /// The allocation is padded to the 16-byte uniform binding alignment.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::BufferTooLarge`] when the allocation exceeds the
    /// device limit.
    pub fn uniform(&self, data: &[u8]) -> Result<GpuBuffer, VulkanError> {
        self.upload_with(
            data,
            wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            "rustmpeg uniform",
            16,
        )
    }

    /// Allocates a zeroed storage buffer of at least `len` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::BufferTooLarge`] when the allocation exceeds the
    /// device limit.
    pub fn allocate(&self, len: usize) -> Result<GpuBuffer, VulkanError> {
        self.check_requested(len)?;
        let size = align_up(len, 4)?;
        self.check_allocation(size)?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustmpeg storage"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ok(GpuBuffer { buffer, size })
    }

    /// Compiles and validates a WGSL compute kernel.
    ///
    /// The shader is parsed and validated by `naga` before it reaches the
    /// driver, so syntax errors and resource-limit violations surface as
    /// [`VulkanError::Kernel`] instead of a device loss or a panic.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Kernel`] when the entry point is missing, the
    /// shader is invalid, or the bindings violate the documented contract.
    pub fn kernel(&self, wgsl: &str, entry: &str) -> Result<Kernel, VulkanError> {
        let module = naga::front::wgsl::parse_str(wgsl).map_err(|error| VulkanError::Kernel {
            entry: entry.to_owned(),
            message: error.emit_to_string(wgsl),
        })?;

        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        validator
            .validate(&module)
            .map_err(|error| VulkanError::Kernel {
                entry: entry.to_owned(),
                message: format!("{error}"),
            })?;

        let entry_point = module
            .entry_points
            .iter()
            .find(|candidate| candidate.name == entry)
            .ok_or_else(|| VulkanError::Kernel {
                entry: entry.to_owned(),
                message: "entry point is not present in the module".to_owned(),
            })?;

        if entry_point.stage != naga::ShaderStage::Compute {
            return Err(VulkanError::Kernel {
                entry: entry.to_owned(),
                message: "entry point is not a compute shader".to_owned(),
            });
        }

        let workgroup_size = entry_point.workgroup_size;
        self.check_workgroup_size(entry, workgroup_size)?;

        let bindings = collect_bindings(entry_point, &module)?;

        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        if let Some(error) = block_on(scope.pop()) {
            return Err(VulkanError::Kernel {
                entry: entry.to_owned(),
                message: error.to_string(),
            });
        }

        Ok(Kernel {
            pipeline,
            bindings,
            workgroup_size,
            entry: entry.to_owned(),
        })
    }

    /// Reads `len` bytes from a GPU buffer.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::BufferTooLarge`] when `len` does not fit the
    /// buffer or the staging allocation, [`VulkanError::Timeout`] when the
    /// queue or the mapping does not complete in time, and
    /// [`VulkanError::Readback`] when the driver refuses the mapping.
    pub fn read(&self, buffer: &GpuBuffer, len: usize) -> Result<Vec<u8>, VulkanError> {
        let (staging, size) = self.prepare_readback(buffer, len)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustmpeg readback"),
            });
        encoder.copy_buffer_to_buffer(&buffer.buffer, 0, &staging, 0, size);
        let submission = self.queue.submit(Some(encoder.finish()));
        self.poll(Some(submission.clone()))?;

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.poll(Some(submission))?;
        receiver
            .recv_timeout(self.timeout)
            .map_err(|_| VulkanError::Timeout(self.timeout))?
            .map_err(|error| VulkanError::readback(error.to_string()))?;

        let view = slice
            .get_mapped_range()
            .map_err(|error| VulkanError::readback(error.to_string()))?;
        let bytes = view[..len].to_vec();
        drop(view);
        staging.unmap();
        Ok(bytes)
    }

    /// Starts a recorded batch of dispatches and read-backs.
    ///
    /// A batch submits once and polls once, so several kernels can run on the
    /// device before the host waits for any result. Record dispatches before
    /// the read-backs they produce.
    pub fn batch(&self) -> Batch<'_> {
        Batch {
            compute: self,
            encoder: self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rustmpeg batch"),
                }),
            reads: Vec::new(),
        }
    }

    fn prepare_readback(
        &self,
        buffer: &GpuBuffer,
        len: usize,
    ) -> Result<(wgpu::Buffer, u64), VulkanError> {
        if len > usize::try_from(buffer.size).unwrap_or(usize::MAX) {
            return Err(VulkanError::Execution(format!(
                "read of {len} bytes exceeds the {} byte buffer",
                buffer.size
            )));
        }
        self.check_requested(len)?;
        let size = align_up(len, 4)?;
        self.check_allocation(size)?;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustmpeg readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok((staging, size))
    }

    /// Waits until every submitted command has completed.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Timeout`] when the device does not make progress
    /// within the configured timeout.
    pub fn wait(&self) -> Result<(), VulkanError> {
        self.poll(None)
    }

    fn poll(&self, submission: Option<wgpu::SubmissionIndex>) -> Result<(), VulkanError> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: submission,
                timeout: Some(self.timeout),
            })
            .map(|_| ())
            .map_err(|error| match error {
                wgpu::PollError::Timeout => VulkanError::Timeout(self.timeout),
                other => VulkanError::execution(other.to_string()),
            })
    }

    fn upload_with(
        &self,
        data: &[u8],
        usage: wgpu::BufferUsages,
        label: &str,
        alignment: usize,
    ) -> Result<GpuBuffer, VulkanError> {
        self.check_requested(data.len())?;
        let size = align_up(data.len(), alignment)?;
        self.check_allocation(size)?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        if !data.is_empty() {
            let padded;
            let bytes = if data.len() % 4 == 0 {
                data
            } else {
                let padded_len = usize::try_from(size).map_err(|_| {
                    VulkanError::Execution("buffer size does not fit in usize".to_owned())
                })?;
                padded = {
                    let mut bytes = data.to_vec();
                    bytes.resize(padded_len, 0);
                    bytes
                };
                &padded
            };
            self.queue.write_buffer(&buffer, 0, bytes);
        }
        Ok(GpuBuffer { buffer, size })
    }

    fn check_requested(&self, len: usize) -> Result<(), VulkanError> {
        let requested = u64::try_from(len).unwrap_or(u64::MAX);
        if requested > self.info.max_buffer_size {
            return Err(VulkanError::BufferTooLarge {
                requested,
                limit: self.info.max_buffer_size,
            });
        }
        Ok(())
    }

    fn check_allocation(&self, size: u64) -> Result<(), VulkanError> {
        if size > self.info.max_buffer_size {
            return Err(VulkanError::BufferTooLarge {
                requested: size,
                limit: self.info.max_buffer_size,
            });
        }
        Ok(())
    }

    fn check_workgroup_size(&self, entry: &str, size: [u32; 3]) -> Result<(), VulkanError> {
        let limit = self.info.max_workgroup_size;
        let product = u64::from(size[0]) * u64::from(size[1]) * u64::from(size[2]);
        if size.contains(&0)
            || size[0] > limit[0]
            || size[1] > limit[1]
            || size[2] > limit[2]
            || product > u64::from(self.info.max_invocations_per_workgroup)
        {
            return Err(VulkanError::Kernel {
                entry: entry.to_owned(),
                message: format!(
                    "workgroup size {size:?} exceeds the device limits {limit:?} / {} invocations",
                    self.info.max_invocations_per_workgroup
                ),
            });
        }
        Ok(())
    }
}

/// A GPU buffer owned by a [`Compute`] device.
#[derive(Debug)]
pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

impl GpuBuffer {
    /// Buffer size in bytes, rounded up to the GPU alignment.
    pub fn size(&self) -> u64 {
        self.size
    }

    pub(crate) fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// Kind of resource bound to one kernel slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindingKind {
    Storage,
    Uniform,
}

/// A compiled compute kernel.
#[derive(Debug)]
pub struct Kernel {
    pipeline: wgpu::ComputePipeline,
    bindings: Vec<BindingKind>,
    workgroup_size: [u32; 3],
    entry: String,
}

impl Kernel {
    /// Entry point name.
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// Workgroup size declared by the shader.
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    /// Number of buffer bindings declared by the shader.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Dispatches the kernel over `workgroups` workgroups.
    ///
    /// Buffers are bound in ascending binding order, starting at `@binding(0)`.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::BindingCount`] when the number of buffers does
    /// not match the kernel signature, [`VulkanError::BufferTooLarge`] when a
    /// bound buffer exceeds the binding limit, or
    /// [`VulkanError::DispatchTooLarge`] when the grid exceeds the device
    /// limit.
    pub fn dispatch(
        &self,
        compute: &Compute,
        buffers: &[&GpuBuffer],
        workgroups: [u32; 3],
    ) -> Result<(), VulkanError> {
        if self.validate_dispatch(compute, buffers, workgroups)? {
            return Ok(());
        }
        let mut encoder = compute
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&self.entry),
            });
        self.record_dispatch(&mut encoder, compute, buffers, workgroups);
        compute.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Validates bindings and the grid. Returns `true` for a zero-sized grid,
    /// which is a valid no-op and needs no command recording.
    fn validate_dispatch(
        &self,
        compute: &Compute,
        buffers: &[&GpuBuffer],
        workgroups: [u32; 3],
    ) -> Result<bool, VulkanError> {
        if buffers.len() != self.bindings.len() {
            return Err(VulkanError::BindingCount {
                expected: self.bindings.len(),
                actual: buffers.len(),
            });
        }

        let limit = compute.info.max_workgroups_per_dimension;
        if workgroups.iter().any(|value| *value > limit) {
            return Err(VulkanError::DispatchTooLarge {
                requested: workgroups,
                limit,
            });
        }

        for (index, buffer) in buffers.iter().enumerate() {
            let (limit, required_usage) = match self.bindings[index] {
                BindingKind::Storage => (
                    compute.info.max_storage_buffer_binding_size,
                    wgpu::BufferUsages::STORAGE,
                ),
                BindingKind::Uniform => (
                    compute.info.max_uniform_buffer_binding_size,
                    wgpu::BufferUsages::UNIFORM,
                ),
            };
            if buffer.size > limit {
                return Err(VulkanError::BufferTooLarge {
                    requested: buffer.size,
                    limit,
                });
            }
            if !buffer.raw().usage().contains(required_usage) {
                return Err(VulkanError::InvalidConfig(format!(
                    "binding {index} requires {required_usage:?} usage"
                )));
            }
        }

        Ok(workgroups.contains(&0))
    }

    fn record_dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        compute: &Compute,
        buffers: &[&GpuBuffer],
        workgroups: [u32; 3],
    ) {
        let layout = self.pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.raw().as_entire_binding(),
            })
            .collect();
        let bind_group = compute
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&self.entry),
                layout: &layout,
                entries: &entries,
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&self.entry),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }
}

/// A recorded batch of dispatches and read-backs submitted together.
///
/// The batch is the pipeline primitive: several kernels run before the host
/// waits for any result, and one submission covers all of them. This is what
/// keeps intermediate planes on the device instead of paying a queue round trip
/// per stage.
#[derive(Debug)]
pub struct Batch<'a> {
    compute: &'a Compute,
    encoder: wgpu::CommandEncoder,
    reads: Vec<PendingRead>,
}

#[derive(Debug)]
struct PendingRead {
    staging: wgpu::Buffer,
    len: usize,
}

impl Batch<'_> {
    /// Records a dispatch into the batch.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Kernel::dispatch`].
    pub fn dispatch(
        &mut self,
        kernel: &Kernel,
        buffers: &[&GpuBuffer],
        workgroups: [u32; 3],
    ) -> Result<(), VulkanError> {
        if kernel.validate_dispatch(self.compute, buffers, workgroups)? {
            return Ok(());
        }
        kernel.record_dispatch(&mut self.encoder, self.compute, buffers, workgroups);
        Ok(())
    }

    /// Records a read-back and returns its index in the [`Batch::submit`]
    /// result vector.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::BufferTooLarge`] when `len` does not fit the
    /// buffer or the staging allocation.
    pub fn read(&mut self, buffer: &GpuBuffer, len: usize) -> Result<usize, VulkanError> {
        let (staging, size) = self.compute.prepare_readback(buffer, len)?;
        self.encoder
            .copy_buffer_to_buffer(&buffer.buffer, 0, &staging, 0, size);
        let index = self.reads.len();
        self.reads.push(PendingRead { staging, len });
        Ok(index)
    }

    /// Number of read-backs recorded so far.
    pub fn read_count(&self) -> usize {
        self.reads.len()
    }

    /// Submits the whole batch and returns the read-back buffers in the order
    /// their indices were assigned by [`Batch::read`].
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::Timeout`] when the device does not finish, and
    /// [`VulkanError::Readback`] when a staging buffer cannot be mapped.
    pub fn submit(self) -> Result<Vec<Vec<u8>>, VulkanError> {
        let Self {
            compute,
            encoder,
            reads,
        } = self;
        let submission = compute.queue.submit(Some(encoder.finish()));
        compute.poll(Some(submission.clone()))?;

        let mut outputs = Vec::with_capacity(reads.len());
        for pending in &reads {
            let slice = pending.staging.slice(..);
            let (sender, receiver) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            compute.poll(Some(submission.clone()))?;
            receiver
                .recv_timeout(compute.timeout)
                .map_err(|_| VulkanError::Timeout(compute.timeout))?
                .map_err(|error| VulkanError::readback(error.to_string()))?;

            let view = slice
                .get_mapped_range()
                .map_err(|error| VulkanError::readback(error.to_string()))?;
            outputs.push(view[..pending.len].to_vec());
            drop(view);
            pending.staging.unmap();
        }
        Ok(outputs)
    }
}

/// Maps a backend error to the codec-facing accelerator error.
pub(crate) fn accelerator_error(error: VulkanError) -> AcceleratorError {
    let message = error.to_string();
    match error {
        VulkanError::Unavailable(detail) => AcceleratorError::Unavailable(detail),
        VulkanError::InvalidConfig(detail) => AcceleratorError::InvalidInput(detail),
        VulkanError::BufferTooLarge { .. } | VulkanError::DispatchTooLarge { .. } => {
            AcceleratorError::LimitExceeded(message)
        }
        VulkanError::Kernel { .. }
        | VulkanError::BindingCount { .. }
        | VulkanError::Timeout(_)
        | VulkanError::Execution(_)
        | VulkanError::Readback(_) => AcceleratorError::Execution(message),
    }
}

fn collect_bindings(
    entry: &naga::EntryPoint,
    module: &naga::Module,
) -> Result<Vec<BindingKind>, VulkanError> {
    let mut bindings: Vec<(u32, BindingKind)> = Vec::new();
    for (_, variable) in module.global_variables.iter() {
        let Some(binding) = variable.binding else {
            continue;
        };
        if binding.group != 0 {
            return Err(VulkanError::Kernel {
                entry: entry.name.clone(),
                message: format!(
                    "binding {} uses group {}, only group 0 is supported",
                    binding.binding, binding.group
                ),
            });
        }
        let kind = match variable.space {
            naga::AddressSpace::Storage { .. } => BindingKind::Storage,
            naga::AddressSpace::Uniform => BindingKind::Uniform,
            _ => continue,
        };
        bindings.push((binding.binding, kind));
    }
    bindings.sort_by_key(|(binding, _)| *binding);

    for (index, (binding, _)) in bindings.iter().enumerate() {
        if *binding != index as u32 {
            return Err(VulkanError::Kernel {
                entry: entry.name.clone(),
                message: format!(
                    "bindings must be contiguous from 0; found {binding} at position {index}"
                ),
            });
        }
    }
    Ok(bindings.into_iter().map(|(_, kind)| kind).collect())
}

fn adapter_score(adapter: &wgpu::Adapter, preference: PowerPreference) -> u8 {
    use wgpu::DeviceType;
    match (preference, adapter.get_info().device_type) {
        (PowerPreference::HighPerformance, DeviceType::DiscreteGpu) => 5,
        (PowerPreference::HighPerformance, DeviceType::IntegratedGpu) => 4,
        (PowerPreference::HighPerformance, DeviceType::VirtualGpu) => 3,
        (PowerPreference::HighPerformance, DeviceType::Cpu) => 1,
        (PowerPreference::HighPerformance, DeviceType::Other) => 2,
        (PowerPreference::LowPower, DeviceType::IntegratedGpu) => 5,
        (PowerPreference::LowPower, DeviceType::DiscreteGpu) => 4,
        (PowerPreference::LowPower, DeviceType::VirtualGpu) => 3,
        (PowerPreference::LowPower, DeviceType::Cpu) => 1,
        (PowerPreference::LowPower, DeviceType::Other) => 2,
    }
}

fn align_up(len: usize, alignment: usize) -> Result<u64, VulkanError> {
    debug_assert!(alignment.is_power_of_two());
    let len = len.max(1);
    let mask = alignment - 1;
    let aligned = len
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or_else(|| VulkanError::Execution("buffer size overflow".to_owned()))?;
    u64::try_from(aligned)
        .map_err(|_| VulkanError::Execution("buffer size does not fit in u64".to_owned()))
}

fn block_on<F: Future>(future: F) -> F::Output {
    struct ThreadWake(std::thread::Thread);

    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_sizes_are_word_aligned() {
        assert_eq!(align_up(0, 4).unwrap(), 4);
        assert_eq!(align_up(1, 4).unwrap(), 4);
        assert_eq!(align_up(4, 4).unwrap(), 4);
        assert_eq!(align_up(5, 4).unwrap(), 8);
        assert_eq!(align_up(17, 16).unwrap(), 32);
    }

    #[test]
    fn alignment_rejects_overflow() {
        assert!(align_up(usize::MAX, 16).is_err());
    }

    #[test]
    fn compute_reports_a_vulkan_adapter_or_fails_cleanly() {
        match Compute::new() {
            Ok(compute) => {
                assert!(compute.info().max_buffer_size > 0);
                assert_eq!(compute.info().backend, "Vulkan");
                assert!(compute.info().max_workgroup_size[0] >= 1);
            }
            Err(error) => {
                // A machine without a Vulkan driver must produce a typed error,
                // never a panic.
                assert!(matches!(error, VulkanError::Unavailable(_)));
            }
        }
    }

    #[test]
    fn invalid_kernel_is_reported_without_touching_the_driver() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let error = compute
            .kernel(
                "@compute @workgroup_size(1) fn main() { let x: u32 = ; }",
                "main",
            )
            .expect_err("syntax error must be rejected");
        assert!(matches!(error, VulkanError::Kernel { .. }));

        let error = compute
            .kernel("@compute @workgroup_size(1) fn other() {}", "missing")
            .expect_err("missing entry point must be rejected");
        assert!(matches!(error, VulkanError::Kernel { .. }));
    }

    #[test]
    fn oversized_allocations_are_rejected_without_panicking() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let error = compute
            .allocate(usize::MAX)
            .expect_err("an unbounded allocation must fail");
        assert!(matches!(error, VulkanError::BufferTooLarge { .. }));
    }

    #[test]
    fn dispatch_limits_are_enforced() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let kernel = compute
            .kernel(
                r#"
                @group(0) @binding(0) var<storage, read_write> data: array<u32>;
                @compute @workgroup_size(1) fn main() {
                    data[0] = 1u;
                }
                "#,
                "main",
            )
            .expect("kernel compiles");
        let buffer = compute.allocate(4).expect("buffer");
        let limit = compute.info().max_workgroups_per_dimension;
        let error = kernel
            .dispatch(&compute, &[&buffer], [limit.saturating_add(1), 1, 1])
            .expect_err("an oversized grid must fail");
        assert!(matches!(error, VulkanError::DispatchTooLarge { .. }));

        // A zero-sized grid is a no-op, not an error.
        kernel
            .dispatch(&compute, &[&buffer], [0, 1, 1])
            .expect("empty dispatch");
    }

    #[test]
    fn dispatch_validates_the_binding_signature() {
        let Ok(compute) = Compute::new() else {
            return;
        };
        let kernel = compute
            .kernel(
                r#"
                @group(0) @binding(0) var<storage, read_write> data: array<u32>;
                @compute @workgroup_size(1) fn main() {
                    data[0] = data[0] + 1u;
                }
                "#,
                "main",
            )
            .expect("kernel compiles");
        assert_eq!(kernel.binding_count(), 1);
        assert_eq!(kernel.workgroup_size(), [1, 1, 1]);

        let error = kernel
            .dispatch(&compute, &[], [1, 1, 1])
            .expect_err("missing binding must be rejected");
        assert!(matches!(
            error,
            VulkanError::BindingCount {
                expected: 1,
                actual: 0
            }
        ));
    }
}
