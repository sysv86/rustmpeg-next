// SPDX-License-Identifier: Apache-2.0
//! Vulkan Compute backend for rustmpeg-next.
//!
//! This crate is the only GPU layer in the workspace. It uses `wgpu` as a safe
//! Vulkan binding and compiles WGSL compute shaders at run time with `naga`,
//! so the build never requires a C/C++ shader compiler or a native SDK.
//!
//! Vulkan Video is deliberately absent: its codec operations cover only AV1,
//! H.264/AVC, H.265/HEVC, and VP9, none of which are in the supported matrix.
//! AV2, H.266/VVC, and APV are accelerated through block-parallel compute
//! kernels instead.
//!
//! # Layers
//!
//! * [`compute`] owns the device, buffers, kernel compilation, dispatch
//!   validation, and synchronous read-back;
//! * [`compute::color`] implements bit-exact RGBA/I420 conversion;
//! * [`compute::metrics`] implements plane SSE/SAD/PSNR metrics;
//! * [`scheduler`] chooses between the CPU and the GPU per frame-level stage.
//!
//! A machine without a Vulkan driver still links and runs: every constructor
//! returns a typed [`VulkanError`] instead of panicking, and the codecs keep
//! their CPU path.

#![forbid(unsafe_code)]
#![recursion_limit = "1024"]

pub mod accelerator;
pub mod compute;
pub mod error;
pub mod scheduler;

pub use accelerator::Accelerator;
pub use compute::color::{
    I420Converter, RgbaConverter, i420_to_rgba, i420_to_rgba_cpu, rgba_to_i420,
};
pub use compute::deblock::DeblockKernel;
pub use compute::metrics::PlaneMetricsKernel;
pub use compute::{
    Compute, ComputeConfig, ComputeInfo, DEFAULT_TIMEOUT, GpuBuffer, Kernel, PowerPreference,
};
pub use error::VulkanError;
pub use scheduler::{
    CpuProfile, FrameSize, GpuProfile, Placement, Scheduler, Stage, StageCost, StageCosts,
};
