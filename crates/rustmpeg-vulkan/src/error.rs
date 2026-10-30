// SPDX-License-Identifier: Apache-2.0
//! Errors returned by the Vulkan Compute backend.

use std::time::Duration;

/// Errors produced by the Vulkan Compute backend.
///
/// Every variant is recoverable: the caller can fall back to the CPU path
/// without losing input data. No library function panics on a driver or input
/// error.
#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    /// No Vulkan compute adapter is available, or the device could not be
    /// created.
    #[error("Vulkan compute is unavailable: {0}")]
    Unavailable(String),

    /// The requested adapter selection or compute configuration is invalid.
    #[error("invalid compute configuration: {0}")]
    InvalidConfig(String),

    /// A WGSL kernel failed to parse or validate.
    #[error("compute kernel `{entry}` is invalid: {message}")]
    Kernel {
        /// Entry point named by the caller.
        entry: String,
        /// Parser or validator diagnostic.
        message: String,
    },

    /// A buffer would exceed the device allocation or binding limit.
    #[error("buffer of {requested} bytes exceeds the device limit of {limit} bytes")]
    BufferTooLarge {
        /// Requested size in bytes.
        requested: u64,
        /// Device limit in bytes.
        limit: u64,
    },

    /// The number of bound buffers does not match the kernel signature.
    #[error("kernel expects {expected} buffer bindings, got {actual}")]
    BindingCount {
        /// Number of buffers declared by the kernel.
        expected: usize,
        /// Number of buffers passed to the dispatch.
        actual: usize,
    },

    /// A dispatch grid exceeds the device limit.
    #[error("dispatch of {requested:?} workgroups exceeds the limit of {limit} per dimension")]
    DispatchTooLarge {
        /// Requested workgroup count.
        requested: [u32; 3],
        /// Device limit per dimension.
        limit: u32,
    },

    /// A GPU submission or mapping operation did not complete in time.
    #[error("GPU operation timed out after {0:?}")]
    Timeout(Duration),

    /// The device rejected or aborted the command.
    #[error("GPU execution failed: {0}")]
    Execution(String),

    /// The result buffer could not be mapped back to host memory.
    #[error("GPU read-back failed: {0}")]
    Readback(String),
}

impl VulkanError {
    /// Creates an execution error from any driver diagnostic.
    pub(crate) fn execution(message: impl Into<String>) -> Self {
        Self::Execution(message.into())
    }

    /// Creates a read-back error from any driver diagnostic.
    pub(crate) fn readback(message: impl Into<String>) -> Self {
        Self::Readback(message.into())
    }
}
