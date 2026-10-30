// SPDX-License-Identifier: Apache-2.0
//! Base error type shared by every rustmpeg-next crate.
//!
//! The core crate is `no_std`, so errors carry owned strings instead of
//! `std::io::Error` values. Codec crates wrap [`CoreError`] in their own
//! typed errors with `#[from]`.

use alloc::string::String;

/// Result alias used by the core primitives.
pub type Result<T> = core::result::Result<T, CoreError>;

/// A recoverable error in a shared primitive.
///
/// Variants are deliberately small and owned: they can cross crate boundaries
/// and survive in async state machines without borrowing the input.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CoreError {
    /// A configuration value is outside its supported range.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// Input bytes or metadata are malformed.
    #[error("invalid data: {0}")]
    InvalidData(String),
    /// The operation is valid but not implemented for this format, layout, or
    /// bit depth.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A resource request exceeds a documented limit.
    #[error("limit exceeded for {what}: {value} > {limit}")]
    LimitExceeded {
        /// Name of the limited resource.
        what: &'static str,
        /// Requested value.
        value: u64,
        /// Maximum accepted value.
        limit: u64,
    },
    /// Checked arithmetic overflowed.
    #[error("arithmetic overflow in {0}")]
    Overflow(&'static str),
    /// A caller-supplied buffer is shorter than the operation requires.
    #[error("buffer too small: need {needed} bytes, got {got}")]
    BufferTooSmall {
        /// Required byte count.
        needed: usize,
        /// Available byte count.
        got: usize,
    },
    /// An index or range does not address the requested plane or row.
    #[error("invalid index: {0}")]
    InvalidIndex(String),
}

impl CoreError {
    /// Builds a [`CoreError::LimitExceeded`] value.
    pub fn limit(what: &'static str, value: u64, limit: u64) -> Self {
        Self::LimitExceeded { what, value, limit }
    }

    /// Builds a [`CoreError::InvalidIndex`] value from displayable parts.
    pub fn index(index: usize, what: &'static str) -> Self {
        use alloc::format;
        Self::InvalidIndex(format!("{what} index {index} is out of range"))
    }
}
