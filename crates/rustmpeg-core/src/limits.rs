// SPDX-License-Identifier: Apache-2.0
//! Checked size arithmetic and shared limits.
//!
//! Every allocation derived from untrusted metadata goes through these helpers.
//! They return typed errors instead of panicking or wrapping.

use crate::error::{CoreError, Result};

/// Largest picture dimension accepted by the codec matrix (16K).
pub const MAX_DIMENSION: u32 = 16_384;

/// Largest number of luma samples in one picture (VVC level 6.2).
pub const MAX_LUMA_SAMPLES: u64 = 35_651_584;

/// Multiplies two sizes with overflow checking.
///
/// # Errors
///
/// Returns [`CoreError::Overflow`] when the product does not fit in `usize`.
pub fn checked_mul(left: usize, right: usize, what: &'static str) -> Result<usize> {
    left.checked_mul(right).ok_or(CoreError::Overflow(what))
}

/// Adds two sizes with overflow checking.
///
/// # Errors
///
/// Returns [`CoreError::Overflow`] when the sum does not fit in `usize`.
pub fn checked_add(left: usize, right: usize, what: &'static str) -> Result<usize> {
    left.checked_add(right).ok_or(CoreError::Overflow(what))
}

/// Byte size of a `width x height` plane with `bytes_per_sample` samples.
///
/// # Errors
///
/// Returns [`CoreError::Overflow`] when the product does not fit in `usize`.
pub fn plane_bytes(width: u32, height: u32, bytes_per_sample: usize) -> Result<usize> {
    let samples = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(CoreError::Overflow("plane sample count"))?;
    let bytes = samples
        .checked_mul(bytes_per_sample as u64)
        .ok_or(CoreError::Overflow("plane byte size"))?;
    usize::try_from(bytes).map_err(|_| CoreError::Overflow("plane byte size"))
}

/// Validates a picture size against [`MAX_DIMENSION`] and
/// [`MAX_LUMA_SAMPLES`].
///
/// # Errors
///
/// Returns [`CoreError::InvalidConfig`] for a zero dimension and
/// [`CoreError::LimitExceeded`] when a limit is exceeded.
pub fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(CoreError::InvalidConfig(
            "picture dimensions must be non-zero".into(),
        ));
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(CoreError::limit(
            "picture dimension",
            u64::from(width.max(height)),
            u64::from(MAX_DIMENSION),
        ));
    }
    let samples = u64::from(width) * u64::from(height);
    if samples > MAX_LUMA_SAMPLES {
        return Err(CoreError::limit("luma samples", samples, MAX_LUMA_SAMPLES));
    }
    Ok(())
}

/// Validates that a row stride can hold `width` samples.
///
/// # Errors
///
/// Returns [`CoreError::InvalidConfig`] when the stride is too small and
/// [`CoreError::Overflow`] when `width * bytes_per_sample` does not fit.
pub fn validate_stride(stride: usize, width: u32, bytes_per_sample: usize) -> Result<()> {
    let minimum = checked_mul(
        usize::try_from(width).map_err(|_| CoreError::Overflow("plane width"))?,
        bytes_per_sample,
        "minimum stride",
    )?;
    if stride < minimum {
        return Err(CoreError::InvalidConfig(
            "plane stride is smaller than one visible row".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_arithmetic_reports_overflow() {
        assert_eq!(checked_mul(6, 7, "test").expect("mul"), 42);
        assert!(checked_mul(usize::MAX, 2, "test").is_err());
        assert!(checked_add(usize::MAX, 1, "test").is_err());
    }

    #[test]
    fn plane_bytes_matches_u64_maths() {
        assert_eq!(plane_bytes(1920, 1080, 1).expect("bytes"), 2_073_600);
        assert_eq!(plane_bytes(1920, 1080, 2).expect("bytes"), 4_147_200);
        assert!(plane_bytes(u32::MAX, u32::MAX, 16).is_err());
    }

    #[test]
    fn dimensions_are_bounded() {
        assert!(validate_dimensions(1920, 1080).is_ok());
        assert!(validate_dimensions(0, 1080).is_err());
        assert!(validate_dimensions(MAX_DIMENSION + 1, 1).is_err());
        assert!(validate_dimensions(8192, 8192).is_err());
    }

    #[test]
    fn stride_must_hold_a_row() {
        assert!(validate_stride(1920, 1920, 1).is_ok());
        assert!(validate_stride(1919, 1920, 1).is_err());
        assert!(validate_stride(3840, 1920, 2).is_ok());
    }
}
