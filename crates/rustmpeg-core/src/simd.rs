// SPDX-License-Identifier: Apache-2.0
//! Portable SIMD primitives used by distortion and quality kernels.
//!
//! The routines use `core::simd`, so they compile to LLVM vector operations on
//! x86_64, aarch64, and wasm32 without target-specific intrinsics or inline
//! assembly. Scalar tails keep every length correct.

use core::simd::Simd;
use core::simd::prelude::{SimdOrd, SimdUint};

/// Preferred vector width for byte operations.
const LANES: usize = 16;

/// Sum of absolute differences between two 8-bit slices.
///
/// The comparison stops at the shorter slice; callers validate equal plane
/// dimensions before calling.
pub fn sad_u8(left: &[u8], right: &[u8]) -> u64 {
    let len = left.len().min(right.len());
    let mut sum = 0_u64;
    let mut index = 0;

    while index + LANES <= len {
        let a = Simd::<u8, LANES>::from_slice(&left[index..index + LANES]);
        let b = Simd::<u8, LANES>::from_slice(&right[index..index + LANES]);
        let difference = a.simd_max(b) - a.simd_min(b);
        sum += u64::from(difference.cast::<u32>().reduce_sum());
        index += LANES;
    }

    while index < len {
        sum += u64::from(left[index].abs_diff(right[index]));
        index += 1;
    }
    sum
}

/// Sum of squared differences between two 8-bit slices.
///
/// The comparison stops at the shorter slice; callers validate equal plane
/// dimensions before calling.
pub fn sse_u8(left: &[u8], right: &[u8]) -> u64 {
    let len = left.len().min(right.len());
    let mut sum = 0_u64;
    let mut index = 0;

    while index + LANES <= len {
        let a = Simd::<u8, LANES>::from_slice(&left[index..index + LANES]);
        let b = Simd::<u8, LANES>::from_slice(&right[index..index + LANES]);
        let difference = (a.simd_max(b) - a.simd_min(b)).cast::<u32>();
        sum += u64::from((difference * difference).reduce_sum());
        index += LANES;
    }

    while index < len {
        let difference = u64::from(left[index].abs_diff(right[index]));
        sum += difference * difference;
        index += 1;
    }
    sum
}

/// Clamps a signed intermediate to an 8-bit sample.
pub const fn clamp_u8(value: i16) -> u8 {
    if value < 0 {
        0
    } else if value > 255 {
        255
    } else {
        value as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn scalar_sad(left: &[u8], right: &[u8]) -> u64 {
        left.iter()
            .zip(right)
            .map(|(a, b)| u64::from(a.abs_diff(*b)))
            .sum()
    }

    fn scalar_sse(left: &[u8], right: &[u8]) -> u64 {
        left.iter()
            .zip(right)
            .map(|(a, b)| {
                let difference = u64::from(a.abs_diff(*b));
                difference * difference
            })
            .sum()
    }

    #[test]
    fn simd_matches_scalar_for_every_tail_length() {
        for len in [0_usize, 1, 3, 15, 16, 17, 31, 32, 33, 257] {
            let left: Vec<u8> = (0..len)
                .map(|index| u8::try_from((index * 37 + 11) % 256).unwrap_or(0))
                .collect();
            let right: Vec<u8> = (0..len)
                .map(|index| u8::try_from((index * 19 + 3) % 256).unwrap_or(0))
                .collect();
            assert_eq!(
                sad_u8(&left, &right),
                scalar_sad(&left, &right),
                "SAD {len}"
            );
            assert_eq!(
                sse_u8(&left, &right),
                scalar_sse(&left, &right),
                "SSE {len}"
            );
        }
    }

    #[test]
    fn shorter_inputs_are_compared_up_to_their_length() {
        assert_eq!(sad_u8(&[1, 2, 3], &[1]), 0);
        assert_eq!(sse_u8(&[], &[1, 2]), 0);
    }

    #[test]
    fn clamp_maps_the_edges() {
        assert_eq!(clamp_u8(-1), 0);
        assert_eq!(clamp_u8(0), 0);
        assert_eq!(clamp_u8(255), 255);
        assert_eq!(clamp_u8(256), 255);
    }
}
