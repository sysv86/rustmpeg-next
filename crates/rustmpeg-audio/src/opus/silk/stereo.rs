// SPDX-License-Identifier: Apache-2.0
//! SILK stereo prediction weights and mid-side reconstruction.
//!
//! Implements RFC 6716, Sections 4.2.7.1, 4.2.7.2, and 4.2.8.

use crate::opus::{OpusError, RangeDecoder};

use super::tables::{
    SILK_STEREO_ONLY_CODE_MID_ICDF, SILK_STEREO_PRED_JOINT_ICDF, SILK_STEREO_PRED_QUANT_Q13,
    SILK_UNIFORM3_ICDF, SILK_UNIFORM5_ICDF,
};

/// Samples of weight interpolation at the start of each frame
/// (RFC 6716, Section 4.2.8: 8 ms).
const INTERPOLATION_MS: usize = 8;

/// Decodes the two Q13 stereo prediction weights.
///
/// # Errors
///
/// Propagates range-decoder failures.
pub(crate) fn decode_weights(decoder: &mut RangeDecoder<'_>) -> Result<[i32; 2], OpusError> {
    let joint = decoder.decode_icdf(&SILK_STEREO_PRED_JOINT_ICDF, 8)? as usize;
    let first_high = decoder.decode_icdf(&SILK_UNIFORM3_ICDF, 8)? as usize;
    let first_low = decoder.decode_icdf(&SILK_UNIFORM5_ICDF, 8)? as i32;
    let second_high = decoder.decode_icdf(&SILK_UNIFORM3_ICDF, 8)? as usize;
    let second_low = decoder.decode_icdf(&SILK_UNIFORM5_ICDF, 8)? as i32;

    let index0 = first_high + 3 * (joint / 5);
    let index1 = second_high + 3 * (joint % 5);

    let weight1 = interpolate_weight(index1, second_low);
    let weight0 = interpolate_weight(index0, first_low) - weight1;
    Ok([weight0, weight1])
}

fn interpolate_weight(index: usize, fraction: i32) -> i32 {
    let base = i32::from(SILK_STEREO_PRED_QUANT_Q13[index]);
    let step = ((i32::from(SILK_STEREO_PRED_QUANT_Q13[index + 1]) - base) * 6554) >> 16;
    base + step * (2 * fraction + 1)
}

/// Decodes the mid-only flag (RFC 6716, Section 4.2.7.2).
///
/// # Errors
///
/// Propagates range-decoder failures.
pub(crate) fn decode_mid_only(decoder: &mut RangeDecoder<'_>) -> Result<bool, OpusError> {
    Ok(decoder.decode_icdf(&SILK_STEREO_ONLY_CODE_MID_ICDF, 8)? != 0)
}

/// Converts mid-side to left-right (RFC 6716, Section 4.2.8).
#[allow(clippy::too_many_arguments)]
///
/// `mid_history` and `side_history` carry the two most recent samples of the
/// previous frame; they are updated to the tail of this frame.
pub(crate) fn mid_side_to_left_right(
    previous_weights: &mut [i32; 2],
    mid_history: &mut [i16; 2],
    side_history: &mut [i16; 2],
    weights: [i32; 2],
    mid: &[i16],
    side: &[i16],
    fs_khz: u8,
    left: &mut [i16],
    right: &mut [i16],
) {
    debug_assert_eq!(mid.len(), side.len());
    debug_assert_eq!(mid.len(), left.len());
    debug_assert_eq!(mid.len(), right.len());

    let interpolation = INTERPOLATION_MS * usize::from(fs_khz);
    let previous = *previous_weights;
    for index in 0..mid.len() {
        let position = index.min(interpolation - 1) as i64;
        let weight0 = i64::from(previous[0])
            + (i64::from(weights[0] - previous[0]) * position) / interpolation as i64;
        let weight1 = i64::from(previous[1])
            + (i64::from(weights[1] - previous[1]) * position) / interpolation as i64;

        let mid_before = if index == 0 {
            i64::from(mid_history[1])
        } else {
            i64::from(mid[index - 1])
        };
        let mid_before2 = if index == 0 {
            i64::from(mid_history[0])
        } else if index == 1 {
            i64::from(mid_history[1])
        } else {
            i64::from(mid[index - 2])
        };
        let side_before = if index == 0 {
            i64::from(side_history[1])
        } else {
            i64::from(side[index - 1])
        };

        let lowpass = (mid_before2 + 2 * mid_before + i64::from(mid[index])) >> 2;
        let common = (weight0 * lowpass) >> 13;
        let coupling = (weight1 * mid_before) >> 13;

        let left_value = mid_before + coupling + side_before + common;
        let right_value = mid_before - coupling - side_before - common;
        left[index] = left_value.clamp(-32768, 32767) as i16;
        right[index] = right_value.clamp(-32768, 32767) as i16;
    }

    if mid.len() >= 2 {
        let end = mid.len();
        *mid_history = [mid[end - 2], mid[end - 1]];
        *side_history = [side[end - 2], side[end - 1]];
    }
    *previous_weights = weights;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_mid_and_side_reconstructs_both_channels() {
        let mut previous = [0_i32; 2];
        let mut mid_history = [0_i16; 2];
        let mut side_history = [0_i16; 2];
        let mid = [1000_i16; 160];
        let side = [0_i16; 160];
        let mut left = [0_i16; 160];
        let mut right = [0_i16; 160];
        mid_side_to_left_right(
            &mut previous,
            &mut mid_history,
            &mut side_history,
            [0, 0],
            &mid,
            &side,
            8,
            &mut left,
            &mut right,
        );
        for index in 2..160 {
            assert_eq!(left[index], 1000);
            assert_eq!(right[index], 1000);
        }
    }
}
