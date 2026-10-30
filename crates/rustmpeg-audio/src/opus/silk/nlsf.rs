// SPDX-License-Identifier: Apache-2.0
//! SILK normalized-LSF reconstruction and LPC conversion.
//!
//! This module implements RFC 6716, Sections 4.2.7.5.2 through 4.2.7.5.8:
//! stage-2 residual dequantization, reconstruction of the normalized LSF
//! vector, stabilization, interpolation, and the fixed-point conversion to
//! Q12 LPC coefficients.  The conversion is integer-only and follows the
//! normative procedure so that every decoder derives identical coefficients
//! from the same indices.

use super::structs::NLSFCodebook;
use super::tables::SILK_LSF_COS_TAB_FIX_Q12;
use super::{MAX_LPC_ORDER, NLSF_QUANT_MAX_AMPLITUDE};

/// Quantization-level adjustment applied to non-zero stage-2 indices
/// (RFC 6716, Section 4.2.7.5.2).
const NLSF_QUANT_LEVEL_ADJ: i32 = 102;

/// Number of LSF coefficients in the polynomial recurrence.
const NLSF_MAX_HALF_ORDER: usize = MAX_LPC_ORDER / 2;

/// Unpacks the stage-2 entropy offsets and prediction weights for one
/// stage-1 codebook vector (RFC 6716, Section 4.2.7.5.2).
pub(crate) fn unpack(
    ec_ix: &mut [i16; MAX_LPC_ORDER],
    pred_q8: &mut [u8; MAX_LPC_ORDER],
    codebook: &NLSFCodebook,
    stage1_index: usize,
) {
    let order = codebook.order as usize;
    let selector = &codebook.ec_sel[stage1_index * order / 2..];
    for index in (0..order).step_by(2) {
        let entry = selector[index / 2];
        ec_ix[index] = i16::from((entry >> 1) & 7) * i16::from(2 * NLSF_QUANT_MAX_AMPLITUDE + 1);
        pred_q8[index] = codebook.pred_q8[index + ((entry & 1) as usize) * (order - 1)];
        ec_ix[index + 1] =
            i16::from((entry >> 5) & 7) * i16::from(2 * NLSF_QUANT_MAX_AMPLITUDE + 1);
        pred_q8[index + 1] =
            codebook.pred_q8[index + (((entry >> 4) & 1) as usize) * (order - 1) + 1];
    }
}

/// Reconstructs the normalized LSF vector from the decoded indices and
/// applies the minimum-spacing stabilization (RFC 6716, Sections 4.2.7.5.2
/// through 4.2.7.5.4).
pub(crate) fn decode(
    nlsf_indices: &[i8; MAX_LPC_ORDER + 1],
    codebook: &NLSFCodebook,
) -> [i16; MAX_LPC_ORDER] {
    let order = codebook.order as usize;
    let stage1 = nlsf_indices[0] as usize;

    let mut ec_ix = [0_i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0_u8; MAX_LPC_ORDER];
    unpack(&mut ec_ix, &mut pred_q8, codebook, stage1);

    let mut res_q10 = [0_i16; MAX_LPC_ORDER];
    let mut out_q10 = 0_i32;
    for index in (0..order).rev() {
        let prediction = (out_q10 * i32::from(pred_q8[index])) >> 8;
        let mut level = i32::from(nlsf_indices[index + 1]) << 10;
        if level > 0 {
            level -= NLSF_QUANT_LEVEL_ADJ;
        } else if level < 0 {
            level += NLSF_QUANT_LEVEL_ADJ;
        }
        out_q10 = prediction + ((level * codebook.quant_step_size_q16) >> 16);
        res_q10[index] = out_q10 as i16;
    }

    let cb1 = &codebook.cb1_nlsf_q8[stage1 * order..];
    let weights = &codebook.cb1_wght_q9[stage1 * order..];
    let mut nlsf = [0_i16; MAX_LPC_ORDER];
    for index in 0..order {
        let center = i32::from(cb1[index]) << 7;
        let residual = (i32::from(res_q10[index]) << 14) / i32::from(weights[index]);
        nlsf[index] = (center + residual).clamp(0, 32767) as i16;
    }
    stabilize(&mut nlsf, order, codebook.delta_min_q15);
    nlsf
}

/// Enforces the minimum spacing of RFC 6716, Section 4.2.7.5.4.
fn stabilize(nlsf: &mut [i16; MAX_LPC_ORDER], order: usize, delta_min: &[i16]) {
    for _ in 0..20 {
        let mut smallest = i32::MAX;
        let mut index = 0;
        for k in 0..=order {
            let current = if k < order { i32::from(nlsf[k]) } else { 32768 };
            let previous = if k == 0 { 0 } else { i32::from(nlsf[k - 1]) };
            let margin = current - previous - i32::from(delta_min[k]);
            if margin < smallest {
                smallest = margin;
                index = k;
            }
        }
        if smallest >= 0 {
            return;
        }
        if index == 0 {
            nlsf[0] = delta_min[0];
        } else if index == order {
            nlsf[order - 1] = (32768 - i32::from(delta_min[order])) as i16;
        } else {
            let step = i32::from(delta_min[index]);
            let mut min_center = step >> 1;
            for value in &delta_min[..index] {
                min_center += i32::from(*value);
            }
            let mut max_center = 32768 - (step >> 1);
            for value in &delta_min[index + 1..=order] {
                max_center -= i32::from(*value);
            }
            let center = ((i32::from(nlsf[index - 1]) + i32::from(nlsf[index]) + 1) >> 1)
                .clamp(min_center, max_center);
            nlsf[index - 1] = (center - (step >> 1)) as i16;
            nlsf[index] = (i32::from(nlsf[index - 1]) + step) as i16;
        }
    }

    // Fallback: sort, then push coefficients apart in both directions.
    nlsf[..order].sort_unstable();
    for index in 0..order {
        let previous = if index == 0 {
            0
        } else {
            i32::from(nlsf[index - 1])
        };
        nlsf[index] = i32::from(nlsf[index]).max(previous + i32::from(delta_min[index])) as i16;
    }
    for index in (0..order).rev() {
        let next = if index + 1 == order {
            32768
        } else {
            i32::from(nlsf[index + 1])
        };
        nlsf[index] = i32::from(nlsf[index]).min(next - i32::from(delta_min[index + 1])) as i16;
    }
}

/// Interpolates between the previous and current normalized LSF vectors for
/// the first half of a 20 ms frame (RFC 6716, Section 4.2.7.5.5).
pub(crate) fn interpolate(
    previous: &[i16; MAX_LPC_ORDER],
    current: &[i16; MAX_LPC_ORDER],
    weight_q2: i8,
    order: usize,
) -> [i16; MAX_LPC_ORDER] {
    let mut nlsf = [0_i16; MAX_LPC_ORDER];
    for index in 0..order {
        let delta = i32::from(current[index]) - i32::from(previous[index]);
        nlsf[index] = (i32::from(previous[index]) + ((i32::from(weight_q2) * delta) >> 2)) as i16;
    }
    nlsf
}

/// Converts normalized LSFs to Q12 LPC coefficients (RFC 6716, Sections
/// 4.2.7.5.6 through 4.2.7.5.8).
pub(crate) fn to_lpc(nlsf: &[i16; MAX_LPC_ORDER], order: usize) -> [i16; MAX_LPC_ORDER] {
    debug_assert!(order == 10 || order == 16);
    const NB_ORDER: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];
    const WB_ORDER: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
    let ordering: &[usize] = if order == 16 { &WB_ORDER } else { &NB_ORDER };

    // Approximate cos(pi*n[k]) with the normative Q12 table, then promote to
    // Q17 and store it in the polynomial evaluation order.
    let mut cosine = [0_i64; MAX_LPC_ORDER];
    for index in 0..order {
        let value = i32::from(nlsf[index]);
        let table_index = (value >> 8) as usize;
        let fraction = value & 255;
        let base = i64::from(SILK_LSF_COS_TAB_FIX_Q12[table_index]);
        let next = i64::from(SILK_LSF_COS_TAB_FIX_Q12[table_index + 1]);
        cosine[ordering[index]] = (base * 256 + (next - base) * i64::from(fraction) + 4) >> 3;
    }

    // Products of the root-pair factors for P(z) and Q(z), Q16, truncated to
    // the symmetric half of each row.
    let half = order / 2;
    let mut p = [0_i64; NLSF_MAX_HALF_ORDER + 1];
    let mut q = [0_i64; NLSF_MAX_HALF_ORDER + 1];
    p[0] = 1 << 16;
    p[1] = -cosine[0];
    q[0] = 1 << 16;
    q[1] = -cosine[1];
    for k in 1..half {
        let mut next_p = [0_i64; NLSF_MAX_HALF_ORDER + 1];
        let mut next_q = [0_i64; NLSF_MAX_HALF_ORDER + 1];
        next_p[0] = p[0];
        next_q[0] = q[0];
        for j in 1..=k + 1 {
            // Row k-1 stores only its symmetric half.  For j = k + 1 the
            // stored half ends at index k - 1, which is the mirror of k + 1
            // (RFC 6716, Section 4.2.7.5.6).
            let previous = if j <= k { p[j] } else { p[k - 1] };
            let previous2 = if j >= 2 { p[j - 2] } else { 0 };
            next_p[j] = previous + previous2 - ((cosine[2 * k] * p[j - 1] + 32768) >> 16);

            let previous = if j <= k { q[j] } else { q[k - 1] };
            let previous2 = if j >= 2 { q[j - 2] } else { 0 };
            next_q[j] = previous + previous2 - ((cosine[2 * k + 1] * q[j - 1] + 32768) >> 16);
        }
        p = next_p;
        q = next_q;
    }

    let mut coefficients = [0_i32; MAX_LPC_ORDER];
    for k in 0..half {
        let difference = q[k + 1] - q[k];
        let sum = p[k + 1] + p[k];
        coefficients[k] = (-difference - sum) as i32;
        coefficients[order - k - 1] = (difference - sum) as i32;
    }

    limit_range(&mut coefficients, order);
    limit_prediction_gain(&mut coefficients, order);

    let mut lpc = [0_i16; MAX_LPC_ORDER];
    for index in 0..order {
        lpc[index] = ((coefficients[index] + 16) >> 5) as i16;
    }
    lpc
}

/// Bandwidth expansion in the Q17 domain (RFC 6716, Section 4.2.7.5.7).
fn expand(coefficients: &mut [i32; MAX_LPC_ORDER], order: usize, mut chirp_q16: i64) {
    let chirp_step = chirp_q16;
    for coefficient in coefficients.iter_mut().take(order) {
        *coefficient = ((i64::from(*coefficient) * chirp_q16) >> 16) as i32;
        chirp_q16 = (chirp_step * chirp_q16 + 32768) >> 16;
    }
}

/// Limits the dynamic range of the LPC coefficients (RFC 6716, Section
/// 4.2.7.5.7).
fn limit_range(coefficients: &mut [i32; MAX_LPC_ORDER], order: usize) {
    let mut saturated = true;
    for _ in 0..10 {
        let mut max_abs = 0_i32;
        let mut index = 0;
        for (position, value) in coefficients.iter().take(order).enumerate() {
            if value.abs() > max_abs {
                max_abs = value.abs();
                index = position;
            }
        }
        let max_abs_q12 = i64::from((max_abs + 16) >> 5).min(163838);
        if max_abs_q12 <= 32767 {
            saturated = false;
            break;
        }
        let numerator = (max_abs_q12 - 32767) << 14;
        let denominator = (max_abs_q12 * (index as i64 + 1)) >> 2;
        let chirp = 65470 - numerator / denominator;
        expand(coefficients, order, chirp);
    }
    if saturated {
        for coefficient in coefficients.iter_mut().take(order) {
            *coefficient = ((*coefficient + 16) >> 5).clamp(-32768, 32767) << 5;
        }
    }
}

/// Limits the prediction gain of the LPC filter (RFC 6716, Section
/// 4.2.7.5.8).
fn limit_prediction_gain(coefficients: &mut [i32; MAX_LPC_ORDER], order: usize) {
    for round in 0..16 {
        if inverse_prediction_gain_is_stable(coefficients, order) {
            return;
        }
        expand(coefficients, order, i64::from(65536 - (2 << round)));
    }
}

/// Fixed-point stability test from RFC 6716, Section 4.2.7.5.8.
fn inverse_prediction_gain_is_stable(coefficients: &[i32; MAX_LPC_ORDER], order: usize) -> bool {
    let mut q12 = [0_i32; MAX_LPC_ORDER];
    let mut dc_response = 0_i64;
    for index in 0..order {
        q12[index] = (coefficients[index] + 16) >> 5;
        dc_response += i64::from(q12[index]);
    }
    if dc_response > 4096 {
        return false;
    }

    let mut q24 = [0_i32; MAX_LPC_ORDER];
    for index in 0..order {
        q24[index] = q12[index] << 12;
    }
    let mut inverse_gain = 1_i64 << 30;
    for k in (0..order).rev() {
        if q24[k].unsigned_abs() > 16_773_022 {
            return false;
        }
        let reflection = (-i64::from(q24[k])) << 7;
        let denominator = (1_i64 << 30) - ((reflection * reflection) >> 32);
        inverse_gain = ((inverse_gain * denominator) >> 32) << 2;
        if inverse_gain < 107_374 {
            return false;
        }
        if k == 0 {
            continue;
        }

        let bits = 64 - denominator.leading_zeros() as i32;
        let fractional_bits = bits - 16;
        let scaled = denominator >> (fractional_bits + 1);
        if scaled == 0 {
            return false;
        }
        let inverse_scaled = ((1_i64 << 29) - 1) / scaled;
        let error =
            (1_i64 << 29) - (((denominator << (15 - fractional_bits)) * inverse_scaled) >> 16);
        let gain = (inverse_scaled << 16) + ((error * inverse_scaled) >> 13);

        let mut next = [0_i32; MAX_LPC_ORDER];
        for index in 0..k {
            let numerator = i64::from(q24[index])
                - ((i64::from(q24[k - index - 1]) * reflection + (1_i64 << 30)) >> 31);
            next[index] = ((numerator * gain + (1_i64 << (bits - 1))) >> bits) as i32;
        }
        q24[..k].copy_from_slice(&next[..k]);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opus::silk::tables_nlsf;

    #[test]
    fn reconstruct_and_convert_produce_stable_coefficients() {
        for (codebook, order) in [
            (&tables_nlsf::SILK_NLSF_CB_NB_MB, 10_usize),
            (&tables_nlsf::SILK_NLSF_CB_WB, 16),
        ] {
            for stage1 in 0..codebook.n_vectors as usize {
                let mut indices = [0_i8; MAX_LPC_ORDER + 1];
                indices[0] = stage1 as i8;
                let nlsf = decode(&indices, codebook);
                for index in 1..order {
                    assert!(
                        nlsf[index] > nlsf[index - 1],
                        "stage-1 {stage1}: LSF {index} is not increasing"
                    );
                }
                let lpc = to_lpc(&nlsf, order);
                assert!(
                    inverse_prediction_gain_is_stable(
                        &lpc.map(|value| i32::from(value) << 5),
                        order
                    ),
                    "stage-1 {stage1}: LPC filter is unstable"
                );
            }
        }
    }

    #[test]
    fn interpolation_moves_towards_the_current_vector() {
        let mut previous = [0_i16; MAX_LPC_ORDER];
        let mut current = [0_i16; MAX_LPC_ORDER];
        for index in 0..10 {
            previous[index] = (1000 + index * 100) as i16;
            current[index] = (2000 + index * 100) as i16;
        }
        let quarter = interpolate(&previous, &current, 1, 10);
        let half = interpolate(&previous, &current, 2, 10);
        for index in 0..10 {
            assert_eq!(
                quarter[index],
                previous[index] + (current[index] - previous[index]) / 4
            );
            assert_eq!(
                half[index],
                previous[index] + (current[index] - previous[index]) / 2
            );
        }
    }
}
