// SPDX-License-Identifier: Apache-2.0
//! LC3 Temporal Noise Shaping analysis and filtering
//! (Bluetooth LC3 v1.0.1 §3.3.8).

use std::f32::consts::PI;

use super::tables::{TNS_FREQ_LIM, TNS_SUB_7M5, TNS_SUB_10M};

/// Reflection-coefficient quantizer step.
const DELTA: f32 = PI / 17.0;

/// One TNS filter as transmitted in the bitstream.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct TnsFilter {
    /// Number of transmitted reflection coefficients (`0` means inactive).
    pub(crate) rc_order: usize,
    /// Quantized reflection coefficients.
    pub(crate) rc_int: [i32; 8],
    /// Dequantized reflection coefficients.
    pub(crate) rc_q: [f32; 8],
}

/// TNS parameters for one channel of one frame.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct TnsParams {
    /// Number of filters (`1` or `2`).
    pub(crate) filters: usize,
    /// Whether the low-order weighting was used.
    pub(crate) lpc_weighting: bool,
    /// The per-filter parameters.
    pub(crate) filter: [TnsFilter; 2],
    /// Bits consumed by TNS in this frame.
    pub(crate) nbits: i32,
}

/// Returns the subdivision boundaries of the filters for a configuration.
fn sub_bands(dt: usize, bw: usize) -> &'static [i32] {
    if dt == 0 {
        TNS_SUB_7M5[bw]
    } else {
        TNS_SUB_10M[bw]
    }
}

/// Estimates the TNS filters and their bit cost (§3.3.8.2 and §3.3.8.3).
pub(crate) fn analyse(
    xs: &[f32],
    dt: usize,
    bw: usize,
    nbits: i32,
    near_nyquist: bool,
) -> TnsParams {
    let limits = TNS_FREQ_LIM[dt][bw];
    let filters = 1 + usize::from(bw >= 3);
    let lpc_weighting = nbits < if dt == 0 { 360 } else { 480 };
    let subs = sub_bands(dt, bw);

    let mut params = TnsParams {
        filters,
        lpc_weighting,
        ..TnsParams::default()
    };

    for f in 0..filters {
        let mut e = [0f32; 3];
        for (s, energy) in e.iter_mut().enumerate() {
            let start = subs[3 * f + s] as usize;
            let stop = subs[3 * f + s + 1] as usize;
            *energy = xs[start..stop].iter().map(|v| v * v).sum();
        }

        // Normalized autocorrelation (Equations 65-67).
        let mut r = [0f32; 9];
        if e.contains(&0.0) {
            r[0] = 3.0;
        } else {
            for (k, value) in r.iter_mut().enumerate() {
                let mut sum = 0f32;
                for (s, &energy) in e.iter().enumerate() {
                    let start = subs[3 * f + s] as usize;
                    let stop = subs[3 * f + s + 1] as usize;
                    let mut band = 0f32;
                    for n in start..stop.saturating_sub(k) {
                        band += xs[n] * xs[n + k];
                    }
                    sum += band / energy;
                }
                *value = sum;
            }
        }

        // Lag window (Equation 68).
        let mut rw = [0f32; 9];
        for (k, (&r, value)) in r.iter().zip(rw.iter_mut()).enumerate() {
            *value = r * (-0.5 * (0.02 * PI * k as f32).powi(2)).exp();
        }

        // Levinson-Durbin recursion.
        let mut a = [0f32; 9];
        let mut err = rw[0];
        a[0] = 1.0;
        let mut tmp = [0f32; 9];
        for k in 1..=8 {
            if err <= 0.0 {
                break;
            }
            let mut sum = 0f32;
            for n in 0..k {
                sum += a[n] * rw[k - n];
            }
            let rc = -sum / err;
            tmp[0] = 1.0;
            for n in 1..k {
                tmp[n] = a[n] + rc * a[k - n];
            }
            tmp[k] = rc;
            a[..=k].copy_from_slice(&tmp[..=k]);
            err *= 1.0 - rc * rc;
        }

        let pred_gain = if err > 0.0 { rw[0] / err } else { 1.0 };
        if pred_gain <= 1.5 || near_nyquist {
            continue;
        }

        // LPC weighting (Equations 70-72).
        let gamma = if lpc_weighting && pred_gain < 2.0 {
            1.0 - 0.15 * (2.0 - pred_gain) / 0.5
        } else {
            1.0
        };
        let mut aw = [0f32; 9];
        let mut power = 1.0f32;
        for (weighted, &coefficient) in aw.iter_mut().zip(&a) {
            *weighted = power * coefficient;
            power *= gamma;
        }

        // Weighted LPC to reflection coefficients.
        let mut tmp1 = aw;
        let mut rc = [0f32; 8];
        let mut tmp2 = [0f32; 9];
        for k in (1..=8).rev() {
            let value = tmp1[k];
            rc[k - 1] = value;
            let e = 1.0 - value * value;
            if e <= 0.0 {
                rc[k - 1..].fill(0.0);
                break;
            }
            for n in 1..k {
                tmp2[n] = (tmp1[n] - value * tmp1[k - n]) / e;
            }
            tmp1[1..k].copy_from_slice(&tmp2[1..k]);
        }

        // Quantization (Equations 73-74).
        let mut filter = TnsFilter::default();
        for k in 0..8 {
            let index = (rc[k].asin() / DELTA).round() as i32 + 8;
            filter.rc_int[k] = index.clamp(0, 16);
            filter.rc_q[k] = (DELTA * (filter.rc_int[k] - 8) as f32).sin();
        }
        let mut order = 8;
        while order > 0 && filter.rc_q[order - 1] == 0.0 {
            order -= 1;
        }
        filter.rc_order = order;
        params.filter[f] = filter;
    }

    let mut nbits_tns = 0i32;
    for filter in params.filter.iter().take(filters) {
        let order = filter.rc_order;
        let (order_bits, coef_bits) = if order > 0 {
            let order_bits = i32::from(TNS_ORDER_BITS[usize::from(lpc_weighting)][order - 1]);
            let coef_bits = (0..order)
                .map(|k| i32::from(TNS_COEF_BITS[k][filter.rc_int[k] as usize]))
                .sum();
            (order_bits, coef_bits)
        } else {
            (0, 0)
        };
        nbits_tns += (2048 + order_bits + coef_bits + 2047) / 2048;
    }
    params.nbits = nbits_tns;
    let _ = limits;
    params
}

/// Writes the TNS order and reflection coefficients to the arithmetic coder
/// (§3.3.13.4.2).
pub(crate) fn write_ac(ac: &mut super::enc_bits::AcEncoder, params: &TnsParams) {
    let weighting = usize::from(params.lpc_weighting);
    for filter in params.filter.iter().take(params.filters) {
        if filter.rc_order == 0 {
            continue;
        }
        let order = filter.rc_order;
        ac.encode(
            super::tables::TNS_ORDER_CUMFREQ[weighting][order - 1],
            super::tables::TNS_ORDER_FREQ[weighting][order - 1],
        );
        for (k, &rc) in filter.rc_int[..order].iter().enumerate() {
            let index = rc.clamp(0, 16) as usize;
            ac.encode(
                super::tables::TNS_COEF_CUMFREQ[k][index],
                super::tables::TNS_COEF_FREQ[k][index],
            );
        }
    }
}

/// Applies the TNS analysis filter to the SNS-shaped spectrum (§3.3.8.4).
pub(crate) fn filter(xs: &mut [f32], dt: usize, bw: usize, params: &TnsParams) {
    let limits = TNS_FREQ_LIM[dt][bw];
    let mut state = [0f32; 8];
    for (f, filter) in params.filter.iter().enumerate().take(params.filters) {
        let order = filter.rc_order;
        if order == 0 {
            continue;
        }
        let start = limits[f] as usize;
        let stop = limits[f + 1] as usize;
        for value in &mut xs[start..stop] {
            let mut t = *value;
            let mut save = t;
            for k in 0..order - 1 {
                let tmp = filter.rc_q[k] * t + state[k];
                t += filter.rc_q[k] * state[k];
                state[k] = save;
                save = tmp;
            }
            t += filter.rc_q[order - 1] * state[order - 1];
            state[order - 1] = save;
            *value = t;
        }
    }
}

/// Bit estimates for the TNS order, scaled by 2048.
static TNS_ORDER_BITS: [[u16; 8]; 2] = super::enc_tables::ENC_TNS_ORDER_BITS;
/// Bit estimates for the TNS coefficients, scaled by 2048.
static TNS_COEF_BITS: [[u16; 17]; 8] = super::enc_tables::ENC_TNS_COEF_BITS;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lc3::tns::TnsSynthesis;

    #[test]
    fn analysis_filter_is_inverted_by_the_synthesis_filter() {
        let mut state = 5u32;
        for dt in 0..2 {
            for bw in 0..5 {
                let ne = *crate::lc3::spec::band_limits(dt, bw)
                    .last()
                    .expect("band limits") as usize;
                let mut x: Vec<f32> = (0..ne)
                    .map(|_| {
                        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                        ((state >> 8) & 0xffff) as f32 / 32768.0 - 1.0
                    })
                    .collect();
                let original = x.clone();
                let mut params = analyse(&x, dt, bw, 600, false);
                for filter in &mut params.filter[..params.filters] {
                    filter.rc_order = 8;
                    for k in 0..8 {
                        filter.rc_int[k] = 8 + ((k as i32) % 3);
                        filter.rc_q[k] = (DELTA * (filter.rc_int[k] - 8) as f32).sin();
                    }
                }
                filter(&mut x, dt, bw, &params);

                let mut synth = TnsSynthesis::new(dt);
                synth.set_for_test(
                    params.filters,
                    [params.filter[0].rc_order, params.filter[1].rc_order],
                    [params.filter[0].rc_int, params.filter[1].rc_int],
                );
                synth.run(&mut x, bw).unwrap();
                let worst = original
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(worst < 1e-3, "dt={dt} bw={bw}: TNS round-trip {worst}");
            }
        }
    }
}
