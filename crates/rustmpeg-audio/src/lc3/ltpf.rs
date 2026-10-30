// SPDX-License-Identifier: Apache-2.0
//! LC3 Long Term Postfilter synthesis (Bluetooth LC3 v1.0.1 §3.4.9).
//!
//! The postfilter is a pitch-prediction filter applied to the MDCT synthesis
//! output. Filter coefficients, gain, and transition handling follow §3.4.9.2
//! through §3.4.9.4.

use super::Lc3Error;
use super::bits::BitReader;
use super::tables::{
    LTPF_D_8K, LTPF_D_16K, LTPF_D_24K, LTPF_D_32K, LTPF_D_48K, LTPF_N_8K, LTPF_N_16K, LTPF_N_24K,
    LTPF_N_32K, LTPF_N_48K, NS, SRATE_KHZ,
};

/// Numerator coefficients for a sample rate and gain index.
fn numerator(sr: usize, gain_ind: usize) -> &'static [f32] {
    match sr {
        0 => &LTPF_N_8K[gain_ind],
        1 => &LTPF_N_16K[gain_ind],
        2 => &LTPF_N_24K[gain_ind],
        3 => &LTPF_N_32K[gain_ind],
        _ => &LTPF_N_48K[gain_ind],
    }
}

/// Denominator coefficients for a sample rate and fractional pitch.
fn denominator(sr: usize, p_fr: usize) -> &'static [f32] {
    match sr {
        0 => &LTPF_D_8K[p_fr],
        1 => &LTPF_D_16K[p_fr],
        2 => &LTPF_D_24K[p_fr],
        3 => &LTPF_D_32K[p_fr],
        _ => &LTPF_D_48K[p_fr],
    }
}

/// LTPF synthesis state for one channel.
pub(crate) struct LtpfSynthesis {
    dt: usize,
    sr: usize,
    ns: usize,
    l_den: usize,
    l_num: usize,
    /// Previous frame filter activity (`mem_ltpf_active`).
    active: bool,
    /// Previous frame pitch in quarter samples at the output rate.
    pitch: i32,
    c_num: Vec<f32>,
    c_den: Vec<f32>,
    /// Last `l_num` samples of the MDCT synthesis input.
    in_hist: Vec<f32>,
    /// Past samples of the final LTPF output.
    out_hist: Vec<f32>,
    /// Current frame LTPF activity flag.
    data_active: bool,
    /// Current frame pitch index.
    pitch_index: u32,
}

impl LtpfSynthesis {
    pub(crate) fn new(dt: usize, sr: usize) -> Self {
        let ns = NS[dt][sr];
        let sr_hz = SRATE_KHZ[sr] * 1000;
        let l_den = (sr_hz / 4000).max(4) as usize;
        let l_num = l_den - 2;
        // The largest pitch (index 511) is 228 samples at 12.8 kHz, i.e. 855
        // samples at 48 kHz, plus the denominator half-width and taps.
        let history = 960;
        Self {
            dt,
            sr,
            ns,
            l_den,
            l_num,
            active: false,
            pitch: 0,
            c_num: vec![0.0; l_num + 1],
            c_den: vec![0.0; l_den + 1],
            in_hist: vec![0.0; l_num],
            out_hist: vec![0.0; history],
            data_active: false,
            pitch_index: 0,
        }
    }

    pub(crate) fn load(&mut self, bits: &mut BitReader<'_>) -> Result<(), Lc3Error> {
        self.data_active = bits.read_bit()?;
        self.pitch_index = bits.read_uint(9)?;
        Ok(())
    }

    pub(crate) fn disable(&mut self) {
        self.data_active = false;
        self.pitch_index = 0;
    }

    /// Applies the postfilter to `x` and returns the filtered frame.
    pub(crate) fn run(&mut self, x: &[f32], nbytes: usize) -> Result<Vec<f32>, Lc3Error> {
        if x.len() != self.ns {
            return Err(Lc3Error::InvalidConfig("LTPF input length"));
        }
        let ns = self.ns;
        let sr = self.sr;

        // Gain tuning (§3.4.9.4). For 7.5 ms frames the bit count is scaled to
        // the 10 ms reference with rounding, `round(nbits * 10 / 7.5)`.
        let nbits = if self.dt == 0 {
            (nbytes as i32 * 32 + 1) / 3
        } else {
            nbytes as i32 * 8
        };
        let base = 3 + sr as i32;
        let g_idx = (nbits / 80).max(base) - base;
        let active = self.data_active && g_idx < 4;
        let gain = if active {
            0.4 - 0.05 * g_idx as f32
        } else {
            0.0
        };
        let gain_ind = g_idx.min(3) as usize;

        let (pitch, p_int, p_fr) = pitch_parts(self.pitch_index, sr);
        let mut c_num = vec![0.0f32; self.l_num + 1];
        let mut c_den = vec![0.0f32; self.l_den + 1];
        if active {
            let num = numerator(sr, gain_ind);
            let den = denominator(sr, p_fr as usize);
            for (value, &coefficient) in c_num.iter_mut().zip(num) {
                *value = 0.85 * gain * coefficient;
            }
            for (value, &coefficient) in c_den.iter_mut().zip(den) {
                *value = gain * coefficient;
            }
        }

        let n_norm = ns / (if self.dt == 0 { 3 } else { 4 });
        let mut y = vec![0.0f32; ns];

        if !self.active && !active {
            y.copy_from_slice(x);
        } else if !self.active && active {
            for n in 0..ns {
                let g = if n < n_norm {
                    n as f32 / n_norm as f32
                } else {
                    1.0
                };
                let f = self.prediction(n, &y, x, &c_num, &c_den, p_int, false);
                y[n] = x[n] - g * f;
            }
        } else if self.active && !active {
            let old_pitch = self.pitch / 4;
            for n in 0..ns {
                let g = if n < n_norm {
                    1.0 - n as f32 / n_norm as f32
                } else {
                    0.0
                };
                let f = self.prediction(n, &y, x, &self.c_num, &self.c_den, old_pitch, false);
                y[n] = x[n] - g * f;
            }
        } else if self.pitch == pitch {
            for n in 0..ns {
                let f = self.prediction(n, &y, x, &c_num, &c_den, p_int, false);
                y[n] = x[n] - f;
            }
        } else {
            let old_pitch = self.pitch / 4;
            let mut y_prime = x.to_vec();
            for n in 0..n_norm {
                let g = 1.0 - n as f32 / n_norm as f32;
                let f = self.prediction(n, &y_prime, x, &self.c_num, &self.c_den, old_pitch, false);
                y_prime[n] = x[n] - g * f;
            }
            for n in 0..ns {
                let g = if n < n_norm {
                    n as f32 / n_norm as f32
                } else {
                    1.0
                };
                let f = self.prediction(n, &y, &y_prime, &c_num, &c_den, p_int, true);
                y[n] = y_prime[n] - g * f;
            }
        }

        let split = ns - self.l_num;
        self.in_hist.copy_from_slice(&x[split..]);
        self.out_hist.copy_within(ns.., 0);
        let history_len = self.out_hist.len();
        self.out_hist[history_len - ns..].copy_from_slice(&y);
        self.active = active;
        self.pitch = pitch;
        self.c_num = c_num;
        self.c_den = c_den;
        Ok(y)
    }

    /// Evaluates the filter prediction at sample `n`.
    ///
    /// `out` holds the final output samples computed so far. `input` is either
    /// the MDCT synthesis input or the intermediate fade-out output, depending
    /// on the transition case.
    #[allow(clippy::too_many_arguments)]
    fn prediction(
        &self,
        n: usize,
        out: &[f32],
        input: &[f32],
        c_num: &[f32],
        c_den: &[f32],
        p_int: i32,
        input_from_history: bool,
    ) -> f32 {
        let mut prediction = 0.0f32;
        for (k, &coefficient) in c_num.iter().enumerate() {
            let index = n as isize - k as isize;
            let sample = if index >= 0 {
                input[index as usize]
            } else if input_from_history {
                self.out_hist[self.out_hist.len() - (-index) as usize]
            } else {
                self.in_hist[self.in_hist.len() - (-index) as usize]
            };
            prediction += coefficient * sample;
        }
        let mut delayed = 0.0f32;
        let center = (self.l_den / 2) as isize;
        for (k, &coefficient) in c_den.iter().enumerate() {
            let index = n as isize - p_int as isize + center - k as isize;
            let sample = if index >= 0 {
                out[index as usize]
            } else {
                self.out_hist[self.out_hist.len() - (-index) as usize]
            };
            delayed += coefficient * sample;
        }
        prediction - delayed
    }
}

/// Computes the scaled pitch and its integer and fractional parts (§3.4.9.4).
fn pitch_parts(pitch_index: u32, sr: usize) -> (i32, i32, i32) {
    let pitch_12k8 = if pitch_index >= 440 {
        ((pitch_index - 283) as i32) << 2
    } else if pitch_index >= 380 {
        (((pitch_index >> 1) as i32 - 63) << 2) + (((pitch_index & 1) as i32) << 1)
    } else {
        (((pitch_index >> 2) as i32 + 32) << 2) + (pitch_index & 3) as i32
    };
    // Scale from 12.8 kHz to the output rate, in quarter samples.
    let ns_10ms = NS[1][sr];
    let scaled = (pitch_12k8 * ns_10ms as i32 + 64) / 128;
    (scaled, scaled / 4, scaled & 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_parts_are_consistent() {
        for index in [0u32, 100, 379, 380, 439, 440, 511] {
            for sr in 0..5 {
                let (pitch, p_int, p_fr) = pitch_parts(index, sr);
                assert_eq!(p_int * 4 + p_fr, pitch);
            }
        }
    }
}
