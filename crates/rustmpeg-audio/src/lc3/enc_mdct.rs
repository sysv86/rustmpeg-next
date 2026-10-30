// SPDX-License-Identifier: Apache-2.0
//! LC3 low-delay MDCT analysis (Bluetooth LC3 v1.0.1 §3.3.4).
//!
//! The analysis transform of Equation 8 is evaluated as a time-domain folding
//! followed by the DCT-IV that the synthesis already uses. For
//! `X(k) = scale * sum_n w(n) t(n) cos(pi/N (n + 1/2 + N/2)(k + 1/2))`, the
//! cosine indices `j = n + N/2` fold into `[0, N)` with the symmetries
//! `C(2N-1-j) = -C(j)` and `C(j-2N) = -C(j)`.

use super::Lc3Error;
use super::mdct::{basis, window};
use super::tables::{I_7M5, I_10M, NS};

/// Per-channel forward MDCT state.
pub(crate) struct MdctAnalysis {
    ns: usize,
    ne: usize,
    zeros: usize,
    window: &'static [f32],
    history: Vec<f32>,
}

impl MdctAnalysis {
    /// Creates an analysis transform for a frame duration and sample rate.
    pub(crate) fn new(dt: usize, sr: usize) -> Result<Self, Lc3Error> {
        let window = window(dt, sr)?;
        let ns = NS[dt][sr];
        if window.len() != 2 * ns {
            return Err(Lc3Error::InvalidConfig("LC3 window length mismatch"));
        }
        let zeros = window
            .iter()
            .rev()
            .take_while(|&&value| value == 0.0)
            .count();
        let ne = encoded_length(dt, sr);
        Ok(Self {
            ns,
            ne,
            zeros,
            window,
            history: vec![0.0; ns],
        })
    }

    /// Transforms one frame of `ns` samples and returns `NE` coefficients.
    pub(crate) fn run(&mut self, frame: &[f32]) -> Result<Vec<f32>, Lc3Error> {
        let ns = self.ns;
        let n2 = 2 * ns;
        if frame.len() != ns {
            return Err(Lc3Error::InvalidConfig("LC3 MDCT frame length"));
        }

        // Equation 6: t holds the newest 2N-Z input samples, then Z zeros.
        let mut t = vec![0f32; n2];
        for (index, value) in t.iter_mut().enumerate().take(n2 - self.zeros) {
            let combined = index + self.zeros;
            *value = if combined < ns {
                self.history[combined]
            } else {
                frame[combined - ns]
            };
        }
        self.history.copy_from_slice(frame);

        // Fold the windowed input into the DCT-IV domain.
        let mut folded = vec![0f32; ns];
        for (n, &sample) in t.iter().enumerate() {
            let j = n + ns / 2;
            let (target, sign) = if j < ns {
                (j, 1.0f32)
            } else if j < n2 {
                (n2 - 1 - j, -1.0f32)
            } else {
                (j - n2, -1.0f32)
            };
            folded[target] += sample * self.window[n] * sign;
        }

        // X(k) = sqrt(2/N) * sum_q folded(q) cos(pi/N (q + 1/2)(k + 1/2)).
        let table = basis(ns);
        let scale = (2.0 / ns as f32).sqrt();
        let mut out = Vec::with_capacity(self.ne);
        for k in 0..self.ne {
            let row = &table[k * ns..(k + 1) * ns];
            let mut sum = 0f32;
            for (&coefficient, &sample) in row.iter().zip(&folded) {
                sum += coefficient * sample;
            }
            out.push(sum * scale);
        }
        Ok(out)
    }
}

/// Number of encoded coefficients for a frame duration and sample rate.
pub(crate) fn encoded_length(dt: usize, sr: usize) -> usize {
    let ns = NS[dt][sr];
    match ns {
        480 => 400,
        360 => 300,
        _ => ns,
    }
}

/// Number of bands for a frame duration and sample rate.
pub(crate) fn band_count(dt: usize, sr: usize) -> usize {
    let limits = match dt {
        0 => I_7M5[sr],
        _ => I_10M[sr],
    };
    limits.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lc3::tables::{W_7M5, W_10M};

    /// Direct double-precision evaluation of the analysis equation.
    fn reference(dt: usize, sr: usize, history: &[f32], frame: &[f32]) -> Vec<f32> {
        let ns = NS[dt][sr];
        let n2 = 2 * ns;
        let window = if dt == 0 { W_7M5[sr] } else { W_10M[sr] };
        let zeros = window.iter().rev().take_while(|&&v| v == 0.0).count();
        let mut t = vec![0f64; n2];
        for (n, slot) in t.iter_mut().enumerate().take(n2 - zeros) {
            let combined = n + zeros;
            *slot = f64::from(if combined < ns {
                history[combined]
            } else {
                frame[combined - ns]
            });
        }
        let pi = std::f64::consts::PI;
        let ne = match ns {
            480 => 400,
            360 => 300,
            _ => ns,
        };
        (0..ne)
            .map(|k| {
                let mut sum = 0f64;
                for (n, (&weight, &value)) in window.iter().zip(&t).enumerate() {
                    let angle =
                        pi / ns as f64 * (n as f64 + 0.5 + ns as f64 / 2.0) * (k as f64 + 0.5);
                    sum += f64::from(weight) * value * angle.cos();
                }
                (sum * (2.0 / ns as f64).sqrt()) as f32
            })
            .collect()
    }

    #[test]
    fn matches_the_direct_analysis_equation() {
        let mut state = 9u32;
        for (dt, row) in NS.iter().enumerate() {
            for (sr, &ns) in row.iter().enumerate() {
                let mut samples = || {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    ((state >> 8) & 0xffff) as f32 / 32_768.0 - 1.0
                };
                let history: Vec<f32> = (0..ns).map(|_| samples()).collect();
                let frame: Vec<f32> = (0..ns).map(|_| samples()).collect();
                let expected = reference(dt, sr, &history, &frame);
                let mut mdct = MdctAnalysis::new(dt, sr).unwrap();
                // Prime the history buffer with the same samples.
                let _ = mdct.run(&history).unwrap();
                let actual = mdct.run(&frame).unwrap();
                let worst = expected
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(worst < 0.05, "dt={dt} sr={sr}: analysis deviation {worst}");
            }
        }
    }
}
