// SPDX-License-Identifier: Apache-2.0
//! LC3 low-delay inverse MDCT (Bluetooth LC3 v1.0.1 §3.4.8 and §3.7.2).
//!
//! The synthesis equation is evaluated as a DCT-IV of the spectrum followed by
//! the time-domain aliasing and windowing steps of §3.4.8. The DCT-IV basis is
//! precomputed once per frame size and shared by all decoders.

use std::sync::OnceLock;

use super::Lc3Error;
use super::tables::{ND, NS, W_7M5, W_10M};

/// Frame sizes used by the supported configurations.
const SIZES: [usize; 9] = [60, 80, 120, 160, 180, 240, 320, 360, 480];

/// Lazily built DCT-IV bases, indexed like [`SIZES`].
static BASES: [OnceLock<Vec<f32>>; 9] = [const { OnceLock::new() }; 9];

fn basis_index(n: usize) -> Option<usize> {
    SIZES.iter().position(|&size| size == n)
}

/// Returns the shared DCT-IV matrix for `n`.
pub(crate) fn basis(n: usize) -> &'static [f32] {
    let index = basis_index(n).expect("LC3 frame size is one of the supported sizes");
    BASES[index].get_or_init(|| {
        let mut table = vec![0f32; n * n];
        let pi = std::f64::consts::PI;
        for j in 0..n {
            for k in 0..n {
                let angle = pi / n as f64 * (j as f64 + 0.5) * (k as f64 + 0.5);
                table[j * n + k] = angle.cos() as f32;
            }
        }
        table
    })
}

/// Returns the window for a frame duration and sample-rate index.
pub(crate) fn window(dt: usize, sr: usize) -> Result<&'static [f32], Lc3Error> {
    match dt {
        0 => Ok(W_7M5[sr]),
        1 => Ok(W_10M[sr]),
        _ => Err(Lc3Error::InvalidConfig("invalid LC3 frame duration")),
    }
}

/// Per-channel inverse MDCT state (window, overlap history).
#[derive(Debug, Clone)]
pub(crate) struct MdctInverse {
    ns: usize,
    nd: usize,
    window: &'static [f32],
    history: Vec<f32>,
}

impl MdctInverse {
    pub(crate) fn new(dt: usize, sr: usize) -> Result<Self, Lc3Error> {
        let window = match dt {
            0 => W_7M5[sr],
            1 => W_10M[sr],
            _ => return Err(Lc3Error::InvalidConfig("invalid LC3 frame duration")),
        };
        let ns = NS[dt][sr];
        if window.len() != 2 * ns {
            return Err(Lc3Error::InvalidConfig("LC3 window length mismatch"));
        }
        if basis_index(ns).is_none() {
            return Err(Lc3Error::InvalidConfig("unsupported LC3 frame size"));
        }
        Ok(Self {
            ns,
            nd: ND[dt][sr],
            window,
            history: vec![0.0; 2 * ns],
        })
    }

    /// Inverse-transforms `x` and returns `ns` time-domain samples.
    pub(crate) fn run(&mut self, x: &[f32]) -> Result<Vec<f32>, Lc3Error> {
        let ns = self.ns;
        let nd = self.nd;
        if x.len() != ns {
            return Err(Lc3Error::InvalidConfig("LC3 MDCT input length"));
        }
        let n2 = 2 * ns;
        let table = basis(ns);

        // DCT-IV of the spectrum: d[j] = sum_k x[k] cos(pi/N (j+1/2)(k+1/2)).
        let mut dct = vec![0f32; ns];
        for (j, value) in dct.iter_mut().enumerate() {
            let row = &table[j * ns..(j + 1) * ns];
            let mut sum = 0f32;
            for (&coefficient, &sample) in row.iter().zip(x) {
                sum += coefficient * sample;
            }
            *value = sum;
        }

        // Time-domain aliasing: t[n] uses d at n + N/2 with the symmetries
        // d(2N-1-j) = -d(j) and d(j+2N) = -d(j).
        let scale = (2.0 / ns as f32).sqrt();
        let mut t = vec![0f32; n2];
        for (n, value) in t.iter_mut().enumerate() {
            let mut j = n + ns / 2;
            let mut sign = 1.0f32;
            if j >= n2 {
                j -= n2;
                sign = -1.0;
            }
            let aliased = if j < ns { dct[j] } else { -dct[n2 - 1 - j] };
            *value = sign * aliased * scale * self.window[n2 - 1 - n];
        }

        let mut y = vec![0f32; ns];
        for index in 0..nd {
            y[index] = t[ns - nd + index] + self.history[2 * ns - nd + index];
        }
        y[nd..].copy_from_slice(&t[ns..2 * ns - nd]);
        self.history = t;
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct double-precision evaluation of the synthesis equation.
    fn reference_transform(dt: usize, sr: usize, x: &[f32]) -> Vec<f32> {
        let ns = NS[dt][sr];
        let n2 = 2 * ns;
        let window = if dt == 0 { W_7M5[sr] } else { W_10M[sr] };
        let scale = (2.0 / ns as f64).sqrt();
        let pi = std::f64::consts::PI;
        let mut t = vec![0f64; n2];
        for (n, value) in t.iter_mut().enumerate() {
            let mut sum = 0f64;
            for (k, &sample) in x.iter().enumerate() {
                let angle = pi / ns as f64 * (n as f64 + 0.5 + ns as f64 / 2.0) * (k as f64 + 0.5);
                sum += f64::from(sample) * angle.cos();
            }
            *value = sum * scale * f64::from(window[n2 - 1 - n]);
        }
        let nd = ND[dt][sr];
        let mut y = vec![0f32; ns];
        for index in 0..nd {
            y[index] = t[ns - nd + index] as f32;
        }
        for index in nd..ns {
            y[index] = t[ns + index - nd] as f32;
        }
        y
    }

    #[test]
    fn matches_the_direct_synthesis_equation() {
        let mut state = 7u32;
        for (dt, sizes) in NS.iter().enumerate() {
            for (sr, &ns) in sizes.iter().enumerate() {
                let x: Vec<f32> = (0..ns)
                    .map(|_| {
                        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                        ((state >> 8) & 0xffff) as f32 / 32_768.0 - 1.0
                    })
                    .collect();
                let expected = reference_transform(dt, sr, &x);
                let mut mdct = MdctInverse::new(dt, sr).unwrap();
                let actual = mdct.run(&x).unwrap();
                let worst = expected
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(worst < 0.02, "dt={dt} sr={sr}: transform deviation {worst}");
            }
        }
    }
}
