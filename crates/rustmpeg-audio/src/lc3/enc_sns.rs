// SPDX-License-Identifier: Apache-2.0
//! LC3 Spectral Noise Shaping analysis and quantization
//! (Bluetooth LC3 v1.0.1 §3.3.7).

use std::f32::consts::PI;

use super::Lc3Error;
use super::sns::idct16;
use super::tables::{
    SNS_HFCB, SNS_LFCB, SNS_MPVQ_OFFSETS, SNS_VQ_FAR_ADJ_GAINS, SNS_VQ_NEAR_ADJ_GAINS,
    SNS_VQ_REG_ADJ_GAINS, SNS_VQ_REG_LF_ADJ_GAINS,
};

/// Number of MPVQ codewords of shape A for the regular shapes.
const SZ_A_REGULAR: u32 = 2_390_004;
/// Number of MPVQ codewords of shape A for the near-outlier shape.
const SZ_A_NEAR: u32 = 15_158_272;

/// Quantized SNS parameters, ready for bitstream multiplexing.
pub(crate) struct SnsQuantized {
    /// Stage 1 low-frequency codebook index.
    pub(crate) ind_lf: usize,
    /// Stage 1 high-frequency codebook index.
    pub(crate) ind_hf: usize,
    /// Stage 2 shape index (`0` regular, `1` regular_lf, `2` near, `3` far).
    pub(crate) shape: usize,
    /// Stage 2 adjustment gain index.
    pub(crate) gain: usize,
    /// Stage 2 MPVQ index of set A.
    pub(crate) idx_a: u32,
    /// Leading sign of set A (`true` when negative).
    pub(crate) ls_a: bool,
    /// Stage 2 MPVQ index of set B.
    pub(crate) idx_b: u32,
    /// Leading sign of set B (`true` when negative).
    pub(crate) ls_b: bool,
    /// Synthesized quantized scale factors.
    pub(crate) scf_q: [f32; 16],
}

impl SnsQuantized {
    /// Joint stage 2 codeword (§3.3.7.3.4.2).
    pub(crate) fn joint_index(&self) -> u32 {
        match self.shape {
            0 => (2 * self.idx_b + u32::from(self.ls_b) + 2) * SZ_A_REGULAR + self.idx_a,
            1 => u32::from(self.gain_lsb()) * SZ_A_REGULAR + self.idx_a,
            2 => self.idx_a,
            _ => SZ_A_NEAR + u32::from(self.gain_lsb()) + 2 * self.idx_a,
        }
    }

    /// Low bit of the adjustment gain index, carried by the joint codeword.
    pub(crate) fn gain_lsb(&self) -> bool {
        self.gain & 1 != 0
    }

    /// Writes the SNS side information (§3.3.10.1, §3.3.13.3).
    pub(crate) fn write_side(
        &self,
        bits: &mut super::enc_bits::BitWriter<'_>,
    ) -> Result<(), Lc3Error> {
        bits.write_uint(self.ind_lf as u32, 5)?;
        bits.write_uint(self.ind_hf as u32, 5)?;
        let shape_msb = self.shape >> 1;
        bits.write_bit(shape_msb != 0)?;
        let gain_msb_bits = 1 + shape_msb as u32;
        bits.write_uint((self.gain >> (self.shape & 1)) as u32, gain_msb_bits)?;
        bits.write_bit(self.ls_a)?;
        let low_bits = 14 - gain_msb_bits;
        let joint = self.joint_index();
        bits.write_uint(joint & ((1u32 << low_bits) - 1), low_bits)?;
        bits.write_uint(joint >> low_bits, 12)?;
        Ok(())
    }
}

/// Estimates the 16 SNS scale factors from the band energies (§3.3.7.2).
pub(crate) fn analyse(eb: &[f32], nb: usize, sr: usize, dt: usize, attack: bool) -> [f32; 16] {
    // §3.3.7.2.1: pad to 64 bands by repeating the lowest bands.
    let mut eb64 = [0f32; 64];
    let n2 = 64 - nb;
    for i in 0..n2 {
        eb64[2 * i] = eb[i];
        eb64[2 * i + 1] = eb[i];
    }
    for i in 0..nb - n2 {
        eb64[2 * n2 + i] = eb[n2 + i];
    }

    // §3.3.7.2.2: smoothing.
    let mut es = [0f32; 64];
    for b in 0..64 {
        es[b] = if b == 0 {
            0.75 * eb64[0] + 0.25 * eb64[1]
        } else if b == 63 {
            0.25 * eb64[62] + 0.75 * eb64[63]
        } else {
            0.25 * eb64[b - 1] + 0.5 * eb64[b] + 0.25 * eb64[b + 1]
        };
    }

    // §3.3.7.2.3: pre-emphasis.
    const TILT: [f32; 5] = [14.0, 18.0, 22.0, 26.0, 30.0];
    let tilt = TILT[sr];
    let mut ep = [0f32; 64];
    for b in 0..64 {
        ep[b] = es[b] * 10f32.powf(b as f32 * tilt / 630.0);
    }

    // §3.3.7.2.4: noise floor.
    let mean: f32 = ep.iter().sum::<f32>() / 64.0;
    let floor = (mean * 1e-4).max(2f32.powi(-32));
    // §3.3.7.2.5: logarithm.
    let mut el = [0f32; 64];
    for b in 0..64 {
        el[b] = (1e-31 + ep[b].max(floor)).log2() * 0.5;
    }

    // §3.3.7.2.6: group four bands with the [1,2,3,3,2,1]/12 kernel.
    const W: [f32; 6] = [
        1.0 / 12.0,
        2.0 / 12.0,
        3.0 / 12.0,
        3.0 / 12.0,
        2.0 / 12.0,
        1.0 / 12.0,
    ];
    let mut e4 = [0f32; 16];
    for (b2, value) in e4.iter_mut().enumerate() {
        let mut sum = 0f32;
        if b2 == 0 {
            sum += W[0] * el[0];
            for (k, &weight) in W.iter().enumerate().skip(1) {
                sum += weight * el[k - 1];
            }
        } else if b2 == 15 {
            for (k, &weight) in W.iter().enumerate().take(5) {
                sum += weight * el[4 * b2 + k - 1];
            }
            sum += W[5] * el[63];
        } else {
            for (k, &weight) in W.iter().enumerate() {
                sum += weight * el[4 * b2 + k - 1];
            }
        }
        *value = sum;
    }

    // §3.3.7.2.7: mean removal and attack handling.
    let mean = e4.iter().sum::<f32>() / 16.0;
    let scf0: [f32; 16] = std::array::from_fn(|n| 0.85 * (e4[n] - mean));
    if !attack {
        return scf0;
    }

    let mut scf1 = [0f32; 16];
    scf1[0] = (scf0[0] + scf0[1] + scf0[2]) / 3.0;
    scf1[1] = (scf0[0] + scf0[1] + scf0[2] + scf0[3]) / 4.0;
    for n in 2..14 {
        scf1[n] = (scf0[n - 2] + scf0[n - 1] + scf0[n] + scf0[n + 1] + scf0[n + 2]) / 5.0;
    }
    scf1[14] = (scf0[12] + scf0[13] + scf0[14] + scf0[15]) / 4.0;
    scf1[15] = (scf0[13] + scf0[14] + scf0[15]) / 3.0;

    let mean = scf1.iter().sum::<f32>() / 16.0;
    let factor = if dt == 0 { 0.3 } else { 0.5 };
    std::array::from_fn(|n| factor * (scf1[n] - mean))
}

/// Orthonormal DCT-II used as the stage 2 analysis rotation.
pub(crate) fn dct16(x: &[f32; 16]) -> [f32; 16] {
    let mut out = [0f32; 16];
    for (k, value) in out.iter_mut().enumerate() {
        let mut sum = 0f32;
        for (n, &sample) in x.iter().enumerate() {
            sum += sample * (PI * k as f32 * (2.0 * n as f32 + 1.0) / 32.0).cos();
        }
        let scale = if k == 0 {
            (1.0f32 / 16.0).sqrt()
        } else {
            (2.0f32 / 16.0).sqrt()
        };
        *value = sum * scale;
    }
    out
}

/// Stage 2 gain sets indexed by shape.
fn gain_set(shape: usize) -> &'static [f32] {
    match shape {
        0 => &SNS_VQ_REG_ADJ_GAINS,
        1 => &SNS_VQ_REG_LF_ADJ_GAINS,
        2 => &SNS_VQ_NEAR_ADJ_GAINS,
        _ => &SNS_VQ_FAR_ADJ_GAINS,
    }
}

/// Adds positive unit pulses up to `target` over `positions`.
fn add_pulses(
    y: &mut [i32; 16],
    positions: usize,
    target: i32,
    x: &[f32; 16],
    corr: &mut f32,
    energy: &mut f32,
) {
    let mut placed: i32 = y[..positions].iter().sum();
    while placed < target {
        let mut best_position = 0usize;
        let mut best_corr = 0f32;
        let mut best_energy = 1f32;
        for (n, &sample) in x.iter().enumerate().take(positions) {
            let candidate_corr = *corr + sample.abs();
            let candidate_energy = *energy + 2.0 * y[n] as f32 + 1.0;
            if candidate_corr * candidate_corr * best_energy
                > best_corr * best_corr * candidate_energy
            {
                best_position = n;
                best_corr = candidate_corr;
                best_energy = candidate_energy;
            }
        }
        y[best_position] += 1;
        *corr += x[best_position].abs();
        *energy += 2.0 * y[best_position] as f32 - 1.0;
        placed += 1;
    }
}

/// Searches the four shape candidates (§3.3.7.3.3.5).
fn search_shapes(x: &[f32; 16]) -> [[i32; 16]; 4] {
    let l1: f32 = x.iter().map(|v| v.abs()).sum();
    let proj = if l1 > 0.0 { 5.0 / l1 } else { 0.0 };
    let mut y = [0i32; 16];
    for (value, &sample) in y.iter_mut().zip(x) {
        *value = (sample.abs() * proj) as i32;
    }
    let mut corr: f32 = y
        .iter()
        .zip(x)
        .map(|(&pulse, &sample)| pulse as f32 * sample.abs())
        .sum();
    let mut energy: f32 = y.iter().map(|&pulse| (pulse * pulse) as f32).sum();

    // Shape 3 (far, K=6) and then shape 2 (near, K=8) over all 16 positions.
    add_pulses(&mut y, 16, 6, x, &mut corr, &mut energy);
    let y3 = y;
    add_pulses(&mut y, 16, 8, x, &mut corr, &mut energy);
    let y2 = y;

    // Shape 1 (regular_lf, K=10) restricted to set A.
    let mut y1 = [0i32; 16];
    y1[..10].copy_from_slice(&y2[..10]);
    let corr1: f32 = y1[..10]
        .iter()
        .zip(&x[..10])
        .map(|(&pulse, &sample)| pulse as f32 * sample.abs())
        .sum();
    let energy1: f32 = y1[..10].iter().map(|&pulse| (pulse * pulse) as f32).sum();
    let mut corr = corr1;
    let mut energy = energy1;
    add_pulses(&mut y1, 10, 10, x, &mut corr, &mut energy);

    // Shape 0 (regular): one pulse in set B at the largest magnitude.
    let mut y0 = y1;
    let mut best = 10usize;
    for n in 11..16 {
        if x[n].abs() > x[best].abs() {
            best = n;
        }
    }
    y0[best] = 1;

    // Apply the target signs to the non-zero entries.
    let mut shapes = [y0, y1, y2, y3];
    for shape in &mut shapes {
        for (value, &sample) in shape.iter_mut().zip(x) {
            if *value != 0 {
                *value = if sample < 0.0 { -*value } else { *value };
            }
        }
    }
    shapes
}

/// Initializes the MPVQ enumeration.
pub(crate) fn mpvq_enum(y: &[i32], k: i32) -> (u32, bool) {
    const SENTINEL: i32 = -1;
    let dim = y.len();
    let mut index = 0u32;
    let mut next_sign = SENTINEL;
    let mut k_acc = 0i32;
    let mut n = 0usize;
    let mut row = SNS_MPVQ_OFFSETS[n][0];
    for pos in (0..dim).rev() {
        let value = y[pos];
        if next_sign != SENTINEL && value != 0 {
            index = 2 * index + next_sign as u32;
        }
        if value < 0 {
            next_sign = 1;
        } else if value > 0 {
            next_sign = 0;
        }
        index += row;
        k_acc += value.abs();
        if pos != 0 {
            n += 1;
        }
        row = SNS_MPVQ_OFFSETS[n][k_acc as usize];
    }
    debug_assert_eq!(k_acc, k);
    (index, next_sign == 1)
}

/// Quantizes the 16 scale factors (§3.3.7.3).
pub(crate) fn quantize(scf: &[f32; 16]) -> Result<SnsQuantized, Lc3Error> {
    // Stage 1: split VQ.
    let mut ind_lf = 0usize;
    let mut best = f32::INFINITY;
    for (i, codebook) in SNS_LFCB.iter().enumerate() {
        let distortion: f32 = codebook
            .iter()
            .zip(&scf[..8])
            .map(|(&c, &s)| (s - c) * (s - c))
            .sum();
        if distortion < best {
            best = distortion;
            ind_lf = i;
        }
    }
    let mut ind_hf = 0usize;
    best = f32::INFINITY;
    for (i, codebook) in SNS_HFCB.iter().enumerate() {
        let distortion: f32 = codebook
            .iter()
            .zip(&scf[8..])
            .map(|(&c, &s)| (s - c) * (s - c))
            .sum();
        if distortion < best {
            best = distortion;
            ind_hf = i;
        }
    }
    let mut st1 = [0f32; 16];
    st1[..8].copy_from_slice(&SNS_LFCB[ind_lf]);
    st1[8..].copy_from_slice(&SNS_HFCB[ind_hf]);
    let residual: [f32; 16] = std::array::from_fn(|n| scf[n] - st1[n]);

    // Stage 2: rotated shape search and gain selection.
    let target = dct16(&residual);
    let shapes = search_shapes(&target);
    let mut candidates = [[0f32; 16]; 4];
    for (&y, out) in shapes.iter().zip(&mut candidates) {
        let norm = y.iter().map(|&v| (v * v) as f32).sum::<f32>().sqrt();
        for (value, pulse) in out.iter_mut().zip(y) {
            *value = if norm > 0.0 { pulse as f32 / norm } else { 0.0 };
        }
    }

    let mut best_shape = 0usize;
    let mut best_gain = 0usize;
    let mut best_distortion = f32::INFINITY;
    for (shape, candidate) in candidates.iter().enumerate() {
        for (gain, &value) in gain_set(shape).iter().enumerate() {
            let distortion: f32 = target
                .iter()
                .zip(candidate)
                .map(|(&t, &c)| {
                    let d = t - value * c;
                    d * d
                })
                .sum();
            if distortion < best_distortion {
                best_distortion = distortion;
                best_shape = shape;
                best_gain = gain;
            }
        }
    }

    let selected = shapes[best_shape];
    let (idx_a, ls_a, idx_b, ls_b) = match best_shape {
        0 => {
            let (a, sa) = mpvq_enum(&selected[..10], 10);
            let (b, sb) = mpvq_enum(&selected[10..], 1);
            (a, sa, b, sb)
        }
        1 => {
            let (a, sa) = mpvq_enum(&selected[..10], 10);
            (a, sa, 0, false)
        }
        2 => {
            let (a, sa) = mpvq_enum(&selected, 8);
            (a, sa, 0, false)
        }
        _ => {
            let (a, sa) = mpvq_enum(&selected, 6);
            (a, sa, 0, false)
        }
    };

    // Reconstruct the quantized scale factors (§3.3.7.3.4.3).
    let norm = selected.iter().map(|&v| (v * v) as f32).sum::<f32>().sqrt();
    let mut unit = [0f32; 16];
    for (value, &pulse) in unit.iter_mut().zip(&selected) {
        *value = pulse as f32 / norm;
    }
    let rotation = idct16(&unit);
    let gain = gain_set(best_shape)[best_gain];
    let scf_q: [f32; 16] = std::array::from_fn(|n| st1[n] + gain * rotation[n]);

    Ok(SnsQuantized {
        ind_lf,
        ind_hf,
        shape: best_shape,
        gain: best_gain,
        idx_a,
        ls_a,
        idx_b,
        ls_b,
        scf_q,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lc3::sns::deenum_mpvq;

    #[test]
    fn mpvq_round_trip() {
        let cases: [(usize, i32); 4] = [(10, 10), (6, 1), (16, 8), (16, 6)];
        let mut state = 12345u32;
        for (dim, k) in cases {
            for _ in 0..50 {
                let mut y = vec![0i32; dim];
                for _ in 0..k {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    let pos = (state >> 8) as usize % dim;
                    y[pos] += 1;
                }
                for value in &mut y {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    if state & 1 == 0 && *value != 0 {
                        *value = -*value;
                    }
                }
                let (index, sign) = mpvq_enum(&y, k);
                let decoded = deenum_mpvq(index, sign, k, dim).unwrap();
                assert_eq!(decoded, y, "dim={dim} k={k} index={index}");
            }
        }
    }

    #[test]
    fn dct_is_orthonormal() {
        let x: [f32; 16] = std::array::from_fn(|n| (n as f32 * 0.37).sin());
        let back = idct16(&dct16(&x));
        let worst = x
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-4, "orthonormality deviation {worst}");
    }

    #[test]
    fn quantization_reconstructs_a_smooth_profile() {
        let scf: [f32; 16] = std::array::from_fn(|n| 1.5 - 0.2 * n as f32);
        let quantized = quantize(&scf).unwrap();
        let worst = scf
            .iter()
            .zip(&quantized.scf_q)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 0.6, "SNS quantization deviation {worst}");
    }
}
