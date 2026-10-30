// SPDX-License-Identifier: Apache-2.0
//! LC3 Spectral Noise Shaping (SNS) synthesis (Bluetooth LC3 v1.0.1 §3.4.7).

use std::f32::consts::PI;

use super::Lc3Error;
use super::bits::BitReader;
use super::spec::band_limits;
use super::tables::{
    SNS_HFCB, SNS_LFCB, SNS_MPVQ_OFFSETS, SNS_VQ_FAR_ADJ_GAINS, SNS_VQ_NEAR_ADJ_GAINS,
    SNS_VQ_REG_ADJ_GAINS, SNS_VQ_REG_LF_ADJ_GAINS,
};

const SQRT2: f32 = std::f32::consts::SQRT_2;

/// SNS synthesis state for one channel.
#[derive(Debug, Clone)]
pub(crate) struct SnsSynthesis {
    dt: usize,
    sr: usize,
    ind_lf: usize,
    ind_hf: usize,
    shape: usize,
    gain: usize,
    idx_a: u32,
    ls_a: bool,
    idx_b: u32,
    ls_b: bool,
}

impl SnsSynthesis {
    pub(crate) fn new(dt: usize, sr: usize) -> Self {
        Self {
            dt,
            sr,
            ind_lf: 0,
            ind_hf: 0,
            shape: 0,
            gain: 0,
            idx_a: 0,
            ls_a: false,
            idx_b: 0,
            ls_b: false,
        }
    }

    /// Reads the SNS side information.
    pub(crate) fn load(&mut self, bits: &mut BitReader<'_>) -> Result<(), Lc3Error> {
        self.ind_lf = bits.read_uint(5)? as usize;
        self.ind_hf = bits.read_uint(5)? as usize;
        let shape_msb = bits.read_bit()?;
        let gain_msb_bits = 1 + u32::from(shape_msb);
        self.gain = bits.read_uint(gain_msb_bits)? as usize;
        self.ls_a = bits.read_bit()?;
        let low_bits = 14 - gain_msb_bits;
        let mut index_joint = bits.read_uint(low_bits)?;
        index_joint |= bits.read_uint(12)? << low_bits;

        if !shape_msb {
            const SZ_A: u32 = 2_390_004;
            if index_joint >= SZ_A * 14 {
                return Err(Lc3Error::Bitstream("invalid SNS joint index"));
            }
            self.idx_a = index_joint % SZ_A;
            index_joint /= SZ_A;
            if index_joint >= 2 {
                self.shape = 0;
                self.idx_b = (index_joint - 2) / 2;
                self.ls_b = (index_joint - 2) % 2 != 0;
            } else {
                self.shape = 1;
                self.gain = (self.gain << 1) + index_joint as usize;
            }
        } else {
            const SZ_A: u32 = 15_158_272;
            const SZ_B: u32 = 1_549_824;
            if index_joint >= SZ_A + SZ_B {
                return Err(Lc3Error::Bitstream("invalid SNS joint index"));
            }
            if index_joint < SZ_A {
                self.shape = 2;
                self.idx_a = index_joint;
            } else {
                self.shape = 3;
                index_joint -= SZ_A;
                self.gain = (self.gain << 1) + (index_joint % 2) as usize;
                self.idx_a = index_joint / 2;
            }
        }
        Ok(())
    }

    /// Applies the SNS shaping to `x`.
    pub(crate) fn run(&self, x: &mut [f32]) -> Result<(), Lc3Error> {
        let scf = self.unquantize()?;
        spectral_shaping(self.dt, self.sr, &scf, true, x);
        Ok(())
    }

    fn unquantize(&self) -> Result<[f32; 16], Lc3Error> {
        let mut y = [0i32; 16];
        match self.shape {
            0 => {
                y[..10].copy_from_slice(&deenum_mpvq(self.idx_a, self.ls_a, 10, 10)?);
                y[10..].copy_from_slice(&deenum_mpvq(self.idx_b, self.ls_b, 1, 6)?);
            }
            1 => {
                y[..10].copy_from_slice(&deenum_mpvq(self.idx_a, self.ls_a, 10, 10)?);
            }
            2 => {
                y.copy_from_slice(&deenum_mpvq(self.idx_a, self.ls_a, 8, 16)?);
            }
            3 => {
                y.copy_from_slice(&deenum_mpvq(self.idx_a, self.ls_a, 6, 16)?);
            }
            _ => return Err(Lc3Error::Bitstream("invalid SNS shape")),
        }
        let norm = (y
            .iter()
            .map(|&v| f32::from(v as i16) * f32::from(v as i16))
            .sum::<f32>())
        .sqrt();
        if norm == 0.0 {
            return Err(Lc3Error::Bitstream("zero SNS shape"));
        }
        let mut shape = [0f32; 16];
        for (dst, &value) in shape.iter_mut().zip(&y) {
            *dst = value as f32 / norm;
        }
        let idct = idct16(&shape);
        let gains: [&[f32]; 4] = [
            &SNS_VQ_REG_ADJ_GAINS,
            &SNS_VQ_REG_LF_ADJ_GAINS,
            &SNS_VQ_NEAR_ADJ_GAINS,
            &SNS_VQ_FAR_ADJ_GAINS,
        ];
        let gain = *gains[self.shape]
            .get(self.gain)
            .ok_or(Lc3Error::Bitstream("invalid SNS gain"))?;
        let mut scf = [0f32; 16];
        for i in 0..8 {
            scf[i] = SNS_LFCB[self.ind_lf][i] + gain * idct[i];
            scf[8 + i] = SNS_HFCB[self.ind_hf][i] + gain * idct[8 + i];
        }
        Ok(scf)
    }
}

/// Applies the interpolated SNS scale factors to `x` (LC3 v1.0.1 §3.3.7.5
/// and §3.4.7.3).
///
/// In inverse (synthesis) mode the factors are negated before interpolation so
/// the shaping gain is always `2^-scf_i`.
pub(crate) fn spectral_shaping(
    dt: usize,
    sr: usize,
    scf: &[f32; 16],
    inverse: bool,
    x: &mut [f32],
) {
    let sign = if inverse { -1.0f32 } else { 1.0f32 };
    let scf_q: [f32; 16] = std::array::from_fn(|i| sign * scf[i]);

    let mut scf_i = [0f32; 64];
    scf_i[0] = scf_q[0];
    scf_i[1] = scf_q[0];
    for i in 0..15 {
        let delta = scf_q[i + 1] - scf_q[i];
        scf_i[2 + 4 * i] = scf_q[i] + delta * 0.125;
        scf_i[3 + 4 * i] = scf_q[i] + delta * 0.375;
        scf_i[4 + 4 * i] = scf_q[i] + delta * 0.625;
        scf_i[5 + 4 * i] = scf_q[i] + delta * 0.875;
    }
    let delta = scf_q[15] - scf_q[14];
    scf_i[62] = scf_q[15] + delta * 0.125;
    scf_i[63] = scf_q[15] + delta * 0.375;

    let i = band_limits(dt, sr);
    let nb = i.len() - 1;
    if nb < 32 {
        let n4 = 32 % nb;
        let n2 = nb - n4;
        for band in 0..n4 {
            scf_i[band] =
                (scf_i[4 * band] + scf_i[4 * band + 1] + scf_i[4 * band + 2] + scf_i[4 * band + 3])
                    * 0.25;
        }
        for band in n4..n4 + n2 {
            scf_i[band] = (scf_i[2 * (n4 + band)] + scf_i[2 * (n4 + band) + 1]) * 0.5;
        }
        let from = 4 * n4 + 2 * n2;
        let count = nb - n4 - n2;
        scf_i.copy_within(from..from + count, n4 + n2);
    } else if nb < 64 {
        let n2 = 64 - nb;
        for band in 0..n2 {
            scf_i[band] = (scf_i[2 * band] + scf_i[2 * band + 1]) * 0.5;
        }
        let from = 2 * n2;
        let count = nb - n2;
        scf_i.copy_within(from..from + count, n2);
    }

    for band in 0..nb {
        let start = i[band] as usize;
        let end = i[band + 1] as usize;
        let gain = 2f32.powf(-scf_i[band]);
        for value in &mut x[start..end] {
            *value *= gain;
        }
    }
}

/// MPVQ de-enumeration (Bluetooth LC3 v1.0.1 §3.3.7.3.3.8).
pub(crate) fn deenum_mpvq(
    mut index: u32,
    mut ls: bool,
    mut pulses: i32,
    n: usize,
) -> Result<Vec<i32>, Lc3Error> {
    let mut y = vec![0i32; n];
    for (pos, i) in (0..n).rev().enumerate() {
        let mut yi = 0i32;
        if index > 0 {
            while pulses - yi >= 0
                && (pulses - yi) < 11
                && index < SNS_MPVQ_OFFSETS[i][(pulses - yi) as usize]
            {
                yi += 1;
                if yi > pulses {
                    return Err(Lc3Error::Bitstream("invalid MPVQ index"));
                }
            }
            if pulses - yi < 0 || (pulses - yi) >= 11 {
                return Err(Lc3Error::Bitstream("invalid MPVQ pulse count"));
            }
            index -= SNS_MPVQ_OFFSETS[i][(pulses - yi) as usize];
        } else {
            yi = pulses;
        }
        y[pos] = if ls { -yi } else { yi };
        pulses -= yi;
        if pulses <= 0 {
            break;
        }
        if yi > 0 {
            ls = index & 1 != 0;
            index >>= 1;
        }
    }
    Ok(y)
}

/// 16-point orthonormal DCT-III, matching `scipy.fftpack.idct(..., norm='ortho')`.
pub(crate) fn idct16(x: &[f32; 16]) -> [f32; 16] {
    let mut out = [0f32; 16];
    let scale = (2.0f32 / 16.0).sqrt();
    for (n, value) in out.iter_mut().enumerate() {
        let mut sum = x[0] / SQRT2;
        for (k, &coefficient) in x.iter().enumerate().skip(1) {
            sum += coefficient * (PI * (n as f32 + 0.5) * k as f32 / 16.0).cos();
        }
        *value = sum * scale;
    }
    out
}
