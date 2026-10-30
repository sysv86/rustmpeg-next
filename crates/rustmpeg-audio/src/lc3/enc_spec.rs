// SPDX-License-Identifier: Apache-2.0
//! LC3 spectral quantization and encoding (Bluetooth LC3 v1.0.1 §3.3.10-3.3.12
//! and the spectral part of §3.3.13.3-3.3.13.5).
//!
//! [`SpecEncoder::analyse`] picks the global gain for a frame, quantizes the
//! TNS-filtered spectrum, estimates the arithmetic-coded size, truncates the
//! spectrum to the bit budget, derives the residual bits and the noise factor.
//! The resulting [`SpecFrame`] then writes the side information, the noise
//! factor, the arithmetic spectrum (interleaved with the backward sign and
//! least-significant-bit planes) and the residual bits.
//!
//! All spectral values are in the 16-bit PCM scale, as produced by the MDCT
//! analysis and shaped by the TNS and SNS stages. Bit estimates are computed in
//! the same `1/2048` bit units as the normative `ac_spec_bits` table.

use super::Lc3Error;
use super::bits::ilog2;
use super::enc_bits::{AcEncoder, BitWriter};
use super::enc_tables::ENC_SPEC_BITS;
use super::spec::band_limits;
use super::tables::{AC_SPEC_CUMFREQ, AC_SPEC_FREQ, AC_SPEC_LOOKUP};

/// Scale of the `ac_spec_bits` estimates: one bit is `BIT_SCALE` units.
const BIT_SCALE: i64 = 2048;

/// Longest LC3 frame payload (400 bytes, §3.2.5).
const MAX_NBYTES: usize = 400;

/// Number of bits available for the last-non-zero index and the arithmetic
/// coder, excluding the arithmetic payload itself (§3.3.10.1, Equation 107).
const fn ari_overhead(nbits_nq: i32, nbits: i32) -> i32 {
    if nbits <= 1280 {
        nbits_nq + 3
    } else if nbits <= 2560 {
        nbits_nq + 4
    } else {
        nbits_nq + 5
    }
}

/// Quantized-gain offset derived from the frame size (§3.3.10.2, Equation 110).
fn gain_offset(fs_idx: i32, nbits: i32) -> i32 {
    let g_off = (nbits / (10 * (1 + fs_idx))).min(115);
    -(g_off + 105 + 5 * (1 + fs_idx))
}

/// Unquantized gain `10^(g_int / 28)` (§3.3.10.3, Equation 114).
fn unquantize_gain(g_int: i32) -> f32 {
    10f32.powf(g_int as f32 / 28.0)
}

/// Rounds to the nearest integer, halves away from zero, as `nint` in the
/// specification (used for Equation 109).
fn nint(value: f32) -> i32 {
    if value >= 0.0 {
        (value + 0.5).floor() as i32
    } else {
        (value - 0.5).ceil() as i32
    }
}

/// Result of the bit-consumption estimate of §3.3.10.4.
struct BitEstimate {
    /// Last non-zero 2-tuple index that still fits the budget (`lastnz_trunc`).
    lastnz: usize,
    /// Estimated bits for the whole (untruncated) spectrum, including the LSB
    /// planes of `lsbMode` (`nbits_est`).
    nbits_est: i32,
    /// Estimated bits up to `lastnz` (`nbits_trunc`).
    nbits_trunc: i32,
}

/// Persistent per-channel encoder state (Equation 108 history).
pub(crate) struct SpecEncoder {
    /// Smoothed offset applied to the spectrum bit budget.
    nbits_offset: f32,
    /// Bit budget of the previous frame (`nbits_spec`).
    prev_nbits_spec: i32,
    /// Estimate of the previous frame before any requantization (`nbits_est`).
    prev_nbits_est: i32,
    /// Whether the previous frame reset the offset (`reset_offset`).
    reset_offset: bool,
}

impl SpecEncoder {
    /// Creates the encoder state for a new channel.
    pub(crate) fn new() -> Self {
        Self {
            nbits_offset: 0.0,
            prev_nbits_spec: 0,
            prev_nbits_est: 0,
            reset_offset: false,
        }
    }

    /// Quantizes `xf` (TNS-filtered, SNS-shaped spectrum on the 16-bit PCM
    /// scale, exactly `NE` coefficients) for a frame of `nbytes` bytes.
    ///
    /// `nbits_overhead` is `nbits_bw + nbits_tns + nbits_ltpf + 38 + 8 + 3`,
    /// that is every side-information field except the arithmetic-coder margin
    /// of Equation 107. `bw_stop` is the last band limit of the detected
    /// bandwidth, used by the noise-level estimator of §3.3.12.
    pub(crate) fn analyse(
        &mut self,
        xf: &[f32],
        dt: usize,
        sr: usize,
        nbytes: usize,
        nbits_overhead: i32,
        bw_stop: usize,
    ) -> Result<SpecFrame, Lc3Error> {
        if dt > 1 || sr > 4 {
            return Err(Lc3Error::Encode(
                "invalid LC3 frame duration or sample rate",
            ));
        }
        if nbytes == 0 || nbytes > MAX_NBYTES {
            return Err(Lc3Error::Encode("invalid LC3 frame size"));
        }
        let ne = *band_limits(dt, sr)
            .last()
            .ok_or(Lc3Error::Encode("missing LC3 band table"))? as usize;
        if ne < 4 || ne % 4 != 0 {
            return Err(Lc3Error::Encode("invalid LC3 band table"));
        }
        if xf.len() != ne {
            return Err(Lc3Error::Encode("spectrum length does not match the frame"));
        }

        let fs_idx = sr as i32;
        let nbits = (nbytes * 8) as i32;

        // §3.3.10.1: bits available for the quantized spectrum.
        let nbits_nq = ilog2(ne as u32 / 2 - 1) + 1;
        let nbits_ari = ari_overhead(nbits_nq, nbits);
        let nbits_spec = nbits - nbits_overhead - nbits_ari;
        if nbits_spec < 0 {
            return Err(Lc3Error::Encode("LC3 spectrum budget is negative"));
        }

        // §3.3.10.2, Equation 108: smoothed offset from the previous frame.
        // The previous values are the raw budget and the estimate taken before
        // any requantization, as required by the note below Equation 109.
        self.nbits_offset = if self.reset_offset {
            0.0
        } else {
            let target = (self.nbits_offset + (self.prev_nbits_spec - self.prev_nbits_est) as f32)
                .clamp(-40.0, 40.0);
            0.8 * self.nbits_offset + 0.2 * target
        };
        // Equation 109: the offset steers the gain search only. Truncation
        // and gain adjustment use the real frame budget below.
        let budget = nint(nbits_spec as f32 + self.nbits_offset);

        let gg_off = gain_offset(fs_idx, nbits);

        // Equation 111: energy in dB of blocks of four coefficients.
        let energies: Vec<f32> = xf
            .as_chunks::<4>()
            .0
            .iter()
            .map(|block| {
                let sum: f32 = block.iter().map(|value| value * value).sum();
                10.0 * (2f32.powi(-31) + sum).log10()
            })
            .collect();
        let xfmax = xf.iter().fold(0f32, |acc, value| acc.max(value.abs()));

        // §3.3.10.2: bisection on the quantized gain index. `gg_ind` is in the
        // transmitted domain, so every comparison uses `gg_ind + gg_off`.
        let target = budget as f32 * 1.4 * 28.0 / 20.0;
        let mut fac = 256i32;
        let mut gg_ind = 255i32;
        for _ in 0..8 {
            fac >>= 1;
            gg_ind -= fac;
            let mut tmp = 0f32;
            let mut iszero = true;
            for energy in energies.iter().rev() {
                let energy = energy * 28.0 / 20.0;
                let gain = (gg_ind + gg_off) as f32;
                if energy < gain {
                    if !iszero {
                        tmp += 2.7 * 28.0 / 20.0;
                    }
                } else {
                    if gain < energy - 43.0 * 28.0 / 20.0 {
                        tmp += 2.0 * energy - 2.0 * gain - 36.0 * 28.0 / 20.0;
                    } else {
                        tmp += energy - gain + 7.0 * 28.0 / 20.0;
                    }
                    iszero = false;
                }
            }
            if tmp > target && !iszero {
                gg_ind += fac;
            }
        }

        // Equation 112: lower bound that keeps the spectrum in range.
        let gg_min = if xfmax > 0.0 {
            (28.0 * (1e-31f32 + xfmax / (32768.0 - 0.375)).log10()).ceil() as i32 - gg_off
        } else {
            0
        };
        self.reset_offset = gg_ind < gg_min || xfmax == 0.0;
        if self.reset_offset {
            gg_ind = gg_min;
        }
        // Defensive clamp: `gg_min` can leave the 8-bit index range for
        // degenerate (near-silent or out-of-range) input. Clamping up to zero
        // never makes the step size larger, so it cannot overflow `xq`; the
        // upper clamp only affects spectra far outside the 16-bit scale, where
        // the quantization below saturates anyway.
        gg_ind = gg_ind.clamp(0, 255);

        // §3.3.10.3: quantization.
        let mut gg = unquantize_gain(gg_ind + gg_off);
        let mut xq = Vec::with_capacity(ne);
        quantize(xf, gg, &mut xq);

        // §3.3.10.4: bit consumption. The smoothed offset only steers the
        // gain search; truncation and the gain adjustment compare against the
        // real frame budget, as in the normative encoder.
        let rate_flag = if nbits > 160 + fs_idx * 160 { 512 } else { 0 };
        let mode_flag = nbits >= 480 + fs_idx * 160;
        let mut estimate = estimate_bits(&xq, ne, rate_flag, mode_flag, nbits_spec);
        let mut lsb_mode = mode_flag && estimate.nbits_est > nbits_spec;
        let pre_requantization_est = estimate.nbits_est;

        // §3.3.10.6: at most one global gain adjustment and requantization.
        let delta = delta_threshold(estimate.nbits_est, fs_idx);
        let delta2 = delta + 2;
        let adjust = (gg_ind < 255 && estimate.nbits_est > nbits_spec)
            || (gg_ind > 0 && estimate.nbits_est < nbits_spec - delta2);
        if adjust {
            if estimate.nbits_est < nbits_spec - delta2 {
                gg_ind -= 1;
            } else if gg_ind == 254 || estimate.nbits_est < nbits_spec + delta {
                gg_ind += 1;
            } else {
                gg_ind += 2;
            }
            gg_ind = gg_ind.max(gg_min).clamp(0, 255);
            gg = unquantize_gain(gg_ind + gg_off);
            quantize(xf, gg, &mut xq);
            estimate = estimate_bits(&xq, ne, rate_flag, mode_flag, nbits_spec);
            lsb_mode = mode_flag && estimate.nbits_est > nbits_spec;
        }

        // §3.3.10.5: truncate the spectrum to the last tuple that fits.
        for value in xq.iter_mut().skip(estimate.lastnz) {
            *value = 0;
        }

        // §3.3.11: residual bits, only when the LSBs are not coded separately.
        let mut res_bits = Vec::new();
        if !lsb_mode {
            let max = (nbits_spec - estimate.nbits_trunc + 4).max(0) as usize;
            for (index, &value) in xq.iter().enumerate() {
                if res_bits.len() >= max {
                    break;
                }
                if value != 0 {
                    res_bits.push(xf[index] >= value as f32 * gg);
                }
            }
        }

        // §3.3.12: noise factor.
        let noise_factor = estimate_noise(xf, &xq, ne, dt, bw_stop, gg);

        self.prev_nbits_spec = nbits_spec;
        self.prev_nbits_est = pre_requantization_est;

        Ok(SpecFrame {
            xq,
            lastnz: estimate.lastnz,
            lsb_mode,
            gg_ind: gg_ind as u8,
            #[cfg(test)]
            gg,
            noise_factor,
            res_bits,
            lsbs: Vec::new(),
            nbits_nq: nbits_nq as u32,
            #[cfg(test)]
            nbits_spec,
            #[cfg(test)]
            nbits_trunc: estimate.nbits_trunc,
            ne,
            rate_flag,
        })
    }
}

/// Quantizes the spectrum with Equation 115, saturating to the 16-bit range.
fn quantize(xf: &[f32], gg: f32, out: &mut Vec<i32>) {
    out.clear();
    for &value in xf {
        let scaled = value / gg;
        let quantized = if scaled >= 0.0 {
            (scaled + 0.375).floor()
        } else {
            (scaled - 0.375).ceil()
        };
        // Hostile input (NaN, infinities, huge magnitudes) saturates here.
        out.push((quantized as i32).clamp(-32768, 32767));
    }
}

/// Estimates the arithmetic-coded size of `xq` (§3.3.10.4).
fn estimate_bits(
    xq: &[i32],
    ne: usize,
    rate_flag: i32,
    mode_flag: bool,
    budget: i32,
) -> BitEstimate {
    let mut lastnz = ne;
    while lastnz > 2 && xq[lastnz - 1] == 0 && xq[lastnz - 2] == 0 {
        lastnz -= 2;
    }

    let budget_scaled = i64::from(budget) * BIT_SCALE;
    let mut nbits_est = 0i64;
    let mut nbits_trunc = 0i64;
    let mut nbits_lsb = 0i64;
    let mut lastnz_trunc = 2usize;
    let mut c = 0usize;

    let mut n = 0usize;
    while n < lastnz {
        let mut t = c + rate_flag as usize;
        if n > ne / 2 {
            t += 256;
        }

        let mut a = i64::from(xq[n].unsigned_abs());
        let mut b = i64::from(xq[n + 1].unsigned_abs());
        let mut lev = 0usize;
        while a.max(b) >= 4 {
            let pki = usize::from(AC_SPEC_LOOKUP[t + lev.min(3) * 1024]);
            nbits_est += i64::from(ENC_SPEC_BITS[pki][16]);
            if lev == 0 && mode_flag {
                nbits_lsb += 2;
            } else {
                nbits_est += 2 * BIT_SCALE;
            }
            a >>= 1;
            b >>= 1;
            lev = (lev + 1).min(3);
        }

        let pki = usize::from(AC_SPEC_LOOKUP[t + lev.min(3) * 1024]);
        let sym = (a + 4 * b) as usize;
        nbits_est += i64::from(ENC_SPEC_BITS[pki][sym]);

        // Signs are estimated from the original magnitudes; in LSB mode the
        // discarded first plane carries the sign of a magnitude of one.
        let a_lsb = i64::from(xq[n].unsigned_abs());
        let b_lsb = i64::from(xq[n + 1].unsigned_abs());
        nbits_est += (a_lsb.min(1) + b_lsb.min(1)) * BIT_SCALE;
        if lev > 0 && mode_flag {
            if a_lsb >> 1 == 0 && xq[n] != 0 {
                nbits_lsb += 1;
            }
            if b_lsb >> 1 == 0 && xq[n + 1] != 0 {
                nbits_lsb += 1;
            }
        }

        if (xq[n] != 0 || xq[n + 1] != 0) && nbits_est <= budget_scaled {
            lastnz_trunc = n + 2;
            nbits_trunc = nbits_est;
        }

        let state = if lev <= 1 {
            1 + (a + b) * (lev as i64 + 1)
        } else {
            12 + lev as i64
        };
        c = (c & 15) * 16 + state as usize;
        n += 2;
    }

    BitEstimate {
        lastnz: lastnz_trunc,
        nbits_est: ((nbits_est + BIT_SCALE - 1) / BIT_SCALE) as i32 + nbits_lsb as i32,
        nbits_trunc: ((nbits_trunc + BIT_SCALE - 1) / BIT_SCALE) as i32,
    }
}

/// Gain-adjustment threshold of §3.3.10.6.
fn delta_threshold(nbits_est: i32, fs_idx: i32) -> i32 {
    const T1: [i32; 5] = [80, 230, 380, 530, 680];
    const T2: [i32; 5] = [500, 1025, 1550, 2075, 2600];
    const T3: [i32; 5] = [850, 1700, 2550, 3400, 4250];

    let index = fs_idx.clamp(0, 4) as usize;
    let (t1, t2, t3) = (T1[index], T2[index], T3[index]);
    if nbits_est < t1 {
        (nbits_est + 48) / 16
    } else if nbits_est < t2 {
        let tmp1 = t1 / 16 + 3;
        let tmp2 = t2 / 48;
        (nbits_est - t1) * (tmp2 - tmp1) / (t2 - t1) + tmp1
    } else if nbits_est < t3 {
        nbits_est / 48
    } else {
        t3 / 48
    }
}

/// Estimates the quantized noise level (§3.3.12).
fn estimate_noise(xf: &[f32], xq: &[i32], ne: usize, dt: usize, bw_stop: usize, gg: f32) -> u8 {
    let nf_start = if dt == 1 { 24 } else { 18 };
    let nf_width = if dt == 1 { 3 } else { 2 };
    let stop = bw_stop.min(ne);

    let mut sum = 0f32;
    let mut count = 0usize;
    for (k, &value) in xf.iter().enumerate().take(stop).skip(nf_start) {
        let low = k.saturating_sub(nf_width);
        let high = (k + nf_width).min(stop - 1);
        if xq[low..=high].iter().all(|&sample| sample == 0) {
            sum += value.abs() / gg;
            count += 1;
        }
    }
    if count == 0 {
        7
    } else {
        let level = 8.0 - 16.0 * sum / count as f32;
        (level.round() as i32).clamp(0, 7) as u8
    }
}

/// Quantized spectrum and side information of one frame.
pub(crate) struct SpecFrame {
    /// Quantized spectrum, zero above `lastnz`.
    pub(crate) xq: Vec<i32>,
    /// Last non-zero 2-tuple index (even, at least two).
    pub(crate) lastnz: usize,
    /// Whether the first LSB plane is coded as residual data.
    pub(crate) lsb_mode: bool,
    /// Quantized global gain index.
    pub(crate) gg_ind: u8,
    /// Unquantized global gain (used by the round-trip tests).
    #[cfg(test)]
    pub(crate) gg: f32,
    /// Three-bit noise factor.
    pub(crate) noise_factor: u8,
    /// Residual bits of §3.3.11, used when `!lsb_mode`.
    pub(crate) res_bits: Vec<bool>,
    /// LSB and sign bits of §3.3.13.4.2, used when `lsb_mode`.
    pub(crate) lsbs: Vec<bool>,
    /// `ceil(log2(NE / 2))`, the width of the last-non-zero field.
    pub(crate) nbits_nq: u32,
    /// Spectrum bit budget (`nbits_spec`), excluding side information.
    #[cfg(test)]
    pub(crate) nbits_spec: i32,
    /// Estimate up to `lastnz`.
    #[cfg(test)]
    pub(crate) nbits_trunc: i32,
    /// Number of coefficients in the frame.
    ne: usize,
    /// `rateFlag` of §3.3.10.4, needed by the arithmetic encoder.
    rate_flag: i32,
}

impl SpecFrame {
    /// Writes `lastnz`, `lsbMode` and the global gain, in that order
    /// (§3.3.13.3).
    pub(crate) fn write_side(&self, bits: &mut BitWriter<'_>) -> Result<(), Lc3Error> {
        let value = ((self.lastnz.max(2) >> 1) - 1) as u32;
        bits.write_uint(value, self.nbits_nq)?;
        bits.write_bit(self.lsb_mode)?;
        bits.write_uint(u32::from(self.gg_ind), 8)
    }

    /// Writes the three-bit noise factor (§3.3.13.3).
    pub(crate) fn write_noise(&self, bits: &mut BitWriter<'_>) -> Result<(), Lc3Error> {
        bits.write_uint(u32::from(self.noise_factor.min(7)), 3)
    }

    /// Encodes the spectrum (§3.3.13.4.2), interleaving arithmetic symbols with
    /// backward sign and LSB planes, and fills `lsbs` in LSB mode.
    pub(crate) fn write_ac(
        &mut self,
        ac: &mut AcEncoder,
        bits: &mut BitWriter<'_>,
    ) -> Result<(), Lc3Error> {
        self.lsbs.clear();
        let mut c = 0usize;

        let mut k = 0usize;
        while k < self.lastnz {
            let mut t = c + self.rate_flag as usize;
            if k > self.ne / 2 {
                t += 256;
            }

            let mut a = self.xq[k].unsigned_abs();
            let mut b = self.xq[k + 1].unsigned_abs();
            let mut lev = 0usize;
            let mut lsb0 = false;
            let mut lsb1 = false;
            while a.max(b) >= 4 {
                let pki = usize::from(AC_SPEC_LOOKUP[t + lev.min(3) * 1024]);
                ac.encode(AC_SPEC_CUMFREQ[pki][16], AC_SPEC_FREQ[pki][16]);
                if self.lsb_mode && lev == 0 {
                    // The first plane is carried by the residual data instead.
                    lsb0 = a & 1 != 0;
                    lsb1 = b & 1 != 0;
                } else {
                    bits.write_bit(a & 1 != 0)?;
                    bits.write_bit(b & 1 != 0)?;
                }
                a >>= 1;
                b >>= 1;
                lev += 1;
            }

            let pki = usize::from(AC_SPEC_LOOKUP[t + lev.min(3) * 1024]);
            let sym = (a + 4 * b) as usize;
            ac.encode(AC_SPEC_CUMFREQ[pki][sym], AC_SPEC_FREQ[pki][sym]);

            let mut a_lsb = self.xq[k].unsigned_abs();
            let mut b_lsb = self.xq[k + 1].unsigned_abs();
            if self.lsb_mode && lev > 0 {
                // In LSB mode the discarded first bitplane carries the sign of
                // a magnitude of one; the arithmetic sign field only encodes
                // the remaining magnitudes (§3.3.13.4.2).
                a_lsb >>= 1;
                b_lsb >>= 1;
                self.lsbs.push(lsb0);
                if a_lsb == 0 && self.xq[k] != 0 {
                    self.lsbs.push(self.xq[k] < 0);
                }
                self.lsbs.push(lsb1);
                if b_lsb == 0 && self.xq[k + 1] != 0 {
                    self.lsbs.push(self.xq[k + 1] < 0);
                }
            }

            if a_lsb > 0 {
                bits.write_bit(self.xq[k] < 0)?;
            }
            if b_lsb > 0 {
                bits.write_bit(self.xq[k + 1] < 0)?;
            }

            // The escape count is capped only for the model index, not for the
            // context update.
            let lev = lev.min(3);
            let state = if lev <= 1 {
                1 + (a + b) * (lev as u32 + 1)
            } else {
                12 + lev as u32
            };
            c = (c & 15) * 16 + state as usize;
            k += 2;
        }
        Ok(())
    }

    /// Writes the residual bits of §3.3.13.5: at most `available` bits, from
    /// `lsbs` in LSB mode and from `res_bits` otherwise.
    pub(crate) fn write_residual(
        &self,
        bits: &mut BitWriter<'_>,
        available: i32,
    ) -> Result<(), Lc3Error> {
        if available <= 0 {
            return Ok(());
        }
        let source = if self.lsb_mode {
            &self.lsbs
        } else {
            &self.res_bits
        };
        let count = (available as usize).min(source.len());
        for &bit in &source[..count] {
            bits.write_bit(bit)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lc3::bits::BitReader;
    use crate::lc3::spec::SpectrumDecoder;

    /// Small deterministic 32-bit linear congruential generator.
    struct Lcg(u32);

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self(seed)
        }

        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0
        }

        /// Uniform value in `[-1, 1)`.
        fn bipolar(&mut self) -> f32 {
            let unit = (self.next() >> 8) as f32 / (1u32 << 24) as f32;
            2.0 * unit - 1.0
        }
    }

    /// Spectrum with a decaying envelope, in the 16-bit PCM scale.
    fn decaying_spectrum(seed: u32, ne: usize, amplitude: f32) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        (0..ne)
            .map(|k| {
                let envelope = 1.0 - 0.8 * (k as f32 / ne as f32);
                amplitude * envelope * rng.bipolar()
            })
            .collect()
    }

    /// Writes a complete spectral payload and checks that the arithmetic data
    /// does not overlap the backward-written side information.
    fn write_payload(frame: &mut SpecFrame, nbytes: usize) -> Vec<u8> {
        let nbits = (nbytes * 8) as i32;
        let mut payload = vec![0u8; nbytes];
        let available;
        {
            let mut writer = BitWriter::new(&mut payload);
            frame.write_side(&mut writer).unwrap();
            frame.write_noise(&mut writer).unwrap();
            let mut ac = AcEncoder::new();
            frame.write_ac(&mut ac, &mut writer).unwrap();
            available = writer.bits_written(nbits) - ac.bits_estimate();
            frame.write_residual(&mut writer, available).unwrap();
            ac.finish();
            let used = ac.bytes().len();
            let bottom = writer.bottom();
            assert!(
                used as i64 <= bottom + 1,
                "arithmetic data overlaps the side information"
            );
            payload[..used].copy_from_slice(ac.bytes());
        }
        assert!(available > 0, "no residual budget in the test frame");
        payload
    }

    /// Decodes a spectral payload and compares it with the quantized spectrum.
    fn check_round_trip(frame: &SpecFrame, payload: &[u8], dt: usize, sr: usize, bw: usize) {
        let nbytes = payload.len();
        let mut reader = BitReader::new(payload).unwrap();
        let mut decoder = SpectrumDecoder::new(dt, sr);
        decoder.load(&mut reader).unwrap();
        let decoded = decoder.decode(&mut reader, bw, nbytes).unwrap();
        let ne = *band_limits(dt, sr).last().unwrap() as usize;
        assert_eq!(decoded.len(), ne);

        let mut worst = 0f32;
        for k in 0..frame.lastnz {
            if frame.xq[k] == 0 {
                continue;
            }
            let scaled = decoded[k] / frame.gg;
            assert_ne!(scaled, 0.0, "coefficient {k} vanished");
            assert_eq!(
                scaled < 0.0,
                frame.xq[k] < 0,
                "sign mismatch at coefficient {k}"
            );
            let deviation = (scaled.abs() - frame.xq[k].unsigned_abs() as f32).abs();
            worst = worst.max(deviation);
            assert!(
                deviation <= 1.0,
                "coefficient {k}: |{}| vs |{}| (deviation {deviation})",
                scaled.abs(),
                frame.xq[k]
            );
        }
        assert!(
            worst <= 1.0,
            "round-trip deviation {worst} exceeds one step"
        );
    }

    #[test]
    fn budget_and_quantization_are_sane_for_all_configurations() {
        let sizes = [20usize, 40, 80, 160, 320, 400];
        let overhead = 52;
        for dt in 0..2 {
            for sr in 0..5 {
                let ne = *band_limits(dt, sr).last().unwrap() as usize;
                for &nbytes in &sizes {
                    let nbits = (nbytes * 8) as i32;
                    let nbits_nq = ilog2(ne as u32 / 2 - 1) + 1;
                    if nbits <= overhead + ari_overhead(nbits_nq, nbits) {
                        continue;
                    }
                    let xf =
                        decaying_spectrum((dt * 977 + sr * 131 + nbytes) as u32 + 1, ne, 1500.0);
                    let mut encoder = SpecEncoder::new();
                    let frame = encoder.analyse(&xf, dt, sr, nbytes, overhead, ne).unwrap();

                    assert_eq!(frame.xq.len(), ne);
                    assert!(frame.gg > 0.0);
                    assert!(frame.noise_factor <= 7);
                    assert_eq!(frame.lastnz % 2, 0);
                    assert!(frame.lastnz >= 2 && frame.lastnz <= ne);
                    let max = (frame.nbits_spec - frame.nbits_trunc + 4).max(0) as usize;
                    assert!(frame.res_bits.len() <= max);
                    assert!(frame.xq[frame.lastnz..].iter().all(|&value| value == 0));

                    // A second frame exercises the offset history.
                    let frame = encoder.analyse(&xf, dt, sr, nbytes, overhead, ne).unwrap();
                    assert_eq!(frame.xq.len(), ne);
                }
            }
        }
    }

    #[test]
    fn silent_spectrum_produces_a_zero_frame() {
        for dt in 0..2 {
            let sr = 1;
            let ne = *band_limits(dt, sr).last().unwrap() as usize;
            let xf = vec![0f32; ne];
            let mut encoder = SpecEncoder::new();
            let mut frame = encoder.analyse(&xf, dt, sr, 40, 52, ne).unwrap();

            assert_eq!(frame.lastnz, 2);
            assert_eq!(frame.gg_ind, 0, "gg_min is zero for a silent spectrum");
            assert_eq!(frame.noise_factor, 7);
            assert!(frame.res_bits.is_empty());
            assert!(!frame.lsb_mode);

            let payload = write_payload(&mut frame, 40);
            let mut reader = BitReader::new(&payload).unwrap();
            let mut decoder = SpectrumDecoder::new(dt, sr);
            decoder.load(&mut reader).unwrap();
            let decoded = decoder.decode(&mut reader, 1, 40).unwrap();
            assert!(decoded.iter().all(|&value| value == 0.0));
        }
    }

    #[test]
    fn low_rate_round_trip_uses_residual_bits() {
        let (dt, sr, nbytes) = (0usize, 1usize, 40usize);
        let ne = *band_limits(dt, sr).last().unwrap() as usize;
        for seed in [1u32, 7, 1234, 98_765] {
            let xf = decaying_spectrum(seed, ne, 900.0);
            let mut encoder = SpecEncoder::new();
            let mut frame = encoder.analyse(&xf, dt, sr, nbytes, 52, ne).unwrap();
            assert!(!frame.lsb_mode, "low-rate frame must use residual bits");

            // The side information must read back unchanged.
            let payload = write_payload(&mut frame, nbytes);
            let mut reader = BitReader::new(&payload).unwrap();
            let lastnz = ((reader.read_uint(frame.nbits_nq).unwrap() + 1) << 1) as usize;
            assert_eq!(lastnz, frame.lastnz);
            assert_eq!(reader.read_bit().unwrap(), frame.lsb_mode);
            assert_eq!(reader.read_uint(8).unwrap(), u32::from(frame.gg_ind));

            check_round_trip(&frame, &payload, dt, sr, 1);
        }
    }

    #[test]
    fn high_rate_round_trip_exercises_lsb_mode() {
        let (dt, sr, nbytes) = (1usize, 4usize, 400usize);
        let ne = *band_limits(dt, sr).last().unwrap() as usize;
        for seed in [3u32, 17, 4242, 555_555] {
            let mut rng = Lcg::new(seed);
            // Dense and loud: the gain cannot grow enough to reach the budget,
            // so the first LSB plane is moved to the residual block.
            let xf: Vec<f32> = (0..ne)
                .map(|_| {
                    if rng.next() & 1 == 0 {
                        30_000.0
                    } else {
                        -30_000.0
                    }
                })
                .collect();
            let mut encoder = SpecEncoder::new();
            let mut frame = encoder.analyse(&xf, dt, sr, nbytes, 52, ne).unwrap();
            assert!(
                frame.lsb_mode,
                "expected LSB mode for a dense loud spectrum"
            );

            let payload = write_payload(&mut frame, nbytes);
            assert!(
                !frame.lsbs.is_empty() || frame.lastnz == 2,
                "LSB mode did not collect any backward bits"
            );
            let mut reader = BitReader::new(&payload).unwrap();
            let lastnz = ((reader.read_uint(frame.nbits_nq).unwrap() + 1) << 1) as usize;
            assert_eq!(lastnz, frame.lastnz);
            assert!(reader.read_bit().unwrap());

            check_round_trip(&frame, &payload, dt, sr, 4);
        }
    }
}
