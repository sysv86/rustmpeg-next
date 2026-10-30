// SPDX-License-Identifier: Apache-2.0
//! LC3 long-term postfilter analysis (Bluetooth LC3 v1.0.1 §3.3.9).
//!
//! The analysis resamples the input frame to 12.8 kHz, high-pass filters it,
//! searches for a pitch lag on a 6.4 kHz decimated copy, refines the lag on the
//! 12.8 kHz signal, and decides whether the postfilter is activated. All
//! signals are kept in the same scale as the 16-bit PCM input; the pipeline is
//! responsible for producing that scale.
//!
//! The processing follows Equations 78-105. Signals are accessed with negative
//! indices for samples of previous frames; missing history reads as zero, both
//! at start-up and for indices older than the retained history, so no input can
//! make the analysis panic.

use super::enc_tables::{LTPF_INTERP_R, LTPF_INTERP_X12K8, RESAMP_FILTER};
use super::tables::{NS, SRATE_KHZ};

/// Longest input frame (`10 ms` at 48 kHz).
const MAX_NS: usize = 480;
/// Longest 12.8 kHz frame (`10 ms`).
const MAX_LEN12_8: usize = 128;
/// Longest 6.4 kHz frame (`10 ms`).
const MAX_LEN6_4: usize = 64;
/// Retained input samples. The resampler needs `240 / P` samples, at most 60.
const RAW_HIST: usize = 64;
/// Retained high-passed samples. The delay line needs at most `D_LTPF = 44`.
const HP_HIST: usize = 64;
/// Retained delayed 12.8 kHz samples. Equation 97 needs indices down to -232.
const DEL_HIST: usize = 288;
/// Retained 6.4 kHz samples. Equation 86 needs indices down to -114.
const X64_HIST: usize = 128;
/// Number of lags searched in Equation 86 (`17..=114`).
const R64_LEN: usize = 114 - 17 + 1;
/// Number of lags in the refined search, including one guard lag per side.
const R12_LEN: usize = 232 - 28 + 1;

/// First searched 6.4 kHz lag.
const KMIN_6_4: usize = 17;
/// Last searched 6.4 kHz lag.
const KMAX_6_4: usize = 114;
/// First searched 12.8 kHz integer lag.
const KMIN_12_8: i32 = 32;
/// Last searched 12.8 kHz integer lag.
const KMAX_12_8: i32 = 228;
/// Fractional-pitch search step boundary in the 12.8 kHz domain.
const FR_HIGH_START: i32 = 157;
/// Second fractional-pitch search step boundary in the 12.8 kHz domain.
const FR_MID_START: i32 = 127;
/// 6.4 kHz decimation filter (§3.3.9.5).
const H2: [f32; 5] = [0.12367964, 0.23535122, 0.28193828, 0.23535122, 0.12367964];

/// Numerator of the 50 Hz high-pass filter (§3.3.9.4, Equation 83).
const HP_B0: f32 = 0.982_794_7;
/// Second numerator coefficient of the 50 Hz high-pass filter.
const HP_B1: f32 = -1.965_589_4;
/// Third numerator coefficient of the 50 Hz high-pass filter.
const HP_B2: f32 = 0.982_794_7;
/// First feedback coefficient of the 50 Hz high-pass filter (sign-inverted).
const HP_A1: f32 = -1.965_293_4;
/// Second feedback coefficient of the 50 Hz high-pass filter (sign-inverted).
const HP_A2: f32 = 0.965_885_46;

/// LTPF parameters produced for one channel and frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LtpfParams {
    /// Whether the pitch lag is transmitted.
    pub(crate) pitch_present: bool,
    /// The 9-bit pitch index of Equation 102 (`0` when absent).
    pub(crate) pitch_index: u16,
    /// The activation bit of §3.3.9.8.
    pub(crate) active: bool,
    /// Bits consumed by the LTPF bitstream: `1` or `11`.
    pub(crate) nbits: i32,
}

/// Per-channel LTPF analysis state, including resampler and filter history.
pub(crate) struct LtpfAnalysis {
    /// Frame duration index: `0` for 7.5 ms, `1` for 10 ms.
    dt: usize,
    /// Sample-rate index: `0` to `4` for 8 to 48 kHz.
    sr: usize,
    /// Upsampling factor `P = 192000 / fs`.
    p: usize,
    /// Resampling scale factor (`0.5` at 8 kHz, `1` otherwise).
    resfac: f32,
    /// Length of one 12.8 kHz frame.
    len12_8: usize,
    /// Length of one 6.4 kHz frame.
    len6_4: usize,
    /// `D_LTPF` delay in 12.8 kHz samples.
    d: i32,
    /// Previous high-pass input sample.
    hp_x1: f32,
    /// Second previous high-pass input sample.
    hp_x2: f32,
    /// Previous high-pass output sample.
    hp_y1: f32,
    /// Second previous high-pass output sample.
    hp_y2: f32,
    /// Raw input history, most recent sample last.
    raw_hist: [f32; RAW_HIST],
    /// High-passed 12.8 kHz history, most recent sample last.
    hp_hist: [f32; HP_HIST],
    /// Delayed 12.8 kHz history, most recent sample last.
    del_hist: [f32; DEL_HIST],
    /// 6.4 kHz history, most recent sample last.
    x64_hist: [f32; X64_HIST],
    /// Final 6.4 kHz pitch lag of the previous frame (Equation 90).
    prev_tcur: usize,
    /// Activation bit of the previous frame.
    mem_ltpf_active: bool,
    /// Normalized correlation of the previous frame.
    mem_nc: f32,
    /// Normalized correlation of the frame before the previous one.
    mem_mem_nc: f32,
    /// Previous pitch lag in quarter samples (`0` when absent).
    mem_pitch: i32,
}

impl LtpfAnalysis {
    /// Creates an analysis state for a frame duration and sample-rate index.
    ///
    /// `dt` is `0` for 7.5 ms and `1` for 10 ms; `sr` is `0` to `4` for 8, 16,
    /// 24, 32, and 48 kHz. Out-of-range indices are clamped so that the
    /// analysis never panics on a bad configuration.
    pub(crate) fn new(dt: usize, sr: usize) -> Self {
        let dt = dt.min(1);
        let sr = sr.min(4);
        let fs_hz = SRATE_KHZ[sr] * 1000;
        let len12_8 = if dt == 0 { 96 } else { 128 };
        Self {
            dt,
            sr,
            p: (192_000 / fs_hz) as usize,
            resfac: if fs_hz == 8_000 { 0.5 } else { 1.0 },
            len12_8,
            len6_4: len12_8 / 2,
            d: if dt == 0 { 44 } else { 24 },
            hp_x1: 0.0,
            hp_x2: 0.0,
            hp_y1: 0.0,
            hp_y2: 0.0,
            raw_hist: [0.0; RAW_HIST],
            hp_hist: [0.0; HP_HIST],
            del_hist: [0.0; DEL_HIST],
            x64_hist: [0.0; X64_HIST],
            prev_tcur: KMIN_6_4,
            mem_ltpf_active: false,
            mem_nc: 0.0,
            mem_mem_nc: 0.0,
            mem_pitch: 0,
        }
    }

    /// Analyses one frame and returns the LTPF parameters.
    ///
    /// `x` holds one frame of `NS[dt][sr]` samples of 16-bit PCM scale; shorter
    /// slices are treated as zero-padded and longer slices are truncated.
    /// `nbytes` is the size of the frame payload and `near_nyquist` is the flag
    /// of §3.3.4.5 that disables the postfilter.
    pub(crate) fn analyse(&mut self, x: &[f32], nbytes: usize, near_nyquist: bool) -> LtpfParams {
        let len12_8 = self.len12_8;
        let len6_4 = self.len6_4;
        let xdel_len = len12_8 + 2;

        // Honour the documented zero-padding: filtering already reads missing
        // samples as zero, but the retained history must advance by a full frame.
        let ns = NS[self.dt][self.sr];
        let mut frame = [0.0f32; MAX_NS];
        let copied = x.len().min(ns);
        frame[..copied].copy_from_slice(&x[..copied]);
        let x = &frame[..ns];

        // §3.3.9.3 resampling and §3.3.9.4 high-pass filtering.
        let mut xhp = [0.0f32; MAX_LEN12_8];
        for (n, slot) in xhp[..len12_8].iter_mut().enumerate() {
            let x12_8 = self.resample(x, n);
            let y = HP_B0 * x12_8 + HP_B1 * self.hp_x1 + HP_B2 * self.hp_x2
                - HP_A1 * self.hp_y1
                - HP_A2 * self.hp_y2;
            self.hp_x2 = self.hp_x1;
            self.hp_x1 = x12_8;
            self.hp_y2 = self.hp_y1;
            self.hp_y1 = y;
            *slot = y;
        }

        // Equation 84: delay the high-passed signal by D_LTPF.
        let mut xdel = [0.0f32; MAX_LEN12_8 + 2];
        for (n, slot) in xdel[..xdel_len].iter_mut().enumerate() {
            *slot = sample(&self.hp_hist, &xhp[..len12_8], n as i32 - self.d);
        }

        // Equation 85: decimate to 6.4 kHz. The reference decimator stores the
        // result on the 16-bit PCM grid; rounding to that grid here keeps the
        // sub-LSB ripple of the polyphase resampler from being mistaken for a
        // periodic signal (it is below the input resolution in any case).
        let mut x64 = [0.0f32; MAX_LEN6_4];
        for (n, slot) in x64[..len6_4].iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (k, &h) in H2.iter().enumerate() {
                let index = (2 * n) as i32 + k as i32 - 3;
                sum += sample(&self.del_hist, &xdel[..xdel_len], index) * h;
            }
            *slot = sum.round();
        }

        // Equations 86-90: autocorrelation and the two lag estimates.
        let mut r64 = [0.0f32; R64_LEN];
        for (offset, slot) in r64.iter_mut().enumerate() {
            let k = (KMIN_6_4 + offset) as i32;
            let mut sum = 0.0f32;
            for n in 0..len6_4 {
                let a = x64[n];
                let b = sample(&self.x64_hist, &x64[..len6_4], n as i32 - k);
                sum += a * b;
            }
            *slot = sum;
        }

        let mut t1 = KMIN_6_4;
        let mut best = f32::NEG_INFINITY;
        let span = (KMAX_6_4 - KMIN_6_4) as f32;
        for (offset, &r) in r64.iter().enumerate() {
            let k = KMIN_6_4 + offset;
            let w = 1.0 - 0.5 * ((k - KMIN_6_4) as f32) / span;
            let weighted = r * w;
            if weighted > best {
                best = weighted;
                t1 = k;
            }
        }

        let t2_lo = KMIN_6_4.max(self.prev_tcur.saturating_sub(4));
        let t2_hi = KMAX_6_4.min(self.prev_tcur + 4);
        let mut t2 = t2_lo;
        let mut best_unweighted = r64[t2_lo - KMIN_6_4];
        for k in t2_lo + 1..=t2_hi {
            let value = r64[k - KMIN_6_4];
            if value > best_unweighted {
                best_unweighted = value;
                t2 = k;
            }
        }

        // Equations 91-93: select the final 6.4 kHz lag.
        let nc1 = normcorr(&self.x64_hist, &x64[..len6_4], len6_4, t1);
        let nc2 = normcorr(&self.x64_hist, &x64[..len6_4], len6_4, t2);
        let tcur = if nc2 <= 0.85 * nc1 { t1 } else { t2 };
        self.prev_tcur = tcur;

        // Equation 94: the pitch-presence flag.
        let pitch_present = normcorr(&self.x64_hist, &x64[..len6_4], len6_4, tcur) > 0.6;

        let mut pitch_index = 0u16;
        let mut pitch = 0i32;
        let mut nc = 0.0f32;

        if pitch_present {
            let tcur = tcur as i32;
            let kmin = (2 * tcur - 4).max(KMIN_12_8);
            let kmax = (2 * tcur + 4).min(KMAX_12_8);
            let r_start = kmin - 4;
            let r_end = kmax + 4;

            // Equations 96-97: autocorrelation around the coarse lag.
            let mut r12 = [0.0f32; R12_LEN];
            for k in r_start..=r_end {
                let mut sum = 0.0f32;
                for n in 0..len12_8 {
                    let a = xdel[n];
                    let b = sample(&self.del_hist, &xdel[..xdel_len], n as i32 - k);
                    sum += a * b;
                }
                r12[(k - r_start) as usize] = sum;
            }

            let mut pitch_int = kmin;
            let mut best_r = r12[(kmin - r_start) as usize];
            for k in kmin + 1..=kmax {
                let value = r12[(k - r_start) as usize];
                if value > best_r {
                    best_r = value;
                    pitch_int = k;
                }
            }

            // Equations 98-101: fractional refinement. The candidate order is
            // zero, then increasing phase, and matches the conformance
            // reference; strict comparison keeps the earliest candidate on ties.
            let mut pitch_fr = 0i32;
            if pitch_int < FR_HIGH_START {
                let candidates: &[i32] = match pitch_int {
                    KMIN_12_8 => &[0, 1, 2, 3],
                    33..=126 => &[0, 1, -1, 2, -2, 3, -3],
                    FR_MID_START..=156 => &[0, 2, -2],
                    _ => &[0],
                };
                let mut best_fr = interpolate_corr(&r12, r_start, pitch_int, 0);
                let mut fr = 0i32;
                for &d in candidates.iter().skip(1) {
                    let value = interpolate_corr(&r12, r_start, pitch_int, d);
                    if value > best_fr {
                        best_fr = value;
                        fr = d;
                    }
                }
                pitch_fr = fr;
            }

            if pitch_fr < 0 {
                pitch_int -= 1;
                pitch_fr += 4;
            }

            // Equation 102: the transmitted pitch index.
            let index = if pitch_int >= FR_HIGH_START {
                pitch_int + 283
            } else if pitch_int >= FR_MID_START {
                2 * pitch_int + pitch_fr / 2 + 126
            } else {
                4 * pitch_int + pitch_fr - 128
            };
            pitch_index = index as u16;
            pitch = 4 * pitch_int + pitch_fr;

            // Equations 103-105: normalized correlation of the refined lag.
            let mut corr = 0.0f32;
            let mut energy = 0.0f32;
            let mut lagged_energy = 0.0f32;
            for n in 0..len12_8 {
                let a = xi(&self.del_hist, &xdel[..xdel_len], n as i32, 0);
                let b = xi(
                    &self.del_hist,
                    &xdel[..xdel_len],
                    n as i32 - pitch_int,
                    pitch_fr,
                );
                corr += a * b;
                energy += a * a;
                lagged_energy += b * b;
            }
            let denom = (energy * lagged_energy).sqrt();
            if denom > 0.0 {
                nc = corr / denom;
            }
        }

        // §3.4.9.4: gain tuning only decides whether the postfilter may run.
        let gain = gain_ltpf(nbytes, self.dt, self.sr);
        let mut active = false;
        if pitch_present && gain > 0.0 {
            if self.mem_ltpf_active {
                let pitch_diff = (pitch - self.mem_pitch).abs();
                active = nc > 0.9 || (nc > 0.84 && pitch_diff < 8 && nc - self.mem_nc > -0.1);
            } else {
                active =
                    (self.dt == 1 || self.mem_mem_nc > 0.94) && self.mem_nc > 0.94 && nc > 0.94;
            }
        }
        if near_nyquist {
            active = false;
        }

        // Retain the signals needed by the next frame. The delay line only
        // advances by one frame; its two extra samples are the first samples of
        // the next frame and must not be stored twice.
        shift_in(&mut self.raw_hist, x);
        shift_in(&mut self.hp_hist, &xhp[..len12_8]);
        shift_in(&mut self.del_hist, &xdel[..len12_8]);
        shift_in(&mut self.x64_hist, &x64[..len6_4]);

        // Update the activation memory of §3.3.9.8.
        self.mem_ltpf_active = active;
        self.mem_mem_nc = self.mem_nc;
        self.mem_nc = nc;
        self.mem_pitch = pitch;

        LtpfParams {
            pitch_present,
            pitch_index,
            active,
            nbits: if pitch_present { 11 } else { 1 },
        }
    }

    /// Evaluates Equation 78 for one 12.8 kHz output sample.
    fn resample(&self, x: &[f32], n: usize) -> f32 {
        let half = 120 / self.p;
        let base = (15 * n) / self.p;
        let phase = (15 * n) % self.p;
        let mut sum = 0.0f32;
        for k in -(half as i32)..=(half as i32) {
            let h = h12_8(self.p as i32 * k - phase as i32);
            if h == 0.0 {
                continue;
            }
            let index = base as i32 + k - half as i32;
            sum += sample(&self.raw_hist, x, index) * h;
        }
        sum * self.resfac * self.p as f32
    }
}

/// Returns `h12.8` of Equation 79 from the normative filter table.
fn h12_8(n: i32) -> f32 {
    if n > -120 && n < 120 {
        RESAMP_FILTER[(n + 119) as usize]
    } else {
        0.0
    }
}

/// Returns `h4` of Equation 100 from the normative filter table.
fn h4(n: i32) -> f32 {
    if n > -16 && n < 16 {
        LTPF_INTERP_R[(n + 15) as usize]
    } else {
        0.0
    }
}

/// Returns `h_i` of Equation 105 from the normative filter table.
fn hi(n: i32) -> f32 {
    if n > -8 && n < 8 {
        LTPF_INTERP_X12K8[(n + 7) as usize]
    } else {
        0.0
    }
}

/// Reads `current[index]` for non-negative indices and the retained history
/// for negative ones. Out-of-range reads return zero.
fn sample(history: &[f32], current: &[f32], index: i32) -> f32 {
    if index >= 0 {
        current.get(index as usize).copied().unwrap_or(0.0)
    } else {
        let history_index = history.len() as i64 + i64::from(index);
        if history_index >= 0 {
            history.get(history_index as usize).copied().unwrap_or(0.0)
        } else {
            0.0
        }
    }
}

/// Shifts `samples` into a history buffer, keeping the most recent entries.
fn shift_in(history: &mut [f32], samples: &[f32]) {
    let h = history.len();
    let s = samples.len();
    if h == 0 || s == 0 {
        return;
    }
    if s >= h {
        history.copy_from_slice(&samples[s - h..]);
    } else {
        history.copy_within(s.., 0);
        history[h - s..].copy_from_slice(samples);
    }
}

/// Normalized correlation of Equation 92, clamped to non-negative values.
fn normcorr(history: &[f32], current: &[f32], len: usize, lag: usize) -> f32 {
    let mut corr = 0.0f32;
    let mut energy = 0.0f32;
    let mut lagged_energy = 0.0f32;
    for n in 0..len {
        let a = sample(history, current, n as i32);
        let b = sample(history, current, n as i32 - lag as i32);
        corr += a * b;
        energy += a * a;
        lagged_energy += b * b;
    }
    let denom = (energy * lagged_energy).sqrt();
    if denom > 0.0 {
        (corr / denom).max(0.0)
    } else {
        0.0
    }
}

/// Evaluates Equation 99 at fractional phase `d`.
fn interpolate_corr(r12: &[f32], r_start: i32, pitch_int: i32, d: i32) -> f32 {
    let mut sum = 0.0f32;
    for m in -4..=4 {
        let lag = pitch_int + m;
        let value = r12.get((lag - r_start) as usize).copied().unwrap_or(0.0);
        sum += value * h4(4 * m - d);
    }
    sum
}

/// Evaluates `xi(n, d)` of Equation 104.
fn xi(history: &[f32], current: &[f32], n: i32, d: i32) -> f32 {
    let mut sum = 0.0f32;
    for k in -2..=2 {
        sum += sample(history, current, n - k) * hi(4 * k - d);
    }
    sum
}

/// Returns the LTPF gain of §3.4.9.4 (`0` disables the postfilter).
fn gain_ltpf(nbytes: usize, dt: usize, sr: usize) -> f32 {
    let nbits = nbytes.saturating_mul(8);
    let t_nbits = if dt == 1 {
        nbits
    } else {
        nbits.saturating_mul(4).saturating_add(1) / 3
    };
    let base = 320 + 80 * sr;
    if t_nbits < base {
        0.4
    } else if t_nbits < base + 80 {
        0.35
    } else if t_nbits < base + 160 {
        0.3
    } else if t_nbits < base + 240 {
        0.25
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;

    /// Small deterministic generator so the tests need no external crate.
    fn pseudo_random(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (*state >> 40) as u32;
        (bits as f32 / 8_388_608.0) * 65_535.0 - 32_768.0
    }

    /// Inverts Equation 102 far enough to recover the integer lag.
    fn decode_pitch_int(index: u16) -> i32 {
        let index = i32::from(index);
        if index >= 440 {
            index - 283
        } else if index >= 380 {
            (index - 126) / 2
        } else {
            (index + 128) / 4
        }
    }

    /// Builds one frame of a 200 Hz harmonic tone at 48 kHz.
    fn harmonic_frame(frame: usize, ns: usize) -> Vec<f32> {
        (0..ns)
            .map(|n| {
                let t = (frame * ns + n) as f32;
                (1..=4)
                    .map(|h| {
                        let h = h as f32;
                        (8_000.0 / h) * (2.0 * PI * 200.0 * h * t / 48_000.0).sin()
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn random_input_is_safe_for_every_configuration() {
        let mut state = 0x0123_4567_89ab_cdefu64;
        for (dt, row) in NS.iter().enumerate() {
            for (sr, &ns) in row.iter().enumerate() {
                let mut analysis = LtpfAnalysis::new(dt, sr);
                for iteration in 0..3 {
                    let x: Vec<f32> = (0..ns).map(|_| pseudo_random(&mut state)).collect();
                    // The last frame exercises the short-input zero-padding path.
                    let input = if iteration == 2 {
                        &x[..x.len() / 3]
                    } else {
                        &x[..]
                    };
                    let params = analysis.analyse(input, 120, false);
                    assert_eq!(
                        params.nbits,
                        if params.pitch_present { 11 } else { 1 },
                        "nbits must match pitch_present for dt={dt}, sr={sr}"
                    );
                    if params.active {
                        assert!(
                            params.pitch_present,
                            "active implies pitch_present for dt={dt}, sr={sr}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn periodic_tone_yields_expected_pitch() {
        let mut analysis = LtpfAnalysis::new(1, 4);
        let mut params = analysis.analyse(&harmonic_frame(0, 480), 80, false);
        for frame in 1..12 {
            params = analysis.analyse(&harmonic_frame(frame, 480), 80, false);
        }

        assert!(params.pitch_present, "a 200 Hz tone must carry a pitch lag");
        assert_eq!(params.nbits, 11);
        // 200 Hz at 12.8 kHz is a lag of 64 samples; allow a two-sample
        // tolerance for the decimated search and fractional refinement.
        let pitch_int = decode_pitch_int(params.pitch_index);
        assert!(
            (pitch_int - 64).abs() <= 2,
            "expected pitch_int near 64, got {pitch_int}"
        );
    }

    #[test]
    fn near_nyquist_disables_activation() {
        let mut enabled = LtpfAnalysis::new(1, 4);
        let mut ever_active = false;
        for frame in 0..8 {
            let params = enabled.analyse(&harmonic_frame(frame, 480), 80, false);
            ever_active |= params.active;
        }
        assert!(
            ever_active,
            "the periodic tone must activate the postfilter when allowed"
        );

        let mut disabled = LtpfAnalysis::new(1, 4);
        for frame in 0..8 {
            let params = disabled.analyse(&harmonic_frame(frame, 480), 80, true);
            assert!(!params.active, "near_nyquist must force the postfilter off");
        }
    }

    #[test]
    fn constant_input_has_no_pitch() {
        let mut analysis = LtpfAnalysis::new(1, 4);
        let dc = [10_000.0f32; 480];
        let mut params = analysis.analyse(&dc, 80, false);
        // The high-pass has a start-up transient, so run enough frames for it
        // to decay. The steady-state resampling ripple of a constant input is
        // below the 6.4 kHz quantization grid, so the detector then sees
        // silence.
        for _ in 0..80 {
            params = analysis.analyse(&dc, 80, false);
        }
        assert!(
            !params.pitch_present,
            "the high-pass filter must remove constant input"
        );
    }
}
