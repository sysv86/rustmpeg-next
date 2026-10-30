// SPDX-License-Identifier: Apache-2.0
//! LC3 encoder front-end detectors (Bluetooth LC3 v1.0.1 §3.3.4.4-§3.3.6).
//!
//! This module provides the band-energy estimation of Equation 10 together
//! with the near-Nyquist flag of Equation 11, the two-stage bandwidth
//! classifier of §3.3.5 (Table 3.6), and the time-domain attack detector of
//! §3.3.6.
//!
//! The attack detector keeps the filter and envelope history of the previous
//! frame, so it is stateful. Every entry point validates its duration and
//! sample-rate indices and its slice lengths; an invalid or too short input
//! yields a neutral result instead of a panic.

use super::spec::band_limits;
use super::tables::NS;

/// Index of the highest supported sample rate (48 kHz).
const MAX_SR: usize = 4;

/// Number of samples per attack-detector block (Equation 16).
const BLOCK_LEN: usize = 40;

/// Downsampled samples per 7.5 ms frame (`MF = 16 * Nms`).
const MF_7M5: usize = 120;

/// Downsampled samples per 10 ms frame (`MF = 16 * Nms`).
const MF_10M: usize = 160;

/// Result of the band-energy estimation.
pub(crate) struct BandEnergy {
    /// Mean square energy of every band (Equation 10).
    pub(crate) eb: Vec<f32>,
    /// Near-Nyquist flag of §3.3.4.5 (always false at 48 kHz).
    pub(crate) near_nyquist: bool,
}

/// Neutral band-energy result for an invalid or too short input.
fn no_bands() -> BandEnergy {
    BandEnergy {
        eb: Vec::new(),
        near_nyquist: false,
    }
}

/// Computes the per-band energies of §3.3.4.4 (mean square per band) and
/// the near-Nyquist flag of §3.3.4.5.
pub(crate) fn band_energies(xf: &[f32], dt: usize, sr: usize) -> BandEnergy {
    if dt > 1 || sr > MAX_SR {
        return no_bands();
    }
    let limits = band_limits(dt, sr);
    if limits.len() < 2 {
        return no_bands();
    }
    let Some(&last) = limits.last() else {
        return no_bands();
    };
    if last < 0 || last as usize > xf.len() {
        return no_bands();
    }
    if limits.iter().any(|&limit| limit < 0) || limits.windows(2).any(|pair| pair[0] > pair[1]) {
        return no_bands();
    }

    let mut eb = Vec::with_capacity(limits.len() - 1);
    for pair in limits.windows(2) {
        let start = pair[0] as usize;
        let stop = pair[1] as usize;
        let width = stop - start;
        if width == 0 {
            return no_bands();
        }
        let sum: f32 = xf[start..stop].iter().map(|value| value * value).sum();
        eb.push(sum / width as f32);
    }

    // §3.3.4.5, Equation 11 and Table 3.5. The detector is only active up to
    // 32 kHz; at 48 kHz the flag stays clear even for a top-heavy spectrum.
    let near_nyquist = if sr <= 3 && !eb.is_empty() {
        let nb = eb.len();
        let nn_idx = match dt {
            1 => nb.saturating_sub(2),
            _ => nb.saturating_sub(4),
        };
        if nn_idx > 0 && nn_idx < nb {
            let upper: f32 = eb[nn_idx..].iter().sum();
            let lower: f32 = eb[..nn_idx].iter().sum();
            upper > 30.0 * lower
        } else {
            false
        }
    } else {
        false
    };

    BandEnergy { eb, near_nyquist }
}

/// Bandwidth regions of Table 3.6 for 10 ms frames, indexed by `Nbw`.
///
/// Entry `sr` holds the `[start, stop]` band ranges of the `sr` candidate
/// bandwidths below the Nyquist rate of that sampling rate.
const REGIONS_10M: [&[(usize, usize)]; 5] = [
    &[],
    &[(53, 63)],
    &[(47, 56), (59, 63)],
    &[(44, 52), (54, 59), (60, 63)],
    &[(41, 49), (51, 55), (57, 60), (61, 63)],
];

/// Bandwidth regions of Table 3.6 for 7.5 ms frames, indexed by `Nbw`.
const REGIONS_7M5: [&[(usize, usize)]; 5] = [
    &[],
    &[(51, 63)],
    &[(45, 55), (58, 63)],
    &[(42, 51), (53, 58), (60, 63)],
    &[(40, 48), (51, 55), (57, 60), (61, 63)],
];

/// Second-stage comparison distances `L` for 10 ms frames.
const L_10M: [usize; 4] = [4, 4, 3, 1];

/// Second-stage comparison distances `L` for 7.5 ms frames.
const L_7M5: [usize; 4] = [4, 4, 3, 2];

/// Second-stage thresholds of Equation 13 as linear ratios `10^(Tc / 10)`.
const TC: [f32; 4] = [31.622_776, 199.526_23, 100.0, 100.0];

/// Detects the coded bandwidth index (0=NB .. 4=FB) per §3.3.5.
pub(crate) fn detect_bandwidth(eb: &[f32], dt: usize, sr: usize) -> usize {
    if sr == 0 {
        return 0;
    }
    if dt > 1 || sr > MAX_SR {
        return sr.min(MAX_SR);
    }

    let bwn = sr;
    let nb = band_limits(dt, sr).len().saturating_sub(1);
    if nb == 0 || eb.len() < nb {
        return bwn;
    }
    let regions = if dt == 1 {
        &REGIONS_10M[sr]
    } else {
        &REGIONS_7M5[sr]
    };
    let distances = if dt == 1 { &L_10M } else { &L_7M5 };

    // Stage 1 (§3.3.5.1, Equation 12): the largest candidate whose region is
    // not quiet. Equation 12 tests the region mean against TQ, with TQ = 20
    // for the first candidate and TQ = 10 for the others.
    let mut bw0 = 0usize;
    for (bw, &(start, stop)) in regions.iter().enumerate() {
        if stop >= eb.len() || start > stop {
            return bwn;
        }
        let width = stop - start + 1;
        let threshold = if bw == 0 { 20.0 } else { 10.0 };
        let sum: f32 = eb[start..=stop].iter().sum();
        if sum / width as f32 >= threshold {
            bw0 = bw + 1;
        }
    }

    // A candidate at the sampling-rate bandwidth is final.
    if bw0 >= bwn {
        return bw0;
    }

    // Stage 2 (Equation 13): look for an energy drop just above the cut-off
    // of the candidate bandwidth. The threshold is precomputed in the linear
    // domain, so the comparison is `EB(n-L) > 10^(Tc/10) * EB(n)`.
    let (start, _) = regions[bw0];
    let distance = distances[bw0];
    let threshold = TC[bw0];
    let first = start.saturating_sub(distance) + 1;
    for i in first..=start + 1 {
        if i >= distance && i < eb.len() && eb[i - distance] > threshold * eb[i] {
            return bw0;
        }
    }
    bwn
}

/// Number of side bits coding the bandwidth value (`0,1,2,2,3`).
pub(crate) fn bandwidth_bits(sr: usize) -> i32 {
    (usize::from(sr > 0) + usize::from(sr > 1) + usize::from(sr > 3)) as i32
}

/// Time-domain attack detector of §3.3.6, with cross-frame state.
pub(crate) struct AttackDetector {
    /// Frame duration index: `0` for 7.5 ms, `1` for 10 ms.
    dt: usize,
    /// Sample-rate index: `0` to `4` for 8 to 48 kHz.
    sr: usize,
    /// Downsampled samples per frame (`MF`).
    mf: usize,
    /// Downsampling factor `NS / MF`.
    factor: usize,
    /// Number of 40-sample energy blocks per frame.
    blocks: usize,
    /// Last downsampled sample of the previous frame (`x_att(-1)`).
    prev_x1: f32,
    /// Second to last downsampled sample (`x_att(-2)`).
    prev_x2: f32,
    /// Last envelope value of the previous frame (`A(-1)`).
    prev_a: f32,
    /// Last block energy of the previous frame (`E(-1)`).
    prev_e: f32,
    /// Attack position of the previous frame (normative `P_att`, -1 if none).
    prev_p: i32,
}

impl AttackDetector {
    /// Creates the detector state for a frame duration and sample rate.
    pub(crate) fn new(dt: usize, sr: usize) -> Self {
        let mf = if dt == 1 { MF_10M } else { MF_7M5 };
        let factor = if dt <= 1 && (3..=MAX_SR).contains(&sr) {
            NS[dt][sr] / mf
        } else {
            1
        };
        Self {
            dt,
            sr,
            mf,
            factor,
            blocks: mf / BLOCK_LEN,
            prev_x1: 0.0,
            prev_x2: 0.0,
            prev_a: 0.0,
            prev_e: 0.0,
            prev_p: -1,
        }
    }

    /// Returns whether §3.3.6.1 activates the detector for this frame size.
    fn enabled(&self, nbytes: usize) -> bool {
        match (self.dt, self.sr) {
            (1, 3) => nbytes > 80,
            (1, 4) => nbytes >= 100,
            (0, 3) => (61..150).contains(&nbytes),
            (0, 4) => (75..150).contains(&nbytes),
            _ => false,
        }
    }

    /// `x` has NS[dt][sr] samples in 16-bit PCM scale.
    /// Returns the attack flag F_att for this frame.
    pub(crate) fn run(&mut self, x: &[f32], nbytes: usize) -> bool {
        if !self.enabled(nbytes) {
            return false;
        }
        let needed = self.factor * self.mf;
        if x.len() < needed {
            return false;
        }

        // §3.3.6.2, Equations 14 and 15: downsample, then high-pass filter.
        // The scaling of the downsample is constant and cancels in the
        // ratio-based attack test, so only the sum is kept.
        let mut xm1 = self.prev_x1;
        let mut xm2 = self.prev_x2;
        let mut energy = [0f32; 4];
        for n in 0..self.mf {
            let base = n * self.factor;
            let xa: f32 = x[base..base + self.factor].iter().sum();
            let xh = 0.375 * xa - 0.5 * xm1 + 0.125 * xm2;
            energy[n / BLOCK_LEN] += xh * xh;
            xm2 = xm1;
            xm1 = xa;
        }
        self.prev_x2 = xm2;
        self.prev_x1 = xm1;

        // §3.3.6.3 and §3.3.6.4, Equations 16-19: block energies, delayed
        // envelope, and the attack decision.
        let mut prev_a = self.prev_a;
        let mut prev_e = self.prev_e;
        let mut p_att = -1i32;
        for (n, &block) in energy.iter().take(self.blocks).enumerate() {
            let envelope = (0.25 * prev_a).max(prev_e);
            if block > 8.5 * envelope {
                p_att = n as i32;
            }
            prev_a = envelope;
            prev_e = block;
        }

        let attack = p_att >= 0 || self.prev_p >= (self.blocks / 2) as i32;
        self.prev_p = p_att;
        self.prev_a = prev_a;
        self.prev_e = prev_e;
        attack
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lc3::tables::SRATE_KHZ;

    /// Builds band energies that are loud below `cut` and silent above it.
    fn band_energies_with_cut(dt: usize, sr: usize, cut: usize) -> Vec<f32> {
        let nb = band_limits(dt, sr).len() - 1;
        (0..nb)
            .map(|band| if band < cut { 100.0 } else { 0.0 })
            .collect()
    }

    /// Sine with `len` samples at `freq` hertz in the 16-bit PCM scale.
    fn sine(freq: f32, sr: usize, len: usize, amplitude: f32) -> Vec<f32> {
        let fs = SRATE_KHZ[sr] as f32 * 1000.0;
        (0..len)
            .map(|n| amplitude * (2.0 * std::f32::consts::PI * freq * n as f32 / fs).sin())
            .collect()
    }

    #[test]
    fn band_energies_are_band_mean_squares() {
        let (dt, sr) = (1, 4);
        let limits = band_limits(dt, sr);
        let ne = *limits.last().unwrap() as usize;
        let result = band_energies(&vec![1.0f32; ne], dt, sr);
        assert_eq!(result.eb.len(), limits.len() - 1);
        assert!(!result.near_nyquist);
        for &energy in &result.eb {
            assert!((energy - 1.0).abs() < 1e-6, "energy {energy}");
        }

        // A factor-two spectrum has a factor-four energy, and the width
        // weighted sum returns the total coefficient energy.
        let result = band_energies(&vec![2.0f32; ne], dt, sr);
        for &energy in &result.eb {
            assert!((energy - 4.0).abs() < 1e-5, "energy {energy}");
        }
        let total: f32 = result
            .eb
            .iter()
            .zip(limits.windows(2))
            .map(|(&energy, pair)| energy * (pair[1] - pair[0]) as f32)
            .sum();
        assert!((total - 4.0 * ne as f32).abs() < 1e-2, "total {total}");
    }

    #[test]
    fn band_energies_rejects_short_input() {
        let limits = band_limits(1, 4);
        let ne = *limits.last().unwrap() as usize;
        let result = band_energies(&vec![1.0f32; ne - 1], 1, 4);
        assert!(result.eb.is_empty());
        assert!(!result.near_nyquist);

        // Invalid indices are neutral as well.
        let result = band_energies(&vec![1.0f32; ne], 2, 9);
        assert!(result.eb.is_empty());
        assert!(!result.near_nyquist);
    }

    #[test]
    fn near_nyquist_uses_the_top_bands_only() {
        // 32 kHz, 10 ms: NB = 64, nn_idx = 62.
        let (dt, sr) = (1, 3);
        let limits = band_limits(dt, sr);
        let ne = *limits.last().unwrap() as usize;
        let nb = limits.len() - 1;
        let nn_idx = nb - 2;

        let mut top = vec![0f32; ne];
        for value in &mut top[limits[nn_idx] as usize..] {
            *value = 1000.0;
        }
        assert!(band_energies(&top, dt, sr).near_nyquist);

        // A flat spectrum has far more energy in the lower bands.
        assert!(!band_energies(&vec![1.0f32; ne], dt, sr).near_nyquist);

        // 7.5 ms at 32 kHz: NB = 64, nn_idx = 60.
        let limits = band_limits(0, sr);
        let ne = *limits.last().unwrap() as usize;
        let mut top = vec![0f32; ne];
        for value in &mut top[limits[60] as usize..] {
            *value = 1000.0;
        }
        assert!(band_energies(&top, 0, sr).near_nyquist);

        // The detector is inactive at 48 kHz, even for a top-heavy spectrum.
        let (dt, sr) = (1, 4);
        let limits = band_limits(dt, sr);
        let ne = *limits.last().unwrap() as usize;
        let mut top = vec![0f32; ne];
        for value in &mut top[limits[62] as usize..] {
            *value = 1.0e4;
        }
        assert!(!band_energies(&top, dt, sr).near_nyquist);
    }

    #[test]
    fn bandwidth_detector_follows_table_regions() {
        // 10 ms, 48 kHz: regions start at 41, 51, 57, and 61, so the cut
        // between loud and silent bands selects the bandwidth.
        let cases = [(40usize, 0usize), (51, 1), (57, 2), (61, 3), (64, 4)];
        for &(cut, expected) in &cases {
            let eb = band_energies_with_cut(1, 4, cut);
            assert_eq!(
                detect_bandwidth(&eb, 1, 4),
                expected,
                "10 ms 48 kHz with cut {cut}"
            );
        }

        // 7.5 ms, 48 kHz: regions start at 40, 51, 57, and 61.
        for &(cut, expected) in &cases {
            let eb = band_energies_with_cut(0, 4, cut);
            assert_eq!(
                detect_bandwidth(&eb, 0, 4),
                expected,
                "7.5 ms 48 kHz with cut {cut}"
            );
        }

        // 10 ms, 32 kHz: regions start at 44, 54, and 60 and the maximum
        // bandwidth is SWB (3).
        for &(cut, expected) in &[(43usize, 0usize), (54, 1), (60, 2), (64, 3)] {
            let eb = band_energies_with_cut(1, 3, cut);
            assert_eq!(
                detect_bandwidth(&eb, 1, 3),
                expected,
                "10 ms 32 kHz with cut {cut}"
            );
        }

        // 10 ms, 24 kHz: regions start at 47 and 59.
        for &(cut, expected) in &[(46usize, 0usize), (59, 1), (64, 2)] {
            let eb = band_energies_with_cut(1, 2, cut);
            assert_eq!(
                detect_bandwidth(&eb, 1, 2),
                expected,
                "10 ms 24 kHz with cut {cut}"
            );
        }

        // 10 ms, 16 kHz: a single region starting at 53.
        for &(cut, expected) in &[(52usize, 0usize), (64, 1)] {
            let eb = band_energies_with_cut(1, 1, cut);
            assert_eq!(
                detect_bandwidth(&eb, 1, 1),
                expected,
                "10 ms 16 kHz with cut {cut}"
            );
        }

        // A loud spectrum always reaches the sampling-rate bandwidth.
        for dt in 0..2 {
            for sr in 0..=4 {
                let nb = band_limits(dt, sr).len() - 1;
                let eb = vec![100.0f32; nb];
                assert_eq!(detect_bandwidth(&eb, dt, sr), sr);
            }
        }

        // Invalid or short inputs fall back to the sampling-rate bandwidth.
        assert_eq!(detect_bandwidth(&[], 1, 4), 4);
        assert_eq!(detect_bandwidth(&[100.0], 1, 4), 4);
        assert_eq!(detect_bandwidth(&[], 0, 0), 0);
    }

    #[test]
    fn bandwidth_bits_match_the_sampling_rate() {
        assert_eq!([0, 1, 2, 2, 3], [0, 1, 2, 3, 4].map(bandwidth_bits));
    }

    #[test]
    fn attack_detector_ignores_a_steady_tone() {
        let (dt, sr) = (1, 4);
        let ns = NS[dt][sr];
        let signal = sine(1000.0, sr, ns, 10_000.0);
        let mut detector = AttackDetector::new(dt, sr);

        // The first frame starts with a zero envelope and may report an
        // attack; a steady tone must not retrigger afterwards.
        let flags: Vec<bool> = (0..10).map(|_| detector.run(&signal, 120)).collect();
        assert!(
            flags[0],
            "the first frame is expected to start the envelope"
        );
        assert!(
            flags[1..].iter().all(|&flag| !flag),
            "unexpected retrigger: {flags:?}"
        );
    }

    #[test]
    fn attack_detector_flags_a_step() {
        let (dt, sr) = (1, 4);
        let ns = NS[dt][sr];
        let silence = vec![0f32; ns];
        let loud = sine(1000.0, sr, ns, 20_000.0);
        let mut detector = AttackDetector::new(dt, sr);

        assert!(!detector.run(&silence, 120));
        assert!(!detector.run(&silence, 120));
        assert!(detector.run(&loud, 120));

        // A step inside one frame, with silence in the first half, also
        // triggers at the transition.
        let mut step = vec![0f32; ns];
        step[ns / 2..].copy_from_slice(&loud[..ns - ns / 2]);
        let mut detector = AttackDetector::new(dt, sr);
        assert!(detector.run(&step, 120));
    }

    #[test]
    fn attack_detector_enabling_follows_the_frame_size() {
        let cases = [
            (1usize, 3usize, 80usize, false),
            (1, 3, 81, true),
            (1, 4, 99, false),
            (1, 4, 100, true),
            (0, 3, 60, false),
            (0, 3, 61, true),
            (0, 3, 149, true),
            (0, 3, 150, false),
            (0, 4, 74, false),
            (0, 4, 75, true),
            (0, 4, 149, true),
            (0, 4, 150, false),
        ];
        for &(dt, sr, nbytes, expected) in &cases {
            let ns = NS[dt][sr];
            let signal = sine(1000.0, sr, ns, 20_000.0);
            let mut detector = AttackDetector::new(dt, sr);
            assert_eq!(
                detector.run(&signal, nbytes),
                expected,
                "dt {dt}, sr {sr}, nbytes {nbytes}"
            );
        }
    }

    #[test]
    fn attack_detector_is_inactive_and_keeps_its_state_below_32k() {
        let (dt, sr) = (1, 1);
        let ns = NS[dt][sr];
        let signal = sine(1000.0, sr, ns, 20_000.0);
        let mut detector = AttackDetector::new(dt, sr);

        let before = (
            detector.prev_x1,
            detector.prev_x2,
            detector.prev_a,
            detector.prev_e,
            detector.prev_p,
        );
        for _ in 0..4 {
            assert!(!detector.run(&signal, 400));
        }
        let after = (
            detector.prev_x1,
            detector.prev_x2,
            detector.prev_a,
            detector.prev_e,
            detector.prev_p,
        );
        assert_eq!(before, after);
        assert_eq!(before.4, -1);
    }

    #[test]
    fn attack_detector_is_deterministic() {
        let (dt, sr) = (1, 4);
        let ns = NS[dt][sr];
        let mut signal = sine(1000.0, sr, ns, 10_000.0);
        signal[ns / 2] = 30_000.0;
        let mut first = AttackDetector::new(dt, sr);
        let mut second = AttackDetector::new(dt, sr);

        for _ in 0..6 {
            assert_eq!(first.run(&signal, 120), second.run(&signal, 120));
        }
        assert_eq!(first.prev_a, second.prev_a);
        assert_eq!(first.prev_e, second.prev_e);
        assert_eq!(first.prev_p, second.prev_p);
        assert_eq!(first.prev_x1, second.prev_x1);
        assert_eq!(first.prev_x2, second.prev_x2);
    }

    #[test]
    fn detectors_are_panic_free_for_invalid_inputs() {
        let mut detector = AttackDetector::new(5, 9);
        assert!(!detector.run(&[], 400));
        assert!(!detector.run(&[1.0, 2.0], 20));
        assert!(band_energies(&[], 2, 0).eb.is_empty());
        assert!(band_energies(&[1.0; 4], 0, 5).eb.is_empty());
    }
}
