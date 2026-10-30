// SPDX-License-Identifier: Apache-2.0
//! Rational polyphase resampler for the SILK output.
//!
//! The resampler itself is non-normative (RFC 6716, Section 4.2.9).  This
//! implementation uses a windowed-sinc low-pass filter with a fixed number of
//! taps per phase and a rational ratio, so the output sample count is exact
//! for every SILK frame duration.

use std::f64::consts::PI;

/// Taps per polyphase branch, in input samples.
const TAPS: usize = 12;

/// Bandwidth of the anti-imaging/anti-aliasing filter relative to the lower
/// Nyquist frequency.
const CUTOFF_SCALE: f64 = 0.92;

/// Streaming rational resampler.
#[derive(Debug, Clone)]
pub(crate) struct Resampler {
    up: usize,
    down: usize,
    phases: Vec<Vec<i32>>,
    history: Vec<i16>,
    fed: i64,
    produced: i64,
    passthrough: bool,
}

impl Resampler {
    /// Creates a resampler from an internal SILK rate to an output rate.
    pub(crate) fn new(input_rate: u32, output_rate: u32) -> Self {
        let divisor = gcd(input_rate, output_rate);
        let up = (output_rate / divisor) as usize;
        let down = (input_rate / divisor) as usize;
        let passthrough = up == down;
        let phases = if passthrough {
            Vec::new()
        } else {
            design(up, down)
        };
        Self {
            up,
            down,
            phases,
            history: Vec::new(),
            fed: 0,
            produced: 0,
            passthrough,
        }
    }

    /// Resamples one block.  `output` must have exactly
    /// `input.len() * output_rate / input_rate` samples.
    pub(crate) fn process(&mut self, input: &[i16], output: &mut [i16]) {
        if self.passthrough {
            debug_assert_eq!(input.len(), output.len());
            output.copy_from_slice(input);
            return;
        }

        let expected = input.len() * self.up / self.down;
        debug_assert_eq!(output.len(), expected);

        self.history.extend_from_slice(input);
        self.fed += input.len() as i64;

        for (offset, sample) in output.iter_mut().enumerate() {
            let index = self.produced + offset as i64;
            let scaled = index * self.down as i64;
            let phase = (scaled % self.up as i64) as usize;
            let position = scaled / self.up as i64;

            let mut accumulator = 0_i64;
            for (tap, &coefficient) in self.phases[phase].iter().enumerate() {
                accumulator +=
                    i64::from(coefficient) * i64::from(self.sample(position - tap as i64));
            }
            *sample = ((accumulator + (1 << 14)) >> 15).clamp(-32768, 32767) as i16;
        }
        self.produced += expected as i64;

        let keep = TAPS.max(1);
        if self.history.len() > keep {
            let drop = self.history.len() - keep;
            self.history.drain(..drop);
        }
    }

    fn sample(&self, index: i64) -> i16 {
        if index < 0 {
            return 0;
        }
        let base = self.fed - self.history.len() as i64;
        if index < base {
            return 0;
        }
        self.history
            .get((index - base) as usize)
            .copied()
            .unwrap_or(0)
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// Builds the Q15 polyphase filter bank.
fn design(up: usize, down: usize) -> Vec<Vec<i32>> {
    let length = up * TAPS;
    let center = (length as f64 - 1.0) / 2.0;
    let cutoff = CUTOFF_SCALE * 0.5 / up.max(down) as f64;

    let mut coefficients = vec![0.0_f64; length];
    for (index, coefficient) in coefficients.iter_mut().enumerate() {
        let position = index as f64 - center;
        let sinc = if position.abs() < 1.0e-9 {
            2.0 * cutoff
        } else {
            (2.0 * PI * cutoff * position).sin() / (PI * position)
        };
        let window = if length > 1 {
            let angle = 2.0 * PI * index as f64 / (length - 1) as f64;
            0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos()
        } else {
            1.0
        };
        *coefficient = sinc * window;
    }

    let sum: f64 = coefficients.iter().sum();
    let scale = up as f64 / sum;
    for coefficient in &mut coefficients {
        *coefficient *= scale;
    }

    let mut phases = Vec::with_capacity(up);
    for phase in 0..up {
        let mut branch = Vec::with_capacity(TAPS);
        for tap in 0..TAPS {
            let index = phase + up * tap;
            let value = (coefficients[index] * 32768.0).round();
            branch.push(value as i32);
        }
        phases.push(branch);
    }
    phases
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resample_block(input_rate: u32, output_rate: u32, input: &[i16]) -> Vec<i16> {
        let mut resampler = Resampler::new(input_rate, output_rate);
        let expected = input.len() * output_rate as usize / input_rate as usize;
        let mut output = vec![0_i16; expected];
        resampler.process(input, &mut output);
        output
    }

    #[test]
    fn constants_are_preserved() {
        for &(input_rate, output_rate) in &[
            (8000, 48000),
            (12000, 48000),
            (16000, 48000),
            (16000, 12000),
            (12000, 8000),
            (16000, 8000),
        ] {
            let input = vec![8000_i16; 480];
            let output = resample_block(input_rate, output_rate, &input);
            let settled = &output[output.len() / 2..];
            let average: f64 =
                settled.iter().map(|value| f64::from(*value)).sum::<f64>() / settled.len() as f64;
            assert!(
                (average - 8000.0).abs() < 60.0,
                "{input_rate} -> {output_rate}: average {average}"
            );
        }
    }

    #[test]
    fn sine_wave_is_preserved() {
        let input_rate = 8_000_u32;
        let output_rate = 48_000_u32;
        let frequency = 440.0_f64;
        let input: Vec<i16> = (0..800)
            .map(|index| {
                let phase =
                    2.0 * std::f64::consts::PI * frequency * index as f64 / f64::from(input_rate);
                (phase.sin() * 12000.0) as i16
            })
            .collect();
        let output = resample_block(input_rate, output_rate, &input);

        // The causal filter delays the signal; find its best alignment and
        // compare against the analytic 48 kHz sine outside the startup
        // transient.
        let expected: Vec<f64> = (0..output.len())
            .map(|index| {
                (2.0 * std::f64::consts::PI * frequency * index as f64 / f64::from(output_rate))
                    .sin()
            })
            .collect();
        let mut best = (0.0_f64, 0_usize);
        for lag in 0..80 {
            let count = output.len() - lag - 100;
            let mut numerator = 0.0;
            let mut left = 0.0;
            let mut right = 0.0;
            for index in 100..count {
                let actual = f64::from(output[index + lag]);
                let reference = expected[index];
                numerator += actual * reference;
                left += actual * actual;
                right += reference * reference;
            }
            let correlation = numerator / (left * right).sqrt();
            if correlation > best.0 {
                best = (correlation, lag);
            }
        }
        assert!(
            best.0 > 0.999,
            "8 -> 48 kHz resampler correlation is only {}",
            best.0
        );
    }

    #[test]
    fn downsampling_preserves_a_tone() {
        for &(input_rate, output_rate, frequency) in &[
            (16_000_u32, 8_000_u32, 600.0_f64),
            (16_000, 12_000, 900.0),
            (12_000, 8_000, 700.0),
        ] {
            let input: Vec<i16> = (0..960)
                .map(|index| {
                    let phase = 2.0 * std::f64::consts::PI * frequency * index as f64
                        / f64::from(input_rate);
                    (phase.sin() * 12000.0) as i16
                })
                .collect();
            let output = resample_block(input_rate, output_rate, &input);
            let mut best = (0.0_f64, 0_usize);
            for lag in 0..40 {
                let count = output.len() - lag - 80;
                let mut numerator = 0.0;
                let mut left = 0.0;
                let mut right = 0.0;
                for index in 80..count {
                    let actual = f64::from(output[index + lag]);
                    let reference = (2.0 * std::f64::consts::PI * frequency * index as f64
                        / f64::from(output_rate))
                    .sin();
                    numerator += actual * reference;
                    left += actual * actual;
                    right += reference * reference;
                }
                let correlation = numerator / (left * right).sqrt();
                if correlation > best.0 {
                    best = (correlation, lag);
                }
            }
            assert!(
                best.0 > 0.995,
                "{input_rate} -> {output_rate} Hz tone correlation is only {}",
                best.0
            );
        }
    }

    #[test]
    fn sample_count_is_exact_for_silk_frame_sizes() {
        for &(input_rate, output_rate) in &[
            (8000, 8000),
            (8000, 12000),
            (8000, 16000),
            (8000, 24000),
            (8000, 48000),
            (12000, 8000),
            (12000, 16000),
            (12000, 24000),
            (12000, 48000),
            (16000, 8000),
            (16000, 12000),
            (16000, 24000),
            (16000, 48000),
        ] {
            for &frame_ms in &[10_u32, 20] {
                let input = vec![0_i16; input_rate as usize / 1000 * frame_ms as usize];
                let output = resample_block(input_rate, output_rate, &input);
                assert_eq!(
                    output.len(),
                    output_rate as usize / 1000 * frame_ms as usize
                );
            }
        }
    }
}
