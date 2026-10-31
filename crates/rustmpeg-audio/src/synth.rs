// SPDX-License-Identifier: Apache-2.0
//! Deterministic procedural audio synthesis.
//!
//! This module renders short stereo clips from a 64-bit seed. It exists to give
//! the FLAC encoder and listening tests varied, reproducible material: every
//! parameter of every voice (root note, scale, chord, tempo, envelope,
//! modulation depth, noise colour, stereo placement) is derived from the seed,
//! so two seeds do not produce the same signal while a recorded seed always
//! reproduces the same samples.
//!
//! Seven families are available: [`Family::Pad`], [`Family::Pluck`],
//! [`Family::Percussion`], [`Family::Glitch`], [`Family::Ambient`],
//! [`Family::Sweep`], and [`Family::Chord`]. Rendering is dependency-free and
//! works at 44.1 kHz or 48 kHz in 16-bit or 24-bit stereo.
//!
//! ```no_run
//! use rustmpeg_audio::synth::{Family, SynthConfig, render};
//!
//! let config = SynthConfig {
//!     seed: 0x5eed,
//!     family: Family::Pad,
//!     ..SynthConfig::default()
//! };
//! let output = render(&config).unwrap();
//! let bytes = output.wav_bytes();
//! # let _ = bytes;
//! ```

use thiserror::Error;

/// Errors returned by the synthesiser.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SynthError {
    /// A configuration field is outside the supported range.
    #[error("invalid synthesis configuration: {0}")]
    InvalidConfig(&'static str),
    /// The requested duration produces no samples.
    #[error("synthesis duration is too short")]
    Empty,
}

/// Sound family rendered by [`render`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Slowly evolving detuned pad chord.
    Pad,
    /// FM plucks and bells with exponential decay.
    Pluck,
    /// Kick, snare, and hat pattern built from noise bursts and pitched drops.
    Percussion,
    /// Abrupt digital glitches, sample-and-hold tones, and bit crushing.
    Glitch,
    /// Coloured noise bed, drone chord, sparse bells, and a feedback delay.
    Ambient,
    /// Exponential frequency sweep or chirp.
    Sweep,
    /// Chord progression played as an arpeggio with a bass line.
    Chord,
}

impl Family {
    /// Every family, in declaration order.
    pub const ALL: [Self; 7] = [
        Self::Pad,
        Self::Pluck,
        Self::Percussion,
        Self::Glitch,
        Self::Ambient,
        Self::Sweep,
        Self::Chord,
    ];

    /// Stable lower-case name used on the command line.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pad => "pad",
            Self::Pluck => "pluck",
            Self::Percussion => "percussion",
            Self::Glitch => "glitch",
            Self::Ambient => "ambient",
            Self::Sweep => "sweep",
            Self::Chord => "chord",
        }
    }

    /// Parses a family name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|family| family.as_str().eq_ignore_ascii_case(name))
    }

    /// Chooses a family deterministically from a seed.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        Self::ALL[(seed.rotate_left(17) % Self::ALL.len() as u64) as usize]
    }
}

/// Parameters for one render.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SynthConfig {
    /// 64-bit seed. Every parameter is derived from it.
    pub seed: u64,
    /// Sound family.
    pub family: Family,
    /// Sample rate in hertz; 44100 or 48000.
    pub sample_rate: u32,
    /// Bit depth; 16 or 24.
    pub bits_per_sample: u8,
    /// Duration in seconds, from 0.05 to 600.
    pub duration_seconds: f64,
    /// Channel count; only stereo is supported.
    pub channels: u8,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            seed: 1,
            family: Family::Pad,
            sample_rate: 48_000,
            bits_per_sample: 16,
            duration_seconds: 4.0,
            channels: 2,
        }
    }
}

impl SynthConfig {
    /// Validates every field.
    pub fn validate(&self) -> Result<(), SynthError> {
        if !matches!(self.sample_rate, 44_100 | 48_000) {
            return Err(SynthError::InvalidConfig(
                "sample rate must be 44100 or 48000 Hz",
            ));
        }
        if !matches!(self.bits_per_sample, 16 | 24) {
            return Err(SynthError::InvalidConfig("bit depth must be 16 or 24"));
        }
        if self.channels != 2 {
            return Err(SynthError::InvalidConfig("only stereo is supported"));
        }
        if !(0.05..=600.0).contains(&self.duration_seconds) {
            return Err(SynthError::InvalidConfig(
                "duration must be between 0.05 and 600 seconds",
            ));
        }
        Ok(())
    }

    /// Number of samples per channel.
    #[must_use]
    pub fn frames(&self) -> usize {
        (self.duration_seconds * f64::from(self.sample_rate)).round() as usize
    }
}

/// Rendered planar audio plus the parameters that produced it.
#[derive(Debug, Clone)]
pub struct SynthOutput {
    /// Planar samples, one vector per channel.
    pub planes: Vec<Vec<i32>>,
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Bit depth.
    pub bits_per_sample: u8,
    /// Family that was rendered.
    pub family: Family,
    /// Seed that was rendered.
    pub seed: u64,
    /// Human-readable summary of the derived parameters.
    pub description: String,
}

impl SynthOutput {
    /// Samples per channel.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.planes.first().map_or(0, Vec::len)
    }

    /// Interleaved little-endian PCM in the declared bit depth.
    #[must_use]
    pub fn interleaved_bytes(&self) -> Vec<u8> {
        let width = usize::from(self.bits_per_sample).div_ceil(8);
        let mut bytes = Vec::with_capacity(self.frames() * self.planes.len() * width);
        for index in 0..self.frames() {
            for plane in &self.planes {
                let sample = plane[index];
                for byte in 0..width {
                    bytes.push((sample >> (8 * byte)) as u8);
                }
            }
        }
        bytes
    }

    /// A complete RIFF/WAVE file with 16-bit or 24-bit PCM.
    #[must_use]
    pub fn wav_bytes(&self) -> Vec<u8> {
        let pcm = self.interleaved_bytes();
        let channels = self.planes.len() as u16;
        let width = u16::from(self.bits_per_sample).div_ceil(8);
        let block_align = channels * width;
        let byte_rate = self.sample_rate * u32::from(block_align);
        let mut wav = Vec::with_capacity(pcm.len() + 44);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&self.sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&u16::from(self.bits_per_sample).to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        wav.extend_from_slice(&pcm);
        wav
    }
}

/// Renders one clip.
pub fn render(config: &SynthConfig) -> Result<SynthOutput, SynthError> {
    config.validate()?;
    let frames = config.frames();
    if frames == 0 {
        return Err(SynthError::Empty);
    }
    let mut render = Render::new(
        config.seed,
        config.sample_rate,
        frames,
        config.duration_seconds,
    );
    let description = match config.family {
        Family::Pad => render.pad(),
        Family::Pluck => render.pluck(),
        Family::Percussion => render.percussion(),
        Family::Glitch => render.glitch(),
        Family::Ambient => render.ambient(),
        Family::Sweep => render.sweep(),
        Family::Chord => render.chord(),
    };
    render.finish();
    let planes = render.to_planes(config.bits_per_sample);
    Ok(SynthOutput {
        planes,
        sample_rate: config.sample_rate,
        bits_per_sample: config.bits_per_sample,
        family: config.family,
        seed: config.seed,
        description,
    })
}

/* ------------------------------------------------------------------------- */
/* Deterministic pseudo-random source                                         */
/* ------------------------------------------------------------------------- */

/// SplitMix64. Small, fast, and fully specified by its arithmetic.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.unit()
    }

    fn index(&mut self, len: usize) -> usize {
        (self.next_u64() % len as u64) as usize
    }

    fn chance(&mut self, probability: f64) -> bool {
        self.unit() < probability
    }

    fn signed(&mut self, scale: f64) -> f64 {
        (self.unit() * 2.0 - 1.0) * scale
    }
}

/* ------------------------------------------------------------------------- */
/* Musical material                                                           */
/* ------------------------------------------------------------------------- */

const SCALES: [&[i32]; 7] = [
    &[0, 2, 4, 5, 7, 9, 11], // major
    &[0, 2, 3, 5, 7, 8, 10], // natural minor
    &[0, 2, 4, 7, 9],        // major pentatonic
    &[0, 3, 5, 7, 10],       // minor pentatonic
    &[0, 2, 3, 5, 7, 9, 10], // dorian
    &[0, 1, 3, 5, 6, 8, 10], // locrian
    &[0, 2, 4, 6, 8, 10],    // whole tone
];

fn midi_to_hz(midi: f64) -> f64 {
    440.0 * (2.0f64).powf((midi - 69.0) / 12.0)
}

fn scale_note(root: i32, scale: &[i32], degree: usize) -> f64 {
    let step = scale[degree % scale.len()];
    let octave = (degree / scale.len()) as i32 * 12;
    midi_to_hz(f64::from(root + step + octave))
}

fn smoothstep(value: f64) -> f64 {
    let x = value.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/* ------------------------------------------------------------------------- */
/* Wavetable                                                                  */
/* ------------------------------------------------------------------------- */

/// One cycle of a band-limited waveform, sampled by phase in `[0, 1)`.
struct Table {
    values: Vec<f64>,
}

impl Table {
    fn new(kind: Waveform) -> Self {
        let size = 2048usize;
        let mut values = vec![0.0f64; size];
        for (index, value) in values.iter_mut().enumerate() {
            let phase = index as f64 / size as f64;
            *value = match kind {
                Waveform::Sine => (std::f64::consts::TAU * phase).sin(),
                Waveform::Triangle => {
                    let mut sum = 0.0;
                    let mut harmonic = 1;
                    while harmonic <= 31 {
                        let sign = if (harmonic / 2) % 2 == 0 { 1.0 } else { -1.0 };
                        sum += sign * (std::f64::consts::TAU * phase * harmonic as f64).sin()
                            / (harmonic * harmonic) as f64;
                        harmonic += 2;
                    }
                    sum * (8.0 / (std::f64::consts::PI * std::f64::consts::PI))
                }
                Waveform::Saw => {
                    let mut sum = 0.0;
                    for harmonic in 1..=48 {
                        sum += (std::f64::consts::TAU * phase * harmonic as f64).sin()
                            / harmonic as f64;
                    }
                    sum * (2.0 / std::f64::consts::PI)
                }
            };
        }
        // Normalize to a peak of one so every table mixes predictably.
        let peak = values
            .iter()
            .fold(0.0f64, |peak, value| peak.max(value.abs()));
        if peak > 0.0 {
            for value in &mut values {
                *value /= peak;
            }
        }
        Self { values }
    }

    fn at(&self, phase: f64) -> f64 {
        let scaled = (phase - phase.floor()) * self.values.len() as f64;
        let index = scaled as usize % self.values.len();
        let next = (index + 1) % self.values.len();
        let fraction = scaled - scaled.floor();
        self.values[index] * (1.0 - fraction) + self.values[next] * fraction
    }
}

#[derive(Clone, Copy)]
enum Waveform {
    Sine,
    Triangle,
    Saw,
}

/* ------------------------------------------------------------------------- */
/* Render canvas                                                              */
/* ------------------------------------------------------------------------- */

struct Render {
    sample_rate: f64,
    frames: usize,
    duration: f64,
    left: Vec<f64>,
    right: Vec<f64>,
    rng: Rng,
}

impl Render {
    fn new(seed: u64, sample_rate: u32, frames: usize, duration: f64) -> Self {
        Self {
            sample_rate: f64::from(sample_rate),
            frames,
            duration,
            left: vec![0.0; frames],
            right: vec![0.0; frames],
            rng: Rng::new(seed),
        }
    }

    fn add(&mut self, index: usize, left: f64, right: f64) {
        if index < self.frames {
            self.left[index] += left;
            self.right[index] += right;
        }
    }

    fn add_mono(&mut self, index: usize, value: f64, pan: f64) {
        let angle = (pan.clamp(-1.0, 1.0) + 1.0) * std::f64::consts::FRAC_PI_4;
        self.add(index, value * angle.cos(), value * angle.sin());
    }

    fn finish(&mut self) {
        // Remove DC offset, then normalize to a consistent headroom.
        let frames = self.frames as f64;
        let left_mean = self.left.iter().sum::<f64>() / frames;
        let right_mean = self.right.iter().sum::<f64>() / frames;
        for index in 0..self.frames {
            self.left[index] -= left_mean;
            self.right[index] -= right_mean;
        }
        let peak = self
            .left
            .iter()
            .chain(self.right.iter())
            .fold(0.0f64, |peak, value| peak.max(value.abs()));
        if peak > 1.0e-9 {
            let gain = 0.89 / peak;
            for value in &mut self.left {
                *value *= gain;
            }
            for value in &mut self.right {
                *value *= gain;
            }
        }
    }

    fn to_planes(&self, bits: u8) -> Vec<Vec<i32>> {
        let limit = (1i64 << (bits - 1)) as f64;
        let convert = |value: f64| -> i32 {
            let scaled = (value.clamp(-1.0, 1.0) * limit).round();
            let bounded = scaled.clamp(-limit, limit - 1.0) as i64;
            bounded as i32
        };
        vec![
            self.left.iter().map(|&value| convert(value)).collect(),
            self.right.iter().map(|&value| convert(value)).collect(),
        ]
    }

    fn root_and_scale(&mut self) -> (i32, &'static [i32]) {
        let root = 33 + self.rng.index(18) as i32; // A1..F#3
        let scale = SCALES[self.rng.index(SCALES.len())];
        (root, scale)
    }

    /// Slow detuned pad chord.
    fn pad(&mut self) -> String {
        let (root, scale) = self.root_and_scale();
        let start = self.rng.index(scale.len());
        let notes = 3 + self.rng.index(3);
        let attack = self.rng.range(0.35, 1.8);
        let release = self.rng.range(0.8, 2.8).min(self.duration * 0.6);
        let detune = self.rng.range(2.0, 16.0);
        let voices = 2 + self.rng.index(2);
        let lfo_rate = self.rng.range(0.04, 0.35);
        let lfo_depth = self.rng.range(0.08, 0.35);
        let cutoff = self.rng.range(500.0, 4200.0);
        let cutoff_lfo = self.rng.range(0.02, 0.12);
        let table = Table::new(if self.rng.chance(0.6) {
            Waveform::Saw
        } else {
            Waveform::Triangle
        });
        let sine = Table::new(Waveform::Sine);
        let mut frequency = Vec::with_capacity(notes);
        for note in 0..notes {
            frequency.push(scale_note(root, scale, start + note * 2));
        }
        let mut phases = vec![0.0f64; notes * voices];
        let mut offsets = vec![0.0f64; notes * voices];
        for (voice, offset) in offsets.iter_mut().enumerate() {
            *offset = self.rng.signed(detune) / 1200.0;
            phases[voice] = self.rng.unit();
        }
        let mut filter = 0.0f64;
        for index in 0..self.frames {
            let time = index as f64 / self.sample_rate;
            let envelope = smoothstep(time / attack)
                * smoothstep((self.duration - time) / release.max(1.0e-3));
            let lfo = (std::f64::consts::TAU * lfo_rate * time).sin();
            let cutoff_now =
                cutoff * (1.0 + 0.6 * (std::f64::consts::TAU * cutoff_lfo * time).sin());
            let coefficient = 1.0 - (-std::f64::consts::TAU * cutoff_now / self.sample_rate).exp();
            let mut value = 0.0f64;
            for (voice, phase) in phases.iter_mut().enumerate() {
                let note = voice / voices;
                let detuned = frequency[note] * (2.0f64).powf(offsets[voice]);
                *phase += detuned / self.sample_rate;
                let tide = sine.at(*phase * 0.5) * 0.18;
                value += table.at(*phase + tide) / (note + 1) as f64;
            }
            filter += coefficient * (value - filter);
            let gain = envelope * (0.55 + lfo_depth * lfo);
            let pan = lfo * 0.35;
            self.add_mono(index, filter * gain * 0.55, pan);
        }
        format!(
            "pad root midi {root}, {} voices, detune {detune:.1} cents, attack {attack:.2}s, cutoff {cutoff:.0} Hz",
            notes * voices
        )
    }

    /// FM plucks and bells with exponential decay.
    fn pluck(&mut self) -> String {
        let (root, scale) = self.root_and_scale();
        let start = self.rng.index(scale.len());
        let tempo = self.rng.range(68.0, 148.0);
        let subdivision = [2.0, 3.0, 4.0][self.rng.index(3)];
        let step = 60.0 / tempo / subdivision;
        let ratio = [1.4, 2.0, 2.76, 3.5, 4.07, 5.43, 7.0][self.rng.index(7)];
        let index_depth = self.rng.range(1.0, 7.0);
        let index_tau = self.rng.range(0.03, 0.25);
        let decay = self.rng.range(0.35, 2.4);
        let brightness = self.rng.range(0.2, 0.7);
        let table = Table::new(Waveform::Sine);
        let mut time = self.rng.range(0.0, step);
        let mut step_index = 0usize;
        let mut note = start;
        let mut first_note = true;
        while time < self.duration {
            let gate = std::mem::take(&mut first_note) || self.rng.chance(0.72);
            if gate {
                let frequency = scale_note(root, scale, note);
                let pan = (step_index % 5) as f64 / 4.0 * 1.4 - 0.7;
                let velocity = self.rng.range(0.35, 1.0);
                let tau = decay * self.rng.range(0.7, 1.3);
                let start_index = (time * self.sample_rate) as usize;
                let length = ((tau * 5.0) * self.sample_rate) as usize;
                let mut carrier = self.rng.unit();
                let mut modulator = self.rng.unit();
                for offset in 0..length {
                    let index = start_index + offset;
                    if index >= self.frames {
                        break;
                    }
                    let local = offset as f64 / self.sample_rate;
                    let decay_env = (-local / tau).exp();
                    let attack = smoothstep(local / 0.004);
                    let modulation = index_depth * (-local / index_tau).exp();
                    modulator += frequency * ratio / self.sample_rate;
                    carrier += frequency / self.sample_rate;
                    let phase = carrier + modulation * table.at(modulator) / std::f64::consts::TAU;
                    let sample = table.at(phase) * decay_env * attack * velocity;
                    self.add_mono(index, sample * (0.6 + brightness * 0.4), pan);
                }
            }
            note += 1 + self.rng.index(3);
            time += step * if self.rng.chance(0.15) { 2.0 } else { 1.0 };
            step_index += 1;
        }
        format!(
            "pluck tempo {tempo:.0} BPM, 1/{subdivision:.0} grid, FM ratio {ratio:.2}, index {index_depth:.2}, decay {decay:.2}s"
        )
    }

    /// Noise and pitched percussion pattern.
    fn percussion(&mut self) -> String {
        let tempo = self.rng.range(84.0, 168.0);
        let step = 60.0 / tempo / 4.0;
        let swing = self.rng.range(0.0, 0.25);
        let kick_probability = self.rng.range(0.25, 0.55);
        let snare_probability = self.rng.range(0.12, 0.35);
        let hat_probability = self.rng.range(0.4, 0.85);
        let kick_tune = self.rng.range(38.0, 62.0);
        let snare_tone = self.rng.range(160.0, 280.0);
        let hat_cutoff = self.rng.range(3000.0, 9000.0);
        let steps = (self.duration / step).ceil() as usize;
        for step_index in 0..steps {
            let swing_offset = if step_index % 2 == 1 {
                swing * step
            } else {
                0.0
            };
            let time = step_index as f64 * step + swing_offset;
            if time >= self.duration {
                break;
            }
            if step_index == 0 || (step_index % 4 == 0 && self.rng.chance(kick_probability)) {
                self.kick(time, kick_tune);
            }
            if (step_index % 4 == 2 || self.rng.chance(0.08)) && self.rng.chance(snare_probability)
            {
                self.snare(time, snare_tone);
            }
            if self.rng.chance(hat_probability) {
                let open = self.rng.chance(0.15);
                self.hat(time, hat_cutoff, open);
            }
        }
        format!(
            "percussion {tempo:.0} BPM, swing {swing:.2}, kick {kick_tune:.0} Hz, snare {snare_tone:.0} Hz, hats {hat_cutoff:.0} Hz"
        )
    }

    fn kick(&mut self, time: f64, tune: f64) {
        let start = (time * self.sample_rate) as usize;
        let decay = self.rng.range(0.10, 0.32);
        let length = (decay * 6.0 * self.sample_rate) as usize;
        let mut phase = 0.0f64;
        let pan = self.rng.signed(0.15);
        for offset in 0..length {
            let index = start + offset;
            if index >= self.frames {
                break;
            }
            let local = offset as f64 / self.sample_rate;
            let frequency = tune + 95.0 * (-local / 0.045).exp();
            phase += frequency / self.sample_rate;
            let sample = (std::f64::consts::TAU * phase).sin()
                * (-local / decay).exp()
                * smoothstep(local / 0.002);
            self.add_mono(index, sample * 0.95, pan);
        }
    }

    fn snare(&mut self, time: f64, tone: f64) {
        let start = (time * self.sample_rate) as usize;
        let decay = self.rng.range(0.07, 0.22);
        let length = (decay * 7.0 * self.sample_rate) as usize;
        let mut noise_state = self.rng.next_u64();
        let mut highpass = 0.0f64;
        let mut previous = 0.0f64;
        let pan = self.rng.signed(0.3);
        let mut phase = 0.0f64;
        for offset in 0..length {
            let index = start + offset;
            if index >= self.frames {
                break;
            }
            let local = offset as f64 / self.sample_rate;
            noise_state = noise_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = ((noise_state >> 33) as f64 / 2_147_483_648.0) - 1.0;
            highpass = 0.92 * (highpass + noise - previous);
            previous = noise;
            phase += tone / self.sample_rate;
            let envelope = (-local / decay).exp() * smoothstep(local / 0.001);
            let sample = highpass * 0.8 + (std::f64::consts::TAU * phase).sin() * 0.25;
            self.add_mono(index, sample * envelope, pan);
        }
    }

    fn hat(&mut self, time: f64, cutoff: f64, open: bool) {
        let start = (time * self.sample_rate) as usize;
        let decay = if open {
            self.rng.range(0.08, 0.2)
        } else {
            self.rng.range(0.012, 0.05)
        };
        let length = (decay * 6.0 * self.sample_rate) as usize;
        let mut noise_state = self.rng.next_u64();
        let mut highpass = 0.0f64;
        let coefficient = 1.0 - (-std::f64::consts::TAU * cutoff / self.sample_rate).exp();
        let pan = self.rng.range(-0.55, 0.55);
        let gain = if open { 0.25 } else { 0.4 };
        for offset in 0..length {
            let index = start + offset;
            if index >= self.frames {
                break;
            }
            let local = offset as f64 / self.sample_rate;
            noise_state = noise_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = ((noise_state >> 33) as f64 / 2_147_483_648.0) - 1.0;
            highpass += coefficient * (noise - highpass);
            let value = noise - highpass;
            let envelope = (-local / decay).exp() * smoothstep(local / 0.0005);
            self.add_mono(index, value * envelope * gain, pan);
        }
    }

    /// Abrupt digital glitch texture.
    fn glitch(&mut self) -> String {
        let bit_depth = 3 + self.rng.index(5) as u32;
        let ring = self.rng.range(30.0, 900.0);
        let base = self.rng.range(60.0, 1200.0);
        let mut index = 0usize;
        let mut segments = 0usize;
        let mut held_noise = 0.0f64;
        let mut held_phase = self.rng.unit();
        while index < self.frames {
            let length = (self.rng.range(0.015, 0.18) * self.sample_rate) as usize;
            let mode = if segments == 0 {
                1 + self.rng.index(4)
            } else {
                self.rng.index(5)
            };
            let pan = self.rng.range(-0.8, 0.8);
            let gain = self.rng.range(0.2, 0.9);
            let segment_phase = self.rng.unit();
            for offset in 0..length {
                let position = index + offset;
                if position >= self.frames {
                    break;
                }
                let local = offset as f64 / self.sample_rate;
                let value = match mode {
                    0 => 0.0,
                    1 => {
                        held_phase += base / self.sample_rate;
                        quantize(
                            (std::f64::consts::TAU * held_phase + segment_phase).sin(),
                            bit_depth,
                        ) * (1.0 - local * 4.0).max(0.0)
                    }
                    2 => {
                        if offset % 64 == 0 {
                            held_noise = self.rng.signed(1.0);
                        }
                        quantize(held_noise, bit_depth)
                    }
                    3 => {
                        held_phase += base / self.sample_rate;
                        let carrier = (std::f64::consts::TAU * held_phase).sin();
                        let modulator = (std::f64::consts::TAU * ring * local).sin();
                        quantize(carrier * modulator, bit_depth)
                    }
                    _ => {
                        if offset % 128 == 0 {
                            held_noise = self.rng.signed(1.0);
                        }
                        let gate = if (offset / 900) % 2 == 0 { 1.0 } else { 0.0 };
                        quantize(held_noise, bit_depth) * gate
                    }
                };
                self.add_mono(position, value * gain, pan);
            }
            index += length;
            segments += 1;
            if self.rng.chance(0.18) {
                index += (self.rng.range(0.005, 0.05) * self.sample_rate) as usize;
            }
        }
        format!(
            "glitch {segments} segments, {bit_depth}-bit crush, ring {ring:.0} Hz, base {base:.0} Hz"
        )
    }

    /// Coloured noise bed, drone chord, sparse bells, and a feedback delay.
    fn ambient(&mut self) -> String {
        let (root, scale) = self.root_and_scale();
        let start = self.rng.index(scale.len());
        let drone_notes = 2 + self.rng.index(3);
        let noise_cutoff = self.rng.range(300.0, 3800.0);
        let noise_level = self.rng.range(0.12, 0.4);
        let drift = self.rng.range(0.01, 0.08);
        let bell_interval = self.rng.range(0.9, 3.2);
        let delay = self.rng.range(0.18, 0.55);
        let feedback = self.rng.range(0.25, 0.58);
        let sine = Table::new(Waveform::Sine);
        let mut drone_phase = vec![0.0f64; drone_notes];
        let mut drone_frequency = Vec::with_capacity(drone_notes);
        for note in 0..drone_notes {
            drone_frequency.push(scale_note(root + 12, scale, start + note * 2));
        }
        let mut noise_states = [self.rng.next_u64(), self.rng.next_u64()];
        let mut filters = [0.0f64; 2];
        let mut bell_time = self.rng.range(0.2, 1.0);
        let mut bell_count = 0usize;
        for index in 0..self.frames {
            let time = index as f64 / self.sample_rate;
            let drift_lfo = (std::f64::consts::TAU * drift * time).sin();
            let coefficient = 1.0
                - (-std::f64::consts::TAU * noise_cutoff * (1.0 + 0.5 * drift_lfo)
                    / self.sample_rate)
                    .exp();
            let mut noise = [0.0f64; 2];
            for channel in 0..2 {
                noise_states[channel] = noise_states[channel]
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let white = ((noise_states[channel] >> 33) as f64 / 2_147_483_648.0) - 1.0;
                filters[channel] += coefficient * (white - filters[channel]);
                noise[channel] = filters[channel];
            }
            let mut drone = 0.0f64;
            for (note, phase) in drone_phase.iter_mut().enumerate() {
                let detune = 1.0 + 0.0015 * (note as f64 + 1.0) * drift_lfo;
                *phase += drone_frequency[note] * detune / self.sample_rate;
                drone += sine.at(*phase) / (note + 1) as f64;
            }
            let bed = 0.55 + 0.45 * drift_lfo;
            self.add(
                index,
                noise[0] * noise_level * bed + drone * 0.22,
                noise[1] * noise_level * bed + drone * 0.22,
            );
            if time >= bell_time {
                let bell_degree = start + self.rng.index(scale.len()) * 2;
                self.bell(time, root, scale, bell_degree);
                bell_time += bell_interval * self.rng.range(0.7, 1.5);
                bell_count += 1;
            }
        }
        apply_delay(
            &mut self.left,
            &mut self.right,
            (delay * self.sample_rate) as usize,
            feedback,
        );
        format!(
            "ambient {drone_notes}-note drone, noise cutoff {noise_cutoff:.0} Hz, delay {delay:.2}s, {bell_count} bells"
        )
    }

    fn bell(&mut self, time: f64, root: i32, scale: &[i32], degree: usize) {
        let start = (time * self.sample_rate) as usize;
        let frequency = scale_note(root + 24, scale, degree);
        let ratio = [1.41, 2.0, 2.76, 3.5, 5.43][self.rng.index(5)];
        let decay = self.rng.range(1.2, 3.5);
        let length = (decay * 3.0 * self.sample_rate) as usize;
        let pan = self.rng.range(-0.7, 0.7);
        let mut carrier = self.rng.unit();
        let mut modulator = self.rng.unit();
        for offset in 0..length {
            let index = start + offset;
            if index >= self.frames {
                break;
            }
            let local = offset as f64 / self.sample_rate;
            let envelope = (-local / decay).exp() * smoothstep(local / 0.01);
            let modulation = 2.5 * (-local / 0.3).exp();
            modulator += frequency * ratio / self.sample_rate;
            carrier += frequency / self.sample_rate;
            let sample = (std::f64::consts::TAU * carrier
                + modulation * (std::f64::consts::TAU * modulator).sin())
            .sin();
            self.add_mono(index, sample * envelope * 0.28, pan);
        }
    }

    /// Exponential sweep with optional second layer and vibrato.
    fn sweep(&mut self) -> String {
        let low = self.rng.range(70.0, 520.0);
        let high = self.rng.range(1400.0, 9500.0);
        let down = self.rng.chance(0.3);
        let (start_hz, end_hz) = if down { (high, low) } else { (low, high) };
        let repeats = 1 + self.rng.index(3);
        let layer_detune = self.rng.range(4.0, 40.0);
        let vibrato_rate = self.rng.range(2.0, 9.0);
        let vibrato_depth = self.rng.range(0.0, 0.02);
        let fade = self.rng.range(0.05, 0.4);
        let sine = Table::new(Waveform::Sine);
        for repeat in 0..repeats {
            let offset_time = repeat as f64 * self.duration / repeats as f64;
            let length = self.duration / repeats as f64;
            let start = (offset_time * self.sample_rate) as usize;
            let count = (length * self.sample_rate) as usize;
            let mut phase = self.rng.unit();
            let mut phase_b = self.rng.unit();
            let pan = self.rng.range(-0.6, 0.6);
            for step in 0..count {
                let index = start + step;
                if index >= self.frames {
                    break;
                }
                let local = step as f64 / self.sample_rate;
                let fraction = local / length;
                let frequency = start_hz * (end_hz / start_hz).powf(fraction);
                let vibrato =
                    1.0 + vibrato_depth * (std::f64::consts::TAU * vibrato_rate * local).sin();
                phase += frequency * vibrato / self.sample_rate;
                phase_b += frequency * (1.0 + layer_detune / 1200.0) / self.sample_rate;
                let envelope = smoothstep(local / fade) * smoothstep((length - local) / fade);
                let value = sine.at(phase) * 0.7 + sine.at(phase_b) * 0.3;
                self.add_mono(index, value * envelope * 0.7, pan);
            }
        }
        format!(
            "sweep {start_hz:.0}->{end_hz:.0} Hz, {repeats} pass(es), vibrato {vibrato_rate:.1} Hz, layer detune {layer_detune:.1} cents"
        )
    }

    /// Chord progression played as an arpeggio with a bass line.
    fn chord(&mut self) -> String {
        let (root, scale) = self.root_and_scale();
        let tempo = self.rng.range(78.0, 160.0);
        let beat = 60.0 / tempo;
        let chord_length = [2.0, 4.0, 8.0][self.rng.index(3)];
        let note_table = Table::new(if self.rng.chance(0.5) {
            Waveform::Triangle
        } else {
            Waveform::Sine
        });
        let bass_table = Table::new(Waveform::Sine);
        let arp_pattern = self.rng.index(4);
        let note_decay = self.rng.range(0.25, 1.4);
        let bass_decay = self.rng.range(0.3, 1.6);
        let mut chord_index = 0usize;
        let mut time = 0.0f64;
        while time < self.duration {
            let degree = self.rng.index(scale.len());
            let chord_degrees = [degree, degree + 2, degree + 4, degree + 6];
            let chord_duration = chord_length * beat;
            let step = beat / 2.0;
            let steps = (chord_duration / step).ceil() as usize;
            for step_index in 0..steps {
                let note_time = time + step_index as f64 * step;
                if note_time >= self.duration {
                    break;
                }
                let position = match arp_pattern {
                    0 => step_index % chord_degrees.len(),
                    1 => chord_degrees.len() - 1 - step_index % chord_degrees.len(),
                    2 => {
                        let cycle = step_index % (chord_degrees.len() * 2 - 2);
                        if cycle < chord_degrees.len() {
                            cycle
                        } else {
                            chord_degrees.len() * 2 - 2 - cycle
                        }
                    }
                    _ => self.rng.index(chord_degrees.len()),
                };
                let frequency = scale_note(root + 12, scale, chord_degrees[position]);
                let pan = if step_index % 2 == 0 { -0.45 } else { 0.45 };
                self.pluck_note(note_time, frequency, note_decay, 0.32, pan, &note_table);
            }
            let bass_frequency = scale_note(root, scale, degree);
            let bass_pan = self.rng.range(-0.15, 0.15);
            self.pluck_note(time, bass_frequency, bass_decay, 0.5, bass_pan, &bass_table);
            time += chord_duration;
            chord_index += 1;
        }
        format!(
            "chord {tempo:.0} BPM, {chord_index} chords of {chord_length:.0} beats, arpeggio pattern {arp_pattern}, note decay {note_decay:.2}s"
        )
    }

    fn pluck_note(
        &mut self,
        time: f64,
        frequency: f64,
        decay: f64,
        gain: f64,
        pan: f64,
        table: &Table,
    ) {
        let start = (time * self.sample_rate) as usize;
        let length = (decay * 4.0 * self.sample_rate) as usize;
        let mut phase = self.rng.unit();
        for offset in 0..length {
            let index = start + offset;
            if index >= self.frames {
                break;
            }
            let local = offset as f64 / self.sample_rate;
            let envelope = (-local / decay).exp() * smoothstep(local / 0.004);
            phase += frequency / self.sample_rate;
            self.add_mono(index, table.at(phase) * envelope * gain, pan);
        }
    }
}

/// Symmetric feedback delay with a little stereo cross-feed.
fn apply_delay(left: &mut [f64], right: &mut [f64], delay: usize, feedback: f64) {
    if delay == 0 || delay >= left.len() {
        return;
    }
    for index in delay..left.len() {
        let delayed_left = left[index - delay];
        let delayed_right = right[index - delay];
        left[index] += delayed_right * 0.35 * feedback + delayed_left * feedback * 0.65;
        right[index] += delayed_left * 0.35 * feedback + delayed_right * feedback * 0.65;
    }
}

/// Quantizes to `bits` of amplitude for the glitch family.
fn quantize(value: f64, bits: u32) -> f64 {
    let levels = (1u32 << (bits - 1)) as f64;
    (value.clamp(-1.0, 1.0) * levels).round() / levels
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::flac::{FlacDecoder, FlacEncoderConfig, encode_flac_to_vec};

    fn config(seed: u64, family: Family, bits: u8) -> SynthConfig {
        SynthConfig {
            seed,
            family,
            bits_per_sample: bits,
            duration_seconds: 0.3,
            ..SynthConfig::default()
        }
    }

    #[test]
    fn rendering_is_deterministic() {
        for family in Family::ALL {
            let cfg = config(0x1234_5678_9abc_def0, family, 16);
            let first = render(&cfg).unwrap();
            let second = render(&cfg).unwrap();
            assert_eq!(first.planes, second.planes, "{}", family.as_str());
            assert_eq!(first.interleaved_bytes(), second.interleaved_bytes());
        }
    }

    #[test]
    fn every_family_produces_signal() {
        for family in Family::ALL {
            let output = render(&config(7, family, 16)).unwrap();
            assert_eq!(output.planes.len(), 2);
            assert!(output.frames() > 1_000);
            let peak = output
                .planes
                .iter()
                .flat_map(|plane| plane.iter())
                .fold(0i32, |peak, &sample| peak.max(sample.abs()));
            assert!(peak > 1_000, "{} rendered silence", family.as_str());
            assert!(!output.description.is_empty());
        }
    }

    #[test]
    fn distinct_seeds_produce_distinct_bytes() {
        let mut seen = HashSet::new();
        for family in Family::ALL {
            for seed in 0..16u64 {
                let output = render(&config(seed, family, 16)).unwrap();
                let digest = hash(&output.interleaved_bytes());
                assert!(
                    seen.insert((family.as_str(), digest)),
                    "{} seed {seed} collided",
                    family.as_str()
                );
            }
        }
    }

    #[test]
    fn families_differ_for_the_same_seed() {
        let mut seen = HashSet::new();
        for family in Family::ALL {
            let output = render(&config(99, family, 16)).unwrap();
            assert!(
                seen.insert(hash(&output.interleaved_bytes())),
                "{} duplicated another family",
                family.as_str()
            );
        }
    }

    #[test]
    fn seed_chooses_a_stable_family() {
        for seed in 0..32u64 {
            let family = Family::from_seed(seed);
            assert_eq!(family, Family::from_seed(seed));
            assert!(Family::from_name(family.as_str()).is_some());
        }
    }

    #[test]
    fn flac_roundtrip_for_every_family() {
        for family in Family::ALL {
            for bits in [16u8, 24] {
                let mut cfg_in = config(4242, family, bits);
                cfg_in.duration_seconds = 0.25;
                let output = render(&cfg_in).unwrap();
                let cfg = FlacEncoderConfig {
                    sample_rate: output.sample_rate,
                    channels: 2,
                    bits_per_sample: bits,
                    block_size: 512,
                    compression_level: 2,
                };
                let (bytes, _) = encode_flac_to_vec(&cfg, &output.planes).unwrap();
                let mut decoder = FlacDecoder::new(std::io::Cursor::new(bytes)).unwrap();
                let mut planes = vec![Vec::new(); 2];
                while let Some(frame) = decoder.next_frame().unwrap() {
                    for (dst, src) in planes.iter_mut().zip(&frame.planes) {
                        dst.extend_from_slice(src);
                    }
                }
                assert_eq!(planes, output.planes, "{} {bits}-bit", family.as_str());
            }
        }
    }

    #[test]
    fn wav_header_is_consistent() {
        let output = render(&config(5, Family::Sweep, 24)).unwrap();
        let wav = output.wav_bytes();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 24);
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            48_000
        );
        assert_eq!(
            u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]) as usize,
            wav.len() - 44
        );
    }

    fn hash(bytes: &[u8]) -> u64 {
        let mut state = 0xcbf2_9ce4_8422_2325u64;
        for &byte in bytes {
            state ^= u64::from(byte);
            state = state.wrapping_mul(0x0000_0100_0000_01b3);
        }
        state
    }
}
