// SPDX-License-Identifier: Apache-2.0
//! LC3 (Low Complexity Communication Codec) decoder.
//!
//! Implementation of the Bluetooth LC3 v1.0.1 decoder for 7.5 ms and 10 ms
//! frames at 8, 16, 24, 32, and 48 kHz, mono or stereo. Each channel is coded
//! independently in the payload (the stereo block is the concatenation of two
//! mono frames), matching the LC3 tools and the specification's per-channel
//! decoder model.
//!
//! The normative tables are transcribed from the liblc3 reference tables
//! (Apache-2.0); see `THIRD_PARTY.md`. The decoder implementation is original
//! safe Rust and follows the decoding process of Bluetooth LC3 v1.0.1 §3.4.

mod bits;
mod bwdet;
mod enc;
mod enc_bits;
mod enc_det;
mod enc_ltpf;
mod enc_mdct;
mod enc_sns;
mod enc_spec;
mod enc_tables;
mod enc_tns;
mod ltpf;
mod mdct;
mod sns;
mod spec;
mod tables;
mod tns;

use thiserror::Error;

use crate::PcmFrame;

use self::bits::BitReader;
use self::enc::ChannelEncoder;
use self::ltpf::LtpfSynthesis;
use self::mdct::MdctInverse;
use self::sns::SnsSynthesis;
use self::spec::SpectrumDecoder;
use self::spec::band_limits;
use self::tns::TnsSynthesis;

/// LC3 frame duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lc3FrameDuration {
    /// 7.5 ms frames.
    Ms7_5,
    /// 10 ms frames.
    Ms10,
}

impl Lc3FrameDuration {
    fn index(self) -> usize {
        match self {
            Self::Ms7_5 => 0,
            Self::Ms10 => 1,
        }
    }
}

/// LC3 decoder configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lc3Config {
    /// Sampling rate in hertz: 8000, 16000, 24000, 32000, or 48000.
    pub sample_rate: u32,
    /// Frame duration.
    pub frame_duration: Lc3FrameDuration,
    /// Channel count (1 or 2).
    pub channels: u8,
    /// Output bit depth (16 or 24).
    pub bit_depth: u8,
}

impl Default for Lc3Config {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            frame_duration: Lc3FrameDuration::Ms10,
            channels: 1,
            bit_depth: 16,
        }
    }
}

impl Lc3Config {
    /// Validates the configuration and returns the sample-rate index.
    pub fn validate(&self) -> Result<usize, Lc3Error> {
        let sr = match self.sample_rate {
            8_000 => 0,
            16_000 => 1,
            24_000 => 2,
            32_000 => 3,
            48_000 => 4,
            _ => {
                return Err(Lc3Error::InvalidConfig(
                    "LC3 sample rate must be 8, 16, 24, 32, or 48 kHz",
                ));
            }
        };
        if self.channels == 0 || self.channels > 2 {
            return Err(Lc3Error::InvalidConfig("LC3 supports one or two channels"));
        }
        if !matches!(self.bit_depth, 16 | 24) {
            return Err(Lc3Error::InvalidConfig(
                "LC3 output bit depth must be 16 or 24",
            ));
        }
        Ok(sr)
    }

    /// Frame size in samples per channel.
    #[must_use]
    pub fn frame_samples(&self) -> usize {
        let sr = self.validate().unwrap_or(0);
        tables::NS[self.frame_duration.index()][sr]
    }
}

/// Errors produced by the LC3 decoder.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Lc3Error {
    /// The decoder configuration is invalid.
    #[error("invalid LC3 configuration: {0}")]
    InvalidConfig(&'static str),
    /// The payload has an invalid size or layout.
    #[error("invalid LC3 payload: {0}")]
    InvalidPayload(&'static str),
    /// The LC3 bitstream is malformed or truncated.
    #[error("LC3 bitstream error: {0}")]
    Bitstream(&'static str),
    /// The encoder cannot represent the requested frame.
    #[error("LC3 encoding error: {0}")]
    Encode(&'static str),
}

/// Packet-loss concealment state for one channel (Bluetooth LC3 v1.0.1
/// Appendix B, low-complexity concealment).
struct ChannelPlc {
    /// Sign-scrambling seed; initialized once and never reset.
    seed: u16,
    /// Number of consecutive concealed frames, including the current one.
    count: u32,
    /// Attenuation factor applied to the retained spectrum.
    alpha: f32,
    /// Last received SNS-shaped spectrum `X_lastGood` (Appendix B.3).
    last_good: Vec<f32>,
}

impl ChannelPlc {
    fn new(ne: usize) -> Self {
        Self {
            seed: 24607,
            count: 1,
            alpha: 1.0,
            last_good: vec![0.0; ne],
        }
    }

    /// Suspends concealment after an error-free frame (Appendix B.3: the
    /// attenuation factor restarts at one; the seed is not reset).
    fn suspend(&mut self) {
        self.count = 1;
        self.alpha = 1.0;
    }

    /// Retains the shaped spectrum of a good frame.
    fn retain(&mut self, x: &[f32]) {
        let count = self.last_good.len().min(x.len());
        self.last_good[..count].copy_from_slice(&x[..count]);
        self.last_good[count..].fill(0.0);
    }

    /// Produces the concealed shaped spectrum for one lost frame.
    fn synthesize(&mut self) -> Vec<f32> {
        self.alpha *= match self.count {
            0..=3 => 1.0,
            4..=7 => 0.9,
            _ => 0.85,
        };
        let mut x = Vec::with_capacity(self.last_good.len());
        for &value in &self.last_good {
            self.seed = 16_831u16.wrapping_add(self.seed.wrapping_mul(12_821));
            let sign = if self.seed & 0x8000 != 0 { -1.0 } else { 1.0 };
            x.push(self.alpha * sign * value);
        }
        self.count += 1;
        x
    }
}

/// State of one LC3 channel decoder.
struct ChannelDecoder {
    spec: SpectrumDecoder,
    tns: TnsSynthesis,
    sns: SnsSynthesis,
    mdct: MdctInverse,
    ltpf: LtpfSynthesis,
    plc: ChannelPlc,
    ns: usize,
    /// Frame size of the last good frame, used by the LTPF transition.
    nbytes: usize,
}

impl ChannelDecoder {
    fn new(dt: usize, sr: usize) -> Result<Self, Lc3Error> {
        let ns = tables::NS[dt][sr];
        let ne = *band_limits(dt, sr)
            .last()
            .ok_or(Lc3Error::InvalidConfig("missing band table"))? as usize;
        Ok(Self {
            spec: SpectrumDecoder::new(dt, sr),
            tns: TnsSynthesis::new(dt),
            sns: SnsSynthesis::new(dt, sr),
            mdct: MdctInverse::new(dt, sr)?,
            ltpf: LtpfSynthesis::new(dt, sr),
            plc: ChannelPlc::new(ne),
            ns,
            nbytes: 20,
        })
    }

    fn decode(
        &mut self,
        payload: &[u8],
        dt: usize,
        sr: usize,
        nbytes: usize,
    ) -> Result<Vec<f32>, Lc3Error> {
        let mut bits = BitReader::new(payload)?;
        let bw = bwdet::read_bandwidth(&mut bits, sr)?;
        self.spec.load(&mut bits)?;
        self.tns.load(&mut bits, bw, nbytes)?;
        let pitch_present = bits.read_bit()?;
        self.sns.load(&mut bits)?;
        if pitch_present {
            self.ltpf.load(&mut bits)?;
        } else {
            self.ltpf.disable();
        }
        let mut x = self.spec.decode(&mut bits, bw, nbytes)?;
        self.tns.run(&mut x, bw)?;
        self.sns.run(&mut x)?;
        let ns = tables::NS[dt][sr];
        if x.len() > ns {
            return Err(Lc3Error::Bitstream("spectrum exceeds frame length"));
        }
        x.resize(ns, 0.0);
        self.plc.retain(&x);
        self.plc.suspend();
        self.nbytes = nbytes;
        let y = self.mdct.run(&x)?;
        self.ltpf.run(&y, nbytes)
    }

    /// Synthesizes one concealed frame (Appendix B.3).
    ///
    /// The last good SNS-shaped spectrum is sign-scrambled and faded out, the
    /// spectrum above `NE` is zeroed, and the LTPF is forced inactive.
    fn conceal(&mut self) -> Result<Vec<f32>, Lc3Error> {
        let mut x = self.plc.synthesize();
        x.resize(self.ns, 0.0);
        self.ltpf.disable();
        let y = self.mdct.run(&x)?;
        self.ltpf.run(&y, self.nbytes)
    }
}

/// LC3 decoder with per-channel history.
pub struct Lc3Decoder {
    config: Lc3Config,
    dt: usize,
    sr: usize,
    channels: Vec<ChannelDecoder>,
    frame_number: u64,
    sample_number: u64,
}

impl Lc3Decoder {
    /// Creates a decoder for `config`.
    pub fn new(config: Lc3Config) -> Result<Self, Lc3Error> {
        let sr = config.validate()?;
        let dt = config.frame_duration.index();
        let channels = (0..config.channels)
            .map(|_| ChannelDecoder::new(dt, sr))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            config,
            dt,
            sr,
            channels,
            frame_number: 0,
            sample_number: 0,
        })
    }

    /// Returns the decoder configuration.
    #[must_use]
    pub fn config(&self) -> &Lc3Config {
        &self.config
    }

    /// Decodes one LC3 frame block into native integer PCM.
    ///
    /// For stereo, `payload` contains the two channel bitstreams concatenated;
    /// each channel occupies `payload.len() / channels` bytes.
    pub fn decode_frame(&mut self, payload: &[u8]) -> Result<PcmFrame, Lc3Error> {
        let channels = usize::from(self.config.channels);
        if payload.len() < 20 * channels {
            return Err(Lc3Error::InvalidPayload("LC3 frame is too short"));
        }
        let base = payload.len() / channels;
        let remainder = payload.len() % channels;
        if base + usize::from(remainder > 0) > 400 {
            return Err(Lc3Error::InvalidPayload("LC3 frame exceeds 400 bytes"));
        }

        let ns = tables::NS[self.dt][self.sr];
        let mut planes = Vec::with_capacity(channels);
        let mut offset = 0usize;
        for (index, decoder) in self.channels.iter_mut().enumerate() {
            let frame_bytes = base + usize::from(index < remainder);
            let start = offset;
            offset += frame_bytes;
            let pcm = decoder.decode(
                &payload[start..start + frame_bytes],
                self.dt,
                self.sr,
                frame_bytes,
            )?;
            planes.push(quantize_plane(&pcm, self.config.bit_depth));
        }
        if planes.iter().any(|plane| plane.len() != ns) {
            return Err(Lc3Error::Bitstream("unexpected LC3 output length"));
        }

        Ok(self.publish(planes))
    }

    /// Conceals one lost or corrupted frame (Bluetooth LC3 v1.0.1 Appendix B).
    ///
    /// Call this instead of [`decode_frame`](Self::decode_frame) when the
    /// receiver's bad-frame indicator is set. The method returns one frame of
    /// PCM and advances the decoder's timestamps, so a stream with losses has
    /// continuous timing:
    ///
    /// ```no_run
    /// # use rustmpeg_audio::{Lc3Config, Lc3Decoder};
    /// let config = Lc3Config::default();
    /// let mut decoder = Lc3Decoder::new(config).unwrap();
    /// # let received: Option<&[u8]> = None;
    /// let frame = match received {
    ///     Some(payload) => decoder.decode_frame(payload),
    ///     None => decoder.conceal_frame(),
    /// }
    /// .expect("valid LC3 state");
    /// # let _ = frame;
    /// ```
    ///
    /// The last received shaped spectrum is sign-scrambled with the
    /// Appendix B.3 pseudo-random sequence and faded out; after four and eight
    /// consecutive lost frames the attenuation factor is multiplied by 0.9 and
    /// 0.85 per frame respectively. The LTPF is kept inactive and the spectrum
    /// above the coded bandwidth is zeroed. The concealment seed is initialized
    /// once and is not reset by good frames.
    pub fn conceal_frame(&mut self) -> Result<PcmFrame, Lc3Error> {
        let ns = tables::NS[self.dt][self.sr];
        let mut planes = Vec::with_capacity(self.channels.len());
        for decoder in &mut self.channels {
            let pcm = decoder.conceal()?;
            planes.push(quantize_plane(&pcm, self.config.bit_depth));
        }
        if planes.iter().any(|plane| plane.len() != ns) {
            return Err(Lc3Error::Bitstream("unexpected LC3 output length"));
        }
        Ok(self.publish(planes))
    }

    /// Wraps decoded planes in a [`PcmFrame`] and advances the timestamps.
    fn publish(&mut self, planes: Vec<Vec<i32>>) -> PcmFrame {
        let ns = tables::NS[self.dt][self.sr];
        let frame = PcmFrame {
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            bits_per_sample: self.config.bit_depth,
            frame_number: self.frame_number,
            sample_number: self.sample_number,
            planes,
        };
        self.frame_number += 1;
        self.sample_number += ns as u64;
        frame
    }
}

/// Converts floating-point PCM to the configured integer bit depth.
fn quantize_plane(samples: &[f32], bit_depth: u8) -> Vec<i32> {
    match bit_depth {
        16 => samples
            .iter()
            .map(|&sample| sample.round().clamp(-32768.0, 32767.0) as i32)
            .collect(),
        _ => samples
            .iter()
            .map(|&sample| (sample * 256.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32)
            .collect(),
    }
}

/// LC3 encoder with per-channel history.
pub struct Lc3Encoder {
    config: Lc3Config,
    dt: usize,
    sr: usize,
    channels: Vec<ChannelEncoder>,
    frame_number: u64,
    sample_number: u64,
}

impl Lc3Encoder {
    /// Creates an encoder for `config`.
    pub fn new(config: Lc3Config) -> Result<Self, Lc3Error> {
        let sr = config.validate()?;
        let dt = config.frame_duration.index();
        let channels = (0..config.channels)
            .map(|_| ChannelEncoder::new(dt, sr))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            config,
            dt,
            sr,
            channels,
            frame_number: 0,
            sample_number: 0,
        })
    }

    /// Returns the encoder configuration.
    #[must_use]
    pub fn config(&self) -> &Lc3Config {
        &self.config
    }

    /// Encodes one block of planar integer PCM into an LC3 payload.
    ///
    /// For stereo the returned payload is the two per-channel bitstreams
    /// concatenated, `2 * nbytes` bytes in total, mirroring
    /// [`Lc3Decoder::decode_frame`].
    ///
    /// Returns [`Lc3Error::Encode`] when `nbytes` is outside `20..=400` or the
    /// frame does not match the configured sample rate, channel count, bit
    /// depth, or frame length.
    pub fn encode_frame(&mut self, frame: &PcmFrame, nbytes: usize) -> Result<Vec<u8>, Lc3Error> {
        if !(20..=400).contains(&nbytes) {
            return Err(Lc3Error::Encode("LC3 frame size must be 20 to 400 bytes"));
        }
        let channels = usize::from(self.config.channels);
        if frame.sample_rate != self.config.sample_rate {
            return Err(Lc3Error::Encode(
                "LC3 frame sample rate does not match the encoder",
            ));
        }
        if frame.channels != self.config.channels {
            return Err(Lc3Error::Encode(
                "LC3 frame channel count does not match the encoder",
            ));
        }
        if frame.bits_per_sample != self.config.bit_depth {
            return Err(Lc3Error::Encode(
                "LC3 frame bit depth does not match the encoder",
            ));
        }
        if frame.planes.len() != channels {
            return Err(Lc3Error::Encode(
                "LC3 frame plane count does not match the encoder",
            ));
        }

        let ns = tables::NS[self.dt][self.sr];
        if frame.planes.iter().any(|plane| plane.len() != ns) {
            return Err(Lc3Error::Encode(
                "LC3 frame plane length does not match the frame duration",
            ));
        }

        let mut payload = Vec::with_capacity(nbytes * channels);
        for (channel, plane) in frame.planes.iter().enumerate() {
            payload.extend_from_slice(&self.channels[channel].encode(
                plane,
                self.config.bit_depth,
                nbytes,
            )?);
        }

        self.frame_number += 1;
        self.sample_number += ns as u64;
        Ok(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::ChannelPlc;

    #[test]
    fn plc_seed_sequence_and_sign_follow_appendix_b() {
        let mut plc = ChannelPlc::new(6);
        plc.retain(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let expected_seeds = [12_874u16, 54_737, 40_420, 48_499, 16_942, 43_909];
        let concealed = plc.synthesize();
        assert_eq!(concealed, [1.0, -2.0, -3.0, -4.0, 5.0, -6.0]);
        // The next call must continue the same sequence.
        plc.retain(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let mut seed = 24_607u16;
        let mut expected = Vec::new();
        for _ in 0..2 * expected_seeds.len() {
            seed = 16_831u16.wrapping_add(seed.wrapping_mul(12_821));
            expected.push(seed);
        }
        let again = plc.synthesize();
        for (index, value) in again.iter().enumerate() {
            let seed = expected[expected_seeds.len() + index];
            let sign = if seed & 0x8000 != 0 { -1.0 } else { 1.0 };
            assert_eq!(*value, sign);
        }
    }

    #[test]
    fn plc_attenuation_follows_the_consecutive_loss_count() {
        let mut plc = ChannelPlc::new(1);
        plc.retain(&[1.0]);
        let mut alpha = Vec::new();
        for _ in 0..9 {
            let _ = plc.synthesize();
            alpha.push(plc.alpha);
        }
        assert_eq!(alpha[..4], [1.0, 1.0, 1.0, 0.9]);
        assert!((alpha[6] - 0.9f32.powi(4)).abs() < 1e-6);
        assert!((alpha[7] - 0.9f32.powi(4) * 0.85).abs() < 1e-6);
        assert!((alpha[8] - 0.9f32.powi(4) * 0.85f32.powi(2)).abs() < 1e-6);
    }

    #[test]
    fn plc_suspend_restarts_attenuation_but_keeps_the_seed() {
        let mut plc = ChannelPlc::new(1);
        plc.retain(&[1.0]);
        for _ in 0..6 {
            let _ = plc.synthesize();
        }
        let seed = plc.seed;
        assert!(plc.alpha < 1.0);
        plc.suspend();
        assert_eq!(plc.count, 1);
        assert_eq!(plc.alpha, 1.0);
        let concealed = plc.synthesize();
        assert_eq!(
            concealed,
            [
                if seed.wrapping_mul(12_821).wrapping_add(16_831) & 0x8000 != 0 {
                    -1.0
                } else {
                    1.0
                }
            ]
        );
    }

    #[test]
    fn plc_retains_only_the_coded_bandwidth() {
        let mut plc = ChannelPlc::new(3);
        plc.retain(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(plc.last_good, [1.0, 2.0, 3.0]);
    }
}
