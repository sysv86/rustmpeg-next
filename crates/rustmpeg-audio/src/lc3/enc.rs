// SPDX-License-Identifier: Apache-2.0
//! LC3 encoder front end (Bluetooth LC3 v1.0.1 §3.3).
//!
//! [`ChannelEncoder`] turns one channel plane of native integer PCM into one
//! LC3 payload. The processing chain follows the normative encoder model:
//! input scaling (§3.3.3), low-delay MDCT analysis (§3.3.4), band energies and
//! bandwidth detection (§3.3.4.4-§3.3.5), attack detection (§3.3.6), spectral
//! noise shaping (§3.3.7), temporal noise shaping (§3.3.8), long-term
//! postfilter analysis (§3.3.9), spectral quantization (§3.3.10-§3.3.12), and
//! bitstream multiplexing with arithmetic coding (§3.3.13).
//!
//! All spectral values stay on the 16-bit PCM scale used by the decoder, which
//! is also the scale of the analysis and synthesis building blocks.

use super::Lc3Error;
use super::enc_bits::{AcEncoder, BitWriter};
use super::enc_det::{self, AttackDetector};
use super::enc_ltpf::LtpfAnalysis;
use super::enc_mdct::MdctAnalysis;
use super::enc_sns;
use super::enc_spec::SpecEncoder;
use super::enc_tns;
use super::sns::spectral_shaping;
use super::spec::band_limits;
use super::tables::NS;

/// Smallest payload size supported by LC3 (§3.2.5).
const MIN_NBYTES: usize = 20;

/// Largest payload size supported by LC3 (§3.2.5).
const MAX_NBYTES: usize = 400;

/// Per-channel LC3 encoder.
pub(crate) struct ChannelEncoder {
    /// Frame duration index: `0` for 7.5 ms, `1` for 10 ms.
    dt: usize,
    /// Sample-rate index: `0` to `4` for 8 to 48 kHz.
    sr: usize,
    /// Forward MDCT with its input history.
    mdct: MdctAnalysis,
    /// Time-domain attack detector.
    attack: AttackDetector,
    /// Long-term postfilter analysis with its history.
    ltpf: LtpfAnalysis,
    /// Spectral quantizer with its gain-offset history.
    spec: SpecEncoder,
}

impl ChannelEncoder {
    /// Creates an encoder for one channel.
    pub(crate) fn new(dt: usize, sr: usize) -> Result<Self, Lc3Error> {
        if dt > 1 {
            return Err(Lc3Error::InvalidConfig("invalid LC3 frame duration"));
        }
        if sr > 4 {
            return Err(Lc3Error::InvalidConfig("invalid LC3 sample rate"));
        }
        Ok(Self {
            dt,
            sr,
            mdct: MdctAnalysis::new(dt, sr)?,
            attack: AttackDetector::new(dt, sr),
            ltpf: LtpfAnalysis::new(dt, sr),
            spec: SpecEncoder::new(),
        })
    }

    /// Encodes one channel plane of native integer PCM into exactly `nbytes`
    /// bytes.
    ///
    /// Returns [`Lc3Error::Encode`] when the frame size, the bit depth, or the
    /// plane length does not match the configuration, and when the arithmetic
    /// coder and the side information would overlap.
    pub(crate) fn encode(
        &mut self,
        pcm: &[i32],
        bit_depth: u8,
        nbytes: usize,
    ) -> Result<Vec<u8>, Lc3Error> {
        // §3.2.5 frame sizes and the supported PCM depths (§3.3.3).
        if !(MIN_NBYTES..=MAX_NBYTES).contains(&nbytes) {
            return Err(Lc3Error::Encode("LC3 frame size must be 20 to 400 bytes"));
        }
        if !matches!(bit_depth, 16 | 24) {
            return Err(Lc3Error::Encode("LC3 PCM bit depth must be 16 or 24"));
        }
        let ns = NS[self.dt][self.sr];
        if pcm.len() != ns {
            return Err(Lc3Error::Encode(
                "LC3 PCM plane length does not match the frame",
            ));
        }

        // §3.3.3, Equations 4-5: scale to the 16-bit PCM range and clip.
        // `xt` is the same signal rounded onto the integer grid; the attack
        // and LTPF detectors operate on that copy.
        let scale = 2f32.powi(16 - i32::from(bit_depth));
        let mut xs = Vec::with_capacity(ns);
        let mut xt = Vec::with_capacity(ns);
        for &sample in pcm {
            let scaled = (sample as f32 * scale).clamp(-32768.0, 32767.0);
            xs.push(scaled);
            xt.push(scaled.round().clamp(-32768.0, 32767.0));
        }

        // §3.3.4: low-delay MDCT analysis.
        let mut xf = self.mdct.run(&xs)?;

        // §3.3.4.4-§3.3.4.5: band energies and the near-Nyquist flag.
        let energy = enc_det::band_energies(&xf, self.dt, self.sr);
        let nb = super::enc_mdct::band_count(self.dt, self.sr);

        // §3.3.6: attack detection.
        let attack = self.attack.run(&xt, nbytes);

        // §3.3.9: LTPF analysis. The near-Nyquist flag comes from the band
        // energies above, so this runs after the MDCT.
        let ltpf = self.ltpf.analyse(&xt, nbytes, energy.near_nyquist);

        // §3.3.5: bandwidth detection.
        let bw = enc_det::detect_bandwidth(&energy.eb, self.dt, self.sr);
        let nbits_bw = enc_det::bandwidth_bits(self.sr);
        let nbits = (8 * nbytes) as i32;

        // §3.3.7: SNS analysis and quantization.
        let scf = enc_sns::analyse(&energy.eb, nb, self.sr, self.dt, attack);
        let sns = enc_sns::quantize(&scf)?;

        // SNS shaping uses the reconstructed factors, matching the decoder.
        spectral_shaping(self.dt, self.sr, &sns.scf_q, false, &mut xf);

        // §3.3.8: TNS analysis and filtering.
        let tns = enc_tns::analyse(&xf, self.dt, bw, nbits, energy.near_nyquist);
        enc_tns::filter(&mut xf, self.dt, bw, &tns);

        // §3.3.10-§3.3.12: quantization of the shaped spectrum.
        let bw_stop = *band_limits(self.dt, bw)
            .last()
            .ok_or(Lc3Error::InvalidConfig("missing band table"))? as usize;
        let nbits_overhead = nbits_bw + tns.nbits + ltpf.nbits + 38 + 8 + 3;
        let mut spec = self
            .spec
            .analyse(&xf, self.dt, self.sr, nbytes, nbits_overhead, bw_stop)?;

        // §3.3.13.3-§3.3.13.5: multiplex the payload.
        let mut payload = vec![0u8; nbytes];
        let coded;
        let last_byte;
        {
            let mut bits = BitWriter::new(&mut payload);

            if nbits_bw > 0 {
                bits.write_uint(bw as u32, nbits_bw as u32)?;
            }
            spec.write_side(&mut bits)?;
            for filter in tns.filter.iter().take(tns.filters) {
                bits.write_bit(filter.rc_order > 0)?;
            }
            bits.write_bit(ltpf.pitch_present)?;
            sns.write_side(&mut bits)?;
            if ltpf.pitch_present {
                bits.write_bit(ltpf.active)?;
                bits.write_uint(u32::from(ltpf.pitch_index), 9)?;
            }
            spec.write_noise(&mut bits)?;

            // Arithmetic-coded spectrum, signs, LSB planes, and residuals.
            let mut ac = AcEncoder::new();
            enc_tns::write_ac(&mut ac, &tns);
            spec.write_ac(&mut ac, &mut bits)?;
            let available = bits.bits_written(nbits) - ac.bits_estimate();
            // §3.3.13.5: this is the exact number of bits the arithmetic
            // stream will occupy once the coder is finished.
            let ac_bits = ac.bits_estimate();
            spec.write_residual(&mut bits, available)?;
            ac.finish();

            // The arithmetic stream grows upwards from bit zero and the side
            // information downwards from the last bit. The final partial
            // arithmetic byte fills the high bits of its byte while a
            // partially written side byte fills the low bits, so the collision
            // test is on bits rather than whole bytes.
            let side_bits = nbits - bits.bits_written(nbits);
            if i64::from(ac_bits) + i64::from(side_bits) > i64::from(nbits) {
                return Err(Lc3Error::Encode("LC3 arithmetic coder overflow"));
            }
            // The side information occupies whole bytes plus a partial byte.
            // The arithmetic stream may end in that shared byte only when its
            // final partial byte fits above the side bits, which the bit
            // count above guarantees. At most `bottom() + 1` bytes are free.
            let bottom = bits.bottom();
            let ac_len = ac.bytes().len() as i64;
            if ac_len > bottom + 1 {
                return Err(Lc3Error::Encode("LC3 arithmetic coder overflow"));
            }
            let Some(last) = ac.bytes().len().checked_sub(1) else {
                return Err(Lc3Error::Encode("LC3 arithmetic coder empty"));
            };
            if last >= payload.len() {
                return Err(Lc3Error::Encode("LC3 arithmetic coder overflow"));
            }
            coded = ac.bytes()[..last].to_vec();
            last_byte = ac.bytes()[last];
            // Bytes below the final one replace the zero padding, while the
            // final byte is merged so that side bits in the same byte survive.
            payload[..last].copy_from_slice(&coded);
            payload[last] |= last_byte;
        }

        Ok(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PcmFrame;
    use crate::lc3::{Lc3Config, Lc3Decoder, Lc3Encoder, Lc3FrameDuration};

    /// Small deterministic linear congruential generator.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }

        /// Uniform value in `[-1, 1)`.
        fn bipolar(&mut self) -> f32 {
            (self.next() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        }
    }

    /// 200 Hz tone plus low-level noise, on the 16-bit integer grid.
    fn test_signal(ns: usize, sr_hz: f32, seed: u64) -> Vec<i32> {
        let mut rng = Lcg::new(seed);
        (0..ns)
            .map(|n| {
                let tone = 8000.0 * (2.0 * std::f32::consts::PI * 200.0 * n as f32 / sr_hz).sin();
                let noise = 100.0 * rng.bipolar();
                (tone + noise).round().clamp(-32768.0, 32767.0) as i32
            })
            .collect()
    }

    #[test]
    fn encodes_every_configuration_to_the_requested_size() {
        let sizes = [20usize, 40, 80, 160, 320, 400];
        for (dt, row) in NS.iter().enumerate() {
            for (sr, &ns) in row.iter().enumerate() {
                let fs = super::super::tables::SRATE_KHZ[sr] as f32 * 1000.0;
                let mut successes = 0usize;
                for &nbytes in &sizes {
                    let signal = test_signal(ns, fs, (dt * 97 + sr * 13 + nbytes) as u64 + 1);
                    let mut encoder = ChannelEncoder::new(dt, sr).unwrap();
                    match encoder.encode(&signal, 16, nbytes) {
                        Ok(payload) => {
                            assert_eq!(payload.len(), nbytes, "dt={dt} sr={sr} nbytes={nbytes}");
                            successes += 1;
                        }
                        Err(Lc3Error::Encode(_)) => {}
                        Err(error) => panic!("unexpected error for dt={dt} sr={sr}: {error}"),
                    }
                }
                assert!(
                    successes >= 3,
                    "only {successes} feasible frame sizes for dt={dt} sr={sr}"
                );
            }
        }
    }

    #[test]
    fn valid_bit_depths_produce_payloads() {
        let (dt, sr) = (1usize, 4usize);
        let ns = NS[dt][sr];
        let signal = test_signal(ns, 48_000.0, 7);
        for bit_depth in [16u8, 24] {
            let mut encoder = ChannelEncoder::new(dt, sr).unwrap();
            let payload = encoder.encode(&signal, bit_depth, 80).unwrap();
            assert_eq!(payload.len(), 80);
        }
    }

    #[test]
    fn round_trip_reproduces_a_tone() {
        let config = Lc3Config {
            sample_rate: 48_000,
            frame_duration: Lc3FrameDuration::Ms10,
            channels: 1,
            bit_depth: 16,
        };
        let ns = 480usize;
        let frames = 40usize;
        let nbytes = 120usize;
        let fs = 48_000.0f64;
        let frequency = 1000.0f64;

        let input: Vec<i32> = (0..frames * ns)
            .map(|n| {
                (8000.0 * (2.0 * std::f64::consts::PI * frequency * n as f64 / fs).sin()).round()
                    as i32
            })
            .collect();

        let mut encoder = Lc3Encoder::new(config).unwrap();
        let mut decoder = Lc3Decoder::new(config).unwrap();
        let mut decoded = Vec::with_capacity(frames * ns);
        for frame_index in 0..frames {
            let start = frame_index * ns;
            let frame = PcmFrame {
                sample_rate: config.sample_rate,
                channels: config.channels,
                bits_per_sample: config.bit_depth,
                frame_number: frame_index as u64,
                sample_number: start as u64,
                planes: vec![input[start..start + ns].to_vec()],
            };
            let payload = encoder.encode_frame(&frame, nbytes).unwrap();
            assert_eq!(payload.len(), nbytes);
            let output = decoder.decode_frame(&payload).unwrap();
            decoded.extend_from_slice(&output.planes[0]);
        }

        // The codec has an algorithmic delay; the first frames are transient.
        let skip = 3 * ns;
        let output: Vec<f64> = decoded[skip..]
            .iter()
            .map(|&value| f64::from(value))
            .collect();
        let reference: Vec<f64> = input[skip..]
            .iter()
            .map(|&value| f64::from(value))
            .collect();

        // Phase-invariant normalized correlation with a sine at the reference
        // frequency: the projection of the output onto the two quadrature
        // components, normalized by the signal energy.
        let mut real = 0.0f64;
        let mut imag = 0.0f64;
        let mut energy = 0.0f64;
        for (index, &value) in output.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * frequency * (skip + index) as f64 / fs;
            real += value * phase.cos();
            imag += value * phase.sin();
            energy += value * value;
        }
        let correlation =
            (real * real + imag * imag).sqrt() / (energy * output.len() as f64 / 2.0).sqrt();

        let input_rms =
            (reference.iter().map(|v| v * v).sum::<f64>() / reference.len() as f64).sqrt();
        let output_rms = (energy / output.len() as f64).sqrt();
        let rms_ratio = output_rms / input_rms;

        println!("round-trip correlation {correlation:.6}, RMS ratio {rms_ratio:.4}");

        assert!(
            correlation > 0.9,
            "decoded tone correlation {correlation} is too low"
        );
        assert!(
            rms_ratio > 1.0 / 3.0 && rms_ratio < 3.0,
            "decoded RMS ratio {rms_ratio} is out of range"
        );
    }

    #[test]
    fn silence_stays_silent() {
        let config = Lc3Config {
            sample_rate: 48_000,
            frame_duration: Lc3FrameDuration::Ms10,
            channels: 1,
            bit_depth: 16,
        };
        let ns = 480usize;
        let mut encoder = Lc3Encoder::new(config).unwrap();
        let mut decoder = Lc3Decoder::new(config).unwrap();
        let mut peak = 0i32;
        for frame_index in 0..20u64 {
            let frame = PcmFrame {
                sample_rate: config.sample_rate,
                channels: config.channels,
                bits_per_sample: config.bit_depth,
                frame_number: frame_index,
                sample_number: frame_index * ns as u64,
                planes: vec![vec![0i32; ns]],
            };
            let payload = encoder.encode_frame(&frame, 120).unwrap();
            let output = decoder.decode_frame(&payload).unwrap();
            if frame_index >= 3 {
                peak = peak.max(
                    output.planes[0]
                        .iter()
                        .map(|value| value.abs())
                        .max()
                        .unwrap_or(0),
                );
            }
        }
        assert!(peak <= 8, "silence produced a peak of {peak}");
    }

    #[test]
    fn rejects_invalid_input() {
        let (dt, sr) = (1usize, 4usize);
        let ns = NS[dt][sr];
        let mut encoder = ChannelEncoder::new(dt, sr).unwrap();
        assert!(encoder.encode(&vec![0i32; ns], 16, 19).is_err());
        assert!(encoder.encode(&vec![0i32; ns], 16, 401).is_err());
        assert!(encoder.encode(&vec![0i32; ns - 1], 16, 80).is_err());
        assert!(encoder.encode(&vec![0i32; ns], 20, 80).is_err());

        assert!(ChannelEncoder::new(2, 0).is_err());
        assert!(ChannelEncoder::new(0, 5).is_err());
    }

    #[test]
    fn stereo_api_concatenates_channels() {
        let config = Lc3Config {
            sample_rate: 48_000,
            frame_duration: Lc3FrameDuration::Ms10,
            channels: 2,
            bit_depth: 16,
        };
        let ns = 480usize;
        let nbytes = 120usize;
        let mut encoder = Lc3Encoder::new(config).unwrap();
        let frame = PcmFrame {
            sample_rate: config.sample_rate,
            channels: config.channels,
            bits_per_sample: config.bit_depth,
            frame_number: 0,
            sample_number: 0,
            planes: vec![test_signal(ns, 48_000.0, 1), test_signal(ns, 48_000.0, 2)],
        };
        let payload = encoder.encode_frame(&frame, nbytes).unwrap();
        assert_eq!(payload.len(), 2 * nbytes);

        // Decode the concatenated block to make sure both halves are complete.
        let mut decoder = Lc3Decoder::new(config).unwrap();
        let output = decoder.decode_frame(&payload).unwrap();
        assert_eq!(output.planes.len(), 2);
        assert_eq!(output.planes[0].len(), ns);

        let mut wrong_rate = frame.clone();
        wrong_rate.sample_rate = 32_000;
        assert!(encoder.encode_frame(&wrong_rate, nbytes).is_err());

        let mut wrong_channels = frame.clone();
        wrong_channels.channels = 1;
        assert!(encoder.encode_frame(&wrong_channels, nbytes).is_err());

        let mut wrong_depth = frame.clone();
        wrong_depth.bits_per_sample = 24;
        assert!(encoder.encode_frame(&wrong_depth, nbytes).is_err());

        let mut wrong_planes = frame.clone();
        wrong_planes.planes.pop();
        assert!(encoder.encode_frame(&wrong_planes, nbytes).is_err());
    }
}
