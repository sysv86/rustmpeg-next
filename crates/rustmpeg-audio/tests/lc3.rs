// SPDX-License-Identifier: Apache-2.0
//! LC3 conformance against liblc3 reference output.
//!
//! The fixtures under `tests/data/lc3` are described in the accompanying
//! `README.md` and `manifest.txt`. The decoder tests compare sample-exact
//! output; the encoder test re-encodes the fixture PCM and measures the
//! round-trip signal-to-noise ratio.

use std::path::{Path, PathBuf};

use rustmpeg_audio::{Lc3Config, Lc3Decoder, Lc3Encoder, Lc3FrameDuration, PcmFrame};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/lc3")
}

fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

/// Splits an LC3 binary file into its configuration and raw frames.
fn parse_stream(data: &[u8]) -> (Lc3Config, usize, Vec<&[u8]>) {
    assert!(data.len() > 18, "short LC3 binary file");
    assert_eq!(read_u16(data, 0), 0xcc1c, "LC3 binary file identifier");
    let header_size = usize::from(read_u16(data, 2));
    let sample_rate = u32::from(read_u16(data, 4)) * 100;
    let channels = read_u16(data, 8) as u8;
    let frame_us = u32::from(read_u16(data, 10)) * 10;
    let _nsamples = u32::from(read_u16(data, 14)) | (u32::from(read_u16(data, 16)) << 16);
    let frame_duration = match frame_us {
        7_500 => Lc3FrameDuration::Ms7_5,
        10_000 => Lc3FrameDuration::Ms10,
        other => panic!("unsupported frame duration {other} us"),
    };
    let mut frames = Vec::new();
    let mut offset = header_size;
    while offset + 2 <= data.len() {
        let size = usize::from(read_u16(data, offset));
        offset += 2;
        frames.push(&data[offset..offset + size]);
        offset += size;
    }
    (
        Lc3Config {
            sample_rate,
            frame_duration,
            channels,
            bit_depth: 16,
        },
        header_size,
        frames,
    )
}

/// Decoder delay in samples, `2 * (ND - NS/2)`.
fn decoder_delay(config: &Lc3Config) -> usize {
    let overlap = match config.frame_duration {
        Lc3FrameDuration::Ms7_5 => config.sample_rate as usize * 2 / 1000,
        Lc3FrameDuration::Ms10 => config.sample_rate as usize / 800,
    };
    2 * overlap
}

fn check_fixture(name: &str) {
    let stream = std::fs::read(data_dir().join(format!("{name}.lc3"))).expect("read stream");
    let reference =
        std::fs::read(data_dir().join(format!("{name}.s16le"))).expect("read reference");
    let (config, _, frames) = parse_stream(&stream);
    let channels = usize::from(config.channels);
    let mut decoder = Lc3Decoder::new(config).expect("decoder");

    let mut pcm = Vec::new();
    for frame in frames {
        let decoded = decoder.decode_frame(frame).expect("decode frame");
        for index in 0..decoded.block_size() {
            for plane in &decoded.planes {
                pcm.push(plane[index]);
            }
        }
    }
    assert_eq!(pcm.len() % channels, 0);

    let delay = decoder_delay(&config) * channels;
    assert!(
        pcm.len() >= delay + reference.len() / 2,
        "{name}: decoded {} samples, need {}",
        pcm.len(),
        delay + reference.len() / 2
    );
    let refs: Vec<i16> = reference
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| i16::from_le_bytes(*bytes))
        .collect();
    let mut worst = 0i32;
    for (index, &expected) in refs.iter().enumerate() {
        let actual = pcm[delay + index];
        worst = worst.max((actual - i32::from(expected)).abs());
        assert!(
            (actual - i32::from(expected)).abs() <= 2,
            "{name}: sample {index} is {actual}, expected {expected}"
        );
    }
    assert!(worst <= 2, "{name}: worst deviation {worst}");
}

#[test]
fn decodes_48k_10ms_stereo() {
    check_fixture("lc3_48k_10ms_stereo_96k");
}

#[test]
fn decodes_48k_10ms_stereo_lsb_mode() {
    check_fixture("lc3_48k_10ms_stereo_256k");
}

#[test]
fn decodes_32k_10ms_mono() {
    check_fixture("lc3_32k_10ms_mono_24k");
}

#[test]
fn decodes_24k_10ms_mono() {
    check_fixture("lc3_24k_10ms_mono_48k");
}

#[test]
fn decodes_16k_75ms_stereo() {
    check_fixture("lc3_16k_75ms_stereo_64k");
}

#[test]
fn decodes_8k_10ms_mono() {
    check_fixture("lc3_8k_10ms_mono_32k");
}

#[test]
fn decodes_48k_75ms_stereo_with_uneven_split() {
    check_fixture("lc3_48k_75ms_stereo_48k");
}

/// Round-trip delay of the encoder and decoder, in samples per channel.
fn round_trip_delay(config: &Lc3Config) -> usize {
    let micros = match config.frame_duration {
        Lc3FrameDuration::Ms7_5 => 4_000,
        Lc3FrameDuration::Ms10 => 2_500,
    };
    config.sample_rate as usize * micros / 1_000_000
}

/// Re-encodes the fixture PCM at the fixture frame size and checks that the
/// decoded result tracks the input at the codec delay.
fn check_encoder_round_trip(name: &str) {
    let stream = std::fs::read(data_dir().join(format!("{name}.lc3"))).expect("read stream");
    let pcm = std::fs::read(data_dir().join(format!("{name}.s16le"))).expect("read pcm");
    let (config, _, frames) = parse_stream(&stream);
    let channels = usize::from(config.channels);
    assert!(
        frames.iter().all(|frame| frame.len() == frames[0].len()),
        "{name}: variable fixture frame size"
    );
    // Stereo fixtures may carry an odd total byte count; the public encoder
    // uses one payload size for every channel, so round down.
    let nbytes = frames[0].len() / channels;
    let ns = config.frame_samples();
    let samples: Vec<i32> = pcm
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| i32::from(i16::from_le_bytes(*bytes)))
        .collect();
    let total = samples.len() / (ns * channels);
    assert!(total >= 5, "{name}: fixture is too short");

    let mut encoder = Lc3Encoder::new(config).expect("encoder");
    let mut decoder = Lc3Decoder::new(config).expect("decoder");
    let mut decoded = Vec::with_capacity(samples.len());
    for index in 0..total {
        let mut planes = vec![Vec::with_capacity(ns); channels];
        for sample in 0..ns {
            for (channel, plane) in planes.iter_mut().enumerate() {
                plane.push(samples[(index * ns + sample) * channels + channel]);
            }
        }
        let frame = PcmFrame {
            sample_rate: config.sample_rate,
            channels: config.channels,
            bits_per_sample: config.bit_depth,
            frame_number: index as u64,
            sample_number: (index * ns) as u64,
            planes,
        };
        let payload = encoder.encode_frame(&frame, nbytes).expect("encode frame");
        assert_eq!(payload.len(), nbytes * channels);
        let output = decoder.decode_frame(&payload).expect("decode frame");
        for sample in 0..ns {
            for plane in &output.planes {
                decoded.push(plane[sample]);
            }
        }
    }

    let lag = round_trip_delay(&config) * channels;
    let mut error = 0.0f64;
    let mut signal = 0.0f64;
    for index in 2..total - 2 {
        let start = index * ns * channels;
        let shifted = start + lag;
        assert!(shifted + ns * channels <= decoded.len());
        for offset in 0..ns * channels {
            let input = f64::from(samples[start + offset]);
            let output = f64::from(decoded[shifted + offset]);
            error += (input - output) * (input - output);
            signal += input * input;
        }
    }
    assert!(error > 0.0, "{name}: decoded output equals the input");
    let snr = 10.0 * (signal / error).log10();
    assert!(snr > 10.0, "{name}: round-trip SNR is only {snr:.2} dB");
}

#[test]
fn encodes_fixtures_with_round_trip_quality() {
    for name in [
        "lc3_48k_10ms_stereo_96k",
        "lc3_48k_10ms_stereo_256k",
        "lc3_32k_10ms_mono_24k",
        "lc3_24k_10ms_mono_48k",
        "lc3_16k_75ms_stereo_64k",
        "lc3_8k_10ms_mono_32k",
        "lc3_48k_75ms_stereo_48k",
    ] {
        check_encoder_round_trip(name);
    }
}

/// Frame indices that the PLC fixture treats as lost or corrupted.
const PLC_LOSSES: &[usize] = &[4, 5, 6, 10, 11, 12, 13, 17];

/// Feeds the fixture stream to the decoder with a loss pattern and compares
/// the concealed output with the liblc3 reference for the same pattern.
#[test]
fn conceals_lost_frames_like_liblc3() {
    let name = "lc3_48k_10ms_stereo_96k";
    let stream = std::fs::read(data_dir().join(format!("{name}.lc3"))).expect("read stream");
    let reference = std::fs::read(data_dir().join(format!("{name}_plc.s16le"))).expect("read plc");
    let (config, _, frames) = parse_stream(&stream);
    let channels = usize::from(config.channels);
    let mut decoder = Lc3Decoder::new(config).expect("decoder");

    let mut pcm = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        let decoded = if PLC_LOSSES.contains(&index) {
            decoder.conceal_frame()
        } else {
            decoder.decode_frame(frame)
        }
        .expect("decode or conceal");
        assert_eq!(decoded.frame_number, index as u64);
        assert_eq!(
            decoded.sample_number,
            (index * config.frame_samples()) as u64
        );
        for sample in 0..decoded.block_size() {
            for plane in &decoded.planes {
                pcm.push(plane[sample]);
            }
        }
    }

    let delay = decoder_delay(&config) * channels;
    let expected: Vec<i16> = reference
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| i16::from_le_bytes(*bytes))
        .collect();
    assert!(
        pcm.len() >= delay + expected.len(),
        "short concealed output"
    );
    for (index, &want) in expected.iter().enumerate() {
        let got = pcm[delay + index];
        assert!(
            (got - i32::from(want)).abs() <= 2,
            "plc sample {index} is {got}, expected {want}"
        );
    }
}

#[test]
fn concealment_before_any_good_frame_is_silent() {
    let name = "lc3_48k_10ms_stereo_96k";
    let stream = std::fs::read(data_dir().join(format!("{name}.lc3"))).expect("read stream");
    let (config, _, _) = parse_stream(&stream);
    let mut decoder = Lc3Decoder::new(config).expect("decoder");
    let frame = decoder.conceal_frame().expect("conceal");
    assert_eq!(frame.frame_number, 0);
    assert_eq!(frame.planes.len(), 2);
    assert!(
        frame
            .planes
            .iter()
            .all(|plane| plane.len() == config.frame_samples())
    );
    assert!(
        frame.planes.iter().flatten().all(|&sample| sample == 0),
        "concealment without a good frame must produce silence"
    );
}
