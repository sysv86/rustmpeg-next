// SPDX-License-Identifier: Apache-2.0
//! SILK decoder integration tests against real Opus fixtures.

#[path = "support/ogg.rs"]
#[allow(dead_code)]
mod ogg;

use std::path::Path;

use rustmpeg_audio::opus::silk::SilkDecoder;
use rustmpeg_audio::{OpusMode, OpusPacket};

const MONO_8K_20MS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/opus/mono_voip_20ms_8k.opus"
);
const MONO_8K_10MS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/opus/mono_voip_10ms_8k.opus"
);
const MONO_60MS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/opus/mono_voip_60ms_10k.opus"
);

/// Decodes a fixture and returns the interleaved PCM and the total number of
/// samples per channel.
fn decode(path: &str, rate: u32, channels: usize) -> (Vec<i16>, usize) {
    let data = std::fs::read(Path::new(path)).expect("fixture");
    let stream = ogg::read_stream(&data).expect("ogg");
    let audio = &stream.packets[2..];
    assert!(!audio.is_empty(), "fixture has audio packets");

    let mut decoder = SilkDecoder::new(rate, channels).expect("decoder");
    let mut pcm = vec![0_i16; (rate as usize / 1000) * 60 * channels];
    let mut output = Vec::new();
    let mut total = 0;
    let mut silent_packets = 0;
    for payload in audio {
        let packet = OpusPacket::parse(&payload.data).expect("packet");
        let info = packet.info();
        assert_eq!(info.mode, OpusMode::Silk, "{path} is not a SILK fixture");
        for frame in packet.frames() {
            let written = decoder
                .decode_frame(
                    frame,
                    info.bandwidth,
                    info.frame_duration_us,
                    info.stereo,
                    &mut pcm,
                )
                .expect("decode");
            assert_eq!(
                written,
                info.frame_duration_us as usize * rate as usize / 1_000_000
            );
            let block = &pcm[..written * channels];
            if block.iter().all(|&sample| sample == 0) {
                silent_packets += 1;
            }
            output.extend_from_slice(block);
            total += written;
        }
    }
    assert!(
        silent_packets * 4 < audio.len(),
        "decoder produced mostly silence"
    );
    (output, total)
}

#[test]
fn silk_nb_20ms_decodes_at_8_and_48_khz() {
    let (narrow, samples) = decode(MONO_8K_20MS, 8_000, 1);
    assert_eq!(narrow.len(), samples);
    assert!(narrow.iter().any(|&sample| sample.abs() > 100));

    let (wide, samples_48) = decode(MONO_8K_20MS, 48_000, 1);
    assert_eq!(wide.len(), samples_48);
    assert_eq!(wide.len(), samples * 6);
}

#[test]
fn silk_nb_10ms_frames_decode() {
    let (pcm, samples) = decode(MONO_8K_10MS, 8_000, 1);
    assert_eq!(pcm.len(), samples);
    assert!(pcm.iter().any(|&sample| sample.abs() > 100));
}

#[test]
fn silk_wb_60ms_frames_decode() {
    let (pcm, samples) = decode(MONO_60MS, 16_000, 1);
    assert_eq!(pcm.len(), samples);
    assert!(pcm.iter().any(|&sample| sample.abs() > 100));

    let (pcm_48, samples_48) = decode(MONO_60MS, 48_000, 1);
    assert_eq!(pcm_48.len(), samples_48);
    assert_eq!(pcm_48.len(), samples * 3);
}

#[test]
fn silk_mono_packet_can_be_duplicated_to_stereo() {
    let (mono, samples) = decode(MONO_8K_20MS, 8_000, 1);
    let (stereo, stereo_samples) = decode(MONO_8K_20MS, 8_000, 2);
    assert_eq!(samples, stereo_samples);
    assert_eq!(stereo.len(), samples * 2);
    for index in 0..mono.len() {
        assert_eq!(stereo[index * 2], mono[index]);
        assert_eq!(stereo[index * 2 + 1], mono[index]);
    }
}

#[test]
fn silk_decoder_rejects_short_output() {
    let data = std::fs::read(Path::new(MONO_8K_20MS)).expect("fixture");
    let stream = ogg::read_stream(&data).expect("ogg");
    let packet = OpusPacket::parse(&stream.packets[2].data).expect("packet");
    let info = packet.info();
    let frame = packet.frames().next().expect("frame");
    let mut decoder = SilkDecoder::new(8_000, 1).expect("decoder");
    let mut pcm = [0_i16; 8];
    assert!(
        decoder
            .decode_frame(
                frame,
                info.bandwidth,
                info.frame_duration_us,
                info.stereo,
                &mut pcm,
            )
            .is_err()
    );
}

#[test]
fn silk_decoder_rejects_invalid_configuration() {
    assert!(SilkDecoder::new(44_100, 1).is_err());
    assert!(SilkDecoder::new(48_000, 3).is_err());
}

#[test]
fn silk_decoder_handles_arbitrary_payloads_without_panicking() {
    use rustmpeg_audio::OpusBandwidth;

    let configurations = [
        (OpusBandwidth::Narrowband, 10_000),
        (OpusBandwidth::Narrowband, 60_000),
        (OpusBandwidth::Mediumband, 20_000),
        (OpusBandwidth::Wideband, 40_000),
    ];
    let mut state = 0x1234_5678_u32;
    let mut decoder = SilkDecoder::new(48_000, 2).expect("decoder");
    let mut pcm = vec![0_i16; 48_000 / 1000 * 60 * 2];
    for iteration in 0..400 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let length = (state as usize >> 8) % 96;
        let payload: Vec<u8> = (0..length)
            .map(|index| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> ((index % 4) * 8)) as u8
            })
            .collect();
        let (bandwidth, duration) = configurations[iteration % configurations.len()];
        let _ = decoder.decode_frame(&payload, bandwidth, duration, iteration % 2 == 0, &mut pcm);
    }
}
