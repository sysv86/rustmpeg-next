// SPDX-License-Identifier: Apache-2.0
//! ISOBMFF and CMAF round-trip tests.

use bytes::Bytes;
use rustmpeg_core::{
    ColorDescription, ColorPrimaries, ColorRange, MatrixCoefficients, TransferFunction,
};
use rustmpeg_format::IsoAudioTrack;
use rustmpeg_format::isobmff::{
    CmafConfig, CmafWriter, IsobmffReader, Mp4Config, Mp4Writer, Sample, SampleEntry, Track,
    TrackKind, VideoTrack,
};

fn video_track(id: u32) -> Track {
    Track {
        id,
        kind: TrackKind::Video,
        timescale: 90_000,
        duration: 0,
        language: "und".into(),
        sample_entry: SampleEntry::new(*b"vvc1", Some(*b"vvcC"), Bytes::from_static(&[1, 2, 3])),
        video: Some(VideoTrack {
            width: 640,
            height: 360,
            color: Some(ColorDescription {
                range: ColorRange::Limited,
                primaries: ColorPrimaries::Bt709,
                transfer: TransferFunction::Sdr,
                matrix: MatrixCoefficients::Bt709,
                chroma_location: rustmpeg_core::ChromaLocation::Left,
            }),
            pixel_aspect_ratio: Some((1, 1)),
        }),
        audio: None,
    }
}

fn audio_track(id: u32) -> Track {
    Track {
        id,
        kind: TrackKind::Audio,
        timescale: 48_000,
        duration: 0,
        language: "eng".into(),
        sample_entry: SampleEntry::new(*b"Opus", Some(*b"dOps"), Bytes::from_static(&[9; 11])),
        video: None,
        audio: Some(IsoAudioTrack {
            channels: 2,
            sample_size: 16,
            sample_rate: 48_000.0,
        }),
    }
}

fn video_sample(dts: i64, pts: i64, keyframe: bool) -> Sample {
    Sample::new(
        1,
        dts,
        pts,
        3_000,
        keyframe,
        Bytes::from(vec![dts as u8; 32]),
    )
}

fn audio_sample(dts: i64) -> Sample {
    Sample::new(2, dts, dts, 960, true, Bytes::from(vec![0xaa; 8]))
}

#[test]
fn mp4_round_trip_preserves_tracks_and_samples() {
    let mut writer = Mp4Writer::new(Mp4Config::default());
    writer.add_track(video_track(1)).expect("video track");
    writer.add_track(audio_track(2)).expect("audio track");
    // Interleave the two tracks so the chunk tables have real boundaries.
    writer
        .write_sample(video_sample(0, 0, true))
        .expect("sample");
    writer.write_sample(audio_sample(0)).expect("sample");
    writer
        .write_sample(video_sample(3_000, 6_000, false))
        .expect("sample");
    writer.write_sample(audio_sample(960)).expect("sample");
    writer
        .write_sample(video_sample(6_000, 3_000, false))
        .expect("sample");

    let file = writer.finish().expect("finish");
    let mut reader = IsobmffReader::new(file).expect("reader");
    assert!(!reader.is_fragmented());
    assert_eq!(reader.tracks().len(), 2);
    assert_eq!(reader.sample_count(), 5);

    let video = reader.track(1).expect("video");
    assert_eq!(video.kind, TrackKind::Video);
    assert_eq!(video.timescale, 90_000);
    assert_eq!(video.sample_entry.format, *b"vvc1");
    assert_eq!(video.sample_entry.config_box, Some(*b"vvcC"));
    assert_eq!(&video.sample_entry.config[..], &[1, 2, 3]);
    let info = video.video.expect("video info");
    assert_eq!((info.width, info.height), (640, 360));
    assert_eq!(info.pixel_aspect_ratio, Some((1, 1)));
    assert_eq!(info.color.expect("color").range, ColorRange::Limited);

    let audio = reader.track(2).expect("audio");
    assert_eq!(audio.kind, TrackKind::Audio);
    assert_eq!(audio.language, "eng");
    assert_eq!(audio.sample_entry.format, *b"Opus");
    assert_eq!(audio.audio.expect("audio info").channels, 2);

    let mut samples = Vec::new();
    while let Some(sample) = reader.next_sample().expect("sample") {
        samples.push(sample);
    }
    assert_eq!(samples.len(), 5);
    let video_samples: Vec<_> = samples
        .iter()
        .filter(|sample| sample.track_id == 1)
        .collect();
    assert_eq!(video_samples.len(), 3);
    assert_eq!(video_samples[0].dts, 0);
    assert_eq!(video_samples[0].pts, 0);
    assert!(video_samples[0].keyframe);
    assert_eq!(video_samples[1].dts, 3_000);
    assert_eq!(video_samples[1].pts, 6_000);
    assert!(!video_samples[1].keyframe);
    assert_eq!(video_samples[2].dts, 6_000);
    assert_eq!(video_samples[2].pts, 3_000);
    assert_eq!(video_samples[2].duration, 3_000);
    assert_eq!(&video_samples[0].data[..], &vec![0_u8; 32]);
    let audio_samples: Vec<_> = samples
        .iter()
        .filter(|sample| sample.track_id == 2)
        .collect();
    assert_eq!(audio_samples.len(), 2);
    assert_eq!(&audio_samples[0].data[..], &[0xaa; 8]);
}

#[test]
fn cmaf_init_and_segments_round_trip() {
    let mut writer = CmafWriter::new(CmafConfig::default());
    writer.add_track(video_track(1)).expect("track");
    let init = writer.init_segment().expect("init");
    let first = writer
        .write_segment(
            1,
            &[video_sample(0, 0, true), video_sample(3_000, 3_000, false)],
        )
        .expect("segment");
    let second = writer
        .write_segment(1, &[video_sample(6_000, 6_000, true)])
        .expect("segment");

    let mut document = Vec::new();
    document.extend_from_slice(&init);
    document.extend_from_slice(&first);
    document.extend_from_slice(&second);
    let mut reader = IsobmffReader::new(Bytes::from(document)).expect("reader");
    assert!(reader.is_fragmented());
    assert_eq!(reader.tracks().len(), 1);
    assert_eq!(reader.sample_count(), 3);
    let mut samples = Vec::new();
    while let Some(sample) = reader.next_sample().expect("sample") {
        samples.push(sample);
    }
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0].dts, 0);
    assert_eq!(samples[1].dts, 3_000);
    assert_eq!(samples[2].dts, 6_000);
    assert!(samples[2].keyframe);
    assert_eq!(&samples[0].data[..], &vec![0_u8; 32]);
}

#[test]
fn cmaf_requires_a_keyframe_first() {
    let mut writer = CmafWriter::new(CmafConfig::default());
    writer.add_track(video_track(1)).expect("track");
    assert!(
        writer
            .write_segment(1, &[video_sample(0, 0, false)])
            .is_err()
    );
}

#[test]
fn mp4_rejects_out_of_order_samples() {
    let mut writer = Mp4Writer::new(Mp4Config::default());
    writer.add_track(video_track(1)).expect("track");
    writer
        .write_sample(video_sample(3_000, 3_000, true))
        .expect("sample");
    assert!(writer.write_sample(video_sample(0, 0, true)).is_err());
}

const NON_FRAGMENTED: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/isobmff/opus-20ms.mp4"
);
const FRAGMENTED: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/isobmff/opus-fragmented.mp4"
);
const MULTI_FRAGMENTED: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/isobmff/opus-multi.mp4"
);
const MULTI_PLAIN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/isobmff/opus-multi-plain.mp4"
);
const FLAC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/isobmff/flac.mp4");

fn decode_fixture(path: &str) -> Vec<(i64, usize)> {
    let data = Bytes::from(std::fs::read(path).expect("fixture"));
    let mut reader = IsobmffReader::new(data).expect("reader");
    let track = &reader.tracks()[0];
    assert_eq!(track.kind, TrackKind::Audio);
    assert_eq!(track.sample_entry.format, *b"Opus");
    assert_eq!(track.sample_entry.config_box, Some(*b"dOps"));
    assert_eq!(track.audio.expect("audio").channels, 1);
    let mut samples = Vec::new();
    while let Some(sample) = reader.next_sample().expect("sample") {
        samples.push((sample.dts, sample.data.len()));
    }
    samples
}

#[test]
fn reads_an_ffmpeg_non_fragmented_mp4_with_an_edit_list() {
    let samples = decode_fixture(NON_FRAGMENTED);
    assert_eq!(
        samples,
        vec![
            (-312, 95),
            (648, 66),
            (1608, 59),
            (2568, 56),
            (3528, 59),
            (4488, 47)
        ]
    );
}

#[test]
fn reads_an_ffmpeg_fragmented_cmaf_file() {
    let samples = decode_fixture(FRAGMENTED);
    assert_eq!(
        samples,
        vec![
            (0, 95),
            (960, 66),
            (1920, 59),
            (2880, 56),
            (3840, 59),
            (4800, 47)
        ]
    );
}

fn read_tracks(path: &str) -> (usize, usize, Vec<(u32, i64, usize)>) {
    let data = Bytes::from(std::fs::read(path).expect("fixture"));
    let mut reader = IsobmffReader::new(data).expect("reader");
    let tracks = reader.tracks().len();
    let count = reader.sample_count();
    let mut samples = Vec::new();
    while let Some(sample) = reader.next_sample().expect("sample") {
        samples.push((sample.track_id, sample.dts, sample.data.len()));
    }
    (tracks, count, samples)
}

#[test]
fn reads_a_multi_track_fragmented_file() {
    let (tracks, count, samples) = read_tracks(MULTI_FRAGMENTED);
    assert_eq!(tracks, 2);
    assert_eq!(count, 12);
    let first: Vec<_> = samples.iter().filter(|(track, _, _)| *track == 1).collect();
    let second: Vec<_> = samples.iter().filter(|(track, _, _)| *track == 2).collect();
    assert_eq!(
        first
            .iter()
            .map(|(_, dts, size)| (*dts, *size))
            .collect::<Vec<_>>(),
        vec![
            (0, 95),
            (960, 66),
            (1920, 59),
            (2880, 56),
            (3840, 59),
            (4800, 47)
        ]
    );
    assert_eq!(
        second
            .iter()
            .map(|(_, dts, size)| (*dts, *size))
            .collect::<Vec<_>>(),
        vec![
            (0, 80),
            (960, 58),
            (1920, 47),
            (2880, 56),
            (3840, 58),
            (4800, 53)
        ]
    );
}

#[test]
fn reads_interleaved_non_fragmented_tracks() {
    let (tracks, count, samples) = read_tracks(MULTI_PLAIN);
    assert_eq!(tracks, 2);
    assert_eq!(count, 12);
    let first: Vec<_> = samples.iter().filter(|(track, _, _)| *track == 1).collect();
    let second: Vec<_> = samples.iter().filter(|(track, _, _)| *track == 2).collect();
    assert_eq!(
        first
            .iter()
            .map(|(_, dts, size)| (*dts, *size))
            .collect::<Vec<_>>(),
        vec![
            (-312, 95),
            (648, 66),
            (1608, 59),
            (2568, 56),
            (3528, 59),
            (4488, 47)
        ]
    );
    assert_eq!(
        second
            .iter()
            .map(|(_, dts, size)| (*dts, *size))
            .collect::<Vec<_>>(),
        vec![
            (-312, 80),
            (648, 58),
            (1608, 47),
            (2568, 56),
            (3528, 58),
            (4488, 53)
        ]
    );
}

#[test]
fn truncated_and_corrupted_documents_never_panic() {
    for path in [NON_FRAGMENTED, FRAGMENTED, MULTI_PLAIN] {
        let data = std::fs::read(path).expect("fixture");
        for length in (0..data.len()).step_by(7) {
            let _ = IsobmffReader::new(Bytes::copy_from_slice(&data[..length]));
        }
        let mut state = 0x9e37_79b9_u32;
        for _ in 0..200 {
            let mut copy = data.clone();
            for _ in 0..4 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let index = (state as usize) % copy.len();
                copy[index] ^= (state >> 8) as u8;
            }
            let _ = IsobmffReader::new(Bytes::from(copy));
        }
    }
}

#[test]
fn reads_flac_in_mp4() {
    let data = Bytes::from(std::fs::read(FLAC).expect("fixture"));
    let mut reader = IsobmffReader::new(data).expect("reader");
    let track = &reader.tracks()[0];
    assert_eq!(track.kind, TrackKind::Audio);
    assert_eq!(track.sample_entry.format, *b"fLaC");
    assert_eq!(track.sample_entry.config_box, Some(*b"dfLa"));
    assert_eq!(track.audio.expect("audio").channels, 2);
    let mut samples = Vec::new();
    while let Some(sample) = reader.next_sample().expect("sample") {
        samples.push((sample.dts, sample.data.len()));
    }
    assert_eq!(
        samples,
        vec![(0, 371), (1024, 346), (2048, 337), (3072, 367), (4096, 289)]
    );
}

#[test]
fn cmaf_video_composition_offsets_round_trip() {
    let mut writer = CmafWriter::new(CmafConfig::default());
    writer.add_track(video_track(1)).expect("track");
    let mut document = Vec::new();
    document.extend_from_slice(&writer.init_segment().expect("init"));
    document.extend_from_slice(
        &writer
            .write_segment(
                1,
                &[
                    video_sample(0, 0, true),
                    video_sample(3_000, 6_000, false),
                    video_sample(6_000, 3_000, false),
                ],
            )
            .expect("segment"),
    );
    let mut reader = IsobmffReader::new(Bytes::from(document)).expect("reader");
    let mut samples = Vec::new();
    while let Some(sample) = reader.next_sample().expect("sample") {
        samples.push(sample);
    }
    assert_eq!(samples.len(), 3);
    assert_eq!(
        samples
            .iter()
            .map(|sample| (sample.dts, sample.pts, sample.keyframe))
            .collect::<Vec<_>>(),
        vec![(0, 0, true), (3_000, 6_000, false), (6_000, 3_000, false)]
    );
}
