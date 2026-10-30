// SPDX-License-Identifier: Apache-2.0
//! Matroska/WebM muxer round-trip tests.
//!
//! The muxer output is read back with the in-tree demuxer. FFmpeg n9.0.2 was
//! used as an external oracle during development: `ffprobe` reported the same
//! tracks and packets, and `ffmpeg -i output -f null -` decoded the remuxed
//! WebM/Opus and Matroska/FLAC files without errors.

use std::path::{Path, PathBuf};

use rustmpeg_format::{
    Bytes, CodecId, DocType, MatroskaReader, MatroskaWriter, MuxerConfig, Packet, Track,
};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/matroska")
}

fn open(name: &str) -> MatroskaReader {
    let data = Bytes::from(std::fs::read(data_dir().join(name)).expect("read fixture"));
    MatroskaReader::new(data).expect("parse fixture")
}

fn read_packets(reader: &mut MatroskaReader) -> Vec<Packet> {
    let mut packets = Vec::new();
    while let Some(packet) = reader.next_packet().expect("read packet") {
        packets.push(packet);
    }
    packets
}

fn remux(reader: &mut MatroskaReader, doc_type: DocType) -> Bytes {
    let mut writer = MatroskaWriter::new(MuxerConfig {
        doc_type,
        ..MuxerConfig::default()
    })
    .expect("create writer");
    for track in reader.tracks() {
        writer.add_track(track.clone()).expect("add track");
    }
    for packet in read_packets(reader) {
        writer.write_packet(packet).expect("write packet");
    }
    writer.finish().expect("finish document")
}

fn assert_track_eq(left: &Track, right: &Track) {
    assert_eq!(left.number, right.number);
    assert_eq!(left.uid, right.uid);
    assert_eq!(left.track_type, right.track_type);
    assert_eq!(left.codec_id, right.codec_id);
    assert_eq!(left.codec, right.codec);
    assert_eq!(left.codec_private, right.codec_private);
    assert_eq!(left.flag_lacing, right.flag_lacing);
    assert_eq!(left.default_duration_ns, right.default_duration_ns);
    assert_eq!(left.codec_delay_ns, right.codec_delay_ns);
    assert_eq!(left.seek_pre_roll_ns, right.seek_pre_roll_ns);
    assert_eq!(left.language, right.language);
    assert_eq!(left.name, right.name);
    assert_eq!(left.audio, right.audio);
    assert_eq!(left.video, right.video);
}

fn assert_packet_eq(left: &Packet, right: &Packet) {
    assert_eq!(left.track, right.track);
    assert_eq!(left.timestamp_ns, right.timestamp_ns);
    assert_eq!(left.duration_ns, right.duration_ns);
    assert_eq!(left.keyframe, right.keyframe);
    assert_eq!(left.discardable, right.discardable);
    assert_eq!(left.invisible, right.invisible);
    assert_eq!(left.discard_padding_ns, right.discard_padding_ns);
    assert_eq!(left.data, right.data);
}

fn assert_round_trip(name: &str, doc_type: DocType) {
    let mut source = open(name);
    let expected_tracks = source.tracks().to_vec();
    let expected_packets = read_packets(&mut source);
    let expected_duration = source.duration_ns();
    let bytes = remux(&mut open(name), doc_type);

    let mut output = MatroskaReader::new(bytes).expect("re-read document");
    assert_eq!(output.doc_type(), doc_type_str(doc_type));
    assert_eq!(output.doc_type_version(), 4);
    assert_eq!(output.timestamp_scale_ns(), 1_000_000);
    assert_eq!(output.duration_ns(), expected_duration);
    assert_eq!(output.tracks().len(), expected_tracks.len());
    for (left, right) in expected_tracks.iter().zip(output.tracks()) {
        assert_track_eq(left, right);
    }
    let packets = read_packets(&mut output);
    assert_eq!(packets.len(), expected_packets.len());
    for (left, right) in expected_packets.iter().zip(&packets) {
        assert_packet_eq(left, right);
    }
}

fn doc_type_str(doc_type: DocType) -> &'static str {
    match doc_type {
        DocType::Matroska => "matroska",
        DocType::Webm => "webm",
    }
}

#[test]
fn round_trips_webm_opus() {
    assert_round_trip("opus.webm", DocType::Webm);
}

#[test]
fn round_trips_matroska_flac() {
    assert_round_trip("flac.mka", DocType::Matroska);
}

#[test]
fn writes_a_seekable_cue_index() {
    let mut source = open("opus_long.webm");
    let bytes = remux(&mut source, DocType::Webm);
    let mut output = MatroskaReader::new(bytes).expect("re-read document");
    assert!(output.is_seekable());
    let total = read_packets(&mut output).len();
    assert_eq!(total, 101);

    output.seek_to(1_000_000_000).expect("seek to 1 s");
    let packets = read_packets(&mut output);
    assert!(!packets.is_empty());
    assert!(packets[0].timestamp_ns <= 1_000_000_000);
    assert!(packets[0].timestamp_ns > 900_000_000);
    assert!(
        packets
            .iter()
            .any(|packet| packet.timestamp_ns >= 1_000_000_000)
    );

    output.seek_to(0).expect("seek to start");
    let first = output.next_packet().expect("packet").expect("first packet");
    assert_eq!(first.timestamp_ns, -6_500_000);
}

#[test]
fn honours_a_finer_timestamp_scale() {
    let mut source = open("opus.webm");
    let tracks = source.tracks().to_vec();
    let expected = read_packets(&mut source);
    let mut writer = MatroskaWriter::new(MuxerConfig {
        doc_type: DocType::Webm,
        timestamp_scale_ns: 1_000,
    })
    .expect("create writer");
    for track in tracks {
        writer.add_track(track).expect("add track");
    }
    for packet in &expected {
        writer.write_packet(packet.clone()).expect("write packet");
    }
    let bytes = writer.finish().expect("finish document");

    let mut output = MatroskaReader::new(bytes).expect("re-read document");
    assert_eq!(output.timestamp_scale_ns(), 1_000);
    assert_eq!(output.duration_ns(), Some(208_000_000));
    let packets = read_packets(&mut output);
    for (left, right) in expected.iter().zip(&packets) {
        assert_eq!(left.timestamp_ns, right.timestamp_ns);
        assert_eq!(left.duration_ns, right.duration_ns);
        assert_eq!(left.discard_padding_ns, right.discard_padding_ns);
        assert_eq!(left.data, right.data);
    }
}

#[test]
fn rejects_tracks_outside_the_rolling_window() {
    let reader = open("opus.webm");
    let mut track = reader.track(1).expect("track 1").clone();
    track.codec_id = "V_VP9".to_owned();
    track.codec = CodecId::Other;
    let mut writer = MatroskaWriter::new(MuxerConfig::default()).expect("create writer");
    assert!(matches!(
        writer.add_track(track),
        Err(rustmpeg_format::FormatError::UnsupportedCodec(_))
    ));

    let mut track = reader.track(1).expect("track 1").clone();
    track.track_type = rustmpeg_format::TrackType::Video;
    assert!(matches!(
        writer.add_track(track),
        Err(rustmpeg_format::FormatError::Structure(_))
    ));

    let mut track = reader.track(1).expect("track 1").clone();
    track.uid = 0;
    assert!(matches!(
        writer.add_track(track),
        Err(rustmpeg_format::FormatError::Structure(_))
    ));
}

#[test]
fn rejects_packets_that_cannot_be_stored() {
    let mut source = open("opus.webm");
    let tracks = source.tracks().to_vec();
    let packets = read_packets(&mut source);

    let mut writer = MatroskaWriter::new(MuxerConfig::default()).expect("create writer");
    for track in tracks {
        writer.add_track(track).expect("add track");
    }
    let mut foreign = packets[0].clone();
    foreign.track = 99;
    assert!(matches!(
        writer.write_packet(foreign),
        Err(rustmpeg_format::FormatError::UnknownTrack(99))
    ));

    let mut empty = packets[0].clone();
    empty.data = Bytes::new();
    assert!(matches!(
        writer.write_packet(empty),
        Err(rustmpeg_format::FormatError::Block(_))
    ));

    // The first packet cannot precede the codec delay by more than the delay.
    let mut early = packets[0].clone();
    early.timestamp_ns = -6_500_001;
    assert!(matches!(
        writer.write_packet(early),
        Err(rustmpeg_format::FormatError::Structure(_))
    ));

    writer
        .write_packet(packets[1].clone())
        .expect("write later packet");
    assert!(matches!(
        writer.write_packet(packets[0].clone()),
        Err(rustmpeg_format::FormatError::Structure(_))
    ));
}

#[test]
fn rejects_a_zero_timestamp_scale() {
    assert!(matches!(
        MatroskaWriter::new(MuxerConfig {
            timestamp_scale_ns: 0,
            ..MuxerConfig::default()
        }),
        Err(rustmpeg_format::FormatError::Structure(_))
    ));
}
