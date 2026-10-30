// SPDX-License-Identifier: Apache-2.0
//! Matroska/WebM demuxer conformance against FFmpeg-produced fixtures.
//!
//! The fixtures are described in `tests/data/matroska/README.md`; the expected
//! packet layout was read from `ffprobe` (`-show_packets`).

use std::path::{Path, PathBuf};

use rustmpeg_format::{Bytes, CodecId, MatroskaReader, TrackType};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/matroska")
}

fn open(name: &str) -> MatroskaReader {
    let data = Bytes::from(std::fs::read(data_dir().join(name)).expect("read fixture"));
    MatroskaReader::new(data).expect("parse fixture")
}

fn packets(reader: &mut MatroskaReader) -> Vec<rustmpeg_format::Packet> {
    let mut packets = Vec::new();
    while let Some(packet) = reader.next_packet().expect("read packet") {
        packets.push(packet);
    }
    packets
}

#[test]
fn reads_the_webm_opus_track() {
    let reader = open("opus.webm");
    assert_eq!(reader.doc_type(), "webm");
    assert_eq!(reader.doc_type_version(), 4);
    assert_eq!(reader.timestamp_scale_ns(), 1_000_000);
    assert_eq!(reader.duration_ns(), Some(208_000_000));
    assert_eq!(reader.title(), None);

    assert_eq!(reader.tracks().len(), 1);
    let track = reader.track(1).expect("track 1");
    assert_eq!(track.track_type, TrackType::Audio);
    assert_eq!(track.codec, CodecId::Opus);
    assert_eq!(track.codec_id, "A_OPUS");
    assert!(!track.flag_lacing);
    assert_eq!(track.default_duration_ns, Some(20_000_000));
    assert_eq!(track.codec_delay_ns, 6_500_000);
    assert_eq!(track.seek_pre_roll_ns, 80_000_000);
    assert_eq!(track.language.as_deref(), Some("und"));
    assert_eq!(track.codec_private.len(), 19);
    assert!(track.codec_private.starts_with(b"OpusHead"));
    let audio = track.audio.as_ref().expect("audio settings");
    assert_eq!(audio.sampling_frequency, 48_000.0);
    assert_eq!(audio.output_sampling_frequency, None);
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.bit_depth, Some(16));
    assert!(track.video.is_none());
    assert!(reader.track(2).is_none());
}

#[test]
fn reads_every_webm_opus_packet() {
    let mut reader = open("opus.webm");
    let packets = packets(&mut reader);
    let sizes = [310, 199, 207, 200, 207, 197, 200, 200, 204, 208, 304];
    assert_eq!(packets.len(), sizes.len());
    assert!(packets.iter().all(|packet| packet.track == 1));
    assert!(packets.iter().all(|packet| packet.keyframe));
    assert!(packets.iter().all(|packet| !packet.discardable));
    assert_eq!(
        packets
            .iter()
            .map(|packet| packet.data.len())
            .collect::<Vec<_>>(),
        sizes
    );
    // Timestamps are presentation timestamps: the Opus pre-skip is removed.
    // Block timestamps are stored in milliseconds, so the second block lands
    // on 21 ms and every following block advances by 20 ms.
    let timestamps = [
        -6_500_000,
        14_500_000,
        34_500_000,
        54_500_000,
        74_500_000,
        94_500_000,
        114_500_000,
        134_500_000,
        154_500_000,
        174_500_000,
        194_500_000,
    ];
    assert_eq!(
        packets
            .iter()
            .map(|packet| packet.timestamp_ns)
            .collect::<Vec<_>>(),
        timestamps
    );
    assert!(
        packets
            .iter()
            .take(10)
            .all(|packet| packet.duration_ns == Some(20_000_000))
    );
    let last = &packets[sizes.len() - 1];
    assert_eq!(last.duration_ns, Some(7_000_000));
    assert_eq!(last.discard_padding_ns, Some(13_500_000));
}

#[test]
fn reads_the_matroska_flac_track() {
    let reader = open("flac.mka");
    assert_eq!(reader.doc_type(), "matroska");
    assert_eq!(reader.duration_ns(), Some(200_000_000));
    let track = reader.track(1).expect("track 1");
    assert_eq!(track.track_type, TrackType::Audio);
    assert_eq!(track.codec, CodecId::Flac);
    assert_eq!(track.codec_id, "A_FLAC");
    assert_eq!(track.codec_delay_ns, 0);
    assert_eq!(track.default_duration_ns, Some(21_333_333));
    assert!(track.codec_private.starts_with(b"fLaC"));
    let audio = track.audio.as_ref().expect("audio settings");
    assert_eq!(audio.sampling_frequency, 48_000.0);
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.bit_depth, Some(16));
}

#[test]
fn reads_every_matroska_flac_packet() {
    let mut reader = open("flac.mka");
    let packets = packets(&mut reader);
    let sizes = [325, 329, 358, 359, 362, 367, 382, 383, 360, 192];
    assert_eq!(
        packets
            .iter()
            .map(|packet| packet.data.len())
            .collect::<Vec<_>>(),
        sizes
    );
    // Block timestamps are stored in milliseconds; the nominal frame duration
    // stays 1024 samples at 48 kHz.
    assert_eq!(packets[0].timestamp_ns, 0);
    assert_eq!(packets[1].timestamp_ns, 21_000_000);
    assert_eq!(packets[2].timestamp_ns, 43_000_000);
    assert!(packets.iter().all(|packet| packet.keyframe));
    // Nine full frames, then the final block carries its own duration so the
    // stream ends exactly on the 200 ms segment duration.
    assert!(
        packets
            .iter()
            .take(9)
            .all(|packet| packet.duration_ns == Some(21_333_333))
    );
    assert_eq!(packets[9].duration_ns, Some(8_000_000));
    let total: u64 = packets
        .iter()
        .map(|packet| packet.duration_ns.unwrap())
        .sum();
    // `DefaultDuration` rounds 1024 samples at 48 kHz to 21,333,333 ns, so the
    // sum lands a few nanoseconds short of the 200 ms segment duration.
    assert!(
        (199_999_990..=200_000_000).contains(&total),
        "total {total}"
    );
}

#[test]
fn seeks_with_the_cue_index() {
    // This fixture spans multiple clusters, so the cue index can place the
    // reader near the target instead of rewinding to the first cluster.
    let mut reader = open("opus_long.webm");
    assert!(reader.is_seekable());
    assert_eq!(reader.duration_ns(), Some(2_008_000_000));
    let total = packets(&mut reader).len();
    assert_eq!(total, 101);

    reader.seek_to(1_000_000_000).expect("seek to 1 s");
    let packets = packets(&mut reader);
    assert!(!packets.is_empty());
    // The reader resumes at the cluster boundary before the target; packets
    // below the target are returned so the caller can decode the pre-roll.
    assert!(packets[0].timestamp_ns <= 1_000_000_000);
    assert!(packets[0].timestamp_ns > 900_000_000);
    assert!(
        packets
            .iter()
            .any(|packet| packet.timestamp_ns >= 1_000_000_000)
    );
    assert!(
        packets
            .windows(2)
            .all(|pair| pair[0].timestamp_ns < pair[1].timestamp_ns)
    );

    // Seeking to the start rewinds to the first cluster and preserves the
    // negative pre-skip timestamp of the first packet.
    reader.seek_to(0).expect("seek to start");
    let first = reader.next_packet().expect("packet").expect("first packet");
    assert_eq!(first.timestamp_ns, -6_500_000);
}

/// Encodes an EBML element with a minimal one- or two-octet size field.
fn element(id: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(id.len() + 9 + payload.len());
    out.extend_from_slice(id);
    let size = payload.len();
    if size < 0x7f {
        out.push(0x80 | size as u8);
    } else {
        assert!(size < 0x3fff, "test element is too large");
        out.push(0x40 | (size >> 8) as u8);
        out.push(size as u8);
    }
    out.extend_from_slice(payload);
    out
}

fn uint(id: &[u8], value: u64) -> Vec<u8> {
    let bytes = value.to_be_bytes();
    let first = bytes.iter().position(|&byte| byte != 0).unwrap_or(7);
    element(id, &bytes[first..])
}

fn sint(id: &[u8], value: i64) -> Vec<u8> {
    let bytes = value.to_be_bytes();
    let mut first = 0;
    // Keep one sign bit so the value decodes identically.
    while first < 7
        && ((bytes[first] == 0 && bytes[first + 1] & 0x80 == 0)
            || (bytes[first] == 0xff && bytes[first + 1] & 0x80 != 0))
    {
        first += 1;
    }
    element(id, &bytes[first..])
}

fn f64_element(id: &[u8], value: f64) -> Vec<u8> {
    element(id, &value.to_bits().to_be_bytes())
}

fn simple_block(track: u64, timestamp: i16, flags: u8, frames: &[&[u8]]) -> Vec<u8> {
    if flags & 0x06 == 0 {
        assert_eq!(frames.len(), 1, "unlaced blocks hold one frame");
    }
    let mut payload = Vec::new();
    // Track numbers in these files fit one octet.
    payload.push(0x80 | track as u8);
    payload.extend_from_slice(&timestamp.to_be_bytes());
    payload.push(flags);
    if frames.len() > 1 {
        payload.push((frames.len() - 1) as u8);
        for frame in &frames[..frames.len() - 1] {
            let mut size = frame.len();
            while size >= 255 {
                payload.push(255);
                size -= 255;
            }
            payload.push(size as u8);
        }
    }
    for frame in frames {
        payload.extend_from_slice(frame);
    }
    payload
}

#[test]
fn reads_unknown_size_clusters_and_laced_blocks() {
    let mut ebml = Vec::new();
    ebml.extend(uint(&[0x42, 0x86], 1)); // EBMLVersion
    ebml.extend(element(&[0x42, 0x82], b"webm")); // DocType
    ebml.extend(uint(&[0x42, 0x87], 4)); // DocTypeVersion
    ebml.extend(uint(&[0x42, 0x85], 2)); // DocTypeReadVersion

    let mut info = Vec::new();
    info.extend(uint(&[0x2a, 0xd7, 0xb1], 1_000_000)); // TimestampScale

    let mut audio = Vec::new();
    audio.extend(f64_element(&[0xb5], 48_000.0));
    audio.extend(uint(&[0x9f], 2));

    let mut entry = Vec::new();
    entry.extend(uint(&[0xd7], 1)); // TrackNumber
    entry.extend(uint(&[0x83], 2)); // TrackType audio
    entry.extend(element(&[0x86], b"A_OPUS"));
    entry.extend(uint(&[0x9c], 1)); // FlagLacing
    entry.extend(uint(&[0x23, 0xe3, 0x83], 20_000_000)); // DefaultDuration
    entry.extend(element(&[0xe1], &audio));
    let tracks = element(&[0xae], &entry);

    // First cluster: unknown size, two Xiph-laced frames.
    let mut first = Vec::new();
    first.extend(uint(&[0xe7], 0));
    first.extend(element(
        &[0xa3],
        &simple_block(1, 0, 0x82, &[b"aa", b"bbb"]),
    ));
    let mut first_cluster = Vec::new();
    first_cluster.extend_from_slice(&[0x1f, 0x43, 0xb6, 0x75, 0xff]);
    first_cluster.extend_from_slice(&first);

    // Second cluster: a laced BlockGroup with a trailing ReferenceBlock and
    // DiscardPadding, which must attach to the last frame only.
    let block = simple_block(1, 0, 0x02, &[b"c", b"d", b"e"]);
    let mut group = Vec::new();
    group.extend(element(&[0xa1], &block));
    group.extend(uint(&[0x9b], 20)); // BlockDuration
    group.extend(sint(&[0x75, 0xa2], 5_000_000)); // DiscardPadding
    group.extend(sint(&[0xfb], -1)); // ReferenceBlock
    let mut second = Vec::new();
    second.extend(uint(&[0xe7], 40));
    second.extend(element(&[0xa0], &group));
    let second_cluster = element(&[0x1f, 0x43, 0xb6, 0x75], &second);

    let mut segment = Vec::new();
    segment.extend(element(&[0x15, 0x49, 0xa9, 0x66], &info));
    segment.extend(element(&[0x16, 0x54, 0xae, 0x6b], &tracks));
    segment.extend(first_cluster);
    segment.extend(second_cluster);

    let mut file = element(&[0x1a, 0x45, 0xdf, 0xa3], &ebml);
    file.extend(element(&[0x18, 0x53, 0x80, 0x67], &segment));

    let mut reader = MatroskaReader::new(Bytes::from(file)).expect("parse synthetic file");
    assert_eq!(reader.doc_type(), "webm");
    assert_eq!(reader.doc_type_version(), 4);
    let packets = packets(&mut reader);
    assert_eq!(packets.len(), 5);
    let sizes: Vec<usize> = packets.iter().map(|packet| packet.data.len()).collect();
    assert_eq!(sizes, [2, 3, 1, 1, 1]);
    let timestamps: Vec<i64> = packets.iter().map(|packet| packet.timestamp_ns).collect();
    assert_eq!(
        timestamps,
        [0, 20_000_000, 40_000_000, 46_666_666, 53_333_332]
    );
    assert!(packets[..2].iter().all(|packet| packet.keyframe));
    assert!(
        packets[2..].iter().all(|packet| !packet.keyframe),
        "ReferenceBlock on the BlockGroup must clear the keyframe flag"
    );
    assert!(
        packets[..4]
            .iter()
            .all(|packet| packet.discard_padding_ns.is_none())
    );
    assert_eq!(packets[4].discard_padding_ns, Some(5_000_000));
    assert!(
        packets[..2]
            .iter()
            .all(|packet| packet.duration_ns == Some(20_000_000))
    );
    assert!(
        packets[2..]
            .iter()
            .all(|packet| packet.duration_ns == Some(6_666_666))
    );
}

#[test]
fn rejects_inputs_outside_the_supported_profile() {
    assert!(MatroskaReader::new(Bytes::new()).is_err());
    assert!(MatroskaReader::new(Bytes::from_static(b"not an ebml file")).is_err());

    // A valid EBML header with an unrelated document type.
    let mut ebml = Vec::new();
    ebml.extend(element(&[0x42, 0x82], b"private"));
    let mut file = element(&[0x1a, 0x45, 0xdf, 0xa3], &ebml);
    file.extend(element(&[0x18, 0x53, 0x80, 0x67], &[]));
    assert!(matches!(
        MatroskaReader::new(Bytes::from(file)),
        Err(rustmpeg_format::FormatError::UnsupportedDocType)
    ));
}
