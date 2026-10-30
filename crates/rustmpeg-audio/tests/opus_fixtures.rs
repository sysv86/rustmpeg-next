// SPDX-License-Identifier: Apache-2.0
//! Fixture tests against real Opus files produced by FFmpeg's `libopus` encoder.
//!
//! The `.opus` files are Ogg containers.  Ogg is outside the supported
//! container matrix, so the reader in `support/ogg.rs` is test-only and exists
//! solely to extract packets and headers for these assertions.
//!
//! `manifest.txt` holds the reference values obtained from
//! `ffprobe -show_packets -show_streams`.  Packet sizes, packet counts, stream
//! duration, pre-skip, and the last packet's end trim are compared against the
//! frames parsed from each packet's TOC.

#[path = "support/ogg.rs"]
mod ogg;

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use rustmpeg_audio::opus::MAX_FRAME_BYTES;
use rustmpeg_audio::{OpusBandwidth, OpusError, OpusMode, OpusPacket, RangeDecoder};

/// Directory holding the fixtures and their ffprobe-derived manifest.
const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/opus");

/// One fixture's expected values, parsed from `manifest.txt`.
#[derive(Debug)]
struct Fixture {
    name: String,
    channels: u8,
    pre_skip: u16,
    frame_samples: u32,
    duration_ts: u32,
    coded_samples: u32,
    configs: BTreeMap<u8, u32>,
    codes: BTreeMap<u8, u32>,
    sizes: Vec<usize>,
    durations: Vec<u32>,
}

#[test]
fn fixtures_match_ffprobe_reference_values() {
    let fixtures = load_manifest();
    assert!(!fixtures.is_empty());

    let mut modes = HashSet::new();
    let mut bandwidths = HashSet::new();
    let mut codes = BTreeMap::new();

    for fixture in &fixtures {
        let path = Path::new(DATA_DIR).join(&fixture.name);
        let data = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        let stream = ogg::read_stream(&data)
            .unwrap_or_else(|error| panic!("{}: cannot read Ogg stream: {error}", fixture.name));

        // Every fixture has exactly two header packets and one audio page.
        assert_eq!(stream.pages.len(), 3, "{}", fixture.name);
        assert!(stream.pages[0].is_bos(), "{}", fixture.name);
        let head = ogg::parse_opus_head(&stream.packets[0].data)
            .unwrap_or_else(|error| panic!("{}: bad OpusHead: {error}", fixture.name));
        assert_eq!(head.version, 1, "{}", fixture.name);
        assert_eq!(head.channels, fixture.channels, "{}", fixture.name);
        assert_eq!(head.pre_skip, fixture.pre_skip, "{}", fixture.name);
        assert_eq!(head.mapping_family, 0, "{}", fixture.name);
        assert_eq!(
            &stream.packets[1].data[..8],
            &b"OpusTags"[..],
            "{}",
            fixture.name
        );

        let audio = &stream.packets[2..];
        assert_eq!(audio.len(), fixture.sizes.len(), "{}", fixture.name);
        assert_eq!(
            stream.pages.last().unwrap().granule,
            i64::from(fixture.duration_ts),
            "{}: final granule",
            fixture.name
        );

        let mut configs = BTreeMap::new();
        let mut observed_codes = BTreeMap::new();
        let mut coded_samples = 0;
        for (index, packet) in audio.iter().enumerate() {
            assert_eq!(packet.page_sequence, 2, "{}", fixture.name);
            assert_eq!(
                packet.granule,
                i64::from(fixture.duration_ts),
                "{}",
                fixture.name
            );
            let parsed = OpusPacket::parse(&packet.data)
                .unwrap_or_else(|error| panic!("{} packet {index}: {error}", fixture.name));
            let info = parsed.info();

            assert_eq!(
                packet.data.len(),
                fixture.sizes[index],
                "{} packet {index} size",
                fixture.name
            );
            assert_eq!(
                info.stereo,
                fixture.channels == 2,
                "{} packet {index} channel count",
                fixture.name
            );
            assert!(
                info.frame_sizes
                    .iter()
                    .all(|&size| usize::from(size) <= MAX_FRAME_BYTES)
            );
            assert!(info.frame_count >= 1 && info.frame_count <= 48);

            // All frames of a packet are inside the packet and contiguous.
            let mut covered = 0;
            for frame in parsed.frames() {
                assert!(frame.len() <= MAX_FRAME_BYTES);
                covered += frame.len();
            }
            assert!(covered + info.padding <= packet.data.len());

            let samples = info.total_duration_us() * 48 / 1000;
            assert_eq!(
                samples, fixture.frame_samples,
                "{} packet {index} duration",
                fixture.name
            );
            coded_samples += samples;

            *configs.entry(info.config).or_insert(0) += 1;
            *observed_codes.entry(packet.data[0] & 3).or_insert(0) += 1;
            modes.insert(info.mode);
            bandwidths.insert(info.bandwidth);
            *codes.entry(packet.data[0] & 3).or_insert(0) += 1;

            // Range decoding of arbitrary symbols is not a bitstream decode in
            // this stage, but it exercises renormalization on real payloads
            // and must never panic or break the range invariant.
            for frame in parsed.frames() {
                let mut decoder = RangeDecoder::new(frame);
                for _ in 0..4 {
                    let _ = decoder.decode_bit_logp(1);
                    let _ = decoder.decode_icdf(&[192, 128, 64, 0], 8);
                    let _ = decoder.decode_uint(1_000);
                    let _ = decoder.decode_raw_bits(4);
                    assert!(decoder.range() > 1 << 23);
                }
            }
        }

        assert_eq!(configs, fixture.configs, "{}: configs", fixture.name);
        assert_eq!(
            observed_codes, fixture.codes,
            "{}: frame-count codes",
            fixture.name
        );
        assert_eq!(
            coded_samples, fixture.coded_samples,
            "{}: summed frame durations",
            fixture.name
        );
        assert_eq!(
            coded_samples,
            fixture.sizes.len() as u32 * fixture.frame_samples,
            "{}: coded total",
            fixture.name
        );

        // RFC 7845: the last packet is trimmed by the final granule position.
        let last_duration = *fixture.durations.last().unwrap();
        assert!(last_duration <= fixture.frame_samples);
        assert_eq!(
            coded_samples - (fixture.frame_samples - last_duration),
            fixture.duration_ts,
            "{}: end trim",
            fixture.name
        );
        assert!(
            fixture.duration_ts >= u32::from(fixture.pre_skip),
            "{}: playable duration",
            fixture.name
        );
    }

    // The committed fixtures cover all three modes, four frame-count codes,
    // both channel counts, CBR, and VBR.
    assert_eq!(modes.len(), 3);
    assert!(modes.contains(&OpusMode::Silk));
    assert!(modes.contains(&OpusMode::Hybrid));
    assert!(modes.contains(&OpusMode::Celt));
    assert!(bandwidths.contains(&OpusBandwidth::Narrowband));
    assert!(bandwidths.contains(&OpusBandwidth::Fullband));
    assert_eq!(codes.keys().copied().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
}

#[test]
fn fixture_header_rejects_bad_magic_and_mapping() {
    assert!(ogg::parse_opus_head(b"NotOpus").is_err());
    let mut head = b"OpusHead\x01\x02\x38\x01\x80\xbb\x00\x00\x00\x00\x00".to_vec();
    assert!(ogg::parse_opus_head(&head).is_ok());
    head[9] = 0;
    assert!(ogg::parse_opus_head(&head).is_err());
    head[9] = 2;
    head[18] = 9;
    assert!(ogg::parse_opus_head(&head).is_err());
}

#[test]
fn opus_head_mapping_families_one_and_255_parse() {
    fn head(family: u8, channels: u8) -> Vec<u8> {
        let mut head = b"OpusHead".to_vec();
        head.extend([1, channels]); // version 1
        head.extend(312u16.to_le_bytes()); // pre-skip
        head.extend(48_000u32.to_le_bytes());
        head.extend(0i16.to_le_bytes()); // output gain
        head.push(family);
        head.push(2); // streams
        head.push(1); // coupled streams
        head.extend((0..channels).collect::<Vec<_>>());
        head
    }

    for family in [1u8, 255] {
        let parsed = ogg::parse_opus_head(&head(family, 2))
            .unwrap_or_else(|error| panic!("mapping family {family}: {error}"));
        assert_eq!(parsed.mapping_family, family);
        assert_eq!(parsed.stream_count, 2);
        assert_eq!(parsed.coupled_count, 1);
        assert_eq!(parsed.channel_mapping, [0, 1]);
    }

    // A family 1 header without its complete mapping table is truncated.
    let mut truncated = head(1, 2);
    truncated.pop();
    assert!(matches!(
        ogg::parse_opus_head(&truncated),
        Err(ogg::OggError::BadOpusHead(_))
    ));
}

#[test]
fn zero_lacing_page_produces_an_empty_packet() {
    let mut page = vec![0u8; 28];
    page[0..4].copy_from_slice(b"OggS");
    page[5] = 0x02; // beginning of stream
    page[14..18].copy_from_slice(&1u32.to_le_bytes()); // serial
    page[26] = 1; // one segment
    page[27] = 0; // zero-length segment
    let crc = ogg::crc32(&page);
    page[22..26].copy_from_slice(&crc.to_le_bytes());

    let pages = ogg::parse_pages(&page).unwrap();
    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].lacing, [0]);
    assert!(pages[0].body.is_empty());

    let stream = ogg::read_stream(&page).unwrap();
    assert_eq!(stream.packets.len(), 1);
    assert!(stream.packets[0].data.is_empty());
}

#[test]
fn empty_ogg_input_is_rejected() {
    assert!(matches!(
        ogg::parse_pages(&[]),
        Err(ogg::OggError::BadHeader(_))
    ));
    assert!(matches!(
        ogg::read_stream(&[]),
        Err(ogg::OggError::BadHeader(_))
    ));
    assert!(ogg::parse_opus_head(&[]).is_err());
}

#[test]
fn bad_page_crc_is_rejected() {
    let mut data = std::fs::read(fixture_path("mono_voip_20ms_16k.opus")).unwrap();
    let pages = ogg::parse_pages(&data).unwrap();
    let last = pages.last().unwrap();
    let index = last.offset + last.length - 1;
    data[index] ^= 0xff;
    match ogg::parse_pages(&data) {
        Err(ogg::OggError::BadCrc { .. }) => {}
        other => panic!("expected BadCrc, got {other:?}"),
    }
}

#[test]
fn truncated_page_is_rejected() {
    let data = std::fs::read(fixture_path("mono_voip_20ms_16k.opus")).unwrap();
    for cut in [1, 10, 100] {
        let truncated = &data[..data.len() - cut];
        match ogg::parse_pages(truncated) {
            Err(ogg::OggError::TruncatedPage(_)) => {}
            other => panic!("cut {cut}: expected TruncatedPage, got {other:?}"),
        }
    }
}

#[test]
fn sequence_gap_is_rejected() {
    let mut data = std::fs::read(fixture_path("mono_voip_20ms_16k.opus")).unwrap();
    let pages = ogg::parse_pages(&data).unwrap();
    let page = &pages[1];
    let replacement = ogg::with_sequence(
        &data[page.offset..page.offset + page.length],
        page.sequence + 2,
    );
    data.splice(page.offset..page.offset + page.length, replacement);
    match ogg::read_stream(&data) {
        Err(ogg::OggError::SequenceGap { .. }) => {}
        other => panic!("expected SequenceGap, got {other:?}"),
    }
}

#[test]
fn unexpected_continuation_flag_is_rejected() {
    let mut data = std::fs::read(fixture_path("mono_voip_20ms_16k.opus")).unwrap();
    let pages = ogg::parse_pages(&data).unwrap();
    let page = &pages[2];
    data[page.offset + 5] |= 0x01;
    data[page.offset + 22..page.offset + 26].fill(0);
    let crc = ogg::crc32(&data[page.offset..page.offset + page.length]);
    data[page.offset + 22..page.offset + 26].copy_from_slice(&crc.to_le_bytes());
    match ogg::read_stream(&data) {
        Err(ogg::OggError::ContinuationWithoutPacket) => {}
        other => panic!("expected ContinuationWithoutPacket, got {other:?}"),
    }
}

#[test]
fn malformed_packets_from_real_fixtures_are_rejected() {
    // A code 1 packet must keep an even payload; dropping one byte breaks it.
    let stream =
        ogg::read_stream(&std::fs::read(fixture_path("mono_voip_40ms_24k.opus")).unwrap()).unwrap();
    let code_1 = stream
        .packets
        .iter()
        .skip(2)
        .find(|packet| packet.data[0] & 3 == 1)
        .expect("fixture contains a code 1 packet");
    let mut broken = code_1.data.clone();
    broken.pop();
    assert!(matches!(
        OpusPacket::parse(&broken),
        Err(OpusError::Malformed(_))
    ));

    // A code 3 packet with a zero frame count is malformed.
    let stream =
        ogg::read_stream(&std::fs::read(fixture_path("mono_voip_60ms_32k_cbr.opus")).unwrap())
            .unwrap();
    let code_3 = stream
        .packets
        .iter()
        .skip(2)
        .find(|packet| packet.data[0] & 3 == 3)
        .expect("fixture contains a code 3 packet");
    let mut broken = code_3.data.clone();
    broken[1] = 0;
    assert!(matches!(
        OpusPacket::parse(&broken),
        Err(OpusError::Malformed(_))
    ));

    // An oversized frame is rejected even if the TOC and framing are valid.
    let mut oversized = vec![0x00];
    oversized.extend(std::iter::repeat_n(0x00, MAX_FRAME_BYTES + 1));
    assert!(matches!(
        OpusPacket::parse(&oversized),
        Err(OpusError::FrameTooLarge { .. })
    ));
}

#[test]
fn arbitrary_ogg_inputs_never_panic() {
    let mut state = 0x0123_4567_89ab_cdefu64;
    for _ in 0..2_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let length = (state as usize) % 400;
        let mut data = vec![0u8; length];
        for byte in &mut data {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        let _ = ogg::parse_pages(&data);
        let _ = ogg::read_stream(&data);
    }
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(DATA_DIR).join(name)
}

fn load_manifest() -> Vec<Fixture> {
    let text = std::fs::read_to_string(Path::new(DATA_DIR).join("manifest.txt"))
        .expect("read fixture manifest");
    let mut fixtures = Vec::new();
    let mut current: Option<Vec<(String, String)>> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let key = parts.next().expect("manifest key");
        let value = parts.collect::<Vec<_>>().join(" ");
        match key {
            "fixture" => {
                assert!(current.is_none(), "nested fixture block");
                current = Some(vec![("name".to_owned(), value)]);
            }
            "end" => {
                let fields = current.take().expect("end without fixture");
                fixtures.push(build_fixture(&fields));
            }
            _ => current
                .as_mut()
                .expect("field outside fixture")
                .push((key.to_owned(), value)),
        }
    }
    assert!(current.is_none(), "unterminated fixture block");
    fixtures
}

fn build_fixture(fields: &[(String, String)]) -> Fixture {
    Fixture {
        name: field(fields, "name").to_owned(),
        channels: field(fields, "channels").parse().unwrap(),
        pre_skip: field(fields, "pre_skip").parse().unwrap(),
        frame_samples: field(fields, "frame_samples").parse().unwrap(),
        duration_ts: field(fields, "duration_ts").parse().unwrap(),
        coded_samples: field(fields, "coded_samples").parse().unwrap(),
        configs: parse_histogram(field(fields, "configs")),
        codes: parse_histogram(field(fields, "codes")),
        sizes: parse_list(field(fields, "sizes")),
        durations: parse_list(field(fields, "durations")),
    }
}

fn field<'a>(fields: &'a [(String, String)], key: &str) -> &'a str {
    fields.iter().find(|(name, _)| name == key).map_or_else(
        || panic!("manifest field {key} is missing"),
        |(_, value)| value,
    )
}

fn parse_list<T: std::str::FromStr>(value: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    value
        .split_whitespace()
        .map(|item| item.parse().unwrap())
        .collect()
}

fn parse_histogram(value: &str) -> BTreeMap<u8, u32> {
    value
        .split_whitespace()
        .map(|item| {
            let (key, count) = item.split_once(':').expect("histogram entry");
            (key.parse().unwrap(), count.parse().unwrap())
        })
        .collect()
}
