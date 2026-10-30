// SPDX-License-Identifier: Apache-2.0
//! Packet framing tests for RFC 6716, Section 3.
//!
//! The expected configuration table is written out independently of the
//! implementation so that a changed mapping is caught.

use rustmpeg_audio::opus::{MAX_FRAME_BYTES, MAX_PACKET_BYTES};
use rustmpeg_audio::{OpusBandwidth, OpusError, OpusMode, OpusPacket};

const N: OpusBandwidth = OpusBandwidth::Narrowband;
const M: OpusBandwidth = OpusBandwidth::Mediumband;
const W: OpusBandwidth = OpusBandwidth::Wideband;
const SW: OpusBandwidth = OpusBandwidth::SuperWideband;
const F: OpusBandwidth = OpusBandwidth::Fullband;
const SILK: OpusMode = OpusMode::Silk;
const HYBRID: OpusMode = OpusMode::Hybrid;
const CELT: OpusMode = OpusMode::Celt;

/// RFC 6716, Table 2: configuration, mode, bandwidth, frame duration.
const CONFIG_TABLE: [(OpusMode, OpusBandwidth, u32); 32] = [
    (SILK, N, 10_000),
    (SILK, N, 20_000),
    (SILK, N, 40_000),
    (SILK, N, 60_000),
    (SILK, M, 10_000),
    (SILK, M, 20_000),
    (SILK, M, 40_000),
    (SILK, M, 60_000),
    (SILK, W, 10_000),
    (SILK, W, 20_000),
    (SILK, W, 40_000),
    (SILK, W, 60_000),
    (HYBRID, SW, 10_000),
    (HYBRID, SW, 20_000),
    (HYBRID, F, 10_000),
    (HYBRID, F, 20_000),
    (CELT, N, 2_500),
    (CELT, N, 5_000),
    (CELT, N, 10_000),
    (CELT, N, 20_000),
    (CELT, W, 2_500),
    (CELT, W, 5_000),
    (CELT, W, 10_000),
    (CELT, W, 20_000),
    (CELT, SW, 2_500),
    (CELT, SW, 5_000),
    (CELT, SW, 10_000),
    (CELT, SW, 20_000),
    (CELT, F, 2_500),
    (CELT, F, 5_000),
    (CELT, F, 10_000),
    (CELT, F, 20_000),
];

#[test]
fn toc_configuration_table_matches_rfc() {
    for (config, &(mode, bandwidth, duration)) in CONFIG_TABLE.iter().enumerate() {
        let config = u8::try_from(config).unwrap();
        let toc = [config << 3];
        let packet = OpusPacket::parse(&toc).unwrap();
        let info = packet.info();
        assert_eq!(info.config, config, "config {config}");
        assert_eq!(info.mode, mode, "config {config}");
        assert_eq!(info.bandwidth, bandwidth, "config {config}");
        assert_eq!(info.frame_duration_us, duration, "config {config}");
        assert!(!info.stereo, "config {config}");
        assert_eq!(info.frame_count, 1, "config {config}");
        assert_eq!(info.frame_sizes, [0], "config {config}");
        assert_eq!(info.padding, 0, "config {config}");
        assert_eq!(info.total_duration_us(), duration, "config {config}");
        assert_eq!(packet.frames().count(), 1, "config {config}");
        assert!(packet.frames().next().unwrap().is_empty());
    }
}

#[test]
fn toc_splits_config_stereo_and_frame_code() {
    // config 5 (SILK MB 20 ms), stereo flag set, code 2.
    let toc = (5 << 3) | 0x04 | 0x02;
    let data = [toc, 0x00];
    let packet = OpusPacket::parse(&data).unwrap();
    let info = packet.info();
    assert_eq!(info.config, 5);
    assert_eq!(info.mode, OpusMode::Silk);
    assert_eq!(info.bandwidth, OpusBandwidth::Mediumband);
    assert_eq!(info.frame_duration_us, 20_000);
    assert!(info.stereo);
    assert_eq!(info.frame_count, 2);
    assert_eq!(info.frame_sizes, [0, 0]);
    assert_eq!(info.total_duration_us(), 40_000);
}

#[test]
fn code_0_single_frame() {
    let data = [0x00, 0x11, 0x22, 0x33];
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().frame_sizes, [3]);
    assert_eq!(packet.frames().collect::<Vec<_>>(), vec![&data[1..]]);
}

#[test]
fn code_0_allows_zero_length_frame() {
    let packet = OpusPacket::parse(&[0x00]).unwrap();
    assert_eq!(packet.info().frame_sizes, [0]);
    assert_eq!(packet.frames().count(), 1);
    assert!(packet.frames().next().unwrap().is_empty());
}

#[test]
fn code_0_enforces_frame_size_limit() {
    let mut data = vec![0x00; 1276];
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().frame_sizes, [1275]);

    data.push(0x00);
    assert!(matches!(
        OpusPacket::parse(&data),
        Err(OpusError::FrameTooLarge { size: 1276 })
    ));
}

#[test]
fn code_1_splits_payload_evenly() {
    let data = [0x01, 0xaa, 0xbb, 0xcc, 0xdd];
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().frame_count, 2);
    assert_eq!(packet.info().frame_sizes, [2, 2]);
    assert_eq!(
        packet.frames().collect::<Vec<_>>(),
        vec![&data[1..3], &data[3..5]]
    );
}

#[test]
fn code_1_allows_two_empty_frames() {
    let packet = OpusPacket::parse(&[0x01]).unwrap();
    assert_eq!(packet.info().frame_sizes, [0, 0]);
    let frames: Vec<_> = packet.frames().collect();
    assert_eq!(frames.len(), 2);
    assert!(frames.iter().all(|frame| frame.is_empty()));
}

#[test]
fn code_1_requires_even_payload() {
    let data = [0x01, 0xaa, 0xbb, 0xcc];
    assert!(matches!(
        OpusPacket::parse(&data),
        Err(OpusError::Malformed(_))
    ));
}

#[test]
fn code_1_enforces_frame_size_limit() {
    let packet = vec![0x01; 2551];
    let parsed = OpusPacket::parse(&packet).unwrap();
    assert_eq!(parsed.info().frame_sizes, [1275, 1275]);

    let packet = vec![0x01; 2553];
    assert!(matches!(
        OpusPacket::parse(&packet),
        Err(OpusError::FrameTooLarge { size: 1276 })
    ));
}

#[test]
fn code_2_one_byte_length() {
    let data = [0x02, 0x03, 0xaa, 0xbb, 0xcc, 0xdd, 0xee];
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().frame_sizes, [3, 2]);
    assert_eq!(
        packet.frames().collect::<Vec<_>>(),
        vec![&data[2..5], &data[5..7]]
    );
}

#[test]
fn code_2_two_byte_length() {
    // 0xfc selects the two-byte form: 0x01 * 4 + 0xfc = 256.
    let mut data = vec![0x02, 0xfc, 0x01];
    data.extend(std::iter::repeat_n(0x5a, 256));
    data.push(0xa5);
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().frame_sizes, [256, 1]);
}

#[test]
fn code_2_allows_two_zero_length_frames() {
    let packet = OpusPacket::parse(&[0x02, 0x00]).unwrap();
    assert_eq!(packet.info().frame_sizes, [0, 0]);
}

#[test]
fn code_2_rejects_length_beyond_payload() {
    // A two-byte packet whose length field is 1..=251 is invalid because no
    // payload bytes remain, and 0 is the only valid two-byte case.
    assert!(matches!(
        OpusPacket::parse(&[0x02, 0x01]),
        Err(OpusError::Malformed(_))
    ));
    assert!(matches!(
        OpusPacket::parse(&[0x02, 0xfb]),
        Err(OpusError::Malformed(_))
    ));
}

#[test]
fn code_2_rejects_truncated_two_byte_length() {
    assert!(matches!(
        OpusPacket::parse(&[0x02, 0xfc]),
        Err(OpusError::Truncated(_))
    ));
}

#[test]
fn code_2_enforces_implicit_frame_limit() {
    let mut data = vec![0x02, 0x00];
    data.extend(std::iter::repeat_n(0x00, 1276));
    assert!(matches!(
        OpusPacket::parse(&data),
        Err(OpusError::FrameTooLarge { size: 1276 })
    ));
}

#[test]
fn code_3_cbr_equal_frames() {
    let data = [0x03, 0x02, 0xaa, 0xbb, 0xcc, 0xdd];
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().frame_count, 2);
    assert_eq!(packet.info().frame_sizes, [2, 2]);
    assert_eq!(
        packet.frames().collect::<Vec<_>>(),
        vec![&data[2..4], &data[4..6]]
    );
}

#[test]
fn code_3_requires_at_least_the_count_byte() {
    assert!(matches!(
        OpusPacket::parse(&[0x03]),
        Err(OpusError::Truncated(_))
    ));
}

#[test]
fn code_3_rejects_zero_frame_count() {
    assert!(matches!(
        OpusPacket::parse(&[0x03, 0x00]),
        Err(OpusError::Malformed(_))
    ));
}

#[test]
fn code_3_rejects_cbr_payload_not_multiple_of_count() {
    assert!(matches!(
        OpusPacket::parse(&[0x03, 0x02, 1, 2, 3]),
        Err(OpusError::Malformed(_))
    ));
}

#[test]
fn code_3_accepts_48_frames_of_2_5_ms() {
    let mut data = vec![0x83, 48]; // config 16 (CELT NB 2.5 ms), code 3.
    data.extend(std::iter::repeat_n(0x00, 48));
    let packet = OpusPacket::parse(&data).unwrap();
    let info = packet.info();
    assert_eq!(info.frame_count, 48);
    assert_eq!(info.frame_sizes, vec![1; 48]);
    assert_eq!(info.total_duration_us(), 120_000);
    assert_eq!(packet.frames().count(), 48);
}

#[test]
fn code_3_vbr_accepts_48_frames() {
    // VBR with 48 zero-length frames: the count byte sets VBR, and 47 explicit
    // one-byte lengths precede the implicit last frame.
    let mut data = vec![0x83, 0xb0]; // config 16 (CELT NB 2.5 ms), code 3.
    data.extend(std::iter::repeat_n(0x00, 47));
    let packet = OpusPacket::parse(&data).unwrap();
    let info = packet.info();
    assert_eq!(info.frame_count, 48);
    assert_eq!(info.frame_sizes, vec![0; 48]);
    assert_eq!(info.total_duration_us(), 120_000);
    assert_eq!(packet.frames().count(), 48);
}

#[test]
fn code_1_and_2_accept_two_60_ms_frames() {
    // Config 3 is SILK narrowband with 60 ms frames; two frames hit the
    // 120 ms packet boundary exactly.
    let data = [(3 << 3) | 0x01, 0xaa, 0xbb];
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().frame_duration_us, 60_000);
    assert_eq!(packet.info().total_duration_us(), 120_000);
    assert_eq!(packet.info().frame_sizes, [1, 1]);

    let data = [(3 << 3) | 0x02, 0x01, 0xaa, 0xbb];
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().total_duration_us(), 120_000);
    assert_eq!(packet.info().frame_sizes, [1, 1]);
}

#[test]
fn frame_size_boundary_accepts_1275_and_rejects_1276() {
    // The two-byte length encoding reaches exactly 1275 (0xff, 0xff).
    let mut data = vec![0x02, 0xff, 0xff];
    data.extend(std::iter::repeat_n(0x00, 1275));
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().frame_sizes, [1275, 0]);

    // The second frame is implicit, so it can carry the first oversized size.
    let mut oversized = vec![0x02, 0x00];
    oversized.extend(std::iter::repeat_n(0x00, 1276));
    assert!(matches!(
        OpusPacket::parse(&oversized),
        Err(OpusError::FrameTooLarge { size: 1276 })
    ));
}

#[test]
fn code_3_rejects_49_frames() {
    let mut data = vec![0x83, 49];
    data.extend(std::iter::repeat_n(0x00, 49));
    assert!(matches!(
        OpusPacket::parse(&data),
        Err(OpusError::TooManyFrames { count: 49 })
    ));
}

#[test]
fn code_3_rejects_more_than_120_ms() {
    // config 2 is SILK NB with 40 ms frames; four frames exceed the cap.
    assert!(matches!(
        OpusPacket::parse(&[0x13, 0x04]),
        Err(OpusError::DurationTooLong {
            duration_us: 160_000
        })
    ));
}

#[test]
fn code_3_vbr_lengths() {
    let data = [0x03, 0x83, 0x02, 0x01, 0xaa, 0xbb, 0xcc, 0xdd];
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().frame_count, 3);
    assert_eq!(packet.info().frame_sizes, [2, 1, 1]);
    assert_eq!(
        packet.frames().collect::<Vec<_>>(),
        vec![&data[4..6], &data[6..7], &data[7..8]]
    );
}

#[test]
fn code_3_vbr_two_byte_length() {
    let mut data = vec![0x03, 0x82, 0xfc, 0x00];
    data.extend(std::iter::repeat_n(0x5a, 255));
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().frame_sizes, [252, 3]);
}

#[test]
fn code_3_vbr_rejects_lengths_beyond_payload() {
    assert!(matches!(
        OpusPacket::parse(&[0x03, 0x82, 0x05, 0xaa, 0xbb]),
        Err(OpusError::Malformed(_))
    ));
}

#[test]
fn code_3_vbr_rejects_truncated_length() {
    assert!(matches!(
        OpusPacket::parse(&[0x03, 0x82, 0xfc]),
        Err(OpusError::Truncated(_))
    ));
}

#[test]
fn code_3_vbr_enforces_last_frame_limit() {
    let mut data = vec![0x03, 0x81];
    data.extend(std::iter::repeat_n(0x00, 1276));
    assert!(matches!(
        OpusPacket::parse(&data),
        Err(OpusError::FrameTooLarge { size: 1276 })
    ));
}

#[test]
fn code_3_cbr_enforces_frame_limit() {
    let mut data = vec![0x03, 0x01];
    data.extend(std::iter::repeat_n(0x00, 1276));
    assert!(matches!(
        OpusPacket::parse(&data),
        Err(OpusError::FrameTooLarge { size: 1276 })
    ));
}

#[test]
fn code_3_single_byte_padding() {
    let data = [0x03, 0x42, 0x03, 0xaa, 0xbb, 0x00, 0x00, 0x00];
    let packet = OpusPacket::parse(&data).unwrap();
    let info = packet.info();
    assert_eq!(info.frame_count, 2);
    assert_eq!(info.frame_sizes, [1, 1]);
    assert_eq!(info.padding, 3);
    assert_eq!(
        packet.frames().collect::<Vec<_>>(),
        vec![&data[3..4], &data[4..5]]
    );
}

#[test]
fn code_3_multi_byte_padding() {
    // 0xff, 0x00 encodes 254 trailing padding bytes and two length bytes.
    let mut data = vec![0x03, 0x41, 0xff, 0x00];
    data.extend(std::iter::repeat_n(0x5a, 22));
    data.extend(std::iter::repeat_n(0x00, 254));
    let packet = OpusPacket::parse(&data).unwrap();
    let info = packet.info();
    assert_eq!(info.frame_count, 1);
    assert_eq!(info.frame_sizes, [22]);
    assert_eq!(info.padding, 254);
    assert_eq!(packet.frames().next().unwrap(), &data[4..26]);
}

#[test]
fn code_3_padding_chain_of_255_bytes() {
    // 0xff, 0xff, 0x01 encodes 254 + 254 + 1 = 509 padding bytes and three
    // padding length bytes (RFC 6716, Section 3.2.5).
    let mut data = vec![0x03, 0x41, 0xff, 0xff, 0x01, 0xaa];
    data.extend(std::iter::repeat_n(0x00, 509));
    let packet = OpusPacket::parse(&data).unwrap();
    let info = packet.info();
    assert_eq!(info.frame_count, 1);
    assert_eq!(info.frame_sizes, [1]);
    assert_eq!(info.padding, 509);
    assert_eq!(packet.frames().next().unwrap(), &data[5..6]);
}

#[test]
fn code_3_accepts_nonzero_padding_bytes() {
    // The encoder must write zeroes, but the decoder must accept any value.
    let data = [0x03, 0x41, 0x02, 0xaa, 0x55, 0x55];
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().padding, 2);
    assert_eq!(packet.info().frame_sizes, [1]);
}

#[test]
fn code_3_rejects_padding_beyond_packet() {
    assert!(matches!(
        OpusPacket::parse(&[0x03, 0x41, 0x05]),
        Err(OpusError::PaddingTooLarge { .. })
    ));
}

#[test]
fn code_3_rejects_truncated_padding_length() {
    assert!(matches!(
        OpusPacket::parse(&[0x03, 0x41, 0xff]),
        Err(OpusError::Truncated(_))
    ));
}

#[test]
fn code_3_vbr_with_padding() {
    let data = [0x03, 0xc2, 0x02, 0x01, 0xaa, 0xbb, 0x00, 0x00];
    let packet = OpusPacket::parse(&data).unwrap();
    assert_eq!(packet.info().padding, 2);
    assert_eq!(packet.info().frame_sizes, [1, 1]);
    assert_eq!(
        packet.frames().collect::<Vec<_>>(),
        vec![&data[4..5], &data[5..6]]
    );
}

#[test]
fn frames_borrow_from_the_packet() {
    let data = [0x02, 0x03, 0x01, 0x02, 0x03, 0x04, 0x05];
    let packet = OpusPacket::parse(&data).unwrap();
    let base = data.as_ptr() as usize;
    let frames: Vec<&[u8]> = packet.frames().collect();
    assert_eq!(frames.len(), 2);
    for frame in &frames {
        let start = frame.as_ptr() as usize;
        assert!(start >= base);
        assert!(start + frame.len() <= base + data.len());
    }
    assert_eq!(frames[0].as_ptr(), data[2..].as_ptr());
}

#[test]
fn empty_packet_is_rejected() {
    assert!(matches!(
        OpusPacket::parse(&[]),
        Err(OpusError::EmptyPacket)
    ));
}

#[test]
fn oversized_packet_is_rejected_before_parsing() {
    let data = vec![0u8; MAX_PACKET_BYTES + 1];
    assert!(matches!(
        OpusPacket::parse(&data),
        Err(OpusError::PacketTooLarge { size }) if size == MAX_PACKET_BYTES + 1
    ));
}

#[test]
fn total_duration_saturates_for_manual_values() {
    let packet = OpusPacket::parse(&[0x00]).unwrap();
    let mut info = packet.info().clone();
    info.frame_duration_us = u32::MAX;
    info.frame_count = 48;
    assert_eq!(info.total_duration_us(), u32::MAX);
}

#[test]
fn all_packets_of_length_at_most_two_are_handled() {
    for value in 0..=0xffu16 {
        check_packet(&[value as u8]);
    }
    for value in 0..=0xffffu16 {
        check_packet(&value.to_le_bytes());
    }
    check_packet(&[]);
}

#[test]
fn pseudo_random_packets_never_panic() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..20_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let length = (state as usize) % 300;
        let mut data = vec![0u8; length];
        for byte in &mut data {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        check_packet(&data);
    }
}

/// Parses a packet and verifies the invariants every successful parse must
/// uphold.  A panic here fails the test.
fn check_packet(data: &[u8]) {
    if let Ok(packet) = OpusPacket::parse(data) {
        let info = packet.info();
        assert_eq!(usize::from(info.frame_count), info.frame_sizes.len());
        let frames: Vec<&[u8]> = packet.frames().collect();
        assert_eq!(usize::from(info.frame_count), frames.len());
        assert!(info.frame_count >= 1 && info.frame_count <= 48);
        assert!(info.total_duration_us() <= 120_000);

        let mut total = 0usize;
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(
                frame.len(),
                usize::from(info.frame_sizes[index]),
                "frame {index} length"
            );
            assert!(frame.len() <= MAX_FRAME_BYTES, "frame {index} size");
            total += frame.len();
            if index > 0 {
                // Frames are contiguous and in packet order.
                assert_eq!(
                    frames[index - 1]
                        .as_ptr()
                        .wrapping_add(frames[index - 1].len()),
                    frame.as_ptr(),
                    "frames {} and {index} are not contiguous",
                    index - 1
                );
            }
        }

        // Every payload byte is covered exactly once, and the packet ends with
        // the declared number of padding bytes.
        let base = data.as_ptr() as usize;
        let first_offset = frames[0].as_ptr() as usize - base;
        assert_eq!(
            first_offset + total + info.padding,
            data.len(),
            "packet span is not exact"
        );
        let last = frames.last().unwrap();
        assert_eq!(
            last.as_ptr() as usize + last.len(),
            base + data.len() - info.padding,
            "last frame does not end at the padding"
        );
    }
}
