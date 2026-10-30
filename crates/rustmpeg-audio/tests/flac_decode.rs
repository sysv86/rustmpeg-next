// SPDX-License-Identifier: Apache-2.0
//! Fixture-based and RFC 9639 known-answer tests for the native FLAC decoder.
//!
//! The fixture hashes are FNV-1a 64 values over the interleaved, left-aligned
//! `s32le` reference output produced by FFmpeg n9.0.2 (`ffmpeg -i in.flac -f
//! s32le -`).  See `tests/data/README.md` for the exact generation commands.

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use rustmpeg_audio::{FlacDecoder, PcmFrame};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

struct Fixture {
    name: &'static str,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    total_samples: u64,
    frames: u64,
    hash: u64,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "stereo16_44k_1024_l8.flac",
        sample_rate: 44_100,
        channels: 2,
        bits_per_sample: 16,
        total_samples: 11_025,
        frames: 11,
        hash: 0x6817_a094_0eb4_38fa,
    },
    Fixture {
        name: "stereo24_96k_4096_l5.flac",
        sample_rate: 96_000,
        channels: 2,
        bits_per_sample: 24,
        total_samples: 14_400,
        frames: 4,
        hash: 0x9b98_c7df_51da_d18e,
    },
    Fixture {
        name: "mono8_8k_192_l0.flac",
        sample_rate: 8_000,
        channels: 1,
        bits_per_sample: 8,
        total_samples: 4_000,
        frames: 21,
        hash: 0x761a_3c54_58f1_9ba5,
    },
    Fixture {
        name: "surround24_48k_1024_l8.flac",
        sample_rate: 48_000,
        channels: 6,
        bits_per_sample: 24,
        total_samples: 4_800,
        frames: 5,
        hash: 0xf2a1_a4bd_e682_77fa,
    },
    Fixture {
        name: "mono32_48k_4096_l5.flac",
        sample_rate: 48_000,
        channels: 1,
        bits_per_sample: 32,
        total_samples: 4_800,
        frames: 2,
        hash: 0xc277_b468_8f36_9d21,
    },
    Fixture {
        name: "stereo32_44k_1024_l8.flac",
        sample_rate: 44_100,
        channels: 2,
        bits_per_sample: 32,
        total_samples: 2_205,
        frames: 3,
        hash: 0x9c42_adb6_beb8_a1fc,
    },
    Fixture {
        name: "mono16_44k_1024_silence.flac",
        sample_rate: 44_100,
        channels: 1,
        bits_per_sample: 16,
        total_samples: 2_205,
        frames: 3,
        hash: 0xfc78_f33e_577c_80b5,
    },
    Fixture {
        name: "mono16_44k_1024_wasted.flac",
        sample_rate: 44_100,
        channels: 1,
        bits_per_sample: 16,
        total_samples: 4_410,
        frames: 5,
        hash: 0x51f8_3211_b006_4305,
    },
    Fixture {
        name: "verbatim8_8k_192_l0.flac",
        sample_rate: 8_000,
        channels: 1,
        bits_per_sample: 8,
        total_samples: 800,
        frames: 5,
        hash: 0x02b9_3eea_eeb2_c69d,
    },
    Fixture {
        name: "midside16_44k_1024_l8.flac",
        sample_rate: 44_100,
        channels: 2,
        bits_per_sample: 16,
        total_samples: 4_410,
        frames: 5,
        hash: 0xd5a2_1774_1071_65c5,
    },
    Fixture {
        name: "leftside16_44k_1024_l8.flac",
        sample_rate: 44_100,
        channels: 2,
        bits_per_sample: 16,
        total_samples: 4_410,
        frames: 5,
        hash: 0xd5a2_1774_1071_65c5,
    },
];

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

/// Overwrites the 36-bit STREAMINFO total sample count; zero means unknown.
fn set_total_samples(stream: &mut [u8], total: u64) {
    let offset = 4 + 4 + 10;
    let packed = u64::from_be_bytes(stream[offset..offset + 8].try_into().unwrap());
    let packed = (packed & !0x0000_000f_ffff_ffff) | (total & 0x0000_000f_ffff_ffff);
    stream[offset..offset + 8].copy_from_slice(&packed.to_be_bytes());
}

/// FNV-1a 64 over the left-aligned interleaved `s32le` representation.
fn hash_frame(mut hash: u64, frame: &PcmFrame) -> u64 {
    let shift = 32 - u32::from(frame.bits_per_sample);
    for sample_index in 0..frame.block_size() {
        for plane in &frame.planes {
            let sample = plane[sample_index] << shift;
            for byte in sample.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
    }
    hash
}

#[test]
fn decodes_fixtures_byte_exactly() {
    for fixture in FIXTURES {
        let path = fixture_path(fixture.name);
        let file = fs::File::open(&path)
            .unwrap_or_else(|error| panic!("failed to open {}: {error}", path.display()));
        let mut decoder = FlacDecoder::new(file)
            .unwrap_or_else(|error| panic!("failed to parse {}: {error}", fixture.name));

        assert_eq!(
            decoder.stream_info().sample_rate,
            fixture.sample_rate,
            "{}",
            fixture.name
        );
        assert_eq!(
            decoder.stream_info().channels,
            fixture.channels,
            "{}",
            fixture.name
        );
        assert_eq!(
            decoder.stream_info().bits_per_sample,
            fixture.bits_per_sample,
            "{}",
            fixture.name
        );
        assert_eq!(
            decoder.stream_info().total_samples,
            Some(fixture.total_samples),
            "{}",
            fixture.name
        );

        let mut hash = FNV_OFFSET;
        let mut frames = 0u64;
        let mut next_frame = 0u64;

        while let Some(frame) = decoder
            .next_frame()
            .unwrap_or_else(|error| panic!("{} failed at frame {frames}: {error}", fixture.name))
        {
            assert_eq!(frame.sample_rate, fixture.sample_rate, "{}", fixture.name);
            assert_eq!(frame.channels, fixture.channels, "{}", fixture.name);
            assert_eq!(
                frame.bits_per_sample, fixture.bits_per_sample,
                "{}",
                fixture.name
            );
            assert_eq!(frame.frame_number, frames, "{}", fixture.name);
            assert_eq!(frame.sample_number, next_frame, "{}", fixture.name);
            assert_eq!(frame.planes.len(), usize::from(fixture.channels));
            assert!(frame.block_size() > 0, "{}", fixture.name);
            for plane in &frame.planes {
                assert_eq!(plane.len(), frame.block_size(), "{}", fixture.name);
            }
            for channel in 0..usize::from(fixture.channels) {
                assert_eq!(frame.plane(channel), Some(frame.planes[channel].as_slice()));
            }

            hash = hash_frame(hash, &frame);
            next_frame += frame.block_size() as u64;
            frames += 1;
        }

        assert_eq!(frames, fixture.frames, "{} frame count", fixture.name);
        assert_eq!(next_frame, fixture.total_samples, "{}", fixture.name);
        assert_eq!(hash, fixture.hash, "{} PCM hash", fixture.name);
        assert_eq!(decoder.next_frame().unwrap(), None);
    }
}

#[test]
fn truncated_fixtures_are_rejected() {
    for fixture in FIXTURES {
        let data = fs::read(fixture_path(fixture.name)).unwrap();
        let truncated = &data[..data.len() - 10];
        let mut decoder = FlacDecoder::new(Cursor::new(truncated.to_vec()))
            .unwrap_or_else(|error| panic!("{} metadata: {error}", fixture.name));
        let mut failed = false;
        loop {
            match decoder.next_frame() {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        assert!(failed, "{} truncation was not detected", fixture.name);
    }
}

#[test]
fn understated_sample_counts_are_rejected_at_the_frame_boundary() {
    // A stream whose last frame is complete but whose STREAMINFO announces
    // more samples than it contains is truncated, not clean.
    for fixture in FIXTURES {
        let mut data = fs::read(fixture_path(fixture.name)).unwrap();
        set_total_samples(&mut data, fixture.total_samples + 1);

        let mut decoder = FlacDecoder::new(Cursor::new(data))
            .unwrap_or_else(|error| panic!("{} metadata: {error}", fixture.name));
        let mut failed = false;
        loop {
            match decoder.next_frame() {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        assert!(
            failed,
            "{} undersized declaration was not detected",
            fixture.name
        );
    }
}

#[test]
fn unknown_sample_counts_end_cleanly() {
    let fixture = &FIXTURES[0];
    let mut data = fs::read(fixture_path(fixture.name)).unwrap();
    set_total_samples(&mut data, 0);

    let mut decoder = FlacDecoder::new(Cursor::new(data)).unwrap();
    assert_eq!(decoder.stream_info().total_samples, None);
    let mut samples = 0u64;
    while let Some(frame) = decoder.next_frame().unwrap() {
        samples += frame.block_size() as u64;
    }
    assert_eq!(samples, fixture.total_samples, "{}", fixture.name);
}

// --------------------------------------------------------------------------
// RFC 9639 Appendix D known-answer tests.
//
// These streams are reproduced as conformance test vectors.  Source: RFC 9639
// (IETF Trust, December 2024), <https://www.rfc-editor.org/rfc/rfc9639>.  Code
// components of IETF documents are provided under the Revised BSD License
// described in Section 4.e of the IETF Trust Legal Provisions.
// --------------------------------------------------------------------------

const RFC9639_EXAMPLE_1: &[u8] = &[
    0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x0f, 0x00,
    0x00, 0x0f, 0x0a, 0xc4, 0x42, 0xf0, 0x00, 0x00, 0x00, 0x01, 0x3e, 0x84, 0xb4, 0x18, 0x07, 0xdc,
    0x69, 0x03, 0x07, 0x58, 0x6a, 0x3d, 0xad, 0x1a, 0x2e, 0x0f, 0xff, 0xf8, 0x69, 0x18, 0x00, 0x00,
    0xbf, 0x03, 0x58, 0xfd, 0x03, 0x12, 0x8b, 0xaa, 0x9a,
];

const RFC9639_EXAMPLE_2: &[u8] = &[
    0x66, 0x4c, 0x61, 0x43, 0x00, 0x00, 0x00, 0x22, 0x00, 0x10, 0x00, 0x10, 0x00, 0x00, 0x17, 0x00,
    0x00, 0x44, 0x0a, 0xc4, 0x42, 0xf0, 0x00, 0x00, 0x00, 0x13, 0xd5, 0xb0, 0x56, 0x49, 0x75, 0xe9,
    0x8b, 0x8d, 0x8b, 0x93, 0x04, 0x22, 0x75, 0x7b, 0x81, 0x03, 0x03, 0x00, 0x00, 0x12, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10,
    0x04, 0x00, 0x00, 0x3a, 0x20, 0x00, 0x00, 0x00, 0x72, 0x65, 0x66, 0x65, 0x72, 0x65, 0x6e, 0x63,
    0x65, 0x20, 0x6c, 0x69, 0x62, 0x46, 0x4c, 0x41, 0x43, 0x20, 0x31, 0x2e, 0x33, 0x2e, 0x33, 0x20,
    0x32, 0x30, 0x31, 0x39, 0x30, 0x38, 0x30, 0x34, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
    0x54, 0x49, 0x54, 0x4c, 0x45, 0x3d, 0xd7, 0xa9, 0xd7, 0x9c, 0xd7, 0x95, 0xd7, 0x9d, 0x81, 0x00,
    0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xf8, 0x69, 0x98, 0x00, 0x0f, 0x99, 0x12,
    0x08, 0x67, 0x01, 0x62, 0x3d, 0x14, 0x42, 0x99, 0x8f, 0x5d, 0xf7, 0x0d, 0x6f, 0xe0, 0x0c, 0x17,
    0xca, 0xeb, 0x21, 0x00, 0x0e, 0xe7, 0xa7, 0x7a, 0x24, 0xa1, 0x59, 0x0c, 0x12, 0x17, 0xb6, 0x03,
    0x09, 0x7b, 0x78, 0x4f, 0xaa, 0x9a, 0x33, 0xd2, 0x85, 0xe0, 0x70, 0xad, 0x5b, 0x1b, 0x48, 0x51,
    0xb4, 0x01, 0x0d, 0x99, 0xd2, 0xcd, 0x1a, 0x68, 0xf1, 0xe6, 0xb8, 0x10, 0xff, 0xf8, 0x69, 0x18,
    0x01, 0x02, 0xa4, 0x02, 0xc3, 0x82, 0xc4, 0x0b, 0xc1, 0x4a, 0x03, 0xee, 0x48, 0xdd, 0x03, 0xb6,
    0x7c, 0x13, 0x30,
];

const RFC9639_EXAMPLE_3: &[u8] = &[
    0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00, 0x1f, 0x00,
    0x00, 0x1f, 0x07, 0xd0, 0x00, 0x70, 0x00, 0x00, 0x00, 0x18, 0xf8, 0xf9, 0xe3, 0x96, 0xf5, 0xcb,
    0xcf, 0xc6, 0xdc, 0x80, 0x7f, 0x99, 0x77, 0x90, 0x6b, 0x32, 0xff, 0xf8, 0x68, 0x02, 0x00, 0x17,
    0xe9, 0x44, 0x00, 0x4f, 0x6f, 0x31, 0x3d, 0x10, 0x47, 0xd2, 0x27, 0xcb, 0x6d, 0x09, 0x08, 0x31,
    0x45, 0x2b, 0xdc, 0x28, 0x22, 0x22, 0x80, 0x57, 0xa3,
];

#[test]
fn decodes_rfc9639_example_1() {
    let mut decoder = FlacDecoder::new(Cursor::new(RFC9639_EXAMPLE_1)).unwrap();
    let info = decoder.stream_info();
    assert_eq!(info.sample_rate, 44_100);
    assert_eq!(info.channels, 2);
    assert_eq!(info.bits_per_sample, 16);
    assert_eq!(info.total_samples, Some(1));

    let frame = decoder.next_frame().unwrap().unwrap();
    assert_eq!(frame.sample_rate, 44_100);
    assert_eq!(frame.channels, 2);
    assert_eq!(frame.bits_per_sample, 16);
    assert_eq!(frame.frame_number, 0);
    assert_eq!(frame.sample_number, 0);
    assert_eq!(frame.block_size(), 1);
    // Verbatim subframes with two and four wasted bits (Appendix D.1.4).
    assert_eq!(frame.planes, vec![vec![25_588], vec![10_416]]);
    assert_eq!(decoder.next_frame().unwrap(), None);
}

#[test]
fn decodes_rfc9639_example_2() {
    // Expected samples from Appendix D.2, including the side-right
    // decorrelation of the first frame and the independent last frame.
    let left = [
        10_372, 18_041, 14_942, 17_876, 15_627, 17_899, 16_242, 18_077, 16_824, 18_263, 17_295,
        -14_418, -15_201, -14_508, -15_195, -14_818, -15_486, -15_349, -16_054,
    ];
    let right = [
        6_070, 10_545, 8_743, 10_449, 9_143, 10_463, 9_502, 10_569, 9_840, 10_680, 10_113, -8_428,
        -8_895, -8_476, -8_896, -8_653, -9_072, -8_958, -9_410,
    ];

    let mut decoder = FlacDecoder::new(Cursor::new(RFC9639_EXAMPLE_2)).unwrap();
    assert_eq!(decoder.stream_info().total_samples, Some(19));

    let first = decoder.next_frame().unwrap().unwrap();
    assert_eq!(first.sample_number, 0);
    assert_eq!(first.block_size(), 16);
    assert_eq!(first.planes[0].as_slice(), &left[..16]);
    assert_eq!(first.planes[1].as_slice(), &right[..16]);

    let second = decoder.next_frame().unwrap().unwrap();
    assert_eq!(second.frame_number, 1);
    assert_eq!(second.sample_number, 16);
    assert_eq!(second.block_size(), 3);
    assert_eq!(second.planes[0].as_slice(), &left[16..]);
    assert_eq!(second.planes[1].as_slice(), &right[16..]);

    assert_eq!(decoder.next_frame().unwrap(), None);
}

#[test]
fn decodes_rfc9639_example_3() {
    // LPC order 3 with a four-partition residual, including an escaped
    // partition (Appendix D.3).
    let expected = [
        0i32, 79, 111, 78, 8, -61, -90, -68, -13, 42, 67, 53, 13, -27, -46, -38, -12, 14, 24, 19,
        6, -4, -5, 0,
    ];

    let mut decoder = FlacDecoder::new(Cursor::new(RFC9639_EXAMPLE_3)).unwrap();
    let info = decoder.stream_info();
    assert_eq!(info.sample_rate, 32_000);
    assert_eq!(info.channels, 1);
    assert_eq!(info.bits_per_sample, 8);
    assert_eq!(info.total_samples, Some(24));

    let frame = decoder.next_frame().unwrap().unwrap();
    assert_eq!(frame.block_size(), 24);
    assert_eq!(frame.planes, vec![expected.to_vec()]);
    assert_eq!(decoder.next_frame().unwrap(), None);
}
