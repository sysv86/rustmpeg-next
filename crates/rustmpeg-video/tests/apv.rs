// SPDX-License-Identifier: Apache-2.0
//! APV encoder/decoder round-trip tests.

use rustmpeg_video::apv::{ApvChromaFormat, ApvConfig, ApvDecoder, ApvEncoder, ApvFrame};

fn make_frame(chroma: ApvChromaFormat, bit_depth: u8, width: u32, height: u32) -> ApvFrame {
    let mut frame = ApvFrame::new(chroma, bit_depth, width, height).expect("frame");
    let maximum = (1_u32 << bit_depth) - 1;
    let sizes: Vec<_> = (0..frame.planes.len())
        .map(|component| frame.plane_size(component).expect("plane"))
        .collect();
    for (component, plane) in frame.planes.iter_mut().enumerate() {
        let (plane_width, plane_height) = sizes[component];
        for y in 0..plane_height {
            for x in 0..plane_width {
                let base = (x * maximum / plane_width.max(1)) as u16;
                let tilt = (y * 37 / plane_height.max(1)) as u16;
                plane[(y * plane_width + x) as usize] =
                    (u32::from(base) + u32::from(tilt) + component as u32 * 17).min(maximum) as u16;
            }
        }
    }
    frame
}

fn psnr(original: &ApvFrame, decoded: &ApvFrame) -> f64 {
    let mut error = 0_f64;
    let mut count = 0_u64;
    for (a, b) in original.planes.iter().zip(&decoded.planes) {
        for (x, y) in a.iter().zip(b) {
            let difference = f64::from(*x) - f64::from(*y);
            error += difference * difference;
            count += 1;
        }
    }
    let mse = error / count as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (f64::from((1_u32 << original.bit_depth) - 1).powi(2) / mse).log10()
    }
}

#[test]
fn round_trip_422_10bit() {
    let frame = make_frame(ApvChromaFormat::Yuv422, 10, 64, 48);
    let encoder = ApvEncoder::new(ApvConfig {
        qp: 24,
        ..ApvConfig::default()
    })
    .expect("encoder");
    let access_unit = encoder.encode_access_unit(&frame).expect("encode");
    let decoded = ApvDecoder::new()
        .decode_access_unit(&access_unit)
        .expect("decode");
    assert_eq!(decoded.width, frame.width);
    assert_eq!(decoded.height, frame.height);
    assert_eq!(decoded.chroma_format, frame.chroma_format);
    assert_eq!(decoded.bit_depth, frame.bit_depth);
    assert!(decoded.is_valid());
    assert!(
        psnr(&frame, &decoded) > 30.0,
        "PSNR {}",
        psnr(&frame, &decoded)
    );
}

#[test]
fn round_trip_all_layouts_and_edges() {
    for (chroma, bit_depth, width, height) in [
        (ApvChromaFormat::Mono, 10, 30, 18),
        (ApvChromaFormat::Yuv422, 10, 30, 18),
        (ApvChromaFormat::Yuv422, 12, 34, 20),
        (ApvChromaFormat::Yuv444, 10, 30, 18),
        (ApvChromaFormat::Yuv4444, 12, 30, 18),
        (ApvChromaFormat::Yuv422, 10, 300, 140),
    ] {
        let frame = make_frame(chroma, bit_depth, width, height);
        let encoder = ApvEncoder::new(ApvConfig {
            qp: 18,
            ..ApvConfig::default()
        })
        .expect("encoder");
        let access_unit = encoder.encode_access_unit(&frame).expect("encode");
        let decoded = ApvDecoder::new()
            .decode_access_unit(&access_unit)
            .expect("decode");
        assert_eq!(decoded.planes.len(), frame.planes.len());
        assert!(decoded.is_valid());
        let quality = psnr(&frame, &decoded);
        assert!(
            quality > 25.0,
            "{chroma} {bit_depth} {width}x{height}: PSNR {quality}"
        );
    }
}

#[test]
fn raw_bitstream_round_trip() {
    let frame = make_frame(ApvChromaFormat::Yuv422, 10, 32, 32);
    let encoder = ApvEncoder::new(ApvConfig::default()).expect("encoder");
    let raw = encoder.encode_raw_bitstream(&frame).expect("encode");
    let frames = ApvDecoder::new()
        .decode_raw_bitstream(&raw)
        .expect("decode");
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].width, 32);
}

const REFERENCE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/apv/qp_F.apv");

fn fnv1a(values: &[u16]) -> u32 {
    let mut hash = 2_166_136_261_u32;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u32::from(byte);
            hash = hash.wrapping_mul(16_777_619);
        }
    }
    hash
}

#[test]
fn decodes_openapv_reference_vector() {
    let data = std::fs::read(REFERENCE).expect("reference stream");
    let frames = ApvDecoder::new()
        .decode_raw_bitstream(&data)
        .expect("decode reference");
    assert_eq!(frames.len(), 2);
    let expected = [
        ([0x93bb_49c5, 0xbd5b_f3c5, 0xbd5b_f3c5]),
        ([0x8bbb_49c5, 0x3b5b_f3c5, 0x3b5b_f3c5]),
    ];
    for (frame, hashes) in frames.iter().zip(expected) {
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert_eq!(frame.chroma_format, ApvChromaFormat::Yuv422);
        assert_eq!(frame.bit_depth, 12);
        assert_eq!(
            [
                fnv1a(&frame.planes[0]),
                fnv1a(&frame.planes[1]),
                fnv1a(&frame.planes[2])
            ],
            hashes
        );
    }
}

#[test]
fn colour_and_capture_time_metadata_round_trip() {
    use rustmpeg_core::ColorDescription;
    let mut frame = make_frame(ApvChromaFormat::Yuv422, 10, 64, 48);
    frame.color = Some(ColorDescription::BT2020_PQ);
    frame.capture_time_distance = 7;
    let encoder = ApvEncoder::new(ApvConfig::default()).expect("encoder");
    let access_unit = encoder.encode_access_unit(&frame).expect("encode");
    let decoded = ApvDecoder::new()
        .decode_access_unit(&access_unit)
        .expect("decode");
    assert_eq!(decoded.color, frame.color);
    assert_eq!(decoded.capture_time_distance, 7);
}

#[test]
fn flat_center_frame_round_trips_exactly() {
    let bit_depth = 10;
    let center = 1_u16 << (bit_depth - 1);
    let mut frame = ApvFrame::new(ApvChromaFormat::Yuv422, bit_depth, 64, 48).expect("frame");
    for plane in &mut frame.planes {
        plane.fill(center);
    }
    let encoder = ApvEncoder::new(ApvConfig {
        qp: 24,
        ..ApvConfig::default()
    })
    .expect("encoder");
    let access_unit = encoder.encode_access_unit(&frame).expect("encode");
    let decoded = ApvDecoder::new()
        .decode_access_unit(&access_unit)
        .expect("decode");
    assert_eq!(decoded.planes, frame.planes);
}

#[test]
fn truncated_and_corrupted_reference_streams_never_panic() {
    let data = std::fs::read(REFERENCE).expect("reference stream");
    let decoder = ApvDecoder::new();
    for length in (0..data.len()).step_by(997) {
        let _ = decoder.decode_raw_bitstream(&data[..length]);
    }
    let mut state = 0x1234_5678_u32;
    for _ in 0..30 {
        let mut copy = data.clone();
        for _ in 0..3 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let index = (state as usize) % copy.len();
            copy[index] ^= (state >> 7) as u8;
        }
        let _ = decoder.decode_raw_bitstream(&copy);
    }
}
