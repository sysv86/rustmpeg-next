// SPDX-License-Identifier: Apache-2.0
//! Synthetic native FLAC streams for code paths that common encoders do not
//! emit, such as variable block size streams and 12/20-bit sample depths.

use std::io::Cursor;

use rustmpeg_audio::{FlacDecoder, FlacError};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    bits: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            bits: 0,
        }
    }

    fn write_bit(&mut self, bit: bool) {
        self.current = (self.current << 1) | u8::from(bit);
        self.bits += 1;
        if self.bits == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.bits = 0;
        }
    }

    fn write_bits(&mut self, value: u64, count: u32) {
        for shift in (0..count).rev() {
            self.write_bit((value >> shift) & 1 != 0);
        }
    }

    fn write_signed(&mut self, value: i64, count: u32) {
        self.write_bits(value as u64, count);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits != 0 {
            self.current <<= 8 - self.bits;
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
    }
    crc
}

fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &byte in data {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

#[derive(Clone, Copy)]
enum Prediction {
    Verbatim,
    Fixed(u32),
    Lpc(u32),
}

struct FrameSpec {
    variable: bool,
    block_size: u8,
    coded_number: u8,
    bits_per_sample: u8,
    bps_code: u8,
    prediction: Prediction,
    samples: Vec<Vec<i64>>,
}

/// Writes a partitioned Rice residual whose samples are all zero.
fn write_zero_residuals(writer: &mut BitWriter, count: u32) {
    writer.write_bits(0, 2); // 4-bit Rice parameters
    writer.write_bits(0, 4); // partition order 0
    writer.write_bits(0, 4); // Rice parameter 0
    for _ in 0..count {
        writer.write_bit(true); // quotient 0
    }
}

/// Overwrites the 36-bit STREAMINFO total sample count; zero means unknown.
fn set_total_samples(stream: &mut [u8], total: u64) {
    let offset = 4 + 4 + 10;
    let packed = u64::from_be_bytes(stream[offset..offset + 8].try_into().unwrap());
    let packed = (packed & !0x0000_000f_ffff_ffff) | (total & 0x0000_000f_ffff_ffff);
    stream[offset..offset + 8].copy_from_slice(&packed.to_be_bytes());
}

fn build_stream(
    min_block_size: u16,
    max_block_size: u16,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    frames: &[FrameSpec],
) -> Vec<u8> {
    let total: u64 = frames.iter().map(|frame| u64::from(frame.block_size)).sum();

    let mut payload = [0u8; 34];
    payload[0..2].copy_from_slice(&min_block_size.to_be_bytes());
    payload[2..4].copy_from_slice(&max_block_size.to_be_bytes());
    let packed = (u64::from(sample_rate) << 44)
        | (u64::from(channels - 1) << 41)
        | (u64::from(bits_per_sample - 1) << 36)
        | total;
    payload[10..18].copy_from_slice(&packed.to_be_bytes());

    let mut stream = b"fLaC".to_vec();
    stream.extend_from_slice(&[0x80, 0x00, 0x00, 0x22]);
    stream.extend_from_slice(&payload);

    for frame in frames {
        stream.extend_from_slice(&build_frame(frame));
    }
    stream
}

fn build_frame(spec: &FrameSpec) -> Vec<u8> {
    let channels = spec.samples.len();
    let mut writer = BitWriter::new();
    writer.write_bits(0x7ffc, 15); // frame sync
    writer.write_bit(spec.variable);
    writer.write_bits(6, 4); // uncommon 8-bit block size
    writer.write_bits(0, 4); // sample rate from STREAMINFO
    writer.write_bits((channels - 1) as u64, 4);
    writer.write_bits(u64::from(spec.bps_code), 3);
    writer.write_bit(false); // reserved
    let mut header = writer.finish();
    assert_eq!(header.len(), 4);
    header.push(spec.coded_number);
    header.push(spec.block_size - 1);
    let header_crc = crc8(&header);
    header.push(header_crc);

    let mut subframes = BitWriter::new();
    for plane in &spec.samples {
        assert_eq!(plane.len(), usize::from(spec.block_size));
        let bits = u32::from(spec.bits_per_sample);
        subframes.write_bit(false);
        match spec.prediction {
            Prediction::Verbatim => {
                subframes.write_bits(1, 6); // verbatim
                subframes.write_bit(false); // no wasted bits
                for &sample in plane {
                    subframes.write_signed(sample, bits);
                }
            }
            Prediction::Fixed(order) => {
                subframes.write_bits(8 + u64::from(order), 6);
                subframes.write_bit(false);
                for &sample in &plane[..order as usize] {
                    subframes.write_signed(sample, bits);
                }
                write_zero_residuals(&mut subframes, spec.block_size as u32 - order);
            }
            Prediction::Lpc(order) => {
                subframes.write_bits(31 + u64::from(order), 6);
                subframes.write_bit(false);
                for &sample in &plane[..order as usize] {
                    subframes.write_signed(sample, bits);
                }
                subframes.write_bits(1, 4); // coefficient precision 2 bits
                subframes.write_signed(0, 5); // right shift 0
                for index in 0..order {
                    // Predict from the directly preceding sample only.
                    let coefficient = if index == 0 { 1 } else { 0 };
                    subframes.write_signed(coefficient, 2);
                }
                write_zero_residuals(&mut subframes, spec.block_size as u32 - order);
            }
        }
    }
    let mut frame = header;
    frame.extend(subframes.finish());
    let frame_crc = crc16(&frame);
    frame.extend_from_slice(&frame_crc.to_be_bytes());
    frame
}

fn mono_frame(
    variable: bool,
    block_size: u8,
    coded_number: u8,
    bits_per_sample: u8,
    bps_code: u8,
    samples: Vec<i64>,
) -> FrameSpec {
    FrameSpec {
        variable,
        block_size,
        coded_number,
        bits_per_sample,
        bps_code,
        prediction: Prediction::Verbatim,
        samples: vec![samples],
    }
}

fn decode(bytes: &[u8]) -> Result<Vec<rustmpeg_audio::PcmFrame>, FlacError> {
    let mut decoder = FlacDecoder::new(Cursor::new(bytes.to_vec()))?;
    let mut frames = Vec::new();
    while let Some(frame) = decoder.next_frame()? {
        frames.push(frame);
    }
    Ok(frames)
}

#[test]
fn decodes_variable_block_size_streams() {
    let frames = [
        mono_frame(true, 16, 0, 16, 0, (0..16).map(i64::from).collect()),
        mono_frame(true, 7, 16, 16, 0, (100..107).map(i64::from).collect()),
    ];
    let stream = build_stream(16, 16, 44_100, 1, 16, &frames);
    let decoded = decode(&stream).unwrap();
    assert_eq!(decoded.len(), 2);

    assert_eq!(decoded[0].block_size(), 16);
    assert_eq!(decoded[0].frame_number, 0);
    assert_eq!(decoded[0].sample_number, 0);
    assert_eq!(decoded[0].planes, vec![(0..16).collect::<Vec<i32>>()]);

    assert_eq!(decoded[1].block_size(), 7);
    assert_eq!(decoded[1].frame_number, 1);
    assert_eq!(decoded[1].sample_number, 16);
    assert_eq!(decoded[1].planes, vec![(100..107).collect::<Vec<i32>>()]);
}

#[test]
fn decodes_twelve_and_twenty_bit_samples() {
    for (bits, bps_code, samples) in [
        (12u8, 2u8, vec![-2048i64, 2047, 0, 1]),
        (20u8, 5u8, vec![-524_288, 524_287, -1, 0]),
    ] {
        let frame = mono_frame(false, 4, 0, bits, bps_code, samples.clone());
        let stream = build_stream(16, 16, 44_100, 1, bits, &[frame]);
        let decoded = decode(&stream).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].bits_per_sample, bits);
        assert_eq!(decoded[0].sample_rate, 44_100);
        let expected: Vec<i32> = samples.iter().map(|&sample| sample as i32).collect();
        assert_eq!(decoded[0].planes, vec![expected]);
    }
}

#[test]
fn decodes_an_id3v2_prefixed_stream() {
    let frame = mono_frame(false, 16, 0, 16, 0, vec![7; 16]);
    let stream = build_stream(16, 16, 44_100, 1, 16, &[frame]);
    let mut prefixed = b"ID3\x04\x00\x00\x00\x00\x00\x04".to_vec();
    prefixed.extend_from_slice(&[0u8; 4]);
    prefixed.extend_from_slice(&stream);
    let decoded = decode(&prefixed).unwrap();
    assert_eq!(decoded[0].planes, vec![vec![7; 16]]);
}

#[test]
fn rejects_a_bad_frame_crc16() {
    let frame = mono_frame(false, 16, 0, 16, 0, (0..16).map(i64::from).collect());
    let mut stream = build_stream(16, 16, 44_100, 1, 16, &[frame]);
    let frame_start = 4 + 4 + 34;
    stream[frame_start + 12] ^= 0x40;
    let error = decode(&stream).err();
    assert!(matches!(error, Some(FlacError::FrameCrcMismatch { .. })));
}

#[test]
fn rejects_a_bad_header_crc8() {
    let frame = mono_frame(false, 16, 0, 16, 0, (0..16).map(i64::from).collect());
    let mut stream = build_stream(16, 16, 44_100, 1, 16, &[frame]);
    let frame_start = 4 + 4 + 34;
    stream[frame_start + 6] ^= 1;
    let error = decode(&stream).err();
    assert!(matches!(error, Some(FlacError::HeaderCrcMismatch { .. })));
}

#[test]
fn rejects_a_non_final_block_below_the_minimum() {
    let frames = [
        mono_frame(true, 7, 0, 16, 0, vec![1; 7]),
        mono_frame(true, 7, 7, 16, 0, vec![2; 7]),
    ];
    let stream = build_stream(16, 16, 44_100, 1, 16, &frames);
    let error = decode(&stream).err();
    assert!(matches!(error, Some(FlacError::InvalidFrame(_))));
}

#[test]
fn rejects_frame_number_discontinuities() {
    let frames = [
        mono_frame(false, 16, 0, 16, 0, vec![1; 16]),
        mono_frame(false, 16, 2, 16, 0, vec![2; 16]),
    ];
    let stream = build_stream(16, 16, 44_100, 1, 16, &frames);
    let error = decode(&stream).err();
    assert!(matches!(error, Some(FlacError::InvalidFrame(_))));
}

#[test]
fn rejects_sample_number_discontinuities() {
    let frames = [
        mono_frame(true, 16, 0, 16, 0, vec![1; 16]),
        mono_frame(true, 7, 17, 16, 0, vec![2; 7]),
    ];
    let stream = build_stream(16, 16, 44_100, 1, 16, &frames);
    let error = decode(&stream).err();
    assert!(matches!(error, Some(FlacError::InvalidFrame(_))));
}

#[test]
fn rejects_block_sizes_above_streaminfo() {
    let frame = mono_frame(false, 17, 0, 16, 0, vec![1; 17]);
    let stream = build_stream(16, 16, 44_100, 1, 16, &[frame]);
    let error = decode(&stream).err();
    assert!(matches!(error, Some(FlacError::InvalidFrame(_))));
}

#[test]
fn decodes_variable_block_fixed_and_lpc_predictors() {
    // A fixed predictor of order 2 continues 10, 20 with a step of 10.
    let fixed = FrameSpec {
        variable: true,
        block_size: 16,
        coded_number: 0,
        bits_per_sample: 16,
        bps_code: 0,
        prediction: Prediction::Fixed(2),
        samples: vec![(0..16).map(|index| 10 + 10 * index).collect()],
    };
    // An LPC predictor of order 2 that weights the previous sample by one.
    let mut lpc_samples = vec![7i64, 9];
    lpc_samples.resize(8, 9);
    let lpc = FrameSpec {
        variable: true,
        block_size: 8,
        coded_number: 16,
        bits_per_sample: 16,
        bps_code: 0,
        prediction: Prediction::Lpc(2),
        samples: vec![lpc_samples.clone()],
    };

    let stream = build_stream(16, 16, 44_100, 1, 16, &[fixed, lpc]);
    let decoded = decode(&stream).unwrap();
    assert_eq!(decoded.len(), 2);

    assert_eq!(
        decoded[0].planes[0],
        (0..16).map(|index| 10 + 10 * index).collect::<Vec<i32>>()
    );
    assert_eq!(decoded[1].sample_number, 16);
    assert_eq!(decoded[1].block_size(), 8);
    assert_eq!(
        decoded[1].planes[0],
        lpc_samples
            .iter()
            .map(|&sample| sample as i32)
            .collect::<Vec<i32>>()
    );
}

#[test]
fn decodes_four_bit_samples_from_streaminfo() {
    let samples = vec![-8i64, 7, 0, -1, 1, -8, 7, 0, -1, 1, -8, 7, 0, -1, 1, 0];
    let frame = mono_frame(false, 16, 0, 4, 0, samples.clone());
    let stream = build_stream(16, 16, 44_100, 1, 4, &[frame]);
    let decoded = decode(&stream).unwrap();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].bits_per_sample, 4);
    assert_eq!(
        decoded[0].planes[0],
        samples
            .iter()
            .map(|&sample| sample as i32)
            .collect::<Vec<i32>>()
    );
}

#[test]
fn rejects_truncation_that_ends_on_a_frame_boundary() {
    let frames = [
        mono_frame(false, 16, 0, 16, 0, vec![1; 16]),
        mono_frame(false, 16, 1, 16, 0, vec![2; 16]),
    ];
    let mut stream = build_stream(16, 16, 44_100, 1, 16, &frames);
    // Announce one sample more than the two complete frames contain.
    set_total_samples(&mut stream, 33);

    let mut decoder = FlacDecoder::new(Cursor::new(stream)).unwrap();
    assert_eq!(decoder.next_frame().unwrap().unwrap().sample_number, 0);
    assert_eq!(decoder.next_frame().unwrap().unwrap().sample_number, 16);
    let error = decoder.next_frame().err();
    assert!(matches!(error, Some(FlacError::Io(_))));
}

#[test]
fn accepts_a_clean_end_of_stream_when_the_total_is_unknown() {
    let frames = [mono_frame(false, 16, 0, 16, 0, vec![1; 16])];
    let mut stream = build_stream(16, 16, 44_100, 1, 16, &frames);
    set_total_samples(&mut stream, 0);

    let mut decoder = FlacDecoder::new(Cursor::new(stream)).unwrap();
    assert_eq!(decoder.stream_info().total_samples, None);
    assert!(decoder.next_frame().unwrap().is_some());
    assert!(decoder.next_frame().unwrap().is_none());
}

#[test]
fn errors_are_latched_for_the_lifetime_of_the_decoder() {
    let frame = mono_frame(false, 16, 0, 16, 0, vec![1; 16]);
    let mut stream = build_stream(16, 16, 44_100, 1, 16, &[frame]);
    let frame_start = 4 + 4 + 34;
    stream[frame_start + 12] ^= 0x40; // corrupt the frame CRC-16

    let mut decoder = FlacDecoder::new(Cursor::new(stream)).unwrap();
    assert!(matches!(
        decoder.next_frame(),
        Err(FlacError::FrameCrcMismatch { .. })
    ));
    // After an error the decoder must be recreated; reuse reports an error
    // instead of decoding garbage.
    assert!(decoder.next_frame().is_err());
}

#[test]
fn fnv_hash_matches_a_reference_vector() {
    // Guard the hash helper used by the fixture tests against accidental
    // changes.  "a" hashes to 0xaf63dc4c8601ec8c in FNV-1a 64.
    let mut hash = FNV_OFFSET;
    for byte in b"a" {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    assert_eq!(hash, 0xaf63_dc4c_8601_ec8c);
}
