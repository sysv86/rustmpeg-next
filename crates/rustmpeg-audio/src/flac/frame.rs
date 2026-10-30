// SPDX-License-Identifier: Apache-2.0
//! FLAC frame header parsing and frame orchestration (RFC 9639, Section 9).

use std::io::Read;

use super::bitreader::BitReader;
use super::subframe;
use super::{
    FlacDecoder, FlacError, MAX_BLOCK_SIZE, MAX_FRAME_BYTES, MAX_FRAME_SAMPLE_RATE,
    MAX_STREAMINFO_SAMPLE_RATE, StreamInfo,
};
use crate::PcmFrame;

/// The 15-bit frame sync code (`0b111111111111100`).
const FRAME_SYNC: u64 = 0x7ffc;

/// Maximum frame number in a fixed block size stream (31 bits).
const MAX_FRAME_NUMBER: u64 = 0x7fff_ffff;

/// Interchannel decorrelation used by a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelAssignment {
    /// Each channel is coded independently.
    Independent,
    /// Left channel plus `left - right`.
    LeftSide,
    /// `left - right` plus right channel.
    RightSide,
    /// Mid channel plus side channel.
    MidSide,
}

/// A fully validated frame header.
#[derive(Debug, Clone, Copy)]
struct FrameHeader {
    variable_block_size: bool,
    block_size: u32,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    assignment: ChannelAssignment,
    coded_number: u64,
}

/// Decodes the next frame, or returns [`None`] at a clean end of stream.
impl<R: Read> FlacDecoder<R> {
    /// Decodes and returns the next audio frame.
    ///
    /// Returns `Ok(None)` only when the input ends exactly on a frame boundary
    /// and every sample announced by STREAMINFO has been decoded, or when
    /// STREAMINFO does not announce a sample count and the input ends on a
    /// frame boundary.  Truncation inside a frame or between frames, checksum
    /// failures, and specification violations are reported as errors.
    ///
    /// Values in the returned [`PcmFrame`] describe the frame that was just
    /// decoded; a frame header may change the sample rate, channel count, or
    /// bit depth, so these fields take precedence over
    /// [`Self::stream_info`].  Frames are returned in decode order with
    /// `sample_number` counting all samples decoded so far.
    ///
    /// # Errors and decoder state
    ///
    /// Any error is latched: the reader may already have consumed part of the
    /// frame, so a decoder that returned an error cannot be reused.  Every
    /// later call returns an error as well.  Create a new decoder to retry or
    /// to resynchronise.
    pub fn next_frame(&mut self) -> Result<Option<PcmFrame>, FlacError> {
        if self.failed {
            return Err(FlacError::InvalidFrame(
                "decoder was used after a previous error",
            ));
        }
        if let Some(frame) = self.pending_frame.take() {
            return Ok(Some(frame));
        }
        match self.decode_frame() {
            Ok(frame) => Ok(frame),
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    /// Implements one frame decode without the latching error wrapper.
    pub(super) fn decode_frame(&mut self) -> Result<Option<PcmFrame>, FlacError> {
        let frame_start = self.reader.consumed();
        self.reader.begin_frame(MAX_FRAME_BYTES);
        if !self.reader.ensure_data()? {
            if let Some(total) = self.info.total_samples
                && self.next_sample < total
            {
                return Err(FlacError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "FLAC stream ended before the announced sample count",
                )));
            }
            self.check_md5()?;
            return Ok(None);
        }

        let header = parse_header(&mut self.reader, &self.info)?;

        if let Some(previous) = self.previous_block_size
            && previous < u32::from(self.info.min_block_size)
        {
            return Err(FlacError::InvalidFrame(
                "only the last frame may be smaller than the minimum block size",
            ));
        }
        match self.blocking_variable {
            Some(variable) if variable != header.variable_block_size => {
                return Err(FlacError::InvalidFrame(
                    "blocking strategy changed inside the stream",
                ));
            }
            None => self.blocking_variable = Some(header.variable_block_size),
            Some(_) => {}
        }
        if header.variable_block_size {
            if self.strict_frame_numbers && header.coded_number != self.next_sample {
                return Err(FlacError::InvalidFrame(
                    "sample number does not match preceding sample count",
                ));
            }
        } else if self.strict_frame_numbers && header.coded_number != self.frame_index {
            return Err(FlacError::InvalidFrame(
                "frame number does not match preceding frame count",
            ));
        }

        let mut subframes = Vec::with_capacity(usize::from(header.channels));
        for channel in 0..header.channels {
            let side = match header.assignment {
                ChannelAssignment::LeftSide => channel == 1,
                ChannelAssignment::RightSide => channel == 0,
                ChannelAssignment::MidSide => channel == 1,
                ChannelAssignment::Independent => false,
            };
            subframes.push(subframe::decode(
                &mut self.reader,
                header.block_size,
                header.bits_per_sample,
                side,
            )?);
        }
        let planes = assemble_planes(header.assignment, subframes, header.bits_per_sample)?;

        self.reader.align_zero()?;
        let computed = self.reader.crc16();
        let stored = self.reader.read_crc16()?;
        if stored != computed {
            return Err(FlacError::FrameCrcMismatch { stored, computed });
        }

        let frame_size = self.reader.consumed() - frame_start;
        if self.info.max_frame_size != 0 && frame_size > u64::from(self.info.max_frame_size) {
            return Err(FlacError::InvalidFrame(
                "frame is larger than the STREAMINFO maximum frame size",
            ));
        }

        let block_size = u64::from(header.block_size);
        let next_sample = self
            .next_sample
            .checked_add(block_size)
            .ok_or(FlacError::LimitExceeded("sample count overflow"))?;
        if let Some(total) = self.info.total_samples
            && next_sample > total
        {
            return Err(FlacError::InvalidFrame(
                "decoded samples exceed the STREAMINFO total sample count",
            ));
        }
        let sample_number = self.next_sample;
        let frame_number = if header.variable_block_size {
            self.frame_index
        } else {
            header.coded_number
        };

        self.next_sample = next_sample;
        self.frame_index = self
            .frame_index
            .checked_add(1)
            .ok_or(FlacError::LimitExceeded("frame count overflow"))?;
        self.previous_block_size = Some(header.block_size);

        self.md5.update(&super::md5::interleaved_bytes(
            &planes,
            header.bits_per_sample,
        ));
        Ok(Some(PcmFrame {
            sample_rate: header.sample_rate,
            channels: header.channels,
            bits_per_sample: header.bits_per_sample,
            frame_number,
            sample_number,
            planes,
        }))
    }
}

/// Parses and validates a frame header, including both checksum fields.
fn parse_header<R: Read>(
    reader: &mut BitReader<R>,
    info: &StreamInfo,
) -> Result<FrameHeader, FlacError> {
    if reader.read_bits(15)? != FRAME_SYNC {
        return Err(FlacError::InvalidFrame("invalid frame sync code"));
    }
    let variable_block_size = reader.read_bit()?;
    let block_size_code = reader.read_bits(4)? as u32;
    let sample_rate_code = reader.read_bits(4)? as u32;
    let channel_code = reader.read_bits(4)? as u32;
    let bit_depth_code = reader.read_bits(3)? as u32;
    if reader.read_bit()? {
        return Err(FlacError::InvalidFrame(
            "reserved frame header bit is not zero",
        ));
    }

    let (channels, assignment) = match channel_code {
        0..=7 => (channel_code as u8 + 1, ChannelAssignment::Independent),
        8 => (2, ChannelAssignment::LeftSide),
        9 => (2, ChannelAssignment::RightSide),
        10 => (2, ChannelAssignment::MidSide),
        _ => {
            return Err(FlacError::UnsupportedFeature(
                "reserved frame header channel assignment",
            ));
        }
    };

    let bits_per_sample = match bit_depth_code {
        0 => info.bits_per_sample,
        1 => 8,
        2 => 12,
        3 => {
            return Err(FlacError::UnsupportedFeature(
                "reserved frame header bit depth",
            ));
        }
        4 => 16,
        5 => 20,
        6 => 24,
        7 => 32,
        _ => return Err(FlacError::InvalidFrame("bit depth code out of range")),
    };
    if !(4..=32).contains(&bits_per_sample) {
        return Err(FlacError::InvalidFrame(
            "frame bit depth is outside 4 to 32 bits",
        ));
    }
    if block_size_code == 0 {
        return Err(FlacError::UnsupportedFeature(
            "reserved frame header block size",
        ));
    }
    if sample_rate_code == 15 {
        return Err(FlacError::InvalidFrame(
            "forbidden frame header sample rate code",
        ));
    }

    let coded_number = read_coded_number(reader, variable_block_size)?;

    let block_size = match block_size_code {
        1 => 192,
        2..=5 => 144u32 << block_size_code,
        6 => u32::from(reader.read_u8()?) + 1,
        7 => {
            let stored = reader.read_bits(16)? as u32;
            if stored == u32::from(u16::MAX) {
                return Err(FlacError::InvalidFrame(
                    "block size 65536 cannot be represented in STREAMINFO",
                ));
            }
            stored + 1
        }
        8..=15 => 1u32 << block_size_code,
        _ => return Err(FlacError::InvalidFrame("block size code out of range")),
    };
    if block_size == 0 || block_size > MAX_BLOCK_SIZE {
        return Err(FlacError::InvalidFrame("invalid block size"));
    }
    if block_size > u32::from(info.max_block_size) {
        return Err(FlacError::InvalidFrame(
            "block size exceeds the STREAMINFO maximum block size",
        ));
    }

    let sample_rate = match sample_rate_code {
        0 => info.sample_rate,
        1 => 88_200,
        2 => 176_400,
        3 => 192_000,
        4 => 8_000,
        5 => 16_000,
        6 => 22_050,
        7 => 24_000,
        8 => 32_000,
        9 => 44_100,
        10 => 48_000,
        11 => 96_000,
        12 => u32::from(reader.read_u8()?) * 1_000,
        13 => reader.read_bits(16)? as u32,
        14 => (reader.read_bits(16)? as u32) * 10,
        _ => return Err(FlacError::InvalidFrame("sample rate code out of range")),
    };
    // Codes 1 to 14 all fit in the 655350 Hz limit imposed by the uncommon
    // forms.  Code 0 delegates to the wider 20-bit STREAMINFO field.
    let sample_rate_limit = if sample_rate_code == 0 {
        MAX_STREAMINFO_SAMPLE_RATE
    } else {
        MAX_FRAME_SAMPLE_RATE
    };
    if sample_rate == 0 || sample_rate > sample_rate_limit {
        return Err(FlacError::InvalidFrame(
            "frame sample rate is zero or out of range",
        ));
    }

    let computed = reader.crc8();
    let stored = reader.read_crc8_byte()?;
    if stored != computed {
        return Err(FlacError::HeaderCrcMismatch { stored, computed });
    }

    Ok(FrameHeader {
        variable_block_size,
        block_size,
        sample_rate,
        channels,
        bits_per_sample,
        assignment,
        coded_number,
    })
}

/// Decodes the UTF-8-like coded frame or sample number (RFC 9639, Section
/// 9.1.5).
fn read_coded_number<R: Read>(
    reader: &mut BitReader<R>,
    variable_block_size: bool,
) -> Result<u64, FlacError> {
    let first = reader.read_u8()?;
    if first < 0x80 {
        return Ok(u64::from(first));
    }

    let (length, payload_mask) = match first {
        0xc0..=0xdf => (2usize, 0x1f),
        0xe0..=0xef => (3, 0x0f),
        0xf0..=0xf7 => (4, 0x07),
        0xf8..=0xfb => (5, 0x03),
        0xfc..=0xfd => (6, 0x01),
        0xfe => (7, 0x00),
        _ => {
            return Err(FlacError::InvalidFrame("invalid coded number leading byte"));
        }
    };

    let mut value = u64::from(first & payload_mask);
    for _ in 1..length {
        let byte = reader.read_u8()?;
        if byte & 0xc0 != 0x80 {
            return Err(FlacError::InvalidFrame(
                "invalid coded number continuation byte",
            ));
        }
        value = (value << 6) | u64::from(byte & 0x3f);
    }

    let minimum = match length {
        2 => 0x80,
        3 => 0x800,
        4 => 0x1_0000,
        5 => 0x20_0000,
        6 => 0x0400_0000,
        7 => 0x8000_0000,
        _ => return Err(FlacError::InvalidFrame("invalid coded number length")),
    };
    if value < minimum {
        return Err(FlacError::InvalidFrame("overlong coded number encoding"));
    }
    if !variable_block_size && (length > 6 || value > MAX_FRAME_NUMBER) {
        return Err(FlacError::InvalidFrame(
            "frame number exceeds the 31-bit limit",
        ));
    }
    Ok(value)
}

/// Undoes interchannel decorrelation and converts samples to native `i32`.
fn assemble_planes(
    assignment: ChannelAssignment,
    subframes: Vec<Vec<i64>>,
    bits_per_sample: u8,
) -> Result<Vec<Vec<i32>>, FlacError> {
    if assignment == ChannelAssignment::Independent {
        let mut planes = Vec::with_capacity(subframes.len());
        for subframe in subframes {
            let mut plane = Vec::with_capacity(subframe.len());
            for sample in subframe {
                plane.push(to_output_sample(sample, bits_per_sample)?);
            }
            planes.push(plane);
        }
        return Ok(planes);
    }

    if subframes.len() != 2 {
        return Err(FlacError::InvalidFrame(
            "stereo decorrelation requires exactly two subframes",
        ));
    }
    let mut subframes = subframes.into_iter();
    let first = subframes
        .next()
        .ok_or(FlacError::InvalidFrame("missing stereo subframe"))?;
    let second = subframes
        .next()
        .ok_or(FlacError::InvalidFrame("missing stereo subframe"))?;
    if first.len() != second.len() {
        return Err(FlacError::InvalidFrame(
            "stereo subframes have different lengths",
        ));
    }

    let mut left = Vec::with_capacity(first.len());
    let mut right = Vec::with_capacity(first.len());
    for (a, b) in first.into_iter().zip(second) {
        let (left_sample, right_sample) = match assignment {
            ChannelAssignment::LeftSide => (Some(a), a.checked_sub(b)),
            ChannelAssignment::RightSide => (b.checked_add(a), Some(b)),
            ChannelAssignment::MidSide => {
                let half = b >> 1;
                (
                    a.checked_add(b & 1)
                        .and_then(|value| value.checked_add(half)),
                    a.checked_sub(half),
                )
            }
            ChannelAssignment::Independent => {
                return Err(FlacError::InvalidFrame("inconsistent channel assignment"));
            }
        };
        let left_sample =
            left_sample.ok_or(FlacError::InvalidFrame("channel decorrelation overflow"))?;
        let right_sample =
            right_sample.ok_or(FlacError::InvalidFrame("channel decorrelation overflow"))?;
        left.push(to_output_sample(left_sample, bits_per_sample)?);
        right.push(to_output_sample(right_sample, bits_per_sample)?);
    }
    Ok(vec![left, right])
}

/// Converts one decorrelated sample to its native integer representation.
fn to_output_sample(sample: i64, bits_per_sample: u8) -> Result<i32, FlacError> {
    let limit = 1i64 << (bits_per_sample - 1);
    if sample < -limit || sample >= limit {
        return Err(FlacError::InvalidFrame(
            "decoded sample is outside the declared bit depth",
        ));
    }
    i32::try_from(sample)
        .map_err(|_| FlacError::InvalidFrame("decoded sample does not fit in 32 bits"))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{ChannelAssignment, assemble_planes, parse_header, read_coded_number};
    use crate::flac::bitreader::BitReader;
    use crate::flac::{FlacError, StreamInfo};

    fn info() -> StreamInfo {
        StreamInfo {
            min_block_size: 16,
            max_block_size: 4096,
            min_frame_size: 0,
            max_frame_size: 0,
            sample_rate: 44_100,
            channels: 2,
            bits_per_sample: 16,
            total_samples: None,
            md5: [0; 16],
        }
    }

    fn decode_number(bytes: &[u8], variable: bool) -> Result<u64, FlacError> {
        let mut reader = BitReader::new(Cursor::new(bytes.to_vec()));
        read_coded_number(&mut reader, variable)
    }

    fn parse(bytes: &[u8]) -> Result<super::FrameHeader, FlacError> {
        let mut reader = BitReader::new(Cursor::new(bytes.to_vec()));
        parse_header(&mut reader, &info())
    }

    /// Builds a byte-aligned header for fixed block strategy, frame number 0.
    fn build_header(
        block_size_code: u8,
        sample_rate_code: u8,
        channel_code: u8,
        bit_depth_code: u8,
        reserved_bit: bool,
        extras: &[u8],
    ) -> Vec<u8> {
        let mut header = vec![
            0xff,
            0xf8,
            (block_size_code << 4) | sample_rate_code,
            (channel_code << 4) | (bit_depth_code << 1) | u8::from(reserved_bit),
            0x00,
        ];
        header.extend_from_slice(extras);
        let crc = crate::flac::crc::crc8(&header);
        header.push(crc);
        header
    }

    #[test]
    fn parses_uncommon_sample_rate_forms() {
        let header = build_header(1, 12, 0, 4, false, &[48]);
        assert_eq!(parse(&header).unwrap().sample_rate, 48_000);

        let header = build_header(1, 13, 0, 4, false, &[0x30, 0x39]);
        assert_eq!(parse(&header).unwrap().sample_rate, 12_345);

        let header = build_header(1, 14, 0, 4, false, &[0x04, 0xd2]);
        assert_eq!(parse(&header).unwrap().sample_rate, 12_340);
    }

    #[test]
    fn accepts_a_streaminfo_sample_rate_above_the_frame_form_limit() {
        // Frame header sample rate code 0 delegates to STREAMINFO, whose field
        // is wider than the 655350 Hz limit of codes 12 to 14.
        let mut high_rate = info();
        high_rate.sample_rate = 1_048_575;
        let header = build_header(1, 0, 0, 4, false, &[]);
        let mut reader = BitReader::new(Cursor::new(header));
        let parsed = parse_header(&mut reader, &high_rate).unwrap();
        assert_eq!(parsed.sample_rate, 1_048_575);
    }

    #[test]
    fn rejects_reserved_and_forbidden_header_values() {
        // Reserved block size code.
        let error = parse(&build_header(0, 9, 0, 4, false, &[])).err();
        assert!(matches!(error, Some(FlacError::UnsupportedFeature(_))));

        // Forbidden sample rate code.
        let error = parse(&build_header(1, 15, 0, 4, false, &[])).err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));

        // Reserved bit depth code.
        let error = parse(&build_header(1, 9, 0, 3, false, &[])).err();
        assert!(matches!(error, Some(FlacError::UnsupportedFeature(_))));

        // Reserved channel assignment.
        let error = parse(&build_header(1, 9, 11, 4, false, &[])).err();
        assert!(matches!(error, Some(FlacError::UnsupportedFeature(_))));

        // Reserved header bit.
        let error = parse(&build_header(1, 9, 0, 4, true, &[])).err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));

        // Block size 65536 is forbidden.
        let error = parse(&build_header(7, 9, 0, 4, false, &[0xff, 0xff])).err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));

        // A zero sample rate is not audio.
        let error = parse(&build_header(1, 13, 0, 4, false, &[0, 0])).err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));
    }

    #[test]
    fn parses_rfc9639_example_1_header() {
        let header = parse(&[0xff, 0xf8, 0x69, 0x18, 0x00, 0x00, 0xbf]).unwrap();
        assert!(!header.variable_block_size);
        assert_eq!(header.block_size, 1);
        assert_eq!(header.sample_rate, 44_100);
        assert_eq!(header.channels, 2);
        assert_eq!(header.bits_per_sample, 16);
        assert_eq!(header.assignment, ChannelAssignment::Independent);
        assert_eq!(header.coded_number, 0);
    }

    #[test]
    fn parses_rfc9639_example_2_header() {
        let header = parse(&[0xff, 0xf8, 0x69, 0x98, 0x00, 0x0f, 0x99]).unwrap();
        assert_eq!(header.block_size, 16);
        assert_eq!(header.assignment, ChannelAssignment::RightSide);
    }

    #[test]
    fn parses_rfc9639_example_3_header() {
        let header = parse(&[0xff, 0xf8, 0x68, 0x02, 0x00, 0x17, 0xe9]).unwrap();
        assert_eq!(header.block_size, 24);
        assert_eq!(header.sample_rate, 32_000);
        assert_eq!(header.channels, 1);
        assert_eq!(header.bits_per_sample, 8);
    }

    #[test]
    fn rejects_a_corrupted_header_crc8() {
        let error = parse(&[0xff, 0xf8, 0x69, 0x18, 0x00, 0x00, 0xbe]).err();
        assert!(matches!(error, Some(FlacError::HeaderCrcMismatch { .. })));
    }

    #[test]
    fn rejects_an_invalid_sync_code() {
        let error = parse(&[0xff, 0xf0, 0x69, 0x18, 0x00, 0x00, 0x00]).err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));
    }

    #[test]
    fn decodes_all_utf8_lengths() {
        assert_eq!(decode_number(&[0x00], true).unwrap(), 0);
        assert_eq!(decode_number(&[0x7f], true).unwrap(), 0x7f);
        assert_eq!(decode_number(&[0xc2, 0x80], true).unwrap(), 0x80);
        assert_eq!(decode_number(&[0xe0, 0xa0, 0x80], true).unwrap(), 0x800);
        assert_eq!(
            decode_number(&[0xf0, 0x90, 0x80, 0x80], true).unwrap(),
            0x1_0000
        );
        assert_eq!(
            decode_number(&[0xf8, 0x88, 0x80, 0x80, 0x80], true).unwrap(),
            0x20_0000
        );
        assert_eq!(
            decode_number(&[0xfc, 0x84, 0x80, 0x80, 0x80, 0x80], true).unwrap(),
            0x0400_0000
        );
        assert_eq!(
            decode_number(&[0xfe, 0x82, 0x80, 0x80, 0x80, 0x80, 0x80], true).unwrap(),
            0x8000_0000
        );
        assert_eq!(
            decode_number(&[0xfd, 0xbf, 0xbf, 0xbf, 0xbf, 0xbf], true).unwrap(),
            0x7fff_ffff
        );
    }

    #[test]
    fn rejects_overlong_and_invalid_coded_numbers() {
        for bytes in [
            &[0xc0, 0x80][..],
            &[0xc1, 0xbf][..],
            &[0xe0, 0x9f, 0xbf][..],
            &[0xfe, 0x81, 0x80, 0x80, 0x80, 0x80, 0x80][..],
            &[0xc2, 0x00][..],
            &[0x80][..],
            &[0xff][..],
        ] {
            let error = decode_number(bytes, true).err();
            assert!(
                matches!(error, Some(FlacError::InvalidFrame(_))),
                "{bytes:02x?} should be rejected"
            );
        }
    }

    #[test]
    fn rejects_seven_byte_frame_numbers() {
        let bytes = [0xfe, 0x82, 0x80, 0x80, 0x80, 0x80, 0x80];
        let error = decode_number(&bytes, false).err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));
    }

    #[test]
    fn undoes_left_side_decorrelation() {
        let planes = assemble_planes(
            ChannelAssignment::LeftSide,
            vec![vec![10, -3], vec![4, -9]],
            16,
        )
        .unwrap();
        assert_eq!(planes, vec![vec![10, -3], vec![6, 6]]);
    }

    #[test]
    fn undoes_right_side_decorrelation() {
        let planes = assemble_planes(
            ChannelAssignment::RightSide,
            vec![vec![4, -9], vec![10, -3]],
            16,
        )
        .unwrap();
        assert_eq!(planes, vec![vec![14, -12], vec![10, -3]]);
    }

    #[test]
    fn undoes_mid_side_decorrelation_with_negative_side() {
        // L = 2, R = 5 gives mid = 3 and side = -3.
        let planes =
            assemble_planes(ChannelAssignment::MidSide, vec![vec![3], vec![-3]], 16).unwrap();
        assert_eq!(planes, vec![vec![2], vec![5]]);
        // L = 5, R = 2 gives mid = 3 and side = 3.
        let planes =
            assemble_planes(ChannelAssignment::MidSide, vec![vec![3], vec![3]], 16).unwrap();
        assert_eq!(planes, vec![vec![5], vec![2]]);
    }

    #[test]
    fn rejects_decorrelated_samples_outside_the_bit_depth() {
        let error = assemble_planes(
            ChannelAssignment::LeftSide,
            vec![vec![0], vec![i64::from(i16::MIN)]],
            8,
        )
        .err();
        assert!(matches!(error, Some(FlacError::InvalidFrame(_))));
    }

    #[test]
    fn copies_independent_planes() {
        let planes = assemble_planes(
            ChannelAssignment::Independent,
            vec![vec![-128, 127], vec![0, 1]],
            8,
        )
        .unwrap();
        assert_eq!(planes, vec![vec![-128, 127], vec![0, 1]]);
    }
}
