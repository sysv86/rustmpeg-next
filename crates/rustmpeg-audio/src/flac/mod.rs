// SPDX-License-Identifier: Apache-2.0
//! Native FLAC decoding.
//!
//! This module implements a streaming decoder for native FLAC bitstreams as
//! defined by [RFC 9639].  It reads the `fLaC` marker, validates and skips the
//! metadata blocks, and decodes audio frames into planar native integer PCM
//! ([`PcmFrame`]).
//!
//! Supported frame features:
//!
//! - fixed and variable block size streams, with common and uncommon block
//!   sizes (including the 8-bit and 16-bit explicit forms);
//! - 1 to 8 channels, both independent and the three stereo decorrelation
//!   modes (left/side, right/side, mid/side);
//! - sample sizes 4, 8, 12, 16, 20, 24, and 32 bits, or the STREAMINFO default;
//! - constant, verbatim, fixed-predictor (orders 0 to 4), and linear-predictor
//!   (orders 1 to 32) subframes;
//! - wasted bits and partitioned Rice residuals with 4-bit or 5-bit
//!   parameters, including escaped partitions;
//! - frame header CRC-8 and frame CRC-16 verification.
//!
//! # Resource limits
//!
//! Metadata is read eagerly and audio frames are decoded lazily.  The decoder
//! enforces these hard caps and reports [`FlacError::LimitExceeded`] when one
//! is exceeded:
//!
//! - 64 MiB of metadata, counting the four-byte header of every block;
//! - 1024 metadata blocks;
//! - a 16 MiB leading ID3v2 area, all skipped tags combined;
//! - 64 MiB per audio frame.
//!
//! A single metadata block cannot exceed 16 MiB minus one byte, because its
//! length field is 24 bits wide (RFC 9639, Section 8.1).
//!
//! # Stream properties
//!
//! RFC 9639, Section 9.1 allows every frame to carry its own sample rate,
//! channel count, and bit depth.  The fields of each [`PcmFrame`] describe the
//! frame that was just decoded and are authoritative for it;
//! [`FlacDecoder::stream_info`] only reports the STREAMINFO defaults.
//!
//! Only one native FLAC stream is supported.  Concatenated streams (a second
//! `fLaC` marker after the audio) and trailing tags after the last frame are
//! not handled: decoding ends at the first frame boundary after the
//! STREAMINFO sample count, or when the input ends.
//!
//! The encoder writes native FLAC streams ([`encode_flac`], [`FlacEncoder`]) and
//! Ogg FLAC streams ([`encode_ogg_flac`]); [`OggFlacDecoder`] reads Ogg FLAC.
//! STREAMINFO MD5 signatures are generated and verified, and seekable readers
//! can seek through [`FlacDecoder::seek_to_sample`].
//!
//! [RFC 9639]: https://www.rfc-editor.org/rfc/rfc9639
//! [`PcmFrame`]: crate::PcmFrame

mod bitreader;
mod bitwriter;
mod crc;
mod encoder;
mod frame;
mod md5;
mod ogg;
mod subframe;

use std::io::{Read, Seek, SeekFrom};

use bitreader::BitReader;
use thiserror::Error;

pub use encoder::{FlacEncoder, FlacEncoderConfig, encode_flac, encode_flac_to_vec};
pub use ogg::{OggFlacDecoder, encode_ogg_flac, encode_ogg_flac_to_vec};

/// Maximum combined size of all metadata block headers and payloads (64 MiB).
const MAX_METADATA_TOTAL: u64 = 64 * 1024 * 1024;

/// Maximum number of metadata blocks in one stream.
const MAX_METADATA_BLOCKS: u32 = 1024;

/// Maximum size of a leading ID3v2 tag and of all skipped tags combined.
const MAX_ID3_BYTES: u64 = 16 * 1024 * 1024;

/// Maximum number of bytes in one FLAC frame.
pub(crate) const MAX_FRAME_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum block size representable in a frame and in STREAMINFO.
pub(crate) const MAX_BLOCK_SIZE: u32 = 65_535;

/// Maximum sample rate representable in STREAMINFO (20-bit field).
pub(crate) const MAX_STREAMINFO_SAMPLE_RATE: u32 = 1_048_575;

/// Maximum sample rate representable by the uncommon frame header forms.
///
/// Sample rate codes 12 to 14 store 8 kHz steps, 16-bit hertz, and 10 Hz
/// steps respectively, so the largest value they encode is 655350 Hz.  Code 0
/// delegates to STREAMINFO, whose field is wider (RFC 9639, Section 8.2).
pub(crate) const MAX_FRAME_SAMPLE_RATE: u32 = 655_350;

/// Errors produced while decoding a native FLAC stream.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FlacError {
    /// An I/O error occurred, including truncated input.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The stream does not start with the `fLaC` marker after any skipped
    /// ID3v2 tag.
    #[error("invalid FLAC stream marker")]
    InvalidMagic,
    /// The STREAMINFO block is missing, misplaced, or contains a forbidden
    /// value.
    #[error("invalid STREAMINFO metadata block: {0}")]
    InvalidStreamInfo(&'static str),
    /// A metadata block is malformed.
    #[error("invalid FLAC metadata: {0}")]
    InvalidMetadata(&'static str),
    /// A frame header, subframe, or residual violates the specification.
    #[error("invalid FLAC frame: {0}")]
    InvalidFrame(&'static str),
    /// The stored frame header CRC-8 does not match the computed value.
    #[error("frame header CRC-8 mismatch: stored 0x{stored:02x}, computed 0x{computed:02x}")]
    HeaderCrcMismatch {
        /// The CRC-8 byte stored in the frame header.
        stored: u8,
        /// The CRC-8 computed over the preceding header bytes.
        computed: u8,
    },
    /// The stored frame CRC-16 does not match the computed value.
    #[error("frame CRC-16 mismatch: stored 0x{stored:04x}, computed 0x{computed:04x}")]
    FrameCrcMismatch {
        /// The CRC-16 stored in the frame footer.
        stored: u16,
        /// The CRC-16 computed over the complete frame except the footer.
        computed: u16,
    },
    /// The stream uses a reserved value or a feature outside this stage.
    #[error("unsupported FLAC feature: {0}")]
    UnsupportedFeature(&'static str),
    /// A configured resource limit was exceeded.
    #[error("FLAC limit exceeded: {0}")]
    LimitExceeded(&'static str),
    /// The encoder configuration is invalid.
    #[error("invalid FLAC encoder configuration: {0}")]
    InvalidConfig(&'static str),
    /// The encoder rejected the supplied PCM data.
    #[error("FLAC encoding error: {0}")]
    EncodeError(&'static str),
    /// The Ogg FLAC mapping is malformed.
    #[error("invalid Ogg FLAC stream: {0}")]
    InvalidOgg(&'static str),
    /// The MD5 signature stored in STREAMINFO does not match the decoded audio.
    #[error("FLAC MD5 mismatch: stored {stored:02x?}, computed {computed:02x?}")]
    Md5Mismatch {
        /// Signature stored in STREAMINFO.
        stored: [u8; 16],
        /// Signature computed over the decoded samples.
        computed: [u8; 16],
    },
}

/// Contents of the mandatory STREAMINFO metadata block.
///
/// These values are the stream defaults.  A frame header may override the
/// sample rate, channel count, and bit depth, so the fields of an individual
/// [`PcmFrame`](crate::PcmFrame) are authoritative for that frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamInfo {
    /// Minimum block size in samples, excluding the last frame.
    pub min_block_size: u16,
    /// Maximum block size in samples.
    pub max_block_size: u16,
    /// Minimum frame size in bytes, or zero when unknown.
    pub min_frame_size: u32,
    /// Maximum frame size in bytes, or zero when unknown.
    pub max_frame_size: u32,
    /// Default sample rate in hertz, from 1 to 1048575.
    pub sample_rate: u32,
    /// Number of channels, from 1 to 8.
    pub channels: u8,
    /// Bits per sample, from 4 to 32.
    pub bits_per_sample: u8,
    /// Total number of interchannel samples, or [`None`] when unknown.
    pub total_samples: Option<u64>,
    /// MD5 checksum of the unencoded audio, or all zeros when unknown.
    ///
    /// This stage parses the value but does not verify it.
    pub md5: [u8; 16],
}

impl StreamInfo {
    pub(crate) fn parse(data: &[u8; 34]) -> Result<Self, FlacError> {
        let min_block_size = u16::from_be_bytes([data[0], data[1]]);
        let max_block_size = u16::from_be_bytes([data[2], data[3]]);
        let min_frame_size = u32::from_be_bytes([0, data[4], data[5], data[6]]);
        let max_frame_size = u32::from_be_bytes([0, data[7], data[8], data[9]]);

        let packed = data[10..18]
            .iter()
            .fold(0u64, |value, &byte| (value << 8) | u64::from(byte));
        let sample_rate = ((packed >> 44) & 0x000f_ffff) as u32;
        let channels = ((packed >> 41) & 0x7) as u8 + 1;
        let bits_per_sample = ((packed >> 36) & 0x1f) as u8 + 1;
        let total = packed & 0x0000_000f_ffff_ffff;

        let mut md5 = [0u8; 16];
        md5.copy_from_slice(&data[18..34]);

        if min_block_size < 16 || max_block_size < 16 {
            return Err(FlacError::InvalidStreamInfo(
                "block size must be between 16 and 65535",
            ));
        }
        if min_block_size > max_block_size {
            return Err(FlacError::InvalidStreamInfo(
                "minimum block size exceeds maximum block size",
            ));
        }
        if max_frame_size != 0 && min_frame_size > max_frame_size {
            return Err(FlacError::InvalidStreamInfo(
                "minimum frame size exceeds maximum frame size",
            ));
        }
        if sample_rate == 0 || sample_rate > MAX_STREAMINFO_SAMPLE_RATE {
            return Err(FlacError::InvalidStreamInfo(
                "sample rate is zero or exceeds 1048575 Hz",
            ));
        }
        if !(4..=32).contains(&bits_per_sample) {
            return Err(FlacError::InvalidStreamInfo(
                "bits per sample must be between 4 and 32",
            ));
        }

        Ok(Self {
            min_block_size,
            max_block_size,
            min_frame_size,
            max_frame_size,
            sample_rate,
            channels,
            bits_per_sample,
            total_samples: (total != 0).then_some(total),
            md5,
        })
    }
}

/// One SEEKTABLE entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SeekPoint {
    sample: u64,
    offset: u64,
    samples: u16,
}

/// Streaming decoder for native FLAC bitstreams.
///
/// The decoder reads the stream header eagerly when it is constructed and then
/// yields one [`PcmFrame`](crate::PcmFrame) per call to [`Self::next_frame`]. It
/// buffers no more than one frame, verifies the STREAMINFO MD5 signature at end
/// of stream by default, and supports sample-accurate seeking for
/// [`Read`] + [`Seek`] sources.
///
/// See the [module documentation](self) for the resource limits and for the
/// handling of per-frame stream properties.
pub struct FlacDecoder<R: Read> {
    reader: BitReader<R>,
    info: StreamInfo,
    frame_index: u64,
    next_sample: u64,
    previous_block_size: Option<u32>,
    blocking_variable: Option<bool>,
    failed: bool,
    md5: md5::Md5,
    verify_md5: bool,
    md5_checked: bool,
    seektable: Vec<SeekPoint>,
    strict_frame_numbers: bool,
    pending_frame: Option<crate::PcmFrame>,
}

impl<R: Read> FlacDecoder<R> {
    /// Creates a decoder and parses the stream marker and all metadata.
    ///
    /// A leading ID3v2 tag is skipped if present.
    ///
    /// The decoder fetches one byte at a time from the source and never reads
    /// ahead, so wrap unbuffered sources such as [`std::fs::File`] in a
    /// [`BufReader`](std::io::BufReader) for reasonable performance.  All
    /// metadata is consumed here; the first audio frame is not read until the
    /// first call to [`Self::next_frame`].
    pub fn new(mut reader: R) -> Result<Self, FlacError> {
        locate_marker(&mut reader)?;
        let (info, seektable) = read_metadata(&mut reader)?;
        Ok(Self {
            reader: BitReader::new(reader),
            info,
            frame_index: 0,
            next_sample: 0,
            previous_block_size: None,
            blocking_variable: None,
            failed: false,
            md5: md5::Md5::new(),
            verify_md5: true,
            md5_checked: false,
            seektable,
            strict_frame_numbers: true,
            pending_frame: None,
        })
    }

    /// Enables or disables MD5 verification of the decoded audio.
    ///
    /// Verification is enabled by default.  It runs when the decoder reaches a
    /// clean end of stream; a stream that stops early is not checked.
    pub fn set_verify_md5(&mut self, enabled: bool) {
        self.verify_md5 = enabled;
    }

    /// Verifies the decoded MD5 signature once, at end of stream.
    fn check_md5(&mut self) -> Result<(), FlacError> {
        if self.md5_checked {
            return Ok(());
        }
        self.md5_checked = true;
        if !self.verify_md5 || self.info.md5 == [0; 16] {
            return Ok(());
        }
        let computed = self.md5.clone().finalize();
        if computed != self.info.md5 {
            return Err(FlacError::Md5Mismatch {
                stored: self.info.md5,
                computed,
            });
        }
        Ok(())
    }

    /// Returns the parsed STREAMINFO metadata, which holds the stream defaults.
    ///
    /// A frame header may change the sample rate, channel count, or bit depth
    /// for the frames that follow it.  The [`PcmFrame`](crate::PcmFrame)
    /// returned by [`Self::next_frame`] carries the values that apply to that
    /// frame, and those fields take precedence over this metadata.
    #[must_use]
    pub fn stream_info(&self) -> &StreamInfo {
        &self.info
    }
}

impl<R: Read + Seek> FlacDecoder<R> {
    /// Seeks to the frame containing `target_sample`.
    ///
    /// The nearest SEEKTABLE point at or before the target is used when one is
    /// present; otherwise decoding restarts at the first frame and skips
    /// forward. The returned value is the sample number of the next frame that
    /// [`Self::next_frame`] will produce (the frame containing the target).
    ///
    /// Seeking disables MD5 verification for the remainder of the stream,
    /// because the digest covers the complete audio. Frame-number and
    /// sample-number consistency checks are relaxed after a seek; frame header
    /// CRCs and per-frame CRC-16 checks still apply.
    pub fn seek_to_sample(&mut self, target_sample: u64) -> Result<u64, FlacError> {
        self.failed = false;
        self.pending_frame = None;
        self.md5 = md5::Md5::new();
        self.md5_checked = true;
        self.strict_frame_numbers = false;
        self.previous_block_size = None;

        let reader = self.reader.get_mut();
        reader.seek(SeekFrom::Start(0))?;
        locate_marker(reader)?;
        let (info, points) = read_metadata(reader)?;
        let first_frame = reader.stream_position()?;
        self.info = info;
        self.seektable = points;
        let target = match self.info.total_samples {
            Some(0) => 0,
            Some(total) => target_sample.min(total - 1),
            None => target_sample,
        };
        let point =
            self.seektable.iter().rev().find(|point| {
                point.sample <= target && point.offset != u64::MAX && point.samples > 0
            });
        let (start_sample, offset) = point.map_or((0, 0), |point| (point.sample, point.offset));
        reader.seek(SeekFrom::Start(first_frame + offset))?;
        self.reader.reset();
        self.next_sample = start_sample;
        self.frame_index = 0;

        loop {
            if self.next_sample > target {
                break;
            }
            match self.decode_frame()? {
                Some(frame) => {
                    let end = frame.sample_number + frame.block_size() as u64;
                    if end > target {
                        let start = frame.sample_number;
                        self.pending_frame = Some(frame);
                        return Ok(start);
                    }
                }
                None => return Ok(self.next_sample),
            }
        }
        Ok(self.next_sample)
    }
}

/// Skips any leading ID3v2 tags and requires the `fLaC` marker.
fn locate_marker<R: Read>(reader: &mut R) -> Result<(), FlacError> {
    let mut skipped_tags = 0u64;
    loop {
        let mut marker = [0u8; 4];
        if let Err(error) = reader.read_exact(&mut marker) {
            return if error.kind() == std::io::ErrorKind::UnexpectedEof {
                Err(FlacError::InvalidMagic)
            } else {
                Err(FlacError::Io(error))
            };
        }
        if marker == *b"fLaC" {
            return Ok(());
        }
        if marker[..3] != *b"ID3" {
            return Err(FlacError::InvalidMagic);
        }

        let mut rest = [0u8; 6];
        reader.read_exact(&mut rest)?;
        let size = syncsafe_size(&rest[2..6])?;
        // Only ID3v2.4 defines a footer; bit 4 of the flags byte carries no
        // footer meaning in earlier versions, whose headers are still skipped
        // for compatibility with tags found in the wild (ID3v2.4, Section 3).
        let footer = if marker[3] == 4 && rest[1] & 0x10 != 0 {
            10
        } else {
            0
        };
        let tag_total = 10u64
            .checked_add(size)
            .and_then(|value| value.checked_add(footer))
            .ok_or(FlacError::LimitExceeded("ID3v2 tag size overflow"))?;
        skipped_tags = skipped_tags
            .checked_add(tag_total)
            .ok_or(FlacError::LimitExceeded("ID3v2 tag size overflow"))?;
        if skipped_tags > MAX_ID3_BYTES {
            return Err(FlacError::LimitExceeded(
                "leading ID3v2 tags exceed the 16 MiB limit",
            ));
        }
        skip_bytes(reader, size + footer)?;
    }
}

/// Decodes a four-byte syncsafe integer.
fn syncsafe_size(bytes: &[u8]) -> Result<u64, FlacError> {
    if bytes.len() != 4 || bytes.iter().any(|byte| byte & 0x80 != 0) {
        return Err(FlacError::InvalidMetadata(
            "invalid syncsafe ID3v2 tag size",
        ));
    }
    Ok((u64::from(bytes[0]) << 21)
        | (u64::from(bytes[1]) << 14)
        | (u64::from(bytes[2]) << 7)
        | u64::from(bytes[3]))
}

/// Parses the metadata block chain and returns STREAMINFO.
fn read_metadata<R: Read>(reader: &mut R) -> Result<(StreamInfo, Vec<SeekPoint>), FlacError> {
    let mut total_size = 0u64;
    let mut block_count = 0u32;
    let mut info: Option<StreamInfo> = None;
    let mut seektable = Vec::new();

    loop {
        if block_count == MAX_METADATA_BLOCKS {
            return Err(FlacError::LimitExceeded(
                "metadata exceeds the 1024 block limit",
            ));
        }
        let first_block = info.is_none();
        let mut header = [0u8; 4];
        if let Err(error) = reader.read_exact(&mut header) {
            return if error.kind() == std::io::ErrorKind::UnexpectedEof && first_block {
                Err(FlacError::InvalidStreamInfo(
                    "missing STREAMINFO metadata block",
                ))
            } else {
                Err(FlacError::Io(error))
            };
        }

        let is_last = header[0] & 0x80 != 0;
        let block_type = header[0] & 0x7f;
        let length =
            (u64::from(header[1]) << 16) | (u64::from(header[2]) << 8) | u64::from(header[3]);

        if block_type == 127 {
            return Err(FlacError::InvalidMetadata(
                "metadata block type 127 is forbidden",
            ));
        }
        // The 24-bit length field cannot express more than 16 MiB minus one
        // byte, so the per-block limit needs no separate check.  The header
        // counts toward the stream total along with the payload.
        total_size = total_size
            .checked_add(4 + length)
            .ok_or(FlacError::LimitExceeded("metadata size overflow"))?;
        if total_size > MAX_METADATA_TOTAL {
            return Err(FlacError::LimitExceeded(
                "metadata exceeds the 64 MiB limit",
            ));
        }

        if first_block {
            if block_type != 0 {
                return Err(FlacError::InvalidStreamInfo(
                    "the first metadata block must be STREAMINFO",
                ));
            }
            if length != 34 {
                return Err(FlacError::InvalidStreamInfo(
                    "STREAMINFO must be exactly 34 bytes",
                ));
            }
            let mut data = [0u8; 34];
            reader.read_exact(&mut data)?;
            info = Some(StreamInfo::parse(&data)?);
        } else {
            if block_type == 0 {
                return Err(FlacError::InvalidMetadata(
                    "duplicate STREAMINFO metadata block",
                ));
            }
            if block_type == 3 {
                if !length.is_multiple_of(18) {
                    return Err(FlacError::InvalidMetadata(
                        "SEEKTABLE length is not a multiple of 18",
                    ));
                }
                let mut data = vec![0u8; length as usize];
                reader.read_exact(&mut data)?;
                for entry in data.as_chunks::<18>().0 {
                    let sample = u64::from_be_bytes(entry[0..8].try_into().unwrap());
                    let offset = u64::from_be_bytes(entry[8..16].try_into().unwrap());
                    let samples = u16::from_be_bytes(entry[16..18].try_into().unwrap());
                    // Placeholder points use all-ones and must be ignorable.
                    if sample == u64::MAX {
                        continue;
                    }
                    seektable.push(SeekPoint {
                        sample,
                        offset,
                        samples,
                    });
                }
            } else {
                skip_bytes(reader, length)?;
            }
        }
        block_count += 1;

        if is_last {
            break;
        }
    }

    let info = info.ok_or(FlacError::InvalidStreamInfo(
        "missing STREAMINFO metadata block",
    ))?;
    seektable.sort_by_key(|point| point.sample);
    Ok((info, seektable))
}

/// Discards exactly `length` bytes using a small stack buffer.
fn skip_bytes<R: Read>(reader: &mut R, length: u64) -> Result<(), FlacError> {
    let mut buffer = [0u8; 8192];
    let mut remaining = length;
    while remaining > 0 {
        let take = remaining.min(buffer.len() as u64) as usize;
        reader.read_exact(&mut buffer[..take])?;
        remaining -= take as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{FlacDecoder, FlacError};

    fn stream_with_metadata(blocks: &[(u8, Vec<u8>)]) -> Vec<u8> {
        let mut out = b"fLaC".to_vec();
        for (index, (kind, payload)) in blocks.iter().enumerate() {
            let last = index + 1 == blocks.len();
            let mut header = [0u8; 4];
            header[0] = (u8::from(last) << 7) | (kind & 0x7f);
            let length = u32::try_from(payload.len()).unwrap();
            header[1..].copy_from_slice(&length.to_be_bytes()[1..]);
            out.extend_from_slice(&header);
            out.extend_from_slice(payload);
        }
        out
    }

    fn streaminfo_payload(min_block: u16, max_block: u16, rate: u32, bits: u8) -> Vec<u8> {
        let mut payload = vec![0u8; 34];
        payload[0..2].copy_from_slice(&min_block.to_be_bytes());
        payload[2..4].copy_from_slice(&max_block.to_be_bytes());
        payload[4..7].copy_from_slice(&[0, 0, 0]);
        payload[7..10].copy_from_slice(&[0, 0, 0]);
        let packed = (u64::from(rate) << 44) | (u64::from(0u8) << 41) | (u64::from(bits - 1) << 36);
        payload[10..18].copy_from_slice(&packed.to_be_bytes());
        payload
    }

    #[test]
    fn rejects_invalid_magic() {
        let error = FlacDecoder::new(Cursor::new(b"OggS".to_vec())).err();
        assert!(matches!(error, Some(FlacError::InvalidMagic)));
    }

    #[test]
    fn rejects_missing_streaminfo() {
        let error = FlacDecoder::new(Cursor::new(b"fLaC".to_vec())).err();
        assert!(matches!(error, Some(FlacError::InvalidStreamInfo(_))));
    }

    #[test]
    fn rejects_streaminfo_that_is_not_first() {
        let stream =
            stream_with_metadata(&[(1, vec![0; 4]), (0, streaminfo_payload(16, 16, 8000, 8))]);
        let error = FlacDecoder::new(Cursor::new(stream)).err();
        assert!(matches!(error, Some(FlacError::InvalidStreamInfo(_))));
    }

    #[test]
    fn rejects_duplicate_streaminfo() {
        let stream = stream_with_metadata(&[
            (0, streaminfo_payload(16, 16, 8000, 8)),
            (0, streaminfo_payload(16, 16, 8000, 8)),
        ]);
        let error = FlacDecoder::new(Cursor::new(stream)).err();
        assert!(matches!(error, Some(FlacError::InvalidMetadata(_))));
    }

    #[test]
    fn rejects_forbidden_metadata_type() {
        let stream =
            stream_with_metadata(&[(0, streaminfo_payload(16, 16, 8000, 8)), (127, vec![])]);
        let error = FlacDecoder::new(Cursor::new(stream)).err();
        assert!(matches!(error, Some(FlacError::InvalidMetadata(_))));
    }

    #[test]
    fn validates_streaminfo_values() {
        for payload in [
            streaminfo_payload(15, 16, 8000, 8),
            streaminfo_payload(32, 16, 8000, 8),
            streaminfo_payload(16, 16, 0, 8),
            streaminfo_payload(16, 16, 8000, 3),
        ] {
            let stream = stream_with_metadata(&[(0, payload)]);
            let error = FlacDecoder::new(Cursor::new(stream)).err();
            assert!(matches!(error, Some(FlacError::InvalidStreamInfo(_))));
        }
    }

    #[test]
    fn accepts_sample_rates_above_the_frame_header_limit() {
        // RFC 9639, Section 8.2 allows up to 1048575 Hz in STREAMINFO even
        // though the uncommon frame header forms stop at 655350 Hz.
        for rate in [655_350, 768_000, 1_048_575] {
            let stream = stream_with_metadata(&[(0, streaminfo_payload(16, 16, rate, 8))]);
            let decoder = FlacDecoder::new(Cursor::new(stream)).unwrap();
            assert_eq!(decoder.stream_info().sample_rate, rate);
        }
        // The STREAMINFO field is 20 bits wide, so the validated limit is the
        // largest value the field can hold and a larger one cannot be encoded.
        assert_eq!(
            u64::from(super::MAX_STREAMINFO_SAMPLE_RATE),
            (1u64 << 20) - 1
        );
    }

    #[test]
    fn accepts_metadata_up_to_the_block_count_limit() {
        let mut blocks = vec![(0, streaminfo_payload(16, 16, 8000, 8))];
        blocks.extend((0..1023).map(|_| (1, Vec::new())));
        assert_eq!(blocks.len(), 1024);
        let stream = stream_with_metadata(&blocks);
        let decoder = FlacDecoder::new(Cursor::new(stream)).unwrap();
        assert_eq!(decoder.stream_info().sample_rate, 8000);
    }

    #[test]
    fn rejects_too_many_metadata_blocks() {
        let mut blocks = vec![(0, streaminfo_payload(16, 16, 8000, 8))];
        blocks.extend((0..1024).map(|_| (1, Vec::new())));
        let stream = stream_with_metadata(&blocks);
        let error = FlacDecoder::new(Cursor::new(stream)).err();
        assert!(matches!(error, Some(FlacError::LimitExceeded(_))));
    }

    /// Generates `fLaC`, one STREAMINFO block, and then an endless run of
    /// maximum sized padding blocks without allocating their payloads.
    struct PaddingBlockStream {
        prologue: Vec<u8>,
        position: u64,
    }

    /// A padding block header whose payload is the largest a 24-bit length can
    /// describe, plus that payload.
    const PADDING_HEADER: [u8; 4] = [0x01, 0xff, 0xff, 0xff];
    const PADDING_PERIOD: u64 = 4 + 0x00ff_ffff;

    impl std::io::Read for PaddingBlockStream {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let prologue_len = self.prologue.len() as u64;
            let mut written = 0;
            while written < buffer.len() {
                let position = self.position;
                if position < prologue_len {
                    buffer[written] = self.prologue[position as usize];
                    written += 1;
                    self.position += 1;
                    continue;
                }
                let offset = (position - prologue_len) % PADDING_PERIOD;
                let room = buffer.len() - written;
                if offset < 4 {
                    let take = (4 - offset).min(room as u64) as usize;
                    let start = offset as usize;
                    buffer[written..written + take]
                        .copy_from_slice(&PADDING_HEADER[start..start + take]);
                    written += take;
                    self.position += take as u64;
                } else {
                    let take = (PADDING_PERIOD - offset).min(room as u64) as usize;
                    buffer[written..written + take].fill(0);
                    written += take;
                    self.position += take as u64;
                }
            }
            Ok(written)
        }
    }

    #[test]
    fn counts_metadata_headers_towards_the_size_limit() {
        // Four payloads of 16 MiB minus one byte still fit in 64 MiB on their
        // own; their four-byte headers are what push the chain over the cap.
        let payload_total = 4 * u64::from(0x00ff_ffffu32);
        assert!(payload_total <= 64 * 1024 * 1024);
        assert!(payload_total + 4 * 4 + 38 > 64 * 1024 * 1024);

        let mut prologue = b"fLaC".to_vec();
        prologue.extend_from_slice(&[0x00, 0x00, 0x00, 0x22]);
        prologue.extend_from_slice(&streaminfo_payload(16, 16, 8000, 8));
        let mut stream = PaddingBlockStream {
            prologue,
            position: 0,
        };
        let error = FlacDecoder::new(&mut stream).err();
        assert!(matches!(error, Some(FlacError::LimitExceeded(_))));
        // The first three padding blocks were skipped and the fourth header,
        // which pushes the total over the cap, was read.
        assert_eq!(
            stream.position,
            42 + 3 * PADDING_PERIOD + 4,
            "the limit must be evaluated from block headers and payloads"
        );
    }

    #[test]
    fn ignores_the_id3v2_footer_flag_before_version_4() {
        // Bit 4 of the flags byte only means "footer present" in ID3v2.4.
        let mut data = b"ID3\x03\x00\x10\x00\x00\x00\x08".to_vec();
        data.extend_from_slice(&[0u8; 8]);
        data.extend_from_slice(&stream_with_metadata(&[(
            0,
            streaminfo_payload(16, 16, 8000, 8),
        )]));
        let decoder = FlacDecoder::new(Cursor::new(data)).unwrap();
        assert_eq!(decoder.stream_info().sample_rate, 8000);
    }

    #[test]
    fn skips_the_id3v2_footer_when_version_4_announces_one() {
        let mut data = b"ID3\x04\x00\x10\x00\x00\x00\x08".to_vec();
        data.extend_from_slice(&[0u8; 8]);
        data.extend_from_slice(b"3DI\x04\x00\x10\x00\x00\x00\x08");
        data.extend_from_slice(&stream_with_metadata(&[(
            0,
            streaminfo_payload(16, 16, 8000, 8),
        )]));
        let decoder = FlacDecoder::new(Cursor::new(data)).unwrap();
        assert_eq!(decoder.stream_info().sample_rate, 8000);
    }

    #[test]
    fn accepts_known_and_unknown_metadata_blocks() {
        let stream = stream_with_metadata(&[
            (0, streaminfo_payload(16, 16, 8000, 8)),
            (1, vec![0; 8]),
            (2, vec![0; 12]),
            (3, vec![0; 18]),
            (4, vec![0; 5]),
            (5, vec![0; 7]),
            (6, vec![0; 9]),
            (99, vec![0; 11]),
        ]);
        let decoder = FlacDecoder::new(Cursor::new(stream)).unwrap();
        assert_eq!(decoder.stream_info().sample_rate, 8000);
        assert_eq!(decoder.stream_info().channels, 1);
        assert_eq!(decoder.stream_info().bits_per_sample, 8);
    }

    #[test]
    fn skips_a_leading_id3v2_tag() {
        let mut data = b"ID3\x04\x00\x00\x00\x00\x00\x08".to_vec();
        data.extend_from_slice(&[0u8; 8]);
        data.extend_from_slice(b"fLaC");
        data.extend_from_slice(
            &stream_with_metadata(&[(0, streaminfo_payload(16, 16, 8000, 8))])[4..],
        );
        let decoder = FlacDecoder::new(Cursor::new(data)).unwrap();
        assert_eq!(decoder.stream_info().sample_rate, 8000);
    }

    #[test]
    fn rejects_oversized_id3v2_tag() {
        let mut data = b"ID3\x04\x00\x00\x7f\x7f\x7f\x7f".to_vec();
        data.extend_from_slice(&[0u8; 16]);
        let error = FlacDecoder::new(Cursor::new(data)).err();
        assert!(matches!(error, Some(FlacError::LimitExceeded(_))));
    }

    #[test]
    fn rejects_truncated_metadata() {
        let stream = stream_with_metadata(&[(0, streaminfo_payload(16, 16, 8000, 8))]);
        let error = FlacDecoder::new(Cursor::new(stream[..20].to_vec())).err();
        assert!(matches!(error, Some(FlacError::Io(_))));
    }
}
