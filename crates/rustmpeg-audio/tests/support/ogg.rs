// SPDX-License-Identifier: Apache-2.0
//! Test-only Ogg reader for Opus fixtures.
//!
//! Ogg is outside the supported container matrix of `rustmpeg-next`, so this
//! module exists only under `tests/`.  It implements the minimum needed to turn
//! the generated `.opus` fixtures into header packets and audio packets for
//! [`rustmpeg_audio::opus`] tests:
//!
//! - Ogg page framing, CRC-32, sequence numbers, and continued packets
//!   (RFC 3533);
//! - Opus identification headers (RFC 7845, Section 5.1).
//!
//! It is not a product API and must not be promoted into the crate.

use thiserror::Error;

/// Ogg capture pattern.
const CAPTURE_PATTERN: &[u8; 4] = b"OggS";

/// Fixed size of an Ogg page header, excluding the segment table.
const PAGE_HEADER_LEN: usize = 27;

/// Mask of the "continued packet" header flag.
const FLAG_CONTINUED: u8 = 0x01;

/// Mask of the "beginning of stream" header flag.
const FLAG_BOS: u8 = 0x02;

/// Maximum size of one assembled packet accepted by this reader.
const MAX_PACKET_SIZE: usize = 1 << 20;

/// Errors produced while reading a test Ogg Opus file.
#[derive(Debug, Error)]
pub enum OggError {
    /// The capture pattern or version is not valid Ogg.
    #[error("invalid Ogg page header: {0}")]
    BadHeader(&'static str),
    /// The page ends before its declared header or body.
    #[error("truncated Ogg page: {0}")]
    TruncatedPage(&'static str),
    /// The stored page checksum does not match the computed one.
    #[error("Ogg page CRC mismatch: stored 0x{stored:08x}, computed 0x{computed:08x}")]
    BadCrc {
        /// Checksum stored in the page header.
        stored: u32,
        /// Checksum computed over the page with the checksum field zeroed.
        computed: u32,
    },
    /// A page sequence number is not the successor of the previous one.
    #[error("Ogg page sequence gap: expected {expected}, found {found}")]
    SequenceGap {
        /// Sequence number that should have appeared.
        expected: u32,
        /// Sequence number found in the page.
        found: u32,
    },
    /// The stream starts without a beginning-of-stream page.
    #[error("Ogg stream does not start with a beginning-of-stream page")]
    MissingBos,
    /// A beginning-of-stream page appears after the first page.
    #[error("unexpected beginning-of-stream page")]
    UnexpectedBos,
    /// A page continues a packet although none is pending.
    #[error("Ogg page continues a packet that does not exist")]
    ContinuationWithoutPacket,
    /// A page starts a packet although the previous page left one pending.
    #[error("Ogg page does not continue the pending packet")]
    MissingContinuation,
    /// The page belongs to a different logical stream.
    #[error("Ogg stream serial mismatch: expected {expected}, found {found}")]
    SerialMismatch {
        /// Serial number of the logical stream.
        expected: u32,
        /// Serial number found in the page.
        found: u32,
    },
    /// A packet grew beyond the reader's limit.
    #[error("Ogg packet exceeds {MAX_PACKET_SIZE} bytes")]
    PacketTooLarge,
    /// The Opus identification header is malformed.
    #[error("invalid OpusHead packet: {0}")]
    BadOpusHead(&'static str),
}

/// One parsed Ogg page.
#[derive(Debug, Clone)]
pub struct OggPage {
    /// Header type flags.
    pub header_type: u8,
    /// Granule position, or -1 when no packet completes on the page.
    pub granule: i64,
    /// Logical stream serial number.
    pub serial: u32,
    /// Page sequence number.
    pub sequence: u32,
    /// Segment table.
    pub lacing: Vec<u8>,
    /// Page body.
    pub body: Vec<u8>,
    /// Offset of the page in the file.
    pub offset: usize,
    /// Total encoded size of the page.
    pub length: usize,
}

/// One packet assembled across pages.
#[derive(Debug, Clone)]
pub struct OggPacket {
    /// Packet bytes.
    pub data: Vec<u8>,
    /// Sequence number of the page on which the packet completed.
    pub page_sequence: u32,
    /// Granule position of that page.
    pub granule: i64,
}

/// A parsed Ogg Opus file.
#[derive(Debug)]
pub struct OggStream {
    /// All pages, in file order.
    pub pages: Vec<OggPage>,
    /// All packets, in stream order.
    pub packets: Vec<OggPacket>,
}

/// Parsed Opus identification header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusHead {
    /// Encapsulation version.
    pub version: u8,
    /// Output channel count.
    pub channels: u8,
    /// Samples at 48 kHz to skip at the start of playback.
    pub pre_skip: u16,
    /// Original input sample rate.
    pub input_sample_rate: u32,
    /// Output gain in Q7.8 dB.
    pub output_gain: i16,
    /// Channel mapping family.
    pub mapping_family: u8,
    /// Number of streams (mapping families 1 and 255).
    pub stream_count: u8,
    /// Number of coupled streams (mapping families 1 and 255).
    pub coupled_count: u8,
    /// Output channel mapping table.
    pub channel_mapping: Vec<u8>,
}

/// Computes the Ogg CRC-32 (polynomial 0x04c11db7, no reflection, initial value
/// and final XOR both zero).
#[must_use]
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0u32;
    for &byte in bytes {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04c1_1db7
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// Parses every page, validating framing and checksums.
///
/// # Errors
///
/// Returns [`OggError`] for empty input, a bad capture pattern or version, a
/// page that ends early, or a checksum mismatch.
pub fn parse_pages(data: &[u8]) -> Result<Vec<OggPage>, OggError> {
    if data.is_empty() {
        return Err(OggError::BadHeader("empty stream"));
    }
    let mut pages = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let header = data
            .get(offset..offset + PAGE_HEADER_LEN)
            .ok_or(OggError::TruncatedPage("page header"))?;
        if &header[0..4] != CAPTURE_PATTERN {
            return Err(OggError::BadHeader("capture pattern"));
        }
        if header[4] != 0 {
            return Err(OggError::BadHeader("unsupported version"));
        }
        let header_type = header[5];
        let granule = i64::from_le_bytes(header[6..14].try_into().expect("slice length"));
        let serial = u32::from_le_bytes(header[14..18].try_into().expect("slice length"));
        let sequence = u32::from_le_bytes(header[18..22].try_into().expect("slice length"));
        let stored_crc = u32::from_le_bytes(header[22..26].try_into().expect("slice length"));
        let segment_count = usize::from(header[26]);
        let lacing = data
            .get(offset + PAGE_HEADER_LEN..offset + PAGE_HEADER_LEN + segment_count)
            .ok_or(OggError::TruncatedPage("segment table"))?
            .to_vec();
        let body_len: usize = lacing.iter().map(|&lace| usize::from(lace)).sum();
        let body_start = offset + PAGE_HEADER_LEN + segment_count;
        let body = data
            .get(body_start..body_start + body_len)
            .ok_or(OggError::TruncatedPage("page body"))?
            .to_vec();
        let length = body_start + body_len - offset;
        let mut checksummed = data[offset..offset + length].to_vec();
        checksummed[22..26].fill(0);
        let computed = crc32(&checksummed);
        if computed != stored_crc {
            return Err(OggError::BadCrc {
                stored: stored_crc,
                computed,
            });
        }
        pages.push(OggPage {
            header_type,
            granule,
            serial,
            sequence,
            lacing,
            body,
            offset,
            length,
        });
        offset += length;
    }
    Ok(pages)
}

/// Parses an Ogg stream and assembles its packets.
///
/// Sequence numbers must be consecutive, the first page must be a
/// beginning-of-stream page, and the continued-packet flag must agree with the
/// lacing values of the previous page (RFC 7845, Section 3).
///
/// # Errors
///
/// Returns [`OggError`] for page framing errors or any of the continuity rules
/// above.
pub fn read_stream(data: &[u8]) -> Result<OggStream, OggError> {
    let pages = parse_pages(data)?;
    let mut packets = Vec::new();
    let mut current = Vec::new();
    let mut pending = false;
    let mut serial = None;
    let mut expected_sequence = None;
    for (index, page) in pages.iter().enumerate() {
        if let Some(expected) = serial {
            if page.serial != expected {
                return Err(OggError::SerialMismatch {
                    expected,
                    found: page.serial,
                });
            }
        } else {
            serial = Some(page.serial);
        }
        if let Some(expected) = expected_sequence
            && page.sequence != expected
        {
            return Err(OggError::SequenceGap {
                expected,
                found: page.sequence,
            });
        }
        expected_sequence = Some(page.sequence.wrapping_add(1));
        if index == 0 {
            if page.header_type & FLAG_BOS == 0 {
                return Err(OggError::MissingBos);
            }
        } else if page.header_type & FLAG_BOS != 0 {
            return Err(OggError::UnexpectedBos);
        }
        let continued = page.header_type & FLAG_CONTINUED != 0;
        if continued != pending {
            return Err(if continued {
                OggError::ContinuationWithoutPacket
            } else {
                OggError::MissingContinuation
            });
        }
        let mut position = 0;
        for &lace in &page.lacing {
            let take = usize::from(lace);
            current.extend_from_slice(&page.body[position..position + take]);
            position += take;
            if current.len() > MAX_PACKET_SIZE {
                return Err(OggError::PacketTooLarge);
            }
            if lace < 255 {
                packets.push(OggPacket {
                    data: std::mem::take(&mut current),
                    page_sequence: page.sequence,
                    granule: page.granule,
                });
                pending = false;
            } else {
                pending = true;
            }
        }
    }
    if pending {
        return Err(OggError::TruncatedPage("final packet is incomplete"));
    }
    Ok(OggStream { pages, packets })
}

/// Parses an Opus identification header (RFC 7845, Section 5.1).
///
/// # Errors
///
/// Returns [`OggError::BadOpusHead`] if the magic, version, channel count,
/// mapping family, or mapping table is invalid.
pub fn parse_opus_head(packet: &[u8]) -> Result<OpusHead, OggError> {
    if packet.len() < 19 {
        return Err(OggError::BadOpusHead("shorter than 19 bytes"));
    }
    if &packet[0..8] != b"OpusHead" {
        return Err(OggError::BadOpusHead("wrong magic"));
    }
    let version = packet[8];
    if version == 0 || version & 0xf0 != 0 {
        return Err(OggError::BadOpusHead("unsupported version"));
    }
    let channels = packet[9];
    if channels == 0 {
        return Err(OggError::BadOpusHead("zero channels"));
    }
    let pre_skip = u16::from_le_bytes([packet[10], packet[11]]);
    let input_sample_rate = u32::from_le_bytes([packet[12], packet[13], packet[14], packet[15]]);
    let output_gain = i16::from_le_bytes([packet[16], packet[17]]);
    let mapping_family = packet[18];
    let (stream_count, coupled_count, channel_mapping) = match mapping_family {
        0 => {
            if channels > 2 {
                return Err(OggError::BadOpusHead("family 0 with more than 2 channels"));
            }
            (1, u8::from(channels == 2), Vec::new())
        }
        1 | 255 => {
            if packet.len() < 21 + usize::from(channels) {
                return Err(OggError::BadOpusHead("truncated channel mapping"));
            }
            let stream_count = packet[19];
            let coupled_count = packet[20];
            if stream_count == 0 || coupled_count > stream_count {
                return Err(OggError::BadOpusHead("invalid stream count"));
            }
            (
                stream_count,
                coupled_count,
                packet[21..21 + usize::from(channels)].to_vec(),
            )
        }
        _ => return Err(OggError::BadOpusHead("unknown channel mapping family")),
    };
    Ok(OpusHead {
        version,
        channels,
        pre_skip,
        input_sample_rate,
        output_gain,
        mapping_family,
        stream_count,
        coupled_count,
        channel_mapping,
    })
}

/// Rewrites a page's sequence number and checksum.
///
/// Used by malformed-input tests; `page` must be exactly one encoded page.
#[must_use]
pub fn with_sequence(page: &[u8], sequence: u32) -> Vec<u8> {
    let mut page = page.to_vec();
    page[18..22].copy_from_slice(&sequence.to_le_bytes());
    page[22..26].fill(0);
    let crc = crc32(&page);
    page[22..26].copy_from_slice(&crc.to_le_bytes());
    page
}

impl OggPage {
    /// Returns `true` if the beginning-of-stream flag is set.
    #[must_use]
    pub fn is_bos(&self) -> bool {
        self.header_type & FLAG_BOS != 0
    }
}
