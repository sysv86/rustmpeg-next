// SPDX-License-Identifier: Apache-2.0
//! Opus bitstream framing and entropy decoding.
//!
//! This stage implements the layer that sits directly below the SILK and CELT
//! decoders:
//!
//! - [`OpusPacket`] parses the table-of-contents byte, frame packing, and Opus
//!   padding of one Opus packet, as defined by RFC 6716, Section 3, and exposes
//!   the compressed frames as borrowed slices;
//! - [`RangeDecoder`] implements the range decoder of RFC 6716, Section 4.1,
//!   including table-based symbols, uniformly distributed integers, and raw
//!   bits.
//!
//! The SILK layer builds on this module in [`silk`], which decodes SILK-only
//! frames to PCM.  CELT, hybrid mode, and packet loss concealment arrive in
//! later stages.  Opus in Ogg (RFC 7845) is outside the supported container
//! matrix; the test suite contains a private Ogg reader used only to obtain
//! packets and headers from test fixtures.
//!
//! # Limits
//!
//! RFC 6716 requires a receiver to reject malformed packets.  This parser
//! enforces the rules `[R1]` through `[R7]` of Section 3.4:
//!
//! - at least one byte per packet, and at most 48 frames per packet;
//! - no frame larger than 1275 bytes;
//! - even payload for code 1, a complete and in-bounds length for code 2;
//! - no more than 120 ms of audio per packet;
//! - consistent CBR frame counts and in-bounds VBR lengths and padding.
//!
//! Frame data must span the packet exactly: the sizes and offsets returned by
//! the parser cover every payload byte and stop at the Opus padding.
//!
//! [RFC 6716]: https://www.rfc-editor.org/rfc/rfc6716

mod range;
pub mod silk;

use thiserror::Error;

pub use range::RangeDecoder;

/// Maximum size of one Opus frame in bytes (RFC 6716, Section 3.2.1, `[R2]`).
pub const MAX_FRAME_BYTES: usize = 1275;

/// Maximum size of one complete Opus packet accepted by the parser.
///
/// The largest packet RFC 6716 permits (48 frames of 1275 bytes plus framing)
/// is far below this bound.  The cap is a denial-of-service guard: it keeps a
/// hostile transport from making the parser allocate for an arbitrarily large
/// slice before the framing rules reject it.
pub const MAX_PACKET_BYTES: usize = 1 << 20;

/// Maximum number of frames in one packet, imposed by the 120 ms duration cap
/// and the shortest supported frame size of 2.5 ms (RFC 6716, Section 3.2.5).
pub const MAX_FRAMES_PER_PACKET: u8 = 48;

/// Maximum audio duration of one packet in microseconds.
pub const MAX_PACKET_DURATION_US: u32 = 120_000;

/// Operating mode encoded by a packet's configuration number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpusMode {
    /// Low-bitrate speech mode using the LP (SILK) layer only.
    Silk,
    /// Medium-bitrate speech mode combining the SILK and CELT layers.
    Hybrid,
    /// Low-delay mode using the MDCT (CELT) layer only.
    Celt,
}

/// Audio bandwidth encoded by a packet's configuration number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpusBandwidth {
    /// Narrowband, approximately 4 kHz.
    Narrowband,
    /// Mediumband, approximately 6 kHz.
    Mediumband,
    /// Wideband, approximately 8 kHz.
    Wideband,
    /// Super-wideband, approximately 12 kHz.
    SuperWideband,
    /// Fullband, approximately 20 kHz.
    Fullband,
}

/// Errors produced while parsing Opus packets or decoding range-coded symbols.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OpusError {
    /// The packet has no bytes, violating `[R1]`.
    #[error("empty Opus packet")]
    EmptyPacket,
    /// The packet exceeds the parser's [`MAX_PACKET_BYTES`] denial-of-service
    /// guard.
    #[error("Opus packet of {size} bytes exceeds the 1 MiB limit")]
    PacketTooLarge {
        /// Size of the offending packet in bytes.
        size: usize,
    },
    /// The packet ends before a required header or frame length field.
    #[error("truncated Opus packet: {0}")]
    Truncated(&'static str),
    /// The packet framing violates a rule of RFC 6716, Section 3.4.
    #[error("malformed Opus packet: {0}")]
    Malformed(&'static str),
    /// A frame is larger than the 1275-byte limit of `[R2]`.
    #[error("Opus frame of {size} bytes exceeds the 1275-byte limit")]
    FrameTooLarge {
        /// Size of the offending frame in bytes.
        size: usize,
    },
    /// A packet signals more than 48 frames, exceeding the 120 ms cap of `[R5]`.
    #[error("Opus packet signals {count} frames, exceeding the 48-frame limit")]
    TooManyFrames {
        /// Number of frames signalled by the packet.
        count: u8,
    },
    /// A packet contains more than 120 ms of audio, violating `[R5]`.
    #[error("Opus packet duration of {duration_us} us exceeds 120 ms")]
    DurationTooLong {
        /// Total audio duration of the packet in microseconds.
        duration_us: u32,
    },
    /// Opus padding or its length bytes do not fit in the packet, violating
    /// `[R6]` or `[R7]`.
    #[error("Opus padding of {padding} bytes exceeds the {available} available bytes")]
    PaddingTooLarge {
        /// Number of header and trailing padding bytes required.
        padding: usize,
        /// Number of bytes available after the TOC and frame count bytes.
        available: usize,
    },
    /// The range decoder state is inconsistent with the requested symbol.
    #[error("range decoder error: {0}")]
    RangeDecoder(&'static str),
    /// A uniformly distributed integer decoded outside its declared range,
    /// which indicates a corrupt frame (RFC 6716, Section 4.1.5).
    #[error("range decoder returned {value}, outside the declared range 0..{ft}")]
    RangeValue {
        /// Value produced by the decoder.
        value: u32,
        /// Total frequency, the exclusive upper bound of the value.
        ft: u32,
    },
    /// The caller's PCM buffer cannot hold the decoded frame.
    #[error("output buffer too small: need {needed} samples, have {available}")]
    OutputTooSmall {
        /// Required number of interleaved samples.
        needed: usize,
        /// Number of samples in the provided buffer.
        available: usize,
    },
}

/// Parameters common to every frame in an Opus packet.
///
/// All frames in a packet share the configuration, mode, bandwidth, frame
/// duration, and channel count; only their compressed sizes differ.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusPacketInfo {
    /// Configuration number from the top five bits of the TOC byte, `0..=31`.
    pub config: u8,
    /// Operating mode selected by `config`.
    pub mode: OpusMode,
    /// Audio bandwidth selected by `config`.
    pub bandwidth: OpusBandwidth,
    /// Duration of each frame in microseconds: 2500, 5000, 10000, 20000,
    /// 40000, or 60000.
    pub frame_duration_us: u32,
    /// Whether the packet codes two channels rather than one.
    pub stereo: bool,
    /// Number of frames in the packet, `1..=48`.
    ///
    /// This always equals `frame_sizes.len()`.
    pub frame_count: u8,
    /// Compressed size of each frame in bytes, in packet order.
    pub frame_sizes: Vec<u16>,
    /// Number of trailing Opus padding bytes.
    ///
    /// This excludes the bytes that encode the padding length, and is zero
    /// when the packet does not set the padding bit.
    pub padding: usize,
}

impl OpusPacketInfo {
    /// Returns the total audio duration of the packet in microseconds.
    ///
    /// This is `frame_duration_us * frame_count`.  [`OpusPacket::parse`] is the
    /// intended constructor: a parsed packet never exceeds
    /// [`MAX_PACKET_DURATION_US`].  Because the fields are public, a manually
    /// constructed value may be arbitrary; this method saturates instead of
    /// panicking on overflow.
    #[must_use]
    pub fn total_duration_us(&self) -> u32 {
        self.frame_duration_us
            .saturating_mul(u32::from(self.frame_count))
    }
}

/// A parsed Opus packet borrowing its compressed data.
///
/// The packet keeps the original byte slice and a table of frame offsets, so
/// [`OpusPacket::frames`] yields sub-slices without copying.  The packet data
/// is not self-delimiting; the transport must supply the packet length, which
/// is the length of the slice passed to [`OpusPacket::parse`].
#[derive(Debug, Clone)]
pub struct OpusPacket<'a> {
    /// Original packet bytes, including the TOC byte and any padding.
    data: &'a [u8],
    /// Shared parameters and frame sizes.
    info: OpusPacketInfo,
    /// Start offset of each frame within `data`, parallel to
    /// `info.frame_sizes`.
    frame_offsets: Vec<usize>,
}

impl<'a> OpusPacket<'a> {
    /// Parses one complete Opus packet.
    ///
    /// The slice must contain exactly one packet, starting with its TOC byte
    /// and ending after the last frame or padding byte.
    ///
    /// # Errors
    ///
    /// Returns an [`OpusError`] if the packet violates any rule of RFC 6716,
    /// Section 3.4, including an empty packet, a truncated length field, an
    /// oversized frame, inconsistent CBR sizing, padding that does not fit, or
    /// a duration above 120 ms.  Packets larger than [`MAX_PACKET_BYTES`] are
    /// rejected before any framing work as a denial-of-service guard.
    pub fn parse(data: &'a [u8]) -> Result<Self, OpusError> {
        if data.len() > MAX_PACKET_BYTES {
            return Err(OpusError::PacketTooLarge { size: data.len() });
        }
        let toc = *data.first().ok_or(OpusError::EmptyPacket)?;
        let config = toc >> 3;
        let stereo = toc & 0x04 != 0;
        let code = toc & 0x03;
        let (mode, bandwidth, frame_duration_us) = config_parameters(config);

        let (payload_start, sizes, padding) = match code {
            // Code 0: one frame filling the rest of the packet.
            0 => (1, vec![data.len() - 1], 0),
            // Code 1: two frames of equal size; the payload must be even.
            1 => {
                let payload = data.len() - 1;
                if payload % 2 != 0 {
                    return Err(OpusError::Malformed("code 1 payload length is not even"));
                }
                let half = payload / 2;
                (1, vec![half, half], 0)
            }
            // Code 2: two frames with an explicit length for the first one.
            2 => {
                let (first, used) = read_frame_length(&data[1..])?;
                let payload_start = 1 + used;
                let remaining = data.len() - payload_start;
                if first > remaining {
                    return Err(OpusError::Malformed(
                        "code 2 first frame length exceeds the payload",
                    ));
                }
                (payload_start, vec![first, remaining - first], 0)
            }
            // Code 3: an arbitrary frame count with optional padding and,
            // for VBR packets, explicit lengths for all but the last frame.
            _ => parse_code_3(data)?,
        };

        let mut frame_offsets = Vec::with_capacity(sizes.len());
        let mut offset = payload_start;
        for &size in &sizes {
            if size > MAX_FRAME_BYTES {
                return Err(OpusError::FrameTooLarge { size });
            }
            frame_offsets.push(offset);
            offset += size;
        }
        if offset != data.len() - padding {
            return Err(OpusError::Malformed(
                "frame data does not span the packet exactly",
            ));
        }

        let frame_count = u8::try_from(sizes.len())
            .map_err(|_| OpusError::Malformed("frame count does not fit in a byte"))?;
        let duration_us = frame_duration_us * u32::from(frame_count);
        if duration_us > MAX_PACKET_DURATION_US {
            return Err(OpusError::DurationTooLong { duration_us });
        }

        let frame_sizes = sizes
            .iter()
            .map(|&size| u16::try_from(size).map_err(|_| OpusError::FrameTooLarge { size }))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            data,
            info: OpusPacketInfo {
                config,
                mode,
                bandwidth,
                frame_duration_us,
                stereo,
                frame_count,
                frame_sizes,
                padding,
            },
            frame_offsets,
        })
    }

    /// Returns the packet's shared parameters and frame sizes.
    #[must_use]
    pub fn info(&self) -> &OpusPacketInfo {
        &self.info
    }

    /// Iterates over the compressed frame payloads without copying.
    ///
    /// The iterator yields one slice per frame, in packet order and with the
    /// sizes reported by [`OpusPacketInfo::frame_sizes`].  A zero-length frame
    /// yields an empty slice, which is valid: any Opus frame may have length
    /// zero (RFC 6716, Section 3.2.1).
    pub fn frames(&self) -> impl Iterator<Item = &'a [u8]> + '_ {
        self.frame_offsets
            .iter()
            .zip(&self.info.frame_sizes)
            .map(move |(&offset, &size)| {
                let end = offset + usize::from(size);
                self.data.get(offset..end).unwrap_or(&[])
            })
    }
}

/// Parses the header of a code 3 packet and returns its payload layout.
///
/// Returns the offset of the first compressed frame, the size of every frame,
/// and the number of trailing padding bytes.
fn parse_code_3(data: &[u8]) -> Result<(usize, Vec<usize>, usize), OpusError> {
    let count_byte = *data
        .get(1)
        .ok_or(OpusError::Truncated("code 3 frame count byte"))?;
    let vbr = count_byte & 0x80 != 0;
    let has_padding = count_byte & 0x40 != 0;
    let count = count_byte & 0x3f;
    if count == 0 {
        return Err(OpusError::Malformed("code 3 frame count is zero"));
    }
    if count > MAX_FRAMES_PER_PACKET {
        return Err(OpusError::TooManyFrames { count });
    }

    let mut pos = 2;
    let mut padding: usize = 0;
    let mut padding_added: usize = 0;
    if has_padding {
        // RFC 6716, Section 3.2.5: 0..=254 add that many bytes; 255 adds 254
        // bytes and requires another length byte.  The length bytes themselves
        // count towards the total added bytes P.
        let mut length_bytes = 0;
        loop {
            let byte = *data
                .get(pos)
                .ok_or(OpusError::Truncated("Opus padding length"))?;
            pos += 1;
            length_bytes += 1;
            padding = padding
                .checked_add(if byte == 255 { 254 } else { usize::from(byte) })
                .ok_or(OpusError::PaddingTooLarge {
                    padding: usize::MAX,
                    available: data.len().saturating_sub(2),
                })?;
            if byte != 255 {
                break;
            }
        }
        let added = padding
            .checked_add(length_bytes)
            .ok_or(OpusError::PaddingTooLarge {
                padding: usize::MAX,
                available: data.len().saturating_sub(2),
            })?;
        if added > data.len() - 2 {
            return Err(OpusError::PaddingTooLarge {
                padding: added,
                available: data.len() - 2,
            });
        }
        padding_added = added;
    }

    // `padding` is bounded by `data.len() - 2`, so this cannot underflow.
    let end = data.len() - padding;
    if end < pos {
        return Err(OpusError::PaddingTooLarge {
            padding: padding_added,
            available: data.len() - 2,
        });
    }

    let count = usize::from(count);
    let sizes = if vbr {
        // The first M-1 frames carry explicit lengths; the last frame consumes
        // every remaining byte before the padding (RFC 6716, [R7]).
        let mut sizes = Vec::with_capacity(count);
        let mut explicit: usize = 0;
        for _ in 0..count - 1 {
            let bytes = data
                .get(pos..end)
                .ok_or(OpusError::Truncated("VBR frame length"))?;
            let (size, used) = read_frame_length(bytes)?;
            pos += used;
            explicit = explicit
                .checked_add(size)
                .ok_or(OpusError::Malformed("VBR frame lengths overflow"))?;
            sizes.push(size);
        }
        if explicit > end - pos {
            return Err(OpusError::Malformed("VBR frame lengths exceed the payload"));
        }
        sizes.push(end - pos - explicit);
        sizes
    } else {
        // Every frame has identical size R/M (RFC 6716, [R6]).
        let remaining = end - pos;
        if !remaining.is_multiple_of(count) {
            return Err(OpusError::Malformed(
                "CBR payload is not a multiple of the frame count",
            ));
        }
        vec![remaining / count; count]
    };

    Ok((pos, sizes, padding))
}

/// Reads a one- or two-byte frame length, returning the length and the number
/// of bytes consumed (RFC 6716, Section 3.2.1).
fn read_frame_length(bytes: &[u8]) -> Result<(usize, usize), OpusError> {
    let first = *bytes.first().ok_or(OpusError::Truncated("frame length"))?;
    if first < 252 {
        Ok((usize::from(first), 1))
    } else {
        let second = *bytes
            .get(1)
            .ok_or(OpusError::Truncated("two-byte frame length"))?;
        Ok((usize::from(second) * 4 + usize::from(first), 2))
    }
}

/// Maps a TOC configuration number to its mode, bandwidth, and frame duration
/// (RFC 6716, Section 3.1, Table 2).
fn config_parameters(config: u8) -> (OpusMode, OpusBandwidth, u32) {
    /// Frame durations of the four SILK configurations in a bandwidth group.
    const SILK_DURATIONS_US: [u32; 4] = [10_000, 20_000, 40_000, 60_000];
    /// Frame durations of the four CELT configurations in a bandwidth group.
    const CELT_DURATIONS_US: [u32; 4] = [2_500, 5_000, 10_000, 20_000];

    match config {
        0..=11 => {
            let bandwidth = match config / 4 {
                0 => OpusBandwidth::Narrowband,
                1 => OpusBandwidth::Mediumband,
                _ => OpusBandwidth::Wideband,
            };
            let index = usize::from(config % 4);
            (OpusMode::Silk, bandwidth, SILK_DURATIONS_US[index])
        }
        12..=15 => {
            let bandwidth = if config < 14 {
                OpusBandwidth::SuperWideband
            } else {
                OpusBandwidth::Fullband
            };
            let duration = if config % 2 == 0 { 10_000 } else { 20_000 };
            (OpusMode::Hybrid, bandwidth, duration)
        }
        _ => {
            let bandwidth = match (config - 16) / 4 {
                0 => OpusBandwidth::Narrowband,
                1 => OpusBandwidth::Wideband,
                2 => OpusBandwidth::SuperWideband,
                _ => OpusBandwidth::Fullband,
            };
            let index = usize::from((config - 16) % 4);
            (OpusMode::Celt, bandwidth, CELT_DURATIONS_US[index])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_table_covers_all_32_entries() {
        for config in 0..32u8 {
            let (mode, bandwidth, duration) = config_parameters(config);
            match config {
                0..=11 => {
                    assert_eq!(mode, OpusMode::Silk, "config {config}");
                    let expected = match config / 4 {
                        0 => OpusBandwidth::Narrowband,
                        1 => OpusBandwidth::Mediumband,
                        _ => OpusBandwidth::Wideband,
                    };
                    assert_eq!(bandwidth, expected, "config {config}");
                    assert_eq!(
                        duration,
                        [10_000, 20_000, 40_000, 60_000][usize::from(config % 4)],
                        "config {config}"
                    );
                }
                12..=15 => {
                    assert_eq!(mode, OpusMode::Hybrid, "config {config}");
                    assert_eq!(
                        bandwidth,
                        if config < 14 {
                            OpusBandwidth::SuperWideband
                        } else {
                            OpusBandwidth::Fullband
                        },
                        "config {config}"
                    );
                    assert_eq!(
                        duration,
                        if config % 2 == 0 { 10_000 } else { 20_000 },
                        "config {config}"
                    );
                }
                _ => {
                    assert_eq!(mode, OpusMode::Celt, "config {config}");
                    let expected = match (config - 16) / 4 {
                        0 => OpusBandwidth::Narrowband,
                        1 => OpusBandwidth::Wideband,
                        2 => OpusBandwidth::SuperWideband,
                        _ => OpusBandwidth::Fullband,
                    };
                    assert_eq!(bandwidth, expected, "config {config}");
                    assert_eq!(
                        duration,
                        [2_500, 5_000, 10_000, 20_000][usize::from((config - 16) % 4)],
                        "config {config}"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_packet_is_rejected() {
        assert!(matches!(
            OpusPacket::parse(&[]),
            Err(OpusError::EmptyPacket)
        ));
    }

    #[test]
    fn code_0_exposes_one_borrowed_frame() {
        let data = [0x00, 0xaa, 0xbb, 0xcc];
        let packet = OpusPacket::parse(&data).unwrap();
        assert_eq!(packet.info().frame_count, 1);
        assert_eq!(packet.info().frame_sizes, vec![3]);
        assert_eq!(packet.frames().collect::<Vec<_>>(), vec![&data[1..]]);
    }

    #[test]
    fn code_3_padding_counts_trailing_bytes_only() {
        // TOC, frame count byte (padding, M = 2), padding length 3,
        // two one-byte frames, three padding bytes.
        let data = [0x03, 0x42, 0x03, 0xaa, 0xbb, 0x00, 0x00, 0x00];
        let packet = OpusPacket::parse(&data).unwrap();
        assert_eq!(packet.info().padding, 3);
        assert_eq!(packet.info().frame_sizes, vec![1, 1]);
        assert_eq!(
            packet.frames().collect::<Vec<_>>(),
            vec![&data[3..4], &data[4..5]]
        );
    }
}
