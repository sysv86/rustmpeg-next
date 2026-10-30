// SPDX-License-Identifier: Apache-2.0
//! ISO base media file format (ISOBMFF) and CMAF.
//!
//! The module implements ISO/IEC 14496-12 box parsing and writing, the sample
//! tables of a non-fragmented `.mp4` file, and fragmented CMAF initialization
//! and media segments.  Sample payloads are zero-copy [`Bytes`] slices of the
//! input document; timestamps are integer ticks in each track's timescale.
//!
//! The container layer is codec-agnostic: a [`SampleEntry`] carries the
//! four-character sample entry code (`vvc1`, `apv1`, `Opus`, `fLaC`, ...), the
//! decoder configuration box type (`vvcC`, `apvC`, `dOps`, `dfLa`, ...), and
//! the configuration payload.  Codec crates own the interpretation of that
//! payload.  Video tracks also carry colour metadata (`colr`) and a pixel
//! aspect ratio (`pasp`) when present.
//!
//! Limits are deliberate denial-of-service guards: at most
//! [`MAX_TRACKS`] tracks, [`MAX_SAMPLES`] samples, and one million boxes are
//! accepted from one document.

mod boxes;
mod cmaf;
mod reader;
mod sample_entry;
mod writer;

pub use cmaf::{CmafConfig, CmafWriter};
pub use reader::IsobmffReader;
pub use writer::{Mp4Config, Mp4Writer};

use bytes::Bytes;
use rustmpeg_core::ColorDescription;

/// Maximum number of tracks accepted from one file.
pub const MAX_TRACKS: usize = 128;

/// Maximum number of samples indexed from one file.
pub const MAX_SAMPLES: usize = 16_000_000;

/// Kind of media carried by a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    /// Video track (`vide` handler).
    Video,
    /// Audio track (`soun` handler).
    Audio,
    /// Subtitle or text track (`subt`, `sbtl`, `text`).
    Subtitle,
    /// Any other handler type, preserved as a four-character code.
    Other([u8; 4]),
}

impl TrackKind {
    pub(crate) fn from_handler(kind: [u8; 4]) -> Self {
        match &kind {
            b"vide" => Self::Video,
            b"soun" => Self::Audio,
            b"subt" | b"sbtl" | b"text" => Self::Subtitle,
            _ => Self::Other(kind),
        }
    }

    pub(crate) fn handler(self) -> [u8; 4] {
        match self {
            Self::Video => *b"vide",
            Self::Audio => *b"soun",
            Self::Subtitle => *b"subt",
            Self::Other(kind) => kind,
        }
    }
}

/// Video geometry and optional colour metadata from `VisualSampleEntry`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoTrack {
    /// Coded width in pixels.
    pub width: u16,
    /// Coded height in pixels.
    pub height: u16,
    /// Colour description from `colr` (`nclx`), when present.
    pub color: Option<ColorDescription>,
    /// Pixel aspect ratio from `pasp`, when present.
    pub pixel_aspect_ratio: Option<(u32, u32)>,
}

/// Audio layout from `AudioSampleEntry`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioTrack {
    /// Channel count.
    pub channels: u16,
    /// Coded sample size in bits.
    pub sample_size: u16,
    /// Output sample rate in hertz.
    pub sample_rate: f64,
}

/// Decoder configuration attached to a sample entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleEntry {
    /// Four-character sample entry code (`vvc1`, `apv1`, `Opus`, ...).
    pub format: [u8; 4],
    /// Four-character decoder configuration box type (`vvcC`, `dOps`, ...).
    pub config_box: Option<[u8; 4]>,
    /// Configuration box payload, zero-copy when read from a file.
    pub config: Bytes,
}

impl SampleEntry {
    /// Creates a sample entry with an optional decoder configuration.
    #[must_use]
    pub fn new(format: [u8; 4], config_box: Option<[u8; 4]>, config: Bytes) -> Self {
        Self {
            format,
            config_box,
            config,
        }
    }
}

/// One track declared by `moov`.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    /// Track identifier used by samples.
    pub id: u32,
    /// Track kind from the media handler.
    pub kind: TrackKind,
    /// Media timescale in ticks per second.
    pub timescale: u32,
    /// Declared track duration in timescale ticks (zero when unknown).
    pub duration: u64,
    /// ISO-639-2/T language code (`und` when unspecified).
    pub language: String,
    /// Sample entry and decoder configuration.
    pub sample_entry: SampleEntry,
    /// Video layout for video tracks.
    pub video: Option<VideoTrack>,
    /// Audio layout for audio tracks.
    pub audio: Option<AudioTrack>,
}

/// One compressed sample in decode order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    /// Track identifier.
    pub track_id: u32,
    /// Decode timestamp in the track timescale.
    pub dts: i64,
    /// Presentation timestamp in the track timescale.
    pub pts: i64,
    /// Sample duration in the track timescale.
    pub duration: u32,
    /// Whether the sample is a random access point.
    pub keyframe: bool,
    /// Zero-copy sample payload.
    pub data: Bytes,
}

impl Sample {
    /// Creates a sample.
    #[must_use]
    pub fn new(
        track_id: u32,
        dts: i64,
        pts: i64,
        duration: u32,
        keyframe: bool,
        data: Bytes,
    ) -> Self {
        Self {
            track_id,
            dts,
            pts,
            duration,
            keyframe,
            data,
        }
    }
}

pub(crate) fn track_by_id(tracks: &[Track], id: u32) -> Option<&Track> {
    tracks.iter().find(|track| track.id == id)
}

pub(crate) fn validate_track(track: &Track) -> Result<(), crate::FormatError> {
    if track.id == 0 {
        return Err(crate::FormatError::Isobmff("track identifier is zero"));
    }
    if track.timescale == 0 {
        return Err(crate::FormatError::Isobmff("track timescale is zero"));
    }
    match track.kind {
        TrackKind::Video if track.video.is_none() => {
            return Err(crate::FormatError::Isobmff(
                "video track has no visual sample entry",
            ));
        }
        TrackKind::Audio if track.audio.is_none() => {
            return Err(crate::FormatError::Isobmff(
                "audio track has no audio sample entry",
            ));
        }
        _ => {}
    }
    Ok(())
}

/// Converts a core colour description to an `nclx` `colr` payload.
pub(crate) fn color_to_nclx(color: &ColorDescription) -> [u8; 11] {
    let primaries = match color.primaries {
        rustmpeg_core::ColorPrimaries::Bt709 => 1_u16,
        rustmpeg_core::ColorPrimaries::Bt2020 => 9,
        rustmpeg_core::ColorPrimaries::DisplayP3 => 12,
        rustmpeg_core::ColorPrimaries::Unknown => 2,
    };
    let transfer = match color.transfer {
        rustmpeg_core::TransferFunction::Sdr => 1_u16,
        rustmpeg_core::TransferFunction::Pq => 16,
        rustmpeg_core::TransferFunction::Hlg => 18,
        rustmpeg_core::TransferFunction::Linear => 8,
        rustmpeg_core::TransferFunction::Srgb => 13,
        rustmpeg_core::TransferFunction::Unknown => 2,
    };
    let matrix = match color.matrix {
        rustmpeg_core::MatrixCoefficients::Bt709 => 1_u16,
        rustmpeg_core::MatrixCoefficients::Bt2020NonConstant => 9,
        rustmpeg_core::MatrixCoefficients::Identity => 0,
        rustmpeg_core::MatrixCoefficients::Unknown => 2,
    };
    let full_range = match color.range {
        rustmpeg_core::ColorRange::Limited => 0_u8,
        rustmpeg_core::ColorRange::Full => 0x80,
    };
    let mut payload = [0_u8; 11];
    payload[..4].copy_from_slice(b"nclx");
    payload[4..6].copy_from_slice(&primaries.to_be_bytes());
    payload[6..8].copy_from_slice(&transfer.to_be_bytes());
    payload[8..10].copy_from_slice(&matrix.to_be_bytes());
    payload[10] = full_range;
    payload
}

/// Parses an `nclx` `colr` payload.
pub(crate) fn nclx_to_color(payload: &[u8]) -> Option<ColorDescription> {
    if payload.len() < 11 || &payload[..4] != b"nclx" {
        return None;
    }
    use rustmpeg_core::{ColorPrimaries, ColorRange, MatrixCoefficients, TransferFunction};
    let primaries = u16::from_be_bytes([payload[4], payload[5]]);
    let transfer = u16::from_be_bytes([payload[6], payload[7]]);
    let matrix = u16::from_be_bytes([payload[8], payload[9]]);
    Some(ColorDescription {
        range: if payload[10] & 0x80 != 0 {
            ColorRange::Full
        } else {
            ColorRange::Limited
        },
        primaries: match primaries {
            1 => ColorPrimaries::Bt709,
            9 => ColorPrimaries::Bt2020,
            12 => ColorPrimaries::DisplayP3,
            _ => ColorPrimaries::Unknown,
        },
        transfer: match transfer {
            1 => TransferFunction::Sdr,
            16 => TransferFunction::Pq,
            18 => TransferFunction::Hlg,
            8 => TransferFunction::Linear,
            13 => TransferFunction::Srgb,
            _ => TransferFunction::Unknown,
        },
        matrix: match matrix {
            1 => MatrixCoefficients::Bt709,
            9 => MatrixCoefficients::Bt2020NonConstant,
            0 => MatrixCoefficients::Identity,
            _ => MatrixCoefficients::Unknown,
        },
        chroma_location: rustmpeg_core::ChromaLocation::Left,
    })
}
