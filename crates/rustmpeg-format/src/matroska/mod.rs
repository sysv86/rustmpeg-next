// SPDX-License-Identifier: Apache-2.0
//! Matroska v4 and WebM demuxer and muxer (IETF RFC 9559).
//!
//! The demuxer reads an in-memory EBML document, validates the document type,
//! exposes the tracks declared in `Tracks`, and yields zero-copy packets from
//! `Cluster`/`SimpleBlock`/`BlockGroup` elements. Packet timestamps are
//! presentation timestamps: the track's `CodecDelay` is subtracted from the
//! stored block timestamp, as required by RFC 9559 §5.1.4.1.25.
//!
//! Supported codecs are the ones in the project's rolling window (Opus, FLAC,
//! LC3, LPCM, VVC, AV1, AV2, APV); other codec IDs are preserved as strings so
//! callers can decide what to do with them. [`MatroskaWriter`] produces a
//! document from the same [`Track`] and [`Packet`] types.

mod block;
mod write;

use std::collections::VecDeque;

use bytes::Bytes;

use crate::ebml::{
    ElementHeader, FormatError, read_float, read_sinteger, read_string, read_uinteger,
};

use self::block::{Block, parse_block};

pub use self::write::{DocType, MatroskaWriter, MuxerConfig};

/// Largest number of tracks accepted in one segment.
const MAX_TRACKS: usize = 64;
/// Largest number of cue points accepted in one segment.
const MAX_CUES: usize = 1_000_000;

/// Matroska `Segment` and top-level element IDs.
mod ids {
    pub const EBML: u64 = 0x1a45_dfa3;
    pub const EBML_MAX_ID_LENGTH: u64 = 0x42f2;
    pub const EBML_MAX_SIZE_LENGTH: u64 = 0x42f3;
    pub const DOC_TYPE: u64 = 0x4282;
    pub const DOC_TYPE_VERSION: u64 = 0x4287;
    pub const DOC_TYPE_READ_VERSION: u64 = 0x4285;
    pub const SEGMENT: u64 = 0x1853_8067;
    pub const SEEK_HEAD: u64 = 0x114d_9b74;
    pub const INFO: u64 = 0x1549_a966;
    pub const TRACKS: u64 = 0x1654_ae6b;
    pub const CLUSTER: u64 = 0x1f43_b675;
    pub const CUES: u64 = 0x1c53_bb6b;
    pub const ATTACHMENTS: u64 = 0x1941_a469;
    pub const CHAPTERS: u64 = 0x1043_a770;
    pub const TAGS: u64 = 0x1254_c367;
    pub const VOID: u64 = 0xec;
    pub const CRC32: u64 = 0xbf;

    pub const TIMESTAMP_SCALE: u64 = 0x002a_d7b1;
    pub const DURATION: u64 = 0x4489;
    pub const TITLE: u64 = 0x7ba9;

    pub const TRACK_ENTRY: u64 = 0xae;
    pub const TRACK_NUMBER: u64 = 0xd7;
    pub const TRACK_UID: u64 = 0x73c5;
    pub const TRACK_TYPE: u64 = 0x83;
    pub const FLAG_LACING: u64 = 0x9c;
    pub const CODEC_ID: u64 = 0x86;
    pub const CODEC_PRIVATE: u64 = 0x63a2;
    pub const DEFAULT_DURATION: u64 = 0x0023_e383;
    pub const NAME: u64 = 0x536e;
    pub const LANGUAGE: u64 = 0x0022_b59c;
    pub const LANGUAGE_BCP47: u64 = 0x0022_b59d;
    pub const CODEC_DELAY: u64 = 0x56aa;
    pub const SEEK_PRE_ROLL: u64 = 0x56bb;
    pub const AUDIO: u64 = 0xe1;
    pub const VIDEO: u64 = 0xe0;

    pub const SAMPLING_FREQUENCY: u64 = 0xb5;
    pub const OUTPUT_SAMPLING_FREQUENCY: u64 = 0x78b5;
    pub const CHANNELS: u64 = 0x9f;
    pub const BIT_DEPTH: u64 = 0x6264;
    pub const PIXEL_WIDTH: u64 = 0xb0;
    pub const PIXEL_HEIGHT: u64 = 0xba;

    pub const CLUSTER_TIMESTAMP: u64 = 0xe7;
    pub const SIMPLE_BLOCK: u64 = 0xa3;
    pub const BLOCK_GROUP: u64 = 0xa0;
    pub const BLOCK: u64 = 0xa1;
    pub const BLOCK_DURATION: u64 = 0x9b;
    pub const REFERENCE_BLOCK: u64 = 0xfb;
    pub const DISCARD_PADDING: u64 = 0x75a2;

    pub const CUE_POINT: u64 = 0xbb;
    pub const CUE_TIME: u64 = 0xb3;
    pub const CUE_TRACK_POSITIONS: u64 = 0xb7;
    pub const CUE_TRACK: u64 = 0xf7;
    pub const CUE_CLUSTER_POSITION: u64 = 0xf1;
}

/// Track kind declared by `TrackType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    /// Video track (`TrackType` 1).
    Video,
    /// Audio track (`TrackType` 2).
    Audio,
    /// Subtitle track (`TrackType` 0x11).
    Subtitle,
    /// Any other registered track type.
    Other(u64),
}

impl TrackType {
    fn from_code(code: u64) -> Self {
        match code {
            1 => Self::Video,
            2 => Self::Audio,
            0x11 => Self::Subtitle,
            other => Self::Other(other),
        }
    }
}

/// Codec classification derived from `CodecID`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecId {
    /// `A_OPUS`.
    Opus,
    /// `A_FLAC`.
    Flac,
    /// `A_LC3`.
    Lc3,
    /// Little-endian integer LPCM (`A_PCM/INT/LIT`).
    PcmInt,
    /// IEEE floating-point LPCM (`A_PCM/FLOAT/IEEE`).
    PcmFloat,
    /// `V_MPEGI/ISO/VVC`.
    Vvc,
    /// `V_AV1`.
    Av1,
    /// `V_AV2`.
    Av2,
    /// `V_MPEGI/ISO/APV`.
    Apv,
    /// Any other codec; the raw string is kept in [`Track::codec_id`].
    Other,
}

impl CodecId {
    fn from_str(codec_id: &str) -> Self {
        match codec_id {
            "A_OPUS" => Self::Opus,
            "A_FLAC" => Self::Flac,
            "A_LC3" => Self::Lc3,
            "A_PCM/INT/LIT" => Self::PcmInt,
            "A_PCM/FLOAT/IEEE" => Self::PcmFloat,
            "V_MPEGI/ISO/VVC" => Self::Vvc,
            "V_AV1" => Self::Av1,
            "V_AV2" => Self::Av2,
            "V_MPEGI/ISO/APV" => Self::Apv,
            _ => Self::Other,
        }
    }
}

/// Audio track settings from the `Audio` element.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioTrack {
    /// Sampling frequency in hertz (default 8000 when omitted).
    pub sampling_frequency: f64,
    /// Output sampling frequency when the track is upsampled.
    pub output_sampling_frequency: Option<f64>,
    /// Channel count.
    pub channels: u8,
    /// Coded bit depth when the codec stores it in the container.
    pub bit_depth: Option<u8>,
}

/// Video track settings from the `Video` element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoTrack {
    /// Coded width in pixels.
    pub width: u32,
    /// Coded height in pixels.
    pub height: u32,
}

/// One track declared in the segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    /// Track number used by blocks.
    pub number: u64,
    /// Track UID.
    pub uid: u64,
    /// Track kind.
    pub track_type: TrackType,
    /// Raw `CodecID` string.
    pub codec_id: String,
    /// Codec classification.
    pub codec: CodecId,
    /// Codec initialization data (`CodecPrivate`), zero-copy.
    pub codec_private: Bytes,
    /// Whether blocks of this track may use lacing.
    pub flag_lacing: bool,
    /// Nominal frame duration in nanoseconds (`DefaultDuration`).
    pub default_duration_ns: Option<u64>,
    /// Decoder delay in nanoseconds (`CodecDelay`).
    pub codec_delay_ns: u64,
    /// Data to decode after a discontinuity before output is valid.
    pub seek_pre_roll_ns: u64,
    /// Track language from `LanguageBCP47` or `Language`.
    pub language: Option<String>,
    /// Informal track name.
    pub name: Option<String>,
    /// Audio settings for audio tracks.
    pub audio: Option<AudioTrack>,
    /// Video settings for video tracks.
    pub video: Option<VideoTrack>,
}

/// One compressed frame from a block, with its presentation timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct Packet {
    /// Track number.
    pub track: u64,
    /// Presentation timestamp in nanoseconds, after subtracting `CodecDelay`.
    ///
    /// The value is signed because a codec delay can place the first packet
    /// before the segment start (for example Opus pre-skip).
    pub timestamp_ns: i64,
    /// Frame duration in nanoseconds when known.
    pub duration_ns: Option<u64>,
    /// Whether the frame is a random access point.
    pub keyframe: bool,
    /// Whether the frame may be dropped during playback.
    pub discardable: bool,
    /// Whether the frame should be decoded but not displayed.
    pub invisible: bool,
    /// Samples to trim from the decoded frame (`DiscardPadding`).
    pub discard_padding_ns: Option<i64>,
    /// Zero-copy frame data.
    pub data: Bytes,
}

/// A cue point used for seeking.
#[derive(Debug, Clone, Copy)]
struct Cue {
    /// Block timestamp in Segment Ticks.
    time_ticks: u64,
    /// Track the cue belongs to.
    track: u64,
    /// Cluster position relative to the start of the segment payload.
    cluster_position: u64,
}

/// Matroska cluster being read.
#[derive(Debug, Clone, Copy)]
struct ClusterState {
    /// Absolute end offset, or `None` for an unknown-size cluster.
    end: Option<usize>,
}

/// Streaming Matroska/WebM demuxer over an in-memory EBML document.
///
/// ```no_run
/// use bytes::Bytes;
/// use rustmpeg_format::MatroskaReader;
///
/// let data = Bytes::from(std::fs::read("input.webm").unwrap());
/// let mut reader = MatroskaReader::new(data).unwrap();
/// for track in reader.tracks() {
///     println!("track {}: {:?} {:?}", track.number, track.track_type, track.codec);
/// }
/// while let Some(packet) = reader.next_packet().unwrap() {
///     println!("{} bytes at {} ns", packet.data.len(), packet.timestamp_ns);
/// }
/// ```
#[derive(Debug)]
pub struct MatroskaReader {
    data: Bytes,
    doc_type: String,
    doc_type_version: u64,
    timestamp_scale_ns: u64,
    duration_ticks: Option<f64>,
    title: Option<String>,
    segment_data_start: usize,
    segment_end: usize,
    tracks: Vec<Track>,
    cues: Vec<Cue>,
    position: usize,
    cluster: Option<ClusterState>,
    cluster_timestamp: u64,
    pending: VecDeque<Packet>,
}

impl MatroskaReader {
    /// Parses the document header, segment metadata, tracks, and cue index.
    ///
    /// Clusters are read lazily by [`next_packet`](Self::next_packet), so this
    /// constructor only scans the top-level element structure.
    pub fn new(data: Bytes) -> Result<Self, FormatError> {
        let mut reader = Self {
            data,
            doc_type: String::new(),
            doc_type_version: 0,
            timestamp_scale_ns: 1_000_000,
            duration_ticks: None,
            title: None,
            segment_data_start: 0,
            segment_end: 0,
            tracks: Vec::new(),
            cues: Vec::new(),
            position: 0,
            cluster: None,
            cluster_timestamp: 0,
            pending: VecDeque::new(),
        };
        reader.parse_ebml_header()?;
        reader.parse_segment()?;
        Ok(reader)
    }

    /// EBML document type (`matroska` or `webm`).
    #[must_use]
    pub fn doc_type(&self) -> &str {
        &self.doc_type
    }

    /// EBML `DocTypeVersion` of the document.
    #[must_use]
    pub fn doc_type_version(&self) -> u64 {
        self.doc_type_version
    }

    /// Segment timestamp scale in nanoseconds.
    #[must_use]
    pub fn timestamp_scale_ns(&self) -> u64 {
        self.timestamp_scale_ns
    }

    /// Segment duration in nanoseconds, when declared in `Info`.
    #[must_use]
    pub fn duration_ns(&self) -> Option<u64> {
        let ticks = self.duration_ticks?;
        let ns = ticks * self.timestamp_scale_ns as f64;
        (ns.is_finite() && ns >= 0.0 && ns <= u64::MAX as f64).then_some(ns as u64)
    }

    /// Segment title, when declared in `Info`.
    #[must_use]
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// All tracks declared in `Tracks`.
    #[must_use]
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Looks up a track by number.
    #[must_use]
    pub fn track(&self, number: u64) -> Option<&Track> {
        self.tracks.iter().find(|track| track.number == number)
    }

    /// Returns whether the file carries a `Cues` index for [`seek_to`](Self::seek_to).
    #[must_use]
    pub fn is_seekable(&self) -> bool {
        !self.cues.is_empty()
    }

    /// Returns the next frame in storage order, or `None` at the end.
    ///
    /// Laced blocks yield one packet per frame. Packets of different tracks
    /// are returned in storage order, which is typically cluster-interleaved.
    /// Laced frames are separated by the track's `DefaultDuration`; when the
    /// track declares none, the frames share the block timestamp.
    pub fn next_packet(&mut self) -> Result<Option<Packet>, FormatError> {
        if let Some(packet) = self.pending.pop_front() {
            return Ok(Some(packet));
        }
        loop {
            if self.position >= self.segment_end {
                return Ok(None);
            }
            if let Some(ClusterState { end: Some(end), .. }) = self.cluster
                && self.position >= end
            {
                self.cluster = None;
                continue;
            }

            let start = self.position;
            let mut cursor = start;
            let header = ElementHeader::read(&self.data, &mut cursor)?;

            if self.cluster.is_some() {
                if is_top_level(header.id) {
                    // Unknown-size cluster ends where the next top-level
                    // element begins; leave the position untouched.
                    self.cluster = None;
                    continue;
                }
                if let Some(ClusterState { end: Some(end), .. }) = self.cluster
                    && header.data_start >= end
                {
                    self.cluster = None;
                    continue;
                }
                let limit = match self.cluster {
                    Some(ClusterState { end: Some(end), .. }) => end,
                    _ => self.segment_end,
                };
                match header.id {
                    ids::CLUSTER_TIMESTAMP => {
                        let payload = self.payload_within(&header, limit)?;
                        self.cluster_timestamp = read_uinteger(&payload)?;
                        self.position = header.data_end()?;
                    }
                    ids::SIMPLE_BLOCK => {
                        let payload = self.payload_within(&header, limit)?;
                        let block = parse_block(&payload, true)?;
                        self.position = header.data_end()?;
                        self.queue_block(&block, None, None)?;
                        if let Some(packet) = self.pending.pop_front() {
                            return Ok(Some(packet));
                        }
                    }
                    ids::BLOCK_GROUP => {
                        let (block, duration, discard_padding) =
                            self.parse_block_group(&header, limit)?;
                        self.position = header.data_end()?;
                        self.queue_block(&block, duration, discard_padding)?;
                        if let Some(packet) = self.pending.pop_front() {
                            return Ok(Some(packet));
                        }
                    }
                    _ => {
                        self.position = self.skip(&header)?;
                    }
                }
                if let Some(ClusterState { end: Some(end), .. }) = self.cluster
                    && self.position > end
                {
                    return Err(FormatError::Structure("cluster overruns its declared size"));
                }
                continue;
            }

            match header.id {
                ids::CLUSTER => {
                    let end = match header.size {
                        Some(_) => {
                            let end = header.data_end()?;
                            if end > self.segment_end {
                                return Err(FormatError::Structure("cluster exceeds the segment"));
                            }
                            Some(end)
                        }
                        None => None,
                    };
                    self.cluster = Some(ClusterState { end });
                    self.cluster_timestamp = 0;
                    self.position = header.data_start;
                }
                ids::INFO => {
                    let payload = self.payload_within(&header, self.segment_end)?;
                    self.parse_info(&payload)?;
                    self.position = header.data_end()?;
                }
                ids::TRACKS => {
                    let payload = self.payload_within(&header, self.segment_end)?;
                    self.parse_tracks(&payload)?;
                    self.position = header.data_end()?;
                }
                ids::CUES => {
                    let payload = self.payload_within(&header, self.segment_end)?;
                    self.parse_cues(&payload)?;
                    self.position = header.data_end()?;
                }
                _ => {
                    self.position = self.skip(&header)?;
                }
            }
        }
    }

    /// Positions the reader at the cluster that precedes `timestamp_ns`.
    ///
    /// `timestamp_ns` is a presentation timestamp (as in
    /// [`Packet::timestamp_ns`]). The function uses the `Cues` index and fails
    /// with [`FormatError::NoSeekIndex`] when the file has none. Reading
    /// resumes at the cluster start; the caller should skip packets whose
    /// timestamp is below the target and should honor each track's
    /// `seek_pre_roll_ns`.
    pub fn seek_to(&mut self, timestamp_ns: i64) -> Result<(), FormatError> {
        let scale = i64::try_from(self.timestamp_scale_ns)
            .map_err(|_| FormatError::Limit("timestamp scale"))?;
        let mut best: Option<(i64, usize)> = None;
        for cue in &self.cues {
            let delay = self
                .track(cue.track)
                .map_or(0, |track| track.codec_delay_ns as i64);
            let ticks =
                i64::try_from(cue.time_ticks).map_err(|_| FormatError::Limit("cue time"))?;
            let presentation = ticks
                .checked_mul(scale)
                .and_then(|value| value.checked_sub(delay))
                .ok_or(FormatError::Limit("cue time"))?;
            if presentation > timestamp_ns {
                continue;
            }
            let position = usize::try_from(cue.cluster_position)
                .ok()
                .and_then(|offset| self.segment_data_start.checked_add(offset))
                .filter(|position| *position < self.segment_end)
                .ok_or(FormatError::Structure("cue points outside the segment"))?;
            if best.is_none_or(|(time, _)| presentation > time) {
                best = Some((presentation, position));
            }
        }
        let (_, position) = best.ok_or(FormatError::NoSeekIndex)?;
        self.position = position;
        self.cluster = None;
        self.cluster_timestamp = 0;
        self.pending.clear();
        Ok(())
    }

    /// Parses the EBML document header and locates the first segment.
    fn parse_ebml_header(&mut self) -> Result<(), FormatError> {
        let mut cursor = 0usize;
        let header = ElementHeader::read(&self.data, &mut cursor)?;
        if header.id != ids::EBML {
            return Err(FormatError::InvalidElement);
        }
        let payload_end = header.data_end()?;
        let mut max_id_length = 4u64;
        let mut max_size_length = 8u64;
        let mut doc_type_version = 1u64;
        let mut read_version = 1u64;
        let mut children = Children::new(&self.data, header.data_start, payload_end)?;
        while let Some(child) = children.next()? {
            let value = children.payload(&child)?;
            match child.id {
                ids::DOC_TYPE => self.doc_type = read_string(&value)?.to_owned(),
                ids::DOC_TYPE_VERSION => doc_type_version = read_uinteger(&value)?,
                ids::DOC_TYPE_READ_VERSION => read_version = read_uinteger(&value)?,
                ids::EBML_MAX_ID_LENGTH => max_id_length = read_uinteger(&value)?,
                ids::EBML_MAX_SIZE_LENGTH => max_size_length = read_uinteger(&value)?,
                _ => {}
            }
        }
        match self.doc_type.as_str() {
            "matroska" | "webm" => {}
            _ => return Err(FormatError::UnsupportedDocType),
        }
        if max_id_length > 4 || max_size_length > 8 {
            return Err(FormatError::Limit("EBML ID or size length"));
        }
        if read_version > 4 || doc_type_version > 4 {
            return Err(FormatError::UnsupportedDocType);
        }
        self.doc_type_version = doc_type_version;

        let mut cursor = header.data_end()?;
        loop {
            if cursor >= self.data.len() {
                return Err(FormatError::Structure("segment is missing"));
            }
            let header = ElementHeader::read(&self.data, &mut cursor)?;
            match header.id {
                ids::SEGMENT => {
                    self.segment_data_start = header.data_start;
                    self.segment_end = match header.size {
                        Some(_) => header.data_end()?,
                        None => self.data.len(),
                    };
                    self.position = self.segment_data_start;
                    return Ok(());
                }
                ids::VOID | ids::CRC32 => {
                    cursor = header.data_end()?;
                }
                _ => return Err(FormatError::Structure("segment is missing")),
            }
        }
    }

    /// Scans top-level segment elements for metadata, tracks, and cues.
    fn parse_segment(&mut self) -> Result<(), FormatError> {
        let mut position = self.segment_data_start;
        let mut first_cluster = None;
        while position < self.segment_end {
            let mut cursor = position;
            let header = ElementHeader::read(&self.data, &mut cursor)?;
            if header.data_start > self.segment_end {
                return Err(FormatError::Structure("element exceeds the segment"));
            }
            let end = header
                .size
                .map(|_| header.data_end())
                .transpose()?
                .unwrap_or(self.segment_end);
            if end > self.segment_end {
                return Err(FormatError::Structure("element exceeds the segment"));
            }
            match header.id {
                ids::INFO => {
                    let payload = self.payload_within(&header, self.segment_end)?;
                    self.parse_info(&payload)?;
                    position = end;
                }
                ids::TRACKS => {
                    let payload = self.payload_within(&header, self.segment_end)?;
                    self.parse_tracks(&payload)?;
                    position = end;
                }
                ids::CUES => {
                    let payload = self.payload_within(&header, self.segment_end)?;
                    self.parse_cues(&payload)?;
                    position = end;
                }
                ids::CLUSTER => {
                    if first_cluster.is_none() {
                        first_cluster = Some(position);
                    }
                    position = match header.size {
                        Some(_) => header.data_end()?,
                        None => skip_unknown_cluster(&self.data, cursor, self.segment_end)?,
                    };
                }
                _ => {
                    position = match header.size {
                        Some(_) => header.data_end()?,
                        None => return Err(FormatError::Structure("unknown-size element")),
                    };
                }
            }
        }
        self.position = first_cluster.unwrap_or(self.segment_end);
        self.cluster = None;
        Ok(())
    }

    /// Advances past an element.
    fn skip(&mut self, header: &ElementHeader) -> Result<usize, FormatError> {
        header.data_end()
    }

    /// Returns an element payload, requiring it to stay inside `limit`.
    fn payload_within(&self, header: &ElementHeader, limit: usize) -> Result<Bytes, FormatError> {
        let end = header.data_end()?;
        if end > limit {
            return Err(FormatError::Structure("element exceeds its parent"));
        }
        Ok(self.data.slice(header.data_start..end))
    }

    fn parse_info(&mut self, data: &Bytes) -> Result<(), FormatError> {
        let mut children = Children::new(data, 0, data.len())?;
        while let Some(child) = children.next()? {
            let value = children.payload(&child)?;
            match child.id {
                ids::TIMESTAMP_SCALE => {
                    let scale = read_uinteger(&value)?;
                    if scale == 0 {
                        return Err(FormatError::Structure("timestamp scale is zero"));
                    }
                    self.timestamp_scale_ns = scale;
                }
                ids::DURATION => {
                    let duration = read_float(&value)?;
                    if !duration.is_finite() || duration < 0.0 {
                        return Err(FormatError::Structure("duration is invalid"));
                    }
                    self.duration_ticks = Some(duration);
                }
                ids::TITLE => self.title = Some(read_string(&value)?.to_owned()),
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_tracks(&mut self, data: &Bytes) -> Result<(), FormatError> {
        let mut children = Children::new(data, 0, data.len())?;
        while let Some(child) = children.next()? {
            if child.id != ids::TRACK_ENTRY {
                continue;
            }
            let entry = children.payload(&child)?;
            let track = parse_track_entry(&entry)?;
            if self.tracks.iter().any(|other| other.number == track.number) {
                return Err(FormatError::Structure("duplicate track number"));
            }
            if self.tracks.len() >= MAX_TRACKS {
                return Err(FormatError::Limit("track count"));
            }
            self.tracks.push(track);
        }
        Ok(())
    }

    fn parse_cues(&mut self, data: &Bytes) -> Result<(), FormatError> {
        let mut children = Children::new(data, 0, data.len())?;
        while let Some(child) = children.next()? {
            if child.id != ids::CUE_POINT {
                continue;
            }
            let point = children.payload(&child)?;
            let mut time = None;
            let mut positions = Vec::new();
            let mut inner = Children::new(&point, 0, point.len())?;
            while let Some(element) = inner.next()? {
                let value = inner.payload(&element)?;
                match element.id {
                    ids::CUE_TIME => time = Some(read_uinteger(&value)?),
                    ids::CUE_TRACK_POSITIONS => {
                        positions.push(parse_cue_positions(&value)?);
                    }
                    _ => {}
                }
            }
            let Some(time) = time else {
                return Err(FormatError::Structure("cue point without a time"));
            };
            for (track, cluster_position) in positions {
                if self.cues.len() >= MAX_CUES {
                    return Err(FormatError::Limit("cue count"));
                }
                self.cues.push(Cue {
                    time_ticks: time,
                    track,
                    cluster_position,
                });
            }
        }
        Ok(())
    }

    /// Parses a `BlockGroup` and returns its Block plus group metadata.
    fn parse_block_group(
        &self,
        header: &ElementHeader,
        limit: usize,
    ) -> Result<(Block, Option<u64>, Option<i64>), FormatError> {
        let end = header.data_end()?;
        if end > limit {
            return Err(FormatError::Structure("block group exceeds the cluster"));
        }
        let mut children = Children::new(&self.data, header.data_start, end)?;
        let mut block = None;
        let mut duration = None;
        let mut discard_padding = None;
        let mut has_reference = false;
        while let Some(child) = children.next()? {
            let value = children.payload(&child)?;
            match child.id {
                ids::BLOCK => {
                    if block.is_some() {
                        return Err(FormatError::Structure("block group has two blocks"));
                    }
                    block = Some(parse_block(&value, false)?);
                }
                ids::BLOCK_DURATION => duration = Some(read_uinteger(&value)?),
                ids::REFERENCE_BLOCK => {
                    has_reference = true;
                }
                ids::DISCARD_PADDING => discard_padding = Some(read_sinteger(&value)?),
                _ => {}
            }
        }
        let mut block = block.ok_or(FormatError::Structure("block group without a block"))?;
        block.keyframe = !has_reference;
        let duration_ns = match duration {
            Some(ticks) => Some(
                ticks
                    .checked_mul(self.timestamp_scale_ns)
                    .ok_or(FormatError::Limit("block duration"))?,
            ),
            None => None,
        };
        Ok((block, duration_ns, discard_padding))
    }

    /// Expands a block into packets and queues them for delivery.
    fn queue_block(
        &mut self,
        block: &Block,
        block_duration_ns: Option<u64>,
        discard_padding_ns: Option<i64>,
    ) -> Result<(), FormatError> {
        let track = self
            .tracks
            .iter()
            .find(|track| track.number == block.track)
            .ok_or(FormatError::UnknownTrack(block.track))?;
        if block.frames.len() > 1 && !track.flag_lacing {
            return Err(FormatError::Structure(
                "lacing used on a track that forbids it",
            ));
        }

        let scale = i64::try_from(self.timestamp_scale_ns)
            .map_err(|_| FormatError::Limit("timestamp scale"))?;
        let cluster = i64::try_from(self.cluster_timestamp)
            .map_err(|_| FormatError::Limit("cluster timestamp"))?;
        let ticks = cluster
            .checked_add(i64::from(block.relative_ticks))
            .ok_or(FormatError::Limit("block timestamp"))?;
        let raw_ns = ticks
            .checked_mul(scale)
            .ok_or(FormatError::Limit("block timestamp"))?;
        let codec_delay =
            i64::try_from(track.codec_delay_ns).map_err(|_| FormatError::Limit("codec delay"))?;
        let base_ns = raw_ns
            .checked_sub(codec_delay)
            .ok_or(FormatError::Limit("block timestamp"))?;

        let frames = block.frames.len() as u64;
        let frame_duration_ns = match block_duration_ns {
            Some(duration) => Some(duration / frames),
            None => track.default_duration_ns,
        };
        let keyframe = block.keyframe;
        let discardable = block.discardable;
        let invisible = block.invisible;

        for (index, frame) in block.frames.iter().enumerate() {
            let offset = frame_duration_ns
                .map(|duration| {
                    duration
                        .checked_mul(index as u64)
                        .ok_or(FormatError::Limit("frame timestamp"))
                })
                .transpose()?
                .unwrap_or(0);
            let timestamp_ns = base_ns
                .checked_add(
                    i64::try_from(offset).map_err(|_| FormatError::Limit("frame timestamp"))?,
                )
                .ok_or(FormatError::Limit("frame timestamp"))?;
            // `DiscardPadding` describes the end of the whole block, so it
            // applies to the last laced frame only.
            let padding = (index + 1 == block.frames.len())
                .then_some(discard_padding_ns)
                .flatten();
            self.pending.push_back(Packet {
                track: block.track,
                timestamp_ns,
                duration_ns: frame_duration_ns,
                keyframe,
                discardable,
                invisible,
                discard_padding_ns: padding,
                data: frame.clone(),
            });
        }
        Ok(())
    }
}

/// Iterator over child elements of a master element.
struct Children<'a> {
    data: &'a Bytes,
    position: usize,
    end: usize,
}

impl<'a> Children<'a> {
    fn new(data: &'a Bytes, start: usize, end: usize) -> Result<Self, FormatError> {
        if start > end || end > data.len() {
            return Err(FormatError::Structure("child range is invalid"));
        }
        Ok(Self {
            data,
            position: start,
            end,
        })
    }

    fn next(&mut self) -> Result<Option<ElementHeader>, FormatError> {
        if self.position >= self.end {
            return Ok(None);
        }
        let header = ElementHeader::read(self.data, &mut self.position)?;
        if self.position > self.end {
            return Err(FormatError::Structure("element header exceeds its parent"));
        }
        let end = header
            .data_end()
            .map_err(|_| FormatError::Structure("unknown-size element inside a master"))?;
        if end > self.end {
            return Err(FormatError::Structure("element exceeds its parent"));
        }
        self.position = end;
        Ok(Some(header))
    }

    fn payload(&self, header: &ElementHeader) -> Result<Bytes, FormatError> {
        let end = header.data_end()?;
        if end > self.end {
            return Err(FormatError::Structure("element exceeds its parent"));
        }
        Ok(self.data.slice(header.data_start..end))
    }
}

/// Element IDs that may appear directly under `Segment`.
fn is_top_level(id: u64) -> bool {
    matches!(
        id,
        ids::SEEK_HEAD
            | ids::INFO
            | ids::TRACKS
            | ids::CLUSTER
            | ids::CUES
            | ids::ATTACHMENTS
            | ids::CHAPTERS
            | ids::TAGS
    )
}

/// Skips an unknown-size cluster by scanning for the next top-level element.
fn skip_unknown_cluster(
    data: &Bytes,
    mut position: usize,
    segment_end: usize,
) -> Result<usize, FormatError> {
    while position < segment_end {
        let start = position;
        let mut cursor = position;
        let header = ElementHeader::read(data, &mut cursor)?;
        if is_top_level(header.id) {
            return Ok(start);
        }
        if header.data_start >= segment_end {
            return Ok(segment_end);
        }
        position = match header.size {
            Some(_) => header.data_end()?,
            None => {
                return Err(FormatError::Structure(
                    "unknown-size element inside a cluster",
                ));
            }
        };
    }
    Ok(segment_end)
}

fn parse_track_entry(data: &Bytes) -> Result<Track, FormatError> {
    let mut number = None;
    let mut uid = 0u64;
    let mut track_type = None;
    let mut flag_lacing = true;
    let mut codec_id = None;
    let mut codec_private = Bytes::new();
    let mut default_duration_ns = None;
    let mut name = None;
    let mut language = None;
    let mut language_bcp47 = None;
    let mut codec_delay_ns = 0u64;
    let mut seek_pre_roll_ns = 0u64;
    let mut audio = None;
    let mut video = None;

    let mut children = Children::new(data, 0, data.len())?;
    while let Some(child) = children.next()? {
        let value = children.payload(&child)?;
        match child.id {
            ids::TRACK_NUMBER => number = Some(read_uinteger(&value)?),
            ids::TRACK_UID => uid = read_uinteger(&value)?,
            ids::TRACK_TYPE => track_type = Some(read_uinteger(&value)?),
            ids::FLAG_LACING => flag_lacing = read_uinteger(&value)? != 0,
            ids::CODEC_ID => codec_id = Some(read_string(&value)?.to_owned()),
            ids::CODEC_PRIVATE => codec_private = value,
            ids::DEFAULT_DURATION => default_duration_ns = Some(read_uinteger(&value)?),
            ids::NAME => name = Some(read_string(&value)?.to_owned()),
            ids::LANGUAGE => language = Some(read_string(&value)?.to_owned()),
            ids::LANGUAGE_BCP47 => language_bcp47 = Some(read_string(&value)?.to_owned()),
            ids::CODEC_DELAY => codec_delay_ns = read_uinteger(&value)?,
            ids::SEEK_PRE_ROLL => seek_pre_roll_ns = read_uinteger(&value)?,
            ids::AUDIO => audio = Some(parse_audio(&value)?),
            ids::VIDEO => video = Some(parse_video(&value)?),
            _ => {}
        }
    }

    let number = number.ok_or(FormatError::Structure("track without a number"))?;
    if number == 0 {
        return Err(FormatError::Structure("track number is zero"));
    }
    let track_type_code = track_type.ok_or(FormatError::Structure("track without a type"))?;
    let track_type = TrackType::from_code(track_type_code);
    let codec_id = codec_id.ok_or(FormatError::Structure("track without a codec ID"))?;
    if codec_id.is_empty() {
        return Err(FormatError::Structure("empty codec ID"));
    }
    match track_type {
        TrackType::Audio if audio.is_none() => {
            return Err(FormatError::Structure("audio track without settings"));
        }
        TrackType::Video if video.is_none() => {
            return Err(FormatError::Structure("video track without settings"));
        }
        _ => {}
    }
    let codec = CodecId::from_str(&codec_id);
    Ok(Track {
        number,
        uid,
        track_type,
        codec_id,
        codec,
        codec_private,
        flag_lacing,
        default_duration_ns,
        codec_delay_ns,
        seek_pre_roll_ns,
        language: language_bcp47.or(language),
        name,
        audio,
        video,
    })
}

fn parse_audio(data: &Bytes) -> Result<AudioTrack, FormatError> {
    let mut sampling_frequency = 8000.0f64;
    let mut output_sampling_frequency = None;
    let mut channels = 1u8;
    let mut bit_depth = None;
    let mut children = Children::new(data, 0, data.len())?;
    while let Some(child) = children.next()? {
        let value = children.payload(&child)?;
        match child.id {
            ids::SAMPLING_FREQUENCY => sampling_frequency = read_float(&value)?,
            ids::OUTPUT_SAMPLING_FREQUENCY => {
                output_sampling_frequency = Some(read_float(&value)?);
            }
            ids::CHANNELS => {
                let value = read_uinteger(&value)?;
                channels = u8::try_from(value).map_err(|_| FormatError::Limit("channel count"))?;
            }
            ids::BIT_DEPTH => {
                let value = read_uinteger(&value)?;
                bit_depth = Some(u8::try_from(value).map_err(|_| FormatError::Limit("bit depth"))?);
            }
            _ => {}
        }
    }
    if !sampling_frequency.is_finite() || sampling_frequency <= 0.0 {
        return Err(FormatError::Structure("invalid sampling frequency"));
    }
    if channels == 0 {
        return Err(FormatError::Structure("audio track has no channels"));
    }
    if let Some(depth) = bit_depth
        && !(1..=64).contains(&depth)
    {
        return Err(FormatError::Structure("invalid bit depth"));
    }
    Ok(AudioTrack {
        sampling_frequency,
        output_sampling_frequency,
        channels,
        bit_depth,
    })
}

fn parse_video(data: &Bytes) -> Result<VideoTrack, FormatError> {
    let mut width = None;
    let mut height = None;
    let mut children = Children::new(data, 0, data.len())?;
    while let Some(child) = children.next()? {
        let value = children.payload(&child)?;
        match child.id {
            ids::PIXEL_WIDTH => width = Some(read_uinteger(&value)?),
            ids::PIXEL_HEIGHT => height = Some(read_uinteger(&value)?),
            _ => {}
        }
    }
    let width = width.ok_or(FormatError::Structure("video track without a width"))?;
    let height = height.ok_or(FormatError::Structure("video track without a height"))?;
    let width = u32::try_from(width).map_err(|_| FormatError::Limit("video width"))?;
    let height = u32::try_from(height).map_err(|_| FormatError::Limit("video height"))?;
    if width == 0 || height == 0 {
        return Err(FormatError::Structure("video track has an empty frame"));
    }
    Ok(VideoTrack { width, height })
}

fn parse_cue_positions(data: &Bytes) -> Result<(u64, u64), FormatError> {
    let mut track = None;
    let mut cluster_position = None;
    let mut children = Children::new(data, 0, data.len())?;
    while let Some(child) = children.next()? {
        let value = children.payload(&child)?;
        match child.id {
            ids::CUE_TRACK => track = Some(read_uinteger(&value)?),
            ids::CUE_CLUSTER_POSITION => {
                cluster_position = Some(read_uinteger(&value)?);
            }
            _ => {}
        }
    }
    let track = track.ok_or(FormatError::Structure("cue without a track"))?;
    let cluster_position =
        cluster_position.ok_or(FormatError::Structure("cue without a cluster position"))?;
    Ok((track, cluster_position))
}
