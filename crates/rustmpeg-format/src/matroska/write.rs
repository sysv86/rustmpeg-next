// SPDX-License-Identifier: Apache-2.0
//! Matroska v4 and WebM muxer (IETF RFC 9559).
//!
//! The muxer writes a complete in-memory EBML document with a known-size
//! `Segment`, one `Cluster` per second (or per 1 MiB of blocks), `SimpleBlock`
//! and `BlockGroup` elements, and a `Cues` index. Packet timestamps are
//! presentation timestamps, as returned by [`MatroskaReader`](super::MatroskaReader);
//! the writer adds each track's `CodecDelay` back to obtain the stored block
//! timestamp.
//!
//! Codecs are limited to the project's rolling window. Lacing is not emitted:
//! every packet becomes one block frame, which is always valid even when the
//! track allows lacing.

use bytes::Bytes;

use super::{AudioTrack, CodecId, FormatError, Packet, Track, TrackType, VideoTrack};

/// EBML document type written by [`MatroskaWriter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocType {
    /// Matroska (`.mkv`, `.mka`).
    Matroska,
    /// WebM.
    Webm,
}

impl DocType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Matroska => "matroska",
            Self::Webm => "webm",
        }
    }
}

/// Configuration for [`MatroskaWriter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MuxerConfig {
    /// Document type written to the EBML header.
    pub doc_type: DocType,
    /// Timestamp scale in nanoseconds. The default is one millisecond, which is
    /// what common Matroska muxers use.
    pub timestamp_scale_ns: u64,
}

impl Default for MuxerConfig {
    fn default() -> Self {
        Self {
            doc_type: DocType::Matroska,
            timestamp_scale_ns: 1_000_000,
        }
    }
}

/// Cluster accumulated before it is serialized.
#[derive(Debug)]
struct PendingCluster {
    timestamp: u64,
    first_track: u64,
    blocks: Vec<u8>,
}

/// Matroska/WebM muxer that produces a complete document in memory.
///
/// ```no_run
/// use rustmpeg_format::matroska::{DocType, MatroskaWriter, MuxerConfig};
///
/// let mut writer = MatroskaWriter::new(MuxerConfig {
///     doc_type: DocType::Webm,
///     ..MuxerConfig::default()
/// })
/// .unwrap();
/// # let track = unimplemented!();
/// # let packets: Vec<rustmpeg_format::Packet> = Vec::new();
/// writer.add_track(track).unwrap();
/// for packet in packets {
///     writer.write_packet(packet).unwrap();
/// }
/// let file = writer.finish().unwrap();
/// std::fs::write("output.webm", file).unwrap();
/// ```
#[derive(Debug)]
pub struct MatroskaWriter {
    config: MuxerConfig,
    tracks: Vec<Track>,
    clusters: Vec<PendingCluster>,
    current: Option<PendingCluster>,
    last_ticks: Option<u64>,
    duration_ticks: f64,
}

impl MatroskaWriter {
    /// Creates an empty document writer, rejecting a zero timestamp scale.
    pub fn new(config: MuxerConfig) -> Result<Self, FormatError> {
        if config.timestamp_scale_ns == 0 {
            return Err(FormatError::Structure("timestamp scale is zero"));
        }
        Ok(Self {
            config,
            tracks: Vec::new(),
            clusters: Vec::new(),
            current: None,
            last_ticks: None,
            duration_ticks: 0.0,
        })
    }

    /// Declares a track. Tracks must be added before the first packet.
    ///
    /// The writer accepts the track description returned by the demuxer, so a
    /// file can be remuxed without rebuilding its metadata. Codec IDs outside
    /// the rolling window are rejected with
    /// [`FormatError::UnsupportedCodec`].
    pub fn add_track(&mut self, track: Track) -> Result<(), FormatError> {
        if !self.clusters.is_empty() || self.current.is_some() {
            return Err(FormatError::Structure(
                "tracks must be added before the first packet",
            ));
        }
        if self.tracks.iter().any(|other| other.number == track.number) {
            return Err(FormatError::Structure("duplicate track number"));
        }
        if self.tracks.len() >= super::MAX_TRACKS {
            return Err(FormatError::Limit("track count"));
        }
        if track.number == 0 {
            return Err(FormatError::Structure("track number is zero"));
        }
        if track.uid == 0 {
            return Err(FormatError::Structure("track UID is zero"));
        }
        let codec = CodecId::from_str(&track.codec_id);
        if codec == CodecId::Other || codec != track.codec {
            return Err(FormatError::UnsupportedCodec(track.codec_id.clone()));
        }
        let codec_kind = codec_kind(codec);
        if !codec_kind.matches(track.track_type) {
            return Err(FormatError::Structure(
                "track type does not match the codec ID",
            ));
        }
        match track.track_type {
            TrackType::Audio => {
                let audio = track
                    .audio
                    .as_ref()
                    .ok_or(FormatError::Structure("audio track without settings"))?;
                validate_audio(audio)?;
            }
            TrackType::Video => {
                let video = track
                    .video
                    .as_ref()
                    .ok_or(FormatError::Structure("video track without settings"))?;
                if video.width == 0 || video.height == 0 {
                    return Err(FormatError::Structure("video track has an empty frame"));
                }
            }
            _ => {
                return Err(FormatError::Structure(
                    "only audio and video tracks can be written",
                ));
            }
        }
        if track.default_duration_ns == Some(0) {
            return Err(FormatError::Structure("default duration is zero"));
        }
        self.tracks.push(track);
        Ok(())
    }

    /// Writes one complete frame. Packets must be added in non-decreasing
    /// presentation timestamp order.
    pub fn write_packet(&mut self, packet: Packet) -> Result<(), FormatError> {
        if self.tracks.is_empty() {
            return Err(FormatError::Structure("packet written before any track"));
        }
        let track = self
            .tracks
            .iter()
            .find(|track| track.number == packet.track)
            .ok_or(FormatError::UnknownTrack(packet.track))?;
        if packet.data.is_empty() {
            return Err(FormatError::Block("empty frame"));
        }
        let codec_delay =
            i64::try_from(track.codec_delay_ns).map_err(|_| FormatError::Limit("codec delay"))?;
        let raw_ns = packet
            .timestamp_ns
            .checked_add(codec_delay)
            .ok_or(FormatError::Limit("block timestamp"))?;
        if raw_ns < 0 {
            return Err(FormatError::Structure(
                "packet timestamp precedes the track's codec delay",
            ));
        }
        let scale = i64::try_from(self.config.timestamp_scale_ns)
            .map_err(|_| FormatError::Limit("timestamp scale"))?;
        let ticks =
            u64::try_from(raw_ns / scale).map_err(|_| FormatError::Limit("block timestamp"))?;
        if self.last_ticks.is_some_and(|last| ticks < last) {
            return Err(FormatError::Structure("packets are not in timestamp order"));
        }

        let duration_ticks = packet
            .duration_ns
            .map(|ns| ns.div_ceil(self.config.timestamp_scale_ns));
        let default_ticks = track
            .default_duration_ns
            .map(|ns| ns.div_ceil(self.config.timestamp_scale_ns));
        let need_duration = duration_ticks.is_some() && duration_ticks != default_ticks;
        let need_group = need_duration || packet.discard_padding_ns.is_some();

        let span_ticks = (1_000_000_000u64 / self.config.timestamp_scale_ns).max(1);
        let start_new = match &self.current {
            None => true,
            Some(cluster) => {
                let relative = ticks - cluster.timestamp;
                relative > i64::from(i16::MAX) as u64
                    || relative >= span_ticks
                    || cluster.blocks.len() + packet.data.len() + 16 > 1 << 20
            }
        };
        let relative = match &self.current {
            Some(cluster) if !start_new => i16::try_from(ticks - cluster.timestamp)
                .map_err(|_| FormatError::Limit("block timestamp"))?,
            _ => 0,
        };
        let block_duration = if need_duration { duration_ticks } else { None };
        let block = encode_block(&packet, relative, need_group, block_duration)?;
        if start_new {
            self.flush_cluster();
            self.current = Some(PendingCluster {
                timestamp: ticks,
                first_track: packet.track,
                blocks: Vec::new(),
            });
        }
        if let Some(cluster) = &mut self.current {
            cluster.blocks.extend_from_slice(&block);
        }
        self.last_ticks = Some(ticks);

        let end = ticks as f64 + duration_ticks.map_or(0.0, |ticks| ticks as f64);
        self.duration_ticks = self.duration_ticks.max(end);
        Ok(())
    }

    /// Serializes the document and returns its bytes.
    pub fn finish(mut self) -> Result<Bytes, FormatError> {
        self.flush_cluster();
        if self.tracks.is_empty() {
            return Err(FormatError::Structure("segment has no tracks"));
        }

        let info = self.info_element()?;
        let tracks = self.tracks_element()?;
        let clusters: Vec<Vec<u8>> = self
            .clusters
            .iter()
            .map(cluster_element)
            .collect::<Result<_, _>>()?;

        let mut cues = Vec::new();
        if !self.clusters.is_empty() {
            // Cue positions depend on the size of `Cues` itself. Encoding the
            // position as a fixed eight-octet integer makes the element size
            // independent of the values, so two passes converge.
            cues = cues_element(&self.clusters, &clusters, 0)?;
            loop {
                let prefix = info
                    .len()
                    .checked_add(tracks.len())
                    .and_then(|length| length.checked_add(cues.len()))
                    .ok_or(FormatError::Limit("segment size"))?;
                let next = cues_element(&self.clusters, &clusters, prefix)?;
                if next.len() == cues.len() {
                    cues = next;
                    break;
                }
                cues = next;
            }
        }

        let segment_size = info
            .len()
            .checked_add(tracks.len())
            .and_then(|length| length.checked_add(cues.len()))
            .ok_or(FormatError::Limit("segment size"))?;
        let mut segment = Vec::with_capacity(segment_size);
        segment.extend_from_slice(&info);
        segment.extend_from_slice(&tracks);
        segment.extend_from_slice(&cues);
        for cluster in &clusters {
            segment.extend_from_slice(cluster);
        }

        let mut document = ebml_header(self.config.doc_type)?;
        document.extend(element(&IDS_SEGMENT, &segment)?);
        Ok(Bytes::from(document))
    }

    fn flush_cluster(&mut self) {
        if let Some(cluster) = self.current.take() {
            self.clusters.push(cluster);
        }
    }

    fn info_element(&self) -> Result<Vec<u8>, FormatError> {
        let mut info = Vec::new();
        info.extend(uint(&IDS_TIMESTAMP_SCALE, self.config.timestamp_scale_ns)?);
        if self.duration_ticks > 0.0 {
            info.extend(float(&IDS_DURATION, self.duration_ticks)?);
        }
        info.extend(string(&IDS_MUXING_APP, "rustmpeg-next")?);
        info.extend(string(&IDS_WRITING_APP, "rustmpeg-next")?);
        element(&IDS_INFO, &info)
    }

    fn tracks_element(&self) -> Result<Vec<u8>, FormatError> {
        let mut tracks = Vec::new();
        for track in &self.tracks {
            tracks.extend(track_element(track)?);
        }
        element(&IDS_TRACKS, &tracks)
    }
}

/// Whether a codec is audio or video, for track/type validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodecKind {
    Audio,
    Video,
}

impl CodecKind {
    fn matches(self, track_type: TrackType) -> bool {
        matches!(
            (self, track_type),
            (Self::Audio, TrackType::Audio) | (Self::Video, TrackType::Video)
        )
    }
}

fn codec_kind(codec: CodecId) -> CodecKind {
    match codec {
        CodecId::Opus | CodecId::Flac | CodecId::Lc3 | CodecId::PcmInt | CodecId::PcmFloat => {
            CodecKind::Audio
        }
        CodecId::Vvc | CodecId::Av1 | CodecId::Av2 | CodecId::Apv | CodecId::Other => {
            CodecKind::Video
        }
    }
}

fn validate_audio(audio: &AudioTrack) -> Result<(), FormatError> {
    if !audio.sampling_frequency.is_finite() || audio.sampling_frequency <= 0.0 {
        return Err(FormatError::Structure("invalid sampling frequency"));
    }
    if let Some(output) = audio.output_sampling_frequency
        && (!output.is_finite() || output <= 0.0)
    {
        return Err(FormatError::Structure("invalid output sampling frequency"));
    }
    if audio.channels == 0 {
        return Err(FormatError::Structure("audio track has no channels"));
    }
    if let Some(depth) = audio.bit_depth
        && !(1..=64).contains(&depth)
    {
        return Err(FormatError::Structure("invalid bit depth"));
    }
    Ok(())
}

fn track_element(track: &Track) -> Result<Vec<u8>, FormatError> {
    let mut entry = Vec::new();
    entry.extend(uint(&IDS_TRACK_NUMBER, track.number)?);
    entry.extend(uint(&IDS_TRACK_UID, track.uid)?);
    let type_code = match track.track_type {
        TrackType::Video => 1,
        TrackType::Audio => 2,
        TrackType::Subtitle => 0x11,
        TrackType::Other(code) => code,
    };
    entry.extend(uint(&IDS_TRACK_TYPE, type_code)?);
    entry.extend(uint(&IDS_FLAG_LACING, u64::from(track.flag_lacing))?);
    entry.extend(string(&IDS_CODEC_ID, &track.codec_id)?);
    if !track.codec_private.is_empty() {
        entry.extend(element(&IDS_CODEC_PRIVATE, &track.codec_private)?);
    }
    if track.codec_delay_ns != 0 {
        entry.extend(uint(&IDS_CODEC_DELAY, track.codec_delay_ns)?);
    }
    if track.seek_pre_roll_ns != 0 {
        entry.extend(uint(&IDS_SEEK_PRE_ROLL, track.seek_pre_roll_ns)?);
    }
    if let Some(duration) = track.default_duration_ns {
        entry.extend(uint(&IDS_DEFAULT_DURATION, duration)?);
    }
    if let Some(language) = &track.language {
        entry.extend(string(&IDS_LANGUAGE_BCP47, language)?);
    }
    if let Some(name) = &track.name {
        entry.extend(string(&IDS_NAME, name)?);
    }
    if let Some(audio) = &track.audio {
        entry.extend(audio_element(audio)?);
    }
    if let Some(video) = &track.video {
        entry.extend(video_element(video)?);
    }
    element(&IDS_TRACK_ENTRY, &entry)
}

fn audio_element(audio: &AudioTrack) -> Result<Vec<u8>, FormatError> {
    let mut settings = Vec::new();
    settings.extend(float(&IDS_SAMPLING_FREQUENCY, audio.sampling_frequency)?);
    if let Some(output) = audio.output_sampling_frequency {
        settings.extend(float(&IDS_OUTPUT_SAMPLING_FREQUENCY, output)?);
    }
    settings.extend(uint(&IDS_CHANNELS, u64::from(audio.channels))?);
    if let Some(depth) = audio.bit_depth {
        settings.extend(uint(&IDS_BIT_DEPTH, u64::from(depth))?);
    }
    element(&IDS_AUDIO, &settings)
}

fn video_element(video: &VideoTrack) -> Result<Vec<u8>, FormatError> {
    let mut settings = Vec::new();
    settings.extend(uint(&IDS_PIXEL_WIDTH, u64::from(video.width))?);
    settings.extend(uint(&IDS_PIXEL_HEIGHT, u64::from(video.height))?);
    element(&IDS_VIDEO, &settings)
}

/// Encodes a `SimpleBlock` or the `Block` inside a `BlockGroup`.
fn encode_block(
    packet: &Packet,
    relative_ticks: i16,
    group: bool,
    duration_ticks: Option<u64>,
) -> Result<Vec<u8>, FormatError> {
    let mut block = Vec::new();
    block.extend(track_vint(packet.track)?);
    block.extend_from_slice(&relative_ticks.to_be_bytes());
    let mut flags = 0u8;
    if !group && packet.keyframe {
        flags |= 0x80;
    }
    if packet.invisible {
        flags |= 0x08;
    }
    if packet.discardable {
        flags |= 0x01;
    }
    block.push(flags);
    block.extend_from_slice(&packet.data);

    if !group {
        return element(&IDS_SIMPLE_BLOCK, &block);
    }

    let mut group = Vec::new();
    group.extend(element(&IDS_BLOCK, &block)?);
    if let Some(duration) = duration_ticks {
        group.extend(uint(&IDS_BLOCK_DURATION, duration)?);
    }
    if !packet.keyframe {
        // The writer does not track cross-frame references, so it marks the
        // frame as non-keyframe with a reference to the preceding tick.
        group.extend(sint(&IDS_REFERENCE_BLOCK, -1)?);
    }
    if let Some(padding) = packet.discard_padding_ns {
        group.extend(sint(&IDS_DISCARD_PADDING, padding)?);
    }
    element(&IDS_BLOCK_GROUP, &group)
}

fn cluster_element(cluster: &PendingCluster) -> Result<Vec<u8>, FormatError> {
    let mut payload = Vec::with_capacity(cluster.blocks.len() + 16);
    payload.extend(uint(&IDS_CLUSTER_TIMESTAMP, cluster.timestamp)?);
    payload.extend_from_slice(&cluster.blocks);
    element(&IDS_CLUSTER, &payload)
}

fn cues_element(
    clusters: &[PendingCluster],
    encoded: &[Vec<u8>],
    prefix: usize,
) -> Result<Vec<u8>, FormatError> {
    let mut cues = Vec::new();
    let mut offset = 0usize;
    for (cluster, bytes) in clusters.iter().zip(encoded) {
        let position = prefix
            .checked_add(offset)
            .ok_or(FormatError::Limit("cue position"))?;
        let mut positions = Vec::new();
        positions.extend(uint(&IDS_CUE_TRACK, cluster.first_track)?);
        positions.extend(fixed_uint(
            &IDS_CUE_CLUSTER_POSITION,
            u64::try_from(position).map_err(|_| FormatError::Limit("cue position"))?,
            8,
        )?);
        let mut point = Vec::new();
        point.extend(uint(&IDS_CUE_TIME, cluster.timestamp)?);
        point.extend(element(&IDS_CUE_TRACK_POSITIONS, &positions)?);
        cues.extend(element(&IDS_CUE_POINT, &point)?);
        offset = offset
            .checked_add(bytes.len())
            .ok_or(FormatError::Limit("cluster offset"))?;
    }
    element(&IDS_CUES, &cues)
}

fn ebml_header(doc_type: DocType) -> Result<Vec<u8>, FormatError> {
    let mut header = Vec::new();
    header.extend(uint(&IDS_EBML_VERSION, 1)?);
    header.extend(uint(&IDS_EBML_READ_VERSION, 1)?);
    header.extend(uint(&IDS_EBML_MAX_ID_LENGTH, 4)?);
    header.extend(uint(&IDS_EBML_MAX_SIZE_LENGTH, 8)?);
    header.extend(string(&IDS_DOC_TYPE, doc_type.as_str())?);
    header.extend(uint(&IDS_DOC_TYPE_VERSION, 4)?);
    header.extend(uint(&IDS_DOC_TYPE_READ_VERSION, 2)?);
    element(&IDS_EBML, &header)
}

// Element IDs, in EBML octet order.
const IDS_EBML: [u8; 4] = [0x1a, 0x45, 0xdf, 0xa3];
const IDS_EBML_VERSION: [u8; 2] = [0x42, 0x86];
const IDS_EBML_READ_VERSION: [u8; 2] = [0x42, 0xf7];
const IDS_EBML_MAX_ID_LENGTH: [u8; 2] = [0x42, 0xf2];
const IDS_EBML_MAX_SIZE_LENGTH: [u8; 2] = [0x42, 0xf3];
const IDS_DOC_TYPE: [u8; 2] = [0x42, 0x82];
const IDS_DOC_TYPE_VERSION: [u8; 2] = [0x42, 0x87];
const IDS_DOC_TYPE_READ_VERSION: [u8; 2] = [0x42, 0x85];
const IDS_SEGMENT: [u8; 4] = [0x18, 0x53, 0x80, 0x67];
const IDS_INFO: [u8; 4] = [0x15, 0x49, 0xa9, 0x66];
const IDS_TIMESTAMP_SCALE: [u8; 3] = [0x2a, 0xd7, 0xb1];
const IDS_DURATION: [u8; 2] = [0x44, 0x89];
const IDS_MUXING_APP: [u8; 2] = [0x4d, 0x80];
const IDS_WRITING_APP: [u8; 2] = [0x57, 0x41];
const IDS_TRACKS: [u8; 4] = [0x16, 0x54, 0xae, 0x6b];
const IDS_TRACK_ENTRY: [u8; 1] = [0xae];
const IDS_TRACK_NUMBER: [u8; 1] = [0xd7];
const IDS_TRACK_UID: [u8; 2] = [0x73, 0xc5];
const IDS_TRACK_TYPE: [u8; 1] = [0x83];
const IDS_FLAG_LACING: [u8; 1] = [0x9c];
const IDS_CODEC_ID: [u8; 1] = [0x86];
const IDS_CODEC_PRIVATE: [u8; 2] = [0x63, 0xa2];
const IDS_DEFAULT_DURATION: [u8; 3] = [0x23, 0xe3, 0x83];
const IDS_LANGUAGE_BCP47: [u8; 3] = [0x22, 0xb5, 0x9d];
const IDS_NAME: [u8; 2] = [0x53, 0x6e];
const IDS_CODEC_DELAY: [u8; 2] = [0x56, 0xaa];
const IDS_SEEK_PRE_ROLL: [u8; 2] = [0x56, 0xbb];
const IDS_AUDIO: [u8; 1] = [0xe1];
const IDS_VIDEO: [u8; 1] = [0xe0];
const IDS_SAMPLING_FREQUENCY: [u8; 1] = [0xb5];
const IDS_OUTPUT_SAMPLING_FREQUENCY: [u8; 2] = [0x78, 0xb5];
const IDS_CHANNELS: [u8; 1] = [0x9f];
const IDS_BIT_DEPTH: [u8; 2] = [0x62, 0x64];
const IDS_PIXEL_WIDTH: [u8; 1] = [0xb0];
const IDS_PIXEL_HEIGHT: [u8; 1] = [0xba];
const IDS_CLUSTER: [u8; 4] = [0x1f, 0x43, 0xb6, 0x75];
const IDS_CLUSTER_TIMESTAMP: [u8; 1] = [0xe7];
const IDS_SIMPLE_BLOCK: [u8; 1] = [0xa3];
const IDS_BLOCK_GROUP: [u8; 1] = [0xa0];
const IDS_BLOCK: [u8; 1] = [0xa1];
const IDS_BLOCK_DURATION: [u8; 1] = [0x9b];
const IDS_REFERENCE_BLOCK: [u8; 1] = [0xfb];
const IDS_DISCARD_PADDING: [u8; 2] = [0x75, 0xa2];
const IDS_CUES: [u8; 4] = [0x1c, 0x53, 0xbb, 0x6b];
const IDS_CUE_POINT: [u8; 1] = [0xbb];
const IDS_CUE_TIME: [u8; 1] = [0xb3];
const IDS_CUE_TRACK_POSITIONS: [u8; 1] = [0xb7];
const IDS_CUE_TRACK: [u8; 1] = [0xf7];
const IDS_CUE_CLUSTER_POSITION: [u8; 1] = [0xf1];

fn element(id: &[u8], payload: &[u8]) -> Result<Vec<u8>, FormatError> {
    let mut out = Vec::with_capacity(id.len().saturating_add(9).saturating_add(payload.len()));
    out.extend_from_slice(id);
    write_size(&mut out, payload.len())?;
    out.extend_from_slice(payload);
    Ok(out)
}

fn write_size(out: &mut Vec<u8>, size: usize) -> Result<(), FormatError> {
    let value = u64::try_from(size).map_err(|_| FormatError::Limit("element size"))?;
    for length in 1..=8u32 {
        let max = (1u64 << (7 * length)) - 2;
        if value <= max {
            let marker = 1u8 << (8 - length);
            let mask = (1u8 << (8 - length)) - 1;
            out.push(marker | (value >> (8 * (length - 1))) as u8 & mask);
            for index in 1..length {
                out.push((value >> (8 * (length - 1 - index))) as u8);
            }
            return Ok(());
        }
    }
    Err(FormatError::Limit("element size"))
}

fn track_vint(track: u64) -> Result<Vec<u8>, FormatError> {
    if track == 0 {
        return Err(FormatError::Structure("track number is zero"));
    }
    let mut out = Vec::new();
    write_size_value(&mut out, track)?;
    Ok(out)
}

/// Writes an EBML variable-length integer with its marker bits.
fn write_size_value(out: &mut Vec<u8>, value: u64) -> Result<(), FormatError> {
    for length in 1..=8u32 {
        let max = (1u64 << (7 * length)) - 1;
        if value <= max {
            let marker = 1u8 << (8 - length);
            let mask = (1u8 << (8 - length)) - 1;
            out.push(marker | (value >> (8 * (length - 1))) as u8 & mask);
            for index in 1..length {
                out.push((value >> (8 * (length - 1 - index))) as u8);
            }
            return Ok(());
        }
    }
    Err(FormatError::Limit("variable-length integer"))
}

fn uint_bytes(value: u64) -> Vec<u8> {
    if value == 0 {
        return vec![0];
    }
    let bytes = value.to_be_bytes();
    let start = bytes.iter().position(|&byte| byte != 0).unwrap_or(7);
    bytes[start..].to_vec()
}

fn uint(id: &[u8], value: u64) -> Result<Vec<u8>, FormatError> {
    element(id, &uint_bytes(value))
}

fn fixed_uint(id: &[u8], value: u64, width: usize) -> Result<Vec<u8>, FormatError> {
    let width = u8::try_from(width).map_err(|_| FormatError::Limit("integer width"))?;
    let bytes = value.to_be_bytes();
    if width > 8
        || bytes[..8 - usize::from(width)]
            .iter()
            .any(|&byte| byte != 0)
    {
        return Err(FormatError::Limit("integer width"));
    }
    element(id, &bytes[8 - usize::from(width)..])
}

fn sint_bytes(value: i64) -> Vec<u8> {
    let bytes = value.to_be_bytes();
    let mut first = 0;
    while first < 7
        && ((bytes[first] == 0 && bytes[first + 1] & 0x80 == 0)
            || (bytes[first] == 0xff && bytes[first + 1] & 0x80 != 0))
    {
        first += 1;
    }
    bytes[first..].to_vec()
}

fn sint(id: &[u8], value: i64) -> Result<Vec<u8>, FormatError> {
    element(id, &sint_bytes(value))
}

fn float(id: &[u8], value: f64) -> Result<Vec<u8>, FormatError> {
    element(id, &value.to_bits().to_be_bytes())
}

fn string(id: &[u8], value: &str) -> Result<Vec<u8>, FormatError> {
    element(id, value.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ebml::read_vint;

    #[test]
    fn element_sizes_round_trip_across_vint_boundaries() {
        for size in [0usize, 1, 126, 127, 1_000, 16_381, 16_382, 1_000_000] {
            let mut encoded = Vec::new();
            write_size(&mut encoded, size).unwrap();
            let mut position = 0;
            let (value, unknown) = read_vint(&encoded, &mut position, false).unwrap();
            assert_eq!(value, size as u64);
            assert_eq!(unknown, None);
            assert_eq!(position, encoded.len());
        }
    }

    #[test]
    fn track_vints_round_trip() {
        for track in [1u64, 126, 127, 128, 16_383, 1 << 20] {
            let encoded = track_vint(track).unwrap();
            let mut position = 0;
            // All-ones has no special meaning for block track numbers; only
            // element sizes reserve it for "unknown".
            let (value, _) = read_vint(&encoded, &mut position, false).unwrap();
            assert_eq!(value, track);
            assert_eq!(position, encoded.len());
        }
        assert!(track_vint(0).is_err());
    }

    #[test]
    fn unsigned_and_signed_integers_use_minimal_width() {
        assert_eq!(uint_bytes(0), [0]);
        assert_eq!(uint_bytes(1), [1]);
        assert_eq!(uint_bytes(0xffff), [0xff, 0xff]);
        assert_eq!(sint_bytes(0), [0]);
        assert_eq!(sint_bytes(-1), [0xff]);
        assert_eq!(sint_bytes(127), [0x7f]);
        assert_eq!(sint_bytes(128), [0x00, 0x80]);
        assert_eq!(sint_bytes(-128), [0x80]);
        assert_eq!(sint_bytes(13_500_000), [0x00, 0xcd, 0xfe, 0x60]);
    }

    #[test]
    fn fixed_uint_rejects_values_that_do_not_fit() {
        assert!(fixed_uint(&IDS_CUE_CLUSTER_POSITION, 1, 8).is_ok());
        assert!(fixed_uint(&IDS_CUE_CLUSTER_POSITION, 1, 1).is_ok());
        assert!(fixed_uint(&IDS_CUE_CLUSTER_POSITION, 256, 1).is_err());
        assert!(fixed_uint(&IDS_CUE_CLUSTER_POSITION, 0, 9).is_err());
    }
}
