// SPDX-License-Identifier: Apache-2.0
//! Non-fragmented ISO base media file writer (`.mp4`).
//!
//! Samples are appended to an `mdat` payload and indexed with the classic
//! sample tables.  The `moov` box is written before `mdat`; chunk offsets are
//! computed after its final size is known.  `stco` is used when every offset
//! fits in 32 bits, otherwise `co64`.

use bytes::Bytes;

use super::boxes::{box_bytes, encode_language, full_box_bytes, push_u16, push_u32, push_u64};
use super::sample_entry::build_stsd;
use super::{MAX_SAMPLES, MAX_TRACKS, Sample, Track, TrackKind, validate_track};
use crate::FormatError;

/// Configuration for [`Mp4Writer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mp4Config {
    /// Major brand written to `ftyp`.
    pub major_brand: [u8; 4],
    /// Minor version written to `ftyp`.
    pub minor_version: u32,
    /// Additional compatible brands.
    pub compatible_brands: Vec<[u8; 4]>,
    /// Movie timescale in ticks per second.
    pub movie_timescale: u32,
}

impl Default for Mp4Config {
    fn default() -> Self {
        Self {
            major_brand: *b"isom",
            minor_version: 0x200,
            compatible_brands: vec![*b"isom", *b"iso2", *b"mp41"],
            movie_timescale: 1000,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Record {
    track_id: u32,
    offset: u64,
    size: u32,
    dts: i64,
    pts: i64,
    duration: u32,
    keyframe: bool,
}

#[derive(Debug, Clone, Copy)]
struct Chunk {
    payload_offset: u64,
    samples: u32,
}

#[derive(Debug)]
struct TrackLayout<'a> {
    track: &'a Track,
    records: Vec<usize>,
    chunks: Vec<Chunk>,
    media_duration: u64,
    presentation_duration: u64,
}

/// Writes a complete non-fragmented `.mp4` document in memory.
///
/// ```no_run
/// use bytes::Bytes;
/// use rustmpeg_format::isobmff::{Mp4Writer, Mp4Config, Sample, SampleEntry, Track, TrackKind, VideoTrack};
///
/// let mut writer = Mp4Writer::new(Mp4Config::default());
/// writer.add_track(Track {
///     id: 1,
///     kind: TrackKind::Video,
///     timescale: 90_000,
///     duration: 0,
///     language: "und".into(),
///     sample_entry: SampleEntry::new(*b"vvc1", Some(*b"vvcC"), Bytes::new()),
///     video: Some(VideoTrack { width: 1920, height: 1080, color: None, pixel_aspect_ratio: None }),
///     audio: None,
/// }).unwrap();
/// writer.write_sample(Sample::new(1, 0, 0, 3000, true, Bytes::from_static(b"frame"))).unwrap();
/// let file = writer.finish().unwrap();
/// std::fs::write("output.mp4", file).unwrap();
/// ```
#[derive(Debug)]
pub struct Mp4Writer {
    config: Mp4Config,
    tracks: Vec<Track>,
    mdat: Vec<u8>,
    records: Vec<Record>,
}

impl Mp4Writer {
    /// Creates an empty writer.
    #[must_use]
    pub fn new(config: Mp4Config) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            mdat: Vec::new(),
            records: Vec::new(),
        }
    }

    /// Declares a track. Tracks must be added before the first sample.
    ///
    /// # Errors
    ///
    /// Rejects duplicate identifiers, zero timescales, unsupported layouts,
    /// and track counts above [`MAX_TRACKS`].
    pub fn add_track(&mut self, track: Track) -> Result<(), FormatError> {
        if !self.records.is_empty() {
            return Err(FormatError::Isobmff(
                "tracks must be added before the first sample",
            ));
        }
        if self.tracks.len() >= MAX_TRACKS {
            return Err(FormatError::IsobmffLimit("track count"));
        }
        if self.tracks.iter().any(|other| other.id == track.id) {
            return Err(FormatError::Isobmff("duplicate track identifier"));
        }
        validate_track(&track)?;
        if !matches!(track.kind, TrackKind::Video | TrackKind::Audio) {
            return Err(FormatError::Isobmff(
                "only audio and video tracks can be written",
            ));
        }
        self.tracks.push(track);
        Ok(())
    }

    /// Appends one sample.
    ///
    /// Samples of each track must be written in non-decreasing decode order.
    ///
    /// # Errors
    ///
    /// Rejects an unknown track, empty payloads, zero durations, negative or
    /// regressing decode timestamps, and more than [`MAX_SAMPLES`] samples.
    pub fn write_sample(&mut self, sample: Sample) -> Result<(), FormatError> {
        if self.records.len() >= MAX_SAMPLES {
            return Err(FormatError::IsobmffLimit("sample count"));
        }
        if super::track_by_id(&self.tracks, sample.track_id).is_none() {
            return Err(FormatError::Isobmff("sample references an unknown track"));
        }
        if sample.data.is_empty() {
            return Err(FormatError::Isobmff("sample is empty"));
        }
        if sample.duration == 0 {
            return Err(FormatError::Isobmff("sample duration is zero"));
        }
        if sample.dts < 0 {
            return Err(FormatError::Isobmff("negative decode timestamp"));
        }
        if sample.pts - sample.dts < i64::from(i32::MIN)
            || sample.pts - sample.dts > i64::from(i32::MAX)
        {
            return Err(FormatError::Isobmff(
                "composition offset does not fit in 32 bits",
            ));
        }
        if let Some(previous) = self
            .records
            .iter()
            .rev()
            .find(|record| record.track_id == sample.track_id)
            && sample.dts < previous.dts
        {
            return Err(FormatError::Isobmff("samples are not in decode order"));
        }
        let size = u32::try_from(sample.data.len())
            .map_err(|_| FormatError::IsobmffLimit("sample size"))?;
        let offset = self.mdat.len() as u64;
        self.mdat.extend_from_slice(&sample.data);
        self.records.push(Record {
            track_id: sample.track_id,
            offset,
            size,
            dts: sample.dts,
            pts: sample.pts,
            duration: sample.duration,
            keyframe: sample.keyframe,
        });
        Ok(())
    }

    /// Serializes the document.
    ///
    /// # Errors
    ///
    /// Rejects a document without tracks and any size or table overflow.
    pub fn finish(self) -> Result<Bytes, FormatError> {
        if self.tracks.is_empty() {
            return Err(FormatError::Isobmff("movie has no tracks"));
        }
        if self.config.movie_timescale == 0 {
            return Err(FormatError::Isobmff("movie timescale is zero"));
        }
        let layouts = self.layouts()?;
        let ftyp = ftyp_bytes(
            self.config.major_brand,
            self.config.minor_version,
            &self.config.compatible_brands,
        );
        let mdat_header = if self.mdat.len() as u64 + 8 <= u64::from(u32::MAX) {
            8_u64
        } else {
            16
        };

        let mut use_co64 = false;
        let (moov, _) = loop {
            let measure = build_moov(
                &layouts,
                &self.records,
                self.config.movie_timescale,
                0,
                use_co64,
            )?;
            let chunk_base = ftyp.len() as u64 + measure.len() as u64 + mdat_header;
            let moov = build_moov(
                &layouts,
                &self.records,
                self.config.movie_timescale,
                chunk_base,
                use_co64,
            )?;
            let fits = layouts.iter().all(|layout| {
                layout
                    .chunks
                    .iter()
                    .all(|chunk| chunk_base + chunk.payload_offset <= u64::from(u32::MAX))
            });
            if !use_co64 && !fits {
                use_co64 = true;
                continue;
            }
            break (moov, chunk_base);
        };

        let mut document = ftyp;
        document.extend_from_slice(&moov);
        document.extend(box_bytes(*b"mdat", &self.mdat));
        Ok(Bytes::from(document))
    }

    fn layouts(&self) -> Result<Vec<TrackLayout<'_>>, FormatError> {
        let mut layouts: Vec<TrackLayout<'_>> = self
            .tracks
            .iter()
            .map(|track| TrackLayout {
                track,
                records: Vec::new(),
                chunks: Vec::new(),
                media_duration: track.duration,
                presentation_duration: track.duration,
            })
            .collect();

        let mut previous_track = None;
        for (index, record) in self.records.iter().enumerate() {
            let layout = layouts
                .iter_mut()
                .find(|layout| layout.track.id == record.track_id)
                .ok_or(FormatError::Isobmff("sample references an unknown track"))?;
            layout.records.push(index);
            if previous_track == Some(record.track_id) {
                let chunk = layout
                    .chunks
                    .last_mut()
                    .ok_or(FormatError::Isobmff("chunk table is empty"))?;
                chunk.samples += 1;
            } else {
                layout.chunks.push(Chunk {
                    payload_offset: record.offset,
                    samples: 1,
                });
            }
            let media_end = u64::try_from(record.dts)
                .ok()
                .and_then(|dts| dts.checked_add(u64::from(record.duration)))
                .ok_or(FormatError::IsobmffLimit("track duration"))?;
            let presentation_end = u64::try_from(record.dts.max(record.pts))
                .ok()
                .and_then(|start| start.checked_add(u64::from(record.duration)))
                .ok_or(FormatError::IsobmffLimit("track duration"))?;
            layout.media_duration = layout.media_duration.max(media_end);
            layout.presentation_duration = layout.presentation_duration.max(presentation_end);
            previous_track = Some(record.track_id);
        }
        Ok(layouts)
    }
}

pub(crate) fn ftyp_bytes(major: [u8; 4], minor: u32, compatible: &[[u8; 4]]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8 + compatible.len() * 4);
    payload.extend_from_slice(&major);
    push_u32(&mut payload, minor);
    for brand in compatible {
        payload.extend_from_slice(brand);
    }
    box_bytes(*b"ftyp", &payload)
}

fn build_moov(
    layouts: &[TrackLayout<'_>],
    records: &[Record],
    movie_timescale: u32,
    chunk_base: u64,
    use_co64: bool,
) -> Result<Vec<u8>, FormatError> {
    let mut movie_duration = 0_u64;
    for layout in layouts {
        let duration = scale_duration(
            layout.presentation_duration,
            layout.track.timescale,
            movie_timescale,
        )?;
        movie_duration = movie_duration.max(duration);
    }
    let next_track = layouts
        .iter()
        .map(|layout| layout.track.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let mut body = Vec::new();
    body.extend_from_slice(&mvhd_bytes(movie_duration, movie_timescale, next_track)?);
    for layout in layouts {
        body.extend_from_slice(&trak(
            layout,
            records,
            movie_timescale,
            chunk_base,
            use_co64,
        )?);
    }
    Ok(box_bytes(*b"moov", &body))
}

pub(crate) fn mvhd_bytes(
    duration: u64,
    timescale: u32,
    next_track: u32,
) -> Result<Vec<u8>, FormatError> {
    let mut payload = Vec::new();
    if duration > u64::from(u32::MAX) {
        push_u64(&mut payload, 0);
        push_u64(&mut payload, 0);
        push_u32(&mut payload, timescale);
        push_u64(&mut payload, duration);
        push_u32(&mut payload, 0x0001_0000);
        push_u16(&mut payload, 0x0100);
        push_u16(&mut payload, 0);
        payload.extend_from_slice(&[0; 8]);
        payload.extend_from_slice(&matrix_bytes());
        payload.extend_from_slice(&[0; 24]);
        push_u32(&mut payload, next_track);
        Ok(full_box_bytes(*b"mvhd", 1, 0, &payload))
    } else {
        push_u32(&mut payload, 0);
        push_u32(&mut payload, 0);
        push_u32(&mut payload, timescale);
        push_u32(&mut payload, duration as u32);
        push_u32(&mut payload, 0x0001_0000);
        push_u16(&mut payload, 0x0100);
        push_u16(&mut payload, 0);
        payload.extend_from_slice(&[0; 8]);
        payload.extend_from_slice(&matrix_bytes());
        payload.extend_from_slice(&[0; 24]);
        push_u32(&mut payload, next_track);
        Ok(full_box_bytes(*b"mvhd", 0, 0, &payload))
    }
}

fn trak(
    layout: &TrackLayout<'_>,
    records: &[Record],
    movie_timescale: u32,
    chunk_base: u64,
    use_co64: bool,
) -> Result<Vec<u8>, FormatError> {
    let track = layout.track;
    let duration = scale_duration(
        layout.presentation_duration,
        track.timescale,
        movie_timescale,
    )?;
    let mut trak = Vec::new();
    trak.extend_from_slice(&tkhd_bytes(track, duration)?);
    trak.extend_from_slice(&mdia(layout, records, chunk_base, use_co64)?);
    Ok(box_bytes(*b"trak", &trak))
}

pub(crate) fn tkhd_bytes(track: &Track, duration: u64) -> Result<Vec<u8>, FormatError> {
    let width = track.video.map_or(0, |video| u32::from(video.width) << 16);
    let height = track.video.map_or(0, |video| u32::from(video.height) << 16);
    let volume = if track.kind == TrackKind::Audio {
        0x0100_u16
    } else {
        0
    };
    let mut payload = Vec::new();
    if duration > u64::from(u32::MAX) {
        push_u64(&mut payload, 0);
        push_u64(&mut payload, 0);
        push_u32(&mut payload, track.id);
        push_u32(&mut payload, 0);
        push_u64(&mut payload, duration);
        payload.extend_from_slice(&[0; 8]);
        push_u16(&mut payload, 0); // layer
        push_u16(&mut payload, 0); // alternate_group
        push_u16(&mut payload, volume);
        push_u16(&mut payload, 0);
        payload.extend_from_slice(&matrix_bytes());
        push_u32(&mut payload, width);
        push_u32(&mut payload, height);
        Ok(full_box_bytes(*b"tkhd", 1, 0x000003, &payload))
    } else {
        push_u32(&mut payload, 0);
        push_u32(&mut payload, 0);
        push_u32(&mut payload, track.id);
        push_u32(&mut payload, 0);
        push_u32(&mut payload, duration as u32);
        payload.extend_from_slice(&[0; 8]);
        push_u16(&mut payload, 0);
        push_u16(&mut payload, 0);
        push_u16(&mut payload, volume);
        push_u16(&mut payload, 0);
        payload.extend_from_slice(&matrix_bytes());
        push_u32(&mut payload, width);
        push_u32(&mut payload, height);
        Ok(full_box_bytes(*b"tkhd", 0, 0x000003, &payload))
    }
}

fn mdia(
    layout: &TrackLayout<'_>,
    records: &[Record],
    chunk_base: u64,
    use_co64: bool,
) -> Result<Vec<u8>, FormatError> {
    let track = layout.track;
    let mut body = Vec::new();
    body.extend_from_slice(&mdhd_bytes(track, layout.media_duration)?);
    body.extend_from_slice(&hdlr_bytes(track));
    body.extend_from_slice(&minf(layout, records, chunk_base, use_co64)?);
    Ok(box_bytes(*b"mdia", &body))
}

pub(crate) fn mdhd_bytes(track: &Track, duration: u64) -> Result<Vec<u8>, FormatError> {
    let mut payload = Vec::new();
    let language = encode_language(
        if track.language.len() == 3 && track.language.bytes().all(|byte| byte.is_ascii_lowercase())
        {
            &track.language
        } else {
            "und"
        },
    );
    if duration > u64::from(u32::MAX) {
        push_u64(&mut payload, 0);
        push_u64(&mut payload, 0);
        push_u32(&mut payload, track.timescale);
        push_u64(&mut payload, duration);
        push_u16(&mut payload, language);
        push_u16(&mut payload, 0);
        Ok(full_box_bytes(*b"mdhd", 1, 0, &payload))
    } else {
        push_u32(&mut payload, 0);
        push_u32(&mut payload, 0);
        push_u32(&mut payload, track.timescale);
        push_u32(&mut payload, duration as u32);
        push_u16(&mut payload, language);
        push_u16(&mut payload, 0);
        Ok(full_box_bytes(*b"mdhd", 0, 0, &payload))
    }
}

pub(crate) fn hdlr_bytes(track: &Track) -> Vec<u8> {
    let name = match track.kind {
        TrackKind::Video => "VideoHandler",
        TrackKind::Audio => "SoundHandler",
        TrackKind::Subtitle => "SubtitleHandler",
        TrackKind::Other(_) => "Handler",
    };
    let mut payload = Vec::with_capacity(24 + name.len());
    push_u32(&mut payload, 0);
    payload.extend_from_slice(&track.kind.handler());
    payload.extend_from_slice(&[0; 12]);
    payload.extend_from_slice(name.as_bytes());
    payload.push(0);
    full_box_bytes(*b"hdlr", 0, 0, &payload)
}

fn minf(
    layout: &TrackLayout<'_>,
    records: &[Record],
    chunk_base: u64,
    use_co64: bool,
) -> Result<Vec<u8>, FormatError> {
    let stbl = stbl(layout, records, chunk_base, use_co64)?;
    Ok(minf_with_stbl(layout.track, &stbl))
}

pub(crate) fn minf_with_stbl(track: &Track, stbl: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    match track.kind {
        TrackKind::Audio => {
            let mut payload = Vec::new();
            push_u16(&mut payload, 0);
            push_u16(&mut payload, 0);
            body.extend_from_slice(&full_box_bytes(*b"smhd", 0, 0, &payload));
        }
        _ => {
            let mut payload = Vec::new();
            push_u16(&mut payload, 0);
            push_u16(&mut payload, 0);
            push_u16(&mut payload, 0);
            push_u16(&mut payload, 0);
            body.extend_from_slice(&full_box_bytes(*b"vmhd", 0, 1, &payload));
        }
    }
    body.extend_from_slice(&dinf_bytes());
    body.extend_from_slice(stbl);
    box_bytes(*b"minf", &body)
}

pub(crate) fn dinf_bytes() -> Vec<u8> {
    let url = full_box_bytes(*b"url ", 0, 1, &[]);
    let mut dref = Vec::new();
    push_u32(&mut dref, 1);
    dref.extend_from_slice(&url);
    let dref = full_box_bytes(*b"dref", 0, 0, &dref);
    let mut body = Vec::new();
    body.extend_from_slice(&dref);
    box_bytes(*b"dinf", &body)
}

fn stbl(
    layout: &TrackLayout<'_>,
    records: &[Record],
    chunk_base: u64,
    use_co64: bool,
) -> Result<Vec<u8>, FormatError> {
    let track_records: Vec<&Record> = layout
        .records
        .iter()
        .map(|index| &records[*index])
        .collect();
    let mut body = Vec::new();
    body.extend_from_slice(&build_stsd(layout.track)?);
    body.extend_from_slice(&stts(&track_records));
    if track_records.iter().any(|record| record.pts != record.dts) {
        body.extend_from_slice(&ctts(&track_records));
    }
    if track_records.iter().any(|record| !record.keyframe) {
        body.extend_from_slice(&stss(&track_records));
    }
    body.extend_from_slice(&stsc(layout));
    body.extend_from_slice(&stsz(&track_records));
    body.extend_from_slice(&stco(layout, chunk_base, use_co64));
    Ok(box_bytes(*b"stbl", &body))
}

fn stts(records: &[&Record]) -> Vec<u8> {
    let mut entries: Vec<(u32, u32)> = Vec::new();
    for record in records {
        match entries.last_mut() {
            Some((count, duration)) if *duration == record.duration => *count += 1,
            _ => entries.push((1, record.duration)),
        }
    }
    let mut payload = Vec::with_capacity(4 + entries.len() * 8);
    push_u32(&mut payload, entries.len() as u32);
    for (count, duration) in entries {
        push_u32(&mut payload, count);
        push_u32(&mut payload, duration);
    }
    full_box_bytes(*b"stts", 0, 0, &payload)
}

fn ctts(records: &[&Record]) -> Vec<u8> {
    let mut entries: Vec<(u32, i64)> = Vec::new();
    let mut negative = false;
    for record in records {
        let offset = record.pts - record.dts;
        negative |= offset < 0;
        match entries.last_mut() {
            Some((count, value)) if *value == offset => *count += 1,
            _ => entries.push((1, offset)),
        }
    }
    let version = u8::from(negative);
    let mut payload = Vec::with_capacity(4 + entries.len() * 8);
    push_u32(&mut payload, entries.len() as u32);
    for (count, offset) in entries {
        push_u32(&mut payload, count);
        push_u32(&mut payload, offset as i32 as u32);
    }
    full_box_bytes(*b"ctts", version, 0, &payload)
}

fn stss(records: &[&Record]) -> Vec<u8> {
    let keyframes: Vec<u32> = records
        .iter()
        .enumerate()
        .filter(|(_, record)| record.keyframe)
        .map(|(index, _)| index as u32 + 1)
        .collect();
    let mut payload = Vec::with_capacity(4 + keyframes.len() * 4);
    push_u32(&mut payload, keyframes.len() as u32);
    for sample in keyframes {
        push_u32(&mut payload, sample);
    }
    full_box_bytes(*b"stss", 0, 0, &payload)
}

fn stsc(layout: &TrackLayout<'_>) -> Vec<u8> {
    let mut entries: Vec<(u32, u32)> = Vec::new();
    for (index, chunk) in layout.chunks.iter().enumerate() {
        match entries.last() {
            Some((_, samples)) if *samples == chunk.samples => {}
            _ => entries.push((index as u32 + 1, chunk.samples)),
        }
    }
    let mut payload = Vec::with_capacity(4 + entries.len() * 12);
    push_u32(&mut payload, entries.len() as u32);
    for (first_chunk, samples) in entries {
        push_u32(&mut payload, first_chunk);
        push_u32(&mut payload, samples);
        push_u32(&mut payload, 1);
    }
    full_box_bytes(*b"stsc", 0, 0, &payload)
}

fn stsz(records: &[&Record]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(12 + records.len() * 4);
    push_u32(&mut payload, 0);
    push_u32(&mut payload, records.len() as u32);
    for record in records {
        push_u32(&mut payload, record.size);
    }
    full_box_bytes(*b"stsz", 0, 0, &payload)
}

fn stco(layout: &TrackLayout<'_>, chunk_base: u64, use_co64: bool) -> Vec<u8> {
    let mut payload = Vec::with_capacity(4 + layout.chunks.len() * 8);
    push_u32(&mut payload, layout.chunks.len() as u32);
    for chunk in &layout.chunks {
        let offset = chunk_base + chunk.payload_offset;
        if use_co64 {
            push_u64(&mut payload, offset);
        } else {
            push_u32(&mut payload, offset as u32);
        }
    }
    full_box_bytes(if use_co64 { *b"co64" } else { *b"stco" }, 0, 0, &payload)
}

pub(crate) fn empty_stbl(track: &Track) -> Result<Vec<u8>, FormatError> {
    let mut body = Vec::new();
    body.extend_from_slice(&build_stsd(track)?);
    let mut stts = Vec::new();
    push_u32(&mut stts, 0);
    body.extend_from_slice(&full_box_bytes(*b"stts", 0, 0, &stts));
    let mut stsc = Vec::new();
    push_u32(&mut stsc, 0);
    body.extend_from_slice(&full_box_bytes(*b"stsc", 0, 0, &stsc));
    let mut stsz = Vec::new();
    push_u32(&mut stsz, 0);
    push_u32(&mut stsz, 0);
    body.extend_from_slice(&full_box_bytes(*b"stsz", 0, 0, &stsz));
    let mut stco = Vec::new();
    push_u32(&mut stco, 0);
    body.extend_from_slice(&full_box_bytes(*b"stco", 0, 0, &stco));
    Ok(box_bytes(*b"stbl", &body))
}

pub(crate) fn trex_bytes(track: &Track) -> Vec<u8> {
    let mut payload = Vec::with_capacity(20);
    push_u32(&mut payload, track.id);
    push_u32(&mut payload, 1);
    push_u32(&mut payload, 0);
    push_u32(&mut payload, 0);
    push_u32(&mut payload, 0);
    full_box_bytes(*b"trex", 0, 0, &payload)
}

pub(crate) fn moov_bytes(mvhd: &[u8], traks: &[Vec<u8>], mvex: Option<&[u8]>) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(mvhd);
    for trak in traks {
        body.extend_from_slice(trak);
    }
    if let Some(mvex) = mvex {
        body.extend_from_slice(mvex);
    }
    box_bytes(*b"moov", &body)
}

pub(crate) fn matrix_bytes() -> [u8; 36] {
    let mut matrix = [0_u8; 36];
    matrix[0..4].copy_from_slice(&0x0001_0000_u32.to_be_bytes());
    matrix[16..20].copy_from_slice(&0x0001_0000_u32.to_be_bytes());
    matrix[32..36].copy_from_slice(&0x4000_0000_u32.to_be_bytes());
    matrix
}

pub(crate) fn scale_duration(duration: u64, from: u32, to: u32) -> Result<u64, FormatError> {
    if from == 0 {
        return Err(FormatError::Isobmff("track timescale is zero"));
    }
    let scaled = (u128::from(duration) * u128::from(to) + u128::from(from / 2)) / u128::from(from);
    u64::try_from(scaled).map_err(|_| FormatError::IsobmffLimit("movie duration"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_scale_with_rounding() {
        assert_eq!(scale_duration(90_000, 90_000, 1000).expect("scale"), 1000);
        assert_eq!(scale_duration(1, 90_000, 1000).expect("scale"), 0);
        assert!(scale_duration(1, 0, 1000).is_err());
    }

    #[test]
    fn matrix_is_identity() {
        let matrix = matrix_bytes();
        assert_eq!(&matrix[0..4], &0x0001_0000_u32.to_be_bytes());
        assert_eq!(&matrix[16..20], &0x0001_0000_u32.to_be_bytes());
        assert_eq!(&matrix[32..36], &0x4000_0000_u32.to_be_bytes());
    }
}
