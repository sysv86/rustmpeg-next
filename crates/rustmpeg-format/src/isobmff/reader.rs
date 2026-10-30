// SPDX-License-Identifier: Apache-2.0
//! ISOBMFF and fragmented CMAF reader.
//!
//! The reader indexes both classic sample tables (`stts`, `stsc`, `stsz`,
//! `stco`/`co64`, `ctts`, `stss`) and fragmented movie fragments
//! (`moof`/`traf`/`tfhd`/`tfdt`/`trun`).  Sample payloads are zero-copy
//! [`Bytes`] slices; the public sample order is file order.

use std::collections::{HashMap, VecDeque};

use bytes::Bytes;

use super::boxes::{
    BoxHeader, Boxes, decode_language, full_box, read_i32, read_u16, read_u32, read_u64,
};
use super::sample_entry;
use super::{
    AudioTrack, MAX_SAMPLES, MAX_TRACKS, Sample, SampleEntry, Track, TrackKind, VideoTrack,
};
use crate::FormatError;

#[derive(Debug, Clone, Copy, Default)]
struct Trex {
    default_description: u32,
    default_duration: u32,
    default_size: u32,
    default_flags: u32,
}

#[derive(Debug, Clone, Copy)]
struct EditEntry {
    segment_duration: u64,
    media_time: i64,
}

#[derive(Debug)]
struct ParsedTrack {
    track: Track,
    samples: Vec<(u64, Sample)>,
    edits: Vec<EditEntry>,
}

/// Streaming-free ISOBMFF/CMAF reader over an in-memory document.
///
/// ```no_run
/// use bytes::Bytes;
/// use rustmpeg_format::isobmff::IsobmffReader;
///
/// let data = Bytes::from(std::fs::read("input.mp4").unwrap());
/// let mut reader = IsobmffReader::new(data).unwrap();
/// for track in reader.tracks() {
///     println!("track {}: {:?}", track.id, track.kind);
/// }
/// while let Some(sample) = reader.next_sample().unwrap() {
///     println!("{} bytes at {} ticks", sample.data.len(), sample.dts);
/// }
/// ```
#[derive(Debug)]
pub struct IsobmffReader {
    movie_timescale: u32,
    duration: Option<u64>,
    tracks: Vec<Track>,
    samples: VecDeque<Sample>,
    fragmented: bool,
}

impl IsobmffReader {
    /// Parses the movie header, sample tables, and movie fragments.
    ///
    /// # Errors
    ///
    /// Returns a typed error for a missing or malformed `moov`, truncated
    /// boxes, inconsistent sample tables, out-of-file sample ranges, or an
    /// exceeded track/sample limit.
    pub fn new(data: Bytes) -> Result<Self, FormatError> {
        let source = data;
        let data = source.as_ref();
        let mut moov: Option<BoxHeader> = None;
        let mut moofs: Vec<BoxHeader> = Vec::new();
        let mut mdat_ranges = Vec::new();
        for header in Boxes::new(data, 0, data.len())? {
            let header = header?;
            match header.kind {
                [b'm', b'o', b'o', b'v'] => {
                    if moov.is_some() {
                        return Err(FormatError::Isobmff("multiple moov boxes"));
                    }
                    moov = Some(header);
                }
                [b'm', b'o', b'o', b'f'] => {
                    if moofs.len() >= MAX_SAMPLES {
                        return Err(FormatError::IsobmffLimit("fragment count"));
                    }
                    moofs.push(header);
                }
                [b'm', b'd', b'a', b't'] => {
                    mdat_ranges.push((header.payload_start as u64, header.payload_end as u64));
                }
                _ => {}
            }
        }
        let moov = moov.ok_or(FormatError::Isobmff("missing moov box"))?;

        let mut movie_timescale = 0_u32;
        let mut movie_duration = None;
        let mut parsed_tracks = Vec::new();
        let mut trex = HashMap::new();
        let mut has_mvex = false;

        for child in Boxes::new(data, moov.payload_start, moov.payload_end)? {
            let child = child?;
            match child.kind {
                [b'm', b'v', b'h', b'd'] => {
                    let (timescale, duration) = parse_mvhd(data, &child)?;
                    movie_timescale = timescale;
                    movie_duration = (duration != 0).then_some(duration);
                }
                [b't', b'r', b'a', b'k'] => {
                    if parsed_tracks.len() >= MAX_TRACKS {
                        return Err(FormatError::IsobmffLimit("track count"));
                    }
                    parsed_tracks.push(parse_trak(data, &source, child)?);
                }
                [b'm', b'v', b'e', b'x'] => {
                    has_mvex = true;
                    parse_mvex(data, &child, &mut trex)?;
                }
                _ => {}
            }
        }
        if movie_timescale == 0 {
            movie_timescale = 1000;
        }
        if parsed_tracks.is_empty() {
            return Err(FormatError::Isobmff("movie has no tracks"));
        }
        let mut tracks = Vec::with_capacity(parsed_tracks.len());
        let mut pending_samples = Vec::new();
        for mut parsed in parsed_tracks {
            apply_edits(
                &mut parsed.samples,
                &parsed.edits,
                movie_timescale,
                parsed.track.timescale,
            )?;
            pending_samples.append(&mut parsed.samples);
            tracks.push(parsed.track);
        }

        let mut next_dts: HashMap<u32, i64> = HashMap::new();
        for moof in &moofs {
            parse_moof(
                data,
                &source,
                *moof,
                &tracks,
                &trex,
                &mut next_dts,
                &mut pending_samples,
            )?;
        }
        if pending_samples.len() > MAX_SAMPLES {
            return Err(FormatError::IsobmffLimit("sample count"));
        }
        pending_samples.sort_by_key(|(offset, _)| *offset);
        mdat_ranges.sort_unstable_by_key(|(start, _)| *start);
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(mdat_ranges.len());
        for (start, end) in mdat_ranges {
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }
        let mut range_index = 0_usize;
        for (offset, sample) in &pending_samples {
            let end = offset
                .checked_add(sample.data.len() as u64)
                .ok_or(FormatError::IsobmffLimit("sample range"))?;
            while range_index < merged.len() && *offset >= merged[range_index].1 {
                range_index += 1;
            }
            let Some(&(start, mdat_end)) = merged.get(range_index) else {
                return Err(FormatError::Isobmff("sample is outside mdat"));
            };
            if *offset < start || end > mdat_end {
                return Err(FormatError::Isobmff("sample is outside mdat"));
            }
        }

        for track in &mut tracks {
            let end = pending_samples
                .iter()
                .filter(|(_, sample)| sample.track_id == track.id)
                .map(|(_, sample)| {
                    let end = sample
                        .dts
                        .max(sample.pts)
                        .saturating_add(i64::from(sample.duration));
                    u64::try_from(end).unwrap_or(0)
                })
                .max()
                .unwrap_or(0);
            track.duration = track.duration.max(end);
        }
        let computed_duration = tracks
            .iter()
            .map(|track| scale_duration(track.duration, track.timescale, movie_timescale))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let duration = movie_duration
            .filter(|duration| *duration != 0)
            .or((computed_duration != 0).then_some(computed_duration));

        Ok(Self {
            movie_timescale,
            duration,
            tracks,
            samples: pending_samples
                .into_iter()
                .map(|(_, sample)| sample)
                .collect(),
            fragmented: has_mvex || !moofs.is_empty(),
        })
    }

    /// Movie timescale in ticks per second.
    #[must_use]
    pub fn movie_timescale(&self) -> u32 {
        self.movie_timescale
    }

    /// Movie duration in movie-timescale ticks, when known.
    #[must_use]
    pub fn duration(&self) -> Option<u64> {
        self.duration
    }

    /// Tracks declared by the movie header.
    #[must_use]
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Looks up a track by identifier.
    #[must_use]
    pub fn track(&self, id: u32) -> Option<&Track> {
        self.tracks.iter().find(|track| track.id == id)
    }

    /// Whether the document is a fragmented CMAF/MP4 file.
    #[must_use]
    pub fn is_fragmented(&self) -> bool {
        self.fragmented
    }

    /// Number of indexed samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Returns the next sample in file order, or `None` at the end.
    ///
    /// # Errors
    ///
    /// The eager parser reports all structural errors in [`new`](Self::new),
    /// so this method currently never fails.
    pub fn next_sample(&mut self) -> Result<Option<Sample>, FormatError> {
        Ok(self.samples.pop_front())
    }
}

fn parse_mvhd(data: &[u8], header: &BoxHeader) -> Result<(u32, u64), FormatError> {
    let (version, _, payload) = full_box(data, header)?;
    match version {
        0 => {
            if payload.len() < 16 {
                return Err(FormatError::Isobmff("truncated mvhd"));
            }
            let timescale = read_u32(payload, 8)?;
            let duration = u64::from(read_u32(payload, 12)?);
            Ok((timescale, duration))
        }
        1 => {
            if payload.len() < 28 {
                return Err(FormatError::Isobmff("truncated mvhd v1"));
            }
            let timescale = read_u32(payload, 16)?;
            let duration = read_u64(payload, 20)?;
            Ok((timescale, duration))
        }
        _ => Err(FormatError::Isobmff("unsupported mvhd version")),
    }
}

fn parse_mvex(
    data: &[u8],
    header: &BoxHeader,
    trex: &mut HashMap<u32, Trex>,
) -> Result<(), FormatError> {
    for child in Boxes::new(data, header.payload_start, header.payload_end)? {
        let child = child?;
        if child.kind != *b"trex" {
            continue;
        }
        let (_, _, payload) = full_box(data, &child)?;
        if payload.len() < 20 {
            return Err(FormatError::Isobmff("truncated trex"));
        }
        let track_id = read_u32(payload, 0)?;
        trex.insert(
            track_id,
            Trex {
                default_description: read_u32(payload, 4)?,
                default_duration: read_u32(payload, 8)?,
                default_size: read_u32(payload, 12)?,
                default_flags: read_u32(payload, 16)?,
            },
        );
    }
    Ok(())
}

fn parse_trak(data: &[u8], bytes: &Bytes, header: BoxHeader) -> Result<ParsedTrack, FormatError> {
    let mut tkhd = None;
    let mut mdia = None;
    let mut edts = None;
    for child in Boxes::new(data, header.payload_start, header.payload_end)? {
        let child = child?;
        match child.kind {
            [b't', b'k', b'h', b'd'] => tkhd = Some(child),
            [b'm', b'd', b'i', b'a'] => mdia = Some(child),
            [b'e', b'd', b't', b's'] => edts = Some(child),
            _ => {}
        }
    }
    let tkhd = tkhd.ok_or(FormatError::Isobmff("track without tkhd"))?;
    let mdia = mdia.ok_or(FormatError::Isobmff("track without mdia"))?;
    let track_id = parse_tkhd(data, &tkhd)?;
    if track_id == 0 {
        return Err(FormatError::Isobmff("track identifier is zero"));
    }

    let mut mdhd = None;
    let mut hdlr = None;
    let mut minf = None;
    for child in Boxes::new(data, mdia.payload_start, mdia.payload_end)? {
        let child = child?;
        match child.kind {
            [b'm', b'd', b'h', b'd'] => mdhd = Some(child),
            [b'h', b'd', b'l', b'r'] => hdlr = Some(child),
            [b'm', b'i', b'n', b'f'] => minf = Some(child),
            _ => {}
        }
    }
    let mdhd = mdhd.ok_or(FormatError::Isobmff("track without mdhd"))?;
    let hdlr = hdlr.ok_or(FormatError::Isobmff("track without hdlr"))?;
    let minf = minf.ok_or(FormatError::Isobmff("track without minf"))?;
    let (timescale, duration, language) = parse_mdhd(data, &mdhd)?;
    if timescale == 0 {
        return Err(FormatError::Isobmff("track timescale is zero"));
    }
    let kind = parse_hdlr(data, &hdlr)?;

    let mut stbl = None;
    for child in Boxes::new(data, minf.payload_start, minf.payload_end)? {
        let child = child?;
        if child.kind == *b"stbl" {
            stbl = Some(child);
            break;
        }
    }
    let stbl = stbl.ok_or(FormatError::Isobmff("track without stbl"))?;
    let (sample_entry, video, audio, samples) = parse_stbl(data, bytes, stbl, kind, track_id)?;
    let edits = match edts {
        Some(header) => parse_edts(data, &header)?,
        None => Vec::new(),
    };

    if kind == TrackKind::Video && video.is_none() {
        return Err(FormatError::Isobmff("video track without visual settings"));
    }
    if kind == TrackKind::Audio && audio.is_none() {
        return Err(FormatError::Isobmff("audio track without audio settings"));
    }
    let track = Track {
        id: track_id,
        kind,
        timescale,
        duration,
        language,
        sample_entry,
        video,
        audio,
    };
    Ok(ParsedTrack {
        track,
        samples,
        edits,
    })
}

fn parse_edts(data: &[u8], header: &BoxHeader) -> Result<Vec<EditEntry>, FormatError> {
    let mut entries = Vec::new();
    for child in Boxes::new(data, header.payload_start, header.payload_end)? {
        let child = child?;
        if child.kind != *b"elst" {
            continue;
        }
        let (version, _, payload) = full_box(data, &child)?;
        if payload.len() < 4 {
            return Err(FormatError::Isobmff("truncated elst"));
        }
        let count = read_u32(payload, 0)? as usize;
        if count > MAX_SAMPLES {
            return Err(FormatError::IsobmffLimit("edit list entry count"));
        }
        let mut position = 4;
        for _ in 0..count {
            let (segment_duration, media_time, extra) = match version {
                0 => {
                    if position + 12 > payload.len() {
                        return Err(FormatError::Isobmff("truncated elst entry"));
                    }
                    (
                        u64::from(read_u32(payload, position)?),
                        i64::from(read_i32(payload, position + 4)?),
                        12,
                    )
                }
                1 => {
                    if position + 20 > payload.len() {
                        return Err(FormatError::Isobmff("truncated elst entry v1"));
                    }
                    (
                        read_u64(payload, position)?,
                        read_u64(payload, position + 8)? as i64,
                        20,
                    )
                }
                _ => return Err(FormatError::Isobmff("unsupported elst version")),
            };
            position += extra;
            entries.push(EditEntry {
                segment_duration,
                media_time,
            });
        }
        if position != payload.len() {
            return Err(FormatError::Isobmff("elst has trailing bytes"));
        }
    }
    Ok(entries)
}

fn apply_edits(
    samples: &mut [(u64, Sample)],
    edits: &[EditEntry],
    movie_timescale: u32,
    track_timescale: u32,
) -> Result<(), FormatError> {
    if edits.is_empty() {
        return Ok(());
    }
    let mut delay = 0_u64;
    let mut media_time = None;
    for edit in edits {
        if edit.media_time == -1 {
            delay = delay
                .checked_add(edit.segment_duration)
                .ok_or(FormatError::IsobmffLimit("edit delay"))?;
        } else if edit.media_time < 0 {
            return Err(FormatError::Isobmff("invalid negative edit media time"));
        } else if media_time.is_some() {
            return Err(FormatError::Isobmff(
                "multiple media edits are not supported",
            ));
        } else {
            media_time = Some(edit.media_time);
        }
    }
    let delay_ticks = i64::try_from(scale_duration(delay, movie_timescale, track_timescale)?)
        .map_err(|_| FormatError::Isobmff("edit delay overflow"))?;
    let shift = delay_ticks
        .checked_sub(media_time.unwrap_or(0))
        .ok_or(FormatError::Isobmff("edit timestamp overflow"))?;
    if shift == 0 {
        return Ok(());
    }
    for (_, sample) in samples.iter_mut() {
        sample.dts = sample
            .dts
            .checked_add(shift)
            .ok_or(FormatError::Isobmff("edit timestamp overflow"))?;
        sample.pts = sample
            .pts
            .checked_add(shift)
            .ok_or(FormatError::Isobmff("edit timestamp overflow"))?;
    }
    Ok(())
}

fn parse_tkhd(data: &[u8], header: &BoxHeader) -> Result<u32, FormatError> {
    let (version, _, payload) = full_box(data, header)?;
    match version {
        0 => {
            if payload.len() < 12 {
                return Err(FormatError::Isobmff("truncated tkhd"));
            }
            read_u32(payload, 8)
        }
        1 => {
            if payload.len() < 20 {
                return Err(FormatError::Isobmff("truncated tkhd v1"));
            }
            read_u32(payload, 16)
        }
        _ => Err(FormatError::Isobmff("unsupported tkhd version")),
    }
}

fn parse_mdhd(data: &[u8], header: &BoxHeader) -> Result<(u32, u64, String), FormatError> {
    let (version, _, payload) = full_box(data, header)?;
    match version {
        0 => {
            if payload.len() < 20 {
                return Err(FormatError::Isobmff("truncated mdhd"));
            }
            let timescale = read_u32(payload, 8)?;
            let duration = u64::from(read_u32(payload, 12)?);
            let language = decode_language(read_u16(payload, 16)?);
            Ok((timescale, duration, language))
        }
        1 => {
            if payload.len() < 32 {
                return Err(FormatError::Isobmff("truncated mdhd v1"));
            }
            let timescale = read_u32(payload, 16)?;
            let duration = read_u64(payload, 20)?;
            let language = decode_language(read_u16(payload, 28)?);
            Ok((timescale, duration, language))
        }
        _ => Err(FormatError::Isobmff("unsupported mdhd version")),
    }
}

fn parse_hdlr(data: &[u8], header: &BoxHeader) -> Result<TrackKind, FormatError> {
    let (_, _, payload) = full_box(data, header)?;
    if payload.len() < 12 {
        return Err(FormatError::Isobmff("truncated hdlr"));
    }
    let mut handler = [0_u8; 4];
    handler.copy_from_slice(&payload[4..8]);
    Ok(TrackKind::from_handler(handler))
}

#[derive(Debug, Clone, Copy)]
struct StscEntry {
    first_chunk: u32,
    samples_per_chunk: u32,
    description: u32,
}

type SampleTable = (
    SampleEntry,
    Option<VideoTrack>,
    Option<AudioTrack>,
    Vec<(u64, Sample)>,
);

fn parse_stbl(
    data: &[u8],
    bytes: &Bytes,
    header: BoxHeader,
    kind: TrackKind,
    track_id: u32,
) -> Result<SampleTable, FormatError> {
    let mut stsd = None;
    let mut stts = None;
    let mut ctts = None;
    let mut stsc = None;
    let mut stsz = None;
    let mut chunk_offsets = None;
    let mut stss = None;
    for child in Boxes::new(data, header.payload_start, header.payload_end)? {
        let child = child?;
        match child.kind {
            [b's', b't', b's', b'd'] => stsd = Some(child),
            [b's', b't', b't', b's'] => stts = Some(child),
            [b'c', b't', b't', b's'] => ctts = Some(child),
            [b's', b't', b's', b'c'] => stsc = Some(child),
            [b's', b't', b's', b'z'] => stsz = Some(child),
            [b's', b't', b'c', b'o'] | [b'c', b'o', b'6', b'4'] => chunk_offsets = Some(child),
            [b's', b't', b's', b's'] => stss = Some(child),
            _ => {}
        }
    }
    let stsd = stsd.ok_or(FormatError::Isobmff("track without stsd"))?;

    // stsd: entry_count followed by sample entries; only the first description
    // is supported because the public Track carries one decoder config.
    let (_, _, stsd_payload) = full_box(data, &stsd)?;
    if stsd_payload.len() < 4 {
        return Err(FormatError::Isobmff("truncated stsd"));
    }
    let entry_count = read_u32(stsd_payload, 0)?;
    if entry_count == 0 {
        return Err(FormatError::Isobmff("sample description is empty"));
    }
    let entry = Boxes::new(data, stsd.payload_start + 8, stsd.payload_end)?
        .next()
        .transpose()?
        .ok_or(FormatError::Isobmff("missing sample entry"))?;
    let (sample_entry, video, audio) = sample_entry::parse(data, bytes, &entry, kind)?;

    let (stts, stsc, stsz, chunk_offsets) = match (stts, stsc, stsz, chunk_offsets) {
        (Some(stts), Some(stsc), Some(stsz), Some(chunk_offsets)) => {
            (stts, stsc, stsz, chunk_offsets)
        }
        (None, None, None, None) => {
            // Fragment initialization segments may carry only `stsd`.
            return Ok((sample_entry, video, audio, Vec::new()));
        }
        _ => {
            return Err(FormatError::Isobmff(
                "incomplete sample table in a non-empty track",
            ));
        }
    };

    let durations = parse_stts(data, &stts)?;
    let composition = ctts
        .as_ref()
        .map(|header| parse_ctts(data, header))
        .transpose()?;
    let entries = parse_stsc(data, &stsc)?;
    let (sample_size, sizes) = parse_stsz(data, &stsz)?;
    let offsets = parse_chunk_offsets(data, &chunk_offsets)?;
    let sync = stss
        .as_ref()
        .map(|header| parse_stss(data, header))
        .transpose()?;

    if entries.iter().any(|entry| entry.description != 1) {
        return Err(FormatError::Isobmff(
            "multiple sample descriptions are not supported",
        ));
    }

    let sample_count = sizes.len();
    let sync = match sync {
        Some(values) => {
            let mut flags = vec![false; sample_count];
            for value in values {
                let index = value as usize;
                if index == 0 || index > sample_count {
                    return Err(FormatError::Isobmff("stss sample is out of range"));
                }
                flags[index - 1] = true;
            }
            Some(flags)
        }
        None => None,
    };
    if durations.len() != sample_count {
        return Err(FormatError::Isobmff("stts and stsz sample counts differ"));
    }
    if let Some(composition) = &composition
        && composition.len() != sample_count
    {
        return Err(FormatError::Isobmff("ctts and stsz sample counts differ"));
    }
    if !offsets.is_empty() && (entries.is_empty() || entries[0].first_chunk != 1) {
        return Err(FormatError::Isobmff(
            "stsc does not start at the first chunk",
        ));
    }
    let mut sample_offsets = Vec::with_capacity(sample_count);
    let mut sample_index = 0_usize;
    let mut entry_index = 0_usize;
    for (chunk_index, chunk_offset) in offsets.iter().enumerate() {
        while entry_index + 1 < entries.len()
            && entries[entry_index + 1].first_chunk <= chunk_index as u32 + 1
        {
            entry_index += 1;
        }
        let entry = &entries[entry_index];
        let mut offset = *chunk_offset;
        for _ in 0..entry.samples_per_chunk {
            if sample_index >= sample_count {
                return Err(FormatError::Isobmff(
                    "stsc describes more samples than stsz",
                ));
            }
            let size = match sample_size {
                0 => sizes[sample_index],
                size => size,
            };
            let end = offset
                .checked_add(u64::from(size))
                .ok_or(FormatError::IsobmffLimit("sample range"))?;
            if end > bytes.len() as u64 {
                return Err(FormatError::Isobmff("sample range exceeds the file"));
            }
            sample_offsets.push((offset, end));
            sample_index += 1;
            offset = end;
        }
    }
    if sample_index != sample_count {
        return Err(FormatError::Isobmff(
            "stsc describes fewer samples than stsz",
        ));
    }

    let mut samples = Vec::with_capacity(sample_count);
    let mut dts = 0_i64;
    for (index, (start, end)) in sample_offsets.into_iter().enumerate() {
        let duration = durations[index];
        let pts = dts
            .checked_add(
                composition
                    .as_ref()
                    .and_then(|values| values.get(index).copied())
                    .unwrap_or(0),
            )
            .ok_or(FormatError::Isobmff("composition timestamp overflow"))?;
        let keyframe = sync.as_ref().is_none_or(|values| values[index]);
        samples.push((
            start,
            Sample {
                track_id,
                dts,
                pts,
                duration,
                keyframe,
                data: bytes.slice(start as usize..end as usize),
            },
        ));
        dts = dts
            .checked_add(i64::from(duration))
            .ok_or(FormatError::Isobmff("decode timestamp overflow"))?;
    }
    Ok((sample_entry, video, audio, samples))
}

fn parse_stts(data: &[u8], header: &BoxHeader) -> Result<Vec<u32>, FormatError> {
    let (_, _, payload) = full_box(data, header)?;
    if payload.len() < 4 {
        return Err(FormatError::Isobmff("truncated stts"));
    }
    let count = read_u32(payload, 0)? as usize;
    if count > MAX_SAMPLES {
        return Err(FormatError::IsobmffLimit("stts entry count"));
    }
    if payload.len() != 4 + count * 8 {
        return Err(FormatError::Isobmff("invalid stts length"));
    }
    let mut durations = Vec::new();
    for index in 0..count {
        let samples = read_u32(payload, 4 + index * 8)?;
        let delta = read_u32(payload, 8 + index * 8)?;
        let new_len = durations
            .len()
            .checked_add(samples as usize)
            .ok_or(FormatError::IsobmffLimit("sample count"))?;
        if new_len > MAX_SAMPLES {
            return Err(FormatError::IsobmffLimit("sample count"));
        }
        durations.resize(new_len, delta);
    }
    Ok(durations)
}

fn parse_ctts(data: &[u8], header: &BoxHeader) -> Result<Vec<i64>, FormatError> {
    let (version, _, payload) = full_box(data, header)?;
    if payload.len() < 4 {
        return Err(FormatError::Isobmff("truncated ctts"));
    }
    let count = read_u32(payload, 0)? as usize;
    if count > MAX_SAMPLES {
        return Err(FormatError::IsobmffLimit("ctts entry count"));
    }
    if payload.len() != 4 + count * 8 {
        return Err(FormatError::Isobmff("invalid ctts length"));
    }
    let mut offsets = Vec::new();
    for index in 0..count {
        let samples = read_u32(payload, 4 + index * 8)?;
        let raw = read_u32(payload, 8 + index * 8)?;
        let offset = match version {
            0 => i64::from(raw),
            1 => i64::from(raw as i32),
            _ => return Err(FormatError::Isobmff("unsupported ctts version")),
        };
        let new_len = offsets
            .len()
            .checked_add(samples as usize)
            .ok_or(FormatError::IsobmffLimit("sample count"))?;
        if new_len > MAX_SAMPLES {
            return Err(FormatError::IsobmffLimit("sample count"));
        }
        offsets.resize(new_len, offset);
    }
    Ok(offsets)
}

fn parse_stsc(data: &[u8], header: &BoxHeader) -> Result<Vec<StscEntry>, FormatError> {
    let (_, _, payload) = full_box(data, header)?;
    if payload.len() < 4 {
        return Err(FormatError::Isobmff("truncated stsc"));
    }
    let count = read_u32(payload, 0)? as usize;
    if count > MAX_SAMPLES {
        return Err(FormatError::IsobmffLimit("stsc entry count"));
    }
    if payload.len() != 4 + count * 12 {
        return Err(FormatError::Isobmff("invalid stsc length"));
    }
    let mut entries = Vec::with_capacity(count);
    let mut previous = 0_u32;
    for index in 0..count {
        let first_chunk = read_u32(payload, 4 + index * 12)?;
        let samples_per_chunk = read_u32(payload, 8 + index * 12)?;
        let description = read_u32(payload, 12 + index * 12)?;
        if first_chunk == 0 || first_chunk <= previous || samples_per_chunk == 0 {
            return Err(FormatError::Isobmff("invalid stsc entry"));
        }
        previous = first_chunk;
        entries.push(StscEntry {
            first_chunk,
            samples_per_chunk,
            description,
        });
    }
    Ok(entries)
}

fn parse_stsz(data: &[u8], header: &BoxHeader) -> Result<(u32, Vec<u32>), FormatError> {
    let (_, _, payload) = full_box(data, header)?;
    if payload.len() < 8 {
        return Err(FormatError::Isobmff("truncated stsz"));
    }
    let sample_size = read_u32(payload, 0)?;
    let count = read_u32(payload, 4)? as usize;
    if count > MAX_SAMPLES {
        return Err(FormatError::IsobmffLimit("sample count"));
    }
    if sample_size == 0 {
        if payload.len() != 8 + count * 4 {
            return Err(FormatError::Isobmff("invalid stsz length"));
        }
        let mut sizes = Vec::with_capacity(count);
        for index in 0..count {
            sizes.push(read_u32(payload, 8 + index * 4)?);
        }
        Ok((0, sizes))
    } else {
        if payload.len() != 8 {
            return Err(FormatError::Isobmff("invalid constant stsz length"));
        }
        Ok((sample_size, vec![sample_size; count]))
    }
}

fn parse_chunk_offsets(data: &[u8], header: &BoxHeader) -> Result<Vec<u64>, FormatError> {
    let (_, _, payload) = full_box(data, header)?;
    if payload.len() < 4 {
        return Err(FormatError::Isobmff("truncated chunk offset table"));
    }
    let count = read_u32(payload, 0)? as usize;
    if count > MAX_SAMPLES {
        return Err(FormatError::IsobmffLimit("chunk offset count"));
    }
    let large = header.kind == *b"co64";
    let width = if large { 8 } else { 4 };
    if payload.len() != 4 + count * width {
        return Err(FormatError::Isobmff("invalid chunk offset table length"));
    }
    let mut offsets = Vec::with_capacity(count);
    for index in 0..count {
        let offset = if large {
            read_u64(payload, 4 + index * 8)?
        } else {
            u64::from(read_u32(payload, 4 + index * 4)?)
        };
        offsets.push(offset);
    }
    Ok(offsets)
}

fn parse_stss(data: &[u8], header: &BoxHeader) -> Result<Vec<u32>, FormatError> {
    let (_, _, payload) = full_box(data, header)?;
    if payload.len() < 4 {
        return Err(FormatError::Isobmff("truncated stss"));
    }
    let count = read_u32(payload, 0)? as usize;
    if count > MAX_SAMPLES {
        return Err(FormatError::IsobmffLimit("stss entry count"));
    }
    if payload.len() != 4 + count * 4 {
        return Err(FormatError::Isobmff("invalid stss length"));
    }
    let mut samples = Vec::with_capacity(count);
    let mut previous = 0_u32;
    for index in 0..count {
        let sample = read_u32(payload, 4 + index * 4)?;
        if sample == 0 || sample <= previous {
            return Err(FormatError::Isobmff("invalid stss entry"));
        }
        previous = sample;
        samples.push(sample);
    }
    Ok(samples)
}

fn parse_moof(
    data: &[u8],
    bytes: &Bytes,
    moof: BoxHeader,
    tracks: &[Track],
    trex: &HashMap<u32, Trex>,
    next_dts: &mut HashMap<u32, i64>,
    samples: &mut Vec<(u64, Sample)>,
) -> Result<(), FormatError> {
    let mut previous_traf_end = moof.start as u64;
    for child in Boxes::new(data, moof.payload_start, moof.payload_end)? {
        let child = child?;
        if child.kind != *b"traf" {
            continue;
        }
        parse_traf(
            data,
            bytes,
            &child,
            moof.start as u64,
            &mut previous_traf_end,
            tracks,
            trex,
            next_dts,
            samples,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn parse_traf(
    data: &[u8],
    bytes: &Bytes,
    traf: &BoxHeader,
    moof_start: u64,
    previous_traf_end: &mut u64,
    tracks: &[Track],
    trex: &HashMap<u32, Trex>,
    next_dts: &mut HashMap<u32, i64>,
    samples: &mut Vec<(u64, Sample)>,
) -> Result<(), FormatError> {
    let mut tfhd = None;
    let mut tfdt = None;
    let mut truns = Vec::new();
    for child in Boxes::new(data, traf.payload_start, traf.payload_end)? {
        let child = child?;
        match child.kind {
            [b't', b'f', b'h', b'd'] => tfhd = Some(child),
            [b't', b'f', b'd', b't'] => tfdt = Some(child),
            [b't', b'r', b'u', b'n'] => truns.push(child),
            _ => {}
        }
    }
    let tfhd = tfhd.ok_or(FormatError::Isobmff("traf without tfhd"))?;
    let (version, flags, payload) = full_box(data, &tfhd)?;
    let _ = version;
    if payload.len() < 4 {
        return Err(FormatError::Isobmff("truncated tfhd"));
    }
    let track_id = read_u32(payload, 0)?;
    if !tracks.iter().any(|track| track.id == track_id) {
        return Err(FormatError::Isobmff("fragment references an unknown track"));
    }
    let defaults = trex.get(&track_id).copied().unwrap_or_default();
    let mut position = 4;
    let base_offset = if flags & 0x000001 != 0 {
        let value = read_u64(payload, position)?;
        position += 8;
        Some(value)
    } else {
        None
    };
    let description = if flags & 0x000002 != 0 {
        let value = read_u32(payload, position)?;
        position += 4;
        Some(value)
    } else {
        None
    };
    if description.unwrap_or(defaults.default_description) != 1 {
        return Err(FormatError::Isobmff(
            "multiple sample descriptions are not supported",
        ));
    }
    let default_duration = if flags & 0x000008 != 0 {
        read_u32(payload, position)?
    } else {
        defaults.default_duration
    };
    position += if flags & 0x000008 != 0 { 4 } else { 0 };
    let default_size = if flags & 0x000010 != 0 {
        read_u32(payload, position)?
    } else {
        defaults.default_size
    };
    position += if flags & 0x000010 != 0 { 4 } else { 0 };
    let default_flags = if flags & 0x000020 != 0 {
        read_u32(payload, position)?
    } else {
        defaults.default_flags
    };
    let base = match base_offset {
        Some(offset) => offset,
        None if flags & 0x020000 != 0 => moof_start,
        None => *previous_traf_end,
    };

    let mut current_dts = match tfdt {
        Some(header) => {
            let (version, _, payload) = full_box(data, &header)?;
            match version {
                0 => {
                    if payload.len() < 4 {
                        return Err(FormatError::Isobmff("truncated tfdt"));
                    }
                    i64::from(read_u32(payload, 0)?)
                }
                1 => {
                    if payload.len() < 8 {
                        return Err(FormatError::Isobmff("truncated tfdt v1"));
                    }
                    i64::try_from(read_u64(payload, 0)?)
                        .map_err(|_| FormatError::Isobmff("tfdt overflow"))?
                }
                _ => return Err(FormatError::Isobmff("unsupported tfdt version")),
            }
        }
        None => *next_dts.get(&track_id).unwrap_or(&0),
    };
    let mut data_position = base;
    for trun in truns {
        let (version, flags, payload) = full_box(data, &trun)?;
        if payload.len() < 4 {
            return Err(FormatError::Isobmff("truncated trun"));
        }
        let count = read_u32(payload, 0)? as usize;
        if count > MAX_SAMPLES {
            return Err(FormatError::IsobmffLimit("sample count"));
        }
        let mut position = 4;
        if flags & 0x000001 != 0 {
            let offset = read_i32(payload, position)?;
            position += 4;
            let absolute = i64::try_from(base)
                .ok()
                .and_then(|base| base.checked_add(i64::from(offset)))
                .ok_or(FormatError::Isobmff("trun data offset overflow"))?;
            if absolute < 0 {
                return Err(FormatError::Isobmff("negative trun data offset"));
            }
            data_position = absolute as u64;
        }
        let first_flags = if flags & 0x000004 != 0 {
            let value = read_u32(payload, position)?;
            position += 4;
            Some(value)
        } else {
            None
        };
        for index in 0..count {
            let duration = if flags & 0x000100 != 0 {
                let value = read_u32(payload, position)?;
                position += 4;
                value
            } else {
                default_duration
            };
            let size = if flags & 0x000200 != 0 {
                let value = read_u32(payload, position)?;
                position += 4;
                value
            } else {
                default_size
            };
            let sample_flags = if flags & 0x000400 != 0 {
                let value = read_u32(payload, position)?;
                position += 4;
                value
            } else if index == 0 {
                first_flags.unwrap_or(default_flags)
            } else {
                default_flags
            };
            let composition = if flags & 0x000800 != 0 {
                let raw = read_u32(payload, position)?;
                position += 4;
                if version == 0 {
                    i64::from(raw)
                } else {
                    i64::from(raw as i32)
                }
            } else {
                0
            };
            let end = data_position
                .checked_add(u64::from(size))
                .ok_or(FormatError::IsobmffLimit("sample range"))?;
            if end > data.len() as u64 {
                return Err(FormatError::Isobmff("sample range exceeds the file"));
            }
            let pts = current_dts
                .checked_add(composition)
                .ok_or(FormatError::Isobmff("composition timestamp overflow"))?;
            samples.push((
                data_position,
                Sample {
                    track_id,
                    dts: current_dts,
                    pts,
                    duration,
                    keyframe: sample_flags & 0x0001_0000 == 0,
                    data: bytes.slice(
                        usize::try_from(data_position).map_err(|_| {
                            FormatError::Isobmff("sample offset exceeds the address space")
                        })?..usize::try_from(end).map_err(|_| {
                            FormatError::Isobmff("sample end exceeds the address space")
                        })?,
                    ),
                },
            ));
            current_dts = current_dts
                .checked_add(i64::from(duration))
                .ok_or(FormatError::Isobmff("decode timestamp overflow"))?;
            data_position = end;
            if samples.len() > MAX_SAMPLES {
                return Err(FormatError::IsobmffLimit("sample count"));
            }
        }
        if position != payload.len() {
            return Err(FormatError::Isobmff("trun has trailing bytes"));
        }
    }
    *previous_traf_end = data_position;
    next_dts.insert(track_id, current_dts);
    Ok(())
}

fn scale_duration(duration: u64, from: u32, to: u32) -> Result<u64, FormatError> {
    if from == 0 {
        return Err(FormatError::Isobmff("track timescale is zero"));
    }
    let value = (u128::from(duration) * u128::from(to) + u128::from(from / 2)) / u128::from(from);
    u64::try_from(value).map_err(|_| FormatError::IsobmffLimit("movie duration"))
}

#[cfg(test)]
mod tests {
    use super::super::boxes::{full_box_bytes, next_box};
    use super::*;

    #[test]
    fn duration_scaling_rounds() {
        assert_eq!(scale_duration(90_000, 90_000, 1000).expect("scale"), 1000);
        assert!(scale_duration(1, 0, 1000).is_err());
    }

    #[test]
    fn co64_offsets_are_read() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&2_u32.to_be_bytes());
        payload.extend_from_slice(&0x1_0000_0000_u64.to_be_bytes());
        payload.extend_from_slice(&0x2_0000_0000_u64.to_be_bytes());
        let boxed = full_box_bytes(*b"co64", 0, 0, &payload);
        let mut position = 0;
        let header = next_box(&boxed, &mut position, boxed.len()).expect("header");
        assert_eq!(
            parse_chunk_offsets(&boxed, &header).expect("offsets"),
            vec![0x1_0000_0000, 0x2_0000_0000]
        );
    }

    #[test]
    fn ctts_version_one_keeps_negative_offsets() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&1_u32.to_be_bytes());
        payload.extend_from_slice(&2_u32.to_be_bytes());
        payload.extend_from_slice(&(-5_i32).to_be_bytes());
        let boxed = full_box_bytes(*b"ctts", 1, 0, &payload);
        let mut position = 0;
        let header = next_box(&boxed, &mut position, boxed.len()).expect("header");
        assert_eq!(parse_ctts(&boxed, &header).expect("offsets"), vec![-5, -5]);
    }
}
