// SPDX-License-Identifier: Apache-2.0
//! Fragmented CMAF writer: initialization segment plus media segments.
//!
//! Each media segment contains one track fragment with a single `trun`, which
//! is the common CMAF layout.  The first sample of a segment must be a random
//! access point, and the fragment uses `default-base-is-moof` so that sample
//! data offsets are relative to the start of `moof`.

use bytes::Bytes;

use super::boxes::{box_bytes, full_box_bytes, push_u32, push_u64};
use super::writer::{
    empty_stbl, ftyp_bytes, hdlr_bytes, mdhd_bytes, minf_with_stbl, moov_bytes, mvhd_bytes,
    tkhd_bytes, trex_bytes,
};
use super::{MAX_TRACKS, Sample, Track, validate_track};
use crate::FormatError;

/// Configuration for [`CmafWriter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CmafConfig {
    /// Major brand of the initialization segment.
    pub major_brand: [u8; 4],
    /// Minor version of the initialization segment.
    pub minor_version: u32,
    /// Compatible brands of the initialization segment.
    pub compatible_brands: Vec<[u8; 4]>,
    /// Movie timescale in ticks per second.
    pub movie_timescale: u32,
}

impl Default for CmafConfig {
    fn default() -> Self {
        Self {
            major_brand: *b"iso6",
            minor_version: 0,
            compatible_brands: vec![*b"iso6", *b"cmfc", *b"cmf2"],
            movie_timescale: 1000,
        }
    }
}

/// Writes a CMAF initialization segment and one media segment per call.
#[derive(Debug)]
pub struct CmafWriter {
    config: CmafConfig,
    tracks: Vec<Track>,
    sequence: u32,
}

impl CmafWriter {
    /// Creates an empty CMAF writer.
    #[must_use]
    pub fn new(config: CmafConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            sequence: 1,
        }
    }

    /// Declares a track. Tracks must be added before the first segment.
    ///
    /// # Errors
    ///
    /// Rejects duplicate identifiers, invalid layouts, and track counts above
    /// [`MAX_TRACKS`].
    pub fn add_track(&mut self, track: Track) -> Result<(), FormatError> {
        if self.tracks.len() >= MAX_TRACKS {
            return Err(FormatError::IsobmffLimit("track count"));
        }
        if self.tracks.iter().any(|other| other.id == track.id) {
            return Err(FormatError::Isobmff("duplicate track identifier"));
        }
        validate_track(&track)?;
        if !matches!(
            track.kind,
            super::TrackKind::Video | super::TrackKind::Audio
        ) {
            return Err(FormatError::Isobmff(
                "only audio and video tracks can be written",
            ));
        }
        self.tracks.push(track);
        Ok(())
    }

    /// Returns the next fragment sequence number.
    #[must_use]
    pub fn sequence_number(&self) -> u32 {
        self.sequence
    }

    /// Builds the initialization segment (`ftyp` followed by `moov`).
    ///
    /// # Errors
    ///
    /// Rejects a document without tracks, a zero movie timescale, or any size
    /// overflow.
    pub fn init_segment(&self) -> Result<Bytes, FormatError> {
        if self.tracks.is_empty() {
            return Err(FormatError::Isobmff("movie has no tracks"));
        }
        if self.config.movie_timescale == 0 {
            return Err(FormatError::Isobmff("movie timescale is zero"));
        }
        let ftyp = ftyp_bytes(
            self.config.major_brand,
            self.config.minor_version,
            &self.config.compatible_brands,
        );
        let next_track = self
            .tracks
            .iter()
            .map(|track| track.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let mvhd = mvhd_bytes(0, self.config.movie_timescale, next_track)?;
        let mut traks = Vec::with_capacity(self.tracks.len());
        for track in &self.tracks {
            traks.push(cmaf_trak(track)?);
        }
        let mut trex = Vec::new();
        for track in &self.tracks {
            trex.extend_from_slice(&trex_bytes(track));
        }
        let mvex = box_bytes(*b"mvex", &trex);
        let moov = moov_bytes(&mvhd, &traks, Some(&mvex));
        let mut init = ftyp;
        init.extend_from_slice(&moov);
        Ok(Bytes::from(init))
    }

    /// Builds one media segment for `track_id`.
    ///
    /// # Errors
    ///
    /// Rejects an unknown track, empty or mixed-track sample lists, a first
    /// sample that is not a keyframe, non-monotonic decode timestamps, zero
    /// durations, or segment payloads above the 32-bit `mdat` limit.
    pub fn write_segment(
        &mut self,
        track_id: u32,
        samples: &[Sample],
    ) -> Result<Bytes, FormatError> {
        if super::track_by_id(&self.tracks, track_id).is_none() {
            return Err(FormatError::Isobmff("segment references an unknown track"));
        }
        if samples.is_empty() {
            return Err(FormatError::Isobmff("segment has no samples"));
        }
        if !samples[0].keyframe {
            return Err(FormatError::Isobmff(
                "CMAF segment must start with a keyframe",
            ));
        }
        let mut previous_dts = None;
        for sample in samples {
            if sample.track_id != track_id {
                return Err(FormatError::Isobmff(
                    "segment contains samples from multiple tracks",
                ));
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
            if previous_dts.is_some_and(|previous| sample.dts < previous) {
                return Err(FormatError::Isobmff("samples are not in decode order"));
            }
            let offset = sample
                .pts
                .checked_sub(sample.dts)
                .ok_or(FormatError::Isobmff("composition offset overflow"))?;
            if offset < i64::from(i32::MIN) || offset > i64::from(i32::MAX) {
                return Err(FormatError::Isobmff(
                    "composition offset does not fit in 32 bits",
                ));
            }
            previous_dts = Some(sample.dts);
        }
        let mut payload = Vec::new();
        for sample in samples {
            payload.extend_from_slice(&sample.data);
        }
        if payload.len() as u64 + 8 > u64::from(u32::MAX) {
            return Err(FormatError::IsobmffLimit("CMAF segment payload"));
        }

        let composition = samples.iter().any(|sample| sample.pts != sample.dts);
        let version = u8::from(samples.iter().any(|sample| sample.pts < sample.dts));
        let mut trun_flags = 0x000001 | 0x000100 | 0x000200 | 0x000400;
        if composition {
            trun_flags |= 0x000800;
        }

        // First pass: measure the moof with a zero data offset.
        let trun = build_trun(samples, trun_flags, version, None)?;
        let mut traf_body = Vec::new();
        traf_body.extend_from_slice(&tfhd(track_id));
        traf_body.extend_from_slice(&tfdt(samples[0].dts as u64));
        traf_body.extend_from_slice(&trun);
        let traf = box_bytes(*b"traf", &traf_body);
        let mut moof_body = Vec::new();
        moof_body.extend_from_slice(&mfhd(self.sequence));
        moof_body.extend_from_slice(&traf);
        let moof = box_bytes(*b"moof", &moof_body);
        let data_offset =
            u32::try_from(moof.len() + 8).map_err(|_| FormatError::IsobmffLimit("moof size"))?;

        // Second pass: the offset is relative to the start of `moof`; the
        // mdat header is eight bytes.  The trun size does not change.
        let trun = build_trun(samples, trun_flags, version, Some(data_offset))?;
        let mut traf_body = Vec::new();
        traf_body.extend_from_slice(&tfhd(track_id));
        traf_body.extend_from_slice(&tfdt(samples[0].dts as u64));
        traf_body.extend_from_slice(&trun);
        let traf = box_bytes(*b"traf", &traf_body);
        let mut moof_body = Vec::new();
        moof_body.extend_from_slice(&mfhd(self.sequence));
        moof_body.extend_from_slice(&traf);
        let moof = box_bytes(*b"moof", &moof_body);

        let mut segment = styp();
        segment.extend_from_slice(&moof);
        segment.extend_from_slice(&box_bytes(*b"mdat", &payload));
        self.sequence = self.sequence.wrapping_add(1);
        Ok(Bytes::from(segment))
    }
}

fn build_trun(
    samples: &[Sample],
    flags: u32,
    version: u8,
    data_offset: Option<u32>,
) -> Result<Vec<u8>, FormatError> {
    let mut payload = Vec::new();
    push_u32(
        &mut payload,
        u32::try_from(samples.len()).map_err(|_| FormatError::IsobmffLimit("sample count"))?,
    );
    if flags & 0x000001 != 0 {
        push_u32(&mut payload, data_offset.unwrap_or(0));
    }
    for sample in samples {
        if flags & 0x000100 != 0 {
            push_u32(&mut payload, sample.duration);
        }
        if flags & 0x000200 != 0 {
            push_u32(
                &mut payload,
                u32::try_from(sample.data.len())
                    .map_err(|_| FormatError::IsobmffLimit("sample size"))?,
            );
        }
        if flags & 0x000400 != 0 {
            let sample_flags = if sample.keyframe {
                2_u32 << 24
            } else {
                (1_u32 << 24) | 0x0001_0000
            };
            push_u32(&mut payload, sample_flags);
        }
        if flags & 0x000800 != 0 {
            let offset = sample.pts - sample.dts;
            if version == 0 && offset < 0 {
                return Err(FormatError::Isobmff(
                    "negative composition offset needs trun version 1",
                ));
            }
            push_u32(&mut payload, offset as i32 as u32);
        }
    }
    Ok(full_box_bytes(*b"trun", version, flags, &payload))
}

fn tfhd(track_id: u32) -> Vec<u8> {
    let mut payload = Vec::new();
    push_u32(&mut payload, track_id);
    push_u32(&mut payload, 1);
    // default-base-is-moof | sample-description-index-present
    full_box_bytes(*b"tfhd", 0, 0x020002, &payload)
}

fn tfdt(base: u64) -> Vec<u8> {
    let mut payload = Vec::new();
    push_u64(&mut payload, base);
    full_box_bytes(*b"tfdt", 1, 0, &payload)
}

fn mfhd(sequence: u32) -> Vec<u8> {
    let mut payload = Vec::new();
    push_u32(&mut payload, sequence);
    full_box_bytes(*b"mfhd", 0, 0, &payload)
}

fn styp() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(b"cmfs");
    push_u32(&mut payload, 0);
    payload.extend_from_slice(b"cmfs");
    payload.extend_from_slice(b"iso6");
    box_bytes(*b"styp", &payload)
}

fn cmaf_trak(track: &Track) -> Result<Vec<u8>, FormatError> {
    let mut body = Vec::new();
    body.extend_from_slice(&tkhd_bytes(track, 0)?);
    let mut mdia = Vec::new();
    mdia.extend_from_slice(&mdhd_bytes(track, 0)?);
    mdia.extend_from_slice(&hdlr_bytes(track));
    mdia.extend_from_slice(&minf_with_stbl(track, &empty_stbl(track)?));
    body.extend_from_slice(&box_bytes(*b"mdia", &mdia));
    Ok(box_bytes(*b"trak", &body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tfhd_declares_moof_relative_offsets() {
        let tfhd = tfhd(7);
        assert_eq!(&tfhd[4..8], b"tfhd");
        // version/flags follow the 8-byte box header
        assert_eq!(&tfhd[8..12], &[0, 0x02, 0x00, 0x02]);
    }

    #[test]
    fn segment_type_box_is_cmaf() {
        let styp = styp();
        assert_eq!(&styp[4..8], b"styp");
        assert_eq!(&styp[8..12], b"cmfs");
    }
}
