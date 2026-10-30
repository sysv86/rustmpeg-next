// SPDX-License-Identifier: Apache-2.0
//! Sample entry and decoder configuration parsing/writing
//! (ISO/IEC 14496-12, Section 8.5; ISO/IEC 14496-15 for codec configs).

use bytes::Bytes;

use super::boxes::{Boxes, full_box_bytes, push_box, push_u16, push_u32, read_u32};
use super::{AudioTrack, SampleEntry, Track, TrackKind, VideoTrack, color_to_nclx, nclx_to_color};
use crate::FormatError;

/// Decoder configuration boxes recognised for video sample entries.
const VIDEO_CONFIG_BOXES: [[u8; 4]; 3] = [*b"vvcC", *b"apvC", *b"av2C"];
/// Decoder configuration boxes recognised for audio sample entries.
const AUDIO_CONFIG_BOXES: [[u8; 4]; 2] = [*b"dOps", *b"dfLa"];

/// Parses a sample entry box declared inside `stsd`.
pub(crate) fn parse(
    data: &[u8],
    bytes: &Bytes,
    header: &super::boxes::BoxHeader,
    kind: TrackKind,
) -> Result<(SampleEntry, Option<VideoTrack>, Option<AudioTrack>), FormatError> {
    let payload = header.payload(data);
    if payload.len() < 8 {
        return Err(FormatError::Isobmff("truncated sample entry"));
    }
    let data_reference_index = u16::from_be_bytes([payload[6], payload[7]]);
    if data_reference_index == 0 {
        return Err(FormatError::Isobmff("sample entry has no data reference"));
    }
    if matches!(&header.kind, b"encv" | b"enca" | b"drms" | b"drmi") {
        return Err(FormatError::UnsupportedSampleEntry(
            String::from_utf8_lossy(&header.kind).into_owned(),
        ));
    }

    let (children_start, video, audio) = match kind {
        TrackKind::Video => {
            if payload.len() < 78 {
                return Err(FormatError::Isobmff("truncated visual sample entry"));
            }
            let width = u16::from_be_bytes([payload[24], payload[25]]);
            let height = u16::from_be_bytes([payload[26], payload[27]]);
            if width == 0 || height == 0 {
                return Err(FormatError::Isobmff("visual sample entry has no size"));
            }
            (
                78,
                Some(VideoTrack {
                    width,
                    height,
                    color: None,
                    pixel_aspect_ratio: None,
                }),
                None,
            )
        }
        TrackKind::Audio => {
            if payload.len() < 28 {
                return Err(FormatError::Isobmff("truncated audio sample entry"));
            }
            let version = u16::from_be_bytes([payload[8], payload[9]]);
            let channels = u16::from_be_bytes([payload[16], payload[17]]);
            let sample_size = u16::from_be_bytes([payload[18], payload[19]]);
            let sample_rate = f64::from(u32::from_be_bytes([
                payload[24],
                payload[25],
                payload[26],
                payload[27],
            ])) / 65_536.0;
            let extra = match version {
                0 => 0,
                1 => 16,
                2 => 36,
                _ => {
                    return Err(FormatError::Isobmff(
                        "unsupported audio sample entry version",
                    ));
                }
            };
            let start = 28_usize
                .checked_add(extra)
                .ok_or(FormatError::IsobmffLimit("sample entry"))?;
            if start > payload.len() {
                return Err(FormatError::Isobmff("truncated audio sample entry"));
            }
            (
                start,
                None,
                Some(AudioTrack {
                    channels,
                    sample_size,
                    sample_rate,
                }),
            )
        }
        _ => (8, None, None),
    };

    let mut entry = SampleEntry::new(header.kind, None, Bytes::new());
    let mut video = video;
    let boxes = Boxes::new(
        data,
        header.payload_start + children_start,
        header.payload_end,
    )?;
    for child in boxes {
        let child = child?;
        if let Some(current) = video.as_mut() {
            match child.kind {
                [b'c', b'o', b'l', b'r'] => {
                    current.color = nclx_to_color(child.payload(data));
                }
                [b'p', b'a', b's', b'p'] => {
                    let payload = child.payload(data);
                    if payload.len() >= 8 {
                        let horizontal = read_u32(data, child.payload_start)?;
                        let vertical = read_u32(data, child.payload_start + 4)?;
                        if horizontal != 0 && vertical != 0 {
                            current.pixel_aspect_ratio = Some((horizontal, vertical));
                        }
                    }
                }
                _ => {}
            }
        }
        let known = match kind {
            TrackKind::Video => VIDEO_CONFIG_BOXES.contains(&child.kind),
            TrackKind::Audio => AUDIO_CONFIG_BOXES.contains(&child.kind),
            _ => false,
        };
        if known && entry.config_box.is_none() {
            entry.config_box = Some(child.kind);
            entry.config = bytes.slice(child.payload_start..child.payload_end);
        }
    }
    Ok((entry, video, audio))
}

/// Builds the sample entry box for a track.
pub(crate) fn build(track: &Track) -> Result<Vec<u8>, FormatError> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&[0; 6]);
    push_u16(&mut payload, 1); // data_reference_index

    match track.kind {
        TrackKind::Video => {
            let video = track.video.ok_or(FormatError::Isobmff(
                "video track has no visual sample entry",
            ))?;
            if video.width == 0 || video.height == 0 {
                return Err(FormatError::Isobmff("video track has an empty frame"));
            }
            push_u16(&mut payload, 0); // pre_defined
            push_u16(&mut payload, 0); // reserved
            payload.extend_from_slice(&[0; 12]); // pre_defined[3]
            push_u16(&mut payload, video.width);
            push_u16(&mut payload, video.height);
            push_u32(&mut payload, 0x0048_0000); // 72 dpi horizontal
            push_u32(&mut payload, 0x0048_0000); // 72 dpi vertical
            push_u32(&mut payload, 0); // reserved
            push_u16(&mut payload, 1); // frame_count
            payload.extend_from_slice(&[0; 32]); // compressorname
            push_u16(&mut payload, 0x0018); // depth
            push_u16(&mut payload, 0xffff); // pre_defined = -1
            if let Some(color) = &video.color {
                push_box(&mut payload, *b"colr", &color_to_nclx(color));
            }
            if let Some((horizontal, vertical)) = video.pixel_aspect_ratio {
                if horizontal == 0 || vertical == 0 {
                    return Err(FormatError::Isobmff("pixel aspect ratio is zero"));
                }
                let mut pasp = Vec::with_capacity(8);
                push_u32(&mut pasp, horizontal);
                push_u32(&mut pasp, vertical);
                push_box(&mut payload, *b"pasp", &pasp);
            }
        }
        TrackKind::Audio => {
            let audio = track.audio.ok_or(FormatError::Isobmff(
                "audio track has no audio sample entry",
            ))?;
            if audio.channels == 0 {
                return Err(FormatError::Isobmff("audio track has no channels"));
            }
            if !audio.sample_rate.is_finite() || audio.sample_rate <= 0.0 {
                return Err(FormatError::Isobmff("invalid audio sample rate"));
            }
            let rate = (audio.sample_rate * 65_536.0).round();
            if rate > f64::from(u32::MAX) {
                return Err(FormatError::Isobmff("audio sample rate is too large"));
            }
            push_u16(&mut payload, 0); // version
            push_u16(&mut payload, 0); // revision
            push_u32(&mut payload, 0); // vendor
            push_u16(&mut payload, audio.channels);
            push_u16(&mut payload, audio.sample_size.max(8));
            push_u16(&mut payload, 0); // pre_defined
            push_u16(&mut payload, 0); // reserved
            push_u32(&mut payload, rate as u32);
        }
        _ => {}
    }

    match (
        track.sample_entry.config_box,
        track.sample_entry.config.is_empty(),
    ) {
        (Some(kind), false) => push_box(&mut payload, kind, &track.sample_entry.config),
        (Some(kind), true) => push_box(&mut payload, kind, &[]),
        (None, false) => {
            return Err(FormatError::Isobmff(
                "decoder configuration without a configuration box",
            ));
        }
        (None, true) => {}
    }

    Ok(super::boxes::box_bytes(track.sample_entry.format, &payload))
}

/// Builds a `stsd` box containing one sample entry.
pub(crate) fn build_stsd(track: &Track) -> Result<Vec<u8>, FormatError> {
    let entry = build(track)?;
    let mut payload = Vec::with_capacity(entry.len() + 8);
    push_u32(&mut payload, 1);
    payload.extend_from_slice(&entry);
    Ok(full_box_bytes(*b"stsd", 0, 0, &payload))
}
