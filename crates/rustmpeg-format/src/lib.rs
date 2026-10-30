// SPDX-License-Identifier: Apache-2.0
//! Container demuxers and muxers plus MoQ transport for rustmpeg-next.
//!
//! The crate provides a strict EBML primitive parser, a Matroska v4 / WebM
//! demuxer and muxer (IETF RFC 8794 and RFC 9559), and ISO base media /
//! CMAF parsing and writing (ISO/IEC 14496-12). Sample payloads are zero-copy
//! byte slices so audio and video decoders can consume them directly.

#![forbid(unsafe_code)]

mod ebml;
pub mod isobmff;
pub mod matroska;

pub use bytes::Bytes;
pub use ebml::FormatError;
pub use isobmff::{
    AudioTrack as IsoAudioTrack, CmafConfig, CmafWriter, IsobmffReader, Mp4Config, Mp4Writer,
    Sample, SampleEntry, Track as IsoTrack, TrackKind, VideoTrack as IsoVideoTrack,
};
pub use matroska::{
    AudioTrack, CodecId, DocType, MatroskaReader, MatroskaWriter, MuxerConfig, Packet, Track,
    TrackType, VideoTrack,
};
