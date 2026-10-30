// SPDX-License-Identifier: Apache-2.0
//! Opus, FLAC, LPCM, LC3 and IAMF audio stack for rustmpeg-next.
//!
//! The audio stage provides a full-cycle FLAC implementation (native and Ogg
//! encode/decode, MD5, seeking), an LC3 decoder, raw LPCM codecs, and the
//! bitstream and entropy layers of Opus.  See [`flac`] for FLAC, [`pcm`] for LPCM, [`opus`] for Opus
//! packet parsing and range decoding, and [`PcmFrame`] for the planar output
//! representation.

#![forbid(unsafe_code)]

pub mod flac;
mod frame;
pub mod lc3;
pub mod opus;
pub mod pcm;

pub use flac::{
    FlacDecoder, FlacEncoder, FlacEncoderConfig, FlacError, OggFlacDecoder, StreamInfo,
    encode_flac, encode_flac_to_vec, encode_ogg_flac, encode_ogg_flac_to_vec,
};
pub use frame::PcmFrame;
pub use lc3::{Lc3Config, Lc3Decoder, Lc3Encoder, Lc3Error, Lc3FrameDuration};
pub use opus::{OpusBandwidth, OpusError, OpusMode, OpusPacket, OpusPacketInfo, RangeDecoder};
pub use pcm::{FloatFrame, PcmData, PcmError, PcmFormat, PcmParams, decode_pcm, encode_pcm};
