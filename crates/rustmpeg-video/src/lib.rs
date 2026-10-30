// SPDX-License-Identifier: Apache-2.0
//! Safe Rust software codecs AV2, H.266/VVC and APV for rustmpeg-next.
//!
//! Implemented codecs:
//!
//! * [`vvc`] — an intra-only H.266/VVC encoder and matching subset decoder;
//! * [`av2`] — an experimental AV2 encoder and matching decoder foundation;
//! * [`apv`] — an intra-only RFC 9924 APV encoder and decoder.
//!
//! The remaining codecs are not implemented yet.
//!
//! The VVC module implements ITU-T H.266 in Rust; its algorithms and tables are
//! derived from the VTM 24.0 reference software (BSD-3-Clause).  See
//! `THIRD_PARTY.md` in the repository root.

#![forbid(unsafe_code)]
#![feature(portable_simd)]

pub mod accelerator;
pub mod apv;
pub mod av2;
mod color;
pub mod frame;
pub mod vvc;

pub use accelerator::{
    AcceleratorError, CpuRgbaToI420, FrameMetrics, FrameMetricsProvider, PlaneMetrics, PlaneView,
    RgbaToI420Converter,
};
pub use apv::{ApvChromaFormat, ApvConfig, ApvDecoder, ApvEncoder, ApvError, ApvFrame};
pub use av2::{Decoder as Av2Decoder, Encoder as Av2Encoder, EncoderConfig as Av2EncoderConfig};
pub use frame::I420Frame;
