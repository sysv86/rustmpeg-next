// SPDX-License-Identifier: Apache-2.0
//! Core primitives for rustmpeg-next.
//!
//! This crate is the bottom of the dependency graph. It is `no_std` + `alloc`,
//! runtime-agnostic, and free of `unsafe`. It provides:
//!
//! * [`Buffer`] — immutable `Arc<[u8]>` buffers with zero-copy slicing;
//! * [`MediaTime`], [`MediaDuration`], [`Timebase`], and [`FrameRate`] — exact
//!   rational time arithmetic;
//! * [`PixelFormat`], [`ColorDescription`], [`VideoFrame`], and
//!   [`AudioFormat`]/[`AudioFrame`] — shared media layout descriptions;
//! * [`Packet`] and [`PacketFlags`] — encoded payloads with timing metadata;
//! * [`CoreError`] — the typed base error;
//! * [`sad_u8`] and [`sse_u8`] — portable SIMD distortion primitives.
//!
//! Codec, container, and GPU crates depend on these types and must not redefine
//! them locally.

#![no_std]
#![forbid(unsafe_code)]
#![feature(portable_simd)]

extern crate alloc;

pub mod audio;
pub mod buffer;
pub mod color;
pub mod error;
pub mod frame;
pub mod limits;
pub mod packet;
pub mod pixel;
pub mod simd;
pub mod time;

pub use audio::{AudioFormat, AudioFrame, MAX_CHANNELS, SampleFormat};
pub use buffer::Buffer;
pub use color::{
    ChromaLocation, ColorDescription, ColorPrimaries, ColorRange, MatrixCoefficients,
    TransferFunction,
};
pub use error::{CoreError, Result};
pub use frame::{Plane, PlaneRef, VideoFrame, VideoFrameView, packed_plane};
pub use limits::{
    MAX_DIMENSION, MAX_LUMA_SAMPLES, checked_add, checked_mul, plane_bytes, validate_dimensions,
    validate_stride,
};
pub use packet::{Packet, PacketFlags};
pub use pixel::PixelFormat;
pub use simd::{clamp_u8, sad_u8, sse_u8};
pub use time::{FrameRate, MediaDuration, MediaTime, Rational, Timebase};
