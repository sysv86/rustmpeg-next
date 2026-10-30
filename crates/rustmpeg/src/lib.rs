// SPDX-License-Identifier: Apache-2.0
//! Convenience entry point for the rustmpeg-next multimedia framework.
//!
//! This crate is a thin facade. It re-exports the workspace crates so an
//! application can depend on `rustmpeg` and reach every component through one
//! dependency:
//!
//! * [`core`] — no_std buffers, time, pixel/colour/frame formats, packets, and
//!   typed errors;
//! * [`format`] — ISOBMFF/CMAF and Matroska/WebM container support;
//! * [`audio`] — Opus, FLAC, LPCM, and LC3 audio;
//! * [`image`] — JPEG XL and OpenEXR;
//! * [`video`] — AV2, H.266/VVC, and APV video codecs;
//! * [`vulkan`] — Vulkan Compute kernels and the CPU/GPU scheduler.
//!
//! The facade contains no implementation of its own and defines no new types.
//! It is not a second implementation of the framework, and it adds no behavior
//! beyond the re-exports below. `rustmpeg-cli` is a binary-only crate and is
//! therefore not re-exported here; install it separately with
//! `cargo install rustmpeg-cli`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub use rustmpeg_audio as audio;
pub use rustmpeg_core as core;
pub use rustmpeg_format as format;
pub use rustmpeg_image as image;
pub use rustmpeg_video as video;
pub use rustmpeg_vulkan as vulkan;
