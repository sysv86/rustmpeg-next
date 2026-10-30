// SPDX-License-Identifier: Apache-2.0
//! Minimal APV decoder example: raw APV bitstream -> planar 16-bit samples.
//!
//! Usage: `cargo run -p rustmpeg-video --example apv_decode -- in.apv out.raw`
//!
//! The output is little-endian `u16` planes in component order, without a
//! container header.  It matches FFmpeg's raw `yuv422p10le`/`yuv422p12le`
//! layout for the same stream.

use std::env;
use std::fs;

use rustmpeg_video::apv::ApvDecoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = env::args().nth(1).unwrap_or_else(|| {
        env::temp_dir()
            .join("apv-example.apv")
            .display()
            .to_string()
    });
    let output = env::args().nth(2).unwrap_or_else(|| {
        env::temp_dir()
            .join("apv-example.raw")
            .display()
            .to_string()
    });
    let data = fs::read(&input)?;
    let frames = ApvDecoder::new().decode_raw_bitstream(&data)?;
    let mut bytes = Vec::new();
    for frame in &frames {
        for plane in &frame.planes {
            for sample in plane {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
        }
    }
    fs::write(&output, &bytes)?;
    let first = frames.first().ok_or("no frames")?;
    println!(
        "decoded {} frame(s): {}x{} {}, {}-bit, {} bytes",
        frames.len(),
        first.width,
        first.height,
        first.chroma_format,
        first.bit_depth,
        bytes.len()
    );
    Ok(())
}
