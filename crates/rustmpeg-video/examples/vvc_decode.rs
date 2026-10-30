// SPDX-License-Identifier: Apache-2.0
//! Decode an H.266/VVC Annex-B stream with `rustmpeg-video::vvc` and write I420.
//!
//! Original rustmpeg-next example, used for external conformance checks.
//!
//! Usage: `vvc_decode <input.vvc> <output.yuv>`

use std::fs;

use rustmpeg_video::vvc::decode_annexb_to_i420;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).ok_or("missing input")?;
    let output = args.get(2).ok_or("missing output")?;
    let data = fs::read(input)?;
    let frames = decode_annexb_to_i420(&[data.as_slice()])?;
    let mut yuv = Vec::new();
    for frame in &frames {
        yuv.extend_from_slice(&frame.y);
        yuv.extend_from_slice(&frame.u);
        yuv.extend_from_slice(&frame.v);
    }
    fs::write(output, &yuv)?;
    eprintln!(
        "decoded {} frames ({} bytes) to {}",
        frames.len(),
        yuv.len(),
        output
    );
    Ok(())
}
