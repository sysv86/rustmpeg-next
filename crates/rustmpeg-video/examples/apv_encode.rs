// SPDX-License-Identifier: Apache-2.0
//! Minimal APV encoder example: one synthetic frame -> raw APV bitstream.
//!
//! Usage:
//! `cargo run -p rustmpeg-video --example apv_encode -- out.apv [width] [height] [400|422|444|4444] [bit_depth] [qp]`

use std::env;
use std::fs;

use rustmpeg_video::apv::{ApvChromaFormat, ApvConfig, ApvEncoder, ApvFrame};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).unwrap_or_else(|| {
        env::temp_dir()
            .join("apv-example.apv")
            .display()
            .to_string()
    });
    let width: u32 = env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let height: u32 = env::args()
        .nth(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(48);
    let chroma = match env::args().nth(4).as_deref().unwrap_or("422") {
        "400" | "mono" => ApvChromaFormat::Mono,
        "422" => ApvChromaFormat::Yuv422,
        "444" => ApvChromaFormat::Yuv444,
        "4444" => ApvChromaFormat::Yuv4444,
        other => return Err(format!("unknown chroma layout {other}").into()),
    };
    let bit_depth: u8 = env::args()
        .nth(5)
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);
    let qp: u8 = env::args()
        .nth(6)
        .and_then(|value| value.parse().ok())
        .unwrap_or(18);

    let mut frame = ApvFrame::new(chroma, bit_depth, width, height)?;
    let maximum = (1_u32 << bit_depth) - 1;
    let sizes: Vec<_> = (0..frame.planes.len())
        .map(|component| frame.plane_size(component).expect("plane"))
        .collect();
    for (component, plane) in frame.planes.iter_mut().enumerate() {
        let (plane_width, plane_height) = sizes[component];
        for y in 0..plane_height {
            for x in 0..plane_width {
                let value = (x * 17 + y * 13 + component as u32 * 29) % (maximum + 1);
                plane[(y * plane_width + x) as usize] = value as u16;
            }
        }
    }

    let encoder = ApvEncoder::new(ApvConfig {
        qp,
        ..ApvConfig::default()
    })?;
    let bitstream = encoder.encode_raw_bitstream(&frame)?;
    fs::write(&path, &bitstream)?;
    println!(
        "wrote {} bytes: {}x{} {}, {}-bit, qp {}",
        bitstream.len(),
        width,
        height,
        chroma,
        bit_depth,
        qp
    );
    Ok(())
}
