// SPDX-License-Identifier: Apache-2.0
//! Decodes an LC3 binary file (liblc3 `lc3bin` format) to interleaved s16le.
//!
//! Usage: `cargo run -p rustmpeg-audio --example lc3_decode -- in.lc3 out.s16le`

use std::env;
use std::fs::File;
use std::io::{BufWriter, Read, Write};

use rustmpeg_audio::{Lc3Config, Lc3Decoder, Lc3FrameDuration};

fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input = args
        .next()
        .ok_or("usage: lc3_decode <input.lc3> <output.s16le>")?;
    let output = args
        .next()
        .ok_or("usage: lc3_decode <input.lc3> <output.s16le>")?;
    let mut data = Vec::new();
    File::open(&input)?.read_to_end(&mut data)?;

    if data.len() < 18 || read_u16(&data, 0) != 0xcc1c {
        return Err("not an LC3 binary file".into());
    }
    let header_size = usize::from(read_u16(&data, 2));
    let sample_rate = u32::from(read_u16(&data, 4)) * 100;
    let channels = read_u16(&data, 8) as u8;
    let frame_us = u32::from(read_u16(&data, 10)) * 10;
    let duration = match frame_us {
        7500 => Lc3FrameDuration::Ms7_5,
        10000 => Lc3FrameDuration::Ms10,
        other => return Err(format!("unsupported LC3 frame duration {other} us").into()),
    };
    let config = Lc3Config {
        sample_rate,
        frame_duration: duration,
        channels,
        bit_depth: 16,
    };
    let mut decoder = Lc3Decoder::new(config)?;
    let mut pos = header_size.max(18);
    let mut out = BufWriter::new(File::create(&output)?);
    let mut frames = 0usize;
    while pos + 2 <= data.len() {
        let nbytes = usize::from(read_u16(&data, pos));
        pos += 2;
        if pos + nbytes > data.len() {
            return Err("truncated LC3 frame".into());
        }
        let frame = decoder.decode_frame(&data[pos..pos + nbytes])?;
        pos += nbytes;
        for index in 0..frame.block_size() {
            for plane in &frame.planes {
                out.write_all(&(plane[index] as i16).to_le_bytes())?;
            }
        }
        frames += 1;
    }
    out.flush()?;
    println!("decoded {frames} LC3 frames, {sample_rate} Hz, {channels} ch, {duration:?}");
    Ok(())
}
