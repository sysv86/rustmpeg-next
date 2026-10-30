// SPDX-License-Identifier: Apache-2.0
//! Encodes interleaved s16le PCM to an LC3 binary file (liblc3 `lc3bin`).
//!
//! Usage: `cargo run -p rustmpeg-audio --example lc3_encode -- <in.s16le> <out.lc3> <sample_rate> <7500|10000> <channels> <nbytes>`
//!
//! The output uses the liblc3 `lc3bin` container: an 18-byte little-endian
//! header followed by `u16` frame sizes and the LC3 payloads. It can be read
//! back with the `lc3_decode` example.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Read, Write};

use rustmpeg_audio::{Lc3Config, Lc3Encoder, Lc3FrameDuration, PcmFrame};

/// Writes the little-endian `lc3bin` header.
fn write_header<W: Write>(
    out: &mut W,
    sample_rate: u32,
    channels: u8,
    frame_us: u32,
    samples: u32,
) -> std::io::Result<()> {
    let fields: [u16; 9] = [
        0xcc1c,
        18,
        (sample_rate / 100) as u16,
        0,
        u16::from(channels),
        (frame_us / 10) as u16,
        0,
        samples as u16,
        (samples >> 16) as u16,
    ];
    for field in fields {
        out.write_all(&field.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let usage =
        "usage: lc3_encode <in.s16le> <out.lc3> <sample_rate> <7500|10000> <channels> <nbytes>";
    let input = args.next().ok_or(usage)?;
    let output = args.next().ok_or(usage)?;
    let sample_rate: u32 = args.next().ok_or(usage)?.parse()?;
    let frame_us: u32 = args.next().ok_or(usage)?.parse()?;
    let channels: u8 = args.next().ok_or(usage)?.parse()?;
    let nbytes: usize = args.next().ok_or(usage)?.parse()?;

    let frame_duration = match frame_us {
        7500 => Lc3FrameDuration::Ms7_5,
        10000 => Lc3FrameDuration::Ms10,
        other => return Err(format!("unsupported LC3 frame duration {other} us").into()),
    };
    let config = Lc3Config {
        sample_rate,
        frame_duration,
        channels,
        bit_depth: 16,
    };
    let mut encoder = Lc3Encoder::new(config)?;
    let ns = config.frame_samples();

    let mut data = Vec::new();
    File::open(&input)?.read_to_end(&mut data)?;
    let interleaved: Vec<i32> = data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i32::from(i16::from_le_bytes(*pair)))
        .collect();
    let per_channel = interleaved.len() / usize::from(channels);
    let frames = per_channel / ns;
    let dropped = per_channel - frames * ns;

    let mut out = BufWriter::new(File::create(&output)?);
    write_header(
        &mut out,
        sample_rate,
        channels,
        frame_us,
        (frames * ns) as u32,
    )?;

    for index in 0..frames {
        let mut planes = vec![Vec::with_capacity(ns); usize::from(channels)];
        let base = index * ns * usize::from(channels);
        for sample in 0..ns {
            for (channel, plane) in planes.iter_mut().enumerate() {
                plane.push(interleaved[base + sample * usize::from(channels) + channel]);
            }
        }
        let frame = PcmFrame {
            sample_rate,
            channels,
            bits_per_sample: 16,
            frame_number: index as u64,
            sample_number: (index * ns) as u64,
            planes,
        };
        let payload = encoder.encode_frame(&frame, nbytes)?;
        out.write_all(&(payload.len() as u16).to_le_bytes())?;
        out.write_all(&payload)?;
    }
    out.flush()?;

    println!(
        "encoded {frames} LC3 frames, {sample_rate} Hz, {channels} ch, {nbytes} bytes; dropped {dropped} trailing frames"
    );
    Ok(())
}
