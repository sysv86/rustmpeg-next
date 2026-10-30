// SPDX-License-Identifier: Apache-2.0
//! FLAC encoder example: synthetic PCM -> native FLAC plus raw reference PCM.
//!
//! Usage:
//! `cargo run -p rustmpeg-audio --example flac_encode -- out.flac [bits] [channels] [samples] [block] [level] [ogg]`
//!
//! The raw reference PCM (`out.flac.raw`) uses the same interleaved
//! little-endian layout as the FLAC MD5 input, so it can be compared with
//! `flac -d --force-raw-format --endian=little --sign=signed`.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use rustmpeg_audio::{FlacEncoderConfig, encode_flac, encode_ogg_flac};

fn signal(channels: usize, bits: u8, len: usize) -> Vec<Vec<i32>> {
    let limit = 1i64 << (bits - 1);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..channels)
        .map(|channel| {
            (0..len)
                .map(|index| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let noise = ((state >> 33) as i64) % (limit / 3).max(1);
                    let tone = ((index as f64 * 0.031 * (channel as f64 + 1.0)).sin()
                        * (limit as f64 * 0.4)) as i64;
                    (tone + noise / 2).clamp(-limit, limit - 1) as i32
                })
                .collect()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| env::temp_dir().join("out.flac").display().to_string());
    let bits: u8 = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let channels: u8 = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let samples: usize = env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(48_000);
    let block_size: u32 = env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let level: u8 = env::args().nth(6).and_then(|s| s.parse().ok()).unwrap_or(5);
    let ogg = env::args().nth(7).is_some_and(|value| value == "ogg");

    let config = FlacEncoderConfig {
        sample_rate: 48_000,
        channels,
        bits_per_sample: bits,
        block_size,
        compression_level: level,
    };
    let planes = signal(usize::from(channels), bits, samples);
    let mut file = BufWriter::new(File::create(&path)?);
    let info = if ogg {
        encode_ogg_flac(&mut file, &config, &planes)?
    } else {
        encode_flac(&mut file, &config, &planes)?
    };
    file.flush()?;

    let bytes_per_sample = usize::from(bits.div_ceil(8));
    let mut raw = BufWriter::new(File::create(format!("{path}.raw"))?);
    for index in 0..samples {
        for plane in &planes {
            let sample = plane[index];
            for byte in 0..bytes_per_sample {
                raw.write_all(&[(sample >> (8 * byte)) as u8])?;
            }
        }
    }
    raw.flush()?;

    let md5: String = info.md5.iter().map(|b| format!("{b:02x}")).collect();
    println!(
        "wrote {path}: {} ch, {} bit, {} Hz, {} samples, block {}, frames {}..{} B, md5 {md5}",
        info.channels,
        info.bits_per_sample,
        info.sample_rate,
        info.total_samples.unwrap_or(0),
        info.max_block_size,
        info.min_frame_size,
        info.max_frame_size
    );
    Ok(())
}
