// SPDX-License-Identifier: Apache-2.0
//! Procedural audio generator example: seed -> FLAC, with optional WAV/PCM.
//!
//! Usage:
//! `cargo run -p rustmpeg-audio --release --example flac_generate -- out.flac [options]`
//!
//! Options:
//! - `--seed <number|random>` (default `1`)
//! - `--family <pad|pluck|percussion|glitch|ambient|sweep|chord|auto>`
//!   (default `auto`, chosen from the seed)
//! - `--duration <seconds>` (default `4`)
//! - `--rate <44100|48000>` (default `48000`)
//! - `--bits <16|24>` (default `16`)
//! - `--level <0..8>` FLAC compression level (default `8`)
//! - `--wav <path>` also write a RIFF/WAVE file
//! - `--pcm <path>` also write interleaved little-endian PCM
//! - `--list-families` print the family names and exit
//!
//! The generated sample data is deterministic for a fixed seed, family, rate,
//! bit depth, and duration. `--seed random` prints the chosen seed so the clip
//! can be reproduced later.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use rustmpeg_audio::synth::{Family, SynthConfig, render};
use rustmpeg_audio::{FlacEncoderConfig, encode_flac};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        env::temp_dir()
            .join("rustmpeg_generate.flac")
            .display()
            .to_string()
    });

    let mut seed_arg = String::from("1");
    let mut family_name = String::from("auto");
    let mut duration = 4.0f64;
    let mut rate = 48_000u32;
    let mut bits = 16u8;
    let mut level = 8u8;
    let mut wav_path: Option<String> = None;
    let mut pcm_path: Option<String> = None;

    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--seed" => seed_arg = value()?,
            "--family" => family_name = value()?,
            "--duration" => duration = value()?.parse()?,
            "--rate" => rate = value()?.parse()?,
            "--bits" => bits = value()?.parse()?,
            "--level" => level = value()?.parse()?,
            "--wav" => wav_path = Some(value()?),
            "--pcm" => pcm_path = Some(value()?),
            "--list-families" => {
                for family in Family::ALL {
                    println!("{}", family.as_str());
                }
                return Ok(());
            }
            other => return Err(format!("unknown option {other}").into()),
        }
    }

    let seed = if seed_arg.eq_ignore_ascii_case("random") {
        random_seed()
    } else {
        seed_arg.parse()?
    };
    let family = if family_name == "auto" {
        Family::from_seed(seed)
    } else {
        Family::from_name(&family_name).ok_or_else(|| format!("unknown family {family_name}"))?
    };
    let config = SynthConfig {
        seed,
        family,
        sample_rate: rate,
        bits_per_sample: bits,
        duration_seconds: duration,
        channels: 2,
    };
    let output = render(&config)?;

    let flac_config = FlacEncoderConfig {
        sample_rate: output.sample_rate,
        channels: 2,
        bits_per_sample: output.bits_per_sample,
        block_size: 4096,
        compression_level: level,
    };
    let file = BufWriter::new(File::create(&path)?);
    let info = encode_flac(file, &flac_config, &output.planes)?;

    if let Some(wav) = wav_path {
        std::fs::write(&wav, output.wav_bytes())?;
    }
    if let Some(pcm) = pcm_path {
        let mut file = BufWriter::new(File::create(&pcm)?);
        file.write_all(&output.interleaved_bytes())?;
        file.flush()?;
    }

    println!(
        "seed {seed} family {} -> {path}: {} ms, {} Hz, {} bit stereo, {} samples, max frame {} B",
        family.as_str(),
        (duration * 1000.0).round() as u64,
        output.sample_rate,
        output.bits_per_sample,
        output.frames(),
        info.max_frame_size
    );
    println!("{}", output.description);
    Ok(())
}

fn random_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ (u64::from(std::process::id()) << 32)
}
