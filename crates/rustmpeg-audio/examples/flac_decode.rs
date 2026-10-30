// SPDX-License-Identifier: Apache-2.0
//! Decodes a native or Ogg FLAC file to left-aligned interleaved 32-bit
//! little-endian PCM and prints the STREAMINFO fields.
//!
//! Usage: `cargo run -p rustmpeg-audio --example flac_decode -- input.(flac|oga) output.s32le`

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use rustmpeg_audio::{FlacDecoder, FlacError, OggFlacDecoder, PcmFrame, StreamInfo};

trait FlacSource {
    fn info(&self) -> &StreamInfo;
    fn next(&mut self) -> Result<Option<PcmFrame>, FlacError>;
}

impl<R: std::io::Read> FlacSource for FlacDecoder<R> {
    fn info(&self) -> &StreamInfo {
        self.stream_info()
    }
    fn next(&mut self) -> Result<Option<PcmFrame>, FlacError> {
        self.next_frame()
    }
}

impl<R: std::io::Read> FlacSource for OggFlacDecoder<R> {
    fn info(&self) -> &StreamInfo {
        self.stream_info()
    }
    fn next(&mut self) -> Result<Option<PcmFrame>, FlacError> {
        self.next_frame()
    }
}

fn decode<D: FlacSource>(
    mut decoder: D,
    mut output: BufWriter<File>,
) -> Result<(), Box<dyn std::error::Error>> {
    let info = decoder.info();
    println!(
        "stream: {} Hz, {} channel(s), {}-bit, {} total samples, MD5 {}",
        info.sample_rate,
        info.channels,
        info.bits_per_sample,
        info.total_samples
            .map_or_else(|| "unknown".to_owned(), |total| total.to_string()),
        hex_md5(&info.md5)
    );

    let mut frames = 0u64;
    let mut samples = 0u64;
    let mut buffer = Vec::new();
    while let Some(frame) = decoder.next()? {
        write_frame(&frame, &mut buffer)?;
        output.write_all(&buffer)?;
        frames += 1;
        samples += frame.block_size() as u64;
    }
    output.flush()?;
    println!("decoded {frames} frame(s), {samples} sample(s) per channel");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input_path = args
        .next()
        .ok_or("usage: flac_decode <input.flac|input.oga> <output.s32le>")?;
    let output_path = args
        .next()
        .ok_or("usage: flac_decode <input.flac|input.oga> <output.s32le>")?;
    if args.next().is_some() {
        return Err("usage: flac_decode <input.flac|input.oga> <output.s32le>".into());
    }

    let mut input = BufReader::new(File::open(&input_path)?);
    let ogg = input.fill_buf()?.starts_with(b"OggS");
    let output = BufWriter::new(File::create(&output_path)?);
    if ogg {
        decode(OggFlacDecoder::new(input)?, output)
    } else {
        decode(FlacDecoder::new(input)?, output)
    }
}

/// Serializes one frame as interleaved native samples shifted into the most
/// significant bits of an `i32`.
fn write_frame(frame: &PcmFrame, buffer: &mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
    let block_size = frame.block_size();
    let shift = 32 - u32::from(frame.bits_per_sample);
    buffer.clear();
    buffer.reserve(
        block_size
            .saturating_mul(usize::from(frame.channels))
            .saturating_mul(4),
    );
    for sample_index in 0..block_size {
        for channel in 0..usize::from(frame.channels) {
            let plane = frame
                .plane(channel)
                .ok_or("frame has fewer planes than channels")?;
            let sample = *plane
                .get(sample_index)
                .ok_or("frame plane is shorter than the block size")?;
            let shifted = sample
                .checked_shl(shift)
                .ok_or("sample does not fit into a left-aligned i32")?;
            buffer.extend_from_slice(&shifted.to_le_bytes());
        }
    }
    Ok(())
}

fn hex_md5(md5: &[u8; 16]) -> String {
    let mut text = String::with_capacity(32);
    for byte in md5 {
        use std::fmt::Write as _;
        let _ = write!(text, "{byte:02x}");
    }
    text
}
