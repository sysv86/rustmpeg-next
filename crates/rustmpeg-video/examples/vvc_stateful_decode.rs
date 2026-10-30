// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! Decode an Annex-B H.266/VVC bitstream file (`<input.vvc>`) using the
//! stateful [`VvcDecoder`] and write the reconstructed YUV frames (`8`, `10`,
//! or `12`-bit planar `4:0:0`..=`4:4:4`) to `<output.yuv>`.
//!
//! Usage:
//!   `cargo run -p rustmpeg-video --example vvc_stateful_decode -- <input.vvc> <output.yuv> [--apply-film-grain 0|1]`

#![forbid(unsafe_code)]

use std::env;
use std::error::Error;
use std::fs;
use std::process::ExitCode;

use rustmpeg_video::vvc::{VvcDecoder, VvcFrame, VvcNalType, parse_annexb_nals};

/// Split a concatenated Annex-B bitstream into per-access-unit slices.
///
/// Splits at `NAL_AUD` (`20`) boundaries when present, or after each VCL picture
/// boundary when AUDs are omitted (e.g. VTM / VVEnc external bitstreams).
fn split_access_units(data: &[u8]) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let nals = parse_annexb_nals(data)?;
    let mut aus: Vec<Vec<u8>> = Vec::new();
    let mut cur_au: Vec<u8> = Vec::new();
    let mut seen_vcl_in_au = false;

    for nal in nals {
        let is_vcl = nal.nal_type.code() <= 11;
        let starts_new_au = nal.nal_type == VvcNalType::Aud
            || (seen_vcl_in_au
                && !is_vcl
                && nal.nal_type != VvcNalType::SuffixAps
                && nal.nal_type != VvcNalType::SuffixSei);
        if starts_new_au && !cur_au.is_empty() {
            aus.push(std::mem::take(&mut cur_au));
            seen_vcl_in_au = false;
        }
        cur_au.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        cur_au.extend_from_slice(&nal.raw);
        if is_vcl {
            seen_vcl_in_au = true;
        }
    }
    if !cur_au.is_empty() {
        aus.push(cur_au);
    }
    Ok(aus)
}

fn append_vvc_frame_raw(out: &mut Vec<u8>, frame: &VvcFrame) {
    let write_plane = |dst: &mut Vec<u8>, data: &[u16], bd: u8| {
        if bd <= 8 {
            dst.extend(data.iter().map(|&v| v.min(255) as u8));
        } else {
            for &v in data {
                dst.extend_from_slice(&v.to_le_bytes());
            }
        }
    };
    write_plane(out, &frame.y.data, frame.bit_depth);
    if frame.chroma_format.is_chroma_enabled() {
        write_plane(out, &frame.u.data, frame.bit_depth);
        write_plane(out, &frame.v.data, frame.bit_depth);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        return Err(
            "usage: vvc_stateful_decode <input.vvc> <output.yuv> [--apply-film-grain 0|1]".into(),
        );
    }
    let input_path = &args[0];
    let output_path = &args[1];
    let mut idx = 2usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--apply-film-grain" => {
                idx += 1;
                let _ = args.get(idx).ok_or("missing --apply-film-grain value")?;
            }
            other => return Err(format!("unknown flag `{other}`").into()),
        }
        idx += 1;
    }

    let data = fs::read(input_path)?;
    let aus = split_access_units(&data)?;
    let refs: Vec<&[u8]> = aus.iter().map(|a| a.as_slice()).collect();

    let mut decoder = VvcDecoder::new();
    let frames = decoder.decode_annexb_stream_vvc_frames(&refs)?;

    let mut out = Vec::new();
    for frame in &frames {
        append_vvc_frame_raw(&mut out, frame);
    }
    fs::write(output_path, &out)?;
    println!(
        "decoded {} frames from {input_path} -> {output_path} ({} bytes)",
        frames.len(),
        out.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    if let Err(err) = run() {
        eprintln!("vvc_stateful_decode: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
