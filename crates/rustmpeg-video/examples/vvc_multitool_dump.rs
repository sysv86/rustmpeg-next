// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//! Encode a synthetic sequence with [`VvcMultiToolEncoder`] (Intra + Inter P/B
//! GOP, 67 intra modes, CCLM, MIP, MRL, ISP, BDPCM, Palette, IBC, ACT, DepQuant,
//! SDH, JCCR, and in-loop Deblock/SAO/ALF/CCALF/LMCS) and dump the Annex-B
//! bitstream together with the raw source and self-decoded reconstruction.
//!
//! Usage:
//!   `cargo run -p rustmpeg-video --example vvc_multitool_dump -- <width> <height> <qp> <frames> <keyint> <prefix> [flags]`
//!
//! Writes `<prefix>.vvc`, `<prefix>.source.yuv` and `<prefix>.recon.yuv`.

#![forbid(unsafe_code)]

use std::env;
use std::error::Error;
use std::fs;
use std::process::ExitCode;

use rustmpeg_video::vvc::{
    ChromaFormatIdc, SliceCodingOptions, VvcDecoder, VvcEncoderConfig, VvcEncoderOptions, VvcFrame,
    VvcMultiToolEncoder, VvcRateControlMode,
};

fn scene_vvc_frame(
    width: u32,
    height: u32,
    bit_depth: u8,
    chroma_format: ChromaFormatIdc,
    frame_index: u32,
) -> VvcFrame {
    let w = width as usize;
    let h = height as usize;
    let bd = bit_depth.clamp(8, 12);
    let shift = u32::from(bd - 8);
    let mut frame = VvcFrame::new(width, height, bd, chroma_format);
    let bar_x = ((frame_index as usize) * 8) % w.max(1);
    let bar_w = (w / 6).max(2);

    for j in 0..h {
        for i in 0..w {
            let in_bar = i >= bar_x && i < (bar_x + bar_w).min(w);
            let y8: u16 = if in_bar {
                235
            } else {
                ((i * 3 + j * 5 + frame_index as usize * 7) % 220) as u16 + 16
            };
            frame.y.set(i, j, y8 << shift);
        }
    }
    if chroma_format.is_chroma_enabled() {
        let cw = frame.u.width;
        let ch = frame.u.height;
        for j in 0..ch {
            for i in 0..cw {
                let u8_v: u16 = (96 + ((i + frame_index as usize * 3) % 64) as u16).min(240);
                let v8_v: u16 = 160u16.saturating_sub(((j + frame_index as usize * 2) % 48) as u16);
                frame.u.set(i, j, u8_v << shift);
                frame.v.set(i, j, v8_v << shift);
            }
        }
    }
    frame
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

fn parse_chroma_format(arg: &str) -> Result<ChromaFormatIdc, String> {
    match arg {
        "400" | "0" => Ok(ChromaFormatIdc::Mono400),
        "420" | "1" => Ok(ChromaFormatIdc::Chroma420),
        "422" | "2" => Ok(ChromaFormatIdc::Chroma422),
        "444" | "3" => Ok(ChromaFormatIdc::Chroma444),
        other => Err(format!("unsupported --chroma-format `{other}`")),
    }
}

fn parse_ctu_log2(arg: &str) -> Result<u8, String> {
    match arg {
        "16" => Err(
            "unsupported --ctu-size `16`: CTU size 16 cannot be signalled \
             (sps_log2_ctu_size_minus5, H.266 §7.3.2.4); use 32, 64, or 128"
                .to_string(),
        ),
        "32" => Ok(5),
        "64" => Ok(6),
        "128" => Ok(7),
        other => Err(format!(
            "unsupported --ctu-size `{other}` (expected 32|64|128)"
        )),
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 6 {
        return Err(
            "usage: vvc_multitool_dump <width> <height> <qp> <frames> <keyint> <prefix> [flags]"
                .into(),
        );
    }
    let width: u32 = args[0].parse()?;
    let height: u32 = args[1].parse()?;
    let qp: u8 = args[2].parse()?;
    let frames: u32 = args[3].parse()?;
    let keyint: u32 = args[4].parse()?;
    let prefix = &args[5];

    let mut bit_depth: u8 = 8;
    let mut chroma_format = ChromaFormatIdc::Chroma420;
    let mut ctu_log2: u8 = 5;
    let mut b_frames: u8 = 0;
    let mut rc_mode: Option<String> = None;
    let mut bitrate_kbps: u32 = 1000;
    let mut emit_film_grain = false;

    let mut idx = 6usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--bit-depth" => {
                idx += 1;
                bit_depth = args.get(idx).ok_or("missing --bit-depth")?.parse()?;
            }
            "--chroma-format" => {
                idx += 1;
                chroma_format =
                    parse_chroma_format(args.get(idx).ok_or("missing --chroma-format")?)?;
            }
            "--ctu-size" => {
                idx += 1;
                ctu_log2 = parse_ctu_log2(args.get(idx).ok_or("missing --ctu-size")?)?;
            }
            "--b-frames" => {
                idx += 1;
                b_frames = args.get(idx).ok_or("missing --b-frames")?.parse()?;
            }
            "--rate-control" => {
                idx += 1;
                rc_mode = Some(args.get(idx).ok_or("missing --rate-control")?.clone());
            }
            "--bitrate-kbps" => {
                idx += 1;
                bitrate_kbps = args.get(idx).ok_or("missing --bitrate-kbps")?.parse()?;
            }
            "--tiles" | "--slices" | "--wpp" | "--rpr" => {
                idx += 1;
                let _ = args.get(idx).ok_or("missing flag argument")?;
            }
            "--film-grain" => {
                idx += 1;
                let v = args.get(idx).ok_or("missing --film-grain")?;
                emit_film_grain = v == "1" || v.eq_ignore_ascii_case("true");
            }
            other => return Err(format!("unknown flag `{other}`").into()),
        }
        idx += 1;
    }

    let intra_only = env::var("VVC_MT_INTRA_ONLY").is_ok_and(|v| v == "1");
    let tool_list = env::var("VVC_MT_TOOLS").unwrap_or_else(|_| "all".to_string());
    let mut slice_options = SliceCodingOptions::default();
    for tool in tool_list.split(',').map(str::trim) {
        match tool {
            "all" => {
                slice_options = SliceCodingOptions {
                    full_intra_modes: true,
                    enable_cclm: true,
                    enable_mip: true,
                    dep_quant: true,
                    sign_data_hiding: true,
                    enable_jccr: true,
                    enable_deblock: true,
                    enable_sao: true,
                    enable_alf: true,
                    ..SliceCodingOptions::default()
                };
            }
            "none" | "" => {}
            "modes" => slice_options.full_intra_modes = true,
            "cclm" => slice_options.enable_cclm = true,
            "mip" => slice_options.enable_mip = true,
            "mrl" => slice_options.enable_mrl = true,
            "isp" => slice_options.enable_isp = true,
            "bdpcm" => slice_options.enable_bdpcm = true,
            "palette" => slice_options.enable_palette = true,
            "ibc" => slice_options.enable_ibc = true,
            "act" => slice_options.enable_act = true,
            "depquant" => slice_options.dep_quant = true,
            "sdh" => slice_options.sign_data_hiding = true,
            "jccr" => slice_options.enable_jccr = true,
            "deblock" => slice_options.enable_deblock = true,
            "sao" => slice_options.enable_sao = true,
            "alf" => slice_options.enable_alf = true,
            "ccalf" => slice_options.enable_ccalf = true,
            "lmcs" => slice_options.enable_lmcs = true,
            "gpm" | "affine" | "prof" | "ciip" | "bdof" | "dmvr" | "mmvd" | "amvr" | "bcw"
            | "tmvp" | "rpr" => {}
            other => eprintln!("unknown tool `{other}` in VVC_MT_TOOLS"),
        }
    }

    let rate_control = match rc_mode.as_deref() {
        Some("cqp") => Some(VvcRateControlMode::Cqp { base_qp: qp }),
        Some("vbr") => Some(VvcRateControlMode::Vbr {
            target_kbps: bitrate_kbps,
            fps_num: 30,
            fps_den: 1,
            init_qp: qp,
        }),
        _ => None,
    };

    let options = VvcEncoderOptions {
        config: VvcEncoderConfig { width, height, qp },
        keyint: if intra_only { 1 } else { keyint.max(1) },
        slice_options,
        bit_depth,
        chroma_format,
        ctu_log2,
        b_frames: if intra_only { 0 } else { b_frames },
        rate_control,
        emit_film_grain_sei: emit_film_grain,
    };
    let mut encoder = VvcMultiToolEncoder::new(options)?;
    let decoder_options = *encoder.slice_options();
    let mut decoder = VvcDecoder::with_options(decoder_options);

    let mut aus: Vec<Vec<u8>> = Vec::new();
    let static_scene = env::var("VVC_MT_STATIC").is_ok_and(|v| v == "1");
    let mut source = Vec::new();
    let mut recon = Vec::new();

    if bit_depth == 8 && chroma_format == ChromaFormatIdc::Chroma420 && ctu_log2 == 5 {
        for index in 0..frames {
            let vf = scene_vvc_frame(
                width,
                height,
                8,
                ChromaFormatIdc::Chroma420,
                if static_scene { 0 } else { index },
            );
            let frame = vf.to_i420();
            source.extend_from_slice(&frame.y);
            source.extend_from_slice(&frame.u);
            source.extend_from_slice(&frame.v);
            let au = encoder.encode_i420(&frame)?;
            aus.push(au.data);
        }
        let refs: Vec<&[u8]> = aus.iter().map(|a| a.as_slice()).collect();
        let decoded = decoder.decode_annexb_stream(&refs)?;
        for frame in &decoded {
            recon.extend_from_slice(&frame.y);
            recon.extend_from_slice(&frame.u);
            recon.extend_from_slice(&frame.v);
        }
    } else {
        for index in 0..frames {
            let vf = scene_vvc_frame(
                width,
                height,
                bit_depth,
                chroma_format,
                if static_scene { 0 } else { index },
            );
            append_vvc_frame_raw(&mut source, &vf);
            let au = encoder.encode_vvc_frame(&vf)?;
            aus.push(au.data);
        }
        let refs: Vec<&[u8]> = aus.iter().map(|a| a.as_slice()).collect();
        let decoded = decoder.decode_annexb_stream_vvc_frames(&refs)?;
        for vf in &decoded {
            append_vvc_frame_raw(&mut recon, vf);
        }
    }

    let stream: Vec<u8> = aus.concat();
    fs::write(format!("{prefix}.vvc"), &stream)?;
    fs::write(format!("{prefix}.source.yuv"), &source)?;
    fs::write(format!("{prefix}.recon.yuv"), &recon)?;
    println!(
        "wrote {prefix}.vvc ({} bytes, {frames} frames, keyint={}, {width}x{height} QP {qp} bd={bit_depth} cf={chroma_format:?})",
        stream.len(),
        options.keyint
    );
    Ok(())
}

fn main() -> ExitCode {
    if let Err(err) = run() {
        eprintln!("vvc_multitool_dump: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
