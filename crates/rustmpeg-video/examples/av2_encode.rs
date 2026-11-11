// SPDX-License-Identifier: Apache-2.0
//! Minimal AV2 encoder example: synthetic frames -> IVF.
//!
//! Usage: `cargo run -p rustmpeg-video --example av2_encode -- <out.ivf> [frames] [width] [height] [qp] [log2_tile_cols] [log2_tile_rows] [deblock 0|1] [420|422|444] [cdef 0|1] [ccso 0|1] [gdf 0|1] [rst 0|1] [newmv 0|1] [switchable 0|1] [compound 0|1] [bit_depth] [vaq 0|1] [delta_q 0|1] [rec.y4m] [vbr_kbps] [warped_motion 0|1] [obmc 0|1] [input.y4m] [palette 0|1] [intrabc 0|1] [filmgrain 0|1] [compound_types] [tx_type_rdo 0|1] [reduced_txtp_set] [dpcm 0|1] [rect 0|1] [ext 0|1] [uneven 0|1] [ccso_sb_reuse 0|1]`
//!
//! `rect 1` enables rectangular `H`/`V` partitions, `ext 1` the extended
//! `H3`/`V3` partitions and `uneven 1` the uneven 4-way `H4A/B`/`V4A/B`
//! partitions (AV2 Spec v1.0.0 §5.20.3.2).
//!
//! `warped_motion 1` enables `WARPMV` + `DELTAWARP` blocks (AV2 §5.20.7.6).
//! `obmc 1` is accepted but has no effect: AV2 v1.0.0 has no OBMC tool.
//! `palette 1` enables palette mode (§5.20.8.1/§5.20.8.4) on intra frames.
//! `intrabc 1` enables local-area Intra Block Copy (§5.20.5.4) on key frames.
//! `rst` is `0`, `1`/`pc` (PC Wiener luma), `ns` (NS Wiener luma),
//! `switch` (per-unit PC/NS), `chroma` (NS luma+chroma), or `pcchroma`.
//! `filmgrain 1` estimates and signals AV2 film grain (§7.21.7) and applies it
//! to the written reconstruction; reference frames stay clean.
//! When `input.y4m` is given, frames are read from it instead of the synthetic
//! generator.
//!
//! `rec.y4m` is optional; when present the encoder's own reconstruction is
//! written so it can be compared byte-for-byte with the reference decoder.

use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use rustmpeg_video::av2::EncodedPacket;
use rustmpeg_video::av2::container::{IvfWriter, RawFrame, Y4mHeader, Y4mReader, Y4mWriter};
use rustmpeg_video::av2::headers::PixelLayout;
use rustmpeg_video::av2::pixel::Pixel;
use rustmpeg_video::{Av2Encoder, Av2EncoderConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).unwrap_or_else(|| {
        env::temp_dir()
            .join("av2-example.ivf")
            .display()
            .to_string()
    });
    let frames: usize = env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let width: usize = env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let height: usize = env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let qp: u16 = env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let log2_tile_cols: u8 = env::args().nth(6).and_then(|s| s.parse().ok()).unwrap_or(0);
    let log2_tile_rows: u8 = env::args().nth(7).and_then(|s| s.parse().ok()).unwrap_or(0);

    let layout = match env::args().nth(9).as_deref() {
        Some("422") => PixelLayout::I422,
        Some("444") => PixelLayout::I444,
        _ => PixelLayout::I420,
    };
    let deblock: bool = env::args().nth(8).is_some_and(|s| s == "1");
    let cdef: bool = env::args().nth(10).is_some_and(|s| s == "1");
    let ccso: bool = env::args().nth(11).is_some_and(|s| s == "1");
    let gdf: bool = env::args().nth(12).is_some_and(|s| s == "1");
    // Loop restoration selector: 0/off, 1/pc (PC Wiener, luma),
    // ns (NS Wiener, luma), switch (per-unit PC/NS), chroma (NS luma+chroma).
    let rst_arg = env::args().nth(13).unwrap_or_default();
    let (rst, restoration_mode, restoration_chroma) = match rst_arg.as_str() {
        "1" | "pc" => (
            true,
            rustmpeg_video::av2::frame::RestorationMode::PcWiener,
            false,
        ),
        "ns" => (
            true,
            rustmpeg_video::av2::frame::RestorationMode::NsWiener,
            false,
        ),
        "switch" => (
            true,
            rustmpeg_video::av2::frame::RestorationMode::Switchable,
            false,
        ),
        "chroma" => (
            true,
            rustmpeg_video::av2::frame::RestorationMode::NsWiener,
            true,
        ),
        "pcchroma" => (
            true,
            rustmpeg_video::av2::frame::RestorationMode::PcWiener,
            true,
        ),
        _ => (
            false,
            rustmpeg_video::av2::frame::RestorationMode::PcWiener,
            false,
        ),
    };
    let newmv: bool = env::args().nth(14).is_some_and(|s| s == "1");
    let switchable: bool = env::args().nth(15).is_some_and(|s| s == "1");
    let compound: bool = env::args().nth(16).is_some_and(|s| s == "1");
    let bit_depth: u8 = env::args()
        .nth(17)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let vaq: bool = env::args().nth(18).is_some_and(|s| s == "1");
    let delta_q: bool = env::args().nth(19).is_some_and(|s| s == "1");
    let rec_path = env::args()
        .nth(20)
        .filter(|path| !path.is_empty() && path != "-");
    let vbr_kbps: u32 = env::args()
        .nth(21)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let warped_motion: bool = env::args().nth(22).is_some_and(|s| s == "1");
    let obmc: bool = env::args().nth(23).is_some_and(|s| s == "1");
    let input_path = env::args()
        .nth(24)
        .filter(|path| !path.is_empty() && path != "-");
    let palette: bool = env::args().nth(25).is_some_and(|s| s == "1");
    let intrabc: bool = env::args().nth(26).is_some_and(|s| s == "1");
    let filmgrain: bool = env::args().nth(27).is_some_and(|s| s == "1");
    let compound_types: u8 = env::args()
        .nth(28)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x0f);
    // Full transform-type set (`enable_tx_type_rdo`), the frame-level
    // `reduced_tx_set` (AV2 Spec v1.0.0 §5.18.7.4) and lossless DPCM
    // (`use_dpcm_y` / `use_dpcm_uv`, §5.20.5.5/§5.20.5.6).
    let tx_type_rdo: bool = env::args().nth(29).is_some_and(|s| s == "1");
    let reduced_txtp_set: u8 = env::args()
        .nth(30)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let dpcm: bool = env::args().nth(31).is_some_and(|s| s == "1");
    let rect_partitions: bool = env::args().nth(32).is_some_and(|s| s == "1");
    let ext_partitions: bool = env::args().nth(33).is_some_and(|s| s == "1");
    let uneven_4way_partitions: bool = env::args().nth(34).is_some_and(|s| s == "1");
    // `ccso_sb_reuse 1` reuses the reference frame's CCSO block flags
    // (`sb_reuse_ccso`, AV2 Spec v1.0.0 §5.18.7.12). Enabled by default; pass
    // `0` to signal the block flags on every inter frame.
    let ccso_sb_reuse: bool = env::args().nth(35).map(|s| s == "1").unwrap_or(true);

    let rate_control = if vbr_kbps > 0 {
        rustmpeg_video::av2::ratectrl::RateControlMode::Vbr {
            target_kbps: vbr_kbps,
            max_qidx: 220,
        }
    } else {
        rustmpeg_video::av2::ratectrl::RateControlMode::ConstantQp { base_qidx: qp }
    };
    let config = Av2EncoderConfig {
        width,
        height,
        layout,
        bit_depth,
        base_qidx: qp,
        rate_control,
        log2_tile_cols,
        log2_tile_rows,
        tools: rustmpeg_video::av2::frame::ToolConfig {
            enable_deblock: deblock,
            enable_cdef: cdef,
            enable_ccso: ccso,
            enable_ccso_sb_reuse: ccso_sb_reuse,
            enable_gdf: gdf,
            enable_restoration: rst,
            restoration_mode,
            restoration_chroma,
            enable_newmv: newmv,
            enable_switchable_interp: switchable,
            enable_compound: compound,
            compound_types,
            enable_compound_masks: compound,
            enable_vaq: vaq,
            enable_delta_q: delta_q,
            enable_warped_motion: warped_motion,
            enable_obmc: obmc,
            enable_palette: palette,
            enable_intrabc: intrabc,
            enable_film_grain: filmgrain,
            enable_tx_type_rdo: tx_type_rdo,
            reduced_txtp_set,
            enable_dpcm: dpcm,
            enable_rect_partitions: rect_partitions,
            enable_ext_partitions: ext_partitions,
            enable_uneven_4way_partitions: uneven_4way_partitions,
            ..Default::default()
        },
        ..Av2EncoderConfig::default()
    };
    let mut writer = IvfWriter::new(
        BufWriter::new(File::create(&path)?),
        width as u16,
        height as u16,
        30,
        1,
        frames as u32,
    )?;
    let mut y4m = match &rec_path {
        Some(p) => Some(Y4mWriter::new(
            BufWriter::new(File::create(p)?),
            Y4mHeader {
                width,
                height,
                framerate_num: 30,
                framerate_den: 1,
                bit_depth,
                layout,
                ..Y4mHeader::default()
            },
        )?),
        None => None,
    };

    match bit_depth {
        8 => {
            let mut encoder = Av2Encoder::<u8>::new(config)?;
            if let Some(path) = &input_path {
                let mut reader = Y4mReader::new(BufReader::new(File::open(path)?))?;
                for _ in 0..frames {
                    let Some(raw) = reader.read_frame::<u8>()? else {
                        break;
                    };
                    let mut packet = encoder.encode_raw_frame(&raw)?;
                    write_packet(&mut writer, &mut y4m, &mut packet, filmgrain)?;
                }
            } else {
                for idx in 0..frames {
                    let raw = RawFrame::generate_synthetic(width, height, 8, layout, idx);
                    let mut packet = encoder.encode_raw_frame(&raw)?;
                    write_packet(&mut writer, &mut y4m, &mut packet, filmgrain)?;
                }
            }
        }
        _ => {
            let mut encoder = Av2Encoder::<u16>::new(config)?;
            if let Some(path) = &input_path {
                let mut reader = Y4mReader::new(BufReader::new(File::open(path)?))?;
                for _ in 0..frames {
                    let Some(raw) = reader.read_frame::<u16>()? else {
                        break;
                    };
                    let mut packet = encoder.encode_raw_frame(&raw)?;
                    write_packet(&mut writer, &mut y4m, &mut packet, filmgrain)?;
                }
            } else {
                for idx in 0..frames {
                    let raw = RawFrame::generate_synthetic(width, height, bit_depth, layout, idx);
                    let mut packet = encoder.encode_raw_frame(&raw)?;
                    write_packet(&mut writer, &mut y4m, &mut packet, filmgrain)?;
                }
            }
        }
    }
    writer.flush()?;
    drop(y4m);
    println!("wrote {frames} frames to {path}");
    Ok(())
}

fn write_packet<P: Pixel, W: std::io::Write>(
    writer: &mut IvfWriter<W>,
    y4m: &mut Option<Y4mWriter<BufWriter<File>>>,
    packet: &mut EncodedPacket<P>,
    filmgrain: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if filmgrain && y4m.is_some() && packet.film_grain.present != 0 {
        rustmpeg_video::av2::filmgrain::apply_film_grain(
            &mut packet.rec_frame,
            &packet.film_grain.data,
            packet.film_grain.seed as u16,
        );
    }
    writer.write_packet(packet.pts, &packet.data)?;
    if packet.film_grain.present != 0 {
        eprintln!(
            "film grain: seed {}, y points {:?}, ar lag {}, ar coeffs y {:?}",
            packet.film_grain.seed,
            packet.film_grain.data.num_points,
            packet.film_grain.data.ar_coeff_lag,
            &packet.film_grain.data.ar_coeffs[0][..4],
        );
    }
    if let Some(y4m) = y4m.as_mut() {
        y4m.write_frame(&packet.rec_frame.to_raw())?;
    }
    eprintln!(
        "frame {}: {:?}, {} bytes, PSNR Y {:.2}, base qidx {}, avg qidx {}, warped {} (warpmv {}, local {}, extend {}), palette {}, intrabc {}, isp {}, newmv {}, subpel {}, compound {} (avg {}, wedge {}, diff {}, cwp {}) inter {} intra {}",
        packet.pts,
        packet.frame_type,
        packet.data.len(),
        packet.stats.psnr_y,
        packet.stats.qidx,
        packet.stats.avg_qidx,
        packet.stats.warped_blocks,
        packet.stats.warpmv_blocks,
        packet.stats.localwarp_blocks,
        packet.stats.extendwarp_blocks,
        packet.stats.palette_blocks,
        packet.stats.intrabc_blocks,
        packet.stats.isp_blocks,
        packet.stats.newmv_blocks,
        packet.stats.switchable_filter_blocks,
        packet.stats.compound_blocks,
        packet.stats.compound_average_blocks,
        packet.stats.compound_wedge_blocks,
        packet.stats.compound_diff_blocks,
        packet.stats.compound_cwp_blocks,
        packet.stats.inter_blocks,
        packet.stats.intra_blocks
    );
    eprintln!(
        "frame {}: tx non-DCT {}, WHT {}, DPCM y {}, DPCM uv {}",
        packet.pts,
        packet.stats.non_dct_tx_blocks,
        packet.stats.wht_blocks,
        packet.stats.dpcm_y_blocks,
        packet.stats.dpcm_uv_blocks
    );
    eprintln!(
        "frame {}: ccso units {}, reuse units {}, edge planes {}; gdf units {}, per-block {}",
        packet.pts,
        packet.stats.ccso_units,
        packet.stats.ccso_reuse_units,
        packet.stats.ccso_edge_planes,
        packet.stats.gdf_units,
        packet.stats.gdf_per_block
    );
    let pc = packet.stats.partition_counts;
    eprintln!(
        "frame {}: partitions none {} h {} v {} h3 {} v3 {} h4a {} h4b {} v4a {} v4b {} split {}",
        packet.pts, pc[0], pc[1], pc[2], pc[3], pc[4], pc[5], pc[6], pc[7], pc[8], pc[9]
    );
    Ok(())
}
