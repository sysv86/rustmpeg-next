// SPDX-License-Identifier: Apache-2.0
//! Minimal AV2 decoder example: IVF -> Y4M.
//!
//! Usage: `cargo run -p rustmpeg-video --example av2_decode -- in.ivf out.y4m [bit_depth] [grain 0|1]`
//!
//! `grain 0` disables output-side film grain synthesis, matching `dav2d
//! --filmgrain 0`; the default is enabled.

use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use rustmpeg_video::Av2Decoder;
use rustmpeg_video::av2::container::{IvfReader, Y4mHeader, Y4mWriter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = env::args().nth(1).unwrap_or_else(|| {
        env::temp_dir()
            .join("av2-example.ivf")
            .display()
            .to_string()
    });
    let output = env::args().nth(2).unwrap_or_else(|| {
        env::temp_dir()
            .join("av2-decoded.y4m")
            .display()
            .to_string()
    });

    let mut ivf = IvfReader::new(BufReader::new(File::open(&input)?))?;
    let width = ivf.header.width as usize;
    let height = ivf.header.height as usize;
    let bit_depth: u8 = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let apply_grain: bool = env::args().nth(4).is_none_or(|s| s != "0");
    if bit_depth > 8 {
        let mut decoder = Av2Decoder::<u16>::new();
        decoder.set_apply_grain(apply_grain);
        let mut writer: Option<Y4mWriter<BufWriter<File>>> = None;
        let mut count = 0usize;
        while let Some(packet) = ivf.read_packet()? {
            let Some(decoded) = decoder.decode_packet(&packet.data)? else {
                continue;
            };
            if writer.is_none() {
                writer = Some(Y4mWriter::new(
                    BufWriter::new(File::create(&output)?),
                    Y4mHeader {
                        width,
                        height,
                        bit_depth: decoded.frame.bit_depth,
                        layout: decoded.frame.layout,
                        ..Y4mHeader::default()
                    },
                )?);
            }
            writer.as_mut().unwrap().write_frame(&decoded.frame)?;
            count += 1;
            eprintln!(
                "frame {count}: {:?}, poc {}, q {}",
                decoded.frame_type, decoded.poc, decoded.qidx
            );
        }
        if let Some(mut w) = writer {
            w.flush()?;
        } else {
            return Err("input contained no coded frames".into());
        }
        println!("decoded {count} frames to {output}");
        return Ok(());
    }
    let mut decoder = Av2Decoder::<u8>::new();
    decoder.set_apply_grain(apply_grain);
    let mut writer: Option<Y4mWriter<BufWriter<File>>> = None;
    let mut count = 0usize;

    while let Some(packet) = ivf.read_packet()? {
        let Some(decoded) = decoder.decode_packet(&packet.data)? else {
            continue;
        };
        if writer.is_none() {
            writer = Some(Y4mWriter::new(
                BufWriter::new(File::create(&output)?),
                Y4mHeader {
                    width,
                    height,
                    bit_depth: decoded.frame.bit_depth,
                    layout: decoded.frame.layout,
                    ..Y4mHeader::default()
                },
            )?);
        }
        writer.as_mut().unwrap().write_frame(&decoded.frame)?;
        count += 1;
        eprintln!(
            "frame {count}: {:?}, poc {}, q {}",
            decoded.frame_type, decoded.poc, decoded.qidx
        );
    }
    if let Some(mut w) = writer {
        w.flush()?;
    } else {
        return Err("input contained no coded frames".into());
    }
    println!("decoded {count} frames to {output}");
    Ok(())
}
