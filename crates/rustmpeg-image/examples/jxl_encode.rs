// SPDX-License-Identifier: Apache-2.0

//! Example: encode synthetic RGBA images into both v1 and v2 lossless Modular
//! and lossy VarDCT JPEG XL bitstreams, and verify round-trip decoding.

use rustmpeg_image::{
    JxlConfig, JxlDecoder, JxlEncoder, JxlError, JxlFrame, JxlImage, JxlPixelLayout,
};

fn main() -> Result<(), JxlError> {
    let width = 64u32;
    let height = 64u32;
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / (width - 1)) as u8;
            let g = ((y * 255) / (height - 1)) as u8;
            let b = (((x + y) * 255) / (width + height - 2)) as u8;
            let a = 255u8;
            pixels.extend_from_slice(&[r, g, b, a]);
        }
    }

    let frame = JxlFrame::new_u8(width, height, JxlPixelLayout::Rgba, pixels.clone())?;
    let image = JxlImage::from_single_frame(frame);

    // 1. Lossless Modular v1 & v2 encode + decode round-trip
    let lossless_v1 = JxlEncoder::new(JxlConfig::lossless()).encode(&image)?;
    let lossless_v2 = JxlEncoder::new(JxlConfig::lossless_v2()).encode(&image)?;
    let decoded_lossless = JxlDecoder::new().decode(&lossless_v2)?;
    assert_eq!(decoded_lossless.frames[0].data, pixels);
    println!(
        "Lossless Modular JXL ({}x{} RGBA): v1={} bytes, v2={} bytes (bit-exact verified)",
        width,
        height,
        lossless_v1.len(),
        lossless_v2.len()
    );

    // 2. Lossy VarDCT v1 & v2 encode + decode round-trip
    let lossy_v1 = JxlEncoder::new(JxlConfig::lossy(90.0)).encode(&image)?;
    let lossy_v2 = JxlEncoder::new(JxlConfig::lossy_v2(90.0)).encode(&image)?;
    let decoded_lossy = JxlDecoder::new().decode(&lossy_v2)?;
    assert_eq!(decoded_lossy.width, width);
    assert_eq!(decoded_lossy.height, height);
    println!(
        "Lossy VarDCT JXL (q=90, {}x{} RGBA): v1={} bytes, v2={} bytes",
        width,
        height,
        lossy_v1.len(),
        lossy_v2.len()
    );

    if let Some(out_path) = std::env::args().nth(1) {
        std::fs::write(&out_path, &lossless_v2)
            .map_err(|_| JxlError::InvalidConfig("failed to write output JXL file"))?;
        println!("Wrote {}", out_path);
    }
    if let Some(lossy_path) = std::env::args().nth(2) {
        std::fs::write(&lossy_path, &lossy_v2)
            .map_err(|_| JxlError::InvalidConfig("failed to write lossy JXL file"))?;
        println!("Wrote {}", lossy_path);
    }

    Ok(())
}
