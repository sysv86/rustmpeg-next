// SPDX-License-Identifier: Apache-2.0

//! Example: probe and decode a JPEG XL file (or an in-memory test bitstream)
//! into a `JxlImage` and a `rustmpeg_core::VideoFrame`.

use rustmpeg_image::{
    JxlConfig, JxlDecoder, JxlEncoder, JxlError, JxlFrame, JxlImage, JxlPixelLayout,
};

fn main() -> Result<(), JxlError> {
    let input_bytes = if let Some(path) = std::env::args().nth(1) {
        std::fs::read(&path)
            .map_err(|_| JxlError::InvalidConfig("failed to read input JXL file"))?
    } else {
        let width = 32u32;
        let height = 32u32;
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                rgb.push((x * 8) as u8);
                rgb.push((y * 8) as u8);
                rgb.push(((x ^ y) * 8) as u8);
            }
        }
        let frame = JxlFrame::new_u8(width, height, JxlPixelLayout::Rgb, rgb)?;
        let image = JxlImage::from_single_frame(frame);
        JxlEncoder::new(JxlConfig::lossless()).encode(&image)?
    };

    let decoder = JxlDecoder::new();
    let (container, header) = decoder.probe(&input_bytes)?;
    println!(
        "Probed JXL: {:?}, {}x{}, xyb_encoded={}, extra_channels={}",
        container.kind,
        header.width_with_orientation(),
        header.height_with_orientation(),
        header.metadata.xyb_encoded,
        header.metadata.ec_info.len(),
    );

    let decoded = decoder.decode(&input_bytes)?;
    if let Some(raw_out) = std::env::args().nth(2) {
        std::fs::write(&raw_out, &decoded.frames[0].data).unwrap();
    }
    let video_frame = decoded.frames[0].to_video_frame()?;
    println!(
        "Decoded {} frame(s); frame[0] -> VideoFrame {:?} {}x{} ({} bytes in plane 0)",
        decoded.frames.len(),
        video_frame.format,
        video_frame.width,
        video_frame.height,
        video_frame.planes[0].data.len(),
    );

    Ok(())
}
