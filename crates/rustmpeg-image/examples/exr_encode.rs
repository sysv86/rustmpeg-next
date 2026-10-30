// SPDX-License-Identifier: Apache-2.0

//! Example demonstrating OpenEXR v2 encoding and decoding across all 11 supported codecs
//! (`None`, `Rle`, `Zips`, `Zip`, `Piz`, `Pxr24`, `B44`, `B44a`, `Dwaa`, `Dwab`, `Zstd`).

use rustmpeg_image::exr::{
    ExrChannelData, ExrChannelDesc, ExrChannelPlane, ExrCompression, ExrDecoder, ExrEncoder,
    ExrImage, ExrPart, ExrSampleType, f32_to_f16,
};

fn main() {
    let width = 64u32;
    let height = 64u32;
    let count = (width * height) as usize;

    let mut r = Vec::with_capacity(count);
    let mut g = Vec::with_capacity(count);
    let mut b = Vec::with_capacity(count);
    let mut a = Vec::with_capacity(count);

    for y in 0..height {
        for x in 0..width {
            let fx = (x as f32) / (width as f32);
            let fy = (y as f32) / (height as f32);
            r.push(f32_to_f16(fx * 4.0));
            g.push(f32_to_f16(fy * 2.0));
            b.push(f32_to_f16((fx + fy) * 0.5));
            a.push(f32_to_f16(1.0));
        }
    }

    for codec in [
        ExrCompression::None,
        ExrCompression::Rle,
        ExrCompression::Zips,
        ExrCompression::Zip,
        ExrCompression::Piz,
        ExrCompression::Pxr24,
        ExrCompression::B44,
        ExrCompression::B44a,
        ExrCompression::Dwaa,
        ExrCompression::Dwab,
        ExrCompression::Zstd,
    ] {
        let planes = vec![
            ExrChannelPlane {
                desc: ExrChannelDesc::new("A", ExrSampleType::Half),
                width,
                height,
                data: ExrChannelData::Half(a.clone()),
            },
            ExrChannelPlane {
                desc: ExrChannelDesc::new("B", ExrSampleType::Half),
                width,
                height,
                data: ExrChannelData::Half(b.clone()),
            },
            ExrChannelPlane {
                desc: ExrChannelDesc::new("G", ExrSampleType::Half),
                width,
                height,
                data: ExrChannelData::Half(g.clone()),
            },
            ExrChannelPlane {
                desc: ExrChannelDesc::new("R", ExrSampleType::Half),
                width,
                height,
                data: ExrChannelData::Half(r.clone()),
            },
        ];

        let part = ExrPart::from_planar_channels(width, height, codec, planes).unwrap();
        let img = ExrImage::from_single_part(part);
        let encoded = ExrEncoder::default().encode(&img).unwrap();
        let decoded = ExrDecoder::default().decode(&encoded).unwrap();
        println!(
            "OpenEXR {:?}: encoded {}x{} RGBA HALF into {} bytes (decoded {} channels)",
            codec,
            width,
            height,
            encoded.len(),
            decoded.parts[0].channels.len()
        );
    }
}
