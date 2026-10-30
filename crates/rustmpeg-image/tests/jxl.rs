// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the `rustmpeg-image` JPEG XL (ISO/IEC 18181-1 & 18181-2)
//! decoder and clean-room encoder (v1 and v2 pipelines).

use rustmpeg_core::{ColorDescription, PixelFormat, VideoFrame};
use rustmpeg_image::jxl::{
    AnimationHeader, JxlBitDepth, JxlColorEncoding, JxlConfig, JxlContainerKind, JxlDecoder,
    JxlEncoder, JxlEncodingMode, JxlEntropyMode, JxlError, JxlFrame, JxlImage, JxlPixelLayout,
    Predictor,
};

fn make_rgb_pattern(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 7 + y * 3) & 0xff) as u8;
            let g = ((x * 5 + y * 11 + 40) & 0xff) as u8;
            let b = ((x * 13 + y * 17 + 90) & 0xff) as u8;
            out.extend_from_slice(&[r, g, b]);
        }
    }
    out
}

fn make_rgba_pattern(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 9 + y * 5) & 0xff) as u8;
            let g = ((x * 3 + y * 13 + 25) & 0xff) as u8;
            let b = ((x * 15 + y * 7 + 80) & 0xff) as u8;
            let a = ((x * 11 + y * 19 + 200) & 0xff) as u8;
            out.extend_from_slice(&[r, g, b, a]);
        }
    }
    out
}

#[test]
fn modular_lossless_roundtrip_all_pixel_layouts_and_predictors() {
    let decoder = JxlDecoder::new();

    for predictor in [
        Predictor::Zero,
        Predictor::West,
        Predictor::North,
        Predictor::AvgWestAndNorth,
        Predictor::Select,
        Predictor::Gradient,
        Predictor::SelfCorrecting,
    ] {
        // 1. Gray 8-bit
        let gray_data: Vec<u8> = (0..(19 * 17))
            .map(|i| ((i * 29 + 7) & 0xff) as u8)
            .collect();
        let gray_frame =
            JxlFrame::new_u8(19, 17, JxlPixelLayout::Gray, gray_data.clone()).expect("gray frame");
        let gray_img = JxlImage::from_single_frame(gray_frame);
        let cfg = JxlConfig {
            mode: JxlEncodingMode::ModularLossless { predictor },
            use_rct: false,
            gaborish: false,
            force_container: false,
            ..JxlConfig::lossless()
        };
        let encoded = JxlEncoder::new(cfg).encode(&gray_img).expect("encode gray");
        let decoded = decoder.decode(&encoded).expect("decode gray");
        assert_eq!(decoded.width, 19);
        assert_eq!(decoded.height, 17);
        assert_eq!(decoded.layout, JxlPixelLayout::Gray);
        assert_eq!(decoded.frames[0].data, gray_data);

        // 2. GrayAlpha 8-bit
        let ga_data: Vec<u8> = (0..(16 * 16 * 2))
            .map(|i| ((i * 13 + 3) & 0xff) as u8)
            .collect();
        let ga_frame =
            JxlFrame::new_u8(16, 16, JxlPixelLayout::GrayAlpha, ga_data.clone()).expect("ga frame");
        let ga_img = JxlImage::from_single_frame(ga_frame);
        let encoded_ga = JxlEncoder::new(JxlConfig {
            mode: JxlEncodingMode::ModularLossless { predictor },
            use_rct: false,
            gaborish: false,
            force_container: false,
            ..JxlConfig::lossless()
        })
        .encode(&ga_img)
        .expect("encode ga");
        let decoded_ga = decoder.decode(&encoded_ga).expect("decode ga");
        assert_eq!(decoded_ga.layout, JxlPixelLayout::GrayAlpha);
        assert_eq!(decoded_ga.frames[0].data, ga_data);

        // 3. RGB 8-bit with YCoCg RCT
        let rgb_data = make_rgb_pattern(23, 19);
        let rgb_frame =
            JxlFrame::new_u8(23, 19, JxlPixelLayout::Rgb, rgb_data.clone()).expect("rgb frame");
        let rgb_img = JxlImage::from_single_frame(rgb_frame);
        let encoded_rgb = JxlEncoder::new(JxlConfig {
            mode: JxlEncodingMode::ModularLossless { predictor },
            use_rct: true,
            gaborish: false,
            force_container: false,
            ..JxlConfig::lossless()
        })
        .encode(&rgb_img)
        .expect("encode rgb");
        let decoded_rgb = decoder.decode(&encoded_rgb).expect("decode rgb");
        assert_eq!(decoded_rgb.layout, JxlPixelLayout::Rgb);
        assert_eq!(decoded_rgb.frames[0].data, rgb_data);

        // 4. RGBA 8-bit with YCoCg RCT
        let rgba_data = make_rgba_pattern(24, 24);
        let rgba_frame =
            JxlFrame::new_u8(24, 24, JxlPixelLayout::Rgba, rgba_data.clone()).expect("rgba frame");
        let rgba_img = JxlImage::from_single_frame(rgba_frame);
        let encoded_rgba = JxlEncoder::new(JxlConfig {
            mode: JxlEncodingMode::ModularLossless { predictor },
            use_rct: true,
            gaborish: false,
            force_container: false,
            ..JxlConfig::lossless()
        })
        .encode(&rgba_img)
        .expect("encode rgba");
        let decoded_rgba = decoder.decode(&encoded_rgba).expect("decode rgba");
        assert_eq!(decoded_rgba.layout, JxlPixelLayout::Rgba);
        assert_eq!(decoded_rgba.frames[0].data, rgba_data);
    }
}

#[test]
fn modular_lossless_v2_ans_lz77_palette_squeeze_and_near_lossless_roundtrip() {
    let decoder = JxlDecoder::new();

    // 1. Palette-eligible synthetic UI icon (<= 256 unique RGBA colors) with lossless_v2()
    let width = 32u32;
    let height = 32u32;
    let mut icon_rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let idx = ((x / 4) + (y / 4) * 8) as u8;
            icon_rgba.extend_from_slice(&[
                idx.wrapping_mul(3),
                idx.wrapping_mul(5),
                255u8.wrapping_sub(idx.wrapping_mul(3)),
                255,
            ]);
        }
    }
    let icon_frame =
        JxlFrame::new_u8(width, height, JxlPixelLayout::Rgba, icon_rgba.clone()).unwrap();
    let icon_img = JxlImage::from_single_frame(icon_frame);
    let icon_encoded = JxlEncoder::new(JxlConfig::lossless_v2())
        .encode(&icon_img)
        .expect("encode icon lossless_v2");
    let icon_decoded = decoder
        .decode(&icon_encoded)
        .expect("decode icon lossless_v2");
    assert_eq!(icon_decoded.frames[0].data, icon_rgba);

    // 2. Continuous-tone image (> 256 colors -> triggers YCoCg + Squeeze + adaptive MA tree + ANS + LZ77)
    let cont_rgb = make_rgb_pattern(48, 40);
    let cont_frame = JxlFrame::new_u8(48, 40, JxlPixelLayout::Rgb, cont_rgb.clone()).unwrap();
    let cont_img = JxlImage::from_single_frame(cont_frame);
    let cont_encoded = JxlEncoder::new(JxlConfig::lossless_v2())
        .encode(&cont_img)
        .expect("encode continuous lossless_v2");
    let cont_decoded = decoder
        .decode(&cont_encoded)
        .expect("decode continuous lossless_v2");
    assert_eq!(cont_decoded.frames[0].data, cont_rgb);

    // 3. Controlled near-lossless Modular (`modular_max_error = 2`)
    let mut near_cfg = JxlConfig::lossless_v2();
    near_cfg.enable_squeeze = false;
    near_cfg.use_rct = false;
    near_cfg.modular_max_error = 2;
    let near_encoded = JxlEncoder::new(near_cfg)
        .encode(&cont_img)
        .expect("encode near-lossless modular");
    let near_decoded = decoder
        .decode(&near_encoded)
        .expect("decode near-lossless modular");
    for (&orig, &rec) in cont_rgb.iter().zip(near_decoded.frames[0].data.iter()) {
        let diff = orig.abs_diff(rec);
        assert!(
            diff <= 2,
            "near-lossless max_error=2 violated: orig={orig}, rec={rec}, diff={diff}"
        );
    }
}

#[test]
fn modular_lossless_16bit_and_float32_roundtrip() {
    let decoder = JxlDecoder::new();
    let encoder = JxlEncoder::new(JxlConfig::lossless());

    // 16-bit RGB
    let width = 17u32;
    let height = 13u32;
    let samples: Vec<u16> = (0..(width * height * 3))
        .map(|i| ((i * 257 + 19) & 0xffff) as u16)
        .collect();
    let frame16 =
        JxlFrame::new_u16(width, height, JxlPixelLayout::Rgb, 16, &samples).expect("u16 frame");
    let expected_bytes = frame16.data.clone();
    let img16 = JxlImage::from_single_frame(frame16);
    let encoded16 = encoder.encode(&img16).expect("encode 16-bit");
    let decoded16 = decoder.decode(&encoded16).expect("decode 16-bit");
    assert_eq!(decoded16.bit_depth, JxlBitDepth::Uint16(16));
    assert_eq!(decoded16.frames[0].data, expected_bytes);

    // 32-bit float RGB
    let mut f32_bytes = Vec::with_capacity((width * height * 3 * 4) as usize);
    for i in 0..(width * height * 3) {
        let val = (i as f32) / ((width * height * 3) as f32);
        f32_bytes.extend_from_slice(&val.to_le_bytes());
    }
    let frame_f32 = JxlFrame {
        width,
        height,
        layout: JxlPixelLayout::Rgb,
        bit_depth: JxlBitDepth::Float32,
        duration_ticks: 0,
        name: String::new(),
        data: f32_bytes.clone(),
    };
    let mut img_f32 = JxlImage::from_single_frame(frame_f32);
    img_f32.color_encoding = JxlColorEncoding::LinearSrgb;
    let encoded_f32 = encoder.encode(&img_f32).expect("encode f32");
    let decoded_f32 = decoder.decode(&encoded_f32).expect("decode f32");
    assert_eq!(decoded_f32.bit_depth, JxlBitDepth::Float32);
    assert_eq!(decoded_f32.color_encoding, JxlColorEncoding::LinearSrgb);
    assert_eq!(decoded_f32.frames[0].data, f32_bytes);
}

#[test]
fn modular_lossless_multigroup_large_image_roundtrip() {
    // 300x280 spans 2x2 = 4 pass groups (> 256x256)
    let width = 300u32;
    let height = 280u32;
    let rgb = make_rgb_pattern(width, height);
    let frame = JxlFrame::new_u8(width, height, JxlPixelLayout::Rgb, rgb.clone()).expect("frame");
    let img = JxlImage::from_single_frame(frame);
    let bytes = JxlEncoder::new(JxlConfig::lossless())
        .encode(&img)
        .expect("encode multigroup modular");
    let decoded = JxlDecoder::new()
        .decode(&bytes)
        .expect("decode multigroup modular");
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.frames[0].data, rgb);
}

#[test]
fn vardct_lossy_v1_and_v2_single_and_multigroup_psnr_and_alpha() {
    let decoder = JxlDecoder::new();

    for cfg in [JxlConfig::lossy(92.0), JxlConfig::lossy_v2(92.0)] {
        let encoder = JxlEncoder::new(cfg);

        // Smooth + textured 64x64 RGBA
        let width = 64u32;
        let height = 64u32;
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let r = ((x * 255) / (width - 1)) as u8;
                let g = ((y * 255) / (height - 1)) as u8;
                let b = (((x + y) * 255) / (width + height - 2)) as u8;
                let a = ((x * 3 + y * 5) & 0xff) as u8;
                rgba.extend_from_slice(&[r, g, b, a]);
            }
        }

        let frame = JxlFrame::new_u8(width, height, JxlPixelLayout::Rgba, rgba.clone())
            .expect("rgba frame");
        let img = JxlImage::from_single_frame(frame);
        let encoded = encoder.encode(&img).expect("encode vardct");
        let decoded = decoder.decode(&encoded).expect("decode vardct");
        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.layout, JxlPixelLayout::Rgba);

        // Compute RGB PSNR and verify lossless alpha extra channel
        let out = &decoded.frames[0].data;
        let mut mse_sum = 0.0f64;
        let num_px = (width * height) as usize;
        for i in 0..num_px {
            for c in 0..3 {
                let diff = f64::from(rgba[i * 4 + c]) - f64::from(out[i * 4 + c]);
                mse_sum += diff * diff;
            }
            // Alpha channel is Modular-coded losslessly even in VarDCT mode
            assert_eq!(out[i * 4 + 3], rgba[i * 4 + 3], "alpha mismatch at px {i}");
        }
        let mse = mse_sum / ((num_px * 3) as f64);
        let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
        assert!(
            psnr > 32.0,
            "expected VarDCT PSNR > 32 dB, got {psnr:.2} dB"
        );

        // Multi-group VarDCT (272x264 -> 2x2 pass groups)
        let mw = 272u32;
        let mh = 264u32;
        let mut mrgb = Vec::with_capacity((mw * mh * 3) as usize);
        for y in 0..mh {
            for x in 0..mw {
                mrgb.push(((x * 255) / mw) as u8);
                mrgb.push(((y * 255) / mh) as u8);
                mrgb.push(128);
            }
        }
        let mframe = JxlFrame::new_u8(mw, mh, JxlPixelLayout::Rgb, mrgb).expect("mframe");
        let mimg = JxlImage::from_single_frame(mframe);
        let mencoded = encoder.encode(&mimg).expect("encode multigroup vardct");
        let mdecoded = decoder.decode(&mencoded).expect("decode multigroup vardct");
        assert_eq!(mdecoded.width, mw);
        assert_eq!(mdecoded.height, mh);
    }
}

#[test]
fn wide_gamut_hdr_and_embedded_icc_profile_roundtrip() {
    let decoder = JxlDecoder::new();
    let width = 32u32;
    let height = 32u32;
    let rgb = make_rgb_pattern(width, height);

    // 1. Display P3, Rec.2020 PQ, Rec.2020 HLG in both Modular and VarDCT v2
    for color_encoding in [
        JxlColorEncoding::DisplayP3,
        JxlColorEncoding::Bt2020Pq,
        JxlColorEncoding::Bt2020Hlg,
    ] {
        let frame = JxlFrame::new_u8(width, height, JxlPixelLayout::Rgb, rgb.clone()).unwrap();
        let mut img = JxlImage::from_single_frame(frame);
        img.color_encoding = color_encoding.clone();

        let mod_bytes = JxlEncoder::new(JxlConfig::lossless_v2())
            .encode(&img)
            .expect("encode modular wide gamut");
        let mod_dec = decoder
            .decode(&mod_bytes)
            .expect("decode modular wide gamut");
        assert_eq!(mod_dec.color_encoding, color_encoding);
        assert_eq!(mod_dec.frames[0].data, rgb);

        let var_bytes = JxlEncoder::new(JxlConfig::lossy_v2(92.0))
            .encode(&img)
            .expect("encode vardct wide gamut");
        let var_dec = decoder
            .decode(&var_bytes)
            .expect("decode vardct wide gamut");
        assert_eq!(var_dec.color_encoding, color_encoding);
        assert_eq!(var_dec.width, width);
        assert_eq!(var_dec.height, height);
    }

    // 2. Embedded ICC profile (ISO/IEC 18181-1 Annex E compressed ICC stream)
    let mut synthetic_icc = vec![0u8; 256];
    // Populate ICC header with valid length, "acsp" signature at 36..40, "RGB " at 16..20, "XYZ " at 20..24
    synthetic_icc[0..4].copy_from_slice(&(256u32).to_be_bytes());
    synthetic_icc[12..16].copy_from_slice(b"mntr");
    synthetic_icc[16..20].copy_from_slice(b"RGB ");
    synthetic_icc[20..24].copy_from_slice(b"XYZ ");
    synthetic_icc[36..40].copy_from_slice(b"acsp");
    for (i, b) in synthetic_icc[128..].iter_mut().enumerate() {
        *b = ((i * 17 + 5) & 0xff) as u8;
    }

    let icc_frame = JxlFrame::new_u8(width, height, JxlPixelLayout::Rgb, rgb.clone()).unwrap();
    let mut icc_img = JxlImage::from_single_frame(icc_frame);
    icc_img.color_encoding = JxlColorEncoding::Icc(synthetic_icc.clone());

    let mut icc_cfg = JxlConfig::lossless_v2();
    icc_cfg.entropy_mode = JxlEntropyMode::Ans;
    let icc_encoded = JxlEncoder::new(icc_cfg)
        .encode(&icc_img)
        .expect("encode embedded ICC");
    let icc_decoded = decoder.decode(&icc_encoded).expect("decode embedded ICC");
    assert_eq!(
        icc_decoded.color_encoding,
        JxlColorEncoding::Icc(synthetic_icc)
    );
    assert_eq!(icc_decoded.frames[0].data, rgb);
}

#[test]
fn container_exif_xmp_animation_and_video_frame_roundtrip() {
    let width = 16u32;
    let height = 16u32;
    let f0_bytes = make_rgb_pattern(width, height);
    let mut f1_bytes = f0_bytes.clone();
    f1_bytes.reverse();

    let mut f0 = JxlFrame::new_u8(width, height, JxlPixelLayout::Rgb, f0_bytes.clone()).unwrap();
    f0.duration_ticks = 10;
    f0.name = "frame0".to_string();

    let mut f1 = JxlFrame::new_u8(width, height, JxlPixelLayout::Rgb, f1_bytes.clone()).unwrap();
    f1.duration_ticks = 20;
    f1.name = "frame1".to_string();

    let image = JxlImage {
        width,
        height,
        layout: JxlPixelLayout::Rgb,
        bit_depth: JxlBitDepth::Uint8(8),
        color_encoding: JxlColorEncoding::Srgb,
        container_kind: JxlContainerKind::Container,
        exif: Some(b"Exif\0\0II*\0\x08\0\0\0".to_vec()),
        xmp: Some(b"<x:xmpmeta>rustmpeg-next</x:xmpmeta>".to_vec()),
        animation: Some(AnimationHeader {
            tps_numerator: 100,
            tps_denominator: 1,
            num_loops: 0,
            have_timecodes: false,
        }),
        frames: vec![f0, f1],
    };

    let bytes = JxlEncoder::new(JxlConfig::lossless_v2())
        .encode(&image)
        .expect("encode container animation");
    let decoded = JxlDecoder::new()
        .decode(&bytes)
        .expect("decode container animation");

    assert_eq!(decoded.container_kind, JxlContainerKind::Container);
    assert_eq!(decoded.exif, image.exif);
    assert_eq!(decoded.xmp, image.xmp);
    assert_eq!(decoded.animation, image.animation);
    assert_eq!(decoded.frames.len(), 2);
    assert_eq!(decoded.frames[0].duration_ticks, 10);
    assert_eq!(decoded.frames[0].name, "frame0");
    assert_eq!(decoded.frames[0].data, f0_bytes);
    assert_eq!(decoded.frames[1].duration_ticks, 20);
    assert_eq!(decoded.frames[1].name, "frame1");
    assert_eq!(decoded.frames[1].data, f1_bytes);

    // VideoFrame conversion round-trip
    let vf = VideoFrame::new(PixelFormat::Rgb8, 8, 16, 16, ColorDescription::default()).unwrap();
    let vf_jxl = JxlEncoder::new(JxlConfig::lossless_v2())
        .encode_video_frame(&vf)
        .expect("encode VideoFrame");
    let decoded_vf = JxlDecoder::new()
        .decode_video_frame(&vf_jxl)
        .expect("decode VideoFrame");
    assert_eq!(decoded_vf.width, 16);
    assert_eq!(decoded_vf.height, 16);
    assert_eq!(decoded_vf.format, PixelFormat::Rgba8);
}

#[test]
fn rejects_jbrd_and_malformed_inputs_without_panicking() {
    let decoder = JxlDecoder::new();

    // Empty and invalid signature
    assert!(matches!(
        decoder.decode(&[]),
        Err(JxlError::UnexpectedEof | JxlError::InvalidSignature)
    ));
    assert!(matches!(
        decoder.decode(&[0x00, 0x01, 0x02, 0x03]),
        Err(JxlError::InvalidSignature)
    ));

    // Truncated bare codestream
    assert!(decoder.decode(&[0xff, 0x0a, 0x08]).is_err());

    // Container with policy-forbidden `jbrd` box
    let mut bad_container = Vec::new();
    bad_container.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
    ]);
    bad_container.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x14, b'f', b't', b'y', b'p', b'j', b'x', b'l', b' ', 0x00, 0x00, 0x00,
        0x00, b'j', b'x', b'l', b' ',
    ]);
    bad_container.extend_from_slice(&[0x00, 0x00, 0x00, 0x0c, b'j', b'b', b'r', b'd', 1, 2, 3, 4]);
    assert!(matches!(
        decoder.decode(&bad_container),
        Err(JxlError::Unsupported(_))
    ));
}

#[test]
fn official_conformance_bitstream_vectors_decode() {
    let decoder = JxlDecoder::new();

    // 1. Extended-size ISO/IEC 18181-2 container with 8x8 VarDCT codestream
    let ext_container: &[u8] = &[
        0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a, 0x00, 0x00, 0x00,
        0x14, 0x66, 0x74, 0x79, 0x70, 0x6a, 0x78, 0x6c, 0x20, 0x00, 0x00, 0x00, 0x00, 0x6a, 0x78,
        0x6c, 0x20, 0x00, 0x00, 0x00, 0x01, 0x6a, 0x78, 0x6c, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x48, 0xff, 0x0a, 0x41, 0x06, 0x00, 0x12, 0x88, 0x02, 0x00, 0xb4, 0x00, 0x55,
        0x0f, 0x00, 0x00, 0xa8, 0x50, 0x19, 0x65, 0xdc, 0xe0, 0xe5, 0x5c, 0xcf, 0x97, 0x1f, 0x3a,
        0x2c, 0xa6, 0x6d, 0x5c, 0x67, 0x68, 0xab, 0x6d, 0x0b, 0x4b, 0x12, 0x45, 0xc6, 0xb1, 0x49,
        0x3a, 0x81, 0x43, 0x92, 0x48, 0xd2, 0x60, 0x20, 0x99, 0xc1, 0x70, 0x5a, 0x25, 0x09,
    ];
    let img_ext = decoder
        .decode(ext_container)
        .expect("decode extended container");
    assert_eq!(img_ext.container_kind, JxlContainerKind::Container);
    assert_eq!(img_ext.width, 8);
    assert_eq!(img_ext.height, 8);
    assert_eq!(img_ext.frames.len(), 1);

    // 2. PQ HDR gradient (1088x64)
    let pq_gradient: &[u8] = &[
        0xff, 0x0a, 0xf8, 0x81, 0x7e, 0x88, 0x80, 0x1f, 0xa0, 0xe8, 0x11, 0x87, 0x03, 0x00, 0x00,
        0x00, 0x40, 0x08, 0x04, 0x01, 0x00, 0xd4, 0x00, 0x00, 0x00, 0x80, 0x05, 0x60, 0x00, 0x02,
        0x4b, 0x38, 0x69, 0x98, 0xca, 0x83, 0xf7, 0x4b, 0x48, 0x98, 0x02, 0x05, 0x8e, 0x1b, 0x37,
        0xc6, 0xdd, 0x18, 0x83, 0xb1, 0xad, 0xbc, 0x00, 0x44, 0x00, 0x80, 0x33, 0x01, 0x00, 0x00,
        0x00, 0x34, 0x54, 0x2f, 0xbc, 0xf4, 0xd2, 0x0b, 0x2f, 0xbd, 0xf4, 0xd2, 0x0b, 0x2f, 0xbd,
        0xf4, 0xc2, 0x4b, 0x2f, 0xbd, 0xf0, 0xd2, 0x0b, 0xa3, 0x44, 0xaf, 0x05, 0xc0, 0x2e, 0xa7,
        0xfa, 0x0f, 0x83, 0x6d, 0x10, 0xf8, 0xc2, 0x47, 0x90, 0x19, 0x94, 0x19, 0xfc, 0x56, 0x06,
        0x03, 0x14, 0x5a, 0x00, 0xa0, 0xd8, 0x0d, 0xec, 0x72, 0xaa, 0xff, 0x30, 0xd8, 0x06, 0x81,
        0x2f, 0x7c, 0x04, 0x99, 0x41, 0x99, 0xc1, 0x6f, 0x65, 0x03, 0x18, 0x0f, 0x04, 0xd0, 0x33,
        0x0b, 0x00,
    ];
    let img_pq = decoder.decode(pq_gradient).expect("decode pq gradient");
    assert_eq!(img_pq.width, 1088);
    assert_eq!(img_pq.height, 64);
    assert_eq!(img_pq.frames.len(), 1);

    // 3. Splines (2048x2048) — verify splines are actually rendered (non-zero pixels along the curve)
    let splines_jxl: &[u8] = &[
        0xff, 0x0a, 0xfa, 0x3f, 0x01, 0x90, 0xd8, 0x63, 0x10, 0x00, 0xe0, 0x00, 0x00, 0x00, 0x40,
        0x00, 0x04, 0x40, 0x00, 0x04, 0x00, 0xa6, 0x01, 0x40, 0x8d, 0x18, 0xc5, 0x92, 0xa4, 0x57,
        0xbc, 0xb7, 0x7b, 0x75, 0x40, 0x4f, 0xfa, 0xba, 0x6b, 0x7d, 0xcb, 0xf8, 0xbf, 0x9b, 0x85,
        0xde, 0x04, 0xe4, 0x03, 0xf5, 0xf6, 0x36, 0x6e, 0xc9, 0xf3, 0x5f, 0xf4, 0x83, 0x02, 0x19,
        0x94, 0x1b, 0xc4, 0x84, 0xed, 0x47, 0x09, 0x43, 0x32, 0x78, 0x26, 0x08, 0x33, 0xaa, 0x25,
        0x10, 0x0c, 0x03, 0x03, 0x03, 0x03,
    ];
    let img_splines = decoder.decode(splines_jxl).expect("decode splines");
    assert_eq!(img_splines.width, 2048);
    assert_eq!(img_splines.height, 2048);
    let non_zero_spline_samples = img_splines.frames[0]
        .data
        .iter()
        .filter(|&&b| b > 0)
        .count();
    assert!(
        non_zero_spline_samples > 100_000,
        "expected rendered spline samples, got {non_zero_spline_samples}"
    );

    // 4. Cropped multi-frame animation with canvas blending (50x80, 4 frames)
    let traffic_jxl: &[u8] = &[
        0xff, 0x0a, 0x78, 0x02, 0x62, 0x08, 0x02, 0xb0, 0x26, 0x08, 0x90, 0x00, 0x00, 0x00, 0x0f,
        0x69, 0x30, 0x4b, 0x00, 0x00, 0x80, 0x00, 0x00, 0xcc, 0x07, 0x4b, 0x28, 0x31, 0xdd, 0x8c,
        0xf5, 0xc1, 0x67, 0x30, 0x3b, 0xaf, 0x10, 0x10, 0x10, 0x38, 0x02, 0x02, 0x02, 0x4a, 0xb2,
        0x80, 0x48, 0x24, 0x42, 0x99, 0xf1, 0x67, 0xf2, 0x77, 0x71, 0xf3, 0x5d, 0xe4, 0x59, 0xa6,
        0x55, 0xe1, 0x8e, 0xe8, 0x88, 0xd8, 0x18, 0x63, 0x8c, 0x24, 0x65, 0xe1, 0x16, 0xc6, 0x18,
        0xaf, 0xb2, 0xe1, 0xb0, 0xbe, 0x80, 0xf0, 0x00, 0x00, 0x94, 0x09, 0x00, 0xc6, 0x8e, 0xec,
        0x0c, 0x23, 0x58, 0xe3, 0x27, 0x21, 0x8e, 0xc0, 0x6f, 0xf6, 0xbe, 0x86, 0x82, 0x3e, 0xf9,
        0x69, 0x37, 0x4b, 0xbf, 0x78, 0xdc, 0x0d, 0xa9, 0xa0, 0x10, 0x0e, 0x77, 0x23, 0xb0, 0x99,
        0xcc, 0xf6, 0x00, 0x7a, 0xc0, 0x78, 0x72, 0x52, 0x95, 0x06, 0x9a, 0x77, 0x5c, 0x24, 0x0c,
        0xc7, 0x29, 0x38, 0x82, 0xfe, 0xf8, 0x81, 0xc1, 0xc7, 0x0e, 0xc8, 0x84, 0x8a, 0x45, 0xc7,
        0xe5, 0x54, 0x9f, 0x86, 0xc5, 0x29, 0x01, 0x6c, 0x95, 0x56, 0x8b, 0x18, 0x3f, 0x61, 0x03,
        0x6a, 0x62, 0xa6, 0x9e, 0x0d, 0x2f, 0x09, 0x91, 0xd4, 0x2d, 0xf0, 0xca, 0x19, 0x2f, 0xd0,
        0x16, 0xa4, 0xaa, 0x0e, 0x20, 0xac, 0x33, 0x4e, 0x44, 0xa1, 0x47, 0xd8, 0xd3, 0x55, 0x30,
        0xc0, 0x94, 0x5c, 0xa4, 0x0c, 0x40, 0x04, 0x2f, 0x7f, 0x18, 0x8e, 0x81, 0xd0, 0x9b, 0xff,
        0x69, 0xde, 0x42, 0x79, 0x70, 0x24, 0x67, 0xe9, 0xc8, 0x9c, 0xc1, 0xeb, 0xcf, 0x36, 0xc6,
        0xca, 0xca, 0xc9, 0xad, 0xef, 0xee, 0x8f, 0xe7, 0xf3, 0x31, 0x7a, 0xe3, 0x5f, 0x77, 0x14,
        0x3a, 0x49, 0x5c, 0x71, 0xbe, 0xd2, 0x81, 0x20, 0xb2, 0x6a, 0xe4, 0x38, 0xb7, 0xe8, 0x42,
        0x41, 0xa2, 0xea, 0x14, 0x14, 0x9d, 0x1f, 0x2a, 0xb4, 0x2d, 0x0a, 0xdb, 0xa3, 0xe0, 0x09,
        0x95, 0x99, 0x67, 0x75, 0x22, 0x2a, 0xa4, 0xbe, 0x85, 0xee, 0xe8, 0xfb, 0x1f, 0x4a, 0xe4,
        0x0d, 0x67, 0x60, 0xb7, 0x0d, 0x3a, 0x58, 0xd2, 0xfd, 0x15, 0x37, 0x7c, 0x36, 0x8d, 0x90,
        0x62, 0x0c, 0x26, 0xf2, 0x11, 0x0f, 0xf0, 0xc1, 0x8e, 0xc1, 0x87, 0xf3, 0xbb, 0x1c, 0xea,
        0xd6, 0x12, 0x0f, 0x30, 0xfe, 0x4f, 0x7e, 0xd8, 0x18, 0x52, 0x3f, 0x10, 0xb9, 0x76, 0x24,
        0xab, 0x77, 0xa4, 0xae, 0x3a, 0xb2, 0xcd, 0xd5, 0x84, 0x17, 0x14, 0x00, 0xdd, 0x87, 0x75,
        0x86, 0x5c, 0xec, 0x43, 0x2a, 0x99, 0x88, 0x7e, 0x2d, 0x8c, 0xb2, 0x40, 0x7c, 0x76, 0x50,
        0x38, 0xf8, 0xcc, 0x78, 0xad, 0xb4, 0xfd, 0x64, 0x22, 0xcf, 0x03, 0x29, 0xa8, 0x3d, 0xf6,
        0x0a, 0x77, 0x10, 0x58, 0x74, 0xe5, 0x1f, 0xf3, 0xc4, 0x4e, 0x64, 0x54, 0x56, 0xab, 0x17,
        0x3c, 0x6d, 0x0c, 0x4d, 0xc9, 0x90, 0xd5, 0xe4, 0xa8, 0xd6, 0x31, 0x41, 0x0c, 0x20, 0x1a,
        0xc3, 0xd3, 0x47, 0x81, 0x95, 0xc4, 0x15, 0x69, 0x1b, 0x5b, 0xa4, 0x38, 0x04, 0x42, 0xc3,
        0xde, 0x61, 0x70, 0x3c, 0x30, 0xf7, 0x88, 0xd9, 0x72, 0x7a, 0x66, 0x19, 0x29, 0xb6, 0xcc,
        0xff, 0xb4, 0x9a, 0xce, 0x2b, 0x16, 0x35, 0xfd, 0x16, 0x57, 0xe1, 0xff, 0xed, 0x7c, 0xca,
        0xb3, 0x7d, 0xe3, 0xdf, 0x1d, 0x10, 0x4d, 0x38, 0x6d, 0x9c, 0xa1, 0xd1, 0x30, 0x2c, 0xd6,
        0x24, 0x15, 0x13, 0x98, 0x89, 0x42, 0x30, 0x39, 0xa9, 0x85, 0x52, 0x1b, 0x0f, 0xff, 0x67,
        0x92, 0xfe, 0xe1, 0x50, 0x54, 0x64, 0xd2, 0xd8, 0x0e, 0xe9, 0xd6, 0x53, 0x12, 0x55, 0xe9,
        0xce, 0x81, 0xed, 0x38, 0xaa, 0x34, 0x9b, 0x83, 0x94, 0x52, 0xab, 0x6a, 0x1d, 0x5b, 0x80,
        0x81, 0x66, 0x11, 0x8b, 0xe5, 0x49, 0xe8, 0xdf, 0x1d, 0x7d, 0x97, 0x64, 0x45, 0xb1, 0x08,
        0x90, 0x90, 0x00, 0x45, 0x06, 0x1a, 0x22, 0x91, 0x64, 0x02, 0x00, 0xf8, 0x00, 0x4b, 0x12,
        0x85, 0x02, 0x05, 0x88, 0x70, 0x6e, 0x1b, 0x63, 0xb0, 0xbc, 0xbb, 0x80, 0x08, 0x00, 0x8e,
        0x3a, 0xcf, 0x0a, 0x75, 0x47, 0x02, 0x31, 0x1b, 0xaf, 0x3d, 0x7e, 0xaa, 0x49, 0xcb, 0x6a,
        0xf0, 0x7e, 0xcf, 0x9b, 0x42, 0x39, 0x03, 0x5f, 0xb9, 0xad, 0xfd, 0x4b, 0xd8, 0x64, 0x84,
        0x62, 0x6b, 0x19, 0xfd, 0x52, 0x3e, 0xd3, 0x15, 0xc7, 0xcb, 0xb3, 0x78, 0x00, 0xa8, 0x03,
        0x08, 0x90, 0x90, 0x60, 0x41, 0x06, 0x54, 0x22, 0xd1, 0x2c, 0x01, 0x00, 0x00, 0x02, 0x00,
        0x5c, 0x01, 0x4b, 0x12, 0x85, 0x02, 0x05, 0x00, 0x35, 0xc9, 0x89, 0x31, 0x4e, 0xfe, 0x05,
        0x64, 0x00, 0x88, 0x68, 0x97, 0x12, 0x70, 0xd8, 0x12, 0x78, 0x00, 0xe4, 0xe2, 0x01, 0xe0,
        0xc8, 0x93, 0xd1, 0xe0, 0xf5, 0x1a, 0x45, 0x45, 0x7e, 0xcd, 0xc0, 0x2a, 0x7e, 0x9d, 0xfe,
        0xe2, 0x7c, 0x3b, 0xab, 0x66, 0x0f, 0xf1, 0x76, 0x50, 0x0c, 0x72, 0x0c, 0x44, 0xf1, 0x1c,
        0x71, 0xbd, 0xfe, 0xd3, 0x4f, 0xdd, 0x80, 0x27, 0x20, 0x1b, 0x19, 0xf0, 0xff, 0x91, 0x5a,
        0x40, 0x09, 0xc3, 0x93, 0x92, 0x69, 0x6e, 0x05, 0xa3, 0xd3, 0x39, 0x0e, 0x3e, 0x5d, 0x08,
        0x90, 0x90, 0x00, 0x45, 0x06, 0x37, 0x22, 0x91, 0x64, 0x01, 0x00, 0x2c, 0x01, 0x4b, 0x12,
        0x85, 0x02, 0x05, 0x88, 0x70, 0x96, 0x14, 0x63, 0xb0, 0x9a, 0xf9, 0x0b, 0xc8, 0x00, 0xe0,
        0xfb, 0xc5, 0x0b, 0x69, 0x13, 0x10, 0xfe, 0xa2, 0x52, 0x45, 0x96, 0xcb, 0x98, 0x00, 0x5b,
        0x35, 0xcb, 0x07, 0x2e, 0x30, 0xa9, 0x4f, 0x95, 0x07, 0xee, 0xfd, 0x04, 0xcb, 0x05, 0x87,
        0x0a, 0x17, 0x1f, 0x34, 0xbb, 0x0a, 0x98, 0x9b, 0x08, 0x8d, 0x08, 0x47, 0x69, 0x66, 0x9d,
        0x81, 0x13, 0xb3, 0xdc, 0xbc, 0x4b, 0xe3, 0xf0, 0x06, 0x2a, 0xe2, 0x4d, 0x05,
    ];
    let img_traffic = decoder
        .decode(traffic_jxl)
        .expect("decode traffic light animation");
    assert_eq!(img_traffic.width, 50);
    assert_eq!(img_traffic.height, 80);
    assert_eq!(img_traffic.frames.len(), 4);
}
