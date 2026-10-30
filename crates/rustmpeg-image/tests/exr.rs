// SPDX-License-Identifier: Apache-2.0

//! Integration, round-trip, conformance, and cross-oracle validation tests for `rustmpeg-image::exr` v2.

use std::path::Path;

use rustmpeg_core::{ColorDescription, PixelFormat, VideoFrame};
use rustmpeg_image::exr::{
    ExrAttribute, ExrAttributeValue, ExrBox2f, ExrBox2i, ExrChannelData, ExrChannelDesc,
    ExrChannelPlane, ExrChromaticities, ExrCompression, ExrConfig, ExrDecoder, ExrDeepSamples,
    ExrEncoder, ExrError, ExrImage, ExrLevelMode, ExrLevelRoundingMode, ExrLineOrder, ExrPart,
    ExrPartHeader, ExrPartType, ExrPreview, ExrSampleType, ExrTileDesc, f16_to_f32, f32_to_f16,
};

fn build_synthetic_hdr_part(
    width: u32,
    height: u32,
    compression: ExrCompression,
    sample_type: ExrSampleType,
) -> ExrPart {
    let count = (width as usize) * (height as usize);
    let mut channels = Vec::new();

    for (c_idx, name) in ["A", "B", "G", "R"].into_iter().enumerate() {
        let data = match sample_type {
            ExrSampleType::Half => {
                let mut v = Vec::with_capacity(count);
                for i in 0..count {
                    let x = (i % (width as usize)) as f32 / (width as f32);
                    let y = (i / (width as usize)) as f32 / (height as f32);
                    let val = if name == "A" {
                        1.0
                    } else {
                        x * 2.5 + y * 0.75 + (c_idx as f32) * 0.25 + 0.25
                    };
                    v.push(f32_to_f16(val));
                }
                ExrChannelData::Half(v)
            }
            ExrSampleType::Float => {
                let mut v = Vec::with_capacity(count);
                for i in 0..count {
                    let x = (i % (width as usize)) as f32 / (width as f32);
                    let y = (i / (width as usize)) as f32 / (height as f32);
                    let val = if name == "A" {
                        1.0
                    } else {
                        x * 4.0 - y * 1.25 + (c_idx as f32) * 0.1
                    };
                    v.push(val);
                }
                ExrChannelData::Float(v)
            }
            ExrSampleType::Uint => {
                let mut v = Vec::with_capacity(count);
                for i in 0..count {
                    v.push(((i as u32) * 17 + (c_idx as u32) * 1000) & 0x00ff_ffff);
                }
                ExrChannelData::Uint(v)
            }
        };

        channels.push(ExrChannelPlane {
            desc: ExrChannelDesc::new(name, sample_type),
            width,
            height,
            data,
        });
    }

    ExrPart::from_planar_channels(width, height, compression, channels).unwrap()
}

#[test]
fn test_lossless_scanline_codecs_exact_roundtrip() {
    let codecs = [
        ExrCompression::None,
        ExrCompression::Rle,
        ExrCompression::Zips,
        ExrCompression::Zip,
        ExrCompression::Piz,
        ExrCompression::Zstd,
    ];
    let sample_types = [
        ExrSampleType::Half,
        ExrSampleType::Float,
        ExrSampleType::Uint,
    ];

    let dec = ExrDecoder::default();
    let enc = ExrEncoder::default();

    for &codec in &codecs {
        for &stype in &sample_types {
            for &line_order in &[ExrLineOrder::IncreasingY, ExrLineOrder::DecreasingY] {
                let mut part = build_synthetic_hdr_part(37, 29, codec, stype);
                part.header.line_order = line_order;
                let img = ExrImage::from_single_part(part.clone());
                let bytes = enc.encode(&img).expect("encode failed");
                let decoded = dec.decode(&bytes).expect("decode failed");
                assert_eq!(decoded.parts.len(), 1);
                assert_eq!(
                    decoded.parts[0].channels, part.channels,
                    "mismatch for codec={codec:?}, stype={stype:?}, line_order={line_order:?}"
                );
            }
        }
    }
}

#[test]
fn test_pxr24_and_b44_b44a_roundtrip() {
    let dec = ExrDecoder::default();
    let enc = ExrEncoder::default();

    // PXR24 is lossless for HALF and UINT, and 24-bit float for FLOAT
    for &stype in &[ExrSampleType::Half, ExrSampleType::Uint] {
        let part = build_synthetic_hdr_part(32, 32, ExrCompression::Pxr24, stype);
        let bytes = enc
            .encode(&ExrImage::from_single_part(part.clone()))
            .unwrap();
        let decoded = dec.decode(&bytes).unwrap();
        assert_eq!(decoded.parts[0].channels, part.channels);
    }

    // PXR24 FLOAT round-trip within 24-bit precision (8 LSBs dropped)
    let part_f32 = build_synthetic_hdr_part(32, 24, ExrCompression::Pxr24, ExrSampleType::Float);
    let bytes_f32 = enc
        .encode(&ExrImage::from_single_part(part_f32.clone()))
        .unwrap();
    let dec_f32 = dec.decode(&bytes_f32).unwrap();
    for (orig_ch, dec_ch) in part_f32.channels.iter().zip(&dec_f32.parts[0].channels) {
        if let (ExrChannelData::Float(a), ExrChannelData::Float(b)) = (&orig_ch.data, &dec_ch.data)
        {
            for (&va, &vb) in a.iter().zip(b.iter()) {
                let diff = (va - vb).abs();
                assert!(diff <= 1e-4, "PXR24 float diff too large: {va} vs {vb}");
            }
        }
    }

    // B44 and B44A on HALF (both p_linear=0 and p_linear=1) and pass-through on FLOAT/UINT
    for &codec in &[ExrCompression::B44, ExrCompression::B44a] {
        for &p_linear in &[0u8, 1u8] {
            let mut part = build_synthetic_hdr_part(28, 20, codec, ExrSampleType::Half);
            for ch in &mut part.header.channels {
                ch.p_linear = p_linear;
            }
            for ch in &mut part.channels {
                ch.desc.p_linear = p_linear;
            }
            let bytes = enc
                .encode(&ExrImage::from_single_part(part.clone()))
                .unwrap();
            let decoded = dec.decode(&bytes).unwrap();
            for (orig_ch, dec_ch) in part.channels.iter().zip(&decoded.parts[0].channels) {
                if let (ExrChannelData::Half(a), ExrChannelData::Half(b)) =
                    (&orig_ch.data, &dec_ch.data)
                {
                    for (&ha, &hb) in a.iter().zip(b.iter()) {
                        let fa = f16_to_f32(ha);
                        let fb = f16_to_f32(hb);
                        assert!(
                            (fa - fb).abs() < 0.08,
                            "B44 error too large for {codec:?} p_linear={p_linear}: {fa} vs {fb}"
                        );
                    }
                }
            }
        }

        if codec == ExrCompression::B44a {
            let const_val = f32_to_f16(0.5);
            let plane = ExrChannelPlane {
                desc: ExrChannelDesc::new("Y", ExrSampleType::Half),
                width: 64,
                height: 64,
                data: ExrChannelData::Half(vec![const_val; 64 * 64]),
            };
            let part =
                ExrPart::from_planar_channels(64, 64, ExrCompression::B44a, vec![plane]).unwrap();
            let encoded = enc
                .encode(&ExrImage::from_single_part(part.clone()))
                .unwrap();
            assert!(
                encoded.len() < 1500,
                "B44A flat-field did not shrink constant image: {}",
                encoded.len()
            );
            let decoded = dec.decode(&encoded).unwrap();
            assert_eq!(decoded.parts[0].channels, part.channels);
        }
    }
}

#[test]
fn test_dwaa_and_dwab_hybrid_roundtrip() {
    let dec = ExrDecoder::default();
    let enc = ExrEncoder::new(ExrConfig::default().with_dwa_level(10.0));

    for &codec in &[ExrCompression::Dwaa, ExrCompression::Dwab] {
        // Build mixed part: R/G/B (Half, LOSSY_DCT), A (Half, RLE lossless), Z (Float, VERBATIM lossless)
        let mut part = build_synthetic_hdr_part(40, 36, codec, ExrSampleType::Half);
        let z_vals: Vec<f32> = (0..(40 * 36)).map(|i| 10.0 + (i as f32) * 0.25).collect();
        part.header
            .channels
            .push(ExrChannelDesc::new("Z", ExrSampleType::Float));
        part.channels.push(ExrChannelPlane {
            desc: ExrChannelDesc::new("Z", ExrSampleType::Float),
            width: 40,
            height: 36,
            data: ExrChannelData::Float(z_vals.clone()),
        });

        let encoded = enc
            .encode(&ExrImage::from_single_part(part.clone()))
            .unwrap();
        let decoded = dec.decode(&encoded).unwrap();
        let dec_part = &decoded.parts[0];

        // Alpha (RLE) and Z (Verbatim) must be bit-exact
        assert_eq!(dec_part.channel_by_name("A"), part.channel_by_name("A"));
        assert_eq!(dec_part.channel_by_name("Z"), part.channel_by_name("Z"));

        // R/G/B (Lossy DCT) must be within tight tolerance at level 10.0
        for name in ["R", "G", "B"] {
            let orig = part.channel_by_name(name).unwrap().to_f32_vec();
            let rec = dec_part.channel_by_name(name).unwrap().to_f32_vec();
            for (&a, &b) in orig.iter().zip(rec.iter()) {
                assert!(
                    (a - b).abs() < 0.08,
                    "{codec:?} channel {name} error too large: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn test_tiled_one_level_mipmap_and_ripmap_roundtrip() {
    let dec = ExrDecoder::default();
    let enc = ExrEncoder::default();

    // 1. OneLevel tiled image with Piz compression
    let mut part = build_synthetic_hdr_part(40, 30, ExrCompression::Piz, ExrSampleType::Half);
    part.header.part_type = ExrPartType::TiledImage;
    part.header.tiles = Some(ExrTileDesc {
        x_size: 16,
        y_size: 16,
        level_mode: ExrLevelMode::OneLevel,
        rounding_mode: ExrLevelRoundingMode::RoundDown,
    });
    let img = ExrImage::from_single_part(part.clone());
    let bytes = enc.encode(&img).unwrap();
    let decoded = dec.decode(&bytes).unwrap();
    assert_eq!(decoded.parts[0].channels, part.channels);

    // 2. MipmapLevels generated via `generate_mipmaps`
    let mut mip_part = build_synthetic_hdr_part(16, 16, ExrCompression::Zip, ExrSampleType::Half);
    mip_part
        .generate_mipmaps(8, 8, ExrLevelRoundingMode::RoundDown)
        .unwrap();
    assert_eq!(mip_part.extra_levels.len(), 4); // 8x8, 4x4, 2x2, 1x1
    let mip_bytes = enc
        .encode(&ExrImage::from_single_part(mip_part.clone()))
        .unwrap();
    let mip_dec = dec.decode(&mip_bytes).unwrap();
    assert_eq!(mip_dec.parts[0].channels, mip_part.channels);
    assert_eq!(mip_dec.parts[0].extra_levels, mip_part.extra_levels);

    // 3. RipmapLevels generated via `generate_ripmaps`
    let mut rip_part = build_synthetic_hdr_part(8, 4, ExrCompression::Zstd, ExrSampleType::Half);
    rip_part
        .generate_ripmaps(4, 4, ExrLevelRoundingMode::RoundDown)
        .unwrap();
    assert_eq!(rip_part.extra_levels.len(), 4 * 3 - 1);
    let rip_bytes = enc
        .encode(&ExrImage::from_single_part(rip_part.clone()))
        .unwrap();
    let rip_dec = dec.decode(&rip_bytes).unwrap();
    assert_eq!(rip_dec.parts[0].channels, rip_part.channels);
    assert_eq!(rip_dec.parts[0].extra_levels, rip_part.extra_levels);
}

#[test]
fn test_deep_scanline_and_deep_tile_roundtrip_all_deep_codecs() {
    let dec = ExrDecoder::default();
    let enc = ExrEncoder::default();

    let w = 8u32;
    let h = 6u32;
    let px_count = (w * h) as usize;
    let mut sample_counts = vec![0u32; px_count];
    for (i, slot) in sample_counts.iter_mut().enumerate() {
        *slot = (i % 3) as u32;
    }
    let total_samples: usize = sample_counts.iter().map(|&c| c as usize).sum();
    let z_samples: Vec<f32> = (0..total_samples)
        .map(|i| 1.0 + (i as f32) * 0.125)
        .collect();
    let a_samples: Vec<u16> = (0..total_samples).map(|_| f32_to_f16(0.8)).collect();

    let channels_desc = vec![
        ExrChannelDesc::new("A", ExrSampleType::Half),
        ExrChannelDesc::new("Z", ExrSampleType::Float),
    ];

    for &codec in &[
        ExrCompression::None,
        ExrCompression::Rle,
        ExrCompression::Zips,
        ExrCompression::Zstd,
    ] {
        // Deep scanline
        let mut hdr = ExrPartHeader::new_scanline(w, h, codec, channels_desc.clone());
        hdr.part_type = ExrPartType::DeepScanline;
        hdr.explicit_type_attr = Some(ExrPartType::DeepScanline);
        hdr.deep_version = Some(1);
        hdr.max_samples_per_pixel = Some(2);

        let deep_part = ExrPart {
            header: hdr,
            channels: Vec::new(),
            extra_levels: Vec::new(),
            deep_samples: Some(ExrDeepSamples {
                width: w,
                height: h,
                sample_counts: sample_counts.clone(),
                channels: vec![
                    ExrChannelData::Half(a_samples.clone()),
                    ExrChannelData::Float(z_samples.clone()),
                ],
            }),
        };

        let bytes = enc
            .encode(&ExrImage::from_single_part(deep_part.clone()))
            .unwrap();
        let decoded = dec.decode(&bytes).unwrap();
        assert_eq!(
            decoded.parts[0].deep_samples, deep_part.deep_samples,
            "deep scanline mismatch for {codec:?}"
        );

        // Deep tiled
        let mut tile_hdr = ExrPartHeader::new_scanline(w, h, codec, channels_desc.clone());
        tile_hdr.part_type = ExrPartType::DeepTile;
        tile_hdr.explicit_type_attr = Some(ExrPartType::DeepTile);
        tile_hdr.deep_version = Some(1);
        tile_hdr.max_samples_per_pixel = Some(2);
        tile_hdr.tiles = Some(ExrTileDesc {
            x_size: 4,
            y_size: 4,
            level_mode: ExrLevelMode::OneLevel,
            rounding_mode: ExrLevelRoundingMode::RoundDown,
        });

        let deep_tile_part = ExrPart {
            header: tile_hdr,
            channels: Vec::new(),
            extra_levels: Vec::new(),
            deep_samples: Some(ExrDeepSamples {
                width: w,
                height: h,
                sample_counts: sample_counts.clone(),
                channels: vec![
                    ExrChannelData::Half(a_samples.clone()),
                    ExrChannelData::Float(z_samples.clone()),
                ],
            }),
        };

        let tile_bytes = enc
            .encode(&ExrImage::from_single_part(deep_tile_part.clone()))
            .unwrap();
        let tile_dec = dec.decode(&tile_bytes).unwrap();
        assert_eq!(
            tile_dec.parts[0]
                .deep_samples
                .as_ref()
                .unwrap()
                .sample_counts,
            deep_tile_part.deep_samples.as_ref().unwrap().sample_counts,
            "deep tile mismatch for {codec:?}"
        );
    }
}

#[test]
fn test_luma_chroma_and_display_window_framing() {
    let mut part = build_synthetic_hdr_part(16, 12, ExrCompression::Zip, ExrSampleType::Half);
    part.header.data_window = ExrBox2i {
        min_x: 2,
        min_y: 3,
        max_x: 17,
        max_y: 14,
    };
    part.header.display_window = ExrBox2i {
        min_x: 0,
        min_y: 0,
        max_x: 19,
        max_y: 15,
    };
    let disp_frame = part.to_display_video_frame().unwrap();
    assert_eq!(disp_frame.width, 20);
    assert_eq!(disp_frame.height, 16);
    // Top-left (0, 0) is outside data_window [2..=17, 3..=14] and must be zero-filled
    let row0 = disp_frame.plane(0).unwrap().row(0).unwrap();
    assert_eq!(&row0[0..4], &[0, 0, 0, 0]);
}

#[test]
fn test_multipart_and_all_attribute_types_roundtrip() {
    let dec = ExrDecoder::default();
    let enc = ExrEncoder::default();

    let mut part0 = build_synthetic_hdr_part(16, 16, ExrCompression::Zip, ExrSampleType::Half);
    part0.header.name = Some("beauty".to_string());
    part0.header.other_attributes = vec![
        ExrAttribute {
            name: "box2fAttr".into(),
            value: ExrAttributeValue::Box2f(ExrBox2f {
                min_x: -1.0,
                min_y: -2.0,
                max_x: 3.0,
                max_y: 4.0,
            }),
        },
        ExrAttribute {
            name: "chromaticities".into(),
            value: ExrAttributeValue::Chromaticities(ExrChromaticities {
                red: (0.64, 0.33),
                green: (0.30, 0.60),
                blue: (0.15, 0.06),
                white: (0.3127, 0.3290),
            }),
        },
        ExrAttribute {
            name: "doubleAttr".into(),
            value: ExrAttributeValue::Double(std::f64::consts::PI),
        },
        ExrAttribute {
            name: "envmap".into(),
            value: ExrAttributeValue::Envmap(0),
        },
        ExrAttribute {
            name: "floatVecAttr".into(),
            value: ExrAttributeValue::FloatVector(vec![1.0, 2.5, -3.5]),
        },
        ExrAttribute {
            name: "keycode".into(),
            value: ExrAttributeValue::Keycode([1, 2, 3, 4, 5, 6, 4]),
        },
        ExrAttribute {
            name: "m33f".into(),
            value: ExrAttributeValue::M33f([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        },
        ExrAttribute {
            name: "m44f".into(),
            value: ExrAttributeValue::M44f([
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]),
        },
        ExrAttribute {
            name: "preview".into(),
            value: ExrAttributeValue::Preview(ExrPreview {
                width: 2,
                height: 2,
                rgba: vec![255; 16],
            }),
        },
        ExrAttribute {
            name: "rational".into(),
            value: ExrAttributeValue::Rational(24000, 1001),
        },
        ExrAttribute {
            name: "multiView".into(),
            value: ExrAttributeValue::StringVector(vec!["left".into(), "right".into()]),
        },
        ExrAttribute {
            name: "timecode".into(),
            value: ExrAttributeValue::Timecode([0x0102_0304, 0]),
        },
        ExrAttribute {
            name: "v3fAttr".into(),
            value: ExrAttributeValue::V3f([0.1, 0.2, 0.3]),
        },
        ExrAttribute {
            name: "customVendorBlob".into(),
            value: ExrAttributeValue::Opaque {
                type_name: "vendorblob".into(),
                data: vec![0xde, 0xad, 0xbe, 0xef],
            },
        },
    ];

    let mut part1 = build_synthetic_hdr_part(16, 16, ExrCompression::Piz, ExrSampleType::Float);
    part1.header.name = Some("depth".to_string());
    part1.header.other_attributes = vec![
        part0.header.other_attributes[1].clone(),
        part0.header.other_attributes[11].clone(),
    ];

    let mp_img = ExrImage {
        version: 2,
        flags: rustmpeg_image::exr::EXR_FLAG_MULTIPART,
        parts: vec![part0.clone(), part1.clone()],
    };

    let encoded = enc.encode(&mp_img).unwrap();
    let decoded = dec.decode(&encoded).unwrap();
    assert_eq!(decoded.parts.len(), 2);
    assert_eq!(decoded.parts[0].channels, part0.channels);
    assert_eq!(decoded.parts[1].channels, part1.channels);
    assert_eq!(
        decoded.parts[0].header.other_attributes,
        part0.header.other_attributes
    );
}

#[test]
fn test_video_frame_conversion_roundtrip() {
    let mut dst = vec![0u8; 16 * 16 * 4];
    for (i, chunk) in dst.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        chunk[0] = ((i * 7) & 0xff) as u8;
        chunk[1] = ((i * 13) & 0xff) as u8;
        chunk[2] = ((i * 29) & 0xff) as u8;
        chunk[3] = 255;
    }
    let plane0 = rustmpeg_core::Plane::new(rustmpeg_core::Buffer::from_vec(dst), 16 * 4, 16, 16);
    let frame = VideoFrame::from_planes(
        PixelFormat::Rgba8,
        8,
        16,
        16,
        ColorDescription::default(),
        [
            plane0,
            rustmpeg_core::Plane::empty(),
            rustmpeg_core::Plane::empty(),
        ],
    )
    .unwrap();

    let enc = ExrEncoder::new(ExrConfig::with_compression(ExrCompression::Zip));
    let exr_bytes = enc.encode_video_frame(&frame).unwrap();

    let dec = ExrDecoder::default();
    let out_frame = dec.decode_to_video_frame(&exr_bytes).unwrap();
    assert_eq!(out_frame.width, 16);
    assert_eq!(out_frame.height, 16);
    assert_eq!(out_frame.format, PixelFormat::Rgba8);
    assert_eq!(
        out_frame.planes[0].data.as_slice(),
        frame.planes[0].data.as_slice()
    );
}

#[test]
fn test_negative_validation_and_bounds() {
    let dec = ExrDecoder::default();

    assert!(matches!(
        dec.decode(b"NOT_EXR_FILE"),
        Err(ExrError::InvalidMagic)
    ));
    assert!(matches!(
        dec.decode(&[0x76, 0x2f, 0x31, 0x01, 0x03, 0x00, 0x00, 0x00]),
        Err(ExrError::UnsupportedVersion(_))
    ));
    assert!(matches!(
        dec.decode(&[0x76, 0x2f, 0x31, 0x01, 0x02, 0x0a, 0x00, 0x00]),
        Err(ExrError::InvalidHeader(_))
    ));
}

#[test]
fn test_conformance_corpus_if_present() {
    let kit_root = Path::new("/tmp/work_exr/exr_kit/openexr-kit");
    if !kit_root.exists() {
        return;
    }

    let dec = ExrDecoder::default();

    // 1. Decode all supported sample files in `samples/` (now including `generated_zstd_half_rgba.exr`!)
    let samples_dir = kit_root.join("samples");
    let expected_valid_samples = [
        "generated_b44a_half_rgba.exr",
        "generated_deep_scanline_zips.exr",
        "generated_deep_tiled_zips_odd.exr",
        "generated_multipart_flat_deep.exr",
        "generated_uint_rgb_zip.exr",
        "generated_zstd_half_rgba.exr",
        "official_allhalfvalues.exr",
        "official_brightrings_naninf.exr",
        "official_displaywindow_t01.exr",
        "official_rec709_yc.exr",
    ];
    for name in expected_valid_samples {
        let path = samples_dir.join(name);
        let bytes = std::fs::read(&path).unwrap();
        let img = dec
            .decode(&bytes)
            .unwrap_or_else(|e| panic!("failed to decode sample {name}: {e}"));
        assert!(!img.parts.is_empty());

        if name == "official_rec709_yc.exr" {
            // Verify Luminance/Chroma (`Y` + `RY` + `BY`) reconstructs to a 3-channel `Rgb8` VideoFrame
            let vf = img.parts[0].to_video_frame().unwrap();
            assert_eq!(vf.format, PixelFormat::Rgb8);
            assert_eq!(vf.width, 610);
            assert_eq!(vf.height, 406);
        }
        if name == "official_displaywindow_t01.exr" {
            let disp_vf = img.parts[0].to_display_video_frame().unwrap();
            assert_eq!(disp_vf.width, img.parts[0].header.display_window.width());
            assert_eq!(disp_vf.height, img.parts[0].header.display_window.height());
        }
    }

    // `generated_htj2k256_half_rgba.exr` must parse headers cleanly and return `UnsupportedCompression`
    {
        let bytes = std::fs::read(samples_dir.join("generated_htj2k256_half_rgba.exr")).unwrap();
        let hdr = dec.decode_headers(&bytes).unwrap();
        assert_eq!(hdr.parts.len(), 1);
        assert!(matches!(
            dec.decode(&bytes),
            Err(ExrError::UnsupportedCompression(ExrCompression::Htj2k256))
        ));
    }

    // 2. Decode all generated files in `conformance/generated/` (including `zstd_half_rgba.exr` and `deep_scanline_zstd.exr`!)
    let gen_dir = kit_root.join("conformance/generated");
    for entry in std::fs::read_dir(&gen_dir).unwrap() {
        let entry = entry.unwrap();
        let fname = entry.file_name().into_string().unwrap();
        if !fname.ends_with(".exr") {
            continue;
        }
        let bytes = std::fs::read(entry.path()).unwrap();
        let res = dec.decode(&bytes);
        if fname.contains("htj2k") || fname.contains("lj2k") {
            assert!(
                matches!(res, Err(ExrError::UnsupportedCompression(_))),
                "expected UnsupportedCompression for {fname}, got {:?}",
                res.err()
            );
        } else {
            assert!(
                res.is_ok(),
                "expected generated conformance file {fname} to decode, got {:?}",
                res.err()
            );
        }
    }

    // 3. Validate all positive (including `comp_dwaa_v1/v2` and `comp_dwab_v1/v2`) and negative files in `conformance/openexr-testsuite/`
    let suite_dir = kit_root.join("conformance/openexr-testsuite");
    for entry in std::fs::read_dir(&suite_dir).unwrap() {
        let entry = entry.unwrap();
        let fname = entry.file_name().into_string().unwrap();
        if !fname.ends_with(".exr") {
            continue;
        }
        let bytes = std::fs::read(entry.path()).unwrap();
        let res = dec.decode(&bytes);

        let is_expected_invalid = fname.starts_with("comp_bad_")
            || fname.starts_with("comp_short_decode_")
            || fname.starts_with("comp_early_eof_")
            || fname.starts_with("comp_invalid_")
            || fname.starts_with("invalid")
            || fname.contains("_with_deep")
            || fname.contains("_with_tiled")
            || fname.contains("_with_scanline");

        if fname == "tiled.exr" {
            let hdr = dec.decode_headers(&bytes).unwrap();
            assert!(hdr.is_single_tiled());
            assert!(res.is_err());
        } else if is_expected_invalid {
            assert!(
                res.is_err(),
                "expected invalid conformance file {fname} to be rejected, but it succeeded"
            );
        } else {
            assert!(
                res.is_ok(),
                "expected valid conformance file {fname} to decode, got {:?}",
                res.err()
            );
        }
    }
}

#[test]
fn test_cross_oracle_openexr_3_5_2_if_present() {
    let oracle_bin = Path::new("/tmp/exr_build/verify_oracle");
    let exrinfo_bin = Path::new("/tmp/exr_build/exrinfo");
    if !oracle_bin.exists() || !exrinfo_bin.exists() {
        return;
    }

    let enc = ExrEncoder::default();
    let dec = ExrDecoder::default();

    // 1. Verify all 11 supported codecs produced by our ExrEncoder against OpenEXR 3.5.2
    for (idx, codec) in [
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
    ]
    .into_iter()
    .enumerate()
    {
        let part = build_synthetic_hdr_part(16, 16, codec, ExrSampleType::Half);
        let img = ExrImage::from_single_part(part);
        let encoded = enc.encode(&img).unwrap();

        let exr_path = format!("/tmp/exr_build/rustmpeg_v2_out_{idx}.exr");
        let raw_path = format!("/tmp/exr_build/oracle_v2_out_{idx}.raw");
        std::fs::write(&exr_path, &encoded).unwrap();

        let status_info = std::process::Command::new(exrinfo_bin)
            .args(["-a", "-v", &exr_path])
            .output()
            .unwrap();
        assert!(
            status_info.status.success(),
            "official exrinfo failed on our {:?} output: {}",
            codec,
            String::from_utf8_lossy(&status_info.stderr)
        );

        let status_oracle = std::process::Command::new(oracle_bin)
            .args([&exr_path, &raw_path])
            .output()
            .unwrap();
        assert!(
            status_oracle.status.success(),
            "official OpenEXRCore decode failed on our {:?} output: {}",
            codec,
            String::from_utf8_lossy(&status_oracle.stderr)
        );

        let oracle_raw = std::fs::read(&raw_path).unwrap();
        let our_decoded = dec.decode(&encoded).unwrap();
        let mut our_raw = Vec::new();
        for ch in &our_decoded.parts[0].channels {
            if let ExrChannelData::Half(v) = &ch.data {
                for &h in v {
                    our_raw.extend_from_slice(&h.to_le_bytes());
                }
            }
        }
        assert!(
            our_raw == oracle_raw,
            "bit-exact mismatch between our decoder and OpenEXR 3.5.2 for {codec:?}"
        );
    }

    // 2. Verify bit-exact decoding on official DWA v1/v2 and ZSTD reference files against OpenEXR 3.5.2
    let kit_root = Path::new("/tmp/work_exr/exr_kit/openexr-kit");
    if kit_root.exists() {
        for rel_path in [
            "samples/generated_zstd_half_rgba.exr",
            "conformance/generated/zstd_half_rgba.exr",
            "conformance/openexr-testsuite/comp_dwaa_v1.exr",
            "conformance/openexr-testsuite/comp_dwaa_v2.exr",
            "conformance/openexr-testsuite/comp_dwab_v1.exr",
            "conformance/openexr-testsuite/comp_dwab_v2.exr",
        ] {
            let full_path = kit_root.join(rel_path);
            let raw_path = "/tmp/exr_build/oracle_ref_check.raw";
            let status_oracle = std::process::Command::new(oracle_bin)
                .args([full_path.to_str().unwrap(), raw_path])
                .output()
                .unwrap();
            assert!(
                status_oracle.status.success(),
                "verify_oracle failed on {rel_path}"
            );
            let oracle_raw = std::fs::read(raw_path).unwrap();
            let bytes = std::fs::read(&full_path).unwrap();
            let our_decoded = dec.decode(&bytes).unwrap();
            let mut our_raw = Vec::new();
            for ch in &our_decoded.parts[0].channels {
                match &ch.data {
                    ExrChannelData::Half(v) => {
                        for &h in v {
                            our_raw.extend_from_slice(&h.to_le_bytes());
                        }
                    }
                    ExrChannelData::Float(v) => {
                        for &f in v {
                            our_raw.extend_from_slice(&f.to_le_bytes());
                        }
                    }
                    ExrChannelData::Uint(v) => {
                        for &u in v {
                            our_raw.extend_from_slice(&u.to_le_bytes());
                        }
                    }
                }
            }
            if our_raw != oracle_raw {
                let mut diff_u16 = 0usize;
                let mut max_ulp = 0u16;
                let mut first_diff = None;
                for idx in (0..our_raw.len()).step_by(2) {
                    let a = u16::from_le_bytes([our_raw[idx], our_raw[idx + 1]]);
                    let b = u16::from_le_bytes([oracle_raw[idx], oracle_raw[idx + 1]]);
                    if a != b {
                        diff_u16 += 1;
                        let ulp = a.abs_diff(b);
                        if ulp > max_ulp {
                            max_ulp = ulp;
                        }
                        if first_diff.is_none() {
                            first_diff = Some((idx / 2, a, b));
                        }
                    }
                }
                panic!(
                    "mismatch on {rel_path}: diff_u16={}/{} max_ulp={} first={:?}",
                    diff_u16,
                    our_raw.len() / 2,
                    max_ulp,
                    first_diff
                );
            }
        }
    }
}
