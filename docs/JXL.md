# JPEG XL (ISO/IEC 18181-1 & 18181-2)

Status: **implemented (v2 decoder + clean-room encoder, CPU + `std::simd`)**. `rustmpeg-image::jxl`
provides a JPEG XL decoder and a clean-room Apache-2.0 encoder written from scratch
in safe Rust (`#![forbid(unsafe_code)]`) with `moxcms` ICCv4/CICP color management.
The AGPL-3.0 `jxl-encoder` crate is strictly excluded by license policy and was not
referenced, transcribed, or linked.

## Scope

- **Container (`ISO/IEC 18181-2`)**:
  - Auto-detection of bare codestreams (`0xFF 0x0A`) and ISO/IEC 18181-2 box
    containers (`JXL `, `ftyp`, `jxll`, `jxlc`, `jxlp`, `Exif`, `xml `, `brob`).
  - Legacy JPEG bitstream reconstruction (`jbrd`) is explicitly rejected with
    `JxlError::Unsupported` per `docs/CONCEPT.md`.
- **Headers and TOC (`ISO/IEC 18181-1` Annex D, E, F)**:
  - `SizeHeader`, `BitDepthHeader` (`8`, `10`, `12`, `16`, and `32`-bit float),
    `ExtraChannelInfo` (alpha, depth, spot, selection mask, black, CFA, thermal),
    `ColourEncoding` (enumerated sRGB, Linear sRGB, Display P3, Rec.2020/BT.2100 PQ,
    Rec.2020/BT.2100 HLG, DCI-P3, BT.709, plus compressed ICC stream decoding and
    encoding via `read_icc` / `decode_icc` and `write_icc` / `encode_icc`),
    `ToneMapping`, `OpsinInverseMatrix`, `AnimationHeader`, `PreviewHeader`,
    `FrameHeader`, `RestorationFilter` (Gaborish and 3-stage Edge-Preserving
    Filter), and `Toc` (single-entry and multi-group/permuted).
- **Entropy coding (`ISO/IEC 18181-1` Annex B & C)**:
  - `HybridUintConfig` (`split_exponent`, `msb_in_token`, `lsb_in_token`),
    canonical prefix histograms (simple 1..4 symbols and complex 2-level code
    lengths), 12-bit rANS with alias table and `0x130000` final state verification,
    2D LZ77 backward references with 120 `SPECIAL_DISTANCES`, and Lehmer-code
    permutations (`read_permutation`).
  - Clean-room `EntropyEncoder`: supports both `JxlEntropyMode::Prefix` and 12-bit
    rANS (`JxlEntropyMode::Ans`, with inverse alias table construction and reverse
    renormalization from `0x0013_0000`), plus 2D LZ77 hash-chain + RLE match
    finding (`lz77: true`), single-stream (`finish`), and multi-group split token
    streams (`encode_split_groups`).
- **Modular engine (`ISO/IEC 18181-1` Annex G & H)**:
  - All 14 spatial predictors (`Zero`, `West`, `North`, `AvgWestAndNorth`,
    `Select`, `Gradient`, `SelfCorrecting`, `NorthEast`, `NorthWest`, `WestWest`,
    `AvgWestAndNorthWest`, `AvgNorthAndNorthWest`, `AvgNorthAndNorthEast`,
    `AvgAll`), `WpHeader` 4-subpredictor weighted self-correcting state machine,
    MA decision trees (`MaConfig`), all 42 reversible colour transforms (`Rct`),
    `Palette` (including `DELTA_PALETTE`), and horizontal/vertical `Squeeze` with
    smooth tendency.
  - Clean-room `encode_modular_subimage_v2` supporting lossless and controlled
    near-lossless (`modular_max_error`) coding across all pixel layouts (`Gray`,
    `GrayAlpha`, `Rgb`, `Rgba`), bit depths (`Uint8`, `Uint16`, `Float32`),
    predictors, automatic lossless `Palette` extraction (`<= 256` colors),
    multi-level Haar-like `Squeeze` decomposition, 4-context `MaConfig` adaptive
    decision trees, and YCoCg RCT (`rct_type = 6`).
- **VarDCT engine (`ISO/IEC 18181-1` Annex I & J)**:
  - `LfChannelDequantization`, `Quantizer`, `HfBlockContext`,
    `LfChannelCorrelation`, all 27 `TransformType`s (`Dct8` through `Dct256`,
    `Hornuss`, `Dct2`, `Dct4`, `Afv0..3`), `DequantMatrixSet`, `DecodedLfGroup`,
    `HfGlobal`, `HfPass`, `decode_hf_pass_group`, recursive 2D DCT/IDCT,
    chroma-from-luma (`CfL`), adaptive LF smoothing, `std::simd::f32x4` Gaborish
    deblocking, 3-stage Edge-Preserving Filter (`EPF`), hierarchical `LfFrame`
    reconstruction, and CIE 1931 XYZ + Bradford gamut-adapted XYB ↔ target RGB
    conversions (`sRGB`, `LinearSrgb`, `DisplayP3`, `Bt2020Pq`, `Bt2020Hlg`).
  - Normative Annex J synthesis features in `JxlDecoder`:
    - `Splines` (`parse_splines`, centripetal Catmull-Rom arc-length upsampling,
      32-point 1D continuous IDCT, and 2D Gaussian `erf` rendering).
    - `Noise` (`parse_noise`, `XorShift128Plus` PRNG, and 5x5 high-pass Laplacian
      convolution).
    - `Patches` reference-frame copying/blending.
  - Clean-room `encode_vardct_frame_sections_v2` supporting lossy XYB VarDCT
    encoding with `extra_precision` DC, least-squares 64x64 `CfL` (`x_from_y`,
    `b_from_y`) estimation, adaptive `Dct16`/`Dct8` varblock selection,
    activity-masked per-block `hf_mul` quantization, inverse-Gaborish
    pre-sharpening (`apply_inverse_gaborish`), EPF `sharpness` map estimation,
    12-bit rANS + LZ77 entropy coding, single- and multi-group TOC layouts, and
    lossless Modular alpha extra channels.
- **Color management (`crates/rustmpeg-image/src/cms.rs`)**:
  - Pure-Rust ICCv4 and CICP color transforms built on `moxcms` (`CmsProfile`,
    `CmsTransform`, `CmsRenderingIntent`).

## API

```rust
use rustmpeg_image::{
    JxlConfig, JxlDecoder, JxlEncoder, JxlFrame, JxlImage, JxlPixelLayout,
};

let frame = JxlFrame::new_u8(64, 64, JxlPixelLayout::Rgba, vec![128u8; 64 * 64 * 4])?;
let image = JxlImage::from_single_frame(frame);

// High-compression v2 Lossless Modular encode + decode (rANS + LZ77 + Palette + MA tree)
let jxl_bytes = JxlEncoder::new(JxlConfig::lossless_v2()).encode(&image)?;
let decoded = JxlDecoder::new().decode(&jxl_bytes)?;

// Full v2 Lossy VarDCT encode + VideoFrame decode (rANS + CfL + Dct16/Dct8 + Gaborish + EPF)
let lossy_bytes = JxlEncoder::new(JxlConfig::lossy_v2(90.0)).encode(&image)?;
let video_frame = JxlDecoder::new().decode_video_frame(&lossy_bytes)?;
# Ok::<(), rustmpeg_image::JxlError>(())
```

## Implementation

`crates/rustmpeg-image/src/`:

```text
src/
  lib.rs            crate root (#![forbid(unsafe_code)], #![feature(portable_simd)])
  cms.rs            pure-Rust ICCv4 / CICP color management via moxcms
  jxl/
    mod.rs          public types (JxlImage, JxlFrame, JxlConfig, JxlEntropyMode, JxlError)
    bits.rs         LSB-first BitReader and BitWriter (U32, U64, F16, Enum, Name)
    container.rs    ISO/IEC 18181-2 container parser/writer and jbrd rejection
    entropy.rs      HybridUint, PrefixHistogram, 12-bit rANS decoder/encoder, 2D LZ77
    headers.rs      ImageHeader, ColourEncoding, read_icc/write_icc, FrameHeader, Toc
    modular.rs      14 predictors, SelfCorrecting, MaConfig, Rct/Palette/Squeeze, v2 encoder
    vardct.rs       27 TransformTypes, DequantMatrixSet, 2D DCT/IDCT, XYB/gamut, Gaborish/EPF, v2 encoder
    decoder.rs      JxlDecoder (canvas blending, patches, splines, noise, LfFrame, orientation, ICC)
    encoder.rs      clean-room JxlEncoder (v1 & v2 ModularLossless and VarDctLossy)
```

## Verification

- `crates/rustmpeg-image/tests/jxl.rs` and module unit tests (23 tests total):
  - Bit-exact round-trip for `ModularLossless` (v1 and v2) across `Gray`,
    `GrayAlpha`, `Rgb`, and `Rgba` layouts, all spatial predictors, 12-bit rANS,
    2D LZ77, automatic `Palette`, `Squeeze`, 4-context `MaConfig` tree, controlled
    near-lossless (`modular_max_error`), 8-bit, 16-bit, and 32-bit float depths,
    single-group and multi-group (`300x280`) frames, ISO/IEC 18181-2 container
    boxes (`Exif`, `xml `), multi-frame animation, and `VideoFrame` conversion.
  - Lossy `VarDctLossy` (v1 and v2) round-trip on single-group (`64x64` RGBA with
    lossless Modular alpha) and multi-group (`272x264` RGB) frames, wide-gamut/HDR
    encodings (`DisplayP3`, `Bt2020Pq`, `Bt2020Hlg`), and embedded Annex E ICC
    profiles (`JxlColorEncoding::Icc`).
  - Pixel-by-pixel verification against the standalone `jxl-oxide 0.12.6`
    reference decoder across all conformance vectors and encoder outputs:
    - `lossless_alpha.jxl` (`1,085,280` samples): `max_diff = 0`, `MAE = 0.000000`, `exact = 100.00%`
    - `lossless_modular.jxl` (`813,960` samples): `max_diff = 0`, `MAE = 0.000000`, `exact = 100.00%`
    - `official_blending_cropped_traffic_light.jxl` (`16,000` samples): `max_diff = 0`, `MAE = 0.000000`, `exact = 100.00%`
    - `official_container_extended_size.jxl` (`192` samples): `max_diff = 0`, `MAE = 0.000000`, `exact = 100.00%`
    - `official_pq_gradient.jxl` (`69,632` samples): `max_diff = 0`, `MAE = 0.000000`, `exact = 100.00%`
    - `official_splines.jxl` (`12,582,912` samples): `max_diff = 0`, `MAE = 0.000000`, `exact = 100.00%`
    - `animation_lossy.jxl` (`25,200` samples): `max_diff = 1`, `MAE = 0.001786`, `exact = 99.82%`
    - `lossy_vardct_d1.jxl` (`813,960` samples): `max_diff = 1`, `MAE = 0.007513`, `exact = 99.25%`
    - `out_lossless_v2.jxl` (`16,384` samples): `max_diff = 0`, `MAE = 0.000000`, `exact = 100.00%` (39.4% smaller than v1)
    - `out_lossy_v2.jxl` (`16,384` samples): `max_diff = 1`, `MAE = 0.004639`, `exact = 99.54%` (26.7% smaller than v1)
