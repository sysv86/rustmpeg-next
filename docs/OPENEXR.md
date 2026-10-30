# OpenEXR (`rustmpeg-image::exr`) — Architecture, Normative Specification, and Verification

Status: **implemented (v2 decoder + encoder; algorithms and constant tables derived in part from OpenEXR 3.5.2 `OpenEXRCore`, BSD-3-Clause)** in `crates/rustmpeg-image/src/exr/`.

## 1. Scope and Module Layout

`rustmpeg-image::exr` implements the OpenEXR v2/v3 container format, full attribute catalog, single-part and multi-part scanline, tiled (`OneLevel`, `MipmapLevels`, `RipmapLevels`), deep scanline (`deepscanline`), and deep tiled (`deeptile`) pipelines, 11 compression codecs (`None`, `Rle`, `Zips`, `Zip`, `Piz`, `Pxr24`, `B44`, `B44a`, `Dwaa`, `Dwab`, `Zstd`), Luminance/Chroma (`Y` + `RY` + `BY`) subsampled reconstruction and encoding, `displayWindow` vs `dataWindow` framing, and automatic mipmap/ripmap pyramid generation under `#![forbid(unsafe_code)]` and the `Apache-2.0 AND BSD-3-Clause` package license, with zero external C/C++ dependencies. The OpenEXRCore-derived scope is recorded in [`../THIRD_PARTY.md`](../THIRD_PARTY.md).

| File | Responsibility | Normative / Reference Grounding |
| :-- | :-- | :-- |
| `crates/rustmpeg-image/src/exr/mod.rs` | Public types (`ExrImage`, `ExrPart`, `ExrChannelPlane`, `ExrChannelData`, `ExrDeepSamples`, `ExrLevelImage`, `ExrConfig`, `ExrError`) | `openexr-kit/docs/01-file-layout.md` – `07-rust-architecture.md` |
| `crates/rustmpeg-image/src/exr/half.rs` | IEEE 754-2008 binary16 (`f16_to_f32`, `f32_to_f16`), `PXR24` 24-bit float (`float_to_float24`), and `B44` perceptual linearity (`b44_convert_from_linear`, `b44_convert_to_linear`) | IEEE 754-2008; `OpenEXRCore/internal_pxr24.c`, `internal_b44_table_init.c` |
| `crates/rustmpeg-image/src/exr/deflate.rs` | Self-contained RFC 1950 (`zlib`) and RFC 1951 (`DEFLATE`) decompressor (stored, fixed Huffman, dynamic Huffman + `Adler-32`) and LZ77 + dynamic/fixed/stored block selector compressor | IETF RFC 1950, IETF RFC 1951 |
| `crates/rustmpeg-image/src/exr/header.rs` | File magic, version/flags validation, single-part and multi-part headers, 29 standard attribute types + `Opaque` pass-through, `TileDesc` level/tile geometry | `openexr-kit/docs/02-header-and-attributes.md`; `OpenEXRCore/parse_header.c`, `validation.c` |
| `crates/rustmpeg-image/src/exr/compression.rs` | Codecs: `None` (`0`), `Rle` (`1`), `Zips` (`2`), `Zip` (`3`), `Piz` (`4`), `Pxr24` (`5`), `B44` (`6`), `B44a` (`7`), plus dispatch to `Dwaa` (`8`), `Dwab` (`9`), and `Zstd` (`13`) + raw-bypass rule (`packed.len() == expected_uncompressed_size`) | `openexr-kit/docs/05-compression-codecs.md`; `OpenEXRCore/compression.c`, `internal_rle.c`, `internal_zip.c`, `internal_piz.c`, `internal_huf.c`, `internal_pxr24.c`, `internal_b44.c` |
| `crates/rustmpeg-image/src/exr/dwa.rs` | `DWAA` (`8`, 32 scanlines) and `DWAB` (`9`, 256 scanlines) hybrid lossy/lossless codec (versions `0`, `1`, `2`, rule block, closed-form perceptual LUTs, 8x8 DCT/IDCT, BT.709 CSC, `UNKNOWN`/`AC`/`DC`/`RLE` substreams) | `openexr-kit/docs/05-compression-codecs.md`, `openexr-piz-dwa-observer-spec.md` §3; `OpenEXRCore/internal_dwa_*.h`, `internal_dwa_table_init.c` |
| `crates/rustmpeg-image/src/exr/zstd.rs` | `ZSTD` (`13`, 1 scanline) codec: 24-byte `zstd-exr` container (`ZSTD_EXR_MAGIC = 8_248_453_963_162_350_458`), 16-bit/32-bit channel sorting, row-wise delta prediction, Blosc 2-byte/4-byte byte shuffle, and pure-Rust RFC 8878 Zstandard frame decompressor/compressor | `openexr-kit/docs/05-compression-codecs.md`; `OpenEXRCore/internal_zstd.c`; IETF RFC 8878 |
| `crates/rustmpeg-image/src/exr/luma_chroma.rs` | Luminance/Chroma (`Y` + `RY` + `BY`) subsampled color conversion (`BT709_CHROMATICITIES`, CIE 1931 XYZ luminance weights, clamped bilinear upsampling, `[1/4, 1/2, 1/4]` tent filter downsampling) | `openexr-kit/docs/02-header-and-attributes.md`, `04-pixel-pipeline.md`; `OpenEXR/ImfRgbaYca.cpp` |
| `crates/rustmpeg-image/src/exr/decoder.rs` | `ExrDecoder` (`decode_headers`, `decode`, `decode_to_video_frame`, `decode_to_display_video_frame`) for flat/deep scanline/tiled single-part and multi-part files | `OpenEXRCore/decoding.c`, `chunk.c`, `unpack.c` |
| `crates/rustmpeg-image/src/exr/encoder.rs` | `ExrEncoder` (`encode`, `encode_video_frame`), `ExrPart::generate_mipmaps()`, and `ExrPart::generate_ripmaps()` for flat/deep scanline/tiled single-part and multi-part files | `OpenEXRCore/encoding.c`, `chunk.c`, `pack.c` |

## 2. Normative Algorithms and Formulas Implemented (Zero Approximations)

1. **File Signature & Version Flags (`header.rs`)**:
   - Magic bytes `0x76, 0x2f, 0x31, 0x01` (`20000630` LE).
   - Version `1` or `2`; valid flags mask `0x0000_1e00` (`0x200` single-part tiled, `0x400` long names up to 255 bytes, `0x800` deep/non-image, `0x1000` multipart).
   - Enforces `OpenEXRCore/parse_header.c` rule: if `0x200` (single-part tiled) is set, `0x800` (deep) and `0x1000` (multipart) must be `0`.
   - Enforces `OpenEXRCore/validation.c` coordinate bounds (`kLargeVal = i32::MAX / 2 = 1_073_741_823`), `pixelAspectRatio` in `[1e-6, 1e6]`, channel sampling divisibility, and multipart shared-attribute consistency (`displayWindow`, `pixelAspectRatio`, `chromaticities`, `timecode`).

2. **Byte Deconstruction + Delta Prediction (`compression.rs`, used by `Rle`, `Zips`, `Zip`)**:
   - Even bytes placed in `scratch[0..(n+1)/2]`, odd bytes in `scratch[(n+1)/2..n]`.
   - Forward delta predictor: `d = (cur - prev + 128 + 256) & 0xff`.
   - Inverse delta predictor: `cur = (prev + d - 128) & 0xff`.

3. **PackBits `RLE` (`1`, `internal_rle.c`)**:
   - Signed byte `count`: if `count < 0`, copy `-count` literal bytes; if `count >= 0`, replicate the following byte `count + 1` times.

4. **Dynamic + Fixed + Stored `zlib`/`DEFLATE` (`deflate.rs`, IETF RFC 1950 / RFC 1951)**:
   - Decompressor supports `BTYPE=00` (stored), `BTYPE=01` (fixed Huffman), and `BTYPE=10` (dynamic Huffman with code-length alphabet order `[16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]`) plus big-endian `Adler-32` verification.
   - Compressor runs 32 KiB sliding-window LZ77 with hash-chain match finding and evaluates `BTYPE=10` (package-merge length-limited dynamic Huffman with RLE code-length symbols `16`, `17`, `18`), `BTYPE=01` (fixed Huffman), and `BTYPE=00` (stored), emitting whichever produces the smallest compressed bitstream.

5. **`PIZ` (`4`, `internal_piz.c` + `internal_huf.c`)**:
   - 65536-bit presence bitmap (`8192` bytes, `bitmap[0] &= !1` so zero is implicit) + compact forward/reverse LUTs (`max_value`).
   - 2D Haar wavelet (`wav2d_encode` / `wav2d_decode`) applied per channel plane across hierarchy levels `p = 1, 2, 4, ...` while `(p << 1) <= min(nx, ny)`:
     - Uses `wenc14` / `wdec14` when `max_value < (1 << 14)` and `wenc16` / `wdec16` (`A_OFFSET = M_OFFSET = 32768`, `MOD_MASK = 65535`) otherwise.
   - 16-bit canonical Huffman (`Huf`, `HUF_ENCSIZE = 65537`, maximum code length `58` bits, table zero-run codes `59..=62` for 2–5 zeros and `63` + 8-bit count for 6–261 zeros, symbol `iM` reserved as run-length escape for 8-bit repeat count).

6. **`PXR24` (`5`, `internal_pxr24.c`)**:
   - Lossless big-endian byte-plane delta transposition for `UINT` (4 planes) and `HALF` (2 planes).
   - Round-to-nearest-even 32-bit `f32` to 24-bit float conversion (`float_to_float24`) for `FLOAT` (3 planes), followed by RFC 1950 `zlib` compression.

7. **`B44` (`6`) and `B44A` (`7`, `internal_b44.c` + `internal_b44_table_init.c`)**:
   - Optional perceptual linearity pre/post-transform when channel `p_linear == 1`:
     - Forward (`b44_convert_from_linear`): `0` for Inf/NaN, `0x7bff` (`HALF_MAX`) for `0x558c..0x8000`, else `f32_to_f16((f16_to_f32(x) / 8.0).exp())`.
     - Inverse (`b44_convert_to_linear`): `0` for Inf/NaN or `x > 0x8000`, else `f32_to_f16(8.0 * f16_to_f32(x).ln())`.
   - 4x4 block sign-magnitude-to-monotonic mapping (`Nan/Inf -> 0x8000`, negative `-> !s`, positive `-> s | 0x8000`), `shift_and_round`, 15 running differences biased by `+32` (`0x20`) packed into 14 bytes (`B44`) or 3 bytes (`0xfc` tag in `B44A` when all 16 samples are identical).

8. **`DWAA` (`8`, 32 scanlines) and `DWAB` (`9`, 256 scanlines, `dwa.rs`)**:
   - **88-byte Chunk Header**: 11 little-endian `u64` counters (`version`, `unknown_uncompressed_size`, `unknown_compressed_size`, `ac_compressed_size`, `dc_compressed_size`, `rle_compressed_size`, `rle_uncompressed_size`, `rle_raw_size`, `ac_count`, `dc_count`, `ac_compression`).
   - **Channel Classification**:
     - `version < 2`: 11 fixed case-insensitive prefix rules (`R`, `G`, `B`, `Y`, `RY`, `BY` -> `LOSSY_DCT`; `A` -> `RLE`; `U`, `V` -> `LOSSY_DCT`).
     - `version >= 2`: serialized rule block (`u16` byte length + NUL-terminated suffix rules) matching the 43 standard suffix rules in `OpenEXRCore/internal_dwa_classifier.h`.
   - **Closed-Form Perceptual LUTs (`dwa_to_nonlinear`, `dwa_to_linear`, `internal_dwa_table_init.c`)**:
     - For `|v| <= 1.0`: `to_nonlinear(v) = sign(v) * |v|^(1 / 2.2)`, `to_linear(v) = sign(v) * |v|^2.2`.
     - For `|v| > 1.0`: `to_nonlinear(v) = sign(v) * (ln(|v|) / 2.2 + 1.0)`, `to_linear(v) = sign(v) * exp(2.2)^(|v| - 1.0)` where `exp(2.2) = 9.02501329156`.
     - Includes the exact 8 single-precision `powf`/`logf` vs `f64` boundary patches so both 65,536-entry tables are 100% bit-identical to `exrcore_dwaToNonLinearTable` and `exrcore_dwaToLinearTable` in `OpenEXRCore/internal_dwa_table.c`.
   - **8x8 Forward & Inverse DCT (`internal_dwa_simd.h`)**:
     - Butterfly constants `DCT_A..DCT_G` (`0x3eb504fb`, `0x3efb14bf`, `0x3eec8361`, `0x3ed4db36`, `0x3e8e39e5`, `0x3e43ef33`, `0x3dc7c60b`) matching `.5f * cosf(k * 3.14159f / 16.0f)` in `dctInverse8x8_scalar`, and `DCT_DC_ONLY = 3.535536e-1` (`0x3eb504fa`) matching `dctInverse8x8DcOnly`.
     - Note on upstream x86_64 inline AVX assembly (`OPENEXR_IMF_HAVE_GCC_INLINE_ASM_AVX`): OpenEXR 3.5.2's `fromHalfZigZag_f16c` inline assembly in `internal_dwa_simd.h` (line 609) duplicates zig-zag index `61` into raster index `(6, 7)` instead of `63` and uses truncated decimal constants (`0x3eb504fa` etc.). `rustmpeg-image` implements the exact portable scalar specification (`dctInverse8x8_scalar` + `fromHalfZigZag_scalar`), achieving 0 ULP (`100.00%` bit-exact across 1,584,900 samples per image) against portable OpenEXR 3.5.2.
   - **BT.709 Color Space Conversion (`csc_forward`, `csc_inverse`)**:
     - Forward: `Y' = 0.2126 R' + 0.7152 G' + 0.0722 B'`, `Cb = -0.1146 R' - 0.3854 G' + 0.5000 B'`, `Cr = 0.5000 R' - 0.4542 G' - 0.0458 B'`.
     - Inverse: `R' = Y' + 1.5747 Cr`, `G' = Y' - 0.1873 Cb - 0.4682 Cr`, `B' = Y' + 1.8556 Cb`.
   - **JPEG Annex K Quantization + Smallest-Bitrate Alternate Selection (`internal_dwa_encoder.h`)**:
     - Base error `base_error = dwaCompressionLevel / 100_000.0` (default `45.0`).
     - Evaluates truncated-sign-magnitude candidates up to 3 neighbors (`trunc1`, `round`, `trunc2`) and selects the candidate with the smallest nonzero bitcount within tolerance `t`, zeroing bit-0 when `abs(v_Clear0 - src) < t`.

9. **`ZSTD` (`13`, 1 scanline, `zstd.rs`)**:
   - **24-Byte `zstd-exr` Container (`OpenEXRCore/internal_zstd.c`)**:
     - `[0..8]`: `magic = 8_248_453_963_162_350_458` (`0x7278_652d_6474_737a`, ASCII `"zstd-exr"` LE).
     - `[8..16]`: `num_two_byte_elems` (`u64` LE); `[16..24]`: `num_four_byte_elems` (`u64` LE).
   - **Three Grid Scenarios**:
     - `DeepSampleCountTable`: treats cumulative `i32` sample counts as a `width x height` 4-byte grid (`num_four_byte_elems = width * height`).
     - `DeepPixels`: splits per-scanline channel runs into `u16` (`Half`) and `u32` (`Float`/`Uint`) pools (`1D` row delta).
     - `FlatScanOrTile`: reorders scanline-interleaved channels into channel-major planes (`sort_2_4_byte_channels`, setting `b2_stride = nx` when uniform across all 2-byte channels and `1` otherwise), applies 2D row-wise delta prediction (`dst[i] = row[i].wrapping_sub(row[i - 1])`), applies Blosc 2-byte (`even`/`odd`) and 4-byte (`b0`/`b1`/`b2`/`b3`) planar byte shuffle, and compresses the combined stream as an RFC 8878 Zstandard frame (`0xFD2FB528`).
   - **Pure-Rust RFC 8878 Zstandard Frame Decoder & Encoder**:
     - Decompresses `Raw_Block` (`0`), `RLE_Block` (`1`), and `Compressed_Block` (`2`) with `Raw`/`RLE`/`Compressed`/`Treeless` Huffman literals (`1X1` and `4X1` streams with FSE-compressed Huffman weights) and `Predefined`/`RLE`/`FSE_Compressed`/`Repeat` sequence tables (`LL`, `OF`, `ML`) + 3-offset repeat history (`[1, 4, 8]`) + optional low-32-bit `XXH64` content checksum verification.
     - Compresses payloads using LZ77 + `Predefined_Mode` FSE sequence encoding (falling back to `Raw_Block` when incompressible).

10. **Luminance/Chroma (`Y` + `RY` + `BY`) & Display Window Framing (`luma_chroma.rs`, `mod.rs`)**:
    - Computes luminance weights `yw = Y_white * inv(M_RGB_to_XYZ)[1]` from the header `chromaticities` attribute (defaulting to Rec.709 `[0.2126390, 0.7151687, 0.0721923]`).
    - Reconstructs full-resolution linear `RGB` from `Y` and subsampled `RY`/`BY` (`x_sampling = y_sampling = 2`) using clamped bilinear upsampling and `R = Y * (1 + RY)`, `B = Y * (1 + BY)`, `G = (Y - R * yw.r - B * yw.b) / yw.g`.
    - `ExrPart::to_display_video_frame()` maps `dataWindow` coordinates into `displayWindow` canvas coordinates (cropping overscan and zero-filling outside `dataWindow`).
    - `ExrPart::generate_mipmaps()` and `ExrPart::generate_ripmaps()` generate complete `MipmapLevels` (`lx == ly`) and `RipmapLevels` (`lx x ly`) resolution pyramids using 2x2 / 2x1 / 1x2 clamped box filtering aligned with `TileDesc.rounding_mode` (`RoundDown` / `RoundUp`).

11. **Deep Data (`deepscanline` and `deeptile`, `OpenEXRCore/decoding.c`)**:
    - Per-scanline cumulative `i32` sample count table (`prev_cum` resets to `0` at the beginning of each scanline row inside a deep chunk), compressed with the part's codec (`None`, `Rle`, `Zips`, `Zstd`).

## 3. Verification and Cross-Oracle Results

Verified against the `openexr-kit` conformance corpus and the official **OpenEXR 3.5.2** C reference library (`libOpenEXRCore` + `exrinfo`):

- **Unit & Integration Tests (`cargo test -p rustmpeg-image`)**: 48 tests pass (14 `exr` unit tests + 11 `exr` integration/conformance/oracle tests + 23 `jxl`/`cms` tests).
- **Cross-Oracle Bit-Exactness (`test_cross_oracle_openexr_3_5_2_if_present`)**:
  - **Part 1 (Encoder -> Oracle)**: Encodes `16x16` RGBA `HALF` images across all 11 codecs (`None`, `Rle`, `Zips`, `Zip`, `Piz`, `Pxr24`, `B44`, `B44a`, `Dwaa`, `Dwab`, `Zstd`) with `ExrEncoder`, validates every file with OpenEXR 3.5.2 `exrinfo -a -v`, and verifies **100% byte-for-byte identity (`0 ULP`)** between `libOpenEXRCore` (`exr_decoding_run`) and `ExrDecoder`.
  - **Part 2 (Reference Corpus -> Oracle vs `ExrDecoder`)**: Decodes official reference files `conformance/generated/zstd_half_rgba.exr`, `conformance/openexr-testsuite/comp_dwaa_v1.exr`, `comp_dwaa_v2.exr`, `comp_dwab_v1.exr`, and `comp_dwab_v2.exr` with both OpenEXR 3.5.2 (`libOpenEXRCore`) and `ExrDecoder`, achieving **100% byte-for-byte identity (`0 ULP` across all 1,584,900 samples per image)**.
- **Conformance Suite (`test_conformance_corpus_if_present`)**:
  - Decodes all 11 files in `openexr-kit/samples/` (`generated_b44a_half_rgba.exr`, `generated_deep_scanline_zips.exr`, `generated_deep_tiled_zips_odd.exr`, `generated_htj2k256_half_rgba.exr` (headers only, `UnsupportedCompression`), `generated_multipart_flat_deep.exr`, `generated_uint_rgb_zip.exr`, `generated_zstd_half_rgba.exr`, `official_allhalfvalues.exr`, `official_brightrings_naninf.exr`, `official_displaywindow_t01.exr`, `official_rec709_yc.exr`), verifying `official_rec709_yc.exr` chroma reconstruction and `official_displaywindow_t01.exr` display-window framing.
  - Decodes all 13 files in `openexr-kit/conformance/generated/` (`b44a_half_rgba.exr`, `deep_scanline_none.exr`, `deep_scanline_rle.exr`, `deep_scanline_zips.exr`, `deep_scanline_zstd.exr`, `deep_tiled_zips.exr`, `deep_tiled_zips_odd.exr`, `htj2k256_half_rgba.exr`, `htj2k32_half_rgba.exr`, `lj2k_half_rgba.exr` (rejected with `UnsupportedCompression`), `multipart_flat_deep.exr`, `uint_rgb_zip.exr`, `zstd_half_rgba.exr`).
  - Validates all 37 files in `openexr-kit/conformance/openexr-testsuite/` (including `comp_dwaa_v1.exr`, `comp_dwaa_v2.exr`, `comp_dwab_v1.exr`, `comp_dwab_v2.exr`, rejecting all malformed bounds/EOF/type-mismatch files, and returning `ExrError::UnsupportedCompression` only on `htj2k`/`lj2k`).

## 4. Remaining Codecs for Wave v3 (`HTJ2K256`, `HTJ2K32`, `LJ2K`)

In accordance with repository policy (zero guessed formulas), the JPEG 2000 family codecs (`HTJ2K256 = 10`, `HTJ2K32 = 11`, `LJ2K = 12`) return `ExrError::UnsupportedCompression` in v2 and require a standalone pure-Rust ITU-T T.814 / ISO/IEC 15444-15 High-Throughput JPEG 2000 block coder (`MEL`, `VLC`, `MagSgn`, 5/3 reversible wavelet, and the OpenEXR `0xFF6F` `omjp` channel-map marker segment from `OpenEXRCore/internal_ht.c` and `openjph`).

## 5. Provenance and Licensing

This module is **not clean-room**. The Rust code is written for this project, but
its container/chunk layout, validation rules, channel-classification rules, DCT
butterfly constants, perceptual-LUT formulas and rounding patches, B44 perceptual
conversion rules, PIZ/canonical-Huffman conventions, Zstd-EXR container layout,
and several constant tables were derived from the OpenEXR 3.5.2 `OpenEXRCore`
sources (`internal_*.c` / `internal_*.h`), which are BSD-3-Clause. The affected
files under `crates/rustmpeg-image/src/exr/` carry
`// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause`. The exact scope, the
copyright notice, and the full license text are in
[`../THIRD_PARTY.md`](../THIRD_PARTY.md) and [`../NOTICE`](../NOTICE).

`deflate.rs` is an independent implementation of IETF RFC 1950 and RFC 1951 and
remains Apache-2.0. OpenEXR 3.5.2 (`libOpenEXRCore`, `exrinfo`) is also used as a
black-box oracle for the conformance results in §3.
