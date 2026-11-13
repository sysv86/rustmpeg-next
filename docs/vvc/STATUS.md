# H.266 / VVC Implementation Status (`rustmpeg-video::vvc`)

## 0c. Chroma dual-tree, CCLM and JCCR (2026-11-13)

The general coding-tree decoder (`decode_general.rs`) now closes the chroma half
of the intra pipeline. Verified byte-exact (`cmp`, full Y+U+V, 3 frames, QP26,
VTM 24.0 `EncoderApp` with `encoder_intra_vtm.cfg`, filters disabled:
`--DeblockingFilterDisable=1 --SAO=0 --ALF=0 --CCALF=0 --LMCSEnable=0`) against
VTM 24.0 `DecoderApp`:

| Stream | CTU | CCLM | JCCR | Result |
| :--- | :--- | :--- | :--- | :--- |
| `vtm8n2_64` | 128 (64x64 picture) | off | off | byte-exact |
| `vtm8n2_128` | 128 | off | off | byte-exact |
| `vtm8_64` | 128 (64x64 picture) | on | on | byte-exact |
| `vtm8_128` | 128 | on | on | byte-exact |

The CABAC regular-bin trace matches VTM bin-for-bin for the complete 3-frame
streams (`D_CABAC` on the trace build, `VVC_BINTRACE` on this decoder), so the
match is a reconstruction match, not an agreement of two desynchronised parses.

Fixes in this change:

- **Dual-tree chroma geometry** (`decode_general.rs`): the split series of each
  node is tracked, and `CclmEnabled` is derived per `H.266 §8.4.4` / VTM
  `CodingUnit::checkCCLMAllowed` (64x64-node split pattern, co-located luma split
  at the same depth, ISP exclusion). Chroma QP mapping for `TuCResMode == 2` uses
  `Qp′CbCr` from `ChromaQpTable[2]` plus `pps_joint_cbcr_qp_offset_value` and
  `sh_joint_cbcr_qp_offset` (`H.266 §8.7.2` Eq. 1125/1128/1135).
- **Normative CCLM** (`H.266 §8.4.5.2.14`): `numTopRight`/`numLeftBelow`,
  `numSampT`/`numSampL`, `numIs4N`, `startPosN`, `pickStepN`, `pickPosN`, the
  per-sample template down-sampling of (368)/(369), the CTU-boundary top
  template (371)–(373), the left template (376)/(377), the min/max group
  parameter derivation (378)–(404), and per-sample application (404). Both
  `sps_chroma_vertical_collocated_flag` values are parsed and honoured.
- **JCCR in the general path** (`H.266 §7.4.12.7`, §8.7.4.2, VTM
  `TU::getICTMode`/`inverseTransformICT`): both components of a joint pair are
  reconstructed together; masks 1/2/3 derive the missing residual
  (`cb = cr >> 1`, `cr = cb >> 1`, `cr = ±cb`) with `ph_joint_cbcr_sign_flag`.
- **Chroma PDPC gating**: PDPC applies only when `nTbW >= 4 && nTbH >= 4`; a
  `8x2` chroma block now predicts unpadded DC, matching VTM.
- **LFNST for chroma**: the zero-out input width is 8 only for 4x4/8x8 and 16
  otherwise (`TrQuant::xInvLfnst`), the chroma LFNST set/transpose mode is the
  actual chroma intra mode (co-located luma mode for CCLM), and LFNST is not
  applied to chroma of a single-tree CU.

The strict 25-case harness (rustmpeg encode → VTM 24.0 / VVdeC 3.3 decode) is
**50/50** after these changes; the full 266-check matrix was not re-run.

## 0b. Post-fix re-measurement (late 2026-10-28, after waves 1–4)

Full matrix (`$ORACLE/vvc-conf-full/run_conformance_full.sh`): **261 passed /
5 failed / 0 skipped** (266 checks). Strict harness
(`$ORACLE/vvc-conf/run_conformance.sh`, 25 cases): **50/50** (25 cases on
both VTM 24.0 `DecoderApp` and VVdeC 3.3 `vvdecapp`).

Byte-exact now: 10-bit, 4:0:0/4:2:2/4:4:4, CTU 64/128 (8/10-bit), MRL, BDPCM,
CCLM on CTU-row boundaries, and the shared 16-bit prediction path (Planar, DC,
angular interpolation, reference filtering, PDPC). Fixes landed in `predict.rs`
(shared `predict_angular_16` now carries the full angular kernel; the duplicated
`slice.rs` `predict_cu_16`/`predict_angular_full_16` copies are deleted),
`gather_references_16` availability clamp, planar term order, PDPC, `[1 2 1]`
reference filter, `slice.rs` (QT z-order + root `split_cu_flag`, MRL availability),
and `hls.rs`/`mod.rs` (`sps_bdpcm_enabled_flag`).

Residual failures (all five are documented below; no other case fails):

- **`b12_420` — oracle limitation, not a stream defect.** VVdeC 3.3 rejects every
  12-bit stream. Our stream fails at `sps_bitdepth_minus8 > 2` ("bit depths above
  10 are not supported"); a VTM 24.0-encoded 12-bit stream is also rejected by
  VVdeC 3.3, failing even earlier in `parseConstraintInfo`
  (`gci_reserved_zero_bit`). VTM 24.0 `DecoderApp` decodes our 12-bit stream
  byte-exactly, so 12-bit has no VVdeC oracle.
- **`ctu16` — typed rejection.** The encoder returns a typed `InvalidConfig`
  (no panic) because a 16-sample CTU cannot be signalled in this subset; the
  harness prints `ENCODE FAIL`.
- **`d_i8`, `d_i10`, `d_ra8` — VTM stock streams outside the decoder subset.**
  Our decoder rejects VTM 24.0 `EncoderApp` output with
  `coded picture not aligned to the CTU grid` (`d_i8`, `d_i10`: VTM uses a
  128-sample CTU for a 64x64 picture), `SPS: more than one temporal sub-layer`
  (`d_ra8`), and, once past those checks, `coding tree with a minimum QT below
  32, MTT, or dual-tree intra`. PH-in-slice-header subset streams are
  byte-exact; decoder interop is proven only for that VTM subset configuration.

The previous duplication note is resolved: there is now a single 16-bit
prediction implementation in `predict.rs`.

## 0a. Pre-fix measured matrix (2026-10-28, before waves 2–3)


External full matrix (64 encodes + 3 VTM-encoder cases, VTM 24.0 `DecoderApp`
and VVdeC 3.3 `vvdecapp`, script kept outside the tree at
`$ORACLE/vvc-conf-full/run_conformance_full.sh`): **195 checks passed,
63 failed**. The strict harness (`$ORACLE/vvc-conf/run_conformance.sh`,
25 cases: strict 8-bit 4:2:0, multi-tool intra, MIP, deblock, inter P) is
**25/25 on both decoders**.

Two defects were found and fixed in this session:

- **Chroma deblocking was not normative.** `filter.rs` now implements
  §8.8.3.6.4/§8.8.3.6.9/§8.8.3.6.10 (`deblock_chroma_edge`,
  `deblock_chroma_edge_region`, `chroma_filter_length`): per-group decision with
  `maxFilterLength` `1`/`3` from §8.8.3.5, `dSam0`/`dSam1`, the strong 3/3
  filter, the asymmetric `1/3` filter at chroma CTB row boundaries, the weak
  filter, and the `1/1 && bS != 2` no-filter rule. This fixed the remaining
  inter chroma mismatches (`mt200_inter`, `mt64_inter_tools`).
- **`intra_mip_flag` used context 0 unconditionally.** The context now follows
  Table 47 / VTM `DeriveCtx::CtxMipFlag`: `left_mip + above_mip`, or `3` for an
  aspect ratio above 2:1 (`mip_flag_ctx_inc`, encoder and decoder grids in
  `slice.rs`). This fixed the `mt_i_mip` CABAC desync.

Measured unfixed clusters (do not claim these as conformant):

- every stream with `bit_depth != 8` or `chroma_format != 4:2:0` differs from
  both reference decoders, which agree with each other; in those streams only
  the first CTU matches and the divergence starts at the first sample of the
  second CTU. This covers 10/12-bit, 4:0:0/4:2:2/4:4:4 and ACT.
- 12-bit `b12_420`: VTM decodes it, VVdeC rejects it (`Missing Parameter Set`,
  `Picture Header missing`).
- `mrl` and `bdpcm` 8-bit intra, and CTU 64: both decoders report
  `DecSlice.cpp: "Expecting a terminating bit"`.
- CTU 16: the encoder panics in `hls.rs:376` (`ctu_log2` 4 rejected; the syntax
  cannot signal a 16-sample CTU).
- decoder-side interop (`VTM EncoderApp -> rustmpeg`): the harness path bug
  (`$tag.src.yuv` vs `$tag.src.source.yuv`) is fixed in the external script, but
  our decoder still rejects real VTM streams (`VCL NAL unit without a preceding
  picture header`, `missing CABAC end_of_slice_one_bit`). No VTM stream has been
  decoded successfully yet.
- `--tiles`, `--slices`, `--wpp`, `--rpr` in `vvc_multitool_dump` are parsed and
  ignored, so the corresponding matrix rows only prove the plain stream.

## 0. Build and verification status (2026-10-28, Waves 0–14 / Waves A–G)

- **Waves 0–7 Baseline (Toolchain-Verified Prior to Waves A–G):**
  - Compiled cleanly (`cargo check`, `cargo clippy -D warnings`, `cargo fmt --check`) with **242** passing VVC unit tests and byte-exact VTM 24.0 / VVdeC 3.3 conformance on the strict 8-bit 4:2:0 subset and QP26 multi-tool Intra/Inter matrix.
- **Waves 8–14 (Waves A–G) Additions (Implemented in Offline Container):**
  - **Wave A (Wave 8 — 10-bit, 4:2:2 / 4:4:4 & ACT)**: `Plane16`, `ChromaFormatIdc` (`4:0:0, 4:2:0, 4:2:2, 4:4:4`), `VvcFrame`, 10-bit transform/quantisation (`qp_bd_offset`, `transform_shift_rect_bd`, `forward_transform_rect_bd`, `inverse_transform_rect_bd`, `quantize_rect_bd`, `dequantize_rect_bd`, `quantize_dep_quant_bd`, `dequantize_dep_quant_bd`), 10-bit intra/inter/filter paths (`predict_block_16`, `interpolate_luma_block_16`, `ALF_CLIP_VALS_10BIT`), 4:2:2/4:4:4 CCLM (`downsample_luma_for_cclm_format`, `predict_cclm_with_format`), and 4:4:4 Adaptive Colour Transform (`DELTA_QP_ACT = [-5, 1, 3]`, `apply_act_forward`, `apply_act_inverse`, `encode_act_flag`, `decode_act_flag`).
  - **Wave B (Wave 9 — General High-Level Syntax & Decoder)**: `VvcActiveApsBank` for `PREFIX_APS_NUT` / `SUFFIX_APS_NUT` (`ALF_APS`, `LMCS_APS`, `SCALING_LIST_APS`), automatic `SliceCodingOptions` derivation in `VvcDecoder` from parsed `VvcSps` / `VvcPps` / `VvcSliceHeader`, and fixes for SPS `sps_ccalf_enabled_flag` / `sps_internal_bit_depth_minus_input_bit_depth` and PH `ph_joint_cbcr_sign_flag`.
  - **Wave C (Wave 10 — Partitions & Parallel Structures)**: `VvcTileGrid` (`VvcPps::derive_tile_grid`, `H.266 §6.5.1 Eq. 15–19`), `sh_entry_offset_len_minus1` & `sh_entry_point_offset_minus1` parsing in `parse_full_slice_header`, and full QT/MTT split context derivation & CABAC syntax (`can_split_coding_tree`, `derive_split_contexts`, `encode_coding_tree_split`, `decode_coding_tree_split`).
  - **Wave D (Wave 11 — Screen Content Coding: Palette & IBC)**: Palette mode (`PALETTE_RUN_TOP_LUT`, `PALETTE_RUN_LEFT_LUT`, `derive_palette_run_ctx`, `PaletteBlock`, `palette_traverse_coord`, `build_palette_from_block`, `reconstruct_palette_block`, `encode_palette_cu_syntax`, `decode_palette_cu_syntax`) and Intra Block Copy (`is_ibc_bv_valid`, `derive_ibc_merge_candidates`, `predict_ibc_block`, `search_ibc_block_vector`, `encode_ibc_cu_syntax`, `decode_ibc_cu_syntax`).
  - **Wave E (Wave 12 — RPR & SEI / Film Grain)**: Reference Picture Resampling (`RPR_SCALE_1X = 1 << 14`, `RprScalingRatio`, `derive_rpr_scaling_ratio`, `interpolate_luma_rpr_block`, `interpolate_chroma_rpr_block`) and general SEI / Film Grain Characteristics (`payloadType == 19`, `VvcFilmGrainSei`, `VvcSeiMessage`, `parse_sei_rbsp`, `build_film_grain_sei_rbsp`, `apply_film_grain_plane`).
  - **Wave F (Wave 13 — Multi-Tool & Inter Conformance Fixes)**:
    1. Fixed the root cause of the Inter P-slice `DecSlice.cpp:253: Expecting a terminating bit` failure at `64x64 QP4 POC1`, `64x64 QP10 POC2`, and `200x120 QP40 POC2`: in `H.266 §7.3.11.10` (`CABACWriter.cpp:2410-2413`, `CABACReader.cpp:2760-2763`), when `!CU::isIntra(cu) && trDepth == 0 && !cbf_cb && !cbf_cr`, `tu_y_coded_flag` (`cbf_y`) is **not** coded in the bitstream and is normatively inferred to `1`.
    2. Fixed `encode_component_full` in `slice.rs` to pass `&CoeffScan::new(w, h).raster` (4x4 grouped diagonal scan) to `dct::quantize_dep_quant`.
    3. Inter P-slice deblocking still uses a constant Bs in the multi-tool writer.
       VTM derives Bs per edge (intra/CBF/MV), so inter deblocking is **disabled
       by default** in the multi-tool inter path until the normative Bs derivation
       is implemented; intra deblocking is conformant.
    4. Wired per-CTU `encode_ctu_sao` / `decode_ctu_sao`, `encode_ctu_alf_luma_fixed` / `decode_ctu_alf_luma_fixed`, `encode_mip_mode` / `decode_mip_mode`, and `encode_joint_cbcr_flag` / `decode_joint_cbcr_flag` into `slice.rs` and `hls.rs`.
  - **Wave G (Wave 14 — Encoder Quality)**: `choose_qt_mtt_partition_rdo` and `VvcRateController` (`VvcRateControlMode::Cqp`, `VvcRateControlMode::Vbr`).
- **Post-merge toolchain verification (2026-10-28):** `cargo fmt`, `cargo
  clippy -D warnings`, `cargo test --workspace` (242 VVC tests) and the license
  audit pass in the main repository. VTM 24.0 and VVdeC 3.3 decode byte-exactly:
  the strict subset; multi-tool intra for `modes`, `cclm`, `depquant`, `sdh`,
  `deblock`, `alf`, and their combinations; and inter P (keyint=4) with
  deblocking disabled. **10/12-bit, 4:2:2/4:4:4, ACT, palette, IBC, RPR, SEI,
  tiles/WPP are implemented but have not been externally conformance-tested.**
- **Offline Container Note:** `rustc`/`cargo` are not installed in the Spark v3 container. Static verification (`scripts/check-licenses.py`, delimiter balance, `#![forbid(unsafe_code)]`, `///` doc coverage) was executed and passed; all compiler and external decoder verification commands for Waves A–G are recorded in [`VERIFICATION_DEBT.md`](../../VERIFICATION_DEBT.md).

## 1. Overview
`crates/rustmpeg-video/src/vvc/` (~22,400 lines of `#![forbid(unsafe_code)]` Rust across 13 submodules) implements an ITU-T H.266 (V4, 01/2026) / ISO/IEC 23090-3 Main 10 software encoder and decoder with normative tables and algorithms derived from VTM 24.0 (`BSD-3-Clause`). All formula and table citations are catalogued in [`CITATION_LOG.md`](CITATION_LOG.md).

## 2. Supported Features & Spec / VTM Citations

| Subsystem | Module | Implemented Tools | Spec Clauses | VTM 24.0 Files |
| :--- | :--- | :--- | :--- | :--- |
| **High-Level Syntax & SEI** | `hls.rs` | NAL unit types `0..=31`, `VvcOpi`, `VvcDci`, `VvcVps`, `VvcSps` (8/10-bit, 4:2:0/4:2:2/4:4:4, ACT, Palette, IBC, RPR), `VvcPps`, `VvcTileGrid`, `VvcAps` & `VvcActiveApsBank` (ALF, LMCS, Scaling List), `VvcRefPicListStruct`, `VvcChromaQpTable`, `VvcPictureHeader`, `VvcSliceHeader` (including entry-point offsets), `VvcSeiMessage` & `VvcFilmGrainSei` (`payloadType == 19`) | §6.5.1, §7.3.2, §7.4.3, Annex D.3.10 | `VLCReader.cpp`, `VLCWriter.cpp`, `PictureParameterSet.cpp`, `SEIread.cpp` |
| **CABAC & Residual** | `cabac.rs`, `residual.rs` | 379-context `CTX_INIT_I`/`P`/`B` tables, 86 context sets, truncated binary, unary max bypass, `EGk` bypass, regular & transform-skip residual coding, DepQuant 4-state trellis (`32040`), Sign Data Hiding (SDH), Joint CbCr flag coding | §7.3.11.11, §8.7.3, §9.3 | `Contexts.cpp`, `ContextModelling.cpp`, `BinEncoder.cpp`, `BinDecoder.cpp`, `CABACWriter.cpp`, `CABACReader.cpp`, `DepQuant.cpp` |
| **Intra & Screen-Content Prediction** | `predict.rs` | `Plane`, `Plane16`, `ChromaFormatIdc`, Planar (`0`), DC (`1`), 65 angular modes (`2..=66`, 8/10-bit), wide-angle (`-14..=80`), 4-tap cubic/Gaussian luma & 2-tap chroma interpolation, PDPC, 6-MPM derivation, MRL (`idx 1, 2`), ISP (`Hor`, `Ver`), CCLM (`81..=83` for 4:2:0, 4:2:2, 4:4:4), MIP, BDPCM, Palette Mode (`PaletteBlock`, `PALETTE_RUN_TOP_LUT`, `PALETTE_RUN_LEFT_LUT`) | §8.4.2–§8.4.5 | `IntraPrediction.cpp`, `Rom.cpp`, `MatrixIntraPrediction.cpp`, `MIPData.h`, `UnitTools.cpp` |
| **Inter, IBC & RPR Prediction** | `inter.rs` | `Mv`, `MvField`, `MotionFieldGrid`, `DecodedPictureBuffer` (DPB), `HmvpTable`, Merge, MMVD, AMVP (`AmvrPrecision`), 8-tap luma & 4-tap chroma MC (8/10-bit, 4:2:0/4:2:2/4:4:4), BCW, BDOF, DMVR, 4/6-param Affine + PROF, CIIP, GPM (64 modes), Intra Block Copy (`is_ibc_bv_valid`, `derive_ibc_merge_candidates`, `predict_ibc_block`), Reference Picture Resampling (`RprScalingRatio`, `derive_rpr_scaling_ratio`, `interpolate_luma_rpr_block`, `interpolate_chroma_rpr_block`) | §8.3.2, §8.5.2–§8.5.8, §8.6.2–§8.6.3 | `InterPrediction.cpp`, `InterpolationFilter.cpp`, `UnitTools.cpp`, `Slice.cpp` |
| **Transforms, Quant & ACT** | `dct.rs`, `lfnst.rs`, `transform.rs`, `tr_tables.rs` | 1D/2D `DCT-2` (`2..=64`), `DST-7` & `DCT-8` (`4..=32`), 8/10-bit rectangular `W x H` transforms & quantisation (`qp_bd_offset`, `transform_shift_rect_bd`), scaling lists, DepQuant, LFNST, SBT (`SbtIdx`), Joint CbCr ICT, Adaptive Colour Transform (`DELTA_QP_ACT`, `apply_act_forward`, `apply_act_inverse`), QT/MTT split geometry & context derivation (`CodingTreeSplit`, `can_split_coding_tree`, `derive_split_contexts`) | §6.4.1–§6.4.3, §7.4.9.4, §8.7.2–§8.7.4, §9.3.4.2.2 | `RomTr.cpp`, `TrQuant.cpp`, `Quant.cpp`, `DepQuant.cpp`, `Buffer.cpp`, `UnitPartitioner.cpp`, `ContextModelling.cpp` |
| **In-Loop Filters & Film Grain** | `filter.rs` | Stage 1 LMCS (`LmcsReshaper`), Stage 2 Deblocking (`DBF_TC_TABLE[66]`, `DBF_BETA_TABLE[64]`, intra/inter BS derivation, 4-sample & long-tap luma/chroma filters), Stage 3 SAO (`BandOffset`, `EdgeOffset`, 8/10-bit), Stage 4 ALF & CCALF (`ALF_FIXED_FILTER_COEFF[64][13]`, `ALF_CLIP_VALS_8BIT`, `ALF_CLIP_VALS_10BIT`), Display-side Film Grain synthesis (`apply_film_grain_plane`) | §8.7.5, §8.8.2–§8.8.5, Annex D.3.10 | `Reshape.cpp`, `DeblockingFilter.cpp`, `SampleAdaptiveOffset.cpp`, `AdaptiveLoopFilter.cpp` |
| **Slice, Rate Control & Codec API** | `slice.rs`, `mod.rs` | Strict intra subset (`VvcEncoder`, `decode_annexb_to_i420`), multi-tool Intra & Inter P/B (`encode_picture_with_options`, `encode_inter_picture`, per-CTU `encode_ctu_sao` / `encode_ctu_alf_luma_fixed`, `encode_coding_tree_split`, `encode_palette_cu_syntax`, `encode_ibc_cu_syntax`), `VvcFrame` (`encode_and_reconstruct_vvc_frame`), `VvcRateController` (`Cqp`, `Vbr`), `VvcMultiToolEncoder`, and stateful `VvcDecoder` | §7.3.11, Annex B | `CABACWriter.cpp`, `CABACReader.cpp`, `DecSlice.cpp`, `EncSlice.cpp` |
