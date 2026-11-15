# H.266 / VVC Implementation Status (`rustmpeg-video::vvc`)

## 0i. Transform blocks larger than 32 samples (2026-11-15)

The general coding-tree decoder now parses and reconstructs transform blocks
larger than 32 samples, which it previously rejected with
`Unsupported("transform block larger than 32 samples")`.

`CoeffScan` already described the grouped scan of a block, but the decoder built
it from the zero-out size (`min(dimension, 32)`), which is only correct while
both dimensions fit in 32 samples. For a 64x64 transform, VVC codes coefficients
only inside the top-left 32x32 region while `last_sig_coeff` and every
`scanPosLast`-derived sub-set index refer to positions in the *full* block
(`blkPos = PosLastX + PosLastY * width`, VTM `CABACReader.cpp:3142-3206`,
`Rom.cpp:355-390`). The scan is therefore built for the full block size, the
coefficient buffer is `width * height` samples, and the sub-block height used by
the LFNST/MTS constraint checks comes from the scan instead of being derived
from the capped coefficient count. The coded region, the sub-set count, the
group grid and `regBinLimit` still use the zero-out area
(`TransformUnit::getTbAreaAfterCoefZeroOut`, `Unit.cpp:894-908`).

Verification (pristine VTM 24.0 `EncoderApp`/`DecoderApp`, `cmp` on Y+U+V):

| Stream | Result |
| :--- | :--- |
| 128x128 10-bit intra, `--Log2MaxTbSize=6`, QP26, filters off, smooth 10-bit gradient source (contains a 64x64 luma transform) | byte-exact; the previous code rejected the stream |
| `d_i8`, `d_i10` (64x64, 3 frames, stock intra cfg) | byte-exact (unchanged) |
| `d_ra8` picture 1 (IDR, 128x128, 8-bit) | byte-exact (unchanged) |

Strict 25-case harness (VTM 24.0 `DecoderApp` + VVdeC 3.3 `vvdecapp`): **50/50**.
Full 266-check matrix: **263 passed / 3 failed / 0 skipped**, the same three
documented failures (`b12_420`, `ctu16`, `d_ra8`). New unit test:
`large_transform_scan_covers_the_top_left_32x32_region_with_the_full_stride`.

Not covered: a transform block with chroma larger than 32 samples (needs 4:2:2
or 4:4:4 with a 64x64 luma transform); only the 4:2:0 geometry is tested, where
a 64x64 luma block maps to a 32x32 chroma block.

**Pre-existing defect found while validating this change:** 128x128 10-bit
streams at QP26 fail with `missing CABAC end_of_slice_one_bit` when the in-loop
filters are enabled, and also when the encoder is limited to
`--Log2MaxTbSize=5` with filters off. The same content at 10-bit, filters off,
default maximum transform size decodes byte-exactly at QP22/26/32, and the
64x64 transform path is byte-exact on its own. The defect is independent of this
change (it reproduces with `--Log2MaxTbSize=5`, where no block is larger than 32)
and is recorded in `VERIFICATION_DEBT.md`; it is not fixed here.

## 0h. Random-access high-level syntax, POC/DPB reordering, and the d_ra8 IDR (2026-11-15)

The general coding-tree decoder now parses the complete random-access high-level
syntax of a VTM 24.0 stream with `sps_max_sub_layers_minus1 = 5`, and the
intra-random-access picture of `d_ra8` reconstructs byte-exactly. The four inter
pictures are still outside the decoder: they are rejected with the typed
`Unsupported("inter slice in the general coding tree")` instead of failing at
the SPS. `d_ra8` therefore remains on the missing list, and its matrix row still
fails.

Implemented in this change:

- **Multi sub-layer SPS.** `profile_tier_level()` is parsed with
  `maxNumSubLayersMinus1` sub-layers: the per-sub-layer `sub_layer_level_present_flag`
  bits, their `ptl_reserved_zero_bit` alignment and the `sub_layer_level_idc`
  values are consumed before `ptl_num_sub_profiles` (§7.3.3). `dpb_parameters()`
  is parsed for every sub-layer, honouring `sps_sublayer_dpb_params_flag`, and
  the single signalled entry is replicated downwards when the flag is 0
  (§7.3.3, VTM `VLCReader.cpp:1337-1362`). `parse_sps_info` and
  `parse_full_sps_rbsp` now share one header parser, so the sub-picture block can
  no longer desynchronise the two views of the SPS.
- **Picture header.** `ph_poc_msb_present_flag`/`ph_poc_msb_cycle_val`, the
  `sps_extra_ph_bit_present_flag` bits, `ph_pic_output_flag`,
  `ph_partition_constraints_override_flag` (with the per-slice-type values),
  `ph_cu_qp_delta_subdiv_intra/inter_slice`,
  `ph_temporal_mvp_enabled_flag`, `ph_collocated_from_l0_flag`/`ph_collocated_ref_idx`,
  `ph_fpel_mmvd_enabled_flag`, `ph_bdof_disabled_flag`, `ph_dmvr_disabled_flag`,
  `ph_prof_disabled_flag`, `ph_qp_delta`, `ph_sao_luma/chroma_enabled_flag`, the
  deblocking override with its six offsets, and `ph_extension_length` are parsed
  and written (§7.3.2.8). Reference picture lists carried in the PH
  (`pps_rpl_info_in_ph_flag = 1`) are parsed with `rpl_sps_flag`/`rpl_idx` or an
  inline `ref_pic_list_struct`, and the picture header writer mirrors all of it.
- **Slice header.** The `sps_extra_sh_bit_present_flag` bits,
  `sh_cu_chroma_qp_offset_enabled_flag`, the slice-header extension and the
  reference list inherited from the picture header are parsed; the collocated
  picture, the `NumRefIdxActive` override and the weighted-prediction presence
  now follow `pps_rpl_info_in_ph_flag` (§7.3.7).
- **POC and DPB.** `PicOrderCntVal` is derived from `ph_pic_order_cnt_lsb`, the
  MSB cycle and `prevTid0POC` exactly as in VTM `VLCReader.cpp:3644-3690`
  (§8.3.1); the DPB is keyed by that POC and sized from
  `dpb_max_dec_pic_buffering_minus1[0]`. Decoded general-tree pictures are held
  in a reorder buffer of `MaxNumReorderPics[Tid]` entries and released in POC
  order, so an access-unit sequence with decode order 0/4/2/1/3 is output as
  0/1/2/3/4 (§C.3.2).
- **Virtual boundaries and weighted prediction** are rejected with typed errors
  when signalled, instead of silently misparsing the remaining header.

Verification (pristine VTM 24.0 build from `vtm24.tar.gz`, `cmp` on Y+U+V):

| Stream | Result |
| :--- | :--- |
| `d_ra8` picture 1 (IDR_N_LP, 128x128, 8-bit, 128-sample CTU, full toolbox, all filters) | byte-exact |
| `d_ra8` pictures 2-5 (B slices) | typed rejection: inter coding units are not implemented |
| `d_i8`, `d_i10` (64x64, 3 frames, stock intra cfg) | byte-exact (unchanged) |

Strict 25-case harness (VTM 24.0 `DecoderApp` + VVdeC 3.3 `vvdecapp`): **50/50**.
Full 266-check matrix: **263 passed / 3 failed / 0 skipped**, the same three
documented failures (`b12_420`, `ctu16`, `d_ra8`). `cargo test --workspace`
passes; the new unit tests cover the multi sub-layer SPS/DPB parse, the PH
override/RPL round trip, the slice-header reference-list inheritance and the
POC output order.

Remaining for `d_ra8`: the general inter coding-unit syntax and its prediction
tools (merge/AMVP, affine, MMVD, SMVD, CIIP, GPM, BDOF, DMVR, SBT), inter motion
compensation with the recorded reconstruction path, inter boundary strength for
deblocking, and the full-size coefficient scan for transform blocks larger than
32 samples. None of that is implemented for the general coding tree; the reduced
`slice::decode_inter_picture` path is a CTU-sized subset decoder and is not a
base for it.

## 0g. 10-bit general coding-tree decode and VTM intra interop (2026-11-15)

The general coding-tree decoder (`decode_general.rs`) now reconstructs 8-bit and
10-bit 4:2:0 pictures, and both VTM-encoder intra rows of the matrix decode
byte-exactly. Streams: VTM 24.0 `EncoderApp` with the stock
`encoder_intra_vtm.cfg` (every coding tool and every in-loop filter enabled),
64x64, 3 frames, QP26, 8-bit (`d_i8`) and 10-bit (`d_i10`), 128-sample CTU on a
64x64 picture. Our output is bit-identical to VTM 24.0 `DecoderApp` over the
full Y+U+V sequence.

Fixes in this change:

- **Bit depth in the general path.** `Gen` now stores `Plane16` planes plus the
  picture bit depth: the unavailable-reference default is `1 << (BitDepth - 1)`,
  samples clip to `ClipBD`, and the angular and MIP predictors take the bit depth
  so their internal clip matches. The 8-bit callers keep `BitDepth == 8`, which
  is sample-identical; new parity tests cover the angular, MIP, SAO, ALF-class
  and LMCS paths.
- **Quantisation.** Dequantisation and the inverse transform receive
  `Qp′ = Qp + QpBdOffset` (`H.266 Eq. 29`) and the picture bit depth. The
  general decoder is handed the *unscaled* `Qp` and applies the offset once;
  `VvcSps::mapped_chroma_qp` returns the scaled `Qp′C`, so the call site
  subtracts `QpBdOffset` first. The previous code added it twice, which broke
  every chroma plane at 10-bit.
- **LMCS.** `LmcsReshaper16` serves both depths. `lmcs_delta_abs_cw` is added to
  `OrgCW` unscaled (`Reshape.cpp:272`), so the previous `<< (BitDepth - 8)`
  scaling was wrong at 10-bit, and chroma residual scaling clips to
  `[-maxVal-1, maxVal]` (`Buffer.cpp:400-410`).
- **SAO.** `sao_offset_abs` is signalled with `Min(BitDepth, 10) - 5` bits, so
  the offset is already expressed in sample units and must not be rescaled by
  `BitDepth - 8` (`SampleAdaptiveOffset.cpp:159-167`). The new
  `apply_sao_region_bounded_16` also reproduces the 8-bit visible-boundary rule
  for edge offsets.
- **ALF and deblocking.** ALF classification uses the `BitDepth + 4` activity
  shift and the 8/10/12-bit clip tables; deblocking uses the bit-depth-aware
  normative filter with the unscaled `Qp` (VTM `QpParam::Qp(0) - qpBdOffset`).
- **Routing and DPB.** 10-bit pictures with a general coding tree reach
  `decode_general` (previously they fell into the CTU-aligned subset decoder and
  were rejected), and the DPB keeps the reconstructed reference planes.

Transform blocks larger than 32 samples are **not** implemented in the general
decoder: VTM codes their coefficients with the full-size grouped scan of
`CoeffCodingContext`, whose `scanPosLast` numbering differs from the capped
32x32 scan used by `CoeffScan`. Such a block now returns a typed
`Unsupported("transform block larger than 32 samples")` instead of a bogus CABAC
error. The matrix's VTM streams do not contain one; VTM 10-bit intra streams at
low QP do, and they are outside the current scope.

Verification (pristine VTM 24.0 build from `vtm24.tar.gz`, `cmp` on Y+U+V):

| Stream | Result |
| :--- | :--- |
| `d_i8` (64x64, 8-bit, 3 frames, stock intra cfg, all filters) | byte-exact |
| `d_i10` (64x64, 10-bit, 3 frames, stock intra cfg, all filters) | byte-exact |
| 10-bit stages LMCS / +SAO / +ALF / +CCALF / +deblock, QP26 | 5/5 byte-exact |
| 10-bit stock cfg, QP26/32/37, 64x64 | 3/3 byte-exact |
| 8-bit stock cfg, QP37, 64x64 | byte-exact |

Strict 25-case harness (VTM 24.0 `DecoderApp` + VVdeC 3.3 `vvdecapp`):
**50/50**. Full 266-check matrix: **263 passed / 3 failed / 0 skipped**. The
three remaining failures are `b12_420` (VVdeC has no 12-bit oracle), `ctu16`
(the 16-sample CTU cannot be signalled and the harness prints `ENCODE FAIL`),
and `d_ra8`.

`d_ra8` (VTM stock `encoder_randomaccess_vtm.cfg`, 128x128, 5 frames, POC order
0/4/2/1/3, temporal ids 0..5) was outside the decoder at this point: the SPS was
rejected at `max_sub_layers_minus1 != 0` before CABAC. Section 0h implements the
multi sub-layer SPS/PH syntax, the RPL-based reference lists, the POC-keyed DPB
and the reorder buffer; the general inter coding-unit syntax remains missing.

## 0f. Deblocking on full-tool intra streams (2026-11-15)

The deblocking filter now derives `maxFilterLengthP`/`maxFilterLengthQ` pair-wise
from the two transform blocks that share an edge (H.266 §8.8.3.3, Eq. 1227 ff.).
The previous code derived each side independently, which admitted unresolved
non-normative edges:

- **Luma.** The rule is "a transform side of at most four samples forces *both*
  lengths to `1`; only otherwise is each side `7` (>= 32 samples) or `3`".
  Independent derivation produced pairs such as `(7, 1)`: the long-tap decision
  was then entered because `maxFilterLengthP > 3`, and a 7/3 long filter was
  applied across a 4-sample transform block where the reference decoders apply
  nothing.
- **Chroma.** The pair is `3/3` only when *both* transform sides span at least
  eight chroma samples, and `1/3` when such an edge also coincides with the upper
  chroma CTB boundary (§8.8.3.3, Eq. 1417-1420). Independent derivation produced
  `(1, 3)` pairs whenever the P side alone was small, which selected the
  asymmetric strong filter instead of the weak filter.

Both defects show up only with the full tool chain: ISP, MRL, MTS, LFNST and SBT
create the 4-sample transform blocks that the reduced-tool configurations never
produce. The first divergent sample of the stock 64x64 QP37 stream is
`(x=12, y=30)`; a VTM deblock trace built from the same source shows the
horizontal edge at `(12, 32)` carrying `maxFilterLengthP = maxFilterLengthQ = 1`
(`sizeQSide = 4`), which the previous derivation turned into a long-tap filter.

Fix: `decode_general.rs` (`luma_lens`, `chroma_lens`, `apply_deblock`) derives the
lengths pair-wise; `filter::deblock_chroma_edge` selects the chroma filter form
from that pair alone (`3/3` strong, `1/3` Eq. 1417-1420, otherwise weak), so the
now-redundant CTB-boundary flag is gone from the chroma edge helpers.

Verified with `cmp` on full Y+U+V, 3 frames, VTM 24.0 `DecoderApp`, stock
`encoder_intra_vtm.cfg` (every tool enabled), 4:2:0 8-bit:

| Stream | 64x64 | 128x128 |
| :--- | :--- | :--- |
| stock, full filter chain, QP22/26/32/37/42 | 5/5 byte-exact | 5/5 byte-exact |
| stock, filters off, QP22/26/32/37/42 | 5/5 byte-exact | 5/5 byte-exact |
| stock, stage chain `base`/`+LMCS`/`+SAO`/`+ALF`/`+CCALF`/`+deblock` at QP26/37 | 12/12 byte-exact | 12/12 byte-exact |

The QP37 64x64 full-chain case that sections 0d and 0e recorded as failing is
byte-exact. Strict 25-case harness (VTM 24.0 `DecoderApp` + VVdeC 3.3 `vvdecapp`):
**50/50**. Full 266-check matrix: **261 passed / 5 failed / 0 skipped**, the same
five documented residual failures (12-bit VVdeC oracle limit, the CTU-16 typed
rejection, and the three VTM-encoder streams outside the decoder subset; section
0g fixes two of those three). No new failure. `cargo test --workspace` passes
(two new deblocking unit tests).

## 0e. LFNST and ISP reconstruction for the stock configuration (2026-11-15)

The general coding-tree decoder now reconstructs the stock `encoder_intra_vtm.cfg`
stream (every tool enabled) byte-exactly. Verified with `cmp` on full Y+U+V, 3
frames, QP26, VTM 24.0 `DecoderApp` output for 64x64 and 128x128 pictures
(`s64.source.yuv` and `ctu128.source.yuv`, 4:2:0 8-bit):

| Stream (VTM encode flags) | 64x64 | 128x128 |
| :--- | :--- | :--- |
| tools off (`--MIP=0 --LFNST=0 --MTS=0 --ISP=0 --MRL=0 --LMChroma=0 --JointCbCr=0 --SBT=0 --BDPCM=0`), filters off | byte-exact | byte-exact |
| `--LFNST=1` only | byte-exact | — |
| `--ISP=1` only | byte-exact | — |
| `--LFNST=1 --ISP=1` | byte-exact | — |
| stock (all tools on), filters off | byte-exact | byte-exact |
| stock (all tools on), full filter chain | byte-exact | byte-exact |

The filters themselves were already exact; the failures were reconstruction
defects in the LFNST and ISP paths at the first combination (`base`, all filters
disabled). Fixes in this change:

- **LFNST set and transpose selection** (`predict.rs` `get_wide_angle`,
  `decode_general.rs` `lfnst_intra_mode`): the §8.4.5.2.7 wide-angle mapping is
  applied before the LFNST mode is renumbered (`+81` for negative wide-angle
  modes, `+14` for `67..80`), using the coding-block size for a luma ISP
  sub-partition (Eq. 320/321). The previous code indexed the LFNST table with the
  raw `0..66` mode, so every wide-angle block (`16x4`, `8x4`, tall `4xN`) used
  the wrong set and the wrong (non-transposed) layout. The prediction
  parameters keep the VTM `getModifiedWideAngle` numbering (`−65` shift); the
  standard's `−67` shift is only used for the LFNST index.
- **LFNST primary-transform zero-out** (`decode_general.rs` `lfnst_skips`):
  `nonZeroW`/`nonZeroH` follow Eq. 1169/1170 (`4` when either TU dimension is
  `4`, else `8`); only the `> 32` case was handled before.
- **ISP implicit MTS** (`decode_general.rs`): `implicitMtsEnabled` for an ISP
  coding unit requires `sps_mts_enabled_flag == 1`. ISP blocks were forced to
  DST-7 even with MTS disabled in the SPS, so `--ISP=1 --MTS=0` streams
  reconstructed with the wrong kernel.
- **ISP reference array lengths** (`predict.rs` `AngularParams::ref_w`/`ref_h`):
  angular prediction for a luma ISP sub-partition uses `refW = nCbW + nTbW` and
  `refH = nCbH + nTbH` (Eq. 315/316). The reference arrays were truncated to
  `2 * nTbW`/`2 * nTbH`, so the §8.4.5.2.15 PDPC reads `mainRef[dX]` and the
  interpolation taps `refMain[deltaInt + x + 3]` ran past the copied samples for
  the wide-angle-mapped modes of a `16x4` coding unit.

Verification after the fix: strict 25-case harness (rustmpeg encode → VTM 24.0
`DecoderApp` + VVdeC 3.3 `vvdecapp`) **50/50**; `cargo test --workspace` passes
(338 VVC tests, 4 new); the full 266-check matrix was not re-run.

Not covered by this change: at QP37 (stock tools, full filter chain) the output
still differed from VTM in 23 of 18432 luma samples of a 64x64 picture, and the
difference was introduced by the deblocking stage. That pre-existing deblocking
defect is fixed in section 0f, which also records the QP sweep that supersedes
this paragraph.

## 0d. In-loop filters for the general decode path (2026-11-15)

The general coding-tree decoder now applies the complete in-loop filter chain:
LMCS (luma inverse mapping and chroma residual scaling), SAO, ALF, CCALF, and
deblocking. Verified byte-exact (`cmp`, full Y+U+V, 3 frames, QP26,
`encoder_intra_vtm.cfg`) against VTM 24.0 `DecoderApp` for 64x64 and 128x128
pictures, using the 3-frame 4:2:0 sources `s64.source.yuv` and
`ctu128.source.yuv`:

| Stage | 64x64 | 128x128 |
| :--- | :--- | :--- |
| base (all filters off) | byte-exact | byte-exact |
| +LMCS | byte-exact | byte-exact |
| +SAO | byte-exact | byte-exact |
| +ALF (no CCALF) | byte-exact | byte-exact |
| +CCALF | byte-exact | byte-exact |
| +deblocking (full chain) | byte-exact | byte-exact |

These streams disable the orthogonal tools that were not yet exact when the
filter stages were verified: `--MIP=0 --LFNST=0 --MTS=0 --ISP=0 --MRL=0
--LMChroma=0 --JointCbCr=0 --SBT=0` (the stock configuration is byte-exact as of
section 0e, at the same QP). The source for both sizes is 4:2:0 8-bit; the exact
commands are in [`VERIFICATION_DEBT.md`](../../VERIFICATION_DEBT.md).

Stock `encoder_intra_vtm.cfg` (all tools on) was **not** byte-exact at the first
combination, `base` with every filter disabled: with only `--LFNST=1` enabled the
stream differed, and with only `--ISP=1` enabled it differed. Those were LFNST and
ISP reconstruction defects that predate this change, not filter stages; they are
fixed in section 0e, which supersedes this paragraph for the stock configuration.

Fixes in this change:

- **LMCS chroma residual scaling** (`filter.rs`, `decode_general.rs`): the chroma
  residual is scaled with symmetric round-half-up around zero and clipped to
  `[-256, 255]` first (`AreaBuf<Pel>::scaleSignal(dir = false)`); the previous
  arithmetic shift rounded negative samples the wrong way. The average reshaped
  luma is derived once per VPDU from the left/top neighbours of the top-left
  luma CU (`Reshape::calculateChromaAdjVpduNei`) and cached as the *average*, not
  as the already-mapped scale. Joint CbCr TUs scale both ICT components whenever
  the TU is joint (`DecCu.cpp:289-292`).
- **PDPC for ISP**: planar/DC prediction applies PDPC for ISP sub-partitions of a
  prediction region of at least `4x4` samples; it was previously skipped for all
  ISP blocks (`IntraPrediction.cpp:390-411`).
- **SAO on the visible picture**: SAO regions are clipped to the visible
  `width x height`, and edge offsets are skipped when a neighbour falls outside
  the visible picture instead of reading the padded allocation
  (`SampleAdaptiveOffset.cpp:308-310`, `:453-515`).
- **ALF**: `clipALF` clips both taps around the current sample, not around each
  other (`AdaptiveLoopFilter.h:72-75`); the inferred ALF virtual boundary at
  `CtbSizeY - 4` (luma) and `CtbSizeY/2 - 2` (chroma) is honoured in both
  classification and filtering, including the `96/64` activity scale and the
  reduced-precision rounding on the two rows adjacent to the boundary
  (`AdaptiveLoopFilter.cpp:884-891`, `:992-1002`, `:1251-1272`); boundary reads
  clamp to the visible picture.
- **CCALF**: per-CTU `ccAlfFilterControlIdc` is parsed with the Cb/Cr context
  groups (`CABACReader.cpp:296-330`), each CTU applies the selected APS filter to
  the ALF-filtered chroma using the pre-ALF luma with the luma virtual-boundary
  row rules, and the `[-128, 127]` residual clip of `filterBlkCcAlf` is applied.
- **Chroma deblocking**: the chroma deblock grid is two chroma samples, each
  decision group spans two lines, chroma beta/tc offsets are taken per component,
  and `pps_chroma_tool_offsets_present_flag == 0` infers `pps_cb/cr_*_offset_div2`
  from the luma offsets (`H.266 §7.4.3.6`). Chroma edges not on the 8-sample
  chroma grid are skipped as in `xEdgeFilterChroma`, and the CTB-row-boundary
  `maxFilterLengthP == 1, maxFilterLengthQ == 3` case uses the asymmetric
  `p0/q0/q1/q2` filter (`isChromaHorCTBBoundary`). Luma deblocking is limited to
  the visible picture.
- **Slice/PPS syntax**: `sh_cb/cr_beta_offset_div2` and `sh_cb/cr_tc_offset_div2`
  are now parsed and carried instead of discarded.

The strict 25-case harness (rustmpeg encode → VTM 24.0 `DecoderApp` + VVdeC 3.3
`vvdecapp`) is **50/50** after these changes. The full 266-check matrix was not
re-run.

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
