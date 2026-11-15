# Verification Debt (`rustmpeg-next` H.266/VVC)

Last updated: 2026-11-15 (random-access high-level syntax + POC/DPB reordering;
`d_ra8` IDR byte-exact, inter coding units still unsupported; matrix 263/3).

## 0f. Random-access headers and the `d_ra8` IDR (2026-11-15)

Executed after adding the multi sub-layer SPS/PTL/DPB syntax, the full picture
header, the slice-header reference-list inheritance, `PicOrderCntVal` derivation
and the POC-ordered output buffer. Oracle: pristine VTM 24.0 build
(`/tmp/opencode/vtm-src`, `DecoderApp`). `rustmpeg` decoder:
`target/release/examples/vvc_stateful_decode`.

```bash
D=/home/nitroarch/vvc-conf-full
BIN=/tmp/opencode/vtm-src/bin/umake/gcc-16.2/x86_64/release
$BIN/DecoderApp -b $D/d_ra8.vvc -o /tmp/opencode/ra8.vtmdec.yuv
python3 -c "d=open('$D/d_ra8.vvc','rb').read(); open('/tmp/opencode/ra8_au1.vvc','wb').write(d[:1270])"
target/release/examples/vvc_stateful_decode /tmp/opencode/ra8_au1.vvc /tmp/opencode/ra8_au1.ours.yuv
head -c 24576 /tmp/opencode/ra8.vtmdec.yuv > /tmp/opencode/ra8_f1.vtm.yuv
cmp /tmp/opencode/ra8_au1.ours.yuv /tmp/opencode/ra8_f1.vtm.yuv   # identical
```

Result:

```text
d_ra8 picture 1 (IDR_N_LP, 128x128, 8-bit, 128 CTU, QP23, all filters)  OK
d_ra8 pictures 2-5 (B slices)  Unsupported("inter slice in the general coding tree")
d_i8 / d_i10 (64x64, 3 frames, stock intra cfg)                          OK (unchanged)
```

The whole 5-picture stream parses its SPS, PPS, APS, embedded picture headers
and slice headers without a desynchronisation; the failure is now the first
inter coding unit, not the SPS. `d_ra8` keeps its matrix row failing.

Other results in this run:

- Strict 25-case harness (`run_conformance.sh`, pristine VTM 24.0 + VVdeC 3.3):
  **50/50**.
- Full 266-check matrix (`run_conformance_full.sh`): **263 passed / 3 failed /
  0 skipped**. Same three failures as the previous run: `b12_420` (VVdeC has no
  12-bit oracle), `ctu16` (`ENCODE FAIL` in the harness), `d_ra8`.
- `cargo test --workspace`: OK. New tests: multi sub-layer SPS/PTL/DPB parse
  (`sps_sub_layer_ptl_and_dpb_parameters_are_parsed`), PH override/RPL/POC-MSB
  round trip, slice-header RPL inheritance from the PH, POC output order.
- `cargo clippy --workspace --all-targets -- -D warnings`: OK.
- `cargo fmt --all -- --check`: OK.
- `python3 scripts/check-licenses.py`: OK.
- Not re-run: 12-bit and 4:2:2/4:4:4 full-tool VTM streams, tiles/WPP/slices/RPR
  VTM streams, GPU paths.

Known gaps introduced or kept by this change (all typed, none silent):

- Inter coding units in the general coding tree (merge/AMVP, affine, MMVD,
  SMVD, CIIP, GPM, BDOF, DMVR, SBT), inter motion compensation and inter
  boundary strength for deblocking.
- Transform blocks larger than 32 samples still return
  `Unsupported("transform block larger than 32 samples")`; the `d_ra8` IDR does
  not contain one, but the full-size grouped scan is still unimplemented.
- Virtual boundaries in the SPS/PH, weighted prediction in the PH/slice header,
  long-term reference pictures, multi-slice/multi-tile pictures, and the
  `ph_cu_chroma_qp_offset_subdiv_*` signalled values are parsed only far enough
  to reject or to ignore them; they are not reconstructed.

## 0e. VTM intra interop at 8- and 10-bit (2026-11-15)

Executed after porting the general coding-tree decoder to 10-bit. Oracle: a
pristine VTM 24.0 build from `~/h266-package/reference/vtm24.tar.gz`
(unpacked to `/tmp/opencode/vtm-src`, `cmake .. -DCMAKE_BUILD_TYPE=Release`,
`make -j12 DecoderApp EncoderApp`). The recorded scratch trees under
`~/h266-package`/`~/h266-package-v2` contain instrumentation prints and, in the
case of `~/h266-package`, no `DecoderApp` binary in
`bin/umake/gcc-16.2/x86_64/release` (`rc=127`), so they are not usable as the
oracle without qualification. The pristine build reproduces every recorded
`.vtm.yuv` reference byte-exactly (`d_i8`, `d_i10`).

`rustmpeg` decoder: `target/release/examples/vvc_stateful_decode`.

```bash
VTM=/tmp/opencode/vtm-src/VVCSoftware_VTM-VTM-24.0
BIN=$VTM/../bin/umake/gcc-16.2/x86_64/release   # built tree
SRC=$DIR/d_i10.src.source.yuv

$BIN/EncoderApp -c $VTM/cfg/encoder_intra_vtm.cfg -i $SRC \
  -wdt 64 -hgt 64 -fr 30 -f 3 --TemporalSubsampleRatio=1 -q 26 \
  --InputBitDepth=10 --InternalBitDepth=10 --InputChromaFormat=420 \
  -b $DIR/d_i10.vvc -o $DIR/d_i10.vtm.yuv
$BIN/DecoderApp -b $DIR/d_i10.vvc -o $DIR/d_i10.dec.yuv
target/release/examples/vvc_stateful_decode $DIR/d_i10.vvc $DIR/d_i10.ours.yuv
cmp $DIR/d_i10.ours.yuv $DIR/d_i10.dec.yuv
```

`--TemporalSubsampleRatio=1` is required for the 3-frame rows: the default ratio
is 8, so `-f 3` alone encodes a single picture and the matrix size check
(`frame_bytes(..., frames=3)`) can never pass. The matrix script
(`~/vvc-conf-full/run_conformance_full.sh`, `decode_vtm`) now passes the flag;
the previous failure text (`rustmpeg decoder failed`) had masked this.

Result (`cmp` on full Y+U+V):

```text
d_i8  64x64  8-bit  3 frames  stock intra cfg, all filters   OK
d_i10 64x64 10-bit  3 frames  stock intra cfg, all filters   OK
10-bit stages QP26: lmcs OK, +sao OK, +alf OK, +ccalf OK, +deblock OK
10-bit stock cfg 64x64: QP26 OK, QP32 OK, QP37 OK
8-bit  stock cfg 64x64: QP37 OK
```

First divergences observed while fixing the 10-bit path (all resolved):

- chroma from the first sample (chroma `Qp′` offset applied twice);
- luma at `(x=0, y=0)` after LMCS (codeword deltas rescaled by `BitDepth - 8`);
- luma at `(x=8, y=0)` with SAO only (SAO offsets rescaled by `BitDepth - 8`).

Not implemented: transform blocks larger than 32 samples. VTM codes them with
the full-size grouped scan of `CoeffCodingContext`; our capped 32x32 `CoeffScan`
desynchronises CABAC, so the decoder now returns
`Unsupported("transform block larger than 32 samples")`. Reproduce with a stock
10-bit intra stream at QP22, which produces such a block.

Other results in this run:

- Strict 25-case harness (`run_conformance.sh`, pristine VTM 24.0 + VVdeC 3.3):
  **50/50**.
- Full 266-check matrix (`run_conformance_full.sh`): **263 passed / 3 failed /
  0 skipped**. Remaining failures: `b12_420` (VVdeC 12-bit limit), `ctu16`
  (typed rejection), `d_ra8` (multi sub-layer random access, not implemented).
- `cargo test --workspace`: OK (new parity tests: SAO region 16 vs 8, ALF class
  16 vs 8, LMCS reshaper 16 vs 8, angular `_bd` vs 8-bit, luma deblock `_bd`).
- `cargo clippy --workspace --all-targets -- -D warnings`: OK.
- `cargo fmt --all -- --check`: OK.
- `python3 scripts/check-licenses.py`: OK.
- Not re-run: 12-bit and 4:2:2/4:4:4 full-tool VTM streams, tiles/WPP/slices/RPR
  VTM streams, GPU paths.

## 0d. Deblocking on full-tool intra streams (2026-11-15)

Executed after fixing the pair-wise `maxFilterLengthP`/`maxFilterLengthQ`
derivation (H.266 §8.8.3.3). Oracles: VTM 24.0 `DecoderApp` and VVdeC 3.3
`vvdecapp`; `rustmpeg` decoder: `target/release/examples/vvc_stateful_decode`.

Sources (4:2:0, 8-bit, 3 frames): `~/vvc-conf/s64.source.yuv` (64x64) and
`~/vvc-conf-full/ctu128.source.yuv` (128x128). Encoder: VTM 24.0 `EncoderApp`
with `encoder_intra_vtm.cfg` (stock, every tool enabled), `-q <QP>
--InternalBitDepth=8 --TemporalSubsampleRatio=1 --FrameRate=30 -f 3`.

```bash
FILTERS_OFF="--DeblockingFilterDisable=1 --SAO=0 --ALF=0 --CCALF=0 --LMCSEnable=0"
FILTERS_ON="--DeblockingFilterDisable=0 --SAO=1 --ALF=1 --CCALF=1 --LMCSEnable=1"
# stages: $FILTERS_OFF plus, in order, --LMCSEnable=1, --SAO=1, --ALF=1, --CCALF=1
```

Result (`cmp` on full Y+U+V, 3 frames):

```text
stock+filters 64  QP22/26/32/37/42   OK  (5/5)
stock+filters 128 QP22/26/32/37/42   OK  (5/5)
stock         64  QP22/26/32/37/42   OK  (5/5, filters off)
stock         128 QP22/26/32/37/42   OK  (5/5, filters off)
stages 64  QP26/37 base/lmcs/sao/alf/ccalf/db   OK (12/12)
stages 128 QP26/37 base/lmcs/sao/alf/ccalf/db   OK (12/12)
```

First divergence of the previously failing case (stock 64x64 QP37, full chain)
was luma `(x=12, y=30)`, introduced by the horizontal edge at `(12, 32)`. A
scratch VTM `DecoderApp` build with an env-guarded print in
`xEdgeFilterLuma` reported `maxFilterLengthP = maxFilterLengthQ = FilterLen::_1`
for that edge (the Q-side transform block is 4 samples high); the previous
per-side derivation produced `(7, 1)` and entered the long-tap path. The oracle
binary was afterwards rebuilt from the unmodified source and re-checked to
produce byte-identical output on the recorded streams.

Other results in this run:

- Strict 25-case harness (`run_conformance.sh`, VTM 24.0 + VVdeC 3.3): **50/50**
  (25 cases on both decoders). Run from a copy with `VTM`/`VVDEC` pointed at the
  recorded builds, since the script's hardcoded paths are stale.
- Full 266-check matrix (`run_conformance_full.sh`): **261 passed / 5 failed /
  0 skipped**, identical to the section 0b baseline; the five failures are the
  documented ones (`b12_420` VVdeC 12-bit limit, `ctu16` typed rejection,
  `d_i8`/`d_i10`/`d_ra8` VTM-encoder streams outside the decoder subset).
- `cargo test --workspace`: OK (two new tests: `luma_lens`, `chroma_lens`).
- `cargo clippy --workspace --all-targets -- -D warnings`: OK.
- `cargo fmt --all -- --check`: OK.
- `python3 scripts/check-licenses.py`: OK.
- Not re-run: 10/12-bit and 4:2:2/4:4:4 full-tool streams, VTM-encoder interop
  (`d_i8`, `d_i10`, `d_ra8`), tiles/WPP/slices/RPR, and GPU paths.

## 0c. LFNST and ISP reconstruction for the stock configuration (2026-11-15)

Executed after fixing the LFNST set/transpose derivation, the LFNST
primary-transform zero-out, the ISP implicit-MTS gate, and the ISP angular
reference lengths. Oracles: VTM 24.0 `DecoderApp` and VVdeC 3.3 `vvdecapp`;
`rustmpeg` decoder: `target/release/examples/vvc_stateful_decode`.

Sources (4:2:0, 8-bit, 3 frames): `~/vvc-conf/s64.source.yuv` (64x64) and
`~/vvc-conf-full/ctu128.source.yuv` (128x128). Encoder: VTM 24.0 `EncoderApp`
with `encoder_intra_vtm.cfg`, `-q 26 --InternalBitDepth=8
--TemporalSubsampleRatio=1 --FrameRate=30 -f 3`.

```bash
TOOLS="--MIP=0 --LFNST=0 --MTS=0 --ISP=0 --MRL=0 --LMChroma=0 --JointCbCr=0 --SBT=0 --BDPCM=0"
FILTERS_OFF="--DeblockingFilterDisable=1 --SAO=0 --ALF=0 --CCALF=0 --LMCSEnable=0"
FILTERS_ON="--DeblockingFilterDisable=0 --SAO=1 --ALF=1 --CCALF=1 --LMCSEnable=1"
```

Result (`cmp` on full Y+U+V, 3 frames):

```text
base/64          OK    base/128        OK
lfnst/64         OK    lfnst+isp/64    OK
isp/64           OK
stock/64         OK    stock/128       OK
stock+filters/64 OK    stock+filters/128 OK
```

`lfnst` = `TOOLS` with `--LFNST=1`, `isp` = `TOOLS` with `--ISP=1`,
`stock` = no tool isolation flags.

Additional sweep (stock tools, both filter settings, same 3-frame sources):
64x64 QP22/QP32/QP37 and 128x128 QP32 are byte-exact with filters off; with the
full filter chain 64x64 QP22, 128x128 QP32 are byte-exact. **64x64 QP37 with the
full filter chain was not**: 23 of 18432 luma samples differed, and the difference
was introduced by deblocking. That defect is fixed and re-verified in section 0d,
which sweeps QP22/26/32/37/42 at both picture sizes with the full chain.

Other results in this run:

- Strict 25-case harness (`run_conformance.sh`, VTM 24.0 + VVdeC 3.3): **50/50**
  (25 cases on both decoders). The script's hardcoded oracle paths were stale
  (`h266-package/reference/.../release/DecoderApp` no longer exists); it was run
  from an unmodified copy with `VTM`/`VVDEC` pointed at the recorded VTM 24.0 /
  VVdeC 3.3 builds.
- `cargo test --workspace`: OK (338 VVC tests, 4 new: `get_wide_angle` numbering,
  `AngularParams::ref_w`/`ref_h`, `lfnst_intra_mode`, `lfnst_skips`).
- `cargo clippy --workspace --all-targets -- -D warnings`: OK.
- `cargo fmt --all -- --check`: OK.
- `python3 scripts/check-licenses.py`: OK.
- Not re-run: the external 266-check matrix, VTM-encoder interop (`d_i8`,
  `d_i10`, `d_ra8`), tiles/WPP/slices/RPR, and GPU paths.
- Not exercised by an oracle stream: implicit intra MTS for a non-ISP TU
  (`sps_mts_enabled_flag == 1 && sps_explicit_mts_intra_enabled_flag == 0`). VTM
  24.0 `EncoderApp` aborts with `xCheckModeSplit: Split decision reusing cannot
  skip quad split` for `--MTSIntraMaxCand=0`, and every configuration used above
  keeps explicit intra MTS enabled, so `implicitMtsEnabled` is only reached
  through ISP. The chroma DCT-2 guard (`cIdx > 0`) and the implicit luma DST-7
  branch follow §8.7.4.1 but are covered by no external oracle; their ISP half is
  verified.

## 0b. General in-loop filter verification (2026-11-15)

Executed after enabling LMCS, SAO, ALF, CCALF, and deblocking in the general
coding-tree decoder. Oracles: VTM 24.0 `DecoderApp` (`-d 8`) and VVdeC 3.3
`vvdecapp`; `rustmpeg` decoder: `target/release/examples/vvc_stateful_decode`.

Sources (4:2:0, 8-bit, 3 frames): `~/vvc-conf/s64.source.yuv` (64x64) and
`~/vvc-conf-full/ctu128.source.yuv` (128x128).

Encoder (`encoder_intra_vtm.cfg`, QP26, `--InternalBitDepth=8`,
`--TemporalSubsampleRatio=1 --FrameRate=30 -f 3`), with the orthogonal tools
disabled so that the pre-filter reconstruction is byte-exact:

```bash
TOOLS="--MIP=0 --LFNST=0 --MTS=0 --ISP=0 --MRL=0 --LMChroma=0 --JointCbCr=0 --SBT=0"
BASE="--DeblockingFilterDisable=1 --SAO=0 --ALF=0 --CCALF=0 --LMCSEnable=0"
# lmcs: $BASE with --LMCSEnable=1
# sao:  $BASE with --SAO=1 --LMCSEnable=1
# alf:  $BASE with --SAO=1 --ALF=1 --LMCSEnable=1
# ccalf:$BASE with --SAO=1 --ALF=1 --CCALF=1 --LMCSEnable=1
# db:   --DeblockingFilterDisable=0 --SAO=1 --ALF=1 --CCALF=1 --LMCSEnable=1
```

Result (`cmp` on full Y+U+V, 3 frames, both picture sizes):

```text
base/64   OK    base/128   OK
lmcs/64   OK    lmcs/128   OK
sao/64    OK    sao/128    OK
alf/64    OK    alf/128    OK
ccalf/64  OK    ccalf/128  OK
db/64     OK    db/128     OK
```

The same twelve combinations were reproduced byte-exactly on a second 3-frame
source (64x64 and 128x128, VTM `EncoderApp`, same flags).

Stock configuration without the tool isolation flags was **not** byte-exact at
that time (first combination `base`, every filter disabled). The failing cases
were streams where only one of the following tools was re-enabled:

- `--LFNST=1`: mismatch in a LFNST-coded transform unit;
- `--ISP=1`: mismatch in an ISP prediction region.

Both were reconstruction defects outside the filter stages (the filters are
disabled in those streams); they are fixed in section 0c, which records the
stock configuration as byte-exact at QP26.

Other results in this run:

- `cargo fmt --all -- --check`: OK.
- `cargo clippy --workspace --all-targets -- -D warnings`: OK.
- `cargo test --workspace`: OK.
- `python3 scripts/check-licenses.py`: OK (all packages permissive).
- Strict 25-case harness (`run_conformance.sh`, VTM 24.0 + VVdeC 3.3):
  **50/50** (25 cases on both decoders), including the inter P cases with
  deblocking after the chroma CTB-boundary filter fix.

## 0a. General intra decoder verification (2026-11-13)

Executed after the chroma dual-tree/CCLM/JCCR change:

- VTM 24.0 `EncoderApp` (`encoder_intra_vtm.cfg`, QP26, 3 frames, filters off:
  `--DeblockingFilterDisable=1 --SAO=0 --ALF=0 --CCALF=0 --LMCSEnable=0`) with
  `LMChroma`/`JointCbCr` off and on, at 64x64 and 128x128 (128 CTU in both
  cases): 4/4 streams byte-exact against VTM 24.0 `DecoderApp`
  (`vtm8n2_64`, `vtm8n2_128`, `vtm8_64`, `vtm8_128`).
- Regular CABAC bins matched VTM bin-for-bin across all three pictures for the
  64x64 and 128x128 streams (VTM trace build `D_CABAC`, decoder `VVC_BINTRACE`).
- Strict 25-case harness (`rustmpeg` encode -> VTM 24.0 `DecoderApp` + VVdeC 3.3
  `vvdecapp`): **50/50**.
- Not re-run: the external 266-check matrix, VTM-encoder interop (`d_i8`,
  `d_i10`, `d_ra8`), tiles/WPP/slices/RPR, and GPU paths. Those rows are not
  claimed by this change.

## 0. Measured Full Matrix (2026-10-28, final after waves 1–4)

Executed outside the offline container with VTM 24.0 `DecoderApp`/`EncoderApp`
and VVdeC 3.3 `vvdecapp`, 266 case-checks: **261 passed, 5 failed, 0 skipped**.
The strict 25-case harness is **50/50** (every case byte-identical on both
decoders).

Every FAIL line from the final run:

```text
FAIL  b12_420: VVdeC decoder rc=255
FAIL  ctu16 [modes,deblock]: ENCODE FAIL
FAIL  d_i8: rustmpeg decoder failed
FAIL  d_i10: rustmpeg decoder failed
FAIL  d_ra8: rustmpeg decoder failed
```

Verified byte-exact against both decoders:

- strict 8-bit 4:2:0 (`s64`, `s128`, `s200`, `s200q40`, `lossless`);
- 8-bit and 10-bit intra tool groups and combinations, including `mip` after the
  `intra_mip_flag` context fix and 10-bit CCLM across CTU-row boundaries;
- 8-bit/10-bit inter P and B;
- the tiles/WPP/slices/RPR rows (the example currently parses and ignores those
  flags, so these rows only re-verify the plain stream);
- the chroma deblocking path after the normative rewrite (§8.8.3.6.4/6.9/6.10);
- the shared 16-bit Planar/DC/angular path after deleting the duplicated
  `slice.rs` kernels.

Residual failures (precise scope — do not cite as conformant):

- **`b12_420`: VVdeC cannot serve as a 12-bit oracle.** VVdeC 3.3 rejects our
  stream at `sps_bitdepth_minus8 > 2` ("bit depths above 10 are not supported").
  A VTM 24.0-encoded 12-bit stream is rejected too (earlier, at
  `parseConstraintInfo`: `gci_reserved_zero_bit`). VTM 24.0 `DecoderApp` decodes
  our stream byte-exactly.
- **`ctu16`: typed rejection.** `--ctu-size 16` returns `InvalidConfig` without
  panicking; the harness records `ENCODE FAIL`.
- **`d_i8` / `d_i10`: stock VTM-stream interop is not yet re-measured.** The
  former `coded picture not aligned to the CTU grid` rejection is gone:
  `validate_decoded_sps` now accepts partial CTUs and the general QT/MTT
  dual-tree decoder (`decode_general.rs`) handles the 128-sample CTU on a 64x64
  picture. `d_i8`/`d_i10` still need a fresh run; they are not claimed passing.
- **`d_ra8`: temporal sub-layers.** Rejected with
  `SPS: more than one temporal sub-layer`; the random-access tree also needs
  inter parsing outside the general intra path.
- **Decoder interop is proven only for a VTM subset configuration.** VTM streams
  carrying the PH in the slice header in that subset decode byte-exactly; stock
  VTM streams outside the verified general-intra configuration do not.

Commands (external, not part of the repo test suite):

```bash
bash $ORACLE/vvc-conf/run_conformance.sh            # 50/50 on both decoders
bash $ORACLE/vvc-conf-full/run_conformance_full.sh  # 261 pass / 5 fail / 0 skip
```


This file records which checks were executed prior to the offline Spark v3 session, what static/license checks were executed inside the offline container, and the exact commands that must be executed in a toolchain-enabled environment (`rustc`/`cargo` + VTM 24.0 + VVdeC 3.3) to close all verification debt. Nothing below is claimed as compiler- or decoder-verified for the Waves A–G code changes unless actually run with a toolchain.

## 1. Executed Checks in Offline Spark v3 Container (Waves A–G)

- `python3 scripts/check-licenses.py` — **PASSED** (all packages permissive, no copyleft; all SPDX headers valid).
- Python static audit across `crates/rustmpeg-video/src/vvc/*.rs` — **PASSED** (balanced `{}`, `[]`, `()`, `#![forbid(unsafe_code)]` preserved, zero `unsafe` blocks, 100% `///` doc-comment coverage on `pub` items).

## 2. Prior Toolchain-Verified Baseline (Before Waves A–G)

- `cargo check --workspace` — **OK**.
- `cargo clippy --workspace --all-targets -- -D warnings` — **OK**.
- `cargo fmt --all -- --check` — **OK**.
- `cargo test --workspace` — **OK** (239 VVC library tests passing prior to Waves A–G; 3 new multi-case unit tests added in Waves A–G).
- Strict subset (`VvcEncoder` / `vvc_dump`) — **PASS** against VTM 24.0 `DecoderApp` and VVdeC 3.3 `vvdecapp` at `64x64 QP26`, `128x64 QP26`, `200x120 QP26`, `200x120 QP40`, and `64x64 QP4`.
- Multi-tool intra (`vvc_multitool_dump`) — **PASS** against VTM 24.0 and VVdeC 3.3 at `64x64 QP26` and `200x120 QP26/QP40` for `modes`, `cclm`, `depquant`, `sdh`, `deblock`.
- Multi-tool inter P (`keyint=4`) — **PASS** against VTM 24.0 and VVdeC 3.3 at `64x64 QP26` and `200x120 QP26`.

## 3. Root-Cause Fixes Applied in Wave F (Wave 13) Pending Toolchain Re-Verification

1. **Inter P-Slice `tu_y_coded_flag` Inference (`slice.rs`)**:
   - Fixed `encode_inter_picture` and `decode_inter_picture` to omit `tu_y_coded_flag` (`cbf_y`) from the CABAC stream and normatively infer `cbf_y = true` when `!CU::isIntra(cu) && trDepth == 0 && !cbf_u && !cbf_v` (`H.266 §7.3.11.10`, PDF p. 96; VTM `CABACWriter.cpp:2410-2413`, `CABACReader.cpp:2760-2763`).
   - This directly addresses the `DecSlice.cpp:253: Expecting a terminating bit` failure on `64x64 QP4 POC1`, `64x64 QP10 POC2`, `200x120 QP40 POC2`, and `VVC_MT_TOOLS=none` inter P-slices.
2. **DepQuant Grouped Scan Order (`slice.rs`)**:
   - Fixed `encode_component_full` (`slice.rs:1641`) to pass `&CoeffScan::new(w, h).raster` instead of `(0..w * h).collect()` to `dct::quantize_dep_quant`.
3. **Inter Deblocking Boundary Strength (`Bs`) & Removal of Forced Restrictions (`slice.rs`, `mod.rs`)**:
   - Added `apply_inter_loop_filters` deriving per-boundary `Bs` (`0` or `1`) from `cbf_y_grid`, `cbf_u_grid`, `cbf_v_grid` (`DeblockingFilter.cpp:790-915`) and removed the forced `dep_quant = true` and `enable_deblock = false` overrides in `VvcMultiToolEncoder::new`.
4. **Per-CTU SAO, ALF, MIP, and JCCR Syntax (`slice.rs`, `hls.rs`, `mod.rs`)**:
   - Wired `encode_ctu_sao` / `decode_ctu_sao`, `encode_ctu_alf_luma_fixed` / `decode_ctu_alf_luma_fixed`, `encode_mip_mode` / `decode_mip_mode`, and `encode_joint_cbcr_flag` / `decode_joint_cbcr_flag`, plus `sps_ccalf_enabled_flag` and `ph_joint_cbcr_sign_flag` in `hls.rs`.

## 4. Required Toolchain Verification Commands (`UNVERIFIED` in Offline Container)

A toolchain-enabled runner must execute the following commands in order:

```bash
# 1. Format, compile, lint, and unit-test workspace
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p rustmpeg-video --lib vvc
cargo test --workspace

# 2. Strict 8-bit 4:2:0 subset conformance (must remain 100% byte-exact)
for case in "64 64 26 s_64_qp26" "128 64 26 s_128x64_qp26" "200 120 26 s_200x120_qp26" "200 120 40 s_200x120_qp40" "64 64 4 s_64_qp4"; do
    set -- $case
    cargo run --release -p rustmpeg-video --example vvc_dump -- "$1" "$2" "$3" "/tmp/$4"
    DecoderApp -b "/tmp/$4.vvc" -o "/tmp/$4.vtm.yuv" && cmp -s "/tmp/$4.recon.yuv" "/tmp/$4.vtm.yuv"
    vvdecapp -b "/tmp/$4.vvc" -o "/tmp/$4.vvdec.yuv" && cmp -s "/tmp/$4.recon.yuv" "/tmp/$4.vvdec.yuv"
done

# 3. Multi-tool intra & inter P conformance (including former failing QP4, QP10, and 200x120 QP40 cases)
for tools in "none" "modes" "cclm" "mip" "depquant" "sdh" "jccr" "deblock" "sao" "alf" "all"; do
    for case in "64 64 26 4 0" "64 64 4 4 0" "64 64 10 4 0" "200 120 26 4 0" "200 120 40 4 0"; do
        set -- $case
        tag="mt_${tools}_${1}x${2}_qp${3}"
        VVC_MT_TOOLS="$tools" cargo run --release -p rustmpeg-video --example vvc_multitool_dump -- "$1" "$2" "$3" "/tmp/$tag" "$4" "$5" 5
        DecoderApp -b "/tmp/$tag.vvc" -o "/tmp/$tag.vtm.yuv" && cmp -s "/tmp/$tag.recon.yuv" "/tmp/$tag.vtm.yuv"
        vvdecapp -b "/tmp/$tag.vvc" -o "/tmp/$tag.vvdec.yuv" && cmp -s "/tmp/$tag.recon.yuv" "/tmp/$tag.vvdec.yuv"
    done
done
```

## 6. Post-merge toolchain verification (2026-10-28)

Executed in `$REPO` after merging
the Waves 8–14 work:

- `cargo fmt --all -- --check` — OK.
- `cargo clippy --workspace --all-targets -- -D warnings` — OK.
- `cargo test --workspace` — OK; **242 VVC tests, 0 failures**.
- `python3 scripts/check-licenses.py` — OK.
- External VTM 24.0 + VVdeC 3.3 harness (8-bit 4:2:0):
  - strict subset (64x64 QP26, 128x64 QP26, 200x120 QP26/QP40, 64x64 QP4): PASS;
  - multi-tool intra (modes, CCLM, DepQuant, SDH, deblock, ALF, combinations;
    64x64 QP26 and 200x120 QP26/QP40): PASS;
  - inter P (keyint=4, 64x64/200x120) with deblocking disabled: PASS.

### Known limitations recorded here

- **Inter deblocking is disabled by default.** VTM derives boundary strength per
  edge from intra/CBF/MV; the multi-tool inter writer still uses a constant Bs,
  which produces small chroma differences. Enabling it requires a normative
  per-edge Bs implementation. Intra deblocking is conformant.
- **10-bit/12-bit, 4:2:2/4:4:4, ACT, palette, IBC, RPR, SEI, tiles/WPP** are
  implemented and unit-tested but were **not** externally conformance-tested in
  this run. Treat them as unverified until VTM/VVdeC checks are added.
- B slices are not emitted by the encoder (P only).
- `gci_no_aps_constraint_flag` is now written as 0 so ALF APS usage is legal;
  the PTL unit-test reference bytes were updated accordingly and VTM accepts the
  resulting SPS.
