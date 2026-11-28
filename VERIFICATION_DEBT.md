# Verification Debt (`rustmpeg-next` H.266/VVC)

## Current verification baseline (commit `79f4f0431287e197909d45df975bcc2493bea2c4`)

Measured on this snapshot, from release examples built in this worktree. The
numbered sections below are historical per-wave records; their counts belong to
the commits named in their headings and are not re-asserted for this snapshot.

```bash
cd "$REPO"
cargo build --release --examples
REPO="$REPO" DIR="$WORK/vvc-matrix" \
  VTM_BIN="$VTM_BIN" VTM_ROOT="$WORK/vtm-src" VVDEC="$VVDEC" \
  bash "$WORK/vvc-conf-full/run_conformance_full.sh"
REPO="$REPO" DIR="$WORK/vvc-strict" \
  VTM="$VTM_BIN" VVDEC="$VVDEC" \
  bash "$WORK/vvc-conf/run_conformance.sh"
python3 scripts/check-licenses.py
```

Results: build pass; full matrix **263 passed / 3 failed / 0 skipped** (266
checks); strict harness **50/50** (25 cases on VTM 24.0 `DecoderApp` and VVdeC
3.3 `vvdecapp`, 0 `ENCODE FAIL`); license audit OK (138 packages, 202 SPDX
headers, all permissive).

Failures, verbatim:

```text
FAIL  b12_420: VVdeC decoder rc=255
FAIL  ctu16 [modes,deblock]: ENCODE FAIL
FAIL  d_ra8: rustmpeg decoder failed
```

- `b12_420`: VVdeC 3.3 rejects `sps_bitdepth_minus8 > 2`; VTM 24.0 decodes the
  same stream byte-exactly. Oracle limitation, not a stream defect.
- `ctu16`: typed `unsupported --ctu-size 16`, no panic (16 cannot be signalled).
- `d_ra8`: `unsupported VVC feature: inter coding unit reconstruction (motion
  compensation)`; all five pictures parse bin-for-bin with VTM (§0i below).
- `d_i8 decoder` and `d_i10 decoder` PASS.

Oracle setup (required for reproduction): the hardcoded paths in both external
scripts are stale. The `h266-package` umake release directory has no
`DecoderApp`; its `bin/DecoderAppStatic` exits with SIGSEGV (rc=139) on every
stream. That run was discarded, not cited. The pristine oracle was rebuilt from
`$WORK/h266-package/reference/vtm24.tar.gz` with
`-DCMAKE_BUILD_TYPE=Release -DENABLE_TRACING=OFF
-DCMAKE_CXX_FLAGS="-Wno-unused-but-set-variable -Wno-unused-variable -Wno-unused-parameter"`;
`$VTM_BIN`/`$WORK/vtm-src` point at that build and `$VVDEC` at the recorded VVdeC
3.3 binary. `$WORK` is the scratch root; `$DAV2D` is unused here.

Not re-run for this snapshot: `cargo test --workspace`, clippy, fmt, the
historical QP sweeps and bin traces, and GPU paths. Claims in the sections below
are records of their own commits, not current verification.

Last updated: 2026-11-28 (re-measured at `79f4f04`: matrix 263/3/0, strict
50/50; d_ra8 inter reconstruction still missing).

## 0i. d_ra8 inter syntax bin-for-bin against VTM (2026-11-15)

The four B pictures of `d_ra8` now consume exactly the same CABAC bins as VTM
24.0. Oracle: the same source tree built with tracing
(`cmake -DENABLE_TRACING=ON -DCMAKE_CXX_FLAGS="-Wno-unused-but-set-variable
-Wno-unused-variable -Wno-unused-parameter"`, `make DecoderApp`); the tracing
build was re-checked to decode identically to the pristine one
(`cmp` of the decoded YUV).

Per-picture census of `D_CABAC` (context id + decoded bin) against our
`VVC_BINTRACE` output, aligned by decoding from the start of the stream:

```text
poc 0 (IDR)  8863 bins, 0 mismatches
poc 4        1941 bins, 0 mismatches
poc 2        1279 bins, 0 mismatches
poc 1         704 bins, 0 mismatches
poc 3         751 bins, 0 mismatches
total       13538 bins
```

(An IDR-picture census is included because it doubles as the control: it was
already exact, so a mismatch anywhere would be an inter-path regression.)

Result of the run itself: `vvc_stateful_decode d_ra8.vvc` now reports
`Unsupported("inter coding unit reconstruction (motion compensation)")` instead
of the earlier slice-level rejection, i.e. every syntax element of all five
pictures is parsed and the decoder stops only at the missing reconstruction
stage. `VVC_INTER_SYNTAX_ONLY=1` parses all five pictures to
`end_of_slice` and emits a frame whose pixels are meaningless by construction
(diagnostic hook only).

Root causes fixed in this increment (all derivation-level, no special cases):

1. `CtxStore::new_i_slice` was used for every picture; inter pictures need
   `new_for_slice(slice_type, cabac_init_flag, SliceQpY)`
   (`CABACReader.cpp:75-88`).
2. `limits()` used the intra-slice SPS partition parameters; an inter slice must
   use the inter values for both channels, and the effective values come from
   the picture header so `ph_partition_constraints_override_flag` applies
   (`Slice.h:1140-1185`, `CodingStructure::signalModeCons`).
3. `coding_tree` applied the dual-tree split rules to inter pictures
   (`dual = sps.dual_tree_intra` instead of `&& slice_type == I`), which forced
   an implicit QT split at the 128x128 CTU root and shifted the first
   `split_cu_flag` context from 3 to 6.
4. `joint_cb_cr` was decoded before `tu_y_coded_flag` and only for joint trees;
   VTM decodes it after the luma cbf, for chroma trees as well, and for an inter
   CU only when both chroma components are coded.
5. `tu_y_coded_flag` is inferred (not coded) for an inter TU at depth 0 with no
   chroma residual.
6. Skip CUs code no `sbt_mode()`/transform tree; `cu_sbt_flag` is coded for
   subblock-merge CUs too; `ts_flag` is not coded in an SBT CU; SBT splits the
   transform area with `getSbtTuTiling` and leaves one tile without residual.
7. Geometric partitioning (2 CUs) codes a truncated-binary split direction and
   two equal-probability candidate indices.

Other results recorded at this wave: strict **50/50**, matrix **263/3/0**
(same as the current baseline above); `cargo test --workspace`, clippy and fmt
OK at that time, not re-run for this snapshot; license audit re-run and OK.
`d_i8`, `d_i10` and the `d_ra8` IDR remain byte-exact.

Not implemented (next stage): merge/AMVP candidate derivation, motion
compensation (8-tap luma / 4-tap chroma at the AMVR precision, BCW), BDOF,
DMVR, PROF, CIIP and geometric-partition blending, SBT residual zero-out area,
inter boundary strength for deblocking, per-picture reference POC lists for TMVP
scaling.

## 0h. 10-bit 128x128 CABAC desynchronisation fixed (2026-11-15)

The defect recorded in section 0g is fixed. Two independent parsing bugs both
produced `missing CABAC end_of_slice_one_bit` on 128x128 10-bit streams:

1. **Maximum transform size.** `transform_tree` compared the *chroma* dimensions
   of an area against `MaxTbSize` and tiled the split with a Z-like recursion.
   VTM compares the *luma* area against `MaxTbSize` capped at 64 and divides
   every component by the luma-derived tile counts, in raster order
   (`QTBTPartitioner::canSplit(TU_MAX_TR_SPLIT)`, `UnitPartitioner.cpp:528-544`;
   `PartitionerImpl::getMaxTuTiling`, `UnitPartitioner.cpp:1155-1195`). With
   `MaxTbSize == 32` a 16x32 chroma area (32x64 luma) was therefore not split
   into 16x16 chroma TUs, which shifted every following cbf flag and coefficient
   bin.
2. **SAO offsets.** `decode_ctu_sao` received a hard-coded 8 as the bit depth, so
   `sao_offset_abs` was read with three bits instead of
   `Min(BitDepth, 10) - 5` (`H.266 §7.3.11.3`, VTM `CABACReader.cpp:343-505`).
   The first stream that enables SAO at 10-bit desynchronised at the first CTU.

Reproduction and result (VTM 24.0 `EncoderApp`/`DecoderApp`, `cmp` on Y+U+V,
128x128 10-bit, `$WORK/vvc-conf-full/ctu128_10.source.yuv`):

```bash
OFF="--DeblockingFilterDisable=1 --SAO=0 --ALF=0 --CCALF=0 --LMCSEnable=0"
# stages: $OFF, then $OFF --LMCSEnable=1, --SAO=1, --ALF=1, --CCALF=1,
#         and finally the full chain (no $OFF)
```

```text
10-bit 128x128 QP26 tb32 filters off            OK  (was: end_of_slice)
10-bit 128x128 QP26 stages base/LMCS/SAO/ALF/CCALF/deblock   OK  (5/5, was 1/6)
10-bit 128x128 QP26 full filter chain           OK  (was: end_of_slice)
10-bit 128x128 QP37 full filter chain           OK  (was: end_of_slice)
10-bit 128x128 QP22/26/32 filters off           OK  (unchanged)
d_i8 / d_i10 (64x64)                            OK  (unchanged)
d_ra8 picture 1 (IDR, 128x128 8-bit)            OK  (unchanged)
```

At this wave `cargo test --workspace`, clippy and fmt were recorded as passing
(not re-run for this snapshot); the license audit was re-run and passes.
Harness results recorded at this wave: strict **50/50**, matrix **263/3/0**
(same three rows: `b12_420`, `ctu16`, `d_ra8`).

## 0g. Transform blocks larger than 32 samples (2026-11-15)

Executed after building the coefficient scan for the full transform block size
instead of the zero-out size, allocating `width * height` coefficients, and
deriving the sub-block height used by the LFNST/MTS constraint checks from the
scan. Oracle: pristine VTM 24.0 build (`$WORK/vtm-src`,
`EncoderApp` + `DecoderApp`). `rustmpeg` decoder:
`target/release/examples/vvc_stateful_decode`.

```bash
D=$WORK/tb
BIN=$WORK/vtm-src/bin/umake/gcc-16.2/x86_64/release
python3 - <<'PY'   # smooth 10-bit gradient source favouring a 64x64 luma transform
import os, struct
W=H=128
d=bytearray()
for y in range(H):
    for x in range(W):
        d+=struct.pack('<H', min(1023, 300+(x*3+y*2)//2+((x*y)//200)))
d+=struct.pack('<H',512)*(2*(W//2)*(H//2))
open(os.environ['WORK'] + '/tb/flat10.yuv','wb').write(d)
PY
$BIN/EncoderApp -c $WORK/vtm-src/cfg/encoder_intra_vtm.cfg \
  -i $D/flat10.yuv -wdt 128 -hgt 128 -fr 30 -f 1 --TemporalSubsampleRatio=1 \
  -q 26 --InputBitDepth=10 --InternalBitDepth=10 --InputChromaFormat=420 \
  --Log2MaxTbSize=6 --DeblockingFilterDisable=1 --SAO=0 --ALF=0 --CCALF=0 \
  --LMCSEnable=0 -b $D/flat10.vvc -o $D/flat10.recon.yuv
target/release/examples/vvc_stateful_decode $D/flat10.vvc $D/flat10.ours.yuv
cmp $D/flat10.ours.yuv $D/flat10.recon.yuv     # identical
```

Result:

```text
flat10 (10-bit 128x128, 64x64 luma transform present)   OK  (previous code: typed rejection)
d_i8 / d_i10 (64x64, 3 frames, stock intra cfg)         OK  (unchanged)
d_ra8 picture 1 (IDR, 128x128, 8-bit)                   OK  (unchanged)
```

Other results recorded at this wave:

- Harness results recorded at this wave (see current baseline above): strict
  **50/50**, matrix **263/3/0**; same three failures (`b12_420`, `ctu16`,
  `d_ra8`).
- `cargo test --workspace`: OK (new test
  `large_transform_scan_covers_the_top_left_32x32_region_with_the_full_stride`).
- `cargo clippy --workspace --all-targets -- -D warnings`: OK.
- `cargo fmt --all -- --check`: OK.
- `python3 scripts/check-licenses.py`: OK.

### Newly found here, fixed in section 0h: 10-bit 128x128 CABAC desynchronisation

While building an acceptance case for the large-transform path, the 10-bit
128x128 streams from `$WORK/vvc-conf-full/ctu128_10.source.yuv` exposed a
**pre-existing** defect, unrelated to the transform size:

| Configuration (128x128, 10-bit, QP26) | Result |
| :-- | :-- |
| stock intra cfg, all filters, default `Log2MaxTbSize` | `missing CABAC end_of_slice_one_bit` |
| stock intra cfg, all filters, `--Log2MaxTbSize=5` | `missing CABAC end_of_slice_one_bit` |
| filters off, default `Log2MaxTbSize` | byte-exact at QP22/26/32 |
| filters off, `--Log2MaxTbSize=5` | `missing CABAC end_of_slice_one_bit` |
| filters off, LMCS/+SAO/+ALF/+CCALF/+deblock chain at `--Log2MaxTbSize=5` | all fail identically |

The `--Log2MaxTbSize=5` rows cannot involve a transform block larger than 32, and
they fail at HEAD as well, so the defect is independent of this change. It does
not affect the matrix (its 10-bit rows use 64x64 pictures) or the large-transform
acceptance case, but it blocks 10-bit 128x128 verification with filters on.
Reproduce exactly as above with `--Log2MaxTbSize=5` and no filter flags. Section
0h localises and fixes it (luma-vs-chroma maximum-transform-size split and the
SAO offset bit depth).

Still not covered:

- Transform blocks with a chroma dimension above 32 (needs 4:2:2/4:4:4 with a
  64x64 luma transform); the 4:2:0 geometry is the only tested one.
- Inter coding units in the general coding tree (merge/AMVP, affine, MMVD,
  SMVD, CIIP, GPM, BDOF, DMVR, SBT), inter motion compensation and inter
  boundary strength for deblocking: `d_ra8` pictures 2-5 are still rejected with
  a typed `Unsupported`.
- Virtual boundaries, weighted prediction, long-term reference pictures,
  multi-slice/multi-tile pictures.

## 0f. Random-access headers and the `d_ra8` IDR (2026-11-15)

Executed after adding the multi sub-layer SPS/PTL/DPB syntax, the full picture
header, the slice-header reference-list inheritance, `PicOrderCntVal` derivation
and the POC-ordered output buffer. Oracle: pristine VTM 24.0 build
(`$WORK/vtm-src`, `DecoderApp`). `rustmpeg` decoder:
`target/release/examples/vvc_stateful_decode`.

```bash
D=$WORK/vvc-conf-full
BIN=$WORK/vtm-src/bin/umake/gcc-16.2/x86_64/release
$BIN/DecoderApp -b $D/d_ra8.vvc -o $WORK/ra8.vtmdec.yuv
python3 -c "d=open('$D/d_ra8.vvc','rb').read(); open('$WORK/ra8_au1.vvc','wb').write(d[:1270])"
target/release/examples/vvc_stateful_decode $WORK/ra8_au1.vvc $WORK/ra8_au1.ours.yuv
head -c 24576 $WORK/ra8.vtmdec.yuv > $WORK/ra8_f1.vtm.yuv
cmp $WORK/ra8_au1.ours.yuv $WORK/ra8_f1.vtm.yuv   # identical
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

Other results recorded at this wave:

- Harness results recorded at this wave (see current baseline above): strict
  **50/50**, matrix **263/3/0**; same three failures as the previous run:
  `b12_420` (VVdeC has no 12-bit oracle), `ctu16` (`ENCODE FAIL`), `d_ra8`.
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
pristine VTM 24.0 build from `$WORK/h266-package/reference/vtm24.tar.gz`
(unpacked to `$WORK/vtm-src`, `cmake .. -DCMAKE_BUILD_TYPE=Release`,
`make -j12 DecoderApp EncoderApp`). The recorded scratch trees under
`$WORK/h266-package`/`$WORK/h266-package-v2` contain instrumentation prints and, in the
case of `$WORK/h266-package`, no `DecoderApp` binary in
`bin/umake/gcc-16.2/x86_64/release` (`rc=127`), so they are not usable as the
oracle without qualification. The pristine build reproduces every recorded
`.vtm.yuv` reference byte-exactly (`d_i8`, `d_i10`).

`rustmpeg` decoder: `target/release/examples/vvc_stateful_decode`.

```bash
VTM=$WORK/vtm-src/VVCSoftware_VTM-VTM-24.0
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
(`$WORK/vvc-conf-full/run_conformance_full.sh`, `decode_vtm`) now passes the flag;
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

Other results recorded at this wave:

- Harness results recorded at this wave (see current baseline above): strict
  **50/50**, matrix **263/3/0**; remaining failures: `b12_420` (VVdeC 12-bit
  limit), `ctu16` (typed rejection), `d_ra8` (multi sub-layer random access).
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

Sources (4:2:0, 8-bit, 3 frames): `$WORK/vvc-conf/s64.source.yuv` (64x64) and
`$WORK/vvc-conf-full/ctu128.source.yuv` (128x128). Encoder: VTM 24.0 `EncoderApp`
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

Other results recorded at this wave:

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

Sources (4:2:0, 8-bit, 3 frames): `$WORK/vvc-conf/s64.source.yuv` (64x64) and
`$WORK/vvc-conf-full/ctu128.source.yuv` (128x128). Encoder: VTM 24.0 `EncoderApp`
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

Other results recorded at this wave:

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

Sources (4:2:0, 8-bit, 3 frames): `$WORK/vvc-conf/s64.source.yuv` (64x64) and
`$WORK/vvc-conf-full/ctu128.source.yuv` (128x128).

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

Other results recorded at this wave:

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

## 0. Measured Full Matrix (2026-10-28, final after waves 1–4; historical, superseded by the current baseline above)

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
- **`d_i8` / `d_i10` (at that time): stock VTM-stream interop had not been
  re-measured.** The former `coded picture not aligned to the CTU grid`
  rejection was gone: `validate_decoded_sps` accepts partial CTUs and the
  general QT/MTT dual-tree decoder handles a 128-sample CTU on a 64x64 picture.
  In the current baseline above both rows PASS byte-exactly.
- **`d_ra8`: temporal sub-layers (at that time).** The
  `SPS: more than one temporal sub-layer` rejection is gone; the current failure
  is missing inter CU reconstruction.
- **Decoder interop scope.** VTM streams carrying the PH in the slice header
  decode byte-exactly, and the stock VTM intra rows `d_i8`/`d_i10` now PASS.
  `d_ra8` remains outside the decoder because inter CU reconstruction is missing.

Commands (external, not part of the repo test suite):

```bash
bash $WORK/vvc-conf/run_conformance.sh            # 50/50 on both decoders
bash $WORK/vvc-conf-full/run_conformance_full.sh  # 261 pass / 5 fail / 0 skip
```


## Historical offline-container and post-merge records (superseded)

These are per-commit records kept for provenance. The baseline at the top of
this file supersedes their status claims; an "unverified" or "not conformance
tested" label below describes that commit, not the current snapshot.

- Offline Spark v3 container (Waves A–G): `python3 scripts/check-licenses.py`
  passed; a Python static audit of `crates/rustmpeg-video/src/vvc/*.rs` passed
  (balanced delimiters, `#![forbid(unsafe_code)]` preserved, no `unsafe` blocks,
  `///` coverage on public items). No compiler or decoder was available there,
  so no compiler/oracle claim was made for those changes in that container.
- Prior toolchain baseline (before Waves A–G), recorded as OK: `cargo check
  --workspace`, clippy, fmt, `cargo test --workspace` (239 VVC tests), the strict
  subset, multi-tool intra and inter P against VTM 24.0 / VVdeC 3.3.
- Wave F fixes (recorded then, not re-run for this snapshot): inter
  `tu_y_coded_flag` inference (H.266 §7.3.11.10), the DepQuant grouped scan
  (`CoeffScan::new(w, h).raster`), per-edge inter boundary strength, and per-CTU
  SAO/ALF/MIP/JCCR syntax.
- Post-merge 2026-10-28 run in `$REPO`, recorded as OK: fmt, clippy,
  `cargo test --workspace` (242 VVC tests), the license audit, strict subset and
  the recorded multi-tool intra/inter checks. Its caveat that 10/12-bit,
  4:2:2/4:4:4, ACT, palette, IBC, RPR, SEI and tiles/WPP had not been externally
  tested describes that run only; the current matrix covers those rows, with the
  three failures listed at the top.

The toolchain command list that was pending in the offline container is kept as
a reproduction recipe with env placeholders. It was not executed in full for the
current snapshot: only the build, matrix, strict harness and license audit from
the current baseline were executed.

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p rustmpeg-video --lib vvc
cargo test --workspace

for case in "64 64 26 s_64_qp26" "128 64 26 s_128x64_qp26" \
            "200 120 26 s_200x120_qp26" "200 120 40 s_200x120_qp40" \
            "64 64 4 s_64_qp4"; do
    set -- $case
    cargo run --release -p rustmpeg-video --example vvc_dump -- "$1" "$2" "$3" "$WORK/$4"
    "$VTM_BIN/DecoderApp" -b "$WORK/$4.vvc" -o "$WORK/$4.vtm.yuv" && cmp -s "$WORK/$4.recon.yuv" "$WORK/$4.vtm.yuv"
    "$VVDEC" -b "$WORK/$4.vvc" -o "$WORK/$4.vvdec.yuv" && cmp -s "$WORK/$4.recon.yuv" "$WORK/$4.vvdec.yuv"
done

for tools in "none" "modes" "cclm" "mip" "depquant" "sdh" "jccr" "deblock" "sao" "alf" "all"; do
    for case in "64 64 26 4 0" "64 64 4 4 0" "64 64 10 4 0" "200 120 26 4 0" "200 120 40 4 0"; do
        set -- $case
        tag="mt_${tools}_${1}x${2}_qp${3}"
        VVC_MT_TOOLS="$tools" cargo run --release -p rustmpeg-video --example vvc_multitool_dump -- "$1" "$2" "$3" "$WORK/$tag" "$4" "$5" 5
        "$VTM_BIN/DecoderApp" -b "$WORK/$tag.vvc" -o "$WORK/$tag.vtm.yuv" && cmp -s "$WORK/$tag.recon.yuv" "$WORK/$tag.vtm.yuv"
        "$VVDEC" -b "$WORK/$tag.vvc" -o "$WORK/$tag.vvdec.yuv" && cmp -s "$WORK/$tag.recon.yuv" "$WORK/$tag.vvdec.yuv"
    done
done
```
