# Verification Debt (`rustmpeg-next` H.266/VVC)

Last updated: 2026-11-13 (general intra decoder verification; 266-check matrix last run 2026-10-28).


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
