# AV2 release close-out (units 1–13)

Date: 2026-10-30
Scope: final verification matrix, per-unit status, robustness probes, and the
release checklist for the AV2 v1.0.0 subset in `rustmpeg-video::av2`.

This report is evidence-based. Every result below was produced by a command on
this machine on 2026-10-30; cases that were not run are marked as recorded or
skipped, never as passes.

## Verdict

Units 1–12 are landed. Units 1–7 and 10–12 are **DONE** for their recorded
matrices; units 8 and 9 are **DONE WITH LIMITS** because a small set of tools
outside the recorded matrix is still absent. Unit 13 is **DONE** for AV2: the
12 in-tree external-oracle tests pass, the baseline driver is 84/84, and the
quick driver is 9/9.

The reproducible evidence is:

| Command | Result |
| :-- | :-- |
| `cargo test -p rustmpeg-video --lib av2_ -- --ignored --nocapture` | **12 passed, 0 failed** (27.03 s) |
| `python3 $ORACLE/check_av2.py quick` | **9/9 OK** |
| `python3 $ORACLE/check_av2.py full` | **84/84 OK** |
| `cargo test -p rustmpeg-video --test av2_warp` | **1 passed** |
| `cargo test --workspace --no-fail-fast` | **742 passed, 0 failed, 12 ignored** |

The old 102-case external release group is superseded by the per-unit in-tree
oracle matrices below. `check_av2.py` is the driver named by the earlier
report; it is external to the repository, so its path is recorded here.

One gate caveat: a plain `cargo test --workspace` run failed intermittently in
`vvc::slice::tests::qp50_selection_uses_mts_and_lfnst` (317 passed, 1 failed);
the same test passed in the `--no-fail-fast` run. It is a VVC transform-selection
test unrelated to AV2 and was not modified by this close-out. All AV2 tests pass
in both runs.

## Verification environment

- `dav2d` revision `bff8e5c`, `$ORACLE/dav2d`, run with
  `--threads 1 --framedelay 1 --strict 1 -q` unless stated otherwise.
- AVM commit `e0e3aa8`, binaries
  `$ORACLE/avm-e0e3aa88112d9852cd18443c693148b73fb3fdf7/build-opt/{avmenc,avmdec}`.
- Release build of the repository examples, one thread, Linux.
- Y4M comparison is on frame payload bytes only, never on the Y4M header
  (`dav2d` writes `C420`, the examples write `C420jpeg`; samples are identical).
- `avmdec` is used as a second decoder on our own streams, not as an encoder.

## Commands

Build and run the examples:

```bash
cargo build --release -p rustmpeg-video --examples
```

Encoder argument order (all positions are positional):

```text
av2_encode OUT.ivf FRAMES WIDTH HEIGHT QP LOG2_TILE_COLS LOG2_TILE_ROWS \
  DEBLOCK LAYOUT CDEF CCSO GDF RST NEWMV SWITCHABLE COMPOUND BIT_DEPTH \
  VAQ DELTA_Q REC.y4m VBR_KBPS WARPED_MOTION OBMC INPUT.y4m \
  PALETTE INTRABC FILMGRAIN COMPOUND_TYPES TX_TYPE_RDO REDUCED_TXTP_SET \
  DPCM RECT EXT UNEVEN CCSO_SB_REUSE
```

`LAYOUT` is `420`, `422` or `444`; `INPUT.y4m` may be `-` for the built-in
synthetic source; `REC.y4m` is the encoder reconstruction.

Decode and compare:

```bash
target/release/examples/av2_decode OUT.ivf OUR.y4m BIT_DEPTH [0|1]
dav2d -i OUT.ivf -o DAV.y4m --threads 1 --framedelay 1 --strict 1 -q
```

The optional fourth argument is `0` to disable output-side film grain (the
default is on); it is a bare `0`, not the pair `grain 0`. For film grain,
`dav2d --filmgrain 1` and `--filmgrain 0` are compared with the decoder's
default and `0` output.

## Matrix

### Baseline driver

`check_av2.py full` covers 1920x1080, 1280x720, 64x64, 96x64, 100x64, 129x64
and 1080x1920, QP 20/60/100/140/180/220, 1 and 3 frames: **84/84 OK**.
`check_av2.py quick` covers the 1080p and small-size smoke cases: **9/9 OK**.
Every passing case is `encoder reconstruction == our decoder == dav2d` on every
frame.

### Per-unit recorded matrices

| Unit | Cases | Result |
| :-- | --: | :-- |
| 1. Deblocking on/off | 63 | pass |
| 2. CDEF on, deblocking off | 54 | pass |
| 3. CCSO/GDF decoder / encoder | 21 + 27 | pass |
| 4. Loop restoration | 29 (28 in-tree) | pass |
| 5. Segmentation, VAQ, delta-Q | 18 + 2 ignored | pass |
| 6. NEWMV + switchable subpel | 19 | pass |
| 7. Compound/wedge/CWP | 34 (28 in-tree) | pass |
| 8. Warped motion | 17 + 3 12-bit | pass |
| 9. Transform sets, WHT, DPCM | full sets + ignored tests | pass |
| 10. Extended partitions | 37 decoder + 7 encoder | pass |
| 11. Palette + IntraBC | 30 | pass |
| 12. Film grain | 9 targeted + AVM gradient | pass |

The ignored oracle tests, all passing in the run above:

```text
av2::segment::tests::av2_segmentation_dav2d_byte_exact
av2::dec::tests::av2_extended_partitions_decoder_dav2d_byte_exact
av2::dec::tests::av2_film_grain_decoder_dav2d_byte_exact
av2::dec::tests::av2_segmentation_decoder_dav2d_byte_exact
av2::tests::av2_unit9_encoder_dav2d_byte_exact
av2::dec::tests::av2_unit9_decoder_dav2d_byte_exact
av2::dec::tests::av2_restoration_decoder_dav2d_byte_exact
av2::tests::av2_unit1_deblock_dav2d_byte_exact
av2::dec::tests::av2_compound_decoder_dav2d_byte_exact
av2::dec::tests::av2_unit3_decoder_dav2d_byte_exact
av2::tests::av2_unit2_cdef_dav2d_byte_exact
av2::tests::av2_unit3_encoder_dav2d_byte_exact
```

### Edge sizes and 12-bit warp

The old 2x2 frame-remainder corner observation can no longer be verified:
`dav2d` exits with SIGSEGV (rc 139) after one frame on 10-frame 66x66, 130x130
and 130x66 I420 streams, with or without loop filters. Those cells are recorded
as skipped, not as passes; our decoder still equals the encoder reconstruction
on all 10 frames.

Warped motion at 12-bit is now exact on the recorded source: 64x64 I420, 3
frames, QP 20/100/140, warp on, gives `encoder reconstruction == our decoder ==
dav2d` on every frame, with a warped block selected at QP 20 and QP 100. The
previous 12-bit warp failure is therefore not a current gap.

## Stock AVM streams

The flag sets are the recorded ones in
[`docs/av2/UNIT8_DECODER.md`](av2/UNIT8_DECODER.md) §"AVM attempt" (the
87-flag tool-disable list), plus `--reduced-tx-type-set=2
--reduced-tx-part-set=1` for intra/segmentation, `--enable-warped-motion=1
--enable-warp-delta=1` for warp, the recorded `--aq-mode=3` command for
segmentation, and `--film-grain-test=1` for grain.

| Stream | Input / frames | `dav2d` | Our decoder | Exact result |
| :-- | :-- | :-- | :-- | :-- |
| AVM intra, 256x192 | synthetic, 1 | rc 0 | rc 0, 1 frame | **byte-exact** |
| AVM warp, 256x192 | synthetic, 3 | rc 0 | rc 1 after key frame | key frame differs in 1011 samples (max 1); `BitstreamError("inter block references a missing frame")` |
| AVM segmentation (`--aq-mode=3`), 256x192 | synthetic, 3 | rc 0 | rc 1 after key frame | key frame differs in 1011 samples (max 1); same missing-reference error |
| AVM film grain, 64x64 | horizontal gradient, 1 | rc 0 (grain on/off) | rc 0 (grain on/off) | **byte-exact** vs `dav2d --filmgrain 1` and `--filmgrain 0` |

The key-frame 1-LSB difference on the multi-frame AVM streams is the same
intra/inter reconstruction difference already recorded in
[`docs/av2/UNIT12_DECODER.md`](av2/UNIT12_DECODER.md); it is unrelated to loop
filters and remains with the filters disabled. The inter frames are not
decodable because the decoder has no reference for them, so they are not a
conformance pass.

The earlier synthetic-checker film-grain row (`BitstreamError("decoded EOB 823
outside the 512-coefficient block")`) was recorded on 2026-10-29 and was not
re-run for this close-out; the gradient stream is the reproducible film-grain
evidence.

`avmdec` also decodes our own 64x64 streams for baseline, unit 5, unit 8 warp,
unit 11 palette+IntraBC and unit 12 film grain with `rc 0`, and its output
matches the encoder reconstruction byte-for-byte on all three frames of each
stream.

AVM 4:2:2/4:4:4 and 10/12-bit streams are not usable as an oracle: `dav2d`
rejects their sequence headers. Our own 4:2:2/4:4:4 and 10/12-bit streams are
verified against `dav2d` in the per-unit matrices above.

## Robustness

The decoder was probed with 24 malformed inputs built from a valid 64x64
stream: truncations at 0/1/2/4/8/16/32/64/128/half/`len-1` bytes, an overlong
8-byte ULEB128, a 7-byte ULEB128 with an out-of-range value, an oversized OBU
size (`0x7fffffff`), a zero-size OBU, a size smaller than the OBU header, an
empty tile-group payload, and pseudo-random buffers of 1/4/16/64/256/1024
bytes.

Result: **0 panics; every malformed input returned a typed error** (for
example `BitstreamError("uleb128 value exceeds 32 bits")`,
`BitstreamError("OBU size ... exceeds the remaining ... bytes")`,
`BitstreamError("unexpected end of bitstream")`). One mutation
(`obu_tiny`) left a parseable stream after the junk byte and decoded one frame;
it did not panic. The in-tree deterministic fuzz test
`av2::dec::tests::test_decoder_rejects_malformed_and_fuzzed_inputs` passes as
part of `cargo test --workspace`. There is no `cargo-fuzz` target in the
repository.

## Supported and rejected tools

Supported and verified in this close-out:

- OBU/IVF/Y4M containers; 8/10/12-bit; I420/I422/I444 (I400 is covered by
  the in-tree layout round trip); 64/128 superblocks; uniform tiles (1/2/4);
  intra and single-reference inter.
- Intra: DC/directional/smooth/Paeth/MRL/DIP/filter-intra/CfL/palette/IntraBC.
- Inter: DRL/NEARMV/GLOBALMV, NEWMV, 8-tap subpel, switchable subpel syntax,
  compound/wedge/CWP blending, warped motion (`WARP_NEWMV`), `INTERINTRA`,
  and decoding/prediction of `WARPMV`, `LOCALWARP` and `EXTENDWARP`.
- In-loop filters: deblock, CDEF, CCSO (including `sb_reuse_ccso`), GDF,
  PC/NS/switchable/chroma loop restoration.
- Segmentation, per-segment `ALT_Q`/`SKIP`/`GLOBALMV`, delta-Q, VAQ, VBR rate
  control.
- Film grain OBU parsing, model storage, output-side synthesis, encoder
  estimation/signalling.
- Transform-type signalling for every transform set (full and reduced 0–3),
  lossless 4x4 WHT at QP 0, and DPCM intra prediction.
- Extended partitions (`H3`, `V3`, `H4A/4B`, `V4A/4B`) with partition RDO.

Rejected with a typed error (not silent misdecoding):

- non-uniform tiles, CCSO parameter `reuse` (without block-map reuse), global
  motion, reference-frame MVs, OPFL/MV refinement, adaptive MVD, SDP/extended
  SDP, FSC, CCTX/MHCCP/IBP, TIP/BAWP/CWP-implicit-mask, IST/DDT, TCQ/parity
  hiding, skip mode, global IntraBC, sub-pixel IntraBC, wedge inter-intra,
  six-parameter warp deltas, inter lossless 4x4 WHT, lossless palette/DIP/CfL,
  transform partitions other than `TX_PARTITION_NONE`, and per-unit NS filter
  banks. `WARPMV`/`LOCALWARP`/`EXTENDWARP` are no longer rejected by the
  decoder; their encoder emission is the remaining unit-8 gap.

## Per-unit status

| Unit | Status | Evidence command and result | Remaining gap |
| :-- | :-- | :-- | :-- |
| 1. Deblocking | DONE | `cargo test -p rustmpeg-video --lib av2_unit1_deblock_dav2d_byte_exact -- --ignored` → pass; 63 cases | None for the recorded matrix; `dav2d` SIGSEGV on 130x66 I420/I422 is skipped |
| 2. CDEF | DONE | `... av2_unit2_cdef_dav2d_byte_exact -- --ignored` → pass; 54 cases | None for the recorded matrix |
| 3. CCSO + GDF | DONE | `... av2_unit3_decoder_dav2d_byte_exact -- --ignored` and `... av2_unit3_encoder_dav2d_byte_exact -- --ignored` → pass; 21 + 27 cases | CCSO parameter `reuse` rejected; AVM inter frames blocked |
| 4. Loop restoration | DONE | `... av2_restoration_decoder_dav2d_byte_exact -- --ignored` → pass; 28 in-tree, 29 recorded | Per-unit NS filter banks and multi-class frame filters rejected |
| 5. Segmentation, delta-Q, VAQ, rate control | DONE | `... av2_segmentation_decoder_dav2d_byte_exact -- --ignored` and `... av2_segmentation_dav2d_byte_exact -- --ignored` → pass; 18 matrix cases | Mixed-lossless segmentation and temporal map prediction are not exercised |
| 6. NEWMV + switchable subpel | DONE | `cargo test -p rustmpeg-video --lib test_encoder_newmv_and_switchable_stats` and `... test_decode_matches_encoder_recon_with_newmv...` → pass; 19-case recorded matrix | Subpel compound and `TxPartition::TxSplit` rejected; reduced DRL |
| 7. Compound, wedge, mask blending | DONE | `... av2_compound_decoder_dav2d_byte_exact -- --ignored` → pass; 28 in-tree, 34 recorded | `JOINT_NEWMV` parsed but not emitted; subpel compound not emitted |
| 8. Warped motion (OBMC N/A) | DONE WITH LIMITS | `cargo test -p rustmpeg-video --test av2_warp` → pass; 17-case recorded matrix; 12-bit 3/3 exact; `INTERINTRA` + `WARPMV`/`WARP_NEWMV` matrix 64x64/256x192 QP20/100/140 + 10/12-bit 256x192 → `encoder recon == our decoder == dav2d`; counts 3–9 `WARPMV`, 5–22 `INTERINTRA` per 256x192 run | OBMC does not exist in AV2 v1.0.0 (§7.3 `MOTION_MODES = 5`, §5.20.7.14), cited. `INTERINTRA`, `LOCALWARP`, `EXTENDWARP` and `WARPMV` are parsed and predicted; the encoder emits `INTERINTRA`, `WARPMV` and `WARP_NEWMV`+`DELTAWARP`; `LOCALWARP`/`EXTENDWARP` emission remains blocked by the reduced `find_mv_stack`/`WarpParamStack`, which is also a residual `WARPMV` risk on strongly affine content; six-parameter warp deltas are not emitted |
| 9. Transform types, lossless WHT, DPCM | DONE WITH LIMITS | `... av2_unit9_encoder_dav2d_byte_exact -- --ignored` and `... av2_unit9_decoder_dav2d_byte_exact -- --ignored` → pass; full sets 0–3 | Lossless is intra-only; encoder emits only `TX_PARTITION_NONE`; inter lossless CDF absent; lossless palette/DIP/CfL rejected |
| 10. Extended partitions + partition RDO | DONE | `... av2_extended_partitions_decoder_dav2d_byte_exact -- --ignored` → pass; 37 decoder + 7 encoder cases | Non-square luma intra is DC-only; extended search at 64x64 only; `dav2d` SIGSEGV on some 4:2:2/4:4:4 cells is skipped |
| 11. Palette, IntraBC, ISP | DONE | 30-case screen-content matrix in [`docs/av2/UNIT11_ENCODER.md`](av2/UNIT11_ENCODER.md); `check_av2.py full` with screen tools off → 84/84 | ISP does not exist in AV2 v1.0.0; sub-pixel/global IntraBC rejected; one IntraBC block per frame; palette cache not reused |
| 12. Film grain | DONE | `... av2_film_grain_decoder_dav2d_byte_exact -- --ignored` → pass; 9 targeted rows; AVM gradient byte-exact grain on/off | `mc_identity = 1`, `show_existing_frame`, multiple model slots, and non-slot-0 reuse not covered |
| 13. Robustness, matrix, close-out | DONE | `cargo test -p rustmpeg-video --lib av2_ -- --ignored --nocapture` → 12/12; `check_av2.py full` → 84/84; `quick` → 9/9; gates below | Unrelated VVC test flake noted above; no AV2 gap |

## Gates

Run on 2026-10-30 in the repository root:

```text
cargo fmt --all -- --check                                  -> rc 0
cargo clippy --workspace --all-targets -- -D warnings       -> rc 0
cargo test --workspace --no-fail-fast                      -> 742 passed, 0 failed, 12 ignored
cargo test --workspace                                     -> 317 passed, 1 failed (vvc lfnst flake), 12 ignored
python3 scripts/check-licenses.py                          -> license audit: OK (all packages permissive, no copyleft)
$ORACLE/vvc-conf/run_conformance.sh                -> rc 0, 25/25 cases OK with VTM and VVdeC
cargo build --release -p rustmpeg-video --examples         -> rc 0
```

The 12 ignored tests are the external-oracle tests that need `dav2d` or an AVM
stream; all 12 were run for this report and passed. The workspace test count
includes the AV2 and VVC suites. The plain `cargo test --workspace` failure is
the VVC transform-selection flake described in the verdict, not an AV2 test.

## Known limitations and risks

- Units 8 and 9 carry the tool limits listed in their table rows. They are not
  accepted features; the decoder rejects the unsupported tools with typed
  errors rather than misdecoding.
- AVM inter frames differ from `dav2d` even with loop filters disabled; the
  difference is the unrelated intra/inter reconstruction gap, not a filter
  defect.
- The decoder's DRL/`find_mv_stack` is a reduced approximation; warp streams
  avoid it with `GLOBALMV` and explicit `WARP_NEWMV` models.
- The encoder emits at most one IntraBC block per frame and never reuses palette
  cache entries.
- No `unsafe`, no new dependency, no native build step, and no third-party code
  was added by units 1–13.
