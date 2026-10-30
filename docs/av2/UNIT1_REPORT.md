# Unit 1 — deblocking loop filter (release close-out)

Date: 2026-10-30
Scope: AV2 in-loop deblocking filter, decoder and encoder, for the v1.0.0
subset in `rustmpeg-video::av2`.

## Verdict

**DONE.** The encoder and decoder now apply the same normative deblocking
filter. Our encoder reconstruction is byte-exact against `dav2d` with
deblocking on **and** off across 8/10/12-bit, 4:2:0/4:2:2/4:4:4, QP 20/100/140,
64x64, 256x192 and 130x66 (4:4:4), 3 frames. AVM `avmenc` streams with
deblocking enabled decode byte-exactly against `dav2d` for the 8-bit 4:2:0
configuration that this AVM build can express without unrelated tools.

## Specification

Implementation and comments cite AV2 Specification v1.0.0:

- §5.18.5.2 `deblocking_filter_params()` syntax.
- §6.17.5.2 deblocking filter parameter semantics.
- §7.17.1 general loop order (`plane`, `pass`; vertical before horizontal).
- §7.17.2 edge deblocking filter process (`onScreen`, `sbEdge`, block and
  transform edges, `skip`, `isSubPuEdge`, q/side averaging, filter-size
  clipping at the coded picture extent).
- §7.17.3 filter maximum width.
- §7.17.4 filter size.
- §7.17.5 adaptive filter strength (`qThr`, `side`, `Side_Thresholds`).
- §7.17.6 adaptive filter strength selection (`lvlSeg`, `DfDeltaQ`).
- §7.17.7.1 sample filtering and §7.17.7.2 filter choice.

## Changes

`crates/rustmpeg-video/src/av2/deblock.rs`:

1. **Coded extent.** Deblocking now runs over `plane.coded_width` x
   `plane.coded_height` (`MiCols * 4` x `MiRows * 4`), not the visible
   `width`/`height`. The bottom/right partial superblock is filtered in full.
   This fixed the 130x66 4:4:4 corner mismatch (12 samples, max 6 at 8-bit QP
   140) and is required by §7.17.2. Before the fix the last 4-line segment was
   truncated to 2 lines and the right-edge samples were treated as visible
   width instead of the reconstructed coded width.
2. **Skip semantics.** `applyFilter` uses `Skips[row][col]` of the block on the
   positive side of the edge, not `prev.skip && curr.skip`. This matches
   §7.17.2 and the dav2d mask construction.
3. **Chroma apply flags.** U and V are enabled by
   `apply_deblocking_filter[2]`/`[3]` (`level_u`/`level_v`) for both directions,
   not by the luma pass flags (§7.17.1).
4. **Zero-threshold compatibility.** The implementation no longer returns early
   when `qThr == 0` or `side == 0`; it lets `filter_choice` decide, exactly as
   dav2d does. The clamp then produces a zero delta for `qThr == 0`. This is a
   deliberate oracle-compatibility decision: §7.17.7.2 describes an early
   return, but real 10/12-bit streams at low qindex otherwise diverge from
   dav2d by 1–3 LSB. The comment in `deblock.rs` records the discrepancy.

`crates/rustmpeg-video/src/av2/mod.rs`:

- New non-ignored test `av2_unit1_deblock_changes_recon_and_levels`: every
  bit depth/layout/QP combination, decoder equality, apply flag, QP 20 no-op,
  QP 140 filtered output.
- New ignored external test `av2_unit1_deblock_dav2d_byte_exact`: the full
  63-case matrix with deblocking on and off, compared with `dav2d`.

No change was needed in `av2/cdef.rs` or `av2/dec/*`. `av2/dec/frame.rs` is
untouched in the final diff.

## Filter level search

AV2 does not signal a filter level. `apply_deblocking_filter[0..3]` is a flag;
the strength is derived from the block `qindex` by the decoder
(§7.17.5/§7.17.6). A search over `level_y`/`level_u`/`level_v` would not change
the output, so the encoder uses the qindex-derived apply flags and non-zero
sentinels only. This is documented on `derive_deblock_params`. The real
strength variation is covered by QP 100/140: `qThr` increases from 100 to 140
for 8/10/12-bit, and QP 20 has `qThr == 0` (a normative no-op).

## Verification

All commands were run in `$REPO`
unless stated otherwise.

```text
cargo fmt --all -- --check                                      -> rc 0
cargo clippy --workspace --all-targets -- -D warnings           -> rc 0
cargo test -p rustmpeg-video av2                                -> 156 passed, 0 failed, 11 ignored
cargo test --workspace                                         -> all suites pass, 0 failed
cargo test -p rustmpeg-video av2_unit1_deblock_dav2d_byte_exact -- --ignored
                                                                -> 1 passed
python3 scripts/check-licenses.py                               -> license audit: OK
bash $ORACLE/vvc-conf/run_conformance.sh                -> rc 0, 25/25 VTM OK, 25/25 VVdeC OK
cargo build --release -p rustmpeg-video --examples              -> rc 0
```

Clippy emits only the expected `unused dependency` manifest warnings from
`rustmpeg-image` manifest warnings are pre-existing; there are no code warnings.

### Encoder matrix versus dav2d

Driver: `$ORACLE/unit1/run_case.py`. Each case encodes 3
frames with the release example, decodes with our decoder and with
`dav2d --threads 1 --framedelay 1 --strict 1 -q`, and compares the encoder
reconstruction, our decoder and `dav2d` on every frame.

Matrix: `{64x64, 256x192, 130x66}` x `{I420, I422, I444}` x
`{8, 10, 12}`-bit x `{20, 100, 140}` QP x 3 frames = **81 cases**.

| Result | Cases |
| :-- | --: |
| Encoder recon == our decoder == dav2d | **63** |
| dav2d SIGSEGV (`rc = -11`) | **18** |

The 18 skips are exactly `130x66` with I420/I422 at all three bit depths and
QPs. `dav2d` also crashes on the same streams with deblocking disabled, so this
is an oracle limitation for that odd-size/subsampled combination, not a
deblocking defect. `130x66` I444 is exact for 8/10/12-bit at QP 20/100/140.

The ignored test repeats the same 63 cases with `enable_deblock` on and off and
asserts both reconstructions against `dav2d`. It also asserts that QP 140
changes the reconstruction, that QP 100 sets the apply flag, and that QP 20 is
a no-op (`qThr == 0`). It passed in 8.4 s.

### AVM streams with deblocking enabled

Tool-disable set: the 87-flag list in
[`UNIT8_DECODER.md`](UNIT8_DECODER.md) plus
`--reduced-tx-type-set=2 --reduced-tx-part-set=1`. The only differences for
this unit are `--enable-deblocking=1` and `--enable-ccso=1` (see the quirk
below); `--enable-lf-sub-pu=0` keeps sub-PU edges out of the stream.

Exact command shape (8-bit, 256x192, 1 frame):

```bash
avmenc --ivf --limit=1 --cpu-used=9 --threads=1 \
  --end-usage=q --qp=100 --min-qp=100 --max-qp=100 --aq-mode=0 \
  --enable-qm=0 --qm-min=0 --qm-max=0 --enable-tcq=0 \
  --lag-in-frames=0 --kf-max-dist=1 --tune-content=default \
  --enable-adaptive-mvd=0 --enable-bawp=0 --enable-bru=0 --enable-ccso=1 \
  --enable-cctx=0 --enable-cdef=0 --enable-cdef-on-skip-txfm=0 \
  --enable-chroma-deltaq=0 --enable-cwp=0 --enable-deblocking=1 \
  --enable-diff-wtd-comp=0 --enable-extended-sdp=0 --enable-ext-partitions=0 \
  --enable-fsc=0 --enable-flex-mvres=0 --enable-gdf=0 --enable-global-motion=0 \
  --enable-ibp=0 --enable-idtx-intra=0 --enable-imp-msk-bld=0 \
  --enable-inter-ddt=0 --enable-interinter-wedge=0 --enable-interintra-comp=0 \
  --enable-interintra-wedge=0 --enable-inter-ist=0 --enable-intrabc=0 \
  --enable-intra-dip=0 --enable-ist=0 --enable-joint-mvd=0 --enable-lf-sub-pu=0 \
  --enable-masked-comp=0 --enable-mhccp=0 --enable-mvd-sign-derive=0 \
  --enable-mv-traj=0 --enable-onesided-comp=0 --enable-opfl-refine=0 \
  --enable-palette=0 --enable-parity-hiding=0 --enable-pc-wiener=0 \
  --enable-ref-frame-mvs=0 --enable-refinemv=0 --enable-restoration=0 \
  --enable-sdp=0 --enable-six-param-warp-delta=0 --enable-skip-mode=0 \
  --enable-tip=0 --enable-tip-refinemv=0 --enable-tx-partition=0 \
  --enable-uneven-4way-partitions=0 --enable-warp-causal=0 \
  --enable-warp-delta=0 --enable-warped-motion=0 --enable-warp-extend=0 \
  --enable-wiener-nonsep=0 --enable-trellis-quant=0 --enable-refmvbank=0 \
  --enable-short-refresh-frame-flags=0 --enable-ext-seg=0 --enable-avg-cdf=0 \
  --enable-mfh-obu-signaling=0 --enable-lcr=0 --enable-operating-point-sets=0 \
  --enable-cropping-window=0 --enable-keyframe-filtering=0 --enable-overlay=0 \
  --enable-sframe=0 --enable-fwd-kf=0 --enable-banding-metadata=0 \
  --enable-intra-edge-filter=0 --enable-mrls=0 --enable-cfl-intra=0 \
  --reduced-tx-type-set=2 --reduced-tx-part-set=1 \
  -y -o out.ivf $ORACLE/u13/avm_in_256x192.y4m
```

Input `u13/avm_in_256x192.y4m`, 8-bit 4:2:0, 1 key frame:

| QP | dav2d | Our decoder | deblock on/off differs |
| --: | :-- | :-- | :-- |
| 100 | rc 0 | rc 0, byte-exact | yes |
| 140 | rc 0 | rc 0, byte-exact | yes |
| 200 | rc 0 | rc 0, byte-exact | yes |

**AVM quirk:** in this AVM build `--enable-ccso=0` makes
`--enable-deblocking=1`/`0` produce byte-identical streams (deblocking is not
applied). With `--enable-ccso=1` deblocking is applied, the stream also carries
CCSO, and both filters decode exactly. This is why the AVM proof keeps CCSO on
while CDEF/GDF/restoration stay off.

Other AVM attempts and exact results:

- 10-bit stream made from the same 8-bit input (`-b 10 --input-bit-depth=8`):
  our decoder matches `dav2d` byte-exactly, but this AVM build emits identical
  deblock-on and deblock-off streams, so it does not prove the deblock filter.
- 12-bit AVM stream: `dav2d` fails with `Error parsing sequence header`
  (oracle cannot parse this AVM sequence header).
- 4:2:2 and 4:4:4 AVM streams: `dav2d` also fails with
  `Error parsing sequence header`.
- 2-frame inter AVM stream: our decoder stops after the key frame with
  `BitstreamError("CCSO sb_reuse plane 0 references ref list entry 0 outside 0")`.
  This is an unrelated CCSO/reference gate, not deblocking.
- A generated 10-bit smooth input decodes with rc 0 but differs from `dav2d`
  even with deblocking off; a detailed 10-bit input stops at
  `BitstreamError("decoded EOB 600 outside the 512-coefficient block")`. Both
  are pre-existing 10-bit AVM/coefficient gaps independent of unit 1.

## RD / speed impact

1920x1080, 10 frames, 1 thread, synthetic source, release build, same machine:

| QP | deblock | fps | bytes | avg PSNR Y |
| --: | :-- | --: | --: | --: |
| 100 | off | 10.42 | 472131 | 40.206 |
| 100 | on | 9.99 | 472141 | 40.182 |
| 140 | off | 11.39 | 181687 | 36.366 |
| 140 | on | 10.76 | 178924 | 36.229 |

At QP 140 deblocking saves about 1.5% of the bytes at a 0.14 dB PSNR cost and
runs about 5% slower in this single-thread run. At QP 100 the difference is
within run-to-run noise. The CPU filter is whole-frame/whole-plane; no GPU path
was added.

## Remaining gaps and risks

- `dav2d` crashes on `130x66` I420/I422 streams (deblock on or off); those 18
  matrix cells are recorded as skipped, not as passes.
- AVM 4:2:2/4:4:4 and 12-bit streams cannot be used as an oracle because
  `dav2d` rejects their sequence headers.
- AVM inter streams are blocked by the unrelated CCSO `sb_reuse` reference
  error; only the 1-frame intra AVM case is used as deblocking evidence.
- Sub-PU deblocking (`allow_df_sub_pu`) is not exercised; the AVM proof keeps
  `--enable-lf-sub-pu=0`, and the encoder never signals it.
- Deblocking combined with segmentation/`CurrentQIndex` per block is not
  exercised by this unit. The AVM proof uses `--aq-mode=0`; the unit 5
  segmentation stream keeps deblocking off.
- The zero-threshold behaviour follows dav2d, not the literal §7.17.7.2 early
  return, because the release acceptance is byte-exactness against dav2d. This
  is documented in the source comment.

## Files changed

- `crates/rustmpeg-video/src/av2/deblock.rs`
- `crates/rustmpeg-video/src/av2/mod.rs`
- `docs/av2/UNIT1_REPORT.md`

Commit: `av2: complete deblocking coverage (unit 1)`.
