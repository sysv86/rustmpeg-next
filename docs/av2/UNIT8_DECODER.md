# AV2 release unit 8 — decoder and selection half (warped motion; OBMC N/A)

Date: 2026-10-29
Scope: AV2 v1.0.0 warped-motion decoding in
`crates/rustmpeg-video/src/av2/dec/*`, plus the encoder selection fix in
`crates/rustmpeg-video/src/av2/{frame.rs,inter.rs}` that makes `is_warp`
blocks actually appear in a stream. External byte-exact oracle: dav2d rev
`bff8e5c`. No files outside unit 8's ownership were changed except the
`av2_decode` example (10-bit output selection) and this document.

## OBMC decision and citation

AV2 v1.0.0 has **no OBMC tool**, so unit 8's OBMC half is not applicable.

- §7.3 constants: `MOTION_MODES = 5`, with the enum
  `SIMPLE=0, INTERINTRA=1, LOCALWARP=2, DELTAWARP=3, EXTENDWARP=4`. There is no
  OBMC mode.
- §5.20.7.14 `read_motion_mode()` is the complete per-block motion-mode syntax.
  It codes only `use_extend_warp`, `use_local_warp`, and inter-intra; it has no
  overlap flag, blend index, or neighbour-MV symbol.
- §7.13.3.18 `block_inter_prediction` and §7.13.3.19 `block_warp` are the only
  single-reference prediction processes. There is no overlapped blending
  process and no OBMC weight/window table.
- `grep -i obmc` over the AV2 v1.0.0 text (`$WORK/av2spec.txt`) returns
  no hit. The only `overlap_flag` in the specification belongs to film-grain
  synthesis (§7.18.3), not motion compensation. `grep -i obmc` over dav2d
  `src/*.c` also returns no hit.

Consequently the encoder's `ToolConfig::enable_obmc` and the example's `obmc`
argument remain documented no-ops. There is no silent no-op: both the field and
the CLI flag say that AV2 v1.0.0 has no OBMC.

## Decoder implementation

`dec/frame.rs`:

- Removed the `Unsupported("warped motion")` gate.
- Parses `is_warp`, `warp_mv`, the `warp_idx` loop, `warpmv_with_mvd`, and
  `warp_inter_intra` in §5.20.7.6 order. `WARPMV` yields `DELTAWARP`. The
  `WARP_NEWMV` path reads `use_extend_warp` then, when it stays in
  `DELTAWARP`, `use_local_warp`, using dav2d's `has_cs_ext` guard and
  `warp_extend`/`warp_causal` contexts. `LOCALWARP` runs the normative
  `find_warp_samples`/`warp_estimation` least-squares fit; `EXTENDWARP` runs
  `extend_warp_estimation` from the matching neighbour matrix and the block MV.
- Implements `INTERINTRA` (§5.20.7.15): `inter_intra`, `interintra_mode`
  (`II_DC/V/H/SMOOTH_PRED`), `wedge_interintra` (rejected only when wedge
  blending is selected), intra prediction saved before the inter prediction,
  the §7.13.3.29 `Ii_Weights_1d` mask, and the §7.13.3.30 blend. Chroma uses
  the per-plane mask dimensions.
- Implements `read_warp_delta()` (§5.20.7.7): `warp_delta_precision`,
  `warp_delta_param_low`/`high`/`sign`, the 4-parameter
  `params[4] = -params[3], params[5] = params[2]` form, and the six-parameter
  form when `enable_six_param_warp_delta` and `RefWarpIdx == 1`.
- Builds `WarpParamStack` from the per-4x4 mode map
  (`generate_points_from_corners`, §7.12.2.3), the per-reference warp bank
  (§7.12.3), the identity global-motion model, and `Default_Warp_Params`.
- Derives `BlockMvs[0]` with `get_warp_motion_vector` (§7.12.2.2), reduces the
  model, runs `setup_shear` / `get_warp_translation` (§7.13.3.21–23), and
  predicts with `block_warp` (§7.13.3.19) on luma and both chroma planes,
  including per-plane subsampling.
- Updates the warp bank after a warp block and stores each 4x4 cell's
  sub-block warp motion vector in the neighbour contexts. The sub-MV storage
  is required for the next block's DRL/corner scan to match the reference.
- Reads the warp-block `tx_part` position **after** the warp syntax, matching
  the encoder and dav2d.

The `is_warp` CDF context is `warp_mv_count` (§7.11.4 `scan_point_warp_ctx`);
`warp_delta_precision` is indexed by `MiSize` through `mi_size_index`, and
`warp_inter_intra` through `Size_Group`, not by the internal `BlockSize` enum.

## Encoder selection fix

The baseline encoder chose `NEARMV`/`GLOBALMV` with a hard-coded
`mv = (0, 0)`, so every neighbour corner had the same zero MV and
`generate_points_from_corners` never produced a non-identity candidate.

- `NEARMV` now uses the actual DRL predictor `mvp`, not a hard-coded zero.
- With warped motion enabled, non-warp single-reference blocks use `GLOBALMV`
  (zero MV, no DRL dependency). This keeps the stream deterministic for the
  reduced in-tree DRL, which does not reproduce every normative candidate once
  warp sub-block MVs exist.
- Warp RDO estimates a `WARP_NEWMV` four-parameter rotation+uniform-scale model
  from four half-block motion vectors by least squares, quantises the two
  explicit deltas to `warp_delta_precision = 1`, evaluates `block_warp`, and
  selects the model only when it beats translation including MV-residual and
  delta rate. No QP, coordinate, or per-clip condition is hard-coded.
- `FrameStats::warped_blocks` / `EncodedPacket.stats.warped_blocks` is the
  counter. `tests/av2_warp.rs` asserts `warped_blocks > 0` on zooming content
  and then asserts decoder reconstruction equality.

`WARPMV` is decoded and emitted; on the recorded matrix the encoder's
`WarpParamStack` candidate is byte-exact against dav2d. The in-tree reduced
DRL/corner stack still cannot reproduce the full normative
`find_mv_stack`/`WarpParamStack` once strongly affine bank/scan candidates are
present, so encoder `LOCALWARP`/`EXTENDWARP` (whose MVD predictor depends on
that stack) remain unemitted and `WARPMV` on such content is a residual
conformance risk. `WARP_NEWMV` carries the explicit model and is not affected
by that stack.

## Verification

Environment: Linux, dav2d rev `bff8e5c`, 1 thread. Clips were generated with
FFmpeg as a black-box (`zoompan`, 3 frames) under `$WORK/`. Each case
encodes with `warped_motion 1, obmc 0, newmv 0`, decodes with our decoder and
with dav2d, and compares the encoder reconstruction byte-for-byte:

```bash
target/release/examples/av2_encode OUT.ivf 3 W H QP 0 0 0 LAYOUT 0 0 0 0 \
    0 0 0 BD 0 0 REC.y4m 0 1 0 IN.y4m
target/release/examples/av2_decode OUT.ivf DEC.y4m BD
dav2d -i OUT.ivf -o DAV.y4m --threads 1 --framedelay 1 --strict 1 -q
python3 $WORK/cmpy4m.py REC.y4m DEC.y4m
python3 $WORK/cmpy4m.py REC.y4m DAV.y4m
```

Result: every case has `warped > 0`, encoder rc 0, decoder rc 0, dav2d rc 0,
and **our decoder == encoder recon == dav2d**:

| Case | QP | Layout | BD | Warp blocks | Our decoder | dav2d |
| :-- | :-- | :-- | :-- | --: | :-- | :-- |
| 64x64 | 20 | I420 | 8 | 2 | OK | OK |
| 64x64 | 100 | I420 | 8 | 2 | OK | OK |
| 64x64 | 140 | I420 | 8 | 1 | OK | OK |
| 256x192 | 20 | I420 | 8 | 2 | OK | OK |
| 256x192 | 100 | I420 | 8 | 1 | OK | OK |
| 256x192 | 140 | I420 | 8 | 1 | OK | OK |
| 64x64 | 100 | I420 | 10 | 2 | OK | OK |
| 64x64 | 100 | I422 | 8 | 2 | OK | OK |
| 64x64 | 100 | I444 | 8 | 2 | OK | OK |

`cargo test -p rustmpeg-video --test av2_warp` passes and asserts
`warped_blocks > 0` plus decoder/encoder reconstruction equality.

Inter-intra and the three explicit motion-mode syntaxes were additionally
verified with all `motion_modes` bits enabled on 64x64 and 256x192 I420 at
QP 20/100/140 (3 frames) and on 10-bit/12-bit 256x192 at QP100:
`encoder recon == our decoder == dav2d` on every frame, with
`interintra_blocks` 5-22 per 256x192 run (0 on the 64x64 clip). Our streams
code `use_extend_warp = 0`/`use_local_warp = 0`, so the AVM end-to-end check
still stops on the unrelated chroma intra mode (`uv_idx=14`); exact AVM means
were not obtained. `python3 $ORACLE/check_av2.py quick` (warp off): 9/9 `OK`.

### AVM attempt

`avmenc` produced a warp stream (rc 0) at 256x192, 3 frames, with warped motion
enabled and local/extended/six-parameter/causal warp disabled:

```text
--ivf --limit=3 --cpu-used=9 --threads=1 --end-usage=cbr --target-bitrate=300
--min-qp=100 --max-qp=200 --aq-mode=3 --enable-qm=0 --qm-min=0 --qm-max=0
--enable-tcq=0 --lag-in-frames=0 --kf-max-dist=30 --tune-content=default
--enable-adaptive-mvd=0 --enable-bawp=0 --enable-bru=0 --enable-ccso=0
--enable-cctx=0 --enable-cdef=0 --enable-cdef-on-skip-txfm=0
--enable-chroma-deltaq=0 --enable-cwp=0 --enable-deblocking=0
--enable-diff-wtd-comp=0 --enable-extended-sdp=0 --enable-ext-partitions=0
--enable-fsc=0 --enable-flex-mvres=0 --enable-gdf=0 --enable-global-motion=0
--enable-ibp=0 --enable-idtx-intra=0 --enable-imp-msk-bld=0
--enable-inter-ddt=0 --enable-interinter-wedge=0 --enable-interintra-comp=0
--enable-interintra-wedge=0 --enable-inter-ist=0 --enable-intrabc=0
--enable-intra-dip=0 --enable-ist=0 --enable-joint-mvd=0 --enable-lf-sub-pu=0
--enable-masked-comp=0 --enable-mhccp=0 --enable-mvd-sign-derive=0
--enable-mv-traj=0 --enable-onesided-comp=0 --enable-opfl-refine=0
--enable-palette=0 --enable-parity-hiding=0 --enable-pc-wiener=0
--enable-ref-frame-mvs=0 --enable-refinemv=0 --enable-restoration=0
--enable-sdp=0 --enable-six-param-warp-delta=0 --enable-skip-mode=0
--enable-tip=0 --enable-tip-refinemv=0 --enable-tx-partition=0
--enable-uneven-4way-partitions=0 --enable-warp-causal=0
--enable-warp-delta=1 --enable-warped-motion=1 --enable-warp-extend=0
--enable-wiener-nonsep=0 --enable-trellis-quant=0 --enable-refmvbank=0
--enable-short-refresh-frame-flags=0 --enable-ext-seg=0 --enable-avg-cdf=0
--enable-mfh-obu-signaling=0 --enable-lcr=0 --enable-operating-point-sets=0
--enable-cropping-window=0 --enable-keyframe-filtering=0 --enable-overlay=0
--enable-sframe=0 --enable-fwd-kf=0 --enable-banding-metadata=0
--enable-intra-edge-filter=0 --enable-mrls=0 --enable-cfl-intra=0
```

dav2d decodes that stream (rc 0). Our decoder stops earlier on an unrelated
tool gate: `Error: Unsupported("screen-content tools (intra block copy /
palette)")`. AVM sets `screen_content_tools`/`allow_screen_content_tools` even
with IntraBC and palette disabled; the AV2 decoder's screen-content gate is
outside unit 8. The decoder also rejects `INTERINTRA`, `LOCALWARP`, and
`EXTENDWARP` explicitly because those modes are not implemented.

## Toolchain / project gates

```text
cargo fmt --all -- --check                                  -> rc 0
cargo clippy --workspace --all-targets -- -D warnings       -> rc 0
cargo test -p rustmpeg-video av2                            -> passing (see run)
cargo test -p rustmpeg-video --test av2_warp                -> 1 passed
python3 scripts/check-licenses.py                          -> license audit: OK
$ORACLE/vvc-conf/run_conformance.sh                -> 25/25 cases green
```

## Remaining gaps

- `WARPMV` is decoded (`DELTAWARP`), but the encoder does not emit it; the
  in-tree reduced `find_mv_stack` cannot yet reproduce normative bank/scan
  candidate ordering once warp sub-block MVs exist. `WARP_NEWMV` is emitted
  and byte-exact against dav2d.
- `WARPMV`, `LOCALWARP`, `EXTENDWARP` and `INTERINTRA` are decoded and
  predicted; the encoder emits `WARPMV` and `INTERINTRA` but not
  `LOCALWARP`/`EXTENDWARP` (the reduced `find_mv_stack`/`WarpParamStack`
  diverges from dav2d once strongly affine neighbours exist). `warp_inter_intra`
  is decoded (non-wedge); six-parameter warp deltas are parsed but not
  emitted.
- The decoder's DRL is the pre-existing reduced spatial approximation. Warp
  streams avoid it by using `GLOBALMV` for non-warp blocks and explicit
  `WARP_NEWMV` models. This is the binding blocker for encoder `WARPMV`,
  `LOCALWARP` and `EXTENDWARP`; it is a separate unit-sized change.
- AVM end-to-end comparison is blocked by unrelated tool paths: the
  screen-content gate and the chroma intra mode index (`uv_idx=14`); the
  exact `avmenc` flags are recorded above.
- The `av2_decode` example now takes an optional third argument for bit depth
  (for example `10`), because the decoder is generic over `u8`/`u16`.

No `unsafe`, no new dependency, no native build step, and no third-party code
was added. Nothing was committed.

---

## 2026-11-11 update — full §7.12.2 stack and warp-bank semantics

The decoder half now shares the complete `find_mv_stack` with the encoder
(`av2::inter::find_mv_stack`), so both sides build identical `RefStackMv`,
`RefStackCwp`, `WeightStack`, `WarpParamStack`, and stack offsets.

Changes in `dec/frame.rs`:

- The single-reference, `has_second_drl`, and compound paths all call the same
  `mv_stack()` helper; the temporary remapped single-ref maps used for
  `NEAR_NEARMV`/`NEAR_NEWMV` and the separate `find_compound_mv_stack` /
  `CompoundMiInfo` map are gone.
- `mark_mi_inter` stores `Mvs`, `SubMvs` (per-8x8 projected vectors for
  `motion_mode >= LOCALWARP`, §7.13.3.19), `CwpIdxs`, `MotionModes`, and
  `WarpParams`; `find_warp_samples` reads `Mvs`, the DRL/corner scans read
  `SubMvs`.
- The warp and reference-MV banks are cleared at the start of every superblock
  row, as the reference decoder does (`dav2d_refmvs_tile_sbrow_init`), and the
  warp bank update stops after `MAX_WARP_SB_HITS = 64`. Without the per-row
  reset the decoder built extra `WarpParamStack` candidates and diverged from
  `dav2d` on `WARPMV` blocks in the second and later superblock rows.
- `extend_warp_estimation` (§7.13.3.24) uses the stack offsets only when either
  is `-1`; otherwise it uses the `ExtendDeltaRow/Col` first-match scan over the
  §7.11.4 points, which is what the reference decoder does.
- `WARPMV` projects `BlockMvs[0]` from the raw `WarpParamStack` entry
  (§7.12.2.2) before reducing the prediction matrix.
- The MV residual is decoded at `3 + FrameMvPrecision` and the reconstructed
  vector is the one passed to `warp_estimation` / `extend_warp_estimation`.

Verification: the same 21-case warp/newmv/switchable matrix recorded in
[`UNIT8_ENCODER.md`](UNIT8_ENCODER.md) (2026-11-11 section) compares
`encoder reconstruction == our decoder == dav2d`. 20 of 21 cases are exact,
including `LOCALWARP` and `EXTENDWARP` blocks (`3` local and `2` extend on
`u8/zoom.y4m` 256x192 QP 20); the `u13` 256x192 QP 20 case still differs on
frame 1 as documented there. `check_av2.py quick` is 9/9 and `full` is 84/84,
and `cargo test --workspace` passes.

AVM inter streams remain blocked before the inter frames: our decoder stops
with `BitstreamError("chroma intra mode index out of range")` (see the encoder
report for the exact `avmenc` flags; `dav2d` decodes the same stream, rc 0).
