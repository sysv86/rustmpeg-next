# AV2 release unit 6 — decoder half (NEWMV residual + switchable subpel filters)

Date: 2026-10-29
Scope: decoder-side `NEWMV` motion-vector residual reconstruction, normative
`find_mv_stack` / DRL prediction, switchable subpel filter parsing and context,
and the `read_tx_partition` position, verified against the encoder
reconstruction and `dav2d`. The encoder/shared half is recorded in
[`UNIT6_ENCODER.md`](UNIT6_ENCODER.md).

## Normative findings

| Item | AV2 v1.0.0 | Reference (read-only) |
| :-- | :-- | :-- |
| Transform-tree entry after inter mode info | §5.20.6.2, §5.20.6.3 | `decode.c` `read_tx_part()` call site |
| MVD magnitude / joint shell | §5.20.7.20, §5.20.7.21 | `decode.c` `assign_mv()`, MV CDFs |
| MVD precision and rounding | §5.20.7.13 | `decode.c` `assign_mv()` |
| `find_mv_stack` / DRL | §7.12.2 | `refmvs.c` `dav2d_refmvs_find()` |
| Scan point / scan col | §7.12.2.6 / §7.12.2.5 | `add_spatial_candidate()`, scan loops |
| Search stack / add reference MV | §7.12.2.12 / §7.12.2.10 | `add_candidate_sngl()` |
| Extra search and clamping | §7.12.2.20, §7.12.2.23 | `dav2d_refmvs_find()` tail |
| `clamp_mv_row` / `clamp_mv_col` | §5.20.9.4–5 | `refmvs.c` |
| Switchable filter syntax | §5.20.7.6 | `decode.c` `has_subpel_filter` |
| Filter context (`TileInterpFilterCdf`) | §8.3 | `get_filter_ctx()` in `env.h`, `boff[]` in `decode.c` |
| 8-tap kernels and rounding | §7.13.3.18, §7.13.3.16 | `mc*` / `recon_tmpl.c` |

## Root causes found

The decoder half was blocked by three independent defects:

1. **TX-partition position.** `dec/frame.rs` read `read_tx_partition()` before
   the MVD and before the switchable subpel filter. The encoder now writes it
   after the complete inter mode info (§5.20.6.2). `dav2d` and the encoder were
   therefore in sync, but our decoder consumed the first MVD symbol as the
   transform-tree flag. With the encoder's `TxNone` flag that symbol was read as
   `1`, which raised `Unsupported("sub-block transform partitioning")`; other
   streams desynchronised into `Unsupported("inter IDTX transform type")`.
2. **Reduced DRL.** The decoder predicted `NEWMV` from
   `build_drl_candidates()` (a weighted left/above list), while the encoder uses
   the normative `find_mv_stack()`. `drl_idx` and the predictor could differ, so
   the reconstructed MV differed from the encoder's even when the MVD syntax
   parsed.
3. **Filter context.** The decoder used the reduced near-corner
   `get_filter_ctx()` wrapper. The normative `boff[0]` / `boff[1]` selection
   depends on the block width/height and on `have_top_in_sb`, so the filter
   symbol was decoded from the wrong CDF. Reverting only this call reproduces
   the switchable-only mismatch: with 192x128, 8-bit, QP 60, 3 frames, the first
   differing luma sample is frame 1 at pixel `(0, 64)` — the inter block at
   64x64 position `(0, 1)` (32x32 position `(0, 2)`); the whole block and the
   chroma below it then diverge.

## What was implemented

- `dec/frame.rs`: the single-reference `NEWMV` path now builds its predictor
  with `find_mv_stack()` over the decoder's per-4x4 `warp_mi` map, using the
  same arguments as the encoder (`use_ref_frame_mvs = 0`, no reference MV bank,
  no TIP, single reference).
- `dec/frame.rs`: `read_tx_partition()` for non-warp single-reference blocks is
  read after the MVD and after the switchable subpel filter. Warp blocks keep it
  after the warp delta and warp-intra syntax, and compound blocks keep the
  encoder's compound order (DRL, flag, MVD, filter). Intra blocks are unchanged.
- `dec/frame.rs`: single-reference and compound subpel filter contexts use
  `get_filter_ctx_for_block()` with the block dimensions, `have_left`,
  `have_top_in_sb = (by4 & (sb_step4 - 1)) != 0`, the compound `+4` base, and
  the reference slot.
- The existing MVD path (`read_mv_residual` with `mv_prec = 3 + FrameMvPrecision`,
  joint-shell magnitude, derived signs) and the `mv_clamp_to_integer` bound are
  unchanged; they now receive the normative predictor.
- The gate for an actual `tx_split = 1` remains: the decoder still rejects real
  sub-block transform trees, which the encoder does not emit by default. Only
  the false positive caused by the wrong symbol position was removed.

## Verification

Environment: Linux, one thread, `dav2d` revision `bff8e5c`
(`$ORACLE/dav2d`) with
`--threads 1 --framedelay 1 --strict 1 -q`. Inputs are the FFmpeg-generated
`zoompan` clips `$ORACLE/u8/zoom64.y4m`, `zoom.y4m`
(256x192) and `zoom64_10.y4m` (10-bit).

Commands per case:

```bash
./target/release/examples/av2_encode OUT.ivf 3 W H QP 0 0 0 420 0 0 0 0 \
  NEWMV SWITCHABLE 0 BIT_DEPTH 0 0 REC.y4m 0 0 0 INPUT.y4m 0 0 0
dav2d -i OUT.ivf -o DAV.y4m --threads 1 --framedelay 1 --strict 1 -q
./target/release/examples/av2_decode OUT.ivf OUR.y4m BIT_DEPTH
python3 $WORK/cmpy4m.py REC.y4m DAV.y4m
python3 $WORK/cmpy4m.py OUR.y4m DAV.y4m
python3 $WORK/cmpy4m.py OUR.y4m REC.y4m
```

Result: **19/19 cases exact on all three comparisons** (`our decoder == encoder
reconstruction == dav2d` on every frame), encoder and decoder rc 0, `dav2d`
rc 0.

| Resolution | QP | NEWMV-only | Switchable-only | Both |
| :-- | --: | :-- | :-- | :-- |
| 64x64 8-bit | 20 | EXACT | EXACT | EXACT |
| 64x64 8-bit | 100 | EXACT | EXACT | EXACT |
| 64x64 8-bit | 140 | EXACT | EXACT | EXACT |
| 256x192 8-bit | 20 | EXACT | EXACT | EXACT |
| 256x192 8-bit | 100 | EXACT | EXACT | EXACT |
| 256x192 8-bit | 140 | EXACT | EXACT | EXACT |
| 64x64 10-bit | 100 | — | — | EXACT |

Baseline regression with both tools off:
`python3 $ORACLE/check_av2.py quick` → 9/9 `OK`.

## Toolchain gates

```text
cargo fmt --all -- --check                                  -> rc 0
cargo clippy --workspace --all-targets -- -D warnings       -> rc 0
cargo test -p rustmpeg-video av2                            -> 132 passed, 0 failed, 3 ignored
python3 scripts/check-licenses.py                           -> license audit: OK (all packages permissive, no copyleft)
bash $ORACLE/vvc-conf/run_conformance.sh            -> rc 0, 25/25 cases OK with VTM and VVdeC
cargo build --release -p rustmpeg-video --examples          -> rc 0
```

The three ignored tests are the external-oracle tests that require `dav2d` or an
AVM stream. They are unchanged by this unit.

## Deliberate limits and remaining risks

- A real `TxPartition::TxSplit` tree is still rejected; the encoder emits
  `TxNone` unless `enable_tx_partition` is set, and no acceptance case uses it.
- The decoder's stack uses the same reduced model as the encoder: no temporal
  `find_mv_stack` scan, no reference MV bank, no TIP, and no compound prediction
  in the conformant path.
- The compound `find_mv_stack` replacement is not part of this unit; compound
  prediction remains non-conformant (unit 7).
- `docs/AV2-RELEASE.md` still lists unit 6 as `BLOCKED` in its per-unit table;
  that release-owner document was not edited by this unit.

Nothing was committed. No `unsafe`, no new dependency, no native build step, and
no third-party code was added.
