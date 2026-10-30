# AV2 release unit 7 — encoder half (compound prediction, wedge, mask blending)

Date: 2026-10-29
Scope: AV2 v1.0.0 compound inter prediction on the encoder/shared side:
reference-pair signalling, `compound_mode`, average/wedge/difference-weighted
and distance-weighted blending, wedge masks, and the compound branch of
`find_mv_stack`. The decoder (`crates/rustmpeg-video/src/av2/dec/*`) is agent
2's half and is not modified here.

## Normative findings

| Item | AV2 v1.0.0 | dav2d reference (read-only) |
| :-- | :-- | :-- |
| Reference-pair syntax (`comp_mode`, `read_compound_ref`) | §5.20.7.10, §5.20.7.11 | `decode.c` compound-ref loop |
| Compound mode symbols (`is_joint`, `compound_mode_non_joint`) | §5.20.7.6 | `decode.c` `comp_mode_joint` / `comp_mode` |
| Compound type (`comp_group_idx`, `compound_type`, `wedge_*`, `mask_type`) | §5.20.7.16, §5.20.7.18 | `decode.c` `comp_type_masked`, `read_wedge_idx` |
| Distance-weighted `cwp_idx` | §5.20.7.6, §8.3 | `decode.c` `cwp_idx` |
| `find_mv_stack` compound branch | §7.12.2, §7.12.2.10, §7.12.2.13, §7.12.2.14, §7.12.2.22, §7.12.2.23 | `refmvs.c` `dav2d_refmvs_find()` |
| Wedge codebook and master mask | §7.13.3.27 | `wedge.c` |
| Difference-weight mask | §7.13.3.28 | `mc_tmpl.c` `w_mask_c` |
| Mask blend | §7.13.3.30 | `recon_tmpl.c` |
| Compound MC rounding (`InterRound0=3`, `InterRound1=7`) | §7.13.3.16 | `mc_tmpl.c` |

## Root causes found

1. **Compound mode-info order.** The compound branch emitted
   `read_tx_partition()` before the MV residual and `read_compound_type()`
   after the switchable subpel filter. The normative order is
   `assign_mv` → `read_compound_type` → `cwp_idx` → subpel filter →
   transform tree. dav2d consumed the partition flag as a compound-type
   symbol, which produced the SIGSEGV on the second inter frame and the
   frame-2 EOB error in AVM. The compound branch now writes the complete
   mode info before the filter and TX partition.
2. **Compound stack.** The encoder had no compound `find_mv_stack` branch.
   It now keeps a parallel per-4x4 compound MI map (`CompoundMiInfo`) and
   implements the scan points, scan col, compound search stack, compound
   add-derived, derived fill, extra search, mixtures, and clamping. The
   `NEARMV_NEARMV` pair and the `NEW_NEWMV` predictor come from candidate 0.
3. **Wedge index mapping.** `WEDGE_IDX_TO_ANGLE_DIST` already stores the
   within-quad angle; the encoder treated it as a global angle, so the
   `wedge_dist1`/`wedge_dist2` branch was wrong. It now derives the global
   angle from `quad * 5 + angle` and uses the `-1` entry of
   `WEDGE_ANGLE_DIST_TO_IDX` to select the distance symbol.
4. **Sequence flags.** `enable_masked_compound` and `enable_cwp` were never
   set, so wedge/difference/CWP syntax was unreachable. They are now
   enabled by `ToolConfig::enable_compound_masks` (the example enables them
   with `compound 1`). The reference MV bank is disabled for compound.
5. **Compound residual precision.** `NEW_NEWMV` residuals are coded at the
   frame MV precision (`mv_prec = 3 + FrameMvPrecision`, shift 1). The
   encoder now emits compound only when both ME MVs are integer-pel; a
   subpel ME result falls back to single prediction instead of truncating
   the residual. This is a deliberate limit of this unit.

## What was implemented

- `inter.rs`: `CompoundMiInfo`, `CompoundMvStack`, `find_compound_mv_stack`
  (compound branch of §7.12.2), plus `compound_mode`/`compound_drl_idx`
  fields on `InterBlockDecision`.
- `frame.rs`: parallel `comp_mi` map; compound RDO for average, difference-
  weighted, wedge, and CWP; reference-pair and compound-type syntax;
  wedge index/distance syntax; compound DRL indices; compound statistics
  (`compound_average_blocks`, `compound_wedge_blocks`,
  `compound_diff_blocks`, `compound_cwp_blocks`); `compound_types` mode
  mask and `enable_compound_masks` tool flag.
- `mod.rs`: sequence-level `enable_masked_compound`/`enable_cwp` and
  reference-MV-bank handling for compound.
- `entropy.rs`: no syntax change was required; the existing compound CDF
  accessors and `comp_ctx`/`compref_ctx`/`comp_group_idx_ctx` are used.
- `examples/av2_encode.rs`: per-frame compound mode counts, `compound_types`
  argument (bit 0 average, 1 wedge, 2 difference-weighted, 3 CWP), and the
  `inter`/`intra` block counts.

## Verification

Environment: Linux, one thread, `dav2d` revision `bff8e5c`
(`$ORACLE/dav2d`), run as
`dav2d -i OUT.ivf -o DAV.y4m --threads 1 --framedelay 1 --strict 1 -q`.
Encoder invocation:

```bash
./target/release/examples/av2_encode OUT.ivf 3 W H QP 0 0 0 420 0 0 0 0 \
  0 0 1 8 0 0 REC.y4m 0 0 0 INPUT.y4m 0 0 0 COMPOUND_TYPES
```

`COMPOUND_TYPES` is the mode mask; `compound 1` enables compound and all
masked/CWP modes, and the final positional argument restricts the RDO.

All results below are `encoder reconstruction == dav2d` on every frame
(3/3 EXACT), `dav2d` rc 0, encoder rc 0. Counts are per stream over the
three frames.

### 256x192, 8-bit, synthetic source

| Mode mask | Mode | QP 20 | QP 100 | QP 140 |
| :-- | :-- | :-- | :-- | :-- |
| 1 | average | 6 | 6 | 4 |
| 2 | wedge | 6 | 6 | 4 |
| 4 | difference-weighted | 6 | 6 | 4 |
| 15 | combined | wedge 6 / cwp 3 | wedge 6 / cwp 3 | wedge 4 / cwp 5 |

All 12 cases: 3/3 exact. The combined cases assert `compound_blocks > 0`,
`compound_wedge_blocks > 0`, and `compound_cwp_blocks > 0`.

### 64x64, 8-bit, constant reference clips

The 64x64 clips are generated so the two references are constants `A=64`
and `B=192`, and frame 2 is the exact blend for the selected mode (average,
wedge index 0, difference mask `38 + |A-B|/16`, or CWP weight 12). This
keeps the ME at integer-pel and isolates the blend syntax.

| Mode mask | Mode | QP 20 | QP 100 | QP 140 |
| :-- | :-- | :-- | :-- | :-- |
| 1 | average | 1 | 1 | 1 |
| 2 | wedge | 1 | 1 | 1 |
| 4 | difference-weighted | 1 | 1 | 1 |
| 8 | distance-weighted | 1 | 1 | 1 |

All 12 cases: 3/3 exact, one compound block per stream in frame 2.

### 10-bit

`256x192`, 10-bit, QP 100, mode mask 15: 3/3 exact,
`compound 9 (wedge 6, cwp 3)` on both inter frames.

### Regression and toolchain gates

```text
cargo fmt --all -- --check                                  -> rc 0
cargo clippy --workspace --all-targets -- -D warnings       -> rc 0
cargo test -p rustmpeg-video av2                            -> rc 0
cargo test --workspace                                      -> rc 0 (718 passed, 0 failed)
python3 scripts/check-licenses.py                           -> license audit: OK
python3 $ORACLE/check_av2.py quick        -> 9/9 OK (compound off)
cargo build --release -p rustmpeg-video --examples          -> rc 0
```

`dav2d` did not crash on any final matrix case. The previous SIGSEGV was
caused by the mode-info ordering defect above; it is fixed.

## Deliberate limits and remaining risks

- Compound residuals are emitted only for integer-pel ME MVs. Subpel ME
  results stay on the single-reference path, so a real-motion clip such as
  `$ORACLE/u8/zoom.y4m` decodes byte-exactly against
  dav2d but may contain no compound blocks. Subpel compound is the next
  step; the mode-info syntax and masks are already in place.
- The encoder uses `NEARMV_NEARMV` for CWP (stack pair) and `NEW_NEWMV` for
  average/wedge/difference (ME pair with the compound-stack predictor).
  `JOINT_NEWMV` is parsed by the shared syntax but not selected.
- The decoder half (`av2/dec/*`) was not modified. Its compound path still
  rejects `enable_cwp` until agent 2 lands the matching change; the
  encoder-side matrix above is verified against `dav2d` only.
- 64x64 coverage uses generated constant clips; the 180-case baseline
  matrix and `check_av2.py quick` remain green with compound off.

Nothing was committed. No `unsafe`, no new dependency, no native build
step, and no third-party code was added.
