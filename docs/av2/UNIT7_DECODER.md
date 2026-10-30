# AV2 release unit 7 — decoder half (compound prediction, wedge, mask blending)

Date: 2026-10-29
Scope: AV2 v1.0.0 compound inter prediction on the decode side: reference-pair
signalling, compound mode info order, average/wedge/difference-weighted and
distance-weighted (CWP) blending, and the compound branch of `find_mv_stack`.
The encoder/shared half is recorded in
[`UNIT7_ENCODER.md`](UNIT7_ENCODER.md). Only
`crates/rustmpeg-video/src/av2/dec/*` and this document are changed by this
half.

## Normative findings

| Item | AV2 v1.0.0 | dav2d reference (read-only) |
| :-- | :-- | :-- |
| Compound reference pair (`comp_mode`, `read_compound_ref`) | §5.20.7.10, §5.20.7.11 | `decode.c` compound-ref loop |
| Compound mode symbols (`is_joint`, `compound_mode_non_joint`, `compound_mode_same_refs`) | §5.20.7.6 | `decode.c` `comp_mode_joint` / `comp_mode` |
| Compound type (`comp_group_idx`, `compound_type`, `wedge_*`, `mask_type`) | §5.20.7.16, §5.20.7.18 | `decode.c` `comp_type_masked`, `read_wedge_idx` |
| Distance-weighted `cwp_idx` and `Cwp_Weighting_Factor` | §5.20.7.6 | `decode.c` `cwp_idx` |
| `assign_mv` for compound modes, `jmvd_scale_mode`, `get_mv_projection` | §5.20.7.13, §7.9.4 | `decode.c` compound `assign_mv` |
| Compound `find_mv_stack` (scan, compound search, derived, extra search, clamp) | §7.12.2, §7.12.2.10, §7.12.2.13–14, §7.12.2.20, §7.12.2.23 | `refmvs.c` `dav2d_refmvs_find()` |
| `has_second_drl` per-list stacks | §5.20.7.6 | `decode.c` two `dav2d_refmvs_find()` calls |
| Wedge codebook and master mask | §7.13.3.27 | `wedge.c` |
| Difference-weight mask | §7.13.3.28 | `mc_tmpl.c` `w_mask_c` |
| Mask blend | §7.13.3.30 | `recon_tmpl.c` |
| Compound MC rounding (`InterRound0=3`, `InterRound1=7`) | §7.13.3.16 | `mc_tmpl.c` |

`Wedge_Bits[MiSize]` is non-zero exactly for blocks whose width and height are
in `8..=64` samples, and `is_thin_4xn_nx4_block()` is
`(w == 4 && h >= 16) || (h == 4 && w >= 16)` (§5.20.7.16). Both are now
implemented explicitly instead of the previous size heuristics.

## Root causes fixed

1. **Compound mode-info order.** The decoder read `read_tx_partition()` before
   the MV residual, `read_compound_type()` and `cwp_idx`, and read the
   switchable subpel filter before the compound type. The normative order is
   `assign_mv` → `read_compound_type` → `cwp_idx` → subpel filter →
   transform tree (§5.20.7.6, §5.20.6.2). This desynchronised the arithmetic
   decoder on every masked/CWP stream.
2. **Missing `jmvd_scale_mode`.** `JOINT_NEWMV` codes `jmvd_scale_mode` between
   `is_joint` and the DRL index (§5.20.7.6). The decoder did not read it and
   projected the whole MV instead of the MVD. Both are fixed; the encoder does
   not currently select `JOINT_NEWMV`, so no existing stream changes.
3. **Compound stack absent.** The decoder used the encoder's reduced
   `build_drl_candidates()` for compound predictors. It now keeps a parallel
   per-4x4 `CompoundMiInfo` map and calls the normative
   `find_compound_mv_stack()` for every non-`has_second_drl` compound mode.
4. **`has_second_drl` list selection.** `NEARMV_NEARMV` and `NEARMV_NEWMV`
   build one single-reference stack per list (§5.20.7.6). The previous
   single-reference stack only looked at the neighbour's list 0, so a compound
   neighbour never contributed its list-1 MV. The decoder now synthesises a
   per-list `WarpMiInfo` map from `CompoundMiInfo` (both lists, `candList`
   semantics of §7.12.2.10) and calls `find_mv_stack()` for each list.
5. **Mask gating and type symbol.** `comp_group_idx` is now coded for every
   non-thin compound block, and `compound_type` only when `Wedge_Bits != 0`
   (§5.20.7.16). The previous `max_dim <= 16` heuristic is gone.
6. **Reference bookkeeping.** `mark_mi_inter()` now records the coded
   references, both MVs, the motion mode, and the CWP factor instead of
   assuming `RefFrame[0] = 0`. `mark_mi_default()` clears the compound map for
   intra blocks so stale neighbours cannot enter the stack.
7. **CWP gate.** `validate_tools()` no longer rejects `enable_cwp`. TIP, BAWP,
   `mv_traj`, implicit mask blending, reference-frame MVs, OPFL/refine-MV,
   adaptive MVD, global motion, inter-intra/local/extended warp, skip mode and
   lossless remain rejected with typed errors.

## What was implemented

- `dec/frame.rs`: `comp_mi` and `ref_mi_scratch` maps, `wedge_bits()`,
  `is_thin_4xn_nx4_block()`, reference-pair loop (unchanged algorithm, now
  followed by the correct syntax order), compound mode/DRL/MV assignment,
  `read_compound_type()`/`read_wedge_mode()`/`cwp_idx`, `jmvd_scale_mode`,
  wedge and difference mask blending for luma and chroma, and the compound
  `find_mv_stack` call.
- `dec/mod.rs`: compound decode documented in the module header; a
  non-ignored encoder/decoder round trip over the compound matrix and an
  ignored dav2d byte-exact test.
- `av2/entropy.rs` and `av2/inter.rs` needed no change: the CDF accessors and
  the compound stack already existed from the encoder half.

## Verification

Environment: Linux, one thread, `dav2d` revision `bff8e5c`
(`$ORACLE/dav2d`), run as
`dav2d -i OUT.ivf -o DAV.y4m --threads 1 --framedelay 1 --strict 1 -q`.
Every case is 3 frames and requires `our decoder == encoder reconstruction ==
dav2d` on every frame. All listed cases assert `compound_blocks > 0`.

### 64x64, 8-bit, constant reference clips

The clips use two constant references (`A=64`, `B=192`) and a frame 2 that is
the exact blend for the selected mode. This keeps the ME at integer pel.

| Mode | QP 20 | QP 100 | QP 140 | Compound blocks |
| :-- | :-- | :-- | :-- | :-- |
| average | EXACT | EXACT | EXACT | 1 |
| wedge | EXACT | EXACT | EXACT | 1 |
| difference-weighted | EXACT | EXACT | EXACT | 1 |
| CWP | EXACT | EXACT | EXACT | 1 |
| combined mask 15 | EXACT | EXACT | EXACT | 1 (CWP) |

### 256x192, 8-bit, synthetic source

| Mode | QP 20 | QP 100 | QP 140 | Compound blocks (3 frames) |
| :-- | :-- | :-- | :-- | :-- |
| average | EXACT | EXACT | EXACT | 6 |
| wedge | EXACT | EXACT | EXACT | 6 |
| difference-weighted | EXACT | EXACT | EXACT | 6 |
| combined mask 15 | EXACT | EXACT | EXACT | 9 (wedge 6, CWP 3) |
| CWP | EXACT | EXACT | EXACT | 12 (CWP) |

### 10-bit

`256x192`, 10-bit, QP 100, combined mask 15: **EXACT**, 9 compound blocks
(wedge 6, CWP 3).

Totals: **34/34 external cases EXACT** (25 pre-existing encoder streams plus 9
freshly generated CWP/combined cases). The in-tree ignored test
`av2_compound_decoder_dav2d_byte_exact` covers 28 of those streams (the 64x64
wedge clip runs at QP 100 only, because the highest-quality run codes it as
intra) and passes; the non-ignored
`test_decode_matches_encoder_recon_compound` covers 28 streams (9 shapes × 3
QPs plus the 10-bit case) and passes.

### Gates

```text
cargo fmt --all -- --check                                  -> rc 0
cargo clippy --workspace --all-targets -- -D warnings       -> rc 0
cargo test -p rustmpeg-video av2                            -> 135 passed, 0 failed, 4 ignored
cargo test --workspace                                     -> all pass (295 rustmpeg-video tests, 4 ignored)
cargo test -p rustmpeg-video av2_compound_decoder_dav2d -- --ignored -> 1 passed
python3 scripts/check-licenses.py                           -> license audit: OK
bash $ORACLE/vvc-conf/run_conformance.sh            -> rc 0, 25/25 cases (50 OK, 0 failures)
```

## Deliberate limits

- `JOINT_NEWMV` is parsed (including `jmvd_scale_mode`) but the encoder does
  not select it; the decoder path is normative but only exercised by future
  streams.
- The per-list single-reference stack synthesised for `has_second_drl` modes
  implements the `candList` selection of §7.12.2.10 but not the temporal or
  reference-bank derived candidates. Those tools stay disabled by the sequence
  header, and the same reduction already exists on the encoder side.
- Reference-frame MVs, OPFL/refine-MV, TIP, BAWP, implicit mask blending,
  adaptive MVD, global motion, inter-intra/local/extended warp, skip mode, and
  lossless remain rejected with typed errors rather than silently misdecoded.
- Subpel compound residuals remain an encoder-side limit: the encoder emits
  compound only for integer-pel ME results, so this half does not prove subpel
  compound against a real-motion AVM stream.

Nothing was committed. No `unsafe`, no new dependency, no native build step,
and no third-party code was added.
