# AV2 release unit 6 — encoder/shared half (NEWMV residual + switchable subpel filters)

Date: 2026-10-29
Scope: AV2 v1.0.0 normative `NEWMV` motion-vector residual coding and
switchable subpel filter selection/signalling on the encoder/shared side, with
external byte-exact verification against `dav2d`. The decoder
(`crates/rustmpeg-video/src/av2/dec/*`) is agent 2's half and is not modified
here.

## Normative findings

| Item | AV2 v1.0.0 | dav2d reference (read-only) |
| :-- | :-- | :-- |
| Single-reference mode syntax (`single_mode`, `YMode`) | §5.20.7.6 | `decode.c` single-ref branch |
| DRL index (`read_drl_idx`) | §5.20.7.6, §5.20.7.8 | `decode.c` DRL branch |
| Assign MV / MVD precision | §5.20.7.13 | `decode.c` `assign_mv()` |
| Joint-shell MV syntax | §5.20.7.20, §5.20.7.21 | `dav2d_msac_decode_symbol_adapt_c`, MV CDFs |
| `find_mv_stack` | §7.12.2 | `dav2d_refmvs_find()` in `refmvs.c` |
| Scan point / scan col | §7.12.2.6 / §7.12.2.5 | `add_spatial_candidate()`, scan loops |
| Search stack / add reference MV | §7.12.2.12 / §7.12.2.10 | `add_candidate_sngl()` |
| Extra search and clamping | §7.12.2.20, §7.12.2.23 | `dav2d_refmvs_find()` tail |
| `clamp_mv_row` / `clamp_mv_col` | §5.20.9.4–5 | `refmvs.c` |
| Switchable filter syntax and `needs_interp_filter` | §5.20.7.6 | `decode.c` `has_subpel_filter` |
| Filter context | §8.3 (`TileInterpFilterCdf`) | `get_filter_ctx()` in `env.h`, `boff[]` in `decode.c` |
| 8-tap kernels and rounding | §7.13.3.18 (`Subpel_Filters`), §7.13.3.16 (`InterRound0/1`) | `mc*` / `recon_tmpl.c` |
| TX partition position | §5.20.6.2, §5.20.6.3 | `read_tx_part()` called after the inter mode info |

## Root causes found

The previous audit reported byte differences on inter frames while the key frame
matched. Four independent defects were found and fixed:

1. **TX-partition order.** The encoder wrote `read_tx_partition()` before the
   inter mode info (before the MVD and the switchable filter). The transform
   tree is entered only after the complete inter mode info (§5.20.6.2), so
   `dav2d` consumed the partition flag as the next mode symbol. Non-warp and
   warp blocks now emit it once, after the MVD / warp delta / warp intra /
   subpel filter.
2. **Filter context at superblock boundaries.** The filter context used the
   near-corner above neighbour and decided availability with
   `by4_in_sb > 0`, where `by4_in_sb = by4 & 31`. For 64x64 superblocks
   (`sb_step4 = 16`) that reports an above neighbour at the first row of every
   odd superblock row, which `dav2d` treats as unavailable. The context also
   ignored the normative far-edge `boff[]` selection. It now uses
   `get_filter_ctx_for_block()`: far-left / far-above first, then near-left /
   near-above, with `have_top_in_sb = (by4 & (sb_step4 - 1)) != 0` and the
   compound `+4` context base.
3. **DRL approximation.** `build_drl_candidates()` ranked a small weighted
   left/above list. It is replaced for the encoder by `find_mv_stack()`
   (single reference, `use_ref_frame_mvs = 0`, `DrlReorder` off, no TIP, no
   compound): the seven spatial scan points, scan-col with the
   `MiColBase` check, `search_stack` pruning, the identity global candidate,
   the >32x32 mixtures, and both clamping passes. The optional reference MV
   bank is disabled in the sequence header when `NEWMV` or switchable filters
   are enabled, because the encoder does not model that tool.
4. **Subpel rounding.** `interpolate_subpel_block()` normalised each axis with
   `(acc + 32) >> 6`. The normative process uses 7-bit filters with
   `InterRound0 = 3` (horizontal, 1/16-pel intermediate) and
   `InterRound1 = 11` for a single reference. `MC_SUBPEL_FILTERS` stores the
   spec filters halved, so the equivalent passes are `(acc + 2) >> 2` and
   `(acc + 512) >> 10`. This removed the remaining 1-LSB differences.

## What was implemented

- `me.rs`: normative two-pass subpel rounding in `interpolate_subpel_block()`
  (all single-reference MC and the filter RDO use it); new tests for a constant
  plane and for the `acc = 30` rounding case that distinguishes the normative
  two-pass from the old single-pass.
- `inter.rs`: normative single-reference `find_mv_stack()` with
  `RefMvStack`, `clamp_mv_row()` / `clamp_mv_col()`, scan-point/scan-col,
  search-stack pruning, extra search, and clamping. New tests for the
  left/above scan order and candidate clamping.
- `frame.rs`: encoder uses `find_mv_stack()` for the predictor; `NEWMV` MVD is
  coded with `mv_prec = 3 + FrameMvPrecision` (quarter-pel frame header,
  joint-shell shift 1); `read_tx_partition()` moved to the transform tree;
  normative filter context; `newmv_blocks` and `switchable_filter_blocks`
  statistics.
- `entropy.rs`: `get_filter_ctx_for_block()` implements the normative
  `boff[0]` / `boff[1]` neighbour selection and compound context base. The old
  `get_filter_ctx()` wrapper is retained for the decoder half.
- `mod.rs`: reference MV bank disabled for `NEWMV` / switchable-filter streams.
- `examples/av2_encode.rs`: prints per-frame `newmv` and `subpel` counts.
- `docs/av2/UNIT6_ENCODER.md`: this report.

## Verification

Environment: Linux, one thread, `dav2d` revision `bff8e5c`
(`$ORACLE/dav2d`), `dav2d --threads 1 --framedelay 1
--strict 1 -q`. Inputs are the FFmpeg-generated `zoompan` clips
`$ORACLE/u8/zoom64.y4m`, `zoom.y4m` (256x192) and
`zoom64_10.y4m` (10-bit), which contain subpixel motion.

Encoder invocation (all positional):

```bash
./target/release/examples/av2_encode OUT.ivf 3 W H QP 0 0 0 420 0 0 0 0 \
  NEWMV SWITCHABLE 0 BIT_DEPTH 0 0 REC.y4m 0 0 0 INPUT.y4m 0 0 0
dav2d -i OUT.ivf -o DAV.y4m --threads 1 --framedelay 1 --strict 1 -q
python3 $WORK/cmpy4m.py REC.y4m DAV.y4m
```

`cmpy4m.py` compares frame payload bytes only (the Y4M header spelling
`C420jpeg` vs `C420` differs; samples are identical).

Result: **19/19 cases byte-exact**, encoder rc 0, `dav2d` rc 0.

| Resolution | QP | NEWMV-only | Switchable-only | Both |
| :-- | --: | :-- | :-- | :-- |
| 64x64 8-bit | 20 | OK, newmv 2 | OK | OK, newmv 2 / subpel 1 |
| 64x64 8-bit | 100 | OK, newmv 1 | OK | OK, newmv 1 / subpel 0 |
| 64x64 8-bit | 140 | OK, newmv 1 | OK | OK, newmv 1 / subpel 0 |
| 256x192 8-bit | 20 | OK, newmv 21 | OK | OK, newmv 20 / subpel 7 |
| 256x192 8-bit | 100 | OK, newmv 15 | OK | OK, newmv 14 / subpel 10 |
| 256x192 8-bit | 140 | OK, newmv 15 | OK | OK, newmv 14 / subpel 13 |
| 64x64 10-bit | 100 | — | — | OK, newmv 1 / subpel 0 |

Counts are per stream over all three frames (`newmv` = `YMode == NEWMV`
blocks, `subpel` = blocks coded with a non-default switchable filter).

`NEWMV` is asserted non-zero in every NEWMV-only and combined case.
Non-default filters are asserted non-zero in the 256x192 combined cases
(7/10/13) and the 64x64 QP 20 combined case. In switchable-only streams all
neighbour MVs are integer, so the three filters are identical at integer phase
and the RDO tie-breaks to the default filter; the filter syntax is still coded
for every inter block. Non-default filters therefore require `NEWMV` to produce
subpel MVs.

Baseline regression: `python3 $ORACLE/check_av2.py quick` →
9/9 `OK` with both tools off.

## Toolchain gates

```text
cargo fmt --all -- --check                                  -> rc 0
cargo clippy --workspace --all-targets -- -D warnings       -> rc 0
python3 scripts/check-licenses.py                           -> license audit: OK (all packages permissive, no copyleft)
cargo build --release -p rustmpeg-video --examples          -> rc 0
cargo test -p rustmpeg-video av2                            -> 129 passed, 3 failed, 3 ignored
```

The three failures are the decoder-half roundtrip tests, not encoder failures:

- `test_decode_matches_encoder_recon_with_newmv`:
  `Unsupported("sub-block transform partitioning")`.
- `test_decode_matches_encoder_recon_with_switchable_subpel`: `luma mismatch`.
- `test_decode_matches_encoder_recon_with_newmv_and_switchable_and_filters`:
  `Unsupported("inter IDTX transform type")`.

`av2_decode` on the new streams stops at the first inter frame with
`Unsupported("sub-block transform partitioning")`. The in-tree decoder still
reads `read_tx_partition()` before the MVD / subpel filter and still uses the
near-corner filter context; aligning it to the normative order and context is
agent 2's half. The encoder-vs-`dav2d` comparison above is exact on every case.

New unit tests:

- `av2::me::tests::test_subpel_constant_plane_is_preserved`
- `av2::me::tests::test_subpel_rounding_matches_normative_two_pass`
- `av2::inter::tests::test_find_mv_stack_scans_left_then_above`
- `av2::inter::tests::test_find_mv_stack_clamps_candidates`
- `av2::tests::test_encoder_newmv_and_switchable_stats` (half-pel ramp; asserts
  `newmv_blocks > 0` and `switchable_filter_blocks > 0`)

## Deliberate limits and remaining risks

- The encoder uses `dav2d`/AV2 single-reference `find_mv_stack` without the
  temporal scan (`use_ref_frame_mvs = 0`), the reference MV bank
  (`enable_refmvbank = 0`), TIP, or compound prediction. Enabling those tools
  requires the corresponding stack processes.
- The encoder still emits only `PARTITION_NONE` and one TX-partition flag per
  coding block; the flag position is normative for blocks up to 64x64.
- Non-default switchable filters need subpel MVs, so they appear in the
  `NEWMV + switchable` cases, not in switchable-only streams.
- The decoder half is not conformant for the new streams until it adopts the
  same mode-info order and filter context.

Nothing was committed. No `unsafe`, no new dependency, no native build step, and
no third-party code was added.
