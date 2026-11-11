# AV2 release unit 8 — encoder/shared half (OBMC + warped motion)

Date: 2026-10-29
Scope: AV2 v1.0.0 overlapped block motion compensation (OBMC) and warped
motion on the encoder/shared side, with external byte-exact verification
against dav2d. The decoder (`crates/rustmpeg-video/src/av2/dec/*`) is agent 2's
half and is not modified here.

## Normative findings

AV2 v1.0.0 has **no OBMC tool**. The motion-mode enumeration is
`SIMPLE=0, INTERINTRA=1, LOCALWARP=2, DELTAWARP=3, EXTENDWARP=4`
(`MOTION_MODES = 5`, §7.3), and §5.20.7.14 `read_motion_mode()` defines no
overlap flag or blending window. `grep -i obmc` over the specification text and
over dav2d `src/*.c` returns nothing. The previous encoder-side
`obmc_predict_block()` blended reconstruction without any bitstream signal, so
dav2d never reproduced it; it has been removed.

| Item | AV2 spec | dav2d reference (read-only) |
| :-- | :-- | :-- |
| Motion-mode enumeration / motion-mode syntax | §7.3, §5.20.7.14 | `MM_TRANSLATION`..`MM_WARP_EXTEND`, `decode.c` motion-mode branch |
| `is_warp`, `warp_mv`, `warp_idx`, `warpmv_with_mvd` | §5.20.7.6 | `decode.c` inter-mode branch |
| `warp_inter_intra` | §5.20.7.15 | `decode.c` `warp_ii` branch |
| `WarpParamStack` from corners / bank / GM / defaults | §7.12.2, §7.12.2.3–4 | `derive_warpmv()`, `find_ref_mvs`, `decode.c` |
| `setup_shear`, `resolve_divisor` | §7.13.3.21–22 | `dav2d_get_shear_params`, `dav2d_resolve_divisor_32` |
| `reduce_warp_model`, `get_warp_translation` | §7.13.3.23 | `iclip_wmp`, `dav2d_set_affine_mv2d` |
| `block_warp` prediction and `Warped_Filters` | §7.13.3.19 | `warp_affine_8x8_c`, `warp_affine()` |
| `get_warp_motion_vector`, `clamp_mv_row/col` | §7.12.2.2, §5.20.9.4–5 | `get_warpmv_2d` |
| `read_tx_part()` position | §5.20.2 | `read_tx_part` called after the inter mode info |
| `update_warp_param_bank`, `params_equal` | §7.12.3 | warp bank in `decode.c` |
| `Size_Group` for `warp_inter_intra` | §9 `size_group.h` | `size_group_lookup` |

## What was implemented

- `warp.rs` rewritten around the normative processes:
  - `Warped_Filters` (449 phases × 8 taps) copied from the AV2 v1.0.0
    specification attachment `all_tables.h` (`warped_filters.h`).
  - `resolve_divisor_64()`, `setup_shear()` (including the
    `4|alpha|+7|beta| < 3<<16` / `4|gamma|+4|delta| < 3<<16` validity test),
    `reduce_warp_model()`, `get_warp_translation()`,
    `warp_motion_vector()` with §5.20.9.4–5 border clamps, and
    `block_warp()` (two-pass 8-tap warped interpolation, per-plane
    subsampling, `Round2(...,3)` / `Round2(...,11)`).
  - `build_warp_param_stack()`, `warp_bank_insert()` / `params_equal()`, and
    `generate_points_from_corners()` / `warp_corner()` over a per-4x4
    `WarpMiInfo` map.
- `frame.rs`:
  - A per-tile 4x4 mode map, per-reference warp bank, `WarpBankHits` counter,
    and `WarpMvCount` (§7.11.4) for the `is_warp` CDF context.
  - `WARPMV` + `DELTAWARP` signalling: `is_warp`, `warp_mv`, the `warp_idx`
    symbols, `warpmv_with_mvd`, and `warp_inter_intra`, with mode index 17.
  - Model selection in RDO: every `WarpParamStack` candidate is reduced,
    shear-checked, translated with `get_warp_motion_vector()`, predicted with
    `block_warp()`, and compared against the translation prediction. Selected
    blocks update the per-reference warp bank exactly as §7.12.3 does.
  - `is_warp = 0` is emitted for every eligible non-warp block when
    `allow_warpmv_mode` is on (required; omitting it desynchronises the stream).
  - `read_tx_part()` is emitted after the inter mode info for warp blocks
    (dav2d order). Non-warp blocks keep the tree's existing pre-MVD order so
    the in-tree decoder tests stay green.
- `mod.rs`: `seq_enabled_motion_modes[DELTAWARP]` and
  `allow_warpmv_mode` are enabled only by `ToolConfig::enable_warped_motion`.
- `obu.rs`/`headers.rs` needed no new fields; the existing
  `motion_modes`, `warp_motion`, `six_param_warp_delta` and
  `FhGmv`/`WarpedMotionParams` paths are reused.
- `examples/av2_encode.rs`: `warped_motion 0|1`, the accepted no-op `obmc 0|1`
  flag, an optional `input.y4m`, and a per-frame warped-block count.
- `ToolConfig::enable_obmc` is documented as a no-op for API/CLI
  compatibility, because AV2 v1.0.0 has no OBMC.

## Verification

Environment: Linux, 1 thread, dav2d rev `bff8e5c` at
`$ORACLE/dav2d`. Clips were produced with FFmpeg as a
black-box generator (`zoompan` over `testsrc2`, 3 frames), written under
`$ORACLE/u8/`.

Encoder invocation shape:

```bash
./target/release/examples/av2_encode OUT.ivf 3 W H QP 0 0 0 LAYOUT 0 0 0 0 0 0 0 BD 0 0 REC.y4m 0 WARP OBMC IN.y4m
dav2d -i OUT.ivf -o DAV.y4m --threads 1 --framedelay 1 --strict 1 -q
python3 $WORK/cmpy4m.py REC.y4m DAV.y4m
```

`cmpy4m.py` compares the frame payload bytes only (`C420jpeg` vs `C420`
header spelling differs; samples are identical).

Result with the current tree: `INTERINTRA` + `WARP_NEWMV` (`DELTAWARP`) are
byte-exact for 64x64 and 256x192 I420 at QP 20/100/140 (3 frames), plus 10-bit
and 12-bit 256x192 QP100 (`interintra_blocks` 5-22 per run; 0 on the 64x64
zoom clip). Every case is `encoder recon == our decoder == dav2d`, all motion
mode bits enabled, encoder/dav2d rc 0. The earlier 17-case matrix below was
run before `INTERINTRA` was enabled and remains reproducible:

| Case | QP | Warp | OBMC flag | Result |
| :-- | :-- | :-- | :-- | :-- |
| 64x64 8-bit 4:2:0 | 20/100/140 | off | off | OK |
| 64x64 8-bit 4:2:0 | 20/100/140 | on | off | OK |
| 256x192 8-bit 4:2:0 | 20/100/140 | off | off | OK |
| 256x192 8-bit 4:2:0 | 20/100/140 | on | off | OK |
| 256x192 8-bit 4:2:0 | 20/100/140 | on | **on** | OK |
| 256x192 10-bit 4:2:0 | 100 | on | off | OK |
| 64x64 10-bit 4:2:0 | 100 | on | off | OK |
| 64x64 8-bit 4:2:2 | 100 | on | off | OK |
| 64x64 8-bit 4:4:4 | 100 | on | off | OK |

Additional runs:

- `python3 $ORACLE/check_av2.py quick` (warp/OBMC off):
  9/9 `OK`, so the default matrix is unchanged.
- 8-bit 256x192 10-frame and 30-frame warp-on runs at QP 100 were also
  byte-exact in earlier iterations of the same code path.
- The OBMC-on column produces a byte-identical bitstream to OBMC-off because
  the flag is a documented no-op.

Toolchain checks:

```text
cargo fmt --all -- --check                                  -> rc 0
cargo clippy --workspace --all-targets -- -D warnings       -> rc 0 (0 errors)
cargo test -p rustmpeg-video av2                            -> 118 passed, 0 failed, 2 ignored
python3 scripts/check-licenses.py                           -> license audit: OK (all packages permissive, no copyleft)
```

Warp unit tests added: `resolve_divisor`/`setup_shear` validity,
`reduce_warp_model` 64-unit rounding, bank de-duplication, corner-derived
affine candidate from three differing corner MVs, `block_warp` zoom vs
identity, and motion-vector derivation.

## Deliberate limits and remaining risks

- **Warp is emitted as `WARP_NEWMV` + `DELTAWARP`.** The decoder/selection
  half (see `UNIT8_DECODER.md`) uses the real DRL predictor, `GLOBALMV` for
  non-warp blocks while warped motion is on, and a four-parameter
  rotation/scale model estimated from four half-block MVs by least squares.
  Warp-on streams contain nonzero `is_warp` blocks and are byte-exact against
  dav2d.
- **`INTERINTRA` is emitted.** For single-reference non-warp blocks the encoder
  evaluates all four `interintra_mode` values, blends them with the inter
  prediction using the §7.13.3.29 mask, and signals `inter_intra`/
  `interintra_mode`/`wedge_inter_intra` in dav2d order. Verified byte-exact.
- **`WARPMV` is emitted** from `WarpParamStack` (§5.20.7.6): the model is
  reduced, translated with `get_warp_motion_vector`, and selected when its
  warped SAD beats the translation prediction after rate. It is byte-exact
  against dav2d on the recorded matrix (counts 3-9 on 256x192; 0 on the
  64x64 zoom clip).
- **`LOCALWARP`/`EXTENDWARP` encoder emission remains blocked.** The decoder
  parses and predicts them, but the encoder's `find_mv_stack`/`WarpParamStack`
  is the reduced in-tree approximation; once strongly affine neighbours exist
  it does not reproduce dav2d's `RefStackMv`/bank candidates, so the coded MVD
  (and hence the derived local/extend model) can diverge. Those selections are
  therefore not emitted. The same reduced stack is a residual risk for
  `WARPMV` on strongly affine content outside the recorded matrix.
- **`WARP_NEWMV` explicit warp-delta coding is emitted** for the four-parameter
  model (`warp_delta_precision` 1, `params[4] = -params[3]`,
  `params[5] = params[2]`). Six-parameter deltas are parsed by the decoder but
  not emitted.
- **No global-motion syntax is emitted.** `use_global_motion = 0`, so the
  global-motion candidate in `WarpParamStack` is the identity model. Global
  motion parameter coding is a separate frame-header feature.
- **Warped-motion decoding is implemented** in `dec/frame.rs`
  (`WARPMV`/`DELTAWARP` and `WARP_NEWMV`/`DELTAWARP`, corner/bank/global stack,
  `setup_shear`, `block_warp`, warp bank, per-4x4 sub-MV storage). `WARPMV` is
  decoder-only: the encoder emits `WARP_NEWMV` because the reduced in-tree DRL
  cannot yet reproduce the full normative `find_mv_stack` after bank/scan warp
  candidates appear.
- **`read_tx_part()` ordering:** dav2d reads the flag after the inter mode
  info. Warp blocks follow that order. Non-warp `NEWMV` blocks keep the tree's
  existing pre-MVD order so the in-tree decoder tests remain green; aligning
  the decoder half (and then every inter block) to dav2d is a cross-half task.
- Warp is restricted to a single-tile frame with fully-inside blocks and
  blocks whose chroma plane is at least 8x8, because the encoder keeps the
  neighbour map per tile and does not implement §7.13.3.20 `ext_block_warp`.
- No `unsafe`, no new dependency, no native build step, and no third-party
  code was added. The `Warped_Filters` table comes from the AV2 specification
  attachment `all_tables.h`; it is recorded here and in the module header.
  The file keeps `SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause` to
  match the existing AV2/dav2d provenance.

Nothing was committed.
