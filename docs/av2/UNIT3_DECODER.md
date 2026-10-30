# AV2 release unit 3 — decoder half (CCSO + GDF)

Date: 2026-10-30
Scope: normative AV2 v1.0.0 Cross-Component Sample Offset (CCSO) and Guided
Detail Filter (GDF) decoding, including CCSO `sb_reuse_ccso`, plus enabling
`sb_reuse` by default in the encoder. Files changed:
`crates/rustmpeg-video/src/av2/dec/{frame.rs,hls.rs,mod.rs}`,
`crates/rustmpeg-video/src/av2/frame.rs`,
`crates/rustmpeg-video/examples/av2_encode.rs`, and this document. The shared
filter helpers `av2/{ccso.rs,gdf.rs,gdf_tables.rs}` were not modified. No
encoder syntax work was undone and nothing was committed.

## Normative references

- CCSO parameters and `sb_reuse_ccso`: AV2 Spec v1.0.0 §5.18.7.12
  `ccso_params()`, semantics §6.17.7.8; block flags §5.20.10.2 `read_ccso()`,
  semantics §6.19.9.2.
- CCSO save/load of `CcsoBlks`: §7.23 Reference frame update process
  (`save_ccso_params`, `load_ccso_params`); application §7.19.1.
- GDF parameters: §5.18.7.9 `gdf_params()`, semantics §6.17.7.5; block flags
  §5.20.10.3 `read_gdf()`, semantics §6.19.9.3; application §7.20.5.

Reference behaviour was read only (never copied) in dav2d rev `bff8e5c`:
`src/ccso.h`, `src/ccso_tmpl.c`, the CCSO map handling in `src/decode.c`
(`prev_ccsomap`, `cur_ccsomap`, `ccso_ref_idx`), `src/gdf_tables.c`, and the GDF
application code.

## CCSO `sb_reuse` decoding

The decoder now keeps the per-slot `CcsoBlks` map in
`ReferenceDpb::ccso_maps`, the same side data the encoder already populated:

1. `decode_frame` reads `sb_reuse_ccso[plane]` and `ccso_ref_idx[plane]` from the
   frame header. `ccso_ref_idx` indexes the reference list, so the DPB slot is
   `refidx[ccso_ref_idx]` (not the raw list index).
2. Before tile decoding, every plane with `sb_reuse_ccso == 1` copies its saved
   map from that slot into the current `ccso_blks`. Missing maps and map-size
   mismatches return a typed `BitstreamError` instead of misdecoding.
3. `read_ccso()` skips the `ccso_blk` symbol for those planes (§5.20.10.2); the
   preloaded map is used unchanged by `ccso_frame_normative()`.
4. After reconstruction, `finish_frame` stores the current map in every slot
   selected by `refresh_frame_flags` (`save_ccso_params`, §7.23).

Band-offset and edge-offset planes, luma and chroma, use the existing normative
`ccso_frame_normative()` (§7.19.1) unchanged: luma guidance is taken from the
pre-CDEF `CurrFrame`, the filtered samples from the CDEF frame, with the
normative band shift, `ccso_score`, `CCSO_Quant_Sz`, `Ccso_Pos`, tile clipping,
and `Ccso_Offset` lookup.

`reuse_ccso` (parameter reuse without block-map reuse) is still rejected with
`Unsupported("CCSO reuse")`. The encoder never emits it, and it was not part of
this unit's acceptance criteria.

## GDF decoding

The filter itself (`gdf_frame_normative`, §7.20.5) was already implemented and
matches dav2d; the decoder-side gap was the `gdf_per_block` gate:

- `gdf_params()` now derives `GdfBlkSize` before reading `gdf_per_block` and
  signals per-block flags exactly when
  `MiCols*4 > GdfBlkSize || MiRows*4 > GdfBlkSize ||
  (disable_loopfilters_across_tiles && (TileRows > 1 || TileCols > 1))`
  (§5.18.7.9). Previously the gate used the visible width/height and ignored
  the tile condition, which desynchronised the tile payload for frames where
  `gdf_unit_matches_sb_size == 0` or loop filters are disabled across tiles.
- `read_gdf()` now reads a `use_gdf` symbol whenever the header selected
  per-block signalling, instead of re-deriving the condition from the frame
  size. Uniform frames still apply GDF to every block without a symbol.

GDF remains luma-only and is applied after CCSO and restoration, using the
pre-CDEF `CurrFrame` outside the current 64-line stripe and the filtered frame
inside it, per §7.20.2.

## Encoder default

`ToolConfig::enable_ccso_sb_reuse` now defaults to `true`, and the
`av2_encode` example treats an absent `ccso_sb_reuse` argument as `1` (passing
`0` still disables it). The 27-case encoder matrix was rerun and still matches
dav2d byte for byte, so the default does not regress conformance.

## Verification

The repeatable in-tree oracle test is:

```bash
DAV2D=$ORACLE/dav2d \
  cargo test -p rustmpeg-video --lib av2_unit3_decoder_dav2d_byte_exact -- --ignored
```

Result: `1 passed; 0 failed` (20.6 s). It runs 21 cases × 3 frames and asserts
`our decoder == encoder reconstruction == dav2d` on every frame:

- 8-bit I420, 64x64 and 256x192, QP 20/100/140, CCSO-only, GDF-only, and both
  (18 cases).
- 10-bit I420, 256x192, QP 100, CCSO+GDF (1 case).
- 8-bit I422 and I444, 256x192, QP 100, CCSO+GDF (2 cases).

Every CCSO case asserts `ccso_units > 0` and `ccso_reuse_units > 0`; every GDF
case asserts `gdf_units > 0`. The non-ignored
`test_decode_matches_encoder_recon_with_ccso_and_gdf` also checks the same
decoder path (CCSO + `sb_reuse` + GDF, 64x64 and 256x192) without the oracle.
Sample telemetry at 256x192, QP 100, 3 frames:

```text
CCSO-only: frame 0 units 3, reuse 0, edge planes 3
           frame 1 units 3, reuse 3, edge planes 3
           frame 2 units 3, reuse 3, edge planes 3
GDF-only:  gdf units 4 per frame, per-block true
both:      ccso units 3/reuse 3/edge planes 3, gdf units 4 per-block true
```

Manual check of the same three streams (encoder reconstruction, `av2_decode`,
and `dav2d --threads 1 --framedelay 1 --strict 1 -q`) agrees on all three
frames for CCSO-only, GDF-only, and both.

## Gates

Run on 2026-10-30 in the repository root:

```text
cargo fmt --all -- --check                                  -> rc 0
cargo clippy --workspace --all-targets -- -D warnings       -> rc 0
cargo test -p rustmpeg-video av2                            -> 154 passed, 0 failed, 10 ignored
cargo test --workspace                                     -> all suites pass, 0 failed
python3 scripts/check-licenses.py                          -> license audit: OK
bash $ORACLE/vvc-conf/run_conformance.sh            -> rc 0, 25/25 cases (50/50 decoder checks) OK
DAV2D=... cargo test ... av2_unit3_encoder_dav2d_byte_exact -- --ignored -> 1 passed
```

The additional ignored test is the new unit 3 decoder oracle. No `unsafe`, no
new dependency, no native build step, and no third-party code was added.
