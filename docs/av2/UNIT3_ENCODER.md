# AV2 release unit 3 — encoder half (CCSO + GDF)

Date: 2026-10-30
Scope: normative AV2 v1.0.0 Cross-Component Sample Offset (CCSO) and Guided
Detail Filter (GDF) selection/signalling in the encoder. Files changed:
`crates/rustmpeg-video/src/av2/{ccso.rs,gdf.rs,frame.rs,mod.rs}`,
`crates/rustmpeg-video/examples/av2_encode.rs`, and this document. No decoder
file (`av2/dec/*`) was edited and nothing was committed.

## Normative references

- CCSO parameters: AV2 Spec v1.0.0 §5.18.7.12 `ccso_params()`, semantics
  §6.17.7.8. Block flags: §5.20.10.2 `read_ccso()`, semantics §6.19.9.2.
- CCSO application: §7.19.1 `Apply CCSO filter process`, including
  `get_ccso_luma`, `ccso_score`, `CCSO_Quant_Sz`, and `Ccso_Pos`.
- GDF parameters: §5.18.7.9 `gdf_params()`, semantics §6.17.7.5. Block flags:
  §5.20.10.3 `read_gdf()`, semantics §6.19.9.3.
- GDF application: §7.20.5 `Apply GDF filter process`.
- Tables are the normative `CCSO_OFFSET`, `CCSO_QUANT_SZ`, and `CCSO_POS`
  (`ccso.rs`) and `GDF_INTRA_PARAMS` / `GDF_INTER_PARAMS` (`gdf_tables.rs`).

Reference behaviour was read only (never copied) in dav2d rev `bff8e5c`:
`src/ccso.h`, `src/ccso_tmpl.c`, the CCSO map handling in `src/decode.c`
(`prev_ccsomap`, `cur_ccsomap`, `ccso_ref_idx`), and `src/gdf_tables.c` plus the
GDF application code.

## CCSO implementation

`ccso.rs` now has an encoder-side solver instead of the previous band-only
heuristic:

- `select_ccso_units()` fixes the per-plane `ccso_blk` map from source activity
  before tile entropy coding, because the flags are part of the tile payload.
  Units with no source variance are skipped.
- `optimize_and_apply_ccso()` searches, per plane, a set of normative
  candidates: band-only (`bo_only=1`, `max_band_log2=3`, all four
  `scale_idx` values) and edge offsets (`bo_only=0`, `quant_idx` 0/1,
  `ext_filter` 0..6, `edge_clf` 0/1, `max_band_log2` 2/3). For each candidate it
  classifies every sample with the §7.19.1 band/edge classifier, picks the
  nearest `Ccso_Offset` index per `(band, cls0, cls1)` class, and keeps the
  candidate with the lowest SSE. The chosen parameters are signalled in
  `ccso_params()`.
- Application uses the existing normative `ccso_frame_normative()` on all
  enabled planes (luma + chroma), after CDEF and before restoration/GDF.
- `sb_reuse_ccso` is implemented: the encoder stores each frame's CCSO map per
  reference slot in `ReferenceDpb::ccso_maps`, and an inter frame can copy the
  reference map and signal `sb_reuse_ccso=1` with `ccso_ref_idx`. It is
  conformant only for `CcsoLumaSizeLog2 == CCSO_LUMA_SIZE_LOG2` (256-luma
  units, `b64size == 4`), which is what the encoder emits.
- Because the in-tree decoder still rejects `sb_reuse` (`dec/frame.rs`), the
  encoder exposes it as opt-in `ToolConfig::enable_ccso_sb_reuse` (default
  `false`, example argument `ccso_sb_reuse 0|1`). The decoder half owns removing
  that rejection.

## GDF implementation

`gdf.rs` keeps the normative `gdf_frame_normative()` (§7.20.5) and adds:

- `select_gdf_units()` chooses the per-block `use_gdf` map from source luma
  variance before tile coding. At 256x192 the GDF block is 128x128, so the
  stream signals `gdf_per_block=1` and four `use_gdf` flags.
- `optimize_and_apply_gdf()` searches `gdf_pic_scale_idx` (scale 1/2) and
  `gdf_pic_qc_idx` (base and base+1), applies the filter with the fixed block
  map, and keeps the lowest-SSE result. It never changes the block map after
  the flags have been written.

## Encoder telemetry

`FrameStats` now reports `ccso_units`, `ccso_reuse_units`,
`ccso_edge_planes`, `gdf_units`, and `gdf_per_block`; the example prints them.
This is what the tests assert and what the commands below record.

## Commands and results

Build:

```bash
cargo build --release -p rustmpeg-video --examples
```

Encoder argument order (the new argument is last):

```text
av2_encode OUT.ivf FRAMES WIDTH HEIGHT QP LOG2_TILE_COLS LOG2_TILE_ROWS \
  DEBLOCK LAYOUT CDEF CCSO GDF RST NEWMV SWITCHABLE COMPOUND BIT_DEPTH \
  VAQ DELTA_Q REC.y4m VBR_KBPS WARPED_MOTION OBMC INPUT.y4m \
  PALETTE INTRABC FILMGRAIN COMPOUND_TYPES TX_TYPE_RDO REDUCED_TXTP_SET \
  DPCM RECT EXT UNEVEN CCSO_SB_REUSE
```

The 27-case matrix (64x64 and 256x192; QP 20/100/140; CCSO-only, GDF-only,
both; 8-bit I420, plus 10-bit I420, 8-bit I422 and 8-bit I444 at QP 100; three
frames; `ccso_sb_reuse=1` on every CCSO case) was run with:

```bash
target/release/examples/av2_encode OUT.ivf 3 W H QP 0 0 0 420 0 CCSO GDF \
  0 0 0 0 BD 0 0 REC.y4m 0 0 0 - 0 0 0 0 0 0 0 0 0 0 CCSO
dav2d -i OUT.ivf -o DAV.y4m --threads 1 --framedelay 1 --strict 1 -q
```

Result: **27/27 PASS**, encoder reconstruction equals `dav2d` byte for byte on
every frame. Sample telemetry (frame 1/2 are inter frames):

```text
64x64   QP 100 CCSO: ccso units 3, reuse units 3, edge planes 1-3; gdf units 0
256x192 QP 100 CCSO: ccso units 3, reuse units 3, edge planes 1-3; gdf units 0
256x192 QP 100 GDF:  ccso units 0, reuse units 0; gdf units 4, per-block true
256x192 QP 100 both: ccso units 3, reuse units 3, edge planes 1-3;
                     gdf units 4, per-block true
```

The repeatable in-tree oracle test is:

```bash
DAV2D=$ORACLE/dav2d \
  cargo test -p rustmpeg-video --lib av2_unit3_encoder_dav2d_byte_exact -- --ignored
```

Result: `1 passed; 0 failed` (23.85 s). It covers the full 27-case matrix and
asserts `ccso_units > 0`, `ccso_reuse_units > 0`, `ccso_edge_planes > 0`,
`gdf_units > 0`, and GDF per-block signalling at 256x192.

Self-consistency with the in-tree decoder (CCSO+GDF, 256x192, QP 100,
`ccso_sb_reuse=0`): `av2_decode` output equals the encoder reconstruction on all
three frames (0 differing frames).

## Gates

Run on 2026-10-30 in the repository root:

```text
cargo fmt --all -- --check                                  -> rc 0
cargo clippy --workspace --all-targets -- -D warnings       -> rc 0
cargo test -p rustmpeg-video av2                            -> 153 passed, 0 failed, 9 ignored
cargo test --workspace                                     -> all suites pass
python3 scripts/check-licenses.py                          -> license audit: OK
python3 $ORACLE/check_av2.py quick        -> 9/9 OK
```

## RD and speed

256x192, 8-bit I420, QP 100, 3 frames, one thread, release build:

| Tools | Bytes (K/I/I) | PSNR Y (K/I/I) | Wall time |
| :-- | :-- | :-- | --: |
| none | 3399 / 1678 / 1876 | 46.74 / 31.77 / 42.42 | 21 ms |
| CCSO | 3435 / 1742 / 1894 | 46.90 / 33.08 / 42.46 | 173 ms |
| GDF | 3400 / 1679 / 1878 | 46.74 / 31.77 / 42.42 | 208 ms |
| both | 3435 / 1743 / 1896 | 46.90 / 33.08 / 42.47 | 325 ms |

CCSO buys +0.16 dB on the key frame and +1.31 dB on the first inter frame for
about 2.6% more bytes. GDF is nearly neutral on this synthetic source (it
changes 6/23 samples by 1 LSB on the two inter frames). The per-plane candidate
search is the dominant cost at this small size; it is CPU-only and uses no
per-block dispatch.

## Remaining gaps

- The decoder half must remove `Unsupported("CCSO sb_reuse")` and implement the
  reference-map copy; until then the encoder keeps `sb_reuse` opt-in.
- The CCSO unit is 256 luma samples when `sb_reuse` is possible (spec
  conformance constraint), so the 64x64 and 256x192 cases carry one CCSO block
  per plane. Per-superblock CCSO units (`ccso_unit_matches_sb_size=1`) are not
  combined with `sb_reuse` because that would violate the `CcsoLumaSizeLog2`
  requirement.
- AVM was not used as an encoder oracle for this half; dav2d is the required
  reconstruction oracle and was run on every case.
