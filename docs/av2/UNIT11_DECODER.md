# AV2 release unit 11 — decoder half (palette, IntraBC, ISP)

Date: 2026-10-29
Scope: AV2 v1.0.0 screen-content decoding in
`crates/rustmpeg-video/src/av2/dec/*`, plus the shared helpers in
`crates/rustmpeg-video/src/av2/screen.rs` and the small CDF/context additions the
decoder needs. External oracle: dav2d rev `bff8e5c` and AVM `e0e3aa8`.
Committed with the IntraBC chroma entropy-context fix described below.

## ISP decision and citation

AV2 v1.0.0 has **no Intra Sub-Partitions tool**, so unit 11's ISP half is not
applicable.

- §5.20.5.3 `intra_frame_mode_info()` is the complete intra-mode syntax. It codes
  `segmentation`, `use_intrabc`, `read_skip()`, `read_gdf()`, `read_cdef()`,
  `read_ccso()`, `read_delta_qindex()`, then `read_intrabc_info()` **or**
  `read_intra_y_mode()` / `read_intra_uv_mode()` / `read_cfl_alphas()` /
  `palette_mode_info()` / `dip_mode_info()`. There is no
  `intra_subpartitions_mode`, split type, or strip count.
- §5.20.7.5 `intra_block_mode_info()` covers the same tools for an intra block in
  an inter frame and likewise has no ISP syntax element.
- §5.20.8 "Coding tools structures" contains palette and transform-type syntax
  only; there is no strip prediction or per-strip coefficient layout process.
- `grep -ni "subpart\|intra sub"` over the converted AV2 v1.0.0 text
  (`$WORK/av2spec.txt`) returns a single generic sentence about
  superblock subpartitioning (§5.20.3) and no ISP tool description. `grep -n
  "\bISP\b"` returns nothing.

`ToolConfig::enable_isp` and `screen.rs::predict_isp_strips` therefore remain
documented non-signalled helpers; the decoder has no ISP path to add.

## Decoder implementation

`dec/frame.rs`:

- Removed the `Unsupported("screen-content tools (intra block copy / palette)")`
  gate from `validate_tools()`. No other gate was removed.
- Per-block `use_intrabc` (§5.20.5.3) with the normative presence condition
  (`allow_intrabc && Block_Width <= 64 && Block_Height <= 64 && MiSize !=
  BLOCK_64X64`), the `TileIntrabcCdf` context from the two spatial neighbours,
  and `read_skip()` only for intra-frame blocks.
- `read_intrabc_info()` (§5.20.5.4): `intrabc_mode`, the `intrabc_drl_mode`
  unary loop over `max_bvp_drl_bits`, `intrabc_precision` when
  `force_integer_mv == 0`, and the MVD through `read_mv_residual()` with
  `MvCtx = MV_INTRABC_CONTEXT`.
- BVP stack: the spatial candidates followed by the four `add_to_ref_bv`
  defaults of §7.12.2.20/§7.12.2.21, capped at
  `max_bvp_drl_bits_minus_1 + 2`, with `lower_mv_precision(ONE_PEL, pred)`
  before the residual is added (§5.20.7.13).
- Reconstruction: an integer block copy into luma and both chroma planes using
  the normative `MiCols * 4` / `MiRows * 4` bounds instead of the visible plane
  size (the gap listed in the release task 11 brief), plus `is_mv_valid()` checks
  (§6.19.7.12: tile bounds, causality, `IBCCoded`, local buffer ring).
  `IBCCoded` and `update_ibc_buffers()` are driven per block (§5.20.4.1), and
  `ibc_buffer_index()` is superblock-size aware (four column slots for 64x64,
  2x2 for 128x128, §5.20.3.1).
- IntraBC residual: when `skip_txfm == 0`, `read_tx_partition` /
  `read_coef_blocks` are decoded with the inter transform-set rules and added
  to the block copy, matching the reference (`dav2d` treats
  `!b->intra || b->intrabc` as inter for transform selection).
- Palette: `palette_mode_info()` after the intra UV mode and CfL alphas and
  before `dip_mode_info()`, in the spec order. `palette_size_y_minus_2`,
  `get_palette_cache()` above/left interleaving, `use_palette_color_cache_y`,
  the first `L(BitDepth)` colour, `palette_num_extra_bits_y`, the shrinking
  `paletteBits ≤ CeilLog2(range)` delta chain, and the final ascending
  `sort(palette_colors_y, 0, size - 1)`.
- `palette_tokens()`: `palette_direction` (only when both block dimensions are
  below 64), `identity_row_y` with `TileIdentityRowYCdf[prevIdentityRow]`, the
  `PaletteSizeY`-selected colour-index CDFs with the
  `get_palette_color_context()` `ColorOrder`, and the on-screen clamp
  (`Min(block, (MiCols - MiCol) * 4)`). The colour map is the block prediction;
  normal residual coding then follows.
- Palette bookkeeping: per-4x4 `PaletteSizes` / `PaletteColors` written across
  each decoded footprint (§5.20.4.1), cleared for non-palette and IntraBC
  blocks.
- `dip_mode_info()` moved after `palette_mode_info()` to match the syntax order
  (§5.20.5.3).
- `decode_tx_part_flag()` now requires `TxMode == TX_MODE_SELECT`
  (`read_tx_size()`, §5.20.6.1); a `TX_MODE_LARGEST` frame codes no
  `tx_do_partition` bit.

`cdf.rs`:

- `CdfContext` gained `dmv: CdfMvContext` (`MV_CONTEXTS = 2`,
  `MV_INTRABC_CONTEXT = 1`, §5.20.7.20 / §8.3.2). It is initialised and reset
  with the other non-coefficient CDFs. `Default_*_Cdf` are copied into each of
  the two contexts, as the spec requires.

`screen.rs` (shared helpers only):

- `palette_color_context()` is public so the decoder uses the exact same
  `get_palette_color_context()` code path as the encoder.
- `IbcBufferState::update_sb()` / `valid_location_sb()` add the
  superblock-size-aware `ibc_buffer_index()` used by the decoder.
- `apply_intrabc_block_bounded()` copies with explicit `MiCols*4`/`MiRows*4`
  source bounds; the existing visible-size entry point is unchanged.
- `lower_mv_precision_to_integer()` implements `lower_mv_precision(ONE_PEL, mv)`.
- The IntraBC search is restricted to the current 64x64 local IBC buffer cell.
  This was required for conformance: `check_valid_local_ibc()` only proves
  causality for the buffer being filled, and a cross-cell candidate made both
  dav2d and AVM reject the stream (AVM: `Invalid intrabc dv`). The encoder still
  emits one IntraBC block per frame; the intra-block syntax and reconstruction
  path are unchanged.

## Verification — our encoder, byte-exact vs dav2d

Commands (all under `$ORACLE/u11x`, generated screen-content
Y4M clips, 3 frames):

```bash
target/release/examples/av2_encode OUT.ivf 3 W H QP 0 0 0 LAYOUT 0 0 0 0 0 0 0 BD \
    0 0 REC.y4m 0 0 0 IN.y4m PALETTE INTRABC
target/release/examples/av2_decode OUT.ivf OUR.y4m BD
dav2d -i OUT.ivf -o DAV.y4m --threads 1 --framedelay 1 --strict 1 -q
```

`OUR.y4m`, the encoder reconstruction `REC.y4m`, and `DAV.y4m` are compared
frame payload by frame payload. Driver: `$WORK/u11_verify.py`.

Result: **24/24 OK** — every case has `our == recon == dav2d` on all three
frames. Combined palette + IntraBC streams are covered separately below.

| Case | palette blocks | IntraBC blocks |
| :-- | --: | --: |
| 64x64 QP 20/100/140, palette | 3 / 1 / 1 | 0 |
| 64x64 QP 20/100/140, IntraBC | 0 | 1 |
| 130x66 QP 20/100/140, palette | 9 / 5 / 5 | 0 |
| 130x66 QP 20/100/140, IntraBC | 0 | 1 |
| 256x192 QP 20/100/140, palette | 30 / 9 / 9 | 0 |
| 256x192 QP 20/100/140, IntraBC | 0 | 1 |

Additional layouts (all `our == recon == dav2d`):

| Case | palette / IntraBC blocks |
| :-- | :-- |
| 256x192 QP 100, 10-bit 4:2:0, palette (flat clip) | 30 / 0 |
| 256x192 QP 100, 10-bit 4:2:0, IntraBC | 0 / 1 |
| 256x192 QP 100, 8-bit 4:2:2, palette | 9 / 0 |
| 256x192 QP 100, 8-bit 4:2:2, IntraBC | 0 / 1 |
| 256x192 QP 100, 8-bit 4:4:4, palette | 9 / 0 |
| 256x192 QP 100, 8-bit 4:4:4, IntraBC | 0 / 1 |

### Combined palette + IntraBC: chroma entropy-context reset

The encoder-side fix is recorded in `docs/av2/UNIT11_ENCODER.md`. The decoder
carried the same defect: `decode_intrabc_block()` updated `ccoef[0..2]` with the
luma 4x4 footprint and luma column/row indices, then reset `uvmode` over the
chroma transform dimensions. AV2 derives the coefficient neighbour arrays from
`coeffs(plane, startX, startY, txSz)` with `x4 = startX >> 2` in **plane**
sample coordinates (§5.20.7.27), and an IntraBC block resets them over the
**plane block size** (`av2_reset_entropy_context`, §5.20.4.1). For 4:2:0 a
16x16 luma IntraBC block has an 8x8 chroma block, i.e. 2x2 chroma 4x4 units, so
the luma-indexed write corrupted the chroma left-context row read by the block
below and left the real chroma columns stale. The reference decoder then used a
different `txb_skip` context and parsed coefficients where the encoder wrote
"all zero".

Fix: the luma context loops no longer write chroma `ccoef`; the chroma reset
writes `ccoef[0..2]` and `uvmode` at chroma 4x4 granularity over
`(bw4 >> ss_x, bh4 >> ss_y)`, clamped to the 32-entry context window and the
frame's chroma dimensions, with the decoded `u_res_ctx`/`v_res_ctx` (0x40 for
the skip IntraBC block). Luma `lcoef` stays at the luma footprint. No positions
are hard-coded and the same helper shape is used on both sides.

Verification (our encoder, combined palette + IntraBC, 3 frames, generated
screen-content clips; `palette` and `intrabc` counts asserted `> 0` on frame 0;
`our == recon == dav2d` on every frame payload):

| Case | Palette | IntraBC | our == recon | our == dav2d | avmdec rc |
| :-- | --: | --: | :-- | :-- | --: |
| 64x64 QP 20 | 3 | 1 | yes | yes | 0 |
| 64x64 QP 100 | 1 | 1 | yes | yes | 0 |
| 64x64 QP 140 | 1 | 1 | yes | yes | 0 |
| 256x192 QP 20 | 30 | 1 | yes | yes | 0 |
| 256x192 QP 100 | 9 | 1 | yes | yes | 0 |
| 256x192 QP 140 | 9 | 1 | yes | yes | 0 |
| 256x192 QP 20, 10-bit 4:2:0 | 21 | 1 | yes | yes | 0 |

Driver: `$WORK/u11_combined_dec_verify.py`; it also re-runs the
palette-only/IntraBC-only 24/24 matrix through `$WORK/u11_verify.py`.
The combined streams are accepted by both reference decoders and match our
encoder reconstruction byte for byte.

## Stock AVM stream: the screen-content gate is gone

This is the critical unblock. The exact tool-disable set used is the unit 8 list
(`docs/av2/UNIT8_DECODER.md`, 87 flags: `--ivf --limit=N --cpu-used=9
--threads=1 --end-usage=cbr ... --enable-intrabc=0 --enable-palette=0
--enable-sdp=0 --enable-tx-partition=0 ... --enable-cfl-intra=0`), with two
additions needed for an intra-only AVM stream to reach block decoding:

```text
--reduced-tx-type-set=2
--reduced-tx-part-set=1
```

Command:

```bash
avmenc $(the 87 flags) --reduced-tx-type-set=2 --reduced-tx-part-set=1 \
    -y -o avm_stock.ivf input.y4m       # 256x192, 1 key frame, 4043 bytes
```

`dav2d` decodes that stream (`rc 0`). Previously our decoder stopped at
`Unsupported("screen-content tools (intra block copy / palette)")` (recorded in
`UNIT8_DECODER.md` §"AVM attempt"). Today it proceeds through the sequence
header, the frame header, the tile layout and into block decoding. Three
**pre-existing, unrelated** high-level-syntax defects were found and fixed in
`dec/hls.rs` on the way, all required to parse any AVM `single_picture_header`
stream:

1. `single_picture_header_flag` (the field this tree calls
   `reduced_still_picture_header`) sets `OrderHintBits = 0`
   (§5.4.6 `sequence_inter_config`). The decoder read one `order_hint` bit that
   the stream does not contain.
2. With the same flag, `frame_size_override_flag = 0` and no bit is coded
   (§5.18.3.1). The decoder read one bit.
3. `sequence_scc_config()` sets `seq_force_screen_content_tools =
   SELECT_SCREEN_CONTENT_TOOLS` and `seq_force_integer_mv = SELECT_INTEGER_MV`
   when `single_picture_header_flag` is set, i.e. both are coded **per frame**
   (§5.4.7). The decoder skipped those two frame-header bits, which
   misaligned `allow_intrabc` and `uniform_tile_spacing_flag`
   (the visible symptom was `Unsupported("non-uniform tile layout")`).

After those fixes the stream parses `allow_screen_content_tools = 1`,
`allow_intrabc = 0`, `uniform_tile_spacing_flag = 1`, `TxMode = LARGEST`
(the `read_tx_size()` gate above was also needed) and stops at the coefficient
syntax:

```text
Error: BitstreamError("decoded EOB 982 outside the 512-coefficient block")
```

That is a coefficient-coding conformance gap (same class as the gate recorded in
`UNIT5_DECODER.md`), not a screen-content, palette, or IntraBC defect: the
stream contains no palette and no IntraBC block. Decoding the full AVM intra
stream therefore remains blocked on coefficient syntax/coding conformance, and
the byte-for-byte AVM comparison cannot be reported yet.

## Gates

```text
cargo fmt --all -- --check                                  -> clean
cargo clippy --workspace --all-targets -- -D warnings       -> clean
cargo test -p rustmpeg-video av2                            -> 119 passed, 0 failed, 2 ignored
python3 scripts/check-licenses.py                          -> license audit: OK
$ORACLE/vvc-conf/run_conformance.sh                -> 25/25 cases OK (VTM + VVdeC)
python3 $ORACLE/check_av2.py quick       -> 9/9 OK
python3 $ORACLE/check_av2.py             -> 84/84 OK
```

## Remaining gaps

- Combined palette + IntraBC frames are not byte-exact against the reference
  because of the encoder-side issue above; the decoder handles the combination
  self-consistently.
- Sub-pixel IntraBC prediction (`MvPrecision != ONE_PEL` with a fractional DV) is
  parsed but rejected with `Unsupported`; every stream in the matrix uses
  integer MVs (the encoder sets `force_integer_mv = 1`).
- The decoder does not implement `allow_global_intrabc` (only the local 64x64
  buffer); `is_mv_valid()` enforces the local constraints, and the global branch
  remains a separate tool.
- The BVP stack uses the direct spatial candidates plus the four normative
  defaults; the full normative `find_mv_stack` reduced/bank ordering is still the
  pre-existing approximation noted for inter prediction.
- Stock AVM intra streams still stop at coefficient syntax (evidence above).

No `unsafe`, no new dependency, no native build step, and no third-party code
was added.
