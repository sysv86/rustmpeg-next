# Unit 11 encoder: palette, IntraBC, and ISP

Status: encoder half complete, including combined palette + IntraBC frames.
Nothing was committed.

## Scope

AV2 v1.0.0 screen-content tools in `crates/rustmpeg-video/src/av2/`:

- Palette mode: `palette_mode_info()` and `palette_tokens()` (AV2 Spec v1.0.0
  §5.20.8.1, §5.20.8.4), `get_palette_cache()` (§5.20.8.1),
  `get_palette_color_context()` (§5.20.8.5), and the palette CDFs in §8.3.2
  (Table 8.1, `Default_Identity_Row_Y_Cdf`).
- Intra Block Copy: `use_intrabc` in `intra_frame_mode_info()`
  (§5.20.5.3), `read_intrabc_info()` (§5.20.5.4), the BVP default candidates
  from §7.12.2.22 (`add_to_ref_bv`), and `is_mv_valid()` / local IBC buffer
  rules from §6.19.7.12 and §5.20.5.14.
- Intra Sub-Partitions: **not present in AV2 v1.0.0.** No syntax element or
  reconstruction process exists in the specification, so ISP is not emitted.
  `ToolConfig::enable_isp` is retained for API compatibility and ignored;
  `predict_isp_strips` is documented as a non-signalled helper.

Reference behavior was checked against dav2d `src/decode.c`
(`read_pal_indices`, `read_pal_plane`, `read_intrabc_info`) and
`src/recon_tmpl.c` (`dav2d_read_pal_plane`, `dav2d_copy_pal_block_y`,
intra-block-copy `mc`). No dav2d code or tables were copied.

## Encoder changes

- `screen.rs` now implements the normative palette bitstream: palette size,
  above/left cache bits, sorted delta-coded colors with a dynamically chosen
  `palette_num_extra_bits_y`, `palette_direction`, `identity_row_y`, the
  `ColorOrder`/`ColorContextHash` index coding, and the local IBC 64x64
  buffer ring with causal `is_mv_valid` checks.
- Two-color palettes are promoted to three entries. This avoids the
  two-symbol `palette_color_idx_y` path and keeps every coded palette on the
  general three-or-more-symbol CDF update.
- `frame.rs` integrates both tools into the existing intra RDO path:
  - IntraBC is evaluated before the intra mode syntax, codes
    `use_intrabc`, `skip_txfm = 1`, BVP mode/DRL/MVD, and an integer copy
    for luma and chroma without residual syntax.
  - Palette is evaluated as an intra mode, forces `DC_PRED`, codes the
    presence flag for every block where `allow_screen_content_tools` makes it
    present (including IntraBC-only frames), then codes palette tokens before
    the transform partition.
  - `PaletteSizes`/`PaletteColors` and `IBCCoded` maps are maintained at 4x4
    granularity for the neighbour contexts.
  - IntraBC neighbour contexts are reset per plane: luma `Lcoef` over the luma
    footprint and chroma `Ccoef` over the chroma plane footprint at chroma 4x4
    indices (`coeffs()` `x4 = startX >> 2`, §5.20.7.27). The earlier
    luma-indexed chroma reset desynchronised the chroma `txb_skip` context of
    the block below the IntraBC block; see the root-cause section below.
- `mod.rs` sets `allow_screen_content_tools`, `allow_intrabc`,
  `allow_local_intrabc = 1` (global IntraBC disabled), `force_integer_mv`,
  and `max_bvp_drl_bits` in the frame header. IntraBC streams also disable the
  reference MV bank so the BVP stack stays at the normative default entries.
- `examples/av2_encode.rs` takes optional `[palette 0|1] [intrabc 0|1]`
  arguments (positions 25 and 26) and prints per-frame palette/IntraBC/ISP
  block counts.

## Verification commands and results

Environment: `$ORACLE/dav2d`, one thread,
`--framedelay 1 --strict 1`. Reconstruction is compared frame-by-frame
(Y4M frame payloads, not headers) against `dav2d` output.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p rustmpeg-video av2
python3 scripts/check-licenses.py
python3 $ORACLE/check_av2.py quick
python3 $ORACLE/check_av2.py
```

Results:

- `cargo fmt --check`: clean.
- `cargo clippy ... -D warnings`: clean (only pre-existing unused-dependency
  manifest warnings).
- `cargo test -p rustmpeg-video av2`: 119 passed, 0 failed, 2 ignored.
- `check-licenses.py`: `license audit: OK`.
- `check_av2.py quick`: 9/9 OK.
- `check_av2.py` full matrix: 84/84 OK with screen-content tools off.

Screen-content matrix (8-bit 4:2:0, 3 frames, palette-only / IntraBC-only /
both; QP 20/100/140; 64x64, 130x66, 256x192):

| Case | Palette blocks | IntraBC blocks | dav2d rc | recon |
| :-- | --: | --: | --: | :-- |
| 64x64, palette | 2 | 0 | 0 | identical |
| 64x64, IntraBC | 0 | 1 | 0 | identical |
| 64x64, both | 2 | 1 | 0 | identical |
| 130x66, palette | 4 | 0 | 0 | identical |
| 130x66, IntraBC | 0 | 1 | 0 | identical |
| 130x66, both | 4 | 1 | 0 | identical |
| 256x192, palette | 8 | 0 | 0 | identical |
| 256x192, IntraBC | 0 | 1 | 0 | identical |
| 256x192, both | 8 | 1 | 0 | identical |

All nine rows repeat identically at QP 20, 100, and 140. Additional layouts
(both tools on, 256x192, QP 100, 3 frames): 10-bit 4:2:0 (8 palette, 1
IntraBC), 8-bit 4:2:2 (8/1), and 8-bit 4:4:4 (8/1), all `dav2d` rc 0 and
frame-identical reconstruction.

### Combined palette + IntraBC: root cause and fix

The combined stream was rejected by `avmdec` (`Invalid intrabc dv`) and
diverged/crashed in `dav2d`. The divergence was not in the IntraBC block
itself: the bit offsets of every block up to and including the IntraBC block
matched the reference. The first mismatch was the **chroma coefficient syntax
of the block below the IntraBC block** (9 bits longer in AVM). Palette is not
the trigger: an IntraBC-only stream from the same clip reproduces the failure,
while IntraBC-only clips whose next block does not read the corrupted chroma
context stay byte-exact.

`encode_intrabc_leaf()` updated the neighbour entropy contexts with the luma
4x4 footprint and luma column/row indices for `Ccoef`:

```text
a.ccoef[0][bx4_in_sb + i] = 0x40;   // luma indices
l.ccoef[0][by4_in_sb + i] = 0x40;   // luma indices, bh4 rows
```

AV2 derives `AboveLevelContext[plane][x4+i]` / `LeftLevelContext[plane][y4+i]`
from `coeffs(plane, startX, startY, txSz)` with `x4 = startX >> 2` in **plane**
sample coordinates (§5.20.7.27), and an IntraBC block (`skip_txfm = 1`) resets
those arrays over the **plane block size** (`av2_reset_entropy_context()` in
AVM, invoked for skip blocks). For 4:2:0 a 16x16 luma IntraBC block has an 8x8
chroma block, i.e. 2x2 chroma 4x4 units. The encoder wrote the zero context over
4 luma rows at luma columns, overwriting the chroma left-context row that the
next block reads, while leaving the real chroma context columns stale. The
reference decoder therefore used a different `txb_skip` context for the next
block's chroma and parsed coefficients where the encoder wrote "all zero".

Fix: the IntraBC context update now resets `ccoef[0..2]` (and `uvmode`) at
chroma 4x4 granularity over `(bw4 >> ss_x, bh4 >> ss_y)`, clamped to the frame,
and no longer writes chroma contexts at luma indices. Luma `lcoef` stays at the
luma footprint. The palette block already used the generic intra path and was
not affected.

Verification (our encoder, combined palette + IntraBC, 3 frames, generated
screen-content clips; `palette` and `intrabc` counts asserted `> 0` on frame 0):

| Case | Palette | IntraBC | dav2d rc | recon == dav2d | avmdec rc |
| :-- | --: | --: | --: | :-- | --: |
| 64x64 QP 20 | 3 | 1 | 0 | yes | 0 |
| 64x64 QP 100 | 1 | 1 | 0 | yes | 0 |
| 64x64 QP 140 | 1 | 1 | 0 | yes | 0 |
| 256x192 QP 20 | 30 | 1 | 0 | yes | 0 |
| 256x192 QP 100 | 9 | 1 | 0 | yes | 0 |
| 256x192 QP 140 | 9 | 1 | 0 | yes | 0 |
| 256x192 QP 20, 10-bit 4:2:0 | 21 | 1 | 0 | yes | 0 |

Driver: `$WORK/u11_combined_verify.py`; it compares Y4M frame payloads
(not headers) and runs both reference decoders on every stream.

Command shape used for each row:

```bash
target/release/examples/av2_encode out.ivf 3 W H QP 0 0 0 LAYOUT \
  0 0 0 0 0 0 0 BITDEPTH 0 0 rec.y4m 0 0 0 input.y4m PALETTE INTRABC
dav2d -i out.ivf -o dav.y4m --threads 1 --framedelay 1 --strict 1 -q
```

The exact run log is in `$WORK/unit11_results.txt` on the build
machine.

## Deliberate limits and risks

- The encoder emits at most one IntraBC block per frame. Later IntraBC blocks
  would see earlier IntraBC motion vectors in the normative BVP candidate
  stack; the encoder currently uses the fixed default candidate list instead
  of reproducing that full stack. The first block still exercises the real
  IntraBC syntax and reconstruction path.
- The reference MV bank is disabled for IntraBC streams for the same reason.
- IntraBC is local-area only (`allow_global_intrabc = 0`); the local 64x64 ring
  is implemented, but the global delayed-SB search is not.
- Palette cache entries are never reused; every cache entry is coded as a
  zero `use_palette_color_cache_y` bit. This is valid syntax but leaves a
  small RD gain on the table.
- Palette tokens always use row-major direction (`palette_direction = 0`) and
  never split the palette into delta-coded ranges beyond the normative sorted
  delta chain.
- Palette and IntraBC are rejected when segmentation is active; the encoder
  does not emit the IntraBC `read_skip()`/segment-id ordering together with
  segmentation.
- No RD or speed measurement is claimed for screen-content input beyond the
  conformance counts above. These tools remain opt-in through
  `ToolConfig::enable_palette` / `ToolConfig::enable_intrabc`.
- Our decoder now decodes palette and IntraBC (`docs/av2/UNIT11_DECODER.md`);
  palette-only and IntraBC-only frames are byte-exact against `dav2d`. The
  combined-stream encoder bug is fixed (above), and the decoder-side analogue is
  fixed in the same change (`docs/av2/UNIT11_DECODER.md`): our decoder, the
  encoder reconstruction, and `dav2d` now agree byte for byte on combined
  palette + IntraBC streams, and `avmdec` accepts them.
