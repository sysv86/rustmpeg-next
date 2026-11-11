# AV2 release unit 9 — decoder half (transform types, lossless WHT, DPCM)

Date: 2026-10-29, updated 2026-11-11
Scope: AV2 v1.0.0 transform-type signalling and kernels for every transform
set (full and reduced 0–3), the lossless 4x4 WHT path at quantizer index 0
(intra and inter, `lossless_tx_size`, `txtp_lossless`, lossless `IDTX`),
DPCM intra prediction (`use_dpcm_y` / `use_dpcm_uv`), the `EobU` V-plane
context, and the `TX_PARTITION_SPLIT`/`HORZ`/`VERT` transform partitions. The
decoder lives in `crates/rustmpeg-video/src/av2/dec/*`. External verification
is byte-exact against `dav2d`.

## Normative findings

| Item | AV2 v1.0.0 | Reference decoder (read-only) |
| :-- | :-- | :-- |
| Transform set derivation | §5.20.8.3 `get_tx_set` | `recon_tmpl.c` `decode_cf` |
| Transform-type syntax | §5.20.8.2 `transform_type` | `recon_tmpl.c` `decode_cf` |
| Inverse-set tables | §5.20.8.2 `Tx_Type_Inter_Inv_Set1/2/3/4`, `Tx_Type_Inv_Long` | `recon_tmpl.c` `txtp_long_tbl`, `txtp_inv_tbl` |
| Intra type table | §9 `Md_Idx_To_Type`, `Size_Class`, `Mode_To_Angle`, `Mrl_Index_To_Delta`, §5.20.8.2 `wide_angle_mapping` | `recon_tmpl.c` `md_idx2type` |
| Lossless transform size | §5.20.6.1 `read_tx_size`: `Lossless` forces `TX_4X4` for intra | `recon_tmpl.c`, `decode.c` `read_tx_size` |
| Lossless 2D inverse | §7.15.4: row/column inverse WHT, `PlaneTxType == IDTX` short path | `itx_tmpl.c` `inv_txfm_add_wht_wht_4x4_c` |
| Lossless dequantization | §7.14.4: `get_q(0, delta<=0) = 64`, `QUANT_TABLE_BITS = 3` → `Round2(|q|*64, 3) = 8*|q|` | `itx_tmpl.c` dequant |
| DPCM syntax | §5.20.5.5 (`use_dpcm_y`, `dpcm_mode_y`), §5.20.5.6 (`use_dpcm_uv`, `dpcm_mode_uv`) | `decode.c` DPCM branch |
| DPCM reconstruction | §7.15.4 cumulative sum over `Residual` (`V_PRED` vertical, else horizontal) | `itx_tmpl.c` `residual_add` cases 1/2 |
| Lossless intra prediction per transform block | §5.20.7.24 `make_intra_prediction` (`x > 0`, `y > 0`, `allowCorners`) | `recon_tmpl.c` per-4x4 `prepare_intra_edges` |
| `EobU` for the V plane | §5.20.7.27 sets `EobU` when the U plane is parsed | `recon_tmpl.c` `if (!pl) t->u_has_cf = eob >= 0` |
| `lossless_tx_size` | §5.20.6.1 (inter, `MiSize != 4x4`, not skipped) | `decode.c` `read_tx_part` |
| `txtp_lossless` | §5.20.8.2 (4x4 inter luma: `WHT_WHT` vs `IDTX`; larger infers `IDTX`) | `recon_tmpl.c` `decode_cf` |
| Lossless `IDTX` residual | §7.15.4 `Residual = Dequant >> (3 - shift)` | `itx_tmpl.c` |
| Lossless chroma transform size | §5.20.7.23 (`TX_4X4` at lossless) | `recon_tmpl.c` `read_coef_blocks` |
| Transform partition syntax | §5.20.6.3 `tx_do_partition`, `tx_partition_type`, `tx_2or3_partition_type` | `decode.c` `read_tx_part` |

Two details resolve in favour of the oracle because the acceptance criterion is
byte-exactness with `dav2d`:

1. **DPCM `IntraJointMode`.** §5.20.5.5 writes `50`/`22` (the `modeIdx` space);
   the neighbour context stores the joint index itself, so the decoder returns
   `45` for `H_PRED` and `17` for `V_PRED`.
2. **`EobU` scope.** With the plane-outer chroma loop the V plane sees the
   `eob` of the *last* U transform block, not the U block at the same position.

## Root causes fixed

1. **Decoder rejected intra transform types, IDTX, and lossless.** `derive_txtp`
   returned `Unsupported` for every intra type except `DCT_DCT`, for inter
   `IDTX`, and for any lossless block other than a 4x4 luma WHT. It now decodes
   the complete `transform_type` syntax and derives all kernels.
2. **Decoder had no lossless path.** Lossless frames were decoded as one
   transform per coding block. The decoder now runs the §5.20.7.24 per-4x4
   transform loop with the 4x4 WHT, per-transform-block prediction, DPCM, and
   the `EobU` rule.
3. **Lossless dequantization was missing the `8 * q` scale.** `decode_coefs`
   now applies `LOSSLESS_DEQUANT_SCALE` (`8`) to every lossless level before the
   inverse WHT.
4. **Encoder IDTX V-plane mismatch.** Inter chroma inherits the luma kernel
   (`av2_get_tx_type`, `plane != 0`, `reduced_tx_set` 0), but the encoder used
   `chosen_txtp` even when the luma block quantized to all zero; `dav2d` then
   infers `DCT_DCT`. The encoder now uses `DCT_DCT` for an all-zero luma block
   and applies the chroma restrictions (`ADST` on a 32-sample-or-longer side,
   16x16 mixed identity/ADST) to the inherited type. No plane or QP special
   case is involved.
5. **Encoder non-DPCM lossless ±1 on frame 2.** The per-4x4 prediction used a
   block-level top-right/bottom-left heuristic. The encoder now mirrors the
   reference decoder's `n_tr` / `n_bl`: it counts the already-reconstructed 4x4
   units in the row above / column to the left, with the superblock and tile
   boundary rules. This is tracked by a per-plane `is_coded` map.
6. **Full inter transform sets were mis-mapped.** `TX_SET_INTER_2` uses
   `inter_tx_type_set2` as its selector and `index/offset_set2`; it is not
   `TX_SET_INTER_1` with a different square index. Corrected, which fixed the
   reduced-set-0 desync at 16x16 and 1080p.

## What was implemented

- `dec/syntax.rs`
  - `derive_txtp` implements `compute_tx_type` for luma intra, luma inter,
    chroma (intra LUT and inter inheritance), and lossless:
    - long 64/32-side sets (`is_long_side_dct` + short 1D index);
    - `TX_SET_INTRA_1` (`intra_tx_type_set1`, 7-way `Md_Idx_To_Type`) and the
      reduced `TX_SET_INTRA_2` (2-way);
    - `TX_SET_DCT_IDTX`, `TX_SET_DCT_IDTX_IDDCT`, `TX_SET_INTER_1`,
      `TX_SET_INTER_2` with the normative inverse tables;
    - `wide_angle_mapping` with the mode substitution and
      `Mrl_Index_To_Delta`;
    - `IDTX_INV` → `IDTX`, the 16x16 mixed-type restriction, and the
      32-sample ADST restriction for chroma.
  - `decode_intra_y_mode` / `decode_intra_uv_mode` decode `use_dpcm_y` /
    `use_dpcm_uv` and `dpcm_mode_y` / `dpcm_mode_uv` instead of rejecting them.
  - `decode_coefs` applies the §7.14.4 lossless dequantization.
- `dec/frame.rs`
  - `decode_lossless_intra_block`: per-4x4 transform loop for luma, U, and V;
    per-transform-block `make_intra_prediction`; WHT levels; DPCM cumulative-sum
    reconstruction through `inv_txfm_add`; `EobU` from the last U block;
    per-4x4 coefficient contexts; 4x4 deblock grid entries.
  - `lossless_tu_edges`: `n_tr` / `n_bl` derivation matching the encoder and
    `dav2d`.
  - Chroma coefficient decode passes the luma kernel (`y_txtp`) to the inter
    chroma derivation.
  - The intra transform tree symbol is not read in lossless frames
    (`TX_MODE_ONLY_4X4`).
- `av2/frame.rs` and `av2/transform.rs`: the encoder fixes described above;
  `IDTX` is now selectable (it was excluded from the inter candidate lists).
  `select_best_tx_type` keeps `DCT_DCT` as the tie-break default.

Update 2026-11-11 (inter lossless, partitions, neighbour context):

- `dec/frame.rs`
  - `decode_lossless_inter_block` logic inside `decode_inter_block`: the luma
    transform units are the 4x4 units of `lossless_tx_size == 0` (or the single
    full-block `IDTX` of `lossless_tx_size == 1`), decoded per unit with their
    own coefficient contexts; chroma is always coded as 4x4 units whose kernel
    is inherited from the co-located luma unit (§5.20.7.23, `txtp_map`).
  - The neighbour context bookkeeping no longer writes one `res_ctx` over the
    whole block footprint when the block has more than one transform unit.
    That write clobbered the per-unit contexts of lossless blocks and made the
    encoder and decoder disagree at the first transform unit of the following
    block.
  - `decode_tx_partition` reads the full transform-tree syntax: the
    `tx_do_partition` flag, the 6-way `tx_partition_type` symbol when the block
    can split in both directions, `tx_2or3_partition_type` for 4-sample-thin
    blocks, and the implied `HORZ`/`VERT` of `4x8`/`8x4`. `SPLIT`, `HORZ` and
    `VERT` are reconstructed per sub-unit; the 4-way and 5-way variants and
    every intra partition are rejected with typed errors.
- `dec/syntax.rs`
  - `derive_txtp` decodes `txtp_lossless` for 4x4 inter luma blocks, infers
    `IDTX` above 4x4, and applies the lossless chroma inheritance
    (`IDTX_INV -> IDTX`).
  - `decode_coefs` keeps the §7.14.4 `8 * q` lossless dequantization.
- `av2/transform.rs`: `inv_txfm_add_lossless_idtx` implements the §7.15.4
  lossless `IDTX` pass-through (`Dequant >> (3 - shift)`), used by the inter
  luma and chroma reconstruction.

## Verification

Environment: Linux, one thread, `dav2d` revision `bff8e5c`
(`$ORACLE/dav2d`), `--threads 1 --framedelay 1 --strict 1
-q`. The encoder's own reconstruction is compared with `dav2d` on the frame
payload bytes; our decoder is compared with both. The encoder command is the
one in [`UNIT9_ENCODER.md`](UNIT9_ENCODER.md) (the last three positional
arguments are `tx_type_rdo`, `reduced_txtp_set`, `dpcm`).

### Lossless WHT + DPCM (quantizer index 0, 3 frames)

| Resolution | Layout | Bit depth | DPCM | Our decoder == encoder rec | == dav2d |
| :-- | :-- | --: | :-- | :-- | :-- |
| 64x64 | I420 | 8 | off | exact | exact |
| 64x64 | I420 | 8 | on | exact | exact |
| 256x192 | I420 | 8 | off | exact | exact |
| 256x192 | I420 | 8 | on | exact | exact |
| 256x192 | I422 | 8 | off | exact | exact |
| 256x192 | I444 | 8 | on | exact | exact |
| 256x192 | I420 | 10 | off | exact | exact |
| 256x192 | I420 | 10 | on | exact | exact |

The previous non-DPCM frame-2 defect (22 luma samples, max 1) is gone; all three
frames of the 256x192 non-DPCM case are exact. Every lossless frame also equals
the source sample by sample, which the in-tree test asserts.

### Transform sets (lossy, 3 frames, 8-bit I420)

| Case | Reduced sets | QPs | Result |
| :-- | :-- | :-- | :-- |
| 64x64 synthetic | 0, 1, 2, 3 | 20, 100, 140 | **12/12 exact** |
| 256x192 synthetic | 0, 1, 2, 3 | 20, 100, 140 | **12/12 exact** |
| 256x192 I422 / I444 | 0 | 100 | **2/2 exact** |
| 256x192 10-bit | 0, 1, 2, 3 | 100 | **4/4 exact** |
| 1920x1080 `testsrc2` | 0 | 20, 100, 140 | **3/3 exact** |
| 1920x1080 `testsrc2` | 2 (IDTX) | 20, 140 | **2/2 exact** |

`IDTX` selection at 1920x1080 QP 140, reduced set 2: 1 non-DCT inter luma block
in frame 1 and 9 in frame 2 (`tx non-DCT` counter; reduced set 2 codes only
`DCT_DCT`/`IDTX`, so these are identity kernels). This is the exact case that
previously differed from `dav2d` in the V plane (64/59 samples, max 44); it is
now exact for our decoder and for the encoder reconstruction.

### Inter lossless and transform partitions (3 frames, byte-exact vs `dav2d`)

| Case | QP | Bit depth | Result |
| :-- | :-- | --: | :-- |
| 64x64 I420 | 0 | 8 | **exact** |
| 256x192 I420 | 0 | 8 | **exact** |
| 256x192 I420 | 0 | 10 | **exact** |
| 64x64 / 256x192, partitions on | 0, 20, 100, 140 | 8 | **8/8 exact** |
| 256x192, partitions on | 100 | 10 | **exact** |

The QP 0 rows code inter frames (`keyint = 60`) with lossless residuals; the
partition rows assert `tx_partition_counts[SPLIT/HORZ/VERT] > 0`. The decoder
output equals the encoder reconstruction and `dav2d` on every frame.

### In-tree tests

```bash
cargo test -p rustmpeg-video --lib test_decoder_unit9_lossless_and_transform_sets
# test result: ok. 1 passed
cargo test -p rustmpeg-video --lib test_encoder_transform_partitions_round_trip
# test result: ok. 1 passed
cargo test -p rustmpeg-video --lib av2_unit9_decoder_dav2d_byte_exact -- --ignored
# test result: ok. 1 passed
cargo test -p rustmpeg-video --lib av2_unit9_encoder_dav2d_byte_exact -- --ignored
# test result: ok. 1 passed
cargo test -p rustmpeg-video --lib av2_unit9_encoder_dav2d_tx_partition -- --ignored
# test result: ok. 1 passed
```

The non-ignored test covers lossless with and without DPCM (including frame 2),
8-bit and 10-bit, reduced sets 0–3 at QP 20/100/140 on 64x64 and 256x192, and
asserts the WHT/DPCM/non-DCT counters. The ignored test repeats the matrix
against `dav2d`.

### Baseline regression

```bash
python3 $ORACLE/check_av2.py quick   # 9/9 OK
python3 $ORACLE/check_av2.py full    # 84/84 OK, exit 0
```

## Toolchain gates

```text
cargo fmt --all -- --check                             -> rc 0
cargo clippy --workspace --all-targets -- -D warnings  -> rc 0
cargo test --workspace                                 -> 800 passed, 0 failed, 13 ignored
cargo test -p rustmpeg-video --lib av2                 -> 159 passed, 0 failed, 12 ignored
cargo test -p rustmpeg-video --lib av2_ -- --ignored   -> 13 passed, 0 failed
python3 scripts/check-licenses.py                      -> license audit: OK
python3 $ORACLE/check_av2.py quick                     -> 9/9 OK, exit 0
python3 $ORACLE/check_av2.py full                      -> 84/84 OK, exit 0
```

The ignored tests are the external-oracle tests that need `dav2d`; the two
unit-9 tests were run for this report and passed.

## Remaining gaps

- **Intra transform partitions are rejected.** Per-transform-block intra
  prediction is not implemented, so `TX_PARTITION_SPLIT`/`HORZ`/`VERT` on
  intra blocks returns a typed error. Inter partitions are decoded.
- **4-way and 5-way partitions are rejected.** `HORZ4`/`VERT4`/`HORZ5`/
  `VERT5` return a typed error; only `SPLIT`/`HORZ`/`VERT` are reconstructed.
- **Inter lossless blocks whose `lossless_tx_size == 1` needs more than one
  32x32 transform** (blocks larger than 32x32) are rejected with a typed
  error; the encoder never produces them.
- **Lossless palette / DIP / CfL are rejected** with typed errors.
- **The long transform sets** (`TX_SET_WIDE_*` / `HIGH_*`) are decoded, but the
  encoder only produces blocks whose base transform matches the block, so the
  long paths are exercised by the decoder's unit tests and the reference
  tables, not by an encoder stream.
- **`TX_SET_INTER_1` / `INTER_2`** are reachable only for 16x16-or-smaller
  inter blocks; they are verified at 16x16 and at 1080p reduced set 0.
- The decoder's full inter-set symbol order follows `dav2d`; `TX_SET_INTER_2`
  has one inverse-table entry (`FLIPADST_ADST`, index 11) that the encoder's
  `symbol2 = idx % 8` cannot express. It is not selected by the encoder.
- Inter blocks are not entered into the per-plane `is_coded` map used for the
  lossless per-transform-block edge derivation; the encoder matches that
  behaviour, so streams agree, but the map is only complete for intra blocks.

