# AV2 release unit 9 — encoder/shared half (transform types, lossless WHT, DPCM)

Date: 2026-10-29, updated 2026-11-11
Scope: AV2 v1.0.0 normative transform-type signalling (`get_tx_set`,
`transform_type`, reduced sets 0–3), the lossless 4x4 WHT path at quantizer
index 0 (intra **and** inter, `lossless_tx_size`, `txtp_lossless`), DPCM intra
prediction (`use_dpcm_y` / `use_dpcm_uv`), and the transform-partition
signalling and RDO (`TX_PARTITION_SPLIT`/`HORZ`/`VERT`). External verification
is byte-exact against `dav2d`.

## Normative findings

| Item | AV2 v1.0.0 | Reference decoder (read-only) |
| :-- | :-- | :-- |
| Transform set derivation | §5.20.8.3 `get_tx_set` | `recon_tmpl.c` `compute_tx_type` |
| Transform-type syntax | §5.20.8.2 `transform_type` | `recon_tmpl.c` `decode_cf` |
| Inverse-set tables | §5.20.8.2 `Tx_Type_Inter_Inv_Set1/2/3/4`, `Tx_Type_Inv_Long` | `recon_tmpl.c` `txtp_long_tbl` |
| Intra type table | §9 `Md_Idx_To_Type`, `Size_Class`, `Mode_To_Angle`, `Mrl_Index_To_Delta`, §5.20.8.2 `wide_angle_mapping` | `recon_tmpl.c`, `decode.c` `custom_mode_list_y` |
| Transform-type CDF selection | §8 Table 8.2 (intra) and Table 8.3 (inter) | `ts->cdf.m.intra_tx_type*`, `inter_tx_type*` |
| Lossless transform size | §5.20.6.1 `read_tx_size`: `Lossless` forces `TX_4X4` for intra; `TX_MODE_ONLY_4X4` | `recon_tmpl.c`, `decode.c` `read_tx_size` |
| Lossless 2D inverse | §7.15.4: row/column inverse WHT (`shift` 3 then 0), `PlaneTxType == IDTX` short path | `itx_tmpl.c` `inv_txfm_add_wht_wht_4x4_c` |
| Lossless dequantization | §7.14.4: `get_q(0, delta<=0) = 64`, `QUANT_TABLE_BITS = 3` → `Round2(|q|*64, 3) = 8*|q|` | `itx_tmpl.c` dequant |
| DPCM syntax | §5.20.5.5 (`use_dpcm_y`, `dpcm_mode_y`), §5.20.5.6 (`use_dpcm_uv`, `dpcm_mode_uv`) | `decode.c` DPCM branch |
| DPCM reconstruction | §7.15.4 cumulative sum over `Residual` (`V_PRED` vertical, else horizontal) | `itx_tmpl.c` `residual_add` cases 1/2 |
| Lossless intra prediction per transform block | §5.20.7.24 `make_intra_prediction` (`x > 0`, `y > 0`, `allowCorners`) | `decode.c` / `recon_tmpl.c` |
| Transform-partition flag presence | §5.20.6.1/§5.20.6.3 (`MiSize > 4x4`, `TxMode == TX_MODE_SELECT`, block ≤ 64x64) | `decode.c` `read_tx_partition` |
| Partition type symbol | §5.20.6.3 `tx_partition_type` (6-way, `txPartition = symbol + 1`); `tx_2or3_partition_type` for 4-sample-thin blocks | `decode.c` `read_tx_part` (`size_to_tx_type_group_vh_lookup`) |
| `lossless_tx_size` | §5.20.6.1: coded when `MiSize != 4x4`, inter (or `fsc`), and not skipped; `1` selects `find_tx_size(Min(32,BW), Min(32,BH))` | `recon_tmpl.c` `read_tx_part` (`tx_size_ll`) |
| `txtp_lossless` | §5.20.8.2: 4x4 inter luma at lossless codes `WHT_WHT` vs `IDTX`; larger inter transforms infer `IDTX` | `recon_tmpl.c` `decode_cf` |
| Lossless `IDTX` reconstruction | §7.15.4: `Residual = Dequant >> (3 - shift)`, `shift = (pels > 256) + (pels > 1024)` | `itx_tmpl.c` |
| Lossless chroma transform size | §5.20.7.23: `txSz = Lossless ? TX_4X4 : ...` (chroma is always 4x4 at lossless) | `recon_tmpl.c` `read_coef_blocks` |
| QP 0 is lossless for every frame type | §5.18.1 `LosslessArray` per segment | `read_frame_header` |

Two details differ between the specification text and the reference decoder and
are resolved in favour of the oracle, because the acceptance criterion is
byte-exactness with `dav2d`:

1. **DPCM `IntraJointMode`.** §5.20.5.5 writes `50` for `H_PRED` and `22` for
   `V_PRED`, which is the `modeIdx` space (`5 + joint index`). The context
   arrays store the joint index itself, so the encoder stores and returns `45`
   and `17` (`6*7 + 3` and `2*7 + 3`, matching `dav2d` `decode.c:1995/1998`).
2. **`EobU` for the V-plane `all_zero` context.** §5.20.7.27 sets `EobU` when
   the U plane is parsed; with the plane-outer chroma loop of §5.20.7.23 the
   value seen by the V plane is the `eob` of the *last* U transform block
   (`dav2d` `recon_tmpl.c:3649`), not the U block at the same position.

## Root causes found in the existing encoder

1. **Lossless streams signalled the wrong transform size.** At quantizer index
   0 the frame header correctly selects `TX_MODE_ONLY_4X4`, but the encoder
   still coded one transform per coding block (16x16/32x32) with the WHT kernel
   and no `lossless_tx_size` symbol. `dav2d` therefore decoded 4x4 transforms
   and lost bitstream sync (it exited with `rc 139`).
2. **Lossless WHT level scaling.** The forward WHT scaled its output by `<< 3`
   and the lossless dequantization passed the levels through unchanged, which
   is self-consistent but 8x away from §7.14.4 (`Dequant = 8 * Quant`). The
   coded levels were 8x too large for any conformant decoder.
3. **DPCM residual not reconstructed.** The encoder fed the differenced
   residual to the inverse WHT with `dpcm_flag = 0`, so its own reconstruction
   was wrong by the cumulative sum (PSNR 11 dB at QP 0). The DPCM flag must be
   packed into the `txtp` high bits (`1` = `H_PRED`, `2` = `V_PRED`), exactly
   as `residual_add` / `dav2d` expect.
4. **Chroma plane size at lossless.** The lossless chroma residual loop used
   the 4x4 transform size as the plane size, so 4:2:0 blocks only coded a 4x4
   chroma region instead of 8x8.
5. **Corner availability for 4x4 transform blocks.** `count_top_right_avail`
   (§5.20.7.24) is available when the top-right samples are inside the current
   coding block (`x4 + 2 <= pw4`, i.e. the row above was already reconstructed)
   or, for the first transform row, from the block-level edge availability.
   The first version used the block-level value for every transform row, which
   changed Paeth/directional predictions.
6. **Chroma `all_zero` context.** `get_skip_ctx` derived `bw * bh > w * h`
   (§5.20.7.27) from a luma-size heuristic; with multiple transform blocks per
   chroma block (lossless) it has to compare the plane residual block with the
   transform block directly.
7. **Transform types were never selected.** `select_best_tx_type` was hard
   wired to `DCT_DCT` for inter and to four kernels for intra, the intra
   transform-type symbol was never coded, and `reduced_txtp_set` was fixed to
   `2` in the frame-header builder, so reduced sets 0/1/3 were unreachable.

## What was implemented

- `transform.rs`
  - Spec `TxType` enum (0..15) ↔ packed kernel id mapping, `SIZE_CLASS`,
    `Md_Idx_To_Type`, `Mode_To_Angle`, `Mrl_Index_To_Delta`, `ANGLE_STEP`,
    `Tx_Type_Inter_Inv_Set1/2/3/4`, `Tx_Type_Inv_Long`, `is_directional_mode`,
    and a spec-accurate `wide_angle_mapping` (which also substitutes the mode).
  - `TxTypeChoice` plus `intra_tx_type_choice` / `intra_tx_type_choices` and
    `inter_tx_type_choice` / `inter_tx_type_choices`, which derive the coded
    symbol(s) from a kernel and enumerate the kernels a block's set can signal.
  - `fwd_wht_wht_4x4` now produces the coded levels of §7.14.4 (no `<< 3`);
    `LOSSLESS_DEQUANT_SCALE` documents the `8 * q` dequantization.
  - `select_best_tx_type` takes the candidate list instead of a fixed kernel
    list and keeps `DCT_DCT` on ties. Candidates are pruned to the 2D kernel
    class (the encoder's quantizer and coefficient scan are 2D-only); `IDTX`
    is included and selectable.
- `entropy.rs`
  - `encode_tx_type_choice` codes `is_long_side_dct`, `intra_tx_type`,
    `inter_tx_type` and `inter_tx_type_offset` for every set
    (`TX_SET_WIDE_64/HIGH_64`, `WIDE_32/HIGH_32`, `INTRA_1`, `INTRA_2`,
    `INTER_1`, `INTER_2`, `DCT_IDTX`, `DCT_IDTX_IDDCT`).
  - `encode_coefs` codes the luma intra transform type at the normative
    position (after the EOB syntax) and honours the inferred cases
    (`fsc_mode`, `!is_inter && eob == 1`, `Lossless`, `reduced_tx_set == 2`
    intra).
  - `encode_intra_y_mode` / `encode_intra_uv_mode` code `use_dpcm_y` /
    `use_dpcm_uv` and `dpcm_mode_y` / `dpcm_mode_uv`, and skip the mode syntax
    for DPCM blocks.
  - `get_skip_ctx` implements the chroma `bw * bh > w * h` test.
- `frame.rs`
  - `encode_intra_lossless_leaf`: per-4x4 transform-block intra prediction
    (`make_intra_prediction`), WHT residual coding, lossless dequantization,
    DPCM differencing and reconstruction, 4x4 chroma transform loop with the
    `EobU` context, and the lossless context updates.
  - `lossless_prediction_costs` / `wht_l1_cost`: the DPCM decision compares the
    selected mode, `V_PRED`-DPCM and `H_PRED`-DPCM by the L1 magnitude of the
    4x4 WHT of the coded residual (the exact level cost at quantizer index 0).
  - `choose_luma_txtp`: set-driven transform-type selection for intra and
    inter, with the DC-only re-run under the inferred `DCT_DCT`.
  - The lossless path is intra-only; `mod.rs` forces key frames for a
    quantizer index of 0 (inter lossless would have to signal
    `lossless_tx_size` and code 4x4 WHT blocks, which the encoder does not
    model yet).
  - The intra transform-tree symbol is no longer written for lossless frames
    (`TX_MODE_ONLY_4X4`), and the presence conditions follow §5.20.6.
  - `FrameStats` gains `non_dct_tx_blocks`, `wht_blocks`, `dpcm_y_blocks` and
    `dpcm_uv_blocks`; `ToolConfig` gains `reduced_txtp_set` and
    `enable_dpcm`.
- `txpart.rs`: the sub-block transform RDO uses the default kernel (the actual
  transform type is chosen by the frame encoder and only `TX_PARTITION_NONE` is
  emitted).
- `examples/av2_encode.rs`: new trailing arguments `tx_type_rdo`,
  `reduced_txtp_set` (0–3) and `dpcm`, plus a per-frame counter line.

Update 2026-11-11 (inter lossless and transform partitions):

- `frame.rs`
  - `EncTxUnit` carries the geometry, kernel, EOB and quantized/dequantized
    levels of one transform unit; the inter residual is built into per-unit
    buffers before the block syntax and coded unit by unit, so the transform
    partition symbol precedes the coefficient blocks and every unit updates
    its own rate contexts.
  - `encode_inter_lossless_leaf` logic inside `encode_inter_leaf`: 4x4
    WHT/`IDTX` units with `txtp_lossless`, the optional full-block `IDTX`
    (`lossless_tx_size = 1`), chroma 4x4 units inheriting the co-located luma
    kernel, and the §7.14.4 `8 * q` dequantization.
  - The lossless inter/intra decision in `encode_leaf_block` (SATD of the
    selected intra mode against the inter predictor) so quantizer-index-0
    frames can code inter blocks.
  - `record_deblock_inter` records the per-unit transform sizes for the
    deblock/CDEF grid.
  - `FrameStats::tx_partition_counts` reports the partition decisions.
- `ratectrl.rs`: a constant quantizer of 0 is kept at 0 for every frame type.
- `txpart.rs`: `select_best_tx_partition` evaluates `NONE`/`SPLIT`/`HORZ`/
  `VERT` with the same cost; `encode_tx_partition_syntax` writes
  `tx_do_partition` plus `tx_partition_type` (6-way) or
  `tx_2or3_partition_type`, and `find_tx_size` / `tx_partition_is_signallable`
  gate the search.
- `transform.rs`: `inv_txfm_add_lossless_idtx` implements the §7.15.4
  `Residual = Dequant >> (3 - shift)` pass-through.
- `tables.rs`: `TX_PART_TYPE_GROUP_VH` (`Size_To_Tx_Type_Group_Vert_And_Horz`).
- `cdf.rs`: accessors for `tx_partition_type`, `tx_2or3_partition_type`,
  `lossless_tx_size` and `lossless_inter_tx_type`.
- `entropy.rs`: `encode_lossless_tx_size`, the `txtp_lossless` symbol, and the
  lossless chroma kernel masking.
- `mod.rs`: the quantizer-index-0 `keyint = 1` forcing is removed; the new
  external-oracle and in-tree transform-partition tests are added.

Update 2026-11-11 (inter lossless and transform partitions):

8. **Quantizer index 0 was not lossless on inter frames.** `RateControlMode::
   ConstantQp` shifted key frames by −8 and inter frames by +4, so a "QP 0"
   stream coded its inter frames at quantizer index 4 plus the GOP offset
   (observed `yac = 6`). A quantizer index of 0 is lossless for every frame
   (§5.18.1), so the offset is now bypassed when `base_qidx == 0`.
9. **Inter lossless was not modelled.** `mod.rs` forced `keyint = 1` at
   quantizer index 0. Inter blocks now code `lossless_tx_size` in the
   transform-tree position and one 4x4 WHT (or `IDTX`, `txtp_lossless`) per
   transform unit, reusing the per-transform-block residual coding of the
   intra path. Blocks up to 16x16 may instead pick the single full-block
   `IDTX` (`lossless_tx_size = 1`, §7.15.4 shift 0); larger blocks always use
   4x4 units. Chroma is coded as 4x4 units whose kernel is inherited from the
   co-located luma unit (§5.20.7.23).
10. **The decoder overwrote the per-unit coefficient contexts.** Its inter
   neighbour bookkeeping wrote one `res_ctx` over the whole block footprint
   after the coefficient loop, replacing the per-transform-unit contexts of
   lossless blocks with the last unit's value (encoder/decoder desync at the
   first TU of the next block). The footprint write now only runs for blocks
   with a single transform unit per plane.
11. **Only `TX_PARTITION_NONE` was emitted.** `select_best_tx_partition` only
   compared `NONE` against `SPLIT` behind a quadrant-uniformity early return,
   and `encode_tx_partition_flag` wrote the flag without the partition type
   symbol. The RDO now evaluates `NONE`/`SPLIT`/`HORZ`/`VERT` with the same
   Lagrangian metric, and the syntax writes `tx_partition_type`
   (6-way) for every block that can split in both directions.
12. **Inter coefficient coding with several transform units.** The encoder's
   luma/chroma residual is computed into per-unit buffers (`EncTxUnit`) before
   the block syntax, so the transform-partition symbol and the coefficient
   blocks stay in the normative order, and the per-unit rate contexts are
   updated at each unit's own position.

## Verification

Environment: Linux, one thread, `dav2d` revision `bff8e5c`
(`$ORACLE/dav2d`), run with
`--threads 1 --framedelay 1 --strict 1 -q`. The encoder's own reconstruction is
compared with the `dav2d` output on the frame payload bytes (the Y4M header
spelling `C420jpeg` vs `C420` differs; samples are identical).

Encoder invocation (all positional, the three unit-9 arguments are the last
three):

```bash
./target/release/examples/av2_encode OUT.ivf 3 W H QP 0 0 0 LAYOUT 0 0 0 0 0 0 0 \
  BIT_DEPTH 0 0 REC.y4m 0 0 0 INPUT.y4m 0 0 0 15 TX_TYPE_RDO REDUCED_TXTP_SET DPCM
dav2d -i OUT.ivf -o DAV.y4m --threads 1 --framedelay 1 --strict 1 -q
```

### Lossless WHT + DPCM (quantizer index 0)

| Resolution | Layout | Bit depth | DPCM | Frames | Result |
| :-- | :-- | --: | :-- | --: | :-- |
| 64x64 | I420 | 8 | off | 3 | **exact** |
| 64x64 | I420 | 8 | on | 3 | **exact** |
| 256x192 | I420 | 8 | on | 3 | **exact** |
| 256x192 | I422 | 8 | on | 3 | **exact** |
| 256x192 | I444 | 8 | on | 3 | **exact** |
| 256x192 | I420 | 10 | on | 3 | **exact** |
| 256x192 | I420 | 8 | off | 3 | **exact** |

Counts for 256x192 I420 8-bit with DPCM on (three frames, per frame):
`WHT 159 / 129 / 72` lossless luma blocks, `dpcm_y 142 / 26 / 30`,
`dpcm_uv 145 / 117 / 62`. Every lossless frame has `PSNR Y 100.00` (the
reconstruction equals the source), which the in-tree test also asserts sample
by sample.

### Transform types and reduced sets (lossy)

`tile256.y4m` is the 64x64 synthetic pattern tiled to 256x192 (steep local
gradients plus the 8x8 detail pattern, so the partitioner produces 16x16
blocks); it is generated outside the repository.

| Case | Reduced set | QP | Frames | Result | Non-DCT blocks |
| :-- | --: | :-- | --: | :-- | --: |
| 64x64 8-bit | 0 / 1 / 3 | 20, 100, 140 | 3 | **12/12 exact** | 2 per stream (frame 0) |
| 64x64 8-bit | 2 | 20, 100, 140 | 3 | **3/3 exact** | 0 (intra types suppressed) |
| 256x192 8-bit tile256 | 0 / 1 / 3 | 20, 100, 140 | 3 | **9/9 exact** | 24 per stream (frame 0) |
| 256x192 8-bit tile256 | 2 | 20, 100, 140 | 3 | **3/3 exact** | 0 |
| 256x192 10-bit tile256 | 0, 3 | 100 | 3 | **2/2 exact** | 24 per stream |
| 16x16 8-bit (inter sets) | 0, 2, 3 | 100 | 3 | **3/3 exact** | 0 (`TX_SET_INTER_2` / `DCT_IDTX` / `DCT_IDTX_IDDCT` still coded) |
| 1920x1080 testsrc2 8-bit | 0, 2 | 20, 100, 140 | 3 | **6/6 exact** | 195–251 (reduced 0) |

The 16x16 frames are the smallest case with 16x16 inter blocks, so they
exercise the `TX_SET_INTER_2` (reduced 0), `TX_SET_DCT_IDTX` (reduced 1/2) and
`TX_SET_DCT_IDTX_IDDCT` (reduced 3) symbols end to end. At 1080p the encoder
also produces 16x16 inter blocks, so the same symbols are exercised at scale.

The in-tree external-oracle test reproduces the lossless 4:2:0 cases (8-bit and
10-bit), the reduced-set cases at 64x64/256x192 (8-bit and one 10-bit), the
non-DCT counter assertion and the 16x16 inter-set cases; the 4:2:2/4:4:4
lossless cases and the tiled-clip counts above were run manually with the
commands in this section. It also asserts the counters:

```bash
cargo test -p rustmpeg-video --lib av2_unit9_encoder_dav2d_byte_exact -- --ignored
# test result: ok. 1 passed
```

### Inter lossless (quantizer index 0, inter frames)

With the rate-control fix a quantizer index of 0 stays lossless on inter
frames, so every frame after the first is an inter frame with a lossless
residual. The QP 0 rows of the matrix below encode three frames with
`keyint = 60` and are byte-exact against `dav2d`:

| Resolution | Layout | Bit depth | Frames | Result |
| :-- | :-- | --: | --: | :-- |
| 64x64 | I420 | 8 | 3 | **exact** |
| 256x192 | I420 | 8 | 3 | **exact** |
| 256x192 | I420 | 10 | 3 | **exact** |

Command:

```bash
DAV2D=$ORACLE/dav2d cargo test -p rustmpeg-video --lib \
  av2_unit9_encoder_dav2d_byte_exact -- --ignored
# test result: ok. 1 passed
```

The in-tree test `test_lossless_encoder_is_exact_and_selects_wht_and_dpcm`
asserts that frame 0 is a key frame, frames 1 and 2 are inter frames, the
reconstruction equals the source sample by sample on every plane, and the
counters are non-zero. It reports `wht = 17`, `dpcm_y = 14`, `dpcm_uv = 17`
and `inter = 1` for the three 64x64 frames.

### Transform partitions

`enable_tx_partition` selects between `TX_PARTITION_NONE`, `SPLIT`, `HORZ` and
`VERT` for inter luma blocks with a base transform larger than 4x4 whose
block can signal the 6-way `tx_partition_type` (min side ≥ 8 samples, block
≤ 64x64). Each candidate transforms, quantizes, reconstructs and scans every
sub-unit; the cost is `SSE + lambda * rate` with a fixed per-unit rate
estimate. `speed >= 7` searches only `NONE`/`SPLIT`, `speed >= 8` emits
`NONE`. The selected partition is written with `tx_do_partition` +
`tx_partition_type` at the normative position and the sub-unit coefficient
blocks follow in raster order; the deblocking grid records the sub-unit
transform sizes.

| Case | QP | Frames | Result |
| :-- | :-- | --: | :-- |
| 64x64 8-bit | 0, 20, 100, 140 | 3 | **4/4 exact** |
| 256x192 8-bit | 0, 20, 100, 140 | 3 | **4/4 exact** |
| 256x192 10-bit | 100 | 3 | **1/1 exact** |

Command:

```bash
DAV2D=$ORACLE/dav2d cargo test -p rustmpeg-video --lib \
  av2_unit9_encoder_dav2d_tx_partition -- --ignored
# test result: ok. 1 passed
```

The test asserts `tx_partition_counts[SPLIT] > 0`, `[HORZ] > 0` and
`[VERT] > 0` over the matrix. The in-tree
`test_encoder_transform_partitions_round_trip` checks
`encoder reconstruction == our decoder` for the same two sizes at QP
0/20/100/140 and reports `[NONE, SPLIT, HORZ, VERT] = [63, 3, 3, 6]` over 75
evaluated inter blocks.

### Baseline regression

```bash
python3 $ORACLE/check_av2.py quick   # 9/9 OK
python3 $ORACLE/check_av2.py full    # 84/84 OK, exit 0
```

The default configuration is unchanged: `enable_tx_type_rdo = false`,
`reduced_txtp_set = 2`, `enable_dpcm = false`, `enable_tx_partition = false`.
The lossless path only affects quantizer index 0, and the transform-partition
RDO is opt-in.

### RD and speed impact

1920x1080 `testsrc2`, 3 frames, 1 thread, release build, `dav2d`-exact in every
case. "RDO off" is `reduced_txtp_set = 2` with transform-type RDO disabled;
"RDO on" is reduced set 0 with the RDO enabled.

| QP | RDO off bytes / PSNR Y / time | RDO on bytes / PSNR Y / time | Δbytes | ΔPSNR | Δtime |
| --: | :-- | :-- | --: | --: | --: |
| 20 | 591 777 / 38.13 / 211 ms | 591 678 / 38.15 / 229–250 ms | −0.02% | +0.02 dB | +8…18% |
| 100 | 185 218 / 37.87 / 186–190 ms | 185 116 / 37.86 / 189–197 ms | −0.06% | −0.01 dB | +0…4% |
| 140 | 59 124 / 36.70 / 155–161 ms | 59 094 / 36.72 / 162–168 ms | −0.05% | +0.02 dB | +4…5% |

At 64x64 (3 frames, synthetic source) the effect is larger in relative terms.
With the RDO off (reduced set 2) and on (reduced set 0): QP 100 1287 → 1241
bytes (−3.6%) at PSNR Y 37.46 → 37.52 (+0.06 dB); QP 140 591 → 576 bytes
(−2.5%) at 32.19 → 32.30 (+0.11 dB); QP 20 3641 → 3625 bytes (−0.4%) at
46.66 → 45.46 dB (−1.2 dB, the compaction heuristic trades distortion for
rate at the low-QP end). The selection cost is dominated by
the candidate count: the 2D kernels of the block's set (up to 7 for
`TX_SET_INTRA_1`) are each forward transformed once per candidate block, and
the evaluation is skipped for transform blocks larger than 16x16 and at
`speed >= 7`.

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
cargo build --release -p rustmpeg-video --examples     -> rc 0
```

New unit tests:

- `av2::transform::tests::test_tx_type_round_trips_through_the_packed_representation`
- `av2::transform::tests::test_intra_tx_type_choices_follow_md_idx_to_type`
- `av2::transform::tests::test_wide_angle_mapping_changes_the_mode`
- `av2::transform::tests::test_inter_tx_type_choices_match_the_inverse_sets`
- `av2::transform::tests::test_fwd_inv_idtx_2d_roundtrip`
- `av2::transform::tests::test_fwd_inv_wht_lossless_roundtrip` (updated for the
  §7.14.4 dequantization scale)
- `av2::tests::test_lossless_encoder_is_exact_and_selects_wht_and_dpcm`
- `av2::tests::test_encoder_selects_non_dct_transform_types`
- `av2::tests::av2_unit9_encoder_dav2d_byte_exact` (external oracle, ignored)

## Deliberate limits and remaining risks

- **Inter lossless is complete for square and rectangular inter blocks.**
  `lossless_tx_size = 1` is only chosen for blocks up to 16x16 (256 samples,
  where the §7.15.4 `IDTX` shift is 0); larger blocks code 4x4 WHT units.
  Streams whose `lossless_tx_size = 1` block needs more than one 32x32
  transform are rejected by the decoder with a typed error, and this encoder
  never produces them.
- **Intra transform partitions are not emitted.** Intra blocks stay
  `TX_PARTITION_NONE`; per-transform-block intra prediction is not modelled.
  The intra partition syntax is parsed so the decoder rejects it precisely.
- **4-way and 5-way partitions are not emitted or decoded.** Only
  `SPLIT`/`HORZ`/`VERT` are selected; `HORZ4`/`VERT4`/`HORZ5`/`VERT5` are
  rejected with a typed error.
- **`IDTX` is selectable and byte-exact.** The forward identity path is
  self-consistent with the normative inverse; the earlier V-plane difference
  with `IDTX` enabled was an inter-chroma bug (the encoder used the luma kernel
  even when the luma block quantized to all zero, while the reference infers
  `DCT_DCT`). Fixed in `frame.rs`.
- **Non-DPCM lossless is byte-exact,** including frame 2; the per-transform
  block top-right/bottom-left availability follows the reference decoder's
  `n_tr`/`n_bl` derivation over a per-plane `is_coded` map. Inter blocks are
  not entered into that map, which matches the decoder's behaviour for the
  encoder's subset.
- **Lossless palette / DIP / CfL are not selected** and are rejected by the
  decoder with typed errors; their per-transform-block reconstruction is not
  implemented.
- **The long transform sets** (`TX_SET_WIDE_*` / `HIGH_*`) are implemented in
  the symbol encoder, but the encoder only produces square-or-rectangular
  coding blocks whose base transform matches the block, so those branches are
  exercised only by the decoder half.
- **`TX_SET_INTER_1` / `INTER_2`** are only reachable for inter blocks of 16x16
  or smaller; the current partitioner emits those only for some content
  (16x16 frames, 1080p), which is why the inter symbols are verified with the
  16x16 and 1080p cases.

Update 2026-10-29 (decoder half): the decoder implements every transform set,
lossless WHT/DPCM, and the `EobU` rule; `IDTX` is selectable and both encoder
mismatches above are fixed. `cargo test -p rustmpeg-video --lib
av2_unit9_decoder_dav2d_byte_exact -- --ignored` passes.

Update 2026-11-11: inter lossless, `lossless_tx_size`, `txtp_lossless`, the
lossless `IDTX` pass-through and the `SPLIT`/`HORZ`/`VERT` transform partitions
are implemented on both sides and verified against `dav2d`. See
[`UNIT9_DECODER.md`](UNIT9_DECODER.md) for the decoder half and the neighbour
context fix.

No `unsafe`, no new dependency, no native build step, and no third-party code
was added.
