# AV2 release unit 5 — encoder/shared-syntax half

Date: 2026-10-29
Scope: AV2 v1.0.0 segmentation syntax and encoder semantics, spatial segment-map
prediction, segment-map temporal coding support, per-segment `SEG_LVL_ALT_Q`,
per-superblock delta-Q, variance adaptive quantization (VAQ), and rate-control
integration.

This half owns the encoder and the shared syntax definitions. The decoder
(`crates/rustmpeg-video/src/av2/dec/*`) is agent 2's half; our decoder still
rejects segmentation in `validate_tools`, so decoder-vs-encoder byte equality is
verified against **dav2d** here, not against our decoder.

## Normative sources and reference symbols

| Item | AV2 spec | dav2d reference (read-only) |
| :-- | :-- | :-- |
| Segment feature enable/value | §5.4.9 `seg_info(numSegments)` | `parse_seg_info`, `src/obu.c` |
| Sequence segment config | §5.4.4 `sequence_segment_config()` | `parse_seq_hdr`, `src/obu.c` |
| Frame segmentation enable, map update, temporal update, `LastActiveSegId`, `SegIdPreSkip` | §5.18.7.1 `segmentation_params()` | `parse_frame_hdr` segmentation branch, `src/obu.c` |
| Spatial segment prediction and context | §5.20.5.8 `read_segment_id()` | `get_cur_frame_segid`, `src/env.h` |
| Temporal prediction from the primary reference | §5.20.7.2 / §5.20.7.4 | `get_prev_frame_segid`, `src/decode.c` |
| `neg_deinterleave` | §5.20.5.8 | `neg_deinterleave`, `src/decode.c` |
| Segment id symbol CDFs | §8.3.7 (`TileSegmentIdCdf`, `TileSegmentIdExtCdf`, `TileSegIdExtFlagCdf`, `TileSegmentIdPredictedCdf`) | `Default_Segment_Id_Cdf` etc., `src/cdf.c` |
| Delta-Q syntax | §5.18.7.8 `delta_q_params()`, §5.20.5.11 `read_delta_qindex()` | delta-q block, `src/decode.c` |
| Segment ALT_Q composition with `CurrentQIndex` | §5.18.1 `get_qindex()` | `init_quant_tables`, `src/decode.c` |

AV2 v1.0.0 has exactly three segment features: `SEG_LVL_ALT_Q`,
`SEG_LVL_SKIP`, `SEG_LVL_GLOBALMV` (§5.18.7.1 constant table). AV1's
`SEG_LVL_ALT_LF` and `SEG_LVL_REF_FRAME` features do **not** exist in AV2
v1.0.0, so "per-segment delta-LF" and "per-segment reference frame" are not
implemented as segment features. This is a spec fact, not an omission. Segment
map prediction *from* a reference frame is implemented as
`load_previous_segment_ids` (`get_prev_frame_segid`).

## What was implemented

- `headers.rs`: AV2 segment feature indices, `MAXQ` per bit depth, and
  `SegmentationDataSet` predicates (`feature_active`, `preskip`,
  `last_active_segid`, `max_segments`).
- New `segment.rs`:
  - `SegmentMap` at 4x4 `MI_SIZE` granularity with the exact
    `read_segment_id()` / `get_cur_frame_segid()` prediction and 3-way context,
    and `get_segment_id()` minimum-over-footprint for reference maps.
  - `neg_deinterleave()` and its exact inverse (`neg_interleave()`) used by the
    encoder to turn a desired segment id into the coded distance.
  - `encode_segment_id()` covering explicit ids, extension ids (`8..16`),
    temporal prediction, and the skipped non-lossless inference rule.
  - `encode_delta_q()` for `delta_q_abs` including the `DELTA_Q_SMALL` escape.
  - `VaqPlan`: 16x16 log-variance analysis, 8 ALT_Q classes, and per-superblock
    QP offsets.
- `frame.rs`:
  - Per-frame segmentation setup (`enabled`, `update_map`, `update_data`,
    `temporal`), per-segment lossless derivation, and `all_lossless` /
    `any_lossless` from the AV2 segment rules.
  - Segment ids coded at the normative positions: before `is_inter` when
    `SegIdPreSkip`, after `skip_txfm` otherwise.
  - Per-superblock delta-Q applied to `CurrentQIndex` and emitted after
    `read_ccso`, before the block mode.
  - Per-superblock dequantizer tables from `get_qindex()` semantics, so the
    encoder reconstruction uses the same quantizer as the decoder.
  - Effective average quantizer (`FrameStats::avg_qidx`) reported to the rate
    controller.
- `ratectrl.rs`: bounded, dead-banded logarithmic rate control with a leaky
  bucket and a per-frame bits budget; `ConstantQp` behaviour is unchanged.
  `frame_budget_bits()` is consumed by the per-superblock delta-Q feedback.
- `examples/av2_encode.rs`: `bit_depth`, `vaq`, `delta_q`, reconstruction-Y4M,
  and VBR CLI arguments.

## Acceptance evidence

Environment: Linux, 1 thread, dav2d rev `bff8e5c` at
`$ORACLE/dav2d`.

Acceptance command shape:

```bash
cargo build --release -p rustmpeg-video --examples
./target/release/examples/av2_encode OUT.ivf 3 W H QP 0 0 0 420 0 0 0 0 0 0 0 BD 1 1 REC.y4m
dav2d -i OUT.ivf -o DAV.y4m --threads 1 --framedelay 1 --strict 1 -q
python3 $WORK/cmpy4m.py REC.y4m DAV.y4m
```

`cmpy4m.py` compares the FRAME payload bytes only (da2d writes `C420`, the
encoder's writer writes `C420jpeg`; the sample bytes are identical).

Result: **18/18 cases byte-exact, encoder rc=0, dav2d rc=0**:

| Bit depth | QP 20 | QP 100 | QP 140 |
| :-- | :-- | :-- | :-- |
| 8-bit 64x64 | OK | OK | OK |
| 8-bit 256x192 | OK | OK | OK |
| 10-bit 64x64 | OK | OK | OK |
| 10-bit 256x192 | OK | OK | OK |
| 12-bit 64x64 | OK | OK | OK |
| 12-bit 256x192 | OK | OK | OK |

All cases are 3 frames (key + 2 inter) with `enable_vaq=1`,
`enable_delta_q=1`. Additional runs:

- 10 and 30 frames at 320x192, QP 100, 8-bit: byte-exact.
- 3 frames at 1920x1080, QP 100, 8-bit: byte-exact.
- Default (segmentation off) full matrix: 84/84 `OK` with
  `python3 $ORACLE/check_av2.py` (all sizes × QP 20/60/100/140/180/220
  × 1/3 frames).
- Quick matrix `python3 $ORACLE/check_av2.py quick`: 9/9 `OK`.

The in-tree external byte-exact test (ignored by default because it needs the
oracle binary) covers 64x64 and 256x192 at 8/10/12-bit:

```bash
cargo test -p rustmpeg-video av2_segmentation_dav2d -- --ignored
# test result: ok. 1 passed; 0 failed; 0 ignored
```

Toolchain checks:

```text
cargo fmt --all -- --check                                  -> rc 0
cargo clippy --workspace --all-targets -- -D warnings       -> rc 0
cargo test -p rustmpeg-video av2                            -> 105 passed, 0 failed, 1 ignored
cargo test --workspace                                      -> all suites pass (265 passed in the av2 lib suite, 1 ignored)
python3 scripts/check-licenses.py                           -> license audit: OK (all packages permissive, no copyleft)
```

## Rate control

`RateController` now has a frame budget (`frame_budget_bits`) and applies a
dead-banded `log2(actual / target)` correction, clamped to ±4 quantizer steps
per frame, with a leaky-bucket buffer correction. `ConstantQp` is untouched, so
constant-QP streams do not change. Unit tests:

- `constant_qp_is_unchanged`
- `vbr_quantizer_is_stable_around_the_target` (alternating ±35% frame sizes:
  quantizer span ≤ 12, tail within ±8 of its mean)
- `cbr_buffer_feedback_is_bounded` (60 kB frames at a 400 kbps target: ≤ 12
  steps per frame)
- `frame_budget_tracks_target_and_keyframes`

The per-superblock delta-Q also uses the frame budget: after each superblock the
actual emitted bits are compared with `budget / number_of_superblocks`, and the
activity offset is nudged by at most one step (bounded ±4).

Measured on 10 synthetic 320x192 frames, 8-bit, one thread:

| QP | VAQ/delta-Q | Total bytes | Mean Y PSNR |
| :-- | :-- | --: | --: |
| 80 | off | 41,656 | 39.48 dB |
| 80 | on | 43,236 | 39.54 dB |
| 100 | off | 27,842 | 37.86 dB |
| 100 | on | 28,278 | 37.78 dB |
| 120 | off | 17,509 | 35.96 dB |
| 120 | on | 18,247 | 36.00 dB |
| 140 | off | 10,545 | 33.71 dB |
| 140 | on | 10,806 | 33.59 dB |

This clip does not show a reliable RD gain from VAQ: the segment-map and delta-Q
side information costs roughly 1–4% and the PSNR change is within noise. The
tool therefore stays opt-in and default-off, and this is recorded as a remaining
risk rather than claimed as a win. The syntax and reconstruction conformance are
independent of that RD result.

## Deliberate limits and remaining risks

- Our decoder does not accept segmentation yet (`dec/frame.rs` still returns
  `Unsupported("segmentation")`). The encoder streams above are verified with
  dav2d; encoder/decoder parity is agent 2's half.
- `segmentation_temporal_update` is implemented and unit-tested, but the
  encoder keeps it off by default (`ToolConfig::enable_seg_temporal`). With the
  current linear GOP the primary reference is the immediately preceding frame;
  a decoder that parses headers one frame ahead (dav2d `--framedelay 1`) has not
  reconstructed that reference yet, derives `PRIMARY_REF_NONE`, and would not
  load the previous segment map. Enabling temporal prediction requires a GOP
  whose primary reference is older than the decoder's frame delay.
- `SEG_LVL_SKIP` and `SEG_LVL_GLOBALMV` are parsed/written by the segment
  syntax and the segment-id coder, but the encoder's AQ path enables only
  `SEG_LVL_ALT_Q`. They are not silently forced.
- Segmentation is not combined with IntraBC or palette mode in this encoder:
  IntraBC has a `read_skip()` before the segment id, and the current encoder
  does not emit that ordering. The two tools are disabled when segmentation is
  on.
- Mixed lossless segments are avoided: with `base_q_idx > 0`, ALT_Q values are
  clamped so every segment stays ≥ 1. A mixed lossless frame would change the
  residual/transform path and is not exercised by these streams.
- Loop filters (deblock/CDEF/CCSO/GDF/restoration) stay off in the segmentation
  verification streams; their segment-aware interaction is covered by the
  filter release units, not this one.
- No `unsafe`, no new dependency, no native build step, and no third-party code
  was added. The new file carries
  `SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause` because the AV2 module
  already carries the recorded dav2d conformance derivation.

Nothing was committed.
