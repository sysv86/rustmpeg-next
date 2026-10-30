# AV2 unit 10 encoder: extended partitions and partition RDO

Status: encoder half complete. The extended partition syntax is emitted, the
partition search is an actual rate-distortion comparison, and the emitted
streams decode byte-exactly with `dav2d` on the recorded matrix.

## Scope and references

- AV2 Specification v1.0.0 §5.20.3.2: `Partition_Subsize`,
  `H_Partition_Midsize`, `is_ext_partition_allowed`,
  `is_uneven_4way_partition_allowed`, `rect_type_implied_by_bsize`,
  `is_partition_allowed`, `init_allowed_partitions`, `partition_implied`,
  `read_partition`, chroma offset propagation, and the extended CDF contexts.
- `dav2d` `src/decode.c` `read_partition` / `decode_sb` (`subb`,
  `has_hv3`, `has_hv4ab`, `pcc->ctx`, `get_partition2_ctx`) as the black-box
  conformance oracle.
- No FFmpeg, AVM, or dav2d source was copied or transcribed. The `subb` table
  values were read only to check our independently written `Partition_Subsize`
  rows.

## Implementation

`crates/rustmpeg-video/src/av2/partition.rs`

- Corrected the `Partition_Subsize` quarter and eighth columns for the small
  rectangular sizes (`Bs8x16`/`Bs16x16`/`Bs32x16` for H3, the mirrored V3 rows,
  `Bs64x32` for H4A/B and `Bs32x64` for V4A/B).
- Added `ext_selection_allowed`, the `has_hv3` / `has_hv4ab` geometry gate
  (split-axis size, 8:1 aspect limit, and subsampled chroma size) used before a
  candidate enters the search.
- `init_allowed_partitions`, `is_partition_allowed`, `partition_implied`,
  `encode_partition_decision`, and the extended CDF contexts follow the
  sequence-header flags and the normative symbol order.

`crates/rustmpeg-video/src/av2/frame.rs`

- Replaced the variance-only decision with a candidate cost
  `J = D + lambda * R` over `NONE`, `H`, `V`, `SPLIT`, `H3`, `V3`, `H4A/B`,
  `V4A/B`, using the existing SAD/SATD prediction cost and the current
  `lambda_sad`. The old `D(quads) >= 0.9 * D(whole)` early exit that suppressed
  extended candidates was removed.
- Extended candidates are searched at 64x64 nodes, where the normative
  geometry gate holds for I420/I422/I444. Smaller nodes keep the regular
  square/rectangular search.
- Added per-partition counters (`FrameStats::partition_counts`) and the
  `rect`/`ext`/`uneven` example flags.

`crates/rustmpeg-video/examples/av2_encode.rs`

- New positional flags 32/33/34: `rect 0|1`, `ext 0|1`, `uneven 0|1`.
- The per-frame log prints the selected partition counts.

### Known encoder limitation

Rectangular luma leaves in extended-partition streams use `DC_PRED`. Wide-angle
directional prediction on non-square blocks is not bit-exact against `dav2d` in
this encoder yet, so the mode search keeps those leaves on the verified mode.
Square blocks keep the full DC/directional/smooth/Paeth search, and default
streams (rectangular partitions disabled) are unchanged. This is recorded as a
remaining gap, not as a completed tool.

## Commands and results

All commands were run in the repository root on 2026-10-29.

```text
cargo fmt --all -- --check                              -> rc 0
cargo clippy --workspace --all-targets -- -D warnings   -> rc 0
cargo test -p rustmpeg-video av2                        -> 150 passed, 0 failed, 7 ignored
python3 scripts/check-licenses.py                       -> license audit: OK
python3 $ORACLE/check_av2.py quick    -> 9/9 OK
python3 $ORACLE/check_av2.py          -> 84/84 OK
```

### Extended-partition matrix (encoder recon vs `dav2d`)

Encoder flags `rect=1 ext=1 uneven=1`, 3 frames, I420, `dav2d --threads 1
--framedelay 1 --strict 1 -q`.

| Size | QP | Result |
| :-- | --: | :-- |
| 64x64 | 20 | MATCH |
| 64x64 | 100 | MATCH |
| 64x64 | 140 | MATCH |
| 256x192 | 20 | MATCH |
| 256x192 | 100 | MATCH |
| 256x192 | 140 | MATCH |
| 64x64, 10-bit | 100 | MATCH |

The 10-bit case selects V3 (`h3=0 v3=1`) and matches both the encoder
reconstruction and our decoder against `dav2d`.

### Selection proof

Synthetic 256x192, QP 100, frame 0:

```text
partitions none 62 h 0 v 20 h3 4 v3 6 h4a 0 h4b 0 v4a 0 v4b 0 split 0
```

`bands2.y4m` (64x64 blocks with 1/8-1/4-1/2-1/8 bands), 256x192, QP 20,
frame 0:

```text
partitions none 48 h 0 v 0 h3 0 v3 0 h4a 3 h4b 3 v4a 3 v4b 3 split 0
```

The `rect`-only stream and the `rect+ext+uneven` stream differ at byte 33
(`cmp` reports a difference), and `dav2d` decodes the extended stream to the
same pixels as the encoder reconstruction on all three frames.

### RD measurement on the 1080p clip

`src_testsrc2_1080p.y4m`, 10 frames, 1 thread, I420. `rect` = `rect=1 ext=0
uneven=0`; `ext` = `rect=1 ext=1 uneven=1`.

| QP | rect kbps | rect PSNR Y | ext kbps | ext PSNR Y | bitrate delta |
| --: | --: | --: | --: | --: | --: |
| 20 | 47398.8 | 37.709 | 46391.2 | 37.878 | -2.1% |
| 60 | 28125.8 | 37.797 | 27534.7 | 37.863 | -2.1% |
| 100 | 13049.3 | 37.555 | 12808.8 | 37.598 | -1.8% |
| 140 | 3396.1 | 36.349 | 3389.1 | 36.347 | -0.2% |

Piecewise-linear BD-rate over the monotonic QP 60/100/140 subset: **-5.7%**
(extended partitions use 5.7% fewer bits at matched PSNR). The QP 20 point is
excluded from the BD-rate because its PSNR (37.709) is below QP 60 (37.797),
which makes the four-point RD curve non-monotonic and the cubic Bjontegaard fit
ill-conditioned; the raw point still shows -2.1% bitrate at +0.169 dB.

Speed on the same clip (10 frames, 1 thread): rect 653/574/461/383 ms and ext
604/559/492/385 ms at QP 20/60/100/140, i.e. within measurement noise of the
rect-only encoder (about 1.0x).

## Remaining gaps

- Non-square luma intra prediction is restricted to `DC_PRED` as described
  above. Enabling the full directional search on rectangular leaves requires a
  bit-exact wide-angle mode/context implementation and is not part of this
  change.
- Extended partitions are searched at 64x64 nodes only; the syntax and allowed
  sets cover the smaller sizes.
- No commit, tag, or push was made.
