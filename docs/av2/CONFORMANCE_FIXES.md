# AV2 conformance fixes: odd-size corner and 12-bit warped motion

Date: 2026-10-29
Scope: the two non-conformances recorded in
[`docs/AV2-RELEASE.md`](../AV2-RELEASE.md) "Failures (not hidden)". Both are
fixed. The supported tool set, the pinned toolchain, and the dependency set are
unchanged.

Oracle: `dav2d` revision `bff8e5c`
(`$ORACLE/dav2d`), run with
`--threads 1 --framedelay 1 --strict 1 -q`. Encoder reconstruction is compared
with our decoder and with `dav2d` on the Y4M frame payloads.

## Defect 1: 1-LSB difference in the bottom-right 2x2 corner

### Symptom

For frames whose coded size leaves a 2x2 remainder in the bottom-right
superblock (`130x66`, `66x66`, `130x130`), a few luma samples at
`(128,64)`..`(129,65)` differed from `dav2d` by 1 LSB after several frames.
Encoder reconstruction and our decoder always agreed with each other; only
`dav2d` differed.

### Root cause

Two related edge rules were wrong.

1. **Inter-prediction edge emulation used the visible plane size.** AV2
   §7.13.2.1 derives `maxX = (MiCols * MI_SIZE) - 1` and
   `maxY = (MiRows * MI_SIZE) - 1`, with `MiCols = 2 * ((FrameWidth + 7) >> 3)`
   and `MiRows = 2 * ((FrameHeight + 7) >> 3)`: the coding grid is padded to a
   multiple of **8** samples. The intra path already used that extent, but
   every inter-prediction, compound, motion-search and warped-motion call
   clamped the reference reads to `Plane::width`/`height`, i.e. the visible
   size. For `130x66` that is `130x66` instead of the coded `136x72`, so motion
   compensation near the right and bottom edges clamped to the wrong sample and
   reconstructed the coded padding differently from the reference decoder.
2. **The encoder overwrote the coded padding of every reference frame.** After
   reconstructing a frame, `encode_frame` replicated the visible last row and
   column over the whole aligned plane before inserting it into the DPB. The
   coded pixels at `x in [width, MiCols*4)` and `y in [height, MiRows*4)` are
   real reconstructed samples that intra and inter prediction may read; the
   decoder keeps them, so the encoder and decoder reference frames could
   diverge as soon as a block used them.

The visible 2x2 corner is the top-left corner of the last 8x8 block. When that
block was intra DC, its average included the coded padding of the block above
and to the left. Those padded samples had drifted by 1 LSB, which changed the
DC average and therefore the visible samples. This is why the error only
appeared after a few inter frames and only for sizes with a 2-pixel remainder.

### Fix

- `Plane` now carries `coded_width`/`coded_height`, set by `Frame::new` to
  `ceil(width / 8) * 8` and `ceil(height / 8) * 8` (chroma shifted by the plane
  subsampling).
- All reference extents passed to `mc_predict_block`,
  `mc_predict_block_prep`, `select_best_subpel_filter`, `estimate_motion_block`
  and `block_warp` use `coded_width`/`coded_height` instead of the visible
  plane size, in both the encoder and the decoder.
- The encoder no longer calls `pad_edges()` on reference frames. Reconstructed
  pixels inside the coded extent are preserved, matching the decoder's DPB.
  Source frames are still padded when read, which only affects the encoder's
  residual choice and not the normative reconstruction.

Files: `av2/frame.rs`, `av2/dec/frame.rs`, `av2/inter.rs`, `av2/warp.rs`
(the `Plane` fields also required adding them to the `Plane` literal in
`av2/ccso.rs`'s unit test).

### Verification

`130x66`, `66x66` and `130x130`, 8-bit, QP 20/100/140, 3 frames:
all nine cases satisfy `encoder reconstruction == our decoder == dav2d`.
The original failure case (`130x66`, 8-bit, QP 100, 10 frames) is exact on every
frame, as are the 10-frame 10-bit and 12-bit variants. `129x64`, `96x64`,
`100x64`, `256x192` remain exact.

## Defect 2: 12-bit warped motion

### Symptom

12-bit warped-motion streams differed from `dav2d` at QP 20 and QP 100 (1 LSB
per sample in 253-281 samples), while 8-bit and 10-bit were exact and QP 140
was exact. The 255 values in the release report were byte-level Y4M deltas of a
1-LSB 16-bit sample difference.

### Root cause

`block_warp` (§7.13.3.19) performs two separable 8-tap passes. The code used
`InterRound0 = 3` and `InterRound1 = 11` for every bit depth. The reference
decoder keeps the intermediate array in 16 bits and derives the pass rounding
from the bit depth (`intermediate_bits = 14 - BitDepth`), so the high-bit-depth
passes round by `BitDepth - 7` and `21 - BitDepth`: `3/11` for 10-bit but
`5/9` for 12-bit. With `3/11` at 12-bit the horizontal intermediate is four
times larger than the reference's and the two rounding stages disagree by
1 LSB. 8-bit and 10-bit were already using the reference's shifts, which is why
only 12-bit failed.

The warp parameter precision and clamping (`reduce_warp_model`, `setup_shear`,
`get_warp_translation`) were checked against §7.13.3.21-§7.13.3.23 and are
unchanged; they were not the cause.

### Fix

`block_warp` now selects `(round0, round1)` from the bit depth:

```text
BitDepth <= 8:  (3, 11)
BitDepth >  8:  (BitDepth - 7, 21 - BitDepth)   // 10-bit: 3/11, 12-bit: 5/9
```

The encoder and decoder share `block_warp`, so the model search, the coded
model and the reconstruction stay consistent.

Note: the literal `InterRound0`/`InterRound1` pair in §7.13.3.16 is `3/11`.
The reference decoder's 16-bit intermediate scaling above is the behaviour that
the recorded conformance oracle produces at 12-bit; matching it is required for
byte-exact output, and the difference from the literal pair is at most 1 LSB.

### Verification

12-bit, 3 frames, warped-motion enabled, zooming 12-bit I420 source:

| Size | QP | Warped blocks (encoder) | Result |
| :-- | --: | --: | :-- |
| 64x64 | 20 | 2 | `recon == ours == dav2d` |
| 64x64 | 100 | 2 | `recon == ours == dav2d` |
| 64x64 | 140 | 2 | `recon == ours == dav2d` |
| 256x192 | 20 | 2 | `recon == ours == dav2d` |
| 256x192 | 100 | 2 | `recon == ours == dav2d` |
| 256x192 | 140 | 1 | `recon == ours == dav2d` |

8-bit and 10-bit warp streams are unchanged and remain exact. The archived
12-bit warp stream `u8_64x64_q100_12b_warp.ivf` now decodes byte-exactly
against `dav2d`; its archived reconstruction was produced by the old encoder
and therefore still reflects the old rounding.

## Regression and gates

The original 102-case release matrix driver was an external harness and is not
in the repository. It was reconstructed from the archived case list in
`$ORACLE/u13/matrix/results_*.json` (same names, sizes, QPs,
bit depths, layouts, frame counts and tool switches) and re-run:

- Release matrix: **102 / 102 pass** (baseline 51, unit 5 18, unit 8 10,
  unit 11 13, unit 12 10).
- Repository `check_av2.py` full matrix: **84 / 84 OK**.
- `cargo fmt --all -- --check`: rc 0.
- `cargo clippy --workspace --all-targets -- -D warnings`: rc 0.
- `cargo test --workspace`: **711 passed, 0 failed, 3 ignored**.
- `python3 scripts/check-licenses.py`: `license audit: OK`.
- `$ORACLE/vvc-conf/run_conformance.sh`: rc 0, **25 / 25 cases OK**
  with VTM and VVdeC.

No new dependency, no native build step, no `unsafe`, and no third-party code
was added.
