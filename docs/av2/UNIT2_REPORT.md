# Unit 2 — CDEF (release close-out)

Date: 2026-10-30
Scope: AV2 constrained directional enhancement filter (CDEF), decoder and
encoder, for the v1.0.0 subset in `rustmpeg-video::av2`.

## Verdict

**DONE.** The encoder and decoder now apply the same CDEF. Our encoder
reconstruction is byte-exact against `dav2d` with CDEF on and deblocking off
across 8/10/12-bit, 4:2:0/4:2:2/4:4:4, QP 20/100/140, 64x64 and 256x192 for
3 frames (**54/54 cases**). AVM `avmenc` streams with CDEF enabled decode
byte-exactly against `dav2d` at 256x192 and 64x64, QP 100/140/200, including a
stream with `CdefStrengths = 3`.

## Specification

Implementation and comments cite AV2 Specification v1.0.0:

- §5.18.7.10 `cdef_params()` syntax (`CdefDamping`, `CdefStrengths`,
  `cdef_on_skip_txfm_frame_enable`, packed y/uv strengths).
- §5.4.9 sequence-header semantics: with `single_picture_header_flag` set,
  `CdefOnSkipTxfm` is inferred to `CDEF_ON_SKIP_TXFM_ADAPTIVE`.
- §5.20.10.1 `read_cdef()` (per-64x64 strength index, `cdef_index0` context).
- §5.20.9.3 `is_inside_filter_region()`.
- §7.18 CDEF process; §7.18.1 CDEF block process (`Skips`, `skipChroma`,
  `LosslessArray`, per-plane order); §7.18.2 CDEF direction process
  (`Div_Table`, `var`); §7.18.3 CDEF filter process (`Cdef_Pri_Taps`,
  `Cdef_Sec_Taps`, `Cdef_Directions`, `Cdef_Uv_Dir`, `constrain`, clipping).

## Changes

`crates/rustmpeg-video/src/av2/cdef.rs`:

1. The §7.18.1 `Skips` input is the per-4x4 non-zero-coefficient mask
   (`lr_noskip`), not the inter skip flag. An 8x8 block with no coded
   coefficients is not filtered even when `skip_txfm == 0`. This fixed the
   1-LSB luma differences on AVM intra streams with flat 8x8 blocks.
2. New unit test `test_cdef_multiple_strengths_change_output` applies the
   filter with strength codes 4/8/13/21 and asserts the luma output changes.

`crates/rustmpeg-video/src/av2/frame.rs`:

- The encoder's per-4x4 grid (`record_deblock`) is now populated when **either**
  deblocking or CDEF is enabled. Before this, CDEF with deblocking off saw the
  default all-skip grid and filtered nothing.

`crates/rustmpeg-video/src/av2/dec/frame.rs` (strictly required for CDEF):

- CDEF index alphabet corrected to `CdefStrengths - 2` symbols (the AVM/dav2d
  `nb_cdef_strengths - 1` `nsyms`), so index values stay inside
  `0..CdefStrengths-1`.
- The 64x64 index map is filled from the aligned CDEF-unit base over the whole
  coding-block extent, not `div_ceil` from the block start. Blocks crossing a
  64x64 boundary now mark every covered unit.

`crates/rustmpeg-video/src/av2/dec/hls.rs` (strictly required for CDEF):

- With `single_picture_header_flag`, `CdefOnSkipTxfm` is now inferred to
  `ADAPTIVE`, so the frame-level `cdef_on_skip_txfm_frame_enable` bit is read.
  Before this, the bit was skipped and every following strength field was
  shifted; this was the root cause of the AVM CDEF mismatch.

`crates/rustmpeg-video/src/av2/mod.rs`:

- New non-ignored test `av2_unit2_cdef_changes_recon_and_varies`: every
  bit-depth/layout/QP/size combination, decoder == encoder reconstruction,
  non-zero strength and recon change at QP 100/140, multiple directions from
  the direction search.
- New ignored external test `av2_unit2_cdef_dav2d_byte_exact`: the full
  54-case matrix with CDEF on, deblocking off, compared with `dav2d`; for
  QP 100/140 it also encodes with CDEF off and asserts the reconstruction
  differs.

No change was needed in `av2/headers.rs`, `av2/obu.rs`, `av2/deblock.rs` or
`examples/av2_encode.rs`.

## Verification

All commands were run in `$REPO`
unless stated otherwise.

```text
cargo fmt --all -- --check                                      -> rc 0
cargo clippy --workspace --all-targets -- -D warnings           -> rc 0
cargo test -p rustmpeg-video av2                                -> 158 passed, 0 failed, 12 ignored
cargo test -p rustmpeg-video --lib av2_unit2_cdef_dav2d_byte_exact -- --ignored
                                                                -> 1 passed (22.16 s)
cargo test -p rustmpeg-video --lib test_cdef_multiple_strengths_change_output
                                                                -> 1 passed
python3 scripts/check-licenses.py                               -> license audit: OK
bash $ORACLE/vvc-conf/run_conformance.sh                -> rc 0, 25/25 VTM OK, 25/25 VVdeC OK
```

Clippy emits only the expected `unused dependency` manifest warnings from
`rustmpeg-image` manifest warnings are pre-existing; there are no code warnings.

### Encoder matrix versus dav2d

Driver: the ignored test above. Each case encodes 3 frames with the release
example, decodes with our decoder and with
`dav2d --threads 1 --framedelay 1 --strict 1 -q`, and compares the encoder
reconstruction, our decoder and `dav2d` on every frame. Deblocking is off so
the comparison isolates CDEF.

Matrix: `{64x64, 256x192}` x `{I420, I422, I444}` x `{8, 10, 12}`-bit x
`{20, 100, 140}` QP x 3 frames = **54 cases**.

| Result | Cases |
| :-- | --: |
| Encoder recon == our decoder == dav2d | **54** |
| Mismatches | **0** |

At QP 100/140 every case selected a non-zero CDEF strength and the CDEF-on
reconstruction differs from the CDEF-off reconstruction (asserted in the
test). QP 20 maps to a qindex where `optimize_and_apply_cdef` signals a zero
strength, so it is a no-op, matching the QP 20 deblocking behaviour.

### AVM streams with CDEF enabled

Tool-disable set: the 87-flag list in
[`UNIT1_REPORT.md`](UNIT1_REPORT.md) §"AVM streams with deblocking enabled",
plus `--reduced-tx-type-set=2 --reduced-tx-part-set=1`. The CDEF run differs
only in:

```text
--enable-cdef=1 --enable-cdef-on-skip-txfm=0 \
--enable-deblocking=0 --enable-ccso=0 --enable-gdf=0 --enable-restoration=0
```

`avmenc` command shape (8-bit, 256x192, 1 frame):

```bash
avmenc --ivf --limit=1 --cpu-used=9 --threads=1 \
  --end-usage=q --qp=100 --min-qp=100 --max-qp=100 --aq-mode=0 \
  --enable-qm=0 --qm-min=0 --qm-max=0 --enable-tcq=0 \
  --lag-in-frames=0 --kf-max-dist=1 --tune-content=default \
  --enable-adaptive-mvd=0 --enable-bawp=0 --enable-bru=0 --enable-ccso=0 \
  --enable-cctx=0 --enable-cdef=1 --enable-cdef-on-skip-txfm=0 \
  --enable-chroma-deltaq=0 --enable-cwp=0 --enable-deblocking=0 \
  --enable-diff-wtd-comp=0 --enable-extended-sdp=0 --enable-ext-partitions=0 \
  --enable-fsc=0 --enable-flex-mvres=0 --enable-gdf=0 --enable-global-motion=0 \
  --enable-ibp=0 --enable-idtx-intra=0 --enable-imp-msk-bld=0 \
  --enable-inter-ddt=0 --enable-interinter-wedge=0 --enable-interintra-comp=0 \
  --enable-interintra-wedge=0 --enable-inter-ist=0 --enable-intrabc=0 \
  --enable-intra-dip=0 --enable-ist=0 --enable-joint-mvd=0 --enable-lf-sub-pu=0 \
  --enable-masked-comp=0 --enable-mhccp=0 --enable-mvd-sign-derive=0 \
  --enable-mv-traj=0 --enable-onesided-comp=0 --enable-opfl-refine=0 \
  --enable-palette=0 --enable-parity-hiding=0 --enable-pc-wiener=0 \
  --enable-ref-frame-mvs=0 --enable-refinemv=0 --enable-restoration=0 \
  --enable-sdp=0 --enable-six-param-warp-delta=0 --enable-skip-mode=0 \
  --enable-tip=0 --enable-tip-refinemv=0 --enable-tx-partition=0 \
  --enable-uneven-4way-partitions=0 --enable-warp-causal=0 \
  --enable-warp-delta=0 --enable-warped-motion=0 --enable-warp-extend=0 \
  --enable-wiener-nonsep=0 --enable-trellis-quant=0 --enable-refmvbank=0 \
  --enable-short-refresh-frame-flags=0 --enable-ext-seg=0 --enable-avg-cdf=0 \
  --enable-mfh-obu-signaling=0 --enable-lcr=0 --enable-operating-point-sets=0 \
  --enable-cropping-window=0 --enable-keyframe-filtering=0 --enable-overlay=0 \
  --enable-sframe=0 --enable-fwd-kf=0 --enable-banding-metadata=0 \
  --enable-intra-edge-filter=0 --enable-mrls=0 --enable-cfl-intra=0 \
  --reduced-tx-type-set=2 --reduced-tx-part-set=1 \
  -y -o out.ivf $ORACLE/u13/avm_in_256x192.y4m
```

| Input / frames | QP | CDEF signalling | `dav2d` | Our decoder |
| :-- | --: | :-- | :-- | :-- |
| `u13/avm_in_256x192.y4m`, 1 | 100 | `CdefStrengths=3`, distinct y/uv strengths | rc 0 | rc 0, **byte-exact** |
| `u13/avm_in_256x192.y4m`, 1 | 140 | CDEF enabled | rc 0 | rc 0, **byte-exact** |
| `u13/avm_in_256x192.y4m`, 1 | 200 | CDEF enabled | rc 0 | rc 0, **byte-exact** |
| `c64_grad.y4m` (64x64 gradient), 1 | 100 | `CdefStrengths=1`, y=56, uv non-zero | rc 0 | rc 0, **byte-exact** |
| `c64_grad.y4m` (64x64 gradient), 1 | 140 | CDEF enabled | rc 0 | rc 0, **byte-exact** |

The QP 100 256x192 stream exercises the multi-strength index path: AVM
signalled three strengths (observed luma strengths 2 and 34 in the first two
slots, chroma strength 8), and our decoder reproduced every sample.

Other AVM attempts and exact results:

- 2-frame and 3-frame inter streams with CDEF on: the key frame is byte-exact;
  the inter frames differ from `dav2d` (68,547 samples, max 202 on frame 1).
  The same difference remains with CDEF disabled at the decoder
  (`dav2d --inloopfilters none` vs our pre-CDEF frame: 68,458 samples, max
  204), so this is the known unrelated inter-reconstruction gap, not CDEF.
- 10-bit AVM streams made from the same 8-bit input, and 12-bit / 4:2:2 /
  4:4:4 AVM streams: `dav2d` rejects the sequence header, as recorded in
  [`UNIT1_REPORT.md`](UNIT1_REPORT.md). They are not used as CDEF evidence.

## RD / speed impact

1920x1080, 10 frames, 1 thread, synthetic source, release build, same machine:

| QP | CDEF | fps | bytes (frame 9) | avg PSNR Y (frame 9) |
| --: | :-- | --: | --: | --: |
| 100 | off | 17.61 | 53585 | 37.27 |
| 100 | on | 5.24 | 53450 | 37.35 |
| 140 | off | 21.31 | 12862 | 36.00 |
| 140 | on | 6.24 | 12707 | 36.25 |

CDEF saves about 0.25% of the bytes at QP 100 and 1.2% at QP 140, with a small
PSNR gain. The encoder-side CDEF search is expensive in this single-thread run
(about 3.4x slower at QP 100, 3.4x at QP 140) because it clones and filters the
frame for each candidate strength. CDEF remains **off by default**; the cost
does not affect the default encode path. Reducing the search (for example,
computing luma SSE without cloning) is a follow-up optimization, not a
conformance issue.

## Remaining gaps and risks

- AVM inter streams are still blocked by the known inter-reconstruction gap;
  CDEF is verified on intra/key frames. The pre-CDEF comparison above proves
  the gap is independent of CDEF.
- AVM 10/12-bit and 4:2:2/4:4:4 streams cannot be used as an oracle because
  `dav2d` rejects their sequence headers. High-bit-depth and chroma-subsampled
  CDEF is verified end-to-end against `dav2d` on our own encoder streams.
- The encoder selects one strength per frame from a small candidate set
  (`4/8/13/21`); the AVM multi-strength index path is exercised by the AVM
  streams, not by our encoder.
- CDEF on skip-txfm blocks is exercised through `on_skiptx = 1` in the unit
  test; the AVM streams use `--enable-cdef-on-skip-txfm=0`.

## Files changed

- `crates/rustmpeg-video/src/av2/cdef.rs`
- `crates/rustmpeg-video/src/av2/frame.rs`
- `crates/rustmpeg-video/src/av2/dec/frame.rs` (strictly required for CDEF)
- `crates/rustmpeg-video/src/av2/dec/hls.rs` (strictly required for CDEF)
- `crates/rustmpeg-video/src/av2/mod.rs`
- `docs/av2/UNIT2_REPORT.md`

Commit: `av2: complete CDEF coverage (unit 2)`.
