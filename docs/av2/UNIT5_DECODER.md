# AV2 release unit 5 — decoder half

Date: 2026-10-29
Scope: AV2 v1.0.0 segmentation parsing and reconstruction in the decoder —
segment map (spatial prediction with the three-way context and
`neg_deinterleave`), temporal prediction from the primary reference, per-segment
`SEG_LVL_ALT_Q` / `SEG_LVL_SKIP` / `SEG_LVL_GLOBALMV`, per-superblock delta-Q,
`qindex` composition, and the segmentation interaction with skip mode,
inter/intra mode selection, and per-segment lossless.

Normative sources:

| Item | AV2 spec | Reference (read-only) |
| :-- | :-- | :-- |
| Sequence segment config | §5.4.4 `sequence_segment_config()` | `parse_seq_hdr`, dav2d `src/obu.c` |
| Segment feature enable/value | §5.4.9 `seg_info()` | `parse_seg_info`, dav2d `src/obu.c` |
| Frame segmentation params | §5.18.7.1 `segmentation_params()` | segmentation branch, dav2d `src/obu.c` |
| Intra segment id | §5.20.5.7 `intra_segment_id()` | `decode_b`, dav2d `src/decode.c` |
| Segment id prediction / `neg_deinterleave` | §5.20.5.8 `read_segment_id()` | `get_cur_frame_segid`, `src/env.h`; `neg_deinterleave`, `src/decode.c` |
| Skip syntax | §5.20.5.10 `read_skip()` | `decode_b` |
| Delta-Q | §5.20.5.11 `read_delta_qindex()` | delta-q block, `src/decode.c` |
| Inter segment id | §5.20.7.2 `inter_segment_id(preSkip)` | `decode_b` |
| Get segment id | §5.20.7.4 `get_segment_id()` | `get_prev_frame_segid` |
| Is-inter / ref frames / modes | §5.20.7.3, §5.20.7.6, §5.20.7.10 | `decode_b` |
| `get_qindex` / lossless | §5.18.1, §7.14.2, §7.14.4 | `init_quant_tables`, `src/decode.c` |
| Primary reference derivation | §7.8 | `derive_pri_sec_ref`, `src/obu.c` |

AV2 v1.0.0 has exactly three segment features. `SEG_LVL_ALT_LF` and
`SEG_LVL_REF_FRAME` from AV1 do not exist in AV2, so per-segment delta-LF and
per-segment reference frame are not implemented as segment features. Segment-map
prediction from a reference frame is `load_previous_segment_ids` (previous-map
prediction), not a feature.

## What was implemented

- `dec/hls.rs`
  - Parses frame segmentation (`enabled`, `reuse_seg_info`, `update_map`,
    `temporal`), including the key/intra rule that `update_map` is inferred to 1
    when `DerivedPrimaryRefFrame == PRIMARY_REF_NONE`.
  - Derives `DerivedPrimaryRefFrame` when not signalled (§7.8) from stored
    reference-slot type, `base_q_idx`, and order hint, so `update_map` /
    `temporal` bits are consumed exactly like dav2d/AVM.
  - Derives `LastActiveSegId`, `SegIdPreSkip`, per-segment `qidx`,
    `LosslessArray`, `HasLosslessSegment`, and `CodedLossless` with the
    per-bit-depth `MaxQ` (`255 + 24 * (BitDepth - 8)`).
  - Removed only the specific gates `unsupported("segmentation")` and
    `unsupported("frame-level delta-q")`. Other unrelated tool gates are
    unchanged.
- `dec/syntax.rs`
  - `SegSyntax` (frame segmentation view) and `decode_segment_id()`: the
    inverse of `encode_segment_id`/`read_segment_id()`; explicit ids, extended
    ids (`8..16`), three-way context, `neg_deinterleave`, temporal
    `seg_id_predicted`, skipped non-lossless inference, and
    `AboveSegPredContext`/`LeftSegPredContext` updates.
  - `decode_delta_qindex()`: the inverse of `encode_delta_q()`, including the
    `DELTA_Q_SMALL` escape and `delta_q_res` shift, clamped to
    `[1, MaxQ]`.
- `dec/frame.rs`
  - A per-tile `CurrentQIndex` and per-segment dequantizer tables recomputed at
    every coded superblock; `get_qindex(0, segment)` semantics for ALT_Q plus
    `CurrentQIndex`, and `get_q_qindex`-equivalent plane DC/AC deltas with the
    correct `MaxQ`.
  - Syntax order matches §5.20.5.3 / §5.20.7.1: `intra_segment_id()` before
    `read_gdf`/`read_cdef`/`read_ccso`/`read_delta_qindex`; inter
    `read_skip()` then `inter_segment_id(0)` (or `inter_segment_id(1)` before
    `is_inter` when `SegIdPreSkip`).
  - Per-segment lossless in the intra predictors (DPCM/lossless path
    eligibility) and per-segment quantizer tables in the coefficient path.
  - `SEG_LVL_SKIP` forces `skip_flag = 1` without consuming a symbol;
    `SEG_LVL_GLOBALMV` forces `is_inter = 1`, `SkipSegFrame`, `GLOBALMV`, and
    suppresses reference/mode/compound symbols.
  - `read_delta_qindex()` is skipped for a whole-superblock skip block
    (`MiSize == SbSize && skip_flag`), as required by §5.20.5.11.
- `dec/mod.rs`
  - `SavedSegmentIds` per reference slot and `RefFrameInfo` per slot.
  - Previous segment map is supplied from the derived primary reference; frame
    output returns the current map and stores it for refreshed slots.

## Acceptance evidence

Environment: Linux, 1 thread, dav2d rev `bff8e5c` at
`$ORACLE/dav2d`.

### Our encoder → our decoder → dav2d

The in-tree external test encodes 3 frames with `enable_vaq=1` and
`enable_delta_q=1`, decodes them with our decoder, and compares both outputs
with the encoder reconstruction and dav2d:

```bash
cargo test -p rustmpeg-video av2_segmentation_decoder_dav2d_byte_exact -- --ignored
# test result: ok. 1 passed; 0 failed; 0 ignored
```

Cases: 64x64 and 256x192, 8/10/12-bit, QP 20/100/140, 3 frames each
(key + 2 inter). **18/18 cases are byte-exact three ways**: encoder
reconstruction == our decoder == dav2d.

### In-tree segmentation/temporal round trip

```bash
cargo test -p rustmpeg-video av2
# 113 passed; 0 failed; 2 ignored
```

Includes `test_decode_matches_encoder_recon_with_segmentation_and_delta_q`
(18 configurations) and
`test_decode_matches_encoder_recon_with_temporal_segment_map`, which asserts
that `segmentation_temporal_update` is actually signalled and that the decoded
frame equals the encoder reconstruction. Unit tests cover explicit ids 0..15
(including the extension flag), spatial three-way prediction, skipped inference
(no coded symbol), preskip with `SEG_LVL_SKIP`, temporal prediction, and every
delta-Q escape range at `delta_q_res` 0 and 2.

### Toolchain / project gates

```text
cargo fmt --all -- --check                                  -> rc 0
cargo clippy --workspace --all-targets -- -D warnings       -> rc 0
cargo test --workspace                                     -> all suites pass
cargo test -p rustmpeg-video vvc                           -> 129 passed, 0 failed
python3 scripts/check-licenses.py                          -> license audit: OK
python3 $ORACLE/check_av2.py quick        -> 9/9 OK
cargo build --release -p rustmpeg-video --examples         -> rc 0
```

The VVC tree was not touched; its 129-test suite remains green.

## AVM (`avmenc`) attempt and exact flags

AVM variance AQ (`--aq-mode=1`) did not emit segmentation in this build for the
short streams tested, even with `--end-usage=q/cq/vbr/cbr`, `--aq-mode=1`,
`--deltaq-mode=1`, `--enable-tpl-model=1`, `--min-qp/--max-qp`, and a natural
256x192 clip (`mandelbrot`). Header inspection showed
`segmentation_enabled = 0` for every frame. Cyclic-refresh AQ
(`--aq-mode=3`) does emit segmentation (`enabled=1`, `update_map=1`,
`LastActiveSegId=2`, `SegIdPreSkip=0`) on inter frames.

Exact command for the segmentation-producing stream (256x192, 8-bit, 3 frames):

```bash
avmenc --ivf --limit=3 --cpu-used=9 --threads=1 \
  --end-usage=cbr --target-bitrate=300 --min-qp=100 --max-qp=200 \
  --aq-mode=3 --enable-qm=0 --qm-min=0 --qm-max=0 \
  --enable-tcq=0 --lag-in-frames=0 --kf-max-dist=30 --tune-content=default \
  --enable-adaptive-mvd=0 --enable-bawp=0 --enable-bru=0 --enable-ccso=0 \
  --enable-cctx=0 --enable-cdef=0 --enable-cdef-on-skip-txfm=0 \
  --enable-chroma-deltaq=0 --enable-cwp=0 --enable-deblocking=0 \
  --enable-diff-wtd-comp=0 --enable-extended-sdp=0 --enable-ext-partitions=0 \
  --enable-fsc=0 --enable-flex-mvres=0 --enable-gdf=0 \
  --enable-global-motion=0 --enable-ibp=0 --enable-idtx-intra=0 \
  --enable-imp-msk-bld=0 --enable-inter-ddt=0 --enable-interinter-wedge=0 \
  --enable-interintra-comp=0 --enable-interintra-wedge=0 \
  --enable-inter-ist=0 --enable-intrabc=0 --enable-intra-dip=0 \
  --enable-ist=0 --enable-joint-mvd=0 --enable-lf-sub-pu=0 \
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
  --enable-qm=0 --enable-mfh-obu-signaling=0 --enable-lcr=0 \
  --enable-operating-point-sets=0 --enable-cropping-window=0 \
  --enable-keyframe-filtering=0 --enable-overlay=0 --enable-sframe=0 \
  --enable-fwd-kf=0 --enable-banding-metadata=0 \
  --enable-intra-edge-filter=0 --enable-mrls=0 --enable-cfl-intra=0 \
  -y -o avm_seg.ivf input.y4m
```

`dav2d` decodes that stream (`rc 0`). Our decoder now parses the segmentation
header (the previously observed `quantization matrices` misalignment was the
missing `DerivedPrimaryRefFrame`; it is fixed), but it then reaches an unrelated
block-syntax gate:

```text
Error: Unsupported("sub-block transform partitioning")
```

The same gate fires with `--aq-mode=0` (segmentation disabled), so it is not a
segmentation defect. AVM’s verbose log reports `Enable txfm partition: 0`, but
the stream still reaches the tx-partition syntax path in our decoder; enabling
that tool is outside unit 5 and outside the decoder files owned here. The AVM
byte-exact comparison therefore remains **blocked by an unrelated unsupported
tool**, not by segmentation/delta-Q parsing. The exact AVM command and error
above are reproducible.

## Remaining gaps and risks

- AVM stock streams that use tools outside the decoder subset (first observed:
  sub-block transform partitioning) cannot be compared byte-exactly yet. The
  segmentation header and segment-id/delta-Q syntax are parsed up to that point.
- Loop-filter interaction with segmentation (deblock/CDEF/CCSO/GDF/restoration
  using `seg_id`/`CurrentQIndex`) belongs to the filter release units; the
  segmentation streams here keep those filters off.
- Mixed-lossless segmentation is handled per §5.18.1, but the encoder avoids it
  and the lossless transform path remains a pre-existing decoder limitation
  (QP 0 streams are still rejected).
- No `unsafe`, no new dependency, no native build step, and no third-party code
  was added. The AV2 module keeps its recorded dav2d conformance derivation.

Nothing was committed.
