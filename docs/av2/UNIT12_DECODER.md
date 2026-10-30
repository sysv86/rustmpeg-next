# Unit 12 decoder: AV2 film grain parsing and synthesis

Status: decoder half complete. Nothing was committed. The shared synthesis
module and the encoder half are covered by
[`UNIT12_ENCODER.md`](UNIT12_ENCODER.md); this document covers the decoder-side
OBU parsing, model storage, and output-side synthesis. The encoder document's
closing note that the decoder still rejects film grain is superseded by this
change.

## Scope and normative references

AV2 v1.0.0 film grain on the decoder side:

- `film_grain_obu()` and `film_grain_model()` — §5.14 and §5.18.10.2.
- `film_grain_config()` in the frame header — §5.18.10.1.
- `fgm_update_flags`, `fgm_chroma_idc`, `save_grain_model()` — §6.13.
- `apply_grain`, `fgm_id`, `grain_seed`, `load_grain_model()` — §6.17.10.1.
- Model semantics and conformance bounds (14 points, increasing x, 4:2:0
  chroma-pair rule) — §6.17.10.2.
- Film grain OBU availability before use — §7.3.8.8.
- Grain is applied to `OutY/OutU/OutV` only, after the output picture is
  prepared, and never to `FrameStore` — §7.21.1 and §7.21.2.
- Synthesis: random number process §7.21.7.2, generate grain §7.21.7.3,
  scaling lookup §7.21.7.4, add noise §7.21.7.5.

The 2048-entry `Gaussian_Sequence` and the synthesis code live in the shared
`av2/filmgrain.rs` module; the decoder only parses, stores, and invokes it.

## Implementation

`dec/hls.rs`:

- `parse_film_grain_obu()` reads `fgm_update_flags`, validates
  `fgm_chroma_idc <= 3` and that it matches the sequence layout (§6.17.10.1),
  parses every set slot with `parse_film_grain_model()`, and requires
  `trailing_bits()`.
- `parse_film_grain_model()` enforces `num_*_points <= 14`, increasing point
  values below 256, the 4:2:0 rule that Cb and Cr points are both present or
  both absent, and reads the per-plane AR coefficients with the extra chroma
  luma coefficient (`numPosChroma`).
- The `Unsupported("film grain synthesis")` return in `film_grain_config()` is
  removed; `apply_grain`, `fgm_id`, and `grain_seed` are parsed as before.

`dec/mod.rs`:

- `Decoder` keeps `grain_models: [Option<FilmGrainData>; 8]`, updated by any
  `OBU_FILM_GRAIN` in the temporal unit and retained across temporal units
  (§6.13: slots may be reused later).
- `frame_grain()` resolves the `fgm_id` referenced by the frame header and
  reports a missing model as a bitstream error.
- `finish_frame()` applies the shared `apply_film_grain()` to a **clone** of the
  reconstruction. The clone becomes `DecodedFrame::frame`; the original clean
  reconstruction enters `ReferenceDpb`, so reference frames and later inter
  predictions stay clean (§7.21.1).
- `set_apply_grain(bool)` mirrors the reference decoder's `apply_grain` option.
  It is enabled by default and only affects the returned output picture.

`dec/frame.rs`:

- Removed only the `film grain synthesis` gate from `validate_tools()`.

`examples/av2_decode.rs`:

- Optional fourth argument `grain 0|1` (default `1`) to exercise the
  grain-off comparison. No encoder example was changed.

## Verification commands and results

Environment: `$ORACLE/dav2d` with `--threads 1
--framedelay 1 --strict 1`, release build, 3 synthetic frames per case.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p rustmpeg-video av2
python3 scripts/check-licenses.py
AVM_FG_IVF=$WORK/hgrad_avm.ivf \
  cargo test -p rustmpeg-video av2_film_grain_decoder_dav2d -- --ignored
```

Results:

- `cargo fmt --check`: clean.
- `cargo clippy ... -D warnings`: clean; only the pre-existing manifest
  warnings for unused workspace dependencies remain.
- `cargo test -p rustmpeg-video av2`: 126 passed, 0 failed, 3 ignored.
- `check-licenses.py`: `license audit: OK (all packages permissive, no
  copyleft)`.
- `cargo test -p rustmpeg-video vvc`: 129 passed, 0 failed; no VVC file was
  touched.
- The ignored film grain test: 1 passed, including the AVM stream.

### Encoder streams with grain

`av2_film_grain_decoder_dav2d_byte_exact` encodes with
`ToolConfig::enable_film_grain`, decodes with grain on and off, and compares
both against dav2d and against the signalled-model synthesis. Every row asserts
`decoder grain-on == dav2d --filmgrain 1 == encoder grained recon` and
`decoder grain-off == dav2d --filmgrain 0 == encoder clean recon`.

| Case | Frames | Grain changes output | grain-on == dav2d fg1 | grain-off == dav2d fg0 |
| :-- | --: | :-- | :-- | :-- |
| 64x64 QP 20, 8-bit | 3 | yes | yes | yes |
| 64x64 QP 100, 8-bit | 3 | yes | yes | yes |
| 64x64 QP 140, 8-bit | 3 | yes | yes | yes |
| 256x192 QP 20, 8-bit | 3 | yes | yes | yes |
| 256x192 QP 100, 8-bit | 3 | yes | yes | yes |
| 256x192 QP 140, 8-bit | 3 | yes | yes | yes |
| 256x192 QP 100, 10-bit | 3 | yes | yes | yes |
| 64x64 QP 100, 4:2:2, 8-bit | 3 | yes | yes | yes |
| 64x64 QP 100, 4:4:4, 8-bit | 3 | yes | yes | yes |

Because the comparison covers inter frames 2 and 3, it also proves that the
grain-on output did not leak into the DPB: those frames decode exactly as the
encoder's clean reconstruction.

### AVM `--film-grain-test=1` stream

`avmenc` defaults enable tools outside this decoder's subset (SDP, extended
SDP, FSC, tx-partition splitting, transposed DIP, and others), so the grain
stream was produced with those unrelated tools disabled:

```bash
avmenc --passes=1 --cpu-used=0 --threads=1 --end-usage=q --qp=100 \
  --limit=1 --fps=30/1 --film-grain-test=1 \
  --enable-sdp=0 --enable-extended-sdp=0 --enable-fsc=0 --enable-cctx=0 \
  --enable-mhccp=0 --enable-ibp=0 --enable-tip=0 --enable-bawp=0 \
  --enable-cwp=0 --enable-opfl-refine=0 --enable-refinemv=0 \
  --enable-adaptive-mvd=0 --enable-mvd-sign-derive=0 --enable-flex-mvres=0 \
  --enable-global-motion=0 --enable-ist=0 --enable-inter-ist=0 \
  --enable-inter-ddt=0 --enable-tcq=0 --enable-parity-hiding=0 \
  --enable-restoration=0 --enable-skip-mode=0 --enable-interintra-comp=0 \
  --enable-interintra-wedge=0 --enable-smooth-interintra=0 \
  --enable-warp-causal=0 --enable-warp-extend=0 \
  --enable-six-param-warp-delta=0 --enable-masked-comp=0 \
  --enable-joint-mvd=0 --enable-imp-msk-bld=0 --enable-mv-traj=0 \
  --enable-ref-frame-mvs=0 --reduced-tx-type-set=2 --enable-cfl-intra=0 \
  --enable-tx-partition=0 --enable-cdef=0 --enable-gdf=0 --enable-ccso=0 \
  --enable-deblocking=0 --enable-lf-sub-pu=0 --enable-intra-dip=0 \
  --enable-mrls=0 --enable-palette=0 --enable-intrabc=0 \
  --enable-intra-edge-filter=0 --enable-smooth-intra=0 \
  -o hgrad_avm.ivf hgrad.y4m
```

`hgrad.y4m` is a 64x64 horizontal luma gradient (U/V 128). The AVM test model
uses `ar_coeff_lag = 2`, 14 luma points and explicit Cb/Cr points,
`scaling_shift = 11`, `ar_coeff_shift = 8`, `overlap_flag = 0`,
`clip_to_restricted_range = 1`, `mc_identity = 0`, `block_size = 0`.

Result: our decoder's grain-on and grain-off output match
`dav2d --filmgrain 1` and `--filmgrain 0` byte-for-byte, and the two outputs
differ, so the stream really carries grain.

Caveat: on the 64x64 synthetic source used by the encoder matrix, this
decoder's **clean** intra reconstruction already differs from dav2d/avmdec in
blocks after the first 16 luma columns. That is a pre-existing intra
reconstruction difference unrelated to unit 12 (both references agree with each
other). A gradient source avoids that path, so the AVM case isolates film grain
and is still byte-exact. A 2-frame AVM inter stream currently fails in inter
coefficient syntax (`decoded EOB ... outside the block`), also unrelated to
grain; the AVM case is therefore one key frame.

## Limits and risks

- `mc_identity`: the specification uses `fg_mc_identity` to select the chroma
  clip range, and this implementation follows it. dav2d hardcodes `is_id = 0`
  in `fg_apply_tmpl.c`, so a stream that sets `mc_identity = 1` with
  `clip_to_restricted_range = 1` could differ from dav2d. All tested streams
  (our encoder and AVM test vector 1) use `mc_identity = 0`.
- `show_existing_frame` output is not implemented by this decoder at all, so
  grain on a shown-existing picture is not covered. `film_grain_config()` is
  parsed for coded frames only, matching the current decoder scope.
- 12-bit and I400 use the generic synthesis path but are not in the targeted
  matrix above.
- The shared synthesis allocates full noise stripes and a noise image per plane;
  this is the straightforward §7.21.7.5 transcription, not the two-line-buffer
  reference implementation.
- Nothing was committed, pushed, tagged, or published.
