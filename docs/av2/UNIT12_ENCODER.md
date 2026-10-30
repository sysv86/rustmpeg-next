# Unit 12 encoder: AV2 film grain signalling and synthesis

Status: encoder/shared half complete. Nothing was committed. The decoder half is
agent 2's `dec/*` change; this document covers the shared synthesis module, the
film grain OBU writer, encoder-side estimation, and the encoder verification.

## Scope and normative references

AV2 v1.0.0 film grain synthesis is split between an OBU that carries reusable
models and per-frame parameters in the frame header:

- `film_grain_obu()` and `film_grain_model()` — §5.14 and §5.18.10.2.
- `film_grain_config()` in the frame header — §5.18.10.1.
- `fgm_update_flags`, `fgm_chroma_idc`, `save_grain_model()` — §6.13.
- `apply_grain`, `fgm_id`, `grain_seed`, `load_grain_model()` — §6.17.10.1.
- Model semantics (points, AR coefficients, overlap, clipping, block size) —
  §6.17.10.2.
- Film grain OBU availability before use — §7.3.8.8.
- Output process: grain is applied to `OutY/OutU/OutV` only, after the output
  picture is prepared, and never to the reference frame store — §7.21.1 and
  §7.21.2.
- Synthesis: random number process §7.21.7.2, generate grain §7.21.7.3,
  scaling lookup §7.21.7.4, add noise §7.21.7.5.

The 2048-entry `Gaussian_Sequence` is the normative table printed in
§7.21.7.3; it is transcribed from the specification, not from a decoder.

Reference behavior was read in dav2d (read-only, no code or tables copied):
`src/obu.c` (`parse_fgm_hdr`), `src/film_grain*`, `src/fg_apply*`,
`src/filmgrain_tmpl.c`, and `src/decode.c` (`f->cur.p.fgm`). The reference
implementation confirms that `fgm_chroma_idc` must match the sequence layout and
that `check_trailing_bits()` follows the model payload.

## Implementation

### Shared module: `av2/filmgrain.rs`

- `get_random_number()` is the exact 16-bit LFSR of §7.21.7.2.
- `generate_grain()` builds the 82x73 luma and 44x38 / 82x73 chroma templates,
  applies the lag-1..3 auto-regressive filter with the luma term for chroma, and
  clips to `GrainMin..GrainMax` (§7.21.7.3).
- `ScalingLuts` performs the piecewise-linear interpolation of §7.21.7.4 and the
  bit-depth-aware `scale_lut()` of §7.21.7.5.
- `apply_film_grain()` implements the 32-line noise stripes, horizontal and
  vertical overlap blending, the `lumaSize` 16/32 block-size switch, chroma
  scaling from luma or from `uv_mult`/`uv_luma_mult`/`uv_offset`, restricted-range
  clipping, and the final luma pass after chroma (§7.21.7.5).
- `write_film_grain_obu()` writes `fgm_update_flags`, `fgm_chroma_idc`, one
  `film_grain_model()`, and the OBU trailing bits.
- `estimate_film_grain_params()` is the encoder-side estimator: it measures the
  source/reconstruction residual per intensity bin, builds five luma scaling
  points at 0/64/128/192/255, fits four lag-1 AR coefficients by regularized
  normal equations, and signals `chroma_scaling_from_luma` with chroma AR
  coefficients fitted from the chroma residual.

### Encoder integration

- `frame.rs` estimates the model after the in-loop filters, signals `apply_grain`
  with a deterministic non-zero 16-bit seed, and keeps the DPB frame clean.
- `mod.rs` emits the `OBU_FILM_GRAIN` in the same temporal unit, after the
  sequence header and before the frame OBU, so the model is available when the
  frame header references it (§7.3.8.8). The model uses slot 0 and is updated on
  every frame that has an active model.
- `EncodedPacket` now carries the signalled `FhFilmGrain`, so callers can apply
  the output-side synthesis without re-parsing the bitstream.
- `examples/av2_encode.rs` takes an optional 27th argument `filmgrain 0|1`. When
  set, the encoder signals grain and the example applies
  `filmgrain::apply_film_grain()` to the reconstruction it writes; the IVF
  reference frames remain clean. `-` in the input/rec positions now means
  "none".

`ToolConfig::enable_film_grain` remains off by default.

## Verification commands and results

Environment: `$ORACLE/dav2d` with `--threads 1
--framedelay 1 --strict 1`, release build, 3 synthetic frames per case.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p rustmpeg-video av2
python3 scripts/check-licenses.py
python3 $ORACLE/check_av2.py quick
python3 $ORACLE/check_av2.py
cargo build --release -p rustmpeg-video --examples
```

Results:

- `cargo fmt --check`: clean.
- `cargo clippy ... -D warnings`: clean; only the pre-existing
  unused-workspace-dependency manifest warnings remain.
- `cargo test -p rustmpeg-video av2`: 124 passed, 0 failed, 2 ignored.
- `check-licenses.py`: `license audit: OK (all packages permissive, no
  copyleft)`.
- `check_av2.py quick`: 9/9 OK.
- `check_av2.py` full matrix: 84/84 OK with film grain off (the default).

### Targeted film grain matrix

Driver: `$WORK/u12_verify.py`. For each row the encoder is run twice on
the same source (grain off and grain on); `dav2d --filmgrain 0` and
`dav2d --filmgrain 1` decode the grain-on stream. Y4M frame payloads are
compared, not headers (`dav2d` writes `C420`, the example writes `C420jpeg`).

| Case | Models | Grain changes pixels | Clean rec == dav2d fg0 | Grained rec == dav2d fg1 |
| :-- | --: | :-- | :-- | :-- |
| 64x64 QP 20, 8-bit | 3/3 | yes | yes | yes |
| 64x64 QP 100, 8-bit | 3/3 | yes | yes | yes |
| 64x64 QP 140, 8-bit | 3/3 | yes | yes | yes |
| 256x192 QP 20, 8-bit | 3/3 | yes | yes | yes |
| 256x192 QP 100, 8-bit | 3/3 | yes | yes | yes |
| 256x192 QP 140, 8-bit | 3/3 | yes | yes | yes |
| 256x192 QP 100, 10-bit | 3/3 | yes | yes | yes |

Additional layouts at 64x64 QP 100, 8-bit, 3 frames: 4:2:2 and 4:4:4 both
decode in `dav2d` and match the encoder output with grain on and off.

Proof that grain is active rather than a zero model (from the encoder log):

```text
film grain: seed 40503, y points [5, 0, 0], ar lag 1, ar coeffs y [-1, 2, 5, 6]
film grain: seed 39653, y points [5, 0, 0], ar lag 1, ar coeffs y [23, -24, -2, 59]
film grain: seed 38803, y points [5, 0, 0], ar lag 1, ar coeffs y [-20, 43, -2, 27]
```

Seeds and AR coefficients are non-zero, `ar_coeff_lag = 1`, and the grained
output differs from the clean output on every frame of the matrix (for example
64x64 QP 20: 4515 / 6085 / 5593 differing bytes on frames 0 / 1 / 2).

### AVM film-grain stream

`avmenc` supports `--film-grain-test=1`. A 64x64, 2-frame stream was generated
from a clean Y4M:

```bash
avmenc --passes=1 --cpu-used=9 --threads=1 --end-usage=q --qp=100 \
  --limit=2 --fps=30/1 --film-grain-test=1 -o avm_fg.ivf input.y4m
```

`dav2d --filmgrain 1` and `--filmgrain 0` both decode it with rc 0, and the two
outputs differ, so the stream really carries grain. Our decoder still rejects
`film grain synthesis` at this point; the decoder half is agent 2's change. Once
it lands, the same stream can be compared against `dav2d --filmgrain 1` using
the shared `apply_film_grain()` implementation.

## RD and speed impact

Measured on the same machine, release build, one thread:

- Bitrate, 256x192 QP 100, 3 frames: 7018 bytes grain off → 7119 bytes grain on
  (+1.4%, mostly the three FGM OBUs and the extra frame-header bits).
- Encoder-only wall time, 1920x1080 QP 100, 1 frame: 98 ms grain off → 99 ms
  grain on. Estimation (residual scan, chroma residual, AR fit) is effectively
  free at this size; grain is off by default, so existing streams are unaffected.
- Encoder-only wall time, 256x192 QP 100, 3 frames: 10 ms in both cases
  (including process startup).
- Output-side synthesis (only when the example writes a grained Y4M) costs about
  35 ms extra on 1920x1080: 99 ms encode-only → 134 ms encode + grained output.

## Limits and risks

- The estimator models quantisation residual, not true film-grain noise. It is a
  variance/correlation fit, not a noise generator matched to a film stock.
- Chroma scaling always uses `chroma_scaling_from_luma`; independent Cb/Cr
  point curves and `uv_mult`/`uv_offset` estimation are not implemented.
- One model slot (slot 0) is updated per frame. Reusing a stored model across
  frames and multiple slots is valid syntax but not used by the encoder.
- The shared synthesis allocates full noise stripes and a noise image per plane.
  This is the straightforward transcription of §7.21.7.5; the specification's
  2-line-buffer implementation is not used yet.
- 12-bit and I400 are implemented in the generic path but not covered by the
  targeted matrix above.
- The decoder-side application of `apply_film_grain()` is owned by agent 2 and
  is not part of this half.
- Nothing was committed, pushed, tagged, or published.
