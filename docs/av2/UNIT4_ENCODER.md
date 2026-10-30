# Unit 4 encoder: AV2 loop restoration

Status: encoder/shared half complete. Nothing was committed. The decoder's
acceptance of NS Wiener, switchable and chroma restoration is agent 2's `dec/*`
change; this document covers the shared filter module, the frame/tile syntax
writer, encoder-side filter selection, and the encoder verification against
`dav2d`.

## Scope and normative references

AV2 v1.0.0 loop restoration is defined by:

- `enable_restoration`, `lr_tools_disable`, `lr_tools_uv_present` —
  §5.4.9 / §6.4.9 (`sequence_filter_config`).
- `lr_params()` and `LoopRestorationSize` — §5.18.7.11 / §6.17.7.7.
- `read_lr()` and `read_lr_unit()` — §5.20.10.4 / §5.20.10.5.
- `read_wienerns_filter()` — §5.20.10.6, including
  `Wiener_Ns_Taps_Present`, `Wiener_Ns_Taps_Min`, `Wiener_Ns_Taps_K`.
- Non-separable Wiener process — §7.20.3, `Wiener_Ns_Config_Y`,
  `Wiener_Ns_Config_Uv`, `get_luma_sample`, `Wiener_Filters_420`.
- Pixel-classified Wiener process — §7.20.4, `Pc_Wiener_Config`,
  `Pc_Wiener_Normalizer`, `Pc_Wiener_Filters`, `get_box_features`,
  `get_qval_given_tskip`.
- Stripe/source sampling — §7.20.1 / §7.20.2 (`get_source_sample`).
- Output/reference order — §7.20 is applied after CDEF and CCSO; the restored
  frame is the reference frame.

`FrameRestorationType` has exactly four values: `RESTORE_NONE`,
`RESTORE_PC_WIENER`, `RESTORE_WIENER_NONSEP`, `RESTORE_SWITCHABLE`
(§6.17.7.7). **AV2 v1.0.0 has no self-guided (`Sgrproj`) restoration**, so the
release-unit wording "SGR-only" has no normative counterpart. The closest
normative case is PC-Wiener-only, which is what the `pc` row below verifies.
`apply_sgrproj_plane` remains an unsignalled helper and is not emitted.

Reference behavior was read in `dav2d` (`src/looprestoration_tmpl.c`,
`src/lr_apply_tmpl.c`, `src/obu.c`, `src/decode.c`) and AVM
(`av2/encoder/bitstream.c`, `av2/common/restoration.*`) without copying code.

## Implementation

### Shared module: `av2/restoration.rs`

- `apply_pc_wiener_normative` / `apply_pc_wiener_units` implement §7.20.4
  exactly: 6x6 box features over the 3x3 2x2-cell grid, per-feature
  normalisation, the `qval` approximation, the pretrained class LUT, the
  12 symmetric tap pairs, and the 7-bit rounding.
- `apply_ns_wiener_plane` implements §7.20.3 for luma (32 taps / 16
  coefficients) and chroma (12 self taps + 12 luma cross taps / 18
  coefficients), including `get_luma_sample` with the 4:2:0/4:2:2
  `Wiener_Filters_420` downsampling.
- `StripeSampler` implements §7.20.2: in-stripe samples come from the
  CDEF/CCSO output, out-of-stripe samples from the deblocked frame clamped to
  two lines above/below the stripe.
- `apply_encoder_restoration` applies exactly the filters that the bitstream
  signalled. It no longer performs a second, un-signalled SSE decision (the
  previous code could signal restoration units and then leave the encoder
  reconstruction unfiltered).
- `encoder_ns_luma_filter` / `encoder_ns_chroma_filter` are fixed,
  non-identity, in-range filters. The chroma filter also exercises the luma
  cross-component taps.

### Frame/tile integration (`frame.rs`, `obu.rs`, `mod.rs`)

- `ToolConfig` gains `restoration_mode` (`PcWiener`, `NsWiener`,
  `Switchable`) and `restoration_chroma`.
- Luma unit size is 64 samples for 64x64 superblocks, 128 for 128x128
  superblocks, and the syntax maximum for 256x256 superblocks. Chroma uses
  64 samples where the stripe-height requirement allows it.
- `lr_types` replaces the old `lr_units` boolean map, so per-unit switchable
  selection is signalled with the dav2d-compatible `rst_switchable` symbols
  (NONE, then PC vs NS).
- NS frame-level filters are signalled once (`frame_filters_on = 1`,
  `num_classes_idx = 0`). Inter frames use `temporal_pred_flag = 1` and copy
  the previous frame's filters, avoiding the filter-bank merge syntax.
- The sequence header enables both PC and NS Wiener (`rst_disable_mask =
  [0, 1]`), so `RESTORE_SWITCHABLE` is signallable; chroma PC is disabled by
  syntax.
- `examples/av2_encode.rs` accepts `rst` values `0`, `1`/`pc`, `ns`,
  `switch`, `chroma`, and `pcchroma`.

### dav2d / AVM frame-filter divergence (recorded, not hidden)

`dav2d` revision `bff8e5c` parses chroma frame-level NS filters differently
from AVM:

- it reads `num_filter_classes_idx` for chroma even though AVM's
  `max_num_classes(chroma) == 1`;
- its group-2 count for chroma is `n_filters - 1`, not zero, so it expects a
  group-selection bit;
- it does not read `wiener_ns_uv_sym` (source comment: "FIXME read sym bit"),
  while AVM does.

The encoder writes the syntax `dav2d` expects for chroma frame filters, which
is what the acceptance oracle decodes byte-exactly. `avmdec` accepts the luma
PC/NS/switchable streams but rejects the chroma-frame-filter stream because of
this reference-decoder divergence. This encoder does not emit the per-unit NS
filter-bank syntax that AVM uses for chroma; that remains a future
compatibility path.

## Verification commands and results

Environment: release build, one thread, `dav2d` `bff8e5c` with
`--threads 1 --framedelay 1 --strict 1 -q`, 3 synthetic frames per case.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p rustmpeg-video av2
cargo test -p rustmpeg-video restoration
python3 scripts/check-licenses.py
python3 $ORACLE/check_av2.py quick
cargo build --release -p rustmpeg-video --examples
```

Results:

- `cargo fmt --check`: clean.
- `cargo clippy --workspace --all-targets -- -D warnings`: rc 0; only the
  pre-existing unused-workspace-dependency manifest warnings remain.
- `cargo test -p rustmpeg-video av2`: 138 passed, 0 failed, 4 ignored.
- `cargo test -p rustmpeg-video restoration`: 9 passed, 0 failed.
- `check-licenses.py`: `license audit: OK (all packages permissive, no
  copyleft)`.
- `check_av2.py quick`: 9/9 OK (restoration off, the default).
- `cargo build --release -p rustmpeg-video --examples`: rc 0.

### Targeted restoration matrix

For each row the encoder writes `rec.y4m`; `dav2d` decodes the IVF and the Y4M
frame payloads are compared (headers differ only in `C420jpeg` vs `C420`).

| Mode | Size | QP | Frames | Bit depth | Encoder recon == dav2d |
| :-- | :-- | --: | --: | --: | :-- |
| `pc` (PC Wiener, luma) | 64x64 | 20/100/140 | 3 | 8 | yes |
| `pc` | 256x192 | 20/100/140 | 3 | 8 | yes |
| `ns` (NS Wiener, luma) | 64x64 | 20/100/140 | 3 | 8 | yes |
| `ns` | 256x192 | 20/100/140 | 3 | 8 | yes |
| `switch` (per-unit PC/NS) | 64x64 | 20/100/140 | 3 | 8 | yes |
| `switch` | 256x192 | 20/100/140 | 3 | 8 | yes |
| `chroma` (NS luma + U/V) | 64x64 | 20/100/140 | 3 | 8 | yes |
| `chroma` | 256x192 | 20/100/140 | 3 | 8 | yes |
| `pc`, `ns`, `switch`, `chroma` | 64x64 | 100 | 3 | 10 | yes |

Additional case: `chroma`, 256x192, QP 100, 8-bit, 2 tiles
(`log2_tile_cols = 1`) is also byte-exact.

Assertions for restoration activity and non-identity parameters:

- `av2::tests::restoration_unit_size_and_count_are_non_zero` checks that
  64x64 and 256x192 have non-zero restoration-unit counts for luma and both
  chroma planes.
- `av2::restoration::tests::test_encoder_ns_filters_are_non_identity_and_in_range`
  checks that the luma and chroma NS filters contain non-zero coefficients and
  every coefficient is inside `Wiener_Ns_Taps_Min..+K`.
- `av2::restoration::tests::test_ns_wiener_luma_changes_gradient` checks that
  the NS filter changes a non-constant gradient (it is not an identity path).
- The `switch` 256x192 case has 12 luma units alternating PC and NS, so both
  tools are present in the bitstream; the 64x64 frame has one unit and selects
  PC.

## RD/size impact

256x192, QP 100, 3 frames, IVF size including the sequence/frame OBUs:
restoration off 6986 bytes, PC luma 6976, NS luma 7026, switchable 6996,
NS luma+chroma 7041. The PC case is slightly smaller because the filtered
reference improves inter prediction enough to offset the unit bits; NS and
chroma add the frame-filter payload. This is a size observation, not an RDO
study.

## Limits and risks

- Chroma restoration uses frame-level NS filters. The per-unit NS filter bank
  and merge syntax (§5.20.10.6 with `frame_filters_on = 0`) are not emitted.
- The encoder uses fixed NS coefficients, not a least-squares/RDO fit per
  frame or unit. The filters are valid, in range, and non-identity, but they
  are not rate-distortion optimised.
- Inter frames reuse the reference frame's NS filters via
  `temporal_pred_flag`. Switching restoration parameters per inter frame is
  not emitted.
- AV2 v1.0.0 has no SGR tool; the `apply_sgrproj_plane` helper is unused and
  unsignalled.
- The `avmdec` divergence for chroma frame filters is described above. Luma
  PC/NS/switchable streams are accepted by both `dav2d` and `avmdec`.
- Nothing was committed, pushed, tagged, or published.
