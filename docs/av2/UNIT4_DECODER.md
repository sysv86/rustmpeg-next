# Unit 4 decoder: AV2 loop restoration

Status: decoder half complete. Nothing was committed. The encoder/shared half is
recorded in [`UNIT4_ENCODER.md`](UNIT4_ENCODER.md); this document covers the
decoder-side frame/unit signalling, filter application, temporal reuse, the
removed gates, and byte-exact verification against `dav2d`.

## Scope and normative references

AV2 v1.0.0 loop restoration is defined by:

- `enable_restoration`, `lr_tools_disable`, `lr_tools_uv_present` —
  §5.4.9 / §6.4.9 (`sequence_filter_config`).
- `lr_params()` — §5.18.7.11; `FrameRestorationType`, `frame_filters_on`,
  `temporal_pred_flag`, `rst_ref_pic_idx`, `num_filter_classes_idx`, and the
  `LoopRestorationSize` half/max/quarter flags.
- `read_lr()` — §5.20.10.4; `count_units_in_frame()`.
- `read_lr_unit()` — §5.20.10.5; per-unit `use_pc_wiener`, `use_wiener_ns`,
  and the `flex_restoration_type` chain for `RESTORE_SWITCHABLE`.
- `read_wienerns_filter()` — §5.20.10.6, including the group/ref/class
  syntax and `Wiener_Ns_Taps_Present`, `Wiener_Ns_Taps_Min`,
  `Wiener_Ns_Taps_K`.
- `FrameRestorationType` semantics — §6.17.7.7.
- Loop restoration process — §7.20; source sampling (§7.20.2), non-separable
  Wiener including luma cross-component chroma taps (§7.20.3), and
  pixel-classified Wiener (§7.20.4).
- Filter order — §7.20 places loop restoration after CDEF/CCSO and before GDF;
  the restored picture is the reference picture.

### No self-guided restoration in AV2 v1.0.0

§6.17.7.7 defines `FrameRestorationType` with exactly four values:

| Value | Name |
| --: | :-- |
| 0 | `RESTORE_NONE` |
| 1 | `RESTORE_PC_WIENER` |
| 2 | `RESTORE_WIENER_NONSEP` |
| 3 | `RESTORE_SWITCHABLE` |

The syntax in §5.18.7.11 (`indexToTool`, `RESTORE_SWITCHABLE_TYPES = 4`) and
§5.20.10.5 (`flex_restoration_type` over tools `0..RESTORE_SWITCHABLE_TYPES-1`)
contains no `Sgrproj`/self-guided branch, and the specification has no
self-guided restoration process. `apply_sgrproj_plane` remains an unsignalled
helper in `av2/restoration.rs` and is never reached from a bitstream.

## Implementation

### Frame-level signalling (`dec/hls.rs`)

`lr_params()` is parsed when `enable_restoration` is set and the frame is not
coded lossless (`CodedLossless`, §5.18.7.11). The decoder now reads:

- per-plane `FrameRestorationType` under `rst_disable_mask`;
- `frame_filters_on` for `RESTORE_WIENER_NONSEP` and `RESTORE_SWITCHABLE`;
- `temporal_pred_flag`, `rst_ref_pic_idx` (with bounds checking), and
  `num_filter_classes_idx`;
- the `LoopRestorationSize` flags;
- the frame-level NS filter bank through
  `read_wienerns_frame_filter()`.

Temporal reuse is resolved during header parsing. `RefFrameInfo` (the decoder's
per-slot reference metadata) now carries the resolved `[NSWienerPlane; 3]`, so
`temporal_pred_flag = 1` copies `num_classes_idx`, `num_classes`, and the filter
bank from `ref_frame_idx[rst_ref_pic_idx]`. For chroma, if the selected
reference has no frame filters for that plane, the other chroma plane is used,
as required by §5.18.7.11 (`refPlane = plane == 1 ? 2 : 1`).

`get_bits_subexp_u()` was added to `dec/bitio.rs` as the exact inverse of the
encoder's `put_bits_subexp_u()` and of the normative sub-exponential code used
by the coefficient and reference-index syntax.

### Unit-level signalling (`dec/frame.rs`)

`decode_lr()` fills a per-plane `Vec<u8>` of restoration types instead of a
boolean PC-only map:

- `RESTORE_PC_WIENER`: `use_pc_wiener` selects PC or none;
- `RESTORE_WIENER_NONSEP`: `use_wiener_ns` selects NS or none;
- `RESTORE_SWITCHABLE`: the first `flex_restoration_type` selects
  `RESTORE_NONE`, the second selects `RESTORE_PC_WIENER` vs
  `RESTORE_WIENER_NONSEP`, matching the `dav2d` symbol order.

Unit counts and positions use `count_units_in_frame(unitSize, frameSize)` on
the MI grid, with chroma subsampling and the tile offsets, as in §5.20.10.4.

### Filter application

`decode_frame()` applies restoration after deblock/CDEF/CCSO and before GDF:

1. PC Wiener over the units selected as `RESTORE_PC_WIENER`, using the
   pretrained class LUT and §7.20.4 feature/`qval` path.
2. NS Wiener over the units selected as `RESTORE_WIENER_NONSEP`, for luma and
   for U/V. Chroma reads 12 self taps plus 12 luma cross taps and uses
   `get_luma_sample()` with the `Wiener_Filters_420` 4:2:0/4:2:2 mapping
   (§7.20.3).
3. GDF on luma, from the same CDEF input.

Both filters sample the CDEF/CCSO output inside the current 64-line stripe
(offset by 8 luma lines) and the deblocked frame outside it, clamped to two
lines above/below the stripe (§7.20.2). The reference picture stored in the DPB
is the restored picture.

### Gates removed and remaining typed limits

The old `validate_tools()` gate that rejected every NS Wiener, switchable and
chroma restoration frame is gone. Two narrow typed limits remain, and both are
reported as `Unsupported` rather than silently misdecoded:

- `frame_filters_on = 0` (per-unit Wiener NS filter bank syntax in
  `read_lr_unit()`); our encoder always signals frame-level filters.
- frame-level NS filter banks with `num_classes > 1`; the filter application
  path uses class 0 only. Our encoder signals one class.

## Verification

Environment: release build, `dav2d` revision `bff8e5c`
(`$ORACLE/dav2d`) with
`--threads 1 --framedelay 1 --strict 1 -q`, synthetic 3-frame sequences.
For every case the three reconstructions are compared on Y4M frame payloads:
our decoder, the encoder reconstruction, and `dav2d`.

| Mode | Size | QP | Bit depth | Tiles | Cases |
| :-- | :-- | --: | --: | --: | --: |
| `pc` (PC Wiener, luma) | 64x64, 256x192 | 20/100/140 | 8 | 1 | 6 |
| `ns` (NS Wiener, luma) | 64x64, 256x192 | 20/100/140 | 8 | 1 | 6 |
| `switch` (per-unit PC/NS) | 64x64, 256x192 | 20/100/140 | 8 | 1 | 6 |
| `chroma` (NS luma+U+V) | 64x64, 256x192 | 20/100/140 | 8 | 1 | 6 |
| `pc`, `ns`, `switch`, `chroma` | 64x64 | 100 | 10 | 1 | 4 |
| `chroma` | 256x192 | 100 | 8 | 2 | 1 |

Result: **29/29 OK** — `our decoder == encoder reconstruction == dav2d` on
every frame. The in-tree ignored harness
`av2::dec::tests::av2_restoration_decoder_dav2d_byte_exact` covers 28 of these
cases (it also runs chroma for PC and switchable modes):

```bash
cargo test -p rustmpeg-video av2_restoration_decoder_dav2d -- --ignored
```

The non-ignored test
`av2::dec::tests::test_decode_matches_encoder_recon_with_restoration` covers
PC/NS/switchable, chroma, 10-bit and a 2-tile chroma case on every run.

## Gates

| Command | Result |
| :-- | :-- |
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | rc 0; only the pre-existing unused-workspace-dependency manifest warnings |
| `cargo test -p rustmpeg-video av2` | 140 passed, 0 failed, 5 ignored |
| `cargo test -p rustmpeg-video av2_restoration_decoder_dav2d -- --ignored` | 1 passed, 0 failed |
| `python3 scripts/check-licenses.py` | `license audit: OK (all packages permissive, no copyleft)` |
| `$ORACLE/vvc-conf/run_conformance.sh` | rc 0, 25/25 cases (50/50 across VTM and VVdeC) |

## Limits and risks

- Per-unit NS filter banks (`frame_filters_on = 0`) and multi-class frame
  filters are rejected with a typed error, not decoded.
- The `dav2d` frame-filter parser does not consume `wiener_ns_uv_sym`; the
  decoder follows the oracle and the encoder, so a stream that relies on the
  AVM symmetric-chroma syntax would diverge.
- No GPU stage was added: loop restoration is a whole-frame CPU filter in this
  change, and the 10% rule was not measured for a GPU path.
- Nothing was committed, pushed, tagged, or published.
