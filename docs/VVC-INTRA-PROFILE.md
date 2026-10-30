# VVC intra profile v1

Status: **complete for the defined subset**. This is not a general H.266/VVC codec.
It is a conformant intra-only profile implemented by `rustmpeg-video`, with a
matching strict decoder. Full VVC remains a separate, much larger effort (see
[Beyond the profile](#beyond-the-profile)).

## Normative scope

| Tool | Value |
| :-- | :-- |
| Profile / tier / level | Main 10 signalling (`general_profile_idc = 1`), Main tier, Annex A level selected from `MaxLumaPs` |
| Chroma / bit depth | 4:2:0, 8-bit, BT.709 limited range, signalled in the VUI |
| Picture type | Every access unit is `IDR_N_LP`; POC LSB is 0 |
| CTU / CU / TU | CTU 32x32, min CU 32x32, one CU == one TU per CTU; quadtree and MTT disabled |
| Slice / tile | One rectangular slice, one tile (`pps_no_pic_partition_flag = 1`), no WPP |
| Intra prediction | PLANAR or DC per CTU, selected by SAD; DM chroma mode; PDPC applied |
| Residual coding | QP 4: transform skip for every component with normative `residual_ts_coding()`. QP > 4: DCT-2, DST-7 or DCT-8 with regular `residual_coding()` |
| Quantisation | `SliceQpY` 4..=63; QP 4 is lossless for 8-bit transform skip |
| In-loop filters | None: deblocking disabled, SAO/ALF/CCALF/LMCS off |
| Transforms | Explicit intra MTS (DST-7/DST-7, DCT-8/DST-7, DST-7/DCT-8, DCT-8/DCT-8) and 8x8 LFNST, selected per luma block by reconstruction SSE; MTS and LFNST are mutually exclusive. Chroma stays DCT-2 |
| Other tools | MIP, ISP, BDPCM, palette, IBC, ACT, CCLM, dependent quantisation, sign hiding, scaling lists, sub-pictures, chroma-QP offsets: off |
| High-level syntax | AUD, VPS, SPS, PPS, PH, SH; in-band parameter sets on every access unit; no OPI/DCI/APS/SEI |
| NAL / transport | Annex-B, 2-byte VVC NAL headers, `nuh_layer_id = 0`, `nuh_temporal_id_plus1 = 1`; no container muxing |

The picture is padded to a 32x32 CTU grid and cropped back by the SPS conformance
window. Visible dimensions must be even: 4:2:0 conformance-window offsets are
expressed in chroma samples, so odd luma sizes cannot be represented exactly.

## Access unit layout

```text
AUD (20)  VPS (14)  SPS (15)  PPS (16)  PH (19)  IDR_N_LP VCL (8)
```

Parameter sets and the picture header are repeated on every IDR, so an Annex-B
file is self-contained. The slice header uses
`sh_picture_header_in_slice_header_flag = 0`, carries `sh_qp_delta`, sets
`sh_ts_residual_coding_disabled_flag = 0`, and ends with `byte_alignment()`.

## Encoder API

```rust
use rustmpeg_video::vvc::{VvcEncoder, VvcEncoderConfig};

let mut encoder = VvcEncoder::new(VvcEncoderConfig {
    width: 1920,
    height: 1080,
    qp: 26,
})?;

let access_unit = encoder.encode_i420(&frame)?; // or encode_rgba(&rgba)
// encode_rgba_with(&rgba, &accelerator) uses the same CPU/GPU seam and is bit-exact.
// access_unit.data is one Annex-B access unit; access_unit.keyframe == true
encoder.finish();
```

- Input: `I420Frame` with exact 4:2:0 plane sizes, or RGBA8 of `width * height * 4`
  bytes (BT.709 limited-range conversion, alpha not coded).
- Output: one `EncodedAccessUnit` per frame, always a keyframe.
- Errors: `VvcError::{InvalidConfig, FrameDimensions, PlaneSize, LimitExceeded,
  Bitstream, Unsupported, EncoderFinished}`.
- Limits: even dimensions in `2..=16384`, at most 35,651,584 luma samples
  (level 6.2 `MaxLumaPs`).

## Decoder contract

`decode_annexb_to_i420` accepts only streams produced by this encoder:

- SPS/PPS are keyed by ID; conflicting duplicate sets are rejected.
- PPS must reference a known SPS.
- PH must reference a known PPS and have
  `ph_gdr_or_irap_pic_flag = 1`, `ph_non_ref_pic_flag = 0`,
  `ph_gdr_pic_flag = 0`, `ph_inter_slice_allowed_flag = 0`, POC LSB 0.
- Exactly one `IDR_N_LP` VCL NAL per access unit; layer 0 and temporal sub-layer 0.
- Unsupported prediction, transform, chroma format, bit depth, CTU size, or
  partition syntax is rejected with `VvcError::Unsupported`.

Foreign conformant streams are not decoded. The decoder is a conformance and
round-trip tool for the profile above, not a general VVC decoder.

## Conformance evidence

Byte-exact reconstruction was verified against three independent decoders:

| Decoder | QP 4 (lossless) | QP 26, 40, 50 (matches reconstruction) |
| :-- | :-- | :-- |
| FFmpeg n9.0.2 native VVC | yes | yes |
| vvdec 3.3 | yes | yes |
| VTM 24.0 `DecoderApp` | yes | yes |

Tested on 200x120 streams; QP 40 and QP 50 exercise explicit MTS and LFNST.
The in-crate suite has 84 tests covering syntax, CABAC, prediction, transform
skip, DCT-2/DST-7/DCT-8, LFNST, parameter-set linkage, malformed input, and
round-trips. `cargo clippy -- -D warnings` and `cargo fmt --check` pass.

## Beyond the profile

A full H.266/VVC implementation still requires:

1. Transform coding: 64-point and non-square transforms, dependent quantisation, sign hiding.
2. Partitioning: CU 4..128, QT/MTT, transform tree, dual tree, multi-slice, tiles,
   WPP, sub-pictures.
3. Inter prediction: P/B slices, DPB/RPL, motion estimation, merge/AMVP, affine,
   weighted prediction, BCW/CIIP/GPM.
4. In-loop filters: deblocking, SAO, ALF, CCALF, LMCS, LADF.
5. Full HLS and decoder: OPI/DCI/APS/SEI, arbitrary parameter sets, 10/12-bit,
   4:2:2/4:4:4, general bitstream decoding.
6. Container integration: ISOBMFF `vvc1`/`vvcC` and Matroska `V_MPEGI/ISO/VVC`.

These are not incremental switches; they require different coding-tree drivers and
additional CABAC context sets. The current intra profile is the stable base.

## Provenance

The profile is original rustmpeg-next Rust code. Its algorithms, integer
matrices, and context tables are derived from VTM 24.0 (BSD-3-Clause, Copyright
(c) 2010-2020 ITU/ISO/IEC) and the ITU-T H.266 specification. See
[THIRD_PARTY.md](../THIRD_PARTY.md).
