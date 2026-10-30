# APV (RFC 9924)

Status: **implemented (experimental, CPU)**.  `rustmpeg-video::apv` provides an
intra-only APV encoder and decoder written from RFC 9924.  No OpenAPV, FFmpeg, or
other implementation source is transcribed; FFmpeg and the OpenAPV reference
bitstream are used only as external oracles.

## Scope

- Profiles: 400-10 (99), 422-10 (33), 422-12 (44), 444-10 (55), 444-12 (66),
  4444-10 (77), 4444-12 (88).
- Chroma layouts: 4:0:0, 4:2:2, 4:4:4, 4:4:4:4.
- Bit depth: 10 and 12 bits, matching the RFC profiles.  The syntax field allows
  a wider range, but there is no profile above 12 bits.
- Tiles of 16x16 macroblocks, 8x8 transform blocks, arbitrary frame cropping,
  and up to 20x20 tiles per frame.
- Differential DC coding, run-level AC coding, the RFC variable-length code,
  the normative 8x8 integer transform, scaling matrices (`use_q_matrix` on
  decode), H.273 colour description, and `capture_time_distance`.
- Decoder input: bare access units, raw APV bitstreams, and reference bitstreams
  from OpenAPV.
- Encoder output: access units, raw bitstreams, and ISOBMFF `apvC` decoder
  configuration records.

Not implemented yet:

- Non-primary PBUs (alpha, depth, preview) are skipped; only the primary frame
  is returned.
- Metadata PBUs and `au_info` are skipped.
- APV companding is rejected (the RFC reserves the field; OpenAPV writes it as
  zero).
- Level luma-rate limits require a frame rate and are not enforced; dimensions,
  tile geometry, profile/chroma/bit-depth, and the syntax limits are enforced.
- Encoder: fixed QP, one tile size per frame, default quantization matrices, no
  rate control, no alpha/depth/preview, no metadata PBUs.
- No GPU stage.  Entropy coding and block decisions are serial; a whole-frame
  transform kernel would need a measured >=10% win including transfer before it
  could join the scheduler.

## API

```rust
use rustmpeg_video::apv::{ApvChromaFormat, ApvConfig, ApvDecoder, ApvEncoder, ApvFrame};

let mut frame = ApvFrame::new(ApvChromaFormat::Yuv422, 10, 1920, 1080)?;
// fill frame.planes[0..3] with 10-bit samples

let encoder = ApvEncoder::new(ApvConfig { qp: 24, ..ApvConfig::default() })?;
// Direct access unit (Matroska block, test vector, direct decoder input):
let access_unit = encoder.encode_access_unit(&frame)?;
// Raw bitstream: 4-byte au_size followed by the access unit.
let raw = encoder.encode_raw_bitstream(&frame)?;
// ISOBMFF: one apv1 sample is a raw bitstream access unit, i.e. `raw` above.
let apvc = encoder.isobmff_config_record(&frame)?;

let decoder = ApvDecoder::new();
let decoded = decoder.decode_access_unit(&access_unit)?;
let frames = decoder.decode_raw_bitstream(&raw)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

An ISOBMFF `apv1` track stores each sample exactly as
`raw_bitstream_access_unit` (a four-byte length followed by the access unit).
The `apvC` box payload is the record returned by
`ApvEncoder::isobmff_config_record`; `rustmpeg-format` writes it through
`SampleEntry::new(*b"apv1", Some(*b"apvC"), config)`.

## Implementation

`crates/rustmpeg-video/src/apv/`:

```text
apv/
  mod.rs        public frame/config/error types, profiles, limits
  bits.rs       MSB-first bit reader/writer
  vlc.rs        RFC 9924 h(v) variable-length code
  transform.rs  8x8 transform, scaling/quantization, zig-zag scan
  syntax.rs     frame header, tile info/header, colour mapping
  decoder.rs    AU/PBU/frame/tile/component decoding
  encoder.rs    fixed-QP encoder and ISOBMFF configuration record
```

## Verification

- `crates/rustmpeg-video/tests/apv.rs` covers all layouts, 10/12-bit, odd frame
  edges, multiple tiles, negative levels, color metadata, and truncation/fuzz
  robustness.
- OpenAPV reference vector `qp_F.apv` (320x240 4:2:2 12-bit, two frames) decodes
  **bit-exactly** against FFmpeg n9.0.2's APV decoder.  Its per-plane checksums
  are asserted in the test suite.
- A second OpenAPV vector, `qp_D.apv` (3840x2160 4:2:2 10-bit, three frames),
  also decodes bit-exactly against FFmpeg; it is not committed because of its
  size.
- Encoder output for 4:0:0, 4:2:2, 4:4:4, and 4:4:4:4 (10 and 12 bit) was decoded
  by FFmpeg's APV decoder and compared byte-for-byte with the in-tree decoder;
  all cases matched exactly.
- An MP4 muxed through `rustmpeg-format` with `apv1`/`apvC` was decoded by
  FFmpeg, confirming the configuration record and the raw-sample framing.

## References

- RFC 9924, *Advanced Professional Video (APV) Codec*: Sections 4.2-4.4
  (formats, tiling, scan), 5.3 (syntax), 6 (decoding), 7.1/7.2 (VLC), 9
  (profiles/levels), Appendix A (raw bitstream).
- APV ISOBMFF binding: OpenAPV `readme/apv_isobmff.md` (`apv1` sample entry,
  `apvC` decoder configuration, sample format).
- H.273 for colour code points.

## Provenance

The codec implementation is original Apache-2.0 work from RFC 9924.  The
transform matrix, level scale, zig-zag table, and profile/level values are
normative RFC data.  The committed `qp_F.apv` test vector is BSD-3-Clause data
from OpenAPV; see `tests/data/apv/README.md`, `THIRD_PARTY.md`, and `NOTICE`.
