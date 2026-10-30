# APV test vectors

`qp_F.apv` is a 20 KiB reference bitstream from the OpenAPV project's public
test suite:

- Source: <https://github.com/AcademySoftwareFoundation/openapv/tree/main/test/bitstream/qp_F.apv>
- License: BSD-3-Clause (OpenAPV, Copyright (c) 2022 Samsung Electronics Co., Ltd.)
- SHA-256: `f32ecf1afde7c8aaa7ee02c9e8249347d8ff07001225bdf54bff8d7f45ce20dc`

The file is a raw APV bitstream containing two 320x240 4:2:2 12-bit frames.
`tests/apv.rs` decodes it with the in-tree decoder and checks the frame count,
geometry, chroma layout, bit depth, and per-plane FNV-1a checksums.  The decoded
planes were verified bit-exact against FFmpeg n9.0.2's APV decoder during
development; FFmpeg is only an external oracle and is not linked or bundled.
