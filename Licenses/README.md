# Licenses

Full license texts for `rustmpeg-next` and for the third-party material that is
included in this repository.

`LICENSE`, `NOTICE`, `THIRD_PARTY.md`, and `LICENSING.md` at the repository root
remain **canonical**. This directory is a convenience mirror of the license
texts only: it does not add, remove, or reinterpret any term, and it does not
replace the root documents. If anything here disagrees with a root document, the
root document wins.

## Project license

The project itself — original code, tests, examples, and documentation — is
licensed under the Apache License 2.0. See [`Apache-2.0.md`](Apache-2.0.md) and
the root [`LICENSE`](../LICENSE).

## Component map

| Component | License file | Scope in this repository |
| :-- | :-- | :-- |
| `rustmpeg-next` (project) | [`Apache-2.0.md`](Apache-2.0.md) | All original code, tests, examples, and documentation. |
| VTM 24.0 | [`BSD-3-Clause.md`](BSD-3-Clause.md) | VVC CABAC, prediction, DCT/DST/LFNST, residual coding, in-loop filters, and high-level syntax derived from VTM in `crates/rustmpeg-video/src/vvc/`. |
| dav2d | [`BSD-2-Clause.md`](BSD-2-Clause.md) | AV2 conformance data, coefficient CDFs, quantiser/transform details, and syntax ordering used by the AV2 encoder in `crates/rustmpeg-video/src/av2/`. |
| OpenEXR 3.5.2 OpenEXRCore | [`BSD-3-Clause.md`](BSD-3-Clause.md) | Container/chunk layout, validation limits, DWA rules and perceptual tables, DCT constants, B44 conversions, PIZ/Huffman conventions, and the Zstd-EXR container used by `crates/rustmpeg-image/src/exr/`. |
| liblc3 | [`Apache-2.0.md`](Apache-2.0.md) | LC3 normative tables only (`crates/rustmpeg-audio/src/lc3/tables.rs`, `enc_tables.rs`). Copyright 2022 Google LLC; the Apache-2.0 text is the same license text. |
| OpenAPV | [`BSD-3-Clause.md`](BSD-3-Clause.md) | The single APV conformance vector `crates/rustmpeg-video/tests/data/apv/qp_F.apv`. No OpenAPV source is copied, compiled, or linked. |
| rusty-opus / RFC 6716 Appendix A | [`BSD-3-Clause.md`](BSD-3-Clause.md) | SILK constant tables only (`crates/rustmpeg-audio/src/opus/silk/`). Normative RFC 6716 data, cross-checked against `rusty-opus`. |

The copyright holder for each entry is recorded in
[`THIRD_PARTY.md`](../THIRD_PARTY.md) and [`NOTICE`](../NOTICE); the texts in this
directory are reproduced without modification.

## Not included here

FFmpeg, libopus, `flac`/libFLAC, and liblc3 are used only as external black-box
oracles for validation. No source code from them is linked, vendored, or copied,
so no license text for them is required in this directory.

## Audit

License compliance is checked with:

```bash
python3 scripts/check-licenses.py
```

The latest result is recorded in [`docs/LICENSE-AUDIT.md`](../docs/LICENSE-AUDIT.md).
