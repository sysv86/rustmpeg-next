# Licensing

## Project license

`rustmpeg-next` is licensed under the **Apache License, Version 2.0**
([`LICENSE`](LICENSE)). All original code, tests, examples, and documentation are
Apache-2.0.

There is **no CLA** and no copyright assignment. Contributions are accepted under
Apache-2.0 on an inbound = outbound basis. There is no separate GPL/LGPL build, no
optional copyleft component, and no runtime loading of copyleft code.

## Third-party code

Only permissive third-party code is included, and every component is recorded in
[`THIRD_PARTY.md`](THIRD_PARTY.md):

- **VTM 24.0** (BSD-3-Clause, © 2010-2020 ITU/ISO/IEC) — VVC algorithms, transforms,
  LFNST tables, residual coding, and high-level syntax used by the VVC module.
- **dav2d** (BSD-2-Clause, © 2018-2026 VideoLAN and dav2d authors) — AV2
  conformance data, coefficient CDFs, quantiser and transform details used by the
  AV2 encoder module.
- **liblc3** (Apache-2.0, Copyright 2022 Google LLC) — the normative LC3 tables in
  `tables.rs` and `enc_tables.rs`, transcribed into the LC3 encoder and decoder.
- **OpenEXR 3.5.2 OpenEXRCore** (BSD-3-Clause, © Contributors to the OpenEXR
  Project) — container/chunk layout, validation limits, DWA classification rules
  and perceptual tables, DCT constants, B44 conversion rules, PIZ/Huffman
  conventions, and the Zstd-EXR container used by `crates/rustmpeg-image/src/exr/`.

No FFmpeg, libav*, GStreamer, libopus, or libFLAC source is included. FFmpeg,
libopus, libFLAC, and liblc3 are used only as external behavioral oracles for
validation. Test fixtures are generated media data, not code, and do not inherit
the license of the tools that produced them.

The latest audit is recorded in [`docs/LICENSE-AUDIT.md`](docs/LICENSE-AUDIT.md).

### Package license expression

The workspace manifest defaults to `license = "Apache-2.0"` for original code.
`crates/rustmpeg-video` declares `license = "Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause"`
because that package contains VTM-derived (BSD-3-Clause) source; its AV2 files
that contain dav2d-derived data carry `Apache-2.0 AND BSD-2-Clause` SPDX headers.
`crates/rustmpeg-image` declares `license = "Apache-2.0 AND BSD-3-Clause"`
because its OpenEXR module contains OpenEXRCore-derived (BSD-3-Clause) material.
The BSD notices are reproduced in `THIRD_PARTY.md`. This is a description of the
combined package, not a relicense of the third-party portions.

## Dependency policy

Allowed licenses: Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, Zlib,
Unlicense, CC0-1.0, Unicode-3.0, and Apache-2.0 WITH LLVM-exception.

Forbidden: GPL, LGPL, AGPL, MPL, SSPL, BUSL, CDLA, and unknown or custom licenses.
Copyleft must not enter the dependency graph even as an unused alternative.

### Decision rule

Compatible permissive code is kept as-is with its notices; do not rewrite it just
to remove attribution. If a license is poisonous, either implement the
functionality from the normative specification or choose a permissive alternative.
Never copy, vendor, link, or transcribe copyleft code.

No `*-sys`/`*_sys` crate that compiles or bundles C/C++, and no dependency build
script that compiles C, C++, or assembly. Generated bindings to OS/API surfaces
without a native build (for example Windows bindings) are allowed. The GPU stack
uses pure-Rust `wgpu` and `naga`. `ring` was removed with `quinn` because it builds
C/assembly.

## Adding code

1. Keep the SPDX header on every source file current.
2. Record new third-party code in `THIRD_PARTY.md` before merging.
3. Do not copy or transcribe code from copyleft or unlisted sources. Reading
   FFmpeg source to understand behavior is allowed; copying is not.
4. Do not add a dependency without checking its license and build script.
5. Run the audit before publishing:

```bash
python3 scripts/check-licenses.py
```

## Why this policy

The project deliberately has a single, permissive license surface. It avoids the
FFmpeg situation where optional GPL components, build flags, and external library
combinations change the license of the resulting binary. There is one build, one
license, and recorded third-party notices.
