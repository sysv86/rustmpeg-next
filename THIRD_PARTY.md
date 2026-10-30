# Third-party code and licenses

`rustmpeg-next` is licensed under Apache-2.0. Included third-party source is
limited to the VTM 24.0 reference software (BSD-3-Clause), dav2d conformance data
(BSD-2-Clause), the normative LC3 tables from liblc3 (Apache-2.0), OpenEXR 3.5.2
`OpenEXRCore` algorithms and tables (BSD-3-Clause), and one OpenAPV conformance
test vector (BSD-3-Clause). Those portions remain under their
original licenses; the project license does not override their terms.

Standalone copies of the license texts are mirrored in [`Licenses/`](Licenses/).
This file, [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`LICENSING.md`](LICENSING.md)
remain canonical; the mirror adds no terms and wins no argument.

The VVC codec's Rust implementation, public API, tests, examples, and integration
are original Apache-2.0 work by the rustmpeg-next contributors. Its algorithms,
integer matrices, and context tables are derived from ITU-T H.266 and VTM 24.0.

## VTM — VVC reference software

- Source: <https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM>
- Copyright (c) 2010-2020, ITU/ISO/IEC
- License: BSD 3-Clause
- Scope: the VVC CABAC engine (all 379 I/P/B context models), intra prediction
  (67 angular modes, wide-angle, PDPC, 6-MPM, MRL, ISP, CCLM, MIP, BDPCM),
  inter prediction (DPB/RPL, spatial/temporal/HMVP merge, MMVD, AMVP, 8-tap luma
  and 4-tap chroma interpolation, BCW, BDOF, DMVR, Affine/PROF, CIIP, GPM),
  transform scan/quantisation, DCT-2 (2..64-point)/DST-7/DCT-8, LFNST
  matrices/tables, Dependent Quantization, Sign Data Hiding, in-loop filters
  (LMCS, Deblocking, SAO, ALF, CCALF), residual coding, and high-level-syntax
  algorithms in `crates/rustmpeg-video/src/vvc/` are derived from VTM 24.0.
  Comments identifying the mirrored VTM functions are retained.

```text
The copyright in this software is being made available under the BSD
License, included below. This software may be subject to other third party
and contributor rights, including patent rights, and no such rights are
granted under this license.

Copyright (c) 2010-2020, ITU/ISO/IEC
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
```

## liblc3 — Bluetooth LC3 reference implementation

- Source: <https://github.com/google/liblc3>
- Copyright 2022 Google LLC
- License: Apache License 2.0
- Scope: the normative tables in `crates/rustmpeg-audio/src/lc3/tables.rs`
  (band-limit tables, low-delay MDCT windows, arithmetic-coding CDFs, SNS
  codebooks and MPVQ offsets, TNS CDFs, LTPF coefficient tables) and
  `crates/rustmpeg-audio/src/lc3/enc_tables.rs` (the `ac_spec_bits`,
  `ac_tns_order_bits`, and `ac_tns_coef_bits` estimates and the LTPF resampler
  tables) are transcribed from liblc3's `src/tables.c`. The encoder and decoder
  algorithms in `crates/rustmpeg-audio/src/lc3/` are an independent Rust
  implementation of Bluetooth LC3 v1.0.1 and do not copy liblc3 source code.
  liblc3 is also used as a black-box oracle to generate the conformance fixtures
  under `crates/rustmpeg-audio/tests/data/lc3/` and to validate the encoder's
  output streams.
- The full Apache License 2.0 text is in [`LICENSE`](LICENSE); it applies to the
  transcribed tables as well.

```text
Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## FFmpeg

FFmpeg is not linked, vendored, or copied. It is used only as an external
behavioral oracle for validation. No FFmpeg source code is included in this
repository.

## Test data

Files under `crates/*/tests/data/` are media data, not source code. Most are
generated fixtures produced by FFmpeg, `flac`, `libopus`, and liblc3 as black-box
generators; they are not derived from, and do not inherit the license of, those
tools. The exception is `crates/rustmpeg-video/tests/data/apv/qp_F.apv`, which is a
conformance test vector copied from OpenAPV and stays under BSD-3-Clause with the
notice below.

## Opus SILK normative tables

- Sources: RFC 6716 Appendix A (normative tables), cross-checked against the
  `rusty-opus` pure-Rust implementation for table arrangement.
- Copyright: IETF Trust and the Opus authors; `rusty-opus` contributors.
- License: BSD 3-Clause
- Scope: the SILK constant tables in
  `crates/rustmpeg-audio/src/opus/silk/tables*.rs` and `define.rs` are normative
  RFC 6716 data. The decoder algorithms are original Rust code implemented from
  the RFC; no libopus C source is copied or linked.

## OpenEXR — OpenEXRCore reference implementation

- Source: <https://github.com/AcademySoftwareFoundation/openexr> (`src/lib/OpenEXRCore`)
- Copyright (c) Contributors to the OpenEXR Project.
- License: BSD 3-Clause
- Scope: the OpenEXR codec in `crates/rustmpeg-image/src/exr/` is derived in part
  from OpenEXR 3.5.2 `OpenEXRCore`. The derived material covers container and
  chunk layout, header validation limits, DWA v0/v1/v2 channel-classification
  rule tables, the closed-form perceptual LUT formulas and their
  rounding-boundary patches, the 8x8 DCT/IDCT butterfly constants, B44/B44A
  perceptual conversion rules, PIZ and canonical-Huffman conventions and
  constants, and the Zstd-EXR container layout with channel sorting, row delta,
  and byte shuffle. The files carrying that derivation
  (`mod.rs`, `header.rs`, `compression.rs`, `decoder.rs`, `encoder.rs`, `dwa.rs`,
  `zstd.rs`, `half.rs`, `luma_chroma.rs`) declare
  `Apache-2.0 AND BSD-3-Clause` SPDX headers. `deflate.rs` is an independent
  implementation of IETF RFC 1950/1951 and remains Apache-2.0. No OpenEXR C/C++
  source is compiled or linked; OpenEXR 3.5.2 is also used as a black-box oracle
  for the conformance results in `docs/OPENEXR.md`.

```text
Copyright (c) Contributors to the OpenEXR Project. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## OpenAPV — APV conformance test vector

- Source: <https://github.com/AcademySoftwareFoundation/openapv/tree/main/test/bitstream/qp_F.apv>
- Copyright (c) 2022 Samsung Electronics Co., Ltd.
- License: BSD 3-Clause
- Scope: `crates/rustmpeg-video/tests/data/apv/qp_F.apv`, a 20 KiB raw APV
  bitstream with two 320x240 4:2:2 12-bit frames. It is used only as decoder
  conformance input. No OpenAPV C source is copied, compiled, or linked; the APV
  codec itself is implemented from RFC 9924. Source, license, and SHA-256 are also
  recorded in `crates/rustmpeg-video/tests/data/apv/README.md`.

```text
Copyright (c) 2022 Samsung Electronics Co., Ltd.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

- Neither the name of the copyright owner, nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## dav2d — AV2 reference decoder

- Source: <https://code.videolan.org/videolan/dav2d>
- Copyright (c) 2018-2026, VideoLAN and dav2d authors
- License: BSD 2-Clause
- Scope: the AV2 encoder conformance work in `crates/rustmpeg-video/src/av2/`
  reuses coefficient CDF values, entropy/coefficient syntax details,
  quantiser/dequantiser tables, transform restrictions, and block/header ordering
  documented by dav2d. The encoder remains an independent Rust implementation; no
  C source is compiled or linked.

```text
Copyright © 2018-2026, VideoLAN and dav2d authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## External oracles (not included)

- **FFmpeg** — behavioral oracle for codec validation; no source included.
- **libopus / opus-tools** — Opus fixtures and packet-boundary checks; no source
  included.
- **flac / libFLAC** — FLAC fixtures and reference PCM; no source included.
- **liblc3 / elc3 / dlc3** — LC3 conformance fixtures and reference PCM; only the
  normative tables are transcribed (see above), no source is compiled or linked.
- **OpenAPV** — APV decoder conformance vectors; one committed vector (see above),
  no source compiled or linked.
- **FFmpeg APV decoder** — bit-exactness oracle for the in-tree decoder; no source
  included.

These tools are not linked or vendored. Their output is test data, not code.

## Dependency audit

All Rust dependencies are permissively licensed. The audit is reproducible with:

```bash
python3 scripts/check-licenses.py
```

It fails on GPL/LGPL/AGPL/MPL/SSPL/BUSL/CDLA or unknown licenses. `quinn` was
removed because it pulled `ring`, which requires a C/assembly build.
