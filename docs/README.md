# Documentation

This directory is reader-facing: everything here is for users, integrators, and
contributors of `rustmpeg-next`. Agent working notes, wave plans, and task
briefs are not kept here; they live in the local, git-ignored `docsagents/`
workspace, which is not part of this repository.

Start with [`CONCEPT.md`](CONCEPT.md) for the product scope and the supported
standard matrix, then pick the topic you need.

Recorded commands use path placeholders instead of machine-specific paths:
`$REPO` is the repository root, `$ORACLE` the directory holding external oracle
builds (`dav2d`, AVM, `vvc-conf`), `$DOCS` the directory holding downloaded
specifications, and `$WORK` a scratch directory. Oracle binaries are never
hard-coded: the tests read `DAV2D`, then fall back to `PATH`, and skip with a
printed reason when neither resolves.

## Architecture and scope

| Document | Content |
| :-- | :-- |
| [`CONCEPT.md`](CONCEPT.md) | Product specification: crate layering, codec/container matrix, rolling-window policy, CPU/GPU split |
| [`../README.md`](../README.md) | Public overview, per-codec usage, requirements, build commands |
| [`../AGENTS.md`](../AGENTS.md) | Contributor and agent rules: provenance, licensing, dependency policy, checks, GPU contract |

## Codecs

| Document | Content |
| :-- | :-- |
| [`VVC-INTRA-PROFILE.md`](VVC-INTRA-PROFILE.md) | Exact H.266/VVC intra subset that is implemented and conformant |
| [`vvc/STATUS.md`](vvc/STATUS.md) | VVC implementation status, measured conformance matrix, known failures |
| [`vvc/CITATION_LOG.md`](vvc/CITATION_LOG.md) | Per-feature citation log: H.266 clause and VTM 24.0 file for every table and algorithm |
| [`av2/UNIT5_DECODER.md`](av2/UNIT5_DECODER.md) | AV2 segmentation, delta-Q, and adaptive quantization — decoder |
| [`av2/UNIT5_ENCODER.md`](av2/UNIT5_ENCODER.md) | AV2 segmentation and VAQ — encoder |
| [`av2/UNIT8_DECODER.md`](av2/UNIT8_DECODER.md) | AV2 warped-motion decoding and selection |
| [`av2/UNIT8_ENCODER.md`](av2/UNIT8_ENCODER.md) | AV2 warped motion; OBMC is not part of AV2 v1.0.0 |
| [`av2/UNIT10_DECODER.md`](av2/UNIT10_DECODER.md) | AV2 extended partitions — decoder parsing, geometry, and `dav2d` matrix |
| [`av2/UNIT10_ENCODER.md`](av2/UNIT10_ENCODER.md) | AV2 extended partitions and partition RDO — encoder |
| [`av2/UNIT11_ENCODER.md`](av2/UNIT11_ENCODER.md) | AV2 palette and Intra Block Copy; ISP is not part of AV2 v1.0.0 |
| [`APV.md`](APV.md) | APV (RFC 9924) intra codec: scope, limits, bitstream, verification |
| [`JXL.md`](JXL.md) | JPEG XL (ISO/IEC 18181-1 & 18181-2) decoder, clean-room encoder, and `moxcms` CMS |
| [`OPENEXR.md`](OPENEXR.md) | OpenEXR v2 decoder and encoder (OpenEXRCore-derived, BSD-3-Clause; `None`, `Rle`, `Zips`, `Zip`, `Piz`, `Pxr24`, `B44`, `B44a`, `Dwaa`, `Dwab`, `Zstd`, tiled, deep, multipart) |
| [`OPUS-DECODER-PLAN.md`](OPUS-DECODER-PLAN.md) | Opus packet/range layer and SILK decoder: implemented stages and remaining waves |

Status at a glance:

- **H.266/VVC** — strict intra-only encoder plus matching subset decoder, and an
  opt-in multi-tool Intra/Inter P writer. The strict 8-bit 4:2:0 subset is
  byte-exact against VTM 24.0 `DecoderApp` and VVdeC 3.3 `vvdecapp`; the full
  measured matrix is recorded in [`vvc/STATUS.md`](vvc/STATUS.md) with its five
  remaining failures listed explicitly. Not a general VVC codec.
- **AV2** — experimental encoder and matching byte-stream decoder. Per-tool
  implementation and conformance status is recorded in the unit documents above;
  superblock-aligned streams are byte-exact against `dav2d` for the recorded
  matrix, while edge partitions and several in-loop tools are still under active
  development. See [`../NOTES.md`](../NOTES.md) for measurements and
  [`../AGENTS.md`](../AGENTS.md#codec-subagent-protocol) for the release gates.
- **APV** — implemented intra encoder and decoder, CPU only.
- **JPEG XL & OpenEXR** — implemented JPEG XL v2 decoder and clean-room encoder (`ModularLossless` + `VarDctLossy`), OpenEXR v2 decoder and encoder (OpenEXRCore-derived, BSD-3-Clause; `None`, `Rle`, `Zips`, `Zip`, `Piz`, `Pxr24`, `B44`, `B44a`, `Dwaa`, `Dwab`, `Zstd`, scanline/tiled/deep/multipart), and `moxcms` ICCv4/CICP color management in `rustmpeg-image`.
- **Audio** — FLAC and LC3 complete; LPCM complete; Opus packet/SILK implemented,
  CELT/hybrid/PLC pending.
- **Containers** — Matroska v4/WebM and ISOBMFF/CMAF implemented; MoQ planned.

## Containers, configuration, and operations

| Document | Content |
| :-- | :-- |
| [`ISOBMFF.md`](ISOBMFF.md) | ISOBMFF/CMAF reader and writers: supported boxes, limits, verification |
| [`configs/CONFIG.md`](configs/CONFIG.md) | Every CLI command and example, environment variables, configuration structs, codec and format options, offline build |
| [`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md) | Benchmark methodology, how to reproduce, and the results table format |
| [`faq/FAQ.md`](faq/FAQ.md) | Short answers: toolchain, no C dependencies, conformance, oracles, adding a codec, licensing, GPU policy |

## Licensing and provenance

| Document | Content |
| :-- | :-- |
| [`LICENSE-AUDIT.md`](LICENSE-AUDIT.md) | Latest audit result and its scope |
| [`../THIRD_PARTY.md`](../THIRD_PARTY.md) | Full third-party license texts, scopes, and attribution |
| [`../LICENSING.md`](../LICENSING.md) | Human-readable policy and decision rule |
| [`../NOTICE`](../NOTICE) | Project and third-party attribution |

## Verification records

| Document | Content |
| :-- | :-- |
| [`../NOTES.md`](../NOTES.md) | Design notes, measured benchmarks, and per-area implementation history |
| [`../VERIFICATION_DEBT.md`](../VERIFICATION_DEBT.md) | What was and was not executed, and the exact external conformance commands |
