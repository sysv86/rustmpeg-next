# AGENTS.md

Operating rules for AI agents and human contributors working in this repository.
Read this file in full before changing anything.

Priority order when documents disagree:

1. This file — process, provenance, and licensing rules.
2. [`docs/CONCEPT.md`](docs/CONCEPT.md) — product scope and supported standards.
3. [`README.md`](README.md) — public overview.
4. [`docs/README.md`](docs/README.md) — index of reader documentation.

Private notes under `docsagents/` are not normative and are never cited as the
source of a rule.

## Mission

`rustmpeg-next` is an independent, full-cycle multimedia framework written in Rust:
demuxing, decoding, filtering, encoding, muxing, and network transport. It targets
functional parity with FFmpeg, but it is **not** a fork, wrapper, or translation of
FFmpeg. Four modules carry recorded permissive derivations: the VVC module from
VTM 24.0 (BSD-3-Clause), the AV2 conformance data from dav2d (BSD-2-Clause), the
LC3 normative tables from liblc3 (Apache-2.0), and the OpenEXR module from
OpenEXR 3.5.2 `OpenEXRCore` (BSD-3-Clause). Everything else is implemented
independently; the exact scope of each derivation is in
[`THIRD_PARTY.md`](THIRD_PARTY.md).

The normative standard is the primary source. We build our own architecture and
code. FFmpeg source may be read to understand behavior, but it is never copied or
transcribed. Approved permissively licensed code is reused only with recorded
provenance and attribution (see [`THIRD_PARTY.md`](THIRD_PARTY.md)); the VVC, AV2,
and LC3 modules currently derive from the sources recorded there.

## Non-negotiable constraints

1. **No copyleft or unlicensed code.** Never copy code from FFmpeg, libav*,
   GStreamer, or another LGPL/GPL/AGPL implementation. Reading their source to
   understand behavior is allowed; transcription and relicensing are not.
2. **Spec-first, with approved reuse.** Implement from normative specifications.
   Reuse from MIT/BSD sources is allowed only when listed in `THIRD_PARTY.md` with
   attribution. The VVC module is derived from VTM 24.0, the AV2 conformance data
   from dav2d, the LC3 tables from liblc3, and the OpenEXR module from OpenEXRCore;
   do not claim clean-room provenance for those parts.
3. **No C/C++ toolchain.** No `*-sys`/`*_sys` crate that compiles or bundles
   C/C++, no `build.rs` that compiles C or C++, no bundled native libraries.
   Generated bindings to OS/API surfaces without a native build (e.g. Windows
   bindings) are allowed. The GPU stack uses `wgpu` and `naga`, both pure Rust.
   `ash` and `gpu-allocator` appear only transitively through `wgpu-hal`; both are
   permissive and add no native build step. `vk-mem`, `cc`, and `bindgen` are not
   used. If a dependency has a native build step, reject it and report it.
4. **No `unsafe`.** Every crate, including `rustmpeg-vulkan`, keeps
   `#![forbid(unsafe_code)]`. The GPU layer uses `wgpu` as a safe Vulkan binding;
   a change that genuinely requires `unsafe` stops and asks first.
5. **Rolling window only.** Do not add codecs, containers, or transports outside
   the matrix in `docs/CONCEPT.md`. No legacy formats, no “just for compatibility”
   modes.
6. **English only.** Code, identifiers, comments, documentation, commit messages,
   pull requests, issues, and test names are written in English.
7. **Latest versions.** Dependency versions in `[workspace.dependencies]` must be
   the latest stable releases on crates.io. Verify before changing them; do not
   guess from memory. Commit `Cargo.lock`.
8. **Pinned nightly.** The toolchain is fixed in `rust-toolchain.toml`. Do not bump
   it as part of an unrelated change.
9. **Layering.** Dependencies point downward only:
   `core → format / audio / image / video → vulkan`. No cycles, no reverse
   dependencies, no new top-level crates without explicit approval.
   `rustmpeg-core` is `no_std` + `alloc`, runtime-agnostic, and must not depend on
   `tokio`, `bytes`, `wgpu`, `naga`, or `std`.
10. **No silent scope expansion.** Implement the requested change. Do not add
    speculative abstractions, configuration knobs, or unrelated refactors.
11. **License gate.** Run `python3 scripts/check-licenses.py` before reporting
    completion. Compatible permissive code stays as-is with its notices; poisonous
    code is rejected and either reimplemented from the specification or replaced
    with a permissive alternative. See “Licensing (complete policy)”.

## Provenance policy

Copyright and patent risk is a first-class engineering concern in this project.
The following rules are mandatory, not advisory.

### Primary sources

Use the normative documents and their official errata:

- ITU-T H.266 / VVC, ITU-T H.267 / ECM (when published)
- ISO/IEC 14496 (MPEG-4 systems, ISOBMFF), ISO/IEC 23008 (HEIF, MIAF), ISO/IEC 18181 (JPEG XL)
- AOMedia AV2 and IAMF specifications
- IETF RFC 6716 (Opus), RFC 9639 (FLAC), RFC 9924 (APV), `moq-transport` drafts
- SMPTE ST 2094-50, SMPTE / IETF APV material
- Bluetooth SIG LC3 / LC3plus specifications
- Matroska / EBML and WebM specifications
- Vulkan specification, WGSL, and Vulkan Compute limits

Cite the exact document and section in the PR description for non-obvious
behavior, for example: `Spec: ITU-T H.266 §8.3.2; test vector: <name>`.

### Allowed

- Reading normative specifications, whitepapers, and published design documents.
- Reading FFmpeg, GStreamer, or other implementation source to understand
  behavior, edge cases, or bitstream layout. Never transcribe code, tables, or
  comments from those sources.
- Running FFmpeg, GStreamer, or reference software as a **black box** on test
  media to compare observable output, bitrate, timestamps, and conformance.
- Using official conformance test vectors when their license permits it. Record
  the source and license of every imported vector.
- Consulting reference software (VTM, AVM) only to resolve a conformance
  question, and only if its license permits it. Do not transcribe anything from
  it.

### Forbidden

- Copying, porting, or transcribing FFmpeg / libav* / GStreamer source.
- Reusing code from a third-party source that is not recorded in `THIRD_PARTY.md`.
- Porting code from copyleft projects or from Rust projects that themselves port
  copyleft code.
- Pasting code from blogs, Stack Overflow, or generated snippets that reproduce a
  known implementation.
- Copying lookup tables, constants, or test data without a recorded, compatible
  source and license.
- Claiming “clean-room” status for work that consulted a restricted source. If
  you have seen restricted source while working on a feature, stop and disclose
  it in the PR; a reviewer must assess the impact.

## Licensing (complete policy)

This section is authoritative for license compliance. Source provenance (which
sources may be read or reused) is defined in “Provenance policy” above; this
section covers the project license, third-party code, dependencies, and audit.

### Project license and contributions

- The project is licensed under **Apache-2.0** ([`LICENSE`](LICENSE)). Original
  code, tests, examples, and documentation are Apache-2.0.
- There is **no CLA** and no copyright assignment. Contributions are accepted
  under Apache-2.0 on an inbound = outbound basis.
- There is one build and one license surface: no GPL/LGPL/AGPL build variants, no
  optional copyleft components, no runtime loading of copyleft code.
- Never relicense third-party code under Apache-2.0. Keep its original license,
  copyright notice, and provenance comments.
- Never add a header that misrepresents authorship or the license of a file.

### SPDX headers

Every Rust source file starts with an SPDX line:

- original code: `// SPDX-License-Identifier: Apache-2.0`;
- VVC files containing VTM-derived algorithms or tables:
  `// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause`;
- AV2 files containing dav2d-derived conformance data or syntax:
  `// SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause`.

Do not remove or alter an existing SPDX line, copyright notice, or derivation
comment. New files need the correct header before they are committed.

### Compatible vs. poisonous

- **Compatible** (Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, Zlib,
  Unlicense, CC0-1.0, Unicode-3.0, Apache-2.0 WITH LLVM-exception): acceptable.
  Keep the code, keep the notices, and do **not** rewrite it just to remove
  attribution. Rewriting compatible code is unnecessary risk.
- **Poisonous** (GPL, LGPL, AGPL, MPL, SSPL, BUSL, CDLA, unknown/custom): must not
  enter the tree. Either implement the functionality ourselves from the normative
  specification, or choose a permissively licensed alternative. Never copy, vendor,
  link, or transcribe poisonous code.

Compatible code that is already present is not a violation. It stays as-is with
its notices (for example, the VVC module's VTM derivation).

### Allowed and forbidden licenses

Allowed: Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, Unlicense,
CC0-1.0, Unicode-3.0, and Apache-2.0 WITH LLVM-exception.

Forbidden: GPL-2.0, GPL-3.0, LGPL-2.1, LGPL-3.0, AGPL, MPL-2.0, SSPL, BUSL,
CDLA, “custom” licenses, and anything without a clear SPDX identifier. Copyleft
must not appear even as an unused alternative in a dual- or triple-licensed crate.

### Third-party code already in the tree

| Source | License | Scope |
| :-- | :-- | :-- |
| [VTM 24.0](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM) | BSD-3-Clause | CABAC, prediction, DCT/DST/LFNST, residual coding, and high-level-syntax algorithms |
| [dav2d](https://code.videolan.org/videolan/dav2d) | BSD-2-Clause | AV2 conformance data, coefficient CDFs, syntax ordering, quantiser/transform details |
| [liblc3](https://github.com/google/liblc3) | Apache-2.0 | LC3 normative tables only (`tables.rs`, `enc_tables.rs`) |
| [OpenEXR 3.5.2](https://github.com/AcademySoftwareFoundation/openexr) | BSD-3-Clause | `rustmpeg-image` EXR container/chunk layout, validation limits, DWA rules and perceptual tables, DCT constants, B44 conversions, PIZ/Huffman conventions, and Zstd-EXR container |
| [OpenAPV](https://github.com/AcademySoftwareFoundation/openapv) | BSD-3-Clause | APV conformance vector `crates/rustmpeg-video/tests/data/apv/qp_F.apv` |
| RFC 6716 Appendix A / `rusty-opus` | BSD-3-Clause | SILK constant tables only; no libopus source is copied or linked |

The full texts and scopes are in [`THIRD_PARTY.md`](THIRD_PARTY.md); attribution is
also in [`NOTICE`](NOTICE). These files must stay current and must not be removed.
All other code in the repository is original Apache-2.0 work.

### Test data and external tools

- Files under `crates/*/tests/data/` are generated media data, not source code.
  They do not inherit the license of the tools that produced them. The OpenAPV
  APV conformance vector is the recorded exception: it is third-party data, and
  its source and BSD-3-Clause license are recorded in `THIRD_PARTY.md` and
  `NOTICE`.
- FFmpeg, libopus, `flac`, and libFLAC are used only as external black-box oracles.
  No source code from them is linked, vendored, or copied.
- Do not import a third-party test corpus unless its license is compatible and the
  source/license is recorded in `THIRD_PARTY.md`.

### Native code, build scripts, and linking

- No `*-sys`/`*_sys` crate that compiles or bundles C/C++, and no `build.rs` that
  compiles C, C++, or assembly.
- Generated bindings to OS/API surfaces without a native build are allowed (for
  example Windows bindings).
- No bundled native libraries and no runtime loading of copyleft libraries.
- `wgpu` (safe Vulkan/WebGPU bindings) and `naga` (WGSL validation) are allowed;
  both are pure Rust. Vulkan Video is not used and `ash` is not a direct
  dependency; `ash` and `gpu-allocator` are transitive through `wgpu-hal`.
  `quinn` was removed because it pulled `ring`, which builds C/assembly.
- If a feature needs a native library, choose a pure-Rust alternative or implement
  it; if that is impossible, stop and report the conflict.

### Patent licensing

Patent licensing is separate from copyright licensing. Do not describe a codec as
royalty-free without citing the organization that makes that claim. AV2, Opus,
IAMF, and APV are designed as royalty-free standards; VVC is covered by
patent pools and must not be described as royalty-free.

### Adding a dependency

1. Verify the latest version on crates.io.
2. Check its license with `cargo metadata` or the crates.io API.
3. Inspect its `build.rs` and dependency tree for native code.
4. Add it to `[workspace.dependencies]` and wire it into exactly one crate.
5. Run `python3 scripts/check-licenses.py`.
6. Update `THIRD_PARTY.md` and `NOTICE` if it includes third-party code.
7. State in the PR why the dependency is necessary.

### Audit, violations, and subagents

The audit command is:

```bash
python3 scripts/check-licenses.py
```

If a violation is found:

1. Do not commit or publish.
2. Remove the poisonous code/dependency, or replace it with a compatible
   alternative, or implement the functionality from the normative specification.
3. Update `THIRD_PARTY.md`, `NOTICE`, SPDX headers, and the audit report.
4. Re-run the audit and the normal test suite.
5. Report exactly what was removed, replaced, or reimplemented.

Subagents must apply this rule before adding or porting code, must not remove
existing third-party notices, and must run the audit before reporting completion.
If a task cannot be completed without a poisonous license, stop and report the
conflict instead of proceeding.

### Files that must stay current

- [`LICENSE`](LICENSE) — Apache-2.0 text.
- [`NOTICE`](NOTICE) — project and third-party attribution.
- [`THIRD_PARTY.md`](THIRD_PARTY.md) — full third-party license texts and scopes.
- [`LICENSING.md`](LICENSING.md) — human-readable policy and decision rule.
- [`Licenses/`](Licenses/) — full license texts (Apache-2.0, BSD-2-Clause,
  BSD-3-Clause) plus the component-to-license map. A convenience mirror only:
  the root files above stay canonical.
- [`docs/LICENSE-AUDIT.md`](docs/LICENSE-AUDIT.md) — latest audit result.
- [`scripts/check-licenses.py`](scripts/check-licenses.py) — repeatable audit.
- `Cargo.toml` files — `license = "Apache-2.0"` on every workspace crate.

This policy is an engineering rule, not legal advice. Escalate unclear cases
instead of guessing.

## Repository layout

```text
crates/
  rustmpeg-core/     no_std buffers, time, pixel/colour/frame/audio formats,
                     packets, typed errors, portable SIMD
  rustmpeg-format/   EBML, Matroska/WebM, and ISOBMFF/CMAF implemented; MoQ planned
  rustmpeg-audio/    Opus/FLAC/LPCM/LC3 implemented; IAMF + OAR planned
  rustmpeg-image/    JPEG XL codec + CMS implemented; OpenEXR planned
  rustmpeg-video/    AV2 experimental encoder + decoder, VVC intra subset plus
                     multi-tool Intra/Inter P, APV intra codec
  rustmpeg-vulkan/   Vulkan Compute kernels, checks, CPU/GPU scheduler
  rustmpeg-cli/      dependency-free CLI bridge for rustmpeg-core
docs/                reader documentation (architecture, codecs, configs, FAQ)
  docs/benchmarks/   benchmark methodology and the CI-generated results table
  docs/configs/      CLI reference, environment variables, config structs, offline build
  docs/faq/          short answers for users and new contributors
  docsagents/        local, git-ignored agent workspace (not part of this repository)
```

## Build and checks

The toolchain is pinned in `rust-toolchain.toml` and requires nightly because the
compute cores use `std::simd`.

```bash
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo fmt --all -- --check
python3 scripts/check-licenses.py
```

All five commands must pass before a change is considered complete. The only
current warnings are `unused dependency` manifest warnings from
`rustmpeg-image`, which still lists dependencies its APIs do not use yet; they
are expected until those APIs land. Do not silence these warnings with `allow`
attributes.

## Coding rules

- Public items require documentation; `missing_docs` is enabled at workspace level.
- Use typed errors (`thiserror`). Library code must not panic on untrusted input.
- Parsers use bounds-checked reads and checked arithmetic. Add explicit size limits
  to resist denial-of-service inputs.
- Never block inside async code. Use `tokio` primitives and `Arc<[u8]>` buffers.
- Prefer zero-copy slices. Do not introduce intermediate buffers without a measured
  reason.
- SIMD code uses `std::simd` and autovectorization. Do not add inline assembly or
  target-specific intrinsics without approval.
- New parsing code needs unit tests and, where applicable, a fuzz target.
- Do not add network access, telemetry, or background tasks to library crates.

## Heterogeneous CPU/GPU execution

This contract is mandatory for every video codec, every codec update, and every
new stage, including AV2, H.266/VVC, and APV. The backend is Vulkan Compute
through `wgpu` + `naga`; Vulkan Video and vendor fixed-function APIs are not
used because they implement none of the codecs in the matrix.

### Default split

Acceleration is scheduled per stage, never per codec and never per block.

| Unit | Owns |
| :-- | :-- |
| **CPU** | Bitstream parsing and writing, CABAC/range/APV entropy coding, block partitioning, mode and RDO decisions, reference management, control flow, and any stage with serial neighbour dependencies. |
| **GPU** | Whole-frame, whole-plane, whole-tile, or large independent-block batches: colour conversion, scaling, tone mapping, distortion metrics, and loop filters once their parameters are already decided. Transforms, prediction, and motion compensation may join this list only when the data is already resident in VRAM and a whole frame or tile can be batched. |

Never dispatch a kernel per coding block (32x32, 64x64, or similar). Upload,
dispatch, synchronization, and read-back cost more than the arithmetic at that
size. The AV2 deblocking kernel is the measured example: it is bit-exact with
the CPU filter, but the transfer-bound path is slower at 640x360 through 4K, so
it stays opt-in and the CPU filter remains the default.

### Decision rule

1. Implement and test the CPU reference first. It is the normative result.
2. Measure both paths at the resolutions the codec targets. The GPU number must
   include allocation, upload, dispatch, synchronization, read-back, and any
   format conversion.
3. Use the GPU only if it is at least 10% faster than the CPU at that size. If
   it is not, keep the CPU path and do not add a GPU kernel "for symmetry".
4. Small pictures, short clips, and machines without a usable adapter always run
   the CPU path.
5. The decision is dynamic: a calibrated scheduler chooses per stage and per
   frame size. It is not a compile-time feature flag and not a global "GPU mode".

### Bit-exactness and failure handling

- A GPU stage must produce byte-identical output to its CPU reference: encoded
  access units, decoded pictures, reconstructed planes, and coefficient arrays.
  Metrics may be reduced on the host, but SSE/SAD and the resulting PSNR must
  match exactly.
- Validate the GPU result against the CPU reference during calibration and in
  the test suite. A mismatch disables the GPU path for the session; it never
  changes the bitstream, the frame, or the reported statistics.
- A GPU error must leave the CPU-side frame untouched. Apply results to caller
  memory only after the submission succeeds, so the caller can retry on the CPU.
- Never silently fall back after writing a partially filtered or converted frame.

### Layering and API

- Codec crates (`rustmpeg-video`) must not depend on `rustmpeg-vulkan`, `wgpu`,
  or `naga`.
- Define a narrow, object-safe trait on the codec side with a CPU implementation
  (see `rustmpeg-video::accelerator` and `av2::deblock::DeblockFilter`).
  Implement it in `rustmpeg-vulkan`.
- Inject the implementation explicitly: an `Option<&dyn Trait>`, `Arc<dyn
  Trait>`, or a dedicated `*_with(...)` method. The default is always `None`,
  which means the CPU path.
- The trait must be `Send + Sync`, must carry its own device handle, and must not
  depend on global state, background tasks, or network access.
- Do not expose `wgpu` types in codec APIs and do not make the GPU mandatory for
  correctness.

### Memory and submission rules

- Batch related dispatches and read-backs into one `Compute::batch()` submission
  (or an equivalent recorder). One queue round trip per pass, not per kernel.
- Keep frames and coefficient planes in VRAM between consecutive GPU stages.
  Upload once and read back once per pass, only when the CPU genuinely needs the
  result for entropy coding, reference management, or output.
- Use padded strides and packed 32-bit words; never assume tightly packed rows.
  Validate every stride, length, and device limit before allocating or binding.
- A new GPU stage must support the same layout and bit-depth combinations as its
  CPU reference. If a combination is not implemented, the caller stays on the
  CPU; a GPU path must not reject valid input that the CPU accepts.

### Scheduler and calibration

- Every new GPU stage gets a `Stage` variant, a `StageCost`, and a calibration
  entry. The fit uses at least two representative frame sizes and a fixed
  per-call cost plus a per-pixel slope.
- Warm up the device before timing. A cold laptop GPU can report a several-fold
  slower rate and permanently bias the scheduler towards the CPU.
- Require the 10% margin. A stage that only matches the CPU stays on the CPU.
- `disable_gpu()` is permanent for the session. `recalibrate()` is explicit and
  may re-enable it after a device or driver change.
- Record measured results in `NOTES.md`. A GPU path with no measurement is
  incomplete.
- An experimental kernel that fails the 10% bar is not wired into automatic
  placement. It remains an explicit opt-in provider, the CPU path stays the
  default, and `NOTES.md` records why. The AV2 deblocking kernel is the current
  example.

### Tests and acceptance for codec work

Every new or updated codec stage follows this sequence:

1. CPU implementation and unit tests.
2. A trait and a CPU reference implementation on the codec side.
3. A WGSL kernel and `rustmpeg-vulkan` implementation.
4. A bit-exact test comparing the GPU output with the CPU reference, including
   odd dimensions, padded strides, and every supported layout/bit depth.
5. An integration test proving that enabling the accelerator does not change the
   bitstream or the reconstructed frame (VVC access-unit equality, AV2 packet and
   `rec_frame` equality).
6. A scheduler stage and calibration. The only exception is an explicit opt-in
   experiment that failed the 10% bar: it must not be selected automatically, must
   stay disabled by default, and must be documented as measured-slower.
7. A benchmark entry and recorded numbers.
8. A headless run where GPU tests skip cleanly because no adapter is present.

A change to bitstream syntax, reconstruction, or filter parameters updates the
CPU reference and the GPU kernel in the same change, and reruns the bit-exact
tests. Do not update one side only.

### Forbidden

- Vulkan Video, VA-API, NVDEC/NVENC, VideoToolbox, DXVA2, or any vendor codec API.
- Entropy coding, bitstream parsing, or block-level RDO on the GPU.
- Per-block dispatch or per-block read-back.
- A GPU-only path without a CPU fallback.
- Claiming a speedup without a release-mode measurement that includes transfer.
- A GPU path that changes the output, silently ignores errors, or requires
  `unsafe`.

## Skills

When the `sysv86/kromite` skill collection is installed in the agent harness, load
the matching skill before starting a task instead of improvising. Typical matches
in this repository: `scope-guard` for any code change, `research` before a
version- or API-specific claim, `strict` for verification work, `subagents`
before splitting work across agents, and `github` before a Git or CI action. Use
the installed text; do not paraphrase it from memory.

If the collection is not installed, or a named skill is absent, proceed without
it and say so. Never invent a skill name, claim to have loaded a skill that is
not present, or present remembered skill text as if it were the installed
version. Skills are guidance only: this file, `docs/CONCEPT.md`, and the licensing
rules win when they disagree.

## Private agent workspace (`docsagents/`)

`docsagents/` is a git-ignored working area for task briefs, wave plans and
reports, oracle transcripts, and review notes. It is not reader documentation, it
is not linked from `README.md` or `docs/`, and it is never a normative source.

- `docsagents/README.md` — purpose, layout, onboarding brief, and the public
  vs. local decision rule.
- `docsagents/SUBAGENTS.md` — the codec subagent protocol.
- `docsagents/vvc/` — VVC wave plans and reports moved out of `docs/vvc/`.

To create it in a fresh clone:

```bash
mkdir -p docsagents/agents docsagents/logs
grep -qxF '/docsagents/' .gitignore || printf '/docsagents/\n' >> .gitignore
```

Then write `docsagents/agents/<name>.md` from the brief template in
`docsagents/README.md` and start the agent with: read `AGENTS.md`, read the
brief, declare the files it will touch, and stop if the brief and the tree
disagree.

Rules for this directory:

- Do not store credentials, tokens, or keys there. Being ignored by Git does not
  make a file secret.
- Do not link to it from public docs; the link would be broken for readers and CI.
- Do not move SPDX headers, third-party notices, normative citations, or the
  license audit into it. Provenance stays public and tracked.
- Keep old plans when adding reports. Do not rewrite history to look clean.
- If the directory is absent, nothing is broken; recreate it from this section.

## Codec subagent protocol

The full protocol is kept in the local, git-ignored workspace at `docsagents/SUBAGENTS.md`;
if that directory is absent, recreate it from the section above. The binding
rules:

1. **One writer per file.** The orchestrator declares ownership before work
   starts. A file may be read by anyone and written by exactly one agent.
   Decoder and encoder halves may be separate agents only when the shared syntax
   and helpers have a single owner.
2. **Order that cannot be parallelised.** CPU reference and tests, then the
   external oracle comparison, then documentation of the result, then any GPU
   stage. Encoder syntax comes before decoder parsing of the same syntax. A GPU
   kernel never exists without its CPU reference, bit-exact test, scheduler entry,
   and measurement.
3. **Parallel only when disjoint.** Independent codecs, independent tools that do
   not share syntax, and docs/benchmark/CI work may run in parallel. Anything
   whose output is another agent's input runs sequentially.
4. **No partial handoff.** A handoff is complete or blocked, never "mostly done".
   A decoder that parses a tool and rejects the rest of the stream, a GPU kernel
   with no CPU reference, an unverified claim in a doc, or an `#[ignore]` without a
   recorded reason is a blocked handoff. Leave the tree buildable, do not weaken
   tests, and report exactly what fails. A blocked report is a valid outcome; a
   false completion is not.
5. **External oracles are mandatory evidence.** VVC is checked with VTM 24.0
   `DecoderApp` and VVdeC 3.3 `vvdecapp`; AV2 with `dav2d` (`--threads 1
   --framedelay 1 --strict 1`) and, for streams this encoder cannot yet produce,
   AVM `avmenc` with the recorded flags. Compare reconstruction bytes, record the
   oracle revision and exact command, and state which cases were skipped. An
   oracle that rejects the feature is not evidence either way. If the binary is
   absent, the case is skipped and reported as skipped, never faked.
6. **Documentation cannot outrun verification.** Public status text is updated
   only from a recorded oracle result. The public conformance summary lives in
   `docs/vvc/STATUS.md`; internal wave reports stay in `docsagents/vvc/`.
7. **Gates are per change and per stage.** The commands in "Build and checks"
   (including the license audit), and for a codec stage the eight acceptance
   steps in "Heterogeneous CPU/GPU execution". A report names the command and its
   result; it never claims a run it did not perform.

## Documentation discipline

- **Every change updates the docs it affects, in the same change.** Behaviour,
  CLI, configuration, status, file layout, licensing, and conformance changes
  must be reflected in `README.md`, `docs/`, `docs/benchmarks/BENCHMARK.md`,
  `docs/configs/CONFIG.md`, `docs/faq/FAQ.md`, `docs/vvc/STATUS.md`,
  `docs/AV2-RELEASE.md`, `NOTES.md`, and this file when the rules change. A code
  change with stale documentation is an incomplete change.
- **Never lie in tests or documentation.** Do not write that something passes,
  is byte-exact, is supported, is measured, or was run unless the command was
  executed and its result recorded. Re-verify before writing. `UNKNOWN`,
  `SKIPPED`, and `BLOCKED` are valid and expected outcomes; a false `PASS` is a
  defect, must be corrected immediately, and must be disclosed in the report.
- **Tests must assert real behaviour.** No weakened assertions, no hard-coded
  QP/position/test-specific branches, no `if false` feature kills, no ignored
  failure without a recorded reason, and no test that only proves the encoder and
  decoder share the same mistake.
- **Duplicate or divergent documentation is a bug.** When two files describe the
  same subject, merge them into one canonical file, keep at most a one-line
  pointer where a stable path is required, and delete the redundant copy. Never
  keep parallel "old" and "new" versions of the same document, and never let two
  files disagree about status, commands, or licensing.
- Public documentation lives in `docs/`; local agent notes live in
  `docsagents/`. Do not link local notes from public docs, and never move
  normative, provenance, or licensing content into `docsagents/`.
- Status numbers (line counts, completion percentages, conformance counts) must
  be generated or measured, never typed by hand.

## Agent workflow

1. Read `AGENTS.md`, `README.md`, `docs/CONCEPT.md`, [`docs/README.md`](docs/README.md),
   and the relevant crate. Load the matching `sysv86/kromite` skill when it is
   available. Read `docsagents/` only when it exists and the task is agent-facing.
2. Check `git status` and the diff. Never overwrite uncommitted user changes.
3. Plan the smallest connected change across code, tests, and docs. For a codec
   change, decide per stage whether it belongs on the CPU or the GPU using the
   contract above; do not assume the previous placement still holds.
4. Implement it. Update affected manifests and documentation in the same change.
   A new GPU stage also needs its CPU reference, bit-exact test, scheduler entry,
   and benchmark update.
5. Run the checks above and report the exact commands and results, including the
   GPU benchmark when a GPU stage changed.
6. Never claim a test passed unless it was run. State what was skipped and why.
7. Do not commit, push, tag, or publish unless explicitly asked.
8. Ask before adding a dependency, creating a crate, changing the toolchain, or
   modifying the supported-format matrix.

## Stop and ask

Stop and request a decision if a task requires any of the following:

- reusing code from a new third-party source without provenance review,
- a copyleft or unclear-license dependency,
- a C/C++ build step,
- any `unsafe` code anywhere,
- a legacy codec or container,
- changing the pinned toolchain or Rust edition,
- publishing, tagging, or force-pushing,
- editing a file that another agent owns without the owner's agreement,
- moving material between `docs/` and `docsagents/` in either direction.
