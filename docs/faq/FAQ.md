# FAQ

Short answers to the questions that come up before someone can build, test, or
extend the tree. Details are in [`CONCEPT.md`](../CONCEPT.md),
[`CONFIG.md`](../configs/CONFIG.md), [`vvc/STATUS.md`](../vvc/STATUS.md), and
[`../../AGENTS.md`](../../AGENTS.md).

## Build and toolchain

**Which toolchain do I need?**

`nightly-2026-10-25`, pinned in [`rust-toolchain.toml`](../../rust-toolchain.toml).
Nightly is required because the compute cores use `std::simd` (`portable_simd`),
which is not stable. `rustup` installs the pinned version automatically when you
run Cargo inside the repository.

**Why is the toolchain pinned instead of using stable?**

Because the SIMD kernels use `portable_simd`, and replacing it with
target-specific intrinsics or inline assembly is forbidden by the project rules.
The pin also keeps formatting, lint, and test results reproducible.

**Can I build without network access?**

Yes. There is no `build.rs`, no C/C++ step, and no system library lookup. Run
`cargo fetch --locked` once while online, then use
`cargo check --workspace --locked --offline`. See
[`CONFIG.md`](../configs/CONFIG.md#offline-build).

**Which checks must pass?**

```bash
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo fmt --all -- --check
python3 scripts/check-licenses.py
```

`rustmpeg-image` still emits unused-dependency warnings because its manifest lists
dependencies its APIs do not use yet. Do not silence them with `allow`
attributes.

**Do I need a GPU?**

No. A Vulkan 1.1 adapter enables the optional compute path; without one the
identical CPU pipeline runs. No Vulkan SDK, shader compiler, or Vulkan Video
extension is needed at build time.

## Dependencies and build model

**Why are C and C++ dependencies forbidden?**

Memory safety and reproducible builds. No `*-sys`/`*_sys` crate, no `build.rs`
that compiles C/C++/assembly, and no bundled native library. That is why `quinn`
was removed (it pulls `ring`, which builds C and assembly), why `vk-mem` is not
used (it builds C++ through `cc`), and why colour management uses `moxcms`
instead of `lcms2`. `wgpu` and `naga` are pure Rust and are the only GPU stack.

**Is this a fork or wrapper of FFmpeg?**

No. It is an independent Apache-2.0 implementation built from normative
specifications. FFmpeg and other implementations are used only as black-box
oracles. The VVC algorithms and tables derive from VTM 24.0 (BSD-3-Clause); AV2
conformance data comes from dav2d (BSD-2-Clause); LC3 tables come from liblc3
(Apache-2.0). Those derivations are recorded in
[`../../THIRD_PARTY.md`](../../THIRD_PARTY.md) and are not claimed as
clean-room work.

**Can I add a crate or dependency?**

Only after checking the latest version, its license (permissive list only), and
its build for native code, then running `python3 scripts/check-licenses.py` and
recording it in `THIRD_PARTY.md` when it contains third-party code. Ask first. See
[`../../AGENTS.md`](../../AGENTS.md#adding-a-dependency).

## Conformance

**Is the VVC codec conformant?**

For the defined subset, yes, within the measured matrix. The strict 8-bit 4:2:0
intra subset is byte-exact against VTM 24.0 `DecoderApp` and VVdeC 3.3
`vvdecapp` (25-case harness, 50/50 across both decoders). The full 266-check
matrix reports 261 passed, 5 failed, 0 skipped. The five failures and their
causes are listed in [`vvc/STATUS.md`](../vvc/STATUS.md); they are not hidden and
must not be described as conformant. It is not a general VVC decoder.

**Why does 12-bit VVC pass VTM but fail VVdeC?**

VVdeC 3.3 rejects every 12-bit stream, including one produced by VTM 24.0, so it
cannot serve as a 12-bit oracle. VTM 24.0 decodes this project's 12-bit stream
byte-exactly. The failure is an oracle limitation, not a stream defect.

**Is AV2 conformant?**

For superblock-aligned sizes and the configurations recorded in
[`../av2/`](../av2/) and [`../../NOTES.md`](../../NOTES.md), the decoded output is
byte-exact against `dav2d`. Edge sizes that are not superblock-aligned are not
`dav2d`-conformant yet, and in-loop tools are still being brought in under the
release-unit briefs. The decoder rejects syntax outside its implemented subset, so
do not claim full AV2 conformance. Per-tool status is in the AV2 unit documents;
a claim without a recorded `dav2d` run is not evidence.

**Which codecs are intentionally absent?**

H.264/AVC, H.265/HEVC, AV1, VP8, VP9, MPEG-2, ProRes, DNxHD, and every legacy
audio and container format. The project uses a rolling window: a new standard
replaces the previous generation rather than accumulating compatibility modes.
See the matrix in [`CONCEPT.md`](../CONCEPT.md).

## Reporting issues

**Why are there two issue forms — a strict one and an easy one?**

They encode different evidence standards, not different classes of users.

- The strict forms (`bug_report`, `feature_request`, `codec_conformance`) cover
  claims that must be actionable: a reproducible defect, a conformance mismatch,
  a performance regression, a security report. Without the command, oracle
  version, stream hash, and `cmp`/first-differing-sample output, such a report
  cannot be triaged or fixed, and it turns the tracker into noise. Those fields
  are mandatory, and a missing reproduction is closed as `status/needs-info`.
- The easy form (`easy.yml`, label `community/newcomer`) is for general users: a
  question, a vague failure, "this surprised me". It has no required fields, so
  the barrier is low. A maintainer triages it and, if it is a real defect, moves
  it to the strict template and relabels it.

The apparent contradiction — "the project demands exact evidence" versus "there
is a form with no requirements" — is deliberate. Evidence is mandatory where a
claim can block a release, a conformance statement, or a security fix; it is not
a gate for asking a question. Both forms land in the same triage queue, and the
label tells maintainers which standard applies.

## Running the external oracles

**How do I verify a stream against VTM or VVdeC?**

```bash
# VTM 24.0
DecoderApp -b stream.vvc -o decoded.yuv
# VVdeC 3.3
vvdecapp -b stream.vvc -o decoded.yuv

# Project harnesses (kept outside the tree)
bash $ORACLE/vvc-conf/run_conformance.sh            # strict 25 cases, 50/50
bash $ORACLE/vvc-conf-full/run_conformance_full.sh  # full 266-check matrix
```

Compare the decoder output byte-for-byte with the reconstruction written by
`vvc_dump` / `vvc_multitool_dump` (`<prefix>.recon.yuv`). An oracle that does not
support the feature (for example VVdeC with 12-bit) is not evidence either way.

**How do I verify AV2 against dav2d?**

```bash
# Encode, then compare the reconstruction with dav2d's output
cargo run -p rustmpeg-video --example av2_encode -- $WORK/out.ivf \
    3 64 64 100 0 0 0 420 0 0 0 0 0 0 0 8 0 0 $WORK/rec.y4m
dav2d -i $WORK/out.ivf -o $WORK/dav.y4m \
    --threads 1 --framedelay 1 --strict 1

# Or run the ignored, byte-exact tests (DAV2D points at the binary)
DAV2D=$ORACLE/dav2d \
  cargo test -p rustmpeg-video av2_segmentation_decoder_dav2d_byte_exact -- --ignored
```

Compare Y4M frame payloads, not headers. AVM is used as a second encoder oracle
for streams the project's encoder cannot yet produce; the exact AVM flags for the
segmentation attempt are recorded in
[`../av2/UNIT5_DECODER.md`](../av2/UNIT5_DECODER.md).

**Are oracle sources allowed in the repository?**

Oracle binaries and streams stay outside the tracked tree. Recorded test data is
allowed only when its license is compatible and its source is recorded in
`THIRD_PARTY.md`.

## Adding a codec

**How do I add or extend a codec?**

1. Confirm the codec is in the supported matrix; a new codec replaces a
   generation, it does not get added for compatibility.
2. Implement from the normative specification and cite the clause and test vector
   in the PR.
3. Put the CPU reference first. No GPU path is written before the CPU result is
   correct and tested.
4. Add unit tests, a fuzz target for new parsers, and limits against hostile
   input.
5. Add the codec-side accelerator trait only if a measured whole-frame GPU stage
   clears the 10% margin.
6. Update `docs/`, `THIRD_PARTY.md`/`NOTICE` if third-party code is involved,
   and run the full check list and the license audit.

**Where do I document a codec?**

Reader-facing status, limits, and usage go in `docs/` (see
[`../README.md`](../README.md)). Wave plans, task briefs, and agent reports go in
the local, git-ignored `docsagents/` workspace (not part of this repository).

## Licensing and provenance

**What license does the project use?**

Apache-2.0 for original code, with the recorded BSD/Apache-2.0 notices for derived
portions. There is no CLA, no GPL/LGPL build variant, and no runtime loading of
copyleft code. `rustmpeg-video` declares
`Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause` because it contains VTM- and
dav2d-derived material.

**Which licenses are acceptable?**

Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, Unlicense, CC0-1.0,
Unicode-3.0, and Apache-2.0 WITH LLVM-exception. GPL, LGPL, AGPL, MPL, SSPL,
BUSL, CDLA, and unclear licenses are rejected.

**May I read FFmpeg source while implementing?**

Yes, to understand behaviour and edge cases. Copying, porting, or transcribing it
is forbidden, and seeing restricted source must be disclosed in the PR. See
[`../../AGENTS.md`](../../AGENTS.md#provenance-policy).

**Is VVC royalty-free?**

No. VVC is covered by patent pools. AV2, Opus, IAMF, and APV are designed as
royalty-free standards; only describe a codec as royalty-free with the
organisation that makes that claim.

## GPU policy

**When does a stage run on the GPU?**

Only when a release-mode measurement that includes allocation, upload, dispatch,
synchronization, and read-back shows it is at least 10% faster than the CPU at
that frame size. The decision is per stage and per frame size, made by a
calibrated scheduler, not a global "GPU mode" switch.

**Can I dispatch one kernel per coding block?**

No. Per-block dispatch and per-block read-back are forbidden; at 32x32 or 64x64
the transfer cost dominates. Entropy coding, bitstream parsing, and block-level
RDO never run on the GPU.

**What happens if the GPU result differs from the CPU?**

The GPU path is disabled for the session and the CPU reference keeps producing
the same output. A GPU path that changes the bitstream, the frame, or the reported
statistics is a bug, not an optimisation.
