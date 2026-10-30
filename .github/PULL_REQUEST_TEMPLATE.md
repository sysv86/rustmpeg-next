## Summary

<!-- What changes, why, and which crate/module owns the change. One paragraph. -->

## Linked issue

<!-- Use "Closes #123" or "Refs #123". A PR without a linked issue needs a
maintainer-approved rationale below. -->

Closes #

## Type of change

- [ ] Bug fix (no behavior change outside the reported defect)
- [ ] New feature within the current `docs/CONCEPT.md` matrix
- [ ] Codec conformance / bit-exactness fix
- [ ] Performance change (measured, see below)
- [ ] Refactor (no functional change; justified below)
- [ ] Documentation / CI / tooling

## Scope guard

- [ ] This diff contains only the change described above; no unrelated
      refactors, renames, formatting churn, or dependency bumps.
- [ ] No legacy codec, container, or transport was added.
- [ ] No `unsafe`, no `*-sys` crate, no C/C++/assembly build step, no `build.rs`
      that compiles native code.
- [ ] No new public API without documentation and tests.
- [ ] No new dependency, or the dependency is justified below and was verified
      as latest stable, permissively licensed, and free of native builds.
- [ ] Layering is unchanged: `core -> format / audio / image / video -> vulkan`.
- [ ] New parsing code has bounds checks, size limits, and tests.
- [ ] For a codec/GPU change, the CPU reference and GPU kernel were updated in
      the same commit and the bit-exact tests were rerun.

## Verification matrix

Paste the exact command and its result. Do not write "passes" without output.

| Check | Command | Result |
| :-- | :-- | :-- |
| Format | `cargo fmt --all -- --check` | |
| Check | `cargo check --workspace` | |
| Clippy | `cargo clippy --workspace --all-targets -- -D warnings` | |
| Tests | `cargo test --workspace` | |
| License audit | `python3 scripts/check-licenses.py` | |
| Docs | `cargo doc --workspace --no-deps` (if public API changed) | |

Skipped checks and the reason:

<!-- Example: "GPU tests skipped: no adapter in this environment." Never claim a
check passed unless it was run. -->

## External oracle results

Required for any bitstream, reconstruction, or filter change. Name the oracle
and version; FFmpeg is a black-box comparator only.

| Oracle and version | Vector / input hash | Command | Result (`cmp` / SHA-256) |
| :-- | :-- | :-- | :-- |
| | | | |

First differing byte/sample, if any:

## Performance claims

- [ ] This PR makes no performance claim.
- [ ] This PR makes a performance claim and the numbers below are from a
      release build, include transfer/allocation/read-back for GPU paths, and
      name the hardware, resolution, and measurement method.

| Path | Resolution / input | Before | After | Delta | Command |
| :-- | :-- | :-- | :-- | :-- | :-- |
| | | | | | |

## Licensing and provenance

- [ ] Every new source file has the correct SPDX header
      (`Apache-2.0`, or `Apache-2.0 AND BSD-3-Clause` / `AND BSD-2-Clause` for
      derived VVC/AV2 files).
- [ ] No existing copyright notice, SPDX line, or derivation comment was removed
      or altered.
- [ ] No copyleft (GPL/LGPL/AGPL/MPL/SSPL/BUSL/CDLA) code, data, or dependency
      was added, directly or transitively.
- [ ] Any reused third-party code or conformance data is permissively licensed,
      recorded in `THIRD_PARTY.md` and `NOTICE`, and attributed in the file.
- [ ] Any new dependency is in `[workspace.dependencies]`, wired into exactly
      one crate, and justified below.
- [ ] Test data is generated or license-compatible, with its source and license
      recorded.
- [ ] No code was transcribed from FFmpeg, libav*, GStreamer, or another
      restricted implementation.

Dependency or provenance justification:

<!-- Why is it necessary, which crate uses it, its license and latest version. -->

## Reviewer notes

<!-- Known limitations, follow-up work, and anything a reviewer should not
assume works. -->

## AI assistance and claims

- [ ] Every factual, performance, conformance, and licensing claim in this PR
      is backed by a command or file that was actually run or read in this
      branch.
- [ ] No generated text, benchmark, or test output was invented or
      extrapolated.
- [ ] If AI assistance was used, I reviewed the full diff and can explain every
      changed line.
