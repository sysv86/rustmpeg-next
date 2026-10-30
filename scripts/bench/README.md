# Benchmark scripts

`scripts/bench/` contains the reproducible benchmark pipeline for rustmpeg. The
driver only uses the Python standard library. All other programs are external
oracles: they are executed as black boxes and are never linked into a rustmpeg
crate.

- `bench.py` — generate clips, encode/decode, measure PSNR/SSIM, compute BD-rate,
  and update `docs/benchmarks/BENCHMARK.md`.
- `fetch_oracles.sh` — clone and build the pinned oracles into a cache directory
  outside the repository.
- `oracles.env` — the single source of truth for oracle revisions.

## Requirements

Common tools:

```sh
sudo apt-get install -y python3 ffmpeg flac
cargo build --release --examples
```

Building the AVM, SVT-AV1, dav2d, and liblc3 oracles additionally needs:

```sh
sudo apt-get install -y build-essential git cmake ninja-build meson nasm yasm \
    pkg-config libxxhash-dev
```

The repository toolchain is pinned in `rust-toolchain.toml` and needs nightly for
`std::simd`. Examples must be built in release mode:

```sh
cargo build --release --examples
```

## Online run

Build the pinned oracles once:

```sh
scripts/bench/fetch_oracles.sh --dest "$HOME/.cache/rustmpeg-bench/oracles"
```

The build is selective when a full oracle set is not needed:

```sh
# Only the AV1 and LC3 references.
scripts/bench/fetch_oracles.sh --dest "$HOME/.cache/rustmpeg-bench/oracles" \
  --only svt,lc3

# Every oracle except AVM (the slowest reference build).
scripts/bench/fetch_oracles.sh --dest "$HOME/.cache/rustmpeg-bench/oracles" \
  --skip avm
```

`--only` and `--skip` take a comma-separated subset of `svt,avm,dav2d,lc3`;
`--only all` (the default) and `--only none` are also accepted. Add
`--best-effort` to turn a failed third-party build into a recorded `SKIP`
instead of a non-zero exit, and `--build-timeout SECONDS` to bound each build.
A skipped oracle is written to `skipped` in `oracles.json`; `bench.py` then lists
the comparison under "Not benchmarked in this run" instead of inventing a row.
`--best-effort` never substitutes a different build or a fabricated version.

Then run the full profile:

```sh
python3 scripts/bench/bench.py \
  --profile full \
  --oracle-dir "$HOME/.cache/rustmpeg-bench/oracles" \
  --out docs/benchmarks/BENCHMARK.md \
  --json /tmp/rustmpeg-bench.json
```

The full profile is 1920x1080, 10 frames, two clips, five AV2 QP points, five AV1
CRF points, FLAC, and three LC3 bitrates. It is intentionally bounded; it is not a
replacement for a large-corpus codec evaluation.

Every external command is bounded by `--command-timeout` (default 1800 s, `0`
disables it), and the whole run is bounded by `--time-budget` (default `0`, no
overall limit). The workflow passes `--command-timeout 2400 --time-budget 8100`
so the job always finishes before the step timeout. A reference sweep that hits
either bound is dropped as a whole and listed under "Not benchmarked in this run"
with the exact reason; a partial sweep is never turned into a row or a BD-rate.
A timeout in a rustmpeg encode/decode still fails the run, because that is a
defect in our own code rather than a slow third-party oracle.

The SVT-AV1 cross-codec sweep runs before the AVM sweep so that a slow AV2
reference encoder cannot starve the cheaper AV1 anchor.

A fast smoke run:

```sh
python3 scripts/bench/bench.py \
  --profile smoke \
  --oracle-dir "$HOME/.cache/rustmpeg-bench/oracles" \
  --out docs/benchmarks/BENCHMARK.md \
  --json /tmp/rustmpeg-bench-smoke.json
```

## Local dry run (64x64, 2 frames)

This is the smallest case that still exercises a real encode, decode, PSNR/SSIM
measurement, BD-rate path, and a table row:

```sh
python3 scripts/bench/bench.py \
  --profile smoke \
  --width 64 --height 64 --frames 2 \
  --clips graphics \
  --allow-missing-oracles \
  --out /tmp/rustmpeg-bench-dry.md \
  --json /tmp/rustmpeg-bench-dry.json \
  --workdir /tmp/rustmpeg-bench-work
```

`--allow-missing-oracles` omits a comparison whose oracle is absent and records
the reason in the "Not benchmarked in this run" table. It never writes invented
rows. The command still fails if the rustmpeg path, FFmpeg metric step, or any
available oracle fails.

To keep and inspect the bitstreams and Y4M files, add `--keep` and a `--workdir`.

## Offline run

If the oracle cache is already populated, no network is needed:

```sh
BENCH_OFFLINE=1 scripts/bench/fetch_oracles.sh \
  --dest "$HOME/.cache/rustmpeg-bench/oracles"

python3 scripts/bench/bench.py \
  --profile full \
  --oracle-dir "$HOME/.cache/rustmpeg-bench/oracles" \
  --out docs/benchmarks/BENCHMARK.md \
  --json /tmp/rustmpeg-bench-offline.json
```

`fetch_oracles.sh` checks that the binaries of every selected oracle exist in
`bin/` and exits non-zero if one is missing. With `--best-effort` a missing or
failed build is recorded as a `SKIP` instead. `bench.py` itself never downloads
anything.

## What each oracle does

| Tool | Built from | Used for |
| :-- | :-- | :-- |
| `SvtAv1EncApp` | SVT-AV1 `v4.2.0` | Encodes the AV1 cross-codec reference. Decoded by FFmpeg. |
| `avmenc` | AVM `v1.0.0` | Encodes the same-codec AV2 reference. |
| `dav2d` | commit `d7df5418` | Decodes AVM AV2 streams. |
| `avmdec` | AVM `v1.0.0` | Fallback AV2 decoder, used only when `dav2d` is absent. The table records which decoder ran. |
| `elc3` / `dlc3` | liblc3 `v1.1.3` | LC3 reference encoder and decoder. |
| `flac` | distribution package | libFLAC reference encoder/decoder. |
| `ffmpeg` | distribution package | Clip generation, AV1 decode, PSNR, SSIM, PCM conversion. |

The exact revisions are in `oracles.env`. Update the pins and
`docs/benchmarks/BENCHMARK.md` in the same change; mixing oracle revisions makes
historical comparisons invalid.

## What lands in `BENCHMARK.md`

`bench.py` writes only between the markers:

```text
<!-- BENCHMARK-TABLE --> ... <!-- /BENCHMARK-TABLE -->
<!-- BENCHMARK-HISTORY --> ... <!-- /BENCHMARK-HISTORY -->
```

Each run records:

- UTC date and profile;
- git commit and whether the working tree was dirty;
- machine, CPU, logical CPU count, and RAM;
- Rust, FFmpeg, and oracle versions;
- source clip SHA-256 prefixes;
- per-point bitrate, PSNR-Y, SSIM, encode/decode MB/s, and bytes;
- BD-rate rows, including the cross-codec label;
- the exact commands that ran.

The previous latest block is moved into a collapsed `<details>` entry. The same
commit is replaced rather than duplicated. The oldest history entries are trimmed
only after 60 runs, with an explicit note; the workflow artifact keeps the full
JSON for each run.

If the native rustmpeg AV2 decoder rejects a tool that the encoder selected, the
driver tries `dav2d`/`avmdec` and accepts their pixels only when they hash-match
the encoder reconstruction. Otherwise it measures the reconstruction and labels
the row `encoder reconstruction (native decode failed: ...)`, with decode MB/s
`n/a`. The table never presents reconstruction pixels as decoded output.

## CI behavior

`.github/workflows/benchmark.yml` runs on:

- `workflow_dispatch` with `profile: full|smoke` (default `full`), an `oracles`
  selection (`auto`, `all`, `none`), and `update_results`;
- a 24-hour schedule;
- pull requests that change the workflow or these scripts, always with the
  `smoke` profile and without committing results.

The workflow restores a cached oracle directory, builds missing oracles from the
pins, builds the examples in release mode, runs `bench.py`, and uploads the JSON,
the Markdown, the oracle manifest, and the oracle build logs as an artifact.

`oracles: auto` builds SVT-AV1 and LC3 for the `smoke` profile and every oracle
for `full`; `all` always attempts every oracle; `none` skips oracle builds
entirely. Builds are attempted with `--best-effort` and a per-build timeout, so a
third-party build that fails or exceeds the timeout is reported as `SKIP` and the
affected comparison is listed under "Not benchmarked in this run". `bench.py` is
always called with `--allow-missing-oracles`, so an absent oracle never becomes a
fabricated row and never fails the job. A failing rustmpeg encode, decode,
frame-count check, or metric parse still fails the job.

It commits `docs/benchmarks/BENCHMARK.md` back to the branch only for schedule and
manual runs after a successful benchmark and only when `update_results` is true.
Because the branch receives frequent CI commits, the step retries fetch/rebase/
push up to five times. If every attempt fails, or the branch is protected, it
emits a warning and the generated document stays in the run artifact. It uses the
repository `GITHUB_TOKEN`; no crates.io token is required.


## Known CI limitations

- GitHub-hosted runners are shared and not CPU-pinned. Absolute MB/s is noisy;
  compare repeated runs and medians.
- AVM is reference software and can take minutes per full sweep. The cache key is
  derived from `oracles.env`; the first run after a pin change rebuilds it. The
  workflow bounds each build command with `--build-timeout 1200` and the
  benchmark with `--command-timeout 2400 --time-budget 8100`; an AVM sweep that
  does not finish inside those bounds is reported as skipped, not approximated.
- AVM and dav2d track a moving AV2 draft. If `dav2d` rejects an AVM stream, the
  run fails rather than publishing a mismatched comparison. Use the recorded
  `avmdec` fallback only when `dav2d` is genuinely unavailable, not to hide a
  version mismatch.
- The full profile is deliberately 10 frames and two clips. It is a regression
  benchmark, not a codec quality verdict. The smoke profile uses 320x240 because
  the current AV2 encoder rejects some 320x180 inter blocks.
- The rustmpeg AV2 decoder does not yet accept every tool its encoder can select.
  Rows fall back to a labelled encoder-reconstruction measurement when no
  independent decoder reproduces the reconstruction exactly.
- Format and image crates have no runnable harness yet; see the table in
  `BENCHMARK.md`.
