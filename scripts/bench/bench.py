#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproducible rustmpeg benchmark driver.

The driver generates deterministic clips with FFmpeg, runs the rustmpeg
examples and the external reference encoders, decodes every stream with an
independent decoder, measures PSNR/SSIM with FFmpeg, computes Bjontregaard-style
BD-rate values, and renders the result into the marked section of
docs/benchmarks/BENCHMARK.md.

Only the Python standard library is required.  External tools are oracles and
are never linked into the project.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE_START = "<!-- BENCHMARK-TABLE -->"
TABLE_END = "<!-- /BENCHMARK-TABLE -->"
HISTORY_START = "<!-- BENCHMARK-HISTORY -->"
HISTORY_END = "<!-- /BENCHMARK-HISTORY -->"
HISTORY_ENTRY = "<!-- BENCH-ENTRY -->"
MAX_HISTORY = 60

PROFILES = {
    "full": {
        "width": 1920,
        "height": 1080,
        "frames": 10,
        "fps": 30,
        "av2_qp": [60, 100, 140, 180, 220],
        "av1_crf": [22, 32, 42, 52, 63],
        "audio_seconds": 2.0,
        "lc3_kbps": [32, 64, 96],
    },
    "smoke": {
        "width": 320,
        "height": 240,
        "frames": 4,
        "fps": 30,
        "av2_qp": [60, 120, 180, 220],
        "av1_crf": [34, 44, 54, 63],
        "audio_seconds": 1.0,
        "lc3_kbps": [32, 64, 96],
    },
}

VIDEO_CLIPS = {
    "graphics": {"noise": None, "seed": 0},
    "noise": {"noise": 12, "seed": 1234},
}

CODEC_INFO = {
    "rustmpeg-av2": {
        "codec": "AV2",
        "implementation": "rustmpeg-video",
        "reference": "AVM (same codec)",
        "decoder": "rustmpeg-video",
        "cross_codec": False,
    },
    "avm-av2": {
        "codec": "AV2",
        "implementation": "AVM avmenc",
        "reference": "AVM",
        "decoder": "dav2d",
        "cross_codec": False,
    },
    "svt-av1": {
        "codec": "AV1",
        "implementation": "SVT-AV1",
        "reference": "SVT-AV1",
        "decoder": "ffmpeg",
        "cross_codec": True,
    },
}


class BenchError(RuntimeError):
    """A benchmark step failed; no partial result may be published."""


class BenchTimeout(BenchError):
    """A single command exceeded its time bound and was killed."""

    def __init__(self, label: str, command: str, seconds: float) -> None:
        self.label = label
        self.command = command
        self.seconds = seconds
        super().__init__(
            f"{label} exceeded the {seconds:.0f}s command limit\n"
            f"command: {command}"
        )


class BenchBudgetExhausted(BenchError):
    """The overall benchmark time budget ran out before this command."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        super().__init__(
            f"benchmark time budget of {seconds:.0f}s was exhausted before "
            "this step"
        )


@dataclass
class Point:
    codec: str
    clip: str
    param: str
    bytes: int
    bitrate_kbps: float
    psnr_y: float
    ssim_all: float
    enc_seconds: float
    dec_seconds: float | None
    enc_mbps: float
    dec_mbps: float | None
    decoder: str


@dataclass
class AudioPoint:
    codec: str
    implementation: str
    reference: str
    param: str
    bytes: int
    ratio: float
    snr_db: float | None
    lossless: bool
    enc_seconds: float
    dec_seconds: float
    enc_mbps: float
    dec_mbps: float
    quality: str
    lag: int = 0


@dataclass
class Runner:
    oracle_dir: Path | None
    target_dir: Path
    env: dict[str, str]
    commands: list[str] = field(default_factory=list)
    allow_missing: bool = False
    keep: bool = False
    command_timeout: float | None = None
    time_budget: float | None = None
    deadline: float | None = None

    def __post_init__(self) -> None:
        if self.oracle_dir is not None:
            bin_dir = self.oracle_dir / "bin"
            self.env["PATH"] = f"{bin_dir}{os.pathsep}{self.env.get('PATH', '')}"
            self.env["LD_LIBRARY_PATH"] = (
                f"{bin_dir}{os.pathsep}{self.env.get('LD_LIBRARY_PATH', '')}"
            )

    def which(self, name: str) -> Path | None:
        found = shutil.which(name, path=self.env.get("PATH"))
        return Path(found) if found else None

    def run(
        self,
        cmd: list[str | Path],
        label: str,
        cwd: Path | None = None,
        check: bool = True,
    ) -> tuple[float, str]:
        argv = [str(part) for part in cmd]
        self.commands.append(shlex.join(argv))
        limit = self.command_timeout
        budget_bound = False
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise BenchBudgetExhausted(self.time_budget or 0)
            if limit is None or remaining <= limit:
                limit = remaining
                budget_bound = True
        started = time.perf_counter()
        print(f"[bench] {label}: {shlex.join(argv)}", file=sys.stderr, flush=True)
        try:
            proc = subprocess.run(
                argv,
                cwd=str(cwd) if cwd else None,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=limit,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - started
            print(
                f"[bench] {label}: killed after {elapsed:.1f}s "
                f"(limit {limit:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
            if budget_bound:
                raise BenchBudgetExhausted(self.time_budget or 0) from None
            raise BenchTimeout(label, shlex.join(argv), limit or elapsed) from None
        elapsed = time.perf_counter() - started
        print(
            f"[bench] {label}: {elapsed:.1f}s exit={proc.returncode}",
            file=sys.stderr,
            flush=True,
        )
        output = proc.stdout or ""
        if check and proc.returncode != 0:
            tail = "\n".join(output.splitlines()[-40:])
            raise BenchError(
                f"{label} failed with exit code {proc.returncode}\n"
                f"command: {shlex.join(argv)}\n{tail}"
            )
        return elapsed, output


def project_binary(runner: Runner, name: str) -> Path:
    path = runner.target_dir / "release" / "examples" / name
    if not path.is_file():
        raise BenchError(
            f"missing benchmark example {path}; run "
            "`cargo build --release --examples` first"
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def y4m_frame_count(path: Path) -> int:
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.readline()
        if not header.startswith(b"YUV4MPEG2"):
            raise BenchError(f"{path} is not a Y4M stream")
        tokens = header.decode("ascii", "replace").split()
        width = height = 0
        chroma = "420jpeg"
        for token in tokens[1:]:
            if token.startswith("W"):
                width = int(token[1:])
            elif token.startswith("H"):
                height = int(token[1:])
            elif token.startswith("C"):
                chroma = token[1:]
        if not width or not height:
            raise BenchError(f"{path}: missing Y4M dimensions")
        sample_bytes = 2 if re.search(r"p(10|12|16)", chroma) else 1
        if chroma.startswith("444"):
            chroma_w, chroma_h = width, height
        elif chroma.startswith("422"):
            chroma_w, chroma_h = (width + 1) // 2, height
        elif chroma.startswith("400") or chroma.startswith("mono"):
            chroma_w = chroma_h = 0
        else:
            chroma_w, chroma_h = (width + 1) // 2, (height + 1) // 2
        frame_bytes = (
            width * height + 2 * chroma_w * chroma_h
        ) * sample_bytes
        frames = 0
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith(b"FRAME"):
                raise BenchError(f"{path}: malformed frame marker at {handle.tell()}")
            if handle.tell() + frame_bytes > size:
                raise BenchError(f"{path}: truncated frame {frames}")
            handle.seek(frame_bytes, os.SEEK_CUR)
            frames += 1
        return frames


def ensure_frames(path: Path, expected: int) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise BenchError(f"encoder/decoder produced no output: {path}")
    actual = y4m_frame_count(path)
    if actual != expected:
        raise BenchError(f"{path}: expected {expected} frames, found {actual}")


def y4m_pixel_sha256(path: Path) -> str:
    """Hash Y4M frame payloads while ignoring header and FRAME metadata."""
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.readline()
        if not header.startswith(b"YUV4MPEG2"):
            raise BenchError(f"{path} is not a Y4M stream")
        tokens = header.decode("ascii", "replace").split()
        width = height = 0
        chroma = "420jpeg"
        for token in tokens[1:]:
            if token.startswith("W"):
                width = int(token[1:])
            elif token.startswith("H"):
                height = int(token[1:])
            elif token.startswith("C"):
                chroma = token[1:]
        sample_bytes = 2 if re.search(r"p(10|12|16)", chroma) else 1
        if chroma.startswith("444"):
            chroma_w, chroma_h = width, height
        elif chroma.startswith("422"):
            chroma_w, chroma_h = (width + 1) // 2, height
        elif chroma.startswith("400") or chroma.startswith("mono"):
            chroma_w = chroma_h = 0
        else:
            chroma_w, chroma_h = (width + 1) // 2, (height + 1) // 2
        frame_bytes = (width * height + 2 * chroma_w * chroma_h) * sample_bytes
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith(b"FRAME") or handle.tell() + frame_bytes > size:
                raise BenchError(f"{path}: malformed Y4M frame")
            digest.update(handle.read(frame_bytes))
    return digest.hexdigest()


def generate_clip(
    runner: Runner, ffmpeg: Path, work: Path, name: str, width: int, height: int,
    frames: int, fps: int,
) -> Path:
    spec = VIDEO_CLIPS[name]
    path = work / f"source-{name}-{width}x{height}-{frames}.y4m"
    if path.is_file():
        ensure_frames(path, frames)
        return path
    filter_chain = "format=yuv420p"
    if spec["noise"]:
        filter_chain = (
            f"noise=alls={spec['noise']}:allf=t+u:"
            f"all_seed={spec['seed']},format=yuv420p"
        )
    runner.run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}",
            "-vf", filter_chain,
            "-frames:v", str(frames),
            "-f", "yuv4mpegpipe", path,
        ],
        f"generate {name} clip",
    )
    ensure_frames(path, frames)
    return path


def measure_quality(
    runner: Runner, ffmpeg: Path, dist: Path, source: Path, label: str,
) -> tuple[float, float]:
    _, output = runner.run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info",
            "-i", dist, "-i", source,
            "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-",
        ],
        f"PSNR {label}",
    )
    match = re.search(r"PSNR y:([0-9.]+).*?average:([0-9.]+)", output)
    if not match:
        raise BenchError(f"could not parse PSNR for {label}:\n{output[-2000:]}")
    psnr_y = float(match.group(1))

    _, output = runner.run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info",
            "-i", dist, "-i", source,
            "-lavfi", "[0:v][1:v]ssim", "-f", "null", "-",
        ],
        f"SSIM {label}",
    )
    match = re.search(r"SSIM .*?All:([0-9.]+)", output)
    if not match:
        raise BenchError(f"could not parse SSIM for {label}:\n{output[-2000:]}")
    return psnr_y, float(match.group(1))


def decode_rustmpeg_av2(
    runner: Runner, ivf: Path, output: Path, reconstruction: Path, frames: int,
    bit_depth: int = 8,
) -> tuple[float | None, str, Path]:
    """Decode a rustmpeg AV2 stream, or fall back to its reconstruction.

    The experimental decoder does not implement every tool its encoder can
    select (for example sub-block transforms).  When native decoding fails, an
    independent AV2 decoder is tried, but its pixels are accepted only if they
    match the encoder reconstruction exactly.  If neither path yields the
    reconstruction, the reconstruction itself is measured and the row is
    labelled accordingly; no metric is invented and decode throughput is `n/a`.
    """
    decode = project_binary(runner, "av2_decode")
    native_error: str = "decode failed"
    try:
        elapsed, _ = runner.run(
            [decode, ivf, output, str(bit_depth)],
            f"rustmpeg AV2 decode {ivf.name}",
        )
        ensure_frames(output, frames)
        return elapsed, "rustmpeg-video", output
    except (BenchTimeout, BenchBudgetExhausted):
        raise
    except BenchError as error:
        lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        if lines:
            native_error = lines[-1]
        if len(native_error) > 140:
            native_error = native_error[:137] + "..."
        output.unlink(missing_ok=True)

    expected_digest = y4m_pixel_sha256(reconstruction)
    for tool_name in ("dav2d", "avmdec"):
        tool = runner.which(tool_name)
        if tool is None:
            continue
        external = output.with_name(output.stem + f"-{tool_name}.y4m")
        try:
            if tool_name == "dav2d":
                elapsed, _ = runner.run(
                    [
                        tool, "-i", ivf, "-o", external, "--muxer",
                        "yuv4mpeg2", "--limit", str(frames), "-q",
                    ],
                    f"dav2d check decode {ivf.name}",
                )
            else:
                elapsed, _ = runner.run(
                    [tool, f"--limit={frames}", "-o", external, ivf],
                    f"avmdec check decode {ivf.name}",
                )
            ensure_frames(external, frames)
            if y4m_pixel_sha256(external) == expected_digest:
                external.replace(output)
                return elapsed, tool_name, output
            native_error += f"; {tool_name} reconstruction mismatch"
        except (BenchTimeout, BenchBudgetExhausted):
            raise
        except BenchError:
            pass
        external.unlink(missing_ok=True)
    return (
        None,
        f"encoder reconstruction (native decode failed: {native_error})",
        reconstruction,
    )


def decode_av2(
    runner: Runner, ivf: Path, output: Path, frames: int, bit_depth: int = 8,
) -> tuple[float, str]:
    dav2d = runner.which("dav2d")
    if dav2d is not None:
        elapsed, log = runner.run(
            [
                dav2d, "-i", ivf, "-o", output, "--muxer", "yuv4mpeg2",
                "--limit", str(frames), "-q",
            ],
            f"dav2d decode {ivf.name}",
        )
        ensure_frames(output, frames)
        return elapsed, "dav2d"
    avmdec = runner.which("avmdec")
    if avmdec is not None:
        elapsed, log = runner.run(
            [avmdec, f"--limit={frames}", "-o", output, ivf],
            f"avmdec decode {ivf.name}",
        )
        ensure_frames(output, frames)
        return elapsed, "avmdec"
    raise BenchError("AV2 decode requires dav2d or avmdec")


def decode_with_ffmpeg(
    runner: Runner, ffmpeg: Path, stream: Path, output: Path, frames: int,
) -> float:
    elapsed, _ = runner.run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", stream, "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", output,
        ],
        f"ffmpeg decode {stream.name}",
    )
    ensure_frames(output, frames)
    return elapsed


def encode_rustmpeg_av2(
    runner: Runner, source: Path, work: Path, width: int, height: int,
    frames: int, qp: int,
) -> tuple[Path, Path, float]:
    out = work / f"rustmpeg-qp{qp}.ivf"
    rec = work / f"rustmpeg-qp{qp}-rec.y4m"
    encode = project_binary(runner, "av2_encode")
    elapsed, _ = runner.run(
        [
            encode, out, str(frames), str(width), str(height), str(qp),
            "0", "0",              # log2 tile cols / rows
            "1", "420",            # deblock, chroma layout
            "1", "1", "1", "1",    # cdef, ccso, gdf, restoration
            "1", "1", "1",         # newmv, switchable interp, compound
            "8",                   # bit depth
            "1", "0",              # vaq, delta_q
            rec, "0",              # reconstruction, vbr kbps
            "1", "0",              # warped motion, obmc
            source,                # input y4m
            "0", "0",              # palette, intrabc
        ],
        f"rustmpeg AV2 qp={qp}",
    )
    ensure_frames(rec, frames)
    if not out.is_file() or out.stat().st_size == 0:
        raise BenchError(f"rustmpeg AV2 produced no bitstream at qp={qp}")
    return out, rec, elapsed


def encode_avm(
    runner: Runner, avmenc: Path, source: Path, work: Path, width: int,
    height: int, frames: int, qp: int,
) -> tuple[Path, float]:
    out = work / f"avm-qp{qp}.ivf"
    elapsed, _ = runner.run(
        [
            avmenc, "--codec=av2", "--passes=1", "--end-usage=q",
            f"--qp={qp}", "--cpu-used=8", "--threads=1",
            f"--limit={frames}", f"--width={width}", f"--height={height}",
            "--ivf", "--quiet", "-o", out, source,
        ],
        f"AVM avmenc qp={qp}",
    )
    if not out.is_file() or out.stat().st_size == 0:
        raise BenchError(f"AVM produced no bitstream at qp={qp}")
    return out, elapsed


def encode_svt(
    runner: Runner, svt: Path, source: Path, work: Path, frames: int, crf: int,
) -> tuple[Path, float]:
    out = work / f"svt-crf{crf}.ivf"
    elapsed, _ = runner.run(
        [
            svt, "-i", source, "-b", out, "--preset", "8",
            "--crf", str(crf), "--keyint", str(max(frames, 1)),
            "--frames", str(frames), "--lp", "1", "--progress", "0",
        ],
        f"SVT-AV1 crf={crf}",
    )
    if not out.is_file() or out.stat().st_size == 0:
        raise BenchError(f"SVT-AV1 produced no bitstream at crf={crf}")
    return out, elapsed


def skip_reason(error: BenchError) -> str:
    """One-line reason recorded in the 'Not benchmarked' table."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return lines[0] if lines else error.__class__.__name__


def discard_partial(directory: Path, prefix: str) -> None:
    """Remove outputs from an aborted sweep so no partial row can be read."""
    for path in directory.glob(f"{prefix}*"):
        if path.is_file():
            path.unlink(missing_ok=True)


def run_video(
    runner: Runner, ffmpeg: Path, work: Path, config: dict, args: argparse.Namespace,
) -> tuple[list[Point], list[dict], dict[str, Path]]:
    width, height = config["width"], config["height"]
    frames, fps = config["frames"], config["fps"]
    raw_bytes = width * height * 3 // 2 * frames
    points: dict[str, dict[str, list[Point]]] = {}
    skipped: list[dict] = []
    sources: dict[str, Path] = {}

    for clip in args.clips:
        clip_dir = work / f"video-{clip}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        sources[clip] = generate_clip(
            runner, ffmpeg, work, clip, width, height, frames, fps
        )

    svt = runner.which("SvtAv1EncApp")
    avmenc = runner.which("avmenc")
    av2_decoder = runner.which("dav2d") or runner.which("avmdec")

    for clip, source in sources.items():
        points[clip] = {}
        clip_dir = work / f"video-{clip}"

        # rustmpeg AV2. A budget stop is recorded and ends the video work; a
        # per-command timeout or a real failure still fails the run, because
        # that is a defect in our own encoder or decoder.
        clip_points: list[Point] = []
        try:
            for qp in config["av2_qp"]:
                out, rec, enc = encode_rustmpeg_av2(
                    runner, source, clip_dir, width, height, frames, qp
                )
                decoded = clip_dir / f"rustmpeg-qp{qp}.y4m"
                dec, decoder, pixels = decode_rustmpeg_av2(
                    runner, out, decoded, rec, frames
                )
                psnr, ssim = measure_quality(
                    runner, ffmpeg, pixels, source, f"rustmpeg {clip} qp={qp}"
                )
                size = out.stat().st_size
                clip_points.append(
                    Point(
                        codec="rustmpeg-av2", clip=clip, param=f"qp={qp}",
                        bytes=size,
                        bitrate_kbps=size * 8 * fps / frames / 1000,
                        psnr_y=psnr, ssim_all=ssim, enc_seconds=enc,
                        dec_seconds=dec, enc_mbps=raw_bytes / enc / 1e6,
                        dec_mbps=None if dec is None else raw_bytes / dec / 1e6,
                        decoder=decoder,
                    )
                )
        except BenchBudgetExhausted as error:
            discard_partial(clip_dir, "rustmpeg-")
            skipped.append(
                {"item": f"rustmpeg AV2 ({clip})", "reason": skip_reason(error)}
            )
            continue
        points[clip]["rustmpeg-av2"] = clip_points

        # SVT-AV1 (AV1): cross-codec comparison. It runs before the AVM sweep so
        # that a slow AV2 reference encoder cannot consume the whole budget.
        if svt is None:
            if not runner.allow_missing:
                raise BenchError(f"AV2 vs SVT-AV1 ({clip}): SvtAv1EncApp not found")
            skipped.append(
                {"item": f"AV2 vs SVT-AV1 ({clip})", "reason": "SvtAv1EncApp not found"}
            )
        else:
            clip_points = []
            try:
                for crf in config["av1_crf"]:
                    out, enc = encode_svt(runner, svt, source, clip_dir, frames, crf)
                    decoded = clip_dir / f"svt-crf{crf}.y4m"
                    dec = decode_with_ffmpeg(runner, ffmpeg, out, decoded, frames)
                    psnr, ssim = measure_quality(
                        runner, ffmpeg, decoded, source, f"SVT-AV1 {clip} crf={crf}"
                    )
                    size = out.stat().st_size
                    clip_points.append(
                        Point(
                            codec="svt-av1", clip=clip, param=f"crf={crf}",
                            bytes=size,
                            bitrate_kbps=size * 8 * fps / frames / 1000,
                            psnr_y=psnr, ssim_all=ssim, enc_seconds=enc,
                            dec_seconds=dec, enc_mbps=raw_bytes / enc / 1e6,
                            dec_mbps=raw_bytes / dec / 1e6, decoder="ffmpeg",
                        )
                    )
            except (BenchTimeout, BenchBudgetExhausted) as error:
                discard_partial(clip_dir, "svt-")
                skipped.append(
                    {"item": f"AV2 vs SVT-AV1 ({clip})", "reason": skip_reason(error)}
                )
            else:
                points[clip]["svt-av1"] = clip_points

        # AVM AV2
        if avmenc is None or av2_decoder is None:
            reason = "avmenc not found" if avmenc is None else "dav2d/avmdec not found"
            if not runner.allow_missing:
                raise BenchError(f"AV2 vs AVM ({clip}): {reason}")
            skipped.append(
                {"item": f"AV2 vs AVM ({clip})", "reason": reason}
            )
        else:
            clip_points = []
            try:
                for qp in config["av2_qp"]:
                    out, enc = encode_avm(
                        runner, avmenc, source, clip_dir, width, height, frames, qp
                    )
                    decoded = clip_dir / f"avm-qp{qp}.y4m"
                    dec, decoder = decode_av2(runner, out, decoded, frames)
                    psnr, ssim = measure_quality(
                        runner, ffmpeg, decoded, source, f"AVM {clip} qp={qp}"
                    )
                    size = out.stat().st_size
                    clip_points.append(
                        Point(
                            codec="avm-av2", clip=clip, param=f"qp={qp}",
                            bytes=size,
                            bitrate_kbps=size * 8 * fps / frames / 1000,
                            psnr_y=psnr, ssim_all=ssim, enc_seconds=enc,
                            dec_seconds=dec, enc_mbps=raw_bytes / enc / 1e6,
                            dec_mbps=raw_bytes / dec / 1e6, decoder=decoder,
                        )
                    )
            except (BenchTimeout, BenchBudgetExhausted) as error:
                discard_partial(clip_dir, "avm-")
                skipped.append(
                    {"item": f"AV2 vs AVM ({clip})", "reason": skip_reason(error)}
                )
            else:
                points[clip]["avm-av2"] = clip_points

    flat: list[Point] = []
    for clip in points:
        for codec in points[clip]:
            flat.extend(points[clip][codec])
    return flat, skipped, sources


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular normal equations")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for entry in range(column, size + 1):
            augmented[column][entry] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                for entry in range(column, size + 1):
                    augmented[row][entry] -= factor * augmented[column][entry]
    return [augmented[row][size] for row in range(size)]


def _polyfit(xs: list[float], ys: list[float], degree: int) -> list[float]:
    size = degree + 1
    matrix = [[0.0] * size for _ in range(size)]
    vector = [0.0] * size
    for row in range(size):
        for column in range(size):
            matrix[row][column] = sum(x ** (row + column) for x in xs)
        vector[row] = sum(y * x**row for x, y in zip(xs, ys))
    return _solve_linear(matrix, vector)


def _poly_integral(coefficients: list[float], low: float, high: float) -> float:
    return sum(
        coefficient * (high ** (power + 1) - low ** (power + 1)) / (power + 1)
        for power, coefficient in enumerate(coefficients)
    )


def bd_rate(anchor: list[tuple[float, float]], test: list[tuple[float, float]]) -> float | None:
    """Return percent BD-rate of test against anchor, or None if not computable.

    Each sample is (PSNR-Y dB, bitrate kbps).  The implementation fits a
    polynomial to log10(rate) over PSNR and integrates the difference on the
    overlapping PSNR interval.  Positive values mean the test encoder needs
    more bits for the same quality.
    """
    if len(anchor) < 3 or len(test) < 3:
        return None
    anchor = sorted(anchor)
    test = sorted(test)
    low = max(anchor[0][0], test[0][0])
    high = min(anchor[-1][0], test[-1][0])
    if high - low < 1e-3:
        return None
    degree = min(3, len(anchor) - 1, len(test) - 1)
    try:
        anchor_fit = _polyfit(
            [point[0] for point in anchor],
            [math.log10(point[1]) for point in anchor],
            degree,
        )
        test_fit = _polyfit(
            [point[0] for point in test],
            [math.log10(point[1]) for point in test],
            degree,
        )
    except ValueError:
        return None
    difference = [
        test_fit[index] - anchor_fit[index]
        for index in range(degree + 1)
    ]
    average = _poly_integral(difference, low, high) / (high - low)
    return (10**average - 1) * 100


def bd_rate_rows(points: list[Point]) -> list[dict]:
    by_key: dict[tuple[str, str], list[Point]] = {}
    for point in points:
        by_key.setdefault((point.clip, point.codec), []).append(point)
    rows: list[dict] = []
    for anchor, test, note in [
        ("avm-av2", "rustmpeg-av2", "same codec (AV2)"),
        ("svt-av1", "rustmpeg-av2", "cross-codec: AV1 anchor vs AV2 test, indicative only"),
    ]:
        for clip in sorted({point.clip for point in points}):
            anchor_points = by_key.get((clip, anchor))
            test_points = by_key.get((clip, test))
            if not anchor_points or not test_points:
                continue
            value = bd_rate(
                [(point.psnr_y, point.bitrate_kbps) for point in anchor_points],
                [(point.psnr_y, point.bitrate_kbps) for point in test_points],
            )
            rows.append(
                {
                    "anchor": CODEC_INFO[anchor]["implementation"],
                    "test": CODEC_INFO[test]["implementation"],
                    "clip": clip,
                    "bd_rate": value,
                    "note": note,
                }
            )
    for anchor, test in [("avm-av2", "rustmpeg-av2"), ("svt-av1", "rustmpeg-av2")]:
        values = [
            row["bd_rate"]
            for row in rows
            if row["anchor"] == CODEC_INFO[anchor]["implementation"]
            and row["test"] == CODEC_INFO[test]["implementation"]
            and row["bd_rate"] is not None
        ]
        if values:
            rows.append(
                {
                    "anchor": CODEC_INFO[anchor]["implementation"],
                    "test": CODEC_INFO[test]["implementation"],
                    "clip": "mean over clips",
                    "bd_rate": statistics.fmean(values),
                    "note": "arithmetic mean of per-clip BD-rate",
                }
            )
    return rows


def write_s16le(path: Path, samples: list[int]) -> None:
    payload = array.array("h", samples)
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


def read_s16le(path: Path) -> list[int]:
    payload = array.array("h")
    payload.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        payload.byteswap()
    return payload.tolist()


def read_s32le(path: Path) -> list[int]:
    payload = array.array("i")
    payload.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        payload.byteswap()
    return payload.tolist()


def aligned_snr(
    reference: list[int], actual: list[int], max_lag: int = 480,
    edge_skip: int = 480,
) -> tuple[float | None, int]:
    """Best SNR over the LC3 algorithmic-delay range.

    LC3 has a known 2.5 ms encoder/decoder delay, and not every implementation
    compensates it at the same point.  Searching +/-10 ms and reporting the
    winning lag keeps the comparison honest instead of picking one arbitrary
    offset; the lag is printed next to the SNR.
    """
    best: tuple[float | None, int] = (None, 0)

    def evaluate(lag: int) -> tuple[float | None, int]:
        start_ref = edge_skip + max(lag, 0)
        start_actual = edge_skip + max(-lag, 0)
        count = min(
            len(reference) - start_ref, len(actual) - start_actual
        ) - edge_skip
        if count <= 0:
            return None, lag
        signal = 0.0
        noise = 0.0
        for index in range(count):
            value = reference[start_ref + index]
            error = value - actual[start_actual + index]
            signal += float(value) * value
            noise += float(error) * error
        value = None if noise == 0.0 else (
            -math.inf if signal == 0.0 else 10 * math.log10(signal / noise)
        )
        return value, lag

    # Coarse search, then refine by one sample around the best coarse lag.
    coarse = list(range(-max_lag, max_lag + 1, 8))
    if coarse[-1] != max_lag:
        coarse.append(max_lag)
    for lag in coarse:
        value, _ = evaluate(lag)
        if value is not None and (best[0] is None or value > best[0]):
            best = (value, lag)
    if best[0] is None:
        return best
    center = best[1]
    best = (None, 0)
    for lag in range(max(-max_lag, center - 8), min(max_lag, center + 8) + 1):
        value, _ = evaluate(lag)
        if value is not None and (best[0] is None or value > best[0]):
            best = (value, lag)
    return best


def run_audio(
    runner: Runner, ffmpeg: Path, work: Path, config: dict, args: argparse.Namespace,
) -> tuple[list[AudioPoint], list[dict]]:
    points: list[AudioPoint] = []
    skipped: list[dict] = []
    audio_dir = work / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    seconds = config["audio_seconds"]
    samples = int(seconds * 48000)

    # FLAC: the example generates deterministic PCM itself and writes the
    # exact reference bytes to out.flac.raw.  libFLAC encodes the same bytes.
    flac_encode = project_binary(runner, "flac_encode")
    flac_decode = project_binary(runner, "flac_decode")
    flac_tool = runner.which("flac")
    rust_flac = audio_dir / "rustmpeg.flac"
    raw_path = Path(str(rust_flac) + ".raw")
    enc, _ = runner.run(
        [flac_encode, rust_flac, "16", "2", str(samples), "4096", "5"],
        "rustmpeg FLAC encode",
    )
    if not rust_flac.is_file() or not raw_path.is_file():
        raise BenchError("rustmpeg FLAC example did not produce its reference PCM")
    raw_bytes = raw_path.stat().st_size
    rust_s32 = audio_dir / "rustmpeg-decoded.s32le"
    dec, _ = runner.run(
        [flac_decode, rust_flac, rust_s32], "rustmpeg FLAC decode"
    )
    source_samples = read_s16le(raw_path)
    decoded_s32 = read_s32le(rust_s32)
    if len(source_samples) != len(decoded_s32):
        raise BenchError("rustmpeg FLAC decode produced a different sample count")
    for index, (source, decoded) in enumerate(zip(source_samples, decoded_s32)):
        if decoded != source << 16 and decoded >> 16 != source:
            raise BenchError(f"rustmpeg FLAC is not bit-exact at sample {index}")
    rust_ratio = rust_flac.stat().st_size / raw_bytes

    if flac_tool is None:
        if not runner.allow_missing:
            raise BenchError("FLAC vs libFLAC: flac command not found")
        skipped.append(
            {"item": "FLAC vs libFLAC", "reason": "flac command not found"}
        )
    else:
        ref_flac = audio_dir / "libflac.flac"
        ref_enc, _ = runner.run(
            [
                flac_tool, "-f", "--silent", "--force-raw-format",
                "--endian=little", "--sign=signed", "--channels=2",
                "--bps=16", "--sample-rate=48000", "-o", ref_flac, raw_path,
            ],
            "libFLAC encode",
        )
        ref_raw = audio_dir / "libflac-decoded.s16le"
        ref_dec, _ = runner.run(
            [
                flac_tool, "-d", "-f", "--silent", "--force-raw-format",
                "--endian=little", "--sign=signed", "-o", ref_raw, ref_flac,
            ],
            "libFLAC decode",
        )
        if ref_raw.read_bytes() != raw_path.read_bytes():
            raise BenchError("libFLAC round-trip is not bit-exact")
        ref_ratio = ref_flac.stat().st_size / raw_bytes
        points.append(
            AudioPoint(
                codec="FLAC", implementation="rustmpeg-audio", reference="libFLAC",
                param="16-bit stereo 48 kHz block 4096 level 5",
                bytes=rust_flac.stat().st_size, ratio=rust_ratio, snr_db=None,
                lossless=True, enc_seconds=enc, dec_seconds=dec,
                enc_mbps=raw_bytes / enc / 1e6, dec_mbps=raw_bytes / dec / 1e6,
                quality="lossless, bit-exact",
            )
        )
        points.append(
            AudioPoint(
                codec="FLAC", implementation="libFLAC", reference="libFLAC",
                param="16-bit stereo 48 kHz default level 5",
                bytes=ref_flac.stat().st_size, ratio=ref_ratio, snr_db=None,
                lossless=True, enc_seconds=ref_enc, dec_seconds=ref_dec,
                enc_mbps=raw_bytes / ref_enc / 1e6,
                dec_mbps=raw_bytes / ref_dec / 1e6,
                quality="lossless, bit-exact",
            )
        )

    # LC3: rustmpeg and liblc3 encode/decode the same 48 kHz s16le PCM.
    lc3_encode = project_binary(runner, "lc3_encode")
    lc3_decode = project_binary(runner, "lc3_decode")
    elc3 = runner.which("elc3")
    dlc3 = runner.which("dlc3")
    pcm_wav = audio_dir / "source.wav"
    pcm_raw = audio_dir / "source.s16le"
    runner.run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency=997:sample_rate=48000:duration={seconds}",
            "-f", "lavfi",
            "-i", f"anoisesrc=color=pink:sample_rate=48000:duration={seconds}:seed=1234:amplitude=0.05",
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=shortest:normalize=0,volume=0.8",
            "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", pcm_wav,
        ],
        "generate LC3 PCM",
    )
    runner.run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", pcm_wav, "-f", "s16le", "-acodec", "pcm_s16le", pcm_raw,
        ],
        "extract LC3 raw PCM",
    )
    reference_pcm = read_s16le(pcm_raw)
    pcm_bytes = len(reference_pcm) * 2

    if elc3 is None or dlc3 is None:
        if not runner.allow_missing:
            raise BenchError("LC3 vs liblc3: elc3/dlc3 not found")
        skipped.append(
            {"item": "LC3 vs liblc3", "reason": "elc3/dlc3 not found"}
        )
    else:
        for kbps in config["lc3_kbps"]:
            nbytes = round(kbps * 1000 * 0.010 / 8)
            rust_lc3 = audio_dir / f"rustmpeg-{kbps}k.lc3"
            rust_pcm = audio_dir / f"rustmpeg-{kbps}k.s16le"
            enc, _ = runner.run(
                [
                    lc3_encode, pcm_raw, rust_lc3, "48000", "10000", "1",
                    str(nbytes),
                ],
                f"rustmpeg LC3 encode {kbps} kbps",
            )
            dec, _ = runner.run(
                [lc3_decode, rust_lc3, rust_pcm],
                f"rustmpeg LC3 decode {kbps} kbps",
            )
            ref_lc3 = audio_dir / f"liblc3-{kbps}k.lc3"
            ref_wav = audio_dir / f"liblc3-{kbps}k.wav"
            ref_pcm = audio_dir / f"liblc3-{kbps}k.s16le"
            ref_enc, _ = runner.run(
                [elc3, "-b", str(kbps * 1000), "-m", "10", pcm_wav, ref_lc3],
                f"liblc3 encode {kbps} kbps",
            )
            ref_dec, _ = runner.run(
                [dlc3, ref_lc3, ref_wav],
                f"liblc3 decode {kbps} kbps",
            )
            runner.run(
                [
                    ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-y", "-i", ref_wav, "-f", "s16le", "-acodec",
                    "pcm_s16le", ref_pcm,
                ],
                f"convert liblc3 PCM {kbps} kbps",
            )
            rust_samples = read_s16le(rust_pcm)
            ref_samples = read_s16le(ref_pcm)
            rust_snr, rust_lag = aligned_snr(reference_pcm, rust_samples)
            ref_snr, ref_lag = aligned_snr(reference_pcm, ref_samples)
            rust_size = rust_lc3.stat().st_size
            ref_size = ref_lc3.stat().st_size
            points.append(
                AudioPoint(
                    codec="LC3", implementation="rustmpeg-audio",
                    reference="liblc3", param=f"{kbps} kbps, 10 ms, mono",
                    bytes=rust_size, ratio=rust_size / pcm_bytes,
                    snr_db=rust_snr, lossless=False, enc_seconds=enc,
                    dec_seconds=dec, enc_mbps=pcm_bytes / enc / 1e6,
                    dec_mbps=pcm_bytes / dec / 1e6,
                    quality=(
                        f"SNR {rust_snr:.2f} dB (lag {rust_lag:+d})"
                        if rust_snr is not None else "bit-exact"
                    ),
                    lag=rust_lag,
                )
            )
            points.append(
                AudioPoint(
                    codec="LC3", implementation="liblc3", reference="liblc3",
                    param=f"{kbps} kbps, 10 ms, mono",
                    bytes=ref_size, ratio=ref_size / pcm_bytes,
                    snr_db=ref_snr, lossless=False, enc_seconds=ref_enc,
                    dec_seconds=ref_dec, enc_mbps=pcm_bytes / ref_enc / 1e6,
                    dec_mbps=pcm_bytes / ref_dec / 1e6,
                    quality=(
                        f"SNR {ref_snr:.2f} dB (lag {ref_lag:+d})"
                        if ref_snr is not None else "bit-exact"
                    ),
                    lag=ref_lag,
                )
            )
    return points, skipped


def machine_info() -> dict:
    cpu = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        memory_gib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1 << 30)
    except (ValueError, OSError):
        memory_gib = 0.0
    logical = os.cpu_count() or 0
    return {
        "hostname": platform.node() or "unknown",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "logical_cpus": logical,
        "memory_gib": round(memory_gib, 1),
    }


def git_info() -> dict:
    def run(*cmd: str) -> str:
        try:
            proc = subprocess.run(
                ["git", "-C", str(ROOT), *cmd],
                capture_output=True, text=True, check=False,
            )
            return proc.stdout.strip() if proc.returncode == 0 else ""
        except OSError:
            return ""

    commit = run("rev-parse", "HEAD") or "no-commit"
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status)}


def oracle_versions(oracle_dir: Path | None) -> dict:
    if oracle_dir is not None:
        manifest = oracle_dir / "oracles.json"
        if manifest.is_file():
            try:
                return json.loads(manifest.read_text())
            except (OSError, json.JSONDecodeError):
                pass
    result = {}
    for tool, flag in [
        ("SvtAv1EncApp", "--version"),
        ("avmenc", "--help"),
        ("dav2d", "--version"),
        ("ffmpeg", "-version"),
        ("flac", "--version"),
    ]:
        found = shutil.which(tool)
        if found:
            proc = subprocess.run(
                [found, flag], capture_output=True, text=True, check=False
            )
            first = (proc.stdout or proc.stderr).splitlines()
            result[tool] = first[0] if first else found
    result.setdefault("svt_av1_pin", result.get("SvtAv1EncApp", "not installed"))
    result.setdefault("avm_pin", result.get("avmenc", "not installed"))
    result.setdefault("dav2d_commit", result.get("dav2d", "not installed"))
    result.setdefault("liblc3_pin", "system elc3/dlc3" if shutil.which("elc3") else "not installed")
    result.setdefault("flac", result.get("flac", "not installed"))
    return result


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def render_video_tables(points: list[Point]) -> list[str]:
    lines: list[str] = []
    clips = sorted({point.clip for point in points})
    for clip in clips:
        lines.append(f"#### Clip `{clip}`")
        lines.append("")
        lines.append(
            "| Codec | Implementation | Decoder | Point | Bitrate (kbps) | "
            "PSNR-Y (dB) | SSIM | Encode MB/s | Decode MB/s | Bytes |"
        )
        lines.append("| :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: |")
        for codec in ("rustmpeg-av2", "avm-av2", "svt-av1"):
            for point in sorted(
                (item for item in points if item.clip == clip and item.codec == codec),
                key=lambda item: item.psnr_y,
            ):
                info = CODEC_INFO[codec]
                label = info["codec"]
                if info["cross_codec"]:
                    label += " (cross-codec)"
                lines.append(
                    f"| {label} | {info['implementation']} | {point.decoder} | "
                    f"`{point.param}` | {fmt(point.bitrate_kbps, 1)} | "
                    f"{fmt(point.psnr_y)} | {fmt(point.ssim_all, 4)} | "
                    f"{fmt(point.enc_mbps, 1)} | {fmt(point.dec_mbps, 1)} | "
                    f"{point.bytes} |"
                )
        lines.append("")
    return lines


def render_bd_rows(rows: list[dict]) -> list[str]:
    lines = [
        "#### BD-rate",
        "",
        "| Anchor | Test | Clip | BD-rate | Note |",
        "| :-- | :-- | :-- | --: | :-- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['anchor']} | {row['test']} | {row['clip']} | "
            f"{pct(row['bd_rate'])} | {row['note']} |"
        )
    lines.append("")
    return lines


def render_audio_table(points: list[AudioPoint]) -> list[str]:
    lines = [
        "#### Audio",
        "",
        "| Codec | Implementation | Reference | Point | Bytes | Ratio | "
        "SNR (dB) | Encode MB/s | Decode MB/s | Quality |",
        "| :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: | :-- |",
    ]
    for point in points:
        snr = "lossless" if point.snr_db is None and point.lossless else fmt(point.snr_db)
        lines.append(
            f"| {point.codec} | {point.implementation} | {point.reference} | "
            f"`{point.param}` | {point.bytes} | {fmt(point.ratio, 4)} | "
            f"{snr} | {fmt(point.enc_mbps, 1)} | {fmt(point.dec_mbps, 1)} | "
            f"{point.quality} |"
        )
    lines.append("")
    if any(point.codec == "FLAC" and point.implementation == "rustmpeg-audio" for point in points):
        lines.append(
            "The rustmpeg FLAC encode time includes the example's deterministic "
            "in-process PCM synthesis, so compare decode throughput, sizes, and "
            "bit-exactness rather than its encode MB/s."
        )
        lines.append("")
    return lines


def render_skips(skipped: list[dict]) -> list[str]:
    system_skips = [
        {
            "item": "Matroska / ISOBMFF (format)",
            "reason": "No runnable format benchmark binary exists and this "
            "pipeline does not modify crates; compare via the format tests instead.",
        },
        {
            "item": "Image codecs (JPEG XL, OpenEXR)",
            "reason": "JPEG XL benchmark row is not wired into this workflow yet.",
        },
        {
            "item": "Opus",
            "reason": "Only packet parsing and the range decoder are implemented; "
            "no encoder/decoder is available to compare.",
        },
    ]
    lines = [
        "#### Not benchmarked in this run",
        "",
        "| Item | Reason |",
        "| :-- | :-- |",
    ]
    for item in skipped + system_skips:
        lines.append(f"| {item['item']} | {item['reason']} |")
    lines.append("")
    return lines


def render_commands(commands: list[str]) -> list[str]:
    lines = [
        "<details>",
        f"<summary>Exact commands ({len(commands)})</summary>",
        "",
        "```sh",
        *commands,
        "```",
        "",
        "</details>",
        "",
    ]
    return lines


def render_block(
    runner: Runner, config: dict, args: argparse.Namespace, video: list[Point],
    audio: list[AudioPoint], bd_rows: list[dict], skipped: list[dict],
    sources: dict[str, Path], started_utc: str,
) -> str:
    git = git_info()
    machine = machine_info()
    oracles = oracle_versions(args.oracle_dir)
    lines = [
        TABLE_START,
        f"### {started_utc} - `{args.profile}` profile",
        "",
        f"- Commit: `{git['commit']}`"
        + (" (working tree dirty)" if git["dirty"] else " (clean)"),
        f"- Content: {config['width']}x{config['height']}, {config['frames']} frames, "
        f"{config['fps']} fps, I420 8-bit; AV2 QP {', '.join(map(str, config['av2_qp']))}; "
        f"AV1 CRF {', '.join(map(str, config['av1_crf']))}",
        f"- Runner: {machine['hostname']} / {machine['platform']} / "
        f"{machine['logical_cpus']} logical CPUs / {machine['memory_gib']} GiB RAM",
        f"- CPU: {machine['cpu']}",
        f"- Rust: `{rust_version()}`",
        f"- FFmpeg: `{oracles.get('ffmpeg', 'unknown')}`",
        f"- Oracles: SVT-AV1 `{oracles.get('svt_av1_pin', 'unknown')}`, "
        f"AVM `{oracles.get('avm_pin', 'unknown')}`, "
        f"dav2d `{str(oracles.get('dav2d_commit', 'unknown'))[:8]}`, "
        f"liblc3 `{oracles.get('liblc3_pin', 'unknown')}`, "
        f"flac `{oracles.get('flac', 'unknown')}`",
        "- Sources: "
        + ", ".join(
            f"`{clip}` sha256 `{sha256_file(path)[:16]}...`"
            for clip, path in sorted(sources.items())
        ),
        "",
        "PSNR and SSIM are measured by FFmpeg against the exact source Y4M. "
        "Bitrate is `(container bytes * 8 * fps) / frame_count`; it includes the "
        "IVF/OBU container header. MB/s uses raw I420 bytes and wall-clock time "
        "for the encode or decode process only.",
        "",
    ]
    if video:
        lines.extend(render_video_tables(video))
        lines.extend(render_bd_rows(bd_rows))
    if audio:
        lines.extend(render_audio_table(audio))
    lines.extend(render_skips(skipped))
    lines.extend(render_commands(runner.commands))
    lines.append(TABLE_END)
    return "\n".join(lines)


def rust_version() -> str:
    try:
        proc = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def replace_section(text: str, start: str, end: str, content: str) -> str:
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end), re.DOTALL
    )
    if not pattern.search(text):
        raise BenchError(f"markers {start} / {end} not found in BENCHMARK.md")
    return pattern.sub(lambda _: content, text, count=1)


def update_document(
    path: Path, block: str, no_history: bool = False,
) -> None:
    if path.is_file():
        text = path.read_text()
    else:
        text = (
            "# Benchmarks\n\n"
            "This file is generated by `scripts/bench/bench.py`. Do not edit the "
            "marked sections by hand.\n\n"
            "## Latest run\n\n"
            f"{TABLE_START}\n_No benchmark run recorded yet._\n{TABLE_END}\n\n"
            "## History\n\n"
            f"{HISTORY_START}\n_No prior runs._\n{HISTORY_END}\n"
        )
    current_match = re.search(
        re.escape(TABLE_START) + r"(.*?)" + re.escape(TABLE_END), text, re.DOTALL
    )
    if current_match is None:
        raise BenchError("BENCHMARK.md is missing the latest-run markers")
    current = current_match.group(0)
    placeholder = "No benchmark run recorded yet" in current

    history_match = re.search(
        re.escape(HISTORY_START) + r"(.*?)" + re.escape(HISTORY_END),
        text,
        re.DOTALL,
    )
    if history_match is None:
        raise BenchError("BENCHMARK.md is missing the history markers")
    old_history = history_match.group(1).strip()
    if no_history or placeholder:
        history = old_history if old_history else "_No prior runs._"
    else:
        commit = re.search(r"Commit: `([^`]+)`", block)
        old_commit = re.search(r"Commit: `([^`]+)`", current)
        if commit and old_commit and commit.group(1) == old_commit.group(1):
            # Re-running the same commit replaces the current entry instead of
            # duplicating it; older entries are still preserved below.
            history = old_history
        else:
            summary = re.search(r"### (.*)", current)
            title = summary.group(1) if summary else "previous run"
            inner = current
            if inner.startswith(TABLE_START):
                inner = inner[len(TABLE_START):]
            if inner.endswith(TABLE_END):
                inner = inner[: -len(TABLE_END)]
            entry = (
                f"{HISTORY_ENTRY}\n<details>\n<summary>{title}</summary>\n\n"
                f"{inner.strip()}\n\n</details>"
            )
            if old_history in ("", "_No prior runs."):
                history = entry
            else:
                history = entry + "\n\n" + old_history
            entries = history.count(HISTORY_ENTRY)
            if entries > MAX_HISTORY:
                # Keep the newest marker-delimited entries and state the trim.
                chunks = history.split(HISTORY_ENTRY)
                history = "".join(
                    HISTORY_ENTRY + chunk
                    for chunk in chunks[1 : MAX_HISTORY + 1]
                )
                history += (
                    f"\n\n_Older entries trimmed at {MAX_HISTORY}; the workflow "
                    "artifact keeps the full JSON for every run._"
                )

    text = replace_section(text, TABLE_START, TABLE_END, block)
    text = replace_section(
        text,
        HISTORY_START,
        HISTORY_END,
        f"{HISTORY_START}\n{history}\n{HISTORY_END}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--width", type=int, help="override profile width")
    parser.add_argument("--height", type=int, help="override profile height")
    parser.add_argument("--frames", type=int, help="override profile frame count")
    parser.add_argument("--fps", type=int, help="override profile frame rate")
    parser.add_argument("--av2-qp", help="comma-separated AV2 QP sweep")
    parser.add_argument("--av1-crf", help="comma-separated AV1 CRF sweep")
    parser.add_argument("--lc3-kbps", help="comma-separated LC3 bitrate sweep")
    parser.add_argument("--audio-seconds", type=float)
    parser.add_argument(
        "--clips", default=",".join(VIDEO_CLIPS),
        help="comma-separated clip names (graphics,noise)",
    )
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        help="directory produced by fetch_oracles.sh (contains bin/ and oracles.json)",
    )
    parser.add_argument(
        "--target-dir", type=Path, default=ROOT / "target",
        help="cargo target directory (default: ./target)",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs/benchmarks/BENCHMARK.md",
        help="benchmark document to update",
    )
    parser.add_argument(
        "--json", type=Path, help="write the complete structured result here",
    )
    parser.add_argument(
        "--workdir", type=Path, help="scratch directory (default: a temp dir)",
    )
    parser.add_argument(
        "--allow-missing-oracles", action="store_true",
        help="skip comparisons whose oracle is not installed; never invent rows",
    )
    parser.add_argument(
        "--command-timeout", type=float, default=1800.0,
        help="kill a single encode/decode/measure command after this many "
        "seconds (0 disables the per-command limit; default: 1800)",
    )
    parser.add_argument(
        "--time-budget", type=float, default=0.0,
        help="stop starting new work after this many seconds and record the "
        "remaining comparisons as skipped (0 disables the overall budget)",
    )
    parser.add_argument(
        "--no-history", action="store_true",
        help="replace the latest block without appending it to history",
    )
    parser.add_argument(
        "--print", action="store_true",
        help="print the rendered block instead of writing the document",
    )
    parser.add_argument("--keep", action="store_true", help="keep the scratch dir")
    args = parser.parse_args()

    config = dict(PROFILES[args.profile])
    for name in ("width", "height", "frames", "fps", "audio_seconds"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.av2_qp:
        config["av2_qp"] = [int(value) for value in args.av2_qp.split(",") if value]
    if args.av1_crf:
        config["av1_crf"] = [int(value) for value in args.av1_crf.split(",") if value]
    if args.lc3_kbps:
        config["lc3_kbps"] = [int(value) for value in args.lc3_kbps.split(",") if value]
    args.clips = [value.strip() for value in args.clips.split(",") if value.strip()]
    for clip in args.clips:
        if clip not in VIDEO_CLIPS:
            parser.error(f"unknown clip '{clip}'")

    if args.workdir:
        work = args.workdir.resolve()
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix="rustmpeg-bench-"))
        cleanup = not args.keep

    env = dict(os.environ)
    runner = Runner(
        oracle_dir=args.oracle_dir.resolve() if args.oracle_dir else None,
        target_dir=args.target_dir.resolve(),
        env=env,
        allow_missing=args.allow_missing_oracles,
        keep=args.keep,
        command_timeout=args.command_timeout or None,
        time_budget=args.time_budget or None,
        deadline=(
            time.monotonic() + args.time_budget if args.time_budget else None
        ),
    )
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    started_at = time.perf_counter()

    try:
        ffmpeg = runner.which("ffmpeg")
        if ffmpeg is None:
            raise BenchError("ffmpeg is required to generate clips and measure quality")
        print(f"benchmark profile: {args.profile} ({config['width']}x{config['height']}, "
              f"{config['frames']} frames)", file=sys.stderr)

        video, video_skips, sources = run_video(runner, ffmpeg, work, config, args)
        try:
            audio, audio_skips = run_audio(runner, ffmpeg, work, config, args)
        except BenchBudgetExhausted as error:
            audio, audio_skips = [], [
                {"item": "Audio (FLAC, LC3)", "reason": skip_reason(error)}
            ]
        if not video and not audio:
            raise BenchError("no benchmark produced a complete result")
        bd_rows = bd_rate_rows(video)
        block = render_block(
            runner, config, args, video, audio, bd_rows,
            video_skips + audio_skips, sources, started,
        )

        elapsed = time.perf_counter() - started_at
        print(f"completed in {elapsed:.1f}s; {len(video)} video points, "
              f"{len(audio)} audio points", file=sys.stderr)

        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(
                    {
                        "generated_at": started,
                        "profile": args.profile,
                        "config": config,
                        "git": git_info(),
                        "machine": machine_info(),
                        "oracles": oracle_versions(args.oracle_dir),
                        "video": [point.__dict__ for point in video],
                        "bd_rate": bd_rows,
                        "audio": [point.__dict__ for point in audio],
                        "skipped": video_skips + audio_skips,
                        "commands": runner.commands,
                    },
                    indent=2,
                )
                + "\n"
            )
        if args.print:
            print(block)
        else:
            update_document(args.out, block, no_history=args.no_history)
            print(f"updated {args.out}", file=sys.stderr)
        return 0
    except BenchError as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        print("no results were written", file=sys.stderr)
        return 1
    finally:
        if cleanup and not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
