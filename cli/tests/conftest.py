"""Shared fixtures: run the CLI as a subprocess on the synthetic data once,
then let several tests inspect the results (events, outputs, caching)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

CLI_ROOT = Path(__file__).resolve().parents[1]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))
EXAMPLES = CLI_ROOT.parent / "examples"
SYNTHETIC_SER = EXAMPLES / "synthetic_planet.ser"
TRUTH_NPY = EXAMPLES / "truth.npy"

# synthetic_planet.ser is ~24 MB and deliberately not in git; generate it once.
if not SYNTHETIC_SER.exists():
    subprocess.run([sys.executable, str(EXAMPLES / "gen_synthetic.py")], check=True)


# -- video fixtures ---------------------------------------------------------
# Video support is optional (see tensorez/video.py): it needs torchcodec plus a
# *shared* FFmpeg build.  The video tests skip with the real reason rather than
# fail when either is missing, since that is a supported configuration.

from tensorez import video  # noqa: E402  (needs CLI_ROOT on sys.path first)


def ffmpeg_exe() -> str | None:
    """The ffmpeg binary, used only to generate test fixtures."""
    directory = video.ffmpeg_dir()
    if directory is not None:
        for name in ("ffmpeg.exe", "ffmpeg"):
            candidate = directory / name
            if candidate.exists():
                return str(candidate)
    return shutil.which("ffmpeg")


def _video_support() -> tuple[bool, str]:
    if ffmpeg_exe() is None:
        return False, "no ffmpeg binary available to generate test videos"
    try:
        video._video_decoder_class()
    except video.VideoSupportError as e:
        return False, str(e).splitlines()[0]
    return True, ""


VIDEO_OK, VIDEO_SKIP_REASON = _video_support()
requires_video = pytest.mark.skipif(not VIDEO_OK, reason=VIDEO_SKIP_REASON)


def marker_frames(count: int = 24, height: int = 32) -> np.ndarray:
    """(N, H, W, 3) uint8 frames that identify themselves two ways.

    Frame i has a flat background of ``i * 8`` and a white bar at column
    ``2 * i``, so a decoder handing back a neighbouring frame -- or the
    preceding keyframe, which is the classic seeking bug -- is caught by
    either the level or the bar position.
    """
    width = 2 * count + 4
    frames = np.zeros((count, height, width, 3), np.uint8)
    for i in range(count):
        frames[i, :, :, :] = i * 8
        frames[i, :, 2 * i : 2 * i + 2, :] = 255
    return frames


def write_video(path: Path, frames: np.ndarray, codec: str = "rawvideo",
                pix_fmt: str = "rgb24", fps: int = 25) -> Path:
    """Encode (N, H, W, 3) uint8 frames with ffmpeg.

    ``rawvideo``/``rgb24`` is lossless and all-intra, so tests can assert exact
    pixels; ``mpeg4`` exercises the inter-coded path where a frame index has to
    be resolved by decoding forward from a keyframe.
    """
    _, height, width, _ = frames.shape
    exe = ffmpeg_exe()
    assert exe is not None, "write_video requires ffmpeg"
    quality = [] if codec == "rawvideo" else ["-qscale:v", "2"]
    subprocess.run(
        [exe, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", codec, *quality, "-pix_fmt", pix_fmt, str(path)],
        input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes(),
        check=True, capture_output=True,
    )
    return path


def write_video_10bit(path: Path, levels: list[int], size: int = 8, fps: int = 5) -> Path:
    """Write a 10-bit video whose frame i is a flat field of ``levels[i]``.

    Fed as gbrp10le -- planar G, B, R in 16-bit containers -- so the exact
    10-bit levels reach the encoder without an intermediate promotion, and
    lossless ffv1 brings them back.  ffv1 is a native LGPL encoder, so this
    works with the plain shared FFmpeg build (x264/x265 would not).
    """
    exe = ffmpeg_exe()
    assert exe is not None, "write_video_10bit requires ffmpeg"
    planes = np.stack([np.full((3, size, size), v, np.uint16) for v in levels])
    subprocess.run(
        [exe, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "gbrp10le",
         "-s", f"{size}x{size}", "-r", str(fps), "-i", "-",
         "-c:v", "ffv1", "-pix_fmt", "gbrp10le", str(path)],
        input=planes.tobytes(), check=True, capture_output=True,
    )
    return path


def run_cli(args: list[str], cwd: Path, events: bool = True) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI.  Tests parse the JSONL event stream, so --events is
    added unless a test is specifically checking the human-readable default."""
    if events and args and args[0] in ("run", "validate") and "--events" not in args:
        args = [*args, "--events"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CLI_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "tensorez", *args],
        cwd=cwd, env=env, capture_output=True, text=True,
    )


def parse_events(stdout: str) -> list[dict]:
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


@dataclass
class CliRun:
    proc: subprocess.CompletedProcess[str]
    events: list[dict]
    seconds: float

    @property
    def run_dir(self) -> Path:
        return Path(self.events[0]["run_dir"])

    def events_of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["event"] == kind]


def write_recipe(path: Path, out_dir: Path, stdevs_above_mean: float = 2.0) -> Path:
    path.write_text(f"""
[lights]
paths = ["{SYNTHETIC_SER.as_posix()}"]

[align]
center_of_mass = true
crop = [192, 192]

[local_lucky]
crossover_wavelength_pixels = 30.0
isoplanatic_patch_pixels = 50.0
stdevs_above_mean = {stdevs_above_mean}
steepness = 3.0

[output]
dir = "{out_dir.as_posix()}"
debug_frames = 3
""")
    return path


@dataclass
class SyntheticRuns:
    workdir: Path
    recipe: Path
    first: CliRun
    second: CliRun
    reselect: CliRun  # same recipe except stdevs_above_mean changed
    reselect_validate: CliRun


def _timed_run(recipe: Path, workdir: Path) -> CliRun:
    t0 = time.monotonic()
    proc = run_cli(["run", str(recipe)], cwd=workdir)
    seconds = time.monotonic() - t0
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return CliRun(proc, parse_events(proc.stdout), seconds)


@pytest.fixture(scope="session")
def synthetic_runs(tmp_path_factory: pytest.TempPathFactory) -> SyntheticRuns:
    workdir = tmp_path_factory.mktemp("synthetic")
    out_dir = workdir / "output"
    recipe = write_recipe(workdir / "recipe.toml", out_dir)

    first = _timed_run(recipe, workdir)
    second = _timed_run(recipe, workdir)

    recipe2 = write_recipe(workdir / "recipe2.toml", out_dir, stdevs_above_mean=1.5)
    proc = run_cli(["validate", str(recipe2)], cwd=workdir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    reselect_validate = CliRun(proc, parse_events(proc.stdout), 0.0)
    reselect = _timed_run(recipe2, workdir)

    return SyntheticRuns(workdir, recipe, first, second, reselect, reselect_validate)
