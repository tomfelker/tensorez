"""[align] per_channel: atmospheric dispersion correction tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import CliRun, parse_events, run_cli
from tensorez import ser
from tensorez.align import center_of_mass
from tensorez.recipe import RecipeError, load_recipe


def _dispersed_ser(path: Path, offsets=((3, 2), (0, 0), (-2, -3)), n=8) -> None:
    """RGB SER of a blob whose channels carry constant artificial offsets
    (fake dispersion), plus per-frame global jitter."""
    rng = np.random.default_rng(11)
    h = w = 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frames = []
    for i in range(n):
        gy, gx = int(rng.integers(-4, 5)), int(rng.integers(-4, 5))  # global jitter
        chans = []
        for (cy, cx) in offsets:
            blob = 0.8 * np.exp(
                (-(yy - h / 2 - cy - gy) ** 2 - (xx - w / 2 - cx - gx) ** 2) / 30.0
            )
            chans.append(blob)
        frame = np.stack(chans, axis=-1) + rng.normal(0, 0.005, (h, w, 3))
        frames.append(np.clip(frame, 0, 1))
    ser.write_ser(path, (np.stack(frames) * 65535).astype(np.uint16), ser.ColorId.RGB)


def _channel_centroids(image_hwc: np.ndarray) -> np.ndarray:
    """(C, 2) centroid (y, x) of each channel, via the pipeline's own CoM."""
    t = torch.from_numpy(image_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    return np.array([center_of_mass(t[:, c : c + 1]) for c in range(t.shape[1])])


def _recipe(tmp_path: Path, ser_path: Path, per_channel: bool) -> Path:
    p = tmp_path / f"r_{per_channel}.toml"
    p.write_text(f"""
[lights]
paths = ["{ser_path.as_posix()}"]
[align]
per_channel = {str(per_channel).lower()}
[local_lucky]
crossover_wavelength_pixels = 8.0
isoplanatic_patch_pixels = 16.0
[output]
dir = "{(tmp_path / 'out').as_posix()}"
debug_frames = 0
""")
    return p


def test_per_channel_removes_dispersion(tmp_path: Path) -> None:
    ser_path = tmp_path / "dispersed.ser"
    _dispersed_ser(ser_path)

    runs = {}
    for per_channel in (False, True):
        proc = run_cli(["run", str(_recipe(tmp_path, ser_path, per_channel))], cwd=tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        runs[per_channel] = CliRun(proc, parse_events(proc.stdout), 0.0)

    def spread(run: CliRun) -> float:
        avg = np.load(run.run_dir / "examples/local_lucky/unweighted_average.npy")
        cent = _channel_centroids(avg)
        return float(np.linalg.norm(cent - cent.mean(axis=0), axis=1).max())

    # Whole-frame alignment keeps the injected ~(3,2)/(-2,-3) channel offsets...
    assert spread(runs[False]) > 2.0, spread(runs[False])
    # ...per-channel alignment removes them to within a pixel.
    assert spread(runs[True]) <= 1.0, spread(runs[True])

    # the dispersion signature is logged
    logs = [e["message"] for e in runs[True].events_of("log")]
    assert any("mean per-channel CoM offset" in m for m in logs)


def test_per_channel_recipe_conflicts(tmp_path: Path) -> None:
    base = """
[lights]
paths = ["x.ser"]
"""
    p = tmp_path / "r.toml"
    p.write_text(base + "[align]\nper_channel = true\ncenter_of_mass = false\n")
    with pytest.raises(RecipeError, match="per_channel requires center_of_mass"):
        load_recipe(p)
    # only_even_shifts is gone (debayering happens on read now); like any
    # removed key it must be rejected as unknown, not silently ignored
    p.write_text(base + "[align]\nonly_even_shifts = true\n")
    with pytest.raises(RecipeError, match="unknown key 'only_even_shifts'"):
        load_recipe(p)


def test_per_channel_mono_is_noop(tmp_path: Path) -> None:
    frames = (np.random.default_rng(5).random((6, 32, 32, 1)) * 65535).astype(np.uint16)
    ser_path = tmp_path / "mono.ser"
    ser.write_ser(ser_path, frames, ser.ColorId.MONO)
    recipe = tmp_path / "r.toml"
    recipe.write_text(f"""
[lights]
paths = ["{ser_path.as_posix()}"]
[align]
per_channel = true
[local_lucky]
crossover_wavelength_pixels = 6.0
isoplanatic_patch_pixels = 10.0
[output]
dir = "{(tmp_path / 'out').as_posix()}"
debug_frames = 0
""")
    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    events = parse_events(proc.stdout)
    assert events[-1]["event"] == "done"
    logs = [e["message"] for e in events if e["event"] == "log"]
    assert any("mono" in m and "no-op" in m for m in logs)
