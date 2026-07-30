"""Dark calibration: master dark subtraction, variance, and caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import CliRun, parse_events, run_cli
from tensorez import ser


def _make_data(tmp_path: Path) -> tuple[Path, Path]:
    """Tiny MONO sequence: a bright blob wandering over a hot background."""
    rng = np.random.default_rng(7)
    h = w = 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    offset = 0.10  # fixed-pattern "thermal" signal the darks must remove
    lights = []
    for i in range(8):
        cy, cx = h / 2 + (i % 3) - 1, w / 2 + (i % 2)
        blob = 0.6 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 12.0)
        frame = blob + offset + rng.normal(0, 0.01, (h, w)).astype(np.float32)
        lights.append(np.clip(frame, 0, 1))
    lights_u16 = (np.stack(lights)[..., None] * 65535).astype(np.uint16)

    darks = offset + rng.normal(0, 0.01, (16, h, w, 1)).astype(np.float32)
    darks_u16 = (np.clip(darks, 0, 1) * 65535).astype(np.uint16)

    lights_path = tmp_path / "lights.ser"
    darks_path = tmp_path / "darks.ser"
    ser.write_ser(lights_path, lights_u16, ser.ColorId.MONO)
    ser.write_ser(darks_path, darks_u16, ser.ColorId.MONO)
    return lights_path, darks_path


def test_darks_subtracted_and_cached(tmp_path: Path) -> None:
    lights_path, darks_path = _make_data(tmp_path)
    recipe = tmp_path / "r.toml"
    recipe.write_text(f"""
[recipe]
version = 0
name = "darks"
[lights]
paths = ["{lights_path.as_posix()}"]
[darks]
paths = ["{darks_path.as_posix()}"]
[local_lucky]
noise_wavelength_pixels = 2.0
crossover_wavelength_pixels = 6.0
isoplanatic_patch_pixels = 10.0
[output]
dir = "{(tmp_path / 'out').as_posix()}"
debug_frames = 0
""")

    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    first = CliRun(proc, parse_events(proc.stdout), 0.0)
    stages = [e["stage"] for e in first.events_of("stage_start")]
    assert stages == ["lights", "darks", "align", "local_lucky", "output"]
    darks_start = [e for e in first.events_of("stage_start") if e["stage"] == "darks"][0]
    assert darks_start["cached"] is False

    # the hot background is gone: corner pixels near zero, blob preserved
    final = np.load(first.run_dir / "final.npy")
    assert abs(float(final[:4, :4].mean())) < 0.02
    assert float(final.max()) > 0.4

    # second run serves the master dark from cache with immediate stage_end
    proc2 = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    second = CliRun(proc2, parse_events(proc2.stdout), 0.0)
    darks_start2 = [e for e in second.events_of("stage_start") if e["stage"] == "darks"][0]
    assert darks_start2["cached"] is True
    assert not [e for e in second.events_of("progress") if e["stage"] == "darks"]
