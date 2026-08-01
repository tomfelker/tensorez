"""Dark calibration: master dark subtraction, variance, and caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import CliRun, parse_events, run_cli
from tensorez import ser


def _make_data(tmp_path: Path) -> tuple[Path, Path]:
    """Tiny MONO sequence: a bright blob wandering over a hot background.

    The darks file starts with 4 saturated junk frames (as if the target were
    still in view); the recipe below must skip them via start_frame or the
    master dark is ruined — which also proves [darks] frame selection works.
    """
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

    junk = np.ones((4, h, w, 1), dtype=np.float32)
    darks = offset + rng.normal(0, 0.01, (16, h, w, 1)).astype(np.float32)
    darks_u16 = (np.clip(np.concatenate([junk, darks]), 0, 1) * 65535).astype(np.uint16)

    lights_path = tmp_path / "lights.ser"
    darks_path = tmp_path / "darks.ser"
    ser.write_ser(lights_path, lights_u16, ser.ColorId.MONO)
    ser.write_ser(darks_path, darks_u16, ser.ColorId.MONO)
    return lights_path, darks_path


def _write_recipe(tmp_path: Path, lights_path: Path, darks_path: Path,
                  name: str = "r", extra: str = "") -> Path:
    recipe = tmp_path / f"{name}.toml"
    recipe.write_text(f"""
[lights]
paths = ["{lights_path.as_posix()}"]
[darks]
paths = ["{darks_path.as_posix()}"]
start_frame = 4
{extra}
[local_lucky]
noise_wavelength_pixels = 2.0
crossover_wavelength_pixels = 6.0
isoplanatic_patch_pixels = 10.0
[output]
dir = "{(tmp_path / 'out').as_posix()}"
debug_frames = 0
""")
    return recipe


def test_darks_subtracted_and_cached(tmp_path: Path) -> None:
    lights_path, darks_path = _make_data(tmp_path)
    recipe = _write_recipe(tmp_path, lights_path, darks_path)

    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    first = CliRun(proc, parse_events(proc.stdout), 0.0)
    stages = [e["stage"] for e in first.events_of("stage_start")]
    assert stages == ["lights", "darks", "align", "local_lucky", "output"]
    darks_start = [e for e in first.events_of("stage_start") if e["stage"] == "darks"][0]
    assert darks_start["cached"] is False

    # the hot background is gone: corner pixels near zero, blob preserved
    result = np.load(first.run_dir / "local_lucky.npy")
    assert abs(float(result[:4, :4].mean())) < 0.02
    assert float(result.max()) > 0.4

    # second run serves the master dark from cache with immediate stage_end
    proc2 = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    second = CliRun(proc2, parse_events(proc2.stdout), 0.0)
    darks_start2 = [e for e in second.events_of("stage_start") if e["stage"] == "darks"][0]
    assert darks_start2["cached"] is True
    assert not [e for e in second.events_of("progress") if e["stage"] == "darks"]


def test_keep_level_leaves_the_pedestal(tmp_path: Path) -> None:
    """keep_level removes the dark's *pattern* but not its level, so the
    calibrated background sits near the dark's mean instead of near zero."""
    lights_path, darks_path = _make_data(tmp_path)
    plain = _write_recipe(tmp_path, lights_path, darks_path, name="plain")
    kept = _write_recipe(tmp_path, lights_path, darks_path, name="kept",
                         extra="keep_level = true")

    runs = []
    for recipe in (plain, kept):
        proc = run_cli(["run", str(recipe)], cwd=tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        runs.append(CliRun(proc, parse_events(proc.stdout), 0.0))

    background = [float(np.load(r.run_dir / "local_lucky.npy")[:4, :4].mean()) for r in runs]
    assert abs(background[0]) < 0.02, background          # subtracted away
    assert abs(background[1] - 0.10) < 0.02, background   # the 0.10 offset survives

    # the two differ only by that constant: same pattern removed either way
    a = np.load(runs[0].run_dir / "local_lucky.npy")
    b = np.load(runs[1].run_dir / "local_lucky.npy")
    assert np.allclose(b - a, float((b - a).mean()), atol=0.02)

    # ...and the level shift is reported, not silent
    assert any("keep_level" in e["message"] for e in runs[1].events_of("log"))
