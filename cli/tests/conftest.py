"""Shared fixtures: run the CLI as a subprocess on the synthetic data once,
then let several tests inspect the results (events, outputs, caching)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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


def run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
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
[recipe]
version = 0
name = "synthetic"

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
