"""Strict recipe validation, including the CLI error path."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import SYNTHETIC_SER, parse_events, run_cli
from tensorez.recipe import RecipeError, load_recipe

MINIMAL = f"""
[recipe]
version = 0
name = "t"

[lights]
paths = ["{SYNTHETIC_SER.as_posix()}"]
"""


def _load(tmp_path: Path, text: str):
    p = tmp_path / "r.toml"
    p.write_text(text)
    return load_recipe(p)


def test_minimal_recipe_defaults(tmp_path: Path) -> None:
    r = _load(tmp_path, MINIMAL)
    assert r.name == "t"
    assert r.lights.debayer == "bilinear"
    assert r.align.center_of_mass is True
    assert r.align.crop is None
    assert r.lucky.crossover_wavelength_pixels == 35.0
    assert r.lucky.stdevs_above_mean == 2.5
    assert r.output.debug_frames == 10
    assert r.darks is None


def test_unknown_key_is_error(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="unknown key 'crossover_wavelength_pixel'"):
        _load(tmp_path, MINIMAL + "\n[lucky]\ncrossover_wavelength_pixel = 30.0\n")


def test_unknown_section_is_error(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match=r"unknown section \[luckyy\]"):
        _load(tmp_path, MINIMAL + "\n[luckyy]\nsteepness = 1.0\n")


def test_wrong_types_are_errors(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match=r"\[lights\] frame_step"):
        _load(tmp_path, MINIMAL.replace('paths = ', 'frame_step = 1.5\npaths = '))
    with pytest.raises(RecipeError, match=r"\[align\] crop"):
        _load(tmp_path, MINIMAL + "\n[align]\ncrop = [512]\n")
    with pytest.raises(RecipeError, match=r"\[lucky\] steepness"):
        _load(tmp_path, MINIMAL + "\n[lucky]\nsteepness = \"sharp\"\n")
    with pytest.raises(RecipeError, match="only version 0"):
        _load(tmp_path, MINIMAL.replace("version = 0", "version = 1"))
    with pytest.raises(RecipeError, match=r"\[lights\] debayer"):
        _load(tmp_path, MINIMAL.replace('paths = ', 'debayer = "vng"\npaths = '))
    with pytest.raises(RecipeError, match="must match"):
        _load(tmp_path, MINIMAL.replace('name = "t"', 'name = "bad name!"'))


def test_relative_paths_resolve_against_recipe_dir(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.ser").write_bytes(b"")
    r = _load(tmp_path, MINIMAL.replace(SYNTHETIC_SER.as_posix(), "data/x.ser"))
    assert r.lights.paths[0] == str(tmp_path / "data" / "x.ser")


def test_cli_unknown_key_emits_error_event(tmp_path: Path) -> None:
    """(e) A recipe with an unknown key fails with nonzero exit + `error` event."""
    recipe = tmp_path / "bad.toml"
    recipe.write_text(MINIMAL + "\n[lucky]\nstdevs_above_meen = 2.0\n")
    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode != 0
    events = parse_events(proc.stdout)
    assert events[-1]["event"] == "error"
    assert "stdevs_above_meen" in events[-1]["message"]
