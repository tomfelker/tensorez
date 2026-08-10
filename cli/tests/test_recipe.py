"""Strict recipe validation, including the CLI error path."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import SYNTHETIC_SER, parse_events, run_cli
from tensorez.recipe import RecipeError, load_recipe

MINIMAL = f"""
[lights]
paths = ["{SYNTHETIC_SER.as_posix()}"]

[local_lucky]
"""


def _load(tmp_path: Path, text: str):
    p = tmp_path / "r.toml"
    p.write_text(text)
    return load_recipe(p)


def test_minimal_recipe_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    r = _load(tmp_path, MINIMAL)
    # the recipe file names the run, and the output directory it produces —
    # which lands in the working directory, like every other relative path
    assert r.name == "r"
    assert r.output_dir == tmp_path / "r"
    assert r.lights.debayer == "bilinear"
    assert r.align.center_of_mass is True
    assert r.align.crop is None
    assert r.local_lucky is not None
    assert r.local_lucky.crossover_wavelength_pixels == 35.0
    assert r.local_lucky.stdevs_above_mean == 2.5
    assert r.lucky_scoring is None
    assert r.lucky_stack is None
    assert r.mfbd is None
    assert r.output.debug_frames == 10
    assert r.darks is None


def test_unknown_key_is_error(tmp_path: Path) -> None:
    # MINIMAL ends inside [local_lucky], so the typo'd key lands there
    with pytest.raises(RecipeError, match="unknown key 'crossover_wavelength_pixel'"):
        _load(tmp_path, MINIMAL + "crossover_wavelength_pixel = 30.0\n")


def test_unknown_section_is_error(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match=r"unknown section \[luckyy\]"):
        _load(tmp_path, MINIMAL + "\n[luckyy]\nsteepness = 1.0\n")
    # the pre-split section names are gone, not silently accepted
    with pytest.raises(RecipeError, match=r"unknown section \[lucky\]"):
        _load(tmp_path, MINIMAL + "\n[lucky]\nsteepness = 1.0\n")
    with pytest.raises(RecipeError, match=r"unknown section \[deconv\]"):
        _load(tmp_path, MINIMAL + "\n[deconv]\ntop_n = 4\n")


def test_producer_branch_rules(tmp_path: Path) -> None:
    no_producer = MINIMAL.replace("[local_lucky]\n", "")
    with pytest.raises(RecipeError, match="nothing produces an output"):
        _load(tmp_path, no_producer)
    with pytest.raises(RecipeError, match=r"\[lucky_stack\] requires \[lucky_scoring\]"):
        _load(tmp_path, MINIMAL + "\n[lucky_stack]\ntop_fractions = [0.1]\n")
    with pytest.raises(RecipeError, match=r"\[mfbd\] frames = 'lucky_top' requires"):
        _load(tmp_path, MINIMAL + "\n[mfbd]\nwavelengths_nm = [550.0]\n"
                                  "pixel_scale_arcsec = 0.25\n")
    with pytest.raises(RecipeError, match=r"\[lucky_stack\] top_fractions"):
        _load(tmp_path, MINIMAL + "\n[lucky_scoring]\n[lucky_stack]\ntop_fractions = [0.0]\n")
    with pytest.raises(RecipeError, match="top_n and top_fraction are mutually exclusive"):
        _load(tmp_path, MINIMAL + "\n[lucky_scoring]\n[mfbd]\ntop_n = 8\ntop_fraction = 0.1\n"
                                  "wavelengths_nm = [550.0]\npixel_scale_arcsec = 0.25\n")
    # frames = "all" needs no scoring
    r = _load(tmp_path, MINIMAL + "\n[mfbd]\nframes = \"all\"\n"
                                  "wavelengths_nm = [550.0]\npixel_scale_arcsec = 0.25\n")
    assert r.mfbd is not None and r.lucky_scoring is None


def test_lucky_fourier_section(tmp_path: Path) -> None:
    # a lone [lucky_fourier] is a valid producer, with defaults
    r = _load(tmp_path, MINIMAL.replace("[local_lucky]", "[lucky_fourier]"))
    assert r.local_lucky is None
    assert r.lucky_fourier is not None
    assert r.lucky_fourier.stdevs_above_mean == 2.5
    assert r.lucky_fourier.steepness == 3.0
    assert r.lucky_fourier.channel_crosstalk == 0.0
    assert r.lucky_fourier.subpixel_align is True
    assert r.lucky_fourier.per_channel is False
    assert r.resolved_dict()["lucky_fourier"]["steepness"] == 3.0
    r = _load(tmp_path, MINIMAL.replace("[local_lucky]", "[lucky_fourier]")
              + "per_channel = true\n")
    assert r.lucky_fourier.per_channel is True
    assert r.resolved_dict()["lucky_fourier"]["per_channel"] is True
    with pytest.raises(RecipeError, match=r"\[lucky_fourier\] unknown key 'top_fraction'"):
        _load(tmp_path, MINIMAL + "\n[lucky_fourier]\ntop_fraction = 0.1\n")
    with pytest.raises(RecipeError, match=r"\[lucky_fourier\] channel_crosstalk"):
        _load(tmp_path, MINIMAL + "\n[lucky_fourier]\nchannel_crosstalk = 1.5\n")


def test_wrong_types_are_errors(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match=r"\[lights\] frame_step"):
        _load(tmp_path, MINIMAL.replace('paths = ', 'frame_step = 1.5\npaths = '))
    with pytest.raises(RecipeError, match=r"\[align\] crop"):
        _load(tmp_path, MINIMAL + "\n[align]\ncrop = [512]\n")
    with pytest.raises(RecipeError, match=r"\[local_lucky\] steepness"):
        _load(tmp_path, MINIMAL + "steepness = \"sharp\"\n")
    with pytest.raises(RecipeError, match=r"\[lights\] debayer"):
        _load(tmp_path, MINIMAL.replace('paths = ', 'debayer = "vng"\npaths = '))
    # the whole [recipe] section is gone: the filename names the run, and
    # there is no schema version to declare
    with pytest.raises(RecipeError, match=r"unknown section \[recipe\]"):
        _load(tmp_path, "[recipe]\nversion = 0\n" + MINIMAL)


def test_output_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    r = _load(tmp_path, MINIMAL + '\n[output]\ndir = "results"\n')
    assert r.output_dir == tmp_path / "results"


def test_relative_paths_resolve_against_working_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not against the recipe's own directory — the shell rule, so that what a
    path means doesn't change when a recipe is copied into a run archive."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.ser").write_bytes(b"")
    (tmp_path / "elsewhere").mkdir()
    text = MINIMAL.replace(SYNTHETIC_SER.as_posix(), "data/x.ser")

    monkeypatch.chdir(tmp_path)
    r = _load(tmp_path / "elsewhere", text)   # recipe over there, cwd here
    assert r.lights.paths[0] == str(tmp_path / "data" / "x.ser")

    monkeypatch.chdir(tmp_path / "elsewhere")
    r = _load(tmp_path / "elsewhere", text)   # same recipe, different cwd
    assert r.lights.paths[0] == str(tmp_path / "elsewhere" / "data" / "x.ser")


def test_cli_unknown_key_emits_error_event(tmp_path: Path) -> None:
    """(e) A recipe with an unknown key fails with nonzero exit + `error` event."""
    recipe = tmp_path / "bad.toml"
    recipe.write_text(MINIMAL + "stdevs_above_meen = 2.0\n")
    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode != 0
    events = parse_events(proc.stdout)
    assert events[-1]["event"] == "error"
    assert "stdevs_above_meen" in events[-1]["message"]
