"""[deconv] stage: recipe validation, end-to-end run, honest science check.

The scientific comparison is reported in two ways because they answer
different questions:

* *as-is* MSE vs truth compares the files a user actually gets;
* *registered* MSE first searches a +/-1.5 px sub-pixel translation, because
  the deconvolved object inherits the residual sub-pixel offset of its
  tip/tilt reference frame while the lucky stack averages such offsets out —
  a global shift is not a sharpness difference.

On the synthetic data the deconvolution beats the lucky stack as-is and is
within a few tens of percent registered; we assert generous bounds rather
than strict superiority so the suite stays robust to numeric drift.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile

from conftest import SYNTHETIC_SER, TRUTH_NPY, CliRun, parse_events, run_cli
from tensorez import ser
from tensorez.recipe import RecipeError, load_recipe

DECONV_RECIPE = f"""
[recipe]
version = 0
name = "deconv"

[lights]
paths = ["{SYNTHETIC_SER}"]

[align]
crop = [192, 192]

[lucky]
crossover_wavelength_pixels = 30.0
isoplanatic_patch_pixels = 50.0
stdevs_above_mean = 2.0

[deconv]
top_n = 6
iterations = 60
diameter_cm = 20.0
pixel_scale_arcsec = 0.25
wavelengths_nm = [700.0, 530.0, 470.0]

[output]
debug_frames = 0
"""


def _load(tmp_path: Path, text: str):
    p = tmp_path / "r.toml"
    p.write_text(text)
    return load_recipe(p)


# -- (a) recipe validation --------------------------------------------------

def test_deconv_recipe_defaults(tmp_path: Path) -> None:
    r = _load(tmp_path, DECONV_RECIPE)
    assert r.deconv is not None
    assert r.deconv.frames == "lucky_top"
    assert r.deconv.psf_model == "kl"
    assert r.deconv.n_modes == 20
    assert r.deconv.frequency_cutoff == (0.2, 0.3)
    assert r.deconv.apodization_border == 0
    assert r.deconv.focal_length_mm == 2800.0
    assert r.deconv.barlow == 1.0
    d = r.resolved_dict()
    assert d["deconv"]["wavelengths_nm"] == [700.0, 530.0, 470.0]

    # Aperture defaults to the author's C11 when not given
    r2 = _load(tmp_path, DECONV_RECIPE.replace("diameter_cm = 20.0\n", ""))
    assert r2.deconv.diameter_cm == 27.94
    assert r2.deconv.central_obscuration_cm == 9.5


def test_deconv_pixel_scale_computed_from_camera(tmp_path: Path) -> None:
    """pixel_size_um + focal_length_mm + barlow -> effective pixel scale."""
    text = DECONV_RECIPE.replace(
        "pixel_scale_arcsec = 0.25",
        "focal_length_mm = 2800.0\nbarlow = 2.0\npixel_size_um = 4.3",
    )
    r = _load(tmp_path, text)
    expected = 206.265 * 4.3 / (2800.0 * 2.0)  # ~0.1584 arcsec/px
    assert r.deconv.pixel_scale_arcsec == pytest.approx(expected)
    d = r.resolved_dict()["deconv"]
    # consumers must see the resolved effective value plus provenance
    assert d["pixel_scale_arcsec"] == pytest.approx(expected)
    assert d["pixel_size_um"] == 4.3
    assert d["barlow"] == 2.0


def test_deconv_recipe_errors(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match=r"\[deconv\] wavelengths_nm"):
        _load(tmp_path, DECONV_RECIPE.replace("wavelengths_nm = [700.0, 530.0, 470.0]\n", ""))
    def in_deconv(extra: str) -> str:
        return DECONV_RECIPE.replace("top_n = 6", f"top_n = 6\n{extra}")

    with pytest.raises(RecipeError, match=r"\[deconv\] n_modes.*full radial degree"):
        _load(tmp_path, in_deconv("n_modes = 15"))
    with pytest.raises(RecipeError, match="unknown key 'iterationz'"):
        _load(tmp_path, in_deconv("iterationz = 3"))
    with pytest.raises(RecipeError, match=r"\[deconv\] frames"):
        _load(tmp_path, in_deconv('frames = "best"'))
    with pytest.raises(RecipeError, match=r"\[deconv\] frequency_cutoff"):
        _load(tmp_path, in_deconv("frequency_cutoff = [0.5, 0.4]"))
    # both pixel-scale forms at once: ambiguous, hard error
    with pytest.raises(RecipeError, match="mutually exclusive"):
        _load(tmp_path, in_deconv("pixel_size_um = 4.3"))
    # neither form: hard error explaining both options
    with pytest.raises(RecipeError, match="pixel scale is required.*pixel_scale_arcsec.*pixel_size_um"):
        _load(tmp_path, DECONV_RECIPE.replace("pixel_scale_arcsec = 0.25\n", ""))


def test_deconv_wavelength_channel_mismatch_is_runtime_error(tmp_path: Path) -> None:
    """3 wavelengths against MONO lights fails with a clear error event."""
    frames = (np.random.default_rng(3).random((6, 32, 32, 1)) * 65535).astype(np.uint16)
    path = tmp_path / "mono.ser"
    ser.write_ser(path, frames, ser.ColorId.MONO)
    recipe = tmp_path / "r.toml"
    recipe.write_text(f"""
[recipe]
version = 0
name = "mono"
[lights]
paths = ["{path}"]
[lucky]
crossover_wavelength_pixels = 6.0
isoplanatic_patch_pixels = 10.0
[deconv]
diameter_cm = 20.0
pixel_scale_arcsec = 0.25
wavelengths_nm = [700.0, 530.0, 470.0]
[output]
dir = "{tmp_path / 'out'}"
debug_frames = 0
""")
    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode != 0
    events = parse_events(proc.stdout)
    assert events[-1]["event"] == "error"
    assert events[-1].get("stage") == "deconv"
    assert "wavelengths_nm" in events[-1]["message"]


# -- (b) end-to-end + (c) science ------------------------------------------

@dataclass
class DeconvRun:
    run: CliRun
    seconds: float


@pytest.fixture(scope="session")
def deconv_run(tmp_path_factory: pytest.TempPathFactory) -> DeconvRun:
    workdir = tmp_path_factory.mktemp("deconv")
    recipe = workdir / "recipe.toml"
    recipe.write_text(
        DECONV_RECIPE.replace(
            "[output]\ndebug_frames = 0",
            f'[output]\ndir = "{workdir / "out"}"\ndebug_frames = 0',
        )
    )
    t0 = time.monotonic()
    proc = run_cli(["run", str(recipe)], cwd=workdir)
    seconds = time.monotonic() - t0
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return DeconvRun(CliRun(proc, parse_events(proc.stdout), seconds), seconds)


def test_deconv_end_to_end(deconv_run: DeconvRun) -> None:
    run = deconv_run.run
    assert run.events[-1]["event"] == "done"
    stages = [e["stage"] for e in run.events_of("stage_start")]
    assert stages == ["lights", "align", "lucky", "deconv", "output"]

    # progress events with loss, throttled
    prog = [e for e in run.events_of("progress") if e["stage"] == "deconv"]
    assert prog and prog[-1]["current"] == prog[-1]["total"] == 60
    assert any("loss" in e.get("message", "") for e in prog)

    # deconv artifacts
    by_name = {(e["stage"], e["name"], e["kind"]): e for e in run.events_of("artifact")}
    assert ("deconv", "loss_history", "array") in by_name
    assert ("deconv", "psf_examples", "preview") in by_name
    assert ("lucky", "lucky_stack", "image") in by_name
    assert ("lucky", "lucky_stack", "preview") in by_name
    assert ("lucky", "frame_scores", "array") in by_name
    for e in run.events_of("artifact"):
        assert (run.run_dir / e["path"]).is_file()

    # loss history is finite and decreased
    loss = np.load(run.run_dir / "stages/deconv/loss_history.npy")
    assert loss.shape == (60,) and np.isfinite(loss).all()
    assert loss[-1] < loss[0]

    # final.* comes from the deconvolution, not the lucky stack
    final = np.load(run.run_dir / "final.npy")
    lucky = tifffile.imread(run.run_dir / "stages/lucky/lucky_stack.tif").astype(np.float32) / 65535.0
    assert final.shape == lucky.shape == (192, 192, 3)
    assert not np.allclose(np.clip(final, 0, 1), lucky, atol=1e-3)

    # frame scores rank the frames the deconv log claims to use
    scores = np.load(run.run_dir / "stages/lucky/frame_scores.npy")
    assert scores.shape == (60,)
    top6 = sorted(int(i) for i in np.argsort(scores)[::-1][:6])
    log_msgs = [e["message"] for e in run.events_of("log")]
    assert any(str(top6) in m for m in log_msgs)


def test_deconv_science_vs_lucky_stack(deconv_run: DeconvRun) -> None:
    """(c) Honest comparison of deconv vs the lucky stack against truth.

    Empirically on this synthetic set (see numbers in the assert messages):
    deconv beats the lucky stack as-is; after registering out the global
    sub-pixel offset both are close, lucky slightly ahead.  We assert sane
    bounds rather than strict superiority.
    """
    import scipy.ndimage as nd

    run = deconv_run.run
    truth = np.load(TRUTH_NPY)
    tc = truth[32:224, 32:224]
    final = np.clip(np.load(run.run_dir / "final.npy"), 0, 1)
    lucky = tifffile.imread(run.run_dir / "stages/lucky/lucky_stack.tif").astype(np.float32) / 65535.0

    def mse(a: np.ndarray) -> float:
        return float(((a - tc) ** 2).mean())

    def registered_mse(a: np.ndarray) -> float:
        best = np.inf
        for dy in np.arange(-1.5, 1.51, 0.25):
            for dx in np.arange(-1.5, 1.51, 0.25):
                shifted = nd.shift(a, (dy, dx, 0), order=3, mode="nearest")
                best = min(best, float(((np.clip(shifted, 0, 1) - tc) ** 2).mean()))
        return best

    mse_deconv, mse_lucky = mse(final), mse(lucky)
    reg_deconv, reg_lucky = registered_mse(final), registered_mse(lucky)

    assert np.isfinite([mse_deconv, mse_lucky, reg_deconv, reg_lucky]).all()
    assert mse_deconv >= 0 and reg_deconv >= 0
    # As-is: deconv has beaten the lucky stack here (0.00144 vs 0.00167 when
    # written); allow headroom but fail if it ever gets meaningfully worse.
    assert mse_deconv < 1.2 * mse_lucky, (mse_deconv, mse_lucky)
    # Registered (fair sharpness metric): lucky was slightly ahead
    # (0.00064 vs 0.00075); deconv must stay in the same league.
    assert reg_deconv < 1.5 * reg_lucky, (reg_deconv, reg_lucky)


# -- (d) frames = "all" vs "lucky_top" --------------------------------------

@pytest.mark.parametrize("frames_mode", ["lucky_top", "all"])
def test_deconv_frame_selection_modes(tmp_path: Path, frames_mode: str) -> None:
    recipe = tmp_path / "r.toml"
    recipe.write_text(f"""
[recipe]
version = 0
name = "modes"
[lights]
paths = ["{SYNTHETIC_SER}"]
end_frame = 10
[align]
crop = [128, 128]
[lucky]
crossover_wavelength_pixels = 20.0
isoplanatic_patch_pixels = 40.0
[deconv]
frames = "{frames_mode}"
top_n = 4
iterations = 6
diameter_cm = 20.0
pixel_scale_arcsec = 0.25
wavelengths_nm = [700.0, 530.0, 470.0]
[output]
dir = "{tmp_path / 'out'}"
debug_frames = 0
""")
    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    events = parse_events(proc.stdout)
    assert events[-1]["event"] == "done"
    n_expected = 10 if frames_mode == "all" else 4
    log_msgs = [e["message"] for e in events if e["event"] == "log"]
    assert any(f"torchmfbd on {n_expected} frame(s)" in m for m in log_msgs)
