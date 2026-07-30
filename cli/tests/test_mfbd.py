"""[mfbd] stage: recipe validation, end-to-end run, honest science check.

The scientific comparison is reported in two ways because they answer
different questions:

* *as-is* MSE vs truth compares the files a user actually gets;
* *registered* MSE first searches a +/-1.5 px sub-pixel translation, because
  the deconvolved object inherits the residual sub-pixel offset of its
  tip/tilt reference frame while the lucky stack averages such offsets out —
  a global shift is not a sharpness difference.

On the synthetic data the deconvolution beats the lucky stack as-is; after
registration the plain stack of the sharpest frames is actually closer to
truth, so we assert generous absolute ceilings rather than strict
superiority, keeping the suite robust to numeric drift.
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

MFBD_RECIPE = f"""
[recipe]
version = 0
name = "mfbd"

[lights]
paths = ["{SYNTHETIC_SER.as_posix()}"]

[align]
crop = [192, 192]

[lucky_scoring]
min_wavelength_pixels = 5.0
max_wavelength_pixels = 50.0

# the same 6 frames as the deconvolution, plainly stacked, for comparison
[lucky_stack]
top_fractions = [0.1]

[mfbd]
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

def test_mfbd_recipe_defaults(tmp_path: Path) -> None:
    r = _load(tmp_path, MFBD_RECIPE)
    assert r.mfbd is not None
    assert r.mfbd.frames == "lucky_top"
    assert r.mfbd.psf_model == "kl"
    assert r.mfbd.n_modes == 20
    assert r.mfbd.frequency_cutoff == (0.2, 0.3)
    assert r.mfbd.apodization_border == 0
    assert r.mfbd.focal_length_mm == 2800.0
    assert r.mfbd.barlow == 1.0
    d = r.resolved_dict()
    assert d["mfbd"]["wavelengths_nm"] == [700.0, 530.0, 470.0]

    # Aperture defaults to the author's C11 when not given
    r2 = _load(tmp_path, MFBD_RECIPE.replace("diameter_cm = 20.0\n", ""))
    assert r2.mfbd.diameter_cm == 27.94
    assert r2.mfbd.central_obscuration_cm == 9.5


def test_mfbd_pixel_scale_computed_from_camera(tmp_path: Path) -> None:
    """pixel_size_um + focal_length_mm + barlow -> effective pixel scale."""
    text = MFBD_RECIPE.replace(
        "pixel_scale_arcsec = 0.25",
        "focal_length_mm = 2800.0\nbarlow = 2.0\npixel_size_um = 4.3",
    )
    r = _load(tmp_path, text)
    expected = 206.265 * 4.3 / (2800.0 * 2.0)  # ~0.1584 arcsec/px
    assert r.mfbd.pixel_scale_arcsec == pytest.approx(expected)
    d = r.resolved_dict()["mfbd"]
    # consumers must see the resolved effective value plus provenance
    assert d["pixel_scale_arcsec"] == pytest.approx(expected)
    assert d["pixel_size_um"] == 4.3
    assert d["barlow"] == 2.0


def test_mfbd_recipe_errors(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match=r"\[mfbd\] wavelengths_nm"):
        _load(tmp_path, MFBD_RECIPE.replace("wavelengths_nm = [700.0, 530.0, 470.0]\n", ""))
    def in_deconv(extra: str) -> str:
        return MFBD_RECIPE.replace("top_n = 6", f"top_n = 6\n{extra}")

    with pytest.raises(RecipeError, match=r"\[mfbd\] n_modes.*full radial degree"):
        _load(tmp_path, in_deconv("n_modes = 15"))
    with pytest.raises(RecipeError, match="unknown key 'iterationz'"):
        _load(tmp_path, in_deconv("iterationz = 3"))
    with pytest.raises(RecipeError, match=r"\[mfbd\] frames"):
        _load(tmp_path, in_deconv('frames = "best"'))
    with pytest.raises(RecipeError, match=r"\[mfbd\] frequency_cutoff"):
        _load(tmp_path, in_deconv("frequency_cutoff = [0.5, 0.4]"))
    # both pixel-scale forms at once: ambiguous, hard error
    with pytest.raises(RecipeError, match="mutually exclusive"):
        _load(tmp_path, in_deconv("pixel_size_um = 4.3"))
    # neither form: hard error explaining both options
    with pytest.raises(RecipeError, match="pixel scale is required.*pixel_scale_arcsec.*pixel_size_um"):
        _load(tmp_path, MFBD_RECIPE.replace("pixel_scale_arcsec = 0.25\n", ""))


def test_mfbd_wavelength_channel_mismatch_is_runtime_error(tmp_path: Path) -> None:
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
paths = ["{path.as_posix()}"]
[lucky_scoring]
[mfbd]
diameter_cm = 20.0
pixel_scale_arcsec = 0.25
wavelengths_nm = [700.0, 530.0, 470.0]
[output]
dir = "{(tmp_path / 'out').as_posix()}"
debug_frames = 0
""")
    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode != 0
    events = parse_events(proc.stdout)
    assert events[-1]["event"] == "error"
    assert events[-1].get("stage") == "mfbd"
    assert "wavelengths_nm" in events[-1]["message"]


def test_undersampling_warns_but_never_fails() -> None:
    """Coarse pixel scales log warnings (severe or Nyquist) instead of raising."""
    from tensorez.deconv import apply_superpixel_scale, check_optics
    from tensorez.recipe import MfbdConfig

    def collect(cfg):
        events: list[tuple[str, str]] = []
        check_optics(cfg, channels=1, log=lambda m, level="info": events.append((level, m)))
        return events

    # lambda/D at 550 nm on a C11 is ~0.41"; 1.0"/px doesn't even span it
    severe = collect(MfbdConfig(pixel_scale_arcsec=1.0, wavelengths_nm=(550.0,)))
    assert any(lvl == "warning" and "SEVERELY undersampled" in m for lvl, m in severe)
    # 0.3"/px spans it with ~1.35 px: below Nyquist but representable
    mild = collect(MfbdConfig(pixel_scale_arcsec=0.3, wavelengths_nm=(550.0,)))
    assert any(lvl == "warning" and "below Nyquist" in m for lvl, m in mild)
    # 0.1"/px is comfortably oversampled: silence
    assert collect(MfbdConfig(pixel_scale_arcsec=0.1, wavelengths_nm=(550.0,))) == []

    # superpixel debayering doubles the effective scale (sensor-referred keys)
    cfg = MfbdConfig(pixel_scale_arcsec=0.2, wavelengths_nm=(550.0,))
    assert apply_superpixel_scale(cfg, True, "superpixel_rgb").pixel_scale_arcsec == 0.4
    assert apply_superpixel_scale(cfg, True, "superpixel_rggb").pixel_scale_arcsec == 0.4
    assert apply_superpixel_scale(cfg, True, "bilinear") is cfg
    assert apply_superpixel_scale(cfg, False, "superpixel_rgb") is cfg


# -- (b) end-to-end + (c) science ------------------------------------------

@dataclass
class MfbdRun:
    run: CliRun
    seconds: float


@pytest.fixture(scope="session")
def mfbd_run(tmp_path_factory: pytest.TempPathFactory) -> MfbdRun:
    workdir = tmp_path_factory.mktemp("mfbd")
    recipe = workdir / "recipe.toml"
    recipe.write_text(
        MFBD_RECIPE.replace(
            "[output]\ndebug_frames = 0",
            f'[output]\ndir = "{(workdir / "out").as_posix()}"\ndebug_frames = 0',
        )
    )
    t0 = time.monotonic()
    proc = run_cli(["run", str(recipe)], cwd=workdir)
    seconds = time.monotonic() - t0
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return MfbdRun(CliRun(proc, parse_events(proc.stdout), seconds), seconds)


def test_mfbd_end_to_end(mfbd_run: MfbdRun) -> None:
    run = mfbd_run.run
    assert run.events[-1]["event"] == "done"
    stages = [e["stage"] for e in run.events_of("stage_start")]
    assert stages == ["lights", "align", "lucky_scoring", "lucky_stack", "mfbd", "output"]

    # progress events with loss, throttled
    prog = [e for e in run.events_of("progress") if e["stage"] == "mfbd"]
    assert prog and prog[-1]["current"] == prog[-1]["total"] == 60
    assert any("loss" in e.get("message", "") for e in prog)

    # every producer published its named product
    by_name = {(e["stage"], e["name"], e["kind"]): e for e in run.events_of("artifact")}
    assert ("mfbd", "loss_history", "array") in by_name
    assert ("mfbd", "psf_examples", "preview") in by_name
    assert ("mfbd", "mfbd", "image") in by_name
    assert ("mfbd", "mfbd", "preview") in by_name
    assert ("lucky_stack", "lucky_stack_p10", "image") in by_name
    assert ("lucky_stack", "lucky_stack_p10", "preview") in by_name
    assert ("lucky_scoring", "frame_scores", "array") in by_name
    for e in run.events_of("artifact"):
        assert (run.run_dir / e["path"]).is_file()

    # loss history is finite and decreased
    loss = np.load(run.run_dir / "stages/mfbd/loss_history.npy")
    assert loss.shape == (60,) and np.isfinite(loss).all()
    assert loss[-1] < loss[0]

    # final.* comes from the deconvolution, not the lucky stack
    final = np.load(run.run_dir / "final.npy")
    lucky = tifffile.imread(
        run.run_dir / "stages/lucky_stack/lucky_stack_p10.tif"
    ).astype(np.float32) / 65535.0
    assert final.shape == lucky.shape == (192, 192, 3)
    assert not np.allclose(np.clip(final, 0, 1), lucky, atol=1e-3)

    # frame scores rank the frames the mfbd log claims to use
    scores = np.load(run.run_dir / "stages/lucky_scoring/frame_scores.npy")
    assert scores.shape == (60,)
    top6 = sorted(int(i) for i in np.argsort(scores)[::-1][:6])
    log_msgs = [e["message"] for e in run.events_of("log")]
    assert any(str(top6) in m for m in log_msgs)


def test_mfbd_science_vs_lucky_stack(mfbd_run: MfbdRun) -> None:
    """(c) Honest comparison of the deconvolution vs the plain lucky stack of
    the same 6 luckiest frames, both against truth.  We assert sane bounds
    rather than strict superiority.
    """
    import scipy.ndimage as nd

    run = mfbd_run.run
    truth = np.load(TRUTH_NPY)
    tc = truth[32:224, 32:224]
    final = np.clip(np.load(run.run_dir / "final.npy"), 0, 1)
    lucky = tifffile.imread(
        run.run_dir / "stages/lucky_stack/lucky_stack_p10.tif"
    ).astype(np.float32) / 65535.0

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
    # As-is: deconv beats the stack (which carries its frames' residual
    # sub-pixel offsets); measured 0.0014 vs 0.0017 when written.
    assert mse_deconv < 1.2 * mse_lucky, (mse_deconv, mse_lucky)
    # Registered: the plain stack of the 6 sharpest frames is extremely close
    # to truth on this synthetic set (9.8e-5 when written) — a good showing
    # for classic lucky imaging; the 60-iteration deconv lands near 1.0e-3.
    # Assert sane absolute ceilings rather than superiority.
    assert reg_lucky < 5e-4, reg_lucky
    assert reg_deconv < 5e-3, reg_deconv


# -- (d) frames = "all" vs "lucky_top" --------------------------------------

@pytest.mark.parametrize("frames_mode", ["lucky_top", "all"])
def test_mfbd_frame_selection_modes(tmp_path: Path, frames_mode: str) -> None:
    recipe = tmp_path / "r.toml"
    recipe.write_text(f"""
[recipe]
version = 0
name = "modes"
[lights]
paths = ["{SYNTHETIC_SER.as_posix()}"]
end_frame = 10
[align]
crop = [128, 128]
[lucky_scoring]
[mfbd]
frames = "{frames_mode}"
top_n = 4
iterations = 6
diameter_cm = 20.0
pixel_scale_arcsec = 0.25
wavelengths_nm = [700.0, 530.0, 470.0]
[output]
dir = "{(tmp_path / 'out').as_posix()}"
debug_frames = 0
""")
    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    events = parse_events(proc.stdout)
    assert events[-1]["event"] == "done"
    n_expected = 10 if frames_mode == "all" else 4
    log_msgs = [e["message"] for e in events if e["event"] == "log"]
    assert any(f"torchmfbd on {n_expected} frame(s)" in m for m in log_msgs)
