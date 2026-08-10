"""The lucky_fourier producer: per-frequency lucky stacking, end to end.

These run on speckle_planet.ser, whose seeing scrambles Fourier *phases*
(off-center speckles).  That matters: synthetic_planet.ser's centered
gaussian blur is zero-phase, so every frame there already has the truth's
phases and per-frequency selection can only lose (it gives up noise
averaging with nothing to recover) — measured, not hypothetical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import SPECKLE_SER, TRUTH_NPY, CliRun, parse_events, run_cli
from tensorez.lucky_fourier import centroid_align_ramp, toroidal_centroid


def write_fourier_recipe(path: Path, out_dir: Path, extra: str = "") -> Path:
    path.write_text(f"""
[lights]
paths = ["{SPECKLE_SER.as_posix()}"]

[align]
center_of_mass = true
crop = [192, 192]

[lucky_fourier]
{extra}stdevs_above_mean = 2.0
steepness = 3.0

[output]
dir = "{out_dir.as_posix()}"
debug_frames = 3
""")
    return path


@dataclass
class FourierRuns:
    workdir: Path
    first: CliRun
    second: CliRun
    second_validate: CliRun
    sub_off: CliRun  # same recipe with subpixel_align = false


@pytest.fixture(scope="module")
def fourier_runs(tmp_path_factory: pytest.TempPathFactory) -> FourierRuns:
    workdir = tmp_path_factory.mktemp("lucky_fourier")
    recipe = write_fourier_recipe(workdir / "fourier.toml", workdir / "output")

    runs = []
    for _ in range(2):
        proc = run_cli(["run", str(recipe)], cwd=workdir)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        runs.append(CliRun(proc, parse_events(proc.stdout), 0.0))

    proc = run_cli(["validate", str(recipe)], cwd=workdir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    second_validate = CliRun(proc, parse_events(proc.stdout), 0.0)

    off_recipe = write_fourier_recipe(workdir / "fourier_off.toml", workdir / "off_output",
                                      extra="subpixel_align = false\n")
    proc = run_cli(["run", str(off_recipe)], cwd=workdir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    sub_off = CliRun(proc, parse_events(proc.stdout), 0.0)

    return FourierRuns(workdir, runs[0], runs[1], second_validate, sub_off)


def test_stage_runs_and_publishes_product(fourier_runs: FourierRuns) -> None:
    run = fourier_runs.first
    stages = [e["stage"] for e in run.events_of("stage_start")]
    assert stages == ["lights", "align", "lucky_fourier", "output"]
    assert run.events[-1]["event"] == "done"
    assert run.events[-1]["products"] == ["lucky_fourier"]

    for name in ("lucky_fourier.npy", "lucky_fourier.tif", "lucky_fourier.png"):
        assert (run.run_dir / name).is_file(), name
        assert (fourier_runs.workdir / "output" / name).is_file(), name

    # the debug spectra and per-frame weight maps announced themselves
    names = {e["name"] for e in run.events_of("artifact")}
    assert {"spectrum_mean", "spectrum_stdev", "uv_total_weight",
            "unweighted_average"} <= names
    frames = [e for e in run.events_of("artifact") if e["kind"] == "sequence_frame"]
    assert {e["frame"] for e in frames} == {0, 1, 2}


def test_pass1_statistics_are_cached(fourier_runs: FourierRuns) -> None:
    """The stage is never fully cached (pass 2 always runs), but pass 1's
    parameter-free magnitude statistics are served from cache on rerun."""
    def stage_start(run: CliRun) -> dict:
        return next(e for e in run.events_of("stage_start")
                    if e["stage"] == "lucky_fourier")

    assert stage_start(fourier_runs.first)["cached"] is False
    assert stage_start(fourier_runs.first)["pass1_cached"] is False
    assert stage_start(fourier_runs.second)["cached"] is False
    assert stage_start(fourier_runs.second)["pass1_cached"] is True

    v = fourier_runs.second_validate.events_of("validate_result")[0]
    by_stage = {s["stage"]: s["cached"] for s in v["stages"]}
    assert by_stage["lucky_fourier_stats"] is True


def test_gate_tweak_reuses_pass1_statistics(fourier_runs: FourierRuns) -> None:
    """Pass 1 depends only on alignment, so changing any gate knob still
    validates as a stats cache hit."""
    recipe = write_fourier_recipe(fourier_runs.workdir / "fourier2.toml",
                                  fourier_runs.workdir / "output")
    text = recipe.read_text().replace("stdevs_above_mean = 2.0",
                                      "stdevs_above_mean = 1.0")
    recipe.write_text(text.replace("steepness = 3.0", "steepness = 10.0"))
    proc = run_cli(["validate", str(recipe)], cwd=fourier_runs.workdir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    v = parse_events(proc.stdout)[-1]
    by_stage = {s["stage"]: s["cached"] for s in v["stages"]}
    assert by_stage["lucky_fourier_stats"] is True


def _gaussian_blob(h: int, w: int, cy: float, cx: float) -> torch.Tensor:
    yy = torch.arange(h, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, -1)
    return torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0**2)).view(1, 1, h, w)


def test_toroidal_centroid_reads_subpixel_position() -> None:
    """The circular centroid comes out of the two first-harmonic bins,
    including its fractional part — and survives wrap-around, which a
    linear center of mass does not."""
    spec = torch.fft.rfft2(_gaussian_blob(64, 64, 40.25, 21.5))
    ybar, xbar = toroidal_centroid(spec, 64, 64)[0, 0]
    assert abs(float(ybar) - 40.25) < 0.01
    assert abs(float(xbar) - 21.5) < 0.01

    # a blob straddling the wrap edge still reads back its (toroidal) position
    spec = torch.fft.rfft2(torch.roll(_gaussian_blob(64, 64, 32.0, 32.0),
                                      shifts=(31, -33), dims=(-2, -1)))
    ybar, xbar = toroidal_centroid(spec, 64, 64)[0, 0]
    assert abs(float(ybar) - 63.0) < 0.01
    assert abs(float(xbar) - 63.0) < 0.01


def test_centroid_align_ramp_centers_exactly() -> None:
    """The ramp puts the centroid on the center pixel (index (size-1)//2),
    and two copies of the same scene at different sub-pixel positions
    become the same spectrum."""
    h = w = 64
    spec_a = torch.fft.rfft2(_gaussian_blob(h, w, 30.3, 33.8))
    spec_b = torch.fft.rfft2(_gaussian_blob(h, w, 32.6, 30.1))

    aligned_a, delta_a = centroid_align_ramp(spec_a, h, w)
    aligned_b, _ = centroid_align_ramp(spec_b, h, w)

    ybar, xbar = toroidal_centroid(aligned_a, h, w)[0, 0]
    assert abs(float(ybar) - 31.0) < 1e-3 and abs(float(xbar) - 31.0) < 1e-3
    # the reported shift is what it applied
    assert torch.allclose(delta_a[0, 0], torch.tensor([31 - 30.3, 31 - 33.8]),
                          atol=1e-3)
    # both copies land on the same aligned spectrum (sub-pixel gaussian
    # shifts are band-limited enough that this holds tightly)
    assert float((aligned_a - aligned_b).abs().max()) < 1e-3 * float(spec_a.abs().max())


def test_centroid_align_joint_channels() -> None:
    """per_channel=False derives one shift from the channel-summed brightness
    and applies it to every channel — dispersion offsets between channels
    survive (deliberately), while the common shift is removed."""
    h = w = 64
    # two channels, dispersed by half a pixel in x
    img = torch.cat([_gaussian_blob(h, w, 30.0, 33.0),
                     _gaussian_blob(h, w, 30.0, 33.5)], dim=1)
    spec = torch.fft.rfft2(img)

    aligned, delta = centroid_align_ramp(spec, h, w, per_channel=False)
    assert delta.shape == (1, 1, 2)  # one shift for the whole frame
    c = toroidal_centroid(aligned, h, w)[0]  # per-channel, post-alignment
    # the mean landed on the target; the half-pixel dispersion is intact
    assert abs(float(c[:, 1].mean()) - 31.0) < 0.01
    assert abs(float(c[1, 1] - c[0, 1]) - 0.5) < 0.01

    per_ch, delta_ch = centroid_align_ramp(spec, h, w, per_channel=True)
    assert delta_ch.shape == (1, 2, 2)
    c = toroidal_centroid(per_ch, h, w)[0]
    # per-channel mode pulls both channels onto the target: dispersion gone
    assert float((c[:, 1] - 31.0).abs().max()) < 0.01


def test_lucky_fourier_beats_unweighted_average(fourier_runs: FourierRuns) -> None:
    """The scientific claim: selecting each frequency from the frames that
    transmitted it best lands closer to the truth than averaging them all."""
    run = fourier_runs.first
    result = np.load(run.run_dir / "lucky_fourier.npy")
    average = np.load(run.run_dir / "examples/lucky_fourier/unweighted_average.npy")
    truth = np.load(TRUTH_NPY)

    h, w = result.shape[:2]
    y = (truth.shape[0] - h) // 2
    x = (truth.shape[1] - w) // 2
    truth_crop = truth[y : y + h, x : x + w]

    mse_fourier = float(((result - truth_crop) ** 2).mean())
    mse_average = float(((average - truth_crop) ** 2).mean())
    assert np.isfinite(mse_fourier)
    assert mse_fourier < mse_average, (mse_fourier, mse_average)


def _register(img_hwc: np.ndarray) -> np.ndarray:
    """Put an image's toroidal centroid on the center pixel (one whole-frame
    shift), so images that follow different centering conventions can be
    compared without a constant offset polluting the MSE."""
    t = torch.from_numpy(img_hwc).permute(2, 0, 1).unsqueeze(0)
    spec = torch.fft.rfft2(t)
    aligned, _ = centroid_align_ramp(spec, t.shape[-2], t.shape[-1], per_channel=False)
    return torch.fft.irfft2(aligned, s=t.shape[-2:]).squeeze(0).permute(1, 2, 0).numpy()


def test_subpixel_align_lands_on_center_pixel(fourier_runs: FourierRuns) -> None:
    """With subpixel_align on, the product's centroid sits on index
    (size-1)//2 — the contract's stated convention — and the run logged the
    residual it corrected."""
    result = np.load(fourier_runs.first.run_dir / "lucky_fourier.npy")
    t = torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0)
    spec = torch.fft.rfft2(t.sum(dim=1, keepdim=True))
    ybar, xbar = toroidal_centroid(spec, t.shape[-2], t.shape[-1])[0, 0]
    h, w = t.shape[-2], t.shape[-1]
    assert abs(float(ybar) - (h - 1) // 2) < 0.2
    assert abs(float(xbar) - (w - 1) // 2) < 0.2

    logs = [e["message"] for e in fourier_runs.first.events_of("log")]
    assert any("subpixel_align corrected" in m for m in logs)
    off_logs = [e["message"] for e in fourier_runs.sub_off.events_of("log")]
    assert not any("subpixel_align corrected" in m for m in off_logs)


def test_subpixel_align_improves_registered_mse(fourier_runs: FourierRuns) -> None:
    """The point of the feature: after removing the constant centering
    offset from both, the sub-pixel-aligned stack is closer to the truth
    than the integer-only one — residual shifts are phase errors, and the
    ramp takes them out before the phases are averaged."""
    on = np.load(fourier_runs.first.run_dir / "lucky_fourier.npy")
    off = np.load(fourier_runs.sub_off.run_dir / "lucky_fourier.npy")
    truth = np.load(TRUTH_NPY)

    h, w = on.shape[:2]
    y = (truth.shape[0] - h) // 2
    x = (truth.shape[1] - w) // 2
    truth_reg = _register(truth[y : y + h, x : x + w])

    mse_on = float(((_register(on) - truth_reg) ** 2).mean())
    mse_off = float(((_register(off) - truth_reg) ** 2).mean())
    assert mse_on < mse_off, (mse_on, mse_off)
