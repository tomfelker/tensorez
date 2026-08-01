"""[lucky_scoring] + [lucky_stack]: score caching, stack outputs, ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch

from conftest import SYNTHETIC_SER, CliRun, parse_events, run_cli
from tensorez.scoring import FrameScorer, ScoringParams


def _recipe(tmp_path: Path) -> Path:
    p = tmp_path / "r.toml"
    p.write_text(f"""
[lights]
paths = ["{SYNTHETIC_SER.as_posix()}"]
end_frame = 20
[align]
crop = [128, 128]
[lucky_scoring]
[lucky_stack]
top_fractions = [0.05, 1.0]
[output]
dir = "{(tmp_path / 'out').as_posix()}"
debug_frames = 0
""")
    return p


def test_scoring_and_stack_end_to_end(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path)
    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run = CliRun(proc, parse_events(proc.stdout), 0.0)

    stages = [e["stage"] for e in run.events_of("stage_start")]
    assert stages == ["lights", "align", "lucky_scoring", "lucky_stack", "output"]

    scores = np.load(run.run_dir / "examples/lucky_scoring/frame_scores.npy")
    assert scores.shape == (20,) and np.isfinite(scores).all()

    # 0.05 of 20 frames -> ceil = 1 (the single best frame); 1.0 -> all 20
    p5 = tifffile.imread(run.run_dir / "lucky_stack_p5.tif")
    p100 = tifffile.imread(run.run_dir / "lucky_stack_p100.tif")
    assert p5.shape == p100.shape == (128, 128, 3)
    logs = [e["message"] for e in run.events_of("log")]
    assert any("lucky_stack_p5 = best 1 of 20" in m for m in logs)
    assert any("lucky_stack_p100 = best 20 of 20" in m for m in logs)

    # one product per fraction, each named for it, all three formats, and
    # copied out of the run dir into the output dir
    done = run.events_of("done")[0]
    assert done["products"] == ["lucky_stack_p5", "lucky_stack_p100"]
    out_dir = Path(done["output_dir"])
    for name in done["products"]:
        for suffix in (".npy", ".tif", ".png"):
            assert (run.run_dir / (name + suffix)).is_file(), name + suffix
            assert (out_dir / (name + suffix)).is_file(), name + suffix
    assert np.array_equal(tifffile.imread(out_dir / "lucky_stack_p5.tif"), p5)

    # the best single frame out-scores the mean of everything on the same
    # metric — the point of lucky imaging
    scorer = FrameScorer(128, 128, ScoringParams())
    to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 65535.0).permute(2, 0, 1).unsqueeze(0)
    assert scorer.score(to_t(p5)) > scorer.score(to_t(p100))

    # second run: scores come from cache, stage is fully cached
    proc2 = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    run2 = CliRun(proc2, parse_events(proc2.stdout), 0.0)
    scoring2 = [e for e in run2.events_of("stage_start") if e["stage"] == "lucky_scoring"][0]
    assert scoring2["cached"] is True
    assert not [e for e in run2.events_of("progress") if e["stage"] == "lucky_scoring"]
    assert np.array_equal(
        scores, np.load(run2.run_dir / "examples/lucky_scoring/frame_scores.npy")
    )
    # ...and the second run's products overwrote the first's in the output dir
    assert run2.run_dir != run.run_dir
    assert Path(run2.events_of("done")[0]["output_dir"]) == out_dir


def test_image_squared_metric_prefers_sharp() -> None:
    """Muller-Buffington: blurring conserves flux but lowers the sum of squares."""
    sharp = torch.zeros(1, 1, 32, 32)
    sharp[0, 0, 16, 16] = 1.0  # all flux in one pixel
    blurred = torch.full((1, 1, 32, 32), 1.0 / (32 * 32))  # same flux, spread out
    scorer = FrameScorer(32, 32, ScoringParams(metric="image_squared"))
    assert scorer.score(sharp) > scorer.score(blurred)
