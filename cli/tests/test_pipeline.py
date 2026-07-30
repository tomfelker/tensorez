"""End-to-end pipeline tests on the synthetic planet sequence."""

from __future__ import annotations

import json

import numpy as np

from conftest import TRUTH_NPY, SyntheticRuns


def test_end_to_end_jsonl(synthetic_runs: SyntheticRuns) -> None:
    """(a) The pipeline runs from a recipe and emits valid JSONL ending in `done`."""
    run = synthetic_runs.first
    assert run.events, "no events emitted"
    assert run.events[0]["event"] == "run_start"
    assert run.events[0]["frame_count"] == 60
    assert "recipe" in run.events[0]
    assert run.events[-1]["event"] == "done"
    assert run.events[-1]["final"] == "final.tif"

    for e in run.events:
        assert "event" in e and "t" in e
        assert isinstance(e["t"], (int, float))
    ts = [e["t"] for e in run.events]
    assert ts == sorted(ts), "event timestamps must be monotonic"

    stages = [e["stage"] for e in run.events_of("stage_start")]
    assert stages == ["lights", "align", "lucky", "output"]
    ends = [e["stage"] for e in run.events_of("stage_end")]
    assert ends == stages

    # artifact paths are relative to run_dir and must exist
    for e in run.events_of("artifact"):
        assert not e["path"].startswith("/")
        assert (run.run_dir / e["path"]).is_file(), e["path"]

    # run dir layout per contract §3
    for name in ("manifest.json", "recipe.toml", "log.txt",
                 "final_preview.png", "final.tif", "final.npy"):
        assert (run.run_dir / name).is_file(), name

    # log.txt mirrors the raw stream
    logged = [json.loads(line) for line in
              (run.run_dir / "log.txt").read_text().splitlines() if line.strip()]
    assert [e["event"] for e in logged] == [e["event"] for e in run.events]

    manifest = json.loads((run.run_dir / "manifest.json").read_text())
    assert manifest["manifest_version"] == 0
    assert manifest["run"]["frame_count"] == 60
    assert {a["path"] for a in manifest["artifacts"]} == \
           {e["path"] for e in run.events_of("artifact")}

    # per-frame debug artifacts for the first debug_frames=3 frames
    frames = [e for e in run.events_of("artifact") if e["kind"] == "sequence_frame"]
    assert {e["frame"] for e in frames} == {0, 1, 2}


def test_lucky_beats_unweighted_average(synthetic_runs: SyntheticRuns) -> None:
    """(b) The core scientific claim: the lucky stack is closer to the truth
    than the unweighted aligned average of the same frames."""
    run = synthetic_runs.first
    final = np.load(run.run_dir / "final.npy")
    average = np.load(run.run_dir / "stages/lucky/unweighted_average.npy")
    truth = np.load(TRUTH_NPY)

    h, w = final.shape[:2]
    y = (truth.shape[0] - h) // 2
    x = (truth.shape[1] - w) // 2
    truth_crop = truth[y : y + h, x : x + w]

    mse_lucky = float(((final - truth_crop) ** 2).mean())
    mse_average = float(((average - truth_crop) ** 2).mean())
    assert mse_lucky < mse_average, (mse_lucky, mse_average)


def test_second_run_uses_cache(synthetic_runs: SyntheticRuns) -> None:
    """(c) Rerunning the identical recipe reports cached stages and is faster."""
    first, second = synthetic_runs.first, synthetic_runs.second

    def cached_flags(run):
        return {e["stage"]: e["cached"] for e in run.events_of("stage_start")}

    assert cached_flags(first)["align"] is False
    assert cached_flags(second)["align"] is True

    lucky_second = [e for e in second.events_of("stage_start") if e["stage"] == "lucky"][0]
    assert lucky_second["pass1_cached"] is True
    # cached stages emit no progress events
    assert not [e for e in second.events_of("progress") if e["stage"] == "align"]
    # pass 1 is skipped: only pass 2 progress remains
    lucky_progress = [e for e in second.events_of("progress") if e["stage"] == "lucky"]
    assert all("pass 2" in e.get("message", "") for e in lucky_progress)

    # Compare the pipeline's own reported time (the `done` event), not subprocess
    # wall clock — interpreter + torch import (~1.5s on small machines) dominates
    # wall clock and makes a ratio assertion flaky.
    first_pipeline_s = first.events_of("done")[0]["seconds"]
    second_pipeline_s = second.events_of("done")[0]["seconds"]
    assert second_pipeline_s < first_pipeline_s * 0.75, (first_pipeline_s, second_pipeline_s)

    # identical recipe -> identical result
    a = np.load(first.run_dir / "final.npy")
    b = np.load(second.run_dir / "final.npy")
    assert np.array_equal(a, b)


def test_selection_change_reuses_pass1_stats(synthetic_runs: SyntheticRuns) -> None:
    """(d) Changing only [lucky] stdevs_above_mean reuses the pass-1 stats cache."""
    validate = synthetic_runs.reselect_validate
    result = [e for e in validate.events if e["event"] == "validate_result"][0]
    cached = {s["stage"]: s["cached"] for s in result["stages"]}
    assert cached["align"] is True
    assert cached["lucky_stats"] is True

    run = synthetic_runs.reselect
    lucky = [e for e in run.events_of("stage_start") if e["stage"] == "lucky"][0]
    assert lucky["pass1_cached"] is True
    lucky_progress = [e for e in run.events_of("progress") if e["stage"] == "lucky"]
    assert all("pass 2" in e.get("message", "") for e in lucky_progress)

    # different selection -> different result than the original run
    a = np.load(synthetic_runs.first.run_dir / "final.npy")
    b = np.load(run.run_dir / "final.npy")
    assert not np.array_equal(a, b)
