"""The fixed pipeline: lights -> darks -> align -> lucky -> output.

Streaming design: no stage ever holds more than a handful of frames in
memory.  The lucky stage is the two-pass scheme from the reference
implementation's ``local_lucky.py``:

* Pass 1 walks all frames computing each frame's per-pixel luckiness and
  feeds it into a Welford accumulator, yielding the per-pixel mean and
  standard deviation of luckiness over time.
* Pass 2 walks the frames again, recomputes each frame's luckiness, converts
  it to a z-score against the pass-1 statistics, gates it through a sigmoid
  (``weight = sigmoid((z - stdevs_above_mean) * steepness)``), and
  accumulates ``sum(weight * frame) / sum(weight)`` — "the average of all
  pixels more than N standard deviations luckier than the mean".

Pass-1 statistics (and the aligned unweighted average, which the luckiness
metric needs as its "known" reference) are cached, so tweaking only the
selection knobs reruns pass 2 alone.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .align import (
    apply_crop,
    apply_frame_shift,
    compute_com_shift,
    compute_com_shift_per_channel,
)
from .artifacts import write_npy, write_preview_png, write_tiff16
from .bayer import bayer_mask
from .cache import CacheEntry
from .events import EventEmitter
from .luckiness import FrequencyBands, FrequencyBandsParams
from .observation import AlignParams, Observation
from .recipe import Recipe
from .sequence import ImageSequence
from .welford import Welford


class PipelineError(RuntimeError):
    def __init__(self, message: str, stage: str | None = None):
        super().__init__(message)
        self.stage = stage


def _utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


class Pipeline:
    def __init__(self, recipe: Recipe, cache_dir: Path, emitter: EventEmitter):
        self.recipe = recipe
        self.cache_dir = Path(cache_dir)
        self.emitter = emitter
        self.run_dir: Path | None = None
        self.stage_summaries: list[dict[str, Any]] = []
        self.artifact_records: list[dict[str, Any]] = []
        self.current_stage: str | None = None

    # -- plumbing -----------------------------------------------------------

    @contextmanager
    def stage(self, name: str, cached: bool, **extra: Any):
        self.current_stage = name
        self.emitter.emit("stage_start", stage=name, cached=cached, **extra)
        t0 = time.monotonic()
        yield
        seconds = round(time.monotonic() - t0, 3)
        self.emitter.emit("stage_end", stage=name, seconds=seconds)
        self.stage_summaries.append({"name": name, "cached": cached, "seconds": seconds})
        self.current_stage = None

    def artifact(
        self,
        stage: str,
        name: str,
        kind: str,
        rel_path: str,
        width: int | None = None,
        height: int | None = None,
        frame: int | None = None,
    ) -> None:
        record: dict[str, Any] = {"stage": stage, "name": name, "kind": kind, "path": rel_path}
        if width is not None:
            record["width"], record["height"] = width, height
        if frame is not None:
            record["frame"] = frame
        self.artifact_records.append(record)
        self.emitter.emit("artifact", **record)

    def preview(self, stage: str, name: str, image: torch.Tensor, normalize: bool = False,
                frame: int | None = None) -> None:
        assert self.run_dir is not None
        rel = f"stages/{stage}/{name}.png"
        w, h = write_preview_png(self.run_dir / rel, image, normalize=normalize)
        kind = "sequence_frame" if frame is not None else "preview"
        self.artifact(stage, name, kind, rel, width=w, height=h, frame=frame)

    # -- cache keys ---------------------------------------------------------

    def _darks_key(self, darks_seq: ImageSequence) -> str:
        return "stage: darks\n" + darks_seq.identity()

    def _align_key(self, lights_seq: ImageSequence, darks_entry: CacheEntry | None) -> str:
        key = "stage: align\nlights:\n" + lights_seq.identity()
        key += f"darks_key: {darks_entry.key_hash if darks_entry else None}\n"
        key += self._align_params().identity()
        return key

    def _lucky_stats_key(self, align_entry: CacheEntry) -> str:
        return (
            "stage: lucky_stats\n"
            f"align_key: {align_entry.key_hash}\n" + self._lucky_params().identity()
            + "stats: mean, stdev, frame_scores (v2)\n"
        )

    def _align_params(self) -> AlignParams:
        a = self.recipe.align
        return AlignParams(
            center_of_mass=a.center_of_mass,
            per_channel=a.per_channel,
            crop=a.crop,
            crop_align=a.crop_align,
            crop_offsets=a.crop_offsets,
        )

    def _lucky_params(self) -> FrequencyBandsParams:
        l = self.recipe.lucky
        return FrequencyBandsParams(
            noise_wavelength_pixels=l.noise_wavelength_pixels,
            crossover_wavelength_pixels=l.crossover_wavelength_pixels,
            isoplanatic_patch_pixels=l.isoplanatic_patch_pixels,
            channel_crosstalk=l.channel_crosstalk,
        )

    def _sequences(self) -> tuple[ImageSequence, ImageSequence | None]:
        r = self.recipe
        try:
            lights = ImageSequence(
                list(r.lights.paths),
                start_frame=r.lights.start_frame,
                frame_step=r.lights.frame_step,
                end_frame=r.lights.end_frame,
                debayer=r.lights.debayer,
            )
        except (FileNotFoundError, ValueError) as e:
            raise PipelineError(str(e), stage="lights")
        darks = None
        if r.darks is not None:
            try:
                # Darks come from the same sensor, so the same debayer mode
                # keeps their geometry and channels matching the lights.
                darks = ImageSequence(list(r.darks.paths), debayer=r.lights.debayer)
            except (FileNotFoundError, ValueError) as e:
                raise PipelineError(str(e), stage="darks")
        return lights, darks

    # -- validate -----------------------------------------------------------

    def validate(self) -> dict[str, Any]:
        """Parse + resolve + report per-stage cache hit/miss; no work."""
        lights, darks = self._sequences()
        stages = []
        darks_entry = None
        if darks is not None:
            darks_entry = CacheEntry(self.cache_dir, "darks", self._darks_key(darks))
            stages.append({"stage": "darks", "cached": darks_entry.complete})
        align_entry = CacheEntry(self.cache_dir, "align", self._align_key(lights, darks_entry))
        stages.append({"stage": "align", "cached": align_entry.complete})
        stats_entry = CacheEntry(self.cache_dir, "lucky_stats", self._lucky_stats_key(align_entry))
        stages.append({"stage": "lucky_stats", "cached": stats_entry.complete})
        return {
            "recipe": self.recipe.resolved_dict(),
            "frame_count": len(lights),
            "stages": stages,
        }

    # -- run ----------------------------------------------------------------

    def run(self) -> Path:
        recipe = self.recipe
        run_started_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()
        lights_seq, darks_seq = self._sequences()

        # Run directory: <output.dir>/<name>/<UTC timestamp>/
        base = Path(recipe.output.dir) / recipe.name
        run_dir = base / _utc_timestamp()
        suffix = 1
        while run_dir.exists():
            suffix += 1
            run_dir = base / f"{_utc_timestamp()}-{suffix}"
        run_dir.mkdir(parents=True)
        self.run_dir = run_dir
        (run_dir / "stages").mkdir()
        shutil.copyfile(recipe.path, run_dir / "recipe.toml")
        self.emitter.open_log(run_dir / "log.txt")

        self.emitter.emit(
            "run_start",
            recipe_path=str(recipe.path),
            recipe=recipe.resolved_dict(),
            run_dir=str(run_dir),
            frame_count=len(lights_seq),
        )

        with self.stage("lights", cached=False):
            layout = lights_seq.color_id.name
            if lights_seq.is_bayer:
                layout += f", debayer {lights_seq.debayer}"
            self.emitter.log(
                f"lights: {len(lights_seq.files)} file(s), {len(lights_seq)} frame(s), "
                f"color layout {layout}"
            )

        dark_mean, dark_variance = self._run_darks(darks_seq)

        obs = Observation(
            lights_seq,
            self._align_params(),
            dark_mean=dark_mean,
            dark_variance=dark_variance,
        )
        darks_entry = (
            CacheEntry(self.cache_dir, "darks", self._darks_key(darks_seq))
            if darks_seq is not None else None
        )
        average_image, align_entry = self._run_align(obs, darks_entry)

        result, frame_scores = self._run_lucky(obs, average_image, align_entry)

        if recipe.deconv is not None:
            result = self._run_deconv(obs, frame_scores)

        with self.stage("output", cached=False):
            write_npy(self.run_dir / "final.npy", result)
            self.artifact("output", "final", "array", "final.npy")
            w, h = write_tiff16(self.run_dir / "final.tif", result)
            self.artifact("output", "final", "image", "final.tif", width=w, height=h)
            w, h = write_preview_png(self.run_dir / "final_preview.png", result)
            self.artifact("output", "final_preview", "preview", "final_preview.png",
                          width=w, height=h)

        manifest = {
            "manifest_version": 0,
            "recipe": recipe.resolved_dict(),
            "run": {
                "started_utc": run_started_utc,
                "seconds": round(self.emitter.now(), 3),
                "frame_count": len(lights_seq),
            },
            "stages": self.stage_summaries,
            "artifacts": self.artifact_records,
        }
        tmp = run_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, run_dir / "manifest.json")

        self.emitter.emit("done", seconds=round(self.emitter.now(), 3), final="final.tif")
        return run_dir

    # -- darks --------------------------------------------------------------

    def _run_darks(self, darks_seq: ImageSequence | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Master dark: streaming Welford mean + per-pixel variance."""
        if darks_seq is None:
            return None, None
        entry = CacheEntry(self.cache_dir, "darks", self._darks_key(darks_seq))
        if entry.complete:
            with self.stage("darks", cached=True):
                pass
            data = entry.load_npz("dark")
            return torch.from_numpy(data["mean"]), torch.from_numpy(data["variance"])

        with self.stage("darks", cached=False):
            welford = Welford()
            total = len(darks_seq)
            for i in range(total):
                welford.update(darks_seq.read_frame(i))
                self.emitter.progress("darks", i + 1, total, message="averaging darks")
            mean, variance = welford.mean, welford.variance
            entry.save_npz("dark", mean=mean.numpy(), variance=variance.numpy())
            entry.mark_complete()
            self.preview("darks", "dark_mean", mean, normalize=True)
        return mean, variance

    # -- align --------------------------------------------------------------

    def _run_align(
        self, obs: Observation, darks_entry: CacheEntry | None
    ) -> tuple[torch.Tensor, CacheEntry]:
        """Per-frame CoM shifts + crop rectangle + aligned unweighted average.

        The average is computed here (not in the lucky stage) because it
        depends only on calibration + geometry, and the luckiness metric
        needs it as its "known" reference before pass 1 can start.
        """
        entry = CacheEntry(self.cache_dir, "align", self._align_key(obs.lights, darks_entry))
        if entry.complete:
            with self.stage("align", cached=True):
                pass
            data = entry.load_npz("align")
            obs.shifts = data["shifts"]  # (N, 2) or (N, C, 2)
            rect = data["rect"]
            obs._rect = tuple(int(v) for v in rect) if rect.size else None
            obs._rect_known = True
            return torch.from_numpy(data["average"]), entry

        with self.stage("align", cached=False):
            p = obs.align_params
            total = len(obs)
            per_channel = p.per_channel and p.center_of_mass
            if per_channel and obs.lights.read_frame(0).shape[1] == 1:
                self.emitter.log(
                    "align: per_channel requested but the lights are mono — "
                    "there is only one channel, so this is a no-op"
                )
                per_channel = False
            shifts: list = []
            average = Welford()
            for i in range(total):
                # Read the calibrated frame once; derive both the shift and
                # the running average from it.
                image = obs.calibrated(i)
                if per_channel:
                    shift = compute_com_shift_per_channel(image)
                elif p.center_of_mass:
                    shift = compute_com_shift(image)
                else:
                    shift = (0, 0)
                shifts.append(shift)
                cooked = apply_crop(apply_frame_shift(image, shift), obs.rect_for(image))
                average.update(cooked)
                self.emitter.progress("align", i + 1, total, message="center-of-mass align")
            obs.shifts = np.asarray(shifts, dtype=np.int32)
            if per_channel:
                # Dispersion signature: how far each channel sits from the
                # channel-mean pointing, averaged over the capture.
                deltas = obs.shifts - obs.shifts.mean(axis=1, keepdims=True)  # (N, C, 2)
                mean_delta = deltas.mean(axis=0)
                msg = ", ".join(
                    f"ch{c}: (dy={mean_delta[c][0]:+.2f}, dx={mean_delta[c][1]:+.2f})"
                    for c in range(mean_delta.shape[0])
                )
                self.emitter.log(f"align: mean per-channel CoM offset vs channel mean: {msg}")
            rect = obs._rect
            entry.save_npz(
                "align",
                shifts=obs.shifts,
                rect=np.asarray(rect if rect is not None else [], dtype=np.int32),
                average=average.mean.numpy(),
            )
            entry.mark_complete()
        return average.mean, entry

    # -- lucky --------------------------------------------------------------

    def _run_lucky(
        self, obs: Observation, average_image: torch.Tensor, align_entry: CacheEntry
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Two-pass lucky stacking.  Returns (stacked image, per-frame scores).

        The scalar score of frame *i* is the spatial mean of its luckiness
        map — a ranking of whole frames that the optional deconv stage uses
        to pick its top-N input frames.  Scores are cached with the pass-1
        statistics and published as ``frame_scores.npy``.
        """
        recipe = self.recipe
        debug_frames = recipe.output.debug_frames
        stats_entry = CacheEntry(self.cache_dir, "lucky_stats", self._lucky_stats_key(align_entry))
        pass1_cached = stats_entry.complete

        h, w = average_image.shape[-2], average_image.shape[-1]
        algo = FrequencyBands(h, w, self._lucky_params(), average_image)
        total = len(obs)

        with self.stage("lucky", cached=False, pass1_cached=pass1_cached):
            self.preview("lucky", "unweighted_average", average_image)
            rel = "stages/lucky/unweighted_average.npy"
            write_npy(self.run_dir / rel, average_image)
            self.artifact("lucky", "unweighted_average", "array", rel)

            if pass1_cached:
                self.emitter.log("lucky: pass 1 statistics loaded from cache")
                stats = stats_entry.load_npz("stats")
                luck_mean = torch.from_numpy(stats["mean"])
                luck_stdev = torch.from_numpy(stats["stdev"])
                frame_scores = stats["frame_scores"]
            else:
                welford = Welford()
                scores: list[float] = []
                for i in range(total):
                    image, dark_variance = obs.read_cooked(i)
                    luckiness = algo.compute(image, dark_variance)
                    welford.update(luckiness)
                    scores.append(float(luckiness.mean()))
                    if i < debug_frames:
                        self.preview("lucky", f"luckiness_{i:08d}", luckiness,
                                     normalize=True, frame=i)
                    self.emitter.progress("lucky", i + 1, total, message="pass 1/2: luckiness statistics")
                luck_mean, luck_stdev = welford.mean, welford.stdev
                frame_scores = np.asarray(scores, dtype=np.float32)
                stats_entry.save_npz(
                    "stats", mean=luck_mean.numpy(), stdev=luck_stdev.numpy(),
                    frame_scores=frame_scores,
                )
                stats_entry.mark_complete()

            rel = "stages/lucky/frame_scores.npy"
            (self.run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(self.run_dir / rel, frame_scores)
            self.artifact("lucky", "frame_scores", "array", rel)
            best = np.argsort(frame_scores)[::-1][: min(10, total)]
            self.emitter.log(
                "lucky: best frames by mean luckiness: "
                + ", ".join(f"{int(i)} ({frame_scores[i]:.4f})" for i in best)
            )

            self.preview("lucky", "luckiness_mean", luck_mean, normalize=True)
            self.preview("lucky", "luckiness_stdev", luck_stdev, normalize=True)

            # Bilinearly-demosaiced Bayer sources: weight each pixel only by
            # channels the sensor actually sampled there, so interpolated
            # pixels don't dilute the stack.  The mask lives in sensor
            # coordinates and rides each frame's shift + crop, exactly like
            # the dark variance does.  (The superpixel modes have no
            # interpolated pixels, and "none" is mono — no mask needed.)
            base_mask: torch.Tensor | None = None
            if obs.lights.is_bayer and obs.lights.debayer == "bilinear":
                first = obs.lights.read_frame(0)
                base_mask = bayer_mask(obs.lights.color_id, first.shape[-2], first.shape[-1])

            # Pass 2: sigmoid-gated weighted average.
            selection = recipe.lucky
            weighted_sum: torch.Tensor | None = None
            total_weight: torch.Tensor | None = None
            for i in range(total):
                image, dark_variance = obs.read_cooked(i)
                luckiness = algo.compute(image, dark_variance)
                z = torch.where(
                    luck_stdev > 0,
                    (luckiness - luck_mean) / luck_stdev,
                    torch.zeros_like(luckiness),
                )
                weight = torch.sigmoid((z - selection.stdevs_above_mean) * selection.steepness)
                if base_mask is not None:
                    sample_mask = apply_crop(
                        apply_frame_shift(base_mask, obs.shifts[i]), obs.rect_for(base_mask)
                    )
                    weight = weight * sample_mask
                if i < debug_frames:
                    self.preview("lucky", f"weight_{i:08d}", weight, normalize=True, frame=i)
                if weighted_sum is None:
                    weighted_sum = torch.zeros_like(image)
                    total_weight = torch.zeros_like(weight)
                weighted_sum += weight * image
                total_weight += weight
                self.emitter.progress("lucky", i + 1, total, message="pass 2/2: weighted stack")

            assert weighted_sum is not None and total_weight is not None
            result = torch.where(
                total_weight > 0, weighted_sum / total_weight, torch.zeros_like(weighted_sum)
            )
            self.preview("lucky", "total_weight", total_weight, normalize=True)
            avg_frames = float(total_weight.mean())
            self.emitter.log(
                f"lucky: average effective frames per pixel: {avg_frames:.2f} of {total}"
            )

            if recipe.deconv is not None:
                # Deconvolution will take over final.*; keep the lucky stack
                # available for comparison as named artifacts.
                rel = "stages/lucky/lucky_stack.tif"
                w2, h2 = write_tiff16(self.run_dir / rel, result)
                self.artifact("lucky", "lucky_stack", "image", rel, width=w2, height=h2)
                self.preview("lucky", "lucky_stack", result)
        return result, frame_scores

    # -- deconv -------------------------------------------------------------

    def _run_deconv(self, obs: Observation, frame_scores: np.ndarray) -> torch.Tensor:
        """Multi-frame blind deconvolution of the luckiest aligned frames.

        Selects the top-N frames by their pass-1 luckiness score (or all
        frames), feeds the same aligned/cropped cooked frames the luckiness
        was computed on to torchmfbd, and returns the reconstructed object,
        which becomes the run's final result.  Not cached: it *is* the final
        product (the frame selection comes free from the cached stats).
        """
        from .deconv import psf_examples_image, run_torchmfbd

        cfg = self.recipe.deconv
        assert cfg is not None
        total = len(obs)

        with self.stage("deconv", cached=False):
            if cfg.frames == "all":
                selected = list(range(total))
            else:
                top_n = cfg.top_n
                if top_n > total:
                    self.emitter.log(
                        f"deconv: top_n={top_n} exceeds the {total} available frames; using all",
                        level="warning",
                    )
                    top_n = total
                order = np.argsort(frame_scores)[::-1][:top_n]
                selected = sorted(int(i) for i in order)
            self.emitter.log(f"deconv: torchmfbd on {len(selected)} frame(s): {selected}")

            frames = torch.cat([obs.read_cooked(i)[0] for i in selected], dim=0)

            def on_progress(current: int, iter_total: int, loss: float | None) -> None:
                message = "torchmfbd" if loss is None else f"torchmfbd loss {loss:.6f}"
                self.emitter.progress(
                    "deconv", current, iter_total or cfg.iterations, message=message
                )

            try:
                result = run_torchmfbd(
                    frames,
                    cfg,
                    basis_dir=(self.cache_dir / "torchmfbd").resolve(),
                    progress=on_progress,
                    log=lambda message, level="info": self.emitter.log(message, level=level),
                )
            except ValueError as e:
                raise PipelineError(str(e), stage="deconv")

            rel = "stages/deconv/loss_history.npy"
            (self.run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(self.run_dir / rel, result.loss_history)
            self.artifact("deconv", "loss_history", "array", rel)

            self.preview("deconv", "psf_examples", psf_examples_image(result.psfs))

        return result.object_nchw
