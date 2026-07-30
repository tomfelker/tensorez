"""Multi-frame blind deconvolution of the selected lucky frames (torchmfbd).

Instead of deconvolving the stacked result, we hand the top-N *individual*
aligned frames (chosen by their pass-1 luckiness scores) to torchmfbd
(Asensio Ramos, arXiv:2505.10639): a MAP multi-frame blind deconvolution
that models each frame's PSF as a diffraction PSF distorted by a wavefront
expanded in Zernike or Karhunen-Loeve modes over the telescope pupil, and
optimizes the mode coefficients so the closed-form (Wiener-like) common
object explains all frames at once.  Color images are handled as one
torchmfbd "object" per channel, each with its own wavelength (so each
channel gets a physically scaled pupil basis), solved jointly in one call.

Unit conversions (torchmfbd is solar-heritage and uses CGS-ish units):

* telescope diameter / central obscuration: **cm** (ours: cm, passed through)
* wavelength: **Angstrom** (ours: nm, x10)
* pixel scale: **arcsec/pixel** (passed through)

torchmfbd requires the pixel scale to oversample the diffraction limit:
``psf_scale = 206265 * lambda / (D * pix) >= 1``; below ~2 the data is
undersampled (coarser than Nyquist) and we emit a warning.

Practical adaptations for planetary (non-solar) targets, found empirically:

* ``apodization_border`` defaults to 0: torchmfbd's apodization subtracts
  the frame mean, tapers the borders, and adds the mean back — fine for
  solar granulation but it paints a planet's dark-sky borders gray.  Our
  aligned crops already have quiet borders.
* the Fourier ``frequency_cutoff`` of the reconstruction filter defaults to
  a conservative (0.2, 0.3) of the diffraction limit — amateur planetary
  data rarely holds signal beyond that, and higher cutoffs ring hard.

Frames are normalized per-frame by their spatial mean before the solve
(torchmfbd convention) and the object is rescaled back to linear light.

Implementation note: torchmfbd tracks its per-iteration loss only in a tqdm
progress bar (its ``self.loss`` attribute is never filled), so we substitute
a tqdm stand-in that captures the postfix values — giving us both progress
callbacks and an honest loss history without touching the library.
"""

from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .recipe import DeconvConfig

ProgressCallback = Callable[[int, int, float | None], None]
LogCallback = Callable[[str, str], None]


@dataclass
class DeconvResult:
    object_nchw: torch.Tensor        # (1, C, H, W) linear light
    psfs: torch.Tensor               # (C, n_frames, H, W), fftshifted (centered)
    loss_history: np.ndarray         # (iterations,) total loss per iteration


def overfill_factor(wavelength_nm: float, diameter_cm: float, pixel_scale_arcsec: float) -> float:
    """torchmfbd's psf_scale: how much finer than lambda/D one pixel is."""
    wavelength_cm = wavelength_nm * 1e-7
    return 206265.0 * wavelength_cm / (diameter_cm * pixel_scale_arcsec)


def check_optics(cfg: DeconvConfig, channels: int, log: LogCallback) -> None:
    """Validate the physical setup; raises ValueError on impossible configs."""
    if len(cfg.wavelengths_nm) != channels:
        raise ValueError(
            f"[deconv] wavelengths_nm has {len(cfg.wavelengths_nm)} entries but the "
            f"image has {channels} channel(s) — provide one wavelength per channel"
        )
    for w in cfg.wavelengths_nm:
        overfill = overfill_factor(w, cfg.diameter_cm, cfg.pixel_scale_arcsec)
        diffraction_arcsec = 206265.0 * w * 1e-7 / cfg.diameter_cm
        if overfill < 1.0:
            max_pix = 206265.0 * w * 1e-7 / cfg.diameter_cm
            raise ValueError(
                f"[deconv] pixel_scale_arcsec={cfg.pixel_scale_arcsec} is too coarse to model "
                f"a {cfg.diameter_cm} cm aperture at {w} nm (lambda/D = {diffraction_arcsec:.3f}\"); "
                f"needs <= {max_pix:.3f} arcsec/pixel"
            )
        if overfill < 2.0:
            log(
                f"deconv: {w} nm is undersampled at {cfg.pixel_scale_arcsec}\"/pix "
                f"(lambda/D = {diffraction_arcsec:.3f}\" spans only {overfill:.2f} px; "
                f"< 2 px is below Nyquist)",
                "warning",
            )


def _build_config(cfg: DeconvConfig, n_pixel: int) -> dict:
    config: dict = {
        "telescope": {
            "diameter": cfg.diameter_cm,
            "central_obscuration": cfg.central_obscuration_cm,
            "spider": 0,
        },
        "images": {
            "n_pixel": n_pixel,
            "pix_size": cfg.pixel_scale_arcsec,
            "apodization_border": cfg.apodization_border,
            "remove_gradient_apodization": False,
        },
        "optimization": {
            "gpu": -1,  # CPU
            "transform": "none",
            "lr_obj": cfg.lr_obj,
            "lr_modes": cfg.lr_modes,
        },
        "initialization": {"object": "contrast", "modes_std": 0.0},
        "annealing": {"type": "linear", "start_pct": 0.0, "end_pct": 0.85},
        "regularization": {"iuwt1": {"variable": "object", "lambda": 0.0, "nbands": 5}},
        "psf": {"model": cfg.psf_model, "nmax_modes": cfg.n_modes},
    }
    for c, wavelength_nm in enumerate(cfg.wavelengths_nm):
        config[f"object{c + 1}"] = {
            "wavelength": wavelength_nm * 10.0,  # nm -> Angstrom
            "cutoff": list(cfg.frequency_cutoff),
            "image_filter": "tophat",
        }
    return config


class _CapturingTqdm:
    """Stand-in for tqdm inside torchmfbd: captures per-iteration loss from
    set_postfix and forwards progress, instead of drawing a progress bar."""

    callback: ProgressCallback | None = None
    losses: list[float] = []

    def __init__(self, iterable=None, **kwargs):
        self._iterable = iterable

    def __iter__(self):
        for i, item in enumerate(self._iterable):
            self._current = i
            yield item

    def set_postfix(self, ordered_dict=None, **kwargs):
        loss = None
        if ordered_dict and "L" in ordered_dict:
            try:
                loss = float(str(ordered_dict["L"]).strip())
            except ValueError:
                loss = None
        _CapturingTqdm.losses.append(loss if loss is not None else float("nan"))
        if _CapturingTqdm.callback is not None:
            total = len(self._iterable) if hasattr(self._iterable, "__len__") else 0
            _CapturingTqdm.callback(self._current + 1, total, loss)

    # torchmfbd never calls anything else on the bar, but be tolerant:
    def __getattr__(self, name):
        return lambda *a, **k: None


def run_torchmfbd(
    frames_nchw: torch.Tensor,
    cfg: DeconvConfig,
    basis_dir: Path,
    progress: ProgressCallback | None = None,
    log: LogCallback | None = None,
) -> DeconvResult:
    """Jointly deconvolve the selected frames; one torchmfbd object per channel.

    ``frames_nchw``: (N, C, H, W) float32 linear light, H == W.
    ``basis_dir``: directory under which torchmfbd caches its precomputed
    KL/Zernike bases (it writes to ``basis/`` relative to the CWD, so we
    temporarily chdir here — keyed by model/aperture/npix/wavelength/modes,
    the bases are reusable across runs).
    """
    if log is None:
        log = lambda message, level="info": None

    n, channels, h, w = frames_nchw.shape
    if h != w:
        raise ValueError(f"[deconv] requires a square crop; got {w}x{h} — set [align] crop")

    check_optics(cfg, channels, log)

    # torchmfbd is chatty on import and construction (loggers + NVML probe);
    # keep our stdout JSONL clean and route nothing to the console.
    logging.getLogger("deconvolution ").disabled = True
    logging.getLogger("modes").disabled = True

    import torchmfbd.deconvolution as tmfbd_deconvolution

    original_tqdm = tmfbd_deconvolution.tqdm
    _CapturingTqdm.callback = progress
    _CapturingTqdm.losses = []
    tmfbd_deconvolution.tqdm = _CapturingTqdm

    basis_dir.mkdir(parents=True, exist_ok=True)
    try:
        with contextlib.chdir(basis_dir), contextlib.redirect_stderr(io.StringIO()):
            decon = tmfbd_deconvolution.Deconvolution(_build_config(cfg, n_pixel=h))
            decon.logger.disabled = True

            # Per-frame normalization by spatial mean (torchmfbd convention);
            # remember the scale to restore linear light afterwards.
            scales = []
            for c in range(channels):
                fr = frames_nchw[:, c].unsqueeze(0)  # (1, N, H, W)
                mean = fr.mean(dim=(-1, -2), keepdim=True).clamp(min=1e-12)
                scales.append(float(mean.mean()))
                decon.add_frames(fr / mean, id_object=c)

            decon.deconvolve(
                infer_object=False,
                optimizer=cfg.optimizer,
                simultaneous_sequences=1,
                n_iterations=cfg.iterations,
            )

            obj = torch.stack(
                [decon.obj[c][0].detach().cpu() * scales[c] for c in range(channels)], dim=0
            ).unsqueeze(0)  # (1, C, H, W)

            # Recompute the per-frame PSFs from the fitted modes (torchmfbd
            # discards them to save memory).  Tip/tilt is referenced to the
            # first frame, exactly as in its final internal evaluation.
            modes = decon.modes.clone()
            modes[:, :, 0:2] -= modes[:, 0:1, 0:2]
            psf_list, _ = decon.compute_psfs(
                modes, [d.to(decon.device) for d in decon.diversity]
            )
            psfs = torch.stack(
                [torch.fft.fftshift(psf_list[c][0].detach().cpu(), dim=(-2, -1))
                 for c in range(channels)],
                dim=0,
            )  # (C, N, H, W), centered
    finally:
        tmfbd_deconvolution.tqdm = original_tqdm
        _CapturingTqdm.callback = None

    return DeconvResult(
        object_nchw=obj.float(),
        psfs=psfs.float(),
        loss_history=np.asarray(_CapturingTqdm.losses, dtype=np.float32),
    )


def psf_examples_image(psfs: torch.Tensor, max_frames: int = 4, tile: int = 64) -> torch.Tensor:
    """Tile the first few per-frame PSFs into one preview image.

    Returns (1, C, tile, tile * k): each channel plane holds its own PSF
    (chromatic differences show as color fringing), frames tiled left to
    right, each tile normalized to its own peak (PSF absolute scale is
    meaningless for display).
    """
    channels, n, h, w = psfs.shape
    k = min(max_frames, n)
    tile = min(tile, h)
    y0, x0 = (h - tile) // 2, (w - tile) // 2
    tiles = []
    for f in range(k):
        patch = psfs[:, f, y0 : y0 + tile, x0 : x0 + tile].clone()
        peak = patch.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-20)
        tiles.append(patch / peak)
    return torch.cat(tiles, dim=-1).unsqueeze(0)
