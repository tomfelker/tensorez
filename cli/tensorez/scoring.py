"""Whole-frame lucky scoring: one scalar per frame, higher = luckier.

This is classic (global) lucky imaging: rank entire frames by a sharpness
metric, then let downstream stages stack or deconvolve only the best ones.
Two metrics, both ports of the legacy explorations:

* ``fourier_bandpass`` — mean FFT magnitude within a wavelength band
  (legacy ``explorations_lucky.rate_image`` / ``FourierFocus``).  A frame
  the seeing briefly left sharp has more energy at planetary-detail scales;
  the band excludes both the DC/low frequencies (dominated by the object's
  overall shape, identical in every frame) and the noise floor.

* ``image_squared`` — the mean of the squared image, the classic sharpness
  metric of Muller & Buffington (1974): blurring conserves flux but spreads
  it out, which strictly decreases the sum of squares.

Scores are computed on the calibrated/aligned/cropped frames and cached
alongside the alignment (same inputs, same lifetime).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .frequency import gaussian_bandpass_mask


@dataclass(frozen=True)
class ScoringParams:
    metric: str = "fourier_bandpass"
    min_wavelength_pixels: float = 5.0
    max_wavelength_pixels: float = 50.0

    def identity(self) -> str:
        out = f"metric: {self.metric}\n"
        if self.metric == "fourier_bandpass":
            out += (
                f"min_wavelength_pixels: {self.min_wavelength_pixels!r}\n"
                f"max_wavelength_pixels: {self.max_wavelength_pixels!r}\n"
            )
        return out


class FrameScorer:
    """Scores frames of a fixed (post-crop) geometry."""

    def __init__(self, height: int, width: int, params: ScoringParams):
        self.params = params
        self.mask: torch.Tensor | None = None
        if params.metric == "fourier_bandpass":
            self.mask = gaussian_bandpass_mask(
                height, width, params.min_wavelength_pixels, params.max_wavelength_pixels
            )

    def score(self, image_nchw: torch.Tensor) -> float:
        if self.params.metric == "image_squared":
            return float(image_nchw.square().mean())
        assert self.mask is not None
        magnitude = torch.fft.rfft2(image_nchw).abs()
        return float((magnitude * self.mask).mean())
