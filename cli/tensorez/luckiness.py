"""The frequency_bands per-pixel luckiness metric.

Lucky imaging wants the moments when the atmosphere briefly acts like a good
lens.  Per frame and per pixel we ask two questions in frequency space:

* Does this region contain *new information* — energy in the "interesting"
  band, between the noise wavelength and the crossover wavelength — that a
  long-exposure-blurred frame would not have?

    new_info = bandpass_interesting(image)^2

* Does it *disagree* with what we already trust — the low-frequency "known"
  band, between the crossover wavelength and the isoplanatic patch scale —
  compared to the average image?  Disagreement there means the frame is
  distorted, not lucky:

    wrong_info = (bandpass_known(image) - bandpass_known(average))^2

If a master dark's per-pixel variance is available it is subtracted from
``new_info``: energy explained by sensor noise is not information.

Both maps are then smoothed to the isoplanatic patch scale (a Gaussian
lowpass — luckiness is a property of a seeing patch, not of single pixels),
and combined as

    luckiness = sqrt(max(0, new_info_lp / (wrong_info_lp + epsilon)))

Finally ``channel_crosstalk`` lerps each channel's luckiness toward the
minimum across channels (1.0 = a pixel is only as lucky as its worst
channel, which suppresses chromatic shimmer at the cost of throughput).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .frequency import apply_frequency_mask, gaussian_bandpass_mask, gaussian_lowpass_mask


@dataclass(frozen=True)
class FrequencyBandsParams:
    noise_wavelength_pixels: float = 2.0
    crossover_wavelength_pixels: float = 35.0
    isoplanatic_patch_pixels: float = 55.0
    channel_crosstalk: float = 0.0
    epsilon: float = 1.0 / (1 << 16)

    def identity(self) -> str:
        return (
            "algorithm: frequency_bands\n"
            f"noise_wavelength_pixels: {self.noise_wavelength_pixels!r}\n"
            f"crossover_wavelength_pixels: {self.crossover_wavelength_pixels!r}\n"
            f"isoplanatic_patch_pixels: {self.isoplanatic_patch_pixels!r}\n"
            f"channel_crosstalk: {self.channel_crosstalk!r}\n"
        )


class FrequencyBands:
    """Precomputed masks + reference for the frequency_bands luckiness."""

    def __init__(self, height: int, width: int, params: FrequencyBandsParams,
                 average_image_nchw: torch.Tensor):
        p = params
        self.params = p
        self.isoplanatic_mask = gaussian_lowpass_mask(height, width, p.isoplanatic_patch_pixels)
        self.known_mask = gaussian_bandpass_mask(
            height, width, p.crossover_wavelength_pixels, p.isoplanatic_patch_pixels
        )
        self.interesting_mask = gaussian_bandpass_mask(
            height, width, p.noise_wavelength_pixels, p.crossover_wavelength_pixels
        )
        self.average_known = apply_frequency_mask(average_image_nchw, self.known_mask)

    def compute(
        self,
        image_nchw: torch.Tensor,
        dark_variance_nchw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-pixel luckiness map, same shape as the input image."""
        p = self.params
        image_known = apply_frequency_mask(image_nchw, self.known_mask)
        image_interesting = apply_frequency_mask(image_nchw, self.interesting_mask)

        new_info = image_interesting.square()
        wrong_info = (image_known - self.average_known).square()

        if dark_variance_nchw is not None:
            new_info = new_info - dark_variance_nchw

        new_info_lp = apply_frequency_mask(new_info, self.isoplanatic_mask)
        wrong_info_lp = apply_frequency_mask(wrong_info, self.isoplanatic_mask)

        luckiness = torch.sqrt(torch.clamp(new_info_lp / (wrong_info_lp + p.epsilon), min=0.0))
        return apply_channel_crosstalk(luckiness, p.channel_crosstalk)


def apply_channel_crosstalk(image_nchw: torch.Tensor, channel_crosstalk: float) -> torch.Tensor:
    if image_nchw.shape[1] == 1 or channel_crosstalk == 0:
        return image_nchw
    channel_min = image_nchw.amin(dim=1, keepdim=True)
    if channel_crosstalk == 1:
        return channel_min.expand_as(image_nchw)
    return torch.lerp(image_nchw, channel_min.expand_as(image_nchw), channel_crosstalk)
