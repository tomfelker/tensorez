"""Lucky Fourier: per-frequency lucky stacking (Fourier amplitude selection).

Classic lucky imaging keeps the best whole frames; ``local_lucky`` keeps the
best pixels.  This producer keeps the best *spatial frequencies*: any given
moment of seeing transmits some (u, v) cells of the object's spectrum nearly
undistorted while scrambling others, and which cells are lucky changes from
frame to frame — so selecting per frequency recovers detail no single frame
(and no whole-frame selection) contains.  The idea is Garrel, Guyon & Baudoz
2012 ("A highly efficient lucky imaging algorithm: image synthesis based on
Fourier amplitude selection"), here with the same smooth sigmoid gate the
``local_lucky`` stage uses instead of their hard top-percentage cut.

Two passes, mirroring ``local_lucky``:

* Pass 1 walks the frames taking each one's ``rfft2`` and accumulating the
  per-(channel, u, v) mean and standard deviation of the spectrum magnitude
  (Welford, cached).  This pass has **no tuning parameters** — the statistics
  depend only on the aligned frames — so the cache survives any gate tweak.

* Pass 2 turns each frame's magnitude into a z-score against those
  statistics, gates it through ``sigmoid((z - stdevs_above_mean) *
  steepness)``, and accumulates the weighted *complex* spectrum:

      F_out(u, v) = sum_i w_i(u, v) * F_i(u, v) / sum_i w_i(u, v)

Averaging complex values (not magnitudes) is the crux: a frame whose
magnitude at a frequency is unusually high is one where the atmosphere
briefly presented that frequency with high contrast and coherent phase, so
the lucky frames' phases agree and reinforce, while a plain average over all
frames lets the scrambled phases of unlucky frames cancel the signal.  The
result image is ``irfft2`` of the gated spectrum.

``channel_crosstalk`` has the same meaning as in ``local_lucky``: 0 gates
each channel independently, 1 gates every channel by the least lucky one
(suppresses chromatic shimmer at the cost of throughput).

``subpixel_align`` fixes what the integer center-of-mass alignment cannot:
the sub-pixel residual.  A residual shift of d corrupts a frame's phase by
exactly 2*pi*f*d — linear in frequency, 90 degrees at Nyquist for half a
pixel — which decoheres precisely the complex average pass 2 exists to
take (magnitudes, and hence pass 1 and the gate, are shift-invariant and
don't care).  The fix stays entirely in frequency space: the circular
("toroidal") centroid of a frame is encoded in the phase of its two
first-harmonic bins, so each frame's spectrum is multiplied by the ramp
that puts that centroid on the center pixel.  Phasors rotate in place;
no bin mixes with any other; nothing is ever resampled back to image
space.  With ``per_channel``, each channel is centered independently,
taking atmospheric dispersion's sub-pixel component out too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .luckiness import apply_channel_crosstalk


@dataclass(frozen=True)
class LuckyFourierParams:
    stdevs_above_mean: float = 2.5
    steepness: float = 3.0
    channel_crosstalk: float = 0.0


def spectrum(image_nchw: torch.Tensor) -> torch.Tensor:
    """The frame's half-plane spectrum, complex, shape (N, C, H, W//2 + 1)."""
    return torch.fft.rfft2(image_nchw)


def gate_weight(
    magnitude: torch.Tensor,
    mag_mean: torch.Tensor,
    mag_stdev: torch.Tensor,
    params: LuckyFourierParams,
) -> torch.Tensor:
    """Per-(u, v) weight: sigmoid of the magnitude's z-score."""
    z = torch.where(
        mag_stdev > 0,
        (magnitude - mag_mean) / mag_stdev,
        torch.zeros_like(magnitude),
    )
    weight = torch.sigmoid((z - params.stdevs_above_mean) * params.steepness)
    return apply_channel_crosstalk(weight, params.channel_crosstalk)


def toroidal_centroid(spec: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Per-channel circular centroid (ybar, xbar) in pixel indices, read from
    the phase of the two first-harmonic bins: for a mass at x0,
    angle(F(0, 1)) = -2*pi*x0/W, and unlike a linear center of mass this is
    well-defined on the torus that torch.roll actually lives on.

    Returns shape (N, C, 2), values in [0, size)."""
    xbar = -spec[..., 0, 1].angle() * width / (2 * math.pi) % width
    ybar = -spec[..., 1, 0].angle() * height / (2 * math.pi) % height
    return torch.stack([ybar, xbar], dim=-1)


def centroid_align_ramp(
    spec: torch.Tensor, height: int, width: int, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift the frame (in frequency space, exactly) so its toroidal centroid
    lands on the center pixel — index (size - 1) // 2, the floor of the true
    center, so the target is a pixel and not a pixel boundary.

    ``per_channel`` aligns each channel to its own centroid (sub-pixel
    atmospheric dispersion correction); otherwise one shift is derived from
    the channel-summed brightness — summing the first-harmonic bins over
    channels IS the channel-summed image's bins — and applied to all.

    Returns (aligned spectrum, applied (dy, dx) shifts of shape (N, C, 2) or
    (N, 1, 2)); the shifts are wrapped to the nearest representative, so
    after integer CoM alignment they are sub-pixel-scale."""
    source = spec if per_channel else spec.sum(dim=1, keepdim=True)
    centroid = toroidal_centroid(source, height, width)
    target = torch.tensor(
        [(height - 1) // 2, (width - 1) // 2], dtype=centroid.dtype
    )
    sizes = torch.tensor([height, width], dtype=centroid.dtype)
    delta = torch.remainder(target - centroid + sizes / 2, sizes) - sizes / 2

    fy = torch.from_numpy(np.fft.fftfreq(height).astype(np.float32)).view(-1, 1)
    fx = torch.from_numpy(np.fft.rfftfreq(width).astype(np.float32)).view(1, -1)
    dy = delta[..., 0].view(*delta.shape[:-1], 1, 1)
    dx = delta[..., 1].view(*delta.shape[:-1], 1, 1)
    # shifting content by +d multiplies the spectrum by exp(-2*pi*i*f*d)
    ramp = torch.exp(-2j * math.pi * (fy * dy + fx * dx))
    return spec * ramp, delta


def spectrum_preview(magnitude: torch.Tensor) -> torch.Tensor:
    """Make a half-plane spectrum map displayable: log-scale it (spectra span
    many decades) and fftshift vertically so DC sits at the left edge's
    center instead of splitting across the top and bottom corners."""
    return torch.fft.fftshift(torch.log1p(magnitude), dim=-2)
