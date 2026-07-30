"""Gaussian frequency-domain masks and filtering, on NCHW torch tensors.

All filtering is done with ``torch.fft.rfft2`` / ``irfft2`` over the last
two (spatial) dimensions, so masks have shape (1, 1, H, W//2 + 1) and
broadcast over batch and channel.

A "lowpass at wavelength L" mask is a Gaussian in frequency magnitude,
normalized to 1 at DC:

    mask(f) = exp(-0.5 * (f * L)^2),   f in cycles/pixel

i.e. features larger than ~L pixels pass, smaller ones are attenuated.
Bandpass masks are built as (1 - lowpass(long)) * lowpass(short), matching
the reference implementation exactly.
"""

from __future__ import annotations

import numpy as np
import torch


def spatial_frequencies(height: int, width: int) -> torch.Tensor:
    """|f| in cycles/pixel for an rfft2 layout, shape (1, 1, H, W//2 + 1)."""
    fy = torch.from_numpy(np.fft.fftfreq(height).astype(np.float32)).view(-1, 1)
    fx = torch.from_numpy(np.fft.rfftfreq(width).astype(np.float32)).view(1, -1)
    return torch.sqrt(fx * fx + fy * fy).view(1, 1, height, -1)


def gaussian_lowpass_mask(height: int, width: int, cutoff_wavelength_pixels: float) -> torch.Tensor:
    freqs = spatial_frequencies(height, width)
    return torch.exp(-0.5 * (freqs * cutoff_wavelength_pixels) ** 2)


def gaussian_bandpass_mask(
    height: int,
    width: int,
    min_wavelength_pixels: float,
    max_wavelength_pixels: float,
) -> torch.Tensor:
    """Passes wavelengths between min (short/fine) and max (long/coarse)."""
    highpass = 1.0 - gaussian_lowpass_mask(height, width, max_wavelength_pixels)
    return highpass * gaussian_lowpass_mask(height, width, min_wavelength_pixels)


def apply_frequency_mask(image_nchw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    h, w = image_nchw.shape[-2], image_nchw.shape[-1]
    spectrum = torch.fft.rfft2(image_nchw)
    return torch.fft.irfft2(spectrum * mask, s=(h, w))
