"""sRGB <-> linear conversion (IEC 61966-2-1) for torch tensors.

Everything inside the pipeline is linear light; the sRGB transfer curve is
applied only when decoding 8-bit sources and when writing 8-bit previews.
"""

from __future__ import annotations

import torch

SRGB_A = 0.055
SRGB_PHI = 12.92
SRGB_K0 = 0.04045
SRGB_GAMMA = 2.4


def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        image <= SRGB_K0,
        image / SRGB_PHI,
        ((image.clamp(min=0.0) + SRGB_A) / (1 + SRGB_A)) ** SRGB_GAMMA,
    )


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        image <= SRGB_K0 / SRGB_PHI,
        image * SRGB_PHI,
        (1 + SRGB_A) * image.clamp(min=0.0) ** (1 / SRGB_GAMMA) - SRGB_A,
    )
