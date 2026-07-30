"""Bilinear demosaic of Bayer mosaics, and Bayer sample masks.

A Bayer sensor records one color per photosite in a repeating 2x2 tile.
Bilinear demosaic reconstructs the missing colors by averaging the nearest
recorded neighbors of each color.  Implemented as: split the mosaic into
three sparse per-channel images (mosaic * channel mask, zero elsewhere) and
convolve each with the classic interpolation kernel:

    R, B:  [[1/4, 1/2, 1/4],          G:  [[  0, 1/4,   0],
            [1/2,  1 , 1/2],               [1/4,  1 , 1/4],
            [1/4, 1/2, 1/4]]               [  0, 1/4,   0]]

At a photosite of the channel's own color the kernel just returns the
recorded value; elsewhere it averages the 2 or 4 nearest samples.  This is
numerically identical to the reference TF implementation's per-tile-phase
kernels, but maps directly onto a single conv2d per channel.

The pipeline's lucky-stacking pass additionally uses the *mask* (which
photosites truly sampled each channel) as a per-channel confidence weight, so
interpolated pixels don't dilute the stack — see ``bayer_mask``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .ser import ColorId

# 2x2 tile of channel indices (0=R, 1=G, 2=B), tile[y][x].
_TILES: dict[ColorId, list[list[int]]] = {
    ColorId.BAYER_RGGB: [[0, 1], [1, 2]],
    ColorId.BAYER_GRBG: [[1, 0], [2, 1]],
}

_KERNEL_RB = torch.tensor(
    [[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], dtype=torch.float32
)
_KERNEL_G = torch.tensor(
    [[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]], dtype=torch.float32
)


def is_supported_bayer(color_id: int) -> bool:
    return ColorId(color_id) in _TILES


def bayer_mask(color_id: int, height: int, width: int) -> torch.Tensor:
    """(1, 3, H, W) float mask: 1 where the sensor truly sampled that channel."""
    tile = _TILES[ColorId(color_id)]
    if height % 2 or width % 2:
        raise ValueError("Bayer images must have even dimensions")
    mask = torch.zeros(1, 3, height, width, dtype=torch.float32)
    for ty in range(2):
        for tx in range(2):
            mask[0, tile[ty][tx], ty::2, tx::2] = 1.0
    return mask


def demosaic(mosaic_hw: torch.Tensor, color_id: int) -> torch.Tensor:
    """Bilinear-demosaic a (1, 1, H, W) mosaic into (1, 3, H, W) RGB."""
    _, _, h, w = mosaic_hw.shape
    mask = bayer_mask(color_id, h, w)
    sparse = mosaic_hw * mask  # (1, 3, H, W), zeros where not sampled

    # One grouped conv: per-channel kernel, reflect padding (correct for the
    # 2x2 tile period at the borders).
    kernels = torch.stack([_KERNEL_RB, _KERNEL_G, _KERNEL_RB]).unsqueeze(1)  # (3,1,3,3)
    padded = F.pad(sparse, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernels, groups=3)
