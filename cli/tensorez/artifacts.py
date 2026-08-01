"""Writing image artifacts: 8-bit sRGB previews, 16-bit linear TIFFs, npy.

Pipeline pixels are float32 linear light NCHW.  Previews get the sRGB
transfer curve (display conversion happens here and only here); TIFF and
npy stay linear.  Single-channel images are broadcast to three channels for
previews so they display neutrally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

from .color import linear_to_srgb


def _to_hwc(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        image = image.squeeze(0)
    return image.permute(1, 2, 0)


def _display_channels(image: torch.Tensor) -> torch.Tensor:
    """4-channel superpixel (R, G1, G2, B) images collapse to RGB for the
    displayable formats (preview PNG, TIFF); the .npy keeps all 4 exact."""
    if image.shape[-3] == 4:
        r, g1, g2, b = image.unbind(dim=-3)
        image = torch.stack([r, (g1 + g2) / 2, b], dim=-3)
    return image


def write_preview_png(path: str | Path, image_nchw: torch.Tensor, normalize: bool = False) -> tuple[int, int]:
    """8-bit sRGB preview.  Returns (width, height).

    ``normalize`` scales the max to 1.0 first — used for debug maps
    (luckiness, weights) whose absolute scale is meaningless.
    """
    image = _display_channels(image_nchw.detach().float())
    if normalize:
        peak = image.max()
        if peak > 0:
            image = image / peak
    image = image.clamp(0.0, 1.0)
    image = linear_to_srgb(image)
    hwc = _to_hwc(image)
    if hwc.shape[-1] == 1:
        hwc = hwc.expand(-1, -1, 3)
    arr = (hwc.numpy() * 255.0).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return arr.shape[1], arr.shape[0]


def write_tiff16(path: str | Path, image_nchw: torch.Tensor) -> tuple[int, int]:
    """16-bit linear-light TIFF (the deliverable).  Returns (width, height)."""
    hwc = _to_hwc(_display_channels(image_nchw.detach().float()).clamp(0.0, 1.0))
    arr = (hwc.numpy() * 65535.0 + 0.5).astype(np.uint16)
    if arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, arr)
    return arr.shape[1], arr.shape[0]


def write_npy(path: str | Path, image_nchw: torch.Tensor) -> None:
    """Exact float32 result, saved HWC (matching truth arrays and stills)."""
    hwc = _to_hwc(image_nchw.detach().float()).contiguous().numpy().astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, hwc)
