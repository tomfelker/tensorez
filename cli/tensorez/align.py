"""Center-of-mass alignment and centered cropping.

Alignment here is deliberately crude and lossless: each frame is translated
by an *integer* pixel shift (a ``torch.roll``, so no resampling and no
information loss) so that its brightness center of mass lands at the image
center.  For a planet floating in dark sky this removes essentially all of
the tip/tilt component of the seeing and any drift, which is all v0 needs
before the per-pixel lucky selection does its work.

The center of mass is computed on the above-average part of the brightness
(background sky pixels shouldn't drag the centroid around), summed across
channels.  Because rolling wraps content around the edges, moving the mass
changes the centroid slightly, so we iterate to convergence (a few steps).

(Bayer mosaics are debayered on read — see bayer.py — so alignment never
needs to worry about preserving a 2x2 mosaic phase.)
"""

from __future__ import annotations

import numpy as np
import torch


def center_of_mass(image_nchw: torch.Tensor) -> tuple[float, float]:
    """(com_y, com_x) in pixels relative to the image center."""
    # Collapse channels; keep only above-average brightness so the sky
    # background doesn't pull the centroid toward the frame center.
    mass = image_nchw.sum(dim=1, keepdim=True)
    mass = (mass - mass.mean()).clamp(min=0.0)

    total = mass.sum()
    if total <= 0:
        return 0.0, 0.0

    h, w = mass.shape[-2], mass.shape[-1]
    ys = torch.linspace(-h / 2.0, h / 2.0, h, dtype=mass.dtype).view(1, 1, h, 1)
    xs = torch.linspace(-w / 2.0, w / 2.0, w, dtype=mass.dtype).view(1, 1, 1, w)
    com_y = float((mass * ys).sum() / total)
    com_x = float((mass * xs).sum() / total)
    return com_y, com_x


def _quantize_shift(value: float) -> int:
    return -int(value)  # truncate toward zero, as the reference did


def compute_com_shift(
    image_nchw: torch.Tensor,
    max_steps: int = 10,
) -> tuple[int, int]:
    """Total (dy, dx) integer roll that centers the image's center of mass.

    Iterates because rolling wraps pixels around the border, slightly moving
    the centroid; converges when the residual shift rounds to zero.
    """
    total_dy, total_dx = 0, 0
    image = image_nchw
    for _ in range(max_steps):
        com_y, com_x = center_of_mass(image)
        dy = _quantize_shift(com_y)
        dx = _quantize_shift(com_x)
        if max(abs(dy), abs(dx)) < 1:
            break
        image = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
        total_dy += dy
        total_dx += dx
    return total_dy, total_dx


def apply_shift(image_nchw: torch.Tensor, shift: tuple[int, int]) -> torch.Tensor:
    dy, dx = shift
    if dy == 0 and dx == 0:
        return image_nchw
    return torch.roll(image_nchw, shifts=(dy, dx), dims=(-2, -1))


def compute_com_shift_per_channel(
    image_nchw: torch.Tensor,
    max_steps: int = 10,
) -> list[tuple[int, int]]:
    """Independent CoM shift per color channel — atmospheric dispersion.

    The atmosphere is a weak prism: red and blue land at slightly different
    altitudes, so each channel's centroid is displaced along the parallactic
    angle.  Centering each channel independently removes the (roughly
    constant) dispersion offset; because the integer shifts dither from
    frame to frame, sub-pixel residuals average out in the stack (and the
    deconv's per-channel tip/tilt modes absorb the rest).
    """
    return [
        compute_com_shift(image_nchw[:, c : c + 1], max_steps=max_steps)
        for c in range(image_nchw.shape[1])
    ]


def apply_frame_shift(image_nchw: torch.Tensor, shift) -> torch.Tensor:
    """Apply a whole-frame shift (2,) or per-channel shifts (C, 2)."""
    arr = np.asarray(shift)
    if arr.ndim == 1:
        return apply_shift(image_nchw, (int(arr[0]), int(arr[1])))
    channels = [
        apply_shift(image_nchw[:, c : c + 1], (int(dy), int(dx)))
        for c, (dy, dx) in enumerate(arr)
    ]
    return torch.cat(channels, dim=1)


def crop_rect(
    height: int,
    width: int,
    crop: tuple[int, int] | None,
    crop_align: int = 2,
    crop_offsets: tuple[int, int] = (0, 0),
) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) of a centered crop, or None for full frame.

    ``crop`` is (w, h); ``crop_offsets`` is (x, y) from the image center.
    The top-left corner is floored to a multiple of ``crop_align``.
    """
    if crop is None:
        return None
    crop_w, crop_h = crop
    if crop_w > width or crop_h > height:
        raise ValueError(f"crop {crop} larger than the {width}x{height} frame")
    x = (width - crop_w) // 2 + crop_offsets[0]
    y = (height - crop_h) // 2 + crop_offsets[1]
    x = (x // crop_align) * crop_align
    y = (y // crop_align) * crop_align
    x = max(0, min(x, width - crop_w))
    y = max(0, min(y, height - crop_h))
    return x, y, crop_w, crop_h


def apply_crop(image_nchw: torch.Tensor, rect: tuple[int, int, int, int] | None) -> torch.Tensor:
    if rect is None:
        return image_nchw
    x, y, w, h = rect
    return image_nchw[..., y : y + h, x : x + w]
