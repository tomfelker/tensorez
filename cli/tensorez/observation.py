"""Observation: calibrated, aligned, cropped access to the light frames.

Ties together the calibration and geometry stages so the lucky-imaging
passes can just ask for "cooked" frame *i*:

1. subtract the master dark (per-pixel mean of the dark frames);
2. roll by that frame's precomputed integer center-of-mass shift;
3. crop to the centered output rectangle.

The master dark's per-pixel *variance* rides along through the same
roll + crop so the luckiness metric can subtract sensor noise where it
actually lands in the aligned frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .align import apply_crop, apply_frame_shift, compute_com_shift, crop_rect
from .sequence import ImageSequence


@dataclass(frozen=True)
class AlignParams:
    center_of_mass: bool = True
    per_channel: bool = False
    crop: tuple[int, int] | None = None
    crop_align: int = 2
    crop_offsets: tuple[int, int] = (0, 0)

    def identity(self) -> str:
        return (
            f"center_of_mass: {self.center_of_mass}\n"
            f"per_channel: {self.per_channel}\n"
            f"crop: {list(self.crop) if self.crop else None}\n"
            f"crop_align: {self.crop_align}\n"
            f"crop_offsets: {list(self.crop_offsets)}\n"
        )


class Observation:
    def __init__(
        self,
        lights: ImageSequence,
        align_params: AlignParams,
        dark_mean: torch.Tensor | None = None,
        dark_variance: torch.Tensor | None = None,
    ):
        self.lights = lights
        self.align_params = align_params
        self.dark_mean = dark_mean
        self.dark_variance = dark_variance
        # Per cooked frame: whole-frame shift (2,) or per-channel (C, 2).
        self.shifts: np.ndarray | None = None
        self._rect: tuple[int, int, int, int] | None = None
        self._rect_known = False

    def __len__(self) -> int:
        return len(self.lights)

    def calibrated(self, index: int) -> torch.Tensor:
        image = self.lights.read_frame(index)
        if self.dark_mean is not None:
            if self.dark_mean.shape[-2:] != image.shape[-2:]:
                raise ValueError(
                    f"dark frames are {tuple(self.dark_mean.shape[-2:])} "
                    f"but lights are {tuple(image.shape[-2:])}"
                )
            image = image - self.dark_mean
        return image

    def compute_shift(self, index: int) -> tuple[int, int]:
        """Center-of-mass shift for one calibrated (pre-crop) frame."""
        if not self.align_params.center_of_mass:
            return (0, 0)
        return compute_com_shift(self.calibrated(index))

    def rect_for(self, image: torch.Tensor) -> tuple[int, int, int, int] | None:
        if not self._rect_known:
            p = self.align_params
            self._rect = crop_rect(
                image.shape[-2], image.shape[-1], p.crop, p.crop_align, p.crop_offsets
            )
            self._rect_known = True
        return self._rect

    def read_cooked(self, index: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(frame, aligned dark variance or None), both (1, C, H, W) cropped.

        Whole-frame or per-channel shifts alike are applied identically to
        the frame and to the dark variance, so the noise map stays glued to
        the pixels it describes.
        """
        if self.shifts is None:
            raise RuntimeError("alignment shifts have not been computed/loaded yet")
        image = self.calibrated(index)
        shift = self.shifts[index]
        image = apply_crop(apply_frame_shift(image, shift), self.rect_for(image))

        variance = None
        if self.dark_variance is not None:
            variance = apply_crop(
                apply_frame_shift(self.dark_variance, shift), self.rect_for(self.dark_variance)
            )
        return image, variance
