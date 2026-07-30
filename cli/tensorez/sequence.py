"""Image sequences: a list of files (SER videos and/or stills) seen as one
flat, frame-steppable sequence.

Terminology kept from the reference implementation:

* a *raw* index addresses any frame in the concatenation of all files;
* a *cooked* index is relative to ``start_frame`` and strides by
  ``frame_step`` (this is what the rest of the pipeline uses).

Frames are returned as float32 linear-light torch tensors of shape
(1, C, H, W).  8-bit stills are assumed sRGB-encoded and are linearized;
16-bit sources are assumed linear.  Bayer SER files are debayered on read
according to the sequence's ``debayer`` mode (see bayer.py); the mode is
ignored for non-Bayer sources.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import ser
from .bayer import DEBAYER_MODES, demosaic, is_supported_bayer, superpixel_rgb, superpixel_rggb
from .color import srgb_to_linear

STILL_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}

SUPPORTED_SER_COLOR_IDS = {
    ser.ColorId.MONO,
    ser.ColorId.RGB,
    ser.ColorId.BAYER_RGGB,
    ser.ColorId.BAYER_GRBG,
}


def _is_ser(path: str) -> bool:
    return path.lower().endswith(".ser")


def read_still(path: str | Path) -> torch.Tensor:
    """Decode a still image to float32 linear light, (1, C, H, W)."""
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[-1] == 4:  # discard alpha
        arr = arr[:, :, :3]

    if arr.dtype == np.uint8:
        t = torch.from_numpy(arr.astype(np.float32) / 255.0)
        t = srgb_to_linear(t)
    elif arr.dtype == np.uint16:
        t = torch.from_numpy(arr.astype(np.float32) / 65535.0)
    elif arr.dtype in (np.float32, np.float64):
        t = torch.from_numpy(arr.astype(np.float32))
    else:
        raise ValueError(f"{path}: unsupported still dtype {arr.dtype}")

    return t.permute(2, 0, 1).unsqueeze(0).contiguous()


def read_ser_frame(path: str, frame_index: int, debayer: str = "bilinear") -> torch.Tensor:
    """Read one SER frame as float32 linear light, (1, C, H, W); debayer Bayer."""
    frame_hwc, header = ser.read_frame(path, frame_index)
    color_id = header.color_id
    t = torch.from_numpy(np.ascontiguousarray(frame_hwc)).permute(2, 0, 1).unsqueeze(0)

    if color_id == ser.ColorId.MONO:
        return t
    if color_id == ser.ColorId.RGB:
        return t
    if is_supported_bayer(color_id):
        if debayer == "bilinear":
            return demosaic(t, color_id)
        if debayer == "superpixel_rgb":
            return superpixel_rgb(t, color_id)
        if debayer == "superpixel_rggb":
            return superpixel_rggb(t, color_id)
        if debayer == "none":
            return t  # keep the mosaic as mono
        raise ValueError(f"unknown debayer mode {debayer!r} (expected one of {DEBAYER_MODES})")
    raise ValueError(f"{path}: unsupported SER color_id {color_id}")


class ImageSequence:
    """A frame-addressable view over globs of SER files and stills."""

    def __init__(
        self,
        paths: list[str],
        start_frame: int = 0,
        frame_step: int = 1,
        end_frame: int | None = None,
        debayer: str = "bilinear",
    ):
        if debayer not in DEBAYER_MODES:
            raise ValueError(f"unknown debayer mode {debayer!r} (expected one of {DEBAYER_MODES})")
        self.debayer = debayer
        self.start_raw_frame = start_frame
        self.raw_frame_step = frame_step

        filenames: list[str] = []
        for pattern in paths:
            matches = sorted(glob.glob(pattern))
            if not matches and os.path.exists(pattern):
                matches = [pattern]
            filenames.extend(matches)
        if not filenames:
            raise FileNotFoundError(f"No files match {paths!r}")

        self.files: list[tuple[str, int]] = []  # (filename, first raw index)
        raw = 0
        for filename in filenames:
            self.files.append((filename, raw))
            raw += self._frame_count_of(filename)

        self.raw_frame_count = raw
        if end_frame is not None:
            self.raw_frame_count = min(self.raw_frame_count, end_frame)
        if self.start_raw_frame >= self.raw_frame_count:
            raise ValueError(
                f"start_frame {start_frame} is beyond the {self.raw_frame_count} available frames"
            )
        self.cooked_frame_count = (
            (self.raw_frame_count - 1 - self.start_raw_frame) // self.raw_frame_step + 1
        )

    @staticmethod
    def _frame_count_of(filename: str) -> int:
        if _is_ser(filename):
            header = ser.read_header(filename)
            if header.color_id not in SUPPORTED_SER_COLOR_IDS:
                raise ValueError(
                    f"{filename}: SER color_id {header.color_id} not supported "
                    f"(supported: MONO, RGB, BAYER_RGGB, BAYER_GRBG)"
                )
            return header.frame_count
        ext = os.path.splitext(filename)[1].lower()
        if ext not in STILL_EXTENSIONS:
            raise ValueError(f"{filename}: unsupported file type {ext!r}")
        return 1

    @property
    def color_id(self) -> ser.ColorId:
        """Color layout of the first file (stills count as RGB)."""
        filename = self.files[0][0]
        if _is_ser(filename):
            return ser.ColorId(ser.read_header(filename).color_id)
        return ser.ColorId.RGB

    @property
    def is_bayer(self) -> bool:
        return is_supported_bayer(self.color_id)

    def identity(self) -> str:
        """Human-readable cache-key contribution: file identities + selection.

        Files are identified by path + size + mtime so an edited or replaced
        file invalidates downstream caches.
        """
        lines = ["ImageSequence:"]
        for filename, _ in self.files:
            st = os.stat(filename)
            lines.append(f"  file: {filename} size={st.st_size} mtime_ns={st.st_mtime_ns}")
        lines.append(f"  start_raw_frame: {self.start_raw_frame}")
        lines.append(f"  raw_frame_step: {self.raw_frame_step}")
        lines.append(f"  raw_frame_count: {self.raw_frame_count}")
        lines.append(f"  debayer: {self.debayer}")
        return "\n".join(lines) + "\n"

    def _raw_to_file(self, raw_index: int) -> tuple[str, int]:
        filename, frame_index = self.files[0]
        for candidate, first_raw in self.files:
            if first_raw <= raw_index:
                filename, frame_index = candidate, raw_index - first_raw
            else:
                break
        return filename, frame_index

    def read_frame(self, cooked_index: int) -> torch.Tensor:
        if not (0 <= cooked_index < self.cooked_frame_count):
            raise IndexError(cooked_index)
        raw_index = self.start_raw_frame + cooked_index * self.raw_frame_step
        filename, frame_index = self._raw_to_file(raw_index)
        if _is_ser(filename):
            return read_ser_frame(filename, frame_index, self.debayer)
        return read_still(filename)

    def __len__(self) -> int:
        return self.cooked_frame_count

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.read_frame(index)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
