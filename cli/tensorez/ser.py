"""SER video file format (the planetary-imaging capture format).

A SER file is a 178-byte header followed by raw frames, followed by optional
per-frame timestamps (which we ignore).  Header layout, little-endian:

    '<14slllllll40s40s40sqq'
    file_id, lu_id, color_id, little_endian, image_width, image_height,
    pixel_depth_per_plane, frame_count, observer, instrument, telescope,
    date_time, date_time_utc

Quirk faithfully ported from the reference implementation: the
``little_endian`` field is interpreted *opposite* to the spec document —
``little_endian == 0`` means the pixel data is little-endian.  Every capture
program apparently agrees on this, so we follow suit.

Pixel values are scaled to [0, 1] by dividing by ``2**bit_depth - 1``
(with a minimum bit depth of 8).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

HEADER_STRUCT = struct.Struct("<14slllllll40s40s40sqq")
assert HEADER_STRUCT.size == 178


class ColorId(IntEnum):
    MONO = 0
    BAYER_RGGB = 8
    BAYER_GRBG = 9
    BAYER_GBRG = 10
    BAYER_BGGR = 11
    BAYER_CYYM = 16
    BAYER_YCMY = 17
    BAYER_YMCY = 18
    BAYER_MYYC = 19
    RGB = 100
    BGR = 101


@dataclass(frozen=True)
class SerHeader:
    file_id: bytes
    lu_id: int
    color_id: int
    little_endian: int
    image_width: int
    image_height: int
    pixel_depth_per_plane: int
    frame_count: int
    observer: bytes
    instrument: bytes
    telescope: bytes
    date_time: int
    date_time_utc: int

    @property
    def num_channels(self) -> int:
        return 3 if self.color_id >= ColorId.RGB else 1

    @property
    def bytes_per_frame(self) -> int:
        bytes_per_channel = self.pixel_depth_per_plane // 8
        return bytes_per_channel * self.num_channels * self.image_width * self.image_height


def read_header(path: str | Path) -> SerHeader:
    with open(path, "rb") as f:
        return SerHeader(*HEADER_STRUCT.unpack(f.read(HEADER_STRUCT.size)))


def read_frame_raw(path: str | Path, frame_index: int) -> tuple[np.ndarray, SerHeader]:
    """Read one frame as its native integer dtype, shape (H, W, C)."""
    header = read_header(path)
    if not (0 <= frame_index < header.frame_count):
        raise IndexError(f"frame {frame_index} out of range for {path} ({header.frame_count} frames)")

    bytes_per_channel = header.pixel_depth_per_plane // 8
    if bytes_per_channel == 1:
        dtype = np.dtype("uint8")
    elif bytes_per_channel == 2:
        dtype = np.dtype("uint16")
    else:
        raise ValueError(f"{path}: unsupported pixel depth {header.pixel_depth_per_plane}")

    # Opposite from the SER spec document, but matching real capture software.
    dtype = dtype.newbyteorder("<" if header.little_endian == 0 else ">")

    with open(path, "rb") as f:
        f.seek(HEADER_STRUCT.size + frame_index * header.bytes_per_frame)
        buf = f.read(header.bytes_per_frame)
    if len(buf) != header.bytes_per_frame:
        raise ValueError(f"{path}: truncated frame {frame_index}")

    frame = np.frombuffer(buf, dtype=dtype)
    return frame.reshape(header.image_height, header.image_width, header.num_channels), header


def read_frame(path: str | Path, frame_index: int) -> tuple[np.ndarray, SerHeader]:
    """Read one frame scaled to float32 [0, 1], shape (H, W, C)."""
    frame, header = read_frame_raw(path, frame_index)
    max_val = (1 << max(header.pixel_depth_per_plane, 8)) - 1
    return frame.astype(np.float32) / max_val, header


def write_ser(
    path: str | Path,
    frames: np.ndarray,
    color_id: ColorId,
    *,
    little_endian_flag: int = 0,
) -> None:
    """Write a SER file from integer frames.

    ``frames`` is (N, H, W, C) uint8 or uint16 (C = 1 for MONO/Bayer, 3 for
    RGB).  Mainly used by tests to synthesize tiny files.
    """
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError("frames must be (N, H, W, C)")
    n, h, w, c = frames.shape
    expected_c = 3 if color_id >= ColorId.RGB else 1
    if c != expected_c:
        raise ValueError(f"color_id {color_id!r} requires {expected_c} channels, got {c}")
    if frames.dtype == np.uint8:
        depth = 8
    elif frames.dtype == np.uint16:
        depth = 16
    else:
        raise ValueError("frames must be uint8 or uint16")

    byte_order = "<" if little_endian_flag == 0 else ">"
    header = HEADER_STRUCT.pack(
        b"LUCAM-RECORDER", 0, int(color_id), little_endian_flag,
        w, h, depth, n, b"", b"", b"", 0, 0,
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(frames, dtype=frames.dtype.newbyteorder(byte_order)).tobytes())
