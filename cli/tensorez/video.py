"""Video files (MP4, AVI, MOV, ...) as frame-addressable image sequences.

Unlike SER, a video file has already been demosaiced and lossily compressed by
tools we do not control, so there is no mosaic left to recover and the
sequence's ``debayer`` mode does not apply to it.  The most that can be
recovered is a canonical encoding that inverts cleanly to linear light, and
8-bit sRGB is exactly that: BT.709, which essentially all consumer video
declares, shares sRGB's primaries and white point, so decoded frames take the
same sRGB -> linear path as any other 8-bit still (see ``sequence.read_still``).

Decoding is torchcodec's, which indexes by frame number and hands back torch
tensors directly.  Frame-accurate seeking is its ``seek_mode="exact"`` (the
default used here): it scans the container up front so that ``decoder[i]`` is
genuinely the i'th frame rather than the nearest preceding keyframe, which is
the part that wrappers built on ffmpeg's ``-ss`` get wrong.  The scan costs a
pass over the file when a video is first opened.

torchcodec deliberately does not bundle FFmpeg -- it loads whatever shared
libraries the system already has (majors 4-8).  On Windows that takes more than
putting ffmpeg.exe on PATH: since Python 3.8 the loader no longer searches PATH
for an extension module's dependencies, so the directory holding the FFmpeg
DLLs has to be handed to ``os.add_dll_directory`` *before* torchcodec is
imported.  ``_video_decoder_class`` does that, and folds both failure modes --
torchcodec missing, FFmpeg missing -- into one actionable error.
"""

from __future__ import annotations

import functools
import os
import shutil
import warnings
from pathlib import Path

import torch

from .color import srgb_to_linear

VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv",
}

FFMPEG_DIR_ENV = "TENSOREZ_FFMPEG_DIR"

# Transfer characteristics that are not (approximately) the sRGB curve.  HDR
# footage decoded as if it were sRGB comes out wrong; we say so rather than
# silently stacking it.
_HDR_TRANSFERS = {"smpte2084", "arib-std-b67", "smpte428", "log100", "log316"}

_SETUP_HINT = (
    "Video support needs torchcodec and FFmpeg's shared libraries (majors "
    "4-8), which torchcodec loads at run time rather than bundling:\n"
    "    pip install tensorez[video]\n"
    "    winget install BtbN.FFmpeg.LGPL.Shared.7.1      (Windows)\n"
    "    apt install ffmpeg  /  brew install ffmpeg      (Linux / macOS)\n"
    "A *shared* build is required.  The common Windows packages (including "
    "Gyan.FFmpeg) are static builds that ship ffmpeg.exe and no DLLs, which "
    "looks installed but cannot be loaded.  If the libraries are not in the "
    f"directory holding ffmpeg on PATH, point {FFMPEG_DIR_ENV} at them."
)


class VideoSupportError(ValueError):
    """Video decoding is unavailable (torchcodec or FFmpeg missing).

    Deliberately a ValueError: the pipeline already reports failures to open
    an input as clean recipe-level errors, so this arrives as an actionable
    message rather than an unexpected-crash traceback.
    """


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def ffmpeg_dir() -> Path | None:
    """The directory holding FFmpeg's shared libraries, or None if not found."""
    override = os.environ.get(FFMPEG_DIR_ENV)
    if override:
        return Path(override)
    exe = shutil.which("ffmpeg")
    if exe is not None:
        # Shared builds keep the libraries beside the executable.  resolve()
        # so that a symlinked shim lands on the real directory, not the shim's.
        return Path(exe).resolve().parent
    return None


# Holding the handle keeps the directory registered; closing it removes it.
_dll_directory: object | None = None
_dll_registered = False


def _register_ffmpeg_dir() -> None:
    """Make FFmpeg's libraries findable by torchcodec's extension modules."""
    global _dll_directory, _dll_registered
    if _dll_registered:
        return
    _dll_registered = True  # set first: one attempt, even if it fails
    directory = ffmpeg_dir()
    if hasattr(os, "add_dll_directory") and directory is not None and directory.is_dir():
        _dll_directory = os.add_dll_directory(str(directory))


def _condensed(error: Exception, limit: int = 200) -> str:
    """One-line gist of an exception.

    torchcodec reports a failed library load as a many-line dump of every shim
    it tried; that detail buries the part the user can act on, so it is
    collapsed to a breadcrumb and the remedy goes first.
    """
    text = " ".join(str(error).split())
    return text if len(text) <= limit else text[:limit] + " ..."


@functools.cache
def _video_decoder_class():
    """Import torchcodec, or explain what is missing."""
    _register_ffmpeg_dir()
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as e:
        raise VideoSupportError(
            f"video support is not installed.\n\n{_SETUP_HINT}\n\n({_condensed(e)})"
        ) from e
    except Exception as e:
        # torchcodec itself imports, but none of its FFmpeg shims could bind.
        found = ffmpeg_dir()
        where = f"looked in {found}" if found else "found no ffmpeg on PATH"
        raise VideoSupportError(
            f"torchcodec is installed but could not load FFmpeg's shared "
            f"libraries ({where}).\n\n{_SETUP_HINT}\n\n({_condensed(e)})"
        ) from e
    return VideoDecoder


def _warn_about_color(path: str, metadata) -> None:
    transfer = (getattr(metadata, "color_transfer_characteristic", None) or "").lower()
    if transfer in _HDR_TRANSFERS:
        warnings.warn(
            f"{path}: transfer characteristic {transfer!r} is not sRGB-like; it will "
            "be linearized as if it were sRGB, which is wrong for HDR footage. "
            "Convert to SDR first for a meaningful stack.",
            stacklevel=3,
        )


@functools.lru_cache(maxsize=8)
def _open(path: str, size: int, mtime_ns: int):
    """Open a decoder, keyed on file identity so a replaced file is reopened.

    Bounded so that a long glob of videos does not hold every one of them open,
    and cached because reopening would repeat the exact-seek scan per frame.
    """
    del size, mtime_ns  # part of the cache key only
    VideoDecoder = _video_decoder_class()
    try:
        decoder = VideoDecoder(path, seek_mode="exact", output_dtype="auto")
    except VideoSupportError:
        raise
    except Exception as e:
        raise ValueError(f"{path}: could not be opened as video: {e}") from e
    _warn_about_color(path, decoder.metadata)
    return decoder


def _decoder(path: str | Path):
    st = os.stat(path)
    return _open(str(path), st.st_size, st.st_mtime_ns)


def _is_mono(metadata) -> bool:
    """True for a grayscale source, which torchcodec expands to 3 channels."""
    return (getattr(metadata, "pixel_format", None) or "").lower().startswith("gray")


def is_mono(path: str | Path) -> bool:
    return _is_mono(_decoder(path).metadata)


def frame_count(path: str | Path) -> int:
    count = _decoder(path).metadata.num_frames
    if not count:
        raise ValueError(f"{path}: contains no decodable video frames")
    return int(count)


def _to_unit_range(frame: torch.Tensor, path: str) -> torch.Tensor:
    """Scale a decoded frame to float32 in [0, 1], whatever depth it came at.

    ``output_dtype="auto"`` hands back the narrowest type that holds the
    source: uint8 for ordinary 8-bit video, float32 for anything deeper.  We
    take that rather than always asking for float32 because torchcodec's
    *8-bit* float conversion is lossy in a way worth avoiding -- it promotes
    the byte with a shift and divides by 65535, i.e. v*256/65535 instead of
    v/255, which is 0.39% low and never quite reaches white.  Dividing the
    exact byte ourselves keeps a video frame identical to a still of the same
    value.  Deeper sources have no such problem: measured against a 10-bit
    source, float32 output is v/1023 to within 4e-6, so it passes through
    untouched and nothing is truncated to 8 bits.
    """
    if frame.dtype == torch.uint8:
        return frame.float() / 255.0
    if frame.dtype == torch.uint16:
        return frame.float() / 65535.0
    if frame.dtype.is_floating_point:
        return frame.float()  # already normalized to [0, 1]
    raise ValueError(f"{path}: unexpected decoded frame dtype {frame.dtype}")


def read_frame(path: str | Path, frame_index: int) -> torch.Tensor:
    """Read one frame as float32 linear light, (1, C, H, W)."""
    decoder = _decoder(path)
    frame = decoder[frame_index]  # (C, H, W), already RGB, depth per the source
    if _is_mono(decoder.metadata):
        frame = frame[:1]  # grayscale decodes to three identical channels
    return srgb_to_linear(_to_unit_range(frame, str(path))).unsqueeze(0).contiguous()
