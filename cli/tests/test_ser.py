"""SER reader round-trips and Bayer handling on tiny generated files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import parse_events, run_cli
from tensorez import ser
from tensorez.bayer import bayer_mask, demosaic
from tensorez.sequence import ImageSequence, read_ser_frame


def test_ser_round_trip_rgb16(tmp_path: Path) -> None:
    """(f) write -> read round-trip preserves pixels exactly."""
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 65536, size=(3, 8, 10, 3), dtype=np.uint16)
    path = tmp_path / "t.ser"
    ser.write_ser(path, frames, ser.ColorId.RGB)

    header = ser.read_header(path)
    assert (header.image_width, header.image_height) == (10, 8)
    assert header.frame_count == 3
    assert header.pixel_depth_per_plane == 16

    for i in range(3):
        raw, _ = ser.read_frame_raw(path, i)
        assert np.array_equal(raw, frames[i])
        scaled, _ = ser.read_frame(path, i)
        assert np.allclose(scaled, frames[i].astype(np.float32) / 65535.0)


def test_ser_round_trip_mono8_and_endianness(tmp_path: Path) -> None:
    frames = np.arange(2 * 4 * 6, dtype=np.uint8).reshape(2, 4, 6, 1)
    for flag in (0, 1):  # 0 = little-endian data (the field is inverted)
        path = tmp_path / f"m{flag}.ser"
        ser.write_ser(path, frames, ser.ColorId.MONO, little_endian_flag=flag)
        raw, header = ser.read_frame_raw(path, 1)
        assert header.num_channels == 1
        assert np.array_equal(raw, frames[1])
    # 16-bit is where byte order actually matters
    frames16 = np.array([[[[0x1234]]]], dtype=np.uint16)
    for flag in (0, 1):
        path = tmp_path / f"e{flag}.ser"
        ser.write_ser(path, frames16, ser.ColorId.MONO, little_endian_flag=flag)
        raw, _ = ser.read_frame_raw(path, 0)
        assert int(raw[0, 0, 0]) == 0x1234


def test_out_of_range_frame(tmp_path: Path) -> None:
    path = tmp_path / "t.ser"
    ser.write_ser(path, np.zeros((1, 4, 4, 1), np.uint8), ser.ColorId.MONO)
    with pytest.raises(IndexError):
        ser.read_frame(path, 1)


def test_bayer_demosaic_flat_field(tmp_path: Path) -> None:
    """A mosaic of a uniform color demosaics back to that color exactly."""
    r, g, b = 0.25, 0.5, 0.75
    mask = bayer_mask(ser.ColorId.BAYER_RGGB, 8, 8)
    flat = torch.tensor([r, g, b]).view(1, 3, 1, 1) * torch.ones(1, 3, 8, 8)
    mosaic = (flat * mask).sum(dim=1, keepdim=True)
    out = demosaic(mosaic, ser.ColorId.BAYER_RGGB)
    assert torch.allclose(out, flat, atol=1e-6)

    # GRBG: same field, different phase
    mask_g = bayer_mask(ser.ColorId.BAYER_GRBG, 8, 8)
    mosaic_g = (flat * mask_g).sum(dim=1, keepdim=True)
    out_g = demosaic(mosaic_g, ser.ColorId.BAYER_GRBG)
    assert torch.allclose(out_g, flat, atol=1e-6)


def test_bayer_ser_read_and_even_shift_enforcement(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 65536, size=(2, 16, 16, 1), dtype=np.uint16)
    path = tmp_path / "bayer.ser"
    ser.write_ser(path, frames, ser.ColorId.BAYER_RGGB)

    image = read_ser_frame(str(path), 0)
    assert image.shape == (1, 3, 16, 16)
    # at R photosites the R channel is the raw sample
    assert torch.allclose(
        image[0, 0, 0::2, 0::2],
        torch.from_numpy(frames[0, 0::2, 0::2, 0].astype(np.float32) / 65535.0),
    )

    seq = ImageSequence([str(path)])
    assert seq.is_bayer

    # pipeline must refuse Bayer lights without only_even_shifts
    recipe = tmp_path / "r.toml"
    recipe.write_text(f"""
[recipe]
version = 0
name = "bayer"
[lights]
paths = ["{path.as_posix()}"]
[output]
dir = "{(tmp_path / 'out').as_posix()}"
""")
    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode != 0
    events = parse_events(proc.stdout)
    assert events[-1]["event"] == "error"
    assert "only_even_shifts" in events[-1]["message"]


def test_frame_selection(tmp_path: Path) -> None:
    frames = np.stack([np.full((4, 4, 1), i, dtype=np.uint8) for i in range(10)])
    path = tmp_path / "t.ser"
    ser.write_ser(path, frames, ser.ColorId.MONO)
    seq = ImageSequence([str(path)], start_frame=2, frame_step=3, end_frame=9)
    # raw indices 2, 5, 8
    assert len(seq) == 3
    values = [float(seq[i].max()) * 255.0 for i in range(3)]
    assert [round(v) for v in values] == [2, 5, 8]
