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


def test_bayer_debayer_modes(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 65536, size=(2, 16, 16, 1), dtype=np.uint16)
    path = tmp_path / "bayer.ser"
    ser.write_ser(path, frames, ser.ColorId.BAYER_RGGB)
    raw = torch.from_numpy(frames[0, :, :, 0].astype(np.float32) / 65535.0)

    # bilinear (the default): full size; at R photosites R is the raw sample
    image = read_ser_frame(str(path), 0)
    assert image.shape == (1, 3, 16, 16)
    assert torch.allclose(image[0, 0, 0::2, 0::2], raw[0::2, 0::2])

    # superpixel_rggb: half size, 4 channels of real photosites (R, G1, G2, B)
    sp4 = read_ser_frame(str(path), 0, "superpixel_rggb")
    assert sp4.shape == (1, 4, 8, 8)
    assert torch.equal(sp4[0, 0], raw[0::2, 0::2])  # R
    assert torch.equal(sp4[0, 1], raw[0::2, 1::2])  # G1 (red row)
    assert torch.equal(sp4[0, 2], raw[1::2, 0::2])  # G2 (blue row)
    assert torch.equal(sp4[0, 3], raw[1::2, 1::2])  # B

    # superpixel_rgb: same but the greens are averaged
    sp3 = read_ser_frame(str(path), 0, "superpixel_rgb")
    assert sp3.shape == (1, 3, 8, 8)
    assert torch.equal(sp3[0, 0], sp4[0, 0])
    assert torch.allclose(sp3[0, 1], (sp4[0, 1] + sp4[0, 2]) / 2)
    assert torch.equal(sp3[0, 2], sp4[0, 3])

    # none: the mosaic passes through as mono
    mono = read_ser_frame(str(path), 0, "none")
    assert mono.shape == (1, 1, 16, 16)
    assert torch.equal(mono[0, 0], raw)

    seq = ImageSequence([str(path)], debayer="superpixel_rgb")
    assert seq.is_bayer
    assert seq.read_frame(0).shape == (1, 3, 8, 8)
    assert "debayer: superpixel_rgb" in seq.identity()


def test_bayer_pipeline_end_to_end(tmp_path: Path) -> None:
    """Bayer lights need no alignment restrictions anymore; every debayer
    mode runs the whole pipeline and yields the expected geometry."""
    rng = np.random.default_rng(2)
    h = w = 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frames = []
    for i in range(6):  # a wandering blob so alignment has something to do
        blob = np.exp(-((yy - h / 2 - i % 3) ** 2 + (xx - w / 2 + i % 2) ** 2) / 18.0)
        noisy = np.clip(blob + rng.normal(0, 0.01, (h, w)), 0, 1)
        frames.append((noisy * 65535).astype(np.uint16)[..., None])
    path = tmp_path / "bayer.ser"
    ser.write_ser(path, np.stack(frames), ser.ColorId.BAYER_GRBG)

    expected = {  # (channels, height/width) of the local_lucky product
        "bilinear": (3, 32),
        "superpixel_rgb": (3, 16),
        "superpixel_rggb": (4, 16),
        "none": (1, 32),
    }
    for debayer, (channels, size) in expected.items():
        recipe = tmp_path / f"r_{debayer}.toml"
        recipe.write_text(f"""
[lights]
paths = ["{path.as_posix()}"]
debayer = "{debayer}"
[local_lucky]
crossover_wavelength_pixels = 4.0
isoplanatic_patch_pixels = 8.0
[output]
dir = "{(tmp_path / 'out').as_posix()}"
debug_frames = 0
""")
        proc = run_cli(["run", str(recipe)], cwd=tmp_path)
        assert proc.returncode == 0, f"{debayer}: " + proc.stdout + proc.stderr
        events = parse_events(proc.stdout)
        assert events[-1]["event"] == "done"
        result = np.load(Path(events[0]["run_dir"]) / "local_lucky.npy")
        assert result.shape == (size, size, channels), debayer


def test_frame_selection(tmp_path: Path) -> None:
    frames = np.stack([np.full((4, 4, 1), i, dtype=np.uint8) for i in range(10)])
    path = tmp_path / "t.ser"
    ser.write_ser(path, frames, ser.ColorId.MONO)
    seq = ImageSequence([str(path)], start_frame=2, frame_step=3, end_frame=9)
    # raw indices 2, 5, 8
    assert len(seq) == 3
    values = [float(seq[i].max()) * 255.0 for i in range(3)]
    assert [round(v) for v in values] == [2, 5, 8]
