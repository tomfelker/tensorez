"""MP4/AVI input: frame-accurate indexing, color handling, graceful absence.

The interesting property is that ``seq[i]`` is genuinely frame i even on an
inter-coded file, where reaching a frame means seeking to the preceding
keyframe and decoding forward.  The fixtures label every frame twice (flat
background level and white-bar position) so a decoder returning a neighbour or
a keyframe fails loudly instead of producing a plausible-looking image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import (
    marker_frames,
    parse_events,
    requires_video,
    run_cli,
    write_video,
    write_video_10bit,
)
from tensorez import video
from tensorez.color import srgb_to_linear
from tensorez.sequence import ImageSequence


def _bar_column(frame: torch.Tensor) -> int:
    """Column of the white bar in a (1, C, H, W) frame."""
    return int(frame[0, 0, 0].argmax())


@requires_video
def test_lossless_video_decodes_exactly(tmp_path: Path) -> None:
    """rawvideo AVI is all-intra and lossless, so every value is predictable:
    8-bit RGB linearized as sRGB, exactly as a PNG still would be."""
    frames = marker_frames()
    path = write_video(tmp_path / "marks.avi", frames)
    seq = ImageSequence([str(path)])

    assert len(seq) == len(frames)
    for i in range(len(frames)):
        frame = seq[i]
        assert frame.shape == (1, 3, 32, 2 * len(frames) + 4)
        assert frame.dtype == torch.float32
        expected = float(srgb_to_linear(torch.tensor(i * 8 / 255.0)))
        # last column is never covered by the bar
        assert torch.allclose(frame[0, :, :, -1], torch.tensor(expected), atol=1e-6)
        assert _bar_column(frame) in (2 * i, 2 * i + 1)


@requires_video
def test_random_access_is_frame_accurate_on_an_inter_coded_file(tmp_path: Path) -> None:
    """The point of the whole exercise: out-of-order reads of a long-GOP file
    return the frames asked for, not the nearest keyframe."""
    frames = marker_frames()
    path = write_video(tmp_path / "marks.mp4", frames, codec="mpeg4", pix_fmt="yuv420p")
    seq = ImageSequence([str(path)])
    assert len(seq) == len(frames)

    # Jump around, including backwards, which is what forces a re-seek.
    for i in (17, 3, 22, 0, 11, 23, 5, 4, 20, 1):
        assert _bar_column(seq[i]) in (2 * i, 2 * i + 1), f"frame {i}"

    # And the same indices read in order agree with the shuffled reads.
    in_order = [seq[i].clone() for i in range(len(seq))]
    for i in (9, 2, 15):
        assert torch.equal(seq[i], in_order[i])


@requires_video
def test_ten_bit_video_keeps_its_depth(tmp_path: Path) -> None:
    """A 10-bit source must not be squashed to 8 on the way in.

    Levels one 10-bit step apart straddling an 8-bit boundary would collide if
    anything truncated; they have to stay ordered and distinct.  The absolute
    values come back as v/1023 because torchcodec normalizes deeper sources
    correctly -- unlike its 8-bit float path, which is why read_frame scales
    bytes itself (see video._to_unit_range).
    """
    levels = [0, 1, 600, 601, 1000, 1023]
    path = write_video_10bit(tmp_path / "deep.mkv", levels)
    seq = ImageSequence([str(path)])
    assert len(seq) == len(levels)

    decoded = [float(seq[i][0, 0, 0, 0]) for i in range(len(levels))]
    for i, level in enumerate(levels):
        expected = float(srgb_to_linear(torch.tensor(level / 1023.0)))
        assert decoded[i] == pytest.approx(expected, rel=1e-4, abs=1e-7), level

    # the pair that 8-bit truncation would merge (600 and 601 both -> 150/255)
    assert decoded[3] > decoded[2]
    assert decoded[1] > decoded[0]


@requires_video
def test_frame_selection_over_a_video(tmp_path: Path) -> None:
    frames = marker_frames()
    path = write_video(tmp_path / "marks.avi", frames)
    seq = ImageSequence([str(path)], start_frame=2, frame_step=5, end_frame=20)
    assert len(seq) == 4  # raw 2, 7, 12, 17
    assert [_bar_column(seq[i]) // 2 for i in range(4)] == [2, 7, 12, 17]


@requires_video
def test_grayscale_video_reads_as_mono(tmp_path: Path) -> None:
    """A gray source decodes to three identical channels; we collapse it so the
    pipeline treats it as the single-channel data it actually is."""
    frames = marker_frames()
    path = write_video(tmp_path / "gray.avi", frames, pix_fmt="gray")
    seq = ImageSequence([str(path)])
    assert not seq.is_bayer
    assert seq[0].shape == (1, 1, 32, 2 * len(frames) + 4)


@requires_video
def test_reencoding_the_file_is_noticed(tmp_path: Path) -> None:
    """Decoders are cached per file, so a replaced file must invalidate them --
    otherwise a re-shot capture would silently decode as the old one."""
    path = tmp_path / "same_name.avi"
    write_video(path, marker_frames(count=8))
    assert len(ImageSequence([str(path)])) == 8

    write_video(path, marker_frames(count=12))
    seq = ImageSequence([str(path)])
    assert len(seq) == 12
    assert _bar_column(seq[11]) in (22, 23)


@requires_video
def test_video_runs_the_whole_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    h = w = 48
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frames = []
    for i in range(8):  # a wandering blob, so alignment has something to do
        blob = 0.8 * np.exp(-((yy - h / 2 - i % 3) ** 2 + (xx - w / 2 + i % 2) ** 2) / 40.0)
        noisy = np.clip(blob + rng.normal(0, 0.005, (h, w)), 0, 1)
        frames.append(np.repeat((noisy * 255).astype(np.uint8)[..., None], 3, axis=-1))
    path = write_video(tmp_path / "planet.mp4", np.stack(frames),
                       codec="mpeg4", pix_fmt="yuv420p")

    recipe = tmp_path / "r.toml"
    recipe.write_text(f"""
[lights]
paths = ["{path.as_posix()}"]
[align]
center_of_mass = true
[local_lucky]
crossover_wavelength_pixels = 6.0
isoplanatic_patch_pixels = 12.0
[output]
dir = "{(tmp_path / 'out').as_posix()}"
debug_frames = 0
""")
    proc = run_cli(["run", str(recipe)], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    events = parse_events(proc.stdout)
    assert events[-1]["event"] == "done"
    result = np.load(Path(events[0]["run_dir"]) / "local_lucky.npy")
    assert result.shape == (h, w, 3)
    assert np.isfinite(result).all()


def test_absent_video_support_is_a_clean_error(tmp_path: Path, monkeypatch) -> None:
    """With torchcodec or FFmpeg missing, a video input must fail as an
    ordinary bad-input error -- the pipeline reports ValueError without a
    traceback -- and the message has to name both halves of the remedy."""
    def unavailable():
        raise video.VideoSupportError("nope\n" + video._SETUP_HINT)

    monkeypatch.setattr(video, "_video_decoder_class", unavailable)
    fake = tmp_path / "capture.mp4"
    fake.write_bytes(b"not really an mp4")

    assert issubclass(video.VideoSupportError, ValueError)
    with pytest.raises(ValueError) as excinfo:
        ImageSequence([str(fake)])
    message = str(excinfo.value)
    assert "tensorez[video]" in message
    assert "shared" in message  # the static-build trap is the common failure
    assert video.FFMPEG_DIR_ENV in message


def test_other_formats_do_not_need_video_support(tmp_path: Path, monkeypatch) -> None:
    """Video is optional: with it unavailable, SER and stills still work."""
    def unavailable():
        raise video.VideoSupportError("nope")

    monkeypatch.setattr(video, "_video_decoder_class", unavailable)
    from tensorez import ser

    path = tmp_path / "t.ser"
    ser.write_ser(path, np.full((3, 4, 4, 1), 7, np.uint8), ser.ColorId.MONO)
    assert len(ImageSequence([str(path)])) == 3


def test_unsupported_extension_names_the_video_formats(tmp_path: Path) -> None:
    junk = tmp_path / "capture.raw"
    junk.write_bytes(b"\0")
    with pytest.raises(ValueError, match=r"\.mp4"):
        ImageSequence([str(junk)])
