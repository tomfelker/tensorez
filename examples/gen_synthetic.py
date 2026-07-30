"""Regenerate examples/truth.npy and examples/synthetic_planet.ser (~24 MB).

A procedural banded planet with a spot, then 60 frames of simulated seeing:
per-frame gaussian blur (sigma ~ |N(1.8, 1.2)| + 0.2, so a few near-sharp
frames), random integer shifts (sigma 3 px), and gaussian noise (sigma 0.01),
written as a 16-bit RGB SER. Deterministic (seed 42).

Usage: python gen_synthetic.py  (from this directory; needs numpy and the
tensorez package on the path for the SER writer, or astrolock's ser module).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'cli'))
from tensorez.ser import write_ser, ColorId  # noqa: E402


def make_truth(H=256, W=256):
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx, r = H / 2, W / 2, 80
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disk = np.clip((r - d) / 3.0, 0, 1)
    bands = 0.75 + 0.25 * np.sin((yy - cy) / 9.0) * disk
    rgb = np.stack([bands, bands * 0.85, bands * 0.65], -1) * disk[..., None]
    spot = np.exp(-(((yy - cy - 25) ** 2 + (xx - cx - 30) ** 2) / 60.0))
    rgb[..., 0] += 0.25 * spot * disk
    rgb[..., 1] -= 0.1 * spot * disk
    return np.clip(rgb, 0, 1)


def blur(img, sigma):
    if sigma <= 0.01:
        return img
    k = max(3, int(sigma * 4) | 1)
    x = np.arange(k) - k // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    out = img
    for axis in (0, 1):
        out = np.apply_along_axis(lambda m: np.convolve(m, g, 'same'), axis, out)
    return out


def main():
    truth = make_truth()
    np.save(os.path.join(HERE, 'truth.npy'), truth.astype(np.float32))
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(60):
        sigma = abs(rng.normal(1.8, 1.2)) + 0.2
        dx, dy = rng.normal(0, 3, 2)
        f = np.roll(truth, (int(round(dy)), int(round(dx))), (0, 1))
        f = blur(f, sigma)
        f = np.clip(f + rng.normal(0, 0.01, f.shape), 0, 1)
        frames.append((f * 65535).astype(np.uint16))
    write_ser(os.path.join(HERE, 'synthetic_planet.ser'), np.stack(frames), ColorId.RGB)
    print('wrote truth.npy and synthetic_planet.ser (60 frames)')


if __name__ == '__main__':
    main()
