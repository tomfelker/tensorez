"""Regenerate examples/truth.npy, examples/synthetic_planet.ser and
examples/speckle_planet.ser (~24 MB each).

A procedural banded planet with a spot, then two 60-frame simulated captures
of it, both deterministic (seed 42):

* synthetic_planet.ser — *magnitude-only* seeing: per-frame centered gaussian
  blur (sigma ~ |N(1.8, 1.2)| + 0.2, so a few near-sharp frames), random
  integer shifts (sigma 3 px), gaussian noise (sigma 0.01).  A centered
  gaussian is zero-phase: every frame keeps the truth's Fourier phases, only
  the magnitudes are damped.  Good for per-pixel/whole-frame lucky tests.

* speckle_planet.ser — *phase-distorting* seeing: each frame is convolved
  with a short-exposure speckle PSF (a handful of gaussian speckles scattered
  by an amount drawn per frame, so a few frames are near-diffraction-limited),
  plus the same shifts and noise.  Off-center speckles scramble Fourier
  phases, which is what lucky_fourier exists to fix — the zero-phase capture
  above gives it nothing to recover, and it (correctly) cannot beat the plain
  average there.

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


def speckle_psf(rng, quality, size=33):
    """Short-exposure seeing PSF: a few gaussian speckles scattered about the
    center. quality in (0, 1]: 1 = one tight, nearly centered speckle
    (a lucky moment); small = many widely scattered speckles. The off-center
    speckles are what distorts the Fourier *phases*."""
    spread = 4.0 * (1.0 - quality) + 0.15
    n_speckles = 1 + int(round(6 * (1 - quality)))
    yy, xx = np.mgrid[0:size, 0:size] - size // 2
    psf = np.zeros((size, size))
    for _ in range(n_speckles):
        dy, dx = rng.normal(0, spread, 2)
        psf += np.exp(-((yy - dy) ** 2 + (xx - dx) ** 2) / (2 * 0.9 ** 2))
    return psf / psf.sum()


def convolve_psf(img, psf):
    """FFT convolution over the spatial axes of an HWC image (the planet sits
    on a wide dark border, so the circular wrap never touches anything)."""
    H, W = img.shape[:2]
    kernel = np.zeros((H, W))
    ph, pw = psf.shape
    kernel[:ph, :pw] = psf
    kernel = np.roll(kernel, (-(ph // 2), -(pw // 2)), (0, 1))
    K = np.fft.rfft2(kernel)
    return np.fft.irfft2(np.fft.rfft2(img, axes=(0, 1)) * K[:, :, None],
                         s=(H, W), axes=(0, 1))


def write_speckle_planet(path=None):
    """The phase-distorting capture; callable on its own (tests regenerate
    just this file when it is missing)."""
    truth = make_truth()
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(60):
        quality = np.clip(abs(rng.normal(0.35, 0.25)), 0.05, 1.0)
        dx, dy = rng.normal(0, 3, 2)
        f = np.roll(truth, (int(round(dy)), int(round(dx))), (0, 1))
        f = convolve_psf(f, speckle_psf(rng, quality))
        f = np.clip(f + rng.normal(0, 0.01, f.shape), 0, 1)
        frames.append((f * 65535).astype(np.uint16))
    write_ser(path or os.path.join(HERE, 'speckle_planet.ser'),
              np.stack(frames), ColorId.RGB)


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
    write_speckle_planet()
    print('wrote truth.npy, synthetic_planet.ser and speckle_planet.ser (60 frames each)')


if __name__ == '__main__':
    main()
