"""Classical deconvolution baselines.

Shared infrastructure.
- Non-blind (known kernel): Wiener and Richardson-Lucy, parameters chosen on val.
- Blind ("phương pháp của đồ án CV"): straight-line motion PSF whose length and angle are
  grid-searched on a validation set by maximising a no-reference sharpness
  score with a ringing penalty, then Richardson-Lucy.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace, rotate
from skimage.restoration import richardson_lucy, wiener


def wiener_rgb(blur: np.ndarray, kernel: np.ndarray, balance: float) -> np.ndarray:
    out = np.stack([wiener(blur[..., c], kernel, balance) for c in range(3)], -1)
    return np.clip(out, 0, 1).astype(np.float32)


def richardson_lucy_rgb(blur: np.ndarray, kernel: np.ndarray, num_iter: int) -> np.ndarray:
    out = np.stack([richardson_lucy(blur[..., c], kernel, num_iter=num_iter) for c in range(3)], -1)
    return np.clip(out, 0, 1).astype(np.float32)


def line_psf(length: int, angle_deg: float, size: int = 41) -> np.ndarray:
    k = np.zeros((size, size), np.float64)
    c = size // 2
    half = max(length, 1) // 2
    k[c, c - half : c + half + 1] = 1.0
    k = rotate(k, angle_deg, reshape=False, order=1)
    k = np.clip(k, 0, None)
    return (k / k.sum()).astype(np.float32)


def sharpness_score(img: np.ndarray, ringing_weight: float = 0.1) -> float:
    gray = img.mean(-1)
    lap = laplace(gray)
    ringing = np.mean(np.abs(np.diff(np.sign(lap), axis=1)))  # crude oscillation proxy
    return float(lap.var() - ringing_weight * ringing)


def blind_rl(blur: np.ndarray, length: int, angle: float, num_iter: int) -> np.ndarray:
    return richardson_lucy_rgb(blur, line_psf(length, angle), num_iter)
