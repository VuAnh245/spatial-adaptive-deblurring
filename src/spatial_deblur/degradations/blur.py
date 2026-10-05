"""Apply uniform (D1) and region-mixed non-uniform (D2) motion blur.

Shared infrastructure. Images are float32 HxWx3 in [0, 1] (sRGB).

    B = sum_r M_r * (K_r conv I) + n,      S(x, y) = sum_r M_r(x, y) s_r
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from .kernels import KERNEL_SIZE, SeverityScale, kernel_for_severity
from .masks import soft_masks


@dataclass
class BlurSample:
    blur: np.ndarray            # H x W x 3, float32 [0, 1]
    severity_map: np.ndarray    # H x W, float32 [0, 1]
    kernels: list = field(default_factory=list)
    severities: list = field(default_factory=list)
    masks: np.ndarray | None = None


def convolve_rgb(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same-size convolution with reflect padding (no dark borders from zero padding)."""
    ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((ph, ph), (pw, pw), (0, 0)), mode="reflect")
    out = np.stack([fftconvolve(padded[..., c], kernel, mode="valid") for c in range(image.shape[-1])], -1)
    return out[: image.shape[0], : image.shape[1]].astype(np.float32)


def blur_uniform(image: np.ndarray, kernel: np.ndarray, s: float, noise_sigma: float,
                 rng: np.random.Generator) -> BlurSample:
    b = convolve_rgb(image, kernel)
    b = np.clip(b + rng.normal(0.0, noise_sigma, b.shape).astype(np.float32), 0.0, 1.0)
    smap = np.full(image.shape[:2], s, dtype=np.float32)
    return BlurSample(b, smap, [kernel], [s], np.ones((1, *image.shape[:2]), np.float32))


def blur_nonuniform(image: np.ndarray, rng: np.random.Generator, scale: SeverityScale,
                    n_regions: int, s_range: tuple[float, float], noise_sigma: float = 0.01,
                    temperature: float | None = None, margin: int = 20,
                    targets: list[float] | None = None, curvature: float | None = None,
                    jitter: float | None = None) -> BlurSample:
    """Region-mixed blur. ``image`` should already include ``margin`` pixels
    of context on each side; the returned sample is cropped back.

    ``targets`` optionally fixes the target severity of each region
    (e.g. the out-of-distribution split forces one region above 0.75).
    """
    h, w = image.shape[:2]
    masks = soft_masks(rng, n_regions, h, w, temperature=temperature)
    if targets is None:
        targets = [rng.uniform(*s_range) for _ in range(n_regions)]
    kernels, sev = [], []
    acc = np.zeros_like(image, dtype=np.float32)
    for r in range(n_regions):
        k, s = kernel_for_severity(rng, targets[r], scale, KERNEL_SIZE, curvature=curvature, jitter=jitter)
        kernels.append(k)
        sev.append(s)
        acc += masks[r][..., None] * convolve_rgb(image, k)
    smap = np.tensordot(np.asarray(sev, np.float32), masks, axes=1)
    acc = np.clip(acc + rng.normal(0.0, noise_sigma, acc.shape).astype(np.float32), 0.0, 1.0)
    if margin:
        sl = (slice(margin, h - margin), slice(margin, w - margin))
        acc, smap, masks = acc[sl], smap[sl], masks[(slice(None), *sl)]
    return BlurSample(acc, smap.astype(np.float32), kernels, sev, masks)
