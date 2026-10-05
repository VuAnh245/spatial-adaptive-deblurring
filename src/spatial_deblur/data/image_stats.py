"""Per-image statistics for EDA and scenario groups.

Shared infrastructure. Thresholds for groups (low light, saturated, noisy,
large / small blur) are fixed after EDA and written to config BEFORE any
evaluation is run.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace, sobel
from skimage.restoration import estimate_sigma


def brightness(img: np.ndarray) -> float:
    """Mean luma (BT.601) of an sRGB image in [0, 1]."""
    return float((0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).mean())


def saturation_ratio(img: np.ndarray, threshold: float = 0.98) -> float:
    return float((img.max(axis=-1) > threshold).mean())


def noise_sigma(img: np.ndarray) -> float:
    return float(estimate_sigma(img, channel_axis=-1, average_sigmas=True))


def gradient_energy(img: np.ndarray) -> float:
    g = img.mean(-1)
    return float(np.mean(sobel(g, 0) ** 2 + sobel(g, 1) ** 2))


def blur_proxy(blur: np.ndarray, sharp: np.ndarray | None = None) -> float:
    """Gradient-energy ratio blur/sharp when a reference exists (lower = blurrier),
    otherwise Laplacian variance of the blurred image."""
    if sharp is not None:
        return gradient_energy(blur) / max(gradient_energy(sharp), 1e-12)
    return float(laplace(blur.mean(-1)).var())


def image_stats(blur: np.ndarray, sharp: np.ndarray | None = None) -> dict[str, float]:
    return {"brightness": brightness(blur), "saturation": saturation_ratio(blur),
            "noise_sigma": noise_sigma(blur), "blur_proxy": blur_proxy(blur, sharp)}
