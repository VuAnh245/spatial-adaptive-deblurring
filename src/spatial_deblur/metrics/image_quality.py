"""Image-quality metrics on full-resolution RGB images in [0, 1].

Shared infrastructure. One implementation for every model, own-trained or
public checkpoint. Numbers taken from papers are never mixed into tables
produced by this module.

RealBlur: PSNR/SSIM must be computed after the official alignment
(``realblur_eval.py``), never with these functions directly.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    return float(structural_similarity(pred, target, channel_axis=-1, data_range=data_range))


def local_psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """PSNR restricted to pixels where ``mask`` is True (e.g. top tertile of S)."""
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    diff = (pred.astype(np.float64) - target.astype(np.float64))[m]
    mse = float(np.mean(diff ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


class LPIPSMetric:
    """LPIPS with the AlexNet backbone for every experiment (lazy import)."""

    def __init__(self, device: str = "cpu"):
        import lpips  # pip install lpips
        import torch

        self._torch = torch
        self.device = device
        self.net = lpips.LPIPS(net="alex").to(device).eval()

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        t = self._torch
        to = lambda a: t.from_numpy(a).permute(2, 0, 1)[None].float().to(self.device) * 2 - 1
        with t.no_grad():
            return float(self.net(to(pred), to(target)).item())
