"""Mechanism analyses on trained checkpoints (no training).

SA-20 oracle (S instead of S_hat), SA-21 shuffled S_hat within the batch,
SA-22 quality of S_hat (MAE, RMSE, pixel Spearman).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


@torch.no_grad()
def predict(model, blur: torch.Tensor, mode: str = "pred", smap: torch.Tensor | None = None):
    """mode: 'pred' (S_hat), 'oracle' (true S), 'shuffle' (S_hat rolled across the batch)."""
    if mode == "pred":
        return model(blur)
    if mode == "oracle":
        s4 = F.interpolate(smap, scale_factor=0.25, mode="area")
        return model(blur, smap_override=s4)
    if mode == "shuffle":
        _, s_hat = model(blur)
        return model(blur, smap_override=torch.roll(s_hat, 1, dims=0))
    raise ValueError(mode)


def map_quality(s_hat: torch.Tensor, smap: torch.Tensor, max_pixels: int = 200_000) -> dict[str, float]:
    up = F.interpolate(s_hat, size=smap.shape[-2:], mode="bilinear", align_corners=False)
    a, b = up.flatten().cpu().numpy(), smap.flatten().cpu().numpy()
    if a.size > max_pixels:
        idx = np.random.default_rng(0).choice(a.size, max_pixels, replace=False)
        a, b = a[idx], b[idx]
    rho = float(spearmanr(a, b).statistic) if (a.std() > 1e-6 and b.std() > 1e-6) else float("nan")
    # Spearman is undefined for a constant map (R = 1); report MAE / RMSE there.
    return {"mae": float(np.abs(a - b).mean()), "rmse": float(np.sqrt(((a - b) ** 2).mean())), "spearman": rho}


def heterogeneity(smap: np.ndarray) -> float:
    """H = std(S) over the image (bins chosen after EDA)."""
    return float(np.std(smap))
