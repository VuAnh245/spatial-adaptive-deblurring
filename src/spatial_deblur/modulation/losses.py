"""Loss for the spatial-adaptation study.

    L = Charbonnier(I_hat, I) + lambda_map * SmoothL1(S_hat, down4(S))

lambda_map = 0.1 for A2/A3, 0 for A1 (same architecture), unused for A0/A0+.
No perceptual or FFT term in architecture comparisons (optional SA-18/19 only).
Gradient from the restoration loss flows into the estimator (no detach);
SA-17 sets ``ModulatedNAFNet.detach_map = True``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def oracle_map(smap: torch.Tensor) -> torch.Tensor:
    """True S at the estimator's 1/4 resolution, exactly as the SA-20 oracle of evaluate.py / analysis.predict."""
    return F.interpolate(smap, scale_factor=0.25, mode="area")


def make_loss(lambda_map: float = 0.1, map_source: str = "pred"):
    """``map_source='oracle'`` (QĐ-10, SA-32): the SFT heads receive the true S (oracle_map) instead of S_hat
    during training; use lambda_map = 0 (the estimator is then unused and untrained)."""
    if map_source not in ("pred", "oracle"):
        raise ValueError(f"map_source must be 'pred' or 'oracle', got {map_source!r}")

    def loss_fn(model, batch):
        if map_source == "oracle":
            out = model(batch["blur"], smap_override=oracle_map(batch["smap"]))
        else:
            out = model(batch["blur"])
        pred, s_hat = out if isinstance(out, tuple) else (out, None)
        l_rec = charbonnier(pred, batch["sharp"])
        parts = {"charbonnier": l_rec.detach()}
        loss = l_rec
        if s_hat is not None and "smap" in batch:
            # L_map is always logged; it only enters the loss when lambda_map > 0
            # (for SFT-Free it shows whether the free map drifts towards S at all).
            target = F.interpolate(batch["smap"], size=s_hat.shape[-2:], mode="area")
            l_map = F.smooth_l1_loss(s_hat.float(), target.float())
            if lambda_map > 0:
                loss = loss + lambda_map * l_map
            parts["l_map"] = l_map.detach()
        parts["loss"] = loss.detach()
        return loss, parts
    return loss_fn
