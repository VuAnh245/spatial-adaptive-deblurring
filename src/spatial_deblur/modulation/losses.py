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


def make_loss(lambda_map: float = 0.1):
    def loss_fn(model, batch):
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
