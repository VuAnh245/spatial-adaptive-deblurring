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


@torch.no_grad()
def val_metrics(model, val_set, device: str = "cpu", max_images: int = 50) -> dict[str, float]:
    """Monitoring metrics of a val pass during training (train.py): PSNR on every image; for models with
    a map, MAE of S_hat on every image and Spearman(S_hat, S) only on images with R >= 2 (undefined for a
    constant map), averaged with nanmean."""
    from ..metrics.image_quality import psnr
    ps, sp, mae = [], [], []
    for i in range(0, len(val_set), max(1, len(val_set) // max_images)):
        s = val_set[i]
        pred, s_hat = model(s["blur"][None].to(device))
        ps.append(psnr(pred[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy(),
                       s["sharp"].permute(1, 2, 0).numpy()))
        if s_hat is not None:
            q = map_quality(s_hat.float().cpu(), s["smap"][None])
            mae.append(q["mae"])
            if s["regions"] > 1:
                sp.append(q["spearman"])
    out = {"val_psnr": float(np.mean(ps))}
    if mae:
        out.update(val_map_mae=float(np.mean(mae)),
                   val_spearman=float(np.nanmean(sp)) if sp and not np.all(np.isnan(sp)) else float("nan"))
    return out


def make_train_diag():
    """Training diagnostics for Trainer(diag_fn=...), logged every ``diag_every`` steps to diag.jsonl:
    ||W|| of every SFT head's last conv (zero at init: shows whether the heads leave zero at all),
    mean |delta_gamma_s| and |beta_s| per stage on that step's batch. L_map comes in separately
    through the loss parts. Variants without SFT heads (Baseline, Baseline-L) log nothing here."""
    def diag(model, phase: str):
        if getattr(model, "sft", None) is None:
            return None
        if phase == "begin":
            model.mod_stats = {}
            return None
        stats, model.mod_stats = model.mod_stats or {}, None
        mid = model.middle_blks[0]  # its residual scales gate how much the mid0 modulation can matter
        row = {"mid_beta_abs": float(mid.beta.detach().abs().mean()), "mid_gamma_abs": float(mid.gamma.detach().abs().mean())}
        for key, head in model.sft.items():
            row[f"w_sft_{key}"] = float(head.conv2.weight.detach().norm())
            if key in stats:
                row[f"dgamma_{key}"], row[f"beta_{key}"] = stats[key]
        return row
    return diag
