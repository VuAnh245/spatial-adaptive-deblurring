"""Why the blur-map estimator E learns nothing (diagnostic, CPU, 2026-10-13; run inline once, saved here to reproduce).

Patches come from the SAME generator as training (blur_nonuniform, s in [0, 0.75], noise 0.01, square grid 3..6)
applied to the sharp frames stored in the original val40 NPZ files (no GoPro download needed):
  (1) label check: Spearman of the label s (R = 1) with the kernel RMS radius and with a hand-made feature computed
      on the BLURRED patch only (mean |horizontal gradient|) -> is the blur level visible in the input at all?
  (2) learnability: the study's BlurMapEstimator and an exploratory variant with the same width (input normalised,
      GroupNorm, Kaiming init, stride-1 first conv) are trained on 256 fixed patches with L_map (SmoothL1, AdamW
      lr 1e-3, batch 8, 600 steps) and scored on 64 held-out patches (pixel and patch-level Spearman, MAE),
      for R = 1 and for R in 1..4.
Results 2026-10-13 (local CPU, torch 2.13, numpy 2.4): (1) rho(s, radius) = 1.00, rho(s, blur gradient) = -0.54;
(2) study E: rho ~ 0 in both settings, test loss not below the constant map; exploratory E: R = 1 rho 0.82,
R in 1..4 rho_pix 0.56 / rho_patch 0.70, test loss 0.0142 vs 0.0205 for the constant map.
The exploratory variant is NOT part of any trained model; using it in training would be a new decision.

    python analysis/estimator_learnability.py --val-dir <folder with the original val40 .npz>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from spatial_deblur.degradations.blur import blur_nonuniform
from spatial_deblur.degradations.kernels import SeverityScale, rms_radius
from spatial_deblur.modulation.modules import BlurMapEstimator, BlurMapEstimatorV2


class ExploratoryEstimator(nn.Module):
    """Same width as BlurMapEstimator; normalised input, GroupNorm, Kaiming init, stride-1 first conv."""

    def __init__(self, hidden: int = 64):
        super().__init__()

        def blk(i, o, s):
            return [nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU()]

        self.net = nn.Sequential(*blk(3, 32, 1), *blk(32, 64, 2), *blk(64, hidden, 2), *blk(hidden, hidden, 1),
                                 *blk(hidden, hidden, 1), nn.Conv2d(hidden, 1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.sigmoid(self.net((x - 0.5) / 0.25))


def patches(sharps, n, regions, seed, scale, patch=256, margin=20):
    rng = np.random.default_rng(seed)
    blur, smap, extra = [], [], []
    for _ in range(n):
        img = sharps[int(rng.integers(len(sharps)))]
        big = patch + 2 * margin
        y, x = int(rng.integers(0, img.shape[0] - big)), int(rng.integers(0, img.shape[1] - big))
        crop = img[y:y + big, x:x + big]
        r, g = int(rng.choice(regions)), int(rng.choice([3, 4, 5, 6]))
        s = blur_nonuniform(crop, rng, scale, r, (0.0, 0.75), 0.01, margin=margin, grid=(g, g))
        blur.append(s.blur)
        smap.append(s.severity_map)
        extra.append((s.severities[0], rms_radius(s.kernels[0]), float(np.abs(np.diff(s.blur.mean(2), axis=1)).mean())))
    return (torch.from_numpy(np.stack(blur)).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(np.stack(smap))[:, None], np.array(extra))


def train_and_score(make_model, data_tr, data_te, steps=600, seed=0):
    torch.manual_seed(seed)
    est = make_model()
    opt = torch.optim.AdamW(est.parameters(), lr=1e-3, betas=(0.9, 0.9), weight_decay=1e-3)
    (btr, s_tr), (bte, s_te) = data_tr, data_te
    down = lambda s, h: F.interpolate(s, size=h.shape[-2:], mode="area")  # noqa: E731
    for _ in range(steps):
        idx = torch.randint(0, len(btr), (8,))
        sh = est(btr[idx])
        loss = F.smooth_l1_loss(sh, down(s_tr[idx], sh))
        opt.zero_grad()
        loss.backward()
        opt.step()
    est.eval()
    with torch.no_grad():
        sh = est(bte)
        t = down(s_te, sh)
    return {"test_loss": F.smooth_l1_loss(sh, t).item(),
            "const_loss": F.smooth_l1_loss(torch.full_like(t, float(s_tr.mean())), t).item(),
            "mae": (sh - t).abs().mean().item(),
            "rho_pix": float(spearmanr(sh.flatten().numpy(), t.flatten().numpy()).statistic),
            "rho_patch": float(spearmanr(sh.mean((1, 2, 3)).numpy(), t.mean((1, 2, 3)).numpy()).statistic),
            "std": sh.std().item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--models", nargs="+", default=["study", "exploratory"], choices=["study", "exploratory", "v2"],
                    help="v2 = BlurMapEstimatorV2, the module SA-33 trains with (QĐ-10 gate C1)")
    a = ap.parse_args()
    candidates = {"study": ("study E", BlurMapEstimator), "exploratory": ("exploratory E", ExploratoryEstimator),
                  "v2": ("estimator v2", BlurMapEstimatorV2)}
    torch.set_num_threads(8)
    scale = SeverityScale(0.0, 8.0)
    sharps = [np.load(f)["sharp"].astype(np.float32) for f in sorted(Path(a.val_dir).glob("*.npz"))[:32]]
    _, _, ex = patches(sharps, 160, [1], 3, scale)
    print(f"(1) R=1: rho(s, kernel RMS radius) {spearmanr(ex[:, 0], ex[:, 1]).statistic:+.2f} | "
          f"rho(s, blur gradient energy) {spearmanr(ex[:, 0], ex[:, 2]).statistic:+.2f}", flush=True)
    t0 = time.time()
    for name, regions in [("R=1", [1]), ("R in 1..4", [1, 2, 3, 4])]:
        tr, te = patches(sharps, 256, regions, 1, scale), patches(sharps, 64, regions, 2, scale)
        for label, make in (candidates[k] for k in a.models):
            r = train_and_score(make, tr[:2], te[:2], a.steps)
            print(f"(2) {name:10s} {label:14s} test loss {r['test_loss']:.4f} (const {r['const_loss']:.4f}) MAE {r['mae']:.3f} "
                  f"rho_pix {r['rho_pix']:+.2f} rho_patch {r['rho_patch']:+.2f} std {r['std']:.3f} | {time.time() - t0:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
