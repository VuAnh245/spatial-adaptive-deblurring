"""Diagnosis of the trained blur-map estimator E (QĐ-04 stop rule: val_spearman ~0 at step 10k).

Runs ONLY ``model.estimator`` of a trained checkpoint on the fixed val set (CPU, no restoration pass) and asks
which of three things E learned:
  (1) nothing but the dataset mean        -> flat calibration, per-image mean of S_hat ~ constant
  (2) the per-image blur LEVEL only        -> mean S_hat tracks mean S across images, within-image std ~ 0
  (3) the spatial map                      -> calibration rises and within-image Spearman > 0
Reference row: a constant predictor equal to the val mean of S (an optimistic "learned the mean" baseline).

    python analysis/estimator_diag.py --ckpt <run>/final.pth --config configs/experiments/SA-13_a2.yaml
Writes <out>/estimator_diag.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from spatial_deblur.data.datasets import FixedSyntheticDataset
from spatial_deblur.modulation.analysis import map_quality
from spatial_deblur.modulation.variants import build_variant
from spatial_deblur.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="final.pth (state_dict) or last.pth (dict with 'model')")
    ap.add_argument("--config", default="configs/experiments/SA-13_a2.yaml")
    ap.add_argument("--variant", default="A2")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--out", default=None, help="default: folder of --ckpt")
    a = ap.parse_args()

    d = load_config(a.config)["data"]
    val_set = FixedSyntheticDataset(Path(d["synthetic_root"]) / "val")
    assert len(val_set) > 0, "no val set - run scripts/generate_testsets.py --splits val first"
    model = build_variant(a.variant)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"] if "model" in st else st)
    est = model.estimator.eval()

    rows, s_all, h_all = [], [], []
    with torch.no_grad():
        for i in range(len(val_set)):
            s = val_set[i]
            smap = s["smap"][None]
            s_hat = est(model.pad(s["blur"][None]))
            up = F.interpolate(s_hat, size=smap.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
            gt = smap[0, 0].numpy()
            q = map_quality(s_hat, smap)
            rows.append({"id": s["id"], "regions": s["regions"], "s_mean": float(gt.mean()), "s_std": float(gt.std()),
                         "s_hat_mean": float(up.mean()), "s_hat_std": float(up.std()),
                         "mae": q["mae"], "spearman": q["spearman"] if s["regions"] > 1 else None})
            idx = np.random.default_rng(i).choice(gt.size, min(gt.size, 20_000), replace=False)
            s_all.append(gt.ravel()[idx])
            h_all.append(up.ravel()[idx])

    s_all, h_all = np.concatenate(s_all), np.concatenate(h_all)
    const = float(np.mean([r["s_mean"] for r in rows]))
    const_mae = float(np.mean([np.abs(val_set[i]["smap"].numpy() - const).mean() for i in range(len(val_set))]))
    edges = np.linspace(0.0, 1.0, a.bins + 1)
    calib = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s_all >= lo) & ((s_all < hi) if hi < 1.0 else (s_all <= hi))
        if m.any():
            calib.append({"S_bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "S_mean": float(s_all[m].mean()),
                          "S_hat_mean": float(h_all[m].mean()), "S_hat_std": float(h_all[m].std())})
    multi = [r for r in rows if r["regions"] > 1]
    summary = {
        "ckpt": a.ckpt, "images": len(rows), "images_R_ge_2": len(multi),
        "mae_model": float(np.mean([r["mae"] for r in rows])),
        "mae_constant_val_mean": const_mae, "constant": const,
        "spearman_within_image_R_ge_2": float(np.nanmean([r["spearman"] for r in multi])) if multi else None,
        "spearman_pixels_pooled": float(spearmanr(s_all, h_all).statistic),
        "spearman_image_means": float(spearmanr([r["s_mean"] for r in rows], [r["s_hat_mean"] for r in rows]).statistic),
        "s_hat_mean_range": [min(r["s_hat_mean"] for r in rows), max(r["s_hat_mean"] for r in rows)],
        "s_mean_range": [min(r["s_mean"] for r in rows), max(r["s_mean"] for r in rows)],
        "within_std_ratio_R_ge_2": float(np.mean([r["s_hat_std"] for r in multi]) / np.mean([r["s_std"] for r in multi]))
        if multi else None,
        "calibration": calib,
    }
    out = Path(a.out) if a.out else Path(a.ckpt).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "estimator_diag.json").write_text(json.dumps({"summary": summary, "per_image": rows}, indent=1))
    print("=== ESTIMATOR DIAG ===")
    for k, v in summary.items():
        if k != "calibration":
            print(f"{k:32s} {v}")
    print("calibration (pixels pooled over val):")
    for c in calib:
        print(f"  S {c['S_bin']}  n {c['n']:7d}  S {c['S_mean']:.3f}  S_hat {c['S_hat_mean']:.3f} +- {c['S_hat_std']:.3f}")


if __name__ == "__main__":
    main()
