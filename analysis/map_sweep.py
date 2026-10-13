"""SA-34 map-quality sweep (QĐ-11): how good must the blur map be for the SFT path to help?

Evaluation only, no training. The SFT-Map trained with the true map (SA-32) is run on every test image with several
maps fed to its SFT heads (all at the estimator's 1/4 resolution, as the SA-20 oracle):
  true            down4(S)
  mix_<a>         a * down4(S) + (1 - a) * mean(down4(S))      a in 0.75, 0.5, 0.25, 0 (0 = only the image's level)
  noise_<sigma>   clip(down4(S) + N(0, sigma), 0, 1)          sigma in 0.05, 0.1, 0.2 (seeded per image)
  prior           0.375 everywhere (mean of s ~ U(0, 0.75); no image information)
  est_v2          S_hat of the SA-33 estimator (BlurMapEstimatorV2): a two-stage model usable without S
  est_v1          S_hat of the SA-13 estimator (original, near-constant): control
Per image and map: PSNR (float, TLC), Spearman and MAE of the map against down4(S) (R >= 2 only for Spearman).
Writes <out>/map_sweep.csv; a summary per map is printed (mean PSNR, mean Spearman, MAE).

    python analysis/map_sweep.py --sa32 <final.pth> --sa33 <final.pth> --sa13 <final.pth> --split test_in --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from spatial_deblur.data.datasets import FixedSyntheticDataset
from spatial_deblur.metrics.image_quality import psnr
from spatial_deblur.models.tlc import convert_tlc
from spatial_deblur.modulation.variants import apply_model_options, build_variant
from spatial_deblur.utils.config import load_config

MIX = (0.75, 0.5, 0.25, 0.0)
NOISE = (0.05, 0.1, 0.2)


def load(config: str, ckpt: str, device: str):
    m = load_config(config)["model"]
    kw = {k: m[k] for k in ("width", "middle_blk_num", "enc_blk_nums", "dec_blk_nums")}
    model = apply_model_options(build_variant(m["variant"], kw, estimator=m.get("estimator", "v1")), m)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa32", required=True)
    ap.add_argument("--sa33", required=True)
    ap.add_argument("--sa13", required=True)
    ap.add_argument("--split", default="test_in")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    d = load_config("configs/experiments/SA-32_a2_oracle_train.yaml")["data"]
    model = load("configs/experiments/SA-32_a2_oracle_train.yaml", a.sa32, "cpu")
    convert_tlc(model, train_size=d["patch"])
    model = model.to(a.device)
    est_v2 = load("configs/experiments/SA-33_a2_estimator_v2.yaml", a.sa33, "cpu").estimator.to(a.device).eval()
    est_v1 = load("configs/experiments/SA-13_a2.yaml", a.sa13, "cpu").estimator.to(a.device).eval()
    ds = FixedSyntheticDataset(Path(d["synthetic_root"]) / a.split)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i in range(len(ds)):
        s = ds[i]
        x = s["blur"][None].to(a.device)
        g = s["sharp"].permute(1, 2, 0).numpy()
        s4 = F.interpolate(s["smap"][None].to(a.device), scale_factor=0.25, mode="area")
        gen = torch.Generator(device="cpu").manual_seed(i)
        maps = {"true": s4}
        maps.update({f"mix_{m:g}": m * s4 + (1 - m) * s4.mean() for m in MIX})
        maps.update({f"noise_{n:g}": (s4 + n * torch.randn(s4.shape, generator=gen).to(a.device)).clamp(0, 1)
                     for n in NOISE})
        maps["prior"] = torch.full_like(s4, 0.375)
        xp = model.pad(x)
        maps["est_v2"] = est_v2(xp)
        maps["est_v1"] = est_v1(xp)
        t4 = s4.flatten().cpu().numpy()
        for name, mp in maps.items():
            pred, _ = model(x, smap_override=mp)
            p = pred[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            mv = F.interpolate(mp, size=s4.shape[-2:], mode="bilinear", align_corners=False).flatten().cpu().numpy()
            rho = float(spearmanr(mv, t4).statistic) if (s["regions"] > 1 and mv.std() > 1e-6) else float("nan")
            rows.append({"id": s["id"], "video": s["group"].rsplit("_", 1)[0], "regions": s["regions"], "map": name,
                         "psnr": psnr(p, g), "map_spearman": rho, "map_mae": float(np.abs(mv - t4).mean())})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}  {(time.time() - t0) / (i + 1):.2f}s/img", flush=True)
    path = out / "map_sweep.csv"
    with path.with_name(path.name + ".part").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    path.with_name(path.name + ".part").replace(path)
    print(f"{len(ds)} images x {len(maps)} maps -> {path} ({time.time() - t0:.0f}s)")
    for name in maps:
        r = [x for x in rows if x["map"] == name]
        rho = [x["map_spearman"] for x in r if np.isfinite(x["map_spearman"])]
        print(f"{name:12s} PSNR {np.mean([x['psnr'] for x in r]):.4f}  map Spearman {np.mean(rho) if rho else float('nan'):.3f}"
              f"  MAE {np.mean([x['map_mae'] for x in r]):.3f}", flush=True)


if __name__ == "__main__":
    main()
