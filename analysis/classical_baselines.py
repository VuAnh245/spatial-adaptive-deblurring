"""SA-29: Wiener and Richardson-Lucy with the TRUE kernel on R = 1 images.

    python analysis/classical_baselines.py --synthetic data/synthetic

Parameters (Wiener balance, RL iterations) are chosen on the val split, then
applied once to test_in. Output: results/SA-29_classical.csv (one row per image).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from spatial_deblur.classical.deconvolution import richardson_lucy_rgb, wiener_rgb
from spatial_deblur.metrics.image_quality import psnr, ssim


def r1_files(folder: Path):
    for f in sorted(folder.glob("*.npz")):
        z = np.load(f)
        if int(z["regions"]) == 1:
            yield f.stem, z


def tune(folder: Path, fn, grid, limit=40):
    scores = {g: [] for g in grid}
    for i, (_, z) in enumerate(r1_files(folder)):
        if i >= limit:
            break
        for g in grid:
            scores[g].append(psnr(fn(z["blur"], z["kernels"][0], g), z["sharp"]))
    return max(grid, key=lambda g: np.mean(scores[g]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="data/synthetic")
    a = ap.parse_args()
    root = Path(a.synthetic)
    bal = tune(root / "val", wiener_rgb, [1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    its = tune(root / "val", richardson_lucy_rgb, [10, 20, 30, 50])
    rows = []
    for stem, z in r1_files(root / "test_in"):
        k = z["kernels"][0]
        for name, out in (("wiener", wiener_rgb(z["blur"], k, bal)), ("richardson_lucy", richardson_lucy_rgb(z["blur"], k, its))):
            rows.append({"id": stem, "group": str(z["group"]), "method": name, "psnr": psnr(out, z["sharp"]),
                         "ssim": ssim(out, z["sharp"]), "S_mean": float(z["smap"].mean())})
    Path("results").mkdir(exist_ok=True)
    with open("results/SA-29_classical.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wiener balance={bal}, RL iterations={its}, {len(rows)} rows")


if __name__ == "__main__":
    main()
