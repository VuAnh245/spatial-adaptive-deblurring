"""Generate the fixed synthetic test sets once (seeded) and store them as .npz.

    python scripts/generate_testsets.py --config configs/experiments/SA-13_a2.yaml [--noise-mult 2]

Splits: val, test_in (300 per R in {1,2,3,4}), test_ood (one region s > 0.75),
test_straight / test_curved (R = 1, same severity bins). Full resolution.
Each file stores blur, sharp, smap, regions, severities, kernels and the
source sequence (group) used for grouped statistics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spatial_deblur.data.datasets import load_rgb
from spatial_deblur.data.manifests import read_list, write_list
from spatial_deblur.degradations.blur import blur_nonuniform
from spatial_deblur.degradations.kernels import SeverityScale
from spatial_deblur.utils.config import load_config


def make(img, rng, scale, r, s_range, noise, targets=None, curvature=None, jitter=None, margin=20):
    padded = np.pad(img, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
    return blur_nonuniform(padded, rng, scale, r, s_range, noise, margin=margin, targets=targets,
                           curvature=curvature, jitter=jitter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--noise-mult", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = cfg["data"]
    scale = SeverityScale(**d["severity_scale"])
    root = Path(d["gopro_root"])
    noise = d["noise_sigma"] * a.noise_mult
    suffix = "" if a.noise_mult == 1 else f"_noise{a.noise_mult:g}x"
    out_root = Path(d["synthetic_root"])
    lo, hi = d["s_train"]
    plans = {
        "val": (read_list(d["sharp_val"]), [(r, d["testsets"]["val"]["per_region"]) for r in d["regions"]], "in"),
        "test_in": (read_list(d["sharp_test"]), [(r, d["testsets"]["test_in"]["per_region"]) for r in d["regions"]], "in"),
        "test_ood": (read_list(d["sharp_test"]), [(0, d["testsets"]["test_ood"]["count"])], "ood"),
        "test_straight": (read_list(d["sharp_test"]), [(1, d["testsets"]["test_straight"]["count"])], "straight"),
        "test_curved": (read_list(d["sharp_test"]), [(1, d["testsets"]["test_curved"]["count"])], "curved"),
    }
    for si, (split, (pool, groups, kind)) in enumerate(plans.items()):
        if a.noise_mult != 1 and split == "val":
            continue
        folder = out_root / (split + suffix)
        folder.mkdir(parents=True, exist_ok=True)
        idx = 0
        for gi, (r, count) in enumerate(groups):
            # spread images evenly across sequences
            picks = np.linspace(0, len(pool) - 1, count).astype(int)
            for j, p in enumerate(picks):
                rng = np.random.default_rng((a.seed, si, gi, j))  # never use hash(): salted per process
                img = load_rgb(root / pool[p])
                kw = {}
                rr = r
                if kind == "ood":
                    rr = int(rng.integers(1, 5))
                    kw["targets"] = [rng.uniform(*d["s_ood"])] + [rng.uniform(lo, hi) for _ in range(rr - 1)]
                elif kind == "straight":
                    kw.update(curvature=0.0, jitter=0.0)
                elif kind == "curved":
                    kw.update(curvature=0.5)
                s = make(img, rng, scale, rr, (lo, hi), noise, **kw)
                np.savez_compressed(folder / f"{idx:05d}.npz", blur=s.blur, sharp=img, smap=s.severity_map,
                                    regions=rr, severities=np.array(s.severities), kernels=np.stack(s.kernels),
                                    group=Path(pool[p]).parts[1],
                                    source=pool[p])
                idx += 1
        files = sorted(str(f) for f in folder.glob("*.npz"))
        write_list(f"manifests/synthetic_{split}{suffix}.txt", files)
        print(f"{split}{suffix}: {idx} images")


if __name__ == "__main__":
    main()
