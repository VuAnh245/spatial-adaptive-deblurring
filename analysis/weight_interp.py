"""SA-36 weight interpolation of the Baseline (QĐ-12): does a model between the public NAFNet and the fine-tuned
Baseline keep both domains?

    theta(alpha) = (1 - alpha) * theta_public + alpha * theta_finetuned     alpha = 0 public (SA-00), 1 = SA-10 seed 1

A0 only: the public checkpoint and SA-10 have the same keys and shapes (no SFT head, no estimator to invent). Every
floating tensor is interpolated; a non-floating one must be equal in both. Evaluation is scripts/evaluate.py (TLC,
FP32, as every other row); the endpoints reuse the SA-00 / SA-10 CSVs of the same evaluation run.

    python analysis/weight_interp.py make --pre <public A0 .pth> --ft <SA-10 final.pth> --alpha 0.5 --out <.pth>
    python analysis/weight_interp.py summary --interp <dir of SA-36 CSVs> --ref <dir with SA-00 / SA-10 CSVs>

``make`` never overwrites a different file. ``summary`` reports the whole curve (no alpha is picked on test data):
mean PSNR on test_in and GoPro per alpha, the video-level difference to each endpoint (QĐ-05, bootstrap CI), and
whether the point lies above the chord joining the endpoints (better than mixing the two endpoints' scores).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import torch

SPLITS = ("test_in", "gopro_test")


def interpolate(pre: dict, ft: dict, alpha: float) -> dict:
    if set(pre) != set(ft):
        raise ValueError(f"key mismatch: {sorted(set(pre) ^ set(ft))[:5]}")
    out = {}
    for k, a in pre.items():
        b = ft[k]
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch at {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
        if a.is_floating_point():
            out[k] = ((1 - alpha) * a.double() + alpha * b.double()).to(a.dtype)
        elif torch.equal(a, b):
            out[k] = a.clone()
        else:
            raise ValueError(f"non-floating tensor {k} differs between the two checkpoints")
    return out


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make(a) -> None:
    assert 0.0 <= a.alpha <= 1.0, "alpha must be in [0, 1]"
    sd = interpolate(torch.load(a.pre, map_location="cpu"), torch.load(a.ft, map_location="cpu"), a.alpha)
    out = Path(a.out)
    part = out.with_name(out.name + ".part")
    torch.save(sd, part)
    if out.exists():
        same = sha256(out) == sha256(part)
        part.unlink()
        assert same, f"{out} exists with different content - not overwritten"
        print(f"exists, identical: {out}")
    else:
        part.replace(out)
    print(f"alpha {a.alpha:g} -> {out} sha256 {sha256(out)}")


def load_csv(path: Path) -> tuple[dict, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["id"]: float(r["psnr"]) for r in rows}, {r["id"]: r.get("video") or r["group"] for r in rows}


def summary(a) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from spatial_deblur.metrics.stats import group_mean, paired_comparison

    ref, interp = Path(a.ref), Path(a.interp)
    curves = {sp: {0.0: load_csv(ref / f"SA-00_eval_{sp}_pred_tlc.csv"), 1.0: load_csv(ref / f"SA-10_eval_{sp}_pred_tlc.csv")}
              for sp in SPLITS}
    for p in sorted(interp.glob("SA-36_a*_eval_*_pred_tlc.csv")):
        m = re.match(r"SA-36_a([0-9.]+)_eval_(.+)_pred_tlc\.csv", p.name)
        if m and m.group(2) in SPLITS:
            curves[m.group(2)][float(m.group(1))] = load_csv(p)
    alphas = sorted(set.intersection(*(set(c) for c in curves.values())))
    for sp, c in curves.items():
        ids = set(c[0.0][0])
        assert all(set(v[0]) == ids for v in c.values()), f"{sp}: CSVs cover different images"
    mean = {sp: {al: float(np.mean(list(curves[sp][al][0].values()))) for al in alphas} for sp in SPLITS}
    print("SA-36 (QĐ-12): theta = (1 - alpha) * public + alpha * Baseline seed 1; PSNR dB, TLC; video-level 95% CI")
    print(f"{'alpha':>6s} {'test_in':>8s} {'GoPro':>8s} | {'test_in vs SA-10 (a=1)':>34s} | {'GoPro vs SA-00 (a=0)':>34s} | chord")
    for al in alphas:
        line = f"{al:6.2f} {mean['test_in'][al]:8.3f} {mean['gopro_test'][al]:8.3f}"
        for sp, end in (("test_in", 1.0), ("gopro_test", 0.0)):
            if al == end:
                line += f" | {'(endpoint)':>34s}"
                continue
            s, v = curves[sp][al]
            r = paired_comparison(group_mean(s, v), group_mean(curves[sp][end][0], v))
            line += f" | {r['mean_diff']:+8.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p {r['wilcoxon_p']:.3f}"
        # chord: linear mix of the endpoint scores with the same alpha
        above = all(mean[sp][al] >= (1 - al) * mean[sp][0.0] + al * mean[sp][1.0] for sp in SPLITS)
        line += f" | {'above' if above else 'not above'}" if 0 < al < 1 else " | -"
        print(line)
    worst = {al: max(mean["test_in"][1.0] - mean["test_in"][al], mean["gopro_test"][0.0] - mean["gopro_test"][al])
             for al in alphas}
    print("largest loss to the better endpoint per alpha (dB): " + ", ".join(f"{al:g}: {w:.3f}" for al, w in worst.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--pre", required=True)
    m.add_argument("--ft", required=True)
    m.add_argument("--alpha", type=float, required=True)
    m.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--interp", required=True)
    s.add_argument("--ref", required=True)
    a = ap.parse_args()
    make(a) if a.cmd == "make" else summary(a)


if __name__ == "__main__":
    main()
