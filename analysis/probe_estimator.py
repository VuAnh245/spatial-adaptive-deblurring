"""Estimator probe: can the blur-map estimator E learn the SPATIAL map with this data generator?

Trains ONLY ``BlurMapEstimator`` with L_map = SmoothL1(S_hat, down4(S)) on the training generator
(SyntheticNonUniformDataset, same config: patches, region masks, severities, noise), batch 8, and logs
MAE and Spearman(S_hat, S) on the fixed val set every ``--val-every`` steps (Spearman on R >= 2 only).
Independent of the main trainer and of NAFNet; runs on CPU.

    python analysis/probe_estimator.py --config configs/experiments/SA-13_a2.yaml --steps 1000

Reading: Spearman rising clearly above 0 means the map is learnable from the blurred image alone, so a flat
val_spearman in SFT-Map training points to the training setup (loss weight, gradient competition, time),
not to the data. Spearman staying near 0 here points to the estimator or the data/labels.
Writes results/probe_E/{log.jsonl, summary.txt}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spatial_deblur.data.datasets import FixedSyntheticDataset, SyntheticNonUniformDataset
from spatial_deblur.degradations.kernels import SeverityScale
from spatial_deblur.modulation.analysis import map_quality
from spatial_deblur.modulation.modules import BlurMapEstimator
from spatial_deblur.utils.config import load_config


@torch.no_grad()
def evaluate(est, val_set, device) -> dict[str, float]:
    est.eval()
    mae, sp = [], []
    for i in range(len(val_set)):
        s = val_set[i]
        s_hat = est(s["blur"][None].to(device))
        q = map_quality(s_hat.cpu(), s["smap"][None])
        mae.append(q["mae"])
        if s["regions"] > 1:  # Spearman is undefined for a constant map
            sp.append(q["spearman"])
    est.train()
    return {"val_map_mae": float(np.mean(mae)), "val_spearman": float(np.nanmean(sp)) if sp else float("nan"),
            "val_s_hat_std": float(s_hat.std())}  # spread of the last prediction: ~0 means a constant map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/SA-13_a2.yaml")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="default: training lr from the config")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/probe_E")
    a = ap.parse_args()

    cfg = load_config(a.config)
    d, t = cfg["data"], cfg["training"]
    lr = a.lr if a.lr is not None else t["lr"]
    torch.manual_seed(a.seed)
    ds = SyntheticNonUniformDataset(d["gopro_root"], d["sharp_train"], SeverityScale(**d["severity_scale"]),
                                    d["s_train"], d["regions"], d["patch"], d["margin"], d["noise_sigma"],
                                    length=a.steps * a.batch, seed=a.seed + 1000,  # not a training seed
                                    mask_grids=d["mask_grid_train"])
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, drop_last=True)
    val_set = FixedSyntheticDataset(Path(d["synthetic_root"]) / "val")
    assert len(val_set) > 0, "no val set - run scripts/generate_testsets.py --splits val first"

    est = BlurMapEstimator().to(a.device)
    opt = torch.optim.AdamW(est.parameters(), lr=lr, betas=tuple(t["betas"]), weight_decay=t["weight_decay"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("w", encoding="utf-8")

    def record(step, extra):
        row = {"step": step, **evaluate(est, val_set, a.device), **extra}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        return row

    print(f"probe E: {sum(p.numel() for p in est.parameters())} params, lr {lr}, batch {a.batch}, "
          f"{a.steps} steps, device {a.device}, val {len(val_set)} images", flush=True)
    rows, t0, running = [record(0, {"train_l_map": None, "sec": 0.0})], time.time(), []
    for step, batch in enumerate(loader, start=1):
        x, smap = batch["blur"].to(a.device), batch["smap"].to(a.device)
        s_hat = est(x)
        target = F.interpolate(smap, size=s_hat.shape[-2:], mode="area")
        loss = F.smooth_l1_loss(s_hat, target)  # same L_map as the study loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running.append(float(loss))
        if step % a.val_every == 0 or step == a.steps:
            rows.append(record(step, {"train_l_map": float(np.mean(running)), "sec": round(time.time() - t0, 1)}))
            running = []
        if step >= a.steps:
            break
    log.close()

    first, last = rows[0], rows[-1]
    text = (f"probe E, lr {lr}, {a.steps} steps: val Spearman {first['val_spearman']:.3f} -> {last['val_spearman']:.3f}, "
            f"val MAE {first['val_map_mae']:.3f} -> {last['val_map_mae']:.3f}, last S_hat std {last['val_s_hat_std']:.4f}")
    (out / "summary.txt").write_text(text + "\n")
    print("\n=== PROBE SUMMARY ===\n" + text, flush=True)


if __name__ == "__main__":
    main()
