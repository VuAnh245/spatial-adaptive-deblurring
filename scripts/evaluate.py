"""Evaluate a trained checkpoint on a fixed split; writes one CSV row per image.

    python scripts/evaluate.py --config configs/experiments/SA-13_a2.yaml \
        --ckpt results/SA-13_a2_seed1/final.pth --split test_in [--mode pred|oracle|shuffle]
    python scripts/evaluate.py ... --split gopro_test          # SA-28 (full-resolution pairs)
    python scripts/evaluate.py ... --classical --split test_in  # SA-29 (R = 1 only, known kernel)

RealBlur-J (SA-28) must go through the official alignment, see metrics/realblur_eval.py.

Columns: ``psnr`` (float, primary, every model alike), ``psnr_u8`` (output rounded to uint8 as in the official
BasicSR evaluation, comparable with SA-00), ``video`` (source recording, QĐ-05) and ``video_in_train`` (the video
has another clip in the training split: known limitation (1) in docs/experiments.md).
The CSV is written to ``<out>.part`` and renamed when every image is done, so a cut job never looks complete.
``--blind`` prints no metric (QĐ-04 order: pair-2 numbers stay hidden until the estimator conclusion is written).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from spatial_deblur.data.datasets import FixedSyntheticDataset, PairedImageDataset
from spatial_deblur.data.manifests import read_list, video_of
from spatial_deblur.metrics.image_quality import local_psnr, psnr, ssim
from spatial_deblur.models.tlc import convert_tlc
from spatial_deblur.modulation.analysis import heterogeneity, map_quality, predict
from spatial_deblur.modulation.variants import build_variant
from spatial_deblur.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=False)
    ap.add_argument("--split", default="test_in")
    ap.add_argument("--mode", default="pred", choices=["pred", "oracle", "shuffle"])
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--pool", default="tlc", choices=["tlc", "global"],
                    help="tlc = primary protocol (docs/experiments.md); global = appendix, A0/A2 only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--blind", action="store_true", help="print progress only, no metric")
    a = ap.parse_args()
    cfg = load_config(a.config)
    m, d = cfg["model"], cfg["data"]
    out = Path(a.out or f"results/{cfg['experiment_id']}_eval_{a.split}_{a.mode}_{a.pool}.csv")
    train_videos = {video_of(p, depth=2) for p in read_list(d["sharp_train"])}

    if a.split == "gopro_test":
        ds = PairedImageDataset(d["gopro_root"], d["gopro_test_pairs"], patch=None, train=False)
    else:
        ds = FixedSyntheticDataset(Path(d["synthetic_root"]) / a.split)

    if a.classical:
        raise SystemExit("SA-29: run `python analysis/classical_baselines.py` (Wiener / RL, true kernel, R = 1)")

    kw = {k: m[k] for k in ("width", "middle_blk_num", "enc_blk_nums", "dec_blk_nums")}
    model = build_variant(m["variant"], kw)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    model = model.eval()
    if a.pool == "tlc":
        convert_tlc(model, train_size=d["patch"])
    model = model.to(a.device).eval()

    rows, t0 = [], time.time()
    torch.set_grad_enabled(False)
    for i in range(len(ds)):
        s = ds[i]
        x = s["blur"][None].to(a.device)
        smap = s.get("smap")
        if a.mode == "shuffle":
            j = (i + 1) % len(ds)  # use another image's estimated map
            s_other = model.estimator(model.pad(ds[j]["blur"][None].to(a.device)))  # = S_hat of forward()
            pred, s_hat = model(x, smap_override=s_other)
        else:
            pred, s_hat = predict(model, x, a.mode, smap[None].to(a.device) if smap is not None else None)
        p = pred[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        g = s["sharp"].permute(1, 2, 0).numpy()
        parts = Path(s["id"]).parts
        group = s.get("group") or (parts[1] if len(parts) > 1 else s["id"])
        video = video_of(group)
        row = {"id": s["id"], "group": group, "video": video, "video_in_train": int(video in train_videos),
               "psnr": psnr(p, g), "psnr_u8": psnr(np.round(p * 255.0) / 255.0, g), "ssim": ssim(p, g)}
        if smap is not None:
            sm = smap[0].numpy()
            row.update(regions=s["regions"], H=heterogeneity(sm), S_mean=float(sm.mean()),
                       psnr_local_high=local_psnr(p, g, sm >= np.quantile(sm, 2 / 3)) if sm.std() > 1e-4 else float("nan"))
            if s_hat is not None:
                row.update({f"map_{k}": v for k, v in map_quality(s_hat.cpu(), smap[None]).items()})
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(ds)}  {(time.time() - t0) / (i + 1):.2f}s/img", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with part.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    part.replace(out)
    tail = "" if a.blind else f"; mean PSNR {np.mean([r['psnr'] for r in rows]):.3f}"
    print(f"{len(rows)} images -> {out} ({time.time() - t0:.0f}s){tail}", flush=True)


if __name__ == "__main__":
    main()
