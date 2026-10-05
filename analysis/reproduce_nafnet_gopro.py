"""SA-00: reproduce the public NAFNet-GoPro-width32 checkpoint on GoPro test.

Gate before any training: the official number (32.87 dB, measured with TLC /
NAFNetLocal) must be reproduced within 0.3 dB. Both global pooling and TLC
are evaluated in one pass so the gap can be reported.

    python analysis/reproduce_nafnet_gopro.py              # all 1,111 pairs
    python analysis/reproduce_nafnet_gopro.py --limit 20   # quick smoke test

Writes results/SA-00_reproduce/{gopro_test_nafnet32_public.csv, summary.txt, meta.txt};
the CSV is rewritten every 100 images so an interrupted run keeps its rows.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from spatial_deblur.data.datasets import load_rgb
from spatial_deblur.engine.inference import restore_full
from spatial_deblur.metrics.image_quality import psnr, ssim
from spatial_deblur.models.public_checkpoints import load_nafnet_public


def q8(a: np.ndarray) -> np.ndarray:
    """Official BasicSR evaluation works on uint8 images."""
    return np.round(a * 255.0) / 255.0


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict]) -> str:
    lines = [f"images: {len(rows)}", "published (TLC): 32.87 dB / 0.9606"]
    for k in ["psnr_blur", "psnr_u8_global", "psnr_u8_tlc", "psnr_float_tlc"]:
        lines.append(f"{k:16s} {np.mean([r[k] for r in rows]):.3f} dB")
    for k in ["ssim_skimage_tlc", "ssim_gauss_tlc"]:
        v = [r[k] for r in rows if k in r]
        lines.append(f"{k:16s} {np.mean(v):.4f}  (n={len(v)}, every 5th image)")
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("psnr_u8_global", "psnr_u8_tlc"):
            by[r["group"]][k].append(r[k])
    lines.append("per sequence  global -> TLC:")
    for g in sorted(by):
        lines.append(f"  {g}: {np.mean(by[g]['psnr_u8_global']):.2f} -> "
                     f"{np.mean(by[g]['psnr_u8_tlc']):.2f} dB  (n={len(by[g]['psnr_u8_tlc'])})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/GOPRO_Large")
    ap.add_argument("--out", default="results/SA-00_reproduce")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    test = Path(a.root) / "test"
    blurs = sorted(test.glob("*/blur/*.png"))
    n_sharp = len(list(test.glob("*/sharp/*.png")))
    n_train = len(list((Path(a.root) / "train").glob("*/sharp/*.png")))
    print(f"test pairs: {len(blurs)} | test sharp: {n_sharp} | train sharp: {n_train}", flush=True)
    assert len(blurs) == n_sharp == 1111 and n_train == 2103, "input data incomplete"
    if a.limit:
        blurs = blurs[:: max(1, len(blurs) // a.limit)][: a.limit]

    m_glob = load_nafnet_public(tlc=False)  # strict=True: key mismatch raises here
    m_tlc = load_nafnet_public(tlc=True)    # official GoPro test setting (NAFNetLocal)
    gpu = torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu"
    print(f"params: {sum(p.numel() for p in m_glob.parameters()) / 1e6:.3f}M | {gpu} | torch {torch.__version__}",
          flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "meta.txt").write_text(f"commit {commit}\ngpu {gpu}\ntorch {torch.__version__}\n")
    csv_path = out / ("gopro_test_nafnet32_public.csv" if not a.limit else f"smoke_{a.limit}.csv")

    rows, t0 = [], time.time()
    for i, bp in enumerate(blurs):
        b, s = load_rgb(bp), load_rgb(Path(str(bp).replace("/blur/", "/sharp/")))
        pt = restore_full(m_tlc, b, device=a.device)
        pg = restore_full(m_glob, b, device=a.device)
        r = {"id": str(bp.relative_to(test)), "group": bp.parts[-3], "psnr_blur": psnr(b, s),
             "psnr_u8_tlc": psnr(q8(pt), s), "psnr_float_tlc": psnr(pt, s), "psnr_u8_global": psnr(q8(pg), s)}
        if i % 5 == 0:  # SSIM is CPU-heavy: every 5th image
            r["ssim_skimage_tlc"] = ssim(q8(pt), s)
            r["ssim_gauss_tlc"] = float(structural_similarity(q8(pt), s, channel_axis=-1, data_range=1.0,
                                        gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        rows.append(r)
        if (i + 1) % 100 == 0:
            write_csv(csv_path, rows)
        if i in (0, 9) or (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"{i + 1:4d}/{len(blurs)}  TLC {np.mean([x['psnr_u8_tlc'] for x in rows]):.3f}  "
                  f"global {np.mean([x['psnr_u8_global'] for x in rows]):.3f}  {el / (i + 1):.2f}s/img  "
                  f"ETA {el / (i + 1) * (len(blurs) - i - 1) / 60:.1f} min", flush=True)

    write_csv(csv_path, rows)
    text = summarize(rows)
    if not a.limit:
        (out / "summary.txt").write_text(text + "\n")
    print("\n=== SUMMARY ===\n" + text + f"\nsaved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
