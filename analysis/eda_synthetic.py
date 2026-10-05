"""Step 4 EDA: statistics of the on-the-fly training distribution (CPU only).

Draws N training patches exactly like ``SyntheticNonUniformDataset`` and reports
what must be fixed in config BEFORE any evaluation:
- balance of R and of the region severities s_r (4 bins over the training range),
- heterogeneity H = std(S) per R, and proposed H-bin thresholds (tertiles, R >= 2),
- how far measured s_r deviates from its target (bisection tolerance),
- CPU cost per patch -> data-loader throughput for the pilot (step 6).

    python analysis/eda_synthetic.py --config configs/experiments/SA-13_a2.yaml --n 500

Writes results/EDA/{eda_samples.csv, summary.txt, s_and_h.png, examples.png}.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from spatial_deblur.data.datasets import load_rgb
from spatial_deblur.data.manifests import read_list
from spatial_deblur.degradations.blur import blur_nonuniform
from spatial_deblur.degradations.kernels import SeverityScale
from spatial_deblur.utils.config import load_config


def draw(i: int, frames, d, scale, seed):
    """Mirror of SyntheticNonUniformDataset.__getitem__ (without augmentation), keeping s_r."""
    rng = np.random.default_rng((seed, i))
    img = load_rgb(Path(d["gopro_root"]) / frames[int(rng.integers(len(frames)))])
    big = d["patch"] + 2 * d["margin"]
    h, w = img.shape[:2]
    y, x = int(rng.integers(0, h - big + 1)), int(rng.integers(0, w - big + 1))
    crop = img[y:y + big, x:x + big]
    r = int(rng.choice(d["regions"]))
    g = int(rng.choice(d["mask_grid_train"]))  # QĐ-02: multi-scale square grid, as in the dataset
    t0 = time.perf_counter()
    targets = [rng.uniform(*d["s_train"]) for _ in range(r)]
    s = blur_nonuniform(crop, rng, scale, r, d["s_train"], d["noise_sigma"], margin=d["margin"], targets=targets,
                        grid=(g, g))
    dt = time.perf_counter() - t0
    sharp = crop[d["margin"]:-d["margin"], d["margin"]:-d["margin"]]
    return r, targets, s, sharp, dt


def region_scale(masks: np.ndarray) -> float:
    """Median equivalent side (sqrt of area, px) of the connected blobs of argmax(masks)."""
    lab = masks.argmax(0)
    sizes = []
    for k in range(masks.shape[0]):
        comp, n = label(lab == k)
        sizes += list(np.bincount(comp.ravel())[1:])
    return float(np.sqrt(np.median(sizes))) if sizes else float("nan")


def full_frame_stats(frames, d, scale, seed, n):
    """Same generator at test-set resolution (full frame + reflect margin, as generate_testsets.py),
    on TRAINING frames with a separate seed: no test image is looked at."""
    out = []
    for j in range(n):
        rng = np.random.default_rng((seed, 99, j))
        img = load_rgb(Path(d["gopro_root"]) / frames[int(rng.integers(len(frames)))])
        m = d["margin"]
        padded = np.pad(img, ((m, m), (m, m), (0, 0)), mode="reflect")
        r = int(rng.integers(2, 5))
        s = blur_nonuniform(padded, rng, scale, r, d["s_train"], d["noise_sigma"], margin=m,
                            grid=tuple(d["mask_grid_test"]))  # QĐ-02: 6 x 10 on full frames
        out.append({"R": r, "H": float(s.severity_map.std()), "scale_px": region_scale(s.masks)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/SA-13_a2.yaml")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-full", type=int, default=60, help="full-frame samples (R>=2) for the train/test scale check")
    ap.add_argument("--seed", type=int, default=12345)  # not a training seed
    ap.add_argument("--out", default="results/EDA")
    a = ap.parse_args()
    d = load_config(a.config)["data"]
    scale = SeverityScale(**d["severity_scale"])
    frames = read_list(d["sharp_train"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, s_all, examples, t_start = [], [], [], time.perf_counter()
    for i in range(a.n):
        r, targets, s, sharp, dt = draw(i, frames, d, scale, a.seed)
        sm = s.severity_map
        rows.append({"i": i, "R": r, "S_mean": float(sm.mean()), "H": float(sm.std()),
                     "scale_px": region_scale(s.masks) if r >= 2 else float("nan"),
                     "s_r": ";".join(f"{v:.4f}" for v in s.severities),
                     "max_abs_err": float(np.max(np.abs(np.array(s.severities) - np.array(targets)))),
                     "blur_sec": dt})
        s_all += list(s.severities)
        if len(examples) < 8 and r == 1 + len(examples) % 4:
            examples.append((s.blur, sharp, sm, r))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{a.n}  {(time.perf_counter() - t_start) / (i + 1):.3f}s/patch (load + blur)", flush=True)
    total = (time.perf_counter() - t_start) / a.n

    with (out / "eda_samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    R = np.array([x["R"] for x in rows])
    H = np.array([x["H"] for x in rows])
    s_all = np.array(s_all)
    lo, hi = d["s_train"]
    edges = np.linspace(lo, hi, 5)
    s_counts = np.histogram(s_all, bins=edges)[0]
    h_multi = H[R >= 2]
    tert = np.quantile(h_multi, [1 / 3, 2 / 3])
    err = np.array([x["max_abs_err"] for x in rows])
    blur_t = np.array([x["blur_sec"] for x in rows])
    lines = [
        f"N = {a.n} patches, config {a.config}, severity_scale {d['severity_scale']}",
        "R counts: " + ", ".join(f"R={k}: {(R == k).sum()}" for k in sorted(set(R))),
        f"s_r bins {np.round(edges, 4).tolist()}: {s_counts.tolist()}  (n={len(s_all)})",
        f"s_r min/max: {s_all.min():.4f} / {s_all.max():.4f}",
        f"|measured - target| s_r: median {np.median(err):.4f}, p95 {np.quantile(err, 0.95):.4f}, max {err.max():.4f}",
        "H = std(S) by R: " + ", ".join(f"R={k}: median {np.median(H[R == k]):.4f}, p90 {np.quantile(H[R == k], 0.9):.4f}"
                                        for k in sorted(set(R))),
        f"proposed H bins (tertiles over R>=2): low < {tert[0]:.4f} <= mid < {tert[1]:.4f} <= high; R=1 -> H=0",
        f"CPU per patch: total {total:.3f}s (blur only median {np.median(blur_t):.3f}s)",
        "data-loader ceiling, batch 8: " + ", ".join(f"{k} workers ~ {k / (8 * total):.2f} it/s" for k in (2, 4)),
    ]
    if a.n_full:
        print(f"full-frame check: {a.n_full} samples ...", flush=True)
        ff = full_frame_stats(frames, d, scale, a.seed, a.n_full)
        Hf, Sf = np.array([x["H"] for x in ff]), np.array([x["scale_px"] for x in ff])
        Sp = np.array([x["scale_px"] for x in rows if x["R"] >= 2])
        lines += [
            f"train patch {d['patch']}px, R>=2: H median {np.median(h_multi):.4f} | region blob side median {np.nanmedian(Sp):.0f}px",
            f"full frame (test-size), R>=2: H median {np.median(Hf):.4f} | region blob side median {np.nanmedian(Sf):.0f}px",
            f"blob-side ratio full/train: {np.nanmedian(Sf) / np.nanmedian(Sp):.2f}x",
        ]
        with (out / "eda_fullframe.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ff[0]))
            w.writeheader()
            w.writerows(ff)
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print("\n=== EDA SUMMARY ===\n" + text, flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(s_all, bins=edges, edgecolor="k"); ax[0].set_title("Mức nhòe của vùng s_r (4 khoảng đều)")
    for k in sorted(set(R)):
        ax[1].hist(H[R == k], bins=30, alpha=0.5, label=f"R = {k}")
    for v in tert:
        ax[1].axvline(v, color="k", ls="--")
    ax[1].set_title("Độ không đồng nhất H = std(S) theo số vùng nhòe R\n(nét đứt: phân vị 33% và 67%, R ≥ 2)")
    ax[1].legend()
    ax[2].hist(np.concatenate([[x["S_mean"]] for x in rows]), bins=30, edgecolor="k"); ax[2].set_title("Giá trị trung bình của bản đồ nhòe thật S trên mỗi patch")
    fig.tight_layout(); fig.savefig(out / "s_and_h.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)))
    for j, (b, sh, sm, r) in enumerate(examples):
        ax[j, 0].imshow(b); ax[j, 0].set_title(f"Ảnh nhòe B, R = {r}")
        ax[j, 1].imshow(sh); ax[j, 1].set_title("Ảnh nét I")
        im = ax[j, 2].imshow(sm, vmin=0, vmax=1, cmap="magma"); ax[j, 2].set_title("Bản đồ nhòe thật S")
        for c in range(3):
            ax[j, c].axis("off")
    fig.colorbar(im, ax=ax[:, 2].tolist(), shrink=0.6)
    fig.savefig(out / "examples.png", dpi=90); plt.close(fig)
    print("saved ->", out, flush=True)


if __name__ == "__main__":
    main()
