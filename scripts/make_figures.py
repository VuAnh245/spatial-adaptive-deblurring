"""Data-independent figures for slides and report (no dataset needed).

    python scripts/make_figures.py --out docs/figures [--image path/to/sharp.png]

Produces:
  fig_kernels.png      kernels at several severities with their RMS radius
  fig_masks.png        soft masks and severity map S for R = 1..4
  fig_blur_examples.png  sharp / uniform blur (D1) / non-uniform blur (D2) + S
Without --image a public-domain sample (skimage astronaut) is used; for the
report, rerun with a GoPro or COCO frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from spatial_deblur.degradations.blur import blur_nonuniform, blur_uniform
from spatial_deblur.degradations.kernels import SeverityScale, kernel_for_severity, rms_radius
from spatial_deblur.degradations.masks import soft_masks

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def load_image(path):
    if path:
        return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    from skimage import data
    return data.astronaut().astype(np.float32) / 255.0


def fig_kernels(out: Path, scale: SeverityScale, seed: int) -> None:
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    fig, axes = plt.subplots(2, len(levels), figsize=(2.2 * len(levels), 4.6))
    for row, curv in enumerate((0.0, None)):
        for j, s in enumerate(levels):
            rng = np.random.default_rng((seed, row, j))
            k, s_meas = kernel_for_severity(rng, s, scale, curvature=curv, jitter=0.0 if curv == 0.0 else None)
            ax = axes[row, j]
            ax.imshow(k, cmap="gray_r")
            ax.set_title(f"s = {s_meas:.2f}\nr = {rms_radius(k):.1f} px", fontsize=9)
            ax.axis("off")
        axes[row, 0].text(-0.15, 0.5, "thẳng" if row == 0 else "cong", transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right")
    fig.suptitle("Kernel nhòe theo mức độ s (bán kính RMS chuẩn hóa)")
    fig.tight_layout()
    fig.savefig(out / "fig_kernels.png", dpi=200)
    plt.close(fig)


def fig_masks(out: Path, seed: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for j, r in enumerate((1, 2, 3, 4)):
        rng = np.random.default_rng((seed, 10, r))
        m = soft_masks(rng, r, 128, 128, temperature=0.1)
        sev = np.linspace(0.1, 0.7, r)
        axes[0, j].imshow(np.argmax(m, 0), cmap="tab10", vmin=0, vmax=9)
        axes[0, j].set_title(f"R = {r}")
        im = axes[1, j].imshow(np.tensordot(sev, m, axes=1), cmap="magma", vmin=0, vmax=0.75)
        axes[1, j].set_title("Bản đồ S(x, y)")
        for a in axes[:, j]:
            a.axis("off")
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8, label="mức nhòe s")
    fig.suptitle("Hàng trên: vùng chiếm ưu thế của mặt nạ mềm. Hàng dưới: bản đồ mức nhòe S(x, y)")
    fig.savefig(out / "fig_masks.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_blur_examples(out: Path, img: np.ndarray, scale: SeverityScale, seed: int) -> None:
    rng = np.random.default_rng((seed, 20))
    k, s = kernel_for_severity(rng, 0.45, scale)
    d1 = blur_uniform(img, k, s, 0.01, rng)
    m = 20
    padded = np.pad(img, ((m, m), (m, m), (0, 0)), mode="reflect")
    d2 = blur_nonuniform(padded, rng, scale, 3, (0.05, 0.7), 0.01, margin=m, targets=[0.05, 0.35, 0.7])
    panels = [(img, "Ảnh nét"), (d1.blur, f"D1: đồng nhất, s = {s:.2f}"), (d2.blur, "D2: không đồng nhất (R = 3)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(t)
        ax.axis("off")
    h = axes[3].imshow(d2.severity_map, cmap="magma", vmin=0, vmax=0.75)
    axes[3].set_title("Bản đồ S của D2")
    axes[3].axis("off")
    fig.colorbar(h, ax=axes[3], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "fig_blur_examples.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--image", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--r-max", type=float, default=8.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scale = SeverityScale(0.0, a.r_max)
    fig_kernels(out, scale, a.seed)
    fig_masks(out, a.seed)
    fig_blur_examples(out, load_image(a.image), scale, a.seed)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
