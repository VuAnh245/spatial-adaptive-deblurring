"""Restore a folder with a trained variant (used to produce M5 outputs for the CV study).

    python scripts/restore_folder.py --config configs/experiments/SA-13_a2.yaml --ckpt results/SA-13_a2_seed1/final.pth \
        --input ../deblurring-for-detection/data/coco_blur/D2_moderate \
        --output ../deblurring-for-detection/results/restored/M5/D2_moderate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from spatial_deblur.engine.inference import restore_full
from spatial_deblur.modulation.variants import build_variant
from spatial_deblur.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    m = load_config(a.config)["model"]
    model = build_variant(m["variant"], {k: m[k] for k in ("width", "middle_blk_num", "enc_blk_nums", "dec_blk_nums")})
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.input).iterdir() if p.suffix.lower() in (".png", ".jpg"))
    for f in files:
        img = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0
        y = restore_full(model, img, a.device, tile=a.tile)
        Image.fromarray((y * 255 + 0.5).astype(np.uint8)).save(out / (f.stem + ".png"))
    print(f"{len(files)} images -> {out}")


if __name__ == "__main__":
    main()
