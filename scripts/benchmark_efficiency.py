"""Efficiency benchmark (params, MACs, latency, FPS, peak VRAM).

    python scripts/benchmark_efficiency.py --model nafnet_public
"""
from __future__ import annotations

import argparse
import json

from spatial_deblur.metrics.efficiency import count_params, latency, macs


def build(name: str):
    from spatial_deblur.models.public_checkpoints import load_nafnet_public
    from spatial_deblur.modulation.variants import build_variant
    if name == "nafnet_public":
        return load_nafnet_public()
    return build_variant(name)  # A0, A0+, A1, A2, A3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()
    m = build(a.model)
    report = {"model": a.model, **count_params(m), "macs_G_256": macs(m), "macs_tool": "fvcore",
              **latency(m, a.height, a.width, device=a.device)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
