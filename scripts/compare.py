"""Paired, group-level comparison of two per-image CSVs (same split).

    python scripts/compare.py results/A2.csv results/A1.csv --metric psnr
"""
from __future__ import annotations

import argparse
import csv
import json

from spatial_deblur.metrics.stats import group_mean, paired_comparison, verdict


def read(path, metric):
    rows = list(csv.DictReader(open(path)))
    return {r["id"]: float(r[metric]) for r in rows}, {r["id"]: r["group"] for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--metric", default="psnr")
    args = ap.parse_args()
    sa, ga = read(args.a, args.metric)
    sb, _ = read(args.b, args.metric)
    res = paired_comparison(group_mean(sa, ga), group_mean(sb, ga))
    res["verdict"] = verdict(res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
