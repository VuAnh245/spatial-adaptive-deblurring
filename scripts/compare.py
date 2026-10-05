"""Paired, group-level comparison of two per-image CSVs (same split).

    python scripts/compare.py results/SA-13_eval_test_in_pred_tlc.csv results/SA-12_eval_test_in_pred_tlc.csv --metric psnr

The JSON keeps every ID next to its readable label (docs/naming.md), e.g.
a_id "SA-13" / a_label "SFT-Map", comparison_label "Phép thử giám sát (chính)",
split_id "test_in" / split_label "Tổng hợp trong phân phối".
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from spatial_deblur.metrics.stats import group_mean, paired_comparison, verdict
from spatial_deblur.utils.labels import (MODE_LABELS, POOL_LABELS, comparison_label, experiment_of, metric_label,
                                         model_label, split_label, variant_of)

STEM = re.compile(r"(?P<exp>SA-\d\d)_eval_(?P<split>.+)_(?P<mode>pred|oracle|shuffle)(?:_(?P<pool>tlc|global))?$")


def read(path, metric):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {r["id"]: float(r[metric]) for r in rows}, {r["id"]: r["group"] for r in rows}


def describe(path: str, side: str) -> dict:
    stem = Path(path).stem
    m = STEM.match(stem)
    out = {f"{side}_id": experiment_of(stem) or stem, f"{side}_variant": variant_of(stem),
           f"{side}_label": model_label(stem), f"{side}_file": Path(path).name}
    if m:
        out.update({f"{side}_mode_id": m["mode"], f"{side}_mode_label": MODE_LABELS[m["mode"]],
                    f"{side}_pool_id": m["pool"], f"{side}_pool_label": POOL_LABELS.get(m["pool"] or "", "")})
    return out, (m["split"] if m else None)


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
    da, split_a = describe(args.a, "a")
    db, split_b = describe(args.b, "b")
    if split_a and split_b and split_a != split_b:
        raise SystemExit(f"different splits: {split_a} vs {split_b}")
    same_model_other_mode = (da["a_variant"] == db["b_variant"]
                             and da.get("a_mode_id") != db.get("b_mode_id"))
    label = (MODE_LABELS[da["a_mode_id"] if da.get("a_mode_id") != "pred" else db["b_mode_id"]]
             if same_model_other_mode else comparison_label(Path(args.a).stem, Path(args.b).stem))
    head = {**da, **db,
            "comparison_label": label,
            "split_id": split_a, "split_label": split_label(split_a) if split_a else None,
            "metric_id": args.metric, "metric_label": metric_label(args.metric)}
    res["verdict_id"] = res.pop("verdict")
    res["verdict_label"] = {"first_better": f"{da['a_label']} tốt hơn", "second_better": f"{db['b_label']} tốt hơn",
                            "no_difference": "Không khác biệt"}[res["verdict_id"]]
    print(json.dumps({**head, **res}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
