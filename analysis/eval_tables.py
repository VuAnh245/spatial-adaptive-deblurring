"""Result tables from the per-image CSVs of scripts/eval_queue.py (QĐ-05, QĐ-09), BLIND for pair 2 by default.

    python analysis/eval_tables.py --dir results/eval --out results/eval/tables_qd09 --unblind
    python analysis/eval_tables.py --dir results/eval --out <new folder>        # blind: SA-11, SA-12 hidden

Inputs: seed-1 CSVs in <dir>/, seed-n CSVs in <dir>/seed<n>/ (stem "<SA-xx>_eval_<split>_<mode>_<pool>").
Blind mode prints and writes nothing about Baseline-L (SA-11) or SFT-Free (SA-12). ``--unblind`` refuses to run
unless the COMMITTED docs/experiments.md has "## Kết luận về bộ ước lượng E" and that file has no uncommitted edit.

Summary: per (model, seed, split, mode, pool) the image count and mean psnr, psnr_u8, ssim, psnr_local_high,
map_mae, map_spearman (gopro_test: also mean PSNR for video_in_train 0 / 1), plus a "mean" row over the seeds of a
model that has several. SA-00 = public NAFNet checkpoint (no fine-tuning), a reference row only.

Comparisons (PSNR, per-image -> per-video means, Wilcoxon over videos, bootstrap CI; QĐ-05), QĐ-09:
  core family per split, TLC, pred: Phép thử giám sát SA-13 vs SA-12 (primary comparison), Phép thử tham số
  SA-13 vs SA-11, Phép thử điều biến SA-12 vs SA-10, Phép thử nhánh kênh SA-14 vs SA-13 (when SA-14 exists).
  - Phép thử giám sát uses the per-image mean over seeds 1-3 when both models have all three, else seed 1
    (reported in "aggregation"); the per-seed differences are listed next to it.
  - the other core comparisons use seed 1 (same seed = same training data stream, paired); a sensitivity row
    with the seed mean is added when a model has several seeds.
  - Holm over the core family of each split; test_in is the PRIMARY endpoint, every other split is secondary.
    Each comparison carries p_raw, p_holm, verdict_raw (stats.verdict on p_raw, the rule used before QĐ-09)
    and verdict_holm (the same rule on p_holm). Conclusions use verdict_holm on test_in.
  secondary, not in the family: SA-13 vs SA-10, SA-20 (oracle) and SA-21 (shuffle) vs SA-13 pred (seed 1),
  SA-10 vs SA-00 (effect of fine-tuning).
H groups (QĐ-03) need data.h_bins in the config; without it that table is skipped and says so.
Writes <out>/{summary.csv, comparisons.json, README.txt}; never writes into an existing non-empty folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from spatial_deblur.metrics.stats import group_mean, paired_comparison, verdict
from spatial_deblur.utils.config import load_config
from spatial_deblur.utils.labels import (MODE_LABELS, POOL_LABELS, comparison_label, experiment_label, model_label,
                                         split_label)

STEM = re.compile(r"(?P<exp>SA-\d\d)_eval_(?P<split>.+)_(?P<mode>pred|oracle|shuffle)_(?P<pool>tlc|global)$")
HIDDEN = {"SA-11", "SA-12"}  # pair 2: Baseline-L, SFT-Free
HEADING = "## Kết luận về bộ ước lượng E"
METRICS = ["psnr", "psnr_u8", "ssim", "psnr_local_high", "map_mae", "map_spearman"]
CORE = [("SA-13", "SA-12"), ("SA-13", "SA-11"), ("SA-12", "SA-10"), ("SA-14", "SA-13")]
SECONDARY = [("SA-13", "SA-10"), ("SA-10", "SA-00")]
PRIMARY_SPLIT = "test_in"
MODAL_TRAINED = {"SA-14", "SA-32", "SA-33", "SA-35"}  # trained on Modal L4; the others on Kaggle T4
CROSS_PLATFORM_DB = 0.02  # declared 2026-10-13 before round-2 numbers: C2 drift 0.011 dB at step 2500


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8").stdout.strip()


def unblind_gate() -> str:
    if HEADING not in git("show", "HEAD:docs/experiments.md"):
        raise SystemExit(f"--unblind refused: '{HEADING}' is not in the committed docs/experiments.md")
    if git("status", "--porcelain", "--", "docs/experiments.md"):
        raise SystemExit("--unblind refused: docs/experiments.md has uncommitted changes")
    return git("log", "-1", "--format=%h %cI", "--", "docs/experiments.md")


def fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def mean(rows, key) -> float:
    v = np.array([fnum(r.get(key)) for r in rows])
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    m, order = len(pvals), np.argsort(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def load(root: Path) -> dict:
    """{(exp, split, mode, pool, seed): rows}."""
    data = {}
    for folder, seed in [(root, 1)] + [(p, int(p.name[4:])) for p in sorted(root.glob("seed[0-9]*")) if p.is_dir()]:
        for p in sorted(folder.glob("SA-*_eval_*.csv")):
            m = STEM.match(p.stem)
            if m:
                with p.open(encoding="utf-8", newline="") as f:
                    data[(m["exp"], m["split"], m["mode"], m["pool"], seed)] = list(csv.DictReader(f))
    return data


def scores(data, exp, split, seeds, mode="pred", pool="tlc"):
    """Per-image PSNR averaged over ``seeds`` (all must exist, same image ids), and id -> video."""
    runs = [data[(exp, split, mode, pool, s)] for s in seeds]
    ids = [{r["id"] for r in rows} for rows in runs]
    assert all(x == ids[0] for x in ids), f"{exp} {split}: seeds cover different images"
    by_seed = [{r["id"]: fnum(r["psnr"]) for r in rows} for rows in runs]
    video = {r["id"]: r.get("video") or r["group"] for r in runs[0]}
    return {k: float(np.mean([s[k] for s in by_seed])) for k in ids[0]}, video


def compare(sa: dict, sb: dict, video: dict) -> dict:
    res = paired_comparison(group_mean(sa, video), group_mean(sb, video))
    shared = sorted(set(sa) & set(sb))
    res["n_images"] = len(shared)
    res["win_rate_images"] = float(np.mean([sa[k] > sb[k] for k in shared]))
    return res


def nanmean(values) -> float:
    v = np.array([fnum(x) for x in values])
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Vietnamese labels on a cp1252 Windows console
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/eval")
    ap.add_argument("--out", required=True, help="new folder for the tables (earlier tables are kept as they are)")
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--config", default="configs/experiments/SA-13_a2.yaml", help="for data.h_bins (QĐ-03)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty - choose a new --out (tables are never overwritten)")
    gate = unblind_gate() if a.unblind else None
    hidden = set() if a.unblind else HIDDEN
    data = load(Path(a.dir))
    seeds_of = defaultdict(set)
    for exp, split, mode, pool, seed in data:
        seeds_of[(exp, split, mode, pool)].add(seed)
    mode_txt = f"UNBLIND (estimator conclusion committed: {gate})" if gate else \
        "BLIND: Baseline-L (SA-11) and SFT-Free (SA-12) hidden until the estimator conclusion is committed"
    print(mode_txt + "\n")

    summary = []
    for (exp, split, mode, pool), seeds in sorted(seeds_of.items(), key=lambda kv: (kv[0][1], kv[0][3], kv[0][2], kv[0][0])):
        groups = [(str(s), data[(exp, split, mode, pool, s)]) for s in sorted(seeds)]
        if len(seeds) > 1:
            groups.append((f"mean{len(seeds)}", None))
        for tag, rows in groups:
            row = {"exp": exp, "label": model_label(exp), "seed": tag, "split": split, "split_label": split_label(split),
                   "mode": mode, "mode_label": MODE_LABELS[mode], "pool": pool, "pool_label": POOL_LABELS[pool]}
            if exp in hidden:
                row.update({"n": len(rows) if rows else "", **{k: "ẩn" for k in METRICS}})
            elif rows is None:  # mean over seeds of the per-seed means
                per = [r for r in summary if r["exp"] == exp and r["split"] == split and r["mode"] == mode
                       and r["pool"] == pool and r["seed"].isdigit()]
                row.update({"n": per[0]["n"], **{k: round(nanmean(p[k] for p in per), 4) for k in METRICS}})
            else:
                row.update({"n": len(rows), **{k: round(mean(rows, k), 4) for k in METRICS}})
                if split == "gopro_test" and rows and "video_in_train" in rows[0]:
                    for flag in ("0", "1"):
                        sub = [r for r in rows if r["video_in_train"] == flag]
                        row[f"psnr_video_in_train_{flag}"] = round(mean(sub, "psnr"), 4)
                        row[f"n_video_in_train_{flag}"] = len(sub)
            summary.append(row)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in summary for k in r))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    print(f"{'split':16s} {'pool':6s} {'mode':8s} {'model':12s} {'seed':6s} {'n':>5s} {'PSNR':>8s} {'PSNR u8':>8s} "
          f"{'SSIM':>7s} {'PSNR nặng':>9s} {'map MAE':>8s} {'map rho':>8s}")
    for r in summary:
        cells = [r[k] if isinstance(r[k], str) else f"{r[k]:.3f}" for k in METRICS]
        print(f"{r['split']:16s} {r['pool']:6s} {r['mode']:8s} {r['label']:12s} {r['seed']:6s} {str(r['n']):>5s} "
              f"{cells[0]:>8s} {cells[1]:>8s} {cells[2]:>7s} {cells[3]:>9s} {cells[4]:>8s} {cells[5]:>8s}")

    comps = []

    def item(split, label, a_key, b_key, family, aggregation, res=None, extra=None):
        it = {"split": split, "split_label": split_label(split), "primary_endpoint": split == PRIMARY_SPLIT,
              "family": family, "comparison_label": label, "a": a_key, "a_label": model_label(a_key.split()[0]),
              "b": b_key, "b_label": model_label(b_key.split()[0]), "metric": "psnr", "group_by": "video",
              "aggregation": aggregation, **(extra or {})}
        it["result"] = "ẩn" if {a_key.split()[0], b_key.split()[0]} & hidden else res
        comps.append(it)
        return it

    for split in sorted({k[1] for k in data}):
        have = lambda e, s=1: (e, split, "pred", "tlc", s) in data  # noqa: E731
        family = []
        for x, y in CORE:
            if not (have(x) and have(y)):
                continue
            label = comparison_label(x, y)
            if (x, y) == ("SA-13", "SA-12") and all(have(x, s) and have(y, s) for s in (1, 2, 3)):
                seeds, agg = (1, 2, 3), "mean of seeds 1-3 per image"
            else:
                seeds, agg = (1,), "seed 1"
            res, extra = None, {}
            if not {x, y} & hidden:
                sa, video = scores(data, x, split, seeds)
                sb, _ = scores(data, y, split, seeds)
                res = compare(sa, sb, video)
                if len(seeds) > 1:
                    extra["per_seed_mean_diff"] = {
                        str(s): compare(scores(data, x, split, (s,))[0], scores(data, y, split, (s,))[0], video)["mean_diff"]
                        for s in seeds}
            family.append(item(split, label, f"{x} pred", f"{y} pred", "core", agg, res, extra))
            if (x, y) != ("SA-13", "SA-12") and not {x, y} & hidden:  # sensitivity: seed means where available
                sx = sorted(s for s in (1, 2, 3) if have(x, s))
                sy = sorted(s for s in (1, 2, 3) if have(y, s))
                if len(sx) > 1 or len(sy) > 1:
                    ra, video = scores(data, x, split, sx)
                    rb, _ = scores(data, y, split, sy)
                    item(split, label + " (độ nhạy: trung bình seed)", f"{x} pred", f"{y} pred", "sensitivity",
                         f"seeds {sx} vs {sy}", compare(ra, rb, video))
        done = [c for c in family if isinstance(c["result"], dict)]
        for c, p in zip(done, holm([c["result"]["wilcoxon_p"] for c in done])):
            c["result"]["p_raw"] = c["result"]["wilcoxon_p"]
            c["result"]["p_holm"] = p
            c["result"]["holm_family_size"] = len(done)
        if have("SA-35") and have("SA-12"):  # platform drift at step 30000: same config/seed, Modal L4 vs Kaggle T4
            sa, video = scores(data, "SA-35", split, (1,))
            item(split, "Lệch nền tảng (SFT-Free Modal so với Kaggle)", "SA-35 pred", "SA-12 pred", "secondary", "seed 1",
                 compare(sa, scores(data, "SA-12", split, (1,))[0], video))
        for x, y in SECONDARY:
            if have(x) and have(y):
                res = None
                if not {x, y} & hidden:
                    sa, video = scores(data, x, split, (1,))
                    res = compare(sa, scores(data, y, split, (1,))[0], video)
                item(split, comparison_label(x, y), f"{x} pred", f"{y} pred", "secondary", "seed 1", res)
        for e in ("SA-13", "SA-33"):
            for m in ("oracle", "shuffle"):
                if (e, split, m, "tlc", 1) in data and have(e):
                    sa, video = scores(data, e, split, (1,), mode=m)
                    item(split, MODE_LABELS[m], f"{e} {m}", f"{e} pred", "secondary", "seed 1",
                         compare(sa, scores(data, e, split, (1,))[0], video))
        # QĐ-10 (exploratory, declared before the runs): SA-32 (oracle mode) and SA-33 (pred) against SFT-Free and
        # Baseline seed 1; "có hiệu quả" only if diff > 3 x SD of the SFT-Free seed means on this split AND CI low > 0.
        sf = [mean(data[("SA-12", split, "pred", "tlc", s)], "psnr") for s in (1, 2, 3) if have("SA-12", s)]
        floor = 3 * float(np.std(sf, ddof=1)) if len(sf) == 3 else None
        for x, mode in (("SA-32", "oracle"), ("SA-33", "pred")):
            if (x, split, mode, "tlc", 1) not in data:
                continue
            for y in ("SA-35", "SA-14", "SA-12", "SA-10", "SA-13"):
                if not have(y):
                    continue
                sa, video = scores(data, x, split, (1,), mode=mode)
                res = compare(sa, scores(data, y, split, (1,))[0], video)
                res["seed_noise_floor_3sd"] = floor
                same_platform = y in MODAL_TRAINED
                res["same_platform"] = same_platform
                ok = floor is not None and res["mean_diff"] > floor and res["ci_low"] > 0
                if not same_platform and abs(res["mean_diff"]) < CROSS_PLATFORM_DB:
                    res["qd10_label"] = "không phân giải được do lệch nền tảng"
                else:
                    res["qd10_label"] = "có hiệu quả" if ok else "không phân biệt được với dao động giữa seed"
                item(split, f"QĐ-10: {experiment_label(x)} so với {model_label(y)}", f"{x} {mode}", f"{y} pred", "qd10",
                     "seed 1", res)

    for c in comps:
        r = c["result"]
        if not isinstance(r, dict):
            continue
        names = {"first_better": f"{c['a_label']} tốt hơn", "second_better": f"{c['b_label']} tốt hơn",
                 "no_difference": "Không khác biệt"}
        r["verdict_raw"] = verdict(r)
        r["verdict_raw_label"] = names[r["verdict_raw"]]
        if "p_holm" in r:
            r["verdict_holm"] = verdict({**r, "wilcoxon_p": r["p_holm"]})
            r["verdict_holm_label"] = names[r["verdict_holm"]]
    (out / "comparisons.json").write_text(json.dumps({"mode": mode_txt, "primary_endpoint": PRIMARY_SPLIT,
                                                      "comparisons": comps}, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    print("\nComparisons (PSNR, groups = video; core family Holm-adjusted per split; primary = test_in):")
    for c in comps:
        r = c["result"]
        if isinstance(r, str) or r is None:
            txt = r or "-"
        else:
            holm_txt = f" p_holm {r['p_holm']:.3f} -> {r['verdict_holm_label']} |" if "p_holm" in r else ""
            txt = (f"diff {r['mean_diff']:+.3f} dB [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p {r['wilcoxon_p']:.3f}"
                   f" ({r['verdict_raw_label']}) |{holm_txt} groups {r['n_groups']} | win img {r['win_rate_images']:.2f}")
        if isinstance(r, dict) and "qd10_label" in r:
            txt += f" | 3SD {r['seed_noise_floor_3sd']:.4f} -> {r['qd10_label']}" if r["seed_noise_floor_3sd"] else ""
        star = "*" if c["primary_endpoint"] and c["family"] == "core" else " "
        print(f" {star}{c['split']:16s} {c['family']:11s} {c['comparison_label'][:34]:34s} {c['aggregation'][:24]:24s} {txt}")
        if c.get("per_seed_mean_diff"):
            print(" " * 20 + "per-seed diff: " + ", ".join(f"seed {s} {v:+.3f}" for s, v in c["per_seed_mean_diff"].items()))

    h_bins = load_config(a.config)["data"].get("h_bins")
    if h_bins is None:
        note = "H groups (QĐ-03): skipped, data.h_bins is not in the config yet"
        print("\n" + note)
    else:
        lo, hi = h_bins
        note = f"H groups (QĐ-03): R=1 | H < {lo} | {lo} <= H < {hi} | H >= {hi} (test_in, TLC, seed 1)"
        print("\n" + note)
        for (exp, split, mode, pool, seed), rows in sorted(data.items()):
            if split != PRIMARY_SPLIT or pool != "tlc" or seed != 1 or exp in hidden:
                continue
            groups = defaultdict(list)
            for r in rows:
                h = fnum(r.get("H"))
                g = "R=1" if r.get("regions") == "1" else ("low" if h < lo else "mid" if h < hi else "high")
                groups[g].append(fnum(r["psnr"]))
            print(f"  {exp} {mode:8s} " + " | ".join(f"{g} {np.mean(groups[g]):.3f} (n {len(groups[g])})"
                                                      for g in ("R=1", "low", "mid", "high") if groups[g]))
    (out / "README.txt").write_text(f"{mode_txt}\nprimary endpoint: {PRIMARY_SPLIT} (QĐ-09)\n{note}\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
