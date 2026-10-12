"""Evaluation queue: every (model, seed, split, mode) job of the study, split over GPUs, with a deadline.

    python scripts/eval_queue.py plan  --ckpts results/eval/ckpts.json --out results/eval [--exclude PATTERN ...]
    python scripts/eval_queue.py run   --shard 0 --shards 2 --device cuda:0 --deadline <unix time> --out results/eval

ckpts.json: {"SA-10": path, "SA-13": path, "SA-13_seed2": path, ..., "SA-00": path}. A bare ID is seed 1.
"SA-00" is the public NAFNet checkpoint saved as an A0 state_dict (reference row: no fine-tuning).
Jobs (QĐ-01: TLC for every model; QĐ-09: test_in is the primary endpoint), in priority order:
  tier 1: every seed of every model (pred) on test_in, the public checkpoint on test_in; seed-1 models on gopro_test
  tier 2: seed-1 models on test_ood / test_straight / test_curved; SFT-Map seed 1 SA-20 (oracle) + SA-21 (shuffle)
          on test_in; seeds 2/3 and the public checkpoint on gopro_test
  tier 3: SFT-Map seed 1 SA-20 / SA-21 on the other synthetic splits; seeds 2/3 on the other synthetic splits
  tier 4: seed-1 models on test_in_noise2x / _noise4x (SA-26), only if those folders exist
  tier 5: Baseline and SFT-Map seed 1 with global pooling on test_in and gopro_test (QĐ-01 appendix)
Output: seed 1 -> <out>/<stem>.csv (as before), seed n -> <out>/seed<n>/<stem>.csv; the stem keeps the pattern
read by compare.py / eval_tables.py. Job id = "<stem>@<seed>"; ``--exclude`` takes fnmatch patterns on it
(e.g. "SA-1[0-3]_eval_gopro_test_pred_tlc@1" when those verified CSVs are copied in instead).
An existing CSV is never overwritten: with the expected row count it counts as done, otherwise the plan fails.
``run`` executes one shard: a job that cannot finish before the deadline at the measured s/img is not started,
a running job is killed at the deadline (no CSV is written, so it reruns in a later session with the same plan).
Nothing here prints a metric: evaluate.py runs with --blind and its output goes to logs/<job>.log.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from spatial_deblur.data.manifests import read_list
from spatial_deblur.utils.config import load_config

MODELS = {"SA-00": "configs/experiments/SA-10_a0.yaml",  # public checkpoint, A0 architecture
          "SA-10": "configs/experiments/SA-10_a0.yaml", "SA-11": "configs/experiments/SA-11_a0plus.yaml",
          "SA-12": "configs/experiments/SA-12_a1.yaml", "SA-13": "configs/experiments/SA-13_a2.yaml",
          "SA-14": "configs/experiments/SA-14_a3.yaml"}
TRAINED = ["SA-10", "SA-11", "SA-12", "SA-13", "SA-14"]
SYN_OTHER = ["test_ood", "test_straight", "test_curved"]
COST = {"pred": 1.0, "oracle": 1.0, "shuffle": 1.15}  # shuffle runs the estimator twice


def parse_ckpts(raw: dict) -> dict[tuple[str, int], str]:
    out = {}
    for key, path in raw.items():
        exp, _, seed = key.partition("_seed")
        assert exp in MODELS, f"unknown experiment in ckpts.json: {key}"
        out[(exp, int(seed or 1))] = path
    return out


def jobs_by_tier(have: set[tuple[str, int]], synthetic_root: Path) -> list[tuple[int, str, int, str, str, str]]:
    """(tier, exp, seed, split, mode, pool) for the checkpoints in ``have``."""
    s1 = [e for e in TRAINED if (e, 1) in have]
    extra = sorted((e, s) for e, s in have if s > 1)
    pub = [("SA-00", 1)] if ("SA-00", 1) in have else []
    j = [(1, e, 1, "test_in", "pred", "tlc") for e in s1]
    j += [(1, e, s, "test_in", "pred", "tlc") for e, s in extra + pub]
    j += [(1, e, 1, "gopro_test", "pred", "tlc") for e in s1]
    j += [(2, e, 1, sp, "pred", "tlc") for sp in SYN_OTHER for e in s1]
    if ("SA-13", 1) in have:
        j += [(2, "SA-13", 1, "test_in", m, "tlc") for m in ("oracle", "shuffle")]
        j += [(3, "SA-13", 1, sp, m, "tlc") for sp in SYN_OTHER for m in ("oracle", "shuffle")]
    j += [(2, e, s, "gopro_test", "pred", "tlc") for e, s in extra + pub]
    j += [(3, e, s, sp, "pred", "tlc") for sp in SYN_OTHER for e, s in extra]
    for noisy in ["test_in_noise2x", "test_in_noise4x"]:
        if any((synthetic_root / noisy).glob("*.npz")):
            j += [(4, e, 1, noisy, "pred", "tlc") for e in s1]
    j += [(5, e, 1, sp, "pred", "global") for sp in ["test_in", "gopro_test"] for e in ("SA-10", "SA-13") if e in s1]
    return j


def n_images(split: str, d: dict) -> int:
    if split == "gopro_test":
        return len(read_list(d["gopro_test_pairs"]))
    return len(list((Path(d["synthetic_root"]) / split).glob("*.npz")))


def stem(exp, split, mode, pool) -> str:
    return f"{exp}_eval_{split}_{mode}_{pool}"  # pattern read by scripts/compare.py


def csv_path(out: Path, job: dict) -> Path:
    return (out if job["seed"] == 1 else out / f"seed{job['seed']}") / f"{job['stem']}.csv"


def csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def plan(a) -> None:
    d = load_config(MODELS["SA-13"])["data"]
    ckpts = parse_ckpts(json.loads(Path(a.ckpts).read_text()))
    missing = [k for k, p in ckpts.items() if not Path(p).exists()]
    assert not missing, f"checkpoint file missing for {missing}"
    out = Path(a.out)
    jobs, load, done, excluded = [], [0.0] * a.shards, 0, 0
    tiered = sorted(jobs_by_tier(set(ckpts), Path(d["synthetic_root"])), key=lambda j: j[0])  # stable: tier order
    for tier, exp, seed, split, mode, pool in tiered:
        job = {"tier": tier, "exp": exp, "seed": seed, "split": split, "mode": mode, "pool": pool,
               "config": MODELS[exp], "ckpt": ckpts[(exp, seed)], "stem": stem(exp, split, mode, pool)}
        job["id"] = f"{job['stem']}@{seed}"
        if any(fnmatch.fnmatch(job["id"], p) for p in a.exclude):
            excluded += 1
            continue
        job["n"] = n_images(split, d)
        assert job["n"] > 0, f"split {split} is empty - generate it first"
        existing = csv_path(out, job)
        if existing.exists():
            rows = csv_rows(existing)
            assert rows == job["n"], f"{existing} has {rows} rows, expected {job['n']} - not overwritten, resolve first"
            done += 1
        shard = min(range(a.shards), key=lambda k: load[k])
        load[shard] += 0 if existing.exists() else job["n"] * COST[mode]
        jobs.append({**job, "shard": shard})
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(jobs, indent=1))
    for k in range(a.shards):
        mine = [x for x in jobs if x["shard"] == k]
        print(f"shard {k}: {len(mine)} jobs, {load[k]:.0f} image passes to run "
              f"(~{load[k] * a.sec_per_img / 3600:.1f} h at {a.sec_per_img} s/img), tiers {sorted({x['tier'] for x in mine})}")
    print(f"total {len(jobs)} jobs ({done} already done, {excluded} excluded) -> {out / 'plan.json'}")


def run(a) -> None:
    out = Path(a.out)
    jobs = [x for x in json.loads((out / "plan.json").read_text()) if x["shard"] == a.shard]
    (out / "logs").mkdir(parents=True, exist_ok=True)
    qlog = (out / f"queue_shard{a.shard}.jsonl").open("a", encoding="utf-8")
    sec_per_img = a.sec_per_img  # prior; replaced by the measured rate after the first job

    def record(job, status, **kw):
        row = {"id": job["id"], "tier": job["tier"], "status": status, "time": round(time.time(), 1), **kw}
        qlog.write(json.dumps(row) + "\n")
        qlog.flush()
        print(f"[shard {a.shard}] {status:16s} {job['id']} {kw if kw else ''}", flush=True)

    env = dict(os.environ)
    if a.device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = a.device.split(":")[1]
    for job in jobs:
        target = csv_path(out, job)
        if target.exists():
            ok = csv_rows(target) == job["n"]
            record(job, "skipped_done" if ok else "conflict_existing")  # never overwritten
            continue
        left = a.deadline - time.time()
        need = job["n"] * COST[job["mode"]] * sec_per_img + 60
        if need > left:
            record(job, "skipped_deadline", need_sec=round(need), left_sec=round(left))
            continue
        cmd = [sys.executable, "-u", "scripts/evaluate.py", "--config", job["config"], "--ckpt", job["ckpt"],
               "--split", job["split"], "--mode", job["mode"], "--pool", job["pool"], "--device", "cuda",
               "--out", str(target), "--blind"]
        t0 = time.time()
        record(job, "start", n=job["n"])
        with (out / "logs" / f"{job['stem']}@{job['seed']}.log").open("w", encoding="utf-8") as log:
            try:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=max(1, left)).returncode
            except subprocess.TimeoutExpired:
                record(job, "cut_deadline", sec=round(time.time() - t0))
                continue
        sec = time.time() - t0
        if rc == 0 and target.exists():
            sec_per_img = sec / (job["n"] * COST[job["mode"]])
            record(job, "done", sec=round(sec), sec_per_img=round(sec_per_img, 2))
        else:
            record(job, "failed", rc=rc, sec=round(sec))
    qlog.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run"])
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--ckpts", default="results/eval/ckpts.json",
                    help="plan: {'SA-10': path, 'SA-13_seed2': path, 'SA-00': public A0 state_dict, ...}")
    ap.add_argument("--exclude", nargs="*", default=[], help="plan: fnmatch patterns on '<stem>@<seed>'")
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--deadline", type=float, default=time.time() + 10 * 3600, help="unix time")
    ap.add_argument("--sec-per-img", type=float, default=2.0, help="prior before the first job is measured")
    a = ap.parse_args()
    plan(a) if a.cmd == "plan" else run(a)


if __name__ == "__main__":
    main()
