"""Step 6 pilot summary and the shared iteration budget (QĐ-04).

    python analysis/pilot_report.py --runs results/SA-13_seed1_pilot results/SA-11_seed1_pilot \
        --gpu-log results/pilot_gpu.csv --val-every 2500 --ckpt-every 2500

Per run: it/s (curves.csv 'sec', first log row excluded as warm-up), mean data_frac, val PSNR at every val
step; the LAST val step (past the 1000-step warmup, i.e. at the peak lr) is compared with step 0
(drop > 0.3 dB, or a non-finite value -> FAIL), grad_norm / amp_scale per log row, abort events,
startup / checkpoint time (events.jsonl),
the last diag.jsonl row. Then GPU use, the step-0 identity check (every variant starts from the same
checkpoint with zero-initialised extras, so step-0 val PSNR must match) and ITERS: one value for all
four models, sized on the SLOWEST run so that one Kaggle session (12 h) holds setup + startup + val +
checkpoints + training with about 1 h to spare.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gpu-log", default=None)
    ap.add_argument("--session-hours", type=float, default=12.0, help="Kaggle session limit")
    ap.add_argument("--reserve-hours", type=float, default=1.0, help="spare time left unused")
    ap.add_argument("--setup-min", type=float, default=10.0, help="bootstrap + data input + val generation")
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--max-drop-db", type=float, default=0.3)
    ap.add_argument("--warmup", type=int, default=1000, help="val freezing is only checked after the warmup")
    ap.add_argument("--freeze-db", type=float, default=0.01, help="|val change| below this after warmup = frozen")
    a = ap.parse_args()

    speeds, val_secs, ckpt_secs, startups, step0, failures = [], [], [], [], {}, []
    for run in map(Path, a.runs):
        cur, val, diag, ev = (rows(run / n) for n in ("curves.csv", "val.jsonl", "diag.jsonl", "events.jsonl"))
        print(f"\n=== {run.name} ===")
        start = next((e for e in ev if e["event"] == "start"), {})
        if start:
            print(f"lr {start.get('lr')} | batch {start.get('batch_size')} x accum {start.get('grad_accum', 1)} | "
                  f"amp {start.get('amp')} | workers {start.get('num_workers')} | "
                  f"account {start.get('account')} | host {start.get('host')} | startup {start.get('startup_sec')} s")
            startups.append(start.get("startup_sec", 0.0))
        ckpt_secs += [e["ckpt_sec"] for e in ev if e["event"] == "ckpt"]
        bad_events = [e for e in ev if e["event"] in ("aborted_nonfinite", "ckpt_skipped_nonfinite", "aborted_val_drop",
                                                       "aborted_stalled", "aborted_amp_scale")]
        if bad_events:
            failures.append(f"{run.name}: events {[(e['event'], e['step']) for e in bad_events]}")
        if len(cur) >= 2:
            first, last = cur[1] if len(cur) > 2 else cur[0], cur[-1]
            its = (last["step"] - first["step"]) / max(last["sec"] - first["sec"], 1e-9)
            speeds.append(its)
            frac = sum(r["data_frac"] for r in cur[1:]) / max(len(cur) - 1, 1)
            print(f"steps {cur[0]['step']}..{last['step']} | {its:.3f} it/s | mean data_frac {frac:.2f}")
            print("last loss parts: " + ", ".join(f"{k}={last[k]:.4f}" for k in ("charbonnier", "l_map", "loss") if k in last))
            if "grad_norm" in last:
                print("grad_norm by step: " + ", ".join(f"{r['step']}:{r['grad_norm']:.3g}" for r in cur))
                scale_min = min(r["amp_scale"] for r in cur)
                print(f"amp_scale min {scale_min:.3g} | max nonfinite_run {max(r.get('nonfinite_run', 0) for r in cur)}")
                if start.get("amp", True) and scale_min < 1.0:  # pilot 3: the scale collapsed towards 0
                    failures.append(f"{run.name}: amp_scale collapsed to {scale_min:.3g}")
                zero_run = best = 0
                for r in cur:
                    zero_run = zero_run + 1 if r["grad_norm"] == 0 else 0
                    best = max(best, zero_run)
                if best >= 2:
                    failures.append(f"{run.name}: grad_norm = 0 in {best} consecutive log rows (training stalled)")
            bad = [r["step"] for r in cur if not all(math.isfinite(r[k]) for k in ("charbonnier", "loss") if k in r)]
            if bad:
                failures.append(f"{run.name}: non-finite training loss from step {bad[0]}")
        else:
            print("not enough curves rows")
        for v in val:
            print("val: " + ", ".join(f"{k}={v[k]:.4f}" if isinstance(v[k], float) else f"{k}={v[k]}" for k in v))
            val_secs.append(v.get("val_sec", 0.0))
        v0 = next((v for v in val if v["step"] == 0), None)
        if v0:
            step0[run.name] = v0["val_psnr"]
        if v0 and len(val) > 1:
            end = val[-1]["val_psnr"]
            drop = v0["val_psnr"] - end
            if not math.isfinite(end):  # NaN / inf is a failure, never "within limit"
                msg = "FAIL: val PSNR is not finite (diverged)"
                failures.append(f"{run.name}: val PSNR {end} at step {val[-1]['step']}")
            elif drop > a.max_drop_db:
                msg = f"FAIL: DROP {drop:.2f} dB > {a.max_drop_db} -> lower lr (range 1e-4..2e-4) and re-pilot"
                failures.append(f"{run.name}: val drop {drop:.2f} dB")
            else:
                msg = "within limit, keep lr"
            print(f"val PSNR step 0 -> {val[-1]['step']}: {v0['val_psnr']:.3f} -> {end:.3f} | {msg}")
        if len(val) >= 2 and val[-1]["step"] > a.warmup and math.isfinite(val[-1]["val_psnr"]):
            change = abs(val[-1]["val_psnr"] - val[-2]["val_psnr"])
            if change < a.freeze_db:  # report-only check: the trainer does not stop on this
                failures.append(f"{run.name}: val frozen after warmup ({val[-2]['step']} -> {val[-1]['step']}: "
                                f"{change:.4f} dB < {a.freeze_db})")
        if diag:
            d = diag[-1]
            stages = [k[len("w_sft_"):] for k in d if k.startswith("w_sft_")]
            print(f"diag rows: {len(diag)} (steps {[r['step'] for r in diag]})")
            if "mid_beta_abs" in d:
                print("middle block mean|beta| by step: " + ", ".join(f"{r['step']}:{r['mid_beta_abs']:.2e}" for r in diag)
                      + f" | mean|gamma| last {d['mid_gamma_abs']:.2e}")
            print("stage    ||W_sft||   |dgamma_s|   |beta_s|")
            for s in stages:
                print(f"{s:8s} {d[f'w_sft_{s}']:10.5f} {d.get(f'dgamma_{s}', float('nan')):11.6f} "
                      f"{d.get(f'beta_{s}', float('nan')):10.6f}")
        else:
            print("diag: none (expected for Baseline / Baseline-L)")

    if a.gpu_log and Path(a.gpu_log).exists():
        util, mem = {}, {}
        for line in csv.reader(open(a.gpu_log)):
            try:
                i, u, m = int(line[0]), float(line[1].split()[0]), float(line[2].split()[0])
            except (ValueError, IndexError):
                continue
            util.setdefault(i, []).append(u)
            mem.setdefault(i, []).append(m)
        print("\n=== GPU ===")
        for i in sorted(util):
            busy = [u for u in util[i] if u > 0] or [0.0]
            print(f"GPU {i}: mean util {sum(busy) / len(busy):.0f}% over busy samples, peak mem {max(mem[i]):.0f} MiB")

    if len(step0) > 1:
        spread = max(step0.values()) - min(step0.values())
        print(f"\n=== step-0 identity check ===\n{step0} | spread {spread:.4f} dB "
              f"({'OK' if spread < 0.01 else 'MISMATCH: variants do not start from the same function'})")

    if speeds:
        slow = min(speeds)
        val_sec = max(val_secs) if val_secs else 0.0
        ckpt_sec = max(ckpt_secs) if ckpt_secs else 5.0
        startup = max(startups) if startups else 120.0
        avail = (a.session_hours - a.reserve_hours) * 3600 - a.setup_min * 60 - startup - val_sec  # val at step 0
        per_it = 1 / slow + val_sec / a.val_every + ckpt_sec / a.ckpt_every
        iters = avail / per_it
        fixed = int(math.floor(iters / 1000) * 1000)
        hours = (a.setup_min * 60 + startup + val_sec + fixed * per_it) / 3600
        print(f"\n=== budget (QĐ-04) ===\nslowest {slow:.3f} it/s | val pass {val_sec:.0f} s every {a.val_every} | "
              f"ckpt {ckpt_sec:.1f} s every {a.ckpt_every} | startup {startup:.0f} s | setup {a.setup_min:.0f} min\n"
              f"session {a.session_hours} h - reserve {a.reserve_hours} h -> {iters:.0f} it -> ITERS = {fixed} "
              f"(expected session length {hours:.2f} h)")
        print("note: SFT-Free has the SFT-Map architecture; Baseline is the lightest model. "
              "If the pilot did not include Baseline-L or SFT-Map, the slowest model was not measured.")

    if len(step0) > 1 and max(step0.values()) - min(step0.values()) >= 0.01:
        failures.append("step-0 identity mismatch")
    print("\n=== PILOT VERDICT ===\n" + ("PASS" if not failures else "FAIL\n- " + "\n- ".join(failures)))


if __name__ == "__main__":
    main()
