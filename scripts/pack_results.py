"""Pack Kaggle results into zips with a SHA-256 manifest, to bring them to the local machine.

    # Z1: the four training runs (attached sad-train versions; may be run once per attached version)
    python scripts/pack_results.py runs --runs SA-13_seed1 SA-10_seed1 SA-12_seed1 SA-11_seed1 \
        --source SA-13_seed1="sad-train v2" SA-10_seed1="sad-train v2" SA-12_seed1="sad-train v3" SA-11_seed1="sad-train v3"
    # Z2: one result folder
    python scripts/pack_results.py dir --dir results/eval --zip /kaggle/working/eval_results.zip

``runs`` copies every run found under /kaggle/input into a staging folder (so the inputs can be swapped between
two calls if Kaggle attaches one notebook version at a time), and builds the zips once all runs are staged:
  runs_light.zip  <run>/{final.pth, config.yaml, curves.csv, val.jsonl, diag.jsonl, events.jsonl}, <run>.log
  runs_last.zip   <run>/last.pth (backup for resuming)
MANIFEST.txt (in both zips and next to them): '#' header lines with source, commit, account, host and last event
of every run, then one line per file: path<TAB>bytes<TAB>sha256. Paths are relative to the extraction folder.
No metric is printed (QĐ-04 order).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

from spatial_deblur.data.manifests import file_sha256

LIGHT = ["final.pth", "config.yaml", "curves.csv", "val.jsonl", "diag.jsonl", "events.jsonl"]


def find_run(run: str, roots: list[Path]) -> Path | None:
    hits = [p.parent for r in roots for p in r.glob(f"**/{run}/final.pth")]
    hits = sorted(set(hits))
    assert len(hits) <= 1, f"{run} found more than once: {hits}"
    return hits[0] if hits else None


def stage_run(run: str, src: Path, stage: Path) -> None:
    dst = stage / run
    dst.mkdir(parents=True, exist_ok=True)
    for name in LIGHT + ["last.pth"]:
        f = src / name
        if f.exists() and not ((dst / name).exists() and (dst / name).stat().st_size == f.stat().st_size):
            shutil.copy2(f, dst / name)
    log = src.parent / f"{run}.log"  # sad-train T3 writes results/<run>.log next to the run folder
    if log.exists():
        shutil.copy2(log, stage / log.name)


def events(run_dir: Path) -> list[dict]:
    p = run_dir / "events.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def manifest_lines(root: Path, files: list[Path]) -> list[str]:
    return [f"{f.relative_to(root).as_posix()}\t{f.stat().st_size}\t{file_sha256(f)}" for f in files]


def write_zip(path: Path, root: Path, files: list[Path], manifest: Path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for f in files + [manifest]:
            kind = zipfile.ZIP_STORED if f.suffix in (".pth", ".png", ".zip") else zipfile.ZIP_DEFLATED
            z.write(f, f.relative_to(root).as_posix() if f != manifest else "MANIFEST.txt", compress_type=kind)
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
    assert bad is None, f"corrupt member {bad} in {path}"


def cmd_runs(a) -> None:
    stage, out = Path(a.stage), Path(a.out)
    roots = [Path(r) for r in a.roots]
    source = dict(s.split("=", 1) for s in a.source)
    status = {}
    for run in a.runs:
        src = find_run(run, roots)
        if src is not None:
            stage_run(run, src, stage)
        staged = stage / run / "final.pth"
        status[run] = "staged" if staged.exists() else "MISSING"
        ev = events(stage / run)
        last = ev[-1] if ev else {}
        print(f"{run:14s} {status[run]:8s} from {src or '(staging only)'} | last event "
              f"{last.get('event')} step {last.get('step')}", flush=True)
    missing = [r for r, s in status.items() if s != "staged"]
    if missing:
        print(f"\nnot complete: {missing}. Attach the sad-train version that holds them and run this cell again "
              f"(runs already staged are kept in {stage}).")
        sys.exit(2)

    head = [f"# packed {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"]
    for run in a.runs:
        ev = events(stage / run)
        start = next((e for e in ev if e.get("event") == "start"), {})
        last = ev[-1] if ev else {}
        head.append(f"# {run}: source {source.get(run, '?')} | commit {str(start.get('git_commit', '?'))[:7]} | "
                    f"account {start.get('account', '?')} | host {start.get('host', '?')} | "
                    f"last event {last.get('event')} step {last.get('step')}")
    light = sorted([stage / r / n for r in a.runs for n in LIGHT if (stage / r / n).exists()] +
                   [stage / f"{r}.log" for r in a.runs if (stage / f"{r}.log").exists()])
    lastp = sorted(stage / r / "last.pth" for r in a.runs if (stage / r / "last.pth").exists())
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "MANIFEST.txt"
    manifest.write_text("\n".join(head + manifest_lines(stage, light + lastp)) + "\n", encoding="utf-8")
    write_zip(out / "runs_light.zip", stage, light, manifest)
    write_zip(out / "runs_last.zip", stage, lastp, manifest)
    print("\n" + "\n".join(head))
    print(f"\nfiles: {len(light)} light + {len(lastp)} last.pth")
    for z in ("runs_light.zip", "runs_last.zip", "MANIFEST.txt"):
        print(f"{z:16s} {(out / z).stat().st_size / 1e6:9.1f} MB  -> {out / z}")


def cmd_dir(a) -> None:
    root, zpath = Path(a.dir), Path(a.zip)
    files = sorted(f for f in root.rglob("*") if f.is_file() and not f.name.endswith(".part")
                   and f.name != "MANIFEST.txt")
    manifest = root / "MANIFEST.txt"
    head = [f"# packed {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} from {root}"] + \
           [f"# {x}" for x in a.note]
    manifest.write_text("\n".join(head + manifest_lines(root, files)) + "\n", encoding="utf-8")
    zpath.parent.mkdir(parents=True, exist_ok=True)
    write_zip(zpath, root, files, manifest)
    print(f"{len(files)} files -> {zpath} ({zpath.stat().st_size / 1e6:.1f} MB), manifest {manifest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("runs")
    r.add_argument("--runs", nargs="+", required=True)
    r.add_argument("--source", nargs="*", default=[], help="run=label, e.g. SA-13_seed1='sad-train v2'")
    r.add_argument("--roots", nargs="+", default=["/kaggle/input"])
    r.add_argument("--stage", default="/kaggle/working/collected")
    r.add_argument("--out", default="/kaggle/working/pack")
    d = sub.add_parser("dir")
    d.add_argument("--dir", required=True)
    d.add_argument("--zip", required=True)
    d.add_argument("--note", nargs="*", default=[], help="extra header lines (commit, notebook version, ...)")
    a = ap.parse_args()
    cmd_runs(a) if a.cmd == "runs" else cmd_dir(a)


if __name__ == "__main__":
    main()
