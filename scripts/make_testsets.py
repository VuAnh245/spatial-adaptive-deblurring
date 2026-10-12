"""Generate the fixed test sets in parallel (one generate_testsets.py process per group) and check their sha.

    python scripts/make_testsets.py --temp /kaggle/temp/synthetic --registry results/testsets_sha.json
    python scripts/make_testsets.py --temp /kaggle/temp/synthetic --noise 2 4 --expect results/testsets_sha.json

Every split folder data/synthetic/<split> becomes a symlink into ``--temp`` (full-resolution .npz files take
roughly 10-20 MB each, too much for /kaggle/working, which is the notebook output). Splits and seeds are those of
generate_testsets.py, so a split is byte-identical whether generated here, alone, or in another session; the
"content sha256" it prints is collected into ``--registry``. ``--expect`` (a registry file or JSON text) makes any
mismatch in image count or sha a failure (exit 1).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from spatial_deblur.utils.config import load_config

GROUPS = [["test_in"], ["test_ood", "test_straight", "test_curved"]]
LINE = re.compile(r"^(?P<split>\S+): (?P<n>\d+) images, content sha256 (?P<sha>[0-9a-f]+)", re.M)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/SA-13_a2.yaml")
    ap.add_argument("--temp", default=None, help="real location of the split folders (symlinked from data/synthetic)")
    ap.add_argument("--noise", nargs="*", type=float, default=[], help="SA-26: extra test_in copies, noise x k")
    ap.add_argument("--registry", default=None, help="write {split: {n, sha}} here")
    ap.add_argument("--expect", default=None, help="registry file or JSON text to compare with")
    a = ap.parse_args()
    root = Path(load_config(a.config)["data"]["synthetic_root"])
    jobs = [(g, 1.0) for g in GROUPS] + [(["test_in"], k) for k in a.noise]
    names = [s + ("" if k == 1 else f"_noise{k:g}x") for g, k in jobs for s in g]

    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        link = root / name
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        if a.temp:
            real = Path(a.temp) / name
            if real.exists():
                shutil.rmtree(real)
            real.mkdir(parents=True)
            link.symlink_to(real.resolve(), target_is_directory=True)
    if a.temp:
        free = shutil.disk_usage(a.temp).free / 1e9
        print(f"free disk at {a.temp}: {free:.0f} GB for {len(names)} splits", flush=True)

    t0, procs = time.time(), []
    for group, k in jobs:
        cmd = [sys.executable, "-u", "scripts/generate_testsets.py", "--config", a.config, "--splits", *group]
        if k != 1:
            cmd += ["--noise-mult", f"{k:g}"]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    found = {}
    for p in procs:
        text = p.communicate()[0]
        if p.returncode != 0:
            print(text[-2000:])
            raise SystemExit(f"generate_testsets.py failed (rc {p.returncode})")
        found.update({m["split"]: {"n": int(m["n"]), "sha": m["sha"]} for m in LINE.finditer(text)})
    print(f"generated in {(time.time() - t0) / 60:.1f} min")
    for name in names:
        r = found.get(name)
        print(f"{name:20s} {r['n'] if r else '-':>5} images  sha {r['sha'] if r else 'MISSING'}")
    assert set(names) <= set(found), f"no sha line for {sorted(set(names) - set(found))}"

    if a.registry:
        Path(a.registry).parent.mkdir(parents=True, exist_ok=True)
        Path(a.registry).write_text(json.dumps(found, indent=1, sort_keys=True))
        print(f"registry -> {a.registry}")
    if a.expect:
        exp = json.loads(Path(a.expect).read_text() if Path(a.expect).exists() else a.expect)
        bad = {s: (found.get(s), v) for s, v in exp.items() if s in names and found.get(s) != v}
        if bad:
            for s, (got, want) in bad.items():
                print(f"MISMATCH {s}: got {got}, registered {want}")
            raise SystemExit(1)
        print(f"sha check OK for {sorted(s for s in exp if s in names)}")


if __name__ == "__main__":
    main()
