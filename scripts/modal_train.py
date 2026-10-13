"""Training on Modal (QĐ-10): SA-32, SA-33, SA-14 and the platform check, with the recipe of QĐ-04 unchanged.

    python -m modal run scripts/modal_train.py::prepare                       # once: data into the Volume
    python -m modal run scripts/modal_train.py::verify_data                   # every PNG + ckpt md5; re-download broken
    python -m modal run --detach scripts/modal_train.py::train --config configs/experiments/SA-12_a1.yaml --tag modalcheck --stop-at 2500
    python -m modal run --detach scripts/modal_train.py::train --config configs/experiments/SA-32_a2_oracle_train.yaml --iterations 1500 --val-every 750 --ckpt-every 750 --tag pilot
    python -m modal run --detach scripts/modal_train.py::train --config configs/experiments/SA-32_a2_oracle_train.yaml
    python -m modal volume get sad-runs results/SA-32_seed1 results/modal/
    python -m modal run scripts/modal_train.py::make_testsets                 # once: test sets, sha must match Kaggle
    python -m modal run --detach scripts/modal_train.py::evaluate_all --max-tier 4 --shards 2
    python -m modal volume get sad-runs eval results/modal_eval/

Guarantees (each checked, not assumed):
- code: the image holds src/, scripts/, configs/, analysis/ of the local repo; launching refuses a working tree with
  uncommitted changes in those folders or a HEAD that is not on origin/main, and GIT_COMMIT = HEAD goes into config.yaml;
- data: xunanhv/sad-data (GoPro subset + public checkpoint, the input of every Kaggle run) and the ORIGINAL val40 from
  xunanhv/sad-train-remaining-input (content sha 96ca0c00be35f399, recomputed from the files before every run);
  manifests are rebuilt with build_manifests.py and must give train 1903 / val 200 / test 1111;
- runtime: torch 2.11.0+cu128, numpy 2.1.3, scipy 1.16.3, scikit-image 0.25.2 as on Kaggle (pair 1-3); TF32 is
  switched off inside train.py; the GPU name is written into config.yaml by train.py;
- results: /vol/results on the Volume "sad-runs", committed every 5 minutes; a rerun with the same config and tag
  resumes from last.pth (same data stream position). ``--stop-at N`` sends SIGTERM once val.jsonl reaches step N
  (train.py saves last.pth and exits), used for the platform check against Kaggle's val at step 2500.
Secret "kaggle" (KAGGLE_USERNAME, KAGGLE_KEY) is needed only by ``prepare``.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
APP = modal.App("sad-train")
VOL = modal.Volume.from_name("sad-runs", create_if_missing=True)
EXPECTED_VAL = "96ca0c00be35f399"
CODE_DIRS = ["src", "scripts", "configs", "analysis"]


def _local_commit() -> str:
    git = lambda *a: subprocess.check_output(["git", "-C", str(REPO), *a], text=True).strip()  # noqa: E731
    dirty = git("status", "--porcelain", "--", *CODE_DIRS)
    if dirty:
        raise SystemExit(f"uncommitted changes in {CODE_DIRS} - commit and push first:\n{dirty}")
    head = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", head):  # lists remote branches holding HEAD; empty = not pushed
        raise SystemExit(f"HEAD {head[:7]} is not on any remote branch - push first")
    return head


COMMIT = _local_commit() if modal.is_local() else os.environ.get("GIT_COMMIT", "unknown")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy==2.1.3", "scipy==1.16.3", "scikit-image==0.25.2", "pyyaml==6.0.3", "pillow", "kaggle")
    .env({"GIT_COMMIT": COMMIT, "PYTHONPATH": "/root/sad/src", "RUN_ACCOUNT": "modal:vuanhit02042k5"})
)
for d in CODE_DIRS:
    image = image.add_local_dir(REPO / d, f"/root/sad/{d}", ignore=["__pycache__", "*.pyc"])


def _val_sha(folder: Path) -> tuple[int, str]:
    import numpy as np
    files = sorted(folder.glob("*.npz"))
    h = hashlib.sha256()
    for f in files:
        z = np.load(f)
        h.update(np.ascontiguousarray(z["blur"]).tobytes())
        h.update(np.ascontiguousarray(z["smap"]).tobytes())
    return len(files), h.hexdigest()[:16]


@APP.function(image=image, volumes={"/vol": VOL}, secrets=[modal.Secret.from_name("kaggle")], cpu=4.0,
              memory=16384, timeout=3 * 3600)
def prepare() -> dict:
    data, bundle = Path("/vol/sad-data"), Path("/vol/bundle")
    if not (data / "DATA_INFO.txt").exists():
        subprocess.run(["kaggle", "kernels", "output", "xunanhv/sad-data", "-p", str(data), "-q"], check=True)
    if not (bundle / "data/synthetic/val").is_dir():
        subprocess.run(["kaggle", "datasets", "download", "xunanhv/sad-train-remaining-input", "-p", str(bundle),
                        "--unzip", "-q"], check=True)
    n_train = len(list((data / "gopro/train").glob("*/sharp/*.png")))
    n_test = len(list((data / "gopro/test").glob("*/blur/*.png")))
    n_val, val_sha = _val_sha(bundle / "data/synthetic/val")
    ckpt = data / "ckpt/NAFNet-GoPro-width32.pth"
    info = {"train_sharp": n_train, "test_pairs": n_test, "val_files": n_val, "val_sha": val_sha,
            "ckpt_bytes": ckpt.stat().st_size if ckpt.exists() else None,
            "data_info": (data / "DATA_INFO.txt").read_text()[:600]}
    VOL.commit()
    print(json.dumps(info, indent=1))
    assert n_train == 2103 and n_test == 1111 and n_val == 40 and val_sha == EXPECTED_VAL, info
    return info


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _bad_pngs(root: Path, deep: bool) -> list[str]:
    """PNG files that are not images. 2026-10-13: the Kaggle download stored a '502 Server Error' HTML page as
    gopro/train/GOPR0374_11_00/sharp/000016.png (332 bytes). deep=True also checks every chunk CRC (PIL verify)."""
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    def bad(p: Path) -> bool:
        try:
            with p.open("rb") as f:
                if f.read(8) != PNG_MAGIC:
                    return True
            if deep:
                with Image.open(p) as im:
                    im.verify()
            return False
        except Exception:
            return True

    files = sorted(root.rglob("*.png"))
    with ThreadPoolExecutor(16) as ex:
        flags = list(ex.map(bad, files))
    return [str(p.relative_to(root)) for p, b in zip(files, flags) if b]


@APP.function(image=image, volumes={"/vol": VOL}, secrets=[modal.Secret.from_name("kaggle")], cpu=4.0,
              memory=16384, timeout=2 * 3600)
def verify_data(repair: bool = True) -> dict:
    """Deep check of /vol/sad-data (every PNG + checkpoint md5 from DATA_INFO.txt); re-download broken files one
    by one (exact --file-pattern, forced, up to 6 tries 30 s apart). Fails if anything is still broken."""
    import re
    data = Path("/vol/sad-data")
    bad = _bad_pngs(data, deep=True)
    print(f"broken PNG before repair: {len(bad)} {bad[:20]}", flush=True)
    for rel in bad if repair else []:
        for attempt in range(6):
            subprocess.run(["kaggle", "kernels", "output", "xunanhv/sad-data", "-p", str(data), "-o", "-q",
                            "--file-pattern", f"^{re.escape(rel)}$"], check=False)
            if not _bad_pngs_one(data / rel):
                print(f"repaired {rel} (try {attempt + 1})", flush=True)
                break
            time.sleep(30)
    still = _bad_pngs(data, deep=True)
    md5 = hashlib.md5((data / "ckpt/NAFNet-GoPro-width32.pth").read_bytes()).hexdigest()
    want = re.search(r"ckpt md5: ([0-9a-f]{32})", (data / "DATA_INFO.txt").read_text())
    info = {"png_total": len(list(data.rglob("*.png"))), "broken_before": len(bad), "broken_after": len(still),
            "still_broken": still[:20], "ckpt_md5_ok": bool(want and md5 == want.group(1))}
    VOL.commit()
    print(json.dumps(info, indent=1), flush=True)
    assert not still and info["ckpt_md5_ok"], info
    return info


def _bad_pngs_one(p: Path) -> bool:
    from PIL import Image
    try:
        with p.open("rb") as f:
            if f.read(8) != PNG_MAGIC:
                return True
        with Image.open(p) as im:
            im.verify()
        return False
    except Exception:
        return True


# ----------------------------------------------------------------------------------------------- evaluation (QĐ-09/10)
# Official test sets = the files generated ONCE on Modal (2026-10-13, commit 0d2d8ce) and kept in /vol/synthetic; never
# regenerated. Kaggle (sad-checks V3, same numpy 2.1.3 / scipy 1.16.3) gave the same sha for the single-region splits
# (test_straight, test_curved) but not for test_in / test_ood (13b027f23ec17c0f / a6e491a2502d099c): the multi-region
# mask path (spline zoom + exp) is not bit-identical across CPUs. regen_val_diff measures the size of that difference.
REGISTERED_TESTSETS = {"test_in": {"n": 400, "sha": "0a83c5daad9657cb"}, "test_ood": {"n": 100, "sha": "978cc452c485da7f"},
                       "test_straight": {"n": 50, "sha": "d6936aac831e864f"}, "test_curved": {"n": 50, "sha": "ee6b893c9aa2c23a"}}
SPLITS = list(REGISTERED_TESTSETS)
# Kaggle-trained checkpoints uploaded to /vol/ckpts/<run>/final.pth (scripts in the evaluation section of docs).
KAGGLE_RUNS = ["SA-10_seed1", "SA-11_seed1", "SA-12_seed1", "SA-13_seed1", "SA-12_seed2", "SA-13_seed2",
               "SA-12_seed3", "SA-13_seed3"]
MODAL_RUNS = ["SA-14_seed1", "SA-32_seed1", "SA-33_seed1"]
# SFT-Free seed 1 retrained on Modal (train --config SA-12_a1.yaml --tag modal) is evaluated as SA-35 (platform control).
MODAL_ALIASES = {"SA-12_seed1_modal": "SA-35_seed1"}


def _link_repo(synthetic: bool) -> None:
    """cwd /root/sad with data/GOPRO_Large, checkpoints, manifests and (optionally) the persisted test sets."""
    os.chdir("/root/sad")
    links = [("data/GOPRO_Large", "/vol/sad-data/gopro"),
             ("checkpoints/NAFNet-GoPro-width32.pth", "/vol/sad-data/ckpt/NAFNet-GoPro-width32.pth")]
    if synthetic:
        links += [(f"data/synthetic/{s}", f"/vol/synthetic/{s}") for s in SPLITS]
    for link, target in links:
        assert Path(target).exists(), f"{target} missing"
        Path(link).parent.mkdir(parents=True, exist_ok=True)
        if not Path(link).is_symlink():
            Path(link).symlink_to(target)
    man = subprocess.run(["python", "scripts/build_manifests.py", "--gopro", "data/GOPRO_Large"],
                         capture_output=True, text=True, check=True).stdout
    assert "train 1903" in man and "test 1111" in man, man


@APP.function(image=image, volumes={"/vol": VOL}, cpu=8.0, memory=32768, timeout=3 * 3600)
def make_testsets() -> dict:
    """Generate the four synthetic test sets ONCE into /vol/synthetic and require the sha registered on Kaggle."""
    _link_repo(synthetic=False)
    Path("/vol/eval").mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["python", "-u", "scripts/make_testsets.py", "--temp", "/vol/synthetic",
                        "--registry", "/vol/eval/testsets_sha.json", "--expect", json.dumps(REGISTERED_TESTSETS)],
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
    VOL.commit()
    assert r.returncode == 0, "test sets differ from the registered sha (or generation failed)"
    return json.loads(Path("/vol/eval/testsets_sha.json").read_text())


@APP.function(image=image, volumes={"/vol": VOL}, cpu=4.0, memory=16384, timeout=3600)
def regen_val_diff() -> dict:
    """Regenerate val (40 images) on this platform into a scratch folder and compare pixel by pixel with the ORIGINAL
    Kaggle val40 (sha 96ca0c00be35f399): how large is the cross-CPU difference of the generator?"""
    import numpy as np
    _link_repo(synthetic=False)
    scratch = Path("/tmp/regen")
    Path("data/synthetic").mkdir(parents=True, exist_ok=True)
    if Path("data/synthetic/val").is_symlink():
        Path("data/synthetic/val").unlink()
    Path("data/synthetic/val").symlink_to(scratch, target_is_directory=True)
    scratch.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["python", "scripts/generate_testsets.py", "--config", "configs/experiments/SA-13_a2.yaml",
                          "--splits", "val"], capture_output=True, text=True, check=True).stdout
    rows = []
    for f in sorted(Path("/vol/bundle/data/synthetic/val").glob("*.npz")):
        a, b = np.load(f), np.load(scratch / f.name)
        d_blur = np.abs(a["blur"].astype(np.float64) - b["blur"].astype(np.float64))
        d_smap = np.abs(a["smap"].astype(np.float64) - b["smap"].astype(np.float64))
        mse = float((d_blur ** 2).mean())
        rows.append({"file": f.name, "regions": int(a["regions"]), "blur_max": float(d_blur.max()),
                     "blur_mean": float(d_blur.mean()), "smap_max": float(d_smap.max()),
                     "psnr_old_vs_new": 10 * np.log10(1 / mse) if mse > 0 else float("inf"),
                     "identical": bool(d_blur.max() == 0 and d_smap.max() == 0)})
    multi = [r for r in rows if r["regions"] > 1]
    info = {"regen_line": out.strip().splitlines()[-1], "images": len(rows), "identical": sum(r["identical"] for r in rows),
            "blur_max_abs_diff": max(r["blur_max"] for r in rows), "smap_max_abs_diff": max(r["smap_max"] for r in rows),
            "min_psnr_old_vs_new_R>=2": min(r["psnr_old_vs_new"] for r in multi) if multi else None}
    Path("/vol/eval/regen_val_diff.json").parent.mkdir(parents=True, exist_ok=True)
    Path("/vol/eval/regen_val_diff.json").write_text(json.dumps({"summary": info, "rows": rows}, indent=1))
    VOL.commit()
    print(json.dumps(info, indent=1), flush=True)
    return info


@APP.function(image=image, volumes={"/vol": VOL}, cpu=4.0, memory=16384, timeout=3600)
def eval_plan(max_tier: int, shards: int) -> dict:
    """ckpts.json (Kaggle runs uploaded to /vol/ckpts, Modal runs that finished, public checkpoint as A0), checkpoint
    sha256 into /vol/eval/meta.json, then eval_queue plan. Seed-1 GoPro CSVs verified on Kaggle are copied in by the
    user (they count as done, never recomputed)."""
    import torch
    _link_repo(synthetic=True)
    for s, want in REGISTERED_TESTSETS.items():
        assert len(list(Path(f"data/synthetic/{s}").glob("*.npz"))) == want["n"], f"{s} incomplete - run make_testsets"
    on_vol = json.loads(Path("/vol/eval/testsets_sha.json").read_text())
    assert all(on_vol.get(s) == want for s, want in REGISTERED_TESTSETS.items()), \
        f"test sets on the Volume are not the official ones: {on_vol}"
    ck = {}
    for run in KAGGLE_RUNS:
        p = Path(f"/vol/ckpts/{run}/final.pth")
        assert p.exists(), f"{p} missing - upload it first"
        ck[run] = p
    for run in MODAL_RUNS:
        p = Path(f"/vol/results/{run}/final.pth")
        if p.exists():
            ck[run] = p
    for run, alias in MODAL_ALIASES.items():
        p = Path(f"/vol/results/{run}/final.pth")
        if p.exists():
            ck[alias] = p
    pub = Path("/vol/ckpts/SA-00_public_A0.pth")
    if not pub.exists():
        import sys
        sys.path.insert(0, "/root/sad/src")
        from spatial_deblur.models.nafnet import load_official_checkpoint
        from spatial_deblur.modulation.variants import build_variant
        m = build_variant("A0")
        rep = load_official_checkpoint(m, "checkpoints/NAFNet-GoPro-width32.pth", strict=False)
        assert not rep.missing_keys and not rep.unexpected_keys, rep
        pub.parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), pub)
    ck["SA-00_seed1"] = pub
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    meta = {"commit": os.environ.get("GIT_COMMIT"), "gpu": "NVIDIA L4", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "testsets": json.loads(Path("/vol/eval/testsets_sha.json").read_text()),
            "ckpt": {run: {"path": str(p), "sha256": sha(p)} for run, p in ck.items()}}
    Path("/vol/eval/meta.json").write_text(json.dumps(meta, indent=1))
    keys = {run.replace("_seed1", ""): str(p) for run, p in ck.items()}  # eval_queue: bare ID = seed 1
    Path("/vol/eval/ckpts.json").write_text(json.dumps(keys, indent=1))
    r = subprocess.run(["python", "scripts/eval_queue.py", "plan", "--ckpts", "/vol/eval/ckpts.json", "--out", "/vol/eval",
                        "--shards", str(shards), "--max-tier", str(max_tier), "--sec-per-img", "1.3"],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:], flush=True)
    VOL.commit()
    assert r.returncode == 0, "plan failed"
    return {"ckpt": {k: v["sha256"][:16] for k, v in meta["ckpt"].items()}, "plan": r.stdout[-1500:]}


@APP.function(image=image, volumes={"/vol": VOL}, gpu="L4", cpu=8.0, memory=32768, timeout=8 * 3600)
def eval_shard(shard: int, shards: int, hours: float) -> str:
    _link_repo(synthetic=True)
    deadline = time.time() + hours * 3600
    stop = threading.Event()

    def keeper():
        while not stop.wait(300):
            VOL.commit()

    threading.Thread(target=keeper, daemon=True).start()
    r = subprocess.run(["python", "-u", "scripts/eval_queue.py", "run", "--shard", str(shard), "--shards", str(shards),
                        "--device", "cuda:0", "--deadline", str(deadline), "--out", "/vol/eval", "--sec-per-img", "1.3"],
                       capture_output=True, text=True)
    stop.set()
    VOL.commit()
    print(r.stdout[-4000:], r.stderr[-2000:], flush=True)
    return r.stdout[-2000:]


@APP.function(image=image, volumes={"/vol": VOL}, cpu=1.0, memory=4096, timeout=10 * 3600)
def evaluate_all(max_tier: int = 4, shards: int = 2, hours: float = 6.0) -> dict:
    """Orchestrator (run with --detach): plan, then the shards in parallel on separate L4 containers. No metric is
    printed; tables are made afterwards with analysis/eval_tables.py on the downloaded CSVs."""
    plan = eval_plan.remote(max_tier, shards)
    print(json.dumps(plan, indent=1), flush=True)
    outs = list(eval_shard.map(range(shards), kwargs={"shards": shards, "hours": hours}))
    VOL.reload()
    status = {}
    for k in range(shards):
        p = Path(f"/vol/eval/queue_shard{k}.jsonl")
        for x in p.read_text().splitlines() if p.exists() else []:
            row = json.loads(x)
            status[row["status"]] = status.get(row["status"], 0) + 1
    print("queue status counts:", status, flush=True)
    return {"status": status, "tails": outs}


@APP.function(image=image, volumes={"/vol": VOL}, gpu="L4", cpu=8.0, memory=32768, timeout=3 * 3600)
def map_sweep(split: str = "test_in") -> str:
    """SA-34 (QĐ-11): analysis/map_sweep.py on the persisted test set with SA-32 (+ SA-33 / SA-13 estimators)."""
    _link_repo(synthetic=True)
    ck = {"sa32": "/vol/results/SA-32_seed1/final.pth", "sa33": "/vol/results/SA-33_seed1/final.pth",
          "sa13": "/vol/ckpts/SA-13_seed1/final.pth"}
    for k, p in ck.items():
        assert Path(p).exists(), f"{p} missing - wait until the run has finished"
    out = Path(f"/vol/eval/map_sweep_{split}")
    r = subprocess.run(["python", "-u", "analysis/map_sweep.py", "--sa32", ck["sa32"], "--sa33", ck["sa33"],
                        "--sa13", ck["sa13"], "--split", split, "--out", str(out)], capture_output=True, text=True)
    sha = {k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in ck.items()}
    (out / "meta.json").write_text(json.dumps({"commit": os.environ.get("GIT_COMMIT"), "ckpt_sha256": sha,
                                               "testsets": REGISTERED_TESTSETS}, indent=1)) if out.exists() else None
    VOL.commit()
    print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
    assert r.returncode == 0, "map sweep failed"
    return r.stdout[-1500:]


@APP.function(image=image, volumes={"/vol": VOL}, gpu="L4", cpu=8.0, memory=32768, timeout=24 * 3600)
def train(config: str, iterations: int = 0, val_every: int = 0, ckpt_every: int = 0, tag: str = "",
          stop_at: int = 0, num_workers: int = 6) -> dict:
    import yaml
    os.chdir("/root/sad")
    Path("/vol/results").mkdir(parents=True, exist_ok=True)
    for link, target in [("data/GOPRO_Large", "/vol/sad-data/gopro"), ("data/synthetic/val", "/vol/bundle/data/synthetic/val"),
                         ("checkpoints/NAFNet-GoPro-width32.pth", "/vol/sad-data/ckpt/NAFNet-GoPro-width32.pth"),
                         ("results", "/vol/results")]:
        assert Path(target).exists(), f"{target} missing - run ::prepare first"
        Path(link).parent.mkdir(parents=True, exist_ok=True)
        if not Path(link).is_symlink():
            Path(link).symlink_to(target)
    n_val, val_sha = _val_sha(Path("data/synthetic/val"))
    assert n_val == 40 and val_sha == EXPECTED_VAL, (n_val, val_sha)
    bad = _bad_pngs(Path("/vol/sad-data"), deep=False)  # fast header check of every PNG before spending GPU time
    assert not bad, f"{len(bad)} broken PNG files {bad[:5]} - run scripts/modal_train.py::verify_data first"
    man = subprocess.run(["python", "scripts/build_manifests.py", "--gopro", "data/GOPRO_Large"],
                         capture_output=True, text=True, check=True).stdout
    assert "train 1903" in man and "test 1111" in man, man
    cfg = yaml.safe_load(Path(config).read_text())
    run = f"{cfg['experiment_id']}_seed{cfg['seed']}" + (f"_{tag}" if tag else "")
    out = Path("results") / run
    cmd = ["python", "-u", "scripts/train.py", "--config", config, "--num-workers", str(num_workers)]
    for flag, val in (("--iterations", iterations), ("--val-every", val_every), ("--ckpt-every", ckpt_every)):
        if val:
            cmd += [flag, str(val)]
    if tag:
        cmd += ["--tag", tag]
    resumed = (out / "last.pth").exists()
    if resumed:
        cmd.append("--resume")
    env = {**os.environ, "RESUME_FROM": str(out / "last.pth") if resumed else ""}
    print(json.dumps({"run": run, "commit": os.environ.get("GIT_COMMIT"), "cmd": cmd, "resumed": resumed}), flush=True)
    log = (Path("/vol/results") / f"{run}.log").open("a")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    stop = threading.Event()

    def keeper():  # commit the Volume every 5 min; stop-at: SIGTERM once val.jsonl reaches the step
        sent = False
        while not stop.wait(30):
            if stop_at and not sent and (out / "val.jsonl").exists():
                steps = [json.loads(x)["step"] for x in (out / "val.jsonl").read_text().splitlines() if x.strip()]
                if steps and max(steps) >= stop_at:
                    proc.send_signal(signal.SIGTERM)
                    sent = True
            if int(time.time()) % 300 < 30:
                VOL.commit()

    th = threading.Thread(target=keeper, daemon=True)
    th.start()
    for line in proc.stdout:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)
    rc = proc.wait()
    stop.set()
    log.close()
    VOL.commit()
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines() if x.strip()] if (out / "events.jsonl").exists() else []
    val = [json.loads(x) for x in (out / "val.jsonl").read_text().splitlines() if x.strip()] if (out / "val.jsonl").exists() else []
    summary = {"run": run, "returncode": rc, "last_event": events[-1] if events else None, "val": val[-3:],
               "hardware": (yaml.safe_load((out / "config.yaml").read_text()) or {}).get("hardware") if (out / "config.yaml").exists() else None}
    print("SUMMARY " + json.dumps(summary), flush=True)
    return summary
