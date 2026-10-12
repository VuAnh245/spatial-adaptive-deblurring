"""Read-only inspection of training artifacts; writes only dated audit outputs.

No training, full evaluation, manifest regeneration, or changes to model code.
Kaggle originals belong in results/artifacts/kaggle/<scriptVersionId>/original/.
Supplied stdout is supplementary evidence, never proof of server version identity.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUNS = {
    "SA-10": ("Baseline", "A0", "SA-10_a0.yaml", "355813018", 2),
    "SA-11": ("Baseline-L", "A0+", "SA-11_a0plus.yaml", "355940350", 3),
    "SA-12": ("SFT-Free", "A1", "SA-12_a1.yaml", "355940350", 3),
    "SA-13": ("SFT-Map", "A2", "SA-13_a2.yaml", "355813018", 2),
    "SA-14": ("SFT-Map+Ch", "A3", "SA-14_a3.yaml", "", None),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8")


def relative(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def supplied_log(path):
    out = {"source": relative(path), "sha256": sha(path), "runs": {}, "version_identity_verified": False}
    active = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "locked commit OK:" in line:
            out["locked_commit_line"] = {"line": line_no, "text": line}
        if "val: 40 images, content sha256" in line:
            out["val_content_hash_line"] = {"line": line_no, "text": line}
        match = re.search(r"(SA-\d+_seed\d+)(?: rc=|: exit)", line)
        if match:
            active = match[1]
            out["runs"].setdefault(active, {"validation_observations": {}, "events": [], "train_samples": []})
        if active is None:
            continue
        row = out["runs"][active]
        if ": exit" in line:
            row["exit_line"] = {"line": line_no, "text": line}
        match = re.search(r"(\{.*\})\s*$", line)
        if not match:
            continue
        try:
            data = json.loads(match[1])
        except json.JSONDecodeError:
            continue
        if "last val:" in line:
            row["validation_observations"][str(data["step"])] = {"line": line_no, **data}
        elif "event" in data:
            item = {"line": line_no, **data}
            if not any(e.get("event") == data["event"] and e.get("step") == data.get("step")
                       and e.get("time") == data.get("time") for e in row["events"]):
                row["events"].append(item)
        elif "rc=" in line:
            row["train_samples"].append({"line": line_no, **data})
    return out


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inspect_checkpoint(final, cfg):
    import torch
    from spatial_deblur.modulation.variants import build_variant

    torch.set_num_threads(2)
    state = torch.load(final, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("final.pth is not the weights-only state_dict written by Trainer.fit")
    bad = [k for k, v in state.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
    if bad:
        raise ValueError(f"nonfinite tensors: {bad[:10]}")
    m = cfg["model"]
    kw = {k: m[k] for k in ("width", "middle_blk_num", "enc_blk_nums", "dec_blk_nums")}
    model = build_variant(m["variant"], kw).eval()
    model.detach_map = m.get("detach_map", False)
    report = model.load_state_dict(state, strict=True)
    with torch.no_grad():
        pred, smap = model(torch.rand(1, 3, 32, 48))
    if pred.shape != (1, 3, 32, 48) or not bool(torch.isfinite(pred).all()):
        raise ValueError("nonfinite or incorrectly shaped smoke inference output")
    if smap is not None and not bool(torch.isfinite(smap).all()):
        raise ValueError("nonfinite map smoke inference output")
    out = {"finite_tensors": True, "missing_keys": report.missing_keys,
           "unexpected_keys": report.unexpected_keys, "smoke_shape": list(pred.shape),
           "map_shape": list(smap.shape) if smap is not None else None,
           "checkpoint_kind": "weights_only", "embedded_step": None}
    last = final.parent / "last.pth"
    if last.exists():
        # Own training outputs, only after checking config and source. NumPy RNG
        # objects in Trainer.save need weights_only=False; do not use on arbitrary downloads.
        resume = torch.load(last, map_location="cpu", weights_only=False)
        required = {"model", "opt", "sched", "scaler", "step", "rng"}
        out["resume_keys_present"] = sorted(required & resume.keys())
        out["resume_step"] = resume.get("step")
        out["resume_complete"] = required <= resume.keys()
        out["final_equals_last_model"] = set(state) == set(resume.get("model", {})) and all(
            torch.equal(v, resume["model"][k]) for k, v in state.items())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="User-local audit date, YYYY-MM-DD")
    ap.add_argument("--v2-log", type=Path)
    ap.add_argument("--v3-log", type=Path)
    args = ap.parse_args()
    evidence = ROOT / "results" / "audit" / args.date
    reports = ROOT / "docs" / "reports"
    evidence.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    snapshot = {"captured_at_utc": now, "local_audit_date": args.date,
                "head": git("rev-parse", "HEAD").strip(), "branch": git("branch", "--show-current").strip(),
                "status": git("status", "--short"), "disk_free_bytes": shutil.disk_usage(ROOT).free,
                "locked_commit": git("rev-parse", "faf4a06").strip()}
    for name, command in {"locked-to-head.diff": ("diff", "faf4a06", "HEAD"),
                          "locked-to-worktree.diff": ("diff", "faf4a06"),
                          "user-worktree.diff": ("diff",),
                          "git-log.txt": ("log", "-n", "5", "--oneline")}.items():
        (evidence / name).write_text(git(*command), encoding="utf-8")
    snapshot["protected_user_files"] = {p: sha(ROOT / p) for p in (
        "docs/experiments.md", "scripts/evaluate.py", "analysis/eval_tables.py", "scripts/eval_queue.py",
        "scripts/make_testsets.py", "scripts/pack_results.py", "scripts/verify_manifest.ps1") if (ROOT / p).exists()}
    write_json(evidence / "repository-snapshot.json", snapshot)

    tooling = {"captured_at_utc": now, "python": sys.version, "python_executable": sys.executable,
               "packages": {}, "kaggle_credential_env_names": [k for k in os.environ if k.startswith("KAGGLE")],
               "credential_file_presence": {str(p): p.exists() for p in (
                   Path.home() / ".kaggle/kaggle.json", Path.home() / ".kaggle/access_token")}}
    for package in ("kaggle", "torch", "numpy", "scipy", "scikit-image", "pyyaml"):
        try:
            tooling["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            tooling["packages"][package] = "Missing"
    spec = importlib.util.find_spec("kaggle")
    if spec and spec.origin:
        api_path = Path(spec.origin).parent / "api/kaggle_api_extended.py"
        source = api_path.read_text(encoding="utf-8")
        excerpts = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name in ("kernels_output", "kernels_list_files", "kernels_pull"):
                excerpts.append(f"# {api_path}:{node.lineno}\n" + ast.get_source_segment(source, node))
        (evidence / "installed-kaggle-methods.txt").write_text("\n\n".join(excerpts), encoding="utf-8")
        tooling["kaggle_api_source"] = str(api_path)
        tooling["kaggle_api_source_sha256"] = sha(api_path)
        tooling["output_and_files_version_binding"] = "Version parsed but not assigned to API request in installed 2.2.4"
    write_json(evidence / "tooling-snapshot.json", tooling)

    logs = {}
    artifacts = []
    for version, src in (("355813018", args.v2_log), ("355940350", args.v3_log)):
        if not src:
            continue
        dst = ROOT / "results/artifacts/kaggle" / version / "supplied" / "stdout-pasted.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha(dst) != sha(src):
            raise FileExistsError(f"Refusing to overwrite original evidence: {dst}")
        if not dst.exists():
            shutil.copyfile(src, dst)
        logs[version] = supplied_log(dst)
        artifacts.append({"path": relative(dst), "bytes": dst.stat().st_size, "sha256": sha(dst),
                          "source_url": f"https://www.kaggle.com/code/xunanhv/sad-train?scriptVersionId={version}",
                          "expected_script_version_id": version, "version_identity_verified": False,
                          "source_kind": "user_supplied_pasted_stdout", "source_local_path": str(src),
                          "imported_at_utc": datetime.fromtimestamp(dst.stat().st_mtime, timezone.utc).isoformat(),
                          "downloaded_at_utc": ""})
    write_json(evidence / "supplied-log-analysis.json", clean(logs))

    from spatial_deblur.utils.config import load_config

    registry, checks = [], {}
    for exp, (label, variant, config_name, version, ordinal) in RUNS.items():
        config_path = ROOT / "configs/experiments" / config_name
        cfg = load_config(config_path)
        run = f"{exp}_seed1"
        supplied = logs.get(version, {}).get("runs", {}).get(run, {})
        finished = [e for e in supplied.get("events", []) if e["event"] == "finished"]
        observed = supplied.get("validation_observations", {})
        original = ROOT / "results/artifacts/kaggle" / version / "original"
        hits = sorted(original.rglob(f"{run}/final.pth")) if version and original.exists() else []
        row = {"run": run, "model": label, "variant": variant, "seed": cfg["seed"],
               "seed_evidence": "repo_config_and_supplied_run_name" if supplied else "repo_config_only",
               "reported_finished_step": finished[-1]["step"] if finished else "",
               "verified_checkpoint_step": "", "expected_commit": "faf4a06", "verified_commit": "",
               "notebook": "xunanhv/sad-train" if version else "", "expected_version_number": ordinal or "",
               "expected_script_version_id": version, "version_identity_verified": False,
               "gpu": "T4 x2 (user screenshot; not artifact-verified)" if version else "",
               "config_path": relative(config_path), "final_path": "", "final_sha256": "",
               "last_path": "", "last_sha256": "", "checkpoint_kind": "",
               "status": "Missing", "training_log_status": "finished_in_supplied_stdout" if finished else "Missing",
               "evaluation_status": "Missing", "final_val_step": "", "final_val_psnr": "",
               "val_n_verified": "", "last_supplied_val_step": max(map(int, observed)) if observed else "",
               "evidence_path": logs.get(version, {}).get("source", ""),
               "finish_evidence_line": finished[-1]["line"] if finished else "",
               "reported_finish_time_utc": datetime.fromtimestamp(finished[-1]["time"], timezone.utc).isoformat() if finished else "",
               "evaluation_ready": False,
               "notes": "No original checkpoint/config/run logs; version-to-source identity not verified"}
        if len(hits) > 1:
            row.update(status="Verification failed", notes="Multiple checkpoint candidates; not choosing silently")
        elif hits:
            final = hits[0]
            row.update(status="Downloaded", final_path=relative(final), final_sha256=sha(final))
            cfg_path = final.parent / "config.yaml"
            try:
                if not cfg_path.exists():
                    raise ValueError("Original config.yaml missing; cannot distinguish A1 from A2 by weights")
                run_cfg = load_config(cfg_path)
                if (run_cfg.get("experiment_id"), run_cfg.get("seed"), run_cfg["model"].get("variant")) != (exp, 1, variant):
                    raise ValueError("Run identity/config mismatch")
                expected_map = 0.1 if variant in ("A2", "A3") else 0.0
                if run_cfg["loss"].get("lambda_map") != expected_map:
                    raise ValueError("Map loss/config identity mismatch")
                commit = run_cfg.get("git_commit", "")
                if not commit.startswith("faf4a06") or run_cfg["training"].get("iterations") != 30000:
                    raise ValueError("Commit or iteration budget mismatch")
                row["verified_commit"] = commit
                row["gpu"] = run_cfg.get("hardware", {}).get("gpu", "")
                checks[run] = inspect_checkpoint(final, run_cfg)
                row["checkpoint_kind"] = checks[run]["checkpoint_kind"]
                last = final.parent / "last.pth"
                if last.exists():
                    row.update(last_path=relative(last), last_sha256=sha(last),
                               verified_checkpoint_step=checks[run].get("resume_step", ""))
                val = final.parent / "val.jsonl"
                if val.exists():
                    vals = read_jsonl(val)
                    final_vals = [v for v in vals if v.get("step") == 30000]
                    if final_vals:
                        row.update(final_val_step=30000, final_val_psnr=final_vals[-1].get("val_psnr", ""))
                ev_path = final.parent / "events.jsonl"
                ev = read_jsonl(ev_path) if ev_path.exists() else []
                provenance_path = original.parent / "download-provenance.json"
                provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
                row["version_identity_verified"] = (provenance.get("version_identity_verified") is True
                    and str(provenance.get("script_version_id")) == version)
                required_logs = ("config.yaml", "curves.csv", "val.jsonl", "events.jsonl")
                logs_complete = all((final.parent / p).is_file() for p in required_logs)
                if (row["version_identity_verified"] and logs_complete and row["final_val_step"] == 30000
                        and any(e.get("event") == "finished" and e.get("step") == 30000 for e in ev)):
                    row.update(status="Verified", notes="Weights/load/smoke/config/events/version verified; dataset identity requires separate manifests check")
                else:
                    row["notes"] = "Checkpoint smoke/config passed; original version identity or finish event still missing"
            except Exception as exc:
                row.update(status="Verification failed", notes=f"{type(exc).__name__}: {exc}")
        registry.append(row)
    hashes = {}
    for row in registry:
        if row["final_sha256"]:
            hashes.setdefault(row["final_sha256"], []).append(row["run"])
    checks["duplicate_final_hashes"] = {h: runs for h, runs in hashes.items() if len(runs) > 1}
    for runs in checks["duplicate_final_hashes"].values():
        for row in registry:
            if row["run"] in runs:
                row.update(status="Verification failed", notes="Identical final checkpoint hash across different runs")
    write_json(evidence / "checkpoint-checks.json", clean(checks))
    write_csv(reports / f"run-registry-{args.date}.csv", registry)

    inventory = []
    for folder in ("results", "manifests", "configs", "docs/figures", "notebooks", "data", "checkpoints", "src", "scripts", "analysis"):
        root = ROOT / folder
        if root.exists():
            for p in sorted(root.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    inventory.append({"path": relative(p), "bytes": p.stat().st_size, "sha256": sha(p)})
    write_csv(reports / f"local-inventory-{args.date}.csv", inventory)
    write_csv(reports / f"artifact-index-{args.date}.csv", artifacts)
    write_json(evidence / "run-registry.json", registry)
    reproduce = ROOT / "results/SA-00_reproduce/gopro_test_nafnet32_public.csv"
    if reproduce.exists():
        with reproduce.open(encoding="utf-8", newline="") as f:
            reproduce_rows = list(csv.DictReader(f))
        means = {}
        for metric in ("psnr_blur", "psnr_float_tlc", "psnr_u8_global", "psnr_u8_tlc", "ssim_skimage_tlc"):
            values = [float(r[metric]) for r in reproduce_rows if r.get(metric)]
            means[metric] = {"n": len(values), "mean": sum(values) / len(values)} if values else {"n": 0}
        write_json(evidence / "sa00-recomputed-summary.json", {"source": relative(reproduce), "sha256": sha(reproduce),
            "rows": len(reproduce_rows), "unique_ids": len({r["id"] for r in reproduce_rows}), "metrics": means})
    print(json.dumps({"evidence": relative(evidence), "registry": relative(reports / f"run-registry-{args.date}.csv"),
                      "finished_in_supplied_logs": sum(bool(r["reported_finished_step"]) for r in registry),
                      "verified_runs": sum(r["status"] == "Verified" for r in registry),
                      "core_model_variants_in_current_document": len(RUNS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
