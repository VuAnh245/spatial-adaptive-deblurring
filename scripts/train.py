"""Train one SA-xx experiment.

    python scripts/train.py --config configs/experiments/SA-13_a2.yaml [--seed 1] [--resume]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from spatial_deblur.data.datasets import FixedSyntheticDataset, SyntheticNonUniformDataset
from spatial_deblur.degradations.kernels import SeverityScale
from spatial_deblur.engine.trainer import Trainer
from spatial_deblur.metrics.image_quality import psnr
from spatial_deblur.models.nafnet import load_official_checkpoint
from spatial_deblur.modulation.analysis import map_quality
from spatial_deblur.modulation.losses import make_loss
from spatial_deblur.modulation.variants import build_variant
from spatial_deblur.utils.config import load_config, validate
from spatial_deblur.utils.seed import set_seed


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    validate(cfg)
    set_seed(cfg["seed"])
    m, d, t = cfg["model"], cfg["data"], cfg["training"]
    if m.get("placement", "all") != "all" or cfg["loss"].get("extra_loss"):
        raise NotImplementedError("SA-15/16/18/19 are optional: implement placement / extra loss first")

    out = Path("results") / f"{cfg['experiment_id']}_seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    cfg["git_commit"] = git_commit()
    cfg["hardware"] = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                       "cuda": str(torch.version.cuda), "pytorch": str(torch.__version__)}

    kw = {k: m[k] for k in ("width", "middle_blk_num", "enc_blk_nums", "dec_blk_nums")}
    model = build_variant(m["variant"], kw)
    model.detach_map = m.get("detach_map", False)
    report = load_official_checkpoint(model, m["init_checkpoint"], strict=False)
    cfg["init_load"] = {"missing": len(report.missing_keys), "unexpected": len(report.unexpected_keys)}
    if report.unexpected_keys:
        raise RuntimeError(f"unexpected keys in checkpoint: {report.unexpected_keys[:5]}")
    if hasattr(model, "param_match_info"):
        cfg["param_match"] = model.param_match_info
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    scale = SeverityScale(**d["severity_scale"])
    ds = SyntheticNonUniformDataset(d["gopro_root"], d["sharp_train"], scale, d["s_train"], d["regions"],
                                    d["patch"], d["margin"], d["noise_sigma"],
                                    length=t["iterations"] * t["batch_size"], seed=cfg["seed"])
    loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"],
                        pin_memory=True, drop_last=True)
    val_set = FixedSyntheticDataset(Path(d["synthetic_root"]) / "val")

    def val_fn(model):
        ps, sp = [], []
        for i in range(0, len(val_set), max(1, len(val_set) // 50)):
            s = val_set[i]
            x = s["blur"][None].to(a.device)
            pred, s_hat = model(x)
            ps.append(psnr(pred[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy(), s["sharp"].permute(1, 2, 0).numpy()))
            if s_hat is not None:
                sp.append(map_quality(s_hat.cpu(), s["smap"][None])["spearman"])
        return {"val_psnr": float(sum(ps) / len(ps)), **({"val_spearman": float(sum(sp) / len(sp))} if sp else {})}

    trainer = Trainer(model, make_loss(cfg["loss"]["lambda_map"]), loader, out, lr=t["lr"],
                      iterations=t["iterations"], warmup=t["warmup"], min_lr=t["min_lr"],
                      betas=tuple(t["betas"]), weight_decay=t["weight_decay"], grad_clip=t["grad_clip"],
                      amp=t["amp"], ckpt_every=t["ckpt_every"], val_fn=val_fn if len(val_set) else None,
                      val_every=t["val_every"], device=a.device)
    if a.resume and trainer.resume():
        print(f"resumed at step {trainer.step}")
    trainer.fit()
    (out / "train_done.json").write_text(json.dumps({"steps": trainer.step}))


if __name__ == "__main__":
    main()
