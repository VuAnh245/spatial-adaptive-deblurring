"""Generic training loop.

Shared infrastructure. AdamW + linear warmup + cosine decay, AMP, gradient
clipping, periodic checkpoint with resume (for interrupted Colab / Kaggle
sessions). The loss is passed in by the study, so study-specific terms
(e.g. L_map) never live here. The final checkpoint is used; validation is
for monitoring only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader


def cosine_with_warmup(optimizer, warmup: int, total: int, min_lr: float, base_lr: float):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total - warmup)
        return (min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))) / base_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Trainer:
    def __init__(self, model: torch.nn.Module, loss_fn: Callable[[torch.nn.Module, dict], tuple],
                 train_loader: DataLoader, out_dir: str | Path, *, lr: float, iterations: int,
                 warmup: int, min_lr: float = 1e-7, betas=(0.9, 0.9), weight_decay: float = 0.0,
                 grad_clip: float = 1.0, amp: bool = True, ckpt_every: int = 5000,
                 log_every: int = 100, val_fn: Callable[[torch.nn.Module], dict] | None = None,
                 val_every: int = 5000, device: str = "cuda"):
        self.model = model.to(device)
        self.loss_fn, self.loader, self.val_fn = loss_fn, train_loader, val_fn
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.sched = cosine_with_warmup(self.opt, warmup, iterations, min_lr, lr)
        self.iterations, self.grad_clip = iterations, grad_clip
        self.ckpt_every, self.log_every, self.val_every = ckpt_every, log_every, val_every
        self.device = device
        self.use_amp = amp and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.step = 0

    # ---------------------------------------------------------------- state
    def save(self, name: str = "last.pth") -> None:
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
                    "step": self.step}, self.out / name)

    def resume(self, path: str | Path | None = None) -> bool:
        path = Path(path) if path else self.out / "last.pth"
        if not path.exists():
            return False
        st = torch.load(path, map_location=self.device)
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"])
        self.scaler.load_state_dict(st["scaler"])
        self.step = st["step"]
        return True

    # ----------------------------------------------------------------- loop
    def _batches(self):
        while True:
            for batch in self.loader:
                yield {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                       for k, v in batch.items()}

    def fit(self) -> None:
        self.model.train()
        log = (self.out / "curves.csv").open("a", encoding="utf-8")
        it = self._batches()
        while self.step < self.iterations:
            batch = next(it)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                loss, parts = self.loss_fn(self.model, batch)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            self.step += 1
            if self.step % self.log_every == 0:
                row = {"step": self.step, "lr": self.sched.get_last_lr()[0], **{k: float(v) for k, v in parts.items()}}
                log.write(json.dumps(row) + "\n")
                log.flush()
            if self.val_fn and self.step % self.val_every == 0:
                self.model.eval()
                with torch.no_grad():
                    metrics = self.val_fn(self.model)
                (self.out / "val.jsonl").open("a").write(json.dumps({"step": self.step, **metrics}) + "\n")
                self.model.train()
            if self.step % self.ckpt_every == 0 or self.step == self.iterations:
                self.save()
        log.close()
        torch.save(self.model.state_dict(), self.out / "final.pth")
