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
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
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
                 val_every: int = 5000, device: str = "cuda",
                 diag_fn: Callable[[torch.nn.Module, str], dict | None] | None = None, diag_every: int = 500,
                 val_at_start: bool = False):
        self.model = model.to(device)
        self.loss_fn, self.loader, self.val_fn = loss_fn, train_loader, val_fn
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.sched = cosine_with_warmup(self.opt, warmup, iterations, min_lr, lr)
        self.iterations, self.grad_clip = iterations, grad_clip
        self.ckpt_every, self.log_every, self.val_every = ckpt_every, log_every, val_every
        # diag_fn(model, "begin") before the forward of every diag_every-th step, diag_fn(model, "end") -> dict
        # after the optimizer step; rows go to diag.jsonl. The study defines what is measured.
        self.diag_fn, self.diag_every = diag_fn, diag_every
        self.device = device
        self.use_amp = amp and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.step = 0
        self.val_at_start = val_at_start  # one val pass before the first step of a fresh run (step 0)
        self.stop = False  # set by a signal handler: save last.pth and leave the loop at the next step

    # ---------------------------------------------------------------- state
    def log_event(self, event: str, **fields) -> None:
        """Append one row to events.jsonl (start, checkpoint, stop, finish; survives resumes)."""
        with (self.out / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, "step": self.step, "time": round(time.time(), 1), **fields}) + "\n")

    def save(self, name: str = "last.pth") -> None:
        """Full training state: model, optimizer, scheduler, GradScaler, step and every RNG state."""
        rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
               "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
                    "step": self.step, "rng": rng}, self.out / name)

    def resume(self, path: str | Path | None = None) -> bool:
        path = Path(path) if path else self.out / "last.pth"
        if not path.exists():
            return False
        # own file, contains numpy / python RNG objects -> weights_only=False (default True since torch 2.6)
        st = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"])
        self.scaler.load_state_dict(st["scaler"])
        self.step = st["step"]
        rng = st.get("rng")
        if rng:
            torch.set_rng_state(rng["torch"].cpu())
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
            if rng["cuda"] and torch.cuda.is_available() and len(rng["cuda"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        return True

    # ----------------------------------------------------------------- loop
    def _batches(self):
        while True:
            for batch in self.loader:
                yield {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                       for k, v in batch.items()}

    def _validate(self) -> None:
        self.model.eval()
        tv = time.time()
        with torch.no_grad():
            metrics = self.val_fn(self.model)
        metrics["val_sec"] = round(time.time() - tv, 2)
        with (self.out / "val.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": self.step, **metrics}) + "\n")
        self.model.train()

    def fit(self) -> bool:
        """Train to ``iterations``. Returns False if stopped early (``self.stop``), True when finished."""
        if self.val_fn and self.val_at_start and self.step == 0:
            self._validate()
        self.model.train()
        log = (self.out / "curves.csv").open("a", encoding="utf-8")
        it = self._batches()
        t0 = t_last = time.time()
        data_wait = 0.0  # time blocked on the data loader since the last log row
        while self.step < self.iterations:
            if self.stop:
                ts = time.time()
                self.save()
                self.log_event("stopped", ckpt_sec=round(time.time() - ts, 2))
                log.close()
                return False
            tw = time.time()
            batch = next(it)
            data_wait += time.time() - tw
            diag_step = self.diag_fn is not None and (self.step + 1) % self.diag_every == 0
            if diag_step:
                self.diag_fn(self.model, "begin")
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
            if diag_step:
                row = self.diag_fn(self.model, "end")
                if row:
                    with (self.out / "diag.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": self.step, **{k: float(v) for k, v in parts.items()}, **row}) + "\n")
            if self.step % self.log_every == 0:
                now = time.time()
                row = {"step": self.step, "lr": self.sched.get_last_lr()[0], **{k: float(v) for k, v in parts.items()},
                       "sec": round(now - t0, 2), "data_frac": round(data_wait / max(now - t_last, 1e-9), 4)}
                t_last, data_wait = now, 0.0
                log.write(json.dumps(row) + "\n")
                log.flush()
            if self.val_fn and self.step % self.val_every == 0:
                self._validate()
            if self.step % self.ckpt_every == 0 or self.step == self.iterations:
                ts = time.time()
                self.save()
                self.log_event("ckpt", ckpt_sec=round(time.time() - ts, 2))
        log.close()
        torch.save(self.model.state_dict(), self.out / "final.pth")
        self.log_event("finished")
        return True
