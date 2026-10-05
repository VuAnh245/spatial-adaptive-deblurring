"""Efficiency benchmark following the spec protocol.

Shared infrastructure. Same GPU, CUDA, PyTorch and precision for every
model; batch 1; 50 warm-up passes, 200 timed passes with
``torch.cuda.synchronize()``; disk I/O and decoding excluded.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def count_params(model: torch.nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_M": total / 1e6, "trainable_params_M": trainable / 1e6}


@torch.no_grad()
def latency(model: torch.nn.Module, height: int = 720, width: int = 1280, warmup: int = 50,
            runs: int = 200, device: str = "cuda", amp: bool = True) -> dict[str, float]:
    model = model.to(device).eval()
    x = torch.rand(1, 3, height, width, device=device)
    use_cuda = device.startswith("cuda")
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (amp and use_cuda) else torch.no_grad()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    with ctx:
        for i in range(warmup + runs):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if use_cuda:
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    med = float(np.median(times))
    out = {"latency_median_ms": med, "latency_p95_ms": float(np.percentile(times, 95)),
           "fps": 1000.0 / med}
    if use_cuda:
        out["peak_vram_GB"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    return out


def macs(model: torch.nn.Module, size: int = 256) -> float | None:
    """MACs in G at size x size using fvcore if installed (report the tool used)."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    x = torch.rand(1, 3, size, size)
    return FlopCountAnalysis(model.eval(), x).total() / 1e9
