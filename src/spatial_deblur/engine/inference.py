"""Full-resolution inference with optional overlapped tiling.

Shared infrastructure. Metrics are always computed on full images; tiling
(512 tiles, 32-64 px overlap, linear blending) is only a memory fallback and
must be reported when used.
"""
from __future__ import annotations

import numpy as np
import torch


def _restore(model, x):
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def restore_full(model: torch.nn.Module, img: np.ndarray, device: str = "cuda",
                 tile: int | None = None, overlap: int = 48) -> np.ndarray:
    """img: H x W x 3 float32 [0, 1] -> restored image of the same shape."""
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].to(device)
    model = model.to(device).eval()
    if tile is None or (x.shape[-1] <= tile and x.shape[-2] <= tile):
        y = _restore(model, x)
    else:
        _, _, h, w = x.shape
        y = torch.zeros_like(x)
        weight = torch.zeros_like(x[:, :1])
        ramp = torch.ones(tile, device=device)
        ramp[:overlap] = torch.linspace(0, 1, overlap + 2, device=device)[1:-1]
        ramp[-overlap:] = torch.linspace(1, 0, overlap + 2, device=device)[1:-1]
        win = (ramp[:, None] * ramp[None, :])[None, None]
        stride = tile - overlap
        ys = list(range(0, max(h - tile, 0) + 1, stride)) + ([h - tile] if h > tile else [])
        xs = list(range(0, max(w - tile, 0) + 1, stride)) + ([w - tile] if w > tile else [])
        for y0 in sorted(set(max(0, v) for v in ys)):
            for x0 in sorted(set(max(0, v) for v in xs)):
                patch = x[..., y0:y0 + tile, x0:x0 + tile]
                wv = win[..., :patch.shape[-2], :patch.shape[-1]]
                y[..., y0:y0 + tile, x0:x0 + tile] += _restore(model, patch) * wv
                weight[..., y0:y0 + tile, x0:x0 + tile] += wv
        y = y / weight.clamp_min(1e-8)
    return y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
