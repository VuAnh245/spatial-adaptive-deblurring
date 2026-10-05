"""Test-time Local Converter (TLC) for NAFNet-style global pooling.

Shared infrastructure. NAFNet's simplified channel attention pools globally
(``nn.AdaptiveAvgPool2d(1)``). Trained on 256x256 patches but tested on
1280x720 frames, global statistics then come from a much larger region than
during training. The official GoPro test config (``NAFNetLocal``) therefore
replaces every global pool with a local average pool whose window equals the
feature size of a ``1.5 x train_size`` input. Published NAFNet GoPro numbers
(32.87 dB for width32) are measured with this conversion.

Ported from megvii-research/NAFNet ``basicsr/models/archs/local_arch.py``
(MIT License), exact (non-"fast_imp") variant only.
Reference: X. Chu, L. Chen, C. Chen, X. Lu, "Improving Image Restoration by
Revisiting Global Information Aggregation", ECCV 2022. arXiv:2112.04491
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAvgPool2d(nn.Module):
    """Local average pooling with replicate padding back to the input size.

    The window is fixed on the first forward call (run on a ``train_size``
    dummy input by :func:`convert_tlc`): ``feature_size * base / train``.
    """

    def __init__(self, base_size: tuple[int, int], train_size: tuple[int, int]):
        super().__init__()
        self.base_size, self.train_size = base_size, train_size
        self.kernel_size: list[int] | None = None

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, base_size={self.base_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size is None:
            self.kernel_size = [x.shape[2] * self.base_size[0] // self.train_size[0],
                                x.shape[3] * self.base_size[1] // self.train_size[1]]
        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)
        n, c, h, w = x.shape
        s = F.pad(x.cumsum(dim=-1).cumsum(dim=-2), (1, 0, 1, 0))
        k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
        out = (s[:, :, k1:, k2:] + s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2]) / (k1 * k2)
        _h, _w = out.shape[2:]
        pad = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
        return F.pad(out, pad, mode="replicate")


def _replace(module: nn.Module, base_size, train_size) -> int:
    n = 0
    for name, m in module.named_children():
        if isinstance(m, nn.AdaptiveAvgPool2d):
            assert m.output_size in (1, (1, 1)), m.output_size
            setattr(module, name, LocalAvgPool2d(base_size, train_size))
            n += 1
        else:
            n += _replace(m, base_size, train_size)
    return n


@torch.no_grad()
def convert_tlc(model: nn.Module, train_size: int = 256, base_scale: float = 1.5) -> nn.Module:
    """Replace global pools in ``model`` (in place) and fix their windows.

    Only ``nn.AdaptiveAvgPool2d(1)`` modules are converted; pooling written as
    ``tensor.mean(dim=(2, 3))`` is left global. Returns the model in eval mode.
    """
    base = int(train_size * base_scale)
    n = _replace(model, (base, base), (train_size, train_size))
    if n == 0:
        raise ValueError("no nn.AdaptiveAvgPool2d found - nothing to convert")
    model.eval()
    dev = next(model.parameters()).device
    model(torch.rand(1, 3, train_size, train_size, device=dev))  # fixes every window size
    return model
