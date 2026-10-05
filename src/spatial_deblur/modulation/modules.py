"""Supervised spatial feature modulation (study contribution).

- ``BlurMapEstimator``: B -> S_hat at 1/4 resolution, sigmoid output.
- ``SpatialSFTHead``: S_l -> (delta_gamma_s, beta_s), per pixel and channel.
- ``ChannelModulation``: GAP(F_l) -> (delta_gamma_c, beta_c), per channel (A3 only).

Modulation is applied right after ``norm1`` of every NAFBlock:

    Gamma = (1 + dgamma_s) * (1 + dgamma_c),  Bias = beta_s + beta_c
    F_tilde = Gamma * LN(F) + Bias

Last layers are zero-initialised, so at step 0 every variant equals the
NAFNet checkpoint it was loaded from (tested in tests/test_modulation.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero(m: nn.Module) -> nn.Module:
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class BlurMapEstimator(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, hidden, 3, 2, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, blur: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(blur))


class SpatialSFTHead(nn.Module):
    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden, 3, 1, 1)
        self.conv2 = _zero(nn.Conv2d(hidden, 2 * channels, 1))

    def forward(self, smap: torch.Tensor, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        s = F.interpolate(smap, size=size, mode="bilinear", align_corners=False)
        dg, b = self.conv2(F.gelu(self.conv1(s))).chunk(2, dim=1)
        return dg, b


class ChannelModulation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 8))
        self.fc2 = _zero(nn.Linear(max(channels // reduction, 8), 2 * channels))

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = feat.mean(dim=(2, 3))
        dg, b = self.fc2(F.gelu(self.fc1(z))).chunk(2, dim=1)
        return dg[..., None, None], b[..., None, None]
