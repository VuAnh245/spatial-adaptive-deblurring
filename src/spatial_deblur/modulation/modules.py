"""Supervised spatial feature modulation (study contribution).

- ``BlurMapEstimator``: B -> S_hat at 1/4 resolution, sigmoid output.
- ``BlurMapEstimatorV2`` (QĐ-10, SA-33 only): same width and output; normalised input, per-pixel LayerNorm2d after
  every conv, Kaiming init, stride-1 first conv. The original estimator collapses to a constant map even when
  trained alone (sad-checks V4, analysis/estimator_learnability.py).
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


class BlurMapEstimatorV2(nn.Module):
    """QĐ-10 (SA-33): BlurMapEstimator with normalised input, LayerNorm2d after each conv, Kaiming init and a
    stride-1 first conv. LayerNorm2d normalises over channels at each pixel (as in NAFNet), so there are no
    image-wide statistics: the output does not depend on the image size (no train-patch / full-frame gap that TLC
    would have to fix). Output: sigmoid map at 1/4 resolution, like BlurMapEstimator."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        from ..models.nafnet import LayerNorm2d

        def blk(i, o, s):
            return [nn.Conv2d(i, o, 3, s, 1), LayerNorm2d(o), nn.GELU()]

        self.net = nn.Sequential(*blk(3, 32, 1), *blk(32, hidden, 2), *blk(hidden, hidden, 2),
                                 *blk(hidden, hidden, 1), *blk(hidden, hidden, 1), nn.Conv2d(hidden, 1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, blur: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net((blur - 0.5) / 0.25))


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
        # A module (not tensor.mean) so models.tlc.convert_tlc localises it like NAFNet's SCA pooling.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, max(channels // reduction, 8), 1)
        self.fc2 = _zero(nn.Conv2d(max(channels // reduction, 8), 2 * channels, 1))

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 1x1 convs = per-position MLP, so the branch also works on a local (TLC) pooled map.
        dg, b = self.fc2(F.gelu(self.fc1(self.pool(feat)))).chunk(2, dim=1)
        return dg, b
