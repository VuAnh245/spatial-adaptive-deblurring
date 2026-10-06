"""NAFNet backbone (port of the official architecture).

Shared infrastructure. Module and parameter names follow the official
implementation (megvii-research/NAFNet, MIT License) so that the public
NAFNet-GoPro-width32 checkpoint loads with ``load_official_checkpoint``.
Verify this at checklist step 4 (GoPro test PSNR close to the published
value) before training anything.

Reference: L. Chen, X. Chu, X. Zhang, J. Sun, "Simple Baselines for Image
Restoration", ECCV 2022. https://github.com/megvii-research/NAFNet
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel LayerNorm for NCHW tensors with affine weight / bias."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Statistics in FP32 even under AMP: (x - mu)^2 overflows FP16 (max 65504) once |x - mu| > ~256.
        # Identical to the plain version in FP32, so FP32 inference (SA-00) is unchanged.
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            y = (x - mu) / torch.sqrt(var + self.eps)
            y = self.weight.float().view(1, -1, 1, 1) * y + self.bias.float().view(1, -1, 1, 1)
        return y.to(dtype)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """NAFBlock. ``modulate`` is an optional callable applied right after
    ``norm1``; it receives (normalised feature, raw block input) and must
    return a tensor of the same shape. ``None`` gives the original block."""

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop: float = 0.0):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()
        ffn = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop) if drop > 0 else nn.Identity()
        # Zero-initialised residual scales: a fresh block is an identity map.
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor, modulate=None) -> torch.Tensor:
        x = self.norm1(inp)
        if modulate is not None:
            x = modulate(x, inp)
        x = self.conv2(self.conv1(x))
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.dropout1(self.conv3(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + self.dropout2(x) * self.gamma


class NAFNet(nn.Module):
    """U-shaped NAFNet. Stage widths: width * 2**i for encoders, width * 2**4 in the middle."""

    def __init__(self, img_channel: int = 3, width: int = 32, middle_blk_num: int = 1,
                 enc_blk_nums=(1, 1, 1, 28), dec_blk_nums=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        self.ups, self.downs = nn.ModuleList(), nn.ModuleList()
        self.enc_channels, self.dec_channels = [], []
        chan = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
            self.enc_channels.append(chan)
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        self.middle_channels = chan
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
            self.dec_channels.append(chan)
        self.padder_size = 2 ** len(self.encoders)

    # The three stage runners exist so subclasses can inject modulation.
    def run_stage(self, blocks: nn.Sequential, x: torch.Tensor, stage: str, idx: int) -> torch.Tensor:
        for blk in blocks:
            x = blk(x)
        return x

    def forward_features(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.intro(inp)
        skips = []
        for i, (enc, down) in enumerate(zip(self.encoders, self.downs)):
            x = self.run_stage(enc, x, "enc", i)
            skips.append(x)
            x = down(x)
        x = self.run_stage(self.middle_blks, x, "mid", 0)
        for i, (dec, up, skip) in enumerate(zip(self.decoders, self.ups, skips[::-1])):
            x = up(x) + skip
            x = self.run_stage(dec, x, "dec", i)
        return self.ending(x)

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pw, 0, ph))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        h, w = inp.shape[-2:]
        x = self.pad(inp)
        out = self.forward_features(x) + x
        return out[..., :h, :w]


def load_official_checkpoint(model: nn.Module, path: str, strict: bool = True):
    """Load an official NAFNet checkpoint (weights stored under 'params').

    Use ``strict=False`` for models that add new modules on top of NAFNet;
    the returned report lists keys that were not found in the checkpoint.
    """
    state = torch.load(path, map_location="cpu")
    state = state.get("params", state.get("state_dict", state))
    return model.load_state_dict(state, strict=strict)


def nafnet_gopro_width32() -> NAFNet:
    """Configuration of the official NAFNet-width32 GoPro model."""
    return NAFNet(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=(1, 1, 1, 28),
                  dec_blk_nums=(1, 1, 1, 1))
