"""Model variants A0, A0+, A1, A2, A3 (study contribution).

| ID  | Spatial branch | Supervised (L_map) | Channel branch |
| A0  | -              | -                  | -              |
| A0+ | - (fewest extra identity NAFBlocks matching A2's extra params within 5%) | - | - |
| A1  | yes            | no (lambda_map = 0)| -              |
| A2  | yes            | yes                | -              |
| A3  | yes            | yes                | yes            |

A1 and A2 share the architecture; they differ only by the loss weight,
so A2 - A1 measures the value of supervision. Supervision is configured in
the loss, not here.

``forward`` returns (I_hat, S_hat). ``smap_override`` replaces S_hat at
inference (oracle test SA-20, shuffle test SA-21).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..models.nafnet import NAFBlock, NAFNet
from .modules import BlurMapEstimator, ChannelModulation, SpatialSFTHead


class ModulatedNAFNet(NAFNet):
    def __init__(self, spatial: bool = True, channel: bool = False, detach_map: bool = False,
                 **nafnet_kwargs):
        super().__init__(**nafnet_kwargs)
        self.spatial, self.channel = spatial, channel
        self.detach_map = detach_map  # SA-17: restoration gradient does not reach the estimator
        stages = {f"enc{i}": c for i, c in enumerate(self.enc_channels)}
        stages["mid0"] = self.middle_channels
        stages.update({f"dec{i}": c for i, c in enumerate(self.dec_channels)})
        self.estimator = BlurMapEstimator() if spatial else None
        self.sft = nn.ModuleDict({k: SpatialSFTHead(c) for k, c in stages.items()}) if spatial else None
        self.chan = nn.ModuleDict({k: ChannelModulation(c) for k, c in stages.items()}) if channel else None
        self._smap: torch.Tensor | None = None
        self.mod_stats: dict | None = None  # filled by run_stage when a dict is set (training diagnostics)

    def run_stage(self, blocks, x, stage, idx):
        key = f"{stage}{idx}"
        sft = self.sft[key](self._smap, x.shape[-2:]) if self.spatial else None
        if self.mod_stats is not None and sft is not None:
            self.mod_stats[key] = (sft[0].detach().abs().mean().item(), sft[1].detach().abs().mean().item())
        for blk in blocks:
            def modulate(xn, inp, sft=sft, key=key):
                gamma, bias = 1.0, 0.0
                if sft is not None:
                    gamma, bias = gamma * (1 + sft[0]), bias + sft[1]
                if self.channel:
                    dgc, bc = self.chan[key](inp)
                    gamma, bias = gamma * (1 + dgc), bias + bc
                return gamma * xn + bias
            x = blk(x, modulate if (self.spatial or self.channel) else None)
        return x

    def forward(self, inp: torch.Tensor, smap_override: torch.Tensor | None = None):
        h, w = inp.shape[-2:]
        x = self.pad(inp)
        s_hat = None
        if self.spatial:
            s_hat = self.estimator(x)
            if smap_override is not None:
                self._smap = smap_override
            else:
                self._smap = s_hat.detach() if self.detach_map else s_hat
        out = self.forward_features(x) + x
        self._smap = None
        return out[..., :h, :w], s_hat


def build_variant(variant: str, nafnet_kwargs: dict | None = None) -> nn.Module:
    kw = dict(width=32, middle_blk_num=1, enc_blk_nums=(1, 1, 1, 28), dec_blk_nums=(1, 1, 1, 1))
    kw.update(nafnet_kwargs or {})
    if variant == "A0":
        return ModulatedNAFNet(spatial=False, channel=False, **kw)
    if variant in ("A1", "A2"):
        return ModulatedNAFNet(spatial=True, channel=False, **kw)
    if variant == "A3":
        return ModulatedNAFNet(spatial=True, channel=True, **kw)
    if variant == "A0+":
        return build_param_matched(build_variant("A2", kw), kw)
    raise ValueError(f"unknown variant {variant}")


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build_param_matched(reference: nn.Module, kw: dict, tolerance: float = 0.05) -> nn.Module:
    """Baseline-L (A0+): A0 + the FEWEST identity-initialised NAFBlocks in one decoder stage whose
    parameters match the extra parameters of ``reference`` (A2 - A0) within ``tolerance`` (QĐ-06).

    Every NAFBlock costs about the same MACs at any stage of this U-Net (7c^2 x pixels is constant),
    so fewest blocks = least added compute. The earlier rule (smallest absolute gap) picked 28 blocks
    at full resolution: far more compute than A2 and out of memory at batch 8.
    NAFBlock residual scales start at zero, so the extra blocks are identity maps and the public
    checkpoint still loads (strict=False for the new keys).
    """
    base = ModulatedNAFNet(spatial=False, channel=False, **kw)
    gap = n_params(reference) - n_params(base)
    best = None
    for i, c in enumerate(base.dec_channels):
        per = n_params(NAFBlock(c))
        for k in range(1, math.ceil(gap * (1 + tolerance) / per) + 1):
            err = abs(k * per - gap) / gap
            if err <= tolerance:
                if best is None or (k, err) < (best[1], best[2]):
                    best = (i, k, err)
                break
    if best is None:
        raise RuntimeError(f"param match failed: no decoder stage within {tolerance:.0%} of the extra {gap} params")
    i, k, err = best
    dec = list(kw["dec_blk_nums"])
    dec[i] += k
    model = ModulatedNAFNet(spatial=False, channel=False, **{**kw, "dec_blk_nums": tuple(dec)})
    model.param_match_info = {"stage": f"dec{i}", "extra_blocks": k, "extra_params": k * n_params(NAFBlock(base.dec_channels[i])),
                              "target_extra": gap, "rel_diff_extra": err,
                              "rel_diff_total": abs(n_params(model) - n_params(reference)) / n_params(reference)}
    return model
