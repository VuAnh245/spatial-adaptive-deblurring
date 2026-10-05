"""Loaders for public checkpoints used for evaluation only (never trained here).

Shared infrastructure. Architecture code for Restormer and MPRNet is NOT
copied into this repo; clone the official repos into ``third_party/``:

    git clone https://github.com/swz30/Restormer third_party/Restormer
    git clone https://github.com/swz30/MPRNet    third_party/MPRNet

Weights (see docs/datasets.md):
    NAFNet-GoPro-width32 -> checkpoints/NAFNet-GoPro-width32.pth
    Restormer            -> checkpoints/motion_deblurring.pth
    MPRNet (Deblurring)  -> checkpoints/model_deblurring.pth

Verify each loader against the published GoPro numbers before use.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nafnet import load_official_checkpoint, nafnet_gopro_width32
from .tlc import convert_tlc

ROOT = Path(__file__).resolve().parents[3]


class PadTo(nn.Module):
    """Pad input to a multiple of ``k`` (reflect), crop the output, keep the first
    output when the model returns a list (MPRNet returns one image per stage)."""

    def __init__(self, model: nn.Module, k: int = 8):
        super().__init__()
        self.model, self.k = model, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.k, (-w) % self.k
        y = self.model(F.pad(x, (0, pw, 0, ph), mode="reflect"))
        y = y[0] if isinstance(y, (list, tuple)) else y
        return y[..., :h, :w]


def _import(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - clone the official repo into third_party/")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip_module(state: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}


def load_nafnet_public(ckpt: str | Path = ROOT / "checkpoints/NAFNet-GoPro-width32.pth",
                       tlc: bool = False) -> nn.Module:
    """``tlc=True`` reproduces the official GoPro test setting (NAFNetLocal)."""
    model = nafnet_gopro_width32()
    load_official_checkpoint(model, str(ckpt), strict=True)
    model.eval()
    return convert_tlc(model) if tlc else model


def load_restormer_public(ckpt: str | Path = ROOT / "checkpoints/motion_deblurring.pth") -> nn.Module:
    arch = _import(ROOT / "third_party/Restormer/basicsr/models/archs/restormer_arch.py", "restormer_arch")
    model = arch.Restormer()  # defaults correspond to the motion-deblurring configuration
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("params", state))
    return PadTo(model, 8).eval()


def load_mprnet_public(ckpt: str | Path = ROOT / "checkpoints/model_deblurring.pth") -> nn.Module:
    arch = _import(ROOT / "third_party/MPRNet/Deblurring/MPRNet.py", "mprnet_arch")
    model = arch.MPRNet()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(_strip_module(state.get("state_dict", state)))
    return PadTo(model, 8).eval()
