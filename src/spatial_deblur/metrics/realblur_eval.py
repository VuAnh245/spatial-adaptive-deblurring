"""Wrapper around the OFFICIAL RealBlur evaluation (alignment + PSNR/SSIM).

Shared infrastructure. The official script is NOT copied into this repo.
Download it once from https://github.com/rimchang/RealBlur
(``evaluation_RealBlur_ecc.py``, CC BY 4.0 dataset; check the repo for the
code licence) into ``third_party/realblur/`` and keep its alignment step
unchanged. Do not re-implement the alignment.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

OFFICIAL = Path(__file__).resolve().parents[3] / "third_party" / "realblur" / "evaluation_RealBlur_ecc.py"


def load_official():
    if not OFFICIAL.exists():
        raise FileNotFoundError(
            f"Official RealBlur evaluation not found at {OFFICIAL}. "
            "Download evaluation_RealBlur_ecc.py from https://github.com/rimchang/RealBlur")
    spec = importlib.util.spec_from_file_location("realblur_official", OFFICIAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
