"""YAML config loading with ``_base_`` inheritance.

Shared infrastructure. A config may list parent files under ``_base_``
(paths relative to itself); children override parents key by key.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REQUIRED_TOP_LEVEL = ("experiment_id", "seed")


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bases = cfg.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = _merge(merged, load_config(path.parent / b))
    return _merge(merged, cfg)


def validate(cfg: dict) -> None:
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise KeyError(f"config missing required keys: {missing}")
