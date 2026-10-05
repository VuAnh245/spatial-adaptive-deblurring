"""Split manifests: ID lists, checksums and scene grouping.

Shared infrastructure. Every split lives in ``manifests/`` as a text file
(one relative path per line) next to a ``.sha256`` file, and is committed.
Splits are made by SCENE / sequence, never by individual frame.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path


def read_list(path: str | Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_list(path: str | Path, items: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(items) + "\n"
    path.write_text(text, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def scene_of(rel_path: str, depth: int = 1) -> str:
    """Scene id = the first ``depth`` directory components (GoPro: sequence folder,
    RealBlur: scene folder). Adjust ``depth`` to the dataset layout."""
    parts = Path(rel_path).parts
    return "/".join(parts[:depth])


def split_by_scene(items: list[str], fraction: float, seed: int, depth: int = 1
                   ) -> tuple[list[str], list[str]]:
    """Select ``fraction`` of SCENES (all their images) -> (selected, rest)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for it in items:
        groups[scene_of(it, depth)].append(it)
    scenes = sorted(groups)
    random.Random(seed).shuffle(scenes)
    k = max(1, round(fraction * len(scenes)))
    chosen = set(scenes[:k])
    sel = [i for s in sorted(chosen) for i in sorted(groups[s])]
    rest = [i for s in sorted(set(scenes) - chosen) for i in sorted(groups[s])]
    return sel, rest


def assert_disjoint_scenes(*splits: list[str], depth: int = 1) -> None:
    seen: dict[str, int] = {}
    for idx, split in enumerate(splits):
        for s in {scene_of(i, depth) for i in split}:
            if s in seen and seen[s] != idx:
                raise AssertionError(f"scene {s} appears in split {seen[s]} and {idx}")
            seen[s] = idx
