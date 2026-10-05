"""PyTorch datasets.

Shared infrastructure.
- ``PairedImageDataset``: blur/sharp pairs listed in a manifest (GoPro, RealBlur,
  HIDE, RSBlur). Each manifest line: ``blur_rel_path<TAB>sharp_rel_path``.
- ``SyntheticNonUniformDataset``: on-the-fly region-mixed blur from sharp
  frames; returns (B, I, S). Used for training; fixed test sets are generated
  once by ``scripts/generate_testsets.py`` and read back with ``FixedSyntheticDataset``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..degradations.blur import blur_nonuniform
from ..degradations.kernels import SeverityScale
from .manifests import read_list


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1) if a.ndim == 3 \
        else torch.from_numpy(np.ascontiguousarray(a))[None]


def augment(arrays: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    """Same flip / 90-degree rotation for every array (no colour jitter, no resize)."""
    if rng.random() < 0.5:
        arrays = [a[:, ::-1] for a in arrays]
    if rng.random() < 0.5:
        arrays = [a[::-1] for a in arrays]
    k = int(rng.integers(0, 4))
    return [np.rot90(a, k) for a in arrays]


class PairedImageDataset(Dataset):
    def __init__(self, root: str | Path, manifest: str | Path, patch: int | None = 256,
                 train: bool = True, seed: int = 0):
        self.root = Path(root)
        self.pairs = [ln.split("\t") for ln in read_list(manifest)]
        self.patch, self.train = patch, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        b_rel, s_rel = self.pairs[i]
        b, s = load_rgb(self.root / b_rel), load_rgb(self.root / s_rel)
        if self.train and self.patch:
            h, w = b.shape[:2]
            y = int(self.rng.integers(0, h - self.patch + 1))
            x = int(self.rng.integers(0, w - self.patch + 1))
            b, s = b[y:y + self.patch, x:x + self.patch], s[y:y + self.patch, x:x + self.patch]
            b, s = augment([b, s], self.rng)
        return {"blur": to_tensor(b), "sharp": to_tensor(s), "id": b_rel}


class SyntheticNonUniformDataset(Dataset):
    """Training data: (B, I, S) generated on the fly from sharp frames."""

    def __init__(self, root: str | Path, sharp_manifest: str | Path, scale: SeverityScale,
                 s_range=(0.0, 0.75), regions=(1, 2, 3, 4), patch: int = 256, margin: int = 20,
                 noise_sigma: float = 0.01, length: int = 100_000, seed: int = 0):
        self.root = Path(root)
        self.frames = read_list(sharp_manifest)
        self.scale, self.s_range, self.regions = scale, tuple(s_range), tuple(regions)
        self.patch, self.margin, self.noise_sigma, self.length = patch, margin, noise_sigma, length
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, i))
        img = load_rgb(self.root / self.frames[int(rng.integers(len(self.frames)))])
        big = self.patch + 2 * self.margin
        h, w = img.shape[:2]
        y, x = int(rng.integers(0, h - big + 1)), int(rng.integers(0, w - big + 1))
        crop = img[y:y + big, x:x + big]
        r = int(rng.choice(self.regions))
        sample = blur_nonuniform(crop, rng, self.scale, r, self.s_range, self.noise_sigma, margin=self.margin)
        sharp = crop[self.margin:-self.margin, self.margin:-self.margin]
        b, s, m = augment([sample.blur, sharp, sample.severity_map], rng)
        return {"blur": to_tensor(b), "sharp": to_tensor(s), "smap": to_tensor(m), "regions": r}


class FixedSyntheticDataset(Dataset):
    """Reads a test set written by ``scripts/generate_testsets.py`` (npz per image)."""

    def __init__(self, folder: str | Path):
        self.files = sorted(Path(folder).glob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        z = np.load(self.files[i])
        return {"blur": to_tensor(z["blur"]), "sharp": to_tensor(z["sharp"]),
                "smap": to_tensor(z["smap"]), "regions": int(z["regions"]),
                "group": str(z["group"]), "id": self.files[i].stem}
