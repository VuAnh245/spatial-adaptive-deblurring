"""Soft region masks for non-uniform blur.

Shared infrastructure. Each region gets a smooth random field; a
temperature-controlled softmax turns the fields into masks that sum to 1
at every pixel. Small temperatures give sharper transitions. Hard
boundaries are avoided because they create edges real blur does not have.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def soft_masks(rng: np.random.Generator, n_regions: int, height: int, width: int,
               grid: int | tuple[int, int] = 6, temperature: float | None = None) -> np.ndarray:
    """Return masks of shape (n_regions, H, W), non-negative, summing to 1.

    ``grid`` controls region size: a (rows, cols) random field upsampled to the
    image gives a few large regions (motion blur varies over objects, not texture).
    An int means a square grid. Pick rows/cols proportional to H/W for square cells
    (QĐ-02: training patches draw 3..6, full test frames use 6 x 10).
    """
    if n_regions < 1:
        raise ValueError("n_regions must be >= 1")
    if n_regions == 1:
        return np.ones((1, height, width), dtype=np.float32)
    gh, gw = (grid, grid) if isinstance(grid, int) else (int(grid[0]), int(grid[1]))
    temperature = rng.uniform(0.05, 0.2) if temperature is None else temperature
    fields = rng.normal(size=(n_regions, gh, gw))
    up = np.stack([zoom(f, (height / gh, width / gw), order=3) for f in fields])
    up = up[:, :height, :width]
    up = (up - up.mean(axis=(1, 2), keepdims=True)) / (up.std(axis=(1, 2), keepdims=True) + 1e-8)
    logits = up / temperature
    logits -= logits.max(axis=0, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=0, keepdims=True)).astype(np.float32)
