"""Motion-blur kernel generation and severity labels.

Shared infrastructure (identical in spatial-adaptive-deblurring and
deblurring-for-detection). Do not put study-specific logic here.

Severity of a kernel K is its RMS radius around the centre of mass,
normalised to [0, 1] with fixed r_min / r_max taken from the config:

    mu  = sum_x K(x) x
    r_K = sqrt( sum_x K(x) ||x - mu||^2 )
    s   = clip( (r_K - r_min) / (r_max - r_min), 0, 1 )

Trajectory length is NOT used as the label: a long trajectory that curls
up produces a compact kernel. Direction is a nuisance factor, not severity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KERNEL_SIZE = 41


@dataclass(frozen=True)
class SeverityScale:
    """Fixed normalisation for the severity label (set once in config)."""

    r_min: float = 0.0
    r_max: float = 8.0

    def normalise(self, r: float) -> float:
        return float(np.clip((r - self.r_min) / (self.r_max - self.r_min), 0.0, 1.0))

    def radius_for(self, s: float) -> float:
        return self.r_min + float(s) * (self.r_max - self.r_min)


def rms_radius(kernel: np.ndarray) -> float:
    """RMS radius of a non-negative kernel around its centre of mass (pixels)."""
    k = np.asarray(kernel, dtype=np.float64)
    total = k.sum()
    if total <= 0:
        raise ValueError("kernel must have positive mass")
    k = k / total
    ys, xs = np.mgrid[0 : k.shape[0], 0 : k.shape[1]]
    my, mx = (k * ys).sum(), (k * xs).sum()
    return float(np.sqrt((k * ((ys - my) ** 2 + (xs - mx) ** 2)).sum()))


def severity(kernel: np.ndarray, scale: SeverityScale) -> float:
    return scale.normalise(rms_radius(kernel))


def random_trajectory(rng: np.random.Generator, n_steps: int = 64,
                      curvature: float | None = None, jitter: float | None = None) -> np.ndarray:
    """Random-walk camera trajectory with momentum (Boracchi-Foi style).

    Returns an (n_steps, 2) array of sub-pixel positions with unit total
    scale; the caller rescales it to reach a target RMS radius.
    """
    curvature = rng.uniform(0.0, 0.6) if curvature is None else curvature
    jitter = rng.uniform(0.0, 0.3) if jitter is None else jitter
    angle = rng.uniform(0, 2 * np.pi)
    velocity = np.array([np.cos(angle), np.sin(angle)])
    pos = np.zeros(2)
    traj = [pos.copy()]
    for _ in range(n_steps - 1):
        turn = rng.normal(0.0, curvature)
        c, s_ = np.cos(turn), np.sin(turn)
        velocity = np.array([c * velocity[0] - s_ * velocity[1], s_ * velocity[0] + c * velocity[1]])
        step = velocity + rng.normal(0.0, jitter, size=2)
        pos = pos + step / max(np.linalg.norm(step), 1e-8)
        traj.append(pos.copy())
    traj = np.asarray(traj)
    traj -= traj.mean(axis=0)
    spread = np.sqrt((traj ** 2).sum(axis=1).mean())
    return traj / max(spread, 1e-8)


def rasterize(traj: np.ndarray, size: int = KERNEL_SIZE) -> np.ndarray:
    """Bilinear splat of a centred trajectory onto a size x size kernel (sums to 1)."""
    k = np.zeros((size, size), dtype=np.float64)
    c = (size - 1) / 2.0
    for y, x in traj[:, ::-1] + c:  # traj columns are (x, y)
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        dy, dx = y - y0, x - x0
        for yy, xx, w in ((y0, x0, (1 - dy) * (1 - dx)), (y0, x0 + 1, (1 - dy) * dx),
                          (y0 + 1, x0, dy * (1 - dx)), (y0 + 1, x0 + 1, dy * dx)):
            if 0 <= yy < size and 0 <= xx < size:
                k[yy, xx] += w
    if k.sum() == 0:
        k[int(c), int(c)] = 1.0
    return (k / k.sum()).astype(np.float32)


def kernel_for_severity(rng: np.random.Generator, s_target: float, scale: SeverityScale,
                        size: int = KERNEL_SIZE, tol: float = 0.02, max_iter: int = 30,
                        curvature: float | None = None, jitter: float | None = None
                        ) -> tuple[np.ndarray, float]:
    """Sample a kernel whose measured severity is close to ``s_target``.

    The trajectory shape is random; only its scale is searched (bisection),
    and the returned severity is measured on the final kernel. Pass
    ``curvature=0, jitter=0`` for the straight-kernel test split.
    """
    traj = random_trajectory(rng, curvature=curvature, jitter=jitter)
    target_r = scale.radius_for(s_target)
    if target_r < 0.5:  # near-delta kernel
        k = rasterize(traj * 1e-3, size)
        return k, severity(k, scale)
    lo, hi = 0.0, (size - 1) / 2.0
    k = rasterize(traj * hi, size)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        k = rasterize(traj * mid, size)
        r = rms_radius(k)
        if abs(r - target_r) < tol * scale.r_max:
            break
        lo, hi = (mid, hi) if r < target_r else (lo, mid)
    return k, severity(k, scale)
