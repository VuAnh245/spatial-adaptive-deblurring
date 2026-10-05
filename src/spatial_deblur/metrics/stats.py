"""Grouped statistics for paired model comparison.

Shared infrastructure. Frames of the same scene (or synthetic images made
from the same source sequence) are correlated, so per-image scores are
averaged per group first and every test runs on group-level values.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def group_mean(scores: dict[str, float], group_of: dict[str, str]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for image_id, value in scores.items():
        buckets[group_of[image_id]].append(value)
    return {g: float(np.mean(v)) for g, v in buckets.items()}


def paired_comparison(a: dict[str, float], b: dict[str, float], n_boot: int = 1000,
                      seed: int = 0) -> dict[str, float]:
    """Compare model A against B on the same groups.

    Returns mean difference (A - B), bootstrap 95% CI over groups,
    Wilcoxon signed-rank p-value and win-rate of A.
    """
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        raise ValueError("need at least two shared groups")
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, size=d.size, replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"n_groups": len(keys), "mean_diff": float(d.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "wilcoxon_p": p, "win_rate": float((d > 0).mean())}


def verdict(result: dict[str, float], min_effect: float = 0.1, alpha: float = 0.05) -> str:
    """Wording rule from the spec: small and non-significant means 'no difference'."""
    if result["wilcoxon_p"] >= alpha and abs(result["mean_diff"]) < min_effect:
        return "no difference"
    return "A better" if result["mean_diff"] > 0 else "B better"
