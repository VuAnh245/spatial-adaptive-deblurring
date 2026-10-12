"""Small graph/label checks for audit; no checkpoint, optimizer, or training run.

The in-memory model is deliberately small and its residual gates are opened
only to test gradient connectivity. Results do not diagnose trained weights.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from spatial_deblur.degradations.blur import blur_nonuniform
    from spatial_deblur.degradations.kernels import SeverityScale, severity
    from spatial_deblur.models.nafnet import NAFBlock
    from spatial_deblur.modulation.analysis import map_quality
    from spatial_deblur.modulation.losses import make_loss
    from spatial_deblur.modulation.variants import build_variant

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1011)
    rng = np.random.default_rng(1011)
    scale = SeverityScale(0, 8)
    sample = blur_nonuniform(rng.random((72, 88, 3), dtype=np.float32), rng, scale, 3,
                            (0, .75), margin=20, targets=[.1, .35, .7], grid=(3, 4))
    reconstructed = np.tensordot(np.asarray(sample.severities, np.float32), sample.masks, axes=1)
    label_error = float(np.max(np.abs(sample.severity_map - reconstructed)))
    radius_error = max(abs(s - severity(k, scale)) for s, k in zip(sample.severities, sample.kernels))
    assert label_error < 1e-6 and radius_error < 1e-6
    assert sample.blur.shape == (32, 48, 3) and sample.severity_map.shape == (32, 48)

    coarse = torch.linspace(.1, .7, 8 * 12).reshape(1, 1, 8, 12)
    target = F.interpolate(coarse, size=(32, 48), mode="bilinear", align_corners=False)
    identical = map_quality(coarse, target)
    constant = map_quality(torch.full_like(coarse, .4), target)
    assert abs(identical["spearman"] - 1) < 1e-8 and identical["mae"] < 1e-7
    assert math.isnan(constant["spearman"])

    kw = {"width": 8, "enc_blk_nums": (1, 1), "dec_blk_nums": (1, 1), "middle_blk_num": 1}
    model = build_variant("A2", kw)
    with torch.no_grad():
        for block in model.modules():
            if isinstance(block, NAFBlock):
                block.beta.fill_(.1)
                block.gamma.fill_(.1)
    batch = {"blur": torch.rand(1, 3, 32, 48), "sharp": torch.rand(1, 3, 32, 48),
             "smap": torch.full((1, 1, 32, 48), .2)}

    def gradient(weight, detach=False):
        model.detach_map = detach
        model.zero_grad(set_to_none=True)
        loss, parts = make_loss(weight)(model, batch)
        loss.backward()
        est = sum(float(p.grad.abs().sum()) for p in model.estimator.parameters() if p.grad is not None)
        heads = sum(float(p.grad.abs().sum()) for p in model.sft.parameters() if p.grad is not None)
        return {"estimator_grad_l1": est, "sft_grad_l1": heads, "loss": float(loss.detach()),
                "charbonnier": float(parts["charbonnier"]), "l_map": float(parts["l_map"])}

    free_zero = gradient(0)
    map_zero = gradient(.1)
    with torch.no_grad():
        for head in model.sft.values():
            head.conv2.weight.normal_(0, .001)
    free_open = gradient(0)
    detached = gradient(0, True)
    with torch.no_grad():
        forward_map = model(batch["blur"])[1]
        estimator_map = model.estimator(model.pad(batch["blur"]))
    estimator_only_equals_forward = torch.equal(forward_map, estimator_map)
    assert estimator_only_equals_forward
    assert free_zero["estimator_grad_l1"] == 0 and free_zero["sft_grad_l1"] > 0
    assert map_zero["estimator_grad_l1"] > 0 and free_open["estimator_grad_l1"] > 0
    assert detached["estimator_grad_l1"] == 0
    result = {
        "scope": "Synthetic toy checks only; no original validation/test data or trained checkpoint",
        "torch": torch.__version__, "cuda": torch.cuda.is_available(), "seed": 1011,
        "label_reconstruction_max_abs_error": label_error, "kernel_severity_max_abs_error": radius_error,
        "bilinear_identity_map_quality": identical, "constant_map_spearman": None,
        "constant_map_spearman_is_undefined": True,
        "free_zero_sft": free_zero, "supervised_zero_sft": map_zero,
        "free_nonzero_sft": free_open, "detached_nonzero_sft": detached,
        "estimator_only_equals_full_forward_map": estimator_only_equals_forward,
        "checks_passed": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
