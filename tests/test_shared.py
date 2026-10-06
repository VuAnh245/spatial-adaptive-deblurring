import numpy as np
import torch

from spatial_deblur.data.manifests import assert_disjoint_scenes, split_by_scene
from spatial_deblur.degradations.blur import blur_nonuniform, blur_uniform
from spatial_deblur.degradations.kernels import SeverityScale, kernel_for_severity, rasterize, rms_radius
from spatial_deblur.degradations.masks import soft_masks
from spatial_deblur.metrics.image_quality import psnr
from spatial_deblur.metrics.stats import paired_comparison
from spatial_deblur.models.nafnet import NAFNet
from spatial_deblur.utils.config import load_config

SCALE = SeverityScale(0.0, 8.0)


def test_kernel_sums_to_one_and_severity_tracks_target():
    rng = np.random.default_rng(0)
    for s in (0.1, 0.4, 0.7):
        k, s_meas = kernel_for_severity(rng, s, SCALE)
        assert abs(k.sum() - 1) < 1e-5
        assert abs(s_meas - s) < 0.06


def test_severity_increases_with_spread():
    traj = np.stack([np.linspace(-1, 1, 64), np.zeros(64)], 1)
    radii = [rms_radius(rasterize(traj * a)) for a in (2, 5, 10)]
    assert radii[0] < radii[1] < radii[2]


def test_masks_sum_to_one():
    m = soft_masks(np.random.default_rng(1), 3, 64, 80)
    assert m.shape == (3, 64, 80)
    assert np.allclose(m.sum(0), 1, atol=1e-5) and (m >= 0).all()


def test_masks_non_square_grid_sum_to_one():
    """QĐ-02: (rows, cols) grids, e.g. 6 x 10 on a full frame, and every training grid size."""
    rng = np.random.default_rng(3)
    m = soft_masks(rng, 4, 76, 132, grid=(6, 10))
    assert m.shape == (4, 76, 132)
    assert np.allclose(m.sum(0), 1, atol=1e-5) and (m >= 0).all()
    for g in (3, 4, 5, 6):
        m = soft_masks(rng, 3, 74, 74, grid=(g, g))
        assert m.shape == (3, 74, 74) and np.allclose(m.sum(0), 1, atol=1e-5)
    img = rng.random((60, 90, 3)).astype(np.float32)
    nu = blur_nonuniform(img, rng, SCALE, 3, (0.0, 0.75), 0.01, margin=8, grid=(2, 3))
    assert nu.blur.shape == (44, 74, 3) and np.allclose(nu.masks.sum(0), 1, atol=1e-5)


def test_blur_maps():
    rng = np.random.default_rng(2)
    img = rng.random((96, 96, 3)).astype(np.float32)
    k, s = kernel_for_severity(rng, 0.3, SCALE)
    u = blur_uniform(img, k, s, 0.0, rng)
    assert np.allclose(u.severity_map, s)
    nu = blur_nonuniform(img, rng, SCALE, 3, (0.0, 0.75), 0.01, margin=8)
    assert nu.blur.shape == (80, 80, 3) and nu.severity_map.shape == (80, 80)
    assert nu.severity_map.min() >= min(nu.severities) - 1e-5
    assert nu.severity_map.max() <= max(nu.severities) + 1e-5


def test_split_by_scene_is_disjoint():
    """Splits are disjoint at VIDEO level: clips GOPRxxxx_yy_NN of one recording stay on one side."""
    from spatial_deblur.data.manifests import video_of
    # 10 videos x 2 clips x 5 frames
    items = [f"train/GOPR{i // 10:04d}_11_{(i // 5) % 2:02d}/sharp/{i:06d}.png" for i in range(100)]
    assert video_of(items[0], 2) == "GOPR0000_11" and video_of("train/scene3/img.png", 2) == "scene3"
    sel, rest = split_by_scene(items, 0.3, seed=0, depth=2)
    assert_disjoint_scenes(sel, rest, depth=2, level="video")
    assert not {video_of(s, 2) for s in sel} & {video_of(r, 2) for r in rest}
    assert len({video_of(s, 2) for s in sel}) == 3 and len(sel) == 30
    # a clip-level split of the same items shares videos and must be rejected at video level
    clip_a = [it for it in items if it.split("/")[1].endswith("_00")]
    clip_b = [it for it in items if it.split("/")[1].endswith("_01")]
    assert_disjoint_scenes(clip_a, clip_b, depth=2, level="scene")
    try:
        assert_disjoint_scenes(clip_a, clip_b, depth=2, level="video")
        raise RuntimeError("video overlap not detected")
    except AssertionError:
        pass


def test_psnr_and_stats():
    a = np.zeros((4, 4, 3), np.float32)
    assert abs(psnr(a + 0.1, a) - 20.0) < 1e-4
    r = paired_comparison({"g1": 30.0, "g2": 31.0, "g3": 29.5}, {"g1": 29.0, "g2": 30.5, "g3": 29.0})
    assert r["mean_diff"] > 0 and r["win_rate"] == 1.0


def test_config_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb: {x: 1, y: 2}\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nb: {y: 3}\n")
    assert load_config(tmp_path / "child.yaml") == {"a": 1, "b": {"x": 1, "y": 3}}


def test_layernorm2d_fp32_statistics():
    """LayerNorm2d equals a channel LayerNorm in FP32 and stays finite on large FP16 inputs
    (the plain FP16 version overflows: (x - mu)^2 > 65504 once |x - mu| > ~256)."""
    import torch.nn.functional as F
    from spatial_deblur.models.nafnet import LayerNorm2d
    torch.manual_seed(0)
    ln = LayerNorm2d(8)
    ln.weight.data.normal_(1, 0.1)
    ln.bias.data.normal_(0, 0.1)
    x = torch.randn(2, 8, 5, 7)
    ref = F.layer_norm(x.permute(0, 2, 3, 1), (8,), ln.weight, ln.bias, ln.eps).permute(0, 3, 1, 2)
    assert torch.allclose(ln(x), ref, atol=1e-5)
    big = (x * 2000).half()
    out = ln(big)
    assert out.dtype == torch.float16 and torch.isfinite(out).all()
    assert torch.allclose(out.float(), ref, atol=1e-2)


def test_nafnet_shapes_and_padding():
    m = NAFNet(width=8, enc_blk_nums=(1, 1, 1, 1), dec_blk_nums=(1, 1, 1, 1))
    x = torch.rand(1, 3, 37, 50)
    assert m(x).shape == x.shape


def test_local_avg_pool_matches_sliding_mean():
    import torch.nn.functional as F
    from spatial_deblur.models.tlc import LocalAvgPool2d
    p = LocalAvgPool2d((6, 6), (4, 4))
    p.kernel_size = [3, 5]
    x = torch.rand(2, 4, 9, 11)
    ref = F.avg_pool2d(x, (3, 5), stride=1)
    ref = F.pad(ref, (2, 2, 1, 1), mode="replicate")
    assert torch.allclose(p(x), ref, atol=1e-5)


def test_tlc_is_identity_at_train_size_and_local_beyond():
    import copy
    from spatial_deblur.models.tlc import LocalAvgPool2d, convert_tlc
    torch.manual_seed(0)
    m = NAFNet(width=8, enc_blk_nums=(1, 1, 1, 1), dec_blk_nums=(1, 1, 1, 1)).eval()
    for blk in m.modules():  # non-zero residual scales so the blocks are not identities
        if hasattr(blk, "beta"):
            blk.beta.data.normal_(0, 0.5); blk.gamma.data.normal_(0, 0.5)
    t = convert_tlc(copy.deepcopy(m), train_size=32)
    assert any(isinstance(x, LocalAvgPool2d) for x in t.modules())
    with torch.no_grad():
        small, big = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 96, 128)
        assert torch.allclose(m(small), t(small), atol=1e-5)
        assert not torch.allclose(m(big), t(big), atol=1e-5)
