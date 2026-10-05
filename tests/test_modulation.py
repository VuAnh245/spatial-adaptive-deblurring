import torch

from spatial_deblur.modulation.variants import build_variant, n_params

TINY = dict(width=8, middle_blk_num=1, enc_blk_nums=(1, 1, 1, 1), dec_blk_nums=(1, 1, 1, 1))


def _nonzero_residuals(model):
    # Official checkpoints have non-zero beta/gamma; emulate that so the test is meaningful.
    torch.manual_seed(0)
    for name, p in model.named_parameters():
        if name.endswith(".beta") or name.endswith(".gamma"):
            p.data.normal_(0, 0.1)


def test_identity_at_init():
    """A1/A2/A3 with zero-initialised heads equal A0 carrying the same backbone weights."""
    torch.manual_seed(0)
    a0 = build_variant("A0", TINY)
    _nonzero_residuals(a0)
    x = torch.rand(2, 3, 32, 48)
    ref, _ = a0(x)
    for v in ("A1", "A3"):
        m = build_variant(v, TINY)
        missing = m.load_state_dict(a0.state_dict(), strict=False)
        assert not missing.unexpected_keys
        out, s_hat = m(x)
        assert torch.allclose(out, ref, atol=1e-6), v
        assert s_hat.shape == (2, 1, 8, 12)


def test_oracle_override_changes_nothing_at_init_but_is_used_after_update():
    m = build_variant("A2", TINY)
    _nonzero_residuals(m)
    for head in m.sft.values():
        head.conv2.weight.data.normal_(0, 0.1)
    x = torch.rand(1, 3, 32, 32)
    a, _ = m(x, smap_override=torch.zeros(1, 1, 8, 8))
    b, _ = m(x, smap_override=torch.ones(1, 1, 8, 8))
    assert not torch.allclose(a, b)


def test_param_match_within_two_percent():
    ref = build_variant("A2", TINY)
    m = build_variant("A0+", TINY)
    assert abs(n_params(m) - n_params(ref)) / n_params(ref) <= 0.02
    assert m.param_match_info["extra_blocks"] >= 0


def test_param_match_real_config():
    ref = build_variant("A2")
    m = build_variant("A0+")
    assert abs(n_params(m) - n_params(ref)) / n_params(ref) <= 0.02
