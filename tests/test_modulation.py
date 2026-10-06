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


def test_param_match_within_five_percent_of_extra():
    """QĐ-06: Baseline-L adds the fewest blocks whose params match SFT-Map's extra params within 5%."""
    ref, base = build_variant("A2", TINY), build_variant("A0", TINY)
    m = build_variant("A0+", TINY)
    extra = n_params(ref) - n_params(base)
    assert abs((n_params(m) - n_params(base)) - extra) / extra <= 0.05
    assert m.param_match_info["extra_blocks"] >= 1


def _conv_gmacs(model, size=256):
    """GMACs of all convolutions for one 3 x size x size image (forward hooks; elementwise ops ignored)."""
    total = [0]

    def hook(mod, inp, out):
        kh, kw = mod.kernel_size
        total[0] += out.numel() // out.shape[0] * (mod.in_channels // mod.groups) * kh * kw

    hs = [mod.register_forward_hook(hook) for mod in model.modules() if isinstance(mod, torch.nn.Conv2d)]
    with torch.no_grad():
        model.eval()(torch.rand(1, 3, size, size))
    for h in hs:
        h.remove()
    return total[0] / 1e9


def test_tlc_converts_every_global_pool_including_a3_channel_branch():
    import copy
    from spatial_deblur.models.tlc import LocalAvgPool2d, convert_tlc
    m = build_variant("A3", TINY).eval()
    _nonzero_residuals(m)
    for mod in list(m.sft.values()) + list(m.chan.values()):  # non-zero heads
        mod.conv2.weight.data.normal_(0, 0.1) if hasattr(mod, "conv2") else mod.fc2.weight.data.normal_(0, 0.1)
    t = convert_tlc(copy.deepcopy(m), train_size=32)
    assert not any(isinstance(x, torch.nn.AdaptiveAvgPool2d) for x in t.modules())
    assert sum(isinstance(x, LocalAvgPool2d) for x in t.chan.modules()) == len(t.chan)
    with torch.no_grad():
        small, big = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 96, 128)
        assert torch.allclose(m(small)[0], t(small)[0], atol=1e-5)
        assert not torch.allclose(m(big)[0], t(big)[0], atol=1e-5)


def test_trainer_logs_l_map_and_sft_diagnostics(tmp_path):
    """Every diag_every steps: ||W|| of each SFT last conv, mean |dgamma_s|, |beta_s| per stage, L_map apart."""
    import json
    from torch.utils.data import DataLoader, Dataset
    from spatial_deblur.engine.trainer import Trainer
    from spatial_deblur.modulation.analysis import make_train_diag
    from spatial_deblur.modulation.losses import make_loss

    class Fake(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {"blur": torch.rand(3, 32, 32, generator=g), "sharp": torch.rand(3, 32, 32, generator=g),
                    "smap": torch.rand(1, 32, 32, generator=g), "regions": 2}

    def run(variant, lam, steps, every):
        m = build_variant(variant, TINY)
        _nonzero_residuals(m)  # like the public checkpoint: residual scales are not zero
        out = tmp_path / variant
        Trainer(m, make_loss(lam), DataLoader(Fake(), batch_size=2), out, lr=1e-3, iterations=steps, warmup=1,
                amp=False, ckpt_every=100, log_every=1, device="cpu",
                diag_fn=make_train_diag(), diag_every=every).fit()
        return m, out

    for variant, lam in (("A2", 0.1), ("A1", 0.0)):  # SFT-Map and SFT-Free
        m, out = run(variant, lam, 4, 2)
        rows = [json.loads(x) for x in (out / "diag.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rows] == [2, 4]
        for key in m.sft:
            assert {f"w_sft_{key}", f"dgamma_{key}", f"beta_{key}"} <= set(rows[0]), key
        assert {"l_map", "charbonnier", "loss"} <= set(rows[0])
        assert rows[-1][f"w_sft_{next(iter(m.sft))}"] > 0  # zero-initialised last conv has moved
        curves = [json.loads(x) for x in (out / "curves.csv").read_text().splitlines()]
        assert len(curves) == 4 and all("l_map" in c for c in curves)  # logged even when lambda_map = 0

    _, out = run("A0", 0.0, 2, 1)  # Baseline: no SFT heads -> no diag file
    assert not (out / "diag.jsonl").exists()


def test_trainer_val_at_start_and_graceful_stop(tmp_path):
    """Val at step 0 of a fresh run. A stop request (SIGTERM in train.py) after step 3 saves last.pth and no
    final.pth. A fresh trainer resumed from it continues at step 3 with the cosine-schedule lr of step 3,
    the same GradScaler scale and the same RNG stream."""
    import json
    import math
    from torch.utils.data import DataLoader, TensorDataset
    from spatial_deblur.engine.trainer import Trainer

    class Wrap(torch.utils.data.Dataset):
        def __init__(self):
            self.t = TensorDataset(torch.rand(8, 3, 32, 32), torch.rand(8, 3, 32, 32))

        def __len__(self):
            return 8

        def __getitem__(self, i):
            b, s = self.t[i]
            return {"blur": b, "sharp": s}

    calls, holder = {"n": 0}, {}

    def loss_fn(model, batch):
        calls["n"] += 1
        if calls["n"] == 3:
            holder["trainer"].stop = True  # takes effect at the top of the next step: stop after step 3
        out, _ = model(batch["blur"])
        loss = (out - batch["sharp"]).abs().mean()
        return loss, {"loss": loss.detach()}

    LR, WARM, ITERS, MIN_LR = 1e-3, 2, 10, 1e-7

    def make(scale):
        m = build_variant("A0", TINY)
        _nonzero_residuals(m)
        tr = Trainer(m, loss_fn, DataLoader(Wrap(), batch_size=2), tmp_path, lr=LR, iterations=ITERS, warmup=WARM,
                     min_lr=MIN_LR, amp=False, log_every=1, val_every=100, device="cpu", val_at_start=True,
                     val_fn=lambda model: {"val_psnr": 1.0})
        tr.scaler = torch.amp.GradScaler("cpu", init_scale=scale)  # enabled scaler, so its state is meaningful
        return tr

    a = make(1234.0)
    holder["trainer"] = a
    assert a.fit() is False and a.step == 3
    assert (tmp_path / "last.pth").exists() and not (tmp_path / "final.pth").exists()
    events = [json.loads(x)["event"] for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events == ["stopped"]
    saved = torch.load(tmp_path / "last.pth", map_location="cpu", weights_only=False)
    assert {"model", "opt", "sched", "scaler", "step", "rng"} <= set(saved)
    assert {"torch", "numpy", "python", "cuda"} <= set(saved["rng"])
    torch.set_rng_state(saved["rng"]["torch"])
    expected_draw = torch.rand(4)

    torch.manual_seed(999)  # scramble the global RNG before resuming
    b = make(1.0)
    holder["trainer"] = b
    assert b.resume() and b.step == 3
    factor = (MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * (3 - WARM) / (ITERS - WARM)))) / LR
    assert math.isclose(b.opt.param_groups[0]["lr"], LR * factor, rel_tol=1e-9)  # lr of the next step (step 4)
    assert b.scaler.get_scale() == a.scaler.get_scale() == 1234.0
    assert torch.equal(torch.rand(4), expected_draw)
    calls["n"] = 100  # no further stop request
    assert b.fit() is True and b.step == ITERS and (tmp_path / "final.pth").exists()
    vals = [json.loads(x)["step"] for x in (tmp_path / "val.jsonl").read_text().splitlines()]
    assert vals == [0]  # val at start only for the fresh run, not after resume


def test_val_metrics_spearman_is_one_for_a_perfect_map():
    """Through the val code path used in training: S_hat = down4(S) gives Spearman ~ 1 on R >= 2 images,
    R = 1 images (constant S) are skipped instead of turning the mean into NaN."""
    import math
    import torch.nn.functional as F
    from spatial_deblur.modulation.analysis import val_metrics

    h, w = 64, 96
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    ramps = [0.1 + 0.6 * (0.3 * yy + 0.7 * xx), 0.7 - 0.5 * (0.8 * yy + 0.2 * xx)]
    val_set = [{"blur": torch.rand(3, h, w), "sharp": torch.rand(3, h, w), "smap": r[None], "regions": 3} for r in ramps]
    val_set.append({"blur": torch.rand(3, h, w), "sharp": torch.rand(3, h, w),
                    "smap": torch.full((1, h, w), 0.4), "regions": 1})
    maps = iter([F.interpolate(s["smap"][None], scale_factor=0.25, mode="area") for s in val_set])

    def perfect_model(x):
        return x, next(maps)  # S_hat = down4(S), exactly what L_map trains towards

    out = val_metrics(perfect_model, val_set, "cpu")
    assert math.isfinite(out["val_spearman"]) and out["val_spearman"] > 0.999, out
    assert out["val_map_mae"] < 0.01, out


def test_trainer_aborts_on_nonfinite_loss_and_keeps_last_good_checkpoint(tmp_path):
    """More than max_nonfinite consecutive non-finite losses -> fit() returns False, event aborted_nonfinite,
    and last.pth still holds the last finite state (later checkpoints are skipped, not overwritten)."""
    import json
    from torch.utils.data import DataLoader, TensorDataset
    from spatial_deblur.engine.trainer import Trainer

    class Wrap(torch.utils.data.Dataset):
        def __init__(self):
            self.t = TensorDataset(torch.rand(8, 3, 32, 32), torch.rand(8, 3, 32, 32))

        def __len__(self):
            return 8

        def __getitem__(self, i):
            b, s = self.t[i]
            return {"blur": b, "sharp": s}

    calls = {"n": 0}

    def loss_fn(model, batch):
        calls["n"] += 1
        out, _ = model(batch["blur"])
        loss = (out - batch["sharp"]).abs().mean()
        if calls["n"] > 4:
            loss = loss * float("nan")  # diverged from step 5 on
        return loss, {"loss": loss.detach()}

    m = build_variant("A0", TINY)
    _nonzero_residuals(m)
    tr = Trainer(m, loss_fn, DataLoader(Wrap(), batch_size=2), tmp_path, lr=1e-3, iterations=200, warmup=1,
                 amp=False, log_every=1, ckpt_every=2, device="cpu", max_nonfinite=50)
    assert tr.fit() is False and tr.abort_reason == "nonfinite_loss"
    saved = torch.load(tmp_path / "last.pth", map_location="cpu", weights_only=False)
    assert saved["step"] == 4 and all(torch.isfinite(v).all() for v in saved["model"].values())
    events = [json.loads(x)["event"] for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events.count("ckpt") == 2 and "ckpt_skipped_nonfinite" in events and events[-1] == "aborted_nonfinite"
    assert not (tmp_path / "final.pth").exists()
    rows = [json.loads(x) for x in (tmp_path / "curves.csv").read_text().splitlines()]
    assert {"grad_norm", "amp_scale", "nonfinite_run"} <= set(rows[0])


def test_trainer_aborts_on_val_drop_below_step0_floor(tmp_path):
    """Collapse with a finite loss (pilot 2): a val pass more than 1 dB below the step-0 val stops the run,
    event aborted_val_drop, last.pth keeps the last checkpoint that passed val; the floor survives a resume."""
    import json
    from torch.utils.data import DataLoader, TensorDataset
    from spatial_deblur.engine.trainer import Trainer

    class Wrap(torch.utils.data.Dataset):
        def __init__(self):
            self.t = TensorDataset(torch.rand(8, 3, 32, 32), torch.rand(8, 3, 32, 32))

        def __len__(self):
            return 8

        def __getitem__(self, i):
            b, s = self.t[i]
            return {"blur": b, "sharp": s}

    def loss_fn(model, batch):
        out, _ = model(batch["blur"])
        loss = (out - batch["sharp"]).abs().mean()
        return loss, {"loss": loss.detach()}

    scripted = iter([30.0, 31.5, 30.2, 28.9])  # steps 0, 2, 4, 6: 28.9 < 30.0 - 1.0 -> abort at step 6

    def make():
        m = build_variant("A0", TINY)
        return Trainer(m, loss_fn, DataLoader(Wrap(), batch_size=2), tmp_path, lr=1e-3, iterations=20, warmup=1,
                       amp=False, log_every=1, val_every=2, ckpt_every=2, device="cpu", val_at_start=True,
                       val_fn=lambda model: {"val_psnr": next(scripted)}, val_floor_drop=1.0)

    tr = make()
    assert tr.fit() is False and tr.abort_reason == "val_drop" and tr.step == 6
    saved = torch.load(tmp_path / "last.pth", map_location="cpu", weights_only=False)
    assert saved["step"] == 4  # step 6 failed val, so its checkpoint was never written
    ev = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in ev] == ["ckpt", "ckpt", "aborted_val_drop"]
    assert ev[-1]["baseline"] == 30.0 and abs(ev[-1]["drop"] - 1.1) < 1e-9
    assert not (tmp_path / "final.pth").exists()

    tr2 = make()
    assert tr2.resume() and tr2.step == 4 and tr2.val_baseline == 30.0  # floor kept across resumes


def test_param_match_real_config():
    """QĐ-06 on NAFNet-32: 2 blocks at dec1 (17,349,731 params); compute stays within 10% of SFT-Map."""
    ref, base = build_variant("A2"), build_variant("A0")
    m = build_variant("A0+")
    assert m.param_match_info["stage"] == "dec1" and m.param_match_info["extra_blocks"] == 2
    assert n_params(m) == 17_349_731
    extra = n_params(ref) - n_params(base)
    assert abs((n_params(m) - n_params(base)) - extra) / extra <= 0.05
    g_plus, g_map = _conv_gmacs(m), _conv_gmacs(ref)
    assert g_plus <= 1.10 * g_map, (g_plus, g_map)
