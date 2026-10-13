# Changelog

## Shared infrastructure

Shared modules (`src/spatial_deblur/{data,degradations,models,classical,engine,metrics,utils}`) are kept identical to
`deblurring-for-detection`. Version: **core-v0.1.0**. Freeze them after the public checkpoint numbers are reproduced
(checklist step 4-5); every later fix is applied to both repos and recorded here.

## core-v0.1.2 - 2026-10-13
- `engine/trainer.Trainer(fail_fast_nonfinite=False)`: in FP32 (no GradScaler), a non-finite loss or gradient norm
  stops the run BEFORE `optimizer.step` (`abort_reason` `nonfinite_loss` / `nonfinite_grad`, events
  `aborted_nonfinite` / `aborted_nonfinite_grad`). Default off, so earlier behaviour is unchanged; the study sets it in
  `configs/_base_/train.yaml` (QĐ-08, commit 862da1d). Apply the same change to `deblurring-for-detection`.
- Study code only (not shared, despite living in `utils/`): `utils/labels.py` gained SA-00 and SA-32 to SA-36 labels.
- Tag `core-v0.1.x` was never created as a git tag; versions are tracked by this file.

## core-v0.1.1 - 2026-10-05
- `models/tlc.py`: Test-time Local Converter (port of NAFNet `local_arch.py`, exact variant).
  The official GoPro test config uses `NAFNetLocal`; without it the public width32 checkpoint scores
  32.294 dB instead of 32.870 dB (published 32.87) on full 1280x720 GoPro test frames (SA-00).
  `load_nafnet_public(tlc=True)`.
  Apply the same change to `deblurring-for-detection`.
- `metrics/stats.verdict` returns IDs `first_better` / `second_better` / `no_difference` (was "A better" /
  "B better"; letters A/B are reserved by docs/naming.md). Labels are added by the caller.
- `degradations/masks.soft_masks(grid=(rows, cols))` and `blur_nonuniform(grid=...)`: non-square grids
  (an int still means a square grid). `SyntheticNonUniformDataset(mask_grids=...)` draws a square grid per patch.
- `engine/trainer.Trainer(diag_fn=..., diag_every=...)`: generic hook, rows written to `diag.jsonl`.
  `curves.csv` rows also carry `sec` (since fit start) and `data_frac` (share of time blocked on the loader);
  `val.jsonl` rows carry `val_sec`. `Trainer(val_at_start=True)` validates at step 0 of a fresh run;
  `events.jsonl` records start / ckpt / stopped / finished; setting `trainer.stop` saves `last.pth` and
  returns `fit() -> False` (used for SIGTERM at the notebook time cap).
  `last.pth` also stores torch / CUDA / numpy / python RNG states; `resume` restores them and loads with
  `weights_only=False` (own file; default became True in torch 2.6).
- `data/manifests`: `video_of`, and `split_by_scene` / `assert_disjoint_scenes(level="video")` (default) check
  splits at source-video level (clips `GOPRxxxx_yy_NN` merged). `build_manifests.py` picks val from videos with a
  single clip in train and none in test. Val changed: old sequences GOPR0374_11_00 + GOPR0386_11_00
  (GOPR0374_11 had 3 more clips in train), content sha256 `5350747d6c56f2b6`; new val GOPR0379_11_00 +
  GOPR0857_11_00, content sha256 `96ca0c00be35f399` (commit ae167ee, Kaggle 2026-10-06).
- `scripts/compare.py --group-by video` (default, QĐ-05) and `win_rate_images`.
- `models/nafnet.LayerNorm2d`: statistics in FP32 inside autocast (FP16 (x - mu)^2 overflows above ~256);
  identical in FP32, so SA-00 is unaffected.
- `engine/trainer`: `curves.csv` adds `grad_norm`, `amp_scale`, `nonfinite_run`; abort after more than
  `max_nonfinite` (50) consecutive non-finite losses (`aborted_nonfinite`, `fit() -> False`, `abort_reason`),
  and periodic checkpoints are skipped (`ckpt_skipped_nonfinite`) while the loss is non-finite.
  Floor guard `Trainer(val_floor_drop=...)`: a val pass below the step-0 val minus the margin (or non-finite) stops
  the run (`aborted_val_drop`, `abort_reason='val_drop'`, last.pth not overwritten; train.py exits 4); the step-0
  baseline is re-read from val.jsonl on resume; the final step is validated before its checkpoint.
  `Trainer(grad_accum=...)`: one optimizer step over several micro-batches (loss / grad_accum).
  Stall guard: grad_norm = 0 at `max_zero_grad_logs` (3) consecutive log rows -> `aborted_stalled`;
  enabled GradScaler with scale < 1 -> `aborted_amp_scale`. train.py exit codes 3 / 4 / 5 / 6.
- Study config: `amp: false` (QĐ-07, FP32 after AMP numeric failures in pilots 2 and 3), `grad_accum: 1`.
  SA-15 redefined as "SFT-Map chỉ ở decoder" (middle block beta ~0.0036 in the public checkpoint).
- Study code: `modulation/analysis.val_metrics` (val pass used by train.py, tested); diag adds middle-block |beta|.
  Training lr 2e-4 -> 1e-4 after pilot 2 diverged at the peak lr (QĐ-04).
- Study code (not shared): Baseline-L param matching QĐ-06 (2 blocks at dec1 instead of 28 at dec3);
  val Spearman only on R >= 2 with nanmean (was NaN), plus `val_map_mae`.
- Test sets reduced for the 5-day schedule (val 50/R, test_in 100/R, ood 100, straight/curved 50).

## core-v0.1.0 - 2026-10-04
- Initial shared infrastructure: kernel generator with RMS-radius severity, soft masks, uniform / non-uniform blur,
  NAFNet port, public checkpoint loaders, trainer with resume, tiled inference, metrics, grouped statistics.
