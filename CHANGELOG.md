# Changelog

## Shared infrastructure

Shared modules (`src/spatial_deblur/{data,degradations,models,classical,engine,metrics,utils}`) are kept identical to
`deblurring-for-detection`. Version: **core-v0.1.0**. Freeze them after the public checkpoint numbers are reproduced
(checklist step 4-5); every later fix is applied to both repos and recorded here.

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
  (GOPR0374_11 had 3 more clips in train), content sha256 `5350747d6c56f2b6`; new val sha256: (fill in after
  the first generation on Kaggle).
- `scripts/compare.py --group-by video` (default, QĐ-05) and `win_rate_images`.
- Study code (not shared): Baseline-L param matching QĐ-06 (2 blocks at dec1 instead of 28 at dec3);
  val Spearman only on R >= 2 with nanmean (was NaN), plus `val_map_mae`.
- Test sets reduced for the 5-day schedule (val 50/R, test_in 100/R, ood 100, straight/curved 50).

## core-v0.1.0 - 2026-10-04
- Initial shared infrastructure: kernel generator with RMS-radius severity, soft masks, uniform / non-uniform blur,
  NAFNet port, public checkpoint loaders, trainer with resume, tiled inference, metrics, grouped statistics.
