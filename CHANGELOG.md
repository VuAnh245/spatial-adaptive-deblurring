# Changelog

## Shared infrastructure

Shared modules (`src/spatial_deblur/{data,degradations,models,classical,engine,metrics,utils}`) are kept identical to
`deblurring-for-detection`. Version: **core-v0.1.0**. Freeze them after the public checkpoint numbers are reproduced
(checklist step 4-5); every later fix is applied to both repos and recorded here.

## core-v0.1.1 - 2026-10-05
- `models/tlc.py`: Test-time Local Converter (port of NAFNet `local_arch.py`, exact variant).
  The official GoPro test config uses `NAFNetLocal`; without it the public width32 checkpoint scores
  clearly below the published 32.87 dB on full 1280x720 frames (exact gap: SA-00 reproduce run).
  `load_nafnet_public(tlc=True)`.
  Apply the same change to `deblurring-for-detection`.
- Test sets reduced for the 5-day schedule (val 50/R, test_in 100/R, ood 100, straight/curved 50).

## core-v0.1.0 - 2026-10-04
- Initial shared infrastructure: kernel generator with RMS-radius severity, soft masks, uniform / non-uniform blur,
  NAFNet port, public checkpoint loaders, trainer with resume, tiled inference, metrics, grouped statistics.
