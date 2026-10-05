# Changelog

## Shared infrastructure

Shared modules (`src/spatial_deblur/{data,degradations,models,classical,engine,metrics,utils}`) are kept identical to
`deblurring-for-detection`. Version: **core-v0.1.0**. Freeze them after the public checkpoint numbers are reproduced
(checklist step 4-5); every later fix is applied to both repos and recorded here.

## core-v0.1.0 - 2026-10-04
- Initial shared infrastructure: kernel generator with RMS-radius severity, soft masks, uniform / non-uniform blur,
  NAFNet port, public checkpoint loaders, trainer with resume, tiled inference, metrics, grouped statistics.
