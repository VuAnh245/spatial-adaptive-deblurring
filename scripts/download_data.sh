#!/usr/bin/env bash
# Download data and checkpoints for spatial-adaptive-deblurring.
# Run on Colab / Kaggle from the repo root. Links: docs/datasets.md
set -euo pipefail
mkdir -p data checkpoints third_party/realblur
pip install -q gdown huggingface_hub

# GoPro (CC BY 4.0) - paired blur/sharp, 1280x720
[ -d data/GOPRO_Large ] || {
  wget -c -O data/GOPRO_Large.zip https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip
  unzip -q data/GOPRO_Large.zip -d data && rm data/GOPRO_Large.zip
}

# NAFNet-GoPro-width32 checkpoint (MIT)
[ -f checkpoints/NAFNet-GoPro-width32.pth ] || \
  gdown 1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj -O checkpoints/NAFNet-GoPro-width32.pth

# RealBlur-J test (CC BY 4.0) - download manually from the official portal, then unzip to data/RealBlur
#   https://cgdata.postech.ac.kr/sharing/YhKdbtvD0
#   https://drive.google.com/drive/folders/1xUNAAVzLhNQuGriKTk1hrE-MT-H_56fq
# Official evaluation script (alignment) -> third_party/realblur/evaluation_RealBlur_ecc.py
#   https://github.com/rimchang/RealBlur
echo "GoPro and NAFNet done. RealBlur-J: download manually (see comments above)."
