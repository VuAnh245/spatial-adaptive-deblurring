# Spatially Adaptive Deblurring for Non-Uniform Motion Blur

**Khử nhòe thích ứng theo không gian cho ảnh nhòe chuyển động không đồng nhất**
Đồ án môn Deep Learning - Vũ Xuân Anh (UTEHY)

## Câu hỏi nghiên cứu

Bản đồ mức nhòe có giám sát có giúp mạng khử nhòe thích ứng theo từng vùng tốt hơn attention
không giám sát, và tốt hơn việc chỉ thêm tham số không?

| Input | Output |
| --- | --- |
| Một ảnh RGB nhòe B (sRGB, [0, 1], H × W bất kỳ) | Ảnh khôi phục Î cùng kích thước + bản đồ mức nhòe Ŝ(x, y) ở 1/4 độ phân giải |

## Biến thể

| ID | Tên đọc được | Mô tả | Config |
| --- | --- | --- | --- |
| A0 | NAFNet gốc (Baseline) | NAFNet-32, không modulation | `configs/experiments/SA-10_a0.yaml` |
| A0+ | NAFNet tăng tham số (Baseline-L) | A0 + 2 NAFBlock khởi tạo đồng nhất ở dec1, phần tham số thêm vào lệch ≤ 5% so với phần thêm của A2 (QĐ-06) | `SA-11_a0plus.yaml` |
| A1 | Điều biến tự do (SFT-Free) | Bộ ước lượng + SFT theo không gian, **không** giám sát (lambda_map = 0) | `SA-12_a1.yaml` |
| A2 | Điều biến theo bản đồ nhòe (SFT-Map), mô hình đề xuất | Như A1 + giám sát L_map | `SA-13_a2.yaml` |
| A3 | Điều biến theo bản đồ nhòe + kênh (SFT-Map+Ch) | A2 + nhánh channel | `SA-14_a3.yaml` |

Quy ước ký hiệu, mã và tên đọc được: `docs/naming.md`.

Toàn bộ thí nghiệm: `docs/experiments.md`. Đặc tả đầy đủ: tài liệu đặc tả đồ án DL.

## Cấu trúc

```text
spatial-adaptive-deblurring/
├── src/spatial_deblur/
│   ├── data/            datasets, manifests, image statistics                 [shared]
│   ├── degradations/    kernels + severity label, soft masks, blur             [shared]
│   ├── models/          NAFNet port, public checkpoint loaders                  [shared]
│   ├── modulation/      blur-map estimator, SFT head, channel branch,
│   │                    variants A0-A3, loss, mechanism analyses               [contribution]
│   ├── classical/       Wiener, Richardson-Lucy                                 [shared]
│   ├── engine/          trainer (AMP, resume), tiled inference                  [shared]
│   ├── metrics/         PSNR/SSIM/LPIPS, grouped stats, efficiency, RealBlur    [shared]
│   └── utils/           config (_base_ inheritance), seed                       [shared]
├── configs/  _base_/ + experiments/SA-xx_*.yaml
├── scripts/  download_data.sh, build_manifests.py, generate_testsets.py,
│             train.py, evaluate.py, compare.py, benchmark_efficiency.py, make_figures.py
├── analysis/            notebooks / scripts producing report tables and figures
├── manifests/           split lists + sha256 (committed)
├── notebooks/           01_eda, 02_kernel_and_mask_stats, 03_results
├── tests/               unit tests
├── docs/                datasets.md, figure_sources.md, experiments.md, architecture.md, figures/
├── third_party/         official RealBlur evaluation (not committed)
├── checkpoints/ data/ results/   (gitignored)
├── pyproject.toml  Makefile  CITATION.cff  CHANGELOG.md  README.md
```

`[contribution]` = đóng góp của đồ án này. Các module còn lại trong `src/spatial_deblur/` là **hạ tầng dùng chung**
với đồ án môn Computer Vision (`deblurring-for-detection`, cùng tác giả), được viết một lần,
đóng băng ở phiên bản `core-v0.1.0` và ghi mọi thay đổi trong `CHANGELOG.md`.

## Bắt đầu

```bash
pip install -e ".[dev]"
make test                                   # unit test hạ tầng + module
bash scripts/download_data.sh               # GoPro, RealBlur-J, checkpoint NAFNet (xem docs/datasets.md)
python scripts/build_manifests.py --gopro data/GOPRO_Large
python scripts/generate_testsets.py --config configs/experiments/SA-13_a2.yaml
python scripts/make_figures.py --out docs/figures
make train CFG=configs/experiments/SA-13_a2.yaml
make eval  CFG=configs/experiments/SA-13_a2.yaml
```

Thứ tự bắt buộc trước khi train: tái lập số liệu checkpoint NAFNet-GoPro-width32 trên GoPro test và
RealBlur-J test (bước 4, 5 trong checklist của đặc tả).

## Dataset và nguồn hình

- Link dataset, checkpoint, giấy phép: `docs/datasets.md`
- Hình cho slide và báo cáo (nguồn paper + hình tự tạo): `docs/figure_sources.md`

## Trích dẫn

Xem `CITATION.cff`. Dataset và model bên ngoài giữ giấy phép gốc (GoPro, RealBlur: CC BY 4.0;
NAFNet: MIT).
