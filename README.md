# Spatially Adaptive Deblurring for Non-Uniform Motion Blur

**Khử nhòe thích ứng theo không gian cho ảnh nhòe chuyển động không đồng nhất**
Đồ án môn Deep Learning - Vũ Xuân Anh (HYUTE)

## Câu hỏi nghiên cứu

Bản đồ mức nhòe có giám sát có giúp mạng khử nhòe thích ứng theo từng vùng tốt hơn attention
không giám sát, và tốt hơn việc chỉ thêm tham số không?

| Input | Output |
| --- | --- |
| Một ảnh RGB nhòe B (sRGB, [0, 1], H × W bất kỳ) | Ảnh khôi phục Î cùng kích thước + bản đồ mức nhòe Ŝ(x, y) ở 1/4 độ phân giải |

Đồ án không đề xuất kiến trúc mới. Đồ án dùng NAFNet-32 (Chen và cộng sự, 2022) và lớp SFT (Wang và cộng sự, 2018) có
sẵn, tự xây bộ sinh nhòe theo vùng kèm bản đồ mức nhòe thật S, rồi làm thí nghiệm có đối chứng. Công trình gần nhất: IKC
(CVPR 2019) và UFPNet (CVPR 2023).

## Kết quả chính (2026-10-14)

PSNR trung bình theo ảnh (dB, suy luận cục bộ TLC, FP32) trên `test_in` (400 ảnh tổng hợp, chỉ tiêu chính theo QĐ-09) và
GoPro test thật (1.111 ảnh):

| Mô hình | `test_in` | GoPro thật |
| --- | --- | --- |
| NAFNet công khai (không fine-tune) | 25,61 | 32,88 |
| Baseline (SA-10) | 33,95 | 28,37 |
| Baseline-L (SA-11) | 33,95 | 28,37 |
| SFT-Free (SA-12), trung bình 3 seed | 33,96 | 28,35 |
| SFT-Map (SA-13), trung bình 3 seed | 33,96 | 28,35 |
| SFT-Map+Ch (SA-14) | 33,97 | 28,37 |
| SFT-Free train lại trên Modal (SA-35, đối chứng cùng nền tảng) | 33,96 | 28,37 |
| SFT-Map huấn luyện với bản đồ thật (SA-32; cần S thật nên chỉ là trần tham chiếu) | 34,45 | - |

(1) Câu trả lời cho câu hỏi nghiên cứu là không. SFT-Map so với SFT-Free chênh -0,001 dB (3 seed), so với Baseline-L
    chênh +0,006 dB (seed 1); cả hai đều không khác biệt sau hiệu chỉnh Holm. Lý do: bản đồ Ŝ của bộ ước lượng E gần như
    hằng số (Spearman 0,04), và xáo trộn bản đồ giữa các ảnh làm PSNR đổi 0,000 dB.
(2) Khi bản đồ đúng, điều biến SFT có lợi. SA-32 hơn SA-35 0,50 dB trên `test_in` (thắng 10/10 video) và 0,64 dB trên
    `test_ood`; mức tăng lớn nhất ở ảnh nhòe không đồng nhất. Theo phép quét SA-34, bản đồ cần đạt Spearman khoảng 0,7 và
    MAE khoảng 0,07 mới hòa vốn với SFT-Free. Bộ ước lượng sửa của SA-33 chỉ đạt 0,37 và 0,15, và SA-33 không khác SFT-Free.
(3) Đánh đổi miền: so với checkpoint công khai, Baseline fine-tune trên ảnh tổng hợp tăng 8,3 dB trên `test_in` nhưng giảm
    4,5 dB trên GoPro thật. Nội suy trọng số (SA-36) không cho điểm nào giữ được cả hai miền.

Chi tiết, khoảng tin cậy (tính theo video) và các quyết định khai báo trước (QĐ-01 đến QĐ-12) nằm trong
`docs/experiments.md`, ở các mục "Kết quả đánh giá đợt 1", "Kết quả đánh giá đợt 2", "Bổ sung sau SA-35" và "QĐ-12".

Phạm vi đã cắt theo QĐ-10 (do ngân sách và thời hạn): RealBlur-J, LPIPS, độ bền với nhiễu (SA-26) và đo tốc độ (SA-31;
chỉ báo số tham số). Code của các phần này vẫn còn trong repo nhưng không được dùng để báo số liệu.

## Biến thể

| ID | Tên đọc được | Mô tả | Config |
| --- | --- | --- | --- |
| A0 | NAFNet gốc (Baseline) | NAFNet-32, không modulation | `configs/experiments/SA-10_a0.yaml` |
| A0+ | NAFNet tăng tham số (Baseline-L) | A0 + 2 NAFBlock khởi tạo đồng nhất ở dec1, phần tham số thêm vào lệch ≤ 5% so với phần thêm của A2 (QĐ-06) | `SA-11_a0plus.yaml` |
| A1 | Điều biến tự do (SFT-Free) | Bộ ước lượng + SFT theo không gian, **không** giám sát (lambda_map = 0) | `SA-12_a1.yaml` |
| A2 | Điều biến theo bản đồ nhòe (SFT-Map), mô hình đề xuất | Như A1 + giám sát L_map | `SA-13_a2.yaml` |
| A3 | Điều biến theo bản đồ nhòe + kênh (SFT-Map+Ch) | A2 + nhánh channel | `SA-14_a3.yaml` |

Thí nghiệm thăm dò sau khi chẩn đoán E (seed 1, chạy trên Modal L4, không thuộc họ phép so sánh chính):

| Mã | Nội dung | Config / script |
| --- | --- | --- |
| SA-32 | SFT-Map huấn luyện với bản đồ thật (QĐ-10) | `SA-32_a2_oracle_train.yaml` |
| SA-33 | SFT-Map với bộ ước lượng sửa `BlurMapEstimatorV2` (QĐ-10) | `SA-33_a2_estimator_v2.yaml` |
| SA-34 | Phép quét chất lượng bản đồ (QĐ-11) | `analysis/map_sweep.py` |
| SA-35 | SFT-Free train lại trên Modal (đối chứng nền tảng) | `SA-12_a1.yaml` |
| SA-36 | Nội suy trọng số Baseline (QĐ-12) | `analysis/weight_interp.py` |

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
│   ├── metrics/         PSNR/SSIM (LPIPS, RealBlur: có code, đã cắt), grouped stats [shared]
│   └── utils/           config (_base_ inheritance), seed                       [shared]
├── configs/  _base_/ + experiments/SA-xx_*.yaml
├── scripts/  download_data.sh, build_manifests.py, generate_testsets.py, make_testsets.py,
│             train.py, evaluate.py, eval_queue.py, compare.py, make_figures.py,
│             modal_train.py (train + đánh giá trên Modal), pack_results.py, verify_manifest.ps1
├── analysis/            eval_tables.py (bảng + thống kê), map_sweep.py, weight_interp.py, chẩn đoán E, audit
├── manifests/           sinh lại bằng build_manifests.py khi chạy (không commit, xem manifests/README.md)
├── notebooks/           03_results (khung báo cáo)
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

Chạy trên máy có GPU:

```bash
pip install -e ".[dev]"
make test                                   # unit test hạ tầng + module
bash scripts/download_data.sh               # GoPro, checkpoint NAFNet (xem docs/datasets.md)
python scripts/build_manifests.py --gopro data/GOPRO_Large
python scripts/generate_testsets.py --config configs/experiments/SA-13_a2.yaml
make train CFG=configs/experiments/SA-13_a2.yaml
make eval  CFG=configs/experiments/SA-13_a2.yaml CKPT=<final.pth>
```

Nơi chạy thực tế: 8 run lõi train trên Kaggle (T4, commit `faf4a06`). SA-14, SA-32, SA-33, SA-35 và toàn bộ phần đánh
giá chạy trên Modal (L4) bằng `scripts/modal_train.py`; script này từ chối chạy khi code chưa commit hoặc chưa push:

```bash
python -m modal run scripts/modal_train.py::prepare          # dữ liệu + checkpoint vào Volume sad-runs (secret kaggle)
python -m modal run scripts/modal_train.py::verify_data
python -m modal run scripts/modal_train.py::make_testsets    # test set chính thức, sinh MỘT lần, kiểm sha
python -m modal run --detach scripts/modal_train.py::train --config configs/experiments/SA-32_a2_oracle_train.yaml
python -m modal run --detach scripts/modal_train.py::evaluate_all --max-tier 4 --shards 2
python -m modal run --detach scripts/modal_train.py::map_sweep
python -m modal run --detach scripts/modal_train.py::weight_interp_all
python analysis/eval_tables.py --dir <CSV đã tải> --out <thư mục mới> --unblind
```

Tái lập checkpoint công khai (SA-00): PSNR uint8 32,870 dB trên GoPro test với TLC (công bố 32,87). Thư mục `results/`
không được commit; số liệu đã kiểm chứng ghi trong `docs/experiments.md`.

## Dataset và nguồn hình

- Link dataset, checkpoint, giấy phép: `docs/datasets.md`
- Hình cho slide và báo cáo (nguồn paper + hình tự tạo): `docs/figure_sources.md`

## Trích dẫn

Xem `CITATION.cff`. Dataset và model bên ngoài giữ giấy phép gốc (GoPro: CC BY 4.0; NAFNet: MIT).
