# Thí nghiệm (study DL)

## SA-00 Tái lập NAFNet gốc trên GoPro (cổng trước khi train) - ĐẠT

NAFNet-GoPro-width32 công khai, GoPro test 1.111 cặp, ảnh đầy đủ 1280×720, Kaggle T4, torch 2.11.0+cu128,
commit `827d60a`. Script tương đương: `analysis/reproduce_nafnet_gopro.py`.

| Chỉ số | Kết quả | Công bố |
| --- | --- | --- |
| PSNR, uint8, suy luận cục bộ (TLC, NAFNetLocal) | **32,870 dB** | 32,87 dB |
| PSNR, float, suy luận cục bộ | 32,882 dB | |
| PSNR, uint8, suy luận toàn cục | 32,294 dB | |
| PSNR ảnh nhòe đầu vào | 25,640 dB | | 
| SSIM (tập con 223 ảnh, skimage) | 0,937 | 0,9606 (implementation BasicSR, không so trực tiếp) |

TLC tăng PSNR ở cả 11 chuỗi (+0,02 đến +1,14 dB). Số ảnh mỗi chuỗi không đều (77 đến 134).

## QĐ-01 Giao thức đánh giá (khai báo 2026-10-05, trước khi train hay đánh giá bất kỳ biến thể nào)

Phương án đã chọn: "TLC chính, global phụ". Quy ước mã và tên: `docs/naming.md`.

- **Chính:** mọi biến thể (A0, A0+, A1, A2, A3) được đánh giá sau `convert_tlc(model, train_size=256)`.
  Mọi bảng, mọi kiểm định và mọi kết luận dùng số TLC.
- **Phụ:** pooling toàn cục, chỉ báo cho A0 và A2, đặt ở phụ lục, để cho thấy kết luận có phụ thuộc
  giao thức hay không.
- TLC chỉ đổi kết quả khi ảnh vượt 384 px (1,5 × patch train). Test set tổng hợp và GoPro test ở độ phân giải
  đầy đủ nên chịu ảnh hưởng. Nếu đổi kích thước patch train, `train_size` của TLC phải đổi theo.
- Bộ ước lượng E và SFT head chỉ gồm conv, TLC không chạm tới. Nhánh channel của A3 dùng
  `nn.AdaptiveAvgPool2d` nên được TLC chuyển như SCA của NAFNet (test: `test_tlc_converts_every_global_pool...`).
- Theo dõi val trong lúc train không dùng để chọn model (chọn checkpoint cuối), nên không phụ thuộc giao thức.

## QĐ-02 Thang vùng nhòe train và test (chốt 2026-10-05, trước Bước 5)

Phương án đã chọn: "Ô vuông + đa thang". Phương án bị loại: "Giữ lưới cũ".

Lý do: mức lệch đã xác định được từ code, không cần chờ số liệu. Với lưới cũ 6 × 6 kéo giãn lên ảnh:
(1) patch train 296 × 296 px có ô khoảng 49 × 49 px;
(2) khung test 1320 × 760 px có ô khoảng 220 × 127 px, vừa lớn hơn vừa không vuông.
Số EDA chỉ dùng để xác nhận và ghi vào báo cáo, không dùng để chọn phương án.

Cài đặt (`configs/_base_/data.yaml`):
(1) `mask_grid_train: [3, 4, 5, 6]`: mỗi patch train rút ngẫu nhiên lưới g × g, ô vuông từ khoảng 99 đến 49 px;
(2) `mask_grid_test: [6, 10]`: test set và val dùng lưới 6 hàng × 10 cột trên khung đầy đủ, ô khoảng 127 × 132 px;
(3) `soft_masks(..., grid=(hàng, cột))`, test `test_masks_non_square_grid_sum_to_one`.

Lệch còn lại: ô test vẫn lớn hơn ô train khoảng 1,3 đến 2,7 lần, vì patch train chỉ rộng 256 px. Ghi vào mục Giới hạn,
kèm số đo cạnh mảng vùng của `analysis/eda_synthetic.py`.

## QĐ-03 Ngưỡng nhóm H (quy tắc chốt 2026-10-05, số ghi vào config trước Bước 5)

Phương án "Theo phân vị 33/67": ngưỡng là phân vị 33% và 67% của H trên các mẫu R ≥ 2 trong 500 mẫu EDA,
làm tròn 2 chữ số. Nhóm "H bằng 0" gồm các ảnh R = 1.

## QĐ-04 Công thức huấn luyện và số iteration chung (quy tắc chốt 2026-10-05, số ITERS ghi sau pilot)

Phương án "Một ITERS cho mọi mô hình, gói vừa một phiên":
(1) công thức giống nhau cho Baseline, Baseline-L, SFT-Free, SFT-Map, SFT-Map+Ch: AdamW lr 2e-4, betas (0,9; 0,9),
    weight decay 1e-3, batch 8, patch 256, warmup 1000 iter rồi cosine về 1e-7, AMP FP16, gradient clip 1,0,
    2 worker DataLoader mỗi tiến trình (hai run chạy chung một phiên T4 × 2);
(2) ITERS tính theo mô hình chậm nhất đo được ở pilot (pilot chạy SFT-Map và Baseline-L song song;
    SFT-Free có cùng kiến trúc với SFT-Map, Baseline nhẹ nhất), sao cho một phiên Kaggle 12 giờ chứa đủ
    khởi động, đánh giá val (mỗi 2500 iter, kèm lần ở bước 0), checkpoint (mỗi 2500 iter) và chừa khoảng 1 giờ dự phòng.
    Tính bằng `analysis/pilot_report.py` (khối "budget");
(3) lr: pilot chạy 1200 iter (qua hết warmup 1000, tức đã ở lr đỉnh 2e-4), val ở bước 0, 600, 1200;
    nếu val PSNR ở bước 1200 tụt quá 0,3 dB so với bước 0 thì hạ lr xuống trong khoảng 1e-4 đến 2e-4
    và chạy lại pilot trước khi khóa; không tune lại sau khi đã train bất kỳ biến thể nào;
(4) chặn thời gian: notebook dừng mọi tiến trình train sau 11 giờ (SIGTERM, lưu `last.pth`), resume ở phiên sau.

ITERS = (điền sau pilot) | VAL_EVERY = 2500 | CKPT_EVERY = 2500

## Log chẩn đoán khi train

`diag.jsonl` (mỗi `diag_every` = 500 iter): ‖W‖ lớp cuối của SFT head theo stage (`w_sft_<stage>`),
trung bình |Δγ_s| và |β_s| theo stage (`dgamma_<stage>`, `beta_<stage>`), kèm `charbonnier`, `l_map`, `loss`.
`curves.csv` (mỗi 100 iter) ghi `l_map` riêng cho mọi biến thể có bộ ước lượng, kể cả SFT-Free (lambda_map = 0, chỉ để theo dõi).
Cảnh báo cần dừng phân tích: `w_sft_*` của SFT-Free hoặc SFT-Map đứng gần 0 sau 10k iter.

| ID | Tên đọc được | Biến thể | Seed | Mức | Config |
| --- | --- | --- | --- | --- | --- |
| SA-10 | Huấn luyện Baseline | A0 | 1 | Lõi | SA-10_a0.yaml |
| SA-11 | Huấn luyện Baseline-L | A0+ | 1 (3 ở bản đầy đủ) | Lõi | SA-11_a0plus.yaml |
| SA-12 | Huấn luyện SFT-Free | A1 | 3 | Lõi | SA-12_a1.yaml |
| SA-13 | Huấn luyện SFT-Map (mô hình đề xuất) | A2 | 3 | Lõi | SA-13_a2.yaml |
| SA-14 | Huấn luyện SFT-Map+Ch | A3 | 1 | Lõi | SA-14_a3.yaml |
| SA-15 | SFT-Map chỉ ở middle | A2 | 1 | Tùy chọn | SA-15_a2_mid_only.yaml |
| SA-16 | SFT-Map ở encoder + middle | A2 | 1 | Tùy chọn | SA-16_a2_enc_mid.yaml |
| SA-17 | SFT-Map cắt gradient vào bộ ước lượng | A2 | 1 | Tùy chọn | SA-17_a2_detach.yaml |
| SA-18 | SFT-Map + loss FFT | A2 | 1 | Tùy chọn | SA-18_a2_fft.yaml |
| SA-19 | SFT-Map + loss perceptual | A2 | 1 | Tùy chọn | SA-19_a2_perceptual.yaml |

Phân tích không train (chạy bằng `scripts/evaluate.py` / `analysis/`):

| ID | Tên | Lệnh |
| --- | --- | --- |
| SA-20 | Trần bản đồ thật (S thay Ŝ) | `evaluate.py --mode oracle` |
| SA-21 | Phép thử xáo trộn (Ŝ của ảnh khác) | `evaluate.py --mode shuffle` |
| SA-22 | Chất lượng bản đồ ước lượng (sai số, tương quan) | cột `map_*` trong CSV |
| SA-23 | PSNR vùng nhòe nặng | cột `psnr_local_high` |
| SA-24 | Phân tích theo số vùng và độ không đồng nhất | nhóm theo cột `regions`, `H` |
| SA-25 | Phân tích theo mức nhòe và ngoài phân phối | `--split test_ood` |
| SA-26 | Độ bền với nhiễu (×2, ×4) | `generate_testsets.py --noise-mult 2` |
| SA-27 | Kernel thẳng và kernel cong | `--split test_straight`, `test_curved` |
| SA-28 | Ảnh nhòe thật (GoPro, RealBlur-J) | `evaluate.py --split gopro_test` |
| SA-29 | So với phương pháp cổ điển (Wiener, Richardson-Lucy; R = 1) | `evaluate.py --classical` |
| SA-30 | Trực quan hệ số điều biến | `analysis/` |
| SA-31 | Hiệu năng tính toán | `benchmark_efficiency.py` |

Mọi kết quả lưu ở `results/<ID>/` với `config.yaml` (có `git_commit`), `metrics.json`, CSV theo ảnh.
