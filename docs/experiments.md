# Thí nghiệm (study DL)

## SA-00 Tái lập checkpoint công khai (cổng trước khi train) - ĐẠT

NAFNet-GoPro-width32 công khai, GoPro test 1.111 cặp, ảnh đầy đủ 1280×720, Kaggle T4, torch 2.11.0+cu128,
commit `827d60a`. Script tương đương: `analysis/reproduce_nafnet_gopro.py`.

| Chỉ số | Kết quả | Công bố |
| --- | --- | --- |
| PSNR, uint8, TLC (NAFNetLocal) | **32,870 dB** | 32,87 dB |
| PSNR, float, TLC | 32,882 dB | |
| PSNR, uint8, pooling toàn cục (không TLC) | 32,294 dB | |
| PSNR ảnh nhòe đầu vào | 25,640 dB | |
| SSIM (tập con 223 ảnh, skimage) | 0,937 | 0,9606 (implementation BasicSR, không so trực tiếp) |

TLC tăng PSNR ở cả 11 chuỗi (+0,02 đến +1,14 dB). Số ảnh mỗi chuỗi không đều (77 đến 134).

| ID | Biến thể | Seed | Mức | Config |
| --- | --- | --- | --- | --- |
| SA-10 | A0 | 1 | Lõi | SA-10_a0.yaml |
| SA-11 | A0+ | 1 (3 ở bản đầy đủ) | Lõi | SA-11_a0plus.yaml |
| SA-12 | A1 | 3 | Lõi | SA-12_a1.yaml |
| SA-13 | A2 (model chính) | 3 | Lõi | SA-13_a2.yaml |
| SA-14 | A3 | 1 | Lõi | SA-14_a3.yaml |
| SA-15 | A2, module chỉ ở middle | 1 | Tùy chọn | SA-15_a2_mid_only.yaml |
| SA-16 | A2, encoder + middle | 1 | Tùy chọn | SA-16_a2_enc_mid.yaml |
| SA-17 | A2, cắt gradient vào bộ ước lượng | 1 | Tùy chọn | SA-17_a2_detach.yaml |
| SA-18 | A2 + FFT loss | 1 | Tùy chọn | SA-18_a2_fft.yaml |
| SA-19 | A2 + perceptual loss | 1 | Tùy chọn | SA-19_a2_perceptual.yaml |

Phân tích không train (chạy bằng `scripts/evaluate.py` / `analysis/`):

| ID | Phân tích | Lệnh |
| --- | --- | --- |
| SA-20 | Oracle (S thay Ŝ) | `evaluate.py --mode oracle` |
| SA-21 | Xáo trộn Ŝ trong batch | `evaluate.py --mode shuffle` |
| SA-22 | MAE, RMSE, Spearman của Ŝ | cột `map_*` trong CSV |
| SA-23 | PSNR cục bộ vùng S cao | cột `psnr_local_high` |
| SA-24 | Theo R và H | nhóm theo cột `regions`, `H` |
| SA-25 | Theo mức nhòe, ngoài phân phối | `--split test_ood` |
| SA-26 | Nhiễu khi test gấp 2, 4 | `generate_testsets.py --noise-mult 2` |
| SA-27 | Kernel thẳng / cong | `--split test_straight`, `test_curved` |
| SA-28 | GoPro, RealBlur-J zero-shot | `evaluate.py --split gopro_test` |
| SA-29 | Wiener, Richardson-Lucy (R = 1) | `evaluate.py --classical` |
| SA-30 | Heatmap Δγ, β theo S | `analysis/` |
| SA-31 | Hiệu năng | `benchmark_efficiency.py` |

Mọi kết quả lưu ở `results/<ID>/` với `config.yaml` (có `git_commit`), `metrics.json`, CSV theo ảnh.
