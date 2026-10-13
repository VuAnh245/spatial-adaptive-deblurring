# Quy ước ký hiệu, mã và tên

Áp dụng cho báo cáo, slide, tài liệu trong `docs/`, tiêu đề hình và bảng xuất từ script.
Không áp dụng cho tên file config, thư mục `results/` và tên biến trong code (giữ nguyên để truy vết).
Bảng xuất ra (CSV, JSON) giữ cả mã lẫn nhãn, ví dụ `a_id: SA-13`, `a_label: SFT-Map`.

## 1. Chữ cái dành riêng

| Chữ | Chỉ dùng cho | Ví dụ |
| --- | --- | --- |
| A | Mã biến thể mô hình | A0, A0+, A1, A2, A3 |
| B | Ảnh nhòe | B = Σ_r M_r ⊙ (K_r ⊛ I) + n |
| M | Mặt nạ của vùng, M_r | Σ_r M_r(x, y) = 1 |

Không dùng A, B, M làm nhãn liệt kê, nhãn phương án, mã hình hay nhãn "phía thứ nhất / thứ hai" của phép so sánh.

## 2. Liệt kê trong câu

Dùng (1), (2), (3). Không dùng chữ cái trong ngoặc làm nhãn liệt kê.
Ví dụ: "giao thức có ba điều kiện: (1) ..., (2) ..., (3) ...".
Ngoại lệ duy nhất: trích nguyên nhãn hình của bài báo khác, ví dụ "Hình 3(d) trong bài báo NAFNet".

## 3. Tiền tố mã

| Tiền tố | Dùng cho | Định dạng | Ví dụ |
| --- | --- | --- | --- |
| SA- | Thí nghiệm | 2 chữ số | SA-13 |
| QĐ- | Quyết định thiết kế. Mỗi phương án gọi bằng tên chữ, không đánh mã | 2 chữ số | QĐ-01, phương án "TLC chính, global phụ" |
| HN- | Hình nguồn ngoài (bài báo, trang dataset), phải ghi `Nguồn: [n]` | 2 chữ số | HN-03 |
| HT- | Hình tự tạo | 2 chữ số | HT-04 |
| CH- | Câu hỏi kiểm tra | 2 chữ số | CH-01 |

## 4. Tên đọc được (bảng chính thức)

### Mô hình

| Mã | Tên đầy đủ | Nhãn ngắn (hình, bảng) |
| --- | --- | --- |
| A0 | NAFNet gốc | Baseline |
| A0+ | NAFNet tăng tham số | Baseline-L |
| A1 | Điều biến tự do | SFT-Free |
| A2 | Điều biến theo bản đồ nhòe, mô hình đề xuất | SFT-Map |
| A3 | Điều biến theo bản đồ nhòe + kênh | SFT-Map+Ch |

### Phép so sánh

| So sánh | Tên |
| --- | --- |
| A1 so với A0 | Phép thử điều biến |
| A2 so với A1 | Phép thử giám sát (chính) |
| A2 so với A0+ | Phép thử tham số |
| A3 so với A2 | Phép thử nhánh kênh |
| SA-20 | Trần bản đồ thật |
| SA-21 | Phép thử xáo trộn |

### Thí nghiệm

| Mã | Tên |
| --- | --- |
| SA-00 | Tái lập NAFNet gốc trên GoPro |
| SA-10 | Huấn luyện Baseline |
| SA-11 | Huấn luyện Baseline-L |
| SA-12 | Huấn luyện SFT-Free |
| SA-13 | Huấn luyện SFT-Map |
| SA-14 | Huấn luyện SFT-Map+Ch |
| SA-15 | SFT-Map chỉ ở decoder (thay "chỉ ở middle" ngày 2026-10-06, xem `docs/architecture.md`) |
| SA-16 | SFT-Map ở encoder + middle |
| SA-17 | SFT-Map cắt gradient vào bộ ước lượng |
| SA-18 | SFT-Map + loss FFT |
| SA-19 | SFT-Map + loss perceptual |
| SA-20 | Trần bản đồ thật |
| SA-21 | Phép thử xáo trộn |
| SA-22 | Chất lượng bản đồ ước lượng |
| SA-23 | PSNR vùng nhòe nặng |
| SA-24 | Phân tích theo số vùng và độ không đồng nhất |
| SA-25 | Phân tích theo mức nhòe và ngoài phân phối |
| SA-26 | Độ bền với nhiễu |
| SA-27 | Kernel thẳng và kernel cong |
| SA-28 | Ảnh nhòe thật (GoPro, RealBlur-J) |
| SA-29 | So với phương pháp cổ điển (Wiener, Richardson-Lucy) |
| SA-30 | Trực quan hệ số điều biến |
| SA-31 | Hiệu năng tính toán |
| SA-32 | SFT-Map huấn luyện với bản đồ thật (QĐ-10, thăm dò) |
| SA-33 | SFT-Map với bộ ước lượng sửa (QĐ-10, thăm dò) |
| SA-34 | Phép quét chất lượng bản đồ (QĐ-11, thăm dò) |
| SA-35 | SFT-Free train lại trên Modal (đối chứng nền tảng) |
| SA-36 | Nội suy trọng số Baseline (QĐ-12, thăm dò) |

### Tập dữ liệu

| Tên trong code | Tên đọc được |
| --- | --- |
| `val` | Tập kiểm định |
| `test_in` | Tổng hợp trong phân phối |
| `test_ood` | Tổng hợp ngoài phân phối |
| `test_straight` | Kernel thẳng |
| `test_curved` | Kernel cong |
| hậu tố `_noise2x`, `_noise4x` | (nhiễu ×2), (nhiễu ×4) |
| `gopro_test` | GoPro thật |
| `realblur_j_test` | RealBlur-J |

### Ký hiệu

| Ký hiệu | Nghĩa |
| --- | --- |
| B | Ảnh nhòe |
| I | Ảnh nét |
| Î | Ảnh khôi phục |
| n | Nhiễu |
| r | Chỉ số vùng |
| R | Số vùng nhòe |
| K_r | Kernel của vùng r |
| M_r | Mặt nạ của vùng r |
| s_r | Mức nhòe của vùng r (một số) |
| S | Bản đồ nhòe thật (một ảnh) |
| Ŝ | Bản đồ nhòe ước lượng |
| H | Độ không đồng nhất, H = std(S) |
| E | Bộ ước lượng bản đồ nhòe |
| SFT head | Đầu điều biến không gian |
| Nhánh channel | Nhánh điều biến kênh |
| Δγ, β | Hệ số nhân, hệ số cộng |
| L_map | Loss bản đồ |
| lambda_map | Trọng số loss bản đồ |

### Thước đo và giao thức

| Tên trong code | Tên đọc được |
| --- | --- |
| TLC (`--pool tlc`) | Suy luận cục bộ |
| global (`--pool global`) | Suy luận toàn cục |
| `psnr_local_high` | PSNR vùng nhòe nặng |
| PSNR trên dải \|∇S\| lớn (`psnr_transition`, chưa cài) | PSNR vùng chuyển tiếp |
| `map_mae` | Sai số bản đồ |
| `map_spearman` | Tương quan bản đồ |

Đơn vị: triệu tham số, GMACs, ms.

Nhòe đồng nhất và nhòe không đồng nhất viết bằng chữ, không dùng mã.
Phương pháp của đồ án Computer Vision gọi là "phương pháp của đồ án CV", không dùng mã của repo đó.

## 5. Trong code

Mọi nhãn hiển thị nằm ở `src/spatial_deblur/utils/labels.py`: `model_label`, `model_name`, `comparison_label`,
`experiment_label`, `split_label`, `metric_label`, `vn_num`. Sửa tên thì sửa đồng thời file đó và file này.
Số thập phân trong hình dùng dấu phẩy (0,45). Kết luận của `scripts/compare.py` trả về mã
(`first_better`, `second_better`, `no_difference`) kèm nhãn, ví dụ "SFT-Map tốt hơn".

## 6. Quyết định thiết kế và câu hỏi kiểm tra

| Mã | Nội dung | Phương án | Trạng thái |
| --- | --- | --- | --- |
| QĐ-01 | Giao thức đánh giá | "TLC chính, global phụ" | Đã chốt 2026-10-05 (`docs/experiments.md`) |
| QĐ-02 | Thang vùng nhòe train và test | Chọn "Ô vuông + đa thang" (loại "Giữ lưới cũ") | Đã chốt 2026-10-05, trước Bước 5 (`docs/experiments.md`). Số EDA chỉ để xác nhận và ghi vào báo cáo |
| QĐ-03 | Ngưỡng nhóm H | "Theo phân vị 33/67" trên H của các mẫu R ≥ 2 (500 mẫu EDA, làm tròn 2 chữ số); nhóm "H bằng 0" gồm các ảnh R = 1 | Đã ghi số 2026-10-13 (sad-checks V3, trước đánh giá test tổng hợp): `data.h_bins: [0.11, 0.18]` |
| QĐ-04 | Công thức huấn luyện và số iteration chung | "Một ITERS cho mọi mô hình, gói vừa một phiên" (batch 8; lr hạ từ 2e-4 xuống 1e-4 sau khi pilot lần hai phân kỳ) | Đã chốt: lr 1e-4, FP32, ITERS = 30.000, chốt chặn 11,5 giờ (`docs/experiments.md`) |
| QĐ-05 | Đơn vị gộp khi kiểm định | "Gộp theo video" (10 nhóm trên GoPro test), báo kèm tỉ lệ thắng theo ảnh | Đã chốt 2026-10-06 |
| QĐ-06 | Quy tắc khớp tham số của Baseline-L | "Ít block nhất, lệch ≤ 5% phần thêm" (2 block ở dec1, 17.349.731 tham số); loại "Khớp tuyệt đối", "Một block ở dec0" | Đã chốt 2026-10-06 |
| QĐ-07 | Độ chính xác số khi train | "FP32 toàn phần" (loại "AMP + dò lỗi"); nếu hết bộ nhớ thì tích lũy gradient 4 × 2 | Đã chốt 2026-10-06 |
| QĐ-08 | Ổn định SFT-Map+Ch (SA-14) | "Nhánh channel đọc đặc trưng chuẩn hóa + dừng trước bước cập nhật hỏng"; loại "Giữ nguyên A3 gốc", "Hạ lr hoặc đổi batch riêng cho A3" | Đã chốt 2026-10-13 (`docs/experiments.md`) |
| QĐ-09 | Chỉ tiêu chính, so sánh nhiều lần, gộp seed | PSNR `test_in` là chỉ tiêu chính; Holm trên họ phép so sánh lõi; Phép thử giám sát gộp 3 seed | Đã chốt 2026-10-13, sau GoPro seed 1, trước test tổng hợp |
| QĐ-10 | Thí nghiệm thăm dò sau chẩn đoán E | "Trần bản đồ thật + bộ ước lượng sửa" (SA-32, SA-33), cổng C1-C4, tiêu chí khai báo trước | Đã chốt 2026-10-13 (`docs/experiments.md`) |
| QĐ-11 | Bản đồ cần tốt đến mức nào | Phép quét SA-34 trên SA-32; đối chứng cùng nền tảng SA-35; quy tắc đọc xuyên nền tảng 0,02 dB | Đã chốt 2026-10-13 (`docs/experiments.md`) |
| QĐ-12 | Đánh đổi miền sau fine-tune | Chọn "Nội suy trọng số Baseline" (SA-36, alpha 0,25/0,5/0,75, chỉ đánh giá); loại "Đóng băng thân mạng + chỉ train SFT" | Đã chốt 2026-10-14 (`docs/experiments.md`) |
| CH-01 | Gradient của SFT head khi lớp cuối khởi tạo 0 | | Nội dung đúng, chờ câu trả lời bằng lời của sinh viên qua CH-03 |
| CH-02 | Vì sao TLC chỉ có tác dụng khi ảnh test lớn hơn patch train | | Đạt |
| CH-03 | Nếu `sft.*.conv2` của SFT-Free vẫn gần 0 sau 10k iter thì đường val so với Baseline thế nào, và có kết luận được gì | | Chờ trả lời |
