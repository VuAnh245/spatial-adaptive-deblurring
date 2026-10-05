# Nguồn hình cho slide và báo cáo

## Quy tắc dùng hình

- **Ưu tiên hình tự tạo** (mục B): không vướng bản quyền, khớp đúng thí nghiệm của mình.
- Hình lấy từ paper hoặc trang dataset (mục A): chỉ dùng để giới thiệu related work hoặc dataset, ghi nguồn ngay dưới hình theo dạng `Nguồn: [n]` (n là số tài liệu tham khảo). Không chỉnh sửa rồi trình bày như hình của mình.
- Sơ đồ kiến trúc của backbone nên **vẽ lại** theo phong cách thống nhất của báo cáo, ghi "Vẽ lại theo [n]".
- Chụp hình từ PDF: mở link arXiv → PDF → phóng to 200% rồi chụp, hoặc dùng `pdftoppm -r 300` để xuất trang rồi cắt. Không chụp từ ảnh thu nhỏ.

## A. Hình từ paper và trang dataset (ghi nguồn khi dùng)

| # | Hình | Dùng cho | Link |
| --- | --- | --- | --- |
| A1 | Ảnh mẫu cặp mờ/nét GoPro; minh họa cách tạo blur bằng trung bình frame | Giới thiệu dataset | https://seungjunnah.github.io/Datasets/gopro ; paper: https://arxiv.org/abs/1612.02177 |
| A2 | Hệ chụp beam splitter và ảnh mẫu RealBlur | Giới thiệu dataset ảnh thật | http://cg.postech.ac.kr/research/RealBlur/ ; PDF: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700188.pdf |
| A3 | Cấu trúc NAFBlock (Fig. 3(d)) và kiến trúc U-Net của NAFNet | Backbone (nên vẽ lại) | https://arxiv.org/abs/2204.04676 |
| A4 | Spatial Feature Transform layer (SFT) | Related work: modulation theo không gian | https://arxiv.org/abs/1804.02815 |
| A5 | FiLM layer (affine theo kênh) | Related work: feature-wise modulation | https://arxiv.org/abs/1709.07871 |
| A6 | Module CBAM (channel + spatial attention) | Đối chứng A1: attention không giám sát | https://arxiv.org/abs/1807.06521 |
| A7 | Squeeze-and-Excitation block | Related work: channel recalibration (nhánh channel A3) | https://arxiv.org/abs/1709.01507 |
| A8 | Kernel quỹ đạo chuyển động ngẫu nhiên | Bộ sinh blur | https://arxiv.org/abs/1711.07064 |
| A9 | Minh họa LPIPS so với PSNR/SSIM | Giải thích metric | https://arxiv.org/abs/1801.03924 |

Mở link arXiv `abs/...`, chọn "View PDF" để có bản đầy đủ độ phân giải. Số hình cụ thể đã kiểm:
NAFNet Fig. 3(d) là NAFBlock. Với các paper khác, tìm hình theo nội dung ở cột "Hình".

## B. Hình tự tạo (ưu tiên dùng)

| # | Hình | Dùng cho | Tạo bằng |
| --- | --- | --- | --- |
| B1 | Kernel theo mức s (thẳng / cong), kèm bán kính RMS | Bộ sinh blur, định nghĩa nhãn s | `scripts/make_figures.py` → `docs/figures/fig_kernels.png` (có sẵn) |
| B2 | Mặt nạ mềm và bản đồ S cho R = 1..4 | Dữ liệu không đồng nhất | `make_figures.py` → `fig_masks.png` (có sẵn) |
| B3 | Ảnh nét / D1 / D2 + bản đồ S | Bài toán, input/output | `make_figures.py --image <frame GoPro>` → `fig_blur_examples.png` |
| B4 | Histogram S, H, R; độ sáng và nhiễu GoPro vs RealBlur | EDA | `notebooks/01_eda.ipynb` |
| B5 | Sơ đồ kiến trúc A2 (vẽ lại) | Kiến trúc | Sơ đồ trong tài liệu đặc tả DL, mục 3; xuất lại bằng draw.io / TikZ |
| B6 | PSNR theo R, theo H, theo mức nhòe; ngoài phân phối | Kết quả chính | `notebooks/03_results.ipynb` từ CSV của `evaluate.py` |
| B7 | Scatter Ŝ vs S; Ŝ cạnh S trên cùng ảnh | Chẩn đoán SA-22 | `analysis/` |
| B8 | Oracle và xáo trộn Ŝ (cột so sánh) | SA-20, SA-21 | `evaluate.py --mode oracle|shuffle` |
| B9 | Heatmap Δγ, β theo S ở từng stage | SA-30 | `analysis/` |
| B10 | Ŝ trên ảnh GoPro / RealBlur thật | Kiểm tra ngoài SA-28 | `analysis/` |
| B11 | Lưới 8 ảnh định tính cố định + crop phóng to; 5 ảnh lỗi | Định tính | `notebooks/03_results.ipynb` (ID trong `manifests/visual_set.txt`) |
| B12 | Bảng params, MACs, latency | Hiệu năng SA-31 | `scripts/benchmark_efficiency.py` |
