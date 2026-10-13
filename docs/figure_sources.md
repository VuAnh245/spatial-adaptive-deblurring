# Nguồn hình cho slide và báo cáo

## Quy tắc dùng hình

- **Ưu tiên hình tự tạo** (mã HT-): không vướng bản quyền, khớp đúng thí nghiệm của mình.
- Hình lấy từ paper hoặc trang dataset (mã HN-): chỉ dùng để giới thiệu related work hoặc dataset, ghi nguồn ngay dưới hình theo dạng `Nguồn: [n]` (n là số tài liệu tham khảo). Không chỉnh sửa rồi trình bày như hình của mình.
- Sơ đồ kiến trúc của backbone nên **vẽ lại** theo phong cách thống nhất của báo cáo, ghi "Vẽ lại theo [n]".
- Chụp hình từ PDF: mở link arXiv → PDF → phóng to 200% rồi chụp, hoặc dùng `pdftoppm -r 300` để xuất trang rồi cắt. Không chụp từ ảnh thu nhỏ.

## Hình nguồn ngoài, mã HN- (ghi nguồn khi dùng)

| Mã | Hình | Dùng cho | Link |
| --- | --- | --- | --- |
| HN-01 | Ảnh mẫu cặp mờ/nét GoPro; minh họa cách tạo blur bằng trung bình frame | Giới thiệu dataset | https://seungjunnah.github.io/Datasets/gopro ; paper: https://arxiv.org/abs/1612.02177 |
| HN-02 | Hệ chụp beam splitter và ảnh mẫu RealBlur | Giới thiệu dataset ảnh thật | http://cg.postech.ac.kr/research/RealBlur/ ; PDF: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700188.pdf |
| HN-03 | Cấu trúc NAFBlock (Hình 3(d) trong bài báo NAFNet) và kiến trúc U-Net của NAFNet | Backbone (nên vẽ lại) | https://arxiv.org/abs/2204.04676 |
| HN-04 | Spatial Feature Transform layer (SFT) | Related work: modulation theo không gian | https://arxiv.org/abs/1804.02815 |
| HN-05 | FiLM layer (affine theo kênh) | Related work: feature-wise modulation | https://arxiv.org/abs/1709.07871 |
| HN-06 | Module CBAM (channel + spatial attention) | Đối chứng SFT-Free (A1): attention không giám sát | https://arxiv.org/abs/1807.06521 |
| HN-07 | Squeeze-and-Excitation block | Related work: channel recalibration (nhánh channel của SFT-Map+Ch, A3) | https://arxiv.org/abs/1709.01507 |
| HN-08 | Kernel quỹ đạo chuyển động ngẫu nhiên | Bộ sinh blur | https://arxiv.org/abs/1711.07064 |
| HN-09 | Minh họa LPIPS so với PSNR/SSIM | Giải thích metric | https://arxiv.org/abs/1801.03924 |

Mở link arXiv `abs/...`, chọn "View PDF" để có bản đầy đủ độ phân giải. Số hình cụ thể đã kiểm:
Hình 3(d) trong bài báo NAFNet là NAFBlock. Với các paper khác, tìm hình theo nội dung ở cột "Hình".

## Hình tự tạo, mã HT- (ưu tiên dùng)

| Mã | Hình | Dùng cho | Tạo bằng |
| --- | --- | --- | --- |
| HT-01 | Kernel theo mức s (thẳng / cong), kèm bán kính RMS | Bộ sinh blur, định nghĩa nhãn s | `scripts/make_figures.py` → `docs/figures/fig_kernels.png` (có sẵn) |
| HT-02 | Mặt nạ M_r và bản đồ nhòe thật S cho R = 1..4 | Dữ liệu không đồng nhất | `make_figures.py` → `fig_masks.png` (có sẵn) |
| HT-03 | Ảnh nét, ảnh nhòe đồng nhất, ảnh nhòe không đồng nhất và bản đồ nhòe thật S | Bài toán, input/output | `make_figures.py --image <frame GoPro>` → `fig_blur_examples.png` |
| HT-04 | Histogram mức nhòe s_r, độ không đồng nhất H, số vùng nhòe R; ảnh mẫu kèm bản đồ nhòe thật S | EDA | `analysis/eda_synthetic.py` → `results/EDA/s_and_h.png`, `examples.png` |
| HT-05 | Sơ đồ kiến trúc SFT-Map (A2, vẽ lại) | Kiến trúc | Sơ đồ trong tài liệu đặc tả DL, mục 3; xuất lại bằng draw.io / TikZ |
| HT-06 | PSNR theo R, theo H, theo mức nhòe; ngoài phân phối | Kết quả chính | `notebooks/03_results.ipynb` từ CSV của `evaluate.py` |
| HT-07 | Scatter Ŝ vs S; Ŝ cạnh S trên cùng ảnh | Chẩn đoán SA-22 | `analysis/` |
| HT-08 | Trần bản đồ thật và phép thử xáo trộn (cột so sánh) | SA-20, SA-21 | `evaluate.py --mode oracle|shuffle` |
| HT-09 | Trực quan hệ số điều biến Δγ, β theo S ở từng stage | SA-30 | `analysis/` |
| HT-10 | Ŝ trên ảnh GoPro / RealBlur thật | Kiểm tra ngoài SA-28 | `analysis/` |
| HT-11 | Lưới 8 ảnh định tính cố định + crop phóng to; 5 ảnh lỗi | Định tính | `notebooks/03_results.ipynb` (ID trong `manifests/visual_set.txt`) |
| HT-12 | Bảng params, MACs, latency | Hiệu năng SA-31 | `scripts/benchmark_efficiency.py` |

Quy ước mã và tên: `docs/naming.md`. So sánh độ sáng và nhiễu GoPro với RealBlur chỉ làm nếu tải RealBlur-J (tùy chọn).

Cập nhật 2026-10-14: HN-02, HN-09, phần RealBlur của HT-10 và latency của HT-12 không dùng (RealBlur-J, LPIPS, SA-31 đã
cắt theo QĐ-10; HT-12 chỉ còn bảng số tham số). HT-11 dùng ID theo quy tắc trong `notebooks/03_results.ipynb`
(`visual_set.txt` không được tạo).
