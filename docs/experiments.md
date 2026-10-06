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
(1) công thức giống nhau cho Baseline, Baseline-L, SFT-Free, SFT-Map, SFT-Map+Ch: AdamW lr 1e-4 (ban đầu 2e-4, hạ theo (3)), betas (0,9; 0,9),
    weight decay 1e-3, batch 8, patch 256, warmup 1000 iter rồi cosine về 1e-7, FP32 (QĐ-07; ban đầu AMP FP16), gradient clip 1,0,
    2 worker DataLoader mỗi tiến trình (hai run chạy chung một phiên T4 × 2);
(2) ITERS tính theo mô hình chậm nhất đo được ở pilot (pilot chạy SFT-Map và Baseline-L song song;
    SFT-Free có cùng kiến trúc với SFT-Map, Baseline nhẹ nhất), sao cho một phiên Kaggle 12 giờ chứa đủ
    khởi động, đánh giá val (mỗi 2500 iter, kèm lần ở bước 0), checkpoint (mỗi 2500 iter) và chừa khoảng 1 giờ dự phòng.
    Tính bằng `analysis/pilot_report.py` (khối "budget");
(3) lr: kiểm tra trên pilot lần một (SFT-Map, 1200 iter, qua hết warmup 1000, tức đã ở lr đỉnh 2e-4, val ở bước
    0, 600, 1200); nếu val PSNR ở bước 1200 tụt quá 0,3 dB so với bước 0 thì hạ lr xuống trong khoảng 1e-4 đến 2e-4
    và chạy lại pilot trước khi khóa; không tune lại sau khi đã train bất kỳ biến thể nào.
    Pilot lần hai (300 iter, SFT-Map và Baseline-L theo QĐ-06 chạy song song, val mới) chỉ đo tốc độ, bộ nhớ,
    val mới và Spearman; ITERS tính từ pilot lần hai;
(4) chặn thời gian: notebook dừng mọi tiến trình train sau 11 giờ (SIGTERM, lưu `last.pth`), resume ở phiên sau.

Diễn biến (2026-10-06):
(1) pilot lần một (SFT-Map chạy một mình, lr 2e-4): dừng ở bước 600 do phiên bị tắt, val 30,96 → 33,32 dB;
(2) pilot lần hai (SFT-Map + Baseline-L theo QĐ-06, lr 2e-4, 1200 iter, val mới): tới bước 600 val tăng
    (34,43 và 33,46 dB) nhưng ở lr đỉnh thì phân kỳ: SFT-Map còn 19,25 dB ở bước 1200, Baseline-L ra NaN.
    Tốc độ 0,959 it/s khi hai run chạy song song (CPU là nút thắt, data_frac 0,40-0,45), ITERS ước tính 36.000;
(3) áp dụng quy tắc (3): **lr = 1e-4**. Kèm hai sửa chống mất ổn định số học không đổi công thức: LayerNorm2d tính
    thống kê bằng FP32 trong autocast; chốt chặn dừng run sau hơn 50 bước liên tiếp có loss không hữu hạn
    (không ghi đè `last.pth` tốt cuối cùng). `curves.csv` ghi thêm `grad_norm`, `amp_scale`;
(4) pilot lần ba: lr 1e-4, 1500 iter, val ở 0 / 750 / 1500, SFT-Map và Baseline-L song song. ITERS khóa sau pilot lần ba;
(5) chốt chặn sàn theo val (`val_floor_drop: 1.0`): nếu một lần val có PSNR thấp hơn val ở bước 0 (tức checkpoint công khai
    ban đầu) quá 1 dB, hoặc không hữu hạn, run dừng ngay, ghi `aborted_val_drop`, không ghi đè `last.pth`, `train.py` thoát mã 4.
    Pilot lần hai cho thấy cần chốt này: SFT-Map sụp từ 34,43 xuống 19,25 dB mà loss vẫn hữu hạn, chốt chặn NaN không bắt được.
    **Đây là mức sàn chống sụp, không phải chọn checkpoint theo val**: mô hình đem đánh giá luôn là checkpoint cuối của run
    chạy hết ITERS; run bị dừng không được thay bằng checkpoint tốt nhất mà phải xử lý theo (3) (hạ lr, chạy lại pilot) và áp
    dụng chung cho mọi biến thể. Checkpoint ở bước có val chỉ được ghi sau khi val đó qua sàn (ckpt_every là bội của val_every),
    và bước cuối luôn được val trước khi ghi.

(6) pilot lần ba (lr 1e-4, AMP): FAIL. GradScaler sụp (scale về gần 0 sau một chuỗi gradient inf), gradient bị làm tròn
    về 0 và cả hai run ngừng học từ khoảng bước 600 (SFT-Map) và 200 (Baseline-L): val bước 750 → 1500 chỉ đổi
    +0,0007 / +0,0004 dB. Phép kiểm tra lr ở mức đỉnh vì vậy chưa thực hiện được. Dẫn tới QĐ-07;
(7) pilot lần bốn: FP32 (QĐ-07), lr 1e-4, 1500 iter, val ở 0 / 750 / 1500, SFT-Map và Baseline-L song song;
    đo lại bộ nhớ đỉnh, it/s và ITERS.

ITERS = (khóa sau pilot lần bốn) | VAL_EVERY = 2500 | CKPT_EVERY = 2500

## QĐ-07 Độ chính xác số khi train (chốt 2026-10-06)

Phương án "FP32 toàn phần" (`amp: false` cho mọi biến thể); loại phương án "AMP + dò lỗi". Lý do:
(1) ba pilot liên tiếp gặp sự cố số học dưới AMP: phân kỳ và NaN ở pilot lần hai, GradScaler sụp ở pilot lần ba;
    sửa LayerNorm2d sang FP32 không chặn được gốc rễ, và op sinh inf trong backward chưa xác định được;
(2) repo NAFNet gốc train bằng FP32;
(3) nút thắt là CPU sinh dữ liệu (GPU chờ dữ liệu khoảng 45% thời gian), nên một phần chi phí FP32 được che đi.
Nếu FP32 hết bộ nhớ ở batch 8: tích lũy gradient, micro-batch 4 × 2 (`grad_accum: 2`), batch hiệu dụng vẫn là 8,
ghi rõ trong config của run. ITERS đo lại ở pilot lần bốn.

Chốt chặn dừng run (ghi trong `events.jsonl`, không ghi đè `last.pth`), mã thoát của `train.py`:
(1) loss không hữu hạn hơn 50 bước liên tiếp: `aborted_nonfinite`, mã 3;
(2) val thấp hơn val ở bước 0 quá 1 dB hoặc không hữu hạn: `aborted_val_drop`, mã 4;
(3) `grad_norm` bằng 0 ở 3 lần log liên tiếp: `aborted_stalled`, mã 5 (không phụ thuộc AMP);
(4) GradScaler bật mà scale < 1: `aborted_amp_scale`, mã 6 (không hoạt động khi FP32).
Val đóng băng sau warmup chỉ được kiểm trong `pilot_report.py`, không làm dừng run, để tránh dừng nhầm ở cuối cosine khi lr rất nhỏ.

## QĐ-05 Đơn vị gộp khi kiểm định (chốt 2026-10-06, trước mọi đánh giá)

Phương án "Gộp theo video": điểm từng ảnh được lấy trung bình theo **video nguồn** (`GOPRxxxx_yy`, gộp các đoạn
`_NN` của cùng một lần quay) rồi mới chạy Wilcoxon, bootstrap CI và tỉ lệ thắng theo nhóm. GoPro test có 11 chuỗi
nhưng chỉ 10 video (`GOPR0384_11_00` và `GOPR0384_11_05` cùng video), nên có **10 nhóm**. Báo kèm tỉ lệ thắng theo ảnh
(`win_rate_images`). Cài ở `scripts/compare.py --group-by video` (mặc định).

## QĐ-06 Quy tắc khớp tham số của Baseline-L (chốt 2026-10-06)

Phương án "Ít block nhất, lệch ≤ 5% phần thêm": Baseline-L = Baseline + ít NAFBlock khởi tạo đồng nhất nhất,
đặt trong một stage decoder, sao cho số tham số thêm vào lệch không quá 5% so với phần SFT-Map thêm vào Baseline.
Kết quả với NAFNet-32: **2 block ở `dec1`** (c = 128, 1/4 độ phân giải), thêm 237.824 tham số so với 230.273 của SFT-Map
(+3,3%), tổng **17.349.731** tham số. Test: GMACs của Baseline-L không vượt SFT-Map quá 10%.

Lý do bỏ quy tắc cũ ("khớp tuyệt đối"): quy tắc cũ chọn 28 block ở `dec3` (độ phân giải đầy đủ), khớp tới 1 tham số
nhưng chi phí tính toán và bộ nhớ vượt xa SFT-Map, gây out of memory ở batch 8 trong pilot lần một, và làm Phép thử
tham số không còn so sánh dung lượng một cách công bằng. Ràng buộc gốc "lệch < 2% tổng tham số" quá lỏng (phần thêm của
SFT-Map chỉ chiếm khoảng 1,35% tổng) nên không phân biệt được các phương án.

## Giới hạn đã biết (đưa vào chương Giới hạn của báo cáo)

(1) Split chính thức của GoPro có train và test chung video: 5 trong 10 video test (6 trong 11 chuỗi test) có đoạn khác
    nằm trong train: GOPR0384_11, GOPR0385_11, GOPR0868_11, GOPR0871_11, GOPR0881_11. Ảnh test tổng hợp được sinh từ
    frame nét của các chuỗi test, nên mô hình đã thấy cảnh tương tự khi train. Mức trùng như nhau cho mọi biến thể,
    không ảnh hưởng phép so sánh, nhưng làm số tuyệt đối lạc quan hơn. Giữ split chính thức để so được với các công bố.
(2) Val chỉ lấy từ các video có đúng một đoạn trong train và không có đoạn nào ở test, nên val không chung video
    với train lẫn test. Val chỉ có 2 video nên chỉ dùng để theo dõi và kiểm tra lr, không để chọn checkpoint.
(3) Thang vùng nhòe: ô vùng ở test vẫn lớn hơn ở train khoảng 1,3 đến 2,7 lần (QĐ-02).
(4) SSIM tính bằng `skimage`, khác cách cài của BasicSR, nên không so trực tiếp với số SSIM công bố (SA-00).

## Log chẩn đoán khi train

`diag.jsonl` (mỗi `diag_every` = 500 iter): ‖W‖ lớp cuối của SFT head theo stage (`w_sft_<stage>`),
trung bình |Δγ_s| và |β_s| theo stage (`dgamma_<stage>`, `beta_<stage>`), kèm `charbonnier`, `l_map`, `loss`.
`curves.csv` (mỗi 100 iter) ghi `l_map` riêng cho mọi biến thể có bộ ước lượng, kể cả SFT-Free (lambda_map = 0, chỉ để theo dõi),
cùng `grad_norm`, `amp_scale`, `nonfinite_run`. `diag.jsonl` ghi thêm `mid_beta_abs`, `mid_gamma_abs` (trung bình |beta|, |gamma|
của block middle, vốn quyết định head `mid0` có tác dụng hay không). Val ghi `val_psnr`, `val_map_mae` (mọi ảnh) và
`val_spearman` (chỉ ảnh R ≥ 2).
Cảnh báo cần dừng phân tích: `w_sft_*` của SFT-Free hoặc SFT-Map đứng gần 0 sau 10k iter.

| ID | Tên đọc được | Biến thể | Seed | Mức | Config |
| --- | --- | --- | --- | --- | --- |
| SA-10 | Huấn luyện Baseline | A0 | 1 | Lõi | SA-10_a0.yaml |
| SA-11 | Huấn luyện Baseline-L | A0+ | 1 (3 ở bản đầy đủ) | Lõi | SA-11_a0plus.yaml |
| SA-12 | Huấn luyện SFT-Free | A1 | 3 | Lõi | SA-12_a1.yaml |
| SA-13 | Huấn luyện SFT-Map (mô hình đề xuất) | A2 | 3 | Lõi | SA-13_a2.yaml |
| SA-14 | Huấn luyện SFT-Map+Ch | A3 | 1 | Lõi | SA-14_a3.yaml |
| SA-15 | SFT-Map chỉ ở decoder (thay "chỉ ở middle": stage middle gần như bị tắt trong checkpoint) | A2 | 1 | Tùy chọn | SA-15_a2_mid_only.yaml (giữ tên file) |
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
