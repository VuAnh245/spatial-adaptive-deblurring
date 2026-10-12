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
(4) chặn thời gian: notebook dừng mọi tiến trình train sau 11,5 giờ tính từ đầu phiên (SIGTERM, lưu `last.pth`), resume ở phiên sau.

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

(8) pilot lần bốn (FP32, lr 1e-4, 1500 iter, commit d017c54): **PASS**. SFT-Map 26,66 → 31,10 → 31,81 dB và Baseline-L
    26,66 → 31,04 → 31,74 dB ở bước 0 / 750 / 1500 (qua lr đỉnh, không tụt, không đóng băng); grad_norm khác 0 ở mọi bước;
    val bước 0 trùng nhau tới 1e-12 dB; 0,834 it/s cho cả hai run (CPU vẫn là nút thắt, data_frac 0,32-0,34; GPU dùng 87-88%);
    bộ nhớ đỉnh 12,5 / 11,5 GiB nên không cần tích lũy gradient. FP32 gần như không chậm hơn AMP (0,864 it/s).

**ITERS = 30.000** (khóa 2026-10-06) | VAL_EVERY = 2500 | CKPT_EVERY = 2500 | grad_accum = 1 | chốt chặn thời gian 11,5 giờ.

Lý do chọn 30.000 thay vì 32.000 do công thức tính ra: tốc độ do CPU sinh dữ liệu quyết định, và máy Kaggle mỗi phiên có thể
khác nhau; ITERS chung cho mọi mô hình nên phải an toàn cho máy chậm nhất. Với 0,834 it/s, 30.000 iter mất khoảng 10,3 giờ
gồm cả chuẩn bị và val, còn khoảng 1,2 giờ trước chốt chặn 11,5 giờ (chốt chặn gửi SIGTERM, lưu `last.pth`, resume ở phiên sau).

Thứ tự chạy: cặp 1 (SFT-Map SA-13 + Baseline SA-10) trước, cặp 2 (SFT-Free SA-12 + Baseline-L SA-11) ở phiên kế tiếp ngay sau đó,
cùng một tài khoản; chỉ chuyển cặp 2 sang tài khoản khác nếu cặp 1 gặp sự cố. Cả hai cặp chạy đúng một commit đã khóa
(notebook `sad-train` checkout commit đó và kiểm tra hash): **`faf4a06`** (khóa 2026-10-06).

Mốc theo dõi `val_spearman` của SFT-Map (SA-13) ở bước **2.500 / 5.000 / 7.500 / 10.000**. Ở pilot lần bốn giá trị này vẫn
quanh 0 sau 1.500 bước (bản đồ mới học mức nhòe trung bình). Theo thiết kế: nếu vẫn đứng yên gần 0 ở bước 10.000 thì dừng
lại phân tích trước khi đọc kết quả của cặp 2; phép thử bộ ước lượng (`analysis/probe_estimator.py`) cho biết bản đồ có học
được với cùng bộ sinh dữ liệu hay không.

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

## Đánh giá (notebook `sad-eval`, khai báo 2026-10-08, trước khi xem số của cặp 2)

Checkpoint: `final.pth` của SA-10, SA-11, SA-12, SA-13 (bước 30.000), giao thức TLC (QĐ-01).
Test set sinh lại trong phiên bằng `scripts/make_testsets.py` và phải khớp mã sha đã ghi ở `sad-setup` (lệch là dừng).
Hàng đợi `scripts/eval_queue.py` chia hai GPU, ưu tiên theo tầng:
(1) bốn mô hình trên `test_in` và GoPro thật;
(2) bốn mô hình trên `test_ood`, `test_straight`, `test_curved`; SA-20 và SA-21 của SFT-Map trên `test_in`;
(3) SA-20 và SA-21 trên ba tập còn lại;
(4) `test_in` nhiễu ×2, ×4 (SA-26), chỉ khi đủ đĩa;
(5) suy luận toàn cục cho Baseline và SFT-Map trên `test_in` và GoPro thật (phụ lục QĐ-01).
CSV theo ảnh có thêm `video`, `video_in_train` (giới hạn (1)) và `psnr_u8` (so được với SA-00). Cột chính vẫn là `psnr`.

Thứ tự đọc kết quả (QĐ-04): `analysis/eval_tables.py` mặc định che mọi số của Baseline-L (SA-11) và SFT-Free (SA-12).
`--unblind` chỉ chạy khi mục "Kết luận về bộ ước lượng E" đã được commit trong file này, và ghi lại commit đó.


## Lịch hoàn thiện dự án (điều chỉnh theo yêu cầu người dùng ngày 2026-10-11)

Múi giờ: Việt Nam (Asia/Ho_Chi_Minh, UTC+07:00). Đây là quyết định về lịch, không thay đổi giao thức nghiên cứu hoặc trạng thái nghiệm thu.

- Tối 13/10/2026: mục tiêu ưu tiên để có kết quả và bản báo cáo/slide có thể rà soát.
- Tối 14/10/2026: hạn hoàn thiện toàn bộ phạm vi dự án đã chốt.
- Ngày 15/10/2026: ôn tập và diễn tập bảo vệ.
- Ngày 16/10/2026: bảo vệ.

Giữ nguyên checklist C01–C55, phạm vi các thí nghiệm và bằng chứng hiện có. Checklist C01–C55 đã chốt trong cuộc hội thoại là baseline; các cập nhật công việc con và bàn giao hiện hành dùng mục tiến độ qua đêm trong docs/experiments.md làm nguồn chính, không tạo checklist song song. Không đánh dấu hoàn thành từ lịch dự kiến, config hoặc notebook đã chuẩn bị. Giữ nguyên các snapshot audit lịch sử; mốc cũ trong snapshot phản ánh kế hoạch tại thời điểm lập snapshot.

### Căn cứ tài nguyên và tiến độ

- Mốc quota đã kiểm tra bằng API Kaggle ngày 11/10 lúc khoảng 23:31 giờ Việt Nam: dùng 5,152729/30 giờ, còn 24,847271 giờ. Đây là số đo tại thời điểm đó, không phải quota thời gian thực. API báo refresh ngày 17/10, không có múi giờ trong trường trả về; không dựa vào kỳ refresh này để hoàn thành trước tối 14/10.
- Chưa xác minh cách tính quota của phiên T4 ×2. Phân biệt thời gian elapsed của phiên, tổng thời gian từng GPU và số quota nền tảng đã trừ; đo chênh lệch quota trước/sau controls hoặc pilot.
- Mốc cập nhật 12/10 lúc 11:12–11:20: SA-10/11/12/13 seed 1 có checkpoint 30.000 bước và bốn GoPro CSV VerifiedLocal đúng phạm vi. Controls V1/357428270 COMPLETE và đã thu/QC local; A1/A2 seed 2 V1/357522661 được SDK báo RUNNING, chưa có log/output công bố. Pilot SA-14 V1/357437244 FAIL nonfinite bước84; chỉ bằng chứng thất bại đã VerifiedLocal. Không Accepted toàn run/dự án.
- Năm run còn thiếu: SA-14_seed1, SA-12_seed2, SA-13_seed2, SA-12_seed3, SA-13_seed3. Thời gian run cũ khoảng 8,88–9,80 giờ; SA-14 chưa có số đo pilot. Ước lượng năm run khoảng 44,5–51,2 giờ GPU vật lý, 27–32 giờ elapsed khi xếp thành ba wave hai GPU; chưa quy đổi chính xác thành quota Kaggle. Không coi các ước lượng này là kết quả pilot.

### Thứ tự và cửa sổ thực hiện dự kiến

| Cửa sổ (giờ Việt Nam) | Công việc và GPU | Phụ thuộc / điều kiện |
| --- | --- | --- |
| Đêm 11/10 – đầu 12/10, ngay khi người dùng sẵn sàng | Controls SA-10/11/12; process/log/output riêng trên hai GPU. Chuẩn bị input và notebook SA-14 song song | Xác nhận Saved Version thực tế; chỉ thu sau khi remote hoàn tất |
| Sau controls, sớm nhất ngày 12/10 | Pilot SA-14 1.500 bước trên GPU 0; GPU 1 không tự giao thêm việc | Input đã xác minh; đo tốc độ, VRAM, ổn định và quota; chưa tự chạy full training |
| Ngày 12/10, điều chỉnh sau pilot A3 thất bại | A1/A2 seed 2 đã gửi: GPU0 SA-13_seed2/GPU1 SA-12_seed2, V1/357522661. Seed3 chuẩn bị cùng hai GPU trong Saved Version mới sau QC/quota. SA-14_seed1 giữ phạm vi nhưng full run chờ chẩn đoán và pilot đạt | Chưa có it/s/step hiện hành để chốt kết thúc. Log cũ 8,9–9,8h/cặp là ước lượng; mỗi run fresh pretrained. Seed3 + A3 + evaluation chưa được đảm bảo bởi 11,50h quota chưa dùng ở mốc11:12 |
| Suốt 12–13/10 | Chuẩn bị báo cáo/slide, đường cong, ảnh định tính, bảng giả thuyết, tái lập và khai báo dùng chung CV song song; evaluation khi checkpoint/data sẵn sàng | Không suy ra kết quả từ lịch; không trộn giao thức; chuẩn bị job riêng nếu dùng GPU trống |
| Tối 13/10 | Mốc ưu tiên: rà soát kết quả đã có và bản báo cáo/slide; liệt kê rõ các C-ID còn thiếu | Kiểm tra bằng artifact thực tế; không tự giảm phạm vi nếu còn thiếu |
| Ngày 14/10 đến tối | Hoàn tất các đánh giá còn lại, thống kê nhiều seed, LPIPS, hình/bảng, báo cáo/slide/demo, tái lập và sao lưu; người dùng thực hiện Git/phát hành được chấp thuận | Ngày dự phòng để hoàn thiện và xử lý vấn đề, không phải lý do trì hoãn training; hạn toàn bộ vẫn có điều kiện tài nguyên và phép đo |
| 15/10; 16/10 | Ôn tập/diễn tập; bảo vệ | Bản hoàn thiện được khóa trước đó |

### Rủi ro cần báo trước hạn tối 14/10

1. Quota: nếu một giờ elapsed tiêu thụ một giờ quota, ngay riêng 27–32 giờ training đã vượt 24,85 giờ còn tại mốc đo, chưa tính controls/pilot/evaluation. Thêm một ngày lịch không bổ sung quota. Cần số đo quota mới và phương án thêm tài nguyên do người dùng quyết định; không tự mua dịch vụ, liên kết tài khoản hoặc đổi môi trường.
2. SA-14: tốc độ/VRAM chưa đo; OOM, bất ổn hoặc run vượt ngân sách một Saved Session có thể làm trễ các wave. Chỉ đề xuất điều chỉnh theo pipeline và quyết định hiện có sau pilot, không tự đổi hyperparameter.
3. Test tổng hợp/robustness/OOD và RealBlur: còn cần kiểm chứng dữ liệu, khả năng công cụ và thời gian. Sai khác số lượng so với đặc tả chưa có đầy đủ bằng chứng chấp thuận; giữ nguyên vấn đề này khi báo cáo tiến độ.
4. LPIPS AlexNet (C52): lớp metric đã có nhưng evaluator chưa xuất LPIPS; CSV hiện tại không đủ để tính metric này. Cần đề xuất sửa tối thiểu công cụ hiện có và đo lượt inference bổ sung, chưa tự sửa/chạy.
5. Thống kê ba seed A2/A1, kết luận giả thuyết (C53) và phát hành A2 (C54) phụ thuộc các checkpoint/evaluation chưa hoàn tất. So sánh A2/A0+ ở phạm vi lõi chỉ seed 1. Báo cáo (C43), hình (C41) và khai báo CV (C55) phải chuẩn bị sớm, không đợi toàn bộ training.

Sau mỗi mốc có bằng chứng, cập nhật đúng C-ID và phạm vi remote/local/verified trong metadata tương thích; giữ nguyên trạng thái phần chưa đạt. Chưa cam kết toàn bộ phạm vi hoàn tất tối 14/10 trước khi có pilot, quota và thời gian đánh giá thực tế.


## Kết luận về bộ ước lượng E

Bản kết luận chuẩn bị lúc khoảng 00:17 ngày 12/10/2026, giờ Việt Nam, trước khi mở kết quả controls SA-11/12. Đây là kết luận mô tả về estimator trên validation đã có, không phải kết luận hiệu quả restoration hoặc nguyên nhân thất bại.

Nguồn: results/artifacts/kaggle/355813018/estimator_diag.json (SHA-256 0301af809b937578d7b9185f606e416d40bd0c27d30b6b9c9497b38ba1358dfa), checkpoint SA-13_seed1/final.pth từ xunanhv/sad-train V2/scriptVersionId 355813018. Diagnostic dùng 40 validation NPZ gốc, 30 ảnh R ≥ 2; không sinh lại split hoặc chọn checkpoint theo validation.

- MAE model = 0,1769647143; MAE bản đồ hằng dùng mean ground truth của chính validation = 0,1773214638. Baseline hằng này là đối chiếu diagnostic có sử dụng thống kê target validation, không phải đối chứng được train.
- Spearman trung bình trong ảnh R ≥ 2 = 0,0476140375; pooled pixels = 0,0362284104. Mean Ŝ qua ảnh chỉ 0,369620–0,377562 so với mean S 0,003943–0,690115; tỷ số độ lệch chuẩn trong ảnh = 0,058103. Calibration gần phẳng quanh Ŝ ≈ 0,373.
- Với checkpoint/validation này, estimator chưa thể hiện bản đồ có biến thiên và tương quan tốt với ground truth; MAE chỉ gần baseline hằng. Chưa có bằng chứng phân biệt nguyên nhân do nhãn/dữ liệu, khả năng nhận dạng blur từ ảnh, tối ưu hóa hay cạnh tranh gradient. Không suy ra dữ liệu không học được, toàn bộ SFT vô ích hoặc implementation sai.
- results/audit/2026-10-11/mechanism-probe.json đã có checks_passed trên mô hình/nhãn toy; đó chỉ là kiểm tra kết nối gradient và consistency toy, không chứng minh estimator đã học trên checkpoint thật. Probe E độc lập chưa chạy (results/probe_E chưa có), oracle/shuffle trên test chưa có CSV verified.

Log training SA-13 diag.jsonl cũng cho thấy head encoder/decoder có norm lớp cuối khác0 (ở bước30k khoảng0,582–1,611), trong khi mid0 khoảng2,045e-5 với mean |delta gamma| khoảng7,14e-7 và |bias| khoảng1,09e-6. Đây là số trên training patches, không phải validation/test; không quy near-constant map thành mọi SFT head không cập nhật và chưa chứng minh quan hệ nhân quả. Source SHA-256 46599fc088d9f68fe340fdb804cbd466017389d168de81ca38bf243f628f1afc.

Giữ gate trong analysis/eval_tables.py::unblind_gate(): người dùng phải commit heading/kết luận này; docs/experiments.md phải không còn sửa chưa commit trước --unblind. AI chưa stage/commit và chưa mở metrics SA-11/12. Việc thêm heading local không tự vượt gate; không dùng compare.py để lách thứ tự đọc kết quả.

## Tiến độ hiện hành — kế thừa checklist C01–C55

Nguồn tiến độ chính cho cập nhật hiện hành: mục này trong docs/experiments.md; kế thừa C01–C55, chỉ cập nhật công việc con. Giữ các snapshot docs/reports/audit-2026-10-11.md và inventory lịch sử nguyên trạng. Mục tổng chưa hoàn tất nếu chỉ công việc con được tích [x]. Các giờ dưới đây là kế hoạch có điều kiện, không phải job thực tế đã bắt đầu.

### Trạng thái kiểm tra lúc 00:09–00:32 ngày 12/10

- Controls xunanhv/sad-eval-gopro-controls/edit đã tồn tại nhưng editor còn cell mẫu, Show Versions = 0, không input, accelerator None, draft off. API kernels_status trả HTTP 404; chưa có Saved Version/scriptVersionId để ghi Running hoặc thu artifact. Hai training ID 355813018/355940350 không phải nguồn evaluation mới.
- Chỉ CSV SA-13 có tại results/eval; controls chưa có CSV local. CSV SA-13 và checkpoint cũ tái sử dụng evidence đã xác minh; không strict-load hoặc audit main lại.
- API quota lúc 00:09:12: tổng 30h, dùng 5,738403624h, reserved 0, còn 24,261596376h. Phải đọc timedelta.total_seconds(); chuỗi repr JSON của SDK bỏ phần ngày, hiển thị totalTimeAllowed 21600s dù giá trị thật là 1 ngày + 6h = 108000s. Refresh API 17/10 sau hạn dự án; timezone không có trong datetime trả về.
- Browser Session options hiển thị khoảng 05:41/30h đã dùng; Active Events có draft recovery T4 ×2 chạy khoảng 33 phút tại mốc quan sát. Người dùng kiểm tra có tác vụ/artifact mới cần giữ trước khi dừng draft; AI chưa dừng phiên. Không đồng nhất active draft recovery với controls Saved Version.
- API qua sandbox mặc định gặp WinError 10013; thao tác chỉ đọc đã chạy được qua môi trường được cấp quyền mạng. Không cần yêu cầu người dùng tải thủ công vì lỗi này.

### Kế hoạch và tiêu chí đến 08:00 (giờ Việt Nam)

| C-ID / công việc con | Trạng thái và ô nghiệm thu | Bắt đầu → kết thúc dự kiến | Phụ thuộc / tài nguyên / người thực hiện | Input → output / tiêu chí và evidence | Song song / khi bị chặn |
| --- | --- | --- | --- | --- | --- |
| C23 — bundle/notebook controls | [x] Đã đạt: bundle và Saved V1/357428270 đã dùng thành công | Worker preflight 12/10; log source đã thu | AI chuẩn bị/thu; người dùng khởi chạy | 57 payload input kiểm hash; evaluator/src/manifests khớp SA-13 | Giữ nguyên source bundle/notebook |
| C24–C26 — GoPro SA-10/11/12 | [x] Đã đạt trong phạm vi CSV GoPro seed 1; remote COMPLETE | 12/10: evaluator bắt đầu khoảng 10:41; GPU0 SA-11 kết thúc khoảng 11:10; worker log 1864 giây | T4 ×2; GPU0 SA-10 rồi SA-11, GPU1 SA-12 | Ba CSV 1.111 ảnh/run; queue returncodes 0; không đồng nhất elapsed Viewer với compute evaluator | Không chạy lại evaluation |
| C24–C26 — thu và kiểm tra local | [x] Đã đạt: ba CSV VerifiedLocal | Thu/QC trong phiên từ 11:12 ngày 12/10 | AI; đúng V1/357428270, official SDK | Archive 622513 bytes SHA-256 872667327f0a92470f1d378388f6d9d8fe36bf0b23204f2da97872d67711b5c3; CRC + manifest 31/31; ID/video/finite metrics/protocol đạt; receipt ở results/artifacts/kaggle/357428270/download-provenance.json | Registry chỉ notes SA-10/11/12; status Verified, evaluation_status Missing, evaluation_ready False; SA-13/14 giữ nguyên |
| C10–C14 — chuẩn bị remaining training | [x] Input/notebook A1/A2 đã chuẩn bị và gắn đúng; [ ] nghiệm thu full runs | A1/A2 seed 2 V1/357522661 được gửi khoảng 10:05 ngày 12/10 | Người dùng Save Version; AI kiểm source/phase/input/GPU | PHASE=wave_a12_seed2, GPU0 SA-13_seed2/GPU1 SA-12_seed2; 30k/run, fresh pretrained, FP32 | A3 vẫn bị chặn riêng; chưa khởi chạy seed 3 |
| C10–C14 — pilot SA-14 | [ ] Blocked: V1/357437244 pilot FAIL, aborted_nonfinite step 84 (51 consecutive); failed-run evidence VerifiedLocal, no checkpoint | 01:50–02:40, ước lượng 35–50 phút + setup | User Save Version; GPU0, cùng runtime/FP32/config khóa; quota đủ | SA-14_seed1_pilot1500, val0/750/1500 → remaining_training_pilot_sa14.zip, logs/curves/events/final/last/runtime/VRAM/pilot report | AI thu/QC sau finished. OOM/nonfinite/runtime mismatch: giữ output, báo blocker; không tự đổi batch/lr |
| C10–C14 — full A1/A2 seed 2 | [ ] Đang chạy: SDK V1/357522661 RUNNING; chưa đủ log/bằng chứng bước training | Job gửi khoảng 10:05 ngày 12/10 → chưa chốt kết thúc; một cặp theo log cũ 8,9–9,8 giờ, ước lượng | T4 ×2, process/log/output riêng | SA-13_seed2 + SA-12_seed2 phải đạt 30k, final/last/config/log và local QC; chưa có output API công bố | Sau checkpoint sẵn sàng mới evaluation; không đánh dấu đạt từ trạng thái Running |
| C10–C14 — tài nguyên bổ sung | [x] Có quota mới và đã đối chiếu serialization; [ ] nguồn GPU bổ sung | Chuẩn bị 00:15–01:00; xử lý song song | User liên kết Colab Pro nếu có hoặc chọn GPU khác; AI kiểm tra thực tế | So quota trước/sau job; ghi GPU-hours vật lý/elapsed/quota riêng. Mốc hiện tại không đủ đảm bảo cả ba wave và evaluation | Nếu có Pro: File → Link to Colab, user xác thực, đo quota sau. Không coi quyền lợi đã nhận trước API/UI xác nhận |
| C41 — đường cong và diagnostic figure | [x] Dữ liệu/nhãn/nguồn được kiểm tra; [ ] QA trực quan / toàn bộ hình | 00:10–00:40 | CPU/AI; SA-10/13 logs và JSON diagnostic đã có | training_curves_seed1_blind.html đã có; estimator_diag_sa13_val40.html mới, source hash đầy đủ; không inference mới, không so total loss | Pair2 giữ blind. Local thiếu matplotlib/kaleido, không cài; người dùng mở HTML/export PNG để duyệt trực quan |
| C43 / C53 — khung báo cáo và bảng giả thuyết | [x] Khung UTF-8/dàn ý báo cáo-slide; [ ] nội dung cuối | 00:10–01:00; bổ sung evidence tiếp đến 08:00 | CPU/AI, kết quả verified và thiếu sót hiện có | notebooks/03_results.ipynb; giữ code cell cũ, không số giả; seed3 chỉ A2/A1; A2/A0+ seed1 | Không đợi training để chuẩn bị; chỉ điền controls sau local QC và unblind gate |
| C53 — kết luận estimator / gate | [x] Mô tả estimator có evidence trước mở cặp 2; [ ] user review/commit và unblind | Controls CSV đã VerifiedLocal; bước phân tích tiếp theo khi gate đạt | Người dùng Git và chạy analysis/eval_tables.py --unblind | Heading ## Kết luận về bộ ước lượng E phải có trong HEAD và docs/experiments.md không còn sửa chưa commit; chưa công bố mean SA-11/12 | AI không Git/không bypass gate; bảng bốn model và kết luận giả thuyết còn chờ |
| C52 và phần cơ chế/qualitative/C55 liên quan | [x] Đã đọc code, xác định blocker; [ ] triển khai/chạy/thu kết quả còn thiếu | Rà soát 00:15–01:10; xử lý đến 08:00 khi có input | AI đề xuất tối thiểu; user chạy/sửa được chấp thuận | Đề xuất cụ thể bên dưới; RealBlur/data/hash testsets/LPIPS/các prediction còn thiếu không đổi trạng thái tổng | Tiếp tục công việc độc lập; không chạy probe, inference hoặc sinh dataset thay user |
| C10–C14, C24–C26, C41/C43/C52–C55 — bàn giao 08:00 | [ ] Mốc 08:00 đã qua; không hồi tố hoàn tất. Bàn giao hiện tại dựa trên evidence sau 11:12 | 07:45–08:00 nếu phiên còn hoạt động | AI đọc log/artifact, user thao tác phụ thuộc | Ghi đúng việc đạt/đang chạy/bị chặn, source version/local/hash, quota và lịch tiếp ngày 12/10 | Nếu chat/app đóng: không watcher nền; lưu trạng thái thực tế trước khi kết thúc, không giả nhận đạt 08:00 |

Các giờ dịch theo thao tác thực tế của người dùng. Nếu launch trễ hoặc pilot FAIL, ghi lại giờ thật/blocker và tiếp tục công việc CPU; không chuyển trạng thái từ bảng lịch.

### Đề xuất sửa tối thiểu vào công cụ hiện có (chưa triển khai)

| Khoảng trống đã đọc | Thành phần cần mở rộng | Thay đổi tối thiểu / giới hạn |
| --- | --- | --- |
| LPIPS AlexNet: class có, evaluator không gọi; local chưa có lpips | scripts/evaluate.py + metrics/image_quality.py + analysis/eval_tables.py | Khởi tạo LPIPSMetric một lần, dùng cùng RGB float prediction đã clamp và GT, [-1,1] qua class hiện có; thêm cột lpips và metadata weights/version/hash. Giữ default cũ và CSV controls đang chạy; flags mới chỉ là đề xuất, chưa tồn tại. Cần user chấp thuận triển khai và pilot tài nguyên; không suy LPIPS từ PSNR/SSIM |
| Chưa lưu prediction/map/crop TLC | scripts/evaluate.py; nếu dùng restore_folder.py thì phải bổ sung lựa chọn pooling rõ ràng | Lưu một danh sách ID/crop chung, prediction đang dùng tính metrics và map float có nguồn/kích thước; không inference global để minh họa số TLC. restore_folder.py hiện gọi restore_full và không convert_tlc. Không dùng generator figures làm prediction |
| RealBlur không đi qua alignment trong evaluator | metrics/realblur_eval.py + evaluate/restore hiện có | Wrapper đòi official evaluation_RealBlur_ecc.py, hiện thiếu third_party, dữ liệu và pair manifest. Lấy official version/hash/license trước; giữ alignment chính thức, không gọi metric GoPro thay thế. Chưa đủ input hoặc phép đo để lập giờ kết thúc |
| Queue chỉ MODELS SA-10..13 và không có seed trong job identity; resume chỉ kiểm file tồn tại | scripts/eval_queue.py và eval_tables.py | Nhận model/run/seed/config/ckpt đã verified theo kế hoạch; A3 seed1 và chỉ A1/A2 seed2/3. Output riêng theo seed để giữ stem parser cũ; xác minh CSV/ID/n/provenance trước skipped_done. Không mở rộng mọi đối chứng thành 3 seed; không ghi đè CSV |
| Test tổng hợp chưa có registry hash local, make_testsets có xóa thư mục | scripts/make_testsets.py dùng generate_testsets.py cũ | Chỉ dùng workspace/temp mới; thu đúng hash/count đã đăng ký ở sad-setup trước khi sinh. Đề xuất preflight chống ghi đè và yêu cầu expect đủ mọi split trước thao tác ghi; chưa chạy hoặc sửa. Config counts không chứng minh artifact hoặc DoD gốc đã đạt |
| Benchmark chi phí mặc định AMP/global, khác bảng chính TLC/FP32 | scripts/benchmark_efficiency.py + metrics/efficiency.py | Cho phép chọn config/pooling/precision rõ ràng, áp dụng convert_tlc trước latency, gọi amp=False cho FP32, ghi warmup50/runs200/batch1/input/GPU/runtime/hash. Dùng hàm cũ, không thay metric/timer; GMACs phải báo tool/operator coverage và kích thước256. Chưa sửa/chạy; MACs=None nếu thiếu fvcore không phải số0 |
| Probe E độc lập vs toy graph probe | analysis/probe_estimator.py / estimator_diag.py / audit_mechanism_probe.py | Không gộp hai chức năng. Diagnostic validation đã có; toy checks đã có; probe_E còn thiếu và là training estimator nhỏ, user phải chạy. Đo pilot CPU/GPU trước khi xếp giờ; không thay thế bằng toy PASS |

### Nhật ký việc con đã đạt / nguồn bàn giao

- 00:09 ngày 12/10: quota API 24,261596376h còn, đọc bằng total_seconds(); controls editor 0 version/không input/cell mẫu, chưa chạy. C24–C26 giữ chưa làm; registry chưa đổi.
- Khoảng 00:15: notebooks/03_results.ipynb sửa markdown mất dấu tiếng Việt, bổ sung khung báo cáo/slide và bảng giả thuyết; các code cell có sẵn giữ nguyên, notebook chưa chạy training/eval.
- Khoảng 00:16: tạo results/eval/estimator_diag_sa13_val40.html từ diagnostic có sẵn, 4.856.134 byte, SHA-256 6368e45f2122998423d39f0b73b1eef3fd428e55049d0f496eecf9e85e1cb3db. Source JSON SHA-256 0301af809b937578d7b9185f606e416d40bd0c27d30b6b9c9497b38ba1358dfa. Notebook JSON/nbformat/compile đạt; chỉ chạy plotting CPU, không inference. QA trực quan vẫn pending; không tích hoàn tất C41 tổng.
- Khoảng 00:17: chuẩn bị kết luận estimator và kế hoạch qua đêm trong tài liệu hiện có. AI chưa chạy training/evaluation, sửa model/config/manifests, xóa file, stage/commit/push hoặc tạo Saved Version. Intake tiếp theo chờ controls phiên thật và output hoàn tất.

- 00:22–00:31 ngày 12/10: controls editor vẫn cell mẫu, không input, Show Versions0; SDK metadata current_version_number0. Chưa có job để thu. sad-setup cũng current_version_number0; chưa xác định nguồn Saved Output hash testsets, không lấy training ID thay thế.
- Khoảng 00:30: sửa duy nhất markdown bị mất dấu trong results/artifacts/kaggle/sad-train-remaining.ipynb, giữ JSON mọi code cell nguyên vẹn; nbformat/compile PASS, không thực thi. SHA-256 notebook hiện a2a07437f132f7f82c5ff1b3e047a0e9fbfc418201972c81bcafbf1047eed2d0. ZIP input không đổi/không đóng gói lại. Phase vẫn pilot_sa14, không tự chuyển wave.
- Khoảng 00:31: notebooks/03_results.ipynb bổ sung tám ID/crop chung chọn không dùng PSNR; kiểm ID/nhãn/tám video PASS. Chỉ chuẩn bị selection, chưa tạo ảnh prediction; C41 tổng vẫn pending. Bảng blind summary.csv/comparisons.json/README.txt ở results/eval/tables đã xuất bằng eval_tables.py, chỉ SA-13 và comparisons rỗng. Lần console Windows cp1252 đầu lỗi sau ghi summary; lần UTF-8 hoàn tất, tái sử dụng summary byte-identical và không ghi đè khác nội dung.
- 00:31:46 ngày 12/10: API quota dùng6,114688122/30h, reserved0, còn23,885311878h; total_seconds. Tăng khoảng0,376284498h từ mốc00:09:12, gần22,58 phút elapsed. Mẫu này phù hợp khoảng một giờ quota/giờ elapsed cho draft GPU đang active, chưa xác nhận mọi Saved Session hoặc cách tính phiên chồng nhau. Controls chưa chạy.

### Tài nguyên bổ sung và giới hạn bàn giao

Ưu tiên dùng quyền lợi Colab Pro/Pro+ hiện có qua **File → Link to Colab** trên Kaggle, người dùng liên kết/xác thực và AI đo quota sau; không tự mua hoặc liên kết tài khoản. [Kaggle Notebook documentation](https://www.kaggle.com/docs/notebooks) mô tả liên kết để có thêm quota; mức nhận thực tế phải xác minh API/UI. Câu hỏi người dùng có Pro đang chờ trả lời, chưa coi đã có tài nguyên bổ sung.

Nếu dùng Colab trực tiếp, không upload rồi chạy nguyên notebook Kaggle: notebook hiện cần hai T4 và /kaggle/input. Chỉ lập adapter input/runtime vào pipeline train.py cũ sau khi biết GPU/VRAM, giữ pretrained/seed/config/manifests; cần user khởi chạy và xác minh môi trường. [Colab FAQ](https://research.google.com/colaboratory/faq.html) nêu GPU/time limits phụ thuộc mức tài nguyên và compute units, không đảm bảo loại GPU. Không lập lịch GPU được bảo đảm khi chưa có session thực tế.

Ở mốc23,885h còn: ba wave training ước27–32h elapsed đã thiếu khoảng3,1–8,1h nếu quota xấp xỉ1:1, chưa có pilot, controls, evaluation và dự phòng. Nhu cầu bổ sung8–12h quota là dự toán, không phải pilot hoặc gói dịch vụ đảm bảo. Nếu billing theo số GPU hoặc SA-14 chậm, thiếu nhiều hơn. Rủi ro vượt tối14/10 tập trung vào seed2/3, SA-14, các testset/RealBlur/LPIPS và đánh giá nhiều seed; không giảm phạm vi hoặc tuyên bố toàn bộ khả thi.

Runtime tạo PPTX chuẩn của skill Presentations (load_workspace_dependencies/artifact-tool) không có tool khả dụng trong phiên. Khung slide9 phần đã có tại notebooks/03_results.ipynb; chưa có PPTX hoặc QA layout. Không cài thư viện thay thế. Đồ thị HTML chuẩn Plotly có thể mở/export để bổ sung báo cáo khi có môi trường render phù hợp.

- Khoảng 00:34–00:36 ngày12/10: đếm tham số CPU theo config bằng build_variant/count_params, không load checkpoint hoặc forward. A0=17.111.907; A0+=17.349.731; A1/A2=17.342.180; A3=17.672.660. A0+ thêm2 block dec1, lệch phần tăngA2 3,2792%; chỉ nghiệm thu phần bảng tham số, chưa benchmark TLC/FP32. Ghi bảng trong notebook kết quả cũ.
- Khoảng 00:36: tạo results/eval/modulation_training_sa13_seed1.html bằng cell notebook kết quả từ60 hàng diag SA-13 gốc; 4.902.357byte, SHA-256 75a74f2b26b8e87b1768b7a056fc3fa67208548b3f4c516ba3113677a97ad794. Source diag SHA-256 46599fc088d9f68fe340fdb804cbd466017389d168de81ca38bf243f628f1afc. TrụcY log, training patches; middlehead gần0 khác encoder/decoder. Chỉ vẽCPU; QA trực quan/C41 tổng pending, chưa thay oracle/shuffle.

### Bàn giao thực tế lúc 00:38 ngày 12/10 (chưa phải mốc 08:00)

- [x] C41/C43/C53, phần chuẩn bị: notebook kết quả có khung báo cáo/slide, bảng giả thuyết, kết luận estimator, hai hình diagnostic và tám ID/crop chung. Bảng tham số CPU đã đếm. Ba notebook đạt schema/compile, chưa chạy training/evaluation. QA trực quan, PPTX, báo cáo cuối và kết luận so sánh vẫn [ ].
- [x] Bảng BLIND sinh bằng công cụ cũ: summary.csv SHA-256 ad6c9236dba93b8aae2cb64952ac6583c725409ec9938b4b883a6c7e848d6b7a; comparisons.json SHA-256 d9df86de5e3c47676f256bbd8f2578a48e7776faac44411a6038bc61442e105a; README.txt SHA-256 be1b2b159ba5d3b42d348cf76207bef158b0c6478470f82cd8f6b809aaeb60d4. Chỉ có SA-13; chưa đủ bảng đối chứng.
- [ ] C24–C26 controls: editor đúng xunanhv/sad-eval-gopro-controls, Show Versions = 0, còn cell mẫu, chưa input. Chưa có job Saved Version hoặc CSV remote/local; đợi người dùng import, gắn input và khởi chạy. Notebook chuẩn bị SHA-256 8217ca858b790e76cd12591e4495cf457e5d1d96b54cabe14f82c3a2864bfde9, không đổi.
- [x] C10–C14, phần notebook/bundle chuẩn bị; [ ] pilot SA-14 và năm full run. Phase vẫn pilot_sa14, không tự chuyển sang full run; ZIP nguồn giữ nguyên.
- [ ] GPU bổ sung. Quota mốc 00:31 còn 23,885h. Recovery Interactive T4 x2 vẫn active khoảng một giờ tại 00:37; người dùng kiểm tra output cần lưu trước khi Stop. AI chưa dừng phiên hoặc tạo job.
- [ ] Hash testsets, RealBlur, LPIPS, oracle/shuffle, prediction TLC, benchmark TLC/FP32 và khai báo chia sẻ CV còn thiếu. Đề xuất tối thiểu đã ghi bên trên; không đổi trạng thái nghiệm thu.
- Registry/index và snapshot audit lịch sử giữ nguyên. Verified ở phạm vi cũ không đổi thành Accepted. Không stage/commit/push hoặc sửa research code/config/manifests.
- Bước người dùng: import notebook controls, upload comparison-input.zip thành private input, gắn sad-data, chọn T4 x2 và Save & Run All. AI thu sau khi xác định đúng version hoàn tất trong phiên đang hoạt động. Không có watcher khi chat/app đóng; chưa tới 08:00 và không ghi nghiệm thu từ lịch.

- 2026-10-12T00:53:11+07:00: controls draft ?? l?u sau khi ng??i d?ng reload. SDK x?c nh?n 4 cell, 3 code cell kh?p nguy?n source notebook local; input xunanhv/sad-eval-gopro-controls-seed1 v? xunanhv/sad-data ?? g?n. C23 ??t ph?n l?u/g?n input; preflight hash/runtime/data ch?a th?c thi. C24?C26 v?n ch?a ch?y, Saved Version = 0, ch?a CSV remote/local; ch?a c?p nh?t registry. Ng??i d?ng ti?p theo Save Version ? Save & Run All v?i GPU T4 x2.

- 2026-10-12T01:05:10+07:00: C23 ??t ph?n t?o Saved Version controls do ng??i d?ng kh?i ch?y: xunanhv/sad-eval-gopro-controls V1/scriptVersionId 357428270. Viewer Edit link tr? /edit/run/357428270; request SDK version_label=v1 x?c nh?n kernel_type=notebook, enable_gpu=True, source code kh?p notebook local. Browser/SDK ??u Queued, GPU T4 x2, Latest Container Image, Output 0 B. C24?C26 c?n pending: ch?a preflight ho?c CSV; ch?a FinishedRemote/DownloadedLocal/VerifiedLocal. Tab Script/JSON ch?a ch?ng minh execution sai; gi? job hi?n t?i, kh?ng t? h?y ho?c ch?y l?i. Input source IDs tr?n notebook ngu?n: datasetVersion 20597346 v? sad-data kernelVersion 355369793.

- 2026-10-12T01:27:54+07:00: C10-C14 input preparation: xunanhv/sad-train-remaining-input created by user; official SDK listing matches prepared ZIP for all 94 files including MANIFEST.txt and 40 validation NPZ, by names and sizes. Remote content hashes remain pending notebook preflight. Pilot notebook PHASE=pilot_sa14; pilot/full training not started. C24-C26: controls V1/357428270 is RUNNING per pinned SDK status; no failure message, no completed output or local verification claimed. Next: user imports the existing sad-train-remaining.ipynb into a separate notebook and attaches this dataset plus verified sad-data output; do not alter the running controls job.

- 2026-10-12T01:33:59+07:00: C10-C14 pilot setup: user imported and attached inputs to xunanhv/sad-train-remaining; SDK confirms notebook, 3 code cells identical to prepared local notebook, PHASE=pilot_sa14, enable_gpu=True, NvidiaTeslaT4, sources xunanhv/sad-train-remaining-input and xunanhv/sad-data, Saved Version count 0. Input hashes/runtime remain pending preflight. Controls V1/357428270 remains RUNNING. GPU quota measured with timedelta.total_seconds(): used 7.464 / 30.000 h, reserved 0.000 h, remaining 22.536 h; this is platform quota, not aggregate physical GPU-hours. Pilot not launched; user next Save Version -> Save & Run All with GPU for this session. No Accepted or local artifact verification claim.

- 2026-10-12T01:37:34+07:00: C10-C14 pilot launch: user supplied pinned URL https://www.kaggle.com/code/xunanhv/sad-train-remaining?scriptVersionId=357437244. Official SDK version_label=v1 confirms Saved Version 1, notebook type, GPU enabled/NvidiaTeslaT4, all three code cells identical to local prepared notebook, PHASE=pilot_sa14, RUNNING and no failure message. Pilot target remains SA-14_seed1_pilot1500 / 1500 steps; full training not started. C24-C26 controls V1/357428270 remains RUNNING. Neither job has remote-completion or local-QC evidence yet; no registry/Accepted update.

- 2026-10-12T01:44:39+07:00: C10-C14 SA-14 pilot V1/357437244 FAILED: preflight passed all 93 bundle files/runtime, original val40 and public pretrained identity; step-0 val PSNR 26.6560. Trainer aborted_nonfinite at step 84 after 51 consecutive non-finite losses, process returncode 3, runtime 108.135 s, peak allocated 11.096 GiB / reserved 12.615 GiB. Failure evidence downloaded to results/artifacts/kaggle/357437244/original/remaining_training_pilot_sa14.zip (4338 bytes; SHA-256 b526e003bc67d0b9d745c1ab2823b17e6d568b5aa201eda100ddc350b70b3675); CRC and existing verify_manifest.ps1 passed 9/9 files; source notebook, session log and download-provenance.json retained. VerifiedLocal scope is failed pilot evidence only, not a training checkpoint. No final.pth/last.pth/train_done.json or throughput estimate; no OOM evidence. Main run blocked on numerical diagnosis; no automatic batch/LR/guard/model changes, rerun or wave1 launch. A1/A2 new seeds remain independent work but unlaunched; resource scheduling still requires actual quota and a user launch decision. Registry and audit snapshots unchanged.

- 2026-10-12T01:51:11+07:00: C10-C14 independent A1/A2 seed preparation: results/artifacts/kaggle/sad-train-a12-seeds.ipynb created (SHA-256 5abc350805d1388fcf89ba8e06c911fe2b5681c7d6f4dd8e322805785c71807a), 4 cells, nbformat and all code/embedded-worker compile checks passed without execution. Default wave_a12_seed2: GPU0 SA-13_seed2 / GPU1 SA-12_seed2. Next fresh Saved Version wave_a12_seed3: GPU0 SA-13_seed3 / GPU1 SA-12_seed3. Locked worker/trainer/input/hyperparameters unchanged, all fresh public pretrained and 30000 steps; old pilot notebook preserved. These phases prepare the four planned A1/A2 seeds independently of the failed A3 pilot; no full job started by AI. Original wave1 including A3 remains blocked pending diagnosis; SA-14 has not been removed. Latest controls V1/357428270 status KernelWorkerStatus.RUNNING; remaining platform quota 21.915 h (not physical GPU-hours). Estimate one paired A1/A2 phase 8.9-9.8 h from old logs, host-dependent; all remaining scope including A3/evaluations is still not guaranteed within quota.

A3 numerical diagnostic proposal for review (not executed, no research code/config change): record input/predicted-image/S_hat finiteness, Charbonnier and map losses, first nonfinite activation in channel/spatial/NAFBlock paths, gradient finiteness/norm before and after existing clipping, and parameter finiteness before/after optimizer update, per step from initialization through the first failure. Keep public pretrained/seed 1/data stream/batch 8/FP32/LR/loss and original guard unchanged so observations can locate the initiating failure. Current evidence has no per-step curves (log interval 100, failure event step 84), no trained checkpoint or intermediate diagnosis. Reading trainer.py shows backward and optimizer step continue for nonfinite loss until the 51-consecutive abort threshold; GradScaler is disabled under FP32. This is a propagation risk, not proof of the initiating A3 cause. Any guard/model/hyperparameter change or additional diagnostic training run must be reviewed with the user before execution.

- 2026-10-12T10:14:51+07:00: C10-C14 A1/A2 seed-2 launch confirmed: user screenshot plus official SDK version_label=v1 confirm xunanhv/sad-train-a12-seeds Saved Version 1, code identical to prepared notebook, PHASE=wave_a12_seed2, GPU enabled/NvidiaTeslaT4, only the intended remaining-training-input dataset and sad-data notebook sources. Connected Viewer Edit link /code/xunanhv/sad-train-a12-seeds/edit/run/357522661 establishes actual scriptVersionId=357522661; pinned Viewer reports Running (about 8 minutes). This is job status only: live log empty/unavailable, no preflight or trainer-step proof, no remote-complete/local checkpoint claim. Planned runs remain GPU0 SA-13_seed2 and GPU1 SA-12_seed2, 30000 steps each.

Resource/schedule exception: connected Viewer and SDK report controls V1/357428270 still RUNNING about 9 hours, far beyond old 35-45 minute estimate; after refreshing pinned Logs page, Loading remains and no log text is available. Pinned list-output log is empty; official live-log stream attempt returned HTTP 400 (session identity not resolved through the supported request), so neither job failure nor current evaluator progress is established. No cancellation or rerun performed. Active Events additionally shows sad-eval-sa13-recovery Interactive T4x2 about 11 hours; user should review and Stop only that unused draft if no active cell or new artifact needs preserving. Existing Saved recovery V1/local evidence stay intact.

Latest measured quota near 10:10 Vietnam: used 24.796 / allowed 30.000 platform hours, reserved 11.906 h, allowed-minus-used 5.204 h. Reserved time is distinct from used quota and physical GPU-hours; it is not evidence of extra credits or guaranteed completion. The 01:51 remaining-quota snapshot is obsolete. Current full pair and all remaining scope are at resource risk; no guaranteed end time or reduction of SA-14/seeds/datasets. Additional Colab Pro/Pro+ availability requested; no purchase, linking, environment change or AI job launch.

- 2026-10-12T10:58:09+07:00: C41 provenance check prompted by user screenshot: training_curves_seed1_blind.html SHA-256 aa1c108b3e778680c94980bccdf12be0b4d78e3ff39f986e282968fb3e3cf3b3. All x/y values in all 10 embedded Plotly traces exactly match original SA-10_seed1/SA-13_seed1 curves.csv and val.jsonl from sad-train V2/355813018. Hashes of curves/val/events/train_done for both runs (8 files) match the original download receipt; each run has 300 training records, 13 validation records and train_done steps=30000. Final synthetic val40 PSNR: SA-10 33.79146857040343 dB, SA-13 33.80335535587285 dB; these are not GoPro-test metrics. No invented points, smoothing, new model execution or figure overwrite. Screenshot confirms first four panels render; full six-panel/export-layout QA remains pending, title contains a cosmetic question-mark character. C41/project not marked fully complete.

- 2026-10-12 11:22:24 (+07:00): C24–C26 đạt CSV GoPro VerifiedLocal cho SA-10/11/12 seed 1, nguồn xunanhv/sad-eval-gopro-controls V1/357428270. Archive/log/notebook nguồn đã tải; CRC và verify_manifest.ps1 31/31 đạt. Mỗi CSV 1111 ID duy nhất, tập ID/pair/video/video_in_train đúng manifests gốc; psnr/psnr_u8/ssim hữu hạn; hash checkpoint/config/evaluator/runtime/data khớp bằng chứng và SA-13 recovery. CSV local ở results/eval/; receipt SHA-256 7755bf797e150a1726887a704af8cb61380829c69bcff5d07771838252138c66. CSV/JSON round-trip và schema 32 cột đủ năm hàng đã kiểm trước metadata write; chỉ notes SA-10/11/12 thay đổi, SA-13/14 giữ nguyên. Không Accepted toàn run/dự án, không mở mean cặp 2.

Điều chỉnh theo tiến độ thực tế lúc 11:12 ngày 12/10: A1/A2 seed 2 V1/357522661 vẫn RUNNING; API chưa công bố log/output nên chưa có số đo it/s hoặc xác nhận checkpoint mới. Controls đã hoàn tất với wall evaluator SA-10 854,264 s, SA-11 903,657 s, SA-12 929,758 s; worker session log kết thúc sau khoảng 1864 s (~31 phút). Viewer từng báo gần 9 giờ chưa giải thích được phần chờ nền tảng trước worker; không kết luận evaluator chạy 9 giờ hoặc phải rerun. Quota API mới used18,499753/30h, reserved0, unused11,500247h; số giảm so với mốc tạm 10:10 là thay đổi số nền tảng trả về, không chứng minh được cấp thêm tài nguyên/cách tính quota. Chưa đủ đảm bảo seed 3, A3 và các evaluation còn lại trước tối14/10. A3 full bị chặn bởi pilot nonfinite; seed3 chưa chạy; GPU bổ sung/Colab vẫn chưa xác nhận.

Việc người dùng làm tiếp độc lập GPU: review/commit kết luận estimator để mở bảng bốn CSV bằng công cụ hiện có, hoặc duyệt hai hình estimator/modulation còn lại thuộc C41. Git/unblind do người dùng thực hiện; AI chưa chạy Git, training, inference hoặc tạo Kaggle version. Các split tổng hợp/robustness/OOD/RealBlur, LPIPS, oracle/shuffle, thống kê seed, latency/GMACs, ảnh TLC/crop, báo cáo/slide/demo, release A2 và khai báo CV vẫn cần evidence; khung hoặc config không thay việc hoàn tất.
