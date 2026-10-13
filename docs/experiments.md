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

## QĐ-08 Ổn định SFT-Map+Ch, SA-14 (chốt 2026-10-13)

Phương án "Nhánh channel đọc đặc trưng chuẩn hóa + dừng trước bước cập nhật hỏng". Loại hai phương án: "Giữ nguyên A3 gốc"
(đã phân kỳ) và "Hạ lr hoặc đổi batch riêng cho A3" (phá công thức chung của QĐ-04).

Căn cứ:
(1) pilot V1/357437244 dừng `aborted_nonfinite` ở bước 84;
(2) chẩn đoán V1/357568385: đầu ra không hữu hạn đầu tiên ở `encoders.3.18.conv3` (lượt 35); trước đó nhánh channel cho
    Δγ khoảng 2,25e10 vì nó nhận **đặc trưng thô** (đầu vào block), và hệ số (1 + Δγ) nhân dồn qua 28 block của enc3;
(3) trainer vẫn chạy `optimizer.step` thêm 50 bước trên gradient NaN (FP32 không có GradScaler để bỏ bước), nên toàn bộ
    tham số hỏng từ lượt 36.

Thay đổi:
(1) `model.chan_input: norm` (mặc định): nhánh channel đọc `norm1(inp)`, tức đặc trưng sau LayerNorm và trước điều biến.
    Giá trị `raw` chỉ giữ để tái hiện pilot hỏng. Áp dụng qua `apply_model_options` ở cả `train.py` và `evaluate.py`;
(2) `training.fail_fast_nonfinite: true`: loss hoặc chuẩn gradient không hữu hạn thì dừng **trước** `optimizer.step`,
    mã thoát 3, sự kiện `aborted_nonfinite` hoặc `aborted_nonfinite_grad`;
(3) các run đã khóa ở `faf4a06` không bị ảnh hưởng: chúng không có bước nào không hữu hạn (`nonfinite_run` luôn bằng 0),
    và kiến trúc của Baseline, Baseline-L, SFT-Free, SFT-Map không đổi.
Test: `test_a3_channel_branch_reads_normalised_feature`, `test_trainer_fail_fast_stops_before_the_bad_update`,
`test_trainer_fail_fast_on_nonfinite_gradient_with_finite_loss`.

Hệ quả: trong báo cáo SA-14 là "SFT-Map+Ch (QĐ-08)", một định nghĩa khác bản A3 ban đầu, train ở commit mới (ghi hash
khi chạy). Công thức QĐ-04 giữ nguyên: lr 1e-4, batch 8, FP32, 30.000 bước, seed 1, khởi tạo từ checkpoint công khai.
Phải qua pilot 1.500 bước (tiêu chí như pilot lần bốn) trước khi chạy full. Nếu chạy trên GPU khác T4 (Colab), tắt TF32
(`torch.backends.cudnn.allow_tf32 = False`, `torch.backends.cuda.matmul.allow_tf32 = False`) và ghi tên GPU.
Phép thử nhánh kênh (SA-14 so với SA-13 seed 1) phải ghi rõ khác commit và, nếu có, khác phần cứng.

## QĐ-09 Chỉ tiêu chính, so sánh nhiều lần và gộp seed (chốt 2026-10-13, trước khi đánh giá test tổng hợp)

(1) Chỉ tiêu chính: PSNR (float, TLC) trên `test_in`. Phép so sánh chính: Phép thử giám sát (SFT-Map so với SFT-Free).
(2) Chỉ tiêu phụ: các phép so sánh còn lại, các tập `test_ood`, `test_straight`, `test_curved`, GoPro thật, SSIM, PSNR vùng
    nhòe nặng, nhóm theo H. Báo cáo bằng cùng phương pháp, ghi rõ là phụ hoặc thăm dò.
(3) Hiệu chỉnh Holm trên họ phép so sánh lõi của mỗi tập: Phép thử giám sát, Phép thử tham số, Phép thử điều biến, và
    Phép thử nhánh kênh khi có SA-14. Luật verdict giữ nguyên (α = 0,05, ngưỡng 0,1 dB), áp dụng trên p đã hiệu chỉnh.
    Báo kèm p chưa hiệu chỉnh.
(4) Gộp seed: Phép thử giám sát dùng trung bình 3 seed cho từng ảnh, rồi trung bình theo video, rồi Wilcoxon; báo kèm
    chênh lệch từng seed. Các phép có Baseline hoặc Baseline-L (1 seed) dùng seed 1 của SFT-Free / SFT-Map, vì cùng seed
    thì cùng dòng dữ liệu train; trung bình seed chỉ là phân tích độ nhạy.
(5) Dòng tham chiếu: checkpoint công khai chưa fine-tune (mã SA-00) được đánh giá trên `test_in` và GoPro thật bằng cùng
    `evaluate.py`, chỉ để mô tả, không thuộc họ phép so sánh.
Cài đặt: `scripts/eval_queue.py` (seed 2/3, SA-00, SA-14), `analysis/eval_tables.py` (Holm, gộp seed, `verdict_raw` và
`verdict_holm`).

Công bố trung thực: QĐ-09 được khai báo **sau** khi đã mở số GoPro seed 1 (Phép thử giám sát -0,024 dB, p = 0,049 chưa
hiệu chỉnh, luật cũ ghi "SFT-Free tốt hơn") và **trước** khi có bất kỳ số nào trên test tổng hợp. Báo cáo nêu cả hai cách đọc
cho GoPro seed 1.

## Giới hạn đã biết (đưa vào chương Giới hạn của báo cáo)

(1) Split chính thức của GoPro có train và test chung video: 5 trong 10 video test (6 trong 11 chuỗi test) có đoạn khác
    nằm trong train: GOPR0384_11, GOPR0385_11, GOPR0868_11, GOPR0871_11, GOPR0881_11. Ảnh test tổng hợp được sinh từ
    frame nét của các chuỗi test, nên mô hình đã thấy cảnh tương tự khi train. Mức trùng như nhau cho mọi biến thể,
    không ảnh hưởng phép so sánh, nhưng làm số tuyệt đối lạc quan hơn. Giữ split chính thức để so được với các công bố.
(2) Val chỉ lấy từ các video có đúng một đoạn trong train và không có đoạn nào ở test, nên val không chung video
    với train lẫn test. Val chỉ có 2 video nên chỉ dùng để theo dõi và kiểm tra lr, không để chọn checkpoint.
(3) Thang vùng nhòe: ô vùng ở test vẫn lớn hơn ở train khoảng 1,3 đến 2,7 lần (QĐ-02).
(4) SSIM tính bằng `skimage`, khác cách cài của BasicSR, nên không so trực tiếp với số SSIM công bố (SA-00).
(5) Fine-tune chỉ trên nhòe tổng hợp (kernel tĩnh chập trong sRGB, trộn theo mặt nạ, nhiễu σ = 0,01; bán kính RMS tối đa khoảng
    6 px) làm mất bớt khả năng gỡ nhòe thật: trên GoPro test cả bốn mô hình seed 1 còn khoảng 28,36 dB (uint8) so với 32,87 dB
    của checkpoint công khai. Mức tụt có ở mọi mức nhòe, 163/1111 ảnh kém hơn ảnh nhòe đầu vào. GoPro thật vì vậy là phép thử
    ngoài phân phối của các mô hình đã fine-tune (QĐ-09: chỉ tiêu phụ). Phép kiểm tra pipeline K1 (sad-checks V3) đã loại trừ
    lỗi đo: trọng số công khai qua đúng đường `build_variant` + `convert_tlc` khớp SA-00 tới 4e-5 dB trên 24 ảnh.
(6) Bộ ước lượng E chưa được kiểm tra khả năng học trước khi train chính; sau 30.000 bước nó cho bản đồ gần hằng
    (xem "Kết luận về bộ ước lượng E").
(7) Sinh lại dữ liệu tổng hợp không cho cùng byte khi môi trường Kaggle ("Latest Container Image") đổi phiên bản thư viện:
    cùng code (`faf4a06` = `862da1d` ở phần sinh dữ liệu), val sinh lại ngày 13/10 có sha `9dd67922cef51b0f` thay vì
    `96ca0c00be35f399`. 40 file val gốc vẫn cho đúng `96ca0c00be35f399`, nên mọi số val khi train không bị ảnh hưởng. Từ đây
    test set được sinh MỘT lần trong notebook đánh giá và lưu lại làm output (không "sinh lại rồi so sha" ở mỗi phiên).
    Bổ sung 2026-10-13: kể cả với cùng numpy 2.1.3 / scipy 1.16.3, sinh trên Modal cho sha khác Kaggle ở `test_in`
    (`0a83c5daad9657cb` so với `13b027f23ec17c0f`) và `test_ood` (`978cc452c485da7f` so với `a6e491a2502d099c`), nhưng trùng ở
    `test_straight` và `test_curved` (một vùng). Phần chỉ chạy khi R >= 2 là sinh mặt nạ mềm (spline zoom, exp), nên khác
    biệt đến từ đường tính của thư viện số theo CPU. **Bộ test chính thức là bộ sinh trên Modal, lưu cố định trong Volume
    `sad-runs:/synthetic`, mọi mô hình đều được chấm trên đúng bộ file này**; sổ sha Kaggle chỉ còn giá trị lịch sử. Độ lớn
    khác biệt được đo bằng `modal_train.py::regen_val_diff` (sinh lại val40 trên Modal và so từng pixel với 40 file gốc).

## Kiểm tra trước đánh giá: sad-checks Version 3 (CPU, commit `862da1d`, 2026-10-13)

(1) K1, pipeline đánh giá: 24 ảnh GoPro test rải đều 10 video. Trọng số công khai qua `build_variant` + nạp như `train.py`
    + `convert_tlc` lệch tối đa 4,4e-5 dB so với SA-00; `final.pth` của SA-10 (CPU) lệch tối đa 5,2e-5 dB so với CSV GoPro
    đã kiểm chứng (GPU). Trung bình 24 ảnh: công khai 33,348 dB, SA-10 28,559 dB. **Đạt**: mức tụt GoPro là do fine-tune.
(2) P1: val sinh lại có sha khác (giới hạn (7)), nên K2 bị chốt chặn bỏ qua và chạy lại ở Version 4 trên 40 file val gốc.
(3) K3, EDA 500 mẫu: R cân bằng (126/127/133/114), s_r phủ đều 4 khoảng; sai số |đo - đích| của s_r trung vị 0,0145;
    ngưỡng H (QĐ-03) 0,1061 và 0,1795, ghi `data.h_bins: [0.11, 0.18]`; cạnh mảng vùng trung vị 69 px trên patch train và
    136 px trên khung đầy đủ, tỉ lệ 1,96 lần (giới hạn (3)).
(4) K4, test tổng hợp sinh trong phiên này (sổ sha tham chiếu): `test_in` 400 ảnh `13b027f23ec17c0f`, `test_ood` 100 ảnh
    `a6e491a2502d099c`, `test_straight` 50 ảnh `d6936aac831e864f`, `test_curved` 50 ảnh `ee6b893c9aa2c23a`.
Bằng chứng: `results/artifacts/kaggle/sad-checks/v3-run/` (zip 1.102.682 byte, MANIFEST 13/13 file khớp SHA-256).

## Chẩn đoán bộ ước lượng E (2026-10-13)

(1) K2, phép thử bộ ước lượng (sad-checks Version 4, CPU; 40 file val gốc sha `96ca0c00be35f399`; bundle mã đóng băng;
    `probe_estimator.py` sha `f761142a…`): E train riêng chỉ với L_map, 1.000 bước, **không có loss khôi phục**, vẫn sụp về
    hằng: lr 1e-4 Spearman 0,011 → −0,002, MAE 0,207 → 0,178, std(Ŝ) 0,0023; lr 1e-3 Spearman → −0,007, std(Ŝ) 0,0008.
    Loss train đứng ở khoảng 0,0224 = loss của bản đồ hằng. Giả thuyết "cạnh tranh gradient với loss khôi phục" bị bác bỏ.
    Bằng chứng: `results/artifacts/kaggle/sad-checks/v4-k2/` (MANIFEST 8/8 khớp).
(2) Chẩn đoán local trên CPU (`analysis/estimator_learnability.py`, cùng bộ sinh nhòe, ảnh nét lấy từ val40):
    nhãn đúng (Spearman giữa s và bán kính RMS của kernel = 1,00) và mức nhòe có trong ảnh đầu vào (một đặc trưng thủ công
    trên ảnh nhòe đạt Spearman −0,54 với s). E của đồ án không học được kể cả khi R = 1 (Spearman khoảng 0, loss không thấp hơn
    bản đồ hằng). Một biến thể thăm dò cùng độ rộng (chuẩn hóa đầu vào, GroupNorm, khởi tạo Kaiming, lớp đầu stride 1) học
    được: R = 1 Spearman 0,82; R từ 1 đến 4 Spearman 0,56 theo pixel, 0,70 theo patch; loss test 0,0142 so với 0,0205.
Kết luận: SFT-Map không thích ứng được theo vùng vì **thiết kế/khởi tạo của bộ ước lượng E** khiến nó không học được bản đồ,
không phải vì dữ liệu, nhãn hay cạnh tranh gradient. Chưa tách được yếu tố nào trong bốn thay đổi là quyết định.
SA-13 giữ nguyên làm kết quả của giao thức đã khóa. SA-14 dùng cùng E nên cũng chịu hạn chế này.
Thu seed 3 (2026-10-13): `xunanhv/sad-train-a12-seeds` Version 2 / scriptVersionId 357688314, COMPLETE. Archive
`results/artifacts/kaggle/357688314/original/remaining_training_wave_a12_seed3.zip` 556.912.598 byte, SHA-256
`5d8c42e9130db8d7b93e38c8042d99708d591649739e960a73b4fafed97fa3c2` (khớp giá trị in trong log phiên), CRC đạt, MANIFEST 22/22.
SA-13_seed3 và SA-12_seed3: returncode 0, sự kiện cuối `finished`, `train_done` 30.000 bước, không dừng sớm, commit `faf4a06`,
Tesla T4 / torch 2.11.0+cu128, 712 tensor hữu hạn; `final.pth` SHA-256 bắt đầu `6a112859ba5bf287` (SA-13) và `94973423c3de5800`
(SA-12). Train lõi: 8/9 run xong (thiếu SA-14). Registry chưa cập nhật cho seed 3.

## QĐ-10 Thí nghiệm thăm dò sau chẩn đoán E (chốt 2026-10-13, trước mọi run SA-32/33)

Đã duyệt kèm: SA-14 (QĐ-08) xếp thứ ba, chỉ chạy nếu chi phí đo ở pilot cho phép; cắt RealBlur-J, LPIPS và SA-26 (đưa vào
hướng phát triển); ngân sách Modal tối đa 30 USD credit, trần tài khoản 42,50 USD.

Phương án "Trần bản đồ thật + bộ ước lượng sửa", hai thí nghiệm post-hoc, seed 1, công thức QĐ-04 không đổi (30.000 bước,
lr 1e-4, batch 8, FP32, khởi tạo checkpoint công khai, cùng dòng dữ liệu seed 1, val40 gốc):
(1) SA-32 "SFT-Map huấn luyện với bản đồ thật" (`SA-32_a2_oracle_train.yaml`): SFT head nhận S thật thu nhỏ 1/4 khi train và
    khi val (`map_source: oracle`), lambda_map 0. Đánh giá bằng `--mode oracle`. Đo trần của ý tưởng, không phụ thuộc E;
(2) SA-33 "SFT-Map với bộ ước lượng sửa" (`SA-33_a2_estimator_v2.yaml`): chỉ thay E bằng `BlurMapEstimatorV2` (chuẩn hóa đầu
    vào, LayerNorm2d theo từng pixel, Kaiming, lớp đầu stride 1; không có thống kê toàn ảnh nên không lệch train/test như
    GroupNorm); lambda_map 0,1 như SA-13.
Cả hai không thay SA-13, không thuộc họ phép so sánh chính của QĐ-09; chạy trên Modal (L4), ghi GPU/commit.

Tiêu chí đọc kết quả, khai báo trước (chỉ tiêu `test_in`, PSNR, TLC):
(1) so với SFT-Free seed 1 (SA-12, cùng kiến trúc, cùng dòng dữ liệu) và Baseline seed 1 (SA-10);
(2) gọi là "có hiệu quả" khi cả hai điều kiện: chênh trung bình theo video > 3 lần độ lệch chuẩn giữa 3 seed của SFT-Free
    trên `test_in`, và CI bootstrap 95% theo video không chứa 0; ngược lại ghi "không phân biệt được với dao động giữa seed";
(3) SA-33 báo thêm chất lượng bản đồ (MAE, Spearman trên `test_in`) và SA-20/21 của chính nó.
Cổng trước khi tiêu tiền: C1 E bản sửa học được trên CPU (Spearman theo patch >= 0,5, R 1-4); C2 SA-12 seed 1 trên Modal tới
bước 2.500 lệch val <= 0,02 dB so với Kaggle; C3 pilot 1.500 bước đạt (không dừng sớm, val bước 1.500 >= bước 0; SA-33 có
val_spearman > 0,2 ở bước 1.500); C4 tổng chi phí dự tính <= 30 USD credit (thứ tự ưu tiên SA-32, SA-33, SA-14).
Kết quả cổng C1 (2026-10-13, CPU local, `estimator_learnability.py --models v2`, đúng module `BlurMapEstimatorV2`):
R = 1 Spearman 0,41 theo pixel / 0,73 theo patch; R từ 1 đến 4 Spearman 0,49 theo pixel / **0,75 theo patch** (ngưỡng 0,5),
loss test 0,0149 so với 0,0205 của bản đồ hằng. **Đạt.** E gốc cùng thiết lập: Spearman khoảng 0.
Kết quả các cổng trên Modal (2026-10-13, commit `c181ed2`, NVIDIA L4, torch 2.11.0+cu128; dữ liệu Volume `sad-runs`:
4.325 PNG hợp lệ sau khi thay 1 file bị Kaggle trả về trang lỗi 502 (`train/GOPR0374_11_00/sharp/000016.png`, tải lại
SHA-256 `a2bdc2d37cd18ea2743b02b124af1fed0ac1f79bb9309603184ee06f41abc0d7`), md5 checkpoint khớp DATA_INFO, val40 sha
`96ca0c00be35f399`):
(1) C2 nền tảng: SA-12 seed 1 trên Modal, val bước 0 = 26,656018 (Kaggle 26,656018), bước 2.500 = 32,537689 (Kaggle
    32,526283), chênh +0,011 dB <= 0,02. **Đạt**;
(2) C3 pilot 1.500 bước (cùng Modal, cùng seed, cùng lịch): SA-32 val 31,269 / 32,114 ở bước 750 / 1.500, **đạt**;
    SA-14 (QĐ-08) 31,079 / 31,813, không có bước không hữu hạn (bản A3 gốc dừng ở bước 84), **đạt**;
    SA-33 31,060 / 31,789, val_spearman 0,003 → 0,090 → 0,141, MAE 0,168 (< 0,177 của bản đồ hằng), **không đạt ngưỡng 0,2**;
(3) C4: 1,73-1,87 it/s trên L4 (data_frac 0), khoảng 4,8 giờ mỗi run 30k; ba run full nằm trong 30 USD.
Cổng bổ sung C3b cho SA-33 (khai báo 2026-10-13 sau pilot, TRƯỚC khi có số của run full): pilot nén lịch cosine vào 1.500
bước nên E chỉ có khoảng 500 bước ở lr cao; SA-33 chạy full nhưng **dừng nếu val_spearman ở bước 5.000 < 0,2**. Đây là điều
chỉnh cổng sau pilot, báo cáo kèm lý do. Kết quả: SA-33 ở bước 2.500 val_spearman 0,254, ở bước 5.000 **0,317, đạt C3b**,
chạy tiếp tới 30.000 (val PSNR 33,034 ở bước 5.000, ngang SFT-Free seed 1 Kaggle 33,001).
Theo dõi val (40 ảnh, chỉ để theo dõi) các run full trên Modal, bước 2.500 / 5.000: SA-32 33,004 / 33,564; SA-33 32,499 / 33,034;
SA-14 32,512 / 32,993. Đối chiếu Kaggle seed 1: SFT-Free 32,526 / 33,001, SFT-Map 32,553 / 33,049.
Cài đặt: `BlurMapEstimatorV2` (modules.py), `build_variant(..., estimator=)`, `make_loss(..., map_source=)`,
`val_metrics(..., map_source=)`, `scripts/modal_train.py`; test `test_estimator_v2_identity_at_init_and_no_image_wide_statistics`,
`test_oracle_map_source_feeds_the_true_map_in_loss_and_val`.

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
Hàng đợi `scripts/eval_queue.py` chia hai GPU, ưu tiên theo tầng (cập nhật 2026-10-13 theo QĐ-09):
(1) mọi mô hình và mọi seed trên `test_in`, checkpoint công khai (SA-00) trên `test_in`; các mô hình seed 1 trên GoPro thật;
(2) các mô hình seed 1 trên `test_ood`, `test_straight`, `test_curved`; SA-20 và SA-21 của SFT-Map seed 1 trên `test_in`;
    seed 2/3 và SA-00 trên GoPro thật;
(3) SA-20 và SA-21 trên ba tập tổng hợp còn lại; seed 2/3 trên ba tập đó;
(4) `test_in` nhiễu ×2, ×4 (SA-26), chỉ khi đủ đĩa;
(5) suy luận toàn cục cho Baseline và SFT-Map seed 1 trên `test_in` và GoPro thật (phụ lục QĐ-01).
Seed n > 1 ghi vào `results/eval/seed<n>/`. CSV đã có (đúng số dòng) được tính là xong, không bao giờ bị ghi đè; bốn CSV GoPro
seed 1 đã kiểm chứng được loại khỏi kế hoạch bằng `--exclude "SA-1[0-3]_eval_gopro_test_pred_tlc@1"`.
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

### Checklist hiện hành và tiêu chí hoàn tất (cập nhật sau 11:56 ngày 12/10)

| C-ID / công việc con | Trạng thái và ô nghiệm thu | Bắt đầu → kết thúc dự kiến | Phụ thuộc / tài nguyên / người thực hiện | Input → output / tiêu chí và evidence | Song song / khi bị chặn |
| --- | --- | --- | --- | --- | --- |
| C23 — bundle/notebook controls | [x] Đã đạt: bundle và Saved V1/357428270 đã dùng thành công | Worker preflight 12/10; log source đã thu | AI chuẩn bị/thu; người dùng khởi chạy | 57 payload input kiểm hash; evaluator/src/manifests khớp SA-13 | Giữ nguyên source bundle/notebook |
| C24–C26 — GoPro SA-10/11/12 | [x] Đã đạt trong phạm vi CSV GoPro seed 1; remote COMPLETE | 12/10: evaluator bắt đầu khoảng 10:41; GPU0 SA-11 kết thúc khoảng 11:10; worker log 1864 giây | T4 ×2; GPU0 SA-10 rồi SA-11, GPU1 SA-12 | Ba CSV 1.111 ảnh/run; queue returncodes 0; không đồng nhất elapsed Viewer với compute evaluator | Không chạy lại evaluation |
| C24–C26 — thu và kiểm tra local | [x] Đã đạt: ba CSV VerifiedLocal | Thu/QC trong phiên từ 11:12 ngày 12/10 | AI; đúng V1/357428270, official SDK | Archive 622513 bytes SHA-256 872667327f0a92470f1d378388f6d9d8fe36bf0b23204f2da97872d67711b5c3; CRC + manifest 31/31; ID/video/finite metrics/protocol đạt; receipt ở results/artifacts/kaggle/357428270/download-provenance.json | Registry chỉ notes SA-10/11/12; status Verified, evaluation_status Missing, evaluation_ready False; SA-13/14 giữ nguyên |
| C10–C14 — checkpoint seed 1 | [x] Đã đạt: SA-10/11/12/13 30000 bước và checkpoint VerifiedLocal; [ ] SA-14 full | Training source V2/355813018 và V3/355940350 | User đã chạy training/strict-load; AI thu/hash/đối chiếu | Receipt inventory hiện còn khớp toàn bộ; checkpoint là scope riêng, không Accepted toàn run | SA-14 pilot fail được lưu riêng, chưa có final30k |
| C10–C14 — chuẩn bị remaining training | [x] Input/notebook A1/A2 đã chuẩn bị và gắn đúng; [ ] nghiệm thu full runs | A1/A2 seed 2 V1/357522661 được gửi khoảng 10:05 ngày 12/10 | Người dùng Save Version; AI kiểm source/phase/input/GPU | PHASE=wave_a12_seed2, GPU0 SA-13_seed2/GPU1 SA-12_seed2; 30k/run, fresh pretrained, FP32 | A3 vẫn bị chặn riêng; seed 3 đã gửi V2/357688314, SDK RUNNING tại23:37 |
| C10–C14 — pilot SA-14 | [ ] Bị chặn: V1/357437244 aborted_nonfinite step84; bằng chứng thất bại VerifiedLocal | Lượt pilot 12/10 đã kết thúc; trainer wall108,135 giây | User đã khởi chạy; GPU0, FP32/config khóa | Archive 4338bytes; CRC/manifest9/9 đạt; không final.pth/last.pth/train_done; không có số đo throughput đủ tin cậy | Chẩn đoán nguồn nonfinite trước pilot/full mới; không coi là OOM hoặc tự đổi batch/LR |
| C10–C14 — chẩn đoán SA-14 | [x] Đã đạt thu và VerifiedLocal bằng chứng diagnostic V1/357568385; [ ] nguyên nhân/ổn định hóa/full30k | Remote COMPLETE; Viewer173,4s, worker133,371s, không dùng làm throughput full | User chạy; AI official SDK transfer/QC; GPU0 | Archive717511bytes SHA256b62fde4663f21fefebedeb13ac135aab903cd35634e41dcb2e2a28b83199b1df; CRC/manifest11/11;846 observations;source4/4/config7fields khớp | First nonfinite attempted-step35 tại encoders.3.18.conv3; bad optimizer contamination sau35, abort counter84; SA14 full vẫn bị chặn |
| C10–C14 — full A1/A2 seed 2 | [x] Đã đạt training30000 và checkpoint VerifiedLocal trong scope hash/finite/strict-load/resume/log; [ ] smoke inference/evaluation | V1/357522661 COMPLETE; GPU0A2 wall34034,544s/GPU1A1 wall34025,124s (~9,45h/run) | User training; AI SDKtransfer/manifest/stateQC, khôngforward | Archive556914349bytes SHA25634d7ce6b8f8b5565f935a22c9ae48597bc14aa1f3a161a986402685adc72deb9;manifest22/22;712weight+2136optim tensors/run finite;final=last-model;step30000 | Registry append2rows giữ5rowscũ; statusVerified scoped,evalMissing/evalreadyFalse; khôngAccepted |
| C10–C14 — tài nguyên bổ sung | [x] SDK23:37: used16,977058/30h,reserved0,unused13,022942h; [ ] GPU bổ sung chưa xác minh | Snapshot12/10 lúcseed3RUNNING | Quota nền tảng, khôngphysicalGPUhours | Seed2wall đo~9,45h/run; seed3 chưa cóthroughput/steps hiện hành | Không coi phép trừ dự báo quota là sốđãđo; chưa bảo đảmSA14full/evaluation trước tối14/10 |
| C10–C14 — A1/A2 seed 3 | [ ] Đang chạy: V2/357688314 SDK RUNNING; số bước worker chưa xác minh | Kiểm tra12/10 lúc23:37 (+07:00); chưa có giờ bắt đầu compute chính xác | User đã Save & Run All; GPU T4 ×2, quota chưa dùng13,022942h | Source/input/PHASE seed3 đã xác minh; GPU0SA13seed3/GPU1SA12seed3; API log/output0, failure trống | Chờ log/steps vàcompletion đểthu/QC; chưaFinishedRemote/VerifiedLocal/Accepted |
| C51 — rà soát commit/push | [x] Đã đạt commit/push mốc hai file docs/experiments.md và scripts/evaluate.py; [ ] tool/notebook/registry untracked và backup | User output xác nhận main→origin/main tại130a870 ngày12/10 | Git do người dùng thực hiện; AI chỉ review/ghi tiến độ | Commit130a870: 2files,65insertions/11deletions; push c273b66..130a870 tới VuAnh245/spatial-adaptive-deblurring; status main...origin/main không ahead/behind | 13file untracked theo status trước push chưa thuộc commit; results/checkpoints vẫn bị ignore, backup độc lập chưa đạt |
| C52 — LPIPS AlexNet | [ ] Chưa làm: chưa có metric/weights-version/runtime verified | Sau khi có input/giao thức/triển khai được chấp thuận | User chạy evaluation; AI chuẩn bị/thu/QC | Giữ đúng prediction TLC/FP32 và normalization theo metric đã chốt | Không suy LPIPS từ CSV PSNR; không đổi giao thức để điền số |
| C54 — phát hành checkpoint A2 | [ ] Chưa làm: có checkpoint local, chưa phát hành | Sau nghiệm thu scope và user quyết định publish | User phát hành/Git; AI chuẩn bị evidence | Checkpoint/seed/config/commit/protocol/license/hash/hướng dẫn phải đi kèm | Private Kaggle hoặc file local không tự là checkpoint đã phát hành |
| C55 — khai báo dùng chung với CV | [ ] Chưa đủ bằng chứng: khung mô tả có, chưa đối chiếu repo CV | Song song phần báo cáo | User cung cấp repo CV; AI đối chiếu source/version/license | Không coi README/CORE_VERSION/CHANGELOG là đã chứng minh đồng nhất hai repo | Giữ phần nghiên cứu DL và phần dùng chung được khai báo rõ |
| C41 — đường cong và diagnostic figure | [x] Dữ liệu/provenance và QA hiển thị đủ6panel training, hai HTML estimator/modulation; [ ] export và chú thích cuối | User cung cấp hai screenshot đủ6panel training trong phiên ngày12/10 | Source/hash giữ nguyên; đối chiếu log JSON-lines có300training/13valrecords mỗi run, không inference | Trục/legend sáu panel đọc được; gradient norm ghi trước clip; cuốiSA10val33,791469/SA13val33,803355,SA13rho0,047614 | Footer còn thông tin SA11/12 hidden đã cũ sau gate; tiêu đề có dấu? và tooltip/cắt nguồn cần xử lý khi export; chưa QA PNG/PDF hoặc toàn C41 |
| C43 / C53 — khung báo cáo và bảng giả thuyết | [x] Có bản nháp phương pháp/thảo luận sơ bộ và bảng GoPro bốn model seed1 đã kiểm chứng; [ ] bản báo cáo/slide cuối và đa seed | Đã bổ sung công thức/đối chứng và diễn giải sau QA6panel ngày12/10 | AI cập nhật notebooks/03_results.ipynb; user review/report | Kết luận estimator trước mở gate giữ nguyên; bảng 1111 ảnh/model và 4 comparisons có source/hash | Không coi dàn ý notebook là báo cáo hoặc slide cuối đã hoàn tất |
| C53 — kết luận estimator / gate | [x] Đã đạt: E commit a8cb0b2, gate mở và bảng bốn model seed1 kiểm chứng; [ ] thống kê ba seed/kết luận cuối | User chạy UNBLIND ngày12/10, exit0; AI kiểm bảng sau đó | Gate đã đạt tại thời điểm xuất bảng; AI không Git | Summary mean khớp bốn CSV, 4 comparisons khớp stats.py; scope GoPro seed1 | Kết luận E đã khóa giữ nguyên; mọi lần UNBLIND tiếp theo cần commit tiến độ mới để file sạch |
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

### GoPro seed 1 — bảng UNBLIND đã kiểm chứng (12/10/2026)

Gate được công cụ xác nhận tại commit a8cb0b2, 2026-10-12T11:31:26+07:00. Cả bốn CSV đã VerifiedLocal, cùng 1111 ID, TLC/FP32 và evaluator/src/manifests. Toàn bộ mean và bốn phép so sánh đã được đối chiếu lại bằng CSV và hàm thống kê hiện có, không chạy model mới.

| Model / run | PSNR float (dB) | PSNR uint8 (dB) | SSIM skimage | Ảnh |
|---|---:|---:|---:|---:|
| A0 / SA-10_seed1 | 28.3699 | 28.3649 | 0.8717 | 1111 |
| A0+ / SA-11_seed1 | 28.3671 | 28.3622 | 0.8717 | 1111 |
| A1 / SA-12_seed1 | 28.3700 | 28.3651 | 0.8717 | 1111 |
| A2 / SA-13_seed1 | 28.3485 | 28.3436 | 0.8713 | 1111 |

Summary gộp theo ảnh; comparisons gộp PSNR float theo 10 source-video bằng implementation video_of(), cho mỗi video cùng trọng số. Vì thế diff comparison không nhất thiết bằng phép trừ hai mean ảnh. Phép thử chính A2-A1: -0.0236103 dB, bootstrap 95% CI [-0.0575773,-0.0017533], Wilcoxon p=0.048828125 chưa hiệu chỉnh; A2-A0+: -0.0193319 dB, p=0.048828125; A1-A0: +0.0015343 dB, p=0.625; A2-A0: -0.0220760 dB, p=0.048828125. Tất cả mới seed 1; hiệu ứng nhỏ, p của ba phép gần 0.05, chưa kết luận ba seed hoặc chấp nhận toàn dự án. Giữ nguyên phép kiểm/ngưỡng đã khai báo, không sửa phương pháp để thay nhãn theo kết quả.

Các nan ở psnr_local_high/map_mae/map_spearman là metric không có ground-truth smap/vùng nhòe trên GoPro; core PSNR/uint8/SSIM đều hữu hạn. H groups vẫn bị chặn bởi data.h_bins chưa có; không tạo H từ CSV GoPro. So với tham chiếu SA-00 TLC uint8 ~32.870 dB trong docs, cả bốn model fine-tune thấp khoảng 4.5 dB, không chỉ A2. Đây là phát hiện cần rà phần chung fine-tune/data/protocol, chưa chứng minh lỗi implementation, quên kiến thức hay SFT thất bại. SSIM SA-00 tập con223 khác giao thức, không so trực tiếp.

Bằng chứng: results/eval/tables/summary.csv SHA-256 4ade4d519c33af7f8f9f28af03de166df4404b0b46640b36d1608569bd0777ab; comparisons.json SHA-256 f4834498fc6a2e3f9a6f5d14f47c407deb15b0995e66830d71145068c407403a; README.txt SHA-256 934823349c1d125bc900c628bcd74e15d00bc9b588f584132515eb4c697fd334. Source receipts: results/artifacts/kaggle/357428270/download-provenance.json và 357357176/download-provenance.json.

- 12/10/2026 lúc 11:44 (+07:00): A1/A2 seed2 V1/357522661 tiếp tục được SDK báo RUNNING; chưa có output/log công bố. C53 chỉ nghiệm thu gate và bảng seed1; C43 bổ sung số thực vào khung báo cáo. Seed3, A3, các split/LPIPS/cơ chế/cost/qualitative và bản báo cáo/slide cuối còn thiếu. Việc ghi tiến độ sau commit a8cb0b2 tạo sửa mới trong docs/experiments.md; không làm mất tính hợp lệ của bảng đã xuất, nhưng cần commit cập nhật trước lần --unblind tiếp theo.

### Bàn giao kiểm tra trước commit/push — 12/10/2026, sau 11:56 (+07:00)

Kiểm tra SDK đúng version_label=v1: controls xunanhv/sad-eval-gopro-controls V1/357428270 COMPLETE, gói và ba CSV đã VerifiedLocal; xunanhv/sad-train-a12-seeds V1/357522661 RUNNING, chưa có log/output API công bố và chưa thu checkpoint seed2. Quota snapshot used8,273493/30h, reserved0, unused21,726507h; không giải thích thay đổi quota thành thêm credits hoặc hứa đủ trước tối14/10.

Đối chiếu lại 5 receipts (355813018,355940350,357357176,357428270,357437244), tổng483 entry inventory: full SHA-256/dung lượng/tồn tại đều khớp, không mismatch. Bốn CSV đã promote trong results/eval vẫn khớp hash receipts; ba bảng UNBLIND vẫn khớp hash đã ghi. CSV/JSON registry đồng nhất 5 hàng/32 cột; core status Verified cho SA-10/11/12/13, aggregate evaluation_status Missing/evaluation_ready False, SA-14 main vẫn Missing vì chưa có full checkpoint. Scope pilot SA-14 VerifiedLocal chỉ là bằng chứng thất bại, không training thành công.

Bốn notebook đã chuẩn bị/khung báo cáo đạt nbformat và compile từng code cell; không thực thi notebook/model trong kiểm tra này. Source/evaluator/config/manifests không bị sửa; không strict-load lại artifacts không thay đổi, không chạy audit main, training/evaluation hoặc Git. Các lần thu Saved Version hiện hành thực hiện remote completion → nguồn version/run chính xác → transfer archive .part/exclusive → local existence/size/hash/CRC → manifest/content/protocol QC → cập nhật metadata scoped sau CSV/JSON round-trip. Giữ originals/receipts và E-conclusion trước unblind; các mốc lịch hoặc trạng thái Running không dùng làm chứng cứ hoàn tất.

Đã ghi tài liệu tại docs/experiments.md, notebooks/03_results.ipynb, registry CSV/JSON và receipts theo phiên bản; đây là bằng chứng local, chưa chứng minh các file đã nằm trong Git remote. .gitignore đang có /results/, /checkpoints/, *.pth và *.pt; chưa có .gitattributes trong repo. File evidence/source notebook mới dưới results không tự được stage/push bởi git add thông thường. Trước push cần user cung cấp status/diff-stat/check-ignore, chọn file evidence nhẹ cho Git và bảo đảm backup độc lập/checkpoint release theo kế hoạch; không tự đổi ignore, force-add, commit/push hoặc phát hành. Backup ngoài máy hiện chưa có bằng chứng.

Ưu tiên tiếp: (1) rà đúng phạm vi commit mốc seed1 đã verified và tài liệu, (2) chẩn đoán nonfinite A3 và nguyên nhân PSNR chung ~28,36 dB sau fine-tune so tham chiếu SA-00 ~32,87, (3) thu seed2 ngay khi hoàn tất trong phiên đang hoạt động, chốt seed3 theo quota thực tế, (4) các test tổng hợp/robustness/OOD/RealBlur, LPIPS, oracle/shuffle, cost/ảnh định tính/report-slide-demo/tái lập/backup còn thiếu. Không tự giảm scope hoặc Accepted toàn dự án; 13/10 là ưu tiên và tối14/10 là hạn toàn bộ có rủi ro.


- 12/10/2026: C51 đạt phần commit/push mốc seed1 theo output Git do người dùng cung cấp. Commit130a870 trên main chứa docs/experiments.md và scripts/evaluate.py (65 insertions,11 deletions); push c273b66..130a870 thành công tới https://github.com/VuAnh245/spatial-adaptive-deblurring.git; status main...origin/main không ahead/behind. Các tool/report/notebook untracked và artifacts bị ignore chưa được đưa lên remote, chưa nghiệm thu tái lập/backup hoặc toàn C51. AI không thực hiện Git. Ghi mốc này tạo sửa mới trong docs/experiments.md sau commit130a870; giữ kết luận E và các snapshot lịch sử nguyên trạng.


- 12/10/2026 khoảng13:30: tiếp tục nguyên phạm vi theo yêu cầu đẩy nhanh. Official SDK pinned v1: A1/A2 seed2 V1/357522661 RUNNING, chưa log/output; controls V1/357428270 COMPLETE đã thu/QC, không tải hoặc chạy lại. Quota used9,681978885/30h, reserved0, unused20,318021115h, không đồng nhất quota với tổng GPU-hours. Chuẩn bị results/artifacts/kaggle/sad-train-sa14-diagnostic.ipynb (21667bytes, SHA256411ff8fe6e919b845c71c5b2d94c8f673f3a51924fa30e8aa3070435a892cb51), worker SHA256de3c8dfe05f49daff437e582318fca1310358d153f42263ba1279948c8c8834d. Notebook dùng nguyên source V2/355813018 và input đã verified, seed1 fresh pretrained, target1500/val750/ckpt750 như pilot, chỉ thêm quan sát trong process cho150 bước đầu; guard/loss/optimizer/config không đổi. Cap60 phút không phải thời gian dự báo; instrumentation làm thay đổi timing, chưa thực thi. Runtime/hook behavior chỉ được xác minh khi người dùng chạy Saved Version; đây chưa là SA-14 ổn định hoặc checkpoint full30k. Giữ notebook pilot cũ và lỗi V1/357437244 nguyên trạng.

- C51 review bổ sung: sáu Python tool untracked đều parse/compile đạt; ba CSV report parse đúng32/90/5rows với10/3/32fields. Đây là static/schema checks, chưa chứng minh toàn bộ hành vi runtime hoặc sẵn sàng stage mọi file. make_testsets có thao tác unlink/rmtree và audit main ghi lại registry/snapshot nên không tự thực thi. C43/C53: cập nhật bảng giả thuyết trong notebook báo cáo theo kết quả GoProseed1 hiện có; phần seed2/3/SA14/cơ chế/LPIPS/chi phí/RealBlur vẫn pending.

Lịch điều kiện từ13:30 ngày12/10: khởi chạy diagnostic riêng ngay khi user gắn input; thời gian chờ Kaggle chưa đo và worker cap60 phút, chỉ GPU0 có training. Seed2 tiếp tục job cũ, chưa đủ step/it/s để chốt giờ kết thúc; sau completion mới intake/QC rồi chọn wave seed3 theo quota, một cặp theo log cũ8,9–9,8h là ước lượng. Chiều12/10 chuẩn bị báo cáo/hình song song và xử lý diagnostic khi có evidence; 13/10 ưu tiên các job dài/test còn thiếu; tối14/10 vẫn là hạn toàn bộ có rủi ro, chưa tự giảm SA14/seed/dataset hoặc gán Accepted.

- 12/10/2026 13:37:30 (+07:00): C10–C14 diagnostic setup được SDK xác minh: xunanhv/sad-train-sa14-diagnostic có3 code cell khớp nguyên source notebook local, PHASE diagnostic_sa14, enable_gpu=True/NvidiaTeslaT4, đúng input sad-train-remaining-input và sad-data; Saved Version count0. Đây là setup đã đạt, chưa preflight runtime/hash, chưa training/diagnostic remote, chưa có artifact để thu. User tiếp theo Save Version→Save & Run All, GPU for this session, rồi gửi exact Saved Version URL. Không tạo version hoặc thực thi notebook bởi AI.

- 12/10/2026 13:40:53 (+07:00): C10–C14 diagnostic launch confirmed. User URL https://www.kaggle.com/code/xunanhv/sad-train-sa14-diagnostic?scriptVersionId=357568385 xác định V1/357568385; official SDK version_label=v1 xác nhận Saved Version1,3 code cell khớp notebook local, GPU enabled/NvidiaTeslaT4, đúng input remaining-training-input và sad-data, RUNNING không failure message. API chưa công bố log/output, chưa có preflight/training-step/diagnostic-content proof và chưa có artifact để thu. User screenshot Logs hiển thị T4×2 và debugger warnings; đó chưa phải lỗi training. A1/A2 seed2 V1/357522661 cũng vẫn RUNNING. AI không thực thi notebook hoặc tạo version; không gán FinishedRemote/VerifiedLocal/Accepted từ trạng thái Running.


### SA-14 numeric diagnostic — V1/357568385, evidence VerifiedLocal ngày12/10

Nguồn chính xác: https://www.kaggle.com/code/xunanhv/sad-train-sa14-diagnostic?scriptVersionId=357568385. Remote COMPLETE do diagnostic capture hoàn tất; training process returncode3/IncompleteRemote. Archive local results/artifacts/kaggle/357568385/original/remaining_training_diagnostic_sa14.zip,717511bytes,SHA256b62fde4663f21fefebedeb13ac135aab903cd35634e41dcb2e2a28b83199b1df; remote log hash khớp transfer. CRC và verify_manifest.ps1 đạt11/11;846 dòng numeric JSONL parse đủ. Worker hashde3c8dfe05f49daff437e582318fca1310358d153f42263ba1279948c8c8834d khớp notebook/source; đã thu source/log và4frozen source file, byte khớp bundle gốc. Bảy field config experiment_id/seed/model/data/training/loss/git_commit khớp pilot V1/357437244; pretrained val0PSNR26,656017615377813. Receipt results/artifacts/kaggle/357568385/download-provenance.json,SHA256b61f34908e69eb3bf3159d652acb12d007a44cec048edc1c72476e87094a1a3b,19inventory entries. Không final.pth/last.pth/train_done.json, không VerifiedLocal checkpoint hoặc Accepted. Registry mainSA14 vẫnMissing và lịch sử không ghi đè.

Quan sát: bước34 loss0,0210317, gradient norm trướcclip1,676999/sauclip~1,0, input/parameter/gradient hữu hạn. Lượt forward35 input blur/sharpFP32 NCHW8×3×256×256 trong[0,1],smap hữu hạn; toàn748parameter tensors trướcforward/optimizer hữu hạn. Bên enc3, outputblock17 absmax~1,76e15; norm1block18 vẫn hữu hạn max~4,54 nhưng channel delta_gamma~2,25e10,bias~2,23e10. SimpleGate vàSCA trướcconv3 lên~3,52e22/~3,44e22; module output không hữu hạn đầu tiên encoders.3.18.conv3. Charbonnier NaN nhưng L_map vẫn0,0323069. Gradient748tensors không hữu hạn; clip không phục hồi; optimizer vẫn cập nhật và748parameter tensors thành không hữu hạn saubước35. Từlượt36 parameter đãhỏng. Trainerabortconsecutive51 khi self.step84 (85attempts). Không dấuhiệuOOM; peakallocated11,096GiB/reserved12,197GiB.

Diễn giải giới hạn: log ủng hộ giả thuyết khuếch đại trong chuỗi encoder/channel nhận feature thô và xác minh sự lan truyền do optimizer vẫn cập nhật gradient NaN. Chưa trực tiếp hook phép nhân x*SCA trướcconv3 hoặc thực hiện can thiệp đối chứng; vị trí output NaN không chứng minh convolution implementation sai hoặc channel là nguyên nhân duy nhất. Chưa dùng timing quan sát để dự báofull run. A1/A2seed2 V1/357522661 vẫnRUNNING ở lầnSDKkiểmtra sauintake, chưa có checkpoint mới để thu.

Phương án sửa tối thiểu để người dùng review, CHƯA ÁP DỤNG: (a) trong ModulatedNAFNet.run_stage, đổi duy nhất đầu vào channel từ self.chan[key](inp) sang self.chan[key](xn), với xn làfeature sauLayerNorm trướcđiềubiến; giữ khởi tạo/modelsize/LR/loss/batch/FP32. Đây là sửa định nghĩaA3, cần ghi quyết định vàsource/patchhash riêng, không gọi cùng frozen snapshot cũ. (b) guardtrainer kiểm loss vàgradient finite trướcoptimizer, fail-fast vàghi event trướcbadupdate; giữ mọi bước hữu hạn, không skipbatch rồi tuyên bốfullcomplete. Guardb sửa cơ chế lantruyền, không tự chữa lỗiforward. Chưa thaymodel/trainer/config/manifest; AGENTS yêu cầu người dùng chấp thuận thayđổi thuật toán/guard và khởi chạy pilot mới. Không tự thayA3gốc bằng biếnthểổn định hoặc fullrun trướcpilot.

C41 visualQA: screenshot estimator/calibration/MAE vàmodulation3panel cho thấy axes/legend/provenance đọc được, nguồn nội dungkhớp bằngchứng. Đã đạt phần hiển thị haiHTML; ảnh đầu cóTranslate popup/tooltip nên chưalàexportcuối. TrainingHTML chỉđãthấy4panelđầu,2panelcuối/PNG-PDF/reportlayout vẫnpending. GiữHTMLgốc, khôngghiđè.


Patch đề xuất để review (chỉ văn bản, chưa áp dụng vào source):

```diff
--- src/spatial_deblur/modulation/variants.py (A3 conditioning)
+++ proposed A3 stability pilot
@@
-                    dgc, bc = self.chan[key](inp)
+                    dgc, bc = self.chan[key](xn)
--- src/spatial_deblur/engine/trainer.py (loss/gradient safeguards)
+++ proposed fail-fast guards
@@ after step_finite is calculated, before parts/backward
+                if not step_finite:
+                    self.abort_reason = "nonfinite_loss"
+                    self.log_event("aborted_nonfinite", attempt=self.step + 1, phase="loss", consecutive=1)
+                    log.close()
+                    return False
@@ after self.scaler.unscale_(self.opt), before gradient clipping
+            if any(not bool(torch.isfinite(p.grad).all())
+                   for p in self.model.parameters() if p.grad is not None):
+                self.abort_reason = "nonfinite_gradient"
+                self.log_event("aborted_nonfinite_gradient", attempt=self.step + 1)
+                log.close()
+                return False
@@ after existing clip_grad_norm_, before scaler.step
+            if not bool(torch.isfinite(grad_norm)):
+                self.abort_reason = "nonfinite_gradient_norm"
+                self.log_event("aborted_nonfinite_gradient_norm", attempt=self.step + 1)
+                log.close()
+                return False
```

Điều kiện nếu được chấp thuận: launcher/bundle riêng dựa trên source gốc và patch/hash ghi rõ, pilot SA-14_seed1_stability1500 khởi tạo pretrained mới; LR1e-4/batch8/FP32/loss/val40/target1500 không đổi. Các run/receipt cũ và job seed2 đang chạy không thay đổi. Guard không bỏ qua batch xấu để tích đủ số bước; gặp lỗi thì abort trước optimizer. Pilot ổn định mới cho phép lập lịch full30k; chưa gọi A3 gốc ổn định hoặc tất cả kết quả cùng frozen code cũ.


- 12/10/2026 19:20–19:23 (+07:00): kiểm officialSDK pinv1 đủ5SavedVersion. A1/A2seed2 V1/357522661 RUNNING/nofailure/APIoutput0/log0; Chrome signed-in pinnedLogs đãcópreflightpass93bundle/val40/runtime/pretrained vàFRESH RUNS SA13seed2GPU0/SA12seed2GPU1. Last visible process rows tại31226,4s/8,6717h đềureturncode null; Viewer33551,5s (~9,32h), chưa thấy actualoptimizersteps/workerlogs. Lưu source log hiểnthị ở results/artifacts/kaggle/357522661/supplied/live-browser-log-2026-10-12T12-23-14-543Z.txt; scope browser evidence, không checkpoint hoặc completedSavedOutput. DiagnosticV1/357568385COMPLETE,controlsV1/357428270COMPLETE,recoveryV1/357357176COMPLETE đềuđãVerifiedLocal đúngscope; pilotV1/357437244ERROR cóevidenceVerifiedLocal, khôngtrainđạt. Khôngtải/checks lặp artifacts đãgiữnguyên.

Quota19:21 used15,7221647147/allowed30h,reserved0,unused14,2778352853h. GiữA1/A2seed2 chạy nguyênrecipe; chưa có lýdo hủy/restart chỉvìseed1GoProkém. Seed3 tiếptục kếhoạch sauseed2QC/quota, chưa tuningA2 trướcprobe_E/oracle/shuffle vàràgapSA00. A3 cầnpilot ổnđịnh hóa theo diff đãreview trướcfull; source/model/guard vẫn chưa đổi vì user chưa chấp thuận cụthể. TheoelapsedViewer, cap11,5h vào khoảng21:30–21:40 ngày12/10 chỉlàshutdownlimit của notebook, khôngphải lời hứa30khoàntất; nếu thiếu target phảithuQCcheckpoint interrupted vàreviewresume cùngrun riêng. Seed3 cặp8,9–9,8h làước lượngoldlogs; còn14,28hquota chưađủ cơsởđảmbảoA3full/evaluations/scope14/10, nguồnGPUbổsung vẫnpending. AI khôngcancel/joblaunch/Git/sửaalgorithm.


### Probe E độc lập — chuẩn bị bằng script hiện có, chưa chạy

12/10/2026: tiếp tục phần cơ chế/estimator trong checklist hiện có, không thayA1/A2/A3 để dò cấu hình. Chuẩn bị results/artifacts/kaggle/sad-probe-estimator.ipynb (20161bytes,SHA2569857f61469447f56f415d1cff05410ffdf778b21c8396275dc76f8420e5af66c). Script analysis/probe_estimator.py giữ nguyên byte,SHA256f761142a9d4d2ad025a644198c5e6609b3fa3f20f93b4361069941f85e49bac0; notebook embed vì bundle cũ chỉ cóanalysis/pilot_report.py, không bắt upload lại dataset. NB/cells/embeddedworker/script compile và source byte-equality/JSONroundtrip đạt, chưa execute.

Inputs reuse xunanhv/sad-train-remaining-input + xunanhv/sad-data;93bundle/runtime/val40 gốc preflight như cũ. Probe chỉtrain freshBlurMapEstimator seed1 trên generatorseed1001 theo script;1000steps,batch8,LRfromconfig1e-4,FP32,val0/250/500/750/1000,workers2,devicecudaGPU0;GPU1 không cótrainjob. KhôngpretrainedNAFNet/resume/checkpointA2 reuse, không đổi generator/valdata/loss/estimatorcode. Output riêng /kaggle/working/remaining_training_probe_estimator_seed1.zip và sad_remaining_probe_estimator_seed1/results/probe_E_seed1_1000; log/summary/runtime/source/worker/MANIFEST. Cap30min làlimit, không thời gian dự báo; capture thấtbại trướcassert. Scriptgốc khônglưucheckpointE, nên scope làdiagnosticmetrics, khôngtrainingcore/Accepted.

Interpretation đãkhai báo trước: rho tăng chỉ hỗ trợ khả năng học trênsetup probe, không tự chứng minh nguyênnhânA2; rho thấp/undefined không chứng minh targetunlearnable. val_s_hat_std của script làảnhvalcuối, khôngmean40. Khi API/Chrome xácnhậncompletion trongphiênactive, thu đúngversion vàQC5logsteps/finiteMAE/provenance; undefinedrho ghi rõ thayvìgán0. ResourceURL tên sad-probe-estimator mớiđềxuất,chưa notebook remote verified; userimport/SaveVersion. Seed2 SDK vẫnRUNNING/output0 lúc chuẩn bị; chưaapplySA14patch, khôngAItraining/eval/Git.


### A1/A2 seed2 — training FinishedRemote, checkpoint VerifiedLocal (12/10/2026)

Source https://www.kaggle.com/code/xunanhv/sad-train-a12-seeds?scriptVersionId=357522661, V1/357522661 COMPLETE; exact3codecells khớp notebook chuẩn bị. SA12seed2/A1GPU1 vàSA13seed2/A2GPU0 đềureturncode0/finished30000/train_done30000,cap_reachedFalse. OfficialSDKtransfer selectedarchive bằng.part/exclusive tới results/artifacts/kaggle/357522661/original/remaining_training_wave_a12_seed2.zip (556914349bytes,SHA25634d7ce6b8f8b5565f935a22c9ae48597bc14aa1f3a161a986402685adc72deb9),hash/size khớpremote log,trướcpromote;CRC/safeZIP extraction/manifest22/22đạt. Originalsaved-source vàsession-log giữnguyên,khôngmirror toàn99Outputcopies.

CheckpointQC bằngCPU đọcstate vàstrict-load dùngsourcefrozen từ remaining-training-input.zip/src,torch2.13.0+cpu; khôngforward/backward hoặcmodel evaluation. Configmodel/data/training/loss/git_commit khớpseed1, chỉseed2/provenance khác.712weighttensors và2136optimizer tensors/run finite;final khớplast.model, resume cómodel/opt/sched/scaler/step/rng vàstep/scheduler30000. Mỗirun300curvesrecords/13valrecords/60diagrecords,loss/grad_norm/valmetricsfinite,nonfinite_run0,khôngabort/stopped. Workingvariants/nafnet khácbytefrozen doCRLF/LF,normalizedsourceequal; strict-load lấy trực tiếpfrozen module,khôngstrict-loadlạiseed1.

SA12seed2 finalSHA256a72ffc159139d41f48bbd4378dfc6bdb82886d825c6a678d5ca6e12ea3df3efb,lastSHA256187d233c329724d4bb4ea5bb425b9d870d409de856a926281d222f98feae7f93,configSHA25695e8cabd1b60efb65d774aa1178a03b920b25f4e90050991a6d456b16ec1fa1c. Val40 cuốiPSNR33,8026487/mapMAE0,6622735/rho-0,0055208;wall34025,124s. SA13seed2 finalSHA256c1875c0052cc604d4b739bbbc989ca53e52f533f4b6c825dc09a4ade06b0c014,lastSHA25676677335d3f14def8b435dec7eb00a3962264d2f366bb8ce5f854bebb0989daa,configSHA2563c997432a3e002e81ec91d3d22695768dc4603d05ef07698216b1e46d964367f. Val40 cuốiPSNR33,7997188/mapMAE0,1768159/rho0,0377062;wall34034,544s. Đây làvalidationmonitoring,khôngGoProtest hoặc kếtluận nhiều seed; estimatorA2rho vẫn thấp,probe_E/oracle/shuffle còn cần.

Receipt results/artifacts/kaggle/357522661/download-provenance.json,SHA2566aa5fd929ce308c1e9ba8966fea30d55b534019f05d3ada8c6b1afe56515262c,26inventory entries. SauQCđạt, dùng helperwrite_csv/write_json cũ vàCSV/JSONroundtrip trướcwrites;registry appendđúng2rows thành7rows/32fields,nămrowscũ giữnguyên kể cảSA14. NewstatusVerified,notesCheckpointVerifiedLocalscope,trainingFinishedRemote/evaluationMissing/evaluation_readyFalse; inference smoke/evaluation chưa chạy, khôngAccepted. Cácsnapshots/index cũ giữnguyên, khôngauditmain/Git/researchcode/config/manifests changes.

Tiến độtraininglõi:6/9runFinishedRemote (4seed1+2seed2);2seed3vàSA14fullcòn thiếu. Ưutiên user tạoSavedVersionmới trên sad-train-a12-seeds, đổi duy nhấtPHASE=wave_a12_seed3, GPU0SA13seed3/GPU1SA12seed3; seed2V1bảo toàn. Freshpublicpretrained/30k, khôngresume. Quota20:32used15,908897633/30h/reserved0/unused14,091102367h; runtimephase3ước9,5–10,5h theohaiwallseed2,host/queue chưađo. Nếuuserlaunch20:45 thìcompute kếtthúcước06:15–07:15 ngày13/10, cộngqueue/setup/packngoàiước lượng; khôngcamkếtdeadline. A3full/pilotapproved/probe/evaluation chưa có đủresourcebảođảm; không tự giảmphạmvi hoặc đổiiteration.

- 12/10/2026 21:13:08 (+07:00): C10–C14 A1/A2 seed 3 đã gửi bởi người dùng. URL thực tế https://www.kaggle.com/code/xunanhv/sad-train-a12-seeds?scriptVersionId=357688314 xác định Saved Version 2; official SDK pin version_label=v2 xác nhận 3 code cell khớp hoàn toàn results/artifacts/kaggle/sad-train-a12-seed3.ipynb (SHA-256 bf95964efe8cbb670cfaaa2991b2b9a48ed789eab480e0a1d46914b087e57392), đúng một PHASE=wave_a12_seed3, GPU enabled/NvidiaTeslaT4 và chỉ hai input sad-train-remaining-input/sad-data. Planned GPU0 SA-13_seed3/A2 và GPU1 SA-12_seed3/A1, fresh public pretrained, target30000/run. SDK QUEUED, failure_message trống, output0/log0; chưa có preflight hoặc optimizer-step proof, chưa có artifact để thu. V1/357522661 và checkpoint seed2 đã verified được giữ nguyên; registry chưa đổi từ trạng thái xếp hàng.

Quota cùng mốc: used15,908897633/30h, reserved0, unused14,091102367h; đây là quota nền tảng, không tổng thời gian GPU vật lý. Ước lượng compute cặp seed3 9,5–10,5h theo wall seed2 ~9,45h/run, chưa đo queue/setup/pack. Nếu compute bắt đầu21:15 ngày12/10, cửa sổ compute ước06:45–07:45 ngày13/10; giờ thực tế phải dịch theo bắt đầu worker và log. Sau lượt này còn khoảng3,6–4,6h quota trước probe/evaluation/SA14: toàn bộ phạm vi tới tối14/10 vẫn chưa được bảo đảm; SA14 full còn bị chặn bởi quyết định ổn định hóa/pilot và tài nguyên. AI không tạo version, chạy training/evaluation, sửa nghiên cứu hoặc thực hiện Git; không có watcher nền khi chat/app đóng.

- 12/10/2026 21:33:20 (+07:00): rà soát tiến độ theo checklist hiện hành, không kiểm hash lại checkpoint cũ. Official SDK pinv2 sad-train-a12-seeds vẫn QUEUED, failure trống, output0/log0; quota used15,908897633/30h, unused14,091102367h, reserved0. Sáu trong chín training lõi đã hoàn tất; seed3 chưa có worker-step evidence, SA14 chưa full. Bốn CSV GoPro seed1 và bảng summary đã có local; đọc lại bảng hiện có cho uint8 PSNR A0=28,3649/A0+=28,3622/A1=28,3651/A2=28,3436. Receipt intake controls/recovery/seed2/SA14diagnostic đều hiện diện ở origin riêng; không chuyển các evidence này thành Accepted toàn dự án.

Đánh giá chất lượng hiện hành: evidence training và GoPro seed1 đủ hỗ trợ kết quả sơ bộ có thể truy vết; chưa chứng minh A2 cải thiện. A2-A1 theo video seed1 -0,0236103dB, rho estimator val40 gần0 và map gần hằng số; chưa kết luận SFT thất bại hay implementation lỗi. Chênh A2 so SA00 uint8 khoảng4,5269dB còn cần giải thích ở phạm vi giao thức đã đối chiếu; SSIMSA00 khác scope/giao thức. Seed2 validation chỉ là monitoring, không thay thế test đa seed. Synthetic/OOD/robustness/RealBlur, oracle/shuffle/probe, LPIPS, chi phí/ảnh định tính, báo cáo/slide/demo/repro/backup/phát hành/khai báoCV còn thiếu theo tiêu chí riêng; không gán phần trăm hoàn thành toàn dự án từ6/9training.

Ưu tiên song song không GPU: C41 review đủ6panel training và chuẩn bị export; C43/C53 hoàn thiện phương pháp, giao thức, bảng giả thuyết, kết quả sơ bộ và giới hạn trong notebook báo cáo hiện có; C51 chuẩn bị review các tool/registry và phương án backup ngoàiGit. Một phiên viết/review45–90phút là ước lượng công việc người dùng, không số đo log. Chuẩn bị probe_E đã có notebook local, chưa có Saved Version remote xác minh. Quyết định sửaSA14 cần user chấp thuận patch cụ thể trước pilot; nguồnGPU thêm chưa xác minh. Sauseed3 ước còn3,6–4,6h quota trước các tác vụ khác, trong khi A3full chưa có số đo; rủi ro vượt tối14/10 vẫn còn. Không tự giảm phạm vi hoặc khởi chạy thí nghiệm/Git.

- C41 QA bổ sung ngày12/10: hai screenshot training_curves_seed1_blind.html hiển thị đủ6panel. Đọc lại log nguồn (curves.csv thực tế JSON từng dòng) cho300records/run,13valrecords/run; cuốiSA10 Charbonnier0,009137704/grad_norm0,008736626/valPSNR33,791468570,SA13 Charbonnier0,009130239/l_map0,019306172/grad_norm0,008857501/valPSNR33,803355356/rho0,047614058; LR cuối1e-7,khôngzero grad tại300điểmlog mỗi run. Gradientnorm là giá trị trả về clip_grad_norm_, tức norm trướcclip, không bằng chứng mọi bước. Validation40 synthetic không so tuyệt đối vớiGoPro1111 hoặc chứng minh overfit. Chỉ QA màn hình đạt; export cuối/footer gate đãcũ/title dấu? vàsourcecắt cần hoàn thiện, không sửa hình nguồn hoặc coiC41toànphần đạt.

- C43/C53 làm song song trong lúc chờseed3: notebooks/03_results.ipynb đã bổ sung mô tả đối chứng, công thức SFT vàloss đúng source modules/losses, freshseed/publicpretrained vàtraining khóa; bổ sung diễn giải training/validation/GoPro/estimator vàgiới hạn suy luận. MụcBLIND cũ được ghi rõ snapshot trướcgate, khôngxóa lịch sử; codecells giữ nguyên. nbformat/codeAST/JSONroundtrip PASS,khôngexecute. Đây là draftnội dung, chưabáo cáo/slide cuối; userreview lập luận vàchuyển thành bài trìnhbày trong khi các bảngseed2/3/SA14/testkhác còn pending. Không có thí nghiệm/Git/model/config/data changes.

- 12/10/2026 23:37:02 (+07:00): officialSDK pinv2 sad-train-a12-seeds/357688314 chuyểnRUNNING,failure trống,APIoutput0/log0. Chỉ xácnhận SavedVersion đangchạy, chưa workerstep/preflight proof. Quota đo thực used16,977058401/30h,unused13,022941599h,reserved0. Traininglõi vẫn6/9finished,2seed3pendingvàSA14fullblocked; GoPro4/4modelseed1VerifiedLocal,4CSVGoProA1/A2seed2/3chưa có. Checklist giữ phạmvi,khôngAccepted; các dự báoquota sauseed3 trướcđây chỉước lượng vàchưađo,không dùng làm sốcòn thực tế. Không kiểmhashlại artifacts khôngđổi,khôngtraining/eval/Git.
