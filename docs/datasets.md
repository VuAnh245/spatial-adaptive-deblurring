# Dataset, checkpoint và giấy phép

Đồ án DL dùng GoPro (nguồn ảnh nét để sinh blur, test ngoài) và RealBlur-J (chỉ test ngoài). Dữ liệu gốc không commit; mọi split nằm trong `manifests/` kèm sha256.
Tải tự động phần có thể bằng `bash scripts/download_data.sh`; RealBlur-J tải tay từ cổng chính thức.

## GoPro (CC BY 4.0)

| Mục | Link |
| --- | --- |
| Trang chính thức | https://seungjunnah.github.io/Datasets/gopro |
| GOPRO_Large (cặp mờ/nét, 2.103 train / 1.111 test, 1280×720) | https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip |
| GOPRO_Large_all (toàn bộ frame nét, tùy chọn) | https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large_all.zip |
| Bản Google Drive (dự phòng) | https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view |
| Paper | Nah et al., CVPR 2017, https://arxiv.org/abs/1612.02177 |

### Bản đã tải và kiểm chứng (2026-10-05)

| Mục | Giá trị |
| --- | --- |
| Nguồn | `huggingface.co/datasets/snah/GOPRO_Large`, commit `592978466ae5` |
| `GOPRO_Large.zip` | 9.540.471.600 byte, sha256 `24532c036712515cf33803704421311c91f6a080e76049d7e5e7fd22389f128e` |
| Phần giải nén | chỉ `train/*/sharp`, `test/*/sharp`, `test/*/blur` (bỏ `train/*/blur`, `*/blur_gamma`: không dùng), 4,3 GB |
| Đếm được | train 22 chuỗi / 2.103 ảnh nét; test 11 chuỗi / 1.111 cặp; 1280×720 RGB |

Lưu ý: split chính thức có chuỗi train và test cắt từ cùng một video gốc (ví dụ `GOPR0384_11_00` ở test, `GOPR0384_11_01..04` ở train). Đây là split chuẩn của cộng đồng, giữ nguyên; ghi vào mục Giới hạn. Thống kê theo nhóm dùng đơn vị chuỗi (11 nhóm ở test).

## RealBlur-J (CC BY 4.0)

| Mục | Link |
| --- | --- |
| Trang dự án | http://cg.postech.ac.kr/research/RealBlur/ |
| Repo + script đánh giá chính thức `evaluation_RealBlur_ecc.py` | https://github.com/rimchang/RealBlur |
| Cổng tải (POSTECH) | https://cgdata.postech.ac.kr/sharing/YhKdbtvD0 |
| Google Drive (dự phòng) | https://drive.google.com/drive/folders/1xUNAAVzLhNQuGriKTk1hrE-MT-H_56fq |
| Paper | Rim et al., ECCV 2020, https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700188.pdf |

Ghi chú: 3.758 cặp train / 980 cặp test cho mỗi bản J và R; dùng bản **J**. Metric phải tính sau bước căn chỉnh của script chính thức (đặt vào `third_party/realblur/`), không tự viết lại.


## Checkpoint công khai

| Model | Link | Ghi chú |
| --- | --- | --- |
| NAFNet-GoPro-width32 | https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view | Repo chính thức báo 32,87 dB PSNR / 0,9606 SSIM trên GoPro test. Repo: https://github.com/megvii-research/NAFNet (MIT). Bản đã tải: 68,7 MB, md5 `1d0ce40c0dfc4bcd3f3d65ecc35fca51` |
| NAFNet-GoPro-width64 (tham chiếu) | https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view | 33,71 dB theo repo chính thức |
| Restormer (motion deblurring) | https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth | Cần code kiến trúc từ https://github.com/swz30/Restormer; chỉ đánh giá, không train |
| MPRNet (deblurring) | https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb/view | Cần code từ https://github.com/swz30/MPRNet; chỉ đánh giá |
| YOLOv8s | `yolov8s.pt`, tự tải khi gọi `YOLO("yolov8s.pt")` | https://docs.ultralytics.com/models/yolov8/ (AGPL-3.0). COCO val mAP50-95 = 44,9 theo trang tài liệu |

## Bố cục thư mục sau khi tải

```text
data/
├── GOPRO_Large/{train,test}/<sequence>/{blur,sharp}/*.png
├── RealBlur/  (RealBlur_J_train_list.txt, RealBlur_J_test_list.txt, ảnh theo scene)
└── synthetic/ (DL)  hoặc  coco_blur/ (CV) - sinh bởi script, không tải
checkpoints/   third_party/realblur/evaluation_RealBlur_ecc.py
```
