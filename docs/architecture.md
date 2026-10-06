# Kiến trúc SFT-Map (A2)

```text
B ──► Blur-map estimator ──► Ŝ (1/4 res) ──► SFT head mỗi stage ──► Δγ_s, β_s
│                                                                     │
└──► NAFNet-32 encoder ─ middle ─ decoder (mỗi NAFBlock: LN → modulation → ...) ──► Î = B + R
```

- Modulation sau `norm1` của mỗi NAFBlock: `F~ = (1 + Δγ_s)(1 + Δγ_c) ⊙ LN(F) + β_s + β_c`.
- Lớp cuối của SFT head và nhánh channel khởi tạo 0: ở bước 0 mọi biến thể trùng checkpoint
  (kiểm bằng `tests/test_modulation.py`).
- **Điều biến hiệu dụng ở 8 stage, không phải 9.** Head `mid0` được nối đúng nhưng không học: trong checkpoint công khai,
  hệ số residual `beta` của block middle có trung bình |beta| = 0,0036 (max 0,027), nhỏ hơn 35 đến 60 lần các block khác
  (0,12 đến 0,21; `encoders.3.27` 0,158, `decoders.3.0` 0,208). Nhánh mang điều biến của block này gần như không ảnh hưởng
  đầu ra, nên gradient về `sft["mid0"]` gần 0 và ‖W‖ của nó đứng yên ở 0 trong cả ba pilot. Đây là đặc điểm của checkpoint,
  không phải lỗi; SA-30 (trực quan hệ số điều biến) báo `mid0` riêng. Vì vậy SA-15 đổi từ "chỉ ở middle" sang "chỉ ở decoder".
- NAFNet-32 GoPro: encoder 4 stage (32 → 64 → 128 → 256 kênh, số block 1, 1, 1, 28), middle 512 kênh (1 block),
  decoder 4 stage (256 → 32 kênh, 1 block mỗi stage).
- A0+: ít NAFBlock khởi tạo đồng nhất nhất trong một stage decoder sao cho phần tham số thêm vào lệch ≤ 5% so với
  phần thêm của A2 (QĐ-06): 2 block ở dec1, 17.349.731 tham số
  (`build_param_matched`), không đặt ở middle vì một block 512 kênh lớn hơn nhiều phần chênh lệch.
