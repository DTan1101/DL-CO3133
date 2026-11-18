# VCC --- Visibility-Gated and Visibility-Calibrated Consistency Learning

Mã nguồn cho bản thảo `manuscripts/main.tex`: *Học nhất quán hiệu chỉnh theo mức nhìn thấy
cho phân loại giống chó, mèo dưới che khuất một phần* (Oxford-IIIT Pet, 37 giống).

Mọi phương pháp đều là một trường hợp riêng của **khung trọng số--đích thống nhất**

$$\mathcal{L} = \frac{1}{B}\sum_i \mathrm{CE}(z_i, y_i) \;+\; \lambda \frac{1}{B}\sum_i w_i\, D(t_i \,\|\, q_i),$$

nên mỗi thí nghiệm chỉ là một file YAML quy định cặp $(w_i, t_i)$.

| Mã | File cấu hình | Trọng số $w_i$ | Đích $t_i$ |
|---|---|---|---|
| E0 | `configs/e0_baseline.yaml` | --- | chỉ CE trên ảnh sạch |
| E1 | `configs/e1_random_erasing.yaml` | --- | CE trên ảnh Random Erasing |
| E2 | `configs/e2_foreground_occlusion.yaml` | --- | CE trên ảnh che theo tiền cảnh |
| E3 | `configs/e3_consistency.yaml` | $1$ | $p$ |
| E4 | `configs/e4_visibility_gating.yaml` | $g(v)$ | $p$ |
| E5 | `configs/e5_vis_conf_gating.yaml` | $g(v)h(c)$ | $p$ |
| **E6 (VCC)** | **`configs/e6_vcc.yaml`** | $h(c)$ | $\alpha(v)p + (1-\alpha(v))u$ |
| E6-SA | `configs/e6sa_soft_augmentation.yaml` | $1$ | $\alpha(v)e_y + (1-\alpha(v))u$ |
| E6-T | `configs/e6t_vcc_temperature.yaml` | $h(c)$ | $\mathrm{softmax}(z/T(v))$ |

với $\alpha(v) = 1-(1-v)^k$ và $v$ là mức nhìn thấy của tiền cảnh, tính **chính xác** từ trimap:
$v = \sum m(1-o) / \sum m$.

## 1. Cài đặt

```bash
conda env create -f environment.yaml && conda activate vcc-pets
```

## 2. Dữ liệu

Tải [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (`images.tar.gz`,
`annotations.tar.gz`) và giải nén vào `datasets/Oxford-IIIT_Pet_Dataset/` sao cho có
`images/` và `annotations/trimaps/`. Sau đó tạo phân chia train/val/test:

```bash
python -m datasets.preprocessing_data --root datasets/Oxford-IIIT_Pet_Dataset --summarize
```

Phân chia chính thức được giữ nguyên: `trainval.txt` chia tiếp 80/20 theo lớp thành
train/val, `test.txt` dùng làm test. Bước này tự chạy khi huấn luyện nếu chưa có file.

## 3. Huấn luyện

```bash
python tools/train.py --config configs/e6_vcc.yaml                    # VCC, 3 hạt giống
python tools/train.py --config configs/e0_baseline.yaml --seeds 0     # một hạt giống
python tools/train.py --config configs/e6_vcc.yaml --debug            # kiểm tra pipeline
```

Ghi đè bất kỳ khoá nào mà không cần sửa file YAML:

```bash
python tools/train.py --config configs/e6_vcc.yaml \
  --set loss.consistency.lambda=2 loss.consistency.target.k=3
```

Kết quả nằm trong `checkpoints/<name>/seed<k>/`: `best.pth`, `last.pth`, `history.json`,
`sweep_test.json`, `cost.json` (thời gian mỗi epoch, bộ nhớ GPU đỉnh), và
`checkpoints/<name>/summary.{json,md}` khi chạy nhiều hạt giống.

### Đổi backbone

```bash
python tools/train.py --config configs/backbones/vcc_resnet50.yaml
python tools/train.py --config configs/e6_vcc.yaml --backbone convnext_tiny
```

Có sẵn trong `configs/backbones/`: ResNet-50, ConvNeXt-Tiny, EfficientNet-B0,
MobileNetV3, ViT-S/16. Mọi tên backbone của `timm` đều dùng được qua `--backbone`.

### Thí nghiệm loại trừ

`configs/ablations/` chứa các ablation của bản thảo: $k \in \{1,3\}$,
$\lambda \in \{0{,}5; 2; 4\}$, $D \in \{\mathrm{JS}, \mathrm{MSE}\}$,
$h \in \{1, p_y, \max_k p_k$, ngưỡng cứng, SoftMatch$\}$, kết hợp cổng bằng tích/min,
chuẩn hoá $\sum wD/\sum w$, tiên nghiệm theo tần suất lớp, nhiệt độ cho mô hình thầy,
mức nhìn thấy ngữ nghĩa $v_\mathrm{sem}$ (CAM), gộp vùng biên trimap, Split-BN.

## 4. Đánh giá

Giao thức test giống nhau cho mọi thí nghiệm: vật che trên tiền cảnh với hạt giống cố
định tại $\rho \in \{0; 0{,}1; 0{,}3; 0{,}5; 0{,}7\}$ (khai báo ở `eval.occlusion`, độc
lập với cách che khi huấn luyện).

```bash
python tools/eval.py --checkpoint checkpoints/e6_vcc/seed0/best.pth --latex
python tools/eval.py --checkpoint checkpoints/e6_vcc/seed0/best.pth \
  --baseline checkpoints/e0_baseline/seed0/sweep_test.json      # thêm RD(ρ)
python tools/eval.py --aggregate "checkpoints/*/seed*/sweep_test.json" --latex
```

Độ đo: Top-1, Top-5, Macro-F1, ECE (15 khoảng), NLL, Brier, $\mathrm{AUC_{occ}}$ và
$\mathrm{RD}(\rho)$. `--latex` in dòng dán trực tiếp vào bảng kết quả của bản thảo.

## 5. Bố cục

```
configs/      mỗi thí nghiệm một file YAML (kế thừa qua khoá `_base_`)
datasets/     preprocessing_data.py (phân chia) · dataset.py (che khuất, v) · dataloader.py
models/       backbone.py (timm/torchvision, Split-BN) · model.py (f_θ, CAM)
utils/        loss.py (khung w--t) · metrics.py (ECE, AUC_occ, RD) · utils.py (cấu hình, log)
trainers/     vcc_trainer.py (vòng huấn luyện batch 2B, quét ρ, tổng hợp nhiều hạt giống)
tools/        train.py · eval.py
options/      train_option.py · eval_option.py (tham số dòng lệnh)
```
