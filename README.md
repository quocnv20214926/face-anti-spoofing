# Real-Time Face Anti-Spoofing & Liveness Detection

Hệ thống phát hiện giả mạo khuôn mặt (Face Anti-Spoofing - FAS) theo thời gian thực sử dụng mạng nơ-ron tích chập nhẹ **MobileNetV3-Small** kết hợp bộ phát hiện khuôn mặt tốc độ cao **OpenCV YuNet**. Hệ thống được thiết kế nhằm tối ưu hóa độ trễ, tài nguyên tính toán và độ chính xác phân loại giữa khuôn mặt thật (*bona fide*) và các hình thức tấn công giả mạo (*presentation attack* như ảnh in, màn hình số).

---

## 🌟 Điểm nổi bật (Key Features)

- **Kiến trúc tinh gọn (Lightweight Architecture)**: Sử dụng MobileNetV3-Small được tinh chỉnh (Transfer Learning) cho bài toán phân loại nhị phân (*Real vs Spoof*), phù hợp triển khai trên cả CPU và GPU.
- **Phát hiện khuôn mặt hiệu năng cao (YuNet Detector)**: Tích hợp mô hình ONNX YuNet qua OpenCV DNN với tốc độ xử lý nhanh, nhận diện chính xác các góc nghiêng và khoảng cách khuôn mặt.
- **Đánh giá theo chuẩn quốc tế (ISO/IEC 30107-3)**: Tích hợp đầy đủ các chỉ số chuyên ngành:
  - **APCER** (*Attack Presentation Classification Error Rate*): Tỷ lệ lọt mẫu giả mạo.
  - **BPCER** (*Bona Fide Presentation Classification Error Rate*): Tỷ lệ nhận nhầm người thật thành giả mạo.
  - **ACER** (*Average Classification Error Rate*): Trung bình sai số, tự động tối ưu hóa ngưỡng quyết định (*optimal threshold*) trên tập kiểm định.
- **Pipeline huấn luyện cân bằng (Robust Training Pipeline)**:
  - Hỗ trợ Mixed Precision (`torch.amp`) tăng tốc tính toán.
  - Chiến lược Freeze/Unfreeze backbone thích ứng chống over-fitting.
  - `WeightedRandomSampler` giải quyết triệt để mất cân bằng dữ liệu giữa các lớp.
  - Data Augmentation đa dạng: RandomResizedCrop, ColorJitter, GaussianBlur, RandomGrayscale mô phỏng môi trường ánh sáng phức tạp.
- **Làm mượt thời gian thực (Temporal Smoothing)**: Sử dụng hàng đợi trượt (*sliding window moving average*) trên luồng video webcam giúp ổn định kết quả dự đoán, triệt tiêu hiện tượng nhấp nháy/giật nhãn.

---

## 📁 Cấu trúc thư mục dự án

```text
CvProject/
├── Data/
│   ├── training/           # Dữ liệu huấn luyện (gồm real/ và spoof/)
│   ├── development/        # Dữ liệu kiểm định & cân chỉnh ngưỡng tối ưu
│   └── evaluation/         # Dữ liệu kiểm thử độc lập (Held-out Test)
├── models/
│   └── face_detection_yunet_2023mar.onnx   # Trọng số YuNet Face Detector
├── scripts/
│   ├── fas_core.py         # Kiến trúc mạng, tiền xử lý, tính toán chỉ số & checkpoint
│   ├── train.py            # Pipeline huấn luyện, early stopping & đánh giá ACER
│   └── webcam.py           # Ứng dụng nhận diện trực tiếp qua webcam
├── artifacts/              # Thư mục lưu checkpoint mô hình (.pt) sau khi train
├── requirements.txt        # Danh sách các thư viện phụ thuộc
└── README.md
```

---

## 🚀 Cài đặt & Chuẩn bị môi trường

### 1. Yêu cầu hệ thống
- Python `>= 3.10`
- Windows / Linux / macOS
- (Tùy chọn) GPU NVIDIA hỗ trợ CUDA để tăng tốc độ huấn luyện.

### 2. Thiết lập môi trường ảo

```powershell
# Khởi tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

## 🏋️ Huấn luyện mô hình (Training & Validation)

Chạy pipeline huấn luyện tự động với toàn bộ các bước tiền xử lý, augment, cân bằng lớp và tìm ngưỡng tối ưu:

```powershell
python scripts/train.py --epochs 12 --batch-size 64 --lr 0.0003
```

### Các tham số tùy chỉnh:

| Tham số | Giá trị mặc định | Mô tả |
| :--- | :---: | :--- |
| `--data` | `Data` | Đường dẫn đến thư mục chứa các tập dữ liệu |
| `--output` | `artifacts/fas_mobilenetv3.pt` | Đường dẫn lưu checkpoint tốt nhất |
| `--epochs` | `12` | Tổng số epoch huấn luyện tối đa |
| `--freeze-epochs` | `2` | Số epoch đóng băng feature extractor ban đầu |
| `--batch-size` | `64` | Kích thước batch (giảm xuống `32` hoặc `16` nếu thiếu RAM/VRAM) |
| `--lr` | `3e-4` | Tốc độ học (Learning Rate) cho bộ tối ưu AdamW |
| `--patience` | `4` | Số epoch chờ cải thiện ACER trước khi kích hoạt Early Stopping |

Sau khi hoàn tất, checkpoint đạt chỉ số ACER thấp nhất trên tập `development` sẽ được lưu lại cùng ngưỡng quyết định tối ưu (`real_threshold`), sau đó tự động đánh giá trên tập `evaluation`.

---

## 📹 Chạy thử nghiệm thời gian thực (Webcam Inference)

Chạy kiểm thử trực tiếp trên camera máy tính:

```powershell
python scripts/webcam.py
```

### Điều khiển & Tùy chọn:
- Nhấn phím **`q`** để dừng ứng dụng và đóng cửa sổ webcam.
- Tùy chỉnh camera hoặc ngưỡng phân loại:
  ```powershell
  # Chọn camera phụ (ID 1) và áp dụng ngưỡng tin cậy 0.6
  python scripts/webcam.py --camera 1 --threshold 0.6 --smoothing 10
  ```

| Tham số | Mặc định | Mô tả |
| :--- | :---: | :--- |
| `--checkpoint` | `artifacts/fas_mobilenetv3.pt` | File mô hình đã huấn luyện |
| `--camera` | `0` | Chỉ số camera đầu vào |
| `--threshold` | *Từ checkpoint* | Ngưỡng xác suất để phân loại khuôn mặt thật |
| `--margin` | `0.25` | Tỷ lệ mở rộng vùng bbox khuôn mặt để bao quát ngữ cảnh xung quanh |
| `--smoothing` | `8` | Số khung hình liên tiếp dùng để làm mượt dự đoán |

---

## 📊 Phương pháp đánh giá (Evaluation Methodology)

Dự án áp dụng tiêu chuẩn đánh giá chống giả mạo khuôn mặt theo ISO/IEC 30107-3:

$$\text{ACER} = \frac{\text{APCER} + \text{BPCER}}{2}$$

- **APCER** (False Acceptance / Tỷ lệ nhận lầm giả mạo): Càng thấp chứng tỏ hệ thống phòng thủ tấn công càng mạnh.
- **BPCER** (False Rejection / Tỷ lệ từ chối người thật): Càng thấp giúp trải nghiệm người dùng không bị gián đoạn.
- **ACER**: Đánh giá toàn diện sự cân bằng giữa khả năng bảo mật và độ tiện dụng.
