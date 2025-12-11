# 🚀 Training SVM on GPU with RAPIDS cuML (CIFAR-10)

Dự án này thực hiện huấn luyện mô hình **Support Vector Machine (SVM)** để phân loại hình ảnh (sử dụng vector đặc trưng trích xuất từ CIFAR-10) trên **GPU** thông qua thư viện [RAPIDS cuML](https://docs.rapids.ai/).

Việc sử dụng `cuML` giúp tăng tốc độ huấn luyện lên gấp nhiều lần so với Scikit-learn chạy trên CPU truyền thống, đặc biệt với dữ liệu có số chiều cao (8192 dimensions).

## 📋 Mục lục

  - [Yêu cầu hệ thống](https://www.google.com/search?q=%23-y%C3%AAu-c%E1%BA%A7u-h%E1%BB%87-th%E1%BB%91ng)
  - [Cài đặt môi trường](https://www.google.com/search?q=%23-c%C3%A0i-%C4%91%E1%BA%B7t-m%C3%B4i-tr%C6%B0%E1%BB%9Dng)
  - [Định dạng dữ liệu](https://www.google.com/search?q=%23-%C4%91%E1%BB%8Bnh-d%E1%BA%A1ng-d%E1%BB%AF-li%E1%BB%87u)
  - [Cách sử dụng](https://www.google.com/search?q=%23-c%C3%A1ch-s%E1%BB%AD-d%E1%BB%A5ng)
  - [Kết quả đầu ra](https://www.google.com/search?q=%23-k%E1%BA%BFt-qu%E1%BA%A3-%C4%91%E1%BA%A7u-ra)

-----

## 💻 Yêu cầu hệ thống

Để chạy được mã nguồn này, bạn cần:

1.  **GPU NVIDIA**: Kiến trúc Pascal trở lên (với bộ nhớ VRAM đủ lớn, khuyến nghị \> 4GB).
2.  **Hệ điều hành**: Linux (Ubuntu 20.04/22.04) hoặc WSL2 trên Windows (RAPIDS không hỗ trợ Windows trực tiếp).
3.  **Driver**: NVIDIA Driver tương thích với CUDA 11.x hoặc 12.x.
4.  **Conda**: Khuyến khích sử dụng Anaconda hoặc Miniconda để quản lý môi trường.

-----

## 🛠 Cài đặt môi trường

Nếu chạy trên Colab, hãy đảm bảo chọn **Runtime \> Change runtime type \> T4 GPU**, sau đó chạy lệnh cài đặt này trong cell đầu tiên:

```python
!pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

-----

## 📂 Định dạng dữ liệu

Script yêu cầu 2 file nhị phân đầu vào (`.bin`) cho tập Train và Test. Dữ liệu được đóng gói (packed) theo cấu trúc sau cho mỗi mẫu (sample):

  * **Label (Nhãn):** 1 byte (unsigned char).
  * **Feature Vector:** 8192 bytes \* 4 (float32) = 32,768 bytes.

**Tổng kích thước mỗi mẫu:** 1 + 32,768 = 32,769 bytes.

Lớp `load_data` trong script sẽ tự động đọc cấu trúc này.

-----

## 🚀 Cách sử dụng

Giả sử file code của bạn tên là `svm_classifier.py`.

### 1\. Lệnh cơ bản

Chạy với các tham số mặc định:

```bash
!python svm_classifier.py \
    --train_file /content/train_features.bin \
    --test_file /content/train_features.bin \
    --output_dir ./output_gpu
```

### 2\. Tùy chỉnh tham số (Nâng cao)

Bạn có thể thay đổi Kernel, tham số C, hoặc lưu model sau khi train:

```bash
python svm_classifier.py \
  --train_file data/train_features.bin \
  --test_file data/test_features.bin \
  --output_dir ./ket_qua_svm \
  --C 50.0 \
  --kernel rbf \
  --save_model
```

### Danh sách tham số (Arguments)

| Tham số | Kiểu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--train_file` | `str` | **Required** | Đường dẫn đến file nhị phân tập huấn luyện. |
| `--test_file` | `str` | **Required** | Đường dẫn đến file nhị phân tập kiểm tra. |
| `--output_dir` | `str` | `./output_gpu` | Thư mục lưu kết quả (báo cáo, hình ảnh). |
| `--C` | `float` | `10.0` | Tham số Regularization của SVM. |
| `--kernel` | `str` | `rbf` | Loại kernel: `linear`, `poly`, `rbf`, `sigmoid`. |
| `--save_model` | `flag` | `False` | Nếu thêm cờ này, model sẽ được lưu thành file `.pkl`. |

-----

## 📊 Kết quả đầu ra

Sau khi chạy xong, script sẽ tạo ra thư mục `output_dir` chứa:

1.  **`confusion_matrix.png`**: Biểu đồ nhiệt thể hiện ma trận nhầm lẫn giữa các lớp dự đoán và thực tế.
2.  **`report.txt`**: Báo cáo chi tiết bao gồm Precision, Recall, F1-Score cho từng lớp và thời gian huấn luyện.
3.  **`svm_model_gpu.pkl`**: File model đã huấn luyện (chỉ có nếu dùng cờ `--save_model`).
4.  **Console Log**: Hiển thị thời gian đọc file, thời gian train, và độ chính xác (Accuracy) ngay trên màn hình terminal.

-----