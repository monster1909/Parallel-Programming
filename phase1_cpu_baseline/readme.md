# Convolutional Autoencoder (CPU Baseline)

Dự án này là một implementation thủ công (from scratch) của mạng nơ-ron tích chập **Autoencoder** bằng ngôn ngữ C++.
Code không sử dụng bất kỳ thư viện Deep Learning nào (như PyTorch, TensorFlow) mà tự cài đặt các thuật toán cốt lõi: Convolution (thông qua `im2col` + `GEMM`), Backpropagation, và tối ưu hóa SGD.

Dự án được tối ưu hóa cho CPU sử dụng **OpenMP** để tính toán song song.

## 📂 Cấu trúc thư mục

```text
phase1_cpu_baseline/
├──Makefile
├── include/            # Các file header (.h)
│   ├── config.h        # Cấu hình hyper-parameters (LR, Epochs,...)
│   ├── common.h        # Các hàm tiện ích (im2col, gemm, memory)
│   ├── autoencoder.h   # Class chính quản lý mô hình
│   ├── layers/         # Định nghĩa các lớp (Conv2D, ReLU, MaxPool...)
│   └── utils/          # Các tiện ích phụ (Timer, Weight Init)
├── src/                # Mã nguồn triển khai (.cpp)
│   ├── main_cpu.cpp    # Hàm main, xử lý tham số dòng lệnh
│   ├── autoencoder.cpp # Logic forward/backward của cả mạng
│   ├── layers/         # Triển khai chi tiết các lớp mạng
│   └── utils/          # Triển khai timer, random
└── README.md           # File hướng dẫn này
```

## 🛠️ Yêu cầu hệ thống

  * **Compiler**: `g++` (Hỗ trợ C++11 trở lên).
  * **Thư viện**: OpenMP (thường có sẵn trong GCC).
  * **Hệ điều hành**: Linux / MacOS / Windows (với MinGW hoặc WSL).
  * **Dữ liệu**: Bộ dữ liệu CIFAR-10 (phiên bản Binary).

## 🚀 Cách biên dịch

Di chuyển vào thư mục `phase1_cpu_baseline` và chạy lệnh sau:

```bash
make
```

-----

## 💾 Chuẩn bị dữ liệu

Dự án sử dụng bộ dữ liệu **CIFAR-10 Binary Version**.

1.  Tải về tại: [CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html) (chọn **CIFAR-10 binary version**).
2.  Giải nén, bạn sẽ thấy các file như `data_batch_1.bin`, `data_batch_2.bin`,...

-----

## 🏃 Hướng dẫn chạy

Chương trình có 2 chế độ: **Train** (Huấn luyện) và **Test** (Kiểm thử/Tái tạo ảnh).

### Xem Thông Tin:
```bash
./autoencoder info
```
### 1\. Chế độ Train (Huấn luyện)

Dùng để học các đặc trưng từ dữ liệu và lưu trọng số vào file.

**Cú pháp:**

```bash
./autoencoder train <danh_sách_file_data> <file_trọng_số_đầu_ra>
```

**Ví dụ:**

```bash
# Huấn luyện trên 1 file batch và lưu vào weights.bin
./autoencoder train data_batch_1.bin weights.bin

# Huấn luyện trên nhiều file (nếu có)
./autoencoder train data_batch_1.bin data_batch_2.bin weights.bin
```

*Quá trình huấn luyện sẽ hiển thị Loss theo từng Step và Epoch.*

### 2\. Chế độ Test (Tái tạo ảnh)

Dùng trọng số đã huấn luyện để nén và giải nén một ảnh đầu vào (định dạng PPM).

**Cú pháp:**

```bash
./autoencoder test <file_trọng_số> <ảnh_đầu_vào.ppm> <ảnh_đầu_ra.ppm>
```

**Ví dụ:**

```bash
./autoencoder test weights.bin input.ppm output.ppm
```

> **Lưu ý:** Ảnh đầu vào phải là định dạng **PPM (P3)** kích thước **32x32**. Bạn có thể dùng GIMP hoặc các công cụ convert online để tạo file PPM.

### 2\. Chế độ Test (Tái tạo ảnh)

```bash
./autoencoder extract <file_trọng_số> <tên_file_đầu_ra.bin> <danh sách file data>
```
**Ví dụ:**
```bash
!./autoencoder extract /content/drive/MyDrive/ckpt/weights_final_1.bin train_features.bin data/data_batch_1.bin data/data_batch_2.bin data/data_batch_3.bin data/data_batch_4.bin data/data_batch_5.bin /content/Parallel-Programming/phase1_cpu_baseline/cifar-10-batches-bin/test_batch.bin
```

-----

## 🧠 Kiến trúc & Kỹ thuật

### 1\. Kiến trúc Mạng (Architecture)

Mô hình nhận đầu vào là ảnh màu 32x32 (CIFAR-10) và cố gắng tái tạo lại nó.

| Lớp (Layer) | Input Shape | Output Shape | Tham số (Kernel/Stride) | Chức năng |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | 3x32x32 | - | - | Ảnh gốc |
| **Conv1** | 3x32x32 | 256x32x32 | K=3, P=1, S=1 | Trích xuất đặc trưng sơ cấp |
| **ReLU + Pool1** | 256x32x32 | 256x16x16 | 2x2 | Giảm chiều dữ liệu |
| **Conv2** | 256x16x16 | 128x16x16 | K=3, P=1, S=1 | Trích xuất đặc trưng sâu hơn |
| **ReLU + Pool2** | 128x16x16 | **128x8x8** | 2x2 | **Latent Space (Vùng ẩn)** |
| **Conv3** | 128x8x8 | 128x8x8 | K=3, P=1, S=1 | Bắt đầu giải mã |
| **Upsample1** | 128x8x8 | 128x16x16 | 2x | Tăng kích thước ảnh |
| **Conv4** | 128x16x16 | 256x16x16 | K=3, P=1, S=1 | Khôi phục chi tiết |
| **Upsample2** | 256x16x16 | 256x32x32 | 2x | Tăng về kích thước gốc |
| **Conv5** | 256x32x32 | **3x32x32** | K=3, P=1, S=1 | Tái tạo ảnh màu RGB |

### 2\. Kỹ thuật triển khai (Core Implementation)

  * **im2col + GEMM**:
      * Phép tính Convolution (tích chập) được chuyển đổi thành phép nhân ma trận (Matrix Multiplication).
      * `im2col`: Biến đổi các vùng ảnh cục bộ (patches) thành các cột của ma trận.
      * `GEMM`: Nhân ma trận trọng số với ma trận `im2col` để ra kết quả.
  * **Memory Management**:
      * Sử dụng `malloc`/`free` trực tiếp để quản lý bộ nhớ, mô phỏng cách hoạt động ở cấp thấp (C-style).
      * Không sử dụng `std::vector` cho các buffer lớn để tăng tốc độ truy cập mảng thô.
  * **Parallelization**:
      * Sử dụng `#pragma omp parallel for` để song song hóa quá trình xử lý theo từng mẫu (sample) trong một Batch.
  * **Backpropagation**:
      * Tính toán gradient thủ công cho từng lớp (Chain Rule).
      * Lưu trữ chỉ số `argmax` tại các lớp MaxPool để phục vụ quá trình Backward (truyền gradient về đúng vị trí pixel lớn nhất).

## ⚙️ Cấu hình (Config)

Bạn có thể thay đổi các tham số trong file `include/config.h`:

  * `BATCH_SIZE`: Số lượng ảnh xử lý cùng lúc (mặc định 32).
  * `EPOCHS`: Số vòng lặp huấn luyện (mặc định 20).
  * `LR` (Learning Rate): Tốc độ học (mặc định 0.001).
  * `F1`, `F2`: Số lượng Filters (kênh) của các lớp Convolution.