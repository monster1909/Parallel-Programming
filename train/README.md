# Hướng Dẫn Training & Inference - CIFAR-10 Autoencoder

## 📋 Tổng Quan

Thư mục này chứa 3 phiên bản tối ưu hóa khác nhau:

- **P2**: Direct Convolution (GPU cơ bản)
- **P3_1**: Im2Col + GEMM (tiling 16x16)
- **P3_2**: Optimized GEMM (tiling 32x32)

---

## ⚠️ LƯU Ý QUAN TRỌNG

**Bạn PHẢI vào đúng thư mục trước khi chạy lệnh make!**

```bash
# ❌ SAI - Chạy từ thư mục train
make clean && make infer_phase2    # KHÔNG HOẠT ĐỘNG!

# ✅ ĐÚNG - Phải cd vào P2 trước
cd P2
make clean && make infer_phase2
./infer_phase2
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1️⃣ Benchmark (Đo Performance)

**Chạy từng phase một:**

```bash
# Từ thư mục train, chạy lần lượt:

# Phase 2
cd P2
make clean && make benchmark_full
./benchmark_full
cd ..

# Phase 3.1
cd P3_1
make clean && make benchmark_full
./benchmark_full
cd ..

# Phase 3.2
cd P3_2
make clean && make benchmark_full
./benchmark_full
cd ..
```

**Hoặc dùng shortcut:**

```bash
cd P2 && make run
cd P3_1 && make run
cd P3_2 && make run
```

---

### 2️⃣ Training (Huấn Luyện Model)

**⚠️ Training mất nhiều thời gian! Cần có GPU CUDA.**

```bash
# Phase 2 - Training cơ bản
cd P2
make clean && make train_phase2
./train_phase2

# Hoặc dùng shortcut
cd P2 && make train
```

**Tương tự cho P3_1 và P3_2:**

```bash
# Phase 3.1
cd P3_1 && make train

# Phase 3.2
cd P3_2 && make train
```

**Output khi training:**
- ✅ Thời gian mỗi epoch
- ✅ Training loss
- ✅ Memory usage
- ✅ Lưu weights vào file `.bin`

**File weights được lưu:**
- P2: `trained_weights_p2.bin`
- P3_1: `trained_weights_p3_v1.bin`
- P3_2: `trained_weights_p3_v2.bin`

---

### 3️⃣ Inference (Chạy Model Đã Train)

**⚠️ Phải train xong mới chạy inference!**

```bash
# Phase 2
cd P2
make clean && make infer_phase2
./infer_phase2

# Hoặc dùng shortcut
cd P2 && make infer
```

**Tương tự cho các phase khác:**

```bash
cd P3_1 && make infer
cd P3_2 && make infer
```

**Output:**
- Thời gian inference
- Reconstruction loss
- Sample images (nếu có)

---

### 4️⃣ Feature Extraction

```bash
# Phase 2
cd P2 && make extract

# Phase 3.1
cd P3_1 && make extract

# Phase 3.2
cd P3_2 && make extract
```

---

## 📋 Tóm Tắt Các Lệnh Make

**Ở bên trong mỗi thư mục P2/P3_1/P3_2:**

| Lệnh | Mô Tả |
|------|-------|
| `make clean` | Xóa file build cũ |
| `make all` | Build tất cả |
| `make run` | Build + chạy benchmark |
| `make train` | Build + chạy training |
| `make infer` | Build + chạy inference |
| `make extract` | Build + extract features |

---

## 🔄 Workflow Đầy Đủ

**Để train và test một phase:**

```bash
# 1. Vào thư mục
cd P2

# 2. Clean build cũ
make clean

# 3. Chạy benchmark
make run

# 4. Training
make train

# 5. Inference
make infer

# 6. Extract features (optional)
make extract
```

---

## 📊 So Sánh Performance Giữa Các Phase

**Chạy tuần tự và lưu kết quả:**

```bash
cd P2 && make run > ../results_p2.txt
cd ../P3_1 && make run > ../results_p3_1.txt
cd ../P3_2 && make run > ../results_p3_2.txt
```

**So sánh các file:**
- `results_p2.txt`
- `results_p3_1.txt`
- `results_p3_2.txt`

---

## ⚙️ Cấu Hình

### GPU Requirements
- CUDA compute capability: **sm_75** (RTX 20xx, GTX 16xx)
- Nếu GPU khác, sửa trong `MAKEFILE`:
  ```makefile
  NVCC_FLAGS = -std=c++11 -arch=sm_XX -O3
  ```

### Dataset
- CIFAR-10 binary format
- Kiểm tra đường dẫn dataset trong code

---

## 🐛 Troubleshooting

### ❌ Lỗi: "No rule to make target 'clean'"
**Nguyên nhân:** Bạn đang ở sai thư mục!

**Giải pháp:** 
```bash
cd P2  # Phải vào thư mục P2/P3_1/P3_2 trước
make clean
```

### ❌ Lỗi: "./infer_phase2: No such file"
**Nguyên nhân:** Chưa build hoặc ở sai thư mục

**Giải pháp:**
```bash
cd P2
make infer_phase2  # Build trước
./infer_phase2     # Rồi mới chạy
```

### ❌ Lỗi: CUDA Out of Memory
**Giải pháp:**
- Giảm batch size trong code
- Dùng GPU có memory lớn hơn

### ❌ File weights không tìm thấy
**Giải pháp:**
- Phải chạy `make train` trước khi `make infer`
- Kiểm tra tên file weights trong code

---

## 📝 Ghi Chú

**Tốc độ (nhanh → chậm):**
- 🏆 **P3_2**: Nhanh nhất (32x32 tiles + optimizations)
- 🥈 **P3_1**: Trung bình (16x16 tiles)
- 🥉 **P2**: Chậm nhất (direct convolution)

**Lưu ý:**
- Training mất nhiều thời gian (có thể vài giờ)
- Cần GPU CUDA để chạy
- So sánh công bằng: dùng cùng epochs và hyperparameters
