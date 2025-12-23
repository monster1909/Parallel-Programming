# 🚀 Hướng Dẫn Chạy Notebook Trên Google Colab

## 📋 Tổng Quan

Notebook `Final_Parallel_prgraming.ipynb` đã được thiết kế sẵn để chạy trên **Google Colab** với GPU Tesla T4 miễn phí.

---

## 🔧 Bước 1: Mở Notebook Trên Colab

### Cách 1: Upload File Trực Tiếp
1. Truy cập: https://colab.research.google.com/
2. Click **File** → **Upload notebook**
3. Chọn file `Final_Parallel_prgraming.ipynb`

### Cách 2: Từ GitHub (Nếu Đã Push)
1. Truy cập: https://colab.research.google.com/
2. Click **File** → **Open notebook**
3. Chọn tab **GitHub**
4. Nhập URL repo: `https://github.com/monster1909/Parallel-Programming`
5. Chọn file `Final_Parallel_prgraming.ipynb`

---

## ⚡ Bước 2: Bật GPU

**⚠️ QUAN TRỌNG: Phải bật GPU trước khi chạy!**

1. Click **Runtime** (hoặc **Môi trường chạy**)
2. Chọn **Change runtime type** (hoặc **Thay đổi loại môi trường chạy**)
3. Trong **Hardware accelerator** (hoặc **Bộ tăng tốc phần cứng**):
   - Chọn **T4 GPU** (hoặc **GPU**)
4. Click **Save** (hoặc **Lưu**)

**Kiểm tra GPU:**
```python
!nvidia-smi
```

Bạn sẽ thấy thông tin GPU Tesla T4.

---

## 📝 Bước 3: Chạy Notebook Tuần Tự

### Cell 1: Kiểm Tra GPU
```python
!nvidia-smi
```
✅ Xác nhận GPU đang hoạt động

### Cell 2: Clone Repository
```python
!git clone -b master https://github.com/monster1909/Parallel-Programming.git
%cd Parallel-Programming
```
✅ Download toàn bộ code và dataset

---

### 🔬 Phần 1: Benchmark/Optimizer

#### Phase 2 (Cells 3-6)
```python
%cd /content/Parallel-Programming/phase2_gpu_basic
!make -f MAKEFILE
!./test_gpu    # Test 1 ảnh
!./run_gpu     # Benchmark 60,000 ảnh
```

#### Phase 3.1 (Cells 7-9)
```python
%cd /content/Parallel-Programming/phase3_gpu_optimized
!make -f MAKEFILE
!./test_gpu
!./run_phase3
```

#### Phase 3.2 (Cells 10-13)
```python
%cd /content/Parallel-Programming/phase3_gpu_optimized_v2
!make -f MAKEFILE
!./test_gpu
!./run_phase3
!./feature_extract
```

---

### 🎓 Phần 2: Training & Inference

#### Training Phase 2 (Cells 14-17)
```python
%cd /content/Parallel-Programming/train/P2
!mkdir -p logs weights
!make -f MAKEFILE run      # Benchmark
!make -f MAKEFILE train    # Training (MẤT NHIỀU THỜI GIAN!)
```

**⚠️ Lưu ý Training:**
- **Mất rất nhiều thời gian** (có thể vài giờ)
- Colab free có giới hạn GPU runtime (~12 giờ/ngày)
- Nếu muốn test nhanh, có thể **dừng sớm** (Runtime → Interrupt execution)

#### Inference Phase 2 (Cell 18)
```python
!make -f MAKEFILE infer    # Sau khi train xong
```

---

#### Training Phase 3.1 (Cells 19-22)
```python
%cd /content/Parallel-Programming/train/P3_1
!mkdir -p logs weights
!make -f MAKEFILE run
!make -f MAKEFILE train    # Training
!make -f MAKEFILE infer    # Inference
```

---

#### Training Phase 3.2 (Cells 23-26)
```python
%cd /content/Parallel-Programming/train/P3_2
!mkdir -p logs weights
!make -f MAKEFILE run
!make -f MAKEFILE train    # Training
!make -f MAKEFILE infer    # Inference
```

---

## 🎯 Cách Chạy Hiệu Quả

### Tùy Chọn 1: Chạy Toàn Bộ (Tự Động)
1. Click **Runtime** → **Run all** (hoặc Ctrl+F9)
2. Chờ tất cả cells chạy xong
3. ⚠️ Training sẽ mất rất nhiều thời gian!

### Tùy Chọn 2: Chạy Từng Phần (Khuyến Nghị)
1. **Chỉ chạy Benchmark** (Cells 1-13):
   - Nhanh, chỉ mất ~5-10 phút
   - Đủ để so sánh performance
   
2. **Chạy Training** (Cells 14+):
   - Chỉ khi cần train thật
   - Mất nhiều giờ!

### Tùy Chọn 3: Test Nhanh
1. Chạy cells: 1, 2, 3, 4, 5 (Phase 2 test)
2. Chạy cells: 7, 8, 9 (Phase 3.1 test)
3. Chạy cells: 10, 11, 12 (Phase 3.2 test)
4. **Bỏ qua training** nếu chỉ muốn xem benchmark

---

## 💡 Tips & Tricks

### 1. Tránh Timeout
- Colab free có giới hạn GPU runtime
- Training lâu → Có thể bị disconnect
- **Giải pháp:** 
  - Giảm số epochs trong code
  - Dùng Colab Pro nếu cần train lâu

### 2. Lưu Kết Quả
```python
# Tải weights về máy sau khi train xong
from google.colab import files
files.download('/content/Parallel-Programming/train/P2/trained_weights_p2.bin')
```

### 3. Xem Logs
```python
# Xem file logs nếu có
!cat /content/Parallel-Programming/train/P2/logs/*.txt
```

### 4. Dừng Training Khi Cần
- Click nút **Stop** ⬛ bên cạnh cell
- Hoặc **Runtime** → **Interrupt execution**

---

## 🐛 Troubleshooting

### ❌ Lỗi: "No GPU detected"
**Giải pháp:** 
1. Runtime → Change runtime type
2. Chọn T4 GPU
3. Save và restart runtime

### ❌ Lỗi: "Disconnected from runtime"
**Giải pháp:**
- Colab timeout do không có hoạt động
- Click **Reconnect** và chạy lại từ đầu

### ❌ Lỗi: "make: command not found"
**Giải pháp:**
- Colab đã có make sẵn
- Kiểm tra lại xem đã cd đúng thư mục chưa

### ❌ Lỗi: "nvcc: command not found"
**Giải pháp:**
- Kiểm tra GPU đã bật chưa
- Restart runtime

---

## 📊 Kết Quả Mong Đợi

### Benchmark Performance:
- **Phase 2**: ~12 giây cho 60,000 ảnh ✅
- **Phase 3.1**: ~1.3 giây cho 60,000 ảnh ✅ (nhanh hơn ~9x)
- **Phase 3.2**: ~0.68 giây cho 60,000 ảnh ✅ (nhanh hơn ~18x)

### Training:
- Mỗi epoch: Vài phút đến vài chục phút
- Total: Vài giờ cho 20 epochs

---

## 📌 Tóm Tắt

| Bước | Thời Gian | Mô Tả |
|------|-----------|-------|
| 1. Mở Colab & bật GPU | 1 phút | Bắt buộc |
| 2. Clone repo (Cell 1-2) | 1 phút | Tự động |
| 3. Benchmark (Cells 3-13) | 5-10 phút | Khuyến nghị chạy |
| 4. Training (Cells 14+) | VÀI GIỜ | Tùy chọn |

**Khuyến nghị:** 
- Lần đầu tiên: Chạy **Cells 1-13** để xem benchmark
- Nếu cần training: Chạy từng phase một, theo dõi kỹ

---

## 🎓 Lưu Ý Cuối

- Notebook đã config sẵn mọi thứ, chỉ cần chạy tuần tự
- Dataset CIFAR-10 đã có trong repo
- GPU T4 trên Colab đủ mạnh để chạy tất cả
- Training mất thời gian nhưng chạy được hoàn toàn trên Colab free!

Chúc bạn thành công! 🚀
