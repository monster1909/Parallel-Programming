# CIFAR-10 Autoencoder - GPU Optimization

## Tổng Quan

Project implement autoencoder cho CIFAR-10 với 3 phases GPU optimization.

## Yêu Cầu

- CUDA 12.6+
- NVIDIA GPU (GTX 1650 trở lên)
- C++17
- CIFAR-10 dataset trong `data/cifar-10-batches-bin/`

**Chạy trên Google Colab**: Xem file `COLAB_SETUP.md` để biết cách setup nhanh.

---

## PHASE 2: GPU Basic (Direct Convolution)

### Bước 1: Compile
```bash
cd phase2_gpu_basic
make -f MAKEFILE clean
make -f MAKEFILE
```

### Bước 2: Test 1 Ảnh (Detailed Timing)
```bash
./test_gpu
```
**Chức năng**: 
- Load 1 ảnh từ CIFAR-10
- Chạy qua autoencoder
- In ra timing chi tiết từng layer (Conv1, ReLU1, MaxPool1, etc.)
- Hiển thị output sample

### Bước 3: Benchmark 60,000 Ảnh
```bash
./run_gpu
```
**Chức năng**:
- Load 1 ảnh từ CIFAR-10
- Chạy 60,000 lần (simulate 60k images)
- Batch size = 1 (process từng ảnh)
- In ra: Total time, average time per image
- So sánh với target <20s

---

## PHASE 3: GPU Optimized (Im2Col + GEMM)

### Bước 1: Compile
```bash
cd phase3_gpu_optimized
make -f MAKEFILE clean
make -f MAKEFILE
```

### Bước 2: Test 1 Ảnh (Detailed Timing)
```bash
./test_gpu
```
**Chức năng**:
- Load 1 ảnh từ CIFAR-10
- Chạy qua autoencoder với Im2Col + GEMM
- In ra timing chi tiết từng layer
- Hiển thị output sample

### Bước 3: Benchmark 60,000 Ảnh
```bash
./run_phase3
```
**Chức năng**:
- Load 1 ảnh từ CIFAR-10
- Chạy 60,000 lần với Im2Col + GEMM
- Batch size = 1
- In ra: Total time, average time per image
- So sánh với Phase 2

---

## PHASE 3 V2: GPU Optimized + Batch + Kernel Fusion

### Bước 1: Compile
```bash
cd phase3_gpu_optimized_v2
make -f MAKEFILE clean
make -f MAKEFILE
```

### Bước 2: Test 1 Ảnh (Detailed Timing)
```bash
./test_gpu
```
**Chức năng**:
- Load 1 ảnh từ CIFAR-10
- Replicate thành batch size = 1
- Chạy qua autoencoder với kernel fusion
- In ra timing chi tiết (ReLU = 0ms vì đã fused)
- Hiển thị output sample

### Bước 3: Benchmark 60,000 Ảnh (Full Autoencoder)
```bash
./run_phase3
```
**Chức năng**:
- Load 1 ảnh từ CIFAR-10
- Replicate thành batch size = 32
- Chạy 1,875 batches (60,000 / 32)
- Mỗi batch: 32 ảnh giống nhau
- In ra: Total time, average time per image
- Progress bar mỗi 100 batches

### Bước 4: Feature Extraction (Encoder Only)
```bash
./feature_extract
```
**Chức năng**:
- Chạy **60,000 ảnh** qua encoder only (không có decoder)
- Batch size = 32 → 1,875 batches
- **Lưu ý**: Code test dùng 1 ảnh mẫu replicate để đo performance
- Output: Latent vectors 8×8×128 cho mỗi ảnh
- Nhanh hơn full autoencoder ~2.8× (vì bỏ decoder)
- In ra: Total time, average time per image, progress bar

---

## So Sánh Các Phases

### Test (1 ảnh):
```bash
# Chạy lần lượt
cd phase2_gpu_basic && ./test_gpu
cd ../phase3_gpu_optimized && ./test_gpu
cd ../phase3_gpu_optimized_v2 && ./test_gpu
```
**Mục đích**: So sánh latency cho single image processing

### Benchmark (60k ảnh):
```bash
# Chạy lần lượt
cd phase2_gpu_basic && ./run_gpu
cd ../phase3_gpu_optimized && ./run_phase3
cd ../phase3_gpu_optimized_v2 && ./run_phase3
```
**Mục đích**: So sánh throughput cho batch processing

### Feature Extraction:
```bash
cd phase3_gpu_optimized_v2 && ./feature_extract
```
**Mục đích**: Đo performance cho inference (encoder only)

---

## Chi Tiết Kỹ Thuật

### Architecture (256/128 channels):
```
Input: 32×32×3
├─ Conv1: 3→256, ReLU, MaxPool(2×2)
├─ Conv2: 256→128, ReLU, MaxPool(2×2)
├─ Latent: 8×8×128 (8,192 dims)
├─ DeConv1: 128→128, ReLU, Upsample(2×)
├─ DeConv2: 128→256, ReLU, Upsample(2×)
└─ FinalConv: 256→3
Output: 32×32×3
```

### Batch Processing (Phase 3 V2):
- **Test**: batch=1 (để so sánh )
- **Benchmark**: batch=32 (tối ưu throughput)
- **Feature Extract**: batch=32 (production use case)

### Kernel Fusion (Phase 3 V2):
- Conv + ReLU fused into single kernel
- Loại bỏ separate ReLU kernel launch
- ReLU time = 0ms trong output

---

## Troubleshooting

### Compile lỗi:
```bash
# Check CUDA version
nvcc --version

# Check CUDA path
which nvcc
```

### Runtime lỗi:
```bash
# Check GPU
nvidia-smi

# Check dataset
ls -la data/cifar-10-batches-bin/
```

