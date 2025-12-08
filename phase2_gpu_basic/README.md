# Phase 2: GPU Basic Autoencoder

## VN

### Giới thiệu

Phase 2 triển khai một **Convolutional Autoencoder** hoàn chỉnh trên GPU sử dụng CUDA. Đây là bước đầu tiên trong việc tối ưu hóa mô hình neural network bằng GPU, với kiến trúc encoder-decoder cơ bản.

### Kiến trúc mô hình

**Autoencoder** bao gồm 2 phần chính:

#### 1. Encoder (Nén ảnh)
- **Conv1**: 3 → 8 channels (3×3 kernel)
- **ReLU**: Hàm kích hoạt
- **MaxPool1**: Giảm kích thước 2× (32×32 → 16×16)
- **Conv2**: 8 → 4 channels (3×3 kernel)
- **ReLU**: Hàm kích hoạt
- **MaxPool2**: Giảm kích thước 2× (16×16 → 8×8)

#### 2. Decoder (Giải nén ảnh)
- **Upsample1**: Tăng kích thước 2× (8×8 → 16×16)
- **Conv3**: 4 → 8 channels (3×3 kernel)
- **Upsample2**: Tăng kích thước 2× (16×16 → 32×32)
- **Conv4**: 8 → 3 channels (3×3 kernel, tái tạo ảnh gốc)

### Cấu trúc thư mục

```
phase2_gpu_basic/
   Include/
      autoencoder.h          # Khai báo class Autoencoder
      utils/
          cuda_utils.h       # Tiện ích kiểm tra lỗi CUDA
          gpu_memory.h       # Quản lý bộ nhớ GPU
   src/
      main_gpu.cu            # Chương trình chính
      autoencoder.cu         # Triển khai Autoencoder
      kernels/               # CUDA kernels
         conv2d.cu          # Convolution 2D
         maxpool.cu         # Max Pooling 2D
         relu.cu            # ReLU activation
         upsample.cu        # Upsampling 2D
      utils/
          cuda_utils.cpp     # Tiện ích CUDA
          gpu_memory.cu      # Quản lý bộ nhớ GPU
   MAKEFILE                   # Build script
   README.md                  # File này
```

### Chi tiết từng thành phần

#### 1. **main_gpu.cu**
- Đọc ảnh thực từ CIFAR-10 dataset (32×32×3)
- Hàm `loadCIFAR10Image()`: Đọc và xử lý binary format CIFAR-10
  - Đọc 1 byte label + 3072 bytes pixel data
  - Chuyển đổi từ [0, 255] sang [0.0, 1.0]
  - Chuẩn hóa định dạng CHW (Channel-Height-Width)
- Khởi tạo Autoencoder trên GPU
- Chạy forward pass với ảnh thực
- Hiển thị kết quả đầu ra

#### 2. **autoencoder.cu**
- **Constructor**: Cấp phát bộ nhớ GPU cho:
  - Input/output buffers
  - Intermediate feature maps (conv, pool, upsample)
  - Weights cho các lớp Conv
- **forward()**: Thực hiện forward pass qua toàn bộ network
- **Destructor**: Giải phóng bộ nhớ GPU

#### 3. **CUDA Kernels**

##### **conv2d.cu**
- Thực hiện convolution 2D với kernel 3×3
- Sử dụng grid-stride loop
- Tối ưu với `#pragma unroll`

##### **maxpool.cu**
- Max pooling với kernel 2×2, stride 2
- Giảm kích thước ảnh xuống 1/2

##### **relu.cu**
- Hàm kích hoạt ReLU: `f(x) = max(0, x)`
- Xử lý song song trên từng phần tử

##### **upsample.cu**
- Nearest neighbor upsampling
- Tăng kích thước ảnh lên 2×
- Mỗi pixel được lặp lại 4 lần (2×2)

#### 4. **Utilities**

##### **gpu_memory.cu**
- `gpu_malloc()`: Cấp phát bộ nhớ GPU
- `gpu_free()`: Giải phóng bộ nhớ GPU
- `gpu_memcpy_h2d()`: Copy CPU → GPU
- `gpu_memcpy_d2h()`: Copy GPU → CPU

##### **cuda_utils.cpp**
- `checkCuda()`: Kiểm tra lỗi CUDA
- `device_synchronize()`: Đồng bộ GPU

### Yêu cầu hệ thống

- **GPU**: NVIDIA GPU với CUDA support
- **CUDA Toolkit**: Version 11.0 trở lên
- **Compiler**: nvcc (đi kèm CUDA Toolkit)
- **C++ Standard**: C++17
- **Dataset**: CIFAR-10 binary format (đặt trong `../data/cifar-10-batches-bin/`)

### Cách biên dịch và chạy

#### 0. Chuẩn bị dataset

Đảm bảo CIFAR-10 dataset có trong thư mục:
```
../data/cifar-10-batches-bin/
   data_batch_1.bin   # 10,000 ảnh training
   data_batch_2.bin   # 10,000 ảnh training
   data_batch_3.bin   # 10,000 ảnh training
   data_batch_4.bin   # 10,000 ảnh training
   data_batch_5.bin   # 10,000 ảnh training
   test_batch.bin     # 10,000 ảnh testing
```

**CIFAR-10 Labels:**
- 0: airplane
- 1: automobile
- 2: bird
- 3: cat
- 4: deer
- 5: dog
- 6: frog
- 7: horse
- 8: ship
- 9: truck

#### 1. Biên dịch chương trình
make -f MAKEFILE clean

```bash
make -f MAKEFILE run_gpu
```


Hoặc với lệnh nvcc trực tiếp:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 \
    src/main_gpu.cu \
    src/autoencoder.cu \
    src/utils/cuda_utils.cpp \
    src/utils/gpu_memory.cu \
    src/kernels/conv2d.cu \
    src/kernels/maxpool.cu \
    src/kernels/relu.cu \
    src/kernels/upsample.cu \
    -I Include -o run_gpu
```

#### 2. Chạy chương trình

```bash
./run_gpu
```

#### 3. Kết quả mẫu

```
===== Phase 2: GPU Basic Autoencoder Test =====
[INFO] Loading image from CIFAR-10 dataset...
[INFO] File: ../data/cifar-10-batches-bin/data_batch_1.bin
[INFO] Image index: 0
[INFO] Image loaded successfully!
[INFO] Image label: 6
[INFO] Image size: 3x32x32

===== INPUT SAMPLE VALUES (first 10 pixels) =====
0.231373 0.168627 0.196078 0.266667 0.384314 0.466667 0.545098 0.568627 0.584314 0.584314

[INFO] Initializing GPU Autoencoder...
[INFO] GPU Autoencoder initialization done.

[INFO] Running GPU forward pass...
[INFO] Running forward pass...
[INFO] Forward pass completed.

===== OUTPUT SAMPLE VALUES (first 10 pixels) =====
0.00121097 0.0022286 0.00264075 0.00316231 0.00327172 0.00349064 0.00360014 0.00370975 0.00370985 0.0037072

===== DONE =====
[INFO] Freeing GPU memory...
[INFO] GPU memory freed.
```

**Giải thích kết quả:**
- Label 6 = "frog" (con ếch trong CIFAR-10)
- Input values: Giá trị pixel thực từ ảnh CIFAR-10 (đã chuẩn hóa [0, 1])
- Output values: Ảnh tái tạo sau khi qua autoencoder

#### 4. Tùy chỉnh ảnh đầu vào

Mở file `src/main_gpu.cu` và chỉnh sửa:

```cpp
// Chọn file batch (1-5 cho training, test_batch.bin cho testing)
string dataPath = "../data/cifar-10-batches-bin/data_batch_1.bin";

// Chọn ảnh trong batch (0-9999)
int imageIndex = 0;
```

**Ví dụ:**
```cpp
// Đọc ảnh thứ 100 từ batch 3
string dataPath = "../data/cifar-10-batches-bin/data_batch_3.bin";
int imageIndex = 100;
```

#### 5. Xóa file thực thi

```bash
make -f MAKEFILE clean
```

### Thông tin GPU

Kiểm tra GPU có sẵn:

```bash
nvidia-smi
```

### Hiệu năng

- **Input size**: 32×32×3 (RGB image)
- **Latent size**: 8×8×4 (compressed 16×)
- **Output size**: 32×32×3 (reconstructed image)
- **GPU Memory**: ~10MB cho buffers và weights
- **Execution time**: < 1ms trên GTX 1650

### Lưu ý kỹ thuật

1. **Memory Management**
   - Tất cả bộ nhớ GPU được quản lý thông qua RAII (Resource Acquisition Is Initialization)
   - Destructor tự động giải phóng khi object bị hủy

2. **Kernel Configuration**
   - Block size: 16×16 threads
   - Grid size: Tự động tính dựa trên kích thước input

3. **Error Handling**
   - Mọi CUDA API call đều được kiểm tra lỗi
   - Sử dụng `checkCuda()` để báo lỗi chi tiết

4. **Optimization**
   - Sử dụng `__restrict__` để tối ưu memory access
   - `#pragma unroll` cho các vòng lặp nhỏ
   - Coalesced memory access pattern

### Troubleshooting

**Lỗi: `nvcc: command not found`**
```bash
# Thêm CUDA vào PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Lỗi: `no CUDA-capable device is detected`**
- Kiểm tra GPU với `nvidia-smi`
- Đảm bảo driver NVIDIA đã được cài đặt

**Lỗi biên dịch: `undefined reference to kernel`**
- Đảm bảo tất cả file `.cu` được thêm vào MAKEFILE
- Kiểm tra tên kernel khớp giữa khai báo và định nghĩa

**Lỗi: `[ERROR] Cannot open file: ../data/cifar-10-batches-bin/data_batch_1.bin`**
- Kiểm tra đường dẫn đến CIFAR-10 dataset
- Đảm bảo file tồn tại: `ls ../data/cifar-10-batches-bin/`
- Download CIFAR-10 từ: https://www.cs.toronto.edu/~kriz/cifar.html

---

## EN

### Introduction

Phase 2 implements a complete **Convolutional Autoencoder** on GPU using CUDA. This is the first step in optimizing neural network models with GPU, featuring a basic encoder-decoder architecture.

### Model Architecture

**Autoencoder** consists of 2 main parts:

#### 1. Encoder (Compression)
- **Conv1**: 3 → 8 channels (3×3 kernel)
- **ReLU**: Activation function
- **MaxPool1**: 2× downsampling (32×32 → 16×16)
- **Conv2**: 8 → 4 channels (3×3 kernel)
- **ReLU**: Activation function
- **MaxPool2**: 2× downsampling (16×16 → 8×8)

#### 2. Decoder (Decompression)
- **Upsample1**: 2× upsampling (8×8 → 16×16)
- **Conv3**: 4 → 8 channels (3×3 kernel)
- **Upsample2**: 2× upsampling (16×16 → 32×32)
- **Conv4**: 8 → 3 channels (3×3 kernel, reconstruct original image)

### Directory Structure

```
phase2_gpu_basic/
   Include/
      autoencoder.h          # Autoencoder class declaration
      utils/
          cuda_utils.h       # CUDA error checking utilities
          gpu_memory.h       # GPU memory management
   src/
      main_gpu.cu            # Main program
      autoencoder.cu         # Autoencoder implementation
      kernels/               # CUDA kernels
         conv2d.cu          # 2D Convolution
         maxpool.cu         # 2D Max Pooling
         relu.cu            # ReLU activation
         upsample.cu        # 2D Upsampling
      utils/
          cuda_utils.cpp     # CUDA utilities
          gpu_memory.cu      # GPU memory management
   MAKEFILE                   # Build script
   README.md                  # This file
```

### Component Details

#### 1. **main_gpu.cu**
- Loads real images from CIFAR-10 dataset (32×32×3)
- `loadCIFAR10Image()` function: Reads and processes CIFAR-10 binary format
  - Reads 1 byte label + 3072 bytes pixel data
  - Converts from [0, 255] to [0.0, 1.0]
  - Normalizes to CHW format (Channel-Height-Width)
- Initializes Autoencoder on GPU
- Runs forward pass with real images
- Displays output results

#### 2. **autoencoder.cu**
- **Constructor**: Allocates GPU memory for:
  - Input/output buffers
  - Intermediate feature maps (conv, pool, upsample)
  - Weights for Conv layers
- **forward()**: Executes forward pass through entire network
- **Destructor**: Frees GPU memory

#### 3. **CUDA Kernels**

##### **conv2d.cu**
- Performs 2D convolution with 3×3 kernel
- Uses grid-stride loop
- Optimized with `#pragma unroll`

##### **maxpool.cu**
- Max pooling with 2×2 kernel, stride 2
- Reduces image size to 1/2

##### **relu.cu**
- ReLU activation function: `f(x) = max(0, x)`
- Parallel processing on each element

##### **upsample.cu**
- Nearest neighbor upsampling
- Increases image size by 2×
- Each pixel is replicated 4 times (2×2)

#### 4. **Utilities**

##### **gpu_memory.cu**
- `gpu_malloc()`: Allocate GPU memory
- `gpu_free()`: Free GPU memory
- `gpu_memcpy_h2d()`: Copy CPU → GPU
- `gpu_memcpy_d2h()`: Copy GPU → CPU

##### **cuda_utils.cpp**
- `checkCuda()`: Check CUDA errors
- `device_synchronize()`: Synchronize GPU

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **CUDA Toolkit**: Version 11.0 or higher
- **Compiler**: nvcc (included with CUDA Toolkit)
- **C++ Standard**: C++17
- **Dataset**: CIFAR-10 binary format (place in `../data/cifar-10-batches-bin/`)

### Build and Run

#### 0. Prepare dataset

Ensure CIFAR-10 dataset is in the directory:
```
../data/cifar-10-batches-bin/
   data_batch_1.bin   # 10,000 training images
   data_batch_2.bin   # 10,000 training images
   data_batch_3.bin   # 10,000 training images
   data_batch_4.bin   # 10,000 training images
   data_batch_5.bin   # 10,000 training images
   test_batch.bin     # 10,000 testing images
```

**CIFAR-10 Labels:**
- 0: airplane
- 1: automobile
- 2: bird
- 3: cat
- 4: deer
- 5: dog
- 6: frog
- 7: horse
- 8: ship
- 9: truck

#### 1. Compile the program

```bash
make -f MAKEFILE run_gpu
```

Or with direct nvcc command:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 \
    src/main_gpu.cu \
    src/autoencoder.cu \
    src/utils/cuda_utils.cpp \
    src/utils/gpu_memory.cu \
    src/kernels/conv2d.cu \
    src/kernels/maxpool.cu \
    src/kernels/relu.cu \
    src/kernels/upsample.cu \
    -I Include -o run_gpu
```

#### 2. Run the program

```bash
./run_gpu
```

#### 3. Sample Output

```
===== Phase 2: GPU Basic Autoencoder Test =====
[INFO] Loading image from CIFAR-10 dataset...
[INFO] File: ../data/cifar-10-batches-bin/data_batch_1.bin
[INFO] Image index: 0
[INFO] Image loaded successfully!
[INFO] Image label: 6
[INFO] Image size: 3x32x32

===== INPUT SAMPLE VALUES (first 10 pixels) =====
0.231373 0.168627 0.196078 0.266667 0.384314 0.466667 0.545098 0.568627 0.584314 0.584314

[INFO] Initializing GPU Autoencoder...
[INFO] GPU Autoencoder initialization done.

[INFO] Running GPU forward pass...
[INFO] Running forward pass...
[INFO] Forward pass completed.

===== OUTPUT SAMPLE VALUES (first 10 pixels) =====
0.00121097 0.0022286 0.00264075 0.00316231 0.00327172 0.00349064 0.00360014 0.00370975 0.00370985 0.0037072

===== DONE =====
[INFO] Freeing GPU memory...
[INFO] GPU memory freed.
```

**Result Explanation:**
- Label 6 = "frog" in CIFAR-10
- Input values: Real pixel values from CIFAR-10 image (normalized to [0, 1])
- Output values: Reconstructed image after passing through autoencoder

#### 4. Customize input image

Open `src/main_gpu.cu` and modify:

```cpp
// Select batch file (1-5 for training, test_batch.bin for testing)
string dataPath = "../data/cifar-10-batches-bin/data_batch_1.bin";

// Select image in batch (0-9999)
int imageIndex = 0;
```

**Example:**
```cpp
// Read image #100 from batch 3
string dataPath = "../data/cifar-10-batches-bin/data_batch_3.bin";
int imageIndex = 100;
```

#### 5. Clean build files

```bash
make -f MAKEFILE clean
```

### GPU Information

Check available GPU:

```bash
nvidia-smi
```

### Performance

- **Input size**: 32×32×3 (RGB image)
- **Latent size**: 8×8×4 (compressed 16×)
- **Output size**: 32×32×3 (reconstructed image)
- **GPU Memory**: ~10MB for buffers and weights
- **Execution time**: < 1ms on GTX 1650

### Technical Notes

1. **Memory Management**
   - All GPU memory is managed through RAII (Resource Acquisition Is Initialization)
   - Destructor automatically frees memory when object is destroyed

2. **Kernel Configuration**
   - Block size: 16×16 threads
   - Grid size: Automatically calculated based on input size

3. **Error Handling**
   - All CUDA API calls are error-checked
   - Uses `checkCuda()` for detailed error reporting

4. **Optimization**
   - Uses `__restrict__` for optimized memory access
   - `#pragma unroll` for small loops
   - Coalesced memory access pattern

### Troubleshooting

**Error: `nvcc: command not found`**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Error: `no CUDA-capable device is detected`**
- Check GPU with `nvidia-smi`
- Ensure NVIDIA driver is installed

**Compilation error: `undefined reference to kernel`**
- Ensure all `.cu` files are added to MAKEFILE
- Check kernel names match between declaration and definition

**Error: `[ERROR] Cannot open file: ../data/cifar-10-batches-bin/data_batch_1.bin`**
- Check the path to CIFAR-10 dataset
- Ensure file exists: `ls ../data/cifar-10-batches-bin/`
- Download CIFAR-10 from: https://www.cs.toronto.edu/~kriz/cifar.html

