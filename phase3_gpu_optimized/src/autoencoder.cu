#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// Include header của bạn
#include "autoencoder.h"
#include "utils/gpu_memory.h"
#include "utils/cuda_utils.h"

// ==========================================
// 1. EXTERNAL KERNEL DECLARATIONS
// ==========================================
// Kernel biến đổi ảnh thành ma trận cột (Im2Col)
void im2col_gpu(const float* data_im, float* data_col, 
                int channels, int height, int width, 
                int ksize, int pad, int stride, 
                int h_out, int w_out);

// Kernel nhân ma trận tối ưu (Tiled GEMM)
extern "C" __global__ void gemm_tiled(const float* A, const float* B, float* C, 
                                      int M, int N, int K);

// Các kernel Activation/Pooling
extern "C" __global__ void relu(float* x, int size);
extern "C" __global__ void maxpool(const float* input, float* output, int H, int W, int C);
extern "C" __global__ void upsample(const float* input, float* output, int H, int W, int C);


// ==========================================
// 2. HELPER FUNCTION: CONV LAYER VIA GEMM
// ==========================================
void forward_conv_layer(const float* d_input, const float* d_weights, float* d_output, float* d_col_buffer,
                        int H, int W, int C_in, int C_out) 
{
    // Tham số cố định theo đề bài: Kernel 3x3, Padding 1, Stride 1
    int ksize = 3; 
    int pad = 1; 
    int stride = 1;
    
    // Output dimensions (giữ nguyên H, W do pad=1, stride=1)
    int H_out = H; 
    int W_out = W;

    // --- BƯỚC 1: Im2Col (Biến đổi Input -> Col Matrix) ---
    // Input: (C_in, H, W) -> Output: (C_in * K * K, H_out * W_out)
    im2col_gpu(d_input, d_col_buffer, C_in, H, W, ksize, pad, stride, H_out, W_out);

    // --- BƯỚC 2: GEMM (Weights * Col) ---
    // Matrix A (Weights): [C_out] x [C_in * K * K]
    // Matrix B (Col):     [C_in * K * K] x [H_out * W_out]
    // Matrix C (Output):  [C_out] x [H_out * W_out] -> (C_out, H, W)
    
    int M = C_out;                      // Rows of A and C (Số filters)
    int N = H_out * W_out;              // Cols of B and C (Số pixels)
    int K = C_in * ksize * ksize;       // Cols of A, Rows of B (Kích thước kernel phẳng)

    // Grid size cho GEMM
    dim3 dimGrid((N + 15)/16, (M + 15)/16);
    dim3 dimBlock(16, 16);
    
    gemm_tiled<<<dimGrid, dimBlock>>>(d_weights, d_col_buffer, d_output, M, N, K);
    
    // Kiểm tra lỗi sau mỗi lớp Conv (tùy chọn, có thể comment để tăng tốc tối đa)
    // checkCuda(cudaGetLastError(), "Gemm Conv Layer execution");
}

// ==========================================
// 3. CONSTRUCTOR & INITIALIZATION
// ==========================================
Autoencoder::Autoencoder(int H_, int W_, int C_)
    : H(H_), W(W_), C(C_)
{
    std::cout << "[INFO] Initializing Phase 3 Autoencoder (Im2Col + GEMM)..." << std::endl;

    // Định nghĩa kích thước các tầng (Architecture Spec)
    int C1 = 256; // Encoder Layer 1
    int C2 = 128; // Encoder Layer 2 (Latent)

    // --- A. KHỞI TẠO TRỌNG SỐ (HOST) ---
    // Dùng He Initialization đơn giản hoặc Random nhỏ
    auto init_vec = [](std::vector<float>& v, int size) {
        v.resize(size);
        for(int i=0; i<size; i++) v[i] = ((float)rand()/RAND_MAX) * 0.02f - 0.01f;
    };

    // 1. Encoder Conv1: 3 -> 256
    init_vec(w_conv1, C1 * C * 3 * 3);
    // 2. Encoder Conv2: 256 -> 128
    init_vec(w_conv2, C2 * C1 * 3 * 3);
    // 3. Decoder Conv1: 128 -> 128
    init_vec(w_dec1, C2 * C2 * 3 * 3);
    // 4. Decoder Conv2: 128 -> 256
    init_vec(w_dec2, C1 * C2 * 3 * 3);
    // 5. Final Conv: 256 -> 3
    init_vec(w_final, C * C1 * 3 * 3);

    // --- B. CẤP PHÁT BỘ NHỚ TRỌNG SỐ (GPU) ---
    d_w_conv1 = (float*) gpu_malloc(w_conv1.size() * sizeof(float));
    d_w_conv2 = (float*) gpu_malloc(w_conv2.size() * sizeof(float));
    d_w_dec1  = (float*) gpu_malloc(w_dec1.size() * sizeof(float));
    d_w_dec2  = (float*) gpu_malloc(w_dec2.size() * sizeof(float));
    d_w_final = (float*) gpu_malloc(w_final.size() * sizeof(float));

    // Copy weights H2D
    gpu_memcpy_h2d(d_w_conv1, w_conv1.data(), w_conv1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_conv2, w_conv2.data(), w_conv2.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_dec1,  w_dec1.data(),  w_dec1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_dec2,  w_dec2.data(),  w_dec2.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_final, w_final.data(), w_final.size() * sizeof(float));

    // --- C. CẤP PHÁT FEATURE MAPS (GPU) ---
    // Input
    d_input     = (float*)gpu_malloc(C * H * W * sizeof(float));
    
    // Encoder Maps
    d_conv1_out = (float*)gpu_malloc(C1 * H * W * sizeof(float));
    d_pool1_out = (float*)gpu_malloc(C1 * (H/2) * (W/2) * sizeof(float));
    d_conv2_out = (float*)gpu_malloc(C2 * (H/2) * (W/2) * sizeof(float));
    d_pool2_out = (float*)gpu_malloc(C2 * (H/4) * (W/4) * sizeof(float)); // Latent Space

    // Decoder Maps
    d_dec1_out  = (float*)gpu_malloc(C2 * (H/4) * (W/4) * sizeof(float)); 
    d_ups1_out  = (float*)gpu_malloc(C2 * (H/2) * (W/2) * sizeof(float));
    d_dec2_out  = (float*)gpu_malloc(C1 * (H/2) * (W/2) * sizeof(float)); // New buffer for Dec2
    d_ups2_out  = (float*)gpu_malloc(C1 * H * W * sizeof(float));
    
    // Output
    d_output    = (float*)gpu_malloc(C * H * W * sizeof(float));

    // --- D. CẤP PHÁT IM2COL BUFFER (Optimization) ---
    // Tính kích thước lớn nhất cần dùng cho buffer này.
    // Max Channels Input = 256 (tại lớp final conv layer input)
    // Max Spatial = 32x32
    // K = 3x3 = 9
    // Size = 256 * 9 * 32 * 32 * 4 bytes ~= 9MB
    size_t max_col_size = 256 * 9 * 32 * 32 * sizeof(float);
    d_col_buffer = (float*)gpu_malloc(max_col_size);
    
    std::cout << "[INFO] GPU Memory Allocated. Ready for Phase 3." << std::endl;
}

// ==========================================
// 4. DESTRUCTOR
// ==========================================
Autoencoder::~Autoencoder() {
    // Free Feature Maps
    gpu_free(d_input);
    gpu_free(d_conv1_out); gpu_free(d_pool1_out);
    gpu_free(d_conv2_out); gpu_free(d_pool2_out);
    gpu_free(d_dec1_out);  gpu_free(d_ups1_out);
    gpu_free(d_dec2_out);  gpu_free(d_ups2_out);
    gpu_free(d_output);
    
    // Free Weights
    gpu_free(d_w_conv1); gpu_free(d_w_conv2);
    gpu_free(d_w_dec1);  gpu_free(d_w_dec2);
    gpu_free(d_w_final);
    
    // Free Opt Buffer
    gpu_free(d_col_buffer);
}

// ==========================================
// 5. HELPER STATIC
// ==========================================
void Autoencoder::device_synchronize() {
    cudaDeviceSynchronize();
}

void Autoencoder::init_weights() {
    // Placeholder nếu muốn re-init ngẫu nhiên lại từ Host
}

// ==========================================
// 6. FORWARD PASS (FULL PIPELINE)
// ==========================================
void Autoencoder::forward(const float* host_input, float* host_output)
{
    // Kích thước không gian
    int H2 = H / 2; int W2 = W / 2;
    int H4 = H / 4; int W4 = W / 4;
    
    // Kích thước kênh
    int C1 = 256;
    int C2 = 128;
    
    // Block size chuẩn cho pooling/upsample
    dim3 block(16, 16);

    // 0. Copy Input H2D
    gpu_memcpy_h2d(d_input, host_input, C * H * W * sizeof(float));

    // ---------------- ENCODER ----------------
    
    // 1. Conv1: Input(3) -> 256 | Size: 32x32
    forward_conv_layer(d_input, d_w_conv1, d_conv1_out, d_col_buffer, H, W, C, C1);
    relu<<<(C1*H*W + 255)/256, 256>>>(d_conv1_out, C1 * H * W);

    // 2. Pool1: 32x32 -> 16x16
    dim3 grid_pool1((W2 + 15) / 16, (H2 + 15) / 16, C1);
    maxpool<<<grid_pool1, block>>>(d_conv1_out, d_pool1_out, H, W, C1);

    // 3. Conv2: 256 -> 128 | Size: 16x16
    forward_conv_layer(d_pool1_out, d_w_conv2, d_conv2_out, d_col_buffer, H2, W2, C1, C2);
    relu<<<(C2*H2*W2 + 255)/256, 256>>>(d_conv2_out, C2 * H2 * W2);

    // 4. Pool2: 16x16 -> 8x8 (LATENT SPACE)
    dim3 grid_pool2((W4 + 15) / 16, (H4 + 15) / 16, C2);
    maxpool<<<grid_pool2, block>>>(d_conv2_out, d_pool2_out, H2, W2, C2);


    // ---------------- DECODER ----------------

    // 5. Decoder Conv1: 128 -> 128 | Size: 8x8
    forward_conv_layer(d_pool2_out, d_w_dec1, d_dec1_out, d_col_buffer, H4, W4, C2, C2);
    relu<<<(C2*H4*W4 + 255)/256, 256>>>(d_dec1_out, C2 * H4 * W4);

    // 6. Upsample1: 8x8 -> 16x16
    dim3 grid_up1((W2 + 15) / 16, (H2 + 15) / 16, C2);
    upsample<<<grid_up1, block>>>(d_dec1_out, d_ups1_out, H4, W4, C2);

    // 7. Decoder Conv2: 128 -> 256 | Size: 16x16
    forward_conv_layer(d_ups1_out, d_w_dec2, d_dec2_out, d_col_buffer, H2, W2, C2, C1);
    relu<<<(C1*H2*W2 + 255)/256, 256>>>(d_dec2_out, C1 * H2 * W2);

    // 8. Upsample2: 16x16 -> 32x32
    dim3 grid_up2((W + 15) / 16, (H + 15) / 16, C1);
    upsample<<<grid_up2, block>>>(d_dec2_out, d_ups2_out, H2, W2, C1);

    // 9. Final Conv: 256 -> 3 | Size: 32x32
    // Lưu ý: Lớp cuối thường không dùng ReLU (để tái tạo dải màu đầy đủ)
    forward_conv_layer(d_ups2_out, d_w_final, d_output, d_col_buffer, H, W, C1, C);


    // ---------------- FINISH ----------------

    // 10. Copy Output D2H
    gpu_memcpy_d2h(host_output, d_output, C * H * W * sizeof(float));
    
    // Đợi GPU hoàn thành để đảm bảo timing chính xác
    cudaDeviceSynchronize();
}