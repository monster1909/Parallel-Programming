#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "autoencoder.h"
#include "utils/gpu_memory.h"
#include "utils/cuda_utils.h"

// Update im2col signature
void im2col_gpu(const float* data_im, float* data_col, int batch_size, int channels, int height, int width, int ksize, int pad, int stride, int h_out, int w_out);
extern "C" __global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K);
extern "C" __global__ void relu(float* x, int size);
extern "C" __global__ void maxpool(const float* input, float* output, int H, int W, int C);
extern "C" __global__ void upsample(const float* input, float* output, int H, int W, int C);

// Helper function: Batch Conv
void forward_conv_layer(const float* d_input, const float* d_weights, float* d_output, float* d_col_buffer,
                        int batch_size, int H, int W, int C_in, int C_out) 
{
    int ksize = 3, pad = 1, stride = 1;
    
    // 1. Im2Col với Batch Size
    im2col_gpu(d_input, d_col_buffer, batch_size, C_in, H, W, ksize, pad, stride, H, W);

    // 2. GEMM
    int M = C_out;              
    int N = batch_size * H * W; // Kích thước chiều ngang ma trận tăng gấp Batch lần
    int K = C_in * ksize * ksize; 

    dim3 dimGrid((N + 15)/16, (M + 15)/16);
    dim3 dimBlock(16, 16);
    gemm_tiled<<<dimGrid, dimBlock>>>(d_weights, d_col_buffer, d_output, M, N, K);
}

// Constructor cập nhật
Autoencoder::Autoencoder(int H_, int W_, int C_, int max_batch_)
    : H(H_), W(W_), C(C_), MAX_B(max_batch_)
{
    std::cout << "[INFO] Init Batch Autoencoder (Max Batch: " << MAX_B << ")..." << std::endl;

    int C1 = 256, C2 = 128;
    // Weights init (Giữ nguyên phần này như cũ)...
    auto init_vec = [](std::vector<float>& v, int size) { v.resize(size, 0.01f); };
    init_vec(w_conv1, C1 * C * 9); init_vec(w_conv2, C2 * C1 * 9);
    init_vec(w_dec1, C2 * C2 * 9); init_vec(w_dec2, C1 * C2 * 9); init_vec(w_final, C * C1 * 9);

    d_w_conv1 = (float*) gpu_malloc(w_conv1.size()*4); gpu_memcpy_h2d(d_w_conv1, w_conv1.data(), w_conv1.size()*4);
    d_w_conv2 = (float*) gpu_malloc(w_conv2.size()*4); gpu_memcpy_h2d(d_w_conv2, w_conv2.data(), w_conv2.size()*4);
    d_w_dec1  = (float*) gpu_malloc(w_dec1.size()*4);  gpu_memcpy_h2d(d_w_dec1, w_dec1.data(), w_dec1.size()*4);
    d_w_dec2  = (float*) gpu_malloc(w_dec2.size()*4);  gpu_memcpy_h2d(d_w_dec2, w_dec2.data(), w_dec2.size()*4);
    d_w_final = (float*) gpu_malloc(w_final.size()*4); gpu_memcpy_h2d(d_w_final, w_final.data(), w_final.size()*4);

    // --- CẤP PHÁT FEATURE MAPS (NHÂN VỚI MAX_B) ---
    // Mấu chốt: Cấp phát đủ chỗ cho cả Batch
    d_input     = (float*)gpu_malloc(MAX_B * C * H * W * 4);
    
    d_conv1_out = (float*)gpu_malloc(MAX_B * C1 * H * W * 4);
    d_pool1_out = (float*)gpu_malloc(MAX_B * C1 * (H/2) * (W/2) * 4);
    d_conv2_out = (float*)gpu_malloc(MAX_B * C2 * (H/2) * (W/2) * 4);
    d_pool2_out = (float*)gpu_malloc(MAX_B * C2 * (H/4) * (W/4) * 4);

    d_dec1_out  = (float*)gpu_malloc(MAX_B * C2 * (H/4) * (W/4) * 4); 
    d_ups1_out  = (float*)gpu_malloc(MAX_B * C2 * (H/2) * (W/2) * 4);
    d_dec2_out  = (float*)gpu_malloc(MAX_B * C1 * (H/2) * (W/2) * 4);
    d_ups2_out  = (float*)gpu_malloc(MAX_B * C1 * H * W * 4);
    
    d_output    = (float*)gpu_malloc(MAX_B * C * H * W * 4);

    // Buffer Im2Col cũng phải đủ lớn cho Batch
    // Size = C_in * 9 * (Batch * H * W)
    size_t max_col_size = 256 * 9 * MAX_B * 32 * 32 * sizeof(float);
    d_col_buffer = (float*)gpu_malloc(max_col_size);
}

Autoencoder::~Autoencoder() {
    gpu_free(d_input); gpu_free(d_output); gpu_free(d_col_buffer);
    gpu_free(d_conv1_out); gpu_free(d_pool1_out); gpu_free(d_conv2_out); gpu_free(d_pool2_out);
    gpu_free(d_dec1_out); gpu_free(d_ups1_out); gpu_free(d_dec2_out); gpu_free(d_ups2_out);
    gpu_free(d_w_conv1); gpu_free(d_w_conv2); gpu_free(d_w_dec1); gpu_free(d_w_dec2); gpu_free(d_w_final);
}

void Autoencoder::forward(const float* host_input, float* host_output, int batch_size)
{
    // Nếu batch_size > MAX_B thì báo lỗi hoặc kẹp lại, ở đây giả sử luôn <= MAX_B
    int B = batch_size;
    
    // Copy Input Batch
    gpu_memcpy_h2d(d_input, host_input, B * C * H * W * sizeof(float));

    // Encoder
    forward_conv_layer(d_input, d_w_conv1, d_conv1_out, d_col_buffer, B, H, W, C, 256);
    relu<<<(B*256*H*W+255)/256, 256>>>(d_conv1_out, B*256*H*W); // Chú ý nhân B

    dim3 block(16, 16);
    // Grid Z phải nhân với Batch Size cho pooling/upsample vì kernel cũ dùng blockIdx.z làm channel
    // Tuy nhiên kernel MaxPool cũ của bạn: int c = blockIdx.z;
    // Nếu Input là (Batch, C, H, W), thì channel "ảo" = Batch * C.
    // Vì vậy ta chỉ cần tăng grid Z lên gấp B lần là kernel cũ vẫn chạy đúng!
    
    dim3 grid_pool1((W/2 + 15)/16, (H/2 + 15)/16, B * 256); 
    maxpool<<<grid_pool1, block>>>(d_conv1_out, d_pool1_out, H, W, 256); // C passed is just purely logic

    forward_conv_layer(d_pool1_out, d_w_conv2, d_conv2_out, d_col_buffer, B, H/2, W/2, 256, 128);
    relu<<<(B*128*(H/2)*(W/2)+255)/256, 256>>>(d_conv2_out, B*128*(H/2)*(W/2));

    dim3 grid_pool2((W/4 + 15)/16, (H/4 + 15)/16, B * 128);
    maxpool<<<grid_pool2, block>>>(d_conv2_out, d_pool2_out, H/2, W/2, 128);

    // Decoder
    forward_conv_layer(d_pool2_out, d_w_dec1, d_dec1_out, d_col_buffer, B, H/4, W/4, 128, 128);
    relu<<<(B*128*(H/4)*(W/4)+255)/256, 256>>>(d_dec1_out, B*128*(H/4)*(W/4));

    dim3 grid_up1((W/2 + 15)/16, (H/2 + 15)/16, B * 128);
    upsample<<<grid_up1, block>>>(d_dec1_out, d_ups1_out, H/4, W/4, 128);

    forward_conv_layer(d_ups1_out, d_w_dec2, d_dec2_out, d_col_buffer, B, H/2, W/2, 128, 256);
    relu<<<(B*256*(H/2)*(W/2)+255)/256, 256>>>(d_dec2_out, B*256*(H/2)*(W/2));

    dim3 grid_up2((W + 15)/16, (H + 15)/16, B * 256);
    upsample<<<grid_up2, block>>>(d_dec2_out, d_ups2_out, H/2, W/2, 256);

    forward_conv_layer(d_ups2_out, d_w_final, d_output, d_col_buffer, B, H, W, 256, 3);

    // Copy Output
    gpu_memcpy_d2h(host_output, d_output, B * C * H * W * sizeof(float));
    cudaDeviceSynchronize();
}