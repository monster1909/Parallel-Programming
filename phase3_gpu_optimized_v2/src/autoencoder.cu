#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "autoencoder.h"
#include "utils/gpu_memory.h"
#include "utils/cuda_utils.h"
#include "utils/gpu_timer.h"
using namespace std;
extern "C" __global__ void gemm_tiled_relu_optimized(const float*, const float*, float*, int, int, int);
extern "C" __global__ void gemm_tiled_optimized(const float*, const float*, float*, int, int, int);
extern "C" __global__ void maxpool(const float*, float*, int, int, int);
extern "C" __global__ void upsample(const float*, float*, int, int, int);
extern void im2col_gpu(const float*, float*, int, int, int, int, int, int, int, int, int);  
extern "C" __global__ void relu(float* x, int size); 
void forward_conv_layer(const float* d_input, const float* d_weights, float* d_output, float* d_col_buffer,
                        int batch_size, int H, int W, int C_in, int C_out) 
{
    int ksize = 3, pad = 1, stride = 1;
    im2col_gpu(d_input, d_col_buffer, batch_size, C_in, H, W, ksize, pad, stride, H, W);
    int M = C_out;
    int N = batch_size * H * W;
    int K = C_in * ksize * ksize; 
    dim3 dimGrid((N + 15)/16, (M + 15)/16);
    dim3 dimBlock(16, 16);
    gemm_tiled_optimized<<<dimGrid, dimBlock>>>(d_weights, d_col_buffer, d_output, M, N, K);
}
void forward_conv_layer_relu(const float* d_input, const float* d_weights, float* d_output, float* d_col_buffer,
                              int batch_size, int H, int W, int C_in, int C_out) 
{
    int ksize = 3, pad = 1, stride = 1;
    im2col_gpu(d_input, d_col_buffer, batch_size, C_in, H, W, ksize, pad, stride, H, W);
    int M = C_out;
    int N = batch_size * H * W;
    int K = C_in * ksize * ksize; 
    dim3 dimGrid((N + 15)/16, (M + 15)/16);
    dim3 dimBlock(16, 16);
    gemm_tiled_relu_optimized<<<dimGrid, dimBlock>>>(d_weights, d_col_buffer, d_output, M, N, K);
}
Autoencoder::Autoencoder(int H_, int W_, int C_, int max_batch_)
    : H(H_), W(W_), C(C_), MAX_B(max_batch_)
{
    cout << "[INFO] Init Batch Autoencoder (Max Batch: " << MAX_B << ")..." << endl;
    int C1 = 256;  
    int C2 = 128;
    auto init_vec = [](vector<float>& v, int size) { v.resize(size, 0.01f); };
    init_vec(w_conv1, C1 * C * 9); init_vec(w_conv2, C2 * C1 * 9);
    init_vec(w_dec1, C2 * C2 * 9); init_vec(w_dec2, C1 * C2 * 9); init_vec(w_final, C * C1 * 9);
    d_w_conv1 = (float*) gpu_malloc(w_conv1.size()*4); gpu_memcpy_h2d(d_w_conv1, w_conv1.data(), w_conv1.size()*4);
    d_w_conv2 = (float*) gpu_malloc(w_conv2.size()*4); gpu_memcpy_h2d(d_w_conv2, w_conv2.data(), w_conv2.size()*4);
    d_w_dec1  = (float*) gpu_malloc(w_dec1.size()*4);  gpu_memcpy_h2d(d_w_dec1, w_dec1.data(), w_dec1.size()*4);
    d_w_dec2  = (float*) gpu_malloc(w_dec2.size()*4);  gpu_memcpy_h2d(d_w_dec2, w_dec2.data(), w_dec2.size()*4);
    d_w_final = (float*) gpu_malloc(w_final.size()*4); gpu_memcpy_h2d(d_w_final, w_final.data(), w_final.size()*4);
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
    size_t max_col_size = C1 * 9 * MAX_B * 32 * 32 * sizeof(float);
    d_col_buffer = (float*)gpu_malloc(max_col_size);
}
Autoencoder::~Autoencoder() {
    gpu_free(d_input); gpu_free(d_output); gpu_free(d_col_buffer);
    gpu_free(d_conv1_out); gpu_free(d_pool1_out); gpu_free(d_conv2_out); gpu_free(d_pool2_out);
    gpu_free(d_dec1_out); gpu_free(d_ups1_out); gpu_free(d_dec2_out); gpu_free(d_ups2_out);
    gpu_free(d_w_conv1); gpu_free(d_w_conv2); gpu_free(d_w_dec1); gpu_free(d_w_dec2); gpu_free(d_w_final);
}
void Autoencoder::forward(const float* host_input, float* host_output, int batch_size, bool verbose)
{
    if (verbose) {
        cout << "\n===== FORWARD PASS START (Batch Size: " << batch_size << ") =====\n";
    }
    int B = batch_size;
    GpuTimer timer;
    float t_conv1, t_relu1, t_pool1;
    float t_conv2, t_relu2, t_pool2;
    float t_dec1, t_relu_dec1, t_up1;
    float t_dec2, t_relu_dec2, t_up2, t_final;
    gpu_memcpy_h2d(d_input, host_input, B * C * H * W * sizeof(float));
    dim3 block(16, 16);
    int C1 = 256;  
    int C2 = 128;
    timer.Start();
    forward_conv_layer_relu(d_input, d_w_conv1, d_conv1_out, d_col_buffer, B, H, W, C, C1);  
    if (verbose) cudaDeviceSynchronize();
    timer.Stop();
    t_conv1 = timer.Elapsed();
    t_relu1 = 0.0f;  
    timer.Start();
    dim3 grid_pool1((W/2 + 15)/16, (H/2 + 15)/16, B * C1); 
    maxpool<<<grid_pool1, block>>>(d_conv1_out, d_pool1_out, H, W, C1);
    if (verbose) cudaDeviceSynchronize();
    timer.Stop();
    t_pool1 = timer.Elapsed();
    timer.Start();
    forward_conv_layer_relu(d_pool1_out, d_w_conv2, d_conv2_out, d_col_buffer, B, H/2, W/2, C1, C2);  
    if (verbose) cudaDeviceSynchronize();
    timer.Stop();
    t_conv2 = timer.Elapsed();
    t_relu2 = 0.0f;  
    timer.Start();
    dim3 grid_pool2((W/4 + 15)/16, (H/4 + 15)/16, B * C2);
    maxpool<<<grid_pool2, block>>>(d_conv2_out, d_pool2_out, H/2, W/2, C2);
    if (verbose) cudaDeviceSynchronize();
    timer.Stop();
    t_pool2 = timer.Elapsed();
    timer.Start();
    forward_conv_layer_relu(d_pool2_out, d_w_dec1, d_dec1_out, d_col_buffer, B, H/4, W/4, C2, C2);  
    if (verbose) cudaDeviceSynchronize();
    timer.Stop();
    t_dec1 = timer.Elapsed();
    t_relu_dec1 = 0.0f;  
    timer.Start();
    dim3 grid_up1((W/2 + 15)/16, (H/2 + 15)/16, B * C2);
    upsample<<<grid_up1, block>>>(d_dec1_out, d_ups1_out, H/4, W/4, C2);
    if (verbose) cudaDeviceSynchronize();
    timer.Stop();
    t_up1 = timer.Elapsed();
    timer.Start();
    forward_conv_layer_relu(d_ups1_out, d_w_dec2, d_dec2_out, d_col_buffer, B, H/2, W/2, C2, C1);  
    if (verbose) cudaDeviceSynchronize();
    timer.Stop();
    t_dec2 = timer.Elapsed();
    t_relu_dec2 = 0.0f;  
    timer.Start();
    dim3 grid_up2((W + 15)/16, (H + 15)/16, B * C1);
    upsample<<<grid_up2, block>>>(d_dec2_out, d_ups2_out, H/2, W/2, C1);
    if (verbose) cudaDeviceSynchronize();
    timer.Stop();
    t_up2 = timer.Elapsed();
    timer.Start();
    forward_conv_layer(d_ups2_out, d_w_final, d_output, d_col_buffer, B, H, W, C1, 3);
    if (verbose) cudaDeviceSynchronize();
    timer.Stop();
    t_final = timer.Elapsed();
    gpu_memcpy_d2h(host_output, d_output, B * C * H * W * sizeof(float));
    if (verbose) cudaDeviceSynchronize();
    if (verbose) {
        float total = t_conv1 + t_relu1 + t_pool1 + t_conv2 + t_relu2 + t_pool2 +
                      t_dec1 + t_relu_dec1 + t_up1 + t_dec2 + t_relu_dec2 + t_up2 + t_final;
        auto pct = [&](float t) { return (t / total) * 100.0f; };
        cout << "\n===== TIME BREAKDOWN =====\n";
        cout << "Conv1:        " << t_conv1 << " ms  (" << pct(t_conv1) << "%)\n";
        cout << "ReLU1:        " << t_relu1 << " ms  (" << pct(t_relu1) << "%)\n";
        cout << "MaxPool1:     " << t_pool1 << " ms  (" << pct(t_pool1) << "%)\n";
        cout << "Conv2:        " << t_conv2 << " ms  (" << pct(t_conv2) << "%)\n";
        cout << "ReLU2:        " << t_relu2 << " ms  (" << pct(t_relu2) << "%)\n";
        cout << "MaxPool2:     " << t_pool2 << " ms  (" << pct(t_pool2) << "%)\n";
        cout << "DecodeConv1:  " << t_dec1 << " ms  (" << pct(t_dec1) << "%)\n";
        cout << "ReLU_Dec1:    " << t_relu_dec1 << " ms  (" << pct(t_relu_dec1) << "%)\n";
        cout << "Upsample1:    " << t_up1 << " ms  (" << pct(t_up1) << "%)\n";
        cout << "DecodeConv2:  " << t_dec2 << " ms  (" << pct(t_dec2) << "%)\n";
        cout << "ReLU_Dec2:    " << t_relu_dec2 << " ms  (" << pct(t_relu_dec2) << "%)\n";
        cout << "Upsample2:    " << t_up2 << " ms  (" << pct(t_up2) << "%)\n";
        cout << "FinalConv:    " << t_final << " ms  (" << pct(t_final) << "%)\n";
        cout << "----------------------------------\n";
        cout << "TOTAL FORWARD TIME: " << total << " ms\n";
        cout << "==================================\n";
    }
}
void Autoencoder::extract_features(const float* host_input, float* host_features, int batch_size, bool verbose)
{
    if (verbose) {
        cout << "\n===== FEATURE EXTRACTION (Encoder Only) - Batch Size: " << batch_size << " =====\n";
    }
    int B = batch_size;
    GpuTimer timer;
    gpu_memcpy_h2d(d_input, host_input, B * C * H * W * sizeof(float));
    dim3 block(16, 16);
    int C1 = 256;
    int C2 = 128;
    float t_total = 0.0f;
    if (verbose) timer.Start();
    forward_conv_layer_relu(d_input, d_w_conv1, d_conv1_out, d_col_buffer, B, H, W, C, C1);
    if (verbose) cudaDeviceSynchronize();
    if (verbose) {
        timer.Stop();
        t_total += timer.Elapsed();
    }
    if (verbose) timer.Start();
    dim3 grid_pool1((W/2 + 15)/16, (H/2 + 15)/16, B * C1);
    maxpool<<<grid_pool1, block>>>(d_conv1_out, d_pool1_out, H, W, C1);
    if (verbose) cudaDeviceSynchronize();
    if (verbose) {
        timer.Stop();
        t_total += timer.Elapsed();
    }
    if (verbose) timer.Start();
    forward_conv_layer_relu(d_pool1_out, d_w_conv2, d_conv2_out, d_col_buffer, B, H/2, W/2, C1, C2);
    if (verbose) cudaDeviceSynchronize();
    if (verbose) {
        timer.Stop();
        t_total += timer.Elapsed();
    }
    if (verbose) timer.Start();
    dim3 grid_pool2((W/4 + 15)/16, (H/4 + 15)/16, B * C2);
    maxpool<<<grid_pool2, block>>>(d_conv2_out, d_pool2_out, H/2, W/2, C2);
    if (verbose) cudaDeviceSynchronize();
    if (verbose) {
        timer.Stop();
        t_total += timer.Elapsed();
    }
    int latent_size = B * C2 * (H/4) * (W/4);
    gpu_memcpy_d2h(host_features, d_pool2_out, latent_size * sizeof(float));
    if (verbose) cudaDeviceSynchronize();
    if (verbose) {
        cout << "\n[FEATURE EXTRACTION] Total time: " << t_total << " ms\n";
        cout << "[FEATURE EXTRACTION] Latent size: " << C2 << "x" << (H/4) << "x" << (W/4) 
             << " = " << (C2 * (H/4) * (W/4)) << " dims per image\n";
        cout << "==================================\n";
    }
}
