#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "autoencoder.h"
#include "utils/gpu_memory.h"
#include "utils/cuda_utils.h"
#include "utils/gpu_timer.h"

using namespace std;

void im2col_gpu(const float* data_im, float* data_col, int batch_size, int channels, int height, int width, int ksize, int pad, int stride, int h_out, int w_out);
extern "C" __global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K);
extern "C" __global__ void relu(float* x, int size);
extern "C" __global__ void maxpool(const float* input, float* output, int H, int W, int C);
extern "C" __global__ void upsample(const float* input, float* output, int H, int W, int C);
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
    gemm_tiled<<<dimGrid, dimBlock>>>(d_weights, d_col_buffer, d_output, M, N, K);
}

Autoencoder::Autoencoder(int H_, int W_, int C_, int max_batch_)
    : H(H_), W(W_), C(C_), MAX_B(max_batch_)
{
    cout << "[INFO] Init Batch Autoencoder (Max Batch: " << MAX_B << ")..." << endl;

    int C1 = 64, C2 = 32;
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

    // Nếu batch_size > MAX_B thì báo lỗi hoặc kẹp lại, ở đây giả sử luôn <= MAX_B
    int B = batch_size;
    
    // Timer for each layer
    GpuTimer timer;

    // Store all layer times
    float t_conv1, t_relu1, t_pool1;
    float t_conv2, t_relu2, t_pool2;
    float t_dec1, t_relu_dec1, t_up1;
    float t_dec2, t_relu_dec2, t_up2, t_final;

    // Copy Input Batch (not counted in timing)
    gpu_memcpy_h2d(d_input, host_input, B * C * H * W * sizeof(float));

    dim3 block(16, 16);

    // Kích thước kênh (phải khớp với constructor)
    int C1 = 64;
    int C2 = 32;

    // Encoder
    timer.Start();
    forward_conv_layer(d_input, d_w_conv1, d_conv1_out, d_col_buffer, B, H, W, C, C1);
    cudaDeviceSynchronize();
    timer.Stop();
    t_conv1 = timer.Elapsed();

    timer.Start();
    relu<<<(B*C1*H*W+255)/256, 256>>>(d_conv1_out, B*C1*H*W);
    cudaDeviceSynchronize();
    timer.Stop();
    t_relu1 = timer.Elapsed();

    timer.Start();
    dim3 grid_pool1((W/2 + 15)/16, (H/2 + 15)/16, B * C1); 
    maxpool<<<grid_pool1, block>>>(d_conv1_out, d_pool1_out, H, W, C1);
    cudaDeviceSynchronize();
    timer.Stop();
    t_pool1 = timer.Elapsed();

    timer.Start();
    forward_conv_layer(d_pool1_out, d_w_conv2, d_conv2_out, d_col_buffer, B, H/2, W/2, C1, C2);
    cudaDeviceSynchronize();
    timer.Stop();
    t_conv2 = timer.Elapsed();

    timer.Start();
    relu<<<(B*C2*(H/2)*(W/2)+255)/256, 256>>>(d_conv2_out, B*C2*(H/2)*(W/2));
    cudaDeviceSynchronize();
    timer.Stop();
    t_relu2 = timer.Elapsed();

    timer.Start();
    dim3 grid_pool2((W/4 + 15)/16, (H/4 + 15)/16, B * C2);
    maxpool<<<grid_pool2, block>>>(d_conv2_out, d_pool2_out, H/2, W/2, C2);
    cudaDeviceSynchronize();
    timer.Stop();
    t_pool2 = timer.Elapsed();

    // Decoder
    timer.Start();
    forward_conv_layer(d_pool2_out, d_w_dec1, d_dec1_out, d_col_buffer, B, H/4, W/4, C2, C2);
    cudaDeviceSynchronize();
    timer.Stop();
    t_dec1 = timer.Elapsed();

    timer.Start();
    relu<<<(B*C2*(H/4)*(W/4)+255)/256, 256>>>(d_dec1_out, B*C2*(H/4)*(W/4));
    cudaDeviceSynchronize();
    timer.Stop();
    t_relu_dec1 = timer.Elapsed();

    timer.Start();
    dim3 grid_up1((W/2 + 15)/16, (H/2 + 15)/16, B * C2);
    upsample<<<grid_up1, block>>>(d_dec1_out, d_ups1_out, H/4, W/4, C2);
    cudaDeviceSynchronize();
    timer.Stop();
    t_up1 = timer.Elapsed();

    timer.Start();
    forward_conv_layer(d_ups1_out, d_w_dec2, d_dec2_out, d_col_buffer, B, H/2, W/2, C2, C1);
    cudaDeviceSynchronize();
    timer.Stop();
    t_dec2 = timer.Elapsed();

    timer.Start();
    relu<<<(B*C1*(H/2)*(W/2)+255)/256, 256>>>(d_dec2_out, B*C1*(H/2)*(W/2));
    cudaDeviceSynchronize();
    timer.Stop();
    t_relu_dec2 = timer.Elapsed();

    timer.Start();
    dim3 grid_up2((W + 15)/16, (H + 15)/16, B * C1);
    upsample<<<grid_up2, block>>>(d_dec2_out, d_ups2_out, H/2, W/2, C1);
    cudaDeviceSynchronize();
    timer.Stop();
    t_up2 = timer.Elapsed();

    timer.Start();
    forward_conv_layer(d_ups2_out, d_w_final, d_output, d_col_buffer, B, H, W, C1, 3);
    cudaDeviceSynchronize();
    timer.Stop();
    t_final = timer.Elapsed();

    gpu_memcpy_d2h(host_output, d_output, B * C * H * W * sizeof(float));
    cudaDeviceSynchronize();

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