#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include "autoencoder.h"
#include "utils/gpu_memory.h"
#include "utils/cuda_utils.h"
#include "utils/gpu_timer.h"
using namespace std;
void im2col_gpu(const float* data_im, float* data_col, 
                int channels, int height, int width, 
                int ksize, int pad, int stride, 
                int h_out, int w_out);

extern "C" __global__ void gemm_tiled(const float* A, const float* B, float* C, 
                                      int M, int N, int K);
extern "C" __global__ void relu(float* x, int size);
extern "C" __global__ void maxpool(const float* input, float* output, int H, int W, int C);
extern "C" __global__ void upsample(const float* input, float* output, int H, int W, int C);
void forward_conv_layer(const float* d_input, const float* d_weights, float* d_output, float* d_col_buffer,
                        int H, int W, int C_in, int C_out) 
{
    int ksize = 3, pad = 1, stride = 1;
    int H_out = H, W_out = W;

    im2col_gpu(d_input, d_col_buffer, C_in, H, W, ksize, pad, stride, H_out, W_out);

    int M = C_out;
    int N = H_out * W_out;
    int K = C_in * ksize * ksize;

    dim3 dimGrid((N + 15)/16, (M + 15)/16);
    dim3 dimBlock(16, 16);
    gemm_tiled<<<dimGrid, dimBlock>>>(d_weights, d_col_buffer, d_output, M, N, K);
}

Autoencoder::Autoencoder(int H_, int W_, int C_)
    : H(H_), W(W_), C(C_)
{
    cout << "[INFO] Initializing Phase 3 Autoencoder (Im2Col + GEMM)..." << endl;

    int C1 = 256;  // Match Phase 1 & 2 specification
    int C2 = 128;

    auto init_vec = [](vector<float>& v, int size) {
        v.resize(size);
        for(int i=0; i<size; i++) v[i] = ((float)rand()/RAND_MAX) * 0.02f - 0.01f;
    };

    init_vec(w_conv1, C1 * C * 3 * 3);
    init_vec(w_conv2, C2 * C1 * 3 * 3);
    init_vec(w_dec1, C2 * C2 * 3 * 3);
    init_vec(w_dec2, C1 * C2 * 3 * 3);
    init_vec(w_final, C * C1 * 3 * 3);

    d_w_conv1 = (float*) gpu_malloc(w_conv1.size() * sizeof(float));
    d_w_conv2 = (float*) gpu_malloc(w_conv2.size() * sizeof(float));
    d_w_dec1  = (float*) gpu_malloc(w_dec1.size() * sizeof(float));
    d_w_dec2  = (float*) gpu_malloc(w_dec2.size() * sizeof(float));
    d_w_final = (float*) gpu_malloc(w_final.size() * sizeof(float));

    gpu_memcpy_h2d(d_w_conv1, w_conv1.data(), w_conv1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_conv2, w_conv2.data(), w_conv2.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_dec1,  w_dec1.data(),  w_dec1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_dec2,  w_dec2.data(),  w_dec2.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_final, w_final.data(), w_final.size() * sizeof(float));

    d_input     = (float*)gpu_malloc(C * H * W * sizeof(float));
    d_conv1_out = (float*)gpu_malloc(C1 * H * W * sizeof(float));
    d_pool1_out = (float*)gpu_malloc(C1 * (H/2) * (W/2) * sizeof(float));
    d_conv2_out = (float*)gpu_malloc(C2 * (H/2) * (W/2) * sizeof(float));
    d_pool2_out = (float*)gpu_malloc(C2 * (H/4) * (W/4) * sizeof(float));
    d_dec1_out  = (float*)gpu_malloc(C2 * (H/4) * (W/4) * sizeof(float)); 
    d_ups1_out  = (float*)gpu_malloc(C2 * (H/2) * (W/2) * sizeof(float));
    d_dec2_out  = (float*)gpu_malloc(C1 * (H/2) * (W/2) * sizeof(float));
    d_ups2_out  = (float*)gpu_malloc(C1 * H * W * sizeof(float));
    d_output    = (float*)gpu_malloc(C * H * W * sizeof(float));

    size_t max_col_size = C1 * 9 * 32 * 32 * sizeof(float);
    d_col_buffer = (float*)gpu_malloc(max_col_size);
    
    cout << "[INFO] GPU Memory Allocated. Ready for Phase 3." << endl;
}

Autoencoder::~Autoencoder() {
    gpu_free(d_input);
    gpu_free(d_conv1_out); gpu_free(d_pool1_out);
    gpu_free(d_conv2_out); gpu_free(d_pool2_out);
    gpu_free(d_dec1_out);  gpu_free(d_ups1_out);
    gpu_free(d_dec2_out);  gpu_free(d_ups2_out);
    gpu_free(d_output);
    gpu_free(d_w_conv1); gpu_free(d_w_conv2);
    gpu_free(d_w_dec1);  gpu_free(d_w_dec2);
    gpu_free(d_w_final);
    gpu_free(d_col_buffer);
}

void Autoencoder::device_synchronize() {
    cudaDeviceSynchronize();
}

void Autoencoder::init_weights() {
}
void Autoencoder::forward(const float* host_input, float* host_output, bool verbose)
{
    if (verbose) {
        cout << "\n===== FORWARD PASS START =====\n";
    }

    int H2 = H / 2, W2 = W / 2;
    int H4 = H / 4, W4 = W / 4;
    int C1 = 256, C2 = 128;  // Match specification
    dim3 block(16, 16);

    GpuTimer timer;
    float t_conv1 = 0, t_relu1 = 0, t_pool1 = 0;
    float t_conv2 = 0, t_relu2 = 0, t_pool2 = 0;
    float t_dec1 = 0, t_relu_dec1 = 0, t_up1 = 0;
    float t_dec2 = 0, t_relu_dec2 = 0, t_up2 = 0, t_final = 0;

    gpu_memcpy_h2d(d_input, host_input, C * H * W * sizeof(float));
    
    // 1. Conv1: Input(3) -> 64 | Size: 32x32
    if (verbose) timer.Start();
    forward_conv_layer(d_input, d_w_conv1, d_conv1_out, d_col_buffer, H, W, C, C1);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_conv1 = timer.Elapsed(); }

    if (verbose) timer.Start();
    relu<<<(C1*H*W + 255)/256, 256>>>(d_conv1_out, C1 * H * W);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_relu1 = timer.Elapsed(); }

    // 2. Pool1: 32x32 -> 16x16
    dim3 grid_pool1((W2 + 15) / 16, (H2 + 15) / 16, C1);
    if (verbose) timer.Start();
    maxpool<<<grid_pool1, block>>>(d_conv1_out, d_pool1_out, H, W, C1);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_pool1 = timer.Elapsed(); }

    // 3. Conv2: 64 -> 32 | Size: 16x16
    if (verbose) timer.Start();
    forward_conv_layer(d_pool1_out, d_w_conv2, d_conv2_out, d_col_buffer, H2, W2, C1, C2);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_conv2 = timer.Elapsed(); }

    if (verbose) timer.Start();
    relu<<<(C2*H2*W2 + 255)/256, 256>>>(d_conv2_out, C2 * H2 * W2);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_relu2 = timer.Elapsed(); }

    // 4. Pool2: 16x16 -> 8x8 (LATENT SPACE)
    dim3 grid_pool2((W4 + 15) / 16, (H4 + 15) / 16, C2);
    if (verbose) timer.Start();
    maxpool<<<grid_pool2, block>>>(d_conv2_out, d_pool2_out, H2, W2, C2);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_pool2 = timer.Elapsed(); }


    // ---------------- DECODER ----------------

    // 5. Decoder Conv1: 32 -> 32 | Size: 8x8
    if (verbose) timer.Start();
    forward_conv_layer(d_pool2_out, d_w_dec1, d_dec1_out, d_col_buffer, H4, W4, C2, C2);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_dec1 = timer.Elapsed(); }

    if (verbose) timer.Start();
    relu<<<(C2*H4*W4 + 255)/256, 256>>>(d_dec1_out, C2 * H4 * W4);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_relu_dec1 = timer.Elapsed(); }

    // 6. Upsample1: 8x8 -> 16x16
    dim3 grid_up1((W2 + 15) / 16, (H2 + 15) / 16, C2);
    if (verbose) timer.Start();
    upsample<<<grid_up1, block>>>(d_dec1_out, d_ups1_out, H4, W4, C2);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_up1 = timer.Elapsed(); }

    // 7. Decoder Conv2: 32 -> 64 | Size: 16x16
    if (verbose) timer.Start();
    forward_conv_layer(d_ups1_out, d_w_dec2, d_dec2_out, d_col_buffer, H2, W2, C2, C1);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_dec2 = timer.Elapsed(); }

    if (verbose) timer.Start();
    relu<<<(C1*H2*W2 + 255)/256, 256>>>(d_dec2_out, C1 * H2 * W2);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_relu_dec2 = timer.Elapsed(); }

    // 8. Upsample2: 16x16 -> 32x32
    dim3 grid_up2((W + 15) / 16, (H + 15) / 16, C1);
    if (verbose) timer.Start();
    upsample<<<grid_up2, block>>>(d_dec2_out, d_ups2_out, H2, W2, C1);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_up2 = timer.Elapsed(); }

    // 9. Final Conv: 64 -> 3 | Size: 32x32
    if (verbose) timer.Start();
    forward_conv_layer(d_ups2_out, d_w_final, d_output, d_col_buffer, H, W, C1, C);
    if (verbose) { cudaDeviceSynchronize(); timer.Stop(); t_final = timer.Elapsed(); }


    gpu_memcpy_d2h(host_output, d_output, C * H * W * sizeof(float));
    if (verbose) {
        cudaDeviceSynchronize();
    }

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