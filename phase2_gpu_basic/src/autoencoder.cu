#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "../include/autoencoder.h"
#include "../include/utils/gpu_memory.h"
#include "../include/utils/cuda_utils.h"

using namespace std;

// ===== KERNEL DECLARATIONS =====
extern "C" __global__
void conv2d(const float* input,
            const float* kernel,
            float* output,
            int H, int W,
            int in_channels, int out_channels);

extern "C" __global__
void relu(float* x, int size);

extern "C" __global__
void maxpool(const float* input,
             float* output,
             int H, int W, int C);

extern "C" __global__
void upsample(const float* input,
              float* output,
              int H, int W, int C);

// ==============================
//       CONSTRUCTOR
// ==============================
Autoencoder::Autoencoder(int H_, int W_, int C_)
    : H(H_), W(W_), C(C_),
      d_input(nullptr), d_conv1_out(nullptr), d_pool1_out(nullptr),
      d_conv2_out(nullptr), d_pool2_out(nullptr),
      d_ups1_out(nullptr), d_decoder1_out(nullptr),
      d_ups2_out(nullptr), d_output(nullptr),
      d_w_conv1(nullptr), d_w_conv2(nullptr),
      d_w_dec1(nullptr), d_w_dec2(nullptr)
{
    cout << "[INFO] Initializing GPU Autoencoder..." << endl;

    // ===== ALLOCATE WEIGHTS (Host side) =====
    w_conv1.resize(8 * C * 3 * 3, 0.01f); // 3→8
    w_conv2.resize(4 * 8 * 3 * 3, 0.01f); // 8→4
    w_dec1.resize(8 * 4 * 3 * 3, 0.01f); // 4→8
    w_dec2.resize(3 * 8 * 3 * 3, 0.01f); // 8→3

    // ===== ALLOCATE WEIGHTS ON GPU =====
    d_w_conv1 = (float*) gpu_malloc(w_conv1.size() * sizeof(float));
    d_w_conv2 = (float*) gpu_malloc(w_conv2.size() * sizeof(float));
    d_w_dec1  = (float*) gpu_malloc(w_dec1.size() * sizeof(float));
    d_w_dec2  = (float*) gpu_malloc(w_dec2.size() * sizeof(float));

    gpu_memcpy_h2d(d_w_conv1, w_conv1.data(), w_conv1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_conv2, w_conv2.data(), w_conv2.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_dec1,  w_dec1.data(),  w_dec1.size()  * sizeof(float));
    gpu_memcpy_h2d(d_w_dec2,  w_dec2.data(),  w_dec2.size()  * sizeof(float));

    // ===== ALLOCATE INTERMEDIATE FEATURE MAPS ON GPU =====
    d_input        = (float*)gpu_malloc(C * H * W * sizeof(float));
    d_conv1_out    = (float*)gpu_malloc(8 * H * W * sizeof(float));
    d_pool1_out    = (float*)gpu_malloc(8 * (H/2) * (W/2) * sizeof(float));
    d_conv2_out    = (float*)gpu_malloc(4 * (H/2) * (W/2) * sizeof(float));
    d_pool2_out    = (float*)gpu_malloc(4 * (H/4) * (W/4) * sizeof(float));
    d_ups1_out     = (float*)gpu_malloc(4 * (H/2) * (W/2) * sizeof(float));
    d_decoder1_out = (float*)gpu_malloc(8 * (H/2) * (W/2) * sizeof(float));
    d_ups2_out     = (float*)gpu_malloc(8 * H * W * sizeof(float));
    d_output       = (float*)gpu_malloc(C * H * W * sizeof(float));

    cout << "[INFO] GPU Autoencoder initialization done." << endl;
}

// ==============================
//        FORWARD PASS
// ==============================
void Autoencoder::forward(const float* host_input, float* host_output)
{
    cout << "[INFO] Running forward pass..." << endl;

    int H2 = H / 2;
    int W2 = W / 2;
    int H4 = H / 4;
    int W4 = W / 4;

    // ===== COPY INPUT =====
    gpu_memcpy_h2d(d_input, host_input, C * H * W * sizeof(float));

    dim3 block(16, 16);

    // ===== CONV 1 =====
    dim3 grid1((W + 15) / 16, (H + 15) / 16, 8);
    conv2d<<<grid1, block>>>(d_input, d_w_conv1, d_conv1_out, H, W, C, 8);
    checkCuda(cudaGetLastError(), "conv1 launch");

    // ===== ReLU 1 =====
    relu<<<(8*H*W + 255)/256, 256>>>(d_conv1_out, 8 * H * W);

    // ===== MaxPool 1 =====
    dim3 grid_pool1((W2 + 15) / 16, (H2 + 15) / 16, 8);
    maxpool<<<grid_pool1, block>>>(d_conv1_out, d_pool1_out, H, W, 8);

    // ===== CONV 2 =====
    dim3 grid2((W2 + 15) / 16, (H2 + 15) / 16, 4);
    conv2d<<<grid2, block>>>(d_pool1_out, d_w_conv2, d_conv2_out, H2, W2, 8, 4);
    checkCuda(cudaGetLastError(), "conv2 launch");

    // ===== ReLU 2 =====
    relu<<<(4*H2*W2 + 255)/256, 256>>>(d_conv2_out, 4 * H2 * W2);

    // ===== MaxPool 2 =====
    dim3 grid_pool2((W4 + 15) / 16, (H4 + 15) / 16, 4);
    maxpool<<<grid_pool2, block>>>(d_conv2_out, d_pool2_out, H2, W2, 4);

    // ===== Upsample 1 =====
    dim3 grid_up1((W2 + 15) / 16, (H2 + 15) / 16, 4);
    upsample<<<grid_up1, block>>>(d_pool2_out, d_ups1_out, H4, W4, 4);

    // ===== DECODER CONV 1 (4→8) =====
    conv2d<<<grid2, block>>>(d_ups1_out, d_w_dec1, d_decoder1_out, H2, W2, 4, 8);

    // ===== Upsample 2 =====
    dim3 grid_up2((W + 15) / 16, (H + 15) / 16, 8);
    upsample<<<grid_up2, block>>>(d_decoder1_out, d_ups2_out, H2, W2, 8);

    // ===== FINAL CONV (8→3) =====
    dim3 grid_final((W + 15) / 16, (H + 15) / 16, C);
    conv2d<<<grid_final, block>>>(d_ups2_out, d_w_dec2, d_output, H, W, 8, C);
    checkCuda(cudaGetLastError(), "final conv");

    // ===== COPY BACK TO CPU =====
    gpu_memcpy_d2h(host_output, d_output, C * H * W * sizeof(float));
    cudaDeviceSynchronize();

    cout << "[INFO] Forward pass completed.\n";
}

// ==============================
//         DESTRUCTOR
// ==============================
Autoencoder::~Autoencoder()
{
    cout << "[INFO] Freeing GPU memory..." << endl;

    gpu_free(d_input);
    gpu_free(d_conv1_out);
    gpu_free(d_pool1_out);
    gpu_free(d_conv2_out);
    gpu_free(d_pool2_out);
    gpu_free(d_ups1_out);
    gpu_free(d_decoder1_out);
    gpu_free(d_ups2_out);
    gpu_free(d_output);

    gpu_free(d_w_conv1);
    gpu_free(d_w_conv2);
    gpu_free(d_w_dec1);
    gpu_free(d_w_dec2);

    cout << "[INFO] GPU memory freed." << endl;
}
