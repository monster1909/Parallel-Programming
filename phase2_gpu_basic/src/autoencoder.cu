#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "../include/autoencoder.h"
#include "../include/utils/gpu_memory.h"
#include "../include/utils/cuda_utils.h"
#include "../include/utils/gpu_timer.h"

using namespace std;

// ===== KERNEL DECLARATIONS =====
extern "C" __global__ void conv2d(const float *input,
                                  const float *kernel,
                                  float *output,
                                  int H, int W,
                                  int in_channels, int out_channels);

extern "C" __global__ void relu(float *x, int size);

extern "C" __global__ void maxpool(const float *input,
                                   float *output,
                                   int H, int W, int C);

extern "C" __global__ void upsample(const float *input,
                                    float *output,
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

    w_conv1.resize(8 * C * 3 * 3, 0.01f);
    w_conv2.resize(4 * 8 * 3 * 3, 0.01f);
    w_dec1.resize(8 * 4 * 3 * 3, 0.01f);
    w_dec2.resize(3 * 8 * 3 * 3, 0.01f);

    d_w_conv1 = (float *)gpu_malloc(w_conv1.size() * sizeof(float));
    d_w_conv2 = (float *)gpu_malloc(w_conv2.size() * sizeof(float));
    d_w_dec1 = (float *)gpu_malloc(w_dec1.size() * sizeof(float));
    d_w_dec2 = (float *)gpu_malloc(w_dec2.size() * sizeof(float));

    gpu_memcpy_h2d(d_w_conv1, w_conv1.data(), w_conv1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_conv2, w_conv2.data(), w_conv2.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_dec1, w_dec1.data(), w_dec1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_dec2, w_dec2.data(), w_dec2.size() * sizeof(float));

    d_input = (float *)gpu_malloc(C * H * W * sizeof(float));
    d_conv1_out = (float *)gpu_malloc(8 * H * W * sizeof(float));
    d_pool1_out = (float *)gpu_malloc(8 * H / 2 * W / 2 * sizeof(float));
    d_conv2_out = (float *)gpu_malloc(4 * H / 2 * W / 2 * sizeof(float));
    d_pool2_out = (float *)gpu_malloc(4 * H / 4 * W / 4 * sizeof(float));
    d_ups1_out = (float *)gpu_malloc(4 * H / 2 * W / 2 * sizeof(float));
    d_decoder1_out = (float *)gpu_malloc(8 * H / 2 * W / 2 * sizeof(float));
    d_ups2_out = (float *)gpu_malloc(8 * H * W * sizeof(float));
    d_output = (float *)gpu_malloc(C * H * W * sizeof(float));

    cout << "[INFO] GPU Autoencoder initialization done.\n";
}

// ==============================
//        FORWARD PASS
// ==============================
void Autoencoder::forward(const float *host_input, float *host_output)
{
    cout << "\n===== FORWARD PASS START =====\n";

    // Show first 10 input pixels
    cout << "[INPUT] First 10 pixels: ";
    for (int i = 0; i < 10; i++)
        cout << host_input[i] << " ";
    cout << "\n\n";

    // Copy input (not counted in timing)
    gpu_memcpy_h2d(d_input, host_input, C * H * W * sizeof(float));

    // Timer for each layer
    GpuTimer timer;

    // Store all layer times
    float t_conv1, t_relu1, t_pool1;
    float t_conv2, t_relu2, t_pool2;
    float t_up1, t_dec1, t_up2, t_final;

    int H2 = H / 2, W2 = W / 2;
    int H4 = H / 4, W4 = W / 4;

    dim3 block(16, 16);

    // ===== CONV 1 =====
    timer.Start();
    conv2d<<<dim3((W + 15) / 16, (H + 15) / 16, 8), block>>>(d_input, d_w_conv1, d_conv1_out, H, W, C, 8);
    cudaDeviceSynchronize();
    timer.Stop();
    t_conv1 = timer.Elapsed();

    // ===== ReLU 1 =====
    timer.Start();
    relu<<<(8 * H * W + 255) / 256, 256>>>(d_conv1_out, 8 * H * W);
    cudaDeviceSynchronize();
    timer.Stop();
    t_relu1 = timer.Elapsed();

    // ===== MaxPool 1 =====
    timer.Start();
    maxpool<<<dim3((W2 + 15) / 16, (H2 + 15) / 16, 8), block>>>(d_conv1_out, d_pool1_out, H, W, 8);
    cudaDeviceSynchronize();
    timer.Stop();
    t_pool1 = timer.Elapsed();

    // ===== CONV 2 =====
    timer.Start();
    conv2d<<<dim3((W2 + 15) / 16, (H2 + 15) / 16, 4), block>>>(d_pool1_out, d_w_conv2, d_conv2_out, H2, W2, 8, 4);
    cudaDeviceSynchronize();
    timer.Stop();
    t_conv2 = timer.Elapsed();

    // ===== ReLU 2 =====
    timer.Start();
    relu<<<(4 * H2 * W2 + 255) / 256, 256>>>(d_conv2_out, 4 * H2 * W2);
    cudaDeviceSynchronize();
    timer.Stop();
    t_relu2 = timer.Elapsed();

    // ===== MaxPool 2 =====
    timer.Start();
    maxpool<<<dim3((W4 + 15) / 16, (H4 + 15) / 16, 4), block>>>(d_conv2_out, d_pool2_out, H2, W2, 4);
    cudaDeviceSynchronize();
    timer.Stop();
    t_pool2 = timer.Elapsed();

    // ===== Upsample 1 =====
    timer.Start();
    upsample<<<dim3((W2 + 15) / 16, (H2 + 15) / 16, 4), block>>>(d_pool2_out, d_ups1_out, H4, W4, 4);
    cudaDeviceSynchronize();
    timer.Stop();
    t_up1 = timer.Elapsed();

    // ===== DECODER CONV 1 =====
    timer.Start();
    conv2d<<<dim3((W2 + 15) / 16, (H2 + 15) / 16, 8), block>>>(d_ups1_out, d_w_dec1, d_decoder1_out, H2, W2, 4, 8);
    cudaDeviceSynchronize();
    timer.Stop();
    t_dec1 = timer.Elapsed();

    // ===== Upsample 2 =====
    timer.Start();
    upsample<<<dim3((W + 15) / 16, (H + 15) / 16, 8), block>>>(d_decoder1_out, d_ups2_out, H2, W2, 8);
    cudaDeviceSynchronize();
    timer.Stop();
    t_up2 = timer.Elapsed();

    // ===== FINAL CONV =====
    timer.Start();
    conv2d<<<dim3((W + 15) / 16, (H + 15) / 16, C), block>>>(d_ups2_out, d_w_dec2, d_output, H, W, 8, C);
    cudaDeviceSynchronize();
    timer.Stop();
    t_final = timer.Elapsed();

    gpu_memcpy_d2h(host_output, d_output, C * H * W * sizeof(float));

    // ===== TOTAL TIME =====
    float total =
        t_conv1 + t_relu1 + t_pool1 +
        t_conv2 + t_relu2 + t_pool2 +
        t_up1 + t_dec1 + t_up2 + t_final;

    auto pct = [&](float t)
    { return (t / total) * 100.0f; };

    cout << "\n===== TIME BREAKDOWN =====\n";
    cout << "Conv1:      " << t_conv1 << " ms  (" << pct(t_conv1) << "%)\n";
    cout << "ReLU1:      " << t_relu1 << " ms  (" << pct(t_relu1) << "%)\n";
    cout << "MaxPool1:   " << t_pool1 << " ms  (" << pct(t_pool1) << "%)\n";
    cout << "Conv2:      " << t_conv2 << " ms  (" << pct(t_conv2) << "%)\n";
    cout << "ReLU2:      " << t_relu2 << " ms  (" << pct(t_relu2) << "%)\n";
    cout << "MaxPool2:   " << t_pool2 << " ms  (" << pct(t_pool2) << "%)\n";
    cout << "Upsample1:  " << t_up1 << " ms  (" << pct(t_up1) << "%)\n";
    cout << "DecodeConv: " << t_dec1 << " ms  (" << pct(t_dec1) << "%)\n";
    cout << "Upsample2:  " << t_up2 << " ms  (" << pct(t_up2) << "%)\n";
    cout << "FinalConv:  " << t_final << " ms  (" << pct(t_final) << "%)\n";

    cout << "----------------------------------\n";
    cout << "TOTAL FORWARD TIME: " << total << " ms\n";
    cout << "==================================\n";
}

// ==============================
//         DESTRUCTOR
// ==============================
Autoencoder::~Autoencoder()
{
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
}
