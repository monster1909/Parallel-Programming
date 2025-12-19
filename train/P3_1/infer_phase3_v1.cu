#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cstring>
#include "../../phase2_gpu_basic/Include/utils/gpu_memory.h"
#include "../../phase2_gpu_basic/Include/utils/cuda_utils.h"
#include "../include/data_loader.h"
#include "../include/mse_loss.h"
using namespace std;
extern "C" __global__ void conv2d(const float *input, const float *kernel, float *output,
                                  int H, int W, int in_channels, int out_channels);
extern "C" __global__ void relu(float *x, int size);
extern "C" __global__ void maxpool(const float *input, float *output, int H, int W, int C);
extern "C" __global__ void upsample(const float *input, float *output, int H, int W, int C);
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <weight_file.bin>" << endl;
        cerr << "Example: " << argv[0] << " weights/phase2_epoch_0.bin" << endl;
        return 1;
    }
    const char* weight_file = argv[1];
    cout << "===== Phase 3.1 Inference =====" << endl;
    cout << "[INFO] Loading weights from: " << weight_file << endl;
    const int H = 32, W = 32, C = 3;
    const int BATCH_SIZE = 32;  
    const int DISPLAY_COUNT = 10;  
    vector<float> w_conv1(256 * C * 3 * 3);
    vector<float> w_conv2(128 * 256 * 3 * 3);
    vector<float> w_dec1(128 * 128 * 3 * 3);
    vector<float> w_dec2(256 * 128 * 3 * 3);
    vector<float> w_final(C * 256 * 3 * 3);
    ifstream file(weight_file, ios::binary);
    if (!file.is_open()) {
        cerr << "[ERROR] Could not open weight file: " << weight_file << endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(w_conv1.data()), w_conv1.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(w_conv2.data()), w_conv2.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(w_dec1.data()), w_dec1.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(w_dec2.data()), w_dec2.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(w_final.data()), w_final.size() * sizeof(float));
    file.close();
    cout << "[INFO] Loaded weights successfully" << endl;
    float *d_w_conv1 = (float*)gpu_malloc(w_conv1.size() * sizeof(float));
    float *d_w_conv2 = (float*)gpu_malloc(w_conv2.size() * sizeof(float));
    float *d_w_dec1 = (float*)gpu_malloc(w_dec1.size() * sizeof(float));
    float *d_w_dec2 = (float*)gpu_malloc(w_dec2.size() * sizeof(float));
    float *d_w_final = (float*)gpu_malloc(w_final.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_conv1, w_conv1.data(), w_conv1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_conv2, w_conv2.data(), w_conv2.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_dec1, w_dec1.data(), w_dec1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_dec2, w_dec2.data(), w_dec2.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_final, w_final.data(), w_final.size() * sizeof(float));
    DataLoader loader("../../Data/cifar-10-batches-bin/", BATCH_SIZE, false, true);  
    float *d_input = (float*)gpu_malloc(C * H * W * sizeof(float));
    float *d_conv1_out = (float*)gpu_malloc(256 * H * W * sizeof(float));
    float *d_pool1_out = (float*)gpu_malloc(256 * (H/2) * (W/2) * sizeof(float));
    float *d_conv2_out = (float*)gpu_malloc(128 * (H/2) * (W/2) * sizeof(float));
    float *d_pool2_out = (float*)gpu_malloc(128 * (H/4) * (W/4) * sizeof(float));
    float *d_dec1_out = (float*)gpu_malloc(128 * (H/4) * (W/4) * sizeof(float));
    float *d_ups1_out = (float*)gpu_malloc(128 * (H/2) * (W/2) * sizeof(float));
    float *d_dec2_out = (float*)gpu_malloc(256 * (H/2) * (W/2) * sizeof(float));
    float *d_ups2_out = (float*)gpu_malloc(256 * H * W * sizeof(float));
    float *d_output = (float*)gpu_malloc(C * H * W * sizeof(float));
    int total_images = loader.get_total_images();
    cout << "[INFO] Running inference on " << total_images << " test images..." << endl;
    cout << "[INFO] Displaying first " << DISPLAY_COUNT << " results..." << endl << endl;
    float total_loss = 0.0f;
    int image_count = 0;
    while (loader.has_next()) {
        float* batch_input = loader.next_batch();
        int current_batch_size = min(BATCH_SIZE, total_images - image_count);
        for (int b = 0; b < current_batch_size; b++) {
            gpu_memcpy_h2d(d_input, batch_input + b * C * H * W, C * H * W * sizeof(float));
            dim3 block(16, 16);
            conv2d<<<dim3((W+15)/16, (H+15)/16, 256), block>>>(d_input, d_w_conv1, d_conv1_out, H, W, C, 256);
            relu<<<(256*H*W+255)/256, 256>>>(d_conv1_out, 256*H*W);
            maxpool<<<dim3((W/2+15)/16, (H/2+15)/16, 256), block>>>(d_conv1_out, d_pool1_out, H, W, 256);
            conv2d<<<dim3((W/2+15)/16, (H/2+15)/16, 128), block>>>(d_pool1_out, d_w_conv2, d_conv2_out, H/2, W/2, 256, 128);
            relu<<<(128*H/2*W/2+255)/256, 256>>>(d_conv2_out, 128*H/2*W/2);
            maxpool<<<dim3((W/4+15)/16, (H/4+15)/16, 128), block>>>(d_conv2_out, d_pool2_out, H/2, W/2, 128);
            conv2d<<<dim3((W/4+15)/16, (H/4+15)/16, 128), block>>>(d_pool2_out, d_w_dec1, d_dec1_out, H/4, W/4, 128, 128);
            upsample<<<dim3((W/2+15)/16, (H/2+15)/16, 128), block>>>(d_dec1_out, d_ups1_out, H/4, W/4, 128);
            conv2d<<<dim3((W/2+15)/16, (H/2+15)/16, 256), block>>>(d_ups1_out, d_w_dec2, d_dec2_out, H/2, W/2, 128, 256);
            upsample<<<dim3((W+15)/16, (H+15)/16, 256), block>>>(d_dec2_out, d_ups2_out, H/2, W/2, 256);
            conv2d<<<dim3((W+15)/16, (H+15)/16, C), block>>>(d_ups2_out, d_w_final, d_output, H, W, 256, C);
            cudaDeviceSynchronize();
            float loss = mse_loss_forward(d_output, d_input, C * H * W);
            total_loss += loss;
            if (image_count < DISPLAY_COUNT) {
                cout << "Image " << (image_count+1) << "/" << total_images 
                     << " - Reconstruction Loss: " << loss << endl;
            }
            image_count++;
        }
        if (image_count % 1000 == 0 && image_count >= DISPLAY_COUNT) {
            cout << "Processed " << image_count << "/" << total_images << " images..." << endl;
        }
    }
    float avg_loss = total_loss / image_count;
    cout << "\n[RESULT] Tested on " << image_count << " images" << endl;
    cout << "[RESULT] Average Reconstruction Loss: " << avg_loss << endl;
    gpu_free(d_w_conv1); gpu_free(d_w_conv2); gpu_free(d_w_dec1);
    gpu_free(d_w_dec2); gpu_free(d_w_final);
    gpu_free(d_input); gpu_free(d_conv1_out); gpu_free(d_pool1_out);
    gpu_free(d_conv2_out); gpu_free(d_pool2_out); gpu_free(d_dec1_out);
    gpu_free(d_ups1_out); gpu_free(d_dec2_out); gpu_free(d_ups2_out);
    gpu_free(d_output);
    cout << "===== Inference Complete =====" << endl;
    return 0;
}
