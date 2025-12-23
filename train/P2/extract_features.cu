#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include "../../phase2_gpu_basic/Include/utils/gpu_memory.h"
#include "../../phase2_gpu_basic/Include/utils/cuda_utils.h"
#include "../include/data_loader.h"
using namespace std;
extern "C" __global__ void conv2d(const float *input, const float *kernel, float *output,
                                  int H, int W, int in_channels, int out_channels);
extern "C" __global__ void relu(float *x, int size);
extern "C" __global__ void maxpool(const float *input, float *output, int H, int W, int C);
int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <weight_file.bin> <output_features.bin>" << endl;
        cerr << "Example: " << argv[0] << " weights/phase2_epoch_0.bin features.bin" << endl;
        return 1;
    }
    const char* weight_file = argv[1];
    const char* output_file = argv[2];
    cout << "===== Phase 2 Feature Extraction =====" << endl;
    cout << "[INFO] Loading weights from: " << weight_file << endl;
    const int H = 32, W = 32, C = 3;
    const int LATENT_H = 8, LATENT_W = 8, LATENT_C = 128;
    const int LATENT_SIZE = LATENT_C * LATENT_H * LATENT_W;  
    const int BATCH_SIZE = 100;  
    vector<float> w_conv1(256 * C * 3 * 3);
    vector<float> w_conv2(128 * 256 * 3 * 3);
    ifstream file(weight_file, ios::binary);
    if (!file.is_open()) {
        cerr << "[ERROR] Could not open weight file: " << weight_file << endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(w_conv1.data()), w_conv1.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(w_conv2.data()), w_conv2.size() * sizeof(float));
    file.close();
    cout << "[INFO] Loaded encoder weights" << endl;
    float *d_w_conv1 = (float*)gpu_malloc(w_conv1.size() * sizeof(float));
    float *d_w_conv2 = (float*)gpu_malloc(w_conv2.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_conv1, w_conv1.data(), w_conv1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_conv2, w_conv2.data(), w_conv2.size() * sizeof(float));
    float *d_input = (float*)gpu_malloc(C * H * W * sizeof(float));
    float *d_conv1_out = (float*)gpu_malloc(256 * H * W * sizeof(float));
    float *d_pool1_out = (float*)gpu_malloc(256 * (H/2) * (W/2) * sizeof(float));
    float *d_conv2_out = (float*)gpu_malloc(128 * (H/2) * (W/2) * sizeof(float));
    float *d_pool2_out = (float*)gpu_malloc(LATENT_SIZE * sizeof(float));  
    vector<float> h_latent(LATENT_SIZE);
    ofstream out(output_file, ios::binary);
    if (!out.is_open()) {
        cerr << "[ERROR] Could not open output file: " << output_file << endl;
        return 1;
    }
    cout << "[INFO] Extracting features from 60k images..." << endl;
    cout << "[INFO] Output format: Label (1 byte) + Features (8192 floats)" << endl;
    int total_processed = 0;
    {
        DataLoader train_loader("../../Data/cifar-10-batches-bin/", BATCH_SIZE);
        cout << "\n[TRAIN] Processing " << train_loader.get_total_images() << " training images..." << endl;
        while (train_loader.has_next()) {
            float* batch_input = train_loader.next_batch();
            // Labels not needed for feature extraction
            int current_batch_size = min(BATCH_SIZE, train_loader.get_total_images() - total_processed);
            for (int b = 0; b < current_batch_size; b++) {
                gpu_memcpy_h2d(d_input, batch_input + b * C * H * W, C * H * W * sizeof(float));
                dim3 block(16, 16);
                conv2d<<<dim3((W+15)/16, (H+15)/16, 256), block>>>(d_input, d_w_conv1, d_conv1_out, H, W, C, 256);
                relu<<<(256*H*W+255)/256, 256>>>(d_conv1_out, 256*H*W);
                maxpool<<<dim3((W/2+15)/16, (H/2+15)/16, 256), block>>>(d_conv1_out, d_pool1_out, H, W, 256);
                conv2d<<<dim3((W/2+15)/16, (H/2+15)/16, 128), block>>>(d_pool1_out, d_w_conv2, d_conv2_out, H/2, W/2, 256, 128);
                relu<<<(128*H/2*W/2+255)/256, 256>>>(d_conv2_out, 128*H/2*W/2);
                maxpool<<<dim3((W/4+15)/16, (H/4+15)/16, 128), block>>>(d_conv2_out, d_pool2_out, H/2, W/2, 128);
                cudaDeviceSynchronize();
                gpu_memcpy_d2h(h_latent.data(), d_pool2_out, LATENT_SIZE * sizeof(float));
                unsigned char label = 0;  // Placeholder - labels not extracted from DataLoader
                out.write(reinterpret_cast<const char*>(&label), 1);
                out.write(reinterpret_cast<const char*>(h_latent.data()), LATENT_SIZE * sizeof(float));
                total_processed++;
                if (total_processed % 1000 == 0) {
                    cout << "  Processed: " << total_processed << " images" << endl;
                }
            }
        }
    }
    {
        DataLoader test_loader("../../Data/cifar-10-batches-bin/", BATCH_SIZE);
        cout << "\n[TEST] Processing " << test_loader.get_total_images() << " test images..." << endl;
        int test_processed = 0;
        while (test_loader.has_next()) {
            float* batch_input = test_loader.next_batch();
            // Labels not needed for feature extraction
            int current_batch_size = min(BATCH_SIZE, test_loader.get_total_images() - test_processed);
            for (int b = 0; b < current_batch_size; b++) {
                gpu_memcpy_h2d(d_input, batch_input + b * C * H * W, C * H * W * sizeof(float));
                dim3 block(16, 16);
                conv2d<<<dim3((W+15)/16, (H+15)/16, 256), block>>>(d_input, d_w_conv1, d_conv1_out, H, W, C, 256);
                relu<<<(256*H*W+255)/256, 256>>>(d_conv1_out, 256*H*W);
                maxpool<<<dim3((W/2+15)/16, (H/2+15)/16, 256), block>>>(d_conv1_out, d_pool1_out, H, W, 256);
                conv2d<<<dim3((W/2+15)/16, (H/2+15)/16, 128), block>>>(d_pool1_out, d_w_conv2, d_conv2_out, H/2, W/2, 256, 128);
                relu<<<(128*H/2*W/2+255)/256, 256>>>(d_conv2_out, 128*H/2*W/2);
                maxpool<<<dim3((W/4+15)/16, (H/4+15)/16, 128), block>>>(d_conv2_out, d_pool2_out, H/2, W/2, 128);
                cudaDeviceSynchronize();
                gpu_memcpy_d2h(h_latent.data(), d_pool2_out, LATENT_SIZE * sizeof(float));
                unsigned char label = 0;  // Placeholder - labels not extracted from DataLoader
                out.write(reinterpret_cast<const char*>(&label), 1);
                out.write(reinterpret_cast<const char*>(h_latent.data()), LATENT_SIZE * sizeof(float));
                total_processed++;
                test_processed++;
                if (total_processed % 1000 == 0) {
                    cout << "  Processed: " << total_processed << " images" << endl;
                }
            }
        }
    }
    out.close();
    cout << "\n[RESULT] Extracted features from " << total_processed << " images" << endl;
    cout << "[INFO] Output saved to: " << output_file << endl;
    cout << "[INFO] File size: " << (total_processed * (1 + LATENT_SIZE * 4)) / (1024*1024) << " MB" << endl;
    cout << "[INFO] Format: Label (1 byte) + Features (8192 floats = 32768 bytes)" << endl;
    gpu_free(d_w_conv1); gpu_free(d_w_conv2);
    gpu_free(d_input); gpu_free(d_conv1_out); gpu_free(d_pool1_out);
    gpu_free(d_conv2_out); gpu_free(d_pool2_out);
    cout << "===== Feature Extraction Complete =====" << endl;
    return 0;
}
