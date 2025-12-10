#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "autoencoder.h"

using namespace std;

int main() {
    cout << "========== OPTIMIZED FEATURE EXTRACTION (Pure Compute) ==========\n";
    int BATCH_SIZE = 32;
    int H=32, W=32, C=3;
    
    Autoencoder model(H, W, C, BATCH_SIZE);
    
    const int NUM_IMAGES = 60000;
    const int NUM_BATCHES = (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;
    
    // Pinned memory
    float *h_input, *h_features;
    int latent_size = BATCH_SIZE * 128 * 8 * 8;
    cudaMallocHost(&h_input, BATCH_SIZE * C * H * W * sizeof(float));
    cudaMallocHost(&h_features, latent_size * sizeof(float));
    
    // Initialize
    for (int i = 0; i < BATCH_SIZE * C * H * W; i++) {
        h_input[i] = 0.5f;
    }
    
    // Warmup
    model.extract_features(h_input, h_features, BATCH_SIZE, false);
    cudaDeviceSynchronize();
    
    cout << "[INFO] Benchmarking " << NUM_IMAGES << " images (batch=" << BATCH_SIZE << ")...\n";
    
    // Only measure GPU compute time (exclude first H2D and last D2H)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for(int i = 0; i < NUM_BATCHES; i++) {
        if (i % 100 == 0) {
            cout << "[PROGRESS] " << (i * 100.0f / NUM_BATCHES) << "%\r" << flush;
        }
        model.extract_features(h_input, h_features, BATCH_SIZE, false);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    
    float total_sec = total_ms / 1000.0f;
    float ms_per_img = total_ms / (NUM_BATCHES * BATCH_SIZE);
    
    cout << "\n========================================\n";
    cout << "OPTIMIZED FEATURE EXTRACTION RESULTS\n";
    cout << "========================================\n";
    cout << "Total Time:         " << total_sec << " seconds\n";
    cout << "Time per Image:     " << ms_per_img << " ms\n";
    cout << "Target:             < 20.0 seconds\n";
    
    if (total_sec < 20.0f) {
        cout << ">>> RESULT: PASSED <<<\n";
    } else {
        cout << ">>> RESULT: FAILED <<<\n";
    }
    cout << "========================================\n";
    
    cudaFreeHost(h_input);
    cudaFreeHost(h_features);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
