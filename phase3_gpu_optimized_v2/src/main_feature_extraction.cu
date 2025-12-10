#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "autoencoder.h"

using namespace std;

int main() {
    cout << "========== PHASE 3 V2: FEATURE EXTRACTION BENCHMARK ==========\n";
    int BATCH_SIZE = 32;
    int H=32, W=32, C=3;
    
    Autoencoder model(H, W, C, BATCH_SIZE);
    
    // Dummy input
    vector<float> batch_input(BATCH_SIZE * C * H * W, 0.5f);
    
    // Output: latent features (batch_size × 128 × 8 × 8)
    int latent_size = BATCH_SIZE * 128 * 8 * 8;
    vector<float> features(latent_size);
    
    // Warmup
    model.extract_features(batch_input.data(), features.data(), BATCH_SIZE, false);
    
    const int NUM_IMAGES = 60000;
    const int NUM_BATCHES = (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;
    const int ACTUAL_IMAGES = NUM_BATCHES * BATCH_SIZE;
    
    cout << "[INFO] Benchmarking feature extraction for " << NUM_IMAGES << " images...\n";
    cout << "[INFO] Using batch size: " << BATCH_SIZE << "\n";
    cout << "[INFO] Number of batches: " << NUM_BATCHES << " (processing " << ACTUAL_IMAGES << " images)\n";
    cout << "[INFO] Running feature extraction (encoder only)...\n";
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for(int i=0; i<NUM_BATCHES; i++) {
        // Show progress every 100 batches
        if (i % 100 == 0) {
            float progress = (i * 100.0f) / NUM_BATCHES;
            int images_processed = i * BATCH_SIZE;
            cout << "[PROGRESS] Batch " << i << "/" << NUM_BATCHES 
                 << " (" << progress << "%) - " << images_processed << " images    \r" << flush;
        }
        model.extract_features(batch_input.data(), features.data(), BATCH_SIZE, false);
    }
    cout << "[PROGRESS] Batch " << NUM_BATCHES << "/" << NUM_BATCHES 
         << " (100.0%) - " << ACTUAL_IMAGES << " images    \n";
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    
    float avg_time_per_img = total_ms / ACTUAL_IMAGES;
    float total_time_seconds = total_ms / 1000.0f;
    float projected_60k_time = (avg_time_per_img * NUM_IMAGES) / 1000.0f;

    cout << "\n========================================\n";
    cout << "FEATURE EXTRACTION RESULTS (60,000 images)\n";
    cout << "========================================\n";
    cout << "Total Time (" << ACTUAL_IMAGES << " imgs): " << total_ms << " ms\n";
    cout << "Total Time:           " << total_time_seconds << " seconds\n";
    cout << "Projected 60k Time:   " << projected_60k_time << " seconds\n";
    cout << "Avg Time per Image:   " << avg_time_per_img << " ms\n";
    cout << "Target Requirement:   < 20.0 seconds\n";
    
    if (projected_60k_time < 20.0f) {
        cout << ">>> RESULT: PASSED (Fast enough) <<<\n";
    } else {
        cout << ">>> RESULT: FAILED (Too slow) <<<\n";
    }
    cout << "========================================\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
