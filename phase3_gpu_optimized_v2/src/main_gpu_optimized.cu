#include <iostream>
#include <vector>
#include "autoencoder.h"
#include "utils/cuda_utils.h"

using namespace std;

int main() {
    cout << "========== PHASE 3 v2: BATCHED OPTIMIZATION ==========\n";
    int BATCH_SIZE = 32; 
    int H=32, W=32, C=3;
    
    Autoencoder model(H, W, C, BATCH_SIZE);
    vector<float> batch_input(BATCH_SIZE * C * H * W, 0.5f);
    vector<float> batch_output(BATCH_SIZE * C * H * W);

    model.forward(batch_input.data(), batch_output.data(), BATCH_SIZE, false);

    const int NUM_IMAGES = 60000;
    const int NUM_BATCHES = (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;
    const int ACTUAL_IMAGES = NUM_BATCHES * BATCH_SIZE;
    
    cout << "[INFO] Benchmarking " << NUM_IMAGES << " images...\n";
    cout << "[INFO] Using batch size: " << BATCH_SIZE << "\n";
    cout << "[INFO] Number of batches: " << NUM_BATCHES << " (processing " << ACTUAL_IMAGES << " images)\n";
    cout << "[INFO] Running GPU forward pass (verbose disabled for benchmark)...\n";
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for(int i=0; i<NUM_BATCHES; i++) {
        model.forward(batch_input.data(), batch_output.data(), BATCH_SIZE, false);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    
    float avg_time_per_img = total_ms / ACTUAL_IMAGES;
    float total_time_seconds = total_ms / 1000.0f;
    float projected_60k_time = (avg_time_per_img * NUM_IMAGES) / 1000.0f;

    cout << "\n========================================\n";
    cout << "BENCHMARK RESULTS (60,000 images)\n";
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