#include <iostream>
#include <vector>
#include "autoencoder.h"
#include "utils/cuda_utils.h"

int main() {
    std::cout << "========== PHASE 3 v2: BATCHED OPTIMIZATION ==========\n";
    int BATCH_SIZE = 64; 
    int H=32, W=32, C=3;
    
    // Init Model
    Autoencoder model(H, W, C, BATCH_SIZE);

    // Tạo Dummy Data cho Batch (64 ảnh)
    std::vector<float> batch_input(BATCH_SIZE * C * H * W, 0.5f);
    std::vector<float> batch_output(BATCH_SIZE * C * H * W);

    // Warmup
    model.forward(batch_input.data(), batch_output.data(), BATCH_SIZE);

    // Benchmark 100 Batches (Tổng cộng 6400 ảnh)
    std::cout << "[INFO] Processing 100 batches (6400 images)...\n";
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i=0; i<100; i++) {
        model.forward(batch_input.data(), batch_output.data(), BATCH_SIZE);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    
    // Tính toán kết quả
    float time_per_6400 = total_ms;
    float time_per_60k = (time_per_6400 / 6400.0f) * 60000.0f / 1000.0f; // Ra giây

    std::cout << "------------------------------------------------\n";
    std::cout << "Total Time (6400 imgs): " << total_ms << " ms\n";
    std::cout << "Projected 60k Time:     " << time_per_60k << " seconds\n";
    std::cout << "Target:                 < 20.0 seconds\n";
    
    if (time_per_60k < 20.0f) std::cout << ">>> RESULT: SUCCESS! <<<\n";
    else std::cout << ">>> RESULT: STILL SLOW <<<\n";
    std::cout << "------------------------------------------------\n";
    return 0;
}