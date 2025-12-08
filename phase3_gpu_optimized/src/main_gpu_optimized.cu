#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>

#include "autoencoder.h"
#include "utils/cuda_utils.h"

using namespace std;
bool loadOneCIFAR10Image(const string& filepath, vector<float>& output) {
    ifstream file(filepath, ios::binary);
    if (!file.is_open()) return false;
    vector<unsigned char> buffer(3073);
    file.read(reinterpret_cast<char*>(buffer.data()), 3073);
    file.close();
    output.resize(3 * 32 * 32);
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 32; h++) {
            for (int w = 0; w < 32; w++) {
                output[(c * 32 + h) * 32 + w] = buffer[1 + (c * 32 * 32) + (h * 32) + w] / 255.0f;
            }
        }
    }
    return true;
}

int main() {
    cout << "========== PHASE 3: BATCH PERFORMANCE TEST ==========\n";

    int H = 32, W = 32, C = 3;
    vector<float> single_img;
    string dataPath = "../data/cifar-10-batches-bin/data_batch_1.bin";
    
    if (!loadOneCIFAR10Image(dataPath, single_img)) {
        single_img.resize(C*H*W, 0.5f);
    }
    
    Autoencoder model(H, W, C);
    vector<float> h_output(C*H*W);

    model.forward(single_img.data(), h_output.data(), false);

    const int NUM_IMAGES = 60000;
    cout << "[INFO] Benchmarking " << NUM_IMAGES << " images...\n";
    cout << "[INFO] Running GPU forward pass (verbose disabled for benchmark)...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i=0; i<NUM_IMAGES; i++) {
        model.forward(single_img.data(), h_output.data(), false);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    
    float avg_time_per_img = total_ms / NUM_IMAGES;
    float total_time_seconds = total_ms / 1000.0f;

    cout << "\n========================================\n";
    cout << "BENCHMARK RESULTS (60,000 images)\n";
    cout << "========================================\n";
    cout << "Total Time:           " << total_ms << " ms\n";
    cout << "Total Time:           " << total_time_seconds << " seconds\n";
    cout << "Avg Time per Image:   " << avg_time_per_img << " ms\n";
    cout << "Target Requirement:   < 20.0 seconds\n";
    
    if (total_time_seconds < 20.0f) {
        cout << ">>> RESULT: PASSED (Fast enough) <<<\n";
    } else {
        cout << ">>> RESULT: FAILED (Too slow) <<<\n";
    }
    cout << "========================================\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}