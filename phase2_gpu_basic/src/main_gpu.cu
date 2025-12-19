#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include "../Include/autoencoder.h"
using namespace std;
bool loadCIFAR10Image(const string& filepath, int imageIndex, vector<float>& output, int& label)
{
    ifstream file(filepath, ios::binary);
    if (!file.is_open())
    {
        cerr << "[ERROR] Cannot open file: " << filepath << endl;
        return false;
    }
    const int imageSize = 3073; 
    const int H = 32;
    const int W = 32;
    const int C = 3;
    file.seekg(imageIndex * imageSize, ios::beg);
    unsigned char labelByte;
    file.read(reinterpret_cast<char*>(&labelByte), 1);
    label = static_cast<int>(labelByte);
    vector<unsigned char> buffer(C * H * W);
    file.read(reinterpret_cast<char*>(buffer.data()), C * H * W);
    if (!file)
    {
        cerr << "[ERROR] Failed to read image data" << endl;
        return false;
    }
    file.close();
    output.resize(C * H * W);
    for (int c = 0; c < C; c++)
    {
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                int srcIdx = (c * H * W) + (y * W) + x; 
                int dstIdx = (c * H + y) * W + x;       
                output[dstIdx] = buffer[srcIdx] / 255.0f;
            }
        }
    }
    return true;
}
int main()
{
    cout << "===== Phase 2: GPU Basic Autoencoder Test =====\n";
    int H = 32;
    int W = 32;
    int C = 3;
    string dataPath = "../Data/cifar-10-batches-bin/data_batch_1.bin";
    int imageIndex = 0; 
    int label;
    vector<float> input;
    cout << "[INFO] Loading image from CIFAR-10 dataset...\n";
    cout << "[INFO] File: " << dataPath << "\n";
    cout << "[INFO] Image index: " << imageIndex << "\n";
    if (!loadCIFAR10Image(dataPath, imageIndex, input, label))
    {
        cerr << "[ERROR] Failed to load CIFAR-10 image\n";
        return -1;
    }
    cout << "[INFO] Image loaded successfully!\n";
    cout << "[INFO] Image label: " << label << "\n";
    cout << "[INFO] Image size: " << C << "x" << H << "x" << W << "\n";
    cout << "\n===== INPUT SAMPLE VALUES (first 10 pixels) =====\n";
    for (int i = 0; i < 10; i++)
        cout << input[i] << " ";
    cout << "\n";
    vector<float> output(C * H * W, 0.0f);
    Autoencoder model(H, W, C);
    model.forward(input.data(), output.data(), false);
    const int NUM_IMAGES = 60000;
    cout << "\n[INFO] Benchmarking " << NUM_IMAGES << " images...\n";
    cout << "[INFO] Running GPU forward pass (verbose disabled for benchmark)...\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < NUM_IMAGES; i++) {
        if (i % 5000 == 0) {
            float progress = (i * 100.0f) / NUM_IMAGES;
            cout << "[PROGRESS] " << i << "/" << NUM_IMAGES 
                 << " (" << progress << "%)    \r" << flush;
        }
        model.forward(input.data(), output.data(), false);
    }
    cout << "[PROGRESS] " << NUM_IMAGES << "/" << NUM_IMAGES << " (100.0%)    \n";
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
    cout << "Target Requirement:    < 20.0 seconds\n";
    if (total_time_seconds < 20.0f) {
        cout << ">>> RESULT: PASSED (Fast enough) <<<\n";
    } else {
        cout << ">>> RESULT: FAILED (Too slow) <<<\n";
    }
    cout << "========================================\n";
    cout << "\n===== OUTPUT SAMPLE VALUES (first 10 pixels from last image) =====\n";
    for (int i = 0; i < 10; i++)
        cout << output[i] << " ";
    cout << "\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cout << "\n===== DONE =====\n";
    return 0;
}
