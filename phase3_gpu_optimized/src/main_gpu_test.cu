#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

#include "autoencoder.h"

using namespace std;

// Load one CIFAR-10 image
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
    cout << "===== Phase 3: GPU Optimized Autoencoder Test (DETAILED TIMING) =====\n";

    int H = 32;
    int W = 32;
    int C = 3;

    // Load image from CIFAR-10
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

    // Output buffer
    vector<float> output(C * H * W, 0.0f);

    // Create autoencoder
    cout << "\n[INFO] Creating autoencoder...\n";
    Autoencoder model(H, W, C);

    // Run detailed timing on 1 image
    cout << "\n========================================\n";
    cout << "DETAILED LAYER TIMING (1 image)\n";
    cout << "========================================\n";
    
    model.forward(input.data(), output.data(), true);  // verbose=true

    // Print sample results
    cout << "\n===== OUTPUT SAMPLE VALUES (first 10 pixels) =====\n";
    for (int i = 0; i < 10; i++)
        cout << output[i] << " ";
    cout << "\n";

    cout << "\n===== DONE =====\n";
    return 0;
}
