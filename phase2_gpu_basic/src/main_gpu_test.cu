#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

#include "../Include/autoencoder.h"

using namespace std;

// Hàm đọc ảnh từ CIFAR-10 binary file
bool loadCIFAR10Image(const string& filepath, int imageIndex, vector<float>& output, int& label)
{
    ifstream file(filepath, ios::binary);
    if (!file.is_open())
    {
        cerr << "[ERROR] Cannot open file: " << filepath << endl;
        return false;
    }

    // Mỗi ảnh CIFAR-10: 1 byte label + 3072 bytes (32x32x3)
    const int imageSize = 3073; // 1 + 32*32*3
    const int H = 32;
    const int W = 32;
    const int C = 3;

    // Di chuyển đến vị trí ảnh cần đọc
    file.seekg(imageIndex * imageSize, ios::beg);

    // Đọc label
    unsigned char labelByte;
    file.read(reinterpret_cast<char*>(&labelByte), 1);
    label = static_cast<int>(labelByte);

    // Đọc 3072 bytes pixel data
    vector<unsigned char> buffer(C * H * W);
    file.read(reinterpret_cast<char*>(buffer.data()), C * H * W);

    if (!file)
    {
        cerr << "[ERROR] Failed to read image data" << endl;
        return false;
    }

    file.close();

    // CIFAR-10 format: [R channel][G channel][B channel] (1024 bytes each)
    // Chuyển đổi từ [0, 255] sang [0, 1] và sắp xếp theo CHW format
    output.resize(C * H * W);
    for (int c = 0; c < C; c++)
    {
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                int srcIdx = (c * H * W) + (y * W) + x; // CIFAR format
                int dstIdx = (c * H + y) * W + x;       // CHW format
                output[dstIdx] = buffer[srcIdx] / 255.0f;
            }
        }
    }

    return true;
}

int main()
{
    cout << "===== Phase 2: GPU Basic Autoencoder Test (QUICK TEST) =====\n";

    int H = 32;
    int W = 32;
    int C = 3;

    // ===== ĐỌC ẢNH TỪ CIFAR-10 =====
    string dataPath = "../Data/cifar-10-batches-bin/data_batch_1.bin";
    int imageIndex = 0; // Đọc ảnh đầu tiên
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

    // In một số giá trị pixel mẫu từ ảnh đầu vào
    cout << "\n===== INPUT SAMPLE VALUES (first 10 pixels) =====\n";
    for (int i = 0; i < 10; i++)
        cout << input[i] << " ";
    cout << "\n";

    // ===== OUTPUT BUFFER =====
    vector<float> output(C * H * W, 0.0f);

    // ===== CREATE AUTOENCODER AND RUN FORWARD =====
    cout << "\n[INFO] Creating autoencoder...\n";
    Autoencoder model(H, W, C);

    // ===== RUN DETAILED TIMING ON 1 IMAGE =====
    cout << "\n========================================\n";
    cout << "DETAILED LAYER TIMING (1 image)\n";
    cout << "========================================\n";
    
    model.forward(input.data(), output.data(), true);  // verbose=true for detailed timing

    // ===== PRINT SAMPLE RESULTS =====
    cout << "\n===== OUTPUT SAMPLE VALUES (first 10 pixels) =====\n";
    for (int i = 0; i < 10; i++)
        cout << output[i] << " ";
    cout << "\n";

    cout << "\n===== DONE =====\n";
    return 0;
}
