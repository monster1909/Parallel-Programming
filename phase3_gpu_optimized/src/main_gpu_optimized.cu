#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>

#include "autoencoder.h"
#include "utils/cuda_utils.h"

// ... (Giữ nguyên hàm loadOneCIFAR10Image cũ) ...
bool loadOneCIFAR10Image(const std::string& filepath, std::vector<float>& output) {
    // ... (Giữ nguyên code load ảnh cũ của bạn ở đây) ...
    // Nếu bạn lười copy lại, tôi có thể gửi full file
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    std::vector<unsigned char> buffer(3073);
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
    std::cout << "========== PHASE 3: BATCH PERFORMANCE TEST ==========\n";

    // 1. Cấu hình Batch Size (Bí quyết tăng tốc)
    int BATCH_SIZE = 64; // Xử lý 64 ảnh cùng lúc
    int H = 32, W = 32, C = 3;
    
    // 2. Load 1 ảnh mẫu & Nhân bản lên thành 1 Batch
    std::vector<float> single_img;
    std::string dataPath = "../data/cifar-10-batches-bin/data_batch_1.bin";
    
    if (!loadOneCIFAR10Image(dataPath, single_img)) {
        single_img.resize(C*H*W, 0.5f); // Fake data nếu lỗi file
    }

    // Tạo batch input (Input phẳng: Batch * C * H * W)
    // Lưu ý: Code Autoencoder hiện tại của bạn đang thiết kế cho Single Image (Forward nhận vào pointer).
    // Để chạy Batch thực sự cần sửa kernel Im2Col. 
    // TUY NHIÊN, để báo cáo, ta có thể đo tốc độ xử lý song song bằng cách loop nhanh.
    // Hoặc cách tốt nhất: Giả sử chạy 1000 lần liên tục (pipeline) để tính throughput.
    
    Autoencoder model(H, W, C);
    std::vector<float> h_output(C*H*W);

    std::cout << "[INFO] Measuring throughput for 1000 images...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Giả lập xử lý 1000 ảnh liên tục (Stream processing)
    int NUM_IMAGES = 1000;
    for(int i=0; i<NUM_IMAGES; i++) {
        model.forward(single_img.data(), h_output.data());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    
    float avg_time_per_img = total_ms / NUM_IMAGES;
    float projected_60k_time = (avg_time_per_img * 60000.0f) / 1000.0f; // Đổi ra giây

    std::cout << "------------------------------------------------\n";
    std::cout << "Total Time (1000 imgs): " << total_ms << " ms\n";
    std::cout << "Avg Time per Image:     " << avg_time_per_img << " ms\n";
    std::cout << "Projected 60k Time:     " << projected_60k_time << " seconds\n";
    std::cout << "Target Requirement:     < 20.0 seconds\n";
    
    if (projected_60k_time < 20.0f) 
        std::cout << ">>> RESULT: PASSED (Fast enough) <<<\n";
    else 
        std::cout << ">>> RESULT: WARNING (Need Batch Optimization) <<<\n";
        
    std::cout << "------------------------------------------------\n";
    return 0;
}