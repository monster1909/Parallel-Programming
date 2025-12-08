#ifndef AUTOENCODER_H
#define AUTOENCODER_H
#include <vector>

class Autoencoder {
public:
    // Thêm tham số max_batch_size (mặc định 64)
    Autoencoder(int H = 32, int W = 32, int C = 3, int max_batch_size = 64);
    ~Autoencoder();

    // Hàm forward giờ nhận thêm tham số batch_size thực tế
    void forward(const float* host_input, float* host_output, int batch_size = 64, bool verbose = true);
    static void device_synchronize();

public: 
    int H, W, C, MAX_B; // Lưu max batch size

    // Pointers (Không đổi tên, nhưng kích thước cấp phát sẽ lớn hơn)
    float *d_input;
    float *d_conv1_out, *d_pool1_out;
    float *d_conv2_out, *d_pool2_out;
    float *d_dec1_out, *d_ups1_out;
    float *d_dec2_out, *d_ups2_out;
    float *d_output;
    float *d_col_buffer;

    // Weights (Giữ nguyên)
    std::vector<float> w_conv1, w_conv2, w_dec1, w_dec2, w_final;
    float *d_w_conv1, *d_w_conv2, *d_w_dec1, *d_w_dec2, *d_w_final;
};
#endif