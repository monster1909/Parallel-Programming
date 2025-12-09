#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <vector>
using namespace std;

class Autoencoder {
public:
    // Constructor với tham số mặc định
    Autoencoder(int H = 32, int W = 32, int C = 3);
    ~Autoencoder();

    void init_weights();
    void forward(const float* host_input, float* host_output, bool verbose = true);

    // Helper static
    static void device_synchronize();

public: 
    int H, W, C;

    // --- GPU Memory Pointers (Feature Maps) ---
    float *d_input;
    
    // Encoder buffers
    float *d_conv1_out;
    float *d_pool1_out;
    float *d_conv2_out;
    float *d_pool2_out; // Latent space
    
    // Decoder buffers
    float *d_dec1_out;
    float *d_ups1_out;
    float *d_dec2_out;
    float *d_ups2_out;
    
    // Final output
    float *d_output;

    // Optimization Buffer (Phase 3 Im2Col)
    float *d_col_buffer;

    // --- Weights (Host Side) ---
    vector<float> w_conv1;
    vector<float> w_conv2;
    vector<float> w_dec1;
    vector<float> w_dec2;
    vector<float> w_final; 

    // --- Weights (Device Side) ---
    float *d_w_conv1;
    float *d_w_conv2;
    float *d_w_dec1;
    float *d_w_dec2;
    float *d_w_final;
};

#endif