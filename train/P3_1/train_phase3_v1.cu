#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Include utilities từ phase3
#include "../../phase3_gpu_optimized/Include/utils/gpu_memory.h"
#include "../../phase3_gpu_optimized/Include/utils/cuda_utils.h"
#include "../../phase3_gpu_optimized/Include/utils/gpu_timer.h"

// Include backward kernels từ train/
#include "../include/backward_kernels.h"
#include "../include/mse_loss.h"
#include "../include/sgd_optimizer.h"

// Include utilities từ train/
#include "../include/data_loader.h"
#include "../include/logger.h"
#include "../../tqdm/tqdm.h"

using namespace std;

// Forward kernel declarations từ phase3 (Im2Col + GEMM)
void im2col_gpu(const float* data_im, float* data_col, 
                int channels, int height, int width, 
                int ksize, int pad, int stride, 
                int h_out, int w_out);

extern "C" __global__ void gemm_tiled(const float* A, const float* B, float* C, 
                                      int M, int N, int K);
extern "C" __global__ void relu(float* x, int size);
extern "C" __global__ void maxpool(const float* input, float* output, int H, int W, int C);
extern "C" __global__ void upsample(const float* input, float* output, int H, int W, int C);

// Helper function for Im2Col + GEMM convolution
void forward_conv_layer(const float* d_input, const float* d_weights, float* d_output, float* d_col_buffer,
                        int H, int W, int C_in, int C_out) 
{
    int ksize = 3, pad = 1, stride = 1;
    int H_out = H, W_out = W;

    im2col_gpu(d_input, d_col_buffer, C_in, H, W, ksize, pad, stride, H_out, W_out);

    int M = C_out;
    int N = H_out * W_out;
    int K = C_in * ksize * ksize;

    dim3 dimGrid((N + 15)/16, (M + 15)/16);
    dim3 dimBlock(16, 16);
    gemm_tiled<<<dimGrid, dimBlock>>>(d_weights, d_col_buffer, d_output, M, N, K);
}

int main() {
    cout << "===== Phase 3_1 Training (Im2Col + GEMM) =====" << endl;
    
    // Hyperparameters
    const int H = 32, W = 32, C = 3;
    const int BATCH_SIZE = 32;
    const int NUM_EPOCHS = 20;
    const float LEARNING_RATE = 0.0001f;
    
    // Initialize data loader and logger
    DataLoader loader("../../Data/cifar-10-batches-bin/", BATCH_SIZE);
    Logger logger("logs/phase3_v1_training.log");
    
    logger.log_training_start(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);
    
    // Allocate weights (256/128 architecture - same as P2)
    vector<float> w_conv1(256 * C * 3 * 3, 0.01f);
    vector<float> w_conv2(128 * 256 * 3 * 3, 0.01f);
    vector<float> w_dec1(128 * 128 * 3 * 3, 0.01f);
    vector<float> w_dec2(256 * 128 * 3 * 3, 0.01f);
    vector<float> w_final(C * 256 * 3 * 3, 0.01f);
    
    // Copy weights to GPU
    float *d_w_conv1 = (float*)gpu_malloc(w_conv1.size() * sizeof(float));
    float *d_w_conv2 = (float*)gpu_malloc(w_conv2.size() * sizeof(float));
    float *d_w_dec1 = (float*)gpu_malloc(w_dec1.size() * sizeof(float));
    float *d_w_dec2 = (float*)gpu_malloc(w_dec2.size() * sizeof(float));
    float *d_w_final = (float*)gpu_malloc(w_final.size() * sizeof(float));
    
    gpu_memcpy_h2d(d_w_conv1, w_conv1.data(), w_conv1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_conv2, w_conv2.data(), w_conv2.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_dec1, w_dec1.data(), w_dec1.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_dec2, w_dec2.data(), w_dec2.size() * sizeof(float));
    gpu_memcpy_h2d(d_w_final, w_final.data(), w_final.size() * sizeof(float));
    
    // Allocate gradient buffers for weights
    float *d_grad_w_conv1 = (float*)gpu_malloc(w_conv1.size() * sizeof(float));
    float *d_grad_w_conv2 = (float*)gpu_malloc(w_conv2.size() * sizeof(float));
    float *d_grad_w_dec1 = (float*)gpu_malloc(w_dec1.size() * sizeof(float));
    float *d_grad_w_dec2 = (float*)gpu_malloc(w_dec2.size() * sizeof(float));
    float *d_grad_w_final = (float*)gpu_malloc(w_final.size() * sizeof(float));
    
    // Allocate activation buffers
    float *d_input = (float*)gpu_malloc(C * H * W * sizeof(float));
    float *d_conv1_out = (float*)gpu_malloc(256 * H * W * sizeof(float));
    float *d_pool1_out = (float*)gpu_malloc(256 * (H/2) * (W/2) * sizeof(float));
    float *d_conv2_out = (float*)gpu_malloc(128 * (H/2) * (W/2) * sizeof(float));
    float *d_pool2_out = (float*)gpu_malloc(128 * (H/4) * (W/4) * sizeof(float));
    float *d_dec1_out = (float*)gpu_malloc(128 * (H/4) * (W/4) * sizeof(float));
    float *d_ups1_out = (float*)gpu_malloc(128 * (H/2) * (W/2) * sizeof(float));
    float *d_dec2_out = (float*)gpu_malloc(256 * (H/2) * (W/2) * sizeof(float));
    float *d_ups2_out = (float*)gpu_malloc(256 * H * W * sizeof(float));
    float *d_output = (float*)gpu_malloc(C * H * W * sizeof(float));
    
    // Im2Col buffer (needed for Im2Col + GEMM)
    size_t max_col_size = 256 * 9 * 32 * 32 * sizeof(float);
    float *d_col_buffer = (float*)gpu_malloc(max_col_size);
    
    cout << "[INFO] Memory allocated, starting training..." << endl;
    
    // Training loop
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        tqdm progress(loader.get_num_batches(), 75, 40);
        
        while (loader.has_next()) {
            float* batch_input = loader.next_batch();
            
            // Zero gradients
            zero_gradient(d_grad_w_conv1, w_conv1.size());
            zero_gradient(d_grad_w_conv2, w_conv2.size());
            zero_gradient(d_grad_w_dec1, w_dec1.size());
            zero_gradient(d_grad_w_dec2, w_dec2.size());
            zero_gradient(d_grad_w_final, w_final.size());
            
            float batch_loss = 0.0f;
            
            // Process each image in batch
            for (int b = 0; b < BATCH_SIZE; b++) {
                // Copy single image to device
                gpu_memcpy_h2d(d_input, batch_input + b * C * H * W, C * H * W * sizeof(float));
                
                // FORWARD PASS (using phase3 Im2Col + GEMM)
                dim3 block(16, 16);
                
                // Conv1 + ReLU + MaxPool (Im2Col + GEMM)
                forward_conv_layer(d_input, d_w_conv1, d_conv1_out, d_col_buffer, H, W, C, 256);
                relu<<<(256*H*W+255)/256, 256>>>(d_conv1_out, 256*H*W);
                maxpool<<<dim3((W/2+15)/16, (H/2+15)/16, 256), block>>>(d_conv1_out, d_pool1_out, H, W, 256);
                
                // Conv2 + ReLU + MaxPool
                forward_conv_layer(d_pool1_out, d_w_conv2, d_conv2_out, d_col_buffer, H/2, W/2, 256, 128);
                relu<<<(128*H/2*W/2+255)/256, 256>>>(d_conv2_out, 128*H/2*W/2);
                maxpool<<<dim3((W/4+15)/16, (H/4+15)/16, 128), block>>>(d_conv2_out, d_pool2_out, H/2, W/2, 128);
                
                // Decoder: Dec1 + Upsample + Dec2 + Upsample + Final
                forward_conv_layer(d_pool2_out, d_w_dec1, d_dec1_out, d_col_buffer, H/4, W/4, 128, 128);
                upsample<<<dim3((W/2+15)/16, (H/2+15)/16, 128), block>>>(d_dec1_out, d_ups1_out, H/4, W/4, 128);
                forward_conv_layer(d_ups1_out, d_w_dec2, d_dec2_out, d_col_buffer, H/2, W/2, 128, 256);
                upsample<<<dim3((W+15)/16, (H+15)/16, 256), block>>>(d_dec2_out, d_ups2_out, H/2, W/2, 256);
                forward_conv_layer(d_ups2_out, d_w_final, d_output, d_col_buffer, H, W, 256, C);
                
                cudaDeviceSynchronize();
                
                // COMPUTE LOSS
                float loss = mse_loss_forward(d_output, d_input, C * H * W);
                batch_loss += loss;
                
                // BACKWARD PASS (Same as P2 - backward kernels are shared!)
                // Allocate gradient buffers for activations
                float *d_grad_output = (float*)gpu_malloc(C * H * W * sizeof(float));
                float *d_grad_ups2_out = (float*)gpu_malloc(256 * H * W * sizeof(float));
                float *d_grad_dec2_out = (float*)gpu_malloc(256 * (H/2) * (W/2) * sizeof(float));
                float *d_grad_ups1_out = (float*)gpu_malloc(128 * (H/2) * (W/2) * sizeof(float));
                float *d_grad_dec1_out = (float*)gpu_malloc(128 * (H/4) * (W/4) * sizeof(float));
                float *d_grad_pool2_out = (float*)gpu_malloc(128 * (H/4) * (W/4) * sizeof(float));
                float *d_grad_conv2_out = (float*)gpu_malloc(128 * (H/2) * (W/2) * sizeof(float));
                float *d_grad_pool1_out = (float*)gpu_malloc(256 * (H/2) * (W/2) * sizeof(float));
                float *d_grad_conv1_out = (float*)gpu_malloc(256 * H * W * sizeof(float));
                float *d_grad_input = (float*)gpu_malloc(C * H * W * sizeof(float));
                
                // MSE Loss Backward
                mse_loss_backward(d_output, d_input, d_grad_output, C * H * W);
                
                // Backward through all layers (same as P2)
                // ... (copy from P2 backward pass)
                
                // Cleanup gradient buffers
                gpu_free(d_grad_output); gpu_free(d_grad_ups2_out); gpu_free(d_grad_dec2_out);
                gpu_free(d_grad_ups1_out); gpu_free(d_grad_dec1_out); gpu_free(d_grad_pool2_out);
                gpu_free(d_grad_conv2_out); gpu_free(d_grad_pool1_out); gpu_free(d_grad_conv1_out);
                gpu_free(d_grad_input);
            }
            
            // Average loss
            batch_loss /= BATCH_SIZE;
            epoch_loss += batch_loss;
            
            // UPDATE WEIGHTS (SGD)
            sgd_update(d_w_conv1, d_grad_w_conv1, LEARNING_RATE, w_conv1.size());
            sgd_update(d_w_conv2, d_grad_w_conv2, LEARNING_RATE, w_conv2.size());
            sgd_update(d_w_dec1, d_grad_w_dec1, LEARNING_RATE, w_dec1.size());
            sgd_update(d_w_dec2, d_grad_w_dec2, LEARNING_RATE, w_dec2.size());
            sgd_update(d_w_final, d_grad_w_final, LEARNING_RATE, w_final.size());
            
            num_batches++;
            progress.update(num_batches, {{"loss", batch_loss}});
        }
        
        float avg_loss = epoch_loss / num_batches;
        logger.log_epoch(epoch, avg_loss);
        
        loader.reset();
        progress.end({{"avg_loss", avg_loss}});
    }
    
    logger.log_training_end();
    
    // Cleanup
    gpu_free(d_w_conv1); gpu_free(d_w_conv2); gpu_free(d_w_dec1); 
    gpu_free(d_w_dec2); gpu_free(d_w_final);
    gpu_free(d_grad_w_conv1); gpu_free(d_grad_w_conv2); gpu_free(d_grad_w_dec1);
    gpu_free(d_grad_w_dec2); gpu_free(d_grad_w_final);
    gpu_free(d_input); gpu_free(d_conv1_out); gpu_free(d_pool1_out);
    gpu_free(d_conv2_out); gpu_free(d_pool2_out); gpu_free(d_dec1_out);
    gpu_free(d_ups1_out); gpu_free(d_dec2_out); gpu_free(d_ups2_out);
    gpu_free(d_output); gpu_free(d_col_buffer);
    
    cout << "===== Training Complete =====" << endl;
    return 0;
}
