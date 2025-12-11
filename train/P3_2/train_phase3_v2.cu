#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

// Include utilities từ phase3
#include "../../phase3_gpu_optimized_v2/Include/utils/gpu_memory.h"
#include "../../phase3_gpu_optimized_v2/Include/utils/cuda_utils.h"
#include "../../phase3_gpu_optimized_v2/Include/utils/gpu_timer.h"

// Include backward kernels từ train/
#include "../include/backward_kernels.h"
#include "../include/mse_loss.h"
#include "../include/sgd_optimizer.h"

// Include utilities từ train/
#include "../include/data_loader.h"
#include "../include/logger.h"

using namespace std;

// Forward kernel declarations từ phase3 (Im2Col + GEMM)
void im2col_gpu(const float* data_im, float* data_col, 
                int batch_size, int channels, int height, int width, 
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

    im2col_gpu(d_input, d_col_buffer, 1, C_in, H, W, ksize, pad, stride, H_out, W_out);

    int M = C_out;
    int N = H_out * W_out;
    int K = C_in * ksize * ksize;

    dim3 dimGrid((N + 15)/16, (M + 15)/16);
    dim3 dimBlock(16, 16);
    gemm_tiled<<<dimGrid, dimBlock>>>(d_weights, d_col_buffer, d_output, M, N, K);
}

int main() {
    cout << "===== Phase 3_2 Training (Im2Col + GEMM) =====" << endl;
    
    // Hyperparameters
    const int H = 32, W = 32, C = 3;
    const int BATCH_SIZE = 32;
    const int NUM_EPOCHS = 20;
    const float LEARNING_RATE = 0.001f;
    
    // Initialize data loader and logger
    DataLoader loader("../../Data/cifar-10-batches-bin/", BATCH_SIZE);
    Logger logger("logs/phase3_v2_training.log");
    
    logger.log_training_start(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);
    
    // Allocate weights (256/128 architecture - same as P2)
    vector<float> w_conv1(256 * C * 3 * 3, 0.001f);
    vector<float> w_conv2(128 * 256 * 3 * 3, 0.001f);
    vector<float> w_dec1(128 * 128 * 3 * 3, 0.001f);
    vector<float> w_dec2(256 * 128 * 3 * 3, 0.001f);
    vector<float> w_final(C * 256 * 3 * 3, 0.001f);
    
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
        
        cout << "Training Epoch " << epoch << "/" << NUM_EPOCHS << "..." << endl;
        auto epoch_start = chrono::high_resolution_clock::now();
        int total_batches = loader.get_num_batches();
        int progress_interval = total_batches / 20;  // 5% intervals
        if (progress_interval == 0) progress_interval = 1;
        
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
                
                // Backward through Final Conv (256→3)
                dim3 grid_final_bw_w((3+15)/16, (3+15)/16, 256*C);
                dim3 grid_final_bw_i((W+15)/16, (H+15)/16, 256);
                conv2d_backward_weights<<<grid_final_bw_w, block>>>(
                    d_grad_output, d_ups2_out, d_grad_w_final,
                    H, W, 256, H, W, C, 3
                );
                conv2d_backward_input<<<grid_final_bw_i, block>>>(
                    d_grad_output, d_w_final, d_grad_ups2_out,
                    H, W, 256, H, W, C, 3
                );
                cudaDeviceSynchronize();
                
                // Backward through Upsample2 (16×16 → 32×32)
                dim3 grid_ups2_bw((W/2+15)/16, (H/2+15)/16, 256);
                upsample_backward<<<grid_ups2_bw, block>>>(
                    d_grad_ups2_out, d_grad_dec2_out, H/2, W/2, 256
                );
                cudaDeviceSynchronize();
                
                // Backward through Dec2 Conv (128→256)
                dim3 grid_dec2_bw_w((3+15)/16, (3+15)/16, 256*128);
                dim3 grid_dec2_bw_i((W/2+15)/16, (H/2+15)/16, 128);
                conv2d_backward_weights<<<grid_dec2_bw_w, block>>>(
                    d_grad_dec2_out, d_ups1_out, d_grad_w_dec2,
                    H/2, W/2, 128, H/2, W/2, 256, 3
                );
                conv2d_backward_input<<<grid_dec2_bw_i, block>>>(
                    d_grad_dec2_out, d_w_dec2, d_grad_ups1_out,
                    H/2, W/2, 128, H/2, W/2, 256, 3
                );
                cudaDeviceSynchronize();
                
                // Backward through Upsample1 (8×8 → 16×16)
                dim3 grid_ups1_bw((W/4+15)/16, (H/4+15)/16, 128);
                upsample_backward<<<grid_ups1_bw, block>>>(
                    d_grad_ups1_out, d_grad_dec1_out, H/4, W/4, 128
                );
                cudaDeviceSynchronize();
                
                // Backward through Dec1 Conv (128→128)
                dim3 grid_dec1_bw_w((3+15)/16, (3+15)/16, 128*128);
                dim3 grid_dec1_bw_i((W/4+15)/16, (H/4+15)/16, 128);
                conv2d_backward_weights<<<grid_dec1_bw_w, block>>>(
                    d_grad_dec1_out, d_pool2_out, d_grad_w_dec1,
                    H/4, W/4, 128, H/4, W/4, 128, 3
                );
                conv2d_backward_input<<<grid_dec1_bw_i, block>>>(
                    d_grad_dec1_out, d_w_dec1, d_grad_pool2_out,
                    H/4, W/4, 128, H/4, W/4, 128, 3
                );
                cudaDeviceSynchronize();
                
                // Backward through MaxPool2 (simplified - no argmax)
                cudaMemcpy(d_grad_conv2_out, d_grad_pool2_out, 
                          128*(H/2)*(W/2)*sizeof(float), cudaMemcpyDeviceToDevice);
                
                // Backward through ReLU2
                relu_backward<<<(128*H/2*W/2+255)/256, 256>>>(
                    d_grad_conv2_out, d_conv2_out, d_grad_conv2_out, 128*H/2*W/2
                );
                cudaDeviceSynchronize();
                
                // Backward through Conv2 (256→128)
                dim3 grid_conv2_bw_w((3+15)/16, (3+15)/16, 128*256);
                dim3 grid_conv2_bw_i((W/2+15)/16, (H/2+15)/16, 256);
                conv2d_backward_weights<<<grid_conv2_bw_w, block>>>(
                    d_grad_conv2_out, d_pool1_out, d_grad_w_conv2,
                    H/2, W/2, 256, H/2, W/2, 128, 3
                );
                conv2d_backward_input<<<grid_conv2_bw_i, block>>>(
                    d_grad_conv2_out, d_w_conv2, d_grad_pool1_out,
                    H/2, W/2, 256, H/2, W/2, 128, 3
                );
                cudaDeviceSynchronize();
                
                // Backward through MaxPool1 (simplified - no argmax)
                cudaMemcpy(d_grad_conv1_out, d_grad_pool1_out,
                          256*H*W*sizeof(float), cudaMemcpyDeviceToDevice);
                
                // Backward through ReLU1
                relu_backward<<<(256*H*W+255)/256, 256>>>(
                    d_grad_conv1_out, d_conv1_out, d_grad_conv1_out, 256*H*W
                );
                cudaDeviceSynchronize();
                
                // Backward through Conv1 (3→256)
                dim3 grid_conv1_bw_w((3+15)/16, (3+15)/16, 256*C);
                dim3 grid_conv1_bw_i((W+15)/16, (H+15)/16, C);
                conv2d_backward_weights<<<grid_conv1_bw_w, block>>>(
                    d_grad_conv1_out, d_input, d_grad_w_conv1,
                    H, W, C, H, W, 256, 3
                );
                conv2d_backward_input<<<grid_conv1_bw_i, block>>>(
                    d_grad_conv1_out, d_w_conv1, d_grad_input,
                    H, W, C, H, W, 256, 3
                );
                cudaDeviceSynchronize();
                
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
            
            // Progress display every 5%
            if (num_batches % progress_interval == 0) {
                auto now = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::seconds>(now - epoch_start).count();
                int percent = (num_batches * 100) / total_batches;
                cout << "  " << percent << "% - Loss: " << batch_loss << " - Time: " << elapsed << "s" << endl;
            }
        }
        
        auto epoch_end = chrono::high_resolution_clock::now();
        auto epoch_time = chrono::duration_cast<chrono::seconds>(epoch_end - epoch_start).count();
        
        float avg_loss = epoch_loss / num_batches;
        cout << "Epoch " << epoch << " complete - Avg Loss: " << avg_loss << " - Time: " << epoch_time << "s" << endl << endl;
        logger.log_epoch(epoch, avg_loss);
        
        loader.reset();
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
