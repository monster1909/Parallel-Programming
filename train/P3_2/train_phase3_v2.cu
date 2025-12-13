#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <cmath>

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

// Helper function for Im2Col + GEMM convolution (single image)
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

// Optimized batch convolution: process entire batch at once
void forward_conv_layer_batch(const float* d_input, const float* d_weights, float* d_output, float* d_col_buffer,
                              int batch_size, int H, int W, int C_in, int C_out) 
{
    int ksize = 3, pad = 1, stride = 1;
    int H_out = H, W_out = W;

    // im2col for entire batch
    im2col_gpu(d_input, d_col_buffer, batch_size, C_in, H, W, ksize, pad, stride, H_out, W_out);

    int M = C_out;
    int N = batch_size * H_out * W_out;  // Batch dimension included
    int K = C_in * ksize * ksize;

    dim3 dimGrid((N + 15)/16, (M + 15)/16);
    dim3 dimBlock(16, 16);
    gemm_tiled<<<dimGrid, dimBlock>>>(d_weights, d_col_buffer, d_output, M, N, K);
}

// Xavier/Glorot initialization helper with fixed seed
void xavier_init(vector<float>& weights, int fan_in, int fan_out) {
    mt19937 gen(42);  // Fixed seed for reproducibility
    float limit = sqrt(6.0f / (fan_in + fan_out));
    uniform_real_distribution<float> dis(-limit, limit);
    
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = dis(gen);
    }
}

int main() {
    cout << "===== Phase 3_2 Training (Im2Col + GEMM) - Optimized Architecture =====" << endl;
    
    // Hyperparameters
    const int H = 32, W = 32, C = 3;
    const int BATCH_SIZE = 32;
    const int NUM_EPOCHS = 20;
    const float LEARNING_RATE = 0.001f;
    
    // ARCHITECTURE CONFIGURATION
    // IMPORTANT: Latent space MUST be (8,8,128) = 8,192 dimensions (project requirement)
    // Therefore C2 MUST be 128 (cannot reduce)
    // We can only reduce C1 from 256 to 128 for faster training
    
    // Option 1: Original (256/128) - Better accuracy, slower training
    // Option 2: Reduced (128/128) - Faster training, still meets requirement (latent = 8,192)
    const bool USE_REDUCED_ARCH = true;  // Set to false for original 256/128
    
    const int C1 = USE_REDUCED_ARCH ? 128 : 256;  // Conv1 output channels (can reduce)
    const int C2 = 128;   // Conv2 output channels (MUST be 128 for latent space requirement)
    
    if (USE_REDUCED_ARCH) {
        cout << "[INFO] Using OPTIMIZED architecture: " << C1 << "/" << C2 << " channels" << endl;
        cout << "[INFO] Note: C1 reduced from 256 to 128 for faster training" << endl;
        cout << "[INFO] Note: C2 kept at 128 to maintain latent space (8,8,128) = 8,192 dims (REQUIRED)" << endl;
    } else {
        cout << "[INFO] Using ORIGINAL architecture: " << C1 << "/" << C2 << " channels" << endl;
        cout << "[INFO] Latent space: (8,8,128) = 8,192 dimensions (project requirement)" << endl;
    }
    
    // OPTIONAL: Reduce number of classes for faster training and better convergence
    // Empty vector = use all 10 classes (full dataset: 50,000 images)
    // Example: {0,1,2,3,4} = use only first 5 classes (25,000 images, ~2x faster)
    const bool USE_SUBSET_CLASSES = true;  // Set to false to use all classes
    vector<int> selected_classes;
    if (USE_SUBSET_CLASSES) {
        // Use first 5 classes: airplane, automobile, bird, cat, deer
        selected_classes = {0, 1, 2, 3, 4};
        cout << "[INFO] Using subset of classes for faster training: " << selected_classes.size() << " classes" << endl;
    }
    
    // Initialize data loader and logger
    DataLoader loader("../../Data/cifar-10-batches-bin/", BATCH_SIZE, true, false, selected_classes);
    Logger logger("logs/phase3_v2_training.log");
    
    logger.log_training_start(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);
    
    // Allocate weights (128/64 architecture - reduced for faster training)
    vector<float> w_conv1(C1 * C * 3 * 3);
    vector<float> w_conv2(C2 * C1 * 3 * 3);
    vector<float> w_dec1(C2 * C2 * 3 * 3);
    vector<float> w_dec2(C1 * C2 * 3 * 3);
    vector<float> w_final(C * C1 * 3 * 3);
    
    // Initialize weights with Xavier initialization (seed 42 for reproducibility)
    cout << "[INFO] Initializing weights with Xavier initialization (seed=42)..." << endl;
    xavier_init(w_conv1, C * 3 * 3, C1 * 3 * 3);        // Conv1: 3→128
    xavier_init(w_conv2, C1 * 3 * 3, C2 * 3 * 3);      // Conv2: 128→64
    xavier_init(w_dec1, C2 * 3 * 3, C2 * 3 * 3);       // Dec1: 64→64
    xavier_init(w_dec2, C2 * 3 * 3, C1 * 3 * 3);       // Dec2: 64→128
    xavier_init(w_final, C1 * 3 * 3, C * 3 * 3);       // Final: 128→3
    
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
    
    // Allocate activation buffers for BATCH processing (using reduced channels)
    float *d_input = (float*)gpu_malloc(BATCH_SIZE * C * H * W * sizeof(float));
    float *d_conv1_out = (float*)gpu_malloc(BATCH_SIZE * C1 * H * W * sizeof(float));
    float *d_pool1_out = (float*)gpu_malloc(BATCH_SIZE * C1 * (H/2) * (W/2) * sizeof(float));
    float *d_conv2_out = (float*)gpu_malloc(BATCH_SIZE * C2 * (H/2) * (W/2) * sizeof(float));
    float *d_pool2_out = (float*)gpu_malloc(BATCH_SIZE * C2 * (H/4) * (W/4) * sizeof(float));
    float *d_dec1_out = (float*)gpu_malloc(BATCH_SIZE * C2 * (H/4) * (W/4) * sizeof(float));
    float *d_ups1_out = (float*)gpu_malloc(BATCH_SIZE * C2 * (H/2) * (W/2) * sizeof(float));
    float *d_dec2_out = (float*)gpu_malloc(BATCH_SIZE * C1 * (H/2) * (W/2) * sizeof(float));
    float *d_ups2_out = (float*)gpu_malloc(BATCH_SIZE * C1 * H * W * sizeof(float));
    float *d_output = (float*)gpu_malloc(BATCH_SIZE * C * H * W * sizeof(float));
    
    // Im2Col buffer for batch (reduced size)
    size_t max_col_size = BATCH_SIZE * C1 * 9 * 32 * 32 * sizeof(float);
    float *d_col_buffer = (float*)gpu_malloc(max_col_size);
    
    // Pre-allocate gradient buffers (ONCE, reused for all images in batch)
    float *d_grad_output = (float*)gpu_malloc(C * H * W * sizeof(float));
    float *d_grad_ups2_out = (float*)gpu_malloc(C1 * H * W * sizeof(float));
    float *d_grad_dec2_out = (float*)gpu_malloc(C1 * (H/2) * (W/2) * sizeof(float));
    float *d_grad_ups1_out = (float*)gpu_malloc(C2 * (H/2) * (W/2) * sizeof(float));
    float *d_grad_dec1_out = (float*)gpu_malloc(C2 * (H/4) * (W/4) * sizeof(float));
    float *d_grad_pool2_out = (float*)gpu_malloc(C2 * (H/4) * (W/4) * sizeof(float));
    float *d_grad_conv2_out = (float*)gpu_malloc(C2 * (H/2) * (W/2) * sizeof(float));
    float *d_grad_pool1_out = (float*)gpu_malloc(C1 * (H/2) * (W/2) * sizeof(float));
    float *d_grad_conv1_out = (float*)gpu_malloc(C1 * H * W * sizeof(float));
    float *d_grad_input = (float*)gpu_malloc(C * H * W * sizeof(float));
    
    cout << "[INFO] Memory allocated (with pre-allocated gradient buffers), starting training..." << endl;
    
    // OPTIMIZATION: Create CUDA streams for parallel backward processing
    const int NUM_STREAMS = 4;  // Process 4 images concurrently
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    cout << "[INFO] Created " << NUM_STREAMS << " CUDA streams for parallel backward pass" << endl;
    
    // Training loop
    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
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
            
            // Gradient buffers are pre-allocated, no need to allocate/free
            
            float batch_loss = 0.0f;
            
            // OPTIMIZED: Process entire batch in forward pass (much faster!)
            // Batch input is already on GPU, no need to copy individual images
            dim3 block(16, 16);
            
            // FORWARD PASS - BATCH PROCESSING (using reduced channels C1/C2)
            // Conv1 + ReLU + MaxPool (Batch)
            forward_conv_layer_batch(batch_input, d_w_conv1, d_conv1_out, d_col_buffer, BATCH_SIZE, H, W, C, C1);
            relu<<<(BATCH_SIZE*C1*H*W+255)/256, 256>>>(d_conv1_out, BATCH_SIZE*C1*H*W);
            // MaxPool: launch for each image in batch (kernels don't support batch, but we can parallelize launches)
            for (int b = 0; b < BATCH_SIZE; b++) {
                maxpool<<<dim3((W/2+15)/16, (H/2+15)/16, C1), block>>>(
                    d_conv1_out + b*C1*H*W, d_pool1_out + b*C1*(H/2)*(W/2), H, W, C1);
            }
            
            // Conv2 + ReLU + MaxPool (Batch)
            forward_conv_layer_batch(d_pool1_out, d_w_conv2, d_conv2_out, d_col_buffer, BATCH_SIZE, H/2, W/2, C1, C2);
            relu<<<(BATCH_SIZE*C2*H/2*W/2+255)/256, 256>>>(d_conv2_out, BATCH_SIZE*C2*H/2*W/2);
            for (int b = 0; b < BATCH_SIZE; b++) {
                maxpool<<<dim3((W/4+15)/16, (H/4+15)/16, C2), block>>>(
                    d_conv2_out + b*C2*(H/2)*(W/2), d_pool2_out + b*C2*(H/4)*(W/4), H/2, W/2, C2);
            }
            
            // Decoder: Dec1 + Upsample + Dec2 + Upsample + Final (Batch)
            forward_conv_layer_batch(d_pool2_out, d_w_dec1, d_dec1_out, d_col_buffer, BATCH_SIZE, H/4, W/4, C2, C2);
            for (int b = 0; b < BATCH_SIZE; b++) {
                upsample<<<dim3((W/2+15)/16, (H/2+15)/16, C2), block>>>(
                    d_dec1_out + b*C2*(H/4)*(W/4), d_ups1_out + b*C2*(H/2)*(W/2), H/4, W/4, C2);
            }
            forward_conv_layer_batch(d_ups1_out, d_w_dec2, d_dec2_out, d_col_buffer, BATCH_SIZE, H/2, W/2, C2, C1);
            for (int b = 0; b < BATCH_SIZE; b++) {
                upsample<<<dim3((W+15)/16, (H+15)/16, C1), block>>>(
                    d_dec2_out + b*C1*(H/2)*(W/2), d_ups2_out + b*C1*H*W, H/2, W/2, C1);
            }
            forward_conv_layer_batch(d_ups2_out, d_w_final, d_output, d_col_buffer, BATCH_SIZE, H, W, C1, C);
            
            // Sync once after entire forward pass
            cudaDeviceSynchronize();
            
            // COMPUTE LOSS for each image (backward still needs per-image processing)
            for (int b = 0; b < BATCH_SIZE; b++) {
                float loss = mse_loss_forward(d_output + b*C*H*W, batch_input + b*C*H*W, C * H * W);
                batch_loss += loss;
            }
            
            // BACKWARD PASS - OPTIMIZED with CUDA Streams for parallel processing
            // Process multiple images concurrently using different streams
            for (int b = 0; b < BATCH_SIZE; b++) {
                int stream_id = b % NUM_STREAMS;  // Round-robin stream assignment
                cudaStream_t stream = streams[stream_id];
                
                // Pointers to current image in batch (using reduced channels)
                float* img_input = batch_input + b * C * H * W;
                float* img_conv1_out = d_conv1_out + b * C1 * H * W;
                float* img_pool1_out = d_pool1_out + b * C1 * (H/2) * (W/2);
                float* img_conv2_out = d_conv2_out + b * C2 * (H/2) * (W/2);
                float* img_pool2_out = d_pool2_out + b * C2 * (H/4) * (W/4);
                float* img_dec1_out = d_dec1_out + b * C2 * (H/4) * (W/4);
                float* img_ups1_out = d_ups1_out + b * C2 * (H/2) * (W/2);
                float* img_dec2_out = d_dec2_out + b * C1 * (H/2) * (W/2);
                float* img_ups2_out = d_ups2_out + b * C1 * H * W;
                float* img_output = d_output + b * C * H * W;
                
                // MSE Loss Backward (kernels launched will use the stream)
                mse_loss_backward(img_output, img_input, d_grad_output, C * H * W);
                
                // Backward through Final Conv (C1→3)
                dim3 grid_final_bw_w((3+15)/16, (3+15)/16, C1*C);
                dim3 grid_final_bw_i((W+15)/16, (H+15)/16, C1);
                conv2d_backward_weights<<<grid_final_bw_w, block, 0, stream>>>(
                    d_grad_output, img_ups2_out, d_grad_w_final,
                    H, W, C1, H, W, C, 3
                );
                conv2d_backward_input<<<grid_final_bw_i, block, 0, stream>>>(
                    d_grad_output, d_w_final, d_grad_ups2_out,
                    H, W, C1, H, W, C, 3
                );
                
                // Backward through Upsample2 (16×16 → 32×32)
                dim3 grid_ups2_bw((W/2+15)/16, (H/2+15)/16, C1);
                upsample_backward<<<grid_ups2_bw, block, 0, stream>>>(
                    d_grad_ups2_out, d_grad_dec2_out, H/2, W/2, C1
                );
                
                // Backward through Dec2 Conv (C2→C1)
                dim3 grid_dec2_bw_w((3+15)/16, (3+15)/16, C1*C2);
                dim3 grid_dec2_bw_i((W/2+15)/16, (H/2+15)/16, C2);
                conv2d_backward_weights<<<grid_dec2_bw_w, block, 0, stream>>>(
                    d_grad_dec2_out, img_ups1_out, d_grad_w_dec2,
                    H/2, W/2, C2, H/2, W/2, C1, 3
                );
                conv2d_backward_input<<<grid_dec2_bw_i, block, 0, stream>>>(
                    d_grad_dec2_out, d_w_dec2, d_grad_ups1_out,
                    H/2, W/2, C2, H/2, W/2, C1, 3
                );
                
                // Backward through Upsample1 (8×8 → 16×16)
                dim3 grid_ups1_bw((W/4+15)/16, (H/4+15)/16, C2);
                upsample_backward<<<grid_ups1_bw, block, 0, stream>>>(
                    d_grad_ups1_out, d_grad_dec1_out, H/4, W/4, C2
                );
                
                // Backward through Dec1 Conv (C2→C2)
                dim3 grid_dec1_bw_w((3+15)/16, (3+15)/16, C2*C2);
                dim3 grid_dec1_bw_i((W/4+15)/16, (H/4+15)/16, C2);
                conv2d_backward_weights<<<grid_dec1_bw_w, block, 0, stream>>>(
                    d_grad_dec1_out, img_pool2_out, d_grad_w_dec1,
                    H/4, W/4, C2, H/4, W/4, C2, 3
                );
                conv2d_backward_input<<<grid_dec1_bw_i, block, 0, stream>>>(
                    d_grad_dec1_out, d_w_dec1, d_grad_pool2_out,
                    H/4, W/4, C2, H/4, W/4, C2, 3
                );
                
                // Backward through MaxPool2 (simplified - no argmax)
                cudaMemcpyAsync(d_grad_conv2_out, d_grad_pool2_out, 
                          C2*(H/2)*(W/2)*sizeof(float), cudaMemcpyDeviceToDevice, stream);
                
                // Backward through ReLU2
                relu_backward<<<(C2*H/2*W/2+255)/256, 256, 0, stream>>>(
                    d_grad_conv2_out, img_conv2_out, d_grad_conv2_out, C2*H/2*W/2
                );
                
                // Backward through Conv2 (C1→C2)
                dim3 grid_conv2_bw_w((3+15)/16, (3+15)/16, C2*C1);
                dim3 grid_conv2_bw_i((W/2+15)/16, (H/2+15)/16, C1);
                conv2d_backward_weights<<<grid_conv2_bw_w, block, 0, stream>>>(
                    d_grad_conv2_out, img_pool1_out, d_grad_w_conv2,
                    H/2, W/2, C1, H/2, W/2, C2, 3
                );
                conv2d_backward_input<<<grid_conv2_bw_i, block, 0, stream>>>(
                    d_grad_conv2_out, d_w_conv2, d_grad_pool1_out,
                    H/2, W/2, C1, H/2, W/2, C2, 3
                );
                
                // Backward through MaxPool1 (simplified - no argmax)
                cudaMemcpyAsync(d_grad_conv1_out, d_grad_pool1_out,
                          C1*H*W*sizeof(float), cudaMemcpyDeviceToDevice, stream);
                
                // Backward through ReLU1
                relu_backward<<<(C1*H*W+255)/256, 256, 0, stream>>>(
                    d_grad_conv1_out, img_conv1_out, d_grad_conv1_out, C1*H*W
                );
                
                // Backward through Conv1 (3→C1)
                dim3 grid_conv1_bw_w((3+15)/16, (3+15)/16, C1*C);
                dim3 grid_conv1_bw_i((W+15)/16, (H+15)/16, C);
                conv2d_backward_weights<<<grid_conv1_bw_w, block, 0, stream>>>(
                    d_grad_conv1_out, img_input, d_grad_w_conv1,
                    H, W, C, H, W, C1, 3
                );
                conv2d_backward_input<<<grid_conv1_bw_i, block, 0, stream>>>(
                    d_grad_conv1_out, d_w_conv1, d_grad_input,
                    H, W, C, H, W, C1, 3
                );
            }
            
            // Synchronize all streams before weight update
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaStreamSynchronize(streams[i]);
            }
            
            // Average loss
            batch_loss /= BATCH_SIZE;
            epoch_loss += batch_loss;
            
            // CLIP GRADIENTS (prevent explosion)
            const float MAX_GRAD_NORM = 0.1f;  // Very conservative to prevent explosion
            clip_gradients(d_grad_w_conv1, w_conv1.size(), MAX_GRAD_NORM);
            clip_gradients(d_grad_w_conv2, w_conv2.size(), MAX_GRAD_NORM);
            clip_gradients(d_grad_w_dec1, w_dec1.size(), MAX_GRAD_NORM);
            clip_gradients(d_grad_w_dec2, w_dec2.size(), MAX_GRAD_NORM);
            clip_gradients(d_grad_w_final, w_final.size(), MAX_GRAD_NORM);
            
            // UPDATE WEIGHTS (SGD)
            sgd_update(d_w_conv1, d_grad_w_conv1, LEARNING_RATE, w_conv1.size());
            sgd_update(d_w_conv2, d_grad_w_conv2, LEARNING_RATE, w_conv2.size());
            sgd_update(d_w_dec1, d_grad_w_dec1, LEARNING_RATE, w_dec1.size());
            sgd_update(d_w_dec2, d_grad_w_dec2, LEARNING_RATE, w_dec2.size());
            sgd_update(d_w_final, d_grad_w_final, LEARNING_RATE, w_final.size());
            
            // Gradient buffers are pre-allocated, no need to free here
            
            num_batches++;
            
            // Progress display every 5%
            if (num_batches % progress_interval == 0) {
                auto now = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::seconds>(now - epoch_start).count();
                int percent = ((num_batches * 20) / total_batches) * 5;  // Round to 5%, 10%, 15%...
                cout << "  " << percent << "% - Loss: " << batch_loss << " - Time: " << elapsed << "s" << endl;
            }
        }
        
        auto epoch_end = chrono::high_resolution_clock::now();
        auto epoch_time = chrono::duration_cast<chrono::seconds>(epoch_end - epoch_start).count();
        
        float avg_loss = epoch_loss / num_batches;
        cout << "Epoch " << epoch << " complete - Avg Loss: " << avg_loss << " - Time: " << epoch_time << "s" << endl << endl;
        logger.log_epoch(epoch, avg_loss);
        
        loader.reset();
        
        // Save weights every epoch
        char filename[256];
        sprintf(filename, "weights/phase3_v2_epoch_%d.bin", epoch);
        
        // Copy weights from GPU to host
        gpu_memcpy_d2h(w_conv1.data(), d_w_conv1, w_conv1.size() * sizeof(float));
        gpu_memcpy_d2h(w_conv2.data(), d_w_conv2, w_conv2.size() * sizeof(float));
        gpu_memcpy_d2h(w_dec1.data(), d_w_dec1, w_dec1.size() * sizeof(float));
        gpu_memcpy_d2h(w_dec2.data(), d_w_dec2, w_dec2.size() * sizeof(float));
        gpu_memcpy_d2h(w_final.data(), d_w_final, w_final.size() * sizeof(float));
        
        // Write to file
        ofstream weight_file(filename, ios::binary);
        if (weight_file.is_open()) {
            weight_file.write(reinterpret_cast<const char*>(w_conv1.data()), w_conv1.size() * sizeof(float));
            weight_file.write(reinterpret_cast<const char*>(w_conv2.data()), w_conv2.size() * sizeof(float));
            weight_file.write(reinterpret_cast<const char*>(w_dec1.data()), w_dec1.size() * sizeof(float));
            weight_file.write(reinterpret_cast<const char*>(w_dec2.data()), w_dec2.size() * sizeof(float));
            weight_file.write(reinterpret_cast<const char*>(w_final.data()), w_final.size() * sizeof(float));
            weight_file.close();
            cout << "[INFO] Saved weights to " << filename << endl;
        } else {
            cerr << "[WARNING] Could not save weights to " << filename << endl;
        }
    }
    
    logger.log_training_end();
    
    // Cleanup CUDA streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cout << "[INFO] Destroyed CUDA streams" << endl;
    
    // Cleanup
    gpu_free(d_w_conv1); gpu_free(d_w_conv2); gpu_free(d_w_dec1); 
    gpu_free(d_w_dec2); gpu_free(d_w_final);
    gpu_free(d_grad_w_conv1); gpu_free(d_grad_w_conv2); gpu_free(d_grad_w_dec1);
    gpu_free(d_grad_w_dec2); gpu_free(d_grad_w_final);
    gpu_free(d_input); gpu_free(d_conv1_out); gpu_free(d_pool1_out);
    gpu_free(d_conv2_out); gpu_free(d_pool2_out); gpu_free(d_dec1_out);
    gpu_free(d_ups1_out); gpu_free(d_dec2_out); gpu_free(d_ups2_out);
    gpu_free(d_output); gpu_free(d_col_buffer);
    // Cleanup pre-allocated gradient buffers
    gpu_free(d_grad_output); gpu_free(d_grad_ups2_out); gpu_free(d_grad_dec2_out);
    gpu_free(d_grad_ups1_out); gpu_free(d_grad_dec1_out); gpu_free(d_grad_pool2_out);
    gpu_free(d_grad_conv2_out); gpu_free(d_grad_pool1_out); gpu_free(d_grad_conv1_out);
    gpu_free(d_grad_input);
    
    cout << "===== Training Complete =====" << endl;
    return 0;
}
