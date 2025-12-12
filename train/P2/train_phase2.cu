#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <cmath>

// Include utilities từ phase2
#include "../../phase2_gpu_basic/Include/utils/gpu_memory.h"
#include "../../phase2_gpu_basic/Include/utils/cuda_utils.h"
#include "../../phase2_gpu_basic/Include/utils/gpu_timer.h"

// Include backward kernels từ train/
#include "../include/backward_kernels.h"
#include "../include/mse_loss.h"
#include "../include/sgd_optimizer.h"

// Include utilities từ train/
#include "../include/data_loader.h"
#include "../include/logger.h"

using namespace std;

// Forward kernel declarations từ phase2
extern "C" __global__ void conv2d(const float *input, const float *kernel, float *output,
                                  int H, int W, int in_channels, int out_channels);
extern "C" __global__ void relu(float *x, int size);
extern "C" __global__ void maxpool(const float *input, float *output, int H, int W, int C);
extern "C" __global__ void upsample(const float *input, float *output, int H, int W, int C);

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
    cout << "===== Phase 2 Training =====" << endl;
    
    // Hyperparameters (FIXED)
    const int H = 32, W = 32, C = 3;
    const int BATCH_SIZE = 32;  
    const int NUM_EPOCHS = 20;   
    const float LEARNING_RATE = 0.001f;  
    
    // Initialize data loader and logger
    DataLoader loader("../../Data/cifar-10-batches-bin/", BATCH_SIZE);
    Logger logger("logs/phase2_training.log");
    
    logger.log_training_start(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);
    
    // Allocate weights
    vector<float> w_conv1(256 * C * 3 * 3);
    vector<float> w_conv2(128 * 256 * 3 * 3);
    vector<float> w_dec1(128 * 128 * 3 * 3);
    vector<float> w_dec2(256 * 128 * 3 * 3);
    vector<float> w_final(C * 256 * 3 * 3);
    
    // Initialize weights with Xavier initialization (seed 42 for reproducibility)
    cout << "[INFO] Initializing weights with Xavier initialization (seed=42)..." << endl;
    xavier_init(w_conv1, C * 3 * 3, 256 * 3 * 3);        // Conv1: 3→256
    xavier_init(w_conv2, 256 * 3 * 3, 128 * 3 * 3);      // Conv2: 256→128
    xavier_init(w_dec1, 128 * 3 * 3, 128 * 3 * 3);       // Dec1: 128→128
    xavier_init(w_dec2, 128 * 3 * 3, 256 * 3 * 3);       // Dec2: 128→256
    xavier_init(w_final, 256 * 3 * 3, C * 3 * 3);        // Final: 256→3
    
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
    
    // Allocate activation buffers (for single image, will process batch sequentially)
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
    
    // ALLOCATE GRADIENT BUFFERS ONCE (FIX!)
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
    
    cout << "[INFO] Memory allocated, starting training..." << endl;
    
    // Training loop
    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        int total_batches = loader.get_num_batches();
        int update_interval = total_batches / 20;  // Update every 5% (100/5 = 20 updates)
        if (update_interval == 0) update_interval = 1;
        
        auto epoch_start = chrono::high_resolution_clock::now();
        
        cout << "Training Epoch " << epoch << "/" << NUM_EPOCHS << "..." << endl;
        
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
                
                // FORWARD PASS (using phase2 kernels)
                dim3 block(16, 16);
                
                // Conv1 + ReLU + MaxPool
                conv2d<<<dim3((W+15)/16, (H+15)/16, 256), block>>>(d_input, d_w_conv1, d_conv1_out, H, W, C, 256);
                relu<<<(256*H*W+255)/256, 256>>>(d_conv1_out, 256*H*W);
                maxpool<<<dim3((W/2+15)/16, (H/2+15)/16, 256), block>>>(d_conv1_out, d_pool1_out, H, W, 256);
                
                // Conv2 + ReLU + MaxPool
                conv2d<<<dim3((W/2+15)/16, (H/2+15)/16, 128), block>>>(d_pool1_out, d_w_conv2, d_conv2_out, H/2, W/2, 256, 128);
                relu<<<(128*H/2*W/2+255)/256, 256>>>(d_conv2_out, 128*H/2*W/2);
                maxpool<<<dim3((W/4+15)/16, (H/4+15)/16, 128), block>>>(d_conv2_out, d_pool2_out, H/2, W/2, 128);
                
                // Decoder: Dec1 + Upsample + Dec2 + Upsample + Final
                conv2d<<<dim3((W/4+15)/16, (H/4+15)/16, 128), block>>>(d_pool2_out, d_w_dec1, d_dec1_out, H/4, W/4, 128, 128);
                upsample<<<dim3((W/2+15)/16, (H/2+15)/16, 128), block>>>(d_dec1_out, d_ups1_out, H/4, W/4, 128);
                conv2d<<<dim3((W/2+15)/16, (H/2+15)/16, 256), block>>>(d_ups1_out, d_w_dec2, d_dec2_out, H/2, W/2, 128, 256);
                upsample<<<dim3((W+15)/16, (H+15)/16, 256), block>>>(d_dec2_out, d_ups2_out, H/2, W/2, 256);
                conv2d<<<dim3((W+15)/16, (H+15)/16, C), block>>>(d_ups2_out, d_w_final, d_output, H, W, 256, C);
                
                cudaDeviceSynchronize();
                
                // COMPUTE LOSS (MSE: target = input for autoencoder)
                float loss = mse_loss_forward(d_output, d_input, C * H * W);
                batch_loss += loss;
                
                // BACKWARD PASS (Full implementation)
                // Note: Gradient buffers already allocated outside loop
                
                // 1. MSE Loss Backward
                mse_loss_backward(d_output, d_input, d_grad_output, C * H * W);
                
                // 2. Backward through Final Conv (256→3)
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
                
                // 3. Backward through Upsample2 (16×16 → 32×32)
                dim3 grid_ups2_bw((W/2+15)/16, (H/2+15)/16, 256);
                upsample_backward<<<grid_ups2_bw, block>>>(
                    d_grad_ups2_out, d_grad_dec2_out, H/2, W/2, 256
                );
                cudaDeviceSynchronize();
                
                // 4. Backward through Dec2 Conv (128→256)
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
                
                // 5. Backward through Upsample1 (8×8 → 16×16)
                dim3 grid_ups1_bw((W/4+15)/16, (H/4+15)/16, 128);
                upsample_backward<<<grid_ups1_bw, block>>>(
                    d_grad_ups1_out, d_grad_dec1_out, H/4, W/4, 128
                );
                cudaDeviceSynchronize();
                
                // 6. Backward through Dec1 Conv (128→128)
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
                
                // 7. Backward through MaxPool2
                // Note: Need argmax from forward pass - simplified here
                // maxpool_backward<<<grid_pool2_bw, block>>>(
                //     d_grad_pool2_out, d_pool2_argmax, d_grad_conv2_out,
                //     H/2, W/2, H/4, W/4, 128
                // );
                // For now, use simple gradient pass-through (need to save argmax in forward)
                cudaMemcpy(d_grad_conv2_out, d_grad_pool2_out, 
                          128*(H/2)*(W/2)*sizeof(float), cudaMemcpyDeviceToDevice);
                
                // 8. Backward through ReLU2
                relu_backward<<<(128*H/2*W/2+255)/256, 256>>>(
                    d_grad_conv2_out, d_conv2_out, d_grad_conv2_out, 128*H/2*W/2
                );
                cudaDeviceSynchronize();
                
                // 9. Backward through Conv2 (256→128)
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
                
                // 10. Backward through MaxPool1
                // maxpool_backward<<<grid_pool1_bw, block>>>(
                //     d_grad_pool1_out, d_pool1_argmax, d_grad_conv1_out,
                //     H, W, H/2, W/2, 256
                // );
                cudaMemcpy(d_grad_conv1_out, d_grad_pool1_out,
                          256*H*W*sizeof(float), cudaMemcpyDeviceToDevice);
                
                // 11. Backward through ReLU1
                relu_backward<<<(256*H*W+255)/256, 256>>>(
                    d_grad_conv1_out, d_conv1_out, d_grad_conv1_out, 256*H*W
                );
                cudaDeviceSynchronize();
                
                // 12. Backward through Conv1 (3→256)
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
            
            num_batches++;
            
            // Print progress every 5%
            if (num_batches % update_interval == 0 || num_batches == total_batches) {
                auto now = chrono::high_resolution_clock::now();
                float elapsed = chrono::duration<float>(now - epoch_start).count();
                int progress_pct = ((num_batches * 20) / total_batches) * 5;  // Round to 5%, 10%, 15%...
                cout << "  " << progress_pct << "% - Loss: " << batch_loss 
                     << " - Time: " << (int)elapsed << "s" << endl;
            }
        }
        
        auto epoch_end = chrono::high_resolution_clock::now();
        float epoch_time = chrono::duration<float>(epoch_end - epoch_start).count();
        
        float avg_loss = epoch_loss / num_batches;
        
        cout << "Epoch " << epoch << " complete - Avg Loss: " << avg_loss 
             << " - Time: " << epoch_time << "s" << endl << endl;
        
        logger.log_epoch(epoch, avg_loss);
        
        loader.reset();
        
        // Save weights every epoch
        char filename[256];
        sprintf(filename, "weights/phase2_epoch_%d.bin", epoch);
        
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
    
    // Cleanup
    gpu_free(d_w_conv1); gpu_free(d_w_conv2); gpu_free(d_w_dec1); 
    gpu_free(d_w_dec2); gpu_free(d_w_final);
    gpu_free(d_grad_w_conv1); gpu_free(d_grad_w_conv2); gpu_free(d_grad_w_dec1);
    gpu_free(d_grad_w_dec2); gpu_free(d_grad_w_final);
    gpu_free(d_input); gpu_free(d_conv1_out); gpu_free(d_pool1_out);
    gpu_free(d_conv2_out); gpu_free(d_pool2_out); gpu_free(d_dec1_out);
    gpu_free(d_ups1_out); gpu_free(d_dec2_out); gpu_free(d_ups2_out);
    gpu_free(d_output);
    
    // Cleanup gradient buffers
    gpu_free(d_grad_output); gpu_free(d_grad_ups2_out); gpu_free(d_grad_dec2_out);
    gpu_free(d_grad_ups1_out); gpu_free(d_grad_dec1_out); gpu_free(d_grad_pool2_out);
    gpu_free(d_grad_conv2_out); gpu_free(d_grad_pool1_out); gpu_free(d_grad_conv1_out);
    gpu_free(d_grad_input);
    
    cout << "===== Training Complete =====" << endl;
    return 0;
}
