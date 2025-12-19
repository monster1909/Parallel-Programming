#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif
#include "../../phase3_gpu_optimized_v2/Include/utils/gpu_memory.h"
#include "../../phase3_gpu_optimized_v2/Include/utils/cuda_utils.h"
#include "../../phase3_gpu_optimized_v2/Include/utils/gpu_timer.h"
#include "../../phase3_gpu_optimized_v2/include/kernels/conv2d_backward_gemm_optimized.h"
#include "../../phase3_gpu_optimized_v2/Include/utils/mse_loss.h"
#include "../../phase3_gpu_optimized_v2/Include/utils/sgd_optimizer.h"
#include "../include/data_loader.h"
#include "../include/logger.h"
using namespace std;
void im2col_gpu(const float* data_im, float* data_col, 
                int batch_size, int channels, int height, int width, 
                int ksize, int pad, int stride, 
                int h_out, int w_out);
extern "C" __global__ void gemm_tiled(const float* A, const float* B, float* C, 
                                      int M, int N, int K);
extern "C" __global__ void gemm_tiled_optimized(const float* A, const float* B, float* C, 
                                                int M, int N, int K);
extern "C" __global__ void gemm_tiled_relu(const float* A, const float* B, float* C, 
                                           int M, int N, int K);
extern "C" __global__ void gemm_tiled_relu_optimized(const float* A, const float* B, float* C, 
                                                     int M, int N, int K);
extern "C" __global__ void relu(float* x, int size);
extern "C" __global__ void maxpool(const float* input, float* output, int H, int W, int C);
extern "C" __global__ void upsample(const float* input, float* output, int H, int W, int C);
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
void forward_conv_layer_optimized(const float* d_input, const float* d_weights, float* d_output, float* d_col_buffer,
                                  int H, int W, int C_in, int C_out) 
{
    int ksize = 3, pad = 1, stride = 1;
    int H_out = H, W_out = W;
    im2col_gpu(d_input, d_col_buffer, 1, C_in, H, W, ksize, pad, stride, H_out, W_out);
    int M = C_out;
    int N = H_out * W_out;
    int K = C_in * ksize * ksize;
    dim3 dimGrid((N + 31)/32, (M + 31)/32);
    dim3 dimBlock(32, 32);
    gemm_tiled_optimized<<<dimGrid, dimBlock>>>(d_weights, d_col_buffer, d_output, M, N, K);
}
void forward_conv_layer_fused_relu(const float* d_input, const float* d_weights, float* d_output, float* d_col_buffer,
                                    int H, int W, int C_in, int C_out) 
{
    int ksize = 3, pad = 1, stride = 1;
    int H_out = H, W_out = W;
    im2col_gpu(d_input, d_col_buffer, 1, C_in, H, W, ksize, pad, stride, H_out, W_out);
    int M = C_out;
    int N = H_out * W_out;
    int K = C_in * ksize * ksize;
    dim3 dimGrid((N + 31)/32, (M + 31)/32);
    dim3 dimBlock(32, 32);
    gemm_tiled_relu_optimized<<<dimGrid, dimBlock>>>(d_weights, d_col_buffer, d_output, M, N, K);
}
void xavier_init(vector<float>& weights, int fan_in, int fan_out) {
    mt19937 gen(42);  
    float limit = sqrt(6.0f / (fan_in + fan_out));
    uniform_real_distribution<float> dis(-limit, limit);
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = dis(gen);
    }
}
void save_ppm(const float* data, const string& filename, int H, int W, int C) {
    ofstream f(filename);
    if (!f) {
        cerr << "[WARNING] Could not save image to " << filename << endl;
        return;
    }
    f << "P3\n" << W << " " << H << "\n255\n";
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int r = (int)(data[(0 * H + h) * W + w] * 255.0f);
            int g = (int)(data[(1 * H + h) * W + w] * 255.0f);
            int b = (int)(data[(2 * H + h) * W + w] * 255.0f);
            r = max(0, min(255, r)); 
            g = max(0, min(255, g)); 
            b = max(0, min(255, b));
            f << r << " " << g << " " << b << " ";
        }
        f << "\n";
    }
    f.close();
    cout << "[INFO] Saved image to " << filename << endl;
}
int main() {
    cout << "===== Phase 3_2 Training (Im2Col + GEMM) =====" << endl;
    const int H = 32, W = 32, C = 3;
    const int BATCH_SIZE = 32;
    const int NUM_EPOCHS = 20;
    const float LEARNING_RATE = 0.001f;
    mkdir("logs", 0755);
    mkdir("weights", 0755);
    DataLoader loader("../../Data/cifar-10-batches-bin/", BATCH_SIZE);
    Logger logger("logs/phase3_v2_training.log");
    logger.log_training_start(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);
    auto total_train_start = chrono::high_resolution_clock::now();
    vector<float> epoch_times;
    float final_loss = 0.0f;
    const int EARLY_STOP_PATIENCE = 3;  
    float best_loss = 1e10f;  
    int epochs_without_improvement = 0;
    bool early_stopped = false;
    vector<float> w_conv1(256 * C * 3 * 3);
    vector<float> w_conv2(128 * 256 * 3 * 3);
    vector<float> w_dec1(128 * 128 * 3 * 3);
    vector<float> w_dec2(256 * 128 * 3 * 3);
    vector<float> w_final(C * 256 * 3 * 3);
    cout << "[INFO] Initializing weights with Xavier initialization (seed=42)..." << endl;
    xavier_init(w_conv1, C * 3 * 3, 256 * 3 * 3);        
    xavier_init(w_conv2, 256 * 3 * 3, 128 * 3 * 3);      
    xavier_init(w_dec1, 128 * 3 * 3, 128 * 3 * 3);       
    xavier_init(w_dec2, 128 * 3 * 3, 256 * 3 * 3);       
    xavier_init(w_final, 256 * 3 * 3, C * 3 * 3);        
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
    float *d_grad_w_conv1 = (float*)gpu_malloc(w_conv1.size() * sizeof(float));
    float *d_grad_w_conv2 = (float*)gpu_malloc(w_conv2.size() * sizeof(float));
    float *d_grad_w_dec1 = (float*)gpu_malloc(w_dec1.size() * sizeof(float));
    float *d_grad_w_dec2 = (float*)gpu_malloc(w_dec2.size() * sizeof(float));
    float *d_grad_w_final = (float*)gpu_malloc(w_final.size() * sizeof(float));
    float *d_input = (float*)gpu_malloc(BATCH_SIZE * C * H * W * sizeof(float));
    float *d_conv1_out = (float*)gpu_malloc(BATCH_SIZE * 256 * H * W * sizeof(float));
    float *d_pool1_out = (float*)gpu_malloc(BATCH_SIZE * 256 * (H/2) * (W/2) * sizeof(float));
    float *d_conv2_out = (float*)gpu_malloc(BATCH_SIZE * 128 * (H/2) * (W/2) * sizeof(float));
    float *d_pool2_out = (float*)gpu_malloc(BATCH_SIZE * 128 * (H/4) * (W/4) * sizeof(float));
    float *d_dec1_out = (float*)gpu_malloc(BATCH_SIZE * 128 * (H/4) * (W/4) * sizeof(float));
    float *d_ups1_out = (float*)gpu_malloc(BATCH_SIZE * 128 * (H/2) * (W/2) * sizeof(float));
    float *d_dec2_out = (float*)gpu_malloc(BATCH_SIZE * 256 * (H/2) * (W/2) * sizeof(float));
    float *d_ups2_out = (float*)gpu_malloc(BATCH_SIZE * 256 * H * W * sizeof(float));
    float *d_output = (float*)gpu_malloc(BATCH_SIZE * C * H * W * sizeof(float));
    size_t max_col_size = BATCH_SIZE * 256 * 9 * 32 * 32 * sizeof(float);
    float *d_col_buffer = (float*)gpu_malloc(max_col_size);
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
    cout << "[INFO] Memory allocated (using memory pool pattern), starting training..." << endl;
    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        cout << "Training Epoch " << epoch << "/" << NUM_EPOCHS << "..." << endl;
        auto epoch_start = chrono::high_resolution_clock::now();
        int total_batches = loader.get_num_batches();
        int progress_interval = total_batches / 20;  
        if (progress_interval == 0) progress_interval = 1;
        while (loader.has_next()) {
            float* batch_input = loader.next_batch();
            zero_gradient(d_grad_w_conv1, w_conv1.size());
            zero_gradient(d_grad_w_conv2, w_conv2.size());
            zero_gradient(d_grad_w_dec1, w_dec1.size());
            zero_gradient(d_grad_w_dec2, w_dec2.size());
            zero_gradient(d_grad_w_final, w_final.size());
            float batch_loss = 0.0f;
            for (int b = 0; b < BATCH_SIZE; b++) {
                gpu_memcpy_h2d(d_input, batch_input + b * C * H * W, C * H * W * sizeof(float));
                dim3 block(16, 16);
                forward_conv_layer_fused_relu(d_input, d_w_conv1, d_conv1_out, d_col_buffer, H, W, C, 256);
                maxpool<<<dim3((W/2+15)/16, (H/2+15)/16, 256), block>>>(d_conv1_out, d_pool1_out, H, W, 256);
                forward_conv_layer_fused_relu(d_pool1_out, d_w_conv2, d_conv2_out, d_col_buffer, H/2, W/2, 256, 128);
                maxpool<<<dim3((W/4+15)/16, (H/4+15)/16, 128), block>>>(d_conv2_out, d_pool2_out, H/2, W/2, 128);
                forward_conv_layer_optimized(d_pool2_out, d_w_dec1, d_dec1_out, d_col_buffer, H/4, W/4, 128, 128);
                upsample<<<dim3((W/2+15)/16, (H/2+15)/16, 128), block>>>(d_dec1_out, d_ups1_out, H/4, W/4, 128);
                forward_conv_layer_optimized(d_ups1_out, d_w_dec2, d_dec2_out, d_col_buffer, H/2, W/2, 128, 256);
                upsample<<<dim3((W+15)/16, (H+15)/16, 256), block>>>(d_dec2_out, d_ups2_out, H/2, W/2, 256);
                forward_conv_layer_optimized(d_ups2_out, d_w_final, d_output, d_col_buffer, H, W, 256, C);
                cudaDeviceSynchronize();
                float loss = mse_loss_forward(d_output, d_input, C * H * W);
                batch_loss += loss;
                mse_loss_backward(d_output, d_input, d_grad_output, C * H * W);
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
                dim3 grid_ups2_bw((W/2+15)/16, (H/2+15)/16, 256);
                upsample_backward<<<grid_ups2_bw, block>>>(
                    d_grad_ups2_out, d_grad_dec2_out, H/2, W/2, 256
                );
                cudaDeviceSynchronize();
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
                dim3 grid_ups1_bw((W/4+15)/16, (H/4+15)/16, 128);
                upsample_backward<<<grid_ups1_bw, block>>>(
                    d_grad_ups1_out, d_grad_dec1_out, H/4, W/4, 128
                );
                cudaDeviceSynchronize();
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
                cudaMemcpy(d_grad_conv2_out, d_grad_pool2_out, 
                          128*(H/2)*(W/2)*sizeof(float), cudaMemcpyDeviceToDevice);
                relu_backward<<<(128*H/2*W/2+255)/256, 256>>>(
                    d_grad_conv2_out, d_conv2_out, d_grad_conv2_out, 128*H/2*W/2
                );
                cudaDeviceSynchronize();
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
                cudaMemcpy(d_grad_conv1_out, d_grad_pool1_out,
                          256*H*W*sizeof(float), cudaMemcpyDeviceToDevice);
                relu_backward<<<(256*H*W+255)/256, 256>>>(
                    d_grad_conv1_out, d_conv1_out, d_grad_conv1_out, 256*H*W
                );
                cudaDeviceSynchronize();
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
            batch_loss /= BATCH_SIZE;
            epoch_loss += batch_loss;
            const float MAX_GRAD_NORM = 0.1f;  
            clip_gradients(d_grad_w_conv1, w_conv1.size(), MAX_GRAD_NORM);
            clip_gradients(d_grad_w_conv2, w_conv2.size(), MAX_GRAD_NORM);
            clip_gradients(d_grad_w_dec1, w_dec1.size(), MAX_GRAD_NORM);
            clip_gradients(d_grad_w_dec2, w_dec2.size(), MAX_GRAD_NORM);
            clip_gradients(d_grad_w_final, w_final.size(), MAX_GRAD_NORM);
            sgd_update(d_w_conv1, d_grad_w_conv1, LEARNING_RATE, w_conv1.size());
            sgd_update(d_w_conv2, d_grad_w_conv2, LEARNING_RATE, w_conv2.size());
            sgd_update(d_w_dec1, d_grad_w_dec1, LEARNING_RATE, w_dec1.size());
            sgd_update(d_w_dec2, d_grad_w_dec2, LEARNING_RATE, w_dec2.size());
            sgd_update(d_w_final, d_grad_w_final, LEARNING_RATE, w_final.size());
            num_batches++;
            if (num_batches % progress_interval == 0) {
                auto now = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::seconds>(now - epoch_start).count();
                int percent = ((num_batches * 20) / total_batches) * 5;  
                cout << "  " << percent << "% - Loss: " << fixed << setprecision(6) << batch_loss 
                     << " - Time: " << elapsed << "s" << endl;
            }
        }
        auto epoch_end = chrono::high_resolution_clock::now();
        auto epoch_time_seconds = chrono::duration_cast<chrono::milliseconds>(epoch_end - epoch_start).count() / 1000.0f;
        epoch_times.push_back(epoch_time_seconds);
        float avg_loss = epoch_loss / num_batches;
        final_loss = avg_loss;  
        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            epochs_without_improvement = 0;
            cout << "Epoch " << epoch << " complete - Avg Loss: " << fixed << setprecision(6) << avg_loss 
                 << " (Best: " << best_loss << ") - Time: " << setprecision(2) << epoch_time_seconds << "s" << endl << endl;
        } else {
            epochs_without_improvement++;
            cout << "Epoch " << epoch << " complete - Avg Loss: " << fixed << setprecision(6) << avg_loss 
                 << " (Best: " << best_loss << ", No improvement: " << epochs_without_improvement << "/" << EARLY_STOP_PATIENCE 
                 << ") - Time: " << setprecision(2) << epoch_time_seconds << "s" << endl << endl;
            if (epochs_without_improvement >= EARLY_STOP_PATIENCE) {
                early_stopped = true;
                cout << "[INFO] Early stopping triggered! No improvement for " << EARLY_STOP_PATIENCE << " epochs." << endl;
                cout << "[INFO] Best loss: " << fixed << setprecision(6) << best_loss << " at epoch " << (epoch - EARLY_STOP_PATIENCE) << endl;
                logger.log_message("Early stopping triggered - No improvement for " + to_string(EARLY_STOP_PATIENCE) + " epochs");
                break;  
            }
        }
        logger.log_epoch(epoch, avg_loss, epoch_time_seconds);
        loader.reset();
        char filename[256];
        sprintf(filename, "weights/phase3_v2_epoch_%d.bin", epoch);
        gpu_memcpy_d2h(w_conv1.data(), d_w_conv1, w_conv1.size() * sizeof(float));
        gpu_memcpy_d2h(w_conv2.data(), d_w_conv2, w_conv2.size() * sizeof(float));
        gpu_memcpy_d2h(w_dec1.data(), d_w_dec1, w_dec1.size() * sizeof(float));
        gpu_memcpy_d2h(w_dec2.data(), d_w_dec2, w_dec2.size() * sizeof(float));
        gpu_memcpy_d2h(w_final.data(), d_w_final, w_final.size() * sizeof(float));
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
    auto total_train_end = chrono::high_resolution_clock::now();
    auto total_time_seconds = chrono::duration_cast<chrono::milliseconds>(total_train_end - total_train_start).count() / 1000.0f;
    if (early_stopped) {
        cout << "\n[INFO] Training stopped early. Generating summary..." << endl;
        final_loss = best_loss;  
    } else {
        cout << "\n[INFO] Training completed. Generating summary..." << endl;
    }
    size_t free_mem = 0, total_mem = 0;
    size_t used_mem_mb = 0, total_mem_mb = 0;
    cudaError_t mem_error = cudaMemGetInfo(&free_mem, &total_mem);
    if (mem_error == cudaSuccess) {
        size_t used_mem = total_mem - free_mem;
        used_mem_mb = used_mem / (1024 * 1024);
        total_mem_mb = total_mem / (1024 * 1024);
    } else {
        cerr << "[WARNING] Could not get GPU memory info: " << cudaGetErrorString(mem_error) << endl;
    }
    bool sample_images_success = false;
    cout << "[INFO] Generating sample reconstructed images..." << endl;
    cout.flush();
    try {
        vector<float> sample_input(C * H * W);
        vector<float> sample_output(C * H * W);
        mt19937 gen(42);
        uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (int i = 0; i < C * H * W; i++) {
            sample_input[i] = dis(gen);
        }
        gpu_memcpy_h2d(d_input, sample_input.data(), C * H * W * sizeof(float));
        dim3 block(16, 16);
        forward_conv_layer_fused_relu(d_input, d_w_conv1, d_conv1_out, d_col_buffer, H, W, C, 256);
        maxpool<<<dim3((W/2+15)/16, (H/2+15)/16, 256), block>>>(d_conv1_out, d_pool1_out, H, W, 256);
        forward_conv_layer_fused_relu(d_pool1_out, d_w_conv2, d_conv2_out, d_col_buffer, H/2, W/2, 256, 128);
        maxpool<<<dim3((W/4+15)/16, (H/4+15)/16, 128), block>>>(d_conv2_out, d_pool2_out, H/2, W/2, 128);
        forward_conv_layer_optimized(d_pool2_out, d_w_dec1, d_dec1_out, d_col_buffer, H/4, W/4, 128, 128);
        upsample<<<dim3((W/2+15)/16, (H/2+15)/16, 128), block>>>(d_dec1_out, d_ups1_out, H/4, W/4, 128);
        forward_conv_layer_optimized(d_ups1_out, d_w_dec2, d_dec2_out, d_col_buffer, H/2, W/2, 128, 256);
        upsample<<<dim3((W+15)/16, (H+15)/16, 256), block>>>(d_dec2_out, d_ups2_out, H/2, W/2, 256);
        forward_conv_layer_optimized(d_ups2_out, d_w_final, d_output, d_col_buffer, H, W, 256, C);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cerr << "[WARNING] CUDA error during forward pass: " << cudaGetErrorString(err) << endl;
        } else {
            gpu_memcpy_d2h(sample_output.data(), d_output, C * H * W * sizeof(float));
            save_ppm(sample_input.data(), "logs/sample_original_phase3_v2.ppm", H, W, C);
            save_ppm(sample_output.data(), "logs/sample_reconstructed_phase3_v2.ppm", H, W, C);
            cout << "[INFO] Sample images saved to logs/sample_original_phase3_v2.ppm and logs/sample_reconstructed_phase3_v2.ppm" << endl;
            sample_images_success = true;
        }
    } catch (const exception& e) {
        cerr << "[WARNING] Exception generating sample images: " << e.what() << ", continuing with summary..." << endl;
    } catch (...) {
        cerr << "[WARNING] Error generating sample images, continuing with summary..." << endl;
    }
    cout << "\n========================================" << endl;
    cout << "       TRAINING SUMMARY" << endl;
    cout << "========================================" << endl;
    cout.flush();
    if (early_stopped) {
        logger.log_message("Training stopped early due to no improvement");
        cout << "\n[EARLY STOP] Training stopped after epoch " << epoch_times.size() 
             << " (no improvement for " << EARLY_STOP_PATIENCE << " epochs)" << endl;
    }
    cout << "\n--- Training Performance ---" << endl;
    cout << "Total training time: " << fixed << setprecision(2) << total_time_seconds << " seconds" << endl;
    cout << "Number of epochs completed: " << epoch_times.size() << endl;
    if (!epoch_times.empty()) {
        float avg_epoch_time = 0.0f;
        for (float t : epoch_times) avg_epoch_time += t;
        avg_epoch_time /= epoch_times.size();
        cout << "Average time per epoch: " << fixed << setprecision(2) << avg_epoch_time << " seconds" << endl;
        cout << "\nTraining Time Per Epoch:" << endl;
        for (size_t i = 0; i < epoch_times.size(); i++) {
            cout << "  Epoch " << setw(3) << (i+1) << ": " << fixed << setprecision(2) 
                 << epoch_times[i] << " seconds" << endl;
        }
    }
    cout << "Final reconstruction loss: " << fixed << setprecision(6) << final_loss << endl;
    cout << "\n--- Memory Usage ---" << endl;
    cout << "GPU Memory Used: " << used_mem_mb << " MB / " << total_mem_mb << " MB" << endl;
    if (total_mem_mb > 0) {
        cout << "GPU Memory Usage: " << fixed << setprecision(1) 
             << (100.0f * used_mem_mb / total_mem_mb) << "%" << endl;
    } else {
        cout << "GPU Memory Usage: N/A" << endl;
    }
    cout << "\n--- Sample Images ---" << endl;
    if (sample_images_success) {
        cout << "✓ Sample reconstructed images saved to:" << endl;
        cout << "  - logs/sample_original_phase3_v2.ppm" << endl;
        cout << "  - logs/sample_reconstructed_phase3_v2.ppm" << endl;
    } else {
        cout << "✗ Sample image generation failed or was disabled" << endl;
    }
    cout << "========================================\n" << endl;
    cout.flush();
    try {
        logger.log_training_summary(total_time_seconds, epoch_times, final_loss, used_mem_mb, total_mem_mb);
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in log_training_summary: " << e.what() << endl;
        cerr << "[INFO] Summary was printed to console anyway" << endl;
    } catch (...) {
        cerr << "[ERROR] Unknown error in log_training_summary" << endl;
        cerr << "[INFO] Summary was printed to console anyway" << endl;
    }
    logger.log_training_end();
    gpu_free(d_w_conv1); gpu_free(d_w_conv2); gpu_free(d_w_dec1); 
    gpu_free(d_w_dec2); gpu_free(d_w_final);
    gpu_free(d_grad_w_conv1); gpu_free(d_grad_w_conv2); gpu_free(d_grad_w_dec1);
    gpu_free(d_grad_w_dec2); gpu_free(d_grad_w_final);
    gpu_free(d_input); gpu_free(d_conv1_out); gpu_free(d_pool1_out);
    gpu_free(d_conv2_out); gpu_free(d_pool2_out); gpu_free(d_dec1_out);
    gpu_free(d_ups1_out); gpu_free(d_dec2_out); gpu_free(d_ups2_out);
    gpu_free(d_output); gpu_free(d_col_buffer);
    gpu_free(d_grad_output); gpu_free(d_grad_ups2_out); gpu_free(d_grad_dec2_out);
    gpu_free(d_grad_ups1_out); gpu_free(d_grad_dec1_out); gpu_free(d_grad_pool2_out);
    gpu_free(d_grad_conv2_out); gpu_free(d_grad_pool1_out); gpu_free(d_grad_conv1_out);
    gpu_free(d_grad_input);
    cout << "===== Training Complete =====" << endl;
    return 0;
}
