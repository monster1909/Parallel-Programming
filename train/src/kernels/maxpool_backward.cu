#include "../include/backward_kernels.h"

// ==============================
// MaxPool Backward
// ==============================

__global__ void maxpool_backward(
    const float* grad_output,  // [C, H_out, W_out]
    const int* argmax,         // Saved from forward pass [C, H_out, W_out]
    float* grad_input,         // [C, H_in, W_in]
    int H_in, int W_in,
    int H_out, int W_out,
    int C
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x >= W_out || y >= H_out || c >= C) return;

    int out_idx = (c * H_out + y) * W_out + x;
    int max_pos = argmax[out_idx];
    
    // Scatter gradient to the position that had max value
    if (max_pos >= 0 && max_pos < H_in * W_in) {
        int in_idx = c * H_in * W_in + max_pos;
        atomicAdd(&grad_input[in_idx], grad_output[out_idx]);
    }
}
