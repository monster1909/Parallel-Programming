#include "../include/backward_kernels.h"

// ==============================
// Upsample Backward
// ==============================

__global__ void upsample_backward(
    const float* grad_output,  // [C, H_out, W_out] (upsampled, H_out = 2*H_in)
    float* grad_input,         // [C, H_in, W_in]
    int H_in, int W_in, int C
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x >= W_in || y >= H_in || c >= C) return;

    int H_out = H_in * 2;
    int W_out = W_in * 2;

    // Sum gradients from the 2x2 block in output
    float sum = 0.0f;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int out_y = y * 2 + dy;
            int out_x = x * 2 + dx;
            int out_idx = (c * H_out + out_y) * W_out + out_x;
            sum += grad_output[out_idx];
        }
    }

    int in_idx = (c * H_in + y) * W_in + x;
    grad_input[in_idx] = sum;
}
