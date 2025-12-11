#include "../include/backward_kernels.h"

// ==============================
// ReLU Backward
// ==============================

__global__ void relu_backward(
    const float* grad_output,
    const float* input_before_relu,  // Saved from forward pass
    float* grad_input,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Gradient is passed through if input > 0, else 0
        grad_input[idx] = (input_before_relu[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}
