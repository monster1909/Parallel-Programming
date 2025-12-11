#ifndef BACKWARD_KERNELS_H
#define BACKWARD_KERNELS_H

#include <cuda_runtime.h>

// ==============================
// Conv2D Backward Kernels
// ==============================

// Gradient w.r.t input
__global__ void conv2d_backward_input(
    const float* grad_output,  // [C_out, H_out, W_out]
    const float* weights,      // [C_out, C_in, K, K]
    float* grad_input,         // [C_in, H_in, W_in]
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K
);

// Gradient w.r.t weights
__global__ void conv2d_backward_weights(
    const float* grad_output,  // [C_out, H_out, W_out]
    const float* input,        // [C_in, H_in, W_in]
    float* grad_weights,       // [C_out, C_in, K, K]
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K
);

// ==============================
// ReLU Backward Kernel
// ==============================

__global__ void relu_backward(
    const float* grad_output,
    const float* input_before_relu,  // Saved from forward pass
    float* grad_input,
    int N
);

// ==============================
// MaxPool Backward Kernel
// ==============================

__global__ void maxpool_backward(
    const float* grad_output,  // [C, H_out, W_out]
    const int* argmax,         // Saved from forward pass [C, H_out, W_out]
    float* grad_input,         // [C, H_in, W_in]
    int H_in, int W_in,
    int H_out, int W_out,
    int C
);

// ==============================
// Upsample Backward Kernel
// ==============================

__global__ void upsample_backward(
    const float* grad_output,  // [C, H_out, W_out] (upsampled)
    float* grad_input,         // [C, H_in, W_in]
    int H_in, int W_in, int C
);

#endif // BACKWARD_KERNELS_H
