#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include <cuda_runtime.h>

// ==============================
// MSE Loss Forward
// ==============================

// Compute MSE loss: sum((output - target)^2) / N
__global__ void mse_loss_forward_kernel(
    const float* output,
    const float* target,
    float* partial_loss,  // Partial sums for reduction
    int N
);

// Host function to compute total loss
float mse_loss_forward(
    const float* d_output,
    const float* d_target,
    int N
);

// ==============================
// MSE Loss Backward
// ==============================

// Compute gradient: 2 * (output - target) / N
__global__ void mse_loss_backward_kernel(
    const float* output,
    const float* target,
    float* grad_output,
    int N
);

// Host function wrapper
void mse_loss_backward(
    const float* d_output,
    const float* d_target,
    float* d_grad_output,
    int N
);

#endif // MSE_LOSS_H
