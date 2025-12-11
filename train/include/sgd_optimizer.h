#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include <cuda_runtime.h>

// ==============================
// SGD Update Kernel
// ==============================

// Update weights: w = w - lr * grad_w
__global__ void sgd_update_kernel(
    float* weights,
    const float* grad_weights,
    float learning_rate,
    int N
);

// Host function wrapper
void sgd_update(
    float* d_weights,
    const float* d_grad_weights,
    float learning_rate,
    int N
);

// ==============================
// Zero Gradient Kernel
// ==============================

// Zero out gradient buffer
__global__ void zero_gradient_kernel(
    float* grad,
    int N
);

// Host function wrapper
void zero_gradient(
    float* d_grad,
    int N
);

#endif // SGD_OPTIMIZER_H
