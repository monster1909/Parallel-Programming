#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H
#include <cuda_runtime.h>
__global__ void sgd_update_kernel(
    float* weights,
    const float* grad_weights,
    float learning_rate,
    int N
);
void sgd_update(
    float* d_weights,
    const float* d_grad_weights,
    float learning_rate,
    int N
);
__global__ void zero_gradient_kernel(
    float* grad,
    int N
);
void zero_gradient(
    float* d_grad,
    int N
);
void clip_gradients(
    float* d_grad,
    int N,
    float max_norm = 1.0f
);
#endif 
