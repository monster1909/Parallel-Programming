#ifndef MSE_LOSS_H
#define MSE_LOSS_H
#include <cuda_runtime.h>
__global__ void mse_loss_forward_kernel(
    const float* output,
    const float* target,
    float* partial_loss,  
    int N
);
float mse_loss_forward(
    const float* d_output,
    const float* d_target,
    int N
);
__global__ void mse_loss_backward_kernel(
    const float* output,
    const float* target,
    float* grad_output,
    int N
);
void mse_loss_backward(
    const float* d_output,
    const float* d_target,
    float* d_grad_output,
    int N
);
#endif 
