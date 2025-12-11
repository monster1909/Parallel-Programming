#include "../include/sgd_optimizer.h"
#include <cuda_runtime.h>

// ==============================
// SGD Update Kernel
// ==============================

__global__ void sgd_update_kernel(
    float* weights,
    const float* grad_weights,
    float learning_rate,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Update: w = w - lr * grad_w
        weights[idx] -= learning_rate * grad_weights[idx];
    }
}

// ==============================
// SGD Update (Host)
// ==============================

void sgd_update(
    float* d_weights,
    const float* d_grad_weights,
    float learning_rate,
    int N
) {
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    
    sgd_update_kernel<<<num_blocks, block_size>>>(
        d_weights, d_grad_weights, learning_rate, N
    );
    cudaDeviceSynchronize();
}

// ==============================
// Zero Gradient Kernel
// ==============================

__global__ void zero_gradient_kernel(
    float* grad,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        grad[idx] = 0.0f;
    }
}

// ==============================
// Zero Gradient (Host)
// ==============================

void zero_gradient(
    float* d_grad,
    int N
) {
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    
    zero_gradient_kernel<<<num_blocks, block_size>>>(d_grad, N);
    cudaDeviceSynchronize();
}
