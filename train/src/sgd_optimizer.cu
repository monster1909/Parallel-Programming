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

// ==============================
// Gradient Clipping
// ==============================

__global__ void compute_grad_norm_kernel(
    const float* grad,
    float* partial_sums,
    int N
) {
    __shared__ float shared_sum[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Compute squared gradient
    float sum = 0.0f;
    if (idx < N) {
        float g = grad[idx];
        sum = g * g;
    }
    
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

__global__ void clip_gradients_kernel(
    float* grad,
    float grad_norm,
    float max_norm,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N && grad_norm > max_norm) {
        float scale = max_norm / (grad_norm + 1e-6f);
        grad[idx] *= scale;
    }
}

void clip_gradients(
    float* d_grad,
    int N,
    float max_norm
) {
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    
    // Allocate temporary buffer for partial sums
    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, num_blocks * sizeof(float));
    
    // Compute gradient norm
    compute_grad_norm_kernel<<<num_blocks, block_size>>>(d_grad, d_partial_sums, N);
    
    // Sum partial results on CPU (simple approach)
    float* h_partial_sums = new float[num_blocks];
    cudaMemcpy(h_partial_sums, d_partial_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_sum += h_partial_sums[i];
    }
    float grad_norm = sqrtf(total_sum);
    
    // Clip gradients if needed
    if (grad_norm > max_norm) {
        clip_gradients_kernel<<<num_blocks, block_size>>>(d_grad, grad_norm, max_norm, N);
        cudaDeviceSynchronize();
    }
    
    // Cleanup
    delete[] h_partial_sums;
    cudaFree(d_partial_sums);
}
