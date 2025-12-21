#include "../../phase2_gpu_basic/Include/utils/sgd_optimizer.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void sgd_update_kernel(
    float* weights,
    const float* grad_weights,
    float learning_rate,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        weights[idx] -= learning_rate * grad_weights[idx];
    }
}

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

__global__ void zero_gradient_kernel(
    float* grad,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad[idx] = 0.0f;
    }
}

void zero_gradient(
    float* d_grad,
    int N
) {
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    zero_gradient_kernel<<<num_blocks, block_size>>>(d_grad, N);
    cudaDeviceSynchronize();
}

__global__ void compute_gradient_norm_kernel(
    const float* grad,
    float* partial_norm,
    int N
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = 0.0f;
    if (idx < N) {
        val = grad[idx] * grad[idx];
    }
    sdata[tid] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_norm[blockIdx.x] = sdata[0];
    }
}

__global__ void clip_gradient_kernel(
    float* grad,
    float scale,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
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
    
    // Compute gradient norm
    float* d_partial_norm;
    cudaMalloc(&d_partial_norm, num_blocks * sizeof(float));
    
    compute_gradient_norm_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_grad, d_partial_norm, N
    );
    
    float* h_partial_norm = new float[num_blocks];
    cudaMemcpy(h_partial_norm, d_partial_norm, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_norm_sq = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_norm_sq += h_partial_norm[i];
    }
    float total_norm = sqrtf(total_norm_sq);
    
    delete[] h_partial_norm;
    cudaFree(d_partial_norm);
    
    // Clip if necessary
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        clip_gradient_kernel<<<num_blocks, block_size>>>(d_grad, scale, N);
        cudaDeviceSynchronize();
    }
}
