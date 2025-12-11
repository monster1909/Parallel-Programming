#include "../include/mse_loss.h"
#include <cuda_runtime.h>
#include <stdio.h>

// ==============================
// MSE Loss Forward Kernel
// ==============================

__global__ void mse_loss_forward_kernel(
    const float* output,
    const float* target,
    float* partial_loss,  // Partial sums for reduction
    int N
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute squared difference
    float diff = 0.0f;
    if (idx < N) {
        float d = output[idx] - target[idx];
        diff = d * d;
    }
    sdata[tid] = diff;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_loss[blockIdx.x] = sdata[0];
    }
}

// ==============================
// MSE Loss Forward (Host)
// ==============================

float mse_loss_forward(
    const float* d_output,
    const float* d_target,
    int N
) {
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    
    // Allocate partial loss buffer
    float* d_partial_loss;
    cudaMalloc(&d_partial_loss, num_blocks * sizeof(float));
    
    // Launch kernel
    mse_loss_forward_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_output, d_target, d_partial_loss, N
    );
    
    // Copy partial results to host and sum
    float* h_partial_loss = new float[num_blocks];
    cudaMemcpy(h_partial_loss, d_partial_loss, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_loss = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_loss += h_partial_loss[i];
    }
    
    // Cleanup
    delete[] h_partial_loss;
    cudaFree(d_partial_loss);
    
    // Return mean squared error
    return total_loss / N;
}

// ==============================
// MSE Loss Backward Kernel
// ==============================

__global__ void mse_loss_backward_kernel(
    const float* output,
    const float* target,
    float* grad_output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Gradient: 2 * (output - target) / N
        grad_output[idx] = 2.0f * (output[idx] - target[idx]) / N;
    }
}

// ==============================
// MSE Loss Backward (Host)
// ==============================

void mse_loss_backward(
    const float* d_output,
    const float* d_target,
    float* d_grad_output,
    int N
) {
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    
    mse_loss_backward_kernel<<<num_blocks, block_size>>>(
        d_output, d_target, d_grad_output, N
    );
    cudaDeviceSynchronize();
}
