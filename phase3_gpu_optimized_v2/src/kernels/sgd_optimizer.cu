#include "../../phase3_gpu_optimized_v2/Include/utils/sgd_optimizer.h"
#include <cuda_runtime.h>
#include <cmath>

// Phase 3_2: Simplified but effective optimizations
__global__ void sgd_update_kernel(
    float* weights,
    const float* grad_weights,
    float learning_rate,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Vectorized access for better memory throughput (only for aligned data)
    if (idx < N / 4) {
        float4* w4 = (float4*)weights;
        const float4* g4 = (const float4*)grad_weights;
        
        float4 w = w4[idx];
        float4 g = g4[idx];
        
        w.x -= learning_rate * g.x;
        w.y -= learning_rate * g.y;
        w.z -= learning_rate * g.z;
        w.w -= learning_rate * g.w;
        
        w4[idx] = w;
    }
    
    // Handle remaining elements
    int base_idx = (N / 4) * 4;
    int remaining_idx = base_idx + (blockIdx.x * blockDim.x + threadIdx.x);
    if (remaining_idx < N && idx >= N / 4) {
        weights[remaining_idx] -= learning_rate * grad_weights[remaining_idx];
    }
}

void sgd_update(
    float* d_weights,
    const float* d_grad_weights,
    float learning_rate,
    int N
) {
    const int block_size = 256;
    const int num_blocks = (N / 4 + block_size - 1) / block_size;
    sgd_update_kernel<<<num_blocks, block_size>>>(
        d_weights, d_grad_weights, learning_rate, N
    );
    // No sync for better pipelining
}

__global__ void zero_gradient_kernel(
    float* grad,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Vectorized zeroing
    if (idx < N / 4) {
        float4* g4 = (float4*)grad;
        g4[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    // Handle remaining elements
    int base_idx = (N / 4) * 4;
    int remaining_idx = base_idx + (blockIdx.x * blockDim.x + threadIdx.x);
    if (remaining_idx < N && idx >= N / 4) {
        grad[remaining_idx] = 0.0f;
    }
}

void zero_gradient(
    float* d_grad,
    int N
) {
    const int block_size = 256;
    const int num_blocks = (N / 4 + block_size - 1) / block_size;
    zero_gradient_kernel<<<num_blocks, block_size>>>(d_grad, N);
}

// Simplified gradient norm computation without cooperative groups
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
    
    // Optimized reduction with unrolling
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
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
    
    // Vectorized clipping
    if (idx < N / 4) {
        float4* g4 = (float4*)grad;
        float4 g = g4[idx];
        
        g.x *= scale;
        g.y *= scale;
        g.z *= scale;
        g.w *= scale;
        
        g4[idx] = g;
    }
    
    // Handle remaining elements
    int base_idx = (N / 4) * 4;
    int remaining_idx = base_idx + (blockIdx.x * blockDim.x + threadIdx.x);
    if (remaining_idx < N && idx >= N / 4) {
        grad[remaining_idx] *= scale;
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
        const int clip_blocks = (N / 4 + block_size - 1) / block_size;
        clip_gradient_kernel<<<clip_blocks, block_size>>>(d_grad, scale, N);
    }
}
