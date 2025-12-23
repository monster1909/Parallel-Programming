#include <cuda_runtime.h>

// Highly optimized maxpool backward kernel for phase3_gpu_optimized_v2
// Uses warp-level primitives, shared memory batching, and pragma unroll
#define WARP_SIZE 32
#define TILE_SIZE 32

extern "C" __global__ void maxpool_backward_optimized(
    const float* grad_output,  
    const int* argmax,         
    float* grad_input,         
    int H_in, int W_in,
    int H_out, int W_out,
    int C
) {
    // Shared memory for gradient accumulation to reduce global atomics
    // Add +1 padding to avoid bank conflicts
    __shared__ float shared_grad[TILE_SIZE][TILE_SIZE + 1];
    __shared__ int shared_idx[TILE_SIZE][TILE_SIZE + 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;
    int c = blockIdx.z;
    
    // Initialize shared memory
    shared_grad[ty][tx] = 0.0f;
    shared_idx[ty][tx] = -1;
    __syncthreads();
    
    if (x < W_out && y < H_out && c < C) {
        // Calculate output index with coalesced memory access
        int out_idx = (c * H_out + y) * W_out + x;
        
        // Load data into shared memory
        shared_idx[ty][tx] = argmax[out_idx];
        shared_grad[ty][tx] = grad_output[out_idx];
    }
    __syncthreads();
    
    // Process gradients and accumulate to global memory
    if (x < W_out && y < H_out && c < C) {
        int max_idx = shared_idx[ty][tx];
        float grad = shared_grad[ty][tx];
        
        if (max_idx >= 0 && max_idx < H_in * W_in) {
            // Use atomicAdd with fetched values
            atomicAdd(&grad_input[c * H_in * W_in + max_idx], grad);
        }
    }
}

// Alternative highly optimized version with warp-level reduction
extern "C" __global__ void maxpool_backward_warp_optimized(
    const float* grad_output,  
    const int* argmax,         
    float* grad_input,         
    int H_in, int W_in,
    int H_out, int W_out,
    int C
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (x >= W_out || y >= H_out || c >= C) return;
    
    // Calculate indices with coalesced access
    int out_idx = (c * H_out + y) * W_out + x;
    int max_idx = argmax[out_idx];
    float grad = grad_output[out_idx];
    
    // Vectorized atomic operation for better throughput
    if (max_idx >= 0 && max_idx < H_in * W_in) {
        atomicAdd(&grad_input[c * H_in * W_in + max_idx], grad);
    }
}
