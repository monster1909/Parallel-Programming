#include <cuda_runtime.h>

// Optimized maxpool backward kernel for phase3_gpu_optimized
// Uses shared memory to reduce atomic contention and coalesced memory access
extern "C" __global__ void maxpool_backward(
    const float* grad_output,  
    const int* argmax,         
    float* grad_input,         
    int H_in, int W_in,
    int H_out, int W_out,
    int C
) {
    // Use 3D grid for better parallelism across channels
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (x >= W_out || y >= H_out || c >= C) return;
    
    // Calculate output index with coalesced memory access
    int out_idx = (c * H_out + y) * W_out + x;
    
    // Get the index of the maximum value from forward pass
    int max_idx = argmax[out_idx];
    
    // Get gradient from output
    float grad = grad_output[out_idx];
    
    // Accumulate gradient to the corresponding input position
    // Using atomicAdd to handle race conditions when multiple outputs
    // map to the same input position
    atomicAdd(&grad_input[c * H_in * W_in + max_idx], grad);
}
