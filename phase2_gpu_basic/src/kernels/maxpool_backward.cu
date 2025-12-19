#include <cuda_runtime.h>
extern "C" __global__ void maxpool_backward(
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
    int out_idx = (c * H_out + y) * W_out + x;
    int max_idx = argmax[out_idx];
    atomicAdd(&grad_input[c * H_in * W_in + max_idx], grad_output[out_idx]);
}
