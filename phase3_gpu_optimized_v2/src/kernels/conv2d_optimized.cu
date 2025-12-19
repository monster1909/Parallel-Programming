#include <cuda_runtime.h>
extern "C" __global__ void conv2d_optimized(
    const float* input, const float* kernel, float* output,
    int H, int W, int in_channels, int out_channels, bool apply_relu = false)
{
    const int TILE_SIZE = 16;
    const int K = 3; 
    const int pad = 1;
    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2]; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    int out_c = blockIdx.z;
    if (row >= H || col >= W) return;
    float sum = 0.0f;
    for (int in_c = 0; in_c < in_channels; in_c++) {
        int load_row = row - pad + ty;
        int load_col = col - pad + tx;
        if (load_row >= 0 && load_row < H && load_col >= 0 && load_col < W) {
            tile[ty][tx] = input[in_c * H * W + load_row * W + load_col];
        } else {
            tile[ty][tx] = 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int ky = 0; ky < K; ky++) {
            #pragma unroll
            for (int kx = 0; kx < K; kx++) {
                int kernel_idx = ((out_c * in_channels + in_c) * K + ky) * K + kx;
                sum += tile[ty + ky][tx + kx] * kernel[kernel_idx];
            }
        }
        __syncthreads();
    }
    if (apply_relu) {
        sum = fmaxf(sum, 0.0f);
    }
    output[out_c * H * W + row * W + col] = sum;
}
void conv2d_fused_optimized(const float* d_input, const float* d_weights, float* d_output,
                            int H, int W, int C_in, int C_out) {
    dim3 block(16, 16);
    dim3 grid((W + 15)/16, (H + 15)/16, C_out);
    conv2d_optimized<<<grid, block>>>(d_input, d_weights, d_output, H, W, C_in, C_out, true);
}
void conv2d_only_optimized(const float* d_input, const float* d_weights, float* d_output,
                           int H, int W, int C_in, int C_out) {
    dim3 block(16, 16);
    dim3 grid((W + 15)/16, (H + 15)/16, C_out);
    conv2d_optimized<<<grid, block>>>(d_input, d_weights, d_output, H, W, C_in, C_out, false);
}
