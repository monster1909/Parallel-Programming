#include <cuda_runtime.h>

// Optimized tile sizes
#define TILE_WIDTH_SMALL 16
#define TILE_WIDTH_LARGE 32

// Original GEMM kernel (16x16 tiles)
extern "C" __global__ void gemm_tiled(const float* A, const float* B, float* C, 
                                      int M, int N, int K) 
{
    __shared__ float ds_A[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL + 1];  // +1 to avoid bank conflicts
    __shared__ float ds_B[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL + 1];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH_SMALL + ty;
    int col = bx * TILE_WIDTH_SMALL + tx;
    float sum = 0.0f;

    for (int p = 0; p < (K - 1) / TILE_WIDTH_SMALL + 1; ++p) {
        if (row < M && p * TILE_WIDTH_SMALL + tx < K)
            ds_A[ty][tx] = A[row * K + p * TILE_WIDTH_SMALL + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (p * TILE_WIDTH_SMALL + ty < K && col < N)
            ds_B[ty][tx] = B[(p * TILE_WIDTH_SMALL + ty) * N + col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH_SMALL; ++i) sum += ds_A[ty][i] * ds_B[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// FUSED GEMM + ReLU kernel (16x16 tiles)
extern "C" __global__ void gemm_tiled_relu(const float* A, const float* B, float* C, 
                                           int M, int N, int K) 
{
    __shared__ float ds_A[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL + 1];
    __shared__ float ds_B[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL + 1];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH_SMALL + ty;
    int col = bx * TILE_WIDTH_SMALL + tx;
    float sum = 0.0f;

    for (int p = 0; p < (K - 1) / TILE_WIDTH_SMALL + 1; ++p) {
        if (row < M && p * TILE_WIDTH_SMALL + tx < K)
            ds_A[ty][tx] = A[row * K + p * TILE_WIDTH_SMALL + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (p * TILE_WIDTH_SMALL + ty < K && col < N)
            ds_B[ty][tx] = B[(p * TILE_WIDTH_SMALL + ty) * N + col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH_SMALL; ++i) sum += ds_A[ty][i] * ds_B[i][tx];
        __syncthreads();
    }
    
    // FUSED: Apply ReLU immediately
    if (row < M && col < N) {
        C[row * N + col] = fmaxf(sum, 0.0f);
    }
}

// OPTIMIZED GEMM with larger tiles (32x32) for better performance
extern "C" __global__ void gemm_tiled_optimized(const float* A, const float* B, float* C, 
                                                int M, int N, int K) 
{
    __shared__ float ds_A[TILE_WIDTH_LARGE][TILE_WIDTH_LARGE + 1];
    __shared__ float ds_B[TILE_WIDTH_LARGE][TILE_WIDTH_LARGE + 1];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH_LARGE + ty;
    int col = bx * TILE_WIDTH_LARGE + tx;
    float sum = 0.0f;

    for (int p = 0; p < (K - 1) / TILE_WIDTH_LARGE + 1; ++p) {
        if (row < M && p * TILE_WIDTH_LARGE + tx < K)
            ds_A[ty][tx] = A[row * K + p * TILE_WIDTH_LARGE + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (p * TILE_WIDTH_LARGE + ty < K && col < N)
            ds_B[ty][tx] = B[(p * TILE_WIDTH_LARGE + ty) * N + col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH_LARGE; ++i) {
            sum += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) C[row * N + col] = sum;
}

// OPTIMIZED FUSED GEMM + ReLU with larger tiles (32x32)
extern "C" __global__ void gemm_tiled_relu_optimized(const float* A, const float* B, float* C, 
                                                     int M, int N, int K) 
{
    __shared__ float ds_A[TILE_WIDTH_LARGE][TILE_WIDTH_LARGE + 1];
    __shared__ float ds_B[TILE_WIDTH_LARGE][TILE_WIDTH_LARGE + 1];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH_LARGE + ty;
    int col = bx * TILE_WIDTH_LARGE + tx;
    float sum = 0.0f;

    for (int p = 0; p < (K - 1) / TILE_WIDTH_LARGE + 1; ++p) {
        if (row < M && p * TILE_WIDTH_LARGE + tx < K)
            ds_A[ty][tx] = A[row * K + p * TILE_WIDTH_LARGE + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (p * TILE_WIDTH_LARGE + ty < K && col < N)
            ds_B[ty][tx] = B[(p * TILE_WIDTH_LARGE + ty) * N + col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH_LARGE; ++i) {
            sum += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads();
    }
    
    // FUSED: Apply ReLU immediately
    if (row < M && col < N) {
        C[row * N + col] = fmaxf(sum, 0.0f);
    }
}