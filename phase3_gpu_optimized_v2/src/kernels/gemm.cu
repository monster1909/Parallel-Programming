#include <cuda_runtime.h>
#define TILE_WIDTH 16

extern "C" __global__ void gemm_tiled(const float* A, const float* B, float* C, 
                                      int M, int N, int K) 
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float sum = 0.0f;

    for (int p = 0; p < (K - 1) / TILE_WIDTH + 1; ++p) {
        if (row < M && p * TILE_WIDTH + tx < K)
            ds_A[ty][tx] = A[row * K + p * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (p * TILE_WIDTH + ty < K && col < N)
            ds_B[ty][tx] = B[(p * TILE_WIDTH + ty) * N + col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; ++i) sum += ds_A[ty][i] * ds_B[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}