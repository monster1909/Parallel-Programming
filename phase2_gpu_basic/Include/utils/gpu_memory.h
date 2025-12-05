#ifndef GPU_MEMORY_H
#define GPU_MEMORY_H

#include <cuda_runtime.h>
#include <iostream>
#include "cuda_utils.h"

using namespace std;

inline void *gpu_malloc(size_t bytes)
{
    void *ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
}

inline void gpu_free(void *ptr)
{
    if (ptr)
        cudaFree(ptr);
}

inline void gpu_memcpy_h2d(void *dst, const void *src, size_t bytes)
{
    checkCuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

inline void gpu_memcpy_d2h(void *dst, const void *src, size_t bytes)
{
    checkCuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}

#endif
