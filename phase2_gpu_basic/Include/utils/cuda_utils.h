#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>

using namespace std;

inline void checkCuda(cudaError_t status, const char *msg = "")
{
    if (status != cudaSuccess)
    {
        cout << "CUDA Error: " << msg << " : "
             << cudaGetErrorString(status) << endl;
        exit(EXIT_FAILURE);
    }
}

#endif
