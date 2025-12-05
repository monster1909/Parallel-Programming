#include <cuda_runtime.h>

extern "C" __global__ void conv2d(const float *__restrict__ input,
                                        const float *__restrict__ kernel,
                                        float *output,
                                        int H, int W,
                                        int in_channels, int out_channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;

    if (x >= W || y >= H)
        return;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ic++)
    {
#pragma unroll
        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {

                int ix = x + kx;
                int iy = y + ky;

                if (ix < 0 || iy < 0 || ix >= W || iy >= H)
                    continue;

                int in_idx = (ic * H + iy) * W + ix;
                int k_idx = ((oc * in_channels + ic) * 3 + (ky + 1)) * 3 + (kx + 1);

                sum += input[in_idx] * kernel[k_idx];
            }
        }
    }

    int out_idx = (oc * H + y) * W + x;
    output[out_idx] = sum;
}
