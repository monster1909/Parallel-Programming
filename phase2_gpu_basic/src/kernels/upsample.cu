extern "C" __global__ void upsample(const float *input,
                                             float *output,
                                             int H, int W, int C)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    int H2 = H * 2;
    int W2 = W * 2;

    if (x >= W2 || y >= H2)
        return;

    int ix = x >> 1;
    int iy = y >> 1;

    int out_idx = (c * H2 + y) * W2 + x;
    int in_idx = (c * H + iy) * W + ix;

    output[out_idx] = input[in_idx];
}
