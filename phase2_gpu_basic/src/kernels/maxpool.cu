extern "C" __global__ void maxpool(const float *input,
                                            float *output,
                                            int H, int W, int C)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    int H2 = H / 2;
    int W2 = W / 2;

    if (ox >= W2 || oy >= H2)
        return;

    float m = -1e20f;

    int base_in = c * H * W;

    int x0 = ox * 2;
    int y0 = oy * 2;

#pragma unroll
    for (int ky = 0; ky < 2; ky++)
    {
        for (int kx = 0; kx < 2; kx++)
        {
            int ix = x0 + kx;
            int iy = y0 + ky;

            float v = input[base_in + iy * W + ix];
            m = (v > m ? v : m);
        }
    }

    output[(c * H2 + oy) * W2 + ox] = m;
}
