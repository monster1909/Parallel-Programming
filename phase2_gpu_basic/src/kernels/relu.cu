extern "C" __global__ void relu(float *x, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float v = x[idx];
        x[idx] = (v > 0.f ? v : 0.f);
    }
}
