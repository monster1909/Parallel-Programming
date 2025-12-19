#include <cuda_runtime.h>
extern "C" __global__ void relu_backward(
    const float* grad_output,
    const float* input_before_relu,  
    float* grad_input,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_input[idx] = (input_before_relu[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}
