#include <cuda_runtime.h>
extern "C" __global__ void conv2d_backward_input(
    const float* grad_output,  
    const float* weights,      
    float* grad_input,         
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c_in = blockIdx.z;
    if (x >= W_in || y >= H_in || c_in >= C_in) return;
    float sum = 0.0f;
    int pad = K / 2;
    for (int c_out = 0; c_out < C_out; c_out++) {
        for (int ky = 0; ky < K; ky++) {
            for (int kx = 0; kx < K; kx++) {
                int out_y = y + pad - ky;
                int out_x = x + pad - kx;
                if (out_y >= 0 && out_y < H_out && out_x >= 0 && out_x < W_out) {
                    int grad_out_idx = (c_out * H_out + out_y) * W_out + out_x;
                    int weight_idx = ((c_out * C_in + c_in) * K + ky) * K + kx;
                    sum += grad_output[grad_out_idx] * weights[weight_idx];
                }
            }
        }
    }
    int grad_in_idx = (c_in * H_in + y) * W_in + x;
    grad_input[grad_in_idx] = sum;
}
extern "C" __global__ void conv2d_backward_weights(
    const float* grad_output,  
    const float* input,        
    float* grad_weights,       
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K
) {
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z / C_in;
    int c_in = blockIdx.z % C_in;
    if (kx >= K || ky >= K || c_out >= C_out || c_in >= C_in) return;
    float sum = 0.0f;
    int pad = K / 2;
    for (int out_y = 0; out_y < H_out; out_y++) {
        for (int out_x = 0; out_x < W_out; out_x++) {
            int in_y = out_y - pad + ky;
            int in_x = out_x - pad + kx;
            if (in_y >= 0 && in_y < H_in && in_x >= 0 && in_x < W_in) {
                int grad_out_idx = (c_out * H_out + out_y) * W_out + out_x;
                int input_idx = (c_in * H_in + in_y) * W_in + in_x;
                sum += grad_output[grad_out_idx] * input[input_idx];
            }
        }
    }
    int grad_w_idx = ((c_out * C_in + c_in) * K + ky) * K + kx;
    atomicAdd(&grad_weights[grad_w_idx], sum);
}
