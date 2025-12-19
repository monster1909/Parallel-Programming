#ifndef CONV2D_BACKWARD_GEMM_OPTIMIZED_H
#define CONV2D_BACKWARD_GEMM_OPTIMIZED_H
void conv2d_backward_weights_gemm_optimized(
    const float* grad_output, const float* input, float* grad_weights, float* col_buffer,
    int H_in, int W_in, int C_in, int H_out, int W_out, int C_out,
    int K, int pad, int stride);
void conv2d_backward_input_gemm_optimized(
    const float* grad_output, const float* weights, float* grad_input, float* col_buffer,
    int H_in, int W_in, int C_in, int H_out, int W_out, int C_out,
    int K, int pad, int stride);
extern "C" __global__ void relu_backward(const float* grad_output, const float* input_before_relu, float* grad_input, int N);
extern "C" __global__ void upsample_backward(const float* grad_output, float* grad_input, int H_in, int W_in, int C);
#endif
