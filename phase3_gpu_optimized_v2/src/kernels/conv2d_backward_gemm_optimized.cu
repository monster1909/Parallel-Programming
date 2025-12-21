#include <cuda_runtime.h>
extern "C" __global__ void col2im_kernel_optimized(
    const float* data_col,
    float* data_im,
    int channels, int height, int width,
    int ksize, int pad, int stride,
    int height_col, int width_col
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = height * width * channels;
    if (index >= total_pixels) return;
    int w = index % width;
    int h = (index / width) % height;
    int c = index / (width * height);
    float val = 0.0f;
    #pragma unroll
    for (int ky = 0; ky < ksize; ky++) {
        #pragma unroll
        for (int kx = 0; kx < ksize; kx++) {
            int h_pad = h + pad;
            int w_pad = w + pad;
            int h_out = (h_pad - ky);
            int w_out = (w_pad - kx);
            if (h_out >= 0 && h_out < height_col && w_out >= 0 && w_out < width_col) {
                int col_idx = ((c * ksize * ksize + ky * ksize + kx) * height_col + h_out) * width_col + w_out;
                val += data_col[col_idx];
            }
        }
    }
    data_im[index] = val;
}
void col2im_gpu_optimized(const float* data_col, float* data_im,
                          int channels, int height, int width,
                          int ksize, int pad, int stride,
                          int h_out, int w_out)
{
    int num_kernels = channels * height * width;
    int threads = 256;
    int blocks = (num_kernels + threads - 1) / threads;
    col2im_kernel_optimized<<<blocks, threads>>>(data_col, data_im, channels, height, width,
                                                  ksize, pad, stride, h_out, w_out);
}
extern "C" __global__ void gemm_tiled(const float* A, const float* B, float* C,
                                      int M, int N, int K);
extern "C" __global__ void gemm_tiled_optimized(const float* A, const float* B, float* C,
                                                int M, int N, int K);
void im2col_gpu(const float* data_im, float* data_col,
                int batch_size, int channels, int height, int width,
                int ksize, int pad, int stride,
                int h_out, int w_out);
void conv2d_backward_weights_gemm_optimized(
    const float* grad_output,  
    const float* input,        
    float* grad_weights,       
    float* col_buffer,         
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K, int pad, int stride
) {
    im2col_gpu(input, col_buffer, 1, C_in, H_in, W_in, K, pad, stride, H_out, W_out);
    int M = C_out;
    int N = C_in * K * K;
    int K_gemm = H_out * W_out;
    // Use 16x16 tiles for better performance with small matrices
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + 15) / 16, (M + 15) / 16);
    gemm_tiled<<<dimGrid, dimBlock>>>(grad_output, col_buffer, grad_weights, M, N, K_gemm);
}
void conv2d_backward_input_gemm_optimized(
    const float* grad_output,  
    const float* weights,      
    float* grad_input,         
    float* col_buffer,         
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K, int pad, int stride
) {
    int M = C_in * K * K;
    int N = H_out * W_out;
    int K_gemm = C_out;
    // Use 16x16 tiles for better performance
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + 15) / 16, (M + 15) / 16);
    gemm_tiled<<<dimGrid, dimBlock>>>(weights, grad_output, col_buffer, M, N, K_gemm);
    cudaDeviceSynchronize();
    col2im_gpu_optimized(col_buffer, grad_input, C_in, H_in, W_in, K, pad, stride, H_out, W_out);
}
