#include <cuda_runtime.h>

extern "C" __global__ void im2col_kernel(const float* data_im, float* data_col,
                                         int channels, int height, int width,
                                         int ksize, int pad, int stride,
                                         int height_col, int width_col) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int input_channels_col = channels * ksize * ksize;
    int num_kernels = height_col * width_col; 

    if (index >= input_channels_col * num_kernels) return;

    int w_out = index % num_kernels;       
    int c_in_k = index / num_kernels;      
    int c = c_in_k / (ksize * ksize);      
    int k_offset = c_in_k % (ksize * ksize);
    int ky = k_offset / ksize;             
    int kx = k_offset % ksize;             
    int h_out = w_out / width_col;
    int w_out_map = w_out % width_col;
    int h_in = h_out * stride - pad + ky;
    int w_in = w_out_map * stride - pad + kx;

    float val = 0.0f;
    if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
        val = data_im[(c * height + h_in) * width + w_in];
    }
    data_col[index] = val;
}

void im2col_gpu(const float* data_im, float* data_col,
                int channels, int height, int width,
                int ksize, int pad, int stride, 
                int h_out, int w_out) 
{
    int num_kernels = channels * ksize * ksize * h_out * w_out;
    int threads = 256;
    int blocks = (num_kernels + threads - 1) / threads;
    im2col_kernel<<<blocks, threads>>>(data_im, data_col, channels, height, width, ksize, pad, stride, h_out, w_out);
}