#include <cuda_runtime.h>

extern "C" __global__ void im2col_kernel(const float* data_im, float* data_col,
                                         int batch_size, int channels, int height, int width,
                                         int ksize, int pad, int stride,
                                         int height_col, int width_col) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Tổng số cột ma trận = Batch * H_out * W_out
    int num_kernels = batch_size * height_col * width_col; 
    int input_channels_col = channels * ksize * ksize;

    if (index >= input_channels_col * num_kernels) return;

    // Giải mã index
    int w_out_idx = index % num_kernels;       
    int c_in_k = index / num_kernels;      
    
    // Tính channel và offset kernel
    int c = c_in_k / (ksize * ksize);      
    int k_offset = c_in_k % (ksize * ksize);
    int ky = k_offset / ksize;             
    int kx = k_offset % ksize;             

    // Giải mã vị trí trong Batch (n, h, w)
    int b = w_out_idx / (height_col * width_col); // Batch index
    int spatial_idx = w_out_idx % (height_col * width_col);
    int h_out = spatial_idx / width_col;
    int w_out_map = spatial_idx % width_col;

    int h_in = h_out * stride - pad + ky;
    int w_in = w_out_map * stride - pad + kx;

    float val = 0.0f;
    if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width) {
        // Offset bộ nhớ input: Batch * ImageSize + Channel * FeatureMap + Row + Col
        int input_offset = b * (channels * height * width) + 
                           (c * height + h_in) * width + w_in;
        val = data_im[input_offset];
    }
    data_col[index] = val;
}

void im2col_gpu(const float* data_im, float* data_col,
                int batch_size, int channels, int height, int width,
                int ksize, int pad, int stride, 
                int h_out, int w_out) 
{
    // Tổng số luồng cần chạy tăng lên theo Batch Size
    int num_kernels = channels * ksize * ksize * batch_size * h_out * w_out;
    int threads = 256;
    int blocks = (num_kernels + threads - 1) / threads;
    
    im2col_kernel<<<blocks, threads>>>(data_im, data_col, batch_size, channels, 
                                       height, width, ksize, pad, stride, h_out, w_out);
}