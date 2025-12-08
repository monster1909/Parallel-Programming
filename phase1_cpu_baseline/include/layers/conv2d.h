#ifndef CONV2D_H
#define CONV2D_H

#include "common.h"

struct Conv2D {
    int in_c, out_c;
    int k, pad, stride;
    int Hin, Win;
    int Hout, Wout;

    float* W;
    float* b;
    float* dW;
    float* db;

    Conv2D();
    void init(int in_c_, int out_c_, int k_, int pad_, int stride_, int Hin_, int Win_);
    void zero_grads();
    void free_all();
    
    // Forward for a single sample (handled inside loop)
    void forward_one(const float* inptr, float* outptr, float* col_buffer);
    
    // Backward for a single sample
    void backward_one(const float* grad_out, const float* input_val, float* grad_in, float* col_buffer);

    void update(float lr, int batch_size);
};

#endif