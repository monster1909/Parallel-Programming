#include "layers/upsample.h"

void upsample2x_forward(const float* in, int C, int H, int W, float* out) {
    int H2 = H*2, W2 = W*2;
    for (int c=0;c<C;++c) {
        for (int h=0; h<H2; ++h) {
            for (int w=0; w<W2; ++w) {
                out[(c*H2 + h) * W2 + w] = in[(c*H + h/2) * W + w/2];
            }
        }
    }
}

void upsample2x_backward(const float* grad_out, int C, int H, int W, float* grad_in) {
    int H2 = H*2, W2 = W*2;
    for (int i=0;i<C*H*W;++i) grad_in[i]=0.0f;
    for (int c=0;c<C;++c) {
        for (int h=0; h<H2; ++h) {
            for (int w=0; w<W2; ++w) {
                grad_in[(c*H + h/2) * W + w/2] += grad_out[(c*H2 + h) * W2 + w];
            }
        }
    }
}