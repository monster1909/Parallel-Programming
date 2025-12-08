#include "layers/maxpool.h"

void maxpool2x2_forward_with_argmax(const float* in, int C, int H, int W, float* out, int* argmax_out) {
    int H2 = H/2, W2 = W/2;
    for (int c=0;c<C;++c) {
        for (int h=0; h<H2; ++h) {
            for (int w=0; w<W2; ++w) {
                int ih = h*2, iw = w*2;
                float best = -1e9f; int best_idx = -1;
                for (int ph=0; ph<2; ++ph) for (int pw=0; pw<2; ++pw) {
                    int r = ih+ph, s = iw+pw;
                    int idx = (c*H + r) * W + s;
                    if (in[idx] > best) { best = in[idx]; best_idx = idx; }
                }
                out[(c*H2 + h) * W2 + w] = best;
                argmax_out[(c*H2 + h) * W2 + w] = best_idx;
            }
        }
    }
}

void maxpool2x2_backward_with_argmax(const float* grad_out, int C, int H, int W, const int* argmax_out, float* grad_in) {
    int H2 = H/2, W2 = W/2;
    int in_size = C*H*W;
    for (int i=0;i<in_size;++i) grad_in[i]=0.0f;
    for (int c=0;c<C;++c) {
        for (int h=0; h<H2; ++h) {
            for (int w=0; w<W2; ++w) {
                float g = grad_out[(c*H2 + h) * W2 + w];
                int idx = argmax_out[(c*H2 + h) * W2 + w];
                grad_in[idx] += g;
            }
        }
    }
}