#include "layers/relu.h"

void relu_forward_inplace(float* X, int N) {
    for (int i=0;i<N;++i) if (X[i] < 0) X[i]=0.0f;
}

void relu_backward_inplace(float* grad_out, const float* X_before_relu, int N) {
    for (int i=0;i<N;++i) if (X_before_relu[i] <= 0.0f) grad_out[i] = 0.0f;
}