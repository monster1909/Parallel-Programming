#ifndef RELU_H
#define RELU_H

void relu_forward_inplace(float* X, int N);
void relu_backward_inplace(float* grad_out, const float* X_before_relu, int N);

#endif