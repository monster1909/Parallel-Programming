#ifndef MAXPOOL_H
#define MAXPOOL_H

void maxpool2x2_forward_with_argmax(const float* in, int C, int H, int W, float* out, int* argmax_out);
void maxpool2x2_backward_with_argmax(const float* grad_out, int C, int H, int W, const int* argmax_out, float* grad_in);

#endif