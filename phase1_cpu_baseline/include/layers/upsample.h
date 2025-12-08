#ifndef UPSAMPLE_H
#define UPSAMPLE_H

void upsample2x_forward(const float* in, int C, int H, int W, float* out);
void upsample2x_backward(const float* grad_out, int C, int H, int W, float* grad_in);

#endif