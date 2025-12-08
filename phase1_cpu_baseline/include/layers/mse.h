#ifndef MSE_H
#define MSE_H

void mse_backward(const float* output_batch, const float* target_batch, float* grad_out, int N, int size_per_sample);
double compute_mse_loss(const float* output, const float* target, int total_elements);

#endif