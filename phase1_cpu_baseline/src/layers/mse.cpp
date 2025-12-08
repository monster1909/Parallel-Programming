#include "layers/mse.h"

void mse_backward(const float* output_batch, const float* target_batch, float* grad_out, int N, int size_per_sample) {
    int total_elems = N * size_per_sample;
    for (int i=0;i<total_elems;++i) {
        grad_out[i] = 2.0f * (output_batch[i] - target_batch[i]) / float(total_elems);
    }
}

double compute_mse_loss(const float* output, const float* target, int total_elements) {
    double loss = 0;
    for (int i=0; i<total_elements; i++) {
        double d = output[i] - target[i];
        loss += d*d;
    }
    return loss / total_elements;
}