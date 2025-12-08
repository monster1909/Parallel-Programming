#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <string>
#include "common.h"
#include "config.h"
#include "layers/conv2d.h"

struct AutoEncoder {
    Conv2D conv1, conv2, conv3, conv4, conv5;

    struct Activations {
        float* feat1; 
        float* feat1_relu; 
        int* pool1_argmax;
        float* pool1; 

        float* feat2; 
        float* feat2_relu;
        int* pool2_argmax; 
        float* latent; 

        float* feat3; 
        float* feat3_relu;
        float* up1; 

        float* feat4; 
        float* feat4_relu;
        float* up2; 

        Activations();
        void alloc(int N);
        void free_all();
    };

    AutoEncoder();
    void init();
    void free_all();

    void forward_batch(const float* input_batch, float* output_batch, Activations &act, int N);
    void backward_batch(const float* input_batch, const float* output_batch, const float* target_batch, Activations &act, int N);
    void sgd_update(float lr, int batch_size);
    
    void save_weights(const std::string &fname);
    void load_weights(const std::string &fname);
    void extract_feature_single(const float* input_img, float* out_latent);
};

#endif