#include "autoencoder.h"
#include "layers/relu.h"
#include "layers/maxpool.h"
#include "layers/upsample.h"
#include "layers/mse.h"
#include <chrono> // [FIX] Thêm thư viện này cho std::chrono

#ifdef _OPENMP
#include <omp.h>
#endif

// Activations Implementation
// Sửa lại thứ tự trong src/autoencoder.cpp
AutoEncoder::Activations::Activations() : 
    feat1(NULL), feat1_relu(NULL), pool1_argmax(NULL), pool1(NULL),
    feat2(NULL), feat2_relu(NULL), pool2_argmax(NULL), latent(NULL),
    feat3(NULL), feat3_relu(NULL), up1(NULL),
    feat4(NULL), feat4_relu(NULL), up2(NULL) 
{}

void AutoEncoder::Activations::alloc(int N) {
    // Sizes
    int h1=IMG_H, w1=IMG_W; 
    int h2=h1/2, w2=w1/2; // after pool1
    int h3=h2/2, w3=w2/2; // after pool2 (latent)

    feat1 = (float*) xmalloc(sizeof(float) * N * F1 * h1 * w1);
    feat1_relu = (float*) xmalloc(sizeof(float) * N * F1 * h1 * w1);
    pool1 = (float*) xmalloc(sizeof(float) * N * F1 * h2 * w2);
    pool1_argmax = (int*) xmalloc(sizeof(int) * N * F1 * h2 * w2);
    
    feat2 = (float*) xmalloc(sizeof(float) * N * F2 * h2 * w2);
    feat2_relu = (float*) xmalloc(sizeof(float) * N * F2 * h2 * w2);
    latent = (float*) xmalloc(sizeof(float) * N * F2 * h3 * w3);
    pool2_argmax = (int*) xmalloc(sizeof(int) * N * F2 * h3 * w3);
    
    feat3 = (float*) xmalloc(sizeof(float) * N * F2 * h3 * w3);
    feat3_relu = (float*) xmalloc(sizeof(float) * N * F2 * h3 * w3);
    up1 = (float*) xmalloc(sizeof(float) * N * F2 * h2 * w2);
    
    feat4 = (float*) xmalloc(sizeof(float) * N * F1 * h2 * w2);
    feat4_relu = (float*) xmalloc(sizeof(float) * N * F1 * h2 * w2);
    up2 = (float*) xmalloc(sizeof(float) * N * F1 * h1 * w1);
}

void AutoEncoder::Activations::free_all() {
    free(feat1); free(pool1); free(pool1_argmax);
    free(feat2); free(latent); free(pool2_argmax);
    free(feat3); free(up1); free(feat4); free(up2);
    free(feat1_relu); free(feat2_relu); free(feat3_relu); free(feat4_relu);
}

// AutoEncoder Implementation
AutoEncoder::AutoEncoder() {}

void AutoEncoder::init() {
    conv1.init(IMG_C, F1, 3, 1, 1, IMG_H, IMG_W);
    conv2.init(F1, F2, 3, 1, 1, IMG_H/2, IMG_W/2);
    conv3.init(F2, F2, 3, 1, 1, IMG_H/4, IMG_W/4);
    conv4.init(F2, F1, 3, 1, 1, IMG_H/2, IMG_W/2);
    conv5.init(F1, IMG_C, 3, 1, 1, IMG_H, IMG_W);
}

void AutoEncoder::free_all() {
    conv1.free_all(); conv2.free_all(); conv3.free_all(); conv4.free_all(); conv5.free_all();
}

void AutoEncoder::forward_batch(const float* input_batch, float* output_batch, Activations &act, int N, DetailedTimer* timer) {
    int in_img_size = IMG_C * IMG_H * IMG_W;
    int feat1_size = F1 * IMG_H * IMG_W;
    int pool1_size = F1 * (IMG_H/2) * (IMG_W/2);
    int feat2_size = F2 * (IMG_H/2) * (IMG_W/2);
    int latent_size = F2 * (IMG_H/4) * (IMG_W/4);
    int feat3_size = latent_size;
    int up1_size = F2 * (IMG_H/2) * (IMG_W/2);
    int feat4_size = F1 * (IMG_H/2) * (IMG_W/2);
    int up2_size = F1 * IMG_H * IMG_W;
    int out_img_size = IMG_C * IMG_H * IMG_W;

    for (int i=0;i<N*out_img_size;++i) output_batch[i]=0.0f;

    #pragma omp parallel for
    for (int n=0;n<N;++n) {
        const float* inptr = input_batch + n * in_img_size;
        bool do_timing = (N == 1 && n == 0 && timer != nullptr);
        // --- Conv1 ---
        float* feat1ptr = act.feat1 + n * feat1_size; 
        
        // [FIX] Cấp phát bộ nhớ đệm TRƯỚC khi bấm giờ
        float* col_buf1 = (float*) malloc(sizeof(float) * conv1.in_c * conv1.k * conv1.k * conv1.Hout * conv1.Wout);
        
        auto t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        conv1.forward_one(inptr, feat1ptr, col_buf1);
        if (do_timing) timer->record_duration("Conv1", t_start);
        
        // [FIX] Giải phóng SAU khi bấm giờ xong
        free(col_buf1);

        // --- ReLU1 ---
        // [FIX] Copy dữ liệu TRƯỚC khi bấm giờ
        memcpy(act.feat1_relu + n * feat1_size, feat1ptr, sizeof(float)*feat1_size);
        
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        relu_forward_inplace(feat1ptr, feat1_size);
        if (do_timing) timer->record_duration("ReLU1", t_start);

        // --- MaxPool1 ---
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        maxpool2x2_forward_with_argmax(feat1ptr, F1, IMG_H, IMG_W, act.pool1 + n * pool1_size, act.pool1_argmax + n * pool1_size);
        if (do_timing) timer->record_duration("MaxPool1", t_start);

        // --- Conv2 ---
        float* feat2ptr = act.feat2 + n * feat2_size;
        
        // [FIX] malloc ra ngoài
        float* col_buf2 = (float*) malloc(sizeof(float) * conv2.in_c * conv2.k * conv2.k * conv2.Hout * conv2.Wout);
        
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        conv2.forward_one(act.pool1 + n * pool1_size, feat2ptr, col_buf2);
        if (do_timing) timer->record_duration("Conv2", t_start);
        
        free(col_buf2);

        // --- ReLU2 ---
        // [FIX] memcpy ra ngoài
        memcpy(act.feat2_relu + n * feat2_size, feat2ptr, sizeof(float)*feat2_size);
        
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        relu_forward_inplace(feat2ptr, feat2_size);
        if (do_timing) timer->record_duration("ReLU2", t_start);

        // --- MaxPool2 ---
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        maxpool2x2_forward_with_argmax(feat2ptr, F2, IMG_H/2, IMG_W/2, act.latent + n * latent_size, act.pool2_argmax + n * latent_size);
        if (do_timing) timer->record_duration("MaxPool2 (Latent)", t_start);

        // --- Conv3 (Decoder Conv1) ---
        float* feat3ptr = act.feat3 + n * feat3_size;
        
        float* col_buf3 = (float*) malloc(sizeof(float) * conv3.in_c * conv3.k * conv3.k * conv3.Hout * conv3.Wout);
        
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        conv3.forward_one(act.latent + n * latent_size, feat3ptr, col_buf3);
        if (do_timing) timer->record_duration("DecodeConv1 (Conv3)", t_start);
        
        free(col_buf3);

        // --- ReLU3 ---
        memcpy(act.feat3_relu + n * feat3_size, feat3ptr, sizeof(float)*feat3_size);
        
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        relu_forward_inplace(feat3ptr, feat3_size);
        if (do_timing) timer->record_duration("ReLU_Dec1 (ReLU3)", t_start);
        
        // --- Upsample1 ---
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        upsample2x_forward(feat3ptr, F2, IMG_H/4, IMG_W/4, act.up1 + n * up1_size);
        if (do_timing) timer->record_duration("Upsample1", t_start);

        // --- Conv4 (Decoder Conv2) ---
        float* feat4ptr = act.feat4 + n * feat4_size;
        
        float* col_buf4 = (float*) malloc(sizeof(float) * conv4.in_c * conv4.k * conv4.k * conv4.Hout * conv4.Wout);
        
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        conv4.forward_one(act.up1 + n * up1_size, feat4ptr, col_buf4);
        if (do_timing) timer->record_duration("DecodeConv2 (Conv4)", t_start);
        
        free(col_buf4);

        // --- ReLU4 ---
        memcpy(act.feat4_relu + n * feat4_size, feat4ptr, sizeof(float)*feat4_size);
        
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        relu_forward_inplace(feat4ptr, feat4_size);
        if (do_timing) timer->record_duration("ReLU_Dec2 (ReLU4)", t_start);
        
        // --- Upsample2 ---
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        upsample2x_forward(feat4ptr, F1, IMG_H/2, IMG_W/2, act.up2 + n * up2_size);
        if (do_timing) timer->record_duration("Upsample2", t_start);

        // --- FinalConv (Conv5) ---
        float* outptr = output_batch + n * out_img_size;
        
        float* col_buf5 = (float*) malloc(sizeof(float) * conv5.in_c * conv5.k * conv5.k * conv5.Hout * conv5.Wout);
        
        t_start = do_timing ? timer->start_point() : std::chrono::high_resolution_clock::time_point();
        conv5.forward_one(act.up2 + n * up2_size, outptr, col_buf5);
        if (do_timing) timer->record_duration("FinalConv (Conv5)", t_start);
        
        free(col_buf5);
    }
}

void AutoEncoder::backward_batch(const float* input_batch, const float* output_batch, const float* target_batch, Activations &act, int N) {
    int out_img_size = IMG_C * IMG_H * IMG_W;
    float* grad_out = (float*) xmalloc(sizeof(float) * N * out_img_size);
    
    // MSE Backward
    mse_backward(output_batch, target_batch, grad_out, N, out_img_size);

    conv1.zero_grads(); conv2.zero_grads(); conv3.zero_grads(); conv4.zero_grads(); conv5.zero_grads();

    // Sizes
    int feat1_size = F1 * IMG_H * IMG_W;
    int pool1_size = F1 * (IMG_H/2) * (IMG_W/2);
    int feat2_size = F2 * (IMG_H/2) * (IMG_W/2);
    int latent_size = F2 * (IMG_H/4) * (IMG_W/4);
    int feat3_size = latent_size;
    int up1_size = F2 * (IMG_H/2) * (IMG_W/2);
    int feat4_size = F1 * (IMG_H/2) * (IMG_W/2);
    int up2_size = F1 * IMG_H * IMG_W;
    int in_img_size = IMG_C * IMG_H * IMG_W;

    #pragma omp parallel for
    for (int n=0;n<N;++n) {
        // Temps
        float* grad_up2 = (float*) xmalloc(sizeof(float) * up2_size);
        float* grad_feat4 = (float*) xmalloc(sizeof(float) * feat4_size);
        float* grad_up1 = (float*) xmalloc(sizeof(float) * up1_size);
        float* grad_feat3 = (float*) xmalloc(sizeof(float) * feat3_size);
        float* grad_latent = (float*) xmalloc(sizeof(float) * latent_size);
        float* grad_feat2 = (float*) xmalloc(sizeof(float) * feat2_size);
        float* grad_pool1 = (float*) xmalloc(sizeof(float) * pool1_size);
        float* grad_feat1 = (float*) xmalloc(sizeof(float) * feat1_size);
        float* grad_input = (float*) xmalloc(sizeof(float) * in_img_size);

        // --- Conv5 Backward ---
        float* col_buf5 = (float*) malloc(sizeof(float) * conv5.in_c * conv5.k * conv5.k * conv5.Hout * conv5.Wout);
        conv5.backward_one(grad_out + n*out_img_size, act.up2 + n*up2_size, grad_up2, col_buf5);
        free(col_buf5);

        upsample2x_backward(grad_up2, F1, IMG_H/2, IMG_W/2, grad_feat4);

        // --- Conv4 Backward ---
        float* col_buf4 = (float*) malloc(sizeof(float) * conv4.in_c * conv4.k * conv4.k * conv4.Hout * conv4.Wout);
        conv4.backward_one(grad_feat4, act.up1 + n*up1_size, grad_up1, col_buf4);
        free(col_buf4);

        upsample2x_backward(grad_up1, F2, IMG_H/4, IMG_W/4, grad_feat3);

        // --- Conv3 Backward ---
        float* col_buf3 = (float*) malloc(sizeof(float) * conv3.in_c * conv3.k * conv3.k * conv3.Hout * conv3.Wout);
        conv3.backward_one(grad_feat3, act.latent + n*latent_size, grad_latent, col_buf3);
        free(col_buf3);

        maxpool2x2_backward_with_argmax(grad_latent, F2, IMG_H/2, IMG_W/2, act.pool2_argmax + n*latent_size, grad_feat2);
        relu_backward_inplace(grad_feat2, act.feat2_relu + n*feat2_size, feat2_size);

        // --- Conv2 Backward ---
        float* col_buf2 = (float*) malloc(sizeof(float) * conv2.in_c * conv2.k * conv2.k * conv2.Hout * conv2.Wout);
        conv2.backward_one(grad_feat2, act.pool1 + n*pool1_size, grad_pool1, col_buf2);
        free(col_buf2);

        maxpool2x2_backward_with_argmax(grad_pool1, F1, IMG_H, IMG_W, act.pool1_argmax + n*pool1_size, grad_feat1);
        relu_backward_inplace(grad_feat1, act.feat1_relu + n*feat1_size, feat1_size);

        // --- Conv1 Backward ---
        float* col_buf1 = (float*) malloc(sizeof(float) * conv1.in_c * conv1.k * conv1.k * conv1.Hout * conv1.Wout);
        conv1.backward_one(grad_feat1, input_batch + n*in_img_size, grad_input, col_buf1);
        free(col_buf1);

        free(grad_up2); free(grad_feat4); free(grad_up1); free(grad_feat3); free(grad_latent);
        free(grad_feat2); free(grad_pool1); free(grad_feat1); free(grad_input);
    }
    free(grad_out);
}

void AutoEncoder::sgd_update(float lr, int batch_size) {
    conv1.update(lr, batch_size);
    conv2.update(lr, batch_size);
    conv3.update(lr, batch_size);
    conv4.update(lr, batch_size);
    conv5.update(lr, batch_size);
}

void AutoEncoder::save_weights(const string &fname) {
    FILE* f = fopen(fname.c_str(),"wb");
    if (!f) { cerr<<"Cannot open "<<fname<<" for writing\n"; return; }
    auto write_conv = [&](Conv2D &c) {
        fwrite(&c.in_c, sizeof(int), 1, f);
        fwrite(&c.out_c, sizeof(int), 1, f);
        fwrite(&c.k, sizeof(int), 1, f);
        int wcount = c.out_c * c.in_c * c.k * c.k;
        fwrite(c.W, sizeof(float), wcount, f);
        fwrite(c.b, sizeof(float), c.out_c, f);
    };
    write_conv(conv1); write_conv(conv2); write_conv(conv3); write_conv(conv4); write_conv(conv5);
    fclose(f);
    cout<<"Weights saved to "<<fname<<"\n";
}

void AutoEncoder::load_weights(const string &fname) {
    FILE* f = fopen(fname.c_str(),"rb");
    if (!f) { cerr<<"Cannot open "<<fname<<" for reading\n"; return; }

    auto read_conv = [&](Conv2D &c) {
        int in_c2, out_c2, k2;
        if (fread(&in_c2, sizeof(int), 1, f) != 1) exit(1);
        if (fread(&out_c2, sizeof(int), 1, f) != 1) exit(1);
        if (fread(&k2, sizeof(int), 1, f) != 1) exit(1);

        if (in_c2 != c.in_c || out_c2 != c.out_c || k2 != c.k) {
            cerr<<"Weight mismatch!\n"; exit(1);
        }
        int wcount = c.out_c * c.in_c * c.k * c.k;
        if (fread(c.W, sizeof(float), wcount, f) != (size_t)wcount) exit(1);
        if (fread(c.b, sizeof(float), c.out_c, f) != (size_t)c.out_c) exit(1);
    };

    read_conv(conv1); read_conv(conv2); read_conv(conv3); read_conv(conv4); read_conv(conv5);
    fclose(f);
    cout<<"Weights loaded from "<<fname<<"\n";
}

void AutoEncoder::extract_feature_single(const float* input_img, float* out_latent) {
    // Allocation on fly for inference (similar to original code's extract logic)
    float* col_buf = (float*) malloc(sizeof(float) * F1 * 9 * 32 * 32); // Estimate max size
    
    // Conv1
    vector<float> feat1(F1 * 32 * 32);
    conv1.forward_one(input_img, feat1.data(), col_buf);
    relu_forward_inplace(feat1.data(), feat1.size());
    
    vector<float> pool1(F1 * 16 * 16);
    vector<int> pool1_arg(F1 * 16 * 16);
    maxpool2x2_forward_with_argmax(feat1.data(), F1, IMG_H, IMG_W, pool1.data(), pool1_arg.data());
    
    // Conv2
    vector<float> feat2(F2 * 16 * 16);
    conv2.forward_one(pool1.data(), feat2.data(), col_buf);
    relu_forward_inplace(feat2.data(), feat2.size());
    
    vector<int> pool2_arg(F2 * 8 * 8);
    maxpool2x2_forward_with_argmax(feat2.data(), F2, IMG_H/2, IMG_W/2, out_latent, pool2_arg.data());

    free(col_buf);
}