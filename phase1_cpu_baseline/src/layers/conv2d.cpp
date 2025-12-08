#include "layers/conv2d.h"
#include "utils/weight_init.h"
#include <cmath>

// Implement common math here to avoid circular deps or define in main utility
void im2col(const float* data_im, int C, int H, int W,
            int K, int pad, int stride, float* data_col) {
    int H_out = (H + 2*pad - K) / stride + 1;
    int W_out = (W + 2*pad - K) / stride + 1;
    int col_index = 0;
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        int ih = oh * stride - pad + kh;
                        int iw = ow * stride - pad + kw;
                        float val = 0.0f;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            val = data_im[(c * H + ih) * W + iw];
                        }
                        data_col[col_index++] = val;
                    }
                }
            }
        }
    }
}

void col2im(const float* data_col, int C, int H, int W,
            int K, int pad, int stride, float* data_im) {
    int H_out = (H + 2*pad - K) / stride + 1;
    int W_out = (W + 2*pad - K) / stride + 1;
    int im_size = C * H * W;
    for (int i=0;i<im_size;++i) data_im[i] = 0.0f;
    int col_index = 0;
    for (int c=0;c<C;++c) {
        for (int kh=0; kh<K; ++kh) {
            for (int kw=0; kw<K; ++kw) {
                for (int oh=0; oh<H_out; ++oh) {
                    for (int ow=0; ow<W_out; ++ow) {
                        int ih = oh * stride - pad + kh;
                        int iw = ow * stride - pad + kw;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            data_im[(c * H + ih) * W + iw] += data_col[col_index];
                        }
                        ++col_index;
                    }
                }
            }
        }
    }
}

void gemm(int M, int N, int K,
          const float* A, bool A_t,
          const float* B, bool B_t,
          float* C) {
    for (int m=0; m<M; ++m) {
        for (int n=0; n<N; ++n) {
            float sum = 0.0f;
            for (int k=0; k<K; ++k) {
                float a = A_t ? A[k * M + m] : A[m * K + k];
                float b = B_t ? B[n * K + k] : B[k * N + n];
                sum += a * b;
            }
            C[m * N + n] = sum;
        }
    }
}

Conv2D::Conv2D() : W(NULL), b(NULL), dW(NULL), db(NULL) {}

void Conv2D::init(int in_c_, int out_c_, int k_, int pad_, int stride_, int Hin_, int Win_) {
    in_c = in_c_; out_c = out_c_; k = k_; pad = pad_; stride = stride_;
    Hin = Hin_; Win = Win_;
    Hout = (Hin + 2*pad - k) / stride + 1;
    Wout = (Win + 2*pad - k) / stride + 1;
    size_t wsize = (size_t)out_c * in_c * k * k;
    W = (float*) xmalloc(sizeof(float) * wsize);
    dW = (float*) xmalloc(sizeof(float) * wsize);
    b = (float*) xmalloc(sizeof(float) * out_c);
    db = (float*) xmalloc(sizeof(float) * out_c);

    float fan_in = in_c * k * k;
    float scale = sqrtf(2.0f / fan_in);
    for (size_t i=0;i<wsize;++i) W[i] = frand(-scale, scale);
    for (int i=0;i<out_c;++i) b[i] = 0.0f;
}

void Conv2D::zero_grads() {
    size_t wsize = (size_t)out_c * in_c * k * k;
    for (size_t i=0;i<wsize;++i) dW[i]=0.0f;
    for (int i=0;i<out_c;++i) db[i]=0.0f;
}

void Conv2D::free_all() {
    free(W); free(dW); free(b); free(db);
}

void Conv2D::forward_one(const float* inptr, float* outptr, float* col_buffer) {
    im2col(inptr, in_c, Hin, Win, k, pad, stride, col_buffer);
    
    int M = out_c;
    int K = in_c * k * k;
    int Ncol = Hout * Wout;
    
    // We need a temp buffer for GEMM output before bias
    // To optimize, gemm can write directly, then we add bias.
    // However, output is (M, Ncol).
    float* tmp_out = (float*) xmalloc(sizeof(float) * M * Ncol);
    gemm(M, Ncol, K, W, false, col_buffer, false, tmp_out);
    
    for (int oc=0; oc<M; ++oc)
        for (int j=0;j<Ncol;++j)
            outptr[oc * Ncol + j] = tmp_out[oc * Ncol + j] + b[oc];
            
    free(tmp_out);
}

void Conv2D::backward_one(const float* grad_out, const float* input_val, float* grad_in, float* col_buffer) {
    // Re-compute col for input (needed for dW)
    im2col(input_val, in_c, Hin, Win, k, pad, stride, col_buffer);

    int M = out_c; 
    int K = in_c * k * k; 
    int Ncol = Hout * Wout;

    // db
    for (int oc=0; oc<M; ++oc) {
        float s=0.0f;
        for (int j=0;j<Ncol;++j) s += grad_out[oc * Ncol + j];
        #pragma omp atomic
        db[oc] += s;
    }
    // dW
    for (int oc=0; oc<M; ++oc) {
        for (int kidx=0; kidx<K; ++kidx) {
            float s=0.0f;
            for (int j=0;j<Ncol;++j) s += grad_out[oc * Ncol + j] * col_buffer[kidx * Ncol + j];
            #pragma omp atomic
            dW[oc * K + kidx] += s;
        }
    }
    
    // grad_in (dx)
    if (grad_in) {
        float* temp = (float*) xmalloc(sizeof(float) * K * Ncol);
        for (int kidx=0; kidx<K; ++kidx) {
            for (int j=0; j<Ncol; ++j) {
                float s=0;
                for (int oc=0; oc<M; ++oc) s += W[oc * K + kidx] * grad_out[oc * Ncol + j];
                temp[kidx * Ncol + j] = s;
            }
        }
        col2im(temp, in_c, Hin, Win, k, pad, stride, grad_in);
        free(temp);
    }
}

void Conv2D::update(float lr, int batch_size) {
    int K = in_c * k * k;
    size_t wsize = (size_t)out_c * K;
    float scale = 1.0f / batch_size;
    for (size_t i=0;i<wsize;++i) W[i] -= lr * dW[i] * scale;
    for (int i=0;i<out_c;++i) b[i] -= lr * db[i] * scale;
}