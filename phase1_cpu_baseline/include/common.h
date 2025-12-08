#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

using namespace std;

// Memory helper
inline void* xmalloc(size_t s) {
    void* p = malloc(s);
    if (!p) { cerr << "malloc failed\n"; exit(1); }
    memset(p, 0, s);
    return p;
}

// Math helpers definitions
void im2col(const float* data_im, int C, int H, int W,
            int K, int pad, int stride, float* data_col);

void col2im(const float* data_col, int C, int H, int W,
            int K, int pad, int stride, float* data_im);

void gemm(int M, int N, int K,
          const float* A, bool A_t,
          const float* B, bool B_t,
          float* C);

#endif