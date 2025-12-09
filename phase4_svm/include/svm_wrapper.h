#ifndef SVM_WRAPPER_H
#define SVM_WRAPPER_H

#include <vector>
#include <string>
#include <iostream>
#include "../src/libsvm/svm.h" // Trỏ tới header của libsvm

// Cấu trúc dữ liệu cho một mẫu ảnh
struct Sample {
    unsigned char label;
    std::vector<float> feat;
};

// Hằng số kích thước đặc trưng
const int FEAT_DIM = 8192;

// Hàm load file binary (đã fix header check)
std::vector<Sample> load_bin_features(const std::string& file);

// Hàm chuyển đổi vector Sample sang định dạng của LIBSVM
svm_problem build_svm_problem(const std::vector<Sample>& data);

// Hàm giải phóng bộ nhớ của svm_problem
void free_svm_problem(svm_problem& prob);

#endif