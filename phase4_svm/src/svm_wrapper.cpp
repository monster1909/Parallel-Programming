#include "../include/svm_wrapper.h"
#include <cstdio>
#include <cstdlib>

using namespace std;

vector<Sample> load_bin_features(const string& file) {
    vector<Sample> data;
    FILE *f = fopen(file.c_str(), "rb");
    if (!f) {
        cerr << "Error: Cannot open " << file << endl;
        exit(1);
    }

    // 1. Đọc Header
    int N_total, dim_feat;
    if (fread(&N_total, sizeof(int), 1, f) != 1) { cerr << "Read header N failed\n"; exit(1); }
    if (fread(&dim_feat, sizeof(int), 1, f) != 1) { cerr << "Read header Dim failed\n"; exit(1); }

    if (dim_feat != FEAT_DIM) {
        cerr << "Dimension mismatch! File: " << dim_feat << ", Code: " << FEAT_DIM << endl;
        exit(1);
    }
    cout << "[Loader] Header: N=" << N_total << ", Dim=" << dim_feat << " (" << file << ")" << endl;

    // 2. Đọc Data
    data.reserve(N_total);
    while (true) {
        unsigned char label;
        size_t r = fread(&label, 1, 1, f);
        if (r != 1) break; // Hết file

        Sample s;
        s.label = label;
        s.feat.resize(FEAT_DIM);

        r = fread(s.feat.data(), sizeof(float), FEAT_DIM, f);
        if (r != FEAT_DIM) {
            cerr << "Warning: Incomplete sample at end of file." << endl;
            break;
        }
        data.push_back(s);
    }
    fclose(f);
    
    if (data.size() != (size_t)N_total) {
        cerr << "Warning: Header said " << N_total << " but read " << data.size() << endl;
    }
    return data;
}

svm_problem build_svm_problem(const vector<Sample>& data) {
    svm_problem prob;
    int N = data.size();
    prob.l = N;
    prob.y = (double*) malloc(N * sizeof(double));
    prob.x = (svm_node**) malloc(N * sizeof(svm_node*));

    for (int i = 0; i < N; i++) {
        prob.y[i] = (double)data[i].label;
        
        // Cấp phát node cho sample thứ i
        svm_node *nodes = (svm_node*) malloc((FEAT_DIM + 1) * sizeof(svm_node));
        for (int d = 0; d < FEAT_DIM; d++) {
            nodes[d].index = d + 1; // LIBSVM index bắt đầu từ 1
            nodes[d].value = data[i].feat[d];
        }
        nodes[FEAT_DIM].index = -1; // Kết thúc sample
        prob.x[i] = nodes;
    }
    return prob;
}

void free_svm_problem(svm_problem& prob) {
    if (prob.y) free(prob.y);
    if (prob.x) {
        for (int i = 0; i < prob.l; i++) {
            free(prob.x[i]);
        }
        free(prob.x);
    }
}