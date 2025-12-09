#include "../include/svm_wrapper.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "Usage: ./train_svm <train_features.bin> <output_model.model>\n";
        return 0;
    }

    string train_file = argv[1];
    string model_file = argv[2];

    // 1. Load Data
    cout << "--- Loading Training Data ---\n";
    vector<Sample> train_data = load_bin_features(train_file);

    // 2. Convert to LIBSVM format
    svm_problem prob = build_svm_problem(train_data);

    // 3. Setup SVM Parameters
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.gamma = 1.0 / FEAT_DIM; // Auto gamma
    param.C = 10;                 // Penalty
    param.cache_size = 1000;      // Cache 1GB (tăng nếu máy mạnh)
    param.eps = 1e-3;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.shrinking = 1;
    param.probability = 0;
    param.degree = 3;
    param.coef0 = 0;
    param.nu = 0.5;
    param.p = 0.1;

    // Check parameters
    const char *err = svm_check_parameter(&prob, &param);
    if (err) {
        cerr << "SVM Parameter Error: " << err << endl;
        return 1;
    }

    // 4. Train
    cout << "--- Training SVM (this may take time) ---\n";
    svm_model* model = svm_train(&prob, &param);

    // 5. Save Model
    if (svm_save_model(model_file.c_str(), model) == 0) {
        cout << "Successfully saved model to: " << model_file << endl;
    } else {
        cerr << "Error saving model!" << endl;
    }

    // 6. Cleanup
    svm_free_and_destroy_model(&model);
    free_svm_problem(prob);

    return 0;
}