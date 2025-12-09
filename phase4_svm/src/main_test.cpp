#include "../include/svm_wrapper.h"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

int main(int argc, char **argv) {
    if (argc < 4) {
        cout << "Usage: ./test_svm <svm_model.model> <test_features.bin> <output_result.csv>\n";
        return 0;
    }

    string model_file = argv[1];
    string test_file = argv[2];
    string csv_file = argv[3];

    // 1. Load Model
    cout << "--- Loading Model ---\n";
    svm_model *model = svm_load_model(model_file.c_str());
    if (!model) {
        cerr << "Failed to load SVM model from " << model_file << endl;
        return 1;
    }

    // 2. Load Test Data
    cout << "--- Loading Test Data ---\n";
    vector<Sample> test_data = load_bin_features(test_file);

    // 3. Prepare CSV Output
    ofstream csv(csv_file);
    if (!csv.is_open()) {
        cerr << "Cannot open " << csv_file << " for writing.\n";
        return 1;
    }
    // Ghi Header CSV
    csv << "id,true_label,predicted_label,is_correct\n";

    // 4. Predict Loop
    cout << "--- Predicting & Exporting CSV ---\n";
    int correct_count = 0;
    int total_count = test_data.size();

    // Node tạm để dự đoán
    svm_node* x_node = (svm_node*) malloc((FEAT_DIM + 1) * sizeof(svm_node));

    for (int i = 0; i < total_count; ++i) {
        // Convert single sample to svm_node
        for (int d = 0; d < FEAT_DIM; d++) {
            x_node[d].index = d + 1;
            x_node[d].value = test_data[i].feat[d];
        }
        x_node[FEAT_DIM].index = -1;

        // Predict
        double pred_val = svm_predict(model, x_node);
        int pred_label = (int)pred_val;
        int true_label = (int)test_data[i].label;

        // Check correct
        bool is_correct = (pred_label == true_label);
        if (is_correct) correct_count++;

        // Write to CSV
        csv << i << "," << true_label << "," << pred_label << "," << (is_correct ? 1 : 0) << "\n";
    }

    // 5. Summary
    double accuracy = 100.0 * correct_count / total_count;
    cout << "------------------------------------------\n";
    cout << "Results saved to: " << csv_file << endl;
    cout << "Correct: " << correct_count << "/" << total_count << endl;
    cout << "Accuracy: " << fixed << setprecision(2) << accuracy << "%\n";
    cout << "------------------------------------------\n";

    // Cleanup
    free(x_node);
    svm_free_and_destroy_model(&model);
    csv.close();

    return 0;
}