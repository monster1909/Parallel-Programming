#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>

using namespace std;

class DataLoader {
public:
    DataLoader(const string& data_dir, int batch_size, bool shuffle = true, bool is_test = false);
    ~DataLoader();
    
    // Batch iteration
    bool has_next();
    float* next_batch();      // Returns pointer to batch on GPU
    int* next_labels();       // Returns pointer to labels on GPU
    unsigned char* get_batch_labels();  // Returns pointer to current batch labels on CPU
    void reset();             // Reset for next epoch
    void shuffle_data();      // Shuffle dataset
    
    // Info
    int get_batch_size() const { return batch_size; }
    int get_num_batches() const { return num_batches; }
    int get_total_images() const { return total_images; }
    
private:
    void load_cifar10_files();
    void load_batch_to_gpu(int batch_idx);
    
    string data_dir;
    int batch_size;
    int current_batch;
    int num_batches;
    int total_images;
    bool shuffle;
    bool is_test;  // true = load test_batch.bin, false = load data_batch_*.bin
    
    // All data in CPU memory
    vector<float> all_images;  // [total_images, 3, 32, 32]
    vector<int> all_labels;    // [total_images]
    vector<int> indices;       // For shuffling
    
    // Current batch on GPU
    float* d_batch_images;
    int* d_batch_labels;
    
    // Current batch labels on CPU (for feature extraction)
    vector<unsigned char> batch_labels_cpu;
};

#endif // DATA_LOADER_H
