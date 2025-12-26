#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>

class DataLoader {
private:
    std::string data_dir;
    int batch_size;
    int current_batch;
    int total_batches;
    std::vector<float> data;
    std::vector<int> labels;
    int num_samples;
    const int IMAGE_SIZE = 32 * 32 * 3;  // CIFAR-10: 32x32x3
    
public:
    DataLoader(const std::string& dir, int bs, bool is_test = false) 
        : data_dir(dir), batch_size(bs), current_batch(0) {
        load_data(is_test);
        total_batches = (num_samples + batch_size - 1) / batch_size;
    }
    
    void load_data(bool is_test = false) {
        // Load CIFAR-10 binary data
        if (is_test) {
            // Load test data: test_batch.bin
            std::string filename = data_dir + "test_batch.bin";
            std::ifstream file(filename, std::ios::binary);
            
            if (!file.is_open()) {
                std::cerr << "[ERROR] Could not open " << filename << std::endl;
                return;
            }
            
            // Test batch has 10000 images
            const int samples_per_file = 10000;
            const int record_size = 1 + 32 * 32 * 3;  // 1 label + 3072 pixels
            
            for (int i = 0; i < samples_per_file; i++) {
                unsigned char record[record_size];
                file.read(reinterpret_cast<char*>(record), record_size);
                
                if (!file) break;
                
                // Store label
                labels.push_back(static_cast<int>(record[0]));
                
                // Store image data (normalize to [0, 1])
                for (int j = 1; j < record_size; j++) {
                    data.push_back(static_cast<float>(record[j]) / 255.0f);
                }
            }
            
            file.close();
        } else {
            // Load training data: data_batch_1.bin to data_batch_5.bin
            for (int file_idx = 1; file_idx <= 5; file_idx++) {
                std::string filename = data_dir + "data_batch_" + std::to_string(file_idx) + ".bin";
                std::ifstream file(filename, std::ios::binary);
                
                if (!file.is_open()) {
                    std::cerr << "[WARNING] Could not open " << filename << std::endl;
                    continue;
                }
                
                // Each CIFAR-10 batch has 10000 images
                // Format: [label (1 byte)][red (1024 bytes)][green (1024 bytes)][blue (1024 bytes)]
                const int samples_per_file = 10000;
                const int record_size = 1 + 32 * 32 * 3;  // 1 label + 3072 pixels
                
                for (int i = 0; i < samples_per_file; i++) {
                    unsigned char record[record_size];
                    file.read(reinterpret_cast<char*>(record), record_size);
                    
                    if (!file) break;
                    
                    // Store label
                    labels.push_back(static_cast<int>(record[0]));
                    
                    // Store image data (normalize to [0, 1])
                    for (int j = 1; j < record_size; j++) {
                        data.push_back(static_cast<float>(record[j]) / 255.0f);
                    }
                }
                
                file.close();
            }
        }
        
        num_samples = labels.size();
        std::string dataset_type = is_test ? "test" : "training";
        std::cout << "[INFO] Loaded " << num_samples << " " << dataset_type << " samples from " << data_dir << std::endl;
    }
    
    bool has_next() const {
        return current_batch < total_batches;
    }
    
    float* next_batch() {
        if (!has_next()) {
            return nullptr;
        }
        
        int start_idx = current_batch * batch_size;
        
        // Return pointer to the batch data
        float* batch_ptr = &data[start_idx * IMAGE_SIZE];
        current_batch++;
        
        return batch_ptr;
    }
    
    void reset() {
        current_batch = 0;
    }
    
    int get_num_batches() const {
        return total_batches;
    }
    
    int get_batch_size() const {
        return batch_size;
    }
    
    int get_total_images() const {
        return num_samples;
    }
    
    // Get labels for the current batch
    int* get_batch_labels() {
        if (current_batch == 0) return nullptr;
        int start_idx = (current_batch - 1) * batch_size;
        return \u0026labels[start_idx];
    }
};

#endif // DATA_LOADER_H
