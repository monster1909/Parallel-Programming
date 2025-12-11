#include "../include/data_loader.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

using namespace std;

DataLoader::DataLoader(const string& data_dir, int batch_size, bool shuffle, bool is_test)
    : data_dir(data_dir), batch_size(batch_size), shuffle(shuffle), is_test(is_test),
      current_batch(0), d_batch_images(nullptr), d_batch_labels(nullptr)
{
    if (is_test) {
        cout << "[DataLoader] Loading CIFAR-10 TEST dataset from: " << data_dir << endl;
    } else {
        cout << "[DataLoader] Loading CIFAR-10 TRAIN dataset from: " << data_dir << endl;
    }
    
    load_cifar10_files();
    
    num_batches = (total_images + batch_size - 1) / batch_size;
    
    // Initialize indices for shuffling
    indices.resize(total_images);
    for (int i = 0; i < total_images; i++) {
        indices[i] = i;
    }
    
    if (shuffle) {
        shuffle_data();
    }
    
    // Allocate GPU memory for one batch
    cudaMalloc(&d_batch_images, batch_size * 3 * 32 * 32 * sizeof(float));
    cudaMalloc(&d_batch_labels, batch_size * sizeof(int));
    
    cout << "[DataLoader] Loaded " << total_images << " images" << endl;
    cout << "[DataLoader] Batch size: " << batch_size << ", Num batches: " << num_batches << endl;
}

DataLoader::~DataLoader() {
    if (d_batch_images) cudaFree(d_batch_images);
    if (d_batch_labels) cudaFree(d_batch_labels);
}

void DataLoader::load_cifar10_files() {
    vector<string> batch_files;
    
    if (is_test) {
        // Test data: 1 file with 10,000 images
        batch_files = {"test_batch.bin"};
    } else {
        // Training data: 5 files with 50,000 images
        batch_files = {
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin"
        };
    }
    
    const int images_per_file = 10000;
    const int image_size = 3 * 32 * 32;
    const int record_size = 1 + image_size;  // 1 byte label + 3072 bytes image
    
    total_images = 0;
    
    cout << "[DataLoader] Loading " << batch_files.size() << " batch file(s)..." << endl;
    
    int file_idx = 0;
    for (const auto& filename : batch_files) {
        string filepath = data_dir + "/" + filename;
        ifstream file(filepath, ios::binary);
        
        if (!file.is_open()) {
            cerr << "[DataLoader] Warning: Could not open " << filepath << endl;
            ++file_idx;
            continue;
        }
        
        for (int i = 0; i < images_per_file; i++) {
            // Read label
            unsigned char label_byte;
            file.read(reinterpret_cast<char*>(&label_byte), 1);
            all_labels.push_back(static_cast<int>(label_byte));
            
            // Read image (3072 bytes: R, G, B channels)
            vector<unsigned char> buffer(image_size);
            file.read(reinterpret_cast<char*>(buffer.data()), image_size);
            
            // Convert to float and normalize [0, 1]
            // CIFAR-10 format: [R channel][G channel][B channel]
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < 32; h++) {
                    for (int w = 0; w < 32; w++) {
                        int src_idx = c * 32 * 32 + h * 32 + w;
                        float pixel = buffer[src_idx] / 255.0f;
                        all_images.push_back(pixel);
                    }
                }
            }
            
            total_images++;
        }
        
        file.close();
        ++file_idx;
    }
    
    cout << "[DataLoader] Loaded " << total_images << " images" << endl;
}

void DataLoader::shuffle_data() {
    random_device rd;
    mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
}

bool DataLoader::has_next() {
    return current_batch < num_batches;
}

float* DataLoader::next_batch() {
    if (!has_next()) {
        return nullptr;
    }
    
    load_batch_to_gpu(current_batch);
    current_batch++;
    
    return d_batch_images;
}

int* DataLoader::next_labels() {
    return d_batch_labels;
}

void DataLoader::load_batch_to_gpu(int batch_idx) {
    const int image_size = 3 * 32 * 32;
    int start_idx = batch_idx * batch_size;
    int end_idx = min(start_idx + batch_size, total_images);
    int actual_batch_size = end_idx - start_idx;
    
    // Prepare batch on CPU
    vector<float> batch_images(batch_size * image_size, 0.0f);
    vector<int> batch_labels(batch_size, 0);
    
    for (int i = 0; i < actual_batch_size; i++) {
        int img_idx = indices[start_idx + i];
        
        // Copy image
        for (int j = 0; j < image_size; j++) {
            batch_images[i * image_size + j] = all_images[img_idx * image_size + j];
        }
        
        // Copy label
        batch_labels[i] = all_labels[img_idx];
    }
    
    // Also fill CPU labels for feature extraction
    batch_labels_cpu.resize(actual_batch_size);
    for (int i = 0; i < actual_batch_size; i++) {
        batch_labels_cpu[i] = static_cast<unsigned char>(batch_labels[i]);
    }
    
    // Copy to GPU
    cudaMemcpy(d_batch_images, batch_images.data(), 
               batch_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch_labels, batch_labels.data(), 
               batch_size * sizeof(int), cudaMemcpyHostToDevice);
}

void DataLoader::reset() {
    current_batch = 0;
    if (shuffle) {
        shuffle_data();
    }
}

unsigned char* DataLoader::get_batch_labels() {
    return batch_labels_cpu.data();
}

