#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip> // For formatting output
#include <algorithm> // For std::min, std::max
#include <chrono> // For extract timing
#include "autoencoder.h"
#include "utils/timer.h"
#include "layers/mse.h"

using namespace std;

// PPM Saver
void save_ppm(const float* data, const string& filename) {
    ofstream f(filename);
    if (!f) return;
    f << "P3\n" << IMG_W << " " << IMG_H << "\n255\n";
    for (int h = 0; h < IMG_H; ++h) {
        for (int w = 0; w < IMG_W; ++w) {
            int r = (int)(data[(0 * IMG_H + h) * IMG_W + w] * 255.0f);
            int g = (int)(data[(1 * IMG_H + h) * IMG_W + w] * 255.0f);
            int b = (int)(data[(2 * IMG_H + h) * IMG_W + w] * 255.0f);
            r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b));
            f << r << " " << g << " " << b << " ";
        }
        f << "\n";
    }
    f.close();
    cout << "Saved image to " << filename << endl;
}

// Data Loader (legacy, only returns data)
vector<float> load_cifar_batches(const vector<string>& files, int &N_out) {
    vector<float> data;
    for (auto &fpath : files) {
        ifstream fin(fpath, ios::binary);
        if (!fin) { cerr<<"Cannot open "<<fpath<<endl; exit(1); }
        const int record = 1 + 3072;
        vector<unsigned char> buf(record);
        int count = 0;
        while (fin.read((char*)buf.data(), record)) {
            data.resize(data.size() + IMG_C*IMG_H*IMG_W);
            int base = (int)data.size() - IMG_C*IMG_H*IMG_W;
            for (int c=0;c<IMG_C;++c) {
                for (int h=0; h<IMG_H; ++h) {
                    for (int w=0; w<IMG_W; ++w) {
                        int v = buf[1 + c*1024 + h*32 + w];
                        data[base + (c*IMG_H + h) * IMG_W + w] = v / 255.0f;
                    }
                }
            }
            ++count;
        }
        cout<<"Loaded "<<count<<" images from "<<fpath<<"\n";
    }
    N_out = (int)(data.size() / (IMG_C*IMG_H*IMG_W));
    return data;
}

// [NEW] Hàm load dữ liệu trả về cả Label để dùng cho SVM/Extract
void load_cifar_batches_with_labels(const vector<string>& files, 
                                    vector<float>& data_out, 
                                    vector<unsigned char>& labels_out, 
                                    int &N_out) {
    data_out.clear();
    labels_out.clear();
    
    for (auto &fpath : files) {
        ifstream fin(fpath, ios::binary);
        if (!fin) { cerr<<"Cannot open "<<fpath<<endl; exit(1); }
        
        const int record_size = 1 + 3072; // 1 byte label + 3072 bytes pixel
        vector<unsigned char> buf(record_size);
        int count = 0;
        
        while (fin.read((char*)buf.data(), record_size)) {
            // Lấy label (byte đầu tiên)
            labels_out.push_back(buf[0]);
            
            // Lấy ảnh
            int current_size = data_out.size();
            data_out.resize(current_size + IMG_C*IMG_H*IMG_W);
            int base = current_size;
            
            // CIFAR format: Label | R... | G... | B...
            for (int c=0; c<IMG_C; ++c) {
                for (int h=0; h<IMG_H; ++h) {
                    for (int w=0; w<IMG_W; ++w) {
                        int v = buf[1 + c*1024 + h*32 + w];
                        data_out[base + (c*IMG_H + h) * IMG_W + w] = v / 255.0f;
                    }
                }
            }
            ++count;
        }
        cout << "Loaded " << count << " images with labels from " << fpath << "\n";
    }
    N_out = (int)labels_out.size();
    cout << "Total labeled images: " << N_out << "\n";
}

// Helper to print sample values
void print_sample_values(const float* data, int size, const std::string& header) {
    std::cout << "\n===== " << header << " (first 10 pixels) =====" << std::endl;
    for (int i = 0; i < std::min(10, size); ++i) {
        std::cout << std::fixed << std::setprecision(6) << data[i] << " ";
    }
    std::cout << "\n" << std::endl;
}


int main(int argc, char** argv) {
    srand(1234);

    if (argc < 2) {
        cout << "Usage:\n  Train: " << argv[0] << " train data_batch_1.bin ... weights.bin\n";
        cout << "  Test : " << argv[0] << " test weights.bin input.ppm output.ppm\n";
        cout << "  Extract: " << argv[0] << " extract weights.bin output.bin data_batch_1.bin ...\n"; // Added Extract usage
        return 0;
    }
    string mode = argv[1];
    AutoEncoder ae;
    ae.init();

    if (mode == "train") {
        if (argc < 4) { cerr << "Args error\n"; return 1; }
        vector<string> files;
        for (int i = 2; i < argc - 1; ++i) files.push_back(argv[i]);
        string weights_out = argv[argc - 1];

        int N;
        vector<float> dataset = load_cifar_batches(files, N);
        if (N < BATCH_SIZE) { cerr << "Not enough data\n"; return 1; }

        int in_size = IMG_C * IMG_H * IMG_W;
        float* input_batch = (float*) xmalloc(sizeof(float) * BATCH_SIZE * in_size);
        float* output_batch = (float*) xmalloc(sizeof(float) * BATCH_SIZE * in_size);

        AutoEncoder::Activations act;
        act.alloc(BATCH_SIZE);

        Timer total_train_timer;
        int steps = N / BATCH_SIZE;
        
        // Detailed train start log
        cout << "\n========================================" << endl;
        cout << "[INFO] Training Start" << endl;
        cout << "[INFO] Data Size: " << N << " images" << endl;
        cout << "[INFO] Batch Size: " << BATCH_SIZE << endl;
        cout << "[INFO] Steps per Epoch: " << steps << endl;
        cout << "========================================\n" << endl;

        for (int epoch=0; epoch<EPOCHS; ++epoch) {
            Timer t;
            double epoch_loss = 0.0;
            for (int step=0; step<steps; ++step) {
                int offset = (step * BATCH_SIZE) % N;
                memcpy(input_batch, dataset.data() + offset * in_size, sizeof(float) * BATCH_SIZE * in_size);

                ae.forward_batch(input_batch, output_batch, act, BATCH_SIZE);

                double loss = compute_mse_loss(output_batch, input_batch, BATCH_SIZE * in_size);
                epoch_loss += loss;

                ae.backward_batch(input_batch, output_batch, input_batch, act, BATCH_SIZE);
                ae.sgd_update(LR, BATCH_SIZE);

                // Re-added step log detail
                if (step % 10 == 0) cout << "Epoch " << epoch+1 << " Step " << step << "/" << steps << " Loss=" << fixed << setprecision(8) << loss << "\n";
            }
            // Re-added epoch end log detail
            cout << "--------------------------------------------------------" << endl;
            cout << "[LOG] Epoch " << epoch+1 << " avg loss=" << fixed << setprecision(8) << (epoch_loss/steps) << " time=" << setprecision(3) << t.get_elapsed_seconds() << "s (" << (t.get_elapsed_seconds() / steps * 1000.0) << " ms/step)\n";
            cout << "--------------------------------------------------------" << endl;
        }
        
        // Re-added total train end log
        cout << "\n========================================" << endl;
        cout << "[INFO] Training Finished" << endl;
        cout << "[INFO] Total Training Time: " << total_train_timer.get_elapsed_seconds() << " seconds" << endl;
        cout << "========================================" << endl;
        
        ae.save_weights(weights_out);
        act.free_all();
        free(input_batch);
        free(output_batch);
        ae.free_all();
    } 
    else if (mode == "test") {
        if (argc != 5) { cerr << "Args error\n"; return 1; }
        string weights_file = argv[2];
        string ppm_in = argv[3];
        string ppm_out = argv[4];

        cout << "\n===== Phase: CPU Autoencoder Test (DETAILED TIMING) =====" << endl;
        cout << "[INFO] Loading weights from: " << weights_file << endl;
        ae.load_weights(weights_file);

        vector<float> img(IMG_C * IMG_H * IMG_W);
        ifstream f(ppm_in);
        if (!f) { cerr << "Cannot open ppm\n"; return 1; }
        
        string magic; int w,h,maxv;
        if (!(f >> magic >> w >> h >> maxv) || magic != "P3" || w != IMG_W || h != IMG_H) {
            cerr << "[ERROR] Invalid PPM format or wrong size! Expected 32x32 P3.\n";
            return 1;
        }

        for (int y=0; y<IMG_H; y++) {
            for (int x=0; x<IMG_W; x++) {
                int r,g,b; 
                if (!(f>>r>>g>>b)) { cerr << "[ERROR] Error reading RGB values.\n"; return 1; }
                img[(0*IMG_H+y)*IMG_W+x] = r/255.0f;
                img[(1*IMG_H+y)*IMG_W+x] = g/255.0f;
                img[(2*IMG_H+y)*IMG_W+x] = b/255.0f;
            }
        }
        f.close();
        cout << "[INFO] Image loaded successfully from: " << ppm_in << endl;

        print_sample_values(img.data(), IMG_C * IMG_H * IMG_W, "INPUT SAMPLE VALUES");
        
        // Detailed layer timing setup (Re-added)
        DetailedTimer d_timer; 
        float out_img[IMG_C * IMG_H * IMG_W];
        AutoEncoder::Activations act;
        act.alloc(1);

        // Run forward (batch=1) with detailed timing
        cout << "[INFO] Starting forward pass (with timing)..." << endl;
        ae.forward_batch(img.data(), out_img, act, 1, &d_timer);
        
        // Print timing report
        d_timer.print_timing_report();

        print_sample_values(out_img, IMG_C * IMG_H * IMG_W, "OUTPUT SAMPLE VALUES");

        act.free_all();
        save_ppm(out_img, ppm_out);
        cout << "\n===== DONE =====" << endl;
    }
    else if (mode == "extract") {
        if (argc < 5) {
            cerr << "Extract mode requires: extract weights.bin output.bin data_batch_1.bin ...\n";
            return 1;
        }

        string weights_file = argv[2];
        string output_file  = argv[3];

        // 1. Load trained weights
        cout << "\n========================================" << endl;
        cout << "[INFO] Starting Feature Extraction (Encoder)" << endl;
        cout << "[INFO] Loading weights from: " << weights_file << endl;
        ae.load_weights(weights_file);

        // 2. Load Data & Labels
        vector<string> data_files;
        for (int i = 4; i < argc; ++i) data_files.push_back(argv[i]);

        vector<float> dataset;
        vector<unsigned char> labels;
        int N;
        load_cifar_batches_with_labels(data_files, dataset, labels, N);

        // 3. Prepare Output File
        // Format: [int TotalImages] [int FeatureDim] 
        //         Loop N: [uint8 Label] [float FeatureVector[8192]]
        FILE* fout = fopen(output_file.c_str(), "wb");
        if (!fout) { cerr << "Cannot open output file " << output_file << "\n"; return 1; }

        int feature_dim = F2 * (IMG_H/2/2) * (IMG_W/2/2); // 128 * 8 * 8 = 8192
        
        fwrite(&N, sizeof(int), 1, fout);
        fwrite(&feature_dim, sizeof(int), 1, fout);

        cout << "[INFO] Extracting features to " << output_file << "...\n";
        cout << "[INFO] Feature Dimension: " << feature_dim << " per image.\n";

        // Buffer for single feature vector
        vector<float> latent_vec(feature_dim);
        int img_size = IMG_C * IMG_H * IMG_W;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N; ++i) {
            const float* img_ptr = dataset.data() + i * img_size;
            
            // Run Encoder
            ae.extract_feature_single(img_ptr, latent_vec.data());

            // Write Label
            fwrite(&labels[i], sizeof(unsigned char), 1, fout);
            
            // Write Features
            fwrite(latent_vec.data(), sizeof(float), feature_dim, fout);

            if ((i+1) % 1000 == 0) {
                cout << "[PROGRESS] Processed " << i+1 << "/" << N << " images\r" << flush;
            }
        }
        cout << "[PROGRESS] Processed " << N << "/" << N << " images. Done.\n";
        fclose(fout);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        cout << "[LOG] Total Extraction time: " << elapsed.count() << "s (" 
             << fixed << setprecision(4) << (elapsed.count()/N)*1000 << " ms/image)\n";
        cout << "========================================" << endl;
        
        return 0;
    }
    return 0;
}