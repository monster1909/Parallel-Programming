#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip> // For formatting output
#include <algorithm>
#include <chrono>
#include <cmath>   // For pow
#include "autoencoder.h"
#include "utils/timer.h"
#include "layers/mse.h"

using namespace std;

// --- Helper Functions (Save/Load) ---
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

vector<float> load_cifar_batches(const vector<string>& files, int &N_out) {
    // ... (Giữ nguyên code load_cifar_batches cũ)
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
    }
    N_out = (int)(data.size() / (IMG_C*IMG_H*IMG_W));
    return data;
}

void load_cifar_batches_with_labels(const vector<string>& files, 
                                    vector<float>& data_out, 
                                    vector<unsigned char>& labels_out, 
                                    int &N_out) {
    // ... (Giữ nguyên code load_cifar_batches_with_labels cũ)
    data_out.clear();
    labels_out.clear();
    for (auto &fpath : files) {
        ifstream fin(fpath, ios::binary);
        if (!fin) { cerr<<"Cannot open "<<fpath<<endl; exit(1); }
        const int record_size = 1 + 3072;
        vector<unsigned char> buf(record_size);
        while (fin.read((char*)buf.data(), record_size)) {
            labels_out.push_back(buf[0]);
            int current_size = data_out.size();
            data_out.resize(current_size + IMG_C*IMG_H*IMG_W);
            int base = current_size;
            for (int c=0; c<IMG_C; ++c) {
                for (int h=0; h<IMG_H; ++h) {
                    for (int w=0; w<IMG_W; ++w) {
                        int v = buf[1 + c*1024 + h*32 + w];
                        data_out[base + (c*IMG_H + h) * IMG_W + w] = v / 255.0f;
                    }
                }
            }
        }
    }
    N_out = (int)labels_out.size();
}

void print_sample_values(const float* data, int size, const std::string& header) {
    std::cout << "\n===== " << header << " (first 10 pixels) =====" << std::endl;
    for (int i = 0; i < std::min(10, size); ++i) {
        std::cout << std::fixed << std::setprecision(6) << data[i] << " ";
    }
    std::cout << "\n" << std::endl;
}

// --- INFO Helper Functions ---

// Tính số lượng tham số cho Conv Layer
long long count_params(const Conv2D& layer) {
    return (long long)layer.out_c * layer.in_c * layer.k * layer.k + layer.out_c; // Weights + Bias
}

// Tính FLOPs (Floating Point Operations) ước lượng
// Công thức chuẩn: 2 * Cin * K * K * Hout * Wout * Cout
long long count_flops(const Conv2D& layer) {
    return 2LL * layer.in_c * layer.k * layer.k * layer.Hout * layer.Wout * layer.out_c;
}

void print_layer_info(const string& name, const string& type, const string& in_shape, const string& out_shape, long long params, long long flops) {
    cout << left << setw(20) << name 
         << left << setw(15) << type
         << left << setw(18) << in_shape
         << left << setw(18) << out_shape
         << right << setw(12) << params
         << right << setw(15) << flops << endl;
}

string shape_str(int c, int h, int w) {
    return to_string(c) + "x" + to_string(h) + "x" + to_string(w);
}

// --- MAIN ---
int main(int argc, char** argv) {
    srand(1234);

    if (argc < 2) {
        cout << "Usage:\n";
        cout << "  Train  : " << argv[0] << " train data_batch_1.bin ... weights.bin\n";
        cout << "  Test   : " << argv[0] << " test weights.bin input.ppm output.ppm\n";
        cout << "  Extract: " << argv[0] << " extract weights.bin output.bin data_batch_1.bin ...\n";
        cout << "  Info   : " << argv[0] << " info\n"; // New Command
        return 0;
    }
    string mode = argv[1];
    AutoEncoder ae;
    ae.init();

    // -------------------------------------------------------------------------
    // MODE: INFO
    // -------------------------------------------------------------------------
    if (mode == "info") {
        cout << "\n==========================================================================================" << endl;
        cout << "                                  MODEL ARCHITECTURE SUMMARY                               " << endl;
        cout << "==========================================================================================" << endl;
        
        cout << left << setw(20) << "Layer Name" 
             << left << setw(15) << "Type"
             << left << setw(18) << "Input Shape"
             << left << setw(18) << "Output Shape"
             << right << setw(12) << "Params"
             << right << setw(15) << "FLOPs" << endl;
        cout << "------------------------------------------------------------------------------------------" << endl;

        long long total_params = 0;
        long long total_flops = 0;

        // --- Layer 1 ---
        long long p1 = count_params(ae.conv1);
        long long f1 = count_flops(ae.conv1);
        print_layer_info("Conv1", "Conv2d 3x3", shape_str(IMG_C, IMG_H, IMG_W), shape_str(F1, IMG_H, IMG_W), p1, f1);
        print_layer_info("ReLU1", "ReLU", shape_str(F1, IMG_H, IMG_W), shape_str(F1, IMG_H, IMG_W), 0, 0);
        print_layer_info("MaxPool1", "MaxPool 2x2", shape_str(F1, IMG_H, IMG_W), shape_str(F1, IMG_H/2, IMG_W/2), 0, 0);
        
        // --- Layer 2 ---
        long long p2 = count_params(ae.conv2);
        long long f2 = count_flops(ae.conv2);
        print_layer_info("Conv2", "Conv2d 3x3", shape_str(F1, IMG_H/2, IMG_W/2), shape_str(F2, IMG_H/2, IMG_W/2), p2, f2);
        print_layer_info("ReLU2", "ReLU", shape_str(F2, IMG_H/2, IMG_W/2), shape_str(F2, IMG_H/2, IMG_W/2), 0, 0);
        print_layer_info("MaxPool2", "MaxPool 2x2", shape_str(F2, IMG_H/2, IMG_W/2), shape_str(F2, IMG_H/4, IMG_W/4), 0, 0);

        // --- Layer 3 (Decoder Start) ---
        long long p3 = count_params(ae.conv3);
        long long f3 = count_flops(ae.conv3);
        print_layer_info("DecodeConv1", "Conv2d 3x3", shape_str(F2, IMG_H/4, IMG_W/4), shape_str(F2, IMG_H/4, IMG_W/4), p3, f3);
        print_layer_info("ReLU_Dec1", "ReLU", shape_str(F2, IMG_H/4, IMG_W/4), shape_str(F2, IMG_H/4, IMG_W/4), 0, 0);
        print_layer_info("Upsample1", "Upsample 2x", shape_str(F2, IMG_H/4, IMG_W/4), shape_str(F2, IMG_H/2, IMG_W/2), 0, 0);

        // --- Layer 4 ---
        long long p4 = count_params(ae.conv4);
        long long f4 = count_flops(ae.conv4);
        print_layer_info("DecodeConv2", "Conv2d 3x3", shape_str(F2, IMG_H/2, IMG_W/2), shape_str(F1, IMG_H/2, IMG_W/2), p4, f4);
        print_layer_info("ReLU_Dec2", "ReLU", shape_str(F1, IMG_H/2, IMG_W/2), shape_str(F1, IMG_H/2, IMG_W/2), 0, 0);
        print_layer_info("Upsample2", "Upsample 2x", shape_str(F1, IMG_H/2, IMG_W/2), shape_str(F1, IMG_H, IMG_W), 0, 0);

        // --- Layer 5 ---
        long long p5 = count_params(ae.conv5);
        long long f5 = count_flops(ae.conv5);
        print_layer_info("FinalConv", "Conv2d 3x3", shape_str(F1, IMG_H, IMG_W), shape_str(IMG_C, IMG_H, IMG_W), p5, f5);

        total_params = p1 + p2 + p3 + p4 + p5;
        total_flops = f1 + f2 + f3 + f4 + f5;

        cout << "------------------------------------------------------------------------------------------" << endl;
        cout << "Total Parameters: " << total_params << endl;
        cout << "Total Estimated FLOPs (Forward): " << total_flops << endl;
        cout << "Latent Vector Size: " << F2 * (IMG_H/4) * (IMG_W/4) << " elements (" << F2 << "x" << IMG_H/4 << "x" << IMG_W/4 << ")" << endl;
        cout << "==========================================================================================\n" << endl;

        // --- BENCHMARK SECTION ---
        cout << "Running Performance Benchmark (50 iterations)... Please wait." << endl;
        
        // Tạo dữ liệu giả ngẫu nhiên
        int batch_size = 1;
        int img_size = IMG_C * IMG_H * IMG_W;
        float* dummy_in = (float*)xmalloc(sizeof(float) * img_size);
        float* dummy_out = (float*)xmalloc(sizeof(float) * img_size);
        for(int i=0; i<img_size; ++i) dummy_in[i] = (float)rand()/RAND_MAX;

        AutoEncoder::Activations act;
        act.alloc(batch_size);
        
        DetailedTimer bench_timer;
        int iterations = 50;

        // Warmup (chạy 5 lần không đo)
        for(int i=0; i<5; ++i) {
            ae.forward_batch(dummy_in, dummy_out, act, batch_size, nullptr);
        }

        // Benchmark loop (đo tích lũy)
        for(int i=0; i<iterations; ++i) {
            ae.forward_batch(dummy_in, dummy_out, act, batch_size, &bench_timer);
        }

        // Tính trung bình
        double total_ms = bench_timer.get_total_time_ms();
        double avg_ms = total_ms / iterations;

        // In báo cáo tùy chỉnh
        // Vì DetailedTimer chỉ lưu tổng, ta cần hack một chút để in trung bình
        // Ta sẽ dùng lại hàm print_timing_report của Timer nhưng hiểu rằng số liệu đó là TỔNG 50 lần,
        // Sau đó in thêm dòng tổng kết chia trung bình.
        
        // Để chuyên nghiệp hơn, ta truy cập map của DetailedTimer (cần sửa Timer class thành public map hoặc getter, 
        // nhưng với code hiện tại, ta có thể dùng print_timing_report và chú thích rõ).
        
        cout << "\n===== PERFORMANCE REPORT (Average over " << iterations << " runs) =====" << endl;
        // In chi tiết (đang là tổng) - ta sẽ tự xử lý hiển thị ở đây nếu Timer::durations là private.
        // Do Timer::durations là private, ta dùng print_timing_report() có sẵn, nhưng lưu ý người dùng đó là Total time.
        // HOẶC TỐT HƠN: Ta sửa Timer một chút để lấy dữ liệu (như bạn đã làm ở các bước trước là đủ).
        // Ở đây tôi dùng cách hiển thị Total Time từ Timer và tính Avg thủ công ở dòng cuối.
        
        bench_timer.print_timing_report(); // In tổng thời gian 50 lần
        
        cout << "\n----------------------------------" << endl;
        cout << "Benchmark Result Summary:" << endl;
        cout << "  > Total Time (" << iterations << " runs): " << fixed << setprecision(2) << total_ms << " ms" << endl;
        cout << "  > Average Latency per Image : " << fixed << setprecision(4) << avg_ms << " ms" << endl;
        cout << "  > Throughput (CPU)          : " << fixed << setprecision(2) << (1000.0 / avg_ms) << " FPS" << endl;
        cout << "==================================" << endl;

        free(dummy_in);
        free(dummy_out);
        act.free_all();
        return 0;
    }

    // -------------------------------------------------------------------------
    // MODE: TRAIN
    // -------------------------------------------------------------------------
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

                if (step % 10 == 0) cout << "Epoch " << epoch+1 << " Step " << step << "/" << steps << " Loss=" << fixed << setprecision(8) << loss << "\n";
            }
            cout << "--------------------------------------------------------" << endl;
            cout << "[LOG] Epoch " << epoch+1 << " avg loss=" << fixed << setprecision(8) << (epoch_loss/steps) << " time=" << setprecision(3) << t.get_elapsed_seconds() << "s (" << (t.get_elapsed_seconds() / steps * 1000.0) << " ms/step)\n";
            cout << "--------------------------------------------------------" << endl;
        }
        
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
    // -------------------------------------------------------------------------
    // MODE: TEST
    // -------------------------------------------------------------------------
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
        
        DetailedTimer d_timer; 
        float out_img[IMG_C * IMG_H * IMG_W];
        AutoEncoder::Activations act;
        act.alloc(1);

        cout << "[INFO] Starting forward pass (with timing)..." << endl;
        ae.forward_batch(img.data(), out_img, act, 1, &d_timer);
        
        d_timer.print_timing_report();

        print_sample_values(out_img, IMG_C * IMG_H * IMG_W, "OUTPUT SAMPLE VALUES");

        act.free_all();
        save_ppm(out_img, ppm_out);
        cout << "\n===== DONE =====" << endl;
    }
    // -------------------------------------------------------------------------
    // MODE: EXTRACT
    // -------------------------------------------------------------------------
    else if (mode == "extract") {
        if (argc < 5) {
            cerr << "Extract mode requires: extract weights.bin output.bin data_batch_1.bin ...\n";
            return 1;
        }

        string weights_file = argv[2];
        string output_file  = argv[3];

        cout << "\n========================================" << endl;
        cout << "[INFO] Starting Feature Extraction (Encoder)" << endl;
        cout << "[INFO] Loading weights from: " << weights_file << endl;
        ae.load_weights(weights_file);

        vector<string> data_files;
        for (int i = 4; i < argc; ++i) data_files.push_back(argv[i]);

        vector<float> dataset;
        vector<unsigned char> labels;
        int N;
        load_cifar_batches_with_labels(data_files, dataset, labels, N);

        FILE* fout = fopen(output_file.c_str(), "wb");
        if (!fout) { cerr << "Cannot open output file " << output_file << "\n"; return 1; }

        int feature_dim = F2 * (IMG_H/2/2) * (IMG_W/2/2); 
        
        fwrite(&N, sizeof(int), 1, fout);
        fwrite(&feature_dim, sizeof(int), 1, fout);

        cout << "[INFO] Extracting features to " << output_file << "...\n";
        cout << "[INFO] Feature Dimension: " << feature_dim << " per image.\n";

        vector<float> latent_vec(feature_dim);
        int img_size = IMG_C * IMG_H * IMG_W;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N; ++i) {
            const float* img_ptr = dataset.data() + i * img_size;
            ae.extract_feature_single(img_ptr, latent_vec.data());
            fwrite(&labels[i], sizeof(unsigned char), 1, fout);
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