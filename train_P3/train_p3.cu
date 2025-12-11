#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../phase3_gpu_optimized_v2/Include/autoencoder.h"

namespace fs = std::filesystem;

// Simple CIFAR-10 loader (binary format)
struct Dataset {
    std::vector<unsigned char> labels;
    std::vector<float> images; // CHW layout, normalized to [0,1]
    int count = 0;
};

constexpr int IMG_H = 32;
constexpr int IMG_W = 32;
constexpr int IMG_C = 3;
constexpr int FEAT_DIM = 128 * 8 * 8; // 8192

bool load_cifar_file(const fs::path &file, Dataset &dst) {
    std::ifstream fin(file, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "[ERROR] Cannot open " << file << "\n";
        return false;
    }

    const int record_bytes = 1 + IMG_C * IMG_H * IMG_W;
    std::vector<unsigned char> buf(record_bytes);

    while (fin.read(reinterpret_cast<char *>(buf.data()), record_bytes)) {
        dst.labels.push_back(buf[0]);
        for (int c = 0; c < IMG_C; ++c) {
            for (int h = 0; h < IMG_H; ++h) {
                for (int w = 0; w < IMG_W; ++w) {
                    int src_idx = 1 + c * IMG_H * IMG_W + h * IMG_W + w;
                    dst.images.push_back(static_cast<float>(buf[src_idx]) / 255.0f);
                }
            }
        }
        dst.count++;
    }
    std::cout << "[LOAD] " << file.filename() << " -> " << dst.count << " images (total)\n";
    return true;
}

Dataset load_dataset(const fs::path &data_dir, bool is_train) {
    Dataset ds;
    if (is_train) {
        for (int i = 1; i <= 5; ++i) {
            fs::path f = data_dir / ("data_batch_" + std::to_string(i) + ".bin");
            load_cifar_file(f, ds);
        }
    } else {
        fs::path f = data_dir / "test_batch.bin";
        load_cifar_file(f, ds);
    }
    return ds;
}

void write_feature_bin(const fs::path &out_file,
                       const std::vector<unsigned char> &labels,
                       const std::vector<float> &features) {
    FILE *f = fopen(out_file.string().c_str(), "wb");
    if (!f) {
        std::cerr << "[ERROR] Cannot open output: " << out_file << "\n";
        return;
    }
    int N = static_cast<int>(labels.size());
    int dim = FEAT_DIM;
    fwrite(&N, sizeof(int), 1, f);
    fwrite(&dim, sizeof(int), 1, f);

    const float *ptr = features.data();
    for (int i = 0; i < N; ++i) {
        fwrite(&labels[i], 1, 1, f);
        fwrite(ptr + i * dim, sizeof(float), dim, f);
    }
    fclose(f);
    std::cout << "[SAVE] " << out_file << " (samples=" << N << ", dim=" << dim << ")\n";
}

void extract_and_save(const Dataset &ds,
                      const fs::path &out_file,
                      int batch_size) {
    Autoencoder model(IMG_H, IMG_W, IMG_C, batch_size);

    const int N = ds.count;
    const int in_per_img = IMG_C * IMG_H * IMG_W;
    std::vector<float> batch_input(batch_size * in_per_img);
    std::vector<float> batch_feat(batch_size * FEAT_DIM);
    std::vector<float> all_features;
    all_features.reserve(static_cast<size_t>(N) * FEAT_DIM);

    int num_batches = (N + batch_size - 1) / batch_size;
    for (int b = 0; b < num_batches; ++b) {
        int start = b * batch_size;
        int cur = std::min(batch_size, N - start);

        // Fill batch (pad with zeros if last batch smaller)
        std::fill(batch_input.begin(), batch_input.end(), 0.0f);
        for (int i = 0; i < cur; ++i) {
            const float *src = ds.images.data() + (start + i) * in_per_img;
            std::copy(src, src + in_per_img, batch_input.begin() + i * in_per_img);
        }

        model.extract_features(batch_input.data(), batch_feat.data(), batch_size, false);

        all_features.insert(all_features.end(),
                            batch_feat.begin(),
                            batch_feat.begin() + cur * FEAT_DIM);

        if (b % 20 == 0) {
            float pct = (b * 100.0f) / num_batches;
            std::cout << "[PROGRESS] " << pct << "% (" << start << "/" << N << ")\r" << std::flush;
        }
    }
    std::cout << "[PROGRESS] 100% (" << N << "/" << N << ")\n";

    write_feature_bin(out_file, ds.labels, all_features);
}

struct Options {
    fs::path data_dir = "../Data/cifar-10-batches-bin";
    fs::path output_dir = "./output";
    int batch = 32;
    bool do_train = true;
    bool do_test = true;
};

Options parse_args(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--data_dir" || a == "-d") && i + 1 < argc) {
            opt.data_dir = argv[++i];
        } else if ((a == "--output" || a == "-o") && i + 1 < argc) {
            opt.output_dir = argv[++i];
        } else if ((a == "--batch" || a == "-b") && i + 1 < argc) {
            opt.batch = std::stoi(argv[++i]);
        } else if (a == "--train-only") {
            opt.do_test = false;
        } else if (a == "--test-only") {
            opt.do_train = false;
        } else {
            std::cout << "Unknown arg: " << a << "\n";
        }
    }
    return opt;
}

int main(int argc, char **argv) {
    Options opt = parse_args(argc, argv);
    fs::create_directories(opt.output_dir);

    std::cout << "=== TRAIN_P3: Feature Extraction using Phase3_v2 Autoencoder ===\n";
    std::cout << "Data dir : " << opt.data_dir << "\n";
    std::cout << "Output dir: " << opt.output_dir << "\n";
    std::cout << "Batch size: " << opt.batch << "\n";

    if (opt.do_train) {
        std::cout << "\n[STEP] Loading TRAIN batches...\n";
        Dataset train_ds = load_dataset(opt.data_dir, true);
        std::cout << "[INFO] Total train images: " << train_ds.count << "\n";
        fs::path out = opt.output_dir / "train_features.bin";
        extract_and_save(train_ds, out, opt.batch);
    }

    if (opt.do_test) {
        std::cout << "\n[STEP] Loading TEST batch...\n";
        Dataset test_ds = load_dataset(opt.data_dir, false);
        std::cout << "[INFO] Total test images: " << test_ds.count << "\n";
        fs::path out = opt.output_dir / "test_features.bin";
        extract_and_save(test_ds, out, opt.batch);
    }

    std::cout << "\nDone. You can feed *.bin into phase4_svm or phase4_svm_gpu.\n";
    return 0;
}

