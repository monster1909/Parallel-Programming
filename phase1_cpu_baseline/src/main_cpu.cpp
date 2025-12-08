#include <iostream>
#include <fstream>
#include <vector>
#include <string>
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

// Data Loader
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

int main(int argc, char** argv) {
    srand(1234);

    if (argc < 2) {
        cout << "Usage:\n  Train: " << argv[0] << " train data_batch_1.bin ... weights.bin\n";
        cout << "  Test : " << argv[0] << " test weights.bin input.ppm output.ppm\n";
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

        int steps = N / BATCH_SIZE;
        cout << "Training: " << EPOCHS << " epochs, " << steps << " steps/epoch\n";

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

                if (step % 10 == 0) cout << "Epoch " << epoch+1 << " Step " << step << " Loss=" << loss << "\n";
            }
            cout << "Epoch " << epoch+1 << " avg loss=" << (epoch_loss/steps) << " time=" << t.get_elapsed_seconds() << "s\n";
        }
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

        ae.load_weights(weights_file);

        vector<float> img(IMG_C * IMG_H * IMG_W);
        ifstream f(ppm_in);
        if (!f) { cerr << "Cannot open ppm\n"; return 1; }
        string magic; int w,h,maxv;
        f >> magic >> w >> h >> maxv;
        for (int y=0; y<IMG_H; y++) {
            for (int x=0; x<IMG_W; x++) {
                int r,g,b; f>>r>>g>>b;
                img[(0*IMG_H+y)*IMG_W+x] = r/255.0f;
                img[(1*IMG_H+y)*IMG_W+x] = g/255.0f;
                img[(2*IMG_H+y)*IMG_W+x] = b/255.0f;
            }
        }
        f.close();

        float out_img[IMG_C * IMG_H * IMG_W];
        AutoEncoder::Activations act;
        act.alloc(1);
        ae.forward_batch(img.data(), out_img, act, 1);
        act.free_all();
        save_ppm(out_img, ppm_out);
    }
    return 0;
}