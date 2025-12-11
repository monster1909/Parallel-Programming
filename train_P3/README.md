# train_P3 — Feature extraction & training assets from Phase3_v2

Thư mục này tạo pipeline trích xuất đặc trưng 8,192 chiều bằng autoencoder tối ưu hóa ở `phase3_gpu_optimized_v2`, sau đó xuất ra file `.bin` đúng định dạng để train SVM (phase4_svm hoặc phase4_svm_gpu).

## Build
```bash
cd train_P3
make
```

Yêu cầu: CUDA (có `nvcc`), C++17, dataset nằm ở `../Data/cifar-10-batches-bin` (mặc định). Path khác có thể truyền bằng tham số.

## Chạy
```bash
./train_p3 \
  --data_dir ../Data/cifar-10-batches-bin \
  --output ./output_features \
  --batch 32
```

Tùy chọn:
- `--train-only` chỉ xuất `train_features.bin`
- `--test-only` chỉ xuất `test_features.bin`
- `--batch <N>` thay đổi batch size (phải ≤ 64 theo cấu hình autoencoder).

## Định dạng output
Mỗi file `.bin` có header và dữ liệu:
1. `int32 N`  : số mẫu.
2. `int32 D`  : số chiều (8192).
3. Với từng mẫu: `uint8 label` + `float32[D]` vector đặc trưng.

File này đưa thẳng vào:
- `phase4_svm/train_svm` và `phase4_svm/test_svm`
- hoặc `phase4_svm_gpu/svm_classifier.py`

## Quy trình gợi ý
1. Build & extract đặc trưng: `./train_p3 ...`
2. Train SVM CPU: `cd ../phase4_svm && make && ./train_svm ../train_P3/output_features/train_features.bin model.model`
3. Test SVM: `./test_svm model.model ../train_P3/output_features/test_features.bin result.csv`
4. (Tùy chọn) Train SVM GPU: dùng script trong `phase4_svm_gpu/` với các file `.bin` vừa tạo.

