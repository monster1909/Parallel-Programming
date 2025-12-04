# tqdm – Thanh tiến trình C++ / C++ Progress Bar

## VN

- Mô-đun này cung cấp một thanh tiến trình (progress bar) viết bằng C++, hoạt động tương tự Python `tqdm`.

- Được sử dụng trong toàn bộ đồ án để theo dõi tiến trình huấn luyện CPU/GPU, trích xuất đặc trưng và các bước xử lý lâu.

### Cách chạy file `test_tqdm.cpp`:

```bash
g++ test_tqdm.cpp tqdm.cpp -std=c++17 -o /tmp/a && /tmp/a
```

Kết quả:
```bash
100% |██████████████████████████| ETA: 0.0s  Speed:    3 it/s  final_loss 0.3624  final_lr: 0.0951  final_acc: 202.0000
```
## EN


- This module provides a lightweight progress bar written in C++, similar to Python’s tqdm.

- It is used throughout the project to monitor CPU/GPU training progress, feature extraction, and other long-running operations.

### How to run `test_tqdm.cpp`:


```bash
g++ test_tqdm.cpp tqdm.cpp -std=c++17 -o /tmp/a && /tmp/a
```

Result:
```bash
100% |██████████████████████████| ETA: 0.0s  Speed:    3 it/s  final_loss 0.3624  final_lr: 0.0951  final_acc: 202.0000
```