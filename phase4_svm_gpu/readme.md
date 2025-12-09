### Cách chạy (Command Line)

Cấu trúc Input đầu vào có dạng:
LABEL[1 byte char] + [Feature Vector (8192 floats)]



Đầu tiên, bạn cần cài đặt các thư viện cần thiết:

```bash
pip install numpy matplotlib seaborn scikit-learn joblib
```

Sau đó, bạn chạy file python với các tham số tương ứng:

**Cách chạy cơ bản:**

```bash
python svm_classifier.py --train_file data/train_features.bin --test_file data/test_features.bin
```

**Cách chạy tùy chỉnh (Thay đổi tham số C, Kernel và lưu model):**

```bash
python svm_classifier.py \
    --train_file data/train_features.bin \
    --test_file data/test_features.bin \
    --output_dir results_phase4 \
    --C 100.0 \
    --kernel rbf \
    --save_model
```

### 3\. Giải thích các tham số (Arguments)

  * `--train_file`: (Bắt buộc) Đường dẫn file chứa feature train.
  * `--test_file`: (Bắt buộc) Đường dẫn file chứa feature test.
  * `--output_dir`: Thư mục chứa kết quả output (mặc định là `./output`).
  * `--C`: Tham số phạt lỗi của SVM. Giá trị càng lớn thì càng cố gắng khớp đúng dữ liệu train (có thể gây overfitting). Đề bài gợi ý thử nghiệm, bạn có thể thử 1, 10, 100.
  * `--kernel`: Loại nhân (kernel). Đề bài yêu cầu `rbf` (mặc định trong code này), nhưng bạn có thể đổi sang `linear` để so sánh.
  * `--save_model`: Nếu thêm cờ này vào dòng lệnh, script sẽ lưu file model `.pkl` (khoảng vài trăm MB) để bạn có thể tái sử dụng mà không cần train lại.