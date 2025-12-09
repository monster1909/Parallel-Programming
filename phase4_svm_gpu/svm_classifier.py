#!/usr/bin/env python3
import argparse
import os
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- THƯ VIỆN GPU ---
try:
    import cuml
    from cuml.svm import SVC as GPU_SVC
    print("[SYSTEM] Đã phát hiện thư viện cuML. Sẵn sàng chạy trên GPU.")
except ImportError:
    print("[ERROR] Không tìm thấy thư viện 'cuml'.")
    print("Vui lòng cài đặt RAPIDS để chạy trên GPU (xem hướng dẫn bên dưới).")
    exit(1)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CẤU HÌNH ---
FEATURE_DIM = 8192
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def parse_args():
    parser = argparse.ArgumentParser(description="Train SVM on GPU (via RAPIDS cuML) for CIFAR-10 Features")
    
    parser.add_argument('--train_file', type=str, required=True, help='Path to train binary file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test binary file')
    parser.add_argument('--output_dir', type=str, default='./output_gpu', help='Output directory')
    parser.add_argument('--C', type=float, default=10.0, help='Regularization parameter')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'rbf', 'poly', 'sigmoid'], 
                        help='Kernel type (default: rbf)')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    
    return parser.parse_args()

def load_data(filename):
    """Đọc dữ liệu binary: 1 byte label + 8192 floats feature"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    print(f"[I/O] Đang đọc file: {filename}...")
    file_size = os.path.getsize(filename)
    sample_size = 1 + (FEATURE_DIM * 4)
    num_samples = file_size // sample_size
    
    labels = np.zeros(num_samples, dtype=np.float32) # cuML thích nhãn kiểu float/int32
    features = np.zeros((num_samples, FEATURE_DIM), dtype=np.float32)

    with open(filename, 'rb') as f:
        buffer = f.read()

    offset = 0
    start_time = time.time()
    for i in range(num_samples):
        labels[i] = struct.unpack_from('B', buffer, offset)[0]
        offset += 1
        features[i] = np.frombuffer(buffer, dtype=np.float32, count=FEATURE_DIM, offset=offset)
        offset += FEATURE_DIM * 4
    
    print(f"   -> Đã đọc {num_samples} mẫu trong {time.time() - start_time:.2f}s")
    return features, labels

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    print("-" * 40)
    X_train, y_train = load_data(args.train_file)
    X_test, y_test = load_data(args.test_file)

    # 2. Khởi tạo và Train Model trên GPU
    print("-" * 40)
    print(f"[GPU TRAIN] Bắt đầu huấn luyện SVM (Kernel: {args.kernel}, C: {args.C})...")
    
    # cuML SVC có API tương tự Sklearn nhưng chạy trên GPU
    clf = GPU_SVC(kernel=args.kernel, C=args.C, gamma='scale', verbose=True)
    
    start_train = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"[DONE] Huấn luyện hoàn tất sau: {train_time:.2f}s")

    # 3. Save Model
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'svm_model_gpu.pkl')
        joblib.dump(clf, model_path)
        print(f"[SAVE] Đã lưu model tại: {model_path}")

    # 4. Evaluate
    print("-" * 40)
    print("[EVAL] Đang dự đoán trên tập Test...")
    
    start_eval = time.time()
    y_pred = clf.predict(X_test)
    eval_time = time.time() - start_eval

    # Chuyển đổi kết quả về CPU numpy array (đề phòng cuML trả về cuDF/cupy array)
    if hasattr(y_pred, 'to_numpy'):
        y_pred = y_pred.to_numpy()
    elif hasattr(y_pred, 'get'): # cupy
        y_pred = y_pred.get()
    
    acc = accuracy_score(y_test, y_pred)
    print(f"   -> Thời gian dự đoán: {eval_time:.2f}s")
    print(f"   -> ĐỘ CHÍNH XÁC: {acc * 100:.2f}%")

    # Lưu báo cáo
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    with open(os.path.join(args.output_dir, 'report.txt'), "w") as f:
        f.write(report)
        f.write(f"\nTraining Time: {train_time:.2f}s")
    
    # 5. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix (GPU SVM) - Acc: {acc*100:.2f}%')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    print(f"[PLOT] Đã lưu Confusion Matrix vào {args.output_dir}")

if __name__ == "__main__":
    main()