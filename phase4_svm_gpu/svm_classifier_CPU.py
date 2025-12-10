#!/usr/bin/env python3
import argparse
import os
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- THƯ VIỆN CHÍNH: THUNDERSVM ---
try:
    from thundersvm import SVC
    print("[SYSTEM] Đã tìm thấy ThunderSVM. Sẵn sàng chạy trên GPU.")
except ImportError:
    print("[ERROR] Chưa cài đặt ThunderSVM.")
    print("Vui lòng chạy: pip install thundersvm")
    exit(1)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CẤU HÌNH ---
FEATURE_DIM = 8192
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def parse_args():
    parser = argparse.ArgumentParser(description="Train SVM using ThunderSVM (GPU)")
    
    parser.add_argument('--train_file', type=str, required=True, help='Path to train binary file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test binary file')
    parser.add_argument('--output_dir', type=str, default='./output_thundersvm', help='Output directory')
    parser.add_argument('--C', type=float, default=10.0, help='Regularization parameter')
    # ThunderSVM hỗ trợ: linear, polynomial, rbf, sigmoid
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
    
    labels = np.zeros(num_samples, dtype=np.float32) 
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

    # 2. Preprocessing
    # Vẫn cần chuẩn hóa để đảm bảo độ chính xác và tốc độ hội tụ
    print("-" * 40)
    print("[PREPROCESS] Chuẩn hóa dữ liệu (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Khởi tạo và Train Model với ThunderSVM
    print("-" * 40)
    print(f"[THUNDERSVM] Bắt đầu huấn luyện (Kernel: {args.kernel}, C: {args.C})...")
    
    # gpu_id=0: Sử dụng GPU đầu tiên
    # n_jobs=-1: Sử dụng tất cả các luồng CPU để hỗ trợ (nếu cần)
    clf = SVC(kernel=args.kernel, C=args.C, gamma='auto', verbose=True, gpu_id=0)
    
    start_train = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"[DONE] Huấn luyện hoàn tất sau: {train_time:.2f}s")

    # 4. Save Model
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'model_thundersvm.pkl')
        # ThunderSVM có phương thức save_to_file riêng, ổn định hơn pickle
        clf.save_to_file(model_path)
        print(f"[SAVE] Đã lưu model tại: {model_path}")

    # 5. Evaluate
    print("-" * 40)
    print("[EVAL] Đang dự đoán trên tập Test...")
    
    start_eval = time.time()
    y_pred = clf.predict(X_test)
    eval_time = time.time() - start_eval

    acc = accuracy_score(y_test, y_pred)
    print(f"   -> Thời gian dự đoán: {eval_time:.2f}s")
    print(f"   -> ĐỘ CHÍNH XÁC: {acc * 100:.2f}%")

    # Lưu báo cáo
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    print(report)
    with open(os.path.join(args.output_dir, 'report.txt'), "w") as f:
        f.write(report)
        f.write(f"\nTraining Time: {train_time:.2f}s")
    
    # 6. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix (ThunderSVM) - Acc: {acc*100:.2f}%')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    print(f"[PLOT] Đã lưu Confusion Matrix vào {args.output_dir}")

if __name__ == "__main__":
    main()