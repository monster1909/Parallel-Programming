#!/usr/bin/env python3
import argparse
import os
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Dùng joblib để lưu model cuML

# --- THAY ĐỔI: DÙNG CUML THAY VÌ THUNDERSVM ---
try:
    from cuml.svm import SVC
    print("[SYSTEM] Đã tìm thấy RAPIDS cuML. Sẵn sàng chạy trên GPU.")
except ImportError:
    print("[ERROR] Chưa cài đặt RAPIDS cuML.")
    print("Vui lòng cài đặt: pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com")
    exit(1)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier

# --- CẤU HÌNH ---
FEATURE_DIM = 8192
TRAIN_SAMPLES = 50000 
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def parse_args():
    parser = argparse.ArgumentParser(description="Train SVM using RAPIDS cuML (GPU)")
    parser.add_argument('--input_file', type=str, required=True, help='Path to binary file')
    parser.add_argument('--output_dir', type=str, default='./output_cuml', help='Output directory')
    parser.add_argument('--C', type=float, default=10.0, help='Regularization parameter')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'rbf', 'poly', 'sigmoid'], 
                        help='Kernel type')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    return parser.parse_args()

def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    print(f"[I/O] Đang đọc file: {filename}...")
    file_size = os.path.getsize(filename)
    sample_size = 1 + (FEATURE_DIM * 4)
    num_samples = file_size // sample_size
    
    # LOAD FEATURES (ignore placeholder labels in file)
    features = np.zeros((num_samples, FEATURE_DIM), dtype=np.float32)

    with open(filename, 'rb') as f:
        buffer = f.read()

    offset = 0
    start_time = time.time()
    for i in range(num_samples):
        offset += 1  # Skip placeholder label
        features[i] = np.frombuffer(buffer, dtype=np.float32, count=FEATURE_DIM, offset=offset)
        offset += FEATURE_DIM * 4
    
    print(f" -> Đã đọc {num_samples} mẫu trong {time.time() - start_time:.2f}s")
    
    # LOAD REAL LABELS từ CIFAR-10 dataset
    print("[I/O] Loading real labels from CIFAR-10 dataset...")
    labels = load_cifar10_labels(num_samples)
    
    return features, labels

def load_cifar10_labels(num_samples):
    """Load labels from CIFAR-10 binary files"""
    labels = []
    
    # CIFAR-10 structure: 1 label byte + 3072 image bytes per sample
    cifar_path = "../../Data/cifar-10-batches-bin/"
    
    # Train files (50k samples)
    for i in range(1, 6):
        batch_file = os.path.join(cifar_path, f"data_batch_{i}.bin")
        if os.path.exists(batch_file):
            with open(batch_file, 'rb') as f:
                data = f.read()
                for j in range(10000):  # 10k per batch
                    offset = j * (1 + 3072)
                    label = struct.unpack_from('B', data, offset)[0]
                    labels.append(label)
    
    # Test file (10k samples)
    test_file = os.path.join(cifar_path, "test_batch.bin")
    if os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            data = f.read()
            for j in range(10000):
                offset = j * (1 + 3072)
                label = struct.unpack_from('B', data, offset)[0]
                labels.append(label)
    
    labels = np.array(labels[:num_samples], dtype=np.float32)
    print(f" -> Loaded {len(labels)} labels, unique classes: {np.unique(labels)}")
    return labels

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    X_all, y_all = load_data(args.input_file)

    # 2. Split Data
    if len(y_all) <= TRAIN_SAMPLES:
        print(f"[ERROR] Không đủ dữ liệu ({len(y_all)}) để split {TRAIN_SAMPLES} train.")
        exit(1)

    X_train = X_all[:TRAIN_SAMPLES]
    y_train = y_all[:TRAIN_SAMPLES]
    X_test = X_all[TRAIN_SAMPLES:]
    y_test = y_all[TRAIN_SAMPLES:]
    
    del X_all, y_all # Giải phóng RAM

    # 3. Preprocessing
    print("[PREPROCESS] Chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Train với cuML
    print(f"[CUML] Training (Kernel: {args.kernel}, C: {args.C})...")
    print("[INFO] Using OneVsRest strategy for multi-class (10 classes)...")
    
    # Wrap cuML SVC với OneVsRestClassifier để hỗ trợ multi-class
    base_clf = SVC(kernel=args.kernel, C=args.C, gamma='auto', verbose=False)
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    
    start_train = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"[DONE] Train time: {train_time:.2f}s")

    # 5. Save Model (Dùng joblib thay vì save_to_file)
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'model_cuml.pkl')
        joblib.dump(clf, model_path)
        print(f"[SAVE] Saved model to: {model_path}")

    # 6. Evaluate
    print("[EVAL] Predicting...")
    start_eval = time.time()
    y_pred = clf.predict(X_test)
    eval_time = time.time() - start_eval

    # Chuyển về numpy (CPU) để tính toán metric sklearn nếu cần thiết
    # (Mặc định cuML trả về có thể là cupy array hoặc numpy array tùy version)
    try:
        y_test_cpu = y_test.get() # Nếu là cupy/gpu array
        y_pred_cpu = y_pred.get()
    except AttributeError:
        y_test_cpu = y_test
        y_pred_cpu = y_pred

    acc = accuracy_score(y_test_cpu, y_pred_cpu)
    print(f" -> Eval time: {eval_time:.2f}s")
    print(f" -> Accuracy: {acc * 100:.2f}%")

    report = classification_report(y_test_cpu, y_pred_cpu, target_names=CLASS_NAMES)
    print(report)
    with open(os.path.join(args.output_dir, 'report.txt'), "w") as f:
        f.write(report)
        f.write(f"\nTraining Time: {train_time:.2f}s")

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test_cpu, y_pred_cpu)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix (cuML) - Acc: {acc*100:.2f}%')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    print(f"[PLOT] Saved matrix to {args.output_dir}")

if __name__ == "__main__":
    main()