#!/usr/bin/env python3
import argparse
import os
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Dùng để lưu model sau khi train
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- HẰNG SỐ CỐ ĐỊNH ---
FEATURE_DIM = 8192  # Số chiều của vector đặc trưng
IMG_SIZE = 32 * 32 * 3  # Kích thước ảnh gốc (nếu cần tham chiếu)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def parse_args():
    """
    Hàm xử lý các tham số dòng lệnh.
    """
    parser = argparse.ArgumentParser(description="Huấn luyện SVM cho bài toán phân lớp ảnh CIFAR-10 dựa trên Feature Vector.")
    
    # Các tham số bắt buộc
    parser.add_argument('--train_file', type=str, required=True, 
                        help='Đường dẫn đến file binary chứa dữ liệu Train (Features + Labels)')
    parser.add_argument('--test_file', type=str, required=True, 
                        help='Đường dẫn đến file binary chứa dữ liệu Test (Features + Labels)')
    
    # Các tham số tùy chọn (có giá trị mặc định)
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='Thư mục để lưu kết quả (model, ảnh confusion matrix)')
    parser.add_argument('--C', type=float, default=10.0, 
                        help='Tham số Regularization cho SVM (mặc định: 10.0)')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'rbf', 'poly'], 
                        help='Loại Kernel sử dụng cho SVM (mặc định: rbf)')
    parser.add_argument('--save_model', action='store_true', 
                        help='Nếu có cờ này, script sẽ lưu model đã train ra file .pkl')

    return parser.parse_args()

def load_data(filename):
    """
    Đọc dữ liệu từ file binary.
    Format: [Label (1 byte)] + [Feature Vector (8192 floats)]
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Không tìm thấy file: {filename}")

    print(f"[INFO] Đang đọc dữ liệu từ: {filename}...")
    file_size = os.path.getsize(filename)
    
    # Tính toán kích thước 1 mẫu: 1 byte label + 8192 * 4 bytes feature
    sample_size = 1 + (FEATURE_DIM * 4)
    num_samples = file_size // sample_size
    
    print(f"   + Kích thước file: {file_size} bytes")
    print(f"   + Số lượng mẫu ước tính: {num_samples}")

    # Cấp phát bộ nhớ trước để tăng tốc độ
    # Label là uint8 (0-255), Feature là float32
    labels = np.zeros(num_samples, dtype=np.uint8)
    features = np.zeros((num_samples, FEATURE_DIM), dtype=np.float32)

    with open(filename, 'rb') as f:
        # Đọc toàn bộ file vào RAM một lần (Batch Read)
        buffer = f.read()

    offset = 0
    feature_bytes_len = FEATURE_DIM * 4

    start_read = time.time()
    for i in range(num_samples):
        # 1. Đọc Label (1 byte unsigned char)
        # struct.unpack_from trả về tuple, lấy phần tử [0]
        labels[i] = struct.unpack_from('B', buffer, offset)[0]
        offset += 1

        # 2. Đọc Feature Vector (8192 floats)
        # Sử dụng frombuffer để map trực tiếp từ memory, nhanh hơn struct.unpack nhiều
        features[i] = np.frombuffer(buffer, dtype=np.float32, count=FEATURE_DIM, offset=offset)
        offset += feature_bytes_len

    print(f"[INFO] Đã đọc xong {num_samples} mẫu trong {time.time() - start_read:.2f}s")
    return features, labels

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Vẽ và lưu Confusion Matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Đã lưu ảnh Confusion Matrix tại: {save_path}")
    plt.close()

def main():
    # 1. Lấy tham số từ dòng lệnh
    args = parse_args()
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load dữ liệu
    try:
        X_train, y_train = load_data(args.train_file)
        X_test, y_test = load_data(args.test_file)
    except Exception as e:
        print(f"[ERROR] Lỗi khi đọc file: {e}")
        return

    # 3. Huấn luyện SVM
    print("\n" + "="*30)
    print(f"BẮT ĐẦU HUẤN LUYỆN (Kernel={args.kernel}, C={args.C})")
    print("="*30)
    
    # Khởi tạo mô hình SVM
    # gamma='scale' là mặc định tốt (1 / (n_features * X.var()))
    # cache_size=1000 (MB) để tăng tốc độ nếu RAM dư dả
    clf = SVC(kernel=args.kernel, C=args.C, gamma='scale', cache_size=2000, verbose=True)
    
    start_train = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"[DONE] Thời gian huấn luyện: {train_time:.2f}s")

    # 4. Lưu Model (Tùy chọn)
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'svm_model.pkl')
        joblib.dump(clf, model_path)
        print(f"[INFO] Đã lưu model tại: {model_path}")

    # 5. Đánh giá trên tập Test
    print("\n" + "="*30)
    print("ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*30)
    
    start_eval = time.time()
    y_pred = clf.predict(X_test)
    eval_time = time.time() - start_eval
    
    acc = accuracy_score(y_test, y_pred)
    print(f"-> Thời gian dự đoán: {eval_time:.2f}s")
    print(f"-> ĐỘ CHÍNH XÁC (ACCURACY): {acc * 100:.2f}%")

    # Lưu báo cáo chi tiết ra file text
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc * 100:.2f}%")
        f.write(f"\nTrain Time: {train_time:.2f}s")
    
    print("\nDetailed Classification Report:")
    print(report)

    # 6. Vẽ Confusion Matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, cm_path)

if __name__ == "__main__":
    main()