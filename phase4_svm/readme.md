## Chạy File.


#### Bước 1. Biên Dịch
Tại Góc Thư Mục Chạy Lệnh:
```
make
```
Sẽ Tạo ra 2 file chạy: train_svm và test_svm.

#### Bước 2: Chạy trainning.
```
./train_svm train_features.bin my_model.model
```

- train_features.bin: File chứa vector đặc trưng của tập Train.

- my_model.model: Tên file model sẽ lưu sau khi train xong.



#### Bước 4: Chạy Testing & Xuất CSV

```
./test_svm my_model.model test_features.bin result_matrix.csv
```
