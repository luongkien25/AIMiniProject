# Báo Cáo Tiến Độ Ngày 11

Cập nhật: 2026-05-10

## 1. Mục Tiêu Ngày 11

Ngày 11 tập trung vào bước chọn mô hình tốt nhất và lưu mô hình bằng `joblib`.

File cần tạo:

```text
models/sentiment_model.joblib
models/issue_model.joblib
```

Mô hình được chọn dựa trên `Macro F1`, không chỉ dựa trên `Accuracy`.

Lý do:

```text
Dataset có nhiều lớp và phân bố nhãn không hoàn toàn cân bằng.
Macro F1 đánh giá công bằng hơn vì mỗi lớp có trọng số ngang nhau.
```

## 2. Mô Hình Sentiment Được Chọn

Best model từ ngày 10:

```text
Task   : Sentiment Classification
Feature: TF-IDF Unigram + Bigram
Model  : Logistic Regression
Accuracy: 0.8636
Macro F1: 0.8570
```

Lý do chọn:

- Có Macro F1 cao nhất trong các mô hình sentiment đã thử.
- Bigram giúp giữ các cụm từ quan trọng như `không tốt`, `giao chậm`, `sai màu`, `bị lỗi`.
- Logistic Regression hoạt động ổn định trên vector TF-IDF.

File đã lưu:

```text
models/sentiment_model.joblib
```

## 3. Mô Hình Issue Được Chọn

Best model từ ngày 10:

```text
Task   : Issue Classification
Feature: TF-IDF Unigram
Model  : Linear SVM
Accuracy: 0.7330
Macro F1: 0.7734
```

Lý do chọn:

- Có Macro F1 cao nhất trong các mô hình issue đã thử.
- Linear SVM phù hợp với dữ liệu văn bản sau khi chuyển thành vector TF-IDF.
- Sau khi gom nhãn issue, mô hình học ổn định hơn và Macro F1 tăng rõ.

File đã lưu:

```text
models/issue_model.joblib
```

## 4. Cách Lưu Mô Hình

Script dùng để lưu model:

```text
src/save_models.py
```

Chạy bằng lệnh:

```powershell
python src/save_models.py
```

Quy trình:

1. Đọc file `data/processed/merged_preprocessed_reviews.csv`.
2. Tạo pipeline gồm `TfidfVectorizer` và model tốt nhất.
3. Train lại model trên toàn bộ dataset hiện có.
4. Lưu model bằng `joblib.dump`.

## 5. Kiểm Tra Model Sau Khi Lưu

Sau khi lưu, hai file `.joblib` đã được kiểm tra bằng `joblib.load`.

Kết quả kiểm tra:

```text
models/sentiment_model.joblib: load được và dự đoán được sentiment_label
models/issue_model.joblib    : load được và dự đoán được issue_label
```

Ví dụ:

```text
"sản phẩm tốt đóng gói đẹp"
-> sentiment: positive

"shop giao sai màu và bị lỗi"
-> sentiment: negative
-> issue: wrong_missing_item

"hộp bị móp khi nhận hàng"
-> sentiment: negative
-> issue: packaging
```

## 6. Kết Luận Ngày 11

Ngày 11 đã hoàn thành bước chọn và lưu mô hình tốt nhất.

Kết quả:

```text
Sentiment model:
models/sentiment_model.joblib

Issue model:
models/issue_model.joblib
```

Hai mô hình này có thể được load lại để dự đoán dữ liệu mới mà không cần train lại từ đầu.
