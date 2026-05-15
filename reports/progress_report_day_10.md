# Báo Cáo Tiến Độ Ngày 10

Cập nhật: 2026-05-10

## 1. Mục Tiêu Ngày 10

Ngày 10 tập trung vào bước đánh giá mô hình.

Các file cần tạo:

```text
reports/confusion_matrix_sentiment.png
reports/confusion_matrix_issue.png
reports/classification_report_sentiment.txt
reports/classification_report_issue.txt
```

Mục tiêu của bước này là đánh giá mô hình bằng các metric:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Confusion matrix

## 2. Mô Hình Được Đánh Giá

### Sentiment Classification

Best model được chọn từ ngày 8:

```text
Task   : Sentiment Classification
Feature: TF-IDF Unigram + Bigram
Model  : Logistic Regression
```

### Issue Classification

Best model được chọn từ ngày 9:

```text
Task   : Issue Classification
Feature: TF-IDF Unigram
Model  : Linear SVM
```

## 3. Kết Quả Sentiment Classification

Kết quả trên tập test:

```text
Accuracy: 0.8636
Macro F1: 0.8570
```

Chi tiết theo từng lớp:

```text
negative: precision 0.88, recall 0.87, f1-score 0.87
neutral : precision 0.81, recall 0.83, f1-score 0.82
positive: precision 0.88, recall 0.88, f1-score 0.88
```

Nhận xét:

- Mô hình phân loại sentiment đạt kết quả tốt và khá cân bằng giữa các lớp.
- `positive` và `negative` có F1-score cao.
- `neutral` thấp hơn một chút vì nhiều bình luận trung lập có từ như `ok`, `tạm`, `ổn`, `5 sao` hoặc nội dung nhận xu.
- Bigram giúp mô hình học được các cụm như `không tốt`, `giao chậm`, `sai màu`, `bị lỗi`.

File đã tạo:

```text
reports/classification_report_sentiment.txt
reports/confusion_matrix_sentiment.png
```

## 4. Kết Quả Issue Classification

Kết quả trên tập test:

```text
Accuracy: 0.7330
Macro F1: 0.7734
```

Chi tiết một số lớp chính:

```text
no_issue          : f1-score 0.74
packaging         : f1-score 0.76
product_quality   : f1-score 0.70
shipping_delivery : f1-score 0.79
spam_irrelevant   : f1-score 0.85
wrong_missing_item: f1-score 0.78
```

Nhận xét:

- Issue classification khó hơn sentiment classification vì số lượng nhãn nhiều hơn.
- Sau khi gom các nhãn issue quá chi tiết, Macro F1 cải thiện rõ.
- Một số nhầm lẫn giữa `no_issue` và `product_quality` vẫn còn, vì nhiều bình luận vừa mô tả sản phẩm vừa không nêu lỗi rõ ràng.
- `spam_irrelevant`, `shipping_delivery`, `wrong_missing_item` có kết quả tương đối tốt.

File đã tạo:

```text
reports/classification_report_issue.txt
reports/confusion_matrix_issue.png
```

## 5. Cách Đọc Các Metric

Accuracy:

```text
Tỷ lệ dự đoán đúng trên toàn bộ tập test.
```

Precision:

```text
Trong các mẫu được mô hình dự đoán là một nhãn, precision cho biết có bao nhiêu mẫu dự đoán đúng.
```

Recall:

```text
Trong các mẫu thật sự thuộc một nhãn, recall cho biết mô hình tìm đúng được bao nhiêu mẫu.
```

F1-score:

```text
Trung bình điều hòa giữa precision và recall.
```

Macro F1:

```text
Trung bình F1-score của tất cả các lớp, mỗi lớp có trọng số ngang nhau.
Macro F1 phù hợp với bài toán có nhiều lớp hoặc dữ liệu không cân bằng.
```

Confusion matrix:

```text
Ma trận nhầm lẫn cho biết mô hình dự đoán đúng và sai ở lớp nào.
Đường chéo chính là dự đoán đúng.
Các ô ngoài đường chéo là các trường hợp mô hình nhầm lớp.
```

## 6. Kết Luận Ngày 10

Ngày 10 đã hoàn thành bước model evaluation cho cả hai bài toán:

- Sentiment classification
- Issue classification

Các mô hình đã được đánh giá bằng classification report và confusion matrix.

Kết quả tốt nhất hiện tại:

```text
Sentiment:
Model   : Logistic Regression
Feature : TF-IDF Unigram + Bigram
Accuracy: 0.8636
Macro F1: 0.8570

Issue:
Model   : Linear SVM
Feature : TF-IDF Unigram
Accuracy: 0.7330
Macro F1: 0.7734
```

Macro F1 được dùng làm chỉ số quan trọng vì dữ liệu có nhiều lớp và phân bố nhãn không hoàn toàn cân bằng.

## 7. Bước Tiếp Theo

Ngày 11 sẽ chọn mô hình tốt nhất theo Macro F1 và lưu lại bằng `joblib`:

```text
models/sentiment_model.joblib
models/issue_model.joblib
```
