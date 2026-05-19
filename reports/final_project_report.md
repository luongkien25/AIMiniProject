# Báo cáo tổng kết dự án

Cập nhật: 19/05/2026

## 1. Mục tiêu

Dự án xây dựng mô hình Machine Learning để phân loại bình luận khách hàng trên Shopee theo hai bài toán:

- Sentiment classification: dự đoán `positive`, `neutral`, `negative`.
- Issue classification: dự đoán vấn đề chính trong bình luận, ví dụ `product_quality`, `packaging`, `wrong_missing_item`.

Đây là bài toán supervised learning vì dữ liệu huấn luyện đã có nhãn.

## 2. Dữ liệu

File dữ liệu chính dùng cho train/evaluate:

```text
data/processed/shopee_reviews_labeled.csv
```

Tổng số dòng: 6508.

Phân bố sentiment:

```text
negative: 3366
positive: 1824
neutral : 1318
```

Phân bố issue:

```text
no_issue           : 2097
product_quality    : 2073
product_attribute  : 758
wrong_missing_item : 483
packaging          : 406
seller_service     : 302
spam_irrelevant    : 146
shipping_delivery  : 126
price_value        : 117
```

## 3. Tiền xử lý

Hàm tiền xử lý chính nằm trong:

```text
src/preprocess.py
```

Các bước chính:

- Chuẩn hóa Unicode và chuyển chữ thường.
- Chuẩn hóa một số teencode phổ biến như `ko`, `k`, `dc`, `sp`.
- Xóa URL, emoji và ký tự nhiễu.
- Giữ lại các từ quan trọng cho sentiment/issue như `khong`, `chua`, `loi`, `cham`, `sai`.

## 4. Mô hình

Baseline tự cài đặt:

```text
Task   : Sentiment Classification
Feature: Unigram + Bigram counts
Model  : Multinomial Naive Bayes from scratch
Result : Accuracy 0.6974, Macro F1 0.6265
```

Baseline này được dùng để minh họa nguyên lý học có giám sát trong text classification: đếm token theo từng lớp, tính prior/likelihood với Laplace smoothing, sau đó dự đoán bằng tổng log-probability. Model final vẫn dùng scikit-learn vì cho kết quả tốt và ổn định hơn.

Trong bản nộp, code chỉ giữ hai hướng mô hình chính: Naive Bayes để làm baseline/giải thích nguyên lý và Linear SVM làm mô hình mạnh nhất. File `src/naive_bayes_model.py` chỉ chứa phần thuật toán Naive Bayes tự cài đặt; việc train và so sánh được đặt trong `src/train_sentiment.py` và `src/train_issue.py`.

Thí nghiệm phụ với bài toán chỉ gồm `negative` và `positive` đạt Accuracy 0.8179 và Macro F1 0.7676. Kết quả này cao hơn baseline 3 lớp vì lớp `neutral` đã bị loại bỏ, nhưng không thay thế được bài toán chính do hệ thống vẫn cần dự đoán đủ 3 sentiment.

Sentiment classification:

```text
Feature: TF-IDF Unigram + Bigram
Model  : Linear SVM
```

Issue classification:

```text
Feature: TF-IDF Word + Char
Model  : Linear SVM + Safe Issue Rules
```

Issue prediction có thêm một số rule an toàn để xử lý các trường hợp rõ ràng như giao sai, thiếu hàng, đóng gói lỗi, giao chậm hoặc nội dung nhận xu.

## 5. Kết quả

Sentiment classification:

```text
Accuracy: 0.8318
Macro F1: 0.7993
```

Issue classification:

```text
Accuracy: 0.7849
Macro F1: 0.7323
```

Các báo cáo chi tiết và confusion matrix nằm trong:

```text
reports/classification_report_sentiment.txt
reports/classification_report_issue.txt
reports/confusion_matrix_sentiment.png
reports/confusion_matrix_issue.png
```

## 6. Các file chính

```text
src/naive_bayes_model.py     Thuật toán Naive Bayes tự cài đặt
src/prepare_data.py      Kiểm tra hoặc tạo lại cột cleaned_review
src/train_sentiment.py   Train và so sánh Naive Bayes/Linear SVM cho sentiment
src/train_issue.py       Train và so sánh Naive Bayes/Linear SVM cho issue
src/evaluate.py          Tạo classification report và confusion matrix
src/save_models.py       Train lại trên toàn bộ dữ liệu và lưu model
src/predict.py           Demo dự đoán bình luận mới
```

Model đã lưu:

```text
models/sentiment_model.joblib
models/issue_model.joblib
```

## 7. Giới hạn

Mô hình issue hiện là single-label classification, nên mỗi bình luận chỉ trả về một issue chính. Với bình luận có nhiều vấn đề cùng lúc, hệ thống chọn issue theo độ ưu tiên.

Các lớp có ít mẫu như `price_value`, `shipping_delivery`, `spam_irrelevant` vẫn khó hơn các lớp lớn. Hướng cải thiện tiếp theo là bổ sung thêm dữ liệu đã review thủ công cho các lớp này.
