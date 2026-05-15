# Báo Cáo Tiến Độ Ngày 12

Cập nhật: 2026-05-10

## 1. Mục Tiêu Ngày 12

Ngày 12 tập trung vào bước inference.

Inference nghĩa là dùng mô hình đã học để dự đoán dữ liệu mới.

Sau ngày 11, hai mô hình tốt nhất đã được lưu:

```text
models/sentiment_model.joblib
models/issue_model.joblib
```

Ngày 12 sử dụng hai file model này để tạo chương trình demo:

```text
src/predict.py
```

## 2. Cách Chạy Demo

Chạy lệnh:

```powershell
python src/predict.py
```

Sau đó nhập một bình luận khách hàng.

Ví dụ:

```text
shop giao chậm, hộp bị móp
```

## 3. Kết Quả Demo

Kết quả thực tế:

```text
Processed text: shop giao chậm hộp bị móp
Sentiment: negative
Issue: packaging
Sentiment confidence: 95.21%
```

## 4. Ý Nghĩa AI Cần Thể Hiện

Đây là bước inference:

```text
Bình luận mới -> preprocess_text -> vector TF-IDF -> model dự đoán nhãn
```

Pipeline dự đoán sentiment:

```text
review_text -> processed_text -> TF-IDF Unigram + Bigram -> Logistic Regression -> sentiment_label
```

Pipeline dự đoán issue:

```text
review_text -> processed_text -> TF-IDF Unigram -> Linear SVM -> issue_label
```

## 5. Nhận Xét

Với câu:

```text
shop giao chậm, hộp bị móp
```

Mô hình dự đoán:

```text
Sentiment: negative
Issue: packaging
```

Kết quả này hợp lý vì bình luận thể hiện trải nghiệm tiêu cực và có vấn đề về hộp/đóng gói.

Nếu một câu có nhiều vấn đề cùng lúc, ví dụ vừa `giao chậm` vừa `hộp bị móp`, mô hình hiện tại chỉ trả về một `issue_label` chính. Đây là giới hạn của bài toán single-label classification.

## 6. Kết Luận Ngày 12

Ngày 12 đã hoàn thành chương trình demo dự đoán dữ liệu mới.

File chính:

```text
src/predict.py
```

Chương trình đã load model đã lưu bằng `joblib` và dự đoán được cả:

- `sentiment_label`
- `issue_label`
