# Báo Cáo Tiến Độ Đến Ngày 9

Cập nhật: 2026-05-10

## 1. Tổng Quan Đề Tài

Đề tài xây dựng mô hình Machine Learning để phân loại bình luận khách hàng.

Dự án gồm hai bài toán học có giám sát:

1. Phân loại cảm xúc:
   `processed_text -> sentiment_label`

2. Phân loại vấn đề:
   `processed_text -> issue_label`

Đây là bài toán học có giám sát vì dữ liệu huấn luyện đã có nhãn
`sentiment_label` và `issue_label`.

## 2. Trạng Thái Dữ Liệu

File dữ liệu chính dùng để train:

```text
data/processed/merged_preprocessed_reviews.csv
```

Tổng số dòng:

```text
3517
```

Nguồn dữ liệu:

```text
shopee_real_scraped: 1323
real_review        : 1102
ai_generated       : 1092
```

Phân bố nhãn cảm xúc:

```text
positive: 1394
negative: 1381
neutral : 742
```

Phân bố nhãn vấn đề sau khi gom nhóm:

```text
product_quality   : 1064
no_issue          : 1021
product_attribute : 307
packaging         : 275
shipping_delivery : 237
wrong_missing_item: 192
seller_service    : 144
price_value       : 142
spam_irrelevant   : 124
general_feedback  : 11
```

## 3. Tóm Tắt Ngày 1 Đến Ngày 7

Đến ngày 7, dự án đã hoàn thành phần baseline:

- Chốt đề tài và xác định đây là bài toán Supervised Learning, Text Classification.
- Chuẩn bị dữ liệu bình luận khách hàng có nhãn.
- Làm sạch dữ liệu: xóa dòng trống, dòng quá ngắn, dòng trùng và chuẩn hóa nhãn.
- Xây dựng hàm tiền xử lý văn bản, giữ dấu tiếng Việt và các từ phủ định quan trọng.
- Tạo cột `processed_text` làm dữ liệu đầu vào cho mô hình.
- Phân tích phân bố nhãn.
- Train baseline sentiment classification bằng TF-IDF Unigram.

Các mô hình đã sử dụng:

- Naive Bayes
- Logistic Regression
- Linear SVM

## 4. Ngày 8: Sentiment Classification Với Bigram

Mục tiêu ngày 8:

Train lại bài toán sentiment classification với hai cách biểu diễn đặc trưng:

```text
TF-IDF Unigram
TF-IDF Unigram + Bigram
```

Đây là bước feature engineering.

TF-IDF giúp chuyển văn bản thành vector số để mô hình Machine Learning có thể học.
Bigram giúp giữ lại các cụm từ quan trọng như:

```text
không tốt
giao chậm
sai màu
bị lỗi
```

Ví dụ:

```text
"tốt"       -> có thể mang nghĩa positive
"không tốt" -> mang nghĩa negative
```

Nếu chỉ dùng unigram, mô hình chỉ nhìn từng từ riêng lẻ. Khi dùng bigram, mô hình
có thể học thêm ý nghĩa của các cụm hai từ.

Kết quả tốt nhất theo Macro F1:

```text
Feature : TF-IDF Unigram + Bigram
Model   : Logistic Regression
Accuracy: 0.8636
Macro F1: 0.8570
```

So sánh với unigram:

```text
Unigram + Logistic Regression:
Accuracy: 0.8509
Macro F1: 0.8440

Unigram + Bigram + Logistic Regression:
Accuracy: 0.8636
Macro F1: 0.8570
```

Nhận xét:

TF-IDF Unigram + Bigram giúp cải thiện kết quả sentiment classification. Điều này
cho thấy việc bổ sung bigram giúp mô hình học được các cụm từ mang ý nghĩa cảm xúc
rõ hơn so với chỉ dùng từng từ đơn.

## 5. Ngày 9: Issue Classification

Mục tiêu ngày 9:

Train bài toán phân loại thứ hai:

```text
processed_text -> issue_label
```

Input vẫn là nội dung bình luận đã tiền xử lý, nhưng output chuyển từ
`sentiment_label` sang `issue_label`.

Ví dụ:

```text
"hộp bị móp" -> packaging
```

Các mô hình đã train:

- Naive Bayes
- Logistic Regression
- Linear SVM

Các feature đã so sánh:

- TF-IDF Unigram
- TF-IDF Unigram + Bigram

## 6. Điều Chỉnh Nhãn Issue

Ban đầu, `issue_label` có quá nhiều nhãn chi tiết. Một số nhãn có rất ít dữ liệu,
làm cho mô hình khó học và làm Macro F1 thấp.

Vì vậy, các nhãn nhỏ có ý nghĩa gần nhau được gom lại thành nhóm lớn hơn.

Nhóm `product_quality` gồm:

```text
damaged_defective
product_quality
device_performance
audio_quality
camera_performance
connectivity_bluetooth
battery_charging
```

Nhóm `product_attribute` gồm:

```text
material_comfort
color_design
size_fit
```

Các nhãn được giữ nguyên:

```text
no_issue
packaging
shipping_delivery
wrong_missing_item
seller_service
price_value
spam_irrelevant
general_feedback
```

Lý do điều chỉnh:

- Một số lớp quá ít mẫu khiến mô hình không học ổn định.
- Các nhãn nhỏ có ý nghĩa gần nhau nên có thể gom lại mà vẫn giữ được ý nghĩa vấn đề.
- Sau khi gom nhãn, bài toán issue classification hợp lý hơn với kích thước dataset hiện tại.

## 7. Kết Quả Ngày 9

Kết quả tốt nhất theo Macro F1 sau khi gom nhãn:

```text
Feature : TF-IDF Unigram
Model   : Linear SVM
Accuracy: 0.7330
Macro F1: 0.7734
```

Trước khi gom nhãn:

```text
Best Macro F1: 0.6209
```

Sau khi gom nhãn:

```text
Best Macro F1: 0.7734
```

Nhận xét:

Hiệu suất issue classification cải thiện rõ sau khi gom các nhãn ít dữ liệu.
Điều này cho thấy thiết kế nhãn là một phần quan trọng trong học có giám sát.
Nếu nhãn quá chi tiết nhưng dữ liệu ít, mô hình sẽ khó học và kết quả Macro F1 thấp.

## 8. Trạng Thái Hiện Tại

Đã hoàn thành đến ngày 9:

- Ngày 8: hoàn thành so sánh TF-IDF Unigram và TF-IDF Unigram + Bigram cho sentiment classification.
- Ngày 9: hoàn thành issue classification với 3 mô hình và 2 loại feature.
- Dữ liệu issue đã được gom nhãn để phù hợp hơn với số lượng mẫu hiện có.
- Kết quả được đánh giá bằng Accuracy và Macro F1.

## 9. Bước Tiếp Theo

Ngày 10 nên tập trung vào đánh giá mô hình:

- Tạo confusion matrix cho sentiment classification.
- Tạo confusion matrix cho issue classification.
- Lưu classification report cho cả hai bài toán.
- Viết nhận xét về các lớp mô hình hay nhầm lẫn.
