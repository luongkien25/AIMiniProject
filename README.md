# Customer Review Sentiment and Issue Classification

Dự án xây dựng mô hình Machine Learning để phân loại bình luận khách hàng trên Shopee theo hai nhiệm vụ:

- Sentiment classification: `positive`, `neutral`, `negative`.
- Issue classification: phân loại vấn đề chính trong bình luận.

## Cấu trúc thư mục

```text
AIMiniProject/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
├── tools/
├── README.md
└── requirements.txt
```

## Dữ liệu chính

File train/evaluate chính:

```text
data/processed/shopee_reviews_labeled.csv
```

File này có 6508 dòng và các cột chính:

- `review_text`: bình luận gốc.
- `cleaned_review`: văn bản sau tiền xử lý.
- `sentiment`: nhãn cảm xúc.
- `issue`: nhãn vấn đề chính.
- `data_source`: nguồn dữ liệu.

## Nhãn

Sentiment labels:

```text
positive
neutral
negative
```

Issue labels:

```text
no_issue
product_quality
product_attribute
wrong_missing_item
packaging
seller_service
shipping_delivery
price_value
spam_irrelevant
```

## Cài đặt

Tạo môi trường ảo và cài thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu đã có `.venv`, chỉ cần activate trước khi chạy script.

## Cách chạy

Kiểm tra hoặc tạo lại cột `cleaned_review`:

```powershell
python src/prepare_data.py
```

Train Naive Bayes baseline và mô hình chính Linear SVM:

```powershell
python src/train_sentiment.py
python src/train_issue.py
```

In metric đánh giá và tạo lại confusion matrix:

```powershell
python src/evaluate.py
```

Notebook EDA và confusion matrix:

```text
notebooks/01_exploration.ipynb
```

Train trên toàn bộ dữ liệu và lưu model:

```powershell
python src/save_models.py
```

Chạy demo dự đoán bình luận mới:

```powershell
python src/predict.py
```

Ví dụ input:

```text
shop giao cham, hop bi mop
```

## Mô hình đang dùng

Để giữ project gọn cho bản nộp, code chỉ giữ lại hai hướng chính:

- `Naive Bayes`: baseline đơn giản, tự cài đặt trong `src/naive_bayes_model.py`.
- `Linear SVM`: mô hình mạnh nhất hiện tại, dùng làm model final.

File `src/naive_bayes_model.py` chỉ chứa phần model/thuật toán. Việc đọc dữ liệu, train và so sánh metric được đặt trong `src/train_sentiment.py` và `src/train_issue.py`.

Baseline tự cài đặt để minh họa nguyên lý thuật toán:

```text
Sentiment baseline: Multinomial Naive Bayes from scratch
Feature           : Unigram + Bigram counts
Accuracy          : 0.6974
Macro F1          : 0.6265
```

Model chính:

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

Model đã lưu:

```text
models/sentiment_model.joblib
models/issue_model.joblib
```

## Kết quả hiện tại

Sentiment classification:

```text
Accuracy: 0.8410
Macro F1: 0.8093
```

Issue classification:

```text
Accuracy: 0.7849
Macro F1: 0.7323
```

Report chính:

```text
reports/final_project_report.md
reports/label_guidelines_report.md
reports/confusion_matrix_sentiment.png
reports/confusion_matrix_issue.png
```

## Ghi chú

Các file audit tạm, script tạo dữ liệu trung gian cũ và report tiến độ theo ngày đã được loại khỏi bản nộp để repo gọn hơn. Các script chính để prepare data, train, evaluate, save model và predict nằm trong `src/`.
