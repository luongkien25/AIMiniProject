# Customer Review Sentiment and Issue Classification

Project này xây dựng mô hình Machine Learning để phân loại bình luận khách hàng theo hai nhiệm vụ:

1. Sentiment classification:
   - Positive
   - Negative
   - Neutral

2. Issue classification:
   - no_issue
   - product_quality
   - delivery
   - packaging
   - wrong_item
   - customer_service
   - price
   - spam_or_irrelevant

## Folder Structure

```text
customer-review-classification/
├── data/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── reports/
├── slides/
├── README.md
└── requirements.txt
```

## Current Data Files

- `data/processed/cleaned_reviews.csv`: dữ liệu đã làm sạch cơ bản.
- `data/processed/preprocessed_reviews.csv`: dữ liệu có thêm cột `processed_text` để train model.

## Labels

Sentiment labels:

- `Positive`
- `Negative`
- `Neutral`

Issue labels:

- `no_issue`
- `product_quality`
- `delivery`
- `packaging`
- `wrong_item`
- `customer_service`
- `price`
- `spam_or_irrelevant`
