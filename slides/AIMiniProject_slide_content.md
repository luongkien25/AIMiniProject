# Nội dung trình bày AIMiniProject

## Slide 1. Title

**Customer Review Sentiment and Issue Classification**

AI Mini Project - Phân loại bình luận Shopee

Người trình bày: `[Tên của bạn]`

## Slide 2. Abstract

- Dự án xây dựng pipeline học máy có giám sát cho bình luận khách hàng trên Shopee.
- Bài toán gồm hai nhiệm vụ: phân loại cảm xúc và phân loại vấn đề chính trong bình luận.
- Pipeline sử dụng tiền xử lý tiếng Việt, đặc trưng TF-IDF, baseline Naive Bayes, Linear SVM và một lớp rule an toàn.
- Kết quả cuối: sentiment Accuracy 0.8318 / Macro F1 0.7993; issue Accuracy 0.7849 / Macro F1 0.7323.

**Gợi ý trình bày:** Dự án biến các bình luận ngắn, nhiễu và không chuẩn hóa thành hai tín hiệu hữu ích: khách hàng đang cảm thấy thế nào và họ đang gặp vấn đề gì.

## Slide 3. Introduction

- Sàn thương mại điện tử nhận rất nhiều bình luận sản phẩm mỗi ngày.
- Bình luận thường ngắn, informal, có teencode, emoji và lỗi chính tả.
- Phân tích thủ công tốn thời gian và khó nhất quán.
- Mục tiêu của project:
  - Dự đoán sentiment: `positive`, `neutral`, `negative`.
  - Dự đoán issue chính: `no_issue`, `product_quality`, `packaging`, `shipping_delivery`, ...
- Đây là bài toán supervised text classification vì dữ liệu huấn luyện đã có nhãn.

**Gợi ý trình bày:** Mình tập trung vào một hệ thống thực dụng, có thể hỗ trợ seller hoặc người quản lý đọc nhanh phản hồi khách hàng.

## Slide 4. Dataset and Labels

- File dữ liệu chính: `data/processed/shopee_reviews_labeled.csv`.
- Tổng số mẫu: 6508 bình luận.
- Cột chính: `review_text`, `cleaned_review`, `sentiment`, `issue`, `data_source`.
- Phân bố sentiment:
  - negative: 3366
  - positive: 1824
  - neutral: 1318
- Issue có 9 nhãn; hai nhóm lớn nhất là `no_issue` và `product_quality`.

**Gợi ý trình bày:** Dữ liệu không cân bằng, nên ngoài Accuracy mình dùng thêm Macro F1 để đánh giá công bằng hơn giữa các lớp.

## Slide 5. Related Work

- Rule-based classification: dễ giải thích và tốt với pattern rõ ràng, nhưng khó bao phủ hết trường hợp thực tế.
- Bag-of-words và TF-IDF: cách biểu diễn văn bản phổ biến cho các mô hình học máy truyền thống.
- Naive Bayes: baseline đơn giản, phù hợp để minh họa nguyên lý xác suất trong text classification.
- Linear SVM: thường hoạt động tốt với vector TF-IDF thưa và là mô hình mạnh nhất trong bản nộp.
- Hybrid approach: mô hình học máy xử lý trường hợp tổng quát, rule sửa các pattern chắc chắn.

**Gợi ý trình bày:** Project không chỉ dùng rule hoặc chỉ dùng model; mình kết hợp hai hướng để tận dụng điểm mạnh của mỗi cách.

## Slide 6. Method Overview

Pipeline:

1. Thu thập và gán nhãn review Shopee.
2. Tiền xử lý văn bản review.
3. Chia train/test theo tỉ lệ 80/20 và stratify theo nhãn.
4. Chuyển văn bản thành vector TF-IDF.
5. Train và so sánh Naive Bayes với Linear SVM.
6. Áp dụng safe issue rules cho các lỗi rõ ràng.
7. Đánh giá bằng Accuracy, Macro F1, classification report và confusion matrix.

**Gợi ý trình bày:** Điểm chính là pipeline có thể chạy lại được bằng script, không chỉ là một thử nghiệm thủ công trong notebook.

## Slide 7. Preprocessing and Features

- File tiền xử lý chính: `src/preprocess.py`.
- Các bước chính:
  - Chuẩn hóa Unicode và chuyển chữ thường.
  - Chuẩn hóa teencode thương mại điện tử như `ko`, `k`, `dc`, `sp`.
  - Xóa URL, emoji và ký tự nhiễu.
  - Giữ lại từ quan trọng như `khong`, `chua`, `loi`, `cham`, `sai`.
- Đặc trưng:
  - Sentiment: TF-IDF unigram + bigram.
  - Issue: TF-IDF word + character n-grams.

**Gợi ý trình bày:** Character n-grams giúp mô hình issue chịu được lỗi chính tả, teencode và các biến thể viết ngắn trong tiếng Việt.

## Slide 8. Models

- Baseline:
  - Multinomial Naive Bayes tự cài đặt.
  - Đặc trưng unigram + bigram count.
  - Dùng để minh họa nguyên lý học có giám sát.
- Các mô hình được giữ trong code nộp:
  - Multinomial Naive Bayes.
  - Linear SVM.
- Mô hình cuối:
  - Sentiment: TF-IDF unigram + bigram + Linear SVM.
  - Issue: TF-IDF word + char + Linear SVM + safe issue rules.

**Gợi ý trình bày:** Linear SVM cho kết quả tốt nhất trên vector TF-IDF thưa trong cả hai bài toán chính.

## Slide 9. Experiments

- Train/test split: 80/20, stratified by label.
- Random state: 62.
- Metric:
  - Accuracy đo tỉ lệ dự đoán đúng tổng thể.
  - Macro F1 đo chất lượng trung bình trên các lớp, hữu ích khi dữ liệu lệch nhãn.
- Sentiment best:
  - TF-IDF unigram + bigram + Linear SVM.
  - Accuracy 0.8318, Macro F1 0.7993.
- Issue best ML-only:
  - TF-IDF word + char + Linear SVM.
  - Accuracy 0.7773, Macro F1 0.7101.
- Final issue thêm safe rules và tăng lên Accuracy 0.7849, Macro F1 0.7323.

**Gợi ý trình bày:** Mình giữ code nộp gọn: Naive Bayes để giải thích nguyên lý, Linear SVM vì cho kết quả tốt nhất trên dữ liệu hiện tại.

## Slide 10. Sentiment Results

- Mô hình cuối: TF-IDF unigram + bigram + Linear SVM.
- Accuracy: 0.8318.
- Macro F1: 0.7993.
- F1 theo lớp:
  - negative: 0.89
  - neutral: 0.66
  - positive: 0.85
- Lớp khó nhất là `neutral` vì nhiều bình luận nằm giữa khen nhẹ, chê nhẹ hoặc chỉ nhận xu.

**Gợi ý trình bày:** Mô hình xử lý tốt các review tích cực hoặc tiêu cực rõ ràng; neutral vẫn là phần dễ nhầm nhất.

## Slide 11. Issue Results

- Mô hình cuối: TF-IDF word + char + Linear SVM + safe issue rules.
- Accuracy: 0.7849.
- Macro F1: 0.7323.
- Nhóm có F1 tốt:
  - `no_issue`: 0.86
  - `product_quality`: 0.80
  - `seller_service`: 0.81
  - `wrong_missing_item`: 0.79
- Nhóm còn khó:
  - `product_attribute`: 0.59
  - `spam_irrelevant`: 0.60
  - `price_value`: ít mẫu, F1 0.67

**Gợi ý trình bày:** Issue classification khó hơn sentiment vì số nhãn nhiều hơn và một số nhãn có ý nghĩa gần nhau.

## Slide 12. Error Analysis

- Sentiment:
  - `neutral` thường bị dự đoán thành `negative`.
  - Một số `positive` bị dự đoán thành `neutral`.
- Issue:
  - `product_quality` và `product_attribute` dễ nhầm với nhau.
  - Một số `no_issue` bị nhầm sang issue liên quan đến sản phẩm.
- Nguyên nhân:
  - Review ngắn và thiếu ngữ cảnh.
  - Cách viết informal, nhiều biến thể.
  - Một review có thể chứa nhiều vấn đề.
  - Bài toán issue hiện là single-label nên chỉ chọn một issue chính.

**Gợi ý trình bày:** Các lỗi này hợp lý với dữ liệu review thật, vì một câu có thể vừa mô tả thuộc tính sản phẩm vừa nói chất lượng sản phẩm.

## Slide 13. What I Implemented

- Chuẩn bị và làm sạch dữ liệu bình luận Shopee đã gán nhãn.
- Xây dựng tiền xử lý trong `src/preprocess.py`.
- Cài baseline Naive Bayes từ đầu trong `src/baseline_naive_bayes.py`.
- Train và so sánh mô hình trong:
  - `src/train_sentiment.py`
  - `src/train_issue.py`
- Tạo report và confusion matrix trong `src/evaluate.py`.
- Lưu model cuối trong thư mục `models/`.
- Viết demo dự đoán bình luận mới trong `src/predict.py`.

**Gợi ý trình bày:** Đây là slide tóm tắt trực tiếp những phần mình đã làm trong project.

## Slide 14. Demo

Input ví dụ:

`shop giao cham, hop bi mop`

Prediction:

- Processed text: `shop giao cham hop bi mop`
- Sentiment: `negative`
- Issue: `packaging`
- Sentiment rule: `clear_negative_rule`
- Issue rule: `packaging_priority_rule`

**Gợi ý trình bày:** Ví dụ này cho thấy model và rule layer cùng hoạt động khi review có khiếu nại rõ ràng.

## Slide 15. Conclusion

- Project đã xây dựng một pipeline hoàn chỉnh để phân loại bình luận Shopee.
- TF-IDF + Linear SVM hoạt động tốt cho cả sentiment và issue classification.
- Safe issue rules giúp cải thiện kết quả issue classification cuối.
- Giới hạn hiện tại:
  - Issue đang là single-label.
  - Một số lớp có ít dữ liệu.
  - Neutral sentiment vẫn khó.
- Hướng phát triển:
  - Bổ sung dữ liệu review thủ công cho lớp nhỏ.
  - Mở rộng sang multi-label issue prediction.
  - Thử mô hình ngôn ngữ tiếng Việt pretrained để so sánh.

**Gợi ý trình bày:** Kết quả hiện tại là một baseline học máy truyền thống khá chắc và có thể mở rộng lên các mô hình mạnh hơn sau này.
