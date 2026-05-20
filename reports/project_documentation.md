# Tài Liệu Mô Tả Dự Án

## 1. Tên đề tài

**Customer Review Sentiment and Issue Classification**

Đề tài xây dựng hệ thống học máy để phân tích bình luận khách hàng trên Shopee. Với mỗi bình luận đầu vào, hệ thống dự đoán hai thông tin chính:

- Cảm xúc của khách hàng đối với sản phẩm hoặc dịch vụ.
- Vấn đề chính được nhắc tới trong bình luận.

## 2. Mô tả bài toán

Trong các sàn thương mại điện tử như Shopee, mỗi sản phẩm có thể nhận được rất nhiều bình luận từ khách hàng. Các bình luận này chứa nhiều thông tin quan trọng về trải nghiệm mua hàng, ví dụ sản phẩm có tốt không, giao hàng có nhanh không, đóng gói có cẩn thận không, shop có hỗ trợ tốt không, hoặc sản phẩm có đúng mô tả không.

Nếu xử lý thủ công, người bán hoặc người phân tích dữ liệu phải đọc từng bình luận rồi tự phân loại. Cách làm này tốn thời gian, khó mở rộng khi số lượng review lớn và dễ thiếu nhất quán giữa các lần đánh giá. Vì vậy, project đặt mục tiêu xây dựng một hệ thống có khả năng tự động phân loại bình luận khách hàng bằng học máy.

Bài toán được chia thành hai nhiệm vụ chính.

Nhiệm vụ thứ nhất là **sentiment classification**, tức là phân loại cảm xúc của bình luận. Mỗi bình luận được gán vào một trong ba nhãn:

```text
positive: bình luận tích cực
neutral : bình luận trung lập
negative: bình luận tiêu cực
```

Ví dụ:

```text
"Sản phẩm đẹp, đúng mô tả, giao hàng nhanh"
→ positive

"Chưa dùng thử nên chưa biết chất lượng thế nào"
→ neutral

"Hàng bị rách, giao chậm, rất thất vọng"
→ negative
```

Nhiệm vụ thứ hai là **issue classification**, tức là phân loại vấn đề chính trong bình luận. Mỗi bình luận được gán một nhãn issue đại diện cho nội dung chính mà khách hàng nhắc tới. Các nhãn issue trong project gồm:

```text
no_issue           : không có vấn đề rõ ràng
product_quality    : vấn đề về chất lượng sản phẩm
product_attribute  : vấn đề về màu sắc, size, chất liệu, kiểu dáng
packaging          : vấn đề về đóng gói
wrong_missing_item : giao sai hàng, thiếu hàng, nhầm phân loại
seller_service     : vấn đề về thái độ hoặc hỗ trợ của shop
shipping_delivery  : vấn đề về vận chuyển, giao hàng
price_value        : vấn đề về giá trị so với giá tiền
spam_irrelevant    : bình luận spam hoặc không liên quan
```

Ví dụ:

```text
"Hộp bị móp, đóng gói sơ sài"
→ packaging

"Đặt màu đen nhưng shop giao màu trắng"
→ wrong_missing_item

"Vải mỏng hơn trong ảnh, form hơi rộng"
→ product_attribute

"Shop không trả lời tin nhắn, hỗ trợ rất kém"
→ seller_service
```

Đây là bài toán **học máy có giám sát** vì dữ liệu huấn luyện đã có nhãn đúng cho sentiment và issue. Mô hình học từ các cặp dữ liệu gồm bình luận và nhãn tương ứng, sau đó dự đoán nhãn cho các bình luận mới.

## 3. Đặc điểm dữ liệu

Dữ liệu đầu vào là bình luận tiếng Việt trên Shopee. Loại dữ liệu này có nhiều đặc điểm gây khó khăn cho mô hình:

- Câu thường ngắn, thiếu ngữ cảnh.
- Có lỗi chính tả, viết tắt và teencode như `ko`, `k`, `dc`, `sp`.
- Có emoji, ký tự đặc biệt, số điểm đánh giá hoặc nội dung thừa từ trang sản phẩm.
- Một bình luận có thể chứa nhiều ý cùng lúc.
- Một số lớp có ít dữ liệu hơn các lớp khác, gây mất cân bằng nhãn.

Ví dụ một bình luận có thể vừa nói:

```text
"giao chậm, hộp bị móp, sản phẩm dùng ổn"
```

Bình luận này có nhiều thông tin: giao hàng, đóng gói và chất lượng sản phẩm. Tuy nhiên, trong project hiện tại, bài toán issue được xử lý theo hướng **single-label classification**, tức là mỗi bình luận chỉ chọn một issue chính.

## 4. Nguồn dữ liệu

Dữ liệu của project được thu thập từ các bình luận công khai trên Shopee. Các bình luận được lấy từ trang sản phẩm Shopee bằng script hỗ trợ trong thư mục `tools/`.

File lấy dữ liệu chính:

```text
tools/take_shopee_data.js
```

Script này được chạy trong trình duyệt trên trang sản phẩm Shopee để quét phần bình luận hiển thị, lọc các bình luận trùng nhau và xuất ra file CSV. File CSV đầu ra có dạng:

```csv
review_id,review_text
1,"shop giao chậm, hộp bị móp"
2,"sản phẩm đúng mô tả, rất đẹp"
```

Trong đó:

- `review_id`: mã thứ tự của bình luận, tự đánh số từ 1.
- `review_text`: nội dung bình luận gốc của khách hàng.

Các file dữ liệu thô được lưu trong thư mục:

```text
data/raw/Shopee/
```

File dữ liệu chính dùng để train và evaluate mô hình:

```text
data/processed/shopee_reviews_labeled.csv
```

File này gồm 6508 dòng và các cột chính:

```text
review_text    : bình luận gốc
cleaned_review : bình luận sau tiền xử lý
sentiment      : nhãn cảm xúc
issue          : nhãn vấn đề chính
data_source    : nguồn hoặc nhóm dữ liệu
```

Dữ liệu sau khi thu thập được gán nhãn theo guideline của project. Nhãn `sentiment` dùng cho bài toán phân loại cảm xúc, còn nhãn `issue` dùng cho bài toán phân loại vấn đề chính.

Ngoài dữ liệu huấn luyện đã gán nhãn, project còn có thể phân tích file bình luận mới chưa có nhãn bằng script:

```text
src/analyze_reviews.py
```

Script này nhận file CSV có dạng `review_id,review_text`, sau đó dùng mô hình đã lưu để dự đoán sentiment và issue cho từng bình luận. Kết quả đầu ra gồm:

```text
review_id
review_text
sentiment
issue
```

Script cũng tạo thêm file tổng hợp thống kê số lượng và tỉ lệ theo sentiment, issue và cặp sentiment-issue.

## 5. Công cụ và thư viện sử dụng

Project sử dụng các công cụ và thư viện chính sau.

**Python**

Python là ngôn ngữ chính dùng để xử lý dữ liệu, huấn luyện mô hình, đánh giá và chạy dự đoán.

**pandas**

Dùng để đọc, xử lý và lưu dữ liệu dạng CSV. pandas hỗ trợ thao tác với bảng dữ liệu, kiểm tra cột, lọc dữ liệu và chuẩn bị dữ liệu cho mô hình.

**scikit-learn**

Dùng cho các bước học máy chính:

- Chia train/test bằng `train_test_split`.
- Biểu diễn văn bản bằng `TfidfVectorizer`.
- Huấn luyện mô hình `LinearSVC`.
- Tính các metric như Accuracy, Macro F1 và confusion matrix.

**joblib**

Dùng để lưu và load mô hình đã huấn luyện:

```text
models/sentiment_model.joblib
models/issue_model.joblib
```

Nhờ đó project có thể train model một lần, lưu lại, rồi dùng lại cho dự đoán bình luận mới.

**matplotlib và seaborn**

Dùng để trực quan hóa kết quả, đặc biệt là confusion matrix cho sentiment và issue classification.

**Jupyter Notebook**

Dùng để khám phá dữ liệu, kiểm tra phân bố nhãn và trình bày một số kết quả phân tích.

Notebook chính:

```text
notebooks/01_exploration.ipynb
```

**JavaScript trong trình duyệt**

Dùng để hỗ trợ lấy bình luận từ trang Shopee. Script `tools/take_shopee_data.js` được paste vào console của trình duyệt khi đang mở trang sản phẩm Shopee, sau đó tự động thu thập bình luận và tải xuống file CSV.

**Git và GitHub**

Dùng để quản lý phiên bản mã nguồn, theo dõi thay đổi trong project và lưu trữ toàn bộ source code.

**PowerPoint**

Dùng để trình bày kết quả project, mô tả bài toán, phương pháp, mô hình và kết quả thực nghiệm.

## 6. Quy trình xử lý tổng quát

Quy trình xử lý của project gồm các bước chính:

```text
Thu thập bình luận Shopee
→ gán nhãn sentiment và issue
→ tiền xử lý văn bản
→ biểu diễn văn bản thành vector
→ huấn luyện mô hình
→ đánh giá trên tập test
→ lưu model
→ dự đoán bình luận mới
```

Ở bước tiền xử lý, bình luận được chuẩn hóa để giảm nhiễu. Các bước chính gồm:

- Chuẩn hóa Unicode.
- Chuyển chữ thường.
- Loại bỏ URL, emoji và ký tự không cần thiết.
- Chuẩn hóa một số teencode phổ biến.
- Giữ lại các từ quan trọng cho việc phân loại như `khong`, `chua`, `loi`, `cham`, `sai`.

Sau tiền xử lý, văn bản được chuyển thành vector số. Với mô hình chính, project sử dụng TF-IDF để biểu diễn review. TF-IDF giúp tăng trọng số cho những từ quan trọng trong một bình luận nhưng không quá phổ biến trong toàn bộ tập dữ liệu.

## 7. Mô hình sử dụng

Project giữ hai hướng mô hình chính.

**Multinomial Naive Bayes**

Đây là mô hình baseline được tự cài đặt trong:

```text
src/naive_bayes_model.py
```

Mô hình này dùng unigram và bigram đếm tần suất. Vai trò của Naive Bayes là tạo mốc so sánh đơn giản và minh họa nguyên lý xác suất trong text classification.

Công thức chính:

```text
y_hat = argmax_c [log P(c) + Σ_t x_t log P(t|c)]
```

Với Laplace smoothing:

```text
P(t|c) = (count(t,c) + alpha) / (Σ_v count(v,c) + alpha|V|)
```

**TF-IDF + Linear SVM**

Đây là mô hình chính của project. Văn bản được chuyển thành vector TF-IDF, sau đó đưa vào Linear SVM để phân loại.

Với sentiment classification:

```text
Feature: TF-IDF word unigram + bigram
Model  : Linear SVM
```

Với issue classification:

```text
Feature: TF-IDF word/char n-grams
Model  : Linear SVM + safe issue rules
```

Linear SVM học một siêu phẳng để chia không gian vector thành các vùng tương ứng với từng lớp. Với mỗi lớp, mô hình tính một điểm:

```text
f_c(x) = w_c^T x + b_c
```

Lớp có điểm cao nhất được chọn làm nhãn dự đoán:

```text
y_hat = argmax_c f_c(x)
```

**Safe issue rules**

Với bài toán issue, project bổ sung một lớp rule an toàn cho các trường hợp rất rõ ràng, ví dụ:

- Giao sai hàng.
- Thiếu hàng.
- Giao chậm.
- Hộp bị móp, bao bì rách.
- Bình luận spam nhận xu.

Rule không thay thế mô hình học máy mà chỉ xử lý các pattern chắc chắn để giảm một số lỗi dễ nhận biết.

## 8. Cách chia train/test và đánh giá

Dữ liệu được chia thành train và test theo tỉ lệ 80/20.

Trong code, project dùng:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=62,
    stratify=y,
)
```

Ý nghĩa:

- `test_size=0.2`: 20% dữ liệu dùng để test, 80% dùng để train.
- `random_state=62`: cố định cách chia để kết quả có thể lặp lại.
- `stratify=y`: giữ tỉ lệ các lớp trong train/test gần giống dữ liệu gốc.

Mô hình chỉ học trên tập train:

```python
model.fit(X_train, y_train)
```

Sau đó dự đoán trên tập test:

```python
y_pred = model.predict(X_test)
```

Như vậy dữ liệu test không được dùng trong quá trình huấn luyện. Tập test chỉ được dùng để đánh giá sau khi model đã train xong.

Các metric chính:

```text
Accuracy
Macro F1
Confusion matrix
```

Accuracy đo tỉ lệ dự đoán đúng tổng thể:

```text
Accuracy = số mẫu dự đoán đúng / tổng số mẫu test
```

Macro F1 là trung bình F1 trên tất cả các lớp:

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Macro F1 quan trọng vì dữ liệu có sự mất cân bằng nhãn. Nếu chỉ dùng Accuracy, mô hình có thể đạt điểm cao bằng cách dự đoán tốt các lớp lớn nhưng bỏ qua các lớp ít mẫu.

## 9. Kết quả hiện tại

Kết quả tốt nhất của sentiment classification:

```text
Model    : TF-IDF word unigram + bigram + Linear SVM
Accuracy : 0.8410
Macro F1 : 0.8093
```

Kết quả tốt nhất của issue classification:

```text
Model    : TF-IDF word/char n-grams + Linear SVM + safe issue rules
Accuracy : 0.7849
Macro F1 : 0.7323
```

Baseline Naive Bayes tự cài đặt cho sentiment:

```text
Model    : From-scratch Multinomial Naive Bayes
Feature  : unigram + bigram counts
Accuracy : 0.6974
Macro F1 : 0.6265
```

Kết quả cho thấy Linear SVM hoạt động tốt hơn Naive Bayes trên dữ liệu review tiếng Việt của project. Nguyên nhân là TF-IDF + Linear SVM tận dụng tốt dữ liệu vector thưa nhiều chiều và học trọng số phân biệt cho nhiều từ/cụm từ cùng lúc.

## 10. Các file chính trong project

```text
src/preprocess.py          Tiền xử lý văn bản
src/prepare_data.py        Kiểm tra hoặc tạo lại cột cleaned_review
src/naive_bayes_model.py   Cài đặt Multinomial Naive Bayes từ đầu
src/train_sentiment.py     Train và đánh giá sentiment model
src/train_issue.py         Train và đánh giá issue model
src/evaluate.py            Tính metric và tạo confusion matrix
src/save_models.py         Train trên toàn bộ dữ liệu và lưu model
src/predict.py             Demo dự đoán một bình luận nhập từ bàn phím
src/analyze_reviews.py     Phân tích hàng loạt bình luận từ file CSV
tools/take_shopee_data.js  Lấy bình luận Shopee từ trình duyệt
```

File dữ liệu chính:

```text
data/processed/shopee_reviews_labeled.csv
```

File model đã lưu:

```text
models/sentiment_model.joblib
models/issue_model.joblib
```

File báo cáo và hình ảnh kết quả:

```text
reports/final_project_report.md
reports/label_guidelines_report.md
reports/confusion_matrix_sentiment.png
reports/confusion_matrix_issue.png
```

## 11. Hạn chế và hướng phát triển

Project hiện tại vẫn có một số hạn chế.

Thứ nhất, issue classification đang là single-label classification. Trong thực tế, một bình luận có thể nhắc tới nhiều vấn đề cùng lúc, ví dụ vừa giao chậm vừa sản phẩm lỗi. Khi đó mô hình chỉ trả về một issue chính nên chưa biểu diễn đầy đủ toàn bộ nội dung bình luận.

Thứ hai, dữ liệu giữa các lớp chưa cân bằng. Một số lớp như `price_value`, `shipping_delivery`, `spam_irrelevant` có ít mẫu hơn các lớp lớn như `no_issue` và `product_quality`, làm mô hình khó học tốt các lớp nhỏ.

Thứ ba, dữ liệu review thực tế có nhiều nhiễu như lỗi chính tả, teencode, nội dung thừa từ trang sản phẩm hoặc review quá ngắn. Điều này làm một số trường hợp bị nhầm giữa các lớp gần nhau, ví dụ `product_quality` và `product_attribute`.

Hướng phát triển tiếp theo:

- Bổ sung thêm dữ liệu đã gán nhãn cho các lớp ít mẫu.
- Chuyển issue classification từ single-label sang multi-label.
- Cải thiện tiền xử lý tiếng Việt và chuẩn hóa teencode.
- Thử nghiệm các mô hình ngôn ngữ mạnh hơn nếu yêu cầu project cho phép.
- Xây dựng giao diện nhỏ để nhập hoặc upload file bình luận và xem kết quả trực quan hơn.
