# Bao cao dinh nghia nhan va quy tac gan nhan

Ngay cap nhat: 17/05/2026

## 1. Muc tieu

Du an phan loai binh luan Shopee theo 2 bai toan:

- Sentiment classification: phan loai cam xuc thanh `positive`, `neutral`, `negative`.
- Issue classification: phan loai van de chinh cua binh luan thanh mot trong cac nhan issue.

Vi mo hinh issue hien tai la single-label, moi binh luan chi duoc gan 1 issue chinh. Do do can co quy tac uu tien khi mot binh luan co nhieu van de.

## 2. Dinh nghia sentiment

### 2.1. `positive`

Binh luan the hien su hai long ro rang, co loi khen, hoac xac nhan san pham dung ky vong.

Vi du:

```text
san pham dep, chat luong tot
giao hang nhanh, dong goi can than
da nhan du hang
san pham nhu hinh
dung mo ta
dang tien, se mua lai
```

### 2.2. `neutral`

Binh luan trung tinh, chu yeu mo ta trang thai, chua danh gia ro tot hay xau, hoac co khen che nhe nhung tong the chua nghieng ro.

Vi du:

```text
da nhan hang chua dung
mau trang, chat lieu vai
tam duoc
binh thuong
ok nhung hoi mong
mau hoi khac anh nhung van dung duoc
nhan xu, hinh anh chi mang tinh minh hoa
```

### 2.3. `negative`

Binh luan the hien su khong hai long, co loi, giao sai hoac thieu hang, chat luong kem, shop khong ho tro, giao hang cham, hoac san pham khong dung ky vong.

Vi du:

```text
san pham bi loi
giao sai mau
thieu hang
dong goi so sai
shop khong phan hoi
khong dang tien
giao hang qua cham
```

## 3. Dinh nghia issue

### 3.1. `no_issue`

Khong co van de cu the. Review chi khen, mo ta san pham, xac nhan nhan hang, hoac nhan xet binh thuong khong neu loi.

Vi du:

```text
mau trang, chat lieu vai
da nhan hang
san pham dung mo ta
hang dep, giao nhanh
```

### 3.2. `product_quality`

Van de ve chat luong, do ben, chuc nang, hu hong, khong su dung duoc, hoac san pham kem.

Vi du:

```text
san pham bi hong
pin nhanh het
khong dung duoc
chat luong te
moi dung da loi
vai rach, nhua de gay
```

### 3.3. `product_attribute`

San pham lech thuoc tinh so voi ky vong hoac mo ta, nhung khong phai truong hop giao nham don hang hoac giao sai phan loai ro rang.

Vi du:

```text
mau hoi khac anh
size hoi nho
vai mong hon tuong tuong
form khong chuan
chat lieu khong giong mo ta
```

### 3.4. `wrong_missing_item`

Giao sai, giao nham, giao thieu, thieu phu kien, hoac khach dat A nhung nhan B.

Vi du:

```text
dat mau trang giao mau den
shop giao sai mau
giao thieu hang
thieu day sac
dat size L giao size M
mua 2 cai giao 1 cai
```

### 3.5. `packaging`

Van de ve dong goi, bao bi, hop, chong soc. Nhan nay dung khi loi chinh nam o khau dong goi.

Vi du:

```text
dong goi so sai
hop bi mop
bao bi rach
khong co chong soc
boc hang khong can than
```

### 3.6. `seller_service`

Van de ve thai do, phan hoi, tu van, ho tro, hoac xu ly khieu nai cua shop.

Vi du:

```text
shop khong rep
shop khong ho tro
tu van sai
khong giai quyet khieu nai
thai do shop te
```

### 3.7. `shipping_delivery`

Van de ve giao hang hoac van chuyen.

Vi du:

```text
giao hang cham
ship lau
doi ca tuan moi nhan
mat don
shipper thai do
```

### 3.8. `price_value`

Van de ve gia tri so voi tien bo ra. Chi dung khi review noi ve gia cao, khong dang tien, khong xung gia, nhung khong neu loi chat luong cu the.

Vi du:

```text
khong dang tien
gia cao
khong xung gia
phi tien
gia dat so voi san pham
```

### 3.9. `spam_irrelevant`

Noi dung khong danh gia san pham that, chu yeu de nhan xu, du ky tu, hoac hinh anh/video khong lien quan. Chi dung nhan nay khi khong co issue that ro rang.

Vi du:

```text
nhan xu
cho du ky tu
hinh anh khong lien quan
video chi mang tinh minh hoa
danh gia de lay xu
```

## 4. Quy tac chon issue chinh

Khi mot binh luan co nhieu van de, chon 1 issue chinh theo cac quy tac sau:

```text
1. Neu dat A nhung nhan B, giao sai, giao nham, giao thieu -> wrong_missing_item
2. Neu loi chinh la dong goi, bao bi, hop -> packaging
3. Neu vua giao sai/thieu vua shop khong ho tro -> wrong_missing_item
4. Neu co loi hoac chat luong cu the -> product_quality
5. Neu chi noi gia cao/khong dang tien, khong neu loi cu the -> price_value
6. Neu co issue that thi khong gan spam_irrelevant
7. spam_irrelevant chi dung khi noi dung chinh la nhan xu/khong lien quan
8. Neu chi mo ta thuoc tinh -> no_issue
9. Neu thuoc tinh lech ky vong hoac lech mo ta -> product_attribute
```

Thu tu uu tien thuc te dang dung trong pipeline du doan:

```text
wrong_missing_item
packaging
seller_service
shipping_delivery
product_quality
price_value
product_attribute
spam_irrelevant
no_issue
```

Ghi chu: `product_quality` duoc uu tien hon `price_value` khi review vua neu loi chat luong cu the vua noi khong dang tien.

## 5. Quy tac sentiment chinh

### 5.1. Gan `positive`

Gan `positive` khi binh luan co loi khen ro hoac xac nhan san pham dung ky vong.

Vi du dau hieu:

```text
dep
tot
ung y
hai long
nen mua
se mua lai
dung mo ta
nhu hinh
nhan du hang
```

### 5.2. Gan `neutral`

Gan `neutral` khi binh luan chua the hien cam xuc ro, chi mo ta, chua dung/chua test, hoac noi dung nhan xu khong kem issue that.

Vi du dau hieu:

```text
chua dung
chua test
chua biet
tam duoc
binh thuong
nhan xu
hinh anh minh hoa
```

### 5.3. Gan `negative`

Gan `negative` khi co loi, khong hai long, giao sai/thieu/cham, shop khong ho tro, dong goi te, khong dang tien, hoac san pham khac ky vong.

Vi du dau hieu:

```text
bi loi
bi hong
giao sai
giao thieu
giao cham
shop khong rep
dong goi so sai
khong dang tien
khac mo ta
```

## 6. Cac truong hop nhap nhang da thong nhat

```text
"ok nhung hoi mong" -> neutral, product_attribute neu can gan issue; neu khong nghiem trong co the no_issue
"tam duoc so voi gia" -> neutral, no_issue
"tot so voi gia" -> positive, no_issue
"mau hoi khac anh" -> negative hoac neutral tuy muc do, issue product_attribute
"da nhan hang chua dung" -> neutral, no_issue
"da nhan du hang" -> positive, no_issue
"san pham nhu hinh" -> positive, no_issue
"dung mo ta" -> positive, no_issue
"dat mau trang giao mau den" -> negative, wrong_missing_item
"shop giao thieu hang, khong rep" -> negative, wrong_missing_item
"hinh anh nhan xu nhung giao hang cham" -> negative, shipping_delivery
"mau trang, chat lieu vai" -> neutral, no_issue
"chat luong te khong dang tien" -> negative, product_quality
```

## 7. Ket qua hien tai sau khi ap dung guideline

Sentiment classification:

```text
Feature: TF-IDF Unigram + Bigram
Model: Linear SVM
Accuracy: 0.8410
Macro F1: 0.8093
```

Issue classification:

```text
Feature: TF-IDF Word + Char
Model: Linear SVM + Safe Issue Rules
Accuracy: 0.7849
Macro F1: 0.7323
```

## 8. Ghi chu cho bao cao

Bo quy tac moi uu tien tinh nhat quan va kha nang ung dung thuc te hon viec toi uu chi so mot cach co lap. Mot so chi so co the thap hon nhe so voi ban cu, nhung nhan duoc gan ro rang hon, dac biet voi cac binh luan Shopee co nhieu van de trong cung mot cau.

Huong cai thien tiep theo la tiep tuc relabel them du lieu theo cung guideline nay, dac biet cac nhom de nham lan:

```text
wrong_missing_item vs product_attribute
packaging vs product_quality
seller_service vs wrong_missing_item
price_value vs product_quality
spam_irrelevant vs issue that
```
