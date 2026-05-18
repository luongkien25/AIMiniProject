from pathlib import Path
import re
import sys
import unicodedata

import joblib

from preprocess import preprocess_text


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SENTIMENT_MODEL_PATH = Path("models/sentiment_model.joblib")
ISSUE_MODEL_PATH = Path("models/issue_model.joblib")


CLEAR_NEGATIVE_PATTERNS = [
    r"\bkhong\s+dung\s+duoc\b",
    r"\bkhong\s+xai\s+duoc\b",
    r"\bkhong\s+hoat\s+dong\b",
    r"\bkhong\s+ket\s+noi\b",
    r"\bkhong\s+nghe\s+duoc\b",
    r"\bkhong\s+len\b",
    r"\bgiao\s+(sai|nham)\b",
    r"\bgiao\s+(cham|lau|qua\s+lau)\b",
    r"\bgiao\s+hang\s+(cham|lau|qua\s+lau)\b",
    r"\bca\s+tuan\s+moi\s+nhan\b",
    r"\bsai\s+(mau|size|kich\s+co)\b",
    r"\bdat\s+.*\bgiao\s+.*\b(mau|size|phan\s+loai)\b",
    r"\bthieu\s+(hang|mon|so\s+luong|phu\s+kien)\b",
    r"\bbi\s+(hong|hu|loi|rach|vo|be|nut|gay|xuoc|meo|mop)\b",
    r"\bpin\s+(yeu|nhanh\s+het|khong\s+trau)\b",
    r"\bkhong\s+hai\s+long\b",
    r"\bkhong\s+dang\s+tien\b",
    r"\bgia\s+(cao|dat|mac)\b",
    r"\bdong\s+goi\s+(so\s+sai|khong\s+can\s+than|khong\s+chac|te)\b",
    r"\bmau\s+(hoi\s+)?khac\b",
    r"\bkhac\s+(anh|hinh|mo\s+ta)\b",
    r"\bqua\s+te\b",
    r"\bkem\s+chat\s+luong\b",
]

POSITIVE_FULFILLMENT_PATTERNS = [
    r"\bda\s+nhan\s+du\s+hang\b",
    r"\bnhan\s+du\s+hang\b",
    r"\bgiao\s+du\s+hang\b",
    r"\bdu\s+so\s+luong\b",
    r"\bsan\s+pham\s+nhu\s+hinh\b",
    r"\bnhu\s+hinh\b",
    r"\bgiong\s+hinh\b",
    r"\bdung\s+mo\s+ta\b",
    r"\bdung\s+voi\s+mo\s+ta\b",
    r"\bdung\s+mau\b",
    r"\bdung\s+size\b",
]

CLEAR_NEUTRAL_PATTERNS = [
    r"\bda\s+nhan\s+hang\s+.*\bchua\b",
    r"\bnhan\s+hang\s+.*\bchua\b",
    r"\bchua\s+(dung|su\s+dung|test|thu|biet)\b",
    r"\btam\s+duoc\b",
    r"\btam\s+on\b",
    r"\bbinh\s+thuong\b",
    r"\bnhan\s+xu\b",
    r"\bcho\s+du\s+ky\s+tu\b",
]

BLOCK_POSITIVE_PATTERNS = [
    *CLEAR_NEGATIVE_PATTERNS,
    r"\bnhung\b",
    r"\btuy\s+nhien\b",
    r"\bmoi\s+toi\b",
    r"\bco\s+dieu\b",
    r"\bhoi\b",
    r"\btam\b",
    r"\bloi\b",
    r"\bxau\b",
    r"\bte\b",
    r"\bkhong\b",
    r"\bko\b",
    r"\bk\b",
]

ISSUE_RULES = [
    (
        "spam_irrelevant",
        [
            r"\bnhan\s+xu\b",
            r"\blay\s+xu\b",
            r"\bkiem\s+xu\b",
            r"\bcho\s+du\s+ky\s+tu\b",
            r"\bhinh\s+anh\s+khong\s+lien\s+quan\b",
            r"\bvideo\s+khong\s+lien\s+quan\b",
            r"\bchi\s+mang\s+tinh\s+minh\s+hoa\b",
            r"\bminh\s+hoa\b",
        ],
    ),
    (
        "wrong_missing_item",
        [
            r"\bgiao\s+(sai|nham|thieu)\b",
            r"\bshop\s+giao\s+(sai|nham|thieu)\b",
            r"\bdat\s+.*\bgiao\s+.*\b(mau|size|phan\s+loai)\b",
            r"\bdat\s+(mau\s+)?(trang|den|hong|xanh|do|vang|ghi|be|nau|tim|cam)\s+"
            r"(giao|ve)\s+(mau\s+)?(trang|den|hong|xanh|do|vang|ghi|be|nau|tim|cam)\b",
            r"\bdat\s+(mau\s+)?(trang|den|hong|xanh|do|vang|ghi)\s+ve\s+mau\b",
            r"\bthieu\s+(hang|mon|so\s+luong|phu\s+kien|day|sac|nut|dau)\b",
            r"\bmua\s+\d+\s+(ma|nhung)\s+(co|giao)\s+\d+\b",
            r"\bkhong\s+du\s+(hang|so\s+luong|phu\s+kien)\b",
            r"\bnham\s+(hang|mau|size|phan\s+loai)\b",
        ],
    ),
    (
        "seller_service",
        [
            r"\bshop\s+(khong|ko|k)\s+(rep|tra\s+loi|phan\s+hoi|ho\s+tro)\b",
            r"\bshop\s+im\b",
            r"\btu\s+van\s+sai\b",
            r"\bkhong\s+ho\s+tro\b",
            r"\bkhong\s+giai\s+quyet\b",
            r"\bthai\s+do\b",
            r"\bcoc\s+can\b",
            r"\bnoi\s+chuyen\s+(te|kho\s+chiu)\b",
        ],
    ),
    (
        "shipping_delivery",
        [
            r"\bgiao\s+(cham|lau|qua\s+lau|khong\s+nhanh)\b",
            r"\bship\s+(cham|lau|qua\s+lau)\b",
            r"\bvan\s+chuyen\s+(cham|lau)\b",
            r"\bdoi\s+.*\bngay\s+moi\s+nhan\b",
            r"\bca\s+tuan\s+moi\s+nhan\b",
            r"\bthat\s+lac\b",
            r"\bmat\s+don\b",
            r"\bshipper\s+(thai\s+do|coc\s+can|khong\s+than\s+thien)\b",
        ],
    ),
    (
        "packaging",
        [
            r"\bdong\s+goi\s+(so\s+sai|khong\s+can\s+than|khong\s+chac|te)\b",
            r"\bgoi\s+(so\s+sai|khong\s+can\s+than|khong\s+chac|te)\b",
            r"\bhop\s+(bi\s+)?(mop|meo|rach|vo|nat)\b",
            r"\bbao\s+bi\s+(rach|mop|meo|vo|nat)\b",
            r"\bkhong\s+(co\s+)?chong\s+soc\b",
            r"\bboc\s+hang\s+(so\s+sai|te)\b",
        ],
    ),
    (
        "price_value",
        [
            r"\bkhong\s+dang\s+tien\b",
            r"\bkhong\s+xung\s+(dang|gia)\b",
            r"\bphi\s+tien\b",
            r"\bgia\s+(cao|dat|mac)\b",
            r"\bdat\s+so\s+voi\s+chat\s+luong\b",
            r"\bchat\s+luong\s+khong\s+xung\s+gia\b",
            r"\btam\s+gia\s+nay\s+khong\s+on\b",
        ],
    ),
    (
        "product_quality",
        [
            r"\bkhong\s+(dung|xai|sai|su\s+dung)\s+duoc\b",
            r"\bkhong\s+hoat\s+dong\b",
            r"\bkhong\s+len\b",
            r"\bkhong\s+ket\s+noi\b",
            r"\bkhong\s+nghe\s+duoc\b",
            r"\bsac\s+khong\s+vao\b",
            r"\bpin\s+(yeu|nhanh\s+het|khong\s+trau|hao)\b",
            r"\bbi\s+(hong|hu|loi|rach|vo|be|nut|gay|xuoc|meo|long)\b",
            r"\bhang\s+(loi|hong|kem\s+chat\s+luong)\b",
            r"\bchat\s+luong\s+(kem|te|khong\s+tot)\b",
            r"\bmoi\s+dung\s+.*\b(hong|loi|rach|vo|be|gay)\b",
        ],
    ),
    (
        "product_attribute",
        [
            r"\bmau\s+(khac|sai|khong\s+giong|khong\s+dung)\b",
            r"\bsize\s+(khac|sai|nho|rong|chat|khong\s+dung)\b",
            r"\bform\s+(nho|rong|xau|khong\s+dep|khong\s+chuan)\b",
            r"\bchat\s+lieu\s+(khac|mong|day|khong\s+giong|khong\s+dung)\b",
            r"\bvai\s+(mong|day|khong\s+dep|khong\s+giong)\b",
            r"\bkhong\s+dung\s+(mau|size|kich\s+thuoc|chat\s+lieu|mau\s+ma)\b",
            r"\bkhac\s+(anh|hinh|mo\s+ta)\b",
            r"\bhoi\s+(mong|day|nho|rong|chat|thua)\b",
        ],
    ),
]

ISSUE_PRIORITY = [
    "wrong_missing_item",
    "packaging",
    "seller_service",
    "shipping_delivery",
    "product_quality",
    "price_value",
    "product_attribute",
    "spam_irrelevant",
]

SAFE_ISSUE_OVERRIDE_LABELS = {
    "packaging",
    "price_value",
    "seller_service",
    "shipping_delivery",
}

STRICT_WRONG_MISSING_PATTERNS = [
    r"\bgiao\s+(sai|nham|thieu)\s+(hang|mau|size|phan\s+loai|san\s+pham)\b",
    r"\bshop\s+giao\s+(sai|nham|thieu)\s+(hang|mau|size|phan\s+loai|san\s+pham)\b",
    r"\bdat\s+(mau\s+)?(trang|den|hong|xanh|do|vang|ghi|be|nau|tim|cam)\s+"
    r"(giao|ve)\s+(mau\s+)?(trang|den|hong|xanh|do|vang|ghi|be|nau|tim|cam)\b",
    r"(?<!gioi\s)\bthieu\s+(hang|mon|so\s+luong|phu\s+kien|day|sac|nut|dau|bom|qua)\b",
    r"\bkhong\s+du\s+(hang|so\s+luong|phu\s+kien)\b",
]

STRICT_QUALITY_VALUE_PATTERNS = [
    r"\bchat\s+luong\s+(te|kem|khong\s+tot|qua\s+te)\b.*"
    r"\b(khong\s+dang\s+tien|phi\s+tien|khong\s+xung)\b",
    r"\b(khong\s+dang\s+tien|phi\s+tien|khong\s+xung)\b.*"
    r"\bchat\s+luong\s+(te|kem|khong\s+tot|qua\s+te)\b",
]

DESCRIPTIVE_NO_ISSUE_PATTERNS = [
    r"^mau\s+\w+\s+chat\s+lieu\s+\w+$",
    r"^mau\s+sac\s+\w+\s+chat\s+lieu\s+\w+$",
    r"^chat\s+lieu\s+\w+\s+mau\s+\w+$",
]

BLOCK_DESCRIPTIVE_NO_ISSUE_PATTERNS = [
    r"\b(khong|ko|k)\b",
    r"\bhoi\b",
    r"\bkhac\b",
    r"\bsai\b",
    r"\bloi\b",
    r"\bhong\b",
    r"\bte\b",
    r"\bkem\b",
    r"\bmong\b",
    r"\bnho\b",
    r"\brong\b",
    r"\bsize\s+chat\b",
    r"\bhoi\s+chat\b",
    r"\bthieu\b",
    r"\bgiao\b",
]


def load_models():
    missing_paths = [
        str(path)
        for path in [SENTIMENT_MODEL_PATH, ISSUE_MODEL_PATH]
        if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise FileNotFoundError(
            f"Cannot find model file(s): {missing}. Run `python src/save_models.py` first."
        )

    sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
    issue_model = joblib.load(ISSUE_MODEL_PATH)
    return sentiment_model, issue_model


def normalize_for_rules(text):
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def has_any(text, patterns):
    return any(re.search(pattern, text) for pattern in patterns)


def get_sentiment_rule_override(review_text, processed_text):
    rule_text = normalize_for_rules(f"{review_text} {processed_text}")

    if has_any(rule_text, CLEAR_NEGATIVE_PATTERNS):
        return "negative", "clear_negative_rule"

    if has_any(rule_text, POSITIVE_FULFILLMENT_PATTERNS) and not has_any(
        rule_text,
        BLOCK_POSITIVE_PATTERNS,
    ):
        return "positive", "positive_fulfillment_rule"

    if has_any(rule_text, CLEAR_NEUTRAL_PATTERNS):
        return "neutral", "clear_neutral_rule"

    return None, None


def get_issue_rule_override(review_text, processed_text):
    rule_text = normalize_for_rules(f"{review_text} {processed_text}")

    matches = []
    for issue_label, patterns in ISSUE_RULES:
        if has_any(rule_text, patterns):
            matches.append(issue_label)

    if matches:
        best_issue = min(
            matches,
            key=lambda issue: ISSUE_PRIORITY.index(issue)
            if issue in ISSUE_PRIORITY
            else len(ISSUE_PRIORITY),
        )
        return best_issue, f"{best_issue}_priority_rule"

    return None, None


def get_safe_issue_rule_override(review_text, processed_text):
    rule_text = normalize_for_rules(f"{review_text} {processed_text}")
    review_rule_text = normalize_for_rules(review_text)

    if has_any(review_rule_text, DESCRIPTIVE_NO_ISSUE_PATTERNS) and not has_any(
        rule_text,
        BLOCK_DESCRIPTIVE_NO_ISSUE_PATTERNS,
    ):
        return "no_issue", "descriptive_no_issue_rule"

    if has_any(rule_text, STRICT_WRONG_MISSING_PATTERNS):
        return "wrong_missing_item", "strict_wrong_missing_item_rule"

    if has_any(rule_text, STRICT_QUALITY_VALUE_PATTERNS):
        return "product_quality", "strict_quality_value_rule"

    issue_label, issue_rule = get_issue_rule_override(review_text, processed_text)
    if issue_label in SAFE_ISSUE_OVERRIDE_LABELS:
        return issue_label, issue_rule

    return None, None


def predict_review(sentiment_model, issue_model, review_text):
    processed_text = preprocess_text(review_text)
    model_sentiment = sentiment_model.predict([processed_text])[0]
    model_issue = issue_model.predict([processed_text])[0]
    rule_sentiment, sentiment_rule = get_sentiment_rule_override(
        review_text,
        processed_text,
    )
    rule_issue, issue_rule = get_safe_issue_rule_override(review_text, processed_text)
    sentiment = rule_sentiment or model_sentiment
    issue = rule_issue or model_issue

    return {
        "processed_text": processed_text,
        "sentiment": sentiment,
        "model_sentiment": model_sentiment,
        "sentiment_rule": sentiment_rule,
        "issue": issue,
        "model_issue": model_issue,
        "issue_rule": issue_rule,
    }


def print_prediction(result):
    sentiment_display = str(result["sentiment"]).capitalize()

    print(f"Processed text: {result['processed_text']}")
    print(f"Sentiment: {sentiment_display}")
    if result["sentiment_rule"] is not None:
        print(f"Sentiment rule: {result['sentiment_rule']}")
    print(f"Issue: {result['issue']}")
    if result["issue_rule"] is not None:
        print(f"Issue rule: {result['issue_rule']}")

    print()


def main():
    sentiment_model, issue_model = load_models()

    print("Demo dự đoán bình luận khách hàng")
    print("Nhập một bình luận rồi nhấn Enter.")
    print("Ví dụ: shop giao chậm, hộp bị móp")
    print("Gõ `exit` hoặc `quit` để thoát.\n")

    while True:
        review_text = input("Review: ").strip()
        if review_text.lower() in {"exit", "quit"}:
            break
        if not review_text:
            print("Vui lòng nhập bình luận không rỗng.\n")
            continue

        result = predict_review(sentiment_model, issue_model, review_text)
        print_prediction(result)


if __name__ == "__main__":
    main()
