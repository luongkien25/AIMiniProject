from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from preprocess import preprocess_text


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "shopee_issue_candidates_review.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "shopee_issue_candidates_reviewed.csv"
CHANGES_PATH = (
    ROOT / "data" / "processed" / "shopee_issue_candidates_reviewed_changes.csv"
)


ISSUE_RULES = [
    (
        "spam_irrelevant",
        [
            r"\bnhan\s+xu\b",
            r"\blay\s+xu\b",
            r"\bkiem\s+xu\b",
            r"\bdanh\s+gia\s+nhan\s+xu\b",
            r"\bcho\s+du\s+ky\s+tu\b",
            r"\bhinh\s+anh\s+khong\s+lien\s+quan\b",
            r"\bvideo\s+khong\s+lien\s+quan\b",
            r"\bhinh\s+anh\s+chi\s+mang\s+tinh\b",
            r"\bvideo\s+chi\s+mang\s+tinh\b",
        ],
    ),
    (
        "wrong_missing_item",
        [
            r"\bgiao\s+(sai|nham|thieu)\b",
            r"\bshop\s+giao\s+(sai|nham|thieu)\b",
            r"\bdat\s+.*\bgiao\s+.*\b(mau|size|phan\s+loai|hang)\b",
            r"\bdat\s+(mau\s+)?(trang|den|hong|xanh|do|vang|ghi|be|nau|tim|cam)\s+"
            r"(giao|ve)\s+(mau\s+)?(trang|den|hong|xanh|do|vang|ghi|be|nau|tim|cam)\b",
            r"\bdat\s+(mau\s+)?(trang|den|hong|xanh|do|vang|ghi)\s+ve\s+mau\b",
            r"\bthieu\s+(hang|mon|so\s+luong|phu\s+kien|day|sac|nut|dau|bom|qua)\b",
            r"\bmua\s+\d+\s+(ma|nhung)\s+(co|giao|nhan)\s+\d+\b",
            r"\bkhong\s+du\s+(hang|so\s+luong|phu\s+kien)\b",
            r"\b(khong|ko|k)\s+(co|duoc|dc)\s+qua\b",
            r"\b(khong|ko|k)\s+duoc\s+tang\b",
            r"\bqua\s+tang\s+(khong|ko|k|chua|thieu|het|sai)\b",
            r"\bthieu\s+qua\b",
            r"\bnham\s+(hang|mau|size|phan\s+loai|san\s+pham)\b",
            r"\bkhong\s+nhan\s+duoc\s+hang\b",
        ],
    ),
    (
        "seller_service",
        [
            r"\bshop\s+(khong|ko|k)\s+(rep|tra\s+loi|phan\s+hoi|ho\s+tro|giai\s+quyet)\b",
            r"\bshop\s+im\b",
            r"\bkhong\s+tra\s+loi\b",
            r"\bkhong\s+phan\s+hoi\b",
            r"\btu\s+van\s+sai\b",
            r"\bkhong\s+ho\s+tro\b",
            r"\bkhong\s+giai\s+quyet\b",
            r"\bdoi\s+tra\s+.*\b(kho|phien|mat\s+thoi\s+gian)\b",
            r"\bthai\s+do\b",
            r"\bcoc\s+can\b",
            r"\bkhong\s+ton\s+trong\b",
            r"\bnoi\s+chuyen\s+(te|kho\s+chiu)\b",
        ],
    ),
    (
        "shipping_delivery",
        [
            r"\bgiao\s+(cham|lau|qua\s+lau|khong\s+nhanh|hang\s+lau)\b",
            r"\bship\s+(cham|lau|qua\s+lau)\b",
            r"\bvan\s+chuyen\s+(cham|lau|qua\s+lau)\b",
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
            r"\bdong\s+goi\s+(so\s+sai|khong\s+can\s+than|khong\s+chac|te|qua\s+te|loi)\b",
            r"\bgoi\s+(hang\s+)?(so\s+sai|khong\s+can\s+than|khong\s+chac|te|qua\s+te)\b",
            r"\bhop\s+(bi\s+)?(mop|meo|rach|vo|nat|be)\b",
            r"\bthung\s+(bi\s+)?(mop|meo|rach|vo|nat|be)\b",
            r"\bbao\s+bi\s+(rach|mop|meo|vo|nat|be)\b",
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
            r"\bvoi\s+gia\s+.*\b(khong\s+on|khong\s+dang|qua\s+te)\b",
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
            r"\bbi\s+(hong|hu|loi|rach|vo|be|nut|gay|xuoc|meo|long|ri|do|ban)\b",
            r"\bhang\s+(loi|hong|kem\s+chat\s+luong)\b",
            r"\bchat\s+luong\s+(kem|te|khong\s+tot)\b",
            r"\bmoi\s+dung\s+.*\b(hong|loi|rach|vo|be|gay)\b",
            r"\bvo\s+hang\b",
            r"\bdut\b",
            r"\bthung\b",
            r"\bmui\s+(hoi|kho\s+chiu|nong)\b",
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
            r"\bhoi\s+(mong|day|nho|rong|chat|thua|khac)\b",
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

NEGATIVE_PATTERNS = [
    r"\bkhong\s+hai\s+long\b",
    r"\bthat\s+vong\b",
    r"\bqua\s+te\b",
    r"\bte\s+qua\b",
    r"\brat\s+te\b",
    r"\bkem\b",
    r"\bxau\b",
    r"\bchan\b",
    r"\bphi\s+tien\b",
    r"\bkhong\s+dang\s+tien\b",
    r"\bkhong\s+xung\b",
    r"\bkhong\s+nen\s+mua\b",
    r"\bkhong\s+on\b",
    r"\bkhong\s+tot\b",
    r"\bkhong\s+duoc\b",
    r"\b(khong|ko|k)\s+(rep|tra\s+loi|ho\s+tro|giai\s+quyet)\b",
    r"\bgiao\s+(sai|nham|thieu|cham|lau)\b",
    r"\bthieu\s+(hang|mon|so\s+luong|phu\s+kien)\b",
    r"\bdong\s+goi\s+(so\s+sai|te|khong\s+can\s+than)\b",
    r"\bkhong\s+(co\s+)?chong\s+soc\b",
    r"\bbi\s+(hong|hu|loi|rach|vo|be|nut|gay|xuoc|meo|long|ri|do|ban)\b",
    r"\bpin\s+(yeu|nhanh\s+het|khong\s+trau|hao)\b",
]

POSITIVE_PATTERNS = [
    r"\brat\s+dep\b",
    r"\bdep\s+lam\b",
    r"\brat\s+tot\b",
    r"\btot\s+lam\b",
    r"\bchat\s+luong\s+tot\b",
    r"\bung\s+y\b",
    r"\bhai\s+long\b",
    r"\bnen\s+mua\b",
    r"\bse\s+mua\s+lai\b",
    r"\bdang\s+tien\b",
    r"\bgiao\s+nhanh\b",
]

NEUTRAL_PATTERNS = [
    r"\bnhan\s+xu\b",
    r"\bcho\s+du\s+ky\s+tu\b",
    r"\bhinh\s+anh\s+khong\s+lien\s+quan\b",
    r"\bvideo\s+khong\s+lien\s+quan\b",
    r"\bchua\s+(dung|su\s+dung|test|thu|biet)\b",
    r"\btam\s+duoc\b",
    r"\btam\s+on\b",
    r"\bbinh\s+thuong\b",
]


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def first_matching_issue(text: str, fallback: str) -> tuple[str, str]:
    matches = []
    for issue, patterns in ISSUE_RULES:
        if has_any(text, patterns):
            matches.append(issue)

    if matches:
        best_issue = min(
            matches,
            key=lambda issue: ISSUE_PRIORITY.index(issue)
            if issue in ISSUE_PRIORITY
            else len(ISSUE_PRIORITY),
        )
        return best_issue, f"{best_issue}_priority_rule"

    return fallback, "keep_candidate"


def review_sentiment(text: str, issue: str, old_sentiment: str) -> tuple[str, str]:
    has_negative = has_any(text, NEGATIVE_PATTERNS)
    has_positive = has_any(text, POSITIVE_PATTERNS)
    has_neutral = has_any(text, NEUTRAL_PATTERNS)

    if issue == "spam_irrelevant" and not has_negative:
        return "neutral", "spam_or_irrelevant_neutral"

    if has_negative:
        return "negative", "clear_complaint_negative"

    if has_positive and not has_neutral:
        return "positive", "clear_praise_positive"

    if has_neutral:
        return "neutral", "neutral_or_not_used"

    if issue != "no_issue":
        return "negative", "issue_candidate_default_negative"

    return old_sentiment, "keep_existing"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    required_columns = {
        "review_text",
        "cleaned_review",
        "sentiment",
        "issue",
        "issue_candidate",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    records = []
    reviewed_rows = []

    for idx, row in df.iterrows():
        old_issue = str(row["issue"]).strip()
        old_sentiment = str(row["sentiment"]).strip()
        text = normalize_text(f"{row.get('review_text', '')} {row.get('cleaned_review', '')}")

        new_issue, issue_reason = first_matching_issue(text, old_issue)
        new_sentiment, sentiment_reason = review_sentiment(
            text,
            new_issue,
            old_sentiment,
        )

        output = row.copy()
        output["cleaned_review"] = preprocess_text(str(row["review_text"]))
        output["issue"] = new_issue
        output["sentiment"] = new_sentiment
        output["review_status"] = "reviewed_by_rule_and_context"
        output["needs_human_review"] = False
        output["review_note"] = f"{issue_reason}; {sentiment_reason}"
        reviewed_rows.append(output)

        if new_issue != old_issue or new_sentiment != old_sentiment:
            records.append(
                {
                    "row_index": idx,
                    "old_issue": old_issue,
                    "new_issue": new_issue,
                    "issue_reason": issue_reason,
                    "old_sentiment": old_sentiment,
                    "new_sentiment": new_sentiment,
                    "sentiment_reason": sentiment_reason,
                    "review_text": row.get("review_text", ""),
                    "matched_patterns": row.get("matched_patterns", ""),
                }
            )

    reviewed_df = pd.DataFrame(reviewed_rows)
    reviewed_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(CHANGES_PATH, index=False, encoding="utf-8-sig")

    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Changes: {CHANGES_PATH}")
    print(f"Rows: {len(reviewed_df)}")
    print(f"Changed rows: {len(records)}")
    print("Issue counts:")
    print(reviewed_df["issue"].value_counts().to_string())
    print("Sentiment counts:")
    print(reviewed_df["sentiment"].value_counts().to_string())
    if records:
        changes = pd.DataFrame(records)
        print("Issue change counts:")
        print(
            changes.groupby(["old_issue", "new_issue"])
            .size()
            .sort_values(ascending=False)
            .head(20)
            .to_string()
        )
        print("Sentiment change counts:")
        print(
            changes.groupby(["old_sentiment", "new_sentiment"])
            .size()
            .sort_values(ascending=False)
            .head(20)
            .to_string()
        )


if __name__ == "__main__":
    main()
