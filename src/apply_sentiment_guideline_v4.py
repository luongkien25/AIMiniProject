from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "shopee_reviews_clean_classified_codex_sentiment_reviewed_preprocessed_v2.csv"
)
OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "shopee_reviews_clean_classified_codex_sentiment_guideline_v4_accuracy.csv"
)
CHANGES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "shopee_reviews_clean_classified_codex_sentiment_guideline_v4_accuracy_changes.csv"
)

MANUAL_SENTIMENT_OVERRIDES = {
    # CSV line 5: the review has clear praise: "rat dep", "chat lieu tot",
    # "rat ung y", "nen mua", so it should stay positive despite "moi nhan".
    3: ("positive", "manual_review_line5_positive"),
}


CLEAR_NEGATIVE_PATTERNS = [
    r"\bkhong\s+dung\s+duoc\b",
    r"\bkhong\s+xai\s+duoc\b",
    r"\bkhong\s+sai\s+duoc\b",
    r"\bkhong\s+hoat\s+dong\b",
    r"\bkhong\s+len\b",
    r"\bkhong\s+nghe\s+duoc\b",
    r"\bkhong\s+nghe\s+thay\b",
    r"\bkhong\s+ket\s+noi\s+duoc\b",
    r"\bkhong\s+ket\s+noi\b",
    r"\bkhong\s+lap\b",
    r"\bket\s+noi\s+chap\s+chon\b",
    r"\bsac\s+khong\s+vao\b",
    r"\bkhong\s+vao\s+pin\b",
    r"\bbat\s+(khong|ko|k)\s+len\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+(hong|hu|loi|rach|vo|be|nut|gay|xuoc|meo|long|thung)\b",
    r"\bhang\s+loi\b",
    r"\bsan\s+pham\s+loi\b",
    r"\bvo\s+nat\b",
    r"\bbe\s+vo\b",
    r"\brach\s+toac\b",
    r"\bbi\s+dut\b",
    r"\bdut\s+day\b",
    r"\bbi\s+lung\s+\d*\s*lo\b",
    r"\blech\s+duong\s+may\b",
    r"\bchi\s+may\s+loi\b",
    r"\bkhong\s+dung\s+mo\s+ta\b",
    r"\bkhong\s+nhu\s+hinh\b",
    r"\bkhong\s+giong\s+hinh\b",
    r"\bkhong\s+dung\s+(mau|size|phan\s+loai|hang)\b",
    r"\bsai\s+(mau|size|kich\s+co)\b",
    r"\bgiao\s+(sai|nham)\b",
    r"\bnham\s+hang\b",
    r"\bgiao\s+thieu\s+(hang|mon|so\s+luong)\b",
    r"\bthieu\s+(mon|so\s+luong|phu\s+kien|nut|dau)\b",
    r"\bmua\s+\d+\s+(ma|nhung)\s+(co|giao)\s+\d+\b",
    r"\bpin\s+(yeu|khong\s+trau|nhanh\s+het)\b",
    r"\bpin\s+.*\bnhanh\s+het\b",
    r"\bnhanh\s+het\s+pin\b",
    r"\brat\s+nhanh\s+het\s+pin\b",
    r"\bmau\s+het\s+pin\b",
    r"\btuot\s+pin\b",
    r"\bhao\s+pin\b",
    r"\btu\s+(tat|tac)\b",
    r"\btac\s+toi\b",
    r"\bkhong\s+hai\s+long\b",
    r"\bkhong\s+nhu\s+y\b",
    r"\bkhong\s+ok\b",
    r"\bko\s+ok\b",
    r"\bk\s+ok\b",
    r"\bqua\s+te\b",
    r"\bte\s+qua\b",
    r"\brat\s+te\b",
    r"\bkem\s+chat\s+luong\b",
    r"\bthat\s+vong\s+(qua|that|ve)\b",
    r"\bshop\s+im\s+re\b",
    r"\bkhong\s+ho\s+tro\b",
    r"\bcoc\s+can\b",
    r"\bkhong\s+nen\s+mua\b",
    r"\blua\s+dao\b",
    r"\bhang\s+fake\b",
    r"\bgia\s+mao\b",
]

POSITIVE_FULFILLMENT_PATTERNS = [
    r"\bda\s+nhan\s+du\s+hang\b",
    r"\bnhan\s+du\s+hang\b",
    r"\bgiao\s+du\s+hang\b",
    r"\bdu\s+so\s+luong\b",
    r"\bdung\s+mo\s+ta\b",
    r"\bdung\s+voi\s+mo\s+ta\b",
    r"\bsan\s+pham\s+nhu\s+hinh\b",
    r"\bnhu\s+hinh\b",
    r"\bgiong\s+hinh\b",
    r"\bdung\s+hinh\b",
    r"\bdung\s+mau\b",
    r"\bdung\s+size\b",
    r"\bdung\s+mau\s+ma\b",
]

STRONG_POSITIVE_PATTERNS = [
    r"\brat\s+dep\b",
    r"\bcuc\s+dep\b",
    r"\bdep\s+lam\b",
    r"\brat\s+tot\b",
    r"\btot\s+lam\b",
    r"\bchat\s+luong\s+tot\b",
    r"\bung\s+y\b",
    r"\bhai\s+long\b",
    r"\bnen\s+mua\b",
    r"\bse\s+mua\s+lai\b",
    r"\bdang\s+tien\b",
    r"\bxung\s+dang\b",
    r"\btot\s+so\s+voi\s+gia\b",
    r"\btot\s+tren\s+gia\b",
    r"\bon\s+so\s+voi\s+gia\b",
    r"\bre\s+ma\s+tot\b",
]

NEUTRAL_PATTERNS = [
    r"\btam\s+duoc\b",
    r"\btam\s+on\b",
    r"\bcung\s+duoc\b",
    r"\bchap\s+nhan\s+duoc\b",
    r"\bbinh\s+thuong\b",
    r"\bmoi\s+nhan\b",
    r"\bda\s+nhan\s+hang\s+.*\bchua\b",
    r"\bnhan\s+hang\s+.*\bchua\b",
    r"\bchua\s+(dung|su\s+dung|test|thu|biet)\b",
    r"\bde\s+xem\b",
    r"\bnhan\s+xu\b",
    r"\blay\s+xu\b",
    r"\bkiem\s+xu\b",
    r"\bcho\s+du\s+ky\s+tu\b",
    r"\bhinh\s+anh\s+khong\s+lien\s+quan\b",
    r"\bvideo\s+khong\s+lien\s+quan\b",
]

MILD_MIXED_PATTERNS = [
    r"\bhoi\s+(mong|day|nho|rong|chat|thua|co|mop|nhan|lau)\b",
    r"\bgiao\s+hoi\s+lau\b",
    r"\bship\s+hoi\s+lau\b",
    r"\bam\s+thanh\s+nho\b",
    r"\bnho\s+xiu\b",
]

BLOCK_POSITIVE_PATTERNS = [
    r"\bnhung\b",
    r"\btuy\s+nhien\b",
    r"\bmoi\s+toi\b",
    r"\bco\s+dieu\b",
    r"\bnhuoc\s+diem\b",
    r"\btam\b",
    r"\bhoi\b",
    r"\bbinh\s+thuong\b",
    r"\bbi\b",
    r"\bloi\b",
    r"\bte\b",
    r"\bxau\b",
    r"\brach\b",
    r"\bthieu\b",
    r"\bsai\b",
    r"\bnham\b",
    r"\bkhac\b",
    r"\bkhong\b",
    r"\bko\b",
    r"\bk\b",
    r"\bban\b",
    r"\bchat\b",
    r"\bchat\s+nich\b",
    r"\bphi\s+tien\b",
    r"\bthat\s+vong\b",
    r"\btuot\s+pin\b",
    r"\bdung\s+(voi\s+)?mo\s+ta\s+(khong|ko|k)\b",
    r"\bdung\s+mo\s+ta\s+la\s+(khong|ko|k)\b",
    r"\bkhong\s+(tot|on|dep|dung|giong|hai\s+long|nhu\s+y)\b",
    r"\bko\s+(tot|on|dep|dung|giong|hai\s+long|nhu\s+y)\b",
    r"\bk\s+(tot|on|dep|dung|giong|hai\s+long|nhu\s+y)\b",
]

ATTRIBUTE_TERMS = [
    "mau",
    "mau sac",
    "chat lieu",
    "kich thuoc",
    "size",
    "phan loai",
    "kieu dang",
    "vai",
    "nhua",
    "trang",
    "den",
    "xanh",
    "do",
    "vang",
    "ghi",
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


def is_attribute_only(text: str) -> bool:
    if len(text.split()) > 12:
        return False
    if has_any(text, CLEAR_NEGATIVE_PATTERNS + STRONG_POSITIVE_PATTERNS):
        return False
    return sum(1 for term in ATTRIBUTE_TERMS if re.search(rf"\b{term}\b", text)) >= 2


def apply_guideline(row: pd.Series) -> tuple[str, str]:
    current_label = str(row["sentiment"]).strip()
    review_text = normalize_text(row.get("review_text", ""))

    # Very long scraped reviews often include product cards/vouchers.
    # Keep existing labels to avoid regex matches from unrelated page text.
    if len(review_text.split()) > 180:
        return current_label, "keep_existing_long_review"

    combined = normalize_text(f"{row.get('review_text', '')} {row.get('cleaned_review', '')}")
    has_clear_negative = has_any(combined, CLEAR_NEGATIVE_PATTERNS)
    has_strong_positive = has_any(combined, STRONG_POSITIVE_PATTERNS)
    has_positive_fulfillment = has_any(combined, POSITIVE_FULFILLMENT_PATTERNS)
    has_neutral = has_any(combined, NEUTRAL_PATTERNS) or is_attribute_only(combined)
    has_mild_mixed = has_any(combined, MILD_MIXED_PATTERNS)
    blocks_positive = has_any(combined, BLOCK_POSITIVE_PATTERNS)

    if has_clear_negative:
        return "negative", "clear_negative"

    if has_positive_fulfillment and not blocks_positive:
        return "positive", "positive_fulfillment"

    if has_strong_positive and not blocks_positive:
        return "positive", "strong_positive"

    if has_neutral:
        return "neutral", "clear_neutral"

    # Keep mild mixed reviews in their existing class unless they were already
    # neutral. This avoids over-expanding neutral when accuracy is the priority.
    if has_mild_mixed and current_label == "neutral":
        return "neutral", "keep_neutral_mild_mixed"

    return current_label, "keep_existing"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    if "sentiment" not in df.columns:
        raise ValueError("Missing sentiment column")

    new_labels = []
    changes = []
    for idx, row in df.iterrows():
        old_label = str(row["sentiment"]).strip()
        new_label, reason = apply_guideline(row)
        if idx in MANUAL_SENTIMENT_OVERRIDES:
            new_label, reason = MANUAL_SENTIMENT_OVERRIDES[idx]
        new_labels.append(new_label)
        if new_label != old_label:
            changes.append(
                {
                    "row_index": idx,
                    "old_sentiment": old_label,
                    "new_sentiment": new_label,
                    "reason": reason,
                    "review_text": row.get("review_text", ""),
                    "cleaned_review": row.get("cleaned_review", ""),
                    "issue": row.get("issue", ""),
                }
            )

    out = df.copy()
    out["sentiment"] = new_labels
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(changes).to_csv(CHANGES_PATH, index=False, encoding="utf-8-sig")

    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Changes: {CHANGES_PATH}")
    print(f"Rows: {len(out)}")
    print(f"Changed rows: {len(changes)}")
    print("Label counts:")
    print(out["sentiment"].value_counts().to_string())
    if changes:
        change_df = pd.DataFrame(changes)
        print("Change counts:")
        print(
            change_df.groupby(["old_sentiment", "new_sentiment"])
            .size()
            .sort_values(ascending=False)
            .to_string()
        )
        print("Reason counts:")
        print(change_df["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
