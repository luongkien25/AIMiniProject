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
    / "shopee_reviews_clean_classified_codex_sentiment_guideline_v3.csv"
)
CHANGES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "shopee_reviews_clean_classified_codex_sentiment_guideline_v3_changes.csv"
)


SERIOUS_NEGATIVE_PATTERNS = [
    r"\bkhong\s+dung\s+duoc\b",
    r"\bkhong\s+su\s+dung\s+duoc\b",
    r"\bkhong\s+xai\s+duoc\b",
    r"\bkhong\s+sai\s+duoc\b",
    r"\bkhong\s+hoat\s+dong\b",
    r"\bkhong\s+len\b",
    r"\bkhong\s+nghe\s+duoc\b",
    r"\bkhong\s+nghe\s+thay\b",
    r"\bkhong\s+sac\s+duoc\b",
    r"\bsac\s+khong\s+vao\b",
    r"\bkhong\s+vao\s+pin\b",
    r"\bkhong\s+ket\s+noi\s+duoc\b",
    r"\bkhong\s+ket\s+noi\b",
    r"\bket\s+noi\s+chap\s+chon\b",
    r"\bket\s+noi\s+khong\s+nhay\b",
    r"\bvan\s+de\s+ket\s+noi\b",
    r"\bkho\s+khan\s+.*\bket\s+noi\b",
    r"\bchi\s+nghe\s+duoc\s+1\s+ben\b",
    r"\bnghe\s+duoc\s+1\s+ben\b",
    r"\bnghe\s+1\s+ben\b",
    r"\bkhong\s+dinh\b",
    r"\bkhong\s+lap\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+hong\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+hu\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+loi\b",
    r"\bhang\s+loi\b",
    r"\bsan\s+pham\s+loi\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+rach\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+vo\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+be\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+nut\b",
    r"\bvo\s+nat\b",
    r"\bbe\s+vo\b",
    r"\brach\s+toac\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+gay\b",
    r"\bgay\s+1\s+ben\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+xuoc\b",
    r"\bvet\s+xuoc\b",
    r"\b\d+\s+vet\b",
    r"\bdut\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+thung\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+meo\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+long\b",
    r"(?<!khong )(?<!ko )(?<!k )(?<!co )(?<!so )\bbi\s+ban\b",
    r"\bban\s+dien\b",
    r"\blech\s+duong\s+may\b",
    r"\bchi\s+may\s+loi\b",
    r"\bkhong\s+dung\s+mo\s+ta\b",
    r"\bkhong\s+dung\s+mau\b",
    r"\bkhong\s+dung\s+size\b",
    r"\bkhong\s+dung\s+phan\s+loai\b",
    r"\bkhong\s+dung\s+hang\b",
    r"\bkhong\s+du\s+hang\b",
    r"\bkhong\s+nhu\s+hinh\b",
    r"\bkhong\s+giong\s+hinh\b",
    r"\bmau\s+(hoi\s+)?khac\s+anh\b(?!\s+chu)",
    r"\bmau\s+(hoi\s+)?khac\s+hinh\b",
    r"\bkhac\s+xa\b",
    r"\bsai\s+mau\b",
    r"\bsai\s+size\b",
    r"\bsai\s+kich\s+co\b",
    r"\bgiao\s+(sai|nham|khac)\s+mau\b",
    r"\bdat\s+mau\s+\w+\s+giao\s+mau\s+\w+\b",
    r"\bdat\s+(mau\s+)?(trang|den|hong|xanh|do|vang|ghi)\s+ve\s+mau\b",
    r"\bgiao\s+nham\b",
    r"\bnham\s+hang\b",
    r"\bthieu\s+mon\b",
    r"\bthieu\s+\d+\b",
    r"\bbi\s+thieu\b",
    r"\bgiao\s+thieu\s+(hang|mon|so\s+luong)\b",
    r"\bthieu\s+(dau|mieng|nut|phu\s+kien|so\s+luong)\b",
    r"\bmat\s+nut\b",
    r"\bhut\s+hang\b",
    r"\bqua\s+te\b",
    r"\bte\s+qua\b",
    r"\brat\s+te\b",
    r"\bkem\s+chat\s+luong\b",
    r"\b(qua|rat)\s+that\s+vong\b",
    r"\bthat\s+vong\s+(qua|that|ve)\b",
    r"\bkhong\s+hai\s+long\b",
    r"\bkhong\s+tot\b",
    r"\bkhong\s+on\b",
    r"\bkhong\s+duoc\s+tot\b",
    r"\bkhong\s+nhu\s+y\b",
    r"\bkhong\s+giong\s+.*\banh\b",
    r"\bkhong\s+dung\s+.*\bhinh\b",
    r"\bkhong\s+nhu\s+.*\bquang\s+cao\b",
    r"\b(muon|can|xin|yeu\s+cau)\s+tra\s+hang\b",
    r"\b(muon|can|xin|yeu\s+cau)\s+hoan\s+tien\b",
    r"\b(muon|can|xin|yeu\s+cau)\s+doi\s+tra\b",
    r"\bdong\s+goi\s+so\s+sai\b",
    r"\bdong\s+goi\s+khong\s+chac\s+chan\b",
    r"\bshop\s+im\s+re\b",
    r"\bkhong\s+ho\s+tro\b",
    r"\bon\s+lon\b",
    r"\bam\s+thanh\s+khong\s+.*\btrong\b",
    r"(?<!khong )(?<!ko )(?<!k )\bkho\s+chiu\b",
    r"\bdau\s+tai\b",
    r"\btit\b",
    r"\bpin\s+yeu\b",
    r"\bpin\s+khong\s+trau\b",
    r"\bpin\s+nhanh\s+het\b(?!\s+chut)",
    r"\bnhanh\s+het\s+pin\b(?!\s+chut)",
    r"\bmau\s+het\s+pin\b",
    r"\bhao\s+pin\b",
    r"\btu\s+tat\b",
    r"\btu\s+tac\b",
    r"\bbat\s+khong\s+len\b",
    r"\bbat\s+k\s+len\b",
    r"\bbat\s+ko\s+len\b",
    r"\bbat\s+ko\s+nen\b",
    r"\btuot\s+pin\b",
    r"\bmong\s+let\b",
    r"\bchat\s+nich\b",
    r"\bkhong\s+the\s+chong\s+nang\b",
    r"\bqua\s+chan\b",
    r"\bchan\s+qua\b",
    r"\brat\s+chan\b",
    r"\bbi\s+lung\s+\d*\s*lo\b",
    r"\bxi\s+ra\b",
    r"\bgiao\s+khong\s+du\b",
    r"\bcach\s+lam\s+viec\s+k\s+ok\b",
    r"\bkhong\s+nen\s+mua\b",
    r"\blua\s+dao\b",
    r"\bhang\s+fake\b",
    r"\bgia\s+mao\b",
]

SPAM_NEUTRAL_PATTERNS = [
    r"\bnhan\s+xu\b",
    r"\blay\s+xu\b",
    r"\bkiem\s+xu\b",
    r"\bdanh\s+gia\s+nhan\s+xu\b",
    r"\bcho\s+du\s+ky\s+tu\b",
    r"\bvideo\s+khong\s+lien\s+quan\b",
    r"\bhinh\s+anh\s+khong\s+lien\s+quan\b",
    r"\bhinh\s+anh\s+chi\s+mang\s+tinh\b",
]

GOOD_VALUE_PATTERNS = [
    r"\btot\s+so\s+voi\s+gia\b",
    r"\btot\s+tren\s+gia\b",
    r"\bon\s+so\s+voi\s+gia\b",
    r"\bdang\s+tien\b",
    r"\bxung\s+dang\b",
    r"\bre\s+ma\s+tot\b",
    r"\bgia\s+tot\b",
    r"\bgia\s+re\s+chat\s+luong\b",
]

STRONG_POSITIVE_PATTERNS = [
    r"\brat\s+dep\b",
    r"\bcuc\s+dep\b",
    r"\bdep\s+lam\b",
    r"\bdep\s+qua\b",
    r"\brat\s+tot\b",
    r"\btot\s+lam\b",
    r"\bchat\s+luong\s+tot\b",
    r"\bung\s+y\b",
    r"\bhai\s+long\b",
    r"\bnen\s+mua\b",
    r"\bse\s+mua\s+lai\b",
    r"\bdong\s+goi\s+can\s+than\b",
    r"\bgiao\s+nhanh\b",
    r"\bshop\s+uy\s+tin\b",
    r"\b10\s+diem\b",
    *GOOD_VALUE_PATTERNS,
]

DESCRIPTION_POSITIVE_PATTERNS = [
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

MILD_NEGATIVE_PATTERNS = [
    r"\bhoi\s+mong\b",
    r"\bhoi\s+day\b",
    r"\bhoi\s+nho\b",
    r"\bhoi\s+rong\b",
    r"\bhoi\s+chat\b",
    r"\bhoi\s+thua\b",
    r"\bhoi\s+co\b",
    r"\bhoi\s+le\s+size\b",
    r"\bhoi\s+mop\b",
    r"\bmop\s+nhe\b",
    r"\bhoi\s+nhan\b",
    r"\bhoi\s+lau\b",
    r"\bgiao\s+hoi\s+lau\b",
    r"\bship\s+hoi\s+lau\b",
    r"\bam\s+thanh\s+nho\b",
    r"\bnho\s+xiu\b",
    r"\bkhong\s+qua\s+xuat\s+sac\b",
    r"\bkhong\s+co\s+gi\s+dac\s+biet\b",
]

NEUTRAL_PATTERNS = [
    r"\btam\s+duoc\b",
    r"\btam\s+on\b",
    r"\bcung\s+duoc\b",
    r"\bchap\s+nhan\s+duoc\b",
    r"\bbinh\s+thuong\b",
    r"\bso\s+voi\s+gia\b",
    r"\btien\s+nao\s+cua\s+nay\b",
    r"\bmoi\s+nhan\b",
    r"\bda\s+nhan\s+hang\s+.*\bchua\b",
    r"\bnhan\s+hang\s+.*\bchua\b",
    r"\bchua\s+dung\b",
    r"\bchua\s+su\s+dung\b",
    r"\bchua\s+test\b",
    r"\bchua\s+thu\b",
    r"\bchua\s+biet\b",
    r"\bde\s+xem\b",
    r"\bdung\s+tam\b",
    r"\bdan\b",
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
]

BLOCK_POSITIVE_PATTERNS = [
    r"\bnhung\b",
    r"\btuy\s+nhien\b",
    r"\bco\s+dieu\b",
    r"\bnhuoc\s+diem\b",
    r"\btam\b",
    r"\bbinh\s+thuong\b",
    r"\bhoi\b",
    r"\bkhong\s+qua\b",
    r"\blau\b",
    r"\bcham\b",
    r"\bmop\b",
    r"\bmeo\b",
    r"\blong\b",
    r"\bye(u|u)\b",
    r"\bkem\b",
    r"\bloi\b",
    r"\bsai\b",
    r"\bnham\b",
    r"\bthieu\b",
    r"\bkhac\b",
    r"\bte\b",
    r"\bxau\b",
    r"\brach\b",
    r"\bthung\b",
    r"\bmong\b",
    r"\brot\b",
    r"\bkhong\s+hai\s+long\b",
    r"\bkhong\s+mong\s+doi\b",
    r"\bkhong\s+ung\s+ho\b",
    r"\bxot\s+tien\b",
    r"\bchi\s+thua\b",
    r"\bkhong\s+net\b",
    r"\bdinh\s+gi\b",
    r"\bqua\s+de\s+rot\b",
    r"\bdanh\s+gia\s+1\s+sao\b",
    r"\bkhong\s+nhan\s+dc\s+hang\b",
    r"\bk\s+nhan\s+dc\s+hang\b",
    r"\bmua\s+2\s+ma\s+co\s+1\b",
    r"\bthai\s+do\b",
    r"\bcoc\s+can\b",
    r"\bhet\s+pin\b",
    r"\bkhong\s+(tot|on|dep|duoc|dung|giong|ro|co|ngon|em)\b",
    r"\bk\s+(tot|on|dep|dc|duoc|dung|giong|ro|co|ngon|em)\b",
    r"\bko\s+(tot|on|dep|dc|duoc|dung|giong|ro|co|ngon|em)\b",
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
    if has_any(text, SERIOUS_NEGATIVE_PATTERNS + STRONG_POSITIVE_PATTERNS):
        return False
    return sum(1 for term in ATTRIBUTE_TERMS if re.search(rf"\b{term}\b", text)) >= 2


def apply_guideline(row: pd.Series) -> tuple[str, str]:
    current_label = str(row["sentiment"]).strip()
    review_text = normalize_text(row.get("review_text", ""))
    if len(review_text.split()) > 180:
        return current_label, "keep_existing_long_review"

    combined = normalize_text(f"{row.get('review_text', '')} {row.get('cleaned_review', '')}")

    # Clear product/order fault wins even if the tone is polite.
    if has_any(combined, SERIOUS_NEGATIVE_PATTERNS):
        return "negative", "clear_negative_issue"

    # Reward/spam-style reviews are neutral unless they contain a clear fault.
    if has_any(combined, SPAM_NEUTRAL_PATTERNS):
        return "neutral", "reward_or_spam_neutral"

    neutral_or_mild = (
        has_any(combined, NEUTRAL_PATTERNS)
        or has_any(combined, MILD_NEGATIVE_PATTERNS)
        or is_attribute_only(combined)
    )
    blocks_positive = has_any(combined, BLOCK_POSITIVE_PATTERNS)

    # Explicitly positive fulfillment/description statements, as long as the
    # full sentence does not contain a mixed/negative turn.
    if has_any(combined, DESCRIPTION_POSITIVE_PATTERNS) and not blocks_positive:
        return "positive", "received_enough_or_matches_description"

    # Strong praise, including good value for money, stays positive despite mild issues.
    if has_any(combined, STRONG_POSITIVE_PATTERNS) and (
        not blocks_positive or has_any(combined, MILD_NEGATIVE_PATTERNS)
    ):
        return "positive", "strong_positive_or_good_value"

    # Mild issue, "tam duoc", "so voi gia", not yet used/tested, or factual attributes.
    if neutral_or_mild:
        return "neutral", "neutral_or_mild_mixed"

    return current_label, "keep_existing"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    if "sentiment" not in df.columns:
        raise ValueError("Missing sentiment column")

    records = []
    new_labels = []
    for idx, row in df.iterrows():
        old_label = str(row["sentiment"]).strip()
        new_label, reason = apply_guideline(row)
        new_labels.append(new_label)
        if new_label != old_label:
            records.append(
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
    pd.DataFrame(records).to_csv(CHANGES_PATH, index=False, encoding="utf-8-sig")

    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Changes: {CHANGES_PATH}")
    print(f"Rows: {len(out)}")
    print(f"Changed rows: {len(records)}")
    print("Label counts:")
    print(out["sentiment"].value_counts().to_string())
    if records:
        changes = pd.DataFrame(records)
        print("Change counts:")
        print(
            changes.groupby(["old_sentiment", "new_sentiment"])
            .size()
            .sort_values(ascending=False)
            .to_string()
        )
        print("Reason counts:")
        print(changes["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
