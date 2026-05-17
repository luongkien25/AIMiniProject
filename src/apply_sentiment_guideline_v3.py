from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from preprocess import preprocess_text


INPUT_PATH = Path("data/processed/shopee_reviews_clean_classified_codex.csv")
OUTPUT_PATH = Path(
    "data/processed/shopee_reviews_clean_classified_codex_sentiment_guideline_v3.csv"
)
CHANGES_PATH = Path(
    "data/processed/shopee_reviews_clean_classified_codex_sentiment_guideline_v3_changes.csv"
)


def normalize_for_rules(value: object) -> str:
    if not isinstance(value, str):
        return ""

    text = value.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.lower()
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    replacements = [
        (r"\b(k|kh|ko|hong|hok|khong)\b", "khong"),
        (r"\b(dc|dk|duoc)\b", "duoc"),
        (r"\b(chx|chua)\b", "chua"),
        (r"\b(bt|bth)\b", "binh thuong"),
        (r"\b(sp)\b", "san pham"),
        (r"\b(mn|mng)\b", "moi nguoi"),
        (r"\b(sop|shope)\b", "shop"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    return re.sub(r"\s+", " ", text).strip()


def has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


CLEAR_NEGATIVE_PATTERNS = [
    r"\bkhong dung mo ta\b",
    r"\bkhong nhu mo ta\b",
    r"\bkhong giong\b",
    r"\bkhac anh\b",
    r"\bkhac hinh\b",
    r"\bkhac hoan toan\b",
    r"\bsai mau\b",
    r"\bsai size\b",
    r"\bsai kich thuoc\b",
    r"\bgiao sai\b",
    r"\bgiao nham\b",
    r"\bgui sai\b",
    r"\bthieu hang\b",
    r"\bgiao thieu\b",
    r"\bgui thieu\b",
    r"\bkhong du so luong\b",
    r"\bmua \d+ giao \d+\b",
    r"\bbi loi\b",
    r"\bloi\b",
    r"\bhong\b",
    r"\bhu\b",
    r"\brach\b",
    r"\bvo\b",
    r"\bbe\b",
    r"\bnut\b",
    r"\bbong troc\b",
    r"\bkhong dung duoc\b",
    r"\bkhong xai duoc\b",
    r"\bkhong su dung duoc\b",
    r"\bkhong an duoc\b",
    r"\bkhong nghe duoc\b",
    r"\bkhong sac\b",
    r"\bsac khong\b",
    r"\bpin nhanh het\b",
    r"\bkhong co tac dung\b",
    r"\bkhong hieu qua\b",
    r"\bcang dung .* tham\b",
    r"\bsam them\b",
    r"\btham them\b",
    r"\bbi lua\b",
    r"\bthat vong\b",
    r"\bkhong hai long\b",
    r"\bqua te\b",
    r"\bchat luong kem\b",
    r"\bkem\b",
]

MILD_NEUTRAL_PATTERNS = [
    r"\bhoi mong\b",
    r"\bhoi nho\b",
    r"\bhoi rong\b",
    r"\bhoi chat\b",
    r"\bhoi lau\b",
    r"\bhoi cham\b",
    r"\bhoi nhat\b",
    r"\bhoi khac\b",
    r"\bhoi mop nhe\b",
    r"\bvan dung duoc\b",
    r"\bvan xai duoc\b",
    r"\bkhong sao\b",
    r"\bcung tam\b",
    r"\btam duoc\b",
    r"\btam on\b",
    r"\bbinh thuong\b",
    r"\bchap nhan duoc\b",
    r"\bkhong qua tot\b",
    r"\bkhong qua te\b",
]

PRICE_NEUTRAL_PATTERNS = [
    r"\bso voi gia\b",
    r"\bphu hop voi gia\b",
    r"\btien nao cua nay\b",
    r"\bgia re nen\b",
    r"\bgia.*tam\b",
]

PRICE_POSITIVE_PATTERNS = [
    r"\bdang tien\b",
    r"\brat dang tien\b",
    r"\bxung dang\b",
    r"\btot so voi gia\b",
    r"\bchat luong tot.*gia\b",
    r"\bgia tot\b",
    r"\bgia re.*chat luong\b",
    r"\bqua re\b",
]

REWARD_NEUTRAL_PATTERNS = [
    r"\bnhan xu\b",
    r"\bcho du ky tu\b",
    r"\bdanh gia.*xu\b",
    r"\bhinh anh.*minh hoa\b",
    r"\bhinh anh.*nhan xu\b",
    r"\bvideo khong lien quan\b",
]

STRONG_POSITIVE_PATTERNS = [
    r"\brat dep\b",
    r"\bdep lam\b",
    r"\bqua dep\b",
    r"\brat tot\b",
    r"\btot lam\b",
    r"\btuyet voi\b",
    r"\brat ung\b",
    r"\bung qua\b",
    r"\bhai long\b",
    r"\bnen mua\b",
    r"\bse mua lai\b",
    r"\bung ho tiep\b",
    r"\bse ung ho\b",
    r"\bchat luong tot\b",
    r"\bsieu dep\b",
    r"\bsieu xinh\b",
    r"\brat xinh\b",
    r"\bxinh lam\b",
]

CONFIRM_POSITIVE_PATTERNS = [
    r"\bda nhan du hang\b",
    r"\bnhan du hang\b",
    r"\bnhan du so luong\b",
    r"\bday du\b",
    r"\bsan pham nhu hinh\b",
    r"\bnhu hinh\b",
    r"\bgiong hinh\b",
    r"\bgiong anh\b",
    r"\bdung mo ta\b",
    r"\bdung voi mo ta\b",
    r"\bdung nhu mo ta\b",
]

ATTRIBUTE_ONLY_PATTERNS = [
    r"\bmau sac\b",
    r"\bchat lieu\b",
    r"\bmau trang\b",
    r"\bmau den\b",
    r"\bsize\b",
    r"\bkich thuoc\b",
]

UNCERTAIN_NEUTRAL_PATTERNS = [
    r"\bchua dung\b",
    r"\bchua su dung\b",
    r"\bchua test\b",
    r"\bchua biet\b",
    r"\bdung them\b",
    r"\bdanh gia sau\b",
    r"\bmoi nhan\b",
    r"\bmoi kiem tra\b",
]


def infer_sentiment(row: pd.Series) -> tuple[str, str]:
    old_sentiment = str(row["sentiment"])
    text = normalize_for_rules(row.get("review_text", ""))
    issue = str(row.get("issue", ""))

    clear_negative = has_any(text, CLEAR_NEGATIVE_PATTERNS)
    mild_neutral = has_any(text, MILD_NEUTRAL_PATTERNS)
    price_neutral = has_any(text, PRICE_NEUTRAL_PATTERNS)
    price_positive = has_any(text, PRICE_POSITIVE_PATTERNS)
    reward_neutral = has_any(text, REWARD_NEUTRAL_PATTERNS)
    strong_positive = has_any(text, STRONG_POSITIVE_PATTERNS)
    confirm_positive = has_any(text, CONFIRM_POSITIVE_PATTERNS)
    uncertain_neutral = has_any(text, UNCERTAIN_NEUTRAL_PATTERNS)
    attribute_only = has_any(text, ATTRIBUTE_ONLY_PATTERNS)

    # 2 + 6: clear error/sai/thieu/hong wins even if tone is soft or has mild praise.
    if clear_negative or issue == "wrong_missing_item":
        return "negative", "clear_error_or_wrong_missing_item"

    # 4: reward/spam without clear error is neutral.
    if reward_neutral or issue == "spam_irrelevant":
        return "neutral", "reward_or_spam_neutral"

    # 5: strong praise plus only mild complaint is positive.
    if strong_positive:
        return "positive", "strong_positive_overall"

    # 3: positive price/value wording is positive.
    if price_positive:
        return "positive", "positive_value_for_money"

    # 7: exact receipt/description confirmations are positive by user guideline.
    if confirm_positive:
        return "positive", "confirmation_positive_guideline"

    # 1 + 3: mild issue / temporary / average / price-normal comments are neutral.
    if mild_neutral or price_neutral or uncertain_neutral:
        return "neutral", "mild_or_uncertain_neutral"

    # 7: bare attribute description without sentiment is neutral.
    if attribute_only and old_sentiment not in {"positive", "negative"}:
        return "neutral", "attribute_only_neutral"

    return old_sentiment, "keep_no_rule_match"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    out_df = df.copy()
    out_df["cleaned_review"] = out_df["review_text"].apply(preprocess_text)

    changes = []
    for index, row in out_df.iterrows():
        old_sentiment = str(row["sentiment"])
        new_sentiment, reason = infer_sentiment(row)
        if old_sentiment == new_sentiment:
            continue
        out_df.at[index, "sentiment"] = new_sentiment
        changes.append(
            {
                "row_index": index,
                "old_sentiment": old_sentiment,
                "new_sentiment": new_sentiment,
                "reason": reason,
                "issue": row.get("issue", ""),
                "review_text": row.get("review_text", ""),
                "cleaned_review": out_df.at[index, "cleaned_review"],
            }
        )

    changes_df = pd.DataFrame(changes)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    changes_df.to_csv(CHANGES_PATH, index=False, encoding="utf-8-sig")

    print("Input:", INPUT_PATH)
    print("Output:", OUTPUT_PATH)
    print("Changes:", CHANGES_PATH)
    print()
    print("Before:")
    print(df["sentiment"].value_counts().to_string())
    print()
    print("After:")
    print(out_df["sentiment"].value_counts().to_string())
    print()
    print("Changed rows:", len(changes_df))
    if len(changes_df):
        print()
        print("Directions:")
        print(
            changes_df.groupby(["old_sentiment", "new_sentiment"])
            .size()
            .sort_values(ascending=False)
            .to_string()
        )
        print()
        print("Reasons:")
        print(changes_df["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
