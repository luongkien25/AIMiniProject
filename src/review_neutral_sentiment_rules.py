from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/shopee_reviews_clean_classified_codex.csv")
OUTPUT_PATH = Path("data/processed/shopee_reviews_clean_classified_codex_sentiment_reviewed.csv")
CHANGES_PATH = Path(
    "data/processed/shopee_reviews_clean_classified_codex_sentiment_review_changes.csv"
)

TEXT_COLUMNS = ["review_text", "cleaned_review"]
SENTIMENT_COLUMN = "sentiment"


def normalize_for_rules(value: object) -> str:
    if not isinstance(value, str):
        return ""

    text = value.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.lower()
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def combined_text(row: pd.Series) -> str:
    parts = [str(row[column]) for column in TEXT_COLUMNS if isinstance(row.get(column), str)]
    return " ".join(parts)


def has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


SERIOUS_NEGATIVE_PATTERNS = [
    r"\bkhong dung mo ta\b",
    r"\bkhong giao dung\b",
    r"\bkhong nhu mo ta\b",
    r"\bkhong giong mo ta\b",
    r"\bkhong giong .{0,25}hinh\b",
    r"\bkhong giong .{0,25}anh\b",
    r"\bko giong .{0,25}hinh\b",
    r"\bko giong .{0,25}anh\b",
    r"\bgiao loai binh thuong\b",
    r"\bkhac anh\b",
    r"\bkhac hinh\b",
    r"\bkhac hoan toan\b",
    r"\bmau khong giong\b",
    r"\bmau ko giong\b",
    r"\bmau kh giong\b",
    r"\bsai mau\b",
    r"\bsai size\b",
    r"\bsai nho\b",
    r"\bsai kich thuoc\b",
    r"\bgiao sai\b",
    r"\bgui sai\b",
    r"\bdat .{0,25} gui\b",
    r"\bthieu hang\b",
    r"\bkhong du so luong\b",
    r"\bgiao khong du\b",
    r"\bgiao thieu\b",
    r"\bgui thieu\b",
    r"\bthieu [a-z0-9]{1,20}\b",
    r"\bbi loi\b",
    r"\bloi\b",
    r"\bhong\b",
    r"\bhu\b",
    r"\bhu roi\b",
    r"\bde hu\b",
    r"\bbi vo\b",
    r"\bde vo\b",
    r"\bvo nat\b",
    r"\bvo be\b",
    r"\bbi be\b",
    r"\bbe nat\b",
    r"\brach\b",
    r"\bnut\b",
    r"\bbi mop\b",
    r"\bmop\b",
    r"\bbi gay\b",
    r"\bgay\b",
    r"\bbi rac\b",
    r"\bbi den\b",
    r"\bbi venh\b",
    r"\bbi boc\b",
    r"\bboc vo\b",
    r"\bboc .{0,20}truoc\b",
    r"\bbong troc\b",
    r"\bkhong dung duoc\b",
    r"\bkhong xai duoc\b",
    r"\bkhong su dung duoc\b",
    r"\bkhong an duoc\b",
    r"\bkhong nghe duoc\b",
    r"\bkhong nghe thay\b",
    r"\bchi nghe duoc .{0,20}1 ben\b",
    r"\bko nghe dc\b",
    r"\bko nghe duoc\b",
    r"\bthinh thoang .{0,20}khong nghe\b",
    r"\bkhong co chuc nang\b",
    r"\bkhong khu mui\b",
    r"\bkhong tac dong\b",
    r"\bcha co tac dung\b",
    r"\bkhong hop\b",
    r"\bkhong phu hop\b",
    r"\bkhong tot\b",
    r"\bkg tot\b",
    r"\bkhong thich\b",
    r"\bkho chiu\b",
    r"\bkha te\b",
    r"\bte\b",
    r"\bhoan thien qua kem\b",
    r"\bchua tot\b",
    r"\bchua hoan thien\b",
    r"\bthiet ke chua hoan thien\b",
    r"\bkhong dang ke\b",
    r"\bkhong dep\b",
    r"\bkhong chac chan\b",
    r"\blong leo\b",
    r"\bkhong nhay\b",
    r"\bkhong muot\b",
    r"\bkhong net\b",
    r"\bphai an manh\b",
    r"\bkhong co tang giam\b",
    r"\bkhong co nut\b",
    r"\bkhong tien\b",
    r"\bchua thay .{0,30}hieu qua\b",
    r"\bkhong thay .{0,30}hieu qua\b",
    r"\bkhong co do\b",
    r"\bchua thay .{0,30}trang\b",
    r"\bkhong thay .{0,30}trang\b",
    r"\bcang dung .{0,30}tham\b",
    r"\btham them\b",
    r"\bsam them\b",
    r"\bhieu qua mo nhat\b",
    r"\bsac .{0,30}ko .{0,20}(len|vo|duoc)\b",
    r"\bsac .{0,30}khong .{0,20}(len|vo|duoc)\b",
    r"\bko .{0,20}sac\b",
    r"\bkhong .{0,20}sac\b",
    r"\bchan sac .{0,20}(khong|ko|kg) tot\b",
    r"\bchap chon\b",
    r"\bkhong he dinh\b",
    r"\bde rot\b",
    r"\bde bi rot\b",
    r"\bde roi\b",
    r"\bde bi roi\b",
    r"\bde bung\b",
    r"\bbung chi\b",
    r"\bbung ca hop\b",
    r"\broi mat\b",
    r"\brot mat\b",
    r"\broi cai hat\b",
    r"\brot cai hat\b",
    r"\brot het\b",
    r"\bro be\b",
    r"\bbe nhu vay\b",
    r"\bnho xiu\b",
    r"\bnho xui\b",
    r"\bmong qua\b",
    r"\bsin mau\b",
    r"\bbay mau\b",
    r"\blem mau\b",
    r"\bcung do\b",
    r"\bmong manh yeu\b",
    r"\bmop nang\b",
    r"\bqua te\b",
    r"\bchat luong kem\b",
    r"\bthat vong\b",
    r"\bkhong hai long\b",
    r"\bkhong hieu qua\b",
    r"\bkhong co chot\b",
    r"\bbi lua\b",
    r"\bthat buc\b",
    r"\btrai nghiem khong tot\b",
    r"\bsieu chat\b",
    r"\bngan qua\b",
    r"\bchua duoc hai long\b",
    r"\bchua dc hai long\b",
    r"\bnon het\b",
    r"\bkhong bao gio mua lai\b",
    r"\bkhong mua lai\b",
    r"\bkien\b",
    r"\bthua loai\b",
    r"\bau tha\b",
    r"\blech lac\b",
    r"\bmau xau\b",
    r"\bchai pin\b",
    r"\bk net\b",
    r"\bcam k net\b",
    r"\bbi tham\b",
    r"\btham moi\b",
    r"\bbanh trang bi dai\b",
    r"\bhoi dau\b",
    r"\bkho tach\b",
    r"\bmuon xiu\b",
    r"\btit\b",
    r"\bbanh bi moc\b",
    r"\bkhong ra trang chinh hang\b",
    r"\bkhong chinh hang\b",
    r"\bfake\b",
    r"\bqua mong\b",
    r"\bgiao qua lau\b",
    r"\bship qua lau\b",
    r"\blau qua\b",
]

MILD_NEGATIVE_PATTERNS = [
    r"\bhoi mong\b",
    r"\bhoi nho\b",
    r"\bhoi rong\b",
    r"\bhoi chat\b",
    r"\bhoi xau\b",
    r"\bhoi lau\b",
    r"\bhoi cham\b",
    r"\bhoi yeu\b",
    r"\bhoi long leo\b",
    r"\bhoi khac\b",
    r"\bhoi nhat\b",
    r"\bhoi dam\b",
    r"\bkhong qua noi bat\b",
    r"\bkhong dac biet\b",
    r"\bchua ro do ben\b",
]

UNCERTAIN_STRONG_NEUTRAL_PATTERNS = [
    r"\bchua dung\b",
    r"\bchua su dung\b",
    r"\bchua test\b",
    r"\bchua thu\b",
    r"\bchua biet\b",
    r"\bdung them\b",
    r"\bdung thu\b",
    r"\bdanh gia sau\b",
    r"\bdanh gia nhan xu\b",
    r"\bnhan xu\b",
    r"\bcho du chu\b",
    r"\bcho du ky tu\b",
    r"\bhinh anh minh hoa\b",
    r"\bvideo khong lien quan\b",
    r"\bkhong biet viet gi\b",
]

RECEIVED_ONLY_PATTERNS = [
    r"\bmoi nhan\b",
    r"\bda nhan\b",
    r"\bnhan du\b",
    r"\bnhan hang\b",
]

MIXED_NEUTRAL_PATTERNS = [
    r"\btam duoc\b",
    r"\btam on\b",
    r"\btam chap nhan\b",
    r"\bnoi chung .{0,20}tam\b",
    r"\bcung tam\b",
    r"\bchap nhan duoc\b",
    r"\bcung ok\b",
    r"\bcung duoc\b",
    r"\bok nhung\b",
    r"\bduoc nhung\b",
    r"\bnhung van\b",
    r"\bnhung cung\b",
    r"\bso voi gia\b",
    r"\bphu hop voi gia\b",
    r"\btien nao cua nay\b",
    r"\bbinh thuong\b",
    r"\bkhong qua tot\b",
    r"\bkhong qua te\b",
    r"\bkhong danh gia qua cao\b",
    r"\bo muc trung binh\b",
]

STRONG_POSITIVE_PATTERNS = [
    r"\bhang dep\b",
    r"\bsan pham dep\b",
    r"\bmau sac dep\b",
    r"\bmau dep\b",
    r"\bvai dep\b",
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
    r"\bdang tien\b",
    r"\bse mua lai\b",
    r"\bse ung ho\b",
    r"\bchat luong tot\b",
    r"\bdong goi ky\b",
    r"\bdong goi chac chan\b",
    r"\bgia ca hop ly\b",
    r"\bgia thanh hop ly\b",
    r"\bmem mai\b",
    r"\bthoang mat\b",
    r"\bde chiu\b",
    r"\bde thuong\b",
    r"\bcute\b",
    r"\bxinh lam\b",
    r"\brat xinh\b",
    r"\bngon lam\b",
    r"\brat ngon\b",
]


def count_matches(text: str, patterns: list[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text))


def serious_negative_is_softened(text: str) -> bool:
    if re.search(r"\bhoi .{0,20}(khac anh|khac hinh)\b", text):
        return has_any(text, MIXED_NEUTRAL_PATTERNS + [r"\bvan\b", r"\bnhin cung\b"])
    if re.search(r"\bhop .{0,20}hoi mop nhe\b", text):
        return has_any(text, [r"\bben trong con nguyen\b", r"\bhang khong sao\b"])
    return False


def infer_sentiment(row: pd.Series) -> tuple[str, str]:
    current = str(row[SENTIMENT_COLUMN])
    text = normalize_for_rules(combined_text(row))

    if not text:
        return current, "keep_empty_text"

    serious_negative = has_any(text, SERIOUS_NEGATIVE_PATTERNS)
    mild_negative = has_any(text, MILD_NEGATIVE_PATTERNS)
    uncertain_strong_neutral = has_any(text, UNCERTAIN_STRONG_NEUTRAL_PATTERNS)
    received_only = has_any(text, RECEIVED_ONLY_PATTERNS)
    mixed_neutral = has_any(text, MIXED_NEUTRAL_PATTERNS)
    positive_score = count_matches(text, STRONG_POSITIVE_PATTERNS)
    issue = str(row.get("issue", ""))

    if serious_negative and not serious_negative_is_softened(text):
        return "negative", "serious_negative_issue"

    if issue == "spam_irrelevant":
        return "neutral", "spam_irrelevant_or_reward_neutral"

    if uncertain_strong_neutral:
        return "neutral", "received_or_not_used_or_reward_review"

    if mixed_neutral or mild_negative:
        if positive_score >= 1 and current == "positive":
            return current, "keep_positive_with_mild_issue"
        return "neutral", "mixed_or_mild_issue_neutral_overall"

    if received_only and positive_score == 0 and current != "negative" and len(text.split()) <= 18:
        return "neutral", "received_only_without_clear_sentiment"

    if current == "neutral" and issue != "no_issue":
        return current, "keep_neutral_with_issue"

    if positive_score >= 2 and not serious_negative:
        return "positive", "strong_positive_overall"

    if positive_score >= 1 and not serious_negative:
        return "positive", "positive_overall"

    return current, "keep_no_rule_match"


def should_apply_change(old_sentiment: str, new_sentiment: str) -> bool:
    if old_sentiment == new_sentiment:
        return False

    # This audit is intentionally scoped to neutral cleanup:
    # neutral -> positive/negative and positive/negative -> neutral.
    return old_sentiment == "neutral" or new_sentiment == "neutral"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    reviewed_df = df.copy()

    change_rows = []
    for index, row in df.iterrows():
        old_sentiment = str(row[SENTIMENT_COLUMN])
        inferred_sentiment, reason = infer_sentiment(row)

        if not should_apply_change(old_sentiment, inferred_sentiment):
            continue

        reviewed_df.at[index, SENTIMENT_COLUMN] = inferred_sentiment
        change_rows.append(
            {
                "row_index": index,
                "old_sentiment": old_sentiment,
                "new_sentiment": inferred_sentiment,
                "reason": reason,
                "issue": row.get("issue", ""),
                "review_text": row.get("review_text", ""),
                "cleaned_review": row.get("cleaned_review", ""),
            }
        )

    changes_df = pd.DataFrame(change_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reviewed_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    changes_df.to_csv(CHANGES_PATH, index=False, encoding="utf-8-sig")

    print("Input:", INPUT_PATH)
    print("Output:", OUTPUT_PATH)
    print("Changes:", CHANGES_PATH)
    print("Rows:", len(df))
    print()
    print("Before sentiment distribution:")
    print(df[SENTIMENT_COLUMN].value_counts().to_string())
    print()
    print("After sentiment distribution:")
    print(reviewed_df[SENTIMENT_COLUMN].value_counts().to_string())
    print()
    print("Changed rows:", len(changes_df))
    if len(changes_df) > 0:
        print()
        print("Change direction:")
        print(
            changes_df.groupby(["old_sentiment", "new_sentiment"])
            .size()
            .sort_values(ascending=False)
            .to_string()
        )
        print()
        print("Reasons:")
        print(changes_df["reason"].value_counts().to_string())
        print()
        print("Sample changes:")
        print(
            changes_df[
                ["row_index", "old_sentiment", "new_sentiment", "reason", "review_text"]
            ]
            .head(30)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
