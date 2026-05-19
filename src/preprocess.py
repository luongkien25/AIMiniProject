import re
import unicodedata


def preprocess_text(text):
    """
    Preprocess Vietnamese customer review text before TF-IDF vectorization.

    Steps:
    1. Return an empty string for missing or non-string input.
    2. Normalize Unicode to keep Vietnamese accents stable.
    3. Convert text to lowercase.
    4. Normalize common Vietnamese e-commerce teencode.
    5. Remove URLs.
    6. Remove common emoji ranges.
    7. Remove noisy special characters while keeping letters, digits, and spaces.
    8. Normalize whitespace.

    Important sentiment/issue words such as "không", "chưa", "kém", "lỗi",
    "tệ", "chậm", "sai", and "hỏng" are not removed.
    """

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Normalize common Shopee-style teencode to stable ASCII tokens.
    text = re.sub(r"\b(không|khong|ko|kh|k|hong|hông|hok|hổng)\b", " khong ", text)
    text = re.sub(r"\b(được|duoc|đc|dc|dk|đk)\b", " duoc ", text)
    text = re.sub(r"\b(chưa|chua|chx|cxua|cưa)\b", " chua ", text)
    text = re.sub(r"\b(bình\s*thường|binh\s*thuong|bt|bth)\b", " binh thuong ", text)
    text = re.sub(r"\b(sản\s*phẩm|san\s*pham|sp)\b", " san pham ", text)
    text = re.sub(r"\b(mọi\s*người|moi\s*nguoi|mn|mng)\b", " moi nguoi ", text)
    text = re.sub(r"\b(shop|sốp|sop|shope)\b", " shop ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)

    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text
