import re
import unicodedata


def preprocess_text(text):
    """
    Preprocess Vietnamese customer review text before TF-IDF vectorization.

    Steps:
    1. Return an empty string for missing or non-string input.
    2. Normalize Unicode to keep Vietnamese accents stable.
    3. Convert text to lowercase.
    4. Remove URLs.
    5. Remove common emoji ranges.
    6. Remove noisy special characters while keeping letters, digits, and spaces.
    7. Normalize whitespace.

    Important sentiment/issue words such as "không", "chưa", "kém", "lỗi",
    "tệ", "chậm", "sai", and "hỏng" are not removed.
    """

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.lower()
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
