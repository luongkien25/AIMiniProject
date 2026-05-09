from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/customer_review_1100_with_spam_multitopic.csv")
CLEANED_DATA_PATH = Path("data/processed/cleaned_reviews.csv")


def resolve_input_path(input_path=None):
    if input_path is not None:
        return Path(input_path)
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    return CLEANED_DATA_PATH


def clean_dataset(input_path=None, output_path=CLEANED_DATA_PATH):
    input_path = resolve_input_path(input_path)
    df = pd.read_csv(input_path)

    required_columns = {
        "review_id",
        "review_text",
        "sentiment_label",
        "issue_label",
        "is_multi_topic",
        "secondary_issue",
        "comment_type",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.split().str.len() >= 3]
    df = df.drop_duplicates(subset=["review_text"])
    
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip()
    df["issue_label"] = df["issue_label"].astype(str).str.strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Input: {input_path}")
    print("Shape:", df.shape)
    print("\nSentiment distribution:")
    print(df["sentiment_label"].value_counts())
    print("\nIssue distribution:")
    print(df["issue_label"].value_counts())
    print(f"\nSaved to {output_path}")

    return df


if __name__ == "__main__":
    clean_dataset()
