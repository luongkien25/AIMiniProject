from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd

from predict import get_sentiment_rule_override
from preprocess import preprocess_text


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "Shopee"
OUTPUT_PATH = ROOT / "data" / "processed" / "shopee_issue_candidates_review.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "shopee_issue_candidates_summary.csv"
SENTIMENT_MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"
REQUIRED_COLUMNS = {
    "page",
    "issue_candidate",
    "score",
    "matched_patterns",
    "review_text",
}


def normalize_for_dedupe(text: object) -> str:
    value = "" if pd.isna(text) else str(text)
    value = value.lower().replace("đ", "d")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_raw_files() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        missing_columns = REQUIRED_COLUMNS.difference(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path.name} missing columns: {missing}")

        df = df[list(REQUIRED_COLUMNS)].copy()
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    raw_df = load_raw_files()
    raw_df["review_text"] = raw_df["review_text"].fillna("").astype(str).str.strip()
    raw_df = raw_df[raw_df["review_text"].str.len() >= 3].copy()
    raw_df["dedupe_key"] = raw_df["review_text"].map(normalize_for_dedupe)
    raw_df = raw_df.sort_values(["score"], ascending=False)
    df = raw_df.drop_duplicates(subset=["dedupe_key"]).copy()

    sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
    df["cleaned_review"] = df["review_text"].map(preprocess_text)
    model_sentiments = sentiment_model.predict(df["cleaned_review"].fillna(""))
    df["model_sentiment"] = model_sentiments

    rule_sentiments = []
    sentiment_rules = []
    for review_text, cleaned_review in zip(df["review_text"], df["cleaned_review"]):
        rule_sentiment, sentiment_rule = get_sentiment_rule_override(
            review_text,
            cleaned_review,
        )
        rule_sentiments.append(rule_sentiment)
        sentiment_rules.append(sentiment_rule)

    df["sentiment_rule"] = sentiment_rules
    df["sentiment"] = [
        rule_sentiment or model_sentiment
        for rule_sentiment, model_sentiment in zip(rule_sentiments, model_sentiments)
    ]
    df["issue"] = df["issue_candidate"].astype(str)
    df["needs_human_review"] = True
    df["review_note"] = ""

    output_columns = [
        "source_file",
        "page",
        "review_text",
        "cleaned_review",
        "sentiment",
        "model_sentiment",
        "sentiment_rule",
        "issue",
        "issue_candidate",
        "score",
        "matched_patterns",
        "needs_human_review",
        "review_note",
    ]
    df[output_columns].to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby("issue")
        .agg(rows=("review_text", "size"), avg_score=("score", "mean"))
        .sort_values("rows", ascending=False)
        .reset_index()
    )
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(f"Raw rows: {len(raw_df)}")
    print(f"Deduped rows: {len(df)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("Issue counts:")
    print(df["issue"].value_counts().to_string())


if __name__ == "__main__":
    main()
