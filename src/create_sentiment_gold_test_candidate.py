from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "shopee_reviews_clean_classified_codex_sentiment_guideline_v4_accuracy.csv"
)
OUTPUT_PATH = ROOT / "data" / "processed" / "sentiment_gold_test_candidate_600.csv"
TEXT_COLUMN = "cleaned_review"
LABEL_COLUMN = "sentiment"
SAMPLES_PER_CLASS = 200
RANDOM_STATE = 62
MAX_REVIEW_WORDS = 180


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    required_columns = {"review_text", TEXT_COLUMN, LABEL_COLUMN, "issue"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    df["source_row_index"] = df.index
    df["_word_count"] = df["review_text"].fillna("").astype(str).str.split().str.len()
    df = df[df["_word_count"] <= MAX_REVIEW_WORDS].copy()
    df = df.drop_duplicates(subset=[TEXT_COLUMN])

    sampled_parts = []
    for label, group in df.groupby(LABEL_COLUMN):
        if len(group) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"Not enough rows for {label}: {len(group)} < {SAMPLES_PER_CLASS}"
            )
        sampled_parts.append(
            group.sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_STATE)
        )

    out = (
        pd.concat(sampled_parts)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    out["gold_sentiment"] = out[LABEL_COLUMN]
    out["review_status"] = "needs_human_review"
    out["review_note"] = ""

    columns = [
        "source_row_index",
        "review_text",
        TEXT_COLUMN,
        "sentiment",
        "gold_sentiment",
        "issue",
        "review_status",
        "review_note",
    ]
    out[columns].to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(out)}")
    print("Gold candidate counts:")
    print(out["gold_sentiment"].value_counts().to_string())


if __name__ == "__main__":
    main()
