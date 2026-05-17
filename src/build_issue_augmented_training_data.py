from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "shopee_reviews_clean_classified_codex_sentiment_guideline_v4_accuracy.csv"
)
CANDIDATES_PATH = (
    ROOT / "data" / "processed" / "shopee_issue_candidates_reviewed.csv"
)
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "shopee_reviews_issue_augmented_reviewed.csv"
)
SUMMARY_PATH = (
    ROOT / "data" / "processed" / "shopee_reviews_issue_augmented_reviewed_summary.csv"
)
CORE_COLUMNS = ["page", "review_text", "cleaned_review", "sentiment", "issue"]


def normalize_for_dedupe(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    base = pd.read_csv(BASE_PATH)
    candidates = pd.read_csv(CANDIDATES_PATH)

    for name, df in [("base", base), ("candidates", candidates)]:
        missing = set(CORE_COLUMNS).difference(df.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{name} missing columns: {missing_text}")

    base_train = base[CORE_COLUMNS].copy()
    base_train["data_source"] = "base_guideline_v4"

    candidate_train = candidates[CORE_COLUMNS].copy()
    candidate_train["data_source"] = "new_shopee_issue_reviewed"

    base_keys = set(base_train["review_text"].map(normalize_for_dedupe))
    candidate_train["dedupe_key"] = candidate_train["review_text"].map(
        normalize_for_dedupe
    )
    before_dedupe = len(candidate_train)
    candidate_train = candidate_train[
        ~candidate_train["dedupe_key"].isin(base_keys)
    ].copy()
    candidate_train = candidate_train.drop_duplicates(subset=["dedupe_key"])
    appended_rows = len(candidate_train)
    candidate_train = candidate_train.drop(columns=["dedupe_key"])

    output = pd.concat([base_train, candidate_train], ignore_index=True)
    output = output.dropna(subset=["cleaned_review", "sentiment", "issue"]).copy()
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary = (
        output.groupby(["data_source", "issue"])
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["data_source", "rows"], ascending=[True, False])
    )
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(f"Base rows: {len(base_train)}")
    print(f"Candidate rows before base-dedupe: {before_dedupe}")
    print(f"Candidate rows appended: {appended_rows}")
    print(f"Output rows: {len(output)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("Final issue counts:")
    print(output["issue"].value_counts().to_string())


if __name__ == "__main__":
    main()
