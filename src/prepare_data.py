from pathlib import Path

import pandas as pd

from preprocess import preprocess_text


CLEANED_DATA_PATH = Path("data/processed/real_reviews.csv")
PREPROCESSED_DATA_PATH = Path("data/processed/realpreprocessed_reviews.csv")


def prepare_dataset(input_path=CLEANED_DATA_PATH, output_path=PREPROCESSED_DATA_PATH):
    df = pd.read_csv(input_path)

    if "review_text" not in df.columns:
        raise ValueError("Dataset must contain a 'review_text' column.")

    df["processed_text"] = df["review_text"].apply(preprocess_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    prepare_dataset()
