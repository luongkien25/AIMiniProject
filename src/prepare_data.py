import os
from pathlib import Path

import pandas as pd

from preprocess import preprocess_text


DATA_PATH = Path(
    os.environ.get("PREPARE_DATA_PATH", "data/processed/shopee_reviews_labeled.csv")
)
OUTPUT_PATH = os.environ.get("PREPARE_OUTPUT_PATH")
TEXT_COLUMN = "review_text"
PREPARED_TEXT_COLUMN = "cleaned_review"


def prepare_dataset(input_path=DATA_PATH, output_path=None):
    input_path = Path(input_path)
    df = pd.read_csv(input_path)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain a '{TEXT_COLUMN}' column.")

    if PREPARED_TEXT_COLUMN in df.columns and output_path is None:
        print(f"Input already contains '{PREPARED_TEXT_COLUMN}'. No file written.")
        print(f"Input: {input_path}")
        print("Shape:", df.shape)
        return df

    df[PREPARED_TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_text)

    output_path = Path(output_path) if output_path is not None else input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("Shape:", df.shape)

    return df


if __name__ == "__main__":
    prepare_dataset(output_path=OUTPUT_PATH)
