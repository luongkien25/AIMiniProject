from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocess import preprocess_text


DATA_PATH = Path("data/processed/merged_preprocessed_reviews.csv")


def train_sentiment_model():
    df = pd.read_csv(DATA_PATH)

    required_columns = {"processed_text", "sentiment_label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["processed_text", "sentiment_label"])

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=10000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=2.0,
                ),
            ),
        ]
    )

    model.fit(df["processed_text"], df["sentiment_label"])
    return model


def predict_review(model, review_text):
    processed_text = preprocess_text(review_text)
    prediction = model.predict([processed_text])[0]

    confidence = None
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba([processed_text])[0]
        confidence = probabilities.max()

    return processed_text, prediction, confidence


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH}. Create the merged preprocessed dataset first."
        )

    print("Training sentiment model from merged review data...")
    model = train_sentiment_model()
    print("Ready. Type a review and press Enter.")
    print("Type `exit` or `quit` to stop.\n")

    while True:
        review_text = input("Review: ").strip()
        if review_text.lower() in {"exit", "quit"}:
            break
        if not review_text:
            print("Please enter a non-empty review.\n")
            continue

        processed_text, prediction, confidence = predict_review(model, review_text)

        print(f"Processed text: {processed_text}")
        if confidence is None:
            print(f"Predicted sentiment: {prediction}\n")
        else:
            print(f"Predicted sentiment: {prediction}")
            print(f"Confidence: {confidence:.2%}\n")


if __name__ == "__main__":
    main()
