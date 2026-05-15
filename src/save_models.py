from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = Path("data/processed/merged_preprocessed_reviews.csv")
MODELS_DIR = Path("models")


def build_sentiment_model():
    return Pipeline(
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


def build_issue_model():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 1),
                    max_features=10000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    C=1.5,
                ),
            ),
        ]
    )


def train_and_save_model(df, label_column, model, output_path):
    required_columns = {"processed_text", label_column}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    train_df = df.dropna(subset=["processed_text", label_column]).copy()
    X = train_df["processed_text"].fillna("")
    y = train_df[label_column].astype(str)

    model.fit(X, y)
    joblib.dump(model, output_path)

    return len(train_df), sorted(y.unique())


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    sentiment_rows, sentiment_labels = train_and_save_model(
        df=df,
        label_column="sentiment_label",
        model=build_sentiment_model(),
        output_path=MODELS_DIR / "sentiment_model.joblib",
    )

    issue_rows, issue_labels = train_and_save_model(
        df=df,
        label_column="issue_label",
        model=build_issue_model(),
        output_path=MODELS_DIR / "issue_model.joblib",
    )

    print("=" * 72)
    print("Saved sentiment model")
    print("Path:", MODELS_DIR / "sentiment_model.joblib")
    print("Training rows:", sentiment_rows)
    print("Labels:", ", ".join(sentiment_labels))
    print("Selection metric: Macro F1")
    print("Selected config: TF-IDF Unigram + Bigram + Logistic Regression")

    print("=" * 72)
    print("Saved issue model")
    print("Path:", MODELS_DIR / "issue_model.joblib")
    print("Training rows:", issue_rows)
    print("Labels:", ", ".join(issue_labels))
    print("Selection metric: Macro F1")
    print("Selected config: TF-IDF Unigram + Linear SVM")


if __name__ == "__main__":
    main()
