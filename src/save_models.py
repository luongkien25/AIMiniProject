from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


SENTIMENT_DATA_PATH = Path(
    "data/processed/shopee_reviews_clean_classified_codex_sentiment_guideline_v4_accuracy.csv"
)
ISSUE_DATA_PATH = Path("data/processed/shopee_reviews_issue_augmented_reviewed.csv")
MODELS_DIR = Path("models")
TEXT_COLUMN = "cleaned_review"


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
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                max_features=10000,
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                max_features=20000,
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    C=1.0,
                ),
            ),
        ]
    )


def train_and_save_model(df, label_column, model, output_path):
    required_columns = {TEXT_COLUMN, label_column}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    train_df = df.dropna(subset=[TEXT_COLUMN, label_column]).copy()
    X = train_df[TEXT_COLUMN].fillna("")
    y = train_df[label_column].astype(str)

    model.fit(X, y)
    joblib.dump(model, output_path)

    return len(train_df), sorted(y.unique())


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    sentiment_df = pd.read_csv(SENTIMENT_DATA_PATH)
    issue_df = pd.read_csv(ISSUE_DATA_PATH)

    sentiment_rows, sentiment_labels = train_and_save_model(
        df=sentiment_df,
        label_column="sentiment",
        model=build_sentiment_model(),
        output_path=MODELS_DIR / "sentiment_model.joblib",
    )

    issue_rows, issue_labels = train_and_save_model(
        df=issue_df,
        label_column="issue",
        model=build_issue_model(),
        output_path=MODELS_DIR / "issue_model.joblib",
    )

    print("=" * 72)
    print("Saved sentiment model")
    print("Path:", MODELS_DIR / "sentiment_model.joblib")
    print("Training rows:", sentiment_rows)
    print("Labels:", ", ".join(sentiment_labels))
    print("Selection metric: Accuracy")
    print("Selected config: TF-IDF Unigram + Bigram + Logistic Regression")

    print("=" * 72)
    print("Saved issue model")
    print("Path:", MODELS_DIR / "issue_model.joblib")
    print("Training rows:", issue_rows)
    print("Labels:", ", ".join(issue_labels))
    print("Selection metric: Macro F1")
    print("Selected config: TF-IDF Word + Char + Linear SVM")


if __name__ == "__main__":
    main()
