import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


DATA_PATH = os.environ.get(
    "SENTIMENT_DATA_PATH",
    "data/processed/shopee_reviews_labeled.csv",
)
TEST_DATA_PATH = os.environ.get("SENTIMENT_TEST_PATH")
TEXT_COLUMN = "cleaned_review"
LABEL_COLUMN = "sentiment"
PRIMARY_METRIC = os.environ.get("SENTIMENT_PRIMARY_METRIC", "accuracy").lower()
RANDOM_STATE = 62
TEST_SIZE = 0.2

FEATURE_NAME = "TF-IDF Unigram + Bigram"


def build_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        sublinear_tf=True,
    )


def build_models():
    return {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            C=1.5,
        ),
    }


def main():
    if PRIMARY_METRIC not in {"accuracy", "macro_f1"}:
        raise ValueError("SENTIMENT_PRIMARY_METRIC must be accuracy or macro_f1")

    df = pd.read_csv(DATA_PATH)

    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    df["_source_row_index"] = df.index

    if TEST_DATA_PATH:
        test_df = pd.read_csv(TEST_DATA_PATH)
        test_label_column = os.environ.get(
            "SENTIMENT_TEST_LABEL_COLUMN",
            "gold_sentiment" if "gold_sentiment" in test_df.columns else LABEL_COLUMN,
        )
        required_test_columns = {TEXT_COLUMN, test_label_column}
        missing_test_columns = required_test_columns.difference(test_df.columns)
        if missing_test_columns:
            missing = ", ".join(sorted(missing_test_columns))
            raise ValueError(f"Missing required test columns: {missing}")

        if "source_row_index" in test_df.columns:
            test_indices = set(test_df["source_row_index"].dropna().astype(int))
            df = df[~df["_source_row_index"].isin(test_indices)].copy()

        train_df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
        test_df = test_df.dropna(subset=[TEXT_COLUMN, test_label_column]).copy()
        X_train = train_df[TEXT_COLUMN].fillna("")
        y_train = train_df[LABEL_COLUMN].astype(str)
        X_test = test_df[TEXT_COLUMN].fillna("")
        y_test = test_df[test_label_column].astype(str)
    else:
        X = df[TEXT_COLUMN].fillna("")
        y = df[LABEL_COLUMN].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    results = []

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    for model_name, model in build_models().items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        results.append(
            {
                "feature": FEATURE_NAME,
                "model": model_name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )

        print("=" * 72)
        print("Task: Sentiment Classification")
        print("Feature:", FEATURE_NAME)
        print("Model:", model_name)
        print("Accuracy:", round(accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))
        print()
        print(classification_report(y_test, y_pred, zero_division=0))

    secondary_metric = "macro_f1" if PRIMARY_METRIC == "accuracy" else "accuracy"
    result_df = pd.DataFrame(results).sort_values(
        [PRIMARY_METRIC, secondary_metric],
        ascending=False,
    )

    print("=" * 72)
    print("Baseline vs final model comparison")
    print(result_df.to_string(index=False))

    best_result = result_df.iloc[0]
    print("=" * 72)
    print("Best result by", PRIMARY_METRIC)
    print("Feature:", best_result["feature"])
    print("Model:", best_result["model"])
    print("Accuracy:", round(best_result["accuracy"], 4))
    print("Macro F1:", round(best_result["macro_f1"], 4))


if __name__ == "__main__":
    main()
