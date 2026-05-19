import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


DATA_PATH = os.environ.get(
    "ISSUE_DATA_PATH",
    "data/processed/shopee_reviews_labeled.csv",
)
TEXT_COLUMN = "cleaned_review"
LABEL_COLUMN = "issue"
RANDOM_STATE = 62
TEST_SIZE = 0.2

FEATURE_NAME = "TF-IDF Word + Char"


def build_vectorizer():
    return FeatureUnion(
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
    )


def build_models():
    return {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            C=1.0,
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()

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
        print("Task: Issue Classification")
        print("Feature:", FEATURE_NAME)
        print("Model:", model_name)
        print("Accuracy:", round(accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))
        print()
        print(classification_report(y_test, y_pred, zero_division=0))

    result_df = pd.DataFrame(results).sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    )

    print("=" * 72)
    print("Baseline vs final model comparison")
    print(result_df.to_string(index=False))

    best_result = result_df.iloc[0]
    print("=" * 72)
    print("Best result by Macro F1")
    print("Feature:", best_result["feature"])
    print("Model:", best_result["model"])
    print("Accuracy:", round(best_result["accuracy"], 4))
    print("Macro F1:", round(best_result["macro_f1"], 4))


if __name__ == "__main__":
    main()
