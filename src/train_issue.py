import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from naive_bayes_model import calculate_metrics, train_naive_bayes


DATA_PATH = os.environ.get(
    "ISSUE_DATA_PATH",
    "data/processed/shopee_reviews_labeled.csv",
)
TEXT_COLUMN = "cleaned_review"
LABEL_COLUMN = "issue"
RANDOM_STATE = 62
TEST_SIZE = 0.2
NAIVE_BAYES_ALPHA = float(os.environ.get("NAIVE_BAYES_ALPHA", "1.0"))
NAIVE_BAYES_MIN_DF = int(os.environ.get("NAIVE_BAYES_MIN_DF", "1"))

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
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            C=1.0,
        ),
    }


def print_naive_bayes_report(X_train, y_train, X_test, y_test):
    model = train_naive_bayes(
        X_train.tolist(),
        y_train.tolist(),
        alpha=NAIVE_BAYES_ALPHA,
        min_df=NAIVE_BAYES_MIN_DF,
    )
    y_pred = model.predict(X_test.tolist())
    accuracy, macro_f1, rows = calculate_metrics(y_test.tolist(), y_pred)

    print("Task: Issue Classification")
    print("Feature: Unigram + Bigram counts")
    print("Model: Naive Bayes")
    print("Implementation: from scratch")
    print("Alpha:", NAIVE_BAYES_ALPHA)
    print("Min DF:", NAIVE_BAYES_MIN_DF)
    print("Vocabulary size:", len(model.vocabulary))
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print()
    print("Per-class metrics:")
    for row in rows:
        print(
            f"{row['label']:>18}  "
            f"precision={row['precision']:.4f}  "
            f"recall={row['recall']:.4f}  "
            f"f1={row['f1']:.4f}  "
            f"support={row['support']}"
        )

    return {
        "feature": "Unigram + Bigram counts",
        "model": "Naive Bayes",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
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
    results.append(print_naive_bayes_report(X_train, y_train, X_test, y_test))

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

    print("Baseline vs final model comparison")
    print(result_df.to_string(index=False))

    best_result = result_df.iloc[0]
    print("Best result by Macro F1")
    print("Feature:", best_result["feature"])
    print("Model:", best_result["model"])
    print("Accuracy:", round(best_result["accuracy"], 4))
    print("Macro F1:", round(best_result["macro_f1"], 4))


if __name__ == "__main__":
    main()
