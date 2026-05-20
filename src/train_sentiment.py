import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from naive_bayes_model import calculate_metrics, train_naive_bayes


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
NAIVE_BAYES_ALPHA = float(os.environ.get("NAIVE_BAYES_ALPHA", "1.0"))
NAIVE_BAYES_MIN_DF = int(os.environ.get("NAIVE_BAYES_MIN_DF", "1"))

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
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            C=0.5,
        ),
    }


def format_naive_bayes_report(
    task_name,
    mode,
    train_rows,
    test_rows,
    vocabulary_size,
    accuracy,
    macro_f1,
    rows,
):
    lines = [
        f"Task: {task_name}",
        f"Mode: {mode}",
        "Model: From-scratch Multinomial Naive Bayes",
        "Feature: Unigram + Bigram counts",
        f"Alpha: {NAIVE_BAYES_ALPHA}",
        f"Min DF: {NAIVE_BAYES_MIN_DF}",
        f"Train rows: {train_rows}",
        f"Test rows: {test_rows}",
        f"Vocabulary size: {vocabulary_size}",
        f"Accuracy: {accuracy:.4f}",
        f"Macro F1: {macro_f1:.4f}",
        "",
        "Per-class metrics:",
    ]

    for row in rows:
        lines.append(
            f"{row['label']:>10}  "
            f"precision={row['precision']:.4f}  "
            f"recall={row['recall']:.4f}  "
            f"f1={row['f1']:.4f}  "
            f"support={row['support']}"
        )

    return "\n".join(lines)


def run_naive_bayes_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    mode,
):
    model = train_naive_bayes(
        X_train.tolist(),
        y_train.tolist(),
        alpha=NAIVE_BAYES_ALPHA,
        min_df=NAIVE_BAYES_MIN_DF,
    )
    y_pred = model.predict(X_test.tolist())
    accuracy, macro_f1, rows = calculate_metrics(y_test.tolist(), y_pred)

    report = format_naive_bayes_report(
        task_name="Sentiment Classification",
        mode=mode,
        train_rows=len(X_train),
        test_rows=len(X_test),
        vocabulary_size=len(model.vocabulary),
        accuracy=accuracy,
        macro_f1=macro_f1,
        rows=rows,
    )
    print(report)

    return {
        "feature": "Unigram + Bigram counts",
        "model": "Naive Bayes",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
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
    results.append(
        run_naive_bayes_experiment(
            X_train,
            y_train,
            X_test,
            y_test,
            mode="3-class sentiment",
        )
    )

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

    print("Baseline vs final model comparison")
    print(result_df.to_string(index=False))

    best_result = result_df.iloc[0]
    print("Best result by", PRIMARY_METRIC)
    print("Feature:", best_result["feature"])
    print("Model:", best_result["model"])
    print("Accuracy:", round(best_result["accuracy"], 4))
    print("Macro F1:", round(best_result["macro_f1"], 4))


if __name__ == "__main__":
    main()
