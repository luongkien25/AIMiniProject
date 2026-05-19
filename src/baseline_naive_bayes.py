import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    os.environ.get("BASELINE_DATA_PATH", "data/processed/shopee_reviews_labeled.csv")
)
REPORT_PATH = Path(
    os.environ.get("BASELINE_REPORT_PATH", "reports/baseline_sentiment_naive_bayes.txt")
)
TEXT_COLUMN = "cleaned_review"
LABEL_COLUMN = "sentiment"
RANDOM_STATE = 62
TEST_SIZE = 0.2
MIN_DF = int(os.environ.get("BASELINE_MIN_DF", "1"))
ALPHA = float(os.environ.get("BASELINE_ALPHA", "1.0"))
BINARY_NEG_POS = os.environ.get("BASELINE_BINARY_NEG_POS", "0") == "1"


def extract_features(text):
    tokens = str(text).split()
    bigrams = [
        f"{tokens[index]} {tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    ]
    return tokens + bigrams


def stratified_train_test_split(df, label_column, test_size, random_state):
    test_parts = []
    for _, group in df.groupby(label_column):
        test_count = max(1, round(len(group) * test_size))
        test_count = min(test_count, len(group) - 1)
        test_parts.append(group.sample(n=test_count, random_state=random_state))

    test_df = pd.concat(test_parts).sample(frac=1, random_state=random_state)
    train_df = df.drop(index=test_df.index)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_vocabulary(texts, min_df):
    document_frequency = Counter()

    for text in texts:
        document_frequency.update(set(extract_features(text)))

    return {
        feature
        for feature, count in document_frequency.items()
        if count >= min_df
    }


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0, min_df=1):
        self.alpha = alpha
        self.min_df = min_df
        self.vocabulary = set()
        self.classes = []
        self.log_prior = {}
        self.log_likelihood = {}
        self.default_log_likelihood = {}

    def fit(self, texts, labels):
        self.vocabulary = build_vocabulary(texts, self.min_df)
        self.classes = sorted(set(labels))

        label_counts = Counter(labels)
        total_documents = len(labels)
        vocabulary_size = len(self.vocabulary)

        feature_counts_by_class = {
            label: Counter()
            for label in self.classes
        }
        total_features_by_class = defaultdict(int)

        for text, label in zip(texts, labels):
            features = [
                feature
                for feature in extract_features(text)
                if feature in self.vocabulary
            ]
            feature_counts_by_class[label].update(features)
            total_features_by_class[label] += len(features)

        for label in self.classes:
            self.log_prior[label] = math.log(label_counts[label] / total_documents)

            denominator = (
                total_features_by_class[label]
                + self.alpha * vocabulary_size
            )
            self.default_log_likelihood[label] = math.log(self.alpha / denominator)
            self.log_likelihood[label] = {}

            for feature in self.vocabulary:
                count = feature_counts_by_class[label][feature]
                probability = (count + self.alpha) / denominator
                self.log_likelihood[label][feature] = math.log(probability)

        return self

    def predict_one(self, text):
        feature_counts = Counter(
            feature
            for feature in extract_features(text)
            if feature in self.vocabulary
        )
        scores = {}

        for label in self.classes:
            score = self.log_prior[label]
            for feature, count in feature_counts.items():
                score += count * self.log_likelihood[label].get(
                    feature,
                    self.default_log_likelihood[label],
                )
            scores[label] = score

        return max(scores, key=scores.get)

    def predict(self, texts):
        return [self.predict_one(text) for text in texts]


def calculate_metrics(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    rows = []
    correct = sum(true == pred for true, pred in zip(y_true, y_pred))
    accuracy = correct / len(y_true)

    for label in labels:
        true_positive = sum(
            true == label and pred == label
            for true, pred in zip(y_true, y_pred)
        )
        false_positive = sum(
            true != label and pred == label
            for true, pred in zip(y_true, y_pred)
        )
        false_negative = sum(
            true == label and pred != label
            for true, pred in zip(y_true, y_pred)
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(true == label for true in y_true),
            }
        )

    macro_f1 = sum(row["f1"] for row in rows) / len(rows)

    return accuracy, macro_f1, rows


def main():
    df = pd.read_csv(DATA_PATH)
    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)

    report_path = REPORT_PATH
    task_mode = "3-class sentiment"
    if BINARY_NEG_POS:
        df = df[df[LABEL_COLUMN].isin(["negative", "positive"])].copy()
        report_path = REPORT_PATH.with_name("baseline_sentiment_naive_bayes_binary.txt")
        task_mode = "binary negative vs positive"

    train_df, test_df = stratified_train_test_split(
        df,
        LABEL_COLUMN,
        TEST_SIZE,
        RANDOM_STATE,
    )

    model = MultinomialNaiveBayes(alpha=ALPHA, min_df=MIN_DF)
    model.fit(train_df[TEXT_COLUMN].tolist(), train_df[LABEL_COLUMN].tolist())
    predictions = model.predict(test_df[TEXT_COLUMN].tolist())

    accuracy, macro_f1, rows = calculate_metrics(
        test_df[LABEL_COLUMN].tolist(),
        predictions,
    )

    lines = [
        "=" * 72,
        "Task: Sentiment Classification",
        f"Mode: {task_mode}",
        "Model: From-scratch Multinomial Naive Bayes",
        "Feature: Unigram + Bigram counts",
        f"Alpha: {ALPHA}",
        f"Min DF: {MIN_DF}",
        f"Train rows: {len(train_df)}",
        f"Test rows: {len(test_df)}",
        f"Vocabulary size: {len(model.vocabulary)}",
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

    content = "\n".join(lines)
    print(content)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content + "\n", encoding="utf-8")
    print()
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
