"""From-scratch Multinomial Naive Bayes model utilities.

This module contains only the model logic. Data loading, train/test split,
report writing, and comparison with Linear SVM live in the train scripts.
"""

import math
from collections import Counter, defaultdict


def extract_features(text):
    tokens = str(text).split()
    bigrams = [
        f"{tokens[index]} {tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    ]
    return tokens + bigrams


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


def train_naive_bayes(texts, labels, alpha=1.0, min_df=1):
    model = MultinomialNaiveBayes(alpha=alpha, min_df=min_df)
    return model.fit(texts, labels)


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
