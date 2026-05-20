from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from predict import get_safe_issue_rule_override


SENTIMENT_DATA_PATH = Path("data/processed/shopee_reviews_labeled.csv")
ISSUE_DATA_PATH = Path("data/processed/shopee_reviews_labeled.csv")
REPORTS_DIR = Path("reports")
RANDOM_STATE = 62
TEST_SIZE = 0.2
TEXT_COLUMN = "cleaned_review"


TASK_CONFIGS = {
    "sentiment": {
        "task_name": "Sentiment Classification",
        "data_path": SENTIMENT_DATA_PATH,
        "label_column": "sentiment",
        "feature_name": "TF-IDF Unigram + Bigram",
        "vectorizer": {
            "ngram_range": (1, 2),
            "max_features": 10000,
            "min_df": 2,
            "sublinear_tf": True,
        },
        "model_name": "Linear SVM",
        "model": LinearSVC(
            class_weight="balanced",
            C=0.5,
        ),
        "labels": ["negative", "neutral", "positive"],
    },
    "issue": {
        "task_name": "Issue Classification",
        "data_path": ISSUE_DATA_PATH,
        "label_column": "issue",
        "feature_name": "TF-IDF Word + Char",
        "vectorizer": FeatureUnion(
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
        "model_name": "Linear SVM + Safe Issue Rules",
        "model": LinearSVC(
            class_weight="balanced",
            C=1.0,
        ),
        "issue_rule_override": "safe",
        "labels": [
            "no_issue",
            "packaging",
            "price_value",
            "product_attribute",
            "product_quality",
            "seller_service",
            "shipping_delivery",
            "spam_irrelevant",
            "wrong_missing_item",
        ],
    },
}


def build_pipeline(config):
    vectorizer_config = config["vectorizer"]
    if isinstance(vectorizer_config, dict):
        vectorizer = TfidfVectorizer(**vectorizer_config)
    else:
        vectorizer = vectorizer_config

    return Pipeline(
        steps=[
            ("features", vectorizer),
            ("classifier", config["model"]),
        ]
    )


def save_confusion_matrix(y_true, y_pred, labels, title, output_path):
    fig_width = max(7, len(labels) * 0.85)
    fig_height = max(5, len(labels) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_task(df, task_key, config):
    required_columns = {TEXT_COLUMN, config["label_column"]}
    if config.get("issue_rule_override") == "safe":
        required_columns.add("review_text")

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns for {task_key}: {missing}")

    task_df = df.dropna(subset=[TEXT_COLUMN, config["label_column"]]).copy()

    X = task_df[TEXT_COLUMN].fillna("")
    y = task_df[config["label_column"]].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if config.get("issue_rule_override") == "safe":
        y_pred = apply_safe_issue_rules(task_df.loc[X_test.index], y_pred)

    labels = [label for label in config["labels"] if label in set(y)]
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    matrix_path = REPORTS_DIR / f"confusion_matrix_{task_key}.png"
    save_confusion_matrix(
        y_test,
        y_pred,
        labels,
        f"{config['task_name']} - Confusion Matrix",
        matrix_path,
    )

    print("Task:", config["task_name"])
    print("Feature:", config["feature_name"])
    print("Model:", config["model_name"])
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Saved confusion matrix:", matrix_path)


def apply_safe_issue_rules(test_df, model_predictions):
    predictions = []
    for model_prediction, (_, row) in zip(model_predictions, test_df.iterrows()):
        rule_issue, _ = get_safe_issue_rule_override(
            str(row.get("review_text", "")),
            str(row.get(TEXT_COLUMN, "")),
        )
        predictions.append(rule_issue or model_prediction)

    return predictions


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    for task_key, config in TASK_CONFIGS.items():
        df = pd.read_csv(config["data_path"])
        evaluate_task(df, task_key, config)


if __name__ == "__main__":
    main()
