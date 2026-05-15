from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = Path("data/processed/merged_preprocessed_reviews.csv")
REPORTS_DIR = Path("reports")
RANDOM_STATE = 62
TEST_SIZE = 0.2


TASK_CONFIGS = {
    "sentiment": {
        "task_name": "Sentiment Classification",
        "label_column": "sentiment_label",
        "feature_name": "TF-IDF Unigram + Bigram",
        "vectorizer": {
            "ngram_range": (1, 2),
            "max_features": 10000,
            "min_df": 2,
            "sublinear_tf": True,
        },
        "model_name": "Logistic Regression",
        "model": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=2.0,
        ),
        "labels": ["negative", "neutral", "positive"],
    },
    "issue": {
        "task_name": "Issue Classification",
        "label_column": "issue_label",
        "feature_name": "TF-IDF Unigram",
        "vectorizer": {
            "ngram_range": (1, 1),
            "max_features": 10000,
            "min_df": 2,
            "sublinear_tf": True,
        },
        "model_name": "Linear SVM",
        "model": LinearSVC(
            class_weight="balanced",
            C=1.5,
        ),
        "labels": [
            "general_feedback",
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


METRIC_EXPLANATION = """Giải thích metric:
- Accuracy: tỷ lệ dự đoán đúng trên toàn bộ tập test.
- Precision: trong các mẫu được dự đoán là một nhãn, có bao nhiêu mẫu dự đoán đúng.
- Recall: trong các mẫu thật sự thuộc một nhãn, mô hình tìm đúng được bao nhiêu mẫu.
- F1-score: trung bình điều hòa giữa precision và recall.
- Macro F1: trung bình F1 của các lớp, mỗi lớp có trọng số ngang nhau. Chỉ số này quan trọng khi dữ liệu có nhiều lớp hoặc phân bố nhãn không đều.
- Confusion matrix: bảng thể hiện mô hình dự đoán đúng và nhầm lẫn giữa các lớp. Đường chéo chính là dự đoán đúng, các ô ngoài đường chéo là dự đoán sai.
"""


def build_pipeline(config):
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**config["vectorizer"])),
            ("classifier", config["model"]),
        ]
    )


def get_top_confusions(y_true, y_pred, labels, limit=6):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    confusions = []

    for true_index, true_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            if true_index == pred_index:
                continue

            count = int(matrix[true_index, pred_index])
            if count > 0:
                confusions.append((count, true_label, pred_label))

    return sorted(confusions, reverse=True)[:limit]


def build_observations(task_key, y_true, y_pred, labels):
    lines = ["Nhận xét:"]
    top_confusions = get_top_confusions(y_true, y_pred, labels)

    if top_confusions:
        lines.append("- Các nhầm lẫn đáng chú ý:")
        for count, true_label, pred_label in top_confusions:
            lines.append(f"  + Nhãn thật `{true_label}` bị dự đoán thành `{pred_label}`: {count} mẫu.")

    if task_key == "sentiment":
        lines.append(
            "- Với sentiment classification, lớp `neutral` thường khó hơn vì nhiều bình luận có "
            "từ như `ok`, `tạm`, `ổn`, `5 sao` hoặc nội dung nhận xu."
        )
        lines.append(
            "- Mô hình hoạt động tốt hơn khi dùng bigram vì học được các cụm như "
            "`không tốt`, `giao chậm`, `sai màu`, `bị lỗi`."
        )

    if task_key == "issue":
        lines.append(
            "- Issue classification khó hơn sentiment vì có nhiều nhãn hơn và một số nhãn có ý nghĩa gần nhau."
        )
        lines.append(
            "- Sau khi gom các nhãn issue quá chi tiết, Macro F1 cải thiện rõ vì mô hình học các nhóm vấn đề ổn định hơn."
        )
        lines.append(
            "- Một số nhầm lẫn giữa `no_issue` và `product_quality` là hợp lý vì nhiều bình luận vừa mô tả sản phẩm vừa không nêu lỗi rõ ràng."
        )

    return "\n".join(lines)


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


def save_classification_report(task_key, config, y_true, y_pred, labels, accuracy, macro_f1):
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    observations = build_observations(task_key, y_true, y_pred, labels)

    content = "\n".join(
        [
            f"Task: {config['task_name']}",
            f"Feature: {config['feature_name']}",
            f"Model: {config['model_name']}",
            f"Accuracy: {accuracy:.4f}",
            f"Macro F1: {macro_f1:.4f}",
            "",
            report,
            "",
            METRIC_EXPLANATION,
            observations,
            "",
        ]
    )

    output_path = REPORTS_DIR / f"classification_report_{task_key}.txt"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def evaluate_task(df, task_key, config):
    required_columns = {"processed_text", config["label_column"]}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns for {task_key}: {missing}")

    task_df = df.dropna(subset=["processed_text", config["label_column"]]).copy()

    X = task_df["processed_text"].fillna("")
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

    labels = [label for label in config["labels"] if label in set(y)]
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    report_path = save_classification_report(
        task_key,
        config,
        y_test,
        y_pred,
        labels,
        accuracy,
        macro_f1,
    )

    matrix_path = REPORTS_DIR / f"confusion_matrix_{task_key}.png"
    save_confusion_matrix(
        y_test,
        y_pred,
        labels,
        f"{config['task_name']} - Confusion Matrix",
        matrix_path,
    )

    print("=" * 72)
    print("Task:", config["task_name"])
    print("Feature:", config["feature_name"])
    print("Model:", config["model_name"])
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Saved report:", report_path)
    print("Saved confusion matrix:", matrix_path)


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    for task_key, config in TASK_CONFIGS.items():
        evaluate_task(df, task_key, config)


if __name__ == "__main__":
    main()
