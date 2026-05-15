from pathlib import Path

import joblib

from preprocess import preprocess_text


SENTIMENT_MODEL_PATH = Path("models/sentiment_model.joblib")
ISSUE_MODEL_PATH = Path("models/issue_model.joblib")


def load_models():
    missing_paths = [
        str(path)
        for path in [SENTIMENT_MODEL_PATH, ISSUE_MODEL_PATH]
        if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise FileNotFoundError(
            f"Cannot find model file(s): {missing}. Run `python src/save_models.py` first."
        )

    sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
    issue_model = joblib.load(ISSUE_MODEL_PATH)
    return sentiment_model, issue_model


def get_confidence(model, processed_text):
    classifier = model.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "predict_proba"):
        return None

    probabilities = model.predict_proba([processed_text])[0]
    return probabilities.max()


def predict_review(sentiment_model, issue_model, review_text):
    processed_text = preprocess_text(review_text)
    sentiment = sentiment_model.predict([processed_text])[0]
    issue = issue_model.predict([processed_text])[0]
    confidence = get_confidence(sentiment_model, processed_text)

    return {
        "processed_text": processed_text,
        "sentiment": sentiment,
        "issue": issue,
        "sentiment_confidence": confidence,
    }


def print_prediction(result):
    sentiment_display = str(result["sentiment"]).capitalize()

    print(f"Processed text: {result['processed_text']}")
    print(f"Sentiment: {sentiment_display}")
    print(f"Issue: {result['issue']}")

    if result["sentiment_confidence"] is not None:
        print(f"Sentiment confidence: {result['sentiment_confidence']:.2%}")

    print()


def main():
    sentiment_model, issue_model = load_models()

    print("Demo dự đoán bình luận khách hàng")
    print("Nhập một bình luận rồi nhấn Enter.")
    print("Ví dụ: shop giao chậm, hộp bị móp")
    print("Gõ `exit` hoặc `quit` để thoát.\n")

    while True:
        review_text = input("Review: ").strip()
        if review_text.lower() in {"exit", "quit"}:
            break
        if not review_text:
            print("Vui lòng nhập bình luận không rỗng.\n")
            continue

        result = predict_review(sentiment_model, issue_model, review_text)
        print_prediction(result)


if __name__ == "__main__":
    main()
