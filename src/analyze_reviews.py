from pathlib import Path
import argparse
from collections import Counter
import csv
import os
import sys

from predict import load_models, predict_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYZED_FIELDS = [
    "review_id",
    "review_text",
    "sentiment",
    "issue",
]

SUMMARY_FIELDS = ["section", "label", "count", "percentage"]

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
ISSUE_ORDER = [
    "no_issue",
    "product_quality",
    "product_attribute",
    "packaging",
    "wrong_missing_item",
    "seller_service",
    "spam_irrelevant",
    "shipping_delivery",
    "price_value",
]


def configure_csv_field_size():
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10


def get_output_path(input_path, output_path):
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}_analyzed{input_path.suffix}")


def get_summary_output_path(input_path, summary_output_path):
    if summary_output_path is not None:
        return summary_output_path
    return input_path.with_name(f"{input_path.stem}_summary.csv")


def get_text_column(fieldnames, requested_column):
    if requested_column:
        if requested_column not in fieldnames:
            raise ValueError(f"Cannot find text column: {requested_column}")
        return requested_column

    if "review_text" in fieldnames:
        return "review_text"

    if len(fieldnames) < 2:
        raise ValueError("Input CSV must have at least two columns: review_id, review_text")

    return fieldnames[1]


def percentage(count, total):
    if total == 0:
        return "0.00"
    return f"{count / total * 100:.2f}"


def ordered_counter_items(counter, preferred_order=None):
    preferred_order = preferred_order or []
    seen = set()

    for label in preferred_order:
        if counter.get(label, 0) > 0:
            seen.add(label)
            yield label, counter[label]

    remaining_items = [
        (label, count)
        for label, count in counter.items()
        if label not in seen and label
    ]
    yield from sorted(remaining_items, key=lambda item: (-item[1], item[0]))


def add_summary_counter_rows(summary_rows, section, counter, total, preferred_order=None):
    for label, count in ordered_counter_items(counter, preferred_order):
        summary_rows.append(
            {
                "section": section,
                "label": label,
                "count": count,
                "percentage": percentage(count, total),
            }
        )


def build_summary_rows(rows):
    total_rows = len(rows)
    analyzed_rows = [row for row in rows if row["note"] != "empty_review"]
    analyzed_count = len(analyzed_rows)
    empty_count = total_rows - analyzed_count

    sentiment_counter = Counter(row["sentiment"] for row in analyzed_rows)
    issue_counter = Counter(row["issue"] for row in analyzed_rows)
    combination_counter = Counter(
        f"{row['sentiment']} | {row['issue']}"
        for row in analyzed_rows
    )

    summary_rows = [
        {
            "section": "overview",
            "label": "total_rows",
            "count": total_rows,
            "percentage": "100.00" if total_rows else "0.00",
        },
        {
            "section": "overview",
            "label": "analyzed_reviews",
            "count": analyzed_count,
            "percentage": percentage(analyzed_count, total_rows),
        },
        {
            "section": "overview",
            "label": "empty_reviews",
            "count": empty_count,
            "percentage": percentage(empty_count, total_rows),
        },
    ]

    add_summary_counter_rows(
        summary_rows,
        "sentiment",
        sentiment_counter,
        analyzed_count,
        SENTIMENT_ORDER,
    )
    add_summary_counter_rows(
        summary_rows,
        "issue",
        issue_counter,
        analyzed_count,
        ISSUE_ORDER,
    )
    add_summary_counter_rows(
        summary_rows,
        "sentiment_issue",
        combination_counter,
        analyzed_count,
    )

    return summary_rows


def write_csv(output_path, fieldnames, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def analyze_csv(input_path, output_path=None, summary_output_path=None, text_column=None):
    input_path = Path(input_path)
    output_path = get_output_path(input_path, Path(output_path) if output_path else None)
    summary_output_path = get_summary_output_path(
        input_path,
        Path(summary_output_path) if summary_output_path else None,
    )

    os.chdir(PROJECT_ROOT)
    sentiment_model, issue_model = load_models()

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError("Input CSV is empty or missing a header row")

        selected_text_column = get_text_column(reader.fieldnames, text_column)

        rows = []
        for row_number, row in enumerate(reader, start=1):
            review_id = row.get("review_id") or str(row_number)
            review_text = str(row.get(selected_text_column, "")).strip()

            if review_text:
                result = predict_review(sentiment_model, issue_model, review_text)
                analyzed_row = {
                    "review_id": review_id,
                    "review_text": review_text,
                    "cleaned_review": result["processed_text"],
                    "sentiment": result["sentiment"],
                    "issue": result["issue"],
                    "model_sentiment": result["model_sentiment"],
                    "model_issue": result["model_issue"],
                    "sentiment_rule": result["sentiment_rule"] or "",
                    "issue_rule": result["issue_rule"] or "",
                    "note": "",
                }
            else:
                analyzed_row = {
                    "review_id": review_id,
                    "review_text": review_text,
                    "cleaned_review": "",
                    "sentiment": "",
                    "issue": "",
                    "model_sentiment": "",
                    "model_issue": "",
                    "sentiment_rule": "",
                    "issue_rule": "",
                    "note": "empty_review",
                }

            rows.append(analyzed_row)

    summary_rows = build_summary_rows(rows)
    write_csv(output_path, ANALYZED_FIELDS, rows)
    write_csv(summary_output_path, SUMMARY_FIELDS, summary_rows)

    return output_path, summary_output_path, len(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Shopee review CSV with saved sentiment and issue models.",
    )
    parser.add_argument(
        "input_csv",
        help="CSV input file with columns review_id and review_text.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path. Default: <input>_analyzed.csv.",
    )
    parser.add_argument(
        "--summary-output",
        help="Summary CSV path. Default: <input>_summary.csv.",
    )
    parser.add_argument(
        "--text-column",
        help="Optional review text column name. Default: review_text or the second column.",
    )
    return parser.parse_args()


def main():
    configure_csv_field_size()
    args = parse_args()
    output_path, summary_output_path, row_count = analyze_csv(
        input_path=args.input_csv,
        output_path=args.output,
        summary_output_path=args.summary_output,
        text_column=args.text_column,
    )
    print(f"Analyzed rows: {row_count}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_output_path}")


if __name__ == "__main__":
    main()
