
"""
CAP-07: Prompt Engineering Benchmark Scoring.

Measures automated execution metrics and supports
human-reviewed semantic correctness evaluation.
"""

import argparse
import csv
import json

from pathlib import Path
from statistics import mean
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "prompt_engineering"
    / "results"
    / "benchmark_v1.0"
)

SCORES = {
    "correct": 1.0,
    "partial": 0.5,
    "incorrect": 0.0,
    "execution_error": 0.0,
}


def percentage(numerator, denominator):

    if not denominator:
        return None

    return round(
        numerator / denominator * 100,
        2,
    )


def average(values):

    values = [
        value
        for value in values
        if value is not None
    ]

    return (
        round(mean(values), 3)
        if values
        else None
    )


def load_results(path):

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def token_summary(generations):

    metrics = {}

    for key in [
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]:

        values = [
            item.get("token_usage", {}).get(key)
            for item in generations
        ]

        available = [
            value
            for value in values
            if value is not None
        ]

        metrics[key] = {
            "recorded_total": (
                sum(available)
                if available
                else None
            ),
            "average_per_recorded_generation": (
                average(available)
            ),
            "recorded_generations": len(available),
            "coverage_pct": percentage(
                len(available),
                len(generations),
            ),
        }

    return metrics


def calculate_metrics(records):

    generations = [
        generation
        for record in records
        for generation in record["generations"]
    ]

    total_cases = len(records)

    total_generations = len(generations)

    status_cases = [
        record
        for record in records
        if record.get("expected_status")
        is not None
    ]

    selected_count = sum(
        record.get("selected_response")
        is not None
        for record in records
    )

    correct_statuses = sum(
        record.get("status_match") is True
        for record in status_cases
    )

    strict_json_count = sum(
        generation.get("strict_json_valid") is True
        for generation in generations
    )

    schema_valid_count = sum(
        generation.get("schema_valid") is True
        for generation in generations
    )

    technical_failures = sum(
        generation.get("execution_status")
        != "success"
        for generation in generations
    )

    selection_counts = Counter(
        record.get("selection_status")
        for record in records
    )

    return {
        "total_cases": total_cases,
        "total_generations": total_generations,
        "selected_responses": selected_count,

        "response_selection_rate_pct": percentage(
            selected_count,
            total_cases,
        ),

        "correct_statuses": correct_statuses,

        "status_accuracy_pct": percentage(
            correct_statuses,
            len(status_cases),
        ),

        "strict_json_validity_pct": percentage(
            strict_json_count,
            total_generations,
        ),

        "schema_validation_rate_pct": percentage(
            schema_valid_count,
            total_generations,
        ),

        "technical_failures": technical_failures,

        "average_generation_latency_seconds": average([
            generation.get("latency_seconds")
            for generation in generations
        ]),

        "average_task_latency_seconds": average([
            record.get("task_latency_seconds")
            for record in records
        ]),

        "selection_counts": dict(selection_counts),

        "token_usage": token_summary(generations),
    }


def create_review_sheet(records, output_path):

    if output_path.exists():

        print(
            "Existing review sheet preserved:",
            output_path,
        )

        return

    fieldnames = [
        "dataset_id",
        "strategy_id",
        "test_id",
        "category",
        "query",
        "gold_answer",
        "expected_status",
        "actual_status",
        "expected_amount",
        "expected_currency",
        "requires_clarification",
        "selected_response",
        "correctness",
        "failure_category",
        "review_notes",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:

            response = record.get(
                "selected_response"
            )

            correctness = ""
            failure_category = ""

            if response is None:

                if record.get(
                    "selection_status"
                ) == "no_consensus":

                    correctness = "incorrect"
                    failure_category = (
                        "consensus_failure"
                    )

                else:

                    correctness = "execution_error"
                    failure_category = (
                        "no_valid_response"
                    )

            writer.writerow({
                "dataset_id": record.get(
                    "dataset_id"
                ),
                "strategy_id": record.get(
                    "strategy_id"
                ),
                "test_id": record.get("test_id"),
                "category": record.get("category"),
                "query": record.get("query"),
                "gold_answer": record.get(
                    "gold_answer"
                ),
                "expected_status": record.get(
                    "expected_status"
                ),
                "actual_status": record.get(
                    "actual_status"
                ),
                "expected_amount": record.get(
                    "expected_amount"
                ),
                "expected_currency": record.get(
                    "expected_currency"
                ),
                "requires_clarification": (
                    record.get(
                        "requires_clarification"
                    )
                ),
                "selected_response": json.dumps(
                    response,
                    ensure_ascii=False,
                ),
                "correctness": correctness,
                "failure_category": failure_category,
                "review_notes": "",
            })


def load_reviews(path):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        return list(csv.DictReader(file))


def record_key(record):

    return (
        record["dataset_id"],
        record["strategy_id"],
        record["test_id"],
    )


def calculate_review_metrics(records, reviews):

    expected_keys = {
        record_key(record)
        for record in records
    }

    relevant_reviews = {}

    for review in reviews:

        key = record_key(review)

        if key not in expected_keys:
            continue

        if key in relevant_reviews:
            raise ValueError(
                f"Duplicate review: {key}"
            )

        relevant_reviews[key] = review

    reviewed = []

    for review in relevant_reviews.values():

        correctness = (
            review.get("correctness") or ""
        ).strip().lower()

        if not correctness:
            continue

        if correctness not in SCORES:

            raise ValueError(
                f"Invalid correctness: {correctness}"
            )

        reviewed.append({
            **review,
            "correctness": correctness,
        })

    counts = Counter(
        item["correctness"]
        for item in reviewed
    )

    failure_counts = Counter(
        item["failure_category"]
        for item in reviewed
        if item["correctness"] != "correct"
        and item.get("failure_category")
    )

    reviewed_count = len(reviewed)

    weighted_score = sum(
        SCORES[item["correctness"]]
        for item in reviewed
    )

    return {
        "total_cases": len(records),

        "reviewed_cases": reviewed_count,

        "pending_reviews": (
            len(records) - reviewed_count
        ),

        "review_coverage_pct": percentage(
            reviewed_count,
            len(records),
        ),

        "correctness_counts": dict(counts),

        "strict_accuracy_reviewed_pct": percentage(
            counts.get("correct", 0),
            reviewed_count,
        ),

        "weighted_accuracy_reviewed_pct": percentage(
            weighted_score,
            reviewed_count,
        ),

        "review_complete": (
            reviewed_count == len(records)
        ),

        "failure_taxonomy": dict(failure_counts),
    }


def run_scoring(run_name):

    run_dir = RESULTS_DIR / run_name

    if not run_dir.exists():

        raise FileNotFoundError(
            f"Benchmark directory not found: {run_dir}"
        )

    all_records = []
    strategy_records = {}

    for strategy_id in ["S0", "S1", "S2", "S3"]:

        result_path = (
            run_dir
            / f"{strategy_id}_results.jsonl"
        )

        if not result_path.exists():
            continue

        records = load_results(result_path)

        strategy_records[strategy_id] = records

        all_records.extend(records)

    if not all_records:

        raise ValueError(
            "No benchmark results found."
        )

    review_path = (
        run_dir / "correctness_review.csv"
    )

    create_review_sheet(
        all_records,
        review_path,
    )

    reviews = load_reviews(review_path)

    summary = {
        "benchmark_version": "1.0",
        "run_name": run_name,
        "total_strategy_cases": len(all_records),
        "strategies": {},
    }

    for strategy_id, records in (
        strategy_records.items()
    ):

        summary["strategies"][strategy_id] = {
            "automated_metrics": calculate_metrics(
                records
            ),
            "semantic_review": calculate_review_metrics(
                records,
                reviews,
            ),
        }

    output_path = (
        run_dir / "benchmark_summary.json"
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nPROMPT ENGINEERING BENCHMARK SUMMARY")
    print("=" * 60)

    for strategy_id, result in (
        summary["strategies"].items()
    ):

        metrics = result["automated_metrics"]
        semantic = result["semantic_review"]

        print(f"\nStrategy: {strategy_id}")

        print(
            "Cases:",
            metrics["total_cases"],
        )

        print(
            "Status accuracy:",
            metrics["status_accuracy_pct"],
        )

        print(
            "Schema validation rate:",
            metrics["schema_validation_rate_pct"],
        )

        print(
            "Average task latency:",
            metrics["average_task_latency_seconds"],
        )

        print(
            "Semantic reviews completed:",
            semantic["reviewed_cases"],
        )

        print(
            "Semantic accuracy:",
            semantic["strict_accuracy_reviewed_pct"],
        )

    print(f"\nSummary saved to: {output_path}")

    print(f"Review sheet saved to: {review_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Score prompt benchmark results."
    )

    parser.add_argument(
        "--run-name",
        default="smoke",
    )

    args = parser.parse_args()

    if (
        Path(args.run_name).name != args.run_name
        or args.run_name in {"", ".", ".."}
    ):
        parser.error("Invalid run name.")

    run_scoring(args.run_name)


if __name__ == "__main__":
    main()
