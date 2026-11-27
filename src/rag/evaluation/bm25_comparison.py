import hashlib
import json
import re
import time

from pathlib import Path
from statistics import mean

from src.rag.retrieval.bm25_retriever import (
    BM25Retriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "real_corpus"
)

EVALUATION_FILE = (
    EVALUATION_DIR
    / "retrieval_eval_v1.0.json"
)

HISTORICAL_FILE = (
    EVALUATION_DIR
    / "bm25_retrieval_results_v1.0.json"
)

OUTPUT_FILE = (
    EVALUATION_DIR
    / "cap13_bm25_comparison_v1.0.json"
)

TOP_K = 10

RECALL_LEVELS = (1, 3, 5, 10)


class OriginalBM25Retriever(BM25Retriever):
    """Reproduce the original text-only tokenizer."""

    def __init__(self, chunk_file=None):

        super().__init__(
            chunk_file=chunk_file,
            include_metadata=False,
        )

    @staticmethod
    def _tokenize(text):

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )


def file_hash(path):

    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def recall_at_k(expected, retrieved, k):

    expected = set(expected)

    if not expected:

        raise ValueError(
            "Gold evidence cannot be empty."
        )

    return (
        len(expected.intersection(retrieved[:k]))
        / len(expected)
    )


def reciprocal_rank(expected, retrieved):

    expected = set(expected)

    for rank, chunk_id in enumerate(
        retrieved,
        start=1,
    ):

        if chunk_id in expected:

            return 1.0 / rank

    return 0.0


def first_relevant_rank(expected, retrieved):

    expected = set(expected)

    for rank, chunk_id in enumerate(
        retrieved,
        start=1,
    ):

        if chunk_id in expected:

            return rank

    return None


def percentile(values, fraction):

    ordered = sorted(values)

    if not ordered:

        return 0.0

    position = (
        len(ordered) - 1
    ) * fraction

    lower = int(position)

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    weight = position - lower

    return (
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


def load_queries():

    dataset = json.loads(
        EVALUATION_FILE.read_text(
            encoding="utf-8-sig"
        )
    )

    queries = dataset["queries"]

    if len(queries) != 20:

        raise ValueError(
            "Expected the frozen 20-query dataset."
        )

    query_ids = [
        item["query_id"]
        for item in queries
    ]

    if len(set(query_ids)) != len(query_ids):

        raise ValueError(
            "Duplicate evaluation query identifiers."
        )

    for item in queries:

        if not item["expected_chunk_ids"]:

            raise ValueError(
                f"Missing gold evidence: "
                f"{item['query_id']}"
            )

    return queries


def validate_corpus(retriever, queries):

    chunk_ids = [
        chunk.chunk_id
        for chunk in retriever.chunks
    ]

    if len(set(chunk_ids)) != len(chunk_ids):

        raise ValueError(
            "Duplicate chunk identifiers in corpus."
        )

    available = set(chunk_ids)

    expected = {
        chunk_id
        for item in queries
        for chunk_id in item["expected_chunk_ids"]
    }

    missing = expected - available

    if missing:

        raise ValueError(
            "Gold evidence missing from corpus: "
            + ", ".join(sorted(missing))
        )

    return len(chunk_ids)


def evaluate(
    name,
    retriever,
    queries,
):

    corpus_size = validate_corpus(
        retriever,
        queries,
    )

    rows = []

    for item in queries:

        query = item["query"]

        expected = item["expected_chunk_ids"]

        expected_documents = set(
            item["expected_document_ids"]
        )

        started = time.perf_counter()

        result = retriever.retrieve(
            query=query,
            top_k=TOP_K,
        )

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        retrieved = [
            candidate.chunk.chunk_id
            for candidate in result.retrieved_chunks
        ]

        documents = [
            candidate.chunk.document_id
            for candidate in result.retrieved_chunks
        ]

        row = {
            "query_id": item["query_id"],
            "category": item["category"],
            "query": query,
            "expected_chunk_ids": expected,
            "expected_document_ids": sorted(
                expected_documents
            ),
            "retrieved_chunk_ids": retrieved,
            "retrieved_document_ids": documents,
            "first_relevant_rank": first_relevant_rank(
                expected,
                retrieved,
            ),
            "reciprocal_rank": reciprocal_rank(
                expected,
                retrieved,
            ),
            "document_hit_at_5": int(
                bool(
                    expected_documents.intersection(
                        documents[:5]
                    )
                )
            ),
            "latency_ms": latency_ms,
        }

        for k in RECALL_LEVELS:

            row[f"recall_at_{k}"] = recall_at_k(
                expected,
                retrieved,
                k,
            )

        rows.append(row)

        print(
            f"{name} {item['query_id']}: "
            f"R@5={row['recall_at_5']:.2f}, "
            f"rank={row['first_relevant_rank']}, "
            f"RR={row['reciprocal_rank']:.3f}"
        )

    summary = {
        "configuration": name,
        "total_queries": len(rows),
        "corpus_chunks": corpus_size,
        "top_k": TOP_K,
        "mrr_at_10": mean(
            row["reciprocal_rank"]
            for row in rows
        ),
        "document_hit_rate_at_5": mean(
            row["document_hit_at_5"]
            for row in rows
        ),
        "mean_latency_ms": mean(
            row["latency_ms"]
            for row in rows
        ),
        "p95_latency_ms": percentile(
            [
                row["latency_ms"]
                for row in rows
            ],
            0.95,
        ),
        "zero_recall_at_5_queries": sum(
            row["recall_at_5"] == 0
            for row in rows
        ),
    }

    for k in RECALL_LEVELS:

        summary[f"recall_at_{k}"] = mean(
            row[f"recall_at_{k}"]
            for row in rows
        )

    return {
        "summary": summary,
        "query_results": rows,
    }


def compare(baseline, candidate):

    baseline_rows = {
        row["query_id"]: row
        for row in baseline["query_results"]
    }

    changes = []

    for row in candidate["query_results"]:

        original = baseline_rows[
            row["query_id"]
        ]

        delta = (
            row["reciprocal_rank"]
            - original["reciprocal_rank"]
        )

        if delta > 1e-12:

            status = "improved"

        elif delta < -1e-12:

            status = "regressed"

        else:

            status = "unchanged"

        changes.append({
            "query_id": row["query_id"],
            "category": row["category"],
            "baseline_rank": original[
                "first_relevant_rank"
            ],
            "candidate_rank": row[
                "first_relevant_rank"
            ],
            "reciprocal_rank_delta": delta,
            "recall_at_5_delta": (
                row["recall_at_5"]
                - original["recall_at_5"]
            ),
            "status": status,
        })

    metrics = [
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "document_hit_rate_at_5",
    ]

    return {
        "baseline": baseline["summary"][
            "configuration"
        ],
        "candidate": candidate["summary"][
            "configuration"
        ],
        "metric_deltas": {
            metric: (
                candidate["summary"][metric]
                - baseline["summary"][metric]
            )
            for metric in metrics
        },
        "improved_queries": sum(
            item["status"] == "improved"
            for item in changes
        ),
        "regressed_queries": sum(
            item["status"] == "regressed"
            for item in changes
        ),
        "unchanged_queries": sum(
            item["status"] == "unchanged"
            for item in changes
        ),
        "query_changes": changes,
    }


def historical_comparison(baseline):

    historical = json.loads(
        HISTORICAL_FILE.read_text(
            encoding="utf-8-sig"
        )
    )

    old = historical["summary"]

    current = baseline["summary"]

    pairs = {
        "recall_at_1": "recall_at_1",
        "recall_at_3": "recall_at_3",
        "recall_at_5": "recall_at_5",
        "mrr": "mrr_at_10",
    }

    deltas = {
        old_name: (
            current[new_name] - old[old_name]
        )
        for old_name, new_name in pairs.items()
    }

    previous_rows = {
        row["query_id"]: row
        for row in historical["query_results"]
    }

    ranking_matches = sum(
        previous_rows[row["query_id"]][
            "retrieved_chunk_ids"
        ] == row["retrieved_chunk_ids"]
        for row in baseline["query_results"]
    )

    return {
        "historical_summary": old,
        "metric_deltas": deltas,
        "metrics_reproduced": all(
            abs(value) < 1e-10
            for value in deltas.values()
        ),
        "identical_query_rankings": ranking_matches,
        "total_queries": len(
            baseline["query_results"]
        ),
    }


def main():

    queries = load_queries()

    original = OriginalBM25Retriever()

    chunk_file = original.chunk_file

    corpus_hash = file_hash(chunk_file)

    configurations = {
        "B0_original": original,
        "B1_normalized": BM25Retriever(
            chunk_file=str(chunk_file),
            include_metadata=False,
        ),
        "B2_metadata": BM25Retriever(
            chunk_file=str(chunk_file),
            include_metadata=True,
        ),
    }

    results = {}

    for name, retriever in configurations.items():

        print("\n========== " + name + " ==========")

        results[name] = evaluate(
            name,
            retriever,
            queries,
        )

    comparisons = {
        "B1_vs_B0": compare(
            results["B0_original"],
            results["B1_normalized"],
        ),
        "B2_vs_B0": compare(
            results["B0_original"],
            results["B2_metadata"],
        ),
        "B2_vs_B1": compare(
            results["B1_normalized"],
            results["B2_metadata"],
        ),
    }

    historical = historical_comparison(
        results["B0_original"]
    )

    if file_hash(chunk_file) != corpus_hash:

        raise RuntimeError(
            "Corpus changed during evaluation."
        )

    report = {
        "experiment": "CAP-13 BM25 ablation",
        "evaluation_dataset": EVALUATION_FILE.name,
        "dataset_sha256": file_hash(
            EVALUATION_FILE
        ),
        "corpus_file": Path(chunk_file).name,
        "corpus_sha256": corpus_hash,
        "total_queries": len(queries),
        "hosted_api_calls": 0,
        "historical_baseline_verification": historical,
        "configurations": results,
        "comparisons": comparisons,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n========== COMPARATIVE SUMMARY ==========")

    print(
        f"{'Configuration':<20}"
        f"{'R@1':>9}"
        f"{'R@3':>9}"
        f"{'R@5':>9}"
        f"{'R@10':>9}"
        f"{'MRR@10':>10}"
    )

    for name, result in results.items():

        summary = result["summary"]

        print(
            f"{name:<20}"
            f"{summary['recall_at_1']:>9.4f}"
            f"{summary['recall_at_3']:>9.4f}"
            f"{summary['recall_at_5']:>9.4f}"
            f"{summary['recall_at_10']:>9.4f}"
            f"{summary['mrr_at_10']:>10.4f}"
        )

    print("\n========== HISTORICAL VERIFICATION ==========")

    print(json.dumps(
        historical,
        indent=2,
    ))

    print("\n========== ENHANCED VS ORIGINAL ==========")

    comparison = comparisons["B2_vs_B0"]

    print(
        "Improved queries:",
        comparison["improved_queries"],
    )

    print(
        "Regressed queries:",
        comparison["regressed_queries"],
    )

    print(
        "Unchanged queries:",
        comparison["unchanged_queries"],
    )

    print(
        "Metric deltas:",
        json.dumps(
            comparison["metric_deltas"],
            indent=2,
        ),
    )

    print("\n========== REGRESSED QUERIES ==========")

    for row in comparison["query_changes"]:

        if row["status"] == "regressed":

            print(json.dumps(row))

    print("\nHosted API calls: 0")

    print("Results saved:", OUTPUT_FILE)


if __name__ == "__main__":

    main()
