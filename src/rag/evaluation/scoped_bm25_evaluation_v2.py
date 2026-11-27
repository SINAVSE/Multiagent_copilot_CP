
from pathlib import Path
from collections import Counter
import hashlib
import json
import statistics
import math
import time

from src.agents.policy_scope import PolicyScope

from src.agents.applicability_registry import (
    ApplicabilityRegistry,
)

from src.rag.retrieval.applicability_filter import (
    ApplicabilityFilter,
)

from src.rag.retrieval.bm25_retriever import (
    BM25Retriever,
)

from src.rag.retrieval.scoped_bm25_retriever import (
    ScopedBM25Retriever,
)

from src.rag.ingestion.vector_ingestion_v2 import (
    preflight,
)


ROOT = Path("data/real_corpus")

CORPUS = ROOT / "expanded_v2"

MANIFEST = (
    CORPUS / "provisional_manifest_v2.jsonl"
)

CHUNKS = (
    CORPUS / "chunking_experiments"
    / "c3_chunks.jsonl"
)

REGISTRY = (
    ROOT / "metadata"
    / "policy_applicability_draft_v2.json"
)

OUTPUT = (
    ROOT / "metadata"
    / "scoped_bm25_evaluation_results_v2.json"
)

EXPECTED_DATASET_HASH = (
    "2678ba31570f0fa854d06da10b850a654"
    "0fff700d61eea2547115a21f1b809fc"
)

K_VALUES = (1, 3, 5, 10)


def sha256(data):

    return hashlib.sha256(data).hexdigest()


def normalize(text):

    return " ".join(text.split())


def locate_dataset():

    matches = list(
        Path("data").rglob(
            "retrieval_eval_v2_draft.json"
        )
    )

    verified = [
        path
        for path in matches
        if sha256(path.read_bytes())
        == EXPECTED_DATASET_HASH
    ]

    if not verified:

        raise ValueError(
            "Frozen V2 evaluation dataset not found."
        )

    return verified[0]


def percentile95(values):

    ordered = sorted(values)

    index = math.ceil(
        len(ordered) * 0.95
    ) - 1

    return ordered[index]


def main():

    chunks, corpus_summary = preflight()

    dataset_path = locate_dataset()

    dataset = json.loads(
        dataset_path.read_text(
            encoding="utf-8-sig"
        )
    )

    assert dataset["manifest_sha256"] == (
        corpus_summary["manifest_sha256"]
    )

    queries = dataset["queries"]

    assert len(queries) == 40

    assert len({
        item["query_id"] for item in queries
    }) == 40

    documents = {
        doc["document_id"]: doc
        for doc in (
            json.loads(line)
            for line in MANIFEST.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        )
    }

    registry = ApplicabilityRegistry(
        MANIFEST, REGISTRY
    )

    selector = ApplicabilityFilter(registry)

    bm25 = BM25Retriever(
        chunk_file=str(CHUNKS)
    )

    scoped = ScopedBM25Retriever(
        bm25, selector
    )

    results = []

    for case in queries:

        query_id = case["query_id"]

        query = case["query"]

        scope = PolicyScope(
            case["organization"],
            case.get("country"),
        )

        expected_ids = set(
            case["expected_document_ids"]
        )

        assert expected_ids

        assert expected_ids.issubset(
            documents
        )

        evidence_spans = []

        for span in case["evidence_spans"]:

            document_id = span["document_id"]

            document = documents[document_id]

            raw = Path(
                document["processed_path"]
            ).read_bytes()

            assert sha256(raw) == (
                document["processed_file_hash"]
            )

            assert span["normalization"] == (
                "whitespace_join"
            )

            text = normalize(
                raw.decode("utf-8-sig")
            )

            passage = text[
                span["start"]:span["end"]
            ]

            assert passage

            assert sha256(
                passage.encode("utf-8")
            ) == span["sha256"]

            evidence_spans.append(
                (document_id, passage)
            )

        assert evidence_spans

        allowed_ids = set(
            selector.allowed_document_ids(scope)
        )

        for strategy in ("B1", "B2"):

            start = time.perf_counter()

            if strategy == "B1":

                result = bm25.retrieve(
                    query=query,
                    top_k=10,
                    where=(
                        scope.organization_filter()
                    ),
                )

            else:

                result = scoped.retrieve(
                    query=query,
                    scope=scope,
                    top_k=10,
                )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            candidates = result.retrieved_chunks

            document_ranks = []

            evidence_ranks = []

            retrieved_ids = []

            for rank, item in enumerate(
                candidates, start=1
            ):

                chunk = item.chunk

                retrieved_ids.append(
                    chunk.document_id
                )

                if chunk.document_id in expected_ids:

                    document_ranks.append(rank)

                normalized_chunk = normalize(
                    chunk.text
                )

                if any(
                    chunk.document_id == doc_id
                    and passage in normalized_chunk
                    for doc_id, passage
                    in evidence_spans
                ):

                    evidence_ranks.append(rank)

            document_first = (
                min(document_ranks)
                if document_ranks else None
            )

            evidence_first = (
                min(evidence_ranks)
                if evidence_ranks else None
            )

            leakage = sum(
                item.chunk.organization
                != scope.organization
                for item in candidates
            )

            violations = sum(
                item.chunk.document_id
                not in allowed_ids
                for item in candidates
            )

            row = {
                "query_id": query_id,
                "strategy": strategy,
                "scope": scope.to_dict(),
                "document_first_rank": document_first,
                "evidence_first_rank": evidence_first,
                "reciprocal_rank": (
                    1.0 / evidence_first
                    if evidence_first else 0.0
                ),
                "elapsed_ms": elapsed_ms,
                "returned_chunks": len(candidates),
                "organization_leakage": leakage,
                "allowlist_violations": violations,
                "gold_document_available": (
                    expected_ids.issubset(
                        allowed_ids
                    )
                    if strategy == "B2"
                    else all(
                        documents[doc_id]["organization"]
                        == scope.organization
                        for doc_id in expected_ids
                    )
                ),
                "document_recall": {
                    str(k): len(
                        expected_ids.intersection(
                            retrieved_ids[:k]
                        )
                    ) / len(expected_ids)
                    for k in K_VALUES
                },
                "evidence_hit": {
                    str(k): int(
                        evidence_first is not None
                        and evidence_first <= k
                    )
                    for k in K_VALUES
                },
                "retrieved_chunk_ids": [
                    item.chunk.chunk_id
                    for item in candidates
                ],
            }

            results.append(row)

            print(
                query_id,
                strategy,
                "document_rank:", document_first,
                "evidence_rank:", evidence_first,
                "latency_ms:", round(elapsed_ms, 2),
            )

    summaries = {}

    for strategy in ("B1", "B2"):

        rows = [
            row for row in results
            if row["strategy"] == strategy
        ]

        latencies = [
            row["elapsed_ms"] for row in rows
        ]

        summary = {
            "queries": len(rows),
            "document_recall": {
                str(k): statistics.mean(
                    row["document_recall"][str(k)]
                    for row in rows
                )
                for k in K_VALUES
            },
            "evidence_hit": {
                str(k): statistics.mean(
                    row["evidence_hit"][str(k)]
                    for row in rows
                )
                for k in K_VALUES
            },
            "mrr_at_10": statistics.mean(
                row["reciprocal_rank"]
                for row in rows
            ),
            "mean_latency_ms": statistics.mean(
                latencies
            ),
            "p95_latency_ms": percentile95(
                latencies
            ),
            "organization_leakage": sum(
                row["organization_leakage"]
                for row in rows
            ),
            "allowlist_violations": sum(
                row["allowlist_violations"]
                for row in rows
            ),
            "gold_document_unavailable": [
                row["query_id"]
                for row in rows
                if not row["gold_document_available"]
            ],
            "evidence_misses_at_5": [
                row["query_id"]
                for row in rows
                if not row["evidence_hit"]["5"]
            ],
        }

        summaries[strategy] = summary

    assert len(results) == 80

    assert summaries["B1"][
        "organization_leakage"
    ] == 0

    assert summaries["B2"][
        "organization_leakage"
    ] == 0

    assert summaries["B2"][
        "allowlist_violations"
    ] == 0

    report = {
        "dataset_sha256": sha256(
            dataset_path.read_bytes()
        ),
        "dataset_status": (
            "draft_integrity_validated"
        ),
        "corpus": corpus_summary,
        "total_executions": len(results),
        "strategies": {
            "B1": "organization_filtered_bm25",
            "B2": "organization_country_filtered_bm25",
        },
        "summaries": summaries,
        "results": results,
        "hosted_ai_api_calls": 0,
        "limitations": [
            "Draft gold evidence requires semantic review.",
            "Evidence hit measures any complete annotated span.",
            "A retrieved passage does not prove answer correctness.",
            "Country classifications are provisional.",
            "No definitive eligibility decisions are evaluated.",
        ],
    }

    OUTPUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== EVALUATION SUMMARY ==========")

    print(
        json.dumps(summaries, indent=2)
    )

    print("\nExecutions:", len(results))
    print("Hosted AI API calls: 0")
    print("Report:", OUTPUT)
    print("SUCCESS: Scoped BM25 evaluation completed.")


if __name__ == "__main__":
    main()
