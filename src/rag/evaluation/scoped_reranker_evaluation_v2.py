import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


from pathlib import Path
from collections import Counter
import hashlib
import json
import statistics
import math
import time

from huggingface_hub import snapshot_download

from src.rag.retrieval.reranker import (
    CrossEncoderReranker,
)

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
    / "scoped_reranker_evaluation_results_v2.json"
)

EXPECTED_DATASET_HASH = (
    "2678ba31570f0fa854d06da10b850a654"
    "0fff700d61eea2547115a21f1b809fc"
)

K_VALUES = (1, 3, 5, 10)

RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_MAX_LENGTH = 512


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

    print("Loading local reranker...")

    model_path = snapshot_download(
        repo_id=RERANKER_MODEL,
        local_files_only=True,
    )

    model_revision = Path(model_path).name

    load_start = time.perf_counter()

    reranker = CrossEncoderReranker(
        model_name=model_path,
        device="cpu",
        max_length=RERANKER_MAX_LENGTH,
        include_metadata=True,
    )

    model_load_ms = (
        time.perf_counter() - load_start
    ) * 1000

    print("Reranker loaded.")
    print("Model revision:", model_revision)

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

        for strategy in ("B2", "B3"):

            start = time.perf_counter()

            if strategy == "B2":

                baseline_result = scoped.retrieve(
                    query=query,
                    scope=scope,
                    top_k=10,
                )

                baseline_ms = (
                    time.perf_counter() - start
                ) * 1000

                result = baseline_result
                elapsed_ms = baseline_ms
                rerank_ms = 0.0

            else:

                reranked = reranker.rerank(
                    query=query,
                    retrieval_result=baseline_result,
                    top_k=10,
                )

                result, diagnostics = selector.select(
                    scope, reranked
                )

                rerank_ms = (
                    time.perf_counter() - start
                ) * 1000

                elapsed_ms = baseline_ms + rerank_ms

                original_ids = [
                    item.chunk.chunk_id
                    for item in baseline_result.retrieved_chunks
                ]

                reranked_ids = [
                    item.chunk.chunk_id
                    for item in result.retrieved_chunks
                ]

                assert len(original_ids) == len(reranked_ids)

                assert set(original_ids) == set(reranked_ids)

                assert len(set(reranked_ids)) == len(reranked_ids)

                assert diagnostics["rejected_candidates"] == 0

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
                "bm25_latency_ms": baseline_ms,
                "reranking_latency_ms": rerank_ms,
                "returned_chunks": len(candidates),
                "organization_leakage": leakage,
                "allowlist_violations": violations,
                "gold_document_available": (
                    expected_ids.issubset(
                        allowed_ids
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

    for strategy in ("B2", "B3"):

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
            "mean_reranking_latency_ms": statistics.mean(
                row["reranking_latency_ms"]
                for row in rows
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

    assert summaries["B3"][
        "organization_leakage"
    ] == 0

    assert summaries["B2"][
        "organization_leakage"
    ] == 0

    for strategy in ("B2", "B3"):

        assert summaries[strategy][
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
            "B2": "organization_country_filtered_bm25",
            "B3": "scoped_bm25_with_cross_encoder",
        },
        "reranker": {
            "model": RERANKER_MODEL,
            "model_revision": model_revision,
            "device": "cpu",
            "max_length": RERANKER_MAX_LENGTH,
            "include_metadata": True,
            "candidate_k": 10,
            "model_load_ms": model_load_ms,
            "local_files_only": True,
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
            "Reranking uses the same ten candidates as B2.",
            "Query and metadata can cause passage truncation.",
            "Model initialization is excluded from query latency.",
            "Reranking cannot recover evidence absent from candidates.",
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
    print("SUCCESS: Scoped reranker evaluation completed.")


if __name__ == "__main__":
    main()
