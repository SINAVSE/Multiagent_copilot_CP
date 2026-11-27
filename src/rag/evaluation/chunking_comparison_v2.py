
import hashlib
import json
import time

from pathlib import Path
from statistics import mean
from collections import Counter

from src.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag.evaluation.bm25_comparison import percentile


ROOT = Path(__file__).resolve().parents[3]

CORPUS = ROOT / "data/real_corpus/expanded_v2"

MANIFEST = CORPUS / "provisional_manifest_v2.jsonl"

DATASET = CORPUS / "retrieval_eval_v2_draft.json"

CHUNK_DIR = CORPUS / "chunking_experiments"

OUTPUT = CORPUS / "chunking_retrieval_comparison_v2.json"

EXPECTED_DATASET_HASH = (
    "2678ba31570f0fa854d06da10b850a6540"
    "fff700d61eea2547115a21f1b809fc"
)

STRATEGIES = ("C1", "C2", "C3", "C4", "C5")

LEVELS = (1, 3, 5, 10)


def file_hash(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def normalize(text):

    return " ".join(text.split())


def load_gold():

    assert file_hash(DATASET) == EXPECTED_DATASET_HASH, (
        "Evaluation dataset changed."
    )

    dataset = json.loads(
        DATASET.read_text(encoding="utf-8-sig")
    )

    assert file_hash(MANIFEST) == (
        dataset["manifest_sha256"]
    )

    rows = {
        row["document_id"]: row
        for row in (
            json.loads(line)
            for line in MANIFEST.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        )
    }

    assert len(rows) == 659

    queries = dataset["queries"]

    assert len(queries) == 40

    assert len({
        q["query_id"] for q in queries
    }) == 40

    gold = []

    for query in queries:

        evidence_items = []

        for evidence in query["evidence_spans"]:

            row = rows[evidence["document_id"]]

            assert row["organization"] == (
                query["organization"]
            )

            path = ROOT / row["processed_path"]

            assert file_hash(path) == (
                row["processed_file_hash"]
            )

            text = normalize(
                path.read_text(encoding="utf-8-sig")
            )

            start = evidence["start"]
            end = evidence["end"]

            assert 0 <= start < end <= len(text)

            passage = text[start:end]

            assert hashlib.sha256(
                passage.encode("utf-8")
            ).hexdigest() == evidence["sha256"]

            evidence_items.append({
                "document_id": evidence["document_id"],
                "passage": passage
            })

        assert evidence_items

        gold.append({
            **query,
            "verified_evidence": evidence_items
        })

    return gold


def expected_chunk_hashes():

    report = json.loads(
        (CHUNK_DIR / "chunking_experiment_summary.json")
        .read_text(encoding="utf-8-sig")
    )

    c5 = json.loads(
        (CHUNK_DIR / "c5_experiment_summary.json")
        .read_text(encoding="utf-8-sig")
    )

    assert report["manifest_sha256"] == file_hash(MANIFEST)

    assert c5["manifest_sha256"] == file_hash(MANIFEST)

    hashes = {
        result["strategy"]: result["chunk_file_sha256"]
        for result in report["results"]
    }

    hashes["C5"] = c5["chunk_file_sha256"]

    return hashes


def evidence_matches(chunk, evidence):

    return (
        chunk.document_id == evidence["document_id"]
        and evidence["passage"] in normalize(chunk.text)
    )


def calculate_availability(retriever, queries):

    by_document = {}

    for chunk in retriever.chunks:

        by_document.setdefault(
            chunk.document_id, []
        ).append(chunk)

    total_spans = 0
    available_spans = 0
    unavailable_queries = []

    for query in queries:

        missing = 0

        for evidence in query["verified_evidence"]:

            total_spans += 1

            candidates = by_document.get(
                evidence["document_id"], []
            )

            available = any(
                evidence_matches(chunk, evidence)
                for chunk in candidates
            )

            available_spans += int(available)

            if not available:
                missing += 1

        if missing:

            unavailable_queries.append({
                "query_id": query["query_id"],
                "missing_complete_spans": missing
            })

    return {
        "total_evidence_spans": total_spans,
        "available_complete_spans": available_spans,
        "availability_rate": (
            available_spans / total_spans
        ),
        "queries_with_unavailable_spans": (
            unavailable_queries
        )
    }


def evaluate(retriever, queries, filtered):

    results = []

    for item in queries:

        where = (
            {"organization": item["organization"]}
            if filtered
            else None
        )

        started = time.perf_counter()

        result = retriever.retrieve(
            query=item["query"],
            top_k=10,
            where=where
        )

        latency = (
            time.perf_counter() - started
        ) * 1000

        retrieved = [
            candidate.chunk
            for candidate in result.retrieved_chunks
        ]

        expected_documents = set(
            item["expected_document_ids"]
        )

        evidence = item["verified_evidence"]

        leakage = sum(
            chunk.organization != item["organization"]
            for chunk in retrieved
        )

        if filtered:

            assert leakage == 0, (
                "Organization leakage: "
                + item["query_id"]
            )

        row = {
            "query_id": item["query_id"],
            "category": item["category"],
            "organization": item["organization"],
            "latency_ms": latency,
            "retrieved_count": len(retrieved),
            "organization_leakage_count": leakage,
            "retrieved_chunk_ids": [
                chunk.chunk_id
                for chunk in retrieved
            ],
            "retrieved_document_ids": [
                chunk.document_id
                for chunk in retrieved
            ],
            "retrieved_organizations": [
                chunk.organization
                for chunk in retrieved
            ],
            "reciprocal_rank": 0.0,
        }

        for rank, chunk in enumerate(
            retrieved, start=1
        ):

            if any(
                evidence_matches(chunk, passage)
                for passage in evidence
            ):

                row["reciprocal_rank"] = 1.0 / rank
                break

        for k in LEVELS:

            candidates = retrieved[:k]

            documents = {
                chunk.document_id
                for chunk in candidates
            }

            row[f"document_recall_at_{k}"] = (
                len(expected_documents & documents)
                / len(expected_documents)
            )

            matched = sum(
                any(
                    evidence_matches(chunk, passage)
                    for chunk in candidates
                )
                for passage in evidence
            )

            row[f"evidence_recall_at_{k}"] = (
                matched / len(evidence)
            )

        results.append(row)

    total_returned = sum(
        row["retrieved_count"]
        for row in results
    )

    total_leakage = sum(
        row["organization_leakage_count"]
        for row in results
    )

    summary = {
        "queries": len(results),
        "filtering": (
            "organization" if filtered else "none"
        ),
        "mrr_at_10": mean(
            row["reciprocal_rank"]
            for row in results
        ),
        "mean_latency_ms": mean(
            row["latency_ms"]
            for row in results
        ),
        "p95_latency_ms": percentile(
            [row["latency_ms"] for row in results],
            0.95
        ),
        "organization_leakage_rate": (
            total_leakage / total_returned
            if total_returned else 0.0
        ),
        "queries_with_organization_leakage": sum(
            row["organization_leakage_count"] > 0
            for row in results
        ),
        "zero_evidence_at_5_queries": sum(
            row["evidence_recall_at_5"] == 0
            for row in results
        ),
        "zero_result_queries": sum(
            row["retrieved_count"] == 0
            for row in results
        )
    }

    for k in LEVELS:

        for metric in ("document", "evidence"):

            key = f"{metric}_recall_at_{k}"

            summary[key] = mean(
                row[key] for row in results
            )

    isolation = [
        row for row in results
        if row["category"] == "organization_isolation"
    ]

    assert len(isolation) == 4

    summary["isolation_query_leakage_count"] = sum(
        row["organization_leakage_count"]
        for row in isolation
    )

    summary["isolation_evidence_recall_at_5"] = mean(
        row["evidence_recall_at_5"]
        for row in isolation
    )

    return {
        "summary": summary,
        "query_results": results
    }


def main():

    queries = load_gold()

    hashes = expected_chunk_hashes()

    configurations = {}

    for strategy in STRATEGIES:

        print(
            "\n========== " + strategy + " ==========",
            flush=True
        )

        chunk_file = (
            CHUNK_DIR
            / (strategy.lower() + "_chunks.jsonl")
        )

        assert file_hash(chunk_file) == hashes[strategy]

        retriever = BM25Retriever(
            chunk_file=str(chunk_file),
            include_metadata=True
        )

        assert len(retriever.chunks) == len({
            chunk.chunk_id
            for chunk in retriever.chunks
        })

        assert len({
            chunk.document_id
            for chunk in retriever.chunks
        }) == 659

        availability = calculate_availability(
            retriever, queries
        )

        unfiltered = evaluate(
            retriever, queries, filtered=False
        )

        filtered = evaluate(
            retriever, queries, filtered=True
        )

        assert file_hash(chunk_file) == hashes[strategy]

        configurations[strategy] = {
            "total_chunks": len(retriever.chunks),
            "chunk_file_sha256": hashes[strategy],
            "evidence_availability": availability,
            "unfiltered": unfiltered,
            "filtered": filtered
        }

        print("Chunks:", len(retriever.chunks))

        print(
            "Complete evidence availability:",
            round(availability["availability_rate"], 4)
        )

        print(
            "Unfiltered:",
            json.dumps(unfiltered["summary"], indent=2)
        )

        print(
            "Filtered:",
            json.dumps(filtered["summary"], indent=2)
        )

        del retriever

    assert file_hash(DATASET) == EXPECTED_DATASET_HASH

    report = {
        "experiment": "CAP-14 chunking comparison",
        "dataset_status": "draft_integrity_validated",
        "dataset_sha256": file_hash(DATASET),
        "manifest_sha256": file_hash(MANIFEST),
        "total_queries": len(queries),
        "retrieval_executions": 400,
        "strategy_count": len(configurations),
        "retrieval_method": "BM25 metadata-aware",
        "evidence_match_method": (
            "complete whitespace-normalized passage "
            "inside an individual source-matched chunk"
        ),
        "configurations": configurations,
        "hosted_ai_api_calls": 0
    }

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8"
    )

    print("\n========== FINAL COMPARATIVE SUMMARY ==========")

    print(
        f"{'Config':<18}"
        f"{'DocR@5':>10}"
        f"{'EvR@1':>10}"
        f"{'EvR@3':>10}"
        f"{'EvR@5':>10}"
        f"{'MRR@10':>10}"
        f"{'Leakage':>10}"
        f"{'P95ms':>10}"
    )

    for strategy, config in configurations.items():

        for mode in ("unfiltered", "filtered"):

            s = config[mode]["summary"]

            name = strategy + "_" + mode

            print(
                f"{name:<18}"
                f"{s['document_recall_at_5']:>10.4f}"
                f"{s['evidence_recall_at_1']:>10.4f}"
                f"{s['evidence_recall_at_3']:>10.4f}"
                f"{s['evidence_recall_at_5']:>10.4f}"
                f"{s['mrr_at_10']:>10.4f}"
                f"{s['organization_leakage_rate']:>10.4f}"
                f"{s['p95_latency_ms']:>10.2f}"
            )

    print("\n========== EVIDENCE AVAILABILITY ==========")

    for strategy, config in configurations.items():

        a = config["evidence_availability"]

        print(
            strategy,
            "| Available:",
            a["available_complete_spans"],
            "/",
            a["total_evidence_spans"],
            "| Rate:",
            round(a["availability_rate"], 4)
        )

    print("\nRetrieval executions: 400")
    print("Hosted AI API calls: 0")
    print("Original corpus modified: NO")
    print("Vector database modified: NO")
    print("Output:", OUTPUT.resolve())
    print("SUCCESS: Chunking retrieval comparison completed.")


if __name__ == "__main__":
    main()
