
"""
CAP-06: Extract gold policy evidence for
real-corpus end-to-end evaluation.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRIEVAL_DATASET = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "real_corpus"
    / "retrieval_eval_v1.0.json"
)

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "real_corpus"
    / "chunks"
    / "gitlab_structure_aware_v4_500_overlap_100.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "real_corpus"
    / "gold_policy_evidence_v1.0.json"
)


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8-sig"
    ) as file:
        return json.load(file)


def load_chunks(path):
    chunks = {}

    with path.open(
        "r",
        encoding="utf-8-sig"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            chunk = json.loads(line)

            metadata = chunk.get("metadata") or {}

            chunk_id = (
                chunk.get("chunk_id")
                or metadata.get("chunk_id")
                or chunk.get("id")
            )

            if chunk_id:
                if chunk_id in chunks:
                    raise ValueError(
                        f"Duplicate chunk ID: {chunk_id}"
                    )

                chunks[chunk_id] = chunk

    return chunks


def extract_gold_evidence():

    dataset = load_json(RETRIEVAL_DATASET)

    chunks = load_chunks(CHUNKS_FILE)

    results = []
    missing_chunks = []
    complete_queries = 0

    print(
        f"Available policy chunks: {len(chunks)}"
    )

    for query in dataset["queries"]:

        evidence = []

        for chunk_id in query["expected_chunk_ids"]:

            chunk = chunks.get(chunk_id)

            if chunk is None:

                missing_chunks.append({
                    "query_id": query["query_id"],
                    "chunk_id": chunk_id,
                })

                continue

            metadata = chunk.get("metadata") or {}

            evidence.append({
                "chunk_id": chunk_id,
                "document_id": (
                    chunk.get("document_id")
                    or metadata.get("document_id")
                ),
                "title": (
                    chunk.get("title")
                    or metadata.get("title")
                ),
                "section_path": (
                    chunk.get("section_path")
                    or metadata.get("section_path")
                    or chunk.get("section")
                ),
                "text": (
                    chunk.get("text")
                    or chunk.get("content")
                    or chunk.get("page_content")
                    or ""
                ),
            })

        evidence_complete = (
            len(evidence)
            == len(query["expected_chunk_ids"])
            and all(item["text"] for item in evidence)
        )

        if evidence_complete:
            complete_queries += 1

        results.append({
            "test_id": query["query_id"],
            "category": query["category"],
            "query": query["query"],
            "expected_document_ids": (
                query["expected_document_ids"]
            ),
            "expected_chunk_ids": (
                query["expected_chunk_ids"]
            ),
            "relevance_notes": (
                query.get("relevance_notes")
            ),
            "evidence_complete": evidence_complete,
            "gold_evidence": evidence,
        })

    output = {
        "dataset_version": "1.0",
        "source_corpus": "GitLab Handbook",
        "source_chunk_file": CHUNKS_FILE.name,
        "total_queries": len(results),
        "complete_evidence_queries": complete_queries,
        "missing_chunks": missing_chunks,
        "evaluation_cases": results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Total queries: {len(results)}")

    print(
        f"Queries with complete evidence: "
        f"{complete_queries}"
    )

    print(
        f"Missing chunk references: "
        f"{len(missing_chunks)}"
    )

    for item in missing_chunks:
        print(
            f"Missing: {item['query_id']} "
            f"- {item['chunk_id']}"
        )

    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_gold_evidence()
