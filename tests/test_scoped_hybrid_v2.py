from pathlib import Path

from unittest.mock import Mock

import pytest

from test_dense_retrieval_v2 import vector

from src.agents.policy_scope import PolicyScope

from src.rag.schemas import DocumentChunk

from src.rag.indexing.vector_store import ChromaVectorStore

from src.rag.retrieval.applicability_filter import (
    ApplicabilityFilter,
)

from src.rag.retrieval.dense_retriever import (
    DenseRetriever,
)

from src.rag.retrieval.bm25_retriever import (
    BM25Retriever,
)

from src.rag.retrieval.scoped_dense_retriever import (
    ScopedDenseRetriever,
)

from src.rag.retrieval.scoped_bm25_retriever import (
    ScopedBM25Retriever,
)

from src.rag.retrieval.scoped_hybrid_retriever import (
    ScopedHybridRetriever,
)


def test_scoped_dense_and_hybrid_integration(tmp_path):

    # Actual expanded-corpus registry.
    # Only synthetic vectors are used in the temporary index.


    from src.agents.applicability_registry import (
        ApplicabilityRegistry,
    )

    import json


    root = tmp_path

    document = {
        "document_id": "SYNTHETIC-INDIA-001",
        "organization": "GitLab",
        "title": "Synthetic India Employee Benefits",
        "source_url": "https://example.org/india",
        "processed_file_hash": "a" * 64,
    }

    manifest = root / "manifest.jsonl"

    manifest.write_text(
        json.dumps(document) + "\n",
        encoding="utf-8",
    )

    import hashlib

    record = {
        **document,
        "source_country": None,
        "candidate_applicability": "COUNTRY_SPECIFIC",
        "candidate_countries": ["IN"],
        "verified_applicability": "UNKNOWN",
        "verified_countries": [],
        "review_status": "pending_content_review",
        "eligible_for_definitive_decision": False,
    }

    registry_path = root / "registry.json"

    registry_path.write_text(
        json.dumps({
            "version": "2.0-draft",
            "manifest_sha256": hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            "total_documents": 1,
            "documents": [record],
        }),
        encoding="utf-8",
    )

    registry = ApplicabilityRegistry(
        manifest, registry_path
    )

    selector = ApplicabilityFilter(registry)

    chunks = [
        DocumentChunk(
            chunk_id="SYNTHETIC-CHUNK-001",
            document_id=document["document_id"],
            organization=document["organization"],
            title=document["title"],
            source_url=document["source_url"],
            text="Employees receive annual leave benefits.",
            chunk_index=0,
        )
    ]

    chunk_file = root / "chunks.jsonl"

    chunk_file.write_text(
        chunks[0].model_dump_json() + "\n",
        encoding="utf-8",
    )

    store = ChromaVectorStore(
        persist_directory=root / "chroma",
        collection_name="cap14_scoped_test",
    )

    store.add_chunks(chunks, [vector(0)])

    embedder = Mock(return_value=vector(0))

    dense = ScopedDenseRetriever(
        DenseRetriever(store, embedder),
        selector,
        chunks,
    )

    bm25 = ScopedBM25Retriever(
        BM25Retriever(chunk_file=chunk_file),
        selector,
    )

    hybrid = ScopedHybridRetriever(
        dense, bm25
    )

    india = PolicyScope("GitLab", "IN")
    germany = PolicyScope("GitLab", "DE")
    unspecified = PolicyScope("GitLab")
    rice = PolicyScope("Rice University", "US")

    result = hybrid.retrieve(
        "annual leave benefits",
        india,
        top_k=1,
    )

    assert len(result.retrieved_chunks) == 1

    assert (
        result.retrieved_chunks[0].chunk.chunk_id
        == chunks[0].chunk_id
    )

    assert result.metadata["dense_candidates"] == 1
    assert result.metadata["bm25_candidates"] == 1

    assert (
        result.metadata["definitive_decision_allowed"]
        is False
    )

    embedder.assert_called_once()

    for scope in (
        germany,
        unspecified,
        rice,
    ):

        embedder.reset_mock()

        result = hybrid.retrieve(
            "annual leave benefits",
            scope,
            top_k=1,
        )

        assert result.retrieved_chunks == []

        embedder.assert_not_called()
