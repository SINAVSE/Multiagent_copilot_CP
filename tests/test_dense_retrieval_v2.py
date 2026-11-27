
from unittest.mock import Mock

import pytest

from src.rag.schemas import DocumentChunk
from src.rag.indexing.vector_store import ChromaVectorStore
from src.rag.retrieval.dense_retriever import DenseRetriever


def vector(index):

    result = [0.0] * 384
    result[index] = 1.0
    return result


@pytest.fixture
def store(tmp_path):

    return ChromaVectorStore(
        persist_directory=tmp_path / "dense_test",
        collection_name="cap14_dense_test",
    )


@pytest.fixture
def populated_store(store):

    records = [
        ("CHUNK-1", "DOC-1", "GitLab"),
        ("CHUNK-2", "DOC-2", "GitLab"),
        ("CHUNK-3", "DOC-3", "Rice University"),
    ]

    chunks = [
        DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            organization=organization,
            title="Synthetic policy " + document_id,
            source_url="https://example.org/" + document_id,
            text="Synthetic content " + chunk_id,
            chunk_index=0,
        )
        for chunk_id, document_id, organization in records
    ]

    store.add_chunks(
        chunks,
        [vector(0), vector(1), vector(0)],
    )

    return store


def test_injected_embedding(populated_store):

    embedder = Mock(return_value=vector(0))

    retriever = DenseRetriever(
        vector_store=populated_store,
        embedding_function=embedder,
    )

    result = retriever.retrieve(
        "  policy question  ",
        top_k=3,
    )

    assert len(result.retrieved_chunks) == 3

    embedder.assert_called_once_with(
        "policy question"
    )


def test_organization_filter(populated_store):

    retriever = DenseRetriever(
        populated_store,
        lambda query: vector(0),
    )

    result = retriever.retrieve(
        "Policy question",
        top_k=2,
        where={"organization": "GitLab"},
    )

    assert len(result.retrieved_chunks) == 2

    assert all(
        item.chunk.organization == "GitLab"
        for item in result.retrieved_chunks
    )

    assert (
        result.retrieved_chunks[0].chunk.chunk_id
        == "CHUNK-1"
    )


def test_document_allowlist(populated_store):

    retriever = DenseRetriever(
        populated_store,
        lambda query: vector(0),
    )

    result = retriever.retrieve(
        "Policy question",
        top_k=2,
        where={
            "$and": [
                {"organization": "GitLab"},
                {
                    "document_id": {
                        "$in": ["DOC-2"]
                    }
                },
            ]
        },
    )

    assert len(result.retrieved_chunks) == 1

    assert (
        result.retrieved_chunks[0].chunk.document_id
        == "DOC-2"
    )


def test_empty_store_skips_embedding(store):

    embedder = Mock()

    retriever = DenseRetriever(
        store, embedder
    )

    result = retriever.retrieve("Policy question")

    assert result.retrieved_chunks == []

    embedder.assert_not_called()


@pytest.mark.parametrize(
    "embedding",
    [
        [0.1] * 383,
        [0.1] * 385,
        [0.0] * 384,
        [float("nan")] * 384,
        [float("inf")] * 384,
        [True] * 384,
        ["0.1"] * 384,
    ],
)
def test_invalid_embedding(
    populated_store, embedding
):

    retriever = DenseRetriever(
        populated_store,
        lambda query: embedding,
    )

    with pytest.raises(ValueError):

        retriever.retrieve("Policy question")


@pytest.mark.parametrize(
    "top_k", [0, -1, True, 1.5]
)
def test_invalid_top_k(populated_store, top_k):

    embedder = Mock()

    retriever = DenseRetriever(
        populated_store, embedder
    )

    with pytest.raises(ValueError):

        retriever.retrieve(
            "Policy question",
            top_k=top_k,
        )

    embedder.assert_not_called()


def test_invalid_embedding_function(store):

    with pytest.raises(TypeError):

        DenseRetriever(store, "invalid")
