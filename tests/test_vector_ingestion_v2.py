
from unittest.mock import Mock

import pytest

from src.rag.schemas import DocumentChunk

from src.rag.indexing.vector_store import (
    ChromaVectorStore,
)

from src.rag.ingestion.vector_ingestion_v2 import (
    ingest_resumable,
    verify_existing,
    validate_embeddings,
)


def make_vector():

    return [0.1] * 384


def mock_embeddings(texts):

    return [
        make_vector()
        for text in texts
    ]


@pytest.fixture
def chunks():

    return [
        DocumentChunk(
            chunk_id=f"TEST-CHUNK-{index}",
            document_id="TEST-DOC-001",
            title="Synthetic Employee Policy",
            organization="GitLab",
            source_url="https://example.org/policy",
            text=f"Synthetic policy content {index}",
            chunk_index=index,
        )
        for index in range(5)
    ]


@pytest.fixture
def store(tmp_path):

    return ChromaVectorStore(
        persist_directory=tmp_path / "chroma_test",
        collection_name="cap14_test_collection",
    )


def test_initial_ingestion(store, chunks):

    embedder = Mock(
        side_effect=mock_embeddings
    )

    result = ingest_resumable(
        store, chunks, embedder, batch_size=2
    )

    assert result["status"] == "completed"
    assert result["verified_chunks"] == 5
    assert result["previously_stored_chunks"] == 0
    assert result["newly_embedded_chunks"] == 5
    assert result["embedding_batches"] == 3

    assert store.count() == 5
    assert embedder.call_count == 3


def test_repeat_ingestion_avoids_embeddings(
    store, chunks
):

    ingest_resumable(
        store, chunks, mock_embeddings
    )

    embedder = Mock(
        side_effect=mock_embeddings
    )

    result = ingest_resumable(
        store, chunks, embedder
    )

    assert result["previously_stored_chunks"] == 5
    assert result["newly_embedded_chunks"] == 0
    assert result["embedding_batches"] == 0

    embedder.assert_not_called()

    assert store.count() == 5


def test_interrupted_ingestion_and_resume(
    store, chunks
):

    failing_embedder = Mock(
        side_effect=[
            [make_vector(), make_vector()],
            RuntimeError("Simulated API interruption"),
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated API interruption",
    ):

        ingest_resumable(
            store,
            chunks,
            failing_embedder,
            batch_size=2,
        )

    assert store.count() == 2

    resumed_embedder = Mock(
        side_effect=mock_embeddings
    )

    result = ingest_resumable(
        store,
        chunks,
        resumed_embedder,
        batch_size=2,
    )

    assert result["previously_stored_chunks"] == 2
    assert result["newly_embedded_chunks"] == 3
    assert result["embedding_batches"] == 2

    assert store.count() == 5

    embedded_texts = [
        text
        for call in resumed_embedder.call_args_list
        for text in call.args[0]
    ]

    assert embedded_texts == [
        chunk.text for chunk in chunks[2:]
    ]


def test_database_reopen(store, chunks):

    ingest_resumable(
        store, chunks, mock_embeddings
    )

    reopened = ChromaVectorStore(
        persist_directory=store.persist_directory,
        collection_name="cap14_test_collection",
    )

    embedder = Mock(
        side_effect=mock_embeddings
    )

    result = ingest_resumable(
        reopened, chunks, embedder
    )

    assert result["verified_chunks"] == 5
    assert result["newly_embedded_chunks"] == 0

    embedder.assert_not_called()


def test_collection_isolation(store, chunks):

    ingest_resumable(
        store, chunks, mock_embeddings
    )

    another_store = ChromaVectorStore(
        persist_directory=store.persist_directory,
        collection_name="cap14_other_collection",
    )

    assert store.count() == 5
    assert another_store.count() == 0


@pytest.mark.parametrize(
    "vectors, expected_count",
    [
        ([], 1),
        ([[0.1] * 383], 1),
        ([[0.1] * 385], 1),
        ([[float("nan")] * 384], 1),
        ([[float("inf")] * 384], 1),
        ([[float("-inf")] * 384], 1),
        ([[0.0] * 384], 1),
        ([[True] * 384], 1),
        ([["0.1"] * 384], 1),
        ([[0.1] * 384] * 2, 1),
    ],
)
def test_invalid_embeddings(vectors, expected_count):

    with pytest.raises(ValueError):

        validate_embeddings(
            vectors, expected_count
        )


def test_valid_embeddings():

    vectors = validate_embeddings(
        [make_vector()], 1
    )

    assert len(vectors) == 1
    assert len(vectors[0]) == 384


def test_invalid_embedding_batch_not_stored(
    store, chunks
):

    embedder = Mock(return_value=[])

    with pytest.raises(ValueError):

        ingest_resumable(
            store, chunks, embedder
        )

    assert store.count() == 0


def test_stored_content_mismatch(store, chunks):

    ingest_resumable(
        store, chunks, mock_embeddings
    )

    store.collection.update(
        ids=[chunks[0].chunk_id],
        documents=["Modified policy content"],
        embeddings=[make_vector()],
    )

    embedder = Mock(
        side_effect=mock_embeddings
    )

    with pytest.raises(
        ValueError,
        match="Stored chunk content mismatch",
    ):

        ingest_resumable(
            store, chunks, embedder
        )

    embedder.assert_not_called()


def test_stored_metadata_mismatch(store, chunks):

    ingest_resumable(
        store, chunks, mock_embeddings
    )

    store.collection.update(
        ids=[chunks[0].chunk_id],
        metadatas=[
            {"organization": "Rice University"}
        ],
    )

    with pytest.raises(
        ValueError,
        match="Stored chunk metadata mismatch",
    ):

        verify_existing(store, chunks)


def test_unexpected_stored_record(store, chunks):

    extra = chunks[0].model_copy(
        update={"chunk_id": "UNEXPECTED-CHUNK"}
    )

    store.add_chunks(
        [extra], [make_vector()]
    )

    with pytest.raises(
        ValueError,
        match="Unexpected stored chunk",
    ):

        verify_existing(store, chunks)


def test_duplicate_chunk_ids(store, chunks):

    embedder = Mock()

    with pytest.raises(
        ValueError,
        match="Duplicate chunk identifiers",
    ):

        ingest_resumable(
            store,
            chunks + [chunks[0]],
            embedder,
        )

    embedder.assert_not_called()


@pytest.mark.parametrize(
    "batch_size",
    [0, -1, True, 1.5],
)
def test_invalid_batch_size(
    store, chunks, batch_size
):

    embedder = Mock()

    with pytest.raises(ValueError):

        ingest_resumable(
            store,
            chunks,
            embedder,
            batch_size=batch_size,
        )

    embedder.assert_not_called()


def test_empty_chunk_inventory(store):

    embedder = Mock()

    with pytest.raises(ValueError):

        ingest_resumable(
            store, [], embedder
        )

    embedder.assert_not_called()
