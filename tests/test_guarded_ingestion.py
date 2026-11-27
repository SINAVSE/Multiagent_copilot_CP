
from unittest.mock import Mock

import pytest

from test_vector_ingestion_v2 import (
    chunks,
    mock_embeddings,
)

from src.rag.ingestion.vector_ingestion_v2 import (
    ingest_guarded_test,
)

from src.rag.ingestion.ingestion_guard import (
    ingestion_lock,
)


@pytest.fixture
def provenance():

    return {
        "embedding_model": "synthetic-test-model",
        "embedding_dimensions": 384,
        "embedding_backend": "synthetic-test",
        "manifest_sha256": "a" * 64,
        "chunk_sha256": "b" * 64,
        "collection_name": "cap14_guarded_test",
    }


def test_guarded_initial_ingestion(
    tmp_path, chunks, provenance
):

    result = ingest_guarded_test(
        tmp_path,
        chunks,
        mock_embeddings,
        provenance,
        batch_size=2,
    )

    assert result["verified_chunks"] == 5
    assert result["embedding_batches"] == 3
    assert result["provenance_status"] == "created"

    assert (tmp_path / "provenance.json").exists()
    assert not (tmp_path / "ingestion.lock").exists()


def test_guarded_resume(
    tmp_path, chunks, provenance
):

    ingest_guarded_test(
        tmp_path,
        chunks,
        mock_embeddings,
        provenance,
    )

    embedder = Mock(
        side_effect=mock_embeddings
    )

    result = ingest_guarded_test(
        tmp_path,
        chunks,
        embedder,
        provenance,
    )

    assert result["previously_stored_chunks"] == 5
    assert result["newly_embedded_chunks"] == 0
    assert result["provenance_status"] == "verified"

    embedder.assert_not_called()


def test_guarded_concurrent_execution(
    tmp_path, chunks, provenance
):

    embedder = Mock()

    with ingestion_lock(tmp_path):

        with pytest.raises(RuntimeError):

            ingest_guarded_test(
                tmp_path,
                chunks,
                embedder,
                provenance,
            )

    embedder.assert_not_called()


def test_guarded_model_mismatch(
    tmp_path, chunks, provenance
):

    ingest_guarded_test(
        tmp_path,
        chunks,
        mock_embeddings,
        provenance,
    )

    modified = dict(provenance)

    modified["embedding_model"] = "different-model"

    embedder = Mock()

    with pytest.raises(
        ValueError,
        match="Embedding provenance mismatch",
    ):

        ingest_guarded_test(
            tmp_path,
            chunks,
            embedder,
            modified,
        )

    embedder.assert_not_called()

    assert not (tmp_path / "ingestion.lock").exists()


def test_guarded_failure_releases_lock(
    tmp_path, chunks, provenance
):

    embedder = Mock(
        side_effect=RuntimeError("Simulated failure")
    )

    with pytest.raises(RuntimeError):

        ingest_guarded_test(
            tmp_path,
            chunks,
            embedder,
            provenance,
        )

    assert not (tmp_path / "ingestion.lock").exists()

    result = ingest_guarded_test(
        tmp_path,
        chunks,
        mock_embeddings,
        provenance,
    )

    assert result["verified_chunks"] == 5


def test_hosted_backend_rejected(
    tmp_path, chunks, provenance
):

    provenance["embedding_backend"] = "hosted"

    embedder = Mock()

    with pytest.raises(ValueError):

        ingest_guarded_test(
            tmp_path,
            chunks,
            embedder,
            provenance,
        )

    embedder.assert_not_called()


def test_invalid_collection_rejected(
    tmp_path, chunks, provenance
):

    provenance["collection_name"] = (
        "employee_operations_copilot"
    )

    embedder = Mock()

    with pytest.raises(ValueError):

        ingest_guarded_test(
            tmp_path,
            chunks,
            embedder,
            provenance,
        )

    embedder.assert_not_called()
