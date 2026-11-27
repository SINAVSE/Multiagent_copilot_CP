
import pytest

from src.rag.ingestion.ingestion_guard import (
    ingestion_lock,
    validate_or_create_provenance,
)


@pytest.fixture
def provenance():

    return {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimensions": 384,
        "embedding_backend": "synthetic-test",
        "manifest_sha256": "a" * 64,
        "chunk_sha256": "b" * 64,
        "collection_name": "test_collection",
    }


def test_lock_created_and_removed(tmp_path):

    with ingestion_lock(tmp_path):

        assert (
            tmp_path / "ingestion.lock"
        ).exists()

    assert not (
        tmp_path / "ingestion.lock"
    ).exists()


def test_concurrent_ingestion_blocked(tmp_path):

    with ingestion_lock(tmp_path):

        with pytest.raises(RuntimeError):

            with ingestion_lock(tmp_path):
                pass


def test_lock_released_after_failure(tmp_path):

    with pytest.raises(ValueError):

        with ingestion_lock(tmp_path):

            raise ValueError("Simulated failure")

    assert not (
        tmp_path / "ingestion.lock"
    ).exists()


def test_create_provenance(tmp_path, provenance):

    result = validate_or_create_provenance(
        tmp_path, provenance, 0
    )

    assert result == "created"


def test_existing_provenance_verified(
    tmp_path, provenance
):

    validate_or_create_provenance(
        tmp_path, provenance, 0
    )

    result = validate_or_create_provenance(
        tmp_path, provenance, 10
    )

    assert result == "verified"


@pytest.mark.parametrize(
    "field, value",
    [
        ("embedding_model", "different-model"),
        ("embedding_backend", "different-backend"),
        ("manifest_sha256", "c" * 64),
        ("chunk_sha256", "d" * 64),
        ("collection_name", "different_collection"),
    ],
)
def test_provenance_mismatch(
    tmp_path, provenance, field, value
):

    validate_or_create_provenance(
        tmp_path, provenance, 0
    )

    modified = dict(provenance)

    modified[field] = value

    with pytest.raises(ValueError):

        validate_or_create_provenance(
            tmp_path, modified, 10
        )


def test_existing_vectors_without_provenance(
    tmp_path, provenance
):

    with pytest.raises(ValueError):

        validate_or_create_provenance(
            tmp_path, provenance, 10
        )


def test_incomplete_provenance(
    tmp_path, provenance
):

    del provenance["embedding_model"]

    with pytest.raises(ValueError):

        validate_or_create_provenance(
            tmp_path, provenance, 0
        )


def test_invalid_embedding_dimensions(
    tmp_path, provenance
):

    provenance["embedding_dimensions"] = 768

    with pytest.raises(ValueError):

        validate_or_create_provenance(
            tmp_path, provenance, 0
        )
