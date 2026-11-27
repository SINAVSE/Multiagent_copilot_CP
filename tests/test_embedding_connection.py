# CAP-11: Explicit opt-in for hosted integration tests.
import os
import pytest

if os.getenv("RUN_LIVE_TESTS") != "1":
    pytest.skip(
        "Set RUN_LIVE_TESTS=1 to enable hosted integration tests.",
        allow_module_level=True,
    )

from config.clients import embeddings


def test_embedding_connection():
    response = embeddings.embed_query(
        "Employee leave policy and annual leave entitlement"
    )

    print("\nEmbedding dimensions:", len(response))

    assert len(response) > 0
