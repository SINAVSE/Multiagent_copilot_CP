# CAP-11: Explicit opt-in for hosted integration tests.
import os
import pytest

if os.getenv("RUN_LIVE_TESTS") != "1":
    pytest.skip(
        "Set RUN_LIVE_TESTS=1 to enable hosted integration tests.",
        allow_module_level=True,
    )

from config.clients import llm


def test_llm_connection():
    response = llm.invoke(
        "Reply with exactly: LLM connection successful"
    )

    print("\nLLM Response:", response.content)

    assert response.content
