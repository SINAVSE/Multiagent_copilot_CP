from config.clients import llm


def test_llm_connection():
    response = llm.invoke(
        "Reply with exactly: LLM connection successful"
    )

    print("\nLLM Response:", response.content)

    assert response.content