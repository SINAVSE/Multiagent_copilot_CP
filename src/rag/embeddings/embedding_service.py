from langchain_openai import OpenAIEmbeddings
from config.clients import embeddings

from src.agents.budgeted_model import invoke_embedding


EXPECTED_EMBEDDING_DIMENSION = 384


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding for a single text.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    vector = invoke_embedding(embeddings.embed_query, text)

    if len(vector) != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Unexpected embedding dimension: {len(vector)}. "
            f"Expected: {EXPECTED_EMBEDDING_DIMENSION}."
        )

    return vector


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts.
    """
    if not texts:
        return []

    if any(not text or not text.strip() for text in texts):
        raise ValueError("Input texts must not contain empty values.")

    vectors = invoke_embedding(embeddings.embed_documents, texts)

    for index, vector in enumerate(vectors):
        if len(vector) != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Unexpected embedding dimension at index {index}: "
                f"{len(vector)}. Expected: "
                f"{EXPECTED_EMBEDDING_DIMENSION}."
            )

    return vectors