from config.clients import embeddings


def test_embedding_connection():
    response = embeddings.embed_query(
        "Employee leave policy and annual leave entitlement"
    )

    print("\nEmbedding dimensions:", len(response))

    assert len(response) > 0