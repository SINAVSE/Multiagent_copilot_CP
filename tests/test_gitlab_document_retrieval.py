
from src.rag.embeddings.embedding_service import embed_texts
from src.rag.indexing.vector_store import ChromaVectorStore


DOCUMENT_IDS = [
    f"GITLAB-{number:04d}"
    for number in range(1, 10)
]

# Broad queries for checking whether each document has retrievable content.
QUERIES = [
    "What does this document say about employee policies and procedures?",
    "What are the requirements, responsibilities and rules described here?",
    "What actions should employees or managers take according to this document?",
]


def test_all_nine_gitlab_documents_are_retrievable():
    store = ChromaVectorStore()

    for document_id in DOCUMENT_IDS:
        retrieved_chunks = []

        for query in QUERIES:
            query_embedding = embed_texts([query])[0]

            results = store.search(
                query_embedding=query_embedding,
                top_k=5,
                where={"document_id": document_id},
            )

            retrieved_chunks.extend(
                results.get("metadatas", [[]])[0]
            )

        retrieved_ids = {
            metadata.get("document_id")
            for metadata in retrieved_chunks
            if metadata
        }

        assert document_id in retrieved_ids, (
            f"No chunks retrieved for {document_id}"
        )

        print(
            f"{document_id}: "
            f"{len(retrieved_chunks)} retrieved results"
        )