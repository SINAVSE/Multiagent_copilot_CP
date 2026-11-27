
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.retrieval.reranker import CrossEncoderReranker


def main():
    query = (
        "What are the meal and incidental expense (M&IE) "
        "per diem rules for sponsored business travel?"
    )

    print("\nUSER QUESTION:")
    print(query)

    # Step 1: Retrieve candidates using Hybrid RRF.
    print("\nRunning hybrid retrieval...")

    retriever = HybridRetriever()

    retrieval_result = retriever.retrieve(
        query=query,
        top_k=10,
        candidate_k=10,
    )

    print("\nHYBRID RETRIEVAL RESULTS:")

    for item in retrieval_result.retrieved_chunks:
        print(
            f"Rank {item.rank}: {item.chunk.chunk_id} "
            f"| RRF score: {item.score:.5f}"
        )

    # Step 2: Load the local Cross-Encoder.
    print("\nLoading Cross-Encoder model...")
    print("First run may download the model.")

    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-base"
    )

    # Step 3: Rerank the retrieved candidates.
    print("\nRunning reranker...")

    reranked_result = reranker.rerank(
        query=query,
        retrieval_result=retrieval_result,
        top_k=5,
    )

    # Step 4: Display reranked results.
    print("\nRERANKED RESULTS:")

    for item in reranked_result.retrieved_chunks:
        print(
            f"Rank {item.rank}: {item.chunk.chunk_id} "
            f"| Relevance score: {item.score:.4f}"
        )

    print("\nReranker model:", reranked_result.metadata["reranker_model"])
    print(
        "Candidates reranked:",
        reranked_result.metadata["reranked_candidates"],
    )

    print("\nTEST COMPLETED")


if __name__ == "__main__":
    main()