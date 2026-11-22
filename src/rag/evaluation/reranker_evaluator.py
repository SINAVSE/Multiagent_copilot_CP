
import json
from pathlib import Path
from statistics import mean

from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.retrieval.reranker import CrossEncoderReranker


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_FILE = (
    PROJECT_ROOT
    / "data/evaluation/real_corpus/retrieval_eval_v1.0.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data/evaluation/real_corpus/"
    / "hybrid_reranker_retrieval_results_v1.0.json"
)


def calculate_recall(expected, retrieved):
    if not expected:
        return 0.0

    return len(set(expected) & set(retrieved)) / len(set(expected))


def calculate_reciprocal_rank(expected, retrieved):
    expected = set(expected)

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected:
            return 1.0 / rank

    return 0.0


def main():
    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        evaluation_data = json.load(file)

    queries = evaluation_data["queries"]

    retriever = HybridRetriever()

    print("Loading Cross-Encoder...")
    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-base"
    )

    query_results = []

    recall_1_values = []
    recall_3_values = []
    recall_5_values = []
    reciprocal_ranks = []

    print(f"\nEvaluating {len(queries)} queries...\n")

    for query_data in queries:
        query_id = query_data["query_id"]
        query = query_data["query"]
        expected = query_data["expected_chunk_ids"]

        # Retrieve 10 hybrid candidates.
        hybrid_result = retriever.retrieve(
            query=query,
            top_k=10,
            candidate_k=10,
        )

        # Rerank all 10 candidates.
        reranked_result = reranker.rerank(
            query=query,
            retrieval_result=hybrid_result,
            top_k=10,
        )

        retrieved_ids = [
            item.chunk.chunk_id
            for item in reranked_result.retrieved_chunks
        ]

        recall_1 = calculate_recall(expected, retrieved_ids[:1])
        recall_3 = calculate_recall(expected, retrieved_ids[:3])
        recall_5 = calculate_recall(expected, retrieved_ids[:5])

        reciprocal_rank = calculate_reciprocal_rank(
            expected,
            retrieved_ids,
        )

        recall_1_values.append(recall_1)
        recall_3_values.append(recall_3)
        recall_5_values.append(recall_5)
        reciprocal_ranks.append(reciprocal_rank)

        query_results.append({
            "query_id": query_id,
            "query": query,
            "expected_chunk_ids": expected,
            "retrieved_chunk_ids": retrieved_ids,
            "recall_at_1": recall_1,
            "recall_at_3": recall_3,
            "recall_at_5": recall_5,
            "reciprocal_rank": reciprocal_rank,
        })

        print(
            f"{query_id}: "
            f"Recall@5={recall_5:.2f}, "
            f"RR={reciprocal_rank:.2f}"
        )

    summary = {
        "total_queries": len(queries),
        "retrieval_method": "hybrid_rrf_with_cross_encoder",
        "hybrid_candidate_k": 10,
        "reranker_model": "BAAI/bge-reranker-base",
        "metrics": {
            "recall_at_1": mean(recall_1_values),
            "recall_at_3": mean(recall_3_values),
            "recall_at_5": mean(recall_5_values),
            "mrr": mean(reciprocal_ranks),
        },
        "query_results": query_results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("HYBRID + RERANKER EVALUATION")
    print("=" * 50)

    for metric, value in summary["metrics"].items():
        print(f"{metric}: {value:.4f}")

    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()