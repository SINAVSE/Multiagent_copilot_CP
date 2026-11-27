import json
from pathlib import Path
from statistics import mean
from src.rag.retrieval.dense_retriever import DenseRetriever
from src.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag.retrieval.hybrid_retriever import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "real_corpus"
    / "retrieval_eval_v1.0.json"
)


def load_evaluation_queries(file_path: Path) -> list[dict]:
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    queries = data["queries"]

    if not queries:
        raise ValueError("Evaluation set contains no queries.")

    return queries


def calculate_recall(
    expected_chunks: set[str],
    retrieved_chunks: list[str],
) -> float:
    if not expected_chunks:
        return 0.0

    retrieved_set = set(retrieved_chunks)

    return len(expected_chunks & retrieved_set) / len(expected_chunks)


def calculate_reciprocal_rank(
    expected_chunks: set[str],
    retrieved_chunks: list[str],
) -> float:
    for rank, chunk_id in enumerate(retrieved_chunks, start=1):
        if chunk_id in expected_chunks:
            return 1.0 / rank

    return 0.0


def evaluate_dense_retrieval(top_k: int = 10) -> dict:
    queries = load_evaluation_queries(EVALUATION_FILE)

    retriever = DenseRetriever()

    recall_scores = {1: [], 3: [], 5: []}
    reciprocal_ranks = []
    query_results = []

    for item in queries:
        query_id = item["query_id"]
        query = item["query"]

        expected_chunks = set(item["expected_chunk_ids"])

        result = retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        retrieved_ids = [
            item.chunk.chunk_id
            for item in result.retrieved_chunks
        ]

        for k in recall_scores:
            recall_scores[k].append(
                calculate_recall(
                    expected_chunks,
                    retrieved_ids[:k],
                )
            )

        reciprocal_rank = calculate_reciprocal_rank(
            expected_chunks,
            retrieved_ids,
        )

        reciprocal_ranks.append(reciprocal_rank)

        query_results.append({
            "query_id": query_id,
            "query": query,
            "expected_chunk_ids": sorted(expected_chunks),
            "retrieved_chunk_ids": retrieved_ids,
            "reciprocal_rank": reciprocal_rank,
        })

        print(
            f"{query_id}: "
            f"Recall@5={recall_scores[5][-1]:.2f}, "
            f"RR={reciprocal_rank:.2f}"
        )

    summary = {
        "retrieval_method": "dense",
        "total_queries": len(queries),
        "recall_at_1": mean(recall_scores[1]),
        "recall_at_3": mean(recall_scores[3]),
        "recall_at_5": mean(recall_scores[5]),
        "mrr": mean(reciprocal_ranks),
    }

    print("\n========== DENSE RETRIEVAL RESULTS ==========")

    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    output_file = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "real_corpus"
        / "dense_retrieval_results_v1.0.json"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": summary,
                "query_results": query_results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nDetailed results saved to: {output_file}")

    return summary

def evaluate_bm25_retrieval(top_k: int = 10) -> dict:
    queries = load_evaluation_queries(EVALUATION_FILE)

    retriever = BM25Retriever()

    recall_scores = {1: [], 3: [], 5: []}
    reciprocal_ranks = []
    query_results = []

    for item in queries:
        query_id = item["query_id"]
        query = item["query"]

        expected_chunks = set(item["expected_chunk_ids"])

        result = retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        retrieved_ids = [
            retrieved.chunk.chunk_id
            for retrieved in result.retrieved_chunks
        ]

        for k in recall_scores:
            recall_scores[k].append(
                calculate_recall(
                    expected_chunks,
                    retrieved_ids[:k],
                )
            )

        reciprocal_rank = calculate_reciprocal_rank(
            expected_chunks,
            retrieved_ids,
        )

        reciprocal_ranks.append(reciprocal_rank)

        query_results.append({
            "query_id": query_id,
            "query": query,
            "expected_chunk_ids": sorted(expected_chunks),
            "retrieved_chunk_ids": retrieved_ids,
            "reciprocal_rank": reciprocal_rank,
        })

        print(
            f"{query_id}: "
            f"Recall@5={recall_scores[5][-1]:.2f}, "
            f"RR={reciprocal_rank:.2f}"
        )

    summary = {
        "retrieval_method": "bm25",
        "total_queries": len(queries),
        "recall_at_1": mean(recall_scores[1]),
        "recall_at_3": mean(recall_scores[3]),
        "recall_at_5": mean(recall_scores[5]),
        "mrr": mean(reciprocal_ranks),
    }

    print("\n========== BM25 RETRIEVAL RESULTS ==========")

    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    output_file = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "real_corpus"
        / "bm25_retrieval_results_v1.0.json"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": summary,
                "query_results": query_results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nDetailed results saved to: {output_file}")

    return summary


def evaluate_hybrid_retrieval(top_k: int = 10) -> dict:
    queries = load_evaluation_queries(EVALUATION_FILE)

    retriever = HybridRetriever()

    recall_scores = {1: [], 3: [], 5: []}
    reciprocal_ranks = []
    query_results = []

    for item in queries:
        query_id = item["query_id"]
        query = item["query"]

        expected_chunks = set(item["expected_chunk_ids"])

        result = retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        retrieved_ids = [
            retrieved.chunk.chunk_id
            for retrieved in result.retrieved_chunks
        ]

        for k in recall_scores:
            recall_scores[k].append(
                calculate_recall(
                    expected_chunks,
                    retrieved_ids[:k],
                )
            )

        reciprocal_rank = calculate_reciprocal_rank(
            expected_chunks,
            retrieved_ids,
        )

        reciprocal_ranks.append(reciprocal_rank)

        query_results.append({
            "query_id": query_id,
            "query": query,
            "expected_chunk_ids": sorted(expected_chunks),
            "retrieved_chunk_ids": retrieved_ids,
            "reciprocal_rank": reciprocal_rank,
        })

        print(
            f"{query_id}: "
            f"Recall@5={recall_scores[5][-1]:.2f}, "
            f"RR={reciprocal_rank:.2f}"
        )

    summary = {
        "retrieval_method": "hybrid_rrf",
        "total_queries": len(queries),
        "recall_at_1": mean(recall_scores[1]),
        "recall_at_3": mean(recall_scores[3]),
        "recall_at_5": mean(recall_scores[5]),
        "mrr": mean(reciprocal_ranks),
    }

    print("\n========== HYBRID RETRIEVAL RESULTS ==========")

    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    output_file = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "real_corpus"
        / "hybrid_retrieval_results_v1.0.json"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": summary,
                "query_results": query_results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nDetailed results saved to: {output_file}")

    return summary

if __name__ == "__main__":
    evaluate_dense_retrieval()
    evaluate_bm25_retrieval()
    evaluate_hybrid_retrieval()