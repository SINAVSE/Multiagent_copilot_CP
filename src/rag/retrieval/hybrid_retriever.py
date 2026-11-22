import time
from typing import Any
from src.rag.retrieval.dense_retriever import DenseRetriever
from src.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag.schemas import RetrievedChunk, RetrievalResult

class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        rrf_k: int = 60,
    ):
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        self.dense_retriever = dense_retriever or DenseRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> RetrievalResult:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if candidate_k is None:
            candidate_k = max(top_k * 2, 10)

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k."
            )

        start_time = time.perf_counter()

        dense_result = self.dense_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            where=where,
        )

        bm25_result = self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            where=where,
        )

        fused_results = {}

        for result in [dense_result, bm25_result]:
            for rank, retrieved in enumerate(
                result.retrieved_chunks,
                start=1,
            ):
                chunk_id = retrieved.chunk.chunk_id

                if chunk_id not in fused_results:
                    fused_results[chunk_id] = {
                        "chunk": retrieved.chunk,
                        "rrf_score": 0.0,
                        "dense_rank": None,
                        "bm25_rank": None,
                    }

                fused_results[chunk_id]["rrf_score"] += (
                    1.0 / (self.rrf_k + rank)
                )

                fused_results[chunk_id][
                    f"{result.retrieval_method}_rank"
                ] = rank

        ranked_results = sorted(
            fused_results.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        retrieved_chunks = []

        for rank, item in enumerate(
            ranked_results[:top_k],
            start=1,
        ):
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=item["chunk"],
                    score=item["rrf_score"],
                    rank=rank,
                    retrieval_method="hybrid",
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            retrieval_method="hybrid",
            elapsed_ms=elapsed_ms,
            metadata={
                "top_k": top_k,
                "candidate_k": candidate_k,
                "rrf_k": self.rrf_k,
                "dense_candidates": len(
                    dense_result.retrieved_chunks
                ),
                "bm25_candidates": len(
                    bm25_result.retrieved_chunks
                ),
                "unique_candidates": len(fused_results),
            },
        )