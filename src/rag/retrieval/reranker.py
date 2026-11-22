
import time
from typing import Any

from sentence_transformers import CrossEncoder

from src.rag.schemas import RetrievedChunk, RetrievalResult


class CrossEncoderReranker:
    """Rerank retrieved chunks using a Cross-Encoder model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str | None = None,
        max_length: int = 512,
    ):
        if max_length < 1:
            raise ValueError("max_length must be at least 1.")

        self.model_name = model_name

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        top_k: int | None = None,
    ) -> RetrievalResult:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1.")

        start_time = time.perf_counter()

        candidates = retrieval_result.retrieved_chunks

        if not candidates:
            return RetrievalResult(
                query=query,
                retrieved_chunks=[],
                retrieval_method="hybrid_reranked",
                elapsed_ms=retrieval_result.elapsed_ms,
                metadata={
                    **retrieval_result.metadata,
                    "reranker_model": self.model_name,
                    "reranked_candidates": 0,
                },
            )

        # Pair the same query with every retrieved chunk.
        pairs = [
            [query, item.chunk.text]
            for item in candidates
        ]

        # Calculate Cross-Encoder relevance scores.
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        if len(scores) != len(candidates):
            raise ValueError(
                "Reranker returned an unexpected number of scores."
            )

        scored_candidates = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        if top_k is not None:
            scored_candidates = scored_candidates[:top_k]

        reranked_chunks = []

        for rank, (original, score) in enumerate(
            scored_candidates,
            start=1,
        ):
            reranked_chunks.append(
                RetrievedChunk(
                    chunk=original.chunk,
                    score=float(score),
                    rank=rank,
                    retrieval_method="cross_encoder",
                )
            )

        rerank_elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return RetrievalResult(
            query=query,
            retrieved_chunks=reranked_chunks,
            retrieval_method="hybrid_reranked",
            elapsed_ms=(
                (retrieval_result.elapsed_ms or 0)
                + rerank_elapsed_ms
            ),
            metadata={
                **retrieval_result.metadata,
                "reranker_model": self.model_name,
                "reranked_candidates": len(candidates),
                "rerank_elapsed_ms": rerank_elapsed_ms,
            },
        )