
import time
import math
from numbers import Real
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
        include_metadata: bool = True,
    ):
        if type(max_length) is not int or max_length < 1:
            raise ValueError(
                "max_length must be a positive integer."
            )

        if type(include_metadata) is not bool:
            raise ValueError(
                "include_metadata must be a boolean."
            )

        self.model_name = model_name
        self.max_length = max_length
        self.include_metadata = include_metadata

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

    @staticmethod
    def build_passage(
        chunk,
        include_metadata=True,
    ) -> str:

        if not include_metadata:
            return chunk.text

        fields = [
            ("Document", chunk.title),
            ("Section", chunk.section_heading),
            ("Section path", chunk.section_path),
        ]

        parts = []

        seen = set()

        for label, value in fields:

            if not value:
                continue

            value = " ".join(value.split())

            if not value:
                continue

            key = value.casefold()

            if key in seen:
                continue

            seen.add(key)

            parts.append(
                f"{label}: {value}"
            )

        if not parts:
            return chunk.text

        parts.append(
            "Policy text: " + chunk.text
        )

        return "\n".join(parts)

    def rerank(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        top_k: int | None = None,
    ) -> RetrievalResult:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k is not None and (
            type(top_k) is not int or top_k < 1
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

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
                    "reranker_include_metadata": self.include_metadata,
                    "reranker_max_length": self.max_length,
                    "reranked_candidates": 0,
                },
            )

        # Pair the same query with every retrieved chunk.
        pairs = [
            [
                query,
                self.build_passage(
                    item.chunk,
                    include_metadata=self.include_metadata,
                ),
            ]
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

        validated_scores = []

        for score in scores:

            if (
                isinstance(score, bool)
                or not isinstance(score, Real)
            ):

                raise ValueError(
                    "Reranker returned a non-numeric score."
                )

            numeric_score = float(score)

            if not math.isfinite(numeric_score):

                raise ValueError(
                    "Reranker returned a non-finite score."
                )

            validated_scores.append(
                numeric_score
            )

        scored_candidates = sorted(
            zip(candidates, validated_scores),
            key=lambda item: item[1],
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
                "reranker_include_metadata": self.include_metadata,
                "reranker_max_length": self.max_length,
                "reranked_candidates": len(candidates),
                "rerank_elapsed_ms": rerank_elapsed_ms,
            },
        )
