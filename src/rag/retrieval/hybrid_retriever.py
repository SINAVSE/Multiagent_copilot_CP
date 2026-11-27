import time
from typing import Any

from src.rag.retrieval.dense_retriever import DenseRetriever
from src.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag.schemas import RetrievedChunk, RetrievalResult
from src.rag.retrieval.chunk_integrity import ChunkIntegrityValidator


class HybridRetriever:

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        rrf_k: int = 60,
    ):

        if type(rrf_k) is not int or rrf_k < 1:

            raise ValueError(
                "rrf_k must be a positive integer."
            )

        self.dense_retriever = (
            dense_retriever or DenseRetriever()
        )

        self.bm25_retriever = (
            bm25_retriever or BM25Retriever()
        )

        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> RetrievalResult:

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        if type(top_k) is not int or top_k < 1:

            raise ValueError(
                "top_k must be a positive integer."
            )

        if candidate_k is None:

            candidate_k = max(
                top_k * 2, 10
            )

        if (
            type(candidate_k) is not int
            or candidate_k < top_k
        ):

            raise ValueError(
                "candidate_k must be an integer "
                "greater than or equal to top_k."
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

        duplicate_counts = {
            "dense": 0,
            "bm25": 0,
        }

        invalid_identifier_counts = {
            "dense": 0,
            "bm25": 0,
        }

        unique_candidate_counts = {
            "dense": 0,
            "bm25": 0,
        }

        retrieval_sources = [
            ("dense", dense_result),
            ("bm25", bm25_result),
        ]

        for method, result in retrieval_sources:

            seen_chunk_ids = set()

            unique_rank = 0

            for retrieved in result.retrieved_chunks:

                chunk = retrieved.chunk

                chunk_id = chunk.chunk_id

                if (
                    not chunk_id.strip()
                    or not chunk.document_id.strip()
                ):

                    invalid_identifier_counts[method] += 1

                    continue

                # Detect conflicting policy evidence
                # before ignoring duplicate identifiers.

                if chunk_id in fused_results:

                    existing_chunk = (
                        fused_results[chunk_id]["chunk"]
                    )

                    fused_results[chunk_id]["chunk"] = (
                        ChunkIntegrityValidator.merge(
                            existing_chunk,
                            chunk,
                        )
                    )

                # Each retrieval method contributes
                # at most once per chunk.

                if chunk_id in seen_chunk_ids:

                    duplicate_counts[method] += 1

                    continue

                seen_chunk_ids.add(chunk_id)

                unique_rank += 1

                if chunk_id not in fused_results:

                    fused_results[chunk_id] = {
                        "chunk": chunk,
                        "rrf_score": 0.0,
                        "dense_rank": None,
                        "bm25_rank": None,
                    }

                fused_results[chunk_id]["rrf_score"] += (
                    1.0 / (
                        self.rrf_k + unique_rank
                    )
                )

                fused_results[chunk_id][
                    f"{method}_rank"
                ] = unique_rank

            unique_candidate_counts[method] = (
                len(seen_chunk_ids)
            )

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

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

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

                "unique_candidates": len(
                    fused_results
                ),

                "dense_duplicates_removed": (
                    duplicate_counts["dense"]
                ),

                "bm25_duplicates_removed": (
                    duplicate_counts["bm25"]
                ),

                "dense_invalid_identifiers_removed": (
                    invalid_identifier_counts["dense"]
                ),

                "bm25_invalid_identifiers_removed": (
                    invalid_identifier_counts["bm25"]
                ),

                "dense_unique_candidates": (
                    unique_candidate_counts["dense"]
                ),

                "bm25_unique_candidates": (
                    unique_candidate_counts["bm25"]
                ),

                "rank_fusion": {
                    chunk_id: {
                        "dense_rank": item["dense_rank"],
                        "bm25_rank": item["bm25_rank"],
                        "rrf_score": item["rrf_score"],
                    }
                    for chunk_id, item
                    in fused_results.items()
                },
            },
        )
