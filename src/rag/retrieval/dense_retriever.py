
import time
from typing import Any
from collections.abc import Callable
from numbers import Real
import math

from src.rag.embeddings.embedding_service import embed_text
from src.rag.indexing.vector_store import ChromaVectorStore
from src.rag.schemas import (
    DocumentChunk,
    RetrievedChunk,
    RetrievalResult,
)


class DenseRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        embedding_function: Callable | None = None,
    ):

        self.vector_store = (
            vector_store
            if vector_store is not None
            else ChromaVectorStore()
        )

        if (
            embedding_function is not None
            and not callable(embedding_function)
        ):

            raise TypeError(
                "embedding_function must be callable."
            )

        self.embedding_function = embedding_function

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> RetrievalResult:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be at least 1.")

        start_time = time.perf_counter()

        if self.vector_store.count() == 0:
            return RetrievalResult(
                query=query,
                retrieved_chunks=[],
                retrieval_method="dense",
                elapsed_ms=0.0,
            )

        embedding_function = (
            self.embedding_function
            if self.embedding_function is not None
            else embed_text
        )

        query_vector = embedding_function(
            query.strip()
        )

        if len(query_vector) != 384:

            raise ValueError(
                "Invalid query embedding dimension."
            )

        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in query_vector
        ):

            raise ValueError(
                "Invalid query embedding values."
            )

        query_vector = [
            float(value) for value in query_vector
        ]

        if not any(query_vector):

            raise ValueError(
                "Query embedding cannot be a zero vector."
            )

        results = self.vector_store.search(
            query_embedding=query_vector,
            top_k=top_k,
            where=where,
        )

        retrieved_chunks = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not (
            len(ids) == len(documents)
            == len(metadatas) == len(distances)
        ):

            raise ValueError(
                "Inconsistent dense retrieval results."
            )

        if len(ids) != len(set(ids)):

            raise ValueError(
                "Duplicate dense retrieval chunk IDs."
            )

        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=metadata["document_id"],
                title=metadata["title"],
                organization=metadata["organization"],
                source_url=metadata["source_url"],
                text=text,
                section_heading=metadata.get("section_heading"),
                section_path=metadata.get("section_path"),
                chunk_index=metadata.get("chunk_index"),
                token_count=metadata.get("token_count"),
            )

            # Chroma cosine distance = 1 - cosine similarity.
            if (
                isinstance(distance, bool)
                or not isinstance(distance, Real)
                or not math.isfinite(distance)
            ):

                raise ValueError(
                    "Invalid dense retrieval distance."
                )

            similarity_score = 1.0 - float(distance)

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=similarity_score,
                    rank=rank,
                    retrieval_method="dense",
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            retrieval_method="dense",
            elapsed_ms=elapsed_ms,
            metadata={
                "top_k": top_k,
                "returned_chunks": len(retrieved_chunks),
                "similarity_metric": "cosine_similarity",
            },
        )
