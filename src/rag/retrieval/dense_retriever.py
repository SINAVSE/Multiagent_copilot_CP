
import time
from typing import Any

from src.rag.embeddings.embedding_service import embed_text
from src.rag.indexing.vector_store import ChromaVectorStore
from src.rag.schemas import (
    DocumentChunk,
    RetrievedChunk,
    RetrievalResult,
)


class DenseRetriever:
    def __init__(self, vector_store: ChromaVectorStore | None = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> RetrievalResult:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        start_time = time.perf_counter()

        if self.vector_store.count() == 0:
            return RetrievalResult(
                query=query,
                retrieved_chunks=[],
                retrieval_method="dense",
                elapsed_ms=0.0,
            )

        query_vector = embed_text(query.strip())

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
            similarity_score = 1.0 - distance

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