
from pathlib import Path

import chromadb

from src.rag.schemas import DocumentChunk


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "vector_store" / "chroma"
COLLECTION_NAME = "employee_operations_copilot"


class ChromaVectorStore:
    """Persistent Chroma vector store for document chunks."""

    def __init__(
        self,
        persist_directory: str | Path = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> None:
        """Add document chunks and their precomputed embeddings."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs found in input.")

        for chunk, vector in zip(chunks, embeddings):
            if len(vector) != 384:
                raise ValueError(
                    f"Invalid embedding dimension for {chunk.chunk_id}: "
                    f"{len(vector)}. Expected 384."
                )

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            batch_embeddings = embeddings[start:start + batch_size]

            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                embeddings=batch_embeddings,
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[
                    {
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "organization": chunk.organization,
                        "source_url": chunk.source_url,
                        "section_heading": chunk.section_heading or "",
                        "section_path": chunk.section_path or "",
                        "chunk_index": chunk.chunk_index or 0,
                        "token_count": chunk.token_count or 0,
                    }
                    for chunk in batch_chunks
                ],
            )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> dict:
        """Search for the most similar chunks."""

        if len(query_embedding) != 384:
            raise ValueError(
                f"Expected a 384-dimensional query embedding, "
                f"received {len(query_embedding)}."
            )

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()