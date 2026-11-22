from pathlib import Path
from src.rag.ingestion.document_loader import load_chunks
from src.rag.ingestion.document_processor import process_chunks
from src.rag.embeddings.embedding_service import embed_texts
from src.rag.indexing.vector_store import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHUNK_FILE = (
    PROJECT_ROOT
    / "data"
    / "real_corpus"
    / "chunks"
    / "gitlab_structure_aware_v4_500_overlap_100.jsonl"
)

BATCH_SIZE = 16


def ingest_chunks(
    chunk_file: str | Path = CHUNK_FILE,
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Load, process, embed and store document chunks in Chroma.
    """

    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    print("Loading document chunks...")

    chunks = load_chunks(chunk_file)
    processed_chunks = process_chunks(chunks)

    if not processed_chunks:
        print("No chunks available for ingestion.")
        return 0

    store = ChromaVectorStore()

    print(f"Total chunks: {len(processed_chunks)}")
    print(f"Existing Chroma records: {store.count()}")

    for start in range(0, len(processed_chunks), batch_size):
        batch = processed_chunks[start:start + batch_size]

        print(
            f"Processing batch "
            f"{start // batch_size + 1}: "
            f"{len(batch)} chunks"
        )

        vectors = embed_texts(
            [chunk.text for chunk in batch]
        )

        store.add_chunks(batch, vectors)

        print(
            f"Progress: "
            f"{min(start + len(batch), len(processed_chunks))}"
            f"/{len(processed_chunks)}"
        )

    final_count = store.count()

    print("\nIngestion completed.")
    print(f"Chroma records: {final_count}")

    return final_count


if __name__ == "__main__":
    ingest_chunks()