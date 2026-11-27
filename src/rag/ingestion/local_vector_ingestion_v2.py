from pathlib import Path

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import argparse
import time

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from src.rag.indexing.vector_store import ChromaVectorStore

from src.rag.ingestion.vector_ingestion_v2 import (
    PROJECT_ROOT,
    EMBEDDING_MODEL,
    preflight,
    ingest_resumable,
)

from src.rag.ingestion.ingestion_guard import (
    ingestion_lock,
    validate_or_create_provenance,
)

LOCAL_STORE_PATH = (
    PROJECT_ROOT / "data" / "vector_store"
    / "chroma_v2_local"
)

LOCAL_COLLECTION = (
    "employee_operations_copilot_v2_local"
)

EXPECTED_REVISION = (
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot",
        action="store_true",
    )

    args = parser.parse_args()

    start = time.perf_counter()

    chunks, summary = preflight()

    model_path = str(
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--BAAI--bge-small-en-v1.5"
        / "snapshots"
        / EXPECTED_REVISION
    )

    model_directory = Path(model_path)

    required_files = [
        "config.json",
        "model.safetensors",
    ]

    missing = [
        name for name in required_files
        if not (model_directory / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required model files missing: "
            + ", ".join(missing)
        )

    print("Required model files verified.")

    revision = os.path.basename(model_path)

    if revision != EXPECTED_REVISION:
        raise ValueError("Unexpected model revision.")

    print("Loading local embedding model...")

    model = SentenceTransformer(
        model_path,
        device="cpu",
        local_files_only=True,
    )

    if model.get_sentence_embedding_dimension() != 384:
        raise ValueError("Unexpected embedding dimension.")

    model.max_seq_length = 512

    print("Model loaded:", EMBEDDING_MODEL)
    print("Model revision:", revision)

    def embedder(texts):

        batch_start = time.perf_counter()

        vectors = model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        elapsed = time.perf_counter() - batch_start

        print(
            "Embedding batch:",
            len(texts),
            "chunks;",
            round(elapsed, 2),
            "seconds",
            flush=True,
        )

        return vectors.tolist()

    provenance = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": 384,
        "embedding_backend": "local-cpu",
        "manifest_sha256": summary["manifest_sha256"],
        "chunk_sha256": summary["chunk_sha256"],
        "collection_name": LOCAL_COLLECTION,
        "model_revision": revision,
        "normalize_embeddings": True,
        "max_seq_length": 512,
    }

    with ingestion_lock(LOCAL_STORE_PATH):

        store = ChromaVectorStore(
            persist_directory=LOCAL_STORE_PATH,
            collection_name=LOCAL_COLLECTION,
        )

        if args.pilot and store.count() > 16:
            raise ValueError(
                "Full ingestion has already started."
            )

        status = validate_or_create_provenance(
            LOCAL_STORE_PATH,
            provenance,
            store.count(),
        )

        print("Provenance:", status)

        selected = chunks[:16] if args.pilot else chunks

        result = ingest_resumable(
            store=store,
            chunks=selected,
            embedder=embedder,
            batch_size=16,
        )

        print("\n========== LOCAL INGESTION RESULTS ==========")

        for key, value in result.items():
            print(key, ":", value)

        print("Database:", LOCAL_STORE_PATH)
        print("Model revision:", revision)
        print("Hosted embedding calls: 0")
        print("Hosted LLM calls: 0")
        print(
            "Total elapsed seconds:",
            round(time.perf_counter() - start, 2),
        )


if __name__ == "__main__":
    main()
