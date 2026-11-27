
from pathlib import Path
from numbers import Real
import argparse
import hashlib
import json
import math

from src.rag.ingestion.document_loader import load_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CORPUS_ROOT = (
    PROJECT_ROOT / "data" / "real_corpus"
    / "expanded_v2"
)

CHUNK_FILE = (
    CORPUS_ROOT / "chunking_experiments"
    / "c3_chunks.jsonl"
)

MANIFEST_FILE = (
    CORPUS_ROOT / "provisional_manifest_v2.jsonl"
)

V2_STORE_PATH = (
    PROJECT_ROOT / "data" / "vector_store"
    / "chroma_v2"
)

V2_COLLECTION = "employee_operations_copilot_v2"

EXPECTED_MANIFEST_HASH = (
    "d171aa77c703d5181e77c043953e643659"
    "fa845af85f45a07d026f553fc35b5c"
)

EXPECTED_CHUNK_HASH = (
    "4ec0840471835b8c65220486f99642878"
    "a2c64942b9e278dab5bd59be417c15e"
)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

EXPECTED_DIMENSION = 384

BATCH_SIZE = 16


def sha256_file(path):

    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def validate_embeddings(vectors, expected_count):

    if len(vectors) != expected_count:

        raise ValueError(
            "Embedding batch count mismatch."
        )

    validated = []

    for index, vector in enumerate(vectors):

        if len(vector) != EXPECTED_DIMENSION:

            raise ValueError(
                "Invalid embedding dimension at index "
                + str(index)
            )

        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in vector
        ):

            raise ValueError(
                "Invalid embedding value at index "
                + str(index)
            )

        converted = [float(value) for value in vector]

        if not any(converted):

            raise ValueError(
                "Zero embedding at index " + str(index)
            )

        validated.append(converted)

    return validated


def preflight():

    manifest_hash = sha256_file(MANIFEST_FILE)
    chunk_hash = sha256_file(CHUNK_FILE)

    if manifest_hash != EXPECTED_MANIFEST_HASH:

        raise ValueError("Frozen manifest hash mismatch.")

    if chunk_hash != EXPECTED_CHUNK_HASH:

        raise ValueError("Frozen chunk hash mismatch.")

    documents = [
        json.loads(line)
        for line in MANIFEST_FILE.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    chunks = load_chunks(CHUNK_FILE)

    if len(documents) != 659:

        raise ValueError("Unexpected document count.")

    if len(chunks) != 3224:

        raise ValueError("Unexpected chunk count.")

    document_ids = {
        item["document_id"]
        for item in documents
    }

    if len(document_ids) != len(documents):

        raise ValueError("Duplicate document identifiers.")

    if {
        chunk.document_id for chunk in chunks
    } != document_ids:

        raise ValueError("Document coverage mismatch.")

    if len({
        chunk.chunk_id for chunk in chunks
    }) != len(chunks):

        raise ValueError("Duplicate chunk identifiers.")

    tokens = []

    for chunk in chunks:

        count = getattr(
            chunk, "embedding_input_tokens", None
        )

        if type(count) is not int or not 0 < count <= 512:

            raise ValueError(
                "Invalid token count: " + chunk.chunk_id
            )

        tokens.append(count)

        if not chunk.text.strip():

            raise ValueError("Empty chunk text.")

    return chunks, {
        "manifest_sha256": manifest_hash,
        "chunk_sha256": chunk_hash,
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EXPECTED_DIMENSION,
        "estimated_input_tokens": sum(tokens),
        "maximum_chunk_tokens": max(tokens),
        "estimated_batches": math.ceil(
            len(chunks) / BATCH_SIZE
        ),
    }


def verify_existing(store, chunks):

    expected = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    existing_ids = set()

    page_size = 100
    total = store.count()

    for offset in range(0, total, page_size):

        result = store.collection.get(
            limit=page_size,
            offset=offset,
            include=[
                "documents",
                "metadatas",
                "embeddings",
            ],
        )

        ids = result["ids"]
        texts = result["documents"]
        metadata = result["metadatas"]
        embeddings = result["embeddings"]

        if not (
            len(ids) == len(texts)
            == len(metadata) == len(embeddings)
        ):

            raise ValueError(
                "Invalid stored record batch."
            )

        validate_embeddings(
            embeddings, len(ids)
        )

        for index, chunk_id in enumerate(ids):

            if chunk_id in existing_ids:

                raise ValueError(
                    "Duplicate stored chunk identifier."
                )

            chunk = expected.get(chunk_id)

            if chunk is None:

                raise ValueError(
                    "Unexpected stored chunk: " + chunk_id
                )

            if texts[index] != chunk.text:

                raise ValueError(
                    "Stored chunk content mismatch: "
                    + chunk_id
                )

            for field in (
                "document_id",
                "organization",
                "title",
                "source_url",
            ):

                if metadata[index].get(field) != getattr(
                    chunk, field
                ):

                    raise ValueError(
                        "Stored chunk metadata mismatch: "
                        + chunk_id + " / " + field
                    )

            existing_ids.add(chunk_id)

    if len(existing_ids) != total:

        raise ValueError(
            "Stored record count mismatch."
        )

    return existing_ids


def ingest_resumable(
    store,
    chunks,
    embedder,
    batch_size=BATCH_SIZE,
):

    if type(batch_size) is not int or batch_size < 1:

        raise ValueError("Invalid batch size.")

    if not chunks:

        raise ValueError("No chunks supplied.")

    if len({
        chunk.chunk_id for chunk in chunks
    }) != len(chunks):

        raise ValueError("Duplicate chunk identifiers.")

    existing_ids = verify_existing(store, chunks)

    pending = [
        chunk
        for chunk in chunks
        if chunk.chunk_id not in existing_ids
    ]

    completed = len(existing_ids)
    embedding_batches = 0

    print("Existing verified chunks:", completed)
    print("Pending chunks:", len(pending))

    for start in range(
        0, len(pending), batch_size
    ):

        batch = pending[start:start + batch_size]

        vectors = embedder(
            [chunk.text for chunk in batch]
        )

        embedding_batches += 1

        vectors = validate_embeddings(
            vectors, len(batch)
        )

        store.add_chunks(batch, vectors)

        completed += len(batch)

        print(
            "Progress:",
            completed,
            "/",
            len(chunks),
        )

    verified_ids = verify_existing(store, chunks)

    expected_ids = {
        chunk.chunk_id for chunk in chunks
    }

    if verified_ids != expected_ids:

        raise ValueError(
            "Final vector database verification failed."
        )

    return {
        "total_chunks": len(chunks),
        "previously_stored_chunks": len(existing_ids),
        "newly_embedded_chunks": len(pending),
        "embedding_batches": embedding_batches,
        "verified_chunks": len(verified_ids),
        "status": "completed",
    }


def ingest_guarded_test(
    directory,
    chunks,
    embedder,
    provenance,
    batch_size=BATCH_SIZE,
):

    from tempfile import gettempdir

    from src.rag.indexing.vector_store import (
        ChromaVectorStore,
    )

    from src.rag.ingestion.ingestion_guard import (
        ingestion_lock,
        validate_or_create_provenance,
    )

    directory = Path(directory).resolve()

    temporary_root = Path(gettempdir()).resolve()

    if not directory.is_relative_to(temporary_root):

        raise ValueError(
            "Test ingestion requires a temporary directory."
        )

    if provenance.get("embedding_backend") != "synthetic-test":

        raise ValueError(
            "Only synthetic test embeddings are supported."
        )

    if provenance.get("collection_name") != (
        "cap14_guarded_test"
    ):

        raise ValueError(
            "Invalid test collection name."
        )

    with ingestion_lock(directory):

        store = ChromaVectorStore(
            persist_directory=directory,
            collection_name="cap14_guarded_test",
        )

        provenance_status = validate_or_create_provenance(
            directory,
            provenance,
            store.count(),
        )

        result = ingest_resumable(
            store=store,
            chunks=chunks,
            embedder=embedder,
            batch_size=batch_size,
        )

        result["provenance_status"] = provenance_status

        return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    chunks, summary = preflight()

    print("========== V2 INGESTION PREFLIGHT ==========")

    for key, value in summary.items():

        print(key, ":", value)

    if args.execute:

        raise SystemExit(
            "Hosted ingestion is disabled pending "
            "permission and quota approval."
        )

    print("\nExecution mode: OFFLINE PREFLIGHT")
    print("Hosted embedding calls: 0")
    print("Vector database modified: No")


if __name__ == "__main__":
    main()
