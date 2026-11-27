
import hashlib
import json
import math
import statistics

from pathlib import Path
from collections import Counter

from transformers import AutoTokenizer

from src.rag.schemas import DocumentChunk
from src.data_ingestion.chunk_documents_structured import (
    split_markdown_sections,
)

ROOT = Path(__file__).resolve().parents[2]

CORPUS_ROOT = ROOT / "data/real_corpus/expanded_v2"
MANIFEST = CORPUS_ROOT / "provisional_manifest_v2.jsonl"
OUTPUT_DIR = CORPUS_ROOT / "chunking_experiments"

TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"

STRATEGIES = {
    "C1": {
        "method": "fixed",
        "size": 256,
        "overlap": 32,
    },
    "C2": {
        "method": "fixed",
        "size": 384,
        "overlap": 64,
    },
    "C3": {
        "method": "fixed",
        "size": 480,
        "overlap": 80,
    },
    "C4": {
        "method": "structured",
        "size": 480,
        "overlap": 80,
    },
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def token_count(text, tokenizer):
    return len(tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    ))


def split_text(text, tokenizer, size, overlap):

    if not 0 <= overlap < size:
        raise ValueError("Invalid overlap.")

    if not text.strip():
        return []

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )

    offsets = encoded["offset_mapping"]

    if not offsets:
        return []

    chunks = []
    start = 0

    while start < len(offsets):

        end = min(start + size, len(offsets))

        while end > start:

            char_start = (
                0 if start == 0
                else offsets[start][0]
            )

            char_end = (
                len(text)
                if end == len(offsets)
                else offsets[end][0]
            )

            chunk_text = text[
                char_start:char_end
            ].strip()

            count = token_count(
                chunk_text,
                tokenizer
            )

            if count <= size:
                break

            end -= 1

        if end <= start or not chunk_text:
            raise ValueError("Invalid token window.")

        chunks.append({
            "text": chunk_text,
            "token_count": count,
            "char_start": char_start,
            "char_end": char_end,
        })

        if end == len(offsets):
            break

        start = max(start + 1, end - overlap)

    return chunks


def create_chunks(text, tokenizer, strategy):

    size = strategy["size"]
    overlap = strategy["overlap"]

    if strategy["method"] == "fixed":

        return split_text(
            text,
            tokenizer,
            size,
            overlap,
        )

    sections = split_markdown_sections(text)

    if not sections:
        raise ValueError("No structured sections found.")

    chunks = []

    for section_index, section in enumerate(sections):

        for chunk in split_text(
            section["text"],
            tokenizer,
            size,
            overlap,
        ):

            chunk["section_heading"] = (
                section["section_heading"]
            )

            chunk["section_path"] = (
                section["section_path"]
            )

            chunk["section_index"] = section_index

            chunks.append(chunk)

    return chunks


def load_documents():

    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    assert len(rows) >= 500

    assert len(rows) == len({
        row["document_id"] for row in rows
    })

    documents = []

    for row in rows:

        path = ROOT / row["processed_path"]

        data = path.read_bytes()

        assert sha256(data) == (
            row["processed_file_hash"]
        ), row["document_id"]

        text = data.decode("utf-8-sig")

        assert text.strip(), row["document_id"]

        documents.append((row, text))

    return documents


def run_strategy(
    name,
    strategy,
    documents,
    tokenizer,
):

    output = OUTPUT_DIR / (
        name.lower() + "_chunks.jsonl"
    )

    temporary = output.with_suffix(".tmp")

    counts = []
    covered = set()
    organizations = Counter()
    chunk_ids = set()
    boundary_warnings = 0

    with temporary.open(
        "w",
        encoding="utf-8"
    ) as stream:

        for row, text in documents:

            chunks = create_chunks(
                text,
                tokenizer,
                strategy,
            )

            assert chunks, row["document_id"]

            if strategy["method"] == "fixed":

                assert chunks[0]["char_start"] == 0

                assert (
                    chunks[-1]["char_end"]
                    == len(text)
                )

                for previous, current in zip(
                    chunks,
                    chunks[1:]
                ):

                    assert (
                        current["char_start"]
                        <= previous["char_end"]
                    )

            else:

                sections = split_markdown_sections(
                    text
                )

                if sum(
                    len(s["text"].split())
                    for s in sections
                ) < len(text.split()) * 0.95:

                    boundary_warnings += 1

            for index, chunk in enumerate(chunks):

                model_tokens = len(
                    tokenizer.encode(
                        chunk["text"],
                        add_special_tokens=True,
                        truncation=False,
                        verbose=False,
                    )
                )

                assert model_tokens <= 512

                assert (
                    chunk["token_count"]
                    <= strategy["size"]
                )

                chunk_id = (
                    row["document_id"]
                    + "-"
                    + name
                    + "-"
                    + str(index + 1).zfill(5)
                )

                assert chunk_id not in chunk_ids

                chunk_ids.add(chunk_id)

                record = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=row["document_id"],
                    title=row["title"],
                    organization=row["organization"],
                    source_url=row["source_url"],
                    text=chunk["text"],
                    section_heading=chunk.get(
                        "section_heading"
                    ),
                    section_path=chunk.get(
                        "section_path"
                    ),
                    chunk_index=index,
                    chunk_size=strategy["size"],
                    overlap=strategy["overlap"],
                    token_count=chunk["token_count"],
                    country=row.get("country"),
                    categories=row.get(
                        "categories", []
                    ),
                    source_id=row["source_id"],
                    corpus_version="2.0-provisional",
                    chunking_strategy=name,
                    embedding_input_tokens=model_tokens,
                    reuse_permission_status=row.get(
                        "redistribution_status",
                        "requires_review"
                    ),
                    char_start=chunk["char_start"],
                    char_end=chunk["char_end"],
                    offset_scope=(
                        "section"
                        if strategy["method"] == "structured"
                        else "document"
                    ),
                    section_index=chunk.get(
                        "section_index"
                    ),
                )

                stream.write(
                    record.model_dump_json()
                    + "\n"
                )

                counts.append(model_tokens)

                organizations[
                    row["organization"]
                ] += 1

            covered.add(row["document_id"])

    assert len(covered) == len(documents)

    temporary.replace(output)

    return {
        "strategy": name,
        "method": strategy["method"],
        "maximum_content_tokens": strategy["size"],
        "overlap": strategy["overlap"],
        "documents_covered": len(covered),
        "total_chunks": len(counts),
        "minimum_embedding_tokens": min(counts),
        "average_embedding_tokens": round(
            statistics.mean(counts), 2
        ),
        "maximum_embedding_tokens": max(counts),
        "total_embedding_input_tokens": sum(counts),
        "estimated_batches_of_16": math.ceil(
            len(counts) / 16
        ),
        "structured_coverage_warnings": (
            boundary_warnings
        ),
        "organization_chunk_counts": dict(
            organizations
        ),
        "chunk_file": str(
            output.relative_to(ROOT)
        ),
        "chunk_file_sha256": sha256(
            output.read_bytes()
        ),
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
        local_files_only=True,
    )

    if not tokenizer.is_fast:
        raise ValueError(
            "Fast tokenizer required for offsets."
        )

    documents = load_documents()

    print("Documents loaded:", len(documents))

    original_tokens = sum(
        token_count(text, tokenizer)
        for _, text in documents
    )

    results = []

    for name, strategy in STRATEGIES.items():

        print("\nRUNNING:", name, flush=True)

        result = run_strategy(
            name,
            strategy,
            documents,
            tokenizer,
        )

        results.append(result)

        for key, value in result.items():

            if key != "organization_chunk_counts":
                print(key, ":", value)

    report = {
        "corpus_version": "2.0-provisional",
        "manifest_sha256": sha256(
            MANIFEST.read_bytes()
        ),
        "tokenizer": TOKENIZER_NAME,
        "document_count": len(documents),
        "original_content_tokens": original_tokens,
        "results": results,
        "hosted_ai_api_calls": 0,
    }

    output = OUTPUT_DIR / (
        "chunking_experiment_summary.json"
    )

    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n========== CHUNKING EXPERIMENT SUMMARY ==========")

    print("Original documents:", len(documents))
    print("Original content tokens:", original_tokens)

    for result in results:

        print(
            result["strategy"],
            "| Chunks:", result["total_chunks"],
            "| Avg tokens:", result["average_embedding_tokens"],
            "| Max tokens:", result["maximum_embedding_tokens"],
            "| Embedding tokens:", result["total_embedding_input_tokens"],
            "| Coverage warnings:", result["structured_coverage_warnings"],
        )

    print("Output:", output.resolve())
    print("Hosted AI API calls: 0")
    print("Original corpus modified: NO")
    print("Vector database modified: NO")
    print("SUCCESS: Chunking experiments completed.")


if __name__ == "__main__":
    main()
