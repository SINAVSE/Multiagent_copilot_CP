
import json
import math
import statistics

from collections import Counter
from transformers import AutoTokenizer

from src.rag.schemas import DocumentChunk

from src.data_ingestion.chunking_experiments_v2 import (
    ROOT,
    MANIFEST,
    OUTPUT_DIR,
    TOKENIZER_NAME,
    load_documents,
    sha256,
)

from src.data_ingestion.enhanced_structured_chunking_v2 import (
    create_enhanced_chunks,
)


def main():

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
        local_files_only=True,
    )

    documents = load_documents()

    output = OUTPUT_DIR / "c5_chunks.jsonl"

    temporary = output.with_suffix(".tmp")

    counts = []
    content_counts = []
    covered = set()
    chunk_ids = set()
    organization_counts = Counter()
    diagnostics = Counter()

    with temporary.open(
        "w", encoding="utf-8"
    ) as stream:

        for row, text in documents:

            chunks, report = create_enhanced_chunks(
                text,
                tokenizer,
                row["source_id"],
            )

            diagnostics[
                "navigation_lines_removed"
            ] += report["removed_navigation_lines"]

            diagnostics[
                "original_sections"
            ] += report["original_sections"]

            diagnostics[
                "merged_sections"
            ] += report["merged_sections"]

            diagnostics[
                "preservation_passed"
            ] += int(
                report["content_preservation_passed"]
            )

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

                assert chunk["token_count"] <= 480

                chunk_id = (
                    row["document_id"]
                    + "-C5-"
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
                    section_heading=chunk["section_heading"],
                    section_path=chunk["section_path"],
                    chunk_index=index,
                    chunk_size=480,
                    overlap=80,
                    token_count=chunk["token_count"],
                    country=row.get("country"),
                    categories=row.get("categories", []),
                    source_id=row["source_id"],
                    corpus_version="2.0-provisional",
                    chunking_strategy="C5",
                    embedding_input_tokens=model_tokens,
                    reuse_permission_status=row.get(
                        "redistribution_status",
                        "requires_review"
                    ),
                    char_start=chunk["char_start"],
                    char_end=chunk["char_end"],
                    offset_scope="merged_section",
                    section_index=chunk["section_index"],
                )

                stream.write(
                    record.model_dump_json() + "\n"
                )

                counts.append(model_tokens)

                content_counts.append(
                    chunk["token_count"]
                )

                organization_counts[
                    row["organization"]
                ] += 1

            covered.add(row["document_id"])

    assert len(covered) == len(documents)

    assert (
        diagnostics["preservation_passed"]
        == len(documents)
    )

    temporary.replace(output)

    summary = {
        "strategy": "C5",
        "method": "enhanced_structured",
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
        "chunks_below_16_content_tokens": sum(
            value < 16 for value in content_counts
        ),
        "chunks_below_32_content_tokens": sum(
            value < 32 for value in content_counts
        ),
        "content_preservation_passed": (
            diagnostics["preservation_passed"]
        ),
        "navigation_lines_removed": (
            diagnostics["navigation_lines_removed"]
        ),
        "original_sections": (
            diagnostics["original_sections"]
        ),
        "merged_sections": (
            diagnostics["merged_sections"]
        ),
        "manifest_sha256": sha256(
            MANIFEST.read_bytes()
        ),
        "chunk_file_sha256": sha256(
            output.read_bytes()
        ),
        "organization_chunk_counts": dict(
            organization_counts
        ),
        "hosted_ai_api_calls": 0,
    }

    report_file = (
        OUTPUT_DIR / "c5_experiment_summary.json"
    )

    report_file.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n========== C5 EXPERIMENT SUMMARY ==========")

    for key, value in summary.items():
        print(key, ":", value)

    print("Original C1-C4 files modified: NO")
    print("Original corpus modified: NO")
    print("Vector database modified: NO")
    print("SUCCESS: C5 experiment completed.")


if __name__ == "__main__":
    main()
