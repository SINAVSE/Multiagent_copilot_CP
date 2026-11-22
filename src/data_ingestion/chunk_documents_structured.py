
import json
import re
from pathlib import Path
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "real_corpus" / "processed" / "gitlab"
)
MANIFEST_FILE = (
    PROJECT_ROOT / "data" / "real_corpus" / "metadata" / "corpus_manifest.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_corpus" / "chunks"

TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_manifest():
    records = {}

    with MANIFEST_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records[record["document_id"]] = record

    return records




def split_markdown_sections(text):
    """Split Markdown while preserving Hugo details hierarchy."""

    sections = []
    heading_stack = []
    current_lines = []
    details_stack = []

    def save_section():
        nonlocal current_lines

        body = "\n".join(current_lines).strip()

        # Remove Markdown heading lines temporarily.
        # A heading-only section should remain in the hierarchy,
        # but should not become a standalone retrieval chunk.
        content_without_headings = re.sub(
            r"(?m)^\s*#{1,6}\s+.*$",
            "",
            body
        ).strip()

        if content_without_headings:
            path_items = [item["title"] for item in heading_stack]

            sections.append({
                "section_heading": (
                    path_items[-1]
                    if path_items
                    else "Document introduction"
                ),
                "section_path": " > ".join(path_items)
                    or "Document introduction",
                "text": body,
            })

        current_lines = []

    for line in text.splitlines():

        # Detect Hugo details opening shortcode.
        opening = re.match(
            r'^\s*\{\{%\s*details\s+summary=["\'](.+?)["\']\s*%\}\}\s*$',
            line
        )

        if opening:
            save_section()

            summary = opening.group(1).strip()

            details_stack.append({
                "summary": summary,
                "base_heading_count": len(heading_stack),
            })

            # Add the shortcode summary as a logical nested heading.
            heading_stack.append({
                "level": 7,
                "title": summary,
            })

            continue

        # Detect Hugo details closing shortcode.
        if re.match(r'^\s*\{\{%\s*/details\s*%\}\}\s*$', line):
            save_section()

            if details_stack:
                details = details_stack.pop()
                base_count = details["base_heading_count"]
                heading_stack[:] = heading_stack[:base_count]

            continue

        # Ignore other shortcode closing markers.
        if re.match(r'^\s*\{\{%\s*/', line):
            continue

        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)

        if match:
            save_section()

            level = len(match.group(1))
            heading = match.group(2).strip()

            # Inside details, the shortcode summary already represents
            # the nested section heading. Avoid duplicating it.
            if (
                details_stack
                and heading == details_stack[-1]["summary"]
            ):
                continue

            heading_stack[:] = [
                item for item in heading_stack
                if item["level"] < level
            ]

            heading_stack.append({
                "level": level,
                "title": heading,
            })

            current_lines.append(line)

        else:
            current_lines.append(line)

    save_section()

    return sections

def split_oversized_section(section, tokenizer):
    """Split a section into overlapping token windows."""
    text = section["text"]
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if len(token_ids) <= CHUNK_SIZE:
        return [{
            **section,
            "text": text,
            "token_count": len(token_ids),
        }]

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(token_ids), step):
        end = min(start + CHUNK_SIZE, len(token_ids))
        chunk_ids = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        )

        chunks.append({
            **section,
            "text": chunk_text,
            "token_count": len(chunk_ids),
        })

        if end == len(token_ids):
            break

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    manifest = load_manifest()

    output_file = OUTPUT_DIR / "gitlab_structure_aware_v4_500_overlap_100.jsonl"

    total_chunks = 0
    document_count = 0

    with output_file.open("w", encoding="utf-8") as output:
        for file_path in sorted(PROCESSED_DIR.glob("*.md")):
            document_id = file_path.stem

            if document_id not in manifest:
                print(f"WARNING: Missing manifest entry: {document_id}")
                continue

            record = manifest[document_id]
            text = file_path.read_text(encoding="utf-8")

            sections = split_markdown_sections(text)
            document_chunks = []

            for section in sections:
                document_chunks.extend(
                    split_oversized_section(section, tokenizer)
                )

            for index, chunk in enumerate(document_chunks):
                chunk_record = {
                    "chunk_id": f"{document_id}-S{index + 1:04d}",
                    "document_id": document_id,
                    "title": record["title"],
                    "organization": record["organization"],
                    "source_url": record["source_url"],
                    "license": record["license"],
                    "section_heading": chunk["section_heading"],
                    "section_path": chunk["section_path"],
                    "chunk_index": index,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "token_count": chunk["token_count"],
                    "text": chunk["text"],
                }

                output.write(
                    json.dumps(chunk_record, ensure_ascii=False) + "\n"
                )

                total_chunks += 1

            document_count += 1

            print(
                f"{document_id}: {len(document_chunks)} chunks "
                f"from {len(sections)} sections"
            )

    print("\nStructure-aware chunking completed")
    print(f"Documents processed: {document_count}")
    print(f"Total chunks: {total_chunks}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()