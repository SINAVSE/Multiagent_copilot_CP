import json
from pathlib import Path
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "real_corpus" / "processed" / "gitlab"
MANIFEST_FILE = PROJECT_ROOT / "data" / "real_corpus" / "metadata" / "corpus_manifest.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_corpus" / "chunks"

TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def load_manifest():
    records = {}

    with MANIFEST_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records[record["document_id"]] = record

    return records


def split_into_chunks(text, tokenizer):
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(token_ids), step):
        end = min(start + CHUNK_SIZE, len(token_ids))

        chunk_ids = token_ids[start:end]

        if not chunk_ids:
            continue

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
        )

        chunks.append({
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

    output_file = OUTPUT_DIR / "gitlab_fixed_800_overlap_100.jsonl"

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

            chunks = split_into_chunks(text, tokenizer)

            for index, chunk in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f"{document_id}-C{index + 1:04d}",
                    "document_id": document_id,
                    "title": record["title"],
                    "organization": record["organization"],
                    "source_url": record["source_url"],
                    "license": record["license"],
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
            print(f"{document_id}: {len(chunks)} chunks")

    print("\nChunking completed")
    print(f"Documents processed: {document_count}")
    print(f"Total chunks: {total_chunks}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()