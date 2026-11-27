
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "real_corpus" / "raw" / "gitlab"
PROCESSED_DIR = PROJECT_ROOT / "data" / "real_corpus" / "processed" / "gitlab"
MANIFEST_FILE = PROJECT_ROOT / "data" / "real_corpus" / "metadata" / "corpus_manifest.jsonl"


def load_manifest():
    records = {}

    with MANIFEST_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records[record["local_path"]] = record

    return records


def clean_markdown(text):
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove YAML frontmatter while retaining the title in the manifest
    text = re.sub(
        r"\A---\s*\n.*?\n---\s*\n",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Remove trailing whitespace
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # Limit excessive blank lines without removing section boundaries
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"


def main():
    manifest = load_manifest()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.rglob("*.md"))

    processed_count = 0

    for file_path in files:
        relative_path = file_path.relative_to(PROJECT_ROOT).as_posix()

        if relative_path not in manifest:
            print(f"WARNING: No manifest record for {relative_path}")
            continue

        record = manifest[relative_path]

        raw_text = file_path.read_text(encoding="utf-8")
        processed_text = clean_markdown(raw_text)

        output_file = PROCESSED_DIR / f"{record['document_id']}.md"

        output_file.write_text(processed_text, encoding="utf-8")

        print(f"Processed: {record['document_id']} -> {output_file.name}")
        processed_count += 1

    print(f"\nTotal documents processed: {processed_count}")


if __name__ == "__main__":
    main()