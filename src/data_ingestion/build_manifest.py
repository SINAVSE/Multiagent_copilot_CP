
import hashlib
import json
import re
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "real_corpus" / "raw" / "gitlab"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "real_corpus"
    / "metadata"
    / "corpus_manifest.jsonl"
)

REPOSITORY_URL = "https://gitlab.com/gitlab-com/content-sites/handbook.git"
REPOSITORY_COMMIT = "e0f0d82bc0e325fb798e0f95523c58f142c1459c"

CATEGORY_MAP = {
    "finance": "expense_and_finance",
    "legal": "legal_and_compliance",
    "people-group": "hr_and_employee_policies",
}


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            sha256.update(block)

    return sha256.hexdigest()


def extract_title(file_path):
    content = file_path.read_text(encoding="utf-8")

    match = re.search(
        r'^title:\s*["\']?(.*?)["\']?\s*$',
        content,
        re.MULTILINE,
    )

    return match.group(1).strip() if match else file_path.stem


def main():
    files = sorted(RAW_DIR.rglob("*.md"))

    if not files:
        raise RuntimeError(f"No Markdown files found in {RAW_DIR}")

    records = []
    seen_hashes = {}

    for index, file_path in enumerate(files, start=1):
        relative_path = file_path.relative_to(RAW_DIR)
        category_root = relative_path.parts[0]

        file_hash = calculate_sha256(file_path)

        if file_hash in seen_hashes:
            print(
                f"WARNING: Duplicate content: {relative_path} "
                f"matches {seen_hashes[file_hash]}"
            )
        else:
            seen_hashes[file_hash] = str(relative_path)

        record = {
            "document_id": f"GITLAB-{index:04d}",
            "title": extract_title(file_path),
            "organization": "GitLab",
            "source_url": ("https://handbook.gitlab.com/handbook/"+ relative_path.with_suffix("").as_posix()),
            "repository_url": REPOSITORY_URL,
            "repository_commit": REPOSITORY_COMMIT,
            "publication_date": None,
            "effective_date": None,
            "retrieved_date": date.today().isoformat(),
            "category": CATEGORY_MAP.get(category_root, "other"),
            "country": "Global",
            "document_type": "handbook_policy_or_guidance",
            "license": "CC-BY-SA-4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "file_hash": file_hash,
            "local_path": str(
                file_path.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "processing_status": "raw_collected_metadata_pending",
        }

        records.append(record)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Documents registered: {len(records)}")
    print(f"Unique content hashes: {len(seen_hashes)}")
    print(f"Manifest created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()