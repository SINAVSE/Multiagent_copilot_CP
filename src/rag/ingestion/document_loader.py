import json
from pathlib import Path

from src.rag.schemas import DocumentChunk


def load_chunks(file_path: str | Path) -> list[DocumentChunk]:
    """
    Load JSONL chunk records and validate them using DocumentChunk.

    Each line in the JSONL file must contain one chunk record.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {file_path}")

    chunks: list[DocumentChunk] = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                chunk = DocumentChunk.model_validate(record)
                chunks.append(chunk)

            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid chunk record at line {line_number}: {exc}"
                ) from exc

    return chunks