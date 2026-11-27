
import re

from src.rag.schemas import DocumentChunk


def normalize_text(text: str) -> str:
    """
    Normalize whitespace without changing the meaning or structure
    of the chunk.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def process_chunk(chunk: DocumentChunk) -> DocumentChunk:
    """
    Normalize chunk text while preserving its metadata.
    """
    normalized_text = normalize_text(chunk.text)

    return chunk.model_copy(
        update={"text": normalized_text}
    )


def process_chunks(
    chunks: list[DocumentChunk],
) -> list[DocumentChunk]:
    """
    Process a collection of document chunks.
    """
    return [
        processed_chunk
        for chunk in chunks
        if (processed_chunk := process_chunk(chunk)).text
    ]