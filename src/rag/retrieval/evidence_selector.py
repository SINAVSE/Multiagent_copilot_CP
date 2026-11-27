import hashlib
import unicodedata

from src.rag.schemas import (
    RetrievedChunk,
    RetrievalResult,
)


class EvidenceSelector:
    """Select unique evidence while preserving ranking."""

    @staticmethod
    def normalize_text(value: str) -> str:

        value = unicodedata.normalize(
            "NFC",
            value,
        )

        return " ".join(value.split())

    @classmethod
    def content_fingerprint(
        cls,
        item: RetrievedChunk,
    ) -> tuple:

        chunk = item.chunk

        normalized_text = cls.normalize_text(
            chunk.text
        )

        digest = hashlib.sha256(
            normalized_text.encode("utf-8")
        ).hexdigest()

        return (
            chunk.document_id,
            chunk.section_path,
            chunk.section_heading,
            digest,
        )

    @classmethod
    def select(
        cls,
        result: RetrievalResult,
        top_k: int = 5,
    ) -> RetrievalResult:

        if type(top_k) is not int or top_k < 1:

            raise ValueError(
                "top_k must be a positive integer."
            )

        selected = []

        seen_chunk_ids = {}

        seen_fingerprints = set()

        duplicate_ids = 0

        duplicate_content = 0

        empty_chunks = 0

        invalid_identifiers = 0

        for item in result.retrieved_chunks:

            chunk = item.chunk

            if (
                not chunk.chunk_id.strip()
                or not chunk.document_id.strip()
            ):

                invalid_identifiers += 1

                continue

            normalized_text = cls.normalize_text(
                chunk.text
            )

            if not normalized_text:

                empty_chunks += 1

                continue

            fingerprint = cls.content_fingerprint(
                item
            )

            if chunk.chunk_id in seen_chunk_ids:

                if (
                    seen_chunk_ids[chunk.chunk_id]
                    != chunk.model_dump()
                ):

                    raise ValueError(
                        "Conflicting evidence uses the "
                        "same chunk identifier."
                    )

                duplicate_ids += 1

                continue

            seen_chunk_ids[chunk.chunk_id] = (
                chunk.model_dump()
            )

            if fingerprint in seen_fingerprints:

                duplicate_content += 1

                continue

            seen_fingerprints.add(fingerprint)

            selected.append(
                item.model_copy(
                    update={
                        "rank": len(selected) + 1,
                    }
                )
            )

        unique_count = len(selected)

        selected = selected[:top_k]

        return result.model_copy(
            update={
                "retrieved_chunks": selected,
                "metadata": {
                    **result.metadata,
                    "evidence_selector": {
                        "input_candidates": len(
                            result.retrieved_chunks
                        ),
                        "unique_candidates": unique_count,
                        "selected_candidates": len(
                            selected
                        ),
                        "duplicate_ids_removed": (
                            duplicate_ids
                        ),
                        "duplicate_content_removed": (
                            duplicate_content
                        ),
                        "empty_chunks_removed": (
                            empty_chunks
                        ),
                        "invalid_identifiers_removed": (
                            invalid_identifiers
                        ),
                        "truncated_candidates": max(
                            0,
                            unique_count - top_k,
                        ),
                    },
                },
            }
        )
