
import re
import unicodedata
import time
from typing import Any

from rank_bm25 import BM25Okapi

from src.rag.ingestion.document_loader import load_chunks
from src.rag.ingestion.document_processor import process_chunks
from src.rag.schemas import (
    RetrievedChunk,
    RetrievalResult,
)


class BM25Retriever:
    def __init__(
        self,
        chunk_file: str | None = None,
        include_metadata: bool = True,
    ):

        if type(include_metadata) is not bool:
            raise ValueError(
                "include_metadata must be a boolean."
            )

        self.include_metadata = include_metadata
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[3]

        self.chunk_file = (
            Path(chunk_file)
            if chunk_file
            else project_root
            / "data"
            / "real_corpus"
            / "chunks"
            / "gitlab_structure_aware_v4_500_overlap_100.jsonl"
        )

        self.chunks = process_chunks(load_chunks(self.chunk_file))

        if not self.chunks:
            raise ValueError("No chunks available for BM25 indexing.")

        self.tokenized_corpus = [
            self._tokenize(
                self.build_searchable_text(
                    chunk,
                    include_metadata=self.include_metadata,
                )
            )
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def build_searchable_text(
        chunk,
        include_metadata: bool = True,
    ) -> str:

        if not include_metadata:
            return chunk.text

        fields = [
            chunk.title,
            chunk.section_heading,
            chunk.section_path,
            chunk.text,
        ]

        unique_fields = []

        seen = set()

        for value in fields:

            if not value:
                continue

            normalized = " ".join(
                value.split()
            )

            if not normalized:
                continue

            key = unicodedata.normalize(
                "NFKC", normalized
            ).casefold()

            if key in seen:
                continue

            seen.add(key)

            unique_fields.append(normalized)

        return "\n".join(unique_fields)

    @staticmethod
    def _tokenize(text: str) -> list[str]:

        normalized = unicodedata.normalize(
            "NFKC", text
        ).casefold()

        normalized = normalized.replace(
            "&", " and "
        )

        return re.findall(
            r"\b\w+\b",
            normalized,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> RetrievalResult:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        start_time = time.perf_counter()

        query_tokens = self._tokenize(query)

        if not query_tokens:
            raise ValueError("Query contains no searchable tokens.")

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        retrieved_chunks = []

        for index in ranked_indices:
            chunk = self.chunks[index]

            if where:
                if any(
                    getattr(chunk, key, None) != value
                    for key, value in where.items()
                ):
                    continue

            score = float(scores[index])

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    rank=len(retrieved_chunks) + 1,
                    retrieval_method="bm25",
                )
            )

            if len(retrieved_chunks) >= top_k:
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            retrieval_method="bm25",
            elapsed_ms=elapsed_ms,
            metadata={
                "top_k": top_k,
                "returned_chunks": len(retrieved_chunks),
                "scoring_algorithm": "BM25Okapi",
                "include_metadata": self.include_metadata,
                "tokenization": "unicode_nfkc_casefold",
            },
        )
