import re
import unicodedata

from src.rag.schemas import RetrievalResult


class PolicyContextSelector:
    """Apply conservative metadata-based policy selection."""

    POLICY_PATTERNS = {
        "sponsored_travel": (
            r"\bsponsored\s+(?:business\s+)?travel\b",
        ),
        "anti_corruption": (
            r"\banti\s*corruption\b",
            r"\banti\s*bribery\b",
        ),
        "anti_fraud": (
            r"\banti\s*fraud\b",
        ),
        "gifts_entertainment": (
            r"\bgifts?\s+(?:and|or)\s+entertainment\b",
        ),
        "global_travel_expense": (
            r"\bglobal\s+travel\s+and\s+expense\b",
        ),
    }

    COMPARISON_PATTERN = (
        r"\b(?:compare|comparison|difference|"
        r"different\s+from|versus|vs|both|"
        r"across\s+policies|multiple\s+policies)\b"
    )

    @staticmethod
    def normalize(value):

        if not value:
            return ""

        value = unicodedata.normalize(
            "NFKC", value
        ).casefold()

        value = value.replace("&", " and ")

        value = re.sub(
            r"[\W_]+", " ", value
        )

        return " ".join(value.split())

    @classmethod
    def detect_contexts(cls, value):

        normalized = cls.normalize(value)

        return [
            context
            for context, patterns
            in cls.POLICY_PATTERNS.items()
            if any(
                re.search(pattern, normalized)
                for pattern in patterns
            )
        ]

    @classmethod
    def is_comparison(cls, query):

        return bool(
            re.search(
                cls.COMPARISON_PATTERN,
                cls.normalize(query),
            )
        )

    @classmethod
    def select(
        cls,
        query: str,
        result: RetrievalResult,
    ) -> RetrievalResult:

        contexts = cls.detect_contexts(query)

        candidates = result.retrieved_chunks

        matching_document_ids = set()

        selected = list(candidates)

        mode = "general"

        if cls.is_comparison(query):

            mode = "comparison"

        elif len(contexts) > 1:

            mode = "multiple_contexts"

        elif len(contexts) == 1:

            requested_context = contexts[0]

            for item in candidates:

                title_contexts = cls.detect_contexts(
                    item.chunk.title
                )

                if requested_context in title_contexts:

                    matching_document_ids.add(
                        item.chunk.document_id
                    )

            if matching_document_ids:

                selected = [
                    item
                    for item in candidates
                    if item.chunk.document_id
                    in matching_document_ids
                ]

                mode = "matched"

            else:

                selected = []

                mode = "requested_context_missing"

        return result.model_copy(
            update={
                "retrieved_chunks": selected,
                "metadata": {
                    **result.metadata,
                    "policy_context_selector": {
                        "mode": mode,
                        "requested_contexts": contexts,
                        "matching_document_ids": sorted(
                            matching_document_ids
                        ),
                        "input_candidates": len(candidates),
                        "selected_candidates": len(selected),
                        "excluded_candidates": (
                            len(candidates) - len(selected)
                        ),
                    },
                },
            }
        )
