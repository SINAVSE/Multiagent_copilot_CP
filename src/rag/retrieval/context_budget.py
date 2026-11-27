from collections.abc import Callable

from src.rag.schemas import RetrievalResult


class EvidenceContextBudget:
    """Select complete evidence within a UTF-8 byte budget."""

    DEFAULT_MAX_CONTEXT_BYTES = 12000

    @classmethod
    def select(
        cls,
        result: RetrievalResult,
        context_builder: Callable[[RetrievalResult], str],
        max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
    ) -> RetrievalResult:

        if (
            type(max_context_bytes) is not int
            or max_context_bytes < 1
        ):

            raise ValueError(
                "max_context_bytes must be a positive integer."
            )

        selected = []

        excluded_chunk_ids = []

        used_bytes = 0

        original_context = context_builder(result)

        original_bytes = len(
            original_context.encode("utf-8")
        )

        for item in result.retrieved_chunks:

            candidate = item.model_copy(
                update={
                    "rank": len(selected) + 1
                }
            )

            proposed = selected + [candidate]

            proposed_result = result.model_copy(
                update={
                    "retrieved_chunks": proposed
                }
            )

            proposed_context = context_builder(
                proposed_result
            )

            proposed_bytes = len(
                proposed_context.encode("utf-8")
            )

            if proposed_bytes > max_context_bytes:

                excluded_chunk_ids.append(
                    item.chunk.chunk_id
                )

                continue

            selected.append(candidate)

            used_bytes = proposed_bytes

        return result.model_copy(
            update={
                "retrieved_chunks": selected,
                "metadata": {
                    **result.metadata,
                    "context_budget": {
                        "budget_unit": "utf8_bytes",
                        "max_context_bytes": max_context_bytes,
                        "original_context_bytes": original_bytes,
                        "selected_context_bytes": used_bytes,
                        "remaining_context_bytes": (
                            max_context_bytes - used_bytes
                        ),
                        "input_chunks": len(
                            result.retrieved_chunks
                        ),
                        "selected_chunks": len(selected),
                        "excluded_chunks": len(
                            excluded_chunk_ids
                        ),
                        "excluded_chunk_ids": excluded_chunk_ids,
                        "budget_limited": bool(
                            excluded_chunk_ids
                        ),
                        "selection_strategy": (
                            "rank_order_whole_chunks"
                        ),
                    },
                },
            }
        )
