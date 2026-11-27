import pytest

from src.rag.generation.answer_generator import AnswerGenerator

from src.rag.retrieval.context_budget import (
    EvidenceContextBudget,
)

from tests.test_evidence_selector import (
    make_item,
    make_result,
)


def make_candidate(chunk_id, text, rank=1):

    return make_item(
        chunk_id=chunk_id,
        text=text,
        rank=rank,
    )


def context_bytes(result):

    return len(
        AnswerGenerator._build_context(
            result
        ).encode("utf-8")
    )


def select(result, budget=12000):

    return EvidenceContextBudget.select(
        result=result,
        context_builder=AnswerGenerator._build_context,
        max_context_bytes=budget,
    )


def metadata(result):

    return result.metadata["context_budget"]


def chunk_ids(result):

    return [
        item.chunk.chunk_id
        for item in result.retrieved_chunks
    ]


def test_empty_evidence():

    result = select(make_result([]))

    assert result.retrieved_chunks == []

    assert metadata(result)["selected_context_bytes"] == 0


@pytest.mark.parametrize(
    "budget",
    [0, -1, True, 1.5, "12000", None],
)
def test_invalid_budget_rejected(budget):

    with pytest.raises(ValueError):

        select(
            make_result([]),
            budget=budget,
        )


def test_all_evidence_fits():

    original = make_result([
        make_candidate("A", "Meal allowance is USD 100."),
        make_candidate("B", "Manager approval required."),
    ])

    result = select(original)

    assert chunk_ids(result) == ["A", "B"]

    assert metadata(result)["budget_limited"] is False


def test_exact_budget_boundary():

    original = make_result([
        make_candidate("A", "Meal allowance is USD 100."),
    ])

    limit = context_bytes(original)

    result = select(
        original,
        budget=limit,
    )

    assert chunk_ids(result) == ["A"]

    assert metadata(result)["remaining_context_bytes"] == 0


def test_one_byte_below_required_budget():

    original = make_result([
        make_candidate("A", "Meal allowance is USD 100."),
    ])

    limit = context_bytes(original) - 1

    result = select(
        original,
        budget=limit,
    )

    assert result.retrieved_chunks == []

    assert metadata(result)["budget_limited"] is True


def test_oversized_chunk_is_not_truncated():

    original = make_result([
        make_candidate("A", "Policy condition. " * 5000),
    ])

    result = select(
        original,
        budget=1000,
    )

    assert result.retrieved_chunks == []

    assert metadata(result)["excluded_chunk_ids"] == ["A"]


def test_smaller_candidate_selected_after_large_candidate():

    original = make_result([
        make_candidate("A", "Large policy text. " * 5000),
        make_candidate("B", "Approval required."),
    ])

    result = select(
        original,
        budget=1000,
    )

    assert chunk_ids(result) == ["B"]


def test_rank_order_priority():

    first = make_candidate(
        "A",
        "First policy condition.",
        rank=1,
    )

    second = make_candidate(
        "B",
        "Second policy condition.",
        rank=2,
    )

    budget = context_bytes(
        make_result([first])
    )

    result = select(
        make_result([first, second]),
        budget=budget,
    )

    assert chunk_ids(result) == ["A"]


def test_combined_context_limit():

    original = make_result([
        make_candidate(
            str(index),
            "Policy condition. " * 20,
            rank=index,
        )
        for index in range(1, 6)
    ])

    result = select(
        original,
        budget=1500,
    )

    assert context_bytes(result) <= 1500

    assert metadata(result)["budget_limited"] is True


def test_policy_text_preserved():

    text = (
        "Meal limit: USD 100.\n"
        "Exception: prior written approval is required."
    )

    original = make_result([
        make_candidate("A", text),
    ])

    result = select(original)

    assert result.retrieved_chunks[0].chunk.text == text


def test_citation_identity_preserved():

    original = make_result([
        make_candidate("A", "Policy text."),
    ])

    result = select(original)

    assert (
        result.retrieved_chunks[0].chunk.model_dump()
        == original.retrieved_chunks[0].chunk.model_dump()
    )


def test_metadata_and_delimiters_counted():

    original = make_result([
        make_candidate("A", "Short policy."),
        make_candidate("B", "Another policy."),
    ])

    result = select(original)

    body_bytes = sum(
        len(item.chunk.text.encode("utf-8"))
        for item in result.retrieved_chunks
    )

    assert context_bytes(result) > body_bytes

    assert (
        metadata(result)["selected_context_bytes"]
        == context_bytes(result)
    )


def test_unicode_byte_accounting():

    original = make_result([
        make_candidate(
            "A",
            "Meal allowance: \u20b91000.",
        ),
    ])

    text = AnswerGenerator._build_context(
        original
    )

    assert len(text.encode("utf-8")) > len(text)

    result = select(original)

    assert (
        metadata(result)["selected_context_bytes"]
        == len(text.encode("utf-8"))
    )


def test_unicode_boundary_rejected():

    original = make_result([
        make_candidate("A", "\u20b9" * 100),
    ])

    character_count = len(
        AnswerGenerator._build_context(original)
    )

    result = select(
        original,
        budget=character_count,
    )

    assert result.retrieved_chunks == []


def test_ranks_reassigned_after_exclusion():

    original = make_result([
        make_candidate("A", "Large text. " * 5000, rank=1),
        make_candidate("B", "Policy B.", rank=2),
        make_candidate("C", "Policy C.", rank=3),
    ])

    result = select(
        original,
        budget=1500,
    )

    assert chunk_ids(result) == ["B", "C"]

    assert [
        item.rank for item in result.retrieved_chunks
    ] == [1, 2]


def test_original_scores_preserved():

    original = make_result([
        make_candidate("A", "Policy A."),
        make_candidate("B", "Policy B."),
    ])

    result = select(original)

    assert [
        item.score for item in result.retrieved_chunks
    ] == [
        item.score for item in original.retrieved_chunks
    ]


def test_original_result_not_modified():

    original = make_result([
        make_candidate("A", "Policy A."),
        make_candidate("B", "Large text. " * 5000),
    ])

    before = original.model_dump()

    select(
        original,
        budget=1000,
    )

    assert original.model_dump() == before


def test_existing_metadata_preserved():

    result = select(make_result([]))

    assert result.metadata[
        "original_metadata"
    ] == "preserved"


def test_budget_metadata_consistency():

    original = make_result([
        make_candidate("A", "Policy A."),
        make_candidate("B", "Large text. " * 5000),
    ])

    result = select(
        original,
        budget=1000,
    )

    info = metadata(result)

    assert info["budget_unit"] == "utf8_bytes"

    assert info["input_chunks"] == 2

    assert info["selected_chunks"] == 1

    assert info["excluded_chunks"] == 1

    assert info["excluded_chunk_ids"] == ["B"]

    assert info["original_context_bytes"] == (
        context_bytes(original)
    )

    assert info["selected_context_bytes"] == (
        context_bytes(result)
    )

    assert (
        info["selected_context_bytes"]
        + info["remaining_context_bytes"]
        == 1000
    )


def test_default_budget():

    result = select(make_result([]))

    assert metadata(result)["max_context_bytes"] == 12000


def test_all_candidates_exceed_budget():

    original = make_result([
        make_candidate("A", "Large text A. " * 5000),
        make_candidate("B", "Large text B. " * 5000),
    ])

    result = select(
        original,
        budget=100,
    )

    assert result.retrieved_chunks == []

    assert metadata(result)["excluded_chunk_ids"] == [
        "A", "B"
    ]

    assert metadata(result)["selected_context_bytes"] == 0


def test_repeatable_selection():

    original = make_result([
        make_candidate("A", "Policy A."),
        make_candidate("B", "Policy B."),
        make_candidate("C", "Policy C."),
    ])

    first = select(original, budget=700)

    second = select(original, budget=700)

    assert first.model_dump() == second.model_dump()
