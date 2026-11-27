from types import SimpleNamespace

import pytest

from src.rag.generation.answer_generator import (
    AnswerGenerator,
)

from tests.test_citation_integrity import (
    QUERY,
    make_generator,
    make_valid_response,
    make_ranked_candidates,
    make_policy_candidates,
)


def with_query(result, query):

    return result.model_copy(
        update={"query": query}
    )


def with_chunks(result, chunks):

    return result.model_copy(
        update={"retrieved_chunks": chunks}
    )


def make_response(chunk_id="CANDIDATE-1"):

    data = make_valid_response()

    data["cited_chunk_ids"] = [
        chunk_id
    ]

    return data


def ids(result):

    return [
        item.chunk.chunk_id
        for item in result.retrieved_chunks
    ]


def test_supplied_evidence_does_not_repeat_retrieval():

    generator, model = make_generator(
        make_response()
    )

    candidates = make_ranked_candidates()

    result = generator.generate(
        query=QUERY,
        retrieval_result=candidates,
    )

    assert result.citations[0].chunk_id == (
        "CANDIDATE-1"
    )

    generator.retriever.retrieve.assert_not_called()

    generator.reranker.rerank.assert_not_called()

    model.invoke.assert_called_once()


def test_supplied_evidence_top_k_enforced():

    generator, _ = make_generator()

    result = generator.prepare_evidence(
        QUERY,
        make_ranked_candidates(),
    )

    assert len(result.retrieved_chunks) == 5


def test_supplied_duplicate_identifiers_removed():

    generator, _ = make_generator()

    candidates = make_ranked_candidates()

    first = candidates.retrieved_chunks[0]

    candidates.retrieved_chunks.insert(
        1,
        first.model_copy()
    )

    result = generator.prepare_evidence(
        QUERY,
        candidates,
    )

    assert ids(result) == [
        "CANDIDATE-1",
        "CANDIDATE-2",
        "CANDIDATE-3",
        "CANDIDATE-4",
        "CANDIDATE-5",
    ]

    assert result.metadata[
        "evidence_selector"
    ]["duplicate_ids_removed"] == 1


def test_conflicting_supplied_identifiers_rejected():

    generator, model = make_generator()

    candidates = make_ranked_candidates()

    first = candidates.retrieved_chunks[0]

    conflicting = first.model_copy(
        update={
            "chunk": first.chunk.model_copy(
                update={
                    "text": "Different policy condition."
                }
            )
        }
    )

    candidates.retrieved_chunks.insert(
        1,
        conflicting,
    )

    with pytest.raises(
        ValueError,
        match="Conflicting evidence",
    ):

        generator.generate(
            query=QUERY,
            retrieval_result=candidates,
        )

    model.invoke.assert_not_called()


def test_supplied_blank_chunk_identifier_excluded():

    generator, _ = make_generator()

    candidates = make_ranked_candidates()

    first = candidates.retrieved_chunks[0]

    candidates.retrieved_chunks[0] = first.model_copy(
        update={
            "chunk": first.chunk.model_copy(
                update={"chunk_id": " "}
            )
        }
    )

    result = generator.prepare_evidence(
        QUERY,
        candidates,
    )

    assert " " not in ids(result)

    assert "CANDIDATE-1" not in ids(result)


def test_supplied_explicit_policy_context_enforced():

    generator, _ = make_generator()

    query = "What is the sponsored travel policy?"

    candidates = with_query(
        make_policy_candidates(),
        query,
    )

    result = generator.prepare_evidence(
        query,
        candidates,
    )

    assert len(result.retrieved_chunks) == 2

    assert all(
        item.chunk.title
        == "Sponsored Business Travel Policy"
        for item in result.retrieved_chunks
    )


def test_missing_supplied_policy_context_refused():

    generator, model = make_generator()

    query = "What is the anti-fraud policy?"

    candidates = with_query(
        make_ranked_candidates(),
        query,
    )

    result = generator.generate(
        query=query,
        retrieval_result=candidates,
    )

    assert result.insufficient_evidence is True

    assert result.citations == []

    model.invoke.assert_not_called()


def test_supplied_comparison_evidence_preserved():

    generator, _ = make_generator()

    query = (
        "Compare sponsored travel and general travel"
    )

    candidates = with_query(
        make_policy_candidates(),
        query,
    )

    result = generator.prepare_evidence(
        query,
        candidates,
    )

    titles = {
        item.chunk.title
        for item in result.retrieved_chunks
    }

    assert "Sponsored Business Travel Policy" in titles

    assert "General Travel Policy" in titles


def test_supplied_evidence_budget_enforced():

    generator, _ = make_generator()

    candidates = make_ranked_candidates()

    first = with_chunks(
        candidates,
        [candidates.retrieved_chunks[0]],
    )

    generator.max_context_bytes = len(
        generator._build_context(first).encode("utf-8")
    )

    result = generator.prepare_evidence(
        QUERY,
        candidates,
    )

    assert ids(result) == ["CANDIDATE-1"]

    assert len(
        generator._build_context(result).encode("utf-8")
    ) <= generator.max_context_bytes


def test_budget_excluded_supplied_citation_rejected():

    generator, model = make_generator(
        make_response("CANDIDATE-2")
    )

    candidates = make_ranked_candidates()

    first = with_chunks(
        candidates,
        [candidates.retrieved_chunks[0]],
    )

    generator.max_context_bytes = len(
        generator._build_context(first).encode("utf-8")
    )

    with pytest.raises(
        ValueError,
        match="not present in retrieved",
    ):

        generator.generate(
            query=QUERY,
            retrieval_result=candidates,
        )


def test_top_k_excluded_supplied_citation_rejected():

    generator, _ = make_generator(
        make_response("CANDIDATE-6")
    )

    with pytest.raises(
        ValueError,
        match="not present in retrieved",
    ):

        generator.generate(
            query=QUERY,
            retrieval_result=make_ranked_candidates(),
        )


def test_supplied_context_budget_exhaustion_refused():

    generator, model = make_generator()

    generator.max_context_bytes = 1

    result = generator.generate(
        query=QUERY,
        retrieval_result=make_ranked_candidates(),
    )

    assert result.insufficient_evidence is True

    assert result.citations == []

    assert result.retrieved_evidence == []

    model.invoke.assert_not_called()


def test_empty_supplied_evidence_refused():

    generator, model = make_generator()

    result = generator.generate(
        query=QUERY,
        retrieval_result=with_chunks(
            make_ranked_candidates(),
            [],
        ),
    )

    assert result.insufficient_evidence is True

    assert result.citations == []

    generator.retriever.retrieve.assert_not_called()

    model.invoke.assert_not_called()


def test_cross_query_evidence_rejected():

    generator, model = make_generator()

    candidates = with_query(
        make_ranked_candidates(),
        "What is the annual leave policy?",
    )

    with pytest.raises(
        ValueError,
        match="different query",
    ):

        generator.generate(
            query=QUERY,
            retrieval_result=candidates,
        )

    model.invoke.assert_not_called()


def test_query_whitespace_normalization():

    generator, _ = make_generator()

    candidates = with_query(
        make_ranked_candidates(),
        "  " + QUERY + "  ",
    )

    result = generator.prepare_evidence(
        QUERY,
        candidates,
    )

    assert len(result.retrieved_chunks) == 5


@pytest.mark.parametrize(
    "value",
    [
        {},
        [],
        "invalid",
        SimpleNamespace(retrieved_chunks=[]),
    ],
)
def test_invalid_supplied_result_type_rejected(value):

    generator, model = make_generator()

    with pytest.raises(
        ValueError,
        match="valid RetrievalResult",
    ):

        generator.generate(
            query=QUERY,
            retrieval_result=value,
        )

    model.invoke.assert_not_called()


def test_original_supplied_evidence_not_mutated():

    generator, _ = make_generator()

    original = make_ranked_candidates()

    before = original.model_dump()

    generator.prepare_evidence(
        QUERY,
        original,
    )

    assert original.model_dump() == before


def test_prepared_evidence_selection_is_idempotent():

    generator, _ = make_generator()

    first = generator.prepare_evidence(
        QUERY,
        make_ranked_candidates(),
    )

    second = generator.prepare_evidence(
        QUERY,
        first,
    )

    assert [
        item.model_dump()
        for item in first.retrieved_chunks
    ] == [
        item.model_dump()
        for item in second.retrieved_chunks
    ]


def test_existing_retrieval_metadata_preserved():

    generator, _ = make_generator()

    original = make_ranked_candidates().model_copy(
        update={
            "metadata": {
                "custom_metric": 123
            }
        }
    )

    result = generator.prepare_evidence(
        QUERY,
        original,
    )

    assert result.metadata["custom_metric"] == 123

    assert "policy_context_selector" in result.metadata

    assert "evidence_selector" in result.metadata

    assert "context_budget" in result.metadata


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "5"],
)
def test_invalid_top_k_rejected(value):

    generator, _ = make_generator()

    with pytest.raises(ValueError):

        AnswerGenerator(
            retriever=generator.retriever,
            reranker=generator.reranker,
            model=generator.model,
            top_k=value,
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 6.5, "10"],
)
def test_invalid_candidate_k_rejected(value):

    generator, _ = make_generator()

    with pytest.raises(ValueError):

        AnswerGenerator(
            retriever=generator.retriever,
            reranker=generator.reranker,
            model=generator.model,
            candidate_k=value,
        )
