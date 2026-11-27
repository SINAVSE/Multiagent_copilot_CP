import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.agents.errors import AgentResponseError

from src.rag.generation.answer_generator import (
    AnswerGenerator,
)

from src.rag.schemas import (
    DocumentChunk,
    RetrievedChunk,
    RetrievalResult,
)


QUERY = "What is the meal allowance?"


def make_chunk(
    chunk_id="TEST-001",
    document_id="DOC-001",
):

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Test Travel Policy",
        organization="Test Organization",
        source_url="https://example.invalid/policy",
        section_heading="Meal Allowance",
        section_path="Travel > Meals",
        text=(
            "TEST POLICY: Meal allowance is INR 1800 "
            "per employee per travel day."
        ),
    )


def make_retrieval_result():

    return RetrievalResult(
        query=QUERY,
        retrieval_method="hybrid_reranked",
        retrieved_chunks=[
            RetrievedChunk(
                chunk=make_chunk(),
                score=0.95,
                rank=1,
                retrieval_method="cross_encoder",
            ),
            RetrievedChunk(
                chunk=make_chunk(
                    "TEST-002",
                    "DOC-002",
                ),
                score=0.75,
                rank=2,
                retrieval_method="cross_encoder",
            ),
        ],
    )


def make_valid_response():

    return {
        "answer": (
            "The meal allowance is INR 1800 "
            "per employee per travel day."
        ),
        "insufficient_evidence": False,
        "missing_information": [],
        "cited_chunk_ids": ["TEST-001"],
    }


def make_generator(data=None, raw_content=None):

    model = Mock()

    if raw_content is None:

        raw_content = json.dumps(
            data if data is not None
            else make_valid_response()
        )

    model.invoke.return_value = SimpleNamespace(
        content=raw_content
    )

    generator = AnswerGenerator(
        retriever=Mock(),
        reranker=Mock(),
        model=model,
    )

    return generator, model


def generate(generator):

    return generator.generate(
        query=QUERY,
        retrieval_result=make_retrieval_result(),
    )


def test_valid_citation_is_accepted():

    generator, model = make_generator()

    result = generate(generator)

    assert result.insufficient_evidence is False

    assert len(result.citations) == 1

    assert result.citations[0].chunk_id == "TEST-001"

    model.invoke.assert_called_once()


def test_citation_metadata_comes_from_source():

    data = make_valid_response()

    data["source_url"] = (
        "https://example.invalid/invented"
    )

    generator, model = make_generator(data)

    result = generate(generator)

    citation = result.citations[0]

    assert citation.document_id == "DOC-001"

    assert citation.title == "Test Travel Policy"

    assert citation.source_url == (
        "https://example.invalid/policy"
    )

    assert citation.section_path == (
        "Travel > Meals"
    )


def test_multiple_valid_citations_are_accepted():

    data = make_valid_response()

    data["cited_chunk_ids"] = [
        "TEST-001",
        "TEST-002",
    ]

    generator, model = make_generator(data)

    result = generate(generator)

    assert [
        citation.chunk_id
        for citation in result.citations
    ] == [
        "TEST-001",
        "TEST-002",
    ]


def test_unknown_citation_is_rejected():

    data = make_valid_response()

    data["cited_chunk_ids"] = [
        "INVENTED-999"
    ]

    generator, model = make_generator(data)

    with pytest.raises(
        ValueError,
        match="not present in retrieved",
    ):
        generate(generator)


def test_mixed_valid_and_invalid_citations_are_rejected():

    data = make_valid_response()

    data["cited_chunk_ids"] = [
        "TEST-001",
        "INVENTED-999",
    ]

    generator, model = make_generator(data)

    with pytest.raises(ValueError):
        generate(generator)


def test_duplicate_citation_ids_are_rejected():

    data = make_valid_response()

    data["cited_chunk_ids"] = [
        "TEST-001",
        "TEST-001",
    ]

    generator, model = make_generator(data)

    with pytest.raises(
        ValueError,
        match="Duplicate citation",
    ):
        generate(generator)


@pytest.mark.parametrize(
    "citation",
    [123, None, True, {}],
)
def test_invalid_citation_types_are_rejected(
    citation,
):

    data = make_valid_response()

    data["cited_chunk_ids"] = [
        citation
    ]

    generator, model = make_generator(data)

    with pytest.raises(
        ValueError,
        match="Citation IDs must be strings",
    ):
        generate(generator)


def test_uncited_sufficient_answer_is_replaced():

    data = make_valid_response()

    data["answer"] = "Unsupported policy conclusion."

    data["cited_chunk_ids"] = []

    generator, model = make_generator(data)

    result = generate(generator)

    assert result.insufficient_evidence is True

    assert result.citations == []

    assert "Supporting policy citation" in (
        result.missing_information
    )

    assert "Unsupported policy conclusion" not in (
        result.answer
    )


def test_uncited_answer_preserves_missing_information():

    data = make_valid_response()

    data["missing_information"] = [
        "Expense category"
    ]

    data["cited_chunk_ids"] = []

    generator, model = make_generator(data)

    result = generate(generator)

    assert result.insufficient_evidence is True

    assert result.missing_information == [
        "Expense category",
        "Supporting policy citation",
    ]


def test_supported_partial_answer_is_preserved():

    data = make_valid_response()

    data["insufficient_evidence"] = True

    data["missing_information"] = [
        "Number of employees"
    ]

    generator, model = make_generator(data)

    result = generate(generator)

    assert result.insufficient_evidence is True

    assert len(result.citations) == 1

    assert result.answer == data["answer"]


def test_insufficient_evidence_without_citations():

    data = make_valid_response()

    data.update({
        "answer": "Insufficient policy evidence.",
        "insufficient_evidence": True,
        "cited_chunk_ids": [],
        "missing_information": [
            "Applicable policy"
        ],
    })

    generator, model = make_generator(data)

    result = generate(generator)

    assert result.insufficient_evidence is True

    assert result.citations == []

    assert result.answer == data["answer"]


@pytest.mark.parametrize(
    "answer",
    ["", " ", "\n\t"],
)
def test_empty_generated_answer_is_rejected(
    answer,
):

    data = make_valid_response()

    data["answer"] = answer

    generator, model = make_generator(data)

    with pytest.raises(
        ValueError,
        match="Answer cannot be empty",
    ):
        generate(generator)


@pytest.mark.parametrize(
    "missing_information",
    [
        [123],
        [None],
        [True],
        [""],
        [" "],
    ],
)
def test_invalid_missing_information_is_rejected(
    missing_information,
):

    data = make_valid_response()

    data["missing_information"] = (
        missing_information
    )

    generator, model = make_generator(data)

    with pytest.raises(
        ValueError,
        match="non-empty strings",
    ):
        generate(generator)


@pytest.mark.parametrize(
    "field",
    [
        "answer",
        "insufficient_evidence",
        "missing_information",
        "cited_chunk_ids",
    ],
)
def test_missing_required_response_fields(field):

    data = make_valid_response()

    del data[field]

    generator, model = make_generator(data)

    with pytest.raises(
        ValueError,
        match="missing required JSON fields",
    ):
        generate(generator)


def test_malformed_json_is_rejected():

    generator, model = make_generator(
        raw_content="Invalid JSON"
    )

    with pytest.raises(AgentResponseError):
        generate(generator)


def test_duplicate_json_keys_are_rejected():

    generator, model = make_generator(
        raw_content=(
            '{"answer":"First answer",'
            '"answer":"Second answer",'
            '"insufficient_evidence":false,'
            '"missing_information":[],'
            '"cited_chunk_ids":["TEST-001"]}'
        )
    )

    with pytest.raises(AgentResponseError):
        generate(generator)


def test_fenced_json_is_accepted():

    fence = chr(96) * 3

    raw_content = (
        fence + "json\n"
        + json.dumps(make_valid_response())
        + "\n" + fence
    )

    generator, model = make_generator(
        raw_content=raw_content
    )

    result = generate(generator)

    assert result.insufficient_evidence is False

    assert len(result.citations) == 1


def test_no_retrieved_evidence_bypasses_model():

    generator, model = make_generator()

    retrieval_result = RetrievalResult(
        query=QUERY,
        retrieval_method="hybrid",
        retrieved_chunks=[],
    )

    result = generator.generate(
        query=QUERY,
        retrieval_result=retrieval_result,
    )

    assert result.insufficient_evidence is True

    assert result.citations == []

    model.invoke.assert_not_called()


def test_supplied_evidence_avoids_duplicate_retrieval():

    generator, model = make_generator()

    generate(generator)

    generator.retriever.retrieve.assert_not_called()

    generator.reranker.rerank.assert_not_called()


def test_prompt_requires_supporting_citations():

    generator, model = make_generator()

    generate(generator)

    prompt = model.invoke.call_args.args[0]

    assert (
        "Every sufficient-evidence answer must include"
        in prompt
    )

    assert "[CHUNK_ID: TEST-001]" in prompt

    assert "[CHUNK_ID: TEST-002]" in prompt


# --------------------------------------------------
# CAP-13: Evidence selection integration
# --------------------------------------------------


def make_ranked_candidates(count=10):

    items = []

    for index in range(1, count + 1):

        chunk = make_chunk(
            chunk_id=f"CANDIDATE-{index}"
        )

        chunk = chunk.model_copy(
            update={
                "text": (
                    f"Distinct policy condition {index}."
                )
            }
        )

        items.append(
            RetrievedChunk(
                chunk=chunk,
                score=1.0 - (index / 100),
                rank=index,
                retrieval_method="cross_encoder",
            )
        )

    return RetrievalResult(
        query=QUERY,
        retrieval_method="hybrid_reranked",
        retrieved_chunks=items,
    )


def test_answer_generator_reranks_complete_candidate_set():

    generator, model = make_generator()

    candidates = make_ranked_candidates()

    generator.retriever.retrieve.return_value = (
        candidates
    )

    generator.reranker.rerank.return_value = (
        candidates
    )

    result = generator.retrieve(QUERY)

    assert (
        generator.reranker.rerank.call_args.kwargs[
            "top_k"
        ] == 10
    )

    assert len(result.retrieved_chunks) == 5

    model.invoke.assert_not_called()


def test_duplicates_do_not_occupy_final_evidence_slots():

    generator, model = make_generator()

    candidates = make_ranked_candidates()

    first = candidates.retrieved_chunks[0]

    candidates.retrieved_chunks.insert(
        1,
        first.model_copy(
            update={"rank": 2}
        )
    )

    generator.retriever.retrieve.return_value = (
        candidates
    )

    generator.reranker.rerank.return_value = (
        candidates
    )

    result = generator.retrieve(QUERY)

    assert [
        item.chunk.chunk_id
        for item in result.retrieved_chunks
    ] == [
        "CANDIDATE-1",
        "CANDIDATE-2",
        "CANDIDATE-3",
        "CANDIDATE-4",
        "CANDIDATE-5",
    ]

    assert result.metadata[
        "evidence_selector"
    ]["duplicate_ids_removed"] == 1


def test_lower_ranked_policy_evidence_is_considered():

    generator, model = make_generator()

    candidates = make_ranked_candidates()

    item = candidates.retrieved_chunks[6]

    candidates.retrieved_chunks[6] = item.model_copy(
        update={
            "chunk": item.chunk.model_copy(
                update={
                    "document_id": "CUSTOM-POLICY-901",
                    "title": "Sponsored Business Travel Policy",
                }
            )
        }
    )

    # CAP-13 query consistency fixture
    candidates = candidates.model_copy(
        update={
            'query': 'What is the sponsored business travel policy?',
        }
    )

    generator.retriever.retrieve.return_value = (
        candidates
    )

    generator.reranker.rerank.return_value = (
        candidates
    )

    result = generator.retrieve(
        "What is the sponsored business travel policy?"
    )

    assert len(result.retrieved_chunks) == 1

    assert result.retrieved_chunks[0].chunk.chunk_id == (
        "CANDIDATE-7"
    )

    assert result.retrieved_chunks[0].rank == 1


def test_evidence_selection_preserves_original_scores():

    generator, model = make_generator()

    candidates = make_ranked_candidates()

    generator.retriever.retrieve.return_value = (
        candidates
    )

    generator.reranker.rerank.return_value = (
        candidates
    )

    result = generator.retrieve(QUERY)

    assert [
        item.score
        for item in result.retrieved_chunks
    ] == [
        item.score
        for item in candidates.retrieved_chunks[:5]
    ]


def test_only_selected_evidence_is_sent_to_model():

    data = make_valid_response()

    data["cited_chunk_ids"] = [
        "CANDIDATE-1"
    ]

    generator, model = make_generator(data)

    candidates = make_ranked_candidates()

    generator.retriever.retrieve.return_value = (
        candidates
    )

    generator.reranker.rerank.return_value = (
        candidates
    )

    result = generator.generate(
        query=QUERY
    )

    prompt = model.invoke.call_args.args[0]

    assert "[CHUNK_ID: CANDIDATE-1]" in prompt

    assert "[CHUNK_ID: CANDIDATE-5]" in prompt

    assert "[CHUNK_ID: CANDIDATE-6]" not in prompt

    assert len(result.retrieved_evidence) == 5


def test_discarded_candidate_cannot_be_cited():

    data = make_valid_response()

    data["cited_chunk_ids"] = [
        "CANDIDATE-6"
    ]

    generator, model = make_generator(data)

    candidates = make_ranked_candidates()

    generator.retriever.retrieve.return_value = (
        candidates
    )

    generator.reranker.rerank.return_value = (
        candidates
    )

    with pytest.raises(
        ValueError,
        match="not present in retrieved",
    ):

        generator.generate(
            query=QUERY
        )


# --------------------------------------------------
# CAP-13: Generic policy context integration
# --------------------------------------------------


def make_policy_candidates():

    candidates = make_ranked_candidates()

    titles = [
        "General Travel Policy",
        "General Expense Policy",
        "Sponsored Business Travel Policy",
        "Anti-Fraud Policy",
        "Anti-Corruption Policy",
    ]

    for index, item in enumerate(
        candidates.retrieved_chunks
    ):

        title = titles[index % len(titles)]

        candidates.retrieved_chunks[index] = (
            item.model_copy(
                update={
                    "chunk": item.chunk.model_copy(
                        update={
                            "document_id": (
                                "CUSTOM-DOC-"
                                + str(index % len(titles))
                            ),
                            "title": title,
                        }
                    )
                }
            )
        )

    return candidates


def test_explicit_policy_context_selects_matching_metadata():

    generator, model = make_generator()

    candidates = make_policy_candidates()

    # CAP-13 query consistency fixture
    candidates = candidates.model_copy(
        update={
            'query': 'What is the anti-fraud policy?',
        }
    )

    generator.retriever.retrieve.return_value = (
        candidates
    )

    generator.reranker.rerank.return_value = (
        candidates
    )

    result = generator.retrieve(
        "What is the anti-fraud policy?"
    )

    assert len(result.retrieved_chunks) == 2

    assert all(
        item.chunk.title == "Anti-Fraud Policy"
        for item in result.retrieved_chunks
    )

    assert result.metadata[
        "policy_context_selector"
    ]["mode"] == "matched"

    model.invoke.assert_not_called()


def test_missing_policy_context_bypasses_model():

    generator, model = make_generator()

    candidates = make_ranked_candidates()

    # CAP-13 query consistency fixture
    candidates = candidates.model_copy(
        update={
            'query': 'What is the sponsored travel policy?',
        }
    )

    generator.retriever.retrieve.return_value = (
        candidates
    )

    generator.reranker.rerank.return_value = (
        candidates
    )

    result = generator.generate(
        query="What is the sponsored travel policy?"
    )

    assert result.insufficient_evidence is True

    assert result.citations == []

    assert result.retrieved_evidence == []

    model.invoke.assert_not_called()


def test_comparison_query_preserves_cross_policy_evidence():

    generator, model = make_generator()

    candidates = make_policy_candidates()

    # CAP-13 query consistency fixture
    candidates = candidates.model_copy(
        update={
            'query': 'Compare sponsored travel and general travel',
        }
    )

    generator.retriever.retrieve.return_value = (
        candidates
    )

    generator.reranker.rerank.return_value = (
        candidates
    )

    result = generator.retrieve(
        "Compare sponsored travel and general travel"
    )

    titles = {
        item.chunk.title
        for item in result.retrieved_chunks
    }

    assert "Sponsored Business Travel Policy" in titles

    assert "General Travel Policy" in titles

    assert result.metadata[
        "policy_context_selector"
    ]["mode"] == "comparison"


def test_unrelated_policy_citation_is_rejected():

    data = make_valid_response()

    data["cited_chunk_ids"] = [
        "CANDIDATE-1"
    ]

    generator, model = make_generator(data)

    candidates = make_policy_candidates()

    # CAP-13 query consistency fixture
    candidates = candidates.model_copy(
        update={
            'query': 'What is the sponsored travel policy?',
        }
    )

    generator.retriever.retrieve.return_value = (
        candidates
    )

    generator.reranker.rerank.return_value = (
        candidates
    )

    with pytest.raises(
        ValueError,
        match="not present in retrieved",
    ):

        generator.generate(
            query="What is the sponsored travel policy?"
        )


# --------------------------------------------------
# CAP-13: Evidence context budget integration
# --------------------------------------------------


def configure_budgeted_generator(
    data=None,
    budget=1000,
    candidates=None,
):

    generator, model = make_generator(data)

    generator.max_context_bytes = budget

    if candidates is None:

        candidates = make_ranked_candidates()

    generator.retriever.retrieve.return_value = (
        candidates
    )

    generator.reranker.rerank.return_value = (
        candidates
    )

    return generator, model


def test_retrieval_enforces_context_budget():

    generator, model = configure_budgeted_generator(
        budget=1000
    )

    result = generator.retrieve(QUERY)

    context = generator._build_context(result)

    assert len(context.encode("utf-8")) <= 1000

    assert result.metadata[
        "context_budget"
    ]["budget_limited"] is True

    model.invoke.assert_not_called()


def test_context_budget_preserves_selection_metadata():

    generator, _ = configure_budgeted_generator()

    result = generator.retrieve(QUERY)

    assert "evidence_selector" in result.metadata

    assert "policy_context_selector" in result.metadata

    assert "context_budget" in result.metadata


def test_excluded_context_not_sent_to_model():

    data = make_valid_response()

    data["cited_chunk_ids"] = [
        "CANDIDATE-1"
    ]

    candidates = make_ranked_candidates()

    first = candidates.model_copy(
        update={
            "retrieved_chunks": [
                candidates.retrieved_chunks[0]
            ]
        }
    )

    budget = len(
        AnswerGenerator._build_context(
            first
        ).encode("utf-8")
    )

    generator, model = configure_budgeted_generator(
        data=data,
        budget=budget,
        candidates=candidates,
    )

    result = generator.generate(
        query=QUERY
    )

    prompt = model.invoke.call_args.args[0]

    assert "[CHUNK_ID: CANDIDATE-1]" in prompt

    assert "[CHUNK_ID: CANDIDATE-2]" not in prompt

    assert len(result.retrieved_evidence) == 1


def test_context_budget_excluded_citation_rejected():

    data = make_valid_response()

    data["cited_chunk_ids"] = [
        "CANDIDATE-2"
    ]

    candidates = make_ranked_candidates()

    first = candidates.model_copy(
        update={
            "retrieved_chunks": [
                candidates.retrieved_chunks[0]
            ]
        }
    )

    budget = len(
        AnswerGenerator._build_context(
            first
        ).encode("utf-8")
    )

    generator, model = configure_budgeted_generator(
        data=data,
        budget=budget,
        candidates=candidates,
    )

    with pytest.raises(
        ValueError,
        match="not present in retrieved",
    ):

        generator.generate(
            query=QUERY
        )


def test_context_budget_exhaustion_bypasses_model():

    generator, model = configure_budgeted_generator(
        budget=1
    )

    result = generator.generate(
        query=QUERY
    )

    assert result.insufficient_evidence is True

    assert result.citations == []

    assert result.retrieved_evidence == []

    model.invoke.assert_not_called()


def test_default_context_budget_configuration():

    generator, model = make_generator()

    assert generator.max_context_bytes == 12000


def test_budgeted_retrieval_can_be_reused():

    generator, model = configure_budgeted_generator(
        data={
            **make_valid_response(),
            "cited_chunk_ids": ["CANDIDATE-1"],
        }
    )

    selected = generator.retrieve(QUERY)

    generator.generate(
        query=QUERY,
        retrieval_result=selected,
    )

    generator.retriever.retrieve.assert_called_once()

    generator.reranker.rerank.assert_called_once()
