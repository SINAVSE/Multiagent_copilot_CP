from unittest.mock import patch

import numpy as np
import pytest

from src.rag.retrieval.reranker import (
    CrossEncoderReranker,
)

from src.rag.schemas import RetrievalResult

from tests.test_evidence_selector import make_item


QUERY = "What is the sponsored travel meal allowance?"


def make_candidate(
    chunk_id="DOC-001-S1",
    title="Sponsored Business Travel Policy",
    heading="Meal Reimbursement",
    section="Travel > Meals",
    text="Employees receive the applicable allowance.",
):

    item = make_item(
        chunk_id=chunk_id,
        document_id=chunk_id.split("-S")[0],
        text=text,
        heading=heading,
        section=section,
    )

    return item.model_copy(
        update={
            "chunk": item.chunk.model_copy(
                update={"title": title}
            )
        }
    )


def make_candidates():

    return [
        make_candidate(),

        make_candidate(
            chunk_id="DOC-002-S1",
            title="Equipment Policy",
            heading="Hardware Expenses",
            section="Equipment",
            text="Employees may purchase approved equipment.",
        ),

        make_candidate(
            chunk_id="DOC-003-S1",
            title="Annual Leave Policy",
            heading="Annual Leave",
            section="Leave",
            text="Employees receive annual leave.",
        ),
    ]


def make_result(candidates=None):

    return RetrievalResult(
        query=QUERY,
        retrieval_method="hybrid",
        retrieved_chunks=(
            make_candidates()
            if candidates is None
            else candidates
        ),
        elapsed_ms=10.0,
        metadata={
            "original_metadata": "preserved",
        },
    )


def create_reranker(
    scores=None,
    include_metadata=True,
    max_length=512,
):

    with patch(
        "src.rag.retrieval.reranker.CrossEncoder"
    ) as factory:

        model = factory.return_value

        model.predict.return_value = (
            [0.9, 0.3, 0.1]
            if scores is None
            else scores
        )

        reranker = CrossEncoderReranker(
            include_metadata=include_metadata,
            max_length=max_length,
        )

    return reranker, model


def test_passage_includes_document_title():

    chunk = make_candidate().chunk

    passage = CrossEncoderReranker.build_passage(
        chunk
    )

    assert "Document: " + chunk.title in passage


def test_passage_includes_section_heading():

    chunk = make_candidate().chunk

    passage = CrossEncoderReranker.build_passage(
        chunk
    )

    assert (
        "Section: " + chunk.section_heading
        in passage
    )


def test_passage_includes_section_path():

    chunk = make_candidate().chunk

    passage = CrossEncoderReranker.build_passage(
        chunk
    )

    assert (
        "Section path: " + chunk.section_path
        in passage
    )


def test_passage_preserves_original_policy_text():

    chunk = make_candidate(
        text="Limit: USD 100.\nApproval required."
    ).chunk

    passage = CrossEncoderReranker.build_passage(
        chunk
    )

    assert passage.endswith(chunk.text)


def test_text_only_passage():

    chunk = make_candidate().chunk

    passage = CrossEncoderReranker.build_passage(
        chunk,
        include_metadata=False,
    )

    assert passage == chunk.text


def test_duplicate_metadata_removed():

    chunk = make_candidate(
        title="Travel Policy",
        heading="TRAVEL POLICY",
        section="Travel  Policy",
    ).chunk

    passage = CrossEncoderReranker.build_passage(
        chunk
    )

    assert passage.casefold().count(
        "travel policy"
    ) == 1


def test_optional_metadata_ignored():

    chunk = make_candidate(
        heading=None,
        section=None,
    ).chunk

    passage = CrossEncoderReranker.build_passage(
        chunk
    )

    assert "None" not in passage

    assert chunk.text in passage


def test_blank_metadata_ignored():

    chunk = make_candidate(
        title=" ",
        heading=" ",
        section=" ",
    ).chunk

    passage = CrossEncoderReranker.build_passage(
        chunk
    )

    assert passage == chunk.text


def test_metadata_aware_model_input():

    reranker, model = create_reranker()

    reranker.rerank(
        QUERY,
        make_result(),
    )

    pairs = model.predict.call_args.args[0]

    assert len(pairs) == 3

    assert pairs[0][0] == QUERY

    assert "Sponsored Business Travel Policy" in pairs[0][1]

    assert "Meal Reimbursement" in pairs[0][1]


def test_text_only_model_input():

    reranker, model = create_reranker(
        include_metadata=False
    )

    original = make_result()

    reranker.rerank(
        QUERY,
        original,
    )

    pairs = model.predict.call_args.args[0]

    assert pairs[0][1] == (
        original.retrieved_chunks[0].chunk.text
    )


def test_correct_score_sorting():

    reranker, _ = create_reranker(
        scores=[0.1, 0.9, 0.5],
    )

    result = reranker.rerank(
        QUERY,
        make_result(),
    )

    assert [
        item.chunk.chunk_id
        for item in result.retrieved_chunks
    ] == [
        "DOC-002-S1",
        "DOC-003-S1",
        "DOC-001-S1",
    ]


def test_negative_scores_are_valid():

    reranker, _ = create_reranker(
        scores=[-3.0, -1.0, -2.0],
    )

    result = reranker.rerank(
        QUERY,
        make_result(),
    )

    assert [
        item.score
        for item in result.retrieved_chunks
    ] == [-1.0, -2.0, -3.0]


def test_zero_scores_are_valid():

    reranker, _ = create_reranker(
        scores=[0.0, 0.0, 0.0],
    )

    result = reranker.rerank(
        QUERY,
        make_result(),
    )

    assert len(result.retrieved_chunks) == 3


def test_numpy_scores_supported():

    reranker, _ = create_reranker(
        scores=np.array(
            [0.1, 0.9, 0.5],
            dtype=np.float32,
        ),
    )

    result = reranker.rerank(
        QUERY,
        make_result(),
    )

    assert result.retrieved_chunks[0].score == (
        pytest.approx(0.9)
    )


@pytest.mark.parametrize(
    "invalid_score",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_scores_rejected(
    invalid_score,
):

    reranker, _ = create_reranker(
        scores=[0.9, invalid_score, 0.1],
    )

    with pytest.raises(
        ValueError,
        match="non-finite score",
    ):

        reranker.rerank(
            QUERY,
            make_result(),
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        "0.8",
        None,
        True,
        False,
        [],
        {},
        np.array([0.8]),
    ],
)
def test_non_numeric_scores_rejected(
    invalid_score,
):

    reranker, _ = create_reranker(
        scores=[0.9, invalid_score, 0.1],
    )

    with pytest.raises(
        ValueError,
        match="non-numeric score",
    ):

        reranker.rerank(
            QUERY,
            make_result(),
        )


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [0.5],
        [0.1, 0.2, 0.3, 0.4],
    ],
)
def test_invalid_score_count(scores):

    reranker, _ = create_reranker(
        scores=scores
    )

    with pytest.raises(
        ValueError,
        match="unexpected number of scores",
    ):

        reranker.rerank(
            QUERY,
            make_result(),
        )


def test_empty_candidates_bypass_model():

    reranker, model = create_reranker()

    result = reranker.rerank(
        QUERY,
        make_result([]),
    )

    assert result.retrieved_chunks == []

    assert result.metadata[
        "reranked_candidates"
    ] == 0

    model.predict.assert_not_called()


def test_top_k_applied_after_ranking():

    reranker, _ = create_reranker(
        scores=[0.1, 0.3, 0.9],
    )

    result = reranker.rerank(
        QUERY,
        make_result(),
        top_k=1,
    )

    assert len(result.retrieved_chunks) == 1

    assert result.retrieved_chunks[0].chunk.chunk_id == (
        "DOC-003-S1"
    )


def test_all_candidates_returned_without_top_k():

    reranker, _ = create_reranker()

    result = reranker.rerank(
        QUERY,
        make_result(),
    )

    assert len(result.retrieved_chunks) == 3


def test_output_ranks_are_sequential():

    reranker, _ = create_reranker()

    result = reranker.rerank(
        QUERY,
        make_result(),
    )

    assert [
        item.rank
        for item in result.retrieved_chunks
    ] == [1, 2, 3]


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "5"],
)
def test_invalid_top_k_rejected(value):

    reranker, model = create_reranker()

    with pytest.raises(ValueError):

        reranker.rerank(
            QUERY,
            make_result(),
            top_k=value,
        )

    model.predict.assert_not_called()


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "512"],
)
def test_invalid_max_length_rejected(value):

    with patch(
        "src.rag.retrieval.reranker.CrossEncoder"
    ) as factory:

        with pytest.raises(ValueError):

            CrossEncoderReranker(
                max_length=value
            )

        factory.assert_not_called()


@pytest.mark.parametrize(
    "value",
    [None, 1, "true", []],
)
def test_invalid_metadata_configuration(value):

    with patch(
        "src.rag.retrieval.reranker.CrossEncoder"
    ) as factory:

        with pytest.raises(ValueError):

            CrossEncoderReranker(
                include_metadata=value
            )

        factory.assert_not_called()


def test_metadata_configuration_recorded():

    reranker, _ = create_reranker(
        include_metadata=False,
        max_length=256,
    )

    result = reranker.rerank(
        QUERY,
        make_result(),
    )

    assert result.metadata[
        "reranker_include_metadata"
    ] is False

    assert result.metadata[
        "reranker_max_length"
    ] == 256


def test_empty_result_records_configuration():

    reranker, _ = create_reranker()

    result = reranker.rerank(
        QUERY,
        make_result([]),
    )

    assert result.metadata[
        "reranker_include_metadata"
    ] is True

    assert result.metadata[
        "reranker_max_length"
    ] == 512


def test_existing_metadata_preserved():

    reranker, _ = create_reranker()

    result = reranker.rerank(
        QUERY,
        make_result(),
    )

    assert result.metadata[
        "original_metadata"
    ] == "preserved"


def test_original_chunk_content_preserved():

    reranker, _ = create_reranker()

    original = make_result()

    result = reranker.rerank(
        QUERY,
        original,
    )

    assert (
        result.retrieved_chunks[0].chunk.model_dump()
        == original.retrieved_chunks[0].chunk.model_dump()
    )


def test_original_retrieval_result_not_modified():

    reranker, _ = create_reranker(
        scores=[0.1, 0.9, 0.5]
    )

    original = make_result()

    before = original.model_dump()

    reranker.rerank(
        QUERY,
        original,
    )

    assert original.model_dump() == before


def test_elapsed_time_is_preserved_and_extended():

    reranker, _ = create_reranker()

    result = reranker.rerank(
        QUERY,
        make_result(),
    )

    assert result.elapsed_ms >= 10.0

    assert result.metadata[
        "rerank_elapsed_ms"
    ] >= 0


def test_equal_scores_preserve_candidate_order():

    reranker, _ = create_reranker(
        scores=[0.5, 0.5, 0.5]
    )

    original = make_result()

    result = reranker.rerank(
        QUERY,
        original,
    )

    assert [
        item.chunk.chunk_id
        for item in result.retrieved_chunks
    ] == [
        item.chunk.chunk_id
        for item in original.retrieved_chunks
    ]


def test_empty_query_rejected():

    reranker, model = create_reranker()

    with pytest.raises(ValueError):

        reranker.rerank(
            " ",
            make_result(),
        )

    model.predict.assert_not_called()


def test_reranked_candidate_count_recorded():

    reranker, _ = create_reranker()

    result = reranker.rerank(
        QUERY,
        make_result(),
        top_k=1,
    )

    assert result.metadata[
        "reranked_candidates"
    ] == 3


def test_model_failure_is_not_silently_ignored():

    reranker, model = create_reranker()

    model.predict.side_effect = RuntimeError(
        "Model execution failed."
    )

    with pytest.raises(
        RuntimeError,
        match="Model execution failed",
    ):

        reranker.rerank(
            QUERY,
            make_result(),
        )
