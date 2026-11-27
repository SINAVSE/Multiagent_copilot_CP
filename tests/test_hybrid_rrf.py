from unittest.mock import Mock

import pytest

from src.rag.retrieval.hybrid_retriever import (
    HybridRetriever,
)

from src.rag.schemas import RetrievalResult

from tests.test_evidence_selector import make_item


QUERY = "What is the travel policy?"


def make_candidate(
    chunk_id,
    text=None,
):

    return make_item(
        chunk_id=chunk_id,
        text=text or f"Policy condition for {chunk_id}.",
    )


def make_retrieval_result(
    method,
    candidates,
):

    return RetrievalResult(
        query=QUERY,
        retrieval_method=method,
        retrieved_chunks=candidates,
    )


def create_retriever(
    dense_candidates=None,
    bm25_candidates=None,
    rrf_k=60,
):

    dense = Mock()

    bm25 = Mock()

    dense.retrieve.return_value = (
        make_retrieval_result(
            "dense",
            dense_candidates or [],
        )
    )

    bm25.retrieve.return_value = (
        make_retrieval_result(
            "bm25",
            bm25_candidates or [],
        )
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        rrf_k=rrf_k,
    )

    return retriever, dense, bm25


def get_scores(result):

    return {
        item.chunk.chunk_id: item.score
        for item in result.retrieved_chunks
    }


def get_ids(result):

    return [
        item.chunk.chunk_id
        for item in result.retrieved_chunks
    ]


def test_empty_retrieval_results():

    retriever, _, _ = create_retriever()

    result = retriever.retrieve(QUERY)

    assert result.retrieved_chunks == []

    assert result.metadata["unique_candidates"] == 0


def test_dense_only_contribution():

    retriever, _, _ = create_retriever(
        dense_candidates=[
            make_candidate("A"),
        ]
    )

    result = retriever.retrieve(QUERY)

    assert get_scores(result)["A"] == pytest.approx(
        1 / 61
    )


def test_bm25_only_contribution():

    retriever, _, _ = create_retriever(
        bm25_candidates=[
            make_candidate("A"),
        ]
    )

    result = retriever.retrieve(QUERY)

    assert get_scores(result)["A"] == pytest.approx(
        1 / 61
    )


def test_cross_retriever_contributions_combined():

    retriever, _, _ = create_retriever(
        dense_candidates=[make_candidate("A")],
        bm25_candidates=[make_candidate("A")],
    )

    result = retriever.retrieve(QUERY)

    assert get_scores(result)["A"] == pytest.approx(
        2 / 61
    )

    assert len(result.retrieved_chunks) == 1


def test_dense_duplicate_does_not_inflate_score():

    candidate = make_candidate("A")

    retriever, _, _ = create_retriever(
        dense_candidates=[
            candidate,
            candidate,
        ]
    )

    result = retriever.retrieve(QUERY)

    assert get_scores(result)["A"] == pytest.approx(
        1 / 61
    )

    assert result.metadata[
        "dense_duplicates_removed"
    ] == 1


def test_bm25_duplicate_does_not_inflate_score():

    candidate = make_candidate("A")

    retriever, _, _ = create_retriever(
        bm25_candidates=[
            candidate,
            candidate,
        ]
    )

    result = retriever.retrieve(QUERY)

    assert get_scores(result)["A"] == pytest.approx(
        1 / 61
    )

    assert result.metadata[
        "bm25_duplicates_removed"
    ] == 1


def test_duplicates_in_both_retrievers():

    candidate = make_candidate("A")

    retriever, _, _ = create_retriever(
        dense_candidates=[
            candidate,
            candidate,
            candidate,
        ],
        bm25_candidates=[
            candidate,
            candidate,
        ],
    )

    result = retriever.retrieve(QUERY)

    assert get_scores(result)["A"] == pytest.approx(
        2 / 61
    )

    assert result.metadata[
        "dense_duplicates_removed"
    ] == 2

    assert result.metadata[
        "bm25_duplicates_removed"
    ] == 1


def test_duplicate_removal_preserves_unique_ranks():

    retriever, _, _ = create_retriever(
        dense_candidates=[
            make_candidate("A"),
            make_candidate("A"),
            make_candidate("B"),
        ],
    )

    result = retriever.retrieve(QUERY)

    scores = get_scores(result)

    assert scores["A"] == pytest.approx(1 / 61)
    assert scores["B"] == pytest.approx(1 / 62)


def test_correct_rrf_score_for_distinct_ranks():

    retriever, _, _ = create_retriever(
        dense_candidates=[
            make_candidate("A"),
            make_candidate("B"),
        ],
        bm25_candidates=[
            make_candidate("B"),
            make_candidate("C"),
            make_candidate("A"),
        ],
    )

    result = retriever.retrieve(QUERY)

    scores = get_scores(result)

    assert scores["A"] == pytest.approx(
        1 / 61 + 1 / 63
    )

    assert scores["B"] == pytest.approx(
        1 / 62 + 1 / 61
    )

    assert scores["C"] == pytest.approx(
        1 / 62
    )

    assert get_ids(result) == ["B", "A", "C"]


def test_cross_retriever_agreement_ranks_higher():

    retriever, _, _ = create_retriever(
        dense_candidates=[
            make_candidate("A"),
            make_candidate("B"),
        ],
        bm25_candidates=[
            make_candidate("C"),
            make_candidate("B"),
        ],
    )

    result = retriever.retrieve(QUERY)

    assert get_ids(result)[0] == "B"


def test_conflicting_dense_identifiers_rejected():

    retriever, _, _ = create_retriever(
        dense_candidates=[
            make_candidate("A", "Allowance is USD 100."),
            make_candidate("A", "Allowance is USD 200."),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Conflicting evidence",
    ):

        retriever.retrieve(QUERY)


def test_conflicting_bm25_identifiers_rejected():

    retriever, _, _ = create_retriever(
        bm25_candidates=[
            make_candidate("A", "Approval required."),
            make_candidate("A", "Approval not required."),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Conflicting evidence",
    ):

        retriever.retrieve(QUERY)


def test_conflicting_cross_retriever_identifiers_rejected():

    retriever, _, _ = create_retriever(
        dense_candidates=[
            make_candidate("A", "Allowance is USD 100."),
        ],
        bm25_candidates=[
            make_candidate("A", "Allowance is USD 200."),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Conflicting evidence",
    ):

        retriever.retrieve(QUERY)


@pytest.mark.parametrize(
    "chunk_id,document_id",
    [
        ("", "DOC-001"),
        (" ", "DOC-001"),
        ("A", ""),
        ("A", " "),
    ],
)
def test_invalid_identifiers_excluded(
    chunk_id,
    document_id,
):

    candidate = make_item(
        chunk_id=chunk_id,
        document_id=document_id,
    )

    retriever, _, _ = create_retriever(
        dense_candidates=[
            candidate,
            make_candidate("VALID"),
        ],
    )

    result = retriever.retrieve(QUERY)

    assert get_ids(result) == ["VALID"]

    assert get_scores(result)["VALID"] == pytest.approx(
        1 / 61
    )

    assert result.metadata[
        "dense_invalid_identifiers_removed"
    ] == 1


def test_top_k_limit():

    retriever, _, _ = create_retriever(
        dense_candidates=[
            make_candidate("A"),
            make_candidate("B"),
            make_candidate("C"),
        ]
    )

    result = retriever.retrieve(
        QUERY,
        top_k=2,
    )

    assert get_ids(result) == ["A", "B"]


def test_output_ranks_are_sequential():

    retriever, _, _ = create_retriever(
        dense_candidates=[
            make_candidate("A"),
            make_candidate("A"),
            make_candidate("B"),
        ]
    )

    result = retriever.retrieve(QUERY)

    assert [
        item.rank
        for item in result.retrieved_chunks
    ] == [1, 2]


def test_custom_rrf_constant():

    retriever, _, _ = create_retriever(
        dense_candidates=[
            make_candidate("A"),
        ],
        rrf_k=20,
    )

    result = retriever.retrieve(QUERY)

    assert get_scores(result)["A"] == pytest.approx(
        1 / 21
    )


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "60"],
)
def test_invalid_rrf_constant(value):

    with pytest.raises(ValueError):

        create_retriever(
            rrf_k=value
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "5"],
)
def test_invalid_top_k(value):

    retriever, dense, bm25 = create_retriever()

    with pytest.raises(ValueError):

        retriever.retrieve(
            QUERY,
            top_k=value,
        )

    dense.retrieve.assert_not_called()
    bm25.retrieve.assert_not_called()


@pytest.mark.parametrize(
    "value",
    [0, 2, True, 6.5, "10"],
)
def test_invalid_candidate_k(value):

    retriever, dense, bm25 = create_retriever()

    with pytest.raises(ValueError):

        retriever.retrieve(
            QUERY,
            top_k=5,
            candidate_k=value,
        )

    dense.retrieve.assert_not_called()
    bm25.retrieve.assert_not_called()


def test_metadata_filter_forwarded():

    retriever, dense, bm25 = create_retriever()

    where = {
        "document_id": "DOC-001"
    }

    retriever.retrieve(
        QUERY,
        top_k=3,
        candidate_k=12,
        where=where,
    )

    for component in [dense, bm25]:

        component.retrieve.assert_called_once_with(
            query=QUERY,
            top_k=12,
            where=where,
        )


def test_default_candidate_count():

    retriever, dense, bm25 = create_retriever()

    retriever.retrieve(
        QUERY,
        top_k=7,
    )

    assert dense.retrieve.call_args.kwargs[
        "top_k"
    ] == 14

    assert bm25.retrieve.call_args.kwargs[
        "top_k"
    ] == 14


def test_default_minimum_candidate_count():

    retriever, dense, bm25 = create_retriever()

    retriever.retrieve(
        QUERY,
        top_k=2,
    )

    assert dense.retrieve.call_args.kwargs[
        "top_k"
    ] == 10

    assert bm25.retrieve.call_args.kwargs[
        "top_k"
    ] == 10


def test_unique_candidate_metrics():

    retriever, _, _ = create_retriever(
        dense_candidates=[
            make_candidate("A"),
            make_candidate("A"),
            make_candidate("B"),
        ],
        bm25_candidates=[
            make_candidate("B"),
            make_candidate("C"),
        ],
    )

    result = retriever.retrieve(QUERY)

    metadata = result.metadata

    assert metadata["dense_candidates"] == 3

    assert metadata["dense_unique_candidates"] == 2

    assert metadata["bm25_candidates"] == 2

    assert metadata["bm25_unique_candidates"] == 2

    assert metadata["unique_candidates"] == 3


def test_fusion_rank_metadata():

    retriever, _, _ = create_retriever(
        dense_candidates=[
            make_candidate("A"),
            make_candidate("B"),
        ],
        bm25_candidates=[
            make_candidate("B"),
        ],
    )

    result = retriever.retrieve(QUERY)

    fusion = result.metadata["rank_fusion"]

    assert fusion["A"]["dense_rank"] == 1
    assert fusion["A"]["bm25_rank"] is None

    assert fusion["B"]["dense_rank"] == 2
    assert fusion["B"]["bm25_rank"] == 1

    assert fusion["B"]["rrf_score"] == pytest.approx(
        1 / 62 + 1 / 61
    )


def test_original_chunk_metadata_preserved():

    candidate = make_candidate("A")

    retriever, _, _ = create_retriever(
        dense_candidates=[candidate],
    )

    result = retriever.retrieve(QUERY)

    assert (
        result.retrieved_chunks[0].chunk.model_dump()
        == candidate.chunk.model_dump()
    )


def test_tied_scores_have_repeatable_order():

    retriever, _, _ = create_retriever(
        dense_candidates=[make_candidate("A")],
        bm25_candidates=[make_candidate("B")],
    )

    first = retriever.retrieve(QUERY)

    second = retriever.retrieve(QUERY)

    assert get_ids(first) == get_ids(second)


def test_empty_query_rejected_before_retrieval():

    retriever, dense, bm25 = create_retriever()

    with pytest.raises(ValueError):

        retriever.retrieve(" ")

    dense.retrieve.assert_not_called()
    bm25.retrieve.assert_not_called()


# --------------------------------------------------
# CAP-13: Dense/BM25 metadata compatibility
# --------------------------------------------------


def update_chunk(candidate, **updates):

    return candidate.model_copy(
        update={
            "chunk": candidate.chunk.model_copy(
                update=updates
            )
        }
    )


def test_dense_and_bm25_optional_metadata_merged():

    candidate = make_candidate("A")

    dense = update_chunk(
        candidate,
        page_number=None,
        chunk_size=None,
        overlap=None,
    )

    bm25 = update_chunk(
        candidate,
        page_number=5,
        chunk_size=500,
        overlap=100,
    )

    retriever, _, _ = create_retriever(
        dense_candidates=[dense],
        bm25_candidates=[bm25],
    )

    result = retriever.retrieve(QUERY)

    chunk = result.retrieved_chunks[0].chunk

    assert chunk.page_number == 5

    assert chunk.chunk_size == 500

    assert chunk.overlap == 100

    assert get_scores(result)["A"] == pytest.approx(
        2 / 61
    )


def test_metadata_enrichment_does_not_inflate_rrf():

    candidate = make_candidate("A")

    first = update_chunk(
        candidate,
        page_number=None,
    )

    second = update_chunk(
        candidate,
        page_number=7,
    )

    retriever, _, _ = create_retriever(
        dense_candidates=[first, second],
    )

    result = retriever.retrieve(QUERY)

    assert get_scores(result)["A"] == pytest.approx(
        1 / 61
    )

    assert (
        result.retrieved_chunks[0].chunk.page_number
        == 7
    )

    assert result.metadata[
        "dense_duplicates_removed"
    ] == 1


def test_conflicting_optional_metadata_rejected():

    candidate = make_candidate("A")

    dense = update_chunk(
        candidate,
        page_number=3,
    )

    bm25 = update_chunk(
        candidate,
        page_number=5,
    )

    retriever, _, _ = create_retriever(
        dense_candidates=[dense],
        bm25_candidates=[bm25],
    )

    with pytest.raises(
        ValueError,
        match="Conflicting evidence",
    ):

        retriever.retrieve(QUERY)


def test_conflicting_organization_rejected():

    candidate = make_candidate("A")

    dense = update_chunk(
        candidate,
        organization="Organization A",
    )

    bm25 = update_chunk(
        candidate,
        organization="Organization B",
    )

    retriever, _, _ = create_retriever(
        dense_candidates=[dense],
        bm25_candidates=[bm25],
    )

    with pytest.raises(
        ValueError,
        match="Conflicting evidence",
    ):

        retriever.retrieve(QUERY)


def test_additional_metadata_survives_rank_fusion():

    candidate = make_candidate("A")

    dense = update_chunk(
        candidate,
        page_number=2,
    )

    bm25 = update_chunk(
        candidate,
        page_number=None,
        category="Travel",
        version="1.0",
    )

    retriever, _, _ = create_retriever(
        dense_candidates=[dense],
        bm25_candidates=[bm25],
    )

    result = retriever.retrieve(QUERY)

    chunk = result.retrieved_chunks[0].chunk

    assert chunk.page_number == 2

    assert chunk.category == "Travel"

    assert chunk.version == "1.0"


def test_metadata_merge_does_not_mutate_inputs():

    candidate = make_candidate("A")

    dense = update_chunk(
        candidate,
        page_number=None,
    )

    bm25 = update_chunk(
        candidate,
        page_number=5,
    )

    dense_before = dense.model_dump()

    bm25_before = bm25.model_dump()

    retriever, _, _ = create_retriever(
        dense_candidates=[dense],
        bm25_candidates=[bm25],
    )

    retriever.retrieve(QUERY)

    assert dense.model_dump() == dense_before

    assert bm25.model_dump() == bm25_before
