import pytest

from src.rag.schemas import (
    DocumentChunk,
    RetrievedChunk,
    RetrievalResult,
)

from src.rag.retrieval.evidence_selector import (
    EvidenceSelector,
)


def make_item(
    chunk_id="TEST-001",
    document_id="DOC-001",
    text="Meal allowance is USD 120 per day.",
    section="Travel > Meals",
    heading="Meals",
    rank=1,
    score=0.95,
):

    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            title="Test Travel Policy",
            organization="Test Organization",
            source_url="https://example.invalid/policy",
            section_path=section,
            section_heading=heading,
            text=text,
        ),
        rank=rank,
        score=score,
        retrieval_method="cross_encoder",
    )


def make_result(items):

    return RetrievalResult(
        query="What is the meal allowance?",
        retrieval_method="hybrid_reranked",
        retrieved_chunks=items,
        elapsed_ms=25.0,
        metadata={
            "original_metadata": "preserved"
        },
    )


def ids(result):

    return [
        item.chunk.chunk_id
        for item in result.retrieved_chunks
    ]


def metrics(result):

    return result.metadata["evidence_selector"]


def test_empty_result():

    result = EvidenceSelector.select(
        make_result([])
    )

    assert result.retrieved_chunks == []

    assert metrics(result)["input_candidates"] == 0


def test_single_chunk_is_preserved():

    result = EvidenceSelector.select(
        make_result([make_item()])
    )

    assert ids(result) == ["TEST-001"]

    assert metrics(result)["selected_candidates"] == 1


@pytest.mark.parametrize(
    "top_k",
    [0, -1, True, 1.5, "5"],
)
def test_invalid_top_k(top_k):

    with pytest.raises(ValueError):

        EvidenceSelector.select(
            make_result([]),
            top_k=top_k,
        )


def test_duplicate_chunk_id_removed():

    result = EvidenceSelector.select(
        make_result([
            make_item(),
            make_item(rank=2, score=0.80),
        ])
    )

    assert ids(result) == ["TEST-001"]

    assert metrics(result)[
        "duplicate_ids_removed"
    ] == 1


def test_highest_ranked_duplicate_is_preserved():

    result = EvidenceSelector.select(
        make_result([
            make_item(score=0.95),
            make_item(score=0.50, rank=2),
        ])
    )

    assert result.retrieved_chunks[0].score == 0.95


def test_conflicting_duplicate_identifier_rejected():

    with pytest.raises(
        ValueError,
        match="Conflicting evidence",
    ):

        EvidenceSelector.select(
            make_result([
                make_item(),
                make_item(
                    text="Meal allowance is USD 200."
                ),
            ])
        )


def test_same_id_different_document_rejected():

    with pytest.raises(ValueError):

        EvidenceSelector.select(
            make_result([
                make_item(),
                make_item(
                    document_id="DOC-002"
                ),
            ])
        )


def test_same_id_different_section_rejected():

    with pytest.raises(ValueError):

        EvidenceSelector.select(
            make_result([
                make_item(),
                make_item(
                    section="Travel > Hotel"
                ),
            ])
        )


def test_identical_content_same_section_removed():

    result = EvidenceSelector.select(
        make_result([
            make_item(),
            make_item(
                chunk_id="TEST-002",
                rank=2,
            ),
        ])
    )

    assert ids(result) == ["TEST-001"]

    assert metrics(result)[
        "duplicate_content_removed"
    ] == 1


def test_whitespace_normalization():

    result = EvidenceSelector.select(
        make_result([
            make_item(),
            make_item(
                chunk_id="TEST-002",
                text=(
                    "Meal  allowance is\n"
                    "USD 120 per day."
                ),
            ),
        ])
    )

    assert ids(result) == ["TEST-001"]


def test_unicode_normalization():

    composed = "Caf\u00e9 expenses."

    decomposed = "Cafe\u0301 expenses."

    assert composed != decomposed

    assert (
        EvidenceSelector.normalize_text(composed)
        == EvidenceSelector.normalize_text(decomposed)
    )

    result = EvidenceSelector.select(
        make_result([
            make_item(text=composed),
            make_item(
                chunk_id="TEST-002",
                text=decomposed,
            ),
        ])
    )

    assert ids(result) == ["TEST-001"]

    assert metrics(result)[
        "duplicate_content_removed"
    ] == 1


def test_identical_content_different_documents_preserved():

    result = EvidenceSelector.select(
        make_result([
            make_item(),
            make_item(
                chunk_id="TEST-002",
                document_id="DOC-002",
            ),
        ])
    )

    assert ids(result) == [
        "TEST-001",
        "TEST-002",
    ]


def test_identical_content_different_sections_preserved():

    result = EvidenceSelector.select(
        make_result([
            make_item(),
            make_item(
                chunk_id="TEST-002",
                section="Travel > Exceptions",
            ),
        ])
    )

    assert len(result.retrieved_chunks) == 2


def test_identical_content_different_headings_preserved():

    result = EvidenceSelector.select(
        make_result([
            make_item(),
            make_item(
                chunk_id="TEST-002",
                heading="International Meals",
            ),
        ])
    )

    assert len(result.retrieved_chunks) == 2


def test_distinct_policy_amounts_preserved():

    result = EvidenceSelector.select(
        make_result([
            make_item(
                text="Meal allowance is USD 120."
            ),
            make_item(
                chunk_id="TEST-002",
                text="Meal allowance is USD 80."
            ),
        ])
    )

    assert len(result.retrieved_chunks) == 2


def test_distinct_approval_conditions_preserved():

    result = EvidenceSelector.select(
        make_result([
            make_item(
                text="Manager approval is required."
            ),
            make_item(
                chunk_id="TEST-002",
                text="Manager approval is not required."
            ),
        ])
    )

    assert len(result.retrieved_chunks) == 2


def test_case_sensitive_policy_terms_preserved():

    result = EvidenceSelector.select(
        make_result([
            make_item(text="Approval is REQUIRED."),
            make_item(
                chunk_id="TEST-002",
                text="Approval is required.",
            ),
        ])
    )

    assert len(result.retrieved_chunks) == 2


def test_blank_content_removed():

    result = EvidenceSelector.select(
        make_result([
            make_item(text=" \n\t "),
            make_item(
                chunk_id="TEST-002"
            ),
        ])
    )

    assert ids(result) == ["TEST-002"]

    assert metrics(result)[
        "empty_chunks_removed"
    ] == 1


@pytest.mark.parametrize(
    "chunk_id,document_id",
    [
        ("", "DOC-001"),
        (" ", "DOC-001"),
        ("TEST-001", ""),
        ("TEST-001", " "),
    ],
)
def test_invalid_identifiers_removed(
    chunk_id,
    document_id,
):

    result = EvidenceSelector.select(
        make_result([
            make_item(
                chunk_id=chunk_id,
                document_id=document_id,
            )
        ])
    )

    assert result.retrieved_chunks == []

    assert metrics(result)[
        "invalid_identifiers_removed"
    ] == 1


def test_top_k_applied_after_duplicate_removal():

    result = EvidenceSelector.select(
        make_result([
            make_item(),
            make_item(),
            make_item(
                chunk_id="TEST-002",
                text="Receipts are required."
            ),
            make_item(
                chunk_id="TEST-003",
                text="Approval is required."
            ),
        ]),
        top_k=3,
    )

    assert ids(result) == [
        "TEST-001",
        "TEST-002",
        "TEST-003",
    ]


def test_top_k_truncation():

    items = [
        make_item(
            chunk_id=f"TEST-{i}",
            text=f"Policy condition {i}",
            rank=i,
        )
        for i in range(1, 8)
    ]

    result = EvidenceSelector.select(
        make_result(items),
        top_k=5,
    )

    assert len(result.retrieved_chunks) == 5

    assert metrics(result)[
        "truncated_candidates"
    ] == 2


def test_ranks_are_sequential_after_selection():

    result = EvidenceSelector.select(
        make_result([
            make_item(rank=2),
            make_item(rank=3),
            make_item(
                chunk_id="TEST-002",
                text="Receipts are required.",
                rank=7,
            ),
        ])
    )

    assert [
        item.rank
        for item in result.retrieved_chunks
    ] == [1, 2]


def test_existing_metadata_is_preserved():

    result = EvidenceSelector.select(
        make_result([make_item()])
    )

    assert result.metadata[
        "original_metadata"
    ] == "preserved"

    assert result.elapsed_ms == 25.0

    assert result.retrieval_method == (
        "hybrid_reranked"
    )


def test_original_retrieval_result_is_not_modified():

    original = make_result([
        make_item(rank=2),
        make_item(rank=3),
    ])

    result = EvidenceSelector.select(original)

    assert len(original.retrieved_chunks) == 2

    assert original.retrieved_chunks[0].rank == 2

    assert "evidence_selector" not in original.metadata

    assert len(result.retrieved_chunks) == 1


def test_selection_metrics_are_consistent():

    result = EvidenceSelector.select(
        make_result([
            make_item(),
            make_item(),
            make_item(
                chunk_id="TEST-002"
            ),
            make_item(
                chunk_id="TEST-003",
                text="Approval is required.",
            ),
            make_item(
                chunk_id="TEST-004",
                text=" ",
            ),
        ]),
        top_k=1,
    )

    counts = metrics(result)

    assert counts["input_candidates"] == 5
    assert counts["unique_candidates"] == 2
    assert counts["selected_candidates"] == 1
    assert counts["duplicate_ids_removed"] == 1
    assert counts["duplicate_content_removed"] == 1
    assert counts["empty_chunks_removed"] == 1
    assert counts["truncated_candidates"] == 1
