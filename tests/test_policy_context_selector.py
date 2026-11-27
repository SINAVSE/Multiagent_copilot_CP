import pytest

from src.rag.retrieval.policy_context_selector import (
    PolicyContextSelector,
)

from tests.test_evidence_selector import (
    make_item,
    make_result,
)


def policy_item(
    document_id,
    title,
    chunk_id=None,
    text="Test policy condition.",
    rank=1,
):

    item = make_item(
        document_id=document_id,
        chunk_id=chunk_id or document_id + "-S1",
        text=text,
        rank=rank,
    )

    return item.model_copy(
        update={
            "chunk": item.chunk.model_copy(
                update={"title": title}
            )
        }
    )


def document_ids(result):

    return [
        item.chunk.document_id
        for item in result.retrieved_chunks
    ]


def selection_metadata(result):

    return result.metadata["policy_context_selector"]


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Sponsored business travel receipts", "sponsored_travel"),
        ("Sponsored-travel per diem", "sponsored_travel"),
        ("SPONSORED TRAVEL POLICY", "sponsored_travel"),
        ("Anti-corruption policy", "anti_corruption"),
        ("Anti bribery policy", "anti_corruption"),
        ("Anti-fraud policy", "anti_fraud"),
        ("Gifts and Entertainment", "gifts_entertainment"),
        ("Gifts & Entertainment", "gifts_entertainment"),
        ("Global Travel and Expense Policy", "global_travel_expense"),
    ],
)
def test_explicit_context_detection(query, expected):

    assert PolicyContextSelector.detect_contexts(
        query
    ) == [expected]


@pytest.mark.parametrize(
    "query",
    [
        "Can I claim USD 100 for dinner?",
        "What is the daily meal allowance?",
        "What is the annual leave policy?",
        "What are the hotel reimbursement limits?",
    ],
)
def test_ambiguous_or_general_queries(query):

    assert PolicyContextSelector.detect_contexts(
        query
    ) == []


@pytest.mark.parametrize(
    "query",
    [
        "Compare sponsored travel and general travel",
        "Sponsored travel versus normal travel",
        "Sponsored travel vs. normal travel",
        "What is the difference between both policies?",
        "Show rules across policies",
        "Compare multiple policies",
    ],
)
def test_comparison_detection(query):

    assert PolicyContextSelector.is_comparison(
        query
    ) is True


def test_metadata_matching_without_fixed_document_id():

    result = PolicyContextSelector.select(
        "Sponsored business travel meal policy",
        make_result([
            policy_item("CUSTOM-100", "General Travel Policy"),
            policy_item("CUSTOM-200", "Sponsored Business Travel Policy"),
        ]),
    )

    assert document_ids(result) == ["CUSTOM-200"]

    assert selection_metadata(result)["mode"] == "matched"


def test_hyphenated_policy_title():

    result = PolicyContextSelector.select(
        "Sponsored travel receipts",
        make_result([
            policy_item("DOC-01", "Sponsored-Travel Policy"),
        ]),
    )

    assert document_ids(result) == ["DOC-01"]


def test_multiple_matching_documents_preserved():

    result = PolicyContextSelector.select(
        "Anti-fraud policy",
        make_result([
            policy_item("ORG-A-01", "Anti-Fraud Policy"),
            policy_item("ORG-B-01", "Corporate Anti-Fraud Policy"),
            policy_item("ORG-C-01", "Expense Policy"),
        ]),
    )

    assert document_ids(result) == [
        "ORG-A-01", "ORG-B-01"
    ]


def test_all_chunks_of_matching_document_preserved():

    result = PolicyContextSelector.select(
        "Sponsored travel policy",
        make_result([
            policy_item(
                "DOC-01",
                "Sponsored Business Travel Policy",
                chunk_id="DOC-01-S1",
            ),
            policy_item(
                "DOC-01",
                "Sponsored Business Travel Policy",
                chunk_id="DOC-01-S2",
            ),
            policy_item("DOC-02", "General Expense Policy"),
        ]),
    )

    assert document_ids(result) == [
        "DOC-01", "DOC-01"
    ]


def test_missing_explicit_context_returns_no_evidence():

    result = PolicyContextSelector.select(
        "Sponsored travel meal allowance",
        make_result([
            policy_item("DOC-01", "General Travel Policy"),
        ]),
    )

    assert result.retrieved_chunks == []

    assert selection_metadata(result)["mode"] == (
        "requested_context_missing"
    )


def test_general_query_preserves_candidates():

    original = make_result([
        policy_item("DOC-01", "General Expense Policy"),
        policy_item("DOC-02", "Travel Policy"),
    ])

    result = PolicyContextSelector.select(
        "Can I claim USD 100 for dinner?",
        original,
    )

    assert document_ids(result) == [
        "DOC-01", "DOC-02"
    ]

    assert selection_metadata(result)["mode"] == "general"


def test_comparison_preserves_multiple_policies():

    result = PolicyContextSelector.select(
        "Compare sponsored travel and general travel",
        make_result([
            policy_item("DOC-01", "Sponsored Travel Policy"),
            policy_item("DOC-02", "General Travel Policy"),
        ]),
    )

    assert document_ids(result) == [
        "DOC-01", "DOC-02"
    ]

    assert selection_metadata(result)["mode"] == (
        "comparison"
    )


def test_multiple_explicit_contexts_preserved():

    result = PolicyContextSelector.select(
        "Explain anti-fraud and anti-corruption policies",
        make_result([
            policy_item("DOC-01", "Anti-Fraud Policy"),
            policy_item("DOC-02", "Anti-Corruption Policy"),
        ]),
    )

    assert document_ids(result) == [
        "DOC-01", "DOC-02"
    ]

    assert selection_metadata(result)["mode"] == (
        "multiple_contexts"
    )


def test_empty_candidates():

    result = PolicyContextSelector.select(
        "Sponsored travel policy",
        make_result([]),
    )

    assert result.retrieved_chunks == []

    assert selection_metadata(result)["mode"] == (
        "requested_context_missing"
    )


def test_original_ranking_is_preserved():

    result = PolicyContextSelector.select(
        "Sponsored travel",
        make_result([
            policy_item("DOC-01", "General Travel Policy", rank=1),
            policy_item("DOC-02", "Sponsored Travel Policy", rank=2),
            policy_item("DOC-03", "Sponsored Travel Policy", rank=3),
        ]),
    )

    assert document_ids(result) == [
        "DOC-02", "DOC-03"
    ]

    assert [
        item.rank for item in result.retrieved_chunks
    ] == [2, 3]


def test_existing_metadata_is_preserved():

    result = PolicyContextSelector.select(
        "Annual leave policy",
        make_result([]),
    )

    assert result.metadata["original_metadata"] == (
        "preserved"
    )


def test_original_result_is_not_mutated():

    original = make_result([
        policy_item("DOC-01", "General Travel Policy"),
    ])

    result = PolicyContextSelector.select(
        "Sponsored travel",
        original,
    )

    assert len(original.retrieved_chunks) == 1

    assert result.retrieved_chunks == []

    assert "policy_context_selector" not in original.metadata


def test_selection_metrics():

    result = PolicyContextSelector.select(
        "Anti-fraud policy",
        make_result([
            policy_item("DOC-01", "Anti-Fraud Policy"),
            policy_item("DOC-02", "Travel Policy"),
            policy_item("DOC-03", "Expense Policy"),
        ]),
    )

    metadata = selection_metadata(result)

    assert metadata["input_candidates"] == 3
    assert metadata["selected_candidates"] == 1
    assert metadata["excluded_candidates"] == 2

    assert metadata["matching_document_ids"] == [
        "DOC-01"
    ]


def test_body_mention_does_not_define_policy_identity():

    result = PolicyContextSelector.select(
        "Sponsored travel policy",
        make_result([
            policy_item(
                "DOC-01",
                "General Expense Policy",
                text=(
                    "See the separate sponsored travel "
                    "policy for applicable limits."
                ),
            ),
        ]),
    )

    assert result.retrieved_chunks == []
