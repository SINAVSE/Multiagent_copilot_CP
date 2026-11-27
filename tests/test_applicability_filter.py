
from copy import deepcopy

import pytest

from test_applicability_registry import (
    registry_files,
    load_registry,
)

from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
)

from src.rag.retrieval.applicability_filter import (
    ApplicabilityFilter,
)

from src.rag.schemas import (
    DocumentChunk,
    RetrievedChunk,
    RetrievalResult,
)


@pytest.fixture
def selector(registry_files):

    return ApplicabilityFilter(
        load_registry(registry_files)
    )


def make_chunk(**updates):

    data = {
        "chunk_id": "CHUNK-001",
        "document_id": "TEST-001",
        "title": "India Employee Benefits",
        "organization": "GitLab",
        "source_url": "https://example.org/policy",
        "text": "Synthetic policy evidence for testing.",
        "chunk_index": 0,
    }

    data.update(updates)

    return DocumentChunk(**data)


def make_result(*chunks):

    candidates = [
        RetrievedChunk(
            chunk=chunk,
            score=0.9,
            rank=index + 1,
            retrieval_method="bm25",
        )
        for index, chunk in enumerate(chunks)
    ]

    return RetrievalResult(
        query="What are the employee benefits?",
        retrieval_method="bm25",
        retrieved_chunks=candidates,
    )


def test_matching_country(selector):

    result = make_result(make_chunk())

    filtered, diagnostics = selector.select(
        PolicyScope("GitLab", "India"),
        result,
    )

    assert len(filtered.retrieved_chunks) == 1
    assert diagnostics["accepted_candidates"] == 1
    assert diagnostics["rejected_candidates"] == 0
    assert diagnostics["eligibility_established"] is False
    assert diagnostics["definitive_decision_allowed"] is False


@pytest.mark.parametrize(
    "country",
    ["Germany", "United States", "Canada", None],
)
def test_country_rejection(selector, country):

    result = make_result(make_chunk())

    filtered, diagnostics = selector.select(
        PolicyScope("GitLab", country),
        result,
    )

    assert filtered.retrieved_chunks == []
    assert diagnostics["rejected_candidates"] == 1
    assert diagnostics["accepted_candidates"] == 0


def test_cross_organization_rejection(selector):

    result = make_result(
        make_chunk(organization="Rice University")
    )

    with pytest.raises(PolicyScopeError):

        selector.select(
            PolicyScope("GitLab", "India"),
            result,
        )


def test_unknown_document_rejection(selector):

    result = make_result(
        make_chunk(document_id="UNKNOWN")
    )

    with pytest.raises(PolicyScopeError):

        selector.select(
            PolicyScope("GitLab", "India"),
            result,
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("title", "Modified policy title"),
        ("source_url", "https://example.org/modified"),
    ],
)
def test_metadata_mismatch(selector, field, value):

    result = make_result(
        make_chunk(**{field: value})
    )

    with pytest.raises(PolicyScopeError):

        selector.select(
            PolicyScope("GitLab", "India"),
            result,
        )


def test_empty_retrieval(selector):

    filtered, diagnostics = selector.select(
        PolicyScope("GitLab", "India"),
        make_result(),
    )

    assert filtered.retrieved_chunks == []
    assert diagnostics["input_candidates"] == 0
    assert diagnostics["accepted_candidates"] == 0
    assert diagnostics["rejected_candidates"] == 0


def test_invalid_scope(selector):

    with pytest.raises(PolicyScopeError):

        selector.select(
            {"organization": "GitLab"},
            make_result(),
        )


def test_invalid_retrieval_result(selector):

    with pytest.raises(PolicyScopeError):

        selector.select(
            PolicyScope("GitLab", "India"),
            {},
        )


def test_invalid_registry():

    with pytest.raises(PolicyScopeError):
        ApplicabilityFilter({})


def test_input_result_not_mutated(selector):

    result = make_result(make_chunk())

    original = deepcopy(result.model_dump())

    selector.select(
        PolicyScope("GitLab", "Germany"),
        result,
    )

    assert result.model_dump() == original


def test_unknown_scope_research_only(registry_files):

    record = registry_files[2]["documents"][0]

    record["candidate_applicability"] = "UNKNOWN"
    record["candidate_countries"] = []

    selector = ApplicabilityFilter(
        load_registry(registry_files)
    )

    filtered, diagnostics = selector.select(
        PolicyScope("GitLab", "Germany"),
        make_result(make_chunk()),
    )

    assert len(filtered.retrieved_chunks) == 1

    assert diagnostics[
        "definitive_decision_allowed"
    ] is False


def test_allowed_document_inventory(selector):

    assert selector.allowed_document_ids(
        PolicyScope("GitLab", "India")
    ) == ("TEST-001",)

    assert selector.allowed_document_ids(
        PolicyScope("GitLab", "Germany")
    ) == ()


def test_mixed_organization_candidates_rejected(selector):

    result = make_result(
        make_chunk(),
        make_chunk(
            chunk_id="CHUNK-002",
            organization="Rice University",
        ),
    )

    with pytest.raises(PolicyScopeError):

        selector.select(
            PolicyScope("GitLab", "India"),
            result,
        )
