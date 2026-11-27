
from copy import deepcopy

import pytest

from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
)

from src.agents.scope_evidence_validator import (
    validate_scoped_tool_results,
)


@pytest.fixture
def scope():

    return PolicyScope("GitLab", "India")


@pytest.fixture
def results(scope):

    return {
        "search_hr_policy": {
            "policy_scope": scope.to_dict(),
            "insufficient_evidence": False,
            "retrieved_evidence": [
                {
                    "chunk_id": "CHUNK-001",
                    "organization": "GitLab",
                    "text": "Supporting policy text.",
                }
            ],
            "citations": [
                {
                    "chunk_id": "CHUNK-001",
                    "organization": "GitLab",
                }
            ],
        }
    }


def test_valid_scoped_results(scope, results):

    validate_scoped_tool_results(scope, results)


def test_invalid_scope(results):

    with pytest.raises(PolicyScopeError):

        validate_scoped_tool_results(
            {"organization": "GitLab"},
            results,
        )


def test_missing_scope_metadata(scope, results):

    del results["search_hr_policy"]["policy_scope"]

    with pytest.raises(PolicyScopeError):
        validate_scoped_tool_results(scope, results)


def test_mismatched_scope(scope, results):

    results["search_hr_policy"]["policy_scope"] = (
        PolicyScope("Rice University").to_dict()
    )

    with pytest.raises(PolicyScopeError):
        validate_scoped_tool_results(scope, results)


def test_mismatched_employment_country(scope, results):

    results["search_hr_policy"]["policy_scope"] = (
        PolicyScope("GitLab", "Germany").to_dict()
    )

    with pytest.raises(PolicyScopeError):
        validate_scoped_tool_results(scope, results)


def test_cross_organization_evidence(scope, results):

    results["search_hr_policy"]["retrieved_evidence"][0][
        "organization"
    ] = "Rice University"

    with pytest.raises(PolicyScopeError):
        validate_scoped_tool_results(scope, results)


def test_cross_organization_citation(scope, results):

    results["search_hr_policy"]["citations"][0][
        "organization"
    ] = "Rice University"

    with pytest.raises(PolicyScopeError):
        validate_scoped_tool_results(scope, results)


def test_missing_evidence_organization(scope, results):

    del results["search_hr_policy"][
        "retrieved_evidence"
    ][0]["organization"]

    with pytest.raises(PolicyScopeError):
        validate_scoped_tool_results(scope, results)


def test_unknown_citation(scope, results):

    results["search_hr_policy"]["citations"][0][
        "chunk_id"
    ] = "UNKNOWN"

    with pytest.raises(PolicyScopeError):
        validate_scoped_tool_results(scope, results)


def test_missing_chunk_id(scope, results):

    del results["search_hr_policy"][
        "retrieved_evidence"
    ][0]["chunk_id"]

    with pytest.raises(PolicyScopeError):
        validate_scoped_tool_results(scope, results)


def test_sufficient_answer_without_citations(scope, results):

    results["search_hr_policy"]["citations"] = []

    with pytest.raises(PolicyScopeError):
        validate_scoped_tool_results(scope, results)


def test_insufficient_evidence_is_safe(scope, results):

    policy = results["search_hr_policy"]

    policy["insufficient_evidence"] = True
    policy["retrieved_evidence"] = []
    policy["citations"] = []

    validate_scoped_tool_results(scope, results)


def test_expense_without_policy_search(scope):

    with pytest.raises(PolicyScopeError):

        validate_scoped_tool_results(
            scope,
            {
                "calculate_expense_eligibility": {
                    "compliance_status": "within_policy",
                }
            },
        )


@pytest.mark.parametrize(
    "status",
    [
        "within_policy",
        "exceeds_policy",
        "requires_approval",
    ],
)
def test_expense_with_insufficient_evidence(
    scope, results, status
):

    results["search_hr_policy"][
        "insufficient_evidence"
    ] = True

    results["calculate_expense_eligibility"] = {
        "compliance_status": status,
    }

    with pytest.raises(PolicyScopeError):
        validate_scoped_tool_results(scope, results)


def test_safe_undetermined_expense(scope, results):

    results["search_hr_policy"][
        "insufficient_evidence"
    ] = True

    results["calculate_expense_eligibility"] = {
        "compliance_status": "unable_to_determine",
    }

    validate_scoped_tool_results(scope, results)


def test_validation_does_not_mutate_results(scope, results):

    original = deepcopy(results)

    validate_scoped_tool_results(scope, results)

    assert results == original
