
from types import SimpleNamespace

import pytest

from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
)

from src.agents.policy_applicability import (
    assess_applicability,
    validate_applicable_evidence,
)


def metadata(
    applicability="COUNTRY_SPECIFIC",
    countries=None,
    organization="GitLab",
    status="verified",
):

    return {
        "document_id": "TEST-001",
        "organization": organization,
        "review_status": status,
        "verified_applicability": applicability,
        "verified_countries": (
            ["IN"] if countries is None else countries
        ),
    }


@pytest.mark.parametrize(
    "country, expected",
    [
        ("India", True),
        ("Germany", False),
        ("United States", False),
        ("Canada", False),
        ("France", False),
        ("Ireland", False),
        ("Singapore", False),
        (None, False),
    ],
)
def test_country_specific_applicability(country, expected):

    result = assess_applicability(
        PolicyScope("GitLab", country),
        metadata(),
    )

    assert result["allowed"] is expected


@pytest.mark.parametrize(
    "country",
    ["India", "Germany", "United States", None],
)
def test_verified_global_policy(country):

    result = assess_applicability(
        PolicyScope("GitLab", country),
        metadata("GLOBAL", []),
    )

    assert result["allowed"] is True
    assert result["eligibility_established"] is False


@pytest.mark.parametrize(
    "country, expected",
    [
        ("India", True),
        ("Germany", True),
        ("United States", False),
        (None, False),
    ],
)
def test_multi_country_policy(country, expected):

    result = assess_applicability(
        PolicyScope("GitLab", country),
        metadata("MULTI_COUNTRY", ["IN", "DE"]),
    )

    assert result["allowed"] is expected


@pytest.mark.parametrize(
    "status",
    [
        "pending_content_review",
        "pending_semantic_review",
        "candidate",
        None,
    ],
)
def test_unverified_policy_rejected(status):

    result = assess_applicability(
        PolicyScope("GitLab", "India"),
        metadata(status=status),
    )

    assert result["allowed"] is False
    assert result["reason"] == "unverified_applicability"


@pytest.mark.parametrize(
    "kind, countries",
    [
        ("COUNTRY_SPECIFIC", []),
        ("COUNTRY_SPECIFIC", ["IN", "DE"]),
        ("MULTI_COUNTRY", ["IN"]),
        ("MULTI_COUNTRY", ["IN", "IN"]),
        ("GLOBAL", ["IN"]),
        ("UNKNOWN", []),
        ("INVALID", []),
        ("COUNTRY_SPECIFIC", ["INVALID"]),
        ("COUNTRY_SPECIFIC", "IN"),
    ],
)
def test_invalid_applicability_metadata(kind, countries):

    result = assess_applicability(
        PolicyScope("GitLab", "India"),
        metadata(kind, countries),
    )

    assert result["allowed"] is False


def test_cross_organization_policy():

    result = assess_applicability(
        PolicyScope("GitLab", "India"),
        metadata(organization="Rice University"),
    )

    assert result["allowed"] is False
    assert result["reason"] == "organization_mismatch"


def test_missing_metadata():

    result = assess_applicability(
        PolicyScope("GitLab", "India"),
        None,
    )

    assert result["allowed"] is False


def test_country_match_is_not_expense_approval():

    result = assess_applicability(
        PolicyScope("GitLab", "India"),
        metadata(),
    )

    assert result["allowed"] is True
    assert result["eligibility_established"] is False


def make_result(organization="GitLab"):

    return SimpleNamespace(
        retrieved_chunks=[
            SimpleNamespace(
                chunk=SimpleNamespace(
                    document_id="TEST-001",
                    organization=organization,
                )
            )
        ]
    )


def test_applicable_evidence():

    result = make_result()

    assert validate_applicable_evidence(
        PolicyScope("GitLab", "India"),
        result,
        {"TEST-001": metadata()},
    ) is result


def test_inapplicable_country_evidence():

    with pytest.raises(PolicyScopeError):

        validate_applicable_evidence(
            PolicyScope("GitLab", "Germany"),
            make_result(),
            {"TEST-001": metadata()},
        )


def test_missing_registry_record():

    with pytest.raises(PolicyScopeError):

        validate_applicable_evidence(
            PolicyScope("GitLab", "India"),
            make_result(),
            {},
        )


def test_cross_organization_evidence():

    with pytest.raises(PolicyScopeError):

        validate_applicable_evidence(
            PolicyScope("GitLab", "India"),
            make_result("Rice University"),
            {"TEST-001": metadata()},
        )


def test_unverified_evidence():

    with pytest.raises(PolicyScopeError):

        validate_applicable_evidence(
            PolicyScope("GitLab", "India"),
            make_result(),
            {
                "TEST-001": metadata(
                    status="pending_semantic_review"
                )
            },
        )


def test_mismatched_registry_document():

    record = metadata()
    record["document_id"] = "OTHER-DOCUMENT"

    with pytest.raises(PolicyScopeError):

        validate_applicable_evidence(
            PolicyScope("GitLab", "India"),
            make_result(),
            {"TEST-001": record},
        )


def test_empty_evidence():

    result = SimpleNamespace(
        retrieved_chunks=[]
    )

    assert validate_applicable_evidence(
        PolicyScope("GitLab", "India"),
        result,
        {},
    ) is result
