
from types import SimpleNamespace

import pytest

from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
    canonical_organization,
    canonical_country,
    organizations_in_query,
    resolve_policy_scope,
    validate_organization_evidence,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("GitLab", "GitLab"),
        (" GITLAB ", "GitLab"),
        ("rice university", "Rice University"),
        ("University of Arizona", "University of Arizona"),
        ("University of Utah", "University of Utah"),
        ("UCONN", "University of Connecticut"),
        ("University of Washington", "University of Washington"),
        ("University of Colorado System", "University of Colorado System"),
    ],
)
def test_organization_normalization(value, expected):

    assert canonical_organization(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "Unknown Company", "Arizona", None, 123],
)
def test_invalid_organizations(value):

    with pytest.raises(PolicyScopeError):

        canonical_organization(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("India", "IN"),
        ("IN", "IN"),
        ("United States", "US"),
        ("USA", "US"),
        ("United Kingdom", "GB"),
        ("Australia", "AU"),
    ],
)
def test_country_normalization(value, expected):

    assert canonical_country(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "Unknown", "USD", 123],
)
def test_invalid_country(value):

    with pytest.raises(PolicyScopeError):

        canonical_country(value)


def test_explicit_organization():

    result = resolve_policy_scope(
        query="What is the meal allowance?",
        organization="GitLab",
    )

    assert result["scope"].organization == "GitLab"

    assert result["scope_changed"] is False


def test_organization_in_query():

    result = resolve_policy_scope(
        query="What is Rice University's leave policy?"
    )

    assert result["scope"].organization == "Rice University"


def test_missing_organization():

    with pytest.raises(PolicyScopeError):

        resolve_policy_scope(
            query="What is the meal allowance?"
        )


def test_conflicting_organization():

    with pytest.raises(PolicyScopeError):

        resolve_policy_scope(
            query="What is GitLab's leave policy?",
            organization="Rice University",
        )


def test_multiple_organizations():

    with pytest.raises(PolicyScopeError):

        resolve_policy_scope(
            query="Compare GitLab and Rice University."
        )


def test_repeated_organization_reference():

    assert organizations_in_query(
        "GitLab policy for GitLab employees"
    ) == {"GitLab"}


def test_no_partial_organization_match():

    assert organizations_in_query(
        "A GitLabExtra application"
    ) == set()


def test_inherited_scope():

    previous = PolicyScope(
        "GitLab", "India"
    ).to_dict()

    result = resolve_policy_scope(
        query="What about dinner?",
        previous_scope=previous,
    )

    assert result["scope"].organization == "GitLab"
    assert result["scope"].employment_country == "IN"
    assert result["scope_changed"] is False


def test_organization_switch_resets_country():

    previous = PolicyScope(
        "GitLab", "India"
    ).to_dict()

    result = resolve_policy_scope(
        query="What is the leave policy?",
        organization="Rice University",
        previous_scope=previous,
    )

    assert result["scope"].organization == "Rice University"
    assert result["scope"].employment_country is None
    assert result["scope_changed"] is True


def test_explicit_query_switch():

    result = resolve_policy_scope(
        query="What is Rice University's policy?",
        previous_scope=PolicyScope(
            "GitLab", "India"
        ).to_dict(),
    )

    assert result["scope_changed"] is True
    assert result["scope"].employment_country is None


def test_country_switch():

    result = resolve_policy_scope(
        query="What is the leave policy?",
        organization="GitLab",
        employment_country="Germany",
        previous_scope=PolicyScope(
            "GitLab", "India"
        ).to_dict(),
    )

    assert result["scope"].employment_country == "DE"
    assert result["scope_changed"] is True


def test_travel_destination_not_employment_country():

    result = resolve_policy_scope(
        query="I am travelling to Germany. Can I claim dinner?",
        previous_scope=PolicyScope(
            "GitLab", "India"
        ).to_dict(),
    )

    assert result["scope"].employment_country == "IN"


def test_currency_not_employment_country():

    result = resolve_policy_scope(
        query="Can I claim USD 100?",
        organization="GitLab",
    )

    assert result["scope"].employment_country is None


def test_scope_filter():

    scope = PolicyScope("GitLab", "India")

    assert scope.organization_filter() == {
        "organization": "GitLab"
    }


def make_result(*organizations):

    return SimpleNamespace(
        retrieved_chunks=[
            SimpleNamespace(
                chunk=SimpleNamespace(
                    organization=organization
                )
            )
            for organization in organizations
        ]
    )


def test_valid_evidence():

    result = make_result("GitLab", "GitLab")

    assert validate_organization_evidence(
        PolicyScope("GitLab"),
        result,
    ) is result


def test_cross_organization_evidence_rejected():

    with pytest.raises(PolicyScopeError):

        validate_organization_evidence(
            PolicyScope("GitLab"),
            make_result("GitLab", "Rice University"),
        )


def test_missing_organization_metadata_rejected():

    with pytest.raises(PolicyScopeError):

        validate_organization_evidence(
            PolicyScope("GitLab"),
            make_result(None),
        )


def test_empty_evidence():

    result = make_result()

    assert validate_organization_evidence(
        PolicyScope("GitLab"),
        result,
    ) is result


def test_unvalidated_scope_rejected():

    with pytest.raises(PolicyScopeError):

        validate_organization_evidence(
            {"organization": "GitLab"},
            make_result("GitLab"),
        )


def test_scope_is_immutable():

    scope = PolicyScope("GitLab")

    with pytest.raises(AttributeError):

        scope.organization = "Rice University"
