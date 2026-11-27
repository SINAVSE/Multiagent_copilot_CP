
import pytest

from src.agents.policy_scope import (
    PolicyScope,
    canonical_country,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Canada", "CA"),
        ("CA", "CA"),
        ("France", "FR"),
        ("FR", "FR"),
        ("Ireland", "IE"),
        ("IE", "IE"),
        ("Singapore", "SG"),
        ("SG", "SG"),
    ],
)
def test_additional_country_normalization(value, expected):

    assert canonical_country(value) == expected


@pytest.mark.parametrize(
    "country, expected",
    [
        ("Canada", "CA"),
        ("France", "FR"),
        ("Ireland", "IE"),
        ("Singapore", "SG"),
    ],
)
def test_additional_employment_country(country, expected):

    scope = PolicyScope("GitLab", country)

    assert scope.employment_country == expected

    assert scope.organization == "GitLab"

    assert scope.organization_filter() == {
        "organization": "GitLab"
    }
