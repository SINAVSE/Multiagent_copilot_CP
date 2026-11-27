
import pytest

from src.agents.policy_scope import PolicyScope

from src.agents.scoped_query_resolver import (
    ScopedQueryResolver,
)


@pytest.fixture
def resolver():

    return ScopedQueryResolver()


@pytest.fixture
def previous_scope():

    return PolicyScope(
        "GitLab", "India"
    ).to_dict()


@pytest.fixture
def previous_context():

    return {
        "topic": "expense",
        "category": "dinner",
        "intent": "expense_assessment",
        "claimed_amount": 100.0,
        "currency": "USD",
        "headcount": 2,
    }


def test_explicit_scope(resolver):

    result = resolver.resolve(
        query="What is the travel policy?",
        organization="GitLab",
        employment_country="India",
    )

    assert result["policy_scope"] == {
        "organization": "GitLab",
        "employment_country": "IN",
    }

    assert not result["requires_clarification"]


def test_scope_from_query(resolver):

    result = resolver.resolve(
        query="Explain the Rice University leave policy."
    )

    assert result["policy_scope"]["organization"] == (
        "Rice University"
    )


def test_missing_scope(resolver):

    result = resolver.resolve(
        query="What is the leave policy?"
    )

    assert result["requires_clarification"]

    assert result["policy_scope"] is None

    assert result["conversation_context"] == {}


def test_conflicting_scope(resolver):

    result = resolver.resolve(
        query="Explain GitLab travel policy.",
        organization="Rice University",
    )

    assert result["requires_clarification"]

    assert result["policy_scope"] is None


def test_multiple_organizations(resolver):

    result = resolver.resolve(
        query="Compare GitLab and Rice University."
    )

    assert result["requires_clarification"]


def test_unknown_organization(resolver):

    result = resolver.resolve(
        query="What is the leave policy?",
        organization="Unknown Company",
    )

    assert result["requires_clarification"]


def test_scope_inheritance(
    resolver, previous_scope
):

    result = resolver.resolve(
        query="What is the leave policy?",
        previous_scope=previous_scope,
    )

    assert result["policy_scope"] == previous_scope

    assert not result["scope_changed"]


def test_same_scope_expense_inheritance(
    resolver, previous_scope, previous_context
):

    result = resolver.resolve(
        query="What about 3 employees?",
        previous_scope=previous_scope,
        previous_context=previous_context,
    )

    context = result["conversation_context"]

    assert context["claimed_amount"] == 100.0
    assert context["currency"] == "USD"
    assert context["headcount"] == 3

    assert not result["scope_changed"]


def test_organization_change(
    resolver, previous_scope, previous_context
):

    result = resolver.resolve(
        query="What is the dinner policy?",
        organization="Rice University",
        previous_scope=previous_scope,
        previous_context=previous_context,
    )

    assert result["scope_changed"]

    assert "claimed_amount" not in (
        result["conversation_context"]
    )

    assert "headcount" not in (
        result["conversation_context"]
    )

    assert result["inherited_fields"] == {}


def test_country_change(
    resolver, previous_scope, previous_context
):

    result = resolver.resolve(
        query="What is the dinner policy?",
        organization="GitLab",
        employment_country="Germany",
        previous_scope=previous_scope,
        previous_context=previous_context,
    )

    assert result["scope_changed"]

    assert result["policy_scope"]["employment_country"] == "DE"

    assert "claimed_amount" not in (
        result["conversation_context"]
    )


def test_same_amount_after_scope_change(
    resolver, previous_scope, previous_context
):

    result = resolver.resolve(
        query="Can I claim the same amount for dinner?",
        organization="Rice University",
        previous_scope=previous_scope,
        previous_context=previous_context,
    )

    assert result["scope_changed"]

    assert result["requires_clarification"]

    assert "claimed_amount" not in (
        result["conversation_context"]
    )


def test_new_amount_after_scope_change(
    resolver, previous_scope, previous_context
):

    result = resolver.resolve(
        query="Can I claim USD 50 for dinner?",
        organization="Rice University",
        previous_scope=previous_scope,
        previous_context=previous_context,
    )

    assert result["scope_changed"]

    assert (
        result["conversation_context"]["claimed_amount"]
        == 50.0
    )

    assert "headcount" not in (
        result["conversation_context"]
    )


def test_unscoped_context_not_inherited(
    resolver, previous_context
):

    result = resolver.resolve(
        query="What about 3 employees?",
        organization="Rice University",
        previous_context=previous_context,
    )

    assert "claimed_amount" not in (
        result["conversation_context"]
    )


def test_travel_destination_not_country_change(
    resolver, previous_scope
):

    result = resolver.resolve(
        query="I am travelling to Germany.",
        previous_scope=previous_scope,
    )

    assert result["policy_scope"]["employment_country"] == "IN"

    assert not result["scope_changed"]


def test_context_not_mutated(
    resolver, previous_scope, previous_context
):

    original_context = dict(previous_context)

    original_scope = dict(previous_scope)

    resolver.resolve(
        query="What about 3 employees?",
        previous_scope=previous_scope,
        previous_context=previous_context,
    )

    assert previous_context == original_context

    assert previous_scope == original_scope


def test_scope_prompt_context(resolver):

    result = resolver.resolve(
        query="What is the travel policy?",
        organization="GitLab",
        employment_country="India",
    )

    assert "Applicable organization: GitLab" in (
        result["resolved_query"]
    )

    assert "Employment country: IN" in (
        result["resolved_query"]
    )


def test_invalid_country(resolver):

    result = resolver.resolve(
        query="What is the travel policy?",
        organization="GitLab",
        employment_country="Unknown",
    )

    assert result["requires_clarification"]


def test_scope_conflict_clears_context(
    resolver, previous_scope, previous_context
):

    result = resolver.resolve(
        query="Explain Rice University travel policy.",
        organization="GitLab",
        previous_scope=previous_scope,
        previous_context=previous_context,
    )

    assert result["requires_clarification"]

    assert result["conversation_context"] == {}

    assert result["inherited_fields"] == {}
