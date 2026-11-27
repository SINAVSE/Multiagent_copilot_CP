
import pytest

from src.agents.query_resolver import QueryResolver


@pytest.fixture
def resolver():
    return QueryResolver()


def test_standalone_policy_query(resolver):

    result = resolver.resolve(
        "What is the annual leave policy?"
    )

    assert result["context_used"] is False

    assert (
        result["conversation_context"]["category"]
        == "annual_leave"
    )


def test_expense_extraction(resolver):

    result = resolver.resolve(
        "Can I claim $100 for a team dinner?"
    )

    context = result["conversation_context"]

    assert context["category"] == "team_dinner"
    assert context["claimed_amount"] == 100
    assert context["currency"] == "USD"


def test_headcount_inheritance(resolver):

    first = resolver.resolve(
        "Can I claim $100 for a team dinner?"
    )

    second = resolver.resolve(
        "There were 3 employees.",
        first["conversation_context"],
    )

    context = second["conversation_context"]

    assert context["claimed_amount"] == 100
    assert context["currency"] == "USD"
    assert context["headcount"] == 3
    assert context["category"] == "team_dinner"
    assert second["context_used"] is True


def test_amount_override(resolver):

    first = resolver.resolve(
        "Can I claim $100 for a team dinner "
        "for 3 employees?"
    )

    second = resolver.resolve(
        "What about $150?",
        first["conversation_context"],
    )

    context = second["conversation_context"]

    assert context["claimed_amount"] == 150
    assert context["headcount"] == 3
    assert context["category"] == "team_dinner"


def test_currency_override(resolver):

    first = resolver.resolve(
        "Can I claim USD 100 for dinner?"
    )

    second = resolver.resolve(
        "What about INR 5000?",
        first["conversation_context"],
    )

    context = second["conversation_context"]

    assert context["claimed_amount"] == 5000
    assert context["currency"] == "INR"


def test_same_amount_new_category(resolver):

    first = resolver.resolve(
        "Can I claim $100 for a team dinner "
        "for 3 employees?"
    )

    second = resolver.resolve(
        "Can I claim the same amount for lunch?",
        first["conversation_context"],
    )

    context = second["conversation_context"]

    assert context["category"] == "lunch"
    assert context["claimed_amount"] == 100
    assert context["currency"] == "USD"

    assert "headcount" not in context


def test_new_expense_does_not_inherit(resolver):

    first = resolver.resolve(
        "Can I claim $100 for a team dinner "
        "for 3 employees?"
    )

    second = resolver.resolve(
        "Can I claim $200 for a hotel?",
        first["conversation_context"],
    )

    context = second["conversation_context"]

    assert context["category"] == "hotel"
    assert context["claimed_amount"] == 200
    assert "headcount" not in context


def test_expense_to_leave_reset(resolver):

    first = resolver.resolve(
        "Can I claim $100 for dinner?"
    )

    second = resolver.resolve(
        "What is the annual leave policy?",
        first["conversation_context"],
    )

    context = second["conversation_context"]

    assert context["topic"] == "leave"
    assert "claimed_amount" not in context
    assert "currency" not in context
    assert second["context_used"] is False


def test_travel_to_leave_reset(resolver):

    first = resolver.resolve(
        "What is the travel meal allowance?"
    )

    second = resolver.resolve(
        "What is the annual leave policy?",
        first["conversation_context"],
    )

    assert second["resolved_query"] == (
        "What is the annual leave policy?"
    )


def test_leave_to_travel_reset(resolver):

    first = resolver.resolve(
        "What is the annual leave policy?"
    )

    second = resolver.resolve(
        "What is the travel meal allowance?",
        first["conversation_context"],
    )

    assert second["context_used"] is False

    assert (
        second["conversation_context"]["category"]
        == "meal"
    )


def test_international_hotel_follow_up(resolver):

    first = resolver.resolve(
        "What is the domestic hotel limit?"
    )

    second = resolver.resolve(
        "What about international travel?",
        first["conversation_context"],
    )

    context = second["conversation_context"]

    assert context["category"] == "hotel"

    assert (
        context["travel_type"]
        == "international"
    )

    assert second["context_used"] is True


def test_hotel_nights_update(resolver):

    first = resolver.resolve(
        "Can I claim $200 for a hotel?"
    )

    second = resolver.resolve(
        "What about two nights?",
        first["conversation_context"],
    )

    context = second["conversation_context"]

    assert context["category"] == "hotel"
    assert context["claimed_amount"] == 200
    assert context["nights"] == 2


def test_unknown_same_amount_reference(resolver):

    result = resolver.resolve(
        "Can I claim the same amount?"
    )

    assert result["requires_clarification"] is True

    assert result["resolution_type"] == (
        "unresolved_amount_reference"
    )


def test_multiple_amounts_detected(resolver):

    result = resolver.resolve(
        "I spent $100 on meals "
        "and $200 on hotel."
    )

    assert result["resolution_type"] == (
        "multiple_amounts"
    )

    assert result["requires_clarification"] is True

    assert "claimed_amount" not in (
        result["conversation_context"]
    )


def test_out_of_scope_context_reset(resolver):

    first = resolver.resolve(
        "Can I claim $100 for dinner?"
    )

    second = resolver.resolve(
        "What is today's weather?",
        first["conversation_context"],
    )

    assert second["context_used"] is False

    assert "claimed_amount" not in (
        second["conversation_context"]
    )


def test_explicit_policy_reference(resolver):

    first = resolver.resolve(
        "Tell me about customer dinner expenses."
    )

    second = resolver.resolve(
        "What is the limit for that?",
        first["conversation_context"],
    )

    assert (
        second["conversation_context"]["category"]
        == "customer_dinner"
    )

    assert second["context_used"] is True


def test_three_turn_inheritance(resolver):

    first = resolver.resolve(
        "Can I claim $100 for a team dinner?"
    )

    second = resolver.resolve(
        "There were 3 employees.",
        first["conversation_context"],
    )

    third = resolver.resolve(
        "What about $150?",
        second["conversation_context"],
    )

    context = third["conversation_context"]

    assert context["category"] == "team_dinner"
    assert context["claimed_amount"] == 150
    assert context["currency"] == "USD"
    assert context["headcount"] == 3


def test_empty_query(resolver):

    result = resolver.resolve("   ")

    assert result["resolution_type"] == (
        "empty_query"
    )

    assert result["resolved_query"] == ""

    assert result["context_used"] is False
