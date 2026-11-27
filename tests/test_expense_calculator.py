import pytest
from src.tools.expense_calculator import (calculate_expense,
    calculate_per_person_expense,
    unable_to_determine,)


def test_expense_within_policy():
    result = calculate_expense(
        claimed_amount=1500,
        policy_limit=1800,
    )

    assert result.compliance_status == "within_policy"
    assert result.eligible_amount == 1500
    assert result.excess_amount == 0
    assert result.approval_required is None


def test_expense_exceeds_policy():
    result = calculate_expense(
        claimed_amount=2200,
        policy_limit=1800,
    )

    assert result.compliance_status == "exceeds_policy"
    assert result.eligible_amount == 1800
    assert result.excess_amount == 400


def test_expense_requires_approval():
    result = calculate_expense(
        claimed_amount=2000,
        policy_limit=3000,
        approval_required=True,
    )

    assert result.compliance_status == "requires_approval"
    assert result.approval_required is True


def test_negative_claimed_amount_is_rejected():
    with pytest.raises(ValueError):
        calculate_expense(
            claimed_amount=-100,
            policy_limit=1800,
        )


def test_negative_policy_limit_is_rejected():
    with pytest.raises(ValueError):
        calculate_expense(
            claimed_amount=1000,
            policy_limit=-1800,
        )


def test_missing_policy_limit_returns_unable_to_determine():
    result = unable_to_determine(
        reason="The applicable hotel expense limit was not found in the policy.",
        claimed_amount=7200,
        currency="INR",
    )

    assert result.compliance_status == "unable_to_determine"
    assert result.claimed_amount == 7200
    assert result.policy_limit is None
    assert result.eligible_amount is None
    assert result.excess_amount is None
    assert result.approval_required is None


def test_missing_required_information_is_recorded():
    result = unable_to_determine(
        reason="The attendee count is required to assess the per-person limit.",
        claimed_amount=9000,
        currency="INR",
        missing_information=["Number of attendees"],
    )

    assert result.compliance_status == "unable_to_determine"
    assert "Number of attendees" in result.missing_information


def test_per_person_expense_within_policy():
    result = calculate_per_person_expense(
        total_claimed_amount=9000,
        headcount=4,
        per_person_policy_limit=2500,
    )

    assert result.compliance_status == "within_policy"
    assert result.per_person_claimed_amount == 2250
    assert result.policy_limit == 10000
    assert result.eligible_amount == 9000
    assert result.excess_amount == 0


def test_per_person_expense_exceeds_policy():
    result = calculate_per_person_expense(
        total_claimed_amount=9000,
        headcount=3,
        per_person_policy_limit=2500,
    )

    assert result.compliance_status == "exceeds_policy"
    assert result.per_person_claimed_amount == 3000
    assert result.policy_limit == 7500
    assert result.eligible_amount == 7500
    assert result.excess_amount == 1500


def test_per_person_expense_requires_approval():
    result = calculate_per_person_expense(
        total_claimed_amount=9000,
        headcount=4,
        per_person_policy_limit=2500,
        approval_required=True,
    )

    assert result.compliance_status == "requires_approval"
    assert result.approval_required is True

@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_claimed_amount_is_rejected(amount):
    with pytest.raises(ValueError):
        calculate_expense(
            claimed_amount=amount,
            policy_limit=1800,
        )


@pytest.mark.parametrize("headcount", [0, -1, 2.5, True])
def test_invalid_headcount_is_rejected(headcount):
    with pytest.raises(ValueError):
        calculate_per_person_expense(
            total_claimed_amount=9000,
            headcount=headcount,
            per_person_policy_limit=2500,
        )


def test_expense_equal_to_policy_limit_is_within_policy():
    result = calculate_expense(
        claimed_amount=1800,
        policy_limit=1800,
    )

    assert result.compliance_status == "within_policy"
    assert result.eligible_amount == 1800
    assert result.excess_amount == 0


def test_per_person_expense_equal_to_limit_is_within_policy():
    result = calculate_per_person_expense(
        total_claimed_amount=10000,
        headcount=4,
        per_person_policy_limit=2500,
    )

    assert result.compliance_status == "within_policy"
    assert result.per_person_claimed_amount == 2500
    assert result.excess_amount == 0