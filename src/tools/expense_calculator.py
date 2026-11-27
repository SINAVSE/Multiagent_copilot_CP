from typing import Literal
from pydantic import BaseModel, Field
import math


class ExpenseCalculationResult(BaseModel):
    """Structured result of a policy-based expense assessment."""

    compliance_status: Literal[
        "within_policy",
        "requires_approval",
        "exceeds_policy",
        "unable_to_determine",
    ]

    approval_required: bool | None = None

    claimed_amount: float | None = None
    policy_limit: float | None = None
    eligible_amount: float | None = None
    excess_amount: float | None = None

    currency: str | None = None

    calculation: str | None = None

    missing_information: list[str] = Field(default_factory=list)

    recommendation: str

    policy_evidence: list[str] = Field(default_factory=list)
    headcount: int | None = None
    per_person_claimed_amount: float | None = None
    per_person_policy_limit: float | None = None

def _validate_amount(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a valid number.")

    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite.")

    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")

def calculate_expense(
    claimed_amount: float,
    policy_limit: float,
    currency: str = "INR",
    approval_required: bool | None = None,
) -> ExpenseCalculationResult:
    """Compare a claimed expense against an explicitly supplied policy limit."""

    _validate_amount(claimed_amount, "Claimed amount")
    _validate_amount(policy_limit, "Policy limit")

    eligible_amount = min(claimed_amount, policy_limit)
    excess_amount = max(claimed_amount - policy_limit, 0)

    if claimed_amount > policy_limit:
        status = "exceeds_policy"
    elif approval_required is True:
        status = "requires_approval"
    else:
        status = "within_policy"

    if status == "exceeds_policy":
        recommendation = "The claimed expense exceeds the supplied policy limit. Review the excess with the appropriate authority."
    elif approval_required:
        recommendation = "The expense is within the supplied limit, but approval is required before proceeding."
    else:
        recommendation = "The expense is within the supplied policy limit. Other applicable conditions still need to be checked."

    return ExpenseCalculationResult(
        compliance_status=status,
        approval_required=approval_required,
        claimed_amount=claimed_amount,
        policy_limit=policy_limit,
        eligible_amount=eligible_amount,
        excess_amount=excess_amount,
        currency=currency,
        calculation=(
            f"Eligible amount = min({claimed_amount}, {policy_limit}) = "
            f"{eligible_amount}; excess = {excess_amount}."
        ),
        recommendation=recommendation,
    )


def unable_to_determine(
    reason: str,
    currency: str | None = None,
    claimed_amount: float | None = None,
    missing_information: list[str] | None = None,
) -> ExpenseCalculationResult:
    """Return an assessment when policy evidence or required information is insufficient."""

    if claimed_amount is not None and claimed_amount < 0:
        raise ValueError("Claimed amount cannot be negative.")

    return ExpenseCalculationResult(
        compliance_status="unable_to_determine",
        approval_required=None,
        claimed_amount=claimed_amount,
        policy_limit=None,
        eligible_amount=None,
        excess_amount=None,
        currency=currency,
        calculation=None,
        missing_information=missing_information or [reason],
        recommendation=reason,
        policy_evidence=[],
    )


def calculate_per_person_expense(
    total_claimed_amount: float,
    headcount: int,
    per_person_policy_limit: float,
    currency: str = "INR",
    approval_required: bool | None = None,
) -> ExpenseCalculationResult:
    """Assess an expense against an explicitly supplied per-person limit."""

    _validate_amount(total_claimed_amount, "Claimed amount")
    _validate_amount(per_person_policy_limit, "Per-person policy limit")

    if isinstance(headcount, bool) or not isinstance(headcount, int):
        raise ValueError("Headcount must be a whole number.")

    if headcount <= 0:
        raise ValueError("Headcount must be greater than zero.")

    per_person_amount = total_claimed_amount / headcount
    total_policy_limit = headcount * per_person_policy_limit

    eligible_amount = min(total_claimed_amount, total_policy_limit)
    excess_amount = max(total_claimed_amount - total_policy_limit, 0)

    if per_person_amount > per_person_policy_limit:
        status = "exceeds_policy"
    elif approval_required is True:
        status = "requires_approval"
    else:
        status = "within_policy"

    if status == "exceeds_policy":
        recommendation = (
            "The per-person expense exceeds the supplied policy limit. "
            "Review the excess with the appropriate authority."
        )
    elif approval_required:
        recommendation = (
            "The expense is within the per-person limit, "
            "but approval is required before proceeding."
        )
    else:
        recommendation = (
            "The expense is within the supplied per-person policy limit. "
            "Other applicable conditions still need to be checked."
        )

    return ExpenseCalculationResult(
        compliance_status=status,
        approval_required=approval_required is True,
        claimed_amount=total_claimed_amount,
        policy_limit=total_policy_limit,
        eligible_amount=eligible_amount,
        excess_amount=excess_amount,
        currency=currency,
        headcount=headcount,
        per_person_claimed_amount=per_person_amount,
        per_person_policy_limit=per_person_policy_limit,
        calculation=(
            f"{total_claimed_amount} / {headcount} = "
            f"{per_person_amount:.2f} per person; "
            f"{headcount} x {per_person_policy_limit} = "
            f"{total_policy_limit} total policy limit."
        ),
        recommendation=recommendation,
    )
