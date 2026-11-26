
from src.agents.planner import extract_expense_details


def test_extract_usd_amount():
    result = extract_expense_details(
        "I spent USD 250 on a customer dinner."
    )

    assert result["claimed_amount"] == 250.0
    assert result["currency"] == "USD"


def test_extract_rupee_amount():
    result = extract_expense_details(
        "I spent ₹1,500 on transport."
    )

    assert result["claimed_amount"] == 1500.0
    assert result["currency"] == "INR"


def test_no_amount_returns_empty_details():
    result = extract_expense_details(
        "What is the company travel policy?"
    )

    assert result == {}