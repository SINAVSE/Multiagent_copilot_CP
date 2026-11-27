import json
from src.agents.planner import Planner
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


class FakePlannerModel:
    def __init__(self, response):
        self.response = response
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt

        class Response:
            def __init__(self, content):
                self.content = content

        return Response(json.dumps(self.response))


def test_general_government_gift_policy_needs_no_clarification():
    model = FakePlannerModel({
        "plan": ["Search the government-official gift policy."],
        "selected_tools": ["search_hr_policy"],
        "missing_information": []
    })

    result = Planner(model=model).plan(
        "What is the policy for gifts to government officials?"
    )

    assert result["selected_tools"] == ["search_hr_policy"]
    assert result["missing_information"] == []


def test_specific_government_gift_identifies_missing_details():
    model = FakePlannerModel({
        "plan": [
            "Retrieve government-official gift restrictions.",
            "Obtain recipient jurisdiction and proposed gift value.",
            "Check whether regulatory matters are pending."
        ],
        "selected_tools": ["search_hr_policy"],
        "missing_information": [
            "Whether the recipient is a U.S. or foreign government official",
            "Proposed gift value and currency",
            "Whether licensing, investigation or regulatory matters are pending"
        ]
    })

    result = Planner(model=model).plan(
        "Can I give a gift to a government official?"
    )

    assert "search_hr_policy" in result["selected_tools"]
    assert len(result["missing_information"]) >= 1