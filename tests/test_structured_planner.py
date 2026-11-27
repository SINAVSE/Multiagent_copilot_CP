import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.agents.planner import Planner
from src.agents.planner_schema import PlannerIntent


def make_payload(
    intent="POLICY_QUERY",
    tools=None,
    missing=None,
):
    return {
        "intent": intent,
        "plan": ["Process the employee request"],
        "selected_tools": (
            ["search_hr_policy"]
            if tools is None
            else tools
        ),
        "missing_information": (
            [] if missing is None else missing
        ),
    }


def make_planner(payload):
    model = Mock()

    model.invoke.return_value = SimpleNamespace(
        content=json.dumps(payload)
    )

    return Planner(model=model), model


# --------------------------------------------------
# Positive intent classification contract
# --------------------------------------------------


@pytest.mark.parametrize(
    "intent,tools,missing,has_file",
    [
        (
            "POLICY_QUERY",
            ["search_hr_policy"],
            [],
            False,
        ),
        (
            "EXPENSE_ELIGIBILITY",
            [
                "search_hr_policy",
                "calculate_expense_eligibility",
            ],
            [],
            False,
        ),
        (
            "RECEIPT_ANALYSIS",
            ["extract_bill_details"],
            [],
            True,
        ),
        (
            "GREETING",
            [],
            [],
            False,
        ),
        (
            "OUT_OF_SCOPE",
            [],
            [],
            False,
        ),
        (
            "CLARIFICATION_REQUIRED",
            [],
            ["Expense category"],
            False,
        ),
    ],
)
def test_supported_intents(
    intent,
    tools,
    missing,
    has_file,
):
    payload = make_payload(
        intent=intent,
        tools=tools,
        missing=missing,
    )

    planner, model = make_planner(payload)

    result = planner.plan(
        query="Example employee request",
        has_file=has_file,
    )

    assert result["intent"] == intent
    assert result["selected_tools"] == tools
    assert result["missing_information"] == missing

    model.invoke.assert_called_once()


def test_all_intents_are_defined():
    assert len(PlannerIntent) == 6


# --------------------------------------------------
# Invalid intent and tool combinations
# --------------------------------------------------


@pytest.mark.parametrize(
    "intent,tools",
    [
        ("UNKNOWN_INTENT", []),
        ("POLICY_QUERY", ["unknown_tool"]),
        (
            "POLICY_QUERY",
            ["search_hr_policy", "search_hr_policy"],
        ),
        (
            "POLICY_QUERY",
            ["calculate_expense_eligibility"],
        ),
        ("POLICY_QUERY", []),
        ("EXPENSE_ELIGIBILITY", []),
        (
            "EXPENSE_ELIGIBILITY",
            ["calculate_expense_eligibility"],
        ),
        (
            "EXPENSE_ELIGIBILITY",
            [
                "calculate_expense_eligibility",
                "search_hr_policy",
            ],
        ),
        ("GREETING", ["search_hr_policy"]),
        ("OUT_OF_SCOPE", ["search_hr_policy"]),
        (
            "CLARIFICATION_REQUIRED",
            ["search_hr_policy"],
        ),
        ("RECEIPT_ANALYSIS", ["search_hr_policy"]),
    ],
)
def test_invalid_intent_tool_combinations(
    intent,
    tools,
):
    planner, model = make_planner(
        make_payload(
            intent=intent,
            tools=tools,
            missing=["Required information"],
        )
    )

    with pytest.raises(
        ValueError,
        match="schema validation",
    ):
        planner.plan(
            query="Example employee request",
            has_file=True,
        )

    model.invoke.assert_called_once()


# --------------------------------------------------
# Schema structure validation
# --------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "intent",
        "plan",
        "selected_tools",
        "missing_information",
    ],
)
def test_missing_required_fields(field):
    payload = make_payload()
    payload.pop(field)

    planner, _ = make_planner(payload)

    with pytest.raises(ValueError):
        planner.plan(query="What is the leave policy?")


@pytest.mark.parametrize(
    "field,value",
    [
        ("plan", "Search policy"),
        ("plan", []),
        ("plan", [123]),
        ("plan", [" "]),
        ("plan", ["Step"] * 11),
        ("selected_tools", "search_hr_policy"),
        ("selected_tools", [123]),
        ("selected_tools", [None]),
        ("missing_information", "Currency"),
        ("missing_information", [123]),
        ("missing_information", [" "]),
        ("missing_information", ["Field"] * 11),
    ],
)
def test_invalid_field_values(field, value):
    payload = make_payload()
    payload[field] = value

    planner, _ = make_planner(payload)

    with pytest.raises(ValueError):
        planner.plan(query="What is the travel policy?")


def test_unexpected_fields_are_rejected():
    payload = make_payload()
    payload["approved"] = True

    planner, _ = make_planner(payload)

    with pytest.raises(ValueError):
        planner.plan(query="Can I claim my expense?")


def test_clarification_requires_missing_information():
    planner, _ = make_planner(
        make_payload(
            intent="CLARIFICATION_REQUIRED",
            tools=[],
            missing=[],
        )
    )

    with pytest.raises(ValueError):
        planner.plan(query="Can I claim it?")


# --------------------------------------------------
# Receipt validation
# --------------------------------------------------


def test_receipt_analysis_requires_file():
    planner, _ = make_planner(
        make_payload(
            intent="RECEIPT_ANALYSIS",
            tools=["extract_bill_details"],
        )
    )

    with pytest.raises(
        ValueError,
        match="uploaded file",
    ):
        planner.plan(
            query="Analyze my receipt",
            has_file=False,
        )


def test_receipt_extraction_must_be_first():
    planner, _ = make_planner(
        make_payload(
            intent="RECEIPT_ANALYSIS",
            tools=[
                "search_hr_policy",
                "extract_bill_details",
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match="first tool",
    ):
        planner.plan(
            query="Analyze my receipt",
            has_file=True,
        )


def test_valid_receipt_assessment():
    tools = [
        "extract_bill_details",
        "search_hr_policy",
        "calculate_expense_eligibility",
    ]

    planner, _ = make_planner(
        make_payload(
            intent="RECEIPT_ANALYSIS",
            tools=tools,
        )
    )

    result = planner.plan(
        query="Assess my uploaded receipt",
        has_file=True,
    )

    assert result["selected_tools"] == tools


# --------------------------------------------------
# CAP-08 expense integrity
# --------------------------------------------------


def test_structured_expense_details_preserved():
    planner, _ = make_planner(
        make_payload(
            intent="EXPENSE_ELIGIBILITY",
            tools=[
                "search_hr_policy",
                "calculate_expense_eligibility",
            ],
        )
    )

    result = planner.plan(
        query="Can I claim USD 100 for dinner?",
        expense_details={
            "claimed_amount": 100.0,
            "currency": "USD",
            "headcount": 3,
        },
    )

    assert result["expense_details"] == {
        "claimed_amount": 100.0,
        "currency": "USD",
        "headcount": 3,
    }


def test_explicit_amount_is_extracted():
    planner, _ = make_planner(
        make_payload(
            intent="EXPENSE_ELIGIBILITY",
            tools=["search_hr_policy"],
        )
    )

    result = planner.plan(
        query="Can I claim USD 100 for dinner?"
    )

    assert result["expense_details"]["claimed_amount"] == 100
    assert result["expense_details"]["currency"] == "USD"


# --------------------------------------------------
# Invalid model responses
# --------------------------------------------------


def test_invalid_json_is_rejected():
    model = Mock()

    model.invoke.return_value = SimpleNamespace(
        content="This is not JSON"
    )

    planner = Planner(model=model)

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        planner.plan(query="What is the leave policy?")


@pytest.mark.parametrize(
    "payload",
    [[], None, 123, "invalid"],
)
def test_non_object_json_is_rejected(payload):
    planner, _ = make_planner(payload)

    with pytest.raises(ValueError):
        planner.plan(query="What is the leave policy?")


def test_non_string_response_is_rejected():
    model = Mock()

    model.invoke.return_value = SimpleNamespace(
        content={"intent": "POLICY_QUERY"}
    )

    planner = Planner(model=model)

    with pytest.raises(
        ValueError,
        match="Unexpected LLM response format",
    ):
        planner.plan(query="What is the leave policy?")


def test_empty_query_does_not_invoke_model():
    planner, model = make_planner(make_payload())

    with pytest.raises(ValueError):
        planner.plan(query=" ")

    model.invoke.assert_not_called()


def test_prompt_contains_supported_intents():
    planner, model = make_planner(make_payload())

    planner.plan(query="What is the leave policy?")

    prompt = model.invoke.call_args.args[0]

    for intent in PlannerIntent:
        assert intent.value in prompt
