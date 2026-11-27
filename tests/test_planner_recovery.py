import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

from src.agents.planner import Planner
from src.agents.errors import (
    AgentResponseError,
    ErrorCode,
)


VALID_PLAN = {
    "intent": "POLICY_QUERY",
    "plan": ["Search employee policy"],
    "selected_tools": ["search_hr_policy"],
    "missing_information": [],
}


def model_response(data):

    return SimpleNamespace(
        content=json.dumps(data)
    )


def create_planner(responses):

    model = Mock()
    model.invoke.side_effect = responses

    return Planner(model=model), model


def test_valid_plan_does_not_retry():

    planner, model = create_planner([
        model_response(VALID_PLAN)
    ])

    result = planner.plan_with_recovery(
        "What is the travel policy?"
    )

    assert result["intent"] == "POLICY_QUERY"
    assert model.invoke.call_count == 1


def test_recovers_from_malformed_json():

    planner, model = create_planner([
        SimpleNamespace(content="Invalid JSON"),
        model_response(VALID_PLAN),
    ])

    callback = Mock()

    result = planner.plan_with_recovery(
        "What is the travel policy?",
        on_error=callback,
    )

    assert result["intent"] == "POLICY_QUERY"
    assert model.invoke.call_count == 2

    callback.assert_called_once()

    event = callback.call_args.args[0]

    assert event["error_code"] == "malformed_json"
    assert event["will_retry"] is True

    second_prompt = model.invoke.call_args_list[1].args[0]

    assert "RESPONSE FORMAT CORRECTION" in second_prompt


def test_recovers_from_invalid_schema():

    planner, model = create_planner([
        model_response({
            **VALID_PLAN,
            "intent": "UNKNOWN_INTENT",
        }),
        model_response(VALID_PLAN),
    ])

    result = planner.plan_with_recovery(
        "What is the travel policy?"
    )

    assert result["intent"] == "POLICY_QUERY"
    assert model.invoke.call_count == 2


def test_recovers_from_empty_response():

    planner, model = create_planner([
        SimpleNamespace(content=""),
        model_response(VALID_PLAN),
    ])

    result = planner.plan_with_recovery(
        "What is the leave policy?"
    )

    assert result["intent"] == "POLICY_QUERY"
    assert model.invoke.call_count == 2


def test_fenced_json_needs_no_retry():

    fence = chr(96) * 3

    content = (
        fence + "json\n"
        + json.dumps(VALID_PLAN)
        + "\n" + fence
    )

    planner, model = create_planner([
        SimpleNamespace(content=content)
    ])

    result = planner.plan_with_recovery(
        "What is the leave policy?"
    )

    assert result["intent"] == "POLICY_QUERY"
    assert model.invoke.call_count == 1


def test_retry_exhaustion_is_bounded():

    planner, model = create_planner([
        SimpleNamespace(content="Invalid JSON"),
        SimpleNamespace(content="Still invalid"),
    ])

    callback = Mock()

    with pytest.raises(AgentResponseError):

        planner.plan_with_recovery(
            "What is the travel policy?",
            on_error=callback,
        )

    assert model.invoke.call_count == 2
    assert callback.call_count == 2

    first = callback.call_args_list[0].args[0]
    second = callback.call_args_list[1].args[0]

    assert first["will_retry"] is True
    assert second["will_retry"] is False


def test_different_errors_share_same_attempt_limit():

    planner, model = create_planner([
        TimeoutError("Test timeout"),
        SimpleNamespace(content="Invalid JSON"),
    ])

    with pytest.raises(AgentResponseError):

        planner.plan_with_recovery(
            "What is the travel policy?"
        )

    assert model.invoke.call_count == 2


def test_timeout_recovery():

    planner, model = create_planner([
        TimeoutError("Test timeout"),
        model_response(VALID_PLAN),
    ])

    result = planner.plan_with_recovery(
        "What is the travel policy?"
    )

    assert result["intent"] == "POLICY_QUERY"
    assert model.invoke.call_count == 2


def test_connection_recovery():

    planner, model = create_planner([
        ConnectionError("Test connection failure"),
        model_response(VALID_PLAN),
    ])

    result = planner.plan_with_recovery(
        "What is the travel policy?"
    )

    assert result["intent"] == "POLICY_QUERY"
    assert model.invoke.call_count == 2


@pytest.mark.parametrize(
    "exception_type,status",
    [
        (openai.AuthenticationError, 401),
        (openai.PermissionDeniedError, 403),
        (openai.RateLimitError, 429),
        (openai.BadRequestError, 400),
    ],
)
def test_non_retryable_api_errors(
    exception_type,
    status,
):

    request = httpx.Request(
        "POST",
        "https://example.invalid",
    )

    response = httpx.Response(
        status,
        request=request,
    )

    error = exception_type(
        "SENSITIVE_TEST_MESSAGE",
        response=response,
        body=None,
    )

    planner, model = create_planner([error])

    callback = Mock()

    with pytest.raises(exception_type):

        planner.plan_with_recovery(
            "What is the travel policy?",
            on_error=callback,
        )

    assert model.invoke.call_count == 1

    event = callback.call_args.args[0]

    assert event["will_retry"] is False
    assert "SENSITIVE_TEST_MESSAGE" not in str(event)


def test_unexpected_error_does_not_retry():

    planner, model = create_planner([
        RuntimeError("Unexpected application error")
    ])

    with pytest.raises(RuntimeError):

        planner.plan_with_recovery(
            "What is the leave policy?"
        )

    assert model.invoke.call_count == 1


def test_empty_query_never_invokes_model():

    planner, model = create_planner([])

    with pytest.raises(ValueError):

        planner.plan_with_recovery("")

    model.invoke.assert_not_called()


def test_retry_budget_can_prevent_recovery():

    planner, model = create_planner([
        SimpleNamespace(content="Invalid JSON")
    ])

    budget = Mock(return_value=False)
    callback = Mock()

    with pytest.raises(AgentResponseError):

        planner.plan_with_recovery(
            "What is the travel policy?",
            retry_allowed=budget,
            on_error=callback,
        )

    assert model.invoke.call_count == 1

    budget.assert_called_once()

    event = callback.call_args.args[0]

    assert event["will_retry"] is False


def test_retry_budget_allows_one_recovery():

    planner, model = create_planner([
        SimpleNamespace(content="Invalid JSON"),
        model_response(VALID_PLAN),
    ])

    budget = Mock(return_value=True)

    result = planner.plan_with_recovery(
        "What is the travel policy?",
        retry_allowed=budget,
    )

    assert result["intent"] == "POLICY_QUERY"
    assert model.invoke.call_count == 2
    budget.assert_called_once()


def test_original_plan_method_remains_single_attempt():

    planner, model = create_planner([
        SimpleNamespace(content="Invalid JSON"),
        model_response(VALID_PLAN),
    ])

    with pytest.raises(AgentResponseError):

        planner.plan(
            "What is the travel policy?"
        )

    assert model.invoke.call_count == 1


def test_recovery_preserves_explicit_expense_details():

    expense_plan = {
        "intent": "EXPENSE_ELIGIBILITY",
        "plan": ["Search policy", "Calculate eligibility"],
        "selected_tools": [
            "search_hr_policy",
            "calculate_expense_eligibility",
        ],
        "missing_information": [],
    }

    planner, model = create_planner([
        SimpleNamespace(content="Invalid JSON"),
        model_response(expense_plan),
    ])

    result = planner.plan_with_recovery(
        "Can I claim $100 for dinner?",
        expense_details={
            "claimed_amount": 100,
            "currency": "USD",
            "headcount": 3,
        },
    )

    assert result["expense_details"] == {
        "claimed_amount": 100,
        "currency": "USD",
        "headcount": 3,
    }

    assert model.invoke.call_count == 2


def test_duplicate_json_keys_trigger_recovery():

    planner, model = create_planner([
        SimpleNamespace(
            content='{"intent":"GREETING","intent":"POLICY_QUERY"}'
        ),
        model_response(VALID_PLAN),
    ])

    result = planner.plan_with_recovery(
        "What is the leave policy?"
    )

    assert result["intent"] == "POLICY_QUERY"
    assert model.invoke.call_count == 2
