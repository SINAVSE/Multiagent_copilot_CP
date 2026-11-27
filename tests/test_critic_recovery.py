import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

from src.agents.critic import Critic
from src.agents.errors import AgentResponseError


APPROVED = {
    "approved": True,
    "issues": [],
    "feedback": "The answer is supported.",
}

REJECTED = {
    "approved": False,
    "issues": ["Unsupported policy claim"],
    "feedback": "Correct the policy claim.",
}

QUERY = "What is the travel policy?"


def response(data):

    return SimpleNamespace(
        content=json.dumps(data)
    )


def create_critic(responses):

    model = Mock()
    model.invoke.side_effect = responses

    critic = Critic(model=model)

    critic.validate_executor_output = Mock(
        return_value={
            "passed": True,
            "issues": [],
        }
    )

    return critic, model


def run_review(critic, **kwargs):

    return critic.review_with_recovery(
        query=QUERY,
        tool_results={},
        final_answer="Test policy answer",
        **kwargs,
    )


def test_approval_needs_no_retry():

    critic, model = create_critic([
        response(APPROVED)
    ])

    result = run_review(critic)

    assert result["approved"] is True
    assert model.invoke.call_count == 1


def test_quality_rejection_is_not_technical_failure():

    critic, model = create_critic([
        response(REJECTED)
    ])

    callback = Mock()

    result = run_review(
        critic,
        on_error=callback,
    )

    assert result["approved"] is False
    assert result["issues"] == REJECTED["issues"]
    assert model.invoke.call_count == 1

    callback.assert_not_called()


def test_deterministic_rejection_bypasses_llm():

    critic, model = create_critic([])

    critic.validate_executor_output.return_value = {
        "passed": False,
        "issues": ["Invalid expense calculation"],
    }

    result = run_review(critic)

    assert result["approved"] is False

    model.invoke.assert_not_called()


def test_malformed_json_recovery():

    critic, model = create_critic([
        SimpleNamespace(content="Invalid JSON"),
        response(APPROVED),
    ])

    callback = Mock()

    result = run_review(
        critic,
        on_error=callback,
    )

    assert result["approved"] is True
    assert model.invoke.call_count == 2

    event = callback.call_args.args[0]

    assert event["error_code"] == "malformed_json"
    assert event["will_retry"] is True

    second_prompt = (
        model.invoke.call_args_list[1].args[0]
    )

    assert "RESPONSE FORMAT CORRECTION" in second_prompt


@pytest.mark.parametrize(
    "invalid",
    [
        {
            **APPROVED,
            "approved": "true",
        },
        {
            **APPROVED,
            "issues": [123],
        },
        {
            **APPROVED,
            "feedback": "",
        },
        {
            **APPROVED,
            "issues": "No issues",
        },
    ],
)
def test_invalid_schema_recovery(invalid):

    critic, model = create_critic([
        response(invalid),
        response(APPROVED),
    ])

    result = run_review(critic)

    assert result["approved"] is True
    assert model.invoke.call_count == 2


def test_empty_response_recovery():

    critic, model = create_critic([
        SimpleNamespace(content=""),
        response(APPROVED),
    ])

    result = run_review(critic)

    assert result["approved"] is True
    assert model.invoke.call_count == 2


def test_fenced_json_does_not_require_retry():

    fence = chr(96) * 3

    content = (
        fence + "json\n"
        + json.dumps(APPROVED)
        + "\n" + fence
    )

    critic, model = create_critic([
        SimpleNamespace(content=content)
    ])

    result = run_review(critic)

    assert result["approved"] is True
    assert model.invoke.call_count == 1


def test_timeout_recovery():

    critic, model = create_critic([
        TimeoutError("Test timeout"),
        response(APPROVED),
    ])

    result = run_review(critic)

    assert result["approved"] is True
    assert model.invoke.call_count == 2


def test_connection_recovery():

    critic, model = create_critic([
        ConnectionError("Test connection failure"),
        response(APPROVED),
    ])

    result = run_review(critic)

    assert result["approved"] is True
    assert model.invoke.call_count == 2


def test_retry_exhaustion_propagates_failure():

    critic, model = create_critic([
        SimpleNamespace(content="Invalid JSON"),
        SimpleNamespace(content="Still invalid"),
    ])

    callback = Mock()

    with pytest.raises(AgentResponseError):

        run_review(
            critic,
            on_error=callback,
        )

    assert model.invoke.call_count == 2
    assert callback.call_count == 2

    assert (
        callback.call_args_list[0].args[0]["will_retry"]
        is True
    )

    assert (
        callback.call_args_list[1].args[0]["will_retry"]
        is False
    )


@pytest.mark.parametrize(
    "exception_type,status",
    [
        (openai.AuthenticationError, 401),
        (openai.PermissionDeniedError, 403),
        (openai.RateLimitError, 429),
        (openai.BadRequestError, 400),
    ],
)
def test_non_retryable_api_failure(
    exception_type,
    status,
):

    request = httpx.Request(
        "POST",
        "https://example.invalid",
    )

    http_response = httpx.Response(
        status,
        request=request,
    )

    error = exception_type(
        "SENSITIVE_TEST_MESSAGE",
        response=http_response,
        body=None,
    )

    critic, model = create_critic([error])

    callback = Mock()

    with pytest.raises(exception_type):

        run_review(
            critic,
            on_error=callback,
        )

    assert model.invoke.call_count == 1

    event = callback.call_args.args[0]

    assert event["will_retry"] is False
    assert "SENSITIVE_TEST_MESSAGE" not in str(event)


def test_shared_retry_budget_prevents_retry():

    critic, model = create_critic([
        SimpleNamespace(content="Invalid JSON")
    ])

    budget = Mock(return_value=False)

    with pytest.raises(AgentResponseError):

        run_review(
            critic,
            retry_allowed=budget,
        )

    assert model.invoke.call_count == 1

    budget.assert_called_once()


def test_shared_retry_budget_allows_retry():

    critic, model = create_critic([
        SimpleNamespace(content="Invalid JSON"),
        response(APPROVED),
    ])

    budget = Mock(return_value=True)

    result = run_review(
        critic,
        retry_allowed=budget,
    )

    assert result["approved"] is True
    assert model.invoke.call_count == 2

    budget.assert_called_once()


def test_different_errors_share_attempt_limit():

    critic, model = create_critic([
        TimeoutError("Timeout"),
        SimpleNamespace(content="Invalid JSON"),
    ])

    with pytest.raises(AgentResponseError):

        run_review(critic)

    assert model.invoke.call_count == 2


def test_duplicate_approval_fields_are_rejected():

    critic, model = create_critic([
        SimpleNamespace(
            content=(
                '{"approved": true, "approved": false, '
                '"issues": [], "feedback": "Invalid"}'
            )
        ),
        response(REJECTED),
    ])

    result = run_review(critic)

    assert result["approved"] is False
    assert model.invoke.call_count == 2


def test_unexpected_internal_failure_does_not_retry():

    critic, model = create_critic([
        RuntimeError("Internal failure")
    ])

    with pytest.raises(RuntimeError):

        run_review(critic)

    assert model.invoke.call_count == 1


def test_legacy_review_still_fails_closed():

    critic, model = create_critic([
        SimpleNamespace(content="Invalid JSON")
    ])

    result = critic.review(
        query=QUERY,
        tool_results={},
    )

    assert result["approved"] is False
    assert "Critic review failed" in result["issues"][0]
    assert model.invoke.call_count == 1
