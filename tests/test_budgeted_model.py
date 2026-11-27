from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.agents.runtime_budget import (
    RuntimeBudget,
    BudgetExceededError,
    budget_scope,
    get_active_budget,
)

from src.agents.budgeted_model import (
    invoke_model,
    invoke_embedding,
)


def make_response(
    input_tokens=100,
    output_tokens=50,
):

    return SimpleNamespace(
        content="Mock model response",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def test_legacy_model_invocation():

    model = Mock()

    model.invoke.return_value = make_response()

    result = invoke_model(
        model,
        "Check policy",
    )

    model.invoke.assert_called_once_with(
        "Check policy"
    )

    assert result is model.invoke.return_value

    assert get_active_budget() is None


def test_budgeted_model_invocation():

    model = Mock()

    model.invoke.return_value = make_response()

    budget = RuntimeBudget()

    with budget_scope(budget):

        result = invoke_model(
            model,
            "Check policy",
        )

    model.invoke.assert_called_once_with(
        "Check policy",
        max_tokens=2048,
    )

    assert result is model.invoke.return_value

    assert budget.llm_calls == 1

    assert budget.total_tokens == 150

    assert budget.input_tokens == 100

    assert budget.output_tokens == 50


def test_output_limit_respects_remaining_budget():

    model = Mock()

    model.invoke.return_value = make_response()

    budget = RuntimeBudget(
        max_tokens=1000,
        max_completion_tokens=500,
    )

    with budget_scope(budget):

        invoke_model(
            model,
            "x" * 700,
        )

    model.invoke.assert_called_once_with(
        "x" * 700,
        max_tokens=172,
    )


def test_exhausted_budget_blocks_model_invocation():

    model = Mock()

    budget = RuntimeBudget(
        max_tokens=500,
        max_completion_tokens=100,
    )

    with budget_scope(budget):

        with pytest.raises(BudgetExceededError):

            invoke_model(
                model,
                "x" * 1000,
            )

    model.invoke.assert_not_called()

    assert budget.llm_calls == 0


def test_failed_invocation_is_accounted():

    model = Mock()

    model.invoke.side_effect = (
        TimeoutError("Mock timeout")
    )

    budget = RuntimeBudget()

    with budget_scope(budget):

        with pytest.raises(TimeoutError):

            invoke_model(
                model,
                "Check policy",
            )

    assert budget.llm_calls == 1

    assert budget.unreported_usage_calls == 1

    assert budget.estimated_tokens > 0

    assert budget.total_tokens == 0


def test_wrapper_does_not_retry_failure():

    model = Mock()

    model.invoke.side_effect = (
        ConnectionError("Mock failure")
    )

    budget = RuntimeBudget()

    with budget_scope(budget):

        with pytest.raises(ConnectionError):

            invoke_model(
                model,
                "Check policy",
            )

    assert model.invoke.call_count == 1


def test_missing_usage_metadata():

    model = Mock()

    model.invoke.return_value = SimpleNamespace(
        content="Mock response"
    )

    budget = RuntimeBudget()

    with budget_scope(budget):

        invoke_model(
            model,
            "Check policy",
        )

    assert budget.llm_calls == 1

    assert budget.estimated_usage_calls == 1

    assert budget.estimated_tokens > 0

    assert budget.total_tokens == 0


def test_malformed_response_still_consumes_tokens():

    model = Mock()

    response = make_response()

    response.content = "Invalid JSON"

    model.invoke.return_value = response

    budget = RuntimeBudget()

    with budget_scope(budget):

        result = invoke_model(
            model,
            "Return JSON",
        )

    assert result.content == "Invalid JSON"

    assert budget.total_tokens == 150

    assert budget.llm_calls == 1


def test_provider_token_overrun_is_not_double_counted():

    model = Mock()

    model.invoke.return_value = make_response(
        input_tokens=400,
        output_tokens=200,
    )

    budget = RuntimeBudget(
        max_tokens=500,
        max_completion_tokens=100,
    )

    with budget_scope(budget):

        with pytest.raises(BudgetExceededError):

            invoke_model(
                model,
                "Check policy",
            )

    assert budget.total_tokens == 600

    assert budget.unreported_usage_calls == 0

    assert budget.estimated_tokens == 0

    assert budget.llm_calls == 1


def test_multiple_models_share_request_budget():

    planner_model = Mock()

    critic_model = Mock()

    planner_model.invoke.return_value = make_response()

    critic_model.invoke.return_value = make_response()

    budget = RuntimeBudget()

    with budget_scope(budget):

        invoke_model(
            planner_model,
            "Create plan",
        )

        invoke_model(
            critic_model,
            "Review answer",
        )

    assert budget.llm_calls == 2

    assert budget.total_tokens == 300


def test_next_call_is_blocked_after_budget_exhaustion():

    model = Mock()

    model.invoke.return_value = make_response(
        input_tokens=400,
        output_tokens=100,
    )

    budget = RuntimeBudget(
        max_tokens=500,
        max_completion_tokens=100,
    )

    with budget_scope(budget):

        invoke_model(
            model,
            "Check policy",
        )

        with pytest.raises(BudgetExceededError):

            invoke_model(
                model,
                "Review answer",
            )

    assert model.invoke.call_count == 1

    assert budget.remaining_tokens == 0


def test_embedding_without_active_budget():

    operation = Mock(
        return_value=[0.1, 0.2]
    )

    result = invoke_embedding(
        operation,
        "Sample query",
    )

    assert result == [0.1, 0.2]

    operation.assert_called_once_with(
        "Sample query"
    )


def test_embedding_call_accounting():

    operation = Mock(
        return_value=[0.1, 0.2]
    )

    budget = RuntimeBudget()

    with budget_scope(budget):

        result = invoke_embedding(
            operation,
            "Sample query",
        )

    assert result == [0.1, 0.2]

    assert budget.embedding_calls == 1

    assert budget.llm_calls == 0

    assert budget.total_tokens == 0


def test_failed_embedding_attempt_is_counted():

    operation = Mock(
        side_effect=TimeoutError(
            "Embedding timeout"
        )
    )

    budget = RuntimeBudget()

    with budget_scope(budget):

        with pytest.raises(TimeoutError):

            invoke_embedding(
                operation,
                "Sample query",
            )

    assert budget.embedding_calls == 1

    operation.assert_called_once()


def test_embedding_wrapper_preserves_arguments():

    operation = Mock(
        return_value=[[0.1], [0.2]]
    )

    budget = RuntimeBudget()

    with budget_scope(budget):

        invoke_embedding(
            operation,
            ["First", "Second"],
            batch_size=2,
        )

    operation.assert_called_once_with(
        ["First", "Second"],
        batch_size=2,
    )

    assert budget.embedding_calls == 1


def test_budget_context_cleanup_after_model_failure():

    model = Mock(
        invoke=Mock(
            side_effect=RuntimeError(
                "Mock failure"
            )
        )
    )

    with pytest.raises(RuntimeError):

        with budget_scope(RuntimeBudget()):

            invoke_model(
                model,
                "Check policy",
            )

    assert get_active_budget() is None
