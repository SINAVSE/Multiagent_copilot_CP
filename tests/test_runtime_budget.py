from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.agents.runtime_budget import (
    RuntimeBudget,
    BudgetExceededError,
    budget_scope,
    get_active_budget,
)


def test_default_configuration():

    budget = RuntimeBudget()

    assert budget.max_agent_steps == 10
    assert budget.agent_steps == 0
    assert budget.remaining_steps == 10


@pytest.mark.parametrize(
    "value",
    [0, -1, 11, 100, True, 2.5],
)
def test_invalid_agent_step_limits(value):

    with pytest.raises(ValueError):

        RuntimeBudget(
            max_agent_steps=value
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 2.5],
)
def test_invalid_token_budget(value):

    with pytest.raises(ValueError):

        RuntimeBudget(max_tokens=value)


def test_invalid_completion_limit():

    with pytest.raises(ValueError):

        RuntimeBudget(
            max_tokens=1000,
            max_completion_tokens=2000,
        )


def test_agent_steps_are_counted():

    budget = RuntimeBudget()

    budget.consume_step("planner")

    budget.consume_step("executor")

    budget.consume_step("critic")

    assert budget.agent_steps == 3
    assert budget.remaining_steps == 7


def test_tenth_step_is_allowed():

    budget = RuntimeBudget()

    for _ in range(10):

        budget.consume_step("critic")

    assert budget.agent_steps == 10
    assert budget.remaining_steps == 0


def test_eleventh_step_is_blocked():

    budget = RuntimeBudget()

    for _ in range(10):

        budget.consume_step("critic")

    with pytest.raises(BudgetExceededError) as error:

        budget.consume_step("executor")

    assert error.value.budget_type == "agent_steps"

    assert budget.agent_steps == 10


def test_lower_configured_step_limit():

    budget = RuntimeBudget(
        max_agent_steps=2
    )

    budget.consume_step("planner")
    budget.consume_step("executor")

    with pytest.raises(BudgetExceededError):

        budget.consume_step("critic")


def test_step_history():

    budget = RuntimeBudget()

    budget.consume_step("planner")

    budget.consume_step("planner_retry")

    assert budget.step_history == [
        {
            "step": 1,
            "agent": "planner",
        },
        {
            "step": 2,
            "agent": "planner_retry",
        },
    ]


def test_model_call_accounting():

    budget = RuntimeBudget()

    estimated_input, output_limit = (
        budget.begin_llm_call("Check policy")
    )

    assert estimated_input > 0

    assert output_limit == 2048

    assert budget.llm_calls == 1

    assert budget.estimated_input_tokens == (
        estimated_input
    )


def test_output_limit_is_reduced_to_remaining_budget():

    budget = RuntimeBudget(
        max_tokens=1000,
        max_completion_tokens=500,
    )

    estimated_input, output_limit = (
        budget.begin_llm_call("x" * 700)
    )

    assert estimated_input == 828

    assert output_limit == 172

    assert (
        estimated_input + output_limit
        <= budget.max_tokens
    )


def test_insufficient_tokens_block_model_attempt():

    budget = RuntimeBudget(
        max_tokens=500,
        max_completion_tokens=100,
    )

    with pytest.raises(BudgetExceededError) as error:

        budget.begin_llm_call("x" * 1000)

    assert error.value.budget_type == "tokens"

    assert budget.llm_calls == 0


def test_provider_usage_metadata():

    budget = RuntimeBudget()

    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
        }
    )

    budget.record_llm_response(
        response,
        estimated_input=200,
        reserved_output=500,
    )

    assert budget.input_tokens == 100

    assert budget.output_tokens == 40

    assert budget.total_tokens == 140

    assert budget.accounted_tokens == 140


def test_openai_response_metadata():

    budget = RuntimeBudget()

    response = SimpleNamespace(
        response_metadata={
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }
    )

    budget.record_llm_response(
        response,
        estimated_input=200,
        reserved_output=500,
    )

    assert budget.total_tokens == 150


def test_total_token_count_can_be_derived():

    budget = RuntimeBudget()

    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 50,
        }
    )

    budget.record_llm_response(
        response,
        estimated_input=200,
        reserved_output=500,
    )

    assert budget.total_tokens == 150


def test_missing_usage_is_estimated():

    budget = RuntimeBudget()

    response = SimpleNamespace(
        content="Mock response"
    )

    budget.record_llm_response(
        response,
        estimated_input=200,
        reserved_output=500,
    )

    assert budget.total_tokens == 0

    assert budget.estimated_tokens == 700

    assert budget.accounted_tokens == 700

    assert budget.estimated_usage_calls == 1


@pytest.mark.parametrize(
    "usage",
    [
        {
            "input_tokens": -100,
            "output_tokens": 50,
        },
        {
            "input_tokens": True,
            "output_tokens": 50,
        },
        {
            "input_tokens": "100",
            "output_tokens": 50,
        },
        {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 20,
        },
        {},
    ],
)
def test_invalid_usage_falls_back_to_estimation(usage):

    budget = RuntimeBudget()

    budget.record_llm_response(
        SimpleNamespace(
            usage_metadata=usage
        ),
        estimated_input=200,
        reserved_output=500,
    )

    assert budget.estimated_tokens == 700

    assert budget.estimated_usage_calls == 1

    assert budget.total_tokens == 0


def test_failed_model_call_has_conservative_usage():

    budget = RuntimeBudget()

    budget.record_failed_llm_call(
        estimated_input=200,
        reserved_output=500,
    )

    assert budget.estimated_tokens == 700

    assert budget.unreported_usage_calls == 1


def test_actual_and_estimated_usage_are_combined():

    budget = RuntimeBudget()

    budget.record_llm_response(
        SimpleNamespace(
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
            }
        ),
        estimated_input=200,
        reserved_output=500,
    )

    budget.record_failed_llm_call(
        estimated_input=200,
        reserved_output=500,
    )

    assert budget.total_tokens == 150

    assert budget.estimated_tokens == 700

    assert budget.accounted_tokens == 850


def test_provider_budget_overrun_is_detected():

    budget = RuntimeBudget(
        max_tokens=500,
        max_completion_tokens=100,
    )

    with pytest.raises(BudgetExceededError):

        budget.record_llm_response(
            SimpleNamespace(
                usage_metadata={
                    "input_tokens": 400,
                    "output_tokens": 200,
                }
            ),
            estimated_input=200,
            reserved_output=100,
        )

    assert budget.total_tokens == 600

    assert budget.remaining_tokens == 0


def test_exact_token_budget_is_allowed():

    budget = RuntimeBudget(
        max_tokens=500,
        max_completion_tokens=100,
    )

    budget.record_llm_response(
        SimpleNamespace(
            usage_metadata={
                "input_tokens": 400,
                "output_tokens": 100,
            }
        ),
        estimated_input=200,
        reserved_output=100,
    )

    assert budget.total_tokens == 500

    assert budget.remaining_tokens == 0

    with pytest.raises(BudgetExceededError):

        budget.begin_llm_call("Next call")


def test_embedding_calls_are_counted_separately():

    budget = RuntimeBudget()

    budget.record_embedding_call()

    budget.record_embedding_call()

    assert budget.embedding_calls == 2

    assert budget.llm_calls == 0

    assert budget.agent_steps == 0


def test_snapshot_contains_budget_metrics():

    budget = RuntimeBudget()

    budget.consume_step("planner")

    snapshot = budget.snapshot()

    assert snapshot["agent_steps"] == 1

    assert snapshot["remaining_steps"] == 9

    assert snapshot["max_tokens"] == 32000


def test_snapshot_does_not_mutate_original_history():

    budget = RuntimeBudget()

    budget.consume_step("planner")

    snapshot = budget.snapshot()

    snapshot["step_history"][0]["agent"] = "changed"

    assert budget.step_history[0]["agent"] == "planner"


def test_budget_scope_is_cleared():

    assert get_active_budget() is None

    budget = RuntimeBudget()

    with budget_scope(budget):

        assert get_active_budget() is budget

    assert get_active_budget() is None


def test_budget_scope_restores_previous_context():

    outer = RuntimeBudget()

    inner = RuntimeBudget()

    with budget_scope(outer):

        with budget_scope(inner):

            assert get_active_budget() is inner

        assert get_active_budget() is outer

    assert get_active_budget() is None


def test_budget_scope_cleans_up_after_failure():

    with pytest.raises(RuntimeError):

        with budget_scope(RuntimeBudget()):

            raise RuntimeError("Test failure")

    assert get_active_budget() is None


def test_concurrent_budget_scopes_are_independent():

    def execute(name):

        budget = RuntimeBudget()

        with budget_scope(budget):

            get_active_budget().consume_step(name)

            return budget.snapshot()

    with ThreadPoolExecutor(max_workers=2) as pool:

        results = list(
            pool.map(
                execute,
                ["planner", "critic"],
            )
        )

    assert results[0]["agent_steps"] == 1

    assert results[1]["agent_steps"] == 1

    assert results[0]["step_history"][0]["agent"] == (
        "planner"
    )

    assert results[1]["step_history"][0]["agent"] == (
        "critic"
    )

    assert get_active_budget() is None


def test_environment_budget_defaults(monkeypatch):

    for name in [
        "COPILOT_MAX_AGENT_STEPS",
        "COPILOT_TOKEN_BUDGET",
        "COPILOT_MAX_COMPLETION_TOKENS",
    ]:
        monkeypatch.delenv(name, raising=False)

    budget = RuntimeBudget.from_environment()

    assert budget.max_agent_steps == 10
    assert budget.max_tokens == 32000
    assert budget.max_completion_tokens == 2048


def test_environment_budget_custom_values(monkeypatch):

    monkeypatch.setenv(
        "COPILOT_MAX_AGENT_STEPS", "5"
    )

    monkeypatch.setenv(
        "COPILOT_TOKEN_BUDGET", "20000"
    )

    monkeypatch.setenv(
        "COPILOT_MAX_COMPLETION_TOKENS", "1000"
    )

    budget = RuntimeBudget.from_environment()

    assert budget.max_agent_steps == 5
    assert budget.max_tokens == 20000
    assert budget.max_completion_tokens == 1000


def test_environment_cannot_exceed_ten_steps(monkeypatch):

    monkeypatch.setenv(
        "COPILOT_MAX_AGENT_STEPS", "11"
    )

    with pytest.raises(ValueError):
        RuntimeBudget.from_environment()


@pytest.mark.parametrize(
    "name,value",
    [
        ("COPILOT_MAX_AGENT_STEPS", "invalid"),
        ("COPILOT_MAX_AGENT_STEPS", "0"),
        ("COPILOT_TOKEN_BUDGET", "-100"),
        ("COPILOT_TOKEN_BUDGET", "abc"),
        ("COPILOT_MAX_COMPLETION_TOKENS", "0"),
        ("COPILOT_MAX_COMPLETION_TOKENS", "1.5"),
    ],
)
def test_invalid_environment_budget(
    monkeypatch,
    name,
    value,
):

    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        RuntimeBudget.from_environment()


def test_environment_completion_limit_validation(
    monkeypatch,
):

    monkeypatch.setenv(
        "COPILOT_TOKEN_BUDGET", "1000"
    )

    monkeypatch.setenv(
        "COPILOT_MAX_COMPLETION_TOKENS", "2000"
    )

    with pytest.raises(ValueError):
        RuntimeBudget.from_environment()

