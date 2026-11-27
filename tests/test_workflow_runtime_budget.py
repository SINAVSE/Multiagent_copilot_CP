import uuid
from types import SimpleNamespace

import pytest

from src.agents.runtime_budget import (
    BudgetExceededError,
    get_active_budget,
)

from src.agents.budgeted_model import (
    invoke_model,
    invoke_embedding,
)

from tests.test_workflow_error_handling import (
    create_workflow,
    run_workflow,
    response,
    invalid_response,
    VALID_PLAN,
    APPROVED,
    REJECTED,
    TOOL_RESULTS,
)


@pytest.fixture(autouse=True)
def reset_budget_environment(monkeypatch):

    for name in [
        "COPILOT_MAX_AGENT_STEPS",
        "COPILOT_TOKEN_BUDGET",
        "COPILOT_MAX_COMPLETION_TOKENS",
    ]:
        monkeypatch.delenv(name, raising=False)


def with_usage(data, input_tokens=100, output_tokens=50):

    result = response(data)

    result.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    return result


def assert_budget_failure(result, reason, node):

    assert result["status"] == "technical_failure"
    assert result["error_code"] == "budget_exhausted"
    assert result["budget_exhausted"] is True

    assert result["budget_exhaustion_reason"] == reason
    assert result["failed_node"] == node

    assert result["is_approved"] is False
    assert result["tool_results"] == {}
    assert result["retrieved_evidence"] == []


# --------------------------------------------------
# Normal execution and actual usage
# --------------------------------------------------


def test_normal_workflow_counts_three_agent_steps():

    workflow, planner, critic = create_workflow(
        planner_responses=[with_usage(VALID_PLAN)],
        critic_responses=[with_usage(APPROVED)],
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"

    budget = result["runtime_budget"]

    assert budget["agent_steps"] == 3
    assert budget["remaining_steps"] == 7
    assert budget["llm_calls"] == 2
    assert budget["total_tokens"] == 300
    assert budget["estimated_tokens"] == 0

    assert result["budget_exhausted"] is False

    assert [
        item["agent"]
        for item in budget["step_history"]
    ] == [
        "planner",
        "executor",
        "critic",
    ]


def test_budget_is_available_inside_executor_thread():

    workflow, planner, critic = create_workflow()

    def execute(**kwargs):

        active = get_active_budget()

        assert active is not None
        assert active.agent_steps == 2

        return TOOL_RESULTS

    workflow.executor.execute.side_effect = execute

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["runtime_budget"]["agent_steps"] == 3


def test_workflow_passes_dynamic_completion_limit():

    workflow, planner, critic = create_workflow()

    result = run_workflow(workflow)

    assert result["status"] == "approved"

    planner_kwargs = planner.invoke.call_args.kwargs
    critic_kwargs = critic.invoke.call_args.kwargs

    assert 0 < planner_kwargs["max_tokens"] <= 2048
    assert 0 < critic_kwargs["max_tokens"] <= 2048


# --------------------------------------------------
# Technical retries
# --------------------------------------------------


def test_planner_retry_consumes_additional_step():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            with_usage(VALID_PLAN),
        ],
        critic_responses=[
            with_usage(APPROVED),
        ],
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"

    assert result["technical_retry_count"] == 1

    budget = result["runtime_budget"]

    assert budget["agent_steps"] == 4
    assert budget["llm_calls"] == 3

    assert planner.invoke.call_count == 2


def test_critic_retry_consumes_additional_step():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            with_usage(VALID_PLAN),
        ],
        critic_responses=[
            invalid_response(),
            with_usage(APPROVED),
        ],
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["technical_retry_count"] == 1
    assert result["runtime_budget"]["agent_steps"] == 4
    assert result["runtime_budget"]["llm_calls"] == 3

    workflow.executor.execute.assert_called_once()


def test_both_technical_retries_are_counted():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            with_usage(VALID_PLAN),
        ],
        critic_responses=[
            invalid_response(),
            with_usage(APPROVED),
        ],
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["technical_retry_count"] == 2
    assert result["runtime_budget"]["agent_steps"] == 5
    assert result["runtime_budget"]["llm_calls"] == 4


# --------------------------------------------------
# Quality correction loop
# --------------------------------------------------


def test_quality_retry_consumes_two_additional_steps():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            with_usage(VALID_PLAN),
        ],
        critic_responses=[
            with_usage(REJECTED),
            with_usage(APPROVED),
        ],
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["retry_count"] == 1

    assert result["runtime_budget"]["agent_steps"] == 5

    assert workflow.executor.execute.call_count == 2


def test_two_quality_retries_consume_seven_steps():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            with_usage(VALID_PLAN),
        ],
        critic_responses=[
            with_usage(REJECTED),
            with_usage(REJECTED),
            with_usage(APPROVED),
        ],
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["retry_count"] == 2
    assert result["runtime_budget"]["agent_steps"] == 7


def test_maximum_existing_recovery_path_nine_steps():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            with_usage(VALID_PLAN),
        ],
        critic_responses=[
            invalid_response(),
            with_usage(REJECTED),
            with_usage(REJECTED),
            with_usage(APPROVED),
        ],
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"

    assert result["technical_retry_count"] == 2
    assert result["retry_count"] == 2

    assert result["runtime_budget"]["agent_steps"] == 9
    assert result["runtime_budget"]["remaining_steps"] == 1


# --------------------------------------------------
# Agent-step exhaustion
# --------------------------------------------------


def test_step_budget_blocks_executor(monkeypatch):

    monkeypatch.setenv(
        "COPILOT_MAX_AGENT_STEPS", "1"
    )

    workflow, planner, critic = create_workflow()

    result = run_workflow(workflow)

    assert_budget_failure(
        result, "agent_steps", "executor"
    )

    assert result["runtime_budget"]["agent_steps"] == 1

    workflow.executor.execute.assert_not_called()
    critic.invoke.assert_not_called()


def test_step_budget_blocks_critic(monkeypatch):

    monkeypatch.setenv(
        "COPILOT_MAX_AGENT_STEPS", "2"
    )

    workflow, planner, critic = create_workflow()

    result = run_workflow(workflow)

    assert_budget_failure(
        result, "agent_steps", "critic"
    )

    assert result["runtime_budget"]["agent_steps"] == 2

    workflow.executor.execute.assert_called_once()
    critic.invoke.assert_not_called()


def test_step_budget_blocks_planner_retry(monkeypatch):

    monkeypatch.setenv(
        "COPILOT_MAX_AGENT_STEPS", "1"
    )

    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            with_usage(VALID_PLAN),
        ]
    )

    result = run_workflow(workflow)

    assert_budget_failure(
        result, "agent_steps", "planner"
    )

    assert planner.invoke.call_count == 1
    assert result["runtime_budget"]["agent_steps"] == 1

    workflow.executor.execute.assert_not_called()


def test_step_budget_blocks_quality_retry(monkeypatch):

    monkeypatch.setenv(
        "COPILOT_MAX_AGENT_STEPS", "3"
    )

    workflow, planner, critic = create_workflow(
        critic_responses=[
            with_usage(REJECTED),
            with_usage(APPROVED),
        ],
    )

    result = run_workflow(workflow)

    assert_budget_failure(
        result, "agent_steps", "executor"
    )

    assert result["runtime_budget"]["agent_steps"] == 3
    assert critic.invoke.call_count == 1

    workflow.executor.execute.assert_called_once()


# --------------------------------------------------
# Token budget exhaustion
# --------------------------------------------------


def test_token_budget_blocks_planner_before_api_call(
    monkeypatch,
):

    monkeypatch.setenv(
        "COPILOT_TOKEN_BUDGET", "100"
    )

    monkeypatch.setenv(
        "COPILOT_MAX_COMPLETION_TOKENS", "50"
    )

    workflow, planner, critic = create_workflow()

    result = run_workflow(workflow)

    assert_budget_failure(
        result, "tokens", "planner"
    )

    planner.invoke.assert_not_called()
    critic.invoke.assert_not_called()
    workflow.executor.execute.assert_not_called()

    assert result["runtime_budget"]["llm_calls"] == 0


def test_provider_reported_overrun_terminates_workflow():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            with_usage(
                VALID_PLAN,
                input_tokens=40000,
                output_tokens=100,
            ),
        ]
    )

    result = run_workflow(workflow)

    assert_budget_failure(
        result, "tokens", "planner"
    )

    assert result["runtime_budget"]["total_tokens"] == 40100

    assert result["runtime_budget"]["remaining_tokens"] == 0

    planner.invoke.assert_called_once()
    workflow.executor.execute.assert_not_called()
    critic.invoke.assert_not_called()


def test_exact_budget_consumption_blocks_next_model_call():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            with_usage(
                VALID_PLAN,
                input_tokens=31000,
                output_tokens=1000,
            ),
        ]
    )

    result = run_workflow(workflow)

    assert_budget_failure(
        result, "tokens", "critic"
    )

    assert result["runtime_budget"]["total_tokens"] == 32000

    assert result["runtime_budget"]["llm_calls"] == 1

    critic.invoke.assert_not_called()


def test_executor_budget_error_is_not_tool_failure():

    workflow, planner, critic = create_workflow()

    workflow.executor.execute.side_effect = (
        BudgetExceededError("tokens")
    )

    result = run_workflow(workflow)

    assert_budget_failure(
        result, "tokens", "executor"
    )

    assert result["technical_retry_count"] == 0

    critic.invoke.assert_not_called()


# --------------------------------------------------
# Executor-internal invocation accounting
# --------------------------------------------------


def test_executor_model_calls_share_workflow_budget():

    from unittest.mock import Mock

    internal_model = Mock()

    internal_model.invoke.return_value = (
        with_usage({"answer": "Mock policy"})
    )

    workflow, planner, critic = create_workflow(
        planner_responses=[with_usage(VALID_PLAN)],
        critic_responses=[with_usage(APPROVED)],
    )

    def execute(**kwargs):

        invoke_model(
            internal_model,
            "Generate policy answer",
        )

        return TOOL_RESULTS

    workflow.executor.execute.side_effect = execute

    result = run_workflow(workflow)

    assert result["status"] == "approved"

    budget = result["runtime_budget"]

    assert budget["llm_calls"] == 3
    assert budget["total_tokens"] == 450
    assert budget["agent_steps"] == 3


def test_embedding_calls_share_workflow_budget():

    from unittest.mock import Mock

    operation = Mock(return_value=[0.1, 0.2])

    workflow, planner, critic = create_workflow()

    def execute(**kwargs):

        invoke_embedding(
            operation,
            "Annual leave policy",
        )

        return TOOL_RESULTS

    workflow.executor.execute.side_effect = execute

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["runtime_budget"]["embedding_calls"] == 1

    operation.assert_called_once()


# --------------------------------------------------
# Persistence and request reset
# --------------------------------------------------


def test_runtime_budget_is_persisted_in_checkpoint():

    workflow, planner, critic = create_workflow()

    thread_id = str(uuid.uuid4())

    result = run_workflow(
        workflow,
        thread_id,
    )

    snapshot = workflow.get_thread_state(thread_id)

    assert snapshot.values["runtime_budget"] == (
        result["runtime_budget"]
    )


def test_budget_resets_for_next_conversation_turn():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            with_usage(VALID_PLAN),
            with_usage(VALID_PLAN),
        ],
        critic_responses=[
            with_usage(APPROVED),
            with_usage(APPROVED),
        ],
    )

    thread_id = str(uuid.uuid4())

    first = run_workflow(workflow, thread_id)
    second = run_workflow(workflow, thread_id)

    assert first["runtime_budget"]["agent_steps"] == 3
    assert second["runtime_budget"]["agent_steps"] == 3

    assert first["runtime_budget"]["total_tokens"] == 300
    assert second["runtime_budget"]["total_tokens"] == 300


def test_budget_context_cleared_after_success():

    workflow, planner, critic = create_workflow()

    run_workflow(workflow)

    assert get_active_budget() is None


def test_budget_context_cleared_after_exhaustion(
    monkeypatch,
):

    monkeypatch.setenv(
        "COPILOT_MAX_AGENT_STEPS", "1"
    )

    workflow, planner, critic = create_workflow()

    result = run_workflow(workflow)

    assert result["budget_exhausted"] is True

    assert get_active_budget() is None


def test_exhaustion_does_not_contaminate_next_request(
    monkeypatch,
):

    monkeypatch.setenv(
        "COPILOT_MAX_AGENT_STEPS", "1"
    )

    workflow, planner, critic = create_workflow(
        planner_responses=[
            with_usage(VALID_PLAN),
            with_usage(VALID_PLAN),
        ],
        critic_responses=[
            with_usage(APPROVED),
        ],
    )

    thread_id = str(uuid.uuid4())

    first = run_workflow(workflow, thread_id)

    assert first["budget_exhausted"] is True

    monkeypatch.setenv(
        "COPILOT_MAX_AGENT_STEPS", "10"
    )

    second = run_workflow(workflow, thread_id)

    assert second["status"] == "approved"
    assert second["budget_exhausted"] is False
    assert second["budget_exhaustion_reason"] is None
    assert second["runtime_budget"]["agent_steps"] == 3


# --------------------------------------------------
# Observability
# --------------------------------------------------


def test_budget_exhaustion_event_is_logged(monkeypatch):

    monkeypatch.setenv(
        "COPILOT_MAX_AGENT_STEPS", "1"
    )

    workflow, planner, critic = create_workflow()

    run_workflow(workflow)

    events = [
        call.args[0]
        for call in workflow.event_logger.log_event.call_args_list
    ]

    budget_events = [
        event for event in events
        if event.get("event") == "budget_exhausted"
    ]

    assert len(budget_events) == 1

    event = budget_events[0]

    assert event["budget_type"] == "agent_steps"
    assert event["node"] == "executor"
    assert event["runtime_budget"]["agent_steps"] == 1


def test_completion_event_includes_runtime_metrics():

    workflow, planner, critic = create_workflow()

    result = run_workflow(workflow)

    events = [
        call.args[0]
        for call in workflow.event_logger.log_event.call_args_list
    ]

    matching_events = [
        event for event in events
        if event.get("runtime_budget") == (
            result["runtime_budget"]
        )
    ]

    assert matching_events


# --------------------------------------------------
# Independent ten-step runtime boundary
# --------------------------------------------------


def test_hard_ten_step_limit_blocks_infinite_quality_loop(
    monkeypatch,
):

    import src.agents.workflow as workflow_module

    # Deliberately relax the quality-retry restriction
    # in this test only.
    monkeypatch.setattr(
        workflow_module,
        "MAX_RETRIES",
        100,
    )

    workflow, planner, critic = create_workflow(
        planner_responses=[
            with_usage(VALID_PLAN),
        ],
        critic_responses=[
            with_usage(REJECTED)
            for _ in range(20)
        ],
    )

    result = run_workflow(workflow)

    assert_budget_failure(
        result,
        "agent_steps",
        "critic",
    )

    budget = result["runtime_budget"]

    assert budget["max_agent_steps"] == 10
    assert budget["agent_steps"] == 10
    assert budget["remaining_steps"] == 0

    assert len(budget["step_history"]) == 10

    assert planner.invoke.call_count == 1
    assert workflow.executor.execute.call_count == 5
    assert critic.invoke.call_count == 4

    assert budget["llm_calls"] == 5
    assert budget["total_tokens"] == 750

    assert result["technical_retry_count"] == 0

    assert get_active_budget() is None


def test_tenth_agent_step_can_complete_successfully(
    monkeypatch,
):

    import src.agents.workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "MAX_RETRIES",
        100,
    )

    # One Planner retry, followed by three quality
    # corrections and final approval.
    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            with_usage(VALID_PLAN),
        ],
        critic_responses=[
            with_usage(REJECTED),
            with_usage(REJECTED),
            with_usage(REJECTED),
            with_usage(APPROVED),
        ],
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["is_approved"] is True

    assert result["budget_exhausted"] is False

    budget = result["runtime_budget"]

    assert budget["agent_steps"] == 10
    assert budget["remaining_steps"] == 0

    assert planner.invoke.call_count == 2
    assert critic.invoke.call_count == 4
    assert workflow.executor.execute.call_count == 4

    assert result["technical_retry_count"] == 1
    assert result["retry_count"] == 3


def test_hard_limit_failure_is_checkpointed(
    monkeypatch,
):

    import src.agents.workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "MAX_RETRIES",
        100,
    )

    workflow, planner, critic = create_workflow(
        planner_responses=[
            with_usage(VALID_PLAN),
        ],
        critic_responses=[
            with_usage(REJECTED)
            for _ in range(20)
        ],
    )

    thread_id = str(uuid.uuid4())

    result = run_workflow(
        workflow,
        thread_id,
    )

    snapshot = workflow.get_thread_state(thread_id)

    assert result["budget_exhausted"] is True

    assert snapshot.values["budget_exhausted"] is True

    assert snapshot.values[
        "budget_exhaustion_reason"
    ] == "agent_steps"

    assert snapshot.values["runtime_budget"][
        "agent_steps"
    ] == 10

    assert snapshot.values["is_approved"] is False

