import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

from langgraph.checkpoint.memory import MemorySaver

from src.agents.workflow import EmployeeOpsWorkflow
from src.agents.query_resolver import QueryResolver
from src.agents.planner import Planner
from src.agents.critic import Critic
from src.agents.errors import ToolExecutionError


QUERY = "What is the annual leave policy?"

VALID_PLAN = {
    "intent": "POLICY_QUERY",
    "plan": ["Search employee policy"],
    "selected_tools": ["search_hr_policy"],
    "missing_information": [],
}

APPROVED = {
    "approved": True,
    "issues": [],
    "feedback": "The policy answer is supported.",
}

REJECTED = {
    "approved": False,
    "issues": ["The answer requires correction."],
    "feedback": "Improve the policy explanation.",
}

TOOL_RESULTS = {
    "search_hr_policy": {
        "answer": "Mock annual leave policy response.",
        "insufficient_evidence": False,
        "citations": [{"chunk_id": "TEST-001"}],
        "retrieved_evidence": [
            {
                "chunk_id": "TEST-001",
                "text": "Mock annual leave policy evidence.",
            }
        ],
    }
}


def response(data):

    return SimpleNamespace(
        content=json.dumps(data)
    )


def invalid_response():

    return SimpleNamespace(
        content="Invalid JSON"
    )


def api_error(exception_type, status):

    request = httpx.Request(
        "POST",
        "https://example.invalid",
    )

    http_response = httpx.Response(
        status,
        request=request,
    )

    return exception_type(
        "SENSITIVE_TEST_MESSAGE",
        response=http_response,
        body=None,
    )


def create_workflow(
    planner_responses=None,
    critic_responses=None,
):

    workflow = EmployeeOpsWorkflow.__new__(
        EmployeeOpsWorkflow
    )

    workflow._app = None
    workflow.checkpointer = MemorySaver()
    workflow.query_resolver = QueryResolver()
    workflow.event_logger = Mock()

    planner_model = Mock()
    critic_model = Mock()

    planner_model.invoke.side_effect = (
        planner_responses
        if planner_responses is not None
        else [response(VALID_PLAN)]
    )

    critic_model.invoke.side_effect = (
        critic_responses
        if critic_responses is not None
        else [response(APPROVED)]
    )

    workflow.planner = Planner(model=planner_model)
    workflow.critic = Critic(model=critic_model)

    workflow.critic.validate_executor_output = Mock(
        return_value={
            "passed": True,
            "issues": [],
        }
    )

    workflow.executor = Mock()
    workflow.executor.execute.return_value = TOOL_RESULTS

    return workflow, planner_model, critic_model


def run_workflow(workflow, thread_id=None):

    return workflow.run(
        query=QUERY,
        thread_id=thread_id or str(uuid.uuid4()),
    )


def assert_safe_failure(result, node, code):

    assert result["status"] == "technical_failure"
    assert result["failed_node"] == node
    assert result["error_code"] == code
    assert result["is_approved"] is False

    assert result["tool_results"] == {}
    assert result["retrieved_evidence"] == []
    assert result["critic_feedback"] is None

    assert "No verified policy or expense decision" in (
        result["final_answer"]
    )

    assert "SENSITIVE_TEST_MESSAGE" not in str(result)


# --------------------------------------------------
# Normal execution
# --------------------------------------------------


def test_successful_workflow_has_no_technical_retries():

    workflow, planner, critic = create_workflow()

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["is_approved"] is True
    assert result["technical_retry_count"] == 0
    assert result["technical_errors"] == []
    assert result["error_code"] is None
    assert result["failed_node"] is None

    assert planner.invoke.call_count == 1
    assert critic.invoke.call_count == 1

    workflow.executor.execute.assert_called_once()


# --------------------------------------------------
# Planner recovery
# --------------------------------------------------


def test_planner_recovers_then_executes_workflow():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            response(VALID_PLAN),
        ]
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["technical_retry_count"] == 1

    assert planner.invoke.call_count == 2
    assert critic.invoke.call_count == 1

    workflow.executor.execute.assert_called_once()

    assert result["technical_errors"][0][
        "error_code"
    ] == "malformed_json"


def test_planner_retry_exhaustion_terminates_workflow():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            invalid_response(),
        ]
    )

    result = run_workflow(workflow)

    assert_safe_failure(
        result,
        "planner",
        "malformed_json",
    )

    assert result["technical_retry_count"] == 1
    assert planner.invoke.call_count == 2

    workflow.executor.execute.assert_not_called()
    critic.invoke.assert_not_called()


@pytest.mark.parametrize(
    "exception_type,status,code",
    [
        (
            openai.AuthenticationError,
            401,
            "authentication_failed",
        ),
        (
            openai.PermissionDeniedError,
            403,
            "authentication_failed",
        ),
        (
            openai.RateLimitError,
            429,
            "rate_limit_exceeded",
        ),
    ],
)
def test_planner_non_retryable_failure(
    exception_type,
    status,
    code,
):

    workflow, planner, critic = create_workflow(
        planner_responses=[
            api_error(exception_type, status)
        ]
    )

    result = run_workflow(workflow)

    assert_safe_failure(result, "planner", code)

    assert result["technical_retry_count"] == 0
    assert planner.invoke.call_count == 1

    workflow.executor.execute.assert_not_called()
    critic.invoke.assert_not_called()


def test_planner_timeout_recovery():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            TimeoutError("Planner timeout"),
            response(VALID_PLAN),
        ]
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["technical_retry_count"] == 1
    assert planner.invoke.call_count == 2
    assert critic.invoke.call_count == 1


# --------------------------------------------------
# Executor failures
# --------------------------------------------------


@pytest.mark.parametrize(
    "tool_name",
    [
        "search_hr_policy",
        "extract_expense_policy",
        "calculate_expense_eligibility",
        "generate_policy_answer",
    ],
)
def test_executor_failure_never_reaches_critic(
    tool_name,
):

    workflow, planner, critic = create_workflow()

    workflow.executor.execute.side_effect = (
        ToolExecutionError(tool_name)
    )

    result = run_workflow(workflow)

    assert_safe_failure(
        result,
        "executor",
        "tool_execution_failed",
    )

    assert result["technical_retry_count"] == 0

    workflow.executor.execute.assert_called_once()
    critic.invoke.assert_not_called()


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        [],
        {},
        {"search_hr_policy": None},
        {"search_hr_policy": "invalid"},
        {"search_hr_policy": {}},
    ],
)
def test_invalid_tool_results_fail_closed(
    invalid_result,
):

    workflow, planner, critic = create_workflow()

    workflow.executor.execute.return_value = (
        invalid_result
    )

    result = run_workflow(workflow)

    assert_safe_failure(
        result,
        "executor",
        "tool_execution_failed",
    )

    critic.invoke.assert_not_called()


# --------------------------------------------------
# Critic technical recovery
# --------------------------------------------------


def test_critic_recovers_without_repeating_executor():

    workflow, planner, critic = create_workflow(
        critic_responses=[
            invalid_response(),
            response(APPROVED),
        ]
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["technical_retry_count"] == 1
    assert result["retry_count"] == 0

    assert critic.invoke.call_count == 2

    workflow.executor.execute.assert_called_once()


def test_critic_exhaustion_clears_unverified_results():

    workflow, planner, critic = create_workflow(
        critic_responses=[
            invalid_response(),
            invalid_response(),
        ]
    )

    result = run_workflow(workflow)

    assert_safe_failure(
        result,
        "critic",
        "malformed_json",
    )

    assert result["technical_retry_count"] == 1
    assert result["retry_count"] == 0

    assert critic.invoke.call_count == 2

    workflow.executor.execute.assert_called_once()


def test_critic_authentication_failure_does_not_retry():

    workflow, planner, critic = create_workflow(
        critic_responses=[
            api_error(openai.AuthenticationError, 401)
        ]
    )

    result = run_workflow(workflow)

    assert_safe_failure(
        result,
        "critic",
        "authentication_failed",
    )

    assert result["technical_retry_count"] == 0
    assert critic.invoke.call_count == 1

    workflow.executor.execute.assert_called_once()


def test_critic_rate_limit_does_not_repeat_executor():

    workflow, planner, critic = create_workflow(
        critic_responses=[
            api_error(openai.RateLimitError, 429)
        ]
    )

    result = run_workflow(workflow)

    assert_safe_failure(
        result,
        "critic",
        "rate_limit_exceeded",
    )

    assert result["technical_retry_count"] == 0
    assert critic.invoke.call_count == 1

    workflow.executor.execute.assert_called_once()


# --------------------------------------------------
# Quality retry and technical retry separation
# --------------------------------------------------


def test_quality_rejection_preserves_existing_retry_loop():

    workflow, planner, critic = create_workflow(
        critic_responses=[
            response(REJECTED),
            response(APPROVED),
        ]
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["retry_count"] == 1
    assert result["technical_retry_count"] == 0

    assert workflow.executor.execute.call_count == 2
    assert critic.invoke.call_count == 2


def test_quality_rejection_stops_after_existing_limit():

    workflow, planner, critic = create_workflow(
        critic_responses=[
            response(REJECTED),
            response(REJECTED),
            response(REJECTED),
        ]
    )

    result = run_workflow(workflow)

    assert result["status"] == "critic_rejected"
    assert result["is_approved"] is False
    assert result["retry_count"] == 3
    assert result["technical_retry_count"] == 0

    assert workflow.executor.execute.call_count == 3


# --------------------------------------------------
# Shared technical retry budget
# --------------------------------------------------


def test_planner_and_critic_share_retry_budget():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            response(VALID_PLAN),
        ],
        critic_responses=[
            invalid_response(),
            response(APPROVED),
        ],
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"
    assert result["technical_retry_count"] == 2

    assert planner.invoke.call_count == 2
    assert critic.invoke.call_count == 2

    assert len(result["technical_errors"]) == 2


def test_shared_budget_exhaustion_across_quality_loop():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            response(VALID_PLAN),
        ],
        critic_responses=[
            invalid_response(),
            response(REJECTED),
            invalid_response(),
            response(APPROVED),
        ],
    )

    result = run_workflow(workflow)

    assert_safe_failure(
        result,
        "critic",
        "malformed_json",
    )

    assert result["technical_retry_count"] == 2
    assert result["retry_count"] == 1

    assert planner.invoke.call_count == 2
    assert critic.invoke.call_count == 3

    assert workflow.executor.execute.call_count == 2

    assert result["technical_errors"][-1][
        "will_retry"
    ] is False


# --------------------------------------------------
# State reset and persistence
# --------------------------------------------------


def test_retry_budget_resets_for_next_request():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            response(VALID_PLAN),
            response(VALID_PLAN),
        ],
        critic_responses=[
            invalid_response(),
            response(APPROVED),
            response(APPROVED),
        ],
    )

    thread_id = str(uuid.uuid4())

    first = run_workflow(workflow, thread_id)

    assert first["technical_retry_count"] == 2

    second = run_workflow(workflow, thread_id)

    assert second["status"] == "approved"
    assert second["technical_retry_count"] == 0
    assert second["technical_errors"] == []
    assert second["error_code"] is None
    assert second["failed_node"] is None


def test_terminal_failure_is_persisted_in_checkpoint():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            invalid_response(),
        ]
    )

    thread_id = str(uuid.uuid4())

    run_workflow(workflow, thread_id)

    snapshot = workflow.get_thread_state(thread_id)

    assert snapshot.values["status"] == "technical_failure"
    assert snapshot.values["failed_node"] == "planner"
    assert snapshot.values["error_code"] == "malformed_json"
    assert snapshot.values["technical_retry_count"] == 1

    assert snapshot.values["is_approved"] is False


def test_failure_does_not_contaminate_next_request():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            invalid_response(),
            invalid_response(),
            response(VALID_PLAN),
        ]
    )

    thread_id = str(uuid.uuid4())

    first = run_workflow(workflow, thread_id)

    assert first["status"] == "technical_failure"

    second = run_workflow(workflow, thread_id)

    assert second["status"] == "approved"
    assert second["error_code"] is None
    assert second["failed_node"] is None
    assert second["technical_retry_count"] == 0
    assert second["technical_errors"] == []


def test_concurrent_requests_have_independent_budgets():

    def execute_request():

        workflow, planner, critic = create_workflow(
            planner_responses=[
                invalid_response(),
                response(VALID_PLAN),
            ],
            critic_responses=[
                invalid_response(),
                response(APPROVED),
            ],
        )

        return run_workflow(workflow)

    with ThreadPoolExecutor(max_workers=2) as pool:

        results = list(
            pool.map(
                lambda _: execute_request(),
                range(2),
            )
        )

    for result in results:

        assert result["status"] == "approved"
        assert result["technical_retry_count"] == 2


# --------------------------------------------------
# Error observability
# --------------------------------------------------


def test_recovery_events_contain_safe_metadata():

    workflow, planner, critic = create_workflow(
        planner_responses=[
            TimeoutError("SENSITIVE_TEST_MESSAGE"),
            response(VALID_PLAN),
        ]
    )

    result = run_workflow(workflow)

    assert result["status"] == "approved"

    events = [
        call.args[0]
        for call in workflow.event_logger.log_event.call_args_list
    ]

    recovery_events = [
        event for event in events
        if event.get("event") == "agent_recovery"
    ]

    assert len(recovery_events) == 1

    event = recovery_events[0]

    assert event["agent"] == "planner"
    assert event["error_code"] == "api_timeout"
    assert event["technical_retry_count"] == 1
    assert event["will_retry"] is True

    assert event["thread_id"]
    assert event["execution_id"]

    assert "SENSITIVE_TEST_MESSAGE" not in str(events)


def test_terminal_failure_event_is_logged():

    workflow, planner, critic = create_workflow()

    workflow.executor.execute.side_effect = (
        ToolExecutionError("search_hr_policy")
    )

    result = run_workflow(workflow)

    assert result["status"] == "technical_failure"

    events = [
        call.args[0]
        for call in workflow.event_logger.log_event.call_args_list
    ]

    failures = [
        event for event in events
        if event.get("event") == "technical_failure"
    ]

    assert len(failures) == 1

    event = failures[0]

    assert event["node"] == "executor"
    assert event["error_code"] == "tool_execution_failed"
    assert event["tool_name"] == "search_hr_policy"
