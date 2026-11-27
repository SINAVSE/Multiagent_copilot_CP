import uuid
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.agents.workflow import EmployeeOpsWorkflow
from src.agents.query_resolver import QueryResolver


@pytest.fixture
def workflow():

    instance = EmployeeOpsWorkflow.__new__(
        EmployeeOpsWorkflow
    )

    instance._app = None
    instance.query_resolver = QueryResolver()
    instance.checkpointer = MemorySaver()
    instance.event_logger = Mock()

    instance.planner = Mock()
    instance.executor = Mock()
    instance.critic = Mock()

    instance.executor.execute.return_value = {
        "search_hr_policy": {
            "answer": "Mock employee policy response.",
            "citations": [
                {"chunk_id": "TEST-001"}
            ],
            "retrieved_evidence": [
                {
                    "chunk_id": "TEST-001",
                    "text": "Mock policy evidence",
                }
            ],
        }
    }

    instance.critic.review.return_value = {
        "approved": True,
        "issues": [],
        "feedback": "",
    }

    return instance


def configure_planner(
    workflow,
    intent,
    tools=None,
    missing=None,
):

    workflow.planner.plan.side_effect = (
        lambda **kwargs: {
            "intent": intent,
            "plan": ["Process employee request"],
            "selected_tools": tools or [],
            "missing_information": missing or [],
            "expense_details": kwargs.get(
                "expense_details", {}
            ),
        }
    )


def assert_direct_response(result):

    assert result["final_answer"]
    assert result["selected_tools"] == []
    assert result["tool_results"] == {}
    assert result["retrieved_evidence"] == []
    assert result["retry_count"] == 0
    assert result["is_approved"] is False
    assert result["critic_feedback"] is None


# --------------------------------------------------
# Direct response routing
# --------------------------------------------------


def test_greeting_bypasses_business_agents(workflow):

    configure_planner(
        workflow,
        "GREETING",
    )

    result = workflow.run(
        query="Hello",
        thread_id=str(uuid.uuid4()),
    )

    assert result["intent"] == "GREETING"
    assert result["status"] == "greeting"
    assert "hello" in result["final_answer"].lower()

    assert_direct_response(result)

    workflow.planner.plan.assert_called_once()
    workflow.executor.execute.assert_not_called()
    workflow.critic.review.assert_not_called()


def test_out_of_scope_returns_explicit_refusal(workflow):

    configure_planner(
        workflow,
        "OUT_OF_SCOPE",
    )

    result = workflow.run(
        query="What is today's weather?",
        thread_id=str(uuid.uuid4()),
    )

    assert result["intent"] == "OUT_OF_SCOPE"
    assert result["status"] == "out_of_scope"

    assert "outside my scope" in (
        result["final_answer"].lower()
    )

    assert_direct_response(result)

    workflow.planner.plan.assert_called_once()
    workflow.executor.execute.assert_not_called()
    workflow.critic.review.assert_not_called()


def test_planner_clarification_bypasses_agents(workflow):

    configure_planner(
        workflow,
        "CLARIFICATION_REQUIRED",
        missing=["Expense category", "Claimed amount"],
    )

    result = workflow.run(
        query="Please assess my expense.",
        thread_id=str(uuid.uuid4()),
    )

    assert result["intent"] == (
        "CLARIFICATION_REQUIRED"
    )

    assert result["status"] == (
        "clarification_required"
    )

    assert "Expense category" in result["final_answer"]
    assert "Claimed amount" in result["final_answer"]

    assert_direct_response(result)

    workflow.planner.plan.assert_called_once()
    workflow.executor.execute.assert_not_called()
    workflow.critic.review.assert_not_called()


def test_missing_receipt_returns_clarification(workflow):

    configure_planner(
        workflow,
        "CLARIFICATION_REQUIRED",
        missing=["Upload your receipt"],
    )

    result = workflow.run(
        query="Analyze my receipt",
        has_file=False,
        thread_id=str(uuid.uuid4()),
    )

    assert result["status"] == (
        "clarification_required"
    )

    assert "Upload your receipt" in (
        result["final_answer"]
    )

    workflow.planner.plan.assert_called_once()
    workflow.executor.execute.assert_not_called()
    workflow.critic.review.assert_not_called()


# --------------------------------------------------
# Business request routing
# --------------------------------------------------


@pytest.mark.parametrize(
    "intent,query,tools,has_file",
    [
        (
            "POLICY_QUERY",
            "What is the annual leave policy?",
            ["search_hr_policy"],
            False,
        ),
        (
            "EXPENSE_ELIGIBILITY",
            "Can I claim $100 for dinner?",
            [
                "search_hr_policy",
                "calculate_expense_eligibility",
            ],
            False,
        ),
        (
            "RECEIPT_ANALYSIS",
            "Analyze my uploaded receipt",
            ["extract_bill_details"],
            True,
        ),
    ],
)
def test_business_intents_use_executor_and_critic(
    workflow,
    intent,
    query,
    tools,
    has_file,
):

    configure_planner(
        workflow,
        intent,
        tools=tools,
    )

    result = workflow.run(
        query=query,
        has_file=has_file,
        file_path=(
            "mock_receipt.pdf" if has_file else None
        ),
        thread_id=str(uuid.uuid4()),
    )

    assert result["intent"] == intent
    assert result["status"] == "approved"
    assert result["is_approved"] is True
    assert result["selected_tools"] == tools

    workflow.planner.plan.assert_called_once()
    workflow.executor.execute.assert_called_once()
    workflow.critic.review.assert_called_once()

    assert (
        workflow.executor.execute.call_args.kwargs[
            "selected_tools"
        ] == tools
    )


# --------------------------------------------------
# Fail-closed routing
# --------------------------------------------------


@pytest.mark.parametrize(
    "intent",
    ["UNKNOWN_INTENT", ""],
)
def test_invalid_intent_never_executes_tools(
    workflow,
    intent,
):

    configure_planner(
        workflow,
        intent,
        tools=["search_hr_policy"],
    )

    with pytest.raises(
        ValueError,
        match="Unsupported or missing Planner intent",
    ):
        workflow.run(
            query="What is the travel policy?",
            thread_id=str(uuid.uuid4()),
        )

    workflow.executor.execute.assert_not_called()
    workflow.critic.review.assert_not_called()


# --------------------------------------------------
# CAP-08 routing compatibility
# --------------------------------------------------


def test_resolver_clarification_bypasses_planner(workflow):

    result = workflow.run(
        query="Can I claim the same amount?",
        thread_id=str(uuid.uuid4()),
    )

    assert result["intent"] == (
        "CLARIFICATION_REQUIRED"
    )

    assert result["status"] == (
        "clarification_required"
    )

    assert_direct_response(result)

    workflow.planner.plan.assert_not_called()
    workflow.executor.execute.assert_not_called()
    workflow.critic.review.assert_not_called()


# --------------------------------------------------
# Persistent state integrity
# --------------------------------------------------


def test_greeting_clears_previous_business_results(
    workflow,
):

    thread_id = str(uuid.uuid4())

    configure_planner(
        workflow,
        "POLICY_QUERY",
        tools=["search_hr_policy"],
    )

    first = workflow.run(
        query="What is the annual leave policy?",
        thread_id=thread_id,
    )

    assert first["is_approved"] is True
    assert first["tool_results"]
    assert first["retrieved_evidence"]

    workflow.executor.execute.reset_mock()
    workflow.critic.review.reset_mock()

    configure_planner(
        workflow,
        "GREETING",
    )

    second = workflow.run(
        query="Hello",
        thread_id=thread_id,
    )

    assert second["intent"] == "GREETING"
    assert second["status"] == "greeting"

    assert_direct_response(second)

    workflow.executor.execute.assert_not_called()
    workflow.critic.review.assert_not_called()


def test_direct_response_state_is_persisted(workflow):

    thread_id = str(uuid.uuid4())

    configure_planner(
        workflow,
        "OUT_OF_SCOPE",
    )

    workflow.run(
        query="What is today's weather?",
        thread_id=thread_id,
    )

    snapshot = workflow.get_thread_state(
        thread_id
    )

    assert snapshot.values["intent"] == (
        "OUT_OF_SCOPE"
    )

    assert snapshot.values["status"] == (
        "out_of_scope"
    )

    assert snapshot.values["tool_results"] == {}

    assert snapshot.values["is_approved"] is False


# --------------------------------------------------
# Routing observability
# --------------------------------------------------


@pytest.mark.parametrize(
    "intent,target",
    [
        ("GREETING", "greeting"),
        ("OUT_OF_SCOPE", "out_of_scope"),
        (
            "CLARIFICATION_REQUIRED",
            "planner_clarification",
        ),
        ("POLICY_QUERY", "executor"),
        ("EXPENSE_ELIGIBILITY", "executor"),
        ("RECEIPT_ANALYSIS", "executor"),
    ],
)
def test_intent_routing_event_is_logged(
    workflow,
    intent,
    target,
):

    state = {
        "intent": intent,
        "thread_id": "test-thread",
        "execution_id": "test-execution",
    }

    actual = workflow.route_after_planner(state)

    assert actual == target

    workflow.event_logger.log_event.assert_called_once_with({
        "event": "intent_routed",
        "thread_id": "test-thread",
        "execution_id": "test-execution",
        "intent": intent,
        "target_node": target,
    })
