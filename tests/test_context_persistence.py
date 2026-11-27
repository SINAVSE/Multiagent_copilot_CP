
import sqlite3
import uuid

from unittest.mock import Mock

import pytest

from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.workflow import EmployeeOpsWorkflow
from src.agents.query_resolver import QueryResolver


def create_test_workflow(database_path):

    workflow = EmployeeOpsWorkflow.__new__(
        EmployeeOpsWorkflow
    )

    workflow._app = None
    workflow.query_resolver = QueryResolver()
    workflow.event_logger = Mock()

    workflow.connection = sqlite3.connect(
        str(database_path),
        check_same_thread=False,
    )

    workflow.checkpointer = SqliteSaver(
        workflow.connection
    )

    workflow.checkpointer.setup()

    # Mock Planner.
    workflow.planner = Mock()

    workflow.planner.plan.side_effect = (
        lambda **kwargs: {
            "plan": ["Retrieve relevant policy"],
            "selected_tools": ["search_hr_policy"],
            "missing_information": [],
            "expense_details": kwargs.get(
                "expense_details", {}
            ),
        }
    )

    # Mock Executor.
    workflow.executor = Mock()

    workflow.executor.execute.return_value = {
        "search_hr_policy": {
            "answer": "Mock policy response.",
            "retrieved_evidence": [],
        }
    }

    # Mock Critic.
    workflow.critic = Mock()

    workflow.critic.review.return_value = {
        "approved": True,
        "issues": [],
        "feedback": "Approved mock response.",
    }

    return workflow


@pytest.fixture
def workflow(tmp_path):

    database_path = (
        tmp_path / "test_context.sqlite"
    )

    instance = create_test_workflow(
        database_path
    )

    try:
        yield instance
    finally:
        instance.close()


def test_context_saved_in_checkpoint(workflow):

    thread_id = str(uuid.uuid4())

    workflow.run(
        query="Can I claim $100 for a team dinner?",
        thread_id=thread_id,
    )

    snapshot = workflow.get_thread_state(
        thread_id
    )

    context = snapshot.values[
        "conversation_context"
    ]

    assert context["category"] == "team_dinner"
    assert context["claimed_amount"] == 100
    assert context["currency"] == "USD"


def test_context_inheritance_across_turns(workflow):

    thread_id = str(uuid.uuid4())

    workflow.run(
        query="Can I claim $100 for a team dinner?",
        thread_id=thread_id,
    )

    result = workflow.run(
        query="There were 3 employees.",
        thread_id=thread_id,
    )

    context = result["conversation_context"]

    assert context["category"] == "team_dinner"
    assert context["claimed_amount"] == 100
    assert context["headcount"] == 3

    assert result["context_resolution"][
        "context_used"
    ] is True


def test_amount_override_across_turns(workflow):

    thread_id = str(uuid.uuid4())

    workflow.run(
        query=(
            "Can I claim $100 for a team dinner "
            "for 3 employees?"
        ),
        thread_id=thread_id,
    )

    result = workflow.run(
        query="What about $150?",
        thread_id=thread_id,
    )

    context = result["conversation_context"]

    assert context["claimed_amount"] == 150
    assert context["headcount"] == 3
    assert context["currency"] == "USD"


def test_topic_switch_clears_context(workflow):

    thread_id = str(uuid.uuid4())

    workflow.run(
        query="Can I claim $100 for dinner?",
        thread_id=thread_id,
    )

    result = workflow.run(
        query="What is the annual leave policy?",
        thread_id=thread_id,
    )

    context = result["conversation_context"]

    assert context["topic"] == "leave"
    assert context["category"] == "annual_leave"

    assert "claimed_amount" not in context
    assert "currency" not in context

    assert result["context_resolution"][
        "context_used"
    ] is False


def test_thread_isolation(workflow):

    thread_one = str(uuid.uuid4())
    thread_two = str(uuid.uuid4())

    workflow.run(
        query="Can I claim $100 for dinner?",
        thread_id=thread_one,
    )

    workflow.run(
        query="Can I claim $500 for a hotel?",
        thread_id=thread_two,
    )

    state_one = workflow.get_thread_state(
        thread_one
    ).values

    state_two = workflow.get_thread_state(
        thread_two
    ).values

    assert state_one["conversation_context"][
        "claimed_amount"
    ] == 100

    assert state_two["conversation_context"][
        "claimed_amount"
    ] == 500

    assert state_one["conversation_context"][
        "category"
    ] == "dinner"

    assert state_two["conversation_context"][
        "category"
    ] == "hotel"


def test_context_survives_workflow_restart(tmp_path):

    database_path = (
        tmp_path / "restart_context.sqlite"
    )

    thread_id = str(uuid.uuid4())

    first = create_test_workflow(
        database_path
    )

    try:

        first.run(
            query=(
                "Can I claim $100 "
                "for a team dinner?"
            ),
            thread_id=thread_id,
        )

    finally:

        first.close()

    # Create a new workflow using the same
    # persisted SQLite database.

    second = create_test_workflow(
        database_path
    )

    try:

        result = second.run(
            query="There were 3 employees.",
            thread_id=thread_id,
        )

        context = result["conversation_context"]

        assert context["claimed_amount"] == 100
        assert context["headcount"] == 3
        assert context["category"] == "team_dinner"

    finally:

        second.close()


def test_assistant_history_not_inherited(workflow):

    result = workflow.run(
        query="Can I claim $100 for dinner?",
        thread_id=str(uuid.uuid4()),
        conversation_history=[
            {
                "role": "assistant",
                "content": (
                    "The previous meal allowance "
                    "was USD 9999."
                ),
            }
        ],
    )

    assert "9999" not in result["query"]

    assert result["conversation_context"][
        "claimed_amount"
    ] == 100


def test_new_thread_has_no_previous_context(workflow):

    workflow.run(
        query="Can I claim $100 for dinner?",
        thread_id=str(uuid.uuid4()),
    )

    result = workflow.run(
        query="What is the annual leave policy?",
        thread_id=str(uuid.uuid4()),
    )

    assert result["context_resolution"][
        "context_used"
    ] is False

    assert "claimed_amount" not in (
        result["conversation_context"]
    )


def test_inherited_expense_details_reach_executor(workflow):

    thread_id = str(uuid.uuid4())

    workflow.run(
        query="Can I claim $100 for a team dinner?",
        thread_id=thread_id,
    )

    workflow.run(
        query="There were 3 employees.",
        thread_id=thread_id,
    )

    details = workflow.executor.execute.call_args.kwargs[
        "expense_details"
    ]

    assert details["claimed_amount"] == 100
    assert details["currency"] == "USD"
    assert details["headcount"] == 3


def test_explicit_ui_details_are_preserved(workflow):

    result = workflow.run(
        query="Assess my team dinner expense.",
        expense_details={
            "claimed_amount": 150,
            "currency": "USD",
            "headcount": 4,
        },
        thread_id=str(uuid.uuid4()),
    )

    details = result["expense_details"]

    assert details["claimed_amount"] == 150
    assert details["currency"] == "USD"
    assert details["headcount"] == 4


def test_conflicting_ui_and_query_details(workflow):

    with pytest.raises(
        ValueError,
        match="Conflicting values",
    ):

        workflow.run(
            query="Can I claim $100 for dinner?",
            expense_details={
                "claimed_amount": 200,
                "currency": "USD",
                "headcount": 2,
            },
            thread_id=str(uuid.uuid4()),
        )

    workflow.planner.plan.assert_not_called()


def test_clarification_skips_all_agents(workflow):

    result = workflow.run(
        query="Can I claim the same amount?",
        thread_id=str(uuid.uuid4()),
    )

    assert result["status"] == "clarification_required"
    assert result["final_answer"]
    assert result["selected_tools"] == []
    assert result["retry_count"] == 0

    workflow.planner.plan.assert_not_called()
    workflow.executor.execute.assert_not_called()
    workflow.critic.review.assert_not_called()


def test_multiple_amounts_skip_agents(workflow):

    result = workflow.run(
        query="I spent $100 on meals and $200 on hotel.",
        thread_id=str(uuid.uuid4()),
    )

    assert result["status"] == "clarification_required"

    workflow.planner.plan.assert_not_called()
    workflow.executor.execute.assert_not_called()
    workflow.critic.review.assert_not_called()


def test_valid_request_uses_normal_workflow(workflow):

    result = workflow.run(
        query="What is the annual leave policy?",
        thread_id=str(uuid.uuid4()),
    )

    assert result["status"] == "approved"

    workflow.planner.plan.assert_called_once()
    workflow.executor.execute.assert_called_once()
    workflow.critic.review.assert_called_once()


def test_clarification_state_is_persisted(workflow):

    thread_id = str(uuid.uuid4())

    workflow.run(
        query="Can I claim the same amount?",
        thread_id=thread_id,
    )

    snapshot = workflow.get_thread_state(thread_id)

    assert snapshot.values["status"] == (
        "clarification_required"
    )

    assert snapshot.values["context_resolution"][
        "requires_clarification"
    ] is True
