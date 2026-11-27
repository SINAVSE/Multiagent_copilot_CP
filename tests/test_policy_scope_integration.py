
import sqlite3

from unittest.mock import Mock

import pytest

from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.workflow import EmployeeOpsWorkflow
from src.agents.query_resolver import QueryResolver
from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
)


@pytest.fixture
def workflow(tmp_path):

    instance = EmployeeOpsWorkflow.__new__(
        EmployeeOpsWorkflow
    )

    instance._app = None

    instance.planner = Mock()
    instance.executor = Mock()
    instance.critic = Mock()

    instance.query_resolver = QueryResolver()
    instance.event_logger = Mock()

    instance.connection = sqlite3.connect(
        str(tmp_path / "scope_test.sqlite"),
        check_same_thread=False,
    )

    instance.checkpointer = SqliteSaver(
        instance.connection
    )

    instance.checkpointer.setup()

    try:

        yield instance

    finally:

        instance.connection.close()


def assert_no_agent_calls(workflow):

    assert workflow.planner.mock_calls == []
    assert workflow.executor.mock_calls == []
    assert workflow.critic.mock_calls == []


def start_expense(workflow, thread_id="thread-1"):

    return workflow.run(
        query="Can I claim USD 100 for dinner?",
        organization="GitLab",
        employment_country="India",
        scoped_mode=True,
        thread_id=thread_id,
    )


def test_scope_persisted_in_checkpoint(workflow):

    result = start_expense(workflow)

    checkpoint = workflow.get_thread_state(
        "thread-1"
    ).values

    expected = {
        "organization": "GitLab",
        "employment_country": "IN",
    }

    assert result["policy_scope"] == expected
    assert checkpoint["policy_scope"] == expected
    assert checkpoint["scoped_mode"] is True

    assert checkpoint["conversation_context"][
        "claimed_amount"
    ] == 100.0

    assert result["is_approved"] is False

    assert_no_agent_calls(workflow)


def test_same_scope_follow_up(workflow):

    start_expense(workflow)

    result = workflow.run(
        query="What about 3 employees?",
        thread_id="thread-1",
    )

    context = result["conversation_context"]

    assert context["claimed_amount"] == 100.0
    assert context["currency"] == "USD"
    assert context["headcount"] == 3

    assert result["policy_scope"]["organization"] == (
        "GitLab"
    )

    assert result["scope_changed"] is False

    assert_no_agent_calls(workflow)


def test_organization_switch_isolation(workflow):

    start_expense(workflow)

    result = workflow.run(
        query="What is the dinner policy?",
        organization="Rice University",
        thread_id="thread-1",
    )

    assert result["scope_changed"] is True

    assert result["policy_scope"] == {
        "organization": "Rice University",
        "employment_country": None,
    }

    assert "claimed_amount" not in (
        result["conversation_context"]
    )

    assert "claimed_amount" not in (
        result["expense_details"]
    )

    assert_no_agent_calls(workflow)


def test_country_switch_isolation(workflow):

    start_expense(workflow)

    result = workflow.run(
        query="What is the dinner policy?",
        organization="GitLab",
        employment_country="Germany",
        thread_id="thread-1",
    )

    assert result["scope_changed"] is True

    assert result["policy_scope"][
        "employment_country"
    ] == "DE"

    assert "claimed_amount" not in (
        result["conversation_context"]
    )

    assert "claimed_amount" not in (
        result["expense_details"]
    )

    assert_no_agent_calls(workflow)


def test_separate_conversation_isolation(workflow):

    start_expense(workflow, "thread-a")

    result = workflow.run(
        query="What is the dinner policy?",
        organization="Rice University",
        scoped_mode=True,
        thread_id="thread-b",
    )

    assert result["policy_scope"]["organization"] == (
        "Rice University"
    )

    assert "claimed_amount" not in (
        result["conversation_context"]
    )

    original = workflow.get_thread_state(
        "thread-a"
    ).values

    assert original["policy_scope"]["organization"] == (
        "GitLab"
    )

    assert original["conversation_context"][
        "claimed_amount"
    ] == 100.0

    assert_no_agent_calls(workflow)


def test_conflicting_scope_blocks_agents(workflow):

    start_expense(workflow)

    result = workflow.run(
        query="Explain Rice University travel policy.",
        organization="GitLab",
        thread_id="thread-1",
    )

    assert result["status"] == "clarification_required"
    assert result["policy_scope"] is None
    assert result["conversation_context"] == {}
    assert result["tool_results"] == {}

    assert "conflicts" in result["final_answer"]

    assert_no_agent_calls(workflow)


def test_missing_scope_blocks_agents(workflow):

    result = workflow.run(
        query="What is the travel policy?",
        scoped_mode=True,
        thread_id="missing-scope",
    )

    assert result["status"] == "clarification_required"
    assert result["policy_scope"] is None

    assert "select your organization" in (
        result["final_answer"]
    )

    assert_no_agent_calls(workflow)


def test_scoped_mode_cannot_silently_downgrade(workflow):

    start_expense(workflow)

    result = workflow.run(
        query="What is the travel policy?",
        scoped_mode=False,
        thread_id="thread-1",
    )

    assert result["scoped_mode"] is True
    assert result["is_approved"] is False

    assert "not yet enabled" in result["final_answer"]

    assert_no_agent_calls(workflow)


def make_policy_result(
    scope,
    evidence_organization="GitLab",
):

    return {
        "search_hr_policy": {
            "policy_scope": scope.to_dict(),
            "answer": "Sample policy answer.",
            "insufficient_evidence": False,
            "retrieved_evidence": [
                {
                    "chunk_id": "TEST-001",
                    "organization": evidence_organization,
                    "text": "Sample policy evidence.",
                }
            ],
            "citations": [
                {
                    "chunk_id": "TEST-001",
                    "organization": evidence_organization,
                }
            ],
        }
    }


def make_state(scope):

    return {
        "scoped_mode": True,
        "policy_scope": scope.to_dict(),
        "query": "What is the travel policy?",
        "selected_tools": ["search_hr_policy"],
        "expense_details": {},
    }


def test_executor_rejects_cross_organization(workflow):

    scope = PolicyScope("GitLab", "India")

    workflow.executor.execute.return_value = (
        make_policy_result(
            scope,
            "Rice University",
        )
    )

    with pytest.raises(PolicyScopeError):

        workflow.executor_node(
            make_state(scope)
        )

    workflow.critic.review.assert_not_called()


def test_critic_rejects_cross_organization(workflow):

    scope = PolicyScope("GitLab", "India")

    state = make_state(scope)

    state["tool_results"] = make_policy_result(
        scope,
        "Rice University",
    )

    with pytest.raises(PolicyScopeError):

        workflow.critic_node(state)

    assert workflow.critic.mock_calls == []


def test_executor_rejects_missing_scope(workflow):

    state = {
        "scoped_mode": True,
        "policy_scope": None,
        "query": "What is the travel policy?",
        "selected_tools": ["search_hr_policy"],
    }

    with pytest.raises(PolicyScopeError):

        workflow.executor_node(state)

    workflow.executor.execute.assert_not_called()


def test_valid_executor_scope(workflow):

    scope = PolicyScope("GitLab", "India")

    workflow.executor.execute.return_value = (
        make_policy_result(scope)
    )

    result = workflow.executor_node(
        make_state(scope)
    )

    assert result["status"] == "executed"

    assert result["retrieved_evidence"][0][
        "organization"
    ] == "GitLab"

    assert workflow.executor.execute.call_args.kwargs[
        "policy_scope"
    ] == scope
