
from src.agents.workflow import EmployeeOpsWorkflow


def make_workflow():

    return EmployeeOpsWorkflow.__new__(
        EmployeeOpsWorkflow
    )


def test_legacy_route_unchanged():

    workflow = make_workflow()

    state = {
        "context_resolution": {},
        "scoped_mode": False,
    }

    assert workflow.route_initial_request(
        state
    ) == "planner"


def test_legacy_route_without_scope_field():

    workflow = make_workflow()

    assert workflow.route_initial_request({
        "context_resolution": {}
    }) == "planner"


def test_scoped_execution_blocked_until_ready():

    workflow = make_workflow()

    state = {
        "scoped_mode": True,
        "policy_scope": {
            "organization": "Rice University",
            "employment_country": "US",
        },
        "context_resolution": {
            "requires_clarification": False,
        },
    }

    assert workflow.route_initial_request(
        state
    ) == "clarification"

    result = workflow.clarification_node(state)

    assert result["is_approved"] is False
    assert result["tool_results"] == {}
    assert result["selected_tools"] == []
    assert result["retrieved_evidence"] == []

    assert "not yet enabled" in result["final_answer"]


def test_missing_organization_clarification():

    workflow = make_workflow()

    state = {
        "scoped_mode": True,
        "policy_scope": None,
        "context_resolution": {
            "requires_clarification": True,
            "clarification_message": (
                "Please select your organization."
            ),
        },
    }

    assert workflow.route_initial_request(
        state
    ) == "clarification"

    result = workflow.clarification_node(state)

    assert result["final_answer"] == (
        "Please select your organization."
    )

    assert result["is_approved"] is False


def test_existing_expense_clarification_unchanged():

    workflow = make_workflow()

    state = {
        "scoped_mode": False,
        "context_resolution": {
            "requires_clarification": True,
            "clarification_message": (
                "Please provide the expense amount."
            ),
        },
    }

    assert workflow.route_initial_request(
        state
    ) == "clarification"

    result = workflow.clarification_node(state)

    assert result["final_answer"] == (
        "Please provide the expense amount."
    )


def test_scoped_conflict_has_no_tool_results():

    workflow = make_workflow()

    state = {
        "scoped_mode": True,
        "context_resolution": {
            "requires_clarification": True,
            "clarification_message": (
                "The selected organization conflicts "
                "with the organization in your question."
            ),
        },
    }

    result = workflow.clarification_node(state)

    assert result["selected_tools"] == []
    assert result["tool_results"] == {}
    assert result["retrieved_evidence"] == []
    assert result["is_approved"] is False
