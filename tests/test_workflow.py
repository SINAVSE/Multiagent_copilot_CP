import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from src.observability.event_logger import EventLogger
from src.agents.workflow import EmployeeOpsWorkflow

class FakePlanner:
    def plan(self, query, has_file=False,expense_details=None,):
        return {
            "plan": ["Search the relevant HR policy"],
            "selected_tools": ["search_hr_policy"],
            "missing_information": [],
        }


class FakeExecutor:
    def __init__(self):
        self.feedback_received = []

    def execute(
        self,
        selected_tools,
        query,
        has_file=False,
        file_path=None,
        critic_feedback=None,
        expense_details=None,
    ):
        self.feedback_received.append(critic_feedback)

        return {
            "search_hr_policy": {
                "answer": "Test policy answer",
                "insufficient_evidence": False,
                "citations": [{"chunk_id": "TEST-001"}],
                "retrieved_evidence": [
                    {"chunk_id": "TEST-001", "text": "Test evidence"}
                ],
            }
        }


class FakeCritic:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def review(self, query, tool_results, final_answer=""):
        approved = next(self.decisions)

        return {
            "approved": approved,
            "issues": [] if approved else ["Insufficient explanation"],
            "feedback": "" if approved else "Please improve the explanation.",
        }


def create_test_workflow(critic_decisions):
    workflow = EmployeeOpsWorkflow.__new__(EmployeeOpsWorkflow)

    workflow._app = None

    workflow.planner = FakePlanner()
    workflow.executor = FakeExecutor()
    workflow.critic = FakeCritic(critic_decisions)

    # Use a real event logger for workflow execution.
    workflow.event_logger = EventLogger()

    # Tests do not need persistent SQLite checkpoints.
    workflow.checkpointer = None

    return workflow


def test_workflow_approved_on_first_attempt():
    workflow = create_test_workflow([True])

    result = workflow.run(query="Test policy question")

    assert result["is_approved"] is True
    assert result["status"] == "approved"
    assert result["retry_count"] == 0
    assert len(workflow.executor.feedback_received) == 1


def test_workflow_retries_and_then_gets_approved():
    workflow = create_test_workflow([False, True])

    result = workflow.run(query="Test policy question")

    assert result["is_approved"] is True
    assert result["status"] == "approved"
    assert result["retry_count"] == 1

    assert len(workflow.executor.feedback_received) == 2

    assert workflow.executor.feedback_received[0] is None
    assert workflow.executor.feedback_received[1] == (
        "Please improve the explanation."
    )


def test_workflow_stops_after_maximum_retries():
    workflow = create_test_workflow([False, False, False])

    result = workflow.run(query="Test policy question")

    assert result["is_approved"] is False
    assert result["status"] == "critic_rejected"
    assert result["retry_count"] == 3

    assert len(workflow.executor.feedback_received) == 3
    assert "could not produce an answer" in result["final_answer"]

    assert len(result["execution_log"]) == 3

def test_workflow_checkpoint_persists_across_instances(tmp_path):
    db_path = tmp_path / "test_checkpoints.sqlite"
    thread_id = "checkpoint-persistence-test"

    # First workflow instance
    workflow1 = EmployeeOpsWorkflow.__new__(EmployeeOpsWorkflow)

    workflow1._app = None
    workflow1.planner = FakePlanner()
    workflow1.executor = FakeExecutor()
    workflow1.critic = FakeCritic([True])
    workflow1.event_logger = EventLogger()

    workflow1.connection = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
    )
    workflow1.checkpointer = SqliteSaver(workflow1.connection)
    workflow1.checkpointer.setup()

    result = workflow1.run(
        query="Test checkpoint persistence",
        thread_id=thread_id,
    )

    assert result["status"] == "approved"

    workflow1.close()

    # Second workflow instance using the same database
    workflow2 = EmployeeOpsWorkflow.__new__(EmployeeOpsWorkflow)

    workflow2._app = None
    workflow2.planner = FakePlanner()
    workflow2.executor = FakeExecutor()
    workflow2.critic = FakeCritic([True])
    workflow2.event_logger = EventLogger()

    workflow2.connection = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
    )
    workflow2.checkpointer = SqliteSaver(workflow2.connection)
    workflow2.checkpointer.setup()

    saved_state = workflow2.get_thread_state(thread_id)

    assert saved_state is not None
    assert saved_state.values["status"] == "approved"
    assert saved_state.values["thread_id"] == thread_id

    workflow2.close()

def test_workflow_preserves_previous_checkpoint_after_failure(tmp_path):
    db_path = tmp_path / "failure_checkpoint.sqlite"
    thread_id = "failure-recovery-test"

    workflow = EmployeeOpsWorkflow.__new__(EmployeeOpsWorkflow)

    workflow._app = None
    workflow.planner = FakePlanner()
    workflow.executor = FakeExecutor()
    workflow.critic = FakeCritic([True])
    workflow.event_logger = EventLogger()

    workflow.connection = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
    )
    workflow.checkpointer = SqliteSaver(workflow.connection)
    workflow.checkpointer.setup()

    # First execution succeeds and saves a checkpoint.
    first_result = workflow.run(
        query="First successful request",
        thread_id=thread_id,
    )

    assert first_result["status"] == "approved"

    # Force the next execution to fail in the planner.
    def failing_plan(*args, **kwargs):
        raise RuntimeError("Simulated planner failure")

    workflow.planner.plan = failing_plan

    try:
        workflow.run(
            query="Second request that fails",
            thread_id=thread_id,
        )
        assert False, "Expected planner failure"
    except RuntimeError as exc:
        assert "Simulated planner failure" in str(exc)

    # Check that a checkpoint is still available.
    saved_state = workflow.get_thread_state(thread_id)

    assert saved_state is not None

    workflow.close()