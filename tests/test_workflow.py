from src.agents.workflow import EmployeeOpsWorkflow


class FakePlanner:
    def plan(self, query, has_file=False):
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

    def review(self, query, tool_results):
        approved = next(self.decisions)

        return {
            "approved": approved,
            "issues": [] if approved else ["Insufficient explanation"],
            "feedback": "" if approved else "Please improve the explanation.",
        }


def create_test_workflow(critic_decisions):
    workflow = EmployeeOpsWorkflow.__new__(EmployeeOpsWorkflow)

    workflow.planner = FakePlanner()
    workflow.executor = FakeExecutor()
    workflow.critic = FakeCritic(critic_decisions)

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
