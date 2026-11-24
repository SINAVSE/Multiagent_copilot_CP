from langgraph.graph import StateGraph, START, END
from src.agents.state import AgentState
from src.agents.planner import Planner
from src.agents.executor import Executor
from src.agents.critic import Critic

MAX_RETRIES = 2

class EmployeeOpsWorkflow:

    def __init__(self):
        self.planner = Planner()
        self.executor = Executor()
        self.critic = Critic()

    def planner_node(self, state: AgentState) -> dict:

        plan_result = self.planner.plan(
            query=state["query"],
            has_file=state.get("has_file", False)
        )

        return {
            "plan": plan_result["plan"],
            "selected_tools": plan_result["selected_tools"],
            "missing_information": plan_result.get(
                "missing_information", []
            ),
            "status": "planned"
        }

    def executor_node(self, state: AgentState) -> dict:

        tool_results = self.executor.execute(
            selected_tools=state.get("selected_tools", []),
            query=state["query"],
            has_file=state.get("has_file", False),
            file_path=state.get("file_path"),
            critic_feedback=state.get("critic_feedback"),
            expense_details=state.get("expense_details"),
        )

        # Collect retrieved policy evidence
        retrieved_evidence = []

        for result in tool_results.values():
            retrieved_evidence.extend(
                result.get("retrieved_evidence", [])
            )

        # Collect generated answer
        final_answer = ""

        for result in tool_results.values():
            if result.get("answer"):
                final_answer = result["answer"]
                break

        return {
            "tool_results": tool_results,
            "retrieved_evidence": retrieved_evidence,
            "final_answer": final_answer,
            "status": "executed"
        }


    def critic_node(self, state: AgentState) -> dict:

        critic_result = self.critic.review(
            query=state["query"],
            tool_results=state.get("tool_results", {})
        )

        approved = critic_result["approved"]
        retry_count = state.get("retry_count", 0)

        if not approved:
            retry_count += 1

        execution_log = state.get("execution_log", [])

        execution_log.append({
            "agent": "critic",
            "approved": approved,
            "issues": critic_result.get("issues", []),
            "feedback": critic_result.get("feedback", "")
        })

        return {
            "is_approved": approved,
            "critic_feedback": critic_result.get("feedback", ""),
            "retry_count": retry_count,
            "status": "approved" if approved else "rejected",
            "execution_log": execution_log
        }

    def route_after_critic(self, state: AgentState) -> str:
        if state.get("is_approved", False):
            return "approved"

        if state.get("retry_count", 0) <= MAX_RETRIES:
            return "retry"

        return "failure"

    def failure_node(self, state: AgentState) -> dict:

        safe_message = (
            "I could not produce an answer that passed the "
            "evidence review after the permitted attempts. "
            "Please review the cited policy evidence or refer "
            "the question to the appropriate HR team."
        )

        return {
            "final_answer": safe_message,
            "is_approved": False,
            "status": "critic_rejected",
        }


    def build(self):

        workflow = StateGraph(AgentState)

        # Register agent nodes
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("critic", self.critic_node)
        workflow.add_node("failure", self.failure_node)


        # Define execution sequence
        workflow.set_entry_point("planner")

        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "critic")
        workflow.add_conditional_edges(
            "critic",
            self.route_after_critic,
            {
                "approved": END,
                "retry": "executor",
                "failure": "failure",
            },
        )
        workflow.add_edge("failure", END)

        return workflow.compile()

    
    def run(
        self,
        query: str,
        has_file: bool = False,
        file_path: str | None = None,
        expense_details: dict | None = None,
    ) -> dict:

        app = self.build()

        initial_state: AgentState = {
            "query": query,
            "has_file": has_file,
            "file_path": file_path,
            "expense_details": expense_details or {},
            "retry_count": 0,
            "execution_log": [],
            "status": "started"
        }

        return app.invoke(initial_state)