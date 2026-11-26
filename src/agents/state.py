from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state passed between Planner, Executor and Critic."""

    # Original user request
    query: str
    has_file: bool
    file_path: str

    # Planner output
    plan: list[str]
    selected_tools: list[str]
    missing_information: list[str]

    # Executor output
    tool_results: dict[str, Any]
    retrieved_evidence: list[dict[str, Any]]

    # Critic output
    critic_feedback: str
    is_approved: bool
    retry_count: int

    # Final response
    final_answer: str
    status: str

    # Execution trace
    execution_log: list[dict[str, Any]]

    # Structured expense details supplied by the user or UI
    expense_details: dict[str, Any]

    # Metadata for observability and debugging
    thread_id: str
    execution_id: str

    # Additional fields can be added as needed for specific tools or workflows
    query: str
