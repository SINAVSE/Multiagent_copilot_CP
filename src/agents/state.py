
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state passed between Planner, Executor and Critic."""

    # Current employee request
    query: str
    user_query: str

    # Previous conversation, kept separate from the current request
    conversation_context: str

    # Uploaded file
    has_file: bool
    file_path: str

    # Planner output
    plan: list[str]
    selected_tools: list[str]
    missing_information: list[str]

    # Structured expense details
    expense_details: dict[str, Any]

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

    # Metadata
    thread_id: str
    execution_id: str