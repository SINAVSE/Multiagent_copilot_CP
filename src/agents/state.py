
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state passed between Planner, Executor and Critic."""

    # Current and resolved user requests
    user_query: str
    query: str
    has_file: bool
    file_path: str | None

    # CAP-08: Persistent conversational context
    conversation_context: dict[str, Any]
    context_resolution: dict[str, Any]

    # Planner output
    intent: str
    plan: list[str]
    selected_tools: list[str]
    missing_information: list[str]

    # Executor output
    tool_results: dict[str, Any]
    retrieved_evidence: list[dict[str, Any]]

    # Critic output
    critic_feedback: str | None
    is_approved: bool
    retry_count: int

    # CAP-11: Technical recovery state
    technical_retry_count: int
    technical_retry_limit: int
    technical_errors: list[dict[str, Any]]
    error_code: str | None
    failed_node: str | None

    # Final response
    final_answer: str
    status: str

    # Execution trace
    execution_log: list[dict[str, Any]]

    # Structured expense details
    expense_details: dict[str, Any]

    # Metadata
    thread_id: str
    execution_id: str
