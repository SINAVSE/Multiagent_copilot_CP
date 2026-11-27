import sqlite3
import time
import uuid

from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.state import AgentState

from src.agents.runtime_budget import (
    RuntimeBudget,
    BudgetExceededError,
    budget_scope,
    get_active_budget,
)

from src.agents.errors import (
    classify_error,
    ToolExecutionError,
)
from src.agents.planner import Planner
from src.agents.executor import Executor
from src.agents.critic import Critic
from src.agents.query_resolver import QueryResolver
from src.observability.event_logger import EventLogger


MAX_RETRIES = 2
MAX_TECHNICAL_RETRIES = 2


class EmployeeOpsWorkflow:

    def __init__(self):
        self._app = None
        self.planner = Planner()
        self.executor = Executor()
        self.critic = Critic()
        self.query_resolver = QueryResolver()

        self.event_logger = EventLogger()

        self.project_root = Path(__file__).resolve().parents[2]

        self.checkpoint_dir = (
            self.project_root / "data" / "checkpoints"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.database_path = (
            self.checkpoint_dir / "employee_ops.sqlite"
        )

        self.connection = sqlite3.connect(
            str(self.database_path),
            check_same_thread=False,
        )

        self.checkpointer = SqliteSaver(self.connection)
        self.checkpointer.setup()

        self._app = None

    # --------------------------------------------------
    # Observability wrapper
    # --------------------------------------------------

    def budget_failure_response(
        self,
        state: AgentState,
        node_name: str,
        exc: BudgetExceededError,
    ) -> dict:

        budget = get_active_budget()

        metrics = (
            budget.snapshot()
            if budget is not None
            else state.get("runtime_budget", {})
        )

        self.event_logger.log_event({
            "event": "budget_exhausted",
            "node": node_name,
            "thread_id": state.get("thread_id"),
            "execution_id": state.get("execution_id"),
            "budget_type": exc.budget_type,
            "runtime_budget": metrics,
        })

        reason = (
            "The maximum permitted agent execution "
            "steps have been reached."
            if exc.budget_type == "agent_steps"
            else
            "The permitted request token budget "
            "has been exhausted."
        )

        return {
            "status": "technical_failure",
            "final_answer": (
                reason
                + " No verified policy or expense "
                "decision has been established. "
                "Please try a more focused request."
            ),
            "error_code": "budget_exhausted",
            "failed_node": node_name,
            "budget_exhausted": True,
            "budget_exhaustion_reason": exc.budget_type,
            "runtime_budget": metrics,
            "is_approved": False,
            "critic_feedback": None,
            "tool_results": {},
            "retrieved_evidence": [],
        }


    def technical_failure_response(
        self,
        state: AgentState,
        node_name: str,
        exc: Exception,
    ) -> dict:

        classification = classify_error(exc)

        self.event_logger.log_event({
            "event": "technical_failure",
            "node": node_name,
            "thread_id": state.get("thread_id"),
            "execution_id": state.get("execution_id"),
            "error_code": classification.code.value,
            "error_type": type(exc).__name__,
            "tool_name": (
                exc.tool_name
                if isinstance(exc, ToolExecutionError)
                else None
            ),
        })

        return {
            "status": "technical_failure",
            "final_answer": (
                classification.safe_message
                + " No verified policy or expense decision "
                "has been established."
            ),
            "error_code": classification.code.value,
            "failed_node": node_name,
            "is_approved": False,
            "critic_feedback": None,
            "tool_results": {},
            "retrieved_evidence": [],
        }


    def _track_node(self, node_name, node_function):

        def tracked_node(state: AgentState):

            start_time = time.perf_counter()

            thread_id = state.get("thread_id", "unknown")
            execution_id = state.get("execution_id", "unknown")

            self.event_logger.log_event({
                "event": "node_started",
                "node": node_name,
                "thread_id": thread_id,
                "execution_id": execution_id,
            })

            try:

                budget = get_active_budget()

                if budget is not None:

                    if node_name in {
                        "planner",
                        "executor",
                        "critic",
                    }:

                        budget.consume_step(
                            node_name
                        )

                result = node_function(state)

                if budget is not None:

                    result["runtime_budget"] = (
                        budget.snapshot()
                    )

                duration_ms = round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                )

                self.event_logger.log_event({
                    "event": "node_completed",
                    "node": node_name,
                    "thread_id": thread_id,
                    "execution_id": execution_id,
                    "duration_ms": duration_ms,
                    "status": result.get("status", "completed"),
                })

                return result

            except BudgetExceededError as exc:

                return self.budget_failure_response(
                    state,
                    node_name,
                    exc,
                )

            except Exception as exc:

                duration_ms = round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                )

                self.event_logger.log_event({
                    "event": "node_failed",
                    "node": node_name,
                    "thread_id": thread_id,
                    "execution_id": execution_id,
                    "duration_ms": duration_ms,
                    "error_type": type(exc).__name__,
                })

                if node_name in {
                    "planner", "executor", "critic"
                }:
                    result = self.technical_failure_response(
                        state,
                        node_name,
                        exc,
                    )

                    errors = list(
                        state.get("technical_errors", [])
                    )

                    classification = classify_error(exc)

                    errors.append({
                        "agent": node_name,
                        "error_code": classification.code.value,
                        "error_type": type(exc).__name__,
                        "will_retry": False,
                    })

                    result["technical_errors"] = errors

                    budget = get_active_budget()

                    if budget is not None:

                        result["runtime_budget"] = (
                            budget.snapshot()
                        )

                    return result

                raise

        return tracked_node

    # --------------------------------------------------
    # Planner
    # --------------------------------------------------

    def invoke_with_recovery(
        self,
        state: AgentState,
        agent,
        recovery_method: str,
        legacy_method: str,
        **kwargs,
    ) -> tuple:

        retry_count = state.get(
            "technical_retry_count", 0
        )

        retry_limit = state.get(
            "technical_retry_limit",
            MAX_TECHNICAL_RETRIES,
        )

        errors = list(
            state.get("technical_errors", [])
        )

        initial_error_count = len(errors)

        def consume_retry():
            nonlocal retry_count

            if retry_count >= retry_limit:
                return False

            budget = get_active_budget()

            if budget is not None:

                budget.consume_step(
                    legacy_method + "_retry"
                )

            retry_count += 1
            return True

        def on_error(event):

            errors.append(dict(event))

            self.event_logger.log_event({
                "event": "agent_recovery",
                "thread_id": state.get("thread_id"),
                "execution_id": state.get("execution_id"),
                "technical_retry_count": retry_count,
                **event,
            })

        try:
            # Real agents expose recovery methods.
            # Older fake and mocked agents retain their legacy API.
            recovery = getattr(
                type(agent),
                recovery_method,
                None,
            )

            if callable(recovery):

                result = getattr(
                    agent, recovery_method
                )(
                    **kwargs,
                    on_error=on_error,
                    retry_allowed=consume_retry,
                )

            else:

                result = getattr(
                    agent, legacy_method
                )(**kwargs)

            return result, {
                "technical_retry_count": retry_count,
                "technical_errors": errors,
            }, None

        except BudgetExceededError as exc:

            return None, {
                "technical_retry_count": retry_count,
                "technical_errors": errors,
            }, exc

        except Exception as exc:

            if (
                len(errors) == initial_error_count
                or errors[-1].get("will_retry") is not False
            ):

                classification = classify_error(exc)

                on_error({
                    "agent": legacy_method,
                    "error_code": classification.code.value,
                    "error_type": type(exc).__name__,
                    "will_retry": False,
                })

            return None, {
                "technical_retry_count": retry_count,
                "technical_errors": errors,
            }, exc


    def planner_node(self, state: AgentState) -> dict:

        plan_result, recovery_state, error = (
            self.invoke_with_recovery(
                state,
                self.planner,
                "plan_with_recovery",
                "plan",
                query=state["query"],
                has_file=state.get("has_file", False),
                expense_details=state.get("expense_details", {}),
            )
        )

        if isinstance(error, BudgetExceededError):

            return {
                **self.budget_failure_response(
                    state, "planner", error
                ),
                **recovery_state,
            }

        if error is not None:

            return {
                **self.technical_failure_response(
                    state, "planner", error
                ),
                **recovery_state,
            }

        return {
            **recovery_state,
            "intent": plan_result["intent"],
            "plan": plan_result["plan"],
            "selected_tools": plan_result["selected_tools"],
            "missing_information": plan_result.get(
                "missing_information", []
            ),
            "expense_details": plan_result.get(
                "expense_details",
                state.get("expense_details", {}),
            ),
            "status": "planned",
        }

    # --------------------------------------------------
    # Executor
    # --------------------------------------------------

    def executor_node(self, state: AgentState) -> dict:

        tool_results = self.executor.execute(
            selected_tools=state.get("selected_tools", []),
            query=state["query"],
            has_file=state.get("has_file", False),
            file_path=state.get("file_path"),
            critic_feedback=state.get("critic_feedback"),
            expense_details=state.get("expense_details"),
        )

        # CAP-11: Validate required tool-result contracts.
        if not isinstance(tool_results, dict):
            raise ToolExecutionError("executor_result")

        for tool_name in state.get("selected_tools", []):

            if tool_name not in tool_results:
                raise ToolExecutionError(tool_name)

            tool_result = tool_results[tool_name]

            if not isinstance(tool_result, dict):
                raise ToolExecutionError(tool_name)

            if not tool_result:
                raise ToolExecutionError(tool_name)

        retrieved_evidence = []

        for result in tool_results.values():

            if isinstance(result, dict):
                retrieved_evidence.extend(
                    result.get("retrieved_evidence", [])
                )

        policy_result = tool_results.get(
            "search_hr_policy", {}
        )

        policy_answer = policy_result.get("answer", "")

        expense_result = tool_results.get(
            "calculate_expense_eligibility", {}
        )

        final_answer = policy_answer

        if expense_result:

            status = expense_result.get("compliance_status")

            if status == "exceeds_policy":

                final_answer += (
                    "\n\n### Expense Calculation\n"
                    f"- Claimed amount: "
                    f"{expense_result.get('currency', '')} "
                    f"{expense_result.get('claimed_amount')}\n"
                    f"- Policy limit: "
                    f"{expense_result.get('currency', '')} "
                    f"{expense_result.get('policy_limit')}\n"
                    f"- Amount within the stated limit: "
                    f"{expense_result.get('currency', '')} "
                    f"{expense_result.get('eligible_amount')}\n"
                    f"- Excess amount: "
                    f"{expense_result.get('currency', '')} "
                    f"{expense_result.get('excess_amount')}\n"
                    "- Compliance status: Exceeds policy limit.\n"
                )

            elif status == "within_policy":

                final_answer += (
                    "\n\nThe claimed amount is within the stated "
                    "monetary policy limit. Other applicable "
                    "conditions and approval requirements must "
                    "still be checked."
                )

            elif status == "requires_approval":

                final_answer += (
                    "\n\nThe claimed amount is within the stated "
                    "monetary limit, but additional approval "
                    "is required."
                )

            elif status == "unable_to_determine":

                final_answer += (
                    "\n\nA definitive expense assessment could not "
                    "be completed because required policy information "
                    "or expense details are missing."
                )

        return {
            "tool_results": tool_results,
            "retrieved_evidence": retrieved_evidence,
            "final_answer": final_answer,
            "status": "executed",
        }

    # --------------------------------------------------
    # Critic
    # --------------------------------------------------

    def critic_node(self, state: AgentState) -> dict:

        critic_result, recovery_state, error = (
            self.invoke_with_recovery(
                state,
                self.critic,
                "review_with_recovery",
                "review",
                query=state["query"],
                tool_results=state.get("tool_results", {}),
                final_answer=state.get("final_answer", ""),
            )
        )

        if isinstance(error, BudgetExceededError):

            return {
                **self.budget_failure_response(
                    state, "critic", error
                ),
                **recovery_state,
            }

        if error is not None:

            return {
                **self.technical_failure_response(
                    state, "critic", error
                ),
                **recovery_state,
            }

        approved = critic_result["approved"]

        retry_count = state.get("retry_count", 0)

        if not approved:
            retry_count += 1

        # Copy the list instead of modifying checkpointed state in place.
        execution_log = list(state.get("execution_log", []))

        execution_log.append({
            "agent": "critic",
            "approved": approved,
            "issues": critic_result.get("issues", []),
            "feedback": critic_result.get("feedback", ""),
        })

        return {
            **recovery_state,
            "is_approved": approved,
            "critic_feedback": critic_result.get("feedback", ""),
            "retry_count": retry_count,
            "status": "approved" if approved else "rejected",
            "execution_log": execution_log,
        }

    # --------------------------------------------------
    # Conditional routing
    # --------------------------------------------------

    def route_after_critic(self, state: AgentState) -> str:

        if state.get("status") == "technical_failure":
            return "technical_failure"

        if state.get("is_approved", False):
            return "approved"

        if state.get("retry_count", 0) <= MAX_RETRIES:
            return "retry"

        return "failure"

    # --------------------------------------------------
    # Failure node
    # --------------------------------------------------

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

    # --------------------------------------------------
    # CAP-08: Clarification routing
    # --------------------------------------------------

    def route_initial_request(
        self, state: AgentState
    ) -> str:

        resolution = state.get(
            "context_resolution", {}
        )

        if resolution.get(
            "requires_clarification", False
        ):
            return "clarification"

        return "planner"

    def clarification_node(
        self, state: AgentState
    ) -> dict:

        resolution = state.get(
            "context_resolution", {}
        )

        message = resolution.get(
            "clarification_message"
        ) or (
            "Please provide additional information "
            "so I can understand your request."
        )

        return {
            "intent": "CLARIFICATION_REQUIRED",
            "final_answer": message,
            "status": "clarification_required",
            "is_approved": False,
            "selected_tools": [],
            "tool_results": {},
            "retrieved_evidence": [],
            "retry_count": 0,
        }

    # --------------------------------------------------
    # Build and compile workflow
    # --------------------------------------------------

    # CAP-10: Intent-based conditional routing.

    def route_after_planner(
        self, state: AgentState
    ) -> str:

        if state.get("status") == "technical_failure":
            return "technical_failure"

        intent = state.get("intent", "")

        routes = {
            "POLICY_QUERY": "executor",
            "EXPENSE_ELIGIBILITY": "executor",
            "RECEIPT_ANALYSIS": "executor",
            "GREETING": "greeting",
            "OUT_OF_SCOPE": "out_of_scope",
            "CLARIFICATION_REQUIRED": "planner_clarification",
        }

        route = routes.get(intent)

        if route is None:
            raise ValueError(
                "Unsupported or missing Planner intent."
            )

        self.event_logger.log_event({
            "event": "intent_routed",
            "thread_id": state.get("thread_id"),
            "execution_id": state.get("execution_id"),
            "intent": intent,
            "target_node": route,
        })

        return route

    def direct_response(
        self, message: str, status: str
    ) -> dict:

        return {
            "final_answer": message,
            "status": status,
            "is_approved": False,
            "selected_tools": [],
            "tool_results": {},
            "retrieved_evidence": [],
            "critic_feedback": None,
            "retry_count": 0,
        }

    def greeting_node(
        self, state: AgentState
    ) -> dict:

        return self.direct_response(
            message=(
                "Hello! How can I help you with employee "
                "policies, travel expenses, leave policies "
                "or expense compliance?"
            ),
            status="greeting",
        )

    def out_of_scope_node(
        self, state: AgentState
    ) -> dict:

        return self.direct_response(
            message=(
                "This request is outside my scope. "
                "I can only assist with employee policies, "
                "HR-related guidance, travel expenses "
                "and expense compliance."
            ),
            status="out_of_scope",
        )

    def planner_clarification_node(
        self, state: AgentState
    ) -> dict:

        missing = state.get(
            "missing_information", []
        )

        message = (
            "Please provide additional information "
            "so I can assist with your request."
        )

        if missing:
            message += (
                " Required information: "
                + "; ".join(missing)
                + "."
            )

        return self.direct_response(
            message=message,
            status="clarification_required",
        )

    def route_after_executor(
        self, state: AgentState
    ) -> str:

        if state.get("status") == "technical_failure":
            return "technical_failure"

        return "critic"


    def build(self):

        if getattr(self, "_app", None) is not None:
            return self._app

        workflow = StateGraph(AgentState)

        workflow.add_node(
            "planner",
            self._track_node("planner", self.planner_node),
        )

        workflow.add_node(
            "executor",
            self._track_node("executor", self.executor_node),
        )

        workflow.add_node(
            "critic",
            self._track_node("critic", self.critic_node),
        )

        workflow.add_node(
            "failure",
            self._track_node("failure", self.failure_node),
        )
        workflow.add_node(
            "clarification",
            self._track_node(
                "clarification",
                self.clarification_node,
            ),
        )

        workflow.add_node(
            "greeting",
            self._track_node(
                "greeting", self.greeting_node
            ),
        )

        workflow.add_node(
            "out_of_scope",
            self._track_node(
                "out_of_scope", self.out_of_scope_node
            ),
        )

        workflow.add_node(
            "planner_clarification",
            self._track_node(
                "planner_clarification",
                self.planner_clarification_node,
            ),
        )

        workflow.add_conditional_edges(
            START,
            self.route_initial_request,
            {
                "planner": "planner",
                "clarification": "clarification",
            },
        )

        workflow.add_edge("clarification", END)
        workflow.add_conditional_edges(
            "planner",
            self.route_after_planner,
            {
                "technical_failure": END,
                "executor": "executor",
                "greeting": "greeting",
                "out_of_scope": "out_of_scope",
                "planner_clarification": (
                    "planner_clarification"
                ),
            },
        )

        workflow.add_edge("greeting", END)
        workflow.add_edge("out_of_scope", END)
        workflow.add_edge(
            "planner_clarification", END
        )
        workflow.add_conditional_edges(
            "executor",
            self.route_after_executor,
            {
                "critic": "critic",
                "technical_failure": END,
            },
        )

        workflow.add_conditional_edges(
            "critic",
            self.route_after_critic,
            {
                "technical_failure": END,
                "approved": END,
                "retry": "executor",
                "failure": "failure",
            },
        )

        workflow.add_edge("failure", END)

        self._app = workflow.compile(
            checkpointer=getattr(self, "checkpointer", None)
        )

        return self._app

    # --------------------------------------------------
    # Run workflow
    # --------------------------------------------------

    def run(
        self,
        query: str,
        has_file: bool = False,
        file_path: str | None = None,
        expense_details: dict | None = None,
        thread_id: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> dict:

        app = self.build()

        thread_id = thread_id or str(uuid.uuid4())
        execution_id = str(uuid.uuid4())

        # CAP-08: Persistent conversational context.
        checkpoint_config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        previous_state = app.get_state(checkpoint_config)

        previous_context = (
            previous_state.values.get("conversation_context", {})
            if previous_state and previous_state.values
            else {}
        )

        resolution = self.query_resolver.resolve(
            query=query,
            previous_context=previous_context,
        )

        contextual_query = resolution["resolved_query"]


        conversation_context = resolution["conversation_context"]

        # CAP-08: Propagate structured expense parameters.
        expense_fields = (
            "claimed_amount",
            "currency",
            "headcount",
        )

        resolved_expense_details = {}

        if (
            conversation_context.get("intent")
            == "expense_assessment"
        ):
            resolved_expense_details = {
                key: conversation_context[key]
                for key in expense_fields
                if key in conversation_context
                and conversation_context[key] is not None
            }

        # Explicit details from the current user request.
        current_fields = self.query_resolver.extract_fields(query)

        provided_details = {
            key: value
            for key, value in (expense_details or {}).items()
            if value is not None
        }

        # Validate conflicting user inputs.
        for key in expense_fields:
            if (
                key in current_fields
                and key in provided_details
                and current_fields[key] != provided_details[key]
            ):
                raise ValueError(
                    f"Conflicting values for {key}. "
                    "Please ensure the query and expense "
                    "form contain consistent information."
                )

        # Current UI details override inherited values.
        resolved_expense_details.update(provided_details)

        # Persist user-provided expense details.
        if resolved_expense_details:
            conversation_context.update(resolved_expense_details)
            conversation_context["intent"] = "expense_assessment"

        context_resolution = {
            "resolution_type": resolution["resolution_type"],
            "context_used": resolution["context_used"],
            "inherited_field_names": list(
                resolution["inherited_fields"].keys()
            ),
            "requires_clarification": resolution[
                "requires_clarification"
            ],
            "clarification_message": resolution.get(
                "clarification_message"
            ),
        }

        self.event_logger.log_event({
            "event": "query_resolved",
            "thread_id": thread_id,
            "execution_id": execution_id,
                        **{
                key: value
                for key, value in context_resolution.items()
                if key != "clarification_message"
            },
        })

        budget = RuntimeBudget.from_environment()

        initial_state: AgentState = {
            "intent": "",
            "query": contextual_query,
            "user_query": query.strip(),
            "conversation_context": conversation_context,
            "context_resolution": context_resolution,
            "has_file": has_file,
            "file_path": file_path,
            "expense_details": resolved_expense_details,
            "plan": [],
            "selected_tools": [],
            "missing_information": [],
            "tool_results": {},
            "retrieved_evidence": [],
            "critic_feedback": None,
            "is_approved": False,
            "retry_count": 0,
            "technical_retry_count": 0,
            "technical_retry_limit": MAX_TECHNICAL_RETRIES,
            "technical_errors": [],
            "error_code": None,
            "failed_node": None,
            "runtime_budget": budget.snapshot(),
            "budget_exhausted": False,
            "budget_exhaustion_reason": None,
            "final_answer": "",
            "execution_log": [],
            "status": "started",
            "thread_id": thread_id,
            "execution_id": execution_id,
        }

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        start_time = time.perf_counter()

        self.event_logger.log_event({
            "event": "workflow_started",
            "thread_id": thread_id,
            "execution_id": execution_id,
        })

        try:
            with budget_scope(budget):
                result = app.invoke(
                    initial_state,
                    config=config,
                )

            duration_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            self.event_logger.log_event({
                "event": "workflow_completed",
                "thread_id": thread_id,
                "execution_id": execution_id,
                "duration_ms": duration_ms,
                "status": result.get("status", "unknown"),
                "retry_count": result.get("retry_count", 0),
                "technical_retry_count": result.get(
                    "technical_retry_count", 0
                ),
                "error_code": result.get("error_code"),
                "failed_node": result.get("failed_node"),
                "runtime_budget": result.get(
                    "runtime_budget", {}
                ),
                "budget_exhausted": result.get(
                    "budget_exhausted", False
                ),
                "budget_exhaustion_reason": result.get(
                    "budget_exhaustion_reason"
                ),
            })

            return result

        except Exception as exc:
            duration_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            self.event_logger.log_event({
                "event": "workflow_failed",
                "thread_id": thread_id,
                "execution_id": execution_id,
                "duration_ms": duration_ms,
                "error_type": type(exc).__name__,
            })

            raise

    # --------------------------------------------------
    # Checkpoint inspection
    # --------------------------------------------------

    def get_thread_state(self, thread_id: str):

        app = self.build()

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        return app.get_state(config)

    # --------------------------------------------------
    # Close database connection
    # --------------------------------------------------

    def close(self):

        if self.connection:
            self.connection.close()
            self.connection = None
