# CAP-12: Runtime Step and Token Budget Enforcement

## 1. Objective

Implement bounded execution and resource-consumption controls
for the Multi-Agent Employee Operations and Expense Compliance
Copilot.

The implementation addresses the mandatory maximum ten
agent-step requirement and introduces configurable chat-token
budget controls for the hosted small language model.

## 2. Problems Addressed

Before CAP-12, the application provided bounded technical
recovery and quality correction retries through CAP-11.

However, it did not have:

- An independent global agent-step ceiling.
- Request-scoped chat-token accounting.
- Pre-invocation token-budget checks.
- Dynamic completion-token allowances.
- Centralized accounting across model invocation paths.
- Runtime resource-consumption metrics.
- Dedicated budget exhaustion termination.

CAP-12 introduces these controls without replacing the
existing Planner-Executor-Critic architecture.

## 3. Architecture

The implementation introduces two reusable components:

- src/agents/runtime_budget.py
- src/agents/budgeted_model.py

RuntimeBudget maintains per-request execution counters,
configured limits, reported usage, estimated usage,
and execution history.

The budget-aware model wrapper performs pre-invocation
checks, supplies a completion-token limit, and accounts
for successful and failed model invocations.

The Workflow activates a fresh runtime budget for every
new user request.

## 4. Agent-Step Counting Contract

An agent step represents an execution attempt by a business
agent within the Planner-Executor-Critic orchestration.

| Operation | Agent steps |
|---|---:|
| Initial Planner execution | 1 |
| Planner technical retry | 1 |
| Executor execution | 1 |
| Initial Critic execution | 1 |
| Critic technical retry | 1 |
| Executor quality correction | 1 |
| Subsequent Critic review | 1 |
| Deterministic greeting response | 0 |
| Deterministic clarification response | 0 |
| Safe failure response | 0 |

Internal model invocations and embedding operations are
accounted for separately.

This interpretation distinguishes agent execution attempts
from individual tool operations and provider requests.

## 5. Independent Hard Step Limit

The maximum configured agent-step limit is ten.

The application checks the remaining step allowance before
each business-agent execution attempt.

Technical retries also consume additional agent steps.

An execution attempt is blocked when no agent steps remain.

The tenth step is permitted. The eleventh step is blocked.

The step ceiling is independent of the existing technical
retry and evidence-quality correction limits.

## 6. Runtime Configuration

| Environment variable | Default |
|---|---:|
| COPILOT_MAX_AGENT_STEPS | 10 |
| COPILOT_TOKEN_BUDGET | 32000 |
| COPILOT_MAX_COMPLETION_TOKENS | 2048 |

Configuration constraints:

- All limits must be positive integers.
- The agent-step limit cannot exceed ten.
- The completion-token limit cannot exceed the chat-token budget.
- Invalid configuration is rejected.

These settings are application-level engineering controls.
They do not represent the team's allocated hosted API quota.

## 7. Request-Scoped Budget Management

Each Workflow request creates an independent RuntimeBudget.

The active budget is propagated through a ContextVar scope.

Budget counters are shared across the applicable agent
and model invocation paths within that request.

The context is cleaned up after successful execution
and failure termination.

A new conversation turn receives a fresh runtime budget.

Budget snapshots are stored in LangGraph state and
persisted through the existing checkpoint mechanism.

## 8. Model Invocation Integration

The common model invocation wrapper is integrated into:

- Planner intent classification and planning.
- Critic evidence verification.
- Policy answer generation.
- Expense policy extraction.

The wrapper preserves standalone model invocation behavior
when no active runtime budget exists.

Within an active request, all identified chat-model paths
use the shared accounting mechanism.

## 9. Pre-Invocation Token Checks

Before a chat-model invocation:

1. Estimate input-token requirements.
2. Determine the remaining accounted token allowance.
3. Calculate the available completion-token allowance.
4. Reject the invocation if no output allowance remains.
5. Supply a bounded max_tokens argument to the model.

The effective completion-token limit is the smaller of
the configured completion limit and the remaining allowance
after estimated input requirements.

## 10. Token Estimation

The input-token estimation mechanism uses UTF-8 byte length
plus a fixed message-framing allowance of 128.

This is a deliberately conservative approximation rather
than exact tokenization for the hosted model.

The approximation may reject requests that could fit
within the provider's actual token allowance.

Exact input-token estimation requires a compatible
model tokenizer and verified provider behavior.

## 11. Reported and Estimated Token Usage

The budget manager recognizes supported usage metadata
returned by the model client.

Tracked values include:

- Reported input tokens.
- Reported output tokens.
- Reported total tokens.
- Estimated input requirements.
- Estimated accounted token consumption.
- Calls with missing or invalid usage metadata.
- Failed calls with unreported usage.

Reported token counts and estimated allowances remain
separate to prevent misleading usage reporting.

When valid provider metadata is unavailable, the
pre-invocation input and completion allowance is
conservatively accounted.

## 12. Failed Invocation Accounting

A failed model request may still have been processed
by the inference provider.

The application therefore accounts for failed calls
using their estimated input and reserved output allowance.

Malformed JSON and schema-invalid responses are accounted
before downstream validation.

The invocation wrapper does not automatically retry
failed requests.

Technical recovery remains controlled by CAP-11.

## 13. Provider-Reported Budget Overrun

If returned usage exceeds the configured accounted
chat-token budget, the application detects the overrun
and terminates safely.

An overrun is recorded rather than hidden or truncated
from the usage metrics.

This post-response check does not prevent tokens that
have already been consumed by the provider.

Consequently, this implementation provides estimated
pre-invocation enforcement with reported-usage
reconciliation, rather than an absolute provider-side
billing guarantee.

## 14. Embedding Invocation Accounting

Application-level query and document embedding invocations
are tracked through the common embedding wrapper.

Embedding invocation counts remain separate from
chat-model invocation counts.

Embedding-token consumption is not included in the
chat-token budget because the current embedding interface
does not provide the required usage metadata.

An application-level embedding invocation may represent
multiple underlying provider requests.

## 15. Retry Integration

CAP-12 preserves the CAP-11 retry policies:

| Retry policy | Maximum |
|---|---:|
| Planner technical retries per operation | 1 |
| Critic technical retries per operation | 1 |
| Shared technical retries per request | 2 |
| Evidence-quality correction retries | 2 |
| SDK automatic retries | 0 |

Technical retries consume additional agent steps
and applicable model-token allowances.

Quality correction loops consume further Executor
and Critic execution steps.

Budget exhaustion is not treated as a retryable
technical model failure.

## 16. Safe Budget Exhaustion

The Workflow uses the existing technical_failure
terminal routing status with additional budget metadata.

Budget exhaustion is identified through:

- error_code = budget_exhausted
- budget_exhausted = true
- budget_exhaustion_reason
- failed_node
- runtime_budget

When a runtime limit is reached:

- Further prohibited agent execution is prevented.
- Unverified tool results are cleared.
- Retrieved evidence is cleared from the final result.
- Approval is set to false.
- A safe user-facing response is generated.
- Runtime metrics remain available for inspection.

No verified policy or expense decision is established
after terminal budget exhaustion.

## 17. Runtime Observability

The application records runtime budget snapshots
and structured budget exhaustion events.

Recorded metrics include:

- Agent steps consumed.
- Remaining agent steps.
- Agent execution history.
- Chat-model invocation attempts.
- Embedding invocation attempts.
- Reported input and output tokens.
- Estimated token allowances.
- Accounted chat-token consumption.
- Remaining token allowance.
- Budget exhaustion classification.

These metrics provide a foundation for the later
evaluation and formal tracing enhancements.

## 18. Streamlit Integration

The application includes a Runtime budget and resource
usage section.

The interface displays:

- Agent steps consumed and configured maximum.
- Accounted chat-token consumption.
- Chat-token budget utilization.
- Chat-model invocation count.
- Embedding invocation count.
- Reported and estimated token usage.
- Remaining execution and token allowances.
- Agent execution history.
- Explicit budget exhaustion messages.

Progress indicators are bounded for safe display when
provider-reported usage exceeds the configured limit.

Reported usage is distinguished from estimated usage.

## 19. Automated Test Coverage

| CAP-12 test category | Tests |
|---|---:|
| Runtime Budget Manager | 40 |
| Budget-aware model invocation | 16 |
| Environment configuration | 10 |
| Workflow runtime budget | 26 |
| Independent ten-step boundary | 3 |
| Streamlit budget presentation | 6 |
| Total new tests | 101 |

The tests cover invalid limits, invocation accounting,
context propagation, retries, budget exhaustion,
checkpoint persistence, and user interface behavior.

External model and embedding operations are mocked
in the dedicated CAP-12 automated tests.

## 20. Independent Ten-Step Boundary Validation

A dedicated forced-loop test deliberately relaxes the
quality-retry restriction using pytest monkeypatch.

The production retry configuration remains unchanged.

The tested execution path is:

1. Planner
2. Executor
3. Critic
4. Executor
5. Critic
6. Executor
7. Critic
8. Executor
9. Critic
10. Executor
11. Critic attempt blocked

The validated outcome establishes that the independent
ten-step ceiling prevents further business-agent execution
even when the quality-retry limit is deliberately relaxed.

Additional tests verify successful approval on the tenth
step and persistence of the terminal budget state.

## 21. Final Offline Regression Results

Execution environment:

- Windows
- Python 3.11.9
- pytest 9.2.0

Recorded regression results:

- Passed: 422
- Failed: 0
- Skipped: 14
- Warnings: 2
- Duration: 24.48 seconds

The skipped entries represent nine live regression cases
and five hosted integration test modules.

The two warnings concern an existing HTTPX TLS
verification configuration deprecation.

The results demonstrate tested software behavior rather
than live-model task accuracy or provider-side billing
guarantees.

## 22. Small-Model Compensation

CAP-12 improves the controllability of the hosted
small-model system through:

- Independent bounded agent execution.
- Controlled technical and quality correction loops.
- Configurable model completion limits.
- Conservative input-token estimation.
- Request-scoped resource accounting.
- Safe termination after budget exhaustion.
- Transparent execution metrics.

These controls do not require modification of model
weights or replacement of the hosted inference model.

## 23. Limitations and Remaining Work

- Provider compliance with dynamic max_tokens limits
  requires live verification.
- Input-token estimation is approximate.
- Provider-reported usage may exceed the estimated budget.
- Embedding-token consumption is not currently included.
- Provider billing costs are not calculated by CAP-12.
- Internal tool and model operations are accounted for
  separately from the ten agent-step ceiling.
- Standalone experiments without an active budget retain
  their existing invocation behavior.
- Configuration currently uses process environment variables.
- Production shared-instance concurrency requires
  additional validation.
- Deferred live-model evaluation remains outstanding.

## 24. Conclusion

CAP-12 introduces an independent hard ten-step execution
ceiling and estimated chat-token budget enforcement.

The implementation integrates with the existing recovery
framework, preserves checkpointed runtime metrics, and
provides transparent resource-consumption information.

The implementation is supported by 101 new automated
tests and a consolidated offline regression result of
422 passing tests with zero failures.
