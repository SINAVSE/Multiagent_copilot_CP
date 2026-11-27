# CAP-10: Conditional LangGraph Routing

## Objective

Implement deterministic intent-based workflow routing to improve
agent execution reliability, prevent unnecessary tool calls, and
provide appropriate responses for greetings, unsupported requests
and clarification requirements.

## IIT Kharagpur Requirement Alignment

- Explicit agent orchestration.
- Conditional workflow execution.
- Responsible AI refusal behavior.
- Small-model compensation through deterministic engineering.
- Controlled tool execution.
- Inference cost optimization.
- Execution observability.
- Automated functional validation.

## Baseline Limitations

Before CAP-10, the Planner always proceeded to the Executor,
regardless of the classified intent.

This could result in unnecessary agent execution and Critic
retries for requests that did not require policy retrieval
or expense assessment.

The Streamlit application also displayed direct-response statuses
as generic workflow warnings.

## Implemented Solution

CAP-10 introduces deterministic routing based on the structured
Planner intent implemented in CAP-09.

The existing CAP-08 context resolution and clarification
mechanisms remain unchanged.

## Conditional Routing Architecture

User Request
    -> Context Resolver
    -> Initial Router
        -> Resolver Clarification -> END
        OR
        -> Structured Planner
        -> Intent Router
            -> Greeting -> END
            -> Out of Scope -> END
            -> Planner Clarification -> END
            -> Executor -> Critic
                -> Approved -> END
                -> Retry -> Executor
                -> Failure -> END

## Intent Routing Matrix

| Intent | Destination | Executor | Critic |
|---|---|---|---|
| POLICY_QUERY | Executor | Yes | Yes |
| EXPENSE_ELIGIBILITY | Executor | Yes | Yes |
| RECEIPT_ANALYSIS | Executor | Yes | Yes |
| GREETING | Greeting node | No | No |
| OUT_OF_SCOPE | Refusal node | No | No |
| CLARIFICATION_REQUIRED | Clarification node | No | No |

The CAP-08 resolver clarification path bypasses all three agents.

## Direct Response Handling

Three new response nodes were introduced:

1. Greeting response node.
2. Out-of-scope refusal node.
3. Planner-driven clarification node.

These nodes generate deterministic responses without additional
language model calls.

Direct response states contain:

- Appropriate intent and workflow status.
- Final user-facing response.
- Empty selected tools.
- Empty tool results.
- Empty retrieved evidence.
- Zero Critic retries.
- No Critic feedback.

The is_approved field remains false because the Critic is
not invoked for these responses.

This does not indicate that the direct response failed.

## Fail-Closed Routing

Unknown or missing Planner intents are rejected before
Executor or Critic execution.

The existing Planner schema remains responsible for validating
intent and tool compatibility.

Graceful error recovery and bounded retries are addressed
in CAP-11.

## Persistent State Integrity

The implementation prevents previous business tool results,
retrieved evidence, Critic feedback and approval status from
contaminating subsequent direct responses.

Checkpoint validation confirms that the latest intent and
response status are persisted correctly.

## Routing Observability

A structured intent_routed event records:

- Conversation thread identifier.
- Workflow execution identifier.
- Classified intent.
- Selected destination node.

Routing events support execution analysis and debugging
without requiring an additional inference call.

Formal tracing integration is addressed in CAP-22.

## Streamlit Enhancements

The application now displays:

- Classified request intent.
- Greeting completion status.
- Explicit out-of-scope status.
- Clarification status.
- Appropriate citation and evidence messaging.
- Critic review not required for direct responses.

Existing policy citations, expense assessment, retrieved evidence
and Critic review displays remain available.

The sidebar workflow description reflects the conditional
routing architecture.

## Files Created

- tests/test_intent_routing.py
- tests/test_streamlit_routing.py
- docs/conditional_routing.md

## Files Modified

- src/agents/workflow.py
- app/streamlit_app.py

## Dedicated Routing Validation

Eighteen automated tests cover:

- Greeting response routing.
- Explicit out-of-scope refusal.
- Planner-driven clarification.
- Missing receipt clarification.
- Policy query routing.
- Expense eligibility routing.
- Receipt analysis routing.
- Invalid and empty intent handling.
- Existing resolver clarification compatibility.
- Previous business result cleanup.
- Persistent response status.
- Routing event generation for all six intents.

The tests use mocked agents and isolated in-memory checkpoints.

## Streamlit Runtime Validation

Five Streamlit AppTest cases validate:

- Greeting status rendering.
- Out-of-scope status rendering.
- Clarification status rendering.
- Citation and evidence messaging.
- Correct display of skipped Critic review.

The tests render simulated workflow results without invoking
the hosted inference API.

## Consolidated Validation Results

| Test Suite | Passed |
|---|---:|
| Streamlit Runtime Validation | 5 |
| Conditional Intent Routing | 18 |
| Structured Planner | 50 |
| Existing Planner | 3 |
| Query Resolver | 18 |
| Context and Intent Persistence | 19 |
| Workflow Orchestration | 3 |
| Executor | 2 |
| Offline Regression Validation | 12 |
| Total | 130 |

Execution results:

- Passed: 130
- Failed: 0
- Skipped: 9
- Warnings: 2
- Execution time: 18.25 seconds
- Hosted inference calls: 0

The nine skipped regression tests require hosted API access.

The two warnings relate to an existing HTTPX TLS configuration
deprecation.

## Small-Model Compensation

The language model classifies the employee request, while
deterministic Python and LangGraph logic control execution.

The model cannot directly determine arbitrary graph destinations.

Validated intent values are mapped to predefined execution
paths, reducing the risk of unnecessary or invalid agent calls.

Direct responses do not require additional language model
generation after classification.

## Responsible AI

- Unsupported requests receive an explicit refusal when
  correctly classified.
- Ambiguous requests can receive clarification responses.
- Unsupported intents cannot proceed to business tool execution.
- Direct responses are not presented as Critic-approved assessments.
- Previous execution results are cleared between requests.

## Expected Cost and Latency Benefits

Greeting, refusal and Planner-driven clarification paths avoid
Executor and Critic invocation.

Resolver-driven clarification avoids Planner invocation as well.

These architectural changes are expected to reduce unnecessary
inference usage.

Actual token savings, cost reduction and latency improvements
have not yet been measured and will be evaluated in CAP-25.

## Limitations and Remaining Work

- Hosted-model intent classification accuracy remains unmeasured.
- Schema-valid but semantically incorrect classification is possible.
- Streamlit tests validate rendering rather than live submission.
- Receipt routing is validated with mocked agents; actual receipt
  extraction and assessment remain part of CAP-16.
- Automatic malformed JSON recovery and tool failure handling
  remain part of CAP-11.
- Runtime step and token budget enforcement remain part of CAP-12.
- Prompt injection resilience requires dedicated adversarial tests.
- Live regression validation is deferred to consolidated evaluation.

## Conclusion

CAP-10 introduced deterministic conditional LangGraph routing,
direct response handling, persistent state safeguards, routing
observability and improved Streamlit response presentation.

All 130 consolidated offline tests passed successfully.

The implementation provides a controlled execution architecture
for subsequent error recovery, budget enforcement and responsible
AI enhancements.
