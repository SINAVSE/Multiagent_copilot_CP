# CAP-08: Context-Aware Query Resolution and Persistent Memory

## Objective

Improve multi-turn conversational accuracy, prevent unrelated historical
information from contaminating current requests, and support persistent
structured conversation state using LangGraph and SQLite.

## IIT Kharagpur Requirement Alignment

- Persistent state across conversation turns.
- Explicit agent orchestration and conditional routing.
- Small-model compensation through deterministic engineering.
- Error prevention and structured data integrity.
- Responsible handling of ambiguous requests.
- Evaluation and observability.

## Baseline Problems

The original workflow concatenated previous user messages and assistant
responses with the current request.

This introduced risks including:

- Unrelated conversation topics influencing retrieval.
- Historical monetary values overriding current claimed amounts.
- Incorrect expense assessment.
- Unnecessary agent execution for ambiguous references.

## Implemented Solution

1. Deterministic context-aware Query Resolver.
2. Structured conversation context stored in LangGraph state.
3. SQLite checkpoint-based persistence.
4. Selective field inheritance across conversation turns.
5. Explicit amount and currency replacement.
6. Headcount inheritance.
7. Topic switching and context reset.
8. Conversation thread isolation.
9. Structured expense parameter propagation.
10. Conflicting UI and query input validation.
11. Conditional clarification routing.
12. Query resolution observability events.
13. Updated multi-turn regression execution.

## Architecture

Current Request
    -> Context Resolver
    -> Previous SQLite Conversation State
    -> Context Extraction and Merging
    -> Conditional Routing
        -> Clarification Response
        OR
        -> Planner
        -> Executor
        -> Critic
    -> Persist Updated State

## Files Created

- src/agents/query_resolver.py
- tests/test_query_resolver.py
- tests/test_context_persistence.py
- docs/context_resolution.md

## Files Modified

- src/agents/state.py
- src/agents/workflow.py
- src/agents/planner.py
- app/streamlit_app.py
- tests/test_baseline_regression.py

## Offline Validation

| Test Category | Passed |
|---|---:|
| Query Resolver | 18 |
| SQLite Persistence | 8 |
| Expense Parameter Propagation | 3 |
| Clarification Routing | 4 |
| Regression Dataset Validation | 12 |
| Total Offline Tests | 45 |

The offline integration tests used mocked agents and temporary
SQLite databases.

The consolidated CAP-08 implementation suite passed all 33 tests
in 13.46 seconds.

No hosted inference calls were required for offline validation.

## Live Regression Validation

Test ID: REG-006

User request:
Can I claim $100 for a business dinner?

Previous conversation concerned the individual travel meal
allowance of USD 120 per day.

| Metric | Baseline | CAP-08 |
|---|---|---|
| Claimed Amount | USD 120 | USD 100 |
| Amount Integrity | FAIL | PASS |
| Regression Result | FAIL | PASS |

Observed CAP-08 Results:

- Context resolution type: standalone
- Previous context used: false
- Resolved claimed amount: USD 100
- Calculator claimed amount: USD 100
- Critic approved response: true
- Retry count: 0
- Workflow status: approved
- Regression status: PASS

The workflow avoided a definitive expense compliance decision
because required information was missing.

Critic approval refers to response review, not business approval
of the expense.

## Execution Metrics

- Previous turn latency: 15.733 seconds
- Current query latency: 22.239 seconds
- Combined workflow latency: 37.972 seconds
- Agent retries: 0

No latency improvement is claimed.

## Small-Model Compensation

The context resolver uses deterministic Python logic rather
than relying exclusively on the hosted language model.

This reduces the risk of context contamination and requires
no additional hosted inference calls for context resolution.

Supported ambiguous requests can return clarification responses
without invoking Planner, Executor or Critic.

## Responsible AI

- Historical assistant responses are not treated as authoritative facts.
- Current explicit user inputs are protected against conflicting values.
- Previous expense details are isolated between conversation threads.
- Ambiguous references can trigger clarification.
- Expense compliance assessment does not constitute actual approval.

## Limitations and Remaining Validation

- Rule-based resolution does not support all natural-language references.
- Complex multi-expense requests require further interpretation.
- Category-specific expense semantics require further validation.
- Clarification recovery across subsequent turns requires further testing.
- Only REG-006 was executed as a live CAP-08 regression.
- Remaining live regression cases are deferred to consolidated evaluation.
- The baseline and enhanced regression setup methodologies differ.
- Overall end-to-end accuracy improvement has not yet been established.
- PII protection and checkpoint retention controls remain future tasks.

## Conclusion

CAP-08 implemented and validated persistent structured conversational
state, selective context inheritance, expense input integrity and
conditional clarification routing.

A total of 45 offline tests and one targeted live regression test
passed successfully.

The targeted REG-006 regression demonstrated correction of the
claimed amount contamination defect.
