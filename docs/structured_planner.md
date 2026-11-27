# CAP-09: Structured Planner Intent Classification

## Objective

Improve the reliability of the Employee Operations and Expense
Compliance Copilot by introducing structured intent classification,
strict Planner output validation, and deterministic tool restrictions.

## IIT Kharagpur Requirement Alignment

- Structured model outputs.
- JSON schema validation.
- Reliable agent orchestration.
- Controlled tool selection.
- Small-model compensation through deterministic engineering.
- Detection of malformed model responses.
- Responsible AI safeguards.
- Automated software testing.

## Baseline Limitations

The original Planner selected tools using an LLM-generated JSON
response with basic dictionary and list validation.

The original implementation did not provide:

- Explicit intent classification.
- Strict validation of every response field.
- Duplicate tool detection.
- Intent-specific tool restrictions.
- Mandatory tool ordering.
- Explicit receipt availability validation.
- Persistent Planner intent.

## Implemented Solution

Introduced six supported Planner intents:

| Intent | Purpose |
|---|---|
| POLICY_QUERY | Retrieve employee policy information |
| EXPENSE_ELIGIBILITY | Assess an employee expense |
| RECEIPT_ANALYSIS | Extract or assess receipt information |
| GREETING | Handle simple greetings |
| OUT_OF_SCOPE | Identify unsupported requests |
| CLARIFICATION_REQUIRED | Request missing information |

## Structured Output Schema

The PlannerOutput Pydantic model validates:

- Mandatory intent.
- Non-empty execution plan.
- Supported tool names.
- Maximum plan length.
- Maximum selected tool count.
- Missing-information list.
- Strict string values.
- Unexpected response fields.
- Duplicate tools.
- Intent-tool compatibility.

The plan is limited to ten entries, and tool selection is limited
to the three available tools.

These schema limits do not replace runtime agent step limits,
which will be implemented separately.

## Tool Compatibility

| Intent | Allowed Tools |
|---|---|
| POLICY_QUERY | Policy search |
| EXPENSE_ELIGIBILITY | Policy search and expense calculation |
| RECEIPT_ANALYSIS | Receipt extraction, policy search and calculation |
| GREETING | None |
| OUT_OF_SCOPE | None |
| CLARIFICATION_REQUIRED | None |

Additional validation ensures that:

- Policy search precedes expense calculation.
- Receipt extraction is the first tool for receipt analysis.
- Receipt analysis requires an uploaded file.
- Clarification requests identify missing information.
- Unsupported and duplicate tools are rejected.

## Planner Integration

The Planner prompt was enhanced with explicit intent definitions,
tool selection rules, structured expense details, and instructions
to treat employee requests as untrusted input.

The model response is parsed as JSON and validated using Pydantic.

Invalid responses raise controlled validation exceptions before
normal tool execution can proceed.

Automatic retry and recovery are addressed in CAP-11.

## Persistent Intent State

The classified intent is propagated into AgentState and stored
using the existing LangGraph SQLite checkpoint mechanism.

The intent is reset at the beginning of every request.

When the CAP-08 Query Resolver bypasses the Planner, the
clarification node explicitly assigns CLARIFICATION_REQUIRED.

This prevents stale intent from remaining in the completed
clarification response.

## Test Compatibility

Existing Workflow test fixtures were updated to support the
mandatory intent field.

The older Workflow tests now use isolated in-memory checkpoints
and explicitly initialized Query Resolver instances.

Production checkpoint behavior continues to use SQLite.

Existing Critic approval and retry assertions were preserved.

## Files Created

- src/agents/planner_schema.py
- tests/test_structured_planner.py
- docs/structured_planner.md

## Files Modified

- src/agents/planner.py
- src/agents/state.py
- src/agents/workflow.py
- tests/test_context_persistence.py
- tests/test_workflow.py

## Validation Results

| Test Suite | Passed |
|---|---:|
| Structured Planner | 50 |
| Existing Planner | 3 |
| Query Resolver | 18 |
| Context and Intent Persistence | 19 |
| Workflow Orchestration | 3 |
| Executor | 2 |
| Offline Regression Validation | 12 |
| Total | 107 |

Consolidated execution results:

- Passed: 107
- Failed: 0
- Skipped: 9
- Warnings: 2
- Execution time: 15.02 seconds
- Hosted inference calls: 0

The skipped tests require hosted API access.

The two warnings relate to an existing HTTPX TLS configuration
deprecation and do not represent test failures.

## Small-Model Compensation

The application does not rely exclusively on the language model
to enforce tool selection rules.

Deterministic schema validation rejects unsupported intents,
invalid tools, incompatible tool combinations, duplicate tools,
invalid response structures, and incorrect tool ordering.

This reduces the risk of executing invalid model-generated plans
without requiring a larger language model.

## Responsible AI

- Unsupported intents cannot contain business tools in a valid plan.
- Expense calculation requires policy retrieval.
- Unexpected approval fields are rejected.
- Missing receipt files are detected.
- Historical intents are not reused as current intent decisions.

These controls validate the Planner output contract. Actual
out-of-scope response routing is implemented separately in CAP-10.

## Limitations and Remaining Work

- Hosted-model semantic intent classification accuracy has not
  yet been measured.
- Mocked tests validate output contracts and workflow behavior.
- Schema-valid output does not guarantee correct semantic intent.
- Prompt injection resistance requires dedicated adversarial tests.
- Greeting and out-of-scope response routing remain part of CAP-10.
- Planner-driven clarification routing remains part of CAP-10.
- Malformed JSON recovery and bounded retries remain part of CAP-11.
- Runtime step limits and token budgets remain part of CAP-12.
- Receipt execution and assessment require further validation.
- Live regression validation is deferred to consolidated evaluation.

## Conclusion

CAP-09 introduced a structured Planner contract with six intent
types, strict schema validation, deterministic tool restrictions,
and persistent intent state.

All 107 consolidated offline tests passed successfully.

The implementation provides the foundation for intent-based
conditional agent routing in CAP-10.
