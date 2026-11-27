# CAP-11: Error Handling and Bounded Recovery

## 1. Objective

Improve the reliability of the Multi-Agent Employee
Operations and Expense Compliance Copilot by implementing
structured error classification, strict response validation,
bounded technical recovery, safe failure termination, and
observable execution.

The implementation compensates for reliability limitations
of a hosted small language model without changing model
weights or replacing the hosted model.

## 2. Problems Addressed

Before CAP-11, the system had the following limitations:

- Malformed model responses could interrupt execution.
- Invalid schema responses lacked controlled recovery.
- Technical Critic failures could be treated as evidence rejection.
- Executor failures could propagate as uncontrolled exceptions.
- Missing tool results were not consistently validated.
- SDK retries could increase inference consumption.
- Some integration tests could invoke hosted services by default.
- Technical failures and quality rejection were not clearly separated.

## 3. Architecture

The application retains its Planner-Executor-Critic architecture.

CAP-11 introduces:

- Shared error classification.
- Shared strict JSON response parser.
- Bounded Planner recovery.
- Bounded Critic recovery.
- Controlled Executor tool exceptions.
- Per-request technical retry accounting.
- Safe terminal Workflow states.
- Structured error events.
- Streamlit technical failure presentation.

## 4. Error Classification

The shared error classification module defines:

| Error code | Retry eligible |
|---|---|
| invalid_response | Yes |
| malformed_json | Yes |
| schema_validation | Yes |
| api_timeout | Yes |
| api_connection | Yes |
| authentication_failed | No |
| rate_limit_exceeded | No |
| api_server_error | Selected transient server failures |
| api_request_failed | No |
| tool_execution_failed | No |
| internal_error | No |

Retry eligibility is subject to the operation-level and
request-level retry budgets.

User-facing error messages are predefined and do not
directly expose raw exception messages.

## 5. Strict JSON Response Parsing

The shared response parser provides:

- JSON object validation.
- Complete Markdown JSON fence support.
- Empty response detection.
- Duplicate JSON key rejection.
- Rejection of NaN and Infinity JSON constants.
- Maximum response length validation.
- Optional structured content-block handling.
- Classification of malformed and invalid responses.

The maximum configured response length is 100,000 characters.

Planner and Critic validation rules remain responsible for
their respective response contracts.

The parser does not silently reconstruct malformed JSON.

## 6. Planner Recovery

The Planner retains its existing single-attempt plan method.

A separate plan_with_recovery method provides bounded
technical recovery.

For recoverable response errors, the second attempt includes
corrective instructions requiring:

- A valid JSON object.
- Supported intent names.
- Supported tool names.
- Required fields and valid field types.
- Correct tool ordering.
- Compliance with uploaded-file requirements.

The original employee query and structured expense details
are preserved during recovery.

The Planner performs a maximum of two model attempts per
recovery operation.

## 7. Executor Failure Handling

Executor tool boundaries classify operational failures
using ToolExecutionError.

Protected operations include:

- Policy retrieval.
- Expense policy extraction.
- Fixed expense calculation.
- Per-person expense calculation.
- Policy answer generation.

The Workflow validates that each selected tool returns a
non-empty dictionary before continuing to Critic review.

The full Executor operation is not automatically retried
after a technical failure.

Insufficient evidence remains a legitimate business outcome
and can produce an unable_to_determine compliance status.

Receipt extraction is intentionally blocked until the
actual extraction tool is implemented in CAP-16.

## 8. Critic Recovery

The Critic distinguishes between:

1. Deterministic evidence validation rejection.
2. Model-based evidence quality rejection.
3. Technical model response or service failure.

A separate review_with_recovery method handles eligible
technical failures.

Malformed Critic responses can be retried without
repeating the complete Executor operation.

If technical recovery is exhausted, the Workflow terminates
without establishing a verified compliance decision.

Genuine quality rejection continues to use the existing
bounded Executor-Critic correction loop.

## 9. Retry and Timeout Configuration

| Configuration | Value |
|---|---|
| SDK automatic retries | 0 |
| Configured request timeout | 60 seconds |
| Planner technical retries per operation | Maximum 1 |
| Critic technical retries per operation | Maximum 1 |
| Shared technical retries per request | Maximum 2 |
| Executor technical retries | 0 |
| Existing quality correction retries | Maximum 2 |

The shared retry allowance is stored in Workflow state.

It is reset for each new request and persists across
agent nodes within the current request.

Technical retries are tracked separately from quality
rejection retries.

These retry limits are engineering design decisions.

The mandatory global agent-step cap and token-budget
enforcement are addressed separately in CAP-12.

## 10. Safe Workflow Termination

The Workflow introduces a technical_failure terminal status.

When an agent operation fails:

- Further inappropriate agent execution is prevented.
- Approval is set to false.
- Unverified tool results are cleared.
- Unverified retrieved evidence is cleared.
- Critic feedback is cleared.
- A safe user-facing response is generated.
- Failure classification and failed agent are recorded.
- Recovery state is retained in the checkpoint.

No verified policy or expense decision is established
following a terminal technical failure.

## 11. State and Observability

New Workflow state fields include:

- technical_retry_count
- technical_retry_limit
- technical_errors
- error_code
- failed_node

Structured recovery events include:

- Agent name.
- Thread identifier.
- Execution identifier.
- Error classification.
- Exception type.
- Retry eligibility.
- Whether recovery will be attempted.
- Consumed technical retry count.

Terminal tool failure events also identify the failed tool
where available.

Raw exception messages are not included in the newly
introduced error metadata.

## 12. Streamlit Integration

The user interface now distinguishes technical failure
from evidence-quality rejection and successful approval.

The interface displays:

- Controlled failure status.
- Safe failure message.
- Failure category.
- Failed execution stage.
- Technical retry count.
- Structured technical error events.
- Absence of verified approval.

## 13. Automated Test Coverage

New CAP-11 automated tests:

| Test category | Tests |
|---|---:|
| Shared parser and error classification | 40 |
| Planner recovery | 20 |
| Critic recovery | 23 |
| Executor failure handling | 15 |
| Workflow recovery | 31 |
| Streamlit recovery presentation | 4 |
| Total new tests | 133 |

Two existing routing test fixtures were updated to return
mock results for every selected tool.

Production tool-result validation was not weakened.

## 14. Final Offline Regression Results

Execution environment:

- Windows
- Python 3.11.9
- pytest 9.2.0

Recorded execution result:

- Passed: 321
- Failed: 0
- Skipped: 14
- Warnings: 2
- Duration: 26.89 seconds

The warnings concern an existing HTTPX TLS verification
configuration deprecation.

The 14 skips represent nine live regression cases and
five hosted integration test modules.

The results validate software behavior under the tested
conditions. They do not establish live-model answer accuracy.

## 15. Live API Test Isolation

Hosted integration tests are disabled by default.

RUN_LIVE_TESTS=1 explicitly enables hosted integration tests.

RUN_LIVE_REGRESSION=1 explicitly enables existing live
regression cases.

Five hosted integration test modules are protected by
module-level opt-in guards.

The previous Critic live script has been converted into
parameterized pytest test cases.

Manual debugging scripts have also been updated with
explicit live execution guards.

Default offline execution:

    python -m pytest -q

A separate hosted LLM connectivity test passed in
4.92 seconds after SDK retries were disabled.

This connectivity check is not included in the final
321-test offline validation result.

## 16. Small-Model Compensation

The implementation addresses small-model limitations through:

- Strict structured response contracts.
- Deterministic schema validation.
- Corrective prompting.
- Bounded model recovery.
- Deterministic business calculations.
- Required tool-result validation.
- Independent evidence review.
- Safe termination instead of fabricated conclusions.

These techniques improve the controllability of the
agentic system without modifying the underlying model.

The improvement in live-model task accuracy has not yet
been quantified through a complete comparative evaluation.

## 17. Limitations and Remaining Work

- The mandatory ten-step execution cap is pending CAP-12.
- Token-budget enforcement is pending CAP-12.
- Actual receipt extraction is pending CAP-16.
- Detailed evidence and compliance consistency improvements
  remain part of subsequent implementation tasks.
- Production shared-instance concurrency requires validation.
- Live-model recovery effectiveness requires further evaluation.
- Hosted API quota consumption is not fully quantified.
- Live test isolation is source-based, not a network sandbox.
- Failures outside the agent nodes, including checkpoint
  infrastructure failures, are not fully covered by the
  new terminal error-handling mechanism.
- Manual script guards were verified with live execution
  disabled.

## 18. Conclusion

CAP-11 introduces a structured and bounded error-handling
framework for the multi-agent copilot.

The system can recover from tested eligible model response
failures, stop safely after operational failures, distinguish
technical failure from quality rejection, and preserve
observable recovery state.

The implementation is supported by 133 new automated tests
and a consolidated offline regression result of 321 passing
tests with zero failures.
