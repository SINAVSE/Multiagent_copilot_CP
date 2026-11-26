
# Functional Specification

## Employee Operations and Expense Compliance Copilot

Version: 1.0
Enhancement: CAP-05
Date: November 27, 2026

## 1. Problem Statement

Employees frequently need to navigate multiple HR,
travel, expense and organizational policy documents
to understand applicable rules.

Manual policy discovery is time-consuming and can
lead to inconsistent interpretation of business rules.

The Employee Operations and Expense Compliance Copilot
provides evidence-grounded policy assistance using
a hosted language model of no more than 8 billion
parameters, supported by retrieval-augmented generation,
specialized tools and multi-agent orchestration.

## 2. Objectives

- Simplify employee policy discovery.
- Provide accurate and citation-grounded answers.
- Support expense eligibility assessments.
- Reduce unsupported AI-generated conclusions.
- Maintain relevant context across conversation turns.
- Demonstrate effective engineering techniques for
  constrained language models.
- Provide transparent and observable agent execution.

## 3. Functional Scope

### FR-01: Employee Policy Assistance

The system shall answer questions related to:

- Employee leave and time-off policies.
- Business travel policies.
- Meal allowances and per diem policies.
- Expense reimbursement.
- Gifts and entertainment.
- Employee benefits covered by the indexed corpus.
- Other supported employee operations policies.

All factual policy claims shall be supported by
retrieved evidence.

### FR-02: Expense Eligibility Assessment

The system shall:

- Identify the applicable expense category.
- Preserve the amount explicitly entered by the user.
- Identify the applicable currency.
- Retrieve relevant policy limits.
- Determine whether mandatory information is missing.
- Perform deterministic expense calculations.
- Explain applicable policy conditions.
- Identify additional approval requirements when
  supported by evidence.

The system shall not assume missing expense details.

For team meals, the system shall determine whether
the amount represents an individual claim or a
total group bill.

For individual travel meals, the system shall consider
cumulative daily expenses when applying daily limits.

### FR-03: Receipt Analysis

Planned capability:

- Accept supported receipt files.
- Extract vendor, date, amount and currency.
- Identify missing or unreadable information.
- Allow users to confirm extracted details.
- Compare confirmed details against policy rules.

Receipt extraction shall not automatically approve
or submit reimbursement claims.

### FR-04: Multi-Turn Conversation

The system shall distinguish between:

- Standalone questions.
- Follow-up questions.
- Topic changes.
- Ambiguous references.

Relevant conversation context shall be retained.

Unrelated previous conversation history shall not
override the current question or user-provided values.

### FR-05: Intent Classification

The system shall classify requests using the
following proposed categories:

- POLICY_QUERY
- EXPENSE_ELIGIBILITY
- RECEIPT_ANALYSIS
- GREETING
- OUT_OF_SCOPE
- CLARIFICATION_REQUIRED

Intent classification shall determine the appropriate
execution path and required tools.

### FR-06: Multi-Agent Execution

The system shall implement:

- Planner for task decomposition and tool selection.
- Executor for retrieval and specialized tools.
- Critic for evidence and consistency validation.

The workflow shall support controlled retries,
persistent state and error handling.

### FR-07: Citation-Grounded Responses

Policy answers shall include:

- Relevant policy information.
- Supporting source document.
- Applicable policy section.
- Source link when available.

The system shall distinguish retrieved facts from
assumptions or missing information.

### FR-08: Clarification Handling

The system shall request clarification when:

- Expense category is ambiguous.
- Claimed amount is missing.
- Currency is required but unspecified.
- Headcount is required but missing.
- Applicable cumulative expenses are unknown.
- Multiple policy interpretations are possible.

The system shall not produce definitive expense
decisions when mandatory information is missing.

### FR-09: Out-of-Scope Handling

The system shall politely refuse unrelated requests.

Examples include:

- Weather forecasts.
- General entertainment requests.
- Unrelated programming questions.
- Unrelated personal recommendations.

Out-of-scope requests should not trigger unnecessary
policy retrieval or expense calculations.

### FR-10: Agent Observability

The system shall provide visibility into:

- Agent plans.
- Selected tools.
- Retrieved policy evidence.
- Policy citations.
- Expense assessment.
- Critic decisions.
- Retry attempts.
- Workflow execution status.
- Performance metrics.

Detailed diagnostic information shall not expose
credentials or unnecessary personal information.

## 4. Responsible AI Boundaries

### RA-01: No Unsupported Policy Claims

The system shall not invent:

- Policy limits.
- Reimbursement amounts.
- Leave entitlements.
- Approval requirements.
- Policy exceptions.

### RA-02: Human-in-the-Loop

The system provides advisory assistance only.

It shall not:

- Approve reimbursement claims.
- Submit expense reports.
- Approve or reject employee leave.
- Make employment decisions.
- Modify HR records.
- Execute financial transactions.

Final decisions remain with authorized human
reviewers and applicable company processes.

### RA-03: Privacy and PII Protection

The system shall minimize the collection,
processing and storage of personal information.

Sensitive information shall not be unnecessarily
included in prompts, logs or execution traces.

### RA-04: Prompt Injection Protection

Instructions contained in retrieved policy
documents or user-provided files shall be treated
as untrusted data.

Such instructions shall not override system
instructions or authorized tool boundaries.

### RA-05: Insufficient Evidence

When relevant policy evidence is unavailable,
the system shall explicitly communicate the
limitation.

It shall not generate an unsupported policy answer.

## 5. Correct Answer Criteria

A correct response shall satisfy the applicable
criteria below:

1. Addresses the current user question.
2. Uses the correct policy category.
3. Preserves explicitly supplied user information.
4. Uses relevant and sufficient policy evidence.
5. Provides appropriate source citations.
6. Represents monetary limits accurately.
7. Applies the correct units and currency.
8. Performs deterministic calculations correctly.
9. Requests clarification when necessary.
10. Avoids unsupported conclusions.
11. Respects responsible AI boundaries.
12. Remains consistent with structured tool output.

## 6. Non-Functional Requirements

- Hosted language model with no more than
  8 billion parameters.
- Structured output and schema validation.
- Controlled retry mechanism.
- Maximum of 10 agent steps per query.
- Configurable token consumption limits.
- Secure credential management.
- Reproducible dependency installation.
- Rate limiting and caching.
- Observable workflow execution.
- Containerized application deployment.

## 7. Evaluation Strategy

System performance shall be evaluated using:

- Original Week 1 evaluation dataset.
- Real-policy evaluation dataset.
- Targeted regression test suite.
- Retrieval Recall@K.
- End-to-end answer accuracy.
- Task completion rate.
- Average tool calls per task.
- Token consumption.
- Cost per query.
- Query latency.
- Failure taxonomy.
- Responsible AI test cases.

## 8. Acceptance Criteria Examples

| Scenario | Expected Behavior |
|---|---|
| Individual travel meal allowance | Return applicable daily limit with citation |
| Traveling team meals | Return applicable per-person, per-meal limit |
| Ambiguous dinner expense | Request required clarification |
| Travel-to-leave topic switch | Retrieve leave policy independently |
| Relevant follow-up question | Retain necessary conversation context |
| Weather question | Return explicit out-of-scope refusal |
| Missing policy evidence | Communicate insufficient information |
| Incorrect calculator amount | Detect inconsistency before approval |
| Malicious document instruction | Ignore unauthorized instructions |
| Receipt with sensitive information | Minimize unnecessary PII exposure |

## 9. Implementation Status

This specification defines the target behavior
of the enhanced Copilot.

Not all requirements are implemented or validated
at the time of writing.

Implementation and verification will be tracked
through the corresponding CAP enhancement tasks.

## 10. Academic Traceability

This document represents a refined functional
specification prepared after exploratory baseline
testing.

It does not replace the original Week 1
pre-implementation specification or evaluation set.

The original artifacts shall remain preserved
for academic traceability.

## 11. Conclusion

The Copilot aims to demonstrate reliable employee
policy assistance and expense compliance support
using a constrained hosted language model.

Its effectiveness will be established through
systematic engineering, evidence-grounded responses,
deterministic tools and measurable evaluation.
