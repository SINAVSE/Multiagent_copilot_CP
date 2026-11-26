
# Baseline Assessment Report

## 1. Project Overview

- Project: Employee Operations and Expense Compliance Copilot
- Enhancement: CAP-04
- Baseline Version: 1.0
- Assessment Date: November 27, 2026
- Language Model: Qwen/Qwen3-8B
- Embedding Model: BAAI/bge-small-en-v1.5
- Agent Framework: LangGraph
- Evaluation Method: Manual exploratory regression testing

## 2. Objective

Establish a functional baseline before implementing
architectural and agentic enhancements.

The baseline enables measurable before-and-after comparisons
of retrieval accuracy, conversation handling, expense
compliance, responsible AI and Critic reliability.

## 3. Existing System Capabilities

- Hybrid retrieval using BM25 and dense vector search.
- Cross-encoder reranking.
- Planner-Executor-Critic agent architecture.
- Persistent conversation state.
- Citation-grounded policy responses.
- Deterministic expense eligibility calculation.
- Streamlit application with agent execution visibility.
- Structured agent feedback and retry mechanism.

## 4. Baseline Environment Validation

| Validation | Result |
|---|---|
| Python version | 3.11.9 |
| Fresh virtual environment installation | Successful |
| Core dependency imports | Successful |
| Expense calculator unit tests | 19/19 passed |
| Unit test execution time | 0.08 seconds |
| Streamlit application execution | Successful |
| Hosted LLM connectivity | Successful |
| Policy retrieval execution | Successful |
| Agent execution visibility | Successful |

## 5. Functional Regression Baseline

| Test ID | Functionality | Recorded Status |
|---|---|---|
| REG-001 | Individual travel meal allowance | PASS |
| REG-002 | Travel-to-leave topic switching | FAIL |
| REG-003 | Follow-up question handling | PARTIAL |
| REG-004 | Out-of-scope refusal | FAIL |
| REG-005 | Ambiguous expense handling | FAIL |
| REG-006 | Claimed amount integrity | FAIL |
| REG-007 | Critic validation | FAIL |
| REG-008 | Retrieval ranking | FAIL |
| REG-009 | Leave-to-travel topic switching | FAIL |

### Baseline Summary

- Total targeted regression cases: 9
- Passed: 1
- Partially passed: 1
- Failed: 7

These results represent targeted exploratory regression
observations and do not represent overall system accuracy.

## 6. Failure Taxonomy

| Defect ID | Description | Severity | Suspected Failure Category | Planned Enhancement |
|---|---|---|---|---|
| DEF-01 | Conversation context contamination | High | Context management | CAP-08 |
| DEF-02 | Missing out-of-scope classification | High | Intent classification | CAP-09, CAP-10 |
| DEF-03 | Incorrect policy terminology | Medium | Reasoning error | CAP-17 |
| DEF-04 | Premature expense classification | High | Reasoning error | CAP-15 |
| DEF-05 | Incorrect claimed amount | Critical | Input integrity | CAP-15 |
| DEF-06 | Critic approves inconsistent assessment | Critical | Validation failure | CAP-17 |
| DEF-07 | Irrelevant policy chunks retrieved | High | Retrieval relevance | CAP-13 |
| DEF-08 | Relevant evidence ranked below irrelevant evidence | High | Retrieval ranking | CAP-13 |
| DEF-09 | Incorrect conclusion without headcount | Critical | Reasoning and validation | CAP-15, CAP-17 |

Root causes remain subject to source-code investigation
and controlled regression experiments.

## 7. Key Observations

### 7.1 Conversation Context Management

The system correctly answers standalone policy questions
but demonstrates inconsistent retrieval when users switch
between unrelated policy topics.

A context-aware query resolution mechanism is required
to distinguish standalone queries from genuine follow-ups.

### 7.2 Retrieval Quality

For an ambiguous business dinner query, five policy chunks
were retrieved.

The relevant traveling team meals policy appeared at rank 4,
while unrelated home office equipment information appeared
at rank 2.

This indicates an opportunity to improve query formulation,
retrieval relevance and reranking.

### 7.3 Expense Input Integrity

The user requested an assessment for a USD 100 business dinner.

However, the expense assessment displayed USD 120 as the
claimed amount.

The source of this discrepancy requires investigation.
Explicit user-provided amounts must not be overridden by
previous conversation values or policy limits.

### 7.4 Critic Validation

The Critic approved an expense assessment despite
inconsistent monetary values and missing headcount.

Evidence grounding alone is insufficient for financial
compliance validation.

Deterministic consistency checks should complement
LLM-based Critic evaluation.

### 7.5 Responsible AI

An unrelated weather question triggered agent execution
and evidence review instead of an explicit scope-based
refusal.

Early intent classification and conditional routing
are required.

## 8. Compensating for Small-Model Limitations

| Observed Limitation | Proposed Engineering Enhancement |
|---|---|
| Context confusion | Context-aware query resolution |
| Incorrect task interpretation | Structured intent classification |
| Irrelevant retrieved information | Hybrid retrieval optimization and reranking |
| Incorrect numerical reasoning | Deterministic calculation tools |
| Inconsistent expense assessment | Input integrity and schema validation |
| Unreliable self-evaluation | Deterministic Critic checks |
| Unnecessary inference calls | Conditional agent routing |
| Unsupported conclusions | Evidence grounding and clarification |

These are proposed mitigations. Their effectiveness
must be established through subsequent evaluation.

## 9. Before-and-After Evaluation Targets

| Metric | Baseline Observation | Target |
|---|---|---|
| Targeted regression cases passed | 1 of 9 | 9 of 9 |
| Relevant team meal policy rank | 4 | Within top 3 |
| Claimed amount integrity | Failed in observed execution | Correct user amount preserved |
| Out-of-scope refusal | Failed | Explicit refusal |
| Conversation topic switching | Failed in observed executions | Correct topic resolution |
| Critic consistency validation | Failed | Detect inconsistent assessment |
| Expense calculator unit tests | 19 of 19 | Maintain all existing tests |

Targets are proposed acceptance criteria, not achieved results.

## 10. Formal Evaluation Metrics

The final evaluation will separately measure:

- End-to-end answer accuracy.
- Retrieval Recall@K.
- Mean Reciprocal Rank.
- Task completion rate.
- Average tool calls per task.
- Average inference token consumption.
- Average cost per query.
- Average query latency.
- Failure distribution by category.
- Responsible AI compliance.

## 11. Evaluation Limitations

- The baseline contains targeted exploratory regression cases.
- Conversation histories are abbreviated reconstructions.
- Some cases represent different assertions from the same execution.
- Results are not statistically representative of overall accuracy.
- Automated workflow replay has not yet been completed.
- The original Week 1 evaluation dataset remains unchanged.
- Formal before-and-after comparisons require consistent
  execution conditions.
- Linux and container deployment remain to be validated.

## 12. Conclusion

Baseline evaluation identified weaknesses in conversation
management, retrieval relevance, expense input integrity,
responsible AI and Critic validation.

The next development phases will introduce targeted
architectural enhancements and measure their effectiveness
using consistent regression and formal evaluation methods.

The objective is to demonstrate how systematic AI engineering
can improve the reliability of a hosted, resource-constrained
8B language model.


## 13. Automated Regression Execution Results

Execution Date: November 27, 2026

The regression test framework executed nine targeted
functional cases against the actual hosted inference
and retrieval workflow.

| Metric | Baseline Result |
|---|---|
| Total regression cases | 9 |
| Passed | 1 |
| Failed | 8 |
| Runtime exceptions | 0 |
| Targeted regression pass rate | 11.11% |
| Critic-approved responses | 8 |
| Critic-rejected responses | 1 |
| Average query latency | 14.02 seconds |
| Maximum query latency | 17.608 seconds |
| Minimum query latency | 1.284 seconds |
| Total measured latency | 126.140 seconds |

### Key Findings

1. Unrelated conversation history affected policy retrieval.
2. The expense calculator captured USD 120 instead of
   the explicitly requested USD 100.
3. Critic approval did not guarantee satisfaction of
   regression acceptance criteria.
4. Out-of-scope questions triggered evidence review retries
   instead of an explicit refusal.
5. Retrieval relevance varied based on conversation context.
6. Ambiguous expense queries required better clarification.

### Evaluation Artifacts

- Frozen dataset:
  data/evaluation/baseline_regression_v1.json

- Frozen automated execution results:
  data/evaluation/baseline_execution_results_v1.json

- Automated test implementation:
  tests/test_baseline_regression.py

### Evaluation Limitations

The regression suite is intentionally defect-focused and
does not represent overall application accuracy.

Some rule-based assertions may produce false negatives.
Conversation histories are reconstructed inputs rather
than complete persistent-state conversation replays.

The original baseline is preserved for traceability.
Any evaluation-rule improvements must be versioned
and documented separately.

Formal evaluation will use the complete evaluation
dataset and independently measured retrieval,
reasoning, performance and responsible AI metrics.
