# Prompt Engineering Baseline Experiment

**Project:** Employee Operations & Expense Compliance Copilot
**Experiment:** Single-Agent, Single-Prompt Comparison
**Version:** 1.0
**Model:** Qwen/Qwen3-8B
**Status:** Planned

## 1. Objective

Evaluate how effectively a small language model handles employee operations and expense compliance requests using three prompting strategies, without Retrieval-Augmented Generation (RAG), external tools, or multi-agent orchestration.

The experiment will identify model weaknesses, compare prompting strategies, and establish a baseline for subsequent RAG and agentic implementations.

## 2. Prompting Strategies

### Strategy 1: Structured Output with Few-Shot Examples
- Provide explicit instructions and a fixed JSON output schema.
- Include representative examples with expected outputs.
- Evaluate structured response validity and decision accuracy.

### Strategy 2: Self-Consistency
- Generate three independent responses to the same query.
- Compare the final decisions and supporting calculations.
- Record disagreements and evaluate whether consistency improves correctness.
- Do not assume majority agreement guarantees correctness.

### Strategy 3: Decomposition Prompting
Break each request into the following tasks:
1. Identify the request type.
2. Extract relevant information.
3. Identify applicable synthetic policy rules.
4. Perform calculations where required.
5. Identify missing information or conflicts.
6. Determine the compliance status.
7. Produce the final structured response.

## 3. Experimental Controls

The following conditions must remain consistent across strategies:

- Same model: Qwen/Qwen3-8B.
- Same synthetic policy baseline: synthetic_policy_v1.0.md.
- Same evaluation dataset: evaluation_v1.0.json.
- Same user queries.
- Same output schema and decision-status definitions.
- Same model generation settings wherever technically possible.
- No RAG, external tools, or multi-agent workflows.
- No changes to the frozen evaluation dataset.

For self-consistency, three independent generations will be performed per query.

## 4. Evaluation Dataset

The frozen evaluation dataset contains 60 test cases across six categories:

- Travel and per-diem.
- Customer gifts and entertainment.
- Team outings.
- Scanned bill and invoice analysis.
- General HR and employee policies.
- Ambiguous, unsafe and exceptional cases.

OCR-dependent cases require actual document fixtures or equivalent supplied document content. Cases that cannot be tested fairly must be marked accordingly rather than assigned assumed results.

## 5. Evaluation Metrics

| Metric | Description |
|---|---|
| Overall accuracy | Correctness compared with gold answers |
| Status accuracy | Correct compliance-status prediction |
| Calculation accuracy | Correct deterministic calculations |
| JSON validity | Compliance with the output schema |
| Clarification accuracy | Correct identification of missing information |
| Refusal accuracy | Correct handling of unsafe or unsupported requests |
| Evidence accuracy | Correct use of the supplied synthetic policy |
| Latency | Response generation time |
| Token usage | Input and output tokens, when available |

## 6. Failure Taxonomy

Record observed failures using these categories:

- Hallucinated policy or unsupported exceptions.
- Arithmetic and calculation errors.
- Incorrect compliance status.
- Missing-information handling failures.
- Incorrect refusal or clarification.
- JSON or schema violations.
- Policy interpretation errors.
- Unsupported evidence or citations.
- Inconsistent responses across repeated generations.
- Failure to follow instructions.

Each failure record must include the test ID, strategy, expected result, actual result, failure category and explanation.

## 7. Result Recording

Save raw model outputs separately for each strategy in JSONL format.

Maintain a comparison dataset containing:

- Test ID.
- Strategy.
- Expected status.
- Actual status.
- Correctness.
- JSON validity.
- Failure category.
- Latency.
- Token usage, if available.

## 8. Success Criteria

The experiment must produce:

1. Results for all eligible evaluation cases.
2. A quantitative comparison of the three prompting strategies.
3. A categorized analysis of small-model failure modes.
4. Representative examples of incorrect outputs.
5. An explanation of the strengths and weaknesses of each strategy.
6. A documented baseline for future RAG and multi-agent comparisons.

The existing evaluation dataset remains frozen as version 1.0 throughout this experiment.

## 9. Scope Limitations

This experiment evaluates prompting alone.

It does not evaluate:

- Document retrieval or RAG.
- Hybrid search or reranking.
- External tool execution.
- Multi-agent planning and orchestration.
- Real employee records or live reimbursement approvals.

All policy decisions use the controlled synthetic policy baseline and must not be represented as actual employer policy decisions.
