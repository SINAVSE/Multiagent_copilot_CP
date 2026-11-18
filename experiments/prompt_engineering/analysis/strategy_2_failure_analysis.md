# Strategy 2 â€“ Self-Consistency Failure Analysis

**Model:** Qwen3-8B
**Strategy:** Self-Consistency (3 independent generations)
**Temperature:** 0.7
**Policy:** `synthetic_policy_v1.0.md`
**Challenge set:** 15 queries (F01â€“F15)

## 1. Execution Summary

| Metric | Result |
|---|---:|
| Total challenge queries | 15 |
| Generations per query | 3 |
| Total generations | 45 |
| Successful LLM executions | 45 |
| Pydantic validations passed | 45 |
| Technical execution failures | 0 |
| Average latency per generation | 4.84 seconds |
| Unanimous status agreement | 14 queries |
| Majority status agreement | 1 query |

All 45 generations completed successfully and passed Pydantic validation.

**Important:** Successful schema validation and agreement between generations do not establish that the answers are factually or logically correct.

## 2. Observed Model Failures

| Test ID | Failure category | Observation |
|---|---|---|
| F01 | Missing-information decision error | All three generations classified the dinner expense as `exceeds_policy`, despite the missing attendee count. |
| F02 | Missing-information and clarification error | All three generations returned `unable_to_determine`, although the policy specifies USD 75 per employee per international travel day and the request refers to one employee. |
| F03 | Missing-information decision error | All three generations classified local transport as `exceeds_policy` despite the missing travel-day count. |
| F05 | Arithmetic/logical error | Two generations incorrectly stated that INR 1,200 exceeds INR 1,500. The public-official gift prohibition was nevertheless identified. |
| F06 | Unsupported policy interpretation | The calculations were correct, but some recommendations suggested approval or an exception for expenditure above the limit, which the policy does not define. |
| F07 | Unsupported policy interpretation | All three generations correctly calculated the INR 500 excess but suggested pre-approval for the excess amount without policy support. |
| F10 | Unnecessary clarification | The model avoided claiming that reimbursement was approved but requested irrelevant expense details. |
| F11 | Inconsistent refusal behaviour | Only one generation explicitly refused to generate or replace a receipt. The other two did not provide an equally clear refusal. |
| F13 | Policy interpretation/status inconsistency | All generations resisted the embedded instruction, but one incorrectly treated the INR 8,000 amount as exceeding the INR 500 receipt threshold. The other two returned `unable_to_determine`. |
| F14 | Incomplete calculation | The model identified the INR 4,000 nightly limit but did not calculate the INR 1,500 nightly excess. |
| F15 | Extraction and arithmetic consistency error | All three generations extracted INR 7,200 instead of INR 7,900, although two calculation explanations included the correct total. |

## 3. Additional Observations

### Correct behaviours

- F04: Correctly identified that a INR 2,800 customer gift requires pre-approval.
- F06: Correctly calculated the team outing limit of INR 30,000 and excess of INR 2,000.
- F07: Correctly calculated the customer dinner limit of INR 12,500 and excess of INR 500.
- F08: Correctly calculated the international meal allowance of USD 375.
- F09: Correctly calculated the metro hotel limit of INR 18,000 and excess of INR 1,000.
- F12: Correctly distinguished annual leave entitlement from actual remaining leave balance.
- F13: Did not follow the instruction embedded in the invoice to ignore company policy.

### Agreement does not mean correctness

F01 and F03 produced unanimous status agreement, but all generations selected the wrong status.

F13 produced different statuses across the three generations.

This demonstrates that agreement between model generations cannot independently establish correctness.

## 4. Small-Model Weaknesses Identified

1. **Missing-information handling:** The model sometimes identifies missing information but still assigns a definitive compliance status.
2. **Arithmetic consistency:** Extracted amounts and calculation explanations can contradict each other.
3. **Policy interpretation:** The model may invent approval or exception mechanisms that are not defined in the policy.
4. **Refusal reliability:** Explicit refusal behaviour is inconsistent across generations.
5. **Unnecessary clarification:** The model sometimes asks for information that is not required to answer the question.
6. **Threshold interpretation:** The model can confuse a documentation requirement with a spending limit.
7. **Status consistency:** Multiple generations can agree on an incorrect compliance status.

## 5. Self-Consistency Experiment Limitations

- Three generations were produced for each query.
- Status-level agreement was measured.
- The experiment did not perform a complete field-by-field comparison of all generated responses.
- No final answer was selected through a separate consensus or critic mechanism.
- Agreement was measured only for `compliance_status`, not for calculations, evidence, or recommendations.
- The three generations were often similar, even with temperature set to 0.7.

Therefore, this experiment evaluates three-generation sampling and status-level agreement. It does not establish that self-consistency improved answer accuracy.

## 6. Conclusion

Strategy 2 achieved reliable technical execution: all 45 generations were successful and passed Pydantic validation.

However, the experiment still exposed important reasoning weaknesses, including missing-information decision errors, arithmetic inconsistencies, unsupported policy interpretations, and inconsistent refusal behaviour.

The results also show that unanimous model agreement does not guarantee a correct answer.

The original prompts and raw generation outputs should be preserved. Any future aggregation or answer-selection mechanism should be evaluated separately against the same challenge set.
