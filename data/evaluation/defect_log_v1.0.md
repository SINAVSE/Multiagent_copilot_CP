# GitLab Policy Copilot — UI Defect Log

**Version:** 1.0
**Status:** Open — initial findings from UI testing
**Project branch:** `feature/project-setup`
**Related dataset:** `data/evaluation/evaluation_gitlab_v1.0.json`

## Purpose

Record defects observed during manual UI testing, their evidence, expected behaviour, fix status and regression coverage. This file is separate from the gold evaluation dataset. Do not record a defect as fixed until the relevant regression test has passed.

## Severity guide

- **Critical:** Unsafe authorization, privacy breach, or a materially harmful policy conclusion.
- **High:** Incorrect compliance result, lost user-provided input, or materially wrong evidence/source.
- **Medium:** Incorrect tool selection, missing clarification, unsupported statement, or confusing answer.
- **Low:** Formatting, duplicate citation, or minor wording issue without changing the decision.

## Defect entries

### DEF-001 — Amount in natural-language query is not extracted
- **Area:** Planner / Executor / Expense Calculator
- **Severity:** High
- **Observed behaviour:** USD 250 and USD 80 were present in customer-meal queries but were not passed into the calculator; the UI asked for amounts already supplied.
- **Expected behaviour:** Preserve amount and currency from the query. Ask only for genuinely missing inputs, such as attendee count for a per-person limit.
- **Evidence:** USD 250 customer dinner and USD 80 business-meal UI tests.
- **Likely cause:** Under investigation; inspect planner extraction and executor argument mapping.
- **Fix status:** Open
- **Regression test:** Repeat both queries with Expense Detail unchecked; verify amount extraction and clarification only for attendee count.

### DEF-002 — Calculator selected for policy-only questions
- **Area:** Intent Planner
- **Severity:** Medium
- **Observed behaviour:** Calculator was selected for informational questions, including per-diem method and customer-gift policy questions, when no numerical assessment was requested or possible.
- **Expected behaviour:** Use retrieval only for policy explanations. Select the calculator only when a numerical assessment is requested and sufficient inputs/rules are available.
- **Evidence:** Per-diem method query; prospective-customer gift policy query; team-outing policy query.
- **Likely cause:** Under investigation; broad keyword triggers may be selecting the calculator.
- **Fix status:** Open
- **Regression test:** Ask “How is per diem calculated?” and “What is the gift policy?”; verify search-only selection.

### DEF-003 — Calculator inputs or policy limits are not propagated reliably
- **Area:** Executor / Policy Extractor / Expense Calculator
- **Severity:** High
- **Observed behaviour:** Some cases returned `unable_to_determine` despite an amount or relevant limit being available; in other cases the calculator result and final answer disagreed.
- **Expected behaviour:** Pass amount, currency, attendee count and policy limit consistently. Identify exactly which input is missing.
- **Evidence:** USD 150 team-member gift; USD 250 and USD 80 meal cases.
- **Likely cause:** Under investigation; trace values from planner output through executor and calculator.
- **Fix status:** Open
- **Regression test:** USD 150 qualifying non-cash gift should calculate USD 125 within cap and USD 25 excess; USD 250 dinner should request attendee count but retain USD 250.

### DEF-004 — Critic accepts contradictions
- **Area:** Critic
- **Severity:** High
- **Observed behaviour:** Critic passed answers that contradicted calculator results or made unsupported conclusions, including declaring an entire gift ineligible when gift type was unknown.
- **Expected behaviour:** Check consistency between user inputs, retrieved evidence, tool outputs and final answer; reject unsupported or contradictory conclusions.
- **Evidence:** Repeated USD 150 team-member gift tests.
- **Likely cause:** Under investigation; semantic consistency checks may be insufficient.
- **Fix status:** Open
- **Regression test:** Provide a correct calculator result with a deliberately contradictory draft answer; critic should reject it.

### DEF-005 — Unsupported conclusion about gift eligibility
- **Area:** Answer Generator
- **Severity:** High
- **Observed behaviour:** The answer applied the cash/cash-equivalent prohibition even though the gift type was unknown, and concluded the entire gift was ineligible.
- **Expected behaviour:** Calculate the cap/excess where possible, then ask whether the gift is cash/cash-equivalent or qualifying non-cash before applying the prohibition.
- **Evidence:** USD 150 team-member gift tests.
- **Likely cause:** Under investigation; conditional policy rule applied without checking its condition.
- **Fix status:** Open
- **Regression test:** Test USD 150 non-cash gift, gift card and cash gift separately.

### DEF-006 — Unsupported per-diem first/last-day rule
- **Area:** Answer Generator / Policy Retrieval
- **Severity:** Medium
- **Observed behaviour:** The answer introduced a first-day/last-day per-diem reduction not established by retrieved evidence.
- **Expected behaviour:** State only destination-specific rates and day-count rules supported by policy. Do not invent partial-day rules.
- **Evidence:** Four-day international per-diem UI tests.
- **Likely cause:** Under investigation; possible unsupported model inference.
- **Fix status:** Open
- **Regression test:** Ask for a four-day trip without destination; verify the system asks for destination and does not invent first/last-day treatment.

### DEF-007 — Incorrect or irrelevant retrieval metadata/source selection
- **Area:** Document Ingestion / Chunking / Retrieval
- **Severity:** High
- **Observed behaviour:** A team-member gifts section appeared under an unrelated WFH parent path; a parent-health-insurance question retrieved an irrelevant bereavement section.
- **Expected behaviour:** Preserve correct section hierarchy and avoid presenting irrelevant material as evidence.
- **Evidence:** Incorrect parent path `Approved WFH Equipment List > Team Member Significant Life Events and Team Member Gifts`; parent insurance query retrieved bereavement content.
- **Likely cause:** Under investigation; inspect heading hierarchy, chunk metadata and retrieval relevance.
- **Fix status:** Open
- **Regression test:** Retrieve team-member gifts and parent health-insurance questions; verify correct metadata and abstention where evidence is absent.

### DEF-008 — Duplicate source quotation/citation
- **Area:** Retrieval / Answer Generator
- **Severity:** Low
- **Observed behaviour:** Some answers repeated the same source section or quotation.
- **Expected behaviour:** Deduplicate identical evidence passages and citations while retaining distinct supporting sections.
- **Evidence:** Parental-leave answer with duplicate section citation; customer-meal answer with repeated USD 80 quotation.
- **Likely cause:** Under investigation; inspect chunk deduplication and citation assembly.
- **Fix status:** Open
- **Regression test:** Repeat parental-leave and business-meal queries; verify each distinct passage is cited once.

### DEF-009 — Fraud-reporting answer uses incorrect source
- **Area:** Retrieval / Answer Generator
- **Severity:** High
- **Observed behaviour:** Fraud-reporting answer cited GITLAB-0004 and GITLAB-0006 instead of expected GITLAB-0005; confidentiality detail was missing.
- **Expected behaviour:** Retrieve GITLAB-0005, “Reporting Suspected Fraud”. Include confidentiality details only if explicitly stated in that source; do not invent reporting channels.
- **Evidence:** Fraud/illegal-conduct UI test.
- **Likely cause:** Under investigation; inspect query matching and source ranking.
- **Fix status:** Open
- **Regression test:** Ask how to report suspected fraud and whether it is confidential; verify source ID and source-backed wording.

### DEF-010 — Evidence gaps are not consistently handled
- **Area:** Retrieval / Answer Generator / Planner
- **Severity:** High
- **Observed behaviour:** Team-outing budget and parent health-insurance coverage were not supported by retrieved evidence. General expense or unrelated HR content risked implying an answer.
- **Expected behaviour:** State that the corpus does not establish the rule, avoid inventing limits/approval requirements, and refer to the policy owner.
- **Evidence:** USD 900 team outing; parent health-insurance coverage.
- **Likely cause:** Under investigation; retrieval fallback may be too permissive.
- **Fix status:** Open
- **Regression test:** Repeat both queries; verify an explicit evidence-gap response without unsupported claims.

### DEF-011 — Approval boundary could be clearer
- **Area:** Answer Generator / Safety
- **Severity:** Medium
- **Observed behaviour:** Prompt-injection test was refused, but the response could state more clearly that the copilot cannot approve expenses, bypass policy or authorize reimbursement.
- **Expected behaviour:** Distinguish a non-binding policy assessment from formal approval, payment authorization or HR decisions.
- **Evidence:** Prompt-injection request to approve a USD 500 expense automatically.
- **Likely cause:** Under investigation; refusal response may be too generic.
- **Fix status:** Open
- **Regression test:** Repeat direct and indirect approval-bypass requests; verify explicit refusal and safe alternative.

## Updating this log

For each new issue, add a new `DEF-###` entry. Preserve the original observation and evidence after a fix. Update the fix status and add:

- **Resolution:** What code/configuration changed.
- **Files changed:** Exact project file paths.
- **Regression result:** Pass/Fail, test ID or command, and date.
- **Verified by:** Test run or reviewer.

Do not mark a defect `Verified` just because code was edited. Re-run the relevant test and, where appropriate, the full automated test suite.
