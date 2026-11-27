# CAP-13: Retrieval and Citation Improvements

## 1. Objective

Improve evidence retrieval, citation integrity, ranking,
context efficiency and reliability of the Employee Operations
and Expense Compliance Copilot.

The system operates under the constraint of a hosted 8B
language model. High-quality retrieval and deterministic
evidence validation reduce dependence on model reasoning.

## 2. Implementation Summary

### 2.1 Citation Integrity

- Shared structured JSON response parsing.
- Validation of required response fields.
- Rejection of unknown and duplicate citation identifiers.
- Citation metadata derived from retrieved evidence.
- Insufficient-evidence handling for uncited sufficient answers.
- No model invocation when evidence is unavailable.

### 2.2 Evidence Selection

- Duplicate chunk identifier removal.
- Exact normalized-content deduplication.
- Preservation of distinct policy conditions.
- Preservation of source provenance and original scores.
- Final top-k selection after duplicate removal.

### 2.3 Policy Context Selection

- Removed hardcoded GitLab document identifier.
- Metadata-based matching of explicit policy contexts.
- Support for five configured policy families.
- Comparison and multi-policy query protection.
- Conservative handling of unavailable explicit policy context.
- General retrieval retained for ambiguous expense queries.

### 2.4 Metadata-Aware BM25

- Document title indexing.
- Section heading and hierarchical section path indexing.
- Unicode NFKC normalization and case folding.
- Duplicate metadata field removal.
- Configurable text-only and metadata-aware indexing.
- Metadata-aware indexing retained as the default configuration.

### 2.5 Hybrid Reciprocal Rank Fusion

- Maximum one contribution per retrieval method per chunk.
- Unique candidate ranks after duplicate removal.
- Strict retrieval parameter validation.
- Invalid chunk identifier exclusion.
- Conflicting evidence detection.
- Rank fusion observability.
- Existing RRF constant of 60 retained.

### 2.6 Cross-Encoder Reranking

- Metadata-aware query-passage preparation.
- Text-only comparison configuration.
- Numeric relevance score validation.
- Rejection of NaN and infinite scores.
- Support for valid negative relevance scores.
- Strict configuration validation.
- Existing BAAI/bge-reranker-base model retained.

### 2.7 Chunk Integrity Validation

- Strict validation of policy identity and content.
- Compatible optional metadata merging.
- Rejection of conflicting non-null metadata.
- Preservation of original chunk objects.
- Offline Dense/BM25 compatibility validation.

### 2.8 Evidence Context Budget

- Configurable UTF-8 evidence byte budget.
- Default maximum context size of 12,000 bytes.
- Whole-chunk selection without policy text truncation.
- Preservation of citation identifiers and policy conditions.
- Context selection and exclusion metrics.
- Complements CAP-12 runtime token budget enforcement.

### 2.9 Supplied Evidence Validation

- Common evidence preparation for normal retrieval and
  directly supplied retrieval results.
- Cross-query evidence reuse protection.
- Policy context validation.
- Duplicate evidence removal.
- Context budget enforcement.
- No repeated retrieval or reranking required.

## 3. Validation Results

| Test Category | Passed |
|---|---:|
| Previous CAP-12 regression suite | 422 |
| Citation integrity | 32 |
| Evidence Selector | 32 |
| Evidence retrieval integration | 6 |
| Policy Context Selector | 33 |
| Policy context integration | 4 |
| Metadata-aware BM25 | 33 |
| Hybrid RRF improvements | 43 |
| Cross-encoder validation | 55 |
| Chunk Integrity Validator | 34 |
| Metadata compatibility integration | 6 |
| Evidence context budget | 27 |
| Context budget integration | 7 |
| Supplied evidence validation | 32 |
| Total | 766 |

Final observed regression execution:

- Passed: 766
- Failed: 0
- Intentionally skipped hosted tests: 14
- Existing HTTPX deprecation warnings: 2
- Execution time: 27.33 seconds
- New CAP-13 tests: 344

The offline suite verifies functional behavior under the
tested conditions. It does not establish live end-to-end
answer accuracy.

## 4. Comparative Retrieval Evaluation

### 4.1 Experimental Design

Dataset: retrieval_eval_v1.0.json

- Frozen real-policy evaluation queries: 20
- Evaluated corpus chunks: 224
- Configurations: 3
- Local retrieval evaluations: 60
- Maximum retrieved candidates: 10
- Hosted API calls: 0

Configurations:

- B0: Original text-only BM25 and original tokenizer.
- B1: Text-only BM25 with Unicode normalization.
- B2: Metadata-aware BM25 with Unicode normalization.

The same corpus and frozen gold labels were used across
all three configurations.

The evaluation artifact records corpus and dataset hashes,
aggregate metrics, query-level rankings and metric deltas.

### 4.2 Baseline Reproducibility

The historical BM25 benchmark was reproduced exactly.

- Recall@1 difference: 0
- Recall@3 difference: 0
- Recall@5 difference: 0
- MRR difference: 0
- Identical query rankings: 20 out of 20

### 4.3 Experimental Results

| Metric | B0 Original | B1 Normalized | B2 Metadata |
|---|---:|---:|---:|
| Recall@1 | 0.5500 | 0.5500 | 0.5500 |
| Recall@3 | 0.7750 | 0.7750 | 0.8000 |
| Recall@5 | 0.8500 | 0.8500 | 0.9000 |
| Recall@10 | 0.9500 | 0.9500 | 0.9500 |
| MRR@10 | 0.7071 | 0.7071 | 0.7354 |
| Document Hit Rate@5 | 0.9500 | 0.9500 | 0.9500 |
| Zero-recall queries@5 | 3 | 3 | 2 |
| Mean latency (ms) | 1.239 | 1.311 | 1.292 |
| P95 latency (ms) | 1.865 | 1.680 | 1.820 |

Recall is calculated as the fraction of expected gold
chunks retrieved within the specified cutoff, averaged
across evaluation queries.

MRR measures the mean reciprocal rank of the first
retrieved gold chunk.

### 4.4 Observed Improvements

- Recall@3 increased by 2.5 percentage points.
- Recall@5 increased by 5 percentage points.
- Relative Recall@5 improvement was approximately 5.88%.
- Absolute MRR improvement was approximately 0.0283.
- Relative MRR improvement was approximately 4.01%.
- Zero-recall queries@5 decreased from three to two.
- Three queries improved in first relevant rank.
- Sixteen queries retained their first relevant rank.
- One query experienced a ranking regression.

Unicode normalization alone did not change retrieval
metrics on the evaluated dataset.

The observed ranking improvements are attributable to
metadata indexing within this controlled comparison.

### 4.5 Significant Query-Level Changes

| Query | Original Rank | Enhanced Rank |
|---|---:|---:|
| R01 | 10 | 3 |
| R02 | 3 | 1 |
| R17 | 3 | 2 |
| R07 | 1 | 2 |
| R08 | 8 | 8 |
| R20 | Not retrieved | Not retrieved |

## 5. Failure Analysis

### 5.1 R07: Cash or Cash-Equivalent Gifts

Classification: Ranking regression.

Expected evidence:
GITLAB-0006-S0004, Prohibited Activities.

Enhanced top-ranked evidence:
GITLAB-0006-S0008, Accepting Gifts and Entertainment
General Guidelines.

The query concerns giving cash-equivalent gifts.
The enhanced first result concerns accepting gifts.

Both contain relevant cash-equivalent restrictions,
but the expected evidence is more directly applicable
to the requested action.

The likely explanation is overlapping lexical and
metadata signals between giving and receiving gifts.

The correct gold evidence remains at rank two.

Decision:
Retain metadata-aware BM25 and evaluate action-sensitive
semantic ranking during the full hybrid benchmark.

### 5.2 R08: Government Official Definition

Classification: Potential incomplete gold-label coverage.

Expected evidence:
GITLAB-0006-S0002, Gifts and Entertainment Policy,
Definitions.

Actual top-ranked evidence in both configurations:
GITLAB-0004-S0004, Anti-Corruption Policy,
Important Concepts.

The actual top-ranked chunk explicitly defines a
government official and represents a potentially valid
alternative source.

The frozen expected chunk remains at rank eight.

Decision:
Preserve the frozen benchmark and its original metrics.
Record the alternative relevant evidence for human
review in a future evaluation dataset version.

Do not retroactively modify the gold labels to improve
reported benchmark performance.

### 5.3 R20: Reimbursement Policy Identification

Classification: Gold-evidence retrieval miss.

Expected evidence:

- GITLAB-0002-S0003, Policy Links and Resources.
- GITLAB-0002-S0004, Ways to spend company money.

Both expected chunks direct employees to the Global
Travel and Expense Policy.

Neither configuration retrieved these expected chunks
within the top ten results.

The enhanced ranking returned unrelated anti-corruption
sections and a specific phone-service reimbursement rule.

This suggests a lexical relevance limitation when the
query asks which policy should be consulted.

Decision:
Evaluate semantic retrieval, query decomposition and
cross-encoder ranking against this case in the full
hybrid benchmark.

## 6. Architecture Decision

Retain metadata-aware BM25 as the default configuration.

The controlled comparison demonstrated better Recall@5
and MRR without additional hosted API consumption.

The text-only configuration remains available for
reproducibility and future ablation experiments.

No hosted model or vector-index changes are required
for the BM25 enhancement.

## 7. Compensating for a Small Language Model

The retrieval architecture reduces reasoning demands
on the hosted 8B model through:

- Improved lexical evidence discovery.
- Hybrid dense and sparse retrieval.
- Cross-encoder relevance ranking.
- Explicit policy context selection.
- Deterministic evidence integrity checks.
- Citation validation.
- Duplicate evidence removal.
- Bounded evidence context.
- Conservative insufficient-evidence responses.
- Prevention of accidental cross-query evidence reuse.

The goal is to supply focused, reliable policy evidence
rather than expecting the model to infer missing facts
or reconcile unrelated policy documents.

## 8. Limitations and Outstanding Evaluation

1. The evaluation dataset contains only 20 queries and
   has informed development decisions. It is not an
   independent held-out test dataset.

2. The current measured corpus contains 224 chunks.
   Mandatory real-document corpus expansion remains
   part of CAP-14.

3. BM25 improvements do not independently establish
   improvements in enhanced hybrid retrieval or
   end-to-end answer accuracy.

4. Cross-encoder metadata may reduce the available
   sequence space for policy text.

5. Policy context detection uses five configured
   families and document-title matching. It is not
   a universal semantic policy classifier.

6. Explicit context selection can exclude relevant
   cross-document evidence.

7. Comparison queries do not guarantee balanced
   representation of every requested policy.

8. Whole-chunk context budgeting may exclude important
   policy exceptions and does not backfill beyond the
   initial top-k selection.

9. The evidence byte budget is not an exact model
   token count or a complete prompt budget.

10. Supplied evidence validation does not independently
    authenticate its origin from the trusted corpus.

11. Revalidating prepared evidence may overwrite
    earlier selection diagnostic counters.

12. Actual Dense/BM25 and ChromaDB compatibility
    remains subject to real-corpus integration testing.

13. Hosted integration and end-to-end evaluations
    remain outstanding under the shared API quota
    constraint.

14. Organization-specific applicability must be
    addressed during corpus expansion to avoid
    combining incompatible employer policies.

15. Retrieval miss classifications are based on
    inspected result excerpts. Detailed causal score
    attribution has not been performed.

## 9. Evaluation Artifacts

Evaluator:
src/rag/evaluation/bm25_comparison.py

Results:
data/evaluation/real_corpus/cap13_bm25_comparison_v1.0.json

Frozen query dataset:
data/evaluation/real_corpus/retrieval_eval_v1.0.json

Historical benchmark:
data/evaluation/real_corpus/bm25_retrieval_results_v1.0.json

## 10. Reproduction Commands

Offline regression:

    python -m pytest -q --tb=short -rs

BM25 comparison:

    python -m src.rag.evaluation.bm25_comparison

Hosted integration tests are intentionally disabled
for offline regression.

## 11. Completion Status

CAP-13 implementation, offline regression validation,
BM25 comparative evaluation and query-level failure
analysis are complete.

Full enhanced hybrid retrieval and end-to-end
validation remain outstanding and must not be
reported as completed.

Next milestone: CAP-14 real-world policy corpus expansion.
