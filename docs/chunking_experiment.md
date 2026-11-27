# Chunking Experiments

## 1. Objective

Evaluate fixed-size and structure-aware chunking strategies for the Employee Operations & Expense Compliance Copilot RAG pipeline.

The experiments compare chunk sizes, overlap settings, and preservation of Markdown document structure.

The objective is to identify a suitable chunking strategy for downstream retrieval evaluation. Chunk count and file size alone are not sufficient to determine retrieval quality.

## 2. Source Corpus

- Organization: GitLab
- Verified source documents: 9
- Source format: Markdown
- Original documents: `data/real_corpus/raw/gitlab/`
- Processed documents: `data/real_corpus/processed/gitlab/`
- Tokenizer: `BAAI/bge-small-en-v1.5`
- Embedding generation: Remote embedding API using `BAAI/bge-small-en-v1.5`
- Token counts are measured using the corresponding Hugging Face tokenizer.

## 3. Fixed-Size Chunking Experiments

| Experiment | Chunk size | Overlap | Total chunks | Output |
|---|---:|---:|---:|---|
| Fixed 1 | 300 | 50 | 165 | `gitlab_fixed_300_overlap_50.jsonl` |
| Fixed 2 | 500 | 100 | 105 | `gitlab_fixed_500_overlap_100.jsonl` |
| Fixed 3 | 800 | 100 | 61 | `gitlab_fixed_800_overlap_100.jsonl` |

Fixed-size chunking divides document text into overlapping token windows. It does not guarantee that chunks align with document section boundaries.

## 4. Structure-Aware Chunking Experiments

| Version | Total chunks | Notes |
|---|---:|---|
| Initial | 240 | Initial Markdown heading-aware implementation |
| V2 | 240 | Hugo shortcode handling adjusted |
| V3 | 241 | Added logical hierarchy for Hugo details sections |
| V4 | 224 | Removed heading-only chunks while retaining parent headings in section paths |

Output files:

- `gitlab_structure_aware_500_overlap_100.jsonl`
- `gitlab_structure_aware_v2_500_overlap_100.jsonl`
- `gitlab_structure_aware_v3_500_overlap_100.jsonl`
- `gitlab_structure_aware_v4_500_overlap_100.jsonl`

## 5. V4 Validation

V4 is the current structure-aware candidate.

| Validation | Result |
|---|---:|
| Documents processed | 9 |
| Total chunks | 224 |
| Unique chunk IDs | 224 |
| Unique document IDs | 9 |
| Minimum token count | 18 |
| Maximum token count | 500 |
| Heading hierarchy | Verified |
| Heading-only chunks | Removed |
| UTF-8 text preservation | Verified |

V4 preserves Markdown heading hierarchy and supports nested Hugo `details` sections. Parent headings are retained in section paths without creating standalone heading-only retrieval chunks.

Each chunk retains source metadata, including document ID, title, organization, source URL, license, section heading, section path, chunk index, and token count.

## 6. Current Assessment

V4 is the current structure-aware candidate, not a proven retrieval winner.

The fixed-size and structure-aware configurations will be compared using the same retrieval evaluation queries and relevance judgments.

The final selection will consider:

- Recall@1, Recall@3, Recall@5, and Recall@10
- Mean Reciprocal Rank (MRR)
- nDCG@k
- Retrieval performance by document category
- Retrieval latency

Retrieval performance will be measured separately from end-to-end answer quality.

## 7. Limitations

- The initial corpus contains only nine documents and is not representative of the final 500–2,000-document target.
- Chunk counts and file sizes do not establish retrieval quality.
- The tokenizer warning for sections longer than 512 tokens occurs during full-section tokenization; generated chunks were separately validated against the 500-token limit.
- Source page numbers are not available for these Markdown documents. Section paths are retained as structural provenance.

## 8. Next Steps

1. Build a chunk catalogue.
2. Create a retrieval evaluation set with expected relevant document and chunk IDs.
3. Implement dense retrieval using the configured embedding API.
4. Implement BM25 sparse retrieval.
5. Evaluate hybrid retrieval.
6. Add a reranker.
7. Compare all chunking configurations using identical evaluation queries.
8. Expand the real source corpus toward 500–2,000 unique documents.
