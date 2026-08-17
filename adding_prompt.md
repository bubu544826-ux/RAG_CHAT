You are a senior RAG / Search Engineer. Work directly in the current codebase to improve the RAG retrieval pipeline.

## Goal

Implement and evaluate these three improvements:

1. **Hybrid Search** — BM25 / keyword search + vector search
2. **Reranking** — rerank retrieved candidates before sending them to the LLM
3. **Query Rewrite / Multi-Query Retrieval** — rewrite or expand user queries to improve recall

Primary metrics:

* Recall@K
* Precision@K
* MRR
* NDCG

The goal is to prove that retrieval quality improves, not just to add features.

---

## 1. Analyze the Current System

First inspect the repository and identify:

* Current retrieval pipeline
* Vector database / search engine
* Embedding model
* Retriever implementation
* Current Top-K settings
* Existing BM25 / full-text search
* Existing reranker
* Existing query rewrite logic
* Evaluation dataset and tests

Do not duplicate functionality that already exists.

Briefly explain the current retrieval architecture and the main retrieval bottlenecks.

Then continue implementation without waiting for approval.

---

## 2. Establish a Baseline

Before modifying retrieval, evaluate the current system.

Measure:

* Recall@1 / @3 / @5 / @10
* Precision@5 / @10
* MRR
* NDCG@5 / @10

If an evaluation dataset already exists, reuse it.

If none exists, create a simple extensible format such as:

```json
{
  "query": "How many annual leave days do employees receive?",
  "relevant_doc_ids": ["doc_123"],
  "relevant_chunk_ids": ["chunk_123_4"]
}
```

Do not fabricate metrics if reliable labeled data is unavailable.

---

## 3. Implement Hybrid Search

Combine:

```text
Vector Search + BM25 / Keyword Search
```

Target flow:

```text
                Vector Search
               /
User Query ----
               \
                BM25 Search
                     ↓
                Result Fusion
                     ↓
                 Candidates
```

Prefer **Reciprocal Rank Fusion (RRF)** for combining rankings:

```text
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Do not directly add raw BM25 and vector similarity scores without normalization.

Support configurable modes:

```text
vector_only
lexical_only
hybrid
```

Prefer existing search/database capabilities instead of introducing unnecessary infrastructure.

---

## 4. Add a Reranker

Use a two-stage retrieval architecture:

```text
Query
  ↓
Hybrid Retrieval
  ↓
Top 30–100 Candidates
  ↓
Reranker
  ↓
Top 5–10
  ↓
LLM
```

The first retrieval stage should have a larger candidate pool than the final context size.

For example:

```text
candidate_k = 50
final_k = 5
```

Use an existing reranker if the project already provides one. Otherwise choose a practical solution that fits the current stack.

Keep useful metadata such as:

```text
document_id
chunk_id
retrieval_score
rerank_score
original_rank
final_rank
retrieval_source
```

---

## 5. Add Query Rewrite / Multi-Query Retrieval

Support two modes.

### Single Rewrite

```text
User Query
→ Rewrite
→ Retrieval
```

### Multi-Query

```text
User Query
→ Generate up to 3 search queries
→ Retrieve in parallel
→ Merge / deduplicate results
→ Rerank
```

Example:

```text
User:
How does employee leave work?

Rewrite:
employee annual leave policy
paid leave calculation rules
employee leave entitlement
```

Query rewriting must preserve important constraints such as:

* Dates
* Numbers
* Product names
* IDs
* Locations
* Negations
* Version numbers

Precise queries such as error codes or file names should be allowed to bypass rewriting.

---

## 6. Target Pipeline

The final retrieval pipeline should roughly be:

```text
User Query
    ↓
Query Rewrite / Multi-Query
    ↓
Vector Search + BM25
    ↓
RRF Fusion
    ↓
Candidate Pool
    ↓
Reranker
    ↓
Top-K Context
    ↓
LLM
```

Keep the implementation modular and compatible with the existing architecture.

Avoid unnecessary refactoring.

---

## 7. Evaluation

Run these experiments:

```text
A. Vector Search baseline
B. Hybrid Search
C. Hybrid + Reranker
D. Hybrid + Reranker + Query Rewrite
```

Produce a comparison like:

| Pipeline          | Recall@5 | Recall@10 | Precision@5 | MRR | NDCG@5 |
| ----------------- | -------: | --------: | ----------: | --: | -----: |
| Vector Baseline   |          |           |             |     |        |
| Hybrid            |          |           |             |     |        |
| Hybrid + Reranker |          |           |             |     |        |
| Full Pipeline     |          |           |             |     |        |

Explain which component improved which metric.

Do not claim improvement without measured evidence.

Also compare retrieval latency where practical.

---

## 8. Tests and Failure Handling

Add tests for:

* RRF ranking
* Hybrid result merging
* Duplicate removal
* Reranker ordering
* Query rewrite constraint preservation
* Multi-query merging
* Empty search results
* Component failures

Use graceful fallbacks:

```text
Query rewrite fails
→ use original query

Reranker fails
→ use fused retrieval ranking

BM25 fails
→ vector search only

Vector search fails
→ lexical search only
```

Do not allow optional retrieval improvements to crash the whole RAG request.

---

## 9. Configuration

Do not hardcode important parameters.

Make settings configurable, for example:

```yaml
retrieval:
  strategy: hybrid

  vector_top_k: 30
  lexical_top_k: 30

  fusion:
    strategy: rrf
    rrf_k: 60

  reranker:
    enabled: true
    candidate_k: 30
    final_k: 5

  query_rewrite:
    enabled: true
    mode: multi_query
    max_queries: 3
```

Use the project's existing configuration system where possible.

---

## Final Deliverable

After implementation, provide:

### Current Architecture

Briefly explain the original retrieval pipeline.

### Changes Made

List the important files changed and what was implemented.

### Evaluation Results

Show:

* Recall@K
* Precision@K
* MRR
* NDCG

for baseline and improved pipelines.

### Recommended Configuration

Provide recommended production values for:

```text
vector_top_k
lexical_top_k
candidate_k
final_top_k
rrf_k
multi_query_count
```

### Remaining Risks

Briefly explain any remaining retrieval weaknesses or limitations.

---

## Engineering Rules

* Inspect the existing implementation before changing it.
* Reuse existing infrastructure whenever possible.
* Avoid unrelated refactoring.
* Preserve backward compatibility.
* Make new functionality configurable.
* Do not fabricate benchmark results.
* Run existing and new tests.
* Fix regressions caused by your changes.
* Directly implement the solution instead of only proposing an architecture.

Prioritize retrieval improvements in this order:

```text
Recall@K
↓
MRR / NDCG
↓
Precision@K
```

The task is complete only when the retrieval improvements are implemented and their impact is evaluated.
