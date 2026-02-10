# Module 3: Indexing Strategies - Instructor Notes

---

## 1. Opening: What is Indexing in RAG?

### Key Slide: "Indexing in RAG: Preparing Knowledge for Retrieval"

> **Opening Line:** "We've learned to embed text and chunk documents. Now: how do we ORGANIZE that knowledge so we can find it when we need it?"

**Core Definition:**
Indexing is the **offline ingestion step** where raw documents are converted into a **searchable structure** that retrieval can query efficiently.

**The Indexing Pipeline:**
```
Documents → Cleaning + Chunking → Embeddings → Metadata Attachment → Index Storage
```

**What Gets Stored in an Index:**
```
chunk_id → embedding vector + metadata + chunk text/reference
```

Optimized for **fast top-k retrieval** at query time.

---

## 2. ⚠️ CRITICAL: Two Types of "Indexing"

### The Confusion Students Will Have

Students often confuse these two completely different concepts:

| | RAG-Level Indexing | DB-Level Indexing |
|---|---|---|
| **What** | How you organize knowledge for retrieval | How vector DBs store/search vectors internally |
| **Examples** | Chunking, summaries, hierarchies, metadata | HNSW, IVF, PQ, DiskANN |
| **Who controls** | You (the developer) | The vector database |
| **Question it answers** | "What knowledge is retrievable?" | "How fast can we search?" |
| **This module** | ✅ **Primary focus** | Mentioned but abstracted away |

> **Teaching Point:** "When you hear 'indexing' in RAG, think about *knowledge organization*, not database internals. The vector DB handles the fast search algorithms for you."

---

## 3. Why Indexing is Required in RAG

### Key Slide: "Why Indexing is Required in RAG"

**Without Indexing:**
- You'd do a near **full scan** over all chunks every query
- High latency, high cost
- Doesn't scale beyond small datasets

**With Indexing:**
- Retrieval is fast (sub-second for millions of docs)
- Good indexing (chunking + metadata) = right context retrieved = fewer hallucinations

**The RAG Architecture Diagram:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      INDEXING (Offline)                         │
│  Documents → Chunking → Embedding → Indexing → Vector Database  │
└─────────────────────────────────────────────────────────────────┘
                                                    ↑
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL (Online)                          │
│  Query → Embed Query → Search Index → Retrieve Top-K → LLM     │
└─────────────────────────────────────────────────────────────────┘
```

> **Key insight:** Indexing happens OFFLINE. Retrieval happens ONLINE at query time.

---

## 4. The Five Indexing Strategies (Demo)

### Framework: LlamaIndex

> **Why LlamaIndex for this module?** It has cleaner abstractions for different index types than LangChain.

### Strategy Overview Table

```
                    Vector    Summary   Tree      Keyword   Hybrid
Small dataset       ✓ Good    ✓ Good    Overkill  ✓ Good    Overkill
Large dataset       ✓ Best    ✗ Slow    ✓ Good    ✓ Fast    ✓ Best
Semantic queries    ✓ Best    ✓ Good    ✓ Good    ✗ Bad     ✓ Best
Exact match (IDs)   ✗ Bad     ✗ Bad     ✗ Bad     ✓ Best    ✓ Good
Hierarchical docs   ✗ Bad     ✓ OK      ✓ Best    ✗ Bad     ✓ Good
```

### 4.1 Vector Index (Flat Index)

**What it is:** Embed all chunks, retrieve by semantic similarity.

```
Build time:  Documents → Chunk → Embed → Store vectors
Query time:  Query → Embed → Find nearest vectors → Return top-K
```

**When to use:**
- Default choice for most RAG applications
- Semantic search (synonyms, paraphrases work)
- Scale: Works to millions of documents

**Code:**
```python
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)
results = index.as_query_engine(similarity_top_k=3).query("How do I reset my password?")
```

### 4.2 Summary Index

**What it is:** Store full documents, LLM evaluates relevance for each.

**How it works:**
- No chunking, no embeddings
- LLM reads each document and judges "is this relevant?"
- O(n) complexity - slow for large datasets

**When to use:**
- Small document collections (<50 docs)
- High-level summarization queries
- When you need full document context

**Warning:** Gets slow quickly. 100 docs = ~20 seconds.

### 4.3 Tree Index (Hierarchical)

**What it is:** Build a tree from leaf chunks up to root summary.

```
        [Root Summary]
             │
    ┌────────┴────────┐
[Branch A]        [Branch B]
    │                 │
┌───┴───┐         ┌───┴───┐
L   L   L         L   L   L
```

**How it works:**
- Leaf nodes = actual document chunks
- Branch nodes = LLM-generated summaries of children
- Root = summary of entire corpus
- Query: Start at root, drill down to relevant leaves

**When to use:**
- Large, hierarchical documents (textbooks, manuals)
- Queries that need navigating structure
- When you want coarse-to-fine retrieval

### 4.4 Keyword Table Index

**What it is:** Traditional inverted index based on keyword extraction.

**How it works:**
- Extract keywords from each document
- Build mapping: keyword → document IDs
- Query: Extract query keywords → Find matching docs

**When to use:**
- Exact matches needed (error codes, IDs, names)
- Fast keyword lookup
- When semantic similarity isn't needed

### 4.5 Hybrid Retrieval (PRODUCTION PATTERN)

**What it is:** Combine vector + keyword search, merge results.

```
Query → Vector Search → Candidates A
    └─→ Keyword Search → Candidates B
                  ↓
        Combine using RRF (Reciprocal Rank Fusion)
                  ↓
            Final Results
```

**Why hybrid?**
- Vector catches meaning ("authentication problem" → "login issue")
- Keyword catches exact matches ("TICK-1234", "Error 504")
- Together: best of both worlds

**Reciprocal Rank Fusion (RRF):**
```python
RRF_score(doc) = Σ 1/(k + rank_in_list)

Example:
  Doc appears rank 2 in vector, rank 5 in keyword
  RRF = 1/(60+2) + 1/(60+5) = 0.016 + 0.015 = 0.031
```

**This is what production systems actually use.**

---

## 5. Demo Walkthrough

### Demo Structure

The demo shows all 5 strategies with the same query:
```
Query: "How do I fix authentication issues after password reset?"
```

**What to highlight:**
1. Same query, different results from each strategy
2. Trade-offs: speed vs quality vs use case
3. Hybrid as the production pattern

### Key Teaching Points During Demo

| Part | Key Point |
|------|-----------|
| Vector Index | "This is your default. Start here." |
| Summary Index | "Notice it's SLOW. Great for small sets." |
| Tree Index | "See how it navigates the hierarchy?" |
| Keyword Index | "Fast, but misses semantics." |
| Hybrid | "THIS is what you deploy to production." |

---

## 6. Why Multiple Indexes Exist in Production RAG

### Key Slide: "Why multiple indexes exist in production RAG"

Now that we understand the individual strategies, let's see how production systems combine them.

Production systems don't use just one index. Here's why:

### 6.1 Different Data Types Need Different Retrieval

| Data Type | Best Approach |
|-----------|---------------|
| Long docs (PDFs) | Recursive chunking + dense vectors |
| FAQs | Keyword/hybrid for exact match |
| Tickets | Recent bias + keyword for error codes |
| Code | Code-aware embeddings + symbol filters |
| Tables | SQL-backed retrieval |

### 6.2 Semantic vs Exact-Match Are Both Needed

```
Dense Vector Index       → Great for MEANING ("how do I reset password?")
Keyword/Inverted Index   → Great for EXACT TERMS ("Error 504", "TICK-1234")
Hybrid                   → Combines both for best results
```

### 6.3 Domain Separation Improves Precision

**Example Indexes:**
- "HR Policy" index
- "Engineering Runbooks" index  
- "Product Specs" index

> **Why?** Reduces noise. Query about vacation policy shouldn't search engineering docs.

### 6.4 Different Freshness/Update Cadences

| Content Type | Update Frequency |
|--------------|------------------|
| Policies | Weekly |
| Incident tickets | Hourly |
| Logs | Continuously |

Separate indexes = separate ingestion pipelines and SLAs.

### 6.5 Governance, Permissions, and Tenancy

- Enterprises isolate indexes by tenant/team/sensitivity
- Routing + metadata filters enforce access control

---

## 7. Query Routing by Intent

### Key Slide: "Why route queries differently based on intent"

**The Key Insight:** Not all queries should go to the same index!

| Query Intent | Example | Best Index | Strategy |
|--------------|---------|------------|----------|
| **Fact lookup** | "What's the PTO limit?" | FAQ index | Keyword + hybrid |
| **Troubleshooting** | "Error 504 on service X" | Incidents index | Recent + keyword |
| **Conceptual** | "Explain indexing in RAG" | Docs index | Dense vector + reranker |
| **Latest status** | "Current outage status?" | Incidents index | Time-sliced, recency weighted |
| **Code questions** | "Where is retry logic?" | Code index | Code-aware embeddings |
| **Metrics** | "What was Q4 churn?" | Tables | SQL-backed retrieval |

### Routing Pattern Overview

```
User Request → Input Normalizer → Router (Intent + Constraints)
                                      │
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
              FAQ Index         Docs Index        Incidents Index
              (Hybrid)          (Dense)           (Recent + Keyword)
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ↓
                            Combine + Rerank
                                      ↓
                              LLM + Generator
                                      ↓
                               Final Answer
```

### What Retrieval Routing Controls

| Decision Point | Implementation | Impact |
|----------------|----------------|--------|
| Which index to query | Vector, keyword, tree, summary, hybrid | Matches intent to strategy |
| Which chunks are eligible | Whole docs vs structure-aware chunks | Preserves context appropriately |
| Which filters apply | Metadata (category, priority, domain) | Prevents irrelevant results |
| How results are ranked | Similarity thresholds, MMR for diversity | Relevant AND sufficient |

> **Warning:** Without routing, RAG becomes blind top-K search—flooding the LLM with irrelevant context.

---

## 8. Common Student Questions

### Q: "Which indexing strategy should I use?"

**A:** Start with Vector Index. It covers 90% of use cases. Graduate to Hybrid for production.

### Q: "When would I use Tree Index?"

**A:** Large hierarchical documents where you need coarse-to-fine retrieval. Rare in practice.

### Q: "Why is Summary Index so slow?"

**A:** It makes an LLM call for every document to judge relevance. O(n) complexity.

### Q: "Can I combine LlamaIndex indexes with LangChain chains?"

**A:** Yes! Use `index.as_retriever()` to get a LangChain-compatible retriever.

### Q: "What about reranking?"

**A:** Covered in Module 4. Reranking takes retrieval results and re-orders by relevance using a cross-encoder.

---

## 9. Key Takeaways (Summary Slide)

1. **Indexing = Knowledge Organization** for efficient retrieval
2. **RAG-level indexing** (this module) ≠ DB-level indexing (HNSW, IVF)
3. **Production systems use multiple indexes** with routing
4. **Vector Index is your default** - works for most cases
5. **Hybrid (vector + keyword) is the production pattern**
6. **Different query intents need different indexes**
7. **Route queries by intent** for best results

---

## 10. Transition to Module 4

> "Now that we know how to organize and retrieve knowledge, Module 4 puts it all together into a complete RAG pipeline with LangChain."

---

## References

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LangChain Integration with LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/langchain/)
- [Pinecone: Vector Database Concepts](https://www.pinecone.io/learn/)
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormack2009reciprocal.pdf)
