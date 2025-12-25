# Hour 2 Exercises: Chunking & Vector Stores

## Exercise 1: Experiment with Chunk Sizes (Easy)

**Task**: Test different chunk sizes and observe how it affects retrieval.

**Steps**:
1. Create vector stores with these chunk sizes: 100, 200, 500, 1000
2. Search for "memory leak in background process"
3. Compare the relevance of top-3 results

```python
chunk_sizes = [100, 200, 500, 1000]
for size in chunk_sizes:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=size//10
    )
    chunks = splitter.split_documents(documents)
    # Build store and search...
```

**Questions**:
- Which chunk size gives the most relevant results?
- What happens when chunks are too small? Too large?
- How does chunk count vary with chunk size?

---

## Exercise 2: Metadata Filtering (Medium)

**Task**: Build a filtered search function that combines similarity search with metadata constraints.

**Requirements**:
```python
def filtered_search(vector_store, query, category=None, priority=None, k=3):
    """
    Search with optional category and priority filters
    
    Args:
        vector_store: Chroma vector store
        query: Search query string
        category: Filter by category (optional)
        priority: Filter by priority (optional)
        k: Number of results
    
    Returns:
        List of matching documents
    """
    # TODO: Build filter dict
    # TODO: Perform filtered search
    # TODO: Return results
    pass
```

**Test cases**:
```python
# All authentication tickets
filtered_search(store, "login failed", category="Authentication")

# High priority only
filtered_search(store, "system down", priority="Critical")

# Combine both filters
filtered_search(store, "database issue", category="Database", priority="High")
```

---

## Exercise 3: Save and Load Vector Stores (Medium)

**Task**: Implement functions to persist and load vector stores.

**FAISS Example**:
```python
def save_faiss_store(store, path="./faiss_index"):
    """Save FAISS store to disk"""
    store.save_local(path)
    print(f"Saved to {path}")

def load_faiss_store(path="./faiss_index", embeddings=None):
    """Load FAISS store from disk"""
    store = LangChainFAISS.load_local(
        path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"Loaded from {path}")
    return store
```

**Steps**:
1. Build a vector store
2. Save it to disk
3. Load it in a new Python session
4. Verify search works

**Why this matters**: In production, you don't want to rebuild embeddings every time!

---

## Exercise 4: Compare FAISS vs Chroma Performance (Medium)

**Task**: Benchmark search performance for both vector stores.

**Metrics to measure**:
- Index build time
- Query latency
- Memory usage
- Disk storage size

```python
import time
import os

def benchmark_store(store_builder, name, queries):
    # Build store
    start = time.time()
    store = store_builder()
    build_time = time.time() - start
    
    # Query latency
    query_times = []
    for query in queries:
        start = time.time()
        results = store.similarity_search(query, k=5)
        query_times.append(time.time() - start)
    
    avg_query_time = sum(query_times) / len(query_times)
    
    print(f"\n{name}:")
    print(f"  Build time: {build_time:.3f}s")
    print(f"  Avg query time: {avg_query_time*1000:.2f}ms")
```

**Test queries**:
```python
test_queries = [
    "authentication failed",
    "database timeout",
    "payment processing error",
    "mobile app crash",
    "email delivery failed"
]
```

---

## Exercise 5: Implement Hybrid Search (Hard)

**Task**: Combine keyword search (BM25) with semantic search for better results.

**Why**: Semantic search finds similar meanings, but sometimes exact keyword matches are important!

**Approach**:
```python
from rank_bm25 import BM25Okapi
import numpy as np

def hybrid_search(query, documents, vector_store, alpha=0.5, k=5):
    """
    Hybrid search combining BM25 (keyword) and semantic search
    
    Args:
        query: Search query
        documents: List of document texts
        vector_store: Vector store for semantic search
        alpha: Weight for semantic search (1-alpha for BM25)
        k: Number of results
    
    Returns:
        Top-k documents ranked by hybrid score
    """
    # BM25 keyword search
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.lower().split())
    
    # Semantic search
    semantic_results = vector_store.similarity_search_with_score(query, k=len(documents))
    semantic_scores = {doc.page_content: score for doc, score in semantic_results}
    
    # Normalize and combine scores
    # TODO: Normalize BM25 scores (0-1)
    # TODO: Normalize semantic scores (0-1)
    # TODO: Compute hybrid score = alpha * semantic + (1-alpha) * bm25
    # TODO: Return top-k
    
    pass
```

**Test**:
- Try alpha=0.5 (equal weight)
- Try alpha=0.8 (favor semantic)
- Try alpha=0.2 (favor keywords)

Compare results for queries like:
- "TICK-005" (exact keyword)
- "memory problems in worker" (semantic)

---

## Exercise 6: Build a Simple Vector Store CLI (Challenge)

**Task**: Create an interactive tool to build, save, and search vector stores.

**Features**:
```
SupportDesk Vector Store Manager
================================
1. Build new vector store
2. Load existing vector store
3. Search tickets
4. Filter by metadata
5. Show statistics
6. Exit

Choose option: _
```

**Implementation hints**:
```python
class VectorStoreManager:
    def __init__(self):
        self.store = None
        self.embeddings = HuggingFaceEmbeddings(...)
    
    def build_store(self, documents, chunk_size=None):
        """Build and save vector store"""
        pass
    
    def load_store(self, path):
        """Load existing vector store"""
        pass
    
    def search(self, query, filters=None, k=5):
        """Search with optional filters"""
        pass
    
    def get_stats(self):
        """Show store statistics"""
        pass
```

---

## Exercise 7: Overlap Analysis (Advanced)

**Task**: Visualize how chunk overlap affects information preservation.

**Steps**:
1. Create chunks with different overlap amounts: 0, 10, 50, 100
2. For each strategy, find chunks that reference a specific ticket
3. Check if critical information spans chunk boundaries

```python
def analyze_overlap(text, chunk_size=200, overlaps=[0, 20, 50, 100]):
    """Analyze how overlap affects chunking"""
    for overlap in overlaps:
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        chunks = splitter.split_text(text)
        
        print(f"\nOverlap: {overlap}")
        print(f"  Total chunks: {len(chunks)}")
        
        # Check for sentence splitting
        broken_sentences = 0
        for chunk in chunks:
            if not chunk.strip().endswith('.'):
                broken_sentences += 1
        
        print(f"  Broken sentences: {broken_sentences}")
```

---

## Bonus: Production Optimizations

### 1. Batch Embedding Generation
```python
# Instead of encoding one by one
for doc in documents:
    embedding = model.encode(doc)
    
# Encode all at once (much faster!)
all_texts = [doc.page_content for doc in documents]
embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=True)
```

### 2. Use FAISS GPU (if available)
```python
import faiss

# CPU version
index = faiss.IndexFlatL2(dimension)

# GPU version (10x faster!)
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, dimension)
```

### 3. Approximate Search for Scale
```python
# Exact search (slow for millions of vectors)
index = faiss.IndexFlatL2(dimension)

# Approximate search with IVF (much faster!)
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
index.train(embeddings)
index.add(embeddings)
```

---

## Solutions

Solutions provided after the workshop. Try to implement these yourself first!

---

## Next Steps

Ready for **Hour 3: RAG Pipeline**? We'll combine everything you've learned to build a complete question-answering system!

---

**Need help?** Ask the instructor or check the demo code for reference implementations.
