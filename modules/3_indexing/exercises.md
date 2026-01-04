# Indexing Strategies Exercises

## Exercise 1: Compare Index Query Performance (Easy)

**Task**: Measure query latency for different LlamaIndex index types.

**Steps**:
```python
import time
from llama_index.core import VectorStoreIndex, SummaryIndex, TreeIndex

def benchmark_index(index, name, test_queries):
    """Benchmark query performance for an index"""
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    query_times = []
    for query in test_queries:
        start = time.time()
        response = query_engine.query(query)
        query_times.append((time.time() - start) * 1000)
    
    avg_query_time = sum(query_times) / len(query_times)
    
    print(f"\n{name}:")
    print(f"  Avg query time: {avg_query_time:.2f}ms")
    print(f"  Queries tested: {len(test_queries)}")

# Test different indexes
test_queries = [
    "How to fix authentication issues?",
    "Database timeout problems",
    "Mobile app crashes"
]

# Build indexes
vector_index = VectorStoreIndex.from_documents(documents)
summary_index = SummaryIndex.from_documents(documents)
tree_index = TreeIndex.from_documents(documents)

# Run benchmarks
benchmark_index(vector_index, "Vector Index", test_queries)
benchmark_index(summary_index, "Summary Index", test_queries)
benchmark_index(tree_index, "Tree Index", test_queries)
```

**Questions**:
- Which index type is fastest for querying?
- Which provides the most accurate results?
- What's the trade-off between speed and quality?

---

## Exercise 2: Customize Retrieval Parameters (Medium)

**Task**: Experiment with different retrieval parameters for each index type.

**For Vector Index**:
```python
from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex.from_documents(documents)

# Test different top_k values
for k in [1, 3, 5, 10]:
    query_engine = vector_index.as_query_engine(similarity_top_k=k)
    response = query_engine.query("How to fix login issues?")
    print(f"\nTop-{k} Results:")
    print(response)
```

**For Summary Index**:
```python
from llama_index.core import SummaryIndex
from llama_index.core.response_synthesizers import ResponseMode

summary_index = SummaryIndex.from_documents(documents)

# Test different response modes
modes = [ResponseMode.REFINE, ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE]

for mode in modes:
    query_engine = summary_index.as_query_engine(response_mode=mode)
    response = query_engine.query("Summarize authentication issues")
    print(f"\nMode: {mode}")
    print(response)
```

**Questions**:
- How does `similarity_top_k` affect answer quality?
- Which response mode works best for summaries?
- What's the latency trade-off?

---

## Exercise 3: Implement Metadata Filtering (Medium)

**Task**: Filter documents by category before retrieval using LlamaIndex metadata filters.

**Implementation**:
```python
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# Build index with metadata
vector_index = VectorStoreIndex.from_documents(documents)

# Create query engine with metadata filter
def filtered_query(query, category=None, priority=None):
    filters = []
    
    if category:
        filters.append(ExactMatchFilter(key="category", value=category))
    
    if priority:
        filters.append(ExactMatchFilter(key="priority", value=priority))
    
    if filters:
        query_engine = vector_index.as_query_engine(
            similarity_top_k=3,
            filters=MetadataFilters(filters=filters)
        )
    else:
        query_engine = vector_index.as_query_engine(similarity_top_k=3)
    
    return query_engine.query(query)

# Test filtered queries
print("All authentication issues:")
response = filtered_query("login failed", category="Authentication")
print(response)

print("\nCritical priority only:")
response = filtered_query("system down", priority="Critical")
print(response)
```

**Test cases**:
- Filter by category only
- Filter by priority only  
- Combine multiple filters
- Compare with unfiltered results

**Questions**:
- How does filtering affect relevance?
- Does it reduce irrelevant results?
- What happens when filters are too restrictive?

---

## Exercise 4: Build a Custom Hybrid Retriever (Medium)

**Task**: Implement your own hybrid retrieval combining vector and keyword search.

**Implementation**:
```python
from llama_index.core import VectorStoreIndex, KeywordTableIndex
import numpy as np

class CustomHybridRetriever:
    def __init__(self, documents):
        # Build both indexes
        self.vector_index = VectorStoreIndex.from_documents(documents)
        self.keyword_index = KeywordTableIndex.from_documents(documents)
        
    def retrieve(self, query, top_k=5, vector_weight=0.7):
        """
        Hybrid retrieval with custom weighting
        
        Args:
            query: Search query
            top_k: Number of results
            vector_weight: Weight for vector search (0-1)
        """
        keyword_weight = 1 - vector_weight
        
        # Get retrievers
        vector_retriever = self.vector_index.as_retriever(similarity_top_k=top_k*2)
        keyword_retriever = self.keyword_index.as_retriever(similarity_top_k=top_k*2)
        
        # Retrieve from both
        vector_nodes = vector_retriever.retrieve(query)
        keyword_nodes = keyword_retriever.retrieve(query)
        
        # Combine scores using RRF (Reciprocal Rank Fusion)
        doc_scores = {}
        
        for rank, node in enumerate(vector_nodes, 1):
            doc_id = node.node.metadata.get('ticket_id', node.node_id)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + (vector_weight / rank)
        
        for rank, node in enumerate(keyword_nodes, 1):
            doc_id = node.node.metadata.get('ticket_id', node.node_id)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + (keyword_weight / rank)
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_docs[:top_k]

# Test hybrid retriever
hybrid = CustomHybridRetriever(documents)

# Test with different weights
for weight in [0.3, 0.5, 0.7, 1.0]:
    print(f"\nVector weight: {weight}")
    results = hybrid.retrieve("authentication failure", vector_weight=weight)
    print(f"Top 3: {[doc_id for doc_id, score in results[:3]]}")
```

**Questions**:
- How do different weights affect results?
- When should you favor vector vs keyword?
- Can you improve the RRF formula?---

## Exercise 5: Index Persistence and Loading (Easy)

**Task**: Practice saving and loading LlamaIndex indexes.

**Steps**:
```python
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

# Build an index
vector_index = VectorStoreIndex.from_documents(documents)

# Save to disk
vector_index.storage_context.persist(persist_dir="./storage")
print("✓ Index saved to ./storage")

# Load from disk
storage_context = StorageContext.from_defaults(persist_dir="./storage")
loaded_index = load_index_from_storage(storage_context)
print("✓ Index loaded from disk")

# Verify it works
query_engine = loaded_index.as_query_engine()
response = query_engine.query("How to fix authentication issues?")
print(f"\nQuery result: {response}")
```

**Why this matters**: 
- Building indexes is expensive
- Production systems should load pre-built indexes
- Much faster startup time
- Saves API costs (no re-embedding)

**Challenge**: Modify the demo to always check if an index exists before building a new one.

---

## Exercise 6: Compare Index Types for Different Queries (Medium)

**Task**: Test which index type works best for different query patterns.

**Query Types**:
```python
query_types = {
    'specific': [
        "What is ticket TICK-001 about?",
        "Show me TICK-015 details"
    ],
    'semantic': [
        "How to fix login problems?",
        "Database connection issues"
    ],
    'summary': [
        "What are the most common authentication problems?",
        "Summarize all database issues"
    ]
}

# Test all indexes
indexes = {
    'vector': VectorStoreIndex.from_documents(documents),
    'summary': SummaryIndex.from_documents(documents),
    'tree': TreeIndex.from_documents(documents),
    'keyword': KeywordTableIndex.from_documents(documents)
}

# Compare results
for query_type, queries in query_types.items():
    print(f"\n{'='*60}")
    print(f"Query Type: {query_type.upper()}")
    print('='*60)
    
    for query in queries:
        print(f"\nQuery: {query}")
        for name, index in indexes.items():
            engine = index.as_query_engine(similarity_top_k=3)
            response = engine.query(query)
            print(f"  {name}: {str(response)[:100]}...")
```

**Analysis**:
- Which index type is best for specific ticket lookups?
- Which works best for semantic similarity?
- Which is best for high-level summaries?
- Can you create decision rules for index selection?

---

## Solutions

Solutions provided after the workshop. Try to implement these yourself first!

These exercises will help you understand:
- When to use each index type
- How to optimize retrieval parameters
- Building hybrid retrieval systems
- Persisting and loading indexes efficiently

---

## Next Steps

Ready for **Module 4: RAG Pipeline**? We'll combine indexing with LLM generation to build a complete question-answering system!

---

**Need help?** Ask the instructor or check the demo code for reference implementations.
