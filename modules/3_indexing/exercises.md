# Indexing Strategies Exercises

> ✅ **Exercise style for this workshop:** keep each solution to a **small edit** (usually 3–15 lines) in existing files.

## Exercise 1: Change the Query (Easy)

**Task**: Modify the demo to search for a different type of issue.

**In the demo code, find this line** (around line 79):
```python
query = "How do I fix authentication issues after password reset?"
```

**Change it to**:
```python
query = "Database connection is timing out"
```

**Run the demo and observe**:
- How do the results differ between Vector, Summary, Tree, and Keyword indexes?
- Which index type gives the most relevant answer?

**Try these queries too**:
- `"Email notifications not being delivered"`
- `"Mobile app crashes on startup"`
- `"Payment processing fails for international cards"`

---

## Exercise 2: Adjust the Number of Results (Easy)

**Task**: Get more search results from the Vector Index.

**In the demo code, find this line** (around line 92):
```python
vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)
```

**Change it to**:
```python
vector_query_engine = vector_index.as_query_engine(similarity_top_k=5)
```

**Run and observe**: Does getting more source documents improve the answer quality?

---

## Exercise 3: Change the Tree Index Branch Factor (Easy)

**Task**: Modify how many branches the Tree Index explores.

**In the demo code, find this line** (around line 158):
```python
tree_query_engine = tree_index.as_query_engine(child_branch_factor=2)
```

**Try different values**:
```python
# Explore only 1 branch (more focused, might miss relevant info)
tree_query_engine = tree_index.as_query_engine(child_branch_factor=1)

# Explore 3 branches (broader search, slower)
tree_query_engine = tree_index.as_query_engine(child_branch_factor=3)
```

**Run and observe**: 
- How does `child_branch_factor=1` affect the answer?
- Is `child_branch_factor=3` noticeably slower?

---

## Exercise 4: Test a Keyword-Specific Query (Easy)

**Task**: See how Keyword Index handles exact term matching.

**Add this code after the Keyword Index section** (around line 195):
```python
# Test keyword-specific query
keyword_query = "TICK-001"
print(f"\nKeyword-specific query: '{keyword_query}'")
keyword_response = keyword_query_engine.query(keyword_query)
print(f"Result: {keyword_response.response}")
```

**Run and observe**: Does the Keyword Index find the exact ticket ID?

---

## Exercise 5: Compare Index Types Side-by-Side (Medium)

**Task**: Run the same query through two index types and compare with minimal edits.

In existing code, add 1-2 test queries and compare `Vector` vs `Keyword` outputs only.
Reference snippet:
```python
import json
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SummaryIndex, TreeIndex, KeywordTableIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# Configure LlamaIndex
Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = OpenAI(model='gpt-4o-mini')

# Load data
with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = [
    Document(
        text=f"Title: {t['title']}\nDescription: {t['description']}\nResolution: {t['resolution']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category']}
    )
    for t in tickets
]

print("Building indexes...")
vector_idx = VectorStoreIndex.from_documents(documents)
keyword_idx = KeywordTableIndex.from_documents(documents)
print("✓ Indexes built\n")

# Compare on 3 queries
test_queries = [
    "authentication login problem",
    "database timeout error",
    "TICK-005"
]

for query in test_queries:
    print("=" * 60)
    print(f"Query: '{query}'")
    print("=" * 60)
    
    # Vector Index
    vec_response = vector_idx.as_query_engine(similarity_top_k=3).query(query)
    print(f"\nVector Index:")
    print(f"  {str(vec_response)[:150]}...")
    
    # Keyword Index
    kw_response = keyword_idx.as_query_engine().query(query)
    print(f"\nKeyword Index:")
    print(f"  {str(kw_response)[:150]}...")
    
    print()
```

**Answer**: Which index works best for "TICK-005" (exact match) vs "authentication login problem" (semantic)?

---

## Exercise 6: Save and Load an Index (Medium)

**Task**: Persist and reload index with a small patch.

In your existing vector-index block, add:
1. one `persist(...)` call
2. one `load_index_from_storage(...)` block
3. one verification query

Reference snippet:
```python
import json
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = OpenAI(model='gpt-4o-mini')

# Load data
with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = [
    Document(
        text=f"Title: {t['title']}\nDescription: {t['description']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category']}
    )
    for t in tickets
]

# Step 1: Build and save
print("Building index...")
vector_index = VectorStoreIndex.from_documents(documents)
vector_index.storage_context.persist(persist_dir="./my_saved_index")
print("✓ Saved to ./my_saved_index")

# Step 2: Load from disk
print("\nLoading index...")
storage_context = StorageContext.from_defaults(persist_dir="./my_saved_index")
loaded_index = load_index_from_storage(storage_context)
print("✓ Loaded from disk")

# Step 3: Test it works
query = "login problem"
response = loaded_index.as_query_engine().query(query)
print(f"\nQuery: '{query}'")
print(f"Result: {response}")
```

**Why this matters**: Building indexes is expensive (API calls). Persisting saves time and money!

---

## Exercise 7: Add Metadata Filtering (Medium)

**Task**: Filter search results by category with one filter object.

Add this to your current query flow:

```python
filters = MetadataFilters(filters=[
    ExactMatchFilter(key="category", value="Authentication")
])
```

Reference snippet:
```python
import json
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = OpenAI(model='gpt-4o-mini')

# Load data
with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = [
    Document(
        text=f"Title: {t['title']}\nDescription: {t['description']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category'], 'priority': t['priority']}
    )
    for t in tickets
]

# Build index
vector_index = VectorStoreIndex.from_documents(documents)

# Query WITHOUT filter
print("Without filter:")
response = vector_index.as_query_engine(similarity_top_k=3).query("system problem")
print(f"  {response}\n")

# Query WITH category filter
print("With 'Authentication' filter:")
filters = MetadataFilters(filters=[
    ExactMatchFilter(key="category", value="Authentication")
])
filtered_engine = vector_index.as_query_engine(similarity_top_k=3, filters=filters)
filtered_response = filtered_engine.query("system problem")
print(f"  {filtered_response}")
```

**Try changing the filter**:
- `value="Database"`
- `value="Performance"`

---

## Exercise 8: Benchmark Index Build Time (Medium)

**Task**: Measure build time with a tiny timing patch.

Add `time.time()` around index construction for at least two indexes.
Reference snippet:
```python
import json
import time
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SummaryIndex, KeywordTableIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = OpenAI(model='gpt-4o-mini')

# Load data
with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = [
    Document(
        text=f"Title: {t['title']}\nDescription: {t['description']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category']}
    )
    for t in tickets
]

print(f"Building indexes for {len(documents)} documents...\n")

# Vector Index
start = time.time()
vector_index = VectorStoreIndex.from_documents(documents)
vector_time = time.time() - start
print(f"Vector Index: {vector_time:.2f}s")

# Keyword Index
start = time.time()
keyword_index = KeywordTableIndex.from_documents(documents)
keyword_time = time.time() - start
print(f"Keyword Index: {keyword_time:.2f}s")

# Summary Index (note: doesn't pre-build, so fast to create)
start = time.time()
summary_index = SummaryIndex.from_documents(documents)
summary_time = time.time() - start
print(f"Summary Index: {summary_time:.2f}s")

print(f"\n→ Vector Index takes longer because it generates embeddings for all documents")
print(f"→ Keyword Index uses LLM to extract keywords from each document")
print(f"→ Summary Index is just storing documents (work happens at query time)")
```

---

## Bonus Exercise: Simple Hybrid Search (Challenge)

**Task**: Combine Vector and Keyword search results.

Small-edit option: combine only top-2 results from each retriever and deduplicate ticket IDs.

**Copy and run this code**:
```python
import json
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, KeywordTableIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = OpenAI(model='gpt-4o-mini')

# Load data
with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = [
    Document(
        text=f"Title: {t['title']}\nDescription: {t['description']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category']}
    )
    for t in tickets
]

# Build both indexes
vector_index = VectorStoreIndex.from_documents(documents)
keyword_index = KeywordTableIndex.from_documents(documents)

query = "authentication timeout error"

# Get retrievers
vector_retriever = vector_index.as_retriever(similarity_top_k=5)
keyword_retriever = keyword_index.as_retriever()

# Retrieve from both
print(f"Query: '{query}'\n")

print("Vector Results:")
vector_nodes = vector_retriever.retrieve(query)
for i, node in enumerate(vector_nodes[:3], 1):
    print(f"  {i}. {node.node.metadata.get('ticket_id', 'N/A')}")

print("\nKeyword Results:")
keyword_nodes = keyword_retriever.retrieve(query)
for i, node in enumerate(keyword_nodes[:3], 1):
    print(f"  {i}. {node.node.metadata.get('ticket_id', 'N/A')}")

# Simple hybrid: combine and deduplicate
seen = set()
hybrid_results = []
for node in vector_nodes + keyword_nodes:
    ticket_id = node.node.metadata.get('ticket_id')
    if ticket_id and ticket_id not in seen:
        seen.add(ticket_id)
        hybrid_results.append(ticket_id)

print(f"\nHybrid Results (combined): {hybrid_results[:5]}")
```

---

## Quick Reference

### Index Types
```python
from llama_index.core import VectorStoreIndex, SummaryIndex, TreeIndex, KeywordTableIndex

# Vector: Semantic similarity search
vector_index = VectorStoreIndex.from_documents(documents)

# Summary: Reads all docs, good for high-level queries (slow for large datasets)
summary_index = SummaryIndex.from_documents(documents)

# Tree: Hierarchical traversal, good for large collections
tree_index = TreeIndex.from_documents(documents)

# Keyword: Exact term matching, no embeddings needed
keyword_index = KeywordTableIndex.from_documents(documents)
```

### Query Engines
```python
# Basic query
engine = index.as_query_engine()
response = engine.query("your question")

# With parameters
engine = index.as_query_engine(similarity_top_k=5)

# Tree Index with branch factor
engine = tree_index.as_query_engine(child_branch_factor=2)
```

### Persistence
```python
# Save
index.storage_context.persist(persist_dir="./storage")

# Load
from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
loaded_index = load_index_from_storage(storage_context)
```

---

## Next Steps

Ready for **Module 4: RAG Pipeline**? We'll combine indexing with LLM generation to build a complete question-answering system!

---

**Questions?** Ask the instructor or refer back to the demo code!
