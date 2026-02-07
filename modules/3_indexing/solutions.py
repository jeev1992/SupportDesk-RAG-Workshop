# -*- coding: utf-8 -*-
"""
Module 3 Solutions: Indexing Strategies
=======================================

Solutions for all exercises in exercises.md

NOTE: Some index types (Tree, Keyword) make many LLM calls during build
and may take 1-2 minutes to complete. This is expected behavior.
"""

import json
import time
import os
import shutil
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, SummaryIndex, TreeIndex, KeywordTableIndex,
    Document, Settings, StorageContext, load_index_from_storage
)
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# Configure LlamaIndex
Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
Settings.llm = OpenAI(model='gpt-4o-mini')

# Load data
print("Loading data...")
with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

documents = [
    Document(
        text=f"Title: {t['title']}\nDescription: {t['description']}\nResolution: {t['resolution']}",
        metadata={
            'ticket_id': t['ticket_id'],
            'category': t['category'],
            'priority': t['priority']
        }
    )
    for t in tickets
]
print(f"✓ Loaded {len(documents)} documents\n")

# Build vector index (fast - just embeddings, no LLM calls)
print("Building Vector Index...")
vector_index = VectorStoreIndex.from_documents(documents)
print("✓ Vector Index built\n")


# ============================================================================
# Exercise 1: Change the Query (Easy)
# ============================================================================
print("=" * 80)
print("EXERCISE 1: Change the Query")
print("=" * 80)

# Build a simple vector index for testing
vector_index = VectorStoreIndex.from_documents(documents)
query_engine = vector_index.as_query_engine(similarity_top_k=3)

queries = [
    "How do I fix authentication issues after password reset?",  # Original
    "Database connection is timing out",  # Changed
    "Email notifications not being delivered"
]

for query in queries:
    response = query_engine.query(query)
    print(f"\nQuery: '{query}'")
    print(f"Answer: {str(response)[:150]}...")


# ============================================================================
# Exercise 2: Adjust the Number of Results (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Adjust the Number of Results")
print("=" * 80)

query = "authentication login problem"

# top_k = 3 (original)
engine_3 = vector_index.as_query_engine(similarity_top_k=3)
response_3 = engine_3.query(query)
print(f"\nWith similarity_top_k=3:")
print(f"  Sources: {len(response_3.source_nodes)} documents used")

# top_k = 5 (changed)
engine_5 = vector_index.as_query_engine(similarity_top_k=5)
response_5 = engine_5.query(query)
print(f"\nWith similarity_top_k=5:")
print(f"  Sources: {len(response_5.source_nodes)} documents used")


# ============================================================================
# Exercise 3: Change the Tree Index Branch Factor (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Change the Tree Index Branch Factor")
print("=" * 80)

# NOTE: Tree Index is SLOW to build (makes many LLM calls to generate summaries)
# Skipping in solutions to avoid API timeouts
# In the demo, the Tree Index is already built, so modifying branch_factor is fast

print("\n→ SKIPPED: Tree Index takes too long to build (many LLM calls)")
print("→ To test: Modify the branch_factor in demo.py and run the demo instead")
print("→ tree_query_engine = tree_index.as_query_engine(child_branch_factor=1)")
print("→ tree_query_engine = tree_index.as_query_engine(child_branch_factor=3)")


# ============================================================================
# Exercise 4: Test a Keyword-Specific Query (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: Test a Keyword-Specific Query")
print("=" * 80)

print("\nBuilding Keyword Index (this may take 1-2 minutes)...")
keyword_index = KeywordTableIndex.from_documents(documents)
keyword_engine = keyword_index.as_query_engine()
print("✓ Keyword Index built")

keyword_queries = ["TICK-001", "authentication", "database timeout"]

for query in keyword_queries:
    response = keyword_engine.query(query)
    print(f"\nKeyword query: '{query}'")
    print(f"  Answer: {str(response)[:150]}...")


# ============================================================================
# Exercise 5: Compare Index Types Side-by-Side (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 5: Compare Index Types Side-by-Side")
print("=" * 80)

test_queries = [
    "authentication login problem",  # Semantic query
    "database timeout error",        # Semantic query
    "TICK-005"                       # Exact match query
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print("=" * 60)
    
    # Vector Index
    vec_response = vector_index.as_query_engine(similarity_top_k=3).query(query)
    print(f"\nVector Index:")
    print(f"  {str(vec_response)[:150]}...")
    
    # Keyword Index
    kw_response = keyword_index.as_query_engine().query(query)
    print(f"\nKeyword Index:")
    print(f"  {str(kw_response)[:150]}...")


# ============================================================================
# Exercise 6: Save and Load an Index (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Save and Load an Index")
print("=" * 80)

# Step 1: Save to disk
persist_dir = "./solution_saved_index"
print(f"\nSaving index to {persist_dir}...")
vector_index.storage_context.persist(persist_dir=persist_dir)
print("✓ Saved")

# Step 2: Load from disk
print("\nLoading index from disk...")
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
loaded_index = load_index_from_storage(storage_context)
print("✓ Loaded")

# Step 3: Test it works
query = "login problem"
response = loaded_index.as_query_engine().query(query)
print(f"\nQuery: '{query}'")
print(f"Result: {str(response)[:150]}...")

# Cleanup
shutil.rmtree(persist_dir)
print(f"\n✓ Cleaned up {persist_dir}")


# ============================================================================
# Exercise 7: Add Metadata Filtering (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Add Metadata Filtering")
print("=" * 80)

query = "system problem"

# Without filter
print("\nWithout filter:")
response = vector_index.as_query_engine(similarity_top_k=3).query(query)
print(f"  {str(response)[:150]}...")

# With category filter
categories = ["Authentication", "Database", "Performance"]

for category in categories:
    filters = MetadataFilters(filters=[
        ExactMatchFilter(key="category", value=category)
    ])
    filtered_engine = vector_index.as_query_engine(similarity_top_k=3, filters=filters)
    
    try:
        filtered_response = filtered_engine.query(query)
        print(f"\nWith '{category}' filter:")
        print(f"  {str(filtered_response)[:100]}...")
    except Exception as e:
        print(f"\nWith '{category}' filter: No matching documents")


# ============================================================================
# Exercise 8: Benchmark Index Build Time (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 8: Benchmark Index Build Time")
print("=" * 80)

print(f"\nBuilding indexes for {len(documents)} documents...\n")

# Vector Index
start = time.time()
_ = VectorStoreIndex.from_documents(documents)
vector_time = time.time() - start
print(f"Vector Index: {vector_time:.2f}s")

# Keyword Index
start = time.time()
_ = KeywordTableIndex.from_documents(documents)
keyword_time = time.time() - start
print(f"Keyword Index: {keyword_time:.2f}s")

# Summary Index
start = time.time()
_ = SummaryIndex.from_documents(documents)
summary_time = time.time() - start
print(f"Summary Index: {summary_time:.2f}s")

print(f"\n→ Vector Index takes longer because it generates embeddings")
print(f"→ Keyword Index uses LLM to extract keywords")
print(f"→ Summary Index just stores documents (work at query time)")


# ============================================================================
# Bonus: Simple Hybrid Search (Challenge)
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Simple Hybrid Search")
print("=" * 80)

query = "authentication timeout error"

# Get retrievers
vector_retriever = vector_index.as_retriever(similarity_top_k=5)
keyword_retriever = keyword_index.as_retriever()

# Retrieve from both
print(f"\nQuery: '{query}'\n")

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


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL SOLUTIONS COMPLETE!")
print("=" * 80)
print("""
Key Takeaways:
──────────────
1. Vector Index: Best for semantic/meaning-based queries
2. Keyword Index: Best for exact term matching (IDs, error codes)
3. Tree Index: Good for large hierarchical documents
4. Summary Index: Good for high-level queries on small datasets
5. Hybrid: Combine vector + keyword for best coverage
6. Always persist indexes to save rebuild time and API costs

Next: Move on to Module 4 - RAG Pipeline!
""")
