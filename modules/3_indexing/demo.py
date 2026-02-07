# -*- coding: utf-8 -*-
"""
Indexing Strategies Demo
========================

This demo teaches:
1. Vector Index (Flat) - Simple vector store with all chunks
2. Summary Index - Index document summaries for better retrieval
3. Tree Index - Hierarchical retrieval from summaries to details
4. Keyword Table Index - Traditional keyword-based retrieval
5. Hybrid Retrieval - Combine multiple retrieval strategies
"""

import json
import os
import httpx
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SummaryIndex, TreeIndex, KeywordTableIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()

# Set longer timeout for httpx (used by OpenAI client)
os.environ["HTTPX_TIMEOUT"] = "300"  # 5 minutes

# Configure LlamaIndex settings with increased timeout and retries
Settings.embed_model = OpenAIEmbedding(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=120,  # 120 second timeout for embedding calls
    max_retries=5  # Retry up to 5 times on failure
)
Settings.llm = OpenAI(
    model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=300,  # 5 minute timeout for LLM calls (Tree/Keyword indexes need more time)
    max_retries=5  # Retry up to 5 times on failure
)

print("="*80)
print("INDEXING STRATEGIES FOR RAG")
print("="*80)
print("\nThis demo compares 5 different indexing approaches:")
print("1. Vector Index - Semantic similarity search")
print("2. Summary Index - Search through document summaries")
print("3. Tree Index - Hierarchical retrieval (summaries → details)")
print("4. Keyword Table Index - Traditional keyword matching")
print("5. Hybrid Retrieval - Combine multiple strategies")

# ============================================================================
# Load Data
# ============================================================================
print("\n" + "="*80)
print("Loading Support Tickets")
print("="*80)

with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

# Convert to LlamaIndex Documents
documents = []
for ticket in tickets:
    # Combine all fields into content
    content = f"""Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}"""
    
    doc = Document(
        text=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority'],
            'title': ticket['title']
        }
    )
    documents.append(doc)

print(f"✓ Loaded {len(documents)} support tickets")

# Test query
query = "How do I fix authentication issues after password reset?"
print(f"\nTest Query: '{query}'")

# ============================================================================
# PART 1: Vector Index (Flat Index)
# ============================================================================
print("\n" + "="*80)
print("PART 1: Vector Index (Flat Index)")
print("="*80)

print("\nVector indexing embeds all chunks and retrieves by semantic similarity.")
print("✓ Simple and effective for most use cases")
print("✓ Fast similarity search with vector databases")
print("✗ No hierarchical structure")
print("✗ May return fragmented chunks\n")

vector_index = VectorStoreIndex.from_documents(documents)
vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)

print("✓ Created vector index")
print(f"\nQuery: '{query}'")
vector_response = vector_query_engine.query(query)

print("\nVector Index Results:")
print(f"Answer: {vector_response.response}\n")
print("Source Documents:")
for i, node in enumerate(vector_response.source_nodes, 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   Score: {node.score:.4f}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# PART 2: Summary Index
# ============================================================================
#
# How Summary Index Works:
# ------------------------
# Unlike Vector Index which uses embeddings for similarity search,
# Summary Index stores full documents and uses the LLM to evaluate relevance.
#
# Storage:
#   [Doc 1: Full ticket text] → No chunking, no embeddings
#   [Doc 2: Full ticket text]
#   [Doc 3: Full ticket text]
#   ...all 50 tickets stored as-is
#
# Query Process (Linear Scan):
#   1. LLM reads EACH document
#   2. LLM decides: "Is this relevant to the query?"
#   3. Relevant documents collected
#   4. LLM synthesizes answer from all relevant docs
#
# Why it's slow:
#   - O(n) complexity - must check every document
#   - Each check = LLM call or LLM attention
#   - 10 docs = 2 sec, 100 docs = 20 sec, 1000 docs = 200 sec!
#
# When to use:
#   - Small collections (<50 documents)
#   - Need full document context (not fragments)
#   - High-level summarization queries
#
print("\n" + "="*80)
print("PART 2: Summary Index")
print("="*80)

print("\nSummary indexing searches through document summaries/titles.")
print("✓ Good for high-level queries")
print("✓ Returns full documents, not fragments")
print("✗ Slower for large datasets (linear scan)")
print("✗ No vector similarity search\n")

# Create summary index:
# - Stores full documents (no chunking)
# - No embeddings generated
# - Documents indexed as-is
summary_index = SummaryIndex.from_documents(documents)

# Query engine with tree_summarize:
# - response_mode="tree_summarize" synthesizes answers hierarchically
# - First summarizes groups of docs, then summarizes summaries
# - Good for getting comprehensive answers from multiple sources
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")

print("✓ Created summary index")
print(f"\nQuery: '{query}'")

# Query execution:
# 1. LLM examines each document for relevance
# 2. Collects all relevant documents
# 3. Uses tree_summarize to synthesize final answer
summary_response = summary_query_engine.query(query)

print("\nSummary Index Results:")
print(f"Answer: {summary_response.response}\n")
print("Source Documents:")
for i, node in enumerate(summary_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# PART 3: Tree Index (Hierarchical)
# ============================================================================
#
# Visual of tree traversal for query "authentication issues after password reset":
#
#                     [Root: All 50 tickets summary]
#                               │
#             ┌─────────────────┼─────────────────┐
#             ▼                 ▼                 ▼
#     [Auth Issues]      [Performance]      [Billing]
#          │                                     
#     ┌────┼────┐                               
#     ▼         ▼                               
# [Ticket 1] [Ticket 5]  ← These become source_nodes
#
# The engine:
# 1. Scores root children → Auth branch scores highest
# 2. Drills into Auth branch
# 3. Finds matching leaves (password reset tickets)
# 4. Returns those as source_nodes
#
print("\n" + "="*80)
print("PART 3: Tree Index (Hierarchical Retrieval)")
print("="*80)

print("\n⚠️  NOTE: Tree Index makes many LLM calls during build.")
print("    Using first 10 documents to reduce API calls.")
print("    This may take 1-2 minutes. Please wait...\n")

print("Tree indexing builds a hierarchical structure from leaf to root.")
print("- Queries start at summary level, then drill down")
print("✓ Preserves document context and relationships")
print("✓ Efficient for large document collections")
print("✗ More complex to build and maintain\n")

# Use subset of documents to reduce LLM calls (Tree Index calls LLM for each node)
tree_documents = documents
print(f"Building Tree Index with {len(tree_documents)} documents...")

# Build tree structure from documents:
# - Leaf nodes: Each ticket becomes a leaf with full text
# - Branch nodes: LLM generates summaries grouping related tickets
# - Root node: Top-level summary of all content
tree_index = TreeIndex.from_documents(tree_documents)

# Create query engine with child_branch_factor=2:
# - At each level, explore the top 2 most relevant branches
# - Higher = broader search (more paths), Lower = focused search (single path)
tree_query_engine = tree_index.as_query_engine(child_branch_factor=2)

print("✓ Created tree index with hierarchical structure")
print(f"\nQuery: '{query}'")

# Query executes hierarchical retrieval:
# 1. Start at root summary
# 2. LLM scores each child branch for relevance
# 3. Explore top 2 branches (child_branch_factor=2)
# 4. Repeat until reaching leaf nodes
# 5. Collect relevant leaves, synthesize answer
tree_response = tree_query_engine.query(query)

print("\nTree Index Results:")
# tree_response.response = The synthesized answer from relevant leaves
print(f"Answer: {tree_response.response}\n")
print("Source Documents:")
# tree_response.source_nodes = The leaf nodes used to generate the answer
for i, node in enumerate(tree_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# PART 4: Keyword Table Index
# ============================================================================
#
# How keywords are extracted:
# ---------------------------
# By default, LlamaIndex uses an LLM call to extract keywords:
#   LLM prompt: "Extract keywords from the following text..."
#   LLM returns: "authentication, password, reset, login, SSO, error"
#
# What gets stored (inverted index):
#   Keyword → [Document IDs]
#   ─────────────────────────
#   "password"  → [ticket_1, ticket_5, ticket_12]
#   "login"     → [ticket_1, ticket_3, ticket_8]
#   "timeout"   → [ticket_7, ticket_15]
#   "billing"   → [ticket_20, ticket_25]
#
# At query time:
#   1. Extract keywords from query (also via LLM)
#   2. Look up matching documents in the keyword table
#   3. Return documents that share keywords with the query
#
# Key difference from vector search:
#   No semantic understanding—"authentication" won't match "login"
#   unless both keywords appear in the same document.
#
print("\n" + "="*80)
print("PART 4: Keyword Table Index")
print("="*80)

print("\n⚠️  NOTE: Keyword Index makes LLM calls to extract keywords.")
print("    Using first 10 documents to reduce API calls.")
print("    This may take 1-2 minutes. Please wait...\n")

print("Keyword indexing extracts keywords and uses exact/fuzzy matching.")
print("✓ No embeddings needed - works without vector DB")
print("✓ Good for keyword-specific queries")
print("✗ No semantic understanding")
print("✗ Misses synonyms and related concepts\n")

# Use subset of documents to reduce LLM calls
keyword_documents = documents
print(f"Building Keyword Index with {len(keyword_documents)} documents...")

# Build keyword index:
# - LLM extracts keywords from each document
# - Stores inverted index: keyword → [doc_ids]
keyword_index = KeywordTableIndex.from_documents(keyword_documents)
keyword_query_engine = keyword_index.as_query_engine()

print("✓ Created keyword table index")
print(f"\nQuery: '{query}'")

# Query process:
# 1. Extract keywords from query via LLM
# 2. Look up documents containing those keywords
# 3. Return matching documents
keyword_response = keyword_query_engine.query(query)

print("\nKeyword Index Results:")
print(f"Answer: {keyword_response.response}\n")
print("Source Documents:")
for i, node in enumerate(keyword_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# PART 5: Hybrid Retrieval (Custom)
# ============================================================================
#
# Why Hybrid Retrieval?
# ---------------------
# Vector search alone misses exact matches (ticket IDs, error codes)
# Keyword search alone misses semantic matches ("login" ≠ "authentication")
# Hybrid combines both → fewer missed results
#
# How it works:
#   Query: "authentication timeout error"
#
#   Vector Search:              Keyword Search:
#   [Doc A: 0.89] (login issue) [Doc C: 3 keywords match]
#   [Doc B: 0.85] (auth error)  [Doc A: 2 keywords match]
#   [Doc C: 0.75] (timeout)     [Doc E: 1 keyword match]
#
#   Fusion → Combine both, remove duplicates:
#   [Doc A] - found by both (high confidence)
#   [Doc B] - semantic match only
#   [Doc C] - found by both (high confidence)
#   [Doc E] - keyword match only
#
print("\n" + "="*80)
print("PART 5: Hybrid Retrieval")
print("="*80)

print("\nHybrid retrieval combines multiple indexes for better results.")
print("- Typically combines vector (semantic) + keyword (exact match)")
print("✓ Best of both worlds - semantic + exact matching")
print("✓ More robust to query variations")
print("✓ Higher overall accuracy")
print("✗ Slower (multiple searches)")
print("✗ Requires result fusion logic\n")

# Simple hybrid approach: Query both and combine results
print("✓ Using Vector + Keyword hybrid approach")
print(f"\nQuery: '{query}'")

# Step 1: Retrieve from vector index (semantic similarity)
# Finds documents that are semantically similar to the query
# Example: "auth issues" → finds "login problems", "SSO failures"
vector_nodes = vector_index.as_retriever(similarity_top_k=5).retrieve(query)

# Step 2: Retrieve from keyword index (exact term matching)
# Finds documents containing the exact query keywords
# Example: "authentication" → finds docs with that exact word
keyword_nodes = keyword_index.as_retriever().retrieve(query)

# Step 3: Simple fusion - combine and deduplicate
# More sophisticated approaches use Reciprocal Rank Fusion (RRF):
#   score(doc) = Σ 1/(k + rank_i) across all retrievers
# Here we use simple deduplication with vector results first (priority)
seen_ids = set()
hybrid_nodes = []

for node in vector_nodes + keyword_nodes:
    # Track by ticket_id to avoid duplicates
    node_id = node.metadata.get('ticket_id', node.node_id)
    if node_id not in seen_ids:
        seen_ids.add(node_id)
        hybrid_nodes.append(node)
# Result: Union of both search methods, duplicates removed
# Documents found by BOTH methods are likely most relevant

print("\nHybrid Retrieval Results (Combined):")
for i, node in enumerate(hybrid_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    if hasattr(node, 'score') and node.score:
        print(f"   Score: {node.score:.4f}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# PART 6: Comparison Summary
# ============================================================================
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print("""
Index Type          | Best For                      | Speed    | Accuracy
---------------------|-------------------------------|----------|----------
Vector Index        | General semantic search       | Fast     | High
Summary Index       | High-level queries            | Slow     | Medium
Tree Index          | Large docs, hierarchical      | Medium   | High
Keyword Index       | Exact keyword matching        | Fast     | Medium
Hybrid Retrieval    | Production systems            | Slow     | Highest

Recommendations:
────────────────
1. START with Vector Index - Works well for 90% of use cases
2. ADD Keyword Index for specific terminology/codes
3. USE Tree Index for very large document collections (1000s+)
4. COMBINE Vector + Keyword for production (Hybrid)
5. AVOID Summary Index for large datasets (doesn't scale)

Production Best Practice:
─────────────────────────
Vector Index + Keyword Index + Reciprocal Rank Fusion (RRF)
""")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)

print("""
Key Takeaways:
──────────────
1. Different indexes optimize for different retrieval patterns
2. Vector indexes are the most versatile starting point
3. Hybrid approaches combine strengths of multiple strategies
4. Consider your data size, query types, and latency requirements
5. Always measure retrieval quality with evaluation metrics
""")

