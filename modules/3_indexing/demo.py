# -*- coding: utf-8 -*-
"""
================================================================================
MODULE 3: Indexing Strategies for RAG
================================================================================

WHY INDEXING MATTERS:
━━━━━━━━━━━━━━━━━━━━━
We learned embeddings (Module 1) and chunking (Module 2).
Now: How do we ORGANIZE documents for efficient retrieval?

Different indexes = Different retrieval behaviors!

THE 5 STRATEGIES WE'LL EXPLORE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Vector Index (Flat)     - Embed everything, similarity search
2. Summary Index           - Store full docs, LLM evaluates relevance
3. Tree Index              - Hierarchical: summaries → details
4. Keyword Table Index     - Traditional inverted index
5. Hybrid Retrieval        - Combine vector + keyword

WHEN TO USE WHAT:
━━━━━━━━━━━━━━━━━
                    Vector    Summary   Tree      Keyword   Hybrid
Small dataset       ✓ Good    ✓ Good    Overkill  ✓ Good    Overkill
Large dataset       ✓ Best    ✗ Slow    ✓ Good    ✓ Fast    ✓ Best
Semantic queries    ✓ Best    ✓ Good    ✓ Good    ✗ Bad     ✓ Best
Exact match (IDs)   ✗ Bad     ✗ Bad     ✗ Bad     ✓ Best    ✓ Good
Hierarchical docs   ✗ Bad     ✓ OK      ✓ Best    ✗ Bad     ✓ Good

FRAMEWORK: LlamaIndex
━━━━━━━━━━━━━━━━━━━━━
This module uses LlamaIndex (not LangChain) because it has
excellent built-in support for different indexing strategies.

LangChain excels at: Chains, agents, integrations
LlamaIndex excels at: Document indexing, retrieval patterns
"""

# =============================================================================
# IMPORTS
# =============================================================================
import json
import os
import httpx
from dotenv import load_dotenv

# LlamaIndex core components
from llama_index.core import (
    VectorStoreIndex,    # Standard embedding-based index
    SummaryIndex,        # Full document storage, LLM-based relevance
    TreeIndex,           # Hierarchical summarization tree
    KeywordTableIndex,   # Inverted keyword index
    Document,            # Document wrapper with text + metadata
    Settings             # Global configuration
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# =============================================================================
# SETUP: Load Environment Variables
# =============================================================================
load_dotenv()

# Set longer timeout for httpx (used by OpenAI client)
# Some index types make MANY LLM calls and need more time
os.environ["HTTPX_TIMEOUT"] = "300"  # 5 minutes

# =============================================================================
# CONFIGURE LLAMAINDEX SETTINGS
# =============================================================================
#
# LlamaIndex uses a Settings singleton to configure:
#   - embed_model: Which embedding model to use
#   - llm: Which LLM to use for queries and index building
#
# These settings apply globally to all indexes we create.
# =============================================================================
Settings.embed_model = OpenAIEmbedding(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=120,      # 2 min timeout for embedding calls
    max_retries=5     # Retry on failure
)
Settings.llm = OpenAI(
    model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=300,      # 5 min timeout (Tree/Keyword indexes are slow!)
    max_retries=5
)

# =============================================================================
# INTRODUCTION
# =============================================================================
print("="*80)
print("MODULE 3: INDEXING STRATEGIES FOR RAG")
print("="*80)
print("\nThis demo compares 5 different indexing approaches:")
print("1. Vector Index - Semantic similarity search (MOST COMMON)")
print("2. Summary Index - Search through full documents with LLM")
print("3. Tree Index - Hierarchical retrieval (summaries → details)")
print("4. Keyword Table Index - Traditional keyword matching")
print("5. Hybrid Retrieval - Combine multiple strategies (PRODUCTION)")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("Loading Support Tickets")
print("="*80)

with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

# -----------------------------------------------------------------------------
# Convert to LlamaIndex Documents
# -----------------------------------------------------------------------------
# LlamaIndex uses Document objects (similar to LangChain's Document)
# Each Document has:
#   - text: The content to index
#   - metadata: Associated data for filtering/display
# -----------------------------------------------------------------------------
documents = []
for ticket in tickets:
    # Combine all fields into content (rich context for embedding)
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

# Test query - we'll use this across all index types
query = "How do I fix authentication issues after password reset?"
print(f"\nTest Query: '{query}'")

# ============================================================================
# PART 1: Vector Index (Flat Index)
# ============================================================================
#
# THE MOST COMMON APPROACH - Start here for most RAG applications!
#
# HOW IT WORKS:
# ─────────────
#   Build time:
#     Document → [Chunk] → [Embed] → Store vector in index
#
#   Query time:
#     Query → [Embed] → [Find nearest vectors] → Return top-K documents
#
# VISUAL:
#   ┌─────────────────────────────────────────────────┐
#   │  Vector Index (Flat)                            │
#   │                                                 │
#   │  Doc 1 ●──────────[0.02, -0.13, 0.89, ...]     │
#   │  Doc 2 ●──────────[0.15, -0.08, 0.72, ...]     │
#   │  Doc 3 ●──────────[-0.11, 0.23, 0.65, ...]     │
#   │  ...                                            │
#   │  Doc N ●──────────[0.08, -0.19, 0.81, ...]     │
#   │                                                 │
#   │  Query ○──────────[0.03, -0.12, 0.88, ...]     │
#   │         └─→ Find K nearest docs by cos sim     │
#   └─────────────────────────────────────────────────┘
#
# PROS:
#   ✓ Simple and effective for most use cases
#   ✓ Fast similarity search (O(1) with approximate methods)
#   ✓ Captures semantic meaning (synonyms, paraphrases)
#
# CONS:
#   ✗ No hierarchical structure
#   ✗ May return fragmented chunks (lost context)
#   ✗ All documents treated equally (no importance weighting)
#
# ============================================================================
print("\n" + "="*80)
print("PART 1: Vector Index (Flat Index)")
print("="*80)

print("\nVector indexing embeds all chunks and retrieves by semantic similarity.")
print("✓ Simple and effective for most use cases")
print("✓ Fast similarity search with vector databases")
print("✗ No hierarchical structure")
print("✗ May return fragmented chunks\n")

# -----------------------------------------------------------------------------
# Build the Vector Index
# -----------------------------------------------------------------------------
# from_documents() handles:
#   1. Chunking (if needed - our docs are small so no chunking)
#   2. Embedding each chunk via Settings.embed_model
#   3. Storing vectors in memory (or a vector store if configured)
# -----------------------------------------------------------------------------
vector_index = VectorStoreIndex.from_documents(documents)

# -----------------------------------------------------------------------------
# Create Query Engine
# -----------------------------------------------------------------------------
# as_query_engine() wraps the index for easy querying:
#   - similarity_top_k=3: Return top 3 most similar documents
#   - The engine handles: embed query → search → synthesize response
# -----------------------------------------------------------------------------
vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)

print("✓ Created vector index")
print(f"\nQuery: '{query}'")
vector_response = vector_query_engine.query(query)

print("\nVector Index Results:")
print(f"Answer: {vector_response.response}\n")
print("Source Documents:")
for i, node in enumerate(vector_response.source_nodes, 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   Score: {node.score:.4f}")  # Similarity score (higher = more similar)
    print(f"   {node.text[:150]}...")

# ============================================================================
# PART 2: Summary Index
# ============================================================================
#
# COMPLETELY DIFFERENT APPROACH - No embeddings at all!
#
# HOW IT WORKS:
# ─────────────
#   Build time:
#     Just store all documents as-is (no embedding!)
#
#   Query time:
#     For EACH document, ask LLM: "Is this relevant to the query?"
#     Collect relevant docs → Synthesize answer
#
# VISUAL - Query Process (Linear Scan):
#   ┌──────────────────────────────────────────────────┐
#   │  Query: "auth issues after password reset"       │
#   │                                                  │
#   │  Doc 1 → [LLM: Relevant?] → YES ────┐           │
#   │  Doc 2 → [LLM: Relevant?] → NO      │           │
#   │  Doc 3 → [LLM: Relevant?] → YES ────┤           │
#   │  Doc 4 → [LLM: Relevant?] → NO      │           │
#   │  ...                                │           │
#   │  Doc N → [LLM: Relevant?] → YES ────┤           │
#   │                                     ▼           │
#   │                              [Synthesize]       │
#   │                                     │           │
#   │                                  Answer         │
#   └──────────────────────────────────────────────────┘
#
# WHY IT'S SLOW:
#   O(n) complexity - must check EVERY document
#   Each check = LLM call (or at least LLM attention)
#   10 docs = 2 sec, 100 docs = 20 sec, 1000 docs = 200 sec!
#
# WHEN TO USE:
#   - Small collections (<50 documents)
#   - Need full document context (not fragments)
#   - High-level summarization queries
#   - When semantic similarity might miss nuanced relevance
#
# WHEN NOT TO USE:
#   - Large datasets (doesn't scale!)
#   - Real-time applications (too slow)
#
# ============================================================================
print("\n" + "="*80)
print("PART 2: Summary Index")
print("="*80)

print("\nSummary indexing searches through document summaries/titles.")
print("✓ Good for high-level queries")
print("✓ Returns full documents, not fragments")
print("✗ Slower for large datasets (linear scan)")
print("✗ No vector similarity search\n")

# -----------------------------------------------------------------------------
# Build the Summary Index
# -----------------------------------------------------------------------------
# NOTE: No embeddings are generated!
# Documents are stored as-is for sequential LLM evaluation
# -----------------------------------------------------------------------------
summary_index = SummaryIndex.from_documents(documents)

# -----------------------------------------------------------------------------
# Query Engine with Tree Summarize
# -----------------------------------------------------------------------------
# response_mode="tree_summarize":
#   1. Collects all relevant documents
#   2. If too many, summarizes in groups
#   3. Combines group summaries into final answer
#
# Good for comprehensive answers that synthesize multiple sources
# -----------------------------------------------------------------------------
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")

print("✓ Created summary index")
print(f"\nQuery: '{query}'")

# Query execution process:
# 1. LLM examines each document for relevance
# 2. Collects all relevant documents
# 3. Uses tree_summarize to synthesize final answer from all relevant docs
summary_response = summary_query_engine.query(query)

print("\nSummary Index Results:")
print(f"Answer: {summary_response.response}\n")
print("Source Documents:")
for i, node in enumerate(summary_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# PART 3: Tree Index (Hierarchical Retrieval)
# ============================================================================
#
# BEST FOR LARGE DOCUMENT COLLECTIONS
#
# HOW IT WORKS:
# ─────────────
#   Build time (EXPENSIVE - many LLM calls!):
#     1. Each document becomes a LEAF node
#     2. LLM groups related docs and creates SUMMARY nodes
#     3. Repeat until one ROOT node summarizes everything
#
#   Query time (EFFICIENT - log(n) traversal):
#     1. Start at root
#     2. LLM scores children: "Which branch is most relevant?"
#     3. Traverse down the best branch(es)
#     4. Reach leaf nodes → return those documents
#
# VISUAL - Tree Structure:
#   ┌────────────────────────────────────────────────────────┐
#   │                                                        │
#   │               [Root: All 50 tickets summary]           │
#   │                          │                             │
#   │        ┌─────────────────┼─────────────────┐           │
#   │        ▼                 ▼                 ▼           │
#   │   [Auth Issues]    [Performance]      [Billing]        │
#   │        │                 │                 │           │
#   │   ┌────┼────┐    ┌───────┼───────┐     ┌──┴──┐        │
#   │   ▼         ▼    ▼       ▼       ▼     ▼     ▼        │
#   │ [T-1]    [T-5] [T-7]   [T-9]  [T-12] [T-20] [T-25]    │
#   │                                                        │
#   │ Query: "auth issues after password reset"              │
#   │        └─→ Root → Auth Issues → T-1, T-5 (leaves)     │
#   └────────────────────────────────────────────────────────┘
#
# WHY IT'S EFFICIENT:
#   Instead of searching 50 docs (O(n)), we traverse ~3 levels (O(log n))
#   For 1000 docs: Vector = 1000 comparisons, Tree = ~10 LLM decisions
#
# TRADE-OFF:
#   Build time: SLOW (many LLM calls to create summaries)
#   Query time: FAST (logarithmic traversal)
#
# WHEN TO USE:
#   - Very large document collections (1000s of docs)
#   - Hierarchically structured content (books, documentation)
#   - When build time is acceptable (offline indexing)
#
# ============================================================================
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

# Use all documents (but warn about LLM costs)
tree_documents = documents
print(f"Building Tree Index with {len(tree_documents)} documents...")

# -----------------------------------------------------------------------------
# Build the Tree Index
# -----------------------------------------------------------------------------
# LlamaIndex will:
#   1. Create leaf nodes from each document
#   2. Call LLM to generate summaries for groups of related docs
#   3. Repeat until a single root summary exists
# This is EXPENSIVE - each grouping requires an LLM call!
# -----------------------------------------------------------------------------
tree_index = TreeIndex.from_documents(tree_documents)

# -----------------------------------------------------------------------------
# Create Query Engine
# -----------------------------------------------------------------------------
# child_branch_factor=2:
#   At each level, explore the top 2 most relevant branches
#   Higher = broader search (more paths explored)
#   Lower = focused search (may miss relevant docs in other branches)
# -----------------------------------------------------------------------------
tree_query_engine = tree_index.as_query_engine(child_branch_factor=2)

print("✓ Created tree index with hierarchical structure")
print(f"\nQuery: '{query}'")

# Query execution (hierarchical traversal):
# 1. Start at root summary
# 2. LLM scores each child branch for relevance to query
# 3. Traverse into top-k branches (child_branch_factor)
# 4. Repeat until reaching leaf nodes
# 5. Collect relevant leaves, synthesize final answer
tree_response = tree_query_engine.query(query)

print("\nTree Index Results:")
print(f"Answer: {tree_response.response}\n")
print("Source Documents:")
for i, node in enumerate(tree_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# PART 4: Keyword Table Index
# ============================================================================
#
# THE TRADITIONAL APPROACH (Pre-embeddings era)
#
# HOW IT WORKS:
# ─────────────
#   Build time:
#     For each document, extract keywords (via LLM or rules)
#     Build inverted index: keyword → [doc_ids]
#
#   Query time:
#     Extract keywords from query
#     Look up documents containing those keywords
#     Return matching documents
#
# VISUAL - Inverted Index:
#   ┌────────────────────────────────────────────────┐
#   │  Keyword Table (Inverted Index)                │
#   │                                                │
#   │  "password"  →  [T-1, T-5, T-12]              │
#   │  "login"     →  [T-1, T-3, T-8]               │
#   │  "timeout"   →  [T-7, T-15]                   │
#   │  "billing"   →  [T-20, T-25]                  │
#   │  "API"       →  [T-30, T-35, T-42]            │
#   │                                                │
#   │  Query: "password reset after login"           │
#   │         └─→ Keywords: {password, login}        │
#   │         └─→ Match: T-1 (both keywords!)       │
#   └────────────────────────────────────────────────┘
#
# KEY LIMITATION - No Semantic Understanding:
#   "authentication" ≠ "login" (unless both in same doc)
#   "reset password" won't find "forgot credentials"
#
# WHEN TO USE:
#   - Exact term matching (error codes, ticket IDs)
#   - No embedding costs needed
#   - Combine with vector search for hybrid approach
#
# WHEN NOT TO USE:
#   - Semantic queries (use Vector Index instead)
#   - User queries with synonyms/paraphrases
#
# ============================================================================
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

# Use all documents
keyword_documents = documents
print(f"Building Keyword Index with {len(keyword_documents)} documents...")

# -----------------------------------------------------------------------------
# Build the Keyword Table Index
# -----------------------------------------------------------------------------
# LlamaIndex uses LLM to extract keywords from each document
# Builds inverted index: keyword → [document IDs]
# Alternative: Use simple regex/rule-based extraction (faster, no LLM)
# -----------------------------------------------------------------------------
keyword_index = KeywordTableIndex.from_documents(keyword_documents)

# Create query engine
keyword_query_engine = keyword_index.as_query_engine()

print("✓ Created keyword table index")
print(f"\nQuery: '{query}'")

# Query process:
# 1. Extract keywords from query (via LLM)
# 2. Look up documents in inverted index
# 3. Return documents containing query keywords
# 4. Synthesize answer from matched documents
keyword_response = keyword_query_engine.query(query)

print("\nKeyword Index Results:")
print(f"Answer: {keyword_response.response}\n")
print("Source Documents:")
for i, node in enumerate(keyword_response.source_nodes[:3], 1):
    print(f"\n{i}. {node.metadata.get('ticket_id', 'Unknown')}")
    print(f"   {node.text[:150]}...")

# ============================================================================
# PART 5: Hybrid Retrieval
# ============================================================================
#
# THE PRODUCTION APPROACH - Best of both worlds!
#
# THE PROBLEM:
# ────────────
#   Vector search alone: Finds "login issues" for "auth problems" ✓
#                        Misses "Ticket T-123" for "T-123" ✗
#
#   Keyword search alone: Finds "Ticket T-123" for "T-123" ✓
#                         Misses "auth problems" for "login issues" ✗
#
# THE SOLUTION - Combine both:
# ────────────────────────────
#   Query: "authentication timeout error T-123"
#
#   Vector Search:              Keyword Search:
#   [Doc A: 0.89] (login)       [Doc C: 3 keywords]
#   [Doc B: 0.85] (auth error)  [Doc A: 2 keywords]
#   [Doc C: 0.75] (timeout)     [Doc E: "T-123" exact match!]
#
#   Fusion:
#   [Doc A] - found by BOTH (high confidence!)
#   [Doc B] - semantic match
#   [Doc C] - found by BOTH (high confidence!)
#   [Doc E] - keyword match (exact ID)
#
# FUSION STRATEGIES:
# ──────────────────
# 1. Simple union (what we do here) - combine and deduplicate
# 2. Reciprocal Rank Fusion (RRF) - score by rank in each list
#      score(doc) = Σ 1/(k + rank_i) for each retriever
# 3. Weighted combination - assign weights to each retriever
#
# ============================================================================
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

print("✓ Using Vector + Keyword hybrid approach")
print(f"\nQuery: '{query}'")

# -----------------------------------------------------------------------------
# Step 1: Retrieve from Vector Index (Semantic)
# -----------------------------------------------------------------------------
# Finds semantically similar documents
# "auth issues" → finds "login problems", "SSO failures", etc.
# -----------------------------------------------------------------------------
vector_nodes = vector_index.as_retriever(similarity_top_k=5).retrieve(query)

# -----------------------------------------------------------------------------
# Step 2: Retrieve from Keyword Index (Exact)
# -----------------------------------------------------------------------------
# Finds documents with exact keyword matches
# "authentication" → finds docs containing that word
# Great for: ticket IDs, error codes, product names
# -----------------------------------------------------------------------------
keyword_nodes = keyword_index.as_retriever().retrieve(query)

# -----------------------------------------------------------------------------
# Step 3: Fusion - Combine and Deduplicate
# -----------------------------------------------------------------------------
# Simple approach: Union of both result sets
# More sophisticated: Reciprocal Rank Fusion (RRF)
#   score(doc) = Σ 1/(k + rank_i) where k=60 typically
#
# Here we prioritize vector results (come first) and dedupe by ticket_id
# -----------------------------------------------------------------------------
seen_ids = set()
hybrid_nodes = []

for node in vector_nodes + keyword_nodes:
    node_id = node.metadata.get('ticket_id', node.node_id)
    if node_id not in seen_ids:
        seen_ids.add(node_id)
        hybrid_nodes.append(node)

# Documents found by BOTH methods are likely most relevant!

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
┌────────────────────┬───────────────────────────────┬──────────┬───────────┐
│ Index Type         │ Best For                      │ Speed    │ Accuracy  │
├────────────────────┼───────────────────────────────┼──────────┼───────────┤
│ Vector Index       │ General semantic search       │ Fast     │ High      │
│ Summary Index      │ High-level queries (small)    │ Slow     │ Medium    │
│ Tree Index         │ Large docs, hierarchical      │ Medium   │ High      │
│ Keyword Index      │ Exact keyword matching        │ Fast     │ Medium    │
│ Hybrid Retrieval   │ Production systems            │ Medium   │ Highest   │
└────────────────────┴───────────────────────────────┴──────────┴───────────┘

DECISION FLOWCHART:
───────────────────
                    ┌─────────────────────┐
                    │ What's your use case?│
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    Small dataset       Large dataset        Production
    (<100 docs)         (1000s+ docs)        (any size)
           │                   │                   │
           ▼                   ▼                   ▼
    Vector Index          Tree Index         Hybrid
    (simple, fast)     (hierarchical)    (Vector + Keyword)

RECOMMENDATIONS:
────────────────
1. START with Vector Index - Works well for 90% of use cases
2. ADD Keyword Index for specific terminology/codes (error codes, IDs)
3. USE Tree Index for very large document collections (1000s+)
4. COMBINE Vector + Keyword for production (Hybrid)
5. AVOID Summary Index for large datasets (doesn't scale)

PRODUCTION BEST PRACTICE:
─────────────────────────
Vector Index + Keyword Index + Reciprocal Rank Fusion (RRF)
""")

# ============================================================================
# DEMO COMPLETE
# ============================================================================
print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)

print("""
KEY TAKEAWAYS:
──────────────
1. Vector Index is your DEFAULT choice
   - Semantic search, fast, effective for most cases

2. Keyword Index is your COMPLEMENT
   - Catches exact matches that vectors miss

3. Hybrid = Vector + Keyword = PRODUCTION QUALITY
   - Best accuracy, slightly slower

4. Tree Index for SCALE
   - Use when dataset is too large for flat vector search

5. Always MEASURE retrieval quality
   - Next module: Evaluation metrics!

NEXT: Module 4 - Building the RAG Pipeline
""")
