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
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SummaryIndex, TreeIndex, KeywordTableIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()

# Configure LlamaIndex settings
Settings.embed_model = OpenAIEmbedding(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    api_key=os.getenv('OPENAI_API_KEY')
)
Settings.llm = OpenAI(
    model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
    api_key=os.getenv('OPENAI_API_KEY')
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
print("\n" + "="*80)
print("PART 2: Summary Index")
print("="*80)

print("\nSummary indexing searches through document summaries/titles.")
print("✓ Good for high-level queries")
print("✓ Returns full documents, not fragments")
print("✗ Slower for large datasets (linear scan)")
print("✗ No vector similarity search\n")

summary_index = SummaryIndex.from_documents(documents)
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")

print("✓ Created summary index")
print(f"\nQuery: '{query}'")
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
print("\n" + "="*80)
print("PART 3: Tree Index (Hierarchical Retrieval)")
print("="*80)

print("\nTree indexing builds a hierarchical structure from leaf to root.")
print("- Queries start at summary level, then drill down")
print("✓ Preserves document context and relationships")
print("✓ Efficient for large document collections")
print("✗ More complex to build and maintain\n")

tree_index = TreeIndex.from_documents(documents)
tree_query_engine = tree_index.as_query_engine(child_branch_factor=2)

print("✓ Created tree index with hierarchical structure")
print(f"\nQuery: '{query}'")
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
print("\n" + "="*80)
print("PART 4: Keyword Table Index")
print("="*80)

print("\nKeyword indexing extracts keywords and uses exact/fuzzy matching.")
print("✓ No embeddings needed - works without vector DB")
print("✓ Good for keyword-specific queries")
print("✗ No semantic understanding")
print("✗ Misses synonyms and related concepts\n")

keyword_index = KeywordTableIndex.from_documents(documents)
keyword_query_engine = keyword_index.as_query_engine()

print("✓ Created keyword table index")
print(f"\nQuery: '{query}'")
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

# Get results from both indexes
vector_nodes = vector_index.as_retriever(similarity_top_k=5).retrieve(query)
keyword_nodes = keyword_index.as_retriever().retrieve(query)

# Simple fusion: combine and deduplicate
seen_ids = set()
hybrid_nodes = []

for node in vector_nodes + keyword_nodes:
    node_id = node.metadata.get('ticket_id', node.node_id)
    if node_id not in seen_ids:
        seen_ids.add(node_id)
        hybrid_nodes.append(node)

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

