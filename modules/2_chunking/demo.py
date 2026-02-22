# -*- coding: utf-8 -*-
"""
Module 2: Chunking & Vector Stores Demo
========================================

This demo teaches:
1. Different chunking strategies for long documents
2. Building a Chroma vector store
3. Comparing retrieval quality across strategies
4. Metadata filtering for targeted search

LEARNING RESOURCES:
- Text Splitting Guide: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- Chroma DB: https://docs.trychroma.com/
- Chunking Best Practices: https://www.pinecone.io/learn/chunking-strategies/

WHY CHUNKING MATTERS:
━━━━━━━━━━━━━━━━━━━━━
The "Goldilocks Problem":
- Too small chunks → Loss of context, noisy embeddings
- Too large chunks → Diluted meaning, irrelevant info retrieved
- Just right → Self-contained units of meaning that retrieve accurately

When do you need chunking?
- Document exceeds embedding model limit (8,191 tokens for OpenAI)
- Document is too long for precise retrieval
- You want to retrieve specific sections, not entire documents
"""

import json
import os
from langchain_text_splitters import (  # Various splitting strategies
    RecursiveCharacterTextSplitter,  # Best general-purpose splitter
    CharacterTextSplitter,  # Simple split by character count
    MarkdownHeaderTextSplitter,  # Splits based on markdown headers
    HTMLHeaderTextSplitter  # Splits based on HTML tags
)
from langchain_experimental.text_splitter import SemanticChunker  # AI-powered semantic chunking
from langchain_community.vectorstores import Chroma  # Vector database
from langchain_openai import OpenAIEmbeddings  # OpenAI embedding function
from langchain_core.documents import Document  # Document abstraction
from dotenv import load_dotenv

# ============================================================================
# SETUP: Load environment and data
# ============================================================================

# Load API keys from .env file (never hardcode API keys!)
load_dotenv()

print("="*80)
print("MODULE 2: CHUNKING & VECTOR STORES")
print("="*80)

# Load our support ticket dataset
# These are relatively SHORT documents - good for demonstrating chunking concepts
# In real scenarios, you'd chunk PDFs, articles, manuals (much longer!)
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"\nLoaded {len(tickets)} support tickets")

# ============================================================================
# PART 1: Chunking Strategies
# ============================================================================
# 
# We'll demo 5 different chunking strategies:
# 1. Fixed-size: Split by character/token count (simple but may break sentences)
# 2. Recursive: Try progressively smaller separators (best general-purpose)
# 3. Semantic: Use embeddings to detect topic changes (expensive but best quality)
# 4. Markdown-aware: Split by headers (great for documentation)
# 5. HTML-aware: Split by HTML tags (great for web content)
#
# ============================================================================
print("\n" + "="*80)
print("PART 1: Chunking Strategies")
print("="*80)

# -----------------------------------------------------------------------------
# First, convert our ticket data into LangChain Document objects
# Document = page_content (text) + metadata (structured info for filtering)
# -----------------------------------------------------------------------------
documents = []
for ticket in tickets:
    # Combine all ticket fields into a single text block
    # TIP: Include all relevant context that helps understand the document
    full_text = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
    """.strip()
    
    # Create Document object with metadata
    # Metadata is CRUCIAL - it enables filtering later!
    # Example: "Find similar tickets, but only in the 'Authentication' category"
    doc = Document(
        page_content=full_text,  # The actual text content
        metadata={
            'ticket_id': ticket['ticket_id'],   # For identifying results
            'category': ticket['category'],      # For category filtering
            'priority': ticket['priority']       # For priority filtering
        }
    )
    documents.append(doc)

print(f"Created {len(documents)} documents")
print(f"\nSample document length: {len(documents[0].page_content)} characters")

# =============================================================================
# STRATEGY 1: Fixed-Size Chunking
# =============================================================================
# 
# HOW IT WORKS:
#   Split text into equal-sized chunks based on character count
#   Overlap ensures we don't lose context at chunk boundaries
#
# EXAMPLE with chunk_size=10, overlap=3:
#   Text: "Hello world, how are you doing today?"
#   Chunk 1: "Hello worl" (chars 0-10)
#   Chunk 2: "orld, how " (chars 7-17, overlaps "orl")
#   Chunk 3: "how are yo" (chars 14-24, overlaps "how")
#   ... and so on
#
# PROS: Simple, predictable chunk sizes
# CONS: May split mid-word or mid-sentence (no semantic awareness)
# =============================================================================
print("\n--- Strategy 1: Fixed-Size Chunking ---")

fixed_splitter = CharacterTextSplitter(
    chunk_size=200,      # Maximum characters per chunk
    chunk_overlap=20,    # Characters to repeat between chunks (10% overlap)
    separator="\n"       # Prefer splitting on newlines when possible
)
fixed_chunks = fixed_splitter.split_documents(documents)

print(f"✓ Created {len(fixed_chunks)} chunks")
print(f"  Chunk size: 200 chars, Overlap: 20 chars")
print(f"  Sample chunk: {fixed_chunks[0].page_content[:100]}...")

# =============================================================================
# STRATEGY 2: Recursive Character Splitting (RECOMMENDED DEFAULT)
# =============================================================================
#
# HOW IT WORKS:
#   1. Try to split on "\n\n" (paragraph breaks) first
#   2. If chunks still too big, try "\n" (line breaks)
#   3. If still too big, try ". " (sentences)
#   4. If still too big, try " " (words)
#   5. Last resort: split character by character
#
# This PRESERVES semantic boundaries when possible!
#
# EXAMPLE:
#   Text: "Paragraph 1. Sentence 1. Sentence 2.\n\nParagraph 2. ..."
#   → First splits on \n\n (paragraphs)
#   → If paragraph too big, splits on sentences
#   → Much better than splitting mid-word!
#
# PROS: Respects natural boundaries, works well for most content
# CONS: Not semantic-aware (doesn't understand meaning)
# =============================================================================
print("\n--- Strategy 2: Recursive Character Splitting ---")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # Max characters per chunk
    chunk_overlap=50,    # 50 char overlap (~17%)
    # Separators tried in ORDER - most specific first!
    separators=[
        "\n\n",  # 1st: Paragraph breaks (best split point)
        "\n",    # 2nd: Line breaks
        ". ",    # 3rd: Sentence boundaries
        " "      # 4th: Word boundaries (last resort for text)
    ]
)
recursive_chunks = recursive_splitter.split_documents(documents)

print(f"✓ Created {len(recursive_chunks)} chunks")
print(f"  Tries to split on paragraph/sentence boundaries")
print(f"  Sample chunk: {recursive_chunks[0].page_content[:100]}...")

# =============================================================================
# STRATEGY 3: Semantic Chunking (Embedding-Based)
# =============================================================================
#
# HOW IT WORKS:
#   1. Split document into sentences
#   2. Get embedding for EACH sentence
#   3. Compare consecutive sentences using cosine similarity
#   4. When similarity DROPS significantly → topic changed → split here!
#
# PROS: Best semantic coherence, natural topic boundaries
# CONS: EXPENSIVE (needs embedding for every sentence), slow
# =============================================================================
print("\n--- Strategy 3: Semantic Chunking ---")
print("  Note: Semantic chunking uses embeddings to find natural break points")

# Initialize OpenAI embeddings for semantic chunker
# IMPORTANT: This costs money! Each sentence needs an embedding API call
embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)

# Demo with a paragraph that has CLEAR topic shifts
# Using completely unrelated domains for clearer separation
demo_paragraph = """
The Mars rover collected soil samples from the Jezero crater last week. Scientists believe these rocks may contain signs of ancient microbial life. NASA plans to retrieve these samples in a future mission. The discovery could reshape our understanding of life in the solar system.

Grandma's apple pie recipe starts with peeling six large Granny Smith apples. Mix flour, sugar, and cinnamon for the filling. Roll the dough thin and crimp the edges carefully. Bake at 375 degrees for 45 minutes until golden brown.

The defendant was charged with breach of contract under Section 12. The plaintiff seeks damages of fifty thousand dollars plus legal fees. Both parties agreed to mediation before proceeding to trial. The judge scheduled the preliminary hearing for next month.
"""

print("\n  📝 Demo Text (3 distinct topics):")
print("  " + "-"*70)
print("  Topic 1: Space exploration (sentences 1-4)")
print("  Topic 2: Cooking recipe (sentences 5-8)")
print("  Topic 3: Legal case (sentences 9-12)")
print("  " + "-"*70)

semantic_splitter = SemanticChunker(
    embeddings=embeddings_model,
    # How to detect "topic change":
    # - "percentile": Split where similarity is in bottom X percentile
    # - "standard_deviation": Split where similarity is X std devs below mean
    # - "interquartile": Split where similarity is below Q1 - 1.5*IQR (outlier detection)
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.0  # Split when similarity drops 1 std dev below mean
)

demo_doc = Document(page_content=demo_paragraph.strip())
semantic_chunks = semantic_splitter.split_documents([demo_doc])
print(f"\n✓ Created {len(semantic_chunks)} chunks")
print("  Note: Semantic chunking results vary based on embedding model and threshold settings")

# Show each semantic chunk
print("\n  📊 Resulting Semantic Chunks:")
print("  " + "-"*70)
for i, chunk in enumerate(semantic_chunks):
    print(f"\n  Chunk {i+1} ({len(chunk.page_content)} chars):")
    print("  " + "~"*60)
    # Show full content for clarity
    for line in chunk.page_content.strip().split('\n'):
        if line.strip():
            print(f"    {line.strip()}")
    print("  " + "~"*60)

print("\n  ✨ The chunker attempts to detect topic shifts between space → cooking → legal")
print("  Adjust breakpoint_threshold_amount (lower = more sensitive) if results vary.")

# =============================================================================
# STRATEGY 4: Markdown Structure-Aware Splitting
# =============================================================================
#
# HOW IT WORKS:
#   Split on markdown headers (#, ##, ###, etc.)
#   Each section becomes a chunk WITH header info in metadata!
#
# EXAMPLE:
#   # Main Title          → Chunk 1, metadata: {"Header 1": "Main Title"}
#   ## Section A          → Chunk 2, metadata: {"Header 1": "Main Title", "Header 2": "Section A"}
#   Content here...
#   ## Section B          → Chunk 3, metadata: {..., "Header 2": "Section B"}
#
# PROS: Preserves document hierarchy, great for documentation
# CONS: Only works for markdown, sections may be too large/small
# =============================================================================
print("\n--- Strategy 4: Markdown Header Splitting ---")

# Sample markdown documentation (simulating a knowledge base article)
markdown_doc = """
# Database Troubleshooting Guide

## Connection Issues

### Timeout Errors
If you encounter timeout errors, check the connection string and ensure the database server is reachable.
Increase the connection timeout value in your configuration.

### Authentication Failures
Verify your credentials are correct. Check for expired passwords or locked accounts.
Ensure the user has proper permissions on the database.

## Performance Problems

### Slow Queries
Analyze query execution plans using EXPLAIN.
Consider adding indexes on frequently queried columns.
Review and optimize JOIN operations.

### High CPU Usage
Monitor long-running queries.
Check for missing indexes causing table scans.
"""

# Define which headers to split on
# Format: (header_marker, metadata_key)
headers_to_split_on = [
    ("#", "Header 1"),    # H1 tags
    ("##", "Header 2"),   # H2 tags  
    ("###", "Header 3"),  # H3 tags
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # Keep headers in the chunk content (usually want True)
)
md_chunks = markdown_splitter.split_text(markdown_doc)

print(f"✓ Created {len(md_chunks)} chunks from markdown")
print(f"  Preserves document structure and header context")
if md_chunks:
    print(f"  Sample chunk with metadata:")
    print(f"    Content: {md_chunks[0].page_content[:80]}...")
    print(f"    Metadata: {md_chunks[0].metadata}")  # Shows header hierarchy!

# =============================================================================
# STRATEGY 5: HTML Structure-Aware Splitting
# =============================================================================
#
# HOW IT WORKS:
#   Same concept as Markdown splitting, but for HTML documents
#   Splits on <h1>, <h2>, <h3> tags and preserves hierarchy in metadata
#
# PROS: Great for web scraping, knowledge bases, wikis
# CONS: Only works for HTML content
# =============================================================================
print("\n--- Strategy 5: HTML Header Splitting ---")

# Sample HTML documentation (simulating a scraped help page)
html_doc = """
<!DOCTYPE html>
<html>
<body>
    <h1>Email Configuration Guide</h1>
    
    <h2>SMTP Settings</h2>
    <p>Configure your SMTP server settings in the admin panel. Use port 587 for TLS or port 465 for SSL.</p>
    
    <h3>Common SMTP Servers</h3>
    <p>Gmail: smtp.gmail.com, Outlook: smtp.office365.com, Yahoo: smtp.mail.yahoo.com</p>
    
    <h2>IMAP Configuration</h2>
    <p>Set up IMAP to sync your emails across devices. Use port 993 for secure connections.</p>
    
    <h3>Folder Mapping</h3>
    <p>Map your email folders to the appropriate IMAP folders for proper synchronization.</p>
</body>
</html>
"""

# Map HTML tags to metadata keys
headers_to_split_on_html = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on_html
)
html_chunks = html_splitter.split_text(html_doc)

print(f"✓ Created {len(html_chunks)} chunks from HTML")
print(f"  Respects HTML semantic structure")
if html_chunks:
    print(f"  Sample chunk with metadata:")
    print(f"    Content: {html_chunks[0].page_content[:80]}...")
    print(f"    Metadata: {html_chunks[0].metadata}")

# =============================================================================
# STRATEGY 6: No Chunking (Whole Documents)
# =============================================================================
#
# WHEN TO USE:
#   - Documents are already short (like our support tickets)
#   - Each document is a self-contained unit
#   - You want to retrieve entire documents, not sections
#
# OUR TICKETS: ~200-400 chars each → No chunking needed!
# LONG PDF: 50,000 chars → Definitely needs chunking!
# =============================================================================
print("\n--- Strategy 6: Whole Documents (No Chunking) ---")
print(f"✓ Using {len(documents)} whole documents")
print(f"  Good for small documents like our tickets")

# ============================================================================
# PART 2: Chroma Vector Store
# ============================================================================
#
# WHY CHROMA?
#   ✓ Automatic persistence (data survives restart)
#   ✓ Metadata storage and filtering
#   ✓ Built-in embedding generation
#   ✓ Collection management
#   ✓ Production-ready features
#
# THIS IS OUR RECOMMENDED APPROACH FOR THE WORKSHOP!
# ============================================================================
print("\n" + "="*80)
print("PART 2: Chroma Vector Store")
print("="*80)

# Use LangChain's embedding wrapper (handles API calls internally)
embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)

query = "Authentication problems after password reset"

print("\nBuilding Chroma vector store...")

# from_documents() handles everything:
#   1. Extracts text from each Document
#   2. Generates embeddings via the embedding model
#   3. Stores vectors + metadata + original text
#   4. Persists to disk (if persist_directory specified)
chroma_store = Chroma.from_documents(
    documents=documents,              # Our LangChain Document objects
    embedding=embeddings_model,       # OpenAI embeddings
    collection_name="support_tickets",# Like a "table" in a database
    persist_directory="./chroma_db"   # Save to disk for persistence
)
print("✓ Chroma store created and persisted")

# -----------------------------------------------------------------------------
# Basic Similarity Search
# -----------------------------------------------------------------------------
print(f"\nSearching in Chroma: '{query}'")
chroma_results = chroma_store.similarity_search(query, k=3)

print(f"\nTop {len(chroma_results)} results:")
for i, doc in enumerate(chroma_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")

# -----------------------------------------------------------------------------
# MMR Search (Maximal Marginal Relevance)
# -----------------------------------------------------------------------------
# PROBLEM: Similarity search can return very similar documents
# SOLUTION: MMR balances relevance AND diversity
#
# EXAMPLE (query: "login issues after password reset"):
#   Normal similarity search top-3 might be:
#     1) "Can't login after reset on web"
#     2) "Login fails after password reset"
#     3) "Password reset causes auth error"
#   These are all relevant, but near-duplicates (low coverage).
#
# Large-scale note (millions of chunks/documents):
#   Normal semantic search often returns multiple top hits from the SAME source
#   document because adjacent chunks are highly similar.
#   You get depth from one document, but poor cross-document coverage.
#
#   MMR top-3 might be:
#     1) "Can't login after reset on web"         (core issue)
#     2) "MFA token invalid after password reset" (different angle)
#     3) "Session cookie not refreshed"           (another angle)
#   Still relevant, but broader context for better troubleshooting/RAG answers.
#
# HOW IT WORKS:
#   1. Get top-k similar documents
#   2. Select first result normally
#   3. For each remaining slot, pick document that is:
#      - Similar to query (relevance)
#      - Different from already-selected docs (diversity)
#
# USE CASE: When you want varied perspectives, not just the "best" match
# -----------------------------------------------------------------------------
print("\n--- Using MMR for Diverse Results ---")
mmr_results = chroma_store.max_marginal_relevance_search(query, k=3)

print(f"\nMMR Results (more diverse):")
for i, doc in enumerate(mmr_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Title: {tickets[int(doc.metadata['ticket_id'].split('-')[1]) - 1]['title']}")

# ============================================================================
# PART 3: Metadata Filtering
# ============================================================================
#
# THE KILLER FEATURE OF VECTOR DATABASES!
#
# Without filtering: Search ALL documents semantically
# With filtering: Search only documents that match criteria
#
# EXAMPLES:
#   - Only high priority tickets
#   - Only tickets from last week
#   - Only tickets in "Authentication" category
#   - Only tickets from specific customer tier
#
# This is MUCH faster than filtering after retrieval!
# ============================================================================
print("\n" + "="*80)
print("PART 3: Metadata Filtering")
print("="*80)

# -----------------------------------------------------------------------------
# Example 1: Filter by category
# -----------------------------------------------------------------------------
# Scenario: User asks about login issues
# We want to search ONLY authentication-related tickets
# -----------------------------------------------------------------------------
print("\nSearching only in 'Authentication' category:")
filtered_results = chroma_store.similarity_search(
    query,
    k=3,
    filter={"category": "Authentication"}  # Only match this category
)

print(f"\nFiltered results ({len(filtered_results)}):")
for i, doc in enumerate(filtered_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")
    print(f"Content: {doc.page_content[:100]}...")

# -----------------------------------------------------------------------------
# Example 2: Filter by priority
# -----------------------------------------------------------------------------
# Scenario: Looking for high-priority issues to prioritize
# Combine semantic search with priority filter
# -----------------------------------------------------------------------------
print("\n\nSearching only 'High' priority tickets:")
high_priority_results = chroma_store.similarity_search(
    "Database performance issues",
    k=3,
    filter={"priority": "High"}  # Only high priority
)

print(f"\nHigh priority results ({len(high_priority_results)}):")
for i, doc in enumerate(high_priority_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Priority: {doc.metadata['priority']}")

# ============================================================================
# PART 4: Comparing Chunking Strategies
# ============================================================================
#
# PURPOSE: Show how different chunking affects retrieval
#
# For our small tickets, chunking doesn't make a huge difference
# BUT for long documents, the choice is critical!
#
# EXPERIMENT FRAMEWORK:
#   1. Build same index with different chunking
#   2. Run same queries
#   3. Compare results
#   4. Measure relevance (which found the right answer?)
# ============================================================================
print("\n" + "="*80)
print("PART 4: Evaluating Chunking Strategies")
print("="*80)

# Build stores with different chunking
print("\nBuilding vector stores with different chunking strategies...")

# Store 1: Whole documents (no chunking)
store_whole = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    collection_name="whole_docs"
)

# Store 2: Fixed-size chunks (may split mid-sentence)
store_fixed = Chroma.from_documents(
    documents=fixed_chunks,
    embedding=embeddings_model,
    collection_name="fixed_chunks"
)

# Store 3: Recursive chunks (splits at natural boundaries)
store_recursive = Chroma.from_documents(
    documents=recursive_chunks,
    embedding=embeddings_model,
    collection_name="recursive_chunks"
)

test_query = "Database connection failures"
print(f"\nTest query: '{test_query}'")

# Compare results from each strategy
stores = [
    ("Whole Documents", store_whole),
    ("Fixed Chunks", store_fixed),
    ("Recursive Chunks", store_recursive)
]

for name, store in stores:
    results = store.similarity_search(test_query, k=1)
    print(f"\n{name}:")
    if results:
        print(f"  Top result: {results[0].page_content[:100]}...")
        print(f"  Length: {len(results[0].page_content)} chars")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("""
KEY TAKEAWAYS:
━━━━━━━━━━━━━━━
1. CHUNKING is crucial for long documents (not critical for short tickets)
   → RecursiveCharacterTextSplitter is a good default
   → SemanticChunker is best quality but expensive

2. CHROMA offers production-ready features
   → Persistence (survives restart)
   → Metadata filtering (search within subsets)
   → MMR for diverse results

3. METADATA FILTERING is powerful
   → Filter by category, priority, date, etc.
   → Faster than post-retrieval filtering

4. ALWAYS EXPERIMENT with your specific data
   → What works for support tickets ≠ what works for PDFs
   → Measure retrieval quality!

NEXT: Module 3 - Indexing Strategies
""")

