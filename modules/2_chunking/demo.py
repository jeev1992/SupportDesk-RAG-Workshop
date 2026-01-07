# -*- coding: utf-8 -*-
"""
Hour 2: Chunking & Vector Stores Demo
======================================

This demo teaches:
1. Different chunking strategies for long documents
2. Building a FAISS vector store
3. Using Chroma for high-level abstraction
4. Comparing retrieval quality across strategies

LEARNING RESOURCES:
- Text Splitting Guide: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- FAISS Documentation: https://github.com/facebookresearch/faiss/wiki
- Chroma DB: https://docs.trychroma.com/
- Chunking Best Practices: https://www.pinecone.io/learn/chunking-strategies/

WHY CHUNKING MATTERS:
- Long documents exceed LLM context windows
- Smaller chunks = more precise retrieval
- Too small = loss of context
- Too large = irrelevant information included
- Goal: Each chunk should be a self-contained unit of meaning
"""

import json
import numpy as np
import os
from openai import OpenAI
import faiss  # Facebook AI Similarity Search - fast vector search library
from langchain_text_splitters import (  # Various splitting strategies
    RecursiveCharacterTextSplitter,  # Best general-purpose splitter
    CharacterTextSplitter,  # Simple split by character count
    MarkdownHeaderTextSplitter,  # Splits based on markdown headers
    HTMLHeaderTextSplitter  # Splits based on HTML tags
)
from langchain_experimental.text_splitter import SemanticChunker  # AI-powered semantic chunking
from langchain_community.vectorstores import Chroma, FAISS as LangChainFAISS  # Vector databases
from langchain_openai import OpenAIEmbeddings  # OpenAI embedding function
from langchain_core.documents import Document  # Document abstraction
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*80)
print("HOUR 2: CHUNKING & VECTOR STORES")
print("="*80)

# Load tickets
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"\nLoaded {len(tickets)} support tickets")

# ============================================================================
# PART 1: Chunking Strategies
# ============================================================================
print("\n" + "="*80)
print("PART 1: Chunking Strategies")
print("="*80)

# Create full documents from tickets
documents = []
for ticket in tickets:
    full_text = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
    """.strip()
    
    doc = Document(
        page_content=full_text,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority']
        }
    )
    documents.append(doc)

print(f"Created {len(documents)} documents")
print(f"\nSample document length: {len(documents[0].page_content)} characters")

# Strategy 1: Fixed-size chunking
print("\n--- Strategy 1: Fixed-Size Chunking ---")
fixed_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator="\n"
)
fixed_chunks = fixed_splitter.split_documents(documents)
print(f"✓ Created {len(fixed_chunks)} chunks")
print(f"  Chunk size: 200 chars, Overlap: 20 chars")
print(f"  Sample chunk: {fixed_chunks[0].page_content[:100]}...")

# Strategy 2: Recursive chunking (smarter)
print("\n--- Strategy 2: Recursive Character Splitting ---")
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
recursive_chunks = recursive_splitter.split_documents(documents)
print(f"✓ Created {len(recursive_chunks)} chunks")
print(f"  Tries to split on paragraph/sentence boundaries")
print(f"  Sample chunk: {recursive_chunks[0].page_content[:100]}...")

# Strategy 3: Semantic chunking (embedding-based) - SKIPPED FOR SPEED
print("\n--- Strategy 3: Semantic Chunking (Skipped) ---")
print("  Note: Semantic chunking uses embeddings to find natural break points")
print("  Skipped in this demo to reduce API calls and runtime")
print("  See exercises.md to implement this yourself!")

# Uncomment to enable semantic chunking:
"""
# Initialize OpenAI embeddings for semantic chunker
embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)

semantic_splitter = SemanticChunker(
    embeddings=embeddings_model,
    breakpoint_threshold_type="percentile"
)
semantic_chunks = semantic_splitter.split_documents(documents)
print(f"✓ Created {len(semantic_chunks)} chunks")
"""

# Strategy 4: Markdown structure-aware chunking
print("\n--- Strategy 4: Markdown Header Splitting ---")

# Create sample markdown documentation
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

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)
md_chunks = markdown_splitter.split_text(markdown_doc)
print(f"✓ Created {len(md_chunks)} chunks from markdown")
print(f"  Preserves document structure and header context")
if md_chunks:
    print(f"  Sample chunk with metadata:")
    print(f"    Content: {md_chunks[0].page_content[:80]}...")
    print(f"    Metadata: {md_chunks[0].metadata}")

# Strategy 5: HTML structure-aware chunking
print("\n--- Strategy 5: HTML Header Splitting ---")

# Create sample HTML documentation
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

# Strategy 6: Whole documents (no chunking)
print("\n--- Strategy 6: Whole Documents (No Chunking) ---")
print(f"✓ Using {len(documents)} whole documents")
print(f"  Good for small documents like our tickets")

# ============================================================================
# PART 2: Building FAISS Index from Scratch
# ============================================================================
print("\n" + "="*80)
print("PART 2: Building FAISS Vector Store")
print("="*80)

# Initialize OpenAI client for embeddings
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
embedding_model_name = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
embedding_dim = 1536  # text-embedding-3-small dimension

# Use whole documents for this example
texts = [doc.page_content for doc in documents]
print(f"\nEncoding {len(texts)} documents...")
response = client.embeddings.create(input=texts, model=embedding_model_name)
embeddings = np.array([data.embedding for data in response.data])

# Create FAISS index
print("\nBuilding FAISS index...")
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
index.add(embeddings.astype('float32'))
print(f"✓ FAISS index created with {index.ntotal} vectors")

# Test search
query = "Authentication problems after password reset"
print(f"\nSearching for: '{query}'")
query_response = client.embeddings.create(input=[query], model=embedding_model_name)
query_embedding = np.array([query_response.data[0].embedding], dtype='float32')

k = 3  # Top-3 results
distances, indices = index.search(query_embedding, k)

print(f"\nTop {k} results:")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
    print(f"\n#{i} - Distance: {dist:.4f}")
    print(f"Ticket: {tickets[idx]['ticket_id']}")
    print(f"Title: {tickets[idx]['title']}")

# ============================================================================
# PART 3: Using LangChain FAISS (High-Level API)
# ============================================================================
print("\n" + "="*80)
print("PART 3: LangChain FAISS Integration")
print("="*80)

# Initialize OpenAI embeddings for LangChain
embeddings_model = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)

print("\nBuilding LangChain FAISS vector store...")
faiss_store = LangChainFAISS.from_documents(
    documents=documents,
    embedding=embeddings_model
)
print("✓ FAISS store created")

# Search using similarity search
print(f"\nSearching: '{query}'")
results = faiss_store.similarity_search(query, k=3)

print(f"\nTop {len(results)} results:")
for i, doc in enumerate(results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")
    print(f"Content: {doc.page_content[:150]}...")

# Search with scores
print("\n--- With Similarity Scores ---")
results_with_scores = faiss_store.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"\n#{i} - Score: {score:.4f}")
    print(f"Ticket: {doc.metadata['ticket_id']}")

# ============================================================================
# PART 4: Using Chroma Vector Store
# ============================================================================
print("\n" + "="*80)
print("PART 4: Chroma Vector Store")
print("="*80)

print("\nBuilding Chroma vector store...")
chroma_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    collection_name="support_tickets",
    persist_directory="./chroma_db"
)
print("✓ Chroma store created and persisted")

# Search
print(f"\nSearching in Chroma: '{query}'")
chroma_results = chroma_store.similarity_search(query, k=3)

print(f"\nTop {len(chroma_results)} results:")
for i, doc in enumerate(chroma_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")

# MMR search (Maximal Marginal Relevance - diverse results)
print("\n--- Using MMR for Diverse Results ---")
mmr_results = chroma_store.max_marginal_relevance_search(query, k=3)

print(f"\nMMR Results (more diverse):")
for i, doc in enumerate(mmr_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Title: {tickets[int(doc.metadata['ticket_id'].split('-')[1]) - 1]['title']}")

# ============================================================================
# PART 5: Filtering with Metadata
# ============================================================================
print("\n" + "="*80)
print("PART 5: Metadata Filtering")
print("="*80)

# Search only in "Authentication" category
print("\nSearching only in 'Authentication' category:")
filtered_results = chroma_store.similarity_search(
    query,
    k=3,
    filter={"category": "Authentication"}
)

print(f"\nFiltered results ({len(filtered_results)}):")
for i, doc in enumerate(filtered_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")
    print(f"Content: {doc.page_content[:100]}...")

# Search only high priority tickets
print("\n\nSearching only 'High' priority tickets:")
high_priority_results = chroma_store.similarity_search(
    "Database performance issues",
    k=3,
    filter={"priority": "High"}
)

print(f"\nHigh priority results ({len(high_priority_results)}):")
for i, doc in enumerate(high_priority_results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Priority: {doc.metadata['priority']}")

# ============================================================================
# PART 6: Comparing Chunking Strategies
# ============================================================================
print("\n" + "="*80)
print("PART 6: Evaluating Chunking Strategies")
print("="*80)

# Build stores with different chunking
print("\nBuilding vector stores with different chunking strategies...")

# Store 1: Whole documents
store_whole = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    collection_name="whole_docs"
)

# Store 2: Fixed chunks
store_fixed = Chroma.from_documents(
    documents=fixed_chunks,
    embedding=embeddings_model,
    collection_name="fixed_chunks"
)

# Store 3: Recursive chunks
store_recursive = Chroma.from_documents(
    documents=recursive_chunks,
    embedding=embeddings_model,
    collection_name="recursive_chunks"
)

test_query = "Database connection failures"
print(f"\nTest query: '{test_query}'")

# Compare results
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

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nKey Takeaways:")
print("1. Chunking is crucial for long documents (not critical for short tickets)")
print("2. FAISS provides fast, efficient similarity search")
print("3. Chroma offers persistence and metadata filtering")
print("4. Different chunking strategies affect retrieval quality")
print("5. Always experiment to find the best approach for your data")
print("\nNext: Hour 3 - Building the RAG Pipeline")
