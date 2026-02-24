# -*- coding: utf-8 -*-
"""
Hour 2 Solutions: Chunking & Vector Stores
==========================================

Solutions for all exercises in exercises.md
Run each section independently or the whole file.
"""

import json
import os
import time
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Load data
print("Loading data...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)

# Create documents
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

print(f"✓ Loaded {len(documents)} documents\n")


# ============================================================================
# Exercise 1: Change the Chunk Size (Easy)
# ============================================================================
print("=" * 80)
print("EXERCISE 1: Change the Chunk Size")
print("=" * 80)

# Original: chunk_size=200, chunk_overlap=20
# Solution: chunk_size=500, chunk_overlap=50

original_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
original_chunks = original_splitter.split_documents(documents)

new_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
new_chunks = new_splitter.split_documents(documents)

print(f"\nOriginal (200 chars): {len(original_chunks)} chunks")
print(f"New (500 chars): {len(new_chunks)} chunks")
print(f"\n→ Larger chunk size = fewer chunks")


# ============================================================================
# Exercise 2: Change the Search Query (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Change the Search Query")
print("=" * 80)

# Build a simple store for testing
store = Chroma.from_documents(documents, embeddings, collection_name="exercise2")

queries = [
    "Database is timing out frequently",
    "Email notifications not working",
    "Payment processing fails"
]

for query in queries:
    results = store.similarity_search(query, k=1)
    if results:
        print(f"\nQuery: '{query}'")
        print(f"  Best match: {results[0].metadata['ticket_id']}")
        print(f"  Category: {results[0].metadata['category']}")


# ============================================================================
# Exercise 3: Adjust Number of Results (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Adjust Number of Results")
print("=" * 80)

query = "Authentication problems"

# k=3 (original)
results_3 = store.similarity_search(query, k=3)
print("\nTop 3 results:")
for i, doc in enumerate(results_3, 1):
    print(f"  {i}. {doc.metadata['ticket_id']}")

# k=5 (changed)
results_5 = store.similarity_search(query, k=5)
print("\nTop 5 results:")
for i, doc in enumerate(results_5, 1):
    print(f"  {i}. {doc.metadata['ticket_id']}")


# ============================================================================
# Exercise 4: Try Different Metadata Filters (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: Try Different Metadata Filters")
print("=" * 80)

# Chroma supports filtering
chroma_store = Chroma.from_documents(
    documents, 
    embeddings, 
    collection_name="filter_test"
)

categories = ["Authentication", "Database", "Performance", "Email"]
query = "system not working"

for category in categories:
    try:
        results = chroma_store.similarity_search(
            query, k=2, filter={"category": category}
        )
        print(f"\nCategory: {category}")
        for doc in results:
            print(f"  {doc.metadata['ticket_id']}: {doc.page_content[:50]}...")
    except Exception as e:
        print(f"\nCategory: {category} - No matches or error")


# ============================================================================
# Exercise 5: Compare Chunk Sizes (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 5: Compare Chunk Sizes")
print("=" * 80)

chunk_sizes = [100, 200, 300, 500, 1000]

print(f"\nTotal documents: {len(documents)}")
print(f"Avg document length: {sum(len(d.page_content) for d in documents) // len(documents)} chars")
print()
print("Chunk Size | # Chunks | Avg Chunk Length")
print("-" * 45)

for size in chunk_sizes:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=size // 10
    )
    chunks = splitter.split_documents(documents)
    avg_len = sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0
    print(f"{size:>10} | {len(chunks):>8} | {avg_len:>16}")


# ============================================================================
# Exercise 6: Add Similarity Scores to Results (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Add Similarity Scores to Results")
print("=" * 80)

query = "login authentication error"
results_with_scores = store.similarity_search_with_score(query, k=5)

print(f"\nQuery: '{query}'")
print("\nResults with distance scores:")
print("-" * 50)
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"#{i} - Distance: {score:.4f}")
    print(f"   Ticket: {doc.metadata['ticket_id']}")
    print(f"   Category: {doc.metadata['category']}")

print("\n→ Lower score = more similar (Chroma uses L2 distance)")


# ============================================================================
# Exercise 7: Filter by Multiple Conditions (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Filter by Multiple Conditions")
print("=" * 80)

query = "system not working"

# Combined filter: category AND priority
combined_filter = {"$and": [{"category": "Authentication"}, {"priority": "High"}]}

try:
    results = chroma_store.similarity_search(query, k=3, filter=combined_filter)
    
    print(f"\nQuery: '{query}'")
    print(f"Filter: Authentication category + High priority")
    print(f"\nResults ({len(results)}):")
    for doc in results:
        print(f"  [{doc.metadata['priority']}] [{doc.metadata['category']}] {doc.metadata['ticket_id']}")
except Exception as e:
    print(f"No results matching filter or error: {e}")


# ============================================================================
# Exercise 8: Save and Load Vector Store (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 8: Save and Load Vector Store")
print("=" * 80)

# Build documents for Chroma
simple_docs = [
    Document(
        page_content=f"{t['title']}. {t['description']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category']}
    )
    for t in tickets
]

# Step 1: Build and save (Chroma persists automatically with persist_directory)
print("\nBuilding vector store...")
persist_dir = "./solution_chroma_db"
chroma_store = Chroma.from_documents(
    simple_docs, 
    embeddings, 
    collection_name="exercise8",
    persist_directory=persist_dir
)
print(f"✓ Saved to {persist_dir}")

# Step 2: Load it back (create new Chroma instance pointing to same directory)
print("\nLoading vector store...")
loaded_store = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings,
    collection_name="exercise8"
)
print("✓ Loaded from disk")

# Step 3: Verify it works
query = "login problem"
results = loaded_store.similarity_search(query, k=3)
print(f"\nSearch results for '{query}':")
for doc in results:
    print(f"  {doc.metadata['ticket_id']}: {doc.page_content[:50]}...")


# ============================================================================
# Cleanup
# ============================================================================
print("\n" + "=" * 80)
print("CLEANUP")
print("=" * 80)

import shutil
if os.path.exists("./solution_chroma_db"):
    # On Windows, Chroma files can stay locked briefly after queries.
    # Release references and retry cleanup once.
    try:
        del loaded_store
    except Exception:
        pass
    try:
        del chroma_store
    except Exception:
        pass

    cleaned = False
    for _ in range(2):
        try:
            shutil.rmtree("./solution_chroma_db")
            cleaned = True
            break
        except PermissionError:
            time.sleep(1)

    if cleaned:
        print("✓ Cleaned up solution_chroma_db")
    else:
        print("⚠ Could not fully clean solution_chroma_db (file lock). You can delete it manually later.")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL SOLUTIONS COMPLETE!")
print("=" * 80)
print("""
Key Takeaways:
──────────────
1. Larger chunk size = fewer chunks, but may lose precision
2. Different queries find different relevant documents
3. Metadata filtering narrows search to specific categories/priorities
4. Chroma returns distance scores (lower = more similar)
5. Persisting vector stores saves embedding computation time

Next: Move on to Module 3 - Indexing Strategies!
""")
