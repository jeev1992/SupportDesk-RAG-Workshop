"""
Hour 2: Chunking & Vector Stores Demo
======================================

This demo teaches:
1. Different chunking strategies for long documents
2. Building a FAISS vector store
3. Using Chroma for high-level abstraction
4. Comparing retrieval quality across strategies
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS as LangChainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

print("="*80)
print("HOUR 2: CHUNKING & VECTOR STORES")
print("="*80)

# Load tickets
with open('../data/synthetic_tickets.json', 'r') as f:
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

# Strategy 3: Whole documents (no chunking)
print("\n--- Strategy 3: Whole Documents (No Chunking) ---")
print(f"✓ Using {len(documents)} whole documents")
print(f"  Good for small documents like our tickets")

# ============================================================================
# PART 2: Building FAISS Index from Scratch
# ============================================================================
print("\n" + "="*80)
print("PART 2: Building FAISS Vector Store")
print("="*80)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = 384

# Use whole documents for this example
texts = [doc.page_content for doc in documents]
print(f"\nEncoding {len(texts)} documents...")
embeddings = model.encode(texts, show_progress_bar=True)

# Create FAISS index
print("\nBuilding FAISS index...")
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
index.add(embeddings.astype('float32'))
print(f"✓ FAISS index created with {index.ntotal} vectors")

# Test search
query = "Authentication problems after password reset"
print(f"\nSearching for: '{query}'")
query_embedding = model.encode([query]).astype('float32')

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

# Initialize embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
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
