# Hour 2 Exercises: Chunking & Vector Stores

> ✅ **Exercise style for this workshop:** keep each solution to a **small edit** (usually 3–15 lines) in existing files.

## Exercise 1: Change the Chunk Size (Easy)

**Task**: Modify the demo to use larger chunks.

**In the demo code, find this line** (around line 87):
```python
fixed_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator="\n"
)
```

**Change it to**:
```python
fixed_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)
```

**Run the demo and observe**:
- How many chunks are created now vs before?
- Are the chunks more meaningful with larger size?

---

## Exercise 2: Change the Search Query (Easy)

**Task**: Test the vector store with a different query.

**In the demo code, find this line** (around line 253):
```python
query = "Authentication problems after password reset"
```

**Change it to**:
```python
query = "Database is timing out frequently"
```

**Run the demo and answer**:
- Do the results match the new query?
- What category do the matching tickets belong to?

**Try these queries too**:
- `"Email notifications not working"`
- `"Payment processing fails"`
- `"Mobile app crashes"`

---

## Exercise 3: Adjust Number of Results (Easy)

**Task**: Get more search results by changing k.

**In the demo code, find this line** (around line 260):
```python
k = 3  # Top-3 results
```

**Change it to**:
```python
k = 5  # Top-5 results
```

**Run and observe**: How do results #4 and #5 compare to the top 3?

---

## Exercise 4: Try Different Metadata Filters (Easy)

**Task**: Search only in a specific category.

**In the demo code, find this line** (around line 355):
```python
filtered_results = chroma_store.similarity_search(
    query,
    k=3,
    filter={"category": "Authentication"}
)
```

**Change the filter to search different categories**:
```python
# Try each of these:
filter={"category": "Database"}
filter={"category": "Performance"}
filter={"category": "Email"}
```

**Run and observe**: Does filtering narrow results appropriately?

---

## Exercise 5: Compare Chunk Sizes (Medium)

**Task**: See how chunk size affects chunk counts with a tiny loop.

In existing code, add a short loop over `[100, 200, 300, 500]` and print chunk counts.
Reference snippet:
```python
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load data
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)

# Create documents
documents = []
for ticket in tickets:
    full_text = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
    """.strip()
    documents.append(Document(page_content=full_text))

print(f"Total documents: {len(documents)}")
print(f"Avg document length: {sum(len(d.page_content) for d in documents) // len(documents)} chars")
print()

# Compare chunk sizes
chunk_sizes = [100, 200, 300, 500, 1000]

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
```

**Answer**: 
- Which chunk size creates the most chunks?
- At what size do you get roughly 1 chunk per ticket?

---

## Exercise 6: Add Similarity Scores to Results (Medium)

**Task**: Display similarity scores alongside search results.

**In the demo code, find the basic search**:
```python
results = chroma_store.similarity_search(query, k=3)

print(f"\nTop {len(results)} results:")
for i, doc in enumerate(results, 1):
    print(f"\n#{i}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
```

**Replace with this version that shows scores**:
```python
# Use similarity_search_with_score instead
results_with_scores = chroma_store.similarity_search_with_score(query, k=3)

print(f"\nTop {len(results_with_scores)} results:")
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"\n#{i} - Distance: {score:.4f}")
    print(f"Ticket: {doc.metadata['ticket_id']}")
    print(f"Category: {doc.metadata['category']}")
```

**Note**: Lower distance = more similar (Chroma uses L2 distance by default)

---

## Exercise 7: Filter by Multiple Conditions (Medium)

**Task**: Search with combined filters (category AND priority) using one filter line.

Use this filter line in your current search block:

```python
combined_filter = {"$and": [{"category": "Authentication"}, {"priority": "High"}]}
```

Reference full snippet:
```python
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Load and create documents
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)

documents = []
for ticket in tickets:
    doc = Document(
        page_content=f"{ticket['title']}. {ticket['description']}",
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority']
        }
    )
    documents.append(doc)

# Create vector store
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
store = Chroma.from_documents(documents, embeddings, collection_name="exercise7")

# Search with combined filter
query = "system not working"

# Chroma uses $and for multiple conditions
combined_filter = {"$and": [{"category": "Authentication"}, {"priority": "High"}]}

results = store.similarity_search(query, k=3, filter=combined_filter)

print(f"Query: '{query}'")
print(f"Filter: High priority + Authentication category")
print(f"\nResults ({len(results)}):")
for doc in results:
    print(f"  [{doc.metadata['priority']}] [{doc.metadata['category']}] {doc.metadata['ticket_id']}")
```

**Try changing the filter to**:
- Database + Critical priority
- Performance + Medium priority

---

## Exercise 8: Save and Load Vector Store (Medium)

**Task**: Persist a vector store and reload it with minimal edits.

Use your existing `Chroma.from_documents(...)` block and only add:
1. `persist_directory=...` on creation
2. one reload block with same collection name
3. one query to verify it works

Reference snippet:
```python
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Load data
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)

documents = [
    Document(
        page_content=f"{t['title']}. {t['description']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category']}
    )
    for t in tickets
]

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Step 1: Build and save (Chroma persists with persist_directory)
print("Building vector store...")
store = Chroma.from_documents(
    documents, 
    embeddings, 
    collection_name="my_collection",
    persist_directory="./my_chroma_db"
)
print("✓ Saved to ./my_chroma_db")

# Step 2: Load it back
print("\nLoading vector store...")
loaded_store = Chroma(
    persist_directory="./my_chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection"
)
print("✓ Loaded from disk")

# Step 3: Verify it works
query = "login problem"
results = loaded_store.similarity_search(query, k=3)
print(f"\nSearch results for '{query}':")
for doc in results:
    print(f"  {doc.metadata['ticket_id']}: {doc.page_content[:50]}...")
```

**Why this matters**: In production, you don't rebuild embeddings every time!

---

## Bonus Exercise: Semantic vs Fixed Chunking (Challenge)

**Task**: Compare how semantic chunking differs from fixed-size chunking.

**Note**: This is optional and takes longer to run.

Small-edit option: run comparison on one short sample text and only print number of chunks.

```python
import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Load a longer document for better comparison
long_text = """
User Authentication System Overview

The authentication system handles user login and session management. Users can 
authenticate using username/password or single sign-on (SSO). The system supports
OAuth 2.0 and SAML protocols for enterprise integration.

Password Security

All passwords are hashed using bcrypt with a cost factor of 12. Password policies
require a minimum of 8 characters with at least one uppercase, lowercase, number,
and special character. Passwords expire after 90 days for compliance.

Session Management

User sessions are stored in Redis with a 24-hour expiration. Each session includes
the user ID, roles, and creation timestamp. Sessions can be invalidated through
the admin panel or via API.

Two-Factor Authentication

Users can enable 2FA using TOTP (Google Authenticator) or SMS verification. 
Backup codes are generated for account recovery. Enterprise accounts require
2FA for all users.
"""

doc = Document(page_content=long_text)

# Fixed chunking
fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
fixed_chunks = fixed_splitter.split_documents([doc])

# Semantic chunking
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
semantic_chunks = semantic_splitter.split_documents([doc])

print("Fixed Chunking Results:")
print(f"Number of chunks: {len(fixed_chunks)}")
for i, chunk in enumerate(fixed_chunks, 1):
    print(f"\nChunk {i} ({len(chunk.page_content)} chars):")
    print(f"  {chunk.page_content[:80]}...")

print("\n" + "="*60)
print("\nSemantic Chunking Results:")
print(f"Number of chunks: {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks, 1):
    print(f"\nChunk {i} ({len(chunk.page_content)} chars):")
    print(f"  {chunk.page_content[:80]}...")
```

---

## Quick Reference

### Chunking
```python
# Fixed size
from langchain_text_splitters import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# Recursive (smarter)
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# Apply to documents
chunks = splitter.split_documents(documents)
```

### Vector Stores
```python
# Chroma (recommended)
from langchain_community.vectorstores import Chroma

# Create with persistence
store = Chroma.from_documents(
    documents, 
    embeddings, 
    persist_directory="./chroma_db"
)

# Load existing store
store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

### Searching
```python
# Basic search
results = store.similarity_search(query, k=3)

# With scores
results = store.similarity_search_with_score(query, k=3)

# With filters (Chroma)
results = store.similarity_search(query, k=3, filter={"category": "Authentication"})

# MMR (diverse results)
results = store.max_marginal_relevance_search(query, k=3)
```

---

## Next Steps

Ready for **Hour 3: Indexing Strategies**? We'll explore different ways to organize and retrieve documents!

---

**Questions?** Ask the instructor or refer back to the demo code!
