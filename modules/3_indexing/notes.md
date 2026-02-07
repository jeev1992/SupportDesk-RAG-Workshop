# Module 3: Indexing Strategies for RAG

## Introduction

Indexing determines how documents are organized and retrieved in a RAG system. The right indexing strategy dramatically impacts retrieval accuracy, speed, and answer quality. This module explores five core approaches using LlamaIndex.

## Why LlamaIndex for This Module?

**Important:** For this module, we're using **LlamaIndex** instead of LangChain to demonstrate indexing strategies. Here's why:

### Framework Choice Rationale

Throughout this workshop:
- **Modules 1, 2, 4, 5**: Use LangChain + OpenAI (consistent framework)
- **Module 3 (this one)**: Uses LlamaIndex (specialized for indexing)

**Why switch frameworks?**
1. **LlamaIndex is purpose-built for indexing** - Clean abstractions for different index types
2. **Easier to teach** - Simple API for Vector, Summary, Tree, Keyword indexes
3. **Better demonstrations** - Native support for all indexing patterns we want to show

### Integration with LangChain

Don't worry! LlamaIndex and LangChain work together seamlessly:

```python
# Using LlamaIndex indexes in LangChain pipelines
from llama_index.core import VectorStoreIndex
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Build index with LlamaIndex
index = VectorStoreIndex.from_documents(documents)

# Convert to LangChain retriever
retriever = index.as_retriever(similarity_top_k=3)

# Use in LangChain chain
llm = ChatOpenAI(model="gpt-4o-mini")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Query as usual
response = qa_chain({"query": "How do I fix auth issues?"})
```

### Key Takeaway

**Different frameworks, same concepts.** Whether you use LangChain or LlamaIndex in production, understanding indexing strategies remains crucial. The patterns you learn here apply universally.

## Understanding Index vs Retrieval

Before diving into strategies, let's clarify a common confusion:

### Index = Storage Structure
**Index** is how you **organize and store** your embedded documents.

Think of it like a library:
- All books (documents) are stored on shelves
- Embedding vectors are stored **flat** in vector storage (FAISS, Chroma, Pinecone)
- The storage itself is typically just a flat array of vectors

```python
# The actual storage is simple
embeddings = [
    [0.23, 0.45, 0.12, ...],  # Doc 1 vector (1536 dims)
    [0.87, 0.34, 0.91, ...],  # Doc 2 vector (1536 dims)
    [0.56, 0.78, 0.23, ...],  # Doc 3 vector (1536 dims)
]
# Stored flat in vector database
```

### Retrieval = Search Logic
**Retrieval** is how you **find and select** relevant documents at query time.

This is where the intelligence lives:
- Vector similarity search (cosine, L2)
- Hierarchical traversal (tree structures)
- Keyword matching
- Hybrid combinations
- Re-ranking
- Filtering by metadata

```python
# The retrieval logic determines HOW to search
vector_retrieval:
    1. Embed query
    2. Compute similarity with all vectors
    3. Return top-K by similarity score
    
tree_retrieval:
    1. Start at root summary
    2. Identify relevant branches
    3. Traverse down to leaf nodes
    4. Return relevant leaves

hybrid_retrieval:
    1. Vector search → candidates A
    2. Keyword search → candidates B
    3. Combine using RRF → final results
```

### The Key Insight

**Storage is flat, retrieval is smart.**

- All these indexing strategies store embeddings in basically the same way (flat vectors)
- The **magic is in the retrieval logic** - how you traverse, filter, and combine results
- Different "indexes" in this module = different **retrieval strategies**, not different storage formats

```
┌─────────────────────────────────────────────────────┐
│              FLAT VECTOR STORAGE                    │
│  [Doc1: [0.23, 0.45, ...]]                         │
│  [Doc2: [0.87, 0.34, ...]]                         │
│  [Doc3: [0.56, 0.78, ...]]                         │
│  ...                                                │
└─────────────────────────────────────────────────────┘
                        ▲
                        │
        ┌───────────────┴───────────────┐
        │                               │
   Vector Index                    Tree Index
   (Similarity                    (Hierarchical
    Search)                        Traversal)
        │                               │
        ▼                               ▼
   Different retrieval logic, same underlying storage
```

### Practical Example

```python
# Same vectors, different retrieval strategies

# Strategy 1: Vector similarity (flat search)
vector_index = VectorStoreIndex.from_documents(docs)
vector_engine = vector_index.as_query_engine(similarity_top_k=3)
# Retrieval: Direct cosine similarity, return top 3

# Strategy 2: Tree hierarchy (structured traversal)  
tree_index = TreeIndex.from_documents(docs)
tree_engine = tree_index.as_query_engine()
# Retrieval: Navigate tree structure, drill down to leaves

# Strategy 3: Keyword matching (no embeddings needed)
keyword_index = KeywordTableIndex.from_documents(docs)
keyword_engine = keyword_index.as_query_engine()
# Retrieval: Extract keywords, match against document keywords

# Same docs, same vectors (where applicable), 
# but DIFFERENT RETRIEVAL LOGIC yields different results!
```

Now you understand: **Indexing strategies are really retrieval strategies.**

## Why Indexing Strategy Matters

### The Challenge
Not all queries need the same retrieval approach:
- **Specific questions**: "What's the API key for X?" → Need exact chunks
- **Broad questions**: "Tell me about authentication" → Need multiple related chunks
- **Exploratory queries**: "What issues are common?" → Need hierarchical browsing

### The Impact
| Bad Indexing | Good Indexing |
|--------------|---------------|
| Returns irrelevant chunks | Finds the right context |
| Misses related information | Connects concepts |
| Slow retrieval | Fast, scalable search |
| High false positives | Precise results |

## Five Indexing Strategies

### 1. Vector Index (Flat Index)

**Description**: Embed all chunks and retrieve by semantic similarity.

**How It Works:**
```
1. Chunk documents → [chunk1, chunk2, ..., chunkN]
2. Embed each chunk → [emb1, emb2, ..., embN]
3. Store in vector database
4. Query: Embed query → Find closest embeddings
```

**Code:**
```python
from llama_index.core import VectorStoreIndex, Document

# Create documents
docs = [Document(text=text, metadata=metadata) for text in texts]

# Create index
index = VectorStoreIndex.from_documents(docs)

# Query
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("How do I reset my password?")
```

**Pros:**
- ✅ Simple and effective
- ✅ Fast with modern vector DBs
- ✅ Works for 90% of use cases
- ✅ Easy to scale

**Cons:**
- ❌ No hierarchical structure
- ❌ May return fragmented chunks
- ❌ All chunks treated equally

**Best For:**
- General purpose RAG
- Q&A systems
- Semantic search
- Getting started quickly

**When to Use:**
- Default choice for most applications
- You have well-chunked documents
- Queries are specific and focused

**Real-World Use Cases:**
```
✓ Customer support ticket search
  Query: "How do I fix login issues?"
  Finds: "Authentication problems", "SSO failures"

✓ FAQ and documentation search
  Query: "What's the refund policy?"
  Finds: Semantically similar policy documents

✓ Chatbot knowledge base
  Query: "My order hasn't arrived"
  Finds: Shipping delay articles, tracking info

✓ Research paper search
  Query: "Machine learning for fraud detection"
  Finds: Related papers even with different terminology

Scale: Works well up to millions of documents
```

**Parameters:**
```python
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Number of chunks to retrieve
    response_mode="compact",  # How to synthesize answer
    streaming=False  # Enable streaming responses
)
```

### 2. Summary Index

**Description**: Store and search through full documents using LLM evaluation.

#### How It Works

Unlike Vector Index which uses embeddings for similarity search, Summary Index stores full documents and uses the LLM to evaluate relevance:

**Storage (No Chunking, No Embeddings):**
```
[Doc 1: Full ticket text] → Stored as-is
[Doc 2: Full ticket text] → Stored as-is
[Doc 3: Full ticket text] → Stored as-is
...all documents stored without modification
```

**Query Process (Linear Scan):**
```
Query: "How do I fix authentication issues?"

Step 1: LLM reads Doc 1 → "Relevant? Yes"
Step 2: LLM reads Doc 2 → "Relevant? No"
Step 3: LLM reads Doc 3 → "Relevant? Yes"
... continues for ALL documents

Step N: Collect relevant docs → [Doc 1, Doc 3, ...]
Step N+1: Synthesize answer from all relevant docs
```

**Why It's Slow:**
```
Complexity: O(n) - must check every document
Each check = LLM call or LLM attention

10 documents:   ~2 seconds
100 documents:  ~20 seconds
1000 documents: ~200 seconds ⚠️ Too slow!
```

**Response Modes:**
```python
# tree_summarize: Hierarchical summarization
# - Summarizes groups of docs, then summarizes summaries
# - Good for comprehensive answers from multiple sources
query_engine = index.as_query_engine(response_mode="tree_summarize")

# compact: Fits as much as possible in one LLM call
query_engine = index.as_query_engine(response_mode="compact")

# refine: Iteratively refines answer with each doc
query_engine = index.as_query_engine(response_mode="refine")
```

**Code:**
```python
from llama_index.core import SummaryIndex

# Create index (no embeddings, stores full docs)
index = SummaryIndex.from_documents(docs)

# Query with tree_summarize for comprehensive answers
query_engine = index.as_query_engine(
    response_mode="tree_summarize"
)
response = query_engine.query("What are common auth issues?")
```

**Pros:**
- ✅ Returns full documents (no fragmentation)
- ✅ Good for high-level queries
- ✅ No embeddings needed (optional)

**Cons:**
- ❌ Slow for large datasets (O(n) scan)
- ❌ No semantic similarity
- ❌ LLM must process all docs

**Best For:**
- Small document collections (<100 docs)
- High-level summarization
- When you want full documents returned

**When to Use:**
- You have <50 documents
- Documents are naturally scoped (tickets, emails)
- Queries need broad context

**Real-World Use Cases:**
```
✓ Executive report summarization
  Query: "Summarize all customer complaints this week"
  Dataset: 20 complaint emails
  Returns: Comprehensive summary across all docs

✓ Meeting notes analysis
  Query: "What decisions were made about the budget?"
  Dataset: 15 meeting transcripts
  Returns: Full context from relevant meetings

✓ Email thread understanding
  Query: "What was the final agreement?"
  Dataset: 30 emails in a negotiation thread
  Returns: Complete thread context

✓ Small knowledge base Q&A
  Query: "What are our vacation policies?"
  Dataset: 25 HR policy documents
  Returns: Full policy documents, not fragments

Scale: <50 documents (gets slow beyond that)
```

**Performance:**
```
10 documents: ~2 seconds
100 documents: ~20 seconds
1000 documents: ~200 seconds ⚠️ Too slow!
```

### 3. Tree Index (Hierarchical)

**Description**: Build a tree structure from leaf nodes (chunks) to root (summary).

**How It Works:**
```
Root: "Support system overview"
  ├─ Branch: "Authentication issues"
  │   ├─ Leaf: "Password reset error"
  │   └─ Leaf: "SSO login failure"
  └─ Branch: "Performance problems"
      ├─ Leaf: "Database timeout"
      └─ Leaf: "Memory leak"

Query Process:
1. Start at root level
2. Find most relevant branch
3. Drill down to specific leaves
4. Return bottom-up synthesized answer
```

### Tree Index: Detailed Example with Physics Textbook

Let's see how LlamaIndex builds and queries a tree structure using a Physics textbook:

#### Input: Physics Textbook Chapters

```
Chapter 1: Classical Mechanics
  - Section 1.1: Newton's Laws of Motion
    • Page 5: First Law (Inertia)
    • Page 7: Second Law (F=ma)
    • Page 10: Third Law (Action-Reaction)
  - Section 1.2: Energy and Work
    • Page 15: Kinetic Energy
    • Page 18: Potential Energy
    • Page 22: Conservation of Energy

Chapter 2: Electromagnetism
  - Section 2.1: Electric Fields
    • Page 45: Coulomb's Law
    • Page 48: Electric Field Strength
  - Section 2.2: Magnetic Fields
    • Page 55: Magnetic Force
    • Page 58: Electromagnetic Induction
```

#### How LlamaIndex Stores Nodes

LlamaIndex creates a **multi-level hierarchy** with two types of indexes:

**Level 1: Leaf Nodes (Detailed Index)**
```python
# These are the actual document chunks - stored with full content
Leaf_1a = Node(
    text="Newton's First Law states that an object at rest stays at rest, 
          and an object in motion stays in motion with constant velocity 
          unless acted upon by a net external force. This is the principle 
          of inertia...",
    metadata={'page': 5, 'section': '1.1', 'chapter': 1},
    node_id='leaf_1a'
)

Leaf_1b = Node(
    text="Newton's Second Law: F=ma. The net force on an object equals 
          its mass times acceleration. This fundamental equation allows 
          us to predict motion...",
    metadata={'page': 7, 'section': '1.1', 'chapter': 1},
    node_id='leaf_1b'
)

# ... more leaf nodes for each page
```

**Level 2: Branch Nodes (Summary Index - Coarse)**
```python
# LlamaIndex generates summaries of child nodes
Branch_1_1 = Node(
    text="Summary: This section covers Newton's Three Laws of Motion, 
          explaining inertia, force-mass-acceleration relationship, 
          and action-reaction pairs.",
    children=['leaf_1a', 'leaf_1b', 'leaf_1c'],
    metadata={'section': '1.1', 'chapter': 1},
    node_id='branch_1_1'
)

Branch_1_2 = Node(
    text="Summary: This section discusses energy concepts including 
          kinetic energy, potential energy, and the law of conservation 
          of energy.",
    children=['leaf_1d', 'leaf_1e', 'leaf_1f'],
    metadata={'section': '1.2', 'chapter': 1},
    node_id='branch_1_2'
)
```

**Level 3: Chapter Nodes (Higher-Level Summary)**
```python
Chapter_1 = Node(
    text="Summary: Classical Mechanics chapter covering Newton's Laws, 
          force, motion, energy, and work.",
    children=['branch_1_1', 'branch_1_2'],
    metadata={'chapter': 1},
    node_id='chapter_1'
)

Chapter_2 = Node(
    text="Summary: Electromagnetism chapter covering electric and 
          magnetic fields, Coulomb's Law, and electromagnetic induction.",
    children=['branch_2_1', 'branch_2_2'],
    metadata={'chapter': 2},
    node_id='chapter_2'
)
```

**Level 4: Root Node (Top-Level Summary)**
```python
Root = Node(
    text="Summary: Complete physics textbook covering Classical Mechanics 
          and Electromagnetism with fundamental laws and principles.",
    children=['chapter_1', 'chapter_2'],
    node_id='root'
)
```

#### Storage Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    LLAMAINDEX TREE STORAGE                  │
└─────────────────────────────────────────────────────────────┘

Storage Layer:
  • All nodes stored in flat database (docstore)
  • Each node has: text, metadata, children, parent pointers
  • Summaries stored at branch/root level
  • Full content stored at leaf level

Tree Structure (logical, not storage):

                    [ROOT]
               "Physics Textbook"
                      │
        ┌─────────────┴─────────────┐
        │                           │
   [Chapter 1]                 [Chapter 2]
  "Classical Mech"          "Electromagnetism"
        │                           │
    ┌───┴───┐                   ┌───┴───┐
    │       │                   │       │
[Sec 1.1] [Sec 1.2]        [Sec 2.1] [Sec 2.2]
"Newton"  "Energy"          "Electric" "Magnetic"
    │       │                   │         │
  ┌─┼─┐   ┌─┼─┐              ┌─┼─┐     ┌─┼─┐
  L L L   L L L              L L       L L
  (Pages) (Pages)          (Pages)   (Pages)

L = Leaf nodes (detailed content)
Summary Index (Coarse) = Chapter, Section nodes
Detailed Index = Leaf nodes
```

#### Query Resolution: Multiple Matching Nodes

**Query Example:** "Explain Newton's Second Law and electromagnetic induction"

This query matches **two different branches** of the tree.

##### Scenario 1: Matching Nodes on Same Path

```
Query: "Explain Newton's First and Second Laws"

Matching nodes:
  • Branch_1_1 (Newton's Laws) - matches both
  • Leaf_1a (First Law) - matches
  • Leaf_1b (Second Law) - matches

Resolution:
┌─────────────────────────────────────────┐
│   SAME PATH - Traverse Down             │
└─────────────────────────────────────────┘

Step 1: Start at root, find Chapter 1 matches
Step 2: Drill into Branch_1_1 (Newton's Laws)
Step 3: Since multiple leaves match AND on same path:
        → Retrieve ALL matching leaves [1a, 1b]
Step 4: Synthesize answer from both leaves

Result: Comprehensive answer using BOTH leaf nodes
```

##### Scenario 2: Matching Nodes on Different Paths

```
Query: "Explain Newton's Second Law and electromagnetic induction"

Matching nodes:
  • Branch_1_1 (Newton's Laws) - matches "Second Law"
    └─ Leaf_1b (F=ma)
  • Branch_2_2 (Magnetic Fields) - matches "induction"
    └─ Leaf_2d (Electromagnetic Induction)

Resolution:
┌─────────────────────────────────────────┐
│   DIFFERENT PATHS - Evaluate Both       │
└─────────────────────────────────────────┘

Step 1: Start at root
Step 2: LLM evaluates: "Which chapters are relevant?"
        → Chapter 1: Relevance score = 0.8
        → Chapter 2: Relevance score = 0.9
Step 3: Explore BOTH chapters (controlled by child_branch_factor)
Step 4: Chapter 1 path:
        - Drill to Branch_1_1 → Retrieve Leaf_1b
Step 5: Chapter 2 path:
        - Drill to Branch_2_2 → Retrieve Leaf_2d
Step 6: Combine results from BOTH paths
Step 7: LLM synthesizes answer using both leaves

Result: Answer covering BOTH topics from different sections
```

#### LlamaIndex Query Algorithm

```python
def query_tree_index(query, root_node, child_branch_factor=1):
    """
    Traverse tree based on relevance
    
    Args:
        child_branch_factor: How many branches to explore at each level
                           1 = single path (greedy)
                           2+ = multiple paths (thorough)
    """
    current_level = [root_node]
    relevant_leaves = []
    
    while current_level:
        # Evaluate all nodes at current level
        scored_nodes = []
        for node in current_level:
            # LLM judges: "Is this node relevant to query?"
            relevance_score = llm.evaluate_relevance(query, node.text)
            scored_nodes.append((node, relevance_score))
        
        # Sort by relevance
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Select top child_branch_factor nodes
        selected = scored_nodes[:child_branch_factor]
        
        # Prepare next level
        next_level = []
        for node, score in selected:
            if node.is_leaf():
                # Reached bottom - this is detailed content
                relevant_leaves.append(node)
            else:
                # Intermediate node - continue traversing
                next_level.extend(node.children)
        
        current_level = next_level
    
    # Synthesize answer from all relevant leaves
    context = "\n\n".join([leaf.text for leaf in relevant_leaves])
    answer = llm.generate_answer(query, context)
    return answer
```

#### Key Insights

**1. Coarse-to-Fine Retrieval:**
- **Coarse (Summary Index)**: High-level summaries at branch/root
- **Fine (Detailed Index)**: Full content at leaf nodes
- Query starts coarse, drills down to fine details

**2. Same Path Logic:**
```python
if all_matching_nodes_on_same_path:
    # Retrieve all matching leaves on that path
    return [leaf for leaf in path if leaf.matches(query)]
    # Example: "First Law and Second Law" → both from Newton section
```

**3. Different Path Logic:**
```python
if matching_nodes_on_different_paths:
    # Evaluate each path's relevance
    # Explore top K paths (K = child_branch_factor)
    # Collect leaves from multiple paths
    # Synthesize combined answer
    # Example: "Newton's Law and Induction" → two different chapters
```

**4. Efficiency:**
- Don't read all documents (unlike Summary Index)
- Only traverse relevant branches
- Good for 1000+ documents with clear hierarchy

**Code:**
```python
from llama_index.core import TreeIndex

# Build tree
index = TreeIndex.from_documents(
    docs,
    num_children=10,  # Branching factor
    build_tree=True
)

# Query (traverses tree)
query_engine = index.as_query_engine(
    child_branch_factor=2  # How many branches to explore per level
)

# Single path query
response = query_engine.query("What is Newton's Second Law?")
# Traverses: Root → Chapter 1 → Section 1.1 → Leaf (F=ma)

# Multi-path query  
response = query_engine.query("Explain force and electromagnetic induction")
# Traverses: Root → Chapter 1 + Chapter 2 → Multiple sections → Multiple leaves
```

**Pros:**
- ✅ Efficient for large collections
- ✅ Natural hierarchical exploration
- ✅ Good for broad → specific queries
- ✅ Preserves relationships

**Cons:**
- ❌ Complex to build and maintain
- ❌ Requires good document structure
- ❌ Slower query time vs flat index

**Best For:**
- Large document collections (1000+)
- Hierarchical content (manuals, wikis)
- Multi-level queries
- Exploration and discovery

**When to Use:**
- You have 1000+ documents
- Documents have natural hierarchy
- Users ask broad then narrow down
- Browsing is important

**Real-World Use Cases:**
```
✓ Technical manuals and documentation
  Query: "Explain section 4.2 of the compliance policy"
  Structure: Manual → Chapters → Sections → Paragraphs
  Traverses: Root → Compliance Chapter → Section 4 → 4.2

✓ Legal document analysis
  Query: "What are the liability clauses?"
  Structure: Contract → Articles → Clauses → Sub-clauses
  Traverses: Root → Liability Article → Specific clauses

✓ Enterprise wiki search
  Query: "How does the authentication system work?"
  Structure: Wiki → Categories → Pages → Sections
  Traverses: Root → Engineering → Auth System → Details

✓ Textbook Q&A
  Query: "What is Newton's Second Law?"
  Structure: Textbook → Chapters → Sections → Paragraphs
  Traverses: Root → Physics → Mechanics → Newton's Laws

Scale: 1000+ documents with natural hierarchy
```

**Parameters:**
```python
TreeIndex.from_documents(
    docs,
    num_children=10,  # Branching factor (lower = deeper tree)
    build_tree=True,  # Auto-build hierarchy
    use_async=True  # Parallel construction
)
```

### 4. Keyword Table Index

**Description**: Extract keywords from each chunk, match query keywords.

**How It Works:**
```
Document: "Users report authentication failures after password reset"
Keywords: [authentication, failures, password, reset, users]

Query: "auth reset problem"
Keywords: [auth, reset, problem]

Match: auth→authentication ✓, reset✓, problem→failures✓
Score: 3/3 keywords matched
```

#### How Keywords Are Extracted

By default, LlamaIndex uses an **LLM call** to extract keywords:

```python
# LlamaIndex sends each document to the LLM with a prompt like:
"Extract keywords from the following text. Return as comma-separated list:
{document_text}"

# LLM returns: "authentication, password, reset, login, SSO, error"
```

**What gets stored (inverted index):**
```
Keyword → [Document IDs]
─────────────────────────
"password"  → [ticket_1, ticket_5, ticket_12]
"login"     → [ticket_1, ticket_3, ticket_8]
"timeout"   → [ticket_7, ticket_15]
"billing"   → [ticket_20, ticket_25]
```

**At query time:**
1. Extract keywords from query (also via LLM or simple tokenization)
2. Look up matching documents in the keyword table
3. Return documents that share keywords with the query

**Key difference from vector search**: No semantic understanding—"authentication" won't match "login" unless both keywords appear in the same document.

**Code:**
```python
from llama_index.core import KeywordTableIndex

# Build keyword index (uses LLM to extract keywords)
index = KeywordTableIndex.from_documents(docs)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("password authentication")
```

**Pros:**
- ✅ No embeddings needed (faster, cheaper)
- ✅ Good for keyword-specific queries
- ✅ Exact term matching
- ✅ Works offline (no API calls)

**Cons:**
- ❌ No semantic understanding
- ❌ Misses synonyms ("car" != "vehicle")
- ❌ Poor for natural language queries
- ❌ Sensitive to wording

**Best For:**
- Code search (function names, error codes)
- Technical documentation
- ID/ticket number lookup
- Exact term matching

**When to Use:**
- You have structured data with IDs
- Queries use specific terminology
- Want to avoid embedding costs
- Offline operation needed

**Real-World Use Cases:**
```
✓ Ticket/Issue ID lookup
  Query: "TICK-001" or "JIRA-4532"
  Returns: Exact ticket with that ID

✓ Error code search
  Query: "HTTP 503 error" or "NullPointerException"
  Returns: Docs containing those exact error codes

✓ Code search
  Query: "function parseJSON" or "class UserAuthentication"
  Returns: Files containing those function/class names

✓ API endpoint lookup
  Query: "/api/v2/users/profile"
  Returns: Documentation for that exact endpoint

✓ Product SKU search
  Query: "SKU-78432-BLK"
  Returns: Product catalog entry

Scale: Any size, very fast, no LLM needed for retrieval
```

**Example Use Cases:**
```python
# Good queries for keyword index:
"TICK-001"  # Exact ID
"HTTP 503 error"  # Specific error code
"function parse_json"  # Code identifier

# Poor queries for keyword index:
"How do I fix login issues?"  # Natural language
"Problems with authentication"  # Synonyms
```

### 5. Hybrid Retrieval

**Description**: Combine multiple retrieval strategies for best results.

#### Why Hybrid?

Vector search and keyword search have complementary weaknesses:

| Search Type | Catches | Misses |
|-------------|---------|--------|
| **Vector** | Semantic matches ("login" ↔ "authentication") | Exact terms (ticket IDs, error codes) |
| **Keyword** | Exact matches ("TICK-001", "HTTP 503") | Synonyms, related concepts |

**Example:**
```
Query: "authentication timeout error TICK-001"

Vector Search finds:
  - "Login session expired after inactivity" (semantic match)
  - "SSO authentication failures" (semantic match)
  - MISSES "TICK-001" (not semantically similar)

Keyword Search finds:
  - "TICK-001: Auth service error" (exact match)
  - "timeout configuration issue" (keyword match)  
  - MISSES "Login session expired" (no keyword overlap)

Hybrid finds ALL of them → fewer missed results
```

**How It Works:**
```
Query: "authentication timeout"

Vector Search:     Keyword Search:
[Doc A: 0.89]      [Doc C: 3 keywords]
[Doc B: 0.85]      [Doc A: 2 keywords]
[Doc C: 0.75]      [Doc E: 1 keyword]

Fusion (RRF - Reciprocal Rank Fusion):
score(doc) = 1/(rank_vector + k) + 1/(rank_keyword + k)

Final Ranking:
[Doc A: 0.032]  # High in both → most confident
[Doc C: 0.029]  # Good in both
[Doc B: 0.015]  # Vector only
```

**Code:**
```python
# Simple hybrid approach
vector_index = VectorStoreIndex.from_documents(docs)
keyword_index = KeywordTableIndex.from_documents(docs)

# Retrieve from both
vector_nodes = vector_index.as_retriever(similarity_top_k=5).retrieve(query)
keyword_nodes = keyword_index.as_retriever().retrieve(query)

# Combine using RRF
def reciprocal_rank_fusion(results_list, k=60):
    scores = {}
    for results in results_list:
        for rank, node in enumerate(results, 1):
            node_id = node.node_id
            if node_id not in scores:
                scores[node_id] = {'node': node, 'score': 0}
            scores[node_id]['score'] += 1 / (rank + k)
    
    sorted_nodes = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
    return [item['node'] for item in sorted_nodes]

# Get final results
hybrid_results = reciprocal_rank_fusion([vector_nodes, keyword_nodes])
```

**Pros:**
- ✅ Best of both worlds
- ✅ More robust to query variations
- ✅ Higher accuracy overall
- ✅ Reduces false negatives

**Cons:**
- ❌ Slower (runs multiple searches)
- ❌ More complex implementation
- ❌ Requires result fusion logic
- ❌ Higher computational cost

**Best For:**
- Production systems
- High-stakes applications
- Diverse query types
- Maximum accuracy required

**When to Use:**
- Accuracy > speed
- Mixed query types (natural + keywords)
- Users vary in query style
- Worth the extra cost

**Real-World Use Cases:**
```
✓ Production customer support
  Query: "auth error TICK-001"
  Vector finds: Related auth troubleshooting articles
  Keyword finds: Exact ticket TICK-001
  Combined: Both the specific ticket AND related solutions

✓ Enterprise search
  Query: "budget meeting Q3 2024"
  Vector finds: Semantically related budget discussions
  Keyword finds: Docs with exact "Q3 2024" mention
  Combined: Comprehensive results across both

✓ E-commerce product search
  Query: "comfortable running shoes size 10"
  Vector finds: Semantically similar athletic footwear
  Keyword finds: Products with exact "size 10" in specs
  Combined: Relevant products in the right size

✓ Medical/Legal search (high stakes)
  Query: "diabetes medication interactions"
  Vector finds: Related drug interaction articles
  Keyword finds: Exact drug names mentioned
  Combined: Can't afford to miss relevant results

Scale: Any size, but 2x retrieval cost
```

**Fusion Strategies:**

**1. Reciprocal Rank Fusion (RRF)** - Recommended
```python
score = 1/(rank + k) + 1/(rank + k)
# k=60 is a good default
```

**2. Weighted Sum**
```python
score = 0.7 * vector_score + 0.3 * keyword_score
# Tune weights based on your data
```

**3. Cascade**
```python
# Use keyword first, then vector for ties
if keyword_results.has_high_match():
    return keyword_results
else:
    return vector_results
```

## Comparison Matrix

| Strategy | Speed | Accuracy | Scale | Semantic | Exact Match | Cost |
|----------|-------|----------|-------|----------|-------------|------|
| Vector | Fast | High | Excellent | ✅ | ❌ | Medium |
| Summary | Slow | Medium | Poor | Partial | ❌ | Low |
| Tree | Medium | High | Excellent | ✅ | ❌ | Medium |
| Keyword | Fast | Medium | Good | ❌ | ✅ | Low |
| Hybrid | Slow | Highest | Good | ✅ | ✅ | High |

## Choosing the Right Strategy

### Decision Tree

```
START
  ↓
Do you have < 100 documents?
  ├─ YES → Summary Index
  └─ NO → Continue
       ↓
Do you need exact keyword matching?
  ├─ YES → Keyword or Hybrid
  └─ NO → Continue
       ↓
Do you have 1000+ documents with hierarchy?
  ├─ YES → Tree Index
  └─ NO → Continue
       ↓
Is maximum accuracy critical?
  ├─ YES → Hybrid (Vector + Keyword)
  └─ NO → Vector Index (default)
```

### Use Case Mapping

| Use Case | Recommended Strategy | Why |
|----------|---------------------|-----|
| Customer Support | Hybrid | Mixed natural + ticket IDs |
| Documentation | Vector or Tree | Semantic understanding needed |
| Code Search | Keyword + Vector | Exact identifiers + concepts |
| Legal/Compliance | Tree | Hierarchical structure |
| Quick Prototype | Vector | Simple, effective |
| FAQ | Summary | Small dataset, full answers |

## Advanced Topics

### 1. Multi-Index Architecture

Use different indexes for different content types:

```python
class MultiIndexRAG:
    def __init__(self):
        self.ticket_index = VectorStoreIndex(...)  # Recent tickets
        self.kb_index = TreeIndex(...)  # Knowledge base
        self.code_index = KeywordTableIndex(...)  # Code docs
    
    def query(self, query):
        # Route based on query type
        if looks_like_ticket_id(query):
            return self.ticket_index.query(query)
        elif is_code_query(query):
            return self.code_index.query(query)
        else:
            # Hybrid across all
            return self.hybrid_query(query)
```

### 2. Dynamic Index Selection

```python
def select_index(query, user_context):
    if user_context.is_admin:
        # Admins get hierarchical browsing
        return tree_index
    elif len(query.split()) < 3:
        # Short queries use keywords
        return keyword_index
    else:
        # Default semantic
        return vector_index
```

### 3. Index Composition

```python
from llama_index.core import ComposableGraph

# Build specialized indexes
auth_index = VectorStoreIndex(auth_docs)
perf_index = VectorStoreIndex(perf_docs)
api_index = KeywordTableIndex(api_docs)

# Compose into graph
graph = ComposableGraph.from_indices(
    TreeIndex,
    [auth_index, perf_index, api_index],
    index_summaries=[
        "Authentication and login issues",
        "Performance and latency problems",
        "API reference and codes"
    ]
)

# Query routes automatically
response = graph.query("How to fix slow API?")
# Routes to: TreeIndex → perf_index OR api_index
```

### 4. Metadata Filtering

Add filters to narrow search:

```python
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# Create index with metadata
docs = [
    Document(text=text, metadata={
        "category": "authentication",
        "priority": "high",
        "date": "2024-01-15"
    })
]

# Query with filters
filters = MetadataFilters(filters=[
    ExactMatchFilter(key="category", value="authentication"),
    ExactMatchFilter(key="priority", value="high")
])

query_engine = index.as_query_engine(filters=filters)
```

## Performance Optimization

### Index Construction

```python
# Parallel document processing
index = VectorStoreIndex.from_documents(
    docs,
    show_progress=True,  # Progress bar
    use_async=True  # Parallel embedding
)

# Batch embedding
from llama_index.core import ServiceContext

service_context = ServiceContext.from_defaults(
    embed_batch_size=100  # Embed 100 docs at once
)
index = VectorStoreIndex.from_documents(docs, service_context=service_context)
```

### Query Optimization

```python
# Cache query results
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_query(query_text):
    return query_engine.query(query_text)

# Adjust retrieval parameters
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Fewer chunks = faster
    response_mode="compact"  # Less LLM processing
)
```

### Storage Optimization

```python
# Persist index to disk
index.storage_context.persist(persist_dir="./storage")

# Load from disk (much faster)
from llama_index.core import load_index_from_storage, StorageContext

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

## Evaluation

### Retrieval Quality

```python
def evaluate_index(index, test_queries):
    metrics = {
        'precision': [],
        'recall': [],
        'latency': []
    }
    
    for query, relevant_ids in test_queries:
        start = time.time()
        results = index.as_retriever(similarity_top_k=5).retrieve(query)
        latency = time.time() - start
        
        retrieved_ids = [r.node.metadata['id'] for r in results]
        
        # Precision@K
        relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
        precision = relevant_retrieved / len(retrieved_ids)
        
        # Recall
        recall = relevant_retrieved / len(relevant_ids)
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['latency'].append(latency)
    
    return {
        'avg_precision': np.mean(metrics['precision']),
        'avg_recall': np.mean(metrics['recall']),
        'avg_latency': np.mean(metrics['latency'])
    }
```

### A/B Testing

```python
# Compare strategies
strategies = {
    'vector': VectorStoreIndex.from_documents(docs),
    'tree': TreeIndex.from_documents(docs),
    'hybrid': create_hybrid_index(docs)
}

for name, index in strategies.items():
    metrics = evaluate_index(index, test_queries)
    print(f"{name}: P={metrics['avg_precision']:.3f}, "
          f"R={metrics['avg_recall']:.3f}, "
          f"Latency={metrics['avg_latency']:.2f}s")
```

## Best Practices

### 1. Start Simple
```python
# Begin with vector index
index = VectorStoreIndex.from_documents(docs)

# If not working well, try hybrid
if precision < 0.7:
    index = create_hybrid_index(docs)
```

### 2. Match Index to Data
- **Structured docs** (manuals) → Tree Index
- **Unstructured text** (articles) → Vector Index
- **Mixed content** → Hybrid

### 3. Tune Parameters
```python
# Retrieval count
similarity_top_k = 3  # Start here
# If too narrow → increase to 5-7
# If too broad → decrease to 1-2

# Response mode
response_mode = "compact"  # Fastest
response_mode = "tree_summarize"  # Best quality
response_mode = "simple_summarize"  # Middle ground
```

### 4. Monitor Performance
```python
import logging

logging.basicConfig(level=logging.INFO)

# LlamaIndex logs query execution
query_engine.query("test")
# Logs: Retrieved 3 nodes in 0.15s
```

## Common Pitfalls

### 1. Wrong Index for Use Case
❌ Using Summary Index for 1000+ documents (too slow)
✅ Use Tree or Vector Index for scale

### 2. No Metadata
❌ Storing raw text only
✅ Add metadata for filtering and debugging

### 3. Over-Engineering
❌ Starting with complex hybrid system
✅ Begin with Vector Index, optimize as needed

### 4. Ignoring Evaluation
❌ Deploying without testing
✅ Create test queries, measure accuracy

## References

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Index Guide](https://docs.llamaindex.ai/en/stable/module_guides/indexing/)
- [RAG Survey Paper](https://arxiv.org/abs/2312.10997)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

## Integrating LlamaIndex with LangChain

Since we're using LlamaIndex for indexing but LangChain elsewhere, here's how to integrate them seamlessly:

### Basic Integration

```python
from llama_index.core import VectorStoreIndex, Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Step 1: Build index with LlamaIndex
docs = [Document(text=text, metadata=meta) for text, meta in doc_data]
index = VectorStoreIndex.from_documents(docs)

# Step 2: Convert to LangChain retriever
retriever = index.as_retriever(
    similarity_top_k=3,
    vector_store_query_mode="default"
)

# Step 3: Use in LangChain chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Step 4: Query
result = qa_chain({"query": "How to fix authentication issues?"})
print(result['result'])
print(f"Sources: {[doc.metadata for doc in result['source_documents']]}")
```

### Advanced Integration: Custom Retriever

```python
from langchain.schema import BaseRetriever, Document as LangChainDoc
from typing import List

class LlamaIndexRetriever(BaseRetriever):
    """Wrap LlamaIndex as LangChain retriever"""
    
    def __init__(self, index, similarity_top_k=3):
        self.index = index
        self.retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    
    def get_relevant_documents(self, query: str) -> List[LangChainDoc]:
        """Retrieve documents using LlamaIndex"""
        nodes = self.retriever.retrieve(query)
        
        # Convert LlamaIndex nodes to LangChain documents
        docs = []
        for node in nodes:
            doc = LangChainDoc(
                page_content=node.text,
                metadata={
                    'score': node.score,
                    'node_id': node.node_id,
                    **node.metadata
                }
            )
            docs.append(doc)
        
        return docs
    
    async def aget_relevant_documents(self, query: str) -> List[LangChainDoc]:
        # Async version
        return self.get_relevant_documents(query)

# Usage
vector_index = VectorStoreIndex.from_documents(docs)
retriever = LlamaIndexRetriever(vector_index, similarity_top_k=5)

# Use in any LangChain chain
from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=retriever,
    return_source_documents=True
)
```

### Hybrid: LlamaIndex Indexing + LangChain Chains

```python
from llama_index.core import VectorStoreIndex, TreeIndex
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Build multiple indexes with LlamaIndex
vector_idx = VectorStoreIndex.from_documents(docs)
tree_idx = TreeIndex.from_documents(docs)

# Convert both to retrievers
vector_retriever = vector_idx.as_retriever(similarity_top_k=3)
tree_retriever = tree_idx.as_retriever()

# Custom LangChain chain that uses both
def hybrid_retrieve(query: str):
    # Get from both indexes
    vector_nodes = vector_retriever.retrieve(query)
    tree_nodes = tree_retriever.retrieve(query)
    
    # Combine and deduplicate
    all_nodes = {node.node_id: node for node in vector_nodes + tree_nodes}
    
    # Convert to LangChain format
    return [
        LangChainDoc(page_content=node.text, metadata=node.metadata)
        for node in all_nodes.values()
    ]

# Use in LangChain template
template = """Answer based on these documents:

{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
llm = ChatOpenAI(model="gpt-4o-mini")
chain = LLMChain(llm=llm, prompt=prompt)

# Query
query = "How to fix authentication?"
docs = hybrid_retrieve(query)
context = "\n\n".join([doc.page_content for doc in docs])
result = chain.run(context=context, question=query)
```

### Using LlamaIndex Query Engine in LangChain Tools

```python
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# Create LlamaIndex query engines
vector_idx = VectorStoreIndex.from_documents(docs)
summary_idx = SummaryIndex.from_documents(docs)

vector_engine = vector_idx.as_query_engine(similarity_top_k=3)
summary_engine = summary_idx.as_query_engine()

# Wrap as LangChain tools
tools = [
    Tool(
        name="Vector Search",
        func=lambda q: str(vector_engine.query(q)),
        description="Search for specific information using semantic similarity. "
                    "Use for targeted questions like 'How to reset password?'"
    ),
    Tool(
        name="Summary Search",
        func=lambda q: str(summary_engine.query(q)),
        description="Get high-level summary across all documents. "
                    "Use for broad questions like 'What are common issues?'"
    )
]

# Create agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Agent will choose the right index strategy
response = agent.run("What's the most common authentication issue?")
# Agent chooses "Summary Search"

response = agent.run("How do I fix ticket TICK-001?")
# Agent chooses "Vector Search"
```

### Sharing Vector Stores Between LlamaIndex and LangChain

```python
from langchain_community.vectorstores import Chroma
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain_openai import OpenAIEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

# Initialize shared Chroma DB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("documents")

# Option 1: Build with LlamaIndex, use in LangChain
llama_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=llama_vector_store)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

# Now use the same Chroma DB in LangChain
langchain_vectorstore = Chroma(
    client=chroma_client,
    collection_name="documents",
    embedding_function=OpenAIEmbeddings()
)
retriever = langchain_vectorstore.as_retriever()

# Option 2: Build with LangChain, use in LlamaIndex
langchain_vectorstore = Chroma.from_documents(
    documents=langchain_docs,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Access same DB from LlamaIndex
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_collection("langchain")
llama_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(llama_vector_store)
```

### Best Practices for Integration

1. **Choose per use case:**
   - Use **LlamaIndex** for: Indexing, diverse retrieval strategies, complex query engines
   - Use **LangChain** for: Chains, agents, prompt templates, integrations

2. **Share vector stores:**
   - Build index once with either framework
   - Access from both frameworks as needed
   - Saves time and API costs

3. **Convert retrievers:**
   - LlamaIndex → LangChain: Use `index.as_retriever()`
   - Wrap in `BaseRetriever` for full control

4. **Leverage strengths:**
   - LlamaIndex: Better for advanced indexing (Tree, Hybrid, etc.)
   - LangChain: Better for orchestration (agents, chains, memory)

### Complete Example: Production RAG System

```python
# indexing.py - Use LlamaIndex for indexing
from llama_index.core import VectorStoreIndex, Document
import os

def build_index(documents):
    """Build and persist index with LlamaIndex"""
    docs = [Document(text=d['text'], metadata=d['metadata']) for d in documents]
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir="./storage")
    return index

# rag_chain.py - Use LangChain for RAG pipeline
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from llama_index.core import load_index_from_storage, StorageContext

def create_rag_chain():
    """Load LlamaIndex and create LangChain pipeline"""
    # Load LlamaIndex index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    
    # Convert to LangChain retriever
    retriever = index.as_retriever(similarity_top_k=3)
    
    # Build LangChain chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": CUSTOM_PROMPT  # Your custom prompt
        }
    )
    
    return qa_chain

# main.py - Use both
index = build_index(my_documents)  # LlamaIndex indexing
qa = create_rag_chain()  # LangChain pipeline
result = qa({"query": "How do I fix auth issues?"})
```

### Key Takeaways

1. **LlamaIndex and LangChain are complementary**, not competitive
2. **Easy integration** via retrievers and vector stores
3. **Choose the right tool** for each part of your pipeline
4. **This workshop**: LlamaIndex for Module 3 indexing, LangChain for everything else
5. **Production**: Mix and match based on your needs

You now have the knowledge to use both frameworks together effectively!

## Next Steps

Now that you understand indexing strategies and how to integrate LlamaIndex with LangChain, proceed to **Module 4: RAG Pipeline** to learn how to combine retrieval with generation for complete question-answering systems.
