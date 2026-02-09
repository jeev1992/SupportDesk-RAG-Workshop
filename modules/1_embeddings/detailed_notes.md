# Module 1: Text Embeddings - Instructor Notes

---

## 1. Understanding Text Embeddings (Opening Slide)

### Why Vector Databases Matter

> **Key Point to Convey:** Generative AI and Large Language Models (LLMs) have become increasingly popular, and one of the most effective ways to manage LLM data is through a vector database.

**Talking Points:**
- Vector databases offer the ideal infrastructure to **store, search, and retrieve** the complex, high-dimensional data that powers LLMs
- This is the foundation of RAG (Retrieval-Augmented Generation) systems
- Without vector databases, LLMs would be limited to their training data cutoff

**Diagram Explanation (Flow):**
```
Content ──┐
          ├──► Embedding Model ──► Vector Embedding ──► Vector Database
Query ────┘                        [0.34, -1.2, 0.34, 1.3, ..., -0.05, 1.142]
                                                              │
                    Query Result ◄────────────────────────────┘
```

---

## 2. Storing Embeddings in Vector Databases

### Multi-Modal Embedding Support

> **Key Point:** Vector databases can store embeddings from multiple modalities - not just text!

| Modality | Embedding Model | Stored As | Search Type |
|----------|-----------------|-----------|-------------|
| **Document/Text** | Text embedding model | Text vector embeddings | Similarity search |
| **Image** | Image embedding model (CLIP, etc.) | Image vector embeddings | Similarity search |
| **Audio** | Audio embedding model | Audio vector embeddings | Similarity search |

**Why This Matters:**
- All modalities get converted to the same format (vectors)
- Enables cross-modal search (e.g., text query → image results)
- Unified infrastructure for all data types

---

## 3. The Vector Store Problem

### Traditional Database Search vs. Text Search

> **Open with a question:** "We know how to search for data in a DB (SQL or NoSQL)... but how do we search for **text data**?"

**The Challenge:**
- Structured data → Easy (WHERE clauses, indexes)
- Text/unstructured data → Hard (no obvious structure to query)

### Can We Do String Match?

> **Ask the class:** "Can we just do string matching?"

**Answer: NO - and here's why:**

#### Problem 1: Synonyms and Rephrasing
| User Searches For | Document Contains | String Match? |
|-------------------|-------------------|---------------|
| "Car" | "Vehicle" | ❌ FAIL |
| "laptop" | "notebook computer" | ❌ FAIL |
| "purchase" | "buy" | ❌ FAIL |

#### Problem 2: No Semantic Understanding
| User Query | Relevant Document | String Match? |
|------------|-------------------|---------------|
| "I am travelling to Mexico, what phone plan should I use" | "International plans" | ❌ FAIL |
| "my app keeps stopping" | "Application crash troubleshooting" | ❌ FAIL |
| "can't get into my account" | "Password reset instructions" | ❌ FAIL |

> **Key Insight:** String matching has **no understanding of meaning** - it's purely lexical!

---

## 4. What is an Embedding?

### Definition

> **Core Concept:** Embeddings create a **vector representation** of a piece of text.

**Simple Explanation:**
- An embedding converts text into a list of numbers (a vector)
- These numbers encode the **semantic meaning** of the text
- Similar meanings → Similar vectors (close in vector space)

**Visual Example:**
```
Input Text          → Embedding Model →    Vector Embeddings
┌──────────┐                          ┌─────────────────────────┐
│ New York │        ┌─────────┐      │ [0.027, -0.011, ..., -0.023] │
├──────────┤   ──►  │EMBEDDING│  ──► ├─────────────────────────┤
│  Paris   │        │  MODEL  │      │ [0.025, -0.009, ..., -0.025] │ ← Similar!
├──────────┤        └─────────┘      ├─────────────────────────┤
│  Animal  │                          │ [-0.011, 0.021, ..., 0.013] │
├──────────┤                          ├─────────────────────────┤
│  Horse   │                          │ [-0.009, 0.019, ..., 0.015] │ ← Similar!
└──────────┘                          └─────────────────────────┘
```

**Notice:** 
- "New York" and "Paris" have similar vectors (both cities)
- "Animal" and "Horse" have similar vectors (related concepts)

### What Can We Do With Embeddings?

1. **Think about text in vector space** - mathematical operations become possible
2. **Semantic search** - find pieces of text that are **most similar in meaning**
3. **Clustering** - group similar documents automatically
4. **Recommendations** - find related content

---

## 5. How Vector Store Works

### The Two-Phase Architecture

> **Key Concept:** Vector stores work in two phases - **Offline** (indexing) and **Online** (querying)

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                       │
│   OFFLINE PHASE                    │    ONLINE PHASE                  │
│   (Indexing - Done Once)           │    (Query Time - Real-time)      │
│                                    │                                   │
│   Documents ──► Embedding ──► Store │   Query ──► Embedding ──► KNN   │
│                  Model      in DB  │               Model      Search  │
│                                    │                    │              │
│   [doc1] ──►  [0.5, 0.8, -0.3, 0.2] ──►  ┌────────┐   │              │
│   [doc2] ──►  [0.8, 0.3, -0.1, 1.0] ──►  │ Vector │◄──┘              │
│   [doc3] ──►  [0.7, 0.6, 0.2, 0.9] ──►   │   DB   │                   │
│                                          └────┬───┘                   │
│   Providers: Cohere, OpenAI,                  │                       │
│   HuggingFace, etc.                           ▼                       │
│                                         Query Results                 │
│                                         (TEXT, not vectors!)          │
└─────────────────────────────────────────────────────────────────────┘
```

### Offline Phase (Indexing)
1. **Take your documents** (support tickets, articles, FAQs, etc.)
2. **Create embeddings** for each document using an embedding model
3. **Store the vectors** in a vector database along with the original text

### Online Phase (Query Time)
1. **User submits a query** (unstructured text)
2. **Embed the query** using the **SAME embedding model**
3. **KNN (K-Nearest Neighbors) search** - find vectors closest to query vector
4. **Return the original text** (not the vectors!) of the most similar documents

> **Important Question to Ask:** "What is the type of the output?"  
> **Answer:** The output is **TEXT, not embeddings!** The vector store retrieves the original documents.

---

## 6. Key Concepts Recap

### The Flow Summary
```
┌────────────────────────────────────────────────────────────────┐
│ 1. Creates an embedding (representation) for the input text   │
│ 2. Store the resulting embedding vectors in vector DB         │
│ 3. At query time:                                              │
│    ○ Embed the unstructured query                              │
│    ○ Retrieve the embedding vectors that are 'most similar'   │
│      to the embedded query                                     │
│    ○ Return the original text associated with those vectors   │
└────────────────────────────────────────────────────────────────┘
```

### Similarity Search (KNN)
- **K-Nearest Neighbors** algorithm finds the K vectors closest to the query
- "Closeness" = high cosine similarity or low euclidean distance
- Returns ranked results by similarity score

---

## 7. Demo 1 - Implementation Notes

### Demo Overview
The demo (`demo.py`) shows:
1. How to generate embeddings using OpenAI's API
2. Computing similarity scores between texts
3. Finding most similar documents (semantic search)
4. Visualizing embedding relationships

### Key Code Walkthrough

#### Part 1: Generate Embeddings
```python
from openai import OpenAI
import numpy as np

# Initialize client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Combine title and description for richer context
ticket_texts = [
    f"{ticket['title']}. {ticket['description']}" 
    for ticket in tickets
]

# Generate embeddings using OpenAI's API
response = client.embeddings.create(input=ticket_texts, model="text-embedding-3-small")

# Convert to numpy array
embeddings = np.array([data.embedding for data in response.data])
# Shape: (num_tickets, 1536) - 1536 dimensions per embedding
```

**Teaching Point:** Always combine relevant fields (title + description) for richer embeddings.

#### Part 2: Compute Similarity
```python
from sklearn.metrics.pairwise import cosine_similarity

# Generate query embedding (SAME model!)
query = "Users can't login after changing password"
query_response = client.embeddings.create(input=[query], model="text-embedding-3-small")
query_embedding = np.array([query_response.data[0].embedding])

# Compute cosine similarity
similarities = cosine_similarity(query_embedding, embeddings)[0]
# Range: -1 to 1 (higher = more similar)
```

**Teaching Point:** ALWAYS use the same embedding model for queries and documents!

#### Part 3: Retrieve Top Results
```python
# Get top-5 most similar
top_k = 5
top_indices = np.argsort(similarities)[::-1][:top_k]

for idx in top_indices:
    print(f"Score: {similarities[idx]:.4f}")
    print(f"Title: {tickets[idx]['title']}")
```

### Expected Output
```
Query: 'Users can't login after changing password'
---------------------------------------------------------
#1 - Similarity: 0.8234
Ticket ID: TKT-001
Title: Password reset not working after email verification

#2 - Similarity: 0.7891
Ticket ID: TKT-015
Title: Login issues after password change
...
```

---

## 8. OpenAI Embedding Models

| Model | Dimensions | Best For | Cost/1M tokens |
|-------|------------|----------|----------------|
| `text-embedding-3-small` | 1536 | General purpose, cost-effective | $0.02 |
| `text-embedding-3-large` | 3072 | Highest quality | $0.13 |
| `text-embedding-ada-002` | 1536 | Legacy (deprecated) | $0.10 |

**Recommendation:** Use `text-embedding-3-small` for most applications.

---

## 9. Similarity Search, KNN & Distance Metrics - Deep Dive

> **Teaching Context:** This section explains the mathematical foundations of how vector databases find "similar" documents.

---

### What is Similarity Search?

**Core Idea:** Given a query vector, find the vectors in our database that are "closest" to it.

```
Query: "How do I reset my password?"
   ↓
[0.23, -0.45, 0.81, ...]  ← Query embedding
   ↓
Compare against ALL stored embeddings
   ↓
Return the K most similar documents
```

**The Big Question:** How do we define "closest" or "most similar"?

---

### What is KNN (K-Nearest Neighbors)?

**KNN** is the algorithm that finds the K vectors closest to your query vector.

```
                    Vector Space (simplified to 2D)
                    
        ^
        |     ⭐ Query
        |    /|\
        |   / | \
        |  /  |  \
        | 📄1 📄2  📄3    ← K=3 nearest neighbors
        |        📄4
        |   📄5       📄6
        |          📄7
        +------------------------->
        
    Result: Return documents 1, 2, 3 (the 3 closest to ⭐)
```

**How KNN Works:**

1. **Compute distance** from query to EVERY vector in the database
2. **Sort** all vectors by distance (ascending)
3. **Return top K** vectors with smallest distance

**The Problem with Brute-Force KNN:**
- 1 million vectors × 1536 dimensions = **1.5 billion calculations per query!**
- This is O(n) - linear time, doesn't scale

**Solution: Approximate Nearest Neighbors (ANN)**
- Use index structures (HNSW, IVF) to avoid checking every vector
- Trade small accuracy loss for massive speed gains
- Most vector DBs use ANN under the hood

---

### Distance & Similarity Metrics Explained

> **Key Insight:** "Distance" and "Similarity" are inverses. Low distance = High similarity.

---

#### 1. Cosine Similarity (Most Common for Embeddings)

**What it measures:** The angle between two vectors (ignoring their length)

**Formula:**
$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|} = \frac{\sum_{i=1}^{n} A_i \times B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$$

**Visual Intuition:**
```
        ^
        |   B
        |  /
        | /  θ = angle between A and B
        |/______ A
        +----------->
        
    cos(0°) = 1     → Same direction (identical meaning)
    cos(90°) = 0    → Perpendicular (unrelated)
    cos(180°) = -1  → Opposite direction (opposite meaning)
```

**Range:** -1 to +1
| Score | Meaning | Example |
|-------|---------|---------|
| 0.95 - 1.0 | Nearly identical | "car" vs "automobile" |
| 0.7 - 0.95 | Very similar | "car" vs "vehicle" |
| 0.5 - 0.7 | Somewhat related | "car" vs "road" |
| 0.0 - 0.5 | Weakly related | "car" vs "apple" |
| < 0 | Opposite/contradictory | Rare with embeddings |

**Python Implementation:**
```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Example
query = np.array([0.5, 0.3, 0.8])
doc = np.array([0.4, 0.35, 0.75])
similarity = cosine_similarity(query, doc)  # ~0.99 (very similar)
```

**Why Cosine is Perfect for Embeddings:**
1. **Magnitude-invariant** - A longer document doesn't automatically seem "more similar"
2. **Normalized comparison** - Fair comparison regardless of vector scale
3. **OpenAI embeddings are unit-normalized** - All have magnitude 1.0
4. **Intuitive interpretation** - Higher = more similar

---

#### 2. Euclidean Distance (L2 Distance)

**What it measures:** The "straight line" distance between two points in space

**Formula:**
$$\text{euclidean\_distance}(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$$

**Visual Intuition:**
```
        ^
        |     B •
        |      \
        |       \ ← This is the Euclidean distance
        |        \
        |    A •--+
        +----------->
        
    Literally the length of a line connecting A and B
```

**Range:** 0 to ∞
- **0** = Identical vectors
- **Larger** = More different

**Python Implementation:**
```python
import numpy as np

def euclidean_distance(vec1, vec2):
    """Compute Euclidean (L2) distance between two vectors."""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))
    # Or simply: np.linalg.norm(vec1 - vec2)

# Example
query = np.array([0.5, 0.3, 0.8])
doc = np.array([0.4, 0.35, 0.75])
distance = euclidean_distance(query, doc)  # ~0.13 (very close)
```

**When to Use Euclidean:**
- When **magnitude matters** (rare for text embeddings)
- Image embeddings where brightness/intensity is meaningful
- Spatial data (actual physical distances)

**Why NOT Euclidean for Text Embeddings:**
- A longer document might have a larger magnitude embedding
- Two semantically identical concepts might have different magnitudes
- Cosine normalizes this away; Euclidean doesn't

---

#### 3. Dot Product (Inner Product)

**What it measures:** Sum of element-wise products (no normalization)

**Formula:**
$$\text{dot\_product}(A, B) = \sum_{i=1}^{n} A_i \times B_i = A \cdot B$$

**Relationship to Cosine:**
```
cosine_similarity = dot_product / (||A|| × ||B||)

If vectors are already normalized (||A|| = ||B|| = 1):
    cosine_similarity = dot_product  ← They're the same!
```

**Why Dot Product is Used:**
- **Faster than cosine** - No need to compute norms
- **OpenAI embeddings are pre-normalized** - Dot product = Cosine similarity
- Many vector DBs use dot product internally for speed

**Python Implementation:**
```python
import numpy as np

def dot_product(vec1, vec2):
    """Compute dot product between two vectors."""
    return np.dot(vec1, vec2)

# For normalized vectors, this equals cosine similarity!
query = np.array([0.5, 0.3, 0.8])
query_normalized = query / np.linalg.norm(query)  # Normalize

doc = np.array([0.4, 0.35, 0.75])
doc_normalized = doc / np.linalg.norm(doc)  # Normalize

similarity = dot_product(query_normalized, doc_normalized)  # Same as cosine!
```

**Teaching Point:** When using OpenAI embeddings, dot product and cosine give the same results because embeddings are already normalized to unit length.

---

#### 4. Manhattan Distance (L1 Distance / Taxicab Distance)

**What it measures:** Sum of absolute differences (like navigating a city grid)

**Formula:**
$$\text{manhattan\_distance}(A, B) = \sum_{i=1}^{n} |A_i - B_i|$$

**Visual Intuition:**
```
        ^
        |     B •
        |     |
        |     |← Can only move along axes
        |     |   (like a taxicab in Manhattan)
        |    A •--+
        +----------->
        
    Manhattan = |x2-x1| + |y2-y1| (sum of horizontal + vertical)
    Euclidean = diagonal line (shorter)
```

**Python Implementation:**
```python
import numpy as np

def manhattan_distance(vec1, vec2):
    """Compute Manhattan (L1) distance between two vectors."""
    return np.sum(np.abs(vec1 - vec2))

# Example
query = np.array([0.5, 0.3, 0.8])
doc = np.array([0.4, 0.35, 0.75])
distance = manhattan_distance(query, doc)  # 0.2 (sum of |diffs|)
```

**When to Use Manhattan:**
- **Sparse vectors** (many zeros) - More robust to outliers
- **High-dimensional data** - Sometimes performs better than Euclidean
- **Discrete features** - Categorical data encoded as numbers

**Rarely used for dense embeddings** - Cosine is preferred

---

### Comparison Summary

| Metric | Measures | Range | Best For | Speed |
|--------|----------|-------|----------|-------|
| **Cosine Similarity** | Angle (direction) | -1 to 1 | Text embeddings | Medium |
| **Dot Product** | Projection | -∞ to ∞ | Normalized embeddings | Fast |
| **Euclidean Distance** | Straight line | 0 to ∞ | Spatial/image data | Medium |
| **Manhattan Distance** | Grid distance | 0 to ∞ | Sparse vectors | Fast |

---

### Which Metric Do Vector Databases Use?

| Database | Default Metric | Options |
|----------|---------------|---------|
| **ChromaDB** | L2 (Euclidean) | cosine, l2, ip (inner product) |
| **Pinecone** | Cosine | cosine, euclidean, dotproduct |
| **Weaviate** | Cosine | cosine, l2-squared, dot, hamming |

**Setting the Metric in ChromaDB:**
```python
import chromadb

client = chromadb.Client()

# Specify distance metric when creating collection
collection = client.create_collection(
    name="support_tickets",
    metadata={"hnsw:space": "cosine"}  # Options: "cosine", "l2", "ip"
)
```

**Setting the Metric in Pinecone:**
```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="...")
pc.create_index(
    name="support-tickets",
    dimension=1536,
    metric="cosine",  # Options: "cosine", "euclidean", "dotproduct"
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

---

### Converting Distance to Similarity

Sometimes you need to convert between them:

```python
# Euclidean distance → Similarity (normalized to 0-1)
def euclidean_to_similarity(distance, max_distance=2.0):
    """Convert Euclidean distance to similarity score."""
    return 1 - (distance / max_distance)

# Cosine distance → Cosine similarity
def cosine_distance_to_similarity(distance):
    """Cosine distance is 1 - cosine_similarity."""
    return 1 - distance

# Example
euclidean_dist = 0.5
similarity = euclidean_to_similarity(euclidean_dist)  # 0.75
```

---

### Practical Example: Comparing Metrics

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Sample embeddings (imagine these are from OpenAI)
query = np.array([[0.5, 0.3, 0.8, 0.1]])
documents = np.array([
    [0.48, 0.32, 0.79, 0.12],  # Very similar
    [0.1, 0.9, 0.1, 0.5],      # Different
    [-0.5, -0.3, -0.8, -0.1],  # Opposite
])

# Cosine Similarity
cos_sim = cosine_similarity(query, documents)[0]
print("Cosine Similarity:", cos_sim)
# Output: [0.999, 0.456, -1.000]

# Euclidean Distance
euc_dist = euclidean_distances(query, documents)[0]
print("Euclidean Distance:", euc_dist)
# Output: [0.045, 1.123, 2.000]

# Notice: Cosine clearly shows doc3 is "opposite" (-1.0)
# Euclidean just shows it's "far" (2.0) - less informative!
```

---

### Key Takeaways for Students

1. **Use Cosine Similarity** for text embeddings - it's the standard
2. **KNN** finds the K closest vectors using your chosen metric
3. **Real vector DBs use ANN** (approximate) for speed, not exact KNN
4. **Dot product = Cosine** when vectors are normalized (like OpenAI's)
5. **Higher similarity = Lower distance** - they're inverses
6. **Always use the same metric** for indexing and querying!

---

## 10. Embedding Best Practices (Comprehensive Guide)

> **Teaching Context:** These are the practical tips that separate beginners from production-ready implementations.

---

### 1. Text Preprocessing

**The Goal:** Clean text without losing semantic meaning.

**DO:**
| Action | Why | Example |
|--------|-----|---------|
| Remove excessive whitespace | Wastes tokens, no meaning | `"Hello    world"` → `"Hello world"` |
| Normalize unicode | Consistent representation | `"café"` (two forms) → single form |
| Keep punctuation | Carries meaning! | `"Let's eat, grandma"` ≠ `"Lets eat grandma"` |
| Preserve case for proper nouns | `"Apple"` (company) ≠ `"apple"` (fruit) | Keep original |

**DON'T:**
| Action | Why NOT | Impact |
|--------|---------|--------|
| Remove stopwords | Embeddings handle them well | "The cat" vs "cat" have different meanings |
| Stem/lemmatize | Loses meaning | "running" → "run" loses tense information |
| Lowercase everything | Loses proper nouns | "US" (country) becomes "us" (pronoun) |
| Strip all special chars | Domain-specific meaning | "@mention", "#hashtag", "$100" |

**Code Example:**
```python
import re
import unicodedata

def preprocess_for_embedding(text: str) -> str:
    """Clean text while preserving semantic meaning."""
    # Normalize unicode (NFC = composed form)
    text = unicodedata.normalize('NFC', text)
    
    # Remove excessive whitespace (but keep single spaces)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # DON'T lowercase, DON'T remove punctuation, DON'T remove stopwords!
    return text

# Example
raw = "  Hello    world!  How's  it   going?  "
clean = preprocess_for_embedding(raw)
# Result: "Hello world! How's it going?"
```

---

### 2. Optimal Text Length

> **Critical for RAG performance** - This directly affects retrieval quality!

**Token Length Guidelines:**

| Range | Quality | Recommendation |
|-------|---------|----------------|
| **< 10 tokens** | Poor | Too short, noisy embeddings |
| **10-50 tokens** | Acceptable | Single sentences, specific queries |
| **50-200 tokens** | Good | Paragraphs, focused content |
| **200-500 tokens** | Optimal | Rich context, best retrieval |
| **500-1000 tokens** | Acceptable | May dilute specific topics |
| **> 1000 tokens** | Declining | Too much mixed content |
| **> 8191 tokens** | Error! | API limit exceeded |

**Why This Matters:**

```
Too Short (5 tokens):
"Password reset" → Embedding lacks context, matches too many things

Too Long (2000 tokens):
"[Entire support article about passwords, security, 2FA, ...]"
→ Embedding is "average" of everything, matches nothing well

Just Right (200 tokens):
"To reset your password, click the 'Forgot Password' link on the login page. 
Enter your email address and we'll send a reset link. The link expires in 24 hours.
If you don't receive the email, check your spam folder."
→ Focused, specific, retrieves accurately
```

**Checking Token Count:**
```python
import tiktoken

def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Count tokens for OpenAI embedding models."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# Check before embedding
text = "Your document text here..."
tokens = count_tokens(text)
print(f"Token count: {tokens}")

if tokens > 8191:
    print("WARNING: Text exceeds token limit! Need to chunk.")
elif tokens < 20:
    print("WARNING: Text may be too short for quality embedding.")
```

---

### 3. Batch Processing

> **Impact:** 10-100x faster, significantly cheaper

**The Problem:**
```python
# BAD: One API call per text
embeddings = []
for text in texts:  # 1000 texts = 1000 API calls!
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embeddings.append(response.data[0].embedding)
# Takes: ~5-10 minutes, many API calls
```

**The Solution:**
```python
# GOOD: Batch up to 2048 texts per call
def batch_embed(texts: list, batch_size: int = 2048) -> list:
    """Efficiently embed texts in batches."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)}")
    
    return all_embeddings

# 1000 texts = 1 API call!
embeddings = batch_embed(texts)
# Takes: ~2-5 seconds
```

**Batch Limits:**
| Constraint | Limit |
|------------|-------|
| Max texts per batch | 2048 |
| Max tokens per text | 8191 |
| Max total tokens per request | ~1M (varies) |

---

### 4. Caching Strategy

> **Why Cache?** Embeddings are deterministic - same text = same embedding. Don't pay twice!

**Simple In-Memory Cache:**
```python
import hashlib
from functools import lru_cache

# Option 1: Using LRU cache decorator
@lru_cache(maxsize=10000)
def get_embedding_cached(text: str) -> tuple:
    """Cache embeddings in memory (returns tuple for hashability)."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return tuple(response.data[0].embedding)
```

**File-Based Cache (Persists Across Sessions):**
```python
import hashlib
import json
import os

CACHE_DIR = "./embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_embedding_with_cache(text: str) -> list:
    """Cache embeddings to disk for persistence."""
    # Create unique hash for this text
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{text_hash}.json")
    
    # Check cache
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    # Generate embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
    
    # Save to cache
    with open(cache_path, 'w') as f:
        json.dump(embedding, f)
    
    return embedding
```

**Database Cache (Production-Ready):**
```python
import sqlite3
import json
import hashlib

class EmbeddingCache:
    """SQLite-based embedding cache for production use."""
    
    def __init__(self, db_path: str = "embedding_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                embedding TEXT,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def get_or_create(self, text: str, model: str = "text-embedding-3-small") -> list:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Check cache
        cursor = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE text_hash = ?",
            (text_hash,)
        )
        row = cursor.fetchone()
        
        if row:
            return json.loads(row[0])
        
        # Generate and cache
        response = client.embeddings.create(model=model, input=text)
        embedding = response.data[0].embedding
        
        self.conn.execute(
            "INSERT INTO embeddings (text_hash, embedding, model) VALUES (?, ?, ?)",
            (text_hash, json.dumps(embedding), model)
        )
        self.conn.commit()
        
        return embedding

# Usage
cache = EmbeddingCache()
embedding = cache.get_or_create("How do I reset my password?")
```

---

### 5. Rate Limit Handling

> **Reality:** OpenAI has rate limits. Your code WILL hit them at scale.

**Using Tenacity for Retry Logic:**
```python
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type
)
from openai import RateLimitError, APIError

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
def get_embedding_with_retry(text: str) -> list:
    """Get embedding with automatic retry on rate limits."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# This will:
# - Wait 1s, then 2s, then 4s, then 8s... up to 60s between retries
# - Give up after 5 attempts
# - Only retry on RateLimitError or APIError
```

**Rate Limit Aware Batching:**
```python
import time
from typing import List

def embed_with_rate_limiting(
    texts: List[str],
    batch_size: int = 100,  # Smaller batches for safety
    requests_per_minute: int = 3000
) -> List[list]:
    """Embed texts while respecting rate limits."""
    embeddings = []
    delay = 60 / requests_per_minute  # Seconds between requests
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            embeddings.extend([item.embedding for item in response.data])
            
        except RateLimitError:
            print("Rate limit hit, waiting 60 seconds...")
            time.sleep(60)
            # Retry this batch
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            embeddings.extend([item.embedding for item in response.data])
        
        # Proactive rate limiting
        time.sleep(delay)
        
        if (i // batch_size) % 10 == 0:
            print(f"Progress: {i + len(batch)}/{len(texts)}")
    
    return embeddings
```

---

### 6. Cost Estimation

> **Know your costs before running large jobs!**

```python
import tiktoken

def estimate_embedding_cost(
    texts: List[str],
    model: str = "text-embedding-3-small"
) -> dict:
    """Estimate cost before embedding."""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Count tokens
    total_tokens = sum(len(encoding.encode(text)) for text in texts)
    
    # Cost per million tokens (as of 2024)
    costs = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10
    }
    
    cost_per_million = costs.get(model, 0.02)
    estimated_cost = (total_tokens / 1_000_000) * cost_per_million
    
    return {
        "total_texts": len(texts),
        "total_tokens": total_tokens,
        "avg_tokens_per_text": total_tokens / len(texts),
        "model": model,
        "estimated_cost_usd": round(estimated_cost, 4)
    }

# Example usage
estimate = estimate_embedding_cost(texts)
print(f"Embedding {estimate['total_texts']} texts")
print(f"Total tokens: {estimate['total_tokens']:,}")
print(f"Estimated cost: ${estimate['estimated_cost_usd']}")

# Output:
# Embedding 10000 texts
# Total tokens: 1,500,000
# Estimated cost: $0.03
```

---

### 7. Best Practices Summary Table

| Practice | Do This | Not This | Impact |
|----------|---------|----------|--------|
| **Preprocessing** | Light cleaning | Heavy NLP | Preserve meaning |
| **Text Length** | 100-500 tokens | <10 or >1000 | Retrieval quality |
| **API Calls** | Batch 100-2048 | One at a time | 100x faster |
| **Caching** | Hash → cache | Re-embed always | Save $$ |
| **Rate Limits** | Retry with backoff | Crash on error | Reliability |
| **Cost** | Estimate first | YOLO | Budget control |
| **Model** | Same everywhere | Mix models | Compatibility |

---

## 11. Common Pitfalls to Warn About

| Pitfall | Why It's Bad | Solution |
|---------|--------------|----------|
| Mixing embedding models | Incompatible vector spaces | Use one model consistently |
| Embedding very long docs | API error / truncation | Chunk into 200-500 tokens |
| Embedding every sentence | Expensive, noisy results | Find optimal chunk size |
| No rate limit handling | API failures | Use retry with backoff |

---

## 12. Visualization in Demo

The demo creates a visualization showing:
1. **Similarity Heatmap** - Pairwise similarities between documents
2. **Query Similarity Bar Chart** - How similar each doc is to the query

**Teaching Point:** These show the TRUE relationships in 1536-dimensional space!

---

## 13. Key Takeaways (Summary Slide)

1. **Embeddings convert text into numerical vectors** that capture semantic meaning
2. **Similar meanings → Similar vectors** (close in vector space)
3. **Vector databases** store and search these embeddings efficiently
4. **String matching fails** for semantic search - no understanding of synonyms or meaning
5. **Two-phase architecture:** Offline (index) + Online (query)
6. **Output is TEXT, not vectors** - the DB retrieves original documents

---

## 14. How Vector Databases Store Data & Metadata

> **Key Concept:** Vector databases don't just store vectors - they store a **complete record** with multiple components.

### The Anatomy of a Vector Store Record

Every record in a vector database typically has **4 components**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VECTOR DATABASE RECORD                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. ID (Primary Key)                                                        │
│     └── "ticket_001" or UUID                                                │
│                                                                              │
│  2. VECTOR (The Embedding)                                                  │
│     └── [0.023, -0.451, 0.812, ..., 0.034]  (1536 floats)                  │
│                                                                              │
│  3. DOCUMENT/TEXT (Original Content)                                       │
│     └── "User cannot login after password reset. Error code 401..."        │
│                                                                              │
│  4. METADATA (Structured Attributes)                                       │
│     └── {"category": "auth", "priority": "high", "date": "2024-01-15"}     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component 1: ID (Primary Key)

**Purpose:** Unique identifier for each record

```python
# Examples of IDs
ids = [
    "ticket_001",           # Custom string ID
    "doc_2024_01_15_001",   # Encoded information in ID
    "550e8400-e29b...",     # UUID (guaranteed unique)
]
```

**Best Practices:**
- Use meaningful IDs when possible (helps debugging)
- UUIDs for high-volume automated ingestion
- Keep IDs consistent across your system (same ID in source DB)

---

### Component 2: Vector (The Embedding)

**Purpose:** The numerical representation used for similarity search

```python
# What a vector looks like
vector = [
    0.0234,   # Dimension 1
    -0.4512,  # Dimension 2
    0.8123,   # Dimension 3
    # ... (1536 total for text-embedding-3-small)
    0.0341    # Dimension 1536
]
```

**Storage Details:**
- Each dimension is typically a **32-bit float** (4 bytes)
- 1536 dimensions × 4 bytes = **6,144 bytes per vector** (~6 KB)
- 1 million vectors ≈ **6 GB** of vector storage alone

**Index Structures (How Search is Optimized):**

| Index Type | How It Works | Trade-off |
|------------|--------------|-----------|
| **Flat (Brute Force)** | Compare query to ALL vectors | 100% accurate, but slow at scale |
| **IVF (Inverted File)** | Cluster vectors, search only relevant clusters | Faster, slight accuracy loss |
| **HNSW (Hierarchical NSW)** | Graph-based navigation through vector space | Very fast, memory-intensive |
| **PQ (Product Quantization)** | Compress vectors, approximate search | Memory efficient, lower accuracy |

> **Teaching Point:** ChromaDB uses HNSW by default. Pinecone auto-selects based on your data.

---

### Component 3: Document (Original Content)

**Purpose:** The actual text that gets returned to the user

```python
# The document is what you get back from search!
document = """
User cannot login after password reset. They receive error code 401 
when attempting to authenticate. The issue started after the 
January security update.
"""
```

**Why Store the Original Document?**
- Vectors are for **searching**
- Documents are for **returning/displaying**
- You can't reverse an embedding back to text!

**Storage Options:**

| Approach | Pros | Cons |
|----------|------|------|
| **Store full text in vector DB** | Simple, self-contained | Increases DB size |
| **Store reference/pointer only** | Smaller DB, source of truth elsewhere | Extra lookup required |
| **Store both** | Fast retrieval + can sync with source | Data duplication |

---

### Component 4: Metadata (Structured Attributes)

**Purpose:** Enable filtering, categorization, and hybrid search

```python
# Rich metadata example
metadata = {
    # Categorical (for filtering)
    "category": "billing",
    "priority": "high",
    "status": "open",
    
    # Temporal (for date-range queries)
    "created_at": "2024-01-15",
    "updated_at": "2024-01-20",
    
    # Numeric (for range queries)
    "ticket_number": 12345,
    "customer_tier": 3,
    
    # Arrays (for multi-value filters)
    "tags": ["payment", "refund", "urgent"],
    
    # Source tracking
    "source_file": "tickets_jan_2024.json",
    "chunk_index": 0
}
```

**Why Metadata is Critical:**

1. **Filtering Before Vector Search** (Pre-filtering)
   ```python
   # Find similar tickets, but ONLY in billing category
   results = collection.query(
       query_texts=["refund not received"],
       where={"category": "billing"},  # Filter first!
       n_results=5
   )
   ```

2. **Filtering After Vector Search** (Post-filtering)
   ```python
   # Get top 100 similar, then filter to high priority
   results = collection.query(
       query_texts=["login issue"],
       n_results=100,
       where={"priority": "high"}  # Applied after similarity
   )
   ```

3. **Combining with Keywords** (Hybrid Search)
   ```python
   # Semantic similarity + exact keyword match
   results = collection.query(
       query_texts=["payment problem"],
       where={"$and": [
           {"status": "open"},
           {"tags": {"$contains": "urgent"}}
       ]}
   )
   ```

---

### How Each Database Stores This

#### ChromaDB Storage Architecture

```
chroma_db/
├── chroma.sqlite3          # Metadata + document storage
└── <collection-uuid>/
    ├── data_level0.bin     # HNSW index (vectors)
    ├── header.bin          # Index configuration
    ├── index_metadata.json # Index stats
    └── length.bin          # Vector lengths
```

- **SQLite** stores: IDs, documents, metadata
- **Binary files** store: HNSW vector index
- Everything in **one directory** - easy to backup/move

#### Pinecone Storage Architecture

```
┌─────────────────────────────────────────────┐
│              Pinecone Cloud                 │
├─────────────────────────────────────────────┤
│  Index: "support-tickets"                   │
│  ├── Namespace: "production"                │
│  │   ├── Vectors (stored in pods/shards)   │
│  │   ├── Metadata (indexed for filtering)  │
│  │   └── Sparse vectors (for hybrid)       │
│  └── Namespace: "staging"                   │
│      └── ...                                │
└─────────────────────────────────────────────┘
```

- **Namespaces** partition data within an index
- **Pods** = compute/storage units (you pay for these)
- **Metadata** indexed separately for fast filtering
- **No document storage** - you store text externally or in metadata

> **Important:** Pinecone doesn't have a "document" field! Store your text in metadata.

```python
# Pinecone: text goes in metadata
index.upsert(vectors=[{
    "id": "ticket_001",
    "values": [0.1, 0.2, ...],
    "metadata": {
        "text": "Your original document here",  # ← Document stored as metadata
        "category": "billing"
    }
}])
```

#### Weaviate Storage Architecture

```
┌─────────────────────────────────────────────┐
│              Weaviate                       │
├─────────────────────────────────────────────┤
│  Class: "SupportTicket"                     │
│  ├── Schema (defined structure)             │
│  │   ├── title: text (vectorized)          │
│  │   ├── description: text (vectorized)    │
│  │   ├── category: text (filterable)       │
│  │   └── priority: int (filterable)        │
│  ├── Objects (actual data)                 │
│  ├── Vector Index (HNSW)                   │
│  └── Inverted Index (for BM25/filtering)   │
└─────────────────────────────────────────────┘
```

- **Schema-based** - must define structure upfront
- **Multiple indexes**: Vector (HNSW) + Inverted (BM25/filtering)
- **Cross-references** - link objects like a graph database
- **Modular vectorizers** - can auto-generate embeddings

---

### Storage Size Estimation

> **Useful for capacity planning discussions**

| Component | Size per Record | 1M Records |
|-----------|-----------------|------------|
| Vector (1536D, float32) | 6 KB | 6 GB |
| ID (avg 36 chars) | 36 bytes | 36 MB |
| Document (avg 500 chars) | 500 bytes | 500 MB |
| Metadata (avg) | 200 bytes | 200 MB |
| **Index overhead** | ~30% of vectors | ~2 GB |
| **Total Estimate** | ~8 KB | **~8-10 GB** |

**Teaching Point:** Vector storage dominates! This is why some DBs offer dimension reduction.

---

## 15. Popular Vector Databases - Comparison

> **Teaching Context:** Students often ask "Which vector database should I use?" Here's a detailed comparison of the 3 most popular options.

---

### 1. ChromaDB

**What it is:** Open-source, lightweight, embedded vector database designed for AI applications.

**Best For:** Prototyping, local development, small-to-medium projects, learning RAG

```python
# Simple ChromaDB example
import chromadb

client = chromadb.Client()
collection = client.create_collection("support_tickets")

# Add documents
collection.add(
    documents=["How to reset password", "Payment failed"],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(query_texts=["login issues"], n_results=2)
```

| Pros | Cons |
|------|------|
| **Zero configuration** - Works out of the box, no server setup | **Limited scalability** - Not designed for billions of vectors |
| **Embedded mode** - Runs in-process, great for notebooks | **Single-node only** - No built-in distributed mode |
| **Automatic embeddings** - Built-in embedding functions | **Fewer indexing options** - Limited ANN algorithm choices |
| **Python-native** - Feels natural for ML/AI developers | **Newer project** - Smaller community than alternatives |
| **Persistent storage** - SQLite backend, survives restarts | **Production concerns** - Less battle-tested at scale |
| **Free & open source** - No licensing costs | **Memory usage** - Loads data into memory |

**When to Choose ChromaDB:**
- Building proof-of-concept or MVP
- Educational projects and workshops (like this one!)
- Applications with < 1 million vectors
- When you want minimal DevOps overhead

---

### 2. Pinecone

**What it is:** Fully managed, cloud-native vector database as a service.

**Best For:** Production workloads, enterprise applications, teams without infrastructure expertise

```python
# Pinecone example
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index("support-tickets")

# Upsert vectors
index.upsert(vectors=[
    {"id": "doc1", "values": [0.1, 0.2, ...], "metadata": {"category": "billing"}},
    {"id": "doc2", "values": [0.3, 0.4, ...], "metadata": {"category": "technical"}}
])

# Query with metadata filtering
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={"category": {"$eq": "billing"}}
)
```

| Pros | Cons |
|------|------|
| **Fully managed** - Zero infrastructure management | **Cost** - Can get expensive at scale ($70+/month for production) |
| **Auto-scaling** - Handles traffic spikes automatically | **Vendor lock-in** - Proprietary, no self-hosting option |
| **High availability** - Built-in replication & redundancy | **Data residency** - Limited region options |
| **Fast performance** - Optimized for low-latency queries | **Cold start** - Serverless tier has latency on first query |
| **Metadata filtering** - Powerful hybrid search capabilities | **Learning curve** - Concepts like pods, replicas, shards |
| **Enterprise features** - SSO, RBAC, audit logs | **Internet dependency** - Requires network connectivity |
| **99.99% SLA** - Production-grade reliability | **Debugging** - Harder to inspect internal state |

**When to Choose Pinecone:**
- Production applications with SLA requirements
- Teams that want to focus on product, not infrastructure
- Need for enterprise security & compliance
- Applications requiring guaranteed uptime

**Pricing Tiers:**
- **Free tier**: 1 index, 100K vectors (good for testing)
- **Standard**: $70/month+ (production workloads)
- **Enterprise**: Custom pricing (dedicated infrastructure)

---

### 3. Weaviate

**What it is:** Open-source vector database with built-in ML models and GraphQL API.

**Best For:** Complex search scenarios, multi-modal applications, teams wanting flexibility

```python
# Weaviate example
import weaviate

client = weaviate.Client("http://localhost:8080")

# Create schema with vectorizer
client.schema.create_class({
    "class": "SupportTicket",
    "vectorizer": "text2vec-openai",  # Auto-generate embeddings!
    "properties": [
        {"name": "title", "dataType": ["text"]},
        {"name": "category", "dataType": ["text"]}
    ]
})

# Add object (embedding generated automatically)
client.data_object.create(
    {"title": "Password reset issue", "category": "auth"},
    "SupportTicket"
)

# Semantic search
result = client.query.get("SupportTicket", ["title", "category"])\
    .with_near_text({"concepts": ["login problems"]})\
    .with_limit(5)\
    .do()
```

| Pros | Cons |
|------|------|
| **Open source** - Self-host or use managed cloud | **Complexity** - Steeper learning curve than ChromaDB |
| **Built-in vectorizers** - OpenAI, Cohere, HuggingFace integrations | **Resource hungry** - Requires more memory/CPU |
| **GraphQL API** - Flexible, powerful query language | **Schema required** - Must define structure upfront |
| **Multi-modal** - Text, images, and more in same DB | **Operational overhead** - Self-hosting requires expertise |
| **Hybrid search** - Combine vector + keyword (BM25) | **Documentation** - Can be overwhelming for beginners |
| **Horizontal scaling** - Distributed architecture | **Cold starts** - Initial schema loading can be slow |
| **Active community** - Strong open-source ecosystem | **Breaking changes** - Newer project, API evolving |
| **Modules system** - Extensible architecture | **Debugging** - Complex queries can be hard to troubleshoot |

**When to Choose Weaviate:**
- Need hybrid search (semantic + keyword)
- Multi-modal applications (text + images)
- Want open-source with cloud option
- Complex data relationships and queries
- Teams with DevOps/infrastructure capabilities

**Deployment Options:**
- **Self-hosted**: Docker, Kubernetes
- **Weaviate Cloud Services (WCS)**: Managed offering
- **Embedded**: New experimental in-process mode

---

### Quick Comparison Table

| Feature | ChromaDB | Pinecone | Weaviate |
|---------|----------|----------|----------|
| **Type** | Embedded/Open-source | Managed SaaS | Open-source + Cloud |
| **Setup Difficulty** | ⭐ Easy | ⭐⭐ Medium | ⭐⭐⭐ Complex |
| **Scalability** | Small-Medium | Large | Large |
| **Cost** | Free | $$ - $$$ | Free (self-host) or $$ |
| **Best For** | Prototyping | Production | Flexibility |
| **Built-in Embeddings** | ✅ Yes | ❌ No | ✅ Yes |
| **Hybrid Search** | ❌ Limited | ✅ Metadata filters | ✅ BM25 + Vector |
| **Self-Hosting** | ✅ Yes | ❌ No | ✅ Yes |
| **Managed Cloud** | ❌ No | ✅ Yes | ✅ Yes |

---

### Recommendation for This Workshop

> **We use ChromaDB** because:
> 1. Zero setup - works immediately in Python
> 2. Perfect for learning concepts
> 3. No API keys or accounts needed for the vector DB itself
> 4. Easy to inspect and debug
> 5. Concepts transfer to other vector DBs

**For production, consider:**
- **Pinecone** if you want zero-ops and have budget
- **Weaviate** if you need hybrid search or want open-source at scale
- **ChromaDB** if your data fits in memory and you want simplicity

---

## References

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone: Vector Embeddings Explained](https://www.pinecone.io/learn/vector-embeddings/)
- [Cosine Similarity - Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)

---

## Next Module

**Module 2: Chunking** - How to split documents optimally for embedding and retrieval.
