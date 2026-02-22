# Module 2: Chunking Strategies - Instructor Notes

---

## 1. Introduction: Breaking Down Long Documents (Opening Slide)

### Connecting to Module 1

> **Opening Line:** "We just covered Embeddings and Vector Stores... but there's a missing piece!"

**Diagram to Draw:**
```
┌─────────────────────┐
│    Embeddings       │ ── We just covered this
├─────────────────────┤
│    Vector Store     │ ── We just covered this
├─────────────────────┤
│    Chunking         │ ── What is this though? 🤔
└─────────────────────┘
```

**The Problem to Pose:**
- What happens when your document is 10,000 tokens but embedding models only accept 8,191?
- What happens when your PDF is 50 pages?
- What happens when you have a complete user manual?

> **Key Question to Ask:** "How do we break these down into pieces suitable for embedding?"

---

## 2. Why Chunking Matters

### The Core Challenge

| Constraint | Limit | Problem If Ignored |
|------------|-------|-------------------|
| Embedding API limit | 8,191 tokens | API error / truncation |
| LLM context window | 4K-128K tokens | Can't fit retrieved docs |
| Retrieval precision | Depends on query | Too much irrelevant content |

### The Goldilocks Problem

```
Too Small (10 tokens):
"password reset" 
→ No context, noisy, matches everything

Too Large (5000 tokens):
"[Entire support manual with 50 topics]"
→ Diluted embedding, irrelevant info returned

Just Right (200-500 tokens):
"To reset your password: 1) Go to login page 2) Click 'Forgot Password'
3) Enter email 4) Check inbox for reset link 5) Create new password
that meets requirements: 8+ chars, 1 uppercase, 1 number"
→ Complete, focused, retrieves accurately
```

**Teaching Point:** Chunking is about finding the OPTIMAL size for your use case!

---

## 3. The 5 Chunking Strategies

> **Core Content:** Walk through each strategy as shown in the slides

---

### Strategy 1: Fixed-Size Chunking

**What It Is:** Split text into uniform segments based on a pre-defined number of characters, words, or tokens.

**Visual (from slide):**
```
"Artificial intelligence is transforming technology and shaping the future."
  └──────────── Chunk 1 ──────────────┘└── Overlap ──┘
                                       └──────────── Chunk 2 ──────────────┘
```

**Key Concept: OVERLAP**
```python
# Without overlap: Information at boundaries is LOST
Chunk 1: "The quick brown fox jumps"
Chunk 2: "over the lazy dog."
# Query "fox jumps over" might not match well!

# With overlap: Boundary information is PRESERVED
Chunk 1: "The quick brown fox jumps over"
Chunk 2: "fox jumps over the lazy dog."
# Query "fox jumps over" now matches both!
```

**Code Example:**
```python
def fixed_size_chunking(text, chunk_size=500, overlap=100):
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Move back by overlap amount
    
    return chunks

# Example
text = "A" * 1000  # 1000 characters
chunks = fixed_size_chunking(text, chunk_size=300, overlap=50)
# Result: 4 chunks
# Chunk 1: chars 0-300
# Chunk 2: chars 250-550 (overlaps 250-300)
# Chunk 3: chars 500-800 (overlaps 500-550)
# Chunk 4: chars 750-1000 (overlaps 750-800)
```

**Using LangChain:**
```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n"
)
chunks = splitter.split_text(text)
```

| Pros | Cons |
|------|------|
| Simple to implement | May split mid-sentence |
| Predictable sizes | Ignores document structure |
| Fast processing | Can break semantic units |
| Easy to understand | No content awareness |

**Best For:** 
- Quick prototypes
- Unstructured text (chat logs, transcripts)
- When you need predictable sizes

**Parameters to Tune:**
| Parameter | Typical Range | Recommendation |
|-----------|---------------|----------------|
| chunk_size | 200-1000 | Start with 300-500 |
| overlap | 10-25% of size | 50-100 tokens |

---

### Strategy 2: Semantic Chunking

**What It Is:** Segment based on **meaning** using embeddings to detect topic shifts.

**The Algorithm (from slide):**
```
Document ──► Segment into sentences/paragraphs
                    │
                    ▼
        Initial first chunk ──► Keep adding segments
                                       │
                    ┌──────────────────┤
                    │                  ▼
                    │      Check cosine similarity with current chunk
                    │                  │
                    │     ┌────────────┴────────────┐
                    │     │                         │
                    │  Similarity HIGH          Similarity DROPS
                    │  (same topic)             (topic changed!)
                    │     │                         │
                    │     ▼                         ▼
                    │  Add to current chunk    Start NEW chunk
                    │                               │
                    └───────────────────────────────┘
                                    │
                                    ▼
                              Final chunks
```

**Visual Example (from slide):**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Artificial intelligence is transforming industries by automating            │
│ processes, enhancing decision-making, and providing insights through       │
│ data analysis.                                                              │ ← Chunk 1 (AI/automation topic)
├─────────────────────────────────────────────────────────────────────────────┤
│ Machine learning, a subset of AI, enables systems to learn                  │
│ and improve from experience without explicit programming.                   │ ← Chunk 2 (ML topic)
├─────────────────────────────────────────────────────────────────────────────┤
│ Deep learning, a branch of machine learning, uses neural networks          │
│ with multiple layers to model complex patterns in data.                    │ ← Chunk 3 (DL topic)
└─────────────────────────────────────────────────────────────────────────────┘
```

**Code Example:**
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # Options: percentile, standard_deviation, interquartile
    breakpoint_threshold_amount=95  # Break when similarity drops below 95th percentile
)

chunks = semantic_splitter.split_text(text)
```

**How It Decides Where to Split:**
```python
# Simplified algorithm
def semantic_chunking(text, threshold=0.7):
    sentences = split_sentences(text)
    embeddings = [get_embedding(s) for s in sentences]
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Compare consecutive sentences
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        
        if similarity >= threshold:
            # Same topic - keep adding
            current_chunk.append(sentences[i])
        else:
            # Topic changed - start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    
    chunks.append(" ".join(current_chunk))
    return chunks
```

| Pros | Cons |
|------|------|
| Preserves semantic coherence | Expensive (needs embeddings) |
| Natural topic boundaries | Variable chunk sizes |
| Adapts to content | Slower processing |
| Best retrieval quality | More complex to implement |

**Best For:**
- High-value content where accuracy matters
- Articles, research papers
- When you need topic-based retrieval

---

### Semantic Chunking Deep Dive: Edge Cases & Solutions

> **Teaching Point:** Students often ask "What about...?" These are the real-world edge cases that make semantic chunking tricky.

#### Step-by-Step Example: How Semantic Chunking Works

```
Document: "AI is transforming healthcare. Machine learning diagnoses diseases. 
           The stock market crashed yesterday. Investors are worried."
```

**Step 1: Split into sentences**
```
Sentence 1: "AI is transforming healthcare."
Sentence 2: "Machine learning diagnoses diseases."
Sentence 3: "The stock market crashed yesterday."
Sentence 4: "Investors are worried."
```

**Step 2: Get embedding for each sentence**
```
Sentence 1 → [0.8, 0.2, 0.1, ...]   (AI/healthcare topic)
Sentence 2 → [0.75, 0.25, 0.15, ...] (AI/healthcare topic - SIMILAR!)
Sentence 3 → [0.1, 0.9, 0.3, ...]   (finance topic - DIFFERENT!)
Sentence 4 → [0.15, 0.85, 0.25, ...] (finance topic - similar to 3)
```

**Step 3: Compare consecutive sentences using cosine similarity**
```
Similarity(Sent1, Sent2) = 0.95  ← HIGH! Same topic, keep together
Similarity(Sent2, Sent3) = 0.20  ← LOW! Topic changed, SPLIT HERE!
Similarity(Sent3, Sent4) = 0.92  ← HIGH! Same topic, keep together
```

**Step 4: Create chunks based on similarity drops**
```
Chunk 1: "AI is transforming healthcare. Machine learning diagnoses diseases."
         (Both about AI/healthcare)

Chunk 2: "The stock market crashed yesterday. Investors are worried."
         (Both about finance)
```

**Visual Diagram:**
```
                    Cosine Similarity Between Consecutive Sentences
                    
Sentence:    1         2         3         4
             │         │         │         │
             ▼         ▼         ▼         ▼
            [AI]─────[ML]      [Stock]───[Investors]
                  │                   │
           sim=0.95              sim=0.92
           (same topic)          (same topic)
                  │                   │
                  └────────┬──────────┘
                           │
                     sim=0.20 ← SPLIT HERE!
                    (topic changed)

Result:
┌──────────────────────────┐    ┌──────────────────────────┐
│        CHUNK 1           │    │        CHUNK 2           │
│ AI is transforming...    │    │ The stock market...      │
│ Machine learning...      │    │ Investors are worried.   │
└──────────────────────────┘    └──────────────────────────┘
     (AI/Healthcare)                   (Finance)
```

---

#### Edge Case 1: Similar Sentences NOT Together

**The Problem:**
```
Sentence 1: "AI transforms healthcare."           (AI topic)
Sentence 2: "The stock market crashed."           (Finance topic)
Sentence 3: "Machine learning diagnoses diseases." (AI topic - but separated!)
Sentence 4: "Investors are panicking."            (Finance topic)
```

**What Basic Semantic Chunking Does:**
```
Compare 1→2: similarity = 0.2  → SPLIT
Compare 2→3: similarity = 0.15 → SPLIT  
Compare 3→4: similarity = 0.2  → SPLIT

Result: 4 separate chunks! 😞
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ AI transforms   │ │ Stock market    │ │ ML diagnoses    │ │ Investors are   │
│ healthcare.     │ │ crashed.        │ │ diseases.       │ │ panicking.      │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘

The two AI sentences are in SEPARATE chunks!
```

**Why This Happens:** Basic semantic chunking only compares CONSECUTIVE sentences - it can't "see" that sentences 1 and 3 are related.

**Solution A: Sliding Window Comparison**
```python
def semantic_chunking_with_window(sentences, window_size=3, threshold=0.7):
    embeddings = [get_embedding(s) for s in sentences]
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Compare new sentence to AVERAGE of current chunk (not just previous)
        chunk_embeddings = [embeddings[j] for j in range(len(current_chunk))]
        chunk_avg_embedding = np.mean(chunk_embeddings, axis=0)
        similarity = cosine_similarity(chunk_avg_embedding, embeddings[i])
        
        if similarity >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    
    return chunks
```

**Solution B: Clustering Approach**
```python
from sklearn.cluster import KMeans

def cluster_based_chunking(sentences, n_clusters=5):
    embeddings = [get_embedding(s) for s in sentences]
    
    # Cluster by meaning (groups similar sentences regardless of position)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings)
    
    # Group sentences by cluster
    chunks = {}
    for i, label in enumerate(labels):
        if label not in chunks:
            chunks[label] = []
        chunks[label].append(sentences[i])
    
    return list(chunks.values())

# Result:
# Cluster 0 (AI): ["AI transforms healthcare.", "ML diagnoses diseases."]
# Cluster 1 (Finance): ["Stock market crashed.", "Investors panicking."]
```

> **Trade-off:** Clustering reorders content - may not preserve narrative flow!

---

#### Edge Case 2: Similar Sentences Exceed Chunk Size

**The Problem:**
```
Sentences 1-20: All about "How to reset your password" (same topic)
Each sentence: ~50 tokens
Total: 1000 tokens

But your chunk_size limit is 500 tokens!
```

**What Basic Semantic Chunking Does:**
```
Compare 1→2: sim=0.95 → keep together
Compare 2→3: sim=0.93 → keep together
Compare 3→4: sim=0.91 → keep together
... (all high similarity - same topic!)
Compare 19→20: sim=0.94 → keep together

Result: ONE chunk of 1000 tokens! 😱
Exceeds your 500 token limit!
```

**Solution: Semantic + Size Limit (Production Approach)**
```python
def semantic_chunking_with_size_limit(text, max_tokens=500, threshold=0.7):
    sentences = split_into_sentences(text)
    embeddings = [get_embedding(s) for s in sentences]
    
    chunks = []
    current_chunk = [sentences[0]]
    current_tokens = count_tokens(sentences[0])
    
    for i in range(1, len(sentences)):
        sentence_tokens = count_tokens(sentences[i])
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        
        # Check BOTH conditions
        would_exceed_limit = (current_tokens + sentence_tokens) > max_tokens
        topic_changed = similarity < threshold
        
        if would_exceed_limit or topic_changed:
            # Start new chunk (either too big OR topic changed)
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_tokens = sentence_tokens
        else:
            # Keep adding
            current_chunk.append(sentences[i])
            current_tokens += sentence_tokens
    
    chunks.append(" ".join(current_chunk))
    return chunks
```

**Result with Size Limit:**
```
Chunk 1: Sentences 1-10 (500 tokens) - password reset part 1
Chunk 2: Sentences 11-20 (500 tokens) - password reset part 2
         ↑ Split due to SIZE, not topic change (both still about password reset)
```

**LangChain's SemanticChunker Handles This:**
```python
from langchain_experimental.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
    # These parameters handle size limits:
    min_chunk_size=100,   # Don't create tiny chunks
    max_chunk_size=500,   # Force split if too large
)
```

---

#### Robust Semantic Chunking Algorithm (Production)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ROBUST SEMANTIC CHUNKING ALGORITHM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  For each new sentence, ask:                                                 │
│                                                                              │
│  ┌─────────────────────────────────────────┐                                │
│  │ 1. Would adding this exceed size limit? │──── YES ──► Start new chunk   │
│  └─────────────────────────────────────────┘                                │
│                    │ NO                                                      │
│                    ▼                                                         │
│  ┌─────────────────────────────────────────┐                                │
│  │ 2. Is similarity below threshold?       │──── YES ──► Start new chunk   │
│  │    (topic changed?)                     │                                │
│  └─────────────────────────────────────────┘                                │
│                    │ NO                                                      │
│                    ▼                                                         │
│           Add to current chunk                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Key Insight:** Production semantic chunking is actually **semantic + size-aware**. Pure semantic chunking without size limits is rarely used in practice!

---

### Strategy 3: Recursive Chunking

**What It Is:** Split hierarchically using multiple separators, only splitting further if chunks are still too large.

**The Algorithm (from slide):**
```
Document ──► Segment by paragraphs/sections
                    │
                    ▼
              Select a segment
                    │
                    ▼
         ┌─────────────────────┐
         │ Size > chunk limit? │
         └─────────┬───────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        NO                   YES
        │                     │
        ▼                     ▼
   Keep as-is          Split further recursively
   (final chunk)       (try next separator)
```

**How It Works:**
```python
# Try separators in order, from most to least specific
separators = [
    "\n\n",     # 1. Try paragraph breaks first
    "\n",       # 2. Then line breaks
    ". ",       # 3. Then sentence endings
    " ",        # 4. Then word boundaries
    ""          # 5. Last resort: character by character
]

# Only split more if chunk is still too large
if len(chunk) > max_size:
    try_next_separator()
```

**Visual Example (from slide):**
```
Paragraph 1:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Artificial intelligence is transforming industries by automating            │
│ processes, enhancing decision-making, and providing insights through       │
│ data analysis. Machine learning, a subset of AI, enables systems to learn  │
│ and improve from experience without explicit programming. Deep learning,   │
│ a branch of machine learning, uses neural networks with multiple layers    │
│ to model complex patterns in data.                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
        ┌─────────────────────────────────────────────────────────────────────┐
        │          If paragraph > chunk limit, split by sentences              │
        └─────────────────────────────────────────────────────────────────────┘
                                    ↓
Paragraph 2:
┌─────────────────────────────────────────────────────────────────────────────┐
│ AI is also improving natural language processing, enabling applications    │
│ like chatbots and virtual assistants.                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
        ┌─────────────────────────────────────────────────────────────────────┐
        │              If < chunk limit, keep as single chunk                  │
        └─────────────────────────────────────────────────────────────────────┘
```

**Code Example:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)
chunks = splitter.split_text(text)
```

| Pros | Cons |
|------|------|
| Preserves natural boundaries | May still break complex structures |
| Works well for most content | Requires tuning for specific formats |
| Best general-purpose choice | Not semantic-aware |
| Respects paragraph/sentence structure | |

**Best For:**
- **General purpose (DEFAULT CHOICE)**
- Articles, documentation, books
- Email threads, support tickets

> **Teaching Point:** "When in doubt, use recursive chunking - it's the Swiss Army knife of chunking strategies!"

---

### Strategy 4: Document Structure-Based Chunking

**What It Is:** Use the inherent structure of documents (headings, sections, paragraphs) to define chunk boundaries.

**Visual (from slide):**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Document                                                                     │
│ ├── Title ──────────────────────► Chunk 1                                   │
│ ├── Introduction ───────────────► Chunk 2                                   │
│ ├── Section #1 ─────────────────► Chunk 3                                   │
│ ├── Section #2 ─────────────────► Chunk 4                                   │
│ └── Conclusion ─────────────────► Chunk 5                                   │
│                                                                              │
│              *Merge with recursive chunking if sections too large            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Example Output (from slide):**
```
┌──────────────────────────────────────────────────────────────────┬─────────┐
│ Title: The Role of Artificial Intelligence in Modern Education  │ Chunk 1 │
├──────────────────────────────────────────────────────────────────┼─────────┤
│ Introduction                                                      │         │
│ Artificial intelligence (AI) is reshaping education by providing │ Chunk 2 │
│ personalized learning experiences and automating admin tasks.    │         │
├──────────────────────────────────────────────────────────────────┼─────────┤
│ Section 1: Personalized Learning                                  │         │
│ AI enables the customization of educational content to meet      │ Chunk 3 │
│ individual student needs, enhancing engagement and comprehension.│         │
├──────────────────────────────────────────────────────────────────┼─────────┤
│ Section 2: Administrative Automation                              │         │
│ From grading to scheduling, AI tools are streamlining admin      │ Chunk 4 │
│ processes, allowing educators to focus more on teaching.         │         │
├──────────────────────────────────────────────────────────────────┼─────────┤
│ Conclusion                                                        │         │
│ The integration of AI in education holds the promise of more     │ Chunk 5 │
│ efficient learning environments and improved student outcomes.   │         │
└──────────────────────────────────────────────────────────────────┴─────────┘
```

**For Markdown Documents:**
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ],
    strip_headers=False  # Keep headers in output
)
chunks = markdown_splitter.split_text(markdown_doc)

# Each chunk now has metadata!
# chunk.metadata = {"Header 1": "Main Title", "Header 2": "Subsection"}
```

**For HTML Documents:**
```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
)
chunks = html_splitter.split_text(html_doc)
```

**For Code:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)
# Splits by function/class definitions instead of arbitrary boundaries
```

| Pros | Cons |
|------|------|
| Preserves hierarchical context | Requires format-specific logic |
| Maintains metadata (headers become metadata) | May create very large or small chunks |
| Works great for structured docs | Can't handle unstructured text |
| Enables section-level filtering | |

**Best For:**
- Documentation (technical docs, wikis, manuals)
- Knowledge bases with clear structure
- Code repositories
- Any content with headers/sections

---

### Strategy 5: LLM-Based Chunking

**What It Is:** Use an LLM to intelligently create semantically isolated and meaningful chunks.

**Visual (from slide):**
```
Document ──► Input to LLM ──► LLM generates chunks ──► Final chunks
                   │
                   ▼
            "Please split this document into
             semantically coherent sections..."
```

**The Idea:**
- Every heuristic approach has upsides and downsides
- LLMs understand context and meaning beyond simple rules
- Let the LLM decide where the natural breakpoints are!

**Code Example:**
```python
from openai import OpenAI

client = OpenAI()

def llm_chunking(document: str, target_chunks: int = 5) -> list:
    """Use an LLM to intelligently chunk a document."""
    
    prompt = f"""Divide the following document into {target_chunks} semantically 
coherent sections. Each section should:
1. Cover a single topic or theme
2. Be self-contained and understandable on its own
3. Have a natural beginning and end

Return ONLY a JSON array of strings, where each string is one chunk.

Document:
{document}

JSON array of chunks:"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    import json
    chunks = json.loads(response.choices[0].message.content)
    return chunks
```

**Advanced Approach - Propositions:**
```python
def proposition_chunking(document: str) -> list:
    """Convert document to atomic propositions then group."""
    
    prompt = """Convert this document into a list of simple, atomic facts.
Each proposition should:
1. Be a single, complete statement
2. Be understandable without context
3. Include all necessary entity names (no pronouns)

Document: {document}

Return as JSON array of propositions:"""
    
    # First: Extract propositions
    propositions = llm_extract_propositions(document)
    
    # Then: Group related propositions into chunks
    chunks = group_similar_propositions(propositions)
    
    return chunks
```

| Pros | Cons |
|------|------|
| Highest semantic accuracy | **Expensive** (LLM calls for each doc!) |
| Understands context beyond heuristics | Slow processing |
| Can handle any document format | Non-deterministic (may vary) |
| Produces human-quality chunks | Rate limits become a concern |

**Best For:**
- High-value, complex documents
- When accuracy is critical and cost is acceptable
- Legal, medical, or research documents
- Small document sets

**Cost Consideration:**
```
100 documents × 2000 tokens each = 200,000 tokens
GPT-4 cost: ~$6-12 per chunking run
vs. Recursive: ~$0 (no LLM needed)
```

> **Teaching Point:** "LLM chunking is the nuclear option - use when nothing else works and you have the budget!"

---

## 4. Chunk Parameters Deep Dive

### Chunk Size Guidelines

| Size (tokens) | Character Approx | Use Case | Trade-off |
|---------------|------------------|----------|-----------|
| 100-200 | 400-800 | Precise Q&A | Fast but loses context |
| 200-500 | 800-2000 | **General RAG (default)** | Balanced |
| 500-1000 | 2000-4000 | Long-form answers | Complete but less precise |
| 1000+ | 4000+ | Summarization | Full docs but slow/expensive |

### Why OVERLAP Matters

```
Without Overlap:
─────────────────┼─────────────────
    Chunk 1      │     Chunk 2
"...password is  │ valid for 24..."
─────────────────┼─────────────────
                 ↑
        Information SPLIT!
        Query "password valid for" won't match!


With Overlap:
─────────────────────┐
    Chunk 1          │
"...password is valid│for 24 hours..."
─────────────────────┘
              ┌─────────────────────
              │     Chunk 2
"...is valid for 24 hours. After..."
              └─────────────────────
                     ↑
        Overlap captures boundary info!
        Query "password valid for" matches BOTH!
```

**Overlap Recommendations:**
| Use Case | Overlap % | Example |
|----------|-----------|---------|
| General | 10-15% | chunk=500, overlap=50-75 |
| High-precision | 20-25% | chunk=500, overlap=100-125 |
| Cost-sensitive | 5-10% | chunk=500, overlap=25-50 |

---

## 5. Cheat Sheet: 5 Chunking Strategies for RAG

> **Reference Slide:** Use this as a quick reference/summary!

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        5 CHUNKING STRATEGIES FOR RAG                            │
├──────────────────┬──────────────────────────────────────────────────────────────┤
│                  │                                                               │
│  1) Fixed-size   │  Split by character/token count                              │
│     chunking     │  + overlap to preserve boundary context                      │
│                  │  ✅ Simple  ❌ May break sentences                           │
│                  │                                                               │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│                  │                                                               │
│  2) Semantic     │  Use embeddings to detect topic shifts                       │
│     chunking     │  Split when cosine similarity drops                          │
│                  │  ✅ Best quality  ❌ Expensive (needs embeddings)            │
│                  │                                                               │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│                  │                                                               │
│  3) Recursive    │  Try separators in order: ¶ → \n → . → space                │
│     chunking     │  Only split further if still too large                       │
│                  │  ✅ Great default  ❌ Not semantic-aware                     │
│                  │                                                               │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│                  │                                                               │
│  4) Structure-   │  Use headers, sections, paragraphs                           │
│     based        │  Preserves document hierarchy                                 │
│                  │  ✅ Keeps metadata  ❌ Needs structured input                │
│                  │                                                               │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│                  │                                                               │
│  5) LLM-based    │  Ask LLM to create semantic chunks                           │
│     chunking     │  Highest intelligence, understands meaning                    │
│                  │  ✅ Best semantics  ❌ Expensive & slow                      │
│                  │                                                               │
└──────────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 6. Choosing the Right Strategy

### Decision Matrix

| Content Type | Recommended Strategy | Chunk Size | Why |
|--------------|---------------------|------------|-----|
| **Support tickets** | Fixed or Recursive | 200-400 | Short, already focused |
| **Documentation** | Structure-based | 400-600 | Has clear sections |
| **Blog posts/articles** | Recursive | 300-500 | Natural paragraphs |
| **Research papers** | Semantic | 400-800 | Topic-focused sections |
| **Code** | Language-aware | 200-400 | Function boundaries |
| **Legal/contracts** | LLM-based or Semantic | 500-800 | High accuracy needed |
| **Chat logs** | Fixed | 200-400 | No natural structure |

### Quick Decision Flow

```
START
  │
  ▼
Is document structured (headers, sections)?
  ├── YES → Use Structure-based chunking
  │
  └── NO
       │
       ▼
  Do you need highest semantic quality?
       ├── YES → Is cost acceptable?
       │           ├── YES → Use LLM-based
       │           └── NO → Use Semantic chunking
       │
       └── NO → Use Recursive chunking (default)
```

---

## 7. Demo Walkthrough

### Demo Overview (`demo.py`)

The demo shows:
1. Different chunking strategies on support ticket data
2. Building a Chroma vector store
3. Comparing retrieval quality across strategies
4. Metadata filtering

---

### Key Code Points to Highlight

**Strategy Comparison:**
```python
# Fixed-size: Simple but may break mid-sentence
fixed_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# Recursive: Smarter, tries natural boundaries
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

# Structure-aware: For markdown/HTML
markdown_splitter = MarkdownHeaderTextSplitter(...)
```

**Building Vector Stores with Chroma:**
```python
# High-level: LangChain + Chroma (handles everything!)
chroma_store = Chroma.from_documents(
    documents, 
    embedding=embeddings_model,
    persist_directory="./chroma_db"  # Optional: persist to disk
)

# Search returns Document objects with metadata
results = chroma_store.similarity_search(query, k=3)
```

**Why MMR matters (teach this explicitly):**

Normal semantic search (`similarity_search`) ranks by relevance only.
That can produce near-duplicate results, which wastes prompt space.

Example query: **"login issues after password reset"**

- Normal top-3 might be:
    1. "Can't login after reset on web"
    2. "Login fails after password reset"
    3. "Password reset causes auth error"

These are all relevant, but very similar to each other.

At production scale (millions of chunks/documents), this gets worse: the top-k can be dominated by multiple chunks from the same source document, which hurts breadth of evidence.

MMR (Maximal Marginal Relevance) improves this by balancing:
1. Query relevance
2. Diversity from already selected results

So MMR top-3 might be:
1. "Can't login after reset on web" (core issue)
2. "MFA token invalid after password reset" (different angle)
3. "Session cookie not refreshed" (another angle)

```python
mmr_results = chroma_store.max_marginal_relevance_search(query, k=3)
```

**Instructor script:**
"Similarity search gives me the most similar chunks; MMR gives me the most useful set of chunks. For RAG, that often means better answers because context is less repetitive and covers more angles."

**Metadata Filtering (key feature!):**
```python
# Search only authentication tickets
results = chroma_store.similarity_search(
    query,
    filter={"category": "Authentication"}
)

# Search only high priority
results = chroma_store.similarity_search(
    query,
    filter={"priority": "High"}
)
```

---

## 8. Common Mistakes to Warn About

| Mistake | Problem | Solution |
|---------|---------|----------|
| Counting by characters | Token limits are in TOKENS not chars | Use tiktoken to count |
| No overlap | Loses boundary information | Always use 10-20% overlap |
| Same size for all content | Different docs need different sizes | Tune per content type |
| Ignoring structure | Breaks headers, lists, code blocks | Use structure-aware splitters |
| Over-chunking short docs | Creates noise | Don't chunk if doc < target size |

---

## 9. Best Practices Summary

1. **Start with Recursive** - It's the best default choice
2. **Use tokens, not characters** - API limits are token-based
3. **Always include overlap** - 10-20% prevents information loss
4. **Match strategy to content** - Structured docs → structure-aware
5. **Test with real queries** - Validate retrieval quality
6. **Preserve metadata** - Enables filtering later
7. **Consider chunk + parent** - Small for search, large for context

---

## 10. Key Takeaways (Summary Slide)

1. **Chunking is CRITICAL** for RAG quality - bad chunks = bad retrieval
2. **5 Strategies:** Fixed, Semantic, Recursive, Structure-based, LLM-based
3. **Overlap** prevents information loss at boundaries
4. **Recursive chunking** is the best default for most cases
5. **Match strategy to content type** - structured docs need structure-aware
6. **Trade-offs exist:** Accuracy ↔ Cost ↔ Speed

---

## References

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Pinecone: Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [Chroma DB](https://docs.trychroma.com/)

---

## Next Module

**Module 3: Indexing** - Building and optimizing vector indexes for fast retrieval.
