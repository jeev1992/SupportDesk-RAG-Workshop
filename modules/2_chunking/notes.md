# Module 2: Chunking Strategies

## Introduction

Chunking is the process of breaking down large documents into smaller, meaningful pieces for embedding and retrieval. Effective chunking is critical to RAG system performance—bad chunking leads to irrelevant results and fragmented context.

## Why Chunking Matters

### The Problem
- **Token Limits**: Embedding models have maximum input lengths (8,191 tokens for OpenAI)
- **Context Windows**: LLMs can only process limited context (4K-128K tokens)
- **Retrieval Precision**: Large chunks return too much irrelevant information
- **Semantic Coherence**: Small chunks lose context and meaning

### The Goal
Find the optimal balance between:
- **Completeness**: Enough context to be meaningful
- **Specificity**: Focused enough to be relevant
- **Efficiency**: Small enough to process quickly

## Chunking Strategies

### 1. Fixed-Size Chunking

**Description**: Split text into equal-sized chunks based on character count or token count.

```python
def fixed_size_chunking(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap for context continuity
    return chunks
```

**Pros:**
- ✅ Simple to implement
- ✅ Predictable chunk sizes
- ✅ Fast processing

**Cons:**
- ❌ May split sentences mid-word
- ❌ Ignores document structure
- ❌ Can break semantic units

**Best For:** Unstructured text, quick prototypes

**Parameters:**
- **chunk_size**: 200-500 tokens (typical)
- **overlap**: 10-20% of chunk_size
- **Measure in tokens, not characters** (use tiktoken)

### 2. Recursive Character Text Splitting

**Description**: Split text hierarchically by trying different separators in order.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],  # Try in order
    length_function=len
)
chunks = splitter.split_text(text)
```

**Splitting Order:**
1. First try: Paragraph breaks (`\n\n`)
2. Then try: Line breaks (`\n`)
3. Then try: Sentence endings (`. `)
4. Then try: Word boundaries (` `)
5. Last resort: Character-by-character

**Pros:**
- ✅ Preserves natural boundaries
- ✅ Maintains semantic coherence
- ✅ Works well for most content

**Cons:**
- ❌ May still break complex structures
- ❌ Requires tuning for specific formats

**Best For:** General purpose text (articles, documentation, emails)

### 3. Semantic Chunking

**Description**: Split based on semantic similarity between sentences.

```python
def semantic_chunking(text, similarity_threshold=0.7):
    sentences = split_into_sentences(text)
    sentence_embeddings = [get_embedding(s) for s in sentences]
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        prev_embedding = sentence_embeddings[i-1]
        curr_embedding = sentence_embeddings[i]
        similarity = cosine_similarity(prev_embedding, curr_embedding)
        
        if similarity >= similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    
    chunks.append(" ".join(current_chunk))
    return chunks
```

**Pros:**
- ✅ Preserves semantic coherence
- ✅ Natural topic boundaries
- ✅ Adapts to content

**Cons:**
- ❌ Expensive (requires embeddings)
- ❌ Variable chunk sizes
- ❌ Slower processing

**Best For:** High-quality retrieval where accuracy is critical

### 4. Document Structure-Aware Splitting

**Description**: Respect document structure (headings, sections, paragraphs).

#### Markdown Splitting
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)
chunks = splitter.split_text(markdown_text)
```

#### HTML Splitting
```python
from langchain_text_splitters import HTMLHeaderTextSplitter

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
)
chunks = splitter.split_text_from_url(url)
```

#### Code Splitting
```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)
chunks = python_splitter.split_text(code)
```

**Pros:**
- ✅ Preserves hierarchical context
- ✅ Maintains metadata (headers, tags)
- ✅ Better for structured documents

**Cons:**
- ❌ Requires format-specific logic
- ❌ May create very large or small chunks

**Best For:** Documentation, wikis, code repositories

### 5. Sentence Window Retrieval

**Description**: Store small chunks but retrieve with surrounding context.

```python
def create_sentence_windows(text, window_size=3):
    sentences = split_into_sentences(text)
    windows = []
    
    for i in range(len(sentences)):
        # Get surrounding sentences
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)
        window = sentences[start:end]
        
        windows.append({
            'search_text': sentences[i],  # Index only this
            'retrieval_text': " ".join(window),  # Return with context
            'position': i
        })
    
    return windows
```

**Pros:**
- ✅ Precise retrieval
- ✅ Rich context for generation
- ✅ Reduces hallucinations

**Cons:**
- ❌ More storage required
- ❌ Complex implementation

**Best For:** Q&A systems, citation-heavy applications

### 6. Parent-Child Chunking

**Description**: Index small chunks, return parent documents.

```python
def create_parent_child_chunks(documents):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    
    indexed_chunks = []
    
    for doc in documents:
        parent_chunks = parent_splitter.split_text(doc)
        
        for parent_chunk in parent_chunks:
            child_chunks = child_splitter.split_text(parent_chunk)
            
            for child in child_chunks:
                indexed_chunks.append({
                    'child_text': child,  # Index this
                    'parent_text': parent_chunk,  # Return this
                    'doc_id': doc.id
                })
    
    return indexed_chunks
```

**Pros:**
- ✅ Precise search + complete context
- ✅ Reduces fragmentation
- ✅ Better answer quality

**Cons:**
- ❌ Increased storage (2x chunks)
- ❌ More complex retrieval logic

**Best For:** Long documents (reports, manuals, legal docs)

## Chunking Parameters

### Chunk Size

| Size (tokens) | Use Case | Pros | Cons |
|---------------|----------|------|------|
| 100-200 | Precise Q&A | Fast, specific | Loses context |
| 200-500 | General RAG | Balanced | May split topics |
| 500-1000 | Long context | Complete info | Less precise |
| 1000+ | Summarization | Full documents | Slow, expensive |

**Rule of Thumb:** 
- Short queries → Smaller chunks (200-400)
- Complex questions → Larger chunks (500-800)

### Chunk Overlap

```python
# Example with 20% overlap
chunk_size = 500
overlap = 100  # 20% of 500

# Chunk 1: tokens 0-500
# Chunk 2: tokens 400-900 (overlaps tokens 400-500)
# Chunk 3: tokens 800-1300 (overlaps tokens 800-900)
```

**Recommendations:**
- **Minimum**: 10% overlap
- **Typical**: 15-20% overlap
- **Maximum**: 25% overlap

**Why Overlap?**
- Prevents splitting important info across chunk boundaries
- Provides context continuity
- Improves retrieval recall

**Trade-offs:**
- More overlap = More storage + cost
- More overlap = Better continuity
- Too much overlap = Redundancy

### Token vs Character Counting

❌ **Wrong**: Count by characters
```python
chunk_size = 500  # characters - misleading!
```

✅ **Right**: Count by tokens
```python
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(encoding.encode(text))

def token_based_chunking(text, max_tokens=500):
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks
```

**Why Tokens?**
- Embeddings and LLMs charge by tokens
- Accurate for API limits
- Consistent across languages

**Rule:** 1 token ≈ 0.75 words (English)

## Advanced Techniques

### 1. Multi-Representation Chunking

Store multiple versions of the same content:

```python
def multi_representation_chunking(document):
    return {
        'full_text': document,
        'summary': generate_summary(document),  # For retrieval
        'keywords': extract_keywords(document),  # For filtering
        'embeddings': {
            'full': get_embedding(document),
            'summary': get_embedding(generate_summary(document))
        }
    }
```

**When to Use:** Complex documents where different retrieval strategies are needed

### 2. Contextual Chunk Enrichment

Add context to each chunk:

```python
def enrich_chunks(chunks, document_metadata):
    enriched = []
    for i, chunk in enumerate(chunks):
        enriched_chunk = {
            'text': chunk,
            'metadata': {
                'document_title': document_metadata['title'],
                'chunk_index': i,
                'total_chunks': len(chunks),
                'section': get_section(chunk),
                'prev_chunk_summary': summarize(chunks[i-1]) if i > 0 else None
            }
        }
        enriched.append(enriched_chunk)
    return enriched
```

**Benefits:**
- Better filtering
- Improved ranking
- Enhanced prompt context

### 3. Adaptive Chunking

Adjust chunk size based on content:

```python
def adaptive_chunking(text):
    if is_code(text):
        return chunk_by_functions(text)
    elif is_table(text):
        return chunk_by_rows(text)
    elif is_dialogue(text):
        return chunk_by_speakers(text)
    else:
        return recursive_chunk(text)
```

## Choosing the Right Strategy

### Decision Matrix

| Content Type | Strategy | Chunk Size | Overlap |
|--------------|----------|------------|---------|
| Articles/Blogs | Recursive | 300-500 | 50-100 |
| Documentation | Structure-Aware | 400-600 | 80-100 |
| Code | Language-Specific | 200-400 | 40-80 |
| Q&A | Semantic | 200-300 | 30-50 |
| Legal/Medical | Parent-Child | 500-800 | 100-150 |
| Chat Logs | Fixed-Size | 200-400 | 40-80 |

### Evaluation Metrics

**1. Retrieval Quality**
```python
def evaluate_chunking(chunks, test_queries):
    relevant_retrieved = 0
    total_retrieved = 0
    
    for query in test_queries:
        results = retrieve(query, chunks, k=5)
        relevant = sum(1 for r in results if r.is_relevant)
        relevant_retrieved += relevant
        total_retrieved += len(results)
    
    precision = relevant_retrieved / total_retrieved
    return precision
```

**2. Context Completeness**
- Can you understand the chunk without external context?
- Does it contain complete sentences?
- Are key concepts defined?

**3. Retrieval Efficiency**
- Average chunk size
- Storage overhead
- Query latency

## Retrieval Diversity: Semantic Search vs MMR

### Problem with Normal Semantic Search

`similarity_search()` optimizes only for relevance to the query, so top results can be near-duplicates.

Example query: **"login issues after password reset"**

- Top 3 from normal semantic search might be:
    1. "Can't login after reset on web"
    2. "Login fails after password reset"
    3. "Password reset causes auth error"

All 3 are relevant, but they are very similar to each other, so you get limited coverage.

At larger scale (millions of chunks/documents), this happens even more: top results often come from different chunks of the **same source document**, which reduces cross-document coverage.

```python
results = chroma_store.similarity_search(
        "login issues after password reset",
        k=3
)
```

### How MMR Improves This

`max_marginal_relevance_search()` balances:
- **Relevance** to the query
- **Diversity** from already selected results

So results stay relevant but cover different subtopics.

Example MMR top 3 could be:
1. "Can't login after reset on web" (core issue)
2. "MFA token invalid after password reset" (different angle)
3. "Session cookie not refreshed" (another angle)

```python
mmr_results = chroma_store.max_marginal_relevance_search(
        "login issues after password reset",
        k=3
)
```

**Rule of thumb:**
- Use `similarity_search` when you want the most similar matches.
- Use MMR when you want relevant **and** non-redundant context (often better for RAG prompts).

## Best Practices

### 1. Preprocessing Text

```python
def preprocess_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove control characters
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
    
    return text.strip()
```

### 2. Add Metadata

```python
def chunk_with_metadata(document):
    chunks = split_document(document)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            'source': document.source,
            'chunk_id': f"{document.id}_chunk_{i}",
            'created_at': datetime.now(),
            'language': detect_language(chunk),
            'tokens': count_tokens(chunk)
        }
    
    return chunks
```

### 3. Validate Chunks

```python
def validate_chunk(chunk, min_tokens=50, max_tokens=1000):
    token_count = count_tokens(chunk)
    
    if token_count < min_tokens:
        return False, "Too short"
    if token_count > max_tokens:
        return False, "Too long"
    if not contains_meaningful_content(chunk):
        return False, "No meaningful content"
    
    return True, "Valid"
```

### 4. Handle Special Cases

```python
def handle_special_content(text):
    # Tables
    if is_table(text):
        return chunk_table_rows(text)
    
    # Lists
    if is_list(text):
        return chunk_list_items(text)
    
    # Code blocks
    if is_code_block(text):
        return chunk_by_functions(text)
    
    # Default
    return recursive_chunk(text)
```

## Common Pitfalls

### 1. Ignoring Document Structure
❌ **Wrong**: Treat all text as plain text
```python
chunks = text.split('\n\n')  # Loses headers, structure
```

✅ **Right**: Preserve hierarchy
```python
chunks = markdown_splitter.split_text(text)  # Keeps headers
```

### 2. Too Small or Too Large
❌ **Wrong**: One-size-fits-all
```python
chunk_size = 100  # Always too small for complex queries
```

✅ **Right**: Test and tune
```python
# A/B test different sizes
for size in [200, 400, 600]:
    evaluate_retrieval(chunk_size=size)
```

### 3. No Overlap
❌ **Wrong**: Hard boundaries
```python
chunks = [text[i:i+500] for i in range(0, len(text), 500)]
```

✅ **Right**: Overlapping chunks
```python
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```

### 4. Ignoring Tokens
❌ **Wrong**: Character-based limits
```python
if len(chunk) > 500:  # Characters don't map to tokens
    split_chunk(chunk)
```

✅ **Right**: Token-based limits
```python
if count_tokens(chunk) > 500:
    split_chunk(chunk)
```

## Performance Optimization

### Parallel Chunking
```python
from concurrent.futures import ProcessPoolExecutor

def chunk_documents_parallel(documents):
    with ProcessPoolExecutor() as executor:
        chunks = executor.map(chunk_document, documents)
    return list(chunks)
```

### Caching
```python
class ChunkCache:
    def __init__(self):
        self.cache = {}
    
    def get_or_create(self, doc_id, doc_text, chunker):
        if doc_id in self.cache:
            return self.cache[doc_id]
        
        chunks = chunker(doc_text)
        self.cache[doc_id] = chunks
        return chunks
```

## Tools and Libraries

### LangChain Text Splitters
```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter
)
```

### Token Counting
```python
import tiktoken

# For OpenAI models
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(text)
```

### Sentence Splitting
```python
import nltk
nltk.download('punkt')

sentences = nltk.sent_tokenize(text)
```

## References

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Chunking Strategies for RAG](https://www.pinecone.io/learn/chunking-strategies/)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)

## Next Steps

Now that you understand chunking, proceed to **Module 3: Indexing Strategies** to learn how to store and retrieve these chunks efficiently.
