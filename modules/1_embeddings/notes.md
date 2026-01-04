# Module 1: Text Embeddings

## Introduction

Text embeddings are numerical representations of text that capture semantic meaning. They are the foundation of modern RAG systems, enabling computers to understand and compare the meaning of text rather than just matching keywords.

## What Are Embeddings?

### Definition
An embedding is a dense vector (array of numbers) that represents text in high-dimensional space. Similar concepts are positioned close to each other in this space, even if they use different words.

**Example:**
```
"cat" → [0.2, -0.5, 0.8, ...]  (1536 dimensions)
"kitten" → [0.19, -0.48, 0.82, ...]  (very similar to "cat")
"dog" → [0.15, -0.3, 0.7, ...]  (somewhat similar)
"computer" → [-0.5, 0.8, -0.2, ...]  (completely different)
```

### Why Embeddings Matter

1. **Semantic Search**: Find documents by meaning, not just keywords
2. **Language Understanding**: Capture context and nuance
3. **Multilingual**: Work across languages
4. **Transfer Learning**: Pre-trained on massive datasets

## OpenAI Embeddings

### Available Models

| Model | Dimensions | Use Case | Cost per 1M tokens |
|-------|------------|----------|-------------------|
| text-embedding-3-small | 1536 | General purpose | $0.02 |
| text-embedding-3-large | 3072 | Highest quality | $0.13 |
| text-embedding-ada-002 | 1536 | Legacy (deprecated) | $0.10 |

**Recommendation**: Use `text-embedding-3-small` for most applications. It offers excellent quality at low cost.

### How to Generate Embeddings

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# Single text
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="How do I reset my password?"
)
embedding = response.data[0].embedding  # List of 1536 floats

# Batch processing (more efficient)
texts = ["text 1", "text 2", "text 3"]
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
embeddings = [item.embedding for item in response.data]
```

## Similarity Metrics

### Cosine Similarity
Measures the angle between two vectors. Range: -1 to 1 (higher is more similar).

**Formula:**
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

**Why Cosine?**
- Normalized by magnitude (good for embeddings)
- Focuses on direction, not length
- Fast to compute
- OpenAI embeddings are already normalized

**Python Implementation:**
```python
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

### Other Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Euclidean Distance | sqrt(Σ(A-B)²) | When magnitude matters |
| Dot Product | A · B | Pre-normalized vectors |
| Manhattan Distance | Σ\|A-B\| | Sparse vectors |

**For OpenAI embeddings, use cosine similarity or dot product** (since they're normalized).

## Embedding Best Practices

### 1. Text Preprocessing

**Do:**
- Remove excessive whitespace
- Normalize unicode characters
- Keep punctuation (it carries meaning)

**Don't:**
- Remove stopwords (embeddings handle them well)
- Stem or lemmatize (can lose meaning)
- Lowercase everything (case matters for acronyms)

### 2. Optimal Text Length

- **Minimum**: 10-20 tokens (too short = noisy)
- **Maximum**: 8191 tokens (API limit)
- **Recommended**: 100-500 tokens per chunk

**Why?** OpenAI models use attention mechanisms that work best with meaningful context.

### 3. Batch Processing

```python
# Bad: One API call per text
for text in texts:
    embedding = client.embeddings.create(input=text, ...)

# Good: Batch up to 2048 texts
batch_size = 2048
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    response = client.embeddings.create(input=batch, ...)
```

**Benefits:**
- 10-100x faster
- Lower API overhead
- More cost-effective

### 4. Caching Strategy

```python
import hashlib
import json

def get_or_create_embedding(text, cache={}):
    # Create hash of text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Check cache
    if text_hash in cache:
        return cache[text_hash]
    
    # Generate embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
    
    # Store in cache
    cache[text_hash] = embedding
    return embedding
```

## Visualizing Embeddings

### Dimensionality Reduction

High-dimensional embeddings (1536D) can't be visualized directly. Use dimensionality reduction:

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
```python
from sklearn.manifold import TSNE

# Reduce 1536D → 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
```

**UMAP (Uniform Manifold Approximation and Projection)**
```python
import umap

# Faster and often better than t-SNE
reducer = umap.UMAP(n_components=2)
embeddings_2d = reducer.fit_transform(embeddings)
```

**PCA (Principal Component Analysis)**
```python
from sklearn.decomposition import PCA

# Linear reduction (faster but less accurate)
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)
```

## Common Pitfalls

### 1. Comparing Different Models
❌ **Wrong**: Compare embeddings from different models
```python
embedding1 = get_embedding("text", model="text-embedding-3-small")
embedding2 = get_embedding("text", model="text-embedding-3-large")
similarity = cosine_similarity(embedding1, embedding2)  # Invalid!
```

✅ **Right**: Use the same model for all embeddings in a project

### 2. Ignoring Token Limits
❌ **Wrong**: Embed very long documents without splitting
```python
long_doc = "..." * 10000  # Way over token limit
embedding = get_embedding(long_doc)  # API error!
```

✅ **Right**: Split documents into chunks (covered in Module 2)

### 3. Over-Embedding
❌ **Wrong**: Embed every sentence separately
```python
for sentence in document.split('.'):
    embeddings.append(get_embedding(sentence))  # Expensive!
```

✅ **Right**: Find optimal chunk size (200-500 tokens typically)

### 4. Not Handling Rate Limits
```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
def get_embedding_with_retry(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
```

## Real-World Applications

### 1. Semantic Search
```python
def semantic_search(query, documents, top_k=5):
    # Embed query
    query_embedding = get_embedding(query)
    
    # Get document embeddings
    doc_embeddings = [get_embedding(doc) for doc in documents]
    
    # Calculate similarities
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]
    
    # Get top K results
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]
```

### 2. Clustering
```python
from sklearn.cluster import KMeans

# Group similar documents
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(embeddings)
```

### 3. Anomaly Detection
```python
def find_outliers(embeddings, threshold=0.3):
    # Calculate centroid
    centroid = np.mean(embeddings, axis=0)
    
    # Find documents far from center
    distances = [
        cosine_similarity(emb, centroid)
        for emb in embeddings
    ]
    
    outliers = [i for i, d in enumerate(distances) if d < threshold]
    return outliers
```

## Performance Considerations

### Cost Optimization

**1. Dimension Reduction** (optional parameter)
```python
# Get 512D embeddings instead of 1536D
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text,
    dimensions=512  # Reduce storage by 66%
)
```

**2. Estimate Costs**
```python
import tiktoken

def estimate_embedding_cost(texts, model="text-embedding-3-small"):
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(encoding.encode(text)) for text in texts)
    
    cost_per_million = 0.02  # text-embedding-3-small
    cost = (total_tokens / 1_000_000) * cost_per_million
    
    print(f"Total tokens: {total_tokens:,}")
    print(f"Estimated cost: ${cost:.4f}")
    return cost
```

### Speed Optimization

**1. Parallel Processing**
```python
from concurrent.futures import ThreadPoolExecutor

def batch_embed_parallel(texts, batch_size=100):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            future = executor.submit(get_embedding, batch)
            futures.append(future)
        
        return [f.result() for f in futures]
```

**2. Async Processing**
```python
import asyncio
from openai import AsyncOpenAI

async def embed_async(texts):
    client = AsyncOpenAI()
    tasks = [
        client.embeddings.create(model="text-embedding-3-small", input=text)
        for text in texts
    ]
    responses = await asyncio.gather(*tasks)
    return [r.data[0].embedding for r in responses]
```

## Evaluation Metrics

### Embedding Quality

**1. Intrinsic Evaluation**
- Cosine similarity between related concepts
- Clustering quality (silhouette score)
- Nearest neighbor accuracy

**2. Extrinsic Evaluation**
- Search accuracy (precision/recall)
- Downstream task performance
- User satisfaction metrics

## Advanced Topics

### 1. Fine-Tuning Embeddings
OpenAI doesn't support fine-tuning for embeddings (as of 2024), but you can:
- Add domain-specific metadata
- Use hybrid search (embeddings + keywords)
- Train a lightweight re-ranking model

### 2. Multimodal Embeddings
Combine text with other modalities:
```python
# CLIP for text + images
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Embed text and image in same space
text_embedding = model.get_text_features(**processor(text=[query]))
image_embedding = model.get_image_features(**processor(images=[image]))
```

### 3. Dynamic Embeddings
Update embeddings as documents change:
```python
class EmbeddingStore:
    def __init__(self):
        self.embeddings = {}
        self.timestamps = {}
    
    def add_or_update(self, doc_id, text):
        self.embeddings[doc_id] = get_embedding(text)
        self.timestamps[doc_id] = datetime.now()
    
    def should_refresh(self, doc_id, max_age_days=30):
        age = datetime.now() - self.timestamps[doc_id]
        return age.days > max_age_days
```

## References

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Text Embeddings: A Comprehensive Guide](https://www.pinecone.io/learn/text-embeddings/)
- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)

## Next Steps

Now that you understand embeddings, proceed to **Module 2: Chunking** to learn how to split documents optimally for embedding and retrieval.
