# Hour 1 Exercises: Embeddings & Similarity Search

> ✅ **Exercise style for this workshop:** keep each solution to a **small edit** (usually 3–15 lines) in existing files.

## Exercise 1: Change the Search Query (Easy)

**Task**: Modify the demo to search for a different type of issue.

**In the demo code, find this line** (around line 103):
```python
query = "Users can't login after changing password"
```

**Change it to**:
```python
query = "Database is running very slowly"
```

**Run the demo and observe**:
- Do the top results match the new query?
- What categories do the matching tickets belong to?

**Try these queries too**:
- `"Payment failed for international customer"`
- `"Mobile app keeps crashing"`
- `"Emails are not being delivered"`

---

## Exercise 2: Adjust the Number of Results (Easy)

**Task**: Get more search results by changing top_k.

**In the demo code, find this line** (around line 137):
```python
top_k = 5
```

**Change it to**:
```python
top_k = 10
```

**Run the demo and answer**:
- At what rank does the similarity score drop below 0.5?
- Are results #8, #9, #10 still relevant to your query?
- What's the similarity score of result #10?

---

## Exercise 3: Add a Similarity Threshold (Easy)

**Task**: Only show results above a certain similarity score.

**Find the results loop in the demo** (around line 145):
```python
for rank, idx in enumerate(top_indices, 1):
    ticket = tickets[idx]
    score = similarities[idx]
    
    print(f"\n#{rank} - Similarity: {score:.4f}")
    # ... rest of printing
```

**Add one line to skip low-scoring results**:
```python
for rank, idx in enumerate(top_indices, 1):
    ticket = tickets[idx]
    score = similarities[idx]
    
    # ADD THIS LINE: Skip results below threshold
    if score < 0.5:
        continue
    
    print(f"\n#{rank} - Similarity: {score:.4f}")
    # ... rest of printing
```

**Test with**:
- A relevant query (should show results)
- An unrelated query like `"How to make pizza"` (should show fewer/no results)

---

## Exercise 4: Compare Two Queries (Easy)

**Task**: See how the same tickets rank differently for different queries.

**Add this code at the end of the demo** (after PART 5):
```python
# Compare two queries
query1 = "Login authentication failed"
query2 = "Slow database performance"

print("\n" + "="*80)
print("COMPARING TWO QUERIES")
print("="*80)

for q in [query1, query2]:
    response = client.embeddings.create(input=[q], model=embedding_model)
    q_emb = np.array([response.data[0].embedding])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argmax(sims)
    
    print(f"\nQuery: '{q}'")
    print(f"  Best match: {tickets[top_idx]['title']}")
    print(f"  Score: {sims[top_idx]:.4f}")
```

**Run it and observe**: Do the queries find appropriate tickets?

---

## Exercise 5: Test Semantic Understanding (Medium)

**Task**: Verify that embeddings understand meaning, not just keywords, with a small patch.

**In `demo.py`, add a short block near the end** (no new file needed):
```python
import numpy as np
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'text-embedding-3-small'

# These mean the SAME thing but use DIFFERENT words
texts = [
    "User authentication failed",      # Original
    "Login credentials rejected",       # Same meaning, different words
    "Cannot sign in to account",        # Same meaning, different words
    "Database connection timeout",      # DIFFERENT topic
]

# Generate embeddings
response = client.embeddings.create(input=texts, model=model)
embeddings = np.array([data.embedding for data in response.data])

# Calculate all pairwise similarities
similarity_matrix = cosine_similarity(embeddings)

# Print results
print("Similarity Matrix:")
print("-" * 50)
for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        if i < j:  # Only print upper triangle
            sim = similarity_matrix[i][j]
            print(f"{sim:.3f}  '{text1[:30]}...' vs '{text2[:30]}...'")
```

Keep this to ~15 lines if you simplify printing.

**Run it and answer**:
- What's the similarity between "authentication failed" and "login rejected"?
- What's the similarity between "authentication failed" and "database timeout"?
- Does this prove embeddings understand meaning, not just keywords?

---

## Exercise 6: Filter by Category (Medium)

**Task**: Add category filtering with one-line logic.

**Use the existing search loop in demo/solutions style and add this ONE line:**

```python
if category_filter and ticket['category'] != category_filter:
    continue
```

If you prefer, use the full snippet below as reference:
```python
import json
import numpy as np
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'text-embedding-3-small'

# Load data
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)

texts = [f"{t['title']}. {t['description']}" for t in tickets]
response = client.embeddings.create(input=texts, model=model)
embeddings = np.array([data.embedding for data in response.data])

def search_with_category(query, category_filter=None, top_k=5):
    """Search tickets, optionally filtering by category"""
    # Get query embedding
    response = client.embeddings.create(input=[query], model=model)
    query_emb = np.array([response.data[0].embedding])
    
    # Calculate similarities
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    # Get results with category filter
    results = []
    for idx in np.argsort(similarities)[::-1]:
        ticket = tickets[idx]
        
        # FILL IN THIS LINE: Skip if category doesn't match filter
        # Hint: if category_filter is set AND ticket category doesn't match, skip
        if category_filter and ticket['category'] != category_filter:
            continue
        
        results.append((ticket, similarities[idx]))
        if len(results) >= top_k:
            break
    
    return results

# Test it
print("All categories:")
for ticket, score in search_with_category("login problem"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")

print("\nOnly 'Authentication' category:")
for ticket, score in search_with_category("login problem", category_filter="Authentication"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")
```

---

## Exercise 7: Batch vs Single Embedding (Medium)

**Task**: Compare speed with a tiny measurement patch.

Use the existing embedding section and add 6–10 lines with `time.time()` around:
- one-by-one embedding calls
- one batched embedding call

Reference snippet:
```python
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'text-embedding-3-small'

texts = [
    "Password reset not working",
    "Database connection timeout", 
    "App crashes on startup",
    "Payment declined error",
    "Email notifications delayed",
]

# Method 1: SLOW - One API call per text
print("Method 1: Single API calls...")
start = time.time()
for text in texts:
    response = client.embeddings.create(input=[text], model=model)
time_slow = time.time() - start
print(f"  Time: {time_slow:.2f} seconds")

# Method 2: FAST - One API call for all texts
print("\nMethod 2: Batch API call...")
start = time.time()
response = client.embeddings.create(input=texts, model=model)
time_fast = time.time() - start
print(f"  Time: {time_fast:.2f} seconds")

# Compare
print(f"\n✓ Batch is {time_slow/time_fast:.1f}x faster!")
print(f"  Always batch your embeddings in production!")
```

**Answer**: How much faster is batching for 5 texts?

---

## Bonus Exercise: Similarity Matrix Heatmap (Challenge)

**Task**: Create a visual heatmap showing how tickets relate to each other.

**This is optional** - only try if you finished the other exercises.

Small-edit option: generate the heatmap for only first 5 tickets and skip cell annotations.

```python
import json
import numpy as np
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load first 10 tickets
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)[:10]

# Generate embeddings
texts = [t['title'] for t in tickets]
response = client.embeddings.create(input=texts, model='text-embedding-3-small')
embeddings = np.array([data.embedding for data in response.data])

# Compute similarity matrix
sim_matrix = cosine_similarity(embeddings)

# Create heatmap
plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(10), [t['ticket_id'] for t in tickets], rotation=45, ha='right')
plt.yticks(range(10), [t['ticket_id'] for t in tickets])
plt.title('Ticket Similarity Matrix')
plt.tight_layout()
plt.savefig('similarity_heatmap.png')
plt.show()
print("✓ Saved as similarity_heatmap.png")
```

---

## Quick Reference

### Key Code Patterns
```python
# Generate embedding for one text
response = client.embeddings.create(input=["your text"], model=model)
embedding = response.data[0].embedding

# Generate embeddings for multiple texts (batch)
response = client.embeddings.create(input=list_of_texts, model=model)
embeddings = [data.embedding for data in response.data]

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], all_embeddings)[0]

# Get top K indices
top_indices = np.argsort(similarities)[::-1][:top_k]
```

---

## Next Steps

Ready for more? Move on to **Hour 2: Chunking & Vector Stores** where we'll:
- Scale to thousands of documents
- Learn efficient storage and retrieval
- Build production-ready vector databases

---

**Questions?** Ask the instructor or refer back to the demo code!
